"""Exact fp32 SFT/SmartAD math and stochastic LoRA expected gradients."""
from copy import deepcopy
import json
import os
from pathlib import Path
from datetime import timedelta

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('peft')

from baselines.common.conversation import conversation, sequence_rows
from baselines.common.fsdp_train import sft_objective, training_microsteps, train_loop
from baselines.common.shared_history import pack_shared_history
from baselines.smartad.data import encode_episode, objective as smartad_objective
from test_sad import CharTokenizer, episode
from test_sad_shared_history_torch import assert_fp32_gradient
from test_validate_shared_history_gpu import tiny_model
from train.modeling import attach_lora


def examples(objective, count=3):
    rows = []
    for i in range(count):
        source = episode('unused')
        source['task_id'] = f'train/{i}'
        source['steps'] = [dict(source['steps'][0], event_id=f'e{j}',
            action_raw=f'Thought: {"check " * (j + 1)}\nAction: look') for j in range(1 + i % 3)]
        row = (encode_episode(source, CharTokenizer()) if objective == 'smartad' else
               conversation(source, CharTokenizer(), source='trusted_teacher_seed', packing='turn'))
        packed = pack_shared_history(row)
        assert [v for v in packed['labels'] if v != -100] == [
            v for turn in row['turns'] for v in turn['labels'] if v != -100]
        assert all(turn['labels'][-1] == 106 for turn in row['turns'])
        assert all(turn['labels'][:turn['prompt_tokens']] == [-100] * turn['prompt_tokens'] for turn in row['turns'])
        rows.append(row)
    return rows


def model_for(adaptation='kang_lora', backend='sdpa', checkpointing=False, dropout=0):
    model = tiny_model()
    model.set_attn_implementation(backend)
    if adaptation == 'kang_lora':
        model = attach_lora(model, rank=64, alpha=128, dropout=dropout, gradient_checkpointing=checkpointing)
        # Exercise gradients of A as well as B; initial PEFT B=0 masks A errors.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'lora_B' in name:
                    p.normal_(std=.01)
    if not checkpointing:
        model.gradient_checkpointing_disable()
    return model


def accumulate(model, rows, indices, objective, scoring, world=1):
    total = 0.0
    fn = smartad_objective if objective == 'smartad' else sft_objective
    from baselines.common.schedule import rank_slots
    units = training_microsteps(rows, indices, objective, scoring)
    # Sequential oracle for rank-slot reductions, including actual dummy graphs.
    for slot in rank_slots(list(range(len(units))), world):
        for index in slot:
            row, weight = units[index if index is not None else 0]
            loss, _ = fn(model, row, 'cpu', scoring=scoring)
            value = loss * (weight / len(indices) if index is not None else 0.)
            total += float(value.detach())
            value.backward()
    return total


@pytest.mark.parametrize('objective', ['sft', 'smartad'])
@pytest.mark.parametrize('adaptation', ['full', 'kang_lora'])
@pytest.mark.parametrize('backend,checkpointing', [('eager', False), ('sdpa', False), ('sdpa', True)])
@pytest.mark.parametrize('indices,world', [([0, 1, 2], 1), ([2], 1), ([2], 2)])
def test_loss_and_every_gradient(objective, adaptation, backend, checkpointing, indices, world):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    rows = examples(objective)
    original = model_for(adaptation, backend, checkpointing)
    packed = deepcopy(original)
    old = accumulate(original, rows, indices, objective, 'per_turn', world)
    new = accumulate(packed, rows, indices, objective, 'shared_history', world)
    assert new == pytest.approx(old, rel=1e-6, abs=1e-6)
    for (name, expected), (other, actual) in zip(original.named_parameters(), packed.named_parameters(), strict=True):
        assert name == other and expected.dtype == actual.dtype == torch.float32
        if expected.grad is None:
            assert actual.grad is None
        else:
            assert_fp32_gradient(actual.grad, expected.grad, name)


@pytest.mark.parametrize('objective', ['sft', 'smartad'])
def test_actual_final_partial_batch_and_accounting(tmp_path, objective):
    rows = examples(objective, 17)  # b'=2; final batch contains one trajectory.
    outputs = {}
    for scoring in ('per_turn', 'shared_history'):
        model = model_for()
        positions = []
        hook = model.register_forward_pre_hook(lambda m, a, kw: positions.append(kw['input_ids'].numel()), with_kwargs=True)
        state = train_loop(model, rows, tmp_path / scoring, objective=objective,
            learning_rate=2e-4, scheduler='linear', warmup_ratio=0, original_batch=4 if objective == 'smartad' else 8,
            epochs=1, packing='turn', scoring=scoring)
        assert state['dropout_equivalence'] == 'exact'
        hook.remove()
        ledger = [json.loads(line) for line in (tmp_path / scoring / 'updates.jsonl').read_text().splitlines()]
        assert ledger[-1]['global_batch'] == state['last_batch'] == 1
        assert state['physical_policy_forward_tokens'] == sum(positions)
        assert state['physical_reference_forward_tokens'] == state['physical_policy_recompute_forward_tokens'] == 0
        if scoring == 'shared_history':
            assert len(positions) == state['examples_seen'] == 17
        outputs[scoring] = (model, state, ledger)
    old, new = outputs['per_turn'], outputs['shared_history']
    for key in ('b_prime', 'total_steps', 'examples_seen', 'supervised_tokens_seen', 'logical_forward_tokens'):
        assert old[1][key] == new[1][key]
    assert new[1]['physical_policy_forward_tokens'] < old[1]['physical_policy_forward_tokens']
    for a, b in zip(old[2], new[2], strict=True):
        assert a['indices'] == b['indices'] and a['lr'] == b['lr']
        assert b['loss'] == pytest.approx(a['loss'], rel=1e-6, abs=1e-6)
    for a, b in zip(old[0].parameters(), new[0].parameters(), strict=True):
        # Nine Adam updates can amplify tiny fp32 summation differences near zero.
        # Loss and every pre-update gradient are checked at 1e-6 above.
        torch.testing.assert_close(a, b, rtol=0, atol=3e-6)


def _distributed_worker(rank, world, rendezvous, output):
    import socket
    import torch.distributed as dist
    # FileStore rendezvous, loopback gloo only; block Python network/API transports.
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', GLOO_SOCKET_IFNAME='lo')
    os.environ.pop('OPENAI_API_KEY', None)
    def forbidden(*args, **kwargs):
        raise AssertionError('External network/API access forbidden')
    socket.create_connection = socket.getaddrinfo = forbidden
    socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = forbidden
    if world > 1:
        dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                                world_size=world, timeout=timedelta(seconds=60))
    try:
        for objective in ('sft', 'smartad'):
            for scoring in ('per_turn', 'shared_history'):
                model = model_for()
                # Two trajectories followed by a partial singleton: test gradient
                # accumulation, multiple updates, and one entirely padded rank.
                import baselines.common.fsdp_train as trainer
                trainer.epoch_batches = lambda *a, **kw: iter([
                    {'epoch': 0, 'indices': [0, 1]}, {'epoch': 0, 'indices': [2]}])
                target = Path(output) / objective / scoring
                state = train_loop(model, examples(objective), target, objective=objective,
                    learning_rate=2e-4, scheduler='constant_with_warmup', warmup_ratio=0,
                    original_batch=8, epochs=1, packing='turn', scoring=scoring,
                    rank=rank, world_size=world, replicated_lora=True)
                torch.save({'params': {n: p.detach().clone() for n, p in model.named_parameters()},
                            'grads': {n: p.grad for n, p in model.named_parameters() if p.requires_grad},
                            'state': state}, target / f'rank-{rank}.pt')
    finally:
        if world > 1:
            dist.destroy_process_group()


def test_two_process_gloo_matches_one_process_lora(tmp_path):
    import torch.multiprocessing as mp
    # Spawn isolates module sampler patches and guarantees actual process collectives.
    mp.spawn(_distributed_worker, args=(1, str(tmp_path / 'store1'), str(tmp_path / 'one')), nprocs=1, join=True)
    mp.spawn(_distributed_worker, args=(2, str(tmp_path / 'store2'), str(tmp_path / 'two')), nprocs=2, join=True)
    for objective in ('sft', 'smartad'):
        for scoring in ('per_turn', 'shared_history'):
            relative = Path(objective) / scoring
            expected = torch.load(tmp_path / 'one' / relative / 'rank-0.pt', weights_only=True)
            ranks = [torch.load(tmp_path / 'two' / relative / f'rank-{r}.pt', weights_only=True) for r in range(2)]
            assert ranks[0]['state']['zero_weight_slots'] > 0
            for actual in ranks:
                for name in expected['params']:
                    torch.testing.assert_close(actual['params'][name], expected['params'][name], rtol=0, atol=1e-6)
                    assert torch.equal(ranks[0]['params'][name], ranks[1]['params'][name])
                for name in expected['grads']:
                    assert_fp32_gradient(actual['grads'][name], expected['grads'][name], name)
            old = [json.loads(s) for s in (tmp_path / 'one' / relative / 'updates.jsonl').read_text().splitlines()]
            new = [json.loads(s) for s in (tmp_path / 'two' / relative / 'updates.jsonl').read_text().splitlines()]
            for a, b in zip(old, new, strict=True):
                assert a['indices'] == b['indices'] and a['global_batch'] == b['global_batch']
                assert b['loss'] == pytest.approx(a['loss'], rel=1e-6, abs=1e-6)


@pytest.mark.parametrize('objective', ['sft', 'smartad'])
def test_gpu_acceptance_path_uses_real_objective_on_cpu(objective):
    from tools import validate_shared_history_gpu as gate
    model = model_for()
    row = examples(objective)[2]
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    record = gate.compare_sft(model, row, parameters, 'cpu', objective=objective)
    summary = gate.summarize([record])
    assert summary['passed']
    for comparison in summary['comparisons'].values():
        assert comparison['max_loss_relative_difference'] < 1e-6
        assert comparison['max_gradient_relative_l2'] < 1e-6
        assert comparison['tokens']['mean_absolute_difference_nats_per_token'] < 1e-6
    expected = accumulate(model, [row], [0], objective, 'per_turn')
    for scoring in gate.SCORINGS:
        path = record['paths'][scoring]
        assert path['loss'] == pytest.approx(expected, rel=1e-6, abs=1e-6)
        for measurement in path['measurements'].values():
            assert measurement['forward_calls'] == (1 if scoring == 'shared_history' else len(row['turns']))
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])


def test_dropout_mean_gradients_agree(record_property):
    from test_shared_history_torch import random_conversation
    # Short, unequal turns keep 256 independent samples fast on one CPU thread.
    # Use actual Gemma attention, Kang LoRA targets, packing and SFT reduction.
    rows = [random_conversation(7, turns=3, prefix=3)]
    model = model_for(dropout=.05).train()
    # A small nonzero adapter keeps this randomly initialized diagnostic near
    # its base while still exercising both A and B gradients (PEFT starts B=0).
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.mul_(.1)
    parameters = [p for p in model.parameters() if p.requires_grad]
    count = 256
    differences, squares, totals = None, None, None
    for seed in range(count):
        gradients = []
        for path, scoring in enumerate(('per_turn', 'shared_history')):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(1000 + 2 * seed + path)  # Disjoint deterministic streams.
            accumulate(model, rows, [0], 'sft', scoring)
            gradients.append(torch.cat([p.grad.flatten() for p in parameters]).double())
        delta = gradients[1] - gradients[0]
        if differences is None:
            differences, squares, totals = torch.zeros_like(delta), torch.zeros_like(delta), torch.zeros_like(delta)
        differences += delta
        squares += delta.square()
        totals += gradients[0]
    mean = differences / count
    variance = ((squares - differences.square() / count) / (count - 1)).clamp_min(0)
    standard_error = (variance / count).sqrt()
    reference = totals / count
    assert torch.isfinite(mean).all() and standard_error.norm() > 0
    record_property('seeds', count)
    record_property('mean_gradient_relative_l2', float(mean.norm() / reference.norm()))
    record_property('standard_error_relative_l2', float(standard_error.norm() / reference.norm()))
    # Six standard errors per element accommodate simultaneous comparisons;
    # the tensor checks also bound the aggregate error and test sensitivity.
    assert torch.all(mean.abs() <= 6 * standard_error + 1e-7)
    # Bound overall sensitivity; individual tensors can have near-zero means.
    assert standard_error.norm() <= .1 * reference.norm()
    offset = 0
    for parameter in parameters:
        stop = offset + parameter.numel()
        error, se = mean[offset:stop], standard_error[offset:stop]
        assert error.norm() <= 4 * se.norm() + 1e-7
        offset = stop


@pytest.mark.parametrize('scoring,world,rank', [('shared_history', 1, 0), ('per_turn', 2, 0),
                                             ('shared_history', 2, 0), ('shared_history', 2, 1)])
def test_nonzero_dropout_trains_reproducibly_per_rank(tmp_path, monkeypatch, scoring, world, rank):
    from test_shared_history_torch import random_conversation
    import baselines.common.fsdp_train as trainer
    # Only collectives are faked here; the separate real Gloo test is unchanged.
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda values: None)
    monkeypatch.setattr(torch.distributed, 'all_gather_object',
                        lambda out, value: out.__setitem__(slice(None), [value] * world))
    monkeypatch.setattr(trainer, 'epoch_batches', lambda *a, **kw: iter([{'epoch': 0, 'indices': [0, 1]}]))
    rows = [random_conversation(i, turns=2, prefix=3) for i in range(2)]
    original = model_for(dropout=.05)
    snapshots, seeds = [], []
    import transformers
    set_seed = transformers.set_seed
    def capture_seed(seed):
        seeds.append(seed)
        set_seed(seed)
    monkeypatch.setattr(transformers, 'set_seed', capture_seed)
    for repeat in range(2):
        model = deepcopy(original)
        torch.manual_seed(900 + repeat)  # Ambient RNG must not affect this run.
        state = train_loop(model, rows, tmp_path / str(repeat), objective='sft',
            learning_rate=2e-4, scheduler='linear', warmup_ratio=0, original_batch=8, epochs=1,
            seed=42, scoring=scoring, rank=rank, world_size=world, replicated_lora=True)
        assert state['dropout_equivalence'] == 'expected_gradient'
        assert state['rng_seed_policy'] == 'seed + rank' and state['rng_seed'] == 42 + rank
        snapshots.append([p.detach().clone() for p in model.parameters() if p.requires_grad])
    assert seeds == [42 + rank, 42 + rank]
    assert any(not torch.equal(p, result) for p, result in zip(
        (p for p in original.parameters() if p.requires_grad), snapshots[0], strict=True))
    assert all(torch.equal(a, b) for a, b in zip(*snapshots, strict=True))


@pytest.mark.parametrize('dropout,expected', [(.05, 'expected_gradient'), (0, 'exact')])
def test_lora_training_provenance_records_equivalence(tmp_path, monkeypatch, dropout, expected):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import baselines.common.fsdp_train as trainer
    from baselines.common import prepared
    from sed.storage import digest
    from test_shared_history_torch import random_conversation
    # Exercise production provenance/export with fake loading and CPU execution.
    # Zero dropout is a diagnostic fixture, never a production recipe override.
    args = SimpleNamespace(snapshot=str(tmp_path / 'base'), base_snapshot=str(tmp_path / 'base'),
        config='fake-config', verified=str(tmp_path / 'verified'), method_data=None,
        objective='sft', adaptation='kang_lora', scoring='shared_history', packing='turn',
        output=str(tmp_path / 'output'), original_batch=8, epochs=1, seed=42, max_length=32768,
        learning_rate=2e-4, scheduler='linear', warmup_ratio=0, warmup_steps=0,
        weight_decay=0, flex_kernel_options=None)
    config = {'training': {'thinking_mode': 'off', 'student_step_token_cap': 1024},
              'environment': {'max_episode_steps': 40}, 'models': {'student': {'revision': 'fake'}},
              'paths': {'manifest': 'fake-manifest', 'split_freeze': 'fake-freeze'}}
    row = random_conversation(7, turns=2, prefix=3)
    row['label_provenance'] = {'source': 'synthetic_test'}
    model = model_for(dropout=dropout)
    for name in ('RANK', 'LOCAL_RANK'):
        monkeypatch.setenv(name, '0')
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda _: None)
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: nullcontext())
    # This numerical/provenance test supplies prepared data at the verification
    # boundary. Real seal refusal and collection paths have separate coverage.
    seal = {'path': 'synthetic-seal', 'sha256': 'fixture', 'encoded_examples_sha256': digest([row])}
    monkeypatch.setattr(prepared, 'load_prepared', lambda _: ((config,
        {'K': 1, 'unique_instances': 1, 'task_groups': 1}, [row], 'fake-template'), seal))
    monkeypatch.setattr(trainer, 'prepare_lora_model', lambda *a, **kw: model)
    monkeypatch.setattr(trainer, 'snapshot_metadata', lambda _: {'fixture': True})
    monkeypatch.setattr(trainer, 'artifact_hashes', lambda _: {})
    captured = {}
    def provenance(output, **kwargs):
        captured.update(command=kwargs['command'], config=deepcopy(kwargs['config']))
        return captured
    monkeypatch.setattr(trainer, 'write_provenance', provenance)
    real_loop = trainer.train_loop
    def cpu_loop(*a, **kw):
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        return real_loop(*a, **{**kw, 'device': 'cpu'})
    monkeypatch.setattr(trainer, 'train_loop', cpu_loop)
    trainer.train_lora(args)
    training = captured['config']['baseline_training']
    assert training['prepared_seal'] == seal
    assert captured['config']['sft']['dropout'] == dropout
    assert training['dropout_equivalence'] == expected
    assert training['rng_seed_policy'] == 'seed + rank' and training['rank_seeds'] == [42]
    state = json.loads((Path(args.output) / 'checkpoint/training_state.json').read_text())
    assert state['dropout_equivalence'] == expected
    assert state['identity_sha256'] == digest(captured)
