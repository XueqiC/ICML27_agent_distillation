"""Continuation tests: unchanged grid trainer, analytic CPU updates, no API/GPU."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repair.artifacts import checkpoint_files
from repair.train_arms import (batch_schedule, accounting, inherited_recipe, run_arm,
                               CHECKPOINTS, load_existing_adapter)
from repair.e2_reader import prepare_examples
from sed.storage import digest, file_hash
from test_repair_e2_reader import adapter_fixture, corpus, read, save


def test_inherit_actual_checkpoint_recipe_and_parameter_names_without_loading(tmp_path):
    theta, snapshot, config = adapter_fixture(tmp_path)
    spec = inherited_recipe(theta, config, snapshot)
    assert spec['optimizer_state']['mode'] == 'identical_reinitialization_for_both'
    assert len(spec['trainable_modules']) == 7 and len(spec['parameters']) == 14
    assert sum(spec['parameters'].values()) == 14 * 64
    assert all(n.endswith('.default.weight') for n in spec['parameters'])
    (theta / 'optimizer.pt').write_bytes(b'never loaded in dry-run')
    assert inherited_recipe(theta, config, snapshot)['optimizer_state']['mode'] == 'restore_for_both'
    (theta / 'optimizer.bin').write_bytes(b'unknown')
    with pytest.raises(ValueError, match='Unrecognized optimizer'):
        inherited_recipe(theta, config, snapshot)


def test_same_order_composition_dry_run_exact_exposure(corpus, tmp_path, monkeypatch):
    import repair.train_arms as training
    def forbidden(*a, **k): raise AssertionError('Model or optimizer load in dry-run')
    monkeypatch.setattr(training, 'load_existing_adapter', forbidden)
    import torch
    monkeypatch.setattr(torch, 'load', forbidden)
    checked = read(corpus)
    manifests, schedules = [], []
    for arm in ('N', 'F'):
        rows, slots = prepare_examples(checked, corpus.tokenizer, arm, packing='turn', max_length=32768)
        batches = batch_schedule(corpus.data['D_repair_train'], slots)
        schedules.append(batches)
        assert len(batches) == 20
        for batch in batches:
            assert len(batch['indices']) == 4
            assert sum(s['bucket'] == 'repair' for s in batch['selections']) == 3
            assert sum(s['bucket'] == 'seed' for s in batch['selections']) == 1
            assert all(s['task_id'] in corpus.data['D_repair_train'] for s in batch['selections'])
        counts = accounting(rows, batches)
        assert set(counts) == set(CHECKPOINTS)
        assert counts[20]['supervised_target_tokens'] == sum(
            rows[i]['supervised_tokens'] for b in batches for i in b['indices'])
        manifest = run_arm(checked, corpus.config, tmp_path / arm, theta0=corpus.theta,
            snapshot=corpus.snapshot, tokenizer=corpus.tokenizer, arm=arm, dry_run=True)
        manifests.append(manifest)
        assert manifest['format'] == 'repair_e2_training_run'
        assert manifest['theta0'] == corpus.data['theta0']
        assert manifest['training_collection_sha256'] == file_hash(corpus.path)
        assert manifest['trainable_parameter_count'] == 896
        assert manifest['training_exposure'] == 0 and manifest['planned_training_exposure'] == 80
        assert [p['update'] for p in manifest['checkpoints']] == [1, 5, 20]
        assert all(p['sha256'] is None and p['status'] == 'planned' for p in manifest['checkpoints'])
        assert json.loads((tmp_path / arm / 'batches.json').read_text()) == batches
        with pytest.raises(FileExistsError):
            run_arm(checked, corpus.config, tmp_path / arm, theta0=corpus.theta,
                snapshot=corpus.snapshot, tokenizer=corpus.tokenizer, arm=arm, dry_run=True)
    assert schedules[0] == schedules[1]
    assert manifests[0]['schedule_sha256'] == manifests[1]['schedule_sha256']
    assert manifests[0]['optimizer_state'] == manifests[1]['optimizer_state']
    assert manifests[0]['scheduler_policy'] == manifests[1]['scheduler_policy']
    assert manifests[0]['checkpoints'][-1]['supervised_target_tokens'] != manifests[1]['checkpoints'][-1]['supervised_target_tokens']


def test_existing_trainer_fixed_batches_adamw_and_loss_normalization(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import baselines.common.fsdp_train as trainer
    # Analytic fake logp with unequal target lengths. The production SFT
    # objective must reduce token means per episode, then average 3:1 episodes.
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.tensor(.3))
            self.base = torch.nn.Parameter(torch.tensor(3.), requires_grad=False)
    def row(n, target):
        return {'input_ids': [1] * (n + 1), 'labels': [-100] + [1] * n, 'packing': 'episode',
                'supervised_tokens': n, 'target': target}
    examples = [row(2, 1.), row(20, -1.)]
    monkeypatch.setattr(trainer, 'batch_logps', lambda model, rows, device, **kw:
                        torch.stack([-((model.adapter - r['target']) ** 2) * r['supervised_tokens'] for r in rows]))
    batches = [{'epoch': 0, 'indices': [0, 0, 0, 1]} for _ in range(20)]
    model, expected = Tiny(), Tiny()
    opt = torch.optim.AdamW([expected.adapter], lr=5e-5, weight_decay=0, foreach=False, fused=False)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: max(0., (20 - i) / 20))
    for _ in range(20):
        opt.zero_grad()
        (.75 * (expected.adapter - 1.) ** 2 + .25 * (expected.adapter + 1.) ** 2).backward()
        torch.nn.utils.clip_grad_norm_([expected.adapter], 1.)
        opt.step(); schedule.step()
    snapshots = []
    state = trainer.train_loop(model, examples, tmp_path / 'tiny', objective='sft', learning_rate=5e-5,
        scheduler='linear', warmup_ratio=0, warmup_steps=0, original_batch=4, seed=1,
        fixed_batches=batches, checkpoint_callback=lambda m, s: snapshots.append(s.copy()))
    assert torch.equal(model.adapter, expected.adapter) and model.base.item() == 3.
    assert state['updates'] == 20 and state['examples_seen'] == 80 and state['b_prime'] == 4
    assert state['supervised_tokens_seen'] == 20 * (3 * 2 + 20)
    assert snapshots[0]['loss_normalization'] == 'per_example_mean_then_global_batch_mean'
    assert [s['updates'] for s in snapshots if s['updates'] in CHECKPOINTS] == [1, 5, 20]
    assert all(p.device.type == 'cpu' for p in model.parameters())


def test_publish_real_adapter_artifacts_at_fixed_updates_and_serving_identity(tmp_path):
    from repair.train_arms import publish_checkpoint
    from harness.lora_adapters import inspect_adapter
    import shutil
    theta, snapshot, config = adapter_fixture(tmp_path)
    inherited = inherited_recipe(theta, config, snapshot)
    prov = json.loads((theta.parent / 'provenance.json').read_text())
    class FakeAdapter:
        def save_pretrained(self, target, safe_serialization):
            assert safe_serialization
            for name in ['adapter_config.json', 'adapter_model.safetensors']:
                shutil.copyfile(theta / name, target / name)
    output = tmp_path / 'saved'; output.mkdir()
    for update in [1, 5, 20]:
        item = publish_checkpoint(FakeAdapter(), output, prov,
            {'updates': update, 'supervised_tokens_seen': update * 10},
            {'optimizer_updates': update, 'processed_context_tokens': update * 100,
             'supervised_target_tokens': update * 10}, inherited)
        inspected = inspect_adapter('arm-N', item['path'], config['models']['student'], snapshot)
        assert item['sha256'] == checkpoint_files(item['path'])
        assert inspected['adapter_weights_sha256'] == item['sha256']['adapter_model.safetensors']
        assert item['trainer_state']['trainable_parameter_count'] == 896
        assert item['trainer_state']['processed_context_tokens'] == update * 100
    assert not (output / 'update-00').exists()
    with pytest.raises(FileExistsError):
        publish_checkpoint(FakeAdapter(), output, prov, {'updates': 1}, {}, inherited)


def test_optimizer_moments_restore_identically_for_both_arms(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    import baselines.common.fsdp_train as trainer
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.3))
    row = {'input_ids': [1, 2], 'labels': [-100, 2], 'supervised_tokens': 1, 'packing': 'episode'}
    monkeypatch.setattr(trainer, 'batch_logps', lambda m, rows, device, **kw: -(m.weight - 1.).square().reshape(1))
    original = Tiny()
    opt = torch.optim.AdamW(original.parameters(), lr=5e-5, weight_decay=0, foreach=False, fused=False)
    (original.weight - 1.).square().backward(); opt.step()
    saved = deepcopy(opt.state_dict())
    # A completed theta0 schedule has zero current LR. Both arms must restart
    # the base LR without throwing away these moments or aliasing them.
    saved['param_groups'][0]['lr'] = 0.
    saved['param_groups'][0]['initial_lr'] = 5e-5
    original_moment = saved['state'][0]['exp_avg'].clone()
    models = [Tiny(), Tiny()]
    for arm, model in zip(['N', 'F'], models):
        trainer.train_loop(model, [row], tmp_path / arm, objective='sft', learning_rate=5e-5,
            scheduler='linear', warmup_ratio=0, warmup_steps=0, original_batch=4, seed=1,
            fixed_batches=[{'indices': [0, 0, 0, 0], 'epoch': 0}], optimizer_state=deepcopy(saved))
    assert torch.equal(models[0].weight, models[1].weight) and models[0].weight.item() != .3
    assert torch.equal(saved['state'][0]['exp_avg'], original_moment)
    ledger = json.loads((tmp_path / 'N' / 'updates.jsonl').read_text().splitlines()[0])
    assert ledger['lr'] == 5e-5
    wrong = deepcopy(saved); wrong['param_groups'][0]['weight_decay'] = .1
    with pytest.raises(ValueError, match='Saved optimizer differs'):
        trainer.train_loop(Tiny(), [row], tmp_path / 'wrong', objective='sft', learning_rate=5e-5,
            scheduler='linear', warmup_ratio=0, original_batch=4,
            fixed_batches=[{'indices': [0, 0, 0, 0]}], optimizer_state=wrong)


@pytest.mark.parametrize('field', ['rank', 'alpha', 'dropout', 'scheduler', 'packing', 'max_length', 'gradient_clip'])
def test_recipe_comes_from_this_tree_not_redeclared(tmp_path, monkeypatch, field):
    from baselines.kang.pipeline import RECIPE
    theta, snapshot, config = adapter_fixture(tmp_path)
    spec = inherited_recipe(theta, config, snapshot)
    assert spec['recipe_sources']['baselines/common/fsdp_train.py'] == file_hash('baselines/common/fsdp_train.py')
    monkeypatch.setitem(RECIPE, field, 'changed_existing_recipe')
    with pytest.raises(ValueError):
        inherited_recipe(theta, config, snapshot)


def test_continuation_loads_exactly_existing_adapter(tmp_path, monkeypatch):
    import torch
    import peft
    from train import modeling
    import baselines.common.shared_history as shared
    theta, snapshot, config = adapter_fixture(tmp_path)
    spec = inherited_recipe(theta, config, snapshot)
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.tensor(.2))
            self.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(use_cache=True))
            self.peft_config = {'default': SimpleNamespace(target_modules=[])}
        def set_attn_implementation(self, name): assert name == 'flex_attention'
        def gradient_checkpointing_enable(self, **kw): assert kw == {'gradient_checkpointing_kwargs': {'use_reentrant': False}}
        def enable_input_require_grads(self): pass
    base = Fake(); calls = []
    def load(*args, **kwargs):
        assert args == (base, theta)
        assert kwargs == {'is_trainable': True, 'local_files_only': True}
        calls.append(args)
        return base
    def forbidden(*a, **kw): raise AssertionError('New adapter creation forbidden')
    monkeypatch.setattr(modeling, 'load_student', lambda *a, **kw: base)
    monkeypatch.setattr(peft.PeftModel, 'from_pretrained', load)
    monkeypatch.setattr(peft, 'get_peft_model', forbidden)
    monkeypatch.setattr(modeling, 'attach_lora', forbidden)
    monkeypatch.setattr(shared, 'validate_model', lambda m: None)
    monkeypatch.setattr(shared, 'configure_flex_attention', lambda *a: None)
    assert load_existing_adapter(snapshot, theta, 'cpu', spec['recipe']) is base
    assert len(calls) == 1
    base.peft_config['extra'] = SimpleNamespace()
    with pytest.raises(ValueError, match='exactly the original'):
        load_existing_adapter(snapshot, theta, 'cpu', spec['recipe'])


def test_cli_dry_run_uses_interchange_gate(corpus, tmp_path, monkeypatch):
    from tools import repair_train_arm as cli
    monkeypatch.setattr(cli, 'load_config', lambda p: corpus.config)
    monkeypatch.setattr(cli, 'load_tokenizer', lambda p: corpus.tokenizer)
    args = ['--theta0', str(corpus.theta), '--snapshot', str(corpus.snapshot), '--collection', str(corpus.path),
            '--repair-splits', str(corpus.repair_splits), '--output', str(tmp_path / 'cli'), '--arm', 'N', '--dry-run',
            '--path-map', '='.join(corpus.path_maps[0])]
    assert cli.main(args) == 0
    corpus.data['dry_run'] = True; save(corpus)
    with pytest.raises(ValueError, match='Synthetic'):
        cli.main(args)


def test_production_orchestration_fixed_checkpoints_with_mock_step(corpus, tmp_path, monkeypatch):
    import contextlib
    import shutil
    import torch
    import torch.distributed as dist
    import transformers
    import baselines.common.fsdp_train as trainer
    import repair.train_arms as training
    inherited = inherited_recipe(corpus.theta, corpus.config, corpus.snapshot)
    class Adapter:
        def named_parameters(self):
            return [(n, torch.nn.Parameter(torch.zeros(count))) for n, count in inherited['parameters'].items()]
        def parameters(self): return [p for _, p in self.named_parameters()]
        def save_pretrained(self, target, safe_serialization):
            assert safe_serialization
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                shutil.copyfile(corpus.theta / name, target / name)
    model = Adapter(); calls = []
    monkeypatch.setenv('WORLD_SIZE', '2'); monkeypatch.setenv('RANK', '0'); monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: None)
    monkeypatch.setattr(transformers, 'set_seed', lambda *a: None)
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(torch.random, 'fork_rng', lambda **kw: contextlib.nullcontext())
    for name in ('init_process_group', 'new_group', 'broadcast', 'broadcast_object_list', 'destroy_process_group'):
        monkeypatch.setattr(dist, name, lambda *a, **kw: None)
    monkeypatch.setattr(training, 'load_existing_adapter', lambda *a: model)
    def step(model, examples, output, **kw):
        calls.append(kw)
        assert kw['original_batch'] == 4 and kw['seed'] == 1 and kw['replicated_lora']
        recipe = inherited['recipe']
        for key in ('objective', 'learning_rate', 'scheduler', 'warmup_ratio', 'warmup_steps',
                    'weight_decay', 'packing', 'scoring'):
            assert kw[key] == recipe[key]
        counts = accounting(examples, kw['fixed_batches'])
        for update in range(1, 21):
            state = {'updates': update, 'examples_seen': update * 4}
            if update in counts:
                state.update(supervised_tokens_seen=counts[update]['supervised_target_tokens'],
                             physical_policy_forward_tokens=counts[update]['processed_context_tokens'])
            kw['checkpoint_callback'](model, state)
        return state
    monkeypatch.setattr(trainer, 'train_loop', step)
    out = tmp_path / 'production-mocked'
    result = run_arm(read(corpus), corpus.config, out, theta0=corpus.theta, snapshot=corpus.snapshot,
                     tokenizer=corpus.tokenizer, arm='N')
    assert len(calls) == 1 and result['status'] == 'complete' and result['training_exposure'] == 80
    for row in result['checkpoints']:
        assert row['status'] == 'saved' and row['sha256'] == checkpoint_files(row['path'])
        assert (out / f"update-{row['update']:02d}" / 'manifest.json').is_file()
    assert sorted(p.name for p in out.glob('update-*')) == ['update-01', 'update-05', 'update-20']
    # A newly created adapter/tensor is refused before any update/publication.
    model.named_parameters = lambda: [('unexpected_adapter.weight', torch.nn.Parameter(torch.zeros(1)))]
    with pytest.raises(ValueError, match='Loaded trainable tensors differ'):
        run_arm(read(corpus), corpus.config, tmp_path / 'bad-model', theta0=corpus.theta,
                snapshot=corpus.snapshot, tokenizer=corpus.tokenizer, arm='F')
    assert len(calls) == 1 and not (tmp_path / 'bad-model').exists()


def test_real_peft_continuation_preserves_every_adapter_tensor(tmp_path, monkeypatch):
    import torch
    import peft
    from train import modeling
    from test_validate_shared_history_gpu import tiny_model
    from baselines.kang.pipeline import RECIPE
    original = modeling.attach_lora(tiny_model(), rank=RECIPE['rank'], alpha=RECIPE['alpha'], dropout=RECIPE['dropout'])
    with torch.no_grad():
        for name, parameter in original.named_parameters():
            if parameter.requires_grad: parameter.normal_(std=.01)
    expected = {n: p.detach().clone() for n, p in original.named_parameters() if p.requires_grad}
    directory = tmp_path / 'adapter'; original.save_pretrained(directory, safe_serialization=True)
    monkeypatch.setattr(modeling, 'load_student', lambda *a, **kw: tiny_model())
    def forbidden(*a, **kw): raise AssertionError('New adapter entrypoint forbidden')
    monkeypatch.setattr(peft, 'get_peft_model', forbidden)
    monkeypatch.setattr(modeling, 'attach_lora', forbidden)
    loaded = load_existing_adapter('unused-local-base', directory, 'cpu', {
        'frozen_base_dtype': 'bfloat16', 'adapter_master_dtype': 'float32', 'attention': 'flex_attention'})
    actual = {n: p for n, p in loaded.named_parameters() if p.requires_grad}
    assert actual.keys() == expected.keys() and set(loaded.peft_config) == {'default'}
    assert all(torch.equal(actual[n], value) for n, value in expected.items())
    assert all(not p.requires_grad for n, p in loaded.named_parameters() if 'lora_' not in n)
