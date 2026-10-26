"""M0c CPU contract tests; all execution receipts come from fake environments."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import random
import shutil

import pytest

from repair.artifacts import seal, checkpoint_files
from repair.e2_reader import BRANCHES, BRANCH_FIELDS, prepare_examples, read_collection
from repair.train_arms import (run_arm, batch_schedule, base_recipe, load_fresh_adapter,
                               initial_adapter_hash, LORA_INIT_SEED)
from sed.storage import digest, file_hash, write_json
from test_repair_e2_reader import build_corpus, read, save

_TEMPLATE = None


@pytest.mark.parametrize('branch,protocol,kind,scope', product(
    ['P0_K20', 'P1_U_mech', 'P0_K0'], ['P0', 'P1'], ['adapter', 'base_model'],
    ['repair_tasks', 'original_support']))
def test_exact_branch_combinations(branch, protocol, kind, scope):
    from repair.e2_reader import branch_metadata
    data = {'schema_version': 2, 'branch': branch, 'protocol': protocol,
            'theta0_kind': kind, 'seed_replay_scope': scope}
    allowed = {('P0_K20', 'P0', 'adapter', 'repair_tasks'),
               ('P1_U_mech', 'P1', 'adapter', 'original_support'),
               ('P0_K0', 'P0', 'base_model', 'repair_tasks')}
    if (branch, protocol, kind, scope) in allowed:
        assert branch_metadata(data) == {k: v for k, v in data.items() if k != 'schema_version'}
    else:
        with pytest.raises(ValueError): branch_metadata(data)


@pytest.fixture
def schema2(tmp_path, tmp_path_factory):
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = build_corpus(tmp_path_factory.mktemp('schema2-template'), task_count=44)
    c = deepcopy(_TEMPLATE)
    old = c.path.parent
    shutil.copytree(old, tmp_path, dirs_exist_ok=True)
    c.local = lambda p: tmp_path / Path(p).relative_to(old)
    c.path_maps = [(str(old), str(tmp_path))]
    for key in ('path', 'theta', 'snapshot', 'repair_splits', 'support', 'view'):
        setattr(c, key, c.local(getattr(c, key)))
    for key in ('manifest', 'split_freeze'):
        c.config['paths'][key] = str(c.local(c.config['paths'][key]))
    return c


def reseal(c):
    path = c.path.with_name('collection.json')
    write_json(path, seal(c.source))
    c.data.update(source_collection_sha256=file_hash(path),
                  source_collection_content_sha256=seal(c.source)['content_sha256'])
    save(c)


def as_branch(c, branch):
    from train.modeling import snapshot_metadata
    from repair.e2_reader import public_episode
    from sed.seed_evidence import checked_episode
    meta = dict(zip(BRANCH_FIELDS, (branch, *BRANCHES[branch])))
    c.data.update(schema_version=2, **meta)
    c.source.update(meta)
    if branch == 'P0_K0':
        base = snapshot_metadata(c.snapshot)
        student = c.config['models']['student']
        theta = {'identity': {'base_model': student['requested_repo'], 'revision': student['revision'],
                              'metadata_sha256': base['metadata_sha256'],
                              'weight_verification': base['weight_verification']},
                 'files': base['metadata_sha256']}
        c.data['theta0'] = c.source['checkpoint'] = theta
        c.theta = None
    if branch == 'P1_U_mech':
        original = json.loads((c.view / 'manifest.json').read_text())
        parent = json.loads((c.support / 'manifest.json').read_text())
        support_ids = {r['task_id'] for r in original['verified']}
        mech = [r for r in parent['verified'] if r['task_id'] not in support_ids][:20]
        order = [r['task_id'] for r in mech[:6]]
        probes = sorted(order, key=lambda t: (digest([20261029, t]), t))[:2]
        train = [t for t in order if t not in probes]
        split = {**c.data['split'], 'failure_order': order, 'D_repair_probe': probes, 'training_candidates': train}
        c.data.update(split=split, D_repair_train=train, D_repair_probe=probes)
        c.source.update(split=split, D_repair_train=train, D_repair_probe=probes)
        references = []
        for receipt in original['verified']:
            episode = checked_episode(c.support, receipt)
            references.append({'task_id': receipt['task_id'], 'episode_id': episode['episode_id'],
                'verified_collection_path': str(c.view), 'verified_manifest_sha256': file_hash(c.view / 'manifest.json'),
                'evidence_root': str(c.support), 'episode_dir': receipt['episode_dir'],
                'episode_sha256': file_hash(c.support / receipt['episode_dir'] / 'episode.json'),
                'receipt': receipt, 'receipt_sha256': digest(receipt)})
        for arm, entries in c.data['arms'].items():
            for entry, tid in zip(entries, train):
                receipt = next(r for r in mech if r['task_id'] == tid)
                episode = public_episode(checked_episode(c.support, receipt))
                # Only the fixture constructs these repairs from fake execution.
                episode['steps'][0]['action_raw'] = 'Thought: Repair ' + arm + '.\nAction: look'
                le_id = digest([arm, tid])
                entry.update(task_id=tid, group_id=receipt['group_id'], episode=episode,
                    episode_sha256=digest(episode), learning_episode_id=le_id, trajectory_id=arm + tid,
                    repair_id='repair-' + le_id, seed_demonstrations=deepcopy(references))
                entry['human_review']['episode_sha256'] = digest(episode)
                entry['human_review']['decision'].update(trajectory_id=arm + tid, episode_id=le_id)
        splits = json.loads(c.repair_splits.read_text()); splits.pop('content_sha256')
        splits['sets']['U_mech'] = {'task_ids': [r['task_id'] for r in mech],
                                    'group_ids': [r['group_id'] for r in mech]}
        splits['E'] = {'group_ids': [r['group_id'] for r in original['verified']]}
        write_json(c.repair_splits, seal(splits))
        c.data['input_hashes'][str(c.repair_splits)] = file_hash(c.repair_splits)
        c.source['input_hashes'] = c.data['input_hashes']
        replay = [r['task_id'] for r in original['verified']]
    else:
        replay = c.data['D_repair_train']
    for entries in c.data['arms'].values():
        for entry in entries:
            entry.update(**meta, design_bucket='G' if branch == 'P1_U_mech' else 'R', seed_design_bucket='D')
    c.data.update(seed_replay_task_ids=replay, seed_replay_tasks_are_repair_tasks=branch != 'P1_U_mech',
        original_teacher_far_policy='excluded_without_replay_pair_check',
        supervision_accounting={'initial_K': 20, 'theta0_seed_training_trajectories': 0 if branch == 'P0_K0' else 20,
                                'training_exposure': 0})
    reseal(c)
    return c


@pytest.mark.parametrize('branch', BRANCHES)
def test_schema2_branches_reader_and_manifests(schema2, branch, tmp_path, monkeypatch):
    c = as_branch(schema2, branch)
    checked = read(c)
    import repair.train_arms as training
    def forbidden(*a, **kw):
        raise AssertionError('No model allocation during dry-run')
    monkeypatch.setattr(training, 'load_fresh_adapter', forbidden)
    monkeypatch.setattr(training, 'load_existing_adapter', forbidden)
    manifests = [run_arm(checked, c.config, tmp_path / arm, theta0=c.theta, snapshot=c.snapshot,
                        tokenizer=c.tokenizer, arm=arm, dry_run=True) for arm in ('N', 'F')]
    for result in manifests:
        for key in BRANCH_FIELDS:
            assert result[key] == c.data[key]
        assert result['seed_replay_task_ids'] == c.data['seed_replay_task_ids']
        assert result['training_exposure'] == 0
        assert result['repair_training_unique_instances'] == 4
        assert result['training_unique_instances'] == (24 if branch == 'P1_U_mech' else 4)
        assert result['replay_trusted_seed_trajectories'] == (20 if branch == 'P1_U_mech' else 4)
    assert manifests[0]['schedule_sha256'] == manifests[1]['schedule_sha256']
    if branch == 'P0_K0':
        assert manifests[0]['optimizer_state']['mode'] == 'fresh_AdamW_for_both'
        assert manifests[0]['initialization_seed'] == LORA_INIT_SEED
        assert manifests[0]['initial_adapter_sha256'] is None  # planned is not an actual tensor hash
        assert [r['update'] for r in manifests[0]['checkpoints']] == [0, 1, 5, 20]
        assert not (tmp_path / 'N/update-00').exists()


@pytest.mark.parametrize('branch', BRANCHES)
@pytest.mark.parametrize('field', BRANCH_FIELDS)
def test_schema2_inconsistent_and_missing_branch_fields(schema2, branch, field):
    c = as_branch(schema2, branch)
    for value in ('invalid', None):
        c.data[field] = value
        save(c)
        with pytest.raises(ValueError):
            read(c)
    del c.data[field]; save(c)
    with pytest.raises(ValueError):
        read(c)


@pytest.mark.parametrize('branch', ['P1_U_mech', 'P0_K0'])
def test_schema1_cannot_masquerade_as_new_branch(schema2, branch):
    c = as_branch(schema2, branch)
    c.data['schema_version'] = 1; save(c)
    with pytest.raises(ValueError, match='Schema 1'):
        read(c)


@pytest.mark.parametrize('where', ['source', 'entry'])
def test_schema1_rejects_conflicting_nested_branch_metadata(schema2, where):
    c = schema2
    target = c.source if where == 'source' else c.data['arms']['N'][0]
    target['branch'] = 'P1_U_mech'
    reseal(c)
    with pytest.raises(ValueError, match='branch metadata'):
        read(c)


@pytest.mark.parametrize('branch', BRANCHES)
@pytest.mark.parametrize('case', ['source_branch', 'entry_branch', 'design', 'seed_design', 'replay_list',
                                 'replay_flag', 'far_policy', 'accounting', 'theta_exposure', 'review', 'hash', 'template'])
def test_schema2_contract_rejections(schema2, branch, case):
    c = as_branch(schema2, branch)
    if case == 'source_branch': c.source['protocol'] = 'wrong'
    elif case == 'entry_branch': c.data['arms']['N'][0]['theta0_kind'] = 'wrong'
    elif case == 'design': c.data['arms']['N'][0]['design_bucket'] = 'wrong'
    elif case == 'seed_design': c.data['arms']['F'][0]['seed_design_bucket'] = 'G'
    elif case == 'replay_list': c.data['seed_replay_task_ids'] = c.data['D_repair_probe']
    elif case == 'replay_flag': c.data['seed_replay_tasks_are_repair_tasks'] = not c.data['seed_replay_tasks_are_repair_tasks']
    elif case == 'far_policy': c.data['original_teacher_far_policy'] = 'allow'
    elif case == 'accounting': c.data['supervision_accounting']['training_exposure'] = 1
    elif case == 'theta_exposure': c.data['supervision_accounting']['theta0_seed_training_trajectories'] = -1
    elif case == 'review': c.data['arms']['N'][0]['human_review']['approved'] = False
    elif case == 'hash': c.data['arms']['F'][0]['episode_sha256'] = '0' * 64
    elif case == 'template': c.tokenizer.chat_template = 'changed'
    reseal(c)
    with pytest.raises(ValueError):
        read(c)


def test_p1_never_opens_unrelated_parent_episodes_or_teacher_files(schema2, monkeypatch):
    c = as_branch(schema2, 'P1_U_mech')
    parent = json.loads((c.support / 'manifest.json').read_text())
    blocked = [(c.support / r['episode_dir']).parent.resolve() for r in parent['verified']
               if r['task_id'] not in c.data['seed_replay_task_ids']]
    original_open = Path.open
    def scoped_open(path, *a, **kw):
        assert not any(path.resolve().is_relative_to(directory) for directory in blocked), str(path)
        return original_open(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', scoped_open)
    checked = read(c)
    schedules = []
    for arm in ('N', 'F'):
        rows, slots = prepare_examples(checked, c.tokenizer, arm, packing='turn', max_length=32768)
        assert len(rows) == 24  # four repairs + K20 once, not four copies of K20
        batches = batch_schedule(c.data['D_repair_train'], slots, checked['seed_replay_task_ids'])
        schedules.append(batches)
        for batch in batches:
            seed = [s for s in batch['selections'] if s['bucket'] == 'seed']
            assert len(seed) == 1 and seed[0]['task_id'] in c.data['seed_replay_task_ids']
            assert seed[0]['task_id'] not in c.data['split']['failure_order']
    assert schedules[0] == schedules[1]


def test_p1_repeated_identical_pool_references_have_no_extra_exposure(schema2):
    c = as_branch(schema2, 'P1_U_mech')
    original = read(c)
    for entries in c.data['arms'].values():
        for entry in entries:
            entry['seed_demonstrations'] += deepcopy(entry['seed_demonstrations'])
    save(c)
    repeated = read(c)
    for arm in ('N', 'F'):
        assert prepare_examples(repeated, c.tokenizer, arm, packing='turn', max_length=32768) == \
               prepare_examples(original, c.tokenizer, arm, packing='turn', max_length=32768)


@pytest.mark.parametrize('case', ['partial', 'different_pool', 'other_support', 'attempt_file',
                                 'parent_metadata', 'probe_group', 'outside_U_mech', 'excluded_group'])
def test_p1_scoped_evidence_failures(schema2, case):
    c = as_branch(schema2, 'P1_U_mech')
    refs = c.data['arms']['N'][0]['seed_demonstrations']
    if case == 'partial':
        for entries in c.data['arms'].values():
            for e in entries: e['seed_demonstrations'].pop()
        c.data['seed_replay_task_ids'].pop()
    elif case == 'different_pool':
        for arm in ('N', 'F'): c.data['arms'][arm][1]['seed_demonstrations'].pop()
    elif case == 'other_support':
        for entries in c.data['arms'].values():
            for e in entries:
                for r in e['seed_demonstrations']: r['verified_manifest_sha256'] = '0' * 64
    elif case == 'attempt_file':
        p = c.support / refs[0]['episode_dir'] / 'episode.json'; p.write_text(p.read_text() + '\n')
    elif case == 'parent_metadata':
        p = c.support / 'identity.json'; write_json(p, {})
    else:
        splits = json.loads(c.repair_splits.read_text()); splits.pop('content_sha256')
        if case == 'probe_group': splits['sets']['D_probe']['group_ids'] = [refs[0]['receipt']['group_id']]
        elif case == 'excluded_group': splits['E']['group_ids'].append(c.data['arms']['N'][0]['group_id'])
        else: splits['sets']['U_mech']['task_ids'].remove(c.data['D_repair_train'][0])
        write_json(c.repair_splits, seal(splits))
        c.data['input_hashes'][str(c.repair_splits)] = file_hash(c.repair_splits)
    reseal(c)
    with pytest.raises(ValueError):
        read(c)


@pytest.mark.parametrize('case', ['revision', 'repo', 'metadata', 'files', 'weights_claim', 'adapter', 'trained', 'merged', 'theta0'])
def test_k0_base_identity_failures(schema2, case):
    c = as_branch(schema2, 'P0_K0')
    if case == 'revision': c.config['models']['student']['revision'] = 'other'
    elif case == 'repo': c.config['models']['student']['requested_repo'] = 'other'
    elif case == 'metadata': (c.snapshot / 'config.json').write_text('{"model_type":"gemma4","changed":true}')
    elif case == 'files': c.data['theta0']['files'] = {}
    elif case == 'weights_claim': c.data['theta0']['identity']['weight_verification'] = 'all weights verified'
    elif case == 'adapter': (c.snapshot / 'adapter_model.safetensors').write_bytes(b'not base')
    elif case == 'trained': (c.snapshot / 'training_state.json').write_text('{}')
    elif case == 'merged': (c.snapshot / 'consistency.json').write_text('{}')
    elif case == 'theta0': c.theta = c.path.parent / 'baseline/checkpoint'
    reseal(c)
    with pytest.raises(ValueError):
        read(c)


def test_uniform_task_then_episode_sampling_preserves_legacy_order():
    tasks, replay = ['r1', 'r2'], ['s1', 's2', 's3']
    slots = {(tid, 'repair'): [i] for i, tid in enumerate(tasks)}
    slots.update({('s1', 'seed'): [2], ('s2', 'seed'): [3, 4, 5], ('s3', 'seed'): [6, 7]})
    rng = random.Random(1)
    expected = []
    for _ in range(20):
        buckets = ['repair', 'repair', 'repair', 'seed']; rng.shuffle(buckets)
        row = []
        for bucket in buckets:
            tid = rng.choice(replay if bucket == 'seed' else tasks)
            row.append(slots[tid, bucket][rng.randrange(len(slots[tid, bucket]))])
        expected.append(row)
    assert [b['indices'] for b in batch_schedule(tasks, slots, replay)] == expected
    legacy_slots = {(tid, bucket): [i] for i, (tid, bucket) in enumerate(
        (t, b) for t in tasks for b in ('repair', 'seed'))}
    assert batch_schedule(tasks, legacy_slots) == batch_schedule(tasks, legacy_slots, tasks)


def test_k0_real_lora_initial_hash_and_update_zero_across_arms(schema2, tmp_path, monkeypatch):
    import torch
    from train import modeling
    from test_validate_shared_history_gpu import tiny_model
    from repair.train_arms import publish_checkpoint
    from harness.lora_adapters import inspect_adapter
    c = as_branch(schema2, 'P0_K0')
    attach = modeling.attach_lora
    calls = []
    def spy(base, **kw):
        calls.append(kw)
        return attach(base, **kw)
    def base(*a, **kw):
        result = tiny_model()
        torch.rand(len(calls) + 11)  # different base-loading consumption in N/F
        return result
    monkeypatch.setattr(modeling, 'load_student', base)
    monkeypatch.setattr(modeling, 'attach_lora', spy)
    hashes, states = [], []
    for arm in ('N', 'F'):
        inherited = base_recipe(c.data['theta0'], c.config, c.snapshot)
        model = load_fresh_adapter(c.snapshot, 'cpu', inherited, c.config['models']['student'])
        hashes.append(initial_adapter_hash(model))
        states.append({n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
        assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
        assert all(not p.requires_grad for n, p in model.named_parameters() if '.lora_' not in n)
        inherited['parameters'] = {n: p.numel() for n, p in states[-1].items()}
        inherited['trainable_modules'] = sorted({n.split('.lora_', 1)[0] for n in states[-1]})
        inherited['sft']['targets'] = sorted(model.peft_config['default'].target_modules)
        prov = {'config': {**c.config, 'sft': inherited['sft']}}
        output = tmp_path / arm; output.mkdir()
        counts = {'optimizer_updates': 0, 'processed_context_tokens': 0, 'supervised_target_tokens': 0}
        item = publish_checkpoint(model, output, prov, {'updates': 0, 'supervised_tokens_seen': 0}, counts, inherited)
        assert item['initial_adapter_sha256'] == hashes[-1]
        assert item['verification_only'] and item['probe_checkpoint_zero'] == 'base_model'
        assert item['sha256'] == checkpoint_files(output / 'update-00/checkpoint')
        inspect_adapter(arm, item['path'], c.config['models']['student'], c.snapshot)
    assert hashes[0] == hashes[1] and all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert calls[0] == calls[1] == dict(rank=64, alpha=128, dropout=.05, targets=modeling.LORA_TARGETS)
    assert inherited['recipe']['learning_rate'] == 5e-5


def test_k0_cli_needs_no_adapter(schema2, tmp_path, monkeypatch):
    from tools import repair_train_arm as cli
    c = as_branch(schema2, 'P0_K0')
    monkeypatch.setattr(cli, 'load_config', lambda p: c.config)
    monkeypatch.setattr(cli, 'load_tokenizer', lambda p: c.tokenizer)
    args = ['--snapshot', str(c.snapshot), '--collection', str(c.path), '--repair-splits', str(c.repair_splits),
            '--output', str(tmp_path / 'cli'), '--arm', 'N', '--dry-run', '--path-map', '='.join(c.path_maps[0])]
    assert cli.main(args) == 0
    result = json.loads((tmp_path / 'cli/manifest.json').read_text())
    assert result['branch'] == 'P0_K0' and result['theta0_kind'] == 'base_model'


def test_k0_production_orchestration_hashes_both_arms_before_any_update(schema2, tmp_path, monkeypatch):
    import contextlib
    import torch
    import torch.distributed as dist
    import transformers
    import baselines.common.fsdp_train as trainer
    import repair.train_arms as training
    from train import modeling
    from test_validate_shared_history_gpu import tiny_model
    c = as_branch(schema2, 'P0_K0')
    checked = read(c)
    monkeypatch.setenv('WORLD_SIZE', '2'); monkeypatch.setenv('RANK', '0'); monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: None)
    monkeypatch.setattr(transformers, 'set_seed', lambda *a: None)
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(torch.random, 'fork_rng', lambda **kw: contextlib.nullcontext())
    for name in ('init_process_group', 'new_group', 'broadcast', 'broadcast_object_list', 'destroy_process_group'):
        monkeypatch.setattr(dist, name, lambda *a, **kw: None)
    monkeypatch.setattr(dist, 'all_gather_object', lambda result, value, **kw: result.__setitem__(slice(None), [value] * 2))
    monkeypatch.setattr(modeling, 'load_student', lambda *a, **kw: tiny_model())
    monkeypatch.setattr(training, 'load_fresh_adapter', lambda snapshot, device, inherited, student:
                        load_fresh_adapter(snapshot, 'cpu', inherited, student))
    def forbidden(*a, **kw):
        raise AssertionError('K0 cannot restore an adapter or optimizer')
    monkeypatch.setattr(training, 'load_existing_adapter', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    calls = []
    def step(model, examples, output, **kw):
        manifest = json.loads((output / 'manifest.json').read_text())
        assert manifest['training_exposure'] == 0
        assert manifest['initial_adapter_sha256'] == initial_adapter_hash(model)
        zero = manifest['checkpoints'][0]
        assert zero['status'] == 'saved' and zero['update'] == 0
        assert zero['initial_adapter_sha256'] == manifest['initial_adapter_sha256']
        assert zero['trainer_state']['passes'] == zero['optimizer_updates'] == 0
        assert Path(zero['path']).is_dir() and zero['sha256'] == checkpoint_files(zero['path'])
        assert kw['optimizer_state'] is None and kw['learning_rate'] == 5e-5
        calls.append(kw)
        counts = training.accounting(examples, kw['fixed_batches'])
        for update in range(1, 21):
            with torch.no_grad():
                next(p for p in model.parameters() if p.requires_grad).add_(.001)
            state = {'updates': update, 'examples_seen': update * 4}
            if update in counts:
                state.update(supervised_tokens_seen=counts[update]['supervised_target_tokens'],
                             physical_policy_forward_tokens=counts[update]['processed_context_tokens'])
            kw['checkpoint_callback'](model, state)
        return state
    monkeypatch.setattr(trainer, 'train_loop', step)
    manifests = [run_arm(checked, c.config, tmp_path / arm, theta0=None, snapshot=c.snapshot,
                        tokenizer=c.tokenizer, arm=arm) for arm in ('N', 'F')]
    assert len(calls) == 2
    assert manifests[0]['initial_adapter_sha256'] == manifests[1]['initial_adapter_sha256']
    assert manifests[0]['schedule_sha256'] == manifests[1]['schedule_sha256']
    for result in manifests:
        assert result['status'] == 'complete' and result['training_exposure'] == 80
        assert result['theta0'] == c.data['theta0']
        assert [r['update'] for r in result['checkpoints']] == [0, 1, 5, 20]
        assert all(r['status'] == 'saved' for r in result['checkpoints'])
    # A divergent initialization fails before any output or optimizer step.
    monkeypatch.setattr(dist, 'all_gather_object', lambda result, value, **kw:
                        result.__setitem__(slice(None), [value, 'changed']))
    with pytest.raises(ValueError, match='initialization differs across ranks'):
        run_arm(checked, c.config, tmp_path / 'bad-rank', theta0=None, snapshot=c.snapshot,
                tokenizer=c.tokenizer, arm='N')
    assert len(calls) == 2 and not (tmp_path / 'bad-rank').exists()
