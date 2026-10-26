"""Branch boundaries use fake evidence, fake models and fake environments only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from repair.artifacts import checked_document, read_json, write_sealed
from repair.branches import (BRANCHES, ADAPTER_KEYS, base_checkpoint, branch_fields,
                             validate_base_serving, validate_coverage_branch)
from repair.dry_fakes import FakeEnvironment, FakeTokenizer, FakeTeacher, fake_review, N, F
from repair.export import export_branch, training_export
from repair.probes import run_probes, validate_serving
from sed.config import load_config
from sed.manifest import build_manifest, freeze_splits, resolve_task
from sed.storage import digest, write_json, file_hash
from tools.repair_coverage import coverage_plan
from test_repair_collect import run


@pytest.fixture
def branch_bank(tmp_path):
    root = tmp_path / 'data'
    for source, n in [('train', 42), ('valid_seen', 1), ('valid_unseen', 1)]:
        for i in range(n):
            directory = root / source / f'room-{i:02d}' / 'trial_0'
            write_json(directory / 'traj_data.json', {'task_type': 'pick_and_place_simple'})
            write_json(directory / 'game.tw-pddl', {'solvable': True})
    manifest = build_manifest(root)
    freeze = freeze_splits(manifest)
    tasks = [resolve_task(manifest, freeze, tid) for tid in sorted(freeze['seed_pool_candidates'])]
    config = load_config('configs/base_e4b.yaml')
    config['environment']['data_root'] = str(root)
    snapshot = tmp_path / 'snapshot'
    write_json(snapshot / 'config.json', {'model_type': 'gemma4'})
    write_json(snapshot / 'tokenizer_config.json', {'chat_template': 'fake'})
    base = base_checkpoint(snapshot, config['models']['student'])
    spec = {'model': config['models']['student']['requested_repo'],
            'revision': config['models']['student']['revision'], 'endpoint': 'http://127.0.0.1:1234/v1'}
    config['server_provenance'] = {'config': {'models': deepcopy(config['models']), 'server': {
        'batch_invariant': True, 'lora_adapters': [], 'merged_model': False, 'checkpoint_kind': 'base',
        'served_model_name': spec['model'], 'endpoint': spec['endpoint'], 'model_path': str(snapshot),
        'model_metadata_sha256': base['files'], 'training_provenance_sha256': None}}}
    seeds = {'K': 20, 'verified': [dict(t, episode_id=f'seed-{i}') for i, t in enumerate(tasks[:20])]}
    splits = {'sets': {'U_mech': {'task_ids': [t['task_id'] for t in tasks[20:40]]},
                       'D_probe': {'group_ids': [tasks[40]['group_id']]}}}
    adapter = {'identity': {k: 'fake-' + k for k in ADAPTER_KEYS}, 'files': {'fake': 'hash'}}
    return dict(config=config, manifest=manifest, freeze=freeze, tasks=tasks, spec=spec,
                splits=splits, seed_collection=seeds, snapshot=snapshot, base=base, adapter=adapter)


def coverage_evidence(bank, branch):
    config = deepcopy(bank['config'])
    spec = deepcopy(bank['spec'])
    if branch != 'P0_K0':
        config['lora_equivalence'] = {'identity': {'adapter': bank['adapter']['identity']}}
        spec['model'] = 'e4b-tsft-k20-e32-lr5e-5'
    tasks = bank['tasks'][20:40] if branch == 'P1_U_mech' else bank['tasks'][:20]
    plan = coverage_plan(tasks, config, spec, 'fake', extra_sampled=2)
    source = 'U_mech' if branch == 'P1_U_mech' else 'K20_seed_tasks'
    config['repair_coverage'] = {**deepcopy(plan), 'task_source': source, 'workers': 1}
    summary = {'student': spec, 'task_source': source, 'protocol': branch_fields(branch)['protocol'],
               'failure_list': [dict(t) for t in tasks[:6]]}
    return summary, {'config': config}, plan


def check(bank, branch, evidence=None):
    summary, provenance, plan = evidence or coverage_evidence(bank, branch)
    return validate_coverage_branch(branch, summary, provenance, plan, bank['seed_collection'],
        bank['splits'], bank['manifest'], bank['freeze'],
        bank['base'] if branch == 'P0_K0' else bank['adapter'], snapshot=bank['snapshot'])


@pytest.mark.parametrize('branch', BRANCHES)
def test_branch_accepts_its_frozen_coverage(branch_bank, branch):
    check(branch_bank, branch)


@pytest.mark.parametrize('branch', ['P1_U_mech', 'P0_K0'])
def test_unassigned_ineligible_manifest_rows_are_not_coverage_tasks(branch_bank, branch):
    # The real ALFWorld manifest includes ineligible trials without assignments.
    branch_bank['manifest']['tasks'].append({'task_id': 'train/ineligible/trial',
        'source_split': 'train', 'group_id': 'train/ineligible', 'eligible': False})
    check(branch_bank, branch)


@pytest.mark.parametrize('branch', BRANCHES)
@pytest.mark.parametrize('change', ['failure_task', 'checkpoint'])
def test_branch_rejects_wrong_failure_or_start(branch_bank, branch, change):
    evidence = coverage_evidence(branch_bank, branch)
    summary, provenance, _ = evidence
    if change == 'failure_task':
        summary['failure_list'][0]['task_id'] = branch_bank['tasks'][41]['task_id']
    elif branch == 'P0_K0':
        provenance['config']['server_provenance']['config']['server']['lora_adapters'] = [{'name': 'unwanted'}]
    else:
        provenance['config']['lora_equivalence']['identity']['adapter'] = {'revision': 'changed'}
    with pytest.raises(ValueError):
        check(branch_bank, branch, evidence)


@pytest.mark.parametrize('branch', ['P1_U_mech', 'P0_K0'])
@pytest.mark.parametrize('change', ['source', 'protocol', 'missing_task', 'extra_task', 'wrong_task',
                                  'seed_count', 'probe_group', 'dev_group', 'plan_provenance'])
def test_new_branches_fail_closed_on_complete_population(branch_bank, branch, change):
    summary, prov, plan = coverage_evidence(branch_bank, branch)
    first = summary['failure_list'][0]
    if change == 'source':
        summary['task_source'] = prov['config']['repair_coverage']['task_source'] = 'other'
    elif change == 'protocol':
        summary['protocol'] = 'P0' if branch == 'P1_U_mech' else 'P1'
    elif change == 'missing_task':
        plan['episodes'].pop()
    elif change == 'extra_task':
        plan['episodes'].append(plan['episodes'][0])
    elif change == 'wrong_task':
        plan['episodes'][-1]['task_id'] = branch_bank['tasks'][41]['task_id']
    elif change == 'seed_count':
        branch_bank['seed_collection']['K'] = 19
    elif change == 'probe_group':
        branch_bank['splits']['sets']['D_probe']['group_ids'].append(first['group_id'])
    elif change == 'dev_group':
        next(t for t in branch_bank['manifest']['tasks'] if t['source_split'] == 'valid_seen')['group_id'] = first['group_id']
    elif change == 'plan_provenance':
        prov['config']['repair_coverage']['episodes'] = []
    if change in ('missing_task', 'extra_task', 'wrong_task'):
        prov['config']['repair_coverage']['episodes'] = deepcopy(plan['episodes'])
    with pytest.raises(ValueError):
        check(branch_bank, branch, (summary, prov, plan))


def test_p1_excludes_seed_groups_even_for_different_trial_ids(branch_bank):
    evidence = coverage_evidence(branch_bank, 'P1_U_mech')
    tid = branch_bank['seed_collection']['verified'][0]['task_id']
    next(t for t in branch_bank['manifest']['tasks'] if t['task_id'] == tid)['group_id'] = branch_bank['tasks'][20]['group_id']
    with pytest.raises(ValueError, match='overlaps'):
        check(branch_bank, 'P1_U_mech', evidence)


@pytest.mark.parametrize('change', ['adapter', 'missing_adapter_field', 'merged', 'fine_tuned', 'name',
                                  'endpoint', 'revision', 'metadata', 'local_metadata', 'batch'])
def test_base_probe_serving_rejections(branch_bank, change):
    cfg, spec = deepcopy(branch_bank['config']), deepcopy(branch_bank['spec'])
    server = cfg['server_provenance']['config']['server']
    if change == 'adapter': server['lora_adapters'] = [{'name': 'sft'}]
    elif change == 'missing_adapter_field': del server['lora_adapters']
    elif change == 'merged': server['merged_model'] = True
    elif change == 'fine_tuned': server['checkpoint_kind'] = 'full_parameter'
    elif change == 'name': spec['model'] = 'sft'
    elif change == 'endpoint': spec['endpoint'] += '-wrong'
    elif change == 'revision': cfg['server_provenance']['config']['models']['student']['revision'] = 'wrong'
    elif change == 'metadata': server['model_metadata_sha256'] = {}
    elif change == 'local_metadata': write_json(branch_bank['snapshot'] / 'tokenizer_config.json', {'changed': True})
    elif change == 'batch': server['batch_invariant'] = False
    with pytest.raises(ValueError):
        validate_serving(cfg, spec, base_model=True, expected=branch_bank['base'])


def test_base_probe_zero_and_updated_checkpoint_requirements(branch_bank, tmp_path):
    collection = {**branch_fields('P0_K0'), 'checkpoint': branch_bank['base'],
                  'tokenizer': {}, 'D_repair_probe': [], 'cases': []}
    calls = []
    result = run_probes(collection, {'content_sha256': 'fake'}, branch_bank['config'], tmp_path / 'probe',
        arm='shared', checkpoint=0, spec=branch_bank['spec'], env_factory=FakeEnvironment,
        quick_runner=lambda *a, **kw: calls.append(a) or {'tasks': []})
    assert calls and result['theta0_kind'] == 'base_model' and result['adapter'] is None
    for arm, cp in [('N', 1), ('N', 0)]:
        with pytest.raises(ValueError):
            run_probes(collection, {}, branch_bank['config'], tmp_path / f'bad-{cp}', arm=arm, checkpoint=cp,
                spec=branch_bank['spec'], env_factory=FakeEnvironment)


@pytest.mark.parametrize('branch', BRANCHES)
def test_export_branch_replay_accounting_and_protocol(tmp_path, branch):
    data = run(tmp_path / branch, branch=branch, reviewer=fake_review)
    export = checked_document(tmp_path / branch / 'e2_training_collection.json')
    assert export_branch(export) == branch_fields(branch)
    assert export['schema_version'] == 2
    original_support = branch == 'P1_U_mech'
    assert export['seed_replay_tasks_are_repair_tasks'] is not original_support
    assert export['original_teacher_far_policy'] == 'excluded_without_replay_pair_check'
    assert data['supervision_accounting']['theta0_seed_training_trajectories'] == (0 if branch == 'P0_K0' else 6)
    assert data['supervision_accounting']['new_task_unique_instances'] == (6 if original_support else 0)
    for n, f in zip(export['arms']['N'], export['arms']['F']):
        assert n['seed_demonstrations'] == f['seed_demonstrations']
        ids = {r['task_id'] for r in n['seed_demonstrations']}
        if original_support:
            assert len(ids) == 6 and not ids & set(data['split']['failure_order'])
        else:
            assert ids == {n['task_id']}
        assert n['design_bucket'] == ('G' if original_support else 'R')
    for path in (tmp_path / branch).glob('cases/*/budget.jsonl'):
        assert all(json.loads(line)['protocol'] == export['protocol'] for line in path.read_text().splitlines())
    legacy = deepcopy(export); legacy['schema_version'] = 1
    if branch == 'P0_K20':
        for key in branch_fields(branch): legacy.pop(key)
        assert export_branch(legacy)['branch'] == 'P0_K20'
    else:
        with pytest.raises(ValueError): export_branch(legacy)
    bad = deepcopy(export); bad.pop('theta0_kind')
    with pytest.raises(ValueError): export_branch(bad)
    bad = deepcopy(data)
    tid = data['D_repair_train'][0]
    if original_support:
        ref = deepcopy(next(iter(data['seed_demonstrations'].values())))
        bad['seed_demonstrations'][tid] = ref
    else:
        bad['seed_demonstrations'][tid][0]['task_id'] = data['D_repair_probe'][0]
    with pytest.raises(ValueError): training_export(bad, collection_sha256='fake')


def test_k0_original_teacher_route_stays_excluded(tmp_path):
    from test_repair_collect import inputs
    data = run(tmp_path / 'k0', branch='P0_K0', reviewer=fake_review,
               seed_actions={t['task_id']: [F] for t in inputs()[1]})
    assert not data['pairs']
    assert any('far_duplicates_original_teacher_seed' in e['reasons'] for e in data['exclusions'])


def test_cli_requires_branch_even_for_dry_run_and_finalize(tmp_path):
    from tools.repair_collect import main
    with pytest.raises(SystemExit) as error:
        main(['--dry-run', '--output', str(tmp_path / 'dry')])
    assert error.value.code == 2 and not (tmp_path / 'dry').exists()
    with pytest.raises(SystemExit):
        main(['--finalize', 'missing.json', '--output', str(tmp_path / 'final')])


@pytest.mark.parametrize('training_section, support_location', [
    ('sft', 'local'), ('sft', 'remote'), ('sft', 'relative'),
    ('baseline_training', 'remote'), ('both', 'remote')])
def test_p1_support_never_opens_parent_non_support_evidence(
        branch_bank, tmp_path, monkeypatch, training_section, support_location):
    from teachers.collect_seeds import collect
    from sed.seed_selection import select_full_train
    from sed.seed_evidence import materialize_full_train_views
    from repair.support import load_original_support
    from test_full_train import factories
    manifest, freeze, cfg = branch_bank['manifest'], branch_bank['freeze'], deepcopy(branch_bank['config'])
    cfg['environment']['max_episode_steps'] = 1
    selection = select_full_train(manifest, freeze)
    bank = (cfg['environment']['data_root'], manifest, freeze, selection, cfg)
    env, teacher = factories(bank, tmp_path / 'calls.jsonl')
    root = tmp_path / 'full'
    collect(selection, manifest, freeze, cfg, root, env_factory=env, teacher_factory=teacher)
    materialize_full_train_views(root, manifest, freeze, [20])
    view = root / 'subsets/k20'
    source, selected = read_json(root / 'manifest.json'), read_json(view / 'manifest.json')
    support_ids = {r['task_id'] for r in selected['verified']}
    other = [r for r in source['verified'] if r['task_id'] not in support_ids]
    splits = {'sets': {'D_probe': {'group_ids': [other[-1]['group_id']]},
                       'U_mech': {'task_ids': [r['task_id'] for r in other[:20]]}}}
    adapter = tmp_path / 'sft/checkpoint'
    recorded_path = {'local': str(view), 'remote': '/original-cluster/verified/subsets/k20',
                     'relative': 'verified/subsets/k20'}[support_location]
    input_path = str(Path(recorded_path) / 'manifest.json')
    if support_location == 'relative':
        input_path = '/original-cluster/' + input_path
    config = {'sft': {'packing': 'episode'}}
    for section in (('sft', 'baseline_training') if training_section == 'both' else (training_section,)):
        config.setdefault(section, {})['verified'] = recorded_path
    provenance = {'config': config, 'inputs': {input_path: file_hash(view / 'manifest.json')}}
    write_json(adapter.parent / 'provenance.json', provenance)
    forbidden = [(root / r['episode_dir']).parent.resolve() for r in other]
    opened = []
    real_open = Path.open
    def guard(path, *a, **kw):
        assert not any(path.resolve().is_relative_to(p) for p in forbidden), 'Opened non-support teacher evidence'
        opened.append(path.resolve())
        return real_open(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', guard)
    result, evidence_root, episodes = load_original_support(view, manifest, freeze, adapter=adapter, splits=splits)
    assert result == selected and evidence_root == root and len(episodes) == 20
    assert {e['task_id'] for e in episodes} == support_ids
    from repair.support import original_support_examples
    from train.dataset import episode_examples
    examples = original_support_examples(view, FakeTokenizer(), manifest, freeze, adapter=adapter, splits=splits)
    assert examples == [row for episode in episodes for row in episode_examples(episode, FakeTokenizer(), packing='turn')]
    assert not any(p.name == 'episode.json' and any(p.is_relative_to(q) for q in forbidden) for p in opened)
    # Identical JSON with different bytes must still fail the training hash binding.
    manifest_bytes = (view / 'manifest.json').read_bytes()
    (view / 'manifest.json').write_bytes(manifest_bytes + b'\n')
    with pytest.raises(ValueError, match='original support'):
        load_original_support(view, manifest, freeze, adapter=adapter, splits=splits)
    (view / 'manifest.json').write_bytes(manifest_bytes)
    write_json(adapter.parent / 'provenance.json', {**provenance, 'inputs': {input_path: 'wrong'}})
    with pytest.raises(ValueError, match='original support'):
        load_original_support(view, manifest, freeze, adapter=adapter, splits=splits)
    # A matching hash stored for another path is not evidence for the recorded support.
    write_json(adapter.parent / 'provenance.json', {**provenance,
        'inputs': {'/unrelated/manifest.json': file_hash(view / 'manifest.json')}})
    with pytest.raises(ValueError, match='original support'):
        load_original_support(view, manifest, freeze, adapter=adapter, splits=splits)


@pytest.mark.parametrize('config, error', [
    ({}, 'needs sft.verified or baseline_training.verified'),
    ({'sft': {}, 'baseline_training': {}}, 'needs sft.verified or baseline_training.verified'),
    ({'sft': {'verified': '/original/support'}, 'baseline_training': {'verified': '/other/support'}},
     'conflicting sft.verified and baseline_training.verified'),
    ({'sft': {'verified': None}, 'baseline_training': {'verified': '/original/support'}}, 'nonempty strings'),
])
def test_p1_support_rejects_missing_or_conflicting_paths(tmp_path, config, error):
    from repair.support import load_original_support
    view, adapter = tmp_path / 'verified', tmp_path / 'training/checkpoint'
    for name in ('manifest.json', 'selection.json', 'identity.json'):
        write_json(view / name, {})
    write_json(adapter.parent / 'provenance.json', {'config': config,
        'inputs': {'/original/support/manifest.json': file_hash(view / 'manifest.json')}})
    with pytest.raises(ValueError, match=error):
        load_original_support(view, {}, {}, adapter=adapter, splits={})


@pytest.mark.parametrize('branch', ['P1_U_mech', 'P0_K0'])
def test_real_mode_collector_cli_wires_branch_with_fake_clients(branch_bank, tmp_path, monkeypatch, branch):
    import tools.repair_collect as cli
    import repair.support as support
    import train.dataset as dataset
    import harness.lora_adapters as adapters
    import harness.alfworld_env as environment
    from repair.dry_fakes import write_seed_demonstrations, O
    from repair.execute import execute_actions
    bank = branch_bank
    summary, prov, plan = coverage_evidence(bank, branch)
    for name, value in [('summary', summary), ('provenance', prov), ('plan', plan), ('failures', {})]:
        write_json(tmp_path / 'coverage' / (name + '.json'), value)
    cfg = deepcopy(bank['config'])
    for name in ('manifest', 'split_freeze'):
        cfg['paths'][name] = str(tmp_path / (name + '.json'))
        write_json(cfg['paths'][name], bank['manifest'] if name == 'manifest' else bank['freeze'])
    split_path = tmp_path / 'splits.json'; write_json(split_path, bank['splits'])
    refs = write_seed_demonstrations(tmp_path / 'support', bank['tasks'][:20])
    collection = {'K': 20, 'verified': [r[0]['receipt'] for r in refs.values()]}
    episodes = [{'task_id': tid, 'episode_id': rows[0]['episode_id'],
                 'steps': [{'action_parsed': a} for a in N]} for tid, rows in refs.items()]
    monkeypatch.setattr(cli, 'load_config', lambda *a: cfg)
    monkeypatch.setattr(cli, 'checked_repair_splits', lambda *a: bank['splits'])
    monkeypatch.setattr(cli, 'load_verified_collection', lambda *a: (collection, {}, tmp_path / 'support'))
    monkeypatch.setattr(cli, 'checked_episodes', lambda *a: iter(episodes))
    monkeypatch.setattr(support, 'load_original_support', lambda *a, **kw: (collection, tmp_path / 'support', episodes))
    monkeypatch.setattr(adapters, 'inspect_adapter', lambda *a: bank['adapter']['identity'])
    monkeypatch.setattr(cli, 'checkpoint_files', lambda *a: bank['adapter']['files'])
    monkeypatch.setattr(dataset, 'load_tokenizer', lambda *a: FakeTokenizer())
    monkeypatch.setattr(environment, 'Environment', lambda *a, **kw: FakeEnvironment())
    calls = []
    def teachers(*a, **kw):
        calls.append(kw)
        return FakeTeacher()
    monkeypatch.setattr(cli, 'make_teachers', teachers)
    tasks = [deepcopy(t) for t in summary['failure_list']]
    def failures(*a, **kw):
        assert kw['branch'] == branch and kw['canary'] is True
        (Path(a[-1]) / 'preflight_budget.jsonl').write_text('')
        return [execute_actions(FakeEnvironment, tasks[0]['task_id'], O, 40)], tasks
    monkeypatch.setattr(cli, 'load_failures', failures)
    args = ['--branch', branch, '--failures', str(tmp_path / 'coverage/failures.json'),
            '--snapshot', str(bank['snapshot']), '--verified', str(tmp_path / 'support'),
            '--repair-splits', str(split_path), '--canary', '--output', str(tmp_path / 'out')]
    if branch == 'P1_U_mech': args.extend(['--theta0', str(tmp_path / 'adapter')])
    assert cli.main(args) == 0
    result = checked_document(tmp_path / 'out/collection.json')
    assert result['branch'] == branch and result['cases'][0]['protocol'] == branch_fields(branch)['protocol']
    assert len(calls) == 1 and calls[0]['single_diagnosis'] is True
    assert len(result['cases'][0]['parent_seed_ids']) == (20 if branch == 'P1_U_mech' else 1)
    assert result['checkpoint'] == (bank['base'] if branch == 'P0_K0' else bank['adapter'])


def test_probe_cli_base_zero_requires_no_lora_report(branch_bank, tmp_path, monkeypatch):
    import tools.repair_probes as cli
    bank = branch_bank
    source = tmp_path / 'collection.json'
    write_sealed(source, {**branch_fields('P0_K0'), 'checkpoint': bank['base'],
        'dry_run': False, 'canary': False, 'D_repair_probe': [bank['tasks'][0]['task_id']]})
    server = tmp_path / 'server.json'; write_json(server, bank['config']['server_provenance'])
    monkeypatch.setattr(cli, 'load_config', lambda *a: deepcopy(bank['config']))
    monkeypatch.setattr(cli, 'load_frozen', lambda *a: (bank['manifest'], bank['freeze']))
    monkeypatch.setattr(cli, 'checked_repair_splits', lambda *a: bank['splits'])
    seen = []
    monkeypatch.setattr(cli, 'run_probes', lambda *a, **kw: seen.append(kw))
    args = ['--collection', str(source), '--arm', 'shared', '--checkpoint', '0',
            '--endpoint', bank['spec']['endpoint'], '--model-name', bank['spec']['model'],
            '--server-provenance', str(server), '--output', str(tmp_path / 'probes')]
    assert cli.main(args) == 0 and seen[0]['checkpoint'] == 0
    with pytest.raises(SystemExit): cli.main(args + ['--lora-check', 'must-not-open.json'])
