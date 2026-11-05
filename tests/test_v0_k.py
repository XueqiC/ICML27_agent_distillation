"""K ladder round regression: fake runtime, no GPU, network or teacher client."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os
import shutil
import subprocess
import sys

import pytest

from repair import synth_tasks as factory, v0_synth
from repair.v0_common import (BINS, QUOTAS, THETA0_CELLS, SyntheticManifest, artifacts,
                             completed, finish, phase, read, round_identity)
from repair.v0_data import check_replay, load_dataset
from repair.v0_expand import expansion_plan
from sed.storage import BlobStore, digest, file_hash, write_json
from tools import v0_round as cli
from test_synth_tasks import small_game
from test_v0_pipeline import arguments, fake_adapter, replay_bundle
from test_v0_round import Pool, manifest_at, seed_ids


@pytest.mark.parametrize('k', [10, 20, 30])
def test_synth_scales_seed_quotas_and_probes(tmp_path, monkeypatch, k):
    args = arguments(tmp_path / 'round', k)
    root = cli.round_root(args)
    evidence = tmp_path / 'receipt'
    evidence.write_text('fake executed seed receipt')
    provenance = {key: str(evidence) for key in ('game_path', 'episode_path', 'events_path')}
    provenance['environment_seed'] = 0
    seeds = [factory.Seed(tid, small_game(), ['finish'], provenance) for tid in seed_ids(k)]
    monkeypatch.setattr(factory, 'load_seeds', lambda *a: (seeds, read(args.seeds)))
    monkeypatch.setattr(factory, 'freeze_heldout', lambda *a: dict(groups=[], sources=[]))

    class FakeFactory(Pool):
        def __init__(self, parent, seeds, directory, *args):
            super().__init__()
            self.parent, self.records = parent, []

        def next_valid(self, *args):
            row = super().next_valid(*args)
            from sed.storage import digest
            ident = digest([self.parent.task_id, row['id']])[:24]
            row.update(id=ident, task_id='train/synth/' + ident, item_fingerprint=ident,
                       parent_seed_task_id=self.parent.task_id)
            self.records.append(deepcopy(row))
            return row

    monkeypatch.setattr(v0_synth, 'SeedFactory', FakeFactory)
    original = v0_synth.subprocess.run

    def fake_workers(command, **kwargs):
        if 'repair.v0_synth' not in command:
            return original(command, **kwargs)
        request = read(command[-1])
        assert request['workers'] == 4 and kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        for tid in request['seed_task_ids']:
            v0_synth.seed_worker((request, tid))

    monkeypatch.setattr(v0_synth.subprocess, 'run', fake_workers)
    result = v0_synth.synthesize(seeds_path=args.seeds, output=root, data_root=tmp_path / 'games',
                                main_checkout=tmp_path / 'source', workers=4)
    assert result['K'] == result['unique_seed_instances'] == result['seed_task_groups'] == k
    assert result['training_items'] == 13 * k and result['heldout_probes'] == 4 * k
    for b in BINS:
        assert result['by_bin'][b]['training'] == QUOTAS[b] * k
        assert result['by_bin'][b]['probes'] == k
    manifest = SyntheticManifest(root / 'synth/manifest.json')
    assert manifest.k == k and read(root / 'round.json') == round_identity(k)
    assert read(root / 'synth/plan.json')['identity']['K'] == k
    assert completed(root / 'synth')


@pytest.mark.parametrize('k,count', [(5, 5), (40, 40), (10, 9), (20, 30), (30, 20), (True, 1)])
def test_invalid_seed_counts_fail_before_round_creation(tmp_path, k, count):
    args = arguments(tmp_path / 'round')
    write_json(args.seeds, dict(K=k, verified=[{}] * count))
    with pytest.raises(ValueError, match='K'):
        cli.round_root(args)
    assert not Path(args.output).exists()


@pytest.mark.parametrize('change', [None, 'groups', 'source_hash', 'source_bytes', 'source_added'])
def test_reusing_freeze_preserves_exclusions_and_checks_all_source_contents(tmp_path, monkeypatch, change):
    old_path, new_path = tmp_path / 'old/view.json', tmp_path / 'new/view.json'
    write_json(old_path, {'task_ids': ['valid_seen/private/trial']})
    write_json(new_path, read(old_path))
    source = dict(path=str(old_path), sha256=file_hash(old_path), groups=1, identity_only=True)
    frozen = dict(groups=['heldout'], sha256=digest(['heldout']), policy='identity only', sources=[source])
    current = dict(frozen, sources=[dict(source, path=str(new_path))])
    path = tmp_path / 'freeze.json'
    write_json(path, frozen)
    if change == 'groups':
        current['groups'] = ['heldout', 'new-heldout']
    elif change == 'source_hash':
        current['sources'][0]['sha256'] = 'a' * 64
    elif change == 'source_bytes':
        write_json(old_path, {'tampered': True})
    elif change == 'source_added':
        current['sources'].append(dict(current['sources'][0], path=str(tmp_path / 'additional.json')))
    monkeypatch.setattr(factory, 'freeze_heldout', lambda *a: current)
    if change:
        with pytest.raises(ValueError):
            v0_synth.synthesis_freeze('unused', 'unused', path)
    else:
        assert v0_synth.synthesis_freeze('unused', 'unused', path) == frozen
        assert v0_synth.synthesis_freeze('unused', 'unused') == current


@pytest.mark.parametrize('k', [10, 20, 30])
def test_synthetic_manifest_rejects_wrong_k_and_round(tmp_path, k):
    args = arguments(tmp_path / 'round', k)
    root = cli.round_root(args)
    manifest = manifest_at(root / 'synth', k=k)
    value = read(manifest.path)
    value['K'] = 30 if k == 10 else 10
    write_json(manifest.path, value)
    with pytest.raises(ValueError, match='K'):
        cli.manifest_for(args, root)
    manifest_at(root / 'synth', k=value['K'])
    with pytest.raises(ValueError, match='K'):
        cli.manifest_for(args, root)


@pytest.mark.parametrize('k', [10, 20, 30])
@pytest.mark.parametrize('mismatch', ['model', 'adapter', 'adapter_name', 'training_state', None])
def test_serving_requires_matching_k_model_and_adapter(tmp_path, monkeypatch, k, mismatch):
    other = 30 if k == 10 else 10
    args = arguments(tmp_path / 'round', k)
    args.K = k
    args.model_name = THETA0_CELLS[other if mismatch == 'model' else k]
    path = tmp_path / THETA0_CELLS[other if mismatch == 'adapter' else k] / 'checkpoint'
    path.mkdir(parents=True)
    write_json(path / 'training_state.json', dict(N_examples=other if mismatch == 'training_state' else k,
                                                 final=True, seed=0, epochs=64 if k == 30 else 32))
    weights = path / 'adapter_model.safetensors'
    weights.write_bytes(b'fake')
    adapter = dict(fake_adapter(other if mismatch == 'adapter_name' else k), path=str(path),
                   adapter_weights_sha256=file_hash(weights))
    calls = []

    def fake_spec(args, config):
        calls.append(True)
        return {}, adapter, []

    monkeypatch.setattr('repair.adapter_eval.serving_spec', fake_spec)
    if mismatch:
        with pytest.raises(ValueError, match='K'):
            cli.serving(args, cli.config_for(args))
        if mismatch == 'model':
            assert not calls
    else:
        _, actual, inputs = cli.serving(args, cli.config_for(args))
        assert actual == adapter and inputs == [weights]


@pytest.mark.parametrize('k', [10, 20, 30])
def test_replay_count_is_seed_count_not_synthetic_count(tmp_path, k):
    bundle = replay_bundle(tmp_path / 'replay', seed_ids(k))
    assert len(check_replay(bundle, seed_ids(k))[1]) == k
    with pytest.raises(ValueError, match='seed receipts'):
        check_replay(bundle, seed_ids(30 if k == 10 else 10))


@pytest.mark.parametrize('k', [10, 30])
@pytest.mark.parametrize('evaluation_overlap', [False, True])
def test_build_automatically_replays_exact_seed_receipts(tmp_path, k, evaluation_overlap):
    from repair.artifacts import seal
    from sed.seed_evidence import artifact_hashes
    from test_supervision import as_episode, record

    root = tmp_path / 'round'
    args = arguments(root, k)
    cli.round_root(args)
    manifest = manifest_at(root / 'synth', k=k)
    collection = tmp_path / 'collection'
    receipts, sources = [], {}
    for i, tid in enumerate(seed_ids(k)):
        episode = as_episode(record(['look'], ['On the cabinet 1, you see an apple 1.'],
                                    task_id=tid, success=True))
        episode.update(seed=0, game_sha256='c' * 64)
        directory = collection / f'attempts/{i}/episode'
        blob = BlobStore(directory / 'blobs').put(dict(fact_source='real_environment',
                                                      executable=True, audit={'won': True}))
        episode['steps'][-1]['verification_ref'] = blob
        write_json(directory / 'episode.json', episode)
        (directory / 'events.jsonl').write_text('{}\n')
        hashes = artifact_hashes(directory)
        receipts.append(dict(task_id=tid, won=True, success=True, episode_dir=str(directory.relative_to(collection)),
                             episode_hashes=hashes))
        sources[tid] = dict(episode_sha256=hashes['episode.json'], events_sha256=hashes['events.jsonl'],
                            game_sha256=episode['game_sha256'])
    source = dict(status='complete', verified=receipts)
    write_json(collection / 'manifest.json', source)
    subset = collection / f'subsets/k{k}'
    selection = dict(K=k, task_order=seed_ids(k))
    identity = dict(selection=selection)
    seeds = dict(K=k, status='complete', verified=receipts, selection_sha256=digest(selection),
        identity_sha256=digest(identity), source_collection=dict(path='../..', manifest_sha256=digest(source)))
    for name, value in (('manifest', seeds), ('selection', selection), ('identity', identity)):
        write_json(subset / (name + '.json'), value)
    write_json(manifest.path, dict(manifest.value, seed_sources=sources,
                                  source_seed_manifest_sha256=file_hash(subset / 'manifest.json')))
    manifest = SyntheticManifest(manifest.path)
    write_json(root / 'synth/plan.json', dict(identity=dict(inputs={str(subset / 'manifest.json'):
                                                               file_hash(subset / 'manifest.json')})))
    args.splits = str(tmp_path / 'splits.json')
    write_json(args.splits, seal(dict(sets={'dev140': dict(group_ids=[seed_ids(k)[0].rsplit('/', 1)[0]]
                                                          if evaluation_overlap else [])})))
    checkpoint = tmp_path / THETA0_CELLS[k] / 'checkpoint'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'fake model')
    write_json(checkpoint / 'adapter_config.json', dict(fake=True))
    hashes = {name: file_hash(checkpoint / name) for name in ('adapter_model.safetensors', 'adapter_config.json')}
    write_json(checkpoint / 'training_state.json', dict(N_examples=k, final=True, seed=0,
        epochs=64 if k == 30 else 32, artifact_sha256=hashes))
    adapter = dict(fake_adapter(k), path=str(checkpoint),
        adapter_weights_sha256=hashes['adapter_model.safetensors'], adapter_config_sha256=hashes['adapter_config.json'])
    labels = root / 'label'
    phase(labels, dict(command='fake-label', inputs={}, theta0=adapter, complete_task_coverage=True))
    write_json(labels / 'rows.json', [])
    finish(labels)
    if evaluation_overlap:
        with pytest.raises(ValueError, match='overlaps evaluation'):
            cli.build_dataset(args, root, manifest, labels)
        assert not (root / 'dataset/manifest.json').exists()
        return
    result = cli.build_dataset(args, root, manifest, labels)
    assert result['K'] == result['counts']['D']['episodes'] == k
    _, rows = check_replay(root / 'dataset/replay', manifest.seeds, adapter)
    assert {r['task_id'] for r in rows} == manifest.seeds
    assert all('SECRET' not in str(r['messages']) for r in rows)
    replay = read(root / 'dataset/replay/manifest.json')
    assert replay['K'] == k and replay['new_teacher_calls'] == replay['actual_training_exposures'] == 0
    assert str(subset / 'manifest.json') in replay['inputs']
    assert str(checkpoint / 'training_state.json') in replay['inputs']


@pytest.mark.parametrize('k', [10, 30])
@pytest.mark.parametrize('mismatch', ['round', 'adapter', 'seed_manifest', None])
def test_rai_preflight_precedes_any_server(tmp_path, k, mismatch):
    project = tmp_path / 'project'
    (project / 'tools').mkdir(parents=True)
    (project / 'scripts').mkdir()
    launcher = project / 'tools/v0_round_rai.sh'
    shutil.copyfile('tools/v0_round_rai.sh', launcher)
    root = tmp_path / 'round'
    args = arguments(root, k)
    cli.round_root(args)
    manifest = manifest_at(root / 'synth', k=k)
    subset = tmp_path / f'k{k}'
    write_json(subset / 'manifest.json', read(args.seeds))
    write_json(manifest.path, dict(manifest.value, source_seed_manifest_sha256=file_hash(subset / 'manifest.json')))
    if mismatch == 'round':
        write_json(root / 'round.json', round_identity(20))
    if mismatch == 'seed_manifest':
        write_json(subset / 'manifest.json', dict(read(args.seeds), tampered=True))
    marker = tmp_path / 'server-would-start'
    checkpoint = tmp_path / THETA0_CELLS[20 if mismatch == 'adapter' else k] / 'checkpoint'
    write_json(checkpoint / 'training_state.json', dict(N_examples=k, final=True, seed=0,
                                                      epochs=64 if k == 30 else 32))
    (project / 'scripts/repair_stage_env.sh').write_text('ENV_PYTHON="' + sys.executable + '"\n')
    (project / 'scripts/repair_serve_stage.sh').write_text('touch "' + str(marker) + '"\n')
    env = dict(PATH=os.defpath, PYTHONPATH=str(Path.cwd() / 'src') + ':' + str(Path.cwd()),
        V0_STAGE='attempt', V0_ROUND_OUT=str(root), VERIFIED=str(subset), MODEL_NAME=THETA0_CELLS[k],
        ADAPTER=str(checkpoint))
    result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True)
    assert (result.returncode == 0) == (mismatch is None)
    assert marker.exists() == (mismatch is None)


@pytest.mark.parametrize('k', [10, 20, 30])
def test_expansion_counts_distinct_failed_items_and_original_seed_parents(tmp_path, k):
    from test_v0_round import task_row
    tasks, attempts, rows = [], [], []
    for i, parent in enumerate(seed_ids(k)):
        task = dict(task_row(f'train/synth/item-{i}'), parent_seed_task_id=parent)
        tasks.append(task)
        attempts.extend([dict(task_id=task['task_id'], heldout_probe=False, success=False)] * 3)
        attempts.append(dict(task_id=task['task_id'], heldout_probe=False, success=True))
        eid = f'repaired-{i}'
        path = tmp_path / (eid + '.json')
        write_json(path, dict(episode_id=eid, verified=True, success=True,
                             rounds=[dict(review=dict(breaking_points=[dict(category='missing_precondition')]))]))
        rows.append(dict(task_id=task['task_id'], v0=dict(bucket='R', episode_id=eid),
                         evidence=dict(episode_ref=str(path))))
    manifest = manifest_at(tmp_path / 'synth', tasks, k)
    attempts.append(dict(task_id='train/synth/unused-probe', heldout_probe=True, success=False))
    plan = expansion_plan(tmp_path / 'unused-dataset', rows, attempts, manifest, variant='D-param')
    assert plan['requested_items'] == len(plan['units']) == k
    assert plan['allocation'] == {'missing_precondition': k}
    assert all(unit['preferred_seed'] in manifest.seeds for unit in plan['units'])


REAL_K20 = Path('/home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1')
REAL_SHA = '4df53e36906111105035aa0bfdcecf153de766b65efc7cae60fb00b1763cfd00'


@pytest.mark.skipif(not (REAL_K20 / 'synth/manifest.json').is_file(), reason='real K20 receipts unavailable')
def test_real_k20_synth_inputs_and_cached_witnesses_are_byte_identical(tmp_path, monkeypatch):
    """Regenerate candidate order/allocation/publication using real sealed evidence.

    No cached result.json is trusted: seed_worker reruns the candidate factory
    and allocation. Copying receipts preserves recorded timeout/negative outcomes
    without repeating timing-sensitive planner executions. Originals are read-only.
    """
    source = REAL_K20 / 'synth'
    assert file_hash(source / 'manifest.json') == REAL_SHA
    assert completed(source)
    before = artifacts(source)
    request = read(source / 'request.json')
    original = v0_synth.subprocess.run

    def cached_workers(command, **kwargs):
        if 'repair.v0_synth' not in command:
            return original(command, **kwargs)
        new = read(command[-1])
        shutil.copytree(source / 'parents', Path(new['out']) / 'parents')
        for tid in new['seed_task_ids']:
            v0_synth.seed_worker((new, tid))

    def no_new_validation(*a, **kw):
        pytest.fail('K20 candidate sequence changed: uncached validation requested')

    monkeypatch.setattr(v0_synth.subprocess, 'run', cached_workers)
    monkeypatch.setattr('repair.v0_validation.isolated_validation', no_new_validation)
    root = tmp_path / 'k20'
    args = SimpleNamespace(output=root, seed=0, seeds=request['seeds'])
    cli.round_root(args)
    result = v0_synth.synthesize(seeds_path=request['seeds'], data_root=request['data_root'], output=root,
        main_checkout='/home/xueqi/hq/projects/ICML27_agent_distillation', workers=4,
        split_freeze=source / 'split_freeze.json')
    assert result == read(source / 'stats.json')
    assert result['synthetic_manifest_sha256'] == REAL_SHA
    for name in ('manifest.json', 'allocation.json', 'attempts.json', 'split_freeze.json'):
        assert (root / 'synth' / name).read_bytes() == (source / name).read_bytes()
    old_plan = read(source / 'plan.json')['identity']
    new_plan = read(root / 'synth/plan.json')['identity']
    # Code provenance necessarily changes; scientific config and inputs do not.
    assert {k: v for k, v in new_plan.items() if k not in ('code_sha256', 'dependencies')} == {
        k: v for k, v in old_plan.items() if k not in ('code_sha256', 'dependencies')}
    assert (root / 'round.json').read_bytes() == (REAL_K20 / 'round.json').read_bytes()
    assert artifacts(source) == before


@pytest.mark.skipif(not (REAL_K20 / 'dataset/manifest.json').is_file(), reason='real K20 dataset unavailable')
@pytest.mark.parametrize('variant', ['D-data', 'D-param'])
def test_real_k20_expansion_allocations_unchanged(variant):
    root = REAL_K20 / ('expand-' + variant)
    identity = read(root / 'plan.json')['identity']
    dataset = REAL_K20 / 'dataset'
    _, rows = load_dataset(dataset)
    manifest = SyntheticManifest(REAL_K20 / 'synth/manifest.json')
    inputs = identity['inputs']
    groups = next((p for p in inputs if p.endswith('/groups.json')), None)
    weights = next((p for p in inputs if p.endswith('/weights.json')), None)
    plan = expansion_plan(dataset, rows, read(REAL_K20 / 'attempt/records.json'), manifest,
                          variant=variant, groups=groups, weights=weights)
    recorded = read(root / 'allocation.json')
    assert recorded == identity['plan']
    # Historical D-data predates the same-category fallback already on HEAD.
    assert set(plan) - set(recorded) <= {'group_sources', 'fallback_rule'}
    assert {k: v for k, v in plan.items() if k in recorded} == recorded
