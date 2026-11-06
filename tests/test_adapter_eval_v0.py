"""Offline V0 identity checks; real artifacts are opt-in and read-only.

Set V0_ADAPTER_EVAL_CHECKPOINT and V0_ADAPTER_EVAL_SYNTH_MANIFEST to also
verify the round-1 checkpoint and its held-out synthetic probe inventory.
Set V0_ADAPTER_EVAL_FILTERED_RUN to audit filtered E4 JSONs without weights.
"""
from copy import deepcopy
import os
from pathlib import Path

import pytest

from repair.adapter_eval import _v0_exposure, leakage_check, training_identity
from repair.artifacts import read_json
from sed.storage import file_hash, write_json


ADAPTER_FILES = [('adapter_model.safetensors', 'adapter_weights_sha256'),
                 ('adapter_config.json', 'adapter_config_sha256')]


def adapter_at(checkpoint):
    return dict(path=str(checkpoint), **{field: file_hash(checkpoint / name)
                                       for name, field in ADAPTER_FILES})


def save_run(adapter, run, update, rows=None):
    checkpoint = Path(adapter['path'])
    if rows is not None:
        path = checkpoint.parent.parent / 'example_sources.json'
        write_json(path, rows)
        run['artifacts_sha256'] = {path.name: file_hash(path)}
    write_json(checkpoint.parent.parent / 'manifest.json', run)
    write_json(checkpoint.parent / 'manifest.json', update)


@pytest.fixture
def v0_run(tmp_path):
    checkpoint = tmp_path / 'v0-run/update-3/checkpoint'
    checkpoint.mkdir(parents=True)
    for name, _ in ADAPTER_FILES:
        (checkpoint / name).write_bytes(b'fake adapter artifact; never loaded')
    adapter = adapter_at(checkpoint)
    rows = [dict(task_id=f'train/synth/{b}', group_id=f'synth/{b}-fingerprint',
                 example_id=b, v0=dict(bucket=b, episode_id=f'episode-{b}')) for b in 'RGS']
    rows.append(dict(task_id='train/seed/trial', group_id='train/seed', example_id='D',
                     v0=dict(bucket='D', episode_id='episode-D')))
    rows.append(dict(rows[0], example_id='R-second-decision'))
    buckets = {b: dict(rows=2 if b == 'R' else 1, episodes=1) for b in 'RGSD'}
    realised = dict(episode_exposures={'episode-R': 2, 'episode-S': 1, 'episode-D': 1},
                    task_groups_seen=3, unique_instances_seen=3)
    run = dict(schema='v0-training-run/1', dataset_sha256='d' * 64,
               episodes=4, task_groups=4, unique_instances=4, buckets=buckets,
               training_exposure=4, realised_exposure=realised,
               input_manifest=dict(schema='v0-train-interchange/1', counts=deepcopy(buckets),
                                   training_exposure=0, synthetic_manifest='/unavailable/synth/manifest.json'))
    update = dict(optimizer_updates=3, sha256={name: adapter[field] for name, field in ADAPTER_FILES},
                  **deepcopy(realised))
    save_run(adapter, run, update, rows)
    return adapter, run, update, rows


def test_v0_inventory_covers_every_bucket_including_unsampled_episodes(v0_run):
    adapter, run, _, rows = v0_run
    identity, inputs = training_identity(adapter)
    assert identity['arm'] == 'v0:' + run['dataset_sha256']
    assert identity['optimizer_updates'] == 3
    assert identity['training_exposure'] == dict(task_ids=sorted({r['task_id'] for r in rows}),
                                                group_ids=sorted({r['group_id'] for r in rows}))
    assert 'train/synth/G' in identity['training_exposure']['task_ids']
    assert 'train/synth' not in identity['training_exposure']['group_ids']
    assert len(inputs) == 3 and inputs[-1].endswith('/example_sources.json')
    assert identity['manifest_sha256'] == file_hash(identity['manifest_path'])
    assert identity['checkpoint_manifest_sha256'] == file_hash(identity['checkpoint_manifest_path'])
    assert any(run['artifacts_sha256']['example_sources.json'] in s for s in identity['exposure_sources'])


def test_v0_variant_and_additive_run_input_update_exposure(v0_run):
    adapter, run, update, _ = v0_run
    run['variant'] = 'D-data'
    run['training_exposure'] = dict(task_ids=['train/prior/trial'])
    run['input_manifest']['training_exposure_groups'] = ['train/replay']
    update['training_exposure'] = dict(task_ids=['train/update/trial', 'train/synth/R'])
    save_run(adapter, run, update)
    identity, _ = training_identity(adapter)
    assert identity['arm'] == 'v0:D-data'
    assert {'train/prior/trial', 'train/update/trial'} <= set(identity['training_exposure']['task_ids'])
    assert {'train/prior', 'train/update', 'train/replay', 'synth/R-fingerprint'} <= set(
        identity['training_exposure']['group_ids'])
    assert 'train/synth' not in identity['training_exposure']['group_ids']


@pytest.mark.parametrize('task', ['train/synth/G', 'train/seed/other-trial'])
def test_v0_disjointness_still_rejects_task_and_group_overlap(v0_run, task):
    adapter, _, _, _ = v0_run
    identity, _ = training_identity(adapter)
    with pytest.raises(ValueError, match='overlaps adapter training exposure'):
        leakage_check([dict(spec=dict(task_id=task))], dict(training=identity))


@pytest.mark.parametrize('name,field', ADAPTER_FILES)
@pytest.mark.parametrize('tamper', ['missing_hash', 'changed_hash', 'missing_evaluated_hash', 'changed_file'])
def test_v0_adapter_hashes_fail_closed(v0_run, name, field, tamper):
    adapter, run, update, _ = v0_run
    if tamper == 'missing_hash':
        del update['sha256'][name]
    elif tamper == 'changed_hash':
        update['sha256'][name] = '0' * 64
    elif tamper == 'missing_evaluated_hash':
        del adapter[field]
    else:
        path = Path(adapter['path']) / name
        path.write_bytes(b'changed evaluated artifact')
        adapter[field] = file_hash(path)
    save_run(adapter, run, update)
    with pytest.raises(ValueError, match='V0 checkpoint manifest differs'):
        training_identity(adapter)


@pytest.mark.parametrize('updates', [None, True, '3', -1, 2])
def test_v0_update_number_must_match_checkpoint_directory(v0_run, updates):
    adapter, run, update, _ = v0_run
    update['optimizer_updates'] = updates
    save_run(adapter, run, update)
    with pytest.raises(ValueError, match='optimizer_updates'):
        training_identity(adapter)


@pytest.mark.parametrize('tamper', ['missing_file', 'missing_hash', 'changed_file'])
def test_v0_exposure_sources_are_required_and_hash_bound(v0_run, tamper):
    adapter, run, update, rows = v0_run
    path = Path(adapter['path']).parent.parent / 'example_sources.json'
    if tamper == 'missing_file':
        path.unlink()
    elif tamper == 'missing_hash':
        run['artifacts_sha256'] = {}
    else:
        write_json(path, rows[:-1])
    save_run(adapter, run, update)
    with pytest.raises((ValueError, FileNotFoundError)):
        training_identity(adapter)


@pytest.mark.parametrize('tamper', ['empty', 'missing_task', 'missing_group', 'missing_episode',
                                   'missing_bucket', 'omitted_episode', 'wrong_group', 'conflicting_group'])
def test_v0_incomplete_or_conflicting_exposure_fails_closed(v0_run, tamper):
    adapter, run, update, rows = v0_run
    if tamper == 'empty':
        rows = []
    elif tamper in ('missing_task', 'missing_group'):
        del rows[0]['task_id' if tamper == 'missing_task' else 'group_id']
    elif tamper in ('missing_episode', 'missing_bucket'):
        del rows[0]['v0']['episode_id' if tamper == 'missing_episode' else 'bucket']
    elif tamper == 'omitted_episode':
        rows = [r for r in rows if r['v0']['bucket'] != 'G']
    elif tamper == 'wrong_group':
        rows[0]['group_id'] = 'train/synth'
    else:
        rows[-1]['group_id'] = 'synth/conflicting-fingerprint'
    save_run(adapter, run, update, rows)
    with pytest.raises(ValueError, match='V0 training exposure'):
        training_identity(adapter)


@pytest.mark.parametrize('source', ['run', 'checkpoint'])
def test_v0_unknown_realised_episode_is_not_silently_dropped(v0_run, source):
    adapter, run, update, _ = v0_run
    accounting = run['realised_exposure'] if source == 'run' else update
    accounting['episode_exposures']['missing-episode'] = 1
    save_run(adapter, run, update)
    with pytest.raises(ValueError, match='unknown exposed episodes'):
        training_identity(adapter)


@pytest.fixture
def filtered_v0_run(v0_run):
    adapter, run, update, rows = v0_run
    run['filter'] = dict(
        kept_episode_ids={'R': ['episode-R'], 'D': ['episode-D']},
        dropped_episode_ids={'G': ['episode-G'], 'S': ['episode-S']},
        buckets={b: dict(rows=2 if b == 'R' else int(b == 'D'), episodes=int(b in 'RD')) for b in 'RGSD'},
        kept_rows=3, drop_buckets=['G', 'S'], keep_episodes_sha256='f' * 64)
    realised = dict(episode_exposures={'episode-R': 2, 'episode-D': 1},
                    unique_instances_seen=2, task_groups_seen=2)
    run['realised_exposure'] = deepcopy(realised)
    update.update(deepcopy(realised))
    update['realised_exposure'] = deepcopy(realised)
    save_run(adapter, run, update)
    return adapter, run, update, rows


@pytest.mark.parametrize('null_filter', [False, True])
def test_v0_no_filter_preserves_inventory_sources_and_scope(v0_run, null_filter):
    adapter, run, update, rows = v0_run
    if null_filter:
        run['filter'] = None
        save_run(adapter, run, update)
    identity, _ = training_identity(adapter)
    assert identity['training_exposure'] == dict(task_ids=sorted({r['task_id'] for r in rows}),
                                                group_ids=sorted({r['group_id'] for r in rows}))
    assert identity['exposure_sources'] == [
        'run.example_sources.json (sha256=' + run['artifacts_sha256']['example_sources.json'] +
        '; all R/G/S/D examples)', 'run.realised_exposure.episode_exposures', 'checkpoint.episode_exposures']
    assert identity['exposure_scope'] == 'declared IDs plus conservative complete R/G/S/D example inventory exclusions'


@pytest.mark.parametrize('keep_hash', ['f' * 64, None])
def test_v0_filtered_inventory_and_leakage(filtered_v0_run, keep_hash):
    adapter, run, update, _ = filtered_v0_run
    if keep_hash is None:
        del run['filter']['keep_episodes_sha256']
        save_run(adapter, run, update)
    identity, inputs = training_identity(adapter)
    assert identity['training_exposure'] == dict(task_ids=['train/seed/trial', 'train/synth/R'],
                                                group_ids=['synth/R-fingerprint', 'train/seed'])
    assert identity['exposure_sources'] == [
        'run.example_sources.json (sha256=' + run['artifacts_sha256']['example_sources.json'] +
        '; R/G/S/D examples of run.filter.kept_episode_ids, keep_episodes_sha256=' + str(keep_hash) +
        '; dropped before training)', 'run.realised_exposure.episode_exposures', 'checkpoint.episode_exposures',
        'checkpoint.realised_exposure.episode_exposures']
    assert identity['exposure_scope'] == ('declared IDs plus conservative kept-episode R/G/S/D inventory exclusions '
                                         '(pre-training filter)')
    assert len(inputs) == 3 and inputs[-1].endswith('/example_sources.json')
    for task in ('train/synth/G', 'train/synth/S'):
        assert leakage_check([dict(spec=dict(task_id=task))], dict(training=identity))['status'] == 'passed'
    for task in ('train/synth/R', 'train/seed/other-trial'):
        with pytest.raises(ValueError, match='overlaps adapter training exposure'):
            leakage_check([dict(spec=dict(task_id=task))], dict(training=identity))


def test_v0_filtered_inventory_keeps_unsampled_and_additive_exclusions(filtered_v0_run):
    adapter, run, update, _ = filtered_v0_run
    for accounting in (run['realised_exposure'], update, update['realised_exposure']):
        accounting.update(episode_exposures={'episode-G': 0, 'episode-S': 0},
                          unique_instances_seen=0, task_groups_seen=0)
    run['training_exposure'] = dict(task_ids=['train/prior/trial'])
    run['input_manifest']['training_exposure_groups'] = ['train/replay']
    update['training_exposure'] = dict(task_ids=['train/synth/G'])
    save_run(adapter, run, update)
    identity, _ = training_identity(adapter)
    assert identity['training_exposure'] == dict(
        task_ids=['train/prior/trial', 'train/seed/trial', 'train/synth/G', 'train/synth/R'],
        group_ids=['synth/G-fingerprint', 'synth/R-fingerprint', 'train/prior', 'train/replay', 'train/seed'])
    assert identity['exposure_sources'][-3:] == ['run.manifest.training_exposure.task_ids',
        'run.input_manifest.manifest.training_exposure_groups', 'checkpoint.manifest.training_exposure.task_ids']


def test_v0_filter_allows_omitted_empty_buckets_and_optional_counts(filtered_v0_run):
    adapter, run, update, _ = filtered_v0_run
    for bucket in 'GS':
        del run['filter']['buckets'][bucket]
    for key in ('kept_rows', 'drop_buckets'):
        del run['filter'][key]
    save_run(adapter, run, update)
    identity, _ = training_identity(adapter)
    assert identity['training_exposure']['task_ids'] == ['train/seed/trial', 'train/synth/R']


@pytest.mark.parametrize('value', [False, 0, '', [], 'filter'])
def test_v0_filter_requires_object(filtered_v0_run, value):
    adapter, run, update, _ = filtered_v0_run
    run['filter'] = value
    save_run(adapter, run, update)
    with pytest.raises(ValueError, match=r'run.filter must be an object'):
        training_identity(adapter)


@pytest.mark.parametrize('key', ['kept_episode_ids', 'dropped_episode_ids'])
@pytest.mark.parametrize('value', ['missing', None, [], {'X': []}, {'R': None}, {'R': 'episode-R'},
                                  {'R': ['']}, {'R': [1]}, {'R': [{}]}, {'R': ['episode-R', 'episode-R']}])
def test_v0_filter_episode_mappings_fail_closed(filtered_v0_run, key, value):
    adapter, run, update, _ = filtered_v0_run
    if value == 'missing':
        del run['filter'][key]
    else:
        run['filter'][key] = value
    save_run(adapter, run, update)
    with pytest.raises(ValueError, match='run.filter.' + key):
        training_identity(adapter)


@pytest.mark.parametrize('tamper', ['kept_dropped_overlap', 'kept_bucket_overlap', 'dropped_bucket_overlap',
                                   'cross_bucket_overlap', 'missing_episode', 'unknown_episode', 'wrong_bucket'])
def test_v0_filter_episode_partition_fails_closed(filtered_v0_run, tamper):
    adapter, run, update, _ = filtered_v0_run
    kept, dropped = run['filter']['kept_episode_ids'], run['filter']['dropped_episode_ids']
    if tamper == 'kept_dropped_overlap':
        dropped['R'] = ['episode-R']
    elif tamper == 'kept_bucket_overlap':
        kept['D'].append('episode-R')
    elif tamper == 'dropped_bucket_overlap':
        dropped['S'].append('episode-G')
    elif tamper == 'cross_bucket_overlap':
        dropped['G'].append('episode-R')
    elif tamper == 'missing_episode':
        del dropped['G']
    elif tamper == 'unknown_episode':
        kept['R'].append('unknown-episode')
    else:
        kept['R'], kept['D'] = kept['D'], kept['R']
    save_run(adapter, run, update)
    with pytest.raises(ValueError, match='run.filter episode (IDs must be disjoint|partition)'):
        training_identity(adapter)


@pytest.mark.parametrize('path,value', [
    (('buckets',), None), (('buckets',), []), (('buckets',), {}), (('buckets',), {'X': {}}),
    (('buckets', 'R'), None), (('buckets', 'R'), {}), (('buckets', 'G'), {}),
    (('buckets', 'R', 'episodes'), 2), (('buckets', 'R', 'episodes'), True),
    (('buckets', 'R', 'rows'), 1), (('buckets', 'R', 'rows'), '2'),
    (('buckets', 'G', 'episodes'), 1), (('buckets', 'G', 'rows'), 1),
    (('kept_rows',), 4), (('kept_rows',), '3'), (('kept_rows',), 3.0),
    (('drop_buckets',), ['R']), (('drop_buckets',), ['D']),
    (('drop_buckets',), None), (('drop_buckets',), 'GS'), (('drop_buckets',), ['X']),
])
def test_v0_filter_counts_and_dropped_buckets_fail_closed(filtered_v0_run, path, value):
    adapter, run, update, _ = filtered_v0_run
    target = run['filter']
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    save_run(adapter, run, update)
    with pytest.raises(ValueError, match='run.filter.' + path[0]):
        training_identity(adapter)


@pytest.mark.parametrize('source', ['run.realised_exposure', 'checkpoint', 'checkpoint.realised_exposure'])
def test_v0_filter_rejects_any_realised_dropped_episode(filtered_v0_run, source):
    adapter, run, update, _ = filtered_v0_run
    accounting = {'run.realised_exposure': run['realised_exposure'], 'checkpoint': update,
                  'checkpoint.realised_exposure': update['realised_exposure']}[source]
    accounting['episode_exposures']['episode-G'] = 1
    accounting.update(unique_instances_seen=3, task_groups_seen=3)
    save_run(adapter, run, update)
    with pytest.raises(ValueError, match=source + r'.episode_exposures contains episodes not kept'):
        training_identity(adapter)


def test_v0_filter_rejects_empty_kept_inventory(filtered_v0_run):
    adapter, run, update, _ = filtered_v0_run
    run['filter'].update(kept_episode_ids={}, dropped_episode_ids={b: [f'episode-{b}'] for b in 'RGSD'},
                         buckets={b: dict(rows=0, episodes=0) for b in 'RGSD'}, kept_rows=0)
    for accounting in (run['realised_exposure'], update, update['realised_exposure']):
        accounting.update(episode_exposures={}, unique_instances_seen=0, task_groups_seen=0)
    save_run(adapter, run, update)
    with pytest.raises(ValueError, match='run.filter must keep at least one episode'):
        training_identity(adapter)


@pytest.mark.parametrize('tamper,message', [
    ('hash', 'hash mismatch'), ('row', 'task/group mismatch'), ('unique_instances', 'run.unique_instances'),
    ('task_groups', 'run.task_groups'), ('episodes', 'run.episodes'), ('run_buckets', 'run.buckets.G.rows'),
    ('input_buckets', 'input_manifest.counts.G.rows'), ('realised', 'run.realised_exposure.unique_instances_seen'),
    ('negative_count', 'counts must be nonnegative integers'), ('unknown_episode', 'unknown exposed episodes'),
])
def test_v0_full_inventory_checks_precede_filter_validation(filtered_v0_run, tamper, message):
    adapter, run, update, rows = filtered_v0_run
    run['filter'] = {}  # Must report the full-inventory failure before this invalid filter.
    if tamper == 'hash':
        run['artifacts_sha256']['example_sources.json'] = '0' * 64
    elif tamper == 'row':
        rows[1]['group_id'] = 'train/synth'  # A dropped G row must still be validated.
        save_run(adapter, run, update, rows)
    elif tamper in ('unique_instances', 'task_groups', 'episodes'):
        run[tamper] -= 1
    elif tamper == 'run_buckets':
        run['buckets']['G']['rows'] = 0
    elif tamper == 'input_buckets':
        run['input_manifest']['counts']['G']['rows'] = 0
    elif tamper == 'realised':
        run['realised_exposure']['unique_instances_seen'] = 1
    elif tamper == 'negative_count':
        run['realised_exposure']['episode_exposures']['episode-G'] = -1
    else:
        run['realised_exposure']['episode_exposures']['unknown-episode'] = 1
    save_run(adapter, run, update)
    with pytest.raises(ValueError, match=message):
        training_identity(adapter)


def test_real_v0_filtered_e4_inventory():
    run_arg = os.environ.get('V0_ADAPTER_EVAL_FILTERED_RUN')
    if not run_arg:
        pytest.skip('Set V0_ADAPTER_EVAL_FILTERED_RUN for read-only filtered E4 audit')
    root = Path(run_arg)
    run_path, update_path, sources_path = root / 'manifest.json', root / 'update-295/manifest.json', root / 'example_sources.json'
    before = {p: file_hash(p) for p in (run_path, update_path, sources_path)}
    run, update, rows = read_json(run_path), read_json(update_path), read_json(sources_path)
    exposure, _, inputs = _v0_exposure(run, update, run_path)
    assert inputs == [str(sources_path)]
    assert len(exposure['task_ids']) == len(exposure['group_ids']) == 59
    assert run['unique_instances'] == len({r['task_id'] for r in rows}) == 308
    assert run['task_groups'] == len({r['group_id'] for r in rows}) == 308
    assert run['episodes'] == len({r['v0']['episode_id'] for r in rows}) == 495
    kept = {e for ids in run['filter']['kept_episode_ids'].values() for e in ids}
    kept_rows = [r for r in rows if r['v0']['episode_id'] in kept]
    assert len(kept) == 75 and len(kept_rows) == run['filter']['kept_rows'] == 1177
    assert exposure == dict(task_ids=sorted({r['task_id'] for r in kept_rows}),
                            group_ids=sorted({r['group_id'] for r in kept_rows}))
    assert before == {p: file_hash(p) for p in before}


def test_real_v0_round1_identity_and_heldout_probes():
    checkpoint_arg = os.environ.get('V0_ADAPTER_EVAL_CHECKPOINT')
    synth_arg = os.environ.get('V0_ADAPTER_EVAL_SYNTH_MANIFEST')
    if not checkpoint_arg or not synth_arg:
        pytest.skip('Set V0_ADAPTER_EVAL_CHECKPOINT and V0_ADAPTER_EVAL_SYNTH_MANIFEST for read-only audit')
    checkpoint, synth_path = Path(checkpoint_arg), Path(synth_arg)
    adapter = adapter_at(checkpoint)
    identity, inputs = training_identity(adapter)
    before = {p: file_hash(p) for p in [*inputs, str(synth_path)]}
    run = read_json(checkpoint.parent.parent / 'manifest.json')
    rows = read_json(checkpoint.parent.parent / 'example_sources.json')
    exposure = identity['training_exposure']
    assert identity['optimizer_updates'] == 2193
    assert identity['arm'] == 'v0:' + run.get('variant', run['dataset_sha256'])
    assert len(exposure['task_ids']) == run['unique_instances'] == 308
    assert len(exposure['group_ids']) == run['task_groups'] == 308
    assert set(exposure['task_ids']) == {r['task_id'] for r in rows}
    assert set(exposure['group_ids']) == {r['group_id'] for r in rows}
    assert {b: len({r['v0']['episode_id'] for r in rows if r['v0']['bucket'] == b})
            for b in 'RGSD'} == dict(R=134, G=166, S=175, D=20)
    # Inventory excludes all 495 episodes, including the seven never sampled.
    seen = {e for e, n in run['realised_exposure']['episode_exposures'].items() if n}
    assert len(seen) == 488
    assert len({r['group_id'] for r in rows if r['v0']['episode_id'] in seen}) == 307
    assert run['realised_exposure']['task_groups_seen'] == 307
    probes = [t for t in read_json(synth_path)['tasks'] if t['heldout_probe'] is True]
    assert len(probes) == len({t['task_id'] for t in probes}) == 71
    assert {t['task_id'] for t in probes}.isdisjoint(exposure['task_ids'])
    assert {t['group_id'] for t in probes}.isdisjoint(exposure['group_ids'])
    report = leakage_check([dict(spec=dict(task_id=t['task_id'])) for t in probes], dict(training=identity))
    assert report['status'] == 'passed'
    assert not report['overlapping_task_ids'] and not report['overlapping_group_ids']
    assert before == {p: file_hash(p) for p in before}
    print('\nReal V0 audit: 495 R/G/S/D episodes, 308 tasks/groups; '
          '488 sampled episodes, 307 sampled groups; 71 probes disjoint by task and explicit group.')
