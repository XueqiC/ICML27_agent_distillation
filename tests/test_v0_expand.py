"""Same-category expansion fallback: fake candidates, no models or network."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from repair import v0_expand
from repair.v0_common import read
from repair.v0_expand import expansion_plan, generate
from sed.storage import write_json
from test_v0_round import SEED, manifest_at, task_row
from tools import v0_round


LOOK = dict(bin='mid', generator='G2', goal_task_type='look_at_obj_in_light')
NEAR = dict(bin='near', generator='G1', destination_receptacle_type='Fridge')
DEAD = dict(bin='near', generator='G3', error_type='wrong_object')
OTHER_SEED = 'train/pick_and_place_simple-Mug-None-CounterTop-2/trial_0'


def source(eid, kind, parent=SEED):
    return dict(source_episode_id=eid, source_synthetic_task_id='train/synth/' + eid,
                preferred_seed=parent, restriction=kind)


def request_at(tmp_path, *, kinds=None):
    original = source('z-original', LOOK)
    # Intentionally unsorted; generation must use episode_id order.
    sources = [original, source('c-fresh', NEAR), source('a-exhausted', DEAD)]
    units = [dict(unit=i, group='category', **source('z-original', kind))
             for i, kind in enumerate(kinds or [LOOK])]
    return dict(seeds='fake', data_root='fake', seed_task_ids=[SEED, OTHER_SEED], seed=0,
        heldout=[], out=str(tmp_path), excluded_fingerprints=['initial', 'probe'],
        plan=dict(units=units, requested_items=len(units), group_sources={'category': sources}))


def fake_candidates(monkeypatch, pools):
    """Each seed has a finite pool; obey the production factory's seen contract."""
    monkeypatch.setattr(v0_expand.factory, 'load_seeds', lambda *a: (
        [SimpleNamespace(task_id=tid) for tid in (SEED, OTHER_SEED)], {}))
    calls = []

    class Factory:
        def __init__(self, parent, seeds, directory, *args, excluded, **kwargs):
            self.parent, self.directory = parent, directory
            self.seen, self.records = set(excluded), []

        def next_valid(self, generator, bin_name, restriction):
            calls.append((self.parent.task_id, restriction))
            for fp in pools.get((self.parent.task_id, generator), []):
                if fp not in self.seen:
                    self.seen.add(fp)
                    row = dict(task_id='train/synth/' + fp, item_fingerprint=fp,
                               generator=generator, bin=bin_name)
                    self.records.append(row)
                    return deepcopy(row)

    monkeypatch.setattr(v0_expand, 'SeedFactory', Factory)
    return calls


def test_fallback_uses_same_category_sorted_repairs_and_fresh_seed_edits(tmp_path, monkeypatch):
    calls = fake_candidates(monkeypatch, {(SEED, 'G1'): ['initial', 'probe', 'fresh-1', 'fresh-2']})
    request = request_at(tmp_path, kinds=[LOOK, LOOK])
    request['plan']['group_sources']['other-category'] = [source('0-forbidden', NEAR)]
    rows = generate(request)
    assert [r['item_fingerprint'] for r in rows] == ['fresh-1', 'fresh-2']
    assert all(r['parent_seed_task_id'] == SEED for r in rows)
    assert all(r['expansion_unit']['group'] == 'category' for r in rows)
    assert all(r['expansion_source'] == source('c-fresh', NEAR) for r in rows)
    log = read(tmp_path / 'allocation_execution.json')
    for row in log:
        assert row['status'] == 'filled'
        assert row['fallback_from'] == source('z-original', LOOK)
        assert row['fallback_to'] == source('c-fresh', NEAR)
        assert [a['source']['source_episode_id'] for a in row['source_attempts']] == [
            'z-original', 'a-exhausted', 'c-fresh']
        assert all(a['tried_seed_task_ids'] == [SEED, OTHER_SEED] for a in row['source_attempts'][:2])
    assert calls[:5] == [(SEED, LOOK), (OTHER_SEED, LOOK), (SEED, DEAD), (OTHER_SEED, DEAD), (SEED, NEAR)]
    assert read(tmp_path / 'summary.json') == dict(requested_items=2, generated_items=2,
        fallback_count=2, fallback_attempted_count=2, unfilled_count=0, strict_fill=False)


def test_all_seeds_exhausted_before_trying_another_source(tmp_path, monkeypatch):
    fake_candidates(monkeypatch, {(OTHER_SEED, 'G2'): ['original-kind']})
    rows = generate(request_at(tmp_path))
    assert rows[0]['parent_seed_task_id'] == OTHER_SEED
    log = read(tmp_path / 'allocation_execution.json')[0]
    assert log['fallback_from'] is log['fallback_to'] is None
    assert len(log['source_attempts']) == 1


@pytest.mark.parametrize('strict', [False, True])
def test_exhausted_category_continues_and_strict_cli_fails_after_writing_evidence(tmp_path, monkeypatch, strict):
    fake_candidates(monkeypatch, {(SEED, 'G1'): ['later-category']})
    request = request_at(tmp_path)
    request['plan']['group_sources']['category'] = [source('a-exhausted', DEAD)]
    request['plan']['units'].append(dict(unit=1, group='later', **source('later-repair', NEAR)))
    request['plan']['requested_items'] = 2
    path = tmp_path / 'request.json'
    write_json(path, request)
    # pytest's autouse offline fixture already prohibits network and live API.
    monkeypatch.setattr('repair.synth_validation.offline', lambda: None)
    if strict:
        with pytest.raises(ValueError, match='candidate space exhausted; requested total not met'):
            v0_expand.main(['--request', str(path), '--strict-fill'])
    else:
        v0_expand.main(['--request', str(path)])
    log = read(tmp_path / 'allocation_execution.json')
    assert [r['status'] for r in log] == ['unfilled_after_fallback', 'filled']
    assert log[0]['fallback_from'] == source('z-original', LOOK)
    assert log[0]['fallback_to'] is None
    assert len(log[0]['source_attempts']) == 2
    assert read(tmp_path / 'generated.json')[0]['task_id'] == 'train/synth/later-category'
    assert (tmp_path / 'attempts.json').exists()
    assert read(tmp_path / 'summary.json') == dict(requested_items=2, generated_items=1,
        fallback_count=0, fallback_attempted_count=1, unfilled_count=1, strict_fill=strict)


def test_fallback_deterministic_across_fresh_outputs_and_source_order(tmp_path, monkeypatch):
    fake_candidates(monkeypatch, {(SEED, 'G1'): ['fresh-1', 'fresh-2']})
    request = request_at(tmp_path / 'one', kinds=[LOOK, LOOK])
    original_request = deepcopy(request)
    left = generate(request)
    assert request == original_request
    request['out'] = str(tmp_path / 'two')
    request['plan']['group_sources']['category'].reverse()
    right = generate(request)
    assert [{k: v for k, v in r.items() if k != 'asset_root'} for r in left] == [
        {k: v for k, v in r.items() if k != 'asset_root'} for r in right]
    for name in ('allocation_execution.json', 'summary.json', 'attempts.json'):
        assert (tmp_path / 'one' / name).read_bytes() == (tmp_path / 'two' / name).read_bytes()


@pytest.mark.parametrize('variant', ['D-param', 'D-data'])
def test_plan_retains_all_eligible_repairs_even_when_not_allocated(tmp_path, monkeypatch, variant):
    tasks = [task_row('train/synth/' + str(i)) for i in range(4)]
    tasks[1]['edit']['destination_receptacle_type'] = 'Fridge'
    manifest = manifest_at(tmp_path, tasks)
    rows = []
    for eid, task, category in [('z', tasks[0], 'same'), ('a', tasks[1], 'same'), ('b', tasks[2], 'other')]:
        path = tmp_path / (eid + '.json')
        write_json(path, dict(episode_id=eid, verified=True, success=True,
            rounds=[dict(review=dict(breaking_points=[dict(category=category)]))]))
        rows.append(dict(v0=dict(bucket='R', episode_id=eid), task_id=task['task_id'],
                         evidence=dict(episode_ref=str(path))))
    rows.append(dict(v0=dict(bucket='S', episode_id='not-a-repair'), task_id=tasks[3]['task_id']))
    attempts = [dict(task_id=tasks[0]['task_id'], success=False, heldout_probe=False)]
    groups, weights = tmp_path / 'groups.json', tmp_path / 'weights.json'
    identity = dict(dataset_sha256='dataset', adapter_sha256='adapter', parameter_names_sha256='params',
                    features_index_sha256='features')
    write_json(tmp_path / 'dataset/manifest.json', dict(theta0=dict(adapter_weights_sha256='adapter')))
    write_json(groups, dict(identity, schema='v0-groups/1', assignment={'a': 'same', 'z': 'same', 'b': 'other'},
        groups=[dict(group_id='same', members=['z', 'a', 'not-a-repair'], consistency=1),
                dict(group_id='other', members=['b'], consistency=1)]))
    # Non-repair membership must still have a consistent group assignment.
    group_doc = read(groups)
    group_doc['assignment']['not-a-repair'] = 'same'
    write_json(groups, group_doc)
    write_json(weights, dict(identity, schema='v0-episode-weights/1', status='valid', weights={'a': 1, 'z': 1, 'b': 1}))
    monkeypatch.setattr(v0_expand, 'consumer_dataset_hash', lambda *a, **kw: 'dataset')
    plan = expansion_plan(tmp_path / 'dataset', rows, attempts, manifest, variant=variant, groups=groups, weights=weights)
    assert plan['allocation'] == {'other': 1, 'same': 0}
    assert len(plan['units']) == 1
    assert [s['source_episode_id'] for s in plan['group_sources']['same']] == ['a', 'z']
    assert plan['group_sources']['same'][0]['restriction'] == NEAR
    assert [s['source_episode_id'] for s in plan['group_sources']['other']] == ['b']
    assert plan == expansion_plan(tmp_path / 'dataset', list(reversed(rows)), attempts, manifest,
                                 variant=variant, groups=groups, weights=weights)


@pytest.mark.parametrize('strict', [False, True])
def test_round_cli_records_strict_fill_and_dry_run(tmp_path, monkeypatch, strict):
    monkeypatch.setattr(v0_round, 'round_root', lambda args: tmp_path)
    monkeypatch.setattr(v0_round, 'manifest_for', lambda *args: None)
    captured = []
    monkeypatch.setattr(v0_expand, 'expand', lambda args, *a: captured.append(args) or {})
    argv = ['expand', '--output', str(tmp_path), '--seeds', 'fake', '--data-root', 'fake',
            '--main-checkout', 'fake', '--dataset', 'fake', '--variant', 'D-param', '--dry-run']
    assert v0_round.main(argv + (['--strict-fill'] if strict else [])) == 0
    assert captured[0].strict_fill is strict and captured[0].dry_run is True
