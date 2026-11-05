from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from repair.p0_agent import freeze_submission
from repair.p0_eval import (mcnemar_exact, compare, predictions, CONDITIONS, evaluation_barrier,
                            evaluate, restore_direction, control_edits, work_units, shard_owner,
                            merge_evaluation, parse_shard, HarnessEvaluation)
from repair.p0_introspect import ToolSession
from repair.p0_pairs import CATEGORIES
from sed.storage import digest, write_json
from p0_fakes import tiny_student, fake_pair, sample_submission


@pytest.mark.parametrize('left,right,p', [([True]*6, [False]*6, .03125),
    ([False]*6, [True]*6, .03125), ([True, False], [False, True], 1.0),
    ([True]*10, [True]*10, 1.0), ([], [], 1.0)])
def test_exact_mcnemar_known_values(left, right, p):
    assert mcnemar_exact(left, right)['p_value'] == p


def test_exact_mcnemar_preserves_unknowns():
    r = mcnemar_exact([True, 'unknown', False], [False, False, 'unknown'])
    assert r['n'] == 1 and r['unknown'] == 2 and r['left_only'] == 1
    with pytest.raises(ValueError):
        mcnemar_exact([True], [])


def rows(n=6):
    return [dict(pair_id=str(i), condition=c, success=c == 'teacher', decision_flip=c == 'teacher',
                 reproduction_matches=True if c == 'none' else None)
            for i in range(n) for c in CONDITIONS]


def test_flip_denominator_uses_only_theta0_reproductions():
    data = rows(3)
    for r in data:
        if r['condition'] == 'none':
            r['reproduction_matches'] = {'0': True, '1': False, '2': None}[r['pair_id']]
        if r['pair_id'] != '0':
            r['decision_flip'] = False
    result = compare(data)
    assert result['reproduction']['rate'] == dict(value=.5, n=2, unknown=1)
    assert result['reproduction']['mismatches'] == ['1']
    assert result['reproduction']['unavailable'] == ['2']
    assert result['metrics']['teacher']['decision_flip_rate'] == dict(value=1.0, n=1, unknown=0, denominator=1)
    assert result['metrics']['random_mean']['decision_flip_rate']['n'] == 1
    assert all(m['rollout_success']['n'] == 3 for m in result['metrics'].values())
    assert result['mcnemar']['none']['n'] == 3


def test_zero_reproductions_leave_flip_unknown_and_rollout_intact():
    data = rows()
    for r in data:
        r['reproduction_matches'] = False if r['condition'] == 'none' else None
    result = compare(data)
    assert result['reproduction']['rate']['value'] == 0
    for m in result['metrics'].values():
        assert m['decision_flip_rate'] == dict(value=None, n=0, unknown=0, denominator=0)
        assert m['rollout_success']['n'] == 6


def test_five_random_controls_are_not_pseudoreplicated():
    report = compare(rows())
    assert report['pairs'] == 6
    assert report['mcnemar']['random_mean']['p_value'] == .03125
    assert all(r['n'] == 6 for r in report['mcnemar']['random_mean']['per_seed'])


def test_comparison_refuses_missing_or_duplicate_conditions():
    with pytest.raises(ValueError):
        compare(rows()[:-1])
    with pytest.raises(ValueError):
        compare(rows() + rows()[:1])


def test_prediction_item_pass_fail():
    submission = dict(predictions=[dict(metric='rollout_gain_vs_none', direction='>', threshold=.1),
        dict(metric='collateral', direction='<=', threshold=.05),
        dict(metric='k20_success', direction='>=', threshold=1)])
    result = predictions(submission, compare(rows()), .1, 1.0)
    assert [r['passed'] for r in result] == [True, False, True]


def test_predictions_preserve_unknown_verification():
    data = rows()
    next(r for r in data if r['condition'] == 'teacher')['success'] = 'unknown'
    prediction = dict(predictions=[dict(metric='rollout_gain_vs_none', direction='>', threshold=0)])
    result = predictions(prediction, compare(data), 0, 1)
    assert result[0]['observed'] is None and result[0]['passed'] is None


def make_freezes(tmp_path, student=None):
    student = student or tiny_student()
    pairs, paths, frozen = [], [], []
    for j, category in enumerate(CATEGORIES):
        diagnosis = [fake_pair(j*10+i, category=category) for i in range(4)]
        tests = [fake_pair(j*10+4+i, category=category, split='test') for i in range(2)]
        pairs.extend(diagnosis + tests)
        session = ToolSession(student, diagnosis, tmp_path / category / 'tools')
        ident = session.call('direction', dict(positive=[dict(state=diagnosis[0]['pair_id'], action='look')],
            negative=[dict(state=diagnosis[1]['pair_id'], action='inventory')], layers=[0]))['data']['direction_id']
        path = tmp_path / category / 'submission.json'
        frozen.append(freeze_submission(path, sample_submission(ident), session,
            category=category, pairs_sha256='pairs', budget={}))
        paths.append(path)
    value = dict(sha256='pairs', pairs=pairs, counts={'pairs': 30})
    return student, value, paths, frozen


def test_all_five_submissions_required_before_any_evaluation(tmp_path):
    student, pairs, paths, _ = make_freezes(tmp_path)
    with pytest.raises(ValueError, match='five'):
        evaluation_barrier(tmp_path / 'evaluation', pairs, paths[:4], student.identity)
    assert not (tmp_path / 'evaluation/evaluation_freeze.json').exists()
    evaluation_barrier(tmp_path / 'evaluation', pairs, paths, student.identity)
    assert (tmp_path / 'evaluation/evaluation_freeze.json').exists()
    with pytest.raises(FileExistsError):
        evaluation_barrier(tmp_path / 'evaluation', pairs, paths, student.identity)


def test_frozen_direction_restore_is_content_bound(tmp_path):
    student, pairs, paths, frozen = make_freezes(tmp_path)
    spec = restore_direction(student, frozen[0])
    assert spec['direction_id'] in student.directions
    changed = deepcopy(frozen[0])
    changed['directions']['0'][0] += 1
    with pytest.raises(ValueError, match='content mismatch'):
        restore_direction(student, changed)


def test_fixed_direction_uses_exact_layers_and_teacher_alpha(tmp_path, monkeypatch):
    student = tiny_student(42)
    student, pairs, paths, frozen = make_freezes(tmp_path, student)
    diagnosis = [p for p in pairs['pairs'] if p['split'] == 'diagnosis' and p['category'] == CATEGORIES[0]]
    original_direction = student.direction
    def checked_direction(positive, negative, layers, *, framing):
        assert framing == 'original_thought'
        assert layers == list(range(20, 42))
        assert [a for _, a in positive] == [p['a_f'] for p in diagnosis]
        assert [a for _, a in negative] == [p['a_r'] for p in diagnosis]
        assert all(state['original_thought']['prefix'] for state, _ in positive + negative)
        return original_direction(positive, negative, layers, framing=framing)
    monkeypatch.setattr(student, 'direction', checked_direction)
    edits = control_edits(student, frozen[0], diagnosis)
    assert edits['fixed']['layers'] == list(range(20, 42))
    assert edits['fixed']['alpha'] == edits['teacher']['alpha']
    assert edits['fixed']['targets'] == edits['teacher']['targets']
    assert len({edits[f'random_{i}']['direction_id'] for i in range(5)}) == 5
    assert all(edits[f'random_{i}']['layers'] == edits['teacher']['layers'] for i in range(5))
    again = control_edits(student, frozen[0], diagnosis)
    assert edits == again


class FakeRuntime:
    def __init__(self, student, pairs, directory):
        # The freeze MUST precede even constructing an evaluator.
        assert (directory.parent.parent / 'evaluation_freeze.json').is_file()
        self.collateral = [dict(pair_id=f'collateral-{i}') for i in range(200)]
        self.seed_ids = [f'train/seed/{i}' for i in range(20)]
        self.seed_sources = {key: {'fixture': True} for key in self.seed_ids}

    def pair(self, pair, edit):
        good = edit is not None and edit['alpha'] == .5
        return dict(success=good, greedy='inventory' if good else 'look', decision_flip=good,
                    reproduction_matches=pair['pair_id'].endswith('4') if edit is None else None)

    def collateral_decision(self, pair_id, edit):
        return dict(pair_id=pair_id, edited='look')

    def k20_task(self, task_id, edit):
        return dict(task_id=task_id, success=True, environment_steps=2)


def fake_edits(student, frozen, diagnosis):
    teacher = restore_direction(student, frozen)
    return {k: None if k == 'none' else dict(teacher, alpha=.5 if k == 'teacher' else .1) for k in CONDITIONS}


def fake_evaluate(student, pairs, paths, directory, **kwargs):
    return evaluate(student, pairs, paths, directory, runtime_factory=FakeRuntime, edits_factory=fake_edits, **kwargs)


@pytest.mark.parametrize('full_plan', [False, True])
def test_evaluation_end_to_end_fake_runtime_no_teacher(tmp_path, full_plan):
    student, pairs, paths, frozen = make_freezes(tmp_path)
    directory = tmp_path / 'evaluation'
    report = fake_evaluate(student, pairs, paths, directory, full_plan=full_plan)
    name = 'full' if full_plan else 'prereg_minimal'
    measured = list(CONDITIONS) if full_plan else ['none', 'teacher']
    assert report['plan'] == name
    assert report['side_measurements'] == dict(collateral=measured, k20=measured,
                                              none_shared_across_categories=not full_plan)
    for category in CATEGORIES:
        assert set(report['collateral'][category]) == set(report['k20'][category]) == set(measured)
        for metric in ('collateral', 'k20'):
            if not full_plan:
                assert report[metric][category]['none'] is report[metric][CATEGORIES[0]]['none']
    plan = json.loads((directory / 'work_plan.json').read_text())
    freeze = json.loads((directory / 'evaluation_freeze.json').read_text())
    assert plan['plan'] == freeze['plan'] == freeze['identity']['plan'] == name
    assert len(plan['units']) == (80 + 5 * 8 * 220 if full_plan else 80 + 6 * 220)
    assert report['teacher_calls'] == 0 and report['complete']
    assert report['pooled']['pairs'] == 10
    assert report['reproduction']['rate'] == dict(value=.5, n=10, unknown=0)
    assert report['pooled']['metrics']['teacher']['decision_flip_rate']['denominator'] == 5
    assert report['pooled']['metrics']['teacher']['rollout_success']['n'] == 10
    assert report['preregistered_numeric_criterion'] is True
    assert report['diagnosis_fidelity']['value'] == 1.0
    assert (tmp_path / 'evaluation/report.md').is_file()
    assert len((tmp_path / 'evaluation/results.jsonl').read_text().splitlines()) == 80
    markdown = (directory / 'report.md').read_text()
    assert f'Plan: `{name}`' in markdown
    assert ('not measured' in markdown) is (not full_plan)
    if not full_plan:
        assert 'only teacher edits have collateral/K20 measured' in markdown


@pytest.mark.parametrize('edit', [None, {'fixture': 'edit'}])
def test_collateral_unit_runs_one_forward_without_implicit_baseline(edit):
    calls = []
    runtime = HarnessEvaluation.__new__(HarnessEvaluation)
    runtime.student = SimpleNamespace(edited=lambda e: nullcontext(),
        greedy=lambda state: calls.append(state) or dict(action='look'))
    runtime.collateral_by_id = {'c': 0}
    runtime.collateral_states = [{'text': 'visible'}]
    assert runtime.collateral_decision('c', edit) == dict(pair_id='c', edited='look')
    assert calls == runtime.collateral_states


def test_shared_none_executed_once_and_reused_for_actual_comparisons(tmp_path):
    calls = Counter()
    class Runtime(FakeRuntime):
        def collateral_decision(self, pair_id, edit):
            calls['collateral', pair_id, edit is None] += 1
            # The saved trajectory action need not match fresh theta0 greedy.
            return dict(pair_id=pair_id, edited='look' if edit is None else 'inventory')

        def k20_task(self, task_id, edit):
            calls['k20', task_id, edit is None] += 1
            return dict(task_id=task_id, success=edit is None, environment_steps=2)

    student, pairs, paths, _ = make_freezes(tmp_path)
    report = evaluate(student, pairs, paths, tmp_path / 'evaluation',
                      runtime_factory=Runtime, edits_factory=fake_edits)
    assert sum(calls.values()) == 6 * (200 + 20)
    assert all(n == (1 if unedited else 5) for (_, _, unedited), n in calls.items())
    for category in CATEGORIES:
        assert report['collateral'][category]['none']['fraction'] == 0
        assert report['collateral'][category]['teacher']['fraction'] == 1
        assert all(r['original'] == 'look' and r['changed'] is True
                   for r in report['collateral'][category]['teacher']['decisions'])
        assert report['k20'][category]['none']['rate']['value'] == 1
        assert report['k20'][category]['teacher']['rate']['value'] == 0
    assert report['complete'] is True
    assert report['preregistered_numeric_criterion'] is False


@pytest.mark.parametrize('count', range(1, 5))
@pytest.mark.parametrize('full_plan', [False, True])
def test_work_units_partition_exactly_once(count, full_plan):
    pairs = dict(pairs=[fake_pair(i, category=c, split='test') for i, c in enumerate(CATEGORIES)])
    units = work_units(pairs, [f'c{i}' for i in range(200)], [f't{i}' for i in range(20)], full_plan=full_plan)
    shards = [[u for u in units if shard_owner(u, count) == i] for i in range(count)]
    assert sum(map(len, shards)) == len(units) == (5 * 8 * 221 if full_plan else 5 * 8 + 6 * 220)
    assert len({digest(u) for shard in shards for u in shard}) == len(units)
    assert all(shard_owner(u, count) == shard_owner(dict(reversed(list(u.items()))), count) for u in units)
    assert {u['kind'] for u in units} == {'pair', 'collateral', 'k20'}


def test_preregistered_freeze_dimensions_have_2040_units_with_720_pair_rollouts():
    # Freeze dimensions only; never open test states or construct a runtime.
    pairs = dict(pairs=[dict(pair_id=f'{c}-{i}', category=c, split='test')
                       for c, n in zip(CATEGORIES, [23, 32, 14, 8, 13]) for i in range(n)])
    collateral, seeds = [f'c{i}' for i in range(200)], [f't{i}' for i in range(20)]
    units = work_units(pairs, collateral, seeds)
    assert len(units) == 2040
    assert Counter(u['kind'] for u in units) == dict(pair=720, collateral=1200, k20=120)
    pair_units = [u for u in units if u['kind'] == 'pair']
    for pair in pairs['pairs']:
        assert {u['condition'] for u in pair_units if u['key'] == pair['pair_id']} == set(CONDITIONS)
        assert all(u['category'] == pair['category'] for u in pair_units if u['key'] == pair['pair_id'])
    shared = [u for u in units if u['kind'] != 'pair' and u['condition'] == 'none']
    assert len(shared) == 220 and all('category' not in u for u in shared)
    assert all(u['condition'] == 'teacher' for u in units if u['kind'] != 'pair' and 'category' in u)
    full = work_units(pairs, collateral, seeds, full_plan=True)
    assert len(full) == 9520
    assert [u for u in full if u['kind'] == 'pair'] == pair_units


@pytest.mark.parametrize('value', ['1/1', '-1/2', '0/0', '0', '1/2/3', '0/a', '0/ 2'])
def test_invalid_shard(value):
    with pytest.raises(ValueError, match='zero-based'):
        parse_shard(value)


@pytest.mark.parametrize('full_plan', [False, True])
def test_sharded_merge_equals_unsharded_and_refuses_missing_duplicate(tmp_path, monkeypatch, full_plan):
    from repair import p0_eval
    # Fix audit time only; no result normalization or production evidence edits.
    monkeypatch.setattr('repair.p0_pairs.utc_now', lambda: '2000-01-01T00:00:00Z')
    student, pairs, paths, _ = make_freezes(tmp_path)
    plain, sharded = tmp_path / 'plain', tmp_path / 'sharded'
    expected = fake_evaluate(student, pairs, paths, plain, full_plan=full_plan)
    fake_evaluate(student, pairs, paths, sharded, shard='0/3', full_plan=full_plan)
    with pytest.raises(ValueError, match='Missing'):
        merge_evaluation(pairs, paths, sharded, full_plan=full_plan)
    assert not (sharded / 'report.json').exists()
    with pytest.raises(FileExistsError):
        fake_evaluate(student, pairs, paths, sharded, shard='0/3', full_plan=full_plan)
    from concurrent.futures import ThreadPoolExecutor
    def run_shard(index):
        independent = SimpleNamespace(identity=student.identity, directions={})
        return fake_evaluate(independent, pairs, paths, sharded, shard=f'{index}/3', full_plan=full_plan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run_shard, [2, 1]))
    path = sharded / 'shards/0000-of-0003/result.json'
    original = json.loads(path.read_text())
    changed = deepcopy(original)
    changed['records'].append(changed['records'][0])
    write_json(path, changed)
    with pytest.raises(ValueError, match='Duplicate evaluation unit'):
        merge_evaluation(pairs, paths, sharded, full_plan=full_plan)
    changed = deepcopy(original)
    plan = json.loads((sharded / 'work_plan.json').read_text())
    baseline_ids = {digest(u) for u in plan['units'] if u['condition'] == 'none' and u['kind'] == 'k20'}
    missing = next(r for r in changed['records'] if r['unit_id'] in baseline_ids)
    changed['records'].remove(missing)
    write_json(path, changed)
    with pytest.raises(ValueError, match='Missing'):
        merge_evaluation(pairs, paths, sharded, full_plan=full_plan)
    assert not (sharded / 'results.jsonl').exists()
    write_json(path, original)
    # No runtime, CUDA or teacher should be involved in the merge.
    monkeypatch.setattr(p0_eval, 'HarnessEvaluation', lambda *a: pytest.fail('runtime during merge'))
    assert merge_evaluation(pairs, paths, sharded, full_plan=full_plan) == expected
    for name in ('report.json', 'report.md', 'results.jsonl'):
        assert (sharded / name).read_bytes() == (plain / name).read_bytes()
    with pytest.raises(FileExistsError):
        merge_evaluation(pairs, paths, sharded, full_plan=full_plan)


@pytest.mark.parametrize('full_plan', [False, True])
def test_shards_and_merge_reject_mixed_plans_before_runtime(tmp_path, full_plan):
    student, pairs, paths, _ = make_freezes(tmp_path)
    directory = tmp_path / 'evaluation'
    fake_evaluate(student, pairs, paths, directory, shard='0/2', full_plan=full_plan)
    with pytest.raises(ValueError, match='freeze differs'):
        fake_evaluate(student, pairs, paths, directory, shard='1/2', full_plan=not full_plan)
    assert not (directory / 'shards/0001-of-0002').exists()
    with pytest.raises(ValueError, match='freeze mismatch'):
        merge_evaluation(pairs, paths, directory, full_plan=not full_plan)
    plan_path = directory / 'work_plan.json'
    plan = json.loads(plan_path.read_text())
    plan['plan'] = 'prereg_minimal' if full_plan else 'full'
    write_json(plan_path, plan)
    with pytest.raises(ValueError, match='freeze mismatch'):
        merge_evaluation(pairs, paths, directory, full_plan=full_plan)


def test_shared_none_unknown_k20_prevents_complete_criterion(tmp_path):
    student, pairs, paths, _ = make_freezes(tmp_path)
    directory = tmp_path / 'evaluation'
    fake_evaluate(student, pairs, paths, directory, shard='0/1')
    plan = json.loads((directory / 'work_plan.json').read_text())
    baseline = next(u for u in plan['units'] if u['kind'] == 'k20' and u['condition'] == 'none')
    path = directory / 'shards/0000-of-0001/result.json'
    result = json.loads(path.read_text())
    next(r for r in result['records'] if r['unit_id'] == digest(baseline))['result']['success'] = 'unknown'
    write_json(path, result)
    report = merge_evaluation(pairs, paths, directory)
    assert report['complete'] is False and report['preregistered_numeric_criterion'] is False
    assert all(report['k20'][c]['none']['rate']['unknown'] == 1 for c in CATEGORIES)


def test_submission_gpu_can_differ_but_checkpoint_cannot(tmp_path):
    student, pairs, paths, _ = make_freezes(tmp_path)
    identity = dict(student.identity, runtime_device=dict(physical_gpu_index=3, device_name='fake'))
    evaluation_barrier(tmp_path / 'eval', pairs, paths, identity)
    with pytest.raises(ValueError, match='student differs'):
        evaluation_barrier(tmp_path / 'bad', pairs, paths, dict(identity, revision='changed'))
