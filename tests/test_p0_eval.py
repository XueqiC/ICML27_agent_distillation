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
from repair.p0_introspect import ToolSession, write_receipt
from repair.p0_pairs import CATEGORIES
from repair.records import TrajectoryRecord
from sed.storage import digest, write_json, file_hash
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
        self.directory = directory
        self.collateral = [dict(pair_id=f'collateral-{i}') for i in range(200)]
        self.seed_ids = [f'train/seed/{i}' for i in range(20)]
        self.seed_sources = {key: {'fixture': True} for key in self.seed_ids}

    def pair(self, pair, edit):
        good = edit is not None and edit['alpha'] == .5
        execution = TrajectoryRecord(task_id=pair['task_id'], initial=None, steps=[],
                                     success=good, termination_reason='SUCCESS' if good else 'FAILED')
        pairdir = self.directory / 'pairs' / pair['pair_id']
        path = pairdir / 'executions' / (execution.trajectory_id + '.json')
        write_receipt(path, execution.to_json())
        data = dict(success=good, termination=execution.termination_reason, environment_steps=0,
                    execution_sha256=file_hash(path))
        key = digest(dict(pair=pair['pair_id'], edit=edit))
        rollout = dict(tool='rollout', framing='free_greedy', data=data, sha256=digest(data),
                       cache_key=key, cache_hit=False, summary='fixture')
        write_receipt(pairdir / 'cache' / (key + '.json'), rollout)
        return dict(success=good, greedy='inventory' if good else 'look', decision_flip=good,
                    reproduction_matches=pair['pair_id'].endswith('4') if edit is None else None, rollout=rollout)

    def collateral_decision(self, pair_id, edit):
        return dict(pair_id=pair_id, edited='look')

    def k20_task(self, task_id, edit):
        return self.seed_result(task_id, True)

    def seed_result(self, task_id, success):
        execution = TrajectoryRecord(task_id=task_id, initial=None, steps=[],
            success=success, termination_reason='SUCCESS' if success is True else 'FAILED', environment_steps=2)
        path = self.directory / 'k20_executions' / (execution.trajectory_id + '.json')
        write_receipt(path, execution.to_json())
        return dict(task_id=task_id, success=success, termination=execution.termination_reason,
                    environment_steps=2, execution_sha256=file_hash(path))


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
            return self.seed_result(task_id, edit is None)

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
    class Runtime(FakeRuntime):
        def k20_task(self, task_id, edit):
            return self.seed_result(task_id, 'unknown' if edit is None and task_id == self.seed_ids[0] else True)
    evaluate(student, pairs, paths, directory, runtime_factory=Runtime, edits_factory=fake_edits, shard='0/1')
    report = merge_evaluation(pairs, paths, directory)
    assert report['complete'] is False and report['preregistered_numeric_criterion'] is False
    assert all(report['k20'][c]['none']['rate']['unknown'] == 1 for c in CATEGORIES)


def test_submission_gpu_can_differ_but_checkpoint_cannot(tmp_path):
    student, pairs, paths, _ = make_freezes(tmp_path)
    identity = dict(student.identity, runtime_device=dict(physical_gpu_index=3, device_name='fake'))
    evaluation_barrier(tmp_path / 'eval', pairs, paths, identity)
    with pytest.raises(ValueError, match='student differs'):
        evaluation_barrier(tmp_path / 'bad', pairs, paths, dict(identity, revision='changed'))


def test_k20_identical_trajectories_reuse_receipt_and_reject_changed_bytes(tmp_path, monkeypatch):
    from repair import execute
    task_id = 'train/group/trial'
    game = tmp_path / task_id / 'game.tw-pddl'
    game.parent.mkdir(parents=True)
    game.write_text('fixture')
    runtime = HarnessEvaluation.__new__(HarnessEvaluation)
    runtime.directory = tmp_path / 'run'
    runtime.seed_sources = {task_id: dict(game_path=str(game), game_sha256=file_hash(game), environment_seed=0)}
    runtime.student = SimpleNamespace(edited=lambda e: nullcontext())
    execution = TrajectoryRecord(task_id=task_id, initial=None, steps=[], success=True, termination_reason='SUCCESS')
    calls = []
    monkeypatch.setattr(execute, 'execute_actions', lambda *a, **kw: calls.append(kw) or execution)
    first = runtime.k20_task(task_id, None)
    path = next((runtime.directory / 'k20_executions').glob('*.json'))
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert runtime.k20_task(task_id, {'alpha': 0}) == first
    assert len(calls) == 4  # Both units really execute reset + rollout.
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    path.write_bytes(before[0] + b' ')
    with pytest.raises(ValueError, match='receipt integrity error'):
        runtime.k20_task(task_id, None)
    assert path.read_bytes() == before[0] + b' '


def test_pair_identical_trajectories_reuse_receipt_and_reject_changed_bytes(tmp_path, monkeypatch):
    from repair.dry_fakes import FakeEnvironment
    from p0_fakes import fake_greedy
    student = tiny_student()
    monkeypatch.setattr(student, 'greedy', fake_greedy)
    pair = fake_pair(split='test')
    session = ToolSession(student, [pair], tmp_path, env_factory=lambda p: FakeEnvironment(), scope='evaluation')
    state = session.state(pair['pair_id'])
    first = session.rollout(state)
    path = next((tmp_path / 'executions').glob('*.json'))
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert session.rollout(state) == first
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    path.write_bytes(before[0] + b' ')
    with pytest.raises(ValueError, match='receipt integrity error'):
        session.rollout(state)
    assert path.read_bytes() == before[0] + b' '


@pytest.mark.parametrize('different', [False, True])
@pytest.mark.parametrize('tool', ['action_scores', 'direction'])
def test_cache_and_direction_publication_verify_existing_bytes(tmp_path, monkeypatch, different, tool):
    from repair import p0_introspect
    student, pair = tiny_student(), fake_pair()
    session = ToolSession(student, [pair], tmp_path)
    original_write = p0_introspect.atomic_write
    competing = []
    def raced_write(path, content, *, exclusive):
        # Another publisher wins after the cache lookup. Never clobber it.
        if not competing:
            competing.append(path)
            original_write(path, content + (b' ' if different else b''), exclusive=True)
        return original_write(path, content, exclusive=exclusive)
    monkeypatch.setattr(p0_introspect, 'atomic_write', raced_write)
    args = (dict(state=pair['pair_id'], candidates=['look']) if tool == 'action_scores' else
            dict(positive=[dict(state=pair['pair_id'], action='look')],
                 negative=[dict(state=pair['pair_id'], action='inventory')], layers=[0]))
    if different:
        with pytest.raises(ValueError, match='receipt integrity error'):
            session.call(tool, args)
        assert competing[0].read_bytes().endswith(b'\n ')
    else:
        first = session.call(tool, args)
        assert session.call(tool, args)['data'] == first['data']
        assert competing


class InterruptedRuntime(FakeRuntime):
    remaining = 3
    executed = []

    def tick(self, kind, key, edit):
        if self.remaining == 0:
            raise RuntimeError('simulated interruption')
        type(self).remaining -= 1
        self.executed.append((kind, key, edit))

    def pair(self, pair, edit):
        self.tick('pair', pair['pair_id'], edit)
        return super().pair(pair, edit)

    def collateral_decision(self, pair_id, edit):
        self.tick('collateral', pair_id, edit)
        return super().collateral_decision(pair_id, edit)

    def k20_task(self, task_id, edit):
        self.tick('k20', task_id, edit)
        return super().k20_task(task_id, edit)


def interrupted_run(student, pairs, paths, directory, *, full_plan=False):
    InterruptedRuntime.remaining, InterruptedRuntime.executed = 3, []
    with pytest.raises(RuntimeError, match='simulated interruption'):
        evaluate(student, pairs, paths, directory, runtime_factory=InterruptedRuntime,
                 edits_factory=fake_edits, shard='0/2', full_plan=full_plan)
    return directory / 'shards/0000-of-0002'


@pytest.mark.parametrize('full_plan', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_resume_skips_completed_and_merge_equals_uninterrupted(tmp_path, monkeypatch, full_plan, legacy):
    monkeypatch.setattr('repair.p0_pairs.utc_now', lambda: '2000-01-01T00:00:00Z')
    original_edits = fake_edits
    def distinguishable_edits(*args):
        edits = original_edits(*args)
        for i, name in enumerate(CONDITIONS):
            if name not in ('none', 'teacher'):
                edits[name]['alpha'] = .01 * (i + 1)
        return edits
    # Distinct condition calls must be distinguishable even when outcomes match.
    monkeypatch.setattr(__name__ + '.fake_edits', distinguishable_edits)
    student, pairs, paths, _ = make_freezes(tmp_path)
    directory, plain = tmp_path / 'resumed', tmp_path / 'plain'
    workdir = interrupted_run(student, pairs, paths, directory, full_plan=full_plan)
    if legacy:
        for path in (workdir / 'units').glob('*.json'):
            row = json.loads(path.read_text())
            write_json(path, dict(unit_id=row['unit_id'], result=row['result']))
    saved = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (workdir / 'units').glob('*.json')}
    frozen = {p: p.read_bytes() for p in directory.glob('*.json')}
    # Finish the other shard with the original path, then resume only the failed one.
    fake_evaluate(student, pairs, paths, directory, shard='1/2', full_plan=full_plan)
    InterruptedRuntime.remaining = 100_000
    previous_calls = list(InterruptedRuntime.executed)
    evaluate(student, pairs, paths, directory, runtime_factory=InterruptedRuntime,
             edits_factory=lambda *a: pytest.fail('recomputed controls'), shard='0/2',
             full_plan=full_plan, resume=True)
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == before for p, before in saved.items())
    assert all(p.read_bytes() == before for p, before in frozen.items())
    assert not any(call in previous_calls for call in InterruptedRuntime.executed[len(previous_calls):])
    result_path = workdir / 'result.json'
    result = json.loads(result_path.read_text())
    assert len(InterruptedRuntime.executed) == len(result['records'])
    events = [json.loads(line) for line in (workdir / 'evaluation.jsonl').read_text().splitlines()]
    assert events[-2]['event'] == 'resume' and len(events[-2]['completed_unit_ids']) == 3
    assert events[-1]['skipped'] == 3 and events[-1]['executed'] == len(result['records']) - 3
    before = (result_path.read_bytes(), result_path.stat().st_mtime_ns)
    evaluate(student, pairs, paths, directory, runtime_factory=lambda *a: pytest.fail('completed shard runtime'),
             shard='0/2', full_plan=full_plan, resume=True)
    assert (result_path.read_bytes(), result_path.stat().st_mtime_ns) == before
    expected = fake_evaluate(student, pairs, paths, plain, full_plan=full_plan)
    assert merge_evaluation(pairs, paths, directory, full_plan=full_plan) == expected
    for name in ('report.json', 'report.md', 'results.jsonl'):
        assert (directory / name).read_bytes() == (plain / name).read_bytes()


@pytest.mark.parametrize('change', ['freeze', 'submissions', 'plan', 'shards', 'runtime'])
def test_resume_refuses_changed_identity_before_runtime(tmp_path, monkeypatch, change):
    from repair import p0_eval
    student, pairs, paths, _ = make_freezes(tmp_path)
    directory = tmp_path / 'run'
    workdir = interrupted_run(student, pairs, paths, directory)
    kwargs = dict(shard='0/2')
    if change == 'freeze':
        path = directory / 'evaluation_freeze.json'
        value = json.loads(path.read_text())
        value['sha256'] = 'changed'
        write_json(path, value)
    elif change == 'submissions':
        path = directory / 'evaluation_freeze.json'
        value = json.loads(path.read_text())
        value['identity']['submissions'][CATEGORIES[0]] = 'changed'
        write_json(path, value)
    elif change == 'plan':
        kwargs['full_plan'] = True
    elif change == 'shards':
        kwargs['shard'] = '0/3'
    else:
        original = p0_eval.provenance
        def changed(*a, **kw):
            value = original(*a, **kw)
            value['code_sha256']['repair/execute.py'] = 'changed'
            return value
        monkeypatch.setattr(p0_eval, 'provenance', changed)
    before = {p: p.read_bytes() for p in (workdir / 'units').glob('*.json')}
    with pytest.raises(ValueError):
        evaluate(student, pairs, paths, directory, runtime_factory=lambda *a: pytest.fail('runtime'),
                 edits_factory=lambda *a: pytest.fail('directions'), resume=True, **kwargs)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (directory / 'resume_runtime.json').exists()


@pytest.mark.parametrize('change', ['schema', 'unit_key', 'result_key', 'execution', 'cache', 'data_hash', 'missing'])
def test_resume_validates_all_completed_units_before_runtime(tmp_path, change):
    student, pairs, paths, _ = make_freezes(tmp_path)
    directory = tmp_path / 'run'
    workdir = interrupted_run(student, pairs, paths, directory)
    path = sorted((workdir / 'units').glob('*.json'))[-1]
    row = json.loads(path.read_text())
    if change == 'schema':
        row['schema'] = 'unsupported'
    elif change == 'unit_key':
        row['unit_id'] = '0' * 64
    elif change == 'result_key':
        row['result']['pair_id'] = 'different'
    elif change == 'data_hash':
        row['result']['rollout']['sha256'] = '0' * 64
    else:
        pairdir = workdir / 'pairs' / row['result']['pair_id']
        target = next((pairdir / ('cache' if change == 'cache' else 'executions')).glob('*.json'))
        if change == 'missing':
            target.unlink()
        else:
            target.write_bytes(target.read_bytes() + b' ')
    write_json(path, row)
    with pytest.raises(ValueError):
        evaluate(student, pairs, paths, directory, shard='0/2', resume=True,
                 runtime_factory=lambda *a: pytest.fail('runtime before full verification'))
    assert len(list((workdir / 'units').glob('*.json'))) == 3
    assert not (workdir / 'result.json').exists()


def test_resume_records_repaired_code_and_merge_accepts_mixed_provenance(tmp_path, monkeypatch):
    from repair import p0_eval
    student, pairs, paths, _ = make_freezes(tmp_path)
    directory = tmp_path / 'run'
    original = p0_eval.provenance
    def old_code(*a, **kw):
        value = original(*a, **kw)
        value['code_sha256']['repair/p0_eval.py'] = 'old-code'
        return value
    monkeypatch.setattr(p0_eval, 'provenance', old_code)
    workdir = interrupted_run(student, pairs, paths, directory)
    fake_evaluate(student, pairs, paths, directory, shard='1/2')
    monkeypatch.setattr(p0_eval, 'provenance', original)
    fake_evaluate(student, pairs, paths, directory, shard='0/2', resume=True)
    freeze = json.loads((directory / 'evaluation_freeze.json').read_text())
    assert freeze['provenance']['code_sha256']['repair/p0_eval.py'] == 'old-code'
    resume = next(json.loads(line) for line in (workdir / 'evaluation.jsonl').read_text().splitlines()
                  if json.loads(line)['event'] == 'resume')
    assert resume['provenance']['code_sha256'] == original()['code_sha256']
    assert merge_evaluation(pairs, paths, directory)['complete'] is True


@pytest.mark.parametrize('change', ['k20_hash', 'shard_summary', 'work_plan', 'conditions', 'dependencies'])
def test_resume_rejects_corrupted_completed_shard(tmp_path, change):
    student, pairs, paths, _ = make_freezes(tmp_path)
    directory = tmp_path / 'run'
    # A small shard that contains a K20 unit; only fake interactions execute.
    units = work_units(pairs, [f'collateral-{i}' for i in range(200)], [f'train/seed/{i}' for i in range(20)])
    index = shard_owner(next(u for u in units if u['kind'] == 'k20'), 100)
    fake_evaluate(student, pairs, paths, directory, shard=f'{index}/100')
    workdir = directory / 'shards' / f'{index:04d}-of-0100'
    if change == 'k20_hash':
        path = next((workdir / 'k20_executions').glob('*.json'))
        path.write_bytes(path.read_bytes() + b' ')
    else:
        path = (workdir / 'result.json' if change == 'shard_summary' else
                directory / {'work_plan': 'work_plan.json', 'conditions': 'conditions.json',
                             'dependencies': 'evaluation_freeze.json'}[change])
        value = json.loads(path.read_text())
        if change == 'shard_summary':
            value['records'] = []
        elif change == 'work_plan':
            value['units'].pop()
        elif change == 'conditions':
            value[CATEGORIES[0]]['directions'] = {}
        else:
            value['provenance']['dependencies']['torch'] = 'changed'
        write_json(path, value)
    with pytest.raises(ValueError):
        evaluate(student, pairs, paths, directory, shard=f'{index}/100', resume=True,
                 runtime_factory=lambda *a: pytest.fail('runtime'))


def test_resume_after_last_unit_before_shard_summary_never_runs_runtime(tmp_path, monkeypatch):
    from repair import p0_eval
    student, pairs, paths, _ = make_freezes(tmp_path)
    directory = tmp_path / 'run'
    original = p0_eval.write_json
    def interrupt_summary(path, value, **kwargs):
        if path.name == 'result.json':
            raise RuntimeError('interrupted summary')
        return original(path, value, **kwargs)
    monkeypatch.setattr(p0_eval, 'write_json', interrupt_summary)
    with pytest.raises(RuntimeError, match='interrupted summary'):
        fake_evaluate(student, pairs, paths, directory, shard='0/100')
    monkeypatch.setattr(p0_eval, 'write_json', original)
    result = evaluate(student, pairs, paths, directory, shard='0/100', resume=True,
                      runtime_factory=lambda *a: pytest.fail('runtime'))
    assert result['units'] > 0
    assert json.loads((directory / 'shards/0000-of-0100/result.json').read_text())['records']


def test_resume_requires_existing_freeze_and_shard(tmp_path):
    student, pairs, paths, _ = make_freezes(tmp_path)
    with pytest.raises(ValueError, match='explicit --shard'):
        fake_evaluate(student, pairs, paths, tmp_path / 'run', resume=True)
    with pytest.raises(ValueError, match='existing evaluation freeze'):
        fake_evaluate(student, pairs, paths, tmp_path / 'run', shard='0/2', resume=True)
    fake_evaluate(student, pairs, paths, tmp_path / 'run', shard='1/100')
    with pytest.raises(ValueError, match='existing shard directory'):
        fake_evaluate(student, pairs, paths, tmp_path / 'run', shard='0/100', resume=True)
