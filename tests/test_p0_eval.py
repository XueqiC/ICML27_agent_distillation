from copy import deepcopy
import json

import pytest

from repair.p0_agent import freeze_submission
from repair.p0_eval import (mcnemar_exact, compare, predictions, CONDITIONS, evaluation_barrier,
                            evaluate, restore_direction, control_edits)
from repair.p0_introspect import ToolSession
from repair.p0_pairs import CATEGORIES
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


def test_evaluation_end_to_end_fake_runtime_no_teacher(tmp_path):
    student, pairs, paths, frozen = make_freezes(tmp_path)
    class Runtime:
        def __init__(self, student, pairs, directory):
            # The freeze MUST precede even constructing an evaluator.
            assert (directory / 'evaluation_freeze.json').is_file()
        def pair(self, pair, edit):
            assert (tmp_path / 'evaluation/evaluation_freeze.json').is_file()
            good = edit is not None and edit['alpha'] == .5
            return dict(success=good, greedy='inventory' if good else 'look',
                        decision_flip=good,
                        reproduction_matches=pair['pair_id'].endswith('4') if edit is None else None)
        def collateral_run(self, edit):
            return dict(fraction=0.0, n=200)
        def k20_run(self, edit):
            return dict(rate=dict(value=1.0, n=20, unknown=0), rows=[])
    def edits(student, frozen, diagnosis):
        teacher = restore_direction(student, frozen)
        return {k: None if k == 'none' else dict(teacher, alpha=.5 if k == 'teacher' else .1) for k in CONDITIONS}
    report = evaluate(student, pairs, paths, tmp_path / 'evaluation', runtime_factory=Runtime, edits_factory=edits)
    assert report['teacher_calls'] == 0 and report['complete']
    assert report['pooled']['pairs'] == 10
    assert report['reproduction']['rate'] == dict(value=.5, n=10, unknown=0)
    assert report['pooled']['metrics']['teacher']['decision_flip_rate']['denominator'] == 5
    assert report['pooled']['metrics']['teacher']['rollout_success']['n'] == 10
    assert report['preregistered_numeric_criterion'] is True
    assert report['diagnosis_fidelity']['value'] == 1.0
    assert (tmp_path / 'evaluation/report.md').is_file()
    assert len((tmp_path / 'evaluation/results.jsonl').read_text().splitlines()) == 80
