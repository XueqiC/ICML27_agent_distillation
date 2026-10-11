"""C5/C6 implementation checks, offline and CPU only."""
import copy
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5c6 as c
import numpy as np
import pytest


def test_finite_law_integration_preserves_nonuniform_weights():
    inc = np.array([[1., 2.], [7., -2.], [4., 4.]])
    laws = np.array([[.25, .75, 0.], [0., 0., 1.]])
    groups = c.exact_groups(inc, laws)
    assert groups.shape == (8, 2, 2)
    assert groups[:, 0] == pytest.approx(np.tile([5.5, -1.], (8, 1)))
    assert groups[:, 1] == pytest.approx(np.tile([4., 4.], (8, 1)))
    assert np.array_equal(groups[0], groups[-1])


def test_bernstein_range_variance_and_nonreference_candidate_count():
    z = np.array([-2., -1., 0., 1., 2., 3., 4., 5.])
    actual = c.empirical_bernstein(z, candidates=8)
    logterm = np.log(4*64/.05)
    expected = np.sqrt(2*np.var(z, ddof=1)*logterm/8)+7*7*logterm/(3*7)
    assert actual['e_stat'] == pytest.approx(expected)
    assert actual['B'] == 7
    assert actual['variance'] == pytest.approx(6.)
    assert c.empirical_bernstein(np.ones(8), 8)['e_stat'] == 0
    with pytest.raises(ValueError):
        c.empirical_bernstein([1.], 8)


def test_calibration_zero_intercept_residual_units_and_rejected_alpha():
    score = np.array([-2., -1., 1., 2.])
    gold = np.array([-4.1, -1.9, 2.1, 3.9])
    fit = c.fit_alignment(score, gold)
    assert fit['alpha'] == pytest.approx(2.)
    assert fit['epsilon_sur'] == pytest.approx(.05)
    assert np.max(abs(gold-fit['alpha']*score)) <= fit['alpha']*fit['epsilon_sur']+1e-15
    assert c.fit_alignment(score, -score)['valid'] is False
    assert c.fit_alignment(np.zeros(4), gold)['valid'] is False


class QuadraticMeter:
    """Known quadratic with an additive random offset, cancelled only by CRN."""
    x = np.array([[.2, -.4, .3]])
    scale = np.array([1., .2, .2])
    linear = np.array([.3, -.2, .4])
    curvature = 1.7

    def evaluate(self, x, category, tail, pairwise):
        assert pairwise
        self.executions = len(x)
        omega = np.tile(tail[:, 0, 0], len(x)//len(tail))
        return x@self.linear+.5*self.curvature*np.sum((x*self.scale)**2, axis=1)+omega


def test_crn_second_differences_against_known_quadratic():
    meter = QuadraticMeter()
    rng = np.random.default_rng(166)
    means = rng.normal(size=(8, 3, 3))
    omega = rng.normal(scale=.01, size=(8, 7, 2))
    h = .003
    measured = c.measure_fd(meter, means, omega, h, 7, 'aligned')
    difference = means[:, 0]-means[:, 1]
    exact = difference@(meter.linear+meter.curvature*meter.x[0]*meter.scale**2)
    exact += .5*h*meter.curvature*(np.sum((means[:, 0]*meter.scale)**2, axis=1)
                                  -np.sum((means[:, 1]*meter.scale)**2, axis=1))
    z = measured['group_provider_values'][:, 0]-measured['group_provider_values'][:, 1]
    assert z == pytest.approx(exact, abs=1e-12)
    assert measured['M_fd'] == pytest.approx(meter.curvature, rel=1e-6)
    assert meter.executions == 5*8*3
    # Offset changes leave paired scores/curvature invariant up to cancellation.
    other = c.measure_fd(meter, means, omega+1., h, 7, 'aligned')
    assert other['group_provider_values'][:, 0]-other['group_provider_values'][:, 1] == pytest.approx(z, abs=1e-12)


def test_certificate_radius_uses_scaled_moments_and_alpha():
    fd = dict(h=.003, group_provider_values=np.tile([2., 0., 1.], (8, 1)),
              M_fd=2., norm2=np.tile([.2, .3, .4], (8, 1)), finite=True)
    cell = dict(task='moment', means=np.zeros((8, 3, 51)), curvature=dict(M2=4., invalid_probes=0),
                second_moments=np.array([.1, .2, .3]), measurements={'aligned':[fd]})
    m = c.pair_measurement(cell, 'aligned', 0, 0, 1, dict(alpha=2., epsilon_sur=.1, valid=True))
    assert m['e_stat'] == 0
    assert m['e_fd'] == pytest.approx(.0015)
    assert m['e_2'] == pytest.approx(.6)
    assert m['radius'] == pytest.approx(.4015)
    assert m['task_radius'] == pytest.approx(.803)
    assert m['certified']
    cell['task']='threshold'
    assert not c.pair_measurement(cell, 'aligned', 0, 0, 1, dict(alpha=2., epsilon_sur=.1, valid=True))['eligible']
    cell['task']='moment';cell['curvature']['invalid_probes']=1
    assert not c.pair_measurement(cell, 'aligned', 0, 0, 1, dict(alpha=2., epsilon_sur=.1, valid=True))['certified']


def test_h_selection_is_task_radius_only_and_ties_choose_small_h():
    options = [dict(h=.003, radius=.1, task_radius=.2, eligible=True, score=100.),
               dict(h=.001, radius=.15, task_radius=.15, eligible=True, score=-100.),
               dict(h=.01, radius=.01, task_radius=.15, eligible=True, score=1000.)]
    assert c.choose_h(options)['h'] == .001
    options[1]['score']=0
    assert c.choose_h(options)['h'] == .001


def test_sign_ties_and_unavailable_do_not_disappear():
    rows = [dict(gold=g, scores={'r':s}) for g,s in [(1.,-1.),(1.,0.),(-1.,-1.),(0.,1.),(1.,None)]]
    assert c.sign_stats(rows, 'r') == dict(rows=5, nonzero_gold=4, available=3, unavailable=1,
                                          wrong=1, ties=1, wrong_rate=1/3)


def test_measurements_need_no_oracle_and_frozen_tree_is_readonly(monkeypatch):
    before = c.hashes()
    manifest, records, calibration, points, menu = c.load_panel()
    o = c.legacy()
    def forbidden(*args, **kwargs):
        raise AssertionError('oracle gradient reached from C6')
    monkeypatch.setattr(o, 'full_gradient', forbidden)
    monkeypatch.setattr(o, 'richardson', forbidden)
    cell = c.measure_cell('warm5/0/moment', manifest, points, np.array([p['law'] for p in menu]), quick=True)
    assert cell['curvature']['M2'] > 0
    assert cell['curvature']['invalid_probes'] == 0
    assert len(calibration) == 8
    held_states = {r['state'] for r in records}
    for key in calibration:
        state = manifest['states'][key.split('/')[0]]
        for name in held_states:
            assert not np.array_equal(o.state_vector(state), o.state_vector(manifest['states'][name]))
    expected_fd = len(c.HS)*5*8*9
    assert cell['tail_evaluations']['aligned_fd_and_score'] == expected_fd
    assert cell['tail_evaluations']['short_fd_and_score'] == expected_fd
    assert cell['tail_evaluations']['full_tail_gold_audit'] == 10*128
    # Check the guard without attempting a filesystem write.
    with pytest.raises(PermissionError, match='read-only'):
        c._readonly_audit('open', (str(c.FROZEN/'untouched'), 'w', os.O_WRONLY))
    assert before == c.hashes()


def test_cost_counts_trajectory_replicas_and_shared_overhead_once():
    cell = dict(tail_evaluations={'aligned_fd_and_score':1080,'short_fd_and_score':1080,
                                 'remainder_curvature':83200,'full_tail_gold_audit':1280},
                tail_update_steps={})
    cal = dict(tail_evaluations={'short_fd_and_score':1080,'full_tail_gold_audit':1280}, tail_update_steps={})
    cost = c.cost_report([cell], [cal], 9)
    assert cost['all_executed_tail_evaluations'] == 89000
    assert cost['by_surrogate']['aligned']['measurement_total'] == 84280
    assert cost['by_surrogate']['short']['measurement_total'] == 86640
    assert cost['per_surrogate_score_endpoints_all_h'] == 216
    assert cost['per_surrogate_score_endpoints_one_h'] == 72
    assert cost['full_tail_audit_reference']['matched_panel_candidate_executions'] == 9144
    assert cost['full_tail_audit_reference']['M8_replicas_per_candidate'] == 1154
