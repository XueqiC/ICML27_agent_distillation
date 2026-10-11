"""Offline CPU regression tests for finite (not linearized) replacements."""
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5c6 as old
import c5c6b as b
import numpy as np
import pytest


class NonlinearMeter:
    x = np.array([[1.]])

    def evaluate(self, x, category, tail, pairwise):
        assert pairwise and len(x) == len(tail)
        self.states, self.tails = x, tail
        return x[:, 0]**2+tail[:, 0, 0]


def test_actual_increment_integration_not_continuation_of_mean():
    meter = NonlinearMeter()
    increments = np.array([[0.], [2.]])
    laws = np.array([[.25, .75], [1., 0.]])
    omega = np.array([[[.1]], [[.8]], [[-.2]]])
    values, atoms, active = b.integrate_replacements(meter, increments, laws, omega, 1, 'test')
    assert active.tolist() == [0, 1]
    assert values[:, 0] == pytest.approx(7.+omega[:, 0, 0])
    assert not np.allclose(values[:, 0], (1.+laws[0]@increments[:, 0])**2+omega[:, 0, 0])
    assert values[:, 0]-values[:, 1] == pytest.approx([6., 6., 6.])
    assert atoms.shape == (3, 2)
    assert len(meter.states) == 6  # Execute each shared support atom once per tail.


def test_bernstein_known_range_survives_zero_observed_variance():
    for r in b.REPLICATES:
        z = np.zeros(r)
        bound = b.bernstein(z, 8, 2.)
        expected = 14*np.log(4*64/.05)/(3*(r-1))
        assert bound['e_stat'] == pytest.approx(expected)
        assert b.bernstein(z, 8, np.ptp(z))['e_stat'] == 0.
        assert bound['variance'] == 0.
    z = np.arange(64)/64
    assert b.bernstein(z, 8, 2.)['variance'] == pytest.approx(np.var(z, ddof=1))
    with pytest.raises(ValueError):
        b.bernstein([0.], 8, 2.)


def test_pair_certificate_threshold_eligible_and_alpha_units():
    values = np.tile([.8, .2], (256, 1))
    cell = dict(task='threshold', returns=[.9, .1], measurements={'aligned':dict(
        provider_values=values, independent_provider_values=values)})
    row = b.pair_record(cell, 'aligned', 256, 0, 1, dict(alpha=2., epsilon_sur=.1, valid=True))
    assert row['eligible'] and row['radii']['a_priori']['certified']
    radius = row['radii']['a_priori']
    assert radius['radius'] == pytest.approx(radius['e_stat']+.1)
    assert radius['task_radius'] == pytest.approx(2*radius['radius'])
    reverse = b.pair_record(cell, 'aligned', 256, 1, 0, dict(alpha=2., epsilon_sur=.1, valid=True))
    assert reverse['score'] == pytest.approx(-row['score'])
    invalid = b.pair_record(cell, 'aligned', 256, 0, 1, dict(alpha=-1., epsilon_sur=None, valid=False))
    assert not invalid['eligible'] and not invalid['radii']['a_priori']['certified']
    assert invalid['radii']['a_priori']['radius'] is None


def test_calibration_uses_finite_gold_differences_and_r_prefix():
    values = np.vstack([np.tile([.1, .3, .5], (64, 1)), np.tile([.2, .3, .4], (192, 1))])
    cell = dict(returns=np.array([.2, .6, 1.]), measurements={'short':dict(provider_values=values)})
    fit = b.calibrate({'cal':cell}, 64)
    assert fit['alpha'] == pytest.approx(2.)
    assert fit['epsilon_sur'] == pytest.approx(0., abs=1e-14)
    assert fit['gold'] == pytest.approx([-.4, -.8, -.4])
    assert b.calibrate({'cal':cell}, 256)['alpha'] == pytest.approx(3.2)


def test_variance_uses_independent_tails_and_keeps_degenerate_pairs():
    rng = np.random.default_rng(1661)
    shared = rng.normal(size=256)
    independent = rng.normal(size=(2, 256))
    z = (1+shared)-shared
    result = b.variance_comparison(z, 1+independent[0]-independent[1])
    assert result['variance_ratio'] < 1e-28
    assert result['independent_variance'] > 1
    zero = b.variance_comparison(np.zeros(256), np.zeros(256))
    assert zero['variance_ratio'] is None
    assert b.variance_summary([result, zero])['zero_independent_variance'] == 1


def test_finite_replacement_measurement_has_no_oracle_and_matches_frozen_gold(monkeypatch):
    before = old.hashes()
    manifest, records, calibration, points, menu = old.load_panel()
    laws = np.array([x['law'] for x in menu])
    def forbidden(*args, **kwargs):
        raise AssertionError('derivative/curvature oracle reached from finite replacements')
    for name in ('full_gradient', 'richardson', 'curvature_search'):
        monkeypatch.setattr(old.legacy(), name, forbidden)
    cell = b.measure_cell('warm5/0/moment', manifest, points, laws)
    repeat = b.measure_cell('warm5/0/moment', manifest, points, laws)
    assert np.array_equal(cell['omega_indices'], repeat['omega_indices'])
    independent = cell['independent_omega_indices']
    assert not np.array_equal(independent[0], independent[1])
    assert not np.array_equal(cell['omega_indices'], independent[0])
    for name in b.SURROGATES:
        m = cell['measurements'][name]
        assert m['provider_values'].shape == (256, 9)
        assert m['atom_values'] @ laws.T == pytest.approx(m['provider_values'])
        assert m['provider_values'] == pytest.approx(repeat['measurements'][name]['provider_values'])
        assert cell['tail_evaluations'][name+'_crn'] == 256*10
        assert cell['tail_evaluations'][name+'_independent'] == 256*14
    names = {x['name']:j for j,x in enumerate(menu)}
    for row in records:
        if row['cell'] == cell['cell']:
            assert cell['returns'][names[row['provider']]]-cell['returns'][0] == pytest.approx(row['gold'], abs=1e-14)
    cal = b.measure_cell(calibration[0], manifest, points, laws, calibration=True)
    cost = b.costs([cell], [cal], laws)
    assert cost['all_executed_tail_evaluations'] == 2*256*(10+14)+256*10+2*1280
    row = next(x for x in cost['by_r_surrogate'] if x['r']==64 and x['surrogate']=='short')
    assert row['measurement_total'] == 2*64*10+1280
    assert row['measurement_update_steps'] == 2*64*10+7*1280
    assert row['independent_variance_diagnostic_evaluations'] == 14*64
    assert cost['full_tail_audit_reference']['matched_panel_candidate_executions'] == 9*1016
    assert before == old.hashes()
