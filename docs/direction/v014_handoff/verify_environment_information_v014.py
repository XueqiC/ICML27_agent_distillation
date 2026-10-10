#!/usr/bin/env python3
"""Finite mathematical checks for v0.14; not neural-agent training.

Checks use exact rational enumeration where feasible, independent LP primal
and dual formulations, and analytic finite-distribution expectations.
"""
import argparse
import itertools
import json
import math
import platform
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial.chebyshev import chebval
from scipy.optimize import linprog
from scipy.stats import binom

SEED = 2026101014


def checked_lp(*args, **kwargs):
    result = linprog(*args, method="highs", **kwargs)
    if not result.success:
        raise AssertionError(result.message)
    return result


def feature_width_primal(phi, g, eps):
    """Optimize over two probability laws directly, not signed dual variables."""
    m, d = phi.shape
    feature_difference = np.hstack([phi.T, -phi.T])
    eq = np.zeros((2, 2*m))
    eq[0, :m], eq[1, m:] = 1, 1
    result = checked_lp(
        np.r_[-g, g],
        A_ub=np.vstack([feature_difference, -feature_difference]),
        b_ub=np.r_[eps, eps],
        A_eq=eq, b_eq=[1, 1], bounds=(0, None),
    )
    p, q = result.x[:m], result.x[m:]
    assert np.max(np.abs((p-q)@phi)-eps) < 1e-8
    return float(-result.fun), p, q


def feature_width_dual(phi, g, eps):
    """Minimize 2*uniform residual + weighted coefficient norm."""
    m, d = phi.shape
    upper = np.column_stack([-phi, phi, -np.ones(m), -np.ones(m)])
    lower = np.column_stack([phi, -phi, np.ones(m), -np.ones(m)])
    result = checked_lp(
        np.r_[eps, eps, 0., 2.],
        A_ub=np.vstack([upper, lower]), b_ub=np.r_[-g, g],
        bounds=[(0, None)]*(2*d)+[(None, None), (0, None)],
    )
    coefficients = result.x[:d]-result.x[d:2*d]
    intercept = result.x[-2]
    residual = float(np.max(np.abs(g-intercept-phi@coefficients)))
    achieved = 2*residual+float(eps@np.abs(coefficients))
    assert abs(achieved-result.fun) < 1e-8
    return float(result.fun), coefficients, residual


def noisy_feature_duality(rng):
    simple = []
    phi = np.array([[0.], [.5], [1.]])
    g = np.array([0., 0., 1.])
    for e in [0., .05, .25, .5, .8, 1.5]:
        pvalue, p, q = feature_width_primal(phi, g, np.array([e]))
        dvalue, b, residual = feature_width_dual(phi, g, np.array([e]))
        expected = min(1., .5+e)
        assert abs(pvalue-expected) < 1e-10
        assert abs(pvalue-dvalue) < 1e-10
        simple.append({
            "tolerance": e, "primal": pvalue, "dual": dvalue,
            "closed_form": expected, "witness_P": p.tolist(),
            "witness_Q": q.tolist(), "coefficient": b.tolist(),
            "uniform_residual": residual,
        })
    errors = []
    for i in range(36):
        m, d = 7, 3
        phi = rng.random((m, d))
        if i % 3 == 0:
            phi[:, 2] = 2*phi[:, 0]  # deliberately dependent features
        eps = np.zeros(d) if i % 2 == 0 else rng.uniform(0, .2, d)
        g = rng.random(m)
        pvalue, _, _ = feature_width_primal(phi, g, eps)
        dvalue, _, _ = feature_width_dual(phi, g, eps)
        errors.append(abs(pvalue-dvalue))
        assert pvalue <= float(np.ptp(g))+1e-9
    assert max(errors) < 1e-8
    return {"explicit_example": simple, "random_LP_cases": len(errors),
            "maximum_primal_dual_residual": max(errors)}


def multiindices(k, degree):
    return [a for a in itertools.product(range(degree+1), repeat=k)
            if sum(a) <= degree]


def parity_witnesses():
    rows = []
    for n in range(1, 9):
        support = list(itertools.product([0, 1], repeat=n))
        even = [z for z in support if sum(z) % 2 == 0]
        odd = [z for z in support if sum(z) % 2 == 1]
        checked_subsets = 0
        for r in range(n):
            for subset in itertools.combinations(range(n), r):
                pm = F(sum(math.prod(z[j] for j in subset) for z in even), len(even))
                qm = F(sum(math.prod(z[j] for j in subset) for z in odd), len(odd))
                assert pm == qm
                checked_subsets += 1
        def payoff(z):
            return F(1+math.prod(1-2*x for x in z), 2)
        vp = sum(map(payoff, even))/len(even)
        vq = sum(map(payoff, odd))/len(odd)
        assert vp == 1 and vq == 0
        # Powers of binary coordinates are idempotent: proper-subset checks
        # imply all total-degree-<n raw moment equalities.
        rows.append({"N": n, "k": n, "matching_proper_subset_tests": checked_subsets,
                     "reference_value": float(vp), "alternative_value": float(vq),
                     "exact_value_gap": str(vp-vq)})
    return rows


def grid_dimensions():
    rows = []
    for k, n in [(1, 1), (1, 4), (2, 2), (2, 3), (3, 2), (3, 3)]:
        y = np.array(list(itertools.product(np.linspace(0, 1, n+1), repeat=k)))
        indices = multiindices(k, n)
        matrix = np.column_stack([np.prod(y**a, axis=1) for a in indices])
        rank = int(np.linalg.matrix_rank(matrix, tol=1e-10))
        count = math.comb(k+n, n)
        assert rank == count == len(indices)
        rows.append({"k": k, "N": n, "alphabet_size": len(y),
                     "numerical_response_rank": rank,
                     "theoretical_minimum_nonconstant_statistics": count-1})
    return rows


def scalar_approximation():
    rows = []
    grid = np.linspace(0, 1, 8001)
    for n in [4, 8, 16, 32]:
        coeff = np.array(
            [math.comb(2*n, n)/4**n] +
            [2*math.comb(2*n, n-j)/4**n for j in range(1, n+1)]
        )
        full = chebval(2*grid-1, coeff)
        assert np.max(np.abs(full-grid**n)) < 1e-12
        for q in range(n+1):
            estimate = chebval(2*grid-1, coeff[:q+1])
            tail = float(coeff[q+1:].sum())
            observed = float(np.max(np.abs(estimate-grid**n)))
            bound = min(1., 2*math.exp(-q*q/n))
            assert abs(observed-tail) < 2e-12
            assert tail <= bound+1e-12
            rows.append({"N": n, "q": q, "Chebyshev_truncation_error": tail,
                         "grid_check_error": observed, "Hoeffding_upper": bound,
                         "parity_best_degree_q_error": .5 if q < n else 0.})
    return rows


def convolve_fraction(a, b):
    result = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i+j] += x*y
    return result


def scalar_exact_decay():
    rows = []
    for degree in range(2, 7):
        p = [F(math.comb(degree, i), 2**(degree-1)) if i % 2 == 0 else F(0)
             for i in range(degree+1)]
        q = [F(math.comb(degree, i), 2**(degree-1)) if i % 2 else F(0)
             for i in range(degree+1)]
        delta = F((-1)**degree*math.factorial(degree),
                  2**(degree-1)*degree**degree)
        for power in range(degree):
            pm = sum(w*F(i, degree)**power for i, w in enumerate(p))
            qm = sum(w*F(i, degree)**power for i, w in enumerate(q))
            assert pm == qm
        for blocks in [1, 2, 4, 8]:
            ps, qs = [F(1)], [F(1)]
            for _ in range(blocks):
                ps, qs = convolve_fraction(ps, p), convolve_fraction(qs, q)
            vp = sum(w*F(i, degree*blocks)**degree for i, w in enumerate(ps))
            vq = sum(w*F(i, degree*blocks)**degree for i, w in enumerate(qs))
            predicted = delta/blocks**(degree-1)
            assert vp-vq == predicted
            rows.append({"N": degree, "independent_blocks": blocks,
                         "enumerated_gap_exact": str(vp-vq),
                         "predicted_gap_exact": str(predicted),
                         "absolute_gap": float(abs(vp-vq))})
    return rows


def coupling_extrema(mu, nu, response):
    m, n = len(mu), len(nu)
    eq = []
    for i in range(m):
        row = np.zeros((m, n))
        row[i, :] = 1
        eq.append(row.ravel())
    for j in range(n):
        row = np.zeros((m, n))
        row[:, j] = 1
        eq.append(row.ravel())
    constraints = dict(A_eq=np.array(eq), b_eq=np.r_[mu, nu], bounds=(0, None))
    low = checked_lp(response.ravel(), **constraints)
    high = checked_lp(-response.ravel(), **constraints)
    return float(low.fun), float(-high.fun)


def coupling_and_null(rng):
    mu, nu = np.array([.2, .3, .5]), np.array([.4, .6])
    additive = np.array([.1, .2, .3])[:, None]+np.array([.1, .4])[None, :]
    lo, hi = coupling_extrema(mu, nu, additive)
    assert abs(hi-lo) < 1e-12
    nonlinear = np.array([[0., .25], [.25, 1.]])
    nlo, nhi = coupling_extrema(np.array([.5, .5]), np.array([.5, .5]), nonlinear)
    assert abs(nlo-.25) < 1e-12 and abs(nhi-.5) < 1e-12
    grads = rng.normal(size=(4, 11))
    theta = rng.normal(size=11)
    updates = []
    # Strict null: every per-example gradient is frozen at the same theta.
    for order in itertools.permutations(range(4)):
        updates.append(theta-.1*sum((grads[i] for i in order), np.zeros(11))/4)
    null_error = max(float(np.max(np.abs(u-updates[0]))) for u in updates)
    assert null_error < 1e-14
    return {"additive_width": hi-lo, "quadratic_pair_min": nlo,
            "quadratic_pair_max": nhi, "quadratic_pair_width": nhi-nlo,
            "strict_frozen_gradient_additive_null_max_residual": null_error,
            "scope": "Finite packet LP and frozen arithmetic; no neural optimizer experiment."}


def adaptive_gate_expectations():
    """Exact one audited stage reached through a random reference first stage.

    Candidate terminal value is Bernoulli(q); reference value is deterministic p.
    Audits use the same correct marginals, and actual update is fresh.
    """
    n, delta = 400, .05
    radius = math.sqrt(2*math.log(2/delta)/n)
    states = [(.2, .6), (.8, .7)]
    rows = []
    expected_gain = expected_lcb = expected_failure = 0.
    for p, q in states:
        k = np.arange(n+1)
        mass = binom.pmf(k, n, q)
        mass /= mass.sum()  # remove accumulated PMF roundoff
        empirical = k/n-p
        lcb = np.maximum(0, empirical-radius)
        selected = empirical > radius
        selection_probability = float(np.clip(mass@selected, 0., 1.))
        gain = selection_probability*(q-p)
        bmean = float(mass@lcb)
        fail = float(mass@(np.abs(k/n-q) > radius))
        expected_gain += gain/2
        expected_lcb += bmean/2
        expected_failure += fail/2
        rows.append({"reference_value": p, "candidate_value": q,
                     "selection_probability": selection_probability,
                     "expected_gain": gain, "expected_selected_LCB": bmean,
                     "audit_failure_probability": fail})
    assert expected_failure <= delta+1e-12
    assert expected_gain >= -delta-1e-12
    assert expected_gain >= expected_lcb-2*delta-1e-12
    # A finite optimistic gate can admit an inferior provider with positive
    # probability; exact unconditional no-degradation is not justified.
    bad_n, bad_delta, p, q = 20, .9, .5, .49
    bad_e = math.sqrt(2*math.log(2/bad_delta)/bad_n)
    minimum_k = math.floor(bad_n*(p+bad_e))+1
    false_selection = float(binom.sf(minimum_k-1, bad_n, q))
    harm = false_selection*(q-p)
    assert false_selection > 0 and harm < 0
    return {
        "scope": "Exact finite-distribution expectations, not an empirical coverage study.",
        "n_per_candidate": n, "delta": delta, "radius": radius, "states": rows,
        "unconditional_expected_gain": expected_gain,
        "expected_sum_selected_LCB": expected_lcb,
        "lower_bound_minus_delta": -delta,
        "lower_bound_LCB_minus_2delta": expected_lcb-2*delta,
        "failure_probability": expected_failure,
        "no_exact_nondegradation_counterexample": {
            "n": bad_n, "delta": bad_delta, "reference": p, "candidate": q,
            "false_selection_probability": false_selection,
            "strictly_negative_expected_gain": harm,
        },
        "gold_cost_example": {
            "M": 4, "delta": .05, "target_radius": .1,
            "paired_replicas_per_candidate": math.ceil(2*math.log(160)/.01),
            "total_pairs_without_reuse": 4*math.ceil(2*math.log(160)/.01),
        },
    }


def robust_ambiguity_minimax():
    # Two worlds prefer opposite actions, with regret gap 0.2.
    regret = np.array([[0., .2], [.2, 0.]])
    # Primal: min threshold t over a policy mixture.
    primal = checked_lp([0., 0., 1.],
                        A_ub=np.column_stack([regret, -np.ones(2)]),
                        b_ub=np.zeros(2),
                        A_eq=[[1., 1., 0.]], b_eq=[1.],
                        bounds=[(0, None), (0, None), (None, None)])
    # Dual: maximize threshold under an environment mixture.
    dual = checked_lp([0., 0., -1.],
                      A_ub=np.column_stack([-regret.T, np.ones(2)]),
                      b_ub=np.zeros(2),
                      A_eq=[[1., 1., 0.]], b_eq=[1.],
                      bounds=[(0, None), (0, None), (None, None)])
    assert abs(primal.fun+.0-.1) < 1e-12
    assert abs(primal.fun+dual.fun) < 1e-12
    return {"primal_regret": float(primal.fun), "dual_regret": float(-dual.fun),
            "policy_mixture": primal.x[:2].tolist(),
            "environment_mixture": dual.x[:2].tolist(),
            "scope": "Mixture-averaged regret; not each sampled policy's regret."}


def plot_results(results, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    selected = [x for x in results["scalar_approximation"] if x["N"] == 16]
    q = np.array([x["q"] for x in selected])
    axes[0].plot(q, [x["Chebyshev_truncation_error"] for x in selected],
                 "o-", color="#2463A0", label="Scalar power: explicit upper bound")
    axes[0].plot(q, [x["parity_best_degree_q_error"] for x in selected],
                 "s-", color="#B55434", label="16-context parity: exact optimum")
    axes[0].set(xlabel="Retained polynomial degree q", ylabel="Uniform approximation error",
                title="Same exact degree, different compressibility")
    axes[0].legend(frameon=False, fontsize=8)

    for degree in [2, 3, 4, 5]:
        rows = [x for x in results["scalar_exact_decay"] if x["N"] == degree]
        axes[1].loglog([x["independent_blocks"] for x in rows],
                       [x["absolute_gap"] for x in rows], "o-", label=f"N={degree}: slope {1-degree}")
    axes[1].set(xlabel="Independent blocks n", ylabel="Absolute learning-value gap",
                title="Exact rational enumeration of decay")
    axes[1].xaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].legend(frameon=False, fontsize=8)

    simple = results["noisy_feature_duality"]["explicit_example"]
    e = [x["tolerance"] for x in simple]
    axes[2].plot(e, [x["primal"] for x in simple], "o-", color="#2463A0",
                 label="Worst pair of feedback laws (LP)")
    axes[2].plot(e, [x["dual"] for x in simple], "x", color="#B55434", markersize=8,
                 label="Stability + approximation (dual LP)")
    axes[2].set(xlabel="Allowed feature-mean error", ylabel="Worst response discrepancy",
                title="Noisy-feature operational duality")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.06, right=.99, bottom=.24, top=.84, wspace=.3)
    fig.text(.5, .06, "Finite mathematical constructions and LP checks. No neural-agent training results.",
             ha="center", fontsize=10, color="#555555")
    names = []
    for suffix in ["png", "svg"]:
        name = f"theory_checks_v014.{suffix}"
        fig.savefig(directory/name, dpi=180, facecolor="white")
        names.append(name)
    plt.close(fig)
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results_v014.json"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {
        "version": "0.14", "seed": SEED,
        "scope": "Finite checks only. No neural training, teacher calls, or external agent runs.",
        "runtime": {"python": platform.python_version(),
                    "numpy": np.__version__, "scipy": scipy.__version__},
        "noisy_feature_duality": noisy_feature_duality(rng),
        "constant_gap_parity": parity_witnesses(),
        "grid_response_dimensions": grid_dimensions(),
        "scalar_approximation": scalar_approximation(),
        "scalar_exact_decay": scalar_exact_decay(),
        "coupling_and_additive_null": coupling_and_null(rng),
        "adaptive_gate_expectations": adaptive_gate_expectations(),
        "robust_ambiguity_minimax": robust_ambiguity_minimax(),
    }
    if args.plot:
        results["figures"] = plot_results(results, args.output.parent)
    results["all_assertions_passed"] = True
    args.output.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({
        "output": str(args.output.resolve()), "all_assertions_passed": True,
        "primal_dual_residual": results["noisy_feature_duality"]["maximum_primal_dual_residual"],
        "parity_orders_checked": [r["N"] for r in results["constant_gap_parity"]],
        "exact_decay_cases": len(results["scalar_exact_decay"]),
        "neural_training_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
