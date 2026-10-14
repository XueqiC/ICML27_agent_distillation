#!/usr/bin/env python3
"""CPU-only checks of the v0.17 additions (response operator, predictive closure,
reuse and estimation cost, compilation, Adam memory) and record checks of the
real-model runs R0, R1, R3 and the noise reference.

Synthetic checks verify derivations; they are not evidence about neural networks.
Record checks recompute the reported numbers from the stored run outputs in
records/ (copied from hpg reports/runs/r03 on 2026-10-14).
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
from fractions import Fraction
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy import optimize

HERE = Path(__file__).resolve().parent
RECORDS = HERE / "records"
SEED = 20261014
CHECKS = []


def check(code, item, claim):
    def wrap(fn):
        CHECKS.append((code, item, claim, fn))
        return fn
    return wrap


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rng_for(tag):
    return np.random.default_rng(np.random.SeedSequence([SEED, sum(map(ord, tag))]))


def err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def weighted_operator(G, w, q):
    """Matrix of the centred response operator in orthonormal coordinates x = sqrt(q) h."""
    m = len(q)
    Pi = np.eye(m) - np.outer(np.sqrt(q), np.sqrt(q))
    return (np.sqrt(w)[:, None] * G * np.sqrt(q)[None, :]) @ Pi, Pi


# ----------------------------------------------------------------- section 3
@check("C1.joint", "3.4", "joint decomposition with an approximate value is exact; "
       "the absolute bound holds; the stopped (coverage) version holds")
def c_joint():
    horizon, z0 = 4, (.2, .03, .08, 2)

    def update(z, y):
        theta, mom, vel, count = z
        grad = theta + .2 * theta ** 3 - y
        mom = .7 * mom + .3 * grad
        vel = .85 * vel + .15 * grad ** 2
        count += 1
        mh, vh = mom / (1 - .7 ** count), vel / (1 - .85 ** count)
        theta = theta - .17 * mh / (math.sqrt(vh) + .01)
        return (float(theta), float(mom), float(vel), count)

    def prob(z, kind):
        return float(sigmoid((.6 if kind == 0 else -.4) + .5 * z[0] - .2 * z[1]))

    def terminal(z):
        return float(sigmoid(1.3 * z[0] - .25 * z[0] ** 2))

    @lru_cache(None)
    def V(t, z):
        if t == horizon:
            return terminal(z)
        p = prob(z, 0)
        return (1 - p) * V(t + 1, update(z, -1)) + p * V(t + 1, update(z, 1))

    def expect(t, z, kind, fun):
        p = prob(z, kind)
        return (1 - p) * fun(t + 1, update(z, -1)) + p * fun(t + 1, update(z, 1))

    def Vhat(t, z):
        return terminal(z) if t == horizon else float(sigmoid(.15 + t * .03 + .4 * z[0]))

    def paths(kind):
        out = [(z0, 1.0, ())]
        levels = [out]
        for t in range(horizon):
            nxt = []
            for z, mass, hist in levels[-1]:
                p = prob(z, kind)
                nxt += [(update(z, -1), mass * (1 - p), hist + (z,)), (update(z, 1), mass * p, hist + (z,))]
            levels.append(nxt)
        return levels

    d0, d1 = paths(0), paths(1)
    L = lambda d: sum(mass * terminal(z) for z, mass, _ in d[horizon])
    gap = L(d0) - L(d1)
    resp = sum(mass * (expect(t, z, 0, Vhat) - expect(t, z, 1, Vhat)) for t in range(horizon) for z, mass, _ in d1[t])
    r = lambda t, z: expect(t, z, 0, Vhat) - Vhat(t, z)
    r0 = sum(mass * r(t, z) for t in range(horizon) for z, mass, _ in d0[t])
    r1 = sum(mass * r(t, z) for t in range(horizon) for z, mass, _ in d1[t])
    joint = resp + r0 - r1
    absolute = sum(mass * abs(expect(t, z, 0, Vhat) - expect(t, z, 1, Vhat)) for t in range(horizon) for z, mass, _ in d1[t])
    absolute += sum(mass * abs(r(t, z)) for d in (d0, d1) for t in range(horizon) for z, mass, _ in d[t])
    # Stopped version: certified set theta <= median of the round-2 parameters (so some paths exit);
    # the exit probability is paid once per protocol.
    cut = max(z0[0], float(np.quantile([z[0] for d in (d0, d1) for z, _, _ in d[3]], .6)))
    cert = lambda z: z[0] <= cut

    def stopped(d, kind, with_response):
        total, exit_mass = 0.0, 0.0
        for z, mass, hist in d[horizon]:
            states = list(hist)
            tau = next((t for t, s in enumerate(states) if not cert(s)), horizon)
            if tau < horizon:
                exit_mass += mass
            # Leaf mass split over time steps: each leaf carries its own path.
        # time-wise sums over nodes before exit
        for t in range(horizon):
            for z, mass, hist in d[t]:
                states = list(hist) + [z]
                if all(cert(s) for s in states):
                    total += mass * abs(r(t, z))
                    if with_response:
                        total += mass * abs(expect(t, z, 0, Vhat) - expect(t, z, 1, Vhat))
        term = sum(mass * abs(terminal(z) - Vhat(horizon, z)) for z, mass, hist in d[horizon]
                   if all(cert(s) for s in hist))
        return total + term + exit_mass, exit_mass

    s0, e0 = stopped(d0, 0, False)
    s1, e1 = stopped(d1, 1, True)
    bound33 = s0 + s1
    ok = (abs(gap - joint) < 1e-12 and abs(gap) <= absolute + 1e-12 and abs(gap) <= bound33 + 1e-12
          and e0 > 0 and e1 > 0)
    return ok, {"return_gap": gap, "certified_cut": cut, "joint_identity_error": abs(gap - joint), "absolute_bound": absolute,
                "stopped_bound": bound33, "exit_probability_reference": e0, "exit_probability_alternative": e1,
                "signed_response_term": resp, "signed_residual_difference": r0 - r1}


# ----------------------------------------------------------------- section 4
@check("C2.spectrum", "4.9", "weighted centred spectral identity; HS^2 <= 1/4; best rank-r residual = sigma_{r+1}")
def c_spectrum():
    rng = rng_for("spectrum")
    p, m, r = 7, 11, 3
    w = rng.uniform(.3, 1.3, p); w /= w.sum()
    q = rng.uniform(.3, 1.3, m); q /= q.sum()
    G = sigmoid(rng.normal(size=(p, m)))
    R, Pi = weighted_operator(G, w, q)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    x = Pi @ rng.normal(size=m); x /= np.linalg.norm(x)
    e1 = abs(np.linalg.norm(R @ x) ** 2 - np.sum(s ** 2 * (Vt @ x) ** 2))
    Rr = (U[:, :r] * s[:r]) @ Vt[:r]
    e2 = abs(np.linalg.norm(R - Rr, 2) - s[r])
    hs = float(np.linalg.norm(R, "fro") ** 2)
    # P - Q with L2 densities: R h equals the weighted score difference
    P, Q = rng.dirichlet(np.ones(m)), rng.dirichlet(np.ones(m))
    h = P / q - Q / q
    e3 = err(R @ (np.sqrt(q) * h), np.sqrt(w) * (G @ (P - Q)))
    ok = e1 < 1e-12 and e2 < 1e-12 and hs <= .25 + 1e-14 and e3 < 1e-12
    return ok, {"spectral_identity_error": e1, "best_rank_residual_error": e2, "hilbert_schmidt_squared": hs,
                "density_difference_error": e3, "singular_values": s.tolist()}


@check("C3.positive", "4.9", "r linear features cannot separate two positive laws whose response gap is >= a*sigma_{r+1}")
def c_positive():
    rng = rng_for("positive")
    p, m, r = 7, 11, 3
    w = rng.uniform(.3, 1.3, p); w /= w.sum()
    q = rng.uniform(.3, 1.3, m); q /= q.sum()
    G = sigmoid(rng.normal(size=(p, m)))
    R, _ = weighted_operator(G, w, q)
    _, s, Vt = np.linalg.svd(R, full_matrices=False)
    features = rng.normal(size=(r, m))
    top = Vt[:r + 1].T
    B = (features * np.sqrt(q)[None, :]) @ top
    v = top @ np.linalg.svd(B, full_matrices=True)[2][-1]
    h = v / np.sqrt(q)
    a = 1.8 / np.max(np.abs(h))
    P, Q = q * (1 + a * h / 2), q * (1 - a * h / 2)
    fe = err(features @ P, features @ Q)
    gap = float(np.linalg.norm(np.sqrt(w) * (G @ (P - Q))))
    ok = min(P.min(), Q.min()) > 0 and abs(P.sum() - 1) < 1e-12 and fe < 1e-12 and gap >= a * s[r] - 1e-13
    return ok, {"feature_match_error": fe, "response_gap": gap, "guaranteed_gap": float(a * s[r]),
                "smallest_probability": float(min(P.min(), Q.min()))}


@check("C4.generic", "4.9", "the generic dimension bound is attained-scale vacuous under the J-candidate mixture measure")
def c_generic():
    rows = []
    for J in (2, 5, 20, 100):
        # J point-mass candidates on distinct messages; rho = their mixture (uniform).
        q = np.full(J, 1 / J)
        h = np.zeros(J); h[0], h[1] = J, -J  # density difference of candidates 1 and 2
        B2 = float(np.sum(q * h ** 2))
        for eps in (0.1, 0.3, 0.7):
            need = B2 / (4 * eps ** 2)
            rows.append({"J": J, "B_squared": B2, "eps": eps, "sufficient_dim": need,
                         "input_dim": J - 1, "vacuous": bool(need > J - 1)})
    ok = all(r["B_squared"] == 2 * r["J"] for r in rows) and all(r["vacuous"] for r in rows if r["eps"] < 1 / math.sqrt(2))
    # A structured response can compress far better than the generic bound says.
    rng = rng_for("generic")
    m, p = 40, 30
    grid = np.linspace(0, 1, m)
    G = sigmoid(np.array([3 * np.sin(2 * np.pi * (k / p) + 2 * grid) for k in range(p)]))
    w, q = np.full(p, 1 / p), np.full(m, 1 / m)
    R, _ = weighted_operator(G, w, q)
    s = np.linalg.svd(R, compute_uv=False)
    return ok, {"rows": rows, "smooth_example_singular_values": s[:6].tolist(),
                "smooth_example_rank_tol_1e-10": int(np.sum(s > 1e-10))}


@check("C5.realization", "4.10", "finite Hankel rank gives an exact linear predictive realization; the conditioned update is Bayes")
def c_realization():
    rng = rng_for("realization")
    n, controls, obs, outputs = 4, 2, 2, 2
    T = {}
    for b in range(controls):
        raw = rng.uniform(.1, 1, size=(obs, n, n))
        raw /= raw.sum(axis=(0, 2), keepdims=True)
        for o in range(obs):
            T[(b, o)] = raw[o]  # T[b,o][i,j] = P(o, next=j | i, b)
    K = {b: sum(T[(b, o)] for o in range(obs)) for b in range(controls)}
    F = rng.uniform(0, 1, size=(n, outputs))
    mu0 = rng.dirichlet(np.ones(n))

    def words(L):
        return [w for l in range(L + 1) for w in itertools.product(range(controls), repeat=l)]

    def Kw(w):
        M = np.eye(n)
        for b in w:
            M = M @ K[b]
        return M

    prefixes, suffixes = words(3), words(3)
    H = np.array([[mu0 @ Kw(p) @ Kw(s) @ F[:, j] for s in suffixes for j in range(outputs)] for p in prefixes])
    H1 = np.column_stack([np.ones(len(prefixes)), H])
    rank = int(np.linalg.matrix_rank(H1, tol=1e-10))
    # Realization from the row space: q(p) = coordinates of row p.
    U, s, Vt = np.linalg.svd(H1, full_matrices=False)
    basis = Vt[:rank]
    coords = H1 @ basis.T
    A = {}
    idx = {p: i for i, p in enumerate(prefixes)}
    for b in range(controls):
        src = [i for p, i in idx.items() if len(p) < 3]
        dst = [idx[p + (b,)] for p, i in idx.items() if len(p) < 3]
        A[b] = np.linalg.lstsq(coords[src], coords[dst], rcond=None)[0].T
    pred_err = 0.0
    for w in words(5):
        if len(w) <= 3:
            continue
        qv = coords[idx[()]]
        for b in w:
            qv = A[b] @ qv
        row = qv @ basis
        truth = np.concatenate([[1.0], [mu0 @ Kw(w) @ Kw(s) @ F[:, j] for s in suffixes for j in range(outputs)]])
        pred_err = max(pred_err, err(row, truth))
    # PSR conditioning: q' = M q / e1^T M q equals the Bayes posterior predictive state.
    psi = np.column_stack([np.ones(n), F])  # tests closed? use full state indicator basis instead
    belief = mu0
    b, o = 1, 0
    joint = belief @ T[(b, o)]
    post = joint / joint.sum()
    M_bo = T[(b, o)]
    q_state = belief  # in the state basis the predictive vector is the belief itself
    q_next = (q_state @ M_bo) / (q_state @ M_bo).sum()
    bayes_err = err(q_next, post)
    ok = rank <= n and pred_err < 1e-9 and bayes_err < 1e-14
    return ok, {"hankel_rank_with_constant": rank, "hidden_states": n, "max_prediction_error_longer_words": pred_err,
                "conditioning_vs_bayes_error": bayes_err}


@check("C6.closure", "4.11", "a statically sufficient feature set need not be closed; the finite-horizon closure recursion is exact")
def c_closure():
    states = [(s0, b) for s0 in (0, 1) for b in (0, 1)]
    K = np.zeros((4, 4))
    for i, (s0, b) in enumerate(states):
        K[i, states.index((b, b))] = 1
    F = np.array([s0 for s0, _ in states], float)
    psi = np.column_stack([np.ones(4), F])
    coeff = np.linalg.lstsq(psi, K @ psi, rcond=None)[0]
    defect = float(np.max(np.abs(K @ psi - psi @ coeff)))
    closed_rank = int(np.linalg.matrix_rank(np.column_stack([psi, K @ F])))
    future_gap = float((K @ F)[1] - (K @ F)[0])
    # Exact error recursion for an approximately closed linear model.
    rng = rng_for("closure")
    n, d, H = 6, 3, 5
    Ks = [rng.dirichlet(np.ones(n), size=n) for _ in range(H)]
    psi2 = np.column_stack([np.ones(n), rng.normal(size=(n, d - 1))])
    As, res = [], []
    for k in Ks:
        Ak = np.linalg.lstsq(psi2, k @ psi2, rcond=None)[0].T
        As.append(Ak); res.append(k @ psi2 - psi2 @ Ak.T)
    mu = rng.dirichlet(np.ones(n))
    q_true, mus = [mu @ psi2], [mu]
    for k in Ks:
        mus.append(mus[-1] @ k); q_true.append(mus[-1] @ psi2)
    qhat = q_true[0] + rng.normal(scale=.01, size=d)
    q0err = q_true[0] - qhat
    for Ak in As:
        qhat = Ak @ qhat
    def prod(a, b):
        M = np.eye(d)
        for Ak in As[a:b]:
            M = Ak @ M
        return M
    rhs = prod(0, H) @ q0err + sum(prod(t + 1, H) @ (mus[t] @ res[t]) for t in range(H))
    rec_err = err(q_true[H] - qhat, rhs)
    ok = defect >= .5 - 1e-12 and closed_rank == 3 and future_gap == 1.0 and rec_err < 1e-12
    return ok, {"closure_defect": defect, "closed_test_rank": closed_rank, "future_gap_equal_current_value": future_gap,
                "recursion_identity_error": rec_err,
                "max_one_step_gain": float(max(np.linalg.norm(a, 2) for a in As))}


@check("C7.bridge", "4.12", "R = C T and ker R = ker T when C has full column rank; the residual version adds B*eps_pred")
def c_bridge():
    rng = rng_for("bridge")
    m, d, p = 9, 3, 6
    rho = rng.dirichlet(np.ones(m))
    Gamma = rng.normal(size=(d, m))
    C = rng.normal(size=(p, d))
    k = rng.normal(size=p)
    g = k[:, None] + C @ Gamma  # responses g_a(y) = k_a + c_a^T Gamma(y)
    Tm = (Gamma - (Gamma @ rho)[:, None]) * np.sqrt(rho)
    Rm = (g - (g @ rho)[:, None]) * np.sqrt(rho)
    fact = err(Rm, C @ Tm)
    null_T = np.linalg.svd(Tm, full_matrices=True)[2][d:].T
    null_R = np.linalg.svd(Rm, full_matrices=True)[2][np.linalg.matrix_rank(Rm):].T
    e_ker = float(np.linalg.norm(Rm @ null_T)) + float(np.linalg.norm(Tm @ null_R))
    # residual version: g = k + C Gamma + e, ||R h - C T h|| <= ||h|| * ||e||_{L2(omega x rho)}
    e = .05 * rng.normal(size=(p, m))
    Re = ((g + e) - ((g + e) @ rho)[:, None]) * np.sqrt(rho)
    x = rng.normal(size=m); x -= np.sqrt(rho) * (np.sqrt(rho) @ x); x /= np.linalg.norm(x)
    lhs = float(np.linalg.norm(Re @ x - C @ Tm @ x) / math.sqrt(p))
    eps_pred = float(math.sqrt(np.mean(e ** 2 @ rho)))
    ok = fact < 1e-12 and e_ker < 1e-10 and lhs <= eps_pred + 1e-12
    return ok, {"factorization_error": fact, "kernel_equality_error": e_ker, "residual_lhs": lhs,
                "residual_bound": eps_pred}


@check("C8.gramian", "4.13", "finite-horizon Hankel singular values equal sqrt eig(Wc^1/2 Wo Wc^1/2), without contraction")
def c_gramian():
    N, split = 5, 2
    A = [np.array([[1.15 + .03 * t, .12], [.04, .75]]) for t in range(N)]
    B = [np.array([[1.], [(-1) ** t * .4]]) for t in range(N)]
    C = [np.array([[1., .25 + .06 * t]]) for t in range(N + 1)]
    cols = []
    for j in range(split):
        col = B[j]
        for k in range(j + 1, split):
            col = A[k] @ col
        cols.append(col)
    reach = np.hstack(cols)
    rows, tr = [], np.eye(2)
    for t in range(split, N + 1):
        rows.append(C[t] @ tr)
        if t < N:
            tr = A[t] @ tr
    obs = np.vstack(rows)
    H = obs @ reach
    Wc, Wo = reach @ reach.T, obs.T @ obs
    ev, vec = np.linalg.eigh(Wc)
    root = (vec * np.sqrt(np.maximum(ev, 0))) @ vec.T
    gsv = np.sqrt(np.maximum(np.linalg.eigvalsh(root @ Wo @ root), 0))[::-1]
    hsv = np.linalg.svd(H, compute_uv=False)
    e = err(hsv, gsv[:len(hsv)])
    gain = float(max(np.linalg.norm(a, 2) for a in A))
    return e < 1e-10 and gain > 1, {"identity_error": e, "hankel_singular_values": hsv.tolist(), "max_one_step_gain": gain}


@check("C9.rare", "4.9", "an average guarantee does not control a rare test; no compression follows from boundedness alone")
def c_rare():
    weight = 1e-6
    rms = math.sqrt(weight)
    Ru = np.eye(8) @ (np.eye(8) - np.ones((8, 8)) / 8)
    rank = int(np.linalg.matrix_rank(Ru))
    return abs(rms - 1e-3) < 1e-15 and rank == 7, {"weighted_rms_error": rms, "rare_test_error": 1.0,
                                                   "messages": 8, "centred_rank": rank}


# ----------------------------------------------------------------- section 6
@check("C10.realizable", "6.14", "a target in the convex hull of a finite legal pool is realized with support <= r+1; "
       "sparse realizations can inflate the density bound")
def c_realizable():
    rng = rng_for("realizable")
    m, r = 30, 4
    phi = rng.normal(size=(r, m))
    weights = rng.dirichlet(np.ones(m))
    target = phi @ weights
    res = optimize.linprog(np.zeros(m), A_eq=np.vstack([phi, np.ones(m)]), b_eq=np.concatenate([target, [1]]),
                           bounds=[(0, None)] * m, method="highs-ds")
    qsol = res.x
    support = int(np.sum(qsol > 1e-10))
    match = err(phi @ qsol, target)
    rho = np.full(m, 1 / m)
    B_dense = float(np.sqrt(np.sum(rho * (weights / rho - 1) ** 2)))
    B_sparse = float(np.sqrt(np.sum(rho * (qsol / rho - 1) ** 2)))
    outside = phi @ np.ones(m) / m + 10 * np.abs(phi).max() * np.ones(r)
    res2 = optimize.linprog(np.zeros(m), A_eq=np.vstack([phi, np.ones(m)]), b_eq=np.concatenate([outside, [1]]),
                            bounds=[(0, None)] * m, method="highs")
    ok = res.status == 0 and support <= r + 1 and match < 1e-9 and res2.status == 2
    return ok, {"support": support, "carathéodory_bound": r + 1, "feature_match_error": match,
                "chi_density_dense_mixture": B_dense, "chi_density_sparse_realization": B_sparse,
                "outside_hull_infeasible": res2.status == 2}


@check("C11.gaussian", "6.15", "GLS risk = tr(B I^-1) = sum of generalized eigenvalues; one-query reduction is Sherman-Morrison")
def c_gaussian():
    rng = rng_for("gaussian")
    design = rng.normal(size=(9, 4))
    I = design.T @ design + .4 * np.eye(4)
    target = rng.normal(size=(3, 4))
    W = np.diag([.2, .3, .5]); Bm = target.T @ W @ target
    inv = np.linalg.inv(I)
    e, v = np.linalg.eigh(I)
    ir = (v * (1 / np.sqrt(e))) @ v.T
    lam = np.linalg.eigvalsh(ir @ Bm @ ir)
    risk = float(np.trace(Bm @ inv))
    a = rng.normal(size=4)
    red = float((a @ inv @ Bm @ inv @ a) / (1 + a @ inv @ a))
    actual = risk - float(np.trace(Bm @ np.linalg.inv(I + np.outer(a, a))))
    # Monte Carlo of the GLS estimator risk (independent confirmation)
    Sigma = np.eye(9)
    theta = rng.normal(size=4)
    ys = design @ theta + rng.normal(size=(200000, 9))
    est = ys @ design @ np.linalg.inv(design.T @ design)
    mc = float(np.mean(np.sum(((est - theta) @ target.T) ** 2 * np.diag(W), axis=1)))
    risk_noreg = float(np.trace(Bm @ np.linalg.inv(design.T @ design)))
    ok = abs(risk - lam.sum()) < 1e-12 and abs(red - actual) < 1e-12 and abs(mc - risk_noreg) / risk_noreg < .02
    return ok, {"risk": risk, "eigen_sum_error": abs(risk - lam.sum()), "one_query_reduction": red,
                "sherman_morrison_error": abs(red - actual), "monte_carlo_risk": mc, "exact_risk_unregularized": risk_noreg}


@check("C12.singular", "6.4", "c-optimal design with singular information: primal = dual, attained")
def c_singular():
    c = np.array([1., 1., 0.])
    pstar = 2 / 3
    I = np.diag([pstar, 4 * (1 - pstar), 0.])
    primal = float(c @ np.linalg.pinv(I) @ c)
    h = np.array([2., 1., 7.])
    dual = float((c @ h) ** 2 / max(h[0] ** 2, 4 * h[1] ** 2))
    grid = np.linspace(.001, .999, 10001)
    gmin = float(np.min(1 / grid + 1 / (4 * (1 - grid))))
    in_range = bool(np.allclose(I @ np.linalg.pinv(I) @ c, c))
    ok = abs(primal - dual) < 1e-12 and gmin >= primal - 1e-12 and np.linalg.matrix_rank(I) == 2 and in_range
    return ok, {"primal": primal, "dual": dual, "grid_minimum": gmin, "unidentified_nuisance_dimensions": 1}


@check("C13.radius", "6.16", "information-compilation radius: invisible uncertainty costs nothing, infeasibility still does; "
       "box uncertainty is maximized at a vertex")
def c_radius():
    C = np.array([[1., 0.]])
    M = np.array([[.2, -100.], [.2, 100.]])
    image = (C @ M.T).ravel()
    legal = np.linspace(0, .1, 101)
    radius = min(max(abs(image - q)) for q in legal)
    M2, legal2 = np.array([0., 1.]), np.linspace(0, 1, 101)
    radius2 = min(max(abs(M2 - q)) for q in legal2)
    # Box: inner max of ||C mu - q|| is at a vertex (convex function on a polytope).
    rng = rng_for("radius")
    r, p = 4, 3
    Cb = rng.normal(size=(p, r))
    lo, hi = rng.uniform(-1, 0, r), rng.uniform(0, 1, r)
    qv = rng.normal(size=p)
    verts = np.array(list(itertools.product(*zip(lo, hi))))
    vmax = float(max(np.linalg.norm(Cb @ v - qv) for v in verts))
    samples = lo + (hi - lo) * rng.uniform(size=(200000, r))
    smax = float(np.max(np.linalg.norm(samples @ Cb.T - qv, axis=1)))
    ok = np.ptp(image) == 0 and abs(radius - .1) < 1e-12 and abs(radius2 - .5) < 1e-12 and smax <= vmax + 1e-12
    return ok, {"invisible_uncertainty_diameter": float(np.ptp(image)), "infeasibility_error": float(radius),
                "two_target_radius": float(radius2), "box_vertex_max": vmax, "box_sampled_max": smax,
                "vertices": int(len(verts))}


# ----------------------------------------------------------------- section 7
B1, B2 = .9, .999


def adam_step(state, grad, t, eps=0., eta=1.):
    theta, m, v = state
    m = B1 * m + (1 - B1) * grad
    v = B2 * v + (1 - B2) * grad ** 2
    theta = theta - eta * (m / (1 - B1 ** t)) / (math.sqrt(v / (1 - B2 ** t)) + eps)
    return (theta, m, v)


@check("C14.memory", "7.6", "standard Adam: two reachable feedbacks give equal immediate parameters and different next steps")
def c_memory():
    one = Fraction(1)
    ga, gb = Fraction(1, 2), Fraction(2103, 790)
    f = lambda g: (Fraction(9, 10) + g) ** 2 / (Fraction(999, 1000) + g ** 2)
    exact_equal = f(ga) == f(gb)
    z1 = adam_step((0., 0., 0.), 1., 1)
    za, zb = adam_step(z1, float(ga), 2), adam_step(z1, float(gb), 2)
    za3, zb3 = adam_step(za, 1., 3), adam_step(zb, 1., 3)
    gap = float(sigmoid(za3[0]) - sigmoid(zb3[0]))
    eps = 1e-8
    zep = adam_step((0., 0., 0.), 1., 1, eps)
    target = adam_step(zep, .5, 2, eps)[0]
    root = optimize.brentq(lambda g: adam_step(zep, g, 2, eps)[0] - target, 2., 4., xtol=1e-15)
    e_eps = abs(adam_step(zep, root, 2, eps)[0] - target)
    next_eps = adam_step(adam_step(zep, .5, 2, eps), 1., 3, eps)[0] - adam_step(adam_step(zep, root, 2, eps), 1., 3, eps)[0]
    # SGD without memory: equal parameters give equal futures by construction.
    ok = exact_equal and abs(za[0] - zb[0]) < 1e-12 and abs(gap) > 3e-3 and e_eps < 1e-12 and abs(next_eps) > .07
    return ok, {"ratio_squared": str(f(ga)), "exact_equal": exact_equal, "checkpoint": list(z1),
                "theta2": za[0], "moments_a": [za[1], za[2]], "moments_b": [zb[1], zb[2]],
                "theta3_gap": za3[0] - zb3[0], "sigmoid_gap": gap, "positive_eps_root": root,
                "positive_eps_next_gap": next_eps, "bias_correction_v_at_t2": 1 / (1 - B2 ** 2)}


@check("C15.adjoint", "7.1", "complete AdamW one-step adjoint (Hessian, bias correction, eps, weight decay) = finite difference")
def c_adjoint():
    state = np.array([.7, .2, .8]); t, eps, wd, lr = 5, .03, .1, .07
    g = state[0] - .3
    mn = B1 * state[1] + (1 - B1) * g
    vn = B2 * state[2] + (1 - B2) * g * g
    aa, bb = 1 / (1 - B1 ** t), 1 / (1 - B2 ** t)
    D = aa / (math.sqrt(bb * vn) + eps)
    E = aa * bb * mn / (2 * math.sqrt(bb * vn) * (math.sqrt(bb * vn) + eps) ** 2)
    s = np.array([.4, -.2, .6])
    u = (1 - B1) * (s[1] - lr * D * s[0]) + 2 * (1 - B2) * g * (s[2] + lr * E * s[0])
    adj = np.array([(1 - lr * wd) * s[0] + u, B1 * (s[1] - lr * D * s[0]), B2 * (s[2] + lr * E * s[0])])

    def obj(z):
        gr = z[0] - .3
        m2 = B1 * z[1] + (1 - B1) * gr
        v2 = B2 * z[2] + (1 - B2) * gr * gr
        th = (1 - lr * wd) * z[0] - lr * aa * m2 / (math.sqrt(bb * v2) + eps)
        return s @ np.array([th, m2, v2])
    h = 1e-6
    num = np.array([(obj(state + h * e) - obj(state - h * e)) / (2 * h) for e in np.eye(3)])
    e = float(np.linalg.norm(adj - num))
    return e < 1e-8, {"finite_difference_error": e}


@check("C23.sampling", "7.1", "state-dependent tail sampling: grad V = sum_w p_w chi_w + sum_w V_w grad p_w; "
       "the costate average alone is wrong; any baseline gives the same sampling term")
def c_sampling():
    # One coordinate of Adam with bias correction and eps > 0. The first tail update consumes the gradient of
    # one of two data points, drawn with a probability that depends on theta and on the momentum m;
    # the second tail update is fixed. Final score F(theta) = sigmoid(2 theta).
    targets, eta, eps, t0 = (-.5, 1.2), .05, 1e-3, 6

    def branch_value(z, w):
        st = adam_step(tuple(z), z[0] - targets[w], t0 + 1, eps, eta)
        st = adam_step(st, st[0] - .4, t0 + 2, eps, eta)
        return float(sigmoid(2 * st[0]))

    def prob1(z):
        return float(sigmoid(2. * z[0] + 3. * z[1]))

    def value(z):
        p1 = prob1(z)
        return (1 - p1) * branch_value(z, 0) + p1 * branch_value(z, 1)

    z, h = np.array([.3, .1, .05]), 1e-6
    fd = lambda f: np.array([(f(z + h * e) - f(z - h * e)) / (2 * h) for e in np.eye(3)])
    full = fd(value)
    chi = [fd(lambda x, w=w: branch_value(x, w)) for w in (0, 1)]
    g1 = fd(prob1)
    p = np.array([1 - prob1(z), prob1(z)])
    vw = np.array([branch_value(z, w) for w in (0, 1)])
    dp = np.array([-g1, g1])
    average = p[0] * chi[0] + p[1] * chi[1]
    sampling = vw[0] * dp[0] + vw[1] * dp[1]
    formula = average + sampling
    base = vw.mean()
    baselined = average + (vw[0] - base) * dp[0] + (vw[1] - base) * dp[1]
    score_form = average + p[0] * vw[0] * (dp[0] / p[0]) + p[1] * vw[1] * (dp[1] / p[1])
    e_formula, e_average = err(formula, full), err(average, full)
    e_base, e_score = err(baselined, formula), err(score_form, formula)
    ok = e_formula < 1e-7 and e_average > 1e-3 and e_base < 1e-12 and e_score < 1e-12 and abs(sampling[2]) < 1e-12
    return ok, {"full_gradient_fd": full.tolist(), "costate_average": average.tolist(),
                "sampling_term": sampling.tolist(), "error_formula": e_formula,
                "error_costate_average_alone": e_average, "baseline_invariance_error": e_base,
                "score_form_error": e_score, "branch_probabilities": p.tolist(), "branch_values": vw.tolist()}


@check("C16.warm", "7.6", "warm relative change of v is (1-b2)(g^2/v-1); immediate share of first-order momentum displacement is 1-b1")
def c_warm():
    rng = rng_for("warm")
    v, g = rng.uniform(.1, 2, 1000), rng.normal(scale=2, size=1000)
    vn = B2 * v + (1 - B2) * g ** 2
    e = err((vn - v) / v, (1 - B2) * (g ** 2 / v - 1))
    total = sum((1 - B1) * B1 ** k for k in range(20000))
    big = float(np.max((1 - B2) * (g ** 2 / v - 1)))
    return e < 1e-12 and abs(total - 1) < 1e-12, {"identity_error": e, "immediate_share": 1 - B1,
                                                  "share_after_10_steps": 1 - B1 ** 10, "max_relative_change_in_sample": big}


# ----------------------------------------------------------------- section 8
@check("C17.catalog", "8.12", "estimated-spectrum residual <= sigma_{r+1} + 2 eps without a gap; catalog run counts")
def c_catalog():
    rng = rng_for("catalog")
    p, m, r, eps = 7, 11, 3, .004
    w = rng.uniform(.3, 1.3, p); w /= w.sum()
    q = rng.uniform(.3, 1.3, m); q /= q.sum()
    G = sigmoid(rng.normal(size=(p, m)))
    Gh = np.clip(G + rng.uniform(-eps, eps, G.shape), 0, 1)
    R, _ = weighted_operator(G, w, q)
    Rh, _ = weighted_operator(Gh, w, q)
    s = np.linalg.svd(R, compute_uv=False)
    Vh = np.linalg.svd(Rh)[2]
    Ph = Vh[:r].T @ Vh[:r]
    noise = float(np.linalg.norm(R - Rh, 2))
    resid = float(np.linalg.norm(R @ (np.eye(m) - Ph), 2))
    hoeff = lambda n, pp, mm, d: math.sqrt(math.log(2 * pp * mm / d) / (2 * n))
    counts = {"m20_n4_p_train1": 20 * 4 * 1, "m20_n4_p_train10": 20 * 4 * 10}
    radii = {f"n{n}": hoeff(n, 10, 20, .05) for n in (4, 16, 64, 256)}
    ok = noise <= eps + 1e-13 and resid <= s[r] + 2 * eps + 1e-13
    return ok, {"operator_noise": noise, "entrywise_bound": eps, "projection_residual": resid,
                "guaranteed": float(s[r] + 2 * eps), "continuation_runs": counts, "hoeffding_radius_p10_m20": radii}


@check("C18.shared", "8.13", "state-conditional inputs need the HS tail sum; the common-input operator bound fails")
def c_shared():
    R1, R2 = np.array([[1., 0.]]), np.array([[0., 1.]])
    stack = np.vstack([R1, R2]) / math.sqrt(2)
    actual = math.sqrt((np.linalg.norm(R1 @ [1., 0.]) ** 2 + np.linalg.norm(R2 @ [0., 1.]) ** 2) / 2)
    wrong, right = float(np.linalg.norm(stack, 2)), float(np.linalg.norm(stack, "fro"))
    rng = rng_for("shared")
    Ms = [rng.normal(size=(5, 8)) for _ in range(4)]
    st = np.vstack(Ms) / 2
    _, s, vt = np.linalg.svd(st, full_matrices=False)
    P = vt[:3].T @ vt[:3]
    hs = sum(np.linalg.norm(M @ (np.eye(8) - P), "fro") ** 2 for M in Ms) / 4
    e = abs(hs - float(np.sum(s[3:] ** 2)))
    return actual > wrong + .2 and actual <= right + 1e-12 and e < 1e-10, {
        "actual_rms": actual, "invalid_common_input_bound": wrong, "hs_bound": right, "tail_identity_error": e}


@check("C19.anchor", "8.14", "anchor reuse bound; paired second moment bounds the operator drift; the debiased paired product")
def c_anchor():
    rng = rng_for("anchor")
    A = rng.normal(size=(7, 10)); E = .02 * rng.normal(size=A.shape)
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    P = vt[:3].T @ vt[:3]
    resid = float(np.linalg.norm((A + E) @ (np.eye(10) - P), 2))
    bound = float(s[3] + np.linalg.norm(E, 2))
    g0, gz = rng.uniform(size=(6, 8)), rng.uniform(size=(6, 8))
    rho, omega = rng.dirichlet(np.ones(8)), rng.dirichlet(np.ones(6))
    delta = gz - g0
    cd = delta - (delta @ rho)[:, None]
    op2 = float(np.linalg.norm(np.sqrt(omega)[:, None] * cd * np.sqrt(rho)[None, :], 2) ** 2)
    second = float(omega @ np.abs(delta) @ rho)  # common-U Bernoulli coupling: E(X-X0)^2 = |g-g0|
    # Debiased product: two conditionally independent paired replicates D1, D2 at the same y.
    # Exact enumeration with Bernoulli outcomes under independent (not common) coupling.
    debias_err = 0.0
    for a in range(6):
        for j in range(8):
            pz, p0 = gz[a, j], g0[a, j]
            vals = {1: pz * (1 - p0), -1: (1 - pz) * p0, 0: pz * p0 + (1 - pz) * (1 - p0)}
            ED = sum(k * pr for k, pr in vals.items())
            Eprod = sum(k1 * k2 * p1 * p2 for k1, p1 in vals.items() for k2, p2 in vals.items())
            debias_err = max(debias_err, abs(Eprod - ED ** 2))
    ok = resid <= bound + 1e-12 and op2 <= second + 1e-12 and debias_err < 1e-15
    return ok, {"residual": resid, "bound": bound, "operator_drift_squared": op2, "paired_second_moment": second,
                "debiased_product_error": debias_err}


@check("C20.scale", "8.14", "at the measured evaluation noise the paired drift radius is near or above the trivial bound")
def c_scale():
    flips, tasks = 15, 300  # noise reference (record check R5)
    W = (math.sqrt(flips) / tasks) ** 2
    M, delta = 10, .05
    rows = []
    for n in (10, 40, 100, 1000):
        hoef = math.sqrt(math.log(M / delta) / (2 * n))
        rows.append({"pairs": n, "hoeffding_term": hoef, "radius": math.sqrt(min(1, W + hoef))})
    target = .03
    need = math.log(M / delta) / (2 * (target ** 2 - W) ** 2)
    floor = math.sqrt(W)
    ok = rows[0]["radius"] > .5 and need > 1e6
    return ok, {"W_bar_eval_noise_only": W, "rows": rows, "pairs_for_radius_0.03": need,
                "estimator_floor_sqrt_W": floor, "trivial_hs_bound": .5}


@check("C21.fidelity", "8.15", "two-fidelity estimator is unbiased with variance v0/n0+v1/n1; optimal allocation; "
       "cost thresholds for fixed and optimal coefficients")
def c_fidelity():
    flip = .0025
    prob = np.array([.5 * (1 - flip), .5 * flip, .5 * flip, .5 * (1 - flip)])
    X, Y = np.array([0., 0., 1., 1.]), np.array([0., 1., 0., 1.])
    var = lambda a: float(prob @ ((a - prob @ a) ** 2))
    vh, v0, v1 = var(X), var(Y), var(X - Y)
    unbiased = abs(prob @ X - (prob @ Y + prob @ (X - Y))) < 1e-15
    c0, c1, ch, mse = .01, 1.01, 1., 1e-4
    S = math.sqrt(v0 * c0) + math.sqrt(v1 * c1)
    n = np.array([math.sqrt(v0 / c0) * S / mse, math.sqrt(v1 / c1) * S / mse])
    alloc_ok = abs(v0 / n[0] + v1 / n[1] - mse) < 1e-14 and abs(n @ [c0, c1] - S * S / mse) < 1e-9
    # Thresholds with c = c_cheap/c_full = 0.1 and equal variances (correlation rho).
    c = .1
    fixed = lambda rho, c1r: math.sqrt(c) + math.sqrt(2 * (1 - rho) * c1r) - 1
    th_fixed = [optimize.brentq(lambda r_: fixed(r_, c1r), 0, 1) for c1r in (1.0, 1.1)]
    mfmc_ratio = lambda rho: (math.sqrt(1 - rho ** 2) + abs(rho) * math.sqrt(c)) ** 2
    th_mfmc = optimize.brentq(lambda r_: mfmc_ratio(r_) - 1, .05, .99)

    # Brute-force the nested two-level estimator: Var = (1/nH)(1 - rho^2) + (rho^2) * (1/nH - 1/nL) ... in closed form,
    # Var(MFMC) = vH/nH + (a^2 vL - 2 a rho sH sL)(1/nH - 1/nL); minimize cost*Var over a and nL/nH.
    def best(rho):
        def f(x):
            a, logr = x
            ratio = math.exp(logr) + 1.0
            var_ = 1 + (a * a - 2 * a * rho) * (1 - 1 / ratio)
            cost = 1 + c * ratio
            return var_ * cost
        sol = optimize.minimize(f, x0=[rho, 1.0], method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
        return min(sol.fun, 1.0)
    brute = {f"{rho:.2f}": best(rho) for rho in (.5, .6, .8, .9, .99)}

    # Same nested estimator when the cheap output of each paired sample is reused (its cost is not charged twice):
    # cost = nH * 1 + c * (nL - nH). Threshold found numerically.
    def best_reuse(rho):
        def f(x):
            a, logr = x
            extra = math.exp(logr)
            nL = 1.0 + extra
            return (1 + (a * a - 2 * a * rho) * (1 - 1 / nL)) * (1 + c * extra)
        vals = [optimize.minimize(f, x0=[rho, l0], method="Nelder-Mead",
                                  options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 40000}).fun for l0 in (-3, 0, 3, 6)]
        return min(min(vals), 1.0)
    th_reuse = optimize.brentq(lambda r_: best_reuse(r_) - 1 + 1e-9, .05, .95)
    formula = {f"{rho:.2f}": min(mfmc_ratio(rho), 1.0) for rho in (.5, .6, .8, .9, .99)}
    agree = max(abs(brute[k] - formula[k]) for k in brute) < 1e-6
    ok = unbiased and alloc_ok and agree and abs(th_mfmc - 2 * math.sqrt(c) / (1 + c)) < 1e-9 and abs(th_reuse - math.sqrt(c)) < 1e-3
    return ok, {"variances": [vh, v0, v1], "cost_ratio_synthetic": float(n @ [c0, c1] / (vh * ch / mse)),
                "fixed_coefficient_threshold_c1_full": th_fixed[0], "fixed_coefficient_threshold_c1_1.1": th_fixed[1],
                "nested_optimal_threshold": th_mfmc, "nested_optimal_threshold_cheap_reused": th_reuse,
                "sqrt_c": math.sqrt(c), "nested_ratio_formula": formula, "nested_ratio_bruteforce": brute,
                "max_saving_with_one_full_run_per_cell_n4": 4}


@check("C22.composed", "8.16", "per-step absolute floor of the composed bound; signed-sum versus per-term certification counts")
def c_composed():
    n, p, m, delta, B = 10, 1, 20, .05, 2.
    eps_n = math.sqrt(math.log(2 * p * m / delta) / (2 * n))
    W = (math.sqrt(15) / 300) ** 2
    drift = math.sqrt(min(1, W + math.sqrt(math.log(10 / delta) / (2 * n))))
    floor = B * (eps_n + drift)
    # Signed-sum estimation: N queries uniformly over R terms, variance sigma^2 R^2 / N (exact; Monte Carlo confirm).
    rng = rng_for("composed")
    R, N, sigma = 20, 400, 1.
    a = rng.normal(size=R)
    reps = 20000
    obs = a[None, :] + sigma * rng.normal(size=(reps, R, N // R)).mean(axis=2) * 1.0
    est = obs.sum(axis=1)
    mc_var = float(np.var(est))
    exact = sigma ** 2 * R ** 2 / N
    per_term_total = R * (sigma ** 2 * R ** 2)  # each term to eps/R at unit eps: queries ~ sigma^2 R^2 per term
    signed_total = sigma ** 2 * R ** 2
    ok = floor > 1 and abs(mc_var - exact) / exact < .05
    return ok, {"eps_n": eps_n, "drift_radius": drift, "per_step_floor": floor, "signed_sum_var_mc": mc_var,
                "signed_sum_var_exact": exact, "queries_per_unit_eps_signed": signed_total,
                "queries_per_unit_eps_per_term_abs": per_term_total}


# ----------------------------------------------------------------- record checks
def _load(path):
    return json.loads((RECORDS / path).read_text())


@check("R1.r0a", "5.11", "R0a: the pre-registered warm criterion fails at all six states; post-hoc ratio against the "
       "signed first-order term")
def r_r0a():
    names = ["state-update-1", "state-update-4", "state-update-8", "checkpoint-tokens-31221",
             "checkpoint-tokens-104070", "checkpoint-tokens-416280"]
    rows = []
    for nm in names:
        d = _load(f"r0a/{nm}.json")
        rows.append({"state": nm, "E_fd": d["E_fd"], "prereg_ratio": d["E_fd_over_first_order"],
                     "prereg_pass": d["preregistered_warm_criterion"],
                     "post_hoc_ratio": d["E_fd"] / abs(d["parameter_first_order_signed"])})
    ok = all(not r["prereg_pass"] for r in rows) and abs(rows[4]["post_hoc_ratio"] - .0972) < 1e-3
    return ok, {"rows": rows}


@check("R2.r0b", "5.11", "R0b: median response non-additivity at update 8 and at p10 (per-token surrogate)")
def r_r0b():
    u8 = _load("r0b/merge_update8_F_tilde_per_token.json")["median_pair_nu_resp"]
    p10 = _load("r0b/merge_p10_F_tilde_per_token.json")["median_pair_nu_resp"]
    g8 = _load("r0b/merge_update8_F_tilde_geo_prob.json")["median_pair_nu_resp"]
    g10 = _load("r0b/merge_p10_F_tilde_geo_prob.json")["median_pair_nu_resp"]
    return u8 is not None and p10 is not None, {"update8_per_token": u8, "p10_per_token": p10,
                                                "update8_geo_prob": g8, "p10_geo_prob": g10}


@check("R3.r3", "7.2", "R3: measured responses at five magnitudes lie inside their diagnostic radii and have the opposite "
       "sign to the restricted-model prediction")
def r_r3():
    d = _load("r3/r3_eval_summary.json")["F_tilde_per_token"]
    rows = [{"arm": a, "c": c, "measured": dF, "diagnostic_radius": rad, "prediction": pred,
             "inside": abs(dF) <= rad, "ratio_pred_over_measured": abs(pred / dF) if dF else None,
             "opposite_sign": dF * pred < 0} for a, c, dF, rad, _, pred in d if a.endswith("noclip")]
    ok = all(r["inside"] and r["opposite_sign"] for r in rows)
    ratios = [r["ratio_pred_over_measured"] for r in rows]
    return ok, {"rows": rows, "ratio_range": [min(ratios), max(ratios)]}


@check("R4.r1", "8.8", "R1: P1 versus P3 paired CRN score, diagnostic and a-priori (strict) radii")
def r_r1():
    out = {}
    for metric in ("F_tilde_geo_prob", "F_tilde_per_token"):
        d = _load(f"r1/merge_p1p3_{metric}.json")
        pr = d["pairs"][0]
        z = np.array(pr["Z"], float)
        out[metric] = {"S_hat": pr["S_hat"], "recomputed_mean": float(z.mean()), "replicas": len(z),
                       "positive_replicas": int(np.sum(z > 0)), "diagnostic_radius": pr["radius_observed_range_DIAGNOSTIC"],
                       "strict_radius": pr["radius_a_priori"], "B": d["B"]}
    g = out["F_tilde_geo_prob"]
    ok = abs(g["S_hat"] - g["recomputed_mean"]) < 1e-12 and g["S_hat"] > g["diagnostic_radius"] and g["strict_radius"] > .3
    return ok, out


@check("R5.noise", "8.14", "noise reference: one checkpoint, two identical servers, dev300 successes and task flips")
def r_noise():
    rows = {}
    for s in "ab":
        with open(RECORDS / f"noise/{s}/per_task.tsv") as fh:
            rows[s] = {r["task_id"]: r["success"] == "True" for r in csv.DictReader(fh, delimiter="\t")}
    a, b = rows["a"], rows["b"]
    flips = sum(a[k] != b[k] for k in a)
    out = {"tasks": len(a), "success_a": sum(a.values()), "success_b": sum(b.values()), "flips": flips,
           "a_only": sum(a[k] and not b[k] for k in a), "b_only": sum(b[k] and not a[k] for k in a),
           "sd_of_score_difference_if_flips_symmetric": math.sqrt(flips) / len(a)}
    return set(a) == set(b) and len(a) == 300, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(HERE / "results_v017.json"))
    args = ap.parse_args()
    results, failed = [], []
    for code, item, claim, fn in CHECKS:
        ok, values = fn()
        results.append({"id": f"v017:{code}", "item": item, "claim": claim, "pass": bool(ok), "values": values})
        if not ok:
            failed.append(code)
        print(f"{'PASS' if ok else 'FAIL'}  v017:{code:<16} item {item:<5} {claim[:80]}")
    meta = {"seed": SEED, "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "status": "PASS" if not failed else "FAIL", "failed": failed}
    Path(args.output).write_text(json.dumps({"meta": meta, "checks": results}, indent=2, ensure_ascii=False,
                                            default=lambda o: o if not isinstance(o, np.generic) else o.item()) + "\n")
    print(json.dumps(meta))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
