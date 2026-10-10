#!/usr/bin/env python3
"""Independent v0.14 checks, derived solely from the two English statements.

No repository modules, original verification programs, or prior results are read.
Every failed assertion is retained; the command writes its report before returning 1.
The only neural model is the explicitly labelled, numpy-only controlled experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools as it
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from fractions import Fraction as F
from functools import lru_cache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGURE_DIR = HERE / 'figs'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplconfig')
os.environ['MPLBACKEND'] = 'Agg'
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
sys.dont_write_bytecode = True


def no_network(*args, **kwargs):
    raise RuntimeError('Independent checks prohibit network access')


socket.create_connection = no_network
socket.getaddrinfo = no_network
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.socket.sendto = no_network

import numpy as np
import scipy
from scipy.optimize import linprog, minimize_scalar
from scipy.special import expit
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt

MASTER = 20261010
TOL = 1e-8
ACTIVE = None


def serial(x):
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    if isinstance(x, np.ndarray):
        return serial(x.tolist())
    if isinstance(x, (F, sp.Basic)):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    return x


class Claim:
    def __init__(self, number, statement, method, kind='exact'):
        global ACTIVE
        ACTIVE = self
        self.id = f'C{number}'
        self.seed = [MASTER, number]
        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed))
        self.data = dict(id=self.id, statement=statement, method=method, kind=kind,
                         n_instances=0, n_assertions=0, worst_residual=0.0,
                         worst_lp_feasibility_residual=0.0, status='PASS',
                         seed=self.seed, notes=[], details={})
        self.failures = []

    def check(self, condition, label, instance=None, residual=0):
        self.data['n_assertions'] += 1
        self.data['worst_residual'] = max(self.data['worst_residual'], float(abs(residual)))
        if not bool(condition):
            self.data['status'] = 'FAIL'
            self.failures.append(serial(dict(claim=self.id, assertion=label,
                                            seed=self.seed, instance=instance,
                                            residual=residual)))

    def eq(self, lhs, rhs, label, instance=None, floating=False):
        if isinstance(lhs, (bool, np.bool_)) or isinstance(rhs, (bool, np.bool_)):
            r = int(bool(lhs) != bool(rhs))
        elif isinstance(lhs, sp.Basic) or isinstance(rhs, sp.Basic):
            r = abs(sp.simplify(lhs-rhs))
        else:
            r = abs(lhs-rhs)
        self.check(r <= TOL if floating else r == 0, label,
                   dict(lhs=lhs, rhs=rhs, inputs=instance), r)

    def le(self, lhs, rhs, label, instance=None, floating=False):
        r = max(0, lhs - rhs)
        self.check(lhs <= rhs + (TOL if floating else 0), label,
                   dict(lhs=lhs, rhs=rhs, inputs=instance), r)

    def count(self, n=1):
        self.data['n_instances'] += n

    def finish(self):
        return dict(self.data, counterexamples=self.failures)


def rationals(rng, shape, den=12):
    return np.array([F(int(v), den) for v in rng.integers(0, den + 1, size=shape).flat],
                    dtype=object).reshape(shape)


def dist(rng, n, positive=True):
    v = rng.integers(1 if positive else 0, 9, size=n)
    if not v.sum():
        v[0] = 1
    return np.array([F(int(x), int(v.sum())) for x in v], dtype=object)


def lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    def arr(a):
        return None if a is None else np.asarray(a, dtype=float)
    r = linprog(arr(c), A_ub=arr(A_ub), b_ub=arr(b_ub), A_eq=arr(A_eq),
                b_eq=arr(b_eq), bounds=bounds, method='highs',
                options={'primal_feasibility_tolerance': 1e-9,
                         'dual_feasibility_tolerance': 1e-9})
    if not r.success:
        raise RuntimeError(f'LP solver status {r.status}')
    residual = 0.0
    if A_eq is not None:
        residual = max(residual, float(np.max(abs(arr(A_eq) @ r.x - arr(b_eq)))))
    if A_ub is not None:
        residual = max(residual, float(np.max(arr(A_ub) @ r.x - arr(b_ub))))
    bs = [(0, None)] * len(c) if bounds is None else bounds
    for x, (lo, hi) in zip(r.x, bs):
        residual = max(residual, 0 if lo is None else lo - x, 0 if hi is None else x - hi)
    if ACTIVE is not None:
        ACTIVE.data['worst_lp_feasibility_residual'] = max(
            ACTIVE.data['worst_lp_feasibility_residual'], residual)
        ACTIVE.check(residual <= TOL, 'LP primal feasibility <= 1e-8',
                     dict(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                          bounds=bs, solution=r.x), residual)
    return r


def pair_width(g, phi, eps=None):
    """Direct 2m-variable probability primal; no signed-measure reformulation."""
    g, phi = np.asarray(g, float), np.asarray(phi, float)
    m = len(g)
    phi = phi.reshape((-1, m))
    eq = [np.r_[np.ones(m), np.zeros(m)], np.r_[np.zeros(m), np.ones(m)]]
    if eps is None:
        eq.extend(np.c_[phi, -phi])
        r = lp(np.r_[-g, g], A_eq=eq, b_eq=[1, 1] + [0] * len(phi))
    else:
        a = np.c_[phi, -phi]
        r = lp(np.r_[-g, g], A_eq=eq, b_eq=[1, 1],
               A_ub=np.r_[a, -a] if len(phi) else None,
               b_ub=np.r_[eps, eps] if len(phi) else None)
    return -r.fun, r.x[:m], r.x[m:]


def approximation(g, features):
    g = np.asarray(g, float)
    h = np.column_stack([np.ones(len(g)), np.asarray(features, float).reshape((-1, len(g))).T])
    r = lp(np.r_[np.zeros(h.shape[1]), 1],
           A_ub=np.r_[np.c_[h, -np.ones(len(g))], np.c_[-h, -np.ones(len(g))]],
           b_ub=np.r_[g, -g], bounds=[(None, None)] * h.shape[1] + [(0, None)])
    return r.fun, r.x[:-1]


def noisy_dual(g, phi, eps, oscillation=True):
    g, phi, eps = np.asarray(g, float), np.asarray(phi, float), np.asarray(eps, float)
    m, d = len(g), len(phi)
    # b=b+ - b-, followed by u,l or c,t; independent from pair_width.
    bmat = np.c_[phi.T, -phi.T]
    if oscillation:
        a = np.r_[np.c_[-bmat, -np.ones(m), np.zeros(m)],
                  np.c_[bmat, np.zeros(m), np.ones(m)]]
        cost = np.r_[eps, eps, 1, -1]
        bounds = [(0, None)] * (2*d) + [(None, None)] * 2
    else:
        a = np.r_[np.c_[-bmat, -np.ones(m), -np.ones(m)],
                  np.c_[bmat, np.ones(m), -np.ones(m)]]
        cost = np.r_[eps, eps, 0, 2]
        bounds = [(0, None)] * (2*d) + [(None, None), (0, None)]
    return lp(cost, A_ub=a, b_ub=np.r_[-g, g], bounds=bounds).fun


@lru_cache(None)
def alphas(k, N):
    return tuple(a for a in it.product(range(N+1), repeat=k) if sum(a) <= N)


def moment(points, weights, alpha):
    return sum((w * math.prod(x**a for x, a in zip(p, alpha))
                for p, w in zip(points, weights)), F(0))


def law_scalar(N):
    points = [(F(i, N),) for i in range(N+1)]
    p = [F(math.comb(N, i), 2**(N-1)) if i % 2 == 0 else F(0) for i in range(N+1)]
    q = [F(math.comb(N, i), 2**(N-1)) if i % 2 else F(0) for i in range(N+1)]
    return points, p, q


def law_parity(N):
    points = list(it.product((F(0), F(1)), repeat=N))
    p = [F(int(sum(x) % 2 == 0), 2**(N-1)) for x in points]
    q = [F(int(sum(x) % 2 == 1), 2**(N-1)) for x in points]
    return points, p, q


def tree(rng, k, depth):
    """History-specific contexts, reused parameters, early stops and chance nodes."""
    if depth == 0 or rng.random() < .12:
        return ('reward', F(int(rng.integers(0, 9)), 8))
    if rng.random() < .25:
        return ('chance', F(int(rng.integers(1, 4)), 4),
                tree(rng, k, depth-1), tree(rng, k, depth-1))
    return ('action', int(rng.integers(k)), tree(rng, k, depth-1), tree(rng, k, depth-1))


def tree_expr(t, ps):
    if t[0] == 'reward':
        return sp.Rational(t[1])
    x = sp.Rational(t[1]) if t[0] == 'chance' else ps[t[1]]
    return sp.expand((1-x)*tree_expr(t[2], ps) + x*tree_expr(t[3], ps))


def poly_dict(expr, ps):
    return {a: F(c) for a, c in sp.Poly(expr, *ps).terms()}


def poly_value(poly, p):
    return sum(c * math.prod(x**a for x, a in zip(p, alpha)) for alpha, c in poly.items())


def affine_moments(points, weights, ws, p0, N):
    k = len(p0)
    aa = alphas(k, N)
    raw = {a: moment(points, weights, a) for a in aa}
    out = {a: math.prod((ws[0]*p0[j])**a[j] for j in range(k)) for a in aa}
    for w in ws[1:]:
        nxt = {}
        for a in aa:
            nxt[a] = sum(math.prod(math.comb(a[j], b[j]) for j in range(k)) *
                         out[tuple(a[j]-b[j] for j in range(k))] * raw[b] * w**sum(b)
                         for b in aa if all(b[j] <= a[j] for j in range(k)))
        out = nxt
    return out


def expected_poly(poly, moments):
    return sum(c*moments[a] for a, c in poly.items())


@lru_cache(None)
def partitions(n):
    if n == 0:
        return ((),)
    out = []
    for blocks in partitions(n-1):
        out.append(blocks + ((n-1,),))
        for j in range(len(blocks)):
            out.append(blocks[:j] + (blocks[j] + (n-1,),) + blocks[j+1:])
    return tuple(out)


def iid_sum_coeffs(alpha, points, weights):
    """Direct enumeration of equal-sample-index patterns in the multinomial sum."""
    axes = [j for j, a in enumerate(alpha) for _ in range(a)]
    coef = [F(0)] * (len(axes)+1)
    raw = {a: moment(points, weights, a) for a in alphas(len(alpha), len(axes))}
    for blocks in partitions(len(axes)):
        val = F(1)
        for block in blocks:
            a = tuple(sum(axes[i] == j for i in block) for j in range(len(alpha)))
            val *= raw[a]
        coef[len(blocks)] += val
    return coef


def average_moment(coef, degree, n):
    fall, total = 1, F(0)
    for b, c in enumerate(coef):
        if b:
            fall *= n-b+1
        total += c*fall
    return total / n**degree


def cumulant(alpha, points, weights):
    axes = [j for j, a in enumerate(alpha) for _ in range(a)]
    result = F(0)
    for blocks in partitions(len(axes)):
        v = F((-1)**(len(blocks)-1) * math.factorial(len(blocks)-1))
        for block in blocks:
            a = tuple(sum(axes[i] == j for i in block) for j in range(len(alpha)))
            v *= moment(points, weights, a)
        result += v
    return result


def process(rng, states=6, rounds=3):
    m, nx = int(rng.integers(2, 5)), 2
    updates = rng.integers(states, size=(rounds, states, m, nx))
    laws = [np.array([[dist(rng, m) for _ in range(states)] for _ in range(rounds)], object)
            for _ in range(2)]
    xi = np.array([F(1, 3), F(2, 3)], object)
    kernels = []
    for law in laws:
        K = np.full((rounds, states, states), F(0), object)
        for t, z, y, x in it.product(range(rounds), range(states), range(m), range(nx)):
            K[t, z, updates[t, z, y, x]] += law[t, z, y]*xi[x]
        kernels.append(K)
    return kernels, dist(rng, states), rationals(rng, (states,)), dict(
        deterministic_updates=updates, message_laws=laws, xi=xi)


def continuations(K, terminal):
    V = [None]*(len(K)+1)
    V[-1] = terminal
    for t in reversed(range(len(K))):
        V[t] = K[t] @ V[t+1]
    return V


def process_stats(K0, KS, init, terminal, certified=None):
    V = continuations(K0, terminal)
    d, signed, absolute = init.copy(), F(0), F(0)
    alive, stopped = init.copy(), F(0)
    for t in range(len(K0)):
        drift = (K0[t]-KS[t]) @ V[t+1]
        signed += d @ drift
        absolute += d @ np.abs(drift)
        if certified is not None:
            alive = alive * certified[t]
            stopped += alive @ np.abs(drift)
            alive = alive @ KS[t]
        d = d @ KS[t]
    return dict(gap=init @ V[0] - d @ terminal, signed=signed, absolute=absolute,
                stopped=stopped, rho=1-sum(alive), reference=init @ V[0],
                alternative=d @ terminal, V=V)


def savefig(name):
    directory = FIGURE_DIR
    directory.mkdir(parents=True,exist_ok=True)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        plt.savefig(directory / f'{name}.{ext}', dpi=160)
    plt.close()


def check_C1(quick=False):
    c = Claim(1, 'Propositions 1–3; equations (1)–(5)',
              'Rational joint tables; independent minimax LPs; garbling search', 'adversarial')
    rng, worst = c.rng, -math.inf
    for i in range(16 if quick else 500):
        ne, npol, nd = int(rng.integers(3, 7)), int(rng.integers(2, 6)), 4
        prior = dist(rng, ne)
        values = rationals(rng, (ne, npol+2))
        if i % 4 == 0:  # common optima, with ties
            values[:, :2] = F(1)
        elif i % 4 == 1:
            values[:, :npol] = F(0)
            for e in range(ne):
                values[e, e % npol] = F(1)
        channel = np.array([dist(rng, nd, False) for _ in range(ne)], object)
        learner = np.array([dist(rng, npol) for _ in range(nd)], object)
        joint = prior[:, None]*channel
        pv = values[:, :npol]
        post_scores = joint.T @ pv
        B = sum(max(row) for row in post_scores)
        oracle = prior @ np.max(pv, axis=1)
        ref = prior @ np.max(values, axis=1)
        learned = sum((post_scores*learner).flat)
        rep, info, alg = ref-oracle, oracle-B, B-learned
        inst = dict(prior=prior, values=values, channel=channel, learner=learner)
        c.eq(ref-learned, rep+info+alg, 'decomposition', inst)
        c.check(min(rep, info, alg) >= 0, 'all three gaps nonnegative', inst)
        common = all(any(all(joint[e,d] == 0 or pv[e,p] == max(pv[e])
                                for e in range(ne)) for p in range(npol)) for d in range(nd))
        c.eq(info == 0, common, 'posterior common optimizer iff zero information gap', inst)
        # The garbling objective is convex in the new channel: search deterministic
        # vertices as well as randomized channels to maximize B(garbled)-B(original).
        garble_best = -math.inf
        candidates = [np.array([dist(rng, 3) for _ in range(nd)], object)]
        if i < (4 if quick else 30):
            candidates += [np.array([[F(int(j == dest)) for j in range(3)] for dest in assignment], object)
                           for assignment in it.product(range(3), repeat=nd)]
        for garble in candidates:
            bg = sum(max(row) for row in garble.T @ post_scores)
            c.le(bg, B, 'data processing', dict(**inst, garble=garble))
            garble_best = max(garble_best, float(bg-B))
        worst = max(worst, garble_best)
        zchan = np.array([dist(rng, 3) for _ in range(nd)], object)
        augmented = sum(max(post_scores[d]*zchan[d,z]) for d in range(nd) for z in range(3))
        c.eq(augmented, B, 'conditionally independent augmentation', dict(**inst, Z_channel=zchan))
        if i < (12 if quick else 220):
            gaps = np.asarray(np.max(pv, axis=1)[:, None]-pv, float)
            primal = lp(np.r_[np.zeros(npol), 1], A_ub=np.c_[gaps, -np.ones(ne)],
                        b_ub=np.zeros(ne), A_eq=[np.r_[np.ones(npol), 0]], b_eq=[1],
                        bounds=[(0, None)]*npol+[(None, None)])
            dual = lp(np.r_[np.zeros(ne), -1], A_ub=np.c_[-gaps.T, np.ones(npol)],
                      b_ub=np.zeros(npol), A_eq=[np.r_[np.ones(ne), 0]], b_eq=[1],
                      bounds=[(0, None)]*ne+[(None, None)])
            c.eq(primal.fun, -dual.fun, 'separate minimax primal and dual', inst, True)
            c.eq(abs(primal.fun) <= TOL, bool(np.any(np.all(gaps == 0, axis=0))),
                 'minimax zero iff common optimizer', inst)
        c.count()
    c.data['details'] = dict(minimax_instances=12 if quick else 220,
                             adversarial_garbling_best_lhs_minus_rhs=worst)
    return c.finish()


def binomial_tail(n, p, lower, upper):
    """Integer numerator summation, not a floating binomial approximation."""
    a, den = p.numerator, p.denominator
    b = den-a
    if not 0 < a < den:
        return F(int(lower <= (n if a else 0) <= upper))
    term, total = b**n, 0
    for j in range(n+1):
        if lower <= j <= upper:
            total += term
        if j < n:
            term = term*(n-j)*a // ((j+1)*b)
    return F(total, den**n)


def walk_recursion(p, h):
    # Solve E_i=1+p E_{i+1}+(1-p) E_{i-1}, absorbing at ±h,
    # and the analogous lower-exit probability by exact tridiagonal elimination.
    n = 2*h-1
    a, b, u = [-(1-p)]*n, [F(1)]*n, [-p]*n
    rhs = [[F(1), F(1-p) if i == 0 else F(0)] for i in range(n)]
    for i in range(1,n):
        f = a[i]/b[i-1]
        b[i] -= f*u[i-1]
        rhs[i] = [rhs[i][j]-f*rhs[i-1][j] for j in range(2)]
    sol = [[F(0), F(0)] for _ in range(n)]
    sol[-1] = [v/b[-1] for v in rhs[-1]]
    for i in reversed(range(n-1)):
        sol[i] = [(rhs[i][j]-u[i]*sol[i+1][j])/b[i] for j in range(2)]
    return sol[h-1]


def check_C2(quick=False):
    c = Claim(2, 'Theorem 4 and compilation',
              'Symbolic payoff; exact binomial sums and absorbing-walk recursion; analytic KL supremum', 'adversarial')
    p, v, D = sp.symbols('p v D', real=True)
    table = (p+(1-p)*(sp.Rational(1,2)-2*v*D)+p*(sp.Rational(1,2)+2*v*D)+1-p)/2
    c.eq(sp.expand(table), sp.Rational(3,4)+v*D*(2*p-1), 'payoff table')
    c.eq(sp.solve(2*D*(1-p)-2*D*p,p)[0], sp.Rational(1,2), 'passive minimax intersection')
    c.eq((2*D*p).subs(p,sp.Rational(1,2)),D,'passive minimax value')
    u = np.linspace(1e-7,.25,20001)
    ratio = 2*u*np.log1p(4*u/(1-2*u))/u**2
    sup = 8*math.log(3)
    c.le(float(ratio.max()),16,'dense KL inequality', floating=True)
    opt = minimize_scalar(lambda x: -(2/x)*math.log((.5+x)/(.5-x)),
                          bounds=(1e-6,.25),method='bounded')
    c.le(-opt.fun,sup,'adversarial KL ratio',floating=True)
    c.data['notes'].append('KL/u² = 8 Σ_{j≥0}(4u²)^j/(2j+1); positive coefficients prove monotonicity; supremum 8 log(3).')
    rows=[]
    for delta in map(F,('0.01','0.05','0.1','0.25','0.4')):
        for den in (8,10,16,32,64):
            d=F(1,den); prob=F(1,2)+2*d
            klout=(1-2*float(delta))*math.log(float((1-delta)/delta))
            klin=(2*float(prob)-1)*math.log(float(prob/(1-prob)))
            bound=klout/klin; loose=klout/(64*float(d)**2)
            # Minimal odd n is globally minimal: the preceding odd test dominates
            # its next even test when ties are counted as failures in both worlds.
            hi=max(1,math.ceil(math.log(1/float(delta))/(8*float(d)**2)))
            lo, high=0,(hi+1)//2
            while lo<high:
                mid=(lo+high)//2; n=2*mid+1
                if binomial_tail(n,prob,0,n//2)<=delta:
                    high=mid
                else:
                    lo=mid+1
            n=2*lo+1
            err=binomial_tail(n,prob,0,n//2)
            opposite_error=binomial_tail(n,1-prob,(n+1)//2,n)
            c.eq(err,opposite_error,'fixed threshold both-world symmetry',dict(n=n,Delta=d,delta=delta))
            c.le(err,delta,'fixed n threshold succeeds',dict(delta=delta,Delta=d,n=n))
            if n>1:
                c.check(binomial_tail(n-1,prob,0,(n-1)//2)>delta,
                        'preceding even n fails',dict(n=n,delta=delta,Delta=d))
            if n>2:
                c.check(binomial_tail(n-2,prob,0,(n-2)//2)>delta,
                        'preceding odd n fails',dict(n=n,delta=delta,Delta=d))
            h=math.ceil(math.log(float((1-delta)/delta))/math.log(float(prob/(1-prob))))
            expected, sprt_error=walk_recursion(prob,h)
            opposite_expected, opposite_lower_exit=walk_recursion(1-prob,h)
            c.eq(expected,opposite_expected,'SPRT expected time in both worlds',dict(h=h,prob=prob))
            c.eq(sprt_error,1-opposite_lower_exit,'SPRT error in both worlds',dict(h=h,prob=prob))
            c.le(sprt_error,delta,'SPRT actual error',dict(h=h,prob=prob))
            c.le(bound,n,'fixed sample lower bound',dict(n=n,Delta=d,delta=delta),True)
            c.le(bound,float(expected),'sequential lower bound',dict(expected=expected,h=h,Delta=d,delta=delta),True)
            c.le(loose,bound,'second lower bound',floating=True)
            n9=math.ceil(math.log(1/float(delta))/(8*float(d)**2))
            err9=binomial_tail(n9,prob,0,n9//2)
            c.le(err9,delta,'equation (9)',dict(n=n9,Delta=d,delta=delta))
            rows.append(dict(Delta=d,delta=delta,min_fixed_n=n,threshold_error=float(err),
                             sprt_boundary=h,sprt_expected_n=float(expected),sprt_error=float(sprt_error),
                             bound8=bound,fixed_ratio=n/bound,sprt_ratio=float(expected)/bound,
                             n9=n9,error9=float(err9)))
            c.count()
    compile_rows=[]
    for dl in map(F,('0.01','0.05','0.1','0.25','0.4')):
        m=math.ceil(32*math.log(2/float(dl)))
        for q in (F(1,8),F(7,8)):
            bad=sum(binomial_tail(m,q,i,i) for i in range(m+1) if abs(F(i,m)-q)>F(1,8))
            c.le(bad,dl,'compilation binomial tail',dict(m=m,q=q,delta_L=dl))
            for i in range(m+1):
                if abs(F(i,m)-q)<=F(1,8):
                    normalized_regret=2*(1-F(i,m)) if q>F(1,2) else 2*F(i,m)
                    c.le(normalized_regret,F(1,2),'compiled regret / Delta',dict(m=m,i=i,q=q))
            compile_rows.append(dict(m=m,q=q,delta_L=dl,error=float(bad)))
    c.data['details']=dict(configurations=rows,compilation=compile_rows,analytic_kl_sup=sup,
                           optimizer_kl_ratio=-opt.fun, minimum_sprt_ratio=min(r['sprt_ratio'] for r in rows))
    return c.finish()


def check_C3(quick=False):
    c=Claim(3,'Propositions 5 and 6','Symbolic state sums and exact packet/binomial enumeration')
    p,eta=sp.symbols('p eta')
    ra=list(map(sp.Rational,('19/30','3/10','9/10','13/30')))
    for weights,target in [([F(3,4),F(1,4),F(0),F(0)],F(43,80)),
                           ([F(0),F(0),F(1,4),F(3,4)],F(41,80))]:
        value=sum(w*(a*p+(1-p)/2) for w,a in zip(weights,ra))
        teacher=sum(w*max(a,sp.Rational(1,2)) for w,a in zip(weights,ra))
        q=sum(w for w,a in zip(weights,ra) if a>sp.Rational(1,2))
        c.eq(sp.expand(value),sp.Rational(1,2)+p/20,'source static value')
        c.eq(teacher,sp.Rational(3,5),'teacher value')
        for j,a in enumerate(ra):
            c.check((a>sp.Rational(1,2))==(j in (0,2)),'teacher statewise optimality',dict(state=j,reward_A=a))
        for m in range(1,31):
            learned=sum(F(math.comb(m,i))*q**i*(1-q)**(m-i)*(F(1,2)+F(i,20*m)) for i in range(m+1))
            c.eq(learned,target,'frequency learner all m',dict(m=m,q=q)); c.count()
    vals=[(1-eta)/2+eta*(a+b)/2 for a,b in it.product((0,1),repeat=2)]
    c.eq(sp.expand(sum(vals)/4),sp.Rational(1,2),'equal packet mean')
    c.eq(sp.expand(sum(x*x for x in vals)/4-sp.Rational(1,4)),eta**2/8,'packet gap')
    c.eq(sp.expand(sum(1-x*x for x in vals)/4-sp.Rational(3,4)),-eta**2/8,'preference reversal')
    c.data['details']=dict(prop5_gap=F(1,40),prop6_gap='eta**2/8')
    return c.finish()


def check_C4(quick=False):
    c=Claim(4,'Theorem 7, equations (14)–(17)','Actual rational update kernels, first-exit recursion, subset hill-climb','adversarial')
    rng=c.rng; best=-math.inf
    for i in range(5 if quick else 45):
        s=int(rng.integers(4,21)); R=int(rng.integers(1,9))
        (K0,KS),init,Ft,meta=process(rng,s,R)
        st=process_stats(K0,KS,init,Ft)
        inst=dict(**meta,initial=init,terminal=Ft)
        c.eq(st['gap'],st['signed'],'telescoping identity',inst)
        c.le(abs(st['gap']),st['absolute'],'absolute composition',inst)
        masks=np.ones((R,s),dtype=int) if i%2==0 else rng.integers(0,2,size=(R,s))
        best_local=-math.inf
        for trial in range(10 if quick else 35):
            candidate=masks.copy()
            if trial:
                candidate[int(rng.integers(R)),int(rng.integers(s))]^=1
            z=process_stats(K0,KS,init,Ft,candidate)
            score=float(abs(z['gap'])-z['stopped']-z['rho'])
            c.le(abs(z['gap']),z['stopped']+z['rho'],'first-exit certificate',dict(**inst,certified=candidate))
            if score>best_local:
                masks,best_local=candidate,score
        best=max(best,best_local)
        # One-boundary family consists of the actual response vectors.
        U=meta['deterministic_updates']; xi=meta['xi']
        for t in range(R):
            z=int(rng.integers(s)); g=np.array([sum(xi[x]*st['V'][t+1][U[t,z,y,x]] for x in range(2))
                                              for y in range(U.shape[2])],object)
            P,Q=meta['message_laws'][0][t,z],meta['message_laws'][1][t,z]
            val0=sum(P[y]*xi[x]*st['V'][t+1][U[t,z,y,x]] for y in range(len(P)) for x in range(2))
            val1=sum(Q[y]*xi[x]*st['V'][t+1][U[t,z,y,x]] for y in range(len(Q)) for x in range(2))
            c.eq(val0-val1,(P-Q)@g,'one-boundary response equivalence',dict(g=g,P=P,Q=Q,U=U[t,z]))
        c.count()
    # Different laws annihilate a whole declared response family and actual
    # one-step updates z+=y realize every function as terminal reward.
    P=np.array([F(1,2),F(0),F(1,2)],object); Q=np.array([F(0),F(1),F(0)],object)
    for g in ([F(1)]*3,[F(0),F(1,2),F(1)]):
        c.eq(P@np.array(g,object),Q@np.array(g,object),'matched response family')
    c.check(P@np.array([F(0),F(0),F(1)],object)!=Q@np.array([F(0),F(0),F(1)],object),'separating response intervention')
    identity=np.array([[F(1),F(0)],[F(0),F(1)]],object)
    flip=identity[:,::-1]
    w=process_stats(np.array([identity,identity]),np.array([flip,flip]),
                    np.array([F(1),F(0)]),np.array([F(0),F(1)]))
    c.eq(w['gap'],F(0),'two-round cancellation')
    c.check(w['absolute']>0,'pointwise mismatch despite final equality')
    c.data['details']=dict(adversarial_exit_best_lhs_minus_rhs=best,cancellation_absolute_drift=w['absolute'])
    return c.finish()


def check_C5(quick=False):
    c=Claim(5,'Theorem 8 and Corollary 9','Exact ranks; independent probability and approximation LPs; transfer LP','adversarial')
    rng=c.rng; maxdual=0.; transferbest=-math.inf
    for i in range(8 if quick else 300):
        m=int(rng.integers(4,13)); d=int(rng.integers(1,m-1))
        phi=rationals(rng,(d,m)); G=rationals(rng,(3,m))
        H=sp.Matrix(np.vstack([np.ones(m,dtype=int),phi]).tolist())
        rank=H.rank(); d0=sp.Matrix(np.vstack([np.ones(m,dtype=int),G]).tolist()).rank()-1
        for g in list(G)+[phi[0]]:
            width,P,Q=pair_width(g,phi); error,_=approximation(g,phi)
            contained=H.col_join(sp.Matrix([list(g)])).rank()==rank
            c.eq(width,2*error,'sharp unrestricted duality',dict(phi=phi,g=g,P=P,Q=Q),True)
            c.eq(width<=TOL,contained,'span criterion versus primal separation',dict(phi=phi,g=g))
            maxdual=max(maxdual,abs(width-2*error))
        c.check(0<=d0<=min(3,m-1),'minimal feature count exact rank',dict(G=G,d0=d0))
        b=rationals(rng,(d,),6)-F(1,2); r=rationals(rng,(m,),12)-F(1,2)
        scale=max(F(1),2*(sum(abs(x) for x in b)+max(abs(x) for x in r)))
        b=b/scale; r=r/scale
        g=F(1,2)+b@phi+r; B=sum(abs(x) for x in b); tau=max(abs(x) for x in r)
        c.check(min(g)>=0 and max(g)<=1,'transfer response lies in [0,1]',dict(g=g,b=b,r=r))
        # Maximize (P-Q)g - B*t - 2tau with |feature difference| <= t.
        f=np.asarray(phi,float); a=np.c_[f,-f,-np.ones(d)]
        a2=np.c_[-f,f,-np.ones(d)]
        opt=lp(np.r_[-np.asarray(g,float),np.asarray(g,float),float(B)],
               A_ub=np.r_[a,a2],b_ub=np.zeros(2*d),
               A_eq=[np.r_[np.ones(m),np.zeros(m+1)],np.r_[np.zeros(m),np.ones(m),0]],b_eq=[1,1])
        score=-opt.fun-2*float(tau); transferbest=max(transferbest,score)
        c.le(score,0,'optimized transfer inequality',dict(phi=phi,b=b,r=r,solution=opt.x),True)
        c.count()
    for m in range(2,9):
        P,Q=dist(rng,m),dist(rng,m); gamma=F(3,5); a=F(1,5)
        events=list(it.product((0,1),repeat=m))
        lhs=max(abs(sum((P[j]-Q[j])*(a+gamma*e[j]) for j in range(m))) for e in events)
        c.eq(lhs,gamma*sum(abs(x) for x in P-Q)/2,'all-event total variation',dict(P=P,Q=Q))
        c.eq(sp.Matrix([[1]*m]+[list(e) for e in events]).rank()-1,m-1,'event response dimension')
    c.data['details']=dict(max_duality_residual=maxdual,adversarial_transfer_best_lhs_minus_rhs=transferbest)
    return c.finish()


def check_C6(quick=False):
    c=Claim(6,'Lemma 10','Exact symbolic tree expansion; randomized and local leaf/context mutations','adversarial')
    rng=c.rng; best=F(0); witness=None
    for i in range(10 if quick else 90):
        k=int(rng.integers(1,4)); N=int(rng.integers(1,6)); ps=sp.symbols(f'p0:{k}')
        t=tree(rng,k,N)
        for j in range(3 if quick else 12):
            candidate=tree(rng,k,N) if j%4==0 else mutate_tree(t,rng,k)
            poly=poly_dict(tree_expr(candidate,ps),ps)
            norm=sum(abs(x) for x in poly.values()); ratio=norm/F(3**N)
            c.le(max(map(sum,poly)),N,'tree total degree',dict(tree=candidate,N=N))
            c.le(norm,3**N,'tree coefficient bound',dict(tree=candidate,poly=poly,N=N))
            for p in rationals(rng,(3,k)):
                value=poly_value(poly,p)
                c.check(0<=value<=1,'tree bounded value',dict(tree=candidate,p=p,value=value))
            if j==0 or ratio>localbest:
                t,localbest=candidate,ratio
            if ratio>best:
                best,witness=ratio,dict(tree=candidate,N=N,poly=poly)
            c.count()
    c.data['best_ratio']=float(best); c.data['details']=dict(exact_best_ratio=best,witness=witness)
    return c.finish()


def mutate_tree(t,rng,k):
    if t[0]=='reward':
        return ('reward',F(int(rng.integers(2))))
    if rng.random()<.3:
        return (t[0],int(rng.integers(k)) if t[0]=='action' else F(int(rng.integers(1,4)),4),t[2],t[3])
    side=int(rng.integers(2)); v=list(t); v[2+side]=mutate_tree(v[2+side],rng,k)
    return tuple(v)


def matched_pair(points, features, rng):
    H=sp.Matrix([[sp.Rational(x) for x in row] for row in features])
    null=H.nullspace()
    if not null:
        raise RuntimeError('Moment support has no null space')
    v=np.array([F(x) for x in null[int(rng.integers(len(null)))]],object)
    P=dist(rng,len(points)); a=min(P[i]/(2*abs(v[i])) for i in range(len(P)) if v[i])
    return P,P+a*v,v


def check_C7(quick=False):
    c=Claim(7,'Theorem 11','Exact null-space laws, independent-block moment convolution and symbolic monomial tasks')
    rng=c.rng; cases=[]
    for i in range(4 if quick else 24):
        k=1+i%2; N=1+i%3; grid=[F(j,N+1) for j in range(N+2)]
        points=list(it.product(grid,repeat=k)); aa=alphas(k,N)
        features=[[math.prod(p[j]**a[j] for j in range(k)) for p in points] for a in aa]
        P,Q,v=matched_pair(points,features,rng)
        R=1+i%5; ws=dist(rng,R+1); p0=list(rationals(rng,(k,)))
        ps=sp.symbols(f'p0:{k}'); poly=poly_dict(tree_expr(tree(rng,k,N),ps),ps)
        mp=affine_moments(points,P,ws,p0,N); mq=affine_moments(points,Q,ws,p0,N)
        inst=dict(points=points,P=P,Q=Q,weights=ws,p0=p0,poly=poly)
        for a in aa:
            c.eq(moment(points,P,a),moment(points,Q,a),'matched input joint moment',inst)
        c.eq(expected_poly(poly,mp),expected_poly(poly,mq),'affine sufficiency',inst)
        cases.append(dict(k=k,N=N,R=R)); c.count()
    for k,N in ((1,4),(2,3),(3,2)):
        ps=sp.symbols(f'p0:{k}'); aa=alphas(k,N)
        points=list(it.product([F(j,N) for j in range(N+1)],repeat=k))
        rows=[[math.prod(p[j]**a[j] for j in range(k)) for p in points] for a in aa]
        for a in aa:
            t=('reward',F(1))
            for j in range(k):
                for _ in range(a[j]):
                    t=('action',j,('reward',F(0)),t)
            expr=tree_expr(t,ps)
            c.eq(expr,math.prod(ps[j]**a[j] for j in range(k)),'necessity task evaluator',dict(alpha=a))
            if not sum(a):
                continue  # total mass cannot differ between probability laws
            idx=aa.index(a); other=rows[:idx]+rows[idx+1:]
            null=sp.Matrix(other).nullspace()
            vector=next(v for v in null if (sp.Matrix([rows[idx]])*v)[0]!=0)
            v=np.array([F(x) for x in vector],object); P=np.full(len(points),F(1,len(points)),object)
            scale=min(P[j]/(2*abs(v[j])) for j in range(len(v)) if v[j]); Q=P+scale*v
            for j,b in enumerate(aa):
                gap=moment(points,P,b)-moment(points,Q,b)
                c.check(gap!=0 if b==a else gap==0,'only target moment differs',dict(target=a,alpha=b,points=points,P=P,Q=Q))
            c.check(sum(P[j]*F(expr.subs(dict(zip(ps,points[j])))) for j in range(len(P))) !=
                    sum(Q[j]*F(expr.subs(dict(zip(ps,points[j])))) for j in range(len(Q))),
                    'necessity learned return separates',dict(alpha=a,P=P,Q=Q,points=points))
            c.count()
    pts=[(F(0),F(0)),(F(0),F(1)),(F(1),F(0)),(F(1),F(1))]
    P=[F(1,2),0,0,F(1,2)]; Q=[0,F(1,2),F(1,2),0]
    c.eq(moment(pts,P,(1,1)),F(1,2),'two-context cross moment P')
    c.eq(moment(pts,Q,(1,1)),F(0),'two-context cross moment Q')
    c.data['details']=dict(affine_configurations=cases,constant_monomial='mass-one tautology; no unequal-mass probability witness')
    return c.finish()


def check_C8(quick=False):
    c=Claim(8,'Theorem 12','Exact finite differences and all multiindices for binary parity')
    rows=[]
    for N in range(1,15):
        pts,P,Q=law_scalar(N)
        c.eq(sum(P),F(1),'scalar P mass'); c.eq(sum(Q),F(1),'scalar Q mass')
        for ell in range(N):
            c.eq(moment(pts,P,(ell,)),moment(pts,Q,(ell,)),'lower scalar moment',dict(N=N,ell=ell))
        delta=moment(pts,P,(N,))-moment(pts,Q,(N,))
        target=F((-1)**N*math.factorial(N),2**(N-1)*N**N)
        c.eq(delta,target,'finite difference formula',dict(N=N,points=pts,P=P,Q=Q))
        rows.append(dict(N=N,delta=delta)); c.count()
    for N in range(1,9 if not quick else 5):
        pts,P,Q=law_parity(N); cache={}
        for a in alphas(N,N-1):
            mask=tuple(int(x>0) for x in a)
            if mask not in cache:
                cache[mask]=(moment(pts,P,mask),moment(pts,Q,mask))
            c.eq(*cache[mask],'all lower joint multiindices',dict(N=N,alpha=a))
        values=[F(int(sum(p)%2==0)) for p in pts]
        c.eq(sum(w*v for w,v in zip(P,values)),F(1),'even law value')
        c.eq(sum(w*v for w,v in zip(Q,values)),F(0),'odd law value')
        for p in rationals(c.rng,(4,N)):
            enumerated=sum(math.prod(p[j] if b[j] else 1-p[j] for j in range(N)) for b in pts if sum(b)%2==0)
            formula=(1+math.prod(1-2*x for x in p))/2
            c.eq(enumerated,formula,'parity action enumeration',dict(p=p,N=N))
        c.count()
    c.data['details']=dict(scalar=rows,parity_all_multiindices=True)
    return c.finish()


def check_C9(quick=False):
    c=Claim(9,'Theorem 13','Exact integer-grid rank; removed-feature LPs; event separation certificates','adversarial')
    configs=[(k,N) for k in range(1,4) for N in range(1,6)]+[(4,N) for N in range(1,4)]
    if quick: configs=[(1,3),(2,2),(3,2),(4,1)]
    rows=[]; dropped=0
    for k,N in configs:
        grid=list(it.product(range(N+1),repeat=k)); aa=alphas(k,N)
        integer=np.array([[math.prod(x[j]**a[j] for j in range(k)) for x in grid] for a in aa],object)
        rank=sp.Matrix(integer.tolist()).to_DM().rank()
        c.eq(rank,math.comb(k+N,N),'exact evaluation rank',dict(k=k,N=N))
        # Row rescaling does not change rational rank; bounded features help LP conditioning.
        table=np.array([[float(F(int(v),N**sum(a))) for v in row] for a,row in zip(aa,integer)])
        for j,a in enumerate(aa):
            if not sum(a): continue
            features=np.delete(table,j,axis=0)
            width,P,Q=pair_width(table[j],features)
            c.check(width>TOL,'dropping each nonconstant feature separates',dict(k=k,N=N,alpha=a,width=width,P=P,Q=Q))
            dropped+=1
        rows.append(dict(k=k,N=N,rank=rank)); c.count()
    barriers=[]
    for k in range(1,5):
        m=2**k
        for rep in range(2 if quick else 8):
            d=int(c.rng.integers(0,m-1)); phi=rationals(c.rng,(d,m))
            H=sp.Matrix(np.vstack([np.ones(m,dtype=int),phi]).tolist())
            vector=H.nullspace()[0]
            event=np.array([int(bool(x>0)) for x in vector])
            width,P,Q=pair_width(event,phi)
            c.eq(width,1,'binary barrier event achieves global upper bound 1',dict(k=k,phi=phi,event=event,P=P,Q=Q),True)
            # Full event enumeration for k<=3; for k=4, attaining the universal
            # [0,1] upper bound certifies the same global optimum without 65536 LPs.
            if k<=3 and rep==0:
                maximum=max(pair_width(e,phi)[0] for e in it.product((0,1),repeat=m))
                c.eq(maximum,1,'enumerated event optimum',floating=True)
            barriers.append(dict(k=k,d=d,width=width))
        while True:
            full=rationals(c.rng,(m-1,m))
            if sp.Matrix(np.vstack([np.ones(m,dtype=int),full]).tolist()).rank()==m: break
        for g in list(np.eye(m))+[c.rng.random(m)]:
            width,_,_=pair_width(g,full)
            c.eq(width,0,'full generic interface discrepancy',dict(k=k,features=full,g=g),True)
        c.count()
    c.data['details']=dict(grid_ranks=rows,dropped_features=dropped,binary_barriers=barriers,
                           event_optimization='All events k<=3; separating event attains universal upper bound for k=4.')
    return c.finish()


def check_C10(quick=False):
    c=Claim(10,'Theorem 14','Exact independent-block convolution; rational local-law hill-climb','adversarial')
    rng=c.rng; best=F(0); best_instance=None; rows=[]
    for trial in range(4 if quick else 24):
        k=1+trial%2; N=1+trial%3; R=[1,2,5,10,25,50][trial%6]
        points=list(it.product((F(0),F(1,2),F(1)),repeat=k))
        P,Q=dist(rng,len(points)),dist(rng,len(points)); ws=dist(rng,R+1)
        p0=list(rationals(rng,(k,))); ps=sp.symbols(f'p0:{k}')
        poly=poly_dict(tree_expr(tree(rng,k,N),ps),ps)
        CJ=sum(abs(v)*sum(a) for a,v in poly.items())
        local=-1
        for step in range(3 if quick else 12):
            candidate=Q.copy()
            if step:
                a,b=rng.choice(len(Q),2,replace=False); move=min(candidate[a],F(1,20))
                candidate[a]-=move; candidate[b]+=move
            delta=max(abs(moment(points,P,a)-moment(points,candidate,a)) for a in alphas(k,N) if sum(a))
            mp=affine_moments(points,P,ws,p0,N); mq=affine_moments(points,candidate,ws,p0,N)
            gap=abs(expected_poly(poly,mp)-expected_poly(poly,mq)); bound=CJ*delta
            inst=dict(points=points,P=P,Q=candidate,weights=ws,p0=p0,poly=poly,N=N,R=R)
            c.le(gap,bound,'round-independent bound',inst)
            c.le(bound,N*3**N*delta,'coefficient envelope',inst)
            ratio=gap/bound if bound else F(0)
            if ratio>local: Q,local=candidate,ratio
            if ratio>best: best,best_instance=ratio,inst
            c.count()
        rows.append(dict(R=R,gap=gap,CJ=CJ,delta=delta,bound=bound))
    # Fixed law/task across all horizons: only the learner changes, bound is fixed.
    points=[(F(0),),(F(1),)]; P=[F(1,4),F(3,4)]; Q=P[::-1]
    sweep=[]
    for R in (1,2,5,10,25,50):
        ws=[F(0)]+[F(1,R)]*R
        mp=affine_moments(points,P,ws,[F(1,2)],2); mq=affine_moments(points,Q,ws,[F(1,2)],2)
        gap=abs(mp[(2,)]-mq[(2,)])
        c.le(gap,F(1),'fixed task bound for each R',dict(R=R))
        sweep.append(dict(R=R,gap=gap,bound=F(1)))
    c.data['best_ratio']=float(best)
    c.data['details']=dict(search=rows,best_exact_ratio=best,best_instance=best_instance,fixed_problem_R_sweep=sweep)
    return c.finish()


def average_expression(poly, points, weights):
    x=sp.Symbol('inverse_n'); out=sp.Integer(0)
    for a,c in poly.items():
        degree=sum(a); coefficients=iid_sum_coeffs(a,points,weights)
        for b,coef in enumerate(coefficients):
            if coef:
                out += sp.Rational(c*coef)*x**(degree-b)*math.prod(1-j*x for j in range(b))
    return sp.Poly(sp.expand(out),x)


def check_C11(quick=False):
    c=Claim(11,'Theorem 15 and equation (31)','Exact sample-index multinomial expansion, cumulants computed independently')
    rows=[]
    for N in range(2,9):
        pts,P,Q=law_scalar(N); cp=iid_sum_coeffs((N,),pts,P); cq=iid_sum_coeffs((N,),pts,Q)
        delta=moment(pts,P,(N,))-moment(pts,Q,(N,))
        for n in range(1,31):
            gap=average_moment(cp,N,n)-average_moment(cq,N,n)
            c.eq(gap,delta/n**(N-1),'exact averaging identity',dict(N=N,n=n)); c.count()
    for k,r in it.product((1,2),(2,3,4)):
        scalar,P0,Q0=law_scalar(r)
        if k==1:
            pts,P,Q=scalar,P0,Q0
        else:
            pts=[(x[0],(x[0]+b)/2) for x in scalar for b in (F(0),F(1))]
            P=[w/2 for w in P0 for _ in range(2)]; Q=[w/2 for w in Q0 for _ in range(2)]
        ps=sp.symbols(f'p0:{k}'); mu=[moment(pts,P,tuple(int(j==i) for j in range(k))) for i in range(k)]
        J=(sum(ps)/k)**(r+1); poly=poly_dict(J,ps)
        gap=average_expression(poly,pts,P)-average_expression(poly,pts,Q)
        for a in alphas(k,r-1):
            if sum(a): c.eq(cumulant(a,pts,P),cumulant(a,pts,Q),'lower cumulants match',dict(k=k,r=r,alpha=a))
        contraction=F(0)
        for a in alphas(k,r):
            if sum(a)!=r: continue
            deriv=J
            for j in range(k): deriv=sp.diff(deriv,ps[j],a[j])
            coefficient=F(deriv.subs(dict(zip(ps,mu))))/math.prod(math.factorial(v) for v in a)
            contraction+=coefficient*(cumulant(a,pts,P)-cumulant(a,pts,Q))
        c.eq(F(gap.nth(r-1)),contraction,'predicted tensor contraction',dict(k=k,r=r,gap=gap.as_expr()))
        for power in range(r-1): c.eq(gap.nth(power),0,'no slower order',dict(k=k,r=r,power=power))
        envelope=sum(abs(F(gap.nth(j))) for j in range(r,gap.degree()+1)) if not gap.is_zero else F(0)
        worst=F(0)
        for n in range(1,221):
            actual=F(gap.eval(F(1,n))); residual=abs(n**(r-1)*actual-contraction)*n
            c.le(residual,envelope,'scaled asymptotic residual bounded for n up to 220',dict(k=k,r=r,n=n,gap=gap.as_expr()))
            worst=max(worst,residual); c.count()
        rows.append(dict(k=k,r=r,mean=mu,predicted_coefficient=contraction,
                         exact_gap_polynomial_in_inverse_n=gap.as_expr(),
                         residual_times_n_bound=envelope,worst_residual_times_n=worst,
                         scaled_gap_n220=F(gap.eval(F(1,220)))*220**(r-1)))
    for r in (2,3,4):
        pts,P,Q=law_scalar(r); p=sp.Symbol('p')
        J=(p-sp.Rational(1,2))**(r+2)
        gap=average_expression(poly_dict(J,(p,)),pts,P)-average_expression(poly_dict(J,(p,)),pts,Q)
        c.eq(gap.nth(r-1),0,'vanishing contraction faster rate',dict(r=r,gap=gap.as_expr()))
        c.check(not gap.is_zero,'nontrivial faster-rate witness',dict(r=r,gap=gap.as_expr()))
        rows.append(dict(vanishing_contraction_r=r,gap=gap.as_expr()))
    pts=[(F(0),),(F(1),)]; P=[F(1,4),F(3,4)]; Q=P[::-1]; poly={(3,):F(1)}
    unequal=average_expression(poly,pts,P)-average_expression(poly,pts,Q)
    target=F(3,4)**3-F(1,4)**3
    c.eq(unequal.nth(0),target,'unequal means limit')
    correction=sum(abs(F(unequal.nth(j))) for j in range(1,unequal.degree()+1))
    for n in range(1,221): c.le(abs(F(unequal.eval(F(1,n)))-target)*n,correction,'unequal means O(1/n)',dict(n=n))
    c.data['details']=dict(cumulant_cases=rows,unequal_means_gap=unequal.as_expr(),unequal_means_limit=target)
    return c.finish()


def check_C12(quick=False):
    c=Claim(12,'Equation (32) and Proposition 16','Symbolic Chebyshev identities, exact tails, full vertex minimax LP','adversarial')
    p=sp.Symbol('p'); best=0.; rows=[]
    for N in range(1,17):
        coeff=[F(math.comb(2*N,N),4**N)]+[F(2*math.comb(2*N,N-j),4**N) for j in range(1,N+1)]
        expansion=sum(sp.Rational(a)*sp.chebyshevt(j,2*p-1) for j,a in enumerate(coeff))
        c.eq(sp.expand(expansion-p**N),0,'Chebyshev polynomial identity',dict(N=N))
        grid=np.linspace(0,1,3001)
        for q in range(N):
            tail=sum(coeff[q+1:]); h=sum(float(coeff[j])*np.polynomial.chebyshev.chebval(2*grid-1,[0]*j+[1]) for j in range(q+1))
            maxerr=float(np.max(abs(grid**N-h)))
            c.eq(maxerr,float(tail),'dense supremum including p=1',dict(N=N,q=q),True)
            exacttail=sum(F(math.comb(2*N,b),4**N) for b in range(2*N+1) if abs(b-N)>q)
            c.eq(tail,exacttail,'Chebyshev tail equals exact binomial tail',dict(N=N,q=q))
            bound=2*math.exp(-q*q/N); c.le(float(tail),bound,'tail inequality',dict(N=N,q=q),True)
            best=max(best,float(tail)/bound); c.count()
        rows.append(dict(N=N,full_identity=True))
    for N in range(1,6):
        vertices=list(it.product((0,1),repeat=N)); g=[int(sum(v)%2==0) for v in vertices]
        phi=[[math.prod(v[j]**a[j] for j in range(N)) for v in vertices]
             for a in alphas(N,N-1) if sum(a)]
        error,_=approximation(g,phi)
        c.eq(error,.5,'best lower-degree parity approximation',dict(N=N),True)
        for p0 in rationals(c.rng,(10,N)):
            parity=(1+math.prod(1-2*x for x in p0))/2
            c.le(abs(parity-F(1,2)),F(1,2),'constant bound over cube',dict(p=p0))
    c.data['details']=dict(exhaustive_tail_best_ratio=best,N_range=[1,16],parity_minimax_error=.5)
    return c.finish()


def check_C13(quick=False):
    c=Claim(13,'Theorem 17 and smooth SGD expansion (35)',
            'Exact symbolic nonlinear pullback, block moments, cubic Taylor identities, logistic segment bound','adversarial')
    rng=c.rng; degree_rows=[]
    for trial in range(3 if quick else 12):
        R=1+trial%3; N=1+trial%2; zs=sp.symbols(f'z0:{R}')
        aa=[int(rng.integers(1,3)) for _ in range(R)]; bb=[int(rng.integers(1,3)) for _ in range(R)]
        p=sp.Rational(1,2)
        for i in range(R):
            lam=sp.Rational(int(rng.integers(1,4)),4)
            p=sp.expand(lam*p**aa[i]+(1-lam)*zs[i]**bb[i])
        response=sp.Poly(sp.expand(p**N),*zs)
        Ds=[N*bb[i]*math.prod(aa[i+1:]) for i in range(R)]
        laws=[law_scalar(D+1) for D in Ds]
        ep=eq=F(0)
        for powers,coef in response.terms():
            vp=vq=F(coef)
            for i,power in enumerate(powers):
                c.le(power,Ds[i],'declared per-block degree',dict(a=aa,b=bb,N=N,powers=powers))
                points,P,Q=laws[i]; vp*=moment(points,P,(power,)); vq*=moment(points,Q,(power,))
            ep+=vp; eq+=vq
        c.eq(ep,eq,'blockwise matching preserves nonlinear value',dict(response=response.as_expr(),D=Ds))
        degree_rows.append(dict(a=aa,b=bb,N=N,D=Ds,actual_degrees=[response.degree(z) for z in zs])); c.count()
    amplification=[]
    for R in range(1,5):
        degree=2**(R-1); z=sp.Symbol('z'); p=z
        for _ in range(R-1): p=sp.expand(p*p)
        c.eq(p,z**degree,'squaring witness pullback')
        pts,P,Q=law_scalar(degree); gap=moment(pts,P,(degree,))-moment(pts,Q,(degree,))
        c.check(gap!=0,'amplified order separation',dict(R=R,degree=degree,gap=gap))
        amplification.append(dict(R=R,degree=degree,gap=gap))
    points=[(F(0),),(F(1,2),),(F(3,4),)]; P=[F(1,3),F(0),F(2,3)]; Q=[F(0),F(1),F(0)]
    c.eq(moment(points,P,(1,)),moment(points,Q,(1,)),'threshold equal mean')
    c.eq(sum(w for (x,),w in zip(points,P) if x>F(1,2)),F(2,3),'threshold P value')
    c.eq(sum(w for (x,),w in zip(points,Q) if x>F(1,2)),0,'threshold Q value')
    th=sp.symbols('t0:3'); etas=[F(1,100),F(1,10),F(1,2),F(1)]
    best=0.; taylor_rows=[]
    for trial in range(3 if quick else 15):
        coeff=rationals(rng,(len(alphas(3,3)),),6)-F(1,2)
        V=sum(sp.Rational(b)*math.prod(t**a for t,a in zip(th,alpha)) for b,alpha in zip(coeff,alphas(3,3)))
        theta=rationals(rng,(3,))-F(1,2); grads=rationals(rng,(6,3))-F(1,2)
        P,Q=dist(rng,6),dist(rng,6); mu=(P-Q)@grads
        MM=sum((P[i]-Q[i])*np.outer(grads[i],grads[i]) for i in range(6))
        at=dict(zip(th,theta)); grad=np.array([F(sp.diff(V,t).subs(at)) for t in th],object)
        H=np.array([[F(sp.diff(V,t,u).subs(at)) for u in th] for t in th],object)
        # Frobenius norm bounds the multilinear operator norm. Squaring the
        # inequality keeps the cubic remainder check entirely rational.
        tensor=[F(sp.diff(V,th[i],th[j],th[k])) for i,j,k in it.product(range(3),repeat=3)]
        m3sq=sum(x*x for x in tensor)
        for eta in etas:
            exact=sum((P[i]-Q[i])*F(V.subs(dict(zip(th,theta-eta*grads[i])))) for i in range(6))
            first=-eta*(grad@mu)+eta**2*sum((H*MM).flat)/2
            remainder=abs(exact-first)
            # sqrt(||G||²)^3 evaluated numerically for the operator-norm bound;
            # the actual difference and first two terms remain exact rationals.
            norms=np.linalg.norm(np.asarray(grads,float),axis=1)
            bound=math.sqrt(float(m3sq))*float(eta)**3/6*float(np.asarray(P+Q,float)@norms**3)
            c.le(float(remainder),bound,'cubic operator-norm remainder',dict(V=V,theta=theta,gradients=grads,P=P,Q=Q,eta=eta),True)
            ratio=float(remainder)/bound if bound else 0; best=max(best,ratio); c.count()
        # Logistic of a linear form: |sigmoid'''|<=1/8 globally, so
        # M3 <= ||w||^3/8 on every relevant segment, without sampling a supremum.
        w=np.asarray(rationals(rng,(3,))-F(1,2),float); t=np.asarray(theta,float); G=np.asarray(grads,float)
        pp,qq=np.asarray(P,float),np.asarray(Q,float); s=expit(w@t)
        gg=s*(1-s)*w; hh=s*(1-s)*(1-2*s)*np.outer(w,w); m3=np.linalg.norm(w)**3/8
        for eta0 in etas:
            eta=float(eta0); exact=(pp-qq)@expit((t-eta*G)@w)
            first=-eta*gg@np.asarray(mu,float)+eta**2/2*np.sum(hh*np.asarray(MM,float))
            rem=abs(exact-first); bound=m3*eta**3/6*((pp+qq)@np.linalg.norm(G,axis=1)**3)
            c.le(rem,bound,'logistic global-third-derivative remainder',dict(w=w,theta=t,G=G,P=P,Q=Q,eta=eta),True)
            best=max(best,rem/bound if bound else 0)
            taylor_rows.append(dict(family='logistic',eta=eta,remainder=rem,bound=bound))
    c.data['best_ratio']=best
    c.data['details']=dict(degrees=degree_rows,amplification=amplification,logistic=taylor_rows,
                           adversarial_search='Independent random restarts over cubics/laws and all declared step sizes; see local search below.')
    # Adversarial perturbations of step size for the last logistic configuration.
    opt=minimize_scalar(lambda e: -(abs((pp-qq)@expit((t-e*G)@w)-(-e*gg@np.asarray(mu,float)+e*e/2*np.sum(hh*np.asarray(MM,float))))
                                   /(m3*e**3/6*((pp+qq)@np.linalg.norm(G,axis=1)**3)+1e-30)),
                        bounds=(.02,1),method='bounded')
    c.le(-opt.fun,1,'adversarial smooth remainder ratio',floating=True)
    c.data['details']['adversarial_logistic_ratio']=-opt.fun
    return c.finish()


def check_C14(quick=False):
    c=Claim(14,'Theorem 18','Direct probability primal, independent oscillation and centered-uniform dual LPs','adversarial')
    rng=c.rng; worst=0.; cases=[]
    for i in range(8 if quick else 220):
        m=int(rng.integers(3,13)); d=int(rng.integers(1,m))
        g=rationals(rng,(m,)); phi=rationals(rng,(d,m)); eps=rationals(rng,(d,),10)
        eps[i%d]=F(0)
        width,P,Q=pair_width(g,phi,eps); dual=noisy_dual(g,phi,eps); centered=noisy_dual(g,phi,eps,False)
        inst=dict(g=g,phi=phi,eps=eps,P=P,Q=Q)
        c.eq(width,dual,'noisy oscillation dual',inst,True)
        c.eq(width,centered,'noisy uniform dual',inst,True)
        c.le(width,float(max(g)-min(g)),'oscillation upper bound',inst,True)
        zero,_,_=pair_width(g,phi,np.zeros(d)); approx,_=approximation(g,phi)
        c.eq(zero,2*approx,'zero-tolerance reduction',inst,True)
        large,_,_=pair_width(g,phi,np.ones(d)*2)
        c.eq(large,float(max(g)-min(g)),'large-tolerance saturation',inst,True)
        worst=max(worst,abs(width-dual),abs(width-centered)); c.count()
    c.data['details']=dict(max_duality_residual=worst,adversarial_method='Primal LP globally maximizes response difference per feature instance.')
    return c.finish()


def coverage_witness():
    n=10000; delta=.05; radius=2*math.sqrt(math.log(4/delta)/(2*n))
    return dict(audited_cells=[0],certified_states=[0,1],initial_state=1,
                message_alphabet=[0,1],feature=[0,1],response_at_both_states=[0,1],
                reference_P_y1=[F(0),F(0)],candidate_P_y1=[F(0),F(1)],
                update='U(z,y)=y',terminal=[0,1],B=[F(1)],tau=[F(0)],rho=F(0),
                audit_n_reference=n,audit_n_candidate=n,delta=delta,
                audit_empirical_discrepancy=F(0),audit_event_probability=F(1),
                true_learning_gap=F(1),claimed_bound39=radius)


def exact_count_learning(q, R=3, power=1):
    """A real finite count-state update, including unreachable-state completion."""
    K=np.full((R,R+1,R+1),F(0),object)
    for t,z in it.product(range(R),range(R+1)):
        K[t,z,z]+=1-q
        K[t,z,min(z+1,R)]+=q
    terminal=np.array([F(z,R)**power for z in range(R+1)],object)
    initial=np.array([F(1)]+[F(0)]*R,object)
    V=continuations(K,terminal)
    return initial@V[0],K,V


def check_C15(quick=False):
    c=Claim(15,'Theorem 19, equations (37)–(40), coverage-condition probe',
            'Independent finite-feature Monte Carlo and exact state-dependent coverage counterexample','monte_carlo')
    rng=c.rng; rows=[]
    count_process_checks=[]
    for q in (F(1,4),F(1,2),F(2,3),F(3,4)):
        for power in (1,2):
            value,K,V=exact_count_learning(q,power=power)
            expected=q if power==1 else q*q+q*(1-q)/3
            c.eq(value,expected,'exact finite count-state training value',dict(q=q,power=power,kernels=K))
            if power==1:
                for t in range(3):
                    for z in range(t+1):
                        c.eq(V[t+1][z+1]-V[t+1][z],F(1,3),
                             'exact covered response c+y/3, tau=0',dict(q=q,t=t,z=z))
            count_process_checks.append(dict(q=q,power=power,value=value))
    configs=[(2,2,2,40,50,.1),(4,3,3,100,80,.05),(3,1,1,1000,1200,.2),(2,2,2,8,8,.4)]
    trials=250 if quick else 3000
    for M,L,d,n0,ns,delta in configs:
        p0=rng.uniform(.15,.85,(L,d)); pj=rng.uniform(.1,.9,(M,L,d))
        D=np.max(abs(pj-p0),axis=(1,2)); e0=math.sqrt(math.log(4*L*d/delta)/(2*n0)); es=math.sqrt(math.log(4*M*L*d/delta)/(2*ns))
        ref=rng.binomial(n0,p0,size=(trials,L,d))/n0
        cand=rng.binomial(ns,pj,size=(trials,M,L,d))/ns
        hat=np.max(abs(cand-ref[:,None]),axis=(2,3)); event=np.all(abs(hat-D)<=e0+es,axis=1)
        chosen=np.argmin(hat,axis=1); selected=D[chosen]
        rate=float(event.mean()); se=math.sqrt(rate*(1-rate)/trials)
        c.check(rate>=1-delta,'audit event frequency (37)',dict(config=[M,L,d,n0,ns,delta],rate=rate,se=se))
        violation=selected-(D.min()+2*(e0+es))
        c.le(float(np.max(violation[event])),0,'selection certificate on audit event (38)',floating=True)
        # Actual three-round finite count process: terminal count/3 has response
        # c+y/3 in each state; all three rounds/cells use the audited Bernoulli law.
        fresh_gap=np.abs(pj[chosen,0,0]-p0[0,0])
        bound=hat[np.arange(trials),chosen]+e0+es
        c.le(float(np.max((fresh_gap-bound)[event])),0,'fresh finite-process certificate (39)',floating=True)
        # For Bernoulli blocks and J=p², exact n=3 average value.
        def value(q): return q*q+q*(1-q)/3
        affine_gap=np.abs(value(pj[chosen,0,0])-value(p0[0,0]))
        c.le(float(np.max((affine_gap-2*bound)[event])),0,'affine certificate (40)',floating=True)
        rows.append(dict(M=M,L=L,d=d,n0=n0,nS=ns,delta=delta,trials=trials,
                         audit_event_frequency=rate,SE=se,e0=e0,eS=es,
                         worst_selection_excess=float(violation[event].max()),
                         worst_fresh_certificate_excess=float((fresh_gap-bound)[event].max())))
        c.count(trials)
    witness=coverage_witness()
    c.le(float(witness['true_learning_gap']),witness['claimed_bound39'],
         'equation (39) with response coverage but unaudited state-dependent provider law',witness,True)
    c.data['notes'].append('FAIL is retained: response approximation at certified states alone does not identify provider laws at unaudited states. Requires certified-state/cell law coverage or an exit penalty.')
    c.data['details']=dict(configurations=rows,exact_count_process=count_process_checks,coverage_probe=witness)
    return c.finish()


def categorical(rng, probabilities):
    return np.sum(rng.random(len(probabilities))[:,None]>np.cumsum(probabilities,axis=1),axis=1)


def adaptive_trial(rng, replicas, R, M, n, delta):
    S=4; rewards=np.linspace(0,1,S)
    baseline=np.array([.2,.3,.3,.2]); K0=np.empty((R,S,S)); candidates=np.empty((R,M,S,S))
    for t in range(R):
        for z in range(S):
            K0[t,z]=.35*np.eye(S)[z]+.65*baseline
            for j in range(M):
                candidates[t,j,z]=K0[t,z]
                # Include slightly negative, high-variance and positive candidates.
                shift=[-.018,-.045,.14][j%3]
                candidates[t,j,z,0]-=shift; candidates[t,j,z,-1]+=shift
    V=continuations(K0,rewards); states=np.ones(replicas,dtype=int)
    sumA=np.zeros(replicas); sumB=np.zeros(replicas); allpass=np.ones(replicas,bool)
    radius=math.sqrt(2*math.log(2*M/(delta/R))/n)
    for t in range(R):
        refprob=K0[t,states]@V[t+1]
        means=np.column_stack([candidates[t,j,states]@V[t+1]-refprob for j in range(M)])
        estimates=np.empty((replicas,M))
        for j in range(M):
            pc=means[:,j]+refprob
            # Antithetic Bernoulli deployments have both correct marginals.
            # Drawing final Bernoulli using K V integrates the full reference tail.
            plus=np.minimum(pc,1-refprob); minus=np.minimum(refprob,1-pc)
            pos=rng.binomial(n,plus)
            neg=rng.binomial(n-pos,np.divide(minus,1-plus,out=np.zeros_like(minus),where=plus<1))
            estimates[:,j]=(pos-neg)/n
        allpass &= np.all(abs(estimates-means)<=radius,axis=1)
        lcb=estimates-radius; choices=np.argmax(np.c_[np.zeros(replicas),lcb],axis=1)-1
        B=np.maximum(0,lcb.max(axis=1)); A=np.where(choices<0,0,means[np.arange(replicas),np.maximum(choices,0)])
        sumA+=A; sumB+=B
        probabilities=K0[t,states].copy()
        for j in range(M):
            mask=choices==j; probabilities[mask]=candidates[t,j,states[mask]]
        states=categorical(rng,probabilities)  # Fresh update, independent of audit.
    terminal=rewards[states]; diff=terminal-V[0][1]
    gain=float(diff.mean()); se=float(diff.std(ddof=1)/math.sqrt(replicas)); eb=float(sumB.mean())
    stat=dict(R=R,M=M,n=n,delta=delta,replicas=replicas,radius=radius,gain=gain,SE=se,
              expected_sum_B=eb,claim1_frequency=float(np.mean(sumA+TOL>=sumB)),
              audit_pass_frequency=float(allpass.mean()),conditional_gain=float(diff[allpass].mean()),
              conditional_expected_sum_B=float(sumB[allpass].mean()),
              conditional_SE=float(diff[allpass].std(ddof=1)/math.sqrt(allpass.sum())),
              initial_reference_value=float(V[0][1]),reference_kernels=K0,candidate_kernels=candidates,
              reference_continuations=V)
    return stat


def check_C16(quick=False):
    c=Claim(16,'Theorem 20, equations (44)–(45)',
            'Actual-state paired full-tail audits, fresh transitions, 100000 independent training runs','monte_carlo')
    rows=[]; replicas=3000 if quick else 100000
    for R,M,n,delta in [(1,2,8,.4),(2,2,12,.4),(3,3,20,.2),(6,3,64,.1),(2,3,1000,.2)]:
        row=adaptive_trial(c.rng,replicas,R,M,n,delta)
        c.le(-delta,row['gain']+4*row['SE'],'unconditional lower bound (44), four SE',row,True)
        c.le(row['expected_sum_B']-2*delta,row['gain']+4*row['SE'],'unconditional LCB lower bound (45), four SE',row,True)
        rate=row['claim1_frequency']; se=math.sqrt(rate*(1-rate)/replicas)
        c.le(1-delta,rate+4*se,'claim 1 predictable drift frequency',row,True)
        row['claim1_SE']=se; rows.append(row); c.count(replicas)
    # Exact point-estimate selection: reference success 1/2, candidate 12/25,
    # two antithetic paired contrasts; select iff their mean is positive.
    probabilities={-1:F(1,2),0:F(1,50),1:F(12,25)}
    selectprob=sum(pa*pb for a,pa in probabilities.items() for b,pb in probabilities.items() if a+b>0)
    point_gain=-F(1,50)*selectprob
    c.check(point_gain<0,'point-estimate selection negative expected gain',dict(contrasts=probabilities,selection_probability=selectprob,gain=point_gain))
    found=[r for r in rows if r['conditional_gain']+4*r['conditional_SE'] < r['conditional_expected_sum_B']-2*r['delta']]
    # A separate exact conditioning-bias witness: first reach a fair terminal
    # bit; at the next state audit two identical Bernoulli tails. At state 0
    # reward is deterministic, whereas at state 1 it has success 1/2.
    # n=1, delta=.4 gives e>1 (no failures), so use n=8 and compute failures.
    n=8; delta=.4; e=math.sqrt(2*math.log(2/delta)/n)
    fail=sum(F(math.comb(n,k),2**n) for k in range(n+1) if abs(F(2*k-n,n))>e)
    # State 0: return 0 and deterministic audit. State 1: return Bernoulli(1/2),
    # two identical antithetically paired tails; candidate never selected on G.
    conditional=F(1,2)*(1-fail)/(2-fail); unconditional=F(1,4)
    c.check(conditional<unconditional,'conditioning can bias terminal mean despite zero selected drifts',dict(failure_probability=fail,conditional=conditional,unconditional=unconditional))
    conditioning_witness=dict(n=n,delta=delta,radius=e,audit_failure_at_state1=fail,
                              conditional_value=conditional,unconditional_value=unconditional)
    # Adversarial one-round exact audit search. Optimize the (45) violation
    # over candidate success probabilities with rational local perturbations;
    # multinomial contrast probabilities are computed by exact convolution.
    best_score=-math.inf; best_search=None
    search_cache={}
    def audit_objective(prob,n,delta):
        key=(prob,n,delta)
        if key in search_cache: return search_cache[key]
        e=math.sqrt(2*math.log(2/delta)/n)
        plus=min(prob,F(1,2)); minus=min(F(1,2),1-prob)
        law={-1:minus,0:1-minus-plus,1:plus}; sums={0:F(1)}
        for _ in range(n):
            nxt={}
            for total,mass in sums.items():
                for value,weight in law.items():
                    nxt[total+value]=nxt.get(total+value,F(0))+mass*weight
            sums=nxt
        selected=sum(mass for total,mass in sums.items() if F(total,n)>e)
        gain=(prob-F(1,2))*selected
        expected_B=sum(float(mass)*max(0,float(F(total,n))-e) for total,mass in sums.items())
        score=expected_B-2*delta-float(gain)
        result=(score,gain,expected_B,e)
        search_cache[key]=result
        return result
    for n,delta in it.product((4,8,12),(0.1,0.2,0.4)):
        for restart in range(2 if quick else 5):
            prob=F(int(c.rng.integers(0,65)),64)
            for iteration in range(3 if quick else 10):
                options=sorted({max(F(0),min(F(1),prob+step)) for step in (-F(1,32),F(0),F(1,32))})
                prob=max(options,key=lambda q:audit_objective(q,n,delta)[0])
                score,gain,eb,e=audit_objective(prob,n,delta)
                inst=dict(candidate_success=prob,reference_success=F(1,2),n=n,delta=delta,
                          exact_gain=gain,expected_B=eb,radius=e)
                c.le(score,0,'adversarial exact one-round (45) search',inst,True)
                if score>best_score: best_score,best_search=score,inst
                c.count()
    c.eq(math.ceil(2*math.log(160)/.01),1016,'audit sample count')
    c.data['details']=dict(configurations=rows,point_estimate_exact_gain=point_gain,
                           conditioning_probe_found_45_violation=bool(found),
                           conditioning_probe_search_configurations=len(rows),
                           zero_drift_conditioning_witness=conditioning_witness,
                           adversarial_best_bound_minus_gain=best_score,adversarial_witness=best_search,
                           adversarial_unique_instances=len(search_cache),
                           replicas_required_per_candidate=1016)
    c.data['notes'].append('Conditional (45) analogue searched, no violation is asserted unless found; the exact additional witness establishes conditioning bias, not a violation of unconditional bounds.')
    return c.finish()


def coupling_width(H,mu,nu,allowed=None):
    H=np.asarray(H,float); a,b=H.shape
    eq=np.r_[np.kron(np.eye(a),np.ones((1,b))),np.kron(np.ones((1,a)),np.eye(b))]
    bounds=[(0,None) if allowed is None or allowed.flat[j] else (0,0) for j in range(a*b)]
    lo=lp(H.ravel(),A_eq=eq,b_eq=np.r_[mu,nu],bounds=bounds)
    hi=lp(-H.ravel(),A_eq=eq,b_eq=np.r_[mu,nu],bounds=bounds)
    return -hi.fun-lo.fun,hi.x.reshape(a,b),lo.x.reshape(a,b)


def check_C17(quick=False):
    c=Claim(17,'Proposition 21 and additive SGD null','Coupling polytope LP extrema, additive approximation LP, common-state gradient check','adversarial')
    rng=c.rng; slacks=[]
    for i in range(8 if quick else 320):
        a,b=map(int,rng.integers(2,6,size=2)); mu,nu=dist(rng,a),dist(rng,b)
        H=rationals(rng,(a,b)); av,bv=rationals(rng,(a,)),rationals(rng,(b,))
        width,hi,lo=coupling_width(H,mu,nu); zero,_,_=coupling_width(av[:,None]+bv[None,:],mu,nu)
        rectangle=H-H[:,[0]]-H[[0],:]+H[0,0]
        c.eq(zero,0,'additive coupling width zero',dict(mu=mu,nu=nu,a=av,b=bv),True)
        c.eq(width>TOL,bool(np.any(rectangle!=0)),'nonadditive iff positive coupling width',dict(H=H,mu=mu,nu=nu))
        phi=np.r_[np.kron(np.eye(a),np.ones((1,b))),np.kron(np.ones((1,a)),np.eye(b))]
        err,_=approximation(H.ravel(),phi)
        c.le(width,2*err,'optimized coupling inequality (46)',dict(H=H,mu=mu,nu=nu,hi=hi,lo=lo),True)
        slacks.append(2*err-width); c.count()
    H=np.array([[0,0],[0,1]]); allowed=np.eye(2,dtype=bool)
    width,_,_=coupling_width(H,[.5,.5],[.5,.5],allowed)
    c.eq(width,0,'forbidden pairing zero width',floating=True)
    c.check(H[0,0]+H[1,1]!=H[0,1]+H[1,0],'forbidden pairing response nonadditive')
    theta=F(1,3); eta=F(1,4); labels=[F(0),F(0),F(1),F(1)]
    packetsA=[[0,1],[2,3]]; packetsB=[[0,2],[1,3]]
    def grad_at(t,packet): return sum(t-labels[i] for i in packet)/len(packet)
    def common(packets): return theta-eta*sum(grad_at(theta,packet)/len(packets) for packet in packets)
    c.eq(common(packetsA),common(packetsB),'exact linear one-common-state additive null')
    def sequential(packets):
        t=theta
        for packet in packets: t-=eta*grad_at(t,packet)
        return t
    sequential_gap=sequential(packetsA)-sequential(packetsB)
    c.check(sequential_gap!=0,'two sequential updates regrouping witness')
    def logistic_common(packets):
        return float(theta)-float(eta)*sum(sum(expit(float(theta))-float(labels[i]) for i in packet)/len(packet)/len(packets) for packet in packets)
    c.eq(logistic_common(packetsA),logistic_common(packetsB),'logistic common-state additive null',floating=True)
    c.data['details']=dict(median_inequality_slack=float(np.median(slacks)),maximum_slack=max(slacks),
                           sequential_linear_parameter_gap=sequential_gap,forbidden_pairing_response=H)
    return c.finish()


def check_C18(quick=False):
    c=Claim(18,'Execution plan §13.1; Theorem 15','Exact independent-block multinomial expectations versus one-parent descendants')
    ns=[1,2,4,8,16,32,64,128,256]; rows=[]
    for family,N in [('scalar',2),('scalar',3),('scalar',4),('parity',2),('parity',3)]:
        pts,P,Q=law_scalar(N) if family=='scalar' else law_parity(N)
        k=len(pts[0]); ps=sp.symbols(f'p0:{k}')
        J=ps[0]**N if family=='scalar' else (1+math.prod(1-2*p for p in ps))/2
        poly=poly_dict(J,ps); independent=average_expression(poly,pts,P)-average_expression(poly,pts,Q)
        base=sum((a-b)*poly_value(poly,p) for p,a,b in zip(pts,P,Q))
        synthetic_points=[tuple((x+F(1,2))/2 for x in p) for p in pts]
        synthetic=sum((a-b)*poly_value(poly,p) for p,a,b in zip(synthetic_points,P,Q))
        c.eq(synthetic,base/2**N,'chosen postprocessing contraction',dict(family=family,N=N))
        for n in ns:
            iid=F(independent.eval(F(1,n)))
            replay=sum((a-b)*poly_value(poly,tuple(sum([x]*n)/n for x in p)) for p,a,b in zip(pts,P,Q))
            descendants=sum((a-b)*poly_value(poly,tuple(sum([x]*n)/n for x in p)) for p,a,b in zip(synthetic_points,P,Q))
            c.eq(iid,base/n**(N-1),'independent evidence exact decay',dict(family=family,N=N,n=n))
            c.eq(replay,base,'replay does not create independent blocks',dict(family=family,N=N,n=n))
            c.eq(descendants,synthetic,'synthetic descendants share one parent',dict(family=family,N=N,n=n))
            rows.append(dict(family=family,N=N,exposure=n,independent_evidence_iid=n,
                             independent_evidence_replay=1,independent_evidence_descendants=1,
                             iid_gap=iid,replay_gap=replay,descendant_gap=descendants)); c.count()
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for col,family in enumerate(('scalar','parity')):
        for N in sorted(set(r['N'] for r in rows if r['family']==family)):
            rr=[r for r in rows if r['family']==family and r['N']==N]; color=f'C{N-2}'
            for field,style,label in [('iid_gap','-','independent'),('replay_gap','--','replay'),('descendant_gap',':','descendants')]:
                y=[abs(float(r[field])) for r in rr]
                axs[0,col].loglog(ns,y,style,color=color,label=f'N={N}, {label}')
                x=ns if field=='iid_gap' else np.ones(len(ns))
                axs[1,col].loglog(x,y,style+'o',markersize=3,color=color,label=f'N={N}, {label}')
        axs[0,col].set(title=family,xlabel='Training exposure (blocks)',ylabel='Absolute learning-value gap')
        axs[1,col].set(xlabel='Independent parent evidence blocks',ylabel='Absolute learning-value gap')
        axs[0,col].legend(fontsize=7)
    savefig('C18_evidence_exposure')
    c.data['details']=dict(curves=rows,channel='Each descendant maps Z to (Z+1/2)/2; labels can be independently permuted inside blocks without changing this statistic.',
                           channel_scope='A constructed information-preserving descendant control, not a universal non-decay claim about every postprocessing channel.')
    return c.finish()


def check_C19(quick=False):
    c=Claim(19,'Execution plan §13.7','Rational response ranks; minimax LP approximation and independent discrepancy LPs','adversarial')
    N=3; grid=list(it.product([F(j,4) for j in range(5)],repeat=N)); m=len(grid)
    avg=[sum(p)/N for p in grid]; scalar=[p[0] for p in grid]
    affine=[(F(0),F(1)),(F(1,4),F(3,4)),(F(1,2),F(1,2)),(F(3,4),F(1,4)),(F(1,4),F(1,2))]
    Gscalar=np.array([[(a+b*x)**N for x in scalar] for a,b in affine],object)
    Gstructured=np.array([[(a+b*x)**N for x in avg] for a,b in affine],object)
    Gparity=np.array([[(1+math.prod(1-2*(a[j]+p[j]/2) for j in range(N)))/2 for p in grid]
                      for a in it.product((F(0),F(1,4)),repeat=N)]+
                     [[(1+math.prod(1-2*x for x in p))/2 for p in grid]],object)
    families=[('scalar affine powers',Gscalar,scalar),('multivariate parity pullbacks',Gparity,None),
              ('structured sum powers',Gstructured,avg)]
    rows=[]; D=7
    for name,G,xs in families:
        rank=sp.Matrix(np.vstack([np.ones(m,dtype=int),G]).tolist()).to_DM().rank()
        # Dictionary fixed before fitting: all multilinear monomials, response
        # vectors, and scalar/structured Chebyshev bases where applicable.
        candidates=list(G)
        if xs is None:
            candidates += [np.array([math.prod(p[j] for j in range(N) if a[j]) for p in grid],object)
                           for a in it.product((0,1),repeat=N) if sum(a)]
        else:
            candidates += [np.array([F(sp.chebyshevt(j,2*x-1)) for x in xs],object) for j in range(1,N+1)]
        selected=[]; remaining=list(range(len(candidates))); curve=[]
        for d in range(1,D+1):
            best=math.inf; chosen=None
            if d < rank:
                for j in remaining:
                    features=selected+[candidates[j]]
                    score=max(approximation(g,features)[0] for g in G)
                    if score<best-1e-12: best,chosen=score,j
                selected.append(candidates[chosen]); remaining.remove(chosen)
            elif d==rank:
                # Add enough independent response directions; zero error has a
                # global lower bound of zero and is therefore exact once attained.
                for j in remaining:
                    if sp.Matrix(np.vstack([np.ones(m,dtype=int),selected,candidates[j]]).tolist()).rank()>len(selected)+1:
                        selected.append(candidates[j]); remaining.remove(j); break
            options=[('greedy finite-dictionary subspace',selected.copy())]
            if xs is not None:
                cheb=[np.array([F(sp.chebyshevt(j,2*x-1)) for x in xs],object) for j in range(1,min(d,N)+1)]
                options.append(('Chebyshev subspace',cheb))
            scores=[(max(approximation(g,phi)[0] for g in G),label,phi) for label,phi in options]
            error,label,features=min(scores,key=lambda v:v[0])
            width=max(pair_width(g,features)[0] for g in G)
            c.eq(width,2*error,'selected subspace sharp discrepancy',dict(family=name,d=d,features=features),True)
            c.le(error,.5,'constant-containing approximation range',dict(family=name,d=d),True)
            curve.append(dict(feature_budget=d,actual_features=len(features),uniform_error=error,
                              discrepancy=width,subspace=label,features=features,
                              optimality='exact: attained universal zero lower bound' if error<=TOL else 'upper bound: best subspace found, not global optimum'))
            c.count()
        c.eq(curve[-1]['uniform_error'],0,'sufficient budget spans finite response family',dict(family=name,rank=rank),True)
        if xs is not None:
            heldout=np.array([[(F(1,8)+F(3,4)*x)**N for x in xs],[(F(1,3)+F(1,3)*x)**N for x in xs]],object)
        else:
            heldout=np.array([[(1+math.prod(1-2*(F(1,8)+p[j]/2) for j in range(N)))/2 for p in grid]],object)
        for entry in curve:
            entry['heldout_uniform_error']=max(approximation(g,entry['features'])[0] for g in heldout)
        rows.append(dict(family=name,maximum_degree=N,grid_points=m,response_count=len(G),
                         span_dimension_including_constant=rank,curve=curve))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for row in rows:
        for ax,field in zip(axs,('uniform_error','discrepancy')):
            ax.plot(range(1,D+1),[r[field] for r in row['curve']],'o-',label=row['family'])
            ax.set(xlabel='Nonconstant feature budget d',ylabel=field.replace('_',' '))
    axs[0].set_title('Best found subspace: upper bounds'); axs[1].set_title('Exact LP discrepancy for selected subspace')
    axs[0].legend(fontsize=8); savefig('C19_response_complexity')
    c.data['details']=dict(N=N,grid=grid,families=rows,
                           scope='Uniform errors are over the complete finite grid. Positive values are subspace-search upper bounds, not globally optimal Grassmannian widths. Held-out responses are not used in selection.')
    return c.finish()


def check_C20(quick=False):
    c=Claim(20,'Finite calibration, omission and coverage controls','Exact finite learning processes with explicit response features','witness')
    # Correct feature: U(z,y)=y, terminal y; fixed Bernoulli providers.
    reference=F(3,4); candidate=F(2,3); gap=abs(reference-candidate)
    c.eq(gap,F(1,12),'covered calibration discrepancy')
    c.le(gap,abs(reference-candidate),'correct-feature population certificate')
    # A possible independently collected finite audit: n=120, counts 90 and 80.
    # Means coincide with population means, so the simultaneous audit event holds.
    n=120; delta=.05; e=2*math.sqrt(math.log(4/delta)/(2*n)); dhat=F(90,n)-F(80,n)
    v0,_,_=exact_count_learning(reference); vs,_,_=exact_count_learning(candidate)
    c.eq(v0-vs,gap,'actual covered three-update process')
    c.le(float(abs(v0-vs)),float(abs(dhat))+e,'correct feature finite-audit certificate (39)',floating=True)
    sq0,_,_=exact_count_learning(reference,power=2); sqs,_,_=exact_count_learning(candidate,power=2)
    c.le(float(abs(sq0-sqs)),2*(float(abs(dhat))+e),'correct feature affine finite-audit certificate (40)',floating=True)
    pts,P,Q=law_parity(2)
    for a in ((1,0),(0,1)):
        c.eq(moment(pts,P,a),moment(pts,Q,a),'all audited marginal features match')
    omitted=moment(pts,P,(1,1))-moment(pts,Q,(1,1))
    c.eq(omitted,F(1,2),'omitted decision-relevant cross feature gap')
    # State 0 is certified; first reference/candidate update both reach state 1.
    # At state 1, the two providers reset terminal state to 0 vs 1.
    K0=np.array([[[F(0),F(1)],[F(0),F(1)]],[[F(1),F(0)],[F(1),F(0)]]],object)
    KS=K0.copy(); KS[1]=np.array([[F(0),F(1)],[F(0),F(1)]],object)
    st=process_stats(K0,KS,np.array([F(1),F(0)]),np.array([F(0),F(1)]),np.array([[1,0],[1,0]]))
    c.check(abs(st['gap'])>st['stopped'],'unpenalized coverage certificate fails as designed',st)
    c.le(abs(st['gap']),st['stopped']+st['rho'],'first-exit penalty repairs certificate',st)
    c.eq(st['rho'],F(1),'exact first-exit probability')
    c.count(3); c.data['details']=dict(correct_feature_gap=gap,omitted_feature_gap=omitted,
                                      correct_feature_audit=dict(n=n,reference_count=90,candidate_count=80,
                                                                 delta=delta,Dhat=dhat,e0_plus_eS=e,
                                                                 certificate39=float(abs(dhat))+e,
                                                                 actual_squared_return_gap=sq0-sqs),
                                      coverage_gap=abs(st['gap']),without_rho_bound=st['stopped'],rho=st['rho'],
                                      with_rho_bound=st['stopped']+st['rho'])
    return c.finish()


def check_C21(quick=False):
    c=Claim(21,'Signed identity (43); fidelity versus improvement (41)',
            'Rational state-dependent kernel selection and signed continuation sums','adversarial')
    worst=-math.inf
    for i in range(5 if quick else 50):
        (K0,K1),init,terminal,meta=process(c.rng,states=4+i%3,rounds=1+i%6)
        choices=c.rng.integers(0,2,size=K0.shape[:2]); KS=np.where(choices[:,:,None],K1,K0)
        st=process_stats(K0,KS,init,terminal)
        c.eq(st['alternative']-st['reference'],-st['signed'],'signed actual-state identity',dict(**meta,choices=choices,initial=init,terminal=terminal))
        gamma=st['reference']-init@terminal; eps=st['absolute']
        score=(gamma-eps)-(st['alternative']-init@terminal); worst=max(worst,float(score))
        c.le(gamma-eps,st['alternative']-init@terminal,'fidelity-improvement inequality',dict(**meta,choices=choices))
        c.count()
    examples=[]
    for ref,cand in [(F(4,5),F(79,100)),(F(2,5),F(39,100)),(F(1,2),F(49,100))]:
        initial=F(1,2); gamma=ref-initial; eps=abs(ref-cand)
        c.le(gamma-eps,cand-initial,'reference usefulness is required',dict(reference=ref,candidate=cand,initial=initial))
        if gamma<=0: c.check(cand<=initial,'faithful ineffective reference gives no improvement')
        examples.append(dict(initial=initial,reference=ref,candidate=cand,gamma=gamma,epsilon=eps,candidate_gain=cand-initial))
    # Exhaustive grid search maximizes the same inequality slack independently.
    for init,ref,cand in it.product([F(j,8) for j in range(9)],repeat=3):
        score=(ref-init-abs(ref-cand))-(cand-init)
        c.le(score,0,'adversarial fidelity grid',dict(initial=init,reference=ref,candidate=cand))
        worst=max(worst,float(score))
    c.data['details']=dict(examples=examples,adversarial_best_lhs_minus_rhs=worst)
    return c.finish()


def check_C22(quick=False):
    c=Claim(22,'Stability control for Theorem 18','Independent probability and noisy dual LPs on binary response','adversarial')
    rows=[]
    for alpha in (F(1),F(1,2),F(1,10),F(1,100)):
        for eps in (F(0),F(1,1000),F(1,100),F(1,20),F(1,5),F(1),F(2)):
            phi=[[F(0),alpha]]; g=[F(0),F(1)]
            value,P,Q=pair_width(g,phi,[eps]); dual=noisy_dual(g,phi,[eps]); expected=min(F(1),eps/alpha)
            c.eq(value,float(expected),'scaled binary stability formula',dict(alpha=alpha,epsilon=eps,P=P,Q=Q),True)
            c.eq(dual,float(expected),'scaled binary dual',dict(alpha=alpha,epsilon=eps),True)
            rows.append(dict(alpha=alpha,epsilon=eps,discrepancy=value,formula=expected)); c.count()
    c.data['details']=dict(configurations=rows)
    return c.finish()


# ---------------------------------------------------------------------------
# C23: CONTROLLED NEURAL EXPERIMENT. No Gemma or external benchmark is used.
# ---------------------------------------------------------------------------
NEURAL_LABEL='controlled neural, not Gemma, not an external benchmark'


def neural_init(k, replicas):
    hidden=np.array([.2,-.1,.3,-.25])
    output=np.array([.5,-.3,.2,.15])
    return [np.broadcast_to(hidden,(replicas,k,4)).copy(),np.zeros((replicas,4)),
            np.broadcast_to(output,(replicas,4)).copy(),
            np.full(replicas,-np.tanh(hidden)@output)]


def neural_forward(params):
    W,b,v,o=params; h=np.tanh(W+b[:,None,:]); logits=np.einsum('nkh,nh->nk',h,v)+o[:,None]
    # Stable sigmoid using NumPy alone, including for very large negative logits.
    probabilities=np.exp(-np.logaddexp(0.,-logits))
    return probabilities,h,logits


def neural_gradient(params, labels):
    probabilities,h,_=neural_forward(params)
    error=(probabilities-labels)/labels.shape[1]
    dh=error[:,:,None]*params[2][:,None,:]*(1-h*h)
    return [dh,dh.sum(axis=1),np.einsum('nk,nkh->nh',error,h),error.sum(axis=1)]


def neural_step(params, labels, eta, optimizer, state, step):
    grads=neural_gradient(params,labels)
    if optimizer=='SGD':
        return [p-eta*g for p,g in zip(params,grads)],state
    first,second=state
    first=[.9*m+.1*g for m,g in zip(first,grads)]
    second=[.999*v+.001*g*g for v,g in zip(second,grads)]
    params=[p-eta*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
            for p,m,v in zip(params,first,second)]
    return params,(first,second)


def tangent_matrix(k):
    params=neural_init(k,1); _,h,_=neural_forward(params)
    h=h[0]; v=params[2][0]; a=(1-h*h)*v
    gram=h@h.T+1+a@a.T+np.diag(np.sum(a*a,axis=1))
    return gram/(4*k)


def neural_return(p,task,N):
    if task=='Prop5': return .5+p[:,0]/20
    if task in ('Prop6','regroup'): return p[:,0]**2
    if task=='moment': return p[:,0]**N
    return (1+np.prod(1-2*p,axis=1))/2


def training_sequences(rng,points,law,R,replicas,quick):
    nonzero=np.flatnonzero(np.asarray(law,float)>0); count=len(nonzero)**R
    if count<=4096:
        indices=np.array(list(it.product(nonzero,repeat=R)),dtype=int)
        probs=np.prod(np.asarray(law,float)[indices],axis=1)
        return np.asarray(points,float)[indices],probs,'complete finite-history enumeration'
    indices=rng.choice(len(points),size=(replicas,R),p=np.asarray(law,float))
    return np.asarray(points,float)[indices],np.full(replicas,1/replicas),'independent training replicas'


def train_sequence(labels,eta,optimizer):
    count,R,k=labels.shape; params=neural_init(k,count)
    state=([np.zeros_like(p) for p in params],[np.zeros_like(p) for p in params])
    affine=np.full((count,k),.5); A=tangent_matrix(k)
    for t in range(R):
        params,state=neural_step(params,labels[:,t],eta,optimizer,state,t+1)
        affine=affine+eta*(labels[:,t]-affine)@A.T
    p,_,_=neural_forward(params)
    return p,affine,params


def check_C23(quick=False):
    c=Claim(23,'Execution plan §9: controlled small neural experiment',
            'Actual NumPy tanh/sigmoid policy, cross-entropy SGD and Adam; exhaustive histories or independent runs','monte_carlo')
    rng=c.rng; replicas=240 if quick else 2400
    configs=[('SGD',eta,R) for eta in (.01,.1,1.,3.) for R in (1,4,40)]
    configs += [('Adam',eta,R) for eta in (.01,.1) for R in (1,4,40)]
    if quick: configs=[('SGD',.01,1),('SGD',1.,4),('Adam',.1,8)]
    # Analytic gradients checked against central differences of actual CE loss.
    params=neural_init(2,1); labels=np.array([[0.,1.]]); gradients=neural_gradient(params,labels)
    maxgrad=0.
    def loss(par):
        _,_,logits=neural_forward(par)
        return float(np.mean(np.logaddexp(0,logits)-labels*logits))
    for block in range(4):
        for index in np.ndindex(params[block].shape):
            plus=[p.copy() for p in params]; minus=[p.copy() for p in params]
            plus[block][index]+=1e-6; minus[block][index]-=1e-6
            numerical=(loss(plus)-loss(minus))/(2e-6)
            residual=abs(numerical-gradients[block][index]); maxgrad=max(maxgrad,residual)
            c.le(residual,1e-7,'numpy CE gradient finite difference',dict(block=block,index=index),True)
    # Common-state strict additive null uses the identical four labels with
    # identical weights, and ONE actual deterministic SGD update.
    base=neural_init(1,1)
    grouped=[np.array([[0.]]),np.array([[1.]])]; mixed=[np.array([[.5]]),np.array([[.5]])]
    def accumulated(packets):
        gs=[neural_gradient(base,z) for z in packets]
        return [p-.2*sum(g[j] for g in gs)/len(gs) for j,p in enumerate(base)]
    null=max(float(np.max(abs(a-b))) for a,b in zip(accumulated(grouped),accumulated(mixed)))
    c.le(null,1e-12,'neural common-state additive SGD null',floating=True)
    cases=[('Prop5',1,[(F(0),),(F(1),)],[F(1,4),F(3,4)],[F(3,4),F(1,4)],1),
           ('Prop6',2,[(F(0),),(F(1,2),),(F(1),)],[F(1,4),F(1,2),F(1,4)],[F(0),F(1),F(0)],2)]
    for N in (2,3,4):
        points,P,Q=law_scalar(N); cases.append(('moment',N,points,P,Q,N))
    for N in (2,3):
        points,P,Q=law_parity(N); cases.append(('parity',N,points,P,Q,N))
    rows=[]
    for task,N,points,P,Q,block_labels in cases:
        k=len(points[0])
        for optimizer,eta,R in configs:
            source=[]
            for law in (P,Q):
                seq,weights,mode=training_sequences(rng,points,law,R,replicas,quick)
                actual,affine,params=train_sequence(seq,eta,optimizer)
                value=neural_return(actual,task,N); prediction=neural_return(affine,task,N)
                c.check(bool(np.all(np.isfinite(value))) and bool(np.all((value>=0)&(value<=1))),
                        'actual neural returns bounded and finite',dict(task=task,N=N,optimizer=optimizer,eta=eta,R=R))
                se=float(value.std(ddof=1)/math.sqrt(len(value))) if mode=='independent training replicas' else 0.
                err=value-prediction
                difference_se=float(err.std(ddof=1)/math.sqrt(len(err))) if mode=='independent training replicas' else 0.
                source.append(dict(mean=float(weights@value),SE=se,affine_mean=float(weights@prediction),
                                   affine_comparison_SE=difference_se,mode=mode,n_training_runs=len(value),
                                   final_probability_mean=weights@actual))
            gap=source[0]['mean']-source[1]['mean']; se=math.hypot(source[0]['SE'],source[1]['SE'])
            pred=source[0]['affine_mean']-source[1]['affine_mean']
            comparison_se=math.hypot(source[0]['affine_comparison_SE'],source[1]['affine_comparison_SE'])
            margin=1e-4+.1*abs(pred)+4*comparison_se
            rows.append(dict(label=NEURAL_LABEL,task=task,N=N,k=k,optimizer=optimizer,step_size=eta,updates=R,
                             labels_per_block=block_labels,exposure_labels=R*block_labels,
                             independent_blocks_per_run=R,sources=source,neural_gap=gap,SE=se,
                             local_affine_SGD_prediction=pred,deviation_from_local_affine=gap-pred,
                             affine_comparison_SE=comparison_se,
                             local_affine_agreement=abs(gap-pred)<=margin,
                             agreement_threshold=margin,
                             comparison_scope='Fixed-initialization tangent affine SGD response; for Adam this is an SGD comparator, not an Adam guarantee.'))
            c.count(sum(s['n_training_runs'] for s in source))
        print(f'{NEURAL_LABEL}: completed {task} N={N}',flush=True)
    # The same four labels regrouped into packets; each macro-round is two
    # sequential updates. Same example multiset, weights and update count.
    regroup=[]
    for optimizer,eta,cycles in configs:
        a=np.tile(np.array([0.,1.]),cycles).reshape(1,2*cycles,1)
        b=np.full_like(a,.5)
        pa,_,_=train_sequence(a,eta,optimizer); pb,_,_=train_sequence(b,eta,optimizer)
        gap=float(neural_return(pa,'regroup',2)[0]-neural_return(pb,'regroup',2)[0])
        regroup.append(dict(label=NEURAL_LABEL,optimizer=optimizer,step_size=eta,cycles=cycles,
                            updates=2*cycles,exposure_labels=4*cycles,source_A_packets=[[0,0],[1,1]],
                            source_B_packets=[[0,1],[0,1]],neural_gap=gap,SE=0.,
                            mode='complete deterministic-history enumeration'))
        c.count(2)
    c.check(any(abs(r['neural_gap'])>1e-8 for r in regroup),'neural sequential regrouping can matter',regroup)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for optimizer,ax in zip(('SGD','Adam'),axs):
        rr=[r for r in rows if r['optimizer']==optimizer]
        for task in ('Prop5','Prop6','moment','parity'):
            subset=[r for r in rr if r['task']==task]
            ax.errorbar([r['local_affine_SGD_prediction'] for r in subset],[r['neural_gap'] for r in subset],
                        yerr=[r['SE'] for r in subset],fmt='o',markersize=4,label=task,alpha=.7)
        limits=ax.get_xlim(); ax.plot(limits,limits,'k--',linewidth=1)
        ax.set(title=optimizer,xlabel='Initial-tangent affine SGD comparator gap',ylabel='Actual final neural return gap')
        ax.legend(fontsize=8)
    fig.suptitle(NEURAL_LABEL,fontsize=11); savefig('C23_controlled_neural')
    c.data['notes']=[NEURAL_LABEL,
        'Finite-law enumeration is exact over training histories; transcendental neural arithmetic is float64. Monte Carlo uncertainty counts independent training runs.',
        'The tangent comparator is derived from the actual initial policy Jacobian. It is only a local approximation and is not the coordinatewise convex-affine theorem for arbitrary SGD or Adam.',
        'Agreement is descriptive: |gap-prediction| <= 1e-4 + 0.1|prediction| + four paired comparison SE. No neural outcome is asserted to follow an affine theorem.']
    cohorts=[]
    for name,predicate in [
            ('SGD, step <= 0.1, at most four updates',lambda r:r['optimizer']=='SGD' and r['step_size']<=.1 and r['updates']<=4),
            ('SGD, step >= 1 or forty updates',lambda r:r['optimizer']=='SGD' and (r['step_size']>=1 or r['updates']>=40)),
            ('Adam, all declared settings',lambda r:r['optimizer']=='Adam')]:
        subset=[r for r in rows if predicate(r)]
        if subset:
            worst=max(subset,key=lambda r:abs(r['deviation_from_local_affine']))
            cohorts.append(dict(label=NEURAL_LABEL,cohort=name,configurations=len(subset),
                                agreements=sum(r['local_affine_agreement'] for r in subset),
                                departures=sum(not r['local_affine_agreement'] for r in subset),
                                largest_absolute_deviation=abs(worst['deviation_from_local_affine']),
                                largest_deviation_case={key:worst[key] for key in ('task','N','optimizer','step_size','updates','neural_gap','SE','local_affine_SGD_prediction')}))
    c.data['details']=dict(label=NEURAL_LABEL,architecture='one-hot context -> 4 tanh units -> sigmoid',
                           initialization_by_k={k:neural_init(k,1) for k in (1,2,3)},
                           optimizer_settings=dict(SGD='plain gradient descent per sampled block',Adam=dict(beta1=.9,beta2=.999,epsilon=1e-8,bias_correction=True)),
                           gradient_max_residual=maxgrad,additive_null_parameter_residual=null,
                           configurations=rows,regrouping=regroup,
                           observed_affine_comparison_cohorts=cohorts,
                           affine_agreement_count=sum(r['local_affine_agreement'] for r in rows),
                           affine_disagreement_count=sum(not r['local_affine_agreement'] for r in rows),
                           full_monte_carlo_replicas_per_source=replicas)
    return c.finish()


CHECKS={f'C{i}':globals()[f'check_C{i}'] for i in range(1,24)}


def all_claims(report):
    return report['claims']+([report['controlled_neural']] if report.get('controlled_neural') else [])


def write_report(output,report):
    # Never read a previous report, even for resume or table generation.
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(serial(report),indent=2,allow_nan=False)+'\n')


def key_number(row):
    d=row.get('details',{}); ident=row['id']
    special={
        'C2':lambda:f"minimum SPRT/lower-bound ratio {d['minimum_sprt_ratio']:.6g}",
        'C3':lambda:'Prop 5 gap 1/40; Prop 6 gap eta²/8',
        'C6':lambda:f"largest coefficient ratio {row['best_ratio']:.6g}",
        'C10':lambda:f"largest gap/bound ratio {row['best_ratio']:.6g}",
        'C15':lambda:f"coverage gap 1 > bound {d['coverage_probe']['claimed_bound39']:.6g}",
        'C16':lambda:f"point-estimate gain {d['point_estimate_exact_gain']}; audit count 1016",
        'C17':lambda:f"median width-bound slack {d['median_inequality_slack']:.6g}",
        'C18':lambda:'independent n^(1−N); replay/descendants constant',
        'C19':lambda:'span dimensions '+', '.join(str(r['span_dimension_including_constant']) for r in d['families']),
        'C20':lambda:'omitted-feature gap 1/2; coverage gap = rho = 1',
        'C22':lambda:'W = min(1, epsilon/alpha)',
        'C23':lambda:f"{d['affine_agreement_count']} local agreements / {d['affine_disagreement_count']} departures; null {d['additive_null_parameter_residual']:.3g}",
    }
    try:
        return special[ident]() if ident in special else f"max assertion residual {row['worst_residual']:.3g}"
    except KeyError:
        return 'execution error: see JSON'


def write_readme(report):
    lines=['# Independent v0.14 numerical checks','',
           'Derived from `SPEC.md`, `Environment_Information_Theory_EN.md`, and execution-plan sections 9, 13.1, and 13.7. No original verification scripts or prior results were read, imported, or copied. The program reads only its own source and the three stated documents for provenance hashes.',
           '', 'Run from the repository root:', '', '```bash',
           "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --output theory/independent_checks/results_independent_v014.json",
           "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --quick --output theory/independent_checks/quick.json",
           "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C5,C14 --output theory/independent_checks/subset.json",
           "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python tests/test_theory_independent_quick.py",'```','',
           'No installation, network, API, GPU, external checkpoint, or repository training module is used. The executable blocks socket connections and DNS, fixes CUDA visibility, and limits numerical library threads to one. All artifacts and caches stay in this directory. A custom output path must also be inside this directory; a bare output filename is resolved here. Quick-run figures go in `figs/quick/`, preserving the full-run plots.',
           '', 'A failing scientific assertion is recorded with its seed, values, and complete witness under `counterexamples`; the run continues and returns exit code 1 after writing JSON. Solver failures and unexpected implementation errors are separately labelled execution errors. No assertion is relaxed to obtain a green run.',
           '',f"Full run: C1–C22 {report['runtime']['finite_seconds']:.2f} s; C23 {report['runtime']['controlled_neural_seconds']:.2f} s. Master seed: {MASTER}. Each claim uses NumPy SeedSequence([master seed, claim number]); `--only` preserves the selected claim's stream.",
           '', '| Claim | Status | Key number |','|---|---|---|']
    for row in all_claims(report): lines.append(f"| {row['id']} | {row['status']} | {key_number(row)} |")
    lines += ['', '## Failure and test interpretation','',
              'C15 preserves the requested coverage-condition probe as a FAIL. Both audited message laws are deterministically zero at cell 0. Training starts at state 1, where the reference always emits 0 and the candidate always emits 1. The same feature and response `g(y)=y` are valid at both states, which are both declared certified; rho is zero. The audit event occurs with probability one, yet the final expected-return gap is 1 and the claimed bound is approximately 0.0296. The connection from (37) to (39) requires explicit visited-state/provider-law coverage by audited cells. C20 independently confirms that a correctly charged first-exit penalty repairs its coverage control.',
              '', 'The quick test intentionally asserts that there are no FAIL statuses, including C15. It therefore fails while this counterexample remains in the theory; it is not skipped, xfailed, or given an exception. A scientific FAIL is different from a process crash. The full JSON is produced in either case.',
              '', '## Methods and limits','',
              '- Finite rational identities use Fraction or SymPy. LPs use independent direct (P,Q), approximation, and oscillation formulations with HiGHS. Feasibility and comparison assertions use 1e-8; worst feasibility residuals are reported separately.',
              '- Binomial probabilities and sequential random-walk recursion are exact rational calculations. Logarithmic/transcendental bounds and dense-grid diagnostics are evaluated numerically. Cumulant rates are checked by exact finite sample-index expansions through n=220, including scalar and bivariate laws and a vanishing leading contraction.',
              '- Inequality searches use LP extrema, deterministic vertex/event enumeration, local perturbations, or bounded scalar optimization. These searches are numerical evidence, not replacements for proofs. The k=4 binary barrier is globally certified by a feasible event achieving the universal upper bound 1; it does not run 65536 redundant event LPs.',
              '- C18 uses an explicit descendant channel `(Z+1/2)/2`; it does not claim every possible postprocessing channel must preserve a nonzero gap. Independent evidence and training exposure are separately recorded.',
              '- C19 reports exact rational span dimensions including constants. Positive approximation errors are upper bounds from the best found dictionary/Chebyshev subspace, not globally optimal subspaces. The uniform error is evaluated on the complete finite grid, held-out responses are labelled separately, and the discrepancy for each selected subspace is independently optimized by LP.',
              '- C16 reports 100000 independent runs per configuration and four-SE comparisons, as requested. The conditional analogue of (45) is searched without asserting a violation when none is found; an additional exact witness demonstrates conditioning bias.',
              '- C23 is **controlled neural, not Gemma, not an external benchmark**. It uses a real one-hidden-layer tanh policy, manual cross-entropy gradients, SGD at four step sizes, and Adam at two step sizes. Complete finite-history enumeration has no sampling SE; otherwise 2400 independent training runs per source are used. Float64 nonlinear arithmetic remains numerical even when histories are enumerated.',
              '- The neural affine comparator is the fixed initial-tangent SGD linearization, explicitly distinguished from the scalar convex-affine theorem and from Adam. Agreement thresholds are fixed in code before execution. Departures are empirical outcomes, not failed affine-theorem assertions.',
              '', '## Figures','',
              '- [C18: evidence and exposure](figs/C18_evidence_exposure.png)',
              '- [C19: response complexity](figs/C19_response_complexity.png)',
              '- [C23: controlled neural](figs/C23_controlled_neural.png)',
              '', 'SVG versions are available alongside the PNGs. Full per-configuration numbers, seeds, provenance hashes, package versions, searches, and counterexamples are in [results_independent_v014.json](results_independent_v014.json).','']
    neural=report.get('controlled_neural')
    if neural and 'observed_affine_comparison_cohorts' in neural.get('details',{}):
        lines += ['## Observed neural comparison', '', NEURAL_LABEL+'. All numbers below have this scope.', '',
                  '| Settings | Agreements | Departures | Largest absolute gap deviation |',
                  '|---|---:|---:|---:|']
        for group in neural['details']['observed_affine_comparison_cohorts']:
            lines.append(f"| {group['cohort']} | {group['agreements']} | {group['departures']} | {group['largest_absolute_deviation']:.6g} |")
        lines += ['', 'These compare the actual neural gap with the fixed initial-tangent SGD comparator using the stated descriptive threshold. They do not assert the convex-affine theorem for the neural optimizer. Adam is compared to SGD linearization to expose optimizer dependence.', '']
    (HERE/'README.md').write_text('\n'.join(lines))


def main(argv=None):
    global FIGURE_DIR
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results_independent_v014.json')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--only',help='Comma-separated claim IDs, e.g. C5,C14')
    args=parser.parse_args(argv)
    FIGURE_DIR=HERE/'figs'/'quick' if args.quick else HERE/'figs'
    output=(HERE/args.output).resolve() if not args.output.is_absolute() and len(args.output.parts)==1 else args.output.resolve()
    if not output.is_relative_to(HERE): parser.error('--output must remain under theory/independent_checks/')
    selected=list(CHECKS) if not args.only else [v.strip().upper() for v in args.only.split(',')]
    if not selected or len(set(selected))!=len(selected) or any(v not in CHECKS for v in selected):
        parser.error('--only requires unique IDs from C1 through C23')
    selected=sorted(selected,key=lambda x:int(x[1:]))  # C23 always follows the finite suite.
    sources=[Path(__file__).resolve(),HERE/'SPEC.md',
             ROOT/'docs/direction/v014_handoff/Environment_Information_Theory_EN.md',
             ROOT/'docs/direction/v014_handoff/Coding_Agent_Execution_Plan_EN.md']
    report=dict(schema_version=1,master_seed=MASTER,quick=args.quick,requested_claims=selected,
                full_run=not args.quick and selected==list(CHECKS),complete=False,
                versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sympy=sp.__version__,matplotlib=matplotlib.__version__),
                provenance=dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                                network='blocked in-process; no network operations requested',CUDA_VISIBLE_DEVICES='',
                                numeric_threads=1,teacher='none',student='C23 controlled numpy policy only'),
                claims=[],controlled_neural=None,counterexamples=[],
                runtime=dict(finite_seconds=0.,controlled_neural_seconds=0.,total_seconds=0.))
    started=time.perf_counter()
    for ident in selected:
        print(f'Starting {ident}',flush=True); then=time.perf_counter()
        try:
            row=CHECKS[ident](args.quick)
        except Exception as exc:
            # Preserve any prior scientific failures even when a later check crashes.
            row=ACTIVE.finish() if ACTIVE is not None and ACTIVE.id==ident else dict(id=ident,statement='execution error',method='incomplete',kind='witness',n_instances=0,worst_residual=0.,notes=[],counterexamples=[])
            row['status']='FAIL'
            row.setdefault('counterexamples',[]).append(dict(claim=ident,seed=[MASTER,int(ident[1:])],
                                                               assertion='execution error (not a mathematical counterexample)',
                                                               error_type=type(exc).__name__,message=str(exc)))
            row['execution_error']=True
        elapsed=time.perf_counter()-then; row['runtime_seconds']=elapsed
        report['counterexamples'].extend(row.pop('counterexamples',[]))
        if ident=='C23':
            report['controlled_neural']=row; report['runtime']['controlled_neural_seconds']+=elapsed
        else:
            report['claims'].append(row); report['runtime']['finite_seconds']+=elapsed
        report['runtime']['total_seconds']=time.perf_counter()-started
        write_report(output,report)
        print(f"{ident}: {row['status']} ({elapsed:.2f}s) — {key_number(row)}",flush=True)
    report['complete']=True
    report['summary']={status:sum(r['status']==status for r in all_claims(report)) for status in ('PASS','FAIL','INFO')}
    write_report(output,report)
    if report['full_run']: write_readme(report)
    print(f"Wrote {output}; {report['summary']}",flush=True)
    return int(bool(report['summary']['FAIL']))


if __name__=='__main__':
    raise SystemExit(main())
