#!/usr/bin/env python3
"""Independent, CPU-only numerical audit of the v0.16 draft. See SPEC.md."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import gzip
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy import integrate, optimize, signal, special, stats

MASTER_SEED = 160160
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BETAS = (0.9, 0.99, 0.995, 0.999, 0.9999)


def rng_for(*path):
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, *path]))


def clean(x):
    """Strict JSON: infinities are explicit strings, never nonstandard JSON numbers."""
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return "+infinity" if x == np.inf else "-infinity" if x == -np.inf else "NaN"
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


@dataclass
class Check:
    claim_id: str
    asserted: str
    data: list = field(default_factory=list)
    assertions: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    def assert_le(self, name, residual, tolerance, case):
        vals = np.atleast_1d(np.asarray(residual, dtype=float))
        bad = (~np.isfinite(vals)) | (vals > tolerance)
        worst = float(np.max(vals)) if np.all(np.isfinite(vals)) else np.inf
        entry = self.assertions.setdefault(name, dict(n=0, worst_residual=-np.inf,
                                                     tolerance=float(tolerance)))
        entry["n"] += int(vals.size)
        entry["worst_residual"] = max(entry["worst_residual"], worst)
        if np.any(bad):
            self.failures.append(dict(assertion=name, case=case, residual=worst,
                                      tolerance=float(tolerance), n_failed=int(bad.sum())))

    def result(self):
        return dict(claim_id=self.claim_id, asserted=self.asserted, n_instances=len(self.data),
                    n_assertions=sum(x["n"] for x in self.assertions.values()),
                    worst_residual=max([0.] + [x["worst_residual"] for x in self.assertions.values()]),
                    status="FAIL" if self.failures else "PASS", assertions=self.assertions,
                    failures=self.failures, data=self.data)


def mean_ci(samples, confidence=0.95):
    samples = np.asarray(samples)
    mean = np.mean(samples, axis=0)
    se = np.std(samples, axis=0, ddof=1) / np.sqrt(len(samples))
    radius = stats.t.ppf((1 + confidence) / 2, len(samples) - 1) * se
    return dict(mean=mean, se=se, ci_low=mean-radius, ci_high=mean+radius,
                confidence=confidence, n_replicates=len(samples))


def kbar(beta, s):
    return (1-beta) / (beta * -np.expm1(np.asarray(s)*np.log(beta)))


def c1(quick=False):
    exact = Check("C1.exact", "Proposition 2.5(a,b): raw-v recursion, both warmth bounds, exact common-factor invariance")
    points = np.unique(np.r_[np.arange(1, 2001), np.geomspace(2001, 10**6, 180).astype(int)])
    for beta in BETAS:
        # lfilter executes the recurrence; it does not evaluate its closed form.
        v = signal.lfilter([1-beta], [1, -beta], np.full(10**6, 1.7**2))
        observed = (1-beta)*1.7**2/(beta*v[points-1])
        theory = kbar(beta, points)
        lower = np.maximum(1/points, 1-beta)/beta
        upper = (1/points + 1-beta)/beta
        row = dict(beta2=beta, s=points, raw_v=v[points-1], observed=observed,
                   kbar=theory, lower=lower, upper=upper, max_upper_over_kbar=max(upper/theory),
                   plateau=(1-beta)/beta)
        idx = len(exact.data)
        exact.data.append(row)
        exact.assert_le("recursion_relative_error", abs(observed/theory-1), 1e-10, idx)
        exact.assert_le("lower_bound_relative_violation", lower/theory-1, 2e-13, idx)
        exact.assert_le("upper_bound_relative_violation", theory/upper-1, 2e-13, idx)
        b = Fraction(str(beta))
        # Rational arithmetic establishes equality, without floating-point tolerance.
        for t in (1, 7, 50):
            vv, dv = Fraction(13, 7), Fraction(11, 19)
            factor = 1-b**t
            residual = (vv+dv)/(b*vv) - ((vv+dv)/factor)/(b*vv/factor)
            exact.assert_le("bias_correction_exact", abs(float(residual)), 0, [idx, t])
    # Boundary: nonstationary second moments require a convolution, not sigma_s^2(1-b^s).
    sigma2 = np.exp(-np.arange(3000)/400)
    v = signal.lfilter([.001], [1, -.999], sigma2)
    naive = sigma2 * -np.expm1(np.arange(1, 3001)*np.log(.999))
    exact.data.append(dict(kind="nonstationary_diagnostic", beta2=.999, s=np.arange(1,3001,10),
                           second_moment=sigma2[::10], expected_v=v[::10], stationary_substitution=naive[::10]))

    mc = Check("C1.moments", "Stationary Gaussian, rescaled t(3), and sparse Gaussian: E v_s in a 99.9% CI")
    checkpoints = (1, 10, 100, 1000, 2000)
    reps = 1500 if quick else 16000
    for family in range(3):
        rng = rng_for(1, family)
        raw = np.zeros((len(BETAS), reps))
        capped = raw.copy()
        beta = np.asarray(BETAS)[:, None]
        cap = 100.
        if family == 1:
            # X=T_3/sqrt(3), E X^2=1; cap X^2 to get a finite-variance CI.
            density = lambda x: np.sqrt(3)*stats.t.pdf(np.sqrt(3)*x, 3)
            tail_prob = 2*stats.t.sf(np.sqrt(3*cap), 3)
            cap_mean = 2*integrate.quad(lambda x: x*x*density(x), 0, np.sqrt(cap), epsabs=1e-12)[0] + cap*tail_prob
            cap_second = 2*integrate.quad(lambda x: x**4*density(x), 0, np.sqrt(cap), epsabs=1e-12)[0] + cap**2*tail_prob
        for s in range(1, max(checkpoints)+1):
            if family == 0:
                g = rng.normal(size=reps)
            elif family == 1:
                g = rng.standard_t(3, size=reps)/np.sqrt(3)
            else:
                g = rng.normal(size=reps)*(rng.random(reps)<.05)/np.sqrt(.05)
            raw = beta*raw + (1-beta)*g*g
            if family == 1:
                capped = beta*capped+(1-beta)*np.minimum(g*g, cap)
            if s not in checkpoints:
                continue
            for j, b in enumerate(BETAS):
                target = -np.expm1(s*np.log(b))
                ci = mean_ci(raw[j], .999)
                method = "Studentized mean (finite fourth moment; asymptotic coverage)"
                if family == 1:
                    # Bernstein applied to independent bounded replicate weighted sums.
                    variance = (cap_second-cap_mean**2)*(1-b)**2*(-np.expm1(2*s*np.log(b)))/(1-b*b)
                    logdelta = np.log(2/.001)
                    radius = np.sqrt(2*variance*logdelta/reps)+2*cap*target*logdelta/(3*reps)
                    estimate = capped[j].mean()+(1-cap_mean)*target
                    ci.update(mean=estimate, ci_low=estimate-radius, ci_high=estimate+radius,
                              raw_untruncated_mean=raw[j].mean(), cap=cap,
                              known_tail_expectation_correction=(1-cap_mean)*target,
                              capped_variance=variance, se=None)
                    method = "Bernstein CI for capped g^2 plus analytic t(3) tail expectation; no finite-fourth-moment assumption"
                idx = len(mc.data)
                mc.data.append(dict(family=("Gaussian", "Student_t3_unit_second_moment", "sparse_Gaussian_p005")[family],
                                    beta2=b, s=s, target=target, method=method, **ci))
                mc.assert_le("target_outside_99_9_CI", max(ci["ci_low"]-target, target-ci["ci_high"]), 0, idx)

    acc = Check("C1.accumulation", "Corrected cumulative upper bound; first failure of OLD c/(W+s) hypothesis")
    R = np.unique(np.r_[np.arange(1,101), np.geomspace(101, 100000, 100).astype(int)])
    for b, W, c in itertools.product(BETAS, (1,4,32), (1.,2.,4.)):
        s = np.arange(1,100001)
        envelope = c/(W+s)+c*(1-b)
        cumulative = np.cumsum(np.minimum(1,envelope))[R-1]
        bound = c*np.log1p(R/W)+c*(1-b)*R
        idx = len(acc.data)
        acc.data.append(dict(kind="corrected_bound", beta2=b, W=W, c=c, R=R, lhs=cumulative, rhs=bound))
        acc.assert_le("accumulation_violation", cumulative-bound, 1e-9, idx)
    for b, W in itertools.product(BETAS, (1,4,32)):
        for c in (1.,1/b,2.):
            s = np.arange(1,100001)
            bad = kbar(b,s)>c/(W+s)
            first = int(s[np.flatnonzero(bad)[0]]) if bad.any() else None
            idx = len(acc.data)
            acc.data.append(dict(kind="old_hypothesis", beta2=b,W=W,c=c,first_failure_s=first,
                                 kbar_at_failure=float(kbar(b,first)) if first else None,
                                 old_rhs=c/(W+first) if first else None,
                                 note="s is elapsed raw-v age, exactly as in the requested OLD hypothesis"))
            acc.assert_le("old_hypothesis_has_counterexample", 0 if first else 1, 0, idx)
            # Also distinguish the natural burn-in interpretation: raw age W+s.
            # It is a diagnostic in addition to, not a replacement for, the
            # literal requested age-s hypothesis above.
            warm_bad=kbar(b,W+s)>c/(W+s)
            warm_first=int(s[np.flatnonzero(warm_bad)[0]]) if warm_bad.any() else None
            acc.data.append(dict(kind="old_hypothesis_after_actual_W_step_burn_in",beta2=b,W=W,c=c,
                                 first_failure_s=warm_first,raw_age=W+warm_first if warm_first else None,
                                 kbar_at_failure=float(kbar(b,W+warm_first)) if warm_first else None,
                                 old_rhs=c/(W+warm_first) if warm_first else None))
    return [exact, mc, acc]


def functional_variance(matrix, c):
    eigen, vectors = np.linalg.eigh(matrix)
    keep = eigen > max(1., np.max(eigen))*1e-12
    projection = vectors.T@c
    if np.linalg.norm(projection[~keep]) > 1e-9*max(1., np.linalg.norm(c)):
        return np.inf, None
    q = vectors[:,keep]@(projection[keep]/eigen[keep])
    return float(c@q), q


def solve_design(ID, features, c, K, n):
    """Independent simplex primal and linear-epigraph quadratic dual; no ridge added."""
    p, nx = len(c), len(features)
    base = K*ID
    if n == 0:
        variance, q = functional_variance(base,c)
        if q is None:
            eigen, vec = np.linalg.eigh(base)
            null = vec[:,eigen <= max(1.,eigen.max())*1e-12]
            h = null@(null.T@c)
            h /= c@h
            return dict(primal=np.inf, dual=np.inf, w=np.ones(nx)/nx, h=h,
                        kkt_residual=0., dual_constraint_residual=abs(c@h-1), primal_success=True, dual_success=True)
        h = q/variance
        dual = 1/(h@base@h)
        return dict(primal=variance,dual=dual,w=np.ones(nx)/nx,h=h,kkt_residual=0.,
                    dual_constraint_residual=abs(c@h-1),primal_success=True,dual_success=True)
    # Common numerical conditioning only. Neither solver initializes from the other's optimum.
    ref = base+n*features.T@features/nx
    ev, U = np.linalg.eigh(ref)
    B = (U/np.sqrt(ev))@U.T
    D, f = B@base@B, features@B
    cb = B@c
    scale = np.linalg.norm(cb)
    u = cb/scale
    def objective(w):
        M = D+n*(f.T*w)@f
        try:
            q = np.linalg.solve(M,u)
        except np.linalg.LinAlgError:
            value,q = functional_variance(M,u)
            if q is None:
                # The true extended objective is infinite on this face. Giving
                # SLSQP that value makes its line search retreat into the domain;
                # no ridge or positive lower bound changes the feasible designs.
                return np.inf, -n*(f@u)**2
        return float(u@q), -n*(f@q)**2
    constraint = dict(type="eq", fun=lambda w: w.sum()-1, jac=lambda w: np.ones(nx))
    sol = optimize.minimize(objective,np.full(nx,1/nx),jac=True,bounds=[(0,1)]*nx,
                            constraints=constraint,method="SLSQP",options=dict(ftol=2e-14,maxiter=2000))
    w = np.maximum(sol.x,0); w /= w.sum()
    val, grad = objective(w)
    kkt = max(0., float(w@grad-grad.min()))/max(val,1e-30)
    if kkt > 2e-7:
        # Same objective/tolerance; preserve first solver diagnostics if a refinement is necessary.
        sol2 = optimize.minimize(objective,w,jac=True,bounds=[(0,1)]*nx,constraints=constraint,
                                 method="SLSQP",options=dict(ftol=1e-15,maxiter=4000))
        if objective(sol2.x)[0] <= val:
            sol = sol2; w = np.maximum(sol.x,0); w /= w.sum()
            val, grad = objective(w)
            kkt = max(0., float(w@grad-grad.min()))/max(val,1e-30)
    # Variable (z,t): minimize z'Dz+n*t^2, u'z=1, t >= +/- f_x'z.
    linear = np.r_[np.c_[-f,np.ones(nx)], np.c_[f,np.ones(nx)]]
    def dual_objective(y):
        z,t = y[:-1],y[-1]
        return float(z@D@z+n*t*t), np.r_[2*D@z,2*n*t]
    start = np.r_[u, np.max(abs(f@u))]
    dual_sol = optimize.minimize(dual_objective,start,jac=True,method="SLSQP",
        constraints=[dict(type="eq",fun=lambda y: u@y[:-1]-1,jac=lambda y: np.r_[u,0.]),
                     dict(type="ineq",fun=lambda y: linear@y,jac=lambda y: linear)],
        options=dict(ftol=2e-14,maxiter=3000))
    attempts=[dict(method="SLSQP",success=bool(dual_sol.success),status=int(dual_sol.status),
                   iterations=int(dual_sol.nit),objective=float(dual_sol.fun))]
    if not dual_sol.success:
        # A second algorithm with an exact Hessian handles SLSQP line-search
        # status 8 without declaring an unsuccessful solver successful.
        hessian=np.zeros((p+1,p+1));hessian[:p,:p]=2*D;hessian[-1,-1]=2*n
        constraints=[optimize.LinearConstraint(np.r_[u,0.][None,:],1.,1.),
                     optimize.LinearConstraint(linear,0.,np.inf)]
        fallback=optimize.minimize(dual_objective,dual_sol.x,jac=True,hess=lambda y:hessian,
            method="trust-constr",constraints=constraints,
            options=dict(gtol=1e-12,xtol=1e-14,barrier_tol=1e-14,maxiter=2000))
        attempts.append(dict(method="trust-constr",success=bool(fallback.success),
                             status=int(fallback.status),iterations=int(fallback.nit),
                             objective=float(fallback.fun)))
        # Polish the active face identified by SLSQP by solving its exact KKT
        # linear system. Convexity plus feasible nonnegative multipliers is
        # an independent optimality certificate, not an optimizer status guess.
        active=np.flatnonzero(abs(linear@dual_sol.x)<1e-7)
        E=np.r_[u,0.][None,:]; L=linear[active]
        system=np.block([[hessian,E.T,-L.T],
                         [E,np.zeros((1,1+len(active)))],
                         [L,np.zeros((len(active),1+len(active)))]])
        rhs=np.r_[np.zeros(p+1),1.,np.zeros(len(active))]
        solution=np.linalg.lstsq(system,rhs,rcond=1e-13)[0]
        y=solution[:p+1];multipliers=solution[p+2:]
        certificate=max(np.max(abs(system@solution-rhs)),abs(E[0]@y-1),
                        max(0.,-np.min(linear@y)),max(0.,-np.min(multipliers,initial=0)))
        certified=bool(certificate<=1e-9)
        attempts.append(dict(method="active_face_KKT_linear_solve",success=certified,
                             status=0 if certified else 1,iterations=1,
                             objective=float(dual_objective(y)[0]),certificate_residual=certificate,
                             active_constraints=active,multipliers=multipliers))
        dual_sol=optimize.OptimizeResult(x=y,success=True,status=0,nit=1) if certified else fallback
    z = dual_sol.x[:-1]
    h = B@z/scale
    h /= c@h
    energy = float(h@base@h+n*np.max((features@h)**2))
    return dict(primal=val*scale**2,dual=1/energy,w=w,h=h,kkt_residual=kkt,
                dual_constraint_residual=max(abs(u@z-1),max(0.,-np.min(linear@dual_sol.x))),
                primal_success=bool(sol.success),dual_success=bool(dual_sol.success),
                primal_iterations=int(sol.nit),dual_iterations=int(dual_sol.nit),dual_attempts=attempts)


def round_counts(weights, total):
    x = np.asarray(weights)*total/np.sum(weights)
    counts = np.floor(x).astype(int)
    left = total-int(counts.sum())
    counts[np.argsort(-(x-counts),kind="stable")[:left]] += 1
    return counts


def c2(quick=False):
    design = Check("C2.design", "Lemma 3.4: independent primal/dual equality (relative 1e-6), KKT, singular n=0 convention")
    risk = Check("C2.risk", "Theorems 3.3/3.5: rounded-design efficient estimator risk in [frak_n/4,frak_n]; exact two-point bounds")
    alias = Check("C2.aliasing", "Corollary 3.7: exactly equal data laws along ker(A_D), every K; triangle lower bound")
    grid = itertools.product((2,3) if quick else (2,3,5,8), (4,) if quick else (4,16), range(3))
    for model_id,(p,nx,kind) in enumerate(grid):
        rng = rng_for(2,model_id)
        A = rng.normal(size=(p+2,p))/np.sqrt(p)
        A[:p] += np.eye(p)
        if kind:
            A[:,-1] = 0  # exact null direction; no numerical approximation to aliasing
        f = rng.normal(size=(nx,p))
        c = rng.normal(size=p)
        if kind == 2:
            c[-1] = 0  # singular but estimable n=0 case
        c /= np.linalg.norm(c)
        ID = A.T@A
        for K,n in itertools.product((1,10,100),(0,10,100)):
            idx = len(design.data)
            sol = solve_design(ID,f,c,K,n)
            row = dict(model_id=model_id,p=p,n_x=nx,kind=("full_rank","aliased","singular_estimable")[kind],
                       K=K,n=n,A_D=A,features=f,c=c,**sol)
            design.data.append(row)
            gap = 0 if np.isinf(sol["primal"]) and np.isinf(sol["dual"]) else abs(sol["primal"]/sol["dual"]-1)
            design.assert_le("primal_dual_relative_gap",gap,1e-6,idx)
            design.assert_le("KKT_Elfving_relative_gap",sol["kkt_residual"],1e-6,idx)
            design.assert_le("dual_feasibility",sol["dual_constraint_residual"],1e-8,idx)
            design.assert_le("solver_success",0 if sol["primal_success"] and sol["dual_success"] else 1,0,idx)
            if np.isinf(sol["primal"]):
                continue
            counts = round_counts(sol["w"],n)
            M = K*ID+(f.T*counts)@f
            variance,q = functional_variance(M,c)
            ridx = len(risk.data)
            if q is None:
                risk.data.append(dict(design_case=idx,counts=counts,rounded_variance=np.inf))
                risk.assert_le("rounded_design_identifiable",np.inf,0,ridx)
                continue
            reps = 3000 if quick else 30000
            # Explicit independent demonstration/query sufficient-statistic noises.
            error = np.sqrt(K)*rng.normal(size=(reps,len(A)))@(A@q)
            error += rng.normal(size=(reps,nx))@(np.sqrt(counts)*(f@q))
            ci = mean_ci(abs(error))
            norm = np.sqrt(sol["dual"])
            h = sol["h"]
            energy = K*h@ID@h+n*np.max((f@h)**2)
            delta = h/np.sqrt(energy)
            separation = abs(c@delta)
            kl_D = .5*np.sum((A@delta)**2)
            kl_Qmax = .5*np.max((f@delta)**2)
            pinsker_bound = .5*separation*(1-np.sqrt((K*kl_D+n*kl_Qmax)/2))
            mahal = np.sqrt(delta@M@delta)
            exact_tv_bound = separation*stats.norm.cdf(-mahal/2)
            exact_risk = np.sqrt(2/np.pi*variance)
            risk.data.append(dict(design_case=idx,counts=counts,frak_n=norm,rounded_variance=variance,
                                  rounding_loss=variance/sol["primal"]-1,exact_MAE=exact_risk,
                                  risk_over_frak_n=ci["mean"]/norm,least_favourable_delta=delta,
                                  KL_D=kl_D,KL_Q_max=kl_Qmax,pinsker_two_point=pinsker_bound,
                                  exact_TV_two_point=exact_tv_bound,**ci))
            risk.assert_le("risk_lower_violation",norm/4-ci["mean"],0,ridx)
            risk.assert_le("risk_upper_violation",ci["mean"]-norm,0,ridx)
            risk.assert_le("exact_risk_upper_violation",exact_risk-norm,0,ridx)
            risk.assert_le("two_point_exceeds_achieved",max(pinsker_bound,exact_tv_bound)-ci["mean"],0,ridx)
            # Fixed 99.9% sampling check, separate from the mathematical [1/4,1] assertion.
            risk.assert_le("MC_vs_exact_MAE_in_SE",abs(ci["mean"]-exact_risk)/ci["se"],stats.norm.ppf(.9995),ridx)
        if kind == 1:
            h = np.zeros(p); h[-1] = 1
            theta = rng.normal(size=p)
            gap = abs(c@h)
            for K in (1,10,100,10**6):
                est = rng.normal(size=100)
                lhs = abs(est-c@theta)+abs(est-c@(theta+h))
                idx = len(alias.data)
                alias.data.append(dict(model_id=model_id,K=K,A_D=A,c=c,theta=theta,h=h,
                                       mean_difference=A@h,covariance_difference=0.,functional_gap=gap,
                                       minimax_lower_bound=gap/2,midpoint_worst_error=gap/2))
                alias.assert_le("identical_means_exact",abs(A@h),0,idx)
                alias.assert_le("triangle_inequality",gap-lhs,2e-14,idx)
    return [design,risk,alias]


def water_fill(weights, seeds, budget):
    W,k = np.asarray(weights,float),np.asarray(seeds,float)
    if budget == 0:
        return np.zeros_like(W)
    if np.any(W<0) or np.any(k<0) or budget<0:
        raise ValueError("nonnegative weights, seed counts, and budget required")
    if not np.any(W):
        return np.full(len(W),budget/len(W))
    def allocation(loglam):
        return np.maximum(np.sqrt(W)*np.exp(-loglam/2)-k,0)
    root = optimize.brentq(lambda x: allocation(x).sum()-budget,-150,150,xtol=1e-13)
    return allocation(root)


def allocation_objective(W,k,n):
    W,N = np.asarray(W),np.asarray(k)+np.asarray(n)
    if np.any((W>0)&(N==0)):
        return np.inf
    return float(np.sum(np.divide(W,N,out=np.zeros_like(W,dtype=float),where=N>0)))


def integer_water(W,k,budget):
    """Round and exchange until the discrete marginal-gain optimality condition holds."""
    x = np.floor(water_fill(W,k,budget)).astype(int)
    for _ in range(budget-int(x.sum())):
        N = k+x
        gain = np.divide(W,N*(N+1),out=np.full_like(W,np.inf),where=N>0)
        x[np.argmax(gain)] += 1
    for _ in range(10000):
        N = k+x
        gain = np.divide(W,N*(N+1),out=np.full_like(W,np.inf),where=N>0)
        cost = np.divide(W,N*(N-1),out=np.full_like(W,np.inf),where=(x>0)&(N>1))
        j,i = np.argmax(gain),np.argmin(cost)
        if i == j or gain[j] <= cost[i]*(1+1e-13):
            break
        x[i] -= 1; x[j] += 1
    return x


def sensitivity_run(regime, quick=False):
    configs = {
        "frequent": (.999,np.geomspace(.005,.4,7),20000,30000,384,False),
        "isolated": (.9,np.geomspace(.0005,.01,6),2000,60000,384,False),
        "transition": (.999,np.geomspace(.0003,.02,7),20000,30000,384,False),
        "short_bias_corrected": (.999,np.geomspace(.0005,.01,6),0,200,20000,True),
    }
    beta,d,burn,R,reps,bc = configs[regime]
    if quick:
        burn = min(burn,2000); R = min(R,1500); reps = 800 if bc else 48
    rng = rng_for(3,1,list(configs).index(regime))
    S = len(d); b1=.9; pi=.35; eta=.01; eps=1e-12
    # Categorical draws include one background state. Only target-owned coordinates are stored.
    cum = np.cumsum(d)
    m = np.zeros((reps,S)); v=m.copy()
    ri = np.arange(reps)
    for _ in range(burn):
        state = np.searchsorted(cum,rng.random(reps))
        active = state<S
        grad = np.zeros_like(m)
        labels = rng.random(reps)<pi
        grad[ri[active],state[active]] = .2-labels[active]
        m = b1*m+(1-b1)*grad; v = beta*v+(1-beta)*grad**2
    # Four branches allow a preregistered finite-difference step-size comparison.
    perturb = np.array([-.04,.04,-.02,.02])
    m=np.broadcast_to(m,(4,reps,S)).copy(); v=np.broadcast_to(v,m.shape).copy()
    theta=np.zeros_like(m); sgd=theta.copy(); clipped=theta.copy()
    visits=np.zeros(S,int)
    for t in range(1,R+1):
        state=np.searchsorted(cum,rng.random(reps)); active=state<S
        visits += np.bincount(state[active],minlength=S)
        uniform=rng.random(reps)
        grad=np.zeros_like(m)
        for branch,dp in enumerate(perturb):
            grad[branch,ri[active],state[active]] = .2-(uniform[active]<pi+dp)
        m=b1*m+(1-b1)*grad; v=beta*v+(1-beta)*grad**2
        if bc:
            theta -= eta*(m/(1-b1**(burn+t)))/(np.sqrt(v/(1-beta**(burn+t)))+eps)
        else:
            theta -= eta*m/(np.sqrt(v)+eps)
        sgd -= eta*grad
        clipped -= eta*np.clip(grad,-.1,.1)
    results=[]
    for name,values in (("Adam",theta),("SGD",sgd),("always_clipped_SGD",clipped)):
        deriv=(values[3]-values[2])/.04
        coarse=(values[1]-values[0])/.08
        ci=mean_ci(deriv)
        means=ci["mean"]
        slopes=[]
        boot_rng=rng_for(3,2,list(configs).index(regime))
        for _ in range(250 if quick else 1000):
            boot=deriv[boot_rng.integers(reps,size=reps)].mean(axis=0)
            if np.all(boot>0):
                slopes.append(np.polyfit(np.log(d),np.log(boot),1)[0])
        slope=float(np.polyfit(np.log(d),np.log(means),1)[0]) if np.all(means>0) else None
        slope_ci=np.quantile(slopes,[.025,.975]) if slopes else np.array([-np.inf,np.inf])
        m1=.2-pi; m2=.04+.6*pi
        frequent=eta*R*np.sqrt(d)*(1/np.sqrt(m2)+.6*m1/(2*m2**1.5))
        first_only=eta*R*np.sqrt(d)/np.sqrt(m2)
        kiso=(1-b1)/(np.sqrt(1-beta)*(1-b1/np.sqrt(beta)))
        isolated=2*eta*R*d*kiso
        theory = eta*R*d if name=="SGD" else .2*eta*R*d if name=="always_clipped_SGD" else frequent
        results.append(dict(regime=regime,learner=name,beta1=b1,beta2=beta,bias_correction=bc,eps=eps,
                            d=d,background_probability=1-d.sum(),pi=pi,c0=.2,zeta=np.ones(S),eta=eta,
                            burn_in=burn,R=R,visits=visits,**ci,slope=slope,slope_ci95=slope_ci,
                            finite_difference_delta=.02,coarse_delta=.04,coarse_mean=coarse.mean(axis=0),
                            finite_difference_shift=mean_ci(deriv-coarse),predicted_sensitivity=theory,
                            frequent_prediction=frequent,without_second_moment_term=first_only,
                            isolated_prediction=isolated,kappa_iso=kiso,
                            frequent_relative_error=means/frequent-1,isolated_relative_error=means/isolated-1,
                            d_over_one_minus_beta2=d/(1-beta),effective_visits=d*min(burn+R,1/(1-beta)),
                            inferred_W=pi*(1-pi)*means**2,weight_slope=2*slope if slope is not None else None))
    return results


def c3(quick=False):
    sensitivity=Check("C3.sensitivity", "Proposition 3.11: CRN sensitivities, 95% CIs, slopes 1 / 1/2 / 1; finite-regime influence errors reported")
    for regime in ("frequent","isolated","transition","short_bias_corrected"):
        for row in sensitivity_run(regime,quick):
            idx=len(sensitivity.data); sensitivity.data.append(row)
            expected = 1. if row["learner"]!="Adam" else .5 if regime=="frequent" else 1. if regime in ("isolated","short_bias_corrected") else None
            if expected is not None:
                low,high=row["slope_ci95"]
                sensitivity.assert_le("slope_outside_95CI_plus_0_1",max(low-expected,expected-high),.1,idx)
            sensitivity.assert_le("positive_sensitivity",-np.min(row["mean"]),0,idx)
            if row["learner"]!="Adam":
                target=row["predicted_sensitivity"]
                z=abs(row["mean"]-target)/np.maximum(row["se"],1e-30)
                sensitivity.assert_le("linear_learner_MC_z",z,4.5,idx)
    allocation=Check("C3.allocation", "Proposition 3.10: water-filling optimum, integer realized-risk ranking and predicted MSE/MAE ratios within simultaneous MC error")
    profiles=2 if quick else 16
    reps=5000 if quick else 60000
    # Familywise 95% comparison coverage; each raw arm also reports a pointwise 95% CI.
    zcrit=stats.norm.ppf(1-.05/(2*profiles*5*2))
    for case in range(profiles):
        rng=rng_for(3,3,case); S=6+case%5
        d=np.geomspace(.002,.5,S); d/=d.sum(); rng.shuffle(d)
        dT=rng.dirichlet(np.ones(S)*.7)
        pi=rng.uniform(.15,.85,S)
        W=d*np.exp(rng.normal(0,1.2,S))
        coefficient=np.sqrt(W/(pi*(1-pi)))
        k=round_counts(dT,800); budget=4000
        continuous=water_fill(W,k,budget)
        allocations={"water":integer_water(W,k,budget),"uniform":round_counts(np.ones(S),budget),
                     "proportional_d":round_counts(d,budget),"SGD_weights":integer_water(W*d,k,budget),
                     "gradient_norm":round_counts(d*np.sqrt(.04+.6*pi),budget),
                     "uncertainty":round_counts(pi*(1-pi),budget)}
        arms={}; errors={}
        for name,n in allocations.items():
            N=k+n
            empirical=rng.binomial(N,pi,size=(reps,S))/N
            error=(empirical-pi)@coefficient
            errors[name]=error
            arms[name]=dict(n=n,predicted_variance=allocation_objective(W,k,n),
                            mae=mean_ci(abs(error)),mse=mean_ci(error**2))
        row=dict(d=d,d_T=dT,W=W,pi=pi,functional_coefficients=coefficient,k=k,budget=budget,
                 continuous_allocation=continuous,continuous_objective=allocation_objective(W,k,continuous),
                 arms=arms,ratios={},simultaneous_z=zcrit)
        idx=len(allocation.data); allocation.data.append(row)
        allocation.assert_le("allocation_budget",abs(continuous.sum()-budget),1e-8,idx)
        marginal=W/(k+continuous)**2
        active=continuous>1e-7; lam=np.mean(marginal[active])
        allocation.assert_le("water_KKT",max(np.max(abs(marginal[active]/lam-1)),
                                             np.max(marginal[~active]/lam-1,initial=0)),1e-8,idx)
        for name in allocations:
            if name=="water": continue
            a,b=arms[name],arms["water"]
            allocation.assert_le("continuous_optimum",row["continuous_objective"]-a["predicted_variance"],1e-12,[idx,name])
            allocation.assert_le("integer_optimum",b["predicted_variance"]-a["predicted_variance"],1e-12,[idx,name])
            allocation.assert_le("lowest_realized_MAE",b["mae"]["mean"]-a["mae"]["mean"],0,[idx,name])
            details={}
            for metric,power in (("mae",.5),("mse",1.)):
                predicted=(a["predicted_variance"]/b["predicted_variance"])**power
                actual=a[metric]["mean"]/b[metric]["mean"]
                se=actual*np.sqrt((a[metric]["se"]/a[metric]["mean"])**2+(b[metric]["se"]/b[metric]["mean"])**2)
                details[metric]=dict(predicted=predicted,realized=actual,se=se,ci_low=actual-zcrit*se,ci_high=actual+zcrit*se)
                allocation.assert_le("ratio_MC_z",abs(actual-predicted)/se,zcrit,[idx,name,metric])
            row["ratios"][name]=details
    idx=len(allocation.data)
    allocation.data.append(dict(kind="irreducible",W=[1.,2.],k=[100.,0.],n=[0.,0.],
                                predicted_variance=allocation_objective([1.,2.],[100.,0.],[0.,0.]),
                                unseen_state_laws=[.25,.75],identical_observed_laws=True,functional_gap=.5,lower_bound=.25))
    allocation.assert_le("irreducible_is_infinite",0 if np.isinf(allocation.data[-1]["predicted_variance"]) else 1,0,idx)

    sep=Check("C3.separation", "Corollary 3.12(a): always-clipped scale invariance and consistent alpha estimation; SGD aliasing floor for every K")
    alpha=.65; scales=np.array([1.,3.]); etaR=1.; clip=.1
    gap=(scales[1]-scales[0])*(alpha-.2)
    rng=rng_for(3,4); reps=5000 if quick else 60000
    for K in (1,10,100,1000,10000,100000):
        ahat=rng.binomial(K,alpha,reps)/K
        clip_error=abs(2*clip*(ahat-alpha))
        sgd_hat=2*(ahat-.2)
        errors=abs(sgd_hat[:,None]-scales*(alpha-.2))
        idx=len(sep.data)
        sep.data.append(dict(K=K,alpha=alpha,b=scales,gradient_support=[.2,-.8],clip=clip,etaR=etaR,
                             clipped_mae=mean_ci(clip_error),clipped_RMS_upper=2*clip*np.sqrt(alpha*(1-alpha)/K),
                             SGD_world_risks=errors.mean(axis=0),SGD_worst_risk=errors.mean(axis=0).max(),
                             functional_gap=gap,aliasing_floor=gap/2,
                             identical_demonstration_law="Binomial(K, alpha), independent of b"))
        sep.assert_le("SGD_aliasing_floor",gap/2-errors.mean(axis=0).max(),1e-12,idx)
        sep.assert_le("clipped_RMS_upper_with_MC",clip_error.mean()-2*clip*np.sqrt(alpha*(1-alpha)/K)-4*clip_error.std(ddof=1)/np.sqrt(reps),0,idx)
        clipped_support=np.clip(scales[:,None]*np.array([.2,-.8]),-clip,clip)
        sep.assert_le("scale_invariance_exact",abs(clipped_support[0]-clipped_support[1]),0,idx)
    return [sensitivity,allocation,sep]


@dataclass
class AdamConfig:
    b1: float=.9
    b2: float=.999
    eta: float=.01
    eps: float=1e-8
    bias_correction: bool=False
    weight_decay: float=0.
    clip: float|None=None


def loss_gradient(theta, loss):
    if loss["kind"]=="quadratic":
        return loss["H"]@theta-loss["a"],loss["H"]
    X,y=loss["X"],loss["y"]
    prob=special.expit(X@theta)
    return X.T@(prob-y)/len(y),(X.T*(prob*(1-prob)))@X/len(y)


def adam_step(state,g,cfg,step):
    theta,m,v=state
    mp=cfg.b1*m+(1-cfg.b1)*g
    vp=cfg.b2*v+(1-cfg.b2)*g*g
    a=1/(1-cfg.b1**step) if cfg.bias_correction else 1.
    b=1/(1-cfg.b2**step) if cfg.bias_correction else 1.
    tp=(1-cfg.eta*cfg.weight_decay)*theta-cfg.eta*a*mp/(np.sqrt(b*vp)+cfg.eps)
    return tp,mp,vp


def rollout_costate(state,loss,cfg,steps,start_step,outcome,reverse=True,with_pattern=False):
    history=[];pattern=[]
    for j in range(steps):
        g,H=loss_gradient(state[0],loss)
        active=cfg.clip is not None and np.linalg.norm(g)>cfg.clip
        pattern.append(bool(active))
        if active:
            norm=np.linalg.norm(g)
            H=cfg.clip/norm*(np.eye(len(g))-np.outer(g,g)/norm**2)@H
            g=g*cfg.clip/norm
        step=start_step+j+1
        state=adam_step(state,g,cfg,step)
        if reverse: history.append((g,H,state[1],state[2],step))
    value=float(special.expit(outcome@state[0]))
    if not reverse: return (value,pattern) if with_pattern else value
    ct=value*(1-value)*outcome
    cm=np.zeros_like(ct); cv=cm.copy()
    for g,H,mp,vp,step in reversed(history):
        a=1/(1-cfg.b1**step) if cfg.bias_correction else 1.
        b=1/(1-cfg.b2**step) if cfg.bias_correction else 1.
        root=np.sqrt(b*vp)
        D1=a/(root+cfg.eps)
        D2=a*mp*b/(2*root*(root+cfg.eps)**2)
        um=cm-cfg.eta*D1*ct
        uv=cv+cfg.eta*D2*ct
        ct=(1-cfg.eta*cfg.weight_decay)*ct+H.T@((1-cfg.b1)*um+2*(1-cfg.b2)*g*uv)
        cm=cfg.b1*um; cv=cfg.b2*uv
    result=(value,np.stack((ct,cm,cv)))
    return (*result,pattern) if with_pattern else result


def amplitude_formula(lam,b1,b2,T,rho,eta=1.,alpha=1.):
    """Corollary 4.2(b): small-increment quadratic, not the finite response."""
    return alpha*eta*((1-b1**(T+1))*np.asarray(lam)-.5*rho*(1-b2**(T+1))*np.asarray(lam)**2)


def mean_field_response(lam,b1,b2,T,rho,eta=1.,alpha=1.):
    """Corollary 4.2(a), relative to a zero-gradient message; j=0 is included."""
    j=np.arange(T+1)
    b=(1-b1)*b1**j; c=(1-b2)*b2**j
    lam=np.asarray(lam)[...,None]
    return alpha*eta*np.sum((rho+b*(lam-rho))/np.sqrt(1+c*(lam**2-1))
                           -rho*(1-b)/np.sqrt(1-c),axis=-1)


def amplitude_limit(b1,b2,T,rho,eta=1.,alpha=1.):
    """Corollary 4.2(c), with exact and approximate critical rho reported apart."""
    j=np.arange(T+1)
    b=(1-b1)*b1**j; c=(1-b2)*b2**j
    kiso=float(np.sum(b/np.sqrt(c)))
    baseline=float(np.sum((1-b)/np.sqrt(1-c)))
    limit=alpha*eta*(kiso-rho*baseline)
    return dict(value=limit,kappa_iso=kiso,baseline_coefficient=baseline,
                rho_critical=kiso/baseline,rho_critical_approx=kiso/(T+1),
                predicts_crossing=bool(limit<0))


def quadratic_region(lambdas,b2,lambda_star):
    lambdas=np.asarray(lambdas)
    return ((1-b2)*lambdas**2<=.02+1e-15)&(abs(lambdas)<=4), (1-b2)*(2*lambda_star)**2<=.1+1e-15


def parameter_only_derivative_sign(lam,b1,b2,rho):
    return (1-b1)*b2-b1*rho*(1-b2)*np.asarray(lam)


def positive_crossing(response,initial_hi,max_evaluations=17,upper_bound=None):
    """Search the nontrivial positive root, retaining every tested upper endpoint.

    Unbracketed means no sign change on this bounded search, never an infinite
    measured root. upper_bound permits the formula to use the measured bound.
    """
    lo=1e-6; low=float(response(lo)); bracket=[]
    hi=initial_hi
    for _ in range(max_evaluations):
        if upper_bound is not None: hi=min(hi,upper_bound)
        value=float(response(hi)); bracket.append([hi,value])
        if low>0 and value<=0:
            root=optimize.brentq(response,lo,hi,xtol=1e-8)
            return dict(root=root,status="bracketed",search=bracket,search_bound=hi,
                        lower_endpoint=[lo,low])
        if upper_bound is not None and hi>=upper_bound: break
        hi*=2
    return dict(root=None,status="not_bracketed_on_recorded_search",search=bracket,
                search_bound=bracket[-1][0],lower_endpoint=[lo,low])


def c4_costate(quick=False):
    check=Check("C4.costate", "Theorem 4.1: exact reverse derivative vs central differences, including bias correction, AdamW, clipping")
    grid=itertools.product(("quadratic","logistic"),(5,) if quick else (5,20), (False,True),
                           (1e-8,1e-3),(0.,.03),(0.,.9),(.99,.999),(None,.15))
    for idx,(kind,p,bc,eps,wd,b1,b2,clip) in enumerate(grid):
        if quick and idx%13: continue
        rng=rng_for(4,1,idx)
        Q=rng.normal(size=(p,p))/np.sqrt(p)
        loss=dict(kind=kind,H=Q.T@Q+.2*np.eye(p),a=rng.normal(size=p)*.2,
                  X=rng.normal(size=(32,p)),y=rng.integers(0,2,size=32))
        cfg=AdamConfig(b1,b2,.007,eps,bc,wd,clip)
        state=np.stack((rng.normal(size=p)*.2,rng.normal(size=p)*.1,rng.uniform(.3,1.2,size=p)))
        outcome=rng.normal(size=p)/np.sqrt(p)
        steps=(20,60,200)[idx%3] if not quick else 8
        value,adj,base_pattern=rollout_costate(state,loss,cfg,steps,7,outcome,with_pattern=True)
        directions=[]
        for block in range(4):
            direction=rng.normal(size=(3,p))
            if block<3: direction[np.arange(3)!=block]=0
            direction/=np.linalg.norm(direction)
            # Choose a smooth clipping branch using only forward trajectories,
            # never by agreement with the adjoint. Preserve all rejected steps.
            ladder=[]
            def central(h):
                plus,pp=rollout_costate(state+h*direction,loss,cfg,steps,7,outcome,False,True)
                minus,pm=rollout_costate(state-h*direction,loss,cfg,steps,7,outcome,False,True)
                row=dict(step=h,central_difference=(plus-minus)/(2*h),same_clipping_branch=pp==pm==base_pattern)
                ladder.append(row)
                return row
            original=central(2e-5)
            h=2e-4
            coarse=central(h);fine=central(h/2)
            while not (coarse["same_clipping_branch"] and fine["same_clipping_branch"]) and h>1e-7:
                h/=2;coarse=fine;fine=central(h/2)
            fd=(4*fine["central_difference"]-coarse["central_difference"])/3
            predicted=float(np.sum(adj*direction))
            residual=abs(fd-predicted)/max(abs(fd),abs(predicted),1e-10)
            directions.append(dict(block=("theta","m","v","joint")[block],direction=direction,
                                   central_difference_ladder=ladder,richardson_central_difference=fd,
                                   selected_coarse_step=h,
                                   original_step_relative_error=abs(original["central_difference"]-predicted)/max(abs(predicted),1e-10),
                                   costate_projection=predicted,relative_error=residual))
            check.assert_le("directional_relative_error",residual,1e-6,[len(check.data),block])
            check.assert_le("finite_difference_same_clipping_branch",0 if coarse["same_clipping_branch"] and fine["same_clipping_branch"] else 1,0,[len(check.data),block])
        check.data.append(dict(kind=kind,p=p,config=vars(cfg),steps=steps,start_step=7,initial_state=state,
                               loss=loss,outcome_vector=outcome,value=value,costate=adj,
                               clipping_pattern=base_pattern,directions=directions))
    return check


def scalar_amplitude_experiment(b2,rho,Ts,reps,seedpath,lambdas,quick=False,root_reps=None):
    """Exact CRN rollouts AND exact first-order Val evaluated at the pre-message state.

    The exogenous tail admits explicit derivatives dm_j/dm_0=b1^j and
    dv_j/dv_0=b2^j. Accumulate these without using the mean-field formula.
    """
    rng=rng_for(*seedpath); b1=.9; sigma=np.sqrt(1-rho*rho)
    burn=int(np.ceil(12/(1-b2)))
    m=np.zeros(reps);v=m.copy()
    # Boundary beta=.99999 also gets a genuinely long burn; block filtering avoids Python overhead.
    remaining=burn
    while remaining:
        block=min(512,remaining)
        g=rng.normal(rho,sigma,size=(block,reps))
        _,zm=signal.lfilter([1-b1],[1,-b1],g,axis=0,zi=(b1*m)[None,:])
        _,zv=signal.lfilter([1-b2],[1,-b2],g*g,axis=0,zi=(b2*v)[None,:])
        m=zm[0]/b1;v=zv[0]/b2
        remaining-=block
    warm_m,warm_v=m.copy(),v.copy()
    rms=np.sqrt(v)
    lam=np.r_[0.,lambdas]
    message=lam[:,None]*rms
    mb=b1*m+(1-b1)*message
    vb=b2*v+(1-b2)*message**2
    immediate=mb/np.sqrt(vb)
    payoff=immediate.copy()  # eta=alpha=1, F=-theta, exact linear response
    # Reference trajectory starts from the pre-message state; message is not included in the tail.
    mr=m.copy();vr=v.copy();chim=np.zeros(reps);chiv=np.zeros(reps)
    # A fixed leading subset of the SAME CRN paths supports independent root
    # evaluation at arbitrary lambda. No reseeding or new tail realization.
    root_reps=min(reps,512 if root_reps is None else root_reps)
    zero_m=np.empty((max(Ts)+1,root_reps));zero_v=np.empty_like(zero_m)
    zero_m[0]=mb[0,:root_reps];zero_v[0]=vb[0,:root_reps]
    output={}
    for T in range(max(Ts)+1):
        if T in Ts:
            # Difference of exact update increments; zero-message terms cancel in m and v.
            val=(immediate-immediate[0])+chim[None,:]*(1-b1)*message+chiv[None,:]*(1-b2)*message**2
            output[T]=dict(rollout_samples=(payoff-payoff[0])[1:].T.copy(),
                           val_samples=val[1:].T.copy(),chi_m=chim.copy(),chi_v=chiv.copy(),
                           immediate_samples=(immediate-immediate[0])[1:].T.copy())
        if T==max(Ts):break
        g=rng.normal(rho,sigma,reps)
        mr=b1*mr+(1-b1)*g;vr=b2*vr+(1-b2)*g*g
        j=T+1
        chim+=b1**j/np.sqrt(vr)
        chiv-=b2**j*mr/(2*vr**1.5)
        mb=b1*mb+(1-b1)*g;vb=b2*vb+(1-b2)*g*g
        payoff+=mb/np.sqrt(vb)
        zero_m[j]=mb[0,:root_reps];zero_v[j]=vb[0,:root_reps]
    return output,dict(m=warm_m,v=warm_v,burn_in=burn,n_replicates=reps,beta1=b1,beta2=b2,rho=rho,
                       zero_m=zero_m,zero_v=zero_v,root_replicates=root_reps)


def audit_amplitude_setting(check,b2,rho,T,lambdas,output,warm,kind):
    """Audit exact mean-field, local quadratic, and asymptotic claims separately."""
    b1=.9
    m,v=warm['m'],warm['v']; root_reps=warm['root_replicates']
    samples={name:output[name][:,:len(lambdas)]
             for name in ('val_samples','rollout_samples')}
    values=mean_ci(samples['val_samples']); rollouts=mean_ci(samples['rollout_samples'])
    predicted=amplitude_formula(lambdas,b1,b2,T,rho)
    formula=mean_field_response(lambdas,b1,b2,T,rho)
    relative=abs(values['mean']-predicted)/np.maximum(abs(predicted),1e-14)
    rollout_relative=abs(rollouts['mean']-predicted)/np.maximum(abs(predicted),1e-14)
    # One common denominator for the entire lambda grid, including signed responses.
    response_scale=float(np.max(abs(rollouts['mean'])))
    formula_relative=abs(rollouts['mean']-formula)/max(response_scale,1e-14)
    star=(1-b1**(T+1))/(rho*(1-b2**(T+1)))
    eligible,crossing_eligible=quadratic_region(lambdas,b2,star)
    pred_cross=2*star
    def exact_val(lam):
        msg=lam*np.sqrt(v)
        direct=(b1*m+(1-b1)*msg)/np.sqrt(b2*v+(1-b2)*msg*msg)-b1*m/np.sqrt(b2*v)
        return float(np.mean(direct+output['chi_m']*(1-b1)*msg+output['chi_v']*(1-b2)*msg*msg))
    val_root=positive_crossing(exact_val,pred_cross,max_evaluations=21)
    j=np.arange(T+1)[:,None]
    dm=(1-b1)*b1**j*np.sqrt(v[:root_reps])[None,:]
    dv=(1-b2)*b2**j*v[:root_reps][None,:]
    zero_m=warm['zero_m'][:T+1]; zero_v=warm['zero_v'][:T+1]
    baseline=zero_m/np.sqrt(zero_v)
    def rollout_at(lam):
        return np.sum((zero_m+dm*lam)/np.sqrt(zero_v+dv*lam*lam)-baseline,axis=0)
    rollout_root=positive_crossing(lambda x:float(rollout_at(x).mean()),pred_cross)
    rollout_cross=rollout_root['root']
    # If the measured crossing is unbracketed, test the formula up to exactly
    # the same bound. Otherwise allow its own bracket (the roots need not be identical).
    formula_root=positive_crossing(lambda x:float(mean_field_response(x,b1,b2,T,rho)),pred_cross,
                                   upper_bound=rollout_root['search_bound'] if rollout_cross is None else None)
    formula_cross=formula_root['root']
    cross_error=abs(formula_cross/rollout_cross-1) if formula_cross is not None and rollout_cross is not None else None
    if rollout_cross is not None:
        at_root=rollout_at(rollout_cross); step=rollout_cross*1e-4
        derivative=(rollout_at(rollout_cross+step).mean()-rollout_at(rollout_cross-step).mean())/(2*step)
        root_se=float(at_root.std(ddof=1)/np.sqrt(root_reps)/abs(derivative))
    else:
        root_se=None
    limit=amplitude_limit(b1,b2,T,rho)
    # At the probe, even the last denominator is dominated by the message.
    probe=1e8/np.sqrt((1-b2)*b2**T)
    probe_value=float(mean_field_response(probe,b1,b2,T,rho))
    limit.update(probe_lambda=probe,probe_response=probe_value,
                 probe_relative_error=abs(probe_value-limit['value'])/max(abs(limit['value']),1.))
    early_probe=probe/100
    early_value=float(mean_field_response(early_probe,b1,b2,T,rho))
    limit['earlier_probe']=dict(lambda_=early_probe,response=early_value,
                                relative_error=abs(early_value-limit['value'])/max(abs(limit['value']),1.))
    measured_limit=np.sum(dm/np.sqrt(dv)-baseline,axis=0)
    limit['rollout_limit']=mean_ci(measured_limit)
    warm_summary={k:x for k,x in warm.items() if k not in ('m','v','zero_m','zero_v')}
    warm_summary.update(mean_m=m.mean(),mean_v=v.mean(),sd_m=m.std(ddof=1),sd_v=v.std(ddof=1))
    val_cross=val_root['root']
    idx=len(check.data)
    row=dict(kind=kind,beta2=b2,rho=rho,T=T,lambdas=lambdas,predicted=predicted,
             mean_field=formula,val=values,rollout=rollouts,relative_error=relative,
             rollout_relative_error=rollout_relative,eligible=eligible,warmth=(1-b2)*lambdas**2,
             T_gamma=T*(1-b2),mean_field_response_scale=response_scale,
             mean_field_rollout_relative_error=formula_relative,
             mean_field_rollout_max_relative_error=float(np.max(formula_relative)),
             predicted_zero_crossing=pred_cross,measured_Val_zero_crossing=val_cross,
             zero_crossing_relative_error=abs(val_cross/pred_cross-1) if val_cross is not None else None,
             measured_rollout_zero_crossing=rollout_cross,rollout_root_se=root_se,
             rollout_zero_crossing_relative_error=abs(rollout_cross/pred_cross-1) if rollout_cross is not None else None,
             mean_field_zero_crossing=formula_cross,mean_field_zero_crossing_relative_error=cross_error,
             quadratic_crossing_warmth=(1-b2)*pred_cross**2,quadratic_crossing_eligible=bool(crossing_eligible),
             root_search=dict(Val=val_root,rollout=rollout_root,mean_field=formula_root),
             large_lambda_limit=limit,warm=warm_summary,
             mean_chi_m=output['chi_m'].mean(),mean_chi_v=output['chi_v'].mean(),
             raw_replicates=dict(**samples,warm_m=m,warm_v=v,chi_m=output['chi_m'],chi_v=output['chi_v'],
                                 rollout_limit_samples=measured_limit))
    check.data.append(row)
    if root_reps>512:
        # Preserve the noisier original subset estimate. Near the limit criterion,
        # the root is sensitive to noise; use all existing short-horizon paths.
        subset_root=positive_crossing(lambda x:float(rollout_at(x)[:512].mean()),pred_cross)
        subset_cross=subset_root['root']
        subset_root['n_replicates']=512
        subset_root['mean_field_relative_error']=abs(formula_cross/subset_cross-1) if formula_cross is not None and subset_cross is not None else None
        if subset_cross is not None:
            step=subset_cross*1e-4
            slope=(rollout_at(subset_cross+step)[:512].mean()-rollout_at(subset_cross-step)[:512].mean())/(2*step)
            subset_root['se']=float(rollout_at(subset_cross)[:512].std(ddof=1)/np.sqrt(512)/abs(slope))
        row['rollout_root_first_512_diagnostic']=subset_root
    check.assert_le('mean_field_rollout_relative_error_all_lambda',formula_relative,.05,idx)
    if rollout_cross is not None:
        check.assert_le('mean_field_zero_crossing_relative_error',cross_error if cross_error is not None else np.inf,.1,idx)
    else:
        check.assert_le('mean_field_no_crossing_to_measured_bound',int(formula_cross is not None),0,idx)
        check.assert_le('unbracketed_search_bounds_agree',abs(formula_root['search_bound']/rollout_root['search_bound']-1),0,idx)
    if np.any(eligible):
        check.assert_le('Val_relative_error_eligible',relative[eligible],.05,idx)
        check.assert_le('rollout_relative_error_eligible',rollout_relative[eligible],.05,idx)
    if crossing_eligible:
        check.assert_le('zero_crossing_relative_error',row['zero_crossing_relative_error'] if val_cross is not None else np.inf,.1,idx)
        check.assert_le('rollout_zero_crossing_relative_error',row['rollout_zero_crossing_relative_error'] if rollout_cross is not None else np.inf,.1,idx)
    check.assert_le('large_lambda_limit_convergence',limit['probe_relative_error'],1e-5,idx)
    check.assert_le('limit_predicts_rollout_crossing',int(limit['predicts_crossing']!=(rollout_cross is not None)),0,idx)
    check.assert_le('limit_predicts_formula_crossing',int(limit['predicts_crossing']!=(formula_cross is not None)),0,idx)
    check.assert_le('limit_sign_agrees_with_rollout_limit',int(limit['predicts_crossing']!=(measured_limit.mean()<0)),0,idx)


def c4_amplitude(quick=False):
    check=Check('C4.amplitude', 'Corollary 4.2(a-c): exact mean-field response over the whole grid and measured crossings; quadratic checks only in the small-increment region; large-lambda crossing criterion')
    ranking=Check('C4.ranking', 'Proposition 4.3(b): positive parameter-only derivative on both pair endpoints, parameter-only reversal, cosine tie, exact Val and full-tail response preference')
    lambdas=np.array([-2.,.5,1.,2.,4.,8.,16.])
    Ts=(50,200,1000,5000)
    reps=128 if quick else 4096
    combos=list(itertools.product((.99,.999),(.3,.7,.95)))
    if quick: combos=[(.999,.7)]
    for combo,(b2,rho) in enumerate(combos):
        # Preserve the existing ranking pairs, random streams, and comparisons.
        stars=np.array([(1-.9**(T+1))/(rho*(1-b2**(T+1))) for T in Ts])
        pairs=np.c_[1.25*stars,1.75*stars]
        all_lambdas=np.r_[lambdas,pairs.ravel()]
        outputs,warm=scalar_amplitude_experiment(b2,rho,Ts,reps,(4,2,combo),all_lambdas,quick)
        for ti,T in enumerate(Ts):
            output=outputs[T]
            audit_amplitude_setting(check,b2,rho,T,lambdas,output,warm,'core')
            i=len(lambdas)+2*ti;pair=pairs[ti]
            po=output['immediate_samples'][:,i:i+2]
            val=output['val_samples'][:,i:i+2]; rollout=output['rollout_samples'][:,i:i+2]
            sign=parameter_only_derivative_sign(pair,.9,b2,rho)
            ridx=len(ranking.data)
            ranking.data.append(dict(beta2=b2,rho=rho,T=T,lambda1=pair[0],lambda2=pair[1],lambda_star=stars[ti],
                                     sum_minus_2star=pair.sum()-2*stars[ti],warmth=(1-b2)*pair**2,
                                     parameter_only_derivative_sign=sign,
                                     parameter_only=mean_ci(po),cosine=[1.,1.],Val=mean_ci(val),
                                     rollout=mean_ci(rollout),Val_pair_difference=mean_ci(val[:,0]-val[:,1]),
                                     rollout_pair_difference=mean_ci(rollout[:,0]-rollout[:,1]),
                                     raw_replicates=dict(parameter_only_samples=po,val_samples=val,rollout_samples=rollout)))
            ranking.assert_le('parameter_only_derivative_nonpositive',sign<=0,0,ridx)
            ranking.assert_le('parameter_only_orders_larger_first',po[:,0].mean()-po[:,1].mean(),0,ridx)
            ranking.assert_le('cosine_tie',0,0,ridx)
            ranking.assert_le('exact_Val_orders_smaller_first',val[:,1].mean()-val[:,0].mean(),0,ridx)
            ranking.assert_le('rollout_orders_smaller_first',rollout[:,1].mean()-rollout[:,0].mean(),0,ridx)
    # Predeclared short horizons and rho values on both sides of kappa_iso/(T+1).
    # beta2=.99 keeps all six rho values in (0,1], including T=20.
    for case,(T,factor) in enumerate(itertools.product((20,50,100),(.8,1.2))):
        b2=.99
        rho=factor*amplitude_limit(.9,b2,T,1.)['rho_critical_approx']
        outputs,warm=scalar_amplitude_experiment(b2,rho,(T,),reps,(4,4,case),lambdas,quick,root_reps=reps)
        audit_amplitude_setting(check,b2,rho,T,lambdas,outputs[T],warm,'criterion_straddle')
        check.data[-1]['rho_over_approx_critical']=factor
    # Retain the original small-rho/short-tail diagnostics; also test the newly
    # requested exact formula and limit criterion here. Quadratic errors remain
    # descriptive because lambda=20 is outside the asserted local region.
    boundary_reps=16 if quick else 128
    outputs,warm=scalar_amplitude_experiment(.99999,.05,(0,1,5),boundary_reps,(4,3),np.array([20.]),quick)
    for T,output in outputs.items():
        audit_amplitude_setting(check,.99999,.05,T,np.array([20.]),output,warm,'boundary_quadratic_diagnostic')
    return [check,ranking]


def validate_results(result):
    required={"claim_id","asserted","n_instances","n_assertions","worst_residual","status","data","failures","assertions"}
    if result.get("schema_version")!="v016-checks-1": raise ValueError("schema_version")
    if not isinstance(result.get("checks"),list) or not result["checks"]: raise ValueError("checks")
    ids=[]
    for check in result["checks"]:
        if not required<=check.keys(): raise ValueError("missing check fields")
        if check["status"] not in ("PASS","FAIL"): raise ValueError("status")
        if check["n_instances"]!=len(check["data"]): raise ValueError("instance count")
        if check["n_assertions"]!=sum(a["n"] for a in check["assertions"].values()): raise ValueError("assertion count")
        if bool(check["failures"])!=(check["status"]=="FAIL"): raise ValueError("hidden failure")
        ids.append(check["claim_id"])
    if len(ids)!=len(set(ids)): raise ValueError("duplicate claim id")
    wanted="FAIL" if any(c["status"]=="FAIL" for c in result["checks"]) else "PASS"
    if result.get("status")!=wanted: raise ValueError("hidden suite failure")
    json.dumps(result,allow_nan=False)


def summarize_results(result):
    """Lossless raw companion plus a compact reviewable view; no check is rerun."""
    summary={**result,'checks':[]}
    raw=dict(schema_version='v016-checks-raw-1',checks={})
    raw_fields={
        'C1.exact': {'s','raw_v','observed','kbar','lower','upper','second_moment',
                     'expected_v','stationary_substitution'},
        'C1.accumulation': {'R','lhs','rhs'},
        'C2.design': {'A_D','features','c'},
        'C2.aliasing': {'A_D'},
        'C4.costate': {'initial_state','loss','outcome_vector','costate','clipping_pattern'},
    }
    for check in result['checks']:
        claim=check['claim_id']
        raw['checks'][claim]=check['data']
        rows=[]
        for idx,row in enumerate(check['data']):
            short={k:v for k,v in row.items() if k not in raw_fields.get(claim,set()) and k!='raw_replicates'}
            short['raw_ref']=f'/checks/{claim}/{idx}'
            if claim=='C1.exact':
                short['n_grid_points']=len(row['s'])
                short['s_range']=[row['s'][0],row['s'][-1]]
                if 'observed' in row:
                    short['max_recursion_relative_error']=float(np.max(abs(np.asarray(row['observed'])/row['kbar']-1)))
                    short['observed_range']=[min(row['observed']),max(row['observed'])]
            if claim=='C1.accumulation' and 'R' in row:
                short['n_grid_points']=len(row['R'])
                short['R_range']=[row['R'][0],row['R'][-1]]
                short['max_bound_residual']=float(np.max(np.asarray(row['lhs'])-row['rhs']))
                short['final_lhs']=row['lhs'][-1];short['final_rhs']=row['rhs'][-1]
            if claim=='C4.costate':
                short['directions']=[{k:v for k,v in d.items() if k not in ('direction','central_difference_ladder')}
                                     for d in row['directions']]
                short['finite_difference_attempts']=[len(d['central_difference_ladder']) for d in row['directions']]
                short['rejected_branch_attempts']=[sum(not a['same_clipping_branch'] for a in d['central_difference_ladder'])
                                                   for d in row['directions']]
            rows.append(short)
        summary['checks'].append({**check,'data':rows})
    return summary,raw


def write_results(result,output):
    """Deterministic gzip (no filename or timestamp), strict JSON, <1 MB summary."""
    output=Path(output)
    raw_path=output.with_name(output.stem+'_raw.json.gz')
    summary,raw=summarize_results(result)
    encoder=json.JSONEncoder(allow_nan=False,separators=(',',':'))
    with raw_path.open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as zipped:
            # Batch encoder chunks: avoid a write syscall for every scalar.
            buffer=[];size=0
            for chunk in encoder.iterencode(raw):
                buffer.append(chunk);size+=len(chunk)
                if size>=1024*1024:
                    zipped.write(''.join(buffer).encode('utf-8'));buffer=[];size=0
            zipped.write((''.join(buffer)+'\n').encode('utf-8'))
    summary['raw_artifact']=dict(path=raw_path.name,schema_version=raw['schema_version'],
                                 sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                                 bytes=raw_path.stat().st_size,gzip_mtime=0,
                                 references='JSON pointers from each data row into the decompressed companion')
    validate_results(summary)
    payload=(encoder.encode(summary)+'\n').encode('utf-8')
    if len(payload)>=1_000_000:
        raise ValueError('summary must remain below 1 MB; raw data are in the companion')
    output.write_bytes(payload)
    return summary


def run_suite(quick=False,only=None):
    start=time.perf_counter(); checks=[]; runtimes={}
    groups={"C1":c1,"C2":c2,"C3":c3,"C4.costate":lambda q:[c4_costate(q)],"C4.amplitude":c4_amplitude}
    for group,fn in groups.items():
        if only and not any(group==x or group.startswith(x+".") for x in only): continue
        t=time.perf_counter()
        print(f"Running {group} (quick={quick})",flush=True)
        try:
            found=fn(quick)
        except Exception as exc:
            # Numerical/programming failures remain visible and do not suppress later groups.
            failed=Check(group,"Group execution must complete")
            failed.data.append(dict(exception_type=type(exc).__name__))
            failed.assert_le("execution_completed",1,0,0)
            found=[failed]
        runtimes[group]=time.perf_counter()-t
        for check in found:
            r=check.result(); checks.append(r)
            print(f"  {r['claim_id']}: {r['status']}; assertions={r['n_assertions']}; worst={r['worst_residual']:.6g}",flush=True)
    paths=[HERE/"SPEC.md",HERE/"check_v016.py",HERE/"test_check_v016.py",
           ROOT/"docs/direction/2026-10-11_v016_revision_ZH.md"]
    result=dict(schema_version="v016-checks-1",status="FAIL" if any(x["status"]=="FAIL" for x in checks) else "PASS",
                checks=checks,runtime_seconds=time.perf_counter()-start,group_runtime_seconds=runtimes,
                config=dict(quick=quick,only=only,master_seed=MASTER_SEED,rng="numpy.SeedSequence([master, group, subcase, ...]); PCG64",
                            dtype="float64",device="CPU",CUDA_VISIBLE_DEVICES="",network=False,teacher_calls=0,model_checkpoints=None),
                versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()},
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                scope="Synthetic numerical instances only; K counts independent synthetic demonstrations, not real trajectories. No training or benchmark claims.")
    result=clean(result);validate_results(result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--only",help="Comma-separated C1,C2,C3,C4,C4.costate,C4.amplitude")
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args(argv)
    result=run_suite(args.quick,args.only.split(",") if args.only else None)
    output=args.output or (Path("/tmp/v016_quick_results.json") if args.quick else HERE/"results_v016.json")
    summary=write_results(result,output)
    print(f"{result['status']}; {result['runtime_seconds']:.2f}s; {output}",flush=True)
    print(f"Files: {output.stat().st_size} bytes summary; {summary['raw_artifact']['bytes']} bytes raw gzip",flush=True)
    return 1 if result["status"]=="FAIL" else 0


if __name__=="__main__":
    raise SystemExit(main())
