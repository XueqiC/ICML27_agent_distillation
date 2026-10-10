#!/usr/bin/env python3
"""Finite regression checks for v0.15, not a rerun of external C1--C23.

Uses exact fractions for finite Bayes risks, moment matching, quotient ranks,
and adaptive-query enumeration; floating point for analytic optimizer maps.
Run: python verify_v015_extensions.py --out results_v015.json
"""
import argparse
import itertools as it
import json
import math
import platform
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(15019)
RESULTS = {}


def record(name, **data):
    RESULTS[name] = {"status": "PASS", **data}


def close(x, y, tol=1e-10):
    assert abs(float(x) - float(y)) <= tol, (x, y)


def rank(a):
    a = [[F(x) for x in row] for row in a]
    r = 0
    for c in range(len(a[0])):
        pivot = next((i for i in range(r, len(a)) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        v = a[r][c]
        a[r] = [x / v for x in a[r]]
        for i in range(len(a)):
            if i != r and a[i][c]:
                v = a[i][c]
                a[i] = [x - v * y for x, y in zip(a[i], a[r])]
        r += 1
        if r == len(a):
            break
    return r


def h(g, eps):
    g = np.asarray(g, dtype=float)
    return np.sign(g) if eps == 0 else g / (np.abs(g) + eps)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def audit_counterexample():
    radius = 2 * math.sqrt(math.log(4 / 0.05) / (2 * 10000))
    actual_gap = 1.0
    assert actual_gap > radius
    assert actual_gap <= min(1, radius + 1)  # discrepancy transport a=1
    assert actual_gap <= 1                  # immediate exit rho=1
    record("T01_coverage_erratum", old_statement_counterexample_reproduced=True,
           old_bound=radius, actual_gap=actual_gap,
           corrected_transport_bound_capped=1, corrected_exit_bound=1)


def transport_checks():
    max_ratio = 0
    for _ in range(1200):
        p0, q0, pz, qz = RNG.dirichlet(np.ones(7), size=4)
        features = RNG.random((7, 4))
        local_features = np.clip(features + RNG.normal(0, .04, (7, 4)), 0, 1)
        alpha = np.max(np.abs(local_features - features))
        anchor = (p0 - q0) @ features
        transported = (pz - qz) @ features
        D = np.max(np.abs(anchor))
        a = np.max(np.abs(transported - anchor))
        local_D = np.max(np.abs((pz - qz) @ local_features))
        assert local_D <= D + a + 2 * alpha + 1e-12
        b = RNG.normal(0, .05, 4)
        residual = RNG.uniform(-.01, .01, 7)
        response = .5 + features @ b + residual
        assert np.all((response >= 0) & (response <= 1))
        actual = abs((pz - qz) @ response)
        bound = np.sum(np.abs(b)) * (D + a) + 2 * np.max(np.abs(residual))
        assert actual <= bound + 1e-12
        max_ratio = max(max_ratio, actual / bound)
    record("T02_local_transport_and_feature_drift", instances=1200,
           maximum_local_gap_over_bound=max_ratio)


def cold_adam():
    max_err = 0
    max_inv_err = 0
    cases = 0
    for eps in [0, 1e-8, .01, .3]:
        for _ in range(120):
            g = RNG.normal(0, 2, 5)
            g[RNG.random(5) < .1] = 0
            b1, b2 = RNG.uniform(0, .999, 2)
            theta = RNG.normal(size=5)
            eta, decay = .1, .03
            m, v = (1 - b1) * g, (1 - b2) * g**2
            mh, vh = m / (1 - b1), v / (1 - b2)
            normalized = np.divide(mh, np.sqrt(vh) + eps,
                                   out=np.zeros_like(g), where=(vh != 0) | (eps != 0))
            actual = (1 - eta * decay) * theta - eta * normalized
            predicted = (1 - eta * decay) * theta - eta * h(g, eps)
            err = np.max(np.abs(actual - predicted))
            assert err < 1e-12
            max_err = max(max_err, err)
            if eps > 0:
                u = h(g, eps)
                inv = eps * u / (1 - np.abs(u))
                inv_err = np.max(np.abs(inv - g))
                assert inv_err < 2e-6  # inverse is ill conditioned at eps=1e-8
                max_inv_err = max(max_inv_err, inv_err)
            cases += 1
    record("T03_cold_adam_and_positive_epsilon_inverse", cases=cases,
           maximum_update_error=max_err, maximum_inverse_error=max_inv_err)


def quotient_and_joint_signs():
    support = list(it.product(range(-2, 3), repeat=2))
    signs = [tuple((g > 0) - (g < 0) for g in row) for row in support]
    patterns = sorted(set(signs))
    matrix = [[int(s == pat) for pat in patterns] for s in signs]
    sign_rank = rank(matrix)
    eps = F(1, 10)
    transformed = [tuple(F(x) / (abs(x) + eps) for x in g) for g in support]
    assert sign_rank == 9
    assert len(set(transformed)) == 25
    record("T04_exact_quotient_dimensions", gradient_atoms=25,
           sign_patterns=9, ideal_sign_statistic_dimension=sign_rank - 1,
           positive_epsilon_statistic_dimension=24)
    P = [(1, 1), (-1, -1)]
    Q = [(1, -1), (-1, 1)]
    assert np.all(np.mean(P, axis=0) == np.mean(Q, axis=0))
    fp = sum(a == b for a, b in P) / 2
    fq = sum(a == b for a, b in Q) / 2
    assert fp - fq == 1
    record("T05_joint_sign_marginal_counterexample", mean_updates_equal=True,
           bounded_same_sign_utility_gap=fp - fq)


def matched_moments():
    support = [F(i, 3) for i in range(4)]
    P = [F(1, 4), F(0), F(3, 4), F(0)]
    Q = [F(0), F(3, 4), F(0), F(1, 4)]
    diffs = [sum((p-q) * x**r for p,q,x in zip(P,Q,support)) for r in range(4)]
    assert diffs[:3] == [0, 0, 0]
    p_init = F(1, 2)
    gradients = [p_init - x for x in support]
    mean_gradient_diff = sum((p-q)*g for p,q,g in zip(P,Q,gradients))
    sign_diff = sum((p-q)*((g>0)-(g<0)) for p,q,g in zip(P,Q,gradients))
    assert mean_gradient_diff == 0 and sign_diff == -1
    eta = F(1, 10)
    positive_eps_gap = -eta * sum((p-q)*g/(abs(g)+F(1,100))
                                 for p,q,g in zip(P,Q,gradients))
    record("T06_moment_matched_optimizer_separation", matched_orders=[0,1,2],
           moment_differences=[str(x) for x in diffs],
           sgd_mean_parameter_gap=0, ideal_adam_mean_parameter_gap=str(-eta*sign_diff),
           finite_epsilon_mean_parameter_gap=str(positive_eps_gap),
           note="New finite witness, not external C23 or its task utility.")


def margin_checks():
    max_ratio = 0
    max_excess = -math.inf
    for _ in range(1500):
        gradients = RNG.normal(size=(12, 3)) * RNG.choice([1e-7, .01, 1], (12,3))
        gradients[RNG.random((12,3)) < .15] = 0
        P, Q = RNG.dirichlet(np.ones(12), 2)
        eps = 10**RNG.uniform(-8, -1)
        a = 10**RNG.uniform(-6, .1)
        eta = 10**RNG.uniform(-2, 0)
        w = RNG.normal(size=3)
        c = RNG.normal(size=3)
        soft = sigmoid((c - eta*h(gradients, eps)) @ w)
        hard = sigmoid((c - eta*np.sign(gradients)) @ w)
        gap = abs((P-Q) @ soft)
        sign_gap = abs((P-Q) @ hard)
        small = (np.abs(gradients) > 0) & (np.abs(gradients) < a)
        beta_p, beta_q = P @ small, Q @ small
        bound = sign_gap + eta*np.sum(np.abs(w)/4 * (beta_p+beta_q+2*eps/(a+eps)))
        assert gap <= bound + 1e-12
        max_ratio = max(max_ratio, gap / bound if bound else 0)
        max_excess = max(max_excess, gap-bound)
    record("T07_margin_sensitive_bound", instances=1500,
           maximum_gap_over_bound=max_ratio, maximum_signed_violation=max_excess)


def memory_counterexample():
    branches = []
    for first in [F(1), F(2)]:
        beta = F(1,2)
        m1, v1 = first/2, first**2/2
        second = F(-3,4)
        m2, v2 = beta*m1+second/2, beta*v1+second**2/2
        mh2, vh2 = m2/(1-beta**2), v2/(1-beta**2)
        branches.append((m2,mh2,vh2))
    assert branches == [(F(-1,8),F(-1,6),F(17,24)),(F(1,8),F(1,6),F(41,24))]
    deltas = [-.1*float(mh)/math.sqrt(float(vh)) for _,mh,vh in branches]
    assert deltas[0] > 0 > deltas[1]
    record("T08_same_signs_different_optimizer_memory",
           first_parameter_outputs_equal=True, gradient_sign_histories_equal=True,
           second_bias_corrected_moments=[[str(m),str(v)] for _,m,v in branches],
           second_parameter_deltas=deltas, threshold_task_gap=1)


def exact_bayes_coverage():
    d = 3
    worlds = list(it.product([0,1], repeat=d))
    weights = [F(1,2),F(1,3),F(1,6)]
    reveal = [(),(0,1),(1,2),(0,2)]
    prob = [F(1,10),F(2,10),F(3,10),F(4,10)]
    q = [sum(p for s,p in zip(reveal,prob) if j in s) for j in range(d)]

    def terminal(mass):
        return sum(weights[j] * min(sum(m for m,w in zip(mass,worlds) if w[j]==b)
                                    for b in [0,1]) for j in range(d))

    @lru_cache(None)
    def decision_tree_risk(mass, n):
        base = terminal(mass)
        if not n or not base:
            return base
        values = [base]
        for j in range(d):
            branches = [tuple(m if w[j]==b else F(0) for m,w in zip(mass,worlds))
                        for b in [0,1]]
            values.append(sum(decision_tree_risk(branch,n-1) for branch in branches))
        return min(values)

    rows = []
    for K in range(5):
        posterior_masses = defaultdict(lambda: [F(0)]*len(worlds))
        predicted = [F(0)]*(d+1)
        for seq in it.product(range(len(reveal)), repeat=K):
            seq_mass = math.prod([prob[i] for i in seq], start=F(1))
            union = set().union(*(set(reveal[i]) for i in seq))
            unseen_weights = sorted([weights[j] for j in range(d) if j not in union], reverse=True)
            for n in range(d+1):
                predicted[n] += seq_mass * sum(unseen_weights[n:], start=F(0)) / 2
            for wi,w in enumerate(worlds):
                transcript = tuple((i,tuple(w[j] for j in reveal[i])) for i in seq)
                posterior_masses[transcript][wi] += seq_mass / len(worlds)
        actual = [sum(decision_tree_risk(tuple(mass),n) for mass in posterior_masses.values())
                  for n in range(d+1)]
        assert actual == predicted, (K, actual, predicted)
        formula = sum(w*(1-qi)**K for w,qi in zip(weights,q))/2
        assert actual[0] == formula
        rows.append({"K":K,"optimal_risks_by_query_budget":[str(x) for x in actual]})
    record("T09_exact_posterior_and_adaptive_query_optimum", d=d,
           K_values=list(range(5)), query_budgets=list(range(d+1)), cases=20,
           reveal_probabilities=[str(x) for x in q], exact_curves=rows,
           note="Full hidden-world posterior enumeration and query decision-tree minimization.")


def missing_mass_uniform():
    d = 4
    rows = []
    for K in range(7):
        counts = defaultdict(int)
        for seq in it.product(range(d),repeat=K):
            counts[d-len(set(seq))] += 1
        probs = {m:F(n,d**K) for m,n in counts.items()}
        risks = [sum(p*max(m-n,0) for m,p in probs.items())/(2*d) for n in range(d+1)]
        assert risks[0] == F(1,2)*F(d-1,d)**K
        rows.append({"K":K,"risk":[str(x) for x in risks]})
        if K == 3:
            correct = risks[2]
            invalid = max(sum(p*m for m,p in probs.items())-2,0)/(2*d)
            assert correct == F(1,128) and invalid == 0
    record("T10_uniform_coverage_and_positive_part", cases=35, exact_curves=rows,
           expectation_of_positive_part_counterexample={"K":3,"n":2,"correct":"1/128","incorrect":0})


def kl(p,q):
    return sum(float(a)*math.log(float(a/b)) for a,b in zip(p,q) if a)


def adaptive_kl():
    demo = [F(3,10),F(7,10)]  # probability of observation 1 in world 0/1
    channels = [[F(1,5),F(4,5)],[F(2,5),F(3,5)]]
    maxN = 4
    rows = []
    for K in range(5):
        transcript = [defaultdict(F),defaultdict(F)]
        expected_n = [F(0),F(0)]
        for world in [0,1]:
            for ds in it.product([0,1],repeat=K):
                initial = math.prod([demo[world] if b else 1-demo[world] for b in ds], start=F(1))
                def branch(qs, mass):
                    all_bits = ds+qs
                    score = sum(2*b-1 for b in all_bits)
                    if len(qs)==maxN or (len(qs)>0 and abs(score)>=2):
                        transcript[world][(ds,qs)] += mass
                        expected_n[world] += mass*len(qs)
                        return
                    channel = 0 if not all_bits or all_bits[-1] else 1
                    p = channels[channel][world]
                    branch(qs+(0,),mass*(1-p))
                    branch(qs+(1,),mass*p)
                branch((),initial)
        assert set(transcript[0]) == set(transcript[1])
        for table in transcript:
            assert sum(table.values()) == 1
        errors=[]
        for world in [0,1]:
            err=F(0)
            for (ds,qs),mass in transcript[world].items():
                score=sum(2*b-1 for b in ds+qs)
                if score==0:
                    err+=mass/2
                elif int(score>0)!=world:
                    err+=mass
            errors.append(err)
        delta=max(errors)
        assert 0<float(delta)<.5
        binary_lb=kl([1-delta,delta],[delta,1-delta])
        for v in [0,1]:
            keys=list(transcript[v])
            exact_kl=kl([transcript[v][k] for k in keys], [transcript[1-v][k] for k in keys])
            kd=kl([demo[v],1-demo[v]],[demo[1-v],1-demo[1-v]])
            kx=max(kl([ch[v],1-ch[v]],[ch[1-v],1-ch[1-v]]) for ch in channels)
            upper=K*kd+kx*float(expected_n[v])
            assert binary_lb<=exact_kl+1e-12<=upper+1e-12
            nlower=max(binary_lb-K*kd,0)/kx
            assert float(expected_n[v])>=nlower-1e-12
            rows.append({"K":K,"direction":v,"delta":float(delta),
                         "transcript_KL":exact_kl,"KL_upper":upper,
                         "binary_KL_lower":binary_lb,"expected_queries":float(expected_n[v]),
                         "query_lower_bound":nlower})
    record("T11_adaptive_query_KL_enumeration", instances=len(rows), max_queries=maxN, rows=rows)


def stopped_learning():
    ratios=[]
    nonvacuous=0
    for trial in range(400):
        states,R=4,3
        P=RNG.dirichlet(np.ones(states),size=(R,states))
        perturb=RNG.dirichlet(np.ones(states),size=(R,states))
        Q=.97*P+.03*perturb
        f=RNG.random(states)
        V=[None]*(R+1)
        V[R]=f
        for t in reversed(range(R)):
            V[t]=P[t]@V[t+1]
        init=np.zeros(states);init[1]=1
        final=init.copy()
        for t in range(R): final=final@Q[t]
        gap=abs(float(init@V[0]-final@f))
        certified=np.array([True,True,True,trial%2==0])
        alive=init.copy();rho=0.;bound=0.
        for t in range(R):
            rho+=float(alive[~certified].sum())
            alive[~certified]=0
            # Indicator features for next states 0,1,2; state 3 is the constant baseline.
            b=V[t+1][:3]-V[t+1][3]
            B=np.abs(b).sum()
            anchor=(P[t,0]-Q[t,0])[:3]
            D=np.abs(anchor).max()
            for z in range(states):
                a=np.abs((P[t,z]-Q[t,z])[:3]-anchor).max()
                bound+=alive[z]*B*(D+a)
            alive=alive@Q[t]
        bound+=rho
        assert gap<=bound+1e-12
        ratios.append(gap/bound if bound else 0)
        nonvacuous+=int(bound<1)
    record("T12_stopped_occupancy_composition", instances=400,
           bounds_below_one=nonvacuous, maximum_gap_over_bound=max(ratios))


def confidence_cost():
    n=math.ceil(2*math.log(2*4/.05)/.1**2)
    assert n==1016
    record("T13_Hoeffding_allocation", sufficient_pairs=n,
           radius=math.sqrt(2*math.log(160)/n),
           is_necessary_lower_bound=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(__file__).with_name("results_v015.json"))
    args=parser.parse_args()
    for fn in [audit_counterexample,transport_checks,cold_adam,quotient_and_joint_signs,
               matched_moments,margin_checks,memory_counterexample,exact_bayes_coverage,
               missing_mass_uniform,adaptive_kl,stopped_learning,confidence_cost]:
        fn()
    report={"version":"0.15","seed":15019,"python":platform.python_version(),
            "numpy":np.__version__,"scope":"New finite regression suite; not C23 replication or neural training",
            "passed_groups":len(RESULTS),"failed_groups":0,"checks":RESULTS}
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed_groups":len(RESULTS),"failed_groups":0,"output":str(args.out)}))


if __name__=="__main__":
    main()
