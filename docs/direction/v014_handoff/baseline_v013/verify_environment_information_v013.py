#!/usr/bin/env python3
"""Exact finite checks for environment-information theory v0.13.
Not a real-agent experiment or evidence of novelty.
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
from scipy.linalg import null_space
from scipy.stats import binom

SEED = 2026100913


def bkl(p, q):
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))


def information_decomposition():
    prior = np.array([.3, .4, .3])
    values = np.array([[.9, .3, .6], [.2, .85, .55], [.5, .4, .8]])
    reference_opt = np.array([1., .95, .9])
    # D has two outcomes. The learner intentionally uses a suboptimal rule.
    observation = np.array([[.9, .1], [.2, .8], [.5, .5]])
    output_policy = [2, 0]
    joint = prior[:, None]*observation
    data_mass = joint.sum(axis=0)
    posterior_values = joint.T@values/data_mass[:, None]
    deployed = sum(np.dot(joint[:, d], values[:, output_policy[d]]) for d in range(2))
    oracle = float(prior@values.max(axis=1))
    data_opt = float(data_mass@posterior_values.max(axis=1))
    representation = float(prior@reference_opt-oracle)
    information = oracle-data_opt
    algorithm = data_opt-deployed
    total = float(prior@reference_opt-deployed)
    assert abs(total-representation-information-algorithm)<1e-12
    # Arbitrary randomized postprocessing of D cannot improve its Bayes value.
    kernel = np.array([[.8, .2, 0.], [.1, .3, .6]])
    compressed = joint@kernel
    compressed_value = float(np.max(compressed.T@values, axis=1).sum())
    assert compressed_value<=data_opt+1e-12
    return {"total_regret": total, "representation_term": representation,
            "information_term": information, "algorithm_term": algorithm,
            "identity_residual": abs(total-representation-information-algorithm),
            "data_Bayes_value": data_opt, "postprocessed_Bayes_value": compressed_value}


def perfect_teacher_counterfactual():
    gap = F(1, 10)
    # Teacher sees s in {0,1}; the student has one aliased context.
    plus = [[F(1), F(1, 2)-2*gap], [F(1, 2)+2*gap, F(1)]]
    minus = [[F(1), F(1, 2)+2*gap], [F(1, 2)-2*gap, F(1)]]
    teacher_plus = (plus[0][0]+plus[1][1])/2
    teacher_minus = (minus[0][0]+minus[1][1])/2

    def value(reward, p):
        return sum((p*r[0]+(1-p)*r[1])/2 for r in reward)

    rows = []
    for p in [F(0), F(1, 4), F(1, 2), F(3, 4), F(1)]:
        jp, jm = value(plus, p), value(minus, p)
        assert jp==F(3, 4)-gap+2*gap*p
        assert jm==F(3, 4)+gap-2*gap*p
        rows.append({"p": float(p), "J_plus": float(jp), "J_minus": float(jm)})
    optimum = F(3, 4)+gap
    passive_floor = gap
    assert teacher_plus==teacher_minus==1
    assert value(plus, F(1, 2))==value(minus,F(1, 2))==F(3,4)
    confidence_delta = .05
    d = float(gap)
    branch_kl = bkl(.5+2*d, .5-2*d)
    kl_conf = bkl(1-confidence_delta, confidence_delta)
    lower_exact = kl_conf/branch_kl
    lower_simplified = kl_conf/(64*d*d)
    n_upper = math.ceil(math.log(1/confidence_delta)/(8*d*d))
    # Tie goes to minus. Use exact Binomial probabilities, including ties.
    threshold = n_upper//2
    err_plus = float(binom.cdf(threshold, n_upper, .5+2*d))
    err_minus = float(binom.sf(threshold, n_upper, .5-2*d))
    assert max(err_plus,err_minus)<=confidence_delta+1e-12
    tau, delta_fit = .125, .025
    compile_n = math.ceil(32*math.log(2/delta_fit))
    radius = math.sqrt(math.log(2/delta_fit)/(2*compile_n))
    assert 2*d*(tau+radius)<=d/2+1e-12
    return {"Delta": d, "teacher_value_both_worlds": 1.,
            "demonstration_law": "s uniform; teacher action matches s; observed reward always one",
            "student_class_optimum": float(optimum),
            "representation_floor_against_teacher": float(1-optimum),
            "passive_information_minimax_regret": float(passive_floor),
            "behavior_cloning_value": .75, "policy_values": rows,
            "active_branch_query": "reset to state 1, execute teacher-dispreferred action 0",
            "per_query_KL": branch_kl, "expected_query_lower_bound_exact": lower_exact,
            "expected_query_lower_bound_simplified": lower_simplified,
            "fixed_query_upper_budget": n_upper, "exact_error_plus": err_plus,
            "exact_error_minus": err_minus,
            "synthetic_compile_samples": compile_n, "target_smoothing_tau": tau,
            "branch_failure_budget": confidence_delta, "compile_failure_budget": delta_fit,
            "joint_success_probability_lower_bound": 1-confidence_delta-delta_fit,
            "compile_probability_radius": radius,
            "certified_compiled_regret": 2*d*(tau+radius),
            "scope": "Fixed two-world query interface; known gap; synthetic target labels after identifying sign."}


def value_equivalent_training_separation():
    # Shared reward rule on four states; E and S have different state frequencies.
    reward_a = [F(19,30),F(3,10),F(9,10),F(13,30)]
    reward_b = [F(1,2)]*4
    real_mass = [F(3,4),F(1,4),F(0),F(0)]
    source_mass = [F(0),F(0),F(1,4),F(3,4)]
    def mean(mass, vals): return sum(p*v for p,v in zip(mass,vals))
    aE,aS=mean(real_mass,reward_a),mean(source_mass,reward_a)
    bE,bS=mean(real_mass,reward_b),mean(source_mass,reward_b)
    teacher_actions=[int(a>b) for a,b in zip(reward_a,reward_b)]
    pE=mean(real_mass,teacher_actions); pS=mean(source_mass,teacher_actions)
    teacher_rewards=[max(a,b) for a,b in zip(reward_a,reward_b)]
    teacherE=mean(real_mass,teacher_rewards);teacherS=mean(source_mass,teacher_rewards)
    assert aE==aS==F(11,20) and bE==bS==F(1,2)
    assert teacherE==teacherS==F(3,5)
    finalE=bE+(aE-bE)*pE;finalS=bE+(aE-bE)*pS
    assert finalE-finalS==F(1,40)
    return {"all_student_policy_values": "J_E(p)=J_S(p)=1/2+p/20 for every p in [0,1]",
            "optimal_teacher_value_both": float(teacherE),
            "teacher_A_frequency_real": float(pE), "teacher_A_frequency_source": float(pS),
            "real_trained_student_value": float(finalE),
            "source_trained_student_value": float(finalS),
            "learning_loss": float(finalE-finalS),
            "scope": "Exact stationary student-class values, plus optimal teacher value; not full Bellman equivalence."}


def learning_telescoping(rng):
    maximum = 0.
    examples = []
    for case in range(40):
        T,S=4,5
        E=rng.dirichlet(np.ones(S),size=(T,S))
        W=rng.dirichlet(np.ones(S),size=(T,S))
        terminal=rng.uniform(size=S)
        initial=rng.dirichlet(np.ones(S))
        V=np.zeros((T+1,S));V[T]=terminal
        VW=np.zeros_like(V);VW[T]=terminal
        for t in reversed(range(T)):
            V[t]=E[t]@V[t+1];VW[t]=W[t]@VW[t+1]
        occupancy=initial.copy();terms=[]
        for t in range(T):
            terms.append(float(occupancy@((E[t]-W[t])@V[t+1])))
            occupancy=occupancy@W[t]
        gap=float(initial@(V[0]-VW[0]))
        residual=abs(gap-sum(terms))
        maximum=max(maximum,residual)
        assert residual<1e-12
        if case==0:examples.append({"actual_gap":gap,"signed_terms":terms,
                                     "sum_of_absolute_terms":sum(abs(t) for t in terms)})
    return {"cases":40,"max_identity_residual":maximum,"examples":examples,
            "scope":"Finite learning-state kernels, not a claim that value-to-go is cheaply observable."}


def first_moment_insufficient():
    eta=.6
    # Same marginal Bernoulli labels, but independent versus antithetic batches.
    ind=list(itertools.product([0.,1.],repeat=2))
    ant=[(0.,1.),(1.,0.)]
    def moments(items):
        labels=np.array(items)
        update=(1-eta)*.5+eta*labels.mean(axis=1)
        return float(update.mean()),float(np.mean(update**2)),float(np.var(update))
    mE,jE,vE=moments(ind);mS,jS,vS=moments(ant)
    assert mE==mS==.5 and abs(jE-jS-eta*eta/8)<1e-12
    return {"eta":eta,"expected_updated_parameter_both":.5,
            "final_value_real":jE,"final_value_source":jS,"loss":jE-jS,
            "predicted_covariance_effect":eta*eta/8,"variance_real":vE,"variance_source":vS,
            "scope":"Two teacher action labels per batch; identical one-label marginals and mean squared-loss gradient."}


def minimal_moments():
    phi=np.array([[0.,0.,1.,1.],[0.,1.,0.,1.]])
    constants=np.ones((1,4))
    target=np.array([0.,0.,1.,1.])
    known=np.vstack([constants,phi[1:]])
    invisible=null_space(known)
    h=invisible@(invisible.T@target)
    h=h/np.max(np.abs(h))*.1
    p=np.ones(4)/4+h;q=np.ones(4)/4-h
    assert np.min(p)>0 and np.min(q)>0
    assert np.max(np.abs(known@(p-q)))<1e-12
    missing=float(target@(p-q))
    assert abs(missing)>.1
    both=np.vstack([constants,phi])
    assert np.linalg.matrix_rank(both)-1==2
    tv=float(np.sum(np.abs(p-q))/2)
    all_events=[np.array(mask) for mask in itertools.product([0.,1.],repeat=4)]
    discrepancy=max(float(v@(p-q)) for v in all_events)
    assert abs(tv-discrepancy)<1e-12
    return {"required_linear_expectation_statistics":2,
            "matched_reduced_statistics_residual":float(np.max(np.abs(known@(p-q)))),
            "omitted_continuation_test_discrepancy":missing,
            "full_event_family_discrepancy":discrepancy,"TV":tv,
            "not_claimed":"bit complexity, neural parameter count, or general query sample complexity"}


def finite_interface_certificate(rng):
    T,S,M=3,4,3
    E=rng.dirichlet(np.ones(S)*2,size=(T,S))
    independent=rng.dirichlet(np.ones(S),size=(T,S))
    candidates=[E.copy(),.5*E+.5*independent,.9*E+.1*independent]
    tests=np.array([[0.,.2,.6,1.],[1.,.6,.2,0.]])
    r=len(tests);L=T*S;n=4096;m=4096;delta=.05
    continuation=np.zeros((r,T+1,S));continuation[:,T,:]=tests
    for f in range(r):
        for t in reversed(range(T)):
            continuation[f,t]=E[t]@continuation[f,t+1]
    # These tests are exactly computed in this finite audit. For unknown systems,
    # a separate continuation-estimation certificate is required.
    real_hat=np.zeros((T,S,r))
    model_hat=np.zeros((M,T,S,r))
    true_discrepancy=np.zeros(M)
    for t in range(T):
        Phi=continuation[:,t+1,:]
        for z in range(S):
            real_hat[t,z]=Phi@(rng.multinomial(n,E[t,z])/n)
            for j,W in enumerate(candidates):
                model_hat[j,t,z]=Phi@(rng.multinomial(m,W[t,z])/m)
                true_discrepancy[j]=max(true_discrepancy[j],
                                        float(np.max(np.abs(Phi@(E[t,z]-W[t,z])))))
    estimated=np.max(np.abs(model_hat-real_hat[None,:,:,:]),axis=(1,2,3))
    er=math.sqrt(math.log(4*L*r/delta)/(2*n))
    ew=math.sqrt(math.log(4*M*L*r/delta)/(2*m))
    selected=int(np.argmin(estimated))
    assert true_discrepancy[selected]<=np.min(true_discrepancy)+2*(er+ew)+1e-12
    initial=np.ones(S)/S
    values=[]
    for W in candidates:
        fvals=[]
        for f in range(r):
            val=tests[f].copy()
            for t in reversed(range(T)):val=W[t]@val
            fvals.append(float(initial@val))
        values.append(fvals)
    actual=float(np.max(np.abs(np.array(values[selected])-np.array(values[0]))))
    certificate=T*(estimated[selected]+er+ew)
    assert actual<=certificate+1e-12
    return {"training_rounds":T,"states":S,"candidates":M,"tests":r,
            "real_queries":L*n,"synthetic_queries":M*L*m,"selected_candidate":selected,
            "estimated_local_discrepancies":estimated.tolist(),
            "true_local_discrepancies":true_discrepancy.tolist(),
            "real_radius":er,"synthetic_radius":ew,
            "actual_max_learning_value_error":actual,"certified_error_upper_bound":certificate,
            "scope":"One finite audit with exact continuation tests and an exact reference candidate as positive control."}


def moment_hierarchy():
    rows=[]
    for order in [2,3,4,5]:
        grid=np.arange(order+1)/order
        even=np.array([math.comb(order,i) if i%2==0 else 0
                       for i in range(order+1)],dtype=float)/2**(order-1)
        odd=np.array([math.comb(order,i) if i%2 else 0
                      for i in range(order+1)],dtype=float)/2**(order-1)
        lower_errors=[float((even-odd)@(grid**r)) for r in range(order)]
        delta_m=float((even-odd)@(grid**order))
        formula=(-1)**order*math.factorial(order)/(2**(order-1)*order**order)
        assert max(abs(x) for x in lower_errors)<1e-12
        assert abs(delta_m-formula)<1e-12
        exposures=[]
        for n in [1,2,4,16]:
            Pe=np.array([1.]);Po=np.array([1.])
            for _ in range(n):
                Pe=np.convolve(Pe,even);Po=np.convolve(Po,odd)
            average=np.arange(len(Pe))/(order*n)
            actual=float((Pe-Po)@(average**order))
            predicted=delta_m/n**(order-1)
            assert abs(actual-predicted)<1e-12
            exposures.append({"independent_blocks":n,"actual_learning_gap":actual,
                              "exact_prediction":predicted})
        rows.append({"deployment_horizon":order,"matched_through_order":order-1,
                     "max_lower_moment_residual":max(abs(x) for x in lower_errors),
                     "first_unmatched_moment":delta_m,"exposure_checks":exposures})
    # A one-step deployment and a nonlinear update invalidate a horizon-only rule.
    p=np.array([1/3,2/3]);z=np.array([0.,.75])
    mean=float(p@z);threshold=float(p@(z>.5))
    assert abs(mean-.5)<1e-12 and threshold==2/3
    return {"sharp_orders":rows,
            "nonlinear_update_boundary":{"deployment_horizon":1,"mean_both":.5,
                                         "threshold_update_value_source_1":threshold,
                                         "threshold_update_value_source_2":0.},
            "scope":"IID complete experience blocks; fixed affine averaging; deployment success p^order. Replayed copies do not increase independent block count."}


def joint_moment_affine_check():
    blocks=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    P=np.array([.5,0.,0.,.5])
    opposite=np.array([0.,.5,.5,0.])
    Q=.9*P+.1*opposite
    multi_indices=[(a,b) for a in range(3) for b in range(3) if 1<=a+b<=2]
    errors=[float((P-Q)@np.prod(blocks**np.array(alpha),axis=1))
            for alpha in multi_indices]
    delta=max(abs(x) for x in errors)
    initial=np.array([.2,.8]);weights=np.array([.1,.2,.3,.4])
    # A two-step task: first context 1; on action 1 use context 2,
    # on action 0 use context 1. Reward when the second action is 1.
    def J(p):return p[0]*p[1]+(1-p[0])*p[0]
    def expected(mass):
        total=0.
        for indices in itertools.product(range(4),repeat=3):
            p=weights[0]*initial+sum(weights[r+1]*blocks[i]
                                     for r,i in enumerate(indices))
            total+=np.prod(mass[list(indices)])*J(p)
        return float(total)
    gap=expected(P)-expected(Q)
    predicted=.05*float(weights[1:]@weights[1:])
    coefficient_constant=5.
    assert abs(gap-predicted)<1e-12
    assert abs(gap)<=coefficient_constant*delta+1e-12
    assert np.max(np.abs((P-Q)@blocks))<1e-12
    return {"task":"J(p)=p1*p2+(1-p1)*p1", "blocks":3,
            "fixed_weights":weights.tolist(), "matched_coordinate_means":True,
            "max_moment_error_through_order_2":delta,
            "actual_learning_gap":gap,"exact_covariance_prediction":predicted,
            "C_J":coefficient_constant,"certified_bound":coefficient_constant*delta,
            "scope":"Exact enumeration of all complete block combinations; joint dependence differs."}


def polynomial_update_degree():
    rows=[]
    for rounds in [2,3,4,5]:
        degree=2**(rounds-1)
        grid=[F(i,degree) for i in range(degree+1)]
        signed=[F((-1)**i*math.comb(degree,i),2**(degree-1))
                for i in range(degree+1)]
        assert all(sum(w*z**d for w,z in zip(signed,grid))==0
                   for d in range(degree))
        # First update copies Z, all later updates square p; deploy once.
        values=[]
        for z in grid:
            p=z
            for _ in range(rounds-1):p=p*p
            values.append(p)
        gap=sum(w*v for w,v in zip(signed,values))
        formula=F((-1)**degree*math.factorial(degree),
                  2**(degree-1)*degree**degree)
        assert gap==formula and gap!=0
        rows.append({"deployment_decisions":1,"updates":rounds,
                     "necessary_moment_order_for_this_pair":degree,
                     "matched_through_order":degree-1,"exact_gap_fraction":str(gap),
                     "gap":float(gap)})
    return {"construction":"p1=Z; p_i=p_(i-1)^2; terminal J(p)=p",
            "cases":rows,"scope":"A permitted polynomial learner family, not a statement that neural optimizers typically require exponential order."}


def plot_checks(results, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "svg.fonttype":"none"})
    fig, axes=plt.subplots(1,2,figsize=(11.5,4.8))
    colors=["#176B87","#A3481D","#5B4B9A","#28774C"]
    for row,color in zip(results["interaction_moment_hierarchy"]["sharp_orders"],colors):
        N=row["deployment_horizon"]
        points=row["exposure_checks"]
        ns=np.array([p["independent_blocks"] for p in points])
        ys=np.array([abs(p["actual_learning_gap"]) for p in points])
        curve=np.geomspace(1,16,100)
        axes[0].loglog(curve,abs(row["first_unmatched_moment"])/curve**(N-1),
                       color=color,linewidth=1.6)
        axes[0].scatter(ns,ys,color=color,s=27,zorder=3,
                        label=rf"$N={N}$: $n^{{-{N-1}}}$")
    axes[0].set(xlabel="Independent experience blocks n",
                ylabel="Absolute learning-value difference",
                title="Affine updates: exact decay law")
    axes[0].set_xticks([1,2,4,8,16],labels=["1","2","4","8","16"])
    axes[0].grid(alpha=.18,which="major")
    axes[0].legend(frameon=False,fontsize=9)

    rows=results["polynomial_update_information_degree"]["cases"]
    updates=np.array([r["updates"] for r in rows])
    degree=np.array([r["necessary_moment_order_for_this_pair"] for r in rows])
    axes[1].plot(updates,degree,marker="o",color="#A3481D",linewidth=2,
                 label=r"Copy Z, then square: $D=2^{R-1}$")
    axes[1].plot(updates,np.ones(len(updates)),marker="s",color="#176B87",
                 linewidth=1.6,label=r"Affine update class: $D=N=1$")
    axes[1].set(xlabel="Number of learning updates R",
                ylabel="Sufficient order; sharp within each class",
                title="One-step deployment: update rule matters",
                xticks=updates,yticks=[1,2,4,8,16])
    axes[1].grid(alpha=.18)
    axes[1].legend(frameon=False,fontsize=8.5,loc="upper left")
    fig.subplots_adjust(bottom=.22,wspace=.28,top=.87)
    fig.text(.5,.055,
             "Finite constructions. Dots: exact enumeration / rational checks. "
             "These are not trained-agent results.",
             ha="center",fontsize=9,color="#555555")
    directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for ext in ["png","svg"]:
        path=directory/f"information_order_checks_v013.{ext}"
        fig.savefig(path,dpi=190,facecolor="white")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return {"files":paths,"matplotlib":matplotlib.__version__,
            "caption":"Exact finite constructions; no empirical neural-agent claim."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("results_v013.json"))
    parser.add_argument("--plot",action="store_true",help="Also save finite-construction PNG/SVG plots.")
    args=parser.parse_args()
    seeds=np.random.SeedSequence(SEED).spawn(2)
    results={"version":"0.13","seed":SEED,
             "runtime":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
             "scope":"Exact finite constructions and limited numerical audits; no LLM, WM post-training or real-agent experiment.",
             "regret_decomposition":information_decomposition(),
             "perfect_teacher_information_gap":perfect_teacher_counterfactual(),
             "evaluation_learning_separation":value_equivalent_training_separation(),
             "learning_process_identity":learning_telescoping(np.random.default_rng(seeds[0])),
             "mean_update_counterexample":first_moment_insufficient(),
             "minimal_linear_information":minimal_moments(),
             "finite_calibration_certificate":finite_interface_certificate(np.random.default_rng(seeds[1])),
             "interaction_moment_hierarchy":moment_hierarchy(),
             "joint_moment_affine_check":joint_moment_affine_check(),
             "polynomial_update_information_degree":polynomial_update_degree()}
    if args.plot:
        results["plot_artifacts"]=plot_checks(results,args.output.parent)
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

