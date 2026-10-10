"""Controlled optimizer experiments. All neural outcomes are empirical diagnostics."""
from __future__ import annotations

import ast
import inspect
import time
from scipy.stats import spearmanr
from common import *
import common as C
from mathematics import exact_history, union_recursion


def weighted_result(probabilities, weights, task, N, mode):
    values=task_value(probabilities,task,N)
    return dict(mean=float(weights@values),
        SE=float(np.std(values,ddof=1)/math.sqrt(len(values))) if 'replicas' in mode else 0.,
        values=values,probabilities=probabilities,mode=mode,n_histories=len(weights))


def check_E0(quick=False):
    g=Group('E0','C23 reproduction and raw artifact bundle',
            'Imported original C23 functions; enumerated histories and independently logged recurrence', 'controlled_neural')
    laws=[]
    allcases=[('Prop5',1,[(F(0),),(F(1),)],[F(1,4),F(3,4)],[F(3,4),F(1,4)]),
        ('Prop6',2,[(F(0),),(F(1,2),),(F(1),)],[F(1,4),F(1,2),F(1,4)],[F(0),F(1),F(0)])]+cases()
    for task,N,points,P,Q in allcases:
        k=len(points[0]); moments=[]
        for alpha in it.product(range(N+1),repeat=k):
            if not 1<=sum(alpha)<=N: continue
            values=[math.prod(v**a for v,a in zip(point,alpha)) for point in points]
            mp=sum(p*v for p,v in zip(P,values));mq=sum(q*v for q,v in zip(Q,values))
            moments.append(dict(alpha=alpha,P=mp,Q=mq,match=mp==mq))
        laws.append(dict(task=task,N=N,support=points,P=P,Q=Q,label_moments=moments))
    rows=[]
    points,P,Q=law_scalar(3)
    for eta in (.01,.1):
        for R in (1,4):
            sources=[]
            for law in (P,Q):
                labels,weights,mode=training_sequences(g.rng,points,law,R,2000,False)
                original,affine,_=train_sequence(labels,eta,'Adam')
                measured=run_network(labels,eta=eta,trace=True)
                discrepancy=float(np.max(abs(original-measured['probabilities'])))
                g.le(discrepancy,1e-12,'C23 imported vs new trace final probabilities',dict(eta=eta,R=R,labels=labels))
                sources.append(dict(law=law,histories=labels,history_probabilities=weights,
                    raw_steps=measured['trace'],final_probabilities=original,
                    task_returns=neural_return(original,'moment',3),
                    affine_probabilities=affine,affine_returns=neural_return(affine,'moment',3),
                    expected_return=float(weights@neural_return(original,'moment',3)),
                    affine_return=float(weights@neural_return(affine,'moment',3)),SE=0.,mode=mode))
                g.count(len(weights))
            gap=sources[0]['expected_return']-sources[1]['expected_return']
            affine_gap=sources[0]['affine_return']-sources[1]['affine_return']
            rows.append(dict(eta=eta,R=R,gap=gap,affine_SGD_comparator_gap=affine_gap,sources=sources))
    reported=[r for r in rows if r['eta']==.1]
    for row,target in zip(reported,(.0769,.1196)):
        g.le(abs(abs(row['gap'])-target),5e-4,'reported approximate C23 gap reproduction',dict(row=row,target=target))
    old_path=ROOT/'docs/direction/v014_handoff/Environment_Information_Theory_EN.md'
    old=old_path.read_text();wording=old.split('### Theorem 19:')[1].split('### 10.1')[0]
    old_hash=digest(old_path)
    g.check(old_hash=='9271ae051b6a31fc15907a0469354e46a8d043493f9cf353daca2809c58ac1e6',
            'exact source version of failed theorem',dict(hash=old_hash))
    source=inspect.getsource(c23)
    tree=ast.parse(source)
    definitions={node.name:ast.get_source_segment(source,node) for node in tree.body
                 if isinstance(node,ast.FunctionDef) and node.name.startswith('check_C')}
    # Historical results from the independent suite are provenance, never an oracle
    # for these new checks. The package verifier and package result files are unread.
    historical_path=ROOT/'theory/independent_checks/results_independent_v014.json'
    historical=json.loads(historical_path.read_text()) if historical_path.exists() else None
    manifest=json.loads((ROOT/'docs/direction/v014_handoff/package_manifest_v014.json').read_text())
    bundle=dict(label='controlled neural, not Gemma',laws=laws,
        scalar_matching='law_scalar(N): orders 1..N-1 match; order N does not',
        architecture='one-hot context -> four tanh units with shared bias -> sigmoid',
        init=neural_init(1,1),dtype='float64',loss='binary cross entropy',reduction='mean over contexts',
        optimizer=dict(name='Adam',beta=[.9,.999],epsilon=1e-8,epsilon_placement='outside sqrt',
            bias_correction=True,decay=0,clipping=None,AMSGrad=False,fused=False,gradient_scaling=False,
            parameter_groups='one common setting',code=inspect.getsource(neural_step)),
        estimand='E_{P^R}[p_R^3] - E_{Q^R}[p_R^3]; deterministic expected deployment return per trained replica',
        affine_estimand='same contrast for p <- p + eta A (label-p), A=tangent_matrix(1)',
        repetition='complete finite-history enumeration; SE=0; no Monte Carlo for these four settings',
        seed=g.seed,raw_runs=rows,
        regroup=dict(packets_A=[[0,0],[1,1]],packets_B=[[0,1],[0,1]],
            labels_A=[0,1],labels_B=[F(1,2),F(1,2)],updates_per_cycle=2),
        original_19_file_manifest=manifest,
        manifest_status='Preserved as supplied metadata; package verification code/results are deliberately not read or rerun.',
        independent_23_check_definitions=definitions,
        historical_independent_results=historical,
        historical_independent_results_sha256=digest(historical_path) if historical else None,
        original_code_commit='7b75c7da11a1fb0811c7c33b3011397cbe83069b',
        source_sha256=digest(ROOT/'theory/independent_checks/check_v014_independent.py'),
        C15=dict(audit_state=0,train_state=1,audit_P=[1,0],audit_Q=[1,0],
            train_P=[1,0],train_Q=[0,1],feature=[0,1],gap=1,
            old_bound=2*math.sqrt(math.log(80)/20000),expected_failure=True),
        exact_old_Theorem19='### Theorem 19:'+wording,old_source_sha256=old_hash)
    path=write_json('c23_bundle.json',bundle)
    g.row['data']=dict(bundle=path,gaps=[{k:r[k] for k in ('eta','R','gap','affine_SGD_comparator_gap')} for r in rows],
        original_source_hash=old_hash,definitions_count=len(definitions))
    figure('E0','C23 reproduced return contrasts',lambda fig,ax:(
        ax.bar([f"eta={r['eta']} R={r['R']}" for r in rows],[r['gap'] for r in rows]),
        ax.set(ylabel='E_P return minus E_Q return')))
    g.row['notes']=['C23 functions reused without modification. The historical 23-check result is provenance only.',
        'The 19-file manifest is included as supplied; package verifier/results were not opened to recompute its hashes.']
    return g.finish()


def baseline_history_audit(g,labels,eta):
    n,R,k=labels.shape;params=neural_init(k,n)
    state=([np.zeros_like(p) for p in params],[np.zeros_like(p) for p in params])
    m=np.zeros_like(flatten(params));v=m.copy(); raw=[]
    for t in range(R):
        theta=flatten(params);gradient=flatten(neural_gradient(params,labels[:,t]))
        predicted,m,v,record=optimizer(theta,gradient,m,v,t+1,eta=eta)
        actual,state=neural_step(params,labels[:,t],eta,'Adam',state,t+1)
        actual_update=flatten(actual)-theta
        expected_update=record['update']
        scale=np.linalg.norm(expected_update,axis=1)
        error=np.linalg.norm(actual_update-expected_update,axis=1)
        relative=np.divide(error,scale,out=np.zeros_like(error),where=scale>0)
        g.check(bool(np.all(error[scale==0]==0)),'zero predicted updates execute exactly zero',dict(labels=labels,step=t))
        g.le(float(relative.max()),1e-12,'predeclared 1e-12 relative actual Adam map audit',
            dict(step=t+1,labels=labels,gradient=gradient,predicted=expected_update,actual=actual_update))
        raw.append(dict(record,theta_before=theta,actual_update=actual_update,
                        actual_theta=flatten(actual),relative_error=relative))
        params=actual
    return raw


def check_E1(quick=False):
    g=Group('E1','Actual finite-epsilon Adam map and one-factor ablations',
            'Per-support and history enumeration against imported neural_step; separate sign limit', 'controlled_neural')
    records=[];ablations=[];max_relative_error=0.
    variants=[('baseline',{}),('inside_sqrt',dict(inside=True))]
    variants += [(f'epsilon_{eps}',dict(eps=eps)) for eps in (1e-6,1e-4,1e-2,1e-1)]
    variants += [('no_bias_correction',dict(correction=False)),('warm_5',dict(warm=5)),
        ('global_clip_0.1',dict(clip=.1)),('sum_reduction',dict(reduction='sum')),
        ('shift_support',dict(shift=True))]
    for task,N,points,P,Q in cases():
        pointsf=np.array(points,float);k=len(pointsf[0]);theta=flatten(neural_init(k,1))
        sources=[]
        for name,law in zip(('P','Q'),(P,Q)):
            support=np.flatnonzero(np.array(law,float));labels=pointsf[support,None,:]
            weights=np.array(law,float)[support]
            gradient=flatten(neural_gradient(neural_init(k,len(support)),labels[:,0]))
            ideal=-.1*np.sign(gradient)
            finite=-.1*gradient/(abs(gradient)+1e-8)
            raw=baseline_history_audit(g,labels,.1)
            max_relative_error=max(max_relative_error,max(float(r['relative_error'].max()) for r in raw))
            measured=run_network(labels,trace=True)
            sign_prob=neural_forward(unflatten(theta+ideal,k))[0]
            inputmom=np.array([weights@(labels[:,0]**power) for power in range(1,N+1)])
            gmom=np.array([weights@(gradient**power) for power in range(1,N+1)])
            sources.append(dict(source=name,support_indices=support,probabilities=weights,
                labels=labels[:,0],gradients=gradient,input_moments=inputmom,gradient_moments=gmom,
                gradient_second_joint=np.einsum('n,ni,nj->ij',weights,gradient,gradient),
                P_g_zero=weights@(gradient==0),P_abs_g_le_epsilon=weights@(abs(gradient)<=1e-8),
                finite_epsilon_updates=finite,expected_update=weights@finite,
                sign_limit_updates=ideal,expected_sign_update=weights@ideal,
                final_probabilities=measured['probabilities'],
                expected_return=float(weights@task_value(measured['probabilities'],task,N)),
                sign_limit_return=float(weights@task_value(sign_prob,task,N)),raw=raw))
            seq,w,mode=training_sequences(g.rng,points,law,4,2000,False)
            history_raw=baseline_history_audit(g,seq,.1)
            max_relative_error=max(max_relative_error,max(float(r['relative_error'].max()) for r in history_raw))
            history_summary=[dict(step=t+1,expected_gradient=w@r['gradient'],
                gradient_moments=np.array([w@(r['gradient']**power) for power in range(1,N+1)]),
                P_g_zero=w@(r['gradient']==0),P_abs_g_le_epsilon=w@(abs(r['gradient'])<=1e-8),
                expected_executed_update=w@r['actual_update'],expected_predicted_update=w@r['update'])
                for t,r in enumerate(history_raw)]
            artifact=write_json(f'raw/E1_{task}{N}_{name}_histories.json',
                dict(histories=seq,weights=w,mode=mode,raw=history_raw,per_step_expectations=history_summary))
            sources[-1]['four_step_history_artifact']=artifact
            g.count(len(support)+len(w))
        transitions=dict(input_moment_differences=sources[0]['input_moments']-sources[1]['input_moments'],
            gradient_moment_differences=sources[0]['gradient_moments']-sources[1]['gradient_moments'],
            expected_update_difference=sources[0]['expected_update']-sources[1]['expected_update'],
            task_return_difference=sources[0]['expected_return']-sources[1]['expected_return'])
        joint_moments=[]
        for order in range(1,N+1):
            worst_joint=0.;witness=None
            for coordinates in it.combinations_with_replacement(range(sources[0]['gradients'].shape[1]),order):
                expectations=[float(s['probabilities']@np.prod(s['gradients'][:,coordinates],axis=1)) for s in sources]
                delta=expectations[0]-expectations[1]
                if abs(delta)>worst_joint:worst_joint=abs(delta);witness=dict(coordinates=coordinates,expectations=expectations)
            joint_moments.append(dict(order=order,max_absolute_joint_moment_difference=worst_joint,witness=witness))
        transitions['all_joint_gradient_moments_through_N']=joint_moments
        records.append(dict(task=task,N=N,sources=sources,transitions=transitions))
        baseline={}
        for variant,options in variants:
            options=options.copy();shift=options.pop('shift',False);warm=options.pop('warm',0)
            used=pointsf*.1+.05 if shift else pointsf
            start=snapshot(k,warm)
            shift_moments=[]
            for alpha in it.product(range(N+1),repeat=k):
                if not 1<=sum(alpha)<N:continue
                original=[math.prod(pj**a for pj,a in zip(point,alpha)) for point in points]
                shifted=[math.prod((F(1,20)+pj/F(10))**a for pj,a in zip(point,alpha)) for point in points]
                before=sum((a-b)*v for a,b,v in zip(P,Q,original))
                after=sum((a-b)*v for a,b,v in zip(P,Q,shifted))
                g.eq(before,0,'declared lower input moments match',dict(task=task,N=N,alpha=alpha))
                g.eq(after,0,'shift retains lower moment equalities',dict(task=task,N=N,alpha=alpha))
                shift_moments.append(dict(alpha=alpha,before=before,after=after))
            if shift:
                gradients=flatten(neural_gradient(neural_init(k,len(used)),used))
                g.check(bool(np.all(np.sign(gradients)==np.sign(gradients[0]))),
                        'shifted support has a common coordinatewise first-gradient sign',dict(gradients=gradients))
            for R in (1,4):
                means=[];raw_sources=[]
                for law in (P,Q):
                    labels,weights,mode=training_sequences(g.rng,used,law,R,2000,False)
                    result=run_network(labels,eta=.1,start=start,trace=(R==1),**options)
                    means.append(float(weights@task_value(result['probabilities'],task,N)))
                    if R==1:raw_sources.append(dict(weights=weights,steps=result['trace']))
                    g.check(bool(np.all(np.isfinite(result['probabilities']))),'finite ablation predictions',dict(variant=variant,task=task,N=N,R=R))
                gap=means[0]-means[1]
                if variant=='baseline':baseline[R]=gap
                ablations.append(dict(task=task,N=N,variant=variant,R=R,gap=gap,
                    change_from_baseline=gap-baseline[R],options=options,
                    start=start,shift='y -> 0.05+0.1*y' if shift else None,
                    preserved_moments=shift_moments if shift else [],per_support=raw_sources));g.count()
    g.row['data']=dict(tolerance=dict(relative=1e-12,dtype='float64',zero_rule='exactly zero'),
        maximum_measured_relative_error=max_relative_error,
        transitions=records,ablations=ablations,
        warm_definition='five actual Adam updates, eta=.03, deterministic third law labels=1/4, full global step retained')
    figure('E1','One-factor Adam ablations, moment N=3, four updates',lambda fig,ax:(
        ax.barh([r['variant'] for r in ablations if r['task']=='moment' and r['N']==3 and r['R']==4],
                [r['gap'] for r in ablations if r['task']=='moment' and r['N']==3 and r['R']==4]),
        ax.set(xlabel='Final expected return contrast')))
    return g.finish()


def check_E2(quick=False):
    g=Group('E2','Optimizer-matched frozen-Jacobian comparison',
            'Five branches from one snapshot; identical within-source histories across branches', 'controlled_neural')
    rows=[];persist=[];derivative_checks=[]
    for k,warm in it.product((1,2,3),(0,3)):
        state=snapshot(k,warm);theta=state['theta'];J=logit_jacobian(unflatten(theta,k))
        maximum=0.
        for coordinate in range(theta.shape[1]):
            plus=theta.copy();minus=theta.copy();plus[0,coordinate]+=1e-6;minus[0,coordinate]-=1e-6
            finite_difference=(neural_forward(unflatten(plus,k))[2]-neural_forward(unflatten(minus,k))[2])/(2e-6)
            maximum=max(maximum,float(np.max(abs(finite_difference-J[:,:,coordinate]))))
        g.le(maximum,1e-8,'analytic frozen logit Jacobian vs central differences',dict(k=k,warm=warm,theta=theta))
        derivative_checks.append(dict(k=k,warm=warm,max_residual=maximum))
    configurations=list(it.product((.01,.1,1.),(1,4,40)))
    if quick: configurations=[(.01,1),(.1,4),(1.,40)]
    for task,N,points,P,Q in cases():
        for eta,R in configurations:
            sources=[]
            for source,law in zip(('P','Q'),(P,Q)):
                labels,weights,mode=training_sequences(g.rng,points,law,R,240 if quick else 2000,quick)
                data=dict(labels=labels,weights=weights);branches={}
                for branch,opt,name in [('actual','Adam','actual_Adam'),('frozen','Adam','frozen_Adam'),
                    ('actual','SGD','actual_SGD'),('frozen','SGD','frozen_SGD')]:
                    result=run_network(labels,eta=eta,opt=opt,branch=branch)
                    summary=weighted_result(result['probabilities'],weights,task,N,mode)
                    g.check(bool(np.all(np.isfinite(summary['values']))) and
                        bool(np.all((summary['values']>=0)&(summary['values']<=1))),
                        'comparator task returns bounded',dict(task=task,N=N,eta=eta,R=R,branch=name))
                    branches[name]={key:summary[key] for key in ('mean','SE','mode','n_histories')}
                    branches[name].update(max_jacobian_drift=float(result['jacobian_drift'].max()),
                        max_logit_residual=float(result['logit_residual'].max()),
                        mean_jacobian_drift=weights@result['jacobian_drift'],
                        mean_logit_residual=weights@result['logit_residual'],
                        effective_frozen_jacobian_drift=0. if branch=='frozen' else None)
                    data[name+'_probabilities']=summary['probabilities'];data[name+'_return']=summary['values']
                    data[name+'_jacobian_drift']=result['jacobian_drift'];data[name+'_logit_residual']=result['logit_residual']
                    data[name+'_final_m']=result['m'];data[name+'_final_v']=result['v']
                    if name=='actual_SGD':
                        affine=weighted_result(result['affine'],weights,task,N,mode)
                        branches['tangent_affine_SGD']={key:affine[key] for key in ('mean','SE','mode','n_histories')}
                        data['tangent_affine_SGD_return']=affine['values']
                path=C.OUT/'raw'/f'E2_{task}{N}_eta{eta}_R{R}_{source}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,**data)
                sources.append(dict(source=source,branches=branches,raw=str(path.relative_to(HERE)),
                    raw_sha256=digest(path),history_sha256=hashlib.sha256(labels.tobytes()).hexdigest()))
                g.count(len(weights))
            gaps={key:sources[0]['branches'][key]['mean']-sources[1]['branches'][key]['mean'] for key in sources[0]['branches']}
            ses={key:math.hypot(sources[0]['branches'][key]['SE'],sources[1]['branches'][key]['SE']) for key in gaps}
            persistent=abs(gaps['frozen_Adam'])>max(1e-10,3*ses['frozen_Adam'])
            rows.append(dict(task=task,N=N,eta=eta,R=R,gaps=gaps,SE=ses,sources=sources,
                frozen_Adam_gap_persists=persistent,initial_snapshot=snapshot(len(points[0])),
                note='Counterfactual actual-network Jacobian is measured along frozen-model parameters; its executed Jacobian is fixed.'))
            if persistent:persist.append(dict(task=task,N=N,eta=eta,R=R,gap=gaps['frozen_Adam'],SE=ses['frozen_Adam']))
        print(f'E2 completed {task} N={N}',flush=True)
    g.row['data']=dict(comparisons=rows,persistent_frozen_gaps=persist,derivative_checks=derivative_checks,
        frozen_definition='logit(theta)=logit(theta0)+J0(theta-theta0); sigmoid and mean CE retained; each branch owns moments',
        tangent_scope='Existing initial-tangent affine-SGD construction; an empirical comparator, not an Adam or convex-affine certificate')
    def draw(fig,ax):
        subset=[r for r in rows if r['task']=='moment' and r['N']==3]
        for key in ('actual_Adam','frozen_Adam','actual_SGD','frozen_SGD','tangent_affine_SGD'):
            ax.plot(range(len(subset)),[r['gaps'][key] for r in subset],marker='o',label=key)
        ax.set_xticks(range(len(subset)),[f"{r['eta']}/{r['R']}" for r in subset],rotation=35)
        ax.set(xlabel='learning rate / updates',ylabel='Expected return contrast, moment N=3');ax.legend(fontsize=8)
    figure('E2','Five branches with matched histories',draw)
    g.info('Persistence under the frozen Jacobian is evidence that evolving features are unnecessary for those particular gaps.')
    return g.finish()


def check_E3(quick=False):
    g=Group('E3','Same-sign history regression in the executed scalar and neural optimizers',
            'Shared E1/E2 optimizer; rational scalar oracle; neural soft-label search', 'controlled_neural')
    scalar=[]
    for epsilon in (0.,1e-8):
        branches=[]
        for first in (1.,2.):
            theta=np.array([[0.]]);m=theta.copy();v=theta.copy();trace=[]
            for step,gradient in enumerate((first,-.75),1):
                theta,m,v,r=optimizer(theta,np.array([[gradient]]),m,v,step,eta=1.,eps=epsilon,beta=(.5,.5))
                trace.append(r)
            exact=exact_history(F(int(first)),F(str(epsilon)))
            g.eq(float(trace[0]['update'][0,0]),float(exact['theta1']),'executed exact first scalar step',exact,tol=1e-14)
            g.eq(float(m[0,0]),float(exact['m2']),'executed scalar m2',exact,tol=1e-14)
            g.eq(float(trace[1]['v_hat'][0,0]),float(exact['vhat2']),'executed scalar vhat2',exact,tol=1e-14)
            branches.append(dict(trace=trace,exact=exact))
        g.check(branches[0]['trace'][1]['update'][0,0]>0>branches[1]['trace'][1]['update'][0,0],
            'executed scalar opposite second directions',branches)
        scalar.append(dict(epsilon=epsilon,branches=branches));g.count(2)
    grid=dict(first_labels=[0.,.1,.2,.3,.4],second_labels=np.linspace(.55,1.,19),eta=[.01,.1],
              epsilon=[0.,1e-8],beta=[.5,.5],context=0)
    witnesses=[];searched=0
    for eps in grid['epsilon']:
        found=None
        for eta,a,b,y2 in it.product(grid['eta'],grid['first_labels'],grid['first_labels'],grid['second_labels']):
            if a>=b:continue
            labels=np.array([[[a],[y2]],[[b],[y2]]])
            r=run_network(labels,eta=eta,eps=eps,beta=(.5,.5),trace=True)
            first=r['trace'][0];second=r['trace'][1];searched+=1
            if (np.array_equal(np.sign(first['gradient'][0]),np.sign(first['gradient'][1])) and
                np.array_equal(np.sign(second['gradient'][0]),np.sign(second['gradient'][1])) and
                second['update'][0,-1]*second['update'][1,-1]<0):
                found=dict(epsilon=eps,eta=eta,labels=labels,trace=r['trace'],
                    separating_coordinate='output bias',probabilities=r['probabilities'])
                break
        if found is not None:
            g.check(found['trace'][1]['update'][0,-1]*found['trace'][1]['update'][1,-1]<0,
                'actual network realizes same-sign histories with opposite second updates',found)
        witnesses.append(found)
    g.count(searched);g.row['data']=dict(scalar=scalar,search_grid=grid,n_searched=searched,neural_witnesses=witnesses)
    if any(w is None for w in witnesses):g.info('No neural witness on one declared epsilon search grid; scalar regression remains valid.')
    figure('E3','Optimizer history: equal gradient signs, different updates',lambda fig,ax:(
        ax.bar(['first=1','first=2'],[scalar[1]['branches'][i]['trace'][1]['update'][0,0] for i in (0,1)]),
        ax.set(ylabel='Second scalar parameter update, epsilon=1e-8')))
    return g.finish()


def state_after_labels(labels,eta=.1):
    r=run_network(np.array(labels,float).reshape(1,-1,1),eta=eta)
    return dict(theta=r['theta'],m=r['m'],v=r['v'],step=len(labels),k=1,pretraining_labels=labels)


def response_panel(state,points,reference,R=6,eta=.1):
    n=len(points);support=np.flatnonzero(np.array(reference,float))
    histories=list(it.product(support,repeat=R-1))
    suffix_weights=np.array([math.prod(float(reference[i]) for i in h) for h in histories])
    labels=np.array([[point]+[points[i] for i in h] for point in points for h in histories],float)
    result=run_network(labels,eta=eta,start=state)
    response=(task_value(result['probabilities'],'moment',3).reshape(n,-1)@suffix_weights)
    first=run_network(np.array(points,float)[:,None,:],eta=eta,start=state)
    features=np.column_stack([np.array(points,float)[:,0],first['probabilities'][:,0]])
    return response,features


def check_E4(quick=False):
    g=Group('E4','Fresh-state coverage diagnostics and anchored response predictions',
            'Actual reference trajectories; exact finite-support/full-tail enumeration at observed states', 'controlled_neural')
    points,P,Q=law_scalar(3);p=np.array(P,float);q=np.array(Q,float)
    reference_history=np.array(points,float)[g.rng.choice(len(points),size=5,p=p),0].tolist()
    anchors=[snapshot(1)]+[state_after_labels(reference_history[:n]) for n in (2,5)]
    def candidate(state):
        prob=float(neural_forward(unflatten(state['theta'],1))[0][0,0])
        mixing=.1+.6*prob
        return (1-mixing)*q+mixing*np.array([0.,0.,0.,1.])
    panels=[]
    for state in anchors:
        response,phi=response_panel(state,points,P)
        coeff=np.linalg.lstsq(np.c_[np.ones(len(phi)),phi],response,rcond=None)[0]
        panels.append(dict(state=state,response=response,phi=phi,coeff=coeff,Q=candidate(state),
            discrepancy=(p-candidate(state))@phi,D=float(np.max(abs((p-candidate(state))@phi)))))
    rows=[]
    for trial in range(6 if quick else 40):
        length=int(g.rng.integers(1,14))
        labels=np.array(points,float)[g.rng.choice(len(points),size=length,p=p),0].tolist()
        state=state_after_labels(labels)
        index=int(np.argmin([np.linalg.norm(state['theta']-a['theta']) for a in anchors]));anchor=panels[index]
        response,phiz=response_panel(state,points,P);qz=candidate(state)
        a=float(np.max(abs((p-qz)@anchor['phi']-anchor['discrepancy'])))
        alpha=float(np.max(abs(phiz-anchor['phi'])))
        coeff=anchor['coeff'];fitted=coeff[0]+phiz@coeff[1:]
        residual=response-fitted;tau=float(max(abs(residual)));B=float(sum(abs(coeff[1:])))
        gold=float((p-qz)@response)
        anchored=float((p-anchor['Q'])@(anchor['coeff'][0]+anchor['phi']@anchor['coeff'][1:]))
        current_laws_frozen_response=float((p-qz)@(anchor['coeff'][0]+anchor['phi']@anchor['coeff'][1:]))
        diagnostic_bound=B*(anchor['D']+a+2*alpha)+2*tau
        g.le(abs(gold),diagnostic_bound,'finite observed message-panel diagnostic bound',
            dict(state=state,response=response,phi=phiz,anchor=anchor,Q=qz),tol=1e-8)
        rows.append(dict(trial=trial,history=labels,state=state,anchor=index,D_x=anchor['D'],a=a,alpha=alpha,
            response=response,phi_z=phiz,coefficients=coeff,residual=residual,tau=tau,B=B,
            gold_contrast=gold,anchor_prediction=anchored,current_laws_prediction=current_laws_frozen_response,
            anchor_error=anchored-gold,diagnostic_bound=diagnostic_bound,
            prediction_failure=abs(anchored-gold)>1e-3));g.count()
    g.row['data']=dict(anchors=panels,observations=rows,prediction_tolerance=1e-3,
        failures=sum(r['prediction_failure'] for r in rows),
        absolute_error_quantiles=np.quantile([abs(r['anchor_error']) for r in rows],[0,.5,.9,1]),
        response_residual_quantiles=np.quantile(np.concatenate([abs(r['residual']) for r in rows]),[0,.5,.9,1]),
        provider_rule='Q_z=(1-lambda)Q3+lambda delta_1; lambda=.1+.6*p(theta_z)',
        message_domain='same four scalar labels at every state; full tail=6 actual updates')
    figure('E4','Observed-state coverage prediction errors',lambda fig,ax:(
        ax.scatter([r['a'] for r in rows],[abs(r['anchor_error']) for r in rows]),
        ax.set(xlabel='Measured discrepancy transport a',ylabel='Absolute anchored prediction error')))
    g.info('Empirical diagnostics on reached states, not a uniform coverage or residual certificate. Prediction failures are retained.')
    return g.finish()


def singleton_union_integer(q_integer,K):
    """Integer subset Möbius inversion; denominator is sum(q)^K."""
    m=len(q_integer);size=1<<m
    masses=np.zeros(size,dtype=np.int64)
    for mask in range(1,size):
        bit=mask&-mask;masses[mask]=masses[mask^bit]+q_integer[bit.bit_length()-1]
    numerators=np.array([int(v)**K for v in masses],dtype=object)
    for bit in range(m):
        stride=1<<bit
        for start in range(0,size,2*stride):
            numerators[start+stride:start+2*stride]-=numerators[start:start+stride]
    return numerators,int(sum(q_integer))**K


def check_E5(quick=False):
    g=Group('E5','Hidden-bit K/query curves and achieved tabular learner risk',
            'Exact integer occupancy inversion, rational risks, actual Adam/SGD logit updates', 'mixed')
    curves=[];learners=[];queries=(0,1,2,4);full_table_checks=[]
    m_values=(4,8) if quick else (4,8,16)
    for m in m_values:
        for qkind,wkind in it.product(('uniform','nonuniform'),repeat=2):
            qi=[1]*m if qkind=='uniform' else list(range(1,m+1))
            wi=[1]*m if wkind=='uniform' else list(range(m,0,-1))
            q=[F(i,sum(qi)) for i in qi];w=[F(i,sum(wi)) for i in wi]
            remaining=np.empty((1<<m,len(queries)),dtype=object)
            for mask in range(1<<m):
                unseen=sorted([wi[j] for j in range(m) if not mask>>j&1],reverse=True)
                for col,n in enumerate(queries):remaining[mask,col]=sum(unseen[n:])
            for K in range(2*m+1):
                numer,den=singleton_union_integer(qi,K)
                g.eq(sum(numer),den,'integer occupancy normalization',dict(m=m,K=K,q=q))
                g.check(all(v>=0 for v in numer),'integer occupancy nonnegative',dict(m=m,K=K,q=q))
                for col,n in enumerate(queries):
                    active=F(int(numer@remaining[:,col]),2*sum(wi)*den)
                    passive=sum(wj*(1-qj)**(K+n) for wj,qj in zip(w,q))/2
                    replay=sum(wj*(1-qj)**K for wj,qj in zip(w,q))/2
                    if n==0:g.eq(active,replay,'A11 vs exact subset enumeration',dict(m=m,K=K,q=q,w=w))
                    g.le(active,passive,'active largest-weight queries no worse than IID passive queries',dict(m=m,K=K,n=n,q=q,w=w))
                    row=dict(m=m,q_profile=qkind,w_profile=wkind,q=q,w=w,K=K,queries=n,
                             active=active,passive=passive,replay=replay,replay_gain=F(0))
                    curves.append(row);g.count()
            print(f'E5 completed m={m} q={qkind} w={wkind}',flush=True)
    # Actual separable tabular logits; sum of observed-context CE terms. The
    # optimizer sees only revealed labels. Both labels are trained explicitly.
    learning=[]
    for opt in ('Adam','SGD'):
        theta=np.zeros((1,2));moment=np.zeros_like(theta);variance=moment.copy()
        labels=np.array([[0.,1.]])
        for step in range(1,13):
            gradient=expit(theta)-labels
            theta,moment,variance,record=optimizer(theta,gradient,moment,variance,step,eta=.1,opt=opt)
            learning.append(dict(record,optimizer=opt,labels=labels,logits=theta.copy()))
        p=expit(theta)[0];observed_error=float((p[0]+1-p[1])/2)
        g.eq(float(p[0]),float(1-p[1]),'tabular bit-reflection symmetry',tol=1e-14)
        for m in m_values:
            # Independently execute complete m-context learners on actual
            # revealed subsets, including arbitrary fair bit worlds. This
            # verifies the separability used to integrate the learner curves.
            masks=np.arange(1<<m) if m<=8 else g.rng.integers(0,1<<m,size=256)
            seen=((masks[:,None]>>np.arange(m))&1).astype(bool)
            revealed_labels=g.rng.integers(0,2,size=seen.shape).astype(float)
            training_targets=np.where(seen,revealed_labels,0.)
            tab_theta=np.zeros(seen.shape);tab_m=tab_theta.copy();tab_v=tab_theta.copy()
            for step in range(1,13):
                consumed=np.where(seen,expit(tab_theta)-training_targets,0.)
                tab_theta,tab_m,tab_v,_=optimizer(tab_theta,consumed,tab_m,tab_v,step,eta=.1,opt=opt)
            probabilities=expit(tab_theta)
            error=np.where(revealed_labels==1,1-probabilities,probabilities)
            expected=np.where(seen,observed_error,.5)
            discrepancy=float(np.max(abs(error-expected)))
            g.le(discrepancy,1e-12,'executed full bit-table learner matches integrated curve',
                dict(m=m,optimizer=opt,masks=masks,revealed_labels=np.where(seen,revealed_labels,-1),logits=tab_theta))
            full_table_checks.append(dict(m=m,optimizer=opt,masks=masks,
                model_visible_labels=np.where(seen,revealed_labels,-1),final_logits=tab_theta,
                max_error=discrepancy,known_bit_error=observed_error))
        for row in curves:
            for access in ('active','passive','replay'):
                floor=float(row[access]);risk=floor+(1-2*floor)*observed_error
                learners.append(dict(m=row['m'],K=row['K'],queries=row['queries'],q_profile=row['q_profile'],
                    w_profile=row['w_profile'],optimizer=opt,access=access,risk=risk,
                    information_floor=floor,observed_bit_error=observed_error,steps=12))
                g.le(floor,risk,'actual learner risk above information floor',learners[-1],tol=1e-12)
    # Adversarial allocation search over weights/q and K: comparisons are exact.
    from common import search,distribution
    def make():return dict(q=distribution(g.rng,4,False),w=distribution(g.rng,4,False),K=int(g.rng.integers(6)),n=int(g.rng.integers(4)))
    def mutate(x):
        x=x.copy();key=g.rng.choice(['q','w','K','n'])
        x[key]=distribution(g.rng,4,False) if key in ('q','w') else int(g.rng.integers(4 if key=='n' else 6));return x
    def evaluate(x):
        states=union_recursion(4,{1<<j:p for j,p in enumerate(x['q'])},x['K'])
        active=sum(p*sum(sorted([x['w'][j] for j in range(4) if not mask>>j&1],reverse=True)[x['n']:])/2 for mask,p in states.items())
        passive=sum(w*(1-q)**(x['K']+x['n']) for w,q in zip(x['w'],x['q']))/2
        g.le(active,passive,'adversarial active/passive comparison',x);g.count()
        return float(active-passive),dict(inputs=x,active=active,passive=passive)
    adv=search(g,make,mutate,evaluate,2 if quick else 6,4 if quick else 20)
    g.row['data']=dict(curves=curves,actual_learners=learners,tabular_training=learning,
        executed_full_table_checks=full_table_checks,adversarial=adv,
        learner_contract='Twelve updates of separable summed CE on each revealed bit; unseen logits remain zero. Deployment samples Bernoulli actions.',
        accounting='K counts complete independent singleton-reveal trajectories; queries are additional reveals; replay creates zero information.',
        distinct_contexts='Each occupancy mask is a set of subcontexts within ONE task family, never a count of new tasks.')
    def draw(fig,ax):
        m=max(m_values)
        for n in queries:
            subset=[r for r in curves if r['m']==m and r['q_profile']==r['w_profile']=='uniform' and r['queries']==n]
            ax.plot([r['K'] for r in subset],[float(r['active']) for r in subset],label=f'active Q={n}')
        subset=[r for r in curves if r['m']==m and r['q_profile']==r['w_profile']=='uniform' and r['queries']==4]
        ax.plot([r['K'] for r in subset],[float(r['passive']) for r in subset],'--',label='passive +4')
        learner=[r for r in learners if r['m']==m and r['q_profile']==r['w_profile']=='uniform' and r['queries']==0 and r['optimizer']=='SGD' and r['access']=='replay']
        ax.plot([r['K'] for r in learner],[r['risk'] for r in learner],':',label='actual SGD, Q=0')
        ax.set(xlabel='K complete trusted trajectories',ylabel='Expected regret',title=f'm={m}; replay overlays Q=0');ax.legend(fontsize=8)
    figure('E5','Hidden-bit information and achieved learner curves',draw)
    return g.finish()


def provider_menu():
    points=np.array([[0,0],[1/3,1/3],[2/3,2/3],[1,1],[.1,.1],[.2,.2],[.8,.8],[.9,.9],[0,1],[1,0]])
    menu=[]
    def add(name,items):
        law=np.zeros(len(points))
        for index,p in items:law[index]=p
        menu.append(dict(name=name,law=law))
    add('reference_even',[(0,.5),(3,.5)])
    add('moment3_P',[(0,.25),(2,.75)])
    add('moment3_Q',[(1,.75),(3,.25)])
    add('low_sign_A',[(4,1.)]);add('low_sign_B',[(5,1.)])
    add('high_sign_A',[(6,1.)]);add('high_sign_B',[(7,1.)])
    add('odd_parity',[(8,.5),(9,.5)])
    add('middle_mixture',[(1,.5),(2,.5)])
    return points,menu


def proxy_panel(g,state,recipe,points,menu,full_R=8):
    k=points.shape[1];reference=menu[0]['law'];support=np.flatnonzero(reference)
    stats={}
    for R in (1,4,full_R):
        histories=list(it.product(support,repeat=R-1));weights=np.array([math.prod(reference[i] for i in h) for h in histories])
        labels=np.array([[point]+[points[i] for i in history] for point in points for history in histories])
        result=run_network(labels,start=state,**recipe)
        stats[R]={task:task_value(result['probabilities'],task,3).reshape(len(points),-1)@weights
                  for task in ('moment','parity','threshold')}
        if R==1:
            initial=np.repeat(state['theta'],len(points),axis=0)
            gradients=flatten(neural_gradient(unflatten(initial,k),points))
            stats['gradient']=gradients;stats['delta']=result['theta']-initial
            stats['state']=np.c_[stats['delta'],result['m'],result['v']]
    return stats


def proxy_features(stats,points,law,ref,task,random_score,step):
    difference=law-ref
    input_features=np.c_[points,points**2,np.prod(points,axis=1)]
    dm=difference@input_features
    signs=np.sign(stats['gradient']);patterns=np.unique(signs,axis=0)
    sign_features=np.c_[signs,np.array([np.all(signs==pat,axis=1) for pat in patterns]).T]
    # Fixed feature size across states, even if some joint patterns disappear.
    sign_vector=difference@signs
    joints=[float(difference@((signs[:,0]==a)&(signs[:,-1]==b))) for a,b in it.product((-1,0,1),repeat=2)]
    return dict(input_moment_error=np.r_[dm,np.linalg.norm(dm)],
        mean_gradient=difference@stats['gradient'],sign_only=np.r_[sign_vector,joints],
        first_delta=difference@stats['delta'],delta_plus_m_v=difference@stats['state'],
        short_tail_1=np.array([difference@stats[1][task]]),
        short_tail_4=np.array([difference@stats[4][task]]),random=np.array([random_score]),
        task_identity=np.array([task=='moment',task=='parity',task=='threshold'],float)*np.linalg.norm(difference),
        training_step=np.array([step*np.linalg.norm(difference)]))


def fit_proxies(records):
    models={};names=list(records[0]['features'])
    for name in names:
        X=np.array([r['features'][name] for r in records]);y=np.array([r['gold'] for r in records])
        scale=np.maximum(np.sqrt(np.mean(X**2,axis=0)),1e-8)
        X=X/scale
        # Fixed ridge and no intercept: a zero provider contrast predicts zero.
        ridge=1e-3;coef=np.linalg.solve(X.T@X+ridge*np.eye(X.shape[1]),X.T@y)
        # Leave-one-cell-out pilot residuals set an empirical abstention threshold.
        errors=[]
        for cell in sorted({r['cell'] for r in records}):
            held=np.array([r['cell']==cell for r in records]);train=~held
            beta=np.linalg.solve(X[train].T@X[train]+ridge*np.eye(X.shape[1]),X[train].T@y[train])
            errors.extend(abs(X[held]@beta-y[held]))
        models[name]=dict(scale=scale,coefficients=coef,ridge=ridge,
            pilot_residual_quantile90=float(np.quantile(errors,.9)),pilot_cv_residuals=errors)
    return models


def proxy_metrics(records,models,g):
    metrics={};predictions=[]
    names=list(models)+['full_tail_gold']
    for name in names:
        gold=np.array([r['gold'] for r in records])
        predicted=gold.copy() if name=='full_tail_gold' else np.array([
            float(np.array(r['features'][name])/models[name]['scale']@models[name]['coefficients']) for r in records])
        threshold=0. if name=='full_tail_gold' else models[name]['pilot_residual_quantile90']
        cells=sorted({r['cell'] for r in records});regrets=[];abstained=[];chosen_gains=[];abstain_regrets=[]
        for cell in cells:
            indices=np.array([i for i,r in enumerate(records) if r['cell']==cell])
            best=indices[np.argmax(predicted[indices])]
            regret=max(0.,float(max(gold[indices])-gold[best]))
            regrets.append(regret);chosen_gains.append(float(gold[best]))
            abstain=bool(predicted[best]<=threshold);abstained.append(abstain)
            abstain_regrets.append(float(max(gold[indices])) if abstain else regret)
        nonzero=abs(gold)>1e-10
        wrong=float(np.mean(predicted[nonzero]*gold[nonzero]<0)) if nonzero.any() else 0.
        abstain_sign=float(np.mean(abs(predicted[nonzero])<=threshold)) if nonzero.any() else 0.
        corr=float(spearmanr(predicted,gold).statistic) if np.std(predicted)>1e-15 and np.std(gold)>1e-15 else None
        slope=float(np.polyfit(predicted,gold,1)[0]) if np.std(predicted)>1e-15 else None
        # Uncertainty counts held-out cells, not correlated provider rows.
        boot=np.array([np.mean(g.rng.choice(regrets,len(regrets),replace=True)) for _ in range(200)]) if regrets else np.zeros(200)
        metrics[name]=dict(n_provider_rows=len(records),n_cells=len(cells),selection_regret=float(np.mean(regrets)),
            regret_bootstrap95=np.quantile(boot,[.025,.975]),mean_selected_gain=float(np.mean(chosen_gains)),
            wrong_sign_rate=wrong,Spearman=corr,MAE=float(np.mean(abs(predicted-gold))),
            RMSE=float(np.sqrt(np.mean((predicted-gold)**2))),calibration_slope=slope,
            empirical_interval_coverage=float(np.mean(abs(predicted-gold)<=threshold)),
            abstention_rate=float(np.mean(abstained)),sign_abstention_rate=abstain_sign,
            selection_regret_with_abstention=float(np.mean(abstain_regrets)),threshold=threshold)
        predictions.append(dict(proxy=name,gold=gold,predicted=predicted,cell=[r['cell'] for r in records],
            provider=[r['provider'] for r in records]))
    return metrics,predictions


def check_E6(quick=False):
    g=Group('E6','G5 held-out provider selection using frozen pilot proxies',
            'Exact one-boundary/full-reference-tail enumeration; independent state/task/recipe holdouts', 'controlled_neural')
    points,menu=provider_menu()
    recipes=[dict(opt=opt,eta=eta,eps=eps) for opt,eta in [('Adam',.01),('Adam',.1),('SGD',.1),('SGD',1.)] for eps in (1e-8,1e-3)]
    states={name:snapshot(2,warm) for name,warm in [('fresh',0),('warm2',2),('warm5',5),('warm9',9)]}
    pilot_recipe_ids=[0,4];pilot_state_ids=['fresh','warm2'];pilot_tasks=['moment','parity']
    manifest=dict(seed=g.seed,full_tail_updates=8,pilot_states=pilot_state_ids,pilot_tasks=pilot_tasks,
        pilot_recipes=pilot_recipe_ids,states=states,recipes=recipes,providers=menu,message_support=points,
        held_out_states=['warm5','warm9'],held_out_tasks=['threshold'],
        held_out_recipes=[1,2,3,5,6,7],
        recipe_shift_axes=dict(epsilon_only=[1,5],learning_rate_only=[2,6],epsilon_and_learning_rate=[3,7]),
        memory_transfer='additional frozen fresh-only -> warm and warm-only -> fresh fits',
        design='One candidate boundary followed by seven IID reference updates. No held-out consequence tunes features, ridge, or abstention.')
    split_path=write_json('proxy_split_freeze.json',manifest);split_hash=digest(C.OUT/'proxy_split_freeze.json')
    cache={}
    def records_for(state_names,recipe_ids,tasks,split):
        rows=[]
        for state_name,recipe_id in it.product(state_names,recipe_ids):
            key=(state_name,recipe_id)
            if key not in cache:
                cache[key]=proxy_panel(g,states[state_name],recipes[recipe_id],points,menu)
            stats=cache[key]
            for task in tasks:
                cell=f'{state_name}/{recipe_id}/{task}'
                for provider in menu:
                    reference=menu[0]['law'];delta=provider['law']-reference
                    gold=float(delta@stats[8][task]);random_score=float(g.rng.normal()) if provider is not menu[0] else 0.
                    features=proxy_features(stats,points,provider['law'],reference,task,random_score,states[state_name]['step'])
                    if provider is menu[0]:features={k:np.zeros_like(v) for k,v in features.items()}
                    rows.append(dict(split=split,cell=cell,state=state_name,recipe=recipe_id,task=task,provider=provider['name'],
                        gold=gold,SE=0.,gold_mode='all finite histories',features=features,
                        reference_return=float(reference@stats[8][task]),candidate_return=float(provider['law']@stats[8][task])))
                    g.count()
        return rows
    pilot=records_for(pilot_state_ids,pilot_recipe_ids,pilot_tasks,'pilot')
    models=fit_proxies(pilot)
    memory_models={name:fit_proxies([r for r in pilot if r['state']==name]) for name in pilot_state_ids}
    model_path=write_json('proxy_models_frozen.json',dict(models=models,memory_transfer_models=memory_models,
        split_sha256=split_hash,pilot=pilot))
    # Held-out measurements begin only after both design and fitted models are written.
    cohorts=[('held_out_states',['warm5','warm9'],pilot_recipe_ids,pilot_tasks,models),
        ('held_out_task',pilot_state_ids,pilot_recipe_ids,['threshold'],models),
        ('held_out_epsilon',pilot_state_ids,[1,5],pilot_tasks,models),
        ('held_out_lr',pilot_state_ids,[2,6],pilot_tasks,models),
        ('held_out_epsilon_lr',pilot_state_ids,[3,7],pilot_tasks,models),
        ('fresh_to_warm',['warm5','warm9'],pilot_recipe_ids,pilot_tasks,memory_models['fresh']),
        ('warm_to_fresh',['fresh'],pilot_recipe_ids,pilot_tasks,memory_models['warm2'])]
    reports={};heldout=[];negative=[]
    for name,state_names,recipe_ids,tasks,fit in cohorts:
        records=records_for(state_names,recipe_ids,tasks,name)
        metrics,predictions=proxy_metrics(records,fit,g)
        reports[name]=dict(metrics=metrics,predictions=predictions);heldout.extend(records)
        simple=min(metrics['random']['selection_regret'],metrics['input_moment_error']['selection_regret'])
        for proxy,m in metrics.items():
            if proxy not in ('random','input_moment_error','full_tail_gold') and m['selection_regret']>=simple-1e-12:
                negative.append(dict(split=name,proxy=proxy,regret=m['selection_regret'],best_simple_regret=simple))
        print(f'E6 completed {name}',flush=True)
    g.check(len(menu)>=6,'provider menu size')
    # Positive-epsilon numeric differences do not justify a sign-matched label:
    # explicitly check the consumed-gradient signs for every evaluated state.
    sign_pairs=[]
    for key,stats in cache.items():
        for a,b in ((4,5),(6,7)):
            equal=np.array_equal(np.sign(stats['gradient'][a]),np.sign(stats['gradient'][b]))
            g.check(equal,'declared same-sign provider pair at this actual state',dict(state=key,indices=[a,b],gradients=stats['gradient'][[a,b]]))
            sign_pairs.append(dict(state=key,pair=[a,b],same_sign=equal))
    raw_path=write_json('raw/E6_held_out_records.json',heldout)
    g.row['data']=dict(split_manifest=split_path,split_sha256=split_hash,frozen_models=model_path,
        raw_held_out=raw_path,reports=reports,negative_results=negative,sign_pair_audit=sign_pairs,
        pilot_rows=len(pilot),held_out_rows=len(heldout),
        uncertainty='Exact history integration (SE=0); cell bootstrap describes variation across the finite held-out panel, not a universal confidence certificate.')
    def draw(fig,ax):
        names=['input_moment_error','sign_only','first_delta','delta_plus_m_v','short_tail_4','random']
        x=np.arange(len(reports));width=.12
        for j,name in enumerate(names):ax.bar(x+(j-2.5)*width,[r['metrics'][name]['selection_regret'] for r in reports.values()],width,label=name)
        ax.set_xticks(x,list(reports),rotation=25,ha='right');ax.set(ylabel='Held-out provider selection regret');ax.legend(fontsize=7)
    figure('E6','Frozen proxy transfer and negative results',draw)
    g.info('Proxy results, including failures to beat simple controls, are empirical. No expanded feature family is claimed sufficient.')
    return g.finish()


def check_E7(quick=False):
    g=Group('E7','Gold-audit paired variance and cost forecast',
            'Correct inverse-CDF coupling; exact nested variance and independent paired pilot', 'forecast')
    points,P,Q=law_scalar(3);points=np.array(points,float);P=np.array(P,float);Q=np.array(Q,float)
    laws=[Q,.5*P+.5*Q,np.array([0.,1.,0.,0.]),np.array([0.,0.,1.,0.])]
    pilot_n=400 if quick else 2400;R=4
    uniforms=g.rng.random((pilot_n,R));deployment=g.rng.random(pilot_n)
    def sample(law,u):return np.searchsorted(np.cumsum(law),u,side='right')
    ref_indices=sample(P,uniforms);ref_labels=points[ref_indices]
    before=time.perf_counter()
    ref_run=run_network(ref_labels)
    ref_return=task_value(ref_run['probabilities'],'moment',3)
    ref_outcome=(deployment<ref_return).astype(float)
    pilot_seconds=time.perf_counter()-before
    contrasts=[];conditional=[];raw=dict(uniforms=uniforms,deployment_uniforms=deployment,
        reference_indices=ref_indices,reference_return=ref_return,reference_outcome=ref_outcome)
    decompositions=[]
    for j,law in enumerate(laws):
        candidate_start=time.perf_counter()
        labels=ref_labels.copy();labels[:,0]=points[sample(law,uniforms[:,0])]
        result=run_network(labels);returns=task_value(result['probabilities'],'moment',3)
        outcomes=(deployment<returns).astype(float);delta=outcomes-ref_outcome
        pilot_seconds+=time.perf_counter()-candidate_start
        contrasts.append(delta);conditional.append(returns-ref_return)
        raw[f'candidate{j}_labels']=labels;raw[f'candidate{j}_returns']=returns;raw[f'candidate{j}_outcomes']=outcomes
        # Exact partition of the common U into block pairs with correct marginals.
        edges=np.unique(np.r_[0.,np.cumsum(P),np.cumsum(law),1.])
        blockpairs=[]
        for left,right in zip(edges[:-1],edges[1:]):
            if right>left:
                u=(left+right)/2
                blockpairs.append((sample(P,u),sample(law,u),float(right-left)))
        pm=np.zeros(len(P));qm=pm.copy()
        block_means=[];block_variances=[];deploy_variances=[];weights=[];raw_exact=[]
        suffix=list(it.product(np.flatnonzero(P),repeat=R-1))
        suffix_weight=np.array([math.prod(P[i] for i in h) for h in suffix])
        for ri,ci,mass in blockpairs:
            pm[ri]+=mass;qm[ci]+=mass
            sequences_ref=np.array([[points[ri]]+[points[i] for i in h] for h in suffix])
            sequences_cand=np.array([[points[ci]]+[points[i] for i in h] for h in suffix])
            pr=task_value(run_network(sequences_ref)['probabilities'],'moment',3)
            pc=task_value(run_network(sequences_cand)['probabilities'],'moment',3)
            d=pc-pr;mean=float(suffix_weight@d)
            block_means.append(mean);weights.append(mass)
            block_variances.append(float(suffix_weight@(d-mean)**2))
            deploy_variances.append(float(suffix_weight@(abs(d)-d*d)))
            raw_exact.append(dict(reference_block=int(ri),candidate_block=int(ci),mass=mass,
                suffix=suffix,weights=suffix_weight,reference_return=pr,candidate_return=pc))
        g.le(float(max(abs(pm-P))),1e-12,'reference coupling marginal',dict(pm=pm,P=P))
        g.le(float(max(abs(qm-law))),1e-12,'candidate coupling marginal',dict(qm=qm,law=law))
        weights=np.array(weights);block_means=np.array(block_means);mean=float(weights@block_means)
        block_variance=float(weights@(block_means-mean)**2)
        suffix_variance=float(weights@block_variances);deployment_variance=float(weights@deploy_variances)
        direct_second=sum(r['mass']*float(r['weights']@abs(r['candidate_return']-r['reference_return'])) for r in raw_exact)
        total=direct_second-mean*mean
        g.eq(block_variance+suffix_variance+deployment_variance,total,'nested law of total variance',raw_exact,tol=1e-12)
        decompositions.append(dict(candidate=j,mean=mean,block=block_variance,suffix=suffix_variance,
            deployment=deployment_variance,total=total,pilot_variance=float(np.var(delta,ddof=1)),
            pilot_mean=float(delta.mean()),pilot_SE=float(np.std(delta,ddof=1)/math.sqrt(pilot_n)),raw_exact=raw_exact))
        g.count(pilot_n)
    elapsed=time.perf_counter()-before
    matrix=np.array(contrasts).T;raw['paired_contrasts']=matrix
    path=C.OUT/'raw/E7_paired_pilot.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**raw)
    M=4;delta=.05;error=.1
    hoeffding=math.ceil(2*math.log(2*M/delta)/error**2)
    g.eq(hoeffding,1016,'specified Hoeffding allocation')
    # Maurer–Pontil fixed-n range-two radius, both tails and four candidates:
    # sqrt(2*S²*log(4M/delta)/n) + 14*log(4M/delta)/(3*(n-1)).
    log=math.log(4*M/delta);s2=float(np.var(matrix,axis=0,ddof=1).max())
    n=2
    def radius(n):return math.sqrt(2*s2*log/n)+14*log/(3*(n-1))
    while radius(n)>error:n+=1
    forecast=[]
    for allocation,replicas in [('Hoeffding',hoeffding),('pilot_variance_Maurer_Pontil',n)]:
        branches=(M+1)*replicas
        forecast.append(dict(allocation=allocation,replicas_per_candidate=replicas,
            shared_reference_training_branches=branches,without_reuse_training_branches=2*M*replicas,
            training_updates=branches*R,deployment_trials=branches,
            controlled_CPU_seconds_projection=pilot_seconds*replicas/pilot_n,
            Gemma_full_FT_SFT_wall_hours=branches*17/60,
            Gemma_full_FT_SFT_GPU_hours=branches*4*17/60,
            Gemma_explore_segment_wall_hours=branches*3.3,
            Gemma_LoRA_SFT_wall_hours=branches*1.5,
            Gemma_140_game_eval_wall_hours_interval=[branches*.7,branches*1.6]))
        forecast[-1]['Gemma_eval_game_count']=140*branches
    g.row['data']=dict(pilot_n=pilot_n,paired_covariance=np.cov(matrix,rowvar=False),
        variance_decomposition=decompositions,raw=str(path.relative_to(HERE)),
        measured_CPU_seconds=pilot_seconds,measured_CPU_seconds_per_pilot_pair_set=pilot_seconds/pilot_n,
        total_pilot_and_exact_decomposition_CPU_seconds=elapsed,
        Hoeffding=dict(M=M,delta=delta,error=error,n=hoeffding),
        empirical_Bernstein=dict(n_forecast=n,max_pilot_sample_variance=s2,radius_at_n=radius(n),
            formula='sqrt(2*S^2*log(4M/delta)/n)+14*log(4M/delta)/(3*(n-1))',
            assumptions='Independent bounded paired contrasts in [-1,1]; fixed n; simultaneous two-sided union bound over M.',
            status='FORECAST: variance from independent pilot; a future fixed-n audit must compute its own actual radius; no optional stopping.'),
        forecast=forecast,
        supplied_hpg_inputs=dict(full_FT_K20_4xB200_minutes=17,explore_1GPU_hours=3.3,
            LoRA_K50_1xB200_hours=1.5,eval_140games_per_checkpoint_hours=[.7,1.6]),
        forecast_assumptions='Each measured hpg segment is scaled separately as a scenario per branch; segments are not summed or claimed interchangeable. CPU timing does not infer B200 throughput.',
        reuse='One reference per independent replica is reused across four candidates; cross-candidate covariance retained. One predeclared audit round.',
        actual_Gemma_run=False)
    figure('E7','Paired contrast variance decomposition',lambda fig,ax:(
        ax.bar(np.arange(M),[r['block'] for r in decompositions],label='block'),
        ax.bar(np.arange(M),[r['suffix'] for r in decompositions],bottom=[r['block'] for r in decompositions],label='suffix'),
        ax.bar(np.arange(M),[r['deployment'] for r in decompositions],bottom=[r['block']+r['suffix'] for r in decompositions],label='deployment'),
        ax.set(xlabel='candidate',ylabel='Variance of paired outcome'),ax.legend()))
    g.info('Gemma numbers are forecasts using the supplied hpg timings, not Gemma results. The variance-based allocation is a pilot forecast, not a completed certificate.')
    return g.finish()
