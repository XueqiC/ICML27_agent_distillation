"""Independent W0--W3 and frozen E6 proxy extensions; old groups are unchanged."""
from __future__ import annotations

from common import *
import common as C
from experiments import (weighted_result, provider_menu, proxy_panel, proxy_features,
                         fit_proxies, proxy_metrics)

W_GRID = (0, 1, 2, 5, 10, 20, 50)
BETA = (.9, .999)


def moment_warmup(g, state, labels):
    """Execute lr=0 Adam, including its counter; compare parameter bytes each step."""
    theta = state['theta'].copy(); m = state['m'].copy(); v = state['v'].copy()
    original = theta.tobytes()
    gradients = []
    for j, label in enumerate(labels):
        grad = flatten(neural_gradient(unflatten(theta, state['k']), np.asarray(label)[None]))
        theta, m, v, _ = optimizer(theta, grad, m, v, state['step']+j+1, eta=0.)
        g.check(theta.tobytes() == original, 'lr=0 warm-up preserves parameters bitwise',
                dict(k=state['k'], W=j+1, label=label, gradient=grad))
        gradients.append(grad[0])
    return dict(theta=theta, m=m, v=v, step=state['step']+len(labels), k=state['k'],
                warmup_labels=np.asarray(labels), warmup_gradients=np.asarray(gradients),
                parent_step=state['step'], parameter_bytes_unchanged=theta.tobytes() == original)


def warmth(gradient, v, beta2=.999):
    return np.divide((1-beta2)*gradient**2, beta2*v,
                     out=np.full_like(gradient, np.inf), where=v > 0)


def lemma_W0(g, quick):
    """Rationalized subtraction avoids cancellation of two almost equal steps."""
    def make():
        p = int(g.rng.integers(1, 9))
        return dict(v=10.**g.rng.uniform(-12, 0, p), m=g.rng.normal(size=p),
                    gradient=g.rng.normal(size=p)*10.**g.rng.uniform(-4, 1),
                    t=int(g.rng.integers(1, 2001)), eta=10.**g.rng.uniform(-3, 0),
                    eps=10.**g.rng.uniform(-12, -2),
                    beta=[(.0, .5), (.5, .5), (.9, .9), (.9, .999), (.99, .9999)][int(g.rng.integers(5))])

    def evaluate(x):
        b1, b2 = x['beta']; v=x['v']; grad=x['gradient']
        a = np.sqrt(b2*v/(1-b2**x['t']))
        n = (b1*x['m']+(1-b1)*grad)/(1-b1**x['t'])
        kappa = warmth(grad, v, b2); root = np.sqrt(1+kappa)
        u = abs(n)/(a+x['eps'])
        actual = -x['eta']*n/(np.sqrt((b2*v+(1-b2)*grad**2)/(1-b2**x['t']))+x['eps'])
        surrogate = -x['eta']*n/(a+x['eps'])
        # Exact algebraic difference, evaluated in float64 without subtracting
        # rounded theta outputs. No absolute tolerance, floors, or dropped cases.
        lhs = x['eta']*abs(n)*a*kappa/((root+1)*(a*root+x['eps'])*(a+x['eps']))
        rhs = x['eta']*u*np.minimum(kappa/2, 1)
        ratio = np.divide(lhs, rhs, out=np.zeros_like(lhs), where=rhs > 0)
        record = dict(**x, a=a, n=n, kappa=kappa, u=u, theta_plus=actual,
                      theta_surrogate=surrogate, lhs=lhs, rhs=rhs, ratio=ratio,
                      direct_subtraction=abs(actual-surrogate))
        g.check(np.all(lhs <= rhs*(1+1e-12)), 'W0 coordinatewise, relative slack 1e-12 only',
                record, max(0., float(np.max(lhs-rhs))))
        g.count()
        return float(ratio.max()), record

    def mutate(x):
        y = dict(x)
        y['v'] = np.clip(x['v']*np.exp(g.rng.normal(0, 1, len(x['v']))), 1e-12, 1.)
        y['gradient'] = x['gradient']*np.exp(g.rng.normal(0, 1, len(x['v'])))
        y['m'] = x['m']+g.rng.normal(0, .1, len(x['v']))
        return y

    best = None
    for _ in range(300 if quick else 5000):
        score, instance = evaluate(make())
        if best is None or score > best['score']:
            best = dict(score=score, instance=instance)
    adversary = search(g, make, mutate, evaluate, restarts=6 if quick else 24, steps=30 if quick else 80)
    if adversary['best']['score'] > best['score']: best = adversary['best']
    return dict(best_ratio=best['score'], best=best, adversarial_search=adversary,
                n_instances=(300 if quick else 5000)+adversary['n_evaluations'],
                status='FAIL' if g.failures else 'PASS',
                subtraction='Float64 rationalized exact theta difference; direct subtraction also retained in witnesses. Relative slack 1e-12, no absolute slack.')


class HeadContinuation:
    """C23 with p=1..4 trainable output-head coordinates; hidden layer is fixed.

    Vectorized exact finite suffix integration. Forward sensitivities give b
    analytically, independently of the finite differences estimating constants.
    """
    def __init__(self, p, task, N, R, eta, t):
        self.p=p; self.task=task; self.N=N; self.R=R; self.eta=eta; self.t=t
        self.k = 1 if task == 'moment' else N
        self.base = snapshot(self.k)['theta'][0]
        self.indices = np.arange(len(self.base)-p, len(self.base))
        params = unflatten(self.base[None], self.k)
        self.J = logit_jacobian(params)[0, :, self.indices].T
        # NumPy's advanced indexing yields (p,k), hence the transpose above.
        self.logit0 = neural_forward(params)[2][0]
        self.q0 = self.base[self.indices]
        self.suffix = np.array(list(it.product((0., 1.), repeat=R))).reshape(-1, R) if R else np.empty((1, 0))

    def probs(self, q):
        return expit(self.logit0+(q-self.q0)@self.J.T)

    def gradient(self, q, labels):
        return (self.probs(q)-labels)@self.J/self.k

    def value(self, q, m, v, tangent=None):
        q=np.atleast_2d(q); m=np.broadcast_to(m, q.shape); v=np.broadcast_to(v, q.shape)
        batch=len(q); histories=len(self.suffix)
        q=np.repeat(q, histories, axis=0).copy()
        m=np.repeat(m, histories, axis=0).copy(); v=np.repeat(v, histories, axis=0).copy()
        if tangent is not None:
            tq, tm, tv = [np.repeat(np.broadcast_to(a, (batch, self.p, self.p)), histories, axis=0).copy() for a in tangent]
        for r in range(self.R):
            prob=self.probs(q)
            labels=np.tile(self.suffix[:, r], batch)[:, None]
            grad=(prob-labels)@self.J/self.k
            if tangent is not None:
                hess=np.einsum('ki,nk,kj->nij', self.J, prob*(1-prob), self.J)/self.k
                dg=np.einsum('nij,njl->nil', hess, tq)
                tm=.9*tm+.1*dg
                tv=.999*tv+.002*grad[:, :, None]*dg
            step=self.t+r+1
            m=.9*m+.1*grad; v=.999*v+.001*grad**2
            a=np.sqrt(v/(1-.999**step)); denom=a+1e-8
            mh=m/(1-.9**step)
            q -= self.eta*mh/denom
            if tangent is not None:
                da=tv/(2*(1-.999**step)*a[:, :, None])
                tq -= self.eta*(tm/(1-.9**step)/denom[:, :, None]-mh[:, :, None]*da/denom[:, :, None]**2)
        prob=self.probs(q)
        values=task_value(prob, self.task, self.N).reshape(batch, histories).mean(axis=1)
        if tangent is None: return values
        dp=prob[:, :, None]*(1-prob[:, :, None])*np.einsum('ki,nij->nkj', self.J, tq)
        if self.task == 'moment':
            derivative=self.N*prob[:, 0, None]**(self.N-1)*dp[:, 0]
        else:
            derivative=sum(-np.prod(np.delete(1-2*prob, k, axis=1), axis=1)[:, None]*dp[:, k] for k in range(self.k))
        return values, derivative.reshape(batch, histories, self.p).mean(axis=1)


def w1_panel(g, p, n, R, eta, task, quick):
    t=int(g.rng.integers(20, 201)); N=3 if task == 'moment' else 2
    cont=HeadContinuation(p, task, N, R, eta, t)
    theta=cont.q0+g.rng.normal(0, .05, p)
    m=g.rng.normal(0, .025, p); v=10.**g.rng.uniform(-2.5, -1.5, p)
    labels=np.repeat(np.linspace(0, 1, n)[:, None], cont.k, axis=1)
    gradients=cont.gradient(theta, labels)
    full_theta=np.repeat(cont.base[None],n,axis=0)
    full_theta[:,cont.indices]=theta
    imported_gradient=flatten(neural_gradient(unflatten(full_theta,cont.k),labels))[:,cont.indices]
    g.check(np.max(abs(gradients-imported_gradient))<=1e-14,
            'W1 restricted-head gradients agree with unchanged C23 neural_gradient',
            dict(theta=full_theta,labels=labels,gradient=gradients,imported_gradient=imported_gradient))
    a=np.sqrt(.999*v/(1-.999**t)); c1=.1/(1-.9**t)
    def surrogate(gs):
        gs=np.atleast_2d(gs)
        return theta-eta*(.9*m+.1*gs)/(1-.9**t)/(a+1e-8), .9*m+.1*gs, .999*v
    def Ffun(gs): return cont.value(*surrogate(gs))
    qtilde, mplus, vbase=surrogate(gradients)
    qactual, _, vplus, _=optimizer(np.broadcast_to(theta, gradients.shape), gradients, m, v, t, eta=eta)
    psi=cont.value(qactual, mplus, vplus)
    zero=np.zeros(p)
    c, b=cont.value(*surrogate(zero), tangent=(-eta*c1*np.diag(1/(a+1e-8)), .1*np.eye(p), np.zeros((p,p))))
    c=float(c[0]); b=b[0]
    h=2e-6
    b_fd=np.array([(Ffun(h*np.eye(p)[i])[0]-Ffun(-h*np.eye(p)[i])[0])/(2*h) for i in range(p)])
    g.check(np.max(abs(b-b_fd)) <= 1e-8, 'W1 analytic forward sensitivity vs independent central difference',
            dict(p=p,n=n,R=R,eta=eta,task=task,b=b,b_fd=b_fd))
    kappa=warmth(gradients, v)
    g.check(np.all(kappa <= 1), 'W1 declared second-moment segment lies below fixed kbar=1', dict(kappa=kappa))
    # Predeclared central differences on both connecting segment families.
    # No constants are enlarged after seeing a response residual or law gap.
    alphas=np.linspace(0, 1, 5 if quick else 9)
    theta_samples=np.concatenate([qactual+x*(qtilde-qactual) for x in alphas])
    m_samples=np.tile(mplus, (len(alphas),1)); v_samples=np.tile(vplus, (len(alphas),1))
    L=np.zeros(p)
    for i in range(p):
        e=np.eye(p)[i]*1e-5
        L[i]=max(abs((cont.value(theta_samples+e,m_samples,v_samples)-cont.value(theta_samples-e,m_samples,v_samples))/(2e-5)))
    relative_samples=np.concatenate([1+x*kappa for x in alphas])
    q_samples=np.tile(qtilde,(len(alphas),1)); v_samples=vbase*relative_samples
    derivatives=[]
    for i in range(p):
        e=vbase*np.eye(p)[i]*1e-4
        derivatives.append((cont.value(q_samples,m_samples,v_samples+e)-cont.value(q_samples,m_samples,v_samples-e))/(2e-4))
    Lv=float(np.max(np.sum(abs(np.array(derivatives)),axis=0)))
    # H_G estimate = max sum_ij |H_ij|, an upper bound on the sampled
    # infinity-norm quadratic form. Sample radial segments AND convex hull.
    nodes=np.vstack([zero, gradients])
    hull=np.vstack([x*gradients for x in alphas]+[g.rng.dirichlet(np.ones(n+1), 8 if quick else 24)@nodes])
    Hg=0.
    for dh in (1e-3, 5e-4):
        Hess=np.empty((len(hull),p,p)); center=Ffun(hull)
        for i in range(p):
            ei=np.eye(p)[i]*dh
            Hess[:,i,i]=(Ffun(hull+ei)-2*center+Ffun(hull-ei))/dh**2
            for j in range(i):
                ej=np.eye(p)[j]*dh
                Hess[:,i,j]=Hess[:,j,i]=(Ffun(hull+ei+ej)-Ffun(hull+ei-ej)-Ffun(hull-ei+ej)+Ffun(hull-ei-ej))/(4*dh**2)
        Hg=max(Hg,float(np.max(np.sum(abs(Hess),axis=(1,2)))))
    u=abs((.9*m+.1*gradients)/(1-.9**t))/(a+1e-8)
    normalization=eta*np.sum(L*u*np.minimum(kappa/2,1),axis=1)
    second=Lv*np.max(np.minimum(kappa,1),axis=1)
    curvature=.5*Hg*np.max(abs(gradients),axis=1)**2
    residual=normalization+second+curvature
    state=dict(theta=theta,m=m,v=v,t=t,trainable_indices=cont.indices,fixed_C23_parameters=cont.base)
    panel=dict(p=p,n_messages=n,R=R,eta=eta,task=task,N=N,state=state,labels=labels,G=gradients,
               psi=psi,c=c,b=b,b_finite_difference=b_fd,L_i=L,L_v=Lv,H_G=Hg,kbar=1.,kappa=kappa,
               r=residual,normalization=normalization,second_moment=second,curvature=curvature,
               pointwise_residual=abs(psi-c-gradients@b),hull_samples=hull)
    g.check(np.all(abs(psi-c-gradients@b) <= residual+1e-12),
            'W1 pointwise empirical consistency with estimated constants',panel)
    pair_rows=[]
    def evaluate(pair, kind='unequal_mean'):
        P,Q=pair; signed=float((P-Q)@psi); dG=(P-Q)@gradients
        prediction=float(b@dG); bound=abs(prediction)+float((P+Q)@residual)
        row=dict(kind=kind,P0=P,P1=Q,mean_gradient_difference=dG,signed_gap=signed,gap=abs(signed),
                 analytic_first_order_prediction=prediction,prediction_error=abs(signed-prediction),bound=bound,
                 ratio=abs(signed)/bound if bound else 0.)
        g.le(abs(signed),bound,'W1 two-law empirical consistency with estimated constants',dict(panel=panel,pair=row),tol=1e-12)
        if kind=='equal_mean': g.le(float(np.max(abs(dG))),1e-14,'W1 equal consumed-gradient means',row)
        pair_rows.append(row);g.count()
        return row['ratio'],row
    P=np.zeros(n);P[[0,-1]]=.5
    Q=np.ones(n)/n if n==2 else np.r_[0,np.full(n-2,1/(n-2)),0]
    evaluate((P,Q),'equal_mean')
    evaluate((g.rng.dirichlet(np.ones(n)),g.rng.dirichlet(np.ones(n))))
    def make(): return g.rng.dirichlet(np.ones(n)),g.rng.dirichlet(np.ones(n))
    def mutate(pair):
        return tuple((law+.15*g.rng.dirichlet(np.ones(n)))/1.15 for law in pair)
    adversary=search(g,make,mutate,evaluate,restarts=2 if quick else 4,steps=6 if quick else 16)
    # Equal-mean adversary stays on the same moment fiber by common mixtures.
    def equal_make(): return (float(g.rng.uniform()),g.rng.dirichlet(np.ones(n)))
    def equal_mutate(x): return (float(np.clip(x[0]+g.rng.normal(0,.2),0,1)),(x[1]+g.rng.dirichlet(np.ones(n)))/2)
    def equal_evaluate(x): return evaluate((x[0]*P+(1-x[0])*x[1],x[0]*Q+(1-x[0])*x[1]),'equal_mean')
    equal_adversary=search(g,equal_make,equal_mutate,equal_evaluate,restarts=2,steps=6 if quick else 16)
    return dict(**panel,pairs=pair_rows,adversarial_search=adversary,equal_mean_search=equal_adversary)


def proposition_W2(g):
    grad=np.array([1.,-1/3,0.]); P=np.array([.25,.75,0.]); Q=np.array([0.,0.,1.])
    rational_mean=F(1,4)*1+F(3,4)*F(-1,3)
    g.eq(rational_mean,F(0),'W2 exact equal-mean rational laws')
    rows=[]; curves=[]
    for eta,w in it.product((.01,.1),(-.7,1.)):
        exact=abs(F(str(w)))*F(str(eta))/2
        ideal=float(abs(w*(P-Q)@(-eta*np.sign(grad))))
        g.le(abs(ideal-float(exact)),1e-15,'W2 ideal sign exact gap',dict(eta=eta,w=w,exact=exact,actual=ideal))
        finite=optimizer(np.zeros(3),grad,np.zeros(3),np.zeros(3),1,eta=eta)[0]
        finite_gap=abs(float(w*(P-Q)@finite))
        eps=F(1,10**8)
        finite_exact=abs(F(str(w))*F(str(eta))*(F(3,4)*F(1,3)/(F(1,3)+eps)-F(1,4)/(1+eps)))
        g.le(abs(finite_gap-float(finite_exact)),1e-15,'W2 finite-epsilon independent rational prediction',dict(exact=finite_exact,actual=finite_gap))
        rows.append(dict(eta=eta,w=w,ideal_gap=ideal,ideal_exact=exact,finite_eps=1e-8,finite_gap=finite_gap,finite_exact=finite_exact))
        for k in np.logspace(-6,2,33):
            v=.001/(.999*k); t=1000
            theta=optimizer(np.zeros(3),grad,np.zeros(3),np.full(3,v),t,eta=eta)[0]
            gap=abs(float(w*(P-Q)@theta))
            a=math.sqrt(.999*v/(1-.999**t)); scale=abs(w)*eta*.1/(1-.9**t)/(a+1e-8)
            curves.append(dict(eta=eta,w=w,kappa_max=k,gap=gap,preconditioned_scale=scale,normalized_gap=gap/scale,v=v,t=t,m=0.))
            g.count()
    small=[x for x in curves if x['eta']==.1 and x['w']==1. and x['kappa_max']<=1e-2]
    slope=float(np.polyfit(np.log([x['kappa_max'] for x in small]),np.log([x['gap'] for x in small]),1)[0])
    normalized=float(np.polyfit(np.log([x['kappa_max'] for x in small]),np.log([x['normalized_gap'] for x in small]),1)[0])
    return dict(cold=rows,warm=curves,loglog_slope=slope,normalized_loglog_slope=normalized,
                note='Same unscaled laws; fixed t=1000,m=0,beta. Varying v changes both kappa and the preconditioner. Raw gap can shrink faster than linearly; normalized gap isolates the kappa factor.')


def fit_inverse(x, y):
    x=np.asarray(x);y=np.asarray(y);keep=(x>0)&np.isfinite(y)&(y>0)
    x=x[keep];y=y[keep]
    if len(x)<2: return dict(n=len(x),c=None,loglog_slope=None,R2=None)
    design=1/x; c=float(design@y/(design@design)); pred=c*design
    total=float(np.sum((y-y.mean())**2))
    return dict(n=len(x),c=c,loglog_slope=float(np.polyfit(np.log(x),np.log(y),1)[0]),
                R2=1-float(np.sum((y-pred)**2))/total if total else None,
                relative_RMSE=float(np.linalg.norm(y-pred)/np.linalg.norm(y)))


def trajectory_warmth(g,quick):
    rows=[];fits=[];starts={};raw={}; replicas=24 if quick else 128; steps=80
    for task,N,pts,P,Q in cases():
        points=np.asarray(pts,float); k=points.shape[1]
        reference=np.asarray(P,float)
        warm_indices=g.rng.choice(len(points),50,p=reference)
        initial=snapshot(k)
        states={W:moment_warmup(g,initial,points[warm_indices[:W]]) for W in W_GRID}
        starts[(task,N)]=states
        for source,law in zip(('P','Q'),(P,Q)):
            indices=g.rng.choice(len(points),size=(replicas,steps),p=np.asarray(law,float))
            for eta,W in it.product((.01,.1),W_GRID):
                state=states[W];theta=np.repeat(state['theta'],replicas,axis=0)
                m=np.repeat(state['m'],replicas,axis=0);v=np.repeat(state['v'],replicas,axis=0)
                series=[];kap=[]
                for s in range(steps):
                    grad=flatten(neural_gradient(unflatten(theta,k),points[indices[:,s]]))
                    ka=warmth(grad,v);kap.append(ka)
                    # Nearest-rank quantiles preserve infinities without inf-inf NaNs.
                    quant=np.quantile(ka,[.1,.25,.5,.75,.9,.99],method='nearest')
                    row=dict(task=task,N=N,source=source,eta=eta,W=W,s=s,evaluation=s+1,
                             prior_gradient_evaluations=W+s,quantiles=quant,infinite_fraction=float(np.mean(~np.isfinite(ka))))
                    rows.append(row);series.append(row)
                    theta,m,v,_=optimizer(theta,grad,m,v,W+s+1,eta=eta)
                fits.append(dict(task=task,N=N,source=source,eta=eta,W=W,
                    fit=fit_inverse([W+r['s'] for r in series if 1<=r['s']<=79],
                                    [r['quantiles'][2] for r in series if 1<=r['s']<=79])))
                key=f'{task}{N}_{source}_eta{eta}_W{W}'
                raw[key+'_kappa']=np.array(kap);raw[key+'_final_theta']=theta
                raw[key+'_final_m']=m;raw[key+'_final_v']=v
                g.count(replicas*steps)
            raw[f'{task}{N}_{source}_label_indices']=indices
        raw[f'{task}{N}_warm_indices']=warm_indices
        print(f'E8 warmth trajectories completed {task} N={N}',flush=True)
    path=C.OUT/'raw'/'E8_trajectories.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**raw)
    return dict(quantile_levels=[.1,.25,.5,.75,.9,.99],per_step=rows,fits=fits,replicas=replicas,
                raw=str(path.relative_to(HERE)),raw_sha256=digest(path),
                snapshots=[dict(task=key[0],N=key[1],states=value) for key,value in starts.items()],
                fit_definition='Least squares median=c/(W+s), s=1..79 prior parameter updates; log-log slope free. Only positive finite medians enter the fit; their count n is reported, and unfittable series have null coefficients. All zero/infinite quantiles remain in per_step/raw. Reference warm-up law=P; shared IID prefixes across W. Nonstationarity and near-zero coordinates retained.'),starts


def gap_warmup(g,starts,quick):
    rows=[]
    for task,N,points,P,Q in cases():
        for eta,R in it.product((.01,.1),(1,4,40)):
            streams=[training_sequences(g.rng,points,law,R,240 if quick else 2000,quick) for law in (P,Q)]
            for W in W_GRID:
                sources=[]; path=C.OUT/'raw'/f'E8_gap_{task}{N}_eta{eta}_R{R}_W{W}.npz';raw={}
                for source,(labels,weights,mode) in zip(('P','Q'),streams):
                    branches={}
                    for branch,name in (('actual','actual_Adam'),('frozen','frozen_Adam')):
                        result=run_network(labels,eta=eta,branch=branch,start=starts[(task,N)][W])
                        summary=weighted_result(result['probabilities'],weights,task,N,mode)
                        branches[name]={key:summary[key] for key in ('mean','SE','mode','n_histories')}
                        raw[source+'_'+name+'_return']=summary['values']
                        if branch=='actual':
                            affine=weighted_result(result['affine'],weights,task,N,mode)
                            branches['tangent_affine_SGD']={key:affine[key] for key in ('mean','SE','mode','n_histories')}
                            raw[source+'_tangent_affine_SGD_return']=affine['values']
                    raw[source+'_labels']=labels;raw[source+'_weights']=weights
                    sources.append(dict(source=source,branches=branches,history_sha256=hashlib.sha256(labels.tobytes()).hexdigest()))
                np.savez_compressed(path,**raw)
                gaps={key:sources[0]['branches'][key]['mean']-sources[1]['branches'][key]['mean'] for key in sources[0]['branches']}
                ses={key:math.hypot(sources[0]['branches'][key]['SE'],sources[1]['branches'][key]['SE']) for key in gaps}
                rows.append(dict(task=task,N=N,eta=eta,R=R,W=W,gaps=gaps,SE=ses,sources=sources,
                                 raw=str(path.relative_to(HERE)),raw_sha256=digest(path)))
                g.count()
        print(f'E8 gap sweep completed {task} N={N}',flush=True)
    return rows


def check_E8(quick=False):
    g=Group('E8','Optimizer warmth W0--W3 and C23 moment warm-up',
            'Float64 adversarial W0; finite-suffix W1 with sampled constants; exact W2; actual trajectories and E2 warm-up matrix',
            'empirical_consistency_and_controlled_neural')
    w0=lemma_W0(g,quick)
    print(f"E8 W0 best ratio {w0['best_ratio']:.12g}",flush=True)
    before=len(g.failures);panels=[]
    # All dimensions, message counts, suffix lengths and rates even in quick mode.
    for p,R,eta in it.product(range(1,5),(0,2,6),(.01,.1)):
        n=2+(p+R+int(eta==.1))%5
        panels.append(w1_panel(g,p,n,R,eta,'moment' if (p+R)%2 else 'parity',quick))
    w1=dict(status='FAIL' if len(g.failures)>before else 'PASS',kind='empirical consistency check',
            panels=panels,n_panels=len(panels),n_law_pairs=sum(len(x['pairs']) for x in panels),
            best_ratio=max(y['ratio'] for x in panels for y in x['pairs']),
            procedure='p=1..4 trainable C23 output-head coordinates, other coordinates fixed; R=0,2,6 IID reference suffix on all-zero/all-one labels, probability 1/2 each. L_i: central theta differences h=1e-5 along actual-to-surrogate segments. L_v: sum absolute relative-v derivatives h=1e-4 along second-moment segments, fixed kbar=1. H_G: sum absolute Hessian entries, central differences h=1e-3 and 5e-4, sampled radial segments and convex-hull Dirichlet points. No post-outcome padding. Analytic b uses forward sensitivities. Two-message equal-mean case is necessarily identical laws for distinct gradients.',
            certificate=False)
    print(f"E8 W1 {w1['status']} on {w1['n_law_pairs']} law pairs",flush=True)
    w2=proposition_W2(g)
    w3,starts=trajectory_warmth(g,quick)
    gaps=gap_warmup(g,starts,quick)
    g.row['best_ratio']=w0['best_ratio']
    g.row['data']=dict(W0=w0,W1=w1,W2=w2,W3=w3,gaps=gaps,W_grid=W_GRID,
                       gap_design='E2 run_network and weighted_result unchanged. Exact histories when <=4096, otherwise independent 2000/source (240 quick). Same histories across W and branches. Warm-up is one declared common IID P prefix, conditional results; no averaging over warm-up prefixes.')
    def draw_gap(fig,ax):
        fig.clear();axes=fig.subplots(2,3)
        for ax,(eta,R) in zip(axes.flat,it.product((.01,.1),(1,4,40))):
            for color,(task,N,_,_,_) in zip(plt.get_cmap('tab10').colors,cases()):
                subset=[r for r in gaps if r['task']==task and r['N']==N and r['eta']==eta and r['R']==R]
                for branch,style in [('actual_Adam','-'),('frozen_Adam','--'),('tangent_affine_SGD',':')]:
                    ax.plot(W_GRID,[abs(r['gaps'][branch]) for r in subset],style,color=color,
                            label=f'{task} N={N}' if branch=='actual_Adam' else None)
            ax.set(title=f'eta={eta}, R={R}',xlabel='lr=0 warm-up W',ylabel='Absolute return gap')
        axes[0,2].legend(fontsize=6)
    figure('E8','C23 gap vs moment warm-up; solid Adam, dashed frozen, dotted affine',draw_gap)
    def draw_warmth(fig,ax):
        for W in (0,5,20,50):
            subset=[r for r in w3['per_step'] if r['task']=='moment' and r['N']==3 and r['source']=='P' and r['eta']==.01 and r['W']==W and r['s']>0]
            ax.loglog([r['W']+r['s'] for r in subset],[r['quantiles'][2] for r in subset],label=f'W={W}')
        ax.set(xlabel='Prior gradient evaluations W+s',ylabel='Median kappa');ax.legend()
    figure('E8_warmth','Actual C23 trajectory warmth (moment N=3, eta=.01)',draw_warmth)
    def draw_w2(fig,ax):
        subset=[r for r in w2['warm'] if r['eta']==.1 and r['w']==1.]
        ax.loglog([r['kappa_max'] for r in subset],[r['gap'] for r in subset],label='raw gap')
        ax.loglog([r['kappa_max'] for r in subset],[r['normalized_gap'] for r in subset],label='gap / preconditioned scale')
        ax.set(xlabel='max kappa',ylabel='Equal-mean response gap');ax.legend()
    figure('E8_W2','W2 cold witness at warm states',draw_w2)
    g.row['notes'].append('PASS gates W0, W1 empirical consistency and implementation identities only; W3 fits and warm-up effects are descriptive, not uniform certificates. Negative trends are retained.')
    return g.finish()


def check_E9(quick=False):
    g=Group('E9','G5 proxy transfer under lr=0 moment warm-up',
            'Unchanged E6 provider/features/gold/fits/metrics; frozen E6 pilot and paired cold/warmed held-out cells',
            'controlled_neural')
    # These contain pilot data only. Never read E6 held-out records to fit or
    # select a model. Reuse the exact originally frozen pilot and split contract.
    original_split=HERE/'proxy_split_freeze.json'
    original_models=HERE/'proxy_models_frozen.json'
    design=json.loads(original_split.read_text()); pilot_contract=json.loads(original_models.read_text())
    points,menu=provider_menu();recipes=design['recipes']
    states={name:snapshot(2,warm) for name,warm in [('fresh',0),('warm2',2),('warm5',5),('warm9',9)]}
    warm_indices=g.rng.choice(len(points),20,p=menu[0]['law'])
    for W in (5,20): states[f'fresh_W{W}']=moment_warmup(g,states['fresh'],points[warm_indices[:W]])
    pilot=pilot_contract['pilot']
    # Refitting the SAME pilot using the SAME E6 function is deterministic.
    models=fit_proxies(pilot)
    memory_models={name:fit_proxies([r for r in pilot if r['state']==name]) for name in ('fresh','warm2')}
    for name,model in models.items():
        for field in ('scale','coefficients'):
            g.check(np.array_equal(model[field],pilot_contract['models'][name][field]),
                    'E9 model fit is byte-identical to frozen E6 pilot fit',dict(proxy=name,field=field))
    # Random controls are paired across cold/warmed arms before any measurement.
    random_scores=g.rng.normal(size=(len(recipes),3,len(menu)))
    random_scores[:,:,0]=0.
    paired_specs=[('held_out_states',[0,4],['moment','parity'],'warm2'),
                  ('held_out_task',[0,4],['threshold'],'combined'),
                  ('held_out_epsilon',[1,5],['moment','parity'],'combined'),
                  ('held_out_lr',[2,6],['moment','parity'],'combined'),
                  ('held_out_epsilon_lr',[3,7],['moment','parity'],'combined')]
    manifest=dict(seed=g.seed,parent_split_sha256=digest(original_split),parent_models_sha256=digest(original_models),
        original_E6_design=design,states=states,recipes=recipes,providers=menu,message_support=points,
        warmup=dict(W=[5,20],parent='fresh',law=menu[0],sampled_indices=warm_indices,eta=0.,beta=BETA,
                    scope='one common IID reference prefix, shared across W; conditional on this declared warm-up'),
        paired_cohorts=[dict(split=s,recipes=r,tasks=t,model=f) for s,r,t,f in paired_specs],
        arms=['fresh','fresh_W5','fresh_W20'],random_control_scores=random_scores,
        pilot_source='Only original frozen E6 pilot rows. No new response is used for fitting.',
        state_holdout='Use original warm2-only pilot model for paired state holdout: fresh and both moment-warmed descendants are all excluded from this fit.',
        gold='Unchanged E6: one candidate boundary + seven IID reference updates, all finite histories, same C23 returns.',
        metric_definition='Unchanged E6 proxy_metrics, including nonzero gold cutoff=1e-10 and zero predictions not counted as wrong-sign. Coverage/abstention retained.',
        aggregation='Equal weight per held-out split for each metric. Lower regret/wrong-sign, higher Spearman is improvement; no significance claim.')
    split_path=write_json('proxy_split_freeze_E9.json',manifest)
    split_hash=digest(C.OUT/'proxy_split_freeze_E9.json')
    model_path=write_json('proxy_models_frozen_E9.json',dict(models=models,memory_transfer_models=memory_models,
        pilot=pilot,split_sha256=split_hash,parent_models_sha256=digest(original_models),
        fitting='Unchanged fit_proxies; original E6 pilot only; frozen before every new held-out panel'))
    model_hash=digest(C.OUT/'proxy_models_frozen_E9.json')
    freeze_events=[dict(event='split_written',sha256=split_hash),dict(event='models_written',sha256=model_hash)]
    cache={};heldout=[]
    def records_for(state_names,recipe_ids,tasks,split):
        rows=[]
        # Enforce the on-disk frozen-before-measurement order, also for --only.
        g.check(digest(C.OUT/'proxy_split_freeze_E9.json')==split_hash and
                digest(C.OUT/'proxy_models_frozen_E9.json')==model_hash,
                'E9 split and coefficients frozen before held-out measurement',dict(split=split))
        for state_name,recipe_id in it.product(state_names,recipe_ids):
            key=(state_name,recipe_id)
            if key not in cache:
                freeze_events.append(dict(event='held_out_panel',state=state_name,recipe=recipe_id))
                cache[key]=proxy_panel(g,states[state_name],recipes[recipe_id],points,menu)
            stats=cache[key]
            for task in tasks:
                cell=f'{state_name}/{recipe_id}/{task}'
                for j,provider in enumerate(menu):
                    reference=menu[0]['law'];delta=provider['law']-reference
                    score=random_scores[recipe_id,('moment','parity','threshold').index(task),j]
                    features=proxy_features(stats,points,provider['law'],reference,task,score,states[state_name]['step'])
                    if j==0: features={name:np.zeros_like(v) for name,v in features.items()}
                    rows.append(dict(split=split,cell=cell,state=state_name,recipe=recipe_id,task=task,provider=provider['name'],
                        gold=float(delta@stats[8][task]),SE=0.,gold_mode='all finite histories',features=features,
                        reference_return=float(reference@stats[8][task]),candidate_return=float(provider['law']@stats[8][task])))
                    g.count()
        heldout.extend(rows)
        return rows
    # Preserve all seven E6 split definitions, including its existing nonzero-lr
    # warm states. The new paired intervention is reported separately below.
    original_cohorts=[('held_out_states',['warm5','warm9'],[0,4],['moment','parity'],models),
        ('held_out_task',['fresh','warm2'],[0,4],['threshold'],models),
        ('held_out_epsilon',['fresh','warm2'],[1,5],['moment','parity'],models),
        ('held_out_lr',['fresh','warm2'],[2,6],['moment','parity'],models),
        ('held_out_epsilon_lr',['fresh','warm2'],[3,7],['moment','parity'],models),
        ('fresh_to_warm',['warm5','warm9'],[0,4],['moment','parity'],memory_models['fresh']),
        ('warm_to_fresh',['fresh'],[0,4],['moment','parity'],memory_models['warm2'])]
    original_reports={}
    for name,ss,rr,tt,fit in original_cohorts:
        records=records_for(ss,rr,tt,'E6_'+name)
        metrics,predictions=proxy_metrics(records,fit,g)
        original_reports[name]=dict(metrics=metrics,predictions=predictions)
    reports={};comparisons=[]
    for split,recipe_ids,tasks,fit_name in paired_specs:
        fit=models if fit_name=='combined' else memory_models[fit_name]
        arms={}
        for state_name in ('fresh','fresh_W5','fresh_W20'):
            records=records_for([state_name],recipe_ids,tasks,split)
            metrics,predictions=proxy_metrics(records,fit,g)
            arms[state_name]=dict(metrics=metrics,predictions=predictions)
        reports[split]=arms
        for proxy in arms['fresh']['metrics']:
            for W in (5,20):
                cold=arms['fresh']['metrics'][proxy]; warm=arms[f'fresh_W{W}']['metrics'][proxy]
                changes={metric:float(warm[metric]-cold[metric]) if cold[metric] is not None and warm[metric] is not None else None
                         for metric in ('selection_regret','wrong_sign_rate','Spearman')}
                comparisons.append(dict(split=split,proxy=proxy,W=W,cold=cold,warm=warm,warm_minus_cold=changes))
        print(f'E9 paired {split} completed',flush=True)
    summary=[]
    for proxy,W in it.product(models,(5,20)):
        subset=[r for r in comparisons if r['proxy']==proxy and r['W']==W]
        metrics={}
        for metric in ('selection_regret','wrong_sign_rate','Spearman'):
            available=[r for r in subset if r['warm_minus_cold'][metric] is not None]
            delta=float(np.mean([r['warm_minus_cold'][metric] for r in available])) if available else None
            metrics[metric]=dict(mean_warm_minus_cold=delta,n_splits=len(available),
                improved=None if delta is None else bool(delta>0 if metric=='Spearman' else delta<0),
                n_splits_improved=sum(r['warm_minus_cold'][metric]>0 if metric=='Spearman' else r['warm_minus_cold'][metric]<0 for r in available))
        summary.append(dict(proxy=proxy,W=W,metrics=metrics,
                            all_three_improve=all(v['improved'] is True for v in metrics.values())))
    sign_pairs=[]
    for key,stats in cache.items():
        for a,b in ((4,5),(6,7)):
            equal=np.array_equal(np.sign(stats['gradient'][a]),np.sign(stats['gradient'][b]))
            g.check(equal,'E9 same-sign E6 provider pair at measured state',dict(state=key,indices=[a,b],gradients=stats['gradient'][[a,b]]))
            sign_pairs.append(dict(state=key,pair=[a,b],same_sign=equal))
    g.check(digest(C.OUT/'proxy_models_frozen_E9.json')==model_hash,'E9 no model modification after held-out measurements')
    raw_path=write_json('raw/E9_held_out_records.json',heldout)
    cheap=('mean_gradient','first_delta','delta_plus_m_v','input_moment_error')
    g.row['data']=dict(split_manifest=split_path,split_sha256=split_hash,frozen_models=model_path,models_sha256=model_hash,
        freeze_events=freeze_events,pilot_rows=len(pilot),held_out_rows=len(heldout),original_E6_splits=original_reports,
        reports=reports,comparisons=comparisons,summary=summary,cheap_proxy_results=[r for r in summary if r['proxy'] in cheap],
        sign_pair_audit=sign_pairs,raw_held_out=raw_path,
        uncertainty='Exact finite history gold; E6 cell bootstrap retained. Paired summary is descriptive across five split panels, not a population certificate.')
    def draw(fig,ax):
        fig.clear();axes=fig.subplots(1,3)
        names=list(models)
        for ax,metric in zip(axes,('selection_regret','wrong_sign_rate','Spearman')):
            for W,offset in ((5,-.18),(20,.18)):
                vals=[next(r for r in summary if r['proxy']==name and r['W']==W)['metrics'][metric]['mean_warm_minus_cold'] for name in names]
                ax.barh(np.arange(len(names))+offset,[np.nan if v is None else v for v in vals],height=.35,label=f'W={W}')
            ax.set_yticks(range(len(names)),names,fontsize=6);ax.axvline(0,color='black',lw=.5)
            ax.set_title(metric,fontsize=9);ax.set_xlabel('Warm minus cold (split mean)',fontsize=8)
        axes[0].legend(fontsize=7)
    figure('E9','All E6 proxies, paired held-out moment warm-up',draw)
    g.info('Warm-up improves a proxy only on the reported metrics/panels. No held-out fitting or implied sufficiency; neutral and adverse outcomes retained.')
    return g.finish()
