"""C1--C6 checks. No legacy helper or frozen proxy is changed by this module.

Conventions: gaps are P minus Q; the parameter increment has a MINUS sign.
Curvature estimates are fixed before comparing a gap with its bound. They are
empirical directional Hessian searches, never certificates of a supremum.
"""
from __future__ import annotations

from common import *
import common as C
from experiments import weighted_result, provider_menu, proxy_panel
from scipy.optimize import linprog


def otlr_figure(name,title,draw,neural=True):
    fig,ax=plt.subplots(figsize=(9,5));draw(fig,ax)
    fig.suptitle(title+('\ncontrolled neural, not Gemma' if neural else '\nfinite-vector Monte Carlo'),fontsize=11)
    fig.tight_layout();directory=C.OUT/'figs';directory.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(directory/f'{name}.{suffix}',dpi=150)
    plt.close(fig)


def line_polynomial(poly, x, h):
    """Exact coefficients of V(x+t h), ascending in t."""
    out = [F(0)] * (1 + max(map(sum, poly)))
    for alpha, coefficient in poly.items():
        term = [coefficient]
        for a, xi, hi in zip(alpha, x, h):
            factor = [F(math.comb(a, j))*xi**(a-j)*hi**j for j in range(a+1)]
            product = [F(0)]*(len(term)+a)
            for i, u in enumerate(term):
                for j, v in enumerate(factor):
                    product[i+j] += u*v
            term = product
        for j, v in enumerate(term):
            out[j] += v
    return out


def check_V8(quick=False):
    g = Group('V8', 'C1 finite Taylor identities and C3 minimal first-order statistics',
              'Fraction polynomial arithmetic; certified logistic Hessians; exact ranks and LP witnesses', 'mixed')
    polynomials = []
    for D in range(2, 6):
        for rep in range(4 if quick else 35):
            degree = 1+rep % 4
            alphas = [a for a in it.product(range(degree+1), repeat=D) if sum(a) <= degree]
            poly = {a: F(int(g.rng.integers(-3, 4)), 3) for a in alphas}
            poly[tuple([degree]+[0]*(D-1))] = F(1)
            x = rational_array(g.rng, (D,), -2, 3, 3)
            increments = rational_array(g.rng, (6, D), -3, 4, 4)
            P, Q = distribution(g.rng, 6), distribution(g.rng, 6)
            coeff = np.array([line_polynomial(poly, x, h) for h in increments], object)
            delta = (P-Q) @ coeff
            gap = sum(delta)
            g.eq(gap, sum(delta[1:]), 'degree-order expansion is exact in Fraction arithmetic')
            radii = np.max(abs(np.r_[x[None], x+increments]), axis=0)
            absolute_poly = {a: abs(c) for a, c in poly.items()}
            # For ||h||_infinity<=1, coefficient sums on [-b_i,b_i]
            # bound every directional derivative. This is rational, not a fit.
            majorant = line_polynomial(absolute_poly, radii, [F(1)]*D)
            bounds = []
            for k in range(degree):
                j = k+1
                M = math.factorial(j)*majorant[j]
                moment = sum((P+Q)*np.array([max(abs(h))**j for h in increments], object))
                bound = M*moment/math.factorial(j)
                residual = abs(gap-sum(delta[1:k+1]))
                g.le(residual, bound, 'coefficient-box polynomial remainder', dict(D=D, degree=degree, k=k))
                bounds.append(dict(k=k, M=M, bound=bound, remainder=residual))
            polynomials.append(dict(D=D, degree=degree, x=x, increments=increments, P=P, Q=Q,
                                    polynomial=[dict(alpha=a, coefficient=c) for a,c in poly.items()],
                                    gap=gap, expansion=delta[1:], box_radii=radii, bounds=bounds))
            g.count()

    def logistic_case(vector):
        D = 3
        A = vector[:9].reshape(D,D); A = (A+A.T)/2
        b = vector[9:12]; x = vector[12:15]
        h = vector[15:33].reshape(6,D)
        P = expit(vector[33:39]); P /= P.sum()
        Q = expit(vector[39:45]); Q /= Q.sum()
        vertices = np.r_[x[None], x+h]
        q = np.einsum('ni,ij,nj->n', vertices, A, vertices)/2 + vertices@b
        v = expit(q)
        gradient = v[0]*(1-v[0])*(A@x+b)
        # ||H sigma(q)|| <= max|sigma''| max_hull||A z+b||²
        #                       + max sigma' ||A||op.
        # Norm of an affine map is convex, so vertices suffice for its maximum.
        L = np.max(np.linalg.norm(vertices@A+b,axis=1))
        M = L*L/(6*math.sqrt(3)) + np.linalg.norm(A,2)/4
        gap = float((P-Q)@v[1:]); first = float(gradient@((P-Q)@h))
        bound = float(M/2*((P+Q)@np.sum(h*h,axis=1)))
        ratio = abs(gap-first)/bound if bound else 0.
        g.le(ratio, 1+1e-12, 'analytic logistic-composition Hessian certificate', vector)
        g.count()
        return ratio, dict(vector=vector, gap=gap, first_order=first, bound=bound, M2=M, ratio=ratio)

    best = None
    for _ in range(80 if quick else 2200):
        ratio, record = logistic_case(g.rng.normal(size=45))
        if best is None or ratio>best['ratio']: best=record
    adversarial = search(g, lambda:g.rng.normal(size=45),
        lambda v:v+g.rng.normal(size=45)*.15, logistic_case,
        restarts=2 if quick else 12, steps=8 if quick else 100)
    if adversarial['best']['score']>best['ratio']: best=adversarial['best']['instance']

    ranks=[]
    for _ in range(12 if quick else 160):
        m=int(g.rng.integers(4,10)); D=int(g.rng.integers(2,6)); u=int(g.rng.integers(1,D+2))
        increments=sp.Matrix(g.rng.integers(-3,4,(m,D)).tolist())
        directions=sp.Matrix(g.rng.integers(-2,3,(D,u)).tolist())
        responses=increments*directions
        joined=sp.ones(m,1).row_join(responses)
        d0=joined.rank()-1
        centered=responses-sp.ones(m,1)*responses[0,:]
        g.eq(d0, centered.rank(), 'exact quotient rank equals centered response rank')
        g.le(d0,min(directions.rank(),m-1),'C3 dimension ceiling')
        basis=sp.Matrix.hstack(*directions.columnspace()) if directions.rank() else sp.zeros(D,0)
        projector=basis*(basis.T*basis).inv()*basis.T if basis.cols else sp.zeros(D,D)
        null=joined.T.nullspace()
        if null:
            perturb=null[0]; w=increments.T*perturb
            g.eq((projector*w).norm(),0,'feature match has zero U projection')
            g.eq((responses.T*perturb).norm(),0,'every first-order term vanishes')
        witness=None
        if d0:
            columns=joined[:,list(joined.rref()[1])]
            features=columns[:,:-1].T
            missed=columns[:,-1]
            A=np.array(features.tolist(),float)
            eq=np.r_[np.c_[A,-A],np.r_[np.ones(m),np.zeros(m)][None,:]]
            rhs=np.r_[np.zeros(len(A)),1.]
            target=np.array(missed,float).ravel()
            lp=linprog(np.r_[-target,target],A_eq=eq,b_eq=rhs,bounds=(0,None),method='highs')
            g.check(lp.success,'fewer-statistics LP is feasible')
            if lp.success:
                P,Q=lp.x[:m],lp.x[m:]; difference=P-Q
                g.le(np.max(abs(eq@lp.x-rhs)),1e-8,'LP matches retained features')
                terms=difference@np.array(responses,float)
                w=difference@np.array(increments,float)
                projected=np.array(projector,float)@w
                g.check(np.linalg.norm(projected)>1e-8 and np.max(abs(terms))>1e-8,
                        'nonzero projection has an original-family witnessing direction')
                witness=dict(P=P,Q=Q,retained_statistics=d0-1,features=features,
                             terms=terms,projection=projected,LP_objective=-lp.fun)
        ranks.append(dict(increments=increments,directions=directions,d0=d0,
                          dim_U=directions.rank(),matched_null=null[:1],LP_witness=witness))
        g.count()
    g.row['data']=dict(polynomials=polynomials,polynomial_norm='infinity',
        box_rule='M_j=j! [t^j] sum_alpha |c_alpha| product_i (b_i+t)^alpha_i; b_i=max over x and x+support |coordinate i|. Exact rational coefficient-sum upper bound, not the exact supremum.',
        logistic=dict(random_instances=80 if quick else 2200,best=best,adversarial=adversarial,
                      certificate='M2=L²/(6 sqrt(3))+||A||op/4; L=max_vertices ||Az+b||2 for q=z^T A z/2+b^T z'),
        ranks=ranks)
    return g.finish()


def isotropic_sums(rng,p,n,trials):
    """Exact rotational recursion for clipped N(0,I/p), without p-vectors.

    Given a partial sum of norm s, the independent next direction has cosine
    2 Beta((p-1)/2,(p-1)/2)-1. The two independent sums' difference has the
    same law as a sum of 2n symmetric vectors. Capture both n and 2n.
    """
    square=np.zeros(trials); one=None
    for i in range(2*n):
        radius=np.minimum(np.sqrt(rng.chisquare(p,trials)/p),1.)
        cosine=rng.choice([-1.,1.],trials) if p==1 else 2*rng.beta((p-1)/2,(p-1)/2,trials)-1
        square=np.maximum(0,square+radius**2+2*np.sqrt(square)*radius*cosine)
        if i==n-1: one=np.sqrt(square)/n
    return one,np.sqrt(square)/n


def spike_norm(indices,signs):
    order=np.argsort(indices,axis=1)
    ix=np.take_along_axis(indices,order,axis=1)
    sg=np.take_along_axis(signs,order,axis=1)
    total=np.zeros(len(ix)); running=sg[:,0].copy()
    for j in range(1,ix.shape[1]):
        same=ix[:,j]==ix[:,j-1]
        total+=np.where(same,0,running**2)
        running=np.where(same,running+sg[:,j],sg[:,j])
    return np.sqrt(total+running**2)


def check_V9(quick=False):
    g=Group('V9','C5 dimension-free one- and two-sample concentration',
            'IID trials; exact implicit high-dimensional sampling; 4-SE binomial margin','monte_carlo')
    trials=800 if quick else 20000; rows=[]
    for p,n,law in it.product((1,10,1000,100000),(8,32,128),('sign','clipped_Gaussian','sparse_spikes')):
        if law=='sign':
            # X=eta * sign_vector * Z; eta=1/sqrt(p), ||X||=1.
            a=(2*g.rng.binomial(n,.65,trials)-n)/n
            b=(2*g.rng.binomial(n,.4,trials)-n)/n
            one=abs(a-.3); two=abs(a-b-.5)
        elif law=='clipped_Gaussian':
            one,two=isotropic_sums(g.rng,p,n,trials)
        else:
            one=[];two=[]
            for start in range(0,trials,1000):
                count=min(1000,trials-start)
                indices=g.rng.integers(p,size=(count,2*n)); signs=g.rng.choice([-1.,1.],(count,2*n))
                one.extend(spike_norm(indices[:,:n],signs[:,:n])/n)
                two.extend(spike_norm(indices,signs)/n)
            one=np.array(one);two=np.array(two)
        for delta in (.05,.2):
            for sample,errors in (('one',one),('two',two)):
                bound=(1+math.sqrt(2*math.log((1 if sample=='one' else 2)/delta)))/math.sqrt(n)
                if sample=='two': bound*=2
                failures=int(np.sum(errors>bound)); rate=failures/trials
                margin=4*math.sqrt(delta*(1-delta)/trials)
                quantile=float(np.quantile(errors,1-delta))
                row=dict(p=p,n=n,law=law,delta=delta,sample=sample,trials=trials,b=1.,
                         failures=failures,failure_rate=rate,coverage=1-rate,bound=bound,
                         quantile=quantile,quantile_over_bound=quantile/bound,four_SE_margin=margin)
                g.le(rate,delta+margin,'C5 failure rate within delta plus 4 SE',row)
                rows.append(row);g.count(trials)
    def draw(fig,ax):
        fig.clear(); axes=fig.subplots(1,2)
        for law in ('sign','clipped_Gaussian','sparse_spikes'):
            for sample,style in (('one','-'),('two','--')):
                chosen=[r for r in rows if r['law']==law and r['sample']==sample and r['n']==32 and r['delta']==.05]
                axes[0].semilogx([r['p'] for r in chosen],[r['coverage'] for r in chosen],style+'o',label=law+'/'+sample)
                axes[1].semilogx([r['p'] for r in chosen],[r['quantile_over_bound'] for r in chosen],style+'o')
        axes[0].axhline(.95,color='gray',ls=':');axes[0].set(xlabel='ambient dimension p',ylabel='coverage',ylim=(.94,1.01))
        axes[1].set(xlabel='ambient dimension p',ylabel='95% quantile / bound',ylim=(0,1))
        axes[0].legend(fontsize=6)
    otlr_figure('V9','C5 bounded-vector concentration (n=32, delta=.05)',draw,neural=False)
    g.row['data']=dict(rows=rows,trials_per_cell=trials,
        laws='sign: biased scalar sign times fixed unit sign vector (eta=1/sqrt(p)), different P/Q means; Gaussian: full-dimensional N(0,I/p) clipped at norm 1; spikes: independent uniform coordinate with fair sign, norm 1. Gaussian and spikes have mean zero.',
        implicit_sampling='Gaussian resultant lengths use exact isotropy/Beta-angle recursion, not a Gaussian approximation. Sparse collisions are explicitly counted. Two samples are independent, though the one/two reported statistics can share draws.',
        max_quantile_over_bound=max(r['quantile_over_bound'] for r in rows),
        caveat='Dimension-free at fixed b; unnormalized sign updates have b=eta sqrt(p). No dimension-free claim for fixed coordinate eta.')
    return g.finish()


OPTIMIZERS = ('SGD','heavy_ball','Nesterov','Adam_cold','Adam_warm5',
              'Adam_warm20','AdamW','clipped_SGD','clipped_Adam','signSGD','Lion')
ETAS = (.001,.003,.01,.03,.1)


def state_vector(state):
    return np.c_[state['theta'],state['m'],state['v']]


def otlr_step(x,gradient,step,eta,opt,clip=1.,eps=1e-8):
    """Complete state, including moments; inherited SGD/Adam maps are unchanged."""
    theta,m,v=np.split(x,3,axis=1)
    gradient=np.asarray(gradient)
    if opt.startswith('clipped_'):
        gradient=gradient*np.minimum(1,clip/np.maximum(np.linalg.norm(gradient,axis=1),1e-300))[:,None]
    if opt in ('SGD','clipped_SGD'):
        a,b,c,_=optimizer(theta,gradient,m,v,step,eta=eta,opt='SGD')
    elif opt in ('heavy_ball','Nesterov'):
        b=.9*m+gradient;c=v.copy()
        a=theta-eta*(b if opt=='heavy_ball' else gradient+.9*b)
    elif opt=='signSGD':
        a=theta-eta*np.sign(gradient);b=m.copy();c=v.copy()
    elif opt=='Lion':
        a=theta-eta*np.sign(.9*m+.1*gradient);b=.99*m+.01*gradient;c=v.copy()
    else:
        with np.errstate(invalid='ignore',divide='ignore'):
            a,b,c,_=optimizer(theta,gradient,m,v,step,eta=eta,eps=eps,
                             decay=.01 if opt=='AdamW' else 0.)
    return np.c_[a,b,c]


def gradient_at(x,labels,k):
    if k==0:
        feature,target=labels[:,0],labels[:,1]
        return ((expit(x[:,0]*feature)-target)*feature)[:,None]
    return flatten(neural_gradient(unflatten(x[:,:4*k+9],k),labels))


def advance(x,labels,k,step,eta,opt,clip=1.,eps=1e-8):
    return otlr_step(x,gradient_at(x,labels,k),step,eta,opt,clip,eps)


def terminal(x,k,task,N):
    if k==0:return expit(x[:,0]+.4)
    return task_value(neural_forward(unflatten(x[:,:4*k+9],k))[0],task,N)


def complete_sequences(points,law,R):
    indices=np.array(list(it.product(np.flatnonzero(law),repeat=R)),int).reshape(-1,R) if R else np.zeros((1,0),int)
    return points[indices],np.prod(law[indices],axis=1)


def continuation(x,labels,weights,k,step,eta,opt,clip,task,N,eps=1e-8,pairwise=False):
    """Finite P tail, or explicitly marked per-replica IID P tails."""
    if pairwise:
        if len(x)!=len(labels):
            if len(x)%len(labels):raise ValueError('pairwise tail batch must tile whole replica sets')
            labels=np.tile(labels,(len(x)//len(labels),1,1))
        value=x.copy()
        for j in range(labels.shape[1]):
            value=advance(value,labels[:,j],k,step+j+1,eta,opt,clip,eps)
        return terminal(value,k,task,N)
    count=len(labels); outputs=[]
    # Batching leaves all histories and exact weights intact.
    for start in range(0,len(x),max(1,8192//count)):
        batch=x[start:start+max(1,8192//count)]
        value=np.repeat(batch,count,axis=0)
        lab=np.tile(labels,(len(batch),1,1))
        for j in range(labels.shape[1]):
            value=advance(value,lab[:,j],k,step+j+1,eta,opt,clip,eps)
        outputs.extend(terminal(value,k,task,N).reshape(len(batch),count)@weights)
    return np.asarray(outputs)


def richardson(fun,x,d,h=.002):
    """Full-state directional gradient contraction, without projecting out moments.

    Central differences at h and h/2, fourth-order Richardson extrapolation.
    The reported error is |fine-coarse|/3 (a differentiation diagnostic, not a
    rigorous bound). Nonfinite probes remain nonfinite; no v clamping.
    """
    fp,fm,hp,hm=fun(np.concatenate((x+h*d,x-h*d,x+h*d/2,x-h*d/2))).reshape(4,len(x))
    coarse=(fp-fm)/(2*h)
    fine=(hp-hm)/h
    return (4*fine-coarse)/3,abs(fine-coarse)/3


def increment_panel(x,points,k,step,eta,opt,clip,eps=1e-8):
    repeated=np.repeat(x,len(points),axis=0)
    labels=np.tile(points,(len(x),1))
    raw=gradient_at(repeated,labels,k)
    increments=otlr_step(repeated,raw,step,eta,opt,clip,eps)-repeated
    return increments.reshape(len(x),len(points),-1),raw.reshape(len(x),len(points),-1)


def curvature_search(fun,x,rays,scale,quick=False):
    """Empirical M2 on actual increment rays; no outcome-dependent inflation.

    Taylor only uses D²V(x+u Delta)[Delta,Delta]. We maximize its normalized
    absolute value over sampled segments. This does not estimate all off-ray
    Hessian eigenvalues or certify unobserved points. Invalid probes are logged.
    """
    rays=rays.reshape(-1,x.shape[1]); x=np.repeat(x,len(rays)//len(x),axis=0)
    norm2=np.sum((rays*scale)**2,axis=1)
    denominator=np.maximum(norm2,1e-300)
    def measure(u):
        h=.002
        center=x+u[:,None]*rays
        f0,fp,fm,hp,hm=fun(np.concatenate((center,center+h*rays,center-h*rays,
                                          center+h/2*rays,center-h/2*rays))).reshape(5,len(center))
        coarse=(fp-2*f0+fm)/h**2
        half=(hp-2*f0+hm)/(h/2)**2
        value=abs((4*half-coarse)/3)/denominator
        error=abs(half-coarse)/(3*denominator)
        return np.where(norm2>1e-28,value,0.),np.where(norm2>1e-28,error,0.)
    best=np.zeros(len(rays)); where=np.full(len(rays),.5); max_error=0.; invalid=0
    for u0 in (.005,.25,.5,.75,.995):
        u=np.full(len(rays),u0);values,errors=measure(u)
        valid=np.isfinite(values);invalid+=int(np.sum(~valid))
        select=valid & (values>best);best[select]=values[select];where[select]=u[select]
        if valid.any(): max_error=max(max_error,float(np.max(errors[valid])))
    radius=.125
    for _ in range(1 if quick else 4):
        for sign in (-1,1):
            u=np.clip(where+sign*radius,.0021,.9979); values,errors=measure(u)
            valid=np.isfinite(values);invalid+=int(np.sum(~valid))
            select=valid & (values>best);best[select]=values[select];where[select]=u[select]
        radius/=2
    return dict(M2=float(np.max(best,initial=0)),max_difference_error=max_error,
                invalid_probes=invalid,n_rays=len(rays),maximizer_u=float(where[np.argmax(best)]) if len(best) else None)


def naive_direction(x,raw,contrast,eta,opt,step,eps):
    p=raw.shape[-1]; out=np.zeros_like(x); dg=np.einsum('m,nmp->np',contrast,raw)
    if 'Adam' in opt:
        # Deliberately naive frozen CURRENT denominator, no state-block terms.
        v=x[:,2*p:]; denom=np.sqrt(v/max(1-.999**max(step-1,1),1e-300))+eps
        out[:,:p]=-eta*dg/denom
    else:
        out[:,:p]=-eta*dg
    return out


def response_cell(g,start,points,P,Q,task,N,opt,eta,R,clip,quick,linear=False):
    k=0 if linear else points.shape[1]; p=1 if linear else 4*k+9; x0=state_vector(start); eps=1e-8
    sources=[]; replicas=48 if quick else 2000
    for law in (P,Q):
        labels,weights,mode=training_sequences(g.rng,points,law,R,replicas,quick)
        if not linear and opt in ('SGD','Adam_cold','Adam_warm5','Adam_warm20','clipped_SGD','clipped_Adam'):
            actual=run_network(labels,start=start,eta=eta,opt='SGD' if opt in ('SGD','clipped_SGD') else 'Adam',
                               clip=clip if opt.startswith('clipped_') else None)
            probabilities=actual['probabilities']
        else:
            x=np.repeat(x0,len(labels),axis=0)
            for t in range(R): x=advance(x,labels[:,t],k,start['step']+t+1,eta,opt,clip)
            probabilities=expit(x[:,:1]+.4) if linear else neural_forward(unflatten(x[:,:p],k))[0]
        summary=weighted_result(probabilities,weights,'moment' if linear else task,N,mode)
        sources.append({a:summary[a] for a in ('mean','SE','mode','n_histories')})
    gap=sources[0]['mean']-sources[1]['mean']; gap_se=math.hypot(sources[0]['SE'],sources[1]['SE'])
    scale=np.r_[np.ones(p),np.full(2*p,eta)]
    total=0.; naive=0.; bound=0.; differr=0.; parts=[]; eligible=True
    if R<=4:
        x=x0.copy(); state_weights=np.ones(1)
        boundary_states=[]
        for t in range(R):
            boundary_states.append((x,state_weights))
            support=np.flatnonzero(Q)
            x=advance(np.repeat(x,len(support),axis=0),np.tile(points[support],(len(x),1)),k,
                      start['step']+t+1,eta,opt,clip)
            state_weights=(state_weights[:,None]*Q[support]).ravel()
    else:
        # Unbiased random-time estimator of the C2 sum: R times an IID boundary
        # chosen uniformly from each independent Q run. Avoids 4**39 histories.
        times=g.rng.integers(R,size=replicas)
        qlabels=points[g.rng.choice(len(points),(replicas,R),p=Q)]
        tails=points[g.rng.choice(len(points),(replicas,R),p=P)]
        x=np.repeat(x0,replicas,axis=0); chosen=np.empty_like(x)
        for t in range(R):
            chosen[times==t]=x[times==t]
            x=advance(x,qlabels[:,t],k,start['step']+t+1,eta,opt,clip)
        first_samples=np.zeros(replicas); naive_samples=np.zeros(replicas); moment_samples=np.zeros(replicas)
        observed_times=np.unique(times)
        curvature_times=set(observed_times[np.linspace(0,len(observed_times)-1,
                            min(len(observed_times),2 if quick else 4),dtype=int)])
        global_M=0.
    for t in range(R):
        if R<=4:
            x,weights=boundary_states[t]
            labels,tail_weights=complete_sequences(points,P,R-t-1)
            fun=lambda a:continuation(a,labels,tail_weights,k,start['step']+t+1,eta,opt,clip,task,N)
        else:
            ids=np.flatnonzero(times==t)
            if not len(ids):continue
            x=chosen[ids];weights=np.full(len(ids),R/replicas)
            labels=tails[ids,t+1:];tail_weights=None
            fun=lambda a:continuation(a,labels,None,k,start['step']+t+1,eta,opt,clip,task,N,pairwise=True)
        inc,raw=increment_panel(x,points,k,start['step']+t+1,eta,opt,clip)
        direction=np.einsum('m,nmd->nd',P-Q,inc)
        score,error=richardson(fun,x,direction)
        # Naive scores are computed by theta-only differentiation, so huge
        # cold denominators cannot perturb parameters by order 1e8.
        rawdir=np.zeros_like(x); rawdir[:,:p]=np.einsum('m,nmp->np',P-Q,raw)
        ndir=naive_direction(x,raw,P-Q,eta,opt,start['step']+t+1,eps)
        size=np.maximum(1,np.linalg.norm(ndir,axis=1)); ns,ne=richardson(fun,x,ndir/size[:,None]);ns*=size
        finite=np.isfinite(score)&np.isfinite(error)&np.isfinite(ns)
        if not finite.all():eligible=False
        # Curvature at predeclared sampled Q states, all supported increment rays.
        selected=np.linspace(0,len(x)-1,min(len(x),2 if quick else 4),dtype=int)
        supported=np.flatnonzero(P+Q)
        cx=x[selected]; rays=inc[selected][:,supported]
        if R<=4:
            cfun=fun
        else:
            clabels=np.repeat(labels[selected],len(supported),axis=0)
            cfun=lambda a:continuation(a,clabels,None,k,start['step']+t+1,eta,opt,clip,task,N,pairwise=True)
        curvature=curvature_search(cfun,cx,rays,scale,quick) if R<=4 or t in curvature_times else None
        moment=np.einsum('m,nm->n',P+Q,np.sum((inc*scale)**2,axis=2))
        if R<=4:
            perbound=curvature['M2']*moment/2
            total+=float(weights@score);naive+=float(weights@ns);bound+=float(weights@perbound)
        else:
            first_samples[ids]=R*score;naive_samples[ids]=R*ns;moment_samples[ids]=moment
            if curvature is not None:global_M=max(global_M,curvature['M2'])
        differr+=float(weights@error)
        parts.append(dict(t=t,Q_states=len(x),exact_P_suffix_histories=len(labels) if R<=4 else None,
                          finite_derivative=bool(finite.all()),curvature=curvature))
    if R>4:
        bound_samples=R*global_M*moment_samples/2
        total=float(np.mean(first_samples));naive=float(np.mean(naive_samples));bound=float(np.mean(bound_samples))
        first_se=float(np.std(first_samples,ddof=1)/math.sqrt(replicas))
        bound_se=float(np.std(bound_samples,ddof=1)/math.sqrt(replicas))
    else:first_se=bound_se=0.
    residual=abs(gap-total); ratio=residual/bound if bound>0 else (0. if residual<1e-13 else math.inf)
    residual_se=math.hypot(gap_se,first_se)
    row=dict(task=task,N=N,optimizer=opt,eta=eta,R=R,gap=gap,gap_SE=gap_se,
        model='direct gradient-law witness realized by linear-logit CE examples (feature,target)' if linear else 'controlled neural, not Gemma; original C23 network',
        first_order=total,first_order_SE=first_se,naive_first_order=naive,
        differentiation_error_estimate=differr,remainder=residual,bound=bound,bound_SE=bound_se,
        ratio=ratio,finite_central_derivatives=eligible,sources=sources,parts=parts,
        residual_SE=residual_se,
        excess_over_bound_after_four_SE=max(0.,residual-bound-4*residual_se) if eligible else None,
        bound_status=('UNAVAILABLE' if not eligible else 'PASS' if ratio<=1 else 'FAIL'),
        first_order_mode='complete Q histories and exact P continuation; full-state directional Richardson' if R<=4 else
            '2000 independent random-time Q boundaries and IID P tails; unbiased finite-difference C2-sum estimator (48 in quick)',
        smoothness='not certified; sign, clipping and cold zero-variance coordinates may violate S1')
    g.count();return row


def slope(rows,field):
    usable=[r for r in rows if r['eta']<=.01 and np.isfinite(r[field]) and abs(r[field])>1e-12]
    return float(np.polyfit(np.log([r['eta'] for r in usable]),np.log([abs(r[field]) for r in usable]),1)[0]) if len(usable)>=3 else None


def check_E10(quick=False):
    g=Group('E10','C2 complete-state linear response across eleven optimizer configurations',
            'Exact R=1,4 histories; independent R=40 runs; full-state Richardson contractions and unpadded segment curvature',
            'controlled_neural')
    rows=[];design=[]
    tasks=cases()+[('clipped_witness',1,[(8.,0.),(8/3,1.),(0.,.5)],[.25,.75,0.],[0.,0.,1.])]
    for task,N,pts,pp,qq in tasks:
        linear=task=='clipped_witness'
        points=np.array(pts,float);P=np.array(pp,float);Q=np.array(qq,float);k=0 if linear else points.shape[1]
        initial=dict(theta=np.zeros((1,1)),m=np.zeros((1,1)),v=np.zeros((1,1)),step=0,k=0) if linear else snapshot(k)
        prefix=points[g.rng.choice(len(points),20,p=P)]
        starts={0:initial}
        for W in (5,20):
            state={a:(b.copy() if isinstance(b,np.ndarray) else b) for a,b in initial.items()}
            for t,labels in enumerate(prefix[:W]):
                old=state['theta'].tobytes(); grad=gradient_at(state_vector(state),labels[None],k)
                state['theta'],state['m'],state['v'],_=optimizer(state['theta'],grad,state['m'],state['v'],t+1,eta=0.)
                g.check(old==state['theta'].tobytes(),'E10 lr=0 moment warm-up preserves theta bytes')
            state['step']=W;starts[W]=state
        raw=gradient_at(np.repeat(state_vector(initial),len(points),axis=0),points,k)
        norms=np.linalg.norm(raw,axis=1); supported=(P+Q)>0
        # Merge roundoff-level ties before selecting a global norm threshold.
        # Splitting mathematically equal C23 norms by a few ulps is not a
        # meaningful way to satisfy the requested clipping frequency.
        unique=[]
        for value in np.sort(norms[supported]):
            if not unique or value-unique[-1]>1e-12*max(1.,abs(value)):unique.append(float(value))
        unique=np.array(unique);candidates=(unique[:-1]+unique[1:])/2
        # Equal weight on the two laws; record both mass and alphabet fractions.
        fractions=np.array([np.dot((P+Q)/2,norms>c) for c in candidates])
        index=int(np.argmin(abs(fractions-.5))) if len(candidates) else None
        clip=1. if linear else float(candidates[index]) if index is not None else float(unique[0])
        active=float((P+Q)/2@(norms>clip))
        design.append(dict(task=task,N=N,points=points,P=P,Q=Q,common_prefix=prefix,
            prefix_law='P, the same declared IID common-prefix law as E8; E10 independent seed',
            clip=clip,clip_active_probability=active,clip_active_message_fraction=float(np.mean(norms[supported]>clip)),
            requested_clip_band_achievable=bool(.3<=active<=.7),initial_norms=norms,
            clip_rule='closest to 50% mass under (P+Q)/2; norms within 1e-12 absolute/relative are treated as ties; both probability and uniform-alphabet fraction recorded',
            distinct_norm_clusters=unique))
        for opt in OPTIMIZERS:
            start=starts[5 if opt=='Adam_warm5' else 20 if opt=='Adam_warm20' else 0]
            for eta,R in it.product(ETAS,(1,4,40)):
                rows.append(response_cell(g,start,points,P,Q,task,N,opt,eta,R,clip,quick,linear))
            print(f'E10 {task}{N} {opt} completed',flush=True)
    summary=[]
    for opt in OPTIMIZERS:
        subset=[r for r in rows if r['optimizer']==opt]
        valid=[r for r in subset if r['finite_central_derivatives'] and np.isfinite(r['first_order'])]
        slopes=[]
        for task,N,_,_,_ in tasks:
            for R in (1,4):
                cell=[r for r in subset if r['task']==task and r['N']==N and r['R']==R]
                slopes.append(dict(task=task,N=N,R=R,remainder_slope=slope(cell,'remainder'),first_order_slope=slope(cell,'first_order')))
        signed=[r for r in valid if abs(r['gap'])>1e-10]
        exact=[r for r in valid if r['R']<=4]
        summary.append(dict(optimizer=opt,max_ratio=max((r['ratio'] for r in valid),default=None),
            max_exact_ratio=max((r['ratio'] for r in exact),default=None),
            status='FAIL' if any(r['bound_status']=='FAIL' for r in subset) else 'UNAVAILABLE' if any(not r['finite_central_derivatives'] for r in subset) else 'PASS',
            exact_bound_failures=sum(r['R']<=4 and r['bound_status']=='FAIL' for r in subset),
            MC_bound_failures=sum(r['R']==40 and r['bound_status']=='FAIL' for r in subset),
            MC_exceedances_after_four_SE=sum(r['R']==40 and r['finite_central_derivatives'] and r['excess_over_bound_after_four_SE']>0 for r in subset),
            bound_failures=sum(r['bound_status']=='FAIL' for r in subset),unavailable=sum(not r['finite_central_derivatives'] for r in subset),
            sign_agreement=float(np.mean([r['gap']*r['first_order']>0 for r in signed])) if signed else None,slopes=slopes))
    c23=[r for r in rows if r['optimizer']=='Adam_cold' and r['task']=='moment' and r['N']==3 and r['eta']==.1 and r['R']==1][0]
    c23['catalog_displayed_plus_sign_prediction']=-c23['first_order']
    c23['sign_note']='C4(d) displayed Prediction has a plus eta multiplying grad(V) dot delta h; the actual increment is -eta h, so C1/C2 give the opposite sign. The increment-based prediction here uses the correct minus sign.'
    g.eq(c23['gap'],.076907458, 'cold Adam C23 gap reproduction',tol=5e-10)
    def draw(fig,ax):
        fig.clear(); axes=fig.subplots(3,4).ravel();fig.set_size_inches(15,10)
        for ax,opt in zip(axes,OPTIMIZERS):
            selected=[r for r in rows if r['optimizer']==opt and r['task']=='moment' and r['N']==3 and r['R']==1]
            x=[r['eta'] for r in selected]
            for field,label in (('remainder','|gap - transformed|'),('first_order','|transformed|'),('naive_first_order','|naive|')):
                ax.loglog(x,[max(abs(r[field]),1e-16) for r in selected],'o-',label=label)
            ax.loglog(x,np.array(x)**2,':',color='gray',label='eta^2 guide')
            ax.loglog(x,np.array(x)*.1,'--',color='gray',label='eta guide');ax.set_title(opt,fontsize=9)
        axes[-1].axis('off');axes[0].legend(fontsize=5)
    otlr_figure('E10','C2 moment N=3, R=1; full matrix and slopes in JSON',draw)
    raw=write_json('raw/E10_cells.json',rows)
    g.row['data']=dict(cells=rows,design=design,optimizer_summary=summary,C23_prediction=c23,raw=raw,
        label='controlled neural, not Gemma',
        limitations=['R=40 exact continuation enumeration is infeasible (up to 4^39 tails per boundary); reported C2 scores are Monte Carlo with SE, not exact gradients or exact gaps.',
            'C2 smoothness is not assumed for sign/clipping kinks, cold Adam zero-gradient coordinates, or a threshold utility. Nonfinite central probes are retained as UNAVAILABLE.',
            'Mhat2 is a sampled/hill-climbed directional segment constant, never padded after observing a gap. Every ratio >1 is a recorded cell FAIL, so E10 cannot receive PASS.',
            'A scalar norm threshold cannot split tied gradient norms; unattainable 30--70% clipping bands are reported, not jittered.',
            'The C4(e) 4/-4/3/0 witness cannot be the original fixed-network label law; it is included across all optimizers/rates/horizons as linear-logit CE with (feature,target)=(8,0),(8/3,1),(0,1/2), plus the direct E11(a) control.'])
    g.info('Empirical response diagnostic. Cell FAIL/UNAVAILABLE records and nonquadratic slopes are scientific negative results; no C2 certificate is claimed.')
    return g.finish()


def full_gradient(fun,x,relative=None):
    """Richardson gradient on every complete-state coordinate."""
    p=x.shape[1]; directions=np.eye(p)
    if relative is not None: directions*=relative[:,None]
    repeated=np.repeat(x,p,axis=0)
    scores,errors=richardson(fun,repeated,directions,h=1e-4)
    if relative is not None:scores/=relative;errors/=relative
    return scores,errors


def check_E11(quick=False):
    g=Group('E11','C4 clipping, warm Adam, Lion and LoRA witnesses',
            'Executed maps, actual C23 continuations, linear-logit controls, bilinear sigmoid LoRA', 'controlled_neural')
    etas=np.array(ETAS); panels={}
    # Direct gradient laws on a linear-logit model. The gradients 4 and -4/3
    # cannot occur in the fixed C23 one-context CE network at this initialization.
    P=np.array([.25,.75,0]);Q=np.array([0.,0.,1.]); gradients=np.array([[4.],[-4/3],[0.]])
    initial=np.array([[.4,0.,0.]])
    values=[]
    for eta in etas:
        updated=otlr_step(np.repeat(initial,3,axis=0),gradients,1,eta,'clipped_SGD',1.)
        gap=float((P-Q)@expit(updated[:,0])); derivative=expit(.4)*(1-expit(.4))
        first=eta*derivative/2
        g.eq(float((P-Q)@gradients[:,0]),0,'C4(e) raw means agree')
        g.eq(float((P-Q)@np.clip(gradients[:,0],-1,1)),-.5,'C4(e) clipped means differ')
        g.le(abs(gap-first),eta**2/(12*math.sqrt(3)),'linear-logit analytic second-order certificate')
        values.append(dict(eta=float(eta),gap=gap,first_order=first,remainder=abs(gap-first)))
        g.count()
    panels['clipped_mean']=dict(rows=values,model='direct gradient-law experiment on sigmoid linear logit theta=.4',
        support=gradients,P=P,Q=Q,remainder_slope=slope(values,'remainder'))

    # Different direction laws, both mean unit direction zero. Their second
    # direction moments differ; changing magnitudes alone would give gap=0.
    gradients=np.array([[2.,0],[-2.,0],[0,4.],[0,-4.]])
    P=np.array([.5,.5,0,0]);Q=np.array([0,0,.5,.5]);x=np.zeros((4,6))
    values=[]
    for eta in etas:
        updated=otlr_step(x,gradients,1,eta,'clipped_SGD',1.)
        gap=float((P-Q)@expit(.4+updated[:,:2]@np.array([1.,2.])))
        first=float((P-Q)@updated[:,:2]@np.array([1.,2.]))*expit(.4)*(1-expit(.4))
        g.eq(first,0,'always-active clipping equal mean unit direction')
        values.append(dict(eta=float(eta),gap=gap,first_order=first,remainder=abs(gap)))
        g.count()
    fitted=slope(values,'remainder');g.le(abs(fitted-2),.01,'always-active witness has quadratic slope')
    panels['always_active']=dict(rows=values,slope=fitted,support=gradients,
        note='Both means of unit direction are zero; different direction second moments yield nonzero eta^2 gap. Identical direction laws with only magnitude changes have exactly zero gap, not a measurable slope.')

    # Symmetric label laws at a warm m=0 state force theta/m mean contrasts to
    # vanish exactly by oddness; general equal-mean warm states do not do so.
    k=1;p=13;state=snapshot(k);state['m'][:]=0;state['v'][:]=.02;state['step']=20
    points=np.array([[0.],[1.],[.25],[.75]])
    P=np.array([.5,.5,0,0]);Q=np.array([0,0,.5,.5]);x=state_vector(state)
    eta=.03;inc,raw=increment_panel(x,points,k,21,eta,'Adam_cold',1.)
    contrast=(P-Q)@inc[0];delta1=(P-Q)@raw[0];delta2=(P-Q)@(raw[0]**2)
    tail=np.full((1,2,1),.25)
    fun=lambda a:continuation(a,tail,np.ones(1),1,21,eta,'Adam_cold',1.,'moment',3)
    grad,error=full_gradient(fun,x,relative=np.r_[np.ones(p),np.full(p,.02),np.full(p,.02)])
    pieces=[float(grad[j*p:(j+1)*p]@contrast[j*p:(j+1)*p]) for j in range(3)]
    expected=.001*float(grad[2*p:]@delta2)
    g.le(np.max(abs(delta1)),1e-15,'warm Adam symmetric equal gradient means')
    g.le(max(abs(pieces[0]),abs(pieces[1])),1e-12,'symmetric warm witness theta/m terms zero')
    g.eq(sum(pieces),expected,'warm Adam explicit v-block first order',tol=1e-12)
    Lv=float(np.sum(abs(grad[2*p:]*state['v'][0])))
    kappa=.001*raw[0]**2/(.999*state['v'])
    w1bound=.999*Lv*float((P+Q)@np.max(kappa,axis=1))
    g.le(abs(expected),w1bound+1e-15,'W1 relative-v Lipschitz kappa covers explicit linear term')
    nextvalues=fun(x+inc[0]);gap=float((P-Q)@nextvalues)
    panels['warm_Adam']=dict(state=state,points=points,P=P,Q=Q,gradient=grad,
        differentiation_error=error,delta1=delta1,delta2=delta2,first_order_blocks=pieces,
        predicted_v_term=expected,first_order=sum(pieces),measured_gap=gap,L_v=Lv,
        kappa=kappa,W1_Lv_kappa_bound=w1bound,
        caveat='Admissible warm state m=0, v=.02, t=20; not claimed as an IID warm-up draw. Symmetry is essential for the zero theta term. W1 bound here covers the explicit linear v contribution, not the entire nonlinear remainder.')
    g.count()

    lion=[]
    points=np.array([[0.],[1.]])
    for dominated in (True,False):
        state=snapshot(1);state['m'][:]=.2 if dominated else .005;x=state_vector(state)
        inc,raw=increment_panel(x,points,1,1,.03,'Lion',1.)
        delta=inc[0,0]-inc[0,1];delta1=raw[0,0]-raw[0,1]
        flips=(np.sign(.9*state['m']+.1*raw[0])!=np.sign(state['m'])).astype(float)
        flipzeros=(.9*state['m']+.1*raw[0])==0
        flips[flipzeros]=.5
        g.le(np.max(abs(delta[p:2*p]-.01*delta1)),1e-15,'Lion m-block carries .01 delta1')
        if dominated:
            g.eq(np.max(abs(delta[:p])),0,'momentum-dominated Lion parameter contrast exactly zero')
            g.eq(np.max(flips),0,'momentum-dominated Lion no flips')
        else:
            g.check(np.max(abs(delta[:p]))>0,'non-dominated Lion control has parameter contrast')
            g.check(np.max(flips)>0,'non-dominated nonzero-momentum Lion has flips')
        lion.append(dict(dominated=dominated,state=state,gradients=raw[0],increment_difference=delta,
                         flip_probability=flips,delta1=delta1,
                         note='The m increment carries delta1. For a locally fixed future sign pattern, its continuation derivative is zero; nonzero m memory does not imply a nonzero local first-order return.'))
        g.count()
    panels['Lion']=lion

    lora=[]
    for rank in (1,2):
        A=g.rng.normal(size=(rank,4));B=np.zeros((4,rank));W0=np.full((4,4),.2);alpha=1.
        projection=np.eye(4)-A.T@np.linalg.solve(A@A.T,A)
        null=g.rng.normal(size=(4,4))@projection
        H=g.rng.normal(size=(4,rank))@A
        magnitude=max(np.max(abs(null+H)),np.max(abs(null-H)),np.max(abs(2*H)))
        null*=.015/magnitude;H*=.015/magnitude
        laws={'null_P':[null+H,null-H],'null_Q':[np.zeros((4,4))],
              'range_P':[H],'range_Q':[np.zeros((4,4))],
              'sign_P':[H+null],'sign_Q':[2*H+null]}
        unmoved=[]
        # Each message contains all sixteen basis-entry observations. Mean CE
        # gradient at W0 is (sigmoid(W0)-labels)/16, exactly the declared Gamma.
        for name,gammas in laws.items():
            labels=[expit(W0)-16*Gamma for Gamma in gammas]
            g.check(all(np.min(y)>=0 and np.max(y)<=1 for y in labels),'LoRA gradients realized by legal soft-label mean CE')
            for opt in ('SGD','heavy_ball','Adam_cold'):
                for Gamma,y in zip(gammas,labels):
                    measured=(expit(W0)-y)/16
                    ga=alpha*B.T@measured;gb=alpha*measured@A.T
                    theta=np.r_[A.ravel(),B.ravel()][None]; grad0=np.r_[ga.ravel(),gb.ravel()][None]
                    updated=otlr_step(np.c_[theta,np.zeros((1,2*theta.shape[1]))],grad0,1,.03,opt)
                    unchanged=updated[0,:A.size].tobytes()==A.ravel().tobytes()
                    g.check(unchanged,'LoRA A bitwise fixed at B=0, cold step 1',dict(rank=rank,law=name,opt=opt))
                    unmoved.append(dict(law=name,optimizer=opt,unchanged=unchanged))
        def value_from_B(bb):return float(np.mean(expit(W0+alpha*bb@A)))
        vprime=expit(.2)*(1-expit(.2)); gradB=alpha*np.full((4,4),vprime/16)@A.T
        values=[]
        for eta in etas:
            def expected(name):
                return float(np.mean([value_from_B(-eta*alpha*Gamma@A.T) for Gamma in laws[name]]))
            gap=expected('null_P')-expected('null_Q')
            first=-eta*alpha*float(np.sum(gradB*(null@A.T)))
            rangefirst=-eta*alpha*float(np.sum(gradB*(H@A.T)))
            rangegap=expected('range_P')-expected('range_Q')
            g.le(abs(first),1e-15,'LoRA null sketch first-order gap zero')
            values.append(dict(eta=float(eta),gap=gap,first_order=first,remainder=abs(gap-first),range_first_order=rangefirst,range_gap=rangegap))
        signs=[np.sign(Gamma@A.T) for name in ('sign_P','sign_Q') for Gamma in laws[name]]
        g.check(np.array_equal(*signs),'LoRA cold-Adam identical sketch-sign laws')
        cold=[]
        for name in ('sign_P','sign_Q'):
            sketch=alpha*laws[name][0]@A.T
            theta=np.r_[A.ravel(),B.ravel()][None];gg=np.r_[np.zeros(A.size),sketch.ravel()][None]
            actual=otlr_step(np.c_[theta,np.zeros((1,2*theta.shape[1]))],gg,1,.03,'Adam_cold')[0,A.size:theta.size]
            predicted=-.03*sketch.ravel()/(abs(sketch.ravel())+1e-8)
            g.le(np.max(abs(actual-predicted)),1e-14,'cold Adam uses h_epsilon of LoRA sketch')
            cold.append(actual)
        # The analytic signal is known nonzero here; retain the small first
        # point (3e-13 for rank 1) and disclose its greater roundoff sensitivity.
        null_slope=float(np.polyfit(np.log(etas[:3]),np.log([r['remainder'] for r in values[:3]]),1)[0])
        g.le(abs(null_slope-2),.005,'LoRA null-sketch gap has quadratic small-rate slope')
        lora.append(dict(rank=rank,A=A,W0=W0,alpha=alpha,Gamma_null=null,Gamma_range=H,
            null_sketch_residual=float(np.max(abs(null@A.T))),rows=values,A_bitwise_audit=unmoved,
            null_gap_slope=null_slope,slope_rule='first three rates; no floor, analytically nonzero second-order witness; tiny values are roundoff-sensitive',cold_Adam_signs=signs,
            finite_epsilon_update_difference=float(np.max(abs(cold[0]-cold[1]))),
            note='Same sign law implies equality only as epsilon -> 0; finite-epsilon h_epsilon sketches are checked exactly. Null mean sketches do not match second moments.'))
        g.check(abs(values[-1]['range_first_order'])>1e-10,'LoRA range sketch gives nonzero first-order gap')
        g.count()
    panels['LoRA']=lora
    def draw(fig,ax):
        fig.clear();axes=fig.subplots(2,3).ravel();fig.set_size_inches(12,7)
        for ax,key in zip(axes[:2],('clipped_mean','always_active')):
            data=panels[key]['rows'];ax.loglog(etas,[abs(r['gap']) for r in data],'o-',label='gap')
            ax.loglog(etas,[max(abs(r['first_order']),1e-16) for r in data],'s--',label='first order')
            ax.loglog(etas,[r['remainder'] for r in data],':',label='remainder');ax.set_title(key);ax.legend(fontsize=7)
        axes[2].bar(['theta','m','v','W1 bound'],[abs(a) for a in pieces]+[w1bound]);axes[2].set_title('warm Adam blocks')
        axes[3].bar(['dominated','control'],[np.linalg.norm(r['increment_difference'][:p]) for r in lion]);axes[3].set_title('Lion parameter contrast')
        for r in lora:axes[4].loglog(etas,[a['remainder'] for a in r['rows']],'o-',label=f"rank {r['rank']}")
        axes[4].set_title('LoRA null-sketch remainder');axes[4].legend();axes[5].axis('off')
    otlr_figure('E11','C4 five catalog witnesses',draw)
    g.row['data']=panels
    g.row['notes'].append('Clipping controls use direct gradient laws on a linear-logit sigmoid; warm Adam and Lion use the original C23 network. LoRA is a separate d=k=4 bilinear sigmoid model. Controlled neural, not Gemma.')
    return g.finish()


def wrong_sign(gold,predicted):
    mask=abs(np.asarray(gold))>1e-10
    return float(np.mean(np.asarray(gold)[mask]*np.asarray(predicted)[mask]<0)) if mask.any() else None


def check_E12(quick=False):
    g=Group('E12','C6 margin on the unchanged E6 held-out panels',
            'Read frozen rows/models; exact seven-update reference continuation; empirical segment curvature', 'controlled_neural')
    paths=[HERE/'proxy_split_freeze.json',HERE/'proxy_models_frozen.json',HERE/'raw/E6_held_out_records.json',
           HERE/'proxy_split_freeze_E9.json',HERE/'proxy_models_frozen_E9.json']
    before={str(path.relative_to(HERE)):digest(path) for path in paths}
    manifest=json.loads(paths[0].read_text());frozen=json.loads(paths[1].read_text());held=json.loads(paths[2].read_text())
    points,menu=provider_menu();ref=menu[0]['law'];tail,tw=complete_sequences(points,ref,7)
    g.check(np.array_equal(points,np.array(manifest['message_support'])),'E12 provider_menu matches frozen E6 support')
    cache={};rows=[]
    for record in held:
        key=record['cell'];state_name=record['state'];recipe_id=record['recipe'];task=record['task']
        if key not in cache:
            state={a:(np.array(b) if a in ('theta','m','v') else b) for a,b in manifest['states'][state_name].items()}
            recipe=manifest['recipes'][recipe_id];eta=recipe['eta'];eps=recipe['eps'];opt=recipe['opt'];x=state_vector(state)
            inc,raw=increment_panel(x,points,2,state['step']+1,eta,opt,1.,eps)
            fun=lambda a:continuation(a,tail,tw,2,state['step']+1,eta,opt,1.,task,3,eps)
            laws=np.array([provider['law'] for provider in menu]);directions=(laws-ref)@inc[0]
            repeated=np.repeat(x,len(laws),axis=0)
            scores,errors=richardson(fun,repeated,directions)
            # Complete coordinate gradient as requested, independently checks
            # every directional contraction. Positive-v relative perturbations.
            relative=np.r_[np.ones(17),np.maximum(abs(x[0,17:34]),1e-5),np.maximum(x[0,34:],1e-8)]
            grad,graderr=full_gradient(fun,x,relative)
            fullscores=directions@grad
            valid=task!='threshold' and bool(np.all(np.isfinite(grad)))
            if valid:
                g.le(float(np.max(abs(scores-fullscores))),2e-6,
                     'full-state gradient agrees with directional Richardson',dict(cell=key))
                curvature=curvature_search(fun,x,inc,np.r_[np.ones(17),np.full(34,eta)],quick)
                norms=np.sum((inc[0]*np.r_[np.ones(17),np.full(34,eta)])**2,axis=1)
                bounds=(laws+ref)@norms*curvature['M2']/2
            else:
                curvature=dict(M2=None,reason='discontinuous threshold continuation' if task=='threshold' else 'no finite central derivative at cold state')
                bounds=[None]*len(menu)
            # The same untouched helper verifies the frozen gold values.
            measured=proxy_panel(g,state,recipe,points,menu)
            cache[key]=dict(scores=fullscores,directional_scores=scores,errors=errors,
                gradient=grad,gradient_error=graderr,bounds=bounds,curvature=curvature,eligible=valid,
                gold=(laws-ref)@measured[8][task])
            print(f'E12 audited {key}',flush=True)
        audit=cache[key];index=next(i for i,p in enumerate(menu) if p['name']==record['provider'])
        g.eq(audit['gold'][index],record['gold'],'E12 gold matches frozen E6 row',tol=1e-14)
        score=float(audit['scores'][index]);bound=audit['bounds'][index]
        margin=bool(audit['eligible'] and abs(score)>bound)
        wrong=bool(abs(record['gold'])>1e-10 and score*record['gold']<0)
        if margin:g.check(not wrong,'C6 no wrong sign above empirical margin',dict(cell=key,provider=record['provider'],score=score,bound=bound,gold=record['gold']))
        fit=frozen['memory_transfer_models']['fresh' if record['split']=='fresh_to_warm' else 'warm2'] if record['split'] in ('fresh_to_warm','warm_to_fresh') else frozen['models']
        predictions={name:float(np.array(record['features'][name])/model['scale']@model['coefficients']) for name,model in fit.items()}
        predictions['full_tail_gold']=record['gold']
        rows.append(dict(split=record['split'],cell=key,provider=record['provider'],task=task,gold=record['gold'],
            oracle_score=score,remainder_bound=bound,margin=margin,eligible=audit['eligible'],wrong_sign=wrong,
            differentiation_error=float(audit['errors'][index]),proxy_predictions=predictions,
            margin_ratio=abs(score)/bound if bound is not None and bound>0 else None))
        g.count()
    summary=[]
    for split in ['overall']+list(dict.fromkeys(r['split'] for r in rows)):
        subset=[r for r in rows if split=='overall' or r['split']==split]
        eligible=[r for r in subset if r['eligible']]; margin=[r for r in eligible if r['margin']]
        proxies={}
        for name in rows[0]['proxy_predictions']:
            failures=[r for r in eligible if abs(r['gold'])>1e-10 and r['proxy_predictions'][name]*r['gold']<0]
            nongold=[r for r in eligible if abs(r['gold'])>1e-10]
            outside=[r for r in nongold if not r['margin']];inside=[r for r in nongold if r['margin']]
            outside_rate=wrong_sign([r['gold'] for r in outside],[r['proxy_predictions'][name] for r in outside])
            inside_rate=wrong_sign([r['gold'] for r in inside],[r['proxy_predictions'][name] for r in inside])
            failure_fraction=float(np.mean([not r['margin'] for r in failures])) if failures else None
            base_fraction=len(outside)/len(nongold) if nongold else None
            proxies[name]=dict(wrong_sign_all_rows=wrong_sign([r['gold'] for r in subset],[r['proxy_predictions'][name] for r in subset]),
                wrong_sign_eligible=wrong_sign([r['gold'] for r in eligible],[r['proxy_predictions'][name] for r in eligible]),
                wrong_sign_margin=wrong_sign([r['gold'] for r in margin],[r['proxy_predictions'][name] for r in margin]),
                failures=len(failures),fraction_failures_outside_margin=failure_fraction,
                fraction_nonzero_gold_outside_margin=base_fraction,
                failure_enrichment_outside_margin=failure_fraction/base_fraction if failure_fraction is not None and base_fraction else None,
                wrong_sign_outside_margin=outside_rate,wrong_sign_inside_margin=inside_rate,
                failures_concentrated_outside=bool(outside_rate>inside_rate) if outside_rate is not None and inside_rate is not None else None)
        summary.append(dict(split=split,rows=len(subset),eligible_rows=len(eligible),margin_rows=len(margin),
            margin_fraction_all_rows=len(margin)/len(subset),margin_fraction_eligible=len(margin)/len(eligible) if eligible else None,
            oracle_wrong_sign_all_rows=wrong_sign([r['gold'] for r in subset],[r['oracle_score'] for r in subset]),
            oracle_wrong_sign_eligible=wrong_sign([r['gold'] for r in eligible],[r['oracle_score'] for r in eligible]),
            oracle_wrong_sign_margin=wrong_sign([r['gold'] for r in margin],[r['oracle_score'] for r in margin]),proxies=proxies))
    bins=[]
    for lo,hi in ((0,.25),(.25,.5),(.5,1),(1,2),(2,math.inf)):
        selected=[r for r in rows if r['eligible'] and r['margin_ratio'] is not None and lo<=r['margin_ratio']<hi]
        bins.append(dict(low=lo,high=hi,n=len(selected),oracle=wrong_sign([r['gold'] for r in selected],[r['oracle_score'] for r in selected]),
            proxies={name:wrong_sign([r['gold'] for r in selected],[r['proxy_predictions'][name] for r in selected]) for name in rows[0]['proxy_predictions']}))
    after={str(path.relative_to(HERE)):digest(path) for path in paths}
    g.check(before==after,'E12 never modifies or refits frozen E6/E9 artifacts')
    def draw(fig,ax):
        x=np.arange(len(bins));width=.18
        for j,name in enumerate(('oracle','mean_gradient','first_delta','delta_plus_m_v')):
            values=[b['oracle'] if name=='oracle' else b['proxies'][name] for b in bins]
            ax.bar(x+(j-1.5)*width,[np.nan if v is None else v for v in values],width,label=name)
        ax.set_xticks(x,[f"{b['low']:g}–{b['high']:g}\n(n={b['n']})" for b in bins]);ax.set(xlabel='|oracle score| / empirical remainder bound',ylabel='wrong-sign rate')
        ax.legend(fontsize=8)
    otlr_figure('E12','Frozen E6 rows: oracle and proxy errors by C6 margin',draw)
    g.row['data']=dict(rows=rows,summary=summary,bins=bins,cell_audits=cache,frozen_sha256_before=before,frozen_sha256_after=after,
        raw=write_json('raw/E12_margin_rows.json',rows),
        limitations=['Threshold-return rows are discontinuous and ineligible for C6; formal finite-difference scores are recorded but no smooth remainder bound is claimed.',
            'All smooth bounds use unpadded sampled/hill-climbed directional curvature, so zero wrong signs above margin is empirical, not a certified guarantee.',
            'E6 zero-prediction convention is preserved: prediction zero is not a wrong sign. Counts with nonzero gold use abs(gold)>1e-10.'])
    g.info('Frozen-panel diagnostic; nonsmooth rows are explicitly ineligible for C6. Empirical margin assertion remains a hard check.')
    return g.finish()
