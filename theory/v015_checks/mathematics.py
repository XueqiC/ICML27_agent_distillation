"""Finite checks derived from the statements; no package verifier dependency."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from scipy.optimize import linprog
from common import *
import common as C


def markov_instance(rng):
    s = int(rng.integers(2, 5)); m = int(rng.integers(2, 5)); R = int(rng.integers(1, 5))
    return dict(s=s, m=m, R=R,
        P=np.array([[distribution(rng, m) for _ in range(s+1)] for _ in range(R)]),
        Q=np.array([[distribution(rng, m) for _ in range(s+1)] for _ in range(R)]),
        U=rng.integers(0, s+1, (R, s+1, m)),
        terminal=rational_array(rng, (s+1,)), initial=distribution(rng, s+1),
        phi=rational_array(rng, (m, m-1)),
        drift=rational_array(rng, (R, s, m, m-1), -2, 3, 24),
        b=rational_array(rng, (R, s, m-1), -3, 4, 12),
        audit_P=distribution(rng, m), audit_Q=distribution(rng, m))


def evaluate_markov(g, x):
    s, m, R = (x[k] for k in ('s', 'm', 'R'))
    P, Q, U = (x[k] for k in ('P', 'Q', 'U'))
    V = [None]*(R+1); V[R] = x['terminal']
    Vq = x['terminal'].copy()
    for t in reversed(range(R)):
        V[t] = np.array([P[t, z]@V[t+1][U[t, z]] for z in range(s+1)])
        Vq = np.array([Q[t, z]@Vq[U[t, z]] for z in range(s+1)])
    gap = abs(x['initial']@(V[0]-Vq))
    phi = x['phi']; delta_audit = (x['audit_P']-x['audit_Q'])@phi
    D = max(abs(delta_audit)); occupancy = x['initial'].copy()
    stopped = occupancy.copy(); rho = stopped[s]; stopped[s] = F(0)
    bound = F(0); bound_drift = F(0); telescoping = F(0); stopped_sum = F(0)
    local = []
    for t in range(R):
        rows = []
        for z in range(s):
            response = V[t+1][U[t, z]]
            b = x['b'][t, z]
            residual = response-phi@b
            c = (max(residual)+min(residual))/2
            residual -= c
            tau = max(abs(residual)); B = sum(abs(b))
            transport = max(abs((P[t, z]-Q[t, z])@phi-delta_audit))
            phiz = phi+x['drift'][t, z]
            alpha = max(abs((phiz-phi).flatten()))
            drift_lhs = max(abs((P[t, z]-Q[t, z])@phiz))
            g.le(drift_lhs, D+transport+2*alpha, 'A5 state-dependent features', x)
            rz = response-phiz@b
            cz = (max(rz)+min(rz))/2; rz -= cz
            rows.append(dict(z=z, c=c, b=b, residual=residual, tau=tau, B=B,
                a=transport, alpha=alpha, tau_z=max(abs(rz)), response=response))
        Bt = max(r['B'] for r in rows); taut = max(r['tau'] for r in rows)
        tauzt = max(r['tau_z'] for r in rows)
        for r in rows:
            z = r['z']
            bound += stopped[z]*(Bt*(D+r['a'])+2*taut)
            bound_drift += stopped[z]*(Bt*(D+r['a']+2*r['alpha'])+2*tauzt)
            stopped_sum += stopped[z]*((P[t, z]-Q[t, z])@r['response'])
        next_occ = np.full(s+1, F(0)); next_stop = next_occ.copy()
        for z in range(s+1):
            telescoping += occupancy[z]*((P[t, z]-Q[t, z])@V[t+1][U[t, z]])
            for y in range(m):
                next_occ[U[t, z, y]] += occupancy[z]*Q[t, z, y]
                next_stop[U[t, z, y]] += stopped[z]*Q[t, z, y]
        # Exit at the terminal time does not count as sigma<R.
        if t+1 < R:
            rho += next_stop[s]
        next_stop[s] = F(0)
        occupancy, stopped = next_occ, next_stop
        local.append(rows)
    g.eq(abs(telescoping), gap, 'full occupancy telescoping', x)
    g.le(abs(x['initial']@(V[0]-Vq)-stopped_sum), rho, 'stopped remainder <= exit probability', x)
    g.le(gap, bound+rho, 'A4 and 39a', x)
    g.le(gap, bound_drift+rho, 'A4 with A5', x)
    # A17 uses independently computed reference benefit and the same bound.
    base = x['initial']@x['terminal']; gamma = x['initial']@V[0]-base
    g.le(gamma-(bound+rho), x['initial']@Vq-base, 'A17 improvement bridge', x)
    g.count()
    record = dict(inputs=x, gap=gap, bound=bound+rho, drift_bound=bound_drift+rho,
                  rho=rho, D=D, local=local, stopped_sum=stopped_sum)
    return float(gap/(bound+rho)) if bound+rho else 0., record


def check_V1(quick=False):
    g = Group('V1', 'Theorem 19 repair; A1–A5, A17 and IID affine (40)',
              'Fraction-valued Markov kernels, backward values and independent forward occupancies')
    worst = None
    for _ in range(30 if quick else 2000):
        score, record = evaluate_markov(g, markov_instance(g.rng))
        if worst is None or score > worst['ratio']:
            worst = dict(ratio=score, **record)
    def mutate(x):
        x = deepcopy(x); t = int(g.rng.integers(x['R'])); z = int(g.rng.integers(x['s']+1))
        x['Q'][t, z] = distribution(g.rng, x['m'])
        return x
    adversarial = search(g, lambda: markov_instance(g.rng), mutate,
        lambda x: evaluate_markov(g, x), 2 if quick else 10, 4 if quick else 25)
    tight=dict(s=2,m=2,R=1,P=np.array([[[F(1),F(0)]]*3],dtype=object),
        Q=np.array([[[F(0),F(1)]]*3],dtype=object),U=np.array([[[0,1]]*3]),
        terminal=np.array([F(0),F(1),F(0)]),initial=np.array([F(1),F(0),F(0)]),
        phi=np.array([[F(0)],[F(1)]]),drift=np.full((1,2,2,1),F(0)),
        b=np.full((1,2,1),F(1)),audit_P=np.array([F(1),F(0)]),audit_Q=np.array([F(1),F(0)]))
    tight_ratio,tight_record=evaluate_markov(g,tight)
    g.eq(tight_record['bound'],tight_record['gap'],'nonzero-transport finite sharp witness',tight)
    old = 2*math.sqrt(math.log(80)/20000)
    regression = dict(audit_P=['1', '0'], audit_Q=['1', '0'],
        train_P=['1', '0'], train_Q=['0', '1'], feature=['0', '1'],
        n0=10000, nS=10000, delta=.05, L=1, M=1, d=1,
        gap=1, old_bound=old, corrected_transport_bound=old+1, corrected_exit_bound=1)
    g.check(old < 1, 'expected failure of old 39: old bound < actual gap', regression)
    g.le(1, old+1, '39a a=1 repair', regression)
    g.le(1, 1, '39a immediate exit rho=1 repair', regression)
    # Search omitted reachable states, with all their contributions illegally zeroed.
    omitted = None
    for denom in range(2, 10):
        reach = F(denom-1, denom)
        actual = reach
        wrong = F(0)
        if actual > wrong:
            omitted = dict(reach_probability=reach, path=['covered -> omitted -> reward'],
                omitted_P=['1', '0'], omitted_Q=['0', '1'], response=['0', '1'],
                actual_gap=actual, invalid_bound=wrong, valid_exit_bound=reach)
    g.check(omitted is not None and omitted['actual_gap'] > omitted['invalid_bound'],
            'omission search must witness failure', omitted)
    # Legitimate warm moments at an identical parameter, followed by the same g.
    warm = []
    for first in (F(1), F(2)):
        warm.append(dict(theta=F(-1), m=first/2, v=first*first/2,
                         next_m=first/4-F(3, 8)))
    g.check(warm[0]['theta'] == warm[1]['theta'] and warm[0]['next_m']*warm[1]['next_m'] < 0,
            'parameter-only coverage misses optimizer response', warm)
    affine_rows = []
    def affine_evaluate(x):
        p, q, weights, p0, N = x
        def expected(law):
            return sum((math.prod(law[y] for y in hist)*
                (weights[0]*p0+sum(weights[i+1]*F(y) for i, y in enumerate(hist)))**N
                for hist in it.product((0, 1), repeat=len(weights)-1)), F(0))
        lhs = abs(expected(p)-expected(q)); D = abs(p[1]-q[1]); rhs = N*D
        g.le(lhs, rhs, '40 under IID convex affine premises', x); g.count()
        return float(lhs/rhs) if rhs else 0., dict(P=p,Q=q,weights=weights,p0=p0,N=N,gap=lhs,bound=rhs)
    def make_affine():
        return [distribution(g.rng, 2), distribution(g.rng, 2), distribution(g.rng, 4),
                F(int(g.rng.integers(13)),12), int(g.rng.integers(1,5))]
    def mutate_affine(x):
        x=deepcopy(x); x[int(g.rng.integers(2))]=distribution(g.rng,2); return x
    for _ in range(10 if quick else 120):
        affine_rows.append(affine_evaluate(make_affine())[1])
    affine_search=search(g,make_affine,mutate_affine,affine_evaluate,2 if quick else 6,5 if quick else 20)
    # Coefficient-weighted uncertainty is essential even with zero residual.
    stability = dict(feature=[F(0), F(1,10000)], response=[F(0),F(1)], B=10000,
                     discrepancy=F(1,10000), residual=0, actual_gap=1)
    g.eq(stability['B']*stability['discrepancy'], F(1), 'noisy-feature stability', stability)
    # Independently enumerate the audit event, not just the population-mean
    # version used above. Conditional on the reference count, the two candidate
    # counts are independent; sum exact binomial probabilities in that order.
    def audit_make():
        return dict(p0=F(int(g.rng.integers(1,8)),8),
            pj=[F(int(g.rng.integers(1,8)),8) for _ in range(2)],
            n=int(g.rng.integers(16,33)),delta=F(int(g.rng.integers(1,5)),10))
    def audit_mutate(x):
        x=deepcopy(x)
        if g.rng.random()<.5:x['p0']=F(int(g.rng.integers(1,8)),8)
        else:x['pj'][int(g.rng.integers(2))]=F(int(g.rng.integers(1,8)),8)
        return x
    def audit_evaluate(x):
        n=x['n'];delta=x['delta'];D=np.array([float(abs(x['p0']-p)) for p in x['pj']])
        radius=math.sqrt(math.log(4/float(delta))/(2*n))+math.sqrt(math.log(8/float(delta))/(2*n))
        laws=[[F(math.comb(n,k))*p**k*(1-p)**(n-k) for k in range(n+1)] for p in [x['p0']]+x['pj']]
        hats=abs(np.arange(n+1)[:,None]/n-np.arange(n+1)[None,:]/n)
        eligible=[abs(hats-d)<=radius for d in D]
        event=F(0);worst_selection=-1.
        for i in range(n+1):
            mass=[sum((law[k] for k in range(n+1) if keep[i,k]),F(0)) for law,keep in zip(laws[1:],eligible)]
            event+=laws[0][i]*mass[0]*mass[1]
            j0,j1=np.meshgrid(np.arange(n+1),np.arange(n+1),indexing='ij')
            chosen=np.where(hats[i,j0]<=hats[i,j1],0,1)
            on_event=eligible[0][i,j0]&eligible[1][i,j1]
            excess=D[chosen]-min(D)-2*radius
            if on_event.any():worst_selection=max(worst_selection,float(excess[on_event].max()))
        g.le(1-event,delta,'37 exact finite binomial audit failure probability',x)
        g.le(worst_selection,0,'38 on simultaneous finite audit event',x,tol=1e-8)
        g.count()
        return float((1-event)/delta),dict(inputs=x,event_probability=event,radius=radius,
            population_discrepancies=D,worst_selection_excess=worst_selection)
    audit_search=search(g,audit_make,audit_mutate,audit_evaluate,2 if quick else 5,2 if quick else 12)
    g.row['data'] = dict(worst_random=worst, adversarial=adversarial, regression=regression,
        omission_witness=omitted, optimizer_state_witness=warm, affine_40=affine_rows,
        affine_adversarial=affine_search, coefficient_stability=stability,sharp_transport_witness=tight_record,
        exact_finite_audit_37_38=audit_search,
        audit_interpretation='Exact population feature means are used for Dbar=D; the old-bound regression separately includes its finite-sample radius.')
    g.row['best_ratio'] = max(worst['ratio'], adversarial['best']['score'],tight_ratio)
    return g.finish()


def check_V2(quick=False):
    g = Group('V2', 'A2 cold Adam/AdamW map, inverse and minimal quotient dimensions',
              'Independent recurrence, rational arithmetic, exact ranks/nullspaces and HiGHS separation', 'mixed')
    examples=[]
    for _ in range(20 if quick else 300):
        gradient=rational_array(g.rng,(int(g.rng.integers(1,5)),),-20,21,7)
        theta=rational_array(g.rng,gradient.shape,-5,6,3)
        eta=F(int(g.rng.integers(1,10)),10); eps=F(int(g.rng.integers(1,10)),100)
        decay=F(int(g.rng.integers(3)),10)
        b1=F(int(g.rng.integers(9)),10); b2=F(int(g.rng.integers(9)),10)
        raw_m=(1-b1)*gradient; raw_v=(1-b2)*gradient**2
        # sqrt(g²)=|g| is exact for these rational squares.
        corrected_v=raw_v/(1-b2)
        roots=np.array([F(math.isqrt(z.numerator),math.isqrt(z.denominator)) for z in corrected_v])
        g.eq(max(abs(roots**2-corrected_v)),0,'rational corrected second moments have exact square roots')
        result=(1-eta*decay)*theta-eta*(raw_m/(1-b1))/(roots+eps)
        formula=(1-eta*decay)*theta-eta*gradient/(abs(gradient)+eps)
        g.eq(max(abs(result-formula)),0,'A7 exact rational AdamW',dict(g=gradient,theta=theta,beta=[b1,b2]))
        u=gradient/(abs(gradient)+eps); inverse=eps*u/(1-abs(u))
        g.eq(max(abs(inverse-gradient)),0,'A8 exact inverse',dict(g=gradient,eps=eps))
        measured=optimizer(np.array(theta,float),np.array(gradient,float),np.zeros(len(theta)),
            np.zeros(len(theta)),1,eta=float(eta),eps=float(eps),beta=(float(b1),float(b2)),decay=float(decay))[0]
        g.eq(float(max(abs(measured-np.array(formula,float)))),0,'A6 float64 vs rational reference',
             dict(gradient=gradient,theta=theta,eps=eps,eta=eta,beta=[b1,b2]),tol=1e-12)
        p=int(g.rng.integers(1,5)); alphabet=np.unique(g.rng.integers(-2,3,(int(g.rng.integers(4,15)),p)),axis=0)
        patterns=np.unique(np.sign(alphabet),axis=0); s=len(patterns)
        indicators=sp.Matrix([[int(np.array_equal(np.sign(a),v)) for v in patterns] for a in alphabet])
        g.eq(indicators.rank(),s,'reachable utility rank',dict(alphabet=alphabet))
        g.eq(sp.Matrix.hstack(sp.ones(len(alphabet),1),indicators[:,:s-1]).rank(),s,'s-1 indicators suffice')
        example=dict(alphabet=alphabet,patterns=patterns,s=s,necessary_and_sufficient=s-1,
                     finite_epsilon_dimension=len(alphabet)-1)
        if s>1:
            # Generic fewer features, not just a dropped indicator.
            features=sp.Matrix(g.rng.integers(-3,4,(max(s-2,0),s)).tolist()) if s>2 else sp.zeros(0,s)
            constraints=sp.ones(1,s).col_join(features)
            null=constraints.nullspace()[0]
            v=np.array([F(int(z.p),int(z.q)) for z in null],dtype=object)
            scale=F(1,2*s)/max(abs(v)); center=np.full(s,F(1,s))
            P=center+scale*v; Q=center-scale*v
            event=int(np.flatnonzero(v)[0]); contrast=P[event]-Q[event]
            g.eq(sum(P-Q),0,'separating perturbation has zero mass')
            g.check(contrast != 0 and min(P)>=0 and min(Q)>=0,'fewer features fail',dict(P=P,Q=Q,features=features))
            A=np.array(constraints.tolist(),float)
            cost=np.zeros(2*s); cost[event]=-1; cost[s+event]=1
            lp=linprog(cost,A_eq=np.r_[np.c_[A,-A],np.r_[np.ones(s),np.zeros(s)][None,:]],
                b_eq=np.r_[np.zeros(len(A)),1],bounds=[(0,None)]*(2*s),method='highs')
            g.check(lp.success,'quotient LP feasible',example)
            if lp.success:
                g.le(float(np.max(abs(np.r_[np.c_[A,-A],np.r_[np.ones(s),np.zeros(s)][None,:]]@lp.x-
                    np.r_[np.zeros(len(A)),1]))),1e-8,'LP equality residual',example)
                g.check(-lp.fun>1e-8,'LP separates a utility',example)
            example.update(features=features,zero_mass_perturbation=v,P=P,Q=Q,utility_indicator=event,contrast=contrast)
        examples.append(example); g.count()
    conditioning=[dict(u=F(10**k-1,10**k),epsilon=F(1,100),
                       inverse_derivative=F(1,100)*10**(2*k)) for k in range(1,13)]
    joint=dict(P=[[1,1],[-1,-1]],Q=[[1,-1],[-1,1]],weights=['1/2','1/2'],
               coordinate_marginals_identical=True,same_sign_utility_P=1,same_sign_utility_Q=0)
    g.eq(np.sign(np.array(joint['P'])).mean(axis=0).sum(),0,'P marginal sign means')
    g.eq(np.sign(np.array(joint['Q'])).mean(axis=0).sum(),0,'Q marginal sign means')
    g.check(all(a*b>0 for a,b in joint['P']) and all(a*b<0 for a,b in joint['Q']),
            'joint sign utility separates equal marginals',joint)
    g.row['data']=dict(quotients=examples,conditioning=conditioning,joint_vs_marginal=joint)
    return g.finish()


def check_V3(quick=False):
    g=Group('V3','A3 margin-sensitive approximation (A9) and sign TV bound',
            'Exact rational clipped sums of piecewise-linear coordinate functions')
    def make():
        n=int(g.rng.integers(3,9)); p=int(g.rng.integers(1,5))
        return dict(G=rational_array(g.rng,(n,p),-8,9,8),P=distribution(g.rng,n),Q=distribution(g.rng,n),
            L=rational_array(g.rng,(p,),0,6,8), c=rational_array(g.rng,(p,),-4,5,8),
            knot=rational_array(g.rng,(p,),-4,5,8),eta=F(int(g.rng.integers(1,9)),8),
            a=F(int(g.rng.integers(1,12)),16),eps=F(int(g.rng.integers(1,10)),100),offset=F(1,4))
    def evaluate(x):
        G,P,Q,L=(x[k] for k in ('G','P','Q','L')); eps=x['eps']; a=x['a']
        sign=np.sign(G); soft=G/(abs(G)+eps)
        def utility(u):
            # |x-knot| is 1-Lipschitz; clipping does not enlarge L_i.
            return np.array([min(F(1),max(F(0),x['offset']+sum(L*abs(row-x['knot'])))) for row in u])
        f=utility(x['c']-x['eta']*soft); f0=utility(x['c']-x['eta']*sign)
        gap=abs((P-Q)@f); sign_gap=abs((P-Q)@f0)
        beta=P@((abs(G)>0)&(abs(G)<a))+Q@((abs(G)>0)&(abs(G)<a))
        bound=sign_gap+x['eta']*sum(L*(beta+2*eps/(a+eps)))
        keys={tuple(v) for v in sign}
        tv=sum(abs(sum((P[i]-Q[i] for i in range(len(G)) if tuple(sign[i])==v),F(0))) for v in keys)/2
        g.le(gap,bound,'A9',x); g.le(sign_gap,tv,'sign gap <= sign TV',x); g.count()
        return float(gap/bound) if bound else 0.,dict(inputs=x,gap=gap,bound=bound,sign_gap=sign_gap,TV=tv)
    def mutate(x):
        x=deepcopy(x); key=g.rng.choice(['a','eps','offset','L','knot'])
        if key in ('L','knot'):
            x[key][int(g.rng.integers(len(x[key])))]+=F(int(g.rng.choice([-1,1])),32)
            if key=='L': x[key]=np.maximum(x[key],F(0))
        else:
            x[key]=max(F(1,10000),x[key]+F(int(g.rng.choice([-1,1])),100))
        return x
    worst=None
    for _ in range(30 if quick else 1000):
        score,rec=evaluate(make())
        if worst is None or score>worst['score']: worst=dict(score=score,instance=rec)
    adv=search(g,make,mutate,evaluate,2 if quick else 12,5 if quick else 40)
    # Separate hill-climb explicitly targets Delta_sign/TV.
    def sign_score(x):
        _,r=evaluate(x)
        return float(r['sign_gap']/r['TV']) if r['TV'] else 0.,r
    tv_search=search(g,make,mutate,sign_score,2 if quick else 8,5 if quick else 20)
    g.row['best_ratio']=max(worst['score'],adv['best']['score'])
    g.row['data']=dict(worst_random=worst,adversarial=adv,sign_TV_adversarial=tv_search)
    return g.finish()


def exact_history(first, eps=F(0), eta=F(1), beta=F(1,2)):
    m1=(1-beta)*first; v1=(1-beta)*first**2
    theta1=-eta*first/(abs(first)+eps)
    m2=beta*m1+(1-beta)*F(-3,4); v2=beta*v1+(1-beta)*F(9,16)
    return dict(first=first,epsilon=eps,theta1=theta1,m1=m1,v1=v1,m2=m2,
        mhat2=m2/(1-beta**2),vhat2=v2/(1-beta**2),
        delta2=-sp.Rational(eta)*sp.Rational(m2/(1-beta**2))/
        (sp.sqrt(sp.Rational(v2/(1-beta**2)))+sp.Rational(eps)))


def check_V4(quick=False):
    g=Group('V4','A4 magnitude-bearing optimizer state and quadratic realization',
            'Fraction moments, symbolic square roots, positive-epsilon grid and beta-grid search')
    a,b=exact_history(F(1)),exact_history(F(2))
    for lhs,rhs in ((a['m2'],F(-1,8)),(b['m2'],F(1,8)),(a['vhat2'],F(17,24)),(b['vhat2'],F(41,24))):
        g.eq(lhs,rhs,'A10 exact moment')
    g.eq(a['theta1'],b['theta1'],'identical ideal first parameters')
    g.check(a['delta2']>0 and b['delta2']<0,'opposite second updates',dict(a=a,b=b))
    variants=[]
    for eps in [F(0),F(1,10**8)]+[F(1,10**k) for k in range(1,13)]:
        aa,bb=exact_history(F(1),eps),exact_history(F(2),eps)
        target2=aa['theta1']+F(3,4)
        realized_g2=[z['theta1']-target2 for z in (aa,bb)]
        realized_m2=[z['m1']/2+gg/2 for z,gg in zip((aa,bb),realized_g2)]
        g.check(aa['delta2']>0 and bb['delta2']<0,'positive epsilon prescribed histories',dict(a=aa,b=bb))
        g.check(realized_m2[0]<0<realized_m2[1],'common quadratic second target realization',
            dict(a=aa,b=bb,target2=target2,g2=realized_g2,m2=realized_m2))
        variants.append(dict(a=aa,b=bb,first_targets=[-1,-2],common_second_target=target2,
                             quadratic_g2=realized_g2,quadratic_m2=realized_m2)); g.count()
    grids=[]
    for b1 in (F(0),F(1,10),F(1,2),F(9,10),F(99,100)):
        for b2 in (F(0),F(1,10),F(1,2),F(9,10),F(999,1000)):
            witness=None
            for _ in range(30 if quick else 300):
                first=F(int(g.rng.integers(1,41)),8); first2=F(int(g.rng.integers(1,41)),8)
                second=-F(int(g.rng.integers(1,41)),16)
                m=[(1-b1)*(b1*x+second) for x in (first,first2)]
                if m[0]*m[1]<0:
                    witness=dict(histories=[[first,second],[first2,second]],m2=m,
                        vhat2=[(1-b2)*(b2*x*x+second*second)/(1-b2*b2) for x in (first,first2)])
                    break
            grids.append(dict(beta1=b1,beta2=b2,found=witness is not None,witness=witness)); g.count()
    g.row['data']=dict(exact=[a,b],positive_epsilon=variants,grid=grids,
        grid_fraction=sum(r['found'] for r in grids)/len(grids),search_trials_per_cell=30 if quick else 300)
    return g.finish()


def union_recursion(d, reveal, K):
    state={0:F(1)}
    for _ in range(K):
        nxt={}
        for mask,p in state.items():
            for S,q in reveal.items(): nxt[mask|S]=nxt.get(mask|S,F(0))+p*q
        state=nxt
    return state


def union_inclusion_exclusion(d,reveal,K):
    result={}
    for mask in range(1<<d):
        val=F(0); subset=mask
        while True:
            mass=sum((p for s,p in reveal.items() if s&~subset==0),F(0))
            val+=(-1)**(mask.bit_count()-subset.bit_count())*mass**K
            if subset==0: break
            subset=(subset-1)&mask
        result[mask]=val
    return result


def hidden_case(g,d,K,n,w,reveal):
    # Enumerate full ordered trajectory histories and every bit world first.
    transcripts={}; worlds=list(it.product((0,1),repeat=d))
    for history in it.product(reveal,repeat=K):
        mass=math.prod(reveal[s] for s in history); mask=0
        for s in history: mask|=s
        for world in worlds:
            known=tuple(world[j] if mask>>j&1 else -1 for j in range(d))
            # History has the same likelihood in all compatible worlds. Its
            # posterior is independently reconstructed, not assumed in A11.
            key=(history,known)
            if key not in transcripts: transcripts[key]=[F(0),[]]
            transcripts[key][0]+=mass/F(2**d); transcripts[key][1].append(world)
    policies=list(it.product((0,1),repeat=d))
    @lru_cache(None)
    def terminal(known):
        posterior=[world for world in worlds if all(a<0 or a==b for a,b in zip(known,world))]
        return min(sum((sum(w[j]*int(action[j]!=world[j]) for j in range(d))
                        for world in posterior),F(0))/len(posterior) for action in policies)
    @lru_cache(None)
    def adaptive(known,budget):
        best=terminal(known)
        if budget:
            posterior=[world for world in worlds if all(a<0 or a==b for a,b in zip(known,world))]
            for j in range(d):
                if known[j]>=0: continue
                value=F(0)
                for label in (0,1):
                    mass=F(sum(world[j]==label for world in posterior),len(posterior))
                    child=list(known); child[j]=label
                    value+=mass*adaptive(tuple(child),budget-1)
                best=min(best,value)
        return best
    @lru_cache(None)
    def posterior_bayes(posterior):
        return min(sum((sum(w[j]*int(action[j]!=world[j]) for j in range(d))
                        for world in posterior),F(0))/len(posterior) for action in policies)
    brute=F(0); queried=F(0)
    for (_,known),(mass,posterior) in transcripts.items():
        # Explicit Bayes action enumeration under each transcript posterior.
        value=posterior_bayes(tuple(posterior))
        g.eq(value,terminal(known),'transcript posterior Bayes optimum',dict(known=known,worlds=posterior,w=w))
        brute+=mass*value; queried+=mass*adaptive(known,n)
    rec=union_recursion(d,reveal,K); ie=union_inclusion_exclusion(d,reveal,K)
    for mask in ie: g.eq(rec.get(mask,F(0)),ie[mask],'occupancy recursion vs inclusion-exclusion',dict(d=d,K=K,reveal=reveal,mask=mask))
    marginal=[sum((p for s,p in reveal.items() if s>>j&1),F(0)) for j in range(d)]
    formula=sum(w[j]*(1-marginal[j])**K for j in range(d))/2
    next_rec=union_recursion(d,reveal,K+1)
    next_risk=sum((p*sum(w[j] for j in range(d) if not mask>>j&1)/2
                   for mask,p in next_rec.items()),F(0))
    marginal_decrease=sum(w[j]*marginal[j]*(1-marginal[j])**K for j in range(d))/2
    g.eq(brute-next_risk,marginal_decrease,'A12 one-demonstration decrease',dict(d=d,K=K,w=w,reveal=reveal))
    greedy=sum((p*sum(sorted((w[j] for j in range(d) if not mask>>j&1),reverse=True)[n:])/2
                for mask,p in rec.items()),F(0))
    g.eq(brute,formula,'A11 transcript Bayes risk',dict(d=d,K=K,w=w,reveal=reveal))
    g.eq(queried,greedy,'A13 all adaptive label-dependent query trees vs greedy',dict(d=d,K=K,n=n,w=w,reveal=reveal))
    em=sum(p*(d-mask.bit_count()) for mask,p in rec.items())
    positive=sum(p*max(0,d-mask.bit_count()-n) for mask,p in rec.items())
    g.le(max(0,em-n),positive,'Jensen positive-part inequality',dict(d=d,K=K,n=n,reveal=reveal))
    g.count()
    return dict(d=d,K=K,n=n,w=w,reveal=reveal,n_transcripts=len(transcripts),
        n_query_nodes=adaptive.cache_info().currsize,Bayes=brute,A11=formula,adaptive=queried,
        greedy=greedy,union_distribution=rec,A12_decrease=marginal_decrease,
        E_M=em,E_positive=positive,positive_E=max(0,em-n))


def check_V5(quick=False):
    g=Group('V5','A5–A6 hidden-bit coverage, hard query budget and adaptivity',
            'All bit worlds, ordered reveal histories, Bayes actions and label-dependent decision trees')
    rows=[]
    configurations=[(d,K,n) for d in range(1,6) for K in range(5) for n in range(4)]
    # Cover each extreme explicitly; random reveal alphabets include dependent multi-bit sets.
    selected=[(5,4,3),(4,2,2),(3,0,1),(2,1,0)] if quick else configurations
    for d,K,n in selected:
        masks=sorted(set([0,(1<<d)-1,int(g.rng.integers(1<<d))]))
        probs=distribution(g.rng,len(masks),zero=False)
        reveal=dict(zip(masks,probs)); w=distribution(g.rng,d,zero=False)
        rows.append(hidden_case(g,d,K,n,w,reveal))
    # Uniform singleton occupancy checks A14 and supplies a strict Jensen witness.
    uniform=[]
    for d in range(2,6):
        for K in range(5):
            reveal={1<<j:F(1,d) for j in range(d)}
            rec=union_recursion(d,reveal,K); ie=union_inclusion_exclusion(d,reveal,K)
            C1=[sum((p for mask,p in rec.items() if mask.bit_count()==c),F(0)) for c in range(d+1)]
            C2=[sum((p for mask,p in ie.items() if mask.bit_count()==c),F(0)) for c in range(d+1)]
            g.eq(max(abs(a-b) for a,b in zip(C1,C2)),0,'C_K distribution two methods')
            for n in range(4):
                val=sum(p*max(0,d-c-n) for c,p in enumerate(C1))/(2*d)
                greedy=sum(p*max(0,d-mask.bit_count()-n)/(2*d) for mask,p in rec.items())
                g.eq(val,greedy,'A14 exact occupancy query curve')
                uniform.append(dict(d=d,K=K,n=n,risk=val,occupancy=C1));g.count()
    strict=next((r for r in rows if r['E_positive']>r['positive_E']),None)
    if strict is None:
        strict=hidden_case(g,2,2,1,[F(1,2)]*2,{1:F(1,2),2:F(1,2)})
    g.check(strict['E_positive']>strict['positive_E'],'strict Jensen witness',strict)
    # Hill-climb both inequalities: Jensen and arbitrary-query gain <= greedy.
    def make():
        return dict(d=4,K=int(g.rng.integers(5)),n=int(g.rng.integers(4)),
                    w=distribution(g.rng,4),law=distribution(g.rng,4))
    def mutate(x):
        x=deepcopy(x); key=g.rng.choice(['K','n','w','law'])
        x[key]=distribution(g.rng,4) if key in ('w','law') else int(g.rng.integers(4 if key=='n' else 5))
        return x
    def evaluate(x):
        rec=union_recursion(4,{1<<j:p for j,p in enumerate(x['law'])},x['K'])
        em=sum(p*(4-mask.bit_count()) for mask,p in rec.items())
        rhs=sum(p*max(0,4-mask.bit_count()-x['n']) for mask,p in rec.items())
        lhs=max(0,em-x['n']);g.le(lhs,rhs,'adversarial Jensen',x)
        worst=-F(1)
        for mask,p in rec.items():
            unseen=[j for j in range(4) if not mask>>j&1]
            greedy=sum(sorted((x['w'][j] for j in unseen),reverse=True)[:x['n']])
            for subset in it.combinations(unseen,min(x['n'],len(unseen))):
                gain=sum(x['w'][j] for j in subset)
                g.le(gain,greedy,'all query subset gains <= greedy',dict(x=x,mask=mask,subset=subset))
                worst=max(worst,gain-greedy)
        g.count();return float(lhs-rhs+worst),dict(inputs=x,jensen_lhs=lhs,jensen_rhs=rhs,query_excess=worst)
    adv=search(g,make,mutate,evaluate,2 if quick else 8,3 if quick else 20)
    g.row['data']=dict(cases=rows,uniform=uniform,strict_Jensen=strict,adversarial=adv)
    return g.finish()


def kl(p,q):
    return sum(float(a)*math.log(float(a/b)) for a,b in zip(p,q) if a)


def adaptive_transcripts(x):
    """Both worlds traverse exactly the same history-dependent randomized rule."""
    demos=x['demo']; channels=x['channels']; K=x['K']; N=x['N']; rule=x['rule']
    leaves={}; expected=np.zeros(2); chain=np.zeros(2)
    def visit(history,probabilities,depth):
        stop=F(1) if depth==N else rule[(sum((i+1)*v for i,v in enumerate(history))+depth)%len(rule)][0]
        if stop:
            leaves[history+(-1,)]=[p*stop for p in probabilities]
        if stop==1: return
        select=rule[(sum(history)+2*depth)%len(rule)][1]
        for query,choose in enumerate((select,1-select)):
            for world in (0,1):
                reach=float(probabilities[world]*(1-stop)*choose)
                expected[world]+=reach
                chain[world]+=reach*kl(channels[query][world],channels[query][1-world])
            for y in range(len(channels[query][0])):
                visit(history+(query+3,y),[probabilities[v]*(1-stop)*choose*channels[query][v][y] for v in (0,1)],depth+1)
    for history in it.product(range(len(demos[0])),repeat=K):
        probabilities=[math.prod(demos[v][y] for y in history) for v in (0,1)]
        visit(tuple(history),probabilities,0)
    values=list(leaves.values()); P=[p[0] for p in values]; Q=[p[1] for p in values]
    errors=[F(0),F(0)]
    for p,q in values:
        if p>q: errors[1]+=q
        elif q>p: errors[0]+=p
        else: errors[0]+=p/2;errors[1]+=q/2
    return dict(leaves=leaves,KL=[kl(P,Q),kl(Q,P)],E_N=expected,
                chain=chain+np.array([K*kl(demos[v],demos[1-v]) for v in (0,1)]),errors=errors)


def check_V6(quick=False):
    g=Group('V6','A7 two-world adaptive stopping and demonstration information credit',
            'Exact rational transcript probabilities; direct log KL versus separately accumulated chain', 'mixed')
    def make():
        # Symmetric informative demo guarantees a useful test despite noisy queries.
        p=F(int(g.rng.integers(6,10)),10)
        return dict(K=int(g.rng.integers(1,3)),N=int(g.rng.integers(1,5)),demo=[[p,1-p],[1-p,p]],
            channels=[[distribution(g.rng,2,False),distribution(g.rng,2,False)] for _ in range(2)],
            rule=[(F(int(g.rng.integers(4)),8),F(int(g.rng.integers(1,8)),8)) for _ in range(7)])
    def evaluate(x):
        r=adaptive_transcripts(x); delta=max(r['errors']); ratios=[]
        for world in (0,1):
            g.eq(sum(p[world] for p in r['leaves'].values()),1,'stopped transcript is normalized',x)
            g.eq(r['KL'][world],r['chain'][world],'direct transcript KL = adaptive chain',x,tol=1e-8)
            kd=kl(x['demo'][world],x['demo'][1-world])
            kx=max(kl(c[world],c[1-world]) for c in x['channels'])
            upper=x['K']*kd+kx*r['E_N'][world]
            g.le(r['KL'][world],upper,'query KL envelope',x,tol=1e-8)
            if 0<delta<F(1,2):
                lower=kl([1-delta,delta],[delta,1-delta])
                g.le(lower,upper,'A15 both directional budgets',dict(inputs=x,transcript=r),tol=1e-8)
                if kx:
                    g.le(max(0,lower-x['K']*kd)/kx,r['E_N'][world],'A16',x,tol=1e-8)
                ratios.append(lower/upper if upper else 0)
        g.count()
        return max(ratios,default=0),dict(inputs=x,**r,delta=delta,premise=0<delta<F(1,2))
    worst=None; premise_count=0
    for _ in range(10 if quick else 160):
        score,r=evaluate(make());premise_count+=r['premise']
        if worst is None or score>worst['score']:worst=dict(score=score,instance=r)
    def mutate(x):
        x=deepcopy(x)
        if g.rng.random()<.5:
            x['channels'][int(g.rng.integers(2))][int(g.rng.integers(2))]=distribution(g.rng,2,False)
        else:
            i=int(g.rng.integers(len(x['rule'])));x['rule'][i]=(F(int(g.rng.integers(5)),8),x['rule'][i][1])
        return x
    adv=search(g,make,mutate,evaluate,2 if quick else 8,3 if quick else 16)
    tight=dict(K=0,N=1,demo=[[F(1,2)]*2]*2,
        channels=[[[F(4,5),F(1,5)],[F(1,5),F(4,5)]]]*2,rule=[(F(0),F(1))])
    tightness,tight_record=evaluate(tight)
    g.eq(tightness,1.,'one informative binary query tightness',tol=1e-8)
    zero=dict(K=2,N=4,demo=[[F(1,2)]*2]*2,
        channels=[[[F(1,3),F(2,3)]]*2]*2,rule=[(F(1,4),F(1,2))])
    zr=adaptive_transcripts(zero)
    g.eq(zr['KL'][0],0.,'zero query and demonstration information')
    g.eq(sum(zr['errors']),F(1),'kappa_X=0 impossibility: sum of errors one')
    g.row['best_ratio']=max(tightness,adv['best']['score'])
    g.row['data']=dict(worst_random=worst,adversarial=adv,tight=tight_record,
        zero_information=zr,random_premise_count=premise_count,
        KL_arithmetic='Exact rational probabilities; transcendental logs evaluated in float64 at tolerance 1e-8')
    return g.finish()


def check_V7(quick=False):
    g=Group('V7','Sharp geometric recursion; failure of unconditional linear-in-R simplification',
            'Exact rational recurrence with slack; restart/hill-climb over eta,L,C,R')
    def make():
        return dict(eta=F(int(g.rng.integers(1,11)),10),L=F(int(g.rng.integers(11)),10),
                    C=F(int(g.rng.integers(1,11)),10),R=int(g.rng.integers(1,41)),
                    slack=F(int(g.rng.integers(1,11)),10))
    def evaluate(x):
        eta,L,c,R=[x[k] for k in ('eta','L','C','R')]
        e=F(0); equality=F(0); slacks=[]
        for t in range(R):
            slack=x.get('slack',F(1));slacks.append(slack)
            e=((1+eta*L)*e+c*eta**2)*slack
            equality=(1+eta*L)*equality+c*eta**2
        bound=c*eta**2*sum((1+eta*L)**s for s in range(R))
        g.le(e,bound,'geometric recursion bound',dict(inputs=x,slacks=slacks));g.eq(equality,bound,'sharpness at equality',x);g.count()
        return float(equality/(R*c*eta**2)),dict(inputs=x,e=e,bound=bound,linear=R*c*eta**2)
    def mutate(x):
        x=x.copy();key=g.rng.choice(['eta','L','C','R','slack'])
        x[key]=int(g.rng.integers(1,61)) if key=='R' else F(int(g.rng.integers(1,11 if key=='slack' else 21)),10)
        return x
    for _ in range(20 if quick else 300): evaluate(make())
    adv=search(g,make,mutate,evaluate,2 if quick else 8,4 if quick else 25)
    def geometric_evaluate(x):
        _,r=evaluate(x)
        return float(r['e']/r['bound']),r
    geometric=search(g,make,mutate,geometric_evaluate,2 if quick else 8,4 if quick else 25)
    witness=dict(eta=F(1,2),L=F(1),C=F(1),R=4)
    _,r=evaluate(witness)
    g.check(r['bound']>r['linear'],'linear simplification expected-failure witness',r)
    g.row['data']=dict(adversarial=adv,geometric_adversarial=geometric,linear_failure=r)
    g.row['best_ratio']=adv['best']['score']
    return g.finish()
