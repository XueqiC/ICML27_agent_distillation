"""Fast implementation tests. Scientific failures must remain visible in JSON."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check_v016", HERE / "check_v016.py")
checks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checks
spec.loader.exec_module(checks)
import numpy as np
import pytest


def test_warmth_recursion_and_bounds():
    for beta in checks.BETAS:
        v=0.
        for s in range(1,300):
            v=beta*v+(1-beta)*4
            k=(1-beta)*4/(beta*v)
            assert k == pytest.approx(checks.kbar(beta,s),rel=1e-10)
            assert max(1/s,1-beta)/beta <= k*(1+1e-12)
            assert k <= (1/s+1-beta)/beta*(1+1e-12)


def test_design_known_optimum_and_singular_cases():
    # No demo information; orthogonal query axes have c-optimal weights |c|/||c||_1.
    ID=np.zeros((2,2)); features=np.eye(2); c=np.array([1.,2.])
    sol=checks.solve_design(ID,features,c,1,10)
    assert sol["w"] == pytest.approx([1/3,2/3],abs=1e-7)
    assert sol["primal"] == pytest.approx(.9,rel=1e-7)
    assert sol["dual"] == pytest.approx(.9,rel=1e-7)
    ID=np.diag([2.,0.])
    missing=checks.solve_design(ID,features,c,10,0)
    assert np.isinf(missing["primal"]) and np.isinf(missing["dual"])
    estimable=checks.solve_design(ID,features,np.array([1.,0.]),10,0)
    assert estimable["primal"] == pytest.approx(.05)
    assert estimable["dual"] == pytest.approx(.05)


def test_water_filling_against_exhaustive_integer_allocation():
    W=np.array([1.,4.,.2]); k=np.array([1,0,8]); budget=9
    continuous=checks.water_fill(W,k,budget)
    assert continuous.sum() == pytest.approx(budget,abs=1e-9)
    integer=checks.integer_water(W,k,budget)
    actual=checks.allocation_objective(W,k,integer)
    exhaustive=min(checks.allocation_objective(W,k,[a,b,budget-a-b])
                   for a in range(budget+1) for b in range(budget-a+1))
    assert actual == pytest.approx(exhaustive,rel=1e-13)
    assert checks.allocation_objective(W,k,continuous) <= actual
    assert np.isinf(checks.allocation_objective([1.],[0.],[0.]))
    assert checks.round_counts([.2,.3,.5],7).sum()==7


@pytest.mark.parametrize("bc,wd,clip",[(False,0.,None),(True,.03,None),(True,.03,.1)])
def test_tiny_costate_all_coordinates(bc,wd,clip):
    state=np.array([[.3,-.2],[.1,.15],[.7,.9]])
    loss=dict(kind="quadratic",H=np.array([[1.2,.2],[.2,.7]]),a=np.array([.2,-.3]))
    cfg=checks.AdamConfig(.9,.99,.01,.001,bc,wd,clip)
    outcome=np.array([.3,-.6])
    _,grad=checks.rollout_costate(state,loss,cfg,4,3,outcome)
    for index in np.ndindex(state.shape):
        direction=np.zeros_like(state); direction[index]=1e-5
        plus=checks.rollout_costate(state+direction,loss,cfg,4,3,outcome,False)
        minus=checks.rollout_costate(state-direction,loss,cfg,4,3,outcome,False)
        assert grad[index] == pytest.approx((plus-minus)/2e-5,rel=1e-6,abs=2e-10)


def test_adam_bias_correction_and_amplitude_formula():
    state=np.zeros((3,2));g=np.array([2.,-3.])
    cfg=checks.AdamConfig(bias_correction=True,eps=.001)
    theta,m,v=checks.adam_step(state,g,cfg,1)
    assert theta == pytest.approx(-cfg.eta*g/(abs(g)+cfg.eps))
    assert m == pytest.approx((1-cfg.b1)*g)
    assert v == pytest.approx((1-cfg.b2)*g*g)
    b1,b2,T,rho=.9,.999,5000,.7
    star=(1-b1**(T+1))/(rho*(1-b2**(T+1)))
    assert checks.amplitude_formula(2*star,b1,b2,T,rho) == pytest.approx(0.,abs=1e-12)
    assert checks.amplitude_formula(star,b1,b2,T,rho)>0


@pytest.mark.parametrize('b1,b2,T,rho',[(.9,.99,0,.4),(.9,.999,20,.7),(0.,.99,50,1.)])
def test_exact_mean_field_against_independent_moment_recursion(b1,b2,T,rho):
    lambdas=np.array([-10.,-2.,0.,.5,1.,4.,100.])
    eta,alpha=.03,1.7
    # Warm moments followed by a real message step and independently iterated
    # mean-field tail moments. This does not use geometric b_j/c_j weights.
    m=b1*rho+(1-b1)*lambdas; v=b2+(1-b2)*lambdas**2
    zero_m=b1*rho;zero_v=b2
    response=m/np.sqrt(v)-zero_m/np.sqrt(zero_v)
    for _ in range(T):
        m=b1*m+(1-b1)*rho;v=b2*v+(1-b2)
        zero_m=b1*zero_m+(1-b1)*rho;zero_v=b2*zero_v+(1-b2)
        response+=m/np.sqrt(v)-zero_m/np.sqrt(zero_v)
    actual=checks.mean_field_response(lambdas,b1,b2,T,rho,eta,alpha)
    assert actual == pytest.approx(alpha*eta*response,rel=2e-12,abs=2e-13)
    assert checks.mean_field_response(0.,b1,b2,T,rho)==pytest.approx(0.,abs=1e-14)


@pytest.mark.parametrize('T',[20,50,100])
def test_large_lambda_limit_straddles_crossing_criterion(T):
    threshold=checks.amplitude_limit(.9,.99,T,1.)['rho_critical_approx']
    for factor in (.8,1.2):
        rho=factor*threshold
        limit=checks.amplitude_limit(.9,.99,T,rho)
        response=lambda x:float(checks.mean_field_response(x,.9,.99,T,rho))
        root=checks.positive_crossing(response,10.)
        assert (root['root'] is not None)==limit['predicts_crossing']==(factor>1)
        assert response(1e10)==pytest.approx(limit['value'],abs=2e-7)
        if factor<1:
            bound=root['search_bound']
            assert np.all(checks.mean_field_response(np.geomspace(1e-6,bound,300),.9,.99,T,rho)>0)
            assert checks.positive_crossing(response,10.,upper_bound=bound)['search_bound']==bound


def test_positive_crossing_excludes_trivial_zero_and_records_bound():
    assert checks.positive_crossing(lambda x:x*(7-x),1.)['root']==pytest.approx(7.)
    assert checks.positive_crossing(lambda x:x,1.,upper_bound=23.)['search_bound']==23.
    assert checks.positive_crossing(lambda x:x,1.,upper_bound=23.)['root'] is None


def test_quadratic_eligibility_and_parameter_only_derivative():
    eligible,crossing=checks.quadratic_region(np.array([-4.,4.,4.01,10.]),.999,5.)
    assert eligible.tolist()==[True,True,False,False]
    assert crossing  # Include equality despite float64 representation of beta2.
    assert checks.quadratic_region([1.,2.],.99,1.)[0].tolist()==[True,False]
    assert checks.quadratic_region([1.],.999,4.9)[1]
    assert not checks.quadratic_region([1.],.999,5.1)[1]
    b1,b2,rho=.9,.99,.7
    threshold=(1-b1)*b2/(b1*rho*(1-b2))
    for lam,expected_sign in ((.9*threshold,1),(1.1*threshold,-1)):
        numerator=checks.parameter_only_derivative_sign(lam,b1,b2,rho)
        def score(x): return (b1*rho+(1-b1)*x)/np.sqrt(b2+(1-b2)*x*x)
        fd=(score(lam+1e-5)-score(lam-1e-5))/2e-5
        assert np.sign(numerator)==np.sign(fd)==expected_sign
        assert fd==pytest.approx(numerator/(b2+(1-b2)*lam**2)**1.5,rel=1e-6)


def test_seedsequence_reproducibility_and_independence():
    assert np.array_equal(checks.rng_for(1,2).normal(size=20),checks.rng_for(1,2).normal(size=20))
    assert not np.array_equal(checks.rng_for(1,2).normal(size=20),checks.rng_for(1,3).normal(size=20))


def test_exogenous_branch_identity_against_executed_rollout():
    lambdas=np.array([-.5,1.,3.])
    outputs,warm=checks.scalar_amplitude_experiment(.9,.7,(0,5),16,(91,),lambdas)
    for T,row in outputs.items():
        j=np.arange(T+1)[:,None]
        for i,lam in enumerate(lambdas):
            m=warm["zero_m"][:T+1]+.1*.9**j*lam*np.sqrt(warm["v"])
            v=warm["zero_v"][:T+1]+.1*.9**j*lam**2*warm["v"]
            reference=warm["zero_m"][:T+1]/np.sqrt(warm["zero_v"][:T+1])
            expected=np.sum(m/np.sqrt(v)-reference,axis=0)
            assert row["rollout_samples"][:,i] == pytest.approx(expected,abs=2e-14)


def test_quick_cli_schema_and_visible_failures(tmp_path):
    output=tmp_path/"quick.json"
    run=subprocess.run([sys.executable,str(HERE/"check_v016.py"),"--quick","--output",str(output)],
                       capture_output=True,text=True,timeout=180)
    assert run.returncode in (0,1),run.stdout+run.stderr
    result=json.loads(output.read_text())
    checks.validate_results(result)
    assert len(result["checks"])==12
    assert result["config"]["quick"] is True
    assert result["config"]["CUDA_VISIBLE_DEVICES"]==""
    assert (run.returncode==1)==(result["status"]=="FAIL")
    # Runtime errors are implementation defects, never expected scientific failures.
    assert not [f for c in result["checks"] for f in c["failures"] if f["assertion"]=="execution_completed"]
    assert output.stat().st_size<1_000_000
    raw_path=tmp_path/result['raw_artifact']['path']
    assert raw_path.name=='quick_raw.json.gz'
    zipped=raw_path.read_bytes()
    assert zipped[4:8]==b'\0'*4  # Gzip mtime is deterministic zero.
    assert not (zipped[3]&8)  # No original filename in the gzip header.
    assert hashlib.sha256(zipped).hexdigest()==result['raw_artifact']['sha256']
    raw=json.loads(gzip.decompress(zipped))
    assert raw['schema_version']=='v016-checks-raw-1'
    amplitude=next(c for c in result['checks'] if c['claim_id']=='C4.amplitude')
    rows=amplitude['data']
    assert len(rows)==13  # Four core, six criterion, three boundary settings.
    assert {row['T'] for row in rows if row['kind']=='criterion_straddle'}=={20,50,100}
    assert amplitude['assertions']['mean_field_rollout_relative_error_all_lambda']['n']==sum(len(row['lambdas']) for row in rows)
    eligible_roots=sum(row['quadratic_crossing_eligible'] for row in rows)
    assert amplitude['assertions']['rollout_zero_crossing_relative_error']['n']==eligible_roots
    assert amplitude['assertions']['limit_predicts_rollout_crossing']['n']==len(rows)
    for idx,row in enumerate(rows):
        assert 'raw_replicates' not in row
        assert row['raw_ref']==f'/checks/C4.amplitude/{idx}'
        samples=np.asarray(raw['checks']['C4.amplitude'][idx]['raw_replicates']['rollout_samples'])
        assert samples.mean(axis=0)==pytest.approx(row['rollout']['mean'])
        scale=max(abs(np.asarray(row['rollout']['mean'])))
        errors=abs(np.asarray(row['mean_field'])-row['rollout']['mean'])/scale
        assert errors==pytest.approx(row['mean_field_rollout_relative_error'])
        if row['measured_rollout_zero_crossing'] is None:
            assert row['mean_field_zero_crossing_relative_error'] is None
            assert row['root_search']['rollout']['search_bound']==row['root_search']['mean_field']['search_bound']
    ranking=next(c for c in result['checks'] if c['claim_id']=='C4.ranking')
    assert ranking['assertions']['parameter_only_derivative_nonpositive']['n']==2*len(ranking['data'])
    damaged=copy.deepcopy(result)
    damaged["checks"][0]["n_assertions"]+=1
    with pytest.raises(ValueError,match="assertion count"):
        checks.validate_results(damaged)
    fail=checks.Check("synthetic.failure","A deliberate false statement")
    fail.data.append(dict(lhs=2,rhs=1));fail.assert_le("inequality",1,0,0)
    bad=dict(schema_version="v016-checks-1",status="FAIL",checks=[fail.result()])
    checks.validate_results(bad)
    bad["checks"][0]["status"]="PASS"
    with pytest.raises(ValueError,match="hidden failure"):
        checks.validate_results(bad)


def test_strict_json_infinity_encoding():
    assert json.dumps(checks.clean(dict(bound=np.inf,raw=np.array([1.,-np.inf]))),allow_nan=False)
    assert checks.clean(np.inf)=="+infinity"


def test_raw_companion_is_lossless_deterministic_and_keeps_failures(tmp_path):
    check=checks.Check('synthetic.failure','Deliberate failure survives serialization')
    check.data.append(dict(mean=[1.,2.],ci_low=[0.,1.],slope=.5,
                           raw_replicates=dict(samples=np.arange(6).reshape(3,2))))
    check.assert_le('inequality',1.,0.,0)
    result=checks.clean(dict(schema_version='v016-checks-1',status='FAIL',checks=[check.result()]))
    a=checks.write_results(result,tmp_path/'a.json')
    b=checks.write_results(result,tmp_path/'b.json')
    assert (tmp_path/'a_raw.json.gz').read_bytes()==(tmp_path/'b_raw.json.gz').read_bytes()
    raw=json.loads(gzip.decompress((tmp_path/'a_raw.json.gz').read_bytes()))
    assert raw['checks']['synthetic.failure']==result['checks'][0]['data']
    assert a['checks'][0]['status']==b['checks'][0]['status']=='FAIL'
    assert a['checks'][0]['failures']==result['checks'][0]['failures']
    assert a['checks'][0]['n_assertions']==1
    assert a['checks'][0]['data'][0]['slope']==.5
    assert 'raw_replicates' not in a['checks'][0]['data'][0]
    assert 'raw_replicates' in result['checks'][0]['data'][0]  # Writer does not mutate its input.
