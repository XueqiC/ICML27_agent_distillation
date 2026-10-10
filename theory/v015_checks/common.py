"""Shared accounting and the explicitly permitted, unchanged C23 import."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools as it
import json
import math
import os
from pathlib import Path
import sys
from fractions import Fraction as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MASTER = 150150
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplconfig')
os.environ['MPLBACKEND'] = 'Agg'
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import scipy
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit

# Initialize matplotlib BEFORE the old module temporarily resets MPLCONFIGDIR.
plt.figure(); plt.close()
spec = importlib.util.spec_from_file_location(
    'independent_c23', ROOT / 'theory/independent_checks/check_v014_independent.py')
c23 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c23)
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplconfig')
law_scalar, law_parity = c23.law_scalar, c23.law_parity
neural_init, neural_forward = c23.neural_init, c23.neural_forward
neural_gradient, neural_step = c23.neural_gradient, c23.neural_step
tangent_matrix, neural_return = c23.tangent_matrix, c23.neural_return
training_sequences, train_sequence, check_C23 = (
    c23.training_sequences, c23.train_sequence, c23.check_C23)

OUT = HERE
QUICK = False
ACTIVE = None


def serial(x):
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [serial(v) for v in x]
    if isinstance(x, np.ndarray):
        return serial(x.tolist())
    if isinstance(x, sp.MatrixBase):
        return serial(x.tolist())
    if isinstance(x, (F, sp.Basic)):
        return str(x)
    if isinstance(x, np.generic):
        return serial(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(data), indent=2, allow_nan=False) + '\n')
    return str(path.relative_to(HERE))


class Group:
    def __init__(self, id, statement, method, kind='exact'):
        global ACTIVE
        ACTIVE = self
        self.id = id
        self.number = int(id[1:]) + (100 if id[0] == 'E' else 0)
        self.seed = [MASTER, self.number]
        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed))
        self.failures = []
        self.row = dict(id=id, statement=statement, method=method, kind=kind,
                        seed=self.seed, n_instances=0, n_assertions=0,
                        worst_residual=0., status='PASS', notes=[], data={})

    def check(self, ok, label, instance=None, residual=0):
        self.row['n_assertions'] += 1
        self.row['worst_residual'] = max(self.row['worst_residual'], abs(float(residual)))
        if not bool(ok):
            self.row['status'] = 'FAIL'
            self.failures.append(serial(dict(group=self.id, seed=self.seed,
                assertion=label, instance=instance, residual=residual)))

    def eq(self, a, b, label, instance=None, tol=0):
        delta = abs(a-b)
        self.check(delta <= tol, label, dict(lhs=a, rhs=b, inputs=instance), delta)

    def le(self, a, b, label, instance=None, tol=0):
        self.check(a-b <= tol, label, dict(lhs=a, rhs=b, inputs=instance), max(0, a-b))

    def count(self, n=1):
        self.row['n_instances'] += n

    def info(self, note):
        if self.row['status'] != 'FAIL':
            self.row['status'] = 'INFO'
        self.row['notes'].append(note)

    def finish(self):
        return dict(self.row, counterexamples=self.failures)


def distribution(rng, n, zero=True):
    a = rng.integers(0 if zero else 1, 10, n)
    if not a.any():
        a[0] = 1
    return np.array([F(int(v), int(a.sum())) for v in a], dtype=object)


def rational_array(rng, shape, low=0, high=13, denominator=12):
    a = rng.integers(low, high, shape)
    return np.array([F(int(v), denominator) for v in a.flat], dtype=object).reshape(a.shape)


def search(group, make, mutate, evaluate, restarts=8, steps=24):
    """Every candidate is checked, including rejected hill-climb proposals."""
    best = None
    count = 0
    for restart in range(restarts):
        x = make()
        score, record = evaluate(x)
        count += 1
        if best is None or score > best['score']:
            best = dict(score=score, instance=record, restart=restart, step=-1)
        for step in range(steps):
            proposal = mutate(x)
            score_new, record_new = evaluate(proposal)
            count += 1
            if score_new > best['score']:
                best = dict(score=score_new, instance=record_new, restart=restart, step=step)
            if score_new >= score:
                x, score = proposal, score_new
    return dict(method='fixed-seed random restarts + greedy hill-climb; all proposals checked',
                restarts=restarts, steps=steps, n_evaluations=count, best=best)


def figure(id, title, draw):
    fig, ax = plt.subplots(figsize=(9, 5))
    draw(fig, ax)
    fig.suptitle(title + '\ncontrolled neural, not Gemma', fontsize=11)
    fig.tight_layout()
    directory = OUT / 'figs'
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(directory / f'{id}.{extension}', dpi=150)
    plt.close(fig)


def flatten(params):
    return np.concatenate([p.reshape(len(p), -1) for p in params], axis=1)


def unflatten(theta, k):
    n = len(theta)
    return [theta[:, :4*k].reshape(n, k, 4), theta[:, 4*k:4*k+4],
            theta[:, 4*k+4:4*k+8], theta[:, -1]]


def logit_jacobian(params):
    """Independent analytic logit derivative, including shared hidden bias."""
    _, h, _ = neural_forward(params)
    n, k, _ = h.shape
    a = (1-h*h)*params[2][:, None, :]
    j = np.zeros((n, k, 4*k+9))
    for context in range(k):
        j[:, context, 4*context:4*context+4] = a[:, context]
    j[:, :, 4*k:4*k+4] = a
    j[:, :, 4*k+4:4*k+8] = h
    j[:, :, -1] = 1
    return j


def optimizer(theta, gradient, m, v, step, eta=.1, opt='Adam', eps=1e-8,
              beta=(.9, .999), correction=True, inside=False, decay=0.):
    """Executed optimizer for all new experiments, including the scalar control."""
    if opt == 'SGD':
        return theta-eta*gradient, m.copy(), v.copy(), dict(gradient=gradient,
            update=-eta*gradient, step=step, m=m, v=v)
    b1, b2 = beta
    m1 = b1*m+(1-b1)*gradient
    v1 = b2*v+(1-b2)*gradient**2
    bc1, bc2 = (1-b1**step, 1-b2**step) if correction else (1., 1.)
    mh, vh = m1/bc1, v1/bc2
    denominator = np.sqrt(vh+eps) if inside else np.sqrt(vh)+eps
    update = -eta*mh/denominator-eta*decay*theta
    return theta+update, m1, v1, dict(gradient=gradient, m=m1, v=v1,
        m_hat=mh, v_hat=vh, bias_correction=[bc1, bc2], denominator=denominator,
        update=update, step=step, decay=decay)


def snapshot(k, warm=0):
    theta = flatten(neural_init(k, 1))
    m = np.zeros_like(theta); v = m.copy()
    history = []
    # Third common law: deterministic soft target 1/4 in every context.
    for t in range(warm):
        labels = np.full((1, k), .25)
        gradient = flatten(neural_gradient(unflatten(theta, k), labels))
        theta, m, v, _ = optimizer(theta, gradient, m, v, t+1, eta=.03)
        history.append(labels[0].tolist())
    return dict(theta=theta, m=m, v=v, step=warm, k=k, pretraining_labels=history)


def run_network(labels, eta=.1, opt='Adam', branch='actual', start=None,
                eps=1e-8, inside=False, correction=True, clip=None,
                reduction='mean', beta=(.9, .999), trace=False):
    n, R, k = labels.shape
    start = snapshot(k) if start is None else start
    theta = np.repeat(start['theta'], n, axis=0)
    m = np.repeat(start['m'], n, axis=0); v = np.repeat(start['v'], n, axis=0)
    initial = theta.copy()
    initial_params = unflatten(initial, k)
    p0, _, logit0 = neural_forward(initial_params)
    J0 = logit_jacobian(initial_params)
    affine = p0.copy(); A = tangent_matrix(k)
    records = []; drifts = []; residuals = []
    for t in range(R):
        params = unflatten(theta, k)
        if branch == 'frozen':
            logits = logit0+np.einsum('nkd,nd->nk', J0, theta-initial)
            gradient = np.einsum('nk,nkd->nd', (expit(logits)-labels[:, t])/k, J0)
        else:
            gradient = flatten(neural_gradient(params, labels[:, t]))
        if reduction == 'sum':
            gradient *= k
        before_clip = gradient.copy()
        if clip is not None:
            gradient *= np.minimum(1, clip/np.maximum(np.linalg.norm(gradient, axis=1), 1e-300))[:, None]
        before = theta.copy()
        theta, m, v, record = optimizer(theta, gradient, m, v, start['step']+t+1,
            eta=eta, opt=opt, eps=eps, inside=inside, correction=correction, beta=beta)
        affine += eta*(labels[:, t]-affine)@A.T
        newparams = unflatten(theta, k)
        J = logit_jacobian(newparams)
        _, _, actual_logits = neural_forward(newparams)
        linear_logits = logit0+np.einsum('nkd,nd->nk', J0, theta-initial)
        drifts.append(np.linalg.norm(J-J0, axis=(1, 2)))
        residuals.append(np.linalg.norm(actual_logits-linear_logits, axis=1))
        if trace:
            records.append(dict(record, theta_before=before, theta_after=theta.copy(),
                labels=labels[:, t], gradient_before_clip=before_clip,
                jacobian_drift=drifts[-1], logit_residual=residuals[-1]))
    if branch == 'frozen':
        probabilities = expit(logit0+np.einsum('nkd,nd->nk', J0, theta-initial))
    else:
        probabilities = neural_forward(unflatten(theta, k))[0]
    return dict(probabilities=probabilities, affine=affine, theta=theta, m=m, v=v,
        step=start['step']+R, trace=records, jacobian_drift=np.array(drifts).T,
        logit_residual=np.array(residuals).T)


def cases():
    answer = []
    for N in (2, 3, 4):
        answer.append(('moment', N, *law_scalar(N)))
    for N in (2, 3):
        answer.append(('parity', N, *law_parity(N)))
    return answer


def task_value(probabilities, task, N=3):
    if task == 'threshold':
        return (probabilities[:, 0] >= .55).astype(float)
    return neural_return(probabilities, task, N)
