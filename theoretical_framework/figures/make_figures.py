#!/usr/bin/env python3
"""Regenerate N01–N20 using committed evidence and exact specified examples.

Run from the repository root with:
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theoretical_framework/figures/make_figures.py
JSON is read from HEAD, never from an untracked/raw result. The only imported
research implementation is the read-only N14 evaluator (and its common/C23
dependencies). No experiment suite, training, network, or GPU is invoked.
"""
from __future__ import annotations

import argparse
import atexit
from fractions import Fraction as F
import importlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
_cache = tempfile.TemporaryDirectory(prefix='.mpl-', dir=HERE)
atexit.register(_cache.cleanup)
os.environ['MPLCONFIGDIR'] = _cache.name


def filesystem_guard(event, args):
    """Fail closed on accidental cache/output writes outside the figure tree."""
    path = None
    if event == 'open':
        name, mode, flags = args
        if isinstance(name, (str, bytes, os.PathLike)):
            candidate = Path(os.fsdecode(name)).resolve()
            if candidate.is_relative_to(ROOT / 'theory/v015_checks/raw'):
                raise RuntimeError('Reading theory/v015_checks/raw is forbidden')
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                path = candidate
    elif event in ('os.mkdir', 'os.remove', 'os.rmdir'):
        # TemporaryDirectory's fd-relative cleanup is confined to our own cache.
        if args[-1] in (-1, None):
            path = Path(os.fsdecode(args[0])).resolve()
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        for name in args[:2]:
            if not Path(os.fsdecode(name)).resolve().is_relative_to(HERE):
                raise RuntimeError('Output outside theoretical_framework/figures is forbidden')
    elif event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
        raise RuntimeError('Figure generation prohibits network access')
    if path is not None and not path.is_relative_to(HERE):
        raise RuntimeError(f'Output outside theoretical_framework/figures is forbidden: {path}')


sys.addaudithook(filesystem_guard)
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import Normalize
from matplotlib.text import Text

# One style block. Every categorical artist receives an explicit color; the
# default cycle contains only black, so an accidental implicit color is obvious.
BLUE, ORANGE, GREEN, PURPLE, SKY, RED, YELLOW, BLACK = (
    '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#222222')
GRAY = '#81878B'
COLORS = (BLUE, ORANGE, GREEN, PURPLE, SKY, RED, YELLOW, BLACK)
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.titlesize': 10, 'lines.linewidth': 1.5,
    'lines.markersize': 4, 'axes.linewidth': .7, 'axes.grid': True,
    'grid.color': '#DCE0E3', 'grid.alpha': .65, 'grid.linewidth': .5,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.prop_cycle': matplotlib.cycler(color=[BLACK]),
    'legend.frameon': False, 'legend.handlelength': 1.8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'svg.hashsalt': 'ICML27-theoretical-framework-N01-N17',
    'savefig.dpi': 300, 'savefig.facecolor': 'white',
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.w_pad': .055,
    'figure.constrained_layout.h_pad': .055,
})

SOURCES = {
    'I': 'theory/independent_checks/results_independent_v014.json',
    'V': 'theory/v015_checks/results_v015_independent.json',
    'P': 'theory/v014_repro/results_v014_rerun.json',
    'T': 'docs/direction/v015_revision/results_v015.json',
}
BASENAMES = (
    'N01_decomposition', 'N02_queries', 'N03_floor', 'N04_static_vs_learning',
    'N05_response', 'N06_duality', 'N07_moment_order', 'N08_rates',
    'N09_approximation', 'N10_regime', 'N11_cold_adam', 'N12_warmth',
    'N13_stability', 'N14_transport', 'N15_adaptive_audit', 'N16_coupling', 'N17_selection',
    'N18_linear_response', 'N19_learner_statistics', 'N20_first_order_audit')
FAMILIES = ('scalar affine powers', 'multivariate parity pullbacks', 'structured sum powers')
FAMILY_NAMES = ('scalar affine powers', 'parity pullbacks', 'structured sum powers')
PROXIES = ('input_moment_error', 'mean_gradient', 'sign_only', 'first_delta',
           'delta_plus_m_v', 'short_tail_1', 'short_tail_4', 'random', 'full_tail_gold')
PROXY_NAMES = ('input moment', 'mean grad', 'sign', 'Δθ', 'Δθ+m,v',
               'short tail 1', 'short tail 4', 'random', 'full tail (gold)')
PROXY_COLORS = {p: BLUE if i < 5 else ORANGE if i < 7 else GRAY for i, p in enumerate(PROXIES)}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def number(x):
    return float(F(x)) if isinstance(x, (str, F)) else float(x)


def array(rows, key):
    return np.array([number(row[key]) for row in rows])


def rational(values):
    return np.array([F(str(v)) for v in np.asarray(values).flat], dtype=object).reshape(np.shape(values))


def only(rows, **filters):
    return [r for r in rows if all(r[k] == v for k, v in filters.items())]


def one(rows, **filters):
    selected = only(rows, **filters)
    require(len(selected) == 1, f'Expected one row for {filters}; found {len(selected)}')
    return selected[0]


def committed_bytes(path):
    return subprocess.check_output(['git', 'cat-file', 'blob', f'HEAD:{path}'], cwd=ROOT)


class Evidence:
    def __init__(self):
        self.documents = {key: json.loads(committed_bytes(path)) for key, path in SOURCES.items()}
        self.claims = {r['id']: r for r in self.documents['I']['claims']}
        self.groups = {r['id']: r for r in self.documents['V']['groups']}
        self.package = self.documents['P']
        self.neural = self.documents['I']['controlled_neural']['details']
        self.captions = {}
        self.notes = []
        self.audits = {}

    def c(self, id):
        return self.claims[id]['details']

    def g(self, id):
        return self.groups[id]['data']


def figure(n, size, footer='Exact finite computations.'):
    fig = plt.figure(figsize=size, layout='constrained')
    grid = fig.add_gridspec(2, n, height_ratios=[1, .055])
    axes = [fig.add_subplot(grid[0, i]) for i in range(n)]
    foot = fig.add_subplot(grid[1, :])
    foot.axis('off')
    foot.text(.5, .25, footer, ha='center', va='center', fontsize=9, color='#555555')
    return fig, axes


def title(ax, text, width=43):
    # Wrapping changes no title words; leave math expressions intact.
    ax.set_title(text if '$' in text else '\n'.join(textwrap.fill(line,width) for line in text.split('\n')), pad=10)


def labels(ax, x, y):
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def legend(ax, handles=None, ncol=1, y=-.25):
    kw = dict(loc='upper center', bbox_to_anchor=(.5, y), ncol=ncol,
              borderaxespad=0, columnspacing=.9, handletextpad=.5)
    return ax.legend(handles=handles, **kw) if handles is not None else ax.legend(**kw)


def note(ax, text, x=.03, y=.97, **kw):
    return ax.text(x, y, text, transform=ax.transAxes, va='top', ha='left', fontsize=9,
                   bbox=dict(facecolor='white', alpha=.92, edgecolor='none', pad=2), **kw)


def identity(ax, low, high):
    ax.plot([low, high], [low, high], '--', color=GRAY, zorder=1, label='y = x')


def handle(color, text, marker=None, ls='-'):
    return Line2D([], [], color=color, marker=marker, ls=ls, label=text)


def group_title(fig, left, right, text):
    """A shared title for two ordinary constrained-layout axes (no dual axis)."""
    spacer='\n'.join(' ' for _ in text.split('\n'))
    left.set_title(spacer,fontsize=10,pad=10)
    right.set_title(spacer,fontsize=10,pad=10)
    shared=fig.text(0,0,text,ha='center',va='bottom',fontsize=10)
    shared.set_in_layout(False)
    # Place after layout. An off-centre Axes title would make constrained_layout
    # reserve the shared title's width against just the narrow left subpanel.
    fig._shared_titles=getattr(fig,'_shared_titles',[])+[(left,right,shared)]


def n01(d):
    prior = rational(['1/2', '1/3', '1/6'])
    values = rational([['.9', '.5', '.6'], ['.4', '.8', '.6'], ['.4', '.5', '.7']])
    likelihood = rational(['.8', '.3', '.5'])
    oracle = prior @ np.max(values, axis=1)
    def bayes(epsilon):
        p = likelihood * (1-epsilon) + (1-likelihood)*epsilon
        return max((prior*p) @ values) + max((prior*(1-p)) @ values)
    optimal = bayes(F(0))
    rep, info = F(1)-oracle, oracle-optimal
    outcomes = [optimal, max(prior @ values), sum(prior @ values)/values.shape[1], bayes(F(1, 4))]
    algorithms = [optimal-outcome for outcome in outcomes]
    totals = [F(1)-outcome for outcome in outcomes]
    require(all(rep+info+a == t for a, t in zip(algorithms, totals)), 'N01 decomposition check failed')
    eps = [F(i, 200) for i in range(101)]
    risk = [oracle-bayes(e) for e in eps]
    require(all(a <= b for a, b in zip(risk, risk[1:])), 'N01 garbling monotonicity failed')
    j = d.package['robust_ambiguity_minimax']
    require(j['primal_regret'] == j['dual_regret'], 'N01 package primal/dual mismatch')
    fig, (a,b,c) = figure(3, (14, 4.5))
    yy = np.arange(len(outcomes)); left = np.zeros(len(yy))
    for vals, col, name in [([rep]*len(yy), GRAY, 'representation'), ([info]*len(yy), BLUE, 'information'),
                             (algorithms, ORANGE, 'algorithm')]:
        floats = [float(v) for v in vals]
        a.barh(yy, floats, left=left, color=col, label=name)
        left += floats
    for y, t in zip(yy, totals): a.text(float(t)+.006, y, f'{float(t):.4f}', va='center')
    a.set_yticks(yy, ['Bayes', 'prior-only', 'uniform random', 'Bayes on garbled D′']); a.invert_yaxis()
    a.set_xlim(0, max(map(float, totals))*1.24); a.set_xlabel('expected regret')
    title(a, '(a) Exact regret decomposition'); legend(a, ncol=1)
    b.plot(list(map(float, eps)), list(map(float, risk)), color=BLUE)
    b.axhline(float(info), color=GRAY, ls='--', label=f'original D: {float(info):.4f}')
    b.axhline(float(risk[-1]), color=ORANGE, ls='--', label=f'no data: {float(risk[-1]):.4f}')
    labels(b, 'flip probability ε', 'information regret'); title(b, '(b) Garbling the transcript'); legend(b)
    weight = np.linspace(0, 1, 201); scale = F('0.2')
    c.plot(weight, float(scale)*np.maximum(weight, 1-weight), color=BLUE, label='worst-case regret (primal)')
    c.plot(weight, float(scale)*np.minimum(weight, 1-weight), '--', color=ORANGE, label='dual objective')
    c.plot(j['policy_mixture'][0], j['primal_regret'], 'o', color=BLACK)
    note(c, f"primal = dual = {j['primal_regret']:g}\n(committed robust ambiguity check)", x=.14, y=.98)
    labels(c, 'mixture weight', 'regret'); title(c, '(c) Minimax duality'); legend(c)
    d.captions['N01'] = ('N01. (a) Exact regret decomposition for four learners on one finite problem: the learner changes only the algorithm term. '
        '(b) Garbling the transcript never lowers the information term; monotone: a garbling can only raise the information term (Prop. 1.4). '
        '(c) Robust ambiguity: min over policy mixtures equals max over priors.')
    return fig


def n02(d):
    rows = d.c('C2')['configurations']; target = F('0.05')
    subset = sorted([r for r in rows if F(r['delta']) == target], key=lambda r: 1/F(r['Delta'])**2)
    fig, (a,b,c) = figure(3, (14, 4.5))
    x = [float(1/F(r['Delta'])**2) for r in subset]
    for key, col, marker, ls, name in [('bound8', BLACK, None, '-', 'bound (8)'),
            ('sprt_expected_n', BLUE, 'o', '-', 'SPRT expected queries'),
            ('min_fixed_n', ORANGE, 's', '', 'smallest successful fixed n'), ('n9', GRAY, None, '--', 'sufficient n (9)')]:
        a.plot(x, array(subset, key), color=col, marker=marker, ls=ls, label=name)
    a.set_xscale('log'); a.set_yscale('log'); note(a, f'δ = {float(target):g}', x=.63, y=.17)
    labels(a, '1 / Δ²', 'number of queries'); title(a, '(a) Queries against inverse squared gap'); legend(a)
    ordered = sorted(rows, key=lambda r: (F(r['delta']), F(r['Delta'])))
    ix = np.arange(len(ordered)); deltas = sorted({F(r['delta']) for r in ordered})
    b.scatter(ix, array(ordered, 'sprt_ratio'), color=BLUE, marker='o', s=20, label='SPRT')
    b.scatter(ix, array(ordered, 'fixed_ratio'), color=ORANGE, marker='s', s=20, label='fixed n')
    b.axhline(1, color=BLACK, lw=1.5, label='lower bound (8)')
    exact = [i for i,r in enumerate(ordered) if r['sprt_ratio'] == 1.0]
    b.scatter(exact, np.ones(len(exact)), s=70, facecolors='none', edgecolors=GREEN, label='SPRT attains the bound')
    centers = []
    for delta in deltas:
        ids = [i for i,r in enumerate(ordered) if F(r['delta']) == delta]
        centers.append(np.mean(ids))
        if ids[0]: b.axvline(ids[0]-.5, color=GRAY, lw=.6)
    b.set_xticks(centers, [f'δ={float(v):g}' for v in deltas], rotation=25)
    b.set_yscale('log'); b.set_ylim(.9, 8)
    labels(b, 'configuration (Δ increasing within group)', 'queries / bound (8)')
    title(b, '(b) Ratios to the lower bound'); legend(b)
    delta = max(F(r['Delta']) for r in rows); m = np.linspace(0, 1, 201)
    plus, minus = 2*float(delta)*(1-m), 2*float(delta)*m
    c.plot(m, plus, color=BLUE, label='world +1'); c.plot(m, minus, color=ORANGE, label='world −1')
    c.plot(m, np.maximum(plus, minus), color=BLACK, lw=2, label='worst case')
    c.axhline(float(delta), color=GRAY, ls='--', label=f'Δ = {delta}'); c.plot(.5, float(delta), 'o', color=GREEN)
    labels(c, 'mean output E p̂', 'expected regret'); title(c, '(c) Passive impossibility'); legend(c, ncol=2)
    d.captions['N02'] = ('N02. (a)–(b) Exact query counts: the SPRT meets the KL lower bound (8), fixed-sample tests and the sufficient n of (9) lie above it, '
                        'and no strategy falls below. (c) Any passive learner has worst-case regret at least Δ.')
    return fig


def n03(d):
    data = d.g('E5'); rows = only(data['curves'], m=8, q_profile='uniform', w_profile='uniform')
    fig, (a,b,c) = figure(3, (14, 4.7), 'Exact hidden-bit calculations and controlled tabular learners; not a real-model result.')
    for key, queries, col, ls, name in [('active',4,BLUE,'-', '+{q} targeted queries'),
          ('passive',4,ORANGE,'--','+{q} extra demonstrations'), ('replay',4,GRAY,':','+{q} replays (gain {gain:g})'),
          ('active',0,BLACK,'-','K demonstrations only')]:
        rr = sorted(only(rows, queries=queries), key=lambda r:r['K'])
        name = name.format(q=rr[0]['queries'], gain=number(rr[0]['replay_gain']))
        if queries==0:name+=rf', $({F(1,2)})({1-F(rr[0]["q"][0])})^K$'
        # Wider dotted line keeps replay visible underneath the coincident baseline.
        a.plot(array(rr,'K'), np.maximum(array(rr,key), 1e-6), color=col, ls=ls,
               lw=2 if key=='replay' else 1.5, marker='s' if key=='replay' else None,
               mfc='white' if key=='replay' else col, markevery=3, label=name)
    a.set_yscale('log'); labels(a, 'K (trusted demonstrations)', 'information-limited risk')
    a.set_xlim(0,max(r['K'] for r in rows))
    note(a,f'committed K = {min(r["K"] for r in rows)}…{max(r["K"] for r in rows)}',x=.03,y=.16)
    title(a, '(a) Independent evidence and query access'); legend(a)
    learners = data['actual_learners']
    for opt,col in [('Adam', BLUE), ('SGD', ORANGE)]:
        for access,mark in [('active','o'),('passive','s'),('replay','^')]:
            rr = only(learners, optimizer=opt, access=access)
            b.scatter(array(rr,'information_floor'), array(rr,'risk'), color=col, marker=mark, s=16, alpha=.45)
    high = max(array(learners,'risk'))*1.04; identity(b, 0, high)
    minimum = min(number(r['risk'])-number(r['information_floor']) for r in learners)
    note(b, f'no point below the diagonal\n(min risk − floor = {minimum:.3g})')
    b.set_xlim(0,high); b.set_ylim(0,high)
    steps = sorted({r['steps'] for r in learners}); require(len(steps)==1, 'N03 learner update counts differ')
    labels(b, 'information floor', f'actual learner risk ({steps[0]} updates)'); title(b, '(b) Learners respect the information floor')
    legend(b, [handle(BLUE,'Adam','o',''),handle(ORANGE,'SGD','o','')]+
           [handle(GRAY,k,m,'') for k,m in [('active','o'),('passive','s'),('replay','^')]]+[handle(GRAY,'information floor',ls='--')], ncol=2)
    rr = only(d.documents['T']['checks']['T11_adaptive_query_KL_enumeration']['rows'], direction=0)
    for r in rr: require(r['binary_KL_lower'] <= r['transcript_KL']+1e-12 <= r['KL_upper']+2e-12, 'N03 KL ordering failed')
    for i,(key,col,name) in enumerate([('binary_KL_lower',BLUE,'binary KL lower'),('transcript_KL',ORANGE,'transcript KL'),('KL_upper',GRAY,'KL upper')]):
        c.bar(array(rr,'K')+(i-1)*.24, array(rr,key), width=.23, color=col, label=name)
    labels(c, 'K', 'nats'); title(c, '(c) Demonstration plus query KL'); legend(c)
    d.captions['N03'] = (f"N03. (a) Exact hidden-bit risk (m={rows[0]['m']}, uniform): targeted queries beat equal numbers of extra demonstrations; replay adds nothing. "
        '(b) Adam/SGD learners never go below the information floor. (c) Demonstration plus query KL always covers kl(1−δ, δ) (A15).')
    return fig


def n04(d):
    fig, (a,b,c) = figure(3, (15, 4.8), d.neural['label'])
    p = np.linspace(0, 1, 201); baseline, slope = F(1,2), F(1,20)
    a.plot(p, float(baseline)+float(slope)*p, color=BLACK,
           label='reference = alternative source\n(identical for every p)')
    students = [F(3,4),F(1,4)]; returns = [baseline+slope*v for v in students]
    for x,y,col,name in zip(students,returns,(BLUE,ORANGE),('reference','alternative')):
        a.plot(float(x),float(y),'o',color=col,ms=6,label=f'trained on {name}: {y}')
    teacher = F(3,5); a.axhline(float(teacher),ls='--',color=GRAY,label='teacher (both sources)')
    a.annotate('',xy=(.88,float(returns[0])),xytext=(.88,float(returns[1])),arrowprops=dict(arrowstyle='|-|',color=BLACK))
    a.text(.9,float(sum(returns)/2),str(returns[0]-returns[1]),va='center')
    labels(a, 'student action-A probability p', 'deployment return'); title(a, '(a) Proposition 5'); legend(a)
    eta = np.linspace(0,1,201)
    for sign,col,name in [(1,BLUE,'J = p²'),(-1,ORANGE,'J = 1 − p²')]:
        b.plot(eta, sign*eta**2/8, color=col, label=name)
    b.axhline(0,color=GRAY,lw=1.5); note(b,'equal label marginals, equal mean\ngradient, equal mean update',y=.19)
    b.set_ylim(-.21,.15); labels(b,'step size η','learning-value gap'); title(b,'(b) Proposition 6'); legend(b, ncol=2)
    configurations = [('SGD',.01),('SGD',.1),('SGD',1.),('Adam',.01),('Adam',.1)]
    groups = [(task,opt,eta) for task in ('Prop5','Prop6') for opt,eta in configurations]
    for j,(updates,col) in enumerate([(1,BLUE),(4,ORANGE)]):
        rr = [one(d.neural['configurations'],task=t,optimizer=o,step_size=e,updates=updates) for t,o,e in groups]
        require(all(r['neural_gap'] > 0 for r in rr),'N04 nonpositive neural gap on log axis')
        c.bar(np.arange(len(groups))+(j-.5)*.37,array(rr,'neural_gap'),width=.35,color=col,label=f'R={rr[0]["updates"]}')
    c.set_xticks(np.arange(len(groups)),[f'{t}\n{o}\n{e:g}' for t,o,e in groups],fontsize=8)
    c.set_yscale('log'); labels(c,'configuration','trained-student return gap (log)')
    title(c,'(c) C23: trained-student gaps (controlled network)'); legend(c,ncol=2,y=-.43)
    d.captions['N04'] = (f'N04. (a) Identical static values for every aliased student, but frequency distillation yields {returns[0]} vs {returns[1]}. '
        '(b) Identical mean updates, but a curvature-dependent gap ±η²/8. (c) In a real tanh network, the statically equivalent sources still give different trained students '
        f'(exact enumeration, R ≤ {max(r["updates"] for r in rr)}).')
    return fig


def telescope(k0, ks, terminal, rounds):
    v = [None]*(rounds+1); v[-1] = terminal
    for t in reversed(range(rounds)): v[t] = k0 @ v[t+1]
    occupancy = np.full(len(terminal),F(0)); occupancy[0]=F(1)
    contributions, bounds = [], []
    for t in range(rounds):
        drift = (k0-ks) @ v[t+1]
        contributions.append(occupancy @ drift); bounds.append(occupancy @ np.abs(drift))
        occupancy = occupancy @ ks
    total = v[0][0]-occupancy @ terminal
    require(sum(contributions)==total,'N05 exact telescoping reproduction failed')
    return contributions, total, sum(bounds)


def n05(d):
    k0 = rational([['.6','.3','.1'],['.2','.6','.2'],['.1','.3','.6']])
    ks = rational([['.4','.4','.2'],['.3','.4','.3'],['.1','.2','.7']])
    contributions,total,bound = telescope(k0,ks,rational(['0','1/2','1']),6)
    cancellation,gap,absolute = telescope(rational([[1,0],[0,1]]),rational([[0,1],[1,0]]),rational([0,1]),2)
    require(absolute == F(d.c('C4')['cancellation_absolute_drift']), 'N05 cancellation JSON check failed')
    fig,(a,b,c) = figure(3,(14,4.5))
    a.bar(range(len(contributions)),list(map(float,contributions)),color=[BLUE if v>=0 else ORANGE for v in contributions])
    a.axhline(float(total),color=BLACK,label=f'final gap = {float(total):.6f}')
    a.axhline(float(bound),color=GRAY,ls='--',label=f'bound (15) = {float(bound):.6f}')
    note(a,r'$\sum_t c_t = L(0) - L(S)$ exactly'+f'\n{float(sum(contributions)):.6f} = {float(total):.6f}',y=.39)
    labels(a,'round t','drift contribution'); title(a,'(a) Telescoping on a 3-state process'); legend(a)
    p,q = rational(['1/2',0,'1/2']),rational([0,1,0])
    responses = rational([[1,1,1],[0,'1/2',1],[0,0,1]])
    for shift,law,col,name in [(-.18,p,BLUE,'P'),(.18,q,ORANGE,'Q')]:
        b.bar(np.arange(len(responses))+shift,[float(law@g) for g in responses],width=.35,color=col,label=name)
    b.set_xticks(range(len(responses)),['g₁','g₂','g₃\nseparating intervention'])
    labels(b,'response','expected final return'); title(b,'(b) One-boundary equivalence (16)'); legend(b,ncol=2)
    c.bar(range(len(cancellation)),list(map(float,cancellation)),color=[ORANGE,BLUE]); c.axhline(0,color=GRAY)
    note(c,f'final gap = {gap}, Σ|drift| = {absolute}\nC4: committed cancellation witness',y=.97)
    c.set_xticks(range(len(cancellation))); labels(c,'round','drift'); title(c,'(c) Cancellation witness')
    d.captions['N05'] = ('N05. (a) The learning-value gap equals the occupancy-weighted sum of per-round response discrepancies (14), bounded by their absolute sum (15). '
        '(b) Two laws are interchangeable for one-boundary replacements exactly when they agree on every response (16). '
        '(c) Per-round discrepancies can cancel, so matching one final mean does not require pointwise matching.')
    return fig


def n06(d):
    y = np.array([F(i,4) for i in range(5)],dtype=object); error=F(1,8)
    g=y*y; h=y-error; p=rational(['1/2',0,0,0,'1/2']); q=rational([0,0,1,0,0])
    require(max(abs(g-h))==error and list((g-h)[[0,2,4]])==[error,-error,error], 'N06 uniform-error/sign reproduction failed')
    require(p@y==q@y==F(1,2) and (p-q)@g==2*error, 'N06 extremal-pair reproduction failed')
    fig,(a,b)=figure(2,(11,4.2))
    x=np.linspace(0,1,401)
    a.plot(x,x*x,color=BLACK,label='g(y) = y²'); a.plot(y.astype(float),g.astype(float),'o',color=BLACK)
    a.plot(x,x-float(error),color=BLUE,label=f'h*(y) = y − {error}')
    a.fill_between(x,x-2*float(error),x,color=BLUE,alpha=.1)
    # The mass strip shares message coordinates, and is explicitly separate from values.
    for law,col,shift,name in [(p,ORANGE,-.008,'P mass'),(q,GREEN,.008,'Q mass')]:
        for xx,mass in zip(y,law):
            if mass:
                a.vlines(float(xx)+shift,-.60,-.60+.18*float(mass),color=col,lw=1.5)
                a.plot(float(xx)+shift,-.60+.18*float(mass),'o',color=col)
                a.text(float(xx)+shift,-.64,str(mass),ha='center',va='top',color=col)
    a.axhline(-.60,color=GRAY,lw=.5); a.set_ylim(-.76,1.1)
    note(a,rf'$E_P\varphi = E_Q\varphi = {p@y}$'+f'\n(P−Q)g = {(p-q)@g} = {int(2*error/error)}·dist∞',x=.02,y=.97)
    labels(a,'message y','value'); title(a,'(a) Worst matched pair attains 2·dist')
    legend(a,[handle(BLACK,'g(y) = y²'),handle(BLUE,f'h*(y), band ±{error}'),handle(ORANGE,'P mass','o'),handle(GREEN,'Q mass','o')],ncol=2)
    residuals=[]
    for i,name in enumerate(FAMILIES):
        rows=one(d.c('C19')['families'],family=name)['curve']
        xx=2*array(rows,'uniform_error'); yy=array(rows,'discrepancy'); residuals.extend(abs(xx-yy))
        b.scatter(xx,yy,color=COLORS[i],marker=MARKERS[i],s=32,label=FAMILY_NAMES[i])
    identity(b,0,.55); b.set_xlim(0,.55); b.set_ylim(0,.55)
    c5=d.claims['C5']
    note(b,f'max |discrepancy − 2·error| = {max(residuals):.2g}\nC5: {c5["n_instances"]} random instances,\nmax residual {c5["details"]["max_duality_residual"]:.2g}')
    labels(b,'2 × best uniform approximation error (LP)','worst-pair discrepancy (primal LP)')
    title(b,'(b) Equality (20) across tested subspaces'); legend(b,ncol=2)
    d.captions['N06'] = ('Thm 8(b): for feature-matched laws, the largest response gap equals twice the uniform distance of the response to the feature span. '
        f'(a) exact {len(y)}-point example; (b) independent primal/dual LPs on all C19 subspaces (and C5’s {c5["n_instances"]} random instances).')
    return fig


def n07(d):
    scalar=d.c('C8')['scalar']; parity=d.package['constant_gap_parity']; c9=d.c('C9')
    closed=[F(math.factorial(r['N']),2**(r['N']-1)*r['N']**r['N']) for r in scalar]
    require(all(abs(F(r['delta']))==v for r,v in zip(scalar,closed)), 'N07 scalar closed-form reproduction failed')
    require(d.claims['C9']['status']=='PASS','N07 full-interface assertion requires C9 PASS')
    fig,(a,b,c)=figure(3,(13,4.2))
    a.semilogy(array(scalar,'N'),abs(array(scalar,'delta')),color=BLUE,marker='o',label='scalar witness (26):\nlower moments equal')
    a.semilogy(array(parity,'N'),array(parity,'exact_value_gap'),color=ORANGE,marker='s',label='parity witness (27):\nall degree < N moments equal')
    a.semilogy(array(scalar,'N'),list(map(float,closed)),color=BLACK,ls='--',label=r'$N!/(2^{N-1}N^N)$')
    labels(a,'order N',r'learning-value gap $|E_P J - E_Q J|$'); title(a,'(a) Each order is needed'); legend(a)
    ranks=c9['grid_ranks']; ks=sorted({r['k'] for r in ranks}); exact=0
    for i,k in enumerate(ks):
        rr=only(ranks,k=k); counts=[math.comb(k+r['N'],r['N']) for r in rr]
        b.scatter(counts,array(rr,'rank'),color=COLORS[i],marker=MARKERS[i],s=30,label=f'k={k}')
        exact+=sum(v==r['rank'] for v,r in zip(counts,rr))
    maxrank=max(r['rank'] for r in ranks); identity(b,1,maxrank*1.3)
    b.set_xscale('log'); b.set_yscale('log'); note(b,f'{exact}/{len(ranks)} equal\n(exact integer rank)')
    labels(b,'C(k+N, N)','rank of monomial matrix on grid'); title(b,'(b) Moment count is exact'); legend(b,ncol=2)
    rows=c9['binary_barriers']; maxd=0
    for i,k in enumerate(sorted({r['k'] for r in rows})):
        rr=only(rows,k=k); full=2**k-1; maxd=max(maxd,full)
        c.plot([0,full,full],[1,1,0],color=COLORS[i],alpha=.7)
        c.scatter(array(rr,'d'),array(rr,'width'),color=COLORS[i],marker=MARKERS[i],s=24,label=f'k={k}')
        c.scatter([full],[0],facecolors='none',edgecolors=COLORS[i],marker='o',s=40)
    c.set_xlim(-.25,maxd+.3); c.set_ylim(-.08,1.1)
    labels(c,'number of nonconstant features d','worst-pair discrepancy'); title(c,r'(c) Fewer than $2^k - 1$ features: error 1')
    handles,_=c.get_legend_handles_labels(); handles.append(Line2D([],[],color=GRAY,marker='o',mfc='none',ls='',label='full interface\n(C9 assertion, PASS)'))
    legend(c,handles,ncol=2)
    d.captions['N07']=(f'Theorems 11–13 in exact arithmetic. (a) scalar witnesses separate at every order with exponentially small gaps (C8, matching the closed form); '
        f'parity laws keep gap {number(parity[0]["exact_value_gap"]):g} (package check). (b) the monomial evaluation rank on the grid equals C(k+N, N) '
        f'in all {len(ranks)} configurations (C9). (c) on binary messages with arbitrary payoffs, any interface with fewer than 2ᵏ − 1 features leaves worst-case error 1 (C9).')
    return fig


def n08(d):
    fig,(a,b,c)=figure(3,(13,4.2))
    curves=d.c('C18')['curves']; family_order=[('scalar',2),('scalar',3),('scalar',4),('parity',2),('parity',3)]
    for i,(family,N) in enumerate(family_order):
        rr=sorted(only(curves,family=family,N=N),key=lambda r:r['exposure']); x=array(rr,'exposure')
        for key,ls in [('iid_gap','-'),('replay_gap','--'),('descendant_gap',':')]:
            a.plot(x,abs(array(rr,key)),color=COLORS[i],ls=ls,marker=MARKERS[i] if key=='iid_gap' else None,
                   label=rf'{family} $N={N}$' if key=='iid_gap' else None)
        if family=='scalar':
            slope=math.log(abs(number(rr[-1]['iid_gap'])/number(rr[0]['iid_gap'])))/math.log(x[-1]/x[0])
            a.annotate(f'slope {slope:.0f}',xy=(x[-2],abs(number(rr[-2]['iid_gap']))),xytext=(-36,7),textcoords='offset points',color=COLORS[i])
    a.set_xscale('log'); a.set_yscale('log'); labels(a,r'training exposure $n$ (blocks)','|learning-value gap|')
    title(a,'(a) Only independent blocks decay:\n'+r'gap $\propto n^{1-N}$')
    handles,_=a.get_legend_handles_labels()
    handles.extend([handle(GRAY,name,ls=ls) for name,ls in [('independent blocks','-'),('replay of one block','--'),('descendants of one block',':')]])
    legend(a,handles,ncol=2)
    cases=d.c('C11')['cumulant_cases']; inv=sp.Symbol('inverse_n'); ns=np.arange(1,221)
    regular=[r for r in cases if 'predicted_coefficient' in r]
    for i,r in enumerate(regular):
        poly=sp.sympify(r['exact_gap_polynomial_in_inverse_n'],locals={'inverse_n':inv})
        coef=sp.Rational(r['predicted_coefficient'])
        require(sp.simplify(poly/inv**(r['r']-1)/coef-1)==0,'N08 cumulant normalization failed')
        ys=np.asarray(sp.lambdify(inv,poly,'numpy')(1/ns),dtype=float)*ns.astype(float)**(r['r']-1)/float(coef)
        b.plot(ns,ys,color=COLORS[i],marker=MARKERS[i],markevery=(i*3,20),label=rf'$k={r["k"]},\ r={r["r"]}$',alpha=.85)
    for i,r in enumerate(r for r in cases if 'vanishing_contraction_r' in r):
        order=r['vanishing_contraction_r']; poly=sp.sympify(r['gap'],locals={'inverse_n':inv})
        ys=sp.lambdify(inv,poly,'numpy')(1/ns)*ns.astype(float)**(order-1)
        b.plot(ns,ys,color=COLORS[i],ls='--',label=rf'zero contraction, $r={order}$')
    b.axhline(1,color=GRAY,ls=':',lw=1.5); b.set_xscale('log'); b.set_ylim(-.12,1.12)
    labels(b,r'independent blocks $n$',r'$n^{r-1}\cdot$gap / predicted coefficient')
    title(b,'(b) Leading term =\n'+r'$\langle D^rJ,\Delta\kappa_r\rangle/(r!\,n^{r-1})$'); legend(b,ncol=2)
    c10=d.c('C10'); sweep=c10['fixed_problem_R_sweep']
    c.plot(array(sweep,'R'),array(sweep,'gap'),color=BLUE,marker='o',label='fixed-problem gap')
    c.plot(array(sweep,'R'),array(sweep,'bound'),color=ORANGE,ls='--',label='fixed-problem bound')
    rr=[r for r in c10['search'] if F(r['bound'])!=0]
    c.scatter(array(rr,'R'),[number(F(r['gap'])/F(r['bound'])) for r in rr],edgecolors=GREEN,facecolors='none',s=25,label='search gap / bound')
    c.axhline(1,color=GRAY,ls=':',lw=1.5); c.set_xscale('log'); c.set_ylim(0,1.1)
    best=F(c10['best_exact_ratio']); note(c,f'max over {d.claims["C10"]["n_instances"]} searched instances:\n{best} ≈ {float(best):.3f}',x=.28,y=.71)
    labels(c,r'rounds $R$','gap or ratio'); title(c,r'(c) $|\mathrm{gap}|\leq C_J\delta_N$, independent of $R$'); legend(c)
    d.captions['N08']=('Theorems 14–15 in exact arithmetic. (a) independent blocks shrink the gap as n¹⁻ᴺ; replaying one block or training on its descendants leaves it constant (C18). '
        f'(b) the leading coefficient equals the first-unmatched-cumulant contraction in all {len(regular)} cases; a zero contraction removes the leading term (C11). '
        f'(c) the round-independent bound holds for R up to {max(r["R"] for r in sweep)} and is nearly tight (C10).')
    return fig


def n09(d):
    rows=d.package['scalar_approximation']; N=16
    coeff=[F(math.comb(2*N,N),4**N)]+[F(2*math.comb(2*N,N-j),4**N) for j in range(1,N+1)]
    fig,(a,b,c)=figure(3,(13,4.2))
    p=np.linspace(0,1,2001); a.plot(p,p**N,color=BLACK,lw=2,label=rf'$p^{{{N}}}$')
    for q,col in zip((4,6,8),('#80B1D3','#377EB8','#084C81')):
        row=one(rows,N=N,q=q); exact=sum(coeff[q+1:])
        require(math.isclose(float(exact),row['Chebyshev_truncation_error'],abs_tol=1e-14),'N09 Chebyshev tail reproduction failed')
        pred=np.polynomial.chebyshev.chebval(2*p-1,list(map(float,coeff[:q+1])))
        require(math.isclose(max(abs(pred-p**N)),float(exact),abs_tol=1e-14),'N09 sampled Chebyshev error differs from exact tail')
        a.plot(p,pred,color=col,label=rf'$h_{{{q}}}$, sup error {row["Chebyshev_truncation_error"]:.3g}')
    parity=one(rows,N=N,q=0)['parity_best_degree_q_error']
    a.axhline(parity,color=GRAY,ls='--',label=f'parity below degree N: constant {parity:g}\n(best error {parity:g})')
    labels(a,'p','value'); title(a,r'(a) $p^{16}\approx$ order-8 polynomial'); legend(a)
    cutoff=1e-3; needed=[]
    ramp=('#86B9D4','#4D92BB','#17678B','#063E63')
    for i,n in enumerate(sorted({r['N'] for r in rows})):
        allrows=sorted(only(rows,N=n),key=lambda r:r['q']); rr=[r for r in allrows if r['q']<n]
        b.semilogy(array(rr,'q'),array(rr,'Chebyshev_truncation_error'),color=ramp[i],marker=MARKERS[i],label=f'N={n}')
        b.semilogy(array(rr,'q'),array(rr,'Hoeffding_upper'),color=ramp[i],ls='--')
        hit=next(r for r in allrows if r['Chebyshev_truncation_error']<cutoff); needed.append((n,hit['q']))
        # q=N has exact zero error; display its threshold crossing, never log(0).
        b.plot(hit['q'],cutoff,'v',color=ramp[i],ms=6)
    b.axhline(parity,color=GRAY,ls=':',label='parity (all N)'); b.axhline(cutoff,color=BLACK,ls=':',label=f'threshold {cutoff:g}')
    c12=d.c('C12'); note(b,f'C12 (N ≤ {max(c12["N_range"])}): exact binomial tail\nmax tail/bound = {c12["exhaustive_tail_best_ratio"]:.3f}',y=.20)
    labels(b,'approximation order q','sup-norm error'); title(b,'(b) Needed order grows like √N')
    handles,_=b.get_legend_handles_labels();handles.extend([handle(GRAY,'Hoeffding upper',ls='--'),handle(GRAY,'first crossing','v','')]);legend(b,handles,ncol=2)
    for i,name in enumerate(FAMILIES):
        family=one(d.c('C19')['families'],family=name); rr=family['curve']
        c.plot(array(rr,'feature_budget'),array(rr,'discrepancy'),color=COLORS[i],label=FAMILY_NAMES[i])
        for r in rr:
            hollow=r['optimality'].startswith('upper bound') and r['discrepancy']!=0
            c.plot(r['feature_budget'],r['discrepancy'],marker=MARKERS[i],color=COLORS[i],mfc='white' if hollow else COLORS[i],ls='')
        span=family['span_dimension_including_constant']-1
        c.axvline(span,color=COLORS[i],ls=':',lw=1.5,ymax=.16+.09*i)
        c.text(span+(.10 if i!=2 else -.10),.04+.04*i,'exact span',color=COLORS[i],ha='left' if i!=2 else 'right')
    c.set_ylim(-.025,.55); labels(c,'number of nonconstant features d','worst-pair discrepancy')
    title(c,r'(c) Parity needs all $2^3 - 1$ directions'); legend(c)
    d.captions['N09']=('Exact degree does not measure approximate information complexity. (a)–(b) Chebyshev truncation of pᴺ has error equal to a binomial tail ≤ 2 exp(−q²/N); '
        f'error {cutoff:g} needs only q = {", ".join(str(q) for _,q in needed)} for N = {", ".join(str(n) for n,_ in needed)} '
        f'(package check; C12 for N ≤ {max(c12["N_range"])}). Parity keeps error {parity:g} below degree N. '
        '(c) On the same grid and degree, scalar and structured families are exactly resolved by 3 features, parity only by 7 '
        '(C19; hollow markers = upper bounds from the best subspace found). Zero-error endpoint threshold crossings are marked at the threshold, outside the log-error curves.')
    return fig


def n10(d):
    fig,(a,b,c)=figure(3,(13,4), 'Exact finite computations and a controlled small network; not a real-model result.')
    rows=only(d.neural['configurations'],optimizer='SGD'); etas=sorted({r['step_size'] for r in rows}); rounds=sorted({r['updates'] for r in rows})
    heat=np.empty((len(etas),len(rounds)))
    for i,eta in enumerate(etas):
        for j,R in enumerate(rounds):
            rr=only(rows,step_size=eta,updates=R); heat[i,j]=math.log10(max(abs(r['deviation_from_local_affine']) for r in rr))
    im=a.pcolormesh(np.arange(len(rounds)+1)-.5,np.arange(len(etas)+1)-.5,heat,cmap='Blues',shading='flat')
    a.invert_yaxis(); a.grid(False)
    for i,eta in enumerate(etas):
        for j,R in enumerate(rounds):
            rr=only(rows,step_size=eta,updates=R)
            a.text(j,i,f'{sum(r["local_affine_agreement"] for r in rr)}/{len(rr)} agree',ha='center',va='center',color='white' if im.norm(heat[i,j])>.55 else BLACK)
    a.set_xticks(range(len(rounds)),rounds); a.set_yticks(range(len(etas)),[f'{v:g}' for v in etas])
    cb=fig.colorbar(im,ax=a,orientation='horizontal',pad=.10,shrink=.9)
    cb.solids.set_rasterized(False)
    cb.set_label('log10 max |actual − affine| gap deviation')
    labels(a,'updates R\nagreement margin = 1e-4 + 0.1|pred| + 4 SE','step size η'); title(a,'SGD: actual gap vs tangent-affine prediction')
    rows=d.c('C13')['logistic']
    for i,eta in enumerate(sorted({r['eta'] for r in rows})):
        rr=only(rows,eta=eta); b.scatter(array(rr,'bound'),array(rr,'remainder'),color=COLORS[i],marker=MARKERS[i],s=20,label=f'η={eta:g}')
    lo=min(array(rows,'remainder'))*.7; hi=max(array(rows,'bound'))*1.2; identity(b,lo,hi)
    b.set_xscale('log'); b.set_yscale('log'); note(b,f'max remainder/bound = {d.claims["C13"]["best_ratio"]:.3f}')
    labels(b,'η³ bound','remainder'); title(b,'Expansion (35): remainder vs η³ bound'); legend(b,ncol=3)
    inp=d.g('V7')['linear_failure']['inputs']; eta,L,C=(F(inp[k]) for k in ('eta','L','C'))
    errors=[]; e=F(0); horizon=range(1,13)
    for R in horizon: e=(1+eta*L)*e+C*eta**2; errors.append(e)
    chosen=inp['R']; require(errors[chosen-1]==F(d.g('V7')['linear_failure']['e']), 'N10 recurrence reproduction failed')
    linear=[R*C*eta**2 for R in horizon]
    c.plot(list(horizon),list(map(float,errors)),color=BLUE,marker='o',label='geometric recursion (equality)')
    c.plot(list(horizon),list(map(float,linear)),color=ORANGE,ls='--',label='R C η²')
    c.annotate(rf'$e_{{{chosen}}} = {errors[chosen-1]} > {linear[chosen-1]}$',xy=(chosen,float(errors[chosen-1])),xytext=(.06,.47),textcoords='axes fraction',arrowprops=dict(arrowstyle='->',color=GRAY))
    amp=d.c('C13')['amplification']
    note(c,'squaring witness: gaps\n'+', '.join(r['gap'] for r in amp[:2])+',\n'+', '.join(r['gap'] for r in amp[2:])+f'\nat R={min(r["R"] for r in amp)}…{max(r["R"] for r in amp)}')
    labels(c,'rounds R','accumulated error'); title(c,'Error accumulation needs stability'); legend(c)
    d.captions['N10']=('Left: the tangent affine-SGD prediction matches the actual controlled network for η≤0.1 and short horizons, and fails at η≥1. '
        'Middle: the third-order remainder of (35) stays below its bound. Right: per-step errors accumulate geometrically; the linear-in-R reading fails without a stability condition.')
    return fig


def n11(d):
    fig,(a,b,c)=figure(3,(13,4), 'Controlled small network and exact optimizer-memory witness; not a real-model result.')
    cells=[('moment',3,.1,1),('moment',3,.1,4),('moment',3,.01,4),('parity',2,.1,4),('moment',4,.1,4)]
    rr=[one(d.g('E2')['comparisons'],task=t,N=n,eta=e,R=r) for t,n,e,r in cells]
    keys=('actual_Adam','frozen_Adam','actual_SGD','tangent_affine_SGD')
    for i,key in enumerate(keys):
        bars=a.bar(np.arange(len(rr))+(i-1.5)*.2,[r['gaps'][key] for r in rr],width=.19,color=COLORS[i],label=key.replace('_',' '))
        if i<2:
            for bar,r in zip(bars,rr):
                negative=bar.get_height()<0
                a.annotate(f'{r["gaps"][key]:.3f}',xy=(bar.get_x()+bar.get_width()/2,bar.get_height()),
                           xytext=(0,-5-11*i if negative else 3+5*i),textcoords='offset points',
                           rotation=0 if negative else 90,ha='center',va='top' if negative else 'bottom',fontsize=8)
    a.axhline(0,color=GRAY); a.margins(y=.32)
    a.set_xticks(range(len(rr)),[f'{r["task"]}\nN={r["N"]}\nη={r["eta"]:g}\nR={r["R"]}' for r in rr])
    a.set_ylabel(r'final return gap $E_P - E_Q$'); title(a,'Same moments, different Adam response'); legend(a,ncol=2,y=-.38)
    rows=only(d.g('E1')['ablations'],task='moment',N=3)
    variants=('baseline','inside_sqrt','epsilon_1e-06','epsilon_0.0001','epsilon_0.01','epsilon_0.1','no_bias_correction','global_clip_0.1','sum_reduction','warm_5','shift_support')
    for i,variant in enumerate(variants):
        if variant in ('warm_5','shift_support'): b.axhspan(i-.48,i+.48,color=YELLOW,alpha=.18,zorder=0)
    for i,(R,col) in enumerate([(1,BLUE),(4,ORANGE)]):
        vals=[one(rows,variant=v,R=R)['gap'] for v in variants]
        b.barh(np.arange(len(variants))+(i-.5)*.35,vals,height=.33,color=col,label=f'R={R}')
    b.set_yticks(range(len(variants)),[v.replace('epsilon_','ε=').replace('_',' ') for v in variants]); b.invert_yaxis(); b.axvline(0,color=GRAY)
    b.set_xlabel('final return gap'); title(b,'One-factor ablations (moment N=3, η=0.1)'); legend(b,ncol=2)
    data=d.g('V4'); exact=data['exact']
    for i,r in enumerate(exact):
        delta=sp.sympify(r['delta2']); first=number(r['theta1']); values=[0,first,first+float(delta)]
        c.plot(range(len(values)),values,color=COLORS[i],marker=MARKERS[i],label=f'G₁={r["first"]}')
        c.annotate(f'm̂₂={r["mhat2"]}, v̂₂={r["vhat2"]}',xy=(2,values[-1]),xytext=(-8,12 if i==0 else -22),textcoords='offset points',ha='right',color=COLORS[i])
    found=[r for r in data['grid'] if r['found']]
    require(len(found)/len(data['grid'])==data['grid_fraction'],'N11 beta-grid fraction check failed')
    require(all(F(r['beta1'])>0 for r in found),'N11 beta-grid sign check failed')
    note(c,f'β-grid witnesses: {len(found)}/{len(data["grid"])} cells,\nall with β₁>0',x=.1,y=.91)
    c.set_xticks([0,1,2]); c.set_ylim(-1.45,.2); labels(c,'step t','θ'); title(c,'Same signs, different memory (A4)'); legend(c,ncol=2)
    d.captions['N11']=('Left: cold Adam separates laws whose first two gradient moments agree, and a frozen-Jacobian Adam reproduces the gap while SGD shows small residual gaps. '
        'Middle: removing sign disagreement (shifted support) nearly removes the gap, and warm moments remove most of it. '
        'Right: two histories with identical sign sequences take opposite second steps because m retains magnitudes.')
    return fig


def n12(d):
    data=d.g('E8'); fig,(a,b,c)=figure(3,(13,4.8), 'Controlled small network, not a real-model result; W1 is an empirical consistency check.')
    horizons=(1,4,40); tasks=(('moment',3,'-'),('parity',2,'--'))
    for task,N,ls in tasks:
        for i,R in enumerate(horizons):
            rr=sorted(only(data['gaps'],task=task,N=N,eta=.1,R=R),key=lambda r:r['W'])
            a.plot(array(rr,'W'),[r['gaps']['actual_Adam'] for r in rr],color=COLORS[i],ls=ls,marker='o' if task=='moment' else 's')
            cold=one(rr,W=0)
            a.axhline(cold['gaps']['tangent_affine_SGD'],color=COLORS[i],ls=':',lw=1.5,alpha=.75)
    a.set_xscale('symlog',linthresh=1); a.set_xticks(data['W_grid'],[str(w) for w in data['W_grid']])
    a.set_xlim(min(data['W_grid']),max(data['W_grid']))
    annotations=[]
    for R in horizons[:2]:
        rr=sorted(only(data['gaps'],task='moment',N=3,eta=.1,R=R),key=lambda r:r['W'])
        annotations.append(f'R={R}: {rr[0]["gaps"]["actual_Adam"]:.4f} → {rr[-1]["gaps"]["actual_Adam"]:.4f}')
    note(a,f'{rr[0]["task"]} N={rr[0]["N"]}\n'+'\n'.join(annotations),x=.30,y=.97)
    a.set_ylim(top=max(r['gaps']['actual_Adam'] for r in only(data['gaps'],eta=.1) if (r['task'],r['N']) in [(t,n) for t,n,_ in tasks])*1.45)
    labels(a,'lr = 0 warm-up W','actual Adam return gap'); title(a,'Gap vs lr=0 warm-up W')
    legend(a,[handle(COLORS[i],f'R={R}') for i,R in enumerate(horizons)]+
           [handle(GRAY,f'{task} N={N}',ls=ls) for task,N,ls in tasks]+[handle(GRAY,'SGD comparator',ls=':')],ncol=2)
    steps=only(data['W3']['per_step'],task='moment',N=3,source='P',eta=.01)
    for i,W in enumerate((0,5,20,50)):
        rr=sorted([r for r in only(steps,W=W) if math.isfinite(float(r['quantiles'][2])) and r['prior_gradient_evaluations']>0 and float(r['quantiles'][2])>0],key=lambda r:r['prior_gradient_evaluations'])
        x=array(rr,'prior_gradient_evaluations'); y=np.array([float(r['quantiles'][2]) for r in rr])
        b.plot(x,y,color=COLORS[i],label=f'W={W}')
        if W==0: coeff=float(np.dot(1/x,y)/np.dot(1/x,1/x)); guide_range=(min(x),max(x))
    x=np.geomspace(*guide_range,200); b.plot(x,coeff/x,'--',color=BLACK,label='least-squares c/x')
    fit=one(data['W3']['fits'],task='moment',N=3,source='P',eta=.01,W=0)['fit']
    require(math.isclose(coeff,fit['c'],rel_tol=1e-12),'N12 inverse-count least-squares reproduction failed')
    b.set_xscale('log'); b.set_yscale('log')
    fits=warmth_fits(d)
    annotation=(f'moment N=3, source P: fitted slope {fit["loglog_slope"]:.2f}, R² {fit["R2"]:.2f}\n'
                f'median over {len(fits)} task/source series (η=0.01, W=0): '
                f'slope {np.median([r["loglog_slope"] for r in fits]):.2f}, R² {np.median([r["R2"] for r in fits]):.3f}')
    b.text(.5,-.51,annotation.replace('-','−'),transform=b.transAxes,ha='center',va='top',fontsize=8)
    labels(b,'prior gradient evaluations W+s','median warmth κ'); title(b,'Warmth decays like 1/(W+s)'); legend(b,ncol=3)
    panels=data['W1']['panels']; etas=sorted({p['eta'] for p in panels}); task_names=sorted({p['task'] for p in panels})
    allx,ally=[],[]
    for p in panels:
        xx=np.asarray(p['r']); yy=np.asarray(p['pointwise_residual'])
        require(np.all(xx>0) and np.all(yy>0),'N12 W1 has nonpositive log-scatter values')
        require(np.all(yy<=xx*(1+1e-12)),'N12 W1 residual exceeds bound')
        c.scatter(xx,yy,color=COLORS[etas.index(p['eta'])],marker=MARKERS[task_names.index(p['task'])],s=22,alpha=.7)
        allx.extend(xx); ally.extend(yy)
    low=min(min(allx),min(ally))*.6; high=max(allx)*1.4; identity(c,low,high)
    c.set_xscale('log'); c.set_yscale('log')
    point_ratio=max(y/x for x,y in zip(allx,ally))
    note(c,f'{data["W1"]["n_law_pairs"]:,} law pairs,\nmax gap/bound {data["W1"]["best_ratio"]:.3f}\npointwise max residual/r {point_ratio:.4f}')
    labels(c,'bound r','pointwise residual'); title(c,'W1: residual vs bound')
    legend(c,[handle(COLORS[i],f'η={eta:g}','o','') for i,eta in enumerate(etas)]+
           [handle(GRAY,task,MARKERS[i],'') for i,task in enumerate(task_names)],ncol=2)
    d.captions['N12']=('Left: an lr = 0 moment warm-up reduces the gap between equal-moment sources at short horizons '
        f'(R={horizons[0]},{horizons[1]}). At R={horizons[-1]} warm-up no longer drives the gap to zero; where the residual is largest, the SGD comparator is nonzero too. '
        'Middle: median warmth is compared with a fitted 1/(W+s) curve regardless of how many prior evaluations were warm-up. '
        f'Right: the W1 pointwise residual never exceeds its bound r and comes within {100*(1-point_ratio):.2f}% of it. '
        f'The separate law-pair max gap/bound is {data["W1"]["best_ratio"]:.3f}; these are empirical consistency checks, not a uniform neural certificate.')
    return fig


def n13(d):
    fig,(a,b)=figure(2,(10,4), 'Exact finite computations, not neural results.')
    rows=d.c('C22')['configurations']; alphas=sorted({F(r['alpha']) for r in rows},reverse=True)
    eps=np.r_[0,np.geomspace(1e-5,2,500)]
    for i,alpha in enumerate(alphas):
        rr=[r for r in rows if F(r['alpha'])==alpha]
        a.plot(eps,np.minimum(1,eps/float(alpha)),color=COLORS[i],label=f'α = {alpha}')
        a.plot(array(rr,'epsilon'),array(rr,'discrepancy'),ls='',marker=MARKERS[i],ms=8,color=COLORS[i],mfc='none',mew=1.3)
        # All right endpoints equal one; stagger direct labels with leader lines.
        a.annotate(f'α = {alpha}',xy=(2,1),xytext=(.75,.74-.12*i),textcoords='axes fraction',color=COLORS[i],arrowprops=dict(arrowstyle='-',color=COLORS[i],lw=.6))
    a.set_xscale('symlog',linthresh=1e-4);a.set_xlim(0,2.2);a.set_ylim(0,1.05)
    note(a,'markers: LP over law pairs\n'+r'lines: $\min\{1,\varepsilon/\alpha\}$',y=.28)
    labels(a,'tolerance ε','Wε(g)'); title(a,'C22: exact representation, small scale ⇒ weak certificate',width=50);legend(a,ncol=4)
    rr=d.package['noisy_feature_duality']['explicit_example']; eps=np.linspace(0,1.6,401)
    baseline=one(rr,tolerance=0.0)['closed_form']; transition=1-baseline
    b.axvspan(0,transition,color=BLUE,alpha=.08); b.axvspan(transition,1.6,color=ORANGE,alpha=.06)
    b.plot(eps,np.minimum(1,baseline+eps),color=GRAY,label=rf'$\min\{{1, {baseline:g}+\varepsilon\}}$')
    b.plot(array(rr,'tolerance'),array(rr,'primal'),'o',color=BLUE,label='primal')
    b.plot(array(rr,'tolerance'),array(rr,'dual'),'s',color=ORANGE,mfc='none',ms=7,label='dual')
    used=next(r['coefficient'][0] for r in rr if r['tolerance']<transition)
    dropped=next(r['coefficient'][0] for r in rr if r['tolerance']>transition)
    b.text(transition/2,.96,f'uses feature\n(b={used:g})',transform=b.get_xaxis_transform(),va='top',ha='center')
    b.text((transition+1.6)/2,.96,f'ignores feature (b={dropped:g})',transform=b.get_xaxis_transform(),va='top',ha='center')
    c14=d.claims['C14']; note(b,f'C14: {c14["n_instances"]} random instances,\nmax primal–dual gap {c14["details"]["max_duality_residual"]:.2g}',x=.34,y=.28)
    b.set_xlim(0,1.6);b.set_ylim(0,1.2);labels(b,'tolerance ε','discrepancy')
    title(b,'Primal = dual; the optimizer drops the feature when ε outweighs its gain',width=49);legend(b,ncol=3)
    d.captions['N13']=('N13. Left: tolerant discrepancy for φ = αg (C22). LP values coincide with min{1, ε/α}. '
        'Right: the explicit example Y = {0, ½, 1}, φ(y) = y, g = 1{y = 1}; primal, dual and closed form coincide. Exact finite computations, not neural results.')
    return fig


def reproduce_transport(d):
    paths=['theory/v015_checks/common.py','theory/v015_checks/mathematics.py',
           'theory/independent_checks/check_v014_independent.py']
    # Imports execute working-tree source. Refuse it unless byte-identical to HEAD.
    for path in paths:
        require((ROOT/path).read_bytes()==committed_bytes(path),f'N14 import is not committed/unmodified: {path}')
    oldcwd=Path.cwd(); oldpath=sys.path.copy(); oldconfig=os.environ['MPLCONFIGDIR']
    try:
        os.chdir(ROOT/'theory/v015_checks'); sys.path.insert(0,str(Path.cwd()))
        common=importlib.import_module('common'); mathematics=importlib.import_module('mathematics')
        g=common.Group('V1','figure','figure'); pairs=[]; scores=[]
        for _ in range(2000):
            score,record=mathematics.evaluate_markov(g,mathematics.markov_instance(g.rng))
            scores.append(score);pairs.append((number(record['bound']),number(record['gap'])))
        expected=d.g('V1')['worst_random']['ratio']
        require(max(scores)==expected,f'N14 reproduction failed: max(score)={max(scores):.17g}, V1 worst_random.ratio={expected:.17g}; N14 will not be drawn')
        require(not g.failures,'N14 finite evaluator assertions failed; N14 will not be drawn')
        require(all(y<=x for x,y in pairs),'N14 recomputed gap exceeds certificate; N14 will not be drawn')
        print(f'N14 reproduction PASS: {len(pairs):,} random instances, max ratio {max(scores):.17g}',flush=True)
        return np.array(pairs)
    finally:
        os.chdir(oldcwd);sys.path[:]=oldpath;os.environ['MPLCONFIGDIR']=oldconfig


def n14(d):
    # Reproduce before allocating the figure or saving either of its outputs.
    pairs=reproduce_transport(d); v1=d.g('V1'); e4=d.g('E4')
    fig,(a,b,c)=figure(3,(15,4.5),'Exact finite computations and a controlled small network; not a real-model result.')
    a.scatter(pairs[:,0],pairs[:,1],color=BLUE,s=16,alpha=.4,label=f'{len(pairs):,} random processes')
    witness=v1['sharp_transport_witness']; x,y=number(witness['bound']),number(witness['gap'])
    a.scatter([x],[y],color=RED,marker='D',s=38,label=f'sharp transport witness\n(D={number(witness["D"]):g}, a={number(witness["local"][0][0]["a"]):g})')
    lim=max(pairs[:,0].max(),x)*1.05;identity(a,0,lim);a.set_xlim(0,lim);a.set_ylim(0,lim)
    group=d.groups['V1']; violations=len(group.get('counterexamples',[]))
    require(group['status']=='PASS' and group['worst_residual']==0 and violations==0,'N14 committed V1 summary reports a failure')
    note(a,f'Committed V1 summary:\n{group["n_instances"]:,} instances, {violations} violations;\nmax ratio {group["best_ratio"]:g}')
    labels(a,'bound (39a)','exact gap');title(a,'V1: (39a) ≥ exact gap');legend(a)
    rr=sorted(e4['observations'],key=lambda r:r['diagnostic_bound']);index=np.arange(len(rr))
    gold=abs(array(rr,'gold_contrast'));bounds=array(rr,'diagnostic_bound')
    error=abs(array(rr,'anchor_prediction')-array(rr,'gold_contrast'));floor=1e-4
    b.plot(index,np.maximum(gold,floor),'o',color=BLUE,label='|gold contrast|')
    b.plot(index,np.maximum(bounds,floor),'-s',color=ORANGE,label='B(D+a+2α)+2τ')
    b.plot(index,np.maximum(error,floor),'^',color=GREEN,mfc='none',label='zero-transport anchored error')
    b.axhline(e4['prediction_tolerance'],color=GRAY,ls='--',label='prediction tolerance')
    covered=int(np.sum(gold<=bounds));missed=int(np.sum(error>e4['prediction_tolerance']))
    note(b,f'|gold| ≤ bound at {covered}/{len(rr)} states;\nzero-transport prediction misses {missed}/{len(rr)}',y=.98)
    b.set_yscale('log');b.set_ylim(floor,bounds.max()*5)
    labels(b,'state index (sorted by diagnostic bound)','contrast, bound, or prediction error');title(b,'E4: transport must be measured');legend(b)
    # C contains independent strip and bar axes, never a second y-axis.
    spec=c.get_subplotspec();c.remove();sub=spec.subgridspec(1,2,width_ratios=[.8,1.25],wspace=.10)
    left=fig.add_subplot(sub[0,0]);right=fig.add_subplot(sub[0,1])
    rng=np.random.default_rng(140014)
    for i,(key,col) in enumerate([('a',BLUE),('alpha',ORANGE)]):
        vals=array(rr,key);left.scatter(i+rng.uniform(-.16,.16,len(vals)),vals,color=col,s=16,alpha=.55)
        median=np.median(vals);left.plot([i-.23,i+.23],[median,median],color=BLACK,lw=2)
        left.text(.5,median+.028,f'{median:.4g}',ha='center')
    medianD=float(np.median([r['D'] for r in e4['anchors']]))
    left.axhline(medianD,color=GRAY,ls='--');left.set_xticks([0,1],['a','α']);left.set_ylabel('transport / feature drift')
    note(left,f'median Dₓ\n{medianD:.3f}',x=.08,y=.97)
    reg=v1['regression']; keys=['gap','corrected_transport_bound','corrected_exit_bound','old_bound']
    vals=[number(reg[k]) for k in keys]; cap=number(reg['gap'])
    bars=right.bar(range(len(vals)),np.minimum(vals,cap),color=[BLACK,BLUE,GREEN,ORANGE])
    bars[-1].set_hatch('///');right.plot([1],[cap],marker='_',color=RED,ms=11,mew=2)
    for i,v in enumerate(vals):
        label=f'{v:.4f}↥' if v>cap else f'{v:.4g}'
        right.text(i,min(v,cap)+(.10 if i==1 else .035),label,ha='center',fontsize=8)
    right.set_xticks(range(len(vals)),['exact\ngap','charge\na','charge\nρ','term\ndeleted'],fontsize=8)
    right.set_ylim(0,cap*1.6)
    right.text(.98,.97,'invalid without\ntransport',transform=right.transAxes,ha='right',va='top',color=RED,fontsize=8)
    right.set_ylabel('gap or certificate (cap marked)')
    group_title(fig,left,right,'Uncovered state: charge a or ρ')
    d.captions['N14']=(f'N14. (A) Exact gap versus the transport-aware certificate on {len(pairs):,} random finite training processes (V1); no point lies above y = x. '
        f'The annotation summarizes all {group["n_instances"]:,} committed V1 instances. (B) In a controlled network (E4), the measured-transport local bound contains '
        f'the gold full-tail contrast at all {len(rr)} reached states, while zero-transport anchor predictions miss {missed}. '
        '(C) Transport and feature drift at reached states, and the uncovered-state witness. Exact finite computations and a controlled small network; not a real-model result.')
    return fig


def bernstein_n(radius, variance, M, delta):
    log=math.log(4*M/delta)
    def valid(n): return math.sqrt(2*variance*log/n)+14*log/(3*(n-1)) <= radius
    high=2
    while not valid(high): high*=2
    low=2
    while low<high:
        mid=(low+high)//2
        if valid(mid): high=mid
        else: low=mid+1
    require(valid(low) and (low==2 or not valid(low-1)), 'N15 smallest Bernstein n check failed')
    return low


def pixel_gaps(fig, ax, bars, pixels=2):
    """Inset each stacked segment by half a 300-dpi output pixel gap.

    Counts/endpoints stay in data coordinates. A post-layout callback sets the
    inset after constrained_layout has settled; PDF uses the same physical gap.
    """
    fig._pixel_gap_specs=getattr(fig,'_pixel_gap_specs',[])+[(ax,bars,pixels)]


def n15(d):
    fig,(a,b,c)=figure(3,(15,4.5),'Finite computations and a controlled small network; not a real-model result.')
    rr=d.c('C16')['configurations'];x=np.arange(len(rr))
    a.errorbar(x,array(rr,'gain'),yerr=2*array(rr,'SE'),fmt='o',color=BLUE,capsize=3,label='gain ± 2 SE')
    for key,col,ls,name in [('minus_delta',ORANGE,'-','bound (44): −δ'),('lcb',GREEN,'--','bound (45): E ΣB − 2δ')]:
        vals=[-r['delta'] if key=='minus_delta' else r['expected_sum_B']-2*r['delta'] for r in rr]
        for j,v in enumerate(vals):a.plot([j-.25,j+.25],[v,v],color=col,ls=ls,label=name if j==0 else None)
    a.axhline(0,color=GRAY);a.set_ylim(-.85,.22)
    a.set_xticks(x,[f'R={r["R"]}, M={r["M"]}\nn={r["n"]}\nδ={r["delta"]:g}' for r in rr],fontsize=8)
    last=rr[-1];a.annotate(f'selects the good candidate:\n{last["gain"]:+.3f}',xy=(x[-1],last['gain']),xytext=(.30,.90),textcoords='axes fraction',arrowprops=dict(arrowstyle='->',color=GRAY))
    replicas={r['replicas'] for r in rr};require(len(replicas)==1,'N15 differing training replica counts')
    a.set_ylabel('measured gain or guarantee');title(a,f'C16: gain ≥ −δ and ≥ E ΣB − 2δ ({rr[0]["replicas"]//1000}k runs each)');legend(a,y=-.39)
    e7=d.g('E7'); h=e7['Hoeffding']; eb=e7['empirical_Bernstein'];M,delta=h['M'],h['delta'];variance=eb['max_pilot_sample_variance']
    require(eb['formula']=='sqrt(2*S^2*log(4M/delta)/n)+14*log(4M/delta)/(3*(n-1))','N15 Bernstein formula changed; review solver before drawing')
    radius=np.linspace(.05,.3,151);hoeffding=2*math.log(2*M/delta)/radius**2
    bernstein=[bernstein_n(v,variance,M,delta) for v in radius]
    hn=math.ceil(2*math.log(2*M/delta)/h['error']**2);bn=bernstein_n(h['error'],variance,M,delta)
    require(hn==h['n'] and bn==eb['n_forecast'],'N15 committed audit-cost reproduction failed')
    b.plot(radius,hoeffding,color=BLUE,label='Hoeffding (sufficient)')
    b.plot(radius,bernstein,color=ORANGE,ls='--',label='empirical Bernstein\n(forecast from pilot variance)')
    for value,col,offset in [(hn,BLUE,(16,18)),(bn,ORANGE,(16,-23))]:
        b.plot(h['error'],value,'o',color=col);b.annotate(f'({h["error"]:g}, {value})',xy=(h['error'],value),xytext=offset,textcoords='offset points',color=col)
    b.set_yscale('log');labels(b,'target radius e','replicas per candidate n');title(b,f'Audit cost at one state, M = {M}, δ = {delta:g}');legend(b)
    variance_rows=e7['variance_decomposition']; yy=np.arange(len(variance_rows));left=np.zeros(len(yy));containers=[]
    for key,col in [('block',BLUE),('suffix',ORANGE),('deployment',GREEN)]:
        vals=array(variance_rows,key);containers.append(c.barh(yy,vals,left=left,color=col,label=key));left+=vals
    for y,r in zip(yy,variance_rows):
        require(math.isclose(r['block']+r['suffix']+r['deployment'],r['total'],abs_tol=1e-12),'N15 variance decomposition mismatch')
        c.text(r['total']+.002,y,f'{r["deployment"]/r["total"]:.0%}',va='center')
    c.set_yticks(yy,[str(r['candidate']) for r in variance_rows]);c.invert_yaxis();c.set_xlim(0,max(left)*1.24)
    labels(c,'paired-contrast variance','candidate');title(c,'Paired-contrast variance: deployment dominates (controlled network)');legend(c,ncol=3)
    pixel_gaps(fig,c,list(zip(*containers)))
    d.captions['N15']=(f'N15. (A) Adaptive actual-state auditing on a finite process (C16): the measured gain (±2 SE, {rr[0]["replicas"]:,} trainings per configuration) '
        f'never falls below −δ or E ΣB − 2δ. (B) Replicas per candidate needed for radius e. Hoeffding’s {hn} at e = {h["error"]:g} is sufficient; '
        'the empirical-Bernstein curve is a forecast from pilot variance. (C) Exact variance decomposition of paired contrasts in the controlled network (E7). '
        'Finite computations and a controlled small network; not a real-model result.')
    return fig


def n16(d):
    fig,(a,b)=figure(2,(11,4.5),'Exact finite computations and a controlled small network; not a real-model result.')
    rows=d.neural['regrouping']; ramps={'SGD':('#8ABBDD','#508DB8','#226594','#084C81'),'Adam':('#E6AE65','#B65D00')}
    endpoints=[]
    for opt in ('SGD','Adam'):
        for i,eta in enumerate(sorted({r['step_size'] for r in only(rows,optimizer=opt)})):
            rr=sorted(only(rows,optimizer=opt,step_size=eta),key=lambda r:r['updates']);col=ramps[opt][i]
            a.plot(array(rr,'updates'),abs(array(rr,'neural_gap')),color=col,ls='-' if opt=='SGD' else '--')
            for r in rr:a.plot(r['updates'],abs(r['neural_gap']),'o',color=col,mfc=col if r['neural_gap']>0 else 'white')
            endpoints.append((abs(rr[-1]['neural_gap']),col,f'{opt} η={eta:g}'))
    a.set_xscale('log');a.set_yscale('log');a.set_ylim(1e-6,1);a.set_xlim(1.7,220)
    # Place direct labels in logarithmic display coordinates, with leader lines.
    prior=-np.inf
    for value,col,name in sorted(endpoints):
        logpos=max(math.log10(value),prior+.34);prior=logpos
        a.annotate(name,xy=(max(r['updates'] for r in rows),value),xytext=(100,10**logpos),color=col,va='center',arrowprops=dict(arrowstyle='-',color=col,lw=.6))
    note(a,f'single common-state step:\nparameter difference = {d.neural["additive_null_parameter_residual"]:g} (exact null)',x=.05,y=.33)
    labels(a,'number of updates','|neural gap|');title(a,'Same labels, different packets: learning-consequence gap',width=51)
    legend(a,[handle(BLUE,'SGD: positive gap','o'),Line2D([],[],color=ORANGE,marker='o',mfc='white',ls='--',label='Adam: negative gap')],ncol=2)
    j=d.package['coupling_and_additive_null'];low,high=j['quadratic_pair_min'],j['quadratic_pair_max']
    # No additive min/max is stored: its x-location is schematic and explicitly labelled.
    b.plot([j['additive_width']],[1],'o',color=BLUE,ms=6)
    b.plot([low,high],[0,0],color=ORANGE,lw=2,marker='|',ms=12)
    b.text(j['additive_width']+.02,1,f'width {j["additive_width"]:g}\n(position schematic)',va='center')
    b.text((low+high)/2,.12,f'width {j["quadratic_pair_width"]:g}',ha='center')
    c17=d.claims['C17'];note(b,f'C17: {c17["n_instances"]} random tables;\n(46) slack median {c17["details"]["median_inequality_slack"]:.2f},\nmax {c17["details"]["maximum_slack"]:.2f}',x=.47,y=.94)
    b.set_yticks([0,1],['quadratic pair H','additive H']);b.set_ylim(-.45,1.6);b.set_xlim(-.03,.6)
    b.set_xlabel('expected packet response ⟨Γ, H⟩');title(b,'Coupling width: 0 iff additive')
    d.captions['N16']=('N16. (A) In a controlled network (C23(e)), grouping the same four labels as {(0,0),(1,1)} versus {(0,1),(0,1)} changes the final return. '
        'The gap is exactly zero for one common-state step and grows with step size and horizon under SGD; under Adam it has the opposite sign and other dynamics. '
        '(B) The range of expected packet response over the coupling polytope: a single point for an additive table, positive width for a non-additive one. '
        'Only the additive width is stored, so its single point has schematic position. Exact finite computations and a controlled small network; not a real-model result.')
    return fig


def n17(d):
    fig,(a,b,c)=figure(3,(16,5),'Exact finite-history computations in a controlled small network; not a real-model result.')
    reports=d.g('E6')['reports'];splits=('held_out_states','fresh_to_warm','held_out_epsilon','held_out_lr')
    heat=np.array([[reports[s]['metrics'][p]['wrong_sign_rate'] for s in splits] for p in PROXIES])
    im=a.pcolormesh(np.arange(len(splits)+1)-.5,np.arange(len(PROXIES)+1)-.5,heat,cmap='Blues',vmin=0,vmax=.6,shading='flat')
    a.invert_yaxis();a.grid(False)
    for i in range(len(PROXIES)):
        for j in range(len(splits)): a.text(j,i,f'{heat[i,j]:.3f}',ha='center',va='center',color='white' if heat[i,j]>.33 else BLACK)
    a.set_yticks(range(len(PROXIES)),PROXY_NAMES)
    for tick,p in zip(a.get_yticklabels(),PROXIES):tick.set_color(PROXY_COLORS[p])
    a.set_xticks(range(len(splits)),['held-out\nstates','fresh →\nwarm','held-out\nε','held-out\nlr'])
    for y in (4.5,6.5):a.axhline(y,color=GRAY,lw=.7)
    a.add_patch(Rectangle((-.48,-.48),3.96,4.96,fill=False,edgecolor=BLACK,ls='--',lw=1.5))
    a.text(1.035,.72,'chance ≈ random row',rotation=90,transform=a.transAxes,ha='left',va='center')
    cb=fig.colorbar(im,ax=a,orientation='horizontal',pad=.14,shrink=.9);cb.set_label('wrong-sign rate');cb.solids.set_rasterized(False)
    title(a,'E6: static statistics are at chance on held-out states')
    ypos=np.arange(len(PROXIES));split_order=('held_out_states','held_out_task')
    # Proxy-group color is fixed across panels; split is encoded by fill/hatching.
    pair_containers=[]
    for j,split in enumerate(split_order):
        metrics=reports[split]['metrics']; vals=np.array([metrics[p]['selection_regret'] for p in PROXIES]);offset=(j-.5)*.38
        bars=b.barh(ypos+offset,vals,height=.34,color=[PROXY_COLORS[p] for p in PROXIES],alpha=1 if j==0 else .45,hatch=None if j==0 else '///',edgecolor='white',linewidth=.48)
        pair_containers.append(bars)
        for y,p,value in zip(ypos+offset,PROXIES,vals):
            interval=metrics[p].get('regret_bootstrap95')
            if interval is not None:
                b.plot(interval,[y,y],color=BLACK,lw=1.5);b.plot(interval,[y,y],'|',color=BLACK,ms=4)
            if split=='held_out_task' and p in ('short_tail_1','short_tail_4'):
                b.annotate(f'{value:.3f}',xy=(value,y),xytext=(4,-9),textcoords='offset points',va='center')
        random=metrics['random']['selection_regret'];b.axvline(random,color=GRAY,ls='--' if j else ':',alpha=.7)
        if j:b.text(random+.003,-.35,f'random {random:.3f}',rotation=0,ha='left',va='center',fontsize=8)
    b.set_yticks(ypos,PROXY_NAMES);b.invert_yaxis();b.set_xlabel('selection regret (bootstrap 95% intervals)')
    for tick,p in zip(b.get_yticklabels(),PROXIES):tick.set_color(PROXY_COLORS[p])
    title(b,'Short tails fail only where the task has a threshold')
    legend(b,[Patch(facecolor=GRAY,label='held-out states'),Patch(facecolor=GRAY,alpha=.45,hatch='///',label='held-out threshold task')])
    spec=c.get_subplotspec();c.remove();sub=spec.subgridspec(1,2,wspace=.08)
    left=fig.add_subplot(sub[0,0]);right=fig.add_subplot(sub[0,1],sharey=left)
    summary=d.g('E9')['summary'];order=('input_moment_error','mean_gradient','first_delta','delta_plus_m_v','sign_only','short_tail_1','short_tail_4','random')
    for ax,key,xname in [(left,'selection_regret','Δ regret'),(right,'wrong_sign_rate','Δ wrong-sign')]:
        for j,W in enumerate((5,20)):
            rr=[one(summary,proxy=p,W=W) for p in order]
            vals=[r['metrics'][key]['mean_warm_minus_cold'] for r in rr]
            ax.scatter(vals,np.arange(len(order))+(j-.5)*.17,c=[PROXY_COLORS[p] for p in order],marker=MARKERS[j],s=24,label=f'W={W}')
        ax.axvline(0,color=GRAY,lw=1.5);lo,hi=ax.get_xlim();ax.axvspan(lo,0,color=GREEN,alpha=.06);ax.set_xlim(lo,hi)
        ax.set_xlabel(xname);ax.set_yticks(range(len(order)),[PROXY_NAMES[PROXIES.index(p)] for p in order]);ax.grid(axis='y',visible=False)
    right.tick_params(labelleft=False);left.invert_yaxis()
    for tick,p in zip(left.get_yticklabels(),order):tick.set_color(PROXY_COLORS[p])
    legend(right,[handle(GRAY,f'W={W}',MARKERS[j],'') for j,W in enumerate((5,20))],ncol=2)
    cheap=d.g('E9')['cheap_proxy_results'];count=sum(r['all_three_improve'] for r in cheap)
    random_rows=[r for r in summary if r['proxy']=='random'];improved={r['metrics']['selection_regret']['n_splits_improved'] for r in random_rows};nsplit={r['metrics']['selection_regret']['n_splits'] for r in random_rows}
    require(len(improved)==len(nsplit)==1,'N17 random split improvements differ by W; annotation needs separate counts')
    note(left,f'random regret falls {next(iter(improved))}/{next(iter(nsplit))} splits\n⇒ providers closer after warm-up\n{count}/{len(cheap)} static proxy/W pairs\nimprove all three metrics',x=-.1,y=-.23)
    group_title(fig,left,right,'E9: warm-up compresses differences,\nstatic proxies stay non-predictive')
    gap=2/300/fig.get_figheight()/b.get_position().height*abs(np.diff(b.get_ylim())[0])
    for bars in pair_containers:
        for bar in bars:
            center=bar.get_y()+bar.get_height()/2
            bar.set_height(.38-gap);bar.set_y(center-bar.get_height()/2)
    d.captions['N17']=('N17. Controlled-learner evidence for CED’s selection rules. (A) Wrong-sign rate of frozen proxies on held-out learner states, warmth shift, ε and learning-rate shifts (E6); '
        'static update statistics are at the random level, measured short-tail responses are near zero. (B) Selection regret: short tails select near-optimally on held-out states but worse than random '
        'on the held-out threshold task; only the full tail is exact everywhere. (C) Effect of an lr = 0 moment warm-up (E9): the spread between providers shrinks (random regret falls) while static '
        'proxies do not improve all metrics. Exact finite-history computations in a controlled small network; not a real-model result.')
    return fig


def finite(value):
    """Missing numeric strings (including 'nan') never become plotted zeros."""
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def warmth_fits(d):
    return [r['fit'] for r in only(d.g('E8')['W3']['fits'],eta=.01,W=0)
            if finite(r['fit']['loglog_slope'])]


OPTIMIZERS = ('SGD','heavy_ball','Nesterov','Adam_cold','Adam_warm5','Adam_warm20',
              'AdamW','clipped_SGD','clipped_Adam','signSGD','Lion')
OPTIMIZER_NAMES = ('SGD','heavy-ball','Nesterov','Adam cold','Adam warm W=5','Adam warm W=20',
                   'AdamW','clipped SGD','clipped Adam','signSGD','Lion')
OPTIMIZER_COLORS = (BLUE,ORANGE,GREEN,PURPLE,SKY,RED,'#8064A2','#A6761D','#6B8E23',BLACK,'#C46C92')


def slope_guide(ax, x, y, slope, label=True):
    xx=np.array([x/10,x]); yy=y*(xx/x)**slope
    ax.plot(xx,yy,':',color=GRAY,lw=.8)
    if label:
        ax.annotate(rf'$\eta^{slope}$',xy=(xx[0],yy[0]),xytext=(0,-12),
                    textcoords='offset points',color=GRAY,ha='center',fontsize=8)


def n18(d):
    data=d.g('E10'); cells=data['cells']; maxima=[]
    for opt in OPTIMIZERS:
        values=[float(r['ratio']) for r in only(cells,optimizer=opt)
                if r['R'] in (1,4) and finite(r['ratio'])]
        require(values,f'N18 no numeric exact ratios for {opt}')
        maximum=max(values); expected=one(data['optimizer_summary'],optimizer=opt)['max_exact_ratio']
        require(math.isclose(maximum,expected,rel_tol=0,abs_tol=1e-12),
                f'N18 reproduction failed for {opt}: {maximum} != {expected}; refusing to draw')
        maxima.append(maximum)
    print('N18 reproduction PASS: all 11 exact-cell maxima match within 1e-12',flush=True)
    fig,(a,b,c)=figure(3,(17,5.5),'Controlled small network; not a real-model result. Bounds use empirical curvature estimates; S1 is not certified.')
    endpoints=[]; selected=('SGD','Adam_cold','Adam_warm20','signSGD')
    for opt in selected:
        ix=OPTIMIZERS.index(opt); col=OPTIMIZER_COLORS[ix]; name=OPTIMIZER_NAMES[ix]
        for R,ls in [(1,'-'),(4,'--')]:
            rr=sorted([r for r in only(cells,task='moment',N=3,optimizer=opt,R=R)
                       if finite(r['remainder']) and float(r['remainder'])>1e-15],key=lambda r:r['eta'])
            require(rr,f'N18 no remainder points for {opt}, R={R}')
            a.loglog(array(rr,'eta'),array(rr,'remainder'),color=col,ls=ls,marker='o' if R==1 else 's',
                     mfc=col if R==1 else 'white')
            slope=one(one(data['optimizer_summary'],optimizer=opt)['slopes'],task='moment',N=3,R=R)['remainder_slope']
            require(finite(slope),f'N18 missing selected slope: {opt}, {R}')
            endpoints.append((rr[-1]['eta'],rr[-1]['remainder'],col,f'{name} R={R}: {slope:.2f}'))
            guide={('Adam_cold',4):1,('signSGD',1):2,('SGD',1):3}.get((opt,R))
            if guide:slope_guide(a,rr[-1]['eta'],rr[-1]['remainder'],guide)
    # Allocate separate label rows next to the line ends; leaders retain identity.
    a.set_xlim(.00075,8); a.set_ylim(1e-15,max(row[1] for row in endpoints)*15)
    for ypos,(x,y,col,text) in zip(np.linspace(.19,.88,len(endpoints)),sorted(endpoints,key=lambda r:r[1])):
        a.annotate(text,xy=(x,y),xytext=(.58,ypos),textcoords='axes fraction',
                   ha='left',va='center',color=col,fontsize=8,
                   arrowprops=dict(arrowstyle='-',color=col,lw=.5))
    labels(a,r'step size $\eta$',r'remainder $|\mathrm{gap}-\mathrm{first\ order}|$')
    title(a,'(a) Remainder order in '+r'$\eta$'+'\n(moment N=3)')
    legend(a,[handle(GRAY,'R=1','o'),handle(GRAY,'R=4','s','--')],ncol=2)
    pred=data['C23_prediction']; keys=('gap','first_order','naive_first_order','remainder','bound')
    names=['measured gap','first order\n(increment)','first order\n(raw mean gradient)',
           'remainder\n|gap − first order|','remainder bound']
    for i,(key,col) in enumerate(zip(keys,(BLUE,GREEN,ORANGE,PURPLE,GRAY))):
        v=abs(pred[key]); require(v>0,f'N18 zero C23 {key} on log axis')
        b.barh(i,v-1e-12,left=1e-12,color=col,height=.6)
        b.annotate(f'{pred[key]:.2g}'.replace('-', '−'),xy=(v,i),xytext=(4,0),
                   textcoords='offset points',va='center',fontsize=9)
    b.set_yticks(range(5),names);b.invert_yaxis();b.set_xscale('log');b.set_xlim(1e-12,1)
    b.set_ylim(5.15,-.65);note(b,f'first order explains {pred["first_order"]/pred["gap"]:.1%} of the gap',y=.08)
    labels(b,'absolute value','');title(b,'(b) C23 cold-Adam gap:\nincrement vs raw gradient')
    bars=c.barh(range(len(OPTIMIZERS)),maxima,color=OPTIMIZER_COLORS,height=.68)
    bars[OPTIMIZERS.index('clipped_SGD')].set_hatch('///')
    c.set_yticks(range(len(OPTIMIZERS)),OPTIMIZER_NAMES);c.invert_yaxis();c.set_xlim(0,1.55)
    c.axvline(1,color=GRAY,ls='--',label='remainder bound')
    c.set_ylim(len(OPTIMIZERS)+2,-.8)
    note(c,'multi-step clipped SGD:\nS1 not guaranteed',x=.05,y=.15)
    mc=only(cells,R=40); exceed=sum(finite(r.get('excess_over_bound_after_four_SE')) and
                                 float(r['excess_over_bound_after_four_SE'])>0 for r in mc)
    labels(c,'max exact remainder / bound','');title(c,'(c) Bound check in exact cells\n(R = 1, 4)')
    legend(c,[handle(GRAY,'remainder bound',ls='--')],y=-.22)
    c.text(.5,-.34,f'R=40 Monte Carlo: {exceed} of {len(mc)} cells\nabove bound after 4 SE (sign-type)',
           transform=c.transAxes,ha='center',va='top',fontsize=9)
    d.captions['N18']=('N18. (A) Remainder orders for moment N=3; fitted slopes are read from E10. '
        'The slope-2 line is a reference guide, while the selected signSGD series is approximately cubic. '
        '(B) The optimizer-increment first-order term explains '+f'{pred["first_order"]/pred["gap"]:.1%}'+
        ' of the C23 cold-Adam gap; the raw mean gradient is nearly blind. (C) All available exact-cell ratios are at most one '
        'except multi-step clipped SGD. Curvature bounds are empirical, and S1 is not certified. '
        'Controlled small network; not a real-model result.')
    return fig


def n19(d):
    data=d.g('E11');fig=plt.figure(figsize=(15,8.5),layout='constrained')
    grid=fig.add_gridspec(3,3,height_ratios=[1,1,.07])
    a,b,c,e,f,t=[fig.add_subplot(grid[i,j]) for i in range(2) for j in range(3)]
    foot=fig.add_subplot(grid[2,:]);foot.axis('off')
    foot.text(.5,.5,'Controlled small networks; not a real-model result. Warm-Adam W1 bound covers only the linear v contribution.',
              ha='center',va='center',fontsize=9,color='#555555')
    block=data['clipped_mean'];rr=block['rows'];x=array(rr,'eta')
    a.loglog(x,array(rr,'gap'),'o',color=BLUE,label='gap')
    a.loglog(x,array(rr,'first_order'),color=ORANGE,label='first order')
    a.loglog(x,array(rr,'remainder'),'-s',color=GREEN,mfc='white',label='remainder')
    slope_guide(a,x[-1],rr[-1]['remainder'],2)
    note(a,'raw means equal; clipped means differ by −1/2\n'+f'remainder slope {block["remainder_slope"]:.3f}',y=.98)
    a.set_ylim(top=max(array(rr,'gap'))*30)
    labels(a,r'step size $\eta$','absolute gap / remainder');title(a,'(a) Clipped SGD: the clipped mean decides');legend(a,ncol=3)
    block=data['always_active'];rr=block['rows'];x=array(rr,'eta')
    require(all(r['first_order']==0 for r in rr),'N19 always-active first order is not identically zero')
    b.loglog(x,array(rr,'gap'),'-o',color=BLUE);slope_guide(b,x[-1],rr[-1]['gap'],2)
    note(b,'first-order term = 0 at every '+r'$\eta$'+f'\ngap slope {block["slope"]:.5f}')
    b.set_ylim(top=max(array(rr,'gap'))*10)
    labels(b,r'step size $\eta$','gap');title(b,'(b) Always-active clipping:\nonly the direction matters')
    warm=data['warm_Adam'];vals=np.r_[warm['first_order_blocks'],warm['measured_gap']]*1e6
    c.bar(range(4),vals,color=[BLUE,ORANGE,GREEN,PURPLE],width=.65)
    c.plot(2,warm['predicted_v_term']*1e6,'D',mfc='white',mec=BLACK,ms=7,label='predicted v term')
    c.axhline(warm['W1_Lv_kappa_bound']*1e6,color=GRAY,ls='--',label=r'W1 bound $L_v\kappa$')
    c.set_xticks(range(4),[r'$\theta$ block','m block','v block','measured\ngap'])
    note(c,rf'equal gradient means ($\delta_1 = {np.max(np.abs(warm["delta1"])):g}$)')
    c.set_ylim(0,max(vals)*1.35);labels(c,'',r'contribution / gap ($10^{-6}$)')
    title(c,r'(c) Warm Adam: $E[G^2]$ enters through $v$');legend(c,ncol=2)
    lion=data['Lion'];increments=[np.asarray(r['increment_difference']).reshape(3,-1) for r in lion]
    for j,(col,name) in enumerate([(BLUE,r'$\theta$ block'),(ORANGE,'m block')]):
        values=[np.max(abs(v[j])) for v in increments]
        # Zero has no bar on the log scale; label it at the displayed floor.
        for i,value in enumerate(values):
            if value>0:e.bar(i+(j-.5)*.3,value-1e-4,bottom=1e-4,width=.28,color=col)
            else:e.text(i+(j-.5)*.3,1.25e-4,'0 (exact)',ha='center',va='bottom',fontsize=9)
    flips=[float(np.mean(r['flip_probability'])) for r in lion]
    note(e,'mean flip fraction: '+', '.join(f'{v:g}' for v in flips))
    e.set_yscale('log');e.set_ylim(1e-4,.25);e.set_xlim(-.5,1.5)
    e.set_xticks([0,1],['momentum-dominated\nm=0.2','control\nm=0.005'])
    labels(e,'',r'max $|$increment difference$|$');title(e,'(d) Lion: momentum dominance\nblinds the parameter step')
    legend(e,[Patch(color=BLUE,label=r'$\theta$ block'),Patch(color=ORANGE,label='m block')],ncol=2)
    annotations=[]
    for i,block in enumerate(data['LoRA']):
        rr=block['rows'];x=array(rr,'eta');col=COLORS[i]
        f.loglog(x,abs(array(rr,'gap')),color=col,ls='-',label=f'rank {block["rank"]}: null')
        f.loglog(x,array(rr,'range_gap'),color=col,ls='--',label=f'rank {block["rank"]}: range')
        f.plot(x,array(rr,'range_first_order'),ls='',marker=MARKERS[i],color=col,mfc='white')
        if i==0:
            slope_guide(f,x[-1],abs(rr[-1]['gap']),2)
            slope_guide(f,x[-1],rr[-1]['range_gap'],1)
        audits=block['A_bitwise_audit']
        annotations.append(f'rank {block["rank"]}: null-space slope {block["null_gap_slope"]:.5f};\nA unchanged: {sum(r["unchanged"] for r in audits)}/{len(audits)}')
    note(f,'\n'.join(annotations),x=.03,y=.60)
    labels(f,r'step size $\eta$','absolute gap');title(f,r'(e) LoRA at $B=0$: only the sketch'+ '\n'+r'$\Gamma A_0^{\mathsf{T}}$ matters')
    handles,_=f.get_legend_handles_labels();handles.append(handle(GRAY,'range first order','o',''));legend(f,handles,ncol=2)
    t.axis('off');title(t,'(f) Learner → first-order statistic')
    table=[('SGD, momentum',r'$E\,G$'),('warm Adam',r'$E\,G$ and $E\,G^2$'),
           ('cold Adam','sign law; '+r'$E\,G$, $E\,G^2$ in $m,v$'),
           ('clipped SGD','clipped mean (direction\nif always active)'),('Lion','flip probability'),
           ('LoRA, step 1',r'sketch $\Gamma A_0^{\mathsf{T}}$'+'\n(or its sign law)')]
    for i,(learner,statistic) in enumerate(table):
        t.text(.0,.92-.15*i,learner,transform=t.transAxes,color=BLACK,va='top')
        t.text(.40,.92-.15*i,statistic,transform=t.transAxes,color=BLACK,va='top')
    d.captions['N19']=('N19. Five witnesses of the learner-specific first-order statistic (Theorem C4). '
        '(A) Equal raw means, different clipped means: first-order gap under clipping. '
        '(B) Always-active clipping: equal mean directions give an O(eta²) gap. '
        '(C) Warm Adam: with equal gradient means, the first-order term is carried entirely by the v block; '
        'the W1 bound covers this linear contribution, not the entire measured gap. '
        '(D) Momentum-dominated Lion: the parameter increment does not depend on the message. '
        '(E) LoRA at initialization: null-space differences are second order, range differences first order. '
        'Controlled small networks; not a real-model result.')
    return fig


AUDIT_METHODS=('oracle','mean_gradient','delta_plus_m_v','short_tail_4','random')
AUDIT_NAMES=('oracle','mean gradient','Δθ+m,v','short tail 4','random')
AUDIT_COLORS=(BLACK,BLUE,PURPLE,ORANGE,GRAY)


def outside_oracle(d):
    bins=d.g('E12')['bins'];summary=one(d.g('E12')['summary'],split='overall')
    outside=[r for r in bins if float(r['high'])<=1];inside=[r for r in bins if r['low']>=1]
    nout=sum(r['n'] for r in outside);nin=sum(r['n'] for r in inside)
    rate=sum(r['n']*r['oracle'] for r in outside)/nout
    combined=(nout*rate+nin*summary['oracle_wrong_sign_margin'])/(nout+nin)
    error=abs(combined-summary['oracle_wrong_sign_eligible'])
    return dict(rate=rate,combined=combined,error=error,valid=error<=.02,nout=nout,nin=nin)


def n20(d):
    concentration=d.g('V9');rows=concentration['rows'];data=d.g('E12');bins=data['bins']
    summary=one(data['summary'],split='overall');check=outside_oracle(d)
    print(f'N20 reproduction {"PASS" if check["valid"] else "DISCREPANCY"}: weighted oracle {check["combined"]:.10f}, '
          f'stored {summary["oracle_wrong_sign_eligible"]:.10f}, difference {check["error"]:.10f} (tolerance 0.02)',flush=True)
    fig,(a,b,c)=figure(3,(16,4.8),'Controlled small network and finite-vector Monte Carlo; not a real-model result.')
    for law,col,name in [('sign',BLUE,'sign'),('clipped_Gaussian',ORANGE,'clipped Gaussian'),('sparse_spikes',GREEN,'sparse spikes')]:
        ps=sorted({r['p'] for r in only(rows,law=law)})
        vals=[max(r['quantile_over_bound'] for r in only(rows,law=law,p=p)) for p in ps]
        a.semilogx(ps,vals,'-o',color=col,label=name)
    maximum=max(r['quantile_over_bound'] for r in rows)
    require(math.isclose(maximum,concentration['max_quantile_over_bound'],rel_tol=0,abs_tol=1e-12),'N20 V9 maximum does not reproduce')
    a.axhline(1,color=GRAY,ls='--',label='C5 bound');a.set_xlim(1,1e5);a.set_ylim(0,1.1)
    note(a,'worst failure rate:\n'+ '\n'.join(rf'$\delta={delta:g}$: {max(r["failure_rate"] for r in only(rows,delta=delta)):.5f}' for delta in (.05,.2))+
         f'\nmax quantile / bound: {maximum:.3f}',y=.37)
    labels(a,'dimension p','max quantile / C5 bound');title(a,'(a) Concentration does not\ndepend on dimension');legend(a,ncol=2)
    containers=[];xs=np.arange(len(bins));width=.16
    for j,(method,name,col) in enumerate(zip(AUDIT_METHODS,AUDIT_NAMES,AUDIT_COLORS)):
        values=[r['oracle'] if method=='oracle' else r['proxies'][method] for r in bins]
        bars=b.bar(xs+(j-2)*width,values,width=width,color=col,label=name);containers.append(bars)
        if method=='oracle':
            for bar,value in zip(bars,values):
                if value==0:b.text(bar.get_x()+bar.get_width()/2,.015,'0',ha='center',va='bottom',fontsize=9)
    b.set_xticks(xs,[f'[{r["low"]:g}, {float(r["high"]):g})\nn={r["n"]}' for r in bins])
    border=next(i for i,r in enumerate(bins) if r['low']>=1)-.5
    b.axvline(border,color=GRAY,ls='--');b.set_ylim(0,.88)
    note(b,'margin > remainder bound (C6)',x=.04,y=.98)
    labels(b,'first-order margin / remainder bound','wrong-sign rate');title(b,'(b) Wrong-sign rate by\nfirst-order margin');legend(b,ncol=3)
    for i,(method,col) in enumerate(zip(AUDIT_METHODS,AUDIT_COLORS)):
        inside=summary['oracle_wrong_sign_margin'] if method=='oracle' else summary['proxies'][method]['wrong_sign_margin']
        outside=check['rate'] if method=='oracle' else summary['proxies'][method]['wrong_sign_outside_margin']
        c.plot(inside,i-.10,'o',color=col,ms=6)
        if method!='oracle' or check['valid']:c.plot(outside,i+.10,'o',mfc='white',mec=col,ms=6)
    c.set_yticks(range(len(AUDIT_METHODS)),AUDIT_NAMES);c.invert_yaxis();c.set_xlim(-.03,.62);c.set_ylim(5.2,-.6)
    note(c,f'margin rows: {summary["margin_rows"]} of {summary["eligible_rows"]} eligible',y=.10)
    labels(c,'wrong-sign rate','');title(c,'(c) Inside vs outside the margin')
    legend(c,[handle(BLACK,'inside margin','o',''),Line2D([],[],color=BLACK,marker='o',mfc='white',ls='',label='outside margin')],ncol=2)
    pixel_gaps(fig,b,list(zip(*containers)))
    d.captions['N20']=('N20. (A) The empirical deviation of the mean increment stays below 0.62 of the C5 bound at every tested '
        'dimension from 1 to 100,000. (B) On the frozen E6 panels, the oracle has no wrong signs in bins at or above one quarter '
        'of the remainder bound; this empirical observation is distinct from the C6 margin > bound condition. '
        'Static proxies remain wrong 42% of the time in the largest-margin bin. (C) Inside the margin first order decides; '
        'outside, measured short tails are more reliable here but retain nonzero error. The outside-margin oracle rate is '
        'reconstructed with the specified bin-row weighting. Controlled small network; not a real-model result.')
    return fig


DESCRIPTIONS = (
    ('exact Fraction instance in N01; P:robust_ambiguity_minimax', 'Regret decomposition, garbling, and minimax duality.'),
    ('I:claims[id=C2].details.configurations; specified passive formula', 'Query counts, bound ratios, and passive impossibility.'),
    ('V:groups[id=E5].data.{curves,actual_learners}; T:checks.T11_adaptive_query_KL_enumeration.rows', 'Hidden-bit information floors, learners, and additive KL budgets.'),
    ('specified analytic formulas; I:controlled_neural.details.configurations', 'Static equivalence versus different trained students.'),
    ('exact Fraction instances in N05; I:claims[id=C4].details.cancellation_absolute_drift', 'Telescoping, response equivalence, and cancellation.'),
    ('exact Fraction instance in N06; I:claims[id=C19].details.families; I:claims[id=C5]', 'Worst matched-pair discrepancy equals twice approximation error.'),
    ('I:claims[id=C8].details.scalar; I:claims[id=C9].details.{grid_ranks,binary_barriers}; P:constant_gap_parity', 'Necessary moment orders, exact ranks, and the binary barrier.'),
    ('I:claims[id=C18].details.curves; I:claims[id=C11].details.cumulant_cases; I:claims[id=C10]', 'Independent-evidence rates, cumulant coefficients, and a bound uniform in rounds.'),
    ('specified exact Chebyshev coefficients; P:scalar_approximation; I:claims[id=C12]; I:claims[id=C19].details.families', 'Approximation complexity versus exact polynomial degree.'),
    ('I:controlled_neural.details.configurations; I:claims[id=C13]; V:groups[id=V7].data.linear_failure', 'Local SGD regime, remainder bounds, and geometric error accumulation.'),
    ('V:groups[id=E2].data.comparisons; V:groups[id=E1].data.ablations; V:groups[id=V4].data', 'Cold Adam, ablations, and magnitude-bearing optimizer memory.'),
    ('V:groups[id=E8].data.{gaps,W3.per_step,W3.fits,W1.panels}', 'Warm-up gaps, inverse-count warmth, and W1 residuals.'),
    ('I:claims[id=C22].details.configurations; P:noisy_feature_duality.explicit_example; I:claims[id=C14]', 'Tolerant duality and the cost of small feature scale.'),
    ('theory/v015_checks/{mathematics,common}.py (read-only, with common’s committed C23 dependency); V:groups[id=V1].data; V:groups[id=E4].data', 'Reproduced transport certificates, reached-state diagnostics, and uncovered states.'),
    ('I:claims[id=C16].details.configurations; V:groups[id=E7].data.{Hoeffding,empirical_Bernstein,variance_decomposition}', 'Adaptive audit gains, replica cost, and variance sources.'),
    ('I:controlled_neural.details.{regrouping,additive_null_parameter_residual}; P:coupling_and_additive_null; I:claims[id=C17]', 'Sequential regrouping effects and additive coupling width.'),
    ('V:groups[id=E6].data.reports; V:groups[id=E9].data.{summary,cheap_proxy_results}', 'Held-out proxy selection, threshold failures, and warm-up transfer.'),
    ('V:groups[id=E10].data.{cells,optimizer_summary,C23_prediction}', 'Optimizer-increment linear response, remainder orders, and empirical bound checks.'),
    ('V:groups[id=E11].data', 'Clipping, warm Adam, Lion, and LoRA first-order statistics.'),
    ('V:groups[id=V9].data; V:groups[id=E12].data.{bins,summary}', 'Dimension-free concentration and actual-state first-order margin audits.'),
)


def discrepancies(d):
    """Compare the spec's rounded annotations to the selected evidence, not vice versa."""
    result=[]
    floor_rows=only(d.g('E5')['curves'],m=8,q_profile='uniform',w_profile='uniform')
    maximum=max(r['K'] for r in floor_rows)
    if maximum!=32:
        result.append(f'N03 A: the spec requests K=0…32, but the committed selected curves contain K=0…{maximum}. '
                      'Only stored values are drawn; the panel is labelled with its supported range, with no extrapolation.')
    rows=d.g('E8')['gaps']
    for R,expected in [(1,.0004),(4,.0016)]:
        selected=sorted(only(rows,task='moment',N=3,eta=.1,R=R),key=lambda r:r['W'])
        start,end=selected[0],selected[-1];actual=end['gaps']['actual_Adam']
        if f'{actual:.4f}'!=f'{expected:.4f}':
            result.append(f'N12 A: moment N={end["N"]}, η={end["eta"]:g}, R={R}, W={end["W"]}: '
                f'spec endpoint {expected:.4f}; data {actual:.10g}. Displayed {start["gaps"]["actual_Adam"]:.4f} → {actual:.4f}.')
    fit=one(d.g('E8')['W3']['fits'],task='moment',N=3,source='P',eta=.01,W=0)['fit']
    if f'{fit["loglog_slope"]:.2f}'!='-1.06' or f'{fit["R2"]:.3f}'!='0.999':
        fits=warmth_fits(d)
        result.append(f'N12 B: moment N=3, source P has slope {fit["loglog_slope"]:.10g} and R² {fit["R2"]:.10g}. '
                      f'The separate medians over {len(fits)} finite-slope task/source series at η=0.01, W=0 are '
                      f'{np.median([r["loglog_slope"] for r in fits]):.10g} and {np.median([r["R2"] for r in fits]):.10g}. Both are now labelled.')
    w1=d.g('E8')['W1'];point=max(y/x for p in w1['panels'] for x,y in zip(p['r'],p['pointwise_residual']))
    result.append(f'N12 C clarification: pointwise max residual/r = {point:.10g} (within {100*(1-point):.3f}%); '
                  f'law-pair max gap/bound = {w1["best_ratio"]:.10g} (within {100*(1-w1["best_ratio"]):.3f}%). '
                  'The spec’s “within 0.2%” applies to the pointwise residual, not the law-pair ratio. Both are labelled separately.')
    selected=[one(d.g('E2')['comparisons'],task=t,N=n,eta=e,R=r) for t,n,e,r in
              [('moment',3,.1,1),('moment',3,.1,4),('moment',3,.01,4),('parity',2,.1,4),('moment',4,.1,4)]]
    sgd=max(abs(r['gaps']['actual_SGD']) for r in selected)
    shift=only(d.g('E1')['ablations'],task='moment',N=3,variant='shift_support')
    result.append(f'N11 wording: the plotted SGD gaps are small, not identically zero (max |actual SGD| = {sgd:.10g}); '
                  f'shifted-support residuals are '+', '.join(f'R={r["R"]}: {r["gap"]:.10g}' for r in shift)+'. Signed values are retained; caption says “small”/“nearly”.')
    slope=one(one(d.g('E10')['optimizer_summary'],optimizer='signSGD')['slopes'],task='moment',N=3,R=1)['remainder_slope']
    result.append(f'N18 A: the requested slope-2 guide is anchored to signSGD R=1, whose stored fitted slope is {slope:.10g} (approximately cubic). '
                  'The requested reference guide and actual fitted slope are both retained.')
    missing=sum(not finite(r['ratio']) for r in d.g('E10')['cells'] if r['R'] in (1,4))
    result.append(f'N18 C: {missing} exact cells have nonnumeric ratios and are omitted, never zero-filled. '
                  'E10 explicitly says smoothness S1 is not certified and curvature estimates are empirical. '
                  'The caption reports observed ratios, not a verified uniform smoothness certificate.')
    warm=d.g('E11')['warm_Adam']
    result.append(f'N19 C: measured gap {warm["measured_gap"]:.10g} exceeds W1 bound {warm["W1_Lv_kappa_bound"]:.10g}; '
                  f'the source caveat says this bound covers only the linear v contribution ({warm["predicted_v_term"]:.10g}), not the full nonlinear gap.')
    check=outside_oracle(d);summary=one(d.g('E12')['summary'],split='overall')
    result.append(f'N20 C: requested bin-row-weighted outside oracle rate = {check["rate"]:.10g}; combining with the inside rate gives '
                  f'{check["combined"]:.10g} versus stored eligible rate {summary["oracle_wrong_sign_eligible"]:.10g} '
                  f'(absolute difference {check["error"]:.10g}; tolerance 0.02: {"PASS" if check["valid"] else "FAIL; reconstructed point omitted"}). '
                  'Bin n counts include zero-gold rows excluded from wrong-sign denominators, so row weighting is approximate. '
                  'Stored proxy rates retain their original denominators.')
    result.append('N20 caption: zero oracle error from ratio 0.25 is an empirical bin observation; the C6 separator remains at ratio 1. '
                  'The outside-margin short-tail-4 error is nonzero; the caption does not claim perfect reliability.')
    return result


def strip_png_timestamps(path):
    raw=path.read_bytes();require(raw[:8]==b'\x89PNG\r\n\x1a\n','Invalid PNG signature')
    chunks=[raw[:8]];position=8
    while position<len(raw):
        length=struct.unpack('>I',raw[position:position+4])[0];kind=raw[position+4:position+8]
        end=position+length+12;payload=raw[position+8:position+8+length]
        keyword=payload.split(b'\0',1)[0].lower()
        is_timestamp=(kind in (b'tIME',b'eXIf') or kind in (b'tEXt',b'zTXt',b'iTXt') and
                      keyword in (b'creation time',b'creationdate',b'moddate',b'date:create',b'date:modify',b'timestamp'))
        if not is_timestamp:chunks.append(raw[position:end])
        position=end
    path.write_bytes(b''.join(chunks))


def text_audit(fig):
    """Audit every visible rendered Text bbox, in 300-dpi pixel coordinates.

    Matplotlib keeps off-view tick artists: exclude ticks outside the axis view,
    plus descendants of invisible parents. No text-pair overlap is whitelisted.
    An Annotation's bbox includes its leader; use Text's bbox for glyphs only.
    """
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    ticks={}
    for ax in fig.axes:
        for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                loc=tick.get_loc()
                visible=(min(limits)<=loc<=max(limits) or any(math.isclose(loc,v,rel_tol=1e-12,abs_tol=0) for v in limits)) and ax.axison
                ticks[id(tick.label1)]=visible;ticks[id(tick.label2)]=visible
            ticks[id(axis.label)]=ax.axison;ticks[id(axis.offsetText)]=ax.axison
    seen=set();items=[];raw_math=[]
    def walk(artist,visible=True):
        visible=visible and artist.get_visible()
        if id(artist) in seen:return
        seen.add(id(artist))
        if isinstance(artist,Text) and visible and ticks.get(id(artist),True) and artist.get_text().strip():
            bbox=Text.get_window_extent(artist,renderer)
            if np.all(np.isfinite(bbox.extents)) and bbox.width>0 and bbox.height>0:
                label=artist.get_text();axis=fig.axes.index(artist.axes) if artist.axes in fig.axes else None
                items.append(dict(text=label,axes=axis,bbox=[round(float(v),3) for v in bbox.extents]))
                if re.search(r'[_^{}]', ''.join(label.split('$')[::2])):raw_math.append(label)
        for child in artist.get_children():walk(child,visible)
    walk(fig)
    overlaps=[]
    for i,first in enumerate(items):
        for j in range(i+1,len(items)):
            second=items[j];a=first['bbox'];b=second['bbox']
            w=min(a[2],b[2])-max(a[0],b[0]);h=min(a[3],b[3])-max(a[1],b[1])
            if w>0 and h>0:
                overlaps.append(dict(first=i,second=j,width_px=round(w,3),height_px=round(h,3)))
    outside=[i for i,r in enumerate(items) if r['bbox'][0]<0 or r['bbox'][1]<0 or
             r['bbox'][2]>fig.bbox.width or r['bbox'][3]>fig.bbox.height]
    return dict(dpi=300,text_count=len(items),texts=items,overlaps=overlaps,outside_canvas=outside,raw_math=raw_math)


def save(fig,basename,caption):
    # Keep all scatter/line/bar artists vector in PDF (no rasterized layers).
    for ax in fig.axes:
        for item in ax.get_children():
            if hasattr(item,'get_rasterized'):
                require(not item.get_rasterized(),f'{basename}: unexpected rasterized artist')
    fig.set_dpi(300)
    # Settle constrained layout once, then use the same positions for PDF/PNG.
    fig.canvas.draw();fig.canvas.draw()
    fig.set_layout_engine('none')
    for left,right,shared in getattr(fig,'_shared_titles',[]):
        p,q=left.get_position(),right.get_position()
        shared.set_position(((p.x0+q.x1)/2,max(p.y1,q.y1)+10/72/fig.get_figheight()))
    for ax,rows,pixels in getattr(fig,'_pixel_gap_specs',[]):
        gap=pixels/ax.bbox.width*(ax.get_xlim()[1]-ax.get_xlim()[0])
        for row in rows:
            for j,bar in enumerate(row):
                leftgap=gap/2 if j else 0;rightgap=gap/2 if j<len(row)-1 else 0
                bar.set_x(bar.get_x()+leftgap);bar.set_width(max(0,bar.get_width()-leftgap-rightgap))
    fig.savefig(HERE/f'{basename}.pdf',metadata={
        'CreationDate':None,'ModDate':None,'Title':basename,'Subject':caption,
        'Creator':'make_figures.py','Author':'',
    })
    fig.savefig(HERE/f'{basename}.png',dpi=300,metadata={
        'Title':basename,'Description':caption,'Software':'make_figures.py',
    })
    strip_png_timestamps(HERE/f'{basename}.png')
    audit=text_audit(fig)
    require(not audit['raw_math'],f'{basename}: unrendered formula text {audit["raw_math"]}')
    print(f'{basename}: {audit["text_count"]} visible text artists, {len(audit["overlaps"])} bbox overlaps, '
          f'{len(audit["outside_canvas"])} outside canvas',flush=True)
    plt.close(fig)
    return audit


def readme(d):
    lines=['# Theoretical framework figures','',
        'Regenerate from the repository root (CPU only, no network):','',
        '```bash',"CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theoretical_framework/figures/make_figures.py",'```','',
        'For a subset, append `--only N12,N14`. Each basename produces a vector PDF with embedded Type 42 fonts and a 300 dpi PNG. '
        'Captions are embedded in PDF Subject and PNG Description metadata. All figures use constrained layout and explicit colors. '
        'The style block uses 9 pt text; a few dense annotations and category labels use 8 pt.','',
        'JSON evidence is loaded from the named blobs in Git HEAD; uncommitted result files and `theory/v015_checks/raw/` are never read. '
        'The N14 working-tree imports must match HEAD byte for byte. N18–N20 require only stored-data reproduction checks; '
        'their specs do not request importing or running otlr.py. Temporary Matplotlib caches are confined to this directory and removed on exit.','',
        'Data-source abbreviations:','']
    lines.extend(f'- **{key}**: `{path}`' for key,path in SOURCES.items())
    lines+=['','| Basename | Node | Data sources / key paths | What it shows |','| --- | --- | --- | --- |']
    for basename,(source,description) in zip(BASENAMES,DESCRIPTIONS):
        lines.append(f'| `{basename}` | {basename[:3]} | {source} | {description} |')
    lines+=['','## Data/spec differences','']+[f'- {s}' for s in discrepancies(d)]
    lines+=['','## Rendering choices and scope','',
        '- N09 B omits exact zero errors from the log curves. A triangle at the threshold marks every first crossing, including the zero-error endpoint for N=4.',
        '- N18 uses a fixed 11-optimizer color order, skips missing numeric values, and verifies all exact-cell maxima before drawing. '
        'N19 shows zero Lion increments as text at the log floor. N20 verifies its reconstructed oracle mixture within the specified 0.02 tolerance.',
        '- N11 retains tiny signed SGD and shifted-support gaps. N12 retains the noisy warmth trajectory and signed SGD comparators without smoothing.',
        '- N14 C uses adjacent strip and bar axes inside the requested panel. Its transport certificate is capped as specified and labelled with its full value. '
        'N14 A plots the 2,000 reproduced random instances plus the sharp witness; its 2,572-instance annotation is explicitly the committed full V1 summary.',
        '- N16 B uses a labelled schematic position for the additive point because the source stores its width but no min/max or absolute response.',
        '- N17 preserves static/measured/control colors across panels; hatching distinguishes held-out splits. The two C dot plots share their row axis. '
        'Only the specified cheap_proxy_results cohort supplies the all-three-metrics count.',
        '- All prescribed panels and data series are included over their committed support (N03’s missing K values are documented above). No data are downsampled. Coincident curves use distinct markers or line styles. '
        'N03 direction 1 and N07 optional package rank markers are omitted, as allowed by their specs.',
        '- Panel titles preserve spec wording, with line breaks for fit. N01–N05 use their panel descriptions where no literal title is specified.',
        '- Exact analytic instances use fractions until plotting; N14 reproduces the seeded evaluator and refuses to draw on any mismatch. '
        'The W1 source describes an empirical consistency check, not a uniform certificate.','',
        '## Determinism','',
        'Fixed `svg.hashsalt`, fixed jitter seed, explicit plotting order, fixed metadata, no PDF creation/modification date, and PNG timestamp stripping. '
        'To verify, run `CUDA_VISIBLE_DEVICES=\'\' theory/.venv/bin/python theoretical_framework/figures/verify_figures.py`. '
        'It regenerates all 20 figures twice, compares every PDF/PNG plus README and the text audit byte for byte, '
        'and writes sizes and SHA-256 hashes to [verification.json](verification.json). '
        'That report records the two-run result for all 42 generated files and the exact interpreter/library environment.','',
        '## Rendered text audit','',
        '`text_audit.json` records the bounding box of every visible text artist at 300 dpi, all intersecting pairs, '
        'out-of-canvas text, and any unrendered formula characters. Off-view ticks and invisible descendants are excluded. '
        'Bounding boxes are conservative, especially for rotated and multiline text. No overlapping pair is silently exempted.','',
        '| Figure | Text artists | Bounding-box overlaps | Outside canvas |','| --- | ---: | ---: | ---: |']
    for basename,audit in d.audits.items():
        lines.append(f'| {basename[:3]} | {audit["text_count"]} | {len(audit["overlaps"])} | {len(audit["outside_canvas"])} |')
    lines+=['','## Output sizes','', '| File | Bytes |','| --- | ---: |']
    for basename in BASENAMES:
        for suffix in ('pdf','png'):
            path=HERE/f'{basename}.{suffix}'
            if path.exists():lines.append(f'| `{path.name}` | {path.stat().st_size:,} |')
    lines.append('')
    (HERE/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',help='Comma-separated node IDs, e.g. N12,N14 (default: all 20)')
    args=parser.parse_args();nodes=[b[:3] for b in BASENAMES]
    requested=set(nodes if args.only is None else [v.strip().upper() for v in args.only.split(',')])
    if not requested or not requested.issubset(nodes):parser.error('--only must contain node IDs N01 through N20')
    evidence=Evidence()
    builders=(n01,n02,n03,n04,n05,n06,n07,n08,n09,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20)
    for basename,builder in zip(BASENAMES,builders):
        node=basename[:3]
        if node not in requested:continue
        print(f'Generating {basename} …',flush=True)
        try:
            fig=builder(evidence)
            evidence.audits[basename]=save(fig,basename,evidence.captions[node])
        except Exception as exc:
            plt.close('all')
            print(f'ERROR {node}: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
            return 1
    (HERE/'text_audit.json').write_text(json.dumps(evidence.audits,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    readme(evidence)
    for message in discrepancies(evidence):print('DATA NOTE:',message,flush=True)
    print(f'Wrote {len(requested)} PDF/PNG pairs and README.md under {HERE.relative_to(ROOT)}.',flush=True)
    return 0


if __name__=='__main__':
    sys.exit(main())
