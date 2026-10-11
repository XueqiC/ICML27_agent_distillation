"""C5/C6 on the unchanged E6 panel (controlled NumPy network, not Gemma).

Only pure numerical helpers are imported from v015. Its check functions/writers
are never called. C6's measurement path does not call any gradient oracle.
Sampled ranges and ray curvature are empirical estimates, not global bounds.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / 'v015_checks'
HS = (1e-3, 3e-3, 1e-2)
GROUPS = 8
DELTA = .05
GOLD_TOL = 1e-10
LARGE_GOLD = .01  # Fixed before examining held-out outcomes.
V_BINS = (0., .25, .5, .75, 1.000000000001)
_CACHE_DIR = None


def _readonly_audit(event, args):
    """Deny writes, including bytecode/cache writes, anywhere in the pinned tree."""
    paths = []
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            paths = [path]
    elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'os.truncate'):
        paths = args[:1]
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        paths = args[:2]
    elif event in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('C5/C6 prohibit network access')
    for path in paths:
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if resolved == FROZEN or FROZEN in resolved.parents:
                raise PermissionError('v015_checks is read-only')


@lru_cache(maxsize=1)
def legacy():
    global _CACHE_DIR
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    sys.addaudithook(_readonly_audit)
    # common.py initializes pyplot and changes MPLCONFIGDIR. Initialize the
    # cached matplotlib config directory first, outside every source tree.
    _CACHE_DIR = tempfile.TemporaryDirectory(prefix='v016-mpl-')
    os.environ['MPLCONFIGDIR'] = _CACHE_DIR.name
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib.pyplot as plt
    plt.figure(); plt.close()
    sys.path.insert(0, str(FROZEN))
    try:
        import otlr
    finally:
        sys.path.remove(str(FROZEN))
        os.environ['MPLCONFIGDIR'] = _CACHE_DIR.name
    if Path(otlr.__file__).resolve() != FROZEN / 'otlr.py':
        raise RuntimeError('unexpected legacy module resolution')
    return otlr


def hashes():
    names = ('common.py', 'experiments.py', 'mathematics.py', 'otlr.py', 'SPEC.md',
             'proxy_split_freeze.json', 'proxy_models_frozen.json',
             'raw/E6_held_out_records.json', 'raw/E12_margin_rows.json',
             'validation_receipt.json', '../independent_checks/check_v014_independent.py')
    return {name: hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() for name in names}


def load_panel():
    manifest = json.loads((FROZEN / 'proxy_split_freeze.json').read_text())
    records = json.loads((FROZEN / 'raw/E6_held_out_records.json').read_text())
    o = legacy()
    # Every original snapshot occurs in a held-out cohort. Use independent
    # calibration states, executed by the unchanged deterministic warmup helper.
    # These local additions never change the frozen evaluation manifest/files.
    manifest['states'] = dict(manifest['states'])
    for warm in (3, 7):
        manifest['states'][f'cal_warm{warm}'] = o.snapshot(2, warm)
    calibration = [f'cal_warm{warm}/{recipe}/{task}' for warm in (3, 7)
                   for recipe in manifest['pilot_recipes'] for task in manifest['pilot_tasks']]
    points, menu = o.provider_menu()
    if not np.array_equal(points, manifest['message_support']):
        raise ValueError('frozen message support changed')
    if [(p['name'], list(p['law'])) for p in menu] != [
            (p['name'], p['law']) for p in manifest['providers']]:
        raise ValueError('frozen provider menu changed')
    return manifest, records, calibration, points, menu


def exact_groups(increments, laws, groups=GROUPS):
    """Finite-law enumeration: weighted exact means, not unweighted samples.

    With fewer support atoms than G, each group integrates the same entire law;
    only omega varies. This does not claim G independent increment draws.
    """
    means = laws @ increments
    return np.repeat(means[None], groups, axis=0)


def empirical_bernstein(z, candidates, delta=DELTA):
    z = np.asarray(z, float)
    if len(z) < 2 or candidates < 1:
        raise ValueError('at least two groups and one nonreference candidate required')
    variance = float(np.var(z, ddof=1))
    width = float(np.ptp(z))
    logterm = float(np.log(4 * candidates**2 / delta))
    radius = np.sqrt(2 * variance * logterm / len(z)) + 7 * width * logterm / (3 * (len(z)-1))
    return dict(e_stat=float(radius), variance=variance, B=width, logterm=logterm)


def fit_alignment(scores, gold):
    """Zero-intercept LS on calibration pairs only; reject nonpositive alpha."""
    scores, gold = np.asarray(scores, float), np.asarray(gold, float)
    denom = float(scores @ scores)
    alpha = float(scores @ gold / denom) if denom > 0 else None
    if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        return dict(alpha=alpha, epsilon_sur=None, valid=False, n_pairs=len(scores))
    residual = np.abs(gold - alpha*scores)
    return dict(alpha=alpha, epsilon_sur=float(np.max(residual, initial=0)/alpha),
                max_calibration_residual=float(np.max(residual, initial=0)),
                valid=True, n_pairs=len(scores))


class TailMeter:
    """Count actual trajectory executions, not vectorized function calls."""
    def __init__(self, key, manifest, points, laws):
        self.o = legacy()
        state, recipe, task = key.split('/')
        self.key, self.task = key, task
        self.state = {k: np.asarray(v) if k in ('theta', 'm', 'v') else v
                      for k, v in manifest['states'][state].items()}
        self.recipe = manifest['recipes'][int(recipe)]
        self.x = self.o.state_vector(self.state)
        self.scale = np.r_[np.ones(17), np.full(34, self.recipe['eta'])]
        self.tail, self.weights = self.o.complete_sequences(points, laws[0], 7)
        self.counts = Counter()
        self.updates = Counter()

    def evaluate(self, x, category, tail=None, pairwise=False):
        labels = self.tail if tail is None else tail
        replicas = len(x) if pairwise else len(x)*len(labels)
        self.counts[category] += replicas
        self.updates[category] += replicas*labels.shape[1]
        return self.o.continuation(x, labels, self.weights, 2, self.state['step']+1,
                                  self.recipe['eta'], self.recipe['opt'], 1., self.task, 3,
                                  self.recipe['eps'], pairwise=pairwise)

    def increments(self, points):
        return self.o.increment_panel(self.x, points, 2, self.state['step']+1,
                                      self.recipe['eta'], self.recipe['opt'], 1., self.recipe['eps'])


def measure_fd(meter, means, omega, h, length, name):
    """CRN scores and measured normalized curvature along [0,h*mu].

    Five forward points (0,h/4,h/2,3h/4,h) stay in nonnegative-v states.
    Three central second differences and center Richardson extrapolation use
    the same omega. Their maximum is fixed without consulting any gold value.
    """
    G, J, D = means.shape
    tails = np.repeat(omega[:, :length], J, axis=0)
    mu = means.reshape(-1, D)
    fractions = np.array([0., .25, .5, .75, 1.])
    states = np.concatenate([meter.x + f*h*mu for f in fractions])
    # Endpoint executions are the score cost; the other four are fd overhead.
    values = meter.evaluate(states, name+'_fd_and_score', tail=tails, pairwise=True).reshape(5, G, J)
    score_values = values[-1]/h
    second = (values[2:]-2*values[1:-1]+values[:-2])/(h/4)**2
    coarse = (values[4]-2*values[2]+values[0])/(h/2)**2
    extrapolated = (4*second[1]-coarse)/3
    norm2 = np.sum((means*meter.scale)**2, axis=-1)
    denom = np.maximum(norm2, 1e-300)
    normalized = np.concatenate([np.abs(second)/denom, (np.abs(extrapolated)/denom)[None]])
    M = float(np.max(np.where(norm2[None] > 1e-28, normalized, 0.)))
    error = float(np.max(np.where(norm2 > 1e-28, abs(second[1]-coarse)/(3*denom), 0.)))
    return dict(h=h, values=values, group_provider_values=score_values,
                M_fd=M, M_fd_difference_error=error, norm2=norm2,
                finite=bool(np.all(np.isfinite(values)) and np.isfinite(M)))


def measure_cell(key, manifest, points, laws, quick=False, calibration=False):
    """Measurement-only path: never calls full_gradient or richardson."""
    from check_v016 import rng_for
    meter = TailMeter(key, manifest, points, laws)
    inc, raw = meter.increments(points)
    means = exact_groups(inc[0], laws)
    state, recipe, _ = key.split('/')
    state_id = manifest['states'][state]['step']
    rng = rng_for(6, state_id, int(recipe))
    indices = rng.choice(len(points), size=(GROUPS, 7), p=laws[0])
    omega = points[indices]
    surrogates = (('short', 1),) if calibration else (('aligned', 7), ('short', 1))
    measurements = {name: [measure_fd(meter, means, omega, h, length, name) for h in HS]
                    for name, length in surrogates}
    if calibration:
        curvature = dict(M2=None, invalid_probes=0, reason='calibration only needs short scores and full-tail gold')
    elif meter.task == 'threshold':
        curvature = dict(M2=None, invalid_probes=0, reason='discontinuous threshold objective')
    else:
        curvature = meter.o.curvature_search(
            lambda x: meter.evaluate(x, 'remainder_curvature'), meter.x, inc, meter.scale, quick)
    moments = laws @ np.sum((inc[0]*meter.scale)**2, axis=1)
    # Outcome evaluation happens AFTER all range/curvature measurements.
    terminal_by_message = meter.evaluate(meter.x+inc[0], 'full_tail_gold_audit')
    returns = laws @ terminal_by_message
    return dict(cell=key, task=meter.task, x=meter.x, increments=inc[0], gradients=raw[0],
                means=means, omega_indices=indices, measurements=measurements,
                curvature=curvature, second_moments=moments, returns=returns,
                tail_evaluations=dict(meter.counts), tail_update_steps=dict(meter.updates))


def pair_measurement(cell, name, hi, j, k, alignment):
    fd = cell['measurements'][name][hi]
    z = fd['group_provider_values'][:, j]-fd['group_provider_values'][:, k]
    stat = empirical_bernstein(z, cell['means'].shape[1]-1)
    e_fd = .5*fd['h']*fd['M_fd']*float(np.max(fd['norm2'][:, j]+fd['norm2'][:, k]))
    M = cell['curvature']['M2']
    e2 = .5*M*(cell['second_moments'][j]+cell['second_moments'][k]) if M is not None else None
    eligible = (cell['task'] != 'threshold' and fd['finite'] and alignment['valid']
                and e2 is not None and np.isfinite(e2) and cell['curvature']['invalid_probes'] == 0)
    alpha, eps = alignment['alpha'], alignment['epsilon_sur']
    radius = stat['e_stat']+e_fd+eps+e2/alpha if eligible else None
    score = float(np.mean(z))
    return dict(h=fd['h'], score=score, **stat, e_fd=e_fd, e_2=e2,
                alpha=alpha, epsilon_sur=eps, radius=radius,
                task_radius=alpha*radius if eligible else None,
                eligible=bool(eligible), certified=bool(eligible and abs(score)>radius),
                Z=z)


def choose_h(options):
    available = [r for r in options if r['eligible']]
    # A radius-only rule; outcomes and margin/coverage never enter selection.
    return min(available, key=lambda r: (r['task_radius'], r['h'])) if available else options[0]


def sign_stats(rows, ranker):
    nonzero = [r for r in rows if abs(r['gold']) > GOLD_TOL]
    finite = [r for r in nonzero if r['scores'][ranker] is not None and np.isfinite(r['scores'][ranker])]
    wrong = sum(r['scores'][ranker]*r['gold'] < 0 for r in finite)
    ties = sum(r['scores'][ranker] == 0 for r in finite)
    return dict(rows=len(rows), nonzero_gold=len(nonzero), available=len(finite),
                unavailable=len(nonzero)-len(finite), wrong=int(wrong), ties=int(ties),
                wrong_rate=wrong/len(finite) if finite else None)


def coverage(rows, name, hi=None):
    chosen = [(r, r['certificates'][name]['selected' if hi is None else 'all_h']
               if hi is None else r['certificates'][name]['all_h'][hi]) for r in rows]
    eligible = [(r, m) for r, m in chosen if m['eligible']]
    certified = [(r, m) for r, m in eligible if m['certified']]
    large = [(r, m) for r, m in eligible if abs(r['gold']) >= LARGE_GOLD]
    missed = sum(not m['certified'] for r, m in large)
    return dict(rows=len(rows), eligible=len(eligible), certified=len(certified),
                fraction_all=len(certified)/len(rows) if rows else None,
                fraction_eligible=len(certified)/len(eligible) if eligible else None,
                wrong_certified=sum(abs(r['gold'])>GOLD_TOL and m['score']*r['gold']<0 for r, m in certified),
                large_gold_threshold=LARGE_GOLD, large_gold_pairs=len(large),
                missed_large_gold=int(missed), missed_large_fraction=missed/len(large) if large else None,
                chosen_h=dict(Counter(str(m['h']) for r, m in eligible)))


def _sum_cost(cells, field):
    result = Counter()
    for cell in cells:
        result.update(cell[field])
    return dict(result)


def cost_report(cells, calibration_cells, providers):
    held = _sum_cost(cells, 'tail_evaluations')
    calibration = _sum_cost(calibration_cells, 'tail_evaluations')
    scoring = len(cells)*providers*GROUPS*len(HS)
    audit = len(cells)*providers*1016
    result = dict(unique_held_cells=len(cells), calibration_cells=len(calibration_cells),
                  providers=providers, held_evaluations=held, calibration_evaluations=calibration,
                  held_update_steps=_sum_cost(cells, 'tail_update_steps'),
                  calibration_update_steps=_sum_cost(calibration_cells, 'tail_update_steps'),
                  all_executed_tail_evaluations=sum(held.values())+sum(calibration.values()),
                  per_surrogate_score_endpoints_all_h=scoring,
                  per_surrogate_score_endpoints_one_h=len(cells)*providers*GROUPS,
                  full_tail_audit_reference=dict(M=4, delta=.05, e=.1, replicas_per_candidate=1016,
                                                matched_panel_candidate_executions=audit,
                                                M8_replicas_per_candidate=int(np.ceil(2*np.log(16/.05)/.1**2))))
    result['by_surrogate'] = {}
    for name in ('aligned', 'short'):
        # Shared e2 is counted once in combined totals and explicitly charged to
        # each stand-alone surrogate comparison. Held gold is validation only.
        setup = held.get(name+'_fd_and_score', 0)+held.get('remainder_curvature', 0)
        cal = sum(calibration.values()) if name == 'short' else 0
        result['by_surrogate'][name] = dict(
            held_score_and_fd=held.get(name+'_fd_and_score', 0),
            held_remainder_curvature=held.get('remainder_curvature', 0),
            calibration_all_executed=cal, measurement_total=setup+cal,
            measurement_over_full_audit=(setup+cal)/audit,
            validation_gold_excluded=held.get('full_tail_gold_audit', 0))
    return result


@lru_cache(maxsize=2)
def measured_panel(quick=False):
    """Frozen measurement/calibration result shared by C5 and C6 in one run."""
    start = time.perf_counter()
    before = hashes()
    manifest, records, calibration_keys, points, menu = load_panel()
    laws = np.array([p['law'] for p in menu])
    cal = {}
    for key in calibration_keys:
        cal[key] = measure_cell(key, manifest, points, laws, quick, calibration=True)
    alignments = []
    for hi, h in enumerate(HS):
        scores, gold = [], []
        for cell in cal.values():
            v = cell['measurements']['short'][hi]['group_provider_values'].mean(axis=0)
            for j, k in itertools.combinations(range(len(menu)), 2):
                scores.append(v[j]-v[k]); gold.append(cell['returns'][j]-cell['returns'][k])
        alignments.append(fit_alignment(scores, gold))
    cells = {}
    for index, key in enumerate(sorted({r['cell'] for r in records})):
        cells[key] = measure_cell(key, manifest, points, laws, quick)
        if index % 8 == 0:
            print(f'  C6 measured {index+1}/40 frozen cells', flush=True)
    pairs, reference_rows = [], []
    unique = list(dict.fromkeys((r['split'], r['cell']) for r in records))
    for split, key in unique:
        cell = cells[key]
        for j, k in itertools.combinations(range(len(menu)), 2):
            certs = {}
            for name in ('aligned', 'short'):
                options = [pair_measurement(cell, name, hi, j, k,
                    dict(alpha=1., epsilon_sur=0., valid=True) if name == 'aligned' else alignments[hi])
                           for hi in range(len(HS))]
                certs[name] = dict(all_h=options, selected=choose_h(options))
            pairs.append(dict(split=split, cell=key, recipe=int(key.split('/')[1]),
                              task=cell['task'], j=j, k=k, gold=float(cell['returns'][j]-cell['returns'][k]),
                              certificates=certs))
        # Preserve E12's orientation candidate minus reference, and its self rows.
        for j in range(len(menu)):
            certs = {}
            for name in ('aligned', 'short'):
                options = [pair_measurement(cell, name, hi, j, 0,
                    dict(alpha=1., epsilon_sur=0., valid=True) if name == 'aligned' else alignments[hi])
                           for hi in range(len(HS))]
                certs[name] = dict(all_h=options, selected=choose_h(options))
            reference_rows.append(dict(split=split, cell=key, j=j, k=0,
                                       gold=float(cell['returns'][j]-cell['returns'][0]), certificates=certs))
    lookup = {(r['split'], r['cell'], r['j']): r for r in reference_rows}
    names = {p['name']: i for i, p in enumerate(menu)}
    gold_errors = [abs(lookup[(r['split'], r['cell'], names[r['provider']])]['gold']-r['gold']) for r in records]
    return dict(manifest=manifest, records=records, calibration_keys=calibration_keys,
                menu=menu, points=points, laws=laws, cells=cells, calibration_cells=cal,
                alignments=alignments, pairs=pairs, reference_rows=reference_rows,
                max_gold_error=max(gold_errors), costs=cost_report(list(cells.values()), list(cal.values()), len(menu)),
                before=before, after=hashes(), runtime_seconds=time.perf_counter()-start)


def c6(quick=False):
    from check_v016 import Check
    p = measured_panel(quick)
    check = Check('C6', 'Corollary 5.4: no wrong aligned empirical first-order certificates on the frozen panel; no oracle gradient used')
    check.assert_le('frozen_imports_and_artifacts_unchanged', p['before'] != p['after'], 0, 0)
    check.assert_le('calibration_heldout_overlap', len(set(p['calibration_keys']) & set(p['cells'])), 0, 0)
    for key, cell in p['calibration_cells'].items():
        check.assert_le('calibration_state_not_in_heldout', any(np.array_equal(cell['x'], held['x'])
                        for held in p['cells'].values()), 0, key)
    check.assert_le('full_tail_gold_matches_frozen_E6', p['max_gold_error'], 1e-14, 0)
    check.assert_le('positive_calibration_alpha', [not a['valid'] for a in p['alignments']], 0, 0)
    for row in p['pairs']:
        for m in row['certificates']['aligned']['all_h']:
            if m['certified']:
                check.assert_le('aligned_certified_wrong_sign', int(abs(row['gold'])>GOLD_TOL and row['gold']*m['score']<0),
                                0, [row['split'], row['cell'], row['j'], row['k'], m['h']])
    check.data.append(dict(kind='protocol', h_grid=HS, groups=GROUPS, requested_n=64,
        increments_per_provider=[int(np.count_nonzero(law)) for law in p['laws']],
        sampling='Exact weighted support enumeration in every group; only the eight reference tails are sampled independently.',
        M=len(p['menu'])-1, delta=DELTA, calibration_cells=p['calibration_keys'],
        calibration_by_h=[dict(h=h, **a) for h, a in zip(HS, p['alignments'])],
        providers=[x['name'] for x in p['menu']], costs=p['costs'], measurement_runtime_seconds=p['runtime_seconds'],
        frozen_sha256_before=p['before'], frozen_sha256_after=p['after'],
        max_frozen_gold_error=p['max_gold_error'],
        limitations=[
            'Empirical certificate only: B=max(Z)-min(Z) over eight groups is not a proven support range.',
            'Curvatures are unpadded ray searches/CRN second differences, not certified global suprema.',
            'delta=.05 is nominal per state and per h; h selection is not a simultaneous three-h confidence guarantee.',
            'Short-tail calibration uses actual full-tail gold, not an unbiased oracle first-order target; transfer to holdouts is unproved.',
            'Threshold objectives are ineligible. Calibration uses separately executed warm3/warm7 snapshots and only the pilot recipes/tasks.',
            'Scores reuse evaluations across pairs and split duplicates; costs count actual unique-cell executions.',
            'The 1016 comparator uses M=4; the panel has M=8 and needs 1154 at the same Hoeffding settings.'],
        calibration_state_source='Unchanged v015 common.snapshot(2,warm), warm in (3,7), deterministic .25 labels, eta=.03 Adam warmup; 10 executed prefix updates total.',
        raw_replicates=dict(cells=p['cells'], calibration_cells=p['calibration_cells'],
                            calibration_states={k:v for k,v in p['manifest']['states'].items() if k.startswith('cal_')})))
    for name in ('aligned', 'short'):
        check.data.append(dict(kind='surrogate', surrogate=name,
            all_pairs=coverage(p['pairs'], name), reference_rows=coverage(p['reference_rows'], name),
            oracle_comparison=dict(margin=56, eligible_reference_rows=396, fraction=56/396),
            all_h=[dict(h=h, all_pairs=coverage(p['pairs'], name, hi),
                        reference_rows=coverage(p['reference_rows'], name, hi)) for hi, h in enumerate(HS)],
            per_split={s: coverage([r for r in p['pairs'] if r['split']==s], name)
                       for s in dict.fromkeys(r['split'] for r in p['pairs'])},
            raw_replicates=dict(pairs=[{k:v for k,v in r.items() if k!='certificates'} |
                dict(certificate=r['certificates'][name]) for r in p['pairs']],
                reference_rows=[{k:v for k,v in r.items() if k!='certificates'} |
                dict(certificate=r['certificates'][name]) for r in p['reference_rows']])) )
    return [check]


def cosine_scores(means, gradient):
    norm = np.linalg.norm(means, axis=1)*np.linalg.norm(gradient)
    return np.divide(means@gradient, norm, out=np.zeros(len(means)), where=norm>1e-28)


def oracle_cell(key, p):
    """C5-only oracle diagnostics. Nothing from this function feeds C6."""
    meter = TailMeter(key, p['manifest'], p['points'], p['laws'])
    cell = p['cells'][key]
    x = meter.x
    relative = np.r_[np.ones(17), np.maximum(abs(x[0, 17:34]), 1e-5), np.maximum(x[0, 34:], 1e-8)]
    grad, graderr = meter.o.full_gradient(lambda a: meter.evaluate(a, 'oracle_gradient'), x, relative)
    mu = p['laws'] @ cell['increments']
    components = np.stack([mu[:, b*17:(b+1)*17]@grad[b*17:(b+1)*17] for b in range(3)], axis=1)
    # The immediate task objective gradient is also measured independently of V.
    task_grad, _ = meter.o.full_gradient(lambda a: meter.o.terminal(a, 2, cell['task'], 3), x,
                                       np.r_[np.ones(17), np.ones(34)*1e-8])
    direction = mu[:, :17] if meter.recipe['opt']=='Adam' else p['laws']@cell['gradients']
    less = cosine_scores(direction, task_grad[:17])
    utility_less = less if meter.recipe['opt']=='Adam' else -less
    contrast = mu-mu[0]
    direct, err = meter.o.richardson(lambda a: meter.evaluate(a, 'oracle_directional_check'),
                                   np.repeat(x, len(mu), axis=0), contrast)
    return dict(gradient=grad, gradient_error=graderr, components=components,
                less=less, utility_less=utility_less, task_gradient=task_grad[:17],
                directional_score=direct, directional_error=err,
                max_contraction_error=float(np.max(abs(components.sum(axis=1)-components[0].sum()-direct))),
                tail_evaluations=dict(meter.counts))


def c5(quick=False):
    from check_v016 import Check
    p = measured_panel(quick)
    check = Check('C5', 'Proposition 4.3: frozen-panel ranker errors and v-channel bins; E12 oracle margin reproduction')
    audits = {key: oracle_cell(key, p) for key in p['cells']}
    ranks = ('less_cosine', 'parameter_only', 'full_state', 'measured_aligned', 'measured_short', 'less_utility_orientation')
    rows = []
    for pair in p['pairs']:
        j, k, key = pair['j'], pair['k'], pair['cell']
        a = audits[key]
        components = a['components'][j]-a['components'][k]
        total = float(np.abs(components).sum())
        v_share = float(abs(components[2])/total) if total>1e-20 and pair['task']!='threshold' else None
        scores = dict(less_cosine=float(a['less'][j]-a['less'][k]), parameter_only=float(components[0]),
                      full_state=float(components.sum()),
                      measured_aligned=pair['certificates']['aligned']['selected']['score'],
                      measured_short=pair['certificates']['short']['selected']['score'],
                      less_utility_orientation=float(a['utility_less'][j]-a['utility_less'][k]))
        rows.append({k:v for k,v in pair.items() if k!='certificates'} |
                    dict(scores=scores, oracle_components=components, v_share=v_share))
    frozen = json.loads((FROZEN / 'raw/E12_margin_rows.json').read_text())
    name_index = {m['name']: i for i, m in enumerate(p['menu'])}
    observed_margin, eligible, frozen_margin = 0, 0, 0
    for i, r in enumerate(frozen):
        a = audits[r['cell']]; j = name_index[r['provider']]
        cell = p['cells'][r['cell']]
        score = float((a['components'][j]-a['components'][0]).sum())
        bound = (.5*cell['curvature']['M2']*(cell['second_moments'][j]+cell['second_moments'][0])
                 if cell['curvature']['M2'] is not None else None)
        valid = r['task']!='threshold' and np.all(np.isfinite(a['gradient']))
        margin = bool(valid and abs(score)>bound)
        eligible += int(valid); observed_margin += int(margin); frozen_margin += int(r['margin'])
        check.assert_le('oracle_score_matches_E12', abs(score-r['oracle_score']), 2e-6, i)
        if r['margin']:
            check.assert_le('no_wrong_sign_on_frozen_E12_margin', int(abs(r['gold'])>GOLD_TOL and score*r['gold']<0), 0, i)
        if margin:
            check.assert_le('no_wrong_sign_on_recomputed_E12_margin', int(abs(r['gold'])>GOLD_TOL and score*r['gold']<0), 0, i)
        if not quick:
            check.assert_le('margin_membership_matches_E12', margin!=r['margin'], 0, i)
    check.assert_le('E12_eligible_rows', abs(eligible-396), 0, 0)
    check.assert_le('frozen_E12_margin_rows', abs(frozen_margin-56), 0, 0)
    if not quick:
        check.assert_le('recomputed_E12_margin_rows', abs(observed_margin-56), 0, 0)
    for key, a in audits.items():
        if p['cells'][key]['task']!='threshold':
            check.assert_le('oracle_coordinate_directional_agreement', a['max_contraction_error'], 2e-6, key)
    def table(subset):
        return {rank: sign_stats(subset, rank) for rank in ranks}
    bins = [dict(lo=lo, hi=min(hi, 1.), rankers=table([r for r in rows if r['v_share'] is not None and lo<=r['v_share']<hi]))
            for lo, hi in zip(V_BINS[:-1], V_BINS[1:])]
    low = [r for r in rows if r['v_share'] is not None and r['v_share']<.5]
    high = [r for r in rows if r['v_share'] is not None and r['v_share']>=.5]
    concentration = {}
    for rank in ('less_cosine', 'parameter_only'):
        l, h = sign_stats(low, rank), sign_stats(high, rank)
        concentration[rank] = dict(low=l, high=h, errors_concentrate_high=(h['wrong_rate']>l['wrong_rate'])
                                   if h['wrong_rate'] is not None and l['wrong_rate'] is not None else None)
    check.data.append(dict(kind='rankers', overall=table(rows),
        per_optimizer_configuration={str(i):dict(config=recipe, rankers=table([r for r in rows if r['recipe']==i]))
                                     for i, recipe in enumerate(p['manifest']['recipes'])},
        per_split={s:table([r for r in rows if r['split']==s]) for s in dict.fromkeys(r['split'] for r in rows)},
        v_share_bins=bins, v_share_unavailable=table([r for r in rows if r['v_share'] is None]),
        concentration=concentration,
        E12=dict(all_rows=len(frozen), eligible=eligible, frozen_margin=frozen_margin, recomputed_margin=observed_margin),
        oracle_tail_evaluations=_sum_cost(list(audits.values()), 'tail_evaluations'),
        conventions=['Gold and all utility scores orient j minus k. Nonzero gold means abs(gold)>1e-10.',
                     'Wrong sign means product<0; exact score ties are separately counted, as in E12.',
                     'Primary LESS follows the requested literal Adam delta-theta / SGD positive loss-gradient convention.',
                     'less_utility_orientation also reports SGD negative loss-gradient so both optimizers point along their updates.',
                     'Bins use oracle CONTRAST components, not the components of either individual provider.',
                     'All pairs retain the frozen split multiplicities; thresholds have no smooth v-share.'],
        raw_replicates=dict(rows=rows, oracle_cells=audits)))
    check.assert_le('frozen_files_unchanged_after_oracles', p['before'] != hashes(), 0, 0)
    return [check]
