"""Finite replacements (Theorem 5.3), separate from unchanged C5/C6.

Integrate phi(x + Delta_atom), never phi(x + E Delta). Frozen numerical
helpers, panel, gold and split conventions are imported through c5c6.
"""
from functools import lru_cache
import itertools
import time

import numpy as np

import c5c6 as old

REPLICATES = (64, 256)
SURROGATES = {'aligned': 7, 'short': 1}
RANGE_WIDTH = 2.  # Both task objectives and both surrogates are in [0, 1].
RANKER = 'paired_replacement'


def integrate_replacements(meter, increments, laws, omega, length, category):
    """One shared tail per replica; exactly weight terminal support outcomes."""
    active = np.flatnonzero(np.any(laws > 0, axis=0))
    r, atoms = len(omega), len(active)
    states = np.tile(meter.x + increments[active], (r, 1))
    tails = np.repeat(omega[:, :length], atoms, axis=0)
    values = meter.evaluate(states, category, tail=tails, pairwise=True).reshape(r, atoms)
    return values @ laws[:, active].T, values, active


def bernstein(z, candidates, width):
    """Theorem 5.3: width is a known range or explicitly empirical diagnostic."""
    z = np.asarray(z, float)
    if z.ndim != 1 or len(z) < 2 or candidates < 1 or width < 0:
        raise ValueError('invalid Bernstein inputs')
    variance = float(np.var(z, ddof=1))
    logterm = float(np.log(4*candidates**2/old.DELTA))
    variance_term = float(np.sqrt(2*variance*logterm/len(z)))
    range_term = float(7*width*logterm/(3*(len(z)-1)))
    return dict(e_stat=variance_term+range_term, variance=variance, B=float(width),
                logterm=logterm, variance_term=variance_term, range_term=range_term)


def variance_comparison(z, independent_z):
    crn, independent = float(np.var(z, ddof=1)), float(np.var(independent_z, ddof=1))
    return dict(crn_variance=crn, independent_variance=independent,
                variance_ratio=crn/independent if independent > 0 else None)


def measure_cell(key, manifest, points, laws, calibration=False):
    """Only actual updates/rollouts: no derivative or curvature calls."""
    from check_v016 import rng_for
    meter = old.TailMeter(key, manifest, points, laws)
    increments, _ = meter.increments(points)
    state, recipe, _ = key.split('/')
    seed_path = (66, manifest['states'][state]['step'], int(recipe))
    r = max(REPLICATES)
    indices = rng_for(*seed_path, 0).choice(len(points), size=(r, 7), p=laws[0])
    names = ('short',) if calibration else tuple(SURROGATES)
    measurements = {}
    for name in names:
        values, atom_values, active = integrate_replacements(
            meter, increments[0], laws, points[indices], SURROGATES[name], name+'_crn')
        measurements[name] = dict(provider_values=values, atom_values=atom_values, active_atoms=active)
    independent_indices = None
    if not calibration:
        independent_indices = np.stack([rng_for(*seed_path, 1, j).choice(
            len(points), size=(r, 7), p=laws[0]) for j in range(len(laws))])
        for name in names:
            # Every candidate has its own independent omega stream; atoms within
            # a candidate still share that omega before exact integration.
            independent = np.empty((r, len(laws)))
            for j, law in enumerate(laws):
                values, _, _ = integrate_replacements(meter, increments[0], law[None],
                    points[independent_indices[j]], SURROGATES[name], name+'_independent')
                independent[:, j] = values[:, 0]
            measurements[name]['independent_provider_values'] = independent
    # Gold is evaluated after measurements and cannot affect seeds/ranges.
    returns = laws @ meter.evaluate(meter.x+increments[0], 'full_tail_gold_audit')
    return dict(cell=key, task=meter.task, x=meter.x, increments=increments[0],
                seed_path=seed_path, omega_indices=indices, independent_omega_indices=independent_indices,
                measurements=measurements, returns=returns,
                tail_evaluations=dict(meter.counts), tail_update_steps=dict(meter.updates))


def calibrate(cells, r):
    """Remark 5.6 targets finite full-tail gold differences, not derivatives."""
    scores, gold = [], []
    for cell in cells.values():
        values = cell['measurements']['short']['provider_values'][:r].mean(axis=0)
        for j, k in itertools.combinations(range(len(values)), 2):
            scores.append(values[j]-values[k])
            gold.append(cell['returns'][j]-cell['returns'][k])
    return dict(**old.fit_alignment(scores, gold), scores=np.array(scores), gold=np.array(gold))


def pair_record(cell, name, r, j, k, alignment):
    values = cell['measurements'][name]
    z = values['provider_values'][:r, j]-values['provider_values'][:r, k]
    independent = values['independent_provider_values'][:r, j]-values['independent_provider_values'][:r, k]
    finite = bool(np.all(np.isfinite(z)))
    eligible = finite and alignment['valid']
    score = float(np.mean(z))
    radii = {}
    for kind, width in (('a_priori', RANGE_WIDTH), ('observed', float(np.ptp(z)))):
        stat = bernstein(z, values['provider_values'].shape[1]-1, width)
        radius = stat['e_stat']+alignment['epsilon_sur'] if eligible else None
        radii[kind] = dict(**stat, radius=radius,
            task_radius=alignment['alpha']*radius if eligible else None,
            certified=bool(eligible and abs(score) > radius))
    return dict(j=j, k=k, gold=float(cell['returns'][j]-cell['returns'][k]), score=score,
                eligible=eligible, alpha=alignment['alpha'], epsilon_sur=alignment['epsilon_sur'],
                radii=radii, **variance_comparison(z, independent), Z=z, Z_independent=independent)


def coverage(rows, kind):
    eligible = [x for x in rows if x['eligible']]
    certified = [x for x in eligible if x['radii'][kind]['certified']]
    large = [x for x in eligible if abs(x['gold']) >= old.LARGE_GOLD]
    missed = sum(not x['radii'][kind]['certified'] for x in large)
    return dict(rows=len(rows), eligible=len(eligible), certified=len(certified),
        fraction_eligible=len(certified)/len(eligible) if eligible else None,
        wrong_certified=sum(abs(x['gold'])>old.GOLD_TOL and x['score']*x['gold']<0 for x in certified),
        certified_gold_ties=sum(abs(x['gold'])<=old.GOLD_TOL for x in certified),
        large_gold_threshold=old.LARGE_GOLD, large_gold_pairs=len(large), missed_large_gold=missed,
        missed_large_fraction=missed/len(large) if large else None)


def variance_summary(rows):
    ratios = [x['variance_ratio'] for x in rows if x['variance_ratio'] is not None]
    crn = sum(x['crn_variance'] for x in rows)
    independent = sum(x['independent_variance'] for x in rows)
    return dict(rows=len(rows), defined_ratios=len(ratios), zero_independent_variance=len(rows)-len(ratios),
        ratio_of_summed_variances=crn/independent if independent > 0 else None,
        pair_ratio_quantiles=dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                                     np.quantile(ratios, [0, .25, .5, .75, 1.]))) if ratios else {},
        fraction_variance_reduced=sum(x<1 for x in ratios)/len(ratios) if ratios else None)


def costs(cells, calibration, laws):
    """Physical trajectory counts; r=64 is a prefix, never double charged."""
    held = old._sum_cost(cells, 'tail_evaluations')
    cal = old._sum_cost(calibration, 'tail_evaluations')
    count, providers = len(cells), len(laws)
    atoms = int(np.count_nonzero(np.any(laws>0, axis=0)))
    unshared_atoms = int(np.count_nonzero(laws))
    comparator = old.cost_report(cells, calibration, providers)['full_tail_audit_reference']
    audit = comparator['matched_panel_candidate_executions']
    rows = []
    for r in REPLICATES:
        for name, length in SURROGATES.items():
            score = count*atoms*r
            cal_score = len(calibration)*atoms*r if name == 'short' else 0
            cal_gold = cal.get('full_tail_gold_audit', 0) if name == 'short' else 0
            total = score+cal_score+cal_gold
            updates = (score+cal_score)*length+7*cal_gold
            diagnostic = count*unshared_atoms*r
            rows.append(dict(r=r, surrogate=name, held_score_evaluations=score,
                logical_candidate_values=count*providers*r,
                unshared_candidate_atom_evaluations=count*unshared_atoms*r,
                calibration_score_evaluations=cal_score, calibration_gold_evaluations=cal_gold,
                measurement_total=total, measurement_update_steps=updates,
                measurement_over_full_audit=total/audit,
                update_steps_over_full_audit=updates/(7*audit),
                independent_variance_diagnostic_evaluations=diagnostic,
                measurement_plus_diagnostic_over_full_audit=(total+diagnostic)/audit))
    return dict(held_evaluations=held, calibration_evaluations=cal,
        held_update_steps=old._sum_cost(cells, 'tail_update_steps'),
        calibration_update_steps=old._sum_cost(calibration, 'tail_update_steps'),
        all_executed_tail_evaluations=sum(held.values())+sum(cal.values()),
        all_executed_tail_update_steps=sum(sum(c['tail_update_steps'].values()) for c in cells+calibration),
        full_tail_audit_reference=comparator, by_r_surrogate=rows,
        unique_held_cells=count, calibration_cells=len(calibration),
        support_atoms=atoms, unshared_provider_support_atoms=unshared_atoms,
        calibration_warmup_updates=10, held_validation_gold_excluded=held['full_tail_gold_audit'],
        reuse='r=64 is the first 64 of 256 IID replicas; splits and pairs reuse each unique cell. '
              'Independent-tail diagnostics are extra executions. No curvature or gradient overhead.')


@lru_cache(maxsize=1)
def measured_panel():
    start = time.perf_counter()
    before = old.hashes()
    manifest, records, cal_keys, points, menu = old.load_panel()
    laws = np.array([p['law'] for p in menu])
    calibration = {key: measure_cell(key, manifest, points, laws, True) for key in cal_keys}
    # Freeze calibration before any held-out measurements or gold evaluations.
    alignments = {r: calibrate(calibration, r) for r in REPLICATES}
    cells = {key: measure_cell(key, manifest, points, laws) for key in sorted({x['cell'] for x in records})}
    provider_ids = {p['name']: j for j, p in enumerate(menu)}
    errors = [abs(cells[x['cell']]['returns'][provider_ids[x['provider']]]-
                  cells[x['cell']]['returns'][0]-x['gold']) for x in records]
    panels = {}
    for r in REPLICATES:
        for name in SURROGATES:
            alignment = dict(alpha=1., epsilon_sur=0., valid=True) if name == 'aligned' else alignments[r]
            pairs, reference = [], []
            for split, key in dict.fromkeys((x['split'], x['cell']) for x in records):
                cell = cells[key]
                meta = dict(split=split, cell=key, recipe=int(key.split('/')[1]), task=cell['task'])
                pairs.extend(dict(**meta, **pair_record(cell, name, r, j, k, alignment))
                             for j, k in itertools.combinations(range(len(menu)), 2))
                reference.extend(dict(**meta, **pair_record(cell, name, r, j, 0, alignment))
                                 for j in range(len(menu)))
            panels[r, name] = dict(pairs=pairs, reference_rows=reference)
    return dict(manifest=manifest, menu=menu, laws=laws, calibration_keys=cal_keys,
        calibration_cells=calibration, cells=cells, alignments=alignments, panels=panels,
        max_gold_error=max(errors), costs=costs(list(cells.values()), list(calibration.values()), laws),
        before=before, after=old.hashes(), runtime_seconds=time.perf_counter()-start)


def c6b(quick=False):
    from check_v016 import Check
    p = measured_panel()  # C6b uses the requested r in quick mode too.
    check = Check('C6b', 'Theorem 5.3: finite CRN replacements; no wrong aligned a-priori certificates')
    check.assert_le('frozen_files_unchanged', p['before'] != p['after'], 0, 0)
    check.assert_le('full_tail_gold_matches_frozen_E6', p['max_gold_error'], 1e-14, 0)
    check.assert_le('calibration_heldout_overlap', len(set(p['calibration_keys']) & set(p['cells'])), 0, 0)
    for key, cell in p['calibration_cells'].items():
        check.assert_le('calibration_state_not_in_heldout', any(np.array_equal(cell['x'], h['x'])
                        for h in p['cells'].values()), 0, key)
    for r, a in p['alignments'].items():
        check.assert_le('positive_calibration_alpha', not a['valid'], 0, r)
    for key, cell in {**p['cells'], **p['calibration_cells']}.items():
        for name, m in cell['measurements'].items():
            for field in ('atom_values', 'provider_values', 'independent_provider_values'):
                if field in m:
                    check.assert_le('bounded_finite_surrogate', np.maximum(-m[field], m[field]-1), 1e-14,
                                    [key, name, field])
    check.data.append(dict(kind='protocol', theorem='5.3', replicates=REPLICATES, M=len(p['menu'])-1,
        delta=old.DELTA, surrogate_range=[0., 1.], difference_range=[-1., 1.], B_a_priori=RANGE_WIDTH,
        sampling='Exact law-weighted terminal outcomes after each actual support-atom increment; '
                 'one IID reference tail per replica shared across all candidates.',
        seeds='SeedSequence([160160,66,warmup_step,recipe,stream,...]); stream 0 CRN, 1 + candidate independent.',
        calibration_cells=p['calibration_keys'], calibration_by_r={str(r):{k:v for k,v in a.items()
            if k not in ('scores', 'gold')} for r, a in p['alignments'].items()},
        providers=[x['name'] for x in p['menu']], costs=p['costs'],
        max_frozen_gold_error=p['max_gold_error'], frozen_sha256_before=p['before'], frozen_sha256_after=p['after'],
        measurement_runtime_seconds=p['runtime_seconds'],
        limitations=['A-priori empirical Bernstein is simultaneous over candidate pairs at ONE fixed state/r/surrogate; '
                     'not jointly over all states or both r values.',
                     'Observed B is a diagnostic, not a known support bound or guaranteed certificate.',
                     'Short alignment is a calibration residual envelope; transfer to held-out states is assumed, not proved.',
                     'Threshold objectives are bounded and eligible here; matched C6/E12 subsets exclude them.',
                     'Zero certified wrong signs is vacuous whenever coverage is zero.',
                     'Split duplicates are retained in tables, not counted as extra independent states.',
                     'The comparator is C6\'s recorded 1016 tails/candidate (M=4); M=8 requires 1154.'],
        raw_replicates=dict(cells=p['cells'], calibration_cells=p['calibration_cells'], alignments=p['alignments'],
                            laws=p['laws'], calibration_states={k:v for k,v in p['manifest']['states'].items()
                                                               if k.startswith('cal_')})))
    for (r, name), panel in p['panels'].items():
        pairs, reference = panel['pairs'], panel['reference_rows']
        smooth = [x for x in pairs if x['task'] != 'threshold']
        matched = [x for x in reference if x['task'] != 'threshold']
        for kind in ('a_priori', 'observed'):
            if name == 'aligned' and kind == 'a_priori':
                for i, x in enumerate(pairs+reference):
                    check.assert_le('aligned_a_priori_certified_wrong_sign',
                        int(x['radii'][kind]['certified'] and abs(x['gold'])>old.GOLD_TOL
                            and x['gold']*x['score']<0), 0, [r, i])
            check.data.append(dict(kind='certificate', r=r, surrogate=name, range_variant=kind,
                all_pairs=coverage(pairs, kind), c6_comparable_pairs=coverage(smooth, kind),
                reference_rows=coverage(reference, kind), e12_comparable_reference_rows=coverage(matched, kind),
                threshold_pairs=coverage([x for x in pairs if x['task']=='threshold'], kind),
                oracle_comparison=dict(certified=56, eligible=396, fraction=56/396),
                c6_comparison=dict(certified=0, eligible=1584),
                variance=variance_summary(pairs),
                per_split={s:coverage([x for x in pairs if x['split']==s], kind)
                           for s in dict.fromkeys(x['split'] for x in pairs)},
                costs=next(x for x in p['costs']['by_r_surrogate'] if x['r']==r and x['surrogate']==name),
                raw_replicates=panel if kind == 'a_priori' else {}))
    return [check]


def c5b(quick=False, baseline=None):
    from check_v016 import Check
    if baseline is None:
        baseline = old.c5(quick)[0].result()
    p = measured_panel()
    original = baseline['data'][0]
    lookup = {(x['split'], x['cell'], x['j'], x['k']): x
              for x in p['panels'][256, 'aligned']['pairs']}
    rows = []
    check = Check('C5b', 'Theorem 5.3 aligned r=256 finite replacement as the fifth C5 ranker')
    for i, row in enumerate(original['raw_replicates']['rows']):
        measured = lookup[row['split'], row['cell'], row['j'], row['k']]
        check.assert_le('same_C5_gold', abs(row['gold']-measured['gold']), 1e-14, i)
        check.assert_le('finite_replacement_score', not np.isfinite(measured['score']), 0, i)
        rows.append({**row, 'scores': {**row['scores'], RANKER: measured['score']}})
    ranks = tuple(original['overall'])+(RANKER,)
    def table(subset):
        return {rank: old.sign_stats(subset, rank) for rank in ranks}
    high = [x for x in rows if x['v_share'] is not None and x['v_share']>=.5]
    high_table = table(high)
    rate = high_table[RANKER]['wrong_rate']
    comparisons = {rank:dict(baseline_wrong=high_table[rank]['wrong'],
        replacement_wrong=high_table[RANKER]['wrong'],
        improves_wrong_rate=rate<high_table[rank]['wrong_rate'], eliminates_wrong_signs=rate==0.)
        for rank in ('parameter_only', 'full_state', 'measured_aligned', 'measured_short')}
    check.data.append(dict(kind='rankers', r=256, surrogate='aligned', overall=table(rows),
        primary_rankers=['less_cosine', 'parameter_only', 'full_state', 'measured_aligned', RANKER],
        per_optimizer_configuration={str(i):dict(config=recipe, rankers=table([x for x in rows if x['recipe']==i]))
                                     for i, recipe in enumerate(p['manifest']['recipes'])},
        per_split={s:table([x for x in rows if x['split']==s]) for s in dict.fromkeys(x['split'] for x in rows)},
        v_share_bins=[dict(lo=lo, hi=min(hi, 1.), rankers=table([x for x in rows if x['v_share'] is not None
                         and lo<=x['v_share']<hi])) for lo, hi in zip(old.V_BINS[:-1], old.V_BINS[1:])],
        v_share_unavailable=table([x for x in rows if x['v_share'] is None]),
        high_v_share=dict(threshold=.5, rankers=high_table, comparisons=comparisons),
        conventions=original['conventions']+['Finite scores are point estimates; ranking correctness does not imply certification.',
            'C5 oracle scores and bins are reused only for reporting, never for C6b measurements or calibration.'],
        raw_replicates=dict(rows=rows)))
    check.assert_le('frozen_files_unchanged_after_rankers', p['before'] != old.hashes(), 0, 0)
    return [check]
