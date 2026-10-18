"""Figures for docs/analysis/baseline_failure_analysis*.md from docs/analysis/figs/baseline_failures_data.json.

Run: cd analysis/baseline_failures && python plot.py      (PNG + PDF into docs/analysis/figs/)
Colours: categorical slots 1-2 (blue = plain SFT reference, orange = the baseline), grey references, one-hue blue ramp for
magnitudes and for the ordinal subgoal stages; every series is also labelled directly.
"""
import json
import os
import statistics as st
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, '..', '..', 'docs', 'analysis', 'figs')
D = json.load(open(os.path.join(FIGS, 'baseline_failures_data.json')))
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GREY, BAND, INK, INK2 = '#8a8985', '#e9e8e4', '#1f1f1e', '#52514e'
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
        '#184f95', '#104281', '#0d366b']
SHORT = {'pick_and_place_simple': 'pick', 'look_at_obj_in_light': 'look', 'pick_clean_then_place_in_recep': 'clean',
         'pick_heat_then_place_in_recep': 'heat', 'pick_cool_then_place_in_recep': 'cool', 'pick_two_obj_and_place': 'two'}
TYPES = D['types']
NOISE = D['log_only']['noise_reference_dev140']
BASE, TEACHER = D['base']['successes'], D['teacher']['successes']
KS = [1, 2, 5, 10, 20, 50, 100]
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 16, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                     'ytick.labelsize': 13, 'legend.fontsize': 13, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'font.family': 'DejaVu Sans'})
CELLS = D['cells']


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + '.png'), dpi=160, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(FIGS, name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def seedsft_end(K=None):
    """Last checkpoint of each pre-specified Seed-SFT schedule (largest token count); not selected on dev."""
    best = {}
    for c in CELLS:
        if c['method'] == 'Seed-SFT' and (K is None or c['K'] == K):
            key = (c['K'], c['seed'])
            if key not in best or c['tokens'] > best[key]['tokens']:
                best[key] = c
    return list(best.values())


def select(method, K=None, **kw):
    out = []
    for c in CELLS:
        if c['method'] != method or (K is not None and c['K'] != K):
            continue
        if all(c.get(k, 0 if k == 'training_seed' else None) == v for k, v in kw.items()):
            out.append(c)
    return out


def leap_last():
    """Last completed iterate per K (K1 and K5 stopped early on empty correction sets); K20/50/100 from exp_log."""
    vals = {}
    for c in select('LEAP privileged'):
        if c['K'] not in (20,) and (c['K'] not in vals or c['iteration'] > vals[c['K']][1]):
            vals[c['K']] = (c['successes'], c['iteration'])
    for K in (20, 50, 100):
        vals[K] = (D['log_only']['leap_dev140'][f'LEAP privileged K{K}'][-1], 3)
    return {K: [v[0]] for K, v in vals.items()}


def refs(ax, xmax_label=None):
    ax.axhspan(BASE - NOISE, BASE + NOISE, color=BAND, zorder=0)
    ax.axhline(BASE, color=GREY, lw=1.6, ls='--', zorder=1)
    ax.axhline(TEACHER, color=GREY, lw=1.6, ls=':', zorder=1)


def fig1_ladder():
    sft = defaultdict(list)
    for c in seedsft_end():
        sft[c['K']].append(c['successes'])
    panels = [('Kang SFT + SAG (2 epochs)', {K: [c['successes'] for c in select('Kang', K, decoding='sag', training_seed=0)]
                                            for K in KS}),
              ('AgentGym BC (1 epoch)', {K: [c['successes'] for c in select('AgentGym', K, iteration=0)] for K in KS}),
              ('AgentEvol, iteration 3', {K: [c['successes'] for c in select('AgentGym', K, iteration=3)] for K in KS}),
              ('LEAP privileged, last iterate', leap_last())]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), sharex=True, sharey=True)
    for ax, (title, vals) in zip(axes.flat, panels):
        refs(ax)
        for series, color, label, dx in [(sft, BLUE, 'Seed-SFT, end of schedule', 0.97), (vals, ORANGE, 'baseline', 1.03)]:
            ks = [K for K in KS if series.get(K)]
            med = [st.median(series[K]) for K in ks]
            ax.plot(ks, med, color=color, lw=2.4, marker='o', ms=9, zorder=3)
            for K in ks:
                ax.scatter([K * dx] * len(series[K]), series[K], s=36, color=color, alpha=0.45, zorder=2, edgecolor='none')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_ylim(0, 140)
        ax.grid(axis='y', color=BAND, lw=1)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        ax.text(1.0, TEACHER + 2, f'teacher {TEACHER}', color=INK2, fontsize=12, va='bottom')
        ax.text(30, BASE - NOISE - 2, f'base {BASE} ± {NOISE} (grey band)', color=INK2, fontsize=12, va='top')
        last = [K for K in KS if vals.get(K)][-1]
        ax.text(last * 1.12, st.median(vals[last]), 'baseline', color=ORANGE, fontsize=13, va='center', fontweight='bold')
        ax.text(100 * 1.12, st.median(sft[100]) + 6, 'Seed-SFT', color=BLUE, fontsize=13, va='center', fontweight='bold')
    for ax in axes[1]:
        ax.set_xlabel('K (trusted demonstrations)')
    for ax in axes[:, 0]:
        ax.set_ylabel('dev140 successes')
    fig.suptitle('Each baseline (orange) against plain SFT trained to the end of its schedule (blue), same support sets',
                 x=0.01, ha='left', fontsize=16)
    fig.tight_layout(rect=(0, 0, 0.97, 0.96))
    save(fig, 'bl_fig1_ladder')


def fig0_overall():
    """All methods in one panel: median over support sets (line) and single cells (dots), with direct labels."""
    sft = defaultdict(list)
    for c in seedsft_end():
        sft[c['K']].append(c['successes'])
    series = [('Seed-SFT (end of schedule)', sft, BLUE, 'o'),
              ('Kang SFT + SAG', {K: [c['successes'] for c in select('Kang', K, decoding='sag', training_seed=0)] for K in KS},
               ORANGE, 's'),
              ('AgentGym BC', {K: [c['successes'] for c in select('AgentGym', K, iteration=0)] for K in KS}, AQUA, '^'),
              ('AgentEvol (iteration 3)', {K: [c['successes'] for c in select('AgentGym', K, iteration=3)] for K in KS},
               '#eda100', 'D'),
              ('LEAP privileged (last iterate)', leap_last(), '#e87ba4', 'v')]
    fig, ax = plt.subplots(figsize=(15, 8.6))
    refs(ax)
    offsets = [0.94, 0.97, 1.0, 1.03, 1.06]
    ends = []
    for (name, vals, color, marker), dx in zip(series, offsets):
        ks = [K for K in KS if vals.get(K)]
        med = [st.median(vals[K]) for K in ks]
        ax.plot([k * dx for k in ks], med, color=color, lw=2.6, marker=marker, ms=10, zorder=3,
                markeredgecolor='white', markeredgewidth=1.2, label=name)
        for K in ks:
            if len(vals[K]) > 1:
                ax.scatter([K * dx] * len(vals[K]), vals[K], s=28, color=color, alpha=0.35, zorder=2, edgecolor='none')
        ends.append([med[-1], name, color])
    lz = D['log_only']['leap_dev140']['LEAP 0% privileged K20'][-1]
    ax.scatter([20 * 1.09], [lz], s=90, marker='v', facecolor='white', edgecolor='#e87ba4', linewidth=2, zorder=4)
    ax.annotate('LEAP 0% privileged', (20 * 1.09, lz), (24, lz - 14), fontsize=12, color=INK2,
                arrowprops=dict(arrowstyle='-', color=GREY, lw=1))
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):  # keep the right-hand labels at least 5 tasks apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 5.5)
    for y, name, color in ends:
        ax.text(100 * 1.17, y, name, color=color if color in (BLUE, ORANGE) else INK, fontsize=13, va='center',
                fontweight='bold')
        if color not in (BLUE, ORANGE):  # low-contrast hues: keep the label in ink, mark the colour beside it
            ax.scatter([100 * 1.13], [y], s=70, color=color, marker='s', zorder=4)
    ax.text(1.0, TEACHER + 2, f'teacher {TEACHER}', color=INK2, fontsize=12, va='bottom')
    ax.text(100 * 1.13, BASE, f'untrained base {BASE}\n(grey band: ± {NOISE} noise)', color=INK2, fontsize=12,
            va='center')
    ax.set_xscale('log')
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(0.85, 300)
    ax.set_ylim(0, 140)
    ax.set_xlabel('K (trusted teacher demonstrations)')
    ax.set_ylabel('dev140 successes (of 140)')
    ax.grid(axis='y', color=BAND, lw=1)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=12)
    ax.set_title('Overall dev140 results of every evaluated method (lines: median over support sets; dots: single cells)',
                 loc='left', fontsize=15)
    save(fig, 'bl_fig0_overall')


def pooled(cells, field):
    tot = Counter()
    for c in cells:
        for k, v in c[field].items():
            if isinstance(v, list):
                tot[k + '_s'] += v[0]
                tot[k + '_n'] += v[1]
            else:
                tot[k] += v
    return tot


def rows_for(K):
    rows = [('Teacher', [D['teacher']]), ('Base (untrained)', [D['base']]),
            ('Seed-SFT, end of schedule', seedsft_end(K)),
            ('Seed-SFT, 5 passes', [c for c in select('Seed-SFT', K) if c['dose'] == 'p05']),
            ('Kang SFT + SAG', select('Kang', K, decoding='sag', training_seed=0)),
            ('AgentGym BC', select('AgentGym', K, iteration=0)),
            ('AgentEvol, iteration 3', select('AgentGym', K, iteration=3))]
    if K == 20:
        rows += [('LEAP privileged, iteration 3', select('LEAP privileged', 20, iteration=3)),
                 ('LEAP 0% privileged, iteration 3', select('LEAP 0% privileged', 20, iteration=3))]
    return [(name, [c for c in cs if 'per_type' in c]) for name, cs in rows]


def fig2_types():
    cmap = LinearSegmentedColormap.from_list('blue', RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), gridspec_kw={'width_ratios': [1, 1]})
    for ax, K in zip(axes, [20, 100]):
        rows = rows_for(K)
        mat = []
        for name, cs in rows:
            t = pooled(cs, 'per_type')
            mat.append([100 * t[ty + '_s'] / t[ty + '_n'] for ty in TYPES])
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect='auto')
        for i, row in enumerate(mat):
            for j, v in enumerate(row):
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=13, color='white' if v >= 55 else INK)
        ax.set_xticks(range(len(TYPES)))
        ax.set_xticklabels([SHORT[t] for t in TYPES])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f'{n}  (n={len(cs)})' if len(cs) > 1 else n for n, cs in rows])
        ax.set_title(f'K = {K}', loc='left', fontweight='bold')
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle('Success rate (%) per task type, pooled over support sets (n = cells); dev140 has 35/13/27/16/25/24 tasks',
                 x=0.01, ha='left', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, 'bl_fig2_types')


STAGES = [('overflow', 'context overflow', INK2), ('search', 'never took the target object', RAMP[3]), ('transform', 'took it, never cleaned/heated/cooled or lit', RAMP[5]),
          ('place', 'ready, never put it in place (pick-two: one of two)', RAMP[8]),
          ('all_subgoals', 'all subgoal actions, still failed', RAMP[11])]


def fig3_stages():
    rows = rows_for(20)
    fig, ax = plt.subplots(figsize=(16, 7.4))
    for i, (name, cs) in enumerate(rows):
        n = 140 * len(cs)
        fs = pooled(cs, 'failure_stage')
        fs['place'] = fs.get('place', 0) + fs.get('second', 0)
        left = 0
        segs = [(k, fs.get(k, 0) / n * 100, color) for k, _, color in STAGES]
        segs.append(('success', sum(c['successes'] for c in cs) / n * 100, AQUA))
        for k, w, color in segs:
            if w <= 0:
                continue
            ax.barh(i, w, left=left, color=color, edgecolor='white', linewidth=2, height=0.72)
            if w >= 6:
                ax.text(left + w / 2, i, f'{w:.0f}', ha='center', va='center', fontsize=12,
                        color='white' if color in (RAMP[8], RAMP[11], AQUA, INK2) else INK)
            left += w
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('share of all dev140 episodes (%), K = 20, pooled over support sets')
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in STAGES] + [plt.Rectangle((0, 0), 1, 1, color=AQUA)]
    labels = [f'failed: {d}' for _, d, _ in STAGES] + ['success']
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.45, -0.13), ncol=2, frameon=False)
    ax.set_title('How far each episode got: furthest subgoal reached by executed actions', loc='left', fontweight='bold')
    save(fig, 'bl_fig3_stages')


def fig4_instability():
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={'width_ratios': [1.05, 1]})
    ax = axes[0]
    ag = defaultdict(dict)
    for c in select('AgentGym'):
        ag[(c['K'], c['seed'])][c['iteration']] = c['successes']
    items = [(f'AgentEvol K{K} s{s}', v[3] - v[0], ORANGE) for (K, s), v in sorted(ag.items()) if 0 in v and 3 in v]
    lp = D['log_only']['leap_dev140']
    leap = defaultdict(dict)
    for c in select('LEAP privileged'):
        leap[c['K']][c['iteration']] = c['successes']
    for K in (2, 10):
        items.append((f'LEAP priv. K{K}', leap[K][3] - leap[K][0], BLUE))
    for name in ['LEAP privileged K20', 'LEAP 0% privileged K20', 'LEAP privileged K50', 'LEAP privileged K100']:
        v = lp[name]
        items.append((name.replace('privileged', 'priv.'), v[-1] - v[0], BLUE))
    ax.axvspan(-NOISE, NOISE, color=BAND, zorder=0)
    ax.axvline(0, color=GREY, lw=1.2)
    for i, (name, dv, color) in enumerate(items):
        ax.plot([0, dv], [i, i], color=color, lw=2, alpha=0.6)
        ax.scatter([dv], [i], color=color, s=80, zorder=3)
        ax.text(dv + (2 if dv >= 0 else -2), i, f'{dv:+d}', va='center', ha='left' if dv >= 0 else 'right', fontsize=12)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _, _ in items], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(-62, 42)
    ax.set_xlabel('last iterate minus iteration 0 (dev140 tasks)', labelpad=26)
    ax.set_title('(a) What the iterations did', loc='left', fontweight='bold')
    ax.text(0, len(items) + 0.2, f'grey band: ±{NOISE} (same adapter, two servers)', ha='center', va='top', fontsize=12,
            color=INK2)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)

    ax = axes[1]
    k1 = [ag[(1, 0)][i] for i in range(4)]
    k5s2 = [c['successes'] for c in select('AgentGym', 5) if c['seed'] == 2 and c['iteration'] in (1, 2, 3)]
    spread = [('same adapter, two servers', [BASE - NOISE / 2, BASE + NOISE / 2], 'reference'),
              ('Seed-SFT end, K20: 3 support sets', [c['successes'] for c in seedsft_end(20)], ''),
              ('Kang K20 s0: 3 training seeds', D['log_only']['kang_training_seeds_k20s0'], ''),
              ('AgentGym BC K20: 3 support sets', [c['successes'] for c in select('AgentGym', 20, iteration=0)], ''),
              ('AgentEvol K1 s0: 4 retrains, same data', k1, ''),
              ('AgentEvol K5 s2 it1-3: same data', k5s2, '')]
    for i, (name, vals, tag) in enumerate(spread):
        ax.plot([min(vals), max(vals)], [i, i], color=GREY, lw=3, solid_capstyle='round', alpha=0.7)
        if tag:
            ax.text(max(vals) + 3, i, f'range {NOISE}', va='center', fontsize=12, color=INK2)
        else:
            ax.scatter(vals, [i] * len(vals), color=BLUE, s=70, zorder=3)
            ax.text(max(vals) + 3, i, f'range {max(vals) - min(vals)}', va='center', fontsize=12, color=INK2)
    ax.set_yticks(range(len(spread)))
    ax.set_yticklabels([n for n, _, _ in spread], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 140)
    ax.set_xlabel('dev140 successes')
    ax.set_title('(b) Spread from things that are not the method', loc='left', fontweight='bold')
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    save(fig, 'bl_fig4_instability')


def fig5_behaviour():
    ag = defaultdict(dict)
    for c in select('AgentGym'):
        ag[(c['K'], c['seed'])][c['iteration']] = c
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    rec = D['iterations']
    for (K, s), its in ag.items():  # share of the K seed TRAIN tasks the iterate re-solved (its own successes, kept as F_m)
        for i in its:
            r = rec.get(f'agentgym-k{K}s{s}', {}).get(str(i), {})
            if r.get('F_m') is not None:
                its[i]['F_share'] = 100 * r['F_m'] / K
    specs = [('successes', 'dev140 successes'), ('inadmissible_per_episode', 'inadmissible actions per episode'),
             ('F_share', '% of its K training tasks re-solved (F_m / K)')]
    for ax, (field, ylabel) in zip(axes, specs):
        for (K, s), its in sorted(ag.items()):
            xs = [i for i in range(4) if i in its and its[i].get(field) is not None]
            ys = [its[i][field] for i in xs]
            color = ORANGE if K <= 10 else BLUE
            ax.plot(xs, ys, color=color, lw=1.8, marker='o', ms=6, alpha=0.75)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['BC', '1', '2', '3'])
        ax.set_xlabel('AgentEvol iteration')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color=BAND)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    axes[0].axhline(BASE, color=GREY, ls='--', lw=1.4)
    axes[0].text(3.05, BASE, 'base', color=INK2, fontsize=12, va='center')
    axes[1].axhline(D['base']['inadmissible_per_episode'], color=GREY, ls='--', lw=1.4)
    axes[1].text(3.05, D['base']['inadmissible_per_episode'], 'base', color=INK2, fontsize=12, va='center')
    fig.suptitle('AgentEvol, one line per support set (13 sets):', x=0.01, ha='left', fontsize=16)
    fig.text(0.335, 0.965, 'K ≤ 10', color=ORANGE, fontweight='bold', fontsize=16, va='center')
    fig.text(0.39, 0.965, 'K ≥ 20', color=BLUE, fontweight='bold', fontsize=16, va='center')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, 'bl_fig5_agentevol')


if __name__ == '__main__':
    fig0_overall()
    fig1_ladder()
    fig2_types()
    fig3_stages()
    fig4_instability()
    fig5_behaviour()
    print('figures written to', os.path.abspath(FIGS))
