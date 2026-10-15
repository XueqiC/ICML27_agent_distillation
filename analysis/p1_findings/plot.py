"""Figures for docs/proposals/P1_empirical_findings*.md, from docs/proposals/figs/p1_findings_data.json."""
import json
import os
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIGS = os.path.join(REPO, 'docs', 'proposals', 'figs')
D = json.load(open(os.path.join(FIGS, 'p1_findings_data.json')))

# Palette: dataviz reference instance (validated: categorical slots 1-3 light; blue ordinal ramp; blue<->red diverging)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
UP, DOWN, NEUTRAL = '#2a78d6', '#e34948', '#8a8984'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
RAMP = {5: '#86b6ef', 10: '#5598e7', 20: '#2a78d6', 50: '#1c5cab', 100: '#104281'}

plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16, 'xtick.labelsize': 14,
                     'ytick.labelsize': 14, 'legend.fontsize': 13, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1.0,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2.0,
                     'savefig.facecolor': SURFACE, 'font.family': 'DejaVu Sans', 'axes.axisbelow': True})
BASE = D['base']['successes']
TEACHER = D['references']['teacher_dev140']
NOISE = D['references']['noise_reference_dev140_cross_config']


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGS, f'{name}.{ext}'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def kval(run):
    return int(re.search(r'k(\d+)', run).group(1))


# Finding A: one static filter (keep environment-verified successes), opposite consequences ------------------------
def fig_a():
    runs = sorted(D['agentevol'], key=lambda r: (kval(r), r))
    steps = ['bc', 'iter-1', 'iter-2', 'iter-3']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={'width_ratios': [1, 1.25]})
    deltas = []
    for r in runs:
        y = [D['agentevol'][r][s]['successes'] for s in steps]
        d = y[-1] - y[0]
        deltas.append((r.replace('agentgym-', ''), d))
        color = UP if d > NOISE else DOWN if d < -NOISE else NEUTRAL
        ax1.plot(range(4), y, color=color, marker='o', markersize=8, alpha=0.9)
        if abs(d) >= 30 or d >= 20:
            dy = {'agentgym-k5s2': 6, 'agentgym-k10s1': -6}.get(r, 0)
            ax1.annotate(r.replace('agentgym-', ''), (3, y[-1]), xytext=(8, dy), textcoords='offset points',
                         va='center', fontsize=13, color=INK2)
    ax1.axhline(BASE, color=INK2, linestyle='--', linewidth=1.5)
    ax1.text(0.02, BASE + 2, f'base student {BASE}', color=INK2, fontsize=13, transform=ax1.get_yaxis_transform())
    ax1.set_xticks(range(4), ['BC', 'iter 1', 'iter 2', 'iter 3'])
    ax1.set_xlim(-0.2, 3.7)
    ax1.set_ylabel('dev140 successes (of 140)')
    ax1.set_title('(a) AgentEvol, 13 support sets (K × seed)')
    deltas.sort(key=lambda t: t[1])
    names, vals = zip(*deltas)
    colors = [UP if v > NOISE else DOWN if v < -NOISE else NEUTRAL for v in vals]
    ax2.bar(range(len(vals)), vals, color=colors, width=0.7)
    ax2.axhspan(-NOISE, NOISE, color=NEUTRAL, alpha=0.18, linewidth=0)
    spread = D['references']['kang_k20s0_training_seed_successes']
    half = (max(spread) - min(spread)) / 2
    for s in (-half, half):
        ax2.axhline(s, color=INK2, linestyle=':', linewidth=1.5)
    ax2.text(-0.4, half + 1.5, f'±{half:.1f}: half the training-seed spread (Kang recipe, K20)', ha='left',
             fontsize=12, color=INK2)
    ax2.text(-0.4, NOISE - 5.0, f'±{NOISE}: cross-server noise', ha='left', fontsize=12, color=INK2)
    ax2.set_xticks(range(len(vals)), names, rotation=55, ha='right')
    ax2.set_ylabel('iter 3 − BC (tasks)')
    ax2.set_title('(b) change caused by the same filter')
    for i, v in enumerate(vals):
        ax2.text(i, v + (1.5 if v >= 0 else -1.5), f'{v:+d}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=12)
    ax2.set_ylim(min(vals) - 9, max(vals) + 9)
    fig.tight_layout()
    save(fig, 'p1_finding_A_agentevol')


# Finding B: the same correct demonstrations, harmful early and helpful later --------------------------------------
def fig_b():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    for k in (5, 10, 20, 50, 100):
        for s in range(3):
            run = f'k{k}s{s}'
            if run not in D['sweep']:
                continue
            cells = sorted(D['sweep'][run].values(), key=lambda c: c['tokens'])
            x = [c['tokens'] for c in cells]
            y = [c['successes'] for c in cells]
            ax.plot(x, y, color=RAMP[k], marker='o', markersize=5, alpha=0.95,
                    label=f'K = {k}' if s == 0 else None)
    ax.set_xscale('log')
    ax.axhspan(BASE - NOISE, BASE + NOISE, color=NEUTRAL, alpha=0.18, linewidth=0)
    ax.axhline(BASE, color=INK2, linestyle='--', linewidth=1.5)
    ax.axhline(TEACHER, color=ORANGE, linestyle='--', linewidth=1.5)
    ax.text(6.2e3, BASE - NOISE - 6, f'base student {BASE} (band ±{NOISE}: cross-server noise)', color=INK2, fontsize=13)
    ax.text(6.2e3, TEACHER + 2, f'teacher {TEACHER}', color=ORANGE, fontsize=13)
    ax.axvspan(3.6e4, 7.7e4, color=DOWN, alpha=0.07, linewidth=0)
    win = [min(c['successes'] for c in D['sweep'][f'k{k}s{s}'].values() if 3.6e4 <= c['tokens'] <= 7.7e4)
           for k in (5, 10, 20, 50) for s in range(3)]
    below, far = sum(v < BASE for v in win), sum(v < BASE - NOISE for v in win)
    ax.text(5.3e4, 21, f'36k–77k tokens: {below} of {len(win)} runs (K = 5…50)\nbelow base, {far} by more than {NOISE}\n'
            '(window drawn after seeing the data)', ha='center', fontsize=12, color=DOWN)
    ax.set_xlabel('supervised tokens seen (log scale): the learner state')
    ax.set_ylabel('dev140 successes (of 140)')
    ax.set_title('Seed-SFT on the same K trusted teacher demonstrations, 3 support seeds per K')
    ax.set_ylim(15, 125)
    ax.legend(loc='lower right', frameon=False, ncol=1)
    fig.tight_layout()
    save(fig, 'p1_finding_B_dose')


# Finding C: static size of the learning signal versus the measured consequence (R1, state z10) ----------------------
def fig_c():
    st = D['r1']['static']
    cands = ['P3', 'P1', 'P2']
    names = {'P1': 'P1 replay of\nK=20 demos', 'P2': 'P2 teacher corrections\non student states',
             'P3': 'P3 self-generated\nsuccesses'}
    cols = {'P1': BLUE, 'P2': ORANGE, 'P3': AQUA}
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.4), gridspec_kw={'width_ratios': [1, 1, 1, 1.5]})
    panels = [('student_loss', '(a) student loss\n(token NLL)', '{:.2f}'),
              ('pre_clip_grad_norm', '(b) gradient norm\nbefore clipping', '{:.1f}'),
              ('update_norm', '(c) parameter update norm\nafter clipping', '{:.4f}')]
    for ax, (key, title, fmt) in zip(axes[:3], panels):
        vals = [st[c][key] for c in cands]
        ax.bar(range(3), vals, color=[cols[c] for c in cands], width=0.65)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha='center', va='bottom', fontsize=13)
        ax.set_xticks(range(3), cands)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.22)
    ax = axes[3]
    pr = D['r1']['paired']
    rows = [('P3', pr['F_tilde_geo_prob:P3-P1']), ('P1', None), ('P2', pr['F_tilde_geo_prob:P2-P1'])]
    for i, (c, p) in enumerate(rows):
        if p is None:
            ax.scatter([i], [0], color=cols[c], s=90, zorder=3)
            continue
        jitter = [(j % 9 - 4) * 0.035 for j in range(len(p['Z']))]
        ax.scatter([i + t for t in jitter], p['Z'], color=cols[c], s=16, alpha=0.45, zorder=2)
        ax.errorbar([i], [p['mean']], yerr=[2 * p['se']], color=INK, capsize=7, marker='D', markersize=8, zorder=4,
                    linewidth=2)
        note = '\n(interim: half)' if c == 'P2' else ''
        ax.text(i + 0.18, p['mean'], f"{p['mean']:+.4f}\n{len(p['Z'])} replicas{note}", va='center', fontsize=12)
    radius = D['r1']['merge_p1p3_geo_prob']['radius_observed_range_DIAGNOSTIC']
    ax.axhspan(-radius, radius, color=NEUTRAL, alpha=0.15, linewidth=0)
    ax.text(2.85, radius + 0.0001, 'diagnostic radius\n(P3 vs P1, 64 replicas)', ha='right', va='bottom', fontsize=11,
            color=INK2)
    ax.axhline(0, color=INK2, linewidth=1.2)
    ax.set_xticks(range(3), cands)
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel('surrogate score minus P1\n(paired by tail seed)')
    ax.set_title('(d) measured consequence at z10\n(CRN replacement, 10-update tail)')
    handles = [plt.Line2D([], [], color=cols[c], marker='s', linestyle='', markersize=12, label=names[c].replace('\n', ' '))
               for c in ['P1', 'P2', 'P3']]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'p1_finding_C_r1')


# Finding D: replay cannot supply missing evidence ------------------------------------------------------------------
def fig_d():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={'width_ratios': [1.15, 1.15, 1]})
    missing = {'pick_cool_then_place_in_recep': ('k5s0', 'cool tasks (25)'),
               'look_at_obj_in_light': ('k5s2', 'look-at-in-light tasks (13)')}
    cols = [ORANGE, BLUE, AQUA]
    for ax, (typ, (lacking, title)) in zip(axes[:2], missing.items()):
        n_tasks = D['base']['per_type'][typ][1]
        order = [lacking] + [r for r in ('k5s0', 'k5s1', 'k5s2') if r != lacking]
        for col, run in zip(cols, order):
            cells = sorted(D['sweep'][run].values(), key=lambda c: c['tokens'])
            x = [c['tokens'] for c in cells]
            y = [c['per_type'][typ][0] for c in cells]
            demos = 0 if run == lacking else 1
            ax.plot(x, y, color=col, marker='o', markersize=6, linewidth=2.6 if demos == 0 else 2.0,
                    label=f'{run}: {demos} demo{"s" if demos == 0 else ""} of this type')
        ax.axhline(D['base']['per_type'][typ][0], color=INK2, linestyle='--', linewidth=1.5)
        ax.text(0.02, D['base']['per_type'][typ][0] + 0.3, 'base student', color=INK2, fontsize=12,
                transform=ax.get_yaxis_transform())
        ax.set_xscale('log')
        ax.set_ylim(-0.5, n_tasks + 0.5)
        ax.set_xlabel('supervised tokens seen (log)')
        ax.set_ylabel('successes on this type')
        ax.set_title(f'({"ab"[list(missing).index(typ)]}) K=5, {title}')
        ax.legend(loc='upper left', frameon=False, fontsize=12)
    ax = axes[2]
    ks = [1, 2, 5, 10, 20, 50, 100]
    for s, marker in zip(range(3), ['o', 's', '^']):
        xs, ys = [], []
        for k in ks:
            run = f'k{k}s{s}'
            if run in D['sweep']:
                xs.append(k)
                ys.append(max(c['successes'] for c in D['sweep'][run].values()))
        ax.plot(xs, ys, color=BLUE, marker=marker, linestyle='', alpha=0.8, markersize=9, label=f'support seed {s}')
    med = []
    for k in ks:
        v = sorted(max(c['successes'] for c in D['sweep'][f'k{k}s{s}'].values()) for s in range(3)
                   if f'k{k}s{s}' in D['sweep'])
        med.append(v[len(v) // 2])
    ax.plot(ks, med, color=INK, linewidth=2.2, label='median of 3 seeds')
    ax.axhline(TEACHER, color=ORANGE, linestyle='--', linewidth=1.5)
    ax.text(1.1, TEACHER + 2, f'teacher {TEACHER}', color=ORANGE, fontsize=12)
    ax.axhline(BASE, color=INK2, linestyle='--', linewidth=1.5)
    ax.text(1.1, BASE - 7, f'base {BASE}', color=INK2, fontsize=12)
    ax.set_xscale('log')
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel('K: trusted demonstrations')
    ax.set_ylabel('best dev140 over all doses')
    ax.set_title('(c) best over all doses, by K')
    ax.set_ylim(40, 125)
    ax.legend(loc='lower right', frameon=False, fontsize=12)
    fig.tight_layout()
    save(fig, 'p1_finding_D_information')


# Finding E: the learner is more than its parameters -----------------------------------------------------------------
def fig_e():
    probe = [r for r in D['probe'] if r['state'] != 'cold']
    names = {'warmup:8': '8 warm-up steps', 'warmup:32': '32 warm-up steps', 'update:1': 'update 1',
             'update:4': 'update 4', 'update:8': 'update 8', 'checkpoint:31221': 'p03 (31k tokens)',
             'checkpoint:104070': 'p10 (104k tokens)', 'checkpoint:416280': 'p40 (416k tokens)'}
    labels = [names[r['state']] for r in probe]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={'width_ratios': [1.3, 1.3, 1]})
    ax = axes[0]
    for i, r in enumerate(probe):
        v = r['adamw_clipped_momentum_share']
        ax.scatter([i] * len(v), v, color=BLUE, s=40, alpha=0.8)
        ax.plot([i - 0.25, i + 0.25], [sum(v) / len(v)] * 2, color=INK, linewidth=2.5)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
    ax.set_ylabel('‖update with an empty batch‖ /\n‖update with the real batch‖')
    ax.set_title('(a) AdamW (clipped): share of each update\nalready fixed by optimizer memory')
    ax = axes[1]
    for key, col, name in [('adamw_clipped_nu_centered', BLUE, 'AdamW, clipped (the recipe)'),
                           ('sgd_unclipped_nu_centered', ORANGE, 'SGD, unclipped')]:
        med = [statistics.median(r[key]) for r in probe]
        ax.plot(range(len(probe)), med, color=col, marker='o', markersize=8, label=name)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
    ax.set_ylim(0, 0.75)
    ax.set_ylabel('non-additivity ν of one update\n(0 = effects of two batches add up)')
    ax.set_title('(b) do the effects of two batches add up?')
    ax.legend(loc='upper right', frameon=False)
    ax = axes[2]
    pw = D['pilot_warm']
    for j, ck in enumerate(['p10', 'p40']):
        c, w = pw[ck]['cold'], pw[ck]['mw32']
        ax.bar([j - 0.18], [c], width=0.34, color=NEUTRAL, label='cold optimizer state' if j == 0 else None)
        ax.bar([j + 0.18], [w], width=0.34, color=BLUE, label='moments warmed, parameters unchanged' if j == 0 else None)
        a, b = pw[ck]['mw32_vs_cold_discordant']
        ax.text(j, max(c, w) + 4, f'{w - c:+d} tasks\n{a}:{b} discordant\np = {pw[ck]["mw32_vs_cold_p"]}', ha='center',
                fontsize=12)
    ax.set_xticks([0, 1], ['p10 checkpoint', 'p40 checkpoint'])
    ax.set_ylim(0, 160)
    ax.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
    ax.set_ylabel('dev140 successes (same server)')
    ax.set_title('(c) same data and parameters,\ndifferent optimizer state')
    ax.legend(loc='lower center', frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    save(fig, 'p1_finding_E_optimizer_state')


if __name__ == '__main__':
    fig_a()
    fig_b()
    fig_c()
    fig_d()
    fig_e()
    print('figures written to', FIGS)
