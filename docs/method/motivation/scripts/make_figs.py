"""Figures for the motivation result pages; reads only docs/method/motivation/data/*.json."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = 'docs/method/motivation/data/'
F = 'docs/method/motivation/figs/'
m1, m2, m3 = (json.load(open(D + f)) for f in ('m1.json', 'm2.json', 'm3.json'))
# Okabe-Ito (colour-blind safe), fixed order
C = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#000000']
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 15, 'axes.labelsize': 14, 'legend.fontsize': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
Ks = ['10', '20', '30', '100']


def save(fig, name):
    fig.tight_layout(); fig.savefig(F + name + '.png', bbox_inches='tight'); fig.savefig(F + name + '.pdf', bbox_inches='tight'); plt.close(fig)


# ---- M1: training tasks vs new tasks ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
k, u = m1['k20'], m1['umech']
labs = ['greedy', 'T=0.7\nseed 0', 'T=0.7\nseed 1', 'T=0.7\nseed 2']
kv = [v / 20 for v in k['by_decoding'].values()]
uv = [v / 20 for v in u['by_decoding'].values()]
xs_k, xs_u = np.arange(4), np.arange(5, 7)
ax[0].bar(xs_k, kv, 0.7, color=C[0]); ax[0].bar(xs_u, uv, 0.7, color=C[1])
for x_, v in list(zip(xs_k, kv)) + list(zip(xs_u, uv)):
    ax[0].text(x_, v + 0.02, f'{v:.0%}', ha='center')
ax[0].set_xticks(list(xs_k) + list(xs_u), labs + labs[:2]); ax[0].set_ylim(0, 1.15); ax[0].set_ylabel('success rate')
ax[0].text(1.5, 1.08, 'own 20 training tasks', ha='center', color=C[0]); ax[0].text(5.5, 1.08, '20 new tasks', ha='center', color=C[1])
ax[0].set_title('theta0 (trajectory SFT, K=20)', pad=18)
bars = [('training tasks:\n= teacher demo', k['equal_to_teacher_demo'] / 20, C[0]),
        ('training tasks:\nsame in 4 decodings', k['identical_across_decodings'] / 20, C[0]),
        ('new tasks:\ngreedy = sampled', u['identical_greedy_sampled'] / 20, C[1])]
for i, (l, v, c) in enumerate(bars):
    ax[1].bar(i, v, 0.6, color=c); ax[1].text(i, v + 0.02, f'{round(v * 20)}/20', ha='center')
ax[1].set_xticks(range(3), [b[0] for b in bars], fontsize=12); ax[1].set_ylim(0, 1.15); ax[1].set_ylabel('fraction of tasks')
ax[1].set_title('Identical action sequences', pad=18)
save(fig, 'm1_train_vs_new')

# ---- M2a: failure modes by K ----
order = [('success', 'success', '#BBBBBB'), ('A_never_seen', 'A never saw target', C[0]),
         ('B_seen_not_taken', 'B saw, not taken', C[5]), ('C_took_no_transform', 'C took, no heat/cool/clean', C[1]),
         ('C_took_no_lamp', 'C took, lamp not used', C[6]), ('D_placed_without_transform', 'D placed without transform', C[3]),
         ('E_transformed_not_placed', 'E transformed, not placed', C[4]), ('E_took_not_placed', 'E took, not placed', '#882255'),
         ('F_two_incomplete', 'F two-object incomplete', C[2])]
fig, ax = plt.subplots(figsize=(10, 5.2))
bottom = np.zeros(4)
for key, lab, col in order:
    vals = np.array([m2['mode_counts'][K].get(key, 0) / sum(m2['mode_counts'][K].values()) for K in Ks])
    ax.bar(range(4), vals, 0.6, bottom=bottom, color=col, label=lab, edgecolor='white', linewidth=1)
    for i, v in enumerate(vals):
        if v >= 0.04 and key != 'success':
            ax.text(i, bottom[i] + v / 2, f'{v:.0%}', ha='center', va='center', fontsize=11, color='white' if col in (C[0], C[3], '#882255') else 'black')
    bottom += vals
ax.set_xticks(range(4), [f'K{K}\n({m2["cells_per_K"][K]} cells)' for K in Ks]); ax.set_ylabel('share of all episodes')
ax.set_ylim(0, 1); ax.set_title('Outcome of every student episode on the 114 teacher-solved dev140 tasks')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
save(fig, 'm2_failure_modes_by_K')

# ---- M2b: phrasing ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
pairs = [('pick_heat_then_place_in_recep', ['heat some X', 'hot X'], 'heat'),
         ('pick_cool_then_place_in_recep', ['cool some X', 'put a cool X'], 'cool'),
         ('pick_two_obj_and_place', ['put two', 'find two'], 'two objects')]
for a, (typ, ps, name) in zip(ax, pairs):
    for j, p in enumerate(ps):
        row = next(r for r in m2['phrasing'] if r['type'] == typ and r['phrasing'] == p)
        ys = [row['solve_rate'][K] for K in Ks]
        a.plot(range(4), ys, marker='o', markersize=8, linewidth=2, color=C[j], label=f'"{p}" ({row["tasks"]} tasks)')
    a.set_xticks(range(4), [f'K{K}' for K in Ks]); a.set_title(name); a.set_ylim(0, 1); a.legend(frameon=False, loc='lower right')
ax[0].set_ylabel('student solve rate (mean over cells)')
save(fig, 'm2_phrasing')

# ---- M2c: feedback + search ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ref = [m2['refused_target_action_in_failures'][K][0] / m2['refused_target_action_in_failures'][K][1] for K in Ks]
rep = [m2['prevalence'][K]['failed_eps_with_repeated_refused_action'][0] / m2['prevalence'][K]['failed_eps_with_repeated_refused_action'][1] for K in Ks]
x = np.arange(4)
ax[0].bar(x - 0.2, ref, 0.38, color=C[3], label='a refused action on the target object')
ax[0].bar(x + 0.2, rep, 0.38, color=C[4], label='the same refused action issued again')
for i in range(4):
    ax[0].text(x[i] - 0.2, ref[i] + 0.01, f'{ref[i]:.0%}', ha='center', fontsize=11)
    ax[0].text(x[i] + 0.2, rep[i] + 0.01, f'{rep[i]:.0%}', ha='center', fontsize=11)
ax[0].set_xticks(x, [f'K{K}' for K in Ks]); ax[0].set_ylim(0, 0.5); ax[0].set_ylabel('share of failed episodes')
ax[0].set_title('Failed episodes containing ...'); ax[0].legend(frameon=False)
sr = m2['search_runs']
s_ = [sr[K]['student_same_type_next_visit_first10'][0] / sr[K]['student_same_type_next_visit_first10'][1] for K in Ks]
t_ = [sr[K]['teacher_same_type_next_visit'][0] / sr[K]['teacher_same_type_next_visit'][1] for K in Ks]
ax[1].bar(x - 0.2, s_, 0.38, color=C[0], label='student, never saw target (first 10 visits)')
ax[1].bar(x + 0.2, t_, 0.38, color=C[2], label='teacher, same tasks (until target seen)')
for i in range(4):
    ax[1].text(x[i] - 0.2, s_[i] + 0.01, f'{s_[i]:.2f}', ha='center', fontsize=11); ax[1].text(x[i] + 0.2, t_[i] + 0.01, f'{t_[i]:.2f}', ha='center', fontsize=11)
ax[1].set_xticks(x, [f'K{K}\n(n={sr[K]["episodes"]})' for K in Ks]); ax[1].set_ylabel('P(next visit is the\nsame receptacle type)')
ax[1].set_title('Search: staying on one receptacle type'); ax[1].legend(frameon=False, loc='upper center', fontsize=11); ax[1].set_ylim(0, 1.3)
save(fig, 'm2_feedback_search')

# ---- M3a: branches ----
fig, ax = plt.subplots(figsize=(10, 4.6))
br = ['A', 'B', 'C', 'D']; names = ['A no hint', 'B generic\nreminder', 'C condition\nhint', 'D condition hint\n+ one action']
x = np.arange(4)
for j, (key, lab) in enumerate([('found', 'target found'), ('picked', 'target picked up'), ('success', 'task completed')]):
    v = [m3['branch_totals'][b][key] for b in br]
    ax.bar(x + (j - 1) * 0.26, v, 0.25, color=C[j], label=lab)
    for i in range(4):
        ax.text(x[i] + (j - 1) * 0.26, v[i] + 0.4, str(v[i]), ha='center', fontsize=11)
ax.set_xticks(x, names); ax.set_ylabel('count out of 56 sampled rollouts'); ax.set_ylim(0, 32)
ax.set_title('CG probe: 7 cases x 8 sampled continuations per branch'); ax.legend(frameon=False, loc='upper left')
save(fig, 'm3_branches')

# ---- M3b: per-case heatmap ----
cases = ['look_at_obj_in_light-Newspaper-None-DeskLamp-224', 'pick_cool_then_place_in_recep-Apple-None-Microwave-19',
         'pick_and_place_simple-Kettle-None-Shelf-1', 'pick_clean_then_place_in_recep-Kettle-None-DiningTable-7',
         'pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18', 'pick_cool_then_place_in_recep-Egg-None-Microwave-5',
         'pick_cool_then_place_in_recep-Potato-None-SideTable-21']
short = ['Newspaper', 'Apple', 'Kettle-Shelf', 'Kettle-DiningTable', 'Lettuce', 'Egg', 'Potato']
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for a, key, title in zip(ax, ['found', 'success'], ['target found (of 8)', 'task completed (of 8)']):
    M = np.array([[m3['per_case'][c][b][key] for b in br] for c in cases])
    im = a.imshow(M, cmap='Blues', vmin=0, vmax=8, aspect='auto')
    for i in range(len(cases)):
        for j in range(4):
            a.text(j, i, str(M[i, j]), ha='center', va='center', color='white' if M[i, j] >= 5 else 'black')
    a.set_xticks(range(4), br); a.set_yticks(range(len(cases)), short if a is ax[0] else [''] * len(cases)); a.set_title(title)
    a.spines[:].set_visible(False)
save(fig, 'm3_cases')
print('figures written')
