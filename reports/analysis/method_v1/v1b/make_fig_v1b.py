"""V1b figure: (a) success by stratum for U / G1 / G with the V1a takeover rate T as a reference tick; (b) per-prefix G vs U."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, 'v1b_r2_analysis.json')))
C = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#000000']
MODE_C = {'U': '#999999', 'G1': C[1], 'G': C[0]}
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'width_ratios': [1.5, 1]})

order = [('seen-middle', 'target seen\nmiddle'), ('seen-floor', 'target seen\nfloor'),
         ('search-middle', 'search\nmiddle'), ('search-floor', 'search\nfloor'), ('CG', 'CG cases')]
allrow = {'n': 38, 'U': r['primary_G_vs_U']['b_mean'], 'G': r['primary_G_vs_U']['a_mean'],
          'G1': r['G_vs_G1']['b_mean'],
          'T': sum(int(x['T_v1a'].split('/')[0]) / int(x['T_v1a'].split('/')[1]) for x in r['per_prefix']) / 38}
rows = [('all', 'all 38', allrow)] + [(k, lab, r['strata'][k]) for k, lab in order]
w = 0.26
for i, (_, lab, row) in enumerate(rows):
    for j, m in enumerate(('U', 'G1', 'G')):
        ax[0].bar(i + (j - 1) * w, row[m], w * 0.92, color=MODE_C[m], label=m if i == 0 else None)
    ax[0].plot([i - 1.5 * w, i + 1.5 * w], [row['T']] * 2, color='black', linewidth=2, linestyle=':',
               label='T (teacher takeover, V1a)' if i == 0 else None)
ax[0].set_xticks(range(len(rows)))
ax[0].set_xticklabels([f'{lab}\n(n={row["n"]})' for _, lab, row in rows], fontsize=10)
ax[0].set_ylabel('task success (sampled, mean over prefixes)')
ax[0].set_ylim(0, 1.22)
ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax[0].legend(frameon=False, fontsize=10, ncol=4, loc='upper center')
ax[0].set_title('(a) success by prefix stratum', loc='left')

pp = r['per_prefix']
ax[1].plot([0, 1], [0, 1], color='#BBBBBB', linewidth=1)
from collections import Counter
cnt = Counter((round(x['U'], 3), round(x['G'], 3)) for x in pp)
for (u, g), n in cnt.items():
    col = C[0] if g > u else C[3] if g < u else '#777777'
    ax[1].scatter(u, g, s=60 * n, color=col, edgecolor='white', linewidth=1.5, zorder=3)
    if n > 1:
        ax[1].text(u + 0.035, g - 0.035, f'×{n}', fontsize=9, color='#333333')
p = r['primary_G_vs_U']
ax[1].text(0.62, 0.08, f'G > U: {p["up"]}\nG < U: {p["down"]}\ntie: {p["tie"]}', fontsize=11)
ax[1].set_xlabel('U: student alone')
ax[1].set_ylabel('G: student + continuous guidance')
ax[1].set_xlim(-0.04, 1.04)
ax[1].set_ylim(-0.04, 1.04)
ax[1].set_aspect('equal')
ax[1].set_title('(b) per prefix (8 samples each)', loc='left')

fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'v1b_r2.{ext}'), dpi=150)
