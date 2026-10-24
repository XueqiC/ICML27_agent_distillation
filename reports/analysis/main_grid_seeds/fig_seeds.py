"""Main-grid figure (user 10-24): converged column only, one line per method = mean over training seeds, teacher and base
student as reference lines. Input: seed_scores.json from seed_scores.py. Output: baseline_perf_seeds.png."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

D = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{D}/seed_scores.json"))
KS = [30, 100, 300, 500]; M = ["tsft", "AD", "SAD", "SmartAD"]
LBL = {"tsft": "Trajectory SFT", "AD": "AD (Kang)", "SAD": "SAD", "SmartAD": "SmartAD"}
COL = {"tsft": "#0072B2", "AD": "#E69F00", "SAD": "#009E73", "SmartAD": "#CC79A7"}  # Okabe-Ito; validator: pass with markers + labels
MK = {"tsft": "o", "AD": "s", "SAD": "^", "SmartAD": "D"}
TEACHER, BASE = 114, 62  # gpt-6-luna single run (reports/runs/teacher-dev140); base gemma-4-12B-it (seedsft-sweep-bi/k5s0)
INK, MUTED = "#222", "#666"
N = 140
pct = lambda x: 100 * x / N

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.6, 5.6))
ax.hlines(pct(TEACHER), 26, 545, color="#333", lw=1.6, zorder=1)
ax.text(27.5, pct(TEACHER) + 1.0, f"teacher {pct(TEACHER):.1f}%", color=INK, fontsize=11, va="bottom")
ax.hlines(pct(BASE), 26, 545, color="#999", lw=1.4, ls="--", zorder=1)
ax.text(27.5, pct(BASE) + 1.0, f"base student {pct(BASE):.1f}%", color=MUTED, fontsize=11, va="bottom")
ends, short = [], 0
for j, m in enumerate(M):
    xs, ys = [], []
    for k in KS:
        runs = S.get(f"conv|{m}|{k}", {})
        if not runs:
            continue
        v = [pct(len(x)) for x in runs.values()]
        x = k * (1 + 0.04 * (j - 1.5))
        if m != "SAD" and len(v) < 3:
            short += 1
        if len(v) > 1:
            ax.scatter([x] * len(v), v, s=22, facecolors="none", edgecolors=COL[m], linewidths=1.1, zorder=2)
        xs.append(x); ys.append(sum(v) / len(v))
    ax.plot(xs, ys, color=COL[m], lw=2, marker=MK[m], ms=8, label=LBL[m], zorder=3)
    ends.append([ys[-1], ys[-1], m, xs[-1]])
# direct labels at the right end, pushed apart to at least 3.2 points and off the teacher line
ends.sort()
for i in range(len(ends)):
    if i:
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 3.2)
    if abs(ends[i][1] - pct(TEACHER)) < 1.8:
        ends[i][1] = pct(TEACHER) + 2.2
for y, ylab, m, xe in ends:
    ax.annotate(f"{LBL[m]} {y:.1f}%", (xe * 1.07, y), xytext=(700, ylab),
                textcoords="data", va="center", fontsize=11, color=INK,
                ha="left", arrowprops=dict(arrowstyle="-", color=COL[m], lw=1, relpos=(0, 0.5), shrinkA=3, shrinkB=2))
ax.set_xscale("log"); ax.set_xlim(26, 1900); ax.set_xticks(KS); ax.set_xticklabels([f"K{k}" for k in KS]); ax.minorticks_off()
ax.set_ylim(28, 97); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0)); ax.grid(axis="y", color="#e8e8e8"); ax.set_axisbelow(True)
ax.set_xlabel("seed trajectories K"); ax.set_ylabel("dev140 success rate")
ax.set_title("Converged column: mean over training seeds", fontsize=14)
ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=11, handlelength=2.2)
note = ("Filled markers: mean over training seeds 0-2; open circles: individual seeds. SAD is seed-independent (one run). "
        "Teacher: gpt-6-luna, one run.")
if short:
    note += f"\nSeeds 1-2 still running: {short} of 12 LoRA points have fewer than 3 seeds so far."
fig.text(0.5, -0.03, note, ha="center", va="top", fontsize=10.5, color="#444", wrap=True)
fig.tight_layout(); fig.savefig(f"{D}/baseline_perf_seeds.png", dpi=150, bbox_inches="tight"); print("ok")
