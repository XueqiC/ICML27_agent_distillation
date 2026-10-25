"""Student-base comparison (user 10-24): three panels side by side, gemma-4-12B-it | E4B-it | E2B-it, converged column,
one line per method (12B: mean over training seeds; E4B/E2B: seed 0), teacher and each panel's base student as reference
lines, one shared legend on top. Inputs: seed_scores.json (12B) and variant_scores.json (E4B/E2B)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

D = os.path.dirname(os.path.abspath(__file__))
S12 = json.load(open(f"{D}/seed_scores.json"))
SV = json.load(open(f"{D}/variant_scores.json")) if os.path.exists(f"{D}/variant_scores.json") else {}
KS = [30, 100, 300, 500]; M = ["tsft", "AD", "SAD", "SmartAD"]
LBL = {"tsft": "Trajectory SFT", "AD": "AD", "SAD": "SAD", "SmartAD": "SmartAD"}
COL = {"tsft": "#0072B2", "AD": "#E69F00", "SAD": "#009E73", "SmartAD": "#CC79A7"}  # Okabe-Ito (validated, + markers)
MK = {"tsft": "o", "AD": "s", "SAD": "^", "SmartAD": "D"}
N, TEACHER = 140, 114
pct = lambda x: 100 * x / N
BASE12 = 62
panels = [("gemma-4-12B-it", "12b"), ("gemma-4-E4B-it", "e4b"), ("gemma-4-E2B-it", "e2b")]

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
handles = {}
for ax, (title, tag) in zip(axes, panels):
    ax.axhline(pct(TEACHER), color="#333", lw=1.6, zorder=1)
    ax.text(27.5, pct(TEACHER) + 1.0, f"teacher {pct(TEACHER):.1f}%", color="#222", fontsize=11, va="bottom")
    base = BASE12 if tag == "12b" else (len(SV[f"{tag}|base"]) if f"{tag}|base" in SV else None)
    if base is not None:
        ax.axhline(pct(base), color="#999", lw=1.4, ls="--", zorder=1)
        ax.text(27.5, pct(base) + 1.0, f"base student {pct(base):.1f}%", color="#666", fontsize=11, va="bottom")
    for j, m in enumerate(M):
        xs, ys, singles = [], [], []
        for k in KS:
            if tag == "12b":
                runs = S12.get(f"conv|{m}|{k}", {}); v = [pct(len(x)) for x in runs.values()]
            else:
                v = [pct(len(SV[f"{tag}|{m}|{k}"]))] if f"{tag}|{m}|{k}" in SV else []
            if not v:
                continue
            x = k * (1 + 0.04 * (j - 1.5))
            if len(v) > 1:
                singles += [(x, y) for y in v]
            xs.append(x); ys.append(sum(v) / len(v))
        if singles:
            ax.scatter(*zip(*singles), s=22, facecolors="none", edgecolors=COL[m], linewidths=1.1, zorder=2)
        if xs:
            h, = ax.plot(xs, ys, color=COL[m], lw=2, marker=MK[m], ms=8, label=LBL[m], zorder=3)
            handles[m] = h
    ax.set_xscale("log"); ax.set_xlim(26, 600); ax.set_xticks(KS); ax.set_xticklabels([f"K{k}" for k in KS]); ax.minorticks_off()
    ax.set_ylim(0, 97)   # 0: the E2B base student (6.4%) must stay visible; ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", color="#e8e8e8"); ax.set_axisbelow(True); ax.set_title(title, fontsize=14)
    ax.set_xlabel("seed trajectories K")
axes[0].set_ylabel("dev140 success rate")
fig.legend([handles[m] for m in M if m in handles], [LBL[m] for m in M if m in handles], loc="upper center", ncol=4,
           frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=12)
fig.text(0.5, -0.03, "Converged column (each cell's 12B ladder rung and lr reused for E4B/E2B). 12B: mean over training seeds "
         "(open circles: single seeds; SAD seed-independent); E4B/E2B: seed 0, K30 and K100"
         + ("" if all(f"{t}|{m}|{k}" in SV for t in ("e4b", "e2b") for m in M for k in (30, 100)) else
            " (missing points still running)") + ". Teacher: gpt-6-luna, one run.",
         ha="center", va="top", fontsize=10.5, color="#444")
fig.tight_layout(); fig.savefig(f"{D}/student_variants.png", dpi=150, bbox_inches="tight"); print("ok")
