import json, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
S = {k: len(v) for k, v in json.load(open("solved_sets.json")).items()}
KS = [30, 100, 300, 500]; M = ["tsft", "AD", "SAD", "SmartAD"]
LBL = {"tsft": "Trajectory SFT", "AD": "AD (Kang)", "SAD": "SAD", "SmartAD": "SmartAD"}
COL = {"tsft": "#0072B2", "AD": "#E69F00", "SAD": "#009E73", "SmartAD": "#CC79A7"}
MK = {"tsft": "o", "AD": "s", "SAD": "^", "SmartAD": "D"}
COLL = {("conv", "tsft", 100), ("conv", "tsft", 300), ("conv", "AD", 100), ("conv", "SmartAD", 100), ("conv", "SmartAD", 300)}
if os.environ.get("AD_K300_COLLAPSE") == "1": COLL.add(("conv", "AD", 300))
OVR = {("conv", "SmartAD", 30)}   # serving check failed, evaluated with the user's recorded approval
DIV = {("paper", "tsft", 100), ("paper", "SAD", 100), ("paper", "AD", 500)}
if os.environ.get("AD_K300"): DIV.add(("conv", "AD", 300))
def wilson(x, n=140, z=1.96):
    p = x / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d * n, (c + h) / d * n
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, col, title in zip(axes, ("conv", "paper"), ("Converged column (doubling ladder)", "Paper-recipe column")):
    ax.axhline(62, color="#888", ls="--", lw=1.5); ax.text(31, 64, "base 62", color="#666", fontsize=11)
    for j, m in enumerate(M):
        xs = [k for k in KS if f"{col}|{m}|{k}" in S]; ys = [S[f"{col}|{m}|{k}"] for k in xs]
        xo = [k * (1 + 0.035 * (j - 1.5)) for k in xs]
        ax.plot(xo, ys, color=COL[m], lw=2, marker=MK[m], ms=8, label=LBL[m], zorder=3)
        for x, k, y in zip(xo, xs, ys):
            lo, hi = wilson(y); ax.vlines(x, lo, hi, color=COL[m], lw=1.2, alpha=0.6)
            if (col, m, k) in COLL:
                ax.plot(x, y, marker=MK[m], ms=13, mfc="none", mec="#222", mew=1.2, zorder=4)
            mark = ("†" if (col, m, k) in DIV else "") + ("‡" if (col, m, k) in OVR else "")
            if mark:
                ax.annotate(mark, (x, y), xytext=(8, 2), textcoords="offset points", fontsize=13, color="#222")
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([f"K{k}" for k in KS]); ax.minorticks_off()
    ax.set_title(title); ax.set_ylim(-3, 140); ax.grid(axis="y", color="#e5e5e5"); ax.set_xlabel("seed trajectories K")
axes[0].set_ylabel("dev140 successes (of 140)")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
fig.text(0.5, -0.02, "Bars: 95% Wilson interval over tasks (single training seed). Ringed: training loss collapsed mid-run and recovered. "
         "†: rerun with halved lr after the first run scored ~0. ‡: serving check failed, evaluated with approval.",
         ha="center", fontsize=11, color="#444")
fig.tight_layout(); fig.savefig("baseline_perf_k.png", dpi=150, bbox_inches="tight"); print("ok")
