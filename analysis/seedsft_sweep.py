"""Seed-SFT dose sweep on dev140 (hpg B200, batch-invariant): table + plots (2026-10-05).

Inputs: the dose grid (configs/seedsft_dose_grid.json), the sweep eval root (<root>/<run>/<run>-<label>/eval, base under
<root>/k5s0/google/*/eval, gate failures as <root>/<run>/<run>-<label>/check/FAILED) and the teacher dev140 run.
Outputs in --out: seedsft_sweep.md (cell table, per-K summaries at matched passes / matched tokens, paired tests vs base),
seedsft_sweep.json, seedsft_sweep.png. Dev (valid_seen) only; the sealed split is never read.
Run: python3 analysis/seedsft_sweep.py --root reports/runs/eval/hpg-seedsft-sweep-bi --teacher reports/runs/teacher-dev140 \
       --out reports/figs/seedsft-sweep
"""
import argparse
import glob
import json
import statistics
from pathlib import Path

from analysis.k_ladder import load_run, mcnemar, wilson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--grid", default="configs/seedsft_dose_grid.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--extra-root", action="append", default=[],
                        help="re-evaluation roots (<root>/<run>-<label>/eval); used only for cells whose gate failed in --root")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = json.loads(Path(args.grid).read_text())["runs"]
    base_dir = glob.glob(str(root / "*" / "google" / "*" / "eval"))
    if len(base_dir) != 1:
        raise ValueError(f"expected one base run, found {base_dir}")
    base = load_run(base_dir[0])["outcomes"]
    teacher = load_run(args.teacher)["outcomes"]
    n = len(base)
    cells = []
    for run in grid:
        for c in run["checkpoints"]:
            cell_dir = root / run["name"] / f"{run['name']}-{c['label']}"
            entry = {"K": run["K"], "seed": run["support_seed"], "label": c["label"], "tokens": c["tokens"],
                     "passes": c["passes"], "status": "missing"}
            if (cell_dir / "check" / "FAILED").exists():
                entry["status"] = "gate_failed"
                for extra in args.extra_root:
                    alt = Path(extra) / f"{run['name']}-{c['label']}"
                    if (alt / "eval" / "summary.json").exists():
                        cell_dir, entry["reevaluated"] = alt, str(Path(extra))
                        entry["status"] = "pending_ok"
                        break
            if entry["status"] in ("missing", "pending_ok") and (cell_dir / "eval" / "summary.json").exists():
                outcomes = load_run(cell_dir / "eval")["outcomes"]
                s = sum(outcomes.values())
                entry.update(status="ok", successes=s, rate=s / len(outcomes), vs_base=mcnemar(outcomes, base),
                             vs_teacher=mcnemar(outcomes, teacher))
            cells.append(entry)
    (out / "seedsft_sweep.json").write_text(json.dumps(
        {"base": sum(base.values()), "teacher": sum(teacher.values()), "episodes": n, "cells": cells}, indent=1))

    labels = ["p03", "p05", "p10", "p20", "p30", "p40", "t100k", "t200k", "t400k"]
    lines = [f"Base {sum(base.values())}/{n} = {100 * sum(base.values()) / n:.1f}%; teacher {sum(teacher.values())}/{n} = "
             f"{100 * sum(teacher.values()) / n:.1f}%. Cells: successes/140; F = equivalence gate failed (not evaluated); "
             "· = merged into a nearby pass point (within 5%); † = evaluated after gate amendment 4 (schema 5).", "",
             "| K | support seed | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for run in grid:
        row = []
        for label in labels:
            cell = next((c for c in cells if c["K"] == run["K"] and c["seed"] == run["support_seed"] and c["label"] == label), None)
            row.append("·" if cell is None else "F" if cell["status"] == "gate_failed" else
                       "?" if cell["status"] == "missing" else
                       str(cell["successes"]) + ("†" if cell.get("reevaluated") else ""))
        lines.append(f"| {run['K']} | s{run['support_seed']} | " + " | ".join(row) + " |")
    lines += ["", "Per-K means over the support seeds that have an evaluated cell (n seeds in brackets):", "",
              "| K | 3 passes | 10 passes | 20 passes | 40 passes | ~400k tokens | best cell (seed, dose): success, vs base only-row/only-base, p |",
              "|---|---|---|---|---|---|---|"]
    for k in sorted({r["K"] for r in grid}):
        def mean_at(pred):
            vals = [c["successes"] for c in cells if c["K"] == k and c["status"] == "ok" and pred(c)]
            return f"{statistics.mean(vals):.1f} [{len(vals)}]" if vals else "—"
        best = max((c for c in cells if c["K"] == k and c["status"] == "ok"), key=lambda c: c["successes"])
        vb = best["vs_base"]
        lines.append(f"| {k} | {mean_at(lambda c: c['label'] == 'p03')} | {mean_at(lambda c: c['label'] == 'p10')} | "
                     f"{mean_at(lambda c: c['label'] == 'p20')} | {mean_at(lambda c: c['label'] == 'p40')} | "
                     f"{mean_at(lambda c: abs(c['tokens'] - 400_000) / 400_000 < 0.05)} | "
                     f"s{best['seed']} {best['label']}: {best['successes']}/{n}, {vb['only_a']}/{vb['only_b']}, p={vb['p']:.2g} |")
    lines += ["", "Serving-noise reference: base vs base on two non-invariant servers flipped 12/140 outcomes; under batch-invariant "
              "serving every number above is exactly reproducible.", ""]
    (out / "seedsft_sweep.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    colour = {5: "#2a78d6", 10: "#eb6834", 20: "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)
    for ax, key, xlabel, title in ((axes[0], "tokens", "supervised tokens seen (log scale)", "Success vs training exposure"),
                                   (axes[1], "passes", "passes over the K seeds (log scale)", "Success vs passes")):
        for name, value, style in (("teacher", sum(teacher.values()) / n, (0, (1, 0))), ("base", sum(base.values()) / n, (0, (4, 3)))):
            ax.axhline(100 * value, color=ink if name == "teacher" else "#8a8984", linestyle=style, linewidth=1.4)
            ax.annotate(f"{name} {100 * value:.1f}%", xy=(1.0, 100 * value), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=8.5, color=muted)
        for k in (5, 10, 20):
            for seed in (0, 1, 2):
                pts = sorted((c[key], 100 * c["rate"]) for c in cells if c["K"] == k and c["seed"] == seed and c["status"] == "ok"
                             and (key == "tokens" or c["label"].startswith("p")))
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour[k], linewidth=1, alpha=0.35,
                            marker="o", markersize=3)
            by_x = {}
            for c in cells:
                if c["K"] == k and c["status"] == "ok" and (key == "tokens" or c["label"].startswith("p")):
                    x = c["passes"] if key == "passes" else round(c["tokens"], -3)
                    by_x.setdefault(c["label"], []).append((c[key], 100 * c["rate"]))
            mean_pts = sorted((statistics.mean(x for x, _ in v), statistics.mean(y for _, y in v)) for v in by_x.values())
            ax.plot([p[0] for p in mean_pts], [p[1] for p in mean_pts], color=colour[k], linewidth=2.2, marker="o",
                    markersize=6, label=f"Seed-SFT K={k} (mean over seeds)")
        ax.set_xscale("log")
        ticks = [3, 5, 10, 20, 30, 40] if key == "passes" else [1e4, 2.5e4, 5e4, 1e5, 2e5, 4e5]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) if key == "passes" else f"{int(t / 1000)}k" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(xlabel, color=muted)
        ax.set_ylabel("dev140 success (%)", color=muted)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.grid(axis="y", color=grid_c, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.text(0.01, 0.01, "ALFWorld valid_seen (dev140), greedy ReAct, thinking off, batch-invariant vLLM on hpg B200; thin lines = "
             "individual support seeds; gate-failed cells omitted", fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "seedsft_sweep.png")
    print(out / "seedsft_sweep.md")


if __name__ == "__main__":
    main()
