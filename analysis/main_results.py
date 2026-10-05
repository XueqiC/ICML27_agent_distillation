"""Main results: success vs K for teacher, base, Seed-SFT (several doses) and every baseline as its own line (2026-10-05).

Inputs: configs/main_results_sources.json (one entry per evaluated cell: method, K, support seed, eval dir with episodes/ and
summary.json) plus the Seed-SFT sweep root(s). Outputs in --out: main_results.md (table: method x K, mean over support seeds with
the per-seed values), main_results.json, main_results.png. Dev (valid_seen) only; sealed episodes are refused by load_run.
Run: python3 analysis/main_results.py --out reports/figs/main-results
"""
import argparse
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from analysis.k_ladder import load_run

# Fixed order so a method keeps its colour when others are added (never cycled).
METHOD_ORDER = ["Seed-SFT (3 passes)", "Seed-SFT (40 passes)", "Seed-SFT (~400k tokens)",
                "Kang (official) + SAG", "AgentGym BC", "AgentEvol (final iter.)", "ETO (final iter.)", "IPR (final iter.)",
                "LEAP (privileged)", "LEAP (0% privileged)"]
PALETTE = ["#9ec5f0", "#2a78d6", "#0b3f8a", "#eb6834", "#1baf7a", "#0f7a52", "#8b5cf6", "#d63f8c", "#b8860b", "#c9a227"]
STYLE = {m: ("-" if m.startswith("Seed-SFT") else "--") for m in METHOD_ORDER}


def seedsft_cells(roots, grid_paths):
    cells = []
    runs = {}
    for g in grid_paths:
        for run in json.loads(Path(g).read_text())["runs"]:
            runs[run["name"]] = run
    for root in roots:
        for path in glob.glob(str(Path(root) / "*" / "k*s*-*" / "eval" / "summary.json")) + \
                    glob.glob(str(Path(root) / "k*s*-*" / "eval" / "summary.json")):
            name = Path(path).parent.parent.name
            m = re.fullmatch(r"k(\d+)s(\d+)-(p\d+|t\d+k)", name)
            if not m or f"k{m.group(1)}s{m.group(2)}" not in runs:
                continue
            k, seed, label = int(m.group(1)), int(m.group(2)), m.group(3)
            run = runs[f"k{k}s{seed}"]
            ck = next(c for c in run["checkpoints"] if c["label"] == label)
            out = load_run(Path(path).parent)["outcomes"]
            cells.append({"K": k, "seed": seed, "label": label, "tokens": ck["tokens"], "passes": ck["passes"],
                          "successes": sum(out.values()), "n": len(out)})
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="configs/main_results_sources.json")
    parser.add_argument("--sweep-root", action="append",
                        default=["reports/runs/eval/hpg-seedsft-sweep-bi", "reports/runs/eval/hpg-seedsft-sweep-bi-reeval-s5"])
    parser.add_argument("--grid", action="append", default=["configs/seedsft_dose_grid.json"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path(args.sources).read_text())
    refs = {}
    for name in ("teacher", "base"):
        o = load_run(sources["references"][name])["outcomes"]
        refs[name] = (sum(o.values()), len(o))
    values = defaultdict(lambda: defaultdict(dict))  # method -> K -> seed -> successes
    for c in seedsft_cells(args.sweep_root, args.grid):
        if c["label"] == "p03":
            values["Seed-SFT (3 passes)"][c["K"]][c["seed"]] = c["successes"]
        if c["label"] == "p40":
            values["Seed-SFT (40 passes)"][c["K"]][c["seed"]] = c["successes"]
        if abs(c["tokens"] - 400_000) / 400_000 < 0.05:
            values["Seed-SFT (~400k tokens)"][c["K"]][c["seed"]] = c["successes"]
    for entry in sources["baselines"]:
        o = load_run(entry["eval_dir"])["outcomes"]
        values[entry["method"]][entry["K"]][entry["seed"]] = sum(o.values())
    n = refs["base"][1]
    lines = [f"Teacher {refs['teacher'][0]}/{n}; base {refs['base'][0]}/{n}. Cells: mean successes over support seeds "
             "[per-seed values]; dev140, batch-invariant serving, hpg B200.", "",
             "| method | " + " | ".join(f"K={k}" for k in sorted({k for m in values.values() for k in m})) + " |",
             "|---|" + "---|" * len({k for m in values.values() for k in m})]
    ks_all = sorted({k for m in values.values() for k in m})
    for method in METHOD_ORDER:
        if method not in values:
            continue
        row = []
        for k in ks_all:
            seeds = values[method].get(k, {})
            row.append(f"{statistics.mean(seeds.values()):.1f} [{', '.join(str(seeds[s]) for s in sorted(seeds))}]" if seeds else "—")
        lines.append(f"| {method} | " + " | ".join(row) + " |")
    (out / "main_results.md").write_text("\n".join(lines) + "\n")
    (out / "main_results.json").write_text(json.dumps({"references": refs, "values": {m: {str(k): v for k, v in d.items()}
                                                                                      for m, d in values.items()}}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=150)
    for name, style in (("teacher", (0, (1, 0))), ("base", (0, (4, 3)))):
        v = 100 * refs[name][0] / refs[name][1]
        ax.axhline(v, color=ink if name == "teacher" else "#8a8984", linestyle=style, linewidth=1.4)
        ax.annotate(f"{name} {v:.1f}%", xy=(1.0, v), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", fontsize=8.5, color=muted)
    for i, method in enumerate(METHOD_ORDER):
        if method not in values:
            continue
        pts = sorted((k, 100 * statistics.mean(s.values()) / n, [100 * x / n for x in s.values()])
                     for k, s in values[method].items() if s)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PALETTE[i], linestyle=STYLE[method], linewidth=2,
                marker="o" if method.startswith("Seed-SFT") else "D", markersize=6, label=method)
        for k, _, seeds in pts:
            ax.scatter([k] * len(seeds), seeds, color=PALETTE[i], s=10, alpha=0.4, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(ks_all)
    ax.set_xticklabels([str(k) for k in ks_all])
    ax.minorticks_off()
    ax.set_xlabel("K (trusted teacher seed trajectories, log scale)", color=muted)
    ax.set_ylabel("dev140 success (%)", color=muted)
    ax.set_title("Success vs K: Seed-SFT doses and baselines (official recipes)", color=ink, fontsize=11, loc="left")
    ax.grid(axis="y", color=grid_c, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.text(0.01, 0.01, "Lines: mean over support seeds; faint dots: individual seeds. ALFWorld valid_seen, greedy ReAct, "
             "thinking off, batch-invariant vLLM on hpg B200.", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "main_results.png", bbox_inches="tight")
    print(out / "main_results.md")


if __name__ == "__main__":
    main()
