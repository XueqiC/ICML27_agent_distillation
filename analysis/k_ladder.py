"""K-ladder table and line plots for dev140 (2026-10-04).

Reads one ladder eval root (one server: base + LoRA students named k{K}s{seed}-p{passes}), the teacher run, and writes
  <out>/k_ladder.md    table: success, paired comparison vs base (McNemar exact), parser/inadmissible counts, per type
  <out>/k_ladder.json  the same numbers
  <out>/k_ladder.png   left: success vs K per training dose; right: success vs passes per K (teacher/base as reference lines)
Run: python3 analysis/k_ladder.py --ladder reports/runs/eval/ladder-s0-rai-dev140 --teacher reports/runs/teacher-dev140 \
       --out reports/figs/ladder-s0
Dev numbers only (valid_seen); the sealed split is never read here.
"""
import argparse
import glob
import json
import math
import re
from pathlib import Path

from analysis.serving import SERVING_NOISE_REFERENCE, SERVING_NOISE_TEXT

TYPES = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
         "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]
SHORT = {"pick_and_place_simple": "pick", "look_at_obj_in_light": "look", "pick_clean_then_place_in_recep": "clean",
         "pick_heat_then_place_in_recep": "heat", "pick_cool_then_place_in_recep": "cool", "pick_two_obj_and_place": "two"}


def batch_invariant_setting(summary, provenance):
    """Missing legacy metadata is unknown, never evidence that the mode was off."""
    config = provenance.get("config", {})
    sources = [summary, config.get("suite", {}),
               config.get("server_provenance", {}).get("config", {}).get("server", {})]
    values = [source["batch_invariant"] for source in sources if "batch_invariant" in source]
    if any(value is not None and type(value) is not bool for value in values):
        raise ValueError("batch_invariant must be boolean or unknown (null)")
    if len(set(values)) > 1:
        raise ValueError("Conflicting batch_invariant summary/provenance")
    return values[0] if values else None


def serving_label(value):
    return "unknown" if value is None else str(int(value))


def guard_serving(rows, *, allow_mixed=False):
    # The black-box teacher does not use our vLLM student server.
    settings = {r["batch_invariant"] for r in rows if r["name"] != "teacher"}
    if len(settings) > 1 and not allow_mixed:
        raise ValueError("Mixed batch_invariant settings; re-run with one setting or pass --allow-mixed-serving to label them")
    return len(settings) > 1


def load_run(eval_dir):
    eval_dir = Path(eval_dir)
    outcomes, steps = {}, {"parser_error": 0, "inadmissible": 0, "legal": 0}
    for path in glob.glob(str(eval_dir / "episodes" / "*" / "episode.json")):
        episode = json.loads(Path(path).read_text())
        if episode["split"] == "test" or episode.get("sealed"):
            raise ValueError("Sealed episode in a dev analysis input")
        outcomes[episode["task_id"]] = bool(episode.get("success"))
        for step in episode["steps"]:
            outcome = step.get("action_outcome")
            if outcome in steps:
                steps[outcome] += 1
    summary = eval_dir / "summary.json"
    provenance = eval_dir / "provenance.json"
    setting = batch_invariant_setting(json.loads(summary.read_text()) if summary.exists() else {},
                                      json.loads(provenance.read_text()) if provenance.exists() else {})
    return {"outcomes": outcomes, "steps": steps, "complete": summary.exists(), "batch_invariant": setting}


def mcnemar(a, b):
    """Exact two-sided McNemar on discordant pairs; a/b dicts task -> bool over the same tasks."""
    keys = sorted(set(a) & set(b))
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    n, k = only_a + only_b, min(only_a, only_b)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n": len(keys), "both": sum(a[k] and b[k] for k in keys), "only_a": only_a, "only_b": only_b,
            "neither": len(keys) - only_a - only_b - sum(a[k] and b[k] for k in keys), "p": p,
            "serving_noise_reference": dict(SERVING_NOISE_REFERENCE)}


def wilson(successes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def task_type(task_id):
    return task_id.split("/")[1].split("-")[0]


def row(name, run, base):
    out = run["outcomes"]
    n, s = len(out), sum(out.values())
    lo, hi = wilson(s, n)
    by_type = {SHORT[t]: (sum(v for k, v in out.items() if task_type(k) == t),
                          sum(1 for k in out if task_type(k) == t)) for t in TYPES}
    entry = {"name": name, "successes": s, "episodes": n, "rate": s / n if n else float("nan"),
             "batch_invariant": run["batch_invariant"],
             "wilson95": [lo, hi], "steps": run["steps"], "by_type": by_type, "complete": run["complete"]}
    if base is not None and name != "base":
        entry["vs_base"] = mcnemar(out, base["outcomes"])
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", required=True, nargs="+", help="eval roots; base is read from the first")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-mixed-serving", action="store_true",
                        help="Permit mixed/unknown batch_invariant settings, labelled in tables and plots")
    args = parser.parse_args(argv)
    roots, out = [Path(r) for r in args.ladder], Path(args.out)
    base_dirs = glob.glob(str(roots[0] / "google" / "*" / "eval"))
    base = load_run(base_dirs[0]) if base_dirs else None
    rows = []
    teacher = load_run(args.teacher)
    rows.append(row("teacher", teacher, base))
    if base:
        rows.append(row("base", base, None))
    students = {}
    paths = [p for root in roots for p in sorted(glob.glob(str(root / "k*s*-p*" / "eval")))]
    for path in paths:
        name = Path(path).parent.name
        match = re.fullmatch(r"k(\d+)s(\d+)-p(\d+)", name)
        if not match or not (Path(path) / "summary.json").exists():
            continue
        k, seed, passes = map(int, match.groups())
        if (k, passes) in students:
            raise ValueError(f"{name} evaluated in more than one root; pass one of them")
        entry = row(f"Seed-SFT K={k} s{seed} {passes} passes", load_run(path), base)
        entry.update(K=k, seed=seed, passes=passes)
        rows.append(entry)
        students[(k, passes)] = entry
    mixed_serving = guard_serving(rows, allow_mixed=args.allow_mixed_serving)
    out.mkdir(parents=True, exist_ok=True)
    (out / "k_ladder.json").write_text(json.dumps(rows, indent=1))

    lines = [SERVING_NOISE_TEXT, ""]
    if mixed_serving:
        lines += ["Mixed serving settings permitted by --allow-mixed-serving; batch_invariant is labelled for every row.", ""]
    lines += ["| row | batch_invariant | success | 95% CI | vs base (both/only base/only row/neither, McNemar p) | parser err | inadmissible | "
             + " | ".join(SHORT[t] for t in TYPES) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * len(TYPES)]
    for r in rows:
        vs = r.get("vs_base")
        vs_text = (f"{vs['both']}/{vs['only_b']}/{vs['only_a']}/{vs['neither']}, p={vs['p']:.3g}" if vs else "—")
        partial = "" if r["complete"] else " (partial)"
        setting = "n/a (teacher)" if r["name"] == "teacher" else serving_label(r["batch_invariant"])
        lines.append(f"| {r['name']}{partial} | {setting} | {r['successes']}/{r['episodes']} = {100 * r['rate']:.1f}% | "
                     f"{100 * r['wilson95'][0]:.0f}–{100 * r['wilson95'][1]:.0f} | {vs_text} | "
                     f"{r['steps']['parser_error']} | {r['steps']['inadmissible']} | "
                     + " | ".join(f"{a}/{b}" for a, b in (r["by_type"][SHORT[t]] for t in TYPES)) + " |")
    (out / "k_ladder.md").write_text("\n".join(lines) + "\n")
    plot_ladder(rows, students, base, out)
    print(out / "k_ladder.md")


def plot_ladder(rows, students, base, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]  # fixed order, one per dose / per K
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    ks = sorted({k for k, _ in students})
    doses = sorted({p for _, p in students})
    ref = [("teacher", rows[0]["rate"], "#0b0b0b", (0, (1, 0)))]
    if base:
        ref.append((f"base student (BI={serving_label(base['batch_invariant'])})", rows[1]["rate"], "#8a8984", (0, (4, 3))))
    for ax, xs, groups, xlabel, title in (
            (axes[0], ks, doses, "K (trusted seed trajectories)", "Success vs K, per training dose"),
            (axes[1], doses, ks, "training passes over the K seeds", "Success vs dose, per K")):
        for name, value, colour, style in ref:
            ax.axhline(100 * value, color=colour, linestyle=style, linewidth=1.5)
            ax.text(xs[-1] if xs else 1, 100 * value + 1.2, f"{name} {100 * value:.1f}%", color=muted,
                    fontsize=8.5, ha="right")
        for i, g in enumerate(groups):
            pts = [(x, students[(x, g) if ax is axes[0] else (g, x)]) for x in xs
                   if ((x, g) if ax is axes[0] else (g, x)) in students]
            if not pts:
                continue
            label = f"{g} passes" if ax is axes[0] else f"K={g}"
            for setting in sorted({p[1]["batch_invariant"] for p in pts}, key=serving_label):
                group = [p for p in pts if p[1]["batch_invariant"] is setting]
                ax.plot([p[0] for p in group], [100 * p[1]["rate"] for p in group],
                        color=palette[i % len(palette)], linewidth=2,
                        marker={True: "o", False: "s", None: "x"}[setting], markersize=6,
                        label=f"{label}, BI={serving_label(setting)}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(xlabel, color=muted)
        ax.set_ylabel("dev140 success (%)", color=muted)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.text(0.01, 0.01, "valid_seen (dev140), greedy, ReAct, thinking off; rai Blackwell GPU1 — preliminary, "
             "paper numbers on hpg B200", fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "k_ladder.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
