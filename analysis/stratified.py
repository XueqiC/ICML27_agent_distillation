"""Numerical Addendum A readout for labeled dev140 evaluation directories."""
import argparse
from pathlib import Path
import re
from statistics import NormalDist

from analysis.k_ladder import guard_serving, load_run, mcnemar, serving_label, wilson
from analysis.mechanism.common import output_guard
from analysis.serving import SERVING_NOISE_TEXT
from analysis.traits import (BUCKETS, Inputs, ROOT, dev_episode, known_outcome, load_traits,
                             provenance, require_dev)
from sed.storage import atomic_write, write_json


def load_eval(directory, traits, inputs):
    """Use k_ladder's reader, but reject duplicates and retain unknown/missing results."""
    directory = Path(directory)
    paths = sorted(directory.glob("episodes/*/episode.json"))
    if not paths:
        raise ValueError("Evaluation directory has no episodes")
    episodes = {}
    for path in paths:
        episode = inputs.read(path)
        dev_episode(episode, traits["games"])
        task_id = episode["task_id"]
        if task_id in episodes:
            raise ValueError("Duplicate task in evaluation row; select one run")
        episodes[task_id] = episode
    for name in ("tasks.json", "summary.json", "provenance.json"):
        if (directory / name).exists():
            value = inputs.read(directory / name)
            if name == "tasks.json":
                for task_id in value["task_ids"]:
                    require_dev(task_id)
                if value != traits["dev140"]:
                    raise ValueError("Evaluation task list differs from frozen dev140")
    run = load_run(directory)
    # load_run coerces missing/null success to False. Correct only these unknowns,
    # without changing its outcome semantics for complete, successful executions.
    for task_id, episode in episodes.items():
        if known_outcome(episode) is None:
            run["outcomes"][task_id] = None
    run["directory"] = str(directory)
    run["outcome_source"] = "recorded episode.success; complete=true and status=ok; audit blobs not required"
    return run


def success_summary(outcomes, task_ids):
    known = {t: outcomes[t] for t in task_ids if type(outcomes.get(t)) is bool}
    n, successes = len(known), sum(known.values())
    return {"n_games": len(task_ids), "n_known": n, "successes": successes,
            "n_missing": sum(t not in outcomes for t in task_ids),
            "n_unknown": sum(t in outcomes and type(outcomes[t]) is not bool for t in task_ids),
            "rate": successes / n if n else None,
            "wilson95": list(wilson(successes, n)) if n else [None, None]}


def paired_difference(row, reference, task_ids):
    ids = [t for t in task_ids if type(row.get(t)) is bool and type(reference.get(t)) is bool]
    result = mcnemar({t: row[t] for t in ids}, {t: reference[t] for t in ids})
    n = len(ids)
    result.update(task_ids=ids, n_unpaired=len(task_ids) - n,
                  difference=(result["only_a"] - result["only_b"]) / n if n else None,
                  difference95=[None, None],
                  difference95_method="Bonferroni-Wilson: subtract simultaneous 97.5% intervals for discordant probabilities")
    if not n:
        result["p"] = None
    else:
        # Paired contrast P(row-only) - P(reference-only), rather than treating
        # marginal row/reference successes as independent. Nominal conservative CI.
        z = NormalDist().inv_cdf(1 - .05 / 4)
        lo_a, hi_a = wilson(result["only_a"], n, z=z)
        lo_b, hi_b = wilson(result["only_b"], n, z=z)
        result["difference95"] = [max(-1., lo_a - hi_b), min(1., hi_a - lo_b)]
    return result


def coverage_context(label, traits):
    match = re.match(r"^k(\d+)s(\d+)-.+$", label)
    if not match:
        return None
    k, seed = map(int, match.groups())
    context = f"k{k}s{seed}"
    if context not in traits["coverage"]:
        raise ValueError("Row K/support seed has no frozen coverage table")
    return context


def build_readout(traits, runs, *, base="base", teacher="teacher", allow_mixed_serving=False):
    if base not in runs or teacher not in runs or base == teacher:
        raise ValueError("Distinct base and teacher reference rows are required")
    mixed = guard_serving([{"name": "teacher" if label == teacher else "student",
                            "batch_invariant": run["batch_invariant"]} for label, run in runs.items()],
                          allow_mixed=allow_mixed_serving)
    contexts = {label: coverage_context(label, traits) for label in runs}
    records = []

    def partition(trait, context, labels):
        for bucket in BUCKETS[trait]:
            def member(row):
                if context is not None:
                    return row["seed_coverage"][context]["bucket"]
                return row.get(trait + "_bucket", row.get(trait))
            ids = [t for t in traits["task_ids"] if member(traits["games"][t]) == bucket]
            entries = {}
            for label in labels:
                outcomes = runs[label]["outcomes"]
                vs_teacher = paired_difference(outcomes, runs[teacher]["outcomes"], ids)
                entries[label] = {**success_summary(outcomes, ids),
                                  "vs_base": paired_difference(outcomes, runs[base]["outcomes"], ids),
                                  "vs_teacher": vs_teacher,
                                  "gap_teacher_minus_row": (-vs_teacher["difference"]
                                                            if vs_teacher["difference"] is not None else None),
                                  "gap95": ([-vs_teacher["difference95"][1], -vs_teacher["difference95"][0]]
                                            if vs_teacher["difference"] is not None else [None, None])}
            records.append({"trait": trait, "coverage_context": context, "bucket": bucket,
                            "n_games": len(ids), "task_groups": len({traits["games"][t]["group_id"] for t in ids}),
                            "task_ids": ids, "rows": entries})

    for trait in BUCKETS:
        if trait != "seed_coverage":
            partition(trait, None, runs)
    for context in sorted({c for c in contexts.values() if c is not None}):
        # References use exactly the matched student's coverage partition.
        labels = [label for label in runs if label in {base, teacher} or contexts[label] == context]
        partition("seed_coverage", context, labels)
    return {"schema_version": 1, "traits_sha256": traits["traits_sha256"], "base": base, "teacher": teacher,
            "unique_instances": len(traits["task_ids"]), "task_groups": traits["task_groups"],
            "mixed_serving": mixed, "paired_difference_orientation": "row minus reference",
            "rate_unit": "proportion", "pairing_unit": "dev game instance; task groups counted separately",
            "gap_denominator": "same known game pairs as vs_teacher",
            "rows": {label: {"directory": run.get("directory"), "batch_invariant": run["batch_invariant"],
                              "summary_present": run["complete"], "coverage_context": contexts[label],
                              "outcome_source": run.get("outcome_source"),
                              "overall": success_summary(run["outcomes"], traits["task_ids"])}
                     for label, run in runs.items()}, "buckets": records}


def markdown(report):
    def num(value):
        return "NA" if value is None else f"{value:.6f}"

    def interval(values):
        return "[" + ", ".join(num(v) for v in values) + "]"

    def paired(value):
        return (f"{value['n']} / {num(value['difference'])} / {interval(value['difference95'])} / "
                f"{num(value['p'])} / {value['only_a']}:{value['only_b']}")

    lines = ["# Dev140 stratified readout", "", f"Traits SHA-256: `{report['traits_sha256']}`.", "",
             "Rates, differences and gaps are proportions. Wilson intervals are 95%. Differences are row minus reference; "
             "paired 95% intervals subtract Bonferroni-Wilson 97.5% intervals for the two discordant probabilities. "
             "McNemar p is exact, two-sided and unadjusted. NA denotes no known denominator.", "",
             "Unknown and missing outcomes are excluded from known denominators; paired comparisons use their own common known games. "
             "Coverage contexts apply only to matching k{K}s{S}-... rows and their base/teacher references. "
             "Intervals and tests use game instances; task groups are counted separately.", "", SERVING_NOISE_TEXT, "",
             "| row | batch_invariant | coverage context | known / games | successes | missing | unknown |",
             "|---|---|---|---|---|---|---|"]
    for label, row in report["rows"].items():
        overall = row["overall"]
        setting = "n/a" if label == report["teacher"] else serving_label(row["batch_invariant"])
        lines.append(f"| {label} | {setting} | {row['coverage_context'] or 'n/a'} | "
                     f"{overall['n_known']} / {overall['n_games']} | {overall['successes']} | "
                     f"{overall['n_missing']} | {overall['n_unknown']} |")
    lines += ["", "Comparison columns: paired n / difference / 95% interval / exact p / only-row:only-reference. "
              "Gap = teacher minus row on known teacher/row pairs.", "",
              "| trait | coverage context | bucket | games / groups | row | known | missing / unknown | successes | rate [Wilson 95%] | vs base | vs teacher | gap [95%] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for bucket in report["buckets"]:
        for label, row in bucket["rows"].items():
            lines.append(f"| {bucket['trait']} | {bucket['coverage_context'] or 'n/a'} | {bucket['bucket']} | "
                         f"{bucket['n_games']} / {bucket['task_groups']} | {label} | {row['n_known']} | "
                         f"{row['n_missing']} / {row['n_unknown']} | {row['successes']} | "
                         f"{num(row['rate'])} {interval(row['wilson95'])} | {paired(row['vs_base'])} | "
                         f"{paired(row['vs_teacher'])} | {num(row['gap_teacher_minus_row'])} {interval(row['gap95'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits", default="data/manifest/dev140_traits.json")
    parser.add_argument("--row", action="append", required=True, metavar="LABEL=EVAL_DIR")
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--teacher-label", default="teacher")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-mixed-serving", action="store_true")
    args = parser.parse_args(argv)
    paths = {}
    for spec in args.row:
        label, sep, path = spec.partition("=")
        if not sep or not re.fullmatch(r"[\w.-]+", label) or not path or label in paths:
            parser.error("Each --row needs a unique LABEL=EVAL_DIR (label: letters, digits, _, -, .)")
        paths[label] = path
    output = output_guard(args.out, inputs=[args.traits, *paths.values()])
    if output.is_relative_to((ROOT / "reports/runs").resolve()):
        raise ValueError("Readout cannot write into reports/runs")
    if output.exists():
        raise FileExistsError("Refusing to overwrite a stratified readout")
    inputs = Inputs()
    inputs.add(args.traits)
    traits = load_traits(args.traits)
    runs = {label: load_eval(path, traits, inputs) for label, path in paths.items()}
    report = build_readout(traits, runs, base=args.base_label, teacher=args.teacher_label,
                           allow_mixed_serving=args.allow_mixed_serving)
    inputs.verify()
    report["provenance"] = provenance(vars(args), inputs)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "stratified.json", report, exclusive=True)
    atomic_write(output / "stratified.md", markdown(report).encode(), exclusive=True)
    print(output / "stratified.md")


if __name__ == "__main__":
    main()
