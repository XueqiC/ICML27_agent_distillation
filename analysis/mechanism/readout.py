"""Section-6 descriptive readout with Wilson intervals and explicit missing evidence."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from analysis.mechanism.common import (completed_branch, gap_episodes, load_episode, new_output, outcome, audit_states,
                                      read_json)
from analysis.mechanism.goals import refuse_sealed
from analysis.mechanism.fce import IncompleteProbe, locate_closest_approach, sample_distance, probe_episode
from analysis.mechanism.interventions import ARMS, arm_outcome
from analysis.mechanism.taxonomy import LABELS, shares
from analysis.serving import SERVING_NOISE_REFERENCE, SERVING_NOISE_TEXT
from sed.config import load_config
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, write_json


def wilson(successes, total):
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    if not total:
        return {"count": successes, "n": total, "rate": None, "ci95": [None, None]}
    z = 1.959963984540054
    p = successes/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return {"count": successes, "n": total, "rate": p, "ci95": [max(0, center-half), min(1, center+half)]}


def paired_mcnemar(first, second):
    keys = {k for k in first.keys() & second.keys() if type(first[k]) is bool and type(second[k]) is bool}
    b = sum(first[k] and not second[k] for k in keys)
    c = sum(second[k] and not first[k] for k in keys)
    n = b+c
    p = min(1., 2*sum(math.comb(n, i) for i in range(min(b, c)+1))/2**n) if n else 1.
    return {"paired_n": len(keys), "first_only": b, "second_only": c, "exact_two_sided_p": p,
            "serving_noise_reference": dict(SERVING_NOISE_REFERENCE)}


def logical_or(a, b):
    return True if a is True or b is True else False if a is False and b is False else None


def verified_arms(entry):
    result = {}
    for arm in ARMS:
        row = entry.get("arms", {}).get(arm, {})
        if row.get("status") == "no_same_type_seed":
            result[arm] = None
        else:
            result[arm] = arm_outcome(row.get("samples", []))["rescue"]
    return result


def decision_cell(arms):
    """Section 6 uses paired excess rescues, already adjusted for each state's I0."""
    experience = logical_or(arms["I5"], arms["I1"])
    information = logical_or(arms["I3"], arms["I2"])
    if experience is None or information is None:
        return "incomplete"
    return { (True, True): "both", (True, False): "M1_cell", (False, True): "M2_cell",
             (False, False): "neither" }[experience, information]


def paired_excess(arms):
    """Unknown controls or arms cannot establish a paired excess rescue."""
    return {arm: (arms[arm] and not arms["I0"]
                  if type(arms[arm]) is bool and type(arms["I0"]) is bool else None)
            for arm in ARMS}


def distance_diagnostics(row):
    t = row.get("t_star")
    return {"task_id": row["task_id"], "status": row.get("status", "missing_fce"),
            "t_star": t, "D_0": row.get("D_0"), "D_t_star": row.get("D_t_star"),
            "turns_remaining_at_t_star": row.get("turns_remaining_at_t_star"),
            "turns_spent_after_t_star_without_progress": row["episode_length"]-t if t is not None else None,
            "budget_fce": row.get("budget_fce", row.get("fce")),
            "budget_fce_status": row.get("budget_fce_status"), "D_curve": row.get("D_curve", {})}


def build_readout(taxonomy, fce, interventions, coverage=None, probes=None):
    coverage = coverage or {}
    probe_report = probes or {}
    probes = {r["task_id"]: r for r in probe_report.get("episodes", [])}
    if "episodes" in coverage:
        coverage = {r["task_id"]: r for r in coverage["episodes"]}
    gap = set(taxonomy["G"]["task_ids"])
    if not set(coverage) <= gap or not set(probes) <= gap:
        raise ValueError("Coverage/probes include tasks outside the frozen gap set")
    taxonomy_rows = taxonomy["runs"]["student"]["episodes"]
    taxonomy_by_id = {r["task_id"]: r for r in taxonomy_rows}
    for row in taxonomy_rows:
        refuse_sealed(row["task_id"])
    fces = {r["task_id"]: r for r in fce["episodes"]}
    entries = {r["task_id"]: r for r in interventions["episodes"]}
    if len(entries) != len(interventions["episodes"]) or len(fces) != len(fce["episodes"]):
        raise ValueError("Duplicate task results")
    if not set(entries) <= gap or not set(fces) <= gap:
        raise ValueError("Results include tasks outside the frozen gap set")
    cohorts = {"primary": [], "F-format": [], "no_seed_task_type": []}
    exclusions = Counter()
    cells, rows, arms_by_task, excess_by_task = Counter(), [], {}, {}
    non_deterministic = []
    for task_id in sorted(gap):
        refuse_sealed(task_id)
        e, f = entries.get(task_id, {}), fces.get(task_id, {})
        if f.get("status") != "closest_approach_found":
            exclusions[f.get("status", "missing_fce")] += 1
            continue
        if not e or e.get("status") in {"no_fce", "no_t_star"}:
            exclusions["missing_interventions"] += 1
            continue
        if e.get("t") != f["t_star"]:
            raise ValueError("Interventions differ from the frozen closest-approach state")
        control = e.get("arms", {}).get("I0", {}).get("samples", [])
        greedy = next((s for s in control if s["sample_index"] == 0), None)
        if e.get("non_deterministic") or (greedy or {}).get("branch", {}).get("non_deterministic"):
            non_deterministic.append(task_id)
        arms = verified_arms(e)
        excess = paired_excess(arms)
        cohort = ("F-format" if "F-format" in taxonomy_by_id[task_id]["labels"] else
                  "no_seed_task_type" if e.get("task_type_covered") is False else "primary")
        cohorts[cohort].append((task_id, arms))
        if cohort != "primary":
            exclusions[cohort + "_separate"] += 1
            continue
        arms_by_task[task_id] = arms
        excess_by_task[task_id] = excess
        cell = decision_cell(excess)
        cells[cell] += 1
        covered = coverage.get(task_id, {}).get("covered", "unknown")
        cov, probe = coverage.get(task_id, {}), probes.get(task_id, {})
        for evidence in (cov, probe):
            if "t_star" in evidence and evidence["t_star"] != f["t_star"]:
                raise ValueError("Coverage/probe state differs from closest approach")
        absorbed = probe.get("absorbed")
        label = {"both": "both_mechanisms_cell", "M1_cell": "M1_requires_coverage",
                 "M2_cell": "M2", "neither": "unresolved", "incomplete": "incomplete"}[cell]
        if cell in {"both", "M1_cell"} and covered is True:
            label = "both_mechanisms" if cell == "both" else "M1"
            if cell == "M1_cell" and type(absorbed) is bool:
                label = "M1b" if absorbed else "M1a"
        if cell in {"both", "M1_cell"} and covered is False:
            label = "M2_with_knowledge_not_in_experience" if cell == "both" else "knowledge_not_in_experience"
        if cell == "neither" and excess["I4"] is True:
            label = "execution_or_capacity"
        elif cell == "neither" and excess["I4"] is None:
            label = "incomplete_I4"
        rows.append({**distance_diagnostics(f), "cell": cell, "label": label, "needed_step_coverage": covered,
                     "coverage_status": cov.get("status", "missing"),
                     "coverage_same_task_type": cov.get("covered_same_task_type"),
                     "coverage_any_task_type": cov.get("covered_any_task_type", covered),
                     "absorbed": absorbed, "probe_status": probe.get("status", "missing"),
                     "absorbed_same_task_type": probe.get("absorbed_same_task_type"),
                     "M1_subtype": ("M1b" if absorbed else "M1a") if covered is True and type(absorbed) is bool
                                      and cell in {"both", "M1_cell"} else None,
                     "task_type_covered": e.get("task_type_covered"), "arms": arms,
                     "paired_excess": excess, "non_deterministic": task_id in non_deterministic})
    rates, paired_rates = {}, {}
    for cohort, values in cohorts.items():
        rates[cohort] = {}
        paired_rates[cohort] = {}
        for arm in ARMS:
            available = [a[arm] for _, a in values if type(a[arm]) is bool]
            rates[cohort][arm] = {**wilson(sum(available), len(available)), "eligible_fce_episodes": len(values),
                                 "eligible_t_star_episodes": len(values),
                                 "unknown": len(values)-len(available),
                                 "gap_denominator": wilson(sum(available), len(gap))}
            matched = [(task, a) for task, a in values if type(a[arm]) is bool and type(a["I0"]) is bool]
            n = len(matched)
            paired_rates[cohort][arm] = {
                "arm": wilson(sum(a[arm] for _, a in matched), n),
                "I0": wilson(sum(a["I0"] for _, a in matched), n),
                "paired_excess": wilson(sum(a[arm] and not a["I0"] for _, a in matched), n),
                "task_ids": [task for task, _ in matched],
                "eligible_fce_episodes": len(values), "unknown_pairs": len(values)-n,
                "eligible_t_star_episodes": len(values),
            }
    budget_rows = [r for r in fce["episodes"] if r.get("budget_fce", r.get("fce")) is not None]
    positions = [r.get("budget_fce", r.get("fce")) for r in budget_rows]
    attributions = Counter(r["label"] for r in rows)
    coverage_cells = {}
    for cell in ("both", "M1_cell", "M2_cell", "neither", "incomplete"):
        values = [r for r in rows if r["cell"] == cell]
        categories = Counter(r["label"] for r in values)
        coverage_cells[cell] = {k: {**wilson(n, len(gap)), "denominator": "all frozen gap episodes"}
                               for k, n in sorted(categories.items())}
    knowledge_rows = [{"task_id": t, "coverage": coverage.get(t), "probe": probes.get(t)} for t in sorted(gap)]
    return {"subject": interventions.get("subject", "student"), "student": interventions.get("student"),
            "gap_size": len(gap), "regression_size": taxonomy.get("R", {}).get("size"),
            "taxonomy_all_failures": taxonomy["runs"]["student"]["shares"],
            "taxonomy_gap": shares([r for r in taxonomy_rows if r["task_id"] in gap]),
            "fce_positions": {"count": len(positions), "histogram": dict(sorted(Counter(positions).items())),
                              "relative_positions": [{"task_id": r["task_id"], "t": r["fce"],
                                "fraction_of_parent": r["fce"]/r["episode_length"] if r["episode_length"] else None}
                                for r in budget_rows]},
            "distance_diagnostics": [distance_diagnostics(fces.get(t, {"task_id": t})) for t in sorted(gap)],
            "turns_without_progress_definition": "Parent episode length minus t*: descriptive turns after the earliest "
                "sampled minimum D. No later probed state has a smaller successful teacher distance; "
                "unprobed/unsolved states do not establish turn-by-turn absence of progress.",
            "exclusions": dict(exclusions), "arm_rescue": rates,
            "paired_arm_rescue": paired_rates,
            "non_deterministic": {"count": len(non_deterministic), "task_ids": non_deterministic,
                                  "excluded": False},
            "decision_cells": {k: {**wilson(cells[k], len(gap)), "denominator": "all frozen gap episodes"}
                               for k in ("both", "M1_cell", "M2_cell", "neither", "incomplete")},
            "coverage_decision_cells": coverage_cells,
            "mechanism_attribution": {k: {**wilson(n, len(gap)), "denominator": "all frozen gap episodes"}
                                      for k, n in sorted(attributions.items())},
            "knowledge_evidence": knowledge_rows, "training_set_fit": probe_report.get("training_set_fit"),
            "episodes": rows, "arms_by_task": arms_by_task, "paired_excess_by_task": excess_by_task,
            "rescue_rule": "greedy succeeds AND at least 3 of 4 T=0.7 samples succeed",
            "paired_excess_rule": "arm rescues AND I0 does not rescue on the same t* state; both outcomes must be known",
            "decision_rule": "Section 6 uses paired excess rescues",
            "coverage_rule": "Task-type overlap alone is not needed-step coverage. No M1 attribution without explicit coverage evidence.",
            "limits": ["Descriptive engineering readout; no mechanism claim.",
                       "M1a/M1b use exact seed-state action agreement, not a general knowledge or causal test; absent probes remain unknown.",
                       "Typed action coverage does not establish matching preconditions. Dose controls remain required before interpretation.",
                       "Wilson intervals use task instances; groups are also reported and may contain dependent trials.",
                       "Missing/unknown continuations are excluded from available-arm denominators and explicitly retained.",
                       "Paired denominators include only states with complete arm and I0 outcomes; unknown I0 is never treated as failure."]}


def validate_intervention_artifacts(value):
    for entry in value["episodes"]:
        refuse_sealed(entry["task_id"])
        for arm_name, arm in entry.get("arms", {}).items():
            for sample in arm.get("samples", []):
                if "episode_path" not in sample:
                    continue
                path = Path(sample["episode_path"])
                ep = load_episode(path)
                if artifact_hashes(path.parent) != sample["episode_hashes"]:
                    raise ValueError("Intervention evidence changed")
                if (ep["task_id"] != entry["task_id"] or ep["branch"] != sample["branch"]
                        or outcome(ep, audit_states(path, ep)) != sample["success"]
                        or ep["branch"]["t"] != entry["t"] or ep["branch"]["arm"] != arm_name
                        or ep["branch"]["sample_index"] != sample["sample_index"]
                        or ep["branch"]["parent_episode"] != entry["parent_episode"]):
                    raise ValueError("Intervention summary differs from real branch")


def validate_fce_artifacts(path, value):
    for row in value["episodes"]:
        refuse_sealed(row["task_id"])
        for t, expected in row["probes"].items():
            probes = [completed_branch(Path(path).parent/"tasks"/digest(row["task_id"])[:16]/
                       f"probes/t{int(t):02d}/sample{i}") for i in range(2)]
            if any(p is None for p in probes) or any(p["success"] is True for p in probes) is not expected:
                raise ValueError("FCE probe summary disagrees with completed teacher samples")
        if row.get("budget_fce_status") == "fce_found":
            t = row["fce"]
            if row["probes"].get(str(t)) is not True or row["probes"].get(str(t+1)) is not False:
                raise ValueError("FCE lacks adjacent 1-to-0 takeover evidence")
        verified = {}
        for t, samples in row["probe_samples"].items():
            actual = []
            for index in range(2):
                receipt = completed_branch(Path(path).parent/"tasks"/digest(row["task_id"])[:16]/
                                           f"probes/t{int(t):02d}/sample{index}")
                actual.append(sample_distance(receipt or {"success": "unknown"}, int(t), index))
                if receipt:
                    episode = load_episode(receipt["episode_path"])
                    if read_json(Path(receipt["episode_path"]).parent/"probe_episode.json") != probe_episode(episode):
                        raise ValueError("Stable probe episode differs from execution evidence")
            if samples != actual:
                raise ValueError("Teacher steps-to-go differ from executed probe episodes")
            verified[int(t)] = actual
            distances = [s["teacher_steps_to_go"] for s in actual if s["success"] is True]
            if row["D_curve"].get(t) != (min(distances) if distances else None):
                raise ValueError("D curve differs from successful teacher samples")
        if row["status"] != "incomplete":
            def probe(t):
                samples = verified.get(t, [])
                if len(samples) != 2 or any(type(s["success"]) is not bool for s in samples):
                    raise IncompleteProbe("Missing teacher samples")
                return [s["teacher_steps_to_go"] for s in samples]
            expected = locate_closest_approach(row["episode_length"], probe)
            for key, expected_value in expected.items():
                # Disk JSON object keys are strings; pure locator uses integers.
                if row.get(key) != json.loads(json.dumps(expected_value)):
                    raise ValueError("Closest-approach summary differs from probe evidence: " + key)


def markdown(result):
    lines = ["# Mechanism readout", "", f"Subject: {result['subject']}; |G| = {result['gap_size']}.", "",
             "Rescue requires greedy success and at least 3/4 sampled successes.", "",
             "Each row compares the arm and I0 on the same completed t* states. Paired excess = arm rescues and I0 does not.", "",
             "| Arm | Raw arm rescues (95% Wilson) | I0 on same states (95% Wilson) | Paired excess (95% Wilson) | Unknown pairs |",
             "|---|---|---|---|---:|"]
    def rate_text(row):
        if not row["n"]:
            return "0/0 (n/a)"
        ci = "–".join(f"{v:.1%}" for v in row["ci95"])
        return f"{row['count']}/{row['n']} = {row['rate']:.1%} ({ci})"
    for arm in ARMS:
        row = result["paired_arm_rescue"]["primary"][arm]
        lines.append(f"| {arm} | {rate_text(row['arm'])} | {rate_text(row['I0'])} | "
                     f"{rate_text(row['paired_excess'])} | {row['unknown_pairs']} |")
    lines += ["", "Taxonomy on the frozen gap set:", "", "| Label | Primary count | Share of known gap failures |",
              "|---|---:|---:|"]
    for label in LABELS:
        row = result["taxonomy_gap"]["primary"][label]
        share = f"{row['share']:.1%}" if row["share"] is not None else "n/a"
        lines.append(f"| {label} | {row['count']} | {share} |")
    cells = result["decision_cells"]
    def cell(key, label):
        row = cells[key]
        ci = "–".join(f"{v:.1%}" for v in row["ci95"]) if row["n"] else "n/a"
        return f"{label}: {row['count']}/{row['n']} ({ci})"
    def coverage_cell(key, label):
        counts = result["coverage_decision_cells"][key]
        return cell(key, label) + ("; " + "; ".join(f"{k}: {v['count']}" for k, v in counts.items()) if counts else "")
    lines += ["", "Section 6: all rescues below are paired excess rescues over I0.", "",
        "| | I5 or I1 rescues | Neither rescues |", "|---|---|---|",
        "| I3 or I2 rescues | " + coverage_cell("both", "Both rescue signals") + " | " + cell("M2_cell", "M2 cell") + " |",
        "| Neither rescues | " + coverage_cell("M1_cell", "Experience rescue signal") + " | " + cell("neither", "I4-only: execution/capacity; else unresolved") + " |",
        "", "Budget FCE positions: `" + json.dumps(result["fce_positions"]["histogram"], sort_keys=True) + "`.",
        "", "Exclusions/pending: `" + json.dumps(result["exclusions"], sort_keys=True) + "`.",
        "", f"Non-deterministic I0 greedy replays retained: {result['non_deterministic']['count']}.",
        "", result["coverage_rule"], "", *result["limits"]]
    lines += ["", "Coverage and absorption (all gap episodes; navigation excluded from both gates):", "",
              "| Task | Coverage status | Same type | Any type | Absorbed | M1 subtype | Attribution |",
              "|---|---|---|---|---|---|---|"]
    attributed = {r["task_id"]: r for r in result["episodes"]}
    for row in result["knowledge_evidence"]:
        cov, probe = row["coverage"] or {}, row["probe"] or {}
        label = attributed.get(row["task_id"], {})
        lines.append("| " + " | ".join(str(x) if x is not None else "unknown" for x in (
            row["task_id"], cov.get("status", "missing"), cov.get("covered_same_task_type"),
            cov.get("covered_any_task_type", cov.get("covered")), probe.get("absorbed"),
            label.get("M1_subtype"), label.get("label", "excluded/pending"))) + " |")
    lines += ["", "Coverage-qualified attributions (all frozen gap episodes):", "",
              "| Attribution | Count / gap size (95% Wilson) |", "|---|---|"]
    for label, row in result["mechanism_attribution"].items():
        lines.append(f"| {label} | {rate_text(row)} |")
    if result["training_set_fit"]:
        fit = result["training_set_fit"]
        lines += ["", "Training-set fit: " + fit["scope"], "",
                  "| Task type | Matches | All states | Unknown | Agreement |", "|---|---:|---:|---:|---:|"]
        for kind, row in [("overall", fit["overall"]), *fit["per_task_type"].items()]:
            rate = f"{row['agreement']:.1%}" if row["agreement"] is not None else "unknown"
            lines.append(f"| {kind} | {row['matches']} | {row['n']} | {row['unknown']} | {rate} |")
    lines += ["", "Closest approach per gap episode (including exclusions/pending):", "",
              "| Task | Status | t* | D(0) | D(t*) | Turns remaining at t* | Turns after t* without progress (descriptive) | Budget FCE |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["distance_diagnostics"]:
        fields = ("task_id", "status", "t_star", "D_0", "D_t_star", "turns_remaining_at_t_star",
                  "turns_spent_after_t_star_without_progress", "budget_fce")
        lines.append("| " + " | ".join(str(row[k]) if row[k] is not None else "undefined" for k in fields) + " |")
    lines += ["", result["turns_without_progress_definition"], "", "D(t) curves (undefined = no successful sample):", ""]
    for row in result["distance_diagnostics"]:
        curve = {str(t): row["D_curve"][t] for t in sorted(row["D_curve"], key=int)}
        lines.append(f"- `{row['task_id']}`: `{json.dumps(curve)}`")
    if "paired_comparison" in result:
        lines += ["", "Student comparison of paired excess rescues (exact McNemar):", "", SERVING_NOISE_TEXT, "",
                  "| Arm | Paired states | First only | Second only | p |", "|---|---:|---:|---:|---:|"]
        for arm, comparison in result["paired_comparison"].items():
            lines.append(f"| {arm} | {comparison['paired_n']} | {comparison['first_only']} | "
                         f"{comparison['second_only']} | {comparison['exact_two_sided_p']:.3g} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--gap-set", required=True)
    parser.add_argument("--fce", required=True)
    parser.add_argument("--interventions", required=True)
    parser.add_argument("--coverage", help="coverage.py coverage.json; validated against executed evidence")
    parser.add_argument("--probes", help="probes.py probes.json; requires --coverage")
    parser.add_argument("--compare-interventions", help="Paired student/base arm comparison on the identical states")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    taxonomy, _ = gap_episodes(args.gap_set, config)
    fce, interventions = read_json(args.fce), read_json(args.interventions)
    if Path(fce["gap_path"]).resolve() != Path(args.gap_set).resolve() or Path(interventions["fce_path"]).resolve() != Path(args.fce).resolve():
        raise ValueError("Readout inputs refer to different populations/closest-approach states")
    validate_intervention_artifacts(interventions)
    validate_fce_artifacts(args.fce, fce)
    from analysis.mechanism.coverage import validate_coverage
    from analysis.mechanism.probes import validate_probes
    if args.probes and not args.coverage:
        parser.error("--probes requires --coverage")
    coverage = validate_coverage(args.coverage, config, fce_path=args.fce, gap_path=args.gap_set) if args.coverage else {}
    probes = validate_probes(args.probes, coverage, config) if args.probes else None
    if coverage and (interventions.get("subject", "student") != "student" or
                     interventions.get("student", {}).get("model") != coverage["student_binding"]["model"]):
        raise ValueError("Coverage/probes must describe the intervention student")
    result = build_readout(taxonomy, fce, interventions, coverage, probes)
    inputs = [args.gap_set, args.fce, args.interventions, *([args.coverage] if args.coverage else []),
              *([args.probes] if args.probes else [])]
    if args.compare_interventions:
        other = read_json(args.compare_interventions)
        validate_intervention_artifacts(other)
        if other["fce_path"] != interventions["fce_path"] or other["gap_path"] != interventions["gap_path"]:
            raise ValueError("Paired comparison requires identical gap and closest-approach states")
        compared = build_readout(taxonomy, fce, other)
        result["paired_comparison"] = {a: paired_mcnemar(
            {k: v[a] for k, v in result["paired_excess_by_task"].items()},
            {k: v[a] for k, v in compared["paired_excess_by_task"].items()}) for a in ARMS}
        inputs.append(args.compare_interventions)
    output = new_output(args.output, config=config, command="readout", inputs=inputs)
    write_json(output/"readout.json", result, exclusive=True)
    (output/"readout.md").write_text(markdown(result))
    print(json.dumps({"output": str(output), "gap_size": result["gap_size"]}))


if __name__ == "__main__":
    main()
