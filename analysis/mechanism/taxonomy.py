"""Automatic, auditable section-2 / Amendment 1 labels; no model calls or environment steps."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import re

from analysis.mechanism.common import (audit_states, frozen_episode, load_episode, new_output,
                                      outcome, output_guard, read_json)
from analysis.mechanism.goals import Facts, normal_type, parse_goal
from sed.config import load_config
from sed.manifest import authorize_task, load_frozen, resolve_task
from sed.storage import file_hash, write_json

LABELS = ("F-format", "F-inadmissible", "F-goal", "F-search", "F-precondition", "F-loop", "F-cap")


def observed_text(text):
    # The task instruction mentions the target by construction; it is not a sighting.
    return re.sub(r"Your task is to:[^\n]*", "", text, flags=re.I)


def action_parts(action):
    action = action or ""
    patterns = (
        (r"(take) (.+) from (.+)",), (r"(move|put) (.+) (?:to|in/on) (.+)",),
        (r"(heat|cool|clean|slice) (.+) with (.+)",), (r"(use|toggle|examine) (.+)",))
    for (pattern,) in patterns:
        m = re.fullmatch(pattern, action)
        if m:
            return m[1], m[2], m[3] if len(m.groups()) == 3 else None
    return action.split(" ", 1)[0], None, None


def entity_type(name):
    return normal_type(re.sub(r"\s+\d+$", "", name or ""))


def classify(episode, audits):
    """Indices are 1-based actions; 0 is reset. Only known failures get labels."""
    from analysis.mechanism.common import authorize_episode
    authorize_episode(episode)
    if len(audits) != len(episode["steps"]) + 1:
        raise ValueError("One audit state per transition and reset is required")
    goal = parse_goal(episode["task_id"])
    success = outcome(episode, audits)
    steps = episode["steps"]
    evidence = {name: [] for name in LABELS}
    parser_indices = [i for i, s in enumerate(steps, 1) if s.get("action_outcome") == "parser_error"]
    inadmissible_indices = [i for i, s in enumerate(steps, 1) if s.get("action_outcome") == "inadmissible"]
    base = {"task_id": episode["task_id"], "episode_id": episode["episode_id"],
            "success": success, "steps": len(steps), "goal": asdict(goal),
            "seed": episode.get("seed"), "game_sha256": episode.get("game_sha256"),
            "max_episode_steps": episode.get("max_episode_steps"),
            "parser_errors": len(parser_indices), "inadmissible": len(inadmissible_indices)}
    if success is not False:
        return dict(base, primary=None, labels=[], triggers={}, evidence={},
                    classification_status="success" if success is True else "unknown")

    def add(label, indices, rule, **details):
        evidence[label].append({"steps": list(indices), "rule": rule, **details})

    tail = []
    for i in range(len(steps), 0, -1):
        if steps[i-1].get("action_outcome") != "parser_error":
            break
        tail.append(i)
    if len(tail) >= 3:
        add("F-format", sorted(tail), "ends_with_at_least_3_parser_errors")
    if steps and len(parser_indices) / len(steps) >= .25:
        add("F-format", parser_indices, "parser_fraction_at_least_0.25", fraction=len(parser_indices)/len(steps))
    if steps and len(inadmissible_indices) / len(steps) >= .25:
        add("F-inadmissible", inadmissible_indices, "inadmissible_fraction_at_least_0.25",
            fraction=len(inadmissible_indices)/len(steps))

    facts = [Facts(a["internal_facts"]) for a in audits]
    goal_type = goal.base_object
    target = normal_type(goal.target_receptacle)
    held_steps = defaultdict(list)
    for i, state in enumerate(facts):
        for obj in state.held():
            if state.types.get(obj) == goal_type:
                held_steps[obj].append(i)
    placements = {"move", "put"}
    transforms = {"heat": ("ishot", "microwave"), "cool": ("iscool", "fridge"),
                  "clean": ("isclean", "sinkbasin"), "slice": ("issliced", "knife")}
    allowed_destinations = {target}
    if goal.movable_receptacle:
        allowed_destinations.add(normal_type(goal.movable_receptacle))
    allowed_destinations.update(tool for verb, (pred, tool) in transforms.items()
                                if verb != "slice" and pred in goal.required)
    for i, step in enumerate(steps, 1):
        before, after = facts[i-1:i+1]
        verb, obj, receptacle = action_parts(step.get("action_parsed"))
        obj_type, recep_type = entity_type(obj), entity_type(receptacle)
        known_types = set(before.types.values()) | set(after.types.values())
        relevant = before.evidence({"holds", "objecttype", "receptacletype"})
        # Amendment 1: only executed placements of the goal object and
        # heat/cool/clean/use commitments count. Failed attempts can still
        # supply precondition evidence below.
        legal = step.get("action_outcome") == "legal"
        if legal and verb in {"heat", "cool", "clean"} and obj_type in known_types and obj_type != goal_type:
            add("F-goal", [i], "manipulated_object_type_mismatch", action=step["action_parsed"],
                actual_type=obj_type, expected_types=[goal_type],
                facts=[f for f in relevant if f"{obj_type}type:" in f.lower()])
        if (legal and verb in placements and obj_type == goal_type
                and recep_type in known_types and recep_type not in allowed_destinations):
            add("F-goal", [i], "placement_receptacle_type_mismatch", action=step["action_parsed"],
                actual_type=recep_type, expected_type=target, allowed_types=sorted(allowed_destinations),
                facts=after.evidence({"inreceptacle"}) if after.get("inreceptacle") != before.get("inreceptacle") else
                [f for f in relevant if f"{recep_type}type:" in f.lower()])
        use_type = target if goal.task_type.startswith("look_at_obj") else goal_type
        if legal and verb in {"use", "toggle"} and obj_type in known_types and obj_type != use_type:
            add("F-goal", [i], "lamp_type_mismatch" if goal.task_type.startswith("look_at_obj") else "used_object_type_mismatch",
                action=step["action_parsed"], actual_type=obj_type, expected_type=use_type,
                facts=[f for f in relevant if f"{obj_type}type:" in f.lower()])
        held = {o for o in before.held() if before.types.get(o) == goal_type}
        if verb in placements and obj_type == goal_type:
            missing = {o: [p for p in goal.required if not before.has(p, o)] for o in held}
            if any(missing.values()):
                add("F-precondition", [i], "put_before_required_transformation", missing=missing,
                    action=step["action_parsed"], facts=before.evidence({"holds", *goal.required}))
        if verb in transforms and obj_type == goal_type and held_steps:
            pred, tool = transforms[verb]
            if not held or recep_type not in ({"knife", "butterknife"} if verb == "slice" else {tool}):
                add("F-precondition", [i], "transform_without_holding_or_at_wrong_receptacle",
                    action=step["action_parsed"], required_tool=tool, held=sorted(held),
                    facts=before.evidence({"holds", pred}))
        if goal.task_type.startswith("look_at_obj") and verb in {"use", "toggle"} and obj_type == target and held_steps and not held:
            add("F-precondition", [i], "lamp_used_without_goal_object_held",
                action=step["action_parsed"], facts=before.evidence({"holds", "istoggled"}))

    observations = [episode["initial"]["visible"]["observation"], *[s["visible"]["observation"] for s in steps]]
    sightings = [i for i, text in enumerate(observations)
                 if re.search(r"\b" + re.escape(goal_type) + r"(?:sliced)?\b", observed_text(text), re.I)]
    if not sightings:
        add("F-search", range(len(observations)), "no_goal_type_in_observation_text",
            goal_type=goal_type, instruction_excluded=True, observations=observations)
    if held_steps and goal.required:
        final = facts[-1]
        if not any(all(final.has(p, obj) for p in goal.required) for obj in held_steps):
            add("F-precondition", [len(steps)], "held_goal_but_required_transformation_missing_at_end",
                held_at=dict(held_steps), required=goal.required,
                facts=final.evidence({"holds", "inreceptacle", *goal.required}))
    if held_steps and goal.task_type.startswith("look_at_obj"):
        final = facts[-1]
        lamps = [args[0] for args in final.get("istoggled") | final.get("toggled")
                 if final.types.get(args[0]) == target]
        held = [o for o in final.held() if final.types.get(o) == goal_type]
        if not lamps or not held:
            add("F-precondition", [len(steps)], "look_requires_held_object_and_toggled_goal_lamp",
                held_at=dict(held_steps), toggled_goal_lamps=lamps,
                facts=final.evidence({"holds", "istoggled", "toggled"}))

    pairs = [(s.get("action_parsed"), s["visible"]["observation"]) for s in steps]
    occurrences = defaultdict(list)
    for i, pair in enumerate(pairs, 1):
        occurrences[pair].append(i)
    for pair, indices in occurrences.items():
        if len(indices) >= 3:
            add("F-loop", indices, "same_action_observation_pair_at_least_3", action=pair[0], observation=pair[1])
    for length in range(2, 5):
        for start in range(len(pairs) - 3*length + 1):
            cycle = pairs[start:start+length]
            if pairs[start:start+3*length] == cycle*3:
                add("F-loop", range(start+1, start+3*length+1), "cycle_repeats_at_least_3", cycle_length=length, cycle=cycle)
    if not any(evidence.values()) and len(steps) >= 40 and episode.get("truncated"):
        add("F-cap", [40], "no_other_rule_and_40_turn_limit")
    labels = [k for k in LABELS if evidence[k]]
    return dict(base, primary=labels[0] if labels else None, labels=labels,
                classification_status="labelled" if labels else "unclassified_failure",
                triggers={k: sorted({i for e in evidence[k] for i in e["steps"]}) for k in labels},
                evidence={k: evidence[k] for k in labels})


def shares(rows):
    failures = [r for r in rows if r["success"] is False]
    n = len(failures)
    counts = Counter(r["primary"] for r in failures)
    all_counts = Counter(k for r in failures for k in r["labels"])
    turns = sum(r["steps"] for r in rows)
    return {"episodes": len(rows), "successes": sum(r["success"] is True for r in rows),
            "failures": n, "unknown": sum(r["success"] == "unknown" for r in rows),
            "unique_instances": len({r["task_id"] for r in rows}),
            "task_groups": len({r["task_id"].rsplit("/", 1)[0] for r in rows}),
            "denominator": "known failed episodes", "unclassified_failures": counts[None],
            "primary": {k: {"count": counts[k], "share": counts[k]/n if n else None} for k in LABELS},
            "all_matching": {k: {"count": all_counts[k], "share": all_counts[k]/n if n else None} for k in LABELS},
            "parser_error_rate": sum(r["parser_errors"] for r in rows)/turns if turns else None}


def classify_run(path, config):
    path = Path(path).resolve()
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    listing = read_json(path / "tasks.json") if (path / "tasks.json").exists() else None
    if listing:
        from eval.run_suite import frozen_tasks
        frozen_tasks(path / "tasks.json", manifest, freeze)
        for task_id in listing["task_ids"]:
            authorize_task(resolve_task(manifest, freeze, task_id), "diagnosis")
    paths = sorted((path / "episodes").glob("*/episode.json"))
    if not paths:
        raise ValueError("Expected a single evaluation directory with episodes/")
    rows = []
    for p in paths:
        episode, _ = frozen_episode(p, config, frozen=(manifest, freeze))
        if "initial" not in episode:
            continue  # Retained explicitly in missing/reset-incomplete task IDs below.
        row = classify(episode, audit_states(p, episode))
        row.update(episode_path=str(p), episode_sha256=file_hash(p))
        # The snapshot/hash must describe the same atomically published episode.
        if read_json(p) != episode:
            raise ValueError("Episode changed during snapshot; retry into a fresh output")
        rows.append(row)
    ids = [r["task_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Run contains duplicate tasks")
    return {"path": str(path), "episodes": rows, "shares": shares(rows),
            "missing_or_reset_incomplete": sorted(set(listing["task_ids"]) - set(ids)) if listing else [],
            "snapshot_complete": bool(listing and set(ids) == set(listing["task_ids"]) and
                                      all(r["success"] != "unknown" for r in rows))}


def comparison(student, reference, name):
    other = {r["task_id"]: r for r in reference["episodes"]}
    matched = [r for r in student["episodes"] if r["task_id"] in other]
    for row in matched:
        if any(row.get(k) != other[row["task_id"]].get(k) for k in ("seed", "game_sha256", "max_episode_steps")):
            raise ValueError("Comparison episodes differ in environment seed, game, or cap")
    ids = [r["task_id"] for r in matched if r["success"] is False and other[r["task_id"]]["success"] is True]
    return {"reference": name, "task_ids": sorted(ids), "size": len(ids), "paired_instances": len(matched),
            "unknown_pairs": sum(r["success"] == "unknown" or other[r["task_id"]]["success"] == "unknown" for r in matched),
            "unpaired_student_tasks": sorted(set(r["task_id"] for r in student["episodes"]) - set(other))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Student evaluation directory")
    parser.add_argument("--compare", help="Teacher or base evaluation directory")
    parser.add_argument("--compare-role", choices=["teacher", "base"], help="Inferred from suite provenance if omitted")
    parser.add_argument("--teacher-run")
    parser.add_argument("--base-run")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", required=True, help="Fresh analysis directory")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    references = {"teacher": args.teacher_run, "base": args.base_run}
    if args.compare:
        provenance = read_json(Path(args.compare) / "provenance.json")
        role = args.compare_role or ("teacher" if provenance["config"].get("suite", {}).get("policy") == "teacher" else "base")
        if references[role]:
            parser.error("Duplicate comparison role")
        references[role] = args.compare
    output_guard(args.output, inputs=[args.run, *[p for p in references.values() if p]])
    runs = {"student": classify_run(args.run, config)}
    for name, path in references.items():
        if path:
            runs[name] = classify_run(path, config)
    result = {"schema_version": 1, "purpose": "diagnosis", "runs": runs,
              "indexing": "reset=0; actions=1..N", "plan_sha256": file_hash("docs/MECHANISM_PLAN.md")}
    for key, role in (("G", "teacher"), ("R", "base")):
        result[key] = comparison(runs["student"], runs[role], role) if role in runs else {
            "reference": role, "task_ids": [], "size": None, "status": "reference_not_supplied"}
    paths = [row["episode_path"] for run in runs.values() for row in run["episodes"] if row["success"] != "unknown"]
    snapshots = {row["episode_path"]: row["episode_sha256"] for run in runs.values() for row in run["episodes"]}
    output = new_output(args.output, config=config, command="taxonomy", inputs=[args.config, *paths],
                        details={"episode_snapshot_sha256": snapshots,
                                 "unknown_episodes": "Per-episode read snapshots; live source may advance after capture"})
    write_json(output / "taxonomy.json", result, exclusive=True)
    samples = {}
    for name, run in runs.items():
        rng = random.Random(0)
        samples[name] = {label: rng.sample([r for r in run["episodes"] if label in r["labels"]],
                            min(10, sum(label in r["labels"] for r in run["episodes"]))) for label in LABELS}
        for label, episodes in samples[name].items():
            enriched = []
            for row in episodes:
                ep = load_episode(row["episode_path"])
                enriched.append(dict(row, review_steps=[
                    {"step": i, "action": ep["steps"][i-1]["action_parsed"] if i else None,
                     "outcome": ep["steps"][i-1]["action_outcome"] if i else "reset",
                     "observation": (ep["steps"][i-1] if i else ep["initial"])["visible"]["observation"]}
                    for i in row["triggers"][label]]))
            samples[name][label] = enriched
    write_json(output / "review_samples.json", samples, exclusive=True)
    lines = ["# Taxonomy sanity check", "", "Automatic labels; review examples are in review_samples.json.", "",
             "| Run | Known failures | " + " | ".join(LABELS) + " |", "|---|---:|" + "---:|"*len(LABELS)]
    for name, run in runs.items():
        summary = run["shares"]
        lines.append(f"| {name} | {summary['failures']} | " + " | ".join(
            f"{summary['primary'][k]['count']} ({summary['primary'][k]['share']:.1%})" if summary['failures'] else "0 (n/a)"
            for k in LABELS) + " |")
    (output / "shares.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "shares": {k: v["shares"] for k, v in runs.items()},
                      "G_size": result["G"]["size"], "R_size": result["R"]["size"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
