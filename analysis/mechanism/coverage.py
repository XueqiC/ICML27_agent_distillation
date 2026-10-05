"""CPU-only needed-pattern coverage from executed t* takeovers and verified seeds."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from analysis.mechanism.common import (audit_states, completed_branch, gap_episodes, load_episode,
                                      outcome, read_json, workspace)
from analysis.mechanism.fce import sample_distance
from analysis.mechanism.goals import normal_type, parse_goal
from analysis.mechanism.policies import validate_continuation
from sed.config import load_config
from sed.manifest import load_frozen
from sed.react import action_outcome, build_messages
from sed.seed_evidence import checked_episode, load_verified_collection, verified_collection_inputs
from sed.storage import digest, file_hash, write_json
from sed.thinking import split_thinking
from sed.visibility import visible_observation

RULE = "Every non-navigation pattern occurs in at least one executed legal seed step; any task type decides coverage."


def action_pattern(action):
    """Public ALFWorld names use the goals.py type normalization, without instance IDs.

    These are typed action patterns, not a claim of equivalent preconditions or
    causal knowledge. Unknown commands stay unknown and cannot establish coverage.
    """
    def kind(entity):
        return normal_type(re.sub(r"\s+\d+$", "", entity))

    action = action or ""
    for regex, template in (
        (r"take (.+) from (.+)", "take {obj:%s} from {%s}"),
        (r"(?:move|put) (.+) (?:to|in/on) (.+)", "move {obj:%s} to {%s}"),
    ):
        match = re.fullmatch(regex, action)
        if match:
            return {"pattern": template % tuple(kind(x) for x in match.groups()), "navigation_only": False}
    match = re.fullmatch(r"(heat|cool|clean|slice) (.+) with (.+)", action)
    if match:
        return {"pattern": f"{match[1]} {{obj:{kind(match[2])}}} with {{{kind(match[3])}}}",
                "navigation_only": False}
    match = re.fullmatch(r"(go to|open|close|use|toggle|examine) (.+)", action)
    if match:
        return {"pattern": f"{match[1]} {{{kind(match[2])}}}",
                "navigation_only": match[1] in {"go to", "examine"}}
    if action in {"look", "inventory"}:
        return {"pattern": action, "navigation_only": True}
    return {"pattern": None, "navigation_only": False}


def seed_states(episode, *, episode_path=None, receipt_sha256=None):
    """Exact teacher-forced prefixes, using the same whitelist as train.dataset.

    Audit fields and state IDs are evidence only, never arguments to a policy.
    Keep *all* training turns for fit, including parser/inadmissible targets.
    """
    goal = parse_goal(episode["task_id"])
    if episode["split"] != "train" or not episode["task_id"].startswith("train/") or episode.get("sealed"):
        raise ValueError("Probe states must come from TRAIN seeds")
    if episode.get("schema_version") != 2:
        raise ValueError("ReAct v2 training seeds required")
    history, rows = [], []
    visible = visible_observation(episode["initial"]["visible"])
    for index, step in enumerate(episode["steps"], 1):
        reply, thinking = split_thinking(step["action_raw"])
        if thinking:
            raise ValueError("Teacher seed contains a hidden thought channel")
        parsed, status = action_outcome(reply, visible["admissible_commands"])
        if parsed != step["action_parsed"] or status != step["action_outcome"]:
            raise ValueError("Seed action differs from shared parser")
        messages = build_messages(history, {}, visible)
        rows.append({"state_id": digest([episode["episode_id"], index]),
            "seed_episode_id": episode["episode_id"], "task_id": episode["task_id"],
            "task_type": goal.task_type, "step": index, "prefix_length": index-1,
            "episode_path": str(episode_path) if episode_path else None, "receipt_sha256": receipt_sha256,
            "seed_action": parsed, "seed_action_outcome": status,
            "demonstrated": status == "legal" and step["environment_steps"] == 1,
            **action_pattern(parsed), "messages_sha256": digest(messages),
            "history": list(history), "visible": visible})
        history.append({"visible": visible, "reply": reply})
        visible = visible_observation(step["visible"])
    return rows


def state_evidence(row):
    return {k: v for k, v in row.items() if k not in {"history", "visible"}}


def load_training_states(verified, config):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, root = load_verified_collection(verified, manifest, freeze)
    states = []
    for receipt in collection["verified"]:
        ep = checked_episode(root, receipt)
        states.extend(seed_states(ep, episode_path=root/receipt["episode_dir"]/"episode.json",
                                  receipt_sha256=digest(receipt)))
    return states, {"path": str(Path(verified).resolve()), "manifest_sha256": file_hash(Path(verified)/"manifest.json"),
                    **{k: collection[k] for k in ("K", "trusted_trajectories", "unique_instances", "task_groups")}}, \
        verified_collection_inputs(verified, manifest, freeze)


def training_binding(population, verified, config):
    """CPU validation of the actual evaluated student's training collection."""
    bindings = []
    for row, _, _ in population:
        parent = read_json(Path(row["episode_path"]).parent/"provenance.json")["config"]
        gate = parent.get("lora_equivalence", {})
        model = parent["suite"]["student"]["model"]
        validate_continuation(dict(config, lora_equivalence=gate), {"model": model},
                              row["episode_path"], verified=verified)
        bindings.append({"model": model, "adapter": gate["identity"]["adapter"]})
    if not bindings or any(b != bindings[0] for b in bindings):
        raise ValueError("Coverage requires one evaluated student checkpoint")
    return bindings[0]


def shortest_takeover(fce_path, row, parent):
    """Read both t* receipts; select shortest real success, tie by sample index."""
    t = row["t_star"]
    candidates, samples = [], []
    for index in range(2):
        root = Path(fce_path).parent/"tasks"/digest(row["task_id"])[:16]/f"probes/t{t:02d}/sample{index}"
        receipt = completed_branch(root)
        if receipt is None:
            samples.append({"sample_index": index, "success": "unknown", "teacher_steps_to_go": None})
            continue
        ep = load_episode(receipt["episode_path"])
        branch = ep["branch"]
        if (ep["task_id"] != parent["task_id"] or branch["parent_episode"] != parent["episode_id"]
                or branch["parent_episode_sha256"] != digest(parent) or branch != receipt["branch"]
                or outcome(ep, audit_states(receipt["episode_path"], ep)) is not receipt["success"]):
            raise ValueError("Takeover differs from its real parent/evidence")
        sample = sample_distance(receipt, t, index)
        samples.append(sample)
        if receipt["success"] is True:
            candidates.append((sample["teacher_steps_to_go"], index, receipt, ep))
    expected = row.get("probe_samples", {}).get(str(t))
    if expected is not None and samples != expected:
        raise ValueError("t* probe summary differs from executed samples")
    if any(s["success"] == "unknown" for s in samples):
        return None, "incomplete_takeover_evidence", samples
    if not candidates:
        return None, "no_successful_takeover", samples
    distance, index, receipt, ep = min(candidates, key=lambda c: c[:2])
    if row.get("D_t_star") != distance:
        raise ValueError("Shortest successful takeover differs from D(t*)")
    selection = {"sample_index": index, "teacher_steps_to_go": distance,
                 "episode_path": receipt["episode_path"], "episode_sha256": file_hash(receipt["episode_path"]),
                 "receipt_sha256": digest(receipt), "tie_break": "lowest sample_index"}
    return (selection, ep["steps"][t:]), "ok", samples


def cover_steps(steps, t, task_type, states):
    index = defaultdict(list)
    for state in states:
        if state["demonstrated"] and state["pattern"] is not None:
            index[state["pattern"]].append(state_evidence(state))
    needed = []
    for offset, step in enumerate(steps, 1):
        pattern = action_pattern(step["action_parsed"])
        matches = index[pattern["pattern"]] if pattern["pattern"] else []
        same = [s for s in matches if s["task_type"] == task_type]
        preferred = same or matches
        needed.append({"step": t+offset, "takeover_step": offset, "action": step["action_parsed"],
                       "action_outcome": step["action_outcome"], **pattern,
                       "covered_same_task_type": bool(same), "covered_any_task_type": bool(matches),
                       "same_task_type": same, "any_task_type": matches,
                       "preferred_evidence": preferred[0] if preferred else None})
    relevant = [r for r in needed if not r["navigation_only"]]
    def covered(key):
        # No substantive step gives no evidence for M1 (avoid vacuous attribution).
        if not relevant:
            return None
        if any(r["pattern"] is not None and not r[key] for r in relevant):
            return False
        return None if any(r["pattern"] is None for r in relevant) else True
    any_type, same_type = covered("covered_any_task_type"), covered("covered_same_task_type")
    return {"needed_steps": needed, "non_navigation_steps": len(relevant),
            "covered": any_type, "covered_any_task_type": any_type, "covered_same_task_type": same_type,
            "status": ("covered" if any_type is True else "not_covered" if any_type is False else
                       "navigation_only" if not relevant else "unknown_pattern")}


def compute_coverage(*, fce_path, verified, config, gap_path=None):
    fce = read_json(fce_path)
    gap_path = gap_path or fce["gap_path"]
    if Path(fce["gap_path"]).resolve() != Path(gap_path).resolve():
        raise ValueError("Coverage/FCE gap populations differ")
    _, population = gap_episodes(gap_path, config)
    binding = training_binding(population, verified, config)
    states, collection, inputs = load_training_states(verified, config)
    fces = {r["task_id"]: r for r in fce["episodes"]}
    if len(fces) != len(fce["episodes"]) or not set(fces) <= {p[1]["task_id"] for p in population}:
        raise ValueError("Invalid FCE population")
    results = []
    for _, parent, _ in population:
        task_id = parent["task_id"]
        row = fces.get(task_id, {})
        entry = {"task_id": task_id, "task_type": parse_goal(task_id).task_type,
                 "t_star": row.get("t_star"), "parent_episode": parent["episode_id"],
                 "covered": None, "covered_same_task_type": None, "covered_any_task_type": None,
                 "needed_steps": [], "selected_takeover": None, "status": "no_t_star",
                 "fce_status": row.get("status", "missing_fce")}
        if row.get("status") == "closest_approach_found":
            if (row.get("parent_episode") != parent["episode_id"] or type(row.get("t_star")) is not int
                    or not 0 <= row["t_star"] < len(parent["steps"])):
                raise ValueError("Invalid closest-approach parent/state")
            chosen, status, samples = shortest_takeover(fce_path, row, parent)
            entry.update(status=status, takeover_samples=samples)
            if chosen:
                selection, steps = chosen
                entry.update(cover_steps(steps, row["t_star"], entry["task_type"], states), selected_takeover=selection)
        results.append(entry)
    return {"schema_version": 1, "purpose": "diagnosis", "rule": RULE,
            "pattern_definition": "verb + object type + receptacle/appliance type; instance numbers removed; put aliases move",
            "preference": "same task type evidence first; report both scopes; any task type is the coverage gate",
            "gap_path": str(Path(gap_path).resolve()), "gap_sha256": file_hash(gap_path),
            "fce_path": str(Path(fce_path).resolve()), "fce_sha256": file_hash(fce_path),
            "student_binding": binding, "training_collection": collection,
            "episodes": results, "status_counts": dict(Counter(e["status"] for e in results)),
            "environment_steps": 0, "teacher_calls": 0, "student_calls": 0, "training_exposure": 0}, inputs


def validate_coverage(path, config, *, fce_path=None, gap_path=None):
    value = read_json(path)
    if ((fce_path and Path(value["fce_path"]).resolve() != Path(fce_path).resolve())
            or (gap_path and Path(value["gap_path"]).resolve() != Path(gap_path).resolve())):
        raise ValueError("Coverage refers to different inputs")
    expected, _ = compute_coverage(fce_path=value["fce_path"], verified=value["training_collection"]["path"],
                                   config=config, gap_path=value["gap_path"])
    if expected != value:
        raise ValueError("Coverage differs from executed evidence")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fce", required=True)
    parser.add_argument("--gap-set")
    parser.add_argument("--verified", required=True)
    parser.add_argument("--output", required=True, help="Owned analysis directory; identical coverage can resume")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result, seed_inputs = compute_coverage(fce_path=args.fce, verified=args.verified, config=config, gap_path=args.gap_set)
    output = workspace(args.output, config=config, command="coverage", identity={"coverage_sha256": digest(result)},
                        inputs=[args.config, result["gap_path"], args.fce, *seed_inputs],
                        details={"student_binding": result["student_binding"], "training_collection": result["training_collection"]})
    if (output/"coverage.json").exists():
        if read_json(output/"coverage.json") != result:
            raise ValueError("Existing coverage changed")
    else:
        write_json(output/"coverage.json", result, exclusive=True)
    print(json.dumps({"output": str(output), "episodes": len(result["episodes"]), "status_counts": result["status_counts"]}))
    return result


if __name__ == "__main__":
    main()
