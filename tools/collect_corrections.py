"""P2/R2 teacher actions on student-visited contexts; supervise the final turn only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from teachers.stage import teacher_stage, count, require_completed

from data.manifest.build_r03_sets import dataset_entry, guarded_task, publish_dataset
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.storage import JsonlLedger, write_json
from teachers.capped import from_config, state_hash
from tools.rollout_pool import read_rollouts


def state_ref(state):
    return {k: state[k] for k in ("episode_id", "step", "state_hash")}


def select_r1_states(records, task_ids, cap=600):
    """Freeze the deterministic prefix before calling a teacher; disclose omissions."""
    by_task = {r["task_id"]: r for r in records}
    if len(by_task) != len(records):
        raise ValueError("R1 expects one p10 rollout per game")
    states = [s for t in task_ids if t in by_task for s in by_task[t]["states"]]
    selected, seen = [], set()
    for s in states:
        if s["state_hash"] not in seen:
            selected.append(state_ref(s))
            seen.add(s["state_hash"])
    return {"states": selected[:cap], "eligible_unique_states": len(selected),
            "omitted_by_cap": max(0, len(selected) - cap), "cap": cap,
            "selection_rule": "pool R1_P2 game order, then zero-based policy step; first unique 600",
            "missing_game_ids": [t for t in task_ids if t not in by_task]}


def resolve_selection(records, selected):
    lookup = {(r["episode_id"], s["step"]): (r, d, s)
              for r, d in records for s in r["states"]}
    resolved, seen = [], set()
    for ref in selected:
        if type(ref["step"]) is not int or ref["step"] < 0:
            raise ValueError("Step must be a nonnegative integer")
        key = (ref.get("episode_id", ref.get("episode")), ref["step"])
        if key not in lookup:
            raise ValueError("Selected state does not exist in rollouts")
        record, directory, state = lookup[key]
        if ref.get("state_hash", state["state_hash"]) != state["state_hash"]:
            raise ValueError("Selected state hash mismatch")
        if state["state_hash"] not in seen:
            resolved.append((record, directory, state))
            seen.add(state["state_hash"])
    return resolved


def correction_episode(original, state, result):
    # Preserve only the actually executed student prefix, never invent a branch.
    episode = deepcopy(original)
    step_index = state["step"]
    episode.update(episode_id="correction-" + state["state_hash"], complete=False,
                   status="unverified_teacher_label", success="unknown", terminated=False, truncated=False)
    episode["steps"] = deepcopy(original["steps"][:step_index])
    episode["steps"].append({"event_id": episode["episode_id"] + ":label",
        "action_raw": "Action: " + result.parsed_action, "action_parsed": result.parsed_action,
        "visible": deepcopy(state["visible_context"]), "reward": None,
        "terminated": False, "truncated": False, "environment_steps": 0,
        "action_outcome": "unexecuted_teacher_label", "success": "unknown",
        # Cache-access billing lives in collection/usage ledgers. Keep the label
        # artifact stable when a later request fails and this prefix is reused.
        "policy_metadata": {k: v for k, v in result.metadata.items()
                            if k not in {"cache_hit", "cache_accessed_at", "billed_usage"}}})
    for name in ("metrics", "policy_calls", "action_outcomes", "termination"):
        episode.pop(name, None)
    episode["policy_turns_attempted"] = step_index + 1
    return episode


@teacher_stage("output", name="collect_corrections")
def collect(records, selected, teacher, output, manifest, freeze, *, source="R1_P2"):
    output = Path(output)
    resolved = resolve_selection(records, selected)
    entries, planned, rejected = [], [], 0
    for record, directory, state in resolved:
        task = guarded_task(manifest, freeze, record["task_id"])
        if state_hash(state["history"], task, state["visible_context"]) != state["state_hash"]:
            raise ValueError("Visible state hash mismatch")
        audit = {"source": source, "parent_seed_ids": [], "parent_rollout_id": record["episode_id"],
                 **state_ref(state), "task_id": task["task_id"]}
        try:
            result = teacher.generate(state["history"], task, state["visible_context"])
        except Exception as exc:
            JsonlLedger(output / "collection.jsonl").append({**audit, "status": "stopped",
                                                           "error_type": type(exc).__name__})
            raise
        if not teacher.dry_run:
            require_completed(result)
        if teacher.dry_run:
            planned.append({**audit, **result.metadata})
            continue
        accepted = (result.parsed_action is not None
                    and result.parsed_action in state["visible_context"]["admissible_commands"])
        JsonlLedger(output / "collection.jsonl").append({**audit, "raw_output": result.raw_text,
            "teacher_metadata": result.metadata, "accepted": accepted,
            "reason": "unexecuted_teacher_label" if accepted else "unparseable_or_inadmissible_action",
            "verification": "unknown", "new_environment_steps": 0})
        if not accepted:
            rejected += 1
            count("rejections")
            continue
        original = json.loads((directory / "episode.json").read_text())
        episode = correction_episode(original, state, result)
        entry = dataset_entry(output, episode, source="teacher_action_at_student_visited_state",
                              supervised_turns=[state["step"]])
        entry.update(state_hash=state["state_hash"], parent_rollout_id=record["episode_id"],
                     source_episode_sha256=record["episode_sha256"], verification="unknown")
        entries.append(entry)
    if teacher.dry_run:
        return {"dry_run": True, "selected_unique_states": len(resolved),
                "would_call": sum(r["would_call"] for r in planned),
                "within_cap": all(r["within_cap"] for r in planned),
                "halted": any(r["halted"] for r in planned), "states": planned}
    return publish_dataset(output, entries, manifest, freeze, source=source,
        selected_unique_states=len(resolved), rejected_labels=rejected, rejections=rejected,
        transport_failures=0, verification="unknown",
        supervision="Only final teacher Action line and native turn end; all student prefix tokens masked",
        new_environment_steps=0, trusted_executed_trajectories=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--pool", default="data/manifest/r03_pool200.json")
    p.add_argument("--rollouts", nargs="+", required=True)
    p.add_argument("--selection", required=True, help="JSON list or object with states/union_states")
    p.add_argument("--source", choices=("R1_P2", "R2"), default="R1_P2")
    p.add_argument("--cap", type=int)
    p.add_argument("--cache", default="data/raw/r03_teacher_cache")
    p.add_argument("--usage-ledger", default="data/raw/r03_teacher_usage.jsonl")
    p.add_argument("--run-id", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    limit = 600 if args.source == "R1_P2" else 1500
    cap = args.cap if args.cap is not None else limit
    if not 0 <= cap <= limit:
        p.error("Cap exceeds preregistered collector ceiling")
    config = load_config(args.config)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    records = read_rollouts(args.rollouts, manifest, freeze, args.pool)
    pool = json.loads(Path(args.pool).read_text())
    allowed = set(pool["subsets"]["R1_P2" if args.source == "R1_P2" else "R2"])
    records = [(r, d) for r, d in records if r["task_id"] in allowed]
    selection = json.loads(Path(args.selection).read_text())
    selected = selection if isinstance(selection, list) else selection.get("states", selection.get("union_states"))
    if selected is None:
        p.error("Selection requires states or union_states")
    teacher = from_config(config, output=args.output, cache_dir=args.cache, usage_ledger=args.usage_ledger,
                          run_id=args.run_id, cap=cap, dry_run=args.dry_run)
    if not args.dry_run:
        config["r03_collector"] = {"run_id": args.run_id, "cap": cap, "source": args.source,
                                  "cache": args.cache, "usage_ledger": args.usage_ledger}
        write_provenance(args.output, command="r03_collect_corrections", config=config,
            inputs=[args.config, args.pool, args.selection, *args.rollouts,
                    config["paths"]["manifest"], config["paths"]["split_freeze"]])
    result = collect(records, selected, teacher, args.output, manifest, freeze, source=args.source)
    if args.dry_run:
        write_json(Path(args.output) / "dry_run.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"episodes", "states"}}, indent=2))


if __name__ == "__main__":
    main()
