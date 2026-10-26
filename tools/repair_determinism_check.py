"""M0a: replay frozen training success episodes twice and test stop reasons.

Run from the worktree with python -m tools.repair_determinism_check. Task IDs
come from --episodes; the frozen manifest authorizes them before any reset.
"""

import argparse
import json
from pathlib import Path

from harness.alfworld_env import Environment
from sed.config import load_config
from sed.manifest import authorize_task, load_frozen, resolve_task, verify_task_files
from sed.provenance import write_provenance
from sed.storage import digest, file_hash, write_json
from sed.visibility import visible_observation

from repair.execute import execute_actions, task_fingerprint
from repair.ledger import BudgetLedger
from repair.minimality import same_execution, same_reset


def _reference_checks(result, episode):
    checks = []
    for index, (actual, expected) in enumerate(zip(
            [result.initial, *result.steps], [episode["initial"], *episode["steps"]])):
        values = {"visible": actual is not None and actual["visible"] == visible_observation(expected["visible"])}
        for key in ("reward", "terminated", "truncated", "audit_sha256"):
            values[key] = actual is not None and actual.get(key) == expected[key]
        if index:
            values["action_parsed"] = actual.get("action_parsed") == expected["action_parsed"]
            if "action_outcome" in expected:
                values["action_outcome"] = actual.get("action_outcome") == expected["action_outcome"]
        checks.append({"step": index, "checks": values, "passed": all(values.values())})
    return {"step_count": len(result.steps) == len(episode["steps"]), "transitions": checks,
            "passed": (result.initial is not None and len(result.steps) == len(episode["steps"])
                       and all(c["passed"] for c in checks))}


def check_episode(env_factory, episode, *, ledger=None):
    """The caller must authorize the frozen training task before calling this."""
    if (episode.get("split") != "train" or episode.get("sealed") or
            not episode["task_id"].startswith("train/")):
        raise ValueError("Determinism acceptance is restricted to training episodes")
    if episode.get("success") is not True or episode.get("complete") is not True:
        raise ValueError("A complete recorded success episode is required")
    actions = [s["action_raw"] if s.get("action_raw") is not None else s["action_parsed"]
               for s in episode["steps"]]
    limit = episode["max_episode_steps"]
    if type(limit) is not int or limit <= 0 or len(actions) > limit:
        raise ValueError("Recorded episode exceeds its step limit")

    def execute(sequence, max_steps):
        # Deliberately no cache: both repetitions must traverse the environment.
        return execute_actions(env_factory, episode["task_id"], sequence, max_steps,
                               ledger=ledger, interaction_kind="task_validation_steps")

    first, second = execute(actions, limit), execute(actions, limit)
    failed = execute(actions[:-1], limit)
    limited = execute(actions, max(0, len(actions) - 1))
    expected_fingerprint = task_fingerprint(first.environment_version, episode["game_sha256"],
                                           episode["initial"]["visible"], episode["seed"])
    reference = [_reference_checks(r, episode) for r in (first, second)]
    checks = {
        "game_sha256_matches_recorded": all(r.game_sha256 == episode["game_sha256"] for r in (first, second)),
        "fingerprint_matches_recorded_initial": all(r.fingerprint == expected_fingerprint for r in (first, second)),
        "seed_matches_recorded": all(r.env_seed == episode["seed"] for r in (first, second)),
        "same_reset": same_reset(first, second),
        "same_visible_observations": ([s["visible"] for s in first.steps] == [s["visible"] for s in second.steps]),
        "identical_execution": same_execution(first, second),
        "recorded_transitions_reproduced": all(r["passed"] for r in reference),
        "success_reproduced_twice": all(r.success is True and r.termination_reason == "SUCCESS" for r in (first, second)),
        "truncated_actions_fail": failed.termination_reason == "FAILED" and failed.success is False,
        "smaller_limit_stops": limited.termination_reason == "STEP_LIMIT" and limited.success is False,
        "three_reasons_distinct": {first.termination_reason, failed.termination_reason, limited.termination_reason}
                                  == {"SUCCESS", "FAILED", "STEP_LIMIT"},
        "diagnostics_same_initial_state": same_reset(first, failed) and same_reset(first, limited),
        "real_replays_no_cache": not any(r.cache_hit for r in (first, second, failed, limited)),
    }
    if "fingerprint" in episode:
        checks["recorded_fingerprint_matches"] = episode["fingerprint"] == expected_fingerprint
    return {"task_id": episode["task_id"], "passed": all(checks.values()), "checks": checks,
            "reference_checks": reference,
            "executions": {"first": first.to_json(), "second": second.to_json(),
                           "failed": failed.to_json(), "step_limit": limited.to_json()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    artifacts = output.parent / (output.stem + "_artifacts")
    if output.exists() or artifacts.exists():
        parser.error("Choose a fresh output path to preserve existing evidence")
    ledger = BudgetLedger(artifacts / "budget.jsonl")
    report = {"schema_version": 1, "milestone": "M0a", "passed": False, "episodes": [],
              "artifacts": str(artifacts), "teacher_calls": 0, "model_execution": False}
    try:
        config = load_config(args.config)
        config["environment"]["data_root"] = args.data_root
        manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
        repair_sources = sorted((Path(__file__).resolve().parents[1] / "repair").glob("*.py"))
        write_provenance(artifacts, command="repair_determinism_check", config=config,
            inputs=[args.config, config["paths"]["manifest"], config["paths"]["split_freeze"],
                    *args.episodes, *repair_sources],
            details={"teacher_calls": 0, "model_execution": False,
                     "student_revision_configured": config["models"]["student"]["revision"],
                     "teacher_alias_configured": config["models"]["teacher"]["provider_model_id"],
                     "teacher_returned_model_id": None, "teacher_timestamp": None,
                     "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze),
                     "repair_code_sha256": {p.name: file_hash(p) for p in repair_sources}})
        ids, groups = [], []
        for path in args.episodes:
            entry = {"episode_path": path, "passed": False}
            try:
                episode = json.loads(Path(path).read_text())
                task = resolve_task(manifest, freeze, episode["task_id"])
                authorize_task(task, "generation")
                if task["task_id"] in ids:
                    raise ValueError("Duplicate task in acceptance list")
                if (episode["seed"] != config["environment"]["seed"] or
                        episode["max_episode_steps"] != config["environment"]["max_episode_steps"]):
                    raise ValueError("Recorded replay configuration differs")
                verify_task_files(args.data_root, task)
                if episode["game_sha256"] != task["game_sha256"]:
                    raise ValueError("Recorded game differs from frozen manifest")
                ids.append(task["task_id"])
                groups.append(task["group_id"])
                def env_factory():
                    return Environment(args.data_root, split="train", seed=episode["seed"],
                                       max_episode_steps=episode["max_episode_steps"])
                entry.update(check_episode(env_factory, episode, ledger=ledger))
            except Exception as exc:
                entry["error_type"] = type(exc).__name__
                ledger.append("rejected_candidate", reason="determinism_input_or_check_error",
                              details={"episode_path": path, "error_type": type(exc).__name__})
            report["episodes"].append(entry)
        report["unique_instances"] = len(set(ids))
        report["task_groups"] = len(set(groups))
        report["checks"] = {"at_least_five_training_instances": len(set(ids)) >= 5,
                            "all_episodes_passed": all(e["passed"] for e in report["episodes"])}
        report["passed"] = all(report["checks"].values())
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        ledger.append("failure", reason="determinism_setup_error", details={"error_type": type(exc).__name__})
    report["budget"] = ledger.summary()
    write_json(output, report, exclusive=True)
    print(json.dumps({"passed": report["passed"], "output": str(output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
