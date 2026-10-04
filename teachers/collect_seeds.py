"""Collect real, verified TRAIN seeds. CLI makes live calls only when explicitly run."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from sed.config import load_config
from sed.manifest import load_frozen, resolve_task, verify_task_files
from sed.provenance import write_provenance, utc_now, code_hashes, versions, safe_config
from sed.runner import _environment, run_episode
from sed.seed_evidence import artifact_hashes, environment_won
from sed.seed_selection import validate_selection
from sed.storage import digest, locked, write_json
from teachers.openai_teacher import Teacher


def usage_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    ends = {r["call_id"] for r in rows if r["status"] != "started"}
    finished = [r for r in rows if r["status"] != "started"]
    interrupted = [r for r in rows if r["status"] == "started" and r["call_id"] not in ends]
    return {"known_billed_tokens": {key: sum((r.get("usage") or {}).get(key, 0) or 0
                                            for r in finished)
                                    for key in ("input_tokens", "output_tokens", "reasoning_tokens")},
            "unknown_usage_calls": sum(r.get("usage") is None for r in finished) + len(interrupted),
            "interrupted_requests": len(interrupted), "ledger_rows": len(rows)}


def collect(selection, manifest, freeze, config, output, *, attempts=3,
            env_factory=None, teacher_factory=None, input_paths=()):
    validate_selection(selection, manifest, freeze)
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be positive")
    config = deepcopy(config)
    output = Path(output)
    env_factory = env_factory or (lambda task: _environment(config, task))
    def default_teacher(task, cache, ledger):
        spec = config["models"]["teacher"]
        return Teacher(cache_dir=cache, usage_ledger=ledger, run_id="pending",
                       provider_model_id=spec["provider_model_id"], sampling=spec["sampling"],
                       max_attempts=spec["max_attempts"], timeout_seconds=spec["timeout_seconds"],
                       secrets_file=config["paths"]["secrets_file"])
    teacher_factory = teacher_factory or default_teacher
    identity = {"selection": selection, "config": safe_config(config), "attempts_per_candidate": attempts,
                "code_sha256": code_hashes(Path(__file__).resolve().parents[1]),
                "dependencies": versions()}
    with locked(output / ".collection.lock"):
        identity_path = output / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Resume identity changed (selection/config/code/dependencies)")
        else:
            write_json(identity_path, identity, exclusive=True)
        if not (output / "provenance.json").exists():
            write_provenance(output, command="collect_seeds", config=config, inputs=input_paths,
                             details={"selection_sha256": digest(selection), "attempts": attempts})
        write_json(output / "selection.json", selection)
        verified, receipts = [], []

        def publish(status):
            value = {"schema_version": 1, "status": status, "K": selection["K"],
                     "K_counts": "trusted_executed_trajectories", "identity_sha256": digest(identity),
                     "selection_sha256": digest(selection), "trusted_trajectories": len(verified),
                     "unique_instances": len({r["task_id"] for r in verified}),
                     "task_groups": len({r["group_id"] for r in verified}),
                     "attempted_unique_instances": len({r["task_id"] for r in receipts}),
                     "attempted_task_groups": len({r["group_id"] for r in receipts}),
                     "attempts": receipts, "verified": verified,
                     "parent_seed_ids": [], "source": "external_frozen_train_pool",
                     "training_exposure": 0}
            write_json(output / "manifest.json", value)
            return value

        for kind in selection["type_order"]:
            accepted = 0
            pending = [(r, r["task_id"]) for r in selection["selected"][kind]]
            replacements = iter(selection["replacements"][kind])
            while pending:
                candidate, selection_parent = pending.pop(0)
                task = resolve_task(manifest, freeze, candidate["task_id"])
                for number in range(1, attempts + 1):
                    attempt_dir = output / "attempts" / (digest(task["task_id"])[:20] + f"-{number}")
                    episode_dir = attempt_dir / "episode"
                    receipt_path = attempt_dir / "receipt.json"
                    if receipt_path.exists():
                        receipt = json.loads(receipt_path.read_text())
                        if artifact_hashes(episode_dir) != receipt["episode_hashes"]:
                            raise ValueError("Attempt evidence changed on resume")
                        if artifact_hashes(attempt_dir / "teacher") != receipt["teacher_hashes"]:
                            raise ValueError("Teacher evidence changed on resume")
                        existing = episode_dir / "episode.json"
                        episode = json.loads(existing.read_text()) if existing.exists() else {}
                        if (receipt["task_id"] != task["task_id"] or receipt["attempt"] != number
                                or receipt["selection_parent_task_id"] != selection_parent
                                or receipt["won"] != environment_won(episode_dir, episode)):
                            raise ValueError("Attempt receipt disagrees with environment evidence")
                    else:
                        attempt_dir.mkdir(parents=True, exist_ok=True)
                        if not (attempt_dir / "started.json").exists():
                            write_json(attempt_dir / "started.json", {"timestamp": utc_now(),
                                       "task_id": task["task_id"], "attempt": number}, exclusive=True)
                        if not episode_dir.exists():
                            verify_task_files(config["environment"]["data_root"], task)
                            # A separate cache per episode attempt permits fresh samples after
                            # failure, while durable completed attempts are reused on resume.
                            policy = teacher_factory(task, attempt_dir / "teacher/cache",
                                                     attempt_dir / "teacher/usage.jsonl")
                            env = env_factory(task)
                            try:
                                run_episode(env, policy, task, config, episode_dir,
                                            policy_condition="seed_teacher_react", purpose="generation",
                                            input_paths=input_paths)
                            except RuntimeError:
                                if not (episode_dir / "episode.json").exists():
                                    raise
                        episode_path = episode_dir / "episode.json"
                        episode = json.loads(episode_path.read_text()) if episode_path.exists() else {}
                        won = environment_won(episode_dir, episode)
                        receipt = {"task_id": task["task_id"], "group_id": task["group_id"],
                                   "task_type": kind, "attempt": number,
                                   "parent_task_id": task["task_id"], "parent_seed_ids": [],
                                   "selection_parent_task_id": selection_parent,
                                   "source": "real_environment_teacher_rollout",
                                   "won": won, "success": episode.get("success", "unknown"),
                                   "status": ("interrupted" if episode.get("status", "started") == "started"
                                              else episode["status"]),
                                   "episode_dir": str(episode_dir.relative_to(output)),
                                   "episode_hashes": artifact_hashes(episode_dir),
                                   "teacher_hashes": artifact_hashes(attempt_dir / "teacher"),
                                   "usage": usage_summary(attempt_dir / "teacher/usage.jsonl"),
                                   "metrics": episode.get("metrics"),
                                   "rejection": None if won else "no_verified_environment_win"}
                        write_json(receipt_path, receipt, exclusive=True)
                    receipts.append(receipt)
                    if receipt["won"]:
                        verified.append(receipt)
                        accepted += 1
                    publish("collecting")
                    if receipt["won"]:
                        break
                if not receipt["won"]:
                    replacement = next(replacements, None)
                    if replacement is None:
                        return publish("exhausted")
                    pending.insert(0, (replacement, selection_parent))
            if accepted != selection["quotas"][kind]:
                return publish("exhausted")
        return publish("complete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    selection = json.loads(Path(args.selection).read_text())
    output = args.output or str(Path(config["paths"]["verified"]) /
                               f"seeds_k{selection['K']}_s{selection['seed']}")
    result = collect(selection, manifest, freeze, config, output, attempts=args.attempts,
                     input_paths=[args.config, args.selection, config["paths"]["manifest"],
                                  config["paths"]["split_freeze"]])
    print(json.dumps({key: result[key] for key in ("status", "trusted_trajectories", "task_groups")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
