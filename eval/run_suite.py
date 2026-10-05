"""Evaluate a frozen task list in isolated worker processes, with no online updates."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import json
import multiprocessing
from pathlib import Path
import time

from sed.config import load_config
from sed.dev_subset import build_dev40, build_dev140
from sed.manifest import authorize_task, load_frozen, resolve_task, verify_task_files
from sed.provenance import write_provenance
from sed.runner import _environment, run_episode
from sed.storage import digest, write_json
from students.vllm_student import Student
from teachers.openai_teacher import Teacher, rate_limit_options


def frozen_tasks(path, manifest, freeze, *, sealed_final=False, policy="student"):
    listing = json.loads(Path(path).read_text())
    ids = listing["task_ids"]
    if not ids or len(set(ids)) != len(ids) or listing["list_sha256"] != digest(ids):
        raise ValueError("Task list is empty, duplicated, or its hash differs")
    if listing["manifest_sha256"] != digest(manifest) or listing["split_freeze_sha256"] != digest(freeze):
        raise ValueError("Task list belongs to a different manifest/freeze")
    tasks = [resolve_task(manifest, freeze, task_id) for task_id in ids]
    for task in tasks:
        authorize_task(task, "evaluation")
        if task["sealed"] or task["split"] == "test":
            if not sealed_final:
                raise ValueError("Sealed tasks require explicit --sealed-final")
            if policy == "teacher":
                raise ValueError("Teacher generation remains forbidden on sealed tasks")
    if listing.get("name") == "dev40" and listing != build_dev40(manifest, freeze):
        raise ValueError("dev40 differs from its fixed seed-0 selection rule")
    if listing.get("name") == "dev140" and listing != build_dev140(manifest, freeze):
        raise ValueError("dev140 differs from its frozen full-dev rule")
    return listing, tasks


def policy_condition(policy, spec, config):
    if policy == "teacher":
        return "teacher"
    server = config.get("server_provenance", {}).get("config", {}).get("server", {})
    name = "merged_lora_student" if server.get("merged_model") else "base_student"
    if server.get("checkpoint_kind") == "full_parameter":
        name = "full_parameter_student"
    if spec["model"] != config["models"]["student"]["requested_repo"]:
        name = "lora_student"
    condition = name + "_thinking_" + str(spec["enable_thinking"]).lower()
    return condition + "_sag" if spec.get("decoding") == "sag" else condition


def _worker(job):
    """One process owns one environment/client at a time; artifacts never collide."""
    task, config, output, policy_name, spec, inputs = job
    output = Path(output)
    try:
        verify_task_files(config["environment"]["data_root"], task)
        if policy_name == "student":
            student_spec = dict(spec)
            decoding = student_spec.pop("decoding", "greedy")
            if decoding == "sag":
                from students.sag_student import SAGStudent
                policy = SAGStudent(log_dir=output / "student_calls", task_id=task["task_id"], **student_spec)
            else:
                policy = Student(log_dir=output / "student_calls", **student_spec)
        else:
            teacher_spec = config["models"]["teacher"]
            policy = Teacher(cache_dir=output / "teacher_cache", usage_ledger=output / "teacher_usage.jsonl",
                             run_id="pending", provider_model_id=teacher_spec["provider_model_id"],
                             sampling=teacher_spec["sampling"], max_attempts=teacher_spec["max_attempts"],
                             timeout_seconds=teacher_spec["timeout_seconds"],
                             **rate_limit_options(teacher_spec, config["suite"]["teacher_rate_limit_dir"]),
                             secrets_file=config["paths"]["secrets_file"], purpose="evaluation")
        condition = policy_condition(policy_name, spec, config)
        run_episode(_environment(config, task), policy, task, config, output,
                    policy_condition=condition, purpose="evaluation", input_paths=inputs)
    except Exception as exc:
        # Failed episodes remain in the denominator. Preserve sanitized preflight
        # failures too; never serialize exception bodies or silently skip tasks.
        if not (output / "episode.json").exists():
            output.mkdir(parents=True, exist_ok=True)
            if not (output / "provenance.json").exists():
                write_provenance(output, command="episode_preflight_failure", config=config,
                                 inputs=inputs, details={"task_id": task["task_id"]})
            write_json(output / "failure.json", {"task_id": task["task_id"],
                       "status": "error", "error_type": type(exc).__name__})
        else:
            episode = json.loads((output / "episode.json").read_text())
            if episode["status"] != "error":
                episode.update(status="error", complete=False, success="unknown", error_type="worker_error")
                write_json(output / "episode.json", episode)
    return task["task_id"]


def task_result(task, directory):
    path = Path(directory) / "episode.json"
    episode = json.loads(path.read_text()) if path.exists() else {}
    failure_path = Path(directory) / "failure.json"
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
    metrics = episode.get("metrics", {})
    row = {"task_id": task["task_id"], "task_type": task["task_type"], "group_id": task["group_id"],
           "split": task["split"], "status": episode.get("status", "error"),
           "error_type": episode.get("error_type", failure.get("error_type")),
           "success": episode.get("success", "unknown"), "steps": metrics.get("steps", 0),
           "environment_steps": metrics.get("environment_steps", 0),
           "environment_steps_confirmed": metrics.get("environment_steps_confirmed", 0),
           "environment_steps_unknown": metrics.get("environment_steps_unknown", 0),
           "parser_errors": metrics.get("parser_errors", 0),
           "inadmissible_actions": metrics.get("inadmissible_actions", 0),
           "legal_actions": metrics.get("legal_actions", 0),
           "failed_policy_calls": metrics.get("failed_policy_calls", 0),
           "wall_time_seconds": metrics.get("wall_time_seconds", 0),
           "policy_latency_seconds": metrics.get("policy_latency_seconds", 0),
           "episode_dir": str(directory)}
    for key in ("prompt_tokens", "completion_tokens", "thinking_tokens", "thinking_chars"):
        item = metrics.get(key, {"known_total": 0, "unknown_calls": 0})
        row[key] = item["known_total"]
        row[key + "_unknown_calls"] = item["unknown_calls"]
    return row


def summarize(rows, wall_time):
    success = sum(r["success"] is True for r in rows)
    unknown = sum(r["success"] not in (True, False) for r in rows)
    totals = {key: sum(r[key] for r in rows) for key in rows[0] if key not in {
        "task_id", "task_type", "group_id", "split", "status", "success", "episode_dir", "error_type"}}
    return {"episodes": len(rows), "unique_instances": len({r["task_id"] for r in rows}),
            "task_groups": len({r["group_id"] for r in rows}), "successes": success,
            "success_rate": success / len(rows), "unknown_success": unknown,
            "completed_episodes": sum(r["status"] == "ok" for r in rows),
            "error_episodes": sum(r["status"] != "ok" for r in rows),
            "totals": totals, "wall_time_seconds": wall_time,
            "counts_by_task_type": dict(sorted(Counter(r["task_type"] for r in rows).items())),
            "interpretation": "Infrastructure/pilot evaluation; unknown outcomes retained in denominator. "
                              "Token totals are known-only; unknown counts and failed calls are separate."}


def run_suite(*, config, tasks_path, output, policy="student", spec=None, workers=4,
              sealed_final=False, inputs=(), worker_fn=_worker):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    if policy not in {"student", "teacher"}:
        raise ValueError("Unknown suite policy")
    if policy == "student" and spec.get("decoding", "greedy") not in {"greedy", "sag"}:
        raise ValueError("Unknown student decoding")
    # The evaluated server is authoritative; the CPU worker's environment is not.
    server = config.get("server_provenance", {}).get("config", {}).get("server", {})
    batch_invariant = server.get("batch_invariant") if policy == "student" else None
    if batch_invariant is not None and type(batch_invariant) is not bool:
        raise ValueError("Server batch_invariant must be a boolean or unknown (null)")
    if policy == "student" and spec.get("decoding") == "sag" and batch_invariant is not True:
        raise ValueError("SAG requires batch-invariant server provenance")
    if policy == "student" and spec["model"] != config["models"]["student"]["requested_repo"]:
        from tools.check_lora_serving import validate_report
        if not config.get("server_provenance") or not config.get("lora_equivalence"):
            raise ValueError("Adapter evaluation requires server provenance and a passing --lora-check")
        validate_report(config["lora_equivalence"], config["server_provenance"], spec["model"], spec["endpoint"])
        checked = config["lora_equivalence"]["identity"]["adapter"]
        if (checked["base_model"] != config["models"]["student"]["requested_repo"]
                or checked["revision"] != config["models"]["student"]["revision"]):
            raise ValueError("Equivalence report differs from configured student identity")
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    listing, tasks = frozen_tasks(tasks_path, manifest, freeze, sealed_final=sealed_final, policy=policy)
    output = Path(output)
    for name in ("raw", "verified"):
        if output.resolve().is_relative_to(Path(config["paths"][name]).resolve()):
            raise ValueError("Evaluation output must be outside training data destinations")
    output.mkdir(parents=True, exist_ok=False)
    inputs = [*inputs, tasks_path, config["paths"]["manifest"], config["paths"]["split_freeze"]]
    effective = deepcopy(config)
    effective["suite"] = {"policy": policy, "student": spec, "workers": workers,
                          "batch_invariant": batch_invariant,
                          "model_name": spec["model"] if spec else None,
                          "teacher_rate_limit_dir": str((output / "data/teacher_rate_limit").resolve()),
                          "sealed_final": sealed_final, "task_list_sha256": listing["list_sha256"]}
    write_provenance(output, command="suite", config=effective, inputs=inputs)
    write_json(output / "tasks.json", listing, exclusive=True)
    directories = [output / "episodes" / f"{i:03d}-{digest(t['task_id'])[:12]}" for i, t in enumerate(tasks)]
    jobs = [(task, effective, str(directory), policy, spec, inputs) for task, directory in zip(tasks, directories)]
    started = time.monotonic()
    # spawn prevents environment RNG/backend state leaking between processes.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for _ in pool.map(worker_fn, jobs):
            pass
    rows = [task_result(t, d) for t, d in zip(tasks, directories)]
    summary = summarize(rows, time.monotonic() - started)
    summary.update(task_list_sha256=listing["list_sha256"], policy=policy, student=spec,
                   batch_invariant=batch_invariant,
                   policy_condition=policy_condition(policy, spec, config))
    write_json(output / "summary.json", summary, exclusive=True)
    with (output / "per_task.tsv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tasks", default="data/manifest/dev40.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=("student", "teacher"), default="student")
    parser.add_argument("--decoding", choices=("greedy", "sag"), default="greedy")
    parser.add_argument("--sag-sampling-seed", type=int, default=0)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", help="Request model: base served name (default) or adapter NAME")
    parser.add_argument("--lora-check", help="Passing check_lora_serving report.json for this server/adapter")
    parser.add_argument("--enable-thinking", choices=("off",), default="off")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sealed-final", action="store_true")
    parser.add_argument("--server-provenance", help="Hash/copy the launcher provenance into this run")
    args = parser.parse_args(argv)
    if args.decoding == "sag" and args.policy != "student":
        parser.error("SAG is a student-only test-time policy")
    if args.max_tokens != 1024:
        parser.error("Stage 2b fixes the student per-step cap at 1024 tokens")
    config = load_config(args.config)
    if not config["environment"]["data_root"]:
        parser.error("Set ALFWORLD_DATA_ROOT")
    model = config["models"]["student"]
    spec = dict(endpoint=args.endpoint, model=args.model_name or model["requested_repo"], revision=model["revision"],
                enable_thinking=args.enable_thinking == "on", max_tokens=args.max_tokens,
                timeout_seconds=args.timeout)
    if args.decoding == "sag":
        spec["decoding"] = "sag"
        spec["sampling_seed"] = args.sag_sampling_seed
    inputs = [args.config]
    if args.server_provenance:
        inputs.append(args.server_provenance)
        server = json.loads(Path(args.server_provenance).read_text())
        config["server_provenance"] = server
    if args.lora_check:
        inputs.append(args.lora_check)
        config["lora_equivalence"] = json.loads(Path(args.lora_check).read_text())
    summary = run_suite(config=config, tasks_path=args.tasks, output=args.output, policy=args.policy,
                        spec=spec if args.policy == "student" else None, workers=args.workers,
                        sealed_final=args.sealed_final, inputs=inputs)
    print(json.dumps(summary, indent=2))
    return int(summary["error_episodes"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
