"""M0b-1 coverage orchestration over the existing pool episode runner.

No teacher, training, model loading or alternative parser/success verifier.
Each invocation is a complete, immutable run (two initial attempts plus the
explicitly requested extras). Use --plan-only before starting a served student.
"""

import argparse
from collections import Counter
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import shlex
import sys
import time

from eval.run_suite import policy_condition, summarize, task_result
from sed.config import load_config
from sed.manifest import authorize_task, frozen_write, load_frozen, resolve_task
from sed.provenance import write_provenance
from sed.storage import JsonlLedger, digest, file_hash, write_json
from students.sampled_student import SampledStudent
from students.vllm_student import Student, context_options
from tools import rollout_pool

FAILURE_SEED = 20261028


def attempts(extra_sampled=0):
    if type(extra_sampled) is not int or not 0 <= extra_sampled <= 2:
        raise ValueError("extra_sampled must be an integer from 0 to 2")
    return [{"attempt": 1, "decoding": "greedy", "sampling_seed": None},
            *[{"attempt": seed + 2, "decoding": "sampled", "sampling_seed": seed}
              for seed in range(extra_sampled + 1)]]


def checked_repair_splits(path, manifest, freeze):
    value = json.loads(Path(path).read_text())
    if (value.get("content_sha256") != digest({k: v for k, v in value.items() if k != "content_sha256"})
            or value["manifest_sha256"] != digest(manifest)
            or value["split_freeze_sha256"] != digest(freeze)):
        raise ValueError("Repair split hash or frozen inputs differ")
    return value


def coverage_tasks(path, manifest, freeze, *, repair_splits, task_set=None):
    """Accept explicit IDs, a frozen listing, or a K20 selection's selected view.

    Replacements/candidate pools supply no implicit extra coverage tasks.
    Probe tasks remain evaluation-only even
    though their underlying ALFWorld source is TRAIN.
    """
    value = json.loads(Path(path).read_text())
    if task_set:
        if task_set != "U_mech" or value != repair_splits:
            raise ValueError("Coverage task set must be the frozen U_mech")
        value = value["sets"][task_set]
    if isinstance(value, list):
        ids = value
    else:
        if value.get("purpose") == "evaluation_only":
            raise ValueError("Evaluation-only tasks cannot supply repair failures")
        if (value.get("manifest_sha256") != digest(manifest)
                or value.get("split_freeze_sha256") != digest(freeze)):
            raise ValueError("Task list differs from frozen inputs")
        if "selected" in value:
            if value.get("selection_sha256") != digest({k: v for k, v in value.items() if k != "selection_sha256"}):
                raise ValueError("Seed selection hash differs")
            selected = [r["task_id"] for rows in value["selected"].values() for r in rows]
            ids = value.get("task_order", selected)
            if set(ids) != set(selected) or len(ids) != len(selected):
                raise ValueError("Seed task order differs from selected tasks")
        else:
            ids = value["task_ids"]
            if value.get("list_sha256") != digest(ids):
                raise ValueError("Task list hash differs")
    if (not ids or any(type(t) is not str for t in ids) or len(ids) != len(set(ids))):
        raise ValueError("Coverage needs nonempty, unique task IDs")
    blocked = set(repair_splits["sets"]["D_probe"]["group_ids"])
    tasks = [resolve_task(manifest, freeze, tid) for tid in ids]
    for task in tasks:
        authorize_task(task, "generation")
        if task["source_split"] != "train" or task["group_id"] in blocked:
            raise ValueError("Coverage requires TRAIN tasks outside D_probe groups")
    if len({t["group_id"] for t in tasks}) != len(tasks):
        raise ValueError("Coverage requires one trial per task group")
    return tasks


def attempt_spec(spec, attempt):
    result = dict(spec)
    result.pop("sampling_seed", None)
    result["decoding"] = attempt["decoding"]
    if attempt["decoding"] == "sampled":
        result["sampling_seed"] = attempt["sampling_seed"]
    return result


def coverage_policy(directory, *, spec):
    spec = dict(spec)
    decoding = spec.pop("decoding")
    cls = SampledStudent if decoding == "sampled" else Student
    return cls(log_dir=Path(directory) / "student_calls", **spec)


def coverage_plan(tasks, config, spec, output, *, extra_sampled=0):
    horizon = config["environment"]["max_episode_steps"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("H must be a positive integer from the config")
    episodes = []
    for attempt in attempts(extra_sampled):
        for task in tasks:
            episodes.append({**attempt, "task_id": task["task_id"], "task_type": task["task_type"],
                "group_id": task["group_id"], "environment_seed": config["environment"]["seed"],
                "H": horizon, "temperature": 0 if attempt["decoding"] == "greedy" else 0.7,
                "top_p": None if attempt["decoding"] == "greedy" else 0.95, "n": 1,
                "turn_0_request_seed": None if attempt["sampling_seed"] is None else
                    int(digest([attempt["sampling_seed"], task["task_id"], 0])[:8], 16)})
    return {"schema_version": 1, "student": spec, "output": str(output), "H": horizon,
        "request_seed_rule": "sha256([sampling_seed, task_id, zero_based_turn])[:8] as unsigned base-16 integer",
        "extra_sampled": extra_sampled, "episodes": episodes,
        "unique_instances": len({t["task_id"] for t in tasks}),
        "task_groups": len({t["group_id"] for t in tasks}),
        "branch_rule": ">=6 distinct verified failed tasks: P0_K20; otherwise need_extra_rollouts before extras, K0_fallback after extras",
        "failure_selection_rule": "one failure per task, earliest failed attempt; task_type then sha256([20261028, task_id]); at most 20",
        "new_teacher_calls": 0, "training_exposure": 0}


def summarize_coverage(records, attempt_summaries, *, extra_sampled):
    attempts(extra_sampled)
    failures = [r for r in records if r["success"] is False and r["status"] == "ok" and r["complete"]]
    first = {}
    for r in sorted(failures, key=lambda r: (r["attempt"], r["episode_id"])):
        first.setdefault(r["task_id"], r)
    ordered = sorted(first.values(), key=lambda r: (r["task_type"], digest([FAILURE_SEED, r["task_id"]]), r["task_id"]))
    failure_list = [{k: r[k] for k in ("task_id", "task_type", "group_id", "episode_id", "attempt",
        "decoding", "sampling_seed", "episode_dir", "episode_sha256", "events_sha256", "provenance_sha256")}
        for r in ordered[:20]]
    decision = "P0_K20" if len(first) >= 6 else "K0_fallback" if extra_sampled else "need_extra_rollouts"
    return {"schema_version": 1, "success_per_attempt": attempt_summaries,
        "distinct_failed_tasks": len(first), "distinct_failed_task_groups": len({r["group_id"] for r in first.values()}),
        "failed_episodes": len(failures), "failures_by_task_type": dict(sorted(Counter(r["task_type"] for r in first.values()).items())),
        "failed_episodes_by_task_type": dict(sorted(Counter(r["task_type"] for r in failures).items())),
        "branch_decision": decision, "failure_list": failure_list, "failure_list_sha256": digest(failure_list),
        "all_failed_task_ids": [r["task_id"] for r in ordered], "failure_order_seed": FAILURE_SEED,
        "selected_failure_count": len(failure_list), "selected_task_fraction": len(failure_list) / len(first) if first else 0,
        "new_teacher_calls": 0, "training_exposure": 0,
        "interpretation": "Coverage only; unknown/error episodes are retained and are not verified task failures. "
                          "Branch labels are the preregistered K20 rule, not evidence of a mechanism effect; "
                          "U_mech runs are P1 and must be reported separately."}


def run_coverage(tasks, config, output, *, spec, env_factory, policy_factory=coverage_policy,
                 inputs=(), extra_sampled=0, workers=1, task_source="explicit_task_list"):
    plan = coverage_plan(tasks, config, spec, output, extra_sampled=extra_sampled)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    for task in tasks:
        authorize_task(task, "generation")
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Coverage needs nonempty, unique task IDs")
    config = deepcopy(config)
    config["repair_coverage"] = {**plan, "task_source": task_source, "workers": workers}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="repair_coverage", config=config, inputs=inputs)
    frozen_write(output / "plan.json", plan)
    records, summaries = [], []
    for attempt in attempts(extra_sampled):
        directory = output / f"attempt-{attempt['attempt']:02d}-{attempt['decoding']}"
        current_spec = attempt_spec(spec, attempt)
        current_config = deepcopy(config)
        current_config["r03_rollout"] = {**attempt,
            "policy_condition": policy_condition("student", current_spec, config)}
        started = time.monotonic()
        # This is the only episode execution path, shared with the pool tool.
        rollout_pool.run_pool(tasks, current_config, directory, env_factory=env_factory,
            policy_factory=partial(policy_factory, spec=current_spec), inputs=inputs, workers=workers)
        rows = []
        for i, task in enumerate(tasks):
            ep_dir = directory / "episodes" / f"{i:03d}-{digest(task['task_id'])[:12]}"
            rows.append(task_result(task, ep_dir))
        summary = {**attempt, **summarize(rows, time.monotonic() - started)}
        summaries.append(summary)
        rollout_path = directory / "rollouts.jsonl"
        for line in rollout_path.read_text().splitlines() if rollout_path.exists() else []:
            original = json.loads(line)
            task = next(t for t in tasks if t["task_id"] == original["task_id"])
            record = rollout_pool.episode_record(directory / original["episode_dir"], task, relative_to=output)
            # Keep the pool format exact. Attempt metadata lives in a separate
            # index, so existing evidence readers can still reconstruct records.
            JsonlLedger(output / "rollouts.jsonl").append(record, unique_key="episode_id")
            JsonlLedger(output / "episode_index.jsonl").append({**attempt,
                "task_id": task["task_id"], "episode_id": record["episode_id"], "episode_dir": record["episode_dir"]})
            records.append({**record, **attempt})
    summary = summarize_coverage(records, summaries, extra_sampled=extra_sampled)
    summary.update(unique_instances=plan["unique_instances"], task_groups=plan["task_groups"],
        task_source=task_source, H=plan["H"], extra_sampled=extra_sampled,
        protocol="P1" if task_source == "U_mech" else "P0", student=spec,
        error_episodes=sum(s["error_episodes"] for s in summaries),
        unknown_success=sum(s["unknown_success"] for s in summaries))
    summary["branch_decision_actionable"] = not (summary["error_episodes"] or summary["unknown_success"])
    frozen_write(output / "failures.json", {"seed": FAILURE_SEED,
        "selection_rule": plan["failure_selection_rule"], "failures": summary["failure_list"],
        "list_sha256": summary["failure_list_sha256"]})
    write_json(output / "summary.json", summary, exclusive=True)
    return summary


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base_e4b.yaml")
    p.add_argument("--tasks", required=True, help="JSON IDs, seed selection, frozen listing or repair splits")
    p.add_argument("--task-set", choices=("U_mech",))
    p.add_argument("--repair-splits", default="data/manifest/repair_v01_splits.json")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model-name", required=True, help="Exact served adapter name (or configured base name)")
    p.add_argument("--server-provenance")
    p.add_argument("--lora-check")
    p.add_argument("--data-root")
    p.add_argument("--extra-sampled", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("workers must be positive")
    config = load_config(args.config)
    if args.data_root:
        config["environment"]["data_root"] = args.data_root
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    splits = checked_repair_splits(args.repair_splits, manifest, freeze)
    tasks = coverage_tasks(args.tasks, manifest, freeze, repair_splits=splits, task_set=args.task_set)
    model = config["models"]["student"]
    inputs = [__file__, args.config, args.tasks, args.repair_splits,
              config["paths"]["manifest"], config["paths"]["split_freeze"]]
    if args.server_provenance:
        config["server_provenance"] = json.loads(Path(args.server_provenance).read_text())
        inputs.append(args.server_provenance)
    spec = dict(endpoint=args.endpoint, model=args.model_name, revision=model["revision"],
                enable_thinking=False, max_tokens=1024, **context_options(config))
    # Constructor only validates the HTTP identity; it never creates a client.
    Student(log_dir=Path(args.output) / "unused", **spec)
    if args.plan_only:
        plan = coverage_plan(tasks, config, spec, args.output, extra_sampled=args.extra_sampled)
        plan["commands"] = [shlex.join([sys.executable, "-m", "tools.repair_coverage",
                                      *[a for a in argv if a != "--plan-only"]])]
        plan["execution_requirements"] = ["installed environment data", "server provenance",
                                           "passing LoRA check for adapter runs"]
        print(json.dumps(plan, indent=2))
        return 0
    if not config["environment"]["data_root"] or not args.server_provenance:
        p.error("Execution requires --data-root/ALFWORLD_DATA_ROOT and --server-provenance")
    if args.model_name != model["requested_repo"]:
        if not args.lora_check:
            p.error("Adapter execution requires --lora-check")
        from tools.check_lora_serving import validate_report
        report = json.loads(Path(args.lora_check).read_text())
        validate_report(report, config["server_provenance"], args.model_name, args.endpoint)
        checked = report["identity"]["adapter"]
        if checked["base_model"] != model["requested_repo"] or checked["revision"] != model["revision"]:
            raise ValueError("Adapter differs from configured student")
        config["lora_equivalence"] = report
        inputs.append(args.lora_check)
    else:
        from repair.branches import validate_base_serving
        validate_base_serving(config, spec)
    task_source = ("U_mech" if set(t["task_id"] for t in tasks) <=
                   set(splits["sets"]["U_mech"]["task_ids"]) else "K20_seed_tasks")
    summary = run_coverage(tasks, config, args.output, spec=spec,
        env_factory=partial(rollout_pool._pool_environment, config=config), inputs=inputs,
        extra_sampled=args.extra_sampled, workers=args.workers, task_source=task_source)
    print(json.dumps(summary, indent=2))
    return int(summary["error_episodes"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
