"""Hash-bound LEAP iter0..3, privileged teacher or 0% privileged ablation."""
import argparse
from copy import deepcopy
from importlib.metadata import version
import os
from pathlib import Path
from teachers.stage import ledger_counts
import signal
import sys

from baselines.common.dose_control import validate_evaluation_init, materialize_init, model_record, SUFFIX
from baselines.common.explore import checked_rollouts
from baselines.common.pipeline import batch_invariant, prepare, update_disclosure as common_disclosure
from baselines.eto.pipeline import stage_run, run_command, served, publish_models, evaluate_pipeline
from baselines.leap.data import seeds, prepare_set, min_subsample
from baselines.leap.expert import collect_one
from baselines.leap.protocol import LABELS, PIN, render, template_hashes, trajectory, expert_source_hashes
from baselines.leap.teacher import TeacherBudget, checked_outcome, read
from baselines.leap.train import EVAL_SAVE_DEVIATION, RECIPE, training_dose
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import versions, write_provenance
from sed.runner import episode_metrics
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, locked, write_json
from teachers.stage import teacher_stage
from teachers.baseline_transport import baseline_transport


def teacher_caps(k, variant):
    policy = baseline_transport("leap")
    def caps(n):
        return {"logical_calls": n, "method_attempts": 3 * n,
                "http_attempts_excluding_rate_limits": 3 * n * policy.http_attempts,
                "http_attempts_upper_bound": 3 * n * (policy.http_attempts + policy.rate_limit_attempts - 1)}
    return {"extraction": caps(k if variant == "privileged" else 0),
        "correction_by_iteration": {str(i): caps(k) for i in range(1, 4)},
        "max_attempts_per_logical_call": 3, "baseline_transport": policy.disclosure()}


def training_command(args, *, snapshot, sets, iteration, output):
    return [args.train_python, "-m", "baselines.leap.train", "--config", args.config,
        "--snapshot", str(snapshot), "--base-snapshot", args.snapshot, "--verified", args.verified,
        "--iteration", str(iteration), "--variant", args.variant, "--output", str(output), "--sets", *map(str, sets)]


def collect_experts(args, config, identity, tasks, output):
    records = []
    root = Path(args.output) / "audit/experts"
    for task in tasks:
        collected = stage_run(root, digest(task["task_id"]), digest([identity, task]),
            lambda a: collect_one(task, config, a / "output"))
        receipt = read(collected / "receipt.json")
        records.append({**receipt, "directory": str(collected), "receipt_sha256": file_hash(collected / "receipt.json")})
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="leap_expert_availability", config=config,
        inputs=[Path(r["directory"]) / "receipt.json" for r in records], details={"audit_only": True})
    write_json(output / "experts.json", records)
    write_json(output / "report.json", {"tasks": len(tasks), "available": sum(r["available"] for r in records),
        "excluded": sum(not r["available"] for r in records), "trusted_trajectories_added": 0})


@teacher_stage("output", name="leap.extract")
def extract_states(args, config, identity, tasks, experts, output):
    budget = TeacherBudget(Path(args.output) / "audit/teacher/extraction", tasks=tasks,
        config=config, kind="extract", iteration=0, run_identity=digest(identity),
        max_output_tokens=args.extraction_tokens, sampling_root=Path(args.output) / "audit/teacher")
    records = read(experts / "experts.json")
    if [r["task_id"] for r in records] != [t["task_id"] for t in tasks]:
        raise ValueError("Expert coverage differs from frozen seed tasks")
    states = []
    for task, record in zip(tasks, records):
        directory = Path(record["directory"])
        if record["task_id"] != task["task_id"] or file_hash(directory / "receipt.json") != record["receipt_sha256"]:
            raise ValueError("Expert availability receipt changed")
        hashes = artifact_hashes(directory)
        hashes.pop("receipt.json")
        if hashes != record["artifact_sha256"]:
            raise ValueError("Expert executed evidence changed")
        row = {"task_id": task["task_id"], "parent_seed_ids": task["parent_seed_ids"],
               "expert_available": record["available"], "state": None, "status": "expert_unavailable"}
        if record["available"]:
            episode = read(directory / "episode/episode.json")
            if episode["success"] is not True or episode["task_id"] != task["task_id"]:
                raise ValueError("Extraction requires the executed expert win")
            result = budget.request(task, render("extract", trajectory(episode, expert=True)))
            row.update(status=result["status"], state=result["labels"], request_directory=result["directory"],
                outcome_sha256=file_hash(Path(result["directory"]) / "outcome.json"))
        states.append(row)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="leap_privileged_extraction", config=config,
        inputs=[experts / "experts.json"], details={"audit_only": True, "caps": budget.binding})
    write_json(output / "states.json", states)
    write_json(output / "report.json", {"tasks": len(tasks), "expert_unavailable": sum(not r["expert_available"] for r in states),
        "extracted": sum(r["status"] == "accepted" for r in states),
        "extraction_failed": sum(r["status"] == "exhausted" for r in states)})


@teacher_stage("output", name="leap.correct")
def correct_rollouts(args, config, identity, tasks, exploration, states, iteration, output):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    rollouts = checked_rollouts(exploration, manifest, freeze)
    if [r["task_id"] for r, _ in rollouts] != [t["task_id"] for t in tasks]:
        raise ValueError("LEAP corrections require exactly the K seed rollouts")
    if (args.variant == "privileged") != (states is not None):
        raise ValueError("Privileged state input differs from named variant")
    state_map = {r["task_id"]: r for r in read(states / "states.json")} if states else {}
    if states and set(state_map) != {t["task_id"] for t in tasks}:
        raise ValueError("Privileged state coverage differs from seed scope")
    budget = TeacherBudget(Path(args.output) / f"audit/teacher/correction-{iteration}", tasks=tasks,
        config=config, kind=args.variant, iteration=iteration, run_identity=digest(identity),
        max_output_tokens=args.correction_tokens, sampling_root=Path(args.output) / "audit/teacher")
    records = []
    for task, (receipt, episode) in zip(tasks, rollouts):
        if receipt["parent_seed_ids"] != task["parent_seed_ids"]:
            raise ValueError("Rollout parent seed differs")
        row = {"task_id": task["task_id"], "episode_sha256": digest(episode), "status": "won"}
        if receipt["won"] is False:
            state = state_map.get(task["task_id"])
            if args.variant == "privileged" and (not state or state["status"] != "accepted"):
                row["status"] = "excluded_no_privileged_state"
            else:
                privileged = state["state"] if state else None
                if state:
                    extracted = checked_outcome(state["request_directory"])
                    if (file_hash(Path(state["request_directory"]) / "outcome.json") != state["outcome_sha256"]
                            or extracted["status"] != "accepted" or extracted["labels"] != privileged
                            or extracted["task_id"] != task["task_id"]):
                        raise ValueError("Privileged state differs from extracted expert evidence")
                result = budget.request(task, render(args.variant, trajectory(episode), privileged),
                    length=len(episode["steps"]), privileged_state=privileged)
                row.update(status=result["status"], request_directory=result["directory"],
                    outcome_sha256=file_hash(Path(result["directory"]) / "outcome.json"))
        records.append(row)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="leap_corrections", config=config,
        inputs=[exploration / "receipts.json", *([states / "states.json"] if states else [])],
        details={"variant": args.variant, "label": LABELS[args.variant], "caps": budget.binding,
                 "correction_verification": "unknown; sampled labels are not executed successes"})
    write_json(output / "corrections.json", records)
    write_json(output / "report.json", {"iteration": iteration, "rollouts": len(rollouts),
        "failed_rollouts": sum(r["won"] is False for r, _ in rollouts),
        "corrections": sum(r["status"] == "accepted" for r in records),
        "rejected_corrections": sum(r["status"] == "exhausted" for r in records),
        "excluded_no_privileged_state": sum(r["status"] == "excluded_no_privileged_state" for r in records),
        "caps": teacher_caps(len(tasks), args.variant)["correction_by_iteration"][str(iteration)]})


def train_pipeline(args, identity, config):
    output, root_identity = Path(args.output), digest(identity)
    tasks = [task for task, _ in seeds(args.verified, config)]
    data = stage_run(output, "data-0", digest([root_identity, "seeds"]),
        lambda a: prepare_set(args.verified, a / "output", config, iteration=0))
    sets, models, states = [data / "examples.json"], [], None
    if args.through == "data-0":
        return
    if args.variant == "privileged":
        experts = stage_run(output / "audit", "expert", digest([root_identity, "expert"]),
            lambda a: collect_experts(args, config, identity, tasks, a / "output"))
        if args.through == "expert":
            return
        states = stage_run(output / "audit", "extract", digest([root_identity, str(experts)]),
            lambda a: extract_states(args, config, identity, tasks, experts, a / "output"))
        if args.through == "extract":
            return
    current = Path(args.snapshot)
    for iteration in range(4):
        if iteration == 0 and "init" in identity:
            current = materialize_init(args, identity)
            models.append(model_record(identity, "iter-0", current, label=LABELS[args.variant]))
            publish_models(output, models)
            write_json(output / "iteration-0.json", {"iteration": 0, "label": models[-1]["label"],
                "checkpoint": str(current), "init": identity["init"], "sft_replaced": True,
                "training_exposure": 0, "example_sets": list(map(str, sets))})
            if args.through in {"train-0", "merge-0"}:
                return
            continue
        report = {"iteration": iteration, "label": LABELS[args.variant]}
        if iteration:
            def explore(attempt):
                with served(args, current, attempt) as endpoint:
                    command = [args.env_python, "-m", "baselines.common.explore", "--config", args.config,
                        "--tasks", str(output / "tasks.json"), "--endpoint", endpoint, "--model-name", args.model_name,
                        "--decoding", "greedy", "--workers", str(args.workers), "--server-provenance",
                        str(attempt / "server/provenance.json"), "--output", str(attempt / "output")]
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = ""
                    run_command(command, attempt / "explore.log", env=env)
            exploration = stage_run(output, f"explore-{iteration}", digest([root_identity, iteration, str(current)]), explore)
            if args.through == f"explore-{iteration}":
                return
            corrections = stage_run(output, f"correct-{iteration}", digest([root_identity, iteration, str(exploration), str(states)]),
                lambda a: correct_rollouts(args, config, identity, tasks, exploration, states, iteration, a / "output"))
            report.update(read(corrections / "report.json"))
            if args.through == f"correct-{iteration}":
                return
            data = stage_run(output, f"data-{iteration}", digest([root_identity, iteration, str(corrections)]),
                lambda a: prepare_set(args.verified, a / "output", config, iteration=iteration,
                    corrections=corrections, exploration=exploration))
            sets.append(data / "examples.json")
        _, aggregation = min_subsample([read(path) for path in sets])
        report.update(aggregation=aggregation, example_sets=list(map(str, sets)),
            inadmissible_corrections=read(data / "report.json")["inadmissible_corrections"],
            init=str(current), dose=training_dose(aggregation["N_examples"]) if aggregation["N_examples"] else None)
        write_json(output / f"iteration-{iteration}.json", report)
        if args.through == f"data-{iteration}":
            return
        if not aggregation["N_examples"]:
            # Upstream cannot fit an empty aggregate. Do not invent seed-only
            # updates or regenerate labels beyond the registered caps.
            write_json(output / "result.json", {"status": "stopped_empty_min_aggregate", "iteration": iteration,
                "final_checkpoint": str(current), "models": models, "training_exposure": 0,
                "reason": "A correction set is empty; official min subsampling leaves no training examples"})
            return
        trained = stage_run(output, f"train-{iteration}", digest([root_identity, iteration, str(current), *map(str, sets)]),
            lambda a: run_command(training_command(args, snapshot=current, sets=sets, iteration=iteration,
                output=a / "output"), a / "train.log"))
        if args.through == f"train-{iteration}":
            return
        merged = stage_run(output, f"merge-{iteration}", digest([root_identity, iteration, str(current), str(trained)]),
            lambda a: run_command([args.train_python, "-m", "baselines.leap.merge", "--config", args.config,
                "--snapshot", str(current), "--adapter", str(trained / "adapter"), "--output", str(a / "output")], a / "merge.log"))
        current = merged / "checkpoint"
        models.append(model_record(identity, f"iter-{iteration}", current, label=LABELS[args.variant]))
        publish_models(output, models)
        write_json(output / f"iteration-{iteration}.json", {**report, "checkpoint": str(current),
            "training": read(trained / "result.json")})
        update_disclosure(output, identity)
        if args.through == f"merge-{iteration}":
            return
    write_json(output / "result.json", {"status": "complete", "label": LABELS[args.variant],
        "final_checkpoint": str(current), "models": models})


def update_disclosure(output, identity):
    common_disclosure(output, identity)
    output = Path(output)
    report = read(output / "disclosure.json")
    report["eval_save_deviation"] = EVAL_SAVE_DEVIATION
    for attempt in report["training_attempts"]:
        provenance = output / attempt["path"] / "output/provenance.json"
        if provenance.exists():
            training = read(provenance)["config"]["leap_training"]
            attempt["dose"] = {key: training[key] for key in ("total_steps", "official_eval_save_step",
                "updates_exceed_official_eval_save_step", "eval_save_deviation") if key in training}
    calls, malformed = [], False
    for path in sorted((output / "audit/teacher").rglob("usage.jsonl")):
        for line in path.read_text().splitlines():
            try:
                import json
                calls.append(json.loads(line))
            except ValueError:
                malformed = True
    starts = [r for r in calls if r["status"] == "started"]
    ends = [r for r in calls if r["status"] not in {"started", "cache_hit"}]
    ended = {r["call_id"] for r in ends}
    usage = [r["usage"] for r in ends if r.get("usage") is not None]
    experts = []
    for attempt in sorted((output / "audit/experts").glob("stages/*/attempt-*")):
        receipt = attempt / "output/receipt.json"
        episode = attempt / "output/episode/episode.json"
        value = read(receipt) if receipt.exists() else {"available": False, "status": "interrupted"}
        ep = read(episode) if episode.exists() else {}
        experts.append({**value, "path": str(attempt.relative_to(output)),
            "task_id": value.get("task_id", ep.get("task_id")),
            "environment_episode_started": bool(ep), "environment_reset_confirmed": "initial" in ep,
            "metrics": episode_metrics(ep) if ep else None,
            "unknown_costs_possible": not receipt.exists() or ep.get("status") not in {None, "ok"}})
    availability = {r["task_id"]: r["available"] for r in experts if r.get("task_id")
                    and r.get("status") != "interrupted"}
    report.update(source="leap_seed_task_dagger", label=identity["recipe"]["label"],
        privileged_teacher=identity["recipe"]["variant"] == "privileged",
        teacher_caps=teacher_caps(identity["K"], identity["recipe"]["variant"]),
        teacher_sampling=read(output / "audit/teacher/sampling.json") if (output / "audit/teacher/sampling.json").exists() else "not_called",
        expert_attempts=experts, expert_availability=availability,
        expert_unavailable=sum(not available for available in availability.values()),
        expert_unavailable_attempts=sum(not r["available"] for r in experts),
        correction_reports=[read(p) for p in sorted((output / "stages").glob("correct-*/attempt-*/output/report.json"))],
        token_accounting="All teacher attempts/cached raw outputs in audit/teacher; original seed usage remains in bound receipts. "
            "Corrected labels have unknown verification. Every retry/failed rollout/expert transition and training exposure is retained.")
    report["totals"].update(teacher_calls=len(starts),
        teacher_returned=sum(r["status"] in {"success", "rejected", "incomplete"} for r in ends),
        teacher_usage_known=len(usage),
        teacher_attempt_slots_reserved=len(list((output / "audit/teacher").glob("*/requests/*/attempt-*/start.json"))),
        teacher_logical_calls=len(list((output / "audit/teacher").glob("*/requests/*/binding.json"))),
        teacher_failed_attempts=sum(r["status"] == "failed" for r in ends),
        teacher_preparation_failures=sum(r.get("request_started") is False for r in ends),
        teacher_unknown_usage=sum(r.get("usage") is None and r.get("request_started") is not False for r in ends)
            + sum(r["call_id"] not in ended for r in starts),
        teacher_log_malformed=malformed,
        **{"teacher_" + key: sum(u.get(key) or 0 for u in usage) for key in
           ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens")},
        expert_environment_steps=sum((r["metrics"] or {}).get("environment_steps", 0) for r in experts),
        expert_environment_episodes=sum(r["environment_episode_started"] for r in experts),
        expert_resets_confirmed=sum(r["environment_reset_confirmed"] for r in experts))
    report["teacher_outcomes"] = {key: sum(ledger_counts(p)[key] for p in (output / "audit/teacher").rglob("usage.jsonl"))
        for key in ("transport_failures", "rejections", "truncated_outputs", "unknown_ledger_rows")}
    write_json(output / "disclosure.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--init-checkpoint", help="Final same-cell Seed-SFT LoRA adapter")
    for name in ("snapshot", "verified", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--variant", choices=list(LABELS), required=True)
    parser.add_argument("--mode", choices=["prepare", "train", "evaluate"], default="train")
    parser.add_argument("--through", choices=["all", "expert", "extract"] + [f"{s}-{i}" for i in range(4)
        for s in ("data", "train", "merge")] + [f"{s}-{i}" for i in range(1, 4) for s in ("explore", "correct")], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--extraction-tokens", type=int, default=4096)
    parser.add_argument("--correction-tokens", type=int, default=16384)
    for name in ("env", "train", "vllm"):
        parser.add_argument(f"--{name}-python", default=os.environ.get(name.upper() + "_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    if min(args.workers, args.extraction_tokens, args.correction_tokens) < 1:
        parser.error("Positive workers and output caps required")
    if args.variant == "zero" and args.through in {"expert", "extract"}:
        parser.error("0% privileged variant does not collect unused oracle state")
    args.gpus = 1
    batch_invariant()
    for name in ("snapshot", "verified", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    config = load_config(args.config)
    if (config["environment"]["max_episode_steps"] != 40 or config["training"]["thinking_mode"] != "off"
            or config["training"]["student_step_token_cap"] != 1024):
        raise ValueError("LEAP fixes X3: 40 turns, OFF, 1024 tokens")
    args.model_name = config["models"]["student"]["requested_repo"]
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"):
        if args.mode == "evaluate":
            validate_evaluation_init(args, config)
            if read(Path(args.output) / "identity.json")["recipe"]["variant"] != args.variant:
                raise ValueError("Evaluation variant label differs from training")
            suffix = SUFFIX if read(Path(args.output) / "identity.json").get("init") else ""
            for model in read(Path(args.output) / "models.json"):
                state = read(Path(model["checkpoint"]) / "training_state.json")
                if (state.get("leap_variant") != args.variant or model["label"] != LABELS[args.variant] + suffix
                        or model["name"] != f"iter-{state.get('leap_iteration')}"):
                    raise ValueError("Published checkpoint variant/iteration differs from evaluation label")
            os.environ["DECODING"] = "greedy"
            evaluate_pipeline(args)
            return
        recipe = {**deepcopy(RECIPE), "pin": PIN, "variant": args.variant, "label": LABELS[args.variant],
            "template_sha256": template_hashes(), "runtime_dependencies": versions(),
            "template_engine_version": version("Jinja2"),
            "expert_source_sha256": expert_source_hashes() if args.variant == "privileged" else {},
            "extraction_tokens": args.extraction_tokens, "correction_tokens": args.correction_tokens}
        identity = prepare(args, config, method="leap", recipe=recipe)
        write_json(Path(args.output) / "teacher_caps.json", teacher_caps(identity["K"], args.variant))
        try:
            if args.mode == "train":
                train_pipeline(args, identity, config)
        finally:
            update_disclosure(args.output, identity)


if __name__ == "__main__":
    main()
