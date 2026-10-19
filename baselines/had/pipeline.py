"""Hash-bound HAD rounds: explore-r (GPU), teacher-r (CPU), train-r (GPU).

Per-stage invocations execute only the selected stage. Paired workers reuse
the same transactions and commands; only the CPU worker invokes the teacher.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys

from baselines.common.pipeline import (batch_invariant, prepare, stage_run, run_command,
    served, publish_models, update_disclosure as shared_disclosure)
from baselines.had import RECIPE
from baselines.had.data import collect, require_cpu_teacher, round_tasks
from sed.config import load_config
from sed.manifest import frozen_write
from sed.storage import digest, file_hash, locked, write_json


def stage(root, name, identity, action, *, selected):
    if name != selected and not (Path(root) / "stages" / name / "complete.json").exists():
        raise ValueError(f"Missing prerequisite {name}; run it in its own stage job")
    return stage_run(root, name, identity, action)


def training_command(args, *, data, snapshot, output, round_index, start_update):
    return [args.train_python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.gpus),
        "-m", "baselines.common.fsdp_train", "--config", args.config, "--objective", "had",
        "--snapshot", str(snapshot), "--base-snapshot", args.snapshot, "--verified", args.verified,
        "--had-data", str(data), "--had-identity", str(Path(args.output) / "identity.json"),
        "--had-round", str(round_index), "--start-update", str(start_update), "--output", str(output),
        "--learning-rate", "1e-5", "--scheduler", "cosine", "--warmup-ratio", "0", "--warmup-steps", "15",
        "--original-batch", "60", "--weight-decay", "0.01", "--beta", "0.5", "--epochs", "1", "--packing", "turn"]


def train_pipeline(args, identity, config, *, execute_stage=stage, publication_lock=locked):
    selected = args.through
    match = re.fullmatch(r"(explore|teacher|train)-(\d+)", selected)
    if not match or int(match[2]) >= 250:
        raise ValueError("HAD_THROUGH must select explore-r, teacher-r or train-r (0 <= r < 250)")
    if match[1] == "teacher":
        require_cpu_teacher()
    output, root_hash = Path(args.output), digest(identity)
    listing = json.loads((output / "tasks.json").read_text())
    snapshot, update = Path(args.snapshot), 0
    for version in range(int(match[2]) + 1):
        tasks = round_tasks(listing, version)
        task_path = output / f"tasks-{version}.json"
        frozen_write(task_path, tasks)
        key = digest([root_hash, version, str(snapshot), update, tasks])
        def explore(attempt):
            with served(args, snapshot, attempt) as endpoint:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""
                command = [args.env_python, "-m", "baselines.had.rollout", "--config", args.config,
                    "--tasks", str(task_path), "--endpoint", endpoint, "--model-name", args.model_name,
                    "--workers", str(args.workers), "--round", str(version),
                    "--server-provenance", str(attempt / "server/provenance.json"), "--output", str(attempt / "output")]
                run_command(command, attempt / "explore.log", env=env)
        explored = execute_stage(output, f"explore-{version}", key, explore, selected=selected)
        if selected == f"explore-{version}":
            return
        labelled = execute_stage(output, f"teacher-{version}", digest([key, str(explored)]),
            lambda a: collect(explored, args.verified, a / "output", config,
                cell=output, identity=identity, round_index=version), selected=selected)
        if selected == f"teacher-{version}":
            return
        trained = execute_stage(output, f"train-{version}", digest([key, str(labelled)]),
            lambda a: run_command(training_command(args, data=labelled / "data.json", snapshot=snapshot,
                output=a / "output", round_index=version, start_update=update), a / "train.log"), selected=selected)
        result = json.loads((trained / "result.json").read_text())
        if result["start_update"] != update or not update < result["end_update"] <= 250:
            raise ValueError("HAD update chain changed")
        update = result["end_update"]
        snapshot = trained / "checkpoint"
        # Both paired workers may reconstruct a completed prefix. Serialize only
        # publication, never the CPU teacher and GPU work behind a cell-wide lock.
        with publication_lock(output / ".publication.lock"):
            publish_models(output, [{"name": f"iter-{version + 1}-{harness}", "checkpoint": str(snapshot),
                "method": "had", "harness": harness, "update": update, "policy_version": version + 1,
                "checkpoint_state_sha256": file_hash(snapshot / "training_state.json")}
                for harness in ("had", "h0")])
            write_json(output / f"iteration-{version}.json", {"iteration": version, "checkpoint": str(snapshot),
                "start_update": result["start_update"], "end_update": update, "examples_seen": result["examples_seen"],
                "unused_at_terminal_budget": result["unused_at_terminal_budget"]})
            if update == 250:
                write_json(output / "result.json", {"status": "complete", "updates": update,
                    "models": json.loads((output / "models.json").read_text()), "evaluation_selection": None})
        if update == 250:
            return
        if selected == f"train-{version}":
            return


def update_disclosure(output, identity):
    output = Path(output)
    shared_disclosure(output, identity)
    disclosure = json.loads((output / "disclosure.json").read_text())
    usage_path = output / "teacher/usage.jsonl"
    usage, partial = [], False
    if usage_path.exists():
        for line in usage_path.read_text().splitlines():
            try:
                usage.append(json.loads(line))
            except json.JSONDecodeError:
                partial = True
    counter_path = output / "teacher/counter.json"
    counter = json.loads(counter_path.read_text()) if counter_path.exists() else {}
    billed = [r for r in usage if r.get("status") in {"success", "rejected"}]
    disclosure.update(method="had", source="student_had_rollouts_and_paired_black_box_teacher",
        teacher_cap=30000, teacher_counter=str(counter_path), teacher_usage_ledger=str(usage_path),
        unknown_teacher_usage_possible=partial or counter.get("pending", False) or counter.get("halted", False),
        policy_version_unit=RECIPE["policy_version_unit"],
        token_accounting="Teacher usage and returned IDs/timestamps in durable ledger; student raw calls in rollout attempts.")
    disclosure["totals"].update(teacher_calls=counter.get("attempts", 0),
        teacher_input_tokens=sum((r.get("usage") or {}).get("input_tokens", 0) or 0 for r in billed),
        teacher_output_tokens=sum((r.get("usage") or {}).get("output_tokens", 0) or 0 for r in billed))
    write_json(output / "disclosure.json", disclosure)


def evaluate_pipeline(args):
    output = Path(args.output)
    identity = json.loads((output / "identity.json").read_text())
    # Evaluate does not call prepare, collect, explore or a training loader.
    if (identity["config_sha256"] != digest(load_config(args.config))
            or identity["snapshot"] != args.snapshot or identity["verified"] != args.verified):
        raise ValueError("HAD evaluation cell identity changed")
    models = json.loads((output / "models.json").read_text())
    for model in models:
        checkpoint = Path(model["checkpoint"])
        if file_hash(checkpoint / "training_state.json") != model["checkpoint_state_sha256"]:
            raise ValueError("Published HAD checkpoint changed")
        def evaluate(attempt):
            env = os.environ.copy()
            env.update(PROJECT_ROOT=str(Path.cwd()), CONFIG=args.config, MODEL_PATH=args.snapshot,
                MERGED_MODEL_PATH=str(checkpoint), FULL_CHECKPOINT_NAME=model["name"],
                HAD_EVAL_HARNESS=model["harness"], VERIFIED=args.verified,
                ENV_PYTHON=args.env_python, TRAIN_PYTHON=args.train_python, VLLM_PYTHON=args.vllm_python,
                TASKS=str(Path("data/manifest/dev140.json").resolve()), OUT=str(attempt / "output"),
                WORKERS=str(args.workers), INCLUDE_BASE="1", READY_TIMEOUT=str(args.ready_timeout),
                DECODING="greedy", BATCH_INVARIANT="1", VLLM_BATCH_INVARIANT="1")
            if env.get("CUDA_VISIBLE_DEVICES"):
                env["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"].split(",")[0]
            run_command(["bash", "-c", "adapters=(); source scripts/eval_adapters.sh"], attempt / "eval.log", env=env)
        stage_run(output, "eval-" + model["name"], digest([identity, model]), evaluate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    for key in ("snapshot", "verified", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate"), default="train")
    parser.add_argument("--through", default="explore-0")
    parser.add_argument("--worker", choices=("gpu", "teacher"),
                        help="Run all rounds as one member of a co-scheduled pair")
    parser.add_argument("--gpu-wait-seconds", type=float, default=1200)
    parser.add_argument("--poll-seconds", type=float, default=1)
    parser.add_argument("--poll-max-seconds", type=float, default=30)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=900)
    for key in ("env", "train", "vllm"):
        parser.add_argument(f"--{key}-python", default=os.environ.get(key.upper() + "_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("Positive workers required")
    if args.worker and args.mode != "train":
        parser.error("--worker requires --mode train")
    batch_invariant()
    for key in ("snapshot", "verified", "output"):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    config = load_config(args.config)
    if file_hash("docs/baselines/NEW_METHOD_SPECS_1019.md") != RECIPE["spec_sha256"]:
        raise ValueError("Pinned HAD implementation specification changed")
    if (config["training"]["thinking_mode"] != "off" or config["training"]["student_step_token_cap"] != 1024
            or config["environment"]["max_episode_steps"] != 40
            or config["models"]["student"]["requested_repo"] != "google/gemma-4-12B-it"
            or not config["models"]["student"]["revision"].startswith("707f0a3b")
            or config["models"]["teacher"]["provider_model_id"] != "gpt-6-luna"):
        raise ValueError("HAD requires the configured paper-1 models and H0 protocol")
    args.model_name = config["models"]["student"]["requested_repo"]
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.worker:
        from baselines.had.coschedule import run_worker
        return run_worker(args, config)
    with locked(Path(args.output) / ".pipeline.lock"):
        if args.mode == "evaluate":
            evaluate_pipeline(args)
            return
        # No independent K ceiling: both nested views and forthcoming full
        # collections must pass the same shared verified-evidence validator.
        identity = prepare(args, config, method="had", recipe=RECIPE, registered_k=None)
        try:
            if args.mode == "train":
                train_pipeline(args, identity, config)
        finally:
            update_disclosure(args.output, identity)


if __name__ == "__main__":
    sys.exit(main())
