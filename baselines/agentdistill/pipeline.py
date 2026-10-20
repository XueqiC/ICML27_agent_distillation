"""Stage-isolated AgentDistill: collect/abstract/cluster/consolidate/publish; evaluate."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
from teachers.stage import ledger_counts
import signal
import subprocess
import sys
import time

from baselines.agentdistill import RECIPE, teacher_cap
from baselines.agentdistill.box import build, load_box
from baselines.agentdistill.collection import collect, tasks_for
from baselines.agentdistill.teacher import Budget, read
from baselines.common.pipeline import batch_invariant, prepare, stage_run, run_command, publish_models
from harness.serve_student import stop_server
from sed.config import load_config
from sed.provenance import code_hashes, versions
from sed.runner import episode_metrics
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, locked, write_json

STAGES = ("collect", "abstract", "cluster", "consolidate", "publish")


def check_external(source):
    """Stage artifacts bind durable run/request evidence outside attempt dirs."""
    for path, expected in read(Path(source) / "external.json").items():
        if artifact_hashes(path) != expected:
            raise ValueError("AgentDistill external run/request evidence changed")


def require_cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("AgentDistill teacher stages require CUDA_VISIBLE_DEVICES=''")
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE"):
        allowed = {""} if key in {"SLURM_JOB_GPUS", "SLURM_STEP_GPUS"} else {"", "0"}
        if os.environ.get(key, "").strip() not in allowed:
            raise ValueError("Teacher-only stages cannot hold a GPU allocation")


def run_stages(args, identity, config):
    require_cpu()
    output = Path(args.output)
    tasks = tasks_for(output, config)
    source = None
    budget = None
    for name in STAGES:
        if name != args.through and not (output / "stages" / name / "complete.json").exists():
            raise ValueError("Run prerequisite in its own job: " + name)
        key = digest([identity, name, artifact_hashes(source) if source else None])
        def action(attempt):
            nonlocal budget
            if name == "publish":
                box = source / "box.json"
                box_hash = file_hash(box)
                load_box(box, box_hash)
                models = [{"name": "mcp-box", "method": "agentdistill", "checkpoint": args.snapshot,
                    "checkpoint_kind": "base", "student": config["models"]["student"],
                    "snapshot_artifact_sha256": identity["snapshot_artifact_sha256"],
                    "box": str(box), "box_sha256": box_hash, "label": "different interface",
                    "training_exposure": 0, "equivalence_required": True}]
                write_json(attempt / "output/models.json", models)
                write_json(attempt / "output/external.json", {str(output / "teacher"): artifact_hashes(output / "teacher")})
                return
            if budget is None:
                budget = Budget(output / "teacher", config, identity)
            if name == "collect":
                collect(tasks, config, output, attempt / "output", identity, budget)
            else:
                build(name, source, attempt / "output", tasks, budget, config)
            evidence = {str(output / "teacher/requests" / ("manager" if name == "collect" else name)):
                        artifact_hashes(output / "teacher/requests" / ("manager" if name == "collect" else name))}
            if name == "collect":
                evidence.update({str(output / "runs"): artifact_hashes(output / "runs"),
                    str(output / "teacher/requests/create"): artifact_hashes(output / "teacher/requests/create")})
            write_json(attempt / "output/external.json", evidence)
        source = stage_run(output, name, key, action)
        check_external(source)
        if name == "publish":
            models = read(source / "models.json")
            publish_models(output, models)
            write_json(output / "result.json", {"status": "complete", "models": models,
                "label": "different interface", "training_exposure": 0, "evaluation_selection": None})
        if args.through == name:
            return


def disclose(output, identity):
    output = Path(output)
    reservations = sorted((output / "runs").rglob("reservation.json"))
    attempts = []
    for reservation in reservations:
        directory = reservation.parent
        episode = read(directory / "episode/episode.json") if (directory / "episode/episode.json").exists() else {}
        receipt = read(directory / "receipt.json") if (directory / "receipt.json").exists() else {}
        attempts.append({**read(reservation), "path": str(directory), "status": receipt.get("status", "incomplete"),
            "success": episode.get("success", "unknown"), "environment_episodes": int(bool(episode)),
            "environment_resets_confirmed": int("initial" in episode),
            "environment_steps": episode_metrics(episode)["environment_steps"] if episode else 0,
            "unknown_costs_possible": not bool(receipt) or receipt.get("status") == "interrupted_unknown"})
    counter_path = output / "teacher/counter.json"
    counter = read(counter_path) if counter_path.exists() else {}
    usage_path = output / "teacher/usage.jsonl"
    usage, partial = [], False
    if usage_path.exists():
        import json
        for line in usage_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get("status") in {"success", "rejected", "incomplete"}:
                    usage.append(row)
            except json.JSONDecodeError:
                partial = True
    calls = counter.get("attempts", 0)
    write_json(output / "disclosure.json", {"method": "agentdistill", "label": "different interface",
        **({"baseline_transport": identity["baseline_transport"]} if "baseline_transport" in identity else {}),
        "K": identity["K"], "unique_instances": identity["unique_instances"], "task_groups": identity["task_groups"],
        "source": "verified_support_train_teacher_runs", "expansion": True, "trusted_trajectories_added": 0,
        "new_tasks": 0, "training_exposure": 0, "attempts": attempts,
        "teacher_cap": teacher_cap(identity["unique_instances"]), "max_runs_per_task": 3,
        "teacher_counter": counter, "teacher_usage_ledger": str(usage_path),
        "teacher_outcomes": ledger_counts(usage_path),
        "unknown_teacher_usage_possible": partial or counter.get("pending", False) or counter.get("halted", False),
        "unknown_usage_calls": calls - sum(r.get("usage") is not None for r in usage),
        "totals": {"teacher_calls": calls,
            **{key: sum(r[key] for r in attempts) for key in ("environment_episodes", "environment_steps", "environment_resets_confirmed")},
            **{"teacher_" + key: sum((r.get("usage") or {}).get(key, 0) or 0 for r in usage)
               for key in ("input_tokens", "output_tokens", "reasoning_tokens")}},
        "accounting": "Every reservation including interruption consumes a run. Raw prompts/outputs, rejections, "
            "returned model IDs, timestamps and usage are retained. Rewritten tool behavior remains unknown; "
            "static validation is not environment success. K counts original trusted executed seed trajectories."})


@contextmanager
def serve_base(args, attempt):
    # shared served() expects a *trained* merged checkpoint; the base-only
    # launcher uses the same harness process manager without --merged-model-path.
    command = [args.env_python, "harness/serve_student.py", "--config", args.config,
        "--python", args.vllm_python, "--model-path", args.snapshot, "--port", "0",
        "--ready-timeout", str(args.ready_timeout), "--output", str(attempt / "server")]
    with (attempt / "launcher.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ready = attempt / "server/ready.json"
            deadline = time.monotonic() + args.ready_timeout + 30
            while not ready.exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("AgentDistill base server not ready")
                time.sleep(1)
            yield read(ready)["endpoint"]
        finally:
            stop_server(process)


def evaluate(args, config):
    output = Path(args.output)
    identity = read(output / "identity.json")
    if (identity["config_sha256"] != digest(config) or identity["snapshot"] != args.snapshot
            or identity["verified"] != args.verified
            or digest(read(Path(args.verified) / "manifest.json")) != identity["collection_sha256"]
            or identity["snapshot_artifact_sha256"] != artifact_hashes(args.snapshot)
            or identity["code_sha256"] != code_hashes(Path(__file__).resolve().parents[2])):
        raise ValueError("AgentDistill evaluation identity changed")
    # Revalidate completed construction without invoking prepare, teacher, or any
    # generation stage. Missing/tampered artifacts fail before server startup.
    source = None
    for name in STAGES:
        key = digest([identity, name, artifact_hashes(source) if source else None])
        if not (output / "stages" / name / "complete.json").exists():
            raise ValueError("Publish the MCP box before evaluation")
        source = stage_run(output, name, key, lambda _: (_ for _ in ()).throw(ValueError("Missing stage")))
        check_external(source)
    models = read(output / "models.json")
    if models != read(source / "models.json"):
        raise ValueError("Published models changed")
    for model in models:
        load_box(model["box"], model["box_sha256"])
        def action(attempt):
            gate = ["-m", "baselines.agentdistill.gate", "--config", args.config,
                "--snapshot", args.snapshot, "--verified", args.verified,
                "--box", model["box"], "--box-sha256", model["box_sha256"]]
            run_command([args.train_python, *gate, "--hf-only", "--output", str(attempt / "hf")], attempt / "hf.log")
            with serve_base(args, attempt) as endpoint:
                server = str(attempt / "server/provenance.json")
                cpu = os.environ.copy()
                cpu["CUDA_VISIBLE_DEVICES"] = ""
                run_command([args.env_python, *gate, "--hf-reference", str(attempt / "hf/reference.json"),
                    "--endpoint", endpoint, "--server-provenance", server, "--output", str(attempt / "check")],
                    attempt / "check.log", env=cpu)
                run_command([args.env_python, "-m", "baselines.agentdistill.evaluation", "--config", args.config,
                    "--tasks", "data/manifest/dev140.json", "--endpoint", endpoint, "--server-provenance", server,
                    "--check", str(attempt / "check/report.json"), "--box", model["box"],
                    "--box-sha256", model["box_sha256"], "--workers", str(args.workers),
                    "--output", str(attempt / "output")], attempt / "eval.log", env=cpu)
        stage_run(output, "eval-" + model["name"], digest([identity, model]), action)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    for key in ("snapshot", "verified", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate"), default="train")
    parser.add_argument("--through", choices=STAGES, default="collect")
    parser.add_argument("--gpus", type=int, choices=(1,), default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=900)
    for key in ("env", "train", "vllm"):
        parser.add_argument("--" + key + "-python", default=os.environ.get(key.upper() + "_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    for key in ("snapshot", "verified", "output"):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    if args.workers < 1 or Path(args.output).is_relative_to(Path(args.verified)):
        parser.error("Positive workers and output outside VERIFIED required")
    config = load_config(args.config)
    if any(Path(args.output).is_relative_to(Path(p).resolve()) for p in
           ("data/verified", "reports/runs", config["paths"]["raw"], config["paths"]["verified"])):
        parser.error("Output must be outside protected data and reports/runs")
    if file_hash("docs/baselines/NEW_METHOD_SPECS_1019.md") != RECIPE["spec_sha256"]:
        raise ValueError("Pinned method specification changed")
    if (config["models"]["teacher"]["provider_model_id"] != "gpt-6-luna"
            or config["models"]["student"]["requested_repo"] != "google/gemma-4-12B-it"
            or not config["models"]["student"]["revision"].startswith("707f0a3b")
            or config["training"]["thinking_mode"] != "off" or config["training"]["student_step_token_cap"] != 1024
            or config["environment"]["max_episode_steps"] != 40):
        raise ValueError("AgentDistill requires pinned paper-1 models and H0")
    if args.mode != "evaluate":
        require_cpu()
    from train.modeling import snapshot_metadata
    snapshot_metadata(args.snapshot)
    if any((Path(args.snapshot) / name).exists() for name in ("training_state.json", "adapter_config.json")):
        raise ValueError("AgentDistill requires the original base snapshot, without a trained checkpoint or adapter")
    batch_invariant()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"):
        if args.mode == "evaluate":
            return evaluate(args, config)
        recipe = {**RECIPE, "runtime_dependencies": versions()}
        identity = prepare(args, config, method="agentdistill", recipe=recipe)
        try:
            if args.mode == "train":
                run_stages(args, identity, config)
        finally:
            disclose(args.output, identity)


if __name__ == "__main__":
    main()
