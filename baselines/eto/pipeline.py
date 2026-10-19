"""Resumable ETO SFT -> (same K greedy rollouts -> pairs -> DPO) x 3.

Each stage publishes a hash-bound completion only on success. Interrupted
attempts are preserved. Resume never trusts a partial checkpoint or silently
uses stale results. Evaluation is a separate dev140-only mode.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from baselines.common.dose_control import validate_evaluation_init, materialize_init, model_record, control_fields
from baselines.common import REGISTERED_K
from baselines.common.explore import training_tasks
from harness.serve_student import stop_server
from sed.config import load_config
from sed.manifest import frozen_write, load_frozen
from sed.provenance import code_hashes, write_provenance
from sed.seed_evidence import checked_episode, load_verified_collection, artifact_hashes
from sed.storage import digest, file_hash, locked, write_json

RECIPE = {"training_seed": 0, "epochs": 3, "max_length": 32768,
          "sft": {"learning_rate": 2e-5, "scheduler": "cosine", "warmup_ratio": 0.03,
                  "original_batch": 64, "weight_decay": 0.0, "packing": "turn"},
          "dpo": {"scheduler": "constant_with_warmup", "warmup_ratio": 0.1,
                  "original_batch": 32, "weight_decay": 0.0, "scoring": "shared_history"},
          "iterations": [{"learning_rate": 1e-6, "beta": 0.1},
                         {"learning_rate": 5e-7, "beta": 0.5}, {"learning_rate": 5e-7, "beta": 0.5}]}


def stage_run(root, name, identity, action):
    directory = Path(root) / "stages" / name
    directory.mkdir(parents=True, exist_ok=True)
    completion = directory / "complete.json"
    if completion.exists():
        row = json.loads(completion.read_text())
        path = directory / row["attempt"]
        if row["identity"] != identity or artifact_hashes(path) != row["artifacts"]:
            raise ValueError("Completed stage identity/artifacts changed")
        return path / "output"
    previous = sorted(directory.glob("attempt-*"))
    attempt = directory / f"attempt-{len(previous):03d}"
    attempt.mkdir(exist_ok=False)
    write_json(attempt / "started.json", {"identity": identity, "previous_attempts": [p.name for p in previous],
                                         "status": "started", "timestamp": time.time()})
    try:
        action(attempt)
    except BaseException as exc:
        write_json(attempt / "failure.json", {"status": "failed", "error_type": type(exc).__name__})
        raise
    write_json(completion, {"identity": identity, "attempt": attempt.name,
                            "artifacts": artifact_hashes(attempt)}, exclusive=True)
    return attempt / "output"


def run_command(command, log, *, env=None):
    with Path(log).open("x") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            status = process.wait()
        except BaseException:
            stop_server(process)
            raise
    if status:
        raise RuntimeError("Stage subprocess failed; inspect its retained log")


def training_command(args, *, objective, snapshot, output, pairs=None, iteration=None, reference=None):
    recipe = dict(RECIPE[objective])
    if objective == "dpo":
        recipe.update(RECIPE["iterations"][iteration - 1])
    command = [args.train_python, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node", str(args.gpus), "-m", "baselines.common.fsdp_train",
               "--config", args.config, "--objective", objective, "--snapshot", str(snapshot),
               "--base-snapshot", args.snapshot, "--output", str(output),
               "--epochs", "3", "--seed", "0", "--max-length", "32768"]
    for key, value in recipe.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    if objective == "sft":
        command += ["--verified", args.verified]
    else:
        command += ["--pairs", str(pairs), "--reference", str(reference if reference is not None else snapshot)]
    return command


@contextmanager
def served(args, snapshot, attempt):
    env = os.environ.copy()  # Private inheritance only, never serialized.
    if env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"].split(",")[0]
    command = [args.env_python, "harness/serve_student.py", "--config", args.config,
               "--python", args.vllm_python, "--model-path", args.snapshot,
               "--merged-model-path", str(snapshot), "--port", "0",
               "--ready-timeout", str(args.ready_timeout), "--output", str(attempt / "server")]
    adapter = (Path(snapshot) / "adapter_config.json").exists()
    if adapter:
        index = command.index("--merged-model-path")
        command[index:index + 2] = ["--lora-adapter", "dose-policy=" + str(snapshot)]
        # Same unmerged HF/vLLM equivalence gate used for Seed-SFT evaluation.
        check = [args.train_python, "tools/check_lora_serving.py", "--config", args.config,
            "--snapshot", args.snapshot, "--adapter", str(snapshot), "--verified", args.verified,
            "--model-name", "dose-policy"]
        run_command(check + ["--hf-only", "--output", str(attempt / "hf")], attempt / "hf.log", env=env)
    with (attempt / "launcher.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            ready = attempt / "server/ready.json"
            deadline = time.monotonic() + args.ready_timeout + 30
            while not ready.exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Exploration server did not become ready")
                time.sleep(1)
            endpoint = json.loads(ready.read_text())["endpoint"]
            if adapter:
                run_command(check + ["--hf-reference", str(attempt / "hf/reference.json"), "--endpoint", endpoint,
                    "--server-provenance", str(attempt / "server/provenance.json"),
                    "--output", str(attempt / "check")], attempt / "check.log", env=env)
            yield endpoint
        finally:
            stop_server(process)


def prepare(args, config):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, selection, root = load_verified_collection(args.verified, manifest, freeze)
    if selection["K"] not in REGISTERED_K:
        raise ValueError("ETO few-shot K must be " + "/".join(map(str, REGISTERED_K)))
    parents = {r["task_id"]: [checked_episode(root, r)["episode_id"]] for r in collection["verified"]}
    ids = list(parents)
    listing = {"name": "eto_seed_train", "task_ids": ids, "parent_seed_ids": parents,
               "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze),
               "list_sha256": digest(ids)}
    training_tasks(listing, manifest, freeze)
    identity = {"recipe": RECIPE, "K": collection["K"], "support_seed": selection["seed"],
                "unique_instances": collection["unique_instances"], "task_groups": collection["task_groups"],
                "verified": str(Path(args.verified).resolve()), "collection_sha256": digest(collection),
                "snapshot": str(Path(args.snapshot).resolve()), "config_sha256": digest(config),
                "snapshot_artifact_sha256": artifact_hashes(args.snapshot),
                "gpus": args.gpus, "tasks_sha256": digest(listing),
                "code_sha256": code_hashes(Path(__file__).resolve().parents[2])}
    output = Path(args.output)
    from baselines.common.dose_control import validate_evaluation_init, bind_init
    bind_init(args, config, identity)
    output.mkdir(parents=True, exist_ok=True)
    frozen_write(output / "identity.json", identity)
    frozen_write(output / "tasks.json", listing)
    if not (output / "provenance.json").exists():
        write_provenance(output, command="eto_pipeline", config={**config, "eto": identity},
            inputs=[args.config, Path(args.verified) / "manifest.json", config["paths"]["manifest"],
                    config["paths"]["split_freeze"]])
    return identity


def update_disclosure(output, identity):
    """Include ALL exploration attempts, including errors and interrupted stages."""
    from sed.runner import episode_metrics
    output = Path(output)
    attempts = []
    for attempt in sorted((output / "stages").glob("explore-*/attempt-*")):
        summary = attempt / "output/summary.json"
        if summary.exists():
            value = json.loads(summary.read_text())
        else:
            episodes = [json.loads(p.read_text()) for p in (attempt / "output/episodes").glob("*/episode.json")]
            replays = [json.loads(p.read_text()) for p in (attempt / "output/episodes").glob("*/replay.json")]
            value = {"environment_episodes": len(episodes),
                     "environment_steps": sum(episode_metrics(e)["environment_steps"] for e in episodes),
                     "replay_environment_episodes": sum(r["environment_episodes"] for r in replays),
                     "replay_environment_steps": sum(r["environment_steps"] for r in replays),
                     "status": "incomplete", "unknown_costs_possible": True}
        attempts.append({"path": str(attempt.relative_to(output)), **value})
    totals = {key: sum(a.get(key, 0) for a in attempts) for key in
              ("environment_episodes", "environment_steps", "replay_environment_episodes", "replay_environment_steps")}
    write_json(output / "disclosure.json", {"K": identity["K"], "trusted_trajectories_added": 0,
              "seed_unique_instances": identity["unique_instances"], "seed_task_groups": identity["task_groups"],
              "source": "student_exploration", "expansion": True, "attempts": attempts, "totals": totals,
              **control_fields(identity),
              "accounting": "Replay is additional real execution. Interrupted attempts remain disclosed; "
                            "retrying tasks never increases K or unique tasks. Training exposure is in each stage result.json."})


def train_pipeline(args, identity):
    output = Path(args.output)
    root_identity = digest(identity)
    def run_train(name, snapshot, objective, pairs=None, iteration=None):
        return stage_run(output, name, digest([root_identity, name, str(snapshot), str(pairs)]),
                  lambda attempt: run_command(training_command(args, objective=objective, snapshot=snapshot,
                           output=attempt / "output", pairs=pairs, iteration=iteration,
                           **({"reference": init_snapshot} if "init" in identity and objective == "dpo" else {})), attempt / "train.log"))
    init_snapshot = (materialize_init(args, identity) if "init" in identity
                     else run_train("sft", args.snapshot, "sft") / "checkpoint")
    stages = [model_record(identity, "sft", init_snapshot)]
    publish_models(output, stages)
    if args.through == "sft":
        return
    snapshot = init_snapshot
    for iteration in range(1, 4):
        def explore(attempt):
            with served(args, snapshot, attempt) as endpoint:
                command = [args.env_python, "-m", "baselines.common.explore", "--config", args.config,
                           "--tasks", str(output / "tasks.json"), "--endpoint", endpoint,
                           "--model-name", args.model_name, "--decoding", "greedy", "--workers", str(args.workers),
                           "--server-provenance", str(attempt / "server/provenance.json"),
                           "--output", str(attempt / "output")]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""
                run_command(command, attempt / "explore.log", env=env)
        exploration = stage_run(output, f"explore-{iteration}", digest([root_identity, str(snapshot)]), explore)
        update_disclosure(output, identity)
        if args.through == f"explore-{iteration}":
            return
        def pairs_action(attempt):
            command = [args.env_python, "-m", "baselines.eto.pairs", "--config", args.config,
                       "--verified", args.verified, "--exploration", str(exploration), "--output", str(attempt / "output")]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            run_command(command, attempt / "pairs.log", env=env)
        pairs = stage_run(output, f"pairs-{iteration}", digest([root_identity, str(exploration)]), pairs_action)
        report = json.loads((pairs / "report.json").read_text())
        write_json(output / f"iteration-{iteration}.json", {"iteration": iteration, **report})
        if report["P"] == 0:
            write_json(output / "result.json", {"status": "stopped", "reason": f"no pairs at iteration {iteration}",
                                               "final_checkpoint": str(snapshot), "models": stages})
            return
        if args.through == f"pairs-{iteration}":
            return
        trained = run_train(f"dpo-{iteration}", snapshot, "dpo", pairs / "pairs.json", iteration)
        snapshot = trained / "checkpoint"
        stages.append(model_record(identity, f"iter-{iteration}", snapshot))
        publish_models(output, stages)
        if args.through == f"dpo-{iteration}":
            return
    write_json(output / "result.json", {"status": "complete", "final_checkpoint": str(snapshot), "models": stages})


def publish_models(output, stages):
    path = Path(output) / "models.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    models = {m["name"]: m for m in existing}
    models.update({m["name"]: m for m in stages})
    write_json(path, list(models.values()))


def evaluate_pipeline(args):
    output = Path(args.output)
    models = json.loads((output / "models.json").read_text())
    for model in models:
        checkpoint = Path(model["checkpoint"])
        state = json.loads((checkpoint / "training_state.json").read_text())
        hashes = artifact_hashes(checkpoint)
        hashes.pop("training_state.json")
        if hashes != state["artifact_sha256"]:
            raise ValueError("Evaluation checkpoint changed")
        def evaluate(attempt):
            env = os.environ.copy()
            env.update(PROJECT_ROOT=str(Path.cwd()), CONFIG=args.config, MODEL_PATH=args.snapshot,
                       MERGED_MODEL_PATH=str(checkpoint), FULL_CHECKPOINT_NAME=model["name"],
                       ENV_PYTHON=args.env_python, TRAIN_PYTHON=args.train_python, VLLM_PYTHON=args.vllm_python,
                       TASKS=str(Path("data/manifest/dev140.json").resolve()), OUT=str(attempt / "output"),
                       WORKERS=str(args.workers), INCLUDE_BASE="1", READY_TIMEOUT=str(args.ready_timeout))
            if env.get("CUDA_VISIBLE_DEVICES"):
                env["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"].split(",")[0]
            script = "adapters=(); source scripts/eval_adapters.sh"
            if model.get("adapter"):
                env.update(MERGED_MODEL_PATH="", INCLUDE_BASE="0", VERIFIED=args.verified,
                    DOSE_ADAPTER=model["name"] + "=" + str(checkpoint))
                script = 'adapters=("$DOSE_ADAPTER"); source scripts/eval_adapters.sh'
            run_command(["bash", "-c", script], attempt / "eval.log", env=env)
        stage_run(output, "eval-" + model["name"], digest(state), evaluate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--init-checkpoint", help="Final same-cell Seed-SFT LoRA adapter")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--verified", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["train", "evaluate", "prepare"], default="train")
    parser.add_argument("--through", choices=["all", "sft"] + [f"{s}-{i}" for i in range(1, 4)
                                                              for s in ("explore", "pairs", "dpo")], default="all")
    parser.add_argument("--gpus", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--env-python", default=os.environ.get("ENV_PYTHON", sys.executable))
    parser.add_argument("--train-python", default=os.environ.get("TRAIN_PYTHON", sys.executable))
    parser.add_argument("--vllm-python", default=os.environ.get("VLLM_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    args.output = str(Path(args.output).resolve())
    args.snapshot = str(Path(args.snapshot).resolve())
    args.verified = str(Path(args.verified).resolve())
    config = load_config(args.config)
    args.model_name = config["models"]["student"]["requested_repo"]
    for key in ("verified", "raw"):
        if Path(args.output).is_relative_to(Path(config["paths"][key]).resolve()):
            parser.error("Baseline output must be separate from original seed storage")
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"):
        if args.mode == "evaluate":
            validate_evaluation_init(args, config)
            evaluate_pipeline(args)
            return
        identity = prepare(args, config)
        try:
            if args.mode == "train":
                train_pipeline(args, identity)
        finally:
            update_disclosure(args.output, identity)


if __name__ == "__main__":
    main()
