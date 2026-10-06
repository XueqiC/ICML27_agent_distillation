"""BC=train-0; AgentEvol trains D_0..D_3 from base and produces untrained D_4."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

from baselines.common.dose_control import validate_evaluation_init, materialize_init, model_record
from baselines.agentgym.data import prepare_dataset
from baselines.common.pipeline import batch_invariant, prepare, update_disclosure
from baselines.common.schedule import dose
from baselines.eto.pipeline import stage_run, run_command, served, publish_models, evaluate_pipeline
from sed.config import load_config
from sed.storage import digest, locked, write_json

RECIPE = {"training_seed": 0, "epochs": 1, "max_length": 32768, "adaptation": "full",
    "learning_rate": 1e-5, "scheduler": "linear", "warmup": "int(0.1 * total_steps)",
    "original_batch": 32, "weight_decay": 0.0, "gradient_clip": 1.0, "packing": "turn",
    "trained_iterations": [0, 1, 2, 3], "untrained_dataset": 4,
    "exploration": "one greedy rollout per seed TRAIN task; batch invariant"}


def publish_datasets(output, datasets):
    path = output / "datasets.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    merged = {row["iteration"]: row for row in existing}
    merged.update({row["iteration"]: row for row in datasets})
    write_json(path, [merged[i] for i in sorted(merged)])


def training_command(args, *, data, output, n_examples):
    plan = dose(n_examples, 32, epochs=1)
    return [args.train_python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.gpus),
        "-m", "baselines.common.fsdp_train", "--config", args.config, "--objective", "sft",
        "--snapshot", args.snapshot, "--base-snapshot", args.snapshot, "--sft-data", str(data),
        "--output", str(output), "--epochs", "1", "--seed", "0", "--max-length", "32768",
        "--learning-rate", "1e-5", "--scheduler", "linear", "--warmup-ratio", "0.1",
        "--warmup-steps", str(int(0.1 * plan["total_steps"])), "--original-batch", "32", "--weight-decay", "0",
        "--packing", RECIPE["packing"]]


def train_pipeline(args, identity, config):
    output, root_identity = Path(args.output), digest(identity)
    data_root = stage_run(output, "data-0", digest([root_identity, "D_0"]),
        lambda a: prepare_dataset(args.verified, a / "output", config))
    data = data_root / "train_iter_0.json"
    datasets, models = [{"iteration": 0, "path": str(data)}], []
    publish_datasets(output, datasets)
    for m in range(4):
        n = len(json.loads(data.read_text()))
        if "init" in identity and m == 0:
            snapshot = materialize_init(args, identity)
            state = {"examples_seen": 0, "supervised_tokens_seen": 0}
        else:
            def command(attempt):
                if "init" in identity:
                    cmd = [args.train_python, "-m", "baselines.agentgym.dose_train", "--config", args.config,
                        "--snapshot", args.snapshot, "--verified", args.verified, "--sft-data", str(data),
                        "--identity", str(output / "identity.json"), "--output", str(attempt / "output")]
                else:
                    cmd = training_command(args, data=data, output=attempt / "output", n_examples=n)
                run_command(cmd, attempt / "train.log")
            trained = stage_run(output, f"train-{m}", digest([root_identity, m, str(data)]), command)
            state = json.loads((trained / "result.json").read_text())
            snapshot = trained / (state["checkpoint"] if "init" in identity else "checkpoint")
        models.append(model_record(identity, "bc" if m == 0 else f"iter-{m}", snapshot,
            **({"adapter": str(snapshot)} if "init" in identity and m > 0 else {})))
        publish_models(output, models)
        iteration = {"iteration": m, "D_m": n, "F_m": None, "training_start": args.snapshot,
            "examples_seen": state["examples_seen"], "supervised_tokens_seen": state["supervised_tokens_seen"],
            "dose": dose(n, 32, epochs=1), "checkpoint": str(snapshot)}
        if "init" in identity:
            iteration.update(training_start=identity["init"]["path"] if m == 0 else args.snapshot,
                dose={"passes": identity["init"]["passes"], "recipe": identity["init"]["recipe"],
                      "realized_passes": state.get("passes"), "reused_init": m == 0})
        # Do not erase an already completed filter's counts when resuming train-m.
        record = output / f"iteration-{m}.json"
        if not record.exists():
            write_json(record, iteration)
        if args.through == f"train-{m}":
            return
        def explore(attempt):
            with served(args, snapshot, attempt) as endpoint:
                command = [args.env_python, "-m", "baselines.common.explore", "--config", args.config,
                    "--tasks", str(output / "tasks.json"), "--endpoint", endpoint, "--model-name",
                    "dose-policy" if "init" in identity and m > 0 else args.model_name,
                    "--decoding", "greedy", "--workers", str(args.workers), "--server-provenance",
                    str(attempt / "server/provenance.json"), "--output", str(attempt / "output")]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""
                run_command(command, attempt / "explore.log", env=env)
        exploration = stage_run(output, f"explore-{m}", digest([root_identity, m, str(snapshot)]), explore)
        if args.through == f"explore-{m}":
            return
        filtered = stage_run(output, f"filter-{m}", digest([root_identity, m, str(data), str(exploration)]),
            lambda a: prepare_dataset(args.verified, a / "output", config,
                previous=data, exploration=exploration, iteration=m + 1))
        report = json.loads((filtered / "report.json").read_text())
        write_json(record, {**iteration, "F_m": report["F_m"], "D_next": report["D_next"],
            "empty_F": report["empty_F"], "filter_report": str(filtered / "report.json")})
        data = filtered / f"train_iter_{m + 1}.json"
        datasets.append({"iteration": m + 1, "path": str(data)})
        publish_datasets(output, datasets)
        update_disclosure(output, identity)
        if args.through == f"filter-{m}":
            return
    write_json(output / "result.json", {"status": "complete", "models": models,
        "train_iter_4": str(data), "train_iter_4_trained": False, "datasets": datasets})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--init-checkpoint", help="Final same-cell Seed-SFT LoRA adapter")
    for name in ("snapshot", "verified", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=["prepare", "train", "evaluate"], default="train")
    parser.add_argument("--through", choices=["all"] + [f"{s}-{i}" for i in range(4)
                        for s in ("train", "explore", "filter")], default="all")
    parser.add_argument("--gpus", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=900)
    for name in ("env", "train", "vllm"):
        parser.add_argument(f"--{name}-python", default=os.environ.get(name.upper() + "_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("AgentGym requires positive workers")
    batch_invariant()
    for name in ("snapshot", "verified", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    config = load_config(args.config)
    args.model_name = config["models"]["student"]["requested_repo"]
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"):
        if args.mode == "evaluate":
            validate_evaluation_init(args, config)
            os.environ["DECODING"] = "greedy"
            evaluate_pipeline(args)
            return
        identity = prepare(args, config, method="agentgym", recipe=RECIPE)
        try:
            if args.mode == "train":
                train_pipeline(args, identity, config)
        finally:
            update_disclosure(args.output, identity)


if __name__ == "__main__":
    main()
