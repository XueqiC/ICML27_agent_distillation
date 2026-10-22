"""Resumable Kang seed SFT. Evaluation is separate through hpg_eval_adapters."""
import argparse
import math
import os
from pathlib import Path
import signal
import sys

from baselines.common.pipeline import batch_invariant, prepare, prepare_training, require_cpu_preparation, update_disclosure
from baselines.eto.pipeline import run_command, stage_run
from sed.config import load_config
from sed.storage import digest, locked, write_json
from sed.startup_timing import startup_timing

RECIPE = {"training_seed": 0, "epochs": 2, "max_length": 32768, "adaptation": "kang_lora",
    "rank": 64, "alpha": 128, "dropout": 0.05, "targets": "q/k/v/o/gate/up/down under .language_model.",
    "dropout_equivalence": "expected_gradient", "rng_seed_policy": "seed + rank",
    "learning_rate": 2e-4, "scheduler": "linear", "warmup_ratio": 0.0, "warmup_steps": 0, "original_batch": 8,
    "weight_decay": 0.0, "gradient_clip": 1.0, "packing": "turn", "scoring": "per_turn", "loss": "assistant_only",
    "first_thought_prefix": False, "sag": {"n": 8, "temperature": 0.4, "top_p": 0.8, "max_tokens": 1024}}


def optimizer_recipe(args):
    """Shared, identity-bound stability overrides; paper settings remain defaults."""
    learning_rate = getattr(args, "learning_rate", RECIPE["learning_rate"])
    warmup_ratio = getattr(args, "warmup_ratio", RECIPE["warmup_ratio"])
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio <= 1:
        raise ValueError("--warmup-ratio must be finite and between 0 and 1")
    return {"learning_rate": learning_rate, "warmup_ratio": warmup_ratio,
            "warmup_steps": 0 if warmup_ratio == 0 else None}


def training_command(args, output):
    optimizer = optimizer_recipe(args)
    learning_rate = "2e-4" if optimizer["learning_rate"] == RECIPE["learning_rate"] else str(optimizer["learning_rate"])
    warmup_ratio = "0" if optimizer["warmup_ratio"] == 0 else str(optimizer["warmup_ratio"])
    training_seed = getattr(args, "training_seed", RECIPE["training_seed"])
    return [args.train_python, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", str(getattr(args, "gpus", 1)), "-m", "baselines.common.fsdp_train", "--config", args.config,
        "--objective", "sft", "--adaptation", "kang_lora", "--snapshot", args.snapshot,
        "--base-snapshot", args.snapshot, "--verified", args.verified, "--output", str(output),
        "--epochs", str(getattr(args, "epochs", RECIPE["epochs"])), "--seed", str(training_seed), "--max-length", "32768", "--learning-rate", learning_rate,
        "--scheduler", "linear", "--warmup-ratio", warmup_ratio,
        *(["--warmup-steps", "0"] if optimizer["warmup_steps"] == 0 else []),
        "--original-batch", "8", "--weight-decay", "0", "--packing", RECIPE["packing"],
        "--scoring", getattr(args, "scoring", "per_turn"),
        *(["--pipeline-output", str(Path(args.output).resolve())] if hasattr(args, "output") else [])]


def train_pipeline(args, identity):
    trained = stage_run(args.output, "train-sft", digest([identity, "sft"]),
        lambda a: run_command(training_command(args, a / "output"), a / "train.log"))
    model = {"name": "kang-sft", "adapter": str(trained / "checkpoint")}
    write_json(Path(args.output) / "models.json", [model])
    write_json(Path(args.output) / "result.json", {"status": "complete", **model,
        "evaluation": "scripts/hpg_eval_adapters.slurm with DECODING=sag, BATCH_INVARIANT=1; equivalence gate required",
        "first_thought_prefix": "not run: missing released end-to-end generation components"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    for name in ("snapshot", "verified", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=["prepare", "train"], default="train")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    parser.add_argument("--learning-rate", type=float, default=RECIPE["learning_rate"])
    parser.add_argument("--warmup-ratio", type=float, default=RECIPE["warmup_ratio"])
    parser.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--scoring", choices=["per_turn", "shared_history"], default="per_turn")
    parser.add_argument("--train-python", default=os.environ.get("TRAIN_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        require_cpu_preparation()
    try:
        optimizer = optimizer_recipe(args)
    except ValueError as error:
        parser.error(str(error))
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    batch_invariant()
    for name in ("snapshot", "verified", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    config = load_config(args.config)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"), startup_timing(
            Path(args.output) / "startup_timing.jsonl", process="pipeline", mode=args.mode, snapshot=args.snapshot):
        identity = prepare(args, config, method="kang", recipe={**RECIPE, **optimizer, "training_seed": args.training_seed,
                                                              "epochs": args.epochs,
                                                              "scoring": args.scoring})
        try:
            if args.mode == "train":
                train_pipeline(args, identity)
            else:
                prepare_training(args, training_command)
        finally:
            update_disclosure(args.output, identity)


if __name__ == "__main__":
    main()
