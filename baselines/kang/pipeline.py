"""Resumable Kang seed SFT. Evaluation is separate through hpg_eval_adapters."""
import argparse
import os
from pathlib import Path
import signal
import sys

from baselines.common.pipeline import batch_invariant, prepare, update_disclosure
from baselines.eto.pipeline import run_command, stage_run
from sed.config import load_config
from sed.storage import digest, locked, write_json

RECIPE = {"training_seed": 0, "epochs": 2, "max_length": 32768, "adaptation": "kang_lora",
    "rank": 64, "alpha": 128, "dropout": 0.05, "targets": "q/k/v/o/gate/up/down under .language_model.",
    "learning_rate": 2e-4, "scheduler": "linear", "warmup_steps": 0, "original_batch": 8,
    "weight_decay": 0.0, "gradient_clip": 1.0, "packing": "turn", "loss": "assistant_only",
    "first_thought_prefix": False, "sag": {"n": 8, "temperature": 0.4, "top_p": 0.8, "max_tokens": 1024}}


def training_command(args, output):
    return [args.train_python, "-m", "baselines.common.fsdp_train", "--config", args.config,
        "--objective", "sft", "--adaptation", "kang_lora", "--snapshot", args.snapshot,
        "--base-snapshot", args.snapshot, "--verified", args.verified, "--output", str(output),
        "--epochs", "2", "--seed", "0", "--max-length", "32768", "--learning-rate", "2e-4",
        "--scheduler", "linear", "--warmup-ratio", "0", "--warmup-steps", "0",
        "--original-batch", "8", "--weight-decay", "0", "--packing", RECIPE["packing"]]


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
    parser.add_argument("--train-python", default=os.environ.get("TRAIN_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    args.gpus = 1
    batch_invariant()
    for name in ("snapshot", "verified", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    config = load_config(args.config)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"):
        identity = prepare(args, config, method="kang", recipe=RECIPE)
        try:
            if args.mode == "train":
                train_pipeline(args, identity)
        finally:
            update_disclosure(args.output, identity)


if __name__ == "__main__":
    main()
