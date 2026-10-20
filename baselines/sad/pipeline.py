"""Hash-bound SAD full tuning and all-checkpoint dev140 evaluation."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

from baselines.common.pipeline import batch_invariant, prepare, update_disclosure
from baselines.eto.pipeline import run_command, stage_run, publish_models
from baselines.sad.data import TOKEN_BUDGET, DENOMINATORS
from sed.config import load_config
from sed.storage import digest, locked, write_json

RECIPE = {"training_seed": 0, "adaptation": "full_parameter_text_fsdp2", "packing": "turn",
    "scoring": "shared_history",
    "learning_rate": 2e-5, "scheduler": "cosine", "warmup_ratio": .03, "original_batch": 64,
    "weight_decay": 0., "gradient_clip": 1., "max_length": 32768,
    "loss": "batch mean of summed one-hot CE on Thought and Action + turn-end spans",
    "lambda_r": 1, "lambda_a": 1, "alpha": 1, "beta": 1, "gamma": 0,
    "curriculum": "stable ascending token complexity; same order each epoch",
    "processed_token_budget": TOKEN_BUDGET, "checkpoint_denominators": list(DENOMINATORS)}


def training_command(args, output):
    return [args.train_python, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", str(args.gpus), "-m", "baselines.common.fsdp_train",
        "--config", args.config, "--objective", "sad", "--snapshot", args.snapshot,
        "--base-snapshot", args.snapshot, "--verified", args.verified, "--output", str(output),
        "--seed", "0", "--max-length", "32768", "--learning-rate", "2e-5",
        "--scheduler", "cosine", "--warmup-ratio", ".03", "--original-batch", "64",
        "--weight-decay", "0", "--packing", "turn", "--processed-token-budget", str(TOKEN_BUDGET),
        "--scoring", args.scoring]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    for name in ("snapshot", "verified", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=["prepare", "train", "evaluate"], default="train")
    parser.add_argument("--through", choices=["prepare", "train", "all"], default="all")
    parser.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--scoring", choices=["per_turn", "shared_history"], default=RECIPE["scoring"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=900)
    for name in ("train", "env", "vllm"):
        parser.add_argument("--" + name + "-python", default=os.environ.get(name.upper() + "_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    batch_invariant()
    for name in ("snapshot", "verified", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    config = load_config(args.config)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"):
        if args.mode == "evaluate":
            from baselines.common.evaluation import evaluate_models
            evaluate_models(args)
            return
        identity = prepare(args, config, method="sad", recipe={**RECIPE, "scoring": args.scoring})
        try:
            if args.mode == "train" and args.through != "prepare":
                trained = stage_run(args.output, "train-sad", digest([identity, "sad"]),
                    lambda a: run_command(training_command(args, a / "output"), a / "train.log"))
                models = json.loads((trained / "models.json").read_text())
                publish_models(args.output, models)
                write_json(Path(args.output) / "result.json", {"status": "complete", "models": models})
        finally:
            update_disclosure(args.output, identity)


if __name__ == "__main__":
    main()
