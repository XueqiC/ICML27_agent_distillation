"""CPU FTP collection followed by the unchanged Kang AD training command."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

from baselines.adftp.collect import collect
from baselines.common.pipeline import batch_invariant, prepare, update_disclosure
from baselines.eto.pipeline import stage_run, run_command
from baselines.kang.pipeline import RECIPE as AD_RECIPE, training_command
from sed.config import load_config
from sed.manifest import load_frozen
from sed.storage import digest, locked, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    for name in ("verified", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--snapshot", help="Required for training/evaluation; not CPU collection")
    parser.add_argument("--mode", choices=["prepare", "train", "evaluate"], default="train")
    parser.add_argument("--through", choices=["collect", "train", "all"], default="collect")
    parser.add_argument("--rate-limit-dir", default="data/adftp_rate_limit")
    parser.add_argument("--subset-k", type=int, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--scoring", choices=["per_turn", "shared_history"], default="per_turn")
    for name in ("train", "env", "vllm"):
        parser.add_argument("--" + name + "-python", default=os.environ.get(name.upper() + "_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    if (args.through != "collect" or args.mode == "evaluate") and not args.snapshot:
        parser.error("Training/evaluation requires --snapshot")
    batch_invariant()
    for name in ("verified", "output", "snapshot"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    config = load_config(args.config)
    if args.mode == "evaluate":
        from baselines.common.evaluation import evaluate_models
        args.output = str(Path(args.output) / "training")
        with locked(Path(args.output) / ".pipeline.lock"):
            evaluate_models(args, decodings=("greedy", "sag"))
        return
    if args.through in {"collect", "all"}:
        if args.mode == "prepare":
            parser.error("Use --through train --mode prepare to prepare training; collection is explicit")
        manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
        collect(args.verified, manifest, freeze, config, args.output,
                rate_limit_dir=args.rate_limit_dir, subset_ks=args.subset_k,
                input_paths=[args.config, Path(args.verified) / "manifest.json",
                             config["paths"]["manifest"], config["paths"]["split_freeze"]])
        if args.through == "collect":
            return
    # GPU-only jobs consume the published collection; never enter the teacher stage.
    collection = Path(args.output) / "collection" if args.through == "all" else Path(args.verified)
    training = SimpleNamespace(**{**vars(args), "training_seed": 0,
                       "output": str(Path(args.output) / "training"), "verified": str(collection)})
    with locked(Path(training.output) / ".pipeline.lock"):
        identity = prepare(training, config, method="adftp", recipe={**AD_RECIPE, "first_thought_prefix": True,
                                                                   "scoring": args.scoring})
        try:
            if args.mode == "train":
                result = stage_run(training.output, "train-sft", digest([identity, "sft"]),
                    lambda a: run_command(training_command(training, a / "output"), a / "train.log"))
                checkpoint = str(result / "checkpoint")
                models = [{"name": "adftp-sft", "adapter": checkpoint, "checkpoint": checkpoint}]
                write_json(Path(training.output) / "models.json", models)
                write_json(Path(args.output) / "models.json", models)
                write_json(Path(training.output) / "result.json", {"status": "complete", "models": models})
        finally:
            update_disclosure(training.output, identity)


if __name__ == "__main__":
    main()
