"""Separate SmartAD CPU collection, GPU selection, LoRA training and dev140 evaluation."""
import argparse
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

from baselines.common.pipeline import batch_invariant, prepare, update_disclosure
from baselines.eto.pipeline import stage_run, run_command
from baselines.kang.pipeline import RECIPE as AD_RECIPE, training_command as ad_command
from baselines.smartad.collect import collect, validate_h0, validate_shard
from baselines.smartad.paths import PathResolver
from baselines.smartad.select import select, validate_selection
from sed.config import load_config
from sed.manifest import load_frozen
from sed.storage import digest, locked, write_json

RECIPE = {k: v for k, v in AD_RECIPE.items() if k not in {"sag", "first_thought_prefix"}}
RECIPE.update(original_batch=4, loss="segment_weighted_token_mean_over_global_batch",
              segment_weights={"reason": 1, "action": 1.5, "final": 2},
              selection="initial_student_macro_turn_NLL", decoding="greedy")


def training_command(args, output):
    command = ad_command(args, output)
    command[command.index("--objective") + 1] = "smartad"
    command[command.index("--original-batch") + 1] = "4"
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--verified", required=True, help="Trusted seeds (collect), pool (select), selected collection (train/evaluate)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                        help="Selection only: relocate an absolute evidence prefix (repeatable; disjoint paths)")
    parser.add_argument("--through", "--stage", dest="stage",
                        choices=["collect", "select", "train", "evaluate"], default="collect")
    parser.add_argument("--mode", choices=["prepare", "train", "evaluate"], default="train")
    parser.add_argument("--rate-limit-dir", default="data/smartad_rate_limit")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Reuse the frozen collection and durable candidate slots")
    parser.add_argument("--subset-k", type=int, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    parser.add_argument("--ready-timeout", type=int, default=900)
    for name in ("train", "env", "vllm"):
        parser.add_argument("--" + name + "-python", default=os.environ.get(name.upper() + "_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.mode == "evaluate":
        args.stage = "evaluate"
    if args.path_map and args.stage != "select":
        parser.error("--path-map is for selection only")
    try:
        PathResolver(args.path_map)
    except ValueError as error:
        parser.error(str(error))
    if args.stage != "collect" and not args.snapshot:
        parser.error("Select/train/evaluate require --snapshot")
    if args.mode == "prepare" and args.stage != "train":
        parser.error("Preparation is for training only")
    if args.subset_k and args.stage != "collect":
        parser.error("--subset-k is frozen at collection; views publish after selection")
    if args.stage != "collect" and (args.shard_count != 1 or args.shard_index != 0 or args.resume):
        parser.error("--shard-index/--shard-count/--resume are for collection only")
    try:
        validate_shard(args.shard_index, args.shard_count)
    except ValueError as error:
        parser.error(str(error))
    for name in ("verified", "output", "snapshot"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    batch_invariant()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    config = load_config(args.config)
    validate_h0(config)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    if args.stage == "collect":
        return collect(args.verified, manifest, freeze, config, args.output,
            rate_limit_dir=args.rate_limit_dir, subset_ks=args.subset_k,
            shard_index=args.shard_index, shard_count=args.shard_count, resume=args.resume,
            input_paths=[args.config, Path(args.verified) / "manifest.json",
                         config["paths"]["manifest"], config["paths"]["split_freeze"]])
    if args.stage == "select":
        return select(args.verified, args.snapshot, config, manifest, freeze, args.output, path_map=args.path_map)
    validate_selection(args.verified, manifest, freeze, args.snapshot)
    training = SimpleNamespace(**{**vars(args), "gpus": 1, "training_seed": 0,
                                  "output": str(Path(args.output) / "training")})
    with locked(Path(training.output) / ".pipeline.lock"):
        if args.stage == "evaluate":
            from baselines.common.evaluation import evaluate_models
            return evaluate_models(training, decodings=("greedy",))
        identity = prepare(training, config, method="smartad", recipe={**RECIPE, "epochs": args.epochs})
        try:
            if args.mode == "train":
                result = stage_run(training.output, "train-sft", digest([identity, "sft"]),
                    lambda a: run_command(training_command(training, a / "output"), a / "train.log"))
                checkpoint = str(result / "checkpoint")
                models = [{"name": "smartad-sft", "adapter": checkpoint, "checkpoint": checkpoint}]
                write_json(Path(training.output) / "models.json", models)
                write_json(Path(args.output) / "models.json", models)
                write_json(Path(training.output) / "result.json", {"status": "complete", "models": models})
        finally:
            update_disclosure(training.output, identity)


if __name__ == "__main__":
    main()
