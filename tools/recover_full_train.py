"""Offline full-TRAIN recovery or read-only classification; never calls a teacher."""
import argparse
import json
from pathlib import Path

from sed.config import load_config
from sed.manifest import load_frozen
from sed.storage import write_json
from teachers.full_train_recovery import DEFAULT_SOURCE, recover_full_train, separate_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default="data/verified/seeds_full_train_r2")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", help="Write audit summary outside source/output (e.g. logs/...)")
    args = parser.parse_args(argv)
    source, output = separate_paths(args.source, args.output)
    if args.summary:
        summary = Path(args.summary).resolve()
        separate_paths(source, summary)
        if summary.is_relative_to(output):
            parser.error("Summary must be outside the recovered collection")
        if args.dry_run and summary.is_relative_to(Path("data/verified").resolve()):
            parser.error("--dry-run cannot write to data/verified")
    config = load_config(args.config)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    result = recover_full_train(source, output, manifest, freeze, config, dry_run=args.dry_run)
    if args.summary:
        write_json(summary, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
