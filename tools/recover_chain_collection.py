"""Audit/copy a chain collection offline; never launches teachers or environments."""
import argparse
import json
import sys

from sed.config import load_config
from sed.manifest import load_frozen
from teachers.chain_recovery import read_json, recover_chain_collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("adftp", "smartad"))
    parser.add_argument("--recover-from", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="Defaults to the exact frozen source config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from pathlib import Path
    config = load_config(args.config) if args.config else read_json(Path(args.source) / "identity.json")["config"]
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    result = recover_chain_collection(args.source, args.output, args.method, manifest, freeze, config,
                                      dry_run=args.dry_run, progress=lambda text: print(text, file=sys.stderr, flush=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
