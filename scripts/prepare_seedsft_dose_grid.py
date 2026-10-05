"""CPU-only, local-tokenizer preparation; no training, API calls or job submission."""

import argparse
import json
import os
from pathlib import Path

from sed.config import load_config
from sed.manifest import load_frozen, frozen_write
from sed.provenance import code_hashes, versions, utc_now
from sed.storage import file_hash
from train.dataset import load_tokenizer
from train.dose_grid import prepare_grid
from train.modeling import snapshot_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="configs/seedsft_dose_grid_v2.json")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--snapshot", required=True, help="Existing pinned local student/tokenizer snapshot")
    parser.add_argument("--data-root", default=".", help="Base directory for the grid's verified paths")
    parser.add_argument("--output", required=True, help="Fresh final grid JSON to use at submission")
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("Final grid output already exists; use a fresh path")
    if Path(args.output).resolve().is_relative_to(Path("data/verified").resolve()):
        parser.error("The grid must be written outside data/verified")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_config(args.config)
    metadata = snapshot_metadata(args.snapshot)
    tokenizer = load_tokenizer(args.snapshot)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    template = json.loads(Path(args.template).read_text())
    grid = prepare_grid(template, tokenizer, manifest, freeze, data_root=args.data_root)
    grid["preparation"] = {"timestamp": utc_now(), "student": config["models"]["student"],
                           "snapshot": metadata, "dependencies": versions(),
                           "code_sha256": code_hashes(Path(__file__).resolve().parents[1]),
                           "inputs": {str(p): file_hash(p) for p in (args.template, args.config,
                               config["paths"]["manifest"], config["paths"]["split_freeze"])},
                           "packing": "episode", "max_length": 32768}
    frozen_write(args.output, grid)
    print(json.dumps({"output": args.output, "runs": len(grid["runs"]), "status": grid["status"]}))


if __name__ == "__main__":
    main()
