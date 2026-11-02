"""Offline seed-only ALFWorld synthetic task factory."""

import argparse
from pathlib import Path

from repair.synth_tasks import build_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-seed", type=int, default=5)
    parser.add_argument("--g3-per-seed", type=int, help="Override G3 attempt count")
    parser.add_argument("--generators", default="G1,G2,G3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--main-checkout", type=Path, help="Read-only manifest source; inferred from --seeds")
    parser.add_argument("--heldout", action="append", type=Path, default=[])
    parser.add_argument("--validation-workers", type=int, default=4, help="CPU validation processes")
    args = parser.parse_args()
    main_checkout = args.main_checkout
    if main_checkout is None:
        main_checkout = next((p for p in args.seeds.resolve().parents if (p / "data/manifest/dev140.json").is_file()), None)
    if main_checkout is None:
        parser.error("Cannot locate dev140; supply --main-checkout")
    stats = build_factory(seeds_path=args.seeds, data_root=args.data_root, out=args.out,
        main_checkout=main_checkout, repo_root=Path(__file__).resolve().parents[1],
        per_seed=args.per_seed, g3_per_seed=args.g3_per_seed, generators=args.generators.split(","),
        seed=args.seed, extra_heldout=args.heldout, validation_workers=args.validation_workers)
    for generator in args.generators.split(","):
        print(generator, stats[generator], flush=True)


if __name__ == "__main__":
    main()
