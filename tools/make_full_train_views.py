"""Materialize hash-bound stratified_nested_v2 prefixes; no teacher or environment."""
import argparse
from pathlib import Path
import sys

# Also support `python tools/make_full_train_views.py` in an uninstalled worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sed.manifest import load_frozen
from sed.seed_evidence import materialize_full_train_views


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[30, 100, 300, 500])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", default="data/manifest/alfworld_manifest.json")
    parser.add_argument("--freeze", default="data/manifest/alfworld_split_freeze.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest, freeze = load_frozen(args.manifest, args.freeze)
    views = materialize_full_train_views(args.source, manifest, freeze, args.ks,
                                         seed=args.seed, dry_run=args.dry_run)
    print(f"{'DRY RUN' if args.dry_run else 'CREATED'}: stratified_nested_v2, seed={args.seed}")
    print(f"Source: {Path(args.source).resolve()}")
    print(f"Verified universe: {views[0]['universe_size']}; full order sha256: {views[0]['full_order_sha256']}")
    print(f"Source manifest sha256: {views[0]['source_manifest_sha256']}")
    print(f"Source selection sha256: {views[0]['source_selection_sha256']}")
    types = views[0]["type_order"]
    print("| K | " + " | ".join(types) + " | Groups | Multi-trial groups |")
    print("| " + " | ".join(["---"] * (len(types) + 3)) + " |")
    for view in views:
        counts = [str(view["counts_by_task_type"][kind]) for kind in types]
        print("| " + " | ".join([str(view["K"]), *counts, str(view["selected_task_groups"]),
                                str(view["multi_trial_groups"])]) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
