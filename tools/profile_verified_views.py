"""Read-only CPU timing of real verified views; prints JSON, makes no API calls."""

import argparse
from collections import Counter
import cProfile
import json
from pathlib import Path
import time

from sed import seed_evidence
from sed.hash_receipts import cache_provenance
from sed.manifest import load_frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", default="data/manifest/alfworld_manifest.json")
    parser.add_argument("--freeze", default="data/manifest/alfworld_split_freeze.json")
    parser.add_argument("--ks", nargs="+", type=int, default=[30, 30, 500, 500])
    parser.add_argument("--profile", help="Save the first load's cProfile outside the source tree")
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    if args.profile and Path(args.profile).resolve().is_relative_to(source):
        parser.error("Profile output must be outside the source collection")
    manifest, freeze = load_frozen(args.manifest, args.freeze)
    original_hash, original_loads = seed_evidence.file_hash, json.loads
    stats = Counter()

    def measured_hash(path):
        start = time.perf_counter()
        value = original_hash(path)
        stats["artifact_and_receipt_hash_calls"] += 1
        stats["hash_thread_seconds"] += time.perf_counter() - start
        return value

    def measured_loads(*args, **kwargs):
        start = time.perf_counter()
        value = original_loads(*args, **kwargs)
        stats["json_loads_calls"] += 1
        stats["json_thread_seconds"] += time.perf_counter() - start
        return value

    seed_evidence.file_hash, json.loads = measured_hash, measured_loads
    try:
        for index, k in enumerate(args.ks):
            stats.clear()
            profiler = cProfile.Profile() if args.profile and index == 0 else None
            if profiler:
                profiler.enable()
            start = time.perf_counter()
            collection, _, _ = seed_evidence.load_verified_collection(source / f"subsets/k{k}", manifest, freeze)
            elapsed = time.perf_counter() - start
            if profiler:
                profiler.disable()
                profiler.dump_stats(args.profile)
            print(json.dumps({"K": k, "index": index, "wall_seconds": elapsed,
                              "profiled": profiler is not None, "hash_cache": cache_provenance(),
                              "trusted_trajectories": collection["trusted_trajectories"], **stats}), flush=True)
    finally:
        seed_evidence.file_hash, json.loads = original_hash, original_loads


if __name__ == "__main__":
    main()
