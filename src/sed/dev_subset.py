"""Immutable seed-0 dev40 derived only from frozen valid_seen assignments."""

import argparse
from collections import Counter

from sed.manifest import TASK_TYPES, frozen_write, load_frozen
from sed.storage import digest


def build_dev40(manifest, freeze, seed=0):
    kinds = sorted(TASK_TYPES, key=lambda t: (digest([seed, "quota", t]), t))
    quotas = {kind: 7 if i < 4 else 6 for i, kind in enumerate(kinds)}
    # Sealed game files/observations are never read by this builder.
    candidates = [t for t in manifest["tasks"] if t["source_split"] == "valid_seen"
                  and t["eligible"] and freeze["assignments"].get(t["task_id"], {}).get("split") == "dev"
                  and not freeze["assignments"][t["task_id"]]["sealed"]]
    chosen = []
    for kind in kinds:
        rows = sorted((t for t in candidates if t["task_type"] == kind),
                      key=lambda t: (digest([seed, "game", t["task_id"]]), t["task_id"]))
        if len(rows) < quotas[kind]:
            raise ValueError(f"Not enough frozen dev games for {kind}")
        chosen.extend(rows[:quotas[kind]])
    chosen.sort(key=lambda t: (digest([seed, "order", t["task_id"]]), t["task_id"]))
    ids = [t["task_id"] for t in chosen]
    return {"schema_version": 1, "name": "dev40", "split": "dev", "source_split": "valid_seen",
            "seed": seed, "selection_rule": "sha256 canonical JSON [seed, quota/game/order, value]; ascending",
            "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze),
            "task_ids": ids, "list_sha256": digest(ids), "quotas": quotas,
            "unique_instances": len(ids), "task_groups": len({t["group_id"] for t in chosen}),
            "counts_by_task_type": dict(sorted(Counter(t["task_type"] for t in chosen).items()))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest/alfworld_manifest.json")
    parser.add_argument("--freeze", default="data/manifest/alfworld_split_freeze.json")
    parser.add_argument("--output", default="data/manifest/dev40.json")
    args = parser.parse_args(argv)
    manifest, freeze = load_frozen(args.manifest, args.freeze)
    subset = build_dev40(manifest, freeze, seed=0)
    frozen_write(args.output, subset)
    print(f"{len(subset['task_ids'])} dev games; list_sha256={subset['list_sha256']}")


if __name__ == "__main__":
    main()
