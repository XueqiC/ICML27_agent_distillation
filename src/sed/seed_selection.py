"""Versioned, immutable TRAIN group sampling; no trajectories are claimed here."""

import argparse
from collections import defaultdict
import random

from sed.manifest import TASK_TYPES, frozen_write, load_frozen, authorize_task, resolve_task
from sed.storage import digest

RULE = "stratified_groups_v1"


def select_seeds(manifest, freeze, *, k=10, seed=0, rule=RULE):
    if rule != RULE or type(k) is not int or k < 1 or type(seed) is not int:
        raise ValueError("Unknown selection rule or invalid K/seed")
    if freeze["manifest_sha256"] != digest(manifest):
        raise ValueError("Manifest differs from freeze")
    types = sorted(TASK_TYPES)
    random.Random(seed).shuffle(types)
    quotas = {kind: k // 6 + int(i < k % 6) for i, kind in enumerate(types)}
    groups = defaultdict(list)
    excluded_groups = {a["group_id"] for a in freeze["assignments"].values()
                       if a["split"] != "train" or a["sealed"]}
    pool = set(freeze["seed_pool_candidates"])
    for row in manifest["tasks"]:
        assignment = freeze["assignments"].get(row["task_id"], {})
        if (row["eligible"] and row["source_split"] == "train"
                and assignment.get("split") == "train" and not assignment.get("sealed", True)
                and row["task_id"] in pool and row["group_id"] not in excluded_groups):
            groups[row["group_id"]].append(row)
    candidates = {kind: [] for kind in types}
    for group, rows in sorted(groups.items()):
        if len({r["task_type"] for r in rows}) != 1:
            raise ValueError("Task group spans multiple types")
        # One instance per group; sibling trials are never replacements.
        row = min(rows, key=lambda r: (digest([seed, "trial", r["task_id"]]), r["task_id"]))
        kind = row["task_type"]
        candidates[kind].append({key: row[key] for key in
                                 ("task_id", "group_id", "task_type", "content_sha256")})
    selected, replacements = {}, {}
    for kind in types:
        rows = sorted(candidates[kind], key=lambda r: r["group_id"])
        random.Random(digest([seed, "groups", kind])).shuffle(rows)
        if len(rows) < quotas[kind]:
            raise ValueError(f"Insufficient TRAIN task groups for {kind}")
        selected[kind], replacements[kind] = rows[:quotas[kind]], rows[quotas[kind]:]
    result = dict(schema_version=1, selection_rule=rule, seed=seed, K=k,
                  K_counts="trusted_executed_trajectories", trusted_trajectories=0,
                  selected_unique_instances=k, selected_task_groups=k,
                  split="train", source_split="train", type_order=types, quotas=quotas,
                  selected=selected, replacements=replacements,
                  manifest_sha256=digest(manifest), split_freeze_sha256=digest(freeze),
                  external_resource="frozen ALFWorld TRAIN pool; one trial per group")
    result["selection_sha256"] = digest(result)
    return result


def validate_selection(selection, manifest, freeze):
    expected = select_seeds(manifest, freeze, k=selection["K"], seed=selection["seed"],
                            rule=selection["selection_rule"])
    if selection != expected:
        raise ValueError("Seed selection differs from its frozen TRAIN sampling rule")
    for rows in [*selection["selected"].values(), *selection["replacements"].values()]:
        for row in rows:
            authorize_task(resolve_task(manifest, freeze, row["task_id"]), "generation")


def validate_extension_selection(small, large):
    """Changing K may move the selected/replacement boundary, never candidates."""
    keys = ("seed", "selection_rule", "type_order", "manifest_sha256", "split_freeze_sha256")
    if (small["K"] >= large["K"] or any(small[key] != large[key] for key in keys)
            or any(small["quotas"][kind] > large["quotas"][kind]
                   or small["selected"][kind] != large["selected"][kind][:small["quotas"][kind]]
                   or small["selected"][kind] + small["replacements"][kind]
                   != large["selected"][kind] + large["replacements"][kind]
                   for kind in small["type_order"])):
        raise ValueError("Selections are not nested per-type prefixes with identical candidate order")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest/alfworld_manifest.json")
    parser.add_argument("--freeze", default="data/manifest/alfworld_split_freeze.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rule", choices=[RULE], default=RULE)
    parser.add_argument("--output")
    parser.add_argument("--extend-selection", help="Check the smaller frozen selection before writing")
    args = parser.parse_args(argv)
    manifest, freeze = load_frozen(args.manifest, args.freeze)
    selected = select_seeds(manifest, freeze, k=args.k, seed=args.seed, rule=args.rule)
    if args.extend_selection:
        import json
        from pathlib import Path
        small = json.loads(Path(args.extend_selection).read_text())
        validate_selection(small, manifest, freeze)
        validate_extension_selection(small, selected)
    path = args.output or f"data/manifest/seeds_k{args.k}_s{args.seed}.json"
    frozen_write(path, selected)
    print(f"Selected {args.k} TRAIN groups; verified trajectories: 0; {path}")
