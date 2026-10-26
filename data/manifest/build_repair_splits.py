"""Freeze repair v0.1 task lists from metadata only; never open an environment.

Exclusions use selected seed/chain tasks and all named R03 tasks/groups.
Candidate and replacement declarations do not establish training exposure.
Missing required sources are errors, never silently empty exclusions.
"""

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath

from sed.manifest import TASK_TYPES, frozen_write, load_frozen
from sed.storage import digest, file_hash

KINDS = sorted(TASK_TYPES)
CHAIN = "reports/runs/adftp-chain/chain_k500_s0_r2/collection"
LEGACY_CHAIN = "reports/runs/adftp-chain/chain_k500_s0/collection"
SEED_GRID = tuple(f"seeds_k{k}_s{s}.json" for k in (5, 10, 20, 100) for s in range(3))
ORDER_RULE = "ascending sha256 of canonical JSON [seed, task_id], task_id tie break"


def selected_task_ids(value, tasks):
    """Read only the selected mapping and/or selected task_order list."""
    if not isinstance(value, dict) or not {"selected", "task_order"} & value.keys():
        raise ValueError("Selection source needs selected or task_order")
    selected, order = None, None
    if "selected" in value:
        if not isinstance(value["selected"], dict):
            raise ValueError("selected must be a mapping of task lists")
        selected = []
        for rows in value["selected"].values():
            if not isinstance(rows, list):
                raise ValueError("selected must be a mapping of task lists")
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("task_id"), str) or row["task_id"] not in tasks:
                    raise ValueError("Selected task absent from the frozen manifest")
                task = tasks[row["task_id"]]
                if row.get("group_id", task["group_id"]) != task["group_id"]:
                    raise ValueError("Selected group differs from the frozen manifest")
                selected.append(row["task_id"])
    if "task_order" in value:
        order = value["task_order"]
        if not isinstance(order, list) or any(not isinstance(t, str) or t not in tasks for t in order):
            raise ValueError("Selected task_order task absent from the frozen manifest")
    if selected is not None and order is not None and Counter(selected) != Counter(order):
        raise ValueError("Selected task_order differs from selected mapping")
    return set(selected if selected is not None else order)


def named_identities(value, tasks):
    """R03 only: include every named task/group, including group-only entries."""
    known_groups = {t["group_id"] for t in tasks.values()}
    ids, groups = set(), set()

    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"task_id", "group_id"}:
                    known = tasks if key == "task_id" else known_groups
                    if not isinstance(child, str) or child not in known:
                        raise ValueError("Source names a task/group absent from the frozen manifest")
                visit(key)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            if item in tasks:
                ids.add(item)
                groups.add(tasks[item]["group_id"])
            elif item in known_groups:
                groups.add(item)
            elif item.startswith(("train/", "valid_seen/", "valid_unseen/")):
                raise ValueError("Source names a task/group absent from the frozen manifest")

    visit(value)
    return ids, groups


def read_source(path, category, manifest, freeze, *, selected_only=True):
    path = Path(path)
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        for key, expected in (("manifest_sha256", digest(manifest)),
                              ("split_freeze_sha256", digest(freeze))):
            if key in value and value[key] != expected:
                raise ValueError(f"Source differs from frozen inputs: {path}")
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    if selected_only:
        ids = selected_task_ids(value, tasks)
        groups = {tasks[t]["group_id"] for t in ids}
    else:
        ids, groups = named_identities(value, tasks)
    ids, groups = sorted(ids), sorted(groups)
    return {"path": str(path), "category": category, "sha256": file_hash(path),
            "extraction_rule": "selected/task_order only" if selected_only else "all named tasks/groups",
            "task_ids": ids, "list_sha256": digest(ids), "unique_instances": len(ids),
            "group_ids": groups, "group_list_sha256": digest(groups), "task_groups": len(groups)}


def exclusion_paths(*, seed_dir, chain_selection, chain_subsets, legacy_chain_collection, r03_paths):
    """Only selection.json views; never collection manifests or provenance."""
    for name in SEED_GRID:
        if not (Path(seed_dir) / name).is_file():
            raise FileNotFoundError(Path(seed_dir) / name)
    seeds = sorted(Path(seed_dir).glob("seeds_k*_s*.json"))

    def view_selections(directory):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        views = sorted(p for p in directory.glob("k*") if p.is_dir())
        if not views:
            raise ValueError(f"No chain subset selection views: {directory}")
        for view in views:
            yield view / "selection.json"
            if (view / "subsets").is_dir():
                yield from view_selections(view / "subsets")

    paths = [(Path(chain_selection), "k500_chain_selection"),
             *[(p, "k500_chain_subset") for p in view_selections(Path(chain_subsets))]]
    legacy = Path(legacy_chain_collection)
    if (legacy / "selection.json").exists():
        paths.append((legacy / "selection.json", "legacy_k500_chain_selection"))
    if (legacy / "subsets").exists():
        paths.extend((p, "legacy_k500_chain_subset") for p in view_selections(legacy / "subsets"))
    paths.extend((p, "seed_selection") for p in seeds)
    paths.extend((Path(p), "r03_pool_surrogate_or_dev") for p in r03_paths)
    for path, _ in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def select(candidates, quotas, seed):
    counts, groups, chosen = Counter(), set(), []
    for task in sorted(candidates, key=lambda t: (digest([seed, t["task_id"]]), t["task_id"])):
        kind = task["task_type"]
        if counts[kind] < quotas[kind] and task["group_id"] not in groups:
            chosen.append(task)
            counts[kind] += 1
            groups.add(task["group_id"])
    return chosen


def task_listing(name, tasks, quotas, seed, manifest, freeze, *, purpose):
    ids = [t["task_id"] for t in tasks]
    counts = {k: sum(t["task_type"] == k for t in tasks) for k in KINDS}
    return {"schema_version": 1, "name": name, "split": "train", "source_split": "train",
            "purpose": purpose, "training_eligible": purpose != "evaluation_only",
            "seed": seed, "selection_rule": ORDER_RULE,
            "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze),
            "task_ids": ids, "list_sha256": digest(ids),
            "group_ids": [t["group_id"] for t in tasks],
            "unique_instances": len(ids), "task_groups": len({t["group_id"] for t in tasks}),
            "quotas": quotas, "counts_by_task_type": counts,
            "shortfalls": {k: quotas[k] - counts[k] for k in KINDS}}


def population_counts(tasks):
    return {"unique_instances": len(tasks), "task_groups": len({t["group_id"] for t in tasks}),
            "instances_by_task_type": {k: sum(t["task_type"] == k for t in tasks) for k in KINDS},
            "groups_by_task_type": {k: len({t["group_id"] for t in tasks if t["task_type"] == k}) for k in KINDS}}


def build_splits(manifest, freeze, sources, dev140):
    if freeze["manifest_sha256"] != digest(manifest):
        raise ValueError("Manifest differs from freeze")
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    if len(tasks) != len(manifest["tasks"]):
        raise ValueError("Duplicate manifest task IDs")
    for t in tasks.values():
        if t["group_id"] != str(PurePosixPath(t["task_id"]).parent):
            raise ValueError("Task group must be the directory without the trial suffix")
    dev_ids = set(dev140["task_ids"])
    if (dev140["list_sha256"] != digest(dev140["task_ids"])
            or dev140["manifest_sha256"] != digest(manifest)
            or dev140["split_freeze_sha256"] != digest(freeze)
            or not dev_ids <= tasks.keys()):
        raise ValueError("dev140 differs from frozen inputs")
    excluded = set().union(*(set(s["task_ids"]) for s in sources))
    if not excluded <= tasks.keys():
        raise ValueError("Unknown exclusion task")
    groups = lambda ids: {tasks[t]["group_id"] for t in ids}
    sealed = {t for t, a in freeze["assignments"].items() if a["sealed"] or a["split"] == "test"}
    eval_ids = {t for t, a in freeze["assignments"].items() if a["split"] != "train"} | dev_ids
    excluded_groups = groups(excluded) | set().union(*(set(s.get("group_ids", [])) for s in sources))
    if not excluded_groups <= {t["group_id"] for t in tasks.values()}:
        raise ValueError("Unknown exclusion group")
    blocked_groups = excluded_groups | groups(eval_ids)
    universe = [t for t in tasks.values() if t["eligible"] and t["source_split"] == "train"
                and freeze["assignments"].get(t["task_id"], {}).get("split") == "train"
                and not freeze["assignments"][t["task_id"]]["sealed"]]
    universe_groups = {t["group_id"] for t in universe}
    if universe_groups & groups(dev_ids | sealed):
        raise ValueError("TRAIN universe overlaps dev140 or sealed groups")
    eligible = [t for t in universe if t["group_id"] not in blocked_groups]
    sources = [{**s, "excluded_train_groups": len(universe_groups &
                (groups(s["task_ids"]) | set(s.get("group_ids", []))))} for s in sources]
    probe_quotas, quick_quotas = dict.fromkeys(KINDS, 10), dict.fromkeys(KINDS, 4)
    mech_quotas = dict(zip(KINDS, (4, 4, 3, 3, 3, 3)))
    probe = select(eligible, probe_quotas, 20261026)
    # First four per type in D_probe hash order, then keep the global hash order.
    quick = select(probe, quick_quotas, 20261026)
    probe_groups = {t["group_id"] for t in probe}
    mech = select([t for t in eligible if t["group_id"] not in probe_groups], mech_quotas, 20261027)
    sets = {name: task_listing(name, rows, quotas, seed, manifest, freeze, purpose=purpose)
            for name, rows, quotas, seed, purpose in (
                ("D_probe", probe, probe_quotas, 20261026, "evaluation_only"),
                ("D_unseen_quick", quick, quick_quotas, 20261026, "evaluation_only"),
                ("U_mech", mech, mech_quotas, 20261027, "mechanism_training_pool"))}
    overlaps = {}
    for name, value in sets.items():
        ids, gs = set(value["task_ids"]), set(value["group_ids"])
        for other, other_ids, other_groups in (("E", excluded, excluded_groups),
                ("dev140", dev_ids, groups(dev_ids)), ("sealed", sealed, groups(sealed))):
            overlaps[f"{name}_vs_{other}"] = {"task_ids": len(ids & other_ids),
                                             "task_groups": len(gs & other_groups)}
    overlaps["D_probe_vs_U_mech"] = {"task_ids": len(set(sets["D_probe"]["task_ids"]) & set(sets["U_mech"]["task_ids"])),
                                    "task_groups": len(probe_groups & set(sets["U_mech"]["group_ids"]))}
    if any(n for pair in overlaps.values() for n in pair.values()):
        raise ValueError("Repair split overlap")
    result = {"schema_version": 1, "name": "repair_v01_splits",
        "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze),
        "selection_rule": {"universe": "eligible frozen TRAIN assignments with source_split=train",
            "exclusions": "groups of selected/task_order seed and K500 chain tasks plus every named R03 task/group; no candidate, replacement or full-order fields",
            "ordering": ORDER_RULE, "deduplication": "one trial per game/scene directory (group_id)",
            "shortfall": "take all available groups of that type; no redistribution",
            "D_unseen_quick": "first four of each type in D_probe hash order (up to 24)"},
        "universe": population_counts(universe),
        "eligible_after_exclusions": population_counts(eligible),
        "exclusion_sources": sources,
        "E": {"task_ids": sorted(excluded), "list_sha256": digest(sorted(excluded)),
              "unique_instances": len(excluded), "task_groups": len(excluded_groups),
              "group_ids": sorted(excluded_groups), "group_list_sha256": digest(sorted(excluded_groups)),
              "excluded_train_groups": len(excluded_groups & universe_groups)},
        "sets": sets, "assertions": {"all_passed": True, "overlap_counts": overlaps,
            "quick_subset_of_probe": set(sets["D_unseen_quick"]["task_ids"]) <= set(sets["D_probe"]["task_ids"]),
            "one_trial_per_group": all(v["unique_instances"] == v["task_groups"] for v in sets.values()),
            "train_only": True, "universe_disjoint_from_dev140_and_sealed": True},
        "new_teacher_calls": 0, "new_environment_steps": 0, "training_exposure": 0}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest/alfworld_manifest.json")
    p.add_argument("--freeze", default="data/manifest/alfworld_split_freeze.json")
    p.add_argument("--seed-dir", default="data/manifest")
    p.add_argument("--full-train", default="data/manifest/seeds_full_train.json",
                   help="candidate declaration recorded as considered_not_excluded")
    p.add_argument("--chain-selection", default=f"{CHAIN}/selection.json")
    p.add_argument("--chain-subsets", default=f"{CHAIN}/subsets")
    p.add_argument("--legacy-chain-collection", default=LEGACY_CHAIN,
                   help="also include this older chain's selection files when present")
    p.add_argument("--r03", nargs="+", default=["data/manifest/r03_pool200.json",
        "data/manifest/r03_surrogate40.json", "data/manifest/r03_surrogate.json",
        "data/manifest/r03_dev300.json"])
    p.add_argument("--dev140", default="data/manifest/dev140.json")
    p.add_argument("--output", default="data/manifest/repair_v01_splits.json")
    args = p.parse_args(argv)
    manifest, freeze = load_frozen(args.manifest, args.freeze)
    paths = exclusion_paths(seed_dir=args.seed_dir, chain_selection=args.chain_selection,
        chain_subsets=args.chain_subsets, legacy_chain_collection=args.legacy_chain_collection,
        r03_paths=args.r03)
    sources = [read_source(path, category, manifest, freeze,
        selected_only=category != "r03_pool_surrogate_or_dev") for path, category in paths]
    full_train = read_source(args.full_train, "full_train_candidate_declaration", manifest, freeze,
                             selected_only=False)
    result = build_splits(manifest, freeze, sources, json.loads(Path(args.dev140).read_text()))
    result["considered_not_excluded"] = [{
        "path": full_train["path"], "sha256": full_train["sha256"], "category": full_train["category"],
        "declared_unique_instances": full_train["unique_instances"],
        "declared_task_groups": full_train["task_groups"], "excluded_groups": 0,
        "reason": "seeds_full_train.json declares candidates for full-TRAIN teacher collection, not actual baseline training exposure"}]
    result["optional_legacy_chain"] = {
        "collection": args.legacy_chain_collection,
        "selection_sources_found": [str(path) for path, category in paths if category.startswith("legacy_")]}
    result.update(builder_sha256=file_hash(__file__), rebuild_arguments=vars(args),
        input_sha256={str(p): file_hash(p) for p in (args.manifest, args.freeze, args.dev140)})
    result["content_sha256"] = digest(result)  # Entire document except this self-hash field.
    frozen_write(args.output, result)
    print(json.dumps({"output": args.output, "sha256": file_hash(args.output),
        "universe": result["universe"],
        "excluded_groups_per_source": [{k: s[k] for k in ("path", "sha256", "task_groups", "excluded_train_groups")}
                                        for s in result["exclusion_sources"]],
        "excluded_groups_union": result["E"]["task_groups"],
        "excluded_train_groups_union": result["E"]["excluded_train_groups"],
        "eligible_after_exclusions": result["eligible_after_exclusions"],
        "counts": {k: v["unique_instances"] for k, v in result["sets"].items()},
        "counts_by_task_type": {k: v["counts_by_task_type"] for k, v in result["sets"].items()},
        "shortfalls": {k: v["shortfalls"] for k, v in result["sets"].items()}}, indent=2))
    return result


if __name__ == "__main__":
    main()
