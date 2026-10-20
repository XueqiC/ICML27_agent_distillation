"""Versioned, immutable TRAIN group sampling; no trajectories are claimed here."""

import argparse
from collections import Counter, defaultdict
import random

from sed.manifest import TASK_TYPES, frozen_write, load_frozen, authorize_task, resolve_task
from sed.storage import digest

RULE = "stratified_groups_v1"
ALL_TRAIN_RULE = "all_train_v1"
NESTED_RULE = "stratified_nested_v2"
NESTED_GRID_K = (30, 100, 300, 500, 1000, 2000)


def is_stratified_nested(selection):
    return selection.get("selection_rule") == NESTED_RULE


def stratified_nested_order(source, *, seed=0):
    """Order verified instances only; no episode content or candidate failures.

    Integer deficits avoid floating-point tie drift. Sorting before each seeded
    shuffle makes the order independent of receipt/dict traversal order.
    """
    if type(seed) is not int:
        raise ValueError("Invalid nested support seed")
    groups, seen, group_types = defaultdict(lambda: defaultdict(list)), set(), {}
    for receipt in source["verified"]:
        row = {key: receipt[key] for key in ("task_id", "group_id", "task_type")}
        tid, group, kind = row["task_id"], row["group_id"], row["task_type"]
        if tid in seen or kind not in TASK_TYPES or group_types.setdefault(group, kind) != kind:
            raise ValueError("Nested universe requires unique instances and single-type groups")
        seen.add(tid)
        groups[kind][group].append(row)
    if not seen:
        raise ValueError("Nested universe has no verified trajectories")
    types = sorted(TASK_TYPES)
    queues = {}
    for kind in types:
        group_order = sorted(groups[kind])
        random.Random(digest([seed, "groups", kind])).shuffle(group_order)
        rows = []
        for group in group_order:
            trials = sorted(groups[kind][group], key=lambda r: r["task_id"])
            random.Random(digest([seed, "instances", kind, group])).shuffle(trials)
            rows.append(trials)
        queues[kind] = [trials[round_] for round_ in range(max(map(len, rows), default=0))
                        for trials in rows if round_ < len(trials)]
    totals = {kind: len(queues[kind]) for kind in types}
    counts, order, size = dict.fromkeys(types, 0), [], len(seen)
    for n in range(1, size + 1):
        # max keeps the first type in the recorded fixed order on exact ties.
        kind = max((t for t in types if counts[t] < totals[t]),
                   key=lambda t: n * totals[t] - size * counts[t])
        order.append(queues[kind][counts[kind]])
        counts[kind] += 1
    return order


def select_nested_verified(source, large, *, k, seed=0, order=None):
    """A prefix of a complete all_train_v1 verified universe (not candidates)."""
    if source.get("status") != "complete" or not is_full_train(large):
        raise ValueError("Nested v2 requires a complete all_train_v1 source")
    if type(seed) is not int or type(k) is not int or not 1 <= k <= len(source["verified"]):
        raise ValueError("Nested K must be positive and at most the verified universe size")
    order = stratified_nested_order(source, seed=seed) if order is None else order
    types = sorted(TASK_TYPES)
    selected = {kind: [r for r in order[:k] if r["task_type"] == kind] for kind in types}
    counts = {kind: len(rows) for kind, rows in selected.items()}
    groups = Counter(r["group_id"] for r in order[:k])
    result = dict(schema_version=1, selection_rule=NESTED_RULE, seed=seed, K=k,
                  K_counts="trusted_executed_trajectories", trusted_trajectories=k,
                  selected_unique_instances=k, selected_task_groups=len(groups),
                  multi_trial_groups=sum(n > 1 for n in groups.values()),
                  split="train", source_split="train", type_order=types,
                  quotas=counts, counts_by_task_type=counts, selected=selected,
                  replacements={kind: [] for kind in types},
                  task_order=[r["task_id"] for r in order[:k]],
                  full_order_sha256=digest([r["task_id"] for r in order]),
                  universe_size=len(order),
                  source_manifest_sha256=digest(source), source_selection_sha256=digest(large),
                  manifest_sha256=large["manifest_sha256"], split_freeze_sha256=large["split_freeze_sha256"],
                  external_resource="verified full TRAIN trajectories; breadth-first sibling trials per type")
    result["selection_sha256"] = digest(result)
    return result


def is_full_train(selection):
    return selection.get("selection_rule") == ALL_TRAIN_RULE


def ordered_candidates(selection):
    """Global full-TRAIN order, also used for deterministic process sharding."""
    return sorted((r for rows in selection["selected"].values() for r in rows),
                  key=lambda r: (digest([selection["seed"], r["task_id"]]), r["task_id"]))


def select_seeds(manifest, freeze, *, k=10, seed=0, rule=RULE):
    if rule not in (RULE, ALL_TRAIN_RULE) or type(k) is not int or k < 1 or type(seed) is not int:
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
    if rule == ALL_TRAIN_RULE:
        rows = sorted((r for group in groups.values() for r in group),
                      key=lambda r: (digest([seed, r["task_id"]]), r["task_id"]))
        selected = {kind: [{key: r[key] for key in
                    ("task_id", "group_id", "task_type", "content_sha256")}
                    for r in rows if r["task_type"] == kind] for kind in sorted(TASK_TYPES)}
        result = dict(schema_version=1, selection_rule=rule, seed=seed, K=len(rows),
                      K_counts="candidates", candidates=len(rows), trusted_trajectories=0,
                      selected_unique_instances=len(rows), selected_task_groups=len(groups),
                      split="train", source_split="train", type_order=sorted(TASK_TYPES),
                      quotas={kind: len(items) for kind, items in selected.items()},
                      selected=selected, replacements={kind: [] for kind in selected},
                      candidate_order=[r["task_id"] for r in rows],
                      manifest_sha256=digest(manifest), split_freeze_sha256=digest(freeze),
                      external_resource="whole eligible frozen ALFWorld TRAIN split; sibling trials included",
                      accounting="Selection K counts candidates only. Trusted K is the number of "
                                 "successfully collected, execution-verified trajectories; no replacements.")
        result["selection_sha256"] = digest(result)
        return result
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


def select_full_train(manifest, freeze):
    """Same all_train_v1 schema as the concurrently developed full-TRAIN collector."""
    return select_seeds(manifest, freeze, rule=ALL_TRAIN_RULE)


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
    parser.add_argument("--rule", choices=[RULE, ALL_TRAIN_RULE], default=RULE)
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
    path = args.output or ("data/manifest/seeds_full_train.json" if is_full_train(selected)
                           else f"data/manifest/seeds_k{args.k}_s{args.seed}.json")
    frozen_write(path, selected)
    print(f"Selected {selected['selected_unique_instances']} TRAIN candidates in "
          f"{selected['selected_task_groups']} groups; verified trajectories: 0; {path}")
