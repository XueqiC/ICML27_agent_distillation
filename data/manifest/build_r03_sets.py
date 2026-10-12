"""Frozen R0–R3 lists and explicit, allowlisted episode-dataset loaders."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from sed.dev_subset import build_dev140
from sed.manifest import TASK_TYPES, authorize_task, frozen_write, load_frozen, resolve_task
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import digest, file_hash

FORMAT = "r03_episode_dataset_v1"


def guarded_task(manifest, freeze, task_id, *, generation=True):
    task = resolve_task(manifest, freeze, task_id)
    # diagnosis rejects sealed tasks even though evaluation would authorize them.
    authorize_task(task, "generation" if generation else "diagnosis")
    if task.get("sealed") or task["source_split"] == "valid_unseen":
        raise ValueError("R0–R3 never uses sealed games")
    return task


def listing(name, tasks, manifest, freeze, **extra):
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task instances")
    counts = dict(sorted(Counter(t["task_type"] for t in tasks).items()))
    return {"schema_version": 1, "name": name, "seed": 0, "task_ids": ids,
            "list_sha256": digest(ids), "manifest_sha256": digest(manifest),
            "split_freeze_sha256": digest(freeze), "unique_instances": len(ids),
            "group_ids": [t["group_id"] for t in tasks],
            "task_groups": len({t["group_id"] for t in tasks}),
            "counts_by_task_type": counts, "quotas": counts, **extra}


def stratified(candidates, count, excluded, tag):
    kinds = sorted(TASK_TYPES, key=lambda k: (digest([0, tag, "quota", k]), k))
    chosen, groups = [], set(excluded)
    for i, kind in enumerate(kinds):
        quota = count // len(kinds) + (i < count % len(kinds))
        rows = sorted((t for t in candidates if t["task_type"] == kind),
                      key=lambda t: (digest([0, tag, t["task_id"]]), t["task_id"]))
        selected = []
        for task in rows:
            if len(selected) == quota:
                break
            if task["group_id"] not in groups:
                groups.add(task["group_id"])
                selected.append(task)
        if len(selected) != quota:
            raise ValueError(f"Insufficient distinct playable groups: {kind}")
        chosen.extend(selected)
    # Round robin makes the 40/80/120 prefixes stratified too.
    by_kind = {k: [t for t in chosen if t["task_type"] == k] for k in kinds}
    return [by_kind[k][i] for i in range(max(map(len, by_kind.values()), default=0))
            for k in kinds if i < len(by_kind[k])]


def surrogate_subset(tasks, count=40):
    """Proportional largest-remainder quotas with seed-0 hash tie breaks/sampling."""
    if not 0 < count <= len(tasks):
        raise ValueError("Surrogate subset size exceeds available trajectories")
    counts = Counter(t["task_type"] for t in tasks)
    quotas = {kind: count * n // len(tasks) for kind, n in counts.items()}
    order = sorted(counts, key=lambda k: (-(count * counts[k] % len(tasks)),
                                         digest([0, "surrogate40", "quota", k]), k))
    for kind in order[:count - sum(quotas.values())]:
        quotas[kind] += 1
    chosen = []
    for kind in sorted(counts):
        rows = sorted((t for t in tasks if t["task_type"] == kind),
                      key=lambda t: (digest([0, "surrogate40", t["task_id"]]), t["task_id"]))
        chosen.extend(rows[:quotas[kind]])
    return sorted(chosen, key=lambda t: t["task_id"])


def build_sets(manifest, freeze, k100, k20, dev140, *, pool_count=200, extra_dev_count=160):
    if freeze["manifest_sha256"] != digest(manifest):
        raise ValueError("Manifest differs from freeze")
    if dev140 != build_dev140(manifest, freeze):
        raise ValueError("dev140 changed")
    large = {r["task_id"]: r for r in k100["verified"]}
    small = {r["task_id"]: r for r in k20["verified"]}
    if len(large) != 100 or len(small) != 20 or not small.keys() <= large.keys():
        raise ValueError("Require nested K100 and K20 verified collections")
    for task_id, receipt in small.items():
        if receipt["episode_hashes"] != large[task_id]["episode_hashes"]:
            raise ValueError("K20 is not the identical executed trajectory in K100")
    seed_tasks = [guarded_task(manifest, freeze, t) for t in large]
    seed_groups = {t["group_id"] for t in seed_tasks}
    candidates = [guarded_task(manifest, freeze, t["task_id"]) for t in manifest["tasks"]
                  if t["eligible"] and t["source_split"] == "train"
                  and freeze["assignments"].get(t["task_id"], {}).get("split") == "train"]
    pool = stratified(candidates, pool_count, seed_groups, "pool")
    heldout = stratified(candidates, extra_dev_count,
                         seed_groups | {t["group_id"] for t in pool}, "dev")
    dev = [guarded_task(manifest, freeze, t, generation=False) for t in dev140["task_ids"]] + heldout
    surrogate = [guarded_task(manifest, freeze, t) for t in sorted(large.keys() - small.keys())]
    if set(Counter(t["task_type"] for t in surrogate).values()) - {13, 14}:
        raise ValueError("Surrogate must have 13–14 trajectories per type")
    sets = {
        "surrogate": listing("r03_surrogate", surrogate, manifest, freeze, split="train",
            source_split="train", purpose="surrogate_only", training_eligible=False,
            trusted_trajectories=80, selection_rule="verified K100 s0 minus identical K20 s0 trajectories"),
        "surrogate40": listing("r03_surrogate40", surrogate_subset(surrogate), manifest, freeze,
            split="train", source_split="train", purpose="surrogate_only", training_eligible=False,
            trusted_trajectories=40, parent_listing="r03_surrogate.json",
            parent_list_sha256=digest([t["task_id"] for t in surrogate]),
            selection_rule="proportional largest remainder; seed-0 sha256 quota ties and task sampling; task_id order"),
        "pool200": listing("r03_pool200", pool, manifest, freeze, split="train", source_split="train",
            selection_rule="seed-0 hash stratification; one game/group; round-robin type order",
            subsets={"R1_P2": [t["task_id"] for t in pool[:40]],
                     "R1_P3": [t["task_id"] for t in pool[:80]],
                     "R2": [t["task_id"] for t in pool[:120]]}),
        "dev300": listing("r03_dev300", dev, manifest, freeze, split="dev",
            source_split="valid_seen+train", purpose="evaluation_only", training_eligible=False,
            selection_rule="unchanged dev140 prefix + seed-0 stratified train holdout",
            dev140_list_sha256=dev140["list_sha256"], train_holdout_ids=[t["task_id"] for t in heldout]),
    }
    group_sets = [seed_groups, {t["group_id"] for t in pool}, {t["group_id"] for t in dev}]
    overlaps = {f"{a}_{b}": len(group_sets[i] & group_sets[j])
                for i, a in enumerate(("k100", "pool", "dev"))
                for j, b in enumerate(("k100", "pool", "dev")) if i < j}
    if any(overlaps.values()):
        raise ValueError("R03 group overlap")
    for value in sets.values():
        value["group_overlap_counts"] = overlaps
        value["source_collection_sha256"] = {"k100": digest(k100), "k20": digest(k20)}
    return sets


def dataset_entry(root, episode, *, source, supervised_turns=None, parent_seed_ids=()):
    """Same schema-2 episode files as verified seeds; separate label eligibility."""
    root = Path(root)
    key = digest([episode["episode_id"], source, supervised_turns])
    path = root / "episodes" / key / "episode.json"
    frozen_write(path, episode)
    return {"episode_file": str(path.relative_to(root)), "episode_sha256": file_hash(path),
            "task_id": episode["task_id"], "source": source,
            "parent_seed_ids": list(parent_seed_ids), "supervised_turns": supervised_turns}


def publish_dataset(root, entries, manifest, freeze, *, source, purpose="training", **extra):
    tasks = [guarded_task(manifest, freeze, e["task_id"]) for e in entries]
    value = {"schema_version": 1, "format": FORMAT, "source": source, "purpose": purpose,
             "training_eligible": purpose == "training", "episodes": entries,
             "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze),
             "episode_count": len(entries), "unique_instances": len({t["task_id"] for t in tasks}),
             "task_groups": len({t["group_id"] for t in tasks}),
             "training_exposure": 0, **extra}
    frozen_write(Path(root) / "manifest.json", value)
    return value


def load_r03_dataset(root, tokenizer, manifest, freeze, *, max_length=32768,
                     packing="episode", purpose="training", task_ids=None):
    """Rows feed train.sft.turn_loss_sum; never supervise arbitrary audit fields."""
    from train.dataset import episode_examples
    root = Path(root).resolve()
    collection = json.loads((root / "manifest.json").read_text())
    if (collection.get("format") != FORMAT or collection["purpose"] != purpose
            or purpose not in {"training", "surrogate_only"}
            or collection["manifest_sha256"] != digest(manifest)
            or collection["split_freeze_sha256"] != digest(freeze)):
        raise ValueError("Dataset purpose or frozen inputs differ")
    entries = collection["episodes"]
    if task_ids is not None:
        if purpose != "surrogate_only" or not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("Require unique surrogate task IDs")
        by_task = {e["task_id"]: e for e in entries}
        if len(by_task) != len(entries) or not set(task_ids) <= by_task.keys():
            raise ValueError("Surrogate task IDs missing or ambiguous in dataset")
        entries = [by_task[t] for t in task_ids]
    examples = []
    for entry in entries:
        path = (root / entry["episode_file"]).resolve()
        if not path.is_relative_to(root) or file_hash(path) != entry["episode_sha256"]:
            raise ValueError("Episode path/hash differs")
        episode = json.loads(path.read_text())
        task = guarded_task(manifest, freeze, episode["task_id"])
        if entry["task_id"] != task["task_id"] or episode["split"] != task["split"]:
            raise ValueError("Episode differs from frozen task")
        turns = entry["supervised_turns"]
        if turns is not None:
            # turn packing masks the ENTIRE preceding student conversation.
            if turns != [len(episode["steps"]) - 1]:
                raise ValueError("Correction dataset must supervise exactly its final turn")
            rows = episode_examples(episode, tokenizer, max_length=max_length, packing="turn",
                                    allow_empty_visible=True)
            rows = [rows[turns[0]]]
        else:
            rows = episode_examples(episode, tokenizer, max_length=max_length, packing=packing,
                                    allow_empty_visible=entry["source"] == "student_real_environment_win")
        for row in rows:
            row["label_provenance"].update(source=entry["source"],
                training_eligible=collection["training_eligible"], parent_seed_ids=entry["parent_seed_ids"],
                dataset_manifest_sha256=file_hash(root / "manifest.json"))
        examples.extend(rows)
    return examples


def load_surrogate_examples(path, tokenizer, manifest, freeze, **kwargs):
    path = Path(path)
    listing_value = json.loads(path.read_text())
    root = path.parent / listing_value["dataset_root"]
    if file_hash(root / "manifest.json") != listing_value["dataset_manifest_sha256"]:
        raise ValueError("Surrogate dataset changed")
    task_ids = listing_value.get("task_ids")
    if task_ids is not None and (listing_value["list_sha256"] != digest(task_ids)
            or listing_value["manifest_sha256"] != digest(manifest)
            or listing_value["split_freeze_sha256"] != digest(freeze)):
        raise ValueError("Surrogate listing differs from frozen inputs")
    return load_r03_dataset(root, tokenizer, manifest, freeze, purpose="surrogate_only",
                            task_ids=task_ids, **kwargs)


def frozen_builder_hash(path):
    """Retain the original builder identity; frozen_write still checks all content."""
    path = Path(path)
    return json.loads(path.read_text())["builder_sha256"] if path.exists() else file_hash(__file__)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest/alfworld_manifest.json")
    p.add_argument("--freeze", default="data/manifest/alfworld_split_freeze.json")
    p.add_argument("--k100", default="data/verified/seeds_k100_s0")
    p.add_argument("--k20", default="data/verified/seeds_k20_s0")
    p.add_argument("--dev140", default="data/manifest/dev140.json")
    p.add_argument("--output-dir", default="data/manifest")
    p.add_argument("--surrogate-root", default="data/verified/r03_surrogate")
    args = p.parse_args(argv)
    manifest, freeze = load_frozen(args.manifest, args.freeze)
    large, ls, evidence = load_verified_collection(args.k100, manifest, freeze)
    small, ss, _ = load_verified_collection(args.k20, manifest, freeze)
    if ls["seed"] != 0 or ss["seed"] != 0:
        raise ValueError("R03 fixes support seed 0")
    sets = build_sets(manifest, freeze, large, small, json.loads(Path(args.dev140).read_text()))
    entries = []
    selected = set(sets["surrogate"]["task_ids"])
    for receipt in sorted(large["verified"], key=lambda r: r["task_id"]):
        if receipt["task_id"] in selected:
            episode = checked_episode(evidence, receipt)
            entry = dataset_entry(args.surrogate_root, episode, source="verified_teacher_surrogate",
                                  parent_seed_ids=[episode["episode_id"]])
            entry["source_episode_sha256"] = receipt["episode_hashes"]["episode.json"]
            entries.append(entry)
    publish_dataset(args.surrogate_root, entries, manifest, freeze, source="K100_s0_minus_K20_s0",
        purpose="surrogate_only", source_collection_sha256=digest(large),
        builder_sha256=frozen_builder_hash(Path(args.surrogate_root) / "manifest.json"),
        verified_evidence_checked=True)
    for name in ("surrogate", "surrogate40"):
        sets[name].update(dataset_root=os.path.relpath(args.surrogate_root, args.output_dir),
            dataset_manifest_sha256=file_hash(Path(args.surrogate_root) / "manifest.json"),
            copied_episode_files=True)
    for name, value in sets.items():
        path = Path(args.output_dir) / f"r03_{name}.json"
        value["builder_sha256"] = frozen_builder_hash(path)
        frozen_write(path, value)
    if not (Path(args.surrogate_root) / "provenance.json").exists():
        from sed.provenance import write_provenance
        write_provenance(args.surrogate_root, command="build_r03_sets", config=vars(args),
            inputs=[__file__, args.manifest, args.freeze, args.dev140,
                    str(Path(args.k100) / "manifest.json"), str(Path(args.k20) / "manifest.json")],
            details={"seed": 0, "training_exposure": 0, "new_teacher_calls": 0,
                     "new_environment_steps": 0, "source_evidence_fully_validated": True})
    print(json.dumps({k: {f: v[f] for f in ("unique_instances", "task_groups", "counts_by_task_type",
                                          "group_overlap_counts")} for k, v in sets.items()}, indent=2))


if __name__ == "__main__":
    main()
