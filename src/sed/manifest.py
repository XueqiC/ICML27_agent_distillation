"""Enumerate the installed data, hash games, and freeze disjoint task groups."""

import argparse
from collections import Counter
import json
from pathlib import Path
import uuid

from sed.config import load_config
from sed.provenance import write_provenance
from sed.storage import canonical_bytes, digest, file_hash, write_json

SOURCES = ("train", "valid_seen", "valid_unseen")
TASK_TYPES = {"pick_and_place_simple", "look_at_obj_in_light",
              "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
              "pick_cool_then_place_in_recep", "pick_two_obj_and_place"}


def build_manifest(data_root):
    root = Path(data_root).resolve()
    tasks = []
    for source in SOURCES:
        if not (root / source).is_dir():
            raise FileNotFoundError(f"Missing source split: {source}")
        # Union retains missing-game trials and unusual game-only directories.
        directories = {p.parent for pattern in ("traj_data.json", "game.tw-pddl")
                       for p in (root / source).rglob(pattern)}
        for directory in sorted(directories):
            files = {}
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    if not path.resolve().is_relative_to(root):
                        raise ValueError("Dataset symlink escapes data root")
                    files[str(path.relative_to(directory))] = file_hash(path)
            task_id = str(directory.relative_to(root))
            traj_path, game_path = directory / "traj_data.json", directory / "game.tw-pddl"
            traj = json.loads(traj_path.read_text()) if traj_path.exists() else {}
            game = json.loads(game_path.read_text()) if game_path.exists() else {}
            kind = traj.get("task_type", "unknown")
            reasons = []
            if not traj_path.exists():
                reasons.append("missing_trajectory_metadata")
            if not game_path.exists():
                reasons.append("missing_game")
            elif game.get("solvable") is not True:
                reasons.append("not_marked_solvable")
            if kind not in TASK_TYPES:
                reasons.append("unsupported_task_type")
            if "movable" in task_id or "Sliced" in task_id:
                reasons.append("unsupported_movable_or_sliced")
            tasks.append({"task_id": task_id, "source_split": source,
                          "group_id": str(directory.parent.relative_to(root)),
                          "task_type": kind, "files": files,
                          "game_sha256": files.get("game.tw-pddl"),
                          "content_sha256": digest(files), "eligible": not reasons,
                          "exclusion_reasons": reasons})
    counts = {}
    for source in SOURCES:
        rows = [r for r in tasks if r["source_split"] == source]
        playable = [r for r in rows if r["eligible"]]
        counts[source] = {"discovered_trials": len(rows), "eligible_games": len(playable),
                          "discovered_groups": len({r["group_id"] for r in rows}),
                          "eligible_groups": len({r["group_id"] for r in playable}),
                          "eligible_by_task_type": dict(sorted(Counter(
                              r["task_type"] for r in playable).items())),
                          "exclusion_reasons": dict(sorted(Counter(
                              reason for r in rows for reason in r["exclusion_reasons"]).items()))}
    return {"schema_version": 1, "environment": "alfworld", "source_splits": list(SOURCES),
            "count_unit": "task_instance_is_one_trial_directory; groups_are_not_independent_trials",
            "counts": counts, "tasks": tasks}


def freeze_splits(manifest, *, dev_source="valid_seen", dev_count=None,
                  sampling_seed=0, sealed_source="valid_unseen"):
    if dev_source not in {"train", "valid_seen"} or sealed_source != "valid_unseen":
        raise ValueError("Stage 1 supports train/valid_seen dev and valid_unseen sealed test")
    if dev_source == "train" and dev_count is None:
        raise ValueError("Train-derived dev needs an explicit group count")
    eligible = [r for r in manifest["tasks"] if r["eligible"]]
    groups = sorted({r["group_id"] for r in eligible if r["source_split"] == dev_source},
                    key=lambda g: digest([sampling_seed, g]))
    if dev_count is not None and (type(dev_count) is not int or not 0 <= dev_count <= len(groups)):
        raise ValueError("dev_count must be a valid number of task groups")
    dev_groups = set(groups if dev_count is None else groups[:dev_count])
    assignments = {}
    for task in eligible:
        source = task["source_split"]
        split = ("test" if source == sealed_source else "dev" if task["group_id"] in dev_groups
                 else "train" if source == "train" else "unused")
        assignments[task["task_id"]] = {"split": split, "sealed": split == "test",
                                        "group_id": task["group_id"]}
    seed_pool = sorted(t for t, value in assignments.items() if value["split"] == "train")
    return {"schema_version": 1, "manifest_sha256": digest(manifest),
            "configuration": {"dev_source": dev_source, "dev_count": dev_count,
                              "dev_count_unit": "task_groups", "sampling_seed": sampling_seed,
                              "sealed_source": sealed_source},
            "seed_pool_candidates": seed_pool, "seed_trajectories_selected": [],
            "K_counts": "trusted_trajectories_not_task_count",
            "assignments": assignments,
            "counts": dict(sorted(Counter(v["split"] for v in assignments.values()).items()))}


def frozen_write(path, value):
    path = Path(path)
    try:
        write_json(path, value, exclusive=True)
    except FileExistsError:
        if path.read_bytes() != canonical_bytes(value) + b"\n":
            raise ValueError(f"Refusing to overwrite a different frozen artifact: {path}") from None


def load_frozen(manifest_path, freeze_path):
    manifest = json.loads(Path(manifest_path).read_text())
    freeze = json.loads(Path(freeze_path).read_text())
    if digest(manifest) != freeze["manifest_sha256"]:
        raise ValueError("Manifest differs from split freeze")
    expected = freeze_splits(manifest, **{k: v for k, v in freeze["configuration"].items()
                                        if k != "dev_count_unit"})
    if freeze != expected:
        raise ValueError("Split assignments differ from the frozen protocol")
    return manifest, freeze


def authorize_task(task, purpose):
    if purpose not in {"generation", "diagnosis", "evaluation", "replay"}:
        raise ValueError("Unknown task purpose")
    split = task.get("split")
    if split not in {"train", "dev", "test"}:
        raise ValueError("Task lacks an authorized frozen split")
    sealed = task.get("sealed", False) or split == "test" or task["task_id"].startswith("valid_unseen/")
    if sealed and purpose != "evaluation":
        raise ValueError("Sealed tasks are for aggregate evaluation only")
    if purpose == "generation" and split != "train":
        raise ValueError("Training generation requires a frozen train task")


def resolve_task(manifest, freeze, task_id):
    assignment = freeze["assignments"].get(task_id)
    if not assignment or assignment["split"] == "unused":
        raise ValueError("Task is not eligible in the frozen protocol")
    task = next(r for r in manifest["tasks"] if r["task_id"] == task_id)
    return dict(task, **{k: v for k, v in assignment.items() if k != "group_id"})


def verify_task_files(data_root, task):
    root = Path(data_root).resolve()
    directory = (root / task["task_id"]).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Task path escapes data root")
    actual = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            if not path.resolve().is_relative_to(root):
                raise ValueError("Dataset symlink escapes data root")
            actual[str(path.relative_to(directory))] = file_hash(path)
    if actual != task["files"]:
        raise ValueError("Task files differ from frozen manifest")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data-root")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.data_root:
        config["environment"]["data_root"] = args.data_root
    root = config["environment"]["data_root"]
    if not root:
        parser.error("Set ALFWORLD_DATA_ROOT or --data-root")
    run_dir = Path(config["paths"]["runs"]) / ("manifest-" + str(uuid.uuid4()))
    write_provenance(run_dir, command="manifest", config=config, inputs=[args.config])
    manifest = build_manifest(root)
    freeze = freeze_splits(manifest, **config["splits"])
    # Preflight both destinations before publishing either.
    for path, value in ((config["paths"]["manifest"], manifest),
                        (config["paths"]["split_freeze"], freeze)):
        if Path(path).exists() and Path(path).read_bytes() != canonical_bytes(value) + b"\n":
            raise ValueError("Existing manifest/freeze differs; choose a new explicit destination")
    frozen_write(config["paths"]["manifest"], manifest)
    frozen_write(config["paths"]["split_freeze"], freeze)
    summary = {"manifest_sha256": digest(manifest), "freeze_sha256": digest(freeze),
               "counts": manifest["counts"], "frozen_counts": freeze["counts"]}
    write_json(run_dir / "result.json", summary, exclusive=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
