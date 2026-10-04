import json

import pytest

from sed.manifest import (authorize_task, build_manifest, freeze_splits, frozen_write,
                          load_frozen, resolve_task, verify_task_files)
from sed.storage import digest


def test_manifest_determinism_hashes_and_counts(data_root):
    first = build_manifest(data_root)
    assert first == build_manifest(data_root)
    assert first["counts"]["train"]["eligible_games"] == 4
    assert all(len(t["files"]) == 3 for t in first["tasks"])
    task = first["tasks"][0]
    (data_root / task["task_id"] / "initial_state.pddl").write_text("changed")
    second = build_manifest(data_root)
    assert digest(first) != digest(second)
    with pytest.raises(ValueError, match="differ"):
        verify_task_files(data_root, task)


def test_split_freeze_groups_disjoint_and_train_only_candidates(data_root):
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest, dev_source="train", dev_count=1, sampling_seed=42)
    assert freeze == freeze_splits(manifest, dev_source="train", dev_count=1, sampling_seed=42)
    assert freeze["counts"] == {"dev": 2, "test": 4, "train": 2, "unused": 4}
    assert all(t.startswith("train/") for t in freeze["seed_pool_candidates"])
    group_splits = {}
    for task in freeze["assignments"].values():
        group_splits.setdefault(task["group_id"], set()).add(task["split"])
        assert task["sealed"] == (task["split"] == "test")
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_missing_and_unsolvable_games_are_disclosed(data_root):
    (data_root / "train/room_a/trial_0/game.tw-pddl").unlink()
    (data_root / "train/room_a/trial_1/game.tw-pddl").write_text('{"solvable": false}')
    counts = build_manifest(data_root)["counts"]["train"]
    assert counts["discovered_trials"] == 4 and counts["eligible_games"] == 2
    assert counts["exclusion_reasons"] == {"missing_game": 1, "not_marked_solvable": 1}


def test_freeze_is_immutable_and_validated(data_root, tmp_path):
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    mp, fp = tmp_path / "m.json", tmp_path / "f.json"
    frozen_write(mp, manifest)
    frozen_write(fp, freeze)
    frozen_write(fp, freeze)
    assert load_frozen(mp, fp) == (manifest, freeze)
    with pytest.raises(ValueError, match="overwrite"):
        frozen_write(fp, {"changed": True})
    key = next(iter(freeze["assignments"]))
    freeze["assignments"][key]["split"] = "test"
    fp.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="assignments"):
        load_frozen(mp, fp)


@pytest.mark.parametrize("purpose", ["generation", "diagnosis", "replay"])
def test_sealed_tasks_rejected(data_root, purpose):
    manifest = build_manifest(data_root)
    task = resolve_task(manifest, freeze_splits(manifest), "valid_unseen/room_a/trial_0")
    with pytest.raises(ValueError, match="Sealed"):
        authorize_task(task, purpose)
    authorize_task(task, "evaluation")


def test_train_dev_generation_boundary(task):
    with pytest.raises(ValueError, match="train"):
        authorize_task(dict(task, split="dev"), "generation")
    authorize_task(dict(task, split="dev"), "diagnosis")
