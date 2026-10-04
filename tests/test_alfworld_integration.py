"""Optional: installed data and both packages required; no downloads or APIs."""

import json
import os
from pathlib import Path

import pytest

from harness.alfworld import Environment
from sed.manifest import load_frozen, resolve_task, verify_task_files


@pytest.mark.integration
def test_real_reset_three_steps_and_replay():
    pytest.importorskip("alfworld")
    pytest.importorskip("textworld")
    root = os.environ.get("ALFWORLD_DATA_ROOT")
    if not root or not Path(root).is_dir():
        pytest.skip("Set ALFWORLD_DATA_ROOT to installed data")
    manifest_path = os.environ.get("SED_MANIFEST", "data/manifest/alfworld_manifest.json")
    freeze_path = os.environ.get("SED_SPLIT_FREEZE", "data/manifest/alfworld_split_freeze.json")
    if not Path(manifest_path).is_file() or not Path(freeze_path).is_file():
        pytest.skip("Build and freeze a local manifest first")
    manifest, freeze = load_frozen(manifest_path, freeze_path)
    task = resolve_task(manifest, freeze, freeze["seed_pool_candidates"][0])
    verify_task_files(root, task)
    env = Environment(root)
    try:
        visible, _, terminated, truncated, audit = env.reset(task["task_id"])
        assert not terminated and not truncated
        assert set(visible) == {"observation", "admissible_commands"}
        for _ in range(3):
            result = env.step("look")
        snapshot = env.snapshot()
        assert env.restore(snapshot) == result
        assert len(env.actions) == 3
        assert audit["internal_facts"]
        assert not audit["oracle_fields_discarded"]  # No expert installed at all.
    finally:
        env.close()
