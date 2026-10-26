from copy import deepcopy
from functools import partial
import json
from pathlib import Path

import pytest
import yaml

from harness.alfworld_env import Environment
from repair_fakes import RepairBackend, TASK_ID, factory
from sed.config import load_config
from sed.manifest import build_manifest, freeze_splits
from sed.runner import ScriptedPolicy, run_episode
from sed.storage import write_json
from tools.repair_determinism_check import check_episode, main


def episode_for(data_root, task_id, directory):
    task = {"task_id": task_id, "split": "train", "sealed": False}
    return run_episode(factory(data_root)(), ScriptedPolicy(["take key", "open door", "finish"]),
                       task, load_config(), directory, policy_condition="scripted_fixture")


def test_determinism_compares_actual_replays_and_recorded_reference(data_root, tmp_path):
    episode = episode_for(data_root, TASK_ID, tmp_path / "episode")
    report = check_episode(factory(data_root), episode)
    assert report["passed"] and all(report["checks"].values())
    assert [report["executions"][k]["termination_reason"] for k in ("first", "failed", "step_limit")] == [
        "SUCCESS", "FAILED", "STEP_LIMIT"]
    bad = deepcopy(episode)
    bad["steps"][1]["visible"]["observation"] = "old or fabricated suffix"
    report = check_episode(factory(data_root), bad)
    assert not report["passed"] and not report["checks"]["recorded_transitions_reproduced"]


def test_determinism_fails_on_drift_and_sealed_episode(data_root, tmp_path):
    episode = episode_for(data_root, TASK_ID, tmp_path / "episode")
    class Drift(RepairBackend):
        generation = 0
        def __init__(self, *args):
            super().__init__(*args)
            type(self).generation += 1
        def state(self):
            state = super().state()
            if len(self.actions) == 2 and type(self).generation > 1:
                state["feedback"] = "drift"
            return state
    report = check_episode(factory(data_root, backend=Drift), episode)
    assert not report["passed"] and not report["checks"]["same_visible_observations"]
    with pytest.raises(ValueError, match="training"):
        check_episode(factory(data_root), dict(episode, split="test", sealed=True))


def test_cli_report_exit_status_and_five_task_gate(data_root, tmp_path, monkeypatch):
    import tools.repair_determinism_check as cli
    # The shared dataset fixture has four train instances; add a fifth.
    directory = data_root / "train/room_c/trial_0"
    directory.mkdir(parents=True)
    (directory / "traj_data.json").write_text('{"task_type": "pick_and_place_simple"}')
    (directory / "game.tw-pddl").write_text('{"solvable": true}')
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    manifest_path, freeze_path = tmp_path / "manifest.json", tmp_path / "freeze.json"
    write_json(manifest_path, manifest)
    write_json(freeze_path, freeze)
    config = load_config()
    config["paths"].update(manifest=str(manifest_path), split_freeze=str(freeze_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    paths = []
    for index, task_id in enumerate(freeze["seed_pool_candidates"]):
        run_dir = tmp_path / f"episode-{index}"
        episode_for(data_root, task_id, run_dir)
        paths.append(str(run_dir / "episode.json"))
    monkeypatch.setattr(cli, "Environment", partial(Environment, backend_factory=RepairBackend))
    base = ["--config", str(config_path), "--data-root", str(data_root), "--episodes"]
    output = tmp_path / "report.json"
    assert main([*base, *paths, "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["passed"] and report["unique_instances"] == 5 and report["task_groups"] == 3
    assert report["teacher_calls"] == 0 and report["model_execution"] is False
    assert report["budget"]["interaction"]["task_validation_steps"] == 50
    provenance = json.loads((tmp_path / "report_artifacts/provenance.json").read_text())
    assert "execute.py" in provenance["details"]["repair_code_sha256"]
    output = tmp_path / "too_few.json"
    assert main([*base, *paths[:1], "--output", str(output)]) == 1
    assert not json.loads(output.read_text())["checks"]["at_least_five_training_instances"]
    # A reference mismatch produces a persisted failing report, not just an exception.
    bad = json.loads(Path(paths[0]).read_text())
    bad["steps"][0]["visible"]["observation"] = "wrong"
    write_json(paths[0], bad)
    output = tmp_path / "mismatch.json"
    assert main([*base, *paths, "--output", str(output)]) == 1
    assert not json.loads(output.read_text())["episodes"][0]["passed"]
