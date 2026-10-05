"""Offline baseline protocol, pair and orchestration contracts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.common.explore import training_tasks, run_exploration, explore_one, checked_rollouts
from baselines.common.schedule import dose, epoch_batches, rank_slots, lr_multiplier
from baselines.eto.pairs import build_pairs
from baselines.eto.pipeline import stage_run, training_command, train_pipeline, RECIPE
from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.manifest import build_manifest, freeze_splits
from sed.runner import ScriptedPolicy
from sed.storage import digest, write_json


def listing_for(manifest, freeze, ids):
    return {"task_ids": ids, "list_sha256": digest(ids), "manifest_sha256": digest(manifest),
            "split_freeze_sha256": digest(freeze), "parent_seed_ids": {t: ["seed-parent"] for t in ids}}


@pytest.mark.parametrize("n,b,expected", [(1, 64, 1), (5, 64, 1), (10, 64, 1), (20, 64, 2),
                                          (32, 32, 4), (1000, 64, 64), (15, 32, 1)])
def test_x5(n, b, expected):
    plan = dose(n, b)
    assert plan["b_prime"] == expected
    assert plan["total_steps"] == 3 * plan["steps_per_epoch"]
    batches = list(epoch_batches(n, expected, epochs=3))
    assert len(batches) == plan["total_steps"]
    for epoch in range(3):
        assert sorted(i for row in batches if row["epoch"] == epoch for i in row["indices"]) == list(range(n))
    assert batches == list(epoch_batches(n, expected, epochs=3, seed=0))
    if n > 1:
        assert batches != list(epoch_batches(n, expected, epochs=3, seed=1))
    for row in batches:
        slots = rank_slots(row["indices"], 4)
        assert all(len(s) == 4 for s in slots)
        assert [i for s in slots for i in s if i is not None] == row["indices"]
    assert lr_multiplier(0, scheduler="cosine", total_steps=plan["total_steps"], warmup_ratio=.03) == 0


def test_dose_rejects_zero():
    with pytest.raises(ValueError):
        dose(0, 32)


@pytest.mark.parametrize("source", ["valid_seen", "valid_unseen", "train_dev"])
def test_explore_refuses_dev_sealed_before_side_effects(data_root, tmp_path, source):
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest, **({"dev_source": "train", "dev_count": 1} if source == "train_dev" else {}))
    task = next(t for t, a in freeze["assignments"].items() if a["split"] == ("test" if source == "valid_unseen" else "dev"))
    listing = listing_for(manifest, freeze, [task])
    with pytest.raises(ValueError):
        training_tasks(listing, manifest, freeze)
    config = load_config()
    config["paths"].update(manifest=str(tmp_path / "m.json"), split_freeze=str(tmp_path / "f.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    path = tmp_path / "tasks.json"
    write_json(path, listing)
    with pytest.raises(ValueError):
        run_exploration(config=config, tasks_path=path, output=tmp_path / "out",
                        spec={"enable_thinking": False, "max_tokens": 1024}, decoding="greedy")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("fail", [False, True])
def test_explore_real_runner_fake_clients_and_replay(data_root, tmp_path, fail):
    manifest, config = build_manifest(data_root), load_config()
    freeze = freeze_splits(manifest)
    config["environment"].update(data_root=str(data_root), max_episode_steps=40)
    config["paths"].update(manifest=str(tmp_path / "m.json"), split_freeze=str(tmp_path / "f.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    task_id = freeze["seed_pool_candidates"][0]
    tasks = tmp_path / "tasks.json"
    write_json(tasks, listing_for(manifest, freeze, [task_id]))
    class Policy(ScriptedPolicy):
        def __init__(self, **kwargs):
            super().__init__(["look"])
        def generate(self, *args):
            if fail:
                raise RuntimeError("sensitive raw exception not logged")
            return super().generate(*args)
    def env_factory(config, task):
        return Environment(data_root, seed=0, max_episode_steps=40, backend_factory=FakeBackend)
    def worker(job):
        return explore_one(job, env_factory=env_factory, policy_factory=Policy)
    out = tmp_path / "exploration"
    result = run_exploration(config=config, tasks_path=tasks, output=out,
             spec={"enable_thinking": False, "max_tokens": 1024}, decoding="greedy", worker_fn=worker)
    receipt, = json.loads((out / "receipts.json").read_text())
    assert receipt["source"] == result["source"] == "student_exploration"
    assert receipt["parent_seed_ids"] == ["seed-parent"]
    assert receipt["trusted_seed"] is False and result["trusted_trajectories_added"] == 0
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["config"]["exploration"]["training_eligible"]
    assert "baselines/common/explore.py" in provenance["code_sha256"]
    if fail:
        assert receipt["won"] == "unknown" and not receipt["training_eligible"]
        with pytest.raises(ValueError, match="Unknown/incomplete"):
            checked_rollouts(out, manifest, freeze)
    else:
        assert receipt["training_eligible"] and receipt["won"] is False
        assert result["environment_steps"] == result["replay_environment_steps"] == 40
        assert result["environment_episodes"] == result["replay_environment_episodes"] == 1
        assert receipt["legal_actions"] == 40
        checked_rollouts(out, manifest, freeze)
        episode_path = out / receipt["episode_dir"] / "episode.json"
        episode_path.write_text("{}")
        with pytest.raises(ValueError, match="evidence changed"):
            checked_rollouts(out, manifest, freeze)


def messages(turns=2):
    result = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    for i in range(turns):
        if i:
            result.append({"role": "user", "content": f"observation {i}"})
        result.append({"role": "assistant", "content": f"Action: action {i}"})
    return result


def test_eto_pair_rule_counts_no_truncation_and_end_roles():
    expert = [{"task_id": f"train/{i}", "won": True, "messages": messages()} for i in range(5)]
    rollouts = [{"task_id": f"train/{i}", "won": value, "messages": messages(40)}
                for i, value in enumerate([False, True, "unknown", False, False])]
    rollouts[0]["messages"].append({"role": "user", "content": "trailing"})
    rollouts[3]["messages"] = messages()[:2]  # Empty rejected continuation after drop.
    expert[4]["messages"].append({"role": "user", "content": "invalid chosen end"})
    pairs, report = build_pairs(expert, rollouts)
    assert {k: report[k] for k in ["win", "tie", "lose", "P", "unknown", "invalid_end"]} == {
        "win": 0, "tie": 1, "lose": 3, "P": 1, "unknown": 1, "invalid_end": 2}
    assert len(pairs[0]["rejected"]) == 79  # All 40 assistant turns, no expert+5 truncation.
    assert pairs[0]["chosen"] == expert[0]["messages"][2:]
    assert pairs[0]["prompt"] == expert[0]["messages"][:2]
    assert pairs[0]["rejected"][-1]["role"] == "assistant"


def test_stage_resume_hashes_and_failed_attempt_preservation(tmp_path):
    calls = []
    def action(attempt):
        calls.append(attempt)
        write_json(attempt / "output/result.json", {"done": True})
    result = stage_run(tmp_path, "sft", "id", action)
    assert stage_run(tmp_path, "sft", "id", action) == result and len(calls) == 1
    with pytest.raises(ValueError, match="changed"):
        stage_run(tmp_path, "sft", "other", action)
    (result / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        stage_run(tmp_path, "sft", "id", action)
    def fail(attempt):
        write_json(attempt / "output/failed.json", {"retained": True})
        raise RuntimeError("failure")
    with pytest.raises(RuntimeError):
        stage_run(tmp_path, "explore", "id", fail)
    completed = stage_run(tmp_path, "explore", "id", action)
    assert completed.parent.name == "attempt-001"
    assert (tmp_path / "stages/explore/attempt-000/output/failed.json").exists()


def test_pipeline_fixed_hyperparameters_and_no_pair_stop(tmp_path, monkeypatch):
    args = SimpleNamespace(train_python="train-python", gpus=4, config="configs/base.yaml",
            snapshot="base", verified="verified", output=str(tmp_path), through="all")
    for iteration in range(1, 4):
        command = training_command(args, objective="dpo", snapshot="policy", output="out", pairs="pairs", iteration=iteration)
        assert command[command.index("--reference") + 1] == command[command.index("--snapshot") + 1] == "policy"
        assert command[command.index("--beta") + 1] == ("0.1" if iteration == 1 else "0.5")
        assert command[command.index("--learning-rate") + 1] == ("1e-06" if iteration == 1 else "5e-07")
        assert "32768" in command and "constant_with_warmup" in command
    names = []
    def stage(root, name, identity, action):
        names.append(name)
        path = tmp_path / name
        path.mkdir()
        if name == "pairs-1":
            write_json(path / "report.json", {"P": 0})
        return path
    monkeypatch.setattr("baselines.eto.pipeline.stage_run", stage)
    monkeypatch.setattr("baselines.eto.pipeline.update_disclosure", lambda *args: None)
    train_pipeline(args, {"recipe": RECIPE})
    assert names == ["sft", "explore-1", "pairs-1"]
    assert json.loads((tmp_path / "result.json").read_text())["reason"] == "no pairs at iteration 1"
