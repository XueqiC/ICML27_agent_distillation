import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.manifest import TASK_TYPES, build_manifest, freeze_splits
from sed.seed_selection import select_seeds, validate_selection
from sed.storage import digest
from teachers.collect_seeds import collect
from teachers.openai_teacher import Teacher


@pytest.fixture
def seed_bank(tmp_path):
    root = tmp_path / "data"
    for source in ("train", "valid_seen", "valid_unseen"):
        for kind in sorted(TASK_TYPES):
            for group in range(5):
                for trial in range(2):
                    directory = root / source / f"{kind}-{group}" / f"trial_{trial}"
                    directory.mkdir(parents=True)
                    (directory / "traj_data.json").write_text(json.dumps({"task_type": kind}))
                    (directory / "game.tw-pddl").write_text(json.dumps({"solvable": True}))
    manifest = build_manifest(root)
    freeze = freeze_splits(manifest)
    config = load_config()
    config["environment"].update(data_root=str(root), max_episode_steps=1)
    return root, manifest, freeze, config


def test_seed_selection_deterministic_groups_and_quotas(seed_bank):
    _, manifest, freeze, _ = seed_bank
    selected = select_seeds(manifest, freeze)
    assert selected == select_seeds(manifest, freeze)
    assert selected != select_seeds(manifest, freeze, seed=1)
    assert [selected["quotas"][t] for t in selected["type_order"]] == [2, 2, 2, 2, 1, 1]
    assert selected["K"] == 10 and selected["trusted_trajectories"] == 0
    rows = [r for kind in selected["type_order"]
            for r in selected["selected"][kind] + selected["replacements"][kind]]
    assert len(rows) == len({r["group_id"] for r in rows}) == 30
    assert all(r["task_id"].startswith("train/") for r in rows)
    assert selected["manifest_sha256"] == digest(manifest)
    assert len([r for rows in select_seeds(manifest, freeze, k=5)["selected"].values() for r in rows]) == 5


def test_seed_selection_excludes_train_derived_dev_and_tamper(seed_bank):
    _, manifest, _, _ = seed_bank
    freeze = freeze_splits(manifest, dev_source="train", dev_count=4)
    selection = select_seeds(manifest, freeze, k=6)
    rows = [r for group in selection["replacements"].values() for r in group]
    rows += [r for group in selection["selected"].values() for r in group]
    assert all(freeze["assignments"][r["task_id"]]["split"] == "train" for r in rows)
    bad = json.loads(json.dumps(selection))
    kind = bad["type_order"][0]
    bad["selected"][kind][0]["task_id"] = "valid_seen/forged/trial"
    with pytest.raises(ValueError, match="frozen TRAIN"):
        validate_selection(bad, manifest, freeze)


def factories(root, config, losing_tasks, calls):
    def environment(task):
        def backend(path, seed):
            result = FakeBackend(path, seed)
            result.win_at = 99 if task["task_id"] in losing_tasks else 1
            return result
        return Environment(root, max_episode_steps=1, backend_factory=backend)
    def teacher(task, cache, ledger):
        class Responses:
            def create(self, **kwargs):
                calls.append((task["task_id"], str(cache)))
                return {"output_text": "Thought: I claim I won.\nAction: look", "model": "gpt-6-luna", "status": "completed",
                        "usage": {"input_tokens": 20, "output_tokens": 10}, "id": "fake"}
        return Teacher(cache_dir=cache, usage_ledger=ledger, run_id="fake",
                       client=SimpleNamespace(responses=Responses()))
    return environment, teacher


def test_collection_replacement_real_win_accounting_resume(seed_bank, tmp_path):
    root, manifest, freeze, config = seed_bank
    selection = select_seeds(manifest, freeze, k=6)
    kind = selection["type_order"][0]
    failed = selection["selected"][kind][0]["task_id"]
    calls = []
    environment, teacher = factories(root, config, {failed}, calls)
    output = tmp_path / "collection"
    result = collect(selection, manifest, freeze, config, output,
                     env_factory=environment, teacher_factory=teacher)
    assert result["status"] == "complete" and result["trusted_trajectories"] == 6
    assert result["task_groups"] == result["unique_instances"] == 6
    assert len(result["attempts"]) == 9 and len(calls) == 9
    rejected = [r for r in result["attempts"] if r["task_id"] == failed]
    assert len(rejected) == 3 and all(r["won"] is False for r in rejected)
    assert len({c[1] for c in calls if c[0] == failed}) == 3
    assert result["verified"][0]["task_id"] == selection["replacements"][kind][0]["task_id"]
    assert all(r["usage"]["known_billed_tokens"]["input_tokens"] == 20 for r in result["attempts"])
    again = collect(selection, manifest, freeze, config, output,
                    env_factory=environment, teacher_factory=teacher)
    assert again == result and len(calls) == 9
    changed = dict(config, changed=True)
    with pytest.raises(ValueError, match="identity changed"):
        collect(selection, manifest, freeze, changed, output)


def test_collection_resume_after_episode_before_receipt(seed_bank, tmp_path, monkeypatch):
    import teachers.collect_seeds as module
    root, manifest, freeze, config = seed_bank
    selection = select_seeds(manifest, freeze, k=1)
    calls = []
    environment, teacher = factories(root, config, set(), calls)
    original = module.write_json
    def interrupt(path, *args, **kwargs):
        if Path(path).name == "receipt.json":
            raise KeyboardInterrupt
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        collect(selection, manifest, freeze, config, tmp_path / "out",
                env_factory=environment, teacher_factory=teacher)
    monkeypatch.setattr(module, "write_json", original)
    result = collect(selection, manifest, freeze, config, tmp_path / "out",
                     env_factory=environment, teacher_factory=teacher)
    assert result["trusted_trajectories"] == 1 and len(calls) == 1


def test_replacement_precedes_next_original_slot(seed_bank, tmp_path):
    root, manifest, freeze, config = seed_bank
    selection = select_seeds(manifest, freeze, k=10)
    kind = selection["type_order"][0]
    first, second = selection["selected"][kind]
    calls = []
    environment, teacher = factories(root, config, {first["task_id"]}, calls)
    result = collect(selection, manifest, freeze, config, tmp_path / "out", attempts=1,
                     env_factory=environment, teacher_factory=teacher)
    assert [r["task_id"] for r in result["attempts"][:3]] == [first["task_id"],
        selection["replacements"][kind][0]["task_id"], second["task_id"]]
    assert result["attempts"][1]["selection_parent_task_id"] == first["task_id"]


def test_interrupted_attempt_is_retained_with_unknown_cost(seed_bank, tmp_path, monkeypatch):
    import teachers.collect_seeds as module
    root, manifest, freeze, config = seed_bank
    selection = select_seeds(manifest, freeze, k=1)
    environment, teacher = factories(root, config, set(), [])
    original = module.run_episode
    def interrupted(env, policy, task, config, output, **kwargs):
        Path(output).mkdir()
        (Path(output) / "episode.json").write_text(json.dumps({"status": "started", "complete": False}))
        policy.ledger.append({"call_id": "interrupted", "status": "started", "usage": None})
        env.close()
        raise KeyboardInterrupt
    monkeypatch.setattr(module, "run_episode", interrupted)
    with pytest.raises(KeyboardInterrupt):
        collect(selection, manifest, freeze, config, tmp_path / "out",
                env_factory=environment, teacher_factory=teacher)
    monkeypatch.setattr(module, "run_episode", original)
    from teachers.openai_teacher import TeacherCallError
    with pytest.raises(TeacherCallError, match="NOT complete"):
        collect(selection, manifest, freeze, config, tmp_path / "out",
                env_factory=environment, teacher_factory=teacher)
    result = collect(selection, manifest, freeze, config, tmp_path / "out",
                     env_factory=environment, teacher_factory=teacher)
    assert len(result["attempts"]) == result["trusted_trajectories"] == 1
    assert list((tmp_path / "out").rglob("transport_failure.json"))
    assert any('"status":"started"' in p.read_text() or '"status": "started"' in p.read_text()
               for p in (tmp_path / "out").rglob("usage.jsonl"))


def test_collection_exhaustion_does_not_claim_k(seed_bank, tmp_path):
    root, manifest, freeze, config = seed_bank
    selection = select_seeds(manifest, freeze, k=1)
    kind = selection["type_order"][0]
    losing = {r["task_id"] for r in selection["selected"][kind] + selection["replacements"][kind]}
    environment, teacher = factories(root, config, losing, [])
    result = collect(selection, manifest, freeze, config, tmp_path / "out", attempts=1,
                     env_factory=environment, teacher_factory=teacher)
    assert result["status"] == "exhausted" and result["trusted_trajectories"] == 0
    assert len(result["attempts"]) == 5
