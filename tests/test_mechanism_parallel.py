"""Spawned fake pipelines: stable artifacts, real file locks, no model transports."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from analysis.mechanism.branch import run_branch
from analysis.mechanism.common import compatible_identity, read_json, workspace
from analysis.mechanism.fce import run_fce
from analysis.mechanism.interventions import run_interventions
from analysis.mechanism.policies import BranchTeacher, BudgetExceeded, TeacherBudget
from analysis.mechanism.readout import validate_intervention_artifacts
from analysis.mechanism.taxonomy import classify_run, comparison
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.manifest import build_manifest, freeze_splits
from sed.runner import ScriptedPolicy, run_episode
from sed.storage import JsonlLedger, digest, write_json
from test_mechanism import BranchBackend, FakeStudent, TASK_ID

REAL_TIME = time.monotonic


class ShortBackend(BranchBackend):
    def state(self):
        row = super().state()
        row["lost"] = self.n >= 4 and not self.won
        return row

    def step(self, action):
        row, reward, done = super().step(action)
        return row, reward, done or row["lost"]


def fake_environment(config, task):
    return Environment(config["environment"]["data_root"], split="dev", backend_factory=ShortBackend)


class Responses:
    def create(self, **kwargs):
        text = kwargs["input"][-1]["content"]
        turn = (len(kwargs["input"])-2)//2
        raw = "Take the apple." if "Do not issue an action" in text else "Action: win" if turn < 1 else "Action: look"
        return SimpleNamespace(output_text=raw, model="gpt-6-luna", id="fake", status="completed",
            usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5), _request_id="request")


def fake_teacher(config, output, budget, episode_key, *, namespace, sample_index, advice=False):
    return BranchTeacher(namespace=namespace, sample_index=sample_index, budget=budget,
        episode_key=episode_key, advice=advice, cache_dir=Path(output)/"cache",
        usage_ledger=Path(output)/"usage.jsonl", run_id="fake", client=SimpleNamespace(responses=Responses()),
        max_attempts=config["models"]["teacher"]["max_attempts"], sampling={"max_output_tokens": 8},
        sleep=lambda _: None, requests_per_minute=100000, tokens_per_minute=1000000000)


def deterministic_branch(env, policy, task, config, parent, t, output, **kwargs):
    """Freeze only external nondeterminism; run the real replay/runner/teacher IO."""
    output = Path(output)
    telemetry = Path(config["paths"]["runs"])/"timing.jsonl"
    key = [task["task_id"], t, kwargs["arm"], kwargs["sample_index"]]
    interrupt = telemetry.parent/"interrupt.json"
    if interrupt.exists() and read_json(interrupt) == key:
        interrupt.unlink()
        write_json(output.parent/"interrupted.json", {"unit": key})
        env.close()
        raise KeyboardInterrupt
    JsonlLedger(telemetry).append({"unit": key, "pid": os.getpid(), "phase": "start", "time": REAL_TIME()})
    # Vary completion order without changing any recorded episode data.
    time.sleep(.02 * (2-kwargs["sample_index"] % 3))
    counter = iter(range(10000))
    def uuid():
        return digest([key, next(counter)])[:32]
    try:
        with patch("uuid.uuid4", uuid), patch("time.monotonic", lambda: 1234.), \
             patch("sed.provenance.utc_now", lambda: "2000-01-01T00:00:00+00:00"), \
             patch("teachers.openai_teacher.utc_now", lambda: "2000-01-01T00:00:00+00:00"):
            return run_branch(env, policy, task, config, parent, t, output, **kwargs)
    finally:
        JsonlLedger(telemetry).append({"unit": key, "pid": os.getpid(), "phase": "end", "time": REAL_TIME()})


@pytest.fixture
def spawned_pipeline(tmp_path):
    data = tmp_path/"data"
    for name in ("train", "valid_seen", "valid_unseen"):
        (data/name).mkdir(parents=True)
    ids = [TASK_ID.replace("trial_0", f"trial_{i}") for i in range(2)]
    assert len(set(ids)) == 2
    for task_id in ids:
        directory = data/task_id
        directory.mkdir(parents=True)
        write_json(directory/"game.tw-pddl", {"solvable": True})
        write_json(directory/"traj_data.json", {"task_type": "pick_heat_then_place_in_recep"})
    manifest = build_manifest(data)
    freeze = freeze_splits(manifest)
    config = load_config()
    config["paths"].update(manifest=str(tmp_path/"manifest.json"), split_freeze=str(tmp_path/"freeze.json"),
                           runs=str(tmp_path/"telemetry"))
    config["environment"]["data_root"] = str(data)
    config["models"]["teacher"]["max_attempts"] = 1
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    parents = []
    for i, task_id in enumerate(ids):
        task = {"task_id": task_id, "split": "dev", "sealed": False, "task_type": "pick_heat_then_place_in_recep"}
        parents.append(run_episode(fake_environment(config, task), ScriptedPolicy(["look"]), task, config,
            tmp_path/f"source/episodes/{i:03d}", policy_condition="fake_student", purpose="diagnosis"))
    write_json(tmp_path/"source/tasks.json", {"task_ids": ids, "list_sha256": digest(ids),
        "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze)})
    classified = classify_run(tmp_path/"source", config)
    teacher = {"episodes": [dict(row, success=True) for row in classified["episodes"]]}
    gap = tmp_path/"taxonomy.json"
    write_json(gap, {"G": comparison(classified, teacher, "teacher"),
                    "runs": {"student": classified, "teacher": teacher}})
    seeds = [{"order": 0, "episode": dict(parents[0], task_id=ids[0].replace("valid_seen", "train"),
                                          split="train", success=True)}]
    fce_args = dict(gap_path=gap, output=tmp_path/"fce", config=config, max_episode=100, max_run=500,
        env_factory=fake_environment, teacher_factory=fake_teacher, branch_fn=deterministic_branch)
    arms_args = dict(gap_path=gap, fce_path=tmp_path/"fce/fce.json", verified=None, output=tmp_path/"arms",
        config=config, spec={}, max_episode=100, max_run=500, env_factory=fake_environment,
        teacher_factory=fake_teacher, student_factory=FakeStudent, branch_fn=deterministic_branch, seeds=seeds)
    return fce_args, arms_args


def artifacts(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in Path(root).rglob("*.json")
            if p.name in {"receipt.json", "result.json", "fce.json", "interventions.json", "shared_teacher_output.json"}}


def assert_overlap(config, *, arm=None, different_tasks=False, same_probe=False):
    rows = [json.loads(line) for line in (Path(config["paths"]["runs"])/"timing.jsonl").read_text().splitlines()]
    starts = {tuple(r["unit"]): r for r in rows if r["phase"] == "start"}
    intervals = [(starts[tuple(r["unit"])], r["time"]) for r in rows if r["phase"] == "end"]
    assert any(a["pid"] != b["pid"] and a["time"] < be and b["time"] < ae
        and (arm is None or a["unit"][2] == b["unit"][2] == arm)
        and (not different_tasks or a["unit"][0] != b["unit"][0])
        and (not same_probe or a["unit"][:3] == b["unit"][:3])
        for a, ae in intervals for b, be in intervals)


def test_process_pipeline_byte_identical_workers_and_real_overlap(spawned_pipeline):
    fce, arms = spawned_pipeline
    outputs = []
    for workers in (1, 4):
        located = run_fce(**fce, workers=workers)
        assert located["complete"] and [r["fce"] for r in located["episodes"]] == [0, 0]
        if workers == 4:
            assert_overlap(fce["config"], arm="takeover", different_tasks=True)
            assert_overlap(fce["config"], arm="takeover", same_probe=True)
        result = run_interventions(**arms, workers=workers)
        validate_intervention_artifacts(result)
        assert result["budget"]["total"] == 4  # One I3 and I4 per parent.
        assert all(row["arms"]["I4"]["rescue"] for row in result["episodes"])
        outputs.append((artifacts(fce["output"]), artifacts(arms["output"])))
        if workers == 4:
            assert_overlap(fce["config"], arm="I0")
        for path in (fce["output"], arms["output"], Path(fce["config"]["paths"]["runs"])):
            shutil.rmtree(path)
    assert outputs[0] == outputs[1]


def test_resume_partial_workspace_different_workers(spawned_pipeline):
    fce, arms = spawned_pipeline
    interrupt = Path(fce["config"]["paths"]["runs"])/"interrupt.json"
    write_json(interrupt, [TASK_ID, 0, "takeover", 1])
    with pytest.raises(KeyboardInterrupt):
        run_fce(**fce)
    receipts = sorted(fce["output"].rglob("receipt.json"))
    assert len(receipts) == 1
    kept = receipts[0]
    old_receipt = kept.read_bytes()
    interrupted = next(fce["output"].rglob("interrupted.json"))
    old_interruption = interrupted.read_bytes()
    assert run_fce(**fce, workers=4)["complete"]
    assert kept.read_bytes() == old_receipt
    assert not list(kept.parent.parent.glob("attempt-001"))
    assert interrupted.read_bytes() == old_interruption
    assert (interrupted.parent.parent/"attempt-001/receipt.json").exists()
    write_json(interrupt, [TASK_ID, 0, "I4", 3])
    with pytest.raises(KeyboardInterrupt):
        run_interventions(**arms, workers=1)
    before = artifacts(arms["output"])
    run_interventions(**arms, workers=4)
    after = artifacts(arms["output"])
    assert all(after[path] == data for path, data in before.items())
    total = read_json(arms["output"]/"budget.json")["total"]
    run_interventions(**arms, workers=1)
    assert artifacts(arms["output"]) == after
    assert read_json(arms["output"]/"budget.json")["total"] == total


@pytest.mark.parametrize("max_episode,max_run", [(1, 20), (100, 5)])
def test_capped_fce_is_identical_and_incomplete(spawned_pipeline, max_episode, max_run):
    fce, _ = spawned_pipeline
    fce.update(max_episode=max_episode, max_run=max_run)
    from teachers.openai_teacher import TeacherCallError
    for workers in (1, 4):
        with pytest.raises(TeacherCallError):
            run_fce(**fce, workers=workers)
        budget = read_json(fce["output"]/"budget.json")
        assert budget["total"] <= max_run
        assert all(n <= max_episode for n in budget["episodes"].values())
        assert not (fce["output"]/"fce.json").exists()
        assert list(fce["output"].rglob("incomplete.json"))
        shutil.rmtree(fce["output"])


@pytest.mark.parametrize("max_episode,max_run", [(1, 20), (100, 1)])
def test_capped_interventions_are_identical(spawned_pipeline, max_episode, max_run):
    fce, arms = spawned_pipeline
    run_fce(**fce)
    arms.update(max_episode=max_episode, max_run=max_run)
    from teachers.openai_teacher import TeacherCallError
    for workers in (1, 4):
        with pytest.raises(TeacherCallError):
            run_interventions(**arms, workers=workers)
        budget = read_json(arms["output"]/"budget.json")
        assert budget["total"] <= max_run
        assert all(n <= max_episode for n in budget["episodes"].values())
        assert not (arms["output"]/"interventions.json").exists()
        shutil.rmtree(arms["output"])


def reserve_many(args):
    path, episode, count = args
    budget = TeacherBudget(path, max_episode=7, max_run=11)
    used = 0
    for index in range(count):
        try:
            budget.reserve(episode, {"index": index})
            used += 1
        except BudgetExceeded:
            pass
    return used


def test_atomic_caps_across_processes(tmp_path):
    path = tmp_path/"budget.json"
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        counts = list(pool.map(reserve_many, [(path, ep, 20) for ep in ("a", "a", "b", "b")]))
    state = TeacherBudget(path, max_episode=7, max_run=11).read()
    assert sum(counts) == state["total"] == len(state["reservations"]) == 11
    assert all(n <= 7 for n in state["episodes"].values())


def test_workspace_legacy_transition_is_pinned(tmp_path, monkeypatch):
    import analysis.mechanism.common as common
    config = load_config()
    output = workspace(tmp_path/"owned", config=config, command="fce", identity={"cap": 1})
    current = read_json(output/"identity.json")
    saved = deepcopy(current)
    saved["code_sha256"] = {"old.py": "legacy"}
    saved["inputs"] = {key.replace(str(common.ROOT), "/old/worktree"): value for key, value in saved["inputs"].items()}
    registry = {"transitions": [{"from": digest(saved["code_sha256"]), "to": digest(current["code_sha256"])}]}
    original_read = common.read_json
    monkeypatch.setattr(common, "read_json", lambda path: registry if Path(path).name == "resume_compatibility.json" else original_read(path))
    write_json(output/"identity.json", saved)
    original = (output/"identity.json").read_bytes()
    workspace(output, config=config, command="fce", identity={"cap": 1})
    assert (output/"identity.json").read_bytes() == original
    assert len(list((output/"resume_compatibility").glob("*.json"))) == 1
    for field in ("settings", "config", "dependencies", "inputs", "code_sha256"):
        changed = deepcopy(current)
        changed[field]["changed"] = True
        assert not compatible_identity(saved, changed)
