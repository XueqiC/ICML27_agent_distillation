"""Offline SmartAD integration: stub Responses, real runner, fake ALFWorld."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.smartad import collect as sm
from baselines.smartad.select import select, validate_selection
from baselines.smartad.teacher import SmartADTeacher, REQUESTED_SAMPLING, CALLS_PER_CANDIDATE
from baselines.smartad.pipeline import training_command
from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.seed_evidence import artifact_hashes, load_verified_collection
from sed.seed_selection import select_seeds, select_full_train
from sed.storage import write_json
from teachers.capped import BudgetExceeded, CollectorHalted
from teachers.collect_seeds import collect as seeds
from teachers.openai_teacher import Teacher, TeacherCallError
from test_stage2b_seeds import seed_bank


class Raw:
    status_code = 200
    def __init__(self, text="Thought: Find it.\nAction: look"):
        self.text = text
    def parse(self):
        return {"output_text": self.text, "model": "gpt-6-luna-fake", "status": "completed",
                "usage": {"input_tokens": 12, "output_tokens": 8}, "id": "fake"}


class Unsupported(Exception):
    status_code = 400
    body = {"error": {"param": "temperature", "code": None, "type": "invalid_request_error"}}


def factory(calls, *, unsupported=False, transport_fail=False, malformed=False):
    def make(config, directory, task, identity, limiter, sampling_root):
        class Responses:
            def create(self, **kwargs):
                calls.append(kwargs)
                assert "SECRET" not in json.dumps(kwargs)
                assert "logprobs" not in kwargs and "tools" not in kwargs
                if transport_fail:
                    raise ValueError("SECRET provider body must never persist")
                if unsupported and "temperature" in kwargs:
                    raise Unsupported("SECRET provider body must never persist")
                return Raw("Thought: Won!" if malformed else "Thought: Find it.\nAction: look")
        return SmartADTeacher(counter_path=directory / "counter.json", cap=CALLS_PER_CANDIDATE,
            cache_dir=directory / "cache", usage_ledger=directory / "usage.jsonl",
            run_id=str(directory), sampling_root=sampling_root, baseline_method="smartad",
            client=SimpleNamespace(responses=SimpleNamespace(with_raw_response=Responses())),
            rate_limit_dir=limiter, requests_per_minute=100000, tokens_per_minute=100000000)
    return make


def setup_source(seed_bank, tmp_path, *, k=1, default=False, full=False, failed_source=False):
    root, manifest, freeze, config = seed_bank
    config["environment"].update(max_episode_steps=40, data_root=str(root))
    config["models"]["teacher"]["sampling"] = {"max_output_tokens": 1024} if default else dict(REQUESTED_SAMPLING)
    selection = select_full_train(manifest, freeze) if full else select_seeds(manifest, freeze, k=k)
    failing = selection["candidate_order"][-1] if failed_source else None
    def env(task):
        def backend(path, seed):
            result = FakeBackend(path, seed)
            result.win_at = 99 if task["task_id"] == failing else 1
            return result
        return Environment(root, max_episode_steps=1 if task["task_id"] == failing else 40, backend_factory=backend)
    def teacher(task, cache, ledger):
        return Teacher(cache_dir=cache, usage_ledger=ledger, run_id=task["task_id"],
            sampling=config["models"]["teacher"]["sampling"],
            client=SimpleNamespace(responses=SimpleNamespace(create=lambda **_: Raw().parse())),
            requests_per_minute=100000, tokens_per_minute=100000000)
    source = tmp_path / "source"
    seeds(selection, manifest, freeze, config, source, env_factory=env, teacher_factory=teacher)
    return source, manifest, freeze, config, env


def test_reuse_nine_new_resume_selection_nested_and_tamper(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=6)
    before = artifact_hashes(source)
    calls = []
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env, teacher_factory=factory(calls), subset_ks=[5])
    pool_root = sm.collect(source, manifest, freeze, config, tmp_path / "pool", **kwargs)
    pool = sm.load_pool(pool_root, manifest, freeze)[0]
    assert len(calls) == 6 * 9 == pool["api_attempts"]
    assert all(t["reuse"] and len(t["candidates"]) == 10 for t in pool["tasks"])
    assert sm.collect(source, manifest, freeze, config, pool_root, **kwargs) == pool_root
    assert len(calls) == 54
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    write_json(snapshot / "config.json", {"model_type": "gemma4_unified"})
    scored = []
    def scorer(episode):
        scored.append(episode["episode_id"])
        nll = .1 if len(scored) % 10 in (2, 3) else 2.0
        return {"nll": nll, "turns": [{"nll_mean": nll}]}
    selected = select(pool_root, snapshot, config, manifest, freeze, tmp_path / "selected", scorer=scorer)
    collection, _, _ = load_verified_collection(selected, manifest, freeze)
    assert collection["fallback_count"] == 0 and collection["api_attempts"] == 54
    assert collection["per_cell_call_cap"] == 6 * 410
    assert collection["known_billed_tokens"]["input_tokens"] == 54 * 12
    assert len(scored) == 60
    assert all(r["smartad_selection"]["chosen_id"] == "candidate-02" for r in collection["verified"])
    subset, _, _ = load_verified_collection(selected / "subsets/k5", manifest, freeze)
    assert subset["K"] == 5 and subset["new_teacher_calls"] == 0
    validate_selection(selected / "subsets/k5", manifest, freeze, snapshot)
    select(pool_root, snapshot, config, manifest, freeze, tmp_path / "selected", scorer=scorer)
    assert len(scored) == 60 and artifact_hashes(source) == before
    write_json(snapshot / "changed.json", {})
    with pytest.raises(ValueError, match="binding changed"):
        validate_selection(selected, manifest, freeze, snapshot)
    candidate = Path(pool["tasks"][0]["candidates"][1]["episode_dir"]) / "episode.json"
    candidate.write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        sm.load_pool(pool_root, manifest, freeze)


def test_mismatched_sampling_ten_new_and_zero_correct_fallback(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, default=True)
    calls = []
    # A losing, truncated rollout is still real execution. Teacher claims do not count.
    def losing(task):
        return Environment(config["environment"]["data_root"], max_episode_steps=1, backend_factory=FakeBackend)
    pool_root = sm.collect(source, manifest, freeze, config, tmp_path / "pool",
        rate_limit_dir=tmp_path / "limiter", env_factory=losing, teacher_factory=factory(calls))
    pool = sm.load_pool(pool_root, manifest, freeze)[0]
    assert len(calls) == 10 and not pool["tasks"][0]["reuse"]
    assert pool["tasks"][0]["reuse_reason"] == "source_collection_sampling_differs"
    assert not any(c["correct"] for c in pool["tasks"][0]["candidates"])
    snapshot = tmp_path / "snapshot"; snapshot.mkdir()
    def no_score(_):
        pytest.fail("Incorrect trajectories must never be scored")
    selected = select(pool_root, snapshot, config, manifest, freeze, tmp_path / "selected", scorer=no_score)
    value, _, _ = load_verified_collection(selected, manifest, freeze)
    original, _, _ = load_verified_collection(source, manifest, freeze)
    assert value["fallback_count"] == 1
    assert value["verified"][0]["episode_hashes"] == original["verified"][0]["episode_hashes"]
    decisions = json.loads((selected / "smartad_selection.json").read_text())
    assert decisions["tasks"][0]["chosen_id"] == "trusted-fallback"
    assert all(r["score"] is None for r in decisions["tasks"][0]["scores"])


def test_explicit_sampling_fallback_then_trusted_reuse(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, default=True)
    calls = []
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env,
                  teacher_factory=factory(calls, unsupported=True))
    root = sm.collect(source, manifest, freeze, config, tmp_path / "pool", **kwargs)
    pool = sm.load_pool(root, manifest, freeze)[0]
    assert pool["tasks"][0]["reuse"] and len(calls) == 10  # one rejected request + nine real rollouts
    assert sum("temperature" in c for c in calls) == 1
    assert pool["sampling_fallback"]["parameter"] == "temperature"
    assert pool["api_attempts"] == 10
    sm.collect(source, manifest, freeze, config, root, **kwargs)
    assert len(calls) == 10
    assert not any("SECRET" in p.read_text() for p in root.rglob("*.json*"))


def test_completed_execution_survives_interrupt_without_resampling(seed_bank, tmp_path, monkeypatch):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    calls = []
    real_run = sm.run_episode
    def interrupt(*args, **kwargs):
        real_run(*args, **kwargs)
        raise KeyboardInterrupt
    monkeypatch.setattr(sm, "run_episode", interrupt)
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env, teacher_factory=factory(calls))
    with pytest.raises(KeyboardInterrupt):
        sm.collect(source, manifest, freeze, config, tmp_path / "pool", **kwargs)
    assert len(calls) == 1
    monkeypatch.setattr(sm, "run_episode", real_run)
    sm.collect(source, manifest, freeze, config, tmp_path / "pool", **kwargs)
    assert len(calls) == 9


def test_transport_failure_halts_and_never_resamples(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    calls = []
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env,
                  teacher_factory=factory(calls, transport_fail=True))
    for error in (TeacherCallError, CollectorHalted):
        with pytest.raises(error):
            sm.collect(source, manifest, freeze, config, tmp_path / "pool", **kwargs)
    assert len(calls) == 1 and not (tmp_path / "pool/candidates.json").exists()


def test_cap_and_cache_hit_at_cap(tmp_path, task, visible):
    calls = []
    teacher = factory(calls)({}, tmp_path / "teacher", task, {}, tmp_path / "limit", tmp_path)
    for i in range(41):
        teacher.generate([], task, {**visible, "observation": str(i)})
    assert teacher._read_counter()["attempts"] == len(calls) == 41
    teacher.generate([], task, {**visible, "observation": "40"})
    assert len(calls) == 41
    with pytest.raises(BudgetExceeded):
        teacher.generate([], task, {**visible, "observation": "42"})
    assert len(calls) == 41


def test_reject_win_with_failed_tool_turn(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    collection, _, root = load_verified_collection(source, manifest, freeze)
    directory = root / collection["verified"][0]["episode_dir"]
    episode = json.loads((directory / "episode.json").read_text())
    assert sm.correct(directory, episode)
    failed = dict(episode["steps"][0], action_outcome="parser_error")
    episode["steps"] = [failed, episode["steps"][0]]
    assert not sm.correct(directory, episode)
    failed["action_outcome"] = "inadmissible"
    assert not sm.correct(directory, episode)


def test_training_recipe_and_launchers():
    args = SimpleNamespace(config="base.yaml", snapshot="base", verified="selected",
                           train_python="python")
    command = training_command(args, "out")
    flags = dict(zip(command[3::2], command[4::2]))
    assert flags["--objective"] == "smartad" and flags["--original-batch"] == "4"
    assert flags["--epochs"] == "2" and flags["--packing"] == "turn"
    assert flags["--adaptation"] == "kang_lora" and flags["--learning-rate"] == "2e-4"
    assert flags["--max-length"] == "32768" and flags["--seed"] == "0"
    launcher = Path("scripts/rai_collect_smartad.sh").read_text()
    assert "--subset-k 5 10 20 50" in launcher and "seeds_full_train_r2" in launcher
    assert "SMARTAD_RATE_LIMIT_DIR" in launcher and "CUDA_VISIBLE_DEVICES=''" in launcher


def test_unknown_attempt_never_reexecutes(seed_bank, tmp_path, monkeypatch):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    calls = []
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(sm, "run_episode", interrupt)
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env, teacher_factory=factory(calls))
    with pytest.raises(KeyboardInterrupt):
        sm.collect(source, manifest, freeze, config, tmp_path / "pool", **kwargs)
    with pytest.raises(TeacherCallError, match="NOT complete"):
        sm.collect(source, manifest, freeze, config, tmp_path / "pool", **kwargs)
    assert not calls


@pytest.mark.parametrize("failed_source", [False, True])
def test_full_train_source_and_publication(data_root, tmp_path, failed_source):
    from sed.config import load_config
    from sed.manifest import build_manifest, freeze_splits
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    bank = (data_root, manifest, freeze, load_config())
    source, manifest, freeze, config, env = setup_source(bank, tmp_path, full=True, failed_source=failed_source)
    calls = []
    root = sm.collect(source, manifest, freeze, config, tmp_path / "pool",
        rate_limit_dir=tmp_path / "limit", env_factory=env, teacher_factory=factory(calls))
    snapshot = tmp_path / "snapshot"; snapshot.mkdir()
    selected = select(root, snapshot, config, manifest, freeze, tmp_path / "selected",
                      scorer=lambda _: {"nll": 1.0})
    collection, selection, _ = load_verified_collection(selected, manifest, freeze)
    assert selection["selection_rule"] == "all_train_v1"
    assert collection["K"] == collection["unique_instances"] == 4 - failed_source
    assert collection["task_groups"] == 2
    assert collection["attempted_unique_instances"] == 4
    assert len(calls) == 9 * (4 - failed_source)
    assert collection["candidate_slots"] == 10 * (4 - failed_source)
    assert sum(not r["won"] for r in collection["attempts"]) == 3 * failed_source
