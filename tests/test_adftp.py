"""Offline collection integration: real runner, capped stub transport, fake ALFWorld."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import FakeBackend
from harness.alfworld_env import Environment
from baselines.adftp import collect as ftp
from baselines.adftp.teacher import COT_PROMPT, FTPTeacher
from sed.seed_evidence import artifact_hashes, load_verified_collection
from sed.seed_selection import select_seeds, select_full_train
from sed.storage import write_json
from teachers.capped import CappedTeacher, CollectorHalted
from teachers.openai_teacher import TeacherCallError
from teachers.collect_seeds import collect as seeds
from test_stage2b_seeds import seed_bank, factories


def stub_factory(calls, *, mismatch=(), fail=()):
    def factory(config, directory, task, identity, rate_limit_dir):
        class Raw:
            status_code = 200
            def __init__(self, text):
                self.text = text
            def parse(self):
                return {"output_text": self.text, "model": "gpt-6-luna-fake-returned", "status": "completed",
                        "usage": {"input_tokens": 12, "output_tokens": 8}, "id": "fake"}
        class Responses:
            def create(self, **kwargs):
                calls.append((task["task_id"], kwargs))
                assert "logprobs" not in kwargs and "tools" not in kwargs
                assert "SECRET" not in json.dumps(kwargs) and task["task_id"] not in json.dumps(kwargs)
                if task["task_id"] in fail:
                    raise ConnectionError("Do not persist secret SDK messages")
                if kwargs["input"][0]["content"] == COT_PROMPT:
                    return Raw("Find the cup.\n\nIgnored later paragraph.")
                prefix = "Unprefixed." if task["task_id"] in mismatch else "Find the cup. Continue."
                return Raw("Thought: " + prefix + "\nAction: look")
        responses = Responses()
        client = SimpleNamespace(responses=SimpleNamespace(with_raw_response=responses))
        return CappedTeacher(counter_path=directory / "counter.json", cap=123,
            cache_dir=directory / "cache", usage_ledger=directory / "usage.jsonl", run_id=task["task_id"],
            teacher_class=FTPTeacher, client=client, sampling={"max_output_tokens": 1024},
            rate_limit_dir=rate_limit_dir, requests_per_minute=10000, tokens_per_minute=10000000)
    return factory


def setup_source(seed_bank, tmp_path, k=6):
    root, manifest, freeze, config = seed_bank
    env, teacher = factories(root, config, set(), [])
    source = tmp_path / "source"
    seeds(select_seeds(manifest, freeze, k=k), manifest, freeze, config, source,
          env_factory=env, teacher_factory=teacher)
    config["environment"]["max_episode_steps"] = 40
    def environment(task):
        def backend(path, seed):
            value = FakeBackend(path, seed)
            value.win_at = 1
            return value
        return Environment(root, max_episode_steps=40, backend_factory=backend)
    return source, manifest, freeze, config, environment


def test_ftp_success_fallback_layout_nested_resume_and_tamper(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    original, _, _ = load_verified_collection(source, manifest, freeze)
    before = artifact_hashes(source)
    failing = original["verified"][0]["task_id"]
    calls = []
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env,
                  teacher_factory=stub_factory(calls, mismatch={failing}), subset_ks=[5])
    result = ftp.collect(source, manifest, freeze, config, tmp_path / "ftp", **kwargs)
    collection, selection, evidence = load_verified_collection(result, manifest, freeze)
    assert collection["fallback_count"] == 1 and collection["ftp_attempt_count"] == 8
    assert collection["api_attempts"] == len(calls) == 16  # 5*2 + 3*2
    assert collection["per_cell_call_cap"] == 6 * 123
    assert collection["unknown_usage_calls"] == 0
    assert collection["known_billed_tokens"]["input_tokens"] == 16 * 12
    assert [r["task_id"] for r in collection["verified"]] == [r["task_id"] for r in original["verified"]]
    fallback = collection["verified"][0]
    assert fallback["episode_hashes"] == original["verified"][0]["episode_hashes"]
    assert all(a["rejection"] == "first_reply_prefix_mismatch" for a in fallback["ftp_attempts"])
    subset, _, _ = load_verified_collection(result / "subsets/k5", manifest, freeze)
    assert subset["K"] == 5 and subset["new_teacher_calls"] == 0
    assert artifact_hashes(source) == before
    assert ftp.collect(source, manifest, freeze, config, tmp_path / "ftp", **kwargs) == result
    assert len(calls) == 16
    first = collection["verified"][1]
    episode = json.loads((evidence / first["episode_dir"] / "episode.json").read_text())
    assert episode["steps"][0]["action_raw"].startswith("Thought: Find the cup.")
    assert "Ignored later paragraph" not in episode["steps"][0]["action_raw"]
    assert episode["steps"][0]["policy_metadata"]["mechanism_teacher_billed_usage"]["input_tokens"] == 12
    cot = next((evidence / fallback["episode_dir"]).parent.glob("teacher/attempt-1/cot.json"))
    cot.write_text('{}')
    from sed.validation_cache import clear_validation_cache
    clear_validation_cache()  # Start a fresh evidence audit after deliberate mutation.
    with pytest.raises(ValueError, match="Attempt evidence changed"):
        load_verified_collection(result, manifest, freeze)


def test_ftp_interrupted_publication_reuses_executed_episode(seed_bank, tmp_path, monkeypatch):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=1)
    calls = []
    real_run = ftp.run_episode
    def interrupt(*args, **kwargs):
        real_run(*args, **kwargs)
        raise KeyboardInterrupt
    monkeypatch.setattr(ftp, "run_episode", interrupt)
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env, teacher_factory=stub_factory(calls))
    with pytest.raises(KeyboardInterrupt):
        ftp.collect(source, manifest, freeze, config, tmp_path / "ftp", **kwargs)
    assert len(calls) == 2
    monkeypatch.setattr(ftp, "run_episode", real_run)
    result = ftp.collect(source, manifest, freeze, config, tmp_path / "ftp", **kwargs)
    assert load_verified_collection(result, manifest, freeze)[0]["fallback_count"] == 0
    assert len(calls) == 2
    config["environment"]["seed"] = 77
    with pytest.raises(ValueError, match="different frozen artifact"):
        ftp.collect(source, manifest, freeze, config, tmp_path / "ftp", **kwargs)


def test_ftp_transport_failure_stays_halted(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=1)
    task_id = load_verified_collection(source, manifest, freeze)[0]["verified"][0]["task_id"]
    calls = []
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env, teacher_factory=stub_factory(calls, fail={task_id}))
    for expected in (TeacherCallError, CollectorHalted):
        with pytest.raises(expected):
            ftp.collect(source, manifest, freeze, config, tmp_path / "ftp", **kwargs)
    assert len(calls) == 1
    assert not (tmp_path / "ftp/collection/manifest.json").exists()
    assert not any("Do not persist" in p.read_text() for p in (tmp_path / "ftp").rglob('*.json*'))


@pytest.mark.parametrize("failed_source", [False, True])
def test_full_train_rule_counts_instances_and_groups_separately(data_root, tmp_path, failed_source):
    from sed.manifest import build_manifest, freeze_splits
    from sed.config import load_config
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=1)
    selection = select_full_train(manifest, freeze)
    failing = {selection["candidate_order"][-1]} if failed_source else set()
    env, teacher = factories(data_root, config, failing, [])
    source = tmp_path / "full"
    # Build the fulltrain worktree's publication schema from real stub executions.
    # This branch does not alter the existing seed collector.
    value = seeds(selection, manifest, freeze, config, source, env_factory=env, teacher_factory=teacher)
    value.update(status="complete", K=len(value["verified"]), selection_rule="all_train_v1",
                 candidates=4, failed_candidates=list(failing))
    write_json(source / "manifest.json", value)
    value, _, _ = load_verified_collection(source, manifest, freeze)
    assert value["K"] == value["unique_instances"] == 4 - failed_source
    assert value["task_groups"] == 2
    config["environment"]["max_episode_steps"] = 40
    calls = []
    result = ftp.collect(source, manifest, freeze, config, tmp_path / "ftp", rate_limit_dir=tmp_path / "limit",
                         env_factory=env, teacher_factory=stub_factory(calls))
    value, _, _ = load_verified_collection(result, manifest, freeze)
    assert value["K"] == 4 - failed_source and value["task_groups"] == 2
    assert len(calls) == 2 * (4 - failed_source)
    assert value["source_reused_failed_attempts"] == 3 * failed_source
    assert value["attempted_unique_instances"] == 4
    assert value["ftp_attempted_unique_instances"] == 4 - failed_source
    assert all(tid not in failing for tid, _ in calls)
