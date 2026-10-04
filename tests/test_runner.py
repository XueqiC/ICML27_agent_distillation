import json

import pytest

from conftest import FakeBackend
from harness.alfworld_env import Environment, ReplayMismatch
from sed.config import load_config
from sed.runner import ScriptedPolicy, replay_episode, run_episode
from sed.storage import BlobStore
from train.dataset import build_action_example


def test_full_episode_logs_and_replays(data_root, tmp_path, task):
    config = load_config()
    env = Environment(data_root, max_episode_steps=3, backend_factory=FakeBackend)
    directory = tmp_path / "run"
    result = run_episode(env, ScriptedPolicy(), task, config, directory, policy_condition="scripted")
    assert result["complete"] and result["truncated"] and len(result["steps"]) == 3
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert len(events) == 4 and sum(e["environment_steps"] for e in events) == 3
    assert (directory / "provenance.json").exists()
    env = Environment(data_root, max_episode_steps=3, backend_factory=FakeBackend)
    assert replay_episode(env, result)["matched"]
    result["steps"][1]["visible"]["observation"] = "different"
    with pytest.raises(ReplayMismatch):
        replay_episode(env, result)
    assert env.backend is None


def test_failure_is_persisted_and_environment_closed(data_root, tmp_path, task):
    class FailedPolicy:
        def generate(self, *args):
            raise RuntimeError("untrusted policy exception")
    env = Environment(data_root, backend_factory=FakeBackend)
    directory = tmp_path / "run"
    with pytest.raises(RuntimeError, match="policy_error"):
        run_episode(env, FailedPolicy(), task, load_config(), directory,
                    policy_condition="scripted")
    episode = json.loads((directory / "episode.json").read_text())
    assert episode["status"] == "error" and not episode["complete"]
    event = json.loads((directory / "events.jsonl").read_text().splitlines()[-1])
    assert event["action_raw"] is None and event["environment_steps"] == 0
    assert env.backend is None


def test_environment_error_does_not_invent_an_observation(data_root, tmp_path, task):
    class FailedStep(FakeBackend):
        def step(self, action):
            self.position += 1  # The failure occurs after a real side effect.
            raise RuntimeError("untrusted backend exception body")
    env = Environment(data_root, backend_factory=FailedStep)
    directory = tmp_path / "run"
    with pytest.raises(RuntimeError, match="step_error"):
        run_episode(env, ScriptedPolicy(), task, load_config(), directory,
                    policy_condition="scripted")
    event = json.loads((directory / "events.jsonl").read_text().splitlines()[-1])
    assert event["environment_steps"] == 1 and event["observation_ref"] is None
    assert event["feedback_source"] == "none"
    verification = BlobStore(directory / "blobs").get(event["verification_ref"])
    assert verification["environment_steps_confirmed"] == "unknown"
    assert verification["executable"] == "unknown"
    assert "untrusted backend" not in json.dumps(verification)
    assert env.backend is None


def test_training_loader_uses_explicit_whitelist(task, visible):
    record = {"task": dict(task, audit="SECRET"), "history": [],
              "visible_context": dict(visible, facts="SECRET", policy_commands="SECRET"),
              "action_parsed": "look", "event_id": "event", "label_source": "teacher_sampled_action",
              "audit": "SECRET", "verification_ref": "SECRET", "hidden": "SECRET"}
    sample = build_action_example(record)
    assert "SECRET" not in json.dumps(sample)
    assert [m["supervised"] for m in sample["messages"]] == [False, True]
    record["task"]["split"] = "test"
    with pytest.raises(ValueError):
        build_action_example(record)


def test_env_overrides_and_pinned_identity(monkeypatch):
    monkeypatch.setenv("ALFWORLD_DATA_ROOT", "/tmp/custom-data")
    monkeypatch.setenv("SED_RAW_DIR", "/tmp/custom-raw")
    config = load_config()
    assert config["environment"]["data_root"] == "/tmp/custom-data"
    assert config["paths"]["raw"] == "/tmp/custom-raw"
    assert config["models"]["student"]["revision"] == "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
    assert config["models"]["teacher"]["account_canary_passed"] is True
