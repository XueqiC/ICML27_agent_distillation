from copy import deepcopy
import json

import pytest

from conftest import FakeBackend
from harness.alfworld_env import Environment, ReplayMismatch
from harness.alfworld_env.environment import partition_state
from sed.storage import digest


def test_oracle_fields_cannot_reach_visible_payload():
    state = {"feedback": "You see a table.", "admissible_commands": ["look"], "won": True,
             "extra.expert_plan": ["SECRET_PLAN"], "policy_commands": ["SECRET_POLICY"],
             "facts": ["SECRET_FACTS"], "pddl": "SECRET_PDDL", "audit": {"hidden": "SECRET"},
             "arbitrary_future_oracle": {"nested": "SECRET_ORACLE"}}
    visible, audit = partition_state(state)
    assert set(visible) == {"observation", "admissible_commands"}
    assert "SECRET" not in json.dumps(visible)
    assert audit["won"] is True and audit["internal_facts"] == ["SECRET_FACTS"]


def test_prefix_restore_and_branch(data_root):
    env = Environment(data_root, backend_factory=FakeBackend)
    initial = env.reset("train/room_a/trial_0")
    env.step("look")
    expected = env.step("go north")
    snapshot = env.snapshot()
    env.step("look")
    restored = env.restore(snapshot)
    assert restored == expected
    assert env.actions == ["look", "go north"]
    assert env.step("look")[0] != initial[0]
    env.close()


@pytest.mark.parametrize("field", ["feedback", "admissible_commands", "_facts"])
def test_replay_divergence_invalidates_branch(data_root, field):
    class Drifting(FakeBackend):
        generation = 0
        def __init__(self, path, seed):
            super().__init__(path, seed)
            type(self).generation += 1
        def state(self):
            value = super().state()
            if type(self).generation > 1 and self.position == 1:
                value[field] = "different" if field == "feedback" else ["different"]
            return value
    env = Environment(data_root, backend_factory=Drifting)
    env.reset("train/room_a/trial_0")
    env.step("look")
    snapshot = env.snapshot()
    with pytest.raises(ReplayMismatch):
        env.restore(snapshot)
    with pytest.raises(RuntimeError):
        env.step("look")


def test_snapshot_tamper_and_game_changes(data_root):
    env = Environment(data_root, backend_factory=FakeBackend)
    env.reset("train/room_a/trial_0")
    snapshot = env.snapshot()
    bad = deepcopy(snapshot)
    bad["seed"] = 5
    with pytest.raises(ReplayMismatch):
        env.restore(bad)
    path = data_root / snapshot["task_id"] / "game.tw-pddl"
    path.write_text('{"solvable": true, "changed": true}')
    with pytest.raises(ReplayMismatch, match="Reset state"):
        env.restore(snapshot)


def test_truncation_is_distinct_from_success(data_root):
    env = Environment(data_root, max_episode_steps=2, backend_factory=FakeBackend)
    env.reset("train/room_a/trial_0")
    env.step("look")
    _, _, terminated, truncated, _ = env.step("look")
    assert not terminated and truncated
    snap = env.snapshot()
    assert env.restore(snap)[3] is True
    with pytest.raises(RuntimeError):
        env.step("look")
    path = data_root / "train/room_a/trial_0/game.tw-pddl"
    path.write_text('{"win_at": 2}')
    env.reset("train/room_a/trial_0")
    env.step("look")
    _, reward, terminated, truncated, _ = env.step("look")
    assert reward == 1 and terminated and not truncated
    env.close()


def test_sealed_branch_rejected(data_root):
    env = Environment(data_root, backend_factory=FakeBackend, split="test")
    env.reset("valid_unseen/room_a/trial_0")
    with pytest.raises(ValueError, match="train/dev"):
        env.snapshot()
    env.close()


@pytest.mark.parametrize("task_id", ["valid_unseen/room_a/trial_0", "valid_seen/room_a/trial_0",
                                    "../outside", "/tmp/outside"])
def test_native_split_and_path_guards(data_root, task_id):
    env = Environment(data_root, backend_factory=FakeBackend)
    with pytest.raises(ValueError):
        env.reset(task_id)
    assert env.backend is None
