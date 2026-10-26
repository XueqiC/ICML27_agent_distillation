import json

import pytest

from repair.align import align_actions, supervised_decisions, supervised_positions
from repair.execute import ResultCache, execute_actions
from repair.ledger import BudgetLedger
from repair.patch import Patch, PatchBlock, compile_patch
from repair_fakes import RepairBackend, TASK_ID, factory
from sed.react import PARSER_FEEDBACK
from sed.runner import ScriptedPolicy, run_episode
from sed.config import load_config


def test_new_execution_supplies_all_suffix_observations_and_clean_prefixes(data_root):
    original = execute_actions(factory(data_root), TASK_ID, ["look", "open door", "finish"], 40)
    patch = Patch(original.trajectory_id, [PatchBlock("key", "replace", ("take key",), start=0, end_exclusive=1)])
    repaired = execute_actions(factory(data_root), TASK_ID, compile_patch(["look", "open door", "finish"], patch), 40)
    assert repaired.success is True and original.success is False
    assert repaired.steps[1]["visible"] != original.steps[1]["visible"]
    assert "Key=True" in repaired.steps[1]["visible"]["observation"]
    context = repaired.decision_context(1)
    assert "take key" in context["history"][0]["reply"]
    assert "open door" not in context["visible_context"]["observation"]
    assert "finish" not in context["visible_context"]["observation"]
    assert "PRIVATE" not in json.dumps(repaired.to_json())
    assert "audit" not in json.dumps(context) and "success" not in json.dumps(context)
    assert set(context["visible_context"]) == {"observation", "admissible_commands"}


def test_delete_target_uses_corrected_prefix(data_root):
    original = ["take key", "inspect", "open door", "finish"]
    patch = Patch("o", [PatchBlock("delete", "delete", start=1, end_exclusive=2)])
    result = execute_actions(factory(data_root), TASK_ID, compile_patch(original, patch), 40)
    indices = supervised_positions(align_actions(original, result.executed_actions))
    assert indices == [1]
    context = result.decision_context(indices[0])
    assert context["history"][0]["reply"] == "Action: take key"
    assert "inspect" not in context["visible_context"]["observation"]
    assert result.steps[indices[0]]["action_parsed"] == "open door"


def test_supervised_decision_mapping_accounts_for_zero_step_parser_feedback(data_root):
    result = execute_actions(factory(data_root), TASK_ID,
                             ["Thought: malformed\nmissing action", "take key", "open door", "finish"], 40)
    decisions = supervised_decisions(["take key", "bad", "open door", "finish"], result)
    assert len(decisions) == 1
    assert decisions[0]["corrected_action_index"] == 1 and decisions[0]["decision_index"] == 2
    assert decisions[0]["action_target"] == "open door"
    assert "Key=True" in decisions[0]["visible_context"]["observation"]
    assert len(decisions[0]["history"]) == 2
    assert "audit" not in json.dumps(decisions)


@pytest.mark.parametrize("actions,limit,reason,success,steps", [
    (["take key", "open door", "finish", "boom"], 40, "SUCCESS", True, 3),
    (["fail", "take key"], 40, "FAILED", False, 1),
    (["look"], 40, "FAILED", False, 1),
    (["take key", "open door", "finish"], 2, "STEP_LIMIT", False, 2),
    (["take key"], 0, "STEP_LIMIT", False, 0),
    ([], 40, "FAILED", False, 0),
])
def test_success_is_only_official_won_and_stop_reasons_differ(data_root, actions, limit, reason, success, steps):
    result = execute_actions(factory(data_root), TASK_ID, actions, limit)
    assert (result.termination_reason, result.success, result.environment_steps) == (reason, success, steps)
    assert all(b.closed for b in RepairBackend.instances)


def test_missing_success_is_unknown_even_with_reward_and_done():
    class Unknown:
        seed, max_episode_steps, game_hash = 0, 40, "fake-game"
        def reset(self, task_id):
            return ({"observation": "Start", "admissible_commands": ["look"]}, 0, False, False, {})
        def step(self, action):
            return ({"observation": "End", "admissible_commands": []}, 10, True, False, {})
        def close(self):
            pass
    result = execute_actions(Unknown, TASK_ID, ["look"], 40)
    assert result.success == "unknown" and result.termination_reason == "FAILED"


def test_parser_feedback_and_inadmissibility_match_runner(data_root, tmp_path, task):
    actions = ["Thought: no action\nstill none", "not admissible", "take key", "open door", "finish"]
    result = execute_actions(factory(data_root), TASK_ID, actions, 40)
    expected = run_episode(factory(data_root)(), ScriptedPolicy(actions), task,
                           load_config(), tmp_path / "runner", policy_condition="scripted")
    for actual, old in zip(result.steps, expected["steps"]):
        for key in ("action_raw", "action_parsed", "action_outcome", "visible", "audit_sha256", "environment_steps"):
            assert actual[key] == old[key]
    assert result.steps[0]["visible"]["observation"] == PARSER_FEEDBACK
    assert result.steps[0]["feedback_source"] == "analytic_verifier"
    assert result.environment_steps == 4
    assert result.steps[1]["action_outcome"] == "inadmissible"
    limited = execute_actions(factory(data_root), TASK_ID, actions, 1)
    assert limited.environment_steps == 0 and limited.termination_reason == "STEP_LIMIT"


def test_environment_error_is_unknown_charged_sanitized_and_not_cached(data_root, tmp_path):
    cache, ledger = ResultCache(), BudgetLedger(tmp_path / "budget.jsonl")
    for _ in range(2):
        result = execute_actions(factory(data_root), TASK_ID, ["look", "boom"], 40, cache=cache, ledger=ledger)
        assert result.success == "unknown" and result.termination_reason == "ENV_ERROR"
        assert result.steps[-1]["visible"] is None and result.environment_steps == 2
        assert result.environment_steps_confirmed == 1 and result.environment_steps_unknown == 1
        assert "PRIVATE" not in json.dumps(result.to_json())
    assert cache.hits == 0
    assert ledger.summary()["interaction"]["candidate_replay_steps"] == 4
    assert "PRIVATE" not in ledger.path.read_text()


@pytest.mark.parametrize("phase", ["factory", "reset", "close"])
def test_non_step_errors_still_settle_failure_rows(data_root, tmp_path, phase):
    ledger = BudgetLedger(tmp_path / "budget.jsonl")
    class Broken(RepairBackend):
        def reset(self):
            if phase == "reset":
                raise RuntimeError("PRIVATE_RESET_ERROR")
            return super().reset()
        def close(self):
            super().close()
            if phase == "close":
                raise RuntimeError("PRIVATE_CLOSE_ERROR")
    def make():
        if phase == "factory":
            raise RuntimeError("PRIVATE_FACTORY_ERROR")
        return factory(data_root, backend=Broken)()
    result = execute_actions(make, TASK_ID, ["look"], 40, ledger=ledger)
    assert result.success == "unknown" and result.termination_reason == "ENV_ERROR"
    assert result.environment_steps == (1 if phase == "close" else 0)
    assert ledger.rows()[-1]["event"] == "failure"
    assert "PRIVATE" not in ledger.path.read_text()


def test_retry_is_linked_and_preserves_actual_cost(data_root, tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.jsonl")
    execute_actions(factory(data_root), TASK_ID, ["look"], 40, ledger=ledger)
    previous = ledger.rows()[-1]["attempt_id"]
    execute_actions(factory(data_root), TASK_ID, ["look"], 40, ledger=ledger, retry_of=previous)
    assert ledger.rows()[-2]["event"] == "retry"
    assert ledger.rows()[-1]["parent_attempt_id"] == previous
    assert ledger.summary()["interaction"]["candidate_replay_steps"] == 2


def test_cache_hits_have_zero_steps_and_independent_evidence(data_root, tmp_path):
    cache, ledger = ResultCache(), BudgetLedger(tmp_path / "budget.jsonl")
    actions = ["take key", "open door", "finish"]
    first = execute_actions(factory(data_root), TASK_ID, actions, 40, cache=cache, ledger=ledger)
    second = execute_actions(factory(data_root), TASK_ID, actions, 40, cache=cache, ledger=ledger)
    assert second.cache_hit and second.success is True
    assert second.environment_steps == sum(s["environment_steps"] for s in second.steps) == 0
    assert second.cache_source_trajectory_id == first.trajectory_id
    assert first.environment_steps == 3 and second.executed_actions == actions
    assert ledger.summary()["interaction"]["candidate_replay_steps"] == 3
    assert [r["event"] for r in ledger.rows()] == ["attempt", "success", "attempt", "cache_hit"]
    second.steps[0]["visible"]["observation"] = "mutated by caller"
    third = execute_actions(factory(data_root), TASK_ID, actions, 40, cache=cache)
    assert third.steps[0]["visible"] == first.steps[0]["visible"]


def test_cache_key_separates_limits_seed_version_game_and_reset_fingerprint(data_root):
    cache = ResultCache()
    actions = ["take key", "open door", "finish"]
    def run(**kwargs):
        return execute_actions(kwargs.pop("factory", factory(data_root)), TASK_ID, actions,
                               kwargs.pop("max_steps", 40), cache=cache, **kwargs)
    run(version="v1")
    assert not run(version="v1", max_steps=1).cache_hit
    assert not run(version="v1", factory=factory(data_root, seed=1)).cache_hit
    assert not run(version="v2").cache_hit
    class DifferentReset(RepairBackend):
        def reset(self):
            self.key = True
            return self.state()
    assert not run(version="v1", factory=factory(data_root, backend=DifferentReset)).cache_hit
    path = data_root / TASK_ID / "game.tw-pddl"
    path.write_text('{"solvable": true, "changed": true}')
    assert not run(version="v1").cache_hit


def test_cache_does_not_alias_whitespace_sensitive_commands(data_root):
    cache = ResultCache()
    execute_actions(factory(data_root), TASK_ID, ["take key"], 40, cache=cache)
    result = execute_actions(factory(data_root), TASK_ID, ["take  key"], 40, cache=cache)
    assert not result.cache_hit and result.steps[0]["action_outcome"] == "inadmissible"


def test_replay_never_uses_snapshots(data_root, monkeypatch):
    from harness.alfworld_env import Environment
    def forbidden(*args):
        pytest.fail("M0a must use reset and replay")
    monkeypatch.setattr(Environment, "snapshot", forbidden)
    monkeypatch.setattr(Environment, "restore", forbidden)
    result = execute_actions(factory(data_root), TASK_ID, ["take key", "open door", "finish"], 40)
    assert result.success is True
