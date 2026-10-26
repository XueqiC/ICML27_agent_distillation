from copy import deepcopy
from dataclasses import replace
import json

import pytest

from repair.driver import repair_one_failure
from repair.execute import execute_actions
from repair.ledger import BudgetLedger
from repair.prompts import diagnosis_prompt, proposal_prompt, rationale_prompt
from repair.runtime import CaseSession, CaseStopped, Limits
from repair.schemas import SchemaError, parse_candidates, parse_diagnosis, parse_label
from repair_fakes import TASK_ID, factory, RepairBackend
from repair_teacher_fakes import DIAGNOSIS, FakeTeacher, proposal, result, label
from sed.config import load_config
from teachers.openai_teacher import TeacherCallError


def setup_case(data_root, tmp_path, task, responses=None, limits=None):
    original = execute_actions(factory(data_root), TASK_ID, ["look", "look", "finish"], 40)
    teacher = FakeTeacher(responses if responses is not None else [DIAGNOSIS, proposal(original)])
    session = CaseSession(task=task, teacher=teacher, ledger=BudgetLedger(tmp_path / "budget.jsonl"),
                          config=load_config("configs/base_e4b.yaml"), limits=limits)
    return original, teacher, session


def test_joint_repair_uses_one_original_confirmation_and_e1(data_root, tmp_path, task):
    original, teacher, session = setup_case(data_root, tmp_path, task)
    outcome = repair_one_failure(original, env_factory=factory(data_root), session=session)
    repair = outcome.repair
    assert repair.repair_status == "VERIFIED"
    assert repair.joint_necessity["both_necessary"] is True
    assert repair.replays_used == 5
    assert [p.kind for p in teacher.requests] == ["diagnosis", "proposal"]
    assert [r["success"] for r in outcome.e1] == [False, False, False, True, True]
    assert outcome.e1[-1]["cache_hit"] is False
    requests = [r for r in session.ledger.rows() if r["event"] == "teacher_request"]
    assert all(len(r["details"]["request"]["template_sha256"]) == 64 for r in requests)
    assert session.ledger.summary()["supervision"]["teacher_calls"]["diagnosis"] == 1


def test_round_two_contains_only_real_replay_feedback_and_full_original_indices(data_root, tmp_path, task):
    original, teacher, session = setup_case(data_root, tmp_path, task)
    bad = proposal(original, blocks=[{"block_id": "only", "op": "replace", "start": 0,
                                      "end_exclusive": 1, "actions": ["take key"]}])
    teacher.responses = [DIAGNOSIS, bad, proposal(original)]
    outcome = repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert outcome.repair.repair_status == "VERIFIED"
    feedback = teacher.requests[2].payload["replay_feedback"]
    assert len(feedback) == 1 and feedback[0]["success"] is False
    assert "Key=True" in feedback[0]["steps"][0]["observation"]
    assert feedback[0]["steps"][0]["action"] == "take key"
    assert feedback[0]["steps"][1]["action"] == "look"
    assert "admissible" not in json.dumps(feedback) and "audit" not in json.dumps(feedback)
    assert [c["round"] for c in outcome.candidates] == [1, 2]


def test_dedupe_and_out_of_bounds_are_logged_without_execution(data_root, tmp_path, task):
    original, teacher, session = setup_case(data_root, tmp_path, task)
    p = proposal(original)
    duplicate = deepcopy(p["candidates"][0])
    duplicate["blocks"][0]["block_id"] = "renamed"
    invalid = deepcopy(duplicate)
    invalid["blocks"][0]["start"] = 200
    p["candidates"].extend([duplicate, invalid])
    teacher.responses = [DIAGNOSIS, p]
    outcome = repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert outcome.repair.replays_used == 5
    assert session.shared["format_retries"] == 0
    reasons = {c.get("reason") for c in outcome.candidates}
    assert {"duplicate_compiled_actions", "range_out_of_bounds_or_empty"} <= reasons


@pytest.mark.parametrize("responses,expected_calls,verified", [
    (["not json", DIAGNOSIS], 3, True), (["not json", "also not json"], 2, False),
])
def test_single_shared_format_retry(data_root, tmp_path, task, responses, expected_calls, verified):
    original, teacher, session = setup_case(data_root, tmp_path, task)
    teacher.responses = [*responses, proposal(original)]
    outcome = repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert len(teacher.requests) == expected_calls
    assert (outcome.repair.repair_status == "VERIFIED") == verified
    assert teacher.requests[1].format_retry is True
    if not verified:
        assert outcome.stop_reason == "format_retry_exhausted"
    assert session.shared["format_retries"] == 1


def test_retry_consumed_by_diagnosis_is_unavailable_to_proposal(data_root, tmp_path, task):
    original, teacher, session = setup_case(data_root, tmp_path, task)
    teacher.responses = ["bad", DIAGNOSIS, "bad", proposal(original)]
    out = repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert len(teacher.requests) == 3 and out.repair.repair_status == "INVALID_PATCH"


@pytest.mark.parametrize("limits,reason", [
    (Limits(total_calls=0), "total_call_cap"), (Limits(diagnosis_calls=0), "diagnosis_call_cap"),
    (Limits(input_tokens=1), "input_token_cap"), (Limits(output_tokens=8191), "output_token_cap"),
])
def test_caps_stop_before_teacher_call(data_root, tmp_path, task, limits, reason):
    original, teacher, session = setup_case(data_root, tmp_path, task, limits=limits)
    out = repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert teacher.requests == [] and out.stop_reason == reason


@pytest.mark.parametrize("limit,expected_calls,expected_replays", [(0, 0, 0), (1, 3, 1), (3, 2, 3)])
def test_replay_budget_reserves_confirmation(data_root, tmp_path, task, limit, expected_calls, expected_replays):
    original, teacher, session = setup_case(data_root, tmp_path, task, limits=Limits(replays=limit))
    teacher.responses.append(proposal(original))
    out = repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert len(teacher.requests) == expected_calls
    assert out.repair.replays_used == expected_replays
    if limit == 3:
        assert out.repair.repair_status == "VERIFIED"
        assert out.repair.minimality_status == "ABLATION_BUDGET_EXHAUSTED"
        assert out.e1[-1]["success"] is True


def test_mismatch_stops_before_teacher(data_root, tmp_path, task):
    original, teacher, session = setup_case(data_root, tmp_path, task)
    class Changed(RepairBackend):
        def reset(self):
            self.key = True
            return self.state()
    outcome = repair_one_failure(original, env_factory=factory(data_root, backend=Changed), session=session)
    assert outcome.repair.repair_status == "NONREPLAYABLE" and not teacher.requests


def test_over_cap_response_is_format_rejected_and_retried(data_root, tmp_path, task):
    original, teacher, session = setup_case(data_root, tmp_path, task)
    teacher.responses = [result(DIAGNOSIS, usage={"input_tokens": 20, "output_tokens": 8193}),
                         DIAGNOSIS, proposal(original)]
    out = repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert out.repair.repair_status == "VERIFIED" and session.shared["format_retries"] == 1
    assert any(r["reason"] == "output_cap" for r in session.ledger.rows())


def test_transport_failure_sanitized_and_not_retried_as_format(data_root, tmp_path, task):
    original, teacher, session = setup_case(data_root, tmp_path, task, responses=[RuntimeError("PRIVATE_SECRET")])
    with pytest.raises(TeacherCallError, match="stage failed"):
        repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert len(teacher.requests) == 1 and session.shared["format_retries"] == 0
    assert "PRIVATE_SECRET" not in session.ledger.path.read_text()


def test_optional_handback_is_real_replay_and_does_not_filter_repair(data_root, tmp_path, task):
    original, teacher, session = setup_case(data_root, tmp_path, task)
    prefixes = []
    def policy(history, visible):
        prefixes.append((history, visible))
        return "fail"
    out = repair_one_failure(original, env_factory=factory(data_root), session=session, student_policy=policy)
    assert out.repair.repair_status == "VERIFIED" and len(out.handbacks) == 2
    assert all(h["handback_success"] is False for h in out.handbacks)
    assert prefixes[0][0][0]["reply"] == "Action: take key"
    assert "Key=True" in prefixes[0][1]["observation"]
    assert out.repair.replays_used == 7


def test_prompt_whitelist_strips_every_forbidden_field(data_root, tmp_path, task):
    original, _, _ = setup_case(data_root, tmp_path, task)
    tainted = deepcopy(task)
    forbidden = {"extra.expert_plan": "ORIGINAL_TEACHER_DEMO", "policy_commands": "COMMAND_SECRET",
                 "hidden_state": "STATE_SECRET", "pddl_facts": "FACT_SECRET", "other_tasks": "OTHER_TASK",
                 "evaluation_data": "SEALED_VALUE", "hidden_success": "HIDDEN_SUCCESS"}
    tainted.update(forbidden)
    trace = original.to_json()
    trace.pop("trajectory_id")
    for step in [trace["initial"], *trace["steps"]]:
        step.update(forbidden)
        step["visible"].update(forbidden)
        step["visible"]["admissible_commands"] = ["ADMISSIBLE_SECRET"]
    from repair.records import TrajectoryRecord
    original = TrajectoryRecord.from_json(trace)
    diagnosis = parse_diagnosis(json.dumps(DIAGNOSIS), original.trajectory_id, 3)
    prompts = [diagnosis_prompt(tainted, original, "command", 40),
               proposal_prompt(tainted, original, diagnosis, "command", 40, executions=[original]),
               rationale_prompt(tainted, "initial", [{"action_index": 0, "reply": "Thought: known",
                   "action": "look", "observation": "public", **forbidden}], "look")]
    serialized = json.dumps([p.messages() for p in prompts])
    for value in [*forbidden, *forbidden.values(), "ADMISSIBLE_SECRET", "admissible_commands", "audit_sha256"]:
        assert value not in serialized
    assert TASK_ID not in serialized


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(unknown="secret"),
    lambda p: p["candidates"].extend([p["candidates"][0]] * 3),
    lambda p: p["candidates"][0]["blocks"].extend([p["candidates"][0]["blocks"][0]] * 3),
    lambda p: p["candidates"][0]["blocks"][0].update(start=True),
    lambda p: p["candidates"][0]["blocks"][0].update(actions=["Action: look"]),
    lambda p: p["candidates"][0].update(original_trajectory_id="other"),
])
def test_patch_schema_is_strict(data_root, tmp_path, task, mutate):
    original, _, _ = setup_case(data_root, tmp_path, task)
    p = proposal(original)
    mutate(p)
    with pytest.raises(SchemaError):
        parse_candidates(json.dumps(p), original.trajectory_id)


def test_label_tokenizer_cap_and_strict_json():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text)
    with pytest.raises(SchemaError):
        parse_label(json.dumps(label("x" * 65)), Tokenizer())
    with pytest.raises(SchemaError):
        parse_label(json.dumps(label("word " * 65)))
    with pytest.raises(SchemaError):
        parse_label('{"reasoning":"x","reasoning":"y","evidence_steps":[],"unsupported":false}')
    assert parse_label(json.dumps(label("word " * 64))).unsupported is False


@pytest.mark.parametrize("change", [{"split": "dev"}, {"sealed": True}, {"task_set": "D_probe"}])
def test_generation_rejects_evaluation_tasks_before_any_call(tmp_path, task, change):
    with pytest.raises(ValueError):
        CaseSession(task={**task, **change}, teacher=FakeTeacher([]),
                    ledger=BudgetLedger(tmp_path / "budget.jsonl"), config=load_config("configs/base_e4b.yaml"))


def test_six_failed_candidates_stay_negative_and_never_use_a_demonstration(data_root, tmp_path, task):
    original, teacher, session = setup_case(data_root, tmp_path, task)
    proposals = []
    for batch in range(2):
        candidates = []
        for i in range(3):
            candidates += proposal(original, blocks=[{"block_id": "p", "op": "replace", "start": 0,
                "end_exclusive": 3, "actions": ["look"] * (1 + batch * 3 + i)}])["candidates"]
        proposals.append({"candidates": candidates})
    teacher.responses = [DIAGNOSIS, *proposals]
    out = repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert out.repair.repair_status == "NO_SUCCESS_WITHIN_BUDGET"
    assert len(out.candidates) == 6 and out.repair.replays_used == 7
    assert len(teacher.requests) == 3


def test_success_choice_uses_executed_distance_not_proposal_order(data_root, tmp_path, task):
    original, teacher, session = setup_case(data_root, tmp_path, task)
    large = proposal(original, blocks=[{"block_id": "large", "op": "replace", "start": 0,
        "end_exclusive": 3, "actions": ["inspect", "take key", "open door", "finish", "boom"]}])
    large["candidates"] += proposal(original)["candidates"]
    teacher.responses = [DIAGNOSIS, large]
    out = repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert out.repair.cost[0] == 2
    assert {b.block_id for b in out.repair.selected_patch.blocks} == {"key", "door"}
    assert len(out.candidates[0]["execution"]["steps"]) == 4  # success stops before boom


def test_case_budget_cannot_be_reset_by_recreating_session(data_root, tmp_path, task):
    _, teacher, session = setup_case(data_root, tmp_path, task)
    with pytest.raises(ValueError, match="already started"):
        CaseSession(task=task, teacher=teacher, ledger=session.ledger, config=session.config)


@pytest.mark.parametrize("n", [3, 4])
def test_e1_measures_each_single_and_each_revert_for_larger_joint_repairs(data_root, tmp_path, task, n):
    actions = ["look", "look", *(["fail"] * (n - 2)), "finish"]
    original = execute_actions(factory(data_root), TASK_ID, actions, 40)
    # A terminal fail means only the executed original prefix can be edited.
    # Use a backend where the bad actions are nonterminal but block success.
    class Multi(RepairBackend):
        def step(self, action):
            result = super().step(action)
            if action == "fail":
                self.lost = False
                return self.state(), 0, False
            if action == "finish" and "fail" in self.actions:
                self.won = False
                return self.state(), 0, False
            return result
    original = execute_actions(factory(data_root, backend=Multi), TASK_ID, actions, 40)
    p = proposal(original)
    p["candidates"][0]["blocks"] += [{"block_id": f"delete-{i}", "op": "delete", "start": i,
        "end_exclusive": i + 1, "actions": []} for i in range(2, n)]
    teacher = FakeTeacher([DIAGNOSIS, p])
    session = CaseSession(task=task, teacher=teacher, ledger=BudgetLedger(tmp_path / "budget.jsonl"),
                          config=load_config("configs/base_e4b.yaml"))
    out = repair_one_failure(original, env_factory=factory(data_root, backend=Multi), session=session)
    assert len(out.repair.selected_patch.blocks) == n
    assert out.repair.joint_necessity["all_necessary"] is True
    assert [row["success"] for row in out.e1] == [False, *([False] * n), True, True]
    assert out.repair.replays_used <= 24


def test_handback_prefix_drift_never_calls_student(data_root, tmp_path, task):
    target = execute_actions(factory(data_root), TASK_ID, ["take key", "open door", "finish"], 40)
    class Drift(RepairBackend):
        def reset(self):
            value = super().reset()
            value["feedback"] += " changed"
            return value
    calls = []
    result = execute_actions(factory(data_root, backend=Drift), TASK_ID, ["take key"], 40,
        continuation=lambda *args: calls.append(args), prefix_reference=target,
        interaction_kind="handback_diagnostic_steps")
    assert not calls and result.success == "unknown"
    assert result.error_type.startswith("handback_prefix_check:")


@pytest.mark.parametrize("allow_retry", [False, True])
def test_existing_capped_transport_429_retries_obey_case_preflight(tmp_path, task, allow_retry):
    from types import SimpleNamespace
    from teachers.capped import CappedTeacher
    from repair.runtime import RepairTeacher
    from repair.prompts import Prompt
    from test_capped_teacher import SequenceClient, http_429
    from test_stage2c_teacher import FakeClock
    clock = FakeClock()
    success = SimpleNamespace(status_code=200, headers={}, parse=lambda: {
        "output_text": json.dumps(DIAGNOSIS), "status": "completed", "model": "fake-model",
        "id": "fake", "_request_id": "fake-request", "usage": {"input_tokens": 13, "output_tokens": 7}})
    sdk = SequenceClient([http_429(), success])
    teacher = CappedTeacher(counter_path=tmp_path / "counter.json", cap=54,
        cache_dir=tmp_path / "cache", usage_ledger=tmp_path / "usage.jsonl", run_id="repair-test",
        client=sdk, teacher_class=RepairTeacher, sampling={"max_output_tokens": 8192},
        clock=clock, sleep=clock.sleep, jitter=lambda lo, hi: 0)
    session = CaseSession(task=task, teacher=teacher, ledger=BudgetLedger(tmp_path / "repair-budget.jsonl"),
        config=load_config("configs/base_e4b.yaml"), limits=Limits(total_calls=2 if allow_retry else 1))
    prompt = Prompt("diagnosis", {"task": "public task"})
    if allow_retry:
        diagnosis = session.request(prompt, lambda raw: parse_diagnosis(raw, "original", 3))
        assert diagnosis.unknowns == DIAGNOSIS["unknowns"]
        assert len(sdk.requests) == 2 and session.calls["transport_retry"] == 1
        assert session.ledger.summary()["supervision"]["teacher_calls"]["retry"] == 1
        assert any(r["event"] == "teacher_transport_retry" for r in session.ledger.rows())
    else:
        with pytest.raises(CaseStopped, match="total_call_cap"):
            session.request(prompt, lambda raw: parse_diagnosis(raw, "original", 3))
        assert len(sdk.requests) == 1
    assert "PRIVATE_PROVIDER_EXCEPTION" not in session.ledger.path.read_text()
    assert all(request["max_output_tokens"] == 8192 for request in sdk.requests)


def test_existing_teacher_cache_hit_is_recorded_in_m0a_ledger(tmp_path, task):
    from types import SimpleNamespace
    from test_capped_teacher import SequenceClient
    from teachers.capped import CappedTeacher
    from repair.budgets import GlobalTeacherBudget
    from repair.runtime import RepairTeacher
    from repair.prompts import Prompt
    sdk = SequenceClient([SimpleNamespace(status_code=200, headers={}, parse=lambda: {
        'output_text': json.dumps(label()), 'status': 'completed', 'model': 'fake-model',
        'id': 'fake', '_request_id': 'fake-request', 'usage': {'input_tokens': 13, 'output_tokens': 7,
            'output_tokens_details': {'reasoning_tokens': 3}}})])
    teacher = CappedTeacher(counter_path=tmp_path / "counter.json", cap=54,
        cache_dir=tmp_path / "cache", usage_ledger=tmp_path / "usage.jsonl", run_id="cache-test",
        client=sdk, teacher_class=RepairTeacher, sampling={"max_output_tokens": 1024})
    budget = GlobalTeacherBudget(tmp_path / 'global.jsonl')
    session = CaseSession(task=task, teacher=teacher, ledger=BudgetLedger(tmp_path / "repair-budget.jsonl"),
        config=load_config("configs/base_e4b.yaml"), global_budget=budget)
    prompt = Prompt("rationale", {"task": "public task", "prefix": [], "current_action": "look"})
    session.request(prompt, parse_label)
    session.request(prompt, parse_label)
    assert len(sdk.requests) == 1
    assert any(r["event"] == "teacher_cache_hit" for r in session.ledger.rows())
    assert session.ledger.summary()["supervision"]["teacher_input_tokens"] == 13
    assert session.ledger.summary()['supervision']['teacher_reasoning_tokens'] == 3
    assert session.ledger.summary()['supervision']['teacher_visible_tokens'] == 4
    results = [r['supervision'] for r in session.ledger.rows() if r['event'] == 'teacher_result']
    assert results[-1]['teacher_output_tokens'] == results[-1]['teacher_reasoning_tokens'] == 0
    assert results[-1]['teacher_visible_tokens'] == 0
    assert budget.summary()['reported_usage'] == {
        'input_tokens': 13, 'output_tokens': 7, 'reasoning_tokens': 3, 'visible_tokens': 4}


@pytest.mark.parametrize('failure_cap,expected_calls', [(64000, 2), (24000, 1)])
def test_reasoning_counts_towards_case_cap_but_not_visible_schema(data_root, tmp_path, task, failure_cap, expected_calls):
    original, teacher, session = setup_case(data_root, tmp_path, task, limits=Limits(output_tokens=failure_cap))
    teacher.responses = [
        result(DIAGNOSIS, usage={'input_tokens': 20, 'output_tokens': 8100, 'reasoning_tokens': 8000}),
        result(proposal(original), usage={'input_tokens': 20, 'output_tokens': 16000, 'reasoning_tokens': 15000})]
    outcome = repair_one_failure(original, env_factory=factory(data_root), session=session)
    assert len(teacher.requests) == expected_calls
    assert (outcome.repair.repair_status == 'VERIFIED') == (expected_calls == 2)
    if expected_calls == 1:
        assert outcome.stop_reason == 'output_token_cap'
    totals = session.ledger.summary()['supervision']
    assert totals['teacher_output_tokens'] == (24100 if expected_calls == 2 else 8100)
    assert totals['teacher_reasoning_tokens'] == (23000 if expected_calls == 2 else 8000)
    assert totals['teacher_visible_tokens'] == totals['teacher_output_tokens'] - totals['teacher_reasoning_tokens']
