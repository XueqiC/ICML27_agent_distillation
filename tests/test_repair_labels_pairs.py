from copy import deepcopy
from dataclasses import replace
import json

import pytest

from repair.driver import repair_one_failure, RepairOutcome
from repair.execute import execute_actions
from repair.labels import LearningEpisode, reconstruct_labels
from repair.ledger import BudgetLedger
from repair.pairs import (PairOutcome, choose_far, construct_pair, coverage_signature,
                          first_pairs, pair_exclusions, review_target)
from repair.patch import Patch, PatchBlock
from repair.records import RepairRecord
from repair.runtime import CaseSession, Limits
from repair.align import align_actions
from repair_fakes import TASK_ID, factory
from repair_teacher_fakes import DIAGNOSIS, FakeTeacher, label, proposal
from sed.config import load_config
from test_r03_sets import TinyTokenizer


def session_for(tmp_path, task, teacher, **kwargs):
    return CaseSession(task=task, teacher=teacher, ledger=BudgetLedger(tmp_path / "budget.jsonl"),
                       config=load_config("configs/base_e4b.yaml"), **kwargs)


def test_labels_rebuild_only_from_current_prefix_preserve_original_bytes(data_root, tmp_path, task):
    actions = ["Thought:  Keep EXACT bytes.  \nAction: take key", "look", "finish"]
    original = execute_actions(factory(data_root), TASK_ID, actions, 40)
    p = proposal(original, blocks=[{"block_id": "door", "op": "replace", "start": 1,
                                    "end_exclusive": 2, "actions": ["open door"]}])
    teacher = FakeTeacher([DIAGNOSIS, p, label("Check the door."), label("Use the open door.", [1])])
    session = session_for(tmp_path, task, teacher)
    repair = repair_one_failure(original, env_factory=factory(data_root), session=session).repair
    output = reconstruct_labels(repair, session=session, review_path=tmp_path / "review.tsv")
    assert output.status == "COMPLETE"
    episode = output.episode
    assert episode.history[0]["reply"].encode() == actions[0].encode()
    assert episode.supervised_decision_ids == [1]
    assert episode.full_trajectory_decision_ids == [0, 1, 2]
    prompts = [p for p in teacher.requests if p.kind == "rationale"]
    assert len(prompts) == 2
    for i, prompt in enumerate(prompts, 1):
        assert len(prompt.payload["prefix"]) == i
        content = json.dumps(prompt.payload)
        assert "diagnosis-only-marker" not in content
        assert "success" not in prompt.payload and "trace" not in prompt.payload
        assert "future" not in prompt.payload
        assert "actions=take key,open door,finish" not in content
    assert "Check the door." in prompts[1].payload["prefix"][1]["reply"]
    assert "door=True" not in json.dumps(prompts[0].payload)
    assert "door=True" in json.dumps(prompts[1].payload)
    assert "human_supported" in output.review_tsv and output.review_tsv == (tmp_path / "review.tsv").read_text()
    assert LearningEpisode.from_json(episode.to_json()).to_json() == episode.to_json()
    assert all(set(m["assistant"]) == {1} and set(m["observation"]) == {0} for m in episode.token_mask)


@pytest.mark.parametrize("bad_label", [label("A future fact.", [1]), label("Use apple 99."),
                                       label("Unknown.", unsupported=True)])
def test_bad_prefix_label_never_becomes_partial_episode(data_root, tmp_path, task, bad_label):
    original = execute_actions(factory(data_root), TASK_ID, ["look", "look", "finish"], 40)
    teacher = FakeTeacher([DIAGNOSIS, proposal(original), bad_label])
    session = session_for(tmp_path, task, teacher)
    repair = repair_one_failure(original, env_factory=factory(data_root), session=session).repair
    out = reconstruct_labels(repair, session=session)
    assert out.status == "UNGROUNDED_TARGET" and out.episode is None
    assert len(teacher.requests) == 3 and len(out.checks) == 1


def test_label_budget_is_atomic_and_shared_format_retry(data_root, tmp_path, task):
    original = execute_actions(factory(data_root), TASK_ID, ["look", "look", "finish"], 40)
    teacher = FakeTeacher(["bad", DIAGNOSIS, proposal(original), "bad", label()])
    session = session_for(tmp_path, task, teacher)
    repair = repair_one_failure(original, env_factory=factory(data_root), session=session).repair
    out = reconstruct_labels(repair, session=session)
    assert out.episode is None and out.status == "LABEL_SCHEMA_FAILED"
    assert len(teacher.requests) == 4 and session.shared["format_retries"] == 1


def test_label_cap_stops_before_next_call_and_action_only_needs_none(data_root, tmp_path, task):
    original = execute_actions(factory(data_root), TASK_ID, ["look", "look", "finish"], 40)
    teacher = FakeTeacher([DIAGNOSIS, proposal(original), label(), label()])
    session = session_for(tmp_path, task, teacher, limits=Limits(rationale_calls=1))
    repair = repair_one_failure(original, env_factory=factory(data_root), session=session).repair
    out = reconstruct_labels(repair, session=session)
    assert out.status == "LABEL_BUDGET_EXHAUSTED" and out.episode is None
    assert len(teacher.requests) == 3 and len(out.checks) == 1
    fresh = session_for(tmp_path / "action-only", task, FakeTeacher([]), limits=Limits(rationale_calls=0))
    assert reconstruct_labels(repair, session=fresh, reasoning=False).episode is not None
    assert fresh.teacher.requests == []


O = ["go to cabinet 1", "open cabinet 1", "take apple 1 from cabinet 1", "look",
     "go to table 1", "put apple 1 in/on table 1"]
N = [*O[:3], "clean apple 1 with sinkbasin 1", *O[4:]]
F = [O[1], O[0], O[2], O[4], N[3], O[5]]


class PublicTaskEnv:
    seed, max_episode_steps, game_hash = 0, 40, "fake-clean-apple-game"
    def __init__(self):
        self.held, self.clean, self.won, self.actions = False, False, False, []
    def state(self, observation):
        return ({"observation": observation, "admissible_commands": sorted(set(O + N + F))},
                0, self.won, False, {"won": self.won, "hidden": "PRIVATE_STATE"})
    def reset(self, task_id):
        return self.state("There are cabinet 1, table 1, sinkbasin 1, apple 1 and apple 2.")
    def step(self, action):
        self.actions.append(action)
        if action.startswith("take apple"):
            self.held = True
        if action.startswith("clean apple"):
            self.clean = True
        if action.startswith("put apple"):
            self.won = self.held and self.clean
        return self.state("You executed " + action + ".")
    def close(self):
        pass


def witnessed_repair(original, actions, block_id="route"):
    target = execute_actions(PublicTaskEnv, TASK_ID, actions, 40)
    patch = Patch(original.trajectory_id, [PatchBlock(block_id, "replace", actions,
                                                    start=0, end_exclusive=len(O))])
    alignment = align_actions(original.executed_actions, target.executed_actions)
    return RepairRecord(original.trajectory_id, patch, target, alignment,
        (alignment.distance, 0, len(actions), patch.patch_id), [], "BEST_OBSERVED", "VERIFIED", {}, original)


def approve(public):
    assert "PRIVATE_STATE" not in json.dumps(public)
    return {"reviewer": "fake-human", "no_rejected_actions": True, "meaningful_route": True,
            "labels_supported": True, "notes": "Checked every public transition and short label."}


def pair_fixture(tmp_path, task):
    task = {**task, "instruction": "Put a clean apple on table 1."}
    original = execute_actions(PublicTaskEnv, TASK_ID,
        ["Thought: Visible choice.\nAction: " + a for a in O], 40)
    near, far = witnessed_repair(original, N, "near"), witnessed_repair(original, F, "far")
    session = session_for(tmp_path, task, FakeTeacher([label("Visible choice.")] * 20))
    nl = reconstruct_labels(near, session=session, tokenizer=TinyTokenizer(), tokenizer_id="fake-student")
    fl = reconstruct_labels(far, session=session, tokenizer=TinyTokenizer(), tokenizer_id="fake-student")
    assert nl.episode and fl.episode
    nr = review_target(near.executed_trajectory, nl.episode, approve)
    fr = review_target(far.executed_trajectory, fl.episode, approve)
    return original, near, far, nl, fl, nr, fr, session


def test_pair_filter_accepts_real_success_same_coverage_and_comparable_labels(tmp_path, task):
    original, near, far, nl, fl, nr, fr, _ = pair_fixture(tmp_path, task)
    assert coverage_signature(near.executed_trajectory) == coverage_signature(far.executed_trajectory)
    assert not pair_exclusions(original, near.executed_trajectory, far.executed_trajectory,
                               nl.episode, fl.episode, nr, fr)
    selected, reasons = choose_far(original, near.executed_trajectory, nl.episode, nr, [(far, fl.episode, fr)])
    assert selected[0] == far and reasons[0]["reasons"] == []
    for episode in (nl.episode, fl.episode):
        assert sum(episode.native_token_mask) == episode.supervised_target_tokens
        assert len(episode.native_token_mask) == len(episode.native_input_ids)
        supervised = [t for t, mask in zip(episode.native_input_ids, episode.native_token_mask) if mask]
        decoded = TinyTokenizer().decode(supervised, skip_special_tokens=False)
        assert decoded.count("<turn|>") == len(episode.history)
        assert "You executed" not in decoded
        assert "PRIVATE_STATE" not in TinyTokenizer().decode(episode.native_input_ids, skip_special_tokens=False)


@pytest.mark.parametrize("change,expected", [
    ("horizon", "task_initial_state_success_standard_or_horizon"),
    ("reset", "task_initial_state_success_standard_or_horizon"),
    ("tokens", "supervised_token_ratio"),
    ("protocol", "label_construction_protocol"),
    ("padding", "far:meaningless_route_or_padding"),
    ("unsupported", "far:unsupported_reasoning"),
    ("review_missing", "far:public_quality_review_missing"),
    ("rejected", "far:environment_rejected_action"),
    ("distance", "action_distance_gap"),
    ("length", "action_length_ratio"),
    ("object", "target_object_instances"),
    ("operations", "required_operation_types"),
    ("success", "far:no_real_success"),
])
def test_pair_filter_exclusions(tmp_path, task, change, expected):
    original, near, far, nl, fl, nr, fr, _ = pair_fixture(tmp_path, task)
    target, episode = far.executed_trajectory, fl.episode
    if change == "horizon":
        target = replace(target, max_steps=39)
    elif change == "reset":
        target = replace(target, env_seed=1)
    elif change == "tokens":
        episode = replace(episode, supervised_target_tokens=999)
    elif change == "protocol":
        episode = replace(episode, label_protocol={"different": True})
    elif change == "padding":
        fr = replace(fr, meaningful_route=False)
    elif change == "unsupported":
        fr = replace(fr, labels_supported=False)
    elif change == "review_missing":
        fr = None
    elif change == "rejected":
        fr = replace(fr, no_rejected_actions=False)
    elif change == "distance":
        target = near.executed_trajectory
    elif change == "length":
        target = replace(target, steps=target.steps[:2] + target.steps)
    elif change == "object":
        target = execute_actions(PublicTaskEnv, TASK_ID, [a.replace("apple 1", "apple 2") for a in F], 40)
    elif change == "operations":
        steps = deepcopy(target.steps)
        steps[4]["action_parsed"] = "look"
        target = replace(target, steps=steps)
    elif change == "success":
        target = replace(target, success=False, termination_reason="FAILED")
    reasons = pair_exclusions(original, near.executed_trajectory, target, nl.episode, episode, nr, fr)
    assert expected in reasons


def test_far_tie_choice_and_first_four_follow_frozen_order(tmp_path, task):
    original, near, far, nl, fl, nr, fr, _ = pair_fixture(tmp_path, task)
    other = replace(far, selected_patch=Patch(original.trajectory_id, [
        replace(far.selected_patch.blocks[0], block_id="other")]))
    # Same execution/labels; final tie is patch ID, independent of response order.
    a = choose_far(original, near.executed_trajectory, nl.episode, nr,
                   [(far, fl.episode, fr), (other, fl.episode, fr)])[0][0]
    b = choose_far(original, near.executed_trajectory, nl.episode, nr,
                   [(other, fl.episode, fr), (far, fl.episode, fr)])[0][0]
    assert a == b and a.selected_patch.patch_id == min(far.selected_patch.patch_id, other.selected_patch.patch_id)
    outcomes = {str(i): PairOutcome("PAIRED", "near", far, fl.episode, [], [], {}) for i in range(6)}
    order = ["5", "3", "1", "2", "0", "4"]
    assert [t for t, _ in first_pairs(order, outcomes)] == order[:4]


def test_construct_pair_one_extra_request_keeps_near_fixed(tmp_path, task):
    original, near, far, nl, _, _, _, session = pair_fixture(tmp_path, task)
    proposed = {"candidates": [{"original_trajectory_id": original.trajectory_id, "blocks": [
        {"block_id": "different", "op": "replace", "start": 0, "end_exclusive": len(O), "actions": F}]}]}
    session.teacher = FakeTeacher([proposed, *[label("Visible choice.")] * 6])
    from repair.schemas import parse_diagnosis
    diagnosis = parse_diagnosis(json.dumps(DIAGNOSIS), original.trajectory_id, len(O))
    near_outcome = RepairOutcome(near, diagnosis, [], [], [], {})
    near_id = near.repair_id
    out = construct_pair(near_outcome, nl, session=session, env_factory=PublicTaskEnv, reviewer=approve,
                         tokenizer=TinyTokenizer(), tokenizer_id="fake-student")
    assert out.status == "PAIRED" and out.near_repair_id == near_id
    assert near_outcome.repair.repair_id == near_id
    assert [r.kind for r in session.teacher.requests].count("far") == 1
    assert out.costs["branch"] == "far" and out.costs["calls"]["proposal"] == 1


def test_bad_near_exits_without_far_call_or_near_substitution(tmp_path, task):
    _, near, _, nl, _, _, _, session = pair_fixture(tmp_path, task)
    session.teacher = FakeTeacher([])
    def bad_near(public):
        return {**approve(public), "no_rejected_actions": False}
    out = construct_pair(RepairOutcome(near, None, [], [], [], {}), nl, session=session,
                         env_factory=PublicTaskEnv, reviewer=bad_near)
    assert out.status == "PAIR_NOT_FOUND" and session.teacher.requests == []
    assert "near:environment_rejected_action" in out.exclusions[0]["reasons"]


def test_whitespace_counts_cannot_pass_strict_token_ratio(tmp_path, task):
    original, near, far, nl, fl, nr, fr, _ = pair_fixture(tmp_path, task)
    approximate = replace(fl.episode, label_protocol={**fl.episode.label_protocol,
        "token_scope": "assistant_text_before_chat_template", "tokenizer": "whitespace_fallback"})
    review = review_target(far.executed_trajectory, approximate, approve)
    reasons = pair_exclusions(original, near.executed_trajectory, far.executed_trajectory,
        nl.episode, approximate, nr, review)
    assert "far:supervised_token_count_requires_student_tokenizer" in reasons


def test_far_cannot_reset_shared_total_call_budget(tmp_path, task):
    _, near, _, nl, _, _, _, session = pair_fixture(tmp_path, task)
    session.limits = Limits(total_calls=session.shared["teacher_calls"])
    session.teacher = FakeTeacher([])
    from repair.schemas import parse_diagnosis
    diagnosis = parse_diagnosis(json.dumps(DIAGNOSIS), near.original_trajectory_id, len(O))
    out = construct_pair(RepairOutcome(near, diagnosis, [], [], [], {}), nl,
        session=session, env_factory=PublicTaskEnv, reviewer=approve,
        tokenizer=TinyTokenizer(), tokenizer_id="fake-student")
    assert out.status == "PAIR_NOT_FOUND" and session.teacher.requests == []
    assert out.exclusions[0]["reasons"] == ["total_call_cap"]
