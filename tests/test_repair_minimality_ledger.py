import json
from types import SimpleNamespace

import pytest

from repair.execute import ResultCache, execute_actions
from repair.ledger import BudgetLedger, summarize
from repair.minimality import MinimalityChecker, joint_necessity_record, repair_cost
from repair.patch import ExtendedRepair, Patch, PatchBlock
from repair.records import PatchRecord, RepairRecord, TaskRecord, TrajectoryRecord
from repair_fakes import RepairBackend, TASK_ID, factory


ORIGINAL = ["look", "look", "finish"]


def joint_patch():
    return Patch("original", [PatchBlock("key", "replace", ("take key",), start=0, end_exclusive=1),
                               PatchBlock("door", "replace", ("open door",), start=1, end_exclusive=2)])


def necessary_patch(block_count):
    original = ["look", "look", *(["fail"] * (block_count - 2)), "finish"]
    patch = Patch("original", [*joint_patch().blocks, *(
        PatchBlock(f"delete_fail_{i}", "delete", start=i, end_exclusive=i + 1)
        for i in range(2, block_count))])
    return original, patch


@pytest.mark.parametrize("block_count", [2, 3, 4])
@pytest.mark.parametrize("complete", [True, False])
def test_joint_necessity_reports_each_final_block_and_missing_evidence(data_root, block_count, complete):
    original, patch = necessary_patch(block_count)
    budget = block_count if complete else block_count - 1
    result = MinimalityChecker(factory(data_root), ablation_replays=budget).run(TASK_ID, original, patch)
    joint = result.joint_necessity
    assert result.repair_status == "VERIFIED" and result.selected_patch == patch
    assert joint["applicable"] is True
    assert joint["block_ids"] == [b.block_id for b in patch.blocks]
    assert joint["all_necessary"] == (True if complete else "unknown")
    assert ("both_necessary" in joint) == (block_count == 2)
    if block_count == 2:
        assert joint["both_necessary"] == joint["all_necessary"]
    assert set(joint["without_block"]) == {b.block_id for b in patch.blocks[:budget]}
    trials = [v for v in result.verification_results if v["phase"] == "ablation"]
    for trial in trials:
        execution = trial["execution"]
        assert joint["without_block"][trial["reverted_block_id"]] == {
            "trajectory_id": execution["trajectory_id"], "success": False,
            "termination_reason": "FAILED", "same_initial_state": True, "cache_hit": False}
    assert joint["scope"] == (
        "These declared blocks, original initial state, fixed suffix and step limit only; "
        "no claim of unique causes or absence of another single-block repair.")


@pytest.mark.parametrize("block_count", [2, 3, 4])
@pytest.mark.parametrize("case,expected", [
    ("success", False), ("success_with_missing", False),
    ("unknown_success", "unknown"), ("environment_error", "unknown"),
    ("different_initial_state", "unknown"), ("stale_patch", "unknown"),
])
def test_joint_necessity_aggregates_only_verifiable_final_patch_evidence(block_count, case, expected):
    _, patch = necessary_patch(block_count)
    full = SimpleNamespace(success=True, trajectory_id="full")
    ablations = [{
        "base_patch_id": patch.patch_id, "reverted_block_id": b.block_id, "same_initial_state": True,
        "execution": {"trajectory_id": f"without_{b.block_id}", "success": False,
                      "termination_reason": "FAILED", "cache_hit": False},
    } for b in patch.blocks]
    evidence = ablations[0]
    if case.startswith("success"):
        evidence["execution"].update(success=True, termination_reason="SUCCESS")
        if case == "success_with_missing":
            ablations.pop()
    elif case == "unknown_success":
        evidence["execution"]["success"] = "unknown"
    elif case == "environment_error":
        evidence["execution"]["termination_reason"] = "ENV_ERROR"
    elif case == "different_initial_state":
        evidence["same_initial_state"] = False
    elif case == "stale_patch":
        evidence["base_patch_id"] = "earlier_patch"
    joint = joint_necessity_record(patch, full, ablations)
    assert joint["all_necessary"] == expected
    assert ("both_necessary" in joint) == (block_count == 2)
    if block_count == 2:
        # Preserve the existing two-block field's requirement for complete evidence.
        assert joint["both_necessary"] == ("unknown" if case == "success_with_missing" else expected)


def test_full_pass_and_joint_necessity_are_scoped_to_declared_blocks(data_root, tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.jsonl")
    checker = MinimalityChecker(factory(data_root), ledger=ledger)
    assert checker.ablation_replays == 12 and checker.total_replays == 24
    result = checker.run(TASK_ID, ORIGINAL, joint_patch())
    assert result.repair_status == "VERIFIED"
    assert result.minimality_status == "BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS"
    assert result.replays_used == 5 and result.ablation_replays_used == 2
    assert result.joint_necessity["both_necessary"] is True
    assert set(result.joint_necessity["without_block"]) == {"key", "door"}
    assert result.cost[:3] == (2, 0, 3)
    assert ledger.summary()["interaction"]["candidate_replay_steps"] == 9
    assert ledger.summary()["interaction"]["edit_ablation_steps"] == 6
    assert result.verification_results[-1]["matches_selected_execution"]


def test_accept_revert_then_restart_from_first_remaining_block(data_root):
    blocks = joint_patch().blocks
    patch = Patch("original", [blocks[0], PatchBlock("extra", "insert_before", ("inspect",), index=1), blocks[1]])
    result = MinimalityChecker(factory(data_root)).run(TASK_ID, ORIGINAL, patch)
    trials = [v for v in result.verification_results if v["phase"] == "ablation"]
    assert [v["reverted_block_id"] for v in trials] == ["key", "extra", "key", "door"]
    assert [v["reverted_block_id"] for v in trials if v["accepted_revert"]] == ["extra"]
    assert result.selected_patch.blocks == blocks
    assert result.joint_necessity["both_necessary"] is True
    assert result.joint_necessity["all_necessary"] is True
    assert set(result.joint_necessity["without_block"]) == {"key", "door"}
    assert result.minimality_status == "BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS"
    assert result.replays_used == 7


@pytest.mark.parametrize("ablation,total,check,status,used", [
    (0, 24, True, "ABLATION_BUDGET_EXHAUSTED", 3),
    (1, 24, True, "ABLATION_BUDGET_EXHAUSTED", 4),
    (12, 3, True, "ABLATION_BUDGET_EXHAUSTED", 3),
    (2, 5, True, "BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS", 5),
    (12, 24, False, "BEST_OBSERVED", 3),
])
def test_status_only_claims_complete_pass(data_root, ablation, total, check, status, used):
    result = MinimalityChecker(factory(data_root), ablation_replays=ablation, total_replays=total).run(
        TASK_ID, ORIGINAL, joint_patch(), check_minimality=check)
    assert result.minimality_status == status and result.replays_used == used
    if status != "BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS":
        assert result.joint_necessity["both_necessary"] == "unknown"


def test_prior_proposals_count_towards_per_failure_cap(data_root):
    result = MinimalityChecker(factory(data_root), total_replays=24).run(
        TASK_ID, ORIGINAL, joint_patch(), replays_used=22)
    assert result.replays_used == 24
    assert result.ablation_replays_used == 0
    assert result.minimality_status == "ABLATION_BUDGET_EXHAUSTED"
    assert result.executed_trajectory.success is True


def test_cache_hit_ablation_does_not_spend_replay_or_step_budget(data_root, tmp_path):
    cache, ledger = ResultCache(), BudgetLedger(tmp_path / "budget.jsonl")
    # These earlier real validation attempts still count towards the total.
    execute_actions(factory(data_root), TASK_ID, ["look", "open door", "finish"], 40, cache=cache, ledger=ledger)
    execute_actions(factory(data_root), TASK_ID, ["take key", "look", "finish"], 40, cache=cache, ledger=ledger)
    result = MinimalityChecker(factory(data_root), cache=cache, ledger=ledger).run(
        TASK_ID, ORIGINAL, joint_patch(), replays_used=2)
    assert result.replays_used == 5 and result.ablation_replays_used == 0 and result.cache_hits == 2
    assert result.minimality_status == "BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS"
    assert result.joint_necessity["both_necessary"] is True
    assert ledger.summary()["interaction"]["edit_ablation_steps"] == 0


def test_environment_error_cannot_establish_necessity_or_irreducibility(data_root):
    class UnknownAblation(RepairBackend):
        def step(self, action):
            if action == "open door" and not self.key:
                raise RuntimeError("PRIVATE")
            return super().step(action)
    result = MinimalityChecker(factory(data_root, backend=UnknownAblation)).run(TASK_ID, ORIGINAL, joint_patch())
    assert result.minimality_status == "BEST_OBSERVED"
    assert result.joint_necessity["both_necessary"] == "unknown"
    assert result.repair_status == "VERIFIED"


def test_changed_reset_cannot_establish_a_repair(data_root):
    class Drift(RepairBackend):
        generation = 0
        def __init__(self, *args):
            super().__init__(*args)
            type(self).generation += 1
        def reset(self):
            state = super().reset()
            state["feedback"] += str(type(self).generation)
            return state
    result = MinimalityChecker(factory(data_root, backend=Drift)).run(TASK_ID, ORIGINAL, joint_patch())
    assert result.repair_status == "NONREPLAYABLE" and result.minimality_status == "BEST_OBSERVED"


@pytest.mark.parametrize("block_count", [2, 3, 4])
def test_final_confirmation_divergence_withdraws_necessity_claim(data_root, block_count):
    class Drift(RepairBackend):
        generation = 0
        def __init__(self, *args):
            super().__init__(*args)
            type(self).generation += 1
        def state(self):
            state = super().state()
            if type(self).generation == block_count + 3 and self.actions:
                state["feedback"] += " drift"
            return state
    original, patch = necessary_patch(block_count)
    result = MinimalityChecker(factory(data_root, backend=Drift)).run(TASK_ID, original, patch)
    assert result.repair_status == "NONREPLAYABLE"
    assert result.minimality_status == "BEST_OBSERVED"
    assert result.joint_necessity["all_necessary"] == "unknown"
    assert ("both_necessary" in result.joint_necessity) == (block_count == 2)
    if block_count == 2:
        assert result.joint_necessity["both_necessary"] == "unknown"
    assert not result.joint_necessity["replay_consistent"]


def test_recorded_original_must_replay_exactly_when_supplied(data_root):
    original = execute_actions(factory(data_root), TASK_ID, ORIGINAL, 40)
    patch = Patch(original.trajectory_id, joint_patch().blocks)
    checker = MinimalityChecker(factory(data_root))
    result = checker.run(TASK_ID, ORIGINAL, patch, original_trajectory=original)
    assert result.repair_status == "VERIFIED"
    original.steps[1]["visible"]["observation"] = "stale trace"
    result = checker.run(TASK_ID, ORIGINAL, patch, original_trajectory=original)
    assert result.repair_status == "NONREPLAYABLE" and result.replays_used == 1


def test_successful_revert_requires_lower_lexicographic_cost(data_root, tmp_path):
    patch = Patch("original", [PatchBlock("keep_look", "insert_before", ("look",), index=0),
                              PatchBlock("fix", "replace", ("take key", "open door"), start=0, end_exclusive=2)])
    # Predeclared ordered_actions([look]) predicate, verified on the public replay.
    def preserved(steps):
        return int(any(s["action"] == "look" and s["outcome"] == "legal" for s in steps))
    ledger = BudgetLedger(tmp_path / "budget.jsonl")
    result = MinimalityChecker(factory(data_root), ledger=ledger).run(
        TASK_ID, ORIGINAL, patch, preserved_subgoals=preserved)
    assert result.selected_patch == patch and result.cost[:3] == (2, -1, 4)
    assert result.minimality_status == "BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS"
    assert result.joint_necessity["both_necessary"] is False
    assert any(r["reason"] == "successful_revert_does_not_lower_cost" for r in ledger.rows())


def test_rejected_candidates_and_extended_variants_are_logged_without_execution(data_root, tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.jsonl")
    def forbidden_factory():
        pytest.fail("Invalid candidate must not reach reset")
    checker = MinimalityChecker(forbidden_factory, ledger=ledger)
    invalid = Patch("original", [PatchBlock("bad", "delete", start=99, end_exclusive=100)])
    result = checker.run(TASK_ID, ORIGINAL, invalid)
    assert result.repair_status == "INVALID_PATCH" and result.replays_used == 0
    assert ledger.rows()[0]["event"] == "rejected_candidate"
    assert ledger.rows()[0]["reason"] == "range_out_of_bounds_or_empty"
    extended = Patch("original", [PatchBlock(str(i), "insert_before", ("look",), index=i) for i in range(5)])
    with pytest.raises(ExtendedRepair):
        checker.run(TASK_ID, ORIGINAL, extended)
    assert ledger.rows()[-1]["details"]["variant"] == "extended_repair"
    assert sum(ledger.summary()["interaction"].values()) == 0


def test_failed_candidate_and_zero_budget_preserve_negative_results(data_root):
    result = MinimalityChecker(factory(data_root)).run(TASK_ID, ORIGINAL, Patch("original", [joint_patch().blocks[0]]))
    assert result.repair_status == "NO_SUCCESS_WITHIN_BUDGET"
    assert result.executed_trajectory.success is False
    result = MinimalityChecker(factory(data_root), total_replays=0).run(TASK_ID, ORIGINAL, joint_patch())
    assert result.replays_used == 0 and result.executed_trajectory is None


def test_cost_counts_tail_after_early_success_and_preserved_predicates(data_root):
    result = execute_actions(factory(data_root), TASK_ID, ["take key", "open door", "finish", "inspect"], 40)
    original = [*ORIGINAL, "inspect"]
    assert repair_cost(original, result, joint_patch())[:3] == (3, 0, 3)
    def preserved(steps):
        assert all(set(s) == {"action", "outcome", "visible"} for s in steps)
        assert "PRIVATE" not in json.dumps(steps)
        return 1
    assert repair_cost(original, result, joint_patch(), preserved)[:3] == (3, -1, 3)


def test_budget_summary_only_settled_three_families_and_unknowns(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.jsonl")
    ledger.append("attempt", settled=False, attempt_id="a", supervision={"teacher_input_tokens": 999})
    prefix = ledger.path.read_bytes()
    ledger.append("failure", attempt_id="a", supervision={"teacher_input_tokens": 10, "teacher_billed_cost": None,
                                                         "teacher_calls": {"proposal": 1}},
                  interaction={"candidate_replay_steps": 3})
    assert ledger.path.read_bytes().startswith(prefix)
    ledger.append("retry", parent_attempt_id="a", supervision={"teacher_calls": {"retry": 1}, "teacher_output_tokens": 2})
    ledger.append("rejected_candidate", reason="overlapping_ranges")
    ledger.append("optimization", optimization={"optimizer_updates": 2, "supervised_target_tokens": 50})
    result = ledger.summary()
    assert result["supervision"]["teacher_input_tokens"] == 10
    assert result["supervision"]["teacher_calls"]["retry"] == 1
    assert result["interaction"]["candidate_replay_steps"] == 3
    assert result["optimization"]["optimizer_updates"] == 2
    assert result["settled_rows"] == 4
    assert result["unknown_quantities"] == {"supervision.teacher_billed_cost": 1}
    with pytest.raises(ValueError):
        ledger.append("duplicate", attempt_id="a")
    with pytest.raises(ValueError):
        ledger.append("bad", interaction={"candidate_replay_steps": -1})
    with pytest.raises(ValueError):
        summarize([ledger.rows()[1], ledger.rows()[1]])


def test_record_roundtrip_stable_ids_and_visible_whitelist(data_root):
    task = TaskRecord(TASK_ID, "fingerprint", "train", "sha256:state", "public goal", 0)
    patch = PatchRecord("original", joint_patch().blocks, "scripted_fixture", 0)
    repair = MinimalityChecker(factory(data_root)).run(TASK_ID, ORIGINAL, joint_patch())
    for record in (task, patch, repair, repair.executed_trajectory):
        value = record.to_json()
        restored = type(record).from_json(json.dumps(value))
        assert restored.to_json() == value and restored.stable_id == record.stable_id
        value[record.id_field] = "tampered"
        with pytest.raises(ValueError):
            type(record).from_json(value)
    assert patch.patch_id == joint_patch().patch_id
    trace = repair.executed_trajectory.to_json()
    trace.pop("trajectory_id")
    trace["steps"][0]["visible"]["hidden_success"] = "PRIVATE"
    trace["steps"][0]["policy_commands"] = "PRIVATE"
    restored = TrajectoryRecord.from_json(trace)
    assert "PRIVATE" not in json.dumps(restored.to_json())
