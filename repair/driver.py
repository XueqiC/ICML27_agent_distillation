"""One bounded repair search. All success and ablation evidence is real execution."""

from dataclasses import dataclass, replace

from sed.storage import digest
from repair.align import canonical_action
from repair.execute import ResultCache, execute_actions
from repair.minimality import MinimalityChecker, repair_cost, same_execution, same_reset
from repair.patch import InvalidPatch, Patch, compile_patch
from repair.prompts import diagnosis_prompt, proposal_prompt
from repair.records import RepairRecord, TrajectoryRecord
from repair.runtime import CaseSession, CaseStopped
from repair.schemas import parse_candidates, parse_diagnosis


@dataclass
class RepairOutcome:
    repair: RepairRecord
    diagnosis: object
    e1: list[dict]
    candidates: list[dict]
    handbacks: list[dict]
    costs: dict
    stop_reason: str | None = None


def evidence_table(record):
    """Unknown stays unknown; a 2-block single is the other block's ablation."""
    blocks = record.selected_patch.blocks
    requested = [("original", ()), *[("single_block", (b.block_id,)) for b in blocks],
                 ("all_blocks", tuple(b.block_id for b in blocks)),
                 ("revert_and_restore", tuple(b.block_id for b in blocks))]
    table = []
    for condition, ids in requested:
        matches = []
        for item in record.verification_results:
            actual = tuple(b["block_id"] for b in item["patch"]["blocks"])
            if set(actual) != set(ids):
                continue
            if condition == "original" and item["phase"] != "original":
                continue
            if condition == "revert_and_restore" and item["phase"] != "confirmation":
                continue
            if condition != "revert_and_restore" and item["phase"] == "confirmation":
                continue
            matches.append(item)
        measured = matches[-1] if matches else None
        execution = measured["execution"] if measured else None
        table.append({"condition": condition, "block_ids": list(ids),
            "success": execution["success"] if execution else "unknown",
            "termination": execution["termination_reason"] if execution else "not_measured",
            "trajectory_id": execution["trajectory_id"] if execution else None,
            "same_initial_state": measured["same_initial_state"] if measured else "unknown",
            "cache_hit": execution["cache_hit"] if execution else False,
            "replay_consistent": record.repair_status != "NONREPLAYABLE"})
    return table


def search_candidates(original, diagnosis, env_factory, session, *, cache, near=None, rounds=2):
    """Yield all observed successes; preserve every rejection and real feedback."""
    if rounds not in (1, 2):
        raise ValueError("Search is bounded by two rounds")
    actions = original.executed_actions
    successes, evidence, executions, rejected, seen = [], [], [], [], set()
    for round_id in range(1, rounds + 1):
        prompt = proposal_prompt(session.task, original, diagnosis, session.action_syntax,
            session.max_steps, executions=executions, rejections=rejected, near=near)
        patches = session.request(prompt, lambda raw: parse_candidates(raw, original.trajectory_id))
        round_success = False
        for patch in patches:
            row = {"round": round_id, "patch": patch.to_json(), "status": "rejected"}
            evidence.append(row)
            try:
                compiled = compile_patch(actions, patch)
            except InvalidPatch as exc:
                row["reason"] = exc.reason
                rejected.append((patch.patch_id, exc.reason))
                session.log("rejected_candidate", candidate_id=patch.patch_id, reason=exc.reason,
                            details={"patch": patch.to_json(), "round": round_id})
                continue
            canonical_hash = digest([canonical_action(a) for a in compiled])
            row["canonical_action_sha256"] = canonical_hash
            if canonical_hash in seen:
                row["reason"] = "duplicate_compiled_actions"
                session.log("rejected_candidate", candidate_id=patch.patch_id,
                            reason=row["reason"], details=row)
                continue
            seen.add(canonical_hash)
            if not session.replay_allowed(reserve_confirmation=near is None):
                row["reason"] = "replay_cap"
                session.log("rejected_candidate", candidate_id=patch.patch_id, reason="replay_cap")
                continue
            result = execute_actions(env_factory, original.task_id, compiled, session.max_steps,
                cache=cache, ledger=session.ledger, candidate_id=patch.patch_id)
            session.shared["replays"] += int(not result.cache_hit)
            session.shared["cache_hits"] += int(result.cache_hit)
            row.update(execution=result.to_json(), status="executed")
            executions.append(result)
            session.log("candidate_evidence", candidate_id=patch.patch_id, details=row)
            if not same_reset(original, result):
                session.stop("NONREPLAYABLE")
            if result.termination_reason == "ENV_ERROR":
                session.stop("ENV_ERROR")
            if result.success is True:
                successes.append((patch, result))
                round_success = True
            else:
                row["reason"] = "candidate_not_verified_successful"
                session.log("rejected_candidate", candidate_id=patch.patch_id, reason=row["reason"])
        if round_success:
            break
    return successes, evidence


def _handback_points(patch, executed_length):
    """Compiled boundaries, including the boundary exposed by a deletion."""
    shift, points = 0, []
    blocks = sorted(patch.blocks, key=lambda b: (b.index if b.op == "insert_before" else b.start,
                                               b.op != "insert_before"))
    for block in blocks:
        start = block.index if block.op == "insert_before" else block.start
        end = start + shift + len(block.actions)
        shift += len(block.actions) - (0 if block.op == "insert_before" else block.end_exclusive - block.start)
        if 0 <= end < executed_length:
            points.append(end)
    return list(dict.fromkeys(points))[:3]


def repair_one_failure(failure_trajectory, *, env_factory, task=None, teacher=None, ledger=None,
                       config=None, session=None, cache=None, student_policy=None):
    """Return RepairOutcome (RepairRecord + E1 evidence), with no training updates.

    ``task`` must come from the caller's frozen training manifest. Student policy,
    if supplied, is a callable (public_history, visible_context) -> reply.
    Use the same session for subsequent label reconstruction (shared retry/caps).
    """
    session = session or CaseSession(task=task, teacher=teacher, ledger=ledger, config=config)
    original = failure_trajectory
    if (original.task_id != session.task["task_id"] or original.max_steps != session.max_steps
            or original.env_seed != session.config["environment"]["seed"]):
        raise ValueError("Failure differs from authorized task/horizon/seed")
    empty = Patch(original.trajectory_id)
    record = RepairRecord(original.trajectory_id, empty, None, None, None, [], "BEST_OBSERVED",
                          "NO_SUCCESS_WITHIN_BUDGET", {}, original)
    diagnosis, candidates, handbacks, stop = None, [], [], None
    cache = cache or ResultCache()
    try:
        if not session.replay_allowed():
            raise CaseStopped("replay_cap")
        replay = execute_actions(env_factory, original.task_id,
            [s["action_raw"] for s in original.steps], session.max_steps, ledger=session.ledger)
        session.shared["replays"] += 1
        validations = [{"phase": "original", "patch": empty.to_json(), "execution": replay.to_json(),
                        "same_initial_state": same_reset(original, replay)}]
        record = replace(record, verification_results=validations)
        if original.success is not False or not same_execution(original, replay):
            session.stop("NONREPLAYABLE")
        if any(s["action_parsed"] is None for s in original.steps):
            # Parser feedback is not an environment action. Do not silently
            # renumber patches or discard those turns to manufacture a repair.
            session.stop("original_contains_parser_feedback")
        diagnosis = session.request(diagnosis_prompt(session.task, original,
            session.action_syntax, session.max_steps), lambda raw: parse_diagnosis(
                raw, original.trajectory_id, len(original.executed_actions)))
        successes, candidates = search_candidates(original, diagnosis, env_factory, session, cache=cache)
        if successes:
            # No unverifiable natural-language diagnosis is counted as p.
            best_patch, best = min(successes, key=lambda pair: repair_cost(original.executed_actions, pair[1], pair[0]))
            # For 3/4-block E1 cases measure each single as well as each revert.
            # Singles are additional observed candidates and can improve N.
            singles = []
            for block in best_patch.blocks if len(best_patch.blocks) > 2 else ():
                if (session.shared["ablations"] >= session.limits.ablations or
                        not session.replay_allowed(reserve_confirmation=True)):
                    break
                single = Patch(original.trajectory_id, (block,))
                result = execute_actions(env_factory, original.task_id,
                    compile_patch(original.executed_actions, single), session.max_steps,
                    cache=cache, ledger=session.ledger, candidate_id=single.patch_id,
                    interaction_kind="edit_ablation_steps")
                session.shared["replays"] += int(not result.cache_hit)
                session.shared["cache_hits"] += int(result.cache_hit)
                session.shared["ablations"] += int(not result.cache_hit)
                singles.append({"phase": "single_block", "patch": single.to_json(),
                                "execution": result.to_json(), "same_initial_state": same_reset(original, result)})
                session.log("single_block_evidence", candidate_id=single.patch_id, details=singles[-1])
                if not same_reset(original, result):
                    session.stop("NONREPLAYABLE")
                if result.success is True:
                    successes.append((single, result))
            best_patch, best = min(successes, key=lambda pair: repair_cost(original.executed_actions, pair[1], pair[0]))
            record = MinimalityChecker(env_factory, max_steps=session.max_steps,
                ablation_replays=session.limits.ablations, total_replays=session.limits.replays,
                cache=cache, ledger=session.ledger).run(original.task_id, original.executed_actions,
                    best_patch, replays_used=session.shared["replays"],
                    ablation_replays_used=session.shared["ablations"], original_trajectory=original,
                    verified_executions=(replay, best))
            record = replace(record, verification_results=record.verification_results + singles)
            session.shared.update(replays=record.replays_used, ablations=record.ablation_replays_used)
            session.shared["cache_hits"] += record.cache_hits
            if record.repair_status == "VERIFIED" and student_policy is not None:
                target = record.executed_trajectory
                for end in _handback_points(record.selected_patch, len(target.steps)):
                    if session.shared["handbacks"] >= session.limits.handbacks or not session.replay_allowed():
                        break
                    result = execute_actions(env_factory, original.task_id, target.executed_actions[:end],
                        session.max_steps, ledger=session.ledger, continuation=student_policy,
                        prefix_reference=target, interaction_kind="handback_diagnostic_steps")
                    session.shared["replays"] += 1
                    session.shared["handbacks"] += 1
                    row = {"prefix_length": end, "handback_success": result.success,
                           "completion_steps": max(0, len(result.steps) - end),
                           "failure_type": result.termination_reason, "execution": result.to_json()}
                    handbacks.append(row)
                    session.log("handback_diagnostic", details=row)
    except CaseStopped as exc:
        stop = str(exc)
        status = ("NONREPLAYABLE" if stop in {"NONREPLAYABLE", "ENV_ERROR"} else
                  "INVALID_PATCH" if stop in {"format_retry_exhausted", "original_contains_parser_feedback"}
                  else record.repair_status)
        record = replace(record, repair_status=status)
    record = replace(record, replays_used=session.shared["replays"],
                     ablation_replays_used=session.shared["ablations"], cache_hits=session.shared["cache_hits"])
    table = evidence_table(record)
    session.log("repair_result", details={"repair": record.to_json(), "e1": table, "stop_reason": stop})
    return RepairOutcome(record, diagnosis, table, candidates, handbacks, session.cost_record(), stop)
