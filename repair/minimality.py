"""Bounded whole-block reverts; no claim of global or within-block minimality."""

from repair.align import align_actions
from repair.execute import execute_actions
from repair.patch import ExtendedRepair, InvalidPatch, Patch, compile_patch
from repair.records import RepairRecord


def repair_cost(original_actions, result, patch, preserved_subgoals=None):
    """Section 7.1 cost uses the executed prefix, never an unexecuted suffix.

    An optional counter must implement predicates registered before proposals;
    it sees only parsed actions and public observations. Unverified p defaults 0.
    """
    public_steps = [{"action": s["action_parsed"], "outcome": s["action_outcome"],
                     "visible": s["visible"]} for s in result.steps]
    p = preserved_subgoals(public_steps) if preserved_subgoals else 0
    if type(p) is not int or p < 0:
        raise ValueError("Preserved subgoal count must be a nonnegative integer")
    actions = result.executed_actions
    return (align_actions(original_actions, actions).distance, -p, len(actions), patch.patch_id)


def same_reset(a, b):
    return (a.initial is not None and b.initial is not None and
            all(getattr(a, k) == getattr(b, k) for k in
                ("environment_version", "game_sha256", "fingerprint", "env_seed")) and
            all(a.initial[k] == b.initial[k] for k in
                ("visible", "reward", "terminated", "truncated", "audit_sha256")))


def same_execution(a, b):
    keys = ("action_parsed", "action_outcome", "visible", "reward", "terminated", "truncated", "audit_sha256")
    return (same_reset(a, b) and a.success == b.success and a.termination_reason == b.termination_reason
            and len(a.steps) == len(b.steps) and all(
                all(left.get(k) == right.get(k) for k in keys) for left, right in zip(a.steps, b.steps)))


def joint_necessity_record(patch, full_result, ablations):
    """Only the 2..4 declared blocks and unchanged compiled suffix are tested."""
    n = len(patch.blocks)
    applicable = 2 <= n <= 4 and full_result is not None and full_result.success is True
    without = {}
    for block in patch.blocks:
        evidence = next((v for v in reversed(ablations) if
                         v.get("base_patch_id") == patch.patch_id and
                         v.get("reverted_block_id") == block.block_id), None)
        if evidence is not None:
            result = evidence["execution"]
            without[block.block_id] = {
                "trajectory_id": result["trajectory_id"], "success": result["success"],
                "termination_reason": result["termination_reason"],
                "same_initial_state": evidence["same_initial_state"],
                "cache_hit": result["cache_hit"]}
    verified = [v for v in without.values() if
                v["same_initial_state"] and type(v["success"]) is bool and
                v["termination_reason"] in {"SUCCESS", "FAILED", "STEP_LIMIT"}]
    known = applicable and len(verified) == n
    necessary = all(v["success"] is False for v in verified) if known else "unknown"
    all_necessary = (False if applicable and any(v["success"] is True for v in verified)
                     else necessary)
    record = {"applicable": applicable, "block_ids": [b.block_id for b in patch.blocks],
            "full_trajectory_id": full_result.trajectory_id if full_result else None,
            "full_success": full_result.success if full_result else "unknown",
            "without_block": without, "all_necessary": all_necessary,
            "scope": "These declared blocks, original initial state, fixed suffix and step limit only; "
                     "no claim of unique causes or absence of another single-block repair."}
    if n == 2:
        record["both_necessary"] = necessary
    return record


class MinimalityChecker:
    def __init__(self, env_factory, *, max_steps=40, ablation_replays=12, total_replays=24,
                 cache=None, ledger=None, version=None):
        for value in (max_steps, ablation_replays, total_replays):
            if type(value) is not int or value < 0:
                raise ValueError("Replay budgets and max_steps must be nonnegative integers")
        self.env_factory, self.max_steps = env_factory, max_steps
        self.ablation_replays, self.total_replays = ablation_replays, total_replays
        self.cache, self.ledger, self.version = cache, ledger, version

    def run(self, task_id, original_actions, patch, *, replays_used=0,
            ablation_replays_used=0, preserved_subgoals=None, check_minimality=True,
            original_trajectory=None):
        """Counters include earlier proposals/retries for this same failure.

        This call checks the original, verifies the candidate, ablates, and
        repeats the selected success without cache. One replay is reserved for
        that confirmation. Cache hits consume neither replay nor step budgets.
        """
        if (type(replays_used) is not int or not 0 <= replays_used <= self.total_replays or
                type(ablation_replays_used) is not int or
                not 0 <= ablation_replays_used <= min(replays_used, self.ablation_replays)):
            raise ValueError("Invalid previously consumed replay budgets")
        original_actions = list(original_actions)
        validations, hits = [], 0
        original, selected, cost = None, None, None
        status, repair_status = "BEST_OBSERVED", "NO_SUCCESS_WITHIN_BUDGET"

        def finish():
            joint = joint_necessity_record(patch, selected, validations)
            joint["replay_consistent"] = repair_status != "NONREPLAYABLE"
            if repair_status == "NONREPLAYABLE":
                joint["all_necessary"] = "unknown"
                if "both_necessary" in joint:
                    joint["both_necessary"] = "unknown"
            return RepairRecord(patch.original_trajectory_id, patch, selected,
                align_actions(original_actions, selected.executed_actions) if selected else None,
                cost, validations, status, repair_status,
                joint, original,
                replays_used, ablation_replays_used, hits)

        def reject(candidate, reason, result):
            if self.ledger is not None:
                self.ledger.append("rejected_candidate", task_id=task_id, candidate_id=candidate.patch_id,
                                   reason=reason, details={"trajectory_id": result.trajectory_id})

        try:
            actions = compile_patch(original_actions, patch)
        except InvalidPatch as exc:
            if self.ledger is not None:
                self.ledger.append("rejected_candidate", task_id=task_id, candidate_id=patch.patch_id,
                                   reason=exc.reason, details={"patch": patch.to_json(),
                                                             "variant": "extended_repair" if
                                                             isinstance(exc, ExtendedRepair) else "default"})
            if isinstance(exc, ExtendedRepair):
                raise
            repair_status = "INVALID_PATCH"
            return finish()

        def replay(candidate, compiled, phase, *, reverted=None, base=None, force=False):
            nonlocal replays_used, ablation_replays_used, hits
            result = execute_actions(self.env_factory, task_id, compiled, self.max_steps,
                cache=None if force else self.cache, ledger=self.ledger, version=self.version,
                candidate_id=candidate.patch_id, interaction_kind=(
                    "edit_ablation_steps" if phase == "ablation" else "candidate_replay_steps"))
            hits += int(result.cache_hit)
            replays_used += int(not result.cache_hit)
            ablation_replays_used += int(phase == "ablation" and not result.cache_hit)
            evidence = {"phase": phase, "patch": candidate.to_json(), "execution": result.to_json(),
                        "base_patch_id": base, "reverted_block_id": reverted,
                        "same_initial_state": original is None or same_reset(original, result),
                        "accepted_revert": False}
            validations.append(evidence)
            return result, evidence

        if replays_used >= self.total_replays:
            return finish()
        original, evidence = replay(Patch(patch.original_trajectory_id), original_actions, "original", force=True)
        if original_trajectory is not None:
            evidence["matches_recorded_original"] = (
                patch.original_trajectory_id == original_trajectory.trajectory_id and
                original_trajectory.task_id == task_id and same_execution(original, original_trajectory))
            if not evidence["matches_recorded_original"]:
                repair_status = "NONREPLAYABLE"
                return finish()
        if original.termination_reason == "ENV_ERROR" or original.success is not False:
            repair_status = "NONREPLAYABLE"
            return finish()
        if replays_used >= self.total_replays:
            return finish()
        selected, evidence = replay(patch, actions, "candidate")
        if not evidence["same_initial_state"]:
            reject(patch, "initial_state_changed", selected)
            repair_status = "NONREPLAYABLE"
            return finish()
        if selected.success is not True:
            reject(patch, "candidate_not_verified_successful", selected)
            return finish()
        repair_status = "VERIFIED"
        cost = repair_cost(original_actions, selected, patch, preserved_subgoals)
        evidence["cost"] = list(cost)
        if check_minimality:
            # A successful revert changes the declared set. Discard the partial
            # pass for status purposes and restart at its first remaining block.
            while True:
                accepted, complete = False, True
                for block in patch.blocks:
                    if (ablation_replays_used >= self.ablation_replays or
                            replays_used >= self.total_replays - 1):
                        status, complete = "ABLATION_BUDGET_EXHAUSTED", False
                        break
                    reverted = Patch(patch.original_trajectory_id,
                                     tuple(b for b in patch.blocks if b.block_id != block.block_id))
                    result, evidence = replay(reverted, compile_patch(original_actions, reverted),
                                              "ablation", reverted=block.block_id, base=patch.patch_id)
                    if result.termination_reason == "ENV_ERROR":
                        reject(reverted, "ablation_environment_error", result)
                        complete = False
                        continue
                    if not evidence["same_initial_state"]:
                        reject(reverted, "initial_state_changed", result)
                        repair_status, complete = "NONREPLAYABLE", False
                        break
                    if result.success is True:
                        new_cost = repair_cost(original_actions, result, reverted, preserved_subgoals)
                        evidence["cost"] = list(new_cost)
                        if new_cost < cost:
                            evidence["accepted_revert"] = True
                            patch, selected, cost, accepted = reverted, result, new_cost, True
                            break
                        reject(reverted, "successful_revert_does_not_lower_cost", result)
                    elif result.success != False:
                        reject(reverted, "ablation_verification_unknown", result)
                        complete = False
                    else:
                        reject(reverted, "reverted_candidate_failed", result)
                if accepted:
                    continue
                if complete:
                    status = "BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS"
                break
        # Always real replay for the repeat confirmation, never a cache hit.
        if replays_used < self.total_replays and repair_status == "VERIFIED":
            confirmation, evidence = replay(patch, compile_patch(original_actions, patch), "confirmation", force=True)
            evidence["matches_selected_execution"] = same_execution(selected, confirmation)
            if not evidence["matches_selected_execution"]:
                repair_status, status = "NONREPLAYABLE", "BEST_OBSERVED"
        elif repair_status == "VERIFIED":
            status = "ABLATION_BUDGET_EXHAUSTED" if check_minimality else "BEST_OBSERVED"
        if repair_status == "NONREPLAYABLE":
            status = "BEST_OBSERVED"
        return finish()
