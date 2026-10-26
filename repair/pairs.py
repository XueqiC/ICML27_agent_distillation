"""Bounded Far search and deterministic E2 pairing; Near is never substituted."""

from dataclasses import dataclass, field
from math import ceil
import re

from sed.storage import digest
from repair.align import align_actions
from repair.driver import search_candidates
from repair.execute import ResultCache
from repair.labels import instance_ids, reconstruct_labels
from repair.minimality import same_reset
from repair.prompts import public_trace
from repair.records import RepairRecord
from repair.runtime import CaseSession, CaseStopped


@dataclass(frozen=True)
class TargetReview:
    trajectory_id: str
    episode_id: str
    reviewer: str
    no_rejected_actions: bool
    meaningful_route: bool
    labels_supported: bool
    notes: str


def review_target(trajectory, episode, reviewer):
    """The injected human-review reader sees public fields only, never an oracle.

    Natural-language support and meaningless detours cannot be proved by regex.
    A missing review remains unknown, which excludes a pair until reviewed.
    """
    if reviewer is None:
        return None
    value = reviewer(review_evidence(trajectory, episode))
    if value is None:
        return None
    fields = {"reviewer", "no_rejected_actions", "meaningful_route", "labels_supported", "notes"}
    if (type(value) is not dict or set(value) != fields or
            any(type(value[k]) is not bool for k in fields - {"reviewer", "notes"}) or
            not isinstance(value["reviewer"], str) or not value["reviewer"].strip() or
            not isinstance(value["notes"], str) or not value["notes"].strip()):
        raise ValueError("Pair review must contain explicit decisions, reviewer and public evidence notes")
    return TargetReview(trajectory.trajectory_id, episode.stable_id, **value)


def review_evidence(trajectory, episode):
    # IDs bind offline approvals to this execution/label artifact. These are
    # review metadata, never teacher or student prompt fields.
    return {"trajectory_id": trajectory.trajectory_id, "episode_id": episode.stable_id,
            "execution": public_trace(trajectory), "labels": episode.history,
            "prefix_checks": episode.prefix_checks}


def coverage_signature(trajectory):
    """Conservative ALFWorld goal-operation signature from executed public text.

    Targets are placed objects (or the held object for a light/use task), not
    all incidental navigation locations/appliances. Unknown syntax is not
    guessed from task paths, task_type, admissible lists or hidden goal fields.
    """
    objects, destinations, operations, held, used = set(), set(), set(), set(), set()
    for step in trajectory.steps:
        action = step["action_parsed"] or ""
        # The harness uses move/to; legacy put/in/on has the same placement
        # signature so object and required-operation comparisons share one key.
        match = (re.fullmatch(r"move (.+? \d+) to (.+? \d+)", action) or
                 re.fullmatch(r"put (.+? \d+) (?:in/on|in|on) (.+? \d+)", action))
        if match:
            objects.add(match[1].lower())
            destinations.add(match[2].lower())
            operations.add(("put", match[1].lower()))
        match = re.fullmatch(r"(clean|heat|cool) (.+? \d+) (?:with|in) (.+? \d+)", action)
        if match:
            operations.add((match[1], match[2].lower()))
        match = re.fullmatch(r"take (.+? \d+) from (.+? \d+)", action)
        if match:
            held.add(match[1].lower())
        match = re.fullmatch(r"use (.+? \d+)", action)
        if match:
            used.add(match[1].lower())
    if not objects and used and held:
        objects = held
        destinations = used
        operations.update(("use_light", obj) for obj in objects)
    if not objects or not destinations:
        return None
    # Instance IDs and the operations must have public execution witnesses;
    # a successful terminal label alone never fills in a missing operation.
    observed = instance_ids("\n".join([trajectory.initial["visible"]["observation"],
        *[s["visible"]["observation"] for s in trajectory.steps if s["visible"]]]))
    if not (objects | destinations) <= observed:
        return None
    relevant = {(op, obj) for op, obj in operations if obj in objects}
    return {"objects": sorted(objects), "destinations": sorted(destinations),
            "operations": sorted(relevant)}


def required_operations(task_text):
    text = task_text.lower()
    result = set()
    for words, operation in ((r"\b(clean|washed)\b", "clean"), (r"\b(heat|hot|heated)\b", "heat"),
                             (r"\b(cool|cold|cooled)\b", "cool"),
                             (r"\b(put|place|move)\b", "put"), (r"\b(light|lamp)\b", "use_light")):
        if re.search(words, text):
            result.add(operation)
    return result


def target_exclusions(trajectory, episode, review):
    reasons = []
    if trajectory.success is not True or trajectory.termination_reason != "SUCCESS":
        reasons.append("no_real_success")
    if not trajectory.steps or any(s.get("feedback_source") not in {
            "real_environment", "cached_real_environment"} for s in trajectory.steps):
        reasons.append("missing_real_execution_witness")
    # A positive parser admissibility classification is NOT proof of validity.
    if any(re.search(r"\b(nothing happens|invalid action|cannot|can't|don't see)\b",
                     s["visible"]["observation"], re.I) for s in trajectory.steps if s["visible"]):
        reasons.append("environment_rejected_action")
    if coverage_signature(trajectory) is None:
        reasons.append("public_goal_coverage_unknown")
    if episode is None or episode.trajectory_id != trajectory.trajectory_id:
        reasons.append("missing_or_mismatched_labels")
        return reasons
    if episode.label_protocol.get("token_scope") != "native_assistant_and_turn_end":
        reasons.append("supervised_token_count_requires_student_tokenizer")
    if not all(c["structural_pass"] for c in episode.prefix_checks):
        reasons.append("prefix_check_failed")
    if review is None or review.trajectory_id != trajectory.trajectory_id or review.episode_id != episode.stable_id:
        reasons.append("public_quality_review_missing")
    else:
        if not review.no_rejected_actions:
            reasons.append("environment_rejected_action")
        if not review.meaningful_route:
            reasons.append("meaningless_route_or_padding")
        if not review.labels_supported:
            reasons.append("unsupported_reasoning")
    return list(dict.fromkeys(reasons))


def pair_exclusions(original, near, far, near_episode, far_episode, near_review, far_review):
    """All nine §14.6 conditions, with independent reasons retained."""
    reasons = ["near:" + r for r in target_exclusions(near, near_episode, near_review)]
    reasons += ["far:" + r for r in target_exclusions(far, far_episode, far_review)]
    if (original.task_id != near.task_id or near.task_id != far.task_id or
            original.max_steps != near.max_steps or near.max_steps != far.max_steps or
            not same_reset(original, near) or not same_reset(near, far)):
        reasons.append("task_initial_state_success_standard_or_horizon")
    n_signature, f_signature = coverage_signature(near), coverage_signature(far)
    if n_signature and f_signature:
        if any(n_signature[k] != f_signature[k] for k in ("objects", "destinations")):
            reasons.append("target_object_instances")
        if n_signature["operations"] != f_signature["operations"]:
            reasons.append("required_operation_types")
        required = required_operations(near_episode.task_text) if near_episode else set()
        if not required or any(not required <= {op for op, _ in signature["operations"]}
                               for signature in (n_signature, f_signature)):
            reasons.append("required_task_operations_missing_or_unknown")
    ln, lf = len(near.executed_actions), len(far.executed_actions)
    if not ln or not 0.8 <= lf / ln <= 1.25:
        reasons.append("action_length_ratio")
    if near_episode and far_episode:
        if (near_episode.task_text != far_episode.task_text or
                near_episode.label_protocol != far_episode.label_protocol or
                near_episode.view != far_episode.view):
            reasons.append("label_construction_protocol")
        tn, tf = near_episode.supervised_target_tokens, far_episode.supervised_target_tokens
        if not tn or not 0.8 <= tf / tn <= 1.25:
            reasons.append("supervised_token_ratio")
    dn = align_actions(original.executed_actions, near.executed_actions).distance
    df = align_actions(original.executed_actions, far.executed_actions).distance
    if df - dn < max(2, ceil(0.15 * max(ln, lf))):
        reasons.append("action_distance_gap")
    return reasons


@dataclass
class PairOutcome:
    status: str
    near_repair_id: str
    far_repair: RepairRecord | None
    far_episode: object
    exclusions: list[dict]
    candidates: list[dict]
    costs: dict
    labelled_candidates: list[dict] = field(default_factory=list)


def choose_far(original, near, near_episode, near_review, candidates):
    """Candidates are (repair, episode, review); no Near reselection is possible."""
    eligible, exclusions = [], []
    for repair, episode, review in candidates:
        far = repair.executed_trajectory
        reasons = pair_exclusions(original, near, far, near_episode, episode, near_review, review)
        exclusions.append({"candidate_id": repair.selected_patch.patch_id, "reasons": reasons})
        if not reasons:
            eligible.append((repair, episode, review))
    if not eligible:
        return None, exclusions
    ln, tn = len(near.executed_actions), near_episode.supervised_target_tokens
    selected = min(eligible, key=lambda r: (
        # N's denominator is fixed: integer distance avoids floating-point
        # asymmetry between equally close shorter and longer candidates.
        abs(len(r[0].executed_trajectory.executed_actions) - ln),
        abs(r[1].supervised_target_tokens - tn), r[0].selected_patch.patch_id))
    return selected, exclusions


def construct_pair(near_outcome, near_labels, *, session, env_factory, reviewer=None,
                   tokenizer=None, tokenizer_id=None, cache=None, proposal_rounds=1,
                   collect_pending=False):
    """Default: one extra alternative-route request (up to three candidates).

    Explicit proposal_rounds=2 enables the design's ceiling of 2x3 for Far.
    Teacher costs are separate; retry and total replay budgets are case-shared.
    Human review may be supplied by an offline TSV reader; absent review yields
    PAIR_NOT_FOUND with retained E1 and no synthetic approvals.
    """
    near = near_outcome.repair
    episode = near_labels.episode
    far_session = CaseSession(task=session.task, teacher=session.teacher, ledger=session.ledger,
        config=session.config, limits=session.limits, branch="far", shared=session.shared,
        global_budget=session.global_budget, output_caps=session.output_caps)
    reviews, evidence, exclusions, chosen = [], [], [], None
    nr = review_target(near.executed_trajectory, episode, reviewer) if episode else None
    quality = target_exclusions(near.executed_trajectory, episode, nr) if near.executed_trajectory else ["missing_near"]
    if collect_pending and nr is None:
        quality = [r for r in quality if r != "public_quality_review_missing"]
    if near.repair_status != "VERIFIED":
        quality.append("near_not_confirmed")
    if quality:
        exclusions.append({"candidate_id": near.selected_patch.patch_id, "reasons": ["near:" + r for r in quality]})
    else:
        try:
            successes, evidence = search_candidates(near.original_trajectory, near_outcome.diagnosis,
                env_factory, far_session, cache=cache or ResultCache(), near=near.executed_trajectory,
                rounds=proposal_rounds)
            for patch, target in successes:
                route_reasons = [r for r in pair_exclusions(near.original_trajectory,
                    near.executed_trajectory, target, episode, None, nr, None)
                    if r != "far:missing_or_mismatched_labels" and not (
                        collect_pending and nr is None and r == "near:public_quality_review_missing")]
                if route_reasons:
                    exclusions.append({"candidate_id": patch.patch_id, "reasons": route_reasons})
                    continue
                alignment = align_actions(near.original_trajectory.executed_actions, target.executed_actions)
                far = RepairRecord(near.original_trajectory_id, patch, target, alignment,
                    (alignment.distance, 0, len(target.executed_actions), patch.patch_id),
                    [], "BEST_OBSERVED", "VERIFIED", {}, near.original_trajectory)
                labels = reconstruct_labels(far, session=far_session, tokenizer=tokenizer,
                    tokenizer_id=tokenizer_id, reasoning=episode.label_protocol["reasoning"])
                review = review_target(target, labels.episode, reviewer) if labels.episode else None
                reviews.append((far, labels.episode, review))
            chosen, filtered = choose_far(near.original_trajectory, near.executed_trajectory, episode, nr, reviews)
            exclusions.extend(filtered)
        except CaseStopped as exc:
            exclusions.append({"candidate_id": None, "reasons": [str(exc)]})
    # Failed, invalid and deduplicated F proposals are exclusions too; retain
    # them alongside the quality/coverage/length/token/distance filter results.
    checked = {row["candidate_id"] for row in exclusions}
    for row in evidence:
        candidate_id = row["patch"]["patch_id"]
        if candidate_id not in checked:
            exclusions.append({"candidate_id": candidate_id,
                "reasons": [row.get("reason", "not_labelled_or_reviewed")]})
    for row in exclusions:
        far_session.log("pair_candidate_filter", candidate_id=row["candidate_id"], details=row)
    status = "PAIRED" if chosen else "PAIR_NOT_FOUND"
    far_session.log(status, details={"near_repair_id": near.repair_id,
        "selected_far_id": chosen[0].repair_id if chosen else None,
        "reviews": [{"trajectory_id": r.trajectory_id, "episode_id": r.episode_id,
                     "reviewer": r.reviewer, "notes": r.notes,
                     "no_rejected_actions": r.no_rejected_actions, "meaningful_route": r.meaningful_route,
                     "labels_supported": r.labels_supported} for r in [nr, *[x[2] for x in reviews]] if r]})
    return PairOutcome(status, near.repair_id, chosen[0] if chosen else None,
                       chosen[1] if chosen else None, exclusions, evidence, far_session.cost_record(),
                       [{"repair": r.to_json(), "episode": e.to_json() if e else None}
                        for r, e, _ in reviews])


def first_pairs(frozen_task_order, outcomes, count=4):
    """Same frozen list for both arms; never sort by post-update performance."""
    if len(set(frozen_task_order)) != len(frozen_task_order):
        raise ValueError("Frozen task order contains duplicates")
    return [(task_id, outcomes[task_id]) for task_id in frozen_task_order
            if task_id in outcomes and outcomes[task_id].status == "PAIRED"][:count]
