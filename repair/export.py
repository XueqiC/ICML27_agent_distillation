"""Versioned, public-only E2 interchange. No trainer, model or optimizer code."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from repair.artifacts import checked_document, write_sealed
from repair.branches import collection_branch
from repair.labels import LearningEpisode
from repair.pairs import TargetReview, pair_exclusions, review_evidence
from repair.records import RepairRecord
from sed.actions import parse_action
from sed.storage import digest, file_hash
from sed.thinking import split_thinking
from sed.visibility import visible_observation

FORMAT = "repair_e2_training_collection"
SCHEMA_VERSION = 2


def export_branch(value):
    """Schema 1 is a legacy P0_K20 adapter export only; never infer P1/K0."""
    fields = collection_branch(value)
    version = value.get('schema_version')
    if (value.get('format') != FORMAT or type(version) is not int or version not in (1, 2)
            or (version == 1 and fields['branch'] != 'P0_K20')
            or (version == 2 and any(k not in value for k in fields))):
        raise ValueError('Unsupported E2 export version/branch')
    return fields


def learning_view(repair, episode):
    """Rebuild the existing schema-2 episode from explicit public fields only."""
    target = repair.executed_trajectory
    if (target is None or target.success is not True or episode.view != "full_trajectory"
            or episode.trajectory_id != target.trajectory_id or episode.repair_id != repair.repair_id
            or episode.task_id != target.task_id
            or episode.full_trajectory_decision_ids != list(range(len(target.steps)))
            or len(episode.history) != len(target.steps)):
        raise ValueError("Mismatched full-trajectory LearningEpisode")
    steps, history = [], []
    previous = visible_observation(target.initial["visible"])
    for i, (turn, step) in enumerate(zip(episode.history, target.steps)):
        reply, hidden = split_thinking(turn["reply"])
        if hidden or reply != turn["reply"] or parse_action(reply) != step["action_parsed"]:
            raise ValueError("Label action or visible reasoning differs from execution")
        history.append({"visible": previous, "reply": reply})
        previous = visible_observation(step["visible"])
        steps.append({"event_id": f"{episode.stable_id}:{i}", "action_raw": reply, "visible": previous})
    if history != episode.student_history:
        raise ValueError("LearningEpisode public history differs from the harness")
    return {"schema_version": 2, "task_id": target.task_id, "split": "train", "sealed": False,
            "initial": {"visible": visible_observation(target.initial["visible"])}, "steps": steps}


def seed_references(verified_path, collection, evidence_root, episodes):
    """Call only after load_verified_collection/checked_episodes have verified receipts."""
    result = {}
    for receipt, episode in zip(collection["verified"], episodes, strict=True):
        tid = receipt["task_id"]
        if tid != episode["task_id"]:
            raise ValueError("Seed receipt and episode identities differ")
        result.setdefault(tid, []).append({
            "task_id": tid, "episode_id": episode["episode_id"],
            "verified_collection_path": str(Path(verified_path).resolve()),
            "verified_manifest_sha256": file_hash(Path(verified_path) / "manifest.json"),
            "evidence_root": str(Path(evidence_root).resolve()),
            "episode_dir": receipt["episode_dir"],
            "episode_sha256": receipt["episode_hashes"]["episode.json"],
            "receipt_sha256": digest(receipt), "receipt": deepcopy(receipt)})
    return result


def training_export(collection, *, collection_sha256):
    """Only selected, approved pairs enter arms; pending evidence stays in collection.json."""
    context = collection_branch(collection)
    original_support = context['seed_replay_scope'] == 'original_support'
    support = collection.get('seed_demonstrations', {})
    support_refs = [r for refs in support.values() for r in refs]
    if original_support:
        if (set(support) & set(collection['split']['failure_order'])
                or any(r['task_id'] != tid for tid, refs in support.items() for r in refs)
                or {r['receipt']['group_id'] for r in support_refs} & {t['group_id'] for t in collection['tasks']}):
            raise ValueError('P1 replay must use original support disjoint from repair tasks/groups')
        if len(support_refs) != collection['K'] or (not collection['dry_run'] and collection['K'] != 20):
            raise ValueError('P1 replay requires the complete original K20 support')
    tids = collection["D_repair_train"]
    probes = collection["D_repair_probe"]
    tasks = {t["task_id"]: t for t in collection["tasks"]}
    if (tids != [p["task_id"] for p in collection["pairs"]] or len(set(tids)) != len(tids)
            or len(tids) > 4 or len(probes) != 2 or set(tids) & set(probes)
            or {tasks[t]["group_id"] for t in tids} & {tasks[t]["group_id"] for t in probes}):
        raise ValueError("Invalid E2 task split/order")
    arms = {"N": [], "F": []}
    blockers = []
    if collection["dry_run"] or collection["canary"]:
        blockers.append("synthetic_or_canary")
    if len(tids) != 4:
        blockers.append("four_approved_pairs_required")
    for pair in collection["pairs"]:
        tid = pair["task_id"]
        repairs = [RepairRecord.from_json(pair[k]["repair"]) for k in ("near", "far")]
        episodes = [LearningEpisode.from_json(pair[k]["episode"]) for k in ("near", "far")]
        reviews = [TargetReview(**pair[k]["review"]) for k in ("near", "far")]
        if pair_exclusions(repairs[0].original_trajectory, *[r.executed_trajectory for r in repairs],
                           *episodes, *reviews):
            raise ValueError("Export requires approved E2 pair evidence")
        refs = support_refs if original_support else support.get(tid, [])
        if not refs:
            blockers.append("missing_seed_demonstrations:" + tid)
        if not original_support and any(ref["task_id"] != tid for ref in refs):
            raise ValueError("Seed demonstration is outside the repair-training task")
        for arm, repair, episode, review in zip(("N", "F"), repairs, episodes, reviews):
            public = learning_view(repair, episode)
            if public["task_id"] != tid:
                raise ValueError("Pair task differs from LearningEpisode")
            evidence = review_evidence(repair.executed_trajectory, episode)
            arms[arm].append({**context, "task_id": tid, "group_id": tasks[tid]["group_id"],
                "learning_episode_id": episode.stable_id, "repair_id": repair.repair_id,
                "trajectory_id": repair.executed_trajectory.trajectory_id,
                "episode": public, "episode_sha256": digest(public),
                "human_review": {"status": "approved", "reviewed": True, "approved": True,
                    "evidence_sha256": digest(evidence), "episode_sha256": digest(public),
                    "decision": asdict(review)},
                "seed_demonstrations": deepcopy(refs), "source_bucket": "successful_repair",
                "design_bucket": 'G' if original_support else 'R', "seed_design_bucket": 'D'})
    return {"format": FORMAT, "schema_version": SCHEMA_VERSION, **context,
        "source_collection_sha256": collection_sha256,
        "source_collection_content_sha256": collection.get("content_sha256", digest(collection)),
        "theta0": deepcopy(collection["checkpoint"]), "tokenizer": deepcopy(collection["tokenizer"]),
        "input_hashes": deepcopy(collection["input_hashes"]),
        "split": deepcopy(collection["split"]), "D_repair_train": tids, "D_repair_probe": probes,
        "K_trusted_seed_trajectories": collection["K"], "unique_instances": len(tids),
        "task_groups": len({tasks[t]["group_id"] for t in tids}),
        "dry_run": collection["dry_run"], "canary": collection["canary"],
        "training_exposure": 0, "ready_for_training": not blockers,
        "training_blockers": blockers, "arms": arms,
        "seed_replay_task_ids": list(support) if original_support else list(tids),
        "seed_replay_tasks_are_repair_tasks": not original_support,
        "original_teacher_far_policy": collection.get('original_teacher_far_policy', 'excluded_without_replay_pair_check'),
        "supervision_accounting": deepcopy(collection.get('supervision_accounting', {}))}


def write_training_export(directory):
    directory = Path(directory)
    path = directory / "collection.json"
    value = training_export(checked_document(path), collection_sha256=file_hash(path))
    write_sealed(directory / "e2_training_collection.json", value)
    return value


def require_approved(entry):
    """Interchange guard for consumers/tests; never treats missing approval as consent."""
    review = entry.get("human_review", {})
    decision = review.get("decision", {})
    if (review.get("status") != "approved" or review.get("reviewed") is not True
            or review.get("approved") is not True
            or any(decision.get(k) is not True for k in
                   ("no_rejected_actions", "meaningful_route", "labels_supported"))
            or not decision.get("reviewer") or not decision.get("notes")
            or decision.get("episode_id") != entry["learning_episode_id"]
            or decision.get("trajectory_id") != entry["trajectory_id"]
            or entry["episode_sha256"] != digest(entry["episode"])
            or review.get("episode_sha256") != entry["episode_sha256"]):
        raise ValueError("Training must refuse unreviewed, unapproved or changed episodes")
