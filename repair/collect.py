"""Frozen E1 collection and offline human-review finalization. No evaluation input."""
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import csv
import io

from sed.manifest import authorize_task
from sed.storage import digest, write_json, JsonlLedger, file_hash
from repair.artifacts import provenance, run_manifest, write_sealed
from repair.budgets import GlobalTeacherBudget
from repair.driver import repair_one_failure
from repair.execute import execute_actions
from repair.labels import LearningEpisode, reconstruct_labels
from repair.ledger import BudgetLedger
from repair.minimality import same_execution
from repair.pairs import construct_pair, choose_far, review_target, review_evidence
from repair.prompts import diagnosis_prompt
from repair.records import RepairRecord
from repair.runtime import CaseSession, CaseStopped, Limits, OUTPUT_CAPS
from repair.schemas import parse_diagnosis
from repair.align import canonical_action
from repair.branches import branch_fields

# The preregistration requires a seeded selection but specifies no numeric seed.
# Freeze this implementation constant before any teacher request, never CLI-tune it.
PROBE_SPLIT_SEED = 20261029


def freeze_probe_split(tasks):
    ids = [t["task_id"] for t in tasks]
    if not 2 <= len(ids) <= 20 or len(set(ids)) != len(ids):
        raise ValueError("Expected 2..20 distinct frozen failures to reserve exactly two probes")
    if len({t["group_id"] for t in tasks}) != len(tasks):
        raise ValueError("Failure groups must be distinct")
    for t in tasks:
        authorize_task(t, "generation")
        if t.get("evaluation_only"):
            raise ValueError("Evaluation task cannot supply repairs")
    probes = sorted(ids, key=lambda t: (digest([PROBE_SPLIT_SEED, t]), t))[:2]
    return {"seed": PROBE_SPLIT_SEED, "rule": "lowest sha256([seed, task_id]); two before teacher calls",
            "failure_order": ids, "D_repair_probe": probes,
            "training_candidates": [t for t in ids if t not in probes]}


def public_review(repair, episode):
    return review_evidence(repair.executed_trajectory, episode)


def review_reader(reviews):
    def read(public):
        return reviews.get(digest(public))
    return read


def finalize_cases(cases, split, *, reviewer=None, seed_actions=None):
    """Recheck all nine pair conditions from retained evidence, with no calls."""
    seed_actions = seed_actions or {}
    pairs, exclusions = {}, []
    for case in cases:
        tid = case["task_id"]
        if "repair" not in case:
            exclusions.append({"task_id": tid, "reasons": [case.get("error_type", "missing_repair")]})
            continue
        near = RepairRecord.from_json(case["repair"])
        ep = LearningEpisode.from_json(case["near_episode"]) if case.get("near_episode") else None
        if near.repair_status != "VERIFIED" or ep is None:
            exclusions.append({"task_id": tid, "reasons": [case.get("label_status", near.repair_status)]})
            continue
        nr = review_target(near.executed_trajectory, ep, reviewer)
        candidates = []
        for item in case.get("labelled_candidates", []):
            far = RepairRecord.from_json(item["repair"])
            fe = LearningEpisode.from_json(item["episode"]) if item["episode"] else None
            if [canonical_action(a) for a in far.executed_trajectory.executed_actions] in seed_actions.get(tid, []):
                exclusions.append({"task_id": tid, "candidate_id": far.repair_id,
                                   "reasons": ["far_duplicates_original_teacher_seed"]})
                continue
            fr = review_target(far.executed_trajectory, fe, reviewer) if fe else None
            candidates.append((far, fe, fr))
        chosen, reasons = choose_far(near.original_trajectory, near.executed_trajectory, ep, nr, candidates)
        exclusions.extend({"task_id": tid, **r} for r in reasons)
        if chosen and tid not in split["D_repair_probe"]:
            far, fe, fr = chosen
            pairs[tid] = {"task_id": tid, "near": {"repair": near.to_json(), "episode": ep.to_json(),
                "review": asdict(nr)}, "far": {"repair": far.to_json(), "episode": fe.to_json(),
                "review": asdict(fr)}}
    selected = [pairs[t] for t in split["training_candidates"] if t in pairs][:4]
    return selected, exclusions


def collect(failures, tasks, config, output, *, env_factory, teacher_factory, tokenizer,
            tokenizer_id, checkpoint, inputs=(), reviewer=None, seed_actions=None, K=None,
            max_teacher_input_tokens=5_000_000, max_teacher_output_tokens=500_000,
            limits=None, canary=False, dry_run=False, tokenizer_metadata=None,
            seed_demonstrations=None, branch='P0_K20'):
    context = branch_fields(branch)
    split = freeze_probe_split(tasks)
    expected = split["failure_order"][:1] if canary else split["failure_order"]
    if [f.task_id for f in failures] not in (expected, split["failure_order"] if canary else expected):
        raise ValueError("Failures differ from frozen task order")
    if canary:
        max_teacher_input_tokens = min(max_teacher_input_tokens, 16_384)
        max_teacher_output_tokens = min(max_teacher_output_tokens, 128)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_sealed(output / "repair_split.json", {**split, **context})  # before teacher requests
    if dry_run and seed_demonstrations is None:
        from repair.dry_fakes import write_seed_demonstrations
        seed_tasks = tasks if branch != 'P1_U_mech' else [
            {**t, 'task_id': t['task_id'] + '-original-support', 'group_id': t['group_id'] + '-original-support'}
            for t in tasks]
        seed_demonstrations = write_seed_demonstrations(output / "synthetic_seeds", seed_tasks)
    config = deepcopy(config)
    case_limits = Limits(format_retry_calls=0) if canary else limits or Limits()
    config['repair_collection'] = {**context, 'case_limits': asdict(case_limits),
        'max_teacher_input_tokens': max_teacher_input_tokens,
        'max_teacher_output_tokens': max_teacher_output_tokens, 'canary': canary, 'dry_run': dry_run,
        'teacher_output_caps': {'diagnosis': 128} if canary else OUTPUT_CAPS,
        'transport_max_attempts': 1 if canary else 20}
    prov = provenance(output, config, command="repair_collect", inputs=inputs)
    manifest = run_manifest(prov, checkpoint=checkpoint,
        tokenizer=tokenizer_metadata or {"id": tokenizer_id, "chat_template_sha256": digest(getattr(tokenizer, "chat_template", None))},
        tasks=tasks, seeds={"repair_probe": PROBE_SPLIT_SEED, "environment": config["environment"]["seed"]},
        K_trusted_seed_trajectories=K, unique_instances=len(tasks), task_groups=len(tasks),
        **context, dry_run=dry_run, canary=canary, training_exposure=0, teacher_responses=[])
    write_json(output / "run_manifest.json", manifest, exclusive=True)
    budget = GlobalTeacherBudget(output / "global_teacher_budget.jsonl",
        input_tokens=max_teacher_input_tokens, output_tokens=max_teacher_output_tokens, context=context)
    teachers = teacher_factory(config, output / "teachers", run_id=prov["run_id"])
    cases, reviews, sheets, stopped = [], {}, [], None
    for index, (failure, task) in enumerate(zip(failures, tasks)):
        if canary and index:
            break
        directory = output / "cases" / f"{index:02d}-{digest(task['task_id'])[:12]}"
        directory.mkdir(parents=True)
        ledger = BudgetLedger(directory / "budget.jsonl", context=context)
        session = CaseSession(task=task, teacher=teachers, ledger=ledger, config=config,
            limits=case_limits,
            global_budget=budget, output_caps={"diagnosis": 128} if canary else None)
        case = {**context, "task_id": task["task_id"], "parent_seed_ids": task.get("parent_seed_ids", []),
                "generation_mode": "repair_U_mech_task" if branch == 'P1_U_mech' else "repair_existing_training_task",
                "design_bucket": 'G' if branch == 'P1_U_mech' else 'R', "training_exposure": 0}
        try:
            if canary:
                replay = execute_actions(env_factory, failure.task_id,
                    [s["action_raw"] for s in failure.steps], failure.max_steps, ledger=ledger)
                if failure.success is not False or not same_execution(failure, replay):
                    raise ValueError("Canary failure is not replayable")
                case["replay"] = replay.to_json()
                diag = session.request(diagnosis_prompt(task, failure, session.action_syntax, session.max_steps),
                    lambda raw: parse_diagnosis(raw, failure.trajectory_id, len(failure.steps)))
                case["diagnosis"] = diag.to_json()
            else:
                outcome = repair_one_failure(failure, env_factory=env_factory, session=session)
                case.update(repair=outcome.repair.to_json(), e1=outcome.e1, candidates=outcome.candidates,
                    handbacks=outcome.handbacks, stop_reason=outcome.stop_reason,
                    diagnosis=outcome.diagnosis.to_json() if outcome.diagnosis else None)
                if outcome.repair.repair_status == "VERIFIED" and not budget.stop_reason:
                    labels = reconstruct_labels(outcome.repair, session=session, tokenizer=tokenizer,
                        tokenizer_id=tokenizer_id, review_path=directory / "near_review.tsv")
                    case.update(near_episode=labels.episode.to_json() if labels.episode else None,
                                label_status=labels.status)
                    if labels.episode:
                        pair = construct_pair(outcome, labels, session=session, env_factory=env_factory,
                            tokenizer=tokenizer, tokenizer_id=tokenizer_id, reviewer=reviewer,
                            collect_pending=True)
                        case.update(pair_status=pair.status, pair_exclusions=pair.exclusions,
                                    far_candidates=pair.candidates, labelled_candidates=pair.labelled_candidates)
                        for r, ep in [(outcome.repair, labels.episode), *[
                            (RepairRecord.from_json(v["repair"]), LearningEpisode.from_json(v["episode"]))
                            for v in pair.labelled_candidates if v["episode"]]]:
                            public = public_review(r, ep)
                            reviews[digest(public)] = public
        except CaseStopped as exc:
            case["stop_reason"] = str(exc)
        except Exception as exc:
            # Keep prior case/transport evidence without propagating SDK bodies.
            case["error_type"] = type(exc).__name__
            stopped = "case_error"
        finally:
            case.update(costs=session.cost_record(), budget=ledger.summary())
            rows = ledger.rows()
            for row in rows:
                if row["event"] == "teacher_result":
                    meta = row["details"]["teacher_metadata"]
                    manifest["teacher_responses"].append({k: meta.get(k) for k in
                        ("provider_model_id", "returned_model_id", "timestamp")})
                if row["event"] == "label_result":
                    for item in csv.DictReader(io.StringIO(row["details"]["review_tsv"]), delimiter="\t"):
                        sheets.append({"task_id": task["task_id"], "protocol": context['protocol'],
                            "collection_branch": branch, "branch": row["details"]["branch"],
                            "episode_id": row["details"]["episode_id"], **item})
            cases.append(case)
            write_json(directory / "case.json", case, exclusive=True)
            JsonlLedger(output / "cases.jsonl").append(case)
        stopped = budget.stop_reason or stopped
        if stopped:
            break
    pairs, exclusions = finalize_cases(cases, split, reviewer=reviewer, seed_actions=seed_actions) if not canary else ([], [])
    exclusions += [{"task_id": c["task_id"], **e} for c in cases for e in c.get("pair_exclusions", [])]
    exclusions += [{"task_id": t, "reasons": [stopped or "canary_only"]}
                   for t in split["failure_order"] if t not in {c["task_id"] for c in cases}]
    from repair.probes import probe_point
    valid_probes = []
    for case in cases:
        if case['task_id'] in split['D_repair_probe']:
            try:
                probe_point(case)
                valid_probes.append(case['task_id'])
            except (ValueError, KeyError):
                pass
    if canary and not stopped:
        stopped = "canary_complete" if cases and "diagnosis" in cases[0] else "canary_failed"
    for row in pairs + exclusions:
        row.update(context)
    value = {"schema_version": 1, **context, "split": split, "tasks": tasks, "cases": cases, "pairs": pairs,
        "D_repair_train": [p["task_id"] for p in pairs], "D_repair_probe": split["D_repair_probe"],
        "exclusions": exclusions, "seed_actions": seed_actions or {}, "checkpoint": checkpoint,
        "seed_demonstrations": seed_demonstrations or {},
        "tokenizer": manifest["tokenizer"], "K": K, "dry_run": dry_run, "canary": canary,
        "status": stopped or "complete", "global_budget": budget.summary(),
        "valid_probe_task_ids": valid_probes,
        "shortfall": {"train_pairs": 4 - len(pairs), "probe_tasks": 2 - len(split["D_repair_probe"]),
                      "probe_execution_evidence": 2 - len(valid_probes)},
        "input_hashes": prov["inputs"],
        "original_teacher_far_policy": "excluded_without_replay_pair_check",
        "seed_replay_tasks_are_repair_tasks": branch != 'P1_U_mech',
        "supervision_accounting": {
            "initial_teacher_demonstrations": K,
            "seed_support_unique_instances": len(seed_demonstrations or {}),
            "seed_support_task_groups": len({r['receipt']['group_id']
                for refs in (seed_demonstrations or {}).values() for r in refs}),
            "theta0_seed_training_trajectories": 0 if branch == 'P0_K0' else K,
            "repair_task_source": 'U_mech' if branch == 'P1_U_mech' else 'K20_seed_tasks',
            "new_task_ids": split['failure_order'] if branch == 'P1_U_mech' else [],
            "new_task_unique_instances": len(tasks) if branch == 'P1_U_mech' else 0,
            "new_task_groups": len({t['group_id'] for t in tasks}) if branch == 'P1_U_mech' else 0,
            "case_costs": [{"task_id": c['task_id'], **context, "budget": c['budget'],
                            "costs": c['costs']} for c in cases],
            "training_exposure": 0}}
    write_sealed(output / "collection.json", value)
    from repair.export import write_training_export
    write_training_export(output)
    write_json(output / "exclusions.json", exclusions, exclusive=True)
    write_json(output / "review_requests.json", reviews, exclusive=True)
    evidence_keys = {v["episode_id"]: k for k, v in reviews.items()}
    for row in sheets:
        row["review_evidence_sha256"] = evidence_keys.get(row["episode_id"])
    with (output / "human_review.tsv").open("x", newline="") as handle:
        fields = list(sheets[0]) if sheets else ["task_id", "episode_id", "human_supported", "human_notes"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(sheets)
    manifest.update(status=value["status"], global_budget=budget.summary(),
                    collection_sha256=digest(value), completed_cases=len(cases),
                    supervision_accounting=value['supervision_accounting'],
                    e2_training_collection_sha256=file_hash(output / "e2_training_collection.json"))
    write_json(output / "run_manifest.json", manifest)
    return value
