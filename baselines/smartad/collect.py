"""CPU-only, durable ten-candidate pools. Selection never invokes a teacher."""
import json
from pathlib import Path

from baselines.adftp.collect import usage_summary
from baselines.common.explore import training_tasks
from baselines.smartad.teacher import (CANDIDATES, CALLS_PER_TASK, REQUESTED_SAMPLING,
                                      SmartADPolicy, default_teacher, effective_sampling, read)
from sed.manifest import frozen_write, resolve_task, verify_task_files
from sed.provenance import code_hashes, safe_config, versions, write_provenance
from sed.react import PROMPT_TEMPLATE, PROMPT_VERSION
from sed.runner import _environment, run_episode
from sed.seed_evidence import artifact_hashes, checked_episode, environment_won, load_verified_collection
from sed.seed_selection import is_full_train
from sed.storage import BlobStore, digest, locked, write_json
from teachers.baseline_transport import baseline_transport
from teachers.openai_teacher import TeacherCallError
from teachers.stage import require_execution, teacher_stage


def correct(directory, episode):
    """Success is a real win with no failed/parser/inadmissible tool turns."""
    if not environment_won(directory, episode):
        return False
    blobs = BlobStore(Path(directory) / "blobs")
    for step in episode["steps"]:
        verification = blobs.get(step["verification_ref"])
        if (step.get("action_outcome") != "legal"
                or verification.get("executable") is not True
                or verification.get("fact_source") != "real_environment"
                or step.get("visible", {}).get("observation", "").strip() == "Nothing happens."):
            return False
    return True


def reuse_check(source_root, receipt, config, sampling):
    """Require identity AND actual successful request receipts, never guesses."""
    episode = checked_episode(source_root, receipt)
    identity = read(Path(source_root) / "identity.json")
    source_config = identity.get("config", {})
    if source_config.get("models", {}).get("teacher", {}).get("provider_model_id") != config["models"]["teacher"]["provider_model_id"]:
        return False, "source_teacher_identity_differs_or_missing"
    source_sampling = source_config.get("models", {}).get("teacher", {}).get("sampling")
    if source_sampling != sampling:
        return False, "source_collection_sampling_differs"
    for section, keys in (("environment", ("max_episode_steps", "seed")),
                          ("training", ("thinking_mode", "student_step_token_cap"))):
        if any(source_config.get(section, {}).get(key) != config[section].get(key) for key in keys):
            return False, "source_h0_configuration_differs"
    directory = Path(source_root) / receipt["episode_dir"]
    if not correct(directory, episode):
        return False, "trusted_win_has_failed_tool_turns"
    rows = []
    for path in sorted((directory.parent / "teacher").rglob("usage.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    successful = {r["call_id"]: r for r in rows if r.get("status") == "success"}
    for step in episode["steps"]:
        row = successful.get(step.get("policy_metadata", {}).get("call_id"))
        if (not row or row.get("sampling") != sampling
                or row.get("provider_model_id") != config["models"]["teacher"]["provider_model_id"]
                or row.get("prompt_template_sha256") != digest(PROMPT_TEMPLATE)):
            return False, "source_call_receipts_do_not_prove_identical_sampling_and_h0"
    return True, "matching_collection_identity_and_executed_call_receipts"


def candidate(config, task, directory, identity, rate_limit_dir, sampling_root, *,
              teacher_factory, env_factory):
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "candidate.json"
    if record_path.exists():
        record = read(record_path)
        if artifact_hashes(directory / "episode") != record["episode_hashes"] or artifact_hashes(directory / "teacher") != record["teacher_hashes"]:
            raise ValueError("SmartAD candidate evidence changed")
        return record
    teacher = teacher_factory(config, directory / "teacher", task, identity, rate_limit_dir, sampling_root)
    teacher.check_health()
    started = directory / "started.json"
    if not started.exists():
        frozen_write(started, {"task_id": task["task_id"], "parent_seed_ids": task["parent_seed_ids"]})
        verify_task_files(config["environment"]["data_root"], task)
        try:
            run_episode(env_factory(task), SmartADPolicy(teacher, task), task, config, directory / "episode",
                        policy_condition="smartad_teacher_h0", purpose="generation",
                        parent_seed_ids=task["parent_seed_ids"])
        except TeacherCallError:
            raise
        except RuntimeError:
            if not (directory / "episode/episode.json").exists():
                raise
    # Never re-enter an existing attempt, even after an interrupted execution.
    # An unresolved attempt stops the collection; it cannot be called again.
    episode_path = directory / "episode/episode.json"
    episode = read(episode_path) if episode_path.exists() else {}
    require_execution(episode)
    teacher.check_health()
    if episode.get("error_type") in {"policy_error", "reset_error"}:
        raise RuntimeError("SmartAD policy/reset failed; inspect retained sanitized execution evidence")
    accepted = correct(directory / "episode", episode)
    record = {"candidate_id": directory.name, "task_id": task["task_id"],
        "episode_dir": str((directory / "episode").resolve()), "reused": False,
        "parent_seed_ids": task["parent_seed_ids"],
        "correct": accepted, "environment_win": environment_won(directory / "episode", episode),
        "success": episode.get("success", "unknown"), "status": episode.get("status", "interrupted"),
        "rejection": None if accepted else "no_verified_win_or_failed_tool_call",
        "metrics": episode.get("metrics"), "episode_hashes": artifact_hashes(directory / "episode"),
        "teacher_hashes": artifact_hashes(directory / "teacher"),
        "api_attempts": teacher._read_counter()["attempts"],
        "usage": usage_summary(directory / "teacher/usage.jsonl")}
    frozen_write(record_path, record)
    return record


def validate_h0(config):
    if (config["environment"]["max_episode_steps"] != 40 or config["training"]["thinking_mode"] != "off"
            or config["training"]["student_step_token_cap"] != 1024):
        raise ValueError("SmartAD requires H0: 40 turns, thinking off, 1024 output tokens")


@teacher_stage("output", name="smartad.collect")
def collect(verified, manifest, freeze, config, output, *, rate_limit_dir,
            env_factory=None, teacher_factory=default_teacher, input_paths=(), subset_ks=()):
    output = Path(output).resolve()
    source, selection, source_root = load_verified_collection(verified, manifest, freeze)
    if not source["verified"]:
        raise ValueError("SmartAD requires at least one trusted support trajectory")
    for directory in (output, Path(rate_limit_dir).resolve()):
        if any(directory.is_relative_to(root) for root in
               (source_root, Path(config["paths"]["raw"]).resolve(), Path(config["paths"]["verified"]).resolve())):
            raise ValueError("SmartAD output/limiter must be outside original seed storage")
    if (is_full_train(selection) and subset_ks) or any(k < 1 or k >= selection["K"] for k in subset_ks):
        raise ValueError("Nested views require smaller positive K and a few-shot source")
    validate_h0(config)
    parents = {r["task_id"]: [checked_episode(source_root, r)["episode_id"]] for r in source["verified"]}
    ids = list(parents)
    training_tasks({"task_ids": ids, "parent_seed_ids": parents, "manifest_sha256": digest(manifest),
                    "split_freeze_sha256": digest(freeze), "list_sha256": digest(ids)}, manifest, freeze)
    identity = {"method": "smartad", "selection": selection, "source_verified": str(Path(verified).resolve()),
        "source_collection_sha256": digest(source), "config": safe_config(config),
        "code_sha256": code_hashes(Path(__file__).resolve().parents[2]), "dependencies": versions(),
        "prompt_version": PROMPT_VERSION, "sampling_requested": REQUESTED_SAMPLING,
        "sampling_fallback": "only explicit HTTP400 unsupported temperature; provider default",
        "baseline_transport": baseline_transport("smartad").disclosure(),
        "candidates_per_task": CANDIDATES, "calls_per_task_cap": CALLS_PER_TASK,
        "per_cell_call_cap": CALLS_PER_TASK * len(ids),
        "subset_ks": sorted(set(subset_ks)), "rate_limit_dir": str(Path(rate_limit_dir).resolve())}
    env_factory = env_factory or (lambda task: _environment(config, task))
    with locked(output / ".pipeline.lock"):
        frozen_write(output / "identity.json", identity)
        if not (output / "provenance.json").exists():
            write_provenance(output, command="smartad_collect", config=config,
                inputs=input_paths, details=identity)
        outcomes = []
        for original in source["verified"]:
            task = resolve_task(manifest, freeze, original["task_id"])
            task["parent_seed_ids"] = parents[task["task_id"]]
            work = output / "work" / digest(task["task_id"])[:20]
            if (work / "outcome.json").exists():
                outcome = read(work / "outcome.json")
                if outcome["source_receipt"] != original:
                    raise ValueError("SmartAD source receipt changed")
                for row in outcome["candidates"]:
                    directory = Path(row["episode_dir"])
                    if (artifact_hashes(directory) != row["episode_hashes"]
                            or artifact_hashes(directory.parent / "teacher") != row["teacher_hashes"]):
                        raise ValueError("SmartAD candidate evidence changed")
                outcomes.append(outcome)
                continue
            # Nine always-needed samples first. A temperature rejection during
            # these calls can establish that a provider-default seed is reusable.
            candidates = [candidate(config, task, work / f"candidate-{i:02d}", digest(identity),
                rate_limit_dir, output, teacher_factory=teacher_factory, env_factory=env_factory)
                for i in range(2, CANDIDATES + 1)]
            reuse, reason = reuse_check(source_root, original, config, effective_sampling(output))
            if reuse:
                first = {"candidate_id": "candidate-01", "task_id": task["task_id"], "reused": True,
                    "correct": True, "environment_win": True, "rejection": None,
                    "episode_dir": str((source_root / original["episode_dir"]).resolve()),
                    "episode_hashes": original["episode_hashes"], "teacher_hashes": original["teacher_hashes"],
                    "parent_seed_ids": task["parent_seed_ids"], "api_attempts": 0}
            else:
                first = candidate(config, task, work / "candidate-01", digest(identity),
                    rate_limit_dir, output, teacher_factory=teacher_factory, env_factory=env_factory)
            outcome = {"task_id": task["task_id"], "source_receipt": original,
                "reuse": reuse, "reuse_reason": reason, "effective_sampling": effective_sampling(output),
                "candidates": [first, *candidates]}
            frozen_write(work / "outcome.json", outcome)
            outcomes.append(outcome)
        fallback = output / "sampling_fallback.json"
        pool = {"schema_version": 1, "status": "complete", "identity_sha256": digest(identity),
            "K": source["K"], "K_counts": "trusted_executed_trajectories",
            "unique_instances": len(ids), "task_groups": source["task_groups"],
            "tasks": outcomes, "work_hashes": artifact_hashes(output / "work"),
            "sampling_fallback": read(fallback) if fallback.exists() else None,
            "api_attempts": sum(c["api_attempts"] for o in outcomes for c in o["candidates"]),
            "new_task_instances": 0, "training_exposure": 0}
        frozen_write(output / "candidates.json", pool)
        write_json(output / "result.json", {"status": "complete", "pool": str(output),
            "api_attempts": pool["api_attempts"], "reused_trusted": sum(o["reuse"] for o in outcomes)})
    return output


def load_pool(root, manifest, freeze):
    root = Path(root).resolve()
    identity, pool = read(root / "identity.json"), read(root / "candidates.json")
    source, selection, source_root = load_verified_collection(identity["source_verified"], manifest, freeze)
    if (pool["status"] != "complete" or pool["identity_sha256"] != digest(identity)
            or digest(source) != identity["source_collection_sha256"] or selection != identity["selection"]
            or pool["work_hashes"] != artifact_hashes(root / "work")):
        raise ValueError("SmartAD pool identity/evidence changed")
    fallback = root / "sampling_fallback.json"
    if pool["sampling_fallback"] != (read(fallback) if fallback.exists() else None):
        raise ValueError("SmartAD sampling decision changed")
    for task in pool["tasks"]:
        if len(task["candidates"]) != CANDIDATES:
            raise ValueError("SmartAD requires exactly ten candidate slots")
        for row in task["candidates"]:
            directory = Path(row["episode_dir"])
            if artifact_hashes(directory) != row["episode_hashes"] or correct(directory, read(directory / "episode.json")) != row["correct"]:
                raise ValueError("SmartAD candidate evidence/correctness changed")
    return pool, identity, source, selection, source_root
