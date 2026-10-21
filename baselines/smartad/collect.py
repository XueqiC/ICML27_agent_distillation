"""CPU-only, durable ten-candidate pools. Selection never invokes a teacher."""
import json
from pathlib import Path

from baselines.adftp.collect import usage_summary
from baselines.common.explore import training_tasks
from baselines.smartad.paths import PathResolver
from baselines.common.collection_views import bind_views, collection_identity, check_support, validate_shard
from baselines.smartad.teacher import (CANDIDATES, CALLS_PER_TASK, REQUESTED_SAMPLING,
                                      SmartADPolicy, default_teacher, effective_sampling, read)
from sed.manifest import frozen_write, resolve_task, verify_task_files
from sed.provenance import code_hashes, safe_config, versions, write_provenance
from sed.react import PROMPT_TEMPLATE, PROMPT_VERSION
from sed.runner import _environment, run_episode
from sed.seed_evidence import artifact_hashes, checked_episode, environment_won, load_verified_collection
from sed.seed_selection import is_full_train, is_stratified_nested
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
    # Shard-count changes and overlapping resumptions still share slot ownership.
    with locked(directory / ".slot.lock"):
        return _candidate(config, task, directory, identity, rate_limit_dir, sampling_root,
                          teacher_factory=teacher_factory, env_factory=env_factory)


def _candidate(config, task, directory, identity, rate_limit_dir, sampling_root, *,
               teacher_factory, env_factory):
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "candidate.json"
    if record_path.exists():
        record = read(record_path)
        from teachers.chain_recovery import evidence_path
        if not record["reused"] and not evidence_path(sampling_root, record["episode_dir"]).is_relative_to(directory):
            raise ValueError("SmartAD candidate episode escapes its slot")
        checked_candidate(record, output=sampling_root)
        return record
    canonical = directory
    collection_identity = read(Path(sampling_root) / "identity.json")
    retry = collection_identity.get("chain_recovery", {}).get("retry_directories", {}).get(
        str(directory.relative_to(sampling_root)))
    from teachers.smartad_resume import fresh_directory
    retry = fresh_directory(sampling_root, directory.relative_to(sampling_root)) or retry
    if retry:
        directory = Path(sampling_root) / retry
        directory.mkdir(parents=True, exist_ok=True)
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
    record = {"candidate_id": canonical.name, "task_id": task["task_id"],
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



def checked_candidate(record, *, output=None):
    directory = Path(record["episode_dir"])
    if output is not None:
        from teachers.chain_recovery import evidence_path
        directory = evidence_path(output, directory)
    if (artifact_hashes(directory) != record["episode_hashes"]
            or artifact_hashes(directory.parent / "teacher") != record["teacher_hashes"]):
        raise ValueError("SmartAD candidate evidence changed")
    return record


def finish_task(config, task, original, source_root, work, identity, output,
                rate_limit_dir, *, teacher_factory, env_factory, owns_first):
    """Resolve the dependent first slot once, after all nine fresh slots settle.

    Any shard may decide reuse or publish final receipts. Only the assigned
    shard can execute candidate 1 if needed. The task lock covers the durable
    decision; the slot lock also guards execution across different partitions.
    No pool-wide lock is held while a teacher/environment is running.
    """
    with locked(work / ".task.lock"):
        if (work / "outcome.json").exists():
            outcome = read(work / "outcome.json")
            if outcome["source_receipt"] != original:
                raise ValueError("SmartAD source receipt changed")
            return outcome
        paths = [work / f"candidate-{i:02d}/candidate.json" for i in range(2, CANDIDATES + 1)]
        if not all(path.exists() for path in paths):
            return None
        candidates = [checked_candidate(read(path), output=output) for path in paths]
        decision_path = work / "reuse.json"
        if decision_path.exists():
            decision = read(decision_path)
            if decision["source_receipt"] != original:
                raise ValueError("SmartAD source receipt changed")
        else:
            sampling = effective_sampling(output)
            reuse, reason = reuse_check(source_root, original, config, sampling)
            decision = {"source_receipt": original, "reuse": reuse, "reuse_reason": reason,
                        "effective_sampling": sampling}
            frozen_write(decision_path, decision)
        first_dir = work / "candidate-01"
        if decision["reuse"]:
            with locked(first_dir / ".slot.lock"):
                first = {"candidate_id": "candidate-01", "task_id": task["task_id"], "reused": True,
                    "correct": True, "environment_win": True, "rejection": None,
                    "episode_dir": str((source_root / original["episode_dir"]).resolve()),
                    "episode_hashes": original["episode_hashes"], "teacher_hashes": original["teacher_hashes"],
                    "parent_seed_ids": task["parent_seed_ids"], "api_attempts": 0}
                frozen_write(first_dir / "candidate.json", first)
        else:
            if not owns_first:
                if not (first_dir / "candidate.json").exists():
                    return None
                first = checked_candidate(read(first_dir / "candidate.json"), output=output)
            else:
                first = candidate(config, task, first_dir, identity, rate_limit_dir, output,
                                  teacher_factory=teacher_factory, env_factory=env_factory)
        outcome = {"task_id": task["task_id"], **decision, "candidates": [first, *candidates]}
        frozen_write(work / "outcome.json", outcome)
        return outcome


def publish(output, identity, source):
    """Only immutable, final slots may enter the pool consumed by selection/views."""
    with locked(output / ".pipeline.lock"):
        paths = [output / "work" / digest(r["task_id"])[:20] / "outcome.json"
                 for r in source["verified"]]
        if not all(path.exists() for path in paths):
            result = {"status": "collecting", "pool": str(output),
                      "final_tasks": sum(path.exists() for path in paths), "tasks": len(paths)}
        else:
            # A completed pool is never rewritten by late/resumed shards.
            if (output / "candidates.json").exists():
                pool = read(output / "candidates.json")
                if (pool["identity_sha256"] != digest(identity)
                        or pool["work_hashes"] != artifact_hashes(output / "work")):
                    raise ValueError("SmartAD pool identity/evidence changed")
            else:
                outcomes = [read(path) for path in paths]
                for original, outcome in zip(source["verified"], outcomes):
                    if (outcome["source_receipt"] != original
                            or len(outcome["candidates"]) != CANDIDATES):
                        raise ValueError("SmartAD task is not final")
                    for row in outcome["candidates"]:
                        checked_candidate(row, output=output)
                fallback = output / "sampling_fallback.json"
                pool = {"schema_version": 1, "status": "complete", "identity_sha256": digest(identity),
                    "K": source["K"], "K_counts": "trusted_executed_trajectories",
                    "unique_instances": len(paths), "task_groups": source["task_groups"],
                    "tasks": outcomes, "work_hashes": artifact_hashes(output / "work"),
                    "sampling_fallback": read(fallback) if fallback.exists() else None,
                    "api_attempts": sum(c["api_attempts"] for o in outcomes for c in o["candidates"]),
                    "new_task_instances": 0, "training_exposure": 0}
                if "chain_recovery" in identity:
                    from teachers.chain_recovery import collection_accounting
                    pool.update(collection_accounting(output, identity))
                frozen_write(output / "candidates.json", pool)
            result = {"status": "complete", "pool": str(output), "api_attempts": pool["api_attempts"],
                      "reused_trusted": sum(o["reuse"] for o in pool["tasks"])}
        write_json(output / "result.json", result)
        return result


@teacher_stage("report_output", name="smartad.collect")
def collect_shard(source, manifest, freeze, config, output, identity, parents, source_root, *,
                  shard_index, shard_count, report_output, rate_limit_dir, teacher_factory, env_factory):
    tasks = []
    for original in source["verified"]:
        task = resolve_task(manifest, freeze, original["task_id"])
        task["parent_seed_ids"] = parents[task["task_id"]]
        tasks.append((task, original, output / "work" / digest(task["task_id"])[:20]))
    identity_sha = digest(identity)

    def finish(task_index):
        task, original, work = tasks[task_index]
        return finish_task(config, task, original, source_root, work, identity_sha, output,
            rate_limit_dir, teacher_factory=teacher_factory, env_factory=env_factory,
            owns_first=(task_index * CANDIDATES + CANDIDATES - 1) % shard_count == shard_index)

    # Deterministic flattened order: (task 0, 2..10, 1), (task 1, 2..10, 1), ...
    # Even one task can occupy nine workers. Slot 1 waits for the nine receipts;
    # defer it if they are not ready, without keeping idle/missing shards alive.
    for slot in range(shard_index, len(tasks) * CANDIDATES, shard_count):
        task_index, offset = divmod(slot, CANDIDATES)
        task, original, work = tasks[task_index]
        if offset < CANDIDATES - 1:
            candidate(config, task, work / f"candidate-{offset + 2:02d}", identity_sha,
                      rate_limit_dir, output, teacher_factory=teacher_factory, env_factory=env_factory)
        finish(task_index)
    # Also recover a crash after the last candidate receipt but before the task
    # outcome/pool was published. Empty shards and changed shard counts can help.
    for task_index in range(len(tasks)):
        finish(task_index)
    result = publish(output, identity, source)
    write_json(report_output / "result.json",
               {**result, "shard_index": shard_index, "shard_count": shard_count})
    return result


def collect(verified, manifest, freeze, config, output, *, rate_limit_dir,
            env_factory=None, teacher_factory=default_teacher, input_paths=(), subset_ks=(),
            shard_index=0, shard_count=1, resume=True, subset_views=()):
    validate_shard(shard_index, shard_count)
    output = Path(output).resolve()
    source, selection, source_root = load_verified_collection(verified, manifest, freeze)
    if not source["verified"]:
        raise ValueError("SmartAD requires at least one trusted support trajectory")
    for directory in (output, Path(rate_limit_dir).resolve()):
        if any(directory.is_relative_to(root) for root in
               (source_root, Path(config["paths"]["raw"]).resolve(), Path(config["paths"]["verified"]).resolve())):
            raise ValueError("SmartAD output/limiter must be outside original seed storage")
    if ((is_full_train(selection) or is_stratified_nested(selection)) and subset_ks) or any(k < 1 or k >= selection["K"] for k in subset_ks):
        raise ValueError("Nested views require smaller positive K and a few-shot source")
    bindings = bind_views(verified, source, selection, manifest, freeze, subset_views, subset_ks=subset_ks)
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
    if bindings:
        identity["subset_views"] = bindings
    env_factory = env_factory or (lambda task: _environment(config, task))
    with locked(output / ".pipeline.lock"):
        if not resume and (output / "identity.json").exists():
            raise ValueError("Collection exists; use --resume with the same frozen identity")
        if (output / "identity.json").exists():
            from teachers.smartad_resume import amend, check_resume_identity, registered
            stored = read(output / "identity.json")
            if registered(output, stored):
                check_resume_identity(stored, identity)
                # After COMPLETE this verifies only the frozen amendment; other
                # shards may already have ordinary candidates in progress.
                amend(output, resume=resume)
        identity = collection_identity(output, identity)
        if not (output / "provenance.json").exists():
            write_provenance(output, command="smartad_collect", config=config,
                inputs=input_paths, details=identity)
    # Reports are worker-local: teacher_stage must never race to overwrite the
    # shared result or describe a partially collected pool as complete.
    report_output = output / "shards" / f"{shard_count}-{shard_index}"
    with locked(report_output / ".shard.lock"):
        collect_shard(source, manifest, freeze, config, output, identity, parents, source_root,
            shard_index=shard_index, shard_count=shard_count, report_output=report_output,
            rate_limit_dir=rate_limit_dir, teacher_factory=teacher_factory, env_factory=env_factory)
    if bindings and (output / "candidates.json").exists():
        materialize_pool_views(output, manifest, freeze, bindings)
    return output


def pool_view(pool, binding):
    tasks = [t for t in pool["tasks"] if t["task_id"] in binding["task_ids"]]
    if {t["task_id"] for t in tasks} != set(binding["task_ids"]):
        raise ValueError("Candidate view task_id mismatch")
    return {"schema_version": 1, "status": "complete", "K": binding["K"],
            "K_counts": "trusted_executed_trajectories", "support_view": binding,
            "parent_pool": {"path": "../..", "sha256": digest(pool)}, "tasks": tasks,
            "unique_instances": len(tasks),
            "task_groups": len({t["source_receipt"]["group_id"] for t in tasks}),
            "new_teacher_calls": 0, "new_environment_steps": 0, "training_exposure": 0}


def materialize_pool_views(root, manifest, freeze, bindings):
    pool = read(root / "candidates.json")
    with locked(root / ".views.lock"):
        for binding in bindings:
            check_support(binding, manifest, freeze)
            frozen_write(root / "subsets" / f"k{binding['K']}" / "candidates.json", pool_view(pool, binding))


def load_pool(root, manifest, freeze, *, resolver=None):
    resolver = resolver or PathResolver()
    root = resolver.resolve(Path(root).resolve())
    pool = read(root / "candidates.json")
    if "parent_pool" in pool:
        parent = root / pool["parent_pool"]["path"]
        full, identity, _, _, _ = load_pool(parent, manifest, freeze, resolver=resolver)
        binding = pool["support_view"]
        if binding not in identity.get("subset_views", []) or digest(full) != pool["parent_pool"]["sha256"]:
            raise ValueError("Candidate view parent/support binding changed")
        source, selection, source_root = check_support(binding, manifest, freeze, resolver=resolver)
        if pool != pool_view(full, binding):
            raise ValueError("Candidate view task_id/evidence mismatch")
        return pool, identity, source, selection, source_root
    identity = read(root / "identity.json")
    if "chain_recovery" in identity:
        # Preserve imported absolute receipts byte-for-byte while consuming the
        # hash-verified local copies, including after an explicit HPG relocation.
        old_root = Path(identity["chain_recovery"]["source"])
        pair = (old_root, root)
        if pair not in resolver.evidence_pairs:
            resolver.evidence_pairs.append(pair)
    source, selection, source_root = load_verified_collection(
        resolver.resolve(identity["source_verified"]), manifest, freeze)
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
            directory = resolver.resolve(row["episode_dir"], row["episode_hashes"])
            resolver.resolve(Path(row["episode_dir"]).parent / "teacher", row["teacher_hashes"])
            if correct(directory, read(directory / "episode.json")) != row["correct"]:
                raise ValueError("SmartAD candidate evidence/correctness changed")
    return pool, identity, source, selection, source_root
