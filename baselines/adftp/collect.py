"""Resumable FTP collection, preserving the original verified slot layout."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from baselines.adftp.teacher import FTPPolicy, FTPTeacher, FTP_PROMPT_VERSION
from baselines.common.explore import training_tasks
from baselines.common.collection_views import (bind_support, bind_views, collection_identity,
    materialize_views, support_reference)
from baselines.common.collection_views import validate_shard
from baselines.eto.pipeline import stage_run
from sed.manifest import frozen_write, resolve_task, verify_task_files
from sed.provenance import code_hashes, safe_config, versions, write_provenance
from sed.runner import _environment, run_episode
from sed.seed_selection import is_full_train, is_stratified_nested
from sed.seed_evidence import artifact_hashes, checked_episode, environment_won, load_verified_collection, materialize_subsets
from sed.seed_evidence import support_view_reference
from sed.storage import digest, locked, write_json, fsync_dir
from teachers.capped import CappedTeacher, CollectorHalted
from teachers.baseline_transport import baseline_transport
from teachers.openai_teacher import TeacherCallError, rate_limit_options
from teachers.stage import teacher_stage, execution_attempt, mark_transport_failure, require_execution, usage_counts


def read(path):
    return json.loads(Path(path).read_text())



def usage_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    # state_link is accounting linkage, never an additional billable request.
    finished = [r for r in rows if r["status"] in {"success", "rejected", "failed", "incomplete", "cache_hit"}]
    ended = {r["call_id"] for r in finished}
    interrupted = [r for r in rows if r["status"] == "started" and r["call_id"] not in ended]
    return {**usage_counts(rows), "known_billed_tokens": {key: sum((r.get("usage") or {}).get(key, 0) or 0 for r in finished)
            for key in ("input_tokens", "output_tokens", "reasoning_tokens")},
        "unknown_usage_calls": sum(r.get("usage") is None for r in finished) + len(interrupted),
        "interrupted_requests": len(interrupted), "ledger_rows": len(rows)}

def default_teacher(config, directory, task, identity, rate_limit_dir):
    spec = config["models"]["teacher"]
    return CappedTeacher(counter_path=directory / "counter.json", cap=123,
        cache_dir=directory / "cache", usage_ledger=directory / "usage.jsonl",
        run_id=digest([identity, task["task_id"]]), teacher_class=FTPTeacher,
        provider_model_id=spec["provider_model_id"],
        sampling={**spec["sampling"], "max_output_tokens": 1024},
        baseline_method="adftp", secrets_file=config["paths"]["secrets_file"],
        **rate_limit_options(spec, rate_limit_dir))


def collect_task(config, task, work, source_root, source_receipt, identity, rate_limit_dir,
                 *, env_factory, teacher_factory):
    """Work survives stage retries. An unfinished episode consumes its slot."""
    work.mkdir(parents=True, exist_ok=True)
    teacher = teacher_factory(config, work / "teacher", task, identity, rate_limit_dir)
    if teacher._read_counter()["halted"]:
        raise CollectorHalted("FTP task has unknown/failed transport usage; inspect its durable ledger")
    accepted = None
    attempts = []
    for number in range(1, 4):
        directory = execution_attempt(work / f"attempt-{number}", start="started.json")
        if not (directory / "started.json").exists():
            frozen_write(directory / "started.json", {"attempt": number, "parent_seed_ids": task["parent_seed_ids"]})
            # Fresh cache namespace per attempt; resume never resamples a started episode.
            teacher.teacher.cache_dir = directory / "cache"
            policy = FTPPolicy(teacher, task, directory)
            verify_task_files(config["environment"]["data_root"], task)
            try:
                run_episode(env_factory(task), policy, task, config, directory / "episode",
                    policy_condition="adftp_teacher_h0", purpose="generation",
                    parent_seed_ids=task["parent_seed_ids"])
            except TeacherCallError:
                mark_transport_failure(directory)
                raise
            except RuntimeError:
                if not (directory / "episode/episode.json").exists():
                    raise
            if teacher._read_counter()["halted"]:
                raise CollectorHalted("FTP transport halted; inspect the durable ledger")
            write_json(directory / "rejection.json", {"reason": policy.rejection})
        episode_file = directory / "episode/episode.json"
        episode = read(episode_file) if episode_file.exists() else {}
        require_execution(episode)
        won = environment_won(directory / "episode", episode)
        rejection = read(directory / "rejection.json") if (directory / "rejection.json").exists() else {}
        attempts.append({"attempt": number, "won": won, "success": episode.get("success", "unknown"),
            "status": episode.get("status", "interrupted"), "metrics": episode.get("metrics"),
            "rejection": None if won else rejection.get("reason") or "no_verified_environment_win",
            "episode_dir": str((directory / "episode").relative_to(work))})
        if won:
            accepted = directory / "episode"
            break
    fallback = accepted is None
    if fallback:
        checked_episode(source_root, source_receipt)
        accepted = Path(source_root) / source_receipt["episode_dir"]
    outcome = {"attempts": attempts, "fallback": fallback, "selected_episode": str(accepted.resolve()),
        "parent_seed_ids": task["parent_seed_ids"], "source_receipt": source_receipt,
        "usage": usage_summary(work / "teacher/usage.jsonl"),
        "api_attempts": teacher._read_counter()["attempts"], "cap": 123}
    write_json(work / "outcome.json", outcome)
    return outcome


def collect(verified, manifest, freeze, config, output, *, rate_limit_dir,
            env_factory=None, teacher_factory=default_teacher, input_paths=(), subset_ks=(),
            subset_views=(), shard_index=0, shard_count=1):
    validate_shard(shard_index, shard_count)
    output = Path(output).resolve()
    for name in ("raw", "verified"):
        if output.is_relative_to(Path(config["paths"][name]).resolve()):
            raise ValueError("FTP output must be outside original seed storage")
    source, selection, source_root = load_verified_collection(verified, manifest, freeze)
    if ((is_full_train(selection) or is_stratified_nested(selection)) and subset_ks) or any(k < 1 or k >= selection["K"] for k in subset_ks):
        raise ValueError("Nested views require smaller positive K and a few-shot source")
    bindings = bind_views(verified, source, selection, manifest, freeze, subset_views, subset_ks=subset_ks)
    for directory in (output, Path(rate_limit_dir).resolve()):
        if (directory.is_relative_to(source_root)
                or directory.is_relative_to(Path(config["paths"]["verified"]).resolve())):
            raise ValueError("FTP output/limiter must not modify source verified storage")
    if (config["environment"]["max_episode_steps"] != 40 or config["training"]["thinking_mode"] != "off"
            or config["training"]["student_step_token_cap"] != 1024):
        raise ValueError("AD-FTP requires H0: 40 turns, thinking off, 1024 output tokens")
    config = deepcopy(config)
    env_factory = env_factory or (lambda task: _environment(config, task))
    parents = {r["task_id"]: [checked_episode(source_root, r)["episode_id"]] for r in source["verified"]}
    ids = list(parents)
    training_tasks({"task_ids": ids, "parent_seed_ids": parents, "manifest_sha256": digest(manifest),
                    "split_freeze_sha256": digest(freeze), "list_sha256": digest(ids)}, manifest, freeze)
    identity = {"selection": selection, "method": "adftp", "prompt_version": FTP_PROMPT_VERSION,
        **support_view_reference(verified, output / "collection", source, selection),
        "baseline_transport": baseline_transport("adftp").disclosure(),
        "source_verified": str(Path(verified).resolve()), "source_collection_sha256": digest(source),
        "config": safe_config(config), "code_sha256": code_hashes(Path(__file__).resolve().parents[2]),
        "dependencies": versions(), "rate_limit_dir": str(Path(rate_limit_dir).resolve()),
        "attempts_per_candidate": 3, "calls_per_task_cap": 123, "per_cell_call_cap": 123 * len(ids)}
    if bindings:
        identity["subset_views"] = bindings
    if selection.get("selection_rule") == "stratified_nested_v2":
        identity["support_view"] = support_reference(bind_support(verified, source, selection), output / "collection")
    with locked(output / ".pipeline.lock"):
        identity = collection_identity(output, identity)
        if not (output / "provenance.json").exists():
            write_provenance(output, command="adftp_collect", config=config, inputs=input_paths,
                             details=identity)
    report_output = output / "shards" / f"{shard_count}-{shard_index}"
    _collect_shard(source, selection, source_root, manifest, freeze, config, output, identity, parents,
        rate_limit_dir=rate_limit_dir, env_factory=env_factory, teacher_factory=teacher_factory,
        subset_ks=subset_ks, bindings=bindings, shard_index=shard_index, shard_count=shard_count,
        report_output=report_output)
    return output / "collection"


@teacher_stage("report_output", name="adftp.collect")
def _collect_shard(source, selection, source_root, manifest, freeze, config, output, identity, parents, *,
                   rate_limit_dir, env_factory, teacher_factory, subset_ks, bindings,
                   shard_index, shard_count, report_output):
    for index, original in enumerate(source["verified"]):
        if index % shard_count != shard_index:
            continue
        task = resolve_task(manifest, freeze, original["task_id"])
        task["parent_seed_ids"] = parents[task["task_id"]]
        key = digest(task["task_id"])[:20]
        work = output / "work" / key
        with locked(output / "task_locks" / (key + ".lock")):
            def action(attempt):
                outcome = collect_task(config, task, work, source_root, original, digest(identity),
                    rate_limit_dir, env_factory=env_factory, teacher_factory=teacher_factory)
                published = attempt / "output"
                shutil.copytree(outcome["selected_episode"], published / "episode")
                # All failed and interrupted FTP evidence is checked by the unchanged loader.
                shutil.copytree(work, published / "teacher")
                if outcome["fallback"]:
                    shutil.copytree(Path(source_root) / original["episode_dir"].rsplit("/", 1)[0] / "teacher",
                                    published / "teacher/trusted_source_teacher")
                write_json(published / "outcome.json", outcome)
            staged = stage_run(output, "collect-" + key, digest([identity, original]), action)
            # Canonical receipts represent one selected verified trajectory per slot.
            # Raw generation attempts remain nested in their hashed teacher evidence.
            destination = output / "collection/attempts" / (key + "-1")
            if not destination.exists():
                # Keep interrupted copies as evidence, then publish a complete sibling atomically.
                destination.parent.mkdir(parents=True, exist_ok=True)
                import tempfile
                partial = Path(tempfile.mkdtemp(prefix="." + key + "-partial-", dir=destination.parent))
                shutil.copytree(staged, partial, dirs_exist_ok=True)
                os.rename(partial, destination)
                fsync_dir(destination.parent)
            elif artifact_hashes(destination) != {**artifact_hashes(staged),
                    **({"receipt.json": artifact_hashes(destination)["receipt.json"]}
                       if (destination / "receipt.json").exists() else {})}:
                raise ValueError("Published FTP evidence changed")
            outcome = read(staged / "outcome.json")
            episode = read(destination / "episode/episode.json")
            receipt = {"task_id": task["task_id"], "group_id": task["group_id"], "task_type": task["task_type"],
                "attempt": 1, "selection_parent_task_id": original["selection_parent_task_id"],
                "parent_task_id": task["task_id"], "parent_seed_ids": task["parent_seed_ids"],
                "source": "adftp_trusted_fallback" if outcome["fallback"] else "adftp_verified_teacher",
                "won": True, "success": True, "status": "ok", "rejection": None,
                "episode_dir": f"attempts/{key}-1/episode", "episode_hashes": artifact_hashes(destination / "episode"),
                "teacher_hashes": artifact_hashes(destination / "teacher"), "usage": outcome["usage"],
                "metrics": episode["metrics"], "ftp_attempts": outcome["attempts"],
                "fallback": outcome["fallback"], "api_attempts": outcome["api_attempts"]}
            frozen_write(destination / "receipt.json", receipt)
    with locked(output / ".pipeline.lock"):
        root = output / "collection"
        paths = [root / "attempts" / (digest(r["task_id"])[:20] + "-1") / "receipt.json" for r in source["verified"]]
        if not all(path.exists() for path in paths):
            result = {"status": "collecting", "final_tasks": sum(path.exists() for path in paths), "tasks": len(paths)}
            write_json(output / "result.json", result)
            return result
        receipts = [read(path) for path in paths]
        ids = [r["task_id"] for r in source["verified"]]
        all_receipts = receipts
        if is_full_train(selection):
            # A full source records unsuccessful seed candidates too. Never generate
            # on them: they lack a trusted support demonstration. Retain their exact
            # execution evidence so the shared full-TRAIN loader can audit coverage.
            by_task = {r["task_id"]: r for r in receipts}
            all_receipts = []
            for original in source["attempts"]:
                if original["task_id"] in by_task:
                    if original["won"]:
                        all_receipts.append(by_task[original["task_id"]])
                    continue
                origin = Path(source_root) / original["episode_dir"]
                destination = root / original["episode_dir"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                for name in ("episode", "teacher"):
                    expected = original[name + "_hashes"]
                    target = destination.parent / name
                    if not target.exists() and (origin.parent / name).is_dir():
                        import tempfile
                        partial = Path(tempfile.mkdtemp(prefix=".inherited-partial-", dir=target.parent))
                        shutil.copytree(origin.parent / name, partial, dirs_exist_ok=True)
                        os.rename(partial, target)
                        fsync_dir(target.parent)
                    if artifact_hashes(target) != expected:
                        raise ValueError("Reused failed candidate evidence changed")
                frozen_write(destination.parent / "receipt.json", original)
                all_receipts.append(original)
        frozen_write(root / "identity.json", identity)
        frozen_write(root / "selection.json", selection)
        if not (root / "provenance.json").exists():
            shutil.copyfile(output / "provenance.json", root / "provenance.json")
        collection = {"schema_version": 1, "status": "complete", "K": source["K"],
            "K_counts": "trusted_executed_trajectories", "identity_sha256": digest(identity),
            "selection_sha256": digest(selection), "trusted_trajectories": len(receipts),
            "unique_instances": len(ids), "task_groups": len({r["group_id"] for r in receipts}),
            "attempted_unique_instances": len({r["task_id"] for r in all_receipts}),
            "attempted_task_groups": len({r["group_id"] for r in all_receipts}),
            "ftp_attempted_unique_instances": len(ids),
            "attempts": all_receipts, "verified": receipts, "parent_seed_ids": [p for v in parents.values() for p in v],
            "source": "adftp_support_tasks_only", "training_exposure": 0,
            "baseline_transport": identity["baseline_transport"],
            "fallback_count": sum(r["fallback"] for r in receipts),
            "ftp_attempt_count": sum(len(r["ftp_attempts"]) for r in receipts),
            "api_attempts": sum(r["api_attempts"] for r in receipts), "per_cell_call_cap": 123 * len(ids),
            "new_environment_steps": sum((a.get("metrics") or {}).get("environment_steps", 0)
                                         for r in receipts for a in r["ftp_attempts"]),
            "unknown_usage_calls": sum(r["usage"]["unknown_usage_calls"] for r in receipts),
            "transport_failures": sum(r["usage"]["transport_failures"] for r in receipts),
            "parser_rejections": sum(r["usage"]["rejections"] for r in receipts),
            "rejections": sum(a["rejection"] in {"empty_first_paragraph", "first_reply_prefix_mismatch"}
                              for r in receipts for a in r["ftp_attempts"]),
            "known_billed_tokens": {key: sum(r["usage"]["known_billed_tokens"][key] for r in receipts)
                                    for key in ("input_tokens", "output_tokens", "reasoning_tokens")},
            "new_verified_trajectories": sum(not r["fallback"] for r in receipts),
            "new_task_instances": 0,
            "accounting": "Canonical attempt=1 is a publication slot, not a new execution. All FTP attempts, "
                "including failures, are in each receipt's hashed teacher bundle. Fallback copies trusted execution; "
                "source costs are reused, never new calls or independent tasks."}
        if is_full_train(selection):
            collection.update(selection_rule=selection["selection_rule"], candidates=selection["candidates"],
                failed_candidates=source["failed_candidates"],
                source_reused_failed_attempts=sum(not r["won"] for r in all_receipts))
        frozen_write(root / "manifest.json", collection)
        load_verified_collection(root, manifest, freeze)
        if subset_ks:
            materialize_subsets(root, manifest, freeze, subset_ks)
        write_json(output / "result.json", {"status": "complete", "verified": str(root),
            "fallback_count": collection["fallback_count"], "api_attempts": collection["api_attempts"]})
        if bindings:
            materialize_views(root, manifest, freeze, bindings)
        return {"status": "complete", "verified": str(root)}
