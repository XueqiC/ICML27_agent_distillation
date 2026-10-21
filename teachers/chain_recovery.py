"""Offline quota-incident recovery. Source evidence is never locked or written.

Small copy/inventory primitives are shared with full_train_recovery, the accepted
full-TRAIN migration precedent. Only this registered incident permits cap credits.

The bounded smartad_interrupted_sample_20261021_v1 amendment in smartad_resume
archives interrupted samples inside the existing registered r2 cell on resume,
then permits one fresh 41-call sample per eligible slot (at most 23 / 943).
It never re-credits quota calls or permits r2 to be used as a recovery source.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from sed.provenance import code_hashes, safe_config, versions, write_provenance
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import BlobStore, digest, file_hash, fsync_dir, locked, write_json
from teachers.full_train_recovery import _copy_checked, _under, read_json, separate_paths, tree_hashes

RULE = "chain_quota_void_20261020_v1"
EXHAUSTED_AT = "2026-10-20T22:45:00+00:00"
TOKENS = ("input_tokens", "output_tokens", "reasoning_tokens")


def credit_file(output, relative):
    return Path(output) / "recovery/credits" / (digest(relative) + ".json")


def extension_file(output, relative):
    return Path(output) / "recovery/extensions" / (digest(relative) + ".json")


def call_budgets(method):
    if method == "adftp":
        from baselines.adftp.teacher import CALLS_PER_ATTEMPT, CALLS_PER_TASK
        return CALLS_PER_ATTEMPT, CALLS_PER_TASK
    if method == "smartad":
        from baselines.smartad.teacher import CALLS_PER_CANDIDATE
        return CALLS_PER_CANDIDATE, CALLS_PER_CANDIDATE
    raise ValueError("Unknown chain recovery method")


def execution_tip(base):
    """Read-only counterpart of execution_attempt; retain superseded failures."""
    retry_root = base.parent / "transport_retries" / base.name
    retry_numbers = sorted(int(p.name.removeprefix("attempt-")) for p in retry_root.glob("attempt-*")
                           if p.is_dir())
    if (retry_numbers != list(range(1, len(retry_numbers) + 1)) or
            any(not (retry_root / f"attempt-{n}/started.json").exists() for n in retry_numbers)):
        raise ValueError("Incomplete execution retry sequence")
    current, index = base, 0
    while (current / "started.json").exists():
        next_directory = base.parent / "transport_retries" / base.name / f"attempt-{index + 1}"
        if not (next_directory / "started.json").exists():
            return current, next_directory
        episode = read_json(current / "episode/episode.json")
        if not episode.get("teacher_transport_failure") or episode.get("complete"):
            raise ValueError("Retry follows a complete/non-transport execution")
        current, index = next_directory, index + 1
    raise ValueError("Missing execution start")


def terminal_failure(directory, task_id):
    """Read the last event's content-addressed audit, never a guessed error."""
    episode = read_json(directory / "episode/episode.json")
    events = ledger_rows(directory / "episode/events.jsonl")
    if not events:
        raise ValueError("Missing execution events")
    event = events[-1]
    if (episode["task_id"] != task_id or event["task_id"] != task_id
            or event["episode_id"] != episode["episode_id"]):
        raise ValueError("Execution event identity mismatch")
    verification = BlobStore(directory / "episode/blobs").get(event["verification_ref"])
    return episode, event, verification.get("audit", {}).get("error_type")


def cap_extension(source, work, record, method, files):
    """Second record of RULE: one fresh execution, with no refund of spent usage.

    Earlier AD-FTP method slots must have completed effective executions. Their
    superseded quota-interrupted executions remain inherited evidence.
    """
    if method == "adftp":
        from baselines.adftp.teacher import ATTEMPTS_PER_CANDIDATE
        bases = [work / f"attempt-{n}" for n in range(1, ATTEMPTS_PER_CANDIDATE + 1)
                 if (work / f"attempt-{n}/started.json").exists()]
        if not bases:
            return None
        tips = [execution_tip(base) for base in bases]
        method_attempt = int(bases[-1].name.removeprefix("attempt-"))
        slot = work
    else:
        slot = source / Path(record["source_counter"]).parent.parent
        if not (slot / "started.json").exists():
            return None
        tips = [execution_tip(slot)]
        method_attempt = 1
    directory, fresh = tips[-1]
    if method == "smartad":
        fresh = slot / "transport_retries/attempt-1"
    episode, event, error = terminal_failure(directory, record["task_id"])
    earlier = [read_json(tip / "episode/episode.json") for tip, _ in tips[:-1]]
    if any(terminal_failure(tip, record["task_id"])[2] == "BudgetExceeded" for tip, _ in tips[:-1]):
        raise ValueError("Cap extension requires BudgetExceeded in the latest attempt")
    if error != "BudgetExceeded":
        return None
    budget, _ = call_budgets(method)
    if (record["final"] or (work / "outcome.json").exists()
            or any((directory / name).exists() for name in ("outcome.json", "candidate.json", "receipt.json"))
            or (slot / "candidate.json").exists()
            or not 0 < record["credit"] <= budget
            or record["old_attempts"] != record["cap"]
            or episode.get("complete") is not False or episode.get("success") != "unknown"
            or not episode.get("teacher_transport_failure")
            or event.get("status") != "error" or event.get("error_type") != "policy_error"
            or any(e.get("complete") is not True or e.get("teacher_transport_failure") for e in earlier)
            or (method == "adftp" and len(bases) != method_attempt)):
        raise ValueError("Ineligible cap extension: quota credit, exhausted cap, no outcome and complete earlier attempts required")
    relative = str(directory.relative_to(source))
    evidence = _under(files, relative)
    completed = {str(tip.relative_to(source)): digest(_under(files, str(tip.relative_to(source))))
                 for tip, _ in tips[:-1]}
    calls = {row["call_id"] for row in episode.get("policy_calls", []) if row.get("call_id")}
    cot = directory / "cot.json"
    if cot.exists():
        calls.add(read_json(cot)["metadata"]["call_id"])
    return {"rule": RULE, "record_type": "cap_extension", "method": method, "task_id": record["task_id"],
            "candidate_id": slot.name if method == "smartad" else None,
            "attempt_id": episode["episode_id"], "attempt_directory": relative,
            "method_attempt": method_attempt, "fresh_attempt_directory": str(fresh.relative_to(source)),
            "source_counter": record["source_counter"], "target_counter": record["target_counter"],
            "calls": record["old_attempts"], "attempt_calls": len(calls), "attempt_call_ids": sorted(calls),
            "credited_calls": record["credit"], "void_calls": record["void_calls"],
            "per_attempt_call_budget": budget, "extension": budget - record["credit"],
            "original_cap": record["cap"], "effective_cap": record["cap"] + budget - record["credit"],
            "remaining_cap": budget, "source_counter_sha256": record["source_counter_sha256"],
            "ledger_prefix_sha256": record["ledger_prefix_sha256"],
            "attempt_evidence_sha256": digest(evidence), "episode_sha256": files[relative + "/episode/episode.json"],
            "terminal_event_sha256": digest(event), "terminal_verification_ref": event["verification_ref"],
            "earlier_complete_attempts": completed}


def seed_counter(credit):
    return {"identity": credit["source_identity"], "attempts": credit["initial_attempts"],
            "halted": False, "pending": False, "chain_credit_sha256": digest(credit),
            "migration_pending": True}


def copy_layout(files, records):
    mutable = {r["source_counter"] for r in records.values() if not r["final"] and
               r["source_counter"] == r["target_counter"]}
    mutable |= {str(Path(p).with_name("usage.jsonl")) for p in mutable}
    destinations, immutable = {}, {}
    for name, sha in files.items():
        active = name.startswith(("work/", "collection/attempts/")) or name == "sampling_fallback.json"
        if name in mutable and name.endswith("/counter.json"):
            active = False  # Archive the original; seed a separately credited counter.
        destination = name if active else "recovery/source/" + name
        destinations[name] = destination
        if name not in mutable or not active:
            immutable[destination] = sha
    return destinations, immutable


def ledger_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def last_calls(rows):
    calls = {}
    for row in rows:
        if row.get("status") != "state_link":
            calls[row["call_id"]] = row
    return calls


def quota_void(row):
    """Legacy ledgers omit provider error bodies; bind the registered incident.

    Non-retryable 429 RateLimitError is the sanitized insufficient-quota signal
    in the frozen baseline transport. Dates alone never qualify a call.
    """
    eligible = (row.get("status") == "failed" and row.get("http_status") == 429
                and row.get("error_type") == "RateLimitError"
                and row.get("retryable") is False and row.get("will_retry") is False
                and "usage" in row and row["usage"] is None
                and not any(row.get(k) for k in ("raw_ref", "response_ref", "output_ref", "raw_text", "output_text")))
    if not eligible:
        return False
    # `timestamp` is captured BEFORE waiting for the shared rate limiter. Its
    # reservation_timestamp is the actual transport admission time (UTC epoch).
    # In the incident some calls queued at 22:43 and were admitted after 22:45.
    admitted = (datetime.fromtimestamp(row["reservation_timestamp"], timezone.utc)
                if row.get("reservation_timestamp") is not None
                else datetime.fromisoformat(row["timestamp"])) if row.get("timestamp") else None
    return admitted is not None and admitted >= datetime.fromisoformat(EXHAUSTED_AT)


def usage(rows):
    calls = last_calls(rows)
    return {"http_calls": len(calls),
            "logical_calls": len({r.get("logical_call_id", r["call_id"]) for r in rows if r["status"] == "started"}),
            "status_counts": dict(Counter(r["status"] for r in calls.values())),
            "known_billed_tokens": {k: sum((r.get("usage") or {}).get(k, 0) or 0 for r in calls.values()) for k in TOKENS},
            "unknown_usage_calls": sum(r.get("usage") is None and not quota_void(r) for r in calls.values()),
            "quota_void_calls": sum(quota_void(r) for r in calls.values())}


def _audit_counter(source, relative, task_id, method, final):
    path = source / relative
    counter = read_json(path)
    ledger = path.with_name("usage.jsonl")
    rows = ledger_rows(ledger)
    identity = counter["identity"]
    if (Path(identity["usage_ledger"]) != ledger or counter["pending"]
            or identity["cap"] != call_budgets(method)[1]
            or any(r.get("run_id") != identity["run_id"] or r.get("task_id", task_id) != task_id for r in rows)):
        raise ValueError("Source counter/ledger identity mismatch or pending request")
    prior = counter.get("recovery")
    if prior and digest(rows[:prior["through_rows"]]) != prior["ledger_prefix_sha256"]:
        raise ValueError("Source acknowledged ledger prefix hash mismatch")
    calls = last_calls(rows)
    starts = {r["call_id"] for r in rows if r["status"] == "started"}
    logical = {r.get("logical_call_id", r["call_id"]) for r in rows if r["status"] == "started"}
    if counter["attempts"] != len(logical) or not starts <= calls.keys():
        raise ValueError("Source counter reservations disagree with ledger")
    if any(cid not in starts and row["status"] != "cache_hit" for cid, row in calls.items()):
        raise ValueError("Source output/failure lacks a reserved start")
    completed = {r.get("logical_call_id", r["call_id"]) for r in calls.values()
                 if r["status"] in {"success", "rejected"}}
    void = []
    for row in calls.values():
        if row["status"] in {"success", "rejected", "cache_hit"}:
            continue
        if quota_void(row):
            chain = [r for r in calls.values() if r.get("logical_call_id", r["call_id"]) ==
                     row.get("logical_call_id", row["call_id"])]
            if len(chain) != 1 or row["call_id"] not in starts:
                raise ValueError("Quota credit must identify exactly one reserved logical call")
            void.append({"call_id": row["call_id"], "logical_call_id": row.get("logical_call_id", row["call_id"]),
                         "failure_sha256": digest(row), "timestamp": row["timestamp"],
                         "reservation_timestamp": row.get("reservation_timestamp")})
            continue
        recovered = (row["status"] == "failed" and row.get("retryable") is True
                     and row.get("will_retry") is True
                     and row.get("logical_call_id", row["call_id"]) in completed)
        sampling = (method == "smartad" and row["status"] == "failed" and row.get("http_status") == 400
                    and row.get("unsupported_sampling_parameter") == "temperature" and prior
                    and row in rows[:prior["through_rows"]])
        if not recovered and not sampling:
            raise ValueError("Non-quota failure or unknown request; refusing automatic recovery")
    if counter["halted"] and not void:
        raise ValueError("Halted counter without a registered quota call")
    return {"rule": RULE, "record_type": "cap_credit", "task_id": task_id, "source_counter": relative,
            "source_counter_sha256": file_hash(path), "source_identity": identity,
            "through_rows": len(rows), "ledger_prefix_sha256": digest(rows),
            "old_attempts": counter["attempts"], "void_calls": void,
            "credit": len(void), "initial_attempts": counter["attempts"] - len(void),
            "cap": identity["cap"], "final": final, "inherited_usage": usage(rows)}, rows


def inspect_source(source, method, manifest, freeze, config, *, progress=lambda _: None):
    from baselines.common.collection_views import bind_support, check_support
    source = Path(source).resolve()
    files = tree_hashes(source)
    progress(f"Hashed {len(files)} source files; validating frozen support")
    old = read_json(source / "identity.json")
    if old.get("method") != method or "chain_recovery" in old:
        raise ValueError("Wrong method or already recovered source; repeated credits are forbidden")
    if old["config"] != safe_config(config):
        raise ValueError("Recovery cannot change the source config, teacher, student or protocol")
    provenance = read_json(source / "provenance.json")
    if provenance["details"] != old:
        raise ValueError("Source identity/provenance hash binding mismatch")
    support, selection, support_root = load_verified_collection(old["source_verified"], manifest, freeze)
    if old["source_collection_sha256"] != digest(support) or old["selection"] != selection:
        raise ValueError("Source support/selection binding mismatch")
    for binding in old.get("subset_views", []):
        path = Path(binding["path"])
        small_collection = read_json(path / "manifest.json")
        # The large view above already validated this entire backing collection.
        # Reconstruct sibling support views with the same canonical publisher,
        # avoiding three more full-TRAIN traversals in a K500 recovery audit.
        if "source_collection" in small_collection and path.parent.parent.resolve() == support_root:
            from sed.seed_evidence import _subset_values
            small = read_json(path / "selection.json")
            expected_identity, expected = _subset_values(support_root, read_json(support_root / "manifest.json"),
                                                         read_json(support_root / "selection.json"), small)
            if (small_collection != expected or read_json(path / "identity.json") != expected_identity
                    or bind_support(path, small_collection, small) != binding):
                raise ValueError("Source support view binding changed")
        else:
            check_support(binding, manifest, freeze)
    progress("Frozen support validated; auditing canonical task/candidate ledgers")
    finals, candidates, records, tasks, calls = [], [], {}, [], {}
    retries, extensions = {}, {}
    for position, original in enumerate(support["verified"]):
        tid = original["task_id"]
        key = digest(tid)[:20]
        work = source / "work" / key
        if method == "adftp":
            receipt_path = source / "collection/attempts" / (key + "-1/receipt.json")
            final = receipt_path.exists()
            if final:
                receipt = read_json(receipt_path)
                if receipt["task_id"] != tid:
                    raise ValueError("Source receipt task mismatch")
                checked_episode(source / "collection", receipt)
                if _under(files, str(receipt_path.parent.relative_to(source)) + "/teacher") != receipt["teacher_hashes"]:
                    raise ValueError("Source receipt teacher evidence hash mismatch")
                stage = source / "stages" / ("collect-" + key)
                completion = read_json(stage / "complete.json")
                if (completion["identity"] != digest([old, original]) or
                        _under(files, str((stage / completion["attempt"]).relative_to(source))) != completion["artifacts"]):
                    raise ValueError("Source completed stage hash mismatch")
                published_work = _under(completion["artifacts"], "output/teacher")
                published_work = {p: sha for p, sha in published_work.items()
                                  if not p.startswith("trusted_source_teacher/")}
                if _under(files, str(work.relative_to(source))) != published_work:
                    raise ValueError("Source final work differs from its completed stage")
            counters = [work / "teacher/counter.json"]
        else:
            final = (work / "outcome.json").exists()
            counters = []
            from baselines.smartad.teacher import CANDIDATES
            for number in range(1, CANDIDATES + 1):
                slot = work / f"candidate-{number:02d}"
                record_path = slot / "candidate.json"
                if record_path.exists():
                    record = read_json(record_path)
                    if record["task_id"] != tid or record["candidate_id"] != slot.name:
                        raise ValueError("Source candidate identity mismatch")
                    from baselines.smartad.collect import checked_candidate, correct
                    checked_candidate(record)
                    episode_dir = Path(record["episode_dir"])
                    if correct(episode_dir, read_json(episode_dir / "episode.json")) != record["correct"]:
                        raise ValueError("Source candidate correctness disagrees with real execution")
                    if record["reused"] and (number != 1 or
                            Path(record["episode_dir"]) != support_root / original["episode_dir"] or
                            any(record[k] != original[k] for k in ("episode_hashes", "teacher_hashes"))):
                        raise ValueError("Source reused candidate differs from its trusted seed")
                    if not record["reused"] and Path(record["episode_dir"]) != slot / "episode":
                        raise ValueError("Source candidate path mismatch")
                    candidates.append(str(slot.relative_to(source)))
                elif (slot / "started.json").exists():
                    episode = read_json(slot / "episode/episode.json")
                    if not episode.get("teacher_transport_failure"):
                        raise ValueError("Unfinalized SmartAD candidate without quota transport evidence")
                    retries[str(slot.relative_to(source))] = str(slot.relative_to(source) / "transport_retries/attempt-1")
                counters.append(slot / "teacher/counter.json")
            if final:
                outcome = read_json(work / "outcome.json")
                expected = [read_json(work / f"candidate-{i:02d}/candidate.json") for i in range(1, CANDIDATES + 1)]
                if outcome["source_receipt"] != original or outcome["candidates"] != expected:
                    raise ValueError("Source outcome/candidate mismatch")
        if final:
            finals.append(key)
        task_records = []
        for counter in counters:
            if not counter.exists():
                continue
            relative = str(counter.relative_to(source))
            slot_final = final if method == "adftp" else (counter.parent.parent / "candidate.json").exists()
            record, rows = _audit_counter(source, relative, tid, method, slot_final)
            if method == "smartad" and str(counter.parent.parent.relative_to(source)) in retries:
                if not record["void_calls"]:
                    raise ValueError("Interrupted candidate has no registered quota call")
                target = retries[str(counter.parent.parent.relative_to(source))] + "/teacher/counter.json"
            else:
                target = relative
            record["target_counter"] = target
            extension = cap_extension(source, work, record, method, files)
            if extension:
                extensions[target] = extension
                record["cap_extension_sha256"] = digest(extension)
            records[target] = record
            task_records.append(record)
            for cid, row in last_calls(rows).items():
                if cid in calls:
                    raise ValueError("Duplicate canonical call_id across tasks/candidates")
                calls[cid] = row
        if not final and (work / "started.json").exists() and not task_records:
            raise ValueError("Unaccounted interrupted task")
        spent = sum(r["old_attempts"] for r in task_records)
        credit = sum(r["credit"] for r in task_records)
        task_extensions = [extensions[r["target_counter"]] for r in task_records if r["target_counter"] in extensions]
        extension_calls = sum(e["extension"] for e in task_extensions)
        def remaining(record):
            return record["cap"] - record["initial_attempts"] + extensions.get(record["target_counter"], {}).get("extension", 0)
        for episode_path in work.rglob("episode/episode.json"):
            episode = read_json(episode_path)
            rejection = episode_path.parent.parent / "rejection.json"
            format_rejection = (method == "adftp" and rejection.exists() and
                read_json(rejection).get("reason") in {"empty_first_paragraph", "first_reply_prefix_mismatch"}
                and not episode.get("teacher_transport_failure"))
            if not format_rejection and (episode.get("status") == "started" or (not episode.get("complete") and
                    (not credit or not episode.get("teacher_transport_failure")))):
                raise ValueError("Unknown/non-quota interrupted execution")
        tasks.append({"task_id": tid, "work_key": key, "chain_position": position,
                      "final": final, "old_attempts": spent, "void_calls": credit,
                      "cap_extensions": task_extensions, "extension_calls": extension_calls,
                      "remaining_cap": old["calls_per_task_cap"] - spent + credit + extension_calls,
                      "fresh_attempt_cap_warning": any(not r["final"] and
                          remaining(r) < call_budgets(method)[0] for r in task_records),
                      "counters": [{"path": r["target_counter"], "final": r["final"],
                                    "remaining_cap": remaining(r),
                                    "extension": extensions.get(r["target_counter"], {}).get("extension", 0),
                                    "void_call_ids": [v["call_id"] for v in r["void_calls"]]} for r in task_records]})
    progress = "collection/manifest.json" if method == "adftp" else "candidates.json"
    progress = progress if progress in files else "result.json"
    inherited = {"http_calls": len(calls), "logical_calls": sum(r["old_attempts"] for r in records.values()),
                 "known_billed_tokens": {k: sum((r.get("usage") or {}).get(k, 0) or 0 for r in calls.values()) for k in TOKENS},
                 "quota_void_calls": sum(r["credit"] for r in records.values()),
                 "unknown_usage_calls": sum(r.get("usage") is None and not quota_void(r) for r in calls.values())}
    inherited["environment_steps"] = sum((read_json(source / p).get("metrics") or {}).get("environment_steps", 0)
        for p in files if p.startswith("work/") and p.endswith("/episode/episode.json"))
    recovery = {"rule": RULE, "exhausted_at": EXHAUSTED_AT, "source": str(source),
                "source_identity_sha256": digest(old), "source_identity_file_sha256": files["identity.json"],
                "source_manifest": progress, "source_manifest_sha256": files.get(progress),
                "source_tree_sha256": digest(files), "final_tasks": finals, "final_candidates": candidates,
                "retry_directories": retries, "credits": {p: digest(r) for p, r in records.items()},
                "cap_extensions": extensions,
                "inherited_usage": inherited}
    recovery["immutable_files_sha256"] = digest(copy_layout(files, records)[1])
    return {"files": files, "old": old, "recovery": recovery, "records": records,
            "summary": {"tasks_imported_final": len(finals), "tasks_to_run": len(tasks) - len(finals),
                        "candidates_imported_final": len(candidates), "tasks": tasks,
                        "candidates_to_run": len(tasks) * CANDIDATES - len(candidates) if method == "smartad" else 0,
                        "credits": inherited["quota_void_calls"], "extensions": len(extensions),
                        "total_extension_calls": sum(e["extension"] for e in extensions.values()),
                        "cap_extensions": list(extensions.values()),
                        "tasks_with_fresh_attempt_cap_warning": sum(t["fresh_attempt_cap_warning"] for t in tasks),
                        "inherited_usage": inherited, "new_teacher_calls": 0}}


def recovery_identity(output, proposed):
    """Called before any collector can instantiate a teacher."""
    stored = read_json(Path(output) / "identity.json")
    recovery = stored["chain_recovery"]
    expected = {**proposed, "chain_recovery": recovery}
    if expected != stored:
        from teachers.smartad_resume import COMPLETE, read_plan, registered
        protocol = lambda row: {k: v for k, v in row.items() if k not in {"code_sha256", "dependencies"}}
        if (not registered(output, stored) or not (Path(output) / COMPLETE).exists()
                or not read_plan(output, stored) or protocol(expected) != protocol(stored)):
            raise ValueError("Recovery identity/code/config changed")
        from sed.manifest import frozen_write
        runtime = {"identity_sha256": digest(stored), "code_sha256": proposed["code_sha256"],
                   "dependencies": proposed["dependencies"]}
        frozen_write(Path(output) / "resume_versions" / (digest(runtime) + ".json"), runtime)
    check_imports(output, stored)
    return stored


def check_imports(output, identity=None):
    output = Path(output)
    identity = identity or read_json(output / "identity.json")
    recovery = identity["chain_recovery"]
    from teachers.smartad_resume import archived_path, check_archives, read_plan
    plan = read_plan(output, identity)
    check_archives(output, plan)
    from teachers import smartad_operator_interrupt as operator
    operator_plan = operator.read_plan(output, identity)
    check_archives(output, operator_plan, complete_path=operator.COMPLETE)
    marker = read_json(output / "recovery/complete.json")
    files = read_json(output / "recovery/source_files.json")
    immutable = read_json(output / "recovery/immutable_files.json")
    if (marker["identity_sha256"] != digest(identity) or digest(files) != recovery["source_tree_sha256"]
            or digest(immutable) != recovery["immutable_files_sha256"]):
        raise ValueError("Recovery inventory/identity hash mismatch")
    for name, sha in immutable.items():
        path = archived_path(output, name, plan) if plan else output / name
        if not path.is_file() or file_hash(path) != sha:
            raise ValueError("Imported evidence missing or changed; refusing duplicate calls")
    for relative, sha in recovery["credits"].items():
        credit = read_json(credit_file(output, relative))
        if digest(credit) != sha:
            raise ValueError("Cap-credit hash mismatch")
        extension = recovery.get("cap_extensions", {}).get(relative)
        if extension and read_json(extension_file(output, relative)) != extension:
            raise ValueError("Cap-extension hash mismatch")
        effective_cap(credit, recovery)
        if not credit["final"]:
            counter = archived_path(output, relative, plan) if plan else output / relative
            if not counter.is_file() or read_json(counter.with_name("cap_credit.json")) != credit:
                raise ValueError("Inherited counter or cap-credit missing/changed")
            rows = ledger_rows(counter.with_name("usage.jsonl"))
            if (len(rows) < credit["through_rows"] or
                    digest(rows[:credit["through_rows"]]) != credit["ledger_prefix_sha256"]):
                raise ValueError("Inherited teacher ledger prefix changed")


def recover_chain_collection(source, output, method, manifest, freeze, config, *, dry_run=False, progress=lambda _: None):
    source, output = separate_paths(source, output)
    plan = inspect_source(source, method, manifest, freeze, config, progress=progress)
    progress("Task audit complete; checking source quiescence")
    old, recovery = plan["old"], plan["recovery"]
    identity = {**old, "code_sha256": code_hashes(Path(__file__).resolve().parents[1]),
                "dependencies": versions(), "chain_recovery": recovery}
    if "support_view" in identity:
        import os
        binding = dict(identity["support_view"])
        absolute = (source / "collection" / binding["path"]).resolve()
        binding["path"] = os.path.relpath(absolute, output / "collection")
        identity["support_view"] = binding
    summary = {**plan["summary"], "source": str(source), "output": str(output), "method": method,
               "recovery_rule": RULE, "source_manifest_sha256": recovery["source_manifest_sha256"],
               "source_identity_sha256": recovery["source_identity_sha256"],
               "source_tree_sha256": recovery["source_tree_sha256"], "identity_sha256": digest(identity),
               "dry_run": dry_run}
    if output.exists():
        if read_json(output / "identity.json") != identity:
            raise ValueError("Existing recovery/source hash mismatch")
        check_imports(output, identity)
        return {**summary, "action": "resume"}
    if tree_hashes(source) != plan["files"]:
        raise ValueError("Source changed during audit; stop all source workers first")
    if dry_run:
        return {**summary, "action": "classify_only"}
    staging = output.with_name("." + output.name + ".recovering")
    separate_paths(source, staging)
    with locked(output.with_name("." + output.name + ".recovery.lock")):
        if output.exists():
            check_imports(output, identity)
            return {**summary, "action": "resume"}
        if staging.exists() and read_json(staging / "identity.json") != identity:
            raise ValueError("Partial recovery identity changed")
        write_json(staging / "identity.json", identity)
        destinations, immutable = copy_layout(plan["files"], plan["records"])
        for name, sha in plan["files"].items():
            destination = destinations[name]
            _copy_checked(source / name, staging / destination, sha)
        for relative, record in plan["records"].items():
            if relative != record["source_counter"]:
                prefix = str(Path(record["source_counter"]).parent)
                for name, sha in _under(plan["files"], prefix).items():
                    if name == "counter.json" or name.startswith("cache/"):
                        continue
                    _copy_checked(source / prefix / name, staging / Path(relative).parent / name, sha)
            write_json(credit_file(staging, relative), record)
            if relative in recovery["cap_extensions"]:
                write_json(extension_file(staging, relative), recovery["cap_extensions"][relative])
            if not record["final"]:
                write_json(staging / Path(relative).parent / "cap_credit.json", record)
                write_json(staging / relative, seed_counter(record))
        write_json(staging / "recovery/source_files.json", plan["files"])
        write_json(staging / "recovery/immutable_files.json", immutable)
        if not (staging / "provenance.json").exists():
            write_provenance(staging, command="recover_chain_collection", config=config,
                             details={"identity_sha256": digest(identity), "recovery": recovery,
                                      "new_teacher_calls": 0, "new_environment_steps": 0})
        if tree_hashes(source) != plan["files"]:
            raise ValueError("Source changed during recovery copy")
        write_json(staging / "recovery/complete.json", {"identity_sha256": digest(identity),
                   "immutable_sha256": digest(immutable)})
        staging.rename(output)
        fsync_dir(output.parent)
    return {**summary, "action": "copied"}


def effective_cap(credit, binding):
    extension = binding.get("cap_extensions", {}).get(credit["target_counter"])
    if extension is None:
        if credit.get("cap_extension_sha256"):
            raise ValueError("Missing registered cap extension")
        return credit["cap"]
    budget, original_cap = call_budgets(extension["method"])
    if (digest(extension) != credit.get("cap_extension_sha256")
            or extension["rule"] != RULE or extension["record_type"] != "cap_extension"
            or any(extension[k] != credit[k] for k in ("task_id", "source_counter", "target_counter",
                "source_counter_sha256", "ledger_prefix_sha256"))
            or credit["final"] or extension["credited_calls"] != credit["credit"]
            or extension["calls"] != credit["old_attempts"] or credit["old_attempts"] != credit["cap"]
            or extension["original_cap"] != credit["cap"] or credit["cap"] != original_cap
            or not 0 < credit["credit"] <= budget or extension["per_attempt_call_budget"] != budget
            or extension["extension"] != extension["per_attempt_call_budget"] - credit["credit"]
            or extension["effective_cap"] != credit["cap"] + extension["extension"]):
        raise ValueError("Cap-extension binding/arithmetic mismatch")
    return extension["effective_cap"]


def inherited_counter(counter_path, value, identity, rows):
    """Return the credited floor and separately registered effective cap.

    The inherited ledger prefix remains byte-identical. It is excluded from the
    current-run health scan, never deleted or counted as new transport. The
    counter's identity keeps the original cap; only the bound extension raises
    the runtime cap. Extensions never lower the reservation floor.
    """
    path = Path(counter_path)
    credit_path = path.with_name("cap_credit.json")
    if not credit_path.exists():
        return value, rows, 0, identity["cap"]
    credit = read_json(credit_path)
    root = path
    for _ in Path(credit["target_counter"]).parts:
        root = root.parent
    binding = read_json(root / "identity.json")["chain_recovery"]
    if digest(credit) != binding["credits"].get(credit["target_counter"]):
        raise ValueError("Cap-credit hash mismatch")
    cap = effective_cap(credit, binding)
    extension = binding.get("cap_extensions", {}).get(credit["target_counter"])
    if extension and read_json(extension_file(root, credit["target_counter"])) != extension:
        raise ValueError("Cap-extension hash mismatch")
    if (len(rows) < credit["through_rows"] or
            digest(rows[:credit["through_rows"]]) != credit["ledger_prefix_sha256"]):
        raise ValueError("Inherited teacher ledger prefix changed")
    if {k: v for k, v in identity.items() if k not in {"run_id", "usage_ledger"}} != {
            k: v for k, v in credit["source_identity"].items() if k not in {"run_id", "usage_ledger"}}:
        raise ValueError("Inherited teacher identity/cap changed")
    if value.get("chain_credit_sha256") != digest(credit):
        raise ValueError("Inherited counter changed or lost cap-credit binding")
    if value.get("migration_pending"):
        if value != seed_counter(credit):
            raise ValueError("Initial credited counter changed")
        value = {**value, "identity": identity}
        value.pop("migration_pending")
    return value, rows[credit["through_rows"]:], credit["initial_attempts"], cap


def evidence_path(output, path):
    output, path = Path(output), Path(path)
    identity = read_json(output / "identity.json")
    recovery = identity.get("chain_recovery")
    if recovery and path.is_relative_to(recovery["source"]):
        path = output / path.relative_to(recovery["source"])
    if path.is_relative_to(output):
        from teachers.smartad_resume import archived_path
        return archived_path(output, path.relative_to(output))
    return path


def collection_accounting(output, identity):
    """Canonical ledgers only; copied stages/receipt bundles never add costs."""
    output = Path(output)
    recovery = identity.get("chain_recovery")
    if not recovery:
        return {}
    from teachers.smartad_resume import original_relative, read_plan
    plan = read_plan(output, identity)
    from teachers.smartad_operator_interrupt import read_plan as read_operator_plan
    operator_plan = read_operator_plan(output, identity)
    archives = {"records": [r for p in (plan, operator_plan) if p for r in p["records"]]}
    rows = []
    records = {p: read_json(credit_file(output, p)) for p in recovery["credits"]}
    skipped = {r["source_counter"] for p, r in records.items() if p != r["source_counter"]}
    for ledger in (output / "work").rglob("usage.jsonl"):
        relative = original_relative(ledger.with_name("counter.json").relative_to(output), archives)
        if relative in skipped:
            continue
        data = ledger_rows(ledger)
        if relative in records:
            data = data[records[relative]["through_rows"]:]
        rows.extend(data)
    current = usage(rows)
    source_files = read_json(output / "recovery/source_files.json")
    current["environment_steps"] = sum((read_json(p).get("metrics") or {}).get("environment_steps", 0)
        for p in (output / "work").rglob("episode/episode.json")
        if original_relative(p.relative_to(output), archives) not in source_files)
    extra = sum(p.get("extra_call_upper_bound", 0) for p in (plan, operator_plan) if p)
    return {"inherited_usage": recovery["inherited_usage"], "new_usage": current,
            "smartad_resume_amendment": plan, "amendment_extra_call_upper_bound": extra,
            "smartad_operator_interrupt_amendment": operator_plan,
            "cap_extensions": list(recovery.get("cap_extensions", {}).values()),
            "total_extension_calls": sum(e["extension"] for e in recovery.get("cap_extensions", {}).values()),
            "effective_per_cell_call_cap": identity["per_cell_call_cap"] + extra +
                sum(e["extension"] for e in recovery.get("cap_extensions", {}).values()),
            "new_teacher_calls": current["http_calls"],
            "new_environment_steps": current["environment_steps"],
            "api_attempts": recovery["inherited_usage"]["logical_calls"] + current["logical_calls"],
            "accounting": "Inherited and new usage are disjoint canonical call_id sets. Quota-void calls "
                          "are inherited, unbilled, and credited only against the call cap. "
                          "Receipt usage includes inherited history; do not add it again or sum support views."}
