"""Bounded in-cell amendment of chain_quota_void_20261020_v1.

Only the registered r2 identity may void an interrupted SmartAD sample. Archive
every byte of the old slot before one fresh step-zero sample, cap 41. The quota
credit is NOT applied again. Completed/imported slots are never amended. Dry
runs acquire no locks, instantiate no teachers, and write no files or caches.
"""
from collections import Counter
from pathlib import Path

from sed.provenance import code_hashes, utc_now, versions
from sed.storage import digest, file_hash, fsync_dir, locked, write_json
from teachers.chain_recovery import (RULE, TOKENS, check_imports, credit_file, last_calls,
                                     ledger_rows, quota_void, read_json, terminal_failure, usage)
from teachers.full_train_recovery import tree_hashes

AMENDMENT = "smartad_interrupted_sample_20261021_v1"
CELL = "chain_k500_s0_r2"
CELL_IDENTITY_SHA256 = "fa77630d6af4610728db88c0d185d2722fa7e73049a8192490a164e02ce93d28"
MAX_RESTARTS = 23
NEW_CAP = 41
PLAN = "recovery/" + AMENDMENT + "/plan.json"
COMPLETE = "recovery/" + AMENDMENT + "/complete.json"


def registered(output, identity):
    recovery = identity.get("chain_recovery", {})
    return (identity.get("method") == "smartad" and recovery.get("rule") == RULE
            and digest(identity) == CELL_IDENTITY_SHA256 and Path(output).name == CELL
            and Path(output).resolve().parent == Path(recovery["source"]).parent)


def read_plan(output, identity=None):
    output = Path(output)
    path = output / PLAN
    if not path.exists():
        return None
    identity = identity or read_json(output / "identity.json")
    plan = read_json(path)
    records = plan["records"]
    if (not registered(output, identity) or plan["rule"] != RULE or plan["amendment"] != AMENDMENT
            or plan["identity_sha256"] != digest(identity)
            or not 0 < len(records) <= MAX_RESTARTS
            or len({r["slot"] for r in records}) != len(records)
            or plan["extra_call_upper_bound"] != len(records) * NEW_CAP
            or plan["spent_and_void_usage"] != spent_and_void_usage(records)):
        raise ValueError("Invalid SmartAD amendment binding/bound")
    retries = identity["chain_recovery"]["retry_directories"]
    for record in records:
        slot = record["slot"]
        if (slot not in retries or slot in identity["chain_recovery"]["final_candidates"]
                or slot != f"work/{digest(record['task_id'])[:20]}/{record['candidate']}"
                or record["new_cap"] != NEW_CAP or record["additional_quota_credit"] != 0
                or record["archive"] != slot + "/attempt_archive/" + AMENDMENT
                or record["fresh"] != slot + "/resampled/attempt-1"
                or record["archived_attempt_sha256"] != digest(record["files"])
                or record["credit_sha256"] != identity["chain_recovery"]["credits"].get(
                    retries[slot] + "/teacher/counter.json")):
            raise ValueError("Invalid SmartAD amendment candidate record")
        for name in record["files"]:
            if Path(name).is_absolute() or ".." in Path(name).parts or name.startswith(("attempt_archive/", "resampled/")):
                raise ValueError("Invalid archived attempt path")
    if (output / COMPLETE).exists() and read_json(output / COMPLETE) != {"plan_sha256": digest(plan)}:
        raise ValueError("SmartAD amendment completion binding changed")
    return plan


def archived_path(output, relative, plan=None):
    """Resolve original evidence during/after atomic per-entry archive moves."""
    output, relative = Path(output), Path(relative)
    plan = read_plan(output) if plan is None else plan
    for record in (plan or {}).get("records", []):
        if relative.is_relative_to(record["slot"]):
            suffix = str(relative.relative_to(record["slot"]))
            if suffix in record["files"]:
                archived = output / record["archive"] / suffix
                if archived.exists():
                    return archived
    return output / relative


def original_relative(relative, plan):
    relative = Path(relative)
    for record in (plan or {}).get("records", []):
        if relative.is_relative_to(record["archive"]):
            return str(Path(record["slot"]) / relative.relative_to(record["archive"]))
    return str(relative)


def check_archives(output, plan):
    output = Path(output)
    complete = (output / COMPLETE).exists()
    for record in (plan or {}).get("records", []):
        for name, sha in record["files"].items():
            path = archived_path(output, Path(record["slot"]) / name, plan)
            if (not path.is_file() or path.is_symlink() or file_hash(path) != sha
                    or (complete and not path.is_relative_to(output / record["archive"]))):
                raise ValueError("SmartAD archived attempt changed/missing")
        if complete and read_json(output / record["slot"] / "amendment.json") != record:
            raise ValueError("SmartAD candidate amendment record changed")
        if complete and tree_hashes(output / record["archive"]) != record["files"]:
            raise ValueError("SmartAD archived attempt inventory changed")


def void_attempt(directory, rows, episode, event, error, *, phase):
    """Voiding a sample never refunds its requests, tokens or transitions."""
    return {"phase": phase, "directory": directory, "disposition": "spent_and_void",
            "episode_id": episode["episode_id"], "parent_seed_ids": event["parent_seed_ids"],
            "terminal_error": error, "terminal_event_sha256": digest(event),
            "terminal_verification_ref": event["verification_ref"],
            "ledger_rows_sha256": digest(rows), "usage": usage(rows),
            "environment_steps": episode["metrics"]["environment_steps"]}


def spent_and_void_usage(records):
    totals = {}
    for phase in ("r1", "r2"):
        attempts = [a for r in records for a in r["void_attempts"] if a["phase"] == phase]
        counts = Counter()
        for attempt in attempts:
            counts.update(attempt["usage"]["status_counts"])
        totals[phase] = {**{k: sum(a["usage"][k] for a in attempts) for k in
                           ("http_calls", "logical_calls", "unknown_usage_calls", "quota_void_calls")},
                        "status_counts": dict(counts),
                        "known_billed_tokens": {k: sum(a["usage"]["known_billed_tokens"][k]
                                                       for a in attempts) for k in TOKENS},
                        "environment_steps": sum(a["environment_steps"] for a in attempts)}
    return totals


def _interruption(output, identity, slot, task_id):
    recovery = identity["chain_recovery"]
    retry = recovery["retry_directories"].get(slot)
    if (retry != slot + "/transport_retries/attempt-1" or slot in recovery["final_candidates"]
            or any((output / slot).rglob("candidate.json"))
            or any((output / slot).rglob("outcome.json"))):
        raise ValueError("Not an unfinished candidate in the recovery classification")
    relative = retry + "/teacher/counter.json"
    credit = read_json(credit_file(output, relative))
    rows = ledger_rows(output / retry / "teacher/usage.jsonl")
    old_rows = rows[:credit["through_rows"]]
    ends = [r for r in old_rows if r.get("status") != "state_link"]
    last = ends[-1] if ends else {}
    if (digest(credit) != recovery["credits"].get(relative) or credit["final"]
            or credit["task_id"] != task_id or credit["cap"] != NEW_CAP
            or credit["source_counter"] != slot + "/teacher/counter.json"
            or credit["target_counter"] != relative or credit.get("cap_extension_sha256")
            or credit["credit"] != 1 or credit["initial_attempts"] != credit["old_attempts"] - 1
            or digest(old_rows) != credit["ledger_prefix_sha256"]
            or not quota_void(last)
            or not any(v["call_id"] == last.get("call_id") and v["failure_sha256"] == digest(last)
                       for v in credit["void_calls"])):
        raise ValueError("Latest inherited outcome is not the bound credited quota-void 429")
    old_counter = read_json(output / slot / "teacher/counter.json")
    counter = read_json(output / relative)
    episode, event, error = terminal_failure(output / slot, task_id)
    if (file_hash(output / slot / "teacher/counter.json") != credit["source_counter_sha256"]
            or old_counter["attempts"] != credit["old_attempts"]
            or counter["pending"] or counter.get("chain_credit_sha256") != digest(credit)
            or episode.get("complete") is not False or episode.get("success") != "unknown"
            or not episode.get("teacher_transport_failure") or event.get("error_type") != "policy_error"
            or event.get("status") != "error"):
        raise ValueError("Interrupted quota attempt/counter evidence mismatch")
    new_rows = rows[credit["through_rows"]:]
    diagnostics = {"slot": slot, "task_id": task_id, "candidate": Path(slot).name,
        "source_old_attempts": credit["old_attempts"], "credited_calls": credit["credit"],
        "inherited_floor": credit["initial_attempts"], "recovery_remaining_cap": NEW_CAP - credit["initial_attempts"],
        "old_attempts": counter["attempts"], "new_cap": NEW_CAP,
        "r2_requests": sum(r["status"] == "started" for r in new_rows)}
    attempts = [void_attempt(slot, old_rows, episode, event, error, phase="r1")]
    ep = ev = err = None
    if (output / retry / "started.json").exists():
        ep, ev, err = terminal_failure(output / retry, task_id)
        diagnostics["active_terminal_error"] = err
        diagnostics["active_last_outcome"] = next(
            (r["status"] for r in reversed(rows) if r["status"] != "state_link"), None)
    if new_rows:
        # Explicit 2026-10-21 decision: only this exhausted descendant of the
        # credited r1 quota interruption qualifies. No other r2 failure does.
        starts = [r for r in new_rows if r["status"] == "started"]
        ends = [r for r in new_rows if r["status"] not in {"started", "state_link"}]
        calls = last_calls(new_rows)
        logical = {r.get("logical_call_id", r["call_id"]) for r in starts}
        protocol = lambda row: {k: v for k, v in row.items() if k not in {"run_id", "usage_ledger"}}
        if (ep is None or ep.get("complete") is not False or ep.get("success") != "unknown"
                or not ep.get("teacher_transport_failure") or ev.get("status") != "error"
                or ev.get("error_type") != "policy_error" or err != "BudgetExceeded"
                or not 0 < credit["initial_attempts"] < NEW_CAP
                or counter["attempts"] != NEW_CAP or counter["identity"]["cap"] != NEW_CAP
                or counter.get("halted") or counter.get("migration_pending") or counter.get("recovery")
                or protocol(counter["identity"]) != protocol(credit["source_identity"])
                or Path(counter["identity"]["usage_ledger"]) != output / retry / "teacher/usage.jsonl"
                or len(logical) != diagnostics["recovery_remaining_cap"]
                or len(starts) != len(logical) or len(ends) != len(starts)
                or len(calls) != len(starts) or {r["call_id"] for r in starts} != set(calls)
                or {r["call_id"] for r in ends} != set(calls)
                or any(r["status"] != "success" and not (
                    r["status"] == "rejected" and r.get("error_type") == "ActionParseError") for r in ends)
                or any(r.get("run_id") != counter["identity"]["run_id"]
                       or r.get("task_id", task_id) != task_id for r in new_rows)
                or any(r.get("logical_call_id", r["call_id"]) not in logical for r in ends)
                or any(p.name != "attempt-1" for p in (output / slot / "transport_retries").iterdir())):
            return diagnostics, "R2 retry is not an unfinished BudgetExceeded execution at the inherited shortened cap"
        attempts.append(void_attempt(retry, new_rows, ep, ev, err, phase="r2"))
        diagnostics["eligibility"] = "r1_quota_void_then_r2_inherited_cap_exhausted"
    elif counter["attempts"] != credit["initial_attempts"]:
        raise ValueError("Credited counter changed without matching requests")
    elif ep is not None:
        if ep.get("complete") is not False or not ep.get("teacher_transport_failure") or err != "BudgetExceeded":
            raise ValueError("Unrecognized no-request r2 interruption")
        attempts.append(void_attempt(retry, new_rows, ep, ev, err, phase="r2"))
    diagnostics.setdefault("eligibility", "r1_quota_void_without_r2_requests")
    files = tree_hashes(output / slot)
    return {**diagnostics, "credit_sha256": digest(credit), "additional_quota_credit": 0,
            "void_attempts": attempts,
            "archive": slot + "/attempt_archive/" + AMENDMENT,
            "fresh": slot + "/resampled/attempt-1", "files": files,
            "archived_attempt_sha256": digest(files)}, None


def classify(output):
    output = Path(output).resolve()
    identity = read_json(output / "identity.json")
    if not registered(output, identity):
        raise ValueError("SmartAD amendment is restricted to the registered existing r2 cell")
    plan = read_plan(output, identity)
    check_imports(output, identity)
    check_archives(output, plan)
    source = read_json(Path(identity["source_verified"]) / "manifest.json")
    if digest(source) != identity["source_collection_sha256"]:
        raise ValueError("Frozen support manifest changed")
    counts = dict(completed=0, imported=0, interrupted_to_restart=0, never_started=0)
    records, refused, diagnostics = [], [], []
    amended = {r["slot"]: r for r in (plan or {}).get("records", [])}
    imports = set(identity["chain_recovery"]["final_candidates"])
    expected = set()
    for receipt in source["verified"]:
        task_id = receipt["task_id"]
        for number in range(1, 11):
            slot = f"work/{digest(task_id)[:20]}/candidate-{number:02d}"
            expected.add(slot)
            directory = output / slot
            if (directory / "candidate.json").exists():
                row = read_json(directory / "candidate.json")
                if row["task_id"] != task_id or row["candidate_id"] != directory.name:
                    raise ValueError("Candidate identity mismatch")
                counts["imported" if slot in imports else "completed"] += 1
            elif slot in amended:
                counts["interrupted_to_restart"] += 1
            elif directory.exists() and tree_hashes(directory):
                record, reason = _interruption(output, identity, slot, task_id)
                diagnostics.append({k: v for k, v in record.items() if k not in {"files"}})
                if reason:
                    refused.append({**diagnostics[-1], "reason": reason})
                else:
                    records.append(record)
                    counts["interrupted_to_restart"] += 1
            else:
                counts["never_started"] += 1
    if not imports <= expected or any(str(p.relative_to(output)) not in expected
                                      for p in (output / "work").glob("*/candidate-*")):
        raise ValueError("Candidate outside frozen support")
    if len(records) + len(amended) > MAX_RESTARTS:
        raise ValueError("SmartAD amendment exceeds 23 candidates / 943 calls")
    if plan and records:
        raise ValueError("Amendment membership is frozen; no second credit/restart")
    bound = (len(records) + len(amended)) * NEW_CAP
    return {"cell": output.name, "rule": RULE, "amendment": AMENDMENT,
            **counts, "refused": len(refused), "extra_call_upper_bound": bound,
            "max_extra_call_bound": MAX_RESTARTS * NEW_CAP,
            "remaining_call_upper_bound": (counts["never_started"] + counts["interrupted_to_restart"]) * NEW_CAP,
            "call_bound_unit": "logical teacher calls; HTTP retries retain the existing transport policy",
            "spent_and_void_usage": spent_and_void_usage([*records, *amended.values()]),
            "ready": not refused, "records": records, "refusals": refused,
            "diagnostics": diagnostics, "identity_sha256": digest(identity)}


def amend(output, *, resume, dry_run=False):
    """Preflight all slots before writes; resume a journaled move, never re-credit."""
    if not resume:
        raise ValueError("SmartAD amendment requires --resume in the existing r2 cell")
    output = Path(output).resolve()
    if dry_run:
        return classify(output)
    # Caller holds .pipeline.lock (the launcher applies before starting shards).
    summary = classify(output)
    if not summary["ready"]:
        raise ValueError("SmartAD amendment refused: active outcomes do not satisfy the registered policy; use --dry-run")
    plan = read_plan(output)
    if plan is None:
        if not summary["records"]:
            return summary
        plan = {"rule": RULE, "amendment": AMENDMENT,
                "identity_sha256": summary["identity_sha256"], "records": summary["records"],
                "extra_call_upper_bound": summary["extra_call_upper_bound"],
                "spent_and_void_usage": summary["spent_and_void_usage"],
                "timestamp": utc_now(), "code_sha256": code_hashes(Path(__file__).resolve().parents[1]),
                "dependencies": versions()}
        write_json(output / PLAN, plan, exclusive=True)
    if (output / COMPLETE).exists():
        return summary
    for record in plan["records"]:
        slot, archive = output / record["slot"], output / record["archive"]
        with locked(slot / ".slot.lock"):
            if (slot / "candidate.json").exists():
                raise ValueError("Candidate completed during amendment; stop workers before resuming")
            check_archives(output, {"records": [record]})
            archive.mkdir(parents=True, exist_ok=True)
            # Rename whole entries (including their lock files). Never copy or
            # truncate an existing ledger; a crash resumes these exact moves.
            entries = {Path(name).parts[0] for name in record["files"]}
            for name in sorted(entries):
                src, dst = slot / name, archive / name
                if dst.exists():
                    if src.exists():
                        raise ValueError("Duplicate live/archive attempt entry")
                    continue
                src.rename(dst)
                fsync_dir(slot)
                fsync_dir(archive)
            path = slot / "amendment.json"
            if path.exists():
                if read_json(path) != record:
                    raise ValueError("Changed candidate amendment")
            else:
                write_json(path, record, exclusive=True)
    check_archives(output, plan)
    write_json(output / COMPLETE, {"plan_sha256": digest(plan)}, exclusive=True)
    return summary


def check_resume_identity(stored, proposed):
    expected = {**proposed, "chain_recovery": stored["chain_recovery"]}
    protocol = lambda row: {k: v for k, v in row.items() if k not in {"code_sha256", "dependencies"}}
    if protocol(stored) != protocol(expected):
        raise ValueError("SmartAD amendment cannot change frozen source/config/sampling/protocol")


def fresh_directory(output, slot):
    plan = read_plan(output)
    for record in (plan or {}).get("records", []):
        if record["slot"] == str(slot):
            if not (Path(output) / COMPLETE).exists():
                raise ValueError("SmartAD amendment archive is not complete")
            return record["fresh"]
    return None


def display(summary):
    return {k: v for k, v in summary.items() if k not in {"records", "diagnostics", "refusals", "identity_sha256"}}
