"""Separate, offline amendment for exactly two operator-terminated r2 samples.

No discovery of additional slots, quota credits, teacher construction or calls.
After publication only the frozen plan/archives are verified, even on dry runs.
"""
from collections import Counter
from pathlib import Path

from sed.provenance import code_hashes, utc_now, versions
from sed.storage import digest, write_json
from teachers import smartad_resume as first
from teachers.chain_recovery import RULE, TOKENS, ledger_rows, read_json, usage
from teachers.full_train_recovery import tree_hashes

AMENDMENT = "smartad_operator_interrupt_20261021_v1"
PLAN = f"recovery/{AMENDMENT}/plan.json"
COMPLETE = f"recovery/{AMENDMENT}/complete.json"
SLOTS = {"work/501b2815d5ab7f82152f/candidate-10": 2,
         "work/581264b0fd84a99a2e1c/candidate-09": 6}
NEW_CAP = 41
EXTRA_CALL_BOUND = 82
CAUSE = "operator_SIGTERM_launcher_stop_terminated_live_shards"


def spent_and_void_usage(records):
    counts = Counter()
    for record in records:
        counts.update(record["usage"]["status_counts"])
    return {"counter_attempts": sum(r["old_attempts"] for r in records),
            "attempts_by_slot": {r["slot"]: r["old_attempts"] for r in records},
            **{k: sum(r["usage"][k] for r in records) for k in
               ("http_calls", "logical_calls", "unknown_usage_calls", "quota_void_calls")},
            "unknown_outcome_in_flight_requests": len(records),
            "unknown_outcome_usage": "unknown; never refunded",
            "refunded_attempts": 0, "status_counts": dict(counts),
            "known_billed_tokens": {k: sum(r["usage"]["known_billed_tokens"][k] for r in records)
                                    for k in TOKENS},
            "environment_steps": sum(r["environment_steps"] for r in records)}


def _first_plan(output, identity):
    if not first.registered(output, identity) or not (output / first.COMPLETE).is_file():
        raise ValueError("SmartAD operator amendment requires the registered cell and COMPLETE first amendment")
    return first.read_plan(output, identity)


def read_plan(output, identity=None):
    output = Path(output)
    if not (output / PLAN).exists():
        if (output / COMPLETE).exists():
            raise ValueError("SmartAD operator completion has no bound plan")
        return None
    identity = identity or read_json(output / "identity.json")
    previous = _first_plan(output, identity)
    plan = read_json(output / PLAN)
    records = plan["records"]
    if (plan["amendment"] != AMENDMENT or plan["rule"] != RULE
            or plan["identity_sha256"] != digest(identity)
            or plan["first_amendment_plan_sha256"] != digest(previous)
            or plan["cause"] != CAUSE or len(records) != 2
            or {r["slot"] for r in records} != set(SLOTS)
            or plan["extra_call_upper_bound"] != EXTRA_CALL_BOUND
            or plan["spent_and_void_usage"] != spent_and_void_usage(records)):
        raise ValueError("Invalid SmartAD operator amendment binding/bound/membership")
    excluded = {r["slot"] for r in previous["records"]} | set(identity["chain_recovery"]["final_candidates"])
    excluded |= set(identity["chain_recovery"]["retry_directories"])
    for record in records:
        slot = record["slot"]
        if (slot in excluded or slot != f"work/{digest(record['task_id'])[:20]}/{record['candidate']}"
                or record["old_attempts"] != SLOTS[slot] or record["new_cap"] != NEW_CAP
                or record["additional_quota_credit"] != 0 or record["disposition"] != "spent_and_void"
                or record["archive"] != f"{slot}/attempt_archive/{AMENDMENT}"
                or record["fresh"] != f"{slot}/resampled/attempt-1"
                or record["archived_attempt_sha256"] != digest(record["files"])):
            raise ValueError("Invalid SmartAD operator candidate record")
        for name in record["files"]:
            if (Path(name).is_absolute() or ".." in Path(name).parts
                    or Path(name).parts[0] in {"attempt_archive", "resampled", "amendment.json", "candidate.json"}):
                raise ValueError("Invalid SmartAD operator archive path")
    if (output / COMPLETE).exists() and read_json(output / COMPLETE) != {"plan_sha256": digest(plan)}:
        raise ValueError("SmartAD operator completion binding changed")
    return plan


def _interruption(output, identity, previous, slot, task_id):
    if slot not in SLOTS:
        raise ValueError("Slot outside the two registered operator interruptions")
    directory = output / slot
    excluded = {r["slot"] for r in previous["records"]} | set(identity["chain_recovery"]["final_candidates"])
    excluded |= set(identity["chain_recovery"]["retry_directories"])
    if (slot in excluded or any(directory.rglob("candidate.json")) or any(directory.rglob("outcome.json"))
            or any((directory / name).exists() for name in
                   ("attempt_archive", "resampled", "amendment.json", "transport_retries", "teacher/cap_credit.json"))):
        raise ValueError("Candidate completed, previously amended, imported or retried")
    # Check the entire inventory before reading; symlinks cannot redirect evidence.
    files = tree_hashes(directory)
    started = read_json(directory / "started.json")
    episode = read_json(directory / "episode/episode.json")
    counter = read_json(directory / "teacher/counter.json")
    ledger = directory / "teacher/usage.jsonl"
    rows = ledger_rows(ledger)
    current = counter["identity"]
    if (started["task_id"] != task_id or episode["task_id"] != task_id
            or not started.get("parent_seed_ids") or episode.get("complete") is not False
            or episode.get("success", "unknown") != "unknown"
            or episode.get("terminated") or episode.get("truncated")
            or current["cap"] != NEW_CAP or Path(current["usage_ledger"]) != ledger
            or counter["attempts"] != SLOTS[slot] or counter.get("pending") is not False
            or counter.get("halted") is not False or counter.get("migration_pending")
            or counter.get("chain_credit_sha256") or not current.get("run_id")
            or any(r.get("run_id") != current["run_id"] or r.get("task_id", task_id) != task_id for r in rows)):
        raise ValueError("Incomplete episode or current counter/run identity mismatch")
    # Earlier starts must each have a success; only the very last row may be
    # unresolved. No rejected/failed/cache-hit or post-interruption rows qualify.
    starts, successes, logical = {}, set(), set()
    for row in rows:
        status, call = row.get("status"), row.get("call_id")
        if not call:
            raise ValueError("Ledger row has no call identity")
        if status == "started":
            lid = row.get("logical_call_id", call)
            if call in starts or lid in logical or set(starts) != successes:
                raise ValueError("Duplicate or unresolved earlier ledger start")
            starts[call] = lid
            logical.add(lid)
        elif status == "success":
            if call not in starts or call in successes or row.get("logical_call_id", call) != starts[call]:
                raise ValueError("Success lacks its unique matching start")
            successes.add(call)
        elif status != "state_link" or call not in successes or set(starts) != successes:
            raise ValueError("Ledger has a non-success outcome or rows after the unresolved start")
    if (not rows or rows[-1].get("status") != "started"
            or set(starts) - successes != {rows[-1]["call_id"]}
            or len(starts) != SLOTS[slot] or rows[-1].get("usage") is not None):
        raise ValueError("Expected exactly one trailing unknown-outcome started row")
    recoveries = directory / "teacher/recoveries"
    if sorted(p.name for p in recoveries.glob("*.json")) != ["0000.json"]:
        raise ValueError("Missing or additional interruption recovery")
    recovery = read_json(recoveries / "0000.json")
    boundary = {"through_rows": len(rows), "ledger_prefix_sha256": digest(rows)}
    prior = recovery["previous_counter"]
    if (recovery.get("reason") != "interruption_reviewed"
            or any(recovery.get(k) != v for k, v in boundary.items())
            or recovery.get("unknown_usage") != "unchanged; no reservations refunded"
            or prior.get("halted") is not True or prior.get("recovery")
            or counter != {**prior, "pending": False, "halted": False, "recovery": boundary}):
        raise ValueError("Recovery is not bound to the full ledger prefix without refund")
    return {"slot": slot, "task_id": task_id, "candidate": directory.name,
            "eligibility": "operator_interruption_reviewed", "disposition": "spent_and_void",
            "old_attempts": counter["attempts"], "new_cap": NEW_CAP, "additional_quota_credit": 0,
            "episode_id": episode["episode_id"], "parent_seed_ids": started["parent_seed_ids"],
            "environment_steps": episode["metrics"]["environment_steps"],
            "run_id": current["run_id"], "ledger_rows_sha256": digest(rows),
            "recovery_sha256": digest(recovery), "unknown_outcome_call_id": rows[-1]["call_id"],
            "usage": usage(rows), "archive": f"{slot}/attempt_archive/{AMENDMENT}",
            "fresh": f"{slot}/resampled/attempt-1", "files": files,
            "archived_attempt_sha256": digest(files)}


def _summary(output, records, refusals=(), *, plan=None):
    result = {"cell": output.name, "rule": RULE, "amendment": AMENDMENT, "cause": CAUSE,
              "eligible": len(records), "refused": len(refusals), "ready": not refusals and len(records) == 2,
              "extra_call_upper_bound": len(records) * NEW_CAP, "max_extra_call_bound": EXTRA_CALL_BOUND,
              "call_bound_unit": "logical teacher calls; HTTP retries retain the existing transport policy",
              "spent_and_void_usage": spent_and_void_usage(records),
              "records": records, "refusals": list(refusals)}
    if plan is not None:
        result.update(applied=(output / COMPLETE).exists(), plan_sha256=digest(plan))
    return result


def classify(output):
    """Read exactly the allowlisted slots, never scan/reclassify running shards."""
    output = Path(output).resolve()
    identity = read_json(output / "identity.json")
    previous = _first_plan(output, identity)
    first.amend(output, resume=True)  # COMPLETE branch is read-only verification.
    source = read_json(Path(identity["source_verified"]) / "manifest.json")
    if digest(source) != identity["source_collection_sha256"]:
        raise ValueError("Frozen support manifest changed")
    tasks = {digest(r["task_id"])[:20]: r["task_id"] for r in source["verified"]}
    records, refusals = [], []
    for slot in SLOTS:
        try:
            records.append(_interruption(output, identity, previous, slot, tasks[Path(slot).parts[1]]))
        except ValueError as error:
            refusals.append({"slot": slot, "reason": str(error)})
        except (KeyError, TypeError, OSError):
            refusals.append({"slot": slot, "reason": "Missing/malformed operator interruption evidence"})
    return _summary(output, records, refusals)


def amend(output, *, resume, dry_run=False):
    if not resume:
        raise ValueError("SmartAD operator amendment requires --resume")
    output = Path(output).resolve()
    plan = read_plan(output)
    if plan is not None:
        first.amend(output, resume=True)
        first.check_archives(output, plan, complete_path=COMPLETE)
        first.check_fresh_directories(output, plan)
        if (output / COMPLETE).exists() or dry_run:
            return _summary(output, plan["records"], plan=plan)
    else:
        summary = classify(output)
        if dry_run:
            return summary
        if not summary["ready"]:
            raise ValueError("SmartAD operator amendment refused; inspect --dry-run --details")
        identity = read_json(output / "identity.json")
        plan = {"rule": RULE, "amendment": AMENDMENT, "cause": CAUSE,
                "identity_sha256": digest(identity),
                "first_amendment_plan_sha256": digest(first.read_plan(output, identity)),
                "records": summary["records"], "extra_call_upper_bound": EXTRA_CALL_BOUND,
                "spent_and_void_usage": summary["spent_and_void_usage"],
                "timestamp": utc_now(), "code_sha256": code_hashes(Path(__file__).resolve().parents[1]),
                "dependencies": versions()}
        write_json(output / PLAN, plan, exclusive=True)
    first.archive_attempts(output, plan, complete_path=COMPLETE)
    return _summary(output, plan["records"], plan=plan)


def resume_if_present(output, *, dry_run=False):
    """Only the explicit offline CLI may enroll these two slots."""
    output = Path(output)
    if (output / PLAN).exists() or (output / COMPLETE).exists():
        return amend(output, resume=True, dry_run=dry_run)
    return None
