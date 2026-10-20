"""Transport failures are stage failures, never rejected teacher labels.

The report is attempt-local. Historical attempts/unknown usage remain in their
original ledgers; these counters do not replace billing or reservation records.
"""
from contextvars import ContextVar
from functools import wraps
from inspect import signature
import json
from pathlib import Path

from sed.storage import digest, locked, write_json
from teachers.openai_teacher import TeacherCallError

_audit = ContextVar("teacher_stage_audit", default=None)


def count(kind, n=1):
    audit = _audit.get()
    if audit is not None:
        audit[kind] = audit.get(kind, 0) + n


def teacher_stage(output_arg, *, name=None):
    """Retain a sanitized report even when no dataset/receipt can be published."""
    def decorate(function):
        sig = signature(function)

        @wraps(function)
        def run(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            output = Path(bound.arguments[output_arg])
            existed = output.exists()
            dry_run = getattr(bound.arguments.get("teacher"), "dry_run", False)
            audit = dict(transport_failures=0, transport_failed_attempts=0,
                         rejections=0, parser_rejections=0, unknown_outcomes=0)
            parent = _audit.get()
            token = _audit.set(audit)
            status, error_type = "complete", None
            try:
                result = function(*args, **kwargs)
                if isinstance(result, dict):
                    if result.get("complete") is False:
                        status = "incomplete"
                    elif result.get("status") in {"collecting", "interrupted", "transport_failed", "budget_exhausted"}:
                        status = result["status"]
                return result
            except BaseException as exc:
                status, error_type = "failed", type(exc).__name__
                if isinstance(exc, (KeyboardInterrupt, SystemExit)) and (
                        audit.get("_active_requests", 0) or audit["transport_failed_attempts"]):
                    audit["transport_failures"] += 1
                if isinstance(exc, TeacherCallError):
                    audit["transport_failures"] = max(1, audit["transport_failures"])
                    raise type(exc)(
                        f"Teacher transport failed ({type(exc).__name__}) in {name or function.__qualname__}; "
                        f"stage NOT complete. Inspect {output / 'teacher_report.json'} "
                        "and retained usage/counter evidence before resuming or rerunning."
                    ) from None
                raise
            finally:
                audit["unknown_outcomes"] += audit.pop("_active_requests", 0)
                _audit.reset(token)
                if parent is not None:
                    for key, value in audit.items():
                        parent[key] = parent.get(key, 0) + value
                publish = not dry_run and (status == "complete" or any(audit.values()) or (not existed and output.exists()))
                if publish:
                    report = {
                        "stage": name or function.__qualname__, "status": status,
                        "error_type": error_type, **audit,
                        "scope": "this stage attempt; historical usage remains in retained evidence"}
                    with locked(output / ".teacher_report.lock"):
                        path = output / "teacher_report.json"
                        if path.exists():
                            previous = json.loads(path.read_text())
                            if previous != report:
                                write_json(output / "teacher_reports" / (digest(previous) + ".json"), previous)
                        write_json(path, report)
                # Keep existing reports useful to their callers, including on
                # failures. Never add a completion marker or fabricate a dataset.
                for filename in ("report.json", "result.json", "summary.json"):
                    path = output / filename
                    if publish and path.exists():
                        value = json.loads(path.read_text())
                        if isinstance(value, dict):
                            write_json(path, {**value, **audit})
        return run
    return decorate


def require_completed(result):
    """A completed empty/malformed OUTPUT is a rejection, not a lost request."""
    if result.metadata.get("response_status") != "completed":
        raise TeacherCallError("Teacher response outcome is incomplete/unknown; inspect retained evidence")
    return result


def require_execution(episode):
    """Old failed/unknown executions must not become negative labels on resume."""
    if episode.get("teacher_transport_failure") or episode.get("status", "started") == "started":
        raise TeacherCallError("Unresolved teacher execution; retain evidence and rerun this stage")


def execution_attempt(directory, *, start):
    """Retry a broken execution in a fresh directory, never a fabricated restore.

    Completed format/environment rejections keep their original method slot.
    Unknown historical starts fail once on discovery, before an explicit rerun.
    """
    base = Path(directory)
    current, index = base, 0
    while (current / start).exists():
        episode_path = current / "episode/episode.json"
        episode = json.loads(episode_path.read_text()) if episode_path.exists() else {}
        failure = current / "transport_failure.json"
        if episode.get("teacher_transport_failure") and not failure.exists():
            write_json(failure, {"status": "failed", "transport_failures": 1, "rejections": 0})
        if not failure.exists():
            receipt_path = current / "receipt.json"
            if receipt_path.exists() and json.loads(receipt_path.read_text()).get("status") in {
                    "interrupted", "interrupted_unknown", "transport_failed"}:
                mark_transport_failure(current)
                raise TeacherCallError(f"Historical unknown teacher execution at {current}; inspect before rerunning")
            ledger = current / "teacher/usage.jsonl"
            if ledger.exists():
                try:
                    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
                except ValueError:
                    mark_transport_failure(current)
                    raise TeacherCallError(f"Unknown teacher usage at {current}; inspect before rerunning") from None
                completed = {r.get("logical_call_id", r.get("call_id")) for r in rows
                             if r.get("status") in {"success", "rejected"}}
                failed = any(r.get("status") in {"failed", "incomplete"} and
                             r.get("logical_call_id", r.get("call_id")) not in completed for r in rows)
                ended = {r.get("call_id") for r in rows if r.get("status") in
                         {"success", "rejected", "failed", "incomplete"}}
                unknown = any(r.get("status") == "started" and r.get("call_id") not in ended for r in rows)
                if failed or unknown:
                    mark_transport_failure(current)
                    raise TeacherCallError(f"Historical teacher transport failure at {current}; inspect before rerunning")
            if episode.get("complete") or (current / "receipt.json").exists():
                return current
            rejection = current / "rejection.json"
            if rejection.exists() and json.loads(rejection.read_text()).get("reason") in {
                    "empty_first_paragraph", "first_reply_prefix_mismatch"}:
                return current
            write_json(failure, {"status": "unknown", "transport_failures": 1, "rejections": 0})
            raise TeacherCallError(f"Unknown teacher execution at {current}; review evidence before rerunning")
        index += 1
        current = base.parent / "transport_retries" / base.name / f"attempt-{index}"
    return current


def mark_transport_failure(directory):
    write_json(Path(directory) / "transport_failure.json",
               {"status": "failed", "transport_failures": 1, "rejections": 0})


def usage_counts(rows):
    """Count HTTP failures/unknown attempts separately from output rejections."""
    ended = {r.get("call_id") for r in rows if r.get("status") in
             {"success", "rejected", "failed", "incomplete"}}
    return {
        "transport_failures": sum(r.get("status") in {"failed", "incomplete"} for r in rows)
            + sum(r.get("status") == "started" and r.get("call_id") not in ended for r in rows),
        "rejections": sum(r.get("status") == "rejected" for r in rows),
    }


def ledger_counts(path):
    """Disclosure counts, including interrupted and partially written ledgers."""
    path = Path(path)
    rows, partial = [], 0
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            rows.append(json.loads(line))
        except ValueError:
            partial += 1
    counts = usage_counts(rows)
    counts["unknown_ledger_rows"] = partial
    return counts


def transport_history(root):
    """Bind failed real executions separately from method-level attempt slots."""
    from sed.seed_evidence import artifact_hashes
    root = Path(root)
    return [{"directory": str(p.parent.relative_to(root)),
             "artifacts": artifact_hashes(p.parent)}
            for p in sorted(root.rglob("transport_failure.json"))]
