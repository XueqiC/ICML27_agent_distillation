"""Offline, evidence-bound migration of interrupted full-TRAIN collections.

The old collection is only read. Reservations are copied without refunds, including
failed requests. A published recovery is immutable and can only be resumed.
"""
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sqlite3

from sed.manifest import resolve_task
from sed.provenance import safe_config, write_provenance
from sed.seed_evidence import artifact_hashes, environment_won
from sed.seed_selection import ordered_candidates, validate_selection
from sed.storage import BlobStore, digest, file_hash, fsync_dir, locked, write_json

RULE = "full_train_valid_complete_copy_v1"
VALID = "valid_complete"
TRANSPORT = "transport_failed"
UNKNOWN = "unknown_interrupted"
DB_PATH = "teacher_budget/reservations.sqlite3"
DEFAULT_SOURCE = "/home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train"


def read_json(path):
    return json.loads(Path(path).read_text())


def separate_paths(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Refusing to write into the source or an overlapping collection")
    return source, output


def tree_hashes(root):
    """No links, including directory links; no locks acquired inside the source."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Recovery requires real directories, not symlinks")
    def paths():
        for directory, dirs, files in os.walk(root):
            for name in dirs + files:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("Symlink in recovery evidence")
            for name in files:
                if not name.endswith(".lock"):
                    yield Path(directory) / name
    # Streaming traversal avoids holding the directory graph of the entire
    # full-TRAIN cache in memory. Bound queued hashes as well as open files.
    result, pending = {}, deque()
    workers = int(os.environ.get("SED_HASH_WORKERS", "16"))
    if workers < 1:
        raise ValueError("SED_HASH_WORKERS must be positive")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path in paths():
            pending.append((str(path.relative_to(root)), pool.submit(file_hash, path)))
            if len(pending) >= workers * 2:
                name, future = pending.popleft()
                result[name] = future.result()
        for name, future in pending:
            result[name] = future.result()
    return result


def _last_calls(rows):
    calls = {}
    for row in rows:
        if row.get("status") == "state_link":
            continue
        if not row.get("call_id"):
            raise ValueError("Teacher ledger row has no call_id")
        calls[row["call_id"]] = row
    return list(calls.values())


def classify_attempt(directory):
    """Classify from local evidence, never from dates or receipt status alone.

    549503d has no logical_call_id. Its retry chain uses the same run/cache key
    and increasing HTTP attempt numbers. A new attempt=1 starts a new chain.
    Only explicit will_retry=true failures followed by a completed chain recover.
    """
    directory = Path(directory)
    episode_path = directory / "episode/episode.json"
    episode = read_json(episode_path) if episode_path.exists() else {}
    ledger = directory / "teacher/usage.jsonl"
    rows, malformed = [], False
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                malformed = True
    calls = _last_calls(rows)
    chains, old_keys = defaultdict(list), {}
    for row in calls:
        if row.get("logical_call_id"):
            key = ("logical", row["logical_call_id"])
        else:
            base = (row.get("run_id"), row.get("cache_key"))
            prior = old_keys.get(base)
            number = row.get("attempt", 1)
            if (prior is None or number <= prior[1]
                    or not all(base)):
                key = ("legacy", row["call_id"])
            else:
                key = prior[0]
            old_keys[base] = key, number
        chains[key].append(row)
    terminal, recovered_rows = [], 0
    for chain in chains.values():
        for index, row in enumerate(chain):
            status = row.get("status")
            if status not in {"failed", "incomplete"}:
                continue
            following = chain[index + 1:]
            recovered = (status == "failed" and row.get("will_retry") is True
                         and following and following[-1].get("status") in {"success", "rejected"}
                         and all(r.get("status") == "failed" and r.get("will_retry") is True
                                 for r in following[:-1]))
            if recovered:
                recovered_rows += 1
            elif status == "incomplete" or row.get("will_retry") is not True or not following:
                terminal.append(row)
    failed = [r for r in calls if r.get("status") in {"failed", "incomplete"}]
    details = {
        "http_status_counts": dict(Counter(str(r.get("http_status")) for r in failed)),
        "error_type_counts": dict(Counter(r.get("error_type") or r["status"] for r in failed)),
        "terminal_http_status_counts": dict(Counter(str(r.get("http_status")) for r in terminal)),
        "terminal_error_type_counts": dict(Counter(r.get("error_type") or r["status"] for r in terminal)),
        "recovered_retry_rows": recovered_rows,
        "unrecovered_retry_rows": len(failed) - len(terminal) - recovered_rows,
    }
    reasons = []
    if malformed:
        reasons.append("malformed_usage_ledger")
    if episode.get("status", "started") == "started":
        reasons.append("episode_started_or_missing")
    if any(r.get("status") == "started" for r in calls):
        reasons.append("teacher_start_without_end")
    if reasons:
        return {**details, "classification": UNKNOWN, "reasons": reasons}, calls
    if terminal:
        reasons.append("terminal_teacher_transport_row")
    teacher_error = episode.get("teacher_transport_failure") or any(
        r.get("error_type") in {"TeacherCallError", "TeacherQuotaError", "TeacherRateLimitError"}
        for r in [episode, *episode.get("policy_calls", [])])
    # The legacy runner records exception class in the event verification blob,
    # while episode.json only says policy_error. This is audit-only evidence.
    events = directory / "episode/events.jsonl"
    if episode.get("error_type") == "policy_error" and events.exists():
        for line in events.read_text().splitlines():
            event = json.loads(line)
            if event.get("error_type") == "policy_error":
                audit = BlobStore(directory / "episode/blobs").get(event["verification_ref"]).get("audit", {})
                teacher_error |= audit.get("error_type") in {
                    "TeacherCallError", "TeacherQuotaError", "TeacherRateLimitError"}
    if teacher_error:
        reasons.append("episode_teacher_call_error")
    if reasons:
        return {**details, "classification": TRANSPORT, "reasons": reasons}, calls
    final = (episode.get("steps") or [episode.get("initial", {})])[-1]
    complete = (episode.get("complete") is True and episode.get("status") == "ok"
                and type(episode.get("success")) is bool
                and (episode.get("terminated") is True or episode.get("truncated") is True)
                and (final.get("terminated") is True or final.get("truncated") is True))
    call_ids = {r["call_id"] for r in calls if r["status"] in {"success", "rejected", "cache_hit"}}
    missing_calls = any(r.get("call_id") not in call_ids for r in episode.get("policy_calls", []))
    if complete and ledger.exists() and not missing_calls:
        verification = BlobStore(directory / "episode/blobs").get(final["verification_ref"])
        real_end = (verification.get("fact_source") == "real_environment"
                    and verification.get("executable") is True)
        # A parser rejection can consume the final harness step-cap turn. It
        # cannot establish a win; the reset/earlier transitions remain real.
        step_cap = (episode.get("truncated") is True and not episode["success"]
                    and len(episode.get("steps", [])) == episode.get("max_episode_steps")
                    and verification.get("failure_reason") == "parser_error")
        if real_end or step_cap:
            if episode["success"] != environment_won(directory / "episode", episode):
                raise ValueError("Episode success disagrees with real environment evidence")
            return {**details, "classification": VALID, "reasons": ["real_environment_end"]}, calls
    return {**details, "classification": UNKNOWN,
            "reasons": ["no_verified_complete_environment_end"]}, calls


def reservation_state(path, *, live=False):
    """Immutable SQLite reader cannot create a journal or lock in the source."""
    path = Path(path)
    for suffix in (() if live else ("-wal", "-journal")):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError("Unsettled SQLite journal; source must be quiescent")
    with sqlite3.connect(path.as_uri() + ("?mode=ro" if live else "?mode=ro&immutable=1"), uri=True) as db:
        db.execute("BEGIN")
        state = json.loads(db.execute("SELECT value FROM budget WHERE id=1").fetchone()[0])
        calls = db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        requests, tokens = db.execute("SELECT COUNT(*), COALESCE(SUM(input_tokens),0) FROM requests").fetchone()
    if (calls, requests, tokens) != (state["logical_calls"], state["transport_reservations"],
                                    state["reserved_input_tokens"]):
        raise ValueError("Reservation totals disagree with database rows")
    return state


def _under(files, prefix):
    prefix += "/"
    return {name[len(prefix):]: sha for name, sha in files.items() if name.startswith(prefix)}


def attempt_files(files):
    result = defaultdict(dict)
    for name, sha in files.items():
        parts = name.split("/", 2)
        if len(parts) == 3 and parts[0] == "attempts":
            result["/".join(parts[:2])][parts[2]] = sha
    return result


def inspect_source(source, manifest, freeze, config):
    """Read-only audit and complete content inventory, including excluded evidence."""
    source = Path(source).resolve()
    files = tree_hashes(source)
    identity, collection, selection = [read_json(source / name) for name in
                                      ("identity.json", "manifest.json", "selection.json")]
    validate_selection(selection, manifest, freeze)
    if (identity["selection"] != selection or selection.get("selection_rule") != "all_train_v1"
            or collection["identity_sha256"] != digest(identity)
            or collection["selection_sha256"] != digest(selection)
            or identity["attempts_per_candidate"] != 3):
        raise ValueError("Source identity/manifest/selection mismatch")
    # Paths and transport retry settings may change in a new code identity.
    # Models, sampling, visible interface and environment semantics may not.
    previous = identity["config"]
    for key in ("provider_model_id", "sampling", "requested_name"):
        if config["models"]["teacher"][key] != previous["models"]["teacher"][key]:
            raise ValueError("Recovery cannot substitute the source teacher or sampling")
    if config["models"]["student"] != previous["models"]["student"]:
        raise ValueError("Recovery cannot substitute the source student")
    for key in set(previous["environment"]) - {"data_root", "python"}:
        if config["environment"].get(key) != previous["environment"][key]:
            raise ValueError("Recovery cannot change environment semantics")
    reservations = reservation_state(source / DB_PATH)
    if reservations["limits"] != {k: identity[k] for k in ("max_calls", "max_input_tokens")}:
        raise ValueError("Source caps disagree with reservation database")
    indexed = {str(Path(r["episode_dir"]).parent): r for r in collection["attempts"]}
    evidence = attempt_files(files)
    directories = sorted(evidence)
    if not set(indexed) <= set(directories):
        raise ValueError("Source manifest references missing attempts")
    order = {r["task_id"]: i for i, r in enumerate(ordered_candidates(selection))}
    records, all_calls = [], {}
    for relative in directories:
        directory = source / relative
        started = read_json(directory / "started.json")
        task = resolve_task(manifest, freeze, started["task_id"])
        number = started["attempt"]
        if (task["task_id"] not in order or type(number) is not int or not 1 <= number <= 3
                or relative != f"attempts/{digest(task['task_id'])[:20]}-{number}"):
            raise ValueError("Source attempt differs from frozen TRAIN selection")
        receipt_file = directory / "receipt.json"
        receipt = read_json(receipt_file) if receipt_file.exists() else None
        if relative in indexed and indexed[relative] != receipt:
            raise ValueError("Source manifest receipt changed (hash mismatch)")
        hashes = evidence[relative]
        if receipt is not None:
            if (receipt["task_id"] != task["task_id"] or receipt["attempt"] != number
                    or receipt["group_id"] != task["group_id"] or receipt["task_type"] != task["task_type"]
                    or receipt["selection_parent_task_id"] != task["task_id"]
                    or receipt["episode_dir"] != relative + "/episode"
                    or receipt["episode_hashes"] != _under(hashes, "episode")
                    or receipt["teacher_hashes"] != _under(hashes, "teacher")):
                raise ValueError("Source attempt evidence changed (hash mismatch)")
        classification, calls = classify_attempt(directory)
        for call in calls:
            if call["call_id"] in all_calls:
                raise ValueError("Duplicate call_id across source attempts")
            all_calls[call["call_id"]] = call
        episode_path = directory / "episode/episode.json"
        episode = read_json(episode_path) if episode_path.exists() else {}
        if episode and (episode["task_id"] != task["task_id"] or episode.get("split") != "train"
                        or episode.get("game_sha256") != task["game_sha256"]):
            raise ValueError("Source episode differs from frozen TRAIN task")
        won = environment_won(directory / "episode", episode)
        if receipt is not None and receipt["won"] != won:
            raise ValueError("Source receipt win disagrees with environment")
        records.append({**classification, "task_id": task["task_id"],
                        "source_attempt": number, "source_path": relative,
                        "source_attempt_sha256": digest(hashes),
                        "source_receipt_sha256": hashes.get("receipt.json"),
                        "source_receipt_status": receipt.get("status") if receipt else None, "won": won})
    records.sort(key=lambda r: (order[r["task_id"]], r["source_attempt"]))
    imported, excluded, counts, won_tasks = [], [], Counter(), set()
    for row in records:
        if row["classification"] != VALID:
            excluded.append(row)
            continue
        tid = row["task_id"]
        if tid in won_tasks:
            raise ValueError("Valid source attempt after first win; refusing ambiguous migration")
        counts[tid] += 1
        imported.append({**row, "attempt": counts[tid],
                         "target_path": f"attempts/{digest(tid)[:20]}-{counts[tid]}"})
        if row["won"]:
            won_tasks.add(tid)
    settled = [r for r in all_calls.values() if r.get("usage") is not None and r["status"] != "started"]
    inherited = {
        "settled_calls": len(settled),
        "settled_usage": {key: sum((r.get("usage") or {}).get(key, 0) or 0 for r in settled)
                          for key in ("input_tokens", "output_tokens", "reasoning_tokens")},
        "last_state_counts": dict(Counter(r["status"] for r in all_calls.values())),
        "unknown_usage_calls": sum(r.get("usage") is None for r in all_calls.values()),
        "reservations": reservations,
        "accounting": "Source SQLite copied intact; all reservations, including excluded attempts, remain spent. "
                      "Settled usage is last state per call_id and is not the reservation counter.",
    }
    class_counts = {key: sum(r["classification"] == key for r in records)
                    for key in (VALID, TRANSPORT, UNKNOWN)}
    errors = sum(r.get("status") == "error" for r in collection["attempts"])
    excluded_errors = sum(r["source_receipt_status"] == "error" for r in excluded)
    summary = {"classification_counts": class_counts, "source_receipts": len(indexed),
               "source_manifest_status": collection["status"],
               "source_reported_attempted_unique_instances": collection["attempted_unique_instances"],
               "source_receipt_unique_instances": len({r["task_id"] for r in collection["attempts"]}),
               "source_error_receipts": errors, "excluded_error_receipts": excluded_errors,
               "errors_reconciled": errors == excluded_errors,
               "excluded_without_error_receipt": len(excluded) - excluded_errors,
               "trusted_trajectories": len(won_tasks), "unique_instances": len(won_tasks),
               "task_groups": len({resolve_task(manifest, freeze, tid)["group_id"] for tid in won_tasks}),
               "attempted_unique_instances": len({r["task_id"] for r in records}),
               "candidates": selection["candidates"],
               "remaining_caps": {"calls": identity["max_calls"] - reservations["logical_calls"],
                                  "reserved_input_tokens": identity["max_input_tokens"] - reservations["reserved_input_tokens"]},
               "inherited_usage": inherited,
               "http_status_counts": dict(sum((Counter(r["http_status_counts"]) for r in records), Counter())),
               "error_type_counts": dict(sum((Counter(r["error_type_counts"]) for r in records), Counter())),
               "recovered_retry_rows": sum(r["recovered_retry_rows"] for r in records),
               "terminal_http_status_counts": dict(sum((Counter(r["terminal_http_status_counts"]) for r in records), Counter())),
               "terminal_error_type_counts": dict(sum((Counter(r["terminal_error_type_counts"]) for r in records), Counter()))}
    if not summary["errors_reconciled"]:
        raise ValueError("Source error receipts do not reconcile with excluded attempts")
    recovery = {"rule": RULE, "source": str(source),
                "source_identity_sha256": files["identity.json"],
                "source_manifest_sha256": files["manifest.json"],
                "source_tree_sha256": digest(files),
                "imported_attempts": imported, "excluded_attempts": excluded,
                "summary": summary, "effective_source_config": safe_config(previous)}
    return {"recovery": recovery, "inherited_usage": inherited, "files": files,
            "selection": selection, "source_identity": identity}


def verify_source(recovery, files, *, full):
    source = Path(recovery["source"])
    if full:
        if tree_hashes(source) != files:
            raise ValueError("Recovery source changed (hash mismatch)")
        return
    for name in ("identity.json", "manifest.json", DB_PATH):
        path = source / name
        if not path.is_file() or path.is_symlink() or file_hash(path) != files[name]:
            raise ValueError("Recovery source changed (hash mismatch)")


def checked_recovery_identity(output, base_identity, *, full=False):
    """Bind current code to the migration; missing imports must never trigger calls."""
    output = Path(output)
    stored = read_json(output / "identity.json")
    if "recovery" not in stored:
        return base_identity
    expected = {**base_identity, "recovery": stored["recovery"],
                "inherited_usage": stored["inherited_usage"]}
    if stored != expected:
        raise ValueError("Resume identity changed (selection/config/code/dependencies/shards/caps)")
    recovery = stored["recovery"]
    files = read_json(output / "recovery/source_files.json")
    if digest(files) != recovery["source_tree_sha256"]:
        raise ValueError("Recovery inventory hash mismatch")
    marker = read_json(output / "recovery/complete.json")
    if marker["identity_sha256"] != digest(stored):
        raise ValueError("Recovery incomplete or identity hash mismatch")
    verify_source(recovery, files, full=full)
    state = reservation_state((output / DB_PATH).resolve(), live=True)
    inherited = stored["inherited_usage"]["reservations"]
    if state["limits"] != inherited["limits"] or any(
            state[k] < inherited[k] for k in ("logical_calls", "reserved_input_tokens", "transport_reservations")):
        raise ValueError("Inherited reservations were reset")
    for entry in recovery["imported_attempts"]:
        relative = entry["target_path"]
        path = output / relative / "receipt.json"
        if not path.is_file() or file_hash(path) != marker["receipt_sha256"][relative]:
            raise ValueError("Imported receipt missing or changed; refusing duplicate API use")
        receipt = read_json(path)
        original = path.parent / "source_receipt.json"
        if entry["source_receipt_sha256"]:
            if not original.is_file() or file_hash(original) != entry["source_receipt_sha256"]:
                raise ValueError("Original imported receipt changed (hash mismatch)")
            original_receipt = read_json(original)
            if any(receipt[k] != original_receipt[k] for k in ("episode_hashes", "teacher_hashes", "won", "task_id")):
                raise ValueError("Imported receipt differs from frozen source evidence")
        if (receipt["task_id"] != entry["task_id"] or receipt["attempt"] != entry["attempt"]
                or receipt["episode_dir"] != relative + "/episode"):
            raise ValueError("Imported receipt differs from frozen recovery order")
        if full:
            if (tree_hashes(path.parent / "episode") != receipt["episode_hashes"]
                    or tree_hashes(path.parent / "teacher") != receipt["teacher_hashes"]):
                raise ValueError("Imported evidence changed (hash mismatch)")
    return expected


def _copy_checked(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(target) != expected:
            raise ValueError("Partial recovery copy changed (hash mismatch)")
        return
    temporary = target.with_name(target.name + ".copying")
    shutil.copyfile(source, temporary)
    if file_hash(source) != expected or file_hash(temporary) != expected:
        raise ValueError("Source changed during copy (hash mismatch)")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(target)
    fsync_dir(target.parent)


def recover_full_train(source, output, manifest, freeze, config, *, dry_run=False):
    """Recovery-or-resume. No environment, teacher construction, or API access."""
    from teachers.collect_seeds import usage_summary
    from teachers.full_train import collection_identity, publish_collection

    source, output = separate_paths(source, output)
    # Reject a changed established source before accepting a new snapshot.
    if output.exists():
        stored = read_json(output / "identity.json")
        if stored.get("recovery", {}).get("source") != str(source):
            raise ValueError("Existing output is not a recovery of this source")
        old_files = read_json(output / "recovery/source_files.json")
        verify_source(stored["recovery"], old_files, full=True)
    plan = inspect_source(source, manifest, freeze, config)
    prior = plan["source_identity"]
    base = collection_identity(plan["selection"], manifest, freeze, config, attempts=3,
        shard_count=prior["shard_count"], max_calls=prior["max_calls"], max_input_tokens=prior["max_input_tokens"])
    identity = {**base, "recovery": plan["recovery"], "inherited_usage": plan["inherited_usage"]}
    summary = {**plan["recovery"]["summary"], "source": str(source), "output": str(output),
               "source_identity_sha256": plan["recovery"]["source_identity_sha256"],
               "source_manifest_sha256": plan["recovery"]["source_manifest_sha256"],
               "source_tree_sha256": plan["recovery"]["source_tree_sha256"],
               "recovery_rule": RULE, "dry_run": dry_run, "new_teacher_calls": 0}
    if output.exists():
        if read_json(output / "identity.json") != identity:
            raise ValueError("Recovery identity changed (hash mismatch)")
        checked_recovery_identity(output, base, full=True)
        return {**summary, "action": "resume"}
    if dry_run:
        return {**summary, "action": "classify_only"}
    staging = output.with_name("." + output.name + ".recovering")
    separate_paths(source, staging)
    # Lock outside the source and outside the published directory; an interrupted
    # copy resumes in staging. No collector can see a half-imported collection.
    with locked(output.with_name("." + output.name + ".recovery.lock")):
        if output.exists():
            checked_recovery_identity(output, base, full=True)
            return {**summary, "action": "resume"}
        if staging.exists():
            if read_json(staging / "identity.json") != identity:
                raise ValueError("Partial recovery identity changed")
        else:
            write_json(staging / "identity.json", identity, exclusive=True)
        write_json(staging / "selection.json", plan["selection"])
        write_json(staging / "recovery/source_files.json", plan["files"])
        # Preserve source metadata/receipts byte-for-byte; imported receipts below
        # are new records with explicit provenance, never in-place source edits.
        for name in ("identity.json", "manifest.json", "selection.json", "provenance.json"):
            if name in plan["files"]:
                _copy_checked(source / name, staging / "recovery/source" / name, plan["files"][name])
        _copy_checked(source / DB_PATH, staging / DB_PATH, plan["files"][DB_PATH])
        receipts, receipt_hashes = [], {}
        evidence = attempt_files(plan["files"])
        for entry in plan["recovery"]["imported_attempts"]:
            src, relative = entry["source_path"], entry["target_path"]
            target = staging / relative
            hashes = evidence[src]
            for name, sha in hashes.items():
                destination = ("source_receipt.json" if name == "receipt.json" else
                               "source_started.json" if name == "started.json" else name)
                _copy_checked(source / src / name, target / destination, sha)
            ep_hashes, teacher_hashes = artifact_hashes(target / "episode"), artifact_hashes(target / "teacher")
            if ep_hashes != _under(hashes, "episode") or teacher_hashes != _under(hashes, "teacher"):
                raise ValueError("Copied attempt hash mismatch")
            if (target / "source_receipt.json").exists():
                receipt = read_json(target / "source_receipt.json")
            else:
                task = resolve_task(manifest, freeze, entry["task_id"])
                episode = read_json(target / "episode/episode.json")
                receipt = {k: task[k] for k in ("task_id", "group_id", "task_type")}
                receipt.update(parent_task_id=task["task_id"], parent_seed_ids=[],
                    selection_parent_task_id=task["task_id"], source="real_environment_teacher_rollout",
                    won=entry["won"], success=episode["success"], status=episode["status"],
                    usage=usage_summary(target / "teacher/usage.jsonl"), metrics=episode.get("metrics"),
                    rejection=None if entry["won"] else "no_verified_environment_win")
            receipt.update(attempt=entry["attempt"], episode_dir=relative + "/episode",
                episode_hashes=ep_hashes, teacher_hashes=teacher_hashes,
                recovery_provenance={"rule": RULE, "source_collection": str(source),
                    **{k: entry[k] for k in ("source_path", "source_attempt", "source_attempt_sha256",
                                            "source_receipt_sha256")}})
            write_json(target / "started.json", {**read_json(target / "source_started.json"),
                                                "attempt": entry["attempt"]})
            write_json(target / "receipt.json", receipt)
            receipts.append(receipt)
            receipt_hashes[relative] = file_hash(target / "receipt.json")
        for index in range(identity["shard_count"]):
            task_ids = set(plan["selection"]["candidate_order"][index::identity["shard_count"]])
            write_json(staging / "shards" / f"{index}.json",
                {"shard_index": index, "status": "collecting",
                 "attempts": [r for r in receipts if r["task_id"] in task_ids]})
        publish_collection(staging, identity, plan["inherited_usage"]["reservations"])
        if not (staging / "provenance.json").exists():
            write_provenance(staging, command="recover_full_train", config=config,
                details={"recovery_rule": RULE, "source_tree_sha256": digest(plan["files"]),
                         "new_teacher_calls": 0, "new_environment_steps": 0})
        verify_source(plan["recovery"], plan["files"], full=True)
        write_json(staging / "recovery/complete.json",
                   {"identity_sha256": digest(identity), "receipt_sha256": receipt_hashes})
        staging.rename(output)
        fsync_dir(output.parent)
    return {**summary, "action": "copied"}
