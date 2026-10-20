"""Offline migration tests: legacy paid attempts must never invoke transport again."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.seed_evidence import artifact_hashes, load_verified_collection
from sed.storage import digest, write_json
from teachers.capped import SharedReservations
from teachers.collect_seeds import collect
from teachers.full_train import preflight
from teachers.full_train_recovery import (
    DB_PATH, TRANSPORT, UNKNOWN, VALID, classify_attempt, read_json,
    recover_full_train, reservation_state, tree_hashes,
)
from teachers.openai_teacher import TeacherCallError
from test_full_train import factories, full_bank
from tools.recover_full_train import main


def ledger_write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def refresh_source(source):
    collection = read_json(source / "manifest.json")
    receipts = []
    for receipt in collection["attempts"]:
        directory = source / receipt["episode_dir"]
        receipt = read_json(directory.parent / "receipt.json")
        receipt["episode_hashes"] = artifact_hashes(directory)
        receipt["teacher_hashes"] = artifact_hashes(directory.parent / "teacher")
        write_json(directory.parent / "receipt.json", receipt)
        receipts.append(receipt)
    collection["attempts"] = receipts
    collection["verified"] = [r for r in receipts if r["won"]]
    collection["budget"] = reservation_state(source / DB_PATH)
    write_json(source / "manifest.json", collection)


@pytest.fixture
def legacy(full_bank, tmp_path):
    root, manifest, freeze, selection, config = full_bank
    source = tmp_path / "source"
    calls = tmp_path / "paid.jsonl"
    ids = selection["candidate_order"]
    seen = {}
    def environment(task):
        tid = task["task_id"]
        seen[tid] = seen.get(tid, 0) + 1
        def backend(path, seed):
            result = FakeBackend(path, seed)
            result.win_at = 1 if tid == ids[3] or tid == ids[0] and seen[tid] == 3 else 99
            return result
        return Environment(root, max_episode_steps=1, backend_factory=backend)
    _, teacher = factories(full_bank, calls)
    result = collect(selection, manifest, freeze, config, source,
                     env_factory=environment, teacher_factory=teacher)
    # Mimic 549503d's paid, recovered rate limit with no logical_call_id.
    receipt = result["verified"][-1]
    path = source / receipt["episode_dir"]
    ledger = path.parent / "teacher/usage.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    original = deepcopy(rows[0])
    SharedReservations(source / DB_PATH).reserve(
        logical_id=original["logical_call_id"], request_id="legacy-retry",
        input_tokens=15, task_id=receipt["task_id"], usage_ledger=ledger)
    failed_start = dict(original, call_id="legacy-retry")
    failed_end = dict(failed_start, status="failed", http_status=429,
                      error_type="RateLimitError", will_retry=True, retryable=True)
    rows = [failed_start, failed_end, *[dict(r, attempt=2) for r in rows]]
    ledger_write(ledger, rows)
    # Two transport failures used up old task attempt slots.
    for tid, number in ((ids[0], 1), (ids[1], 2)):
        receipt = next(r for r in result["attempts"] if (r["task_id"], r["attempt"]) == (tid, number))
        directory = source / receipt["episode_dir"]
        episode = read_json(directory / "episode.json")
        episode.update(complete=False, status="error", success="unknown",
                       error_type="policy_error", teacher_transport_failure=True)
        write_json(directory / "episode.json", episode)
        receipt.update(status="error", success="unknown", won=False)
        write_json(directory.parent / "receipt.json", receipt)
        ledger = directory.parent / "teacher/usage.jsonl"
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        rows[-1].update(status="failed", error_type="RateLimitError", http_status=429,
                        will_retry=False, usage=None, retryable=tid == ids[0])
        ledger_write(ledger, rows)
    for ledger in source.glob("attempts/*/teacher/usage.jsonl"):
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        for row in rows:
            row.pop("logical_call_id", None)
        ledger_write(ledger, rows)
    refresh_source(source)
    return source, calls


@pytest.mark.parametrize("failure,expected", [
    ({"status": "failed", "http_status": 429, "retryable": False}, TRANSPORT),
    ({"status": "failed", "http_status": 429, "retryable": True}, TRANSPORT),
    ({"status": "failed", "error_type": "APITimeoutError"}, TRANSPORT),
    ({"status": "failed", "http_status": 503}, TRANSPORT),
    ({"status": "failed", "error_type": "APIConnectionError"}, TRANSPORT),
    ({"status": "incomplete", "error_type": "IncompleteResponse"}, TRANSPORT),
    ({"status": "failed", "will_retry": True}, TRANSPORT),
    ({"status": "started"}, UNKNOWN),
])
def test_classifies_terminal_and_unknown_rows(legacy, failure, expected):
    source, _ = legacy
    receipt = read_json(source / "manifest.json")["verified"][-1]
    directory = (source / receipt["episode_dir"]).parent
    ledger = directory / "teacher/usage.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[-1].update(failure)
    ledger_write(ledger, rows)
    result, _ = classify_attempt(directory)
    assert result["classification"] == expected
    if expected == TRANSPORT:
        assert sum(result["terminal_error_type_counts"].values()) >= 1


def test_classifies_valid_loss_win_retries_and_episode_errors(legacy):
    source, _ = legacy
    receipts = read_json(source / "manifest.json")["attempts"]
    for won in (False, True):
        receipt = next(r for r in receipts if r["status"] == "ok" and r["won"] == won)
        directory = (source / receipt["episode_dir"]).parent
        assert classify_attempt(directory)[0]["classification"] == VALID
    directory = (source / receipts[-1]["episode_dir"]).parent
    result, _ = classify_attempt(directory)
    assert result["classification"] == VALID and result["recovered_retry_rows"] == 1
    episode = read_json(directory / "episode/episode.json")
    episode.update(status="error", complete=False, error_type="TeacherCallError")
    write_json(directory / "episode/episode.json", episode)
    assert classify_attempt(directory)[0]["classification"] == TRANSPORT
    episode.update(status="started", error_type=None)
    write_json(directory / "episode/episode.json", episode)
    assert classify_attempt(directory)[0]["classification"] == UNKNOWN


def test_terminal_failure_cannot_be_forgiven_by_later_success(legacy):
    source, _ = legacy
    receipt = read_json(source / "manifest.json")["verified"][-1]
    directory = (source / receipt["episode_dir"]).parent
    ledger = directory / "teacher/usage.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[1]["will_retry"] = False
    ledger_write(ledger, rows)
    assert classify_attempt(directory)[0]["classification"] == TRANSPORT
    rows[1]["will_retry"] = True
    for row in rows[2:]:
        row["attempt"] = 1  # A different logical call even with the same cache key.
    ledger_write(ledger, rows)
    assert classify_attempt(directory)[0]["classification"] == TRANSPORT


def test_unreceipted_complete_and_started_attempts(full_bank, legacy, tmp_path):
    source, _ = legacy
    _, manifest, freeze, _, config = full_bank
    collection = read_json(source / "manifest.json")
    win = collection["verified"][-1]
    failure = next(r for r in collection["attempts"] if r["status"] == "error")
    for receipt in (win, failure):
        (source / receipt["episode_dir"]).parent.joinpath("receipt.json").unlink()
    collection["attempts"] = [r for r in collection["attempts"] if r not in (win, failure)]
    collection["verified"] = [r for r in collection["verified"] if r != win]
    write_json(source / "manifest.json", collection)
    episode_path = source / failure["episode_dir"] / "episode.json"
    episode = read_json(episode_path)
    episode["status"] = "started"
    write_json(episode_path, episode)
    output = tmp_path / "r2"
    summary = recover_full_train(source, output, manifest, freeze, config)
    assert summary["classification_counts"] == {VALID: 8, TRANSPORT: 1, UNKNOWN: 1}
    assert summary["source_error_receipts"] == 1 and summary["excluded_without_error_receipt"] == 1
    imported = next(r for r in read_json(output / "manifest.json")["verified"] if r["task_id"] == win["task_id"])
    assert imported["recovery_provenance"]["source_receipt_sha256"] is None
    assert imported["won"] is True


@pytest.mark.parametrize("new_win", [False, True])
def test_import_renumber_resume_bound_first_win_and_consumers(full_bank, legacy, tmp_path, new_win):
    source, paid = legacy
    _, manifest, freeze, selection, config = full_bank
    output = tmp_path / "seeds_full_train_r2"
    before = tree_hashes(source)
    summary = recover_full_train(source, output, manifest, freeze, config)
    assert summary["classification_counts"] == {VALID: 8, TRANSPORT: 2, UNKNOWN: 0}
    assert summary["errors_reconciled"] and summary["source_error_receipts"] == 2
    identity = read_json(output / "identity.json")
    assert identity["recovery"]["rule"] == "full_train_valid_complete_copy_v1"
    assert len(identity["recovery"]["imported_attempts"]) == 8
    assert len(identity["recovery"]["excluded_attempts"]) == 2
    assert identity["inherited_usage"]["settled_calls"] == 8
    assert reservation_state(output / DB_PATH) == reservation_state(source / DB_PATH)
    assert not any(p.is_symlink() for p in output.rglob("*"))
    for entry in identity["recovery"]["imported_attempts"]:
        receipt = read_json(output / entry["target_path"] / "receipt.json")
        assert receipt["attempt"] == entry["attempt"]
        assert receipt["recovery_provenance"]["source_attempt"] == entry["source_attempt"]
        for subdir in ("episode", "teacher"):
            assert artifact_hashes(output / entry["target_path"] / subdir) == artifact_hashes(
                source / entry["source_path"] / subdir)
        source_receipt = source / entry["source_path"] / "receipt.json"
        assert source_receipt.read_bytes() == (output / entry["target_path"] / "source_receipt.json").read_bytes()
        assert source_receipt.stat().st_ino != (output / entry["target_path"] / "source_receipt.json").stat().st_ino
    migrated = read_json(output / "manifest.json")
    ids = selection["candidate_order"]
    first = [r for r in migrated["attempts"] if r["task_id"] == ids[0]]
    assert [r["attempt"] for r in first] == [1, 2]
    assert [r["recovery_provenance"]["source_attempt"] for r in first] == [2, 3]
    # No transport construction or environment reset for ANY imported attempt,
    # including losses and a recovered legacy retry. Only task 1 slot 3 is new.
    new_calls = tmp_path / "new.jsonl"
    environment, teacher = factories(full_bank, new_calls, losing=() if new_win else ids)
    def checked_env(task):
        assert task["task_id"] == ids[1]
        return environment(task)
    def checked_teacher(task, cache, ledger):
        assert task["task_id"] == ids[1] and cache.parent.parent.name.endswith("-3")
        return teacher(task, cache, ledger)
    preflight(selection, manifest, freeze, config, output, resume=True)
    result = collect(selection, manifest, freeze, config, output, resume=True,
                     env_factory=checked_env, teacher_factory=checked_teacher)
    assert len(new_calls.read_text().splitlines()) == 1
    assert len(paid.read_text().splitlines()) == 10
    assert result["status"] == "complete" and result["K"] == 2 + int(new_win)
    assert result["budget"]["logical_calls"] == summary["inherited_usage"]["reservations"]["logical_calls"] + 1
    assert all(r["attempt"] <= 3 for r in result["attempts"])
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed/imported attempts must never call a teacher or reset an environment")
    assert collect(selection, manifest, freeze, config, output, resume=True,
                   env_factory=forbidden, teacher_factory=forbidden) == result
    assert recover_full_train(source, output, manifest, freeze, config)["action"] == "resume"
    assert tree_hashes(source) == before
    assert load_verified_collection(output, manifest, freeze)[0] == result
    from baselines.agentgym.data import seed_rows
    assert len(seed_rows(output, config)) == result["K"]
    from train.dose_grid import prepare_grid
    from test_r03_sets import TinyTokenizer
    tokenizer = TinyTokenizer()
    tokenizer.chat_template = "fake Gemma OFF template"
    template = read_json("configs/seedsft_dose_grid_full_train.json")
    assert template["runs"][0]["verified"] == "data/verified/seeds_full_train_r2"
    template["runs"][0]["verified"] = str(output)
    assert prepare_grid(template, tokenizer, manifest, freeze)["runs"][0]["K"] == result["K"]


@pytest.mark.parametrize("change", ["source_receipt", "source_blob", "source_identity", "missing_import"])
def test_changed_source_or_missing_import_refuses(full_bank, legacy, tmp_path, change):
    source, _ = legacy
    _, manifest, freeze, selection, config = full_bank
    output = tmp_path / "r2"
    recover_full_train(source, output, manifest, freeze, config)
    entry = read_json(output / "identity.json")["recovery"]["imported_attempts"][0]
    if change == "source_receipt":
        path = source / entry["source_path"] / "receipt.json"
    elif change == "source_blob":
        path = next((source / entry["source_path"] / "episode/blobs").rglob("*.json"))
    elif change == "source_identity":
        path = source / "identity.json"
    else:
        path = output / entry["target_path"] / "receipt.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed|mismatch"):
        recover_full_train(source, output, manifest, freeze, config)
    if change == "missing_import":
        path.unlink()
        with pytest.raises(ValueError, match="refusing duplicate"):
            collect(selection, manifest, freeze, config, output, resume=True)


def test_refuses_source_overlap_and_dry_run_is_read_only(full_bank, legacy, tmp_path):
    source, _ = legacy
    _, manifest, freeze, _, config = full_bank
    before = tree_hashes(source)
    for output in (source, source / "r2", source.parent):
        with pytest.raises(ValueError, match="source"):
            recover_full_train(source, output, manifest, freeze, config)
    output = tmp_path / "dry-output"
    summary = recover_full_train(source, output, manifest, freeze, config, dry_run=True)
    assert summary["new_teacher_calls"] == 0 and summary["action"] == "classify_only"
    assert not output.exists() and tree_hashes(source) == before
    with pytest.raises(ValueError, match="source"):
        main(["--source", str(source), "--output", str(output), "--dry-run",
              "--summary", str(source / "summary.json")])


@pytest.mark.parametrize("limit", ["max_calls", "max_input_tokens"])
def test_caps_are_inherited_without_refunds(full_bank, legacy, tmp_path, limit):
    source, _ = legacy
    _, manifest, freeze, selection, config = full_bank
    # Restrict the fake source's cap to exactly the calls already reserved.
    import sqlite3
    state = reservation_state(source / DB_PATH)
    counter = "logical_calls" if limit == "max_calls" else "reserved_input_tokens"
    cap = state[counter]
    state["limits"][limit] = cap
    with sqlite3.connect(source / DB_PATH) as db:
        db.execute("UPDATE budget SET value=? WHERE id=1", (json.dumps(state),))
    identity = read_json(source / "identity.json")
    identity[limit] = cap
    write_json(source / "identity.json", identity)
    collection = read_json(source / "manifest.json")
    collection["identity_sha256"] = digest(identity)
    collection["budget"] = state
    write_json(source / "manifest.json", collection)
    output = tmp_path / "r2"
    summary = recover_full_train(source, output, manifest, freeze, config)
    remaining = "calls" if limit == "max_calls" else "reserved_input_tokens"
    assert summary["remaining_caps"][remaining] == 0
    new_calls = tmp_path / "forbidden.jsonl"
    env, teacher = factories(full_bank, new_calls)
    with pytest.raises(TeacherCallError):
        collect(selection, manifest, freeze, config, output, resume=True,
                env_factory=env, teacher_factory=teacher, **{limit: cap})
    assert not new_calls.exists()
    result = read_json(output / "manifest.json")
    assert result["status"] == "budget_exhausted"
    assert result["budget"][counter] == cap
    assert reservation_state(source / DB_PATH) == state
    with pytest.raises(ValueError, match="identity changed"):
        collect(selection, manifest, freeze, config, output, resume=True, **{limit: cap + 1})


def test_recovery_preserves_32_shard_ownership(full_bank, legacy, tmp_path):
    source, _ = legacy
    _, manifest, freeze, selection, config = full_bank
    identity = read_json(source / "identity.json")
    identity["shard_count"] = 32
    write_json(source / "identity.json", identity)
    collection = read_json(source / "manifest.json")
    collection["identity_sha256"] = digest(identity)
    write_json(source / "manifest.json", collection)
    output = tmp_path / "r2"
    recover_full_train(source, output, manifest, freeze, config)
    assert len(list((output / "shards").glob("*.json"))) == 32
    preflight(selection, manifest, freeze, config, output, shard_count=32, resume=True)
    calls = tmp_path / "new-calls.jsonl"
    env, teacher = factories(full_bank, calls)
    for index in range(4):
        collect(selection, manifest, freeze, config, output, shard_count=32,
                shard_index=index, resume=True, env_factory=env, teacher_factory=teacher)
    assert len(calls.read_text().splitlines()) == 1
    assert read_json(output / "manifest.json")["status"] == "complete"


def test_interrupted_copy_can_resume_without_exposing_partial_collection(full_bank, legacy, tmp_path, monkeypatch):
    source, _ = legacy
    _, manifest, freeze, _, config = full_bank
    import teachers.full_train_recovery as recovery
    original = recovery._copy_checked
    count = 0
    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 8:
            raise KeyboardInterrupt
        return original(*args)
    output = tmp_path / "r2"
    with monkeypatch.context() as patch:
        patch.setattr(recovery, "_copy_checked", interrupted)
        with pytest.raises(KeyboardInterrupt):
            recover_full_train(source, output, manifest, freeze, config)
    assert not output.exists()
    assert recover_full_train(source, output, manifest, freeze, config)["action"] == "copied"


def test_r2_launcher_fixed_shards_and_offline_preflight(tmp_path):
    fake, ledger = tmp_path / "python", tmp_path / "calls.jsonl"
    fake.write_text("#!" + sys.executable + "\n" + '''
import fcntl, json, os, sys
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
with open(os.environ["INVOCATIONS"], "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(sys.argv[1:]) + "\\n")
''')
    fake.chmod(0o755)
    env = dict(os.environ, PROJECT_ROOT=str(Path.cwd()), ENV_PYTHON=str(fake),
               COLLECTION_ROOT=str(tmp_path / "collections"), LOG_ROOT=str(tmp_path / "logs"),
               RECOVERY_SOURCE=str(tmp_path / "source"), INVOCATIONS=str(ledger))
    command = ["bash", "scripts/rai_collect_full_train_r2.sh"]
    result = subprocess.run([*command, "--preflight"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 2 and rows[0][0] == "tools/recover_full_train.py"
    assert "--preflight" in rows[1] and "--resume" in rows[1]
    assert rows[1][rows[1].index("--shard-count") + 1] == "32"
    assert rows[1][rows[1].index("--output") + 1].endswith("seeds_full_train_r2")
    assert subprocess.run([*command, "--workers", "8"], env=env, capture_output=True).returncode == 2
    assert len(ledger.read_text().splitlines()) == 2
    assert "data/verified/seeds_full_train_r2" in Path("scripts/hpg_baseline_cells.slurm").read_text()
    assert "seeds_full_train_r2" in Path("scripts/rai_collect_adftp.sh").read_text()
