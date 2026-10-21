"""Offline in-cell amendment: fake teacher/environment, real durable evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.smartad import collect as sm
from sed.storage import BlobStore, digest, write_json
from teachers import smartad_resume as amendment
from teachers.chain_recovery import (check_imports, collection_accounting, credit_file,
                                     read_json, recover_chain_collection)
from teachers.full_train_recovery import tree_hashes
from teachers.openai_teacher import TeacherCallError
from test_chain_recovery import quota_factory, long_environment
from test_smartad import Raw, factory, setup_source
from test_stage2b_seeds import seed_bank


@pytest.fixture
def cell(seed_bank, tmp_path, monkeypatch):
    monkeypatch.setenv("SED_HASH_WORKERS", "1")
    monkeypatch.setattr("teachers.openai_teacher.utc_now", lambda: "2026-10-20T22:45:30+00:00")
    support, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=2)
    old, new, calls = tmp_path / "old", tmp_path / "r2", []
    kw = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env)
    sm.collect(support, manifest, freeze, config, old, shard_count=20, shard_index=0,
               teacher_factory=factory(calls), **kw)
    with pytest.raises(TeacherCallError):
        sm.collect(support, manifest, freeze, config, old, shard_count=20, shard_index=1,
            teacher_factory=quota_factory(factory(calls), [], fail=True, fail_after=30),
            **{**kw, "env_factory": long_environment(config)})
    recover_chain_collection(old, new, "smartad", manifest, freeze, config)
    identity = read_json(new / "identity.json")
    monkeypatch.setattr(amendment, "CELL", new.name)
    monkeypatch.setattr(amendment, "CELL_IDENTITY_SHA256", digest(identity))
    slot, retry = next(iter(identity["chain_recovery"]["retry_directories"].items()))
    return dict(support=support, manifest=manifest, freeze=freeze, config=config, old=old,
                new=new, calls=calls, kw=kw, identity=identity, slot=slot, retry=retry)


def run(cell, **kwargs):
    return sm.collect(cell["support"], cell["manifest"], cell["freeze"], cell["config"], cell["new"],
        teacher_factory=factory(cell["calls"]), **{**cell["kw"], **kwargs})


def test_dry_run_no_writes_and_step_zero_restart(cell):
    root, old = cell["new"], cell["old"]
    before, source = tree_hashes(root), tree_hashes(old)
    summary = amendment.amend(root, resume=True, dry_run=True)
    assert tree_hashes(root) == before
    assert summary["imported"] == summary["interrupted_to_restart"] == 1
    assert summary["completed"] == 0 and summary["never_started"] == 18
    assert summary["extra_call_upper_bound"] == 41 and summary["remaining_call_upper_bound"] == 19 * 41
    record = summary["records"][0]
    assert record["source_old_attempts"] == 31 and record["old_attempts"] == 30
    assert record["additional_quota_credit"] == 0
    slot_before = tree_hashes(root / cell["slot"])
    imports = {p: tree_hashes(root / p) for p in cell["identity"]["chain_recovery"]["final_candidates"]}
    calls = len(cell["calls"])
    # 32 shards cover every untouched slot. A second pass resolves deferred 01.
    for _ in range(2):
        for index in range(32):
            run(cell, shard_count=32, shard_index=index)
    assert (root / "candidates.json").exists()
    assert tree_hashes(root / record["archive"]) == slot_before
    assert all(tree_hashes(root / p) == hashes for p, hashes in imports.items())
    assert tree_hashes(old) == source
    assert read_json(root / "identity.json") == cell["identity"]
    fresh = root / record["fresh"]
    counter = read_json(fresh / "teacher/counter.json")
    assert counter["attempts"] == 1 and counter["identity"]["cap"] == 41
    assert counter["identity"]["sampling"] == read_json(
        root / record["archive"] / "teacher/counter.json")["identity"]["sampling"]
    assert not (fresh / "teacher/cap_credit.json").exists()
    episode = read_json(fresh / "episode/episode.json")
    assert episode["complete"] is True and episode["metrics"]["environment_steps"] == 1
    assert len(cell["calls"]) - calls == 17  # 2 trusted first slots reused.
    accounting = collection_accounting(root, cell["identity"])
    assert accounting["inherited_usage"]["http_calls"] == 32
    assert accounting["new_teacher_calls"] == 17
    assert accounting["amendment_extra_call_upper_bound"] == 41
    check_imports(root)
    frozen, calls = tree_hashes(root), len(cell["calls"])
    amendment.amend(root, resume=True)
    assert tree_hashes(root) == frozen and len(cell["calls"]) == calls
    run(cell)
    assert len(cell["calls"]) == calls


@pytest.mark.parametrize("failure", ["wrong_cell", "no_resume", "unclassified", "completed",
                                    "different_latest_outcome", "credit_changed", "pending"])
def test_refusals_before_writes(cell, monkeypatch, failure):
    root, retry = cell["new"], cell["retry"]
    if failure == "wrong_cell":
        monkeypatch.setattr(amendment, "CELL_IDENTITY_SHA256", "other")
    elif failure == "unclassified":
        directory = root / "work/unknown/candidate-02"
        write_json(directory / "started.json", {})
    elif failure == "completed":
        write_json(root / cell["slot"] / "candidate.json", {})
    elif failure == "different_latest_outcome":
        ledger = root / retry / "teacher/usage.jsonl"
        rows = ledger.read_text().splitlines()
        rows[-1] = '{"status": "failed", "call_id": "unknown"}'
        ledger.write_text("\n".join(rows) + "\n")
    elif failure == "credit_changed":
        path = credit_file(root, retry + "/teacher/counter.json")
        value = read_json(path)
        write_json(path, {**value, "credit": 2})
    elif failure == "pending":
        path = root / retry / "teacher/counter.json"
        write_json(path, {**read_json(path), "pending": True})
    before = tree_hashes(root)
    with pytest.raises((ValueError, KeyError)):
        amendment.amend(root, resume=failure != "no_resume")
    assert tree_hashes(root) == before


def exhaust_r2(cell, monkeypatch, *, parse_rejection=False):
    # Reproduce the real bug: inherited floor 30 + 11 fresh calls = cap 41.
    from teachers.capped import BudgetExceeded
    make = factory(cell["calls"])
    def teacher(*args):
        result = make(*args)
        responses = result.teacher.client.client.responses.with_raw_response
        create, first = responses.create, True
        def request(**kwargs):
            nonlocal first
            response = create(**kwargs)
            if first and parse_rejection:
                response = Raw("Thought: no action")
            first = False
            return response
        responses.create = request
        return result
    with monkeypatch.context() as patch:
        patch.setattr(amendment, "CELL_IDENTITY_SHA256", "disabled")
        with pytest.raises(BudgetExceeded):
            sm.collect(cell["support"], cell["manifest"], cell["freeze"], cell["config"], cell["new"],
                teacher_factory=teacher, shard_count=20, shard_index=1,
                **{**cell["kw"], "env_factory": long_environment(cell["config"])})


@pytest.mark.parametrize("parse_rejection", [False, True])
def test_inherited_shortened_cap_retry_is_spent_void_and_restarted(cell, monkeypatch, parse_rejection):
    exhaust_r2(cell, monkeypatch, parse_rejection=parse_rejection)
    before = tree_hashes(cell["new"])
    source = tree_hashes(cell["old"])
    summary = amendment.amend(cell["new"], resume=True, dry_run=True)
    assert summary["refused"] == 0 and summary["interrupted_to_restart"] == 1 and summary["ready"]
    assert summary["extra_call_upper_bound"] == 41
    record = summary["records"][0]
    assert record["r2_requests"] == record["recovery_remaining_cap"] == 11
    assert record["old_attempts"] == 41
    assert record["eligibility"] == "r1_quota_void_then_r2_inherited_cap_exhausted"
    spent = summary["spent_and_void_usage"]
    assert spent["r1"]["http_calls"] == 31 and spent["r1"]["quota_void_calls"] == 1
    assert spent["r2"]["http_calls"] == spent["r2"]["logical_calls"] == 11
    assert spent["r2"]["status_counts"] == ({"success": 10, "rejected": 1} if parse_rejection else {"success": 11})
    assert spent["r2"]["known_billed_tokens"] == {"input_tokens": 132, "output_tokens": 88, "reasoning_tokens": 0}
    assert spent["r2"]["environment_steps"] == 11 - int(parse_rejection)
    assert [a["disposition"] for a in record["void_attempts"]] == ["spent_and_void"] * 2
    assert [a["phase"] for a in record["void_attempts"]] == ["r1", "r2"]
    assert tree_hashes(cell["new"]) == before
    prior = collection_accounting(cell["new"], cell["identity"])
    amendment.amend(cell["new"], resume=True)
    assert tree_hashes(cell["new"] / record["archive"]) == record["files"]
    assert tree_hashes(cell["old"]) == source
    accounting = collection_accounting(cell["new"], cell["identity"])
    assert accounting["inherited_usage"] == prior["inherited_usage"]
    assert accounting["new_usage"] == prior["new_usage"]
    assert accounting["smartad_resume_amendment"]["spent_and_void_usage"] == spent
    assert accounting["new_teacher_calls"] == 11
    assert amendment.amend(cell["new"], resume=True)["plan_sha256"] == digest(
        accounting["smartad_resume_amendment"])
    assert amendment.read_plan(cell["new"])["spent_and_void_usage"] == spent
    run(cell, shard_count=20, shard_index=1)
    counter = read_json(cell["new"] / record["fresh"] / "teacher/counter.json")
    assert counter["attempts"] == 1 and counter["identity"]["cap"] == 41
    assert not (cell["new"] / record["fresh"] / "teacher/cap_credit.json").exists()
    assert collection_accounting(cell["new"], cell["identity"])["new_teacher_calls"] == 12
    frozen = tree_hashes(cell["new"])
    amendment.amend(cell["new"], resume=True)
    assert tree_hashes(cell["new"]) == frozen


@pytest.mark.parametrize("failure", ["other_terminal_error", "complete", "known_success", "other_event",
    "counter_not_exhausted", "changed_cap", "changed_protocol", "halted", "unknown_outcome",
    "other_rejection", "unresolved_start", "missing_start", "wrong_task", "another_retry", "receipt"])
def test_other_r2_outcomes_remain_refused(cell, monkeypatch, failure):
    from teachers.chain_recovery import ledger_rows
    exhaust_r2(cell, monkeypatch)
    root, retry = cell["new"], cell["new"] / cell["retry"]
    counter_path = retry / "teacher/counter.json"
    counter = read_json(counter_path)
    episode_path = retry / "episode/episode.json"
    episode = read_json(episode_path)
    ledger_path = retry / "teacher/usage.jsonl"
    rows = ledger_rows(ledger_path)
    if failure in {"other_terminal_error", "other_event"}:
        events_path = retry / "episode/events.jsonl"
        events = ledger_rows(events_path)
        if failure == "other_terminal_error":
            blobs = BlobStore(retry / "episode/blobs")
            verification = blobs.get(events[-1]["verification_ref"])
            verification["audit"]["error_type"] = "CollectorHalted"
            events[-1]["verification_ref"] = blobs.put(verification)
        else:
            events[-1]["error_type"] = "environment_error"
        import json
        events_path.write_text("".join(json.dumps(r) + "\n" for r in events))
    elif failure in {"complete", "known_success"}:
        episode["complete" if failure == "complete" else "success"] = True
        write_json(episode_path, episode)
    elif failure in {"counter_not_exhausted", "changed_cap", "changed_protocol", "halted"}:
        if failure == "counter_not_exhausted": counter["attempts"] -= 1
        if failure == "changed_cap": counter["identity"]["cap"] += 1
        if failure == "changed_protocol": counter["identity"]["sampling"]["temperature"] = 0.5
        if failure == "halted": counter["halted"] = True
        write_json(counter_path, counter)
    elif failure == "another_retry":
        write_json(retry.parent / "attempt-2/started.json", {})
    elif failure == "receipt":
        write_json(retry / "candidate.json", {})
    else:
        row = next(r for r in reversed(rows) if r["status"] == "success")
        if failure == "unknown_outcome": row["status"] = "failed"
        if failure == "other_rejection": row.update(status="rejected", error_type="OtherError")
        if failure == "unresolved_start": rows.remove(row)
        if failure == "missing_start":
            rows.remove(next(r for r in rows if r["status"] == "started" and r["call_id"] == row["call_id"]))
        if failure == "wrong_task": row["task_id"] = "unknown"
        import json
        ledger_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = tree_hashes(root)
    with pytest.raises(ValueError):
        amendment.amend(root, resume=True)
    assert tree_hashes(root) == before


def test_archive_crash_is_resumable_without_second_credit(cell, monkeypatch):
    root = cell["new"]
    original, moved = Path.rename, 0
    def crash(path, target):
        nonlocal moved
        result = original(path, target)
        moved += 1
        if moved == 2:
            raise KeyboardInterrupt
        return result
    monkeypatch.setattr(Path, "rename", crash)
    with pytest.raises(KeyboardInterrupt):
        amendment.amend(root, resume=True)
    plan = read_json(root / amendment.PLAN)
    assert not (root / amendment.COMPLETE).exists()
    monkeypatch.setattr(Path, "rename", original)
    amendment.amend(root, resume=True)
    assert read_json(root / amendment.PLAN) == plan
    before = tree_hashes(root)
    amendment.amend(root, resume=True)
    assert tree_hashes(root) == before
    record = plan["records"][0]
    assert tree_hashes(root / record["archive"]) == record["files"]
    # Tampering must not lead to another sample or a new cap.
    write_json(root / record["archive"] / "started.json", {})
    with pytest.raises(ValueError, match="archived attempt"):
        amendment.amend(root, resume=True)


def test_repeated_recovery_guard_stays(cell, tmp_path):
    with pytest.raises(ValueError, match="already recovered"):
        recover_chain_collection(cell["new"], tmp_path / "r3", "smartad", cell["manifest"],
                                 cell["freeze"], cell["config"])


def test_unknown_transport_failure_still_aborts_shard(cell, monkeypatch):
    # Explicit design: do not spend across tasks after unresolved infrastructure failure.
    monkeypatch.setattr(amendment, "CELL_IDENTITY_SHA256", "disabled")
    calls = len(cell["calls"])
    with pytest.raises(TeacherCallError):
        sm.collect(cell["support"], cell["manifest"], cell["freeze"], cell["config"], cell["new"],
            teacher_factory=factory(cell["calls"], transport_fail=True), **cell["kw"])
    assert len(cell["calls"]) == calls + 1
    assert not list((cell["new"] / "work").glob("*/candidate-04/candidate.json"))


def test_bound_and_protocol_refused_before_archiving(cell, monkeypatch):
    before = tree_hashes(cell["new"])
    monkeypatch.setattr(amendment, "MAX_RESTARTS", 0)
    with pytest.raises(ValueError, match="exceeds"):
        amendment.amend(cell["new"], resume=True)
    assert tree_hashes(cell["new"]) == before
    monkeypatch.setattr(amendment, "MAX_RESTARTS", 23)
    cell["config"]["environment"]["seed"] += 1
    with pytest.raises(ValueError, match="frozen"):
        run(cell)
    assert tree_hashes(cell["new"]) == before


def test_offline_cli_never_launches_or_calls(cell, capsys, monkeypatch):
    import subprocess
    from tools.amend_smartad_resume import main
    def forbidden(*args, **kwargs):
        pytest.fail("Offline classification must not launch anything")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    before = tree_hashes(cell["new"])
    assert main(["--output", str(cell["new"]), "--resume", "--dry-run"]) == 0
    assert '"interrupted_to_restart": 1' in capsys.readouterr().out
    assert tree_hashes(cell["new"]) == before


def test_code_upgrade_keeps_identity_and_freezes_protocol(cell, monkeypatch):
    from teachers.chain_recovery import recovery_identity
    amendment.amend(cell["new"], resume=True)
    proposed = {k: v for k, v in cell["identity"].items() if k != "chain_recovery"}
    proposed["code_sha256"] = {"amendment": "new-code"}
    assert recovery_identity(cell["new"], proposed) == cell["identity"]
    assert len(list((cell["new"] / "resume_versions").glob("*.json"))) == 1
    with pytest.raises(ValueError, match="identity/code/config"):
        recovery_identity(cell["new"], {**proposed, "per_cell_call_cap": 1000000})
    assert read_json(cell["new"] / "identity.json") == cell["identity"]


def forbid_classification(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Completed amendment must not classify candidates again")
    monkeypatch.setattr(amendment, "classify", forbidden)


def test_late_shard_verifies_completed_amendment_while_other_slot_runs(cell, monkeypatch):
    root = cell["new"]
    amendment.amend(root, resume=True)
    frozen = tree_hashes(root / "recovery")
    plan = amendment.read_plan(root)
    # An offline amendment has not materialized unused fresh destinations yet.
    assert not (root / plan["records"][0]["fresh"]).exists()
    forbid_classification(monkeypatch)
    execute, interleaved = sm.run_episode, []
    slot_a = Path(cell["slot"]).with_name("candidate-04")
    slot_b = slot_a.with_name("candidate-05")

    def start_b(*args, **kwargs):
        if not interleaved:
            interleaved.append(True)
            assert (root / slot_a / "started.json").exists()
            assert not (root / slot_a / "candidate.json").exists()
            assert str(slot_a) not in cell["identity"]["chain_recovery"]["retry_directories"]
            run(cell, shard_count=20, shard_index=3)
            assert (root / slot_b / "candidate.json").exists()
        return execute(*args, **kwargs)

    monkeypatch.setattr(sm, "run_episode", start_b)
    calls = len(cell["calls"])
    run(cell, shard_count=20, shard_index=2)
    assert interleaved and len(cell["calls"]) == calls + 2
    assert (root / slot_a / "candidate.json").exists()
    assert tree_hashes(root / "recovery") == frozen
    before = tree_hashes(root)
    assert amendment.amend(root, resume=True) == {
        "applied": True, "ready": True, "cell": root.name, "rule": amendment.RULE,
        "amendment": amendment.AMENDMENT, "extra_call_upper_bound": 41, "plan_sha256": digest(plan)}
    assert tree_hashes(root) == before


def test_resume_after_unamended_execution_interruption_never_recredits(cell, monkeypatch):
    root = cell["new"]
    amendment.amend(root, resume=True)
    plan = amendment.read_plan(root)
    frozen = tree_hashes(root / "recovery")
    execute = sm.run_episode

    def interrupt(*args, **kwargs):
        execute(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(sm, "run_episode", interrupt)
    calls = len(cell["calls"])
    with pytest.raises(KeyboardInterrupt):
        run(cell, shard_count=20, shard_index=2)
    slot = Path(cell["slot"]).with_name("candidate-04")
    assert (root / slot / "started.json").exists()
    assert not (root / slot / "candidate.json").exists()
    assert len(cell["calls"]) == calls + 1
    accounting = collection_accounting(root, cell["identity"])
    before = tree_hashes(root)
    summary = amendment.amend(root, resume=True, dry_run=True)
    assert summary["ready"] and summary["refused"] == 0 and summary["records"] == []
    assert summary["in_progress"] == summary["interrupted_to_restart"] == 1
    assert summary["extra_call_upper_bound"] == 41 and summary["never_started"] == 17
    assert summary["remaining_call_upper_bound"] == 19 * 41
    assert summary["diagnostics"] == [{"slot": str(slot), "task_id": plan["records"][0]["task_id"],
                                        "candidate": slot.name, "status": "in_progress"}]
    assert tree_hashes(root) == before
    monkeypatch.setattr(sm, "run_episode", execute)
    forbid_classification(monkeypatch)
    for _ in range(2):
        run(cell, shard_count=20, shard_index=2)
    assert (root / slot / "candidate.json").exists()
    assert len(cell["calls"]) == calls + 1
    assert tree_hashes(root / "recovery") == frozen
    assert amendment.read_plan(root) == plan
    assert collection_accounting(root, cell["identity"]) == accounting
    assert not (root / slot / "amendment.json").exists()
    assert not (root / slot / "teacher/cap_credit.json").exists()


@pytest.mark.parametrize("tamper", ["archive", "completion", "missing_plan", "fresh_file", "fresh_symlink",
                                  "fresh_task", "fresh_receipt"])
def test_complete_verification_refuses_changed_evidence(cell, monkeypatch, tamper):
    root = cell["new"]
    amendment.amend(root, resume=True)
    record = amendment.read_plan(root)["records"][0]
    fresh = root / record["fresh"]
    if tamper == "archive":
        write_json(root / record["archive"] / "started.json", {})
    elif tamper == "completion":
        write_json(root / amendment.COMPLETE, {"plan_sha256": "changed"})
    elif tamper == "missing_plan":
        (root / amendment.PLAN).unlink()
    elif tamper in {"fresh_file", "fresh_symlink"}:
        fresh.parent.mkdir(parents=True)
        if tamper == "fresh_file":
            fresh.write_text("not a directory")
        else:
            fresh.symlink_to(root / record["archive"], target_is_directory=True)
    elif tamper == "fresh_task":
        write_json(fresh / "started.json", {"task_id": "other"})
    else:
        write_json(root / record["slot"] / "candidate.json", {
            "task_id": record["task_id"], "candidate_id": record["candidate"], "reused": False,
            "episode_dir": str(fresh / "episode")})
    forbid_classification(monkeypatch)
    calls = len(cell["calls"])
    with pytest.raises(ValueError, match="SmartAD"):
        amendment.amend(root, resume=True)
    with pytest.raises(ValueError, match="SmartAD"):
        run(cell, shard_count=20, shard_index=2)
    assert len(cell["calls"]) == calls


@pytest.mark.parametrize("complete,dry_run", [(False, False), (False, True), (True, False), (True, True)])
def test_launcher_amendment_preflight_and_completed_resume(cell, monkeypatch, capsys, complete, dry_run):
    from baselines.common import collection_launcher as launcher
    root = cell["new"]
    if complete:
        amendment.amend(root, resume=True)
        slot = Path(cell["slot"]).with_name("candidate-04")
        write_json(root / slot / "started.json", {"task_id": "in-progress-fixture"})
        if not dry_run:
            forbid_classification(monkeypatch)
    monkeypatch.setattr(launcher, "load_config", lambda _: cell["config"])
    monkeypatch.setattr(launcher, "load_frozen", lambda *_: (cell["manifest"], cell["freeze"]))
    monkeypatch.setattr(launcher.signal, "signal", lambda *_: None)
    commands = []

    def fake_process(command, **kwargs):
        assert (root / amendment.COMPLETE).exists()
        commands.append(command)
        write_json(root / "candidates.json", {"status": "complete"})
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_process)
    before, calls = tree_hashes(root), len(cell["calls"])
    launcher.main(["--method", "smartad", "--cell", root.name, "--source-view", str(cell["support"]),
        "--output-root", str(root.parent), "--rate-limit-dir", str(cell["kw"]["rate_limit_dir"]),
        "--log-root", str(root.parent / "logs"), "--shards", "2", "--resume",
        *(["--dry-run"] if dry_run else [])])
    summary = json.loads(capsys.readouterr().out)["cells"][0]["amendment"]
    assert summary["ready"] and summary["extra_call_upper_bound"] == 41
    assert len(cell["calls"]) == calls
    if dry_run:
        assert not commands and tree_hashes(root) == before
        # Launcher verification after COMPLETE must not scan running slots.
        assert summary.get("in_progress", 0) == 0
    else:
        assert len(commands) == 2
        assert all("--resume" in command for command in commands)
        assert summary.get("applied", False) is complete


def test_nonplanned_in_progress_slot_still_refused_before_complete(cell):
    root = cell["new"]
    slot = Path(cell["slot"]).with_name("candidate-04")
    write_json(root / slot / "started.json", {})
    before = tree_hashes(root)
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="Not an unfinished candidate"):
            amendment.amend(root, resume=True, dry_run=dry_run)
    assert tree_hashes(root) == before
