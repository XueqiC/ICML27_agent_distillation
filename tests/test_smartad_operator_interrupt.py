"""Two operator interruptions, with fake transport/environment and real journals."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.smartad import collect as sm
from sed.storage import digest, write_json
from teachers import smartad_operator_interrupt as operator, smartad_resume as first
from teachers.chain_recovery import collection_accounting, ledger_rows, read_json
from teachers.full_train_recovery import tree_hashes
from teachers.recover import recover
from test_chain_recovery import long_environment
from test_smartad import factory
from test_smartad_resume import cell, run
from test_stage2b_seeds import seed_bank


@pytest.fixture
def interrupted(cell, monkeypatch):
    root = cell["new"]
    first.amend(root, resume=True)
    first_frozen = tree_hashes(root / "recovery" / first.AMENDMENT)
    slots = {str(Path(cell["slot"]).with_name("candidate-04")): 2,
             str(Path(cell["slot"]).with_name("candidate-05")): 6}
    monkeypatch.setattr(operator, "SLOTS", slots)
    for shard, (slot, attempts) in enumerate(slots.items(), start=2):
        make = factory(cell["calls"])

        def teacher(*args):
            value = make(*args)
            responses = value.teacher.client.client.responses.with_raw_response
            request, count = responses.create, 0

            def interrupt(**kwargs):
                nonlocal count
                count += 1
                if count == attempts:
                    raise KeyboardInterrupt
                return request(**kwargs)

            responses.create = interrupt
            return value

        with pytest.raises(KeyboardInterrupt):
            sm.collect(cell["support"], cell["manifest"], cell["freeze"], cell["config"], root,
                teacher_factory=teacher, shard_count=20, shard_index=shard,
                **{**cell["kw"], "env_factory": long_environment(cell["config"])})
        recover(root / slot / "teacher/counter.json", reason="interruption_reviewed")
    assert tree_hashes(root / "recovery" / first.AMENDMENT) == first_frozen
    return cell


def test_two_slots_dry_run_apply_collect_accounting_and_idempotence(interrupted):
    cell, root = interrupted, interrupted["new"]
    before, calls = tree_hashes(root), len(cell["calls"])
    summary = operator.amend(root, resume=True, dry_run=True)
    assert summary["eligible"] == 2 and summary["refused"] == 0 and summary["ready"]
    assert summary["extra_call_upper_bound"] == 82
    spent = summary["spent_and_void_usage"]
    assert spent["counter_attempts"] == spent["http_calls"] == spent["logical_calls"] == 8
    assert spent["unknown_usage_calls"] == spent["unknown_outcome_in_flight_requests"] == 2
    assert spent["refunded_attempts"] == spent["quota_void_calls"] == 0
    assert spent["status_counts"] == {"success": 6, "started": 2}
    assert spent["known_billed_tokens"]["input_tokens"] == 6 * 12
    assert spent["environment_steps"] == 6
    assert tree_hashes(root) == before and len(cell["calls"]) == calls
    frozen_first = tree_hashes(root / "recovery" / first.AMENDMENT)
    first_plan = first.read_plan(root)
    accounting_before = collection_accounting(root, cell["identity"])
    operator.amend(root, resume=True)
    plan = operator.read_plan(root)
    assert plan["spent_and_void_usage"] == spent
    assert plan["first_amendment_plan_sha256"] == digest(first_plan)
    for record in plan["records"]:
        assert tree_hashes(root / record["archive"]) == record["files"]
        assert read_json(root / record["slot"] / "amendment.json") == record
        assert not (root / record["fresh"]).exists()
        assert first.fresh_directory(root, record["slot"]) == record["fresh"]
    assert first.fresh_directory(root, first_plan["records"][0]["slot"]) == first_plan["records"][0]["fresh"]
    assert first.fresh_directory(root, Path(cell["slot"]).with_name("candidate-06")) is None
    accounting = collection_accounting(root, cell["identity"])
    assert accounting["new_usage"] == accounting_before["new_usage"]
    assert accounting["new_usage"]["unknown_usage_calls"] == 2
    assert accounting["amendment_extra_call_upper_bound"] == 41 + 82
    assert accounting["effective_per_cell_call_cap"] == accounting_before["effective_per_cell_call_cap"] + 82
    for index in (2, 3):
        run(cell, shard_count=20, shard_index=index)
    assert len(cell["calls"]) == calls + 2
    for record in plan["records"]:
        fresh = root / record["fresh"]
        counter = read_json(fresh / "teacher/counter.json")
        assert counter["attempts"] == 1 and counter["identity"]["cap"] == 41
        assert counter["identity"]["run_id"] != record["run_id"]
        assert not (fresh / "teacher/cap_credit.json").exists()
        ep = read_json(fresh / "episode/episode.json")
        assert ep["complete"] and ep["metrics"]["environment_steps"] == 1
        assert read_json(root / record["slot"] / "candidate.json")["episode_dir"] == str(fresh / "episode")
    before = tree_hashes(root)
    for dry_run in (False, True, False):
        assert operator.amend(root, resume=True, dry_run=dry_run)["applied"]
    assert tree_hashes(root) == before
    assert tree_hashes(root / "recovery" / first.AMENDMENT) == frozen_first
    assert first.read_plan(root) == first_plan and operator.read_plan(root) == plan
    assert read_json(root / "identity.json") == cell["identity"]
    accounting = collection_accounting(root, cell["identity"])
    assert accounting["new_teacher_calls"] == accounting_before["new_teacher_calls"] + 2
    assert accounting["new_usage"]["unknown_usage_calls"] == 2


@pytest.mark.parametrize("failure", ["missing_recovery", "completed", "episode_complete", "known_success",
    "post_start_link", "post_start_success", "post_start_started", "old_run", "rejected",
    "missing_success", "recovery_reason", "recovery_hash", "recovery_boundary", "refunded",
    "halted", "extra_recovery", "no_resume", "wrong_cell", "first_membership"])
def test_refusals_do_not_write(interrupted, monkeypatch, failure):
    root = interrupted["new"]
    slot = next(iter(operator.SLOTS))
    directory = root / slot
    counter_path, recovery_path = directory / "teacher/counter.json", directory / "teacher/recoveries/0000.json"
    if failure == "missing_recovery":
        recovery_path.unlink()
    elif failure == "completed":
        write_json(directory / "candidate.json", {})
    elif failure in {"episode_complete", "known_success"}:
        path = directory / "episode/episode.json"
        value = read_json(path)
        value["complete" if failure == "episode_complete" else "success"] = True
        write_json(path, value)
    elif failure in {"recovery_reason", "recovery_hash", "recovery_boundary", "extra_recovery"}:
        value = read_json(recovery_path)
        if failure == "recovery_reason": value["reason"] = "transport_fixed"
        if failure == "recovery_hash": value["ledger_prefix_sha256"] = "changed"
        if failure == "recovery_boundary": value["through_rows"] -= 1
        write_json(recovery_path.with_name("0001.json") if failure == "extra_recovery" else recovery_path, value)
    elif failure in {"refunded", "halted"}:
        value = read_json(counter_path)
        if failure == "refunded": value["attempts"] -= 1
        if failure == "halted": value["halted"] = True
        write_json(counter_path, value)
    elif failure == "wrong_cell":
        monkeypatch.setattr(first, "CELL_IDENTITY_SHA256", "wrong")
    elif failure == "first_membership":
        monkeypatch.setattr(operator, "SLOTS", {interrupted["slot"]: 2, list(operator.SLOTS)[1]: 6})
    elif failure != "no_resume":
        path = directory / "teacher/usage.jsonl"
        rows = ledger_rows(path)
        if failure.startswith("post_start_"):
            status = {"link": "state_link", "success": "success", "started": "started"}[failure[11:]]
            rows.append({**rows[-1], "status": status})
        if failure == "old_run": rows[0]["run_id"] = "old"
        if failure == "rejected": rows[1]["status"] = "rejected"
        if failure == "missing_success": rows.pop(1)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = tree_hashes(root)
    with pytest.raises(ValueError):
        operator.amend(root, resume=failure != "no_resume")
    assert tree_hashes(root) == before


def test_extra_slot_refused_without_scanning_other_workers(interrupted):
    root = interrupted["new"]
    slot = str(Path(interrupted["slot"]).with_name("candidate-06"))
    write_json(root / slot / "started.json", {"task_id": "other worker in progress"})
    before = tree_hashes(root)
    with pytest.raises(ValueError, match="outside the two"):
        operator._interruption(root, interrupted["identity"], first.read_plan(root), slot, "other")
    assert operator.amend(root, resume=True, dry_run=True)["eligible"] == 2
    assert tree_hashes(root) == before


@pytest.mark.parametrize("move", [1, 2, 4, 6])
def test_archive_crash_resumes_exact_plan(interrupted, monkeypatch, move):
    root = interrupted["new"]
    rename, count = Path.rename, 0

    def crash(path, target):
        nonlocal count
        result = rename(path, target)
        count += 1
        if count == move:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(Path, "rename", crash)
    with pytest.raises(KeyboardInterrupt):
        operator.amend(root, resume=True)
    plan = operator.read_plan(root)
    assert not (root / operator.COMPLETE).exists()
    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(operator, "classify", lambda *_: pytest.fail("Frozen plan must not reclassify"))
    before = tree_hashes(root)
    assert not operator.amend(root, resume=True, dry_run=True)["applied"]
    assert tree_hashes(root) == before
    operator.amend(root, resume=True)
    assert operator.read_plan(root) == plan
    for record in plan["records"]:
        assert tree_hashes(root / record["archive"]) == record["files"]


def forbid_classification(monkeypatch):
    for module in (operator, first):
        monkeypatch.setattr(module, "classify", lambda *_: pytest.fail("COMPLETE is verify-only"))


@pytest.mark.parametrize("boundary", ["plan", "slot", "complete", "unplanned_file"])
def test_publication_crashes_and_changed_inventory(interrupted, monkeypatch, boundary):
    root = interrupted["new"]
    publish = write_json

    def crash(path, value, **kwargs):
        result = publish(path, value, **kwargs)
        if ((boundary in {"plan", "unplanned_file"} and path == root / operator.PLAN)
                or (boundary == "slot" and path.name == "amendment.json")
                or (boundary == "complete" and path == root / operator.COMPLETE)):
            raise KeyboardInterrupt
        return result

    with monkeypatch.context() as patch:
        patch.setattr(operator, "write_json", crash)
        patch.setattr(first, "write_json", crash)
        with pytest.raises(KeyboardInterrupt):
            operator.amend(root, resume=True)
    plan = operator.read_plan(root)
    forbid_classification(monkeypatch)
    if boundary == "unplanned_file":
        publish(root / plan["records"][0]["slot"] / "unexpected.json", {})
        before = tree_hashes(root)
        for dry_run in (True, False):
            with pytest.raises(ValueError, match="inventory changed"):
                operator.amend(root, resume=True, dry_run=dry_run)
        assert tree_hashes(root) == before and not (root / operator.COMPLETE).exists()
    else:
        operator.amend(root, resume=True)
        assert operator.read_plan(root) == plan
        assert read_json(root / operator.COMPLETE) == {"plan_sha256": digest(plan)}


def test_staggered_shards_only_verify_both_frozen_amendments(interrupted, monkeypatch):
    cell, root = interrupted, interrupted["new"]
    operator.amend(root, resume=True)
    plan, frozen = operator.read_plan(root), tree_hashes(root / "recovery")
    forbid_classification(monkeypatch)
    execute, interleaved = sm.run_episode, []

    def start_second(*args, **kwargs):
        if not interleaved:
            interleaved.append(True)
            assert (root / plan["records"][0]["fresh"] / "started.json").exists()
            assert not (root / plan["records"][0]["slot"] / "candidate.json").exists()
            run(cell, shard_count=20, shard_index=3)
            assert (root / plan["records"][1]["slot"] / "candidate.json").exists()
        return execute(*args, **kwargs)

    monkeypatch.setattr(sm, "run_episode", start_second)
    calls = len(cell["calls"])
    run(cell, shard_count=20, shard_index=2)
    assert interleaved and len(cell["calls"]) == calls + 2
    assert tree_hashes(root / "recovery") == frozen
    assert operator.amend(root, resume=True, dry_run=True)["applied"]


@pytest.mark.parametrize("tamper", ["archive", "first_archive", "completion", "missing_plan",
    "extra_slot", "changed_bound", "fresh_symlink", "fresh_receipt"])
def test_complete_verification_refuses_tampering(interrupted, monkeypatch, tamper):
    cell, root = interrupted, interrupted["new"]
    operator.amend(root, resume=True)
    plan = operator.read_plan(root)
    record = plan["records"][0]
    if tamper in {"archive", "first_archive"}:
        selected = first.read_plan(root)["records"][0] if tamper == "first_archive" else record
        write_json(root / selected["archive"] / "started.json", {})
    elif tamper == "completion":
        write_json(root / operator.COMPLETE, {"plan_sha256": "changed"})
    elif tamper == "missing_plan":
        (root / operator.PLAN).unlink()
    elif tamper in {"extra_slot", "changed_bound"}:
        if tamper == "extra_slot": plan["records"].append({**record, "slot": "work/extra/candidate-05"})
        else: plan["extra_call_upper_bound"] += 41
        write_json(root / operator.PLAN, plan)
    elif tamper == "fresh_symlink":
        fresh = root / record["fresh"]
        fresh.parent.mkdir(parents=True)
        fresh.symlink_to(root / record["archive"], target_is_directory=True)
    else:
        write_json(root / record["slot"] / "candidate.json", {})
    forbid_classification(monkeypatch)
    before, calls = tree_hashes(root) if tamper != "fresh_symlink" else None, len(cell["calls"])
    for dry_run in (False, True):
        with pytest.raises((ValueError, FileNotFoundError), match="SmartAD|No such"):
            operator.amend(root, resume=True, dry_run=dry_run)
    with pytest.raises((ValueError, FileNotFoundError)):
        run(cell, shard_count=20, shard_index=4)
    assert len(cell["calls"]) == calls
    if before is not None:
        assert tree_hashes(root) == before


@pytest.mark.parametrize("complete", [False, True])
def test_cli_selects_new_amendment_without_calls_launches_or_writes(interrupted, monkeypatch, capsys, complete):
    import subprocess
    from tools.amend_smartad_resume import main
    root = interrupted["new"]
    if complete:
        operator.amend(root, resume=True)
        forbid_classification(monkeypatch)
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("No launches"))
    before, calls = tree_hashes(root), len(interrupted["calls"])
    assert main(["--amendment", operator.AMENDMENT, "--output", str(root), "--resume", "--dry-run"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["eligible"] == 2 and summary["refused"] == 0 and summary["extra_call_upper_bound"] == 82
    assert tree_hashes(root) == before and len(interrupted["calls"]) == calls


@pytest.mark.parametrize("dry_run", [False, True])
def test_launcher_verifies_both_after_complete(interrupted, monkeypatch, capsys, dry_run):
    from baselines.common import collection_launcher as launcher
    cell, root = interrupted, interrupted["new"]
    operator.amend(root, resume=True)
    forbid_classification(monkeypatch)
    monkeypatch.setattr(launcher, "load_config", lambda _: cell["config"])
    monkeypatch.setattr(launcher, "load_frozen", lambda *_: (cell["manifest"], cell["freeze"]))
    monkeypatch.setattr(launcher.signal, "signal", lambda *_: None)
    commands = []

    def process(command, **kwargs):
        commands.append(command)
        write_json(root / "candidates.json", {"status": "complete"})
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(launcher.subprocess, "Popen", process)
    before = tree_hashes(root)
    launcher.main(["--method", "smartad", "--cell", root.name, "--source-view", str(cell["support"]),
        "--output-root", str(root.parent), "--rate-limit-dir", str(cell["kw"]["rate_limit_dir"]),
        "--log-root", str(root.parent / "logs"), "--shards", "2", "--resume",
        *(["--dry-run"] if dry_run else [])])
    result = json.loads(capsys.readouterr().out)["cells"][0]
    assert result["amendment"]["applied"] and result["operator_interrupt_amendment"]["applied"]
    assert result["operator_interrupt_amendment"]["extra_call_upper_bound"] == 82
    assert len(commands) == (0 if dry_run else 2)
    if dry_run:
        assert tree_hashes(root) == before
