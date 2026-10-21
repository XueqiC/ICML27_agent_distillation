"""Offline migration with real durable collectors and fake provider/environment."""
from pathlib import Path

import pytest

from baselines.adftp import collect as ftp
from baselines.smartad import collect as sm
from sed.seed_evidence import artifact_hashes, load_verified_collection, materialize_subsets
from sed.storage import digest, write_json
from teachers.chain_recovery import credit_file, read_json, recover_chain_collection
from teachers.openai_teacher import TeacherCallError
from test_adftp import stub_factory
from test_smartad import factory, setup_source
from test_stage2b_seeds import seed_bank


class RateLimitError(Exception):
    status_code = 429
    body = {"error": {"code": "insufficient_quota"}}


def quota_factory(make, calls, *, fail=False, error=RateLimitError, fail_after=0):
    def wrapped(*args):
        teacher = make(*args)
        responses = teacher.teacher.client.client.responses.with_raw_response
        original = responses.create
        successes = 0
        def create(**kwargs):
            nonlocal successes
            calls.append(args[2]["task_id"])
            if fail and successes >= fail_after:
                raise error("SECRET provider text")
            successes += 1
            return original(**kwargs)
        responses.create = create
        return teacher
    return wrapped


def make_ftp(seed_bank, tmp_path, monkeypatch, *, error=RateLimitError, nested=False):
    monkeypatch.setattr("teachers.openai_teacher.utc_now", lambda: "2026-10-20T22:45:30+00:00")
    support, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=2, full=nested)
    if nested:
        from sed.seed_evidence import materialize_full_train_views
        materialize_full_train_views(support, manifest, freeze, [1, 2], seed=0)
        small = support / "subsets/k1"
        support = support / "subsets/k2"
    else:
        materialize_subsets(support, manifest, freeze, [1])
        small = support / "subsets/k1"
    old, new = tmp_path / "old", tmp_path / "new"
    calls = []
    kw = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env,
              subset_views=[f"1={small}"])
    ftp.collect(support, manifest, freeze, config, old, shard_count=2, shard_index=0,
                teacher_factory=quota_factory(stub_factory([]), calls), **kw)
    with pytest.raises(TeacherCallError):
        ftp.collect(support, manifest, freeze, config, old, shard_count=2, shard_index=1,
                    teacher_factory=quota_factory(stub_factory([]), calls, fail=True, error=error, fail_after=1), **kw)
    return support, manifest, freeze, config, env, old, new, calls, kw


@pytest.mark.parametrize("nested", [False, True])
def test_ftp_exact_credit_import_and_resume(seed_bank, tmp_path, monkeypatch, nested):
    support, manifest, freeze, config, _, old, new, calls, kw = make_ftp(seed_bank, tmp_path, monkeypatch, nested=nested)
    before = artifact_hashes(old)
    imported = next((old / "collection/attempts").iterdir())
    imported_hashes = artifact_hashes(imported)
    args = old, new, "adftp", manifest, freeze, config
    plan = recover_chain_collection(*args, dry_run=True)
    assert not new.exists() and artifact_hashes(old) == before
    assert plan["tasks_imported_final"] == plan["tasks_to_run"] == 1
    assert plan["inherited_usage"]["quota_void_calls"] == 1
    unfinished = next(r for r in plan["tasks"] if not r["final"])
    assert unfinished["old_attempts"] == 2 and unfinished["remaining_cap"] == 122
    recover_chain_collection(*args)
    assert artifact_hashes(new / "collection/attempts" / imported.name) == imported_hashes
    count = len(calls)
    root = ftp.collect(support, manifest, freeze, config, new,
                       teacher_factory=quota_factory(stub_factory([]), calls), **kw)
    assert len(calls) == count + 2
    assert all(tid == unfinished["task_id"] for tid in calls[count:])
    collection = load_verified_collection(root, manifest, freeze)[0]
    assert collection["inherited_usage"]["http_calls"] == 4
    assert collection["new_usage"]["http_calls"] == 2
    assert collection["inherited_usage"]["known_billed_tokens"]["input_tokens"] == 36
    assert collection["new_usage"]["known_billed_tokens"]["input_tokens"] == 24
    assert load_verified_collection(root / "subsets/k1", manifest, freeze)[0]["new_teacher_calls"] == 0
    work = new / "work" / unfinished["work_key"]
    assert (work / "attempt-1/episode/episode.json").is_file()
    assert (work / "transport_retries/attempt-1/attempt-1/episode/episode.json").is_file()
    counter = read_json(work / "teacher/counter.json")
    assert counter["attempts"] == 3
    assert artifact_hashes(old) == before
    ftp.collect(support, manifest, freeze, config, new,
                teacher_factory=quota_factory(stub_factory([]), calls), **kw)
    assert len(calls) == count + 2
    assert recover_chain_collection(*args)["action"] == "resume"
    assert artifact_hashes(new / "collection/attempts" / imported.name) == imported_hashes


@pytest.mark.parametrize("tamper", ["source", "source_work", "import", "credit", "prefix"])
def test_hash_refusals(seed_bank, tmp_path, monkeypatch, tamper):
    support, manifest, freeze, config, _, old, new, calls, kw = make_ftp(seed_bank, tmp_path, monkeypatch)
    args = old, new, "adftp", manifest, freeze, config
    if tamper in {"source", "source_work"}:
        if tamper == "source":
            next((old / "collection/attempts").rglob("episode.json")).write_text("{}")
        else:
            next((old / "work").rglob("outcome.json")).write_text("{}")
        with pytest.raises(ValueError):
            recover_chain_collection(*args)
        assert not new.exists()
        return
    plan = recover_chain_collection(*args)
    row = next(r for r in plan["tasks"] if not r["final"])
    if tamper == "import":
        next((new / "collection/attempts").rglob("receipt.json")).unlink()
    elif tamper == "credit":
        path = credit_file(new, row["counters"][0]["path"])
        record = read_json(path)
        record["credit"] += 1
        write_json(path, record)
    else:
        (new / "work" / row["work_key"] / "teacher/usage.jsonl").write_text("")
    count = len(calls)
    with pytest.raises(ValueError):
        ftp.collect(support, manifest, freeze, config, new,
                    teacher_factory=quota_factory(stub_factory([]), calls), **kw)
    assert len(calls) == count


def test_nonquota_refused(seed_bank, tmp_path, monkeypatch):
    _, manifest, freeze, config, _, old, new, _, _ = make_ftp(seed_bank, tmp_path, monkeypatch, error=ValueError)
    with pytest.raises(ValueError, match="Non-quota"):
        recover_chain_collection(old, new, "adftp", manifest, freeze, config)
    assert not new.exists()


def test_format_rejections_are_final_without_credit(seed_bank, tmp_path):
    support, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    tid = load_verified_collection(support, manifest, freeze)[0]["verified"][0]["task_id"]
    old, new = tmp_path / "old", tmp_path / "new"
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env)
    ftp.collect(support, manifest, freeze, config, old,
                teacher_factory=stub_factory([], mismatch={tid}), **kwargs)
    plan = recover_chain_collection(old, new, "adftp", manifest, freeze, config)
    assert plan["tasks_imported_final"] == 1 and plan["tasks_to_run"] == 0
    assert plan["inherited_usage"]["quota_void_calls"] == 0
    ftp.collect(support, manifest, freeze, config, new,
                teacher_factory=lambda *a: pytest.fail("final task constructed a teacher"), **kwargs)


def test_launcher_recovery_dry_run(seed_bank, tmp_path, monkeypatch, capsys):
    import json
    from baselines.common import collection_launcher as launcher
    support, manifest, freeze, config, _, old, new, _, kw = make_ftp(seed_bank, tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "load_config", lambda _: config)
    monkeypatch.setattr(launcher, "load_frozen", lambda *a: (manifest, freeze))
    launcher.main(["--method", "adftp", "--cell", new.name, "--source-view", str(support),
                   "--subset-view", kw["subset_views"][0], "--output-root", str(tmp_path),
                   "--rate-limit-dir", str(kw["rate_limit_dir"]), "--recover-from", str(old), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)["cells"][0]
    assert plan["recovery"]["tasks_imported_final"] == 1
    assert plan["recovery"]["tasks_to_run"] == 1
    assert not new.exists()


def test_interrupted_copy_resumes_and_counter_floor(seed_bank, tmp_path, monkeypatch):
    from teachers import chain_recovery as recovery
    support, manifest, freeze, config, _, old, new, calls, kw = make_ftp(seed_bank, tmp_path, monkeypatch)
    original = recovery._copy_checked
    copied = 0
    def interrupt(*args):
        nonlocal copied
        copied += 1
        if copied == 4:
            raise KeyboardInterrupt
        return original(*args)
    monkeypatch.setattr(recovery, "_copy_checked", interrupt)
    with pytest.raises(KeyboardInterrupt):
        recover_chain_collection(old, new, "adftp", manifest, freeze, config)
    assert not new.exists()
    monkeypatch.setattr(recovery, "_copy_checked", original)
    plan = recover_chain_collection(old, new, "adftp", manifest, freeze, config)
    row = next(r for r in plan["tasks"] if not r["final"])
    work = new / "work" / row["work_key"]
    teacher = stub_factory([])(config, work / "teacher", {"task_id": row["task_id"]}, "test", kw["rate_limit_dir"])
    assert teacher._read_counter()["attempts"] == 1
    value = teacher._read_counter()
    value["attempts"] = 0
    write_json(work / "teacher/counter.json", value)
    assert teacher._read_counter()["attempts"] == 1
    from teachers.recover import recover
    recover(work / "teacher/counter.json", reason="transport_fixed")
    assert teacher._read_counter()["attempts"] == 1
    with pytest.raises(ValueError, match="already recovered"):
        recover_chain_collection(new, tmp_path / "r3", "adftp", manifest, freeze, config)


def test_smartad_candidate_slots(seed_bank, tmp_path, monkeypatch):
    monkeypatch.setattr("teachers.openai_teacher.utc_now", lambda: "2026-10-20T22:45:30+00:00")
    support, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=2)
    materialize_subsets(support, manifest, freeze, [1])
    old, new = tmp_path / "old", tmp_path / "new"
    calls = []
    kw = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env,
              subset_views=[f"1={support}/subsets/k1"])
    good = quota_factory(factory([]), calls)
    for index in [0, *range(10, 20)]:
        sm.collect(support, manifest, freeze, config, old, shard_count=20, shard_index=index,
                   teacher_factory=good, **kw)
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    def two_steps(task):
        def backend(path, seed):
            value = FakeBackend(path, seed)
            value.win_at = 2
            return value
        return Environment(config["environment"]["data_root"], max_episode_steps=40, backend_factory=backend)
    with pytest.raises(TeacherCallError):
        sm.collect(support, manifest, freeze, config, old, shard_count=20, shard_index=1,
                   teacher_factory=quota_factory(factory([]), calls, fail=True, fail_after=1),
                   **{**kw, "env_factory": two_steps})
    before = artifact_hashes(old)
    originals = {str(p.relative_to(old)): artifact_hashes(p.parent) for p in (old / "work").rglob("candidate.json")}
    candidate_path = next((old / "work").rglob("candidate.json"))
    original_bytes = candidate_path.read_bytes()
    candidate_record = read_json(candidate_path)
    write_json(candidate_path, {**candidate_record, "correct": not candidate_record["correct"]})
    with pytest.raises(ValueError, match="correctness"):
        recover_chain_collection(old, new, "smartad", manifest, freeze, config)
    candidate_path.write_bytes(original_bytes)
    plan = recover_chain_collection(old, new, "smartad", manifest, freeze, config)
    assert plan["tasks_imported_final"] == 1 and plan["candidates_imported_final"] == 11
    assert plan["inherited_usage"]["quota_void_calls"] == 1
    count = len(calls)
    sm.collect(support, manifest, freeze, config, new, teacher_factory=good, **kw)
    assert len(calls) - count == 8
    for name, hashes in originals.items():
        assert artifact_hashes((new / name).parent) == hashes
    pool = sm.load_pool(new, manifest, freeze)[0]
    assert pool["inherited_usage"]["http_calls"] == 12
    assert pool["new_usage"]["http_calls"] == 8
    assert (new / "subsets/k1/candidates.json").is_file()
    assert artifact_hashes(old) == before
    sm.collect(support, manifest, freeze, config, new, teacher_factory=good, **kw)
    assert len(calls) - count == 8
    # Selection must resolve imported absolute paths into the copied evidence.
    from baselines.smartad.select import select, validate_selection
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    selected = select(new, snapshot, config, manifest, freeze, tmp_path / "selected", scorer=lambda _: {"nll": 1.0})
    assert load_verified_collection(selected, manifest, freeze)[0]["K"] == 2
    validate_selection(selected, manifest, freeze, snapshot)


@pytest.mark.parametrize("change", [
    {"http_status": 500}, {"retryable": True}, {"usage": {"input_tokens": 1}},
    {"timestamp": "2026-10-20T22:44:59+00:00"}, {"raw_ref": "sha256:output"},
    {"error_type": "ConnectionError"},
])
def test_void_predicate_is_exact(change):
    from teachers.chain_recovery import quota_void
    row = {"status": "failed", "http_status": 429, "error_type": "RateLimitError", "retryable": False,
           "will_retry": False, "usage": None, "timestamp": "2026-10-20T22:45:30+00:00"}
    assert quota_void(row)
    assert not quota_void({**row, **change})


def test_quota_timestamp_uses_transport_admission():
    from teachers.chain_recovery import quota_void
    row = {"status": "failed", "http_status": 429, "error_type": "RateLimitError", "retryable": False,
           "will_retry": False, "usage": None, "timestamp": "2026-10-20T22:43:30+00:00",
           "reservation_timestamp": 1792536337.2320633}
    assert quota_void(row)
    assert not quota_void({**row, "reservation_timestamp": 1792536200.0})


@pytest.fixture
def cap_evidence(tmp_path):
    """Minimal content-addressed execution evidence for eligibility decisions."""
    from sed.storage import BlobStore
    from baselines.adftp.teacher import CALLS_PER_TASK
    source = tmp_path / "cap-source"
    work = source / "work/task"
    for number in range(1, 4):
        directory = work / f"attempt-{number}"
        write_json(directory / "started.json", {"attempt": number})
        complete = number < 3
        episode = {"task_id": "task", "episode_id": f"episode-{number}", "complete": complete,
                   "success": False if complete else "unknown", "teacher_transport_failure": not complete}
        write_json(directory / "episode/episode.json", episode)
        ref = BlobStore(directory / "episode/blobs").put({"audit": {
            "error_type": None if complete else "BudgetExceeded"}})
        write_json(directory / "episode/events.jsonl", {"task_id": "task", "episode_id": episode["episode_id"],
                   "verification_ref": ref, "status": "ok" if complete else "error", "error_type": "policy_error"})
    record = {"task_id": "task", "final": False, "credit": 1, "void_calls": [{"call_id": "void"}],
              "old_attempts": CALLS_PER_TASK, "cap": CALLS_PER_TASK,
              "source_counter": "work/task/teacher/counter.json", "target_counter": "work/task/teacher/counter.json",
              "source_counter_sha256": "counter-hash", "ledger_prefix_sha256": "ledger-hash"}
    return source, work, record


@pytest.mark.parametrize("credits", [1, 2, 7])
def test_cap_extension_exact_arithmetic(cap_evidence, credits):
    from baselines.adftp.teacher import CALLS_PER_ATTEMPT
    from teachers.chain_recovery import RULE, cap_extension, effective_cap
    source, work, record = cap_evidence
    record["credit"] = credits
    extension = cap_extension(source, work, record, "adftp", artifact_hashes(source))
    assert extension["rule"] == RULE and extension["record_type"] == "cap_extension"
    assert extension["task_id"] == "task" and extension["attempt_id"] == "episode-3"
    assert extension["calls"] == record["cap"]
    assert extension["extension"] == CALLS_PER_ATTEMPT - credits
    assert extension["original_cap"] + credits + extension["extension"] == record["cap"] + CALLS_PER_ATTEMPT
    assert extension["effective_cap"] - (record["old_attempts"] - credits) == CALLS_PER_ATTEMPT
    assert extension["fresh_attempt_directory"] == "work/task/transport_retries/attempt-3/attempt-1"
    assert len(extension["earlier_complete_attempts"]) == 2
    binding = {"cap_extensions": {record["target_counter"]: extension}}
    record["cap_extension_sha256"] = digest(extension)
    assert effective_cap(record, binding) == extension["effective_cap"]
    extension["extension"] += 1
    with pytest.raises(ValueError, match="binding/arithmetic"):
        effective_cap(record, binding)


@pytest.mark.parametrize("refusal", ["no_credit", "too_many_credits", "not_exhausted", "outcome", "final",
                                    "earlier_incomplete", "missing_earlier", "latest_complete", "known_outcome"])
def test_cap_extension_refusals(cap_evidence, refusal):
    from teachers.chain_recovery import cap_extension
    source, work, record = cap_evidence
    if refusal == "no_credit":
        record["credit"] = 0
    elif refusal == "too_many_credits":
        record["credit"] = record["cap"]
    elif refusal == "not_exhausted":
        record["old_attempts"] -= 1
    elif refusal == "outcome":
        write_json(work / "outcome.json", {})
    elif refusal == "final":
        record["final"] = True
    elif refusal == "missing_earlier":
        (work / "attempt-2/started.json").unlink()
    else:
        path = work / ("attempt-2" if refusal == "earlier_incomplete" else "attempt-3") / "episode/episode.json"
        episode = read_json(path)
        episode.update({"success": False} if refusal == "known_outcome" else {"complete": refusal == "latest_complete"})
        write_json(path, episode)
    with pytest.raises(ValueError, match="Ineligible cap extension"):
        cap_extension(source, work, record, "adftp", artifact_hashes(source))


@pytest.mark.parametrize("error", ["TeacherCallError", "CollectorHalted", "CollectionBudgetExceeded", None])
def test_cap_extension_requires_exact_terminal_error(cap_evidence, error):
    from sed.storage import BlobStore
    from teachers.chain_recovery import cap_extension
    source, work, record = cap_evidence
    directory = work / "attempt-3/episode"
    event = read_json(directory / "events.jsonl")
    event["verification_ref"] = BlobStore(directory / "blobs").put({"audit": {"error_type": error}})
    write_json(directory / "events.jsonl", event)
    assert cap_extension(source, work, record, "adftp", artifact_hashes(source)) is None


def test_cap_extension_refuses_budget_failure_before_latest(cap_evidence):
    from sed.storage import BlobStore
    from teachers.chain_recovery import cap_extension
    source, work, record = cap_evidence
    directory = work / "attempt-2/episode"
    event = read_json(directory / "events.jsonl")
    event["verification_ref"] = BlobStore(directory / "blobs").put({"audit": {"error_type": "BudgetExceeded"}})
    write_json(directory / "events.jsonl", event)
    with pytest.raises(ValueError, match="latest attempt"):
        cap_extension(source, work, record, "adftp", artifact_hashes(source))


def test_cap_extension_refuses_changed_terminal_blob(cap_evidence):
    from teachers.chain_recovery import cap_extension
    source, work, record = cap_evidence
    directory = work / "attempt-3/episode"
    ref = read_json(directory / "events.jsonl")["verification_ref"].removeprefix("sha256:")
    write_json(directory / "blobs" / ref[:2] / (ref + ".json"), {})
    with pytest.raises(ValueError, match="Blob hash mismatch"):
        cap_extension(source, work, record, "adftp", artifact_hashes(source))


def long_environment(config):
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    return lambda task: Environment(config["environment"]["data_root"],
        max_episode_steps=config["environment"]["max_episode_steps"], backend_factory=FakeBackend)


def test_ftp_cap_extension_fresh_execution_and_inherited_usage(seed_bank, tmp_path, monkeypatch):
    from baselines.adftp.teacher import CALLS_PER_ATTEMPT, CALLS_PER_TASK
    from teachers.capped import BudgetExceeded
    from teachers.recover import recover
    from teachers.chain_recovery import extension_file
    monkeypatch.setattr("teachers.openai_teacher.utc_now", lambda: "2026-10-20T22:45:30+00:00")
    support, manifest, freeze, config, _ = setup_source(seed_bank, tmp_path)
    old, new, calls = tmp_path / "old", tmp_path / "new", []
    kw = dict(rate_limit_dir=tmp_path / "limiter", env_factory=long_environment(config))
    with pytest.raises(TeacherCallError):
        ftp.collect(support, manifest, freeze, config, old,
                    teacher_factory=quota_factory(stub_factory([]), calls, fail=True, fail_after=1), **kw)
    counter = next((old / "work").glob("*/teacher/counter.json"))
    recover(counter, reason="transport_fixed")
    good = quota_factory(stub_factory([]), calls)
    with pytest.raises(BudgetExceeded):
        ftp.collect(support, manifest, freeze, config, old, teacher_factory=good, **kw)
    before = artifact_hashes(old)
    plan = recover_chain_collection(old, new, "adftp", manifest, freeze, config, dry_run=True)
    assert not new.exists() and artifact_hashes(old) == before
    assert plan["extensions"] == 1 and plan["credits"] == 1
    assert plan["tasks_with_fresh_attempt_cap_warning"] == 0
    extension = plan["cap_extensions"][0]
    assert extension["attempt_calls"] == CALLS_PER_ATTEMPT - 2
    assert plan["total_extension_calls"] == CALLS_PER_ATTEMPT - 1
    assert plan["tasks"][0]["remaining_cap"] == CALLS_PER_ATTEMPT
    recover_chain_collection(old, new, "adftp", manifest, freeze, config)
    ext_path = extension_file(new, extension["target_counter"])
    write_json(ext_path, {**extension, "extension": extension["extension"] + 1})
    with pytest.raises(ValueError, match="Cap-extension hash"):
        ftp.collect(support, manifest, freeze, config, new, teacher_factory=good, **kw)
    assert len(calls) == CALLS_PER_TASK
    write_json(ext_path, extension)
    root = ftp.collect(support, manifest, freeze, config, new, teacher_factory=good, **kw)
    result = load_verified_collection(root, manifest, freeze)[0]
    assert len(calls) == CALLS_PER_TASK + CALLS_PER_ATTEMPT == result["api_attempts"]
    assert result["inherited_usage"]["http_calls"] == CALLS_PER_TASK
    assert result["inherited_usage"]["known_billed_tokens"]["input_tokens"] == (CALLS_PER_TASK - 1) * 12
    assert result["new_teacher_calls"] == CALLS_PER_ATTEMPT
    assert result["new_usage"]["known_billed_tokens"]["input_tokens"] == CALLS_PER_ATTEMPT * 12
    assert result["total_extension_calls"] == CALLS_PER_ATTEMPT - 1
    assert artifact_hashes(new / extension["attempt_directory"]) == artifact_hashes(old / extension["attempt_directory"])
    assert read_json(new / extension["fresh_attempt_directory"] / "episode/episode.json")["complete"] is True
    assert read_json(new / extension["target_counter"])["attempts"] == extension["effective_cap"]
    from sed.manifest import resolve_task
    task = resolve_task(manifest, freeze, extension["task_id"])
    teacher = good(config, (new / extension["target_counter"]).parent, task, "test", kw["rate_limit_dir"])
    with pytest.raises(BudgetExceeded):
        teacher.generate([], task, {"observation": "No extra attempt", "admissible_commands": ["look"]})
    with pytest.raises(BudgetExceeded):
        recover(new / extension["target_counter"], reason="transport_fixed")
    ftp.collect(support, manifest, freeze, config, new, teacher_factory=good, **kw)
    assert len(calls) == CALLS_PER_TASK + CALLS_PER_ATTEMPT
    assert artifact_hashes(old) == before


def test_smartad_cap_extension_same_slot_and_usage(seed_bank, tmp_path, monkeypatch):
    from baselines.smartad.teacher import CALLS_PER_CANDIDATE
    from teachers.capped import BudgetExceeded
    from teachers.recover import recover
    from teachers.chain_recovery import collection_accounting
    monkeypatch.setattr("teachers.openai_teacher.utc_now", lambda: "2026-10-20T22:45:30+00:00")
    support, manifest, freeze, config, _ = setup_source(seed_bank, tmp_path)
    old, new, calls = tmp_path / "old", tmp_path / "new", []
    def interrupted(*args):
        teacher = factory(calls)(*args)
        response = teacher.teacher.client.client.responses.with_raw_response
        original, failures = response.create, 0
        def create(**kwargs):
            nonlocal failures
            if failures < 2:
                failures += 1
                raise RateLimitError("SECRET")
            return original(**kwargs)
        response.create = create
        generate = teacher.generate
        def acknowledged(*args):
            while True:
                try:
                    return generate(*args)
                except BudgetExceeded:
                    raise
                except TeacherCallError:
                    recover(teacher.counter_path, reason="transport_fixed")
        teacher.generate = acknowledged
        return teacher
    def long_win(task):
        from conftest import FakeBackend
        from harness.alfworld_env import Environment
        def backend(path, seed):
            value = FakeBackend(path, seed)
            value.win_at = config["environment"]["max_episode_steps"]
            return value
        return Environment(config["environment"]["data_root"], max_episode_steps=config["environment"]["max_episode_steps"], backend_factory=backend)
    kw = dict(rate_limit_dir=tmp_path / "limiter", env_factory=long_win, shard_count=10, shard_index=1)
    with pytest.raises(BudgetExceeded):
        sm.collect(support, manifest, freeze, config, old, teacher_factory=interrupted, **kw)
    before = artifact_hashes(old)
    plan = recover_chain_collection(old, new, "smartad", manifest, freeze, config)
    assert plan["credits"] == 2 and plan["extensions"] == 1
    extension = plan["cap_extensions"][0]
    assert extension["candidate_id"] == "candidate-03"
    assert extension["extension"] == CALLS_PER_CANDIDATE - 2
    assert plan["tasks"][0]["counters"][0]["remaining_cap"] == CALLS_PER_CANDIDATE
    sm.collect(support, manifest, freeze, config, new, teacher_factory=factory(calls), **kw)
    usage = collection_accounting(new, read_json(new / "identity.json"))
    assert usage["inherited_usage"]["http_calls"] == CALLS_PER_CANDIDATE
    assert usage["inherited_usage"]["known_billed_tokens"]["input_tokens"] == (CALLS_PER_CANDIDATE - 2) * 12
    assert usage["new_teacher_calls"] == config["environment"]["max_episode_steps"]
    assert usage["new_usage"]["known_billed_tokens"]["input_tokens"] == usage["new_teacher_calls"] * 12
    assert artifact_hashes(old) == before
    slot = new / Path(extension["attempt_directory"])
    assert read_json(slot / "candidate.json")["correct"] is True
    for filename, sha in artifact_hashes(old / extension["attempt_directory"]).items():
        assert artifact_hashes(slot)[filename] == sha
    assert read_json(new / extension["fresh_attempt_directory"] / "episode/episode.json")["complete"] is True
    count = len(calls)
    sm.collect(support, manifest, freeze, config, new, teacher_factory=factory(calls), **kw)
    assert len(calls) == count
