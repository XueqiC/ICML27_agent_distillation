"""Upstream-equivalent baseline retries, with no network or real sleeps."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from teachers.baseline_transport import SOURCES, baseline_transport
from teachers.openai_teacher import Teacher, TeacherCallError, TeacherRateLimitError, TeacherQuotaError
from teachers.stage import teacher_stage
from test_capped_teacher import RawClient, SequenceClient, capped, http_429, usage_rows
from test_stage2c_teacher import FakeClock
from test_stage2b_seeds import seed_bank, factories
from test_full_train import full_bank
from test_stage3d import budget, STATE, correction
from test_teacher import FakeClient, response


def http_error(status, headers=None):
    error = Exception("PRIVATE_PROVIDER_ERROR")
    error.status_code, error.response = status, NS(headers=headers or {})
    return error


@pytest.mark.parametrize("method", SOURCES)
@pytest.mark.parametrize("capped_path", [False, True])
def test_client_receives_sdk_timeout(tmp_path, task, visible, monkeypatch, method, capped_path):
    import openai
    constructors = []
    def client(**kwargs):
        constructors.append(kwargs)
        return RawClient()
    monkeypatch.setattr(openai, "OpenAI", client)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-stub")
    if capped_path:
        teacher = capped(tmp_path, None, baseline_method=method)
    else:
        teacher = Teacher(cache_dir=tmp_path / "cache", usage_ledger=tmp_path / "usage.jsonl",
                          run_id="baseline", baseline_method=method)
    result = teacher.generate([], task, visible)
    timeout = constructors[0]["timeout"]
    assert (timeout.read, timeout.write, timeout.pool, timeout.connect) == (600, 600, 600, 5)
    assert constructors[0]["max_retries"] == 0  # audited retries, no hidden SDK attempts
    assert result.metadata["baseline_transport"] == baseline_transport(method).disclosure()


def transport(tmp_path, results, **kwargs):
    client, clock = SequenceClient(results), FakeClock()
    teacher = capped(tmp_path, client, baseline_method="had", cap=1,
                     clock=clock, sleep=clock.sleep, jitter=lambda low, high: 0, **kwargs)
    return teacher, client, clock


@pytest.mark.parametrize("failure", [TimeoutError(), ConnectionError(), *[http_error(s) for s in (408, 409, 500, 503)]])
def test_http_attempts_recover_in_one_slot_and_survive_restart(tmp_path, task, visible, failure):
    teacher, client, clock = transport(tmp_path, [failure, failure, RawClient().create()])
    teacher.generate([], task, visible)
    assert len(client.requests) == 3 and clock.waits == [.5, 1.0]
    assert teacher._read_counter()["attempts"] == 1
    teacher.check_health()
    # Reconstruction must accept recovered non-429 attempts, not mark the cell halted.
    again = capped(tmp_path, client, baseline_method="had", cap=1)
    assert again.generate([], task, visible).metadata["cache_hit"]
    rows = usage_rows(tmp_path)
    assert len({r["logical_call_id"] for r in rows if r["status"] == "started"}) == 1
    assert len({r["call_id"] for r in rows if r["status"] == "started"}) == 3
    assert "PRIVATE_PROVIDER_ERROR" not in (tmp_path / "usage.jsonl").read_text()


def test_interleaved_429s_do_not_spend_http_budget_or_slots(tmp_path, task, visible):
    teacher, client, clock = transport(tmp_path, [http_429(headers={"retry-after": "7"}),
        TimeoutError(), http_429(), http_error(503), http_429(), RawClient().create()])
    teacher.generate([], task, visible)
    assert len(client.requests) == 6 and teacher._read_counter()["attempts"] == 1
    assert clock.waits == [7, .5, 0, 1, 0]
    assert [r["http_attempt"] for r in usage_rows(tmp_path) if r["status"] == "started"] == [1, 1, 2, 2, 3, 3]
    teacher.check_health()


@pytest.mark.parametrize("headers,expected", [({"retry-after-ms": "1250"}, 1.25),
    ({"retry-after": "3"}, 3), ({"retry-after": "61"}, .5)])
def test_sdk_retry_after(tmp_path, task, visible, headers, expected):
    teacher, _, clock = transport(tmp_path, [http_error(503, headers), RawClient().create()])
    teacher.generate([], task, visible)
    assert clock.waits == [expected]


@pytest.mark.parametrize("status,headers,count", [(400, {}, 1), (503, {"x-should-retry": "false"}, 1),
    (400, {"x-should-retry": "true"}, 3), (503, {}, 3)])
def test_fail_loud_after_exact_retry_budget(tmp_path, task, visible, status, headers, count):
    teacher, client, _ = transport(tmp_path, [http_error(status, headers)] * 3)
    @teacher_stage("output")
    def stage(output):
        teacher.generate([], task, visible)
        (output / "data.json").write_text("must not publish")
    output = tmp_path / "stage"
    with pytest.raises(TeacherCallError, match="NOT complete"):
        stage(output)
    assert len(client.requests) == count
    assert teacher._read_counter()["halted"]
    assert not (output / "data.json").exists()
    report = json.loads((output / "teacher_report.json").read_text())
    assert report["status"] == "failed" and report["rejections"] == 0
    assert report["transport_failed_attempts"] == count


@pytest.mark.parametrize("quota,attempts,error", [(False, 20, TeacherRateLimitError), (True, 1, TeacherQuotaError)])
def test_429_halts_without_another_slot(tmp_path, task, visible, quota, attempts, error):
    body = {"code": "insufficient_quota" if quota else "rate_limit_exceeded"}
    teacher, client, _ = transport(tmp_path, [http_429(body=body)] * 20)
    with pytest.raises(error):
        teacher.generate([], task, visible)
    assert len(client.requests) == attempts
    assert teacher._read_counter()["attempts"] == 1 and teacher._read_counter()["halted"]


@pytest.mark.parametrize("kind", ["extract", "privileged", "zero"])
@pytest.mark.parametrize("prefix,tries,failures,rejections", [
    ([TimeoutError()] * 2, 1, 0, 0),
    ([TimeoutError()] * 3, 2, 1, 0),
    ([response("bad")], 2, 0, 1),
    ([TimeoutError()] * 3 + [response("bad")], 3, 1, 1),
    ([http_429()] * 4, 1, 0, 0)])
def test_leap_method_try_consumption(tmp_path, task, kind, prefix, tries, failures, rejections):
    good = STATE if kind == "extract" else correction()
    client = FakeClient([*prefix, response(good)])
    teacher, task = budget(tmp_path, task, client, kind=kind)
    result = teacher.request(task, [{"role": "user", "content": "visible request"}],
                             **({} if kind == "extract" else {"length": 2}))
    assert result["status"] == "accepted" and result["attempts_reserved"] == tries
    assert result["transport_failures"] == failures and result["rejections"] == rejections
    assert len(client.requests) == len(prefix) + 1
    assert teacher.request(task, [{"role": "user", "content": "visible request"}],
                           **({} if kind == "extract" else {"length": 2})) == result


def test_leap_all_transport_tries_exhaust_loudly_without_redraw(tmp_path, task):
    client = FakeClient([TimeoutError("PRIVATE")] * 9)
    teacher, task = budget(tmp_path, task, client)
    @teacher_stage("output")
    def stage(output):
        teacher.request(task, [], length=2)
    with pytest.raises(TeacherCallError, match="NOT complete"):
        stage(tmp_path / "stage")
    assert len(client.requests) == 9
    assert len(list(teacher.root.rglob("failure.json"))) == 3
    assert not list(teacher.root.rglob("outcome.json"))
    with pytest.raises(TeacherCallError):
        teacher.request(task, [], length=2)
    assert len(client.requests) == 9
    report = json.loads((tmp_path / "stage/teacher_report.json").read_text())
    assert report["transport_failed_attempts"] == 9 and report["rejections"] == 0


@pytest.mark.parametrize("quota", [False, True])
def test_leap_429_does_not_advance_method_try(tmp_path, task, quota):
    client = FakeClient([http_429(body={"code": "insufficient_quota" if quota else "rate_limit_exceeded"})] * 20)
    teacher, task = budget(tmp_path, task, client)
    with pytest.raises(TeacherCallError):
        teacher.request(task, [], length=2)
    attempts = list(teacher.root.glob("requests/*/attempt-*"))
    assert len(attempts) == 1
    assert json.loads((attempts[0] / "failure.json").read_text())["method_try_consumed"] is False
    assert len(client.requests) == (1 if quota else 20)


@pytest.mark.parametrize("full", [False, True])
def test_verified_seed_loading_and_seed_client_timeout_unchanged(seed_bank, full_bank, tmp_path, monkeypatch, full):
    import openai
    from sed.seed_evidence import load_verified_collection, artifact_hashes
    from sed.seed_selection import select_seeds, select_full_train
    from sed.storage import digest
    from teachers.collect_seeds import collect
    root, manifest, freeze, config = (*full_bank[:3], full_bank[-1]) if full else seed_bank
    original = deepcopy(config)
    assert config["models"]["teacher"]["timeout_seconds"] == 60
    constructors = []
    def client(**kwargs):
        constructors.append(kwargs)
        return RawClient()
    monkeypatch.setattr(openai, "OpenAI", client)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-stub")
    env, _ = factories(root, config, set(), [])
    collection = tmp_path / "verified"
    selection = select_full_train(manifest, freeze) if full else select_seeds(manifest, freeze, k=5)
    collect(selection, manifest, freeze, config, collection, env_factory=env)
    loaded = load_verified_collection(collection, manifest, freeze)
    before = artifact_hashes(collection)
    # Exercise every baseline's opt-in with the same config still in memory.
    for method in SOURCES:
        teacher = Teacher(cache_dir=tmp_path / method, usage_ledger=tmp_path / (method + ".jsonl"),
                          run_id=method, baseline_method=method)
        assert teacher.timeout_seconds == 600
    assert config == original and digest(config) == digest(original)
    assert all(c["timeout"] == 60 and c["max_retries"] == 0 for c in constructors)
    assert len(constructors) == (selection["candidates"] if full else 5)
    assert load_verified_collection(collection, manifest, freeze) == loaded
    assert artifact_hashes(collection) == before


def test_baseline_factory_wiring(tmp_path, task):
    from sed.config import load_config
    from teachers.capped import from_config
    from baselines.coa.teacher import budget as coa_budget
    from baselines.adftp.collect import default_teacher
    from baselines.agentdistill.teacher import Budget
    config = load_config()
    original = deepcopy(config)
    for method in ("gad", "had", "steca"):
        teacher = from_config(config, output=tmp_path / method, cache_dir=tmp_path / method / "cache",
            usage_ledger=tmp_path / method / "usage.jsonl", run_id=method, cap=1, baseline_method=method)
        assert teacher.identity["baseline_transport"] == baseline_transport(method).disclosure()
        assert teacher.teacher.timeout_seconds == 600
    for evaluation in (False, True):
        teacher = coa_budget(config, tmp_path / ("coa" + str(evaluation)), {}, [task], evaluation=evaluation)
        assert teacher.teacher.transport_policy == baseline_transport("coa")
    teacher = default_teacher(config, tmp_path / "adftp", task, {}, tmp_path / "rate_limit")
    assert teacher.teacher.transport_policy == baseline_transport("adftp")
    teacher = Budget(tmp_path / "agentdistill", config, {"unique_instances": 1})
    assert teacher.client.teacher.transport_policy == baseline_transport("agentdistill")
    assert config == original
