import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest

from teachers.capped import CappedTeacher, BudgetExceeded, CollectorHalted
from teachers.openai_teacher import TeacherCallError
from teachers.capped import SharedReservations, CollectionBudgetExceeded
from teachers.openai_teacher import Teacher
from teachers.rate_limit import reserved_tokens
from test_stage2c_teacher import FakeClock


class RawClient:
    max_retries = 0
    def __init__(self, texts=None, status=200, error=None):
        self.texts = list(texts or ["Action: look"] * 20)
        self.status, self.error, self.requests = status, error, []
        self.responses = NS(with_raw_response=self)
    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        raw = self.texts.pop(0)
        parsed = {"output_text": raw, "status": "completed", "model": "gpt-6-luna",
                  "id": "fake", "_request_id": "fake-request", "usage": {
                      "input_tokens": 13, "output_tokens": 7, "total_tokens": 20}}
        return NS(status_code=self.status, headers={}, parse=lambda: parsed)


def capped(tmp_path, client, *, run="test", cap=2, dry_run=False, **kwargs):
    return CappedTeacher(counter_path=tmp_path / (run + ".counter.json"), cap=cap,
        cache_dir=tmp_path / "cache", usage_ledger=tmp_path / "usage.jsonl", run_id=run,
        client=client, dry_run=dry_run, sleep=kwargs.pop("sleep", lambda _: None), **kwargs)


def test_cap_restart_cache_and_cross_run_dedupe(tmp_path, task, visible):
    client = RawClient()
    teacher = capped(tmp_path, client, cap=1)
    first = teacher.generate([], task, visible)
    assert len(client.requests) == 1 and first.metadata["usage"]["total_tokens"] == 20
    assert capped(tmp_path, client, cap=1).generate([], task, visible).metadata["cache_hit"]
    assert capped(tmp_path, client, run="R2", cap=0).generate([], task, visible).metadata["cache_hit"]
    with pytest.raises(BudgetExceeded):
        teacher.generate([], task, dict(visible, observation="another"))
    assert len(client.requests) == 1
    assert json.loads((tmp_path / "test.counter.json").read_text())["attempts"] == 1
    counter = tmp_path / "test.counter.json"
    counter.unlink()
    with pytest.raises(BudgetExceeded):
        teacher.generate([], task, dict(visible, observation="another"))


@pytest.mark.parametrize("status", [201, 400, 401, 403, 408, 409, 500, 503])
def test_non_200_never_retries_and_halts_restart(tmp_path, task, visible, status):
    client = RawClient(status=status)
    with pytest.raises(TeacherCallError):
        capped(tmp_path, client).generate([], task, visible)
    with pytest.raises(CollectorHalted):
        capped(tmp_path, client).generate([], task, visible)
    assert len(client.requests) == 1
    assert json.loads((tmp_path / "test.counter.json").read_text())["attempts"] == 1
    rows = [json.loads(s) for s in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert rows[-1]["http_status"] == status and rows[-1]["will_retry"] is False


def test_empty_and_timeout_raise_no_secret_persistence(tmp_path, task, visible):
    client = RawClient(error=TimeoutError("SECRET_RAW_EXCEPTION"))
    with pytest.raises(TeacherCallError, match="sanitized"):
        capped(tmp_path, client).generate([], task, visible)
    assert "SECRET_RAW_EXCEPTION" not in (tmp_path / "usage.jsonl").read_text()
    with pytest.raises(TeacherCallError, match="Empty"):
        capped(tmp_path / "empty", RawClient(texts=[""])).generate([], task, visible)


def test_dry_run_reads_cache_dedupes_without_transport_or_counter(tmp_path, task, visible):
    client = RawClient()
    teacher = capped(tmp_path, client, cap=1, dry_run=True)
    assert teacher.generate([], task, visible).metadata["would_call"]
    assert not teacher.generate([], task, visible).metadata["would_call"]
    assert not teacher.generate([], task, dict(visible, observation="new")).metadata["within_cap"]
    assert not client.requests and not list(tmp_path.iterdir())


def test_concurrent_collectors_cannot_exceed_cap(tmp_path, task, visible):
    client = RawClient()
    def run(i):
        try:
            capped(tmp_path, client, cap=1).generate([], task, dict(visible, observation=str(i)))
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(run, [0, 1])) == [False, True]
    assert len(client.requests) == 1


def test_pending_is_unknown_not_free_budget(tmp_path, task, visible):
    t = capped(tmp_path, RawClient())
    value = {"identity": t.identity, "attempts": 1, "pending": True, "halted": False}
    (tmp_path / "test.counter.json").write_text(json.dumps(value))
    with pytest.raises(CollectorHalted):
        t.generate([], task, visible)


@pytest.mark.parametrize("allow_retry", [True, False])
def test_shared_reservations_bound_retry_inputs_and_count_one_logical_call(tmp_path, task, visible, allow_retry):
    class RetryClient(RawClient):
        def create(self, **kwargs):
            if not self.requests:
                self.requests.append(kwargs)
                raise TimeoutError("SECRET_RETRY_EXCEPTION")
            return super().create(**kwargs)
    client = RetryClient()
    teacher = Teacher(cache_dir=tmp_path / "cache", usage_ledger=tmp_path / "usage.jsonl",
                      run_id="shared", client=client, max_attempts=2, sleep=lambda _: None)
    tokens = reserved_tokens(teacher.prompt([], task, visible), 0)
    budget = SharedReservations(tmp_path / "budget.sqlite3", max_calls=1,
                                max_input_tokens=tokens * (2 if allow_retry else 1))
    teacher.reservation_budget = budget
    if allow_retry:
        result = teacher.generate([], task, visible)
        assert result.parsed_action == "look"
        assert teacher.generate([], task, visible).metadata["cache_hit"]
        assert len(client.requests) == 2
    else:
        with pytest.raises(CollectionBudgetExceeded):
            teacher.generate([], task, visible)
        assert len(client.requests) == 1
    summary = budget.snapshot()
    assert summary["logical_calls"] == 1
    assert summary["transport_reservations"] == (2 if allow_retry else 1)
    assert summary["reserved_input_tokens"] == tokens * summary["transport_reservations"]
    assert "SECRET_RETRY_EXCEPTION" not in (tmp_path / "usage.jsonl").read_text()
    restarted = SharedReservations(tmp_path / "budget.sqlite3", **summary["limits"])
    assert restarted.snapshot() == summary
    with pytest.raises(ValueError, match="caps changed"):
        SharedReservations(tmp_path / "budget.sqlite3", max_calls=2)


class SequenceClient:
    max_retries = 0

    def __init__(self, results):
        self.results, self.requests = list(results), []
        self.responses = NS(with_raw_response=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def http_429(*, body=None, headers=None, raw=False):
    body = body if body is not None else {"error": {"code": "rate_limit_exceeded"}}
    headers = {"x-request-id": "req_stub_429", **(headers or {})}
    if raw:
        return NS(status_code=429, headers=headers, parse=lambda: body)
    # Same public error attributes as the SDK; no live HTTP client.
    class RateLimitError(Exception):
        status_code = 429
    error = RateLimitError("PRIVATE_PROVIDER_EXCEPTION")
    error.body, error.response, error.request_id = body, NS(headers=headers), "req_stub_429"
    return error


def retry_teacher(tmp_path, results, **kwargs):
    clock, client = FakeClock(), SequenceClient(results)
    teacher = capped(tmp_path, client, clock=clock, sleep=clock.sleep,
                     jitter=kwargs.pop("jitter", lambda low, high: high / 2), **kwargs)
    return teacher, clock, client


def usage_rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("body", [{}, {"error": {"code": "rate_limit_exceeded"}},
                                  {"type": "rate_limit_exceeded"}])
def test_rate_limit_retries_one_logical_slot_and_recovers_counter(tmp_path, task, visible, raw, body):
    success = RawClient().create()
    teacher, clock, client = retry_teacher(tmp_path, [http_429(raw=raw, body=body), success], cap=1)
    result = teacher.generate([], task, visible)
    assert result.parsed_action == "look" and result.metadata["attempt"] == 2
    assert len(client.requests) == 2 and clock.waits == [2.5]
    assert client.requests[0] == client.requests[1]
    rows = usage_rows(tmp_path)
    assert [r["status"] for r in rows] == ["started", "failed", "started", "success", "state_link"]
    attempts = rows[:-1]
    assert len({r["logical_call_id"] for r in attempts}) == 1
    assert len({r["call_id"] for r in attempts}) == 2
    tokens = reserved_tokens(teacher.teacher.prompt([], task, visible), 512)
    assert all(r["reserved_tokens"] == tokens for r in attempts)
    assert rows[1]["usage"] is None and rows[1]["request_id"] == "req_stub_429"
    assert rows[1]["retryable"] and rows[1]["will_retry"]
    assert rows[3]["usage"]["total_tokens"] == 20
    assert rows[-1]["attempts_reserved"] == 1
    state = json.loads(teacher.counter_path.read_text())
    assert state["attempts"] == 1 and not state["halted"] and not state["pending"]
    # A resolved 429 is safe on restart, even when only the ledger survives.
    for remove_counter in (False, True):
        if remove_counter:
            teacher.counter_path.unlink()
        restarted = capped(tmp_path, client, cap=1, clock=clock, sleep=clock.sleep)
        assert restarted.generate([], task, visible).metadata["cache_hit"]
        with pytest.raises(BudgetExceeded):
            restarted.generate([], task, dict(visible, observation="new logical call"))
    assert len(client.requests) == 2
    assert "PRIVATE_PROVIDER" not in (tmp_path / "usage.jsonl").read_text()


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("header,expected", [("17", 17), ("0", 0), ("350", 350),
    ("Tue, 14 Nov 2023 22:13:30 GMT", 10), ("NaN", 2.5), ("invalid", 2.5)])
def test_capped_retry_after(tmp_path, task, visible, raw, header, expected):
    teacher, clock, client = retry_teacher(tmp_path, [
        http_429(raw=raw, headers={"Retry-After": header}), RawClient().create()])
    teacher.generate([], task, visible)
    assert clock.waits == [expected] and len(client.requests) == 2
    assert usage_rows(tmp_path)[1]["wait_seconds"] == expected


def test_twenty_consecutive_rate_limits_halt_with_nonrefundable_reservations(tmp_path, task, visible):
    bounds = []
    def jitter(low, high):
        bounds.append((low, high))
        return high / 2
    teacher, clock, client = retry_teacher(tmp_path, [http_429() for _ in range(21)],
                                          cap=1, jitter=jitter)
    tokens = reserved_tokens(teacher.teacher.prompt([], task, visible), 0)
    budget = SharedReservations(tmp_path / "budget.sqlite3", max_calls=1, max_input_tokens=20 * tokens)
    teacher.teacher.reservation_budget = budget
    with pytest.raises(TeacherCallError, match="sanitized"):
        teacher.generate([], task, visible)
    assert len(client.requests) == 20
    assert bounds == [(0, min(5 * 2 ** i, 300)) for i in range(19)]
    assert clock.waits == [high / 2 for _, high in bounds]
    rows = usage_rows(tmp_path)
    assert len(rows) == 40 and all(r["usage"] is None for r in rows)
    assert len({r["logical_call_id"] for r in rows}) == 1
    assert len({r["call_id"] for r in rows}) == 20
    assert all(r["will_retry"] for r in rows[1:-1:2])
    assert not rows[-1]["will_retry"] and rows[-1]["wait_seconds"] == 0
    assert budget.snapshot()["logical_calls"] == 1
    assert budget.snapshot()["transport_reservations"] == 20
    assert budget.snapshot()["reserved_input_tokens"] == 20 * tokens
    state = json.loads(teacher.counter_path.read_text())
    assert state["halted"] and state["attempts"] == 1 and not state["pending"]
    for remove_counter in (False, True):
        if remove_counter:
            teacher.counter_path.unlink()
        with pytest.raises(CollectorHalted):
            capped(tmp_path, client, cap=1).generate([], task, visible)
    assert len(client.requests) == 20


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("body", [
    {"error": {"code": "insufficient_quota"}}, {"type": "insufficient_quota"},
    {"error": {"code": "billing_hard_limit_reached"}},
    {"error": {"type": "credit_balance_exhausted"}},
    {"error": {"code": "rate_limit_exceeded", "message": "Check your billing details"}},
    {"message": "Insufficient credits. PRIVATE_PROVIDER_BODY"},
    "insufficient_quota PRIVATE_PROVIDER_BODY",
])
def test_quota_429_halts_immediately_without_retry_or_body_leak(tmp_path, task, visible, raw, body):
    teacher, clock, client = retry_teacher(tmp_path, [
        http_429(raw=raw, body=body, headers={"Retry-After": "12"}), RawClient().create()])
    with pytest.raises(TeacherCallError, match="sanitized"):
        teacher.generate([], task, visible)
    assert len(client.requests) == 1 and not clock.waits
    failed = usage_rows(tmp_path)[-1]
    assert failed["http_status"] == 429 and not failed["retryable"] and not failed["will_retry"]
    assert failed["wait_seconds"] == 0 and failed["usage"] is None
    with pytest.raises(CollectorHalted):
        capped(tmp_path, client).generate([], task, visible)
    assert all("PRIVATE_PROVIDER" not in p.read_text() for p in tmp_path.rglob("*") if p.is_file())


@pytest.mark.parametrize("allow_retry", [False, True])
def test_capped_retries_obey_shared_input_cap(tmp_path, task, visible, allow_retry):
    teacher, _, client = retry_teacher(tmp_path, [http_429(), RawClient().create()], cap=1)
    tokens = reserved_tokens(teacher.teacher.prompt([], task, visible), 0)
    budget = SharedReservations(tmp_path / "budget.sqlite3", max_calls=1,
                                max_input_tokens=tokens * (2 if allow_retry else 1))
    teacher.teacher.reservation_budget = budget
    if allow_retry:
        assert teacher.generate([], task, visible).parsed_action == "look"
    else:
        with pytest.raises(CollectionBudgetExceeded):
            teacher.generate([], task, visible)
        with pytest.raises(CollectorHalted):
            teacher.generate([], task, visible)
    attempts = 2 if allow_retry else 1
    assert len(client.requests) == attempts
    assert len([r for r in usage_rows(tmp_path) if r["status"] == "started"]) == attempts
    state = budget.snapshot()
    assert state["logical_calls"] == 1 and state["transport_reservations"] == attempts
    assert state["reserved_input_tokens"] == attempts * tokens
    assert state["stopped"] == (None if allow_retry else "reserved_input_token_cap")


def test_interrupted_rate_limit_backoff_halts_even_without_counter(tmp_path, task, visible):
    teacher, _, client = retry_teacher(tmp_path, [http_429(), RawClient().create()])
    def interrupt(_):
        raise KeyboardInterrupt()
    teacher.teacher.sleep = interrupt
    with pytest.raises(KeyboardInterrupt):
        teacher.generate([], task, visible)
    teacher.counter_path.unlink()
    with pytest.raises(CollectorHalted):
        capped(tmp_path, client).generate([], task, visible)
    assert len(client.requests) == 1


@pytest.mark.parametrize("failure", [TimeoutError(), ConnectionError(), NS(status_code=503, headers={})])
def test_other_failure_after_429_still_halts(tmp_path, task, visible, failure):
    teacher, _, client = retry_teacher(tmp_path, [http_429(), failure, RawClient().create()])
    with pytest.raises(TeacherCallError):
        teacher.generate([], task, visible)
    assert len(client.requests) == 2 and not usage_rows(tmp_path)[-1]["will_retry"]
    with pytest.raises(CollectorHalted):
        capped(tmp_path, client).generate([], task, visible)


def test_legacy_ledger_failed_429_stays_halted(tmp_path, task, visible):
    teacher, _, client = retry_teacher(tmp_path, [http_429(), RawClient().create()])
    teacher.generate([], task, visible)
    # Old no-retry runs have no logical IDs and cannot be revived by this change.
    rows = usage_rows(tmp_path)[:2]
    for row in rows:
        row.pop("logical_call_id")
    rows[-1]["will_retry"] = False
    (tmp_path / "usage.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    teacher.counter_path.unlink()
    with pytest.raises(CollectorHalted):
        capped(tmp_path, client).generate([], task, visible)


def test_next_uncached_call_after_recovered_429_uses_next_slot(tmp_path, task, visible):
    teacher, clock, client = retry_teacher(tmp_path, [http_429(), RawClient().create(),
                                                    http_429(), RawClient().create()], cap=2)
    teacher.generate([], task, visible)
    restarted = capped(tmp_path, client, cap=2, clock=clock, sleep=clock.sleep)
    restarted.generate([], task, dict(visible, observation="second logical call"))
    state = json.loads(teacher.counter_path.read_text())
    assert len(client.requests) == 4 and state["attempts"] == 2 and not state["halted"]
    starts = [r for r in usage_rows(tmp_path) if r["status"] == "started"]
    assert len(starts) == 4 and len({r["logical_call_id"] for r in starts}) == 2


def test_capped_429_reservations_still_consume_limiter_capacity(tmp_path, task, visible):
    teacher, clock, client = retry_teacher(tmp_path, [http_429(), RawClient().create()],
                                          requests_per_minute=1, cap=1)
    teacher.generate([], task, visible)
    assert clock.waits == [2.5, 57.5] and len(client.requests) == 2
    rows = usage_rows(tmp_path)
    assert rows[2]["limiter_wait_seconds"] == 57.5
    assert rows[0]["reserved_tokens"] == rows[2]["reserved_tokens"]


def test_installed_sdk_rate_limit_error_retries_with_stub_client(tmp_path, task, visible):
    import httpx
    from openai import RateLimitError
    response = httpx.Response(429, headers={"Retry-After": "9"},
                              request=httpx.Request("POST", "https://example.invalid/responses"))
    error = RateLimitError("PRIVATE_SDK_MESSAGE", response=response,
                          body={"code": "rate_limit_exceeded", "type": "tokens"})
    teacher, clock, client = retry_teacher(tmp_path, [error, RawClient().create()])
    assert teacher.generate([], task, visible).parsed_action == "look"
    assert clock.waits == [9] and len(client.requests) == 2
    assert "PRIVATE_SDK_MESSAGE" not in (tmp_path / "usage.jsonl").read_text()
