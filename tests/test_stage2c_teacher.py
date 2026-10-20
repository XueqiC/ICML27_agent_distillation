import json
import multiprocessing
from types import SimpleNamespace as NS

import pytest

from teachers.openai_teacher import Teacher, TeacherCallError
from teachers.rate_limit import RateLimiter, reserved_tokens
from test_teacher import FakeClient, response


class FakeClock:
    def __init__(self):
        self.now = 1_700_000_000.0
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class HTTPError(Exception):
    def __init__(self, status, headers=None):
        super().__init__("sk-never-log-error-body")
        self.status_code = status
        self.response = NS(headers=headers or {})


class RateLimitError(Exception):
    pass


def make_teacher(tmp_path, results, **kwargs):
    clock = FakeClock()
    client = FakeClient(results)
    model = Teacher(cache_dir=tmp_path / "cache", usage_ledger=tmp_path / "usage.jsonl",
                    run_id="fake", client=client, clock=clock, sleep=clock.sleep,
                    jitter=lambda low, high: high / 2, **kwargs)
    return model, clock, client


def ledger(tmp_path):
    return [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("headers,wait", [({}, 1), ({"Retry-After": "7"}, 7),
    ({"Retry-After": "Tue, 14 Nov 2023 22:13:30 GMT"}, 10),
    ({"Retry-After": "90"}, 90),
    ({"retry-after": "2", "x-ratelimit-reset-requests": "1m2.5s",
      "x-ratelimit-reset-tokens": "250ms"}, 62.5),
    ({"x-ratelimit-reset-tokens": "1700000011"}, 11),
    ({"Retry-After": "NaN", "x-ratelimit-reset-tokens": "secret-invalid"}, 1)])
def test_429_header_delays_and_sanitized_ledger(tmp_path, task, visible, headers, wait):
    model, clock, client = make_teacher(tmp_path, [HTTPError(429, headers), response()])
    assert model.generate([], task, visible).metadata["attempt"] == 2
    assert clock.waits == [wait]
    rows = ledger(tmp_path)
    assert [r["status"] for r in rows] == ["started", "failed", "started", "success"]
    assert rows[1]["http_status"] == 429 and rows[1]["error_type"] == "HTTPError"
    assert rows[1]["wait_seconds"] == wait and rows[1]["will_retry"] is True
    assert len(client.requests) == 2
    model.generate([], task, visible)
    assert len(client.requests) == 2 and ledger(tmp_path)[-1]["status"] == "cache_hit"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            text = path.read_text()
            assert "sk-never-log-error-body" not in text and "secret-invalid" not in text
            assert "Retry-After" not in text


def test_ten_attempts_full_jitter_cap_and_unknown_usage(tmp_path, task, visible):
    model, clock, client = make_teacher(tmp_path, [HTTPError(429) for _ in range(10)])
    bounds = []
    def jitter(low, high):
        bounds.append((low, high))
        return high * .25
    model.jitter = jitter
    with pytest.raises(TeacherCallError):
        model.generate([], task, visible)
    assert bounds == [(0, n) for n in [2, 4, 8, 16, 32, 60, 60, 60, 60]]
    assert clock.waits == [high * .25 for _, high in bounds]
    assert len(client.requests) == 10
    rows = ledger(tmp_path)
    assert len(rows) == 20 and all(r["usage"] is None for r in rows)
    assert rows[-1]["wait_seconds"] == 0 and rows[-1]["will_retry"] is False


@pytest.mark.parametrize("error", [HTTPError(500), HTTPError(502), HTTPError(503), HTTPError(504),
                                  HTTPError(599), TimeoutError(), ConnectionError(), RateLimitError()])
def test_transient_failures_retry(tmp_path, task, visible, error):
    model, clock, client = make_teacher(tmp_path, [error, response()])
    assert model.generate([], task, visible).metadata["attempt"] == 2
    assert clock.waits == [1] and len(client.requests) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_nonretryable_fail_fast(tmp_path, task, visible, status):
    model, clock, client = make_teacher(tmp_path, [HTTPError(status, {"Retry-After": "30"}), response()])
    with pytest.raises(TeacherCallError):
        model.generate([], task, visible)
    assert not clock.waits and len(client.requests) == 1
    assert ledger(tmp_path)[-1]["wait_seconds"] == 0


def test_failed_call_cooldown_shared_with_another_worker(tmp_path, task, visible):
    model, clock, _ = make_teacher(tmp_path, [HTTPError(429, {"Retry-After": "12"})], max_attempts=1)
    with pytest.raises(TeacherCallError):
        model.generate([], task, visible)
    other = RateLimiter(tmp_path / "rate_limit", "gpt-6-luna", clock=clock, sleep=clock.sleep)
    result = other.acquire(1)
    assert clock.waits == [12] and result["limiter_wait_seconds"] == 12


@pytest.mark.parametrize("prompt,cap,expected", [
    ([], 2048, 2057),  # Two JSON characters, one framing allowance.
    ([{"role": "user", "content": "ab"}], 2048, 2072),  # 32 characters / 4.
    ([{"role": "user", "content": "abc"}], 2048, 2073),  # Round 33 / 4 up.
    ([{"role": "user", "content": "é🙂"}], 2048, 2072),  # Characters, not bytes.
    ([{"role": "user", "content": "é🙂" * 100}], 512, 586),
    ([{"role": "system", "content": "é🙂"}, {"role": "user", "content": "abc"}],
     512, 553),  # 66 characters, three framing allowances.
])
def test_token_reservation_character_estimate(prompt, cap, expected):
    assert reserved_tokens(prompt, cap) == expected


def test_token_reservations_and_changed_limits_fail_closed(tmp_path):
    clock = FakeClock()
    limiter = RateLimiter(tmp_path, "model", requests_per_minute=3, tokens_per_minute=100,
                          clock=clock, sleep=clock.sleep)
    limiter.acquire(60)
    assert limiter.acquire(60)["limiter_wait_seconds"] == 60
    with pytest.raises(ValueError, match="exceeds"):
        limiter.acquire(101)
    with pytest.raises(ValueError, match="configuration changed"):
        RateLimiter(tmp_path, "model", clock=clock, sleep=clock.sleep).acquire(1)


@pytest.mark.parametrize("text,status", [("Action: look", "success"), ("invalid reply", "rejected")])
def test_raw_response_rate_limit_headers_allowlisted_per_attempt(tmp_path, task, visible, text, status):
    counters = {"x-ratelimit-limit-requests": 500, "x-ratelimit-limit-tokens": 200000,
                "x-ratelimit-remaining-requests": 499, "x-ratelimit-remaining-tokens": 198000}
    headers = {k.upper(): str(v) for k, v in counters.items()}
    headers.update({"Authorization": "Bearer sk-never-log-header",
                    "Set-Cookie": "sk-never-log-cookie", "x-private": "sk-never-log-private"})
    parsed = []
    def parse():
        parsed.append(True)
        return response(text)
    raw = NS(headers=headers, parse=parse)
    model, clock, client = make_teacher(tmp_path, [
        HTTPError(429, {**headers, "X-RATELIMIT-REMAINING-REQUESTS": "0", "Retry-After": "7"}), raw])
    # Match the installed SDK: headers live on the raw wrapper, not its parsed body.
    client.responses = NS(with_raw_response=NS(create=client.create))
    result = model.generate([], task, visible)
    rows = ledger(tmp_path)
    assert [r["status"] for r in rows] == ["started", "failed", "started", status]
    assert rows[1]["rate_limit_headers"] == dict(counters, **{"x-ratelimit-remaining-requests": 0})
    assert rows[3]["rate_limit_headers"] == counters
    assert all("rate_limit_headers" not in rows[i] for i in (0, 2))
    assert clock.waits == [7] and len(client.requests) == 2 and parsed == [True]
    assert result.metadata["usage"]["input_tokens"] == 13
    assert result.parsed_action == ("look" if status == "success" else None)
    if status == "success":
        model.generate([], task, visible)
        assert len(client.requests) == 2 and ledger(tmp_path)[-1]["status"] == "cache_hit"
        assert "rate_limit_headers" not in ledger(tmp_path)[-1]
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "sk-never-log" not in path.read_text()


def test_partial_and_malformed_rate_limit_headers(tmp_path, task, visible):
    reply = response()
    reply.headers = {"X-RateLimit-Limit-Requests": " 500 ",
                     "x-ratelimit-limit-tokens": "sk-never-log-header",
                     "x-ratelimit-remaining-requests": "-1",
                     "x-ratelimit-remaining-tokens": "NaN"}
    model, _, _ = make_teacher(tmp_path, [reply])
    model.generate([], task, visible)
    assert ledger(tmp_path)[-1]["rate_limit_headers"] == {"x-ratelimit-limit-requests": 500}
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "sk-never-log" not in path.read_text()


@pytest.mark.parametrize("headers,expected", [
    ({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "3.5s"}, 3.5),
    ({"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "250ms"}, .25),
    ({"x-ratelimit-remaining-tokens": "1", "x-ratelimit-reset-tokens": "1m2s"}, 62),
    ({"x-ratelimit-remaining-requests": "0", "Retry-After": "7"}, 7),
    ({"x-ratelimit-remaining-tokens": "0"}, 60),
    ({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "invalid"}, 60),
    ({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2s",
      "x-ratelimit-remaining-tokens": "1", "x-ratelimit-reset-tokens": "11s"}, 11),
    ({"x-ratelimit-remaining-requests": "1", "x-ratelimit-remaining-tokens": "180000"}, 0),
    ({"x-ratelimit-remaining-tokens": "invalid", "x-ratelimit-remaining-requests": "-1"}, 0),
])
def test_provider_remaining_pauses_shared_limiter_after_success(tmp_path, task, visible, headers, expected):
    raw = NS(headers=headers, parse=response)
    model, clock, client = make_teacher(tmp_path, [raw])
    client.responses = NS(with_raw_response=client)
    model.generate([], task, visible)
    assert not clock.waits  # The current result returns immediately.
    other = RateLimiter(tmp_path / "rate_limit", "gpt-6-luna", clock=clock, sleep=clock.sleep)
    reservation = other.acquire(1)
    assert reservation["limiter_wait_seconds"] == expected
    assert clock.waits == ([expected] if expected else [])


def test_cluster_directory_override_shares_across_run_paths(tmp_path, task, visible, monkeypatch):
    from teachers.openai_teacher import rate_limit_options
    clock = FakeClock()
    cluster = tmp_path / "cluster"
    monkeypatch.setenv("TEACHER_RATE_LIMIT_DIR", str(cluster))
    options = rate_limit_options({"rate_limit": {"requests_per_minute": 1}}, tmp_path / "run1")
    one = Teacher(cache_dir=tmp_path / "cache1", usage_ledger=tmp_path / "usage1.jsonl",
                  run_id="one", client=FakeClient(), clock=clock, sleep=clock.sleep, **options)
    two = Teacher(cache_dir=tmp_path / "cache2", usage_ledger=tmp_path / "usage2.jsonl",
                  run_id="two", client=FakeClient(), clock=clock, sleep=clock.sleep,
                  rate_limit_dir=tmp_path / "run2", requests_per_minute=1)
    assert one.limiter.path == two.limiter.path and one.limiter.path.parent == cluster
    one.generate([], task, visible)
    two.generate([], task, visible)
    assert clock.waits == [60]
    assert len(json.loads(one.limiter.path.read_text())["reservations"]) == 1
    assert not (tmp_path / "run1").exists() and not (tmp_path / "run2").exists()
    monkeypatch.delenv("TEACHER_RATE_LIMIT_DIR")
    default = RateLimiter(tmp_path / "default", "gpt-6-luna")
    assert default.path.parent == tmp_path / "default"
    monkeypatch.setenv("TEACHER_RATE_LIMIT_DIR", "")
    assert RateLimiter(tmp_path / "default", "gpt-6-luna").path == default.path


def limiter_worker(directory, now, barrier, queue, rpm, tpm):
    # Shared virtual time makes this a real spawned-process locking test without
    # waiting minutes. Each reservation timestamp is captured under the file lock.
    def clock():
        with now.get_lock():
            return now.value
    def sleep(seconds):
        with now.get_lock():
            now.value += seconds
    limiter = RateLimiter(directory, "model", requests_per_minute=rpm, tokens_per_minute=tpm,
                          clock=clock, sleep=sleep)
    barrier.wait(timeout=20)
    queue.put([limiter.acquire(40) for _ in range(6)])


@pytest.mark.parametrize("rpm,tpm", [(3, 100), (2, 1000)])
def test_multiprocess_shared_request_and_token_limits(tmp_path, rpm, tpm):
    ctx = multiprocessing.get_context("spawn")
    now, barrier, queue = ctx.Value("d", 1_700_000_000), ctx.Barrier(4), ctx.Queue()
    workers = [ctx.Process(target=limiter_worker, args=(str(tmp_path), now, barrier, queue, rpm, tpm))
               for _ in range(4)]
    try:
        for worker in workers:
            worker.start()
        rows = [r for _ in workers for r in queue.get(timeout=30)]
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
        assert len(rows) == 24 and any(r["limiter_wait_seconds"] > 0 for r in rows)
        for row in rows:
            window = [r for r in rows if row["reservation_timestamp"] - 60
                      < r["reservation_timestamp"] <= row["reservation_timestamp"]]
            assert len(window) <= rpm
            assert sum(r["reserved_tokens"] for r in window) <= tpm
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
        queue.close()
