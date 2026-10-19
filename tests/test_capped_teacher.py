import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest

from teachers.capped import CappedTeacher, BudgetExceeded, CollectorHalted
from teachers.openai_teacher import TeacherCallError
from teachers.capped import SharedReservations, CollectionBudgetExceeded
from teachers.openai_teacher import Teacher
from teachers.rate_limit import reserved_tokens


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
        client=client, dry_run=dry_run, sleep=lambda _: None, **kwargs)


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


@pytest.mark.parametrize("status", [201, 400, 429, 500])
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
