"""Output-cap exhaustion is a model output; all transports/environments are fake."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.runner import replay_episode, run_episode
from sed.storage import digest, write_json
from teachers.capped import BudgetExceeded, CollectorHalted, SharedReservations
from teachers.openai_teacher import TeacherCallError
from teachers.stage import execution_attempt, ledger_counts, teacher_stage
from test_capped_teacher import RawClient, SequenceClient, capped, http_429
from test_full_train import full_bank
from test_stage2b_seeds import seed_bank
from test_stage3d import STATE, budget, correction
from test_teacher import FakeClient, response, teacher


def read(path):
    return json.loads(Path(path).read_text())


def truncated_response(text="", *, tokens=1024, details="max_output_tokens"):
    value = response(text)
    value.status = "incomplete"
    value.usage.output_tokens = tokens
    value.usage.total_tokens = value.usage.input_tokens + tokens
    value.usage.output_tokens_details.reasoning_tokens = tokens
    if details != "absent":
        value.incomplete_details = details
        if isinstance(details, str):
            value.incomplete_details = NS(reason=details)
    return value


def http_response(value):
    return NS(status_code=200, headers={}, parse=lambda: value)


@pytest.mark.parametrize("text", ["", "Thought: unfinished", "Action: look"])
@pytest.mark.parametrize("details,tokens", [
    ("max_output_tokens", 1024),
    ({"reason": "max_output_tokens"}, 1023),  # Explicit reason is authoritative.
    ("absent", 1024),
    (None, 1024),
    ("absent", 1025),
])
def test_cap_output_retains_text_usage_and_flag_without_cache(tmp_path, task, visible, text, details, tokens):
    client = FakeClient([truncated_response(text, details=details, tokens=tokens), response()])
    model = teacher(tmp_path / "teacher", client=client, sampling={"max_output_tokens": 1024})

    @teacher_stage("output")
    def collect(output):
        return model.generate([], task, visible)

    result = collect(tmp_path / "stage")
    assert result.raw_text == text and result.truncated
    assert result.metadata["truncated"] is True
    assert result.metadata["response_status"] == "incomplete"
    assert result.metadata["max_output_tokens"] == 1024
    assert result.metadata["usage"] == result.metadata["billed_usage"]
    assert result.metadata["usage"]["output_tokens"] == tokens
    assert result.metadata["usage"]["reasoning_tokens"] == tokens
    assert not list(model.cache_dir.glob("*.json"))
    rows = [json.loads(line) for line in model.ledger.path.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["started", "rejected"]
    assert rows[-1]["error_type"] == "TruncatedOutput"
    assert rows[-1]["usage"] == result.metadata["usage"]
    saved = model.call_blobs.get(rows[-1]["result_ref"])
    assert saved["raw_text"] == text and saved["metadata"]["truncated"] is True
    report = read(tmp_path / "stage/teacher_report.json")
    assert report["status"] == "complete" and report["truncated_outputs"] == 1
    assert report["transport_failures"] == report["transport_failed_attempts"] == report["unknown_outcomes"] == 0
    assert report["parser_rejections"] == int(text != "Action: look")
    assert ledger_counts(model.ledger.path) == {
        "transport_failures": 0, "rejections": int(text != "Action: look"),
        "truncated_outputs": 1, "unknown_ledger_rows": 0}
    # Same prompt needs a new call; only the subsequent completed output is cached.
    assert not model.generate([], task, visible).truncated
    assert model.generate([], task, visible).metadata["cache_hit"]
    assert len(client.requests) == 2


@pytest.mark.parametrize("status,details,tokens,usage", [
    ("incomplete", "content_filter", 1024, "known"),
    ("incomplete", "other_reason", 1024, "known"),
    ("incomplete", {}, 1024, "known"),
    ("incomplete", {"reason": None}, 1024, "known"),
    ("incomplete", "absent", 1023, "known"),
    ("incomplete", "max_output_tokens", 1024, "missing"),
    ("incomplete", "max_output_tokens", 1024, "empty"),
    ("incomplete", "max_output_tokens", 1024, "missing_output"),
    ("completed", None, 1024, "missing"),
    ("completed", None, 1024, "missing_input"),
    ("failed", "max_output_tokens", 1024, "known"),
    ("cancelled", "max_output_tokens", 1024, "known"),
    ("in_progress", "max_output_tokens", 1024, "known"),
    ("unknown", "max_output_tokens", 1024, "known"),
    (None, "max_output_tokens", 1024, "known"),
])
def test_other_outcomes_fail_loud_and_halt_collector(tmp_path, task, visible, status, details, tokens, usage):
    value = truncated_response("Action: look", details=details, tokens=tokens)
    value.status = status
    if usage == "missing":
        value.usage = None
    elif usage == "empty":
        value.usage = {}
    elif usage == "missing_output":
        del value.usage.output_tokens
    elif usage == "missing_input":
        del value.usage.input_tokens
    client = SequenceClient([http_response(value), RawClient().create()])
    options = {"sampling": {"max_output_tokens": 1024}}
    model = capped(tmp_path / "teacher", client, **options)

    @teacher_stage("output")
    def collect(output):
        return model.generate([], task, visible)

    with pytest.raises(TeacherCallError, match="NOT complete"):
        collect(tmp_path / "stage")
    with pytest.raises(CollectorHalted):
        capped(tmp_path / "teacher", client, **options).generate([], task, visible)
    assert len(client.requests) == 1
    assert not list(model.teacher.cache_dir.glob("*.json"))
    report = read(tmp_path / "stage/teacher_report.json")
    assert report["status"] == "failed"
    assert report["transport_failures"] == report["transport_failed_attempts"] == report["unknown_outcomes"] == 1
    assert report["truncated_outputs"] == report["rejections"] == 0
    assert ledger_counts(model.teacher.ledger.path)["transport_failures"] == 1


@pytest.mark.parametrize("after_429", [False, True])
def test_capped_truncation_consumes_slot_without_halting_on_restart(tmp_path, task, visible, after_429):
    client = SequenceClient(([http_429()] if after_429 else []) +
                            [http_response(truncated_response()), RawClient().create()])
    reservations = SharedReservations(tmp_path / "reservations.sqlite3", max_calls=2)
    options = {"sampling": {"max_output_tokens": 1024}, "reservation_budget": reservations}
    model = capped(tmp_path, client, **options)

    @teacher_stage("output")
    def collect(output):
        return model.generate([], task, visible)

    assert collect(tmp_path / "stage").truncated
    state = read(model.counter_path)
    assert state["attempts"] == 1 and not state["pending"] and not state["halted"]
    report = read(tmp_path / "stage/teacher_report.json")
    assert report["truncated_outputs"] == 1 and report["transport_failures"] == 0
    assert report["transport_failed_attempts"] == int(after_429)
    snapshot = reservations.snapshot()
    assert snapshot["logical_calls"] == 1
    assert snapshot["transport_reservations"] == 1 + int(after_429)
    # Rebuild from the durable ledger, including a 429 recovered by truncation.
    model.counter_path.unlink()
    restarted = capped(tmp_path, client, **options)
    restarted.check_health()
    assert restarted._read_counter()["attempts"] == 1
    assert not restarted.generate([], task, visible).metadata["cache_hit"]
    assert len(client.requests) == 2 + int(after_429)
    assert reservations.snapshot()["logical_calls"] == 2
    assert reservations.snapshot()["reserved_input_tokens"] > snapshot["reserved_input_tokens"]
    assert restarted.generate([], task, visible).metadata["cache_hit"]
    with pytest.raises(BudgetExceeded):
        restarted.generate([], task, dict(visible, observation="next request"))
    assert not restarted._read_counter()["halted"]


@pytest.mark.parametrize("capped_cache", [False, True])
def test_truncated_result_cannot_be_reused_from_success_cache(tmp_path, task, visible, capped_cache):
    client = FakeClient([response()])
    model = teacher(tmp_path, client=client)
    model.generate([], task, visible)
    path = next(model.cache_dir.glob("*.json"))
    cached = read(path)
    cached["result"]["metadata"].update(response_status="incomplete", truncated=True)
    cached["result_sha256"] = digest(cached["result"])
    write_json(path, cached)
    consumer = capped(tmp_path, RawClient()) if capped_cache else model
    with pytest.raises(TeacherCallError, match="incomplete/unknown"):
        consumer.generate([], task, visible)
    assert len(client.requests) == 1


def test_completed_output_at_cap_is_not_truncated(tmp_path, task, visible):
    value = truncated_response("Action: look")
    value.status = "completed"
    model = teacher(tmp_path, client=FakeClient([value]), sampling={"max_output_tokens": 1024})
    assert not model.generate([], task, visible).truncated
    assert model.generate([], task, visible).metadata["cache_hit"]
    assert ledger_counts(model.ledger.path)["truncated_outputs"] == 0


def test_h0_runner_records_parser_failure_then_continues(data_root, tmp_path, task):
    client = FakeClient([truncated_response(), response()])
    model = teacher(tmp_path / "teacher", client=client, sampling={"max_output_tokens": 1024})

    def backend(path, seed):
        result = FakeBackend(path, seed)
        result.win_at = 1
        return result

    def environment():
        return Environment(data_root, max_episode_steps=2, backend_factory=backend)

    @teacher_stage("output")
    def collect(output):
        return run_episode(environment(), model, task, load_config(), output / "episode",
                           policy_condition="teacher")

    result = collect(tmp_path / "stage")
    assert result["complete"] and result["success"] and len(client.requests) == 2
    assert result["action_outcomes"] == ["parser_error", "legal"]
    assert [step["environment_steps"] for step in result["steps"]] == [0, 1]
    assert result["steps"][0]["action_raw"] == ""
    assert result["steps"][0]["policy_metadata"]["truncated"] is True
    assert result["metrics"]["completion_tokens"]["known_total"] == 1041
    assert result["metrics"]["thinking_tokens"]["known_total"] == 1034
    report = read(tmp_path / "stage/teacher_report.json")
    assert report["truncated_outputs"] == report["parser_rejections"] == report["rejections"] == 1
    assert report["transport_failures"] == 0
    assert replay_episode(environment(), result)["matched"]


@pytest.mark.parametrize("kind", ["extract", "privileged", "zero"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_leap_truncation_is_validator_rejection_and_consumes_try(tmp_path, task, kind, exhausted):
    first = truncated_response(tokens=4096)
    replies = [first] * 3 if exhausted else [first, response(STATE if kind == "extract" else correction())]
    client = FakeClient(replies)
    limited, task = budget(tmp_path / "teacher", task, client, kind=kind)
    messages = [{"role": "user", "content": "Stub request"}]
    kwargs = {} if kind == "extract" else {"length": 2}

    @teacher_stage("output")
    def collect(output):
        return limited.request(task, messages, **kwargs)

    result = collect(tmp_path / "stage")
    assert result["status"] == ("exhausted" if exhausted else "accepted")
    assert result["rejections"] == (3 if exhausted else 1)
    assert result["attempts_reserved"] == len(client.requests) == (3 if exhausted else 2)
    assert result["transport_failures"] == 0
    attempt = read(Path(result["directory"]) / "attempt-0/result.json")
    assert attempt["accepted"] is False and attempt["result"]["raw_text"] == ""
    assert attempt["result"]["metadata"]["truncated"] is True
    assert not list(limited.root.rglob("failure.json"))
    report = read(tmp_path / "stage/teacher_report.json")
    assert report["transport_failures"] == report["transport_failed_attempts"] == 0
    assert report["truncated_outputs"] == report["rejections"] == report["parser_rejections"] == (3 if exhausted else 1)
    assert limited.request(task, messages, **kwargs) == result
    assert len(client.requests) == (3 if exhausted else 2)
    if exhausted:
        assert not list(limited.sampling_root.glob("cache/*.json"))


def test_execution_resume_recognizes_truncation_as_finished(tmp_path, task, visible):
    directory = tmp_path / "attempt"
    write_json(directory / "started.json", {"status": "started"})
    model = teacher(directory / "teacher", client=FakeClient([truncated_response()]),
                    sampling={"max_output_tokens": 1024})
    model.generate([], task, visible)
    write_json(directory / "episode/episode.json", {"complete": True, "status": "ok"})
    assert execution_attempt(directory, start="started.json") == directory
    assert not (directory / "transport_failure.json").exists()


@pytest.mark.parametrize("full", [False, True])
def test_seed_and_full_train_collectors_keep_truncated_step_and_resume(seed_bank, full_bank, tmp_path, full):
    from sed.seed_selection import select_seeds
    from teachers.collect_seeds import collect
    from teachers.openai_teacher import Teacher

    root, manifest, freeze, full_selection, config = full_bank
    if not full:
        root, manifest, freeze, config = seed_bank
    config["environment"]["max_episode_steps"] = 2
    selection = full_selection if full else select_seeds(manifest, freeze, k=1)
    clients = []

    def environment(task):
        def backend(path, seed):
            value = FakeBackend(path, seed)
            value.win_at = 1
            return value
        return Environment(root, max_episode_steps=2, backend_factory=backend)

    def policy(task, cache, ledger):
        client = FakeClient([truncated_response(), response()])
        clients.append(client)
        return Teacher(cache_dir=cache, usage_ledger=ledger, run_id="stub-seed",
                       client=client, sampling={"max_output_tokens": 1024})

    output = tmp_path / "collection"
    kwargs = {"env_factory": environment, "teacher_factory": policy}
    result = collect(selection, manifest, freeze, config, output, **kwargs)
    assert result["status"] == "complete" and result["trusted_trajectories"] == len(clients) > 0
    assert not list(output.rglob("transport_failure.json"))
    for path in (output / "attempts").glob("*/episode/episode.json"):
        episode = read(path)
        assert episode["action_outcomes"] == ["parser_error", "legal"]
        assert episode["success"] is True
    report = read(output / "teacher_report.json")
    assert report["truncated_outputs"] == report["parser_rejections"] == report["rejections"] == len(clients)
    assert report["transport_failures"] == 0
    calls = sum(len(client.requests) for client in clients)
    resumed = collect(selection, manifest, freeze, config, output, resume=True, **kwargs)
    assert resumed["trusted_trajectories"] == result["trusted_trajectories"]
    assert sum(len(client.requests) for client in clients) == calls


def truncated_evaluation_worker(job):
    from test_teacher_failloud import failed_evaluation_worker
    return failed_evaluation_worker(job, truncated_response())


def test_evaluation_aggregates_truncations_from_worker_ledgers(data_root, tmp_path):
    from eval.run_suite import run_suite
    from test_suite import fixture_list

    tasks, manifest, freeze = fixture_list(data_root, tmp_path)
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=1)
    config["models"]["teacher"]["sampling"]["max_output_tokens"] = 1024
    for name, value in (("manifest", manifest), ("split_freeze", freeze)):
        config["paths"][name] = str(tmp_path / (name + ".json"))
        write_json(config["paths"][name], value)
    output = tmp_path / "evaluation"
    result = run_suite(config=config, tasks_path=tasks, policy="teacher", workers=1,
                       output=output, worker_fn=truncated_evaluation_worker)
    assert result["completed_episodes"] == 4
    report = read(output / "teacher_report.json")
    assert report["truncated_outputs"] == report["parser_rejections"] == report["rejections"] == 4
    assert report["transport_failures"] == report["transport_failed_attempts"] == 0
    assert read(output / "summary.json")["truncated_outputs"] == 4


def test_worker_failure_report_keeps_earlier_truncation(data_root, tmp_path, monkeypatch):
    import eval.run_suite as suite
    from sed.manifest import resolve_task
    from teachers.openai_teacher import Teacher
    from test_suite import fixture_list

    listing, manifest, freeze = fixture_list(data_root, tmp_path)
    task = resolve_task(manifest, freeze, read(listing)["task_ids"][0])
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=2)
    config["models"]["teacher"].update(max_attempts=1, sampling={"max_output_tokens": 1024})
    config["suite"] = {"teacher_rate_limit_dir": str(tmp_path / "limiter")}
    client = FakeClient([truncated_response(), TimeoutError("PRIVATE_ERROR")])
    monkeypatch.setattr(suite, "Teacher", lambda **kw: Teacher(client=client, **kw))
    monkeypatch.setattr(suite, "_environment", lambda config, task:
                        Environment(data_root, split="dev", max_episode_steps=2, backend_factory=FakeBackend))
    output = tmp_path / "worker"
    with pytest.raises(TeacherCallError):
        suite._worker((task, config, str(output), "teacher", {}, []))
    report = read(output / "teacher_report.json")
    assert report["status"] == "failed"
    assert report["transport_failures"] == report["truncated_outputs"] == report["rejections"] == 1
    assert len(client.requests) == 2
