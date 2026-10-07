"""CPU-only policy overflow, accounting, replay and stage-exit regressions."""
import csv
import json
from pathlib import Path

import httpx
import pytest

from baselines.common.explore import checked_rollouts, explore_one, run_exploration
from conftest import FakeBackend
from eval.run_suite import run_suite
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.manifest import build_manifest, freeze_splits
from sed.react import build_messages
from sed.runner import replay_episode, run_episode
from sed.storage import BlobStore, write_json
from students.vllm_student import ContextOverflow, Student, context_options, context_rejection
from test_stage3a import listing_for
from test_suite import fixture_list


VLLM_400 = ("This model's maximum context length is 32768 tokens. However, you requested "
            "32824 tokens (31800 in the messages, 1024 in the completion). "
            "Please reduce the length of the messages or completion. SECRET_HTTP_BODY")
VLLM_NEW_400 = ("'max_tokens' or 'max_completion_tokens' is too large: 1024. "
                "This model's maximum context length is 32768 tokens and your request has "
                "31800 input tokens (1024 > 32768 - 31800). SECRET_HTTP_BODY")
VLLM_027_400 = ("This model's maximum context length is 32768 tokens. However, you requested "
               "1024 output tokens and your prompt contains 31800 input tokens, for a total of 32824 tokens.")


class OverflowServer:
    def __init__(self, precheck, *, message=VLLM_400):
        self.precheck, self.message = precheck, message
        self.chats, self.tokenizations = [], []

    def __call__(self, request):
        body = json.loads(request.content)
        assert request.method == "POST"
        if request.url.path == "/tokenize":
            self.tokenizations.append(body)
            # Exact boundary is allowed, the next prompt exceeds the limit.
            return httpx.Response(200, json={"count": 31744 if not self.chats else 31800,
                                            "max_model_len": 32768})
        assert request.url.path == "/v1/chat/completions"
        self.chats.append(body)
        if len(self.chats) > 1:
            assert not self.precheck, "Precheck must stop before an inference request"
            return httpx.Response(400, json={"error": {"message": self.message,
                                        "type": "BadRequestError", "code": 400}})
        return httpx.Response(200, json={"model": body["model"], "choices": [{
            "message": {"content": "Thought: " + "spam " * 1023}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 31744, "completion_tokens": 1024}})


def student(directory, server, *, precheck=False):
    return Student(endpoint="http://fake/v1", model="configured", revision="pinned", enable_thinking=False,
        log_dir=directory / "student_calls", transport=httpx.MockTransport(server),
        max_model_len=32768 if precheck else None)


@pytest.mark.parametrize("precheck,message", [(True, VLLM_400), (False, VLLM_400),
    (False, VLLM_NEW_400), (False, VLLM_027_400)])
def test_overflow_known_failure_stops_calls_and_preserves_replay(data_root, tmp_path, task, precheck, message):
    server = OverflowServer(precheck, message=message)
    directory = tmp_path / "episode"
    result = run_episode(Environment(data_root, backend_factory=FakeBackend),
        student(directory, server, precheck=precheck), task, load_config(), directory, policy_condition="student")
    assert result["success"] is False and result["complete"] and result["status"] == "ok"
    assert result["terminated"] and not result["truncated"] and "error_type" not in result
    assert result["termination_reason"] == "context_overflow"
    assert result["termination_step_index"] == 1  # Zero-based policy turn, including parser failures.
    assert result["termination_prompt_tokens"] == 31800
    assert len(server.chats) == (1 if precheck else 2)
    assert len(result["steps"]) == 1 and result["metrics"]["steps"] == 2
    assert result["metrics"]["environment_steps"] == result["metrics"]["environment_steps_unknown"] == 0
    assert result["metrics"]["failed_policy_calls"] == 0
    assert result["metrics"]["context_overflow_episodes"] == 1
    assert result["metrics"]["prompt_tokens"]["known_total"] == 31744  # No invented inference usage.
    assert result["metrics"]["completion_tokens"]["known_total"] == 1024
    replay = replay_episode(Environment(data_root, backend_factory=FakeBackend), result)
    assert replay["matched"] and replay["steps_replayed"] == 0 and replay["complete_episode"]
    rows = [json.loads(line) for line in (directory / "student_calls/calls.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["started", "completed", "started", "rejected"]
    meta = rows[-1]
    assert meta["step_index"] == 1 and meta["rendered_prompt_tokens"] == 31800
    assert meta["max_model_len"] == 32768 and meta["max_tokens"] == 1024
    assert meta["termination_reason"] == "context_overflow"
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    terminal = events[-1]
    assert terminal["status"] == "rejected" and terminal["error_type"] == "context_overflow"
    assert terminal["terminated"] and terminal["environment_steps"] == 0
    assert terminal["feedback_source"] == "analytic_verifier"
    verification = BlobStore(directory / "blobs").get(terminal["verification_ref"])
    assert verification["success"] is False and verification["environment_steps_confirmed"] == 0
    assert all("SECRET_HTTP_BODY" not in p.read_text() for p in directory.rglob("*") if p.is_file())
    if precheck:
        assert len(server.tokenizations) == 2
        assert server.tokenizations[0] == {k: server.chats[0][k] for k in
            ("model", "messages", "chat_template_kwargs")} | {
                "add_generation_prompt": True, "add_special_tokens": False}


def test_precheck_rejects_first_prompt_without_inference(tmp_path, visible):
    requests = []
    def handle(request):
        requests.append(request)
        assert request.url.path == "/tokenize"
        return httpx.Response(200, json={"count": 33000, "max_model_len": 32768})
    policy = student(tmp_path, handle, precheck=True)
    with pytest.raises(ContextOverflow) as error:
        policy.generate([], {}, visible)
    assert len(requests) == 1
    assert error.value.metadata["step_index"] == 0
    assert error.value.metadata["rendered_prompt_tokens"] == 33000
    policy.close()


@pytest.mark.parametrize("precheck", [False, True])
def test_ordinary_completion_request_unchanged(tmp_path, visible, precheck):
    requests = []
    def handle(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 31744, "max_model_len": 32768})
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "configured", "choices": [{
            "message": {"content": "Action: look"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 31744, "completion_tokens": 3}})
    policy = student(tmp_path, handle, precheck=precheck)
    result = policy.generate([], {}, visible)
    assert result.parsed_action == "look"
    assert requests == [{"model": "configured", "messages": build_messages([], {}, visible),
        "temperature": 0, "max_tokens": 1024, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}, "skip_special_tokens": False}]
    policy.close()


@pytest.mark.parametrize("status,body", [(400, "Invalid model SECRET_HTTP_BODY"),
    (400, "Invalid maximum context length configuration SECRET_HTTP_BODY"),
    (401, VLLM_400), (403, VLLM_400), (429, VLLM_400), (500, VLLM_400), (None, None)])
@pytest.mark.parametrize("precheck", [False, True])
def test_other_http_and_transport_errors_remain_unknown(data_root, tmp_path, task, status, body, precheck):
    def handle(request):
        if status is None:
            raise httpx.ConnectError("SECRET_HTTP_BODY", request=request)
        return httpx.Response(status, text=body)
    directory = tmp_path / "episode"
    with pytest.raises(RuntimeError, match="policy_error"):
        run_episode(Environment(data_root, backend_factory=FakeBackend), student(directory, handle, precheck=precheck),
            task, load_config(), directory, policy_condition="student")
    episode = json.loads((directory / "episode.json").read_text())
    assert episode["success"] == "unknown" and episode["status"] == "error" and not episode["complete"]
    assert episode["metrics"]["context_overflow_episodes"] == 0
    assert episode["metrics"]["failed_policy_calls"] == 1
    row = json.loads((directory / "student_calls/calls.jsonl").read_text().splitlines()[-1])
    assert row["status"] == "failed" and row["error_type"] == ("ConnectError" if status is None else "HTTPStatusError")
    assert all("SECRET_HTTP_BODY" not in p.read_text() for p in directory.rglob("*") if p.is_file())


def frozen_config(tmp_path, data_root, manifest, freeze):
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=40)
    config["server_provenance"] = {"config": {"server": {
        "command": ["vllm", "serve", "local-model", "--max-model-len", "32768"],
        "batch_invariant": True}}}
    for key, value in (("manifest", manifest), ("split_freeze", freeze)):
        path = tmp_path / (key + ".json")
        write_json(path, value)
        config["paths"][key] = str(path)
    return config


def test_exploration_overflow_is_eligible_known_negative(data_root, tmp_path):
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    config = frozen_config(tmp_path, data_root, manifest, freeze)
    tasks = tmp_path / "tasks.json"
    write_json(tasks, listing_for(manifest, freeze, freeze["seed_pool_candidates"][:1]))
    spec = dict(endpoint="http://fake/v1", model=config["models"]["student"]["requested_repo"],
        revision="pinned", enable_thinking=False, max_tokens=1024)
    def worker(job):
        return explore_one(job, env_factory=lambda c, t: Environment(data_root, backend_factory=FakeBackend),
            policy_factory=lambda **kw: Student(**kw, transport=httpx.MockTransport(OverflowServer(True))))
    output = tmp_path / "explore"
    summary = run_exploration(config=config, tasks_path=tasks, output=output, spec=spec,
                             decoding="greedy", worker_fn=worker)
    assert summary["episodes"] == summary["training_eligible_episodes"] == summary["context_overflow_episodes"] == 1
    assert summary["error_episodes"] == summary["unknown_success"] == summary["replay_errors"] == 0
    receipt, episode = checked_rollouts(output, manifest, freeze)[0]
    assert receipt["won"] is False and episode["success"] is False
    assert receipt["termination_reason"] == "context_overflow"
    assert not list(output.rglob("failure.json"))


def overflow_suite_worker(job):
    """Spawn-safe fake server and environment; retain actual suite worker logic."""
    import socket
    import eval.run_suite as suite
    def forbidden(*args, **kwargs):
        raise AssertionError("No test network")
    socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = forbidden
    socket.getaddrinfo = socket.create_connection = forbidden
    suite._environment = lambda c, t: Environment(c["environment"]["data_root"],
        split=t["split"], backend_factory=FakeBackend)
    suite.Student = lambda **kw: Student(**kw, transport=httpx.MockTransport(OverflowServer(True)))
    return suite._worker(job)


def test_eval_overflow_summary_tsv_and_successful_exit(data_root, tmp_path, monkeypatch):
    import eval.run_suite as suite
    tasks, manifest, freeze = fixture_list(data_root, tmp_path)
    config = frozen_config(tmp_path, data_root, manifest, freeze)
    spec = dict(endpoint="http://fake/v1", model=config["models"]["student"]["requested_repo"],
        revision="pinned", enable_thinking=False, max_tokens=1024)
    output = tmp_path / "eval"
    summary = run_suite(config=config, tasks_path=tasks, output=output, spec=spec,
                        workers=2, worker_fn=overflow_suite_worker)
    assert summary["episodes"] == summary["completed_episodes"] == summary["context_overflow_episodes"] == 4
    assert summary["error_episodes"] == summary["unknown_success"] == summary["successes"] == 0
    assert summary["totals"]["context_overflow_episodes"] == 4 and summary["batch_invariant"] is True
    with (output / "per_task.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert all(r["context_overflow_episodes"] == "1" and r["success"] == "False"
        and r["termination_reason"] == "context_overflow" and r["termination_step_index"] == "1"
        and r["termination_prompt_tokens"] == "31800" for r in rows)
    assert not list(output.rglob("failure.json"))
    monkeypatch.setenv("ALFWORLD_DATA_ROOT", str(data_root))
    monkeypatch.setattr(suite, "run_suite", lambda **kwargs: summary)
    assert suite.main(["--output", str(tmp_path / "cli")]) == 0


def test_context_limit_from_existing_server_provenance():
    assert context_options({"server_provenance": {"config": {"server": {
        "command": ["vllm", "--max-model-len", "16384"]}}}}) == {"max_model_len": 16384}
    assert context_options({}) == {}


def test_server_count_bounds_are_not_misreported_as_exact():
    response = httpx.Response(400, text=VLLM_027_400.replace("contains 31800", "contains at least 31800"))
    counts = context_rejection(response)
    assert counts["rendered_prompt_tokens"] == 31800 and counts["prompt_token_count_kind"] == "lower_bound"
    response = httpx.Response(400, text="This model's maximum context length is 32768 tokens. "
        "However, you requested 1024 output tokens and your prompt contains 1000000 characters "
        "(more than 126976 characters, which is the upper bound for 31744 input tokens).")
    counts = context_rejection(response)
    assert counts["rendered_prompt_tokens"] is None and counts["prompt_token_count_kind"] == "unknown"
