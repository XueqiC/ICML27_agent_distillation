import json
from pathlib import Path

import httpx
import pytest

from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.react import action_outcome, build_messages, parse_action, ActionParseError, PARSER_FEEDBACK
from sed.runner import replay_episode, run_episode
from sed.storage import BlobStore
from sed.thinking import split_thinking
from students.vllm_student import Student, separate_response, StudentCallError
from teachers.openai_teacher import TeacherResult


@pytest.mark.parametrize("raw,command", [
    ("Thought: inspect\nAction: look", "look"),
    ("Action: look\nThought: revise\nAction: go north\ntrailing text", "go north"),
    ("  Thought: inspect\r\n \tAction \t:   look  \r\n", "look"),
    ("Action: go  north", "go  north"),
])
def test_parser_last_line_and_whitespace(raw, command):
    assert parse_action(raw) == command


@pytest.mark.parametrize("raw", ["look", "Thought: no command", "Action: \nlook", "ACTION: look",
                                     "Action: look\nAction:  ", None, "prefix Action: look"])
def test_missing_action(raw):
    with pytest.raises(ActionParseError):
        parse_action(raw)


def test_no_fuzzy_matching():
    assert action_outcome("Action: LOOK", ["look"]) == ("LOOK", "inadmissible")
    assert action_outcome("Action: look", ["look"]) == ("look", "legal")
    assert action_outcome("look", ["look"]) == (None, "parser_error")


@pytest.mark.parametrize("message,shape", [
    ({"content": "Thought: visible\nAction: look", "reasoning": "PRIVATE\nAction: go north"}, "reasoning"),
    ({"content": "Thought: visible\nAction: look", "reasoning_content": "PRIVATE\nAction: go north"}, "reasoning_content"),
    ({"content": "<|channel>thought\nPRIVATE\nAction: go north<channel|>Thought: visible\nAction: look<turn|>"}, "raw_channels"),
    ({"content": "PRIVATE\nAction: go north<channel|>Thought: visible\nAction: look"}, "raw_channels"),
])
def test_thinking_separation(message, shape):
    reply, thinking, actual = separate_response(message)
    assert reply == "Thought: visible\nAction: look" and "PRIVATE" in thinking
    assert actual == shape and parse_action(reply) == "look"


@pytest.mark.parametrize("text", ["<|channel>thought\nPRIVATE\nAction: look", "thought\nAction: look"])
def test_truncated_thinking_has_no_react_reply(text):
    reply, thinking = split_thinking(text)
    assert reply == "" and "Action: look" in thinking


def test_history_only_visible_reply(visible, task):
    history = [{"visible": dict(visible, audit="SECRET_AUDIT", extra={"expert_plan": "SECRET_PLAN"}),
                "reply": "<|channel>thought\nSECRET_THINKING<channel|>Thought: visible\nAction: look",
                "thinking": "SECRET_THINKING", "reasoning_content": "SECRET_THINKING",
                "policy_metadata": {"thinking_ref": "SECRET_REF"}, "won": True}]
    messages = build_messages(history, dict(task, audit="SECRET_TASK"), dict(visible, facts="SECRET_FACTS"))
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "Thought: visible\nAction: look"
    assert "SECRET" not in json.dumps(messages) and task["task_id"] not in json.dumps(messages)


class Replies:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []
    def generate(self, history, task, visible):
        self.inputs.append((history, task, visible))
        raw = next(self.replies)
        return TeacherResult(raw, action_outcome(raw, visible["admissible_commands"])[0], {})


def test_parser_feedback_consumes_turn_without_step_and_inadmissible_executes(data_root, tmp_path, task):
    class TrackingBackend(FakeBackend):
        actions = []
        def step(self, action):
            self.actions.append(action)
            return super().step(action)
    env = Environment(data_root, max_episode_steps=3, backend_factory=TrackingBackend)
    policy = Replies(["missing", "Thought: try\nAction: invented Command", "Action: look"])
    directory = tmp_path / "run"
    result = run_episode(env, policy, task, load_config(), directory, policy_condition="test")
    assert TrackingBackend.actions == ["invented Command", "look"]
    assert policy.inputs[1][2]["observation"] == PARSER_FEEDBACK
    assert policy.inputs[1][0][0]["reply"] == "missing"
    assert result["truncated"] and result["metrics"]["environment_steps"] == 2
    assert [s["action_outcome"] for s in result["steps"]] == ["parser_error", "inadmissible", "legal"]
    assert result["metrics"]["parser_errors"] == result["metrics"]["inadmissible_actions"] == 1
    events = [json.loads(s) for s in (directory / "events.jsonl").read_text().splitlines()]
    assert events[1]["environment_steps"] == 0 and events[1]["error_type"] == "parser_error"
    assert events[1]["feedback_source"] == "analytic_verifier" and events[2]["inadmissible"] is True
    assert env.backend is None
    replay = replay_episode(Environment(data_root, max_episode_steps=3, backend_factory=FakeBackend), result)
    assert replay["matched"] and replay["steps_replayed"] == 2


def test_forty_parser_errors_terminate_without_environment_steps(data_root, tmp_path, task):
    class NoStep(FakeBackend):
        def step(self, action):
            pytest.fail("Parser errors must not call step")
    result = run_episode(Environment(data_root, backend_factory=NoStep), Replies(["bad"] * 40),
                         task, load_config(), tmp_path / "run", policy_condition="test")
    assert result["metrics"]["steps"] == result["metrics"]["parser_errors"] == 40
    assert result["metrics"]["environment_steps"] == 0 and result["truncated"]


class FakeChatServer:
    """OpenAI-compatible HTTP handler, entirely in memory (no sockets)."""
    def __init__(self, shape="reasoning", finish="stop"):
        self.requests = []
        self.shape, self.finish = shape, finish
    def __call__(self, request):
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        self.requests.append(body)
        reply = "Thought: inspect the room\nAction: look"
        message = {"role": "assistant", "content": reply}
        if self.shape == "raw":
            message["content"] = "<|channel>thought\nPRIVATE\nAction: wrong<channel|>" + reply
        else:
            message[self.shape] = "PRIVATE\nAction: wrong"
        return httpx.Response(200, json={"model": body["model"], "id": "chat_fake", "created": 123,
            "choices": [{"message": message, "finish_reason": self.finish}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 17}})


@pytest.mark.parametrize("shape", ["reasoning", "reasoning_content", "raw"])
@pytest.mark.parametrize("enabled", [False, True])
def test_student_http_end_to_end(data_root, tmp_path, task, shape, enabled):
    if enabled:
        (data_root / task["task_id"] / "game.tw-pddl").write_text('{"win_at": 2}')
    server = FakeChatServer(shape)
    config = load_config()
    directory = tmp_path / "run"
    policy = Student(endpoint="http://fake-vllm/v1", model=config["models"]["student"]["requested_repo"],
                     revision=config["models"]["student"]["revision"], enable_thinking=enabled,
                     max_tokens=123, log_dir=directory / "student_calls", transport=httpx.MockTransport(server))
    result = run_episode(Environment(data_root, backend_factory=FakeBackend, max_episode_steps=2), policy,
                         task, config, directory, policy_condition="base_student")
    assert result["complete"] and result["metrics"]["prompt_tokens"]["known_total"] == 26
    assert result["success"] is enabled
    assert result["metrics"]["thinking_tokens"]["unknown_calls"] == 2
    for body in server.requests:
        assert body["temperature"] == 0 and body["max_tokens"] == 123
        assert body["chat_template_kwargs"] == {"enable_thinking": enabled}
        assert body["skip_special_tokens"] is False
        assert "PRIVATE" not in json.dumps(body) and "SECRET" not in json.dumps(body)
        assert task["task_id"] not in json.dumps(body)
    assert server.requests[1]["messages"][2] == {"role": "assistant", "content": "Thought: inspect the room\nAction: look"}
    events = [json.loads(s) for s in (directory / "events.jsonl").read_text().splitlines()]
    assert all(e["teacher_tokens_in"] == 0 for e in events)
    meta = BlobStore(directory / "blobs").get(events[1]["policy_metadata_ref"])
    assert meta["thinking_chars"] > 0 and meta["finish_reason"] == "stop"
    assert "PRIVATE" in BlobStore(directory / "student_calls/blobs").get(meta["thinking_ref"])
    assert meta["latency_seconds"] >= 0 and meta["student_revision"] == config["models"]["student"]["revision"]


def test_student_http_error_sanitized(tmp_path, visible, task):
    def handler(request):
        return httpx.Response(500, text="SECRET_RAW_ERROR")
    policy = Student(endpoint="http://fake/v1", model="configured", revision="pinned", enable_thinking=True,
                     log_dir=tmp_path / "calls", transport=httpx.MockTransport(handler))
    with pytest.raises(StudentCallError) as exc:
        policy.generate([], task, visible)
    assert "SECRET" not in str(exc.value)
    assert all("SECRET_RAW_ERROR" not in p.read_text() for p in tmp_path.rglob("*") if p.is_file())
    ledger = [json.loads(s) for s in (tmp_path / "calls/calls.jsonl").read_text().splitlines()]
    assert [r["status"] for r in ledger] == ["started", "failed"] and ledger[-1]["usage"] is None
    policy.close()


def test_failed_environment_step_retains_student_usage(data_root, tmp_path, task):
    class FailedStep(FakeBackend):
        def step(self, action):
            raise RuntimeError("unknown environment side effect")
    config = load_config()
    student = config["models"]["student"]
    directory = tmp_path / "run"
    policy = Student(endpoint="http://fake/v1", model=student["requested_repo"], revision=student["revision"],
                     enable_thinking=True, log_dir=directory / "student_calls",
                     transport=httpx.MockTransport(FakeChatServer()))
    with pytest.raises(RuntimeError, match="step_error"):
        run_episode(Environment(data_root, backend_factory=FailedStep), policy, task, config,
                    directory, policy_condition="base_student")
    result = json.loads((directory / "episode.json").read_text())
    assert result["metrics"]["completion_tokens"]["known_total"] == 17
    assert result["metrics"]["legal_actions"] == result["metrics"]["steps"] == 1
    assert result["metrics"]["environment_steps_unknown"] == 1
    assert result["success"] == "unknown"


def test_student_thinking_only_finish_and_reported_token_count(tmp_path, visible, task):
    def handler(request):
        return httpx.Response(200, json={"model": "configured", "choices": [{
            "message": {"content": None, "reasoning": "PRIVATE\nAction: look"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                      "completion_tokens_details": {"reasoning_tokens": 7}}})
    policy = Student(endpoint="http://fake/v1", model="configured", revision="pinned", enable_thinking=True,
                     log_dir=tmp_path / "calls", transport=httpx.MockTransport(handler))
    result = policy.generate([], task, visible)
    assert result.raw_text == "" and result.parsed_action is None
    assert result.metadata["finish_reason"] == "length" and result.metadata["thinking_tokens"] == 7
    policy.close()
