import json
from types import SimpleNamespace as NS

import pytest

from teachers.openai_teacher import Teacher, TeacherCallError, read_api_key
from sed.react import build_messages


def response(text="Action: look"):
    return NS(output_text=text, model="gpt-6-luna", id="resp_fake", _request_id="req_fake",
              status="completed", usage=NS(input_tokens=13, output_tokens=17, total_tokens=30,
              input_tokens_details=NS(cached_tokens=2), output_tokens_details=NS(reasoning_tokens=10)))


class FakeClient:
    def __init__(self, results=None):
        self.results = list(results or [response()])
        self.requests = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def teacher(tmp_path, **kwargs):
    return Teacher(cache_dir=tmp_path / "cache", usage_ledger=tmp_path / "usage.jsonl",
                   run_id="test-run", client=kwargs.pop("client", FakeClient()), sleep=lambda _: None,
                   **kwargs)


def test_disk_cache_metadata_and_no_privileged_requests(tmp_path, task, visible):
    client = FakeClient()
    first = teacher(tmp_path, client=client).generate([], task, visible)
    fresh = teacher(tmp_path, client=client)
    cached = fresh.generate([], task, visible)
    assert first.raw_text == "Action: look" and first.parsed_action == "look"
    assert first.metadata["usage"]["reasoning_tokens"] == 10
    assert first.metadata["returned_model_id"] == "gpt-6-luna"
    assert cached.metadata["cache_hit"] and cached.metadata["billed_usage"]["input_tokens"] == 0
    assert len(client.requests) == 1
    assert set(client.requests[0]) == {"model", "input", "store", "max_output_tokens"}
    statuses = [json.loads(r)["status"] for r in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert statuses == ["started", "success", "cache_hit"]


def test_cache_keys_cover_context_prompt_history_task_and_sampling(tmp_path, task, visible):
    model = teacher(tmp_path)
    key = model.cache_key([], task, visible)
    keys = [model.cache_key([], dict(task, task_id="train/another"), visible),
            model.cache_key([], dict(task, instruction="different"), visible),
            model.cache_key([{"visible": visible, "action": "look"}], task, visible),
            model.cache_key([], task, dict(visible, observation="different intervention")),
            model.cache_key([], task, dict(visible, admissible_commands=["look"])),
            teacher(tmp_path, sampling={"max_output_tokens": 256}).cache_key([], task, visible),
            teacher(tmp_path, prompt_template="changed").cache_key([], task, visible)]
    assert key not in keys and len(set(keys)) == len(keys)
    # Audit cannot alter prompts or create covert intervention channels.
    assert key == model.cache_key([], dict(task, hidden="SECRET"), dict(visible, audit="SECRET"))


def test_teacher_uses_shared_react_conversation_and_visible_thought_in_cache(tmp_path, task, visible):
    model = teacher(tmp_path)
    history = [{"visible": visible, "reply": "Thought: inspect\nAction: look", "thinking": "SECRET"}]
    assert model.prompt(history, task, visible) == build_messages(history, task, visible)
    assert model.cache_key(history, task, visible) != model.cache_key(
        [{"visible": visible, "reply": "Thought: changed\nAction: look"}], task, visible)


def test_prompt_allowlist_hides_audit_and_task_filename(tmp_path, task, visible):
    model = teacher(tmp_path)
    polluted = dict(visible, policy_commands=["SECRET"], extra={"expert_plan": "SECRET"}, facts="SECRET")
    prompt = json.dumps(model.prompt([{"visible": polluted, "action": "look", "audit": "SECRET"}],
                          dict(task, mechanism_id="SECRET", audit="SECRET"), polluted))
    assert "SECRET" not in prompt and task["task_id"] not in prompt
    assert "policy_commands" not in prompt and "admissible_commands" in prompt


def test_bounded_retries_account_for_every_attempt_without_secrets(tmp_path, task, visible, monkeypatch, capsys):
    secret = "sk-test-secret-never-log-123456"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    client = FakeClient([TimeoutError(secret), TimeoutError(secret), TimeoutError(secret)])
    model = teacher(tmp_path, client=client, max_attempts=2)
    with pytest.raises(TeacherCallError) as error:
        model.generate([], task, visible)
    assert len(client.requests) == 2
    assert secret not in str(error.value) and secret not in capsys.readouterr().out
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text()
    rows = [json.loads(r) for r in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["started", "failed", "started", "failed"]
    assert all(r["usage"] is None for r in rows)
    assert all(r["prompt_ref"].startswith("sha256:") for r in rows)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("status,param,error_type,code,unsupported", [
    (400, "temperature", "invalid_request_error", None, True),
    (400, "top_p", "invalid_request_error", None, True),
    (400, "temperature", None, "unsupported_parameter", True),
    (400, "top_p", None, "unsupported_value", True),
    (400, "input", "invalid_request_error", None, False),
    (400, "temperature", "other_error", None, False),
    (400, "top_p", "other_error", None, False),
    (400, "temperature", None, None, False),
    (400, "temperature", "invalid_request_error", "other_code", False),
    (422, "temperature", "invalid_request_error", None, False),
])
def test_sampling_fallback_flag_is_narrow_sanitized_and_not_retryable(
        tmp_path, task, visible, capsys, nested, status, param, error_type, code, unsupported):
    class FakeAPIError(Exception):
        status_code = status
    failure = FakeAPIError("PRIVATE_PROVIDER_MESSAGE")
    error_body = {"type": error_type, "param": param, "code": code,
                  "message": "PRIVATE_PROVIDER_MESSAGE", "extra": "PRIVATE_PROVIDER_BODY"}
    failure.body = {"error": error_body} if nested else error_body
    client = FakeClient([failure, response()])
    with pytest.raises(TeacherCallError) as caught:
        teacher(tmp_path, client=client, max_attempts=3).generate([], task, visible)
    assert len(client.requests) == 1
    rows = [json.loads(r) for r in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["started", "failed"]
    failed = rows[-1]
    if unsupported:
        assert failed["unsupported_sampling_parameter"] == param
    else:
        assert "unsupported_sampling_parameter" not in failed
    assert not failed["retryable"] and not failed["will_retry"]
    assert failed["wait_seconds"] == failed["provider_wait_seconds"] == 0
    assert failed["usage"] is None
    assert "body" not in failed and "message" not in failed
    assert "PRIVATE_PROVIDER" not in str(caught.value)
    captured = capsys.readouterr()
    assert "PRIVATE_PROVIDER" not in captured.out + captured.err
    assert all("PRIVATE_PROVIDER" not in p.read_text() for p in tmp_path.rglob("*") if p.is_file())


def test_echoed_secret_is_redacted(tmp_path, task, visible, monkeypatch):
    secret = "sk-test-echo-123456"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    result = teacher(tmp_path, client=FakeClient([response(secret)])).generate([], task, visible)
    assert secret not in result.raw_text
    assert all(secret not in p.read_text() for p in tmp_path.rglob("*") if p.is_file())


def test_invalid_output_is_retained_but_not_cached(tmp_path, task, visible):
    model = teacher(tmp_path, client=FakeClient([response("first\nsecond")]))
    result = model.generate([], task, visible)
    assert result.parsed_action is None and result.metadata["parser_error"] == "ActionParseError"
    assert not list((tmp_path / "cache").glob("*.json"))
    assert "rejected" in (tmp_path / "usage.jsonl").read_text()
    assert any("first" in p.read_text() for p in (tmp_path / "teacher_blobs").rglob("*.json"))


def test_credentials_env_precedes_file_without_shell_execution(tmp_path, monkeypatch):
    secret_file = tmp_path / "keys.env"
    secret_file.write_text("export OPENAI_API_KEY='fake-file-key' # comment\n")
    assert read_api_key(secret_file) == "fake-file-key"
    monkeypatch.setenv("OPENAI_API_KEY", "fake-env-key")
    assert read_api_key(secret_file) == "fake-env-key"
    monkeypatch.delenv("OPENAI_API_KEY")
    secret_file.write_text("OPENAI_API_KEY='$(touch /tmp/DO_NOT_EXECUTE)'\n")
    assert read_api_key(secret_file) == "$(touch /tmp/DO_NOT_EXECUTE)"


def test_sealed_rejected_before_cache_or_client(tmp_path, task, visible):
    client = FakeClient()
    model = teacher(tmp_path, client=client, purpose="evaluation")
    with pytest.raises(ValueError, match="sealed"):
        model.generate([], dict(task, split="test", sealed=True), visible)
    assert not client.requests and not (tmp_path / "usage.jsonl").exists()


@pytest.mark.parametrize("sampling", [{"logprobs": True}, {"top_logprobs": 5},
                                     {"include": ["reasoning.encrypted_content"]}])
def test_privileged_sampling_rejected(tmp_path, sampling):
    with pytest.raises(ValueError):
        teacher(tmp_path, sampling=sampling)


def test_missing_credentials_fail_offline_and_are_accounted(tmp_path, task, visible):
    model = Teacher(cache_dir=tmp_path / "cache", usage_ledger=tmp_path / "usage.jsonl",
                    run_id="test-run", secrets_file=tmp_path / "absent.env")
    with pytest.raises(TeacherCallError, match="initialized"):
        model.generate([], task, visible)
    rows = [json.loads(r) for r in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["request_started"] is False


def test_retry_success_and_cache_corruption(tmp_path, task, visible):
    model = teacher(tmp_path, client=FakeClient([TimeoutError("safe"), response()]))
    result = model.generate([], task, visible)
    assert result.metadata["attempt"] == 2
    path = next((tmp_path / "cache").glob("*.json"))
    cached = json.loads(path.read_text())
    cached["result"]["raw_text"] = "altered"
    path.write_text(json.dumps(cached))
    with pytest.raises(TeacherCallError, match="integrity"):
        model.generate([], task, visible)
