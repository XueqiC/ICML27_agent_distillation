import hashlib
import json

import httpx
import pytest

from sed.storage import BlobStore
from students.sampled_student import SampledStudent
from students.sag_student import SAGStudent
from students.vllm_student import Student


def test_sampled_requests_match_sag_seed_rule_and_visible_messages(tmp_path, task, visible):
    calls = []
    def server(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"model": "adapter", "id": "fake", "created": 123,
            "choices": [{"message": {"content": "Action: look"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    kwargs = dict(endpoint="http://fake/v1", model="adapter", revision="pinned",
        enable_thinking=False, max_tokens=1024, log_dir=tmp_path, transport=httpx.MockTransport(server))
    history = []
    with_hidden = {**task, "extra.expert_plan": ["SECRET"], "internal_state": "SECRET"}
    for sampling_seed in (0, 1, 2):
        student = SampledStudent(sampling_seed=sampling_seed, **kwargs)
        sag = SAGStudent(sampling_seed=sampling_seed, **kwargs)
        for turn in range(3):
            history = [{"visible": visible, "reply": "Action: look"}] * turn
            reply = student.generate(history, with_hidden, {**visible, "hidden_success": "SECRET"})
            body = calls[-1]
            canonical = json.dumps([sampling_seed, task["task_id"], turn], sort_keys=True,
                                   separators=(",", ":"), ensure_ascii=False).encode()
            seed = int(hashlib.sha256(canonical).hexdigest()[:8], 16)
            assert body["seed"] == seed == sag.request_sampling(history, task)["seed"]
            assert {k: body[k] for k in ("temperature", "top_p", "n")} == {"temperature": 0.7, "top_p": 0.95, "n": 1}
            assert body["messages"] == Student(**kwargs).messages(history, task, visible)
            assert "SECRET" not in json.dumps(body) and task["task_id"] not in json.dumps(body["messages"])
            assert body["max_tokens"] == 1024 and body["chat_template_kwargs"] == {"enable_thinking": False}
            assert reply.parsed_action == "look" and reply.metadata["seed"] == seed
            assert BlobStore(tmp_path / "blobs").get(reply.metadata["request_ref"]) == body
        assert student.spec["request_seed_rule"] == sag.spec["request_seed_rule"]
        student.close()
    assert len(calls) == 9


def test_seed_identity_validation_and_pool_factory(tmp_path, task):
    from tools.rollout_pool import _pool_policy
    kwargs = dict(endpoint="http://fake/v1", model="adapter", revision="pinned", enable_thinking=False)
    with pytest.raises(ValueError, match="integer"):
        SampledStudent(sampling_seed=True, log_dir=tmp_path, **kwargs)
    student = SampledStudent(log_dir=tmp_path, task_id=task["task_id"], **kwargs)
    assert student.request_sampling([], {}) == student.request_sampling([], task)
    with pytest.raises(ValueError, match="identity"):
        SampledStudent(log_dir=tmp_path, **kwargs).request_sampling([], {})
    assert isinstance(_pool_policy(tmp_path, spec={**kwargs, "decoding": "sampled", "sampling_seed": 2}), SampledStudent)
    with pytest.raises(ValueError, match="label samples"):
        _pool_policy(tmp_path, spec={**kwargs, "decoding": "sampled", "label_samples": 1})


def test_suite_cli_selects_sampled_and_rejects_teacher(monkeypatch, tmp_path):
    from eval import run_suite as suite
    captured = {}
    monkeypatch.setenv("ALFWORLD_DATA_ROOT", str(tmp_path))
    def fake_suite(**kwargs):
        captured.update(kwargs)
        return {"error_episodes": 0}
    monkeypatch.setattr(suite, "run_suite", fake_suite)
    assert suite.main(["--decoding", "sampled", "--sampling-seed", "2", "--output", str(tmp_path / "out")]) == 0
    assert captured["spec"]["decoding"] == "sampled" and captured["spec"]["sampling_seed"] == 2
    assert suite.policy_condition("student", captured["spec"], captured["config"]).endswith("_sampled")
    with pytest.raises(SystemExit):
        suite.main(["--decoding", "sampled", "--policy", "teacher", "--output", str(tmp_path / "out")])


def test_suite_worker_uses_sampled_class(monkeypatch, tmp_path, task):
    from eval import run_suite as suite
    from sed.config import load_config
    captured = {}
    monkeypatch.setattr(suite, "verify_task_files", lambda *a: None)
    monkeypatch.setattr(suite, "_environment", lambda *a: object())
    def fake_episode(env, policy, *args, **kwargs):
        captured.update(policy=policy, condition=kwargs["policy_condition"])
    monkeypatch.setattr(suite, "run_episode", fake_episode)
    spec = dict(endpoint="http://fake/v1", model="adapter", revision="pinned", enable_thinking=False,
                decoding="sampled", sampling_seed=2)
    suite._worker((task, load_config(), tmp_path, "student", spec, []))
    assert isinstance(captured["policy"], SampledStudent)
    assert captured["policy"].task_id == task["task_id"]
    assert captured["policy"].sampling_seed == 2
    assert captured["condition"].endswith("_sampled")
