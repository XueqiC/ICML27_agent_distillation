from copy import deepcopy
import json
from pathlib import Path

import pytest

from eval.run_suite import frozen_tasks, run_suite, summarize
from sed.config import load_config
from sed.dev_subset import build_dev40
from sed.manifest import build_manifest, freeze_splits, frozen_write, load_frozen
from sed.storage import digest, write_json


def test_dev40_determinism_and_frozen_artifact(tmp_path):
    manifest, freeze = load_frozen("data/manifest/alfworld_manifest.json", "data/manifest/alfworld_split_freeze.json")
    first = build_dev40(manifest, freeze)
    assert first == build_dev40(manifest, freeze)
    assert first == json.loads(Path("data/manifest/dev40.json").read_text())
    assert first["seed"] == 0 and first["list_sha256"] == digest(first["task_ids"])
    assert len(first["task_ids"]) == len(set(first["task_ids"])) == 40
    assert sorted(first["quotas"].values()) == [6, 6, 7, 7, 7, 7]
    assert first["quotas"] == first["counts_by_task_type"]
    assert all(t.startswith("valid_seen/") and freeze["assignments"][t]["split"] == "dev" for t in first["task_ids"])
    shuffled = dict(manifest, tasks=list(reversed(manifest["tasks"])))
    assert build_dev40(shuffled, freeze)["task_ids"] == first["task_ids"]
    path = tmp_path / "dev40.json"
    frozen_write(path, first)
    frozen_write(path, first)
    with pytest.raises(ValueError, match="overwrite"):
        frozen_write(path, dict(first, seed=1))


def fixture_list(data_root, tmp_path, split="dev"):
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    ids = sorted(t for t, a in freeze["assignments"].items() if a["split"] == split)
    listing = {"task_ids": ids, "list_sha256": digest(ids), "manifest_sha256": digest(manifest),
               "split_freeze_sha256": digest(freeze)}
    path = tmp_path / "tasks.json"
    write_json(path, listing)
    return path, manifest, freeze


def test_sealed_refusal_before_outputs(data_root, tmp_path):
    path, manifest, freeze = fixture_list(data_root, tmp_path, split="test")
    with pytest.raises(ValueError, match="sealed-final"):
        frozen_tasks(path, manifest, freeze)
    assert len(frozen_tasks(path, manifest, freeze, sealed_final=True)[1]) == 4
    with pytest.raises(ValueError, match="Teacher"):
        frozen_tasks(path, manifest, freeze, sealed_final=True, policy="teacher")


def test_list_tamper_refused(data_root, tmp_path):
    path, manifest, freeze = fixture_list(data_root, tmp_path)
    listing = json.loads(path.read_text())
    listing["task_ids"].reverse()
    path.write_text(json.dumps(listing))
    with pytest.raises(ValueError, match="hash differs"):
        frozen_tasks(path, manifest, freeze)


def fake_worker(job):
    """Spawn-safe test worker: real suite machinery with fake HTTP/environment."""
    import os
    import socket
    import httpx
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    from students.vllm_student import Student
    from test_react_student import FakeChatServer
    import eval.run_suite as suite
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden in spawned test worker")
    socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = forbidden
    socket.getaddrinfo = socket.create_connection = forbidden
    suite._environment = lambda config, task: Environment(config["environment"]["data_root"],
        split=task["split"], max_episode_steps=2, backend_factory=FakeBackend)
    suite.Student = lambda **kwargs: Student(**kwargs, transport=httpx.MockTransport(FakeChatServer()))
    result = suite._worker(job)
    (Path(job[2]) / "test_pid.txt").write_text(str(os.getpid()))
    return result


def fake_teacher_worker(job):
    import socket
    from types import SimpleNamespace
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    from teachers.openai_teacher import Teacher
    from test_teacher import response
    import eval.run_suite as suite
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden in spawned test worker")
    socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = forbidden
    socket.getaddrinfo = socket.create_connection = forbidden
    suite._environment = lambda config, task: Environment(config["environment"]["data_root"],
        split=task["split"], max_episode_steps=1, backend_factory=FakeBackend)
    suite.Teacher = lambda **kwargs: Teacher(**kwargs, client=SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_kwargs: response())))
    return suite._worker(job)


def test_teacher_suite_workers_share_run_limiter_and_config(data_root, tmp_path):
    path, manifest, freeze = fixture_list(data_root, tmp_path)
    config = load_config()
    assert config["models"]["teacher"]["max_attempts"] == 10
    assert config["models"]["teacher"]["rate_limit"] == {
        "requests_per_minute": 450, "tokens_per_minute": 180000}
    config["models"]["teacher"]["rate_limit"] = {"requests_per_minute": 10, "tokens_per_minute": 100000}
    config["environment"].update(data_root=str(data_root), max_episode_steps=1)
    for name, data in (("manifest", manifest), ("split_freeze", freeze)):
        destination = tmp_path / (name + ".json")
        write_json(destination, data)
        config["paths"][name] = str(destination)
    output = tmp_path / "teacher-suite"
    summary = run_suite(config=config, tasks_path=path, output=output, policy="teacher",
                        workers=2, worker_fn=fake_teacher_worker)
    assert summary["episodes"] == summary["completed_episodes"] == 4
    assert summary["batch_invariant"] is None
    states = list(output.glob("**/teacher_rate_limit/*.json"))
    assert len(states) == 1 and states[0].parent == output / "data/teacher_rate_limit"
    state = json.loads(states[0].read_text())
    assert state["limits"] == config["models"]["teacher"]["rate_limit"]
    assert len(state["reservations"]) == 4
    ledgers = list(output.glob("episodes/*/teacher_usage.jsonl"))
    assert len(ledgers) == 4
    assert all(json.loads(p.read_text().splitlines()[0])["reserved_tokens"] > 2048 for p in ledgers)


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize("batch_invariant", [None, False, True])
def test_process_suite_with_fake_server_and_environment(data_root, tmp_path, monkeypatch, adapter, batch_invariant):
    path, manifest, freeze = fixture_list(data_root, tmp_path)
    config = load_config()
    config["environment"]["data_root"] = str(data_root)
    for name, data in (("manifest", manifest), ("split_freeze", freeze)):
        destination = tmp_path / (name + ".json")
        write_json(destination, data)
        config["paths"][name] = str(destination)
    student = config["models"]["student"]
    spec = {"endpoint": "http://fake/v1", "model": student["requested_repo"],
            "revision": student["revision"], "enable_thinking": True}
    # A local launcher flag must never override recorded server provenance.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0" if batch_invariant else "1")
    server = {"config": {"server": {}}}
    if batch_invariant is not None:
        server["config"]["server"]["batch_invariant"] = batch_invariant
    config["server_provenance"] = server
    if adapter:
        from harness.lora_adapters import adapter_identity
        from tools.check_lora_serving import summarize_prompts
        record = {"name": "sft", "base_model": student["requested_repo"], "revision": student["revision"],
                  "adapter_config_sha256": "c", "adapter_weights_sha256": "w",
                  "training_state_sha256": "s", "training_provenance_sha256": "p"}
        server["config"]["server"].update(lora_adapters=[record],
                   endpoint=spec["endpoint"], served_model_name=student["requested_repo"])
        spec["model"] = "sft"
        config["server_provenance"] = server
        report = summarize_prompts([
            {"hf_logprobs": [-1], "hf_base_logprobs": [-2.01], "adapter_vllm_logprobs": [-1.01],
             "base_vllm_logprobs": [-2], "input_ids": [2, 106], "prompt_tokens": 1,
             "input_sha256": digest([2, 106]),
             "source": {"task_id": "train/task/1", "turn_index": i,
                        "training_eligible": True}} for i in range(8)])
        config["lora_equivalence"] = {**report, "model_name": "sft",
            "server_provenance_sha256": digest(server), "identity": {"adapter": adapter_identity(record)}}
    output = tmp_path / "suite"
    summary = run_suite(config=config, tasks_path=path, output=output, spec=spec, workers=2, worker_fn=fake_worker)
    assert summary["episodes"] == summary["completed_episodes"] == 4
    assert summary["batch_invariant"] is batch_invariant
    assert summary["unique_instances"] == 4 and summary["task_groups"] == 2
    assert summary["totals"]["steps"] == summary["totals"]["legal_actions"] == 8
    assert summary["totals"]["prompt_tokens"] == 104
    assert summary["totals"]["thinking_tokens_unknown_calls"] == 8
    assert len((output / "per_task.tsv").read_text().splitlines()) == 5
    import os
    pids = {int(p.read_text()) for p in output.glob("episodes/*/test_pid.txt")}
    assert os.getpid() not in pids and len(pids) == 2
    assert not list(output.glob("**/failure.json"))
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["config"]["suite"]["model_name"] == spec["model"]
    assert provenance["config"]["suite"]["batch_invariant"] is batch_invariant
    assert all(json.loads(p.read_text())["config"]["suite"]["batch_invariant"] is batch_invariant
               for p in output.glob("episodes/*/provenance.json"))
    assert provenance["config"]["suite"]["student"]["model"] == spec["model"]
    with pytest.raises(FileExistsError):
        run_suite(config=config, tasks_path=path, output=output, spec=spec)
    with pytest.raises(ValueError, match="training data"):
        run_suite(config=config, tasks_path=path, output=Path(config["paths"]["raw"]) / "forbidden", spec=spec)


def test_unknown_results_stay_in_denominator():
    rows = [{"task_id": str(i), "group_id": "same_group", "task_type": "kind", "split": "dev",
             "success": success, "status": "error" if success == "unknown" else "ok",
             "steps": 2, "episode_dir": "/tmp/fake"} for i, success in enumerate((True, False, "unknown"))]
    summary = summarize(rows, 1.5)
    assert summary["success_rate"] == 1 / 3 and summary["unknown_success"] == 1
    assert summary["task_groups"] == 1 and summary["error_episodes"] == 1


@pytest.mark.parametrize("name", [None, "sft-checkpoint"])
def test_cli_model_name_and_provenance(tmp_path, monkeypatch, name):
    import eval.run_suite as suite
    monkeypatch.setenv("ALFWORLD_DATA_ROOT", str(tmp_path))
    captured = {}
    def fake_suite(**kwargs):
        captured.update(kwargs)
        return {"error_episodes": 0}
    monkeypatch.setattr(suite, "run_suite", fake_suite)
    args = ["--output", str(tmp_path / "out")]
    if name:
        args += ["--model-name", name]
    assert suite.main(args) == 0
    assert captured["spec"]["model"] == (name or load_config()["models"]["student"]["requested_repo"])
    assert captured["spec"]["max_tokens"] == 1024 and captured["spec"]["enable_thinking"] is False
    assert suite.policy_condition("student", captured["spec"], captured["config"]) == (
        "lora_student_thinking_false" if name else "base_student_thinking_false")


def test_adapter_suite_refuses_missing_equivalence_before_output(tmp_path):
    with pytest.raises(ValueError, match="passing --lora-check"):
        run_suite(config=load_config(), tasks_path="data/manifest/dev40.json", output=tmp_path / "out",
                  spec={"model": "sft", "endpoint": "http://127.0.0.1:12345/v1"})
    assert not (tmp_path / "out").exists()
