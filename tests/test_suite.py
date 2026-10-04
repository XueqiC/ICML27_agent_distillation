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


def test_process_suite_with_fake_server_and_environment(data_root, tmp_path):
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
    output = tmp_path / "suite"
    summary = run_suite(config=config, tasks_path=path, output=output, spec=spec, workers=2, worker_fn=fake_worker)
    assert summary["episodes"] == summary["completed_episodes"] == 4
    assert summary["unique_instances"] == 4 and summary["task_groups"] == 2
    assert summary["totals"]["steps"] == summary["totals"]["legal_actions"] == 8
    assert summary["totals"]["prompt_tokens"] == 104
    assert summary["totals"]["thinking_tokens_unknown_calls"] == 8
    assert len((output / "per_task.tsv").read_text().splitlines()) == 5
    import os
    pids = {int(p.read_text()) for p in output.glob("episodes/*/test_pid.txt")}
    assert os.getpid() not in pids and len(pids) == 2
    assert not list(output.glob("**/failure.json"))
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
