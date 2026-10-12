import json
from functools import partial
import os

import pytest

from conftest import FakeBackend
from data.manifest.build_r03_sets import listing, load_r03_dataset, dataset_entry, publish_dataset
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.manifest import build_manifest, freeze_splits, resolve_task
from sed.runner import ScriptedPolicy
from sed.storage import write_json
from tools.rollout_pool import PoolStudent, filter_successes, read_rollouts, run_pool
from test_r03_sets import TinyTokenizer


def sampled_rollout(tmp_path, data_root, *, label_samples=2, fail_label=False):
    import httpx
    from sed.storage import file_hash
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    task = resolve_task(manifest, freeze, freeze["seed_pool_candidates"][0])
    checkpoint = tmp_path / "checkpoint-tokens-104070"
    checkpoint.mkdir()
    write_json(checkpoint / "training_state.json", {"updates": 115, "supervised_tokens": 104070})
    (checkpoint / "adapter_model.safetensors").write_bytes(b"mock identity only")
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=2)
    config["lora_equivalence"] = {"identity": {"adapter": {
        "training_state_sha256": file_hash(checkpoint / "training_state.json"),
        "adapter_weights_sha256": file_hash(checkpoint / "adapter_model.safetensors")}}}
    calls = []
    def server(request):
        body = json.loads(request.content)
        calls.append(body)
        index = (len(calls) - 1) % (label_samples + 1)
        if fail_label and index == 1:
            return httpx.Response(500, text="PRIVATE_SERVER_ERROR")
        content = "Action: look" if index == 0 else " Action: go north \n" if index == 1 else "malformed"
        return httpx.Response(200, json={"id": str(len(calls)), "model": "p10", "created": 123,
            "choices": [{"message": {"content": content}, "finish_reason": "length" if index == 2 else "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 4,
                      "completion_tokens_details": {"reasoning_tokens": 0}}})
    output = tmp_path / "sampled"
    run_pool([task], config, output,
        env_factory=lambda _: Environment(data_root, max_episode_steps=2, backend_factory=FakeBackend),
        policy_factory=lambda directory: PoolStudent(label_samples=label_samples,
            endpoint="http://mock/v1", model="p10", revision="student-rev", enable_thinking=False,
            max_tokens=1024, log_dir=directory / "student_calls", transport=httpx.MockTransport(server)))
    pool = tmp_path / "sampled-pool.json"
    write_json(pool, listing("r03_pool200", [task], manifest, freeze,
        subsets={key: [task["task_id"]] for key in ("R1_P2", "R1_P3", "R2")}))
    records = read_rollouts([output / "rollouts.jsonl"], manifest, freeze, pool)
    return manifest, freeze, output, pool, records, calls, checkpoint


@pytest.mark.parametrize("label_samples", [0, 2])
def test_sampled_labels_never_drive_episode_and_retain_calls(tmp_path, data_root, label_samples):
    m, f, output, pool, records, calls, _ = sampled_rollout(tmp_path, data_root, label_samples=label_samples)
    record, directory = records[0]
    assert len(record["states"]) == 2 and len(calls) == 2 * (1 + label_samples)
    for index, state in enumerate(record["states"]):
        assert state["student_action"] == "look" and state["environment_steps"] == 1
        assert len(state["label_samples"]) == label_samples
        assert all(turn["reply"] == "Action: look" for turn in state["history"])
        group = calls[index * (1 + label_samples):(index + 1) * (1 + label_samples)]
        assert group[0]["temperature"] == 0
        for call in group:
            assert call["messages"] == state["messages"]
            assert call["model"] == "p10" and call["max_tokens"] == 1024
            assert call["chat_template_kwargs"] == {"enable_thinking": False}
            assert "SECRET" not in json.dumps(call)
        for sample, call in zip(state["label_samples"], group[1:]):
            assert all(call[k] == v for k, v in PoolStudent.label_sampling.items())
            assert sample["metadata"]["prompt_tokens"] == 50
            assert sample["metadata"]["completion_tokens"] == 4
            assert sample["metadata"]["returned_model_id"] == "p10"
    ledger = [json.loads(s) for s in (directory / "student_calls/calls.jsonl").read_text().splitlines()]
    assert sum(r["status"] == "completed" for r in ledger) == len(calls)
    individual = next((output / "episode_jsonl").glob("*.jsonl"))
    assert read_rollouts([individual], m, f, pool)[0][0]["states"] == record["states"]
    if label_samples:
        record["states"][0]["label_samples"][0]["reply"] = "tampered"
        (output / "rollouts.jsonl").write_text(json.dumps(record) + "\n")
        with pytest.raises(ValueError, match="evidence"):
            read_rollouts([output / "rollouts.jsonl"], m, f, pool)


def test_failed_label_is_logged_without_affecting_greedy_episode(tmp_path, data_root):
    _, _, _, _, records, calls, _ = sampled_rollout(tmp_path, data_root, fail_label=True)
    record, directory = records[0]
    assert record["complete"] and record["status"] == "ok" and len(calls) == 6
    assert all(s["student_action"] == "look" for s in record["states"])
    assert all(s["label_samples"][0]["status"] == "failed" for s in record["states"])
    assert all(s["label_samples"][1]["status"] == "completed" for s in record["states"])
    assert "PRIVATE_SERVER_ERROR" not in json.dumps(record)
    assert "PRIVATE_SERVER_ERROR" not in (directory / "student_calls/calls.jsonl").read_text()


def test_negative_label_count_cli_rejected_before_loading_inputs(tmp_path):
    from tools.rollout_pool import main
    with pytest.raises(SystemExit):
        main(["run", "--label-samples", "-1", "--output", str(tmp_path / "unused")])


@pytest.mark.parametrize("workers", ["0", "-1"])
def test_invalid_worker_count_cli_rejected_before_loading_inputs(tmp_path, workers):
    from tools.rollout_pool import main
    with pytest.raises(SystemExit):
        main(["run", "--workers", workers, "--output", str(tmp_path / "unused")])


def fake_worker_environment(task, *, data_root, fail_task):
    if task["task_id"] == fail_task:
        raise ValueError("PRIVATE_FAILURE_MESSAGE")
    return Environment(data_root, max_episode_steps=2, backend_factory=FakeBackend)


def fake_worker_policy(directory):
    policy = ScriptedPolicy(["look"])
    policy.spec = {"pid": os.getpid(), "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}
    return policy


@pytest.mark.parametrize("workers", [1, 2])
def test_rollout_workers_isolate_episodes_keep_order_and_failed_attempts(tmp_path, data_root, workers):
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    tasks = [resolve_task(manifest, freeze, t) for t in freeze["seed_pool_candidates"][:3]]
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=2)
    output = tmp_path / "rollout"
    result = run_pool(tasks, config, output, workers=workers,
        env_factory=partial(fake_worker_environment, data_root=data_root, fail_task=tasks[1]["task_id"]),
        policy_factory=fake_worker_policy)
    assert result["games"] == 3 and result["failed_episodes"] == 1
    assert json.loads((output / "failures.jsonl").read_text()) == {
        "task_id": tasks[1]["task_id"], "error_type": "ValueError"}
    pool = tmp_path / "pool.json"
    write_json(pool, listing("r03_pool200", tasks, manifest, freeze))
    records = read_rollouts([output / "rollouts.jsonl"], manifest, freeze, pool)
    assert [record["task_id"] for record, _ in records] == [t["task_id"] for t in (tasks[0], tasks[2])]
    assert all(len(record["states"]) == 2 for record, _ in records)
    for _, directory in records:
        provenance = json.loads((directory / "provenance.json").read_text())
        worker = provenance["details"]["policy_spec"]
        assert (worker["pid"] == os.getpid()) is (workers == 1)
        assert worker["cuda"] == ""
        assert provenance["config"]["r03_rollout"]["workers"] == workers
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["config"]["r03_rollout"]["workers"] == workers


@pytest.fixture
def rollout(tmp_path, data_root):
    m = build_manifest(data_root)
    f = freeze_splits(m)
    task = resolve_task(m, f, f["seed_pool_candidates"][0])
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=3)
    def backend(path, seed):
        value = FakeBackend(path, seed)
        value.win_at = 2
        return value
    def environment(task):
        return Environment(data_root, max_episode_steps=3, backend_factory=backend)
    output = tmp_path / "rollout"
    run_pool([task], config, output, env_factory=environment,
             policy_factory=lambda _: ScriptedPolicy(["look"]))
    pool = tmp_path / "pool.json"
    write_json(pool, listing("r03_pool200", [task], m, f, subsets={"R1_P2": [task["task_id"]],
        "R1_P3": [task["task_id"]], "R2": [task["task_id"]]}))
    records = read_rollouts([output / "rollouts.jsonl"], m, f, pool)
    return m, f, task, output, pool, records


def test_rollout_prompt_history_reward_success_and_filter(rollout, tmp_path):
    m, f, task, output, pool, records = rollout
    record, _ = records[0]
    assert len(record["states"]) == 2 and record["success"] is True
    first, last = record["states"]
    assert first["history"] == [] and first["reward"] == 0 and not first["success"]
    assert last["student_action"] == "look" and last["reward"] == 1 and last["success"] is True
    assert last["history"][0]["reply"] == "Action: look"
    assert all("SECRET" not in json.dumps(s["messages"]) for s in record["states"])
    destination = tmp_path / "p3"
    result = filter_successes(records, destination, m, f)
    assert result["trusted_executed_trajectories"] == 1 and result["new_teacher_calls"] == 0
    rows = load_r03_dataset(destination, TinyTokenizer(), m, f)
    assert len(rows) == 1 and rows[0]["turn_count"] == 2


def test_rollout_tamper_rejected(rollout):
    m, f, task, output, pool, records = rollout
    path = output / "rollouts.jsonl"
    row = json.loads(path.read_text())
    row["states"][0]["reward"] = 100
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="evidence"):
        read_rollouts([path], m, f, pool)


def test_individual_episode_jsonl_resolves_evidence(rollout):
    m, f, _, output, pool, records = rollout
    individual = next((output / "episode_jsonl").glob("*.jsonl"))
    loaded = read_rollouts([individual], m, f, pool)
    assert loaded[0][0]["states"] == records[0][0]["states"]


def test_false_success_flag_is_not_environment_evidence(rollout, tmp_path):
    m, f, task, output, pool, records = rollout
    row, directory = records[0]
    path = directory / "episode.json"
    ep = json.loads(path.read_text())
    ep["steps"] = ep["steps"][:1]
    ep["success"] = True  # final verification blob still says won=False
    write_json(path, ep)
    result = filter_successes([(row, directory)], tmp_path / "p3", m, f)
    assert result["episode_count"] == 0


def test_p3_empty_student_turn_stays_context_only(rollout, tmp_path):
    m, f, _, _, _, records = rollout
    ep = json.loads((records[0][1] / "episode.json").read_text())
    ep["steps"][0]["action_raw"] = ""
    destination = tmp_path / "empty_history"
    entry = dataset_entry(destination, ep, source="student_real_environment_win")
    publish_dataset(destination, [entry], m, f, source="R1_P3")
    row = load_r03_dataset(destination, TinyTokenizer(), m, f)[0]
    assert row["empty_visible_turns_masked"] == 1
    assert len(row["assistant_spans"]) == 1


def test_parser_feedback_keeps_shared_harness_semantics(tmp_path, data_root):
    m = build_manifest(data_root)
    f = freeze_splits(m)
    task = resolve_task(m, f, f["seed_pool_candidates"][0])
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=2)
    output = tmp_path / "bad"
    run_pool([task], config, output,
        env_factory=lambda _: Environment(data_root, max_episode_steps=2, backend_factory=FakeBackend),
        policy_factory=lambda _: ScriptedPolicy(["missing\naction"]))
    record = json.loads((output / "rollouts.jsonl").read_text())
    assert len(record["states"]) == 2
    assert all(s["action_outcome"] == "parser_error" and s["environment_steps"] == 0 for s in record["states"])


def test_failed_policy_attempt_keeps_prompt_and_unknown_reward(tmp_path, data_root):
    m = build_manifest(data_root)
    f = freeze_splits(m)
    task = resolve_task(m, f, f["seed_pool_candidates"][0])
    config = load_config()
    class BrokenPolicy:
        def generate(self, *args):
            raise RuntimeError("PRIVATE_FAILURE_MESSAGE")
    output = tmp_path / "failed"
    result = run_pool([task], config, output,
        env_factory=lambda _: Environment(data_root, max_episode_steps=2, backend_factory=FakeBackend),
        policy_factory=lambda _: BrokenPolicy())
    assert result["failed_episodes"] == 1
    record = json.loads((output / "rollouts.jsonl").read_text())
    assert record["states"] == [] and len(record["failed_attempts"]) == 1
    failure = record["failed_attempts"][0]
    assert failure["reward"] is None and failure["success"] == "unknown"
    assert failure["messages"][-1]["role"] == "user"
    assert "PRIVATE_FAILURE_MESSAGE" not in json.dumps(record)
