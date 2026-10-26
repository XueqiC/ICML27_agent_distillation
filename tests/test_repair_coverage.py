from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest
import yaml

from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.manifest import TASK_TYPES, build_manifest, freeze_splits, resolve_task
from sed.storage import digest, file_hash, write_json
from students.sampled_student import SampledStudent
from students.vllm_student import Student
from tools import repair_coverage as coverage, rollout_pool


@pytest.fixture
def coverage_bank(tmp_path):
    root = tmp_path / "data"
    for source, count in (("train", 8), ("valid_seen", 1), ("valid_unseen", 1)):
        for i in range(count):
            path = root / source / f"room-{i}" / "trial_0"
            write_json(path / "traj_data.json", {"task_type": sorted(TASK_TYPES)[i % 6]})
            write_json(path / "game.tw-pddl", {"solvable": True})
    manifest = build_manifest(root)
    freeze = freeze_splits(manifest)
    tasks = [resolve_task(manifest, freeze, t) for t in freeze["seed_pool_candidates"]]
    config = load_config("configs/base_e4b.yaml")
    config["environment"].update(data_root=str(root), max_episode_steps=2)
    for field, value in (("manifest", manifest), ("split_freeze", freeze)):
        config["paths"][field] = str(tmp_path / f"{field}.json")
        write_json(config["paths"][field], value)
    splits = {"manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze),
              "sets": {"D_probe": {"group_ids": [tasks[-1]["group_id"]]}}}
    ids = [t["task_id"] for t in tasks[:6]]
    splits["sets"]["U_mech"] = {"name": "U_mech", "task_ids": ids, "list_sha256": digest(ids),
        "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze)}
    splits["content_sha256"] = digest(splits)
    write_json(tmp_path / "splits.json", splits)
    write_json(tmp_path / "tasks.json", ids)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    spec = dict(endpoint="http://fake/v1", model="adapter", revision="pinned", enable_thinking=False,
                max_tokens=1024)
    return manifest, freeze, tasks, config, spec, splits


class ActionBackend(FakeBackend):
    def step(self, action):
        self.win_at = 1 if action == "go north" else 99
        return super().step(action)


def test_real_pool_runner_fake_student_environment_and_records(coverage_bank, tmp_path, monkeypatch):
    m, f, tasks, config, spec, _ = coverage_bank
    calls, pool_calls = [], []
    original = rollout_pool.run_pool
    def track(*args, **kwargs):
        pool_calls.append(args[1]["r03_rollout"])
        return original(*args, **kwargs)
    monkeypatch.setattr(rollout_pool, "run_pool", track)
    def server(request):
        body = json.loads(request.content)
        calls.append(body)
        # Greedy parser failure is retained; sampled policy genuinely wins.
        reply = "Action: go north" if body["temperature"] else "malformed\nreply"
        return httpx.Response(200, json={"model": "adapter", "choices": [
            {"message": {"content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4}})
    def policy(directory, *, spec):
        spec = dict(spec)
        cls = SampledStudent if spec.pop("decoding") == "sampled" else Student
        return cls(log_dir=directory / "student_calls", transport=httpx.MockTransport(server), **spec)
    output = tmp_path / "run"
    summary = coverage.run_coverage(tasks[:6], config, output, spec=spec,
        env_factory=lambda _: Environment(config["environment"]["data_root"], max_episode_steps=2,
                                          backend_factory=ActionBackend), policy_factory=policy)
    assert [a["decoding"] for a in pool_calls] == ["greedy", "sampled"]
    assert [s["successes"] for s in summary["success_per_attempt"]] == [0, 6]
    assert [s["episodes"] for s in summary["success_per_attempt"]] == [6, 6]
    assert summary["branch_decision"] == "P0_K20" and summary["distinct_failed_tasks"] == 6
    assert summary["failed_episodes"] == 6 and summary["error_episodes"] == 0
    assert summary["H"] == 2 and summary["task_groups"] == summary["unique_instances"] == 6
    assert summary["new_teacher_calls"] == summary["training_exposure"] == 0
    assert summary["success_per_attempt"][0]["totals"]["parser_errors"] == 12
    assert summary["success_per_attempt"][0]["totals"]["environment_steps"] == 0
    records = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines()]
    assert len(records) == 12
    for record in records:
        task = next(t for t in tasks if t["task_id"] == record["task_id"])
        assert record == rollout_pool.episode_record(output / record["episode_dir"], task, relative_to=output)
        assert "attempt" not in record
        provenance = json.loads((output / record["episode_dir"] / "provenance.json").read_text())
        assert "p10" not in provenance["details"]["policy_condition"]
        assert provenance["details"]["policy_spec"]["model"] == "adapter"
        assert provenance["details"]["policy_spec"]["revision"] == "pinned"
    failures = json.loads((output / "failures.json").read_text())
    assert failures["list_sha256"] == digest(failures["failures"])
    assert [r["task_id"] for r in failures["failures"]] == [t["task_id"] for t in sorted(tasks[:6],
        key=lambda t: (t["task_type"], digest([20261028, t["task_id"]])))]
    assert all(r["attempt"] == 1 for r in failures["failures"])
    assert all("SECRET" not in json.dumps(b) for b in calls)
    sampled = [b for b in calls if b["temperature"]]
    assert len(sampled) == 6 and all(b["top_p"] == 0.95 and b["n"] == 1 for b in sampled)
    assert {b["seed"] for b in sampled} == {int(digest([0, t["task_id"], 0])[:8], 16) for t in tasks[:6]}


def failure_record(task, attempt, success=False):
    return {"task_id": task, "task_type": "kind", "group_id": task.rsplit("/", 1)[0],
        "episode_id": f"{task}-{attempt}", "attempt": attempt, "decoding": "greedy" if attempt == 1 else "sampled",
        "sampling_seed": None if attempt == 1 else attempt - 2, "episode_dir": f"episode-{attempt}",
        "episode_sha256": "episode-hash", "events_sha256": "event-hash", "provenance_sha256": "provenance-hash",
        "status": "ok", "complete": True, "success": success}


@pytest.mark.parametrize("count,extras,decision", [(5, 0, "need_extra_rollouts"),
    (5, 1, "K0_fallback"), (5, 2, "K0_fallback"), (6, 0, "P0_K20"), (6, 2, "P0_K20")])
def test_distinct_task_threshold_not_episode_threshold(count, extras, decision):
    records = [failure_record(f"train/{i}/trial", a) for a in range(1, 3 + extras) for i in range(count)]
    records.extend([failure_record("train/unknown/trial", 1, "unknown"),
                    {**failure_record("train/error/trial", 1), "status": "error", "complete": False}])
    result = coverage.summarize_coverage(records[::-1], [], extra_sampled=extras)
    assert result["branch_decision"] == decision
    assert result["distinct_failed_tasks"] == count
    assert len(result["failure_list"]) == count
    assert all(r["attempt"] == 1 for r in result["failure_list"])
    assert result["failures_by_task_type"] == {"kind": count}
    assert result["failed_episodes"] == count * (2 + extras)
    many = coverage.summarize_coverage([failure_record(f"train/{i}/trial", 1) for i in range(25)], [], extra_sampled=0)
    assert len(many["failure_list"]) == 20 and many["distinct_failed_tasks"] == 25


def test_extras_use_seeds_one_two_and_unknowns_remain_unknown(coverage_bank, tmp_path):
    _, _, tasks, config, spec, _ = coverage_bank
    specs = []
    class BrokenPolicy:
        def generate(self, *args):
            raise RuntimeError("SECRET_EXCEPTION")
    def policy(directory, *, spec):
        specs.append(spec)
        return BrokenPolicy()
    def environment(task):
        if task == tasks[1]:
            raise ValueError("SECRET_ENV_EXCEPTION")
        return Environment(config["environment"]["data_root"], max_episode_steps=2, backend_factory=ActionBackend)
    out = tmp_path / "unknowns"
    result = coverage.run_coverage(tasks[:2], config, out, spec=spec,
        env_factory=environment, policy_factory=policy, extra_sampled=2)
    assert [s.get("sampling_seed") for s in specs] == [None, 0, 1, 2]
    assert result["distinct_failed_tasks"] == 0
    assert result["error_episodes"] == result["unknown_success"] == 8
    assert result["branch_decision"] == "K0_fallback"
    assert result["branch_decision_actionable"] is False
    assert all(s["episodes"] == 2 and s["success_rate"] == 0 for s in result["success_per_attempt"])
    assert all("SECRET" not in p.read_text() for p in out.rglob("*.json*"))
    assert len(list(out.glob("attempt-*/failures.jsonl"))) == 4


def test_plan_only_exact_schedule_no_client_environment_or_output(coverage_bank, tmp_path, monkeypatch, capsys):
    _, _, tasks, config, _, _ = coverage_bank
    config["environment"].update(data_root="", max_episode_steps=40)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    def forbidden(*args, **kwargs):
        raise AssertionError("Plan touched a client or runner")
    monkeypatch.setattr(httpx, "Client", forbidden)
    monkeypatch.setattr(rollout_pool, "run_pool", forbidden)
    output = tmp_path / "plan"
    args = ["--config", str(tmp_path / "config.yaml"), "--tasks", str(tmp_path / "tasks.json"),
        "--repair-splits", str(tmp_path / "splits.json"), "--endpoint", "http://fake/v1",
        "--model-name", "adapter", "--extra-sampled", "2", "--output", str(output), "--plan-only"]
    assert coverage.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["episodes"]) == 24 and plan["H"] == 40
    assert [e["sampling_seed"] for e in plan["episodes"][::6]] == [None, 0, 1, 2]
    assert "--plan-only" not in plan["commands"][0] and "--extra-sampled 2" in plan["commands"][0]
    assert not output.exists()
    with pytest.raises(SystemExit):
        coverage.main([a if a != "2" else "3" for a in args])


def test_protected_tasks_rejected_before_execution(coverage_bank, tmp_path):
    m, f, tasks, _, _, splits = coverage_bank
    path = tmp_path / "bad.json"
    for tid in (tasks[-1]["task_id"], "valid_seen/room-0/trial_0", "valid_unseen/room-0/trial_0"):
        write_json(path, [tid])
        with pytest.raises(ValueError):
            coverage.coverage_tasks(path, m, f, repair_splits=splits)
    write_json(path, [tasks[0]["task_id"]] * 2)
    with pytest.raises(ValueError, match="unique"):
        coverage.coverage_tasks(path, m, f, repair_splits=splits)
    selected = coverage.coverage_tasks(tmp_path / "splits.json", m, f, repair_splits=splits, task_set="U_mech")
    assert selected == tasks[:6]
    bad = deepcopy(splits)
    bad["sets"]["U_mech"]["task_ids"].reverse()
    write_json(path, bad)
    with pytest.raises(ValueError, match="hash"):
        coverage.checked_repair_splits(path, m, f)


def test_seed_selection_runs_selected_view_only(coverage_bank, tmp_path):
    m, f, tasks, _, _, splits = coverage_bank
    value = {"manifest_sha256": digest(m), "split_freeze_sha256": digest(f),
        "selected": {"kind": [{"task_id": t["task_id"]} for t in tasks[:2]]},
        "task_order": [t["task_id"] for t in tasks[:2]][::-1],
        "replacements": {"kind": [{"task_id": tasks[-1]["task_id"]}]}}
    value["selection_sha256"] = digest(value)
    path = tmp_path / "seed-view.json"
    write_json(path, value)
    assert coverage.coverage_tasks(path, m, f, repair_splits=splits) == tasks[:2][::-1]
    value["task_order"].append(tasks[-1]["task_id"])
    value["selection_sha256"] = digest({k: v for k, v in value.items() if k != "selection_sha256"})
    write_json(path, value)
    with pytest.raises(ValueError, match="selected"):
        coverage.coverage_tasks(path, m, f, repair_splits=splits)


@pytest.mark.parametrize("extra", [-1, 3, True, 1.5])
def test_extra_cap(extra):
    with pytest.raises(ValueError):
        coverage.attempts(extra)
