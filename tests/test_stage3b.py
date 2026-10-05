"""Stage 3b offline receipts, orchestration, and in-memory HTTP policies."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from baselines.agentgym.data import accumulate, prepare_dataset, load_dataset
from baselines.agentgym import pipeline as agentgym
from baselines.kang import pipeline as kang
from baselines.common.schedule import dose, lr_multiplier, warmup_count
from baselines.common.pipeline import batch_invariant, update_disclosure
from sed.react import build_messages
from sed.storage import write_json
from students.sag_student import SAGStudent, vote
from students.vllm_student import Student, StudentCallError
from test_stage2b_seeds import seed_bank, factories


@pytest.mark.parametrize("n,steps,warmup", [(5, 5, 0), (10, 10, 1), (20, 10, 1), (27, 9, 0)])
def test_agentgym_floor_warmup(n, steps, warmup):
    plan = dose(n, 32, epochs=1)
    assert plan["total_steps"] == steps
    assert warmup_count(steps, .1, int(.1 * steps)) == warmup
    values = [lr_multiplier(s, scheduler="linear", total_steps=steps, warmup_ratio=.1,
                            warmup_steps=warmup) for s in range(steps + 2)]
    assert values[0] == (0 if warmup else 1)
    assert values[steps:] == [0, 0]
    assert lr_multiplier(0, scheduler="linear", total_steps=9, warmup_ratio=.1) == 0  # HF ceil default unchanged.


def test_accumulating_filter_exact_order_strict_threshold_and_empty():
    seeds = [{"task_id": "a", "source": "seed", "id": "seed-a"}, {"task_id": "b", "id": "seed-b"}]
    rollouts = [{"task_id": task, "reward": reward, "training_eligible": True, "id": i}
                for i, (task, reward) in enumerate([("a", .99), ("b", 1), ("a", 1), ("b", 1), ("a", None)])]
    before = deepcopy(seeds)
    d1, report = accumulate(seeds, rollouts, ["a", "b"])
    assert d1 == [rollouts[1], rollouts[2], *seeds] and seeds == before
    assert (report["D_m"], report["F_m"], report["D_next"]) == (2, 2, 4)
    assert [d["decision"] for d in report["decisions"]] == ["reward_not_above_0.99", "accepted", "accepted",
        "duplicate_task_within_iteration", "unknown_or_unverified"]
    d2, _ = accumulate(d1, rollouts, ["a", "b"])
    assert d2 == [rollouts[1], rollouts[2], *d1]  # Repeated task AND identical trajectory survive across rounds.
    d3, report = accumulate(d2, [], ["a", "b"])
    assert d3 == d2 and report["empty_F"] and report["F_m"] == 0
    with pytest.raises(ValueError, match="only seed"):
        accumulate(seeds, [{"task_id": "dev"}], ["a", "b"])


def test_pure_rule_matches_pinned_official_filter():
    """Execute only the filter AST, using fake jsonlines and a memory filesystem."""
    import ast
    import io
    source = Path("third_party/AgentGym/agentenv/utils/agentevol_filter.py").read_text()
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))
                 and (not isinstance(n, ast.FunctionDef) or n.name in {"extract_category", "filter_jsonl"})]
    seeds = [{"item_id": "alfworld_0", "task_id": "a", "id": "seed"}]
    rows = [{"item_id": "alfworld_0", "task_id": "a", "reward": value,
             "training_eligible": True, "id": i} for i, value in enumerate([0, .99, 1, 1])]
    class Reader:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter(self, **kwargs): return iter(rows)
    class Writer(io.StringIO):
        def close(self): pass
    result = Writer()
    scope = {"os": SimpleNamespace(listdir=lambda _: ["inference.jsonl"], path=__import__("os").path),
        "jsonlines": SimpleNamespace(open=lambda _: Reader()), "json": json, "print": lambda *a: None,
        "open": lambda path, mode, **kw: io.StringIO(json.dumps(seeds)) if mode == "r" else result}
    exec(compile(tree, "pinned_agentgym_filter", "exec"), scope)
    scope["filter_jsonl"]("fake", "current", "next", True)
    actual, _ = accumulate(seeds, rows, ["a"])
    assert actual == json.loads(result.getvalue())


def test_checked_dataset_chain_keeps_seed_and_rollout_evidence(seed_bank, tmp_path):
    from teachers.collect_seeds import collect
    from sed.seed_selection import select_seeds
    from baselines.common.explore import run_exploration, explore_one
    from sed.runner import ScriptedPolicy
    from sed.storage import digest
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    root, manifest, freeze, config = seed_bank
    config["paths"].update(manifest=str(tmp_path / "m.json"), split_freeze=str(tmp_path / "f.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    env, teacher = factories(root, config, set(), [])
    verified = tmp_path / "verified"
    collection = collect(select_seeds(manifest, freeze, k=5), manifest, freeze, config, verified,
                         env_factory=env, teacher_factory=teacher)
    config["environment"]["max_episode_steps"] = 40
    d0 = tmp_path / "d0"
    prepare_dataset(verified, d0, config)
    seeds = load_dataset(d0 / "train_iter_0.json", config)
    ids = [r["task_id"] for r in seeds]
    tasks = tmp_path / "tasks.json"
    write_json(tasks, {"task_ids": ids, "list_sha256": digest(ids), "manifest_sha256": digest(manifest),
        "split_freeze_sha256": digest(freeze), "parent_seed_ids": {r["task_id"]: r["parent_seed_ids"] for r in seeds}})
    def backend(path, seed):
        value = FakeBackend(path, seed)
        value.win_at = 1
        return value
    def worker(job):
        return explore_one(job, env_factory=lambda c, t: Environment(root, max_episode_steps=40, backend_factory=backend),
                           policy_factory=lambda **kw: ScriptedPolicy(["look"]))
    exploration = tmp_path / "explore"
    run_exploration(config=config, tasks_path=tasks, output=exploration,
        spec={"enable_thinking": False, "max_tokens": 1024}, decoding="greedy", worker_fn=worker)
    d1 = tmp_path / "d1"
    report = prepare_dataset(verified, d1, config, previous=d0 / "train_iter_0.json", exploration=exploration, iteration=1)
    assert (report["F_m"], report["D_next"]) == (5, 10)
    rows = load_dataset(d1 / "train_iter_1.json", config)
    assert rows[5:] == seeds and all(r["source"] == "student_exploration" for r in rows[:5])
    assert "SECRET" not in json.dumps([r["messages"] for r in rows])
    # Persisted conversation tampering is rejected, not merely re-hashed.
    rows[0]["messages"][-1]["content"] = "Action: go north"
    write_json(d1 / "train_iter_1.json", rows)
    with pytest.raises(ValueError, match="checked source"):
        load_dataset(d1 / "train_iter_1.json", config)
    assert collection["K"] == 5


@pytest.mark.parametrize("successes", [0, 2])
def test_agentgym_four_base_restarts_empty_rounds_and_resume(tmp_path, monkeypatch, successes):
    args = SimpleNamespace(output=str(tmp_path), verified="verified", snapshot="BASE", config="config",
        train_python="TRAIN", env_python="ENV", model_name="gemma", workers=1, gpus=4, through="all")
    commands, prepared = [], []
    def prepare(verified, output, config, previous=None, exploration=None, iteration=0):
        output.mkdir(parents=True)
        # Even with every round empty, all four train/explore rounds must run.
        rows = [{"task_id": "a"}] * (5 + successes * iteration)
        write_json(output / f"train_iter_{iteration}.json", rows)
        write_json(output / "report.json", {"F_m": successes, "D_next": len(rows), "empty_F": successes == 0})
        prepared.append(iteration)
    @contextmanager
    def server(*args): yield "http://fake/v1"
    def command(cmd, log, **kwargs):
        commands.append(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True)
        n = len(json.loads(Path(cmd[cmd.index("--sft-data") + 1]).read_text())) if "--sft-data" in cmd else 0
        write_json(out / "result.json", {"examples_seen": n, "supervised_tokens_seen": n * 20})
    monkeypatch.setattr(agentgym, "prepare_dataset", prepare)
    monkeypatch.setattr(agentgym, "served", server)
    monkeypatch.setattr(agentgym, "run_command", command)
    monkeypatch.setattr(agentgym, "update_disclosure", lambda *a: None)
    agentgym.train_pipeline(args, {"K": 5}, {})
    assert prepared == list(range(5))
    training = [c for c in commands if "baselines.common.fsdp_train" in c]
    exploration = [c for c in commands if "baselines.common.explore" in c]
    assert len(training) == len(exploration) == 4
    for m, cmd in enumerate(training):
        assert cmd[cmd.index("--snapshot") + 1] == cmd[cmd.index("--base-snapshot") + 1] == "BASE"
        assert cmd[cmd.index("--sft-data") + 1].endswith(f"train_iter_{m}.json")
        assert cmd[cmd.index("--warmup-steps") + 1] == str(int(.1 * dose(5 + successes * m, 32, epochs=1)["total_steps"]))
    agentgym.train_pipeline(args, {"K": 5}, {})
    assert len(commands) == 8  # Actual stage hash checks reused all completed artifacts.
    result = json.loads((tmp_path / "result.json").read_text())
    assert not result["train_iter_4_trained"] and result["train_iter_4"].endswith("train_iter_4.json")
    for m in range(4):
        record = json.loads((tmp_path / f"iteration-{m}.json").read_text())
        assert record["F_m"] == successes and record["D_m"] == record["examples_seen"] == 5 + successes * m
    args.through = "train-0"
    agentgym.train_pipeline(args, {"K": 5}, {})
    assert len(json.loads((tmp_path / "datasets.json").read_text())) == 5


def test_kang_command_and_training_attempt_disclosure(tmp_path):
    args = SimpleNamespace(train_python="TRAIN", config="config", snapshot="BASE", verified="verified")
    command = kang.training_command(args, tmp_path / "out")
    values = {command[i]: command[i + 1] for i in range(len(command) - 1) if command[i].startswith("--")}
    assert values["--adaptation"] == "kang_lora" and values["--epochs"] == "2"
    assert values["--learning-rate"] == "2e-4" and values["--original-batch"] == "8"
    assert values["--scheduler"] == "linear" and values["--warmup-steps"] == "0"
    assert values["--seed"] == "0" and values["--max-length"] == "32768"
    assert values["--packing"] == kang.RECIPE["packing"] == "turn"
    assert "torch.distributed.run" not in command
    for i in range(2):
        path = tmp_path / f"stages/train-sft/attempt-{i:03d}/output"
        path.mkdir(parents=True)
        (path / "updates.jsonl").write_text(json.dumps({"examples_seen": 3, "supervised_tokens_seen": 7}) + "\n")
    write_json(path / "result.json", {"logical_forward_tokens": 15})
    update_disclosure(tmp_path, {"K": 5, "unique_instances": 5, "task_groups": 5})
    report = json.loads((tmp_path / "disclosure.json").read_text())
    assert report["totals"]["examples_seen"] == 6 and report["totals"]["supervised_tokens_seen"] == 14
    assert report["training_attempts"][0]["unknown_exposure_possible"]
    assert report["totals"]["teacher_calls"] == 0


@pytest.mark.parametrize("replies,expected,reason", [
    (["bad", "Action: b", "Action: a", "Action: a"], 2, "largest_group"),
    (["Action: b", "Action: a", "Action: a", "Action: b"], 0, "largest_group"),
    (["Action: x", "bad", "Thought: ok\nAction: b", "Action: a"], 2, "all_singletons_first_passing"),
    (["Action: x", "bad", "Action:"], 0, "no_passing_sample_0"),
])
def test_sag_vote(replies, expected, reason):
    result = vote(replies, ["a", "b"])
    assert result["selected_sample"] == expected and result["reason"] == reason


def response(replies):
    return {"id": "fake", "model": "gemma", "created": 1,
        "choices": [{"message": {"content": text}, "finish_reason": "stop", "index": i} for i, text in enumerate(replies)],
        "usage": {"prompt_tokens": 50, "completion_tokens": 80}}


def test_default_request_body_byte_identical(tmp_path):
    """Literal pre-Stage-3b request schema, including order and omitted defaults."""
    visible = {"observation": "room", "admissible_commands": ["look"]}
    history = [{"visible": visible, "reply": "Thought: visible\nAction: look"}]
    expected = {"model": "gemma", "messages": build_messages(history, {}, visible), "temperature": 0,
        "max_tokens": 1024, "stream": False, "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": False}
    captured = []
    def handler(request):
        captured.append(request.content)
        return httpx.Response(200, json=response(["Action: look"]))
    student = Student(endpoint="http://fake/v1", model="gemma", revision="pinned", enable_thinking=False,
        log_dir=tmp_path, transport=httpx.MockTransport(handler))
    result = student.generate(history, {}, visible)
    assert captured == [httpx.Request("POST", "http://fake/v1/chat/completions", json=expected).content]
    assert "sag" not in result.metadata and "decoding" not in result.metadata
    student.close()


def test_sag_request_candidates_thinking_and_total_usage(tmp_path):
    visible = {"observation": "room", "admissible_commands": ["look", "go north"]}
    replies = ["bad", "Action: go north", "Thought: ok\nAction: look", "Action: look", "Action: x", "bad", "bad", "bad"]
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        value = response(replies)
        value["choices"][7]["message"]["reasoning"] = "PRIVATE_THINKING"
        return httpx.Response(200, json=value)
    student = SAGStudent(endpoint="http://fake/v1", model="gemma", revision="pinned", enable_thinking=False,
        log_dir=tmp_path, transport=httpx.MockTransport(handler))
    result = student.generate([], {"task_id": "train/fake"}, visible)
    assert result.raw_text == replies[2] and result.parsed_action == "look"
    assert {k: captured[0][k] for k in ("n", "temperature", "top_p", "max_tokens")} == {
        "n": 8, "temperature": .4, "top_p": .8, "max_tokens": 1024}
    assert "PRIVATE_THINKING" not in json.dumps(captured)
    assert result.metadata["sag"]["selected_sample"] == 2 and len(result.metadata["sag"]["candidates"]) == 8
    assert result.metadata["completion_tokens"] == 80 and result.metadata["thinking_tokens"] is None
    assert result.metadata["thinking_chars"] == len("PRIVATE_THINKING")
    student.close()


def test_sag_short_response_fails_and_batch_invariance_guard(tmp_path, monkeypatch):
    student = SAGStudent(endpoint="http://fake/v1", model="gemma", revision="pinned", enable_thinking=False,
        log_dir=tmp_path, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response(["Action: look"]))))
    with pytest.raises(StudentCallError):
        student.generate([], {"task_id": "train/fake"}, {"observation": "room", "admissible_commands": ["look"]})
    student.close()
    monkeypatch.setenv("BATCH_INVARIANT", "0")
    with pytest.raises(ValueError): batch_invariant()
    monkeypatch.setenv("BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    batch_invariant()
    assert __import__("os").environ["VLLM_BATCH_INVARIANT"] == "1"


@pytest.mark.parametrize("decoding", ["greedy", "sag"])
def test_suite_cli_keeps_greedy_spec_unchanged(tmp_path, monkeypatch, decoding):
    import eval.run_suite as suite
    monkeypatch.setenv("ALFWORLD_DATA_ROOT", str(tmp_path))
    captured = {}
    def fake_suite(**kwargs):
        captured.update(kwargs)
        return {"error_episodes": 0}
    monkeypatch.setattr(suite, "run_suite", fake_suite)
    assert suite.main(["--output", str(tmp_path / "out"), "--policy", "student", "--decoding", decoding]) == 0
    spec = captured["spec"]
    assert (spec.get("decoding") == "sag") == (decoding == "sag")
    assert spec["max_tokens"] == 1024 and not spec["enable_thinking"]
    assert suite.policy_condition("student", spec, captured["config"]).endswith("_sag") == (decoding == "sag")


def test_sag_worker_records_vote_and_executes_only_winner(data_root, tmp_path, monkeypatch):
    import eval.run_suite as suite
    import students.sag_student as sag_module
    from sed.config import load_config
    from sed.manifest import build_manifest, freeze_splits, resolve_task
    from harness.alfworld_env import Environment
    from conftest import FakeBackend
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    task_id = next(t for t, a in freeze["assignments"].items() if a["split"] == "dev")
    task = resolve_task(manifest, freeze, task_id)
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=40)
    actions = []
    class Backend(FakeBackend):
        def __init__(self, *args):
            super().__init__(*args)
            self.win_at = 1
        def step(self, action):
            actions.append(action)
            return super().step(action)
    monkeypatch.setattr(suite, "_environment", lambda c, t: Environment(data_root, split=t["split"],
        max_episode_steps=40, backend_factory=Backend))
    result = response(["Action: go north", "Action: look", "Action: look"] + ["bad"] * 5)
    monkeypatch.setattr(sag_module, "SAGStudent", lambda **kw: SAGStudent(**kw,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=result))))
    spec = {"endpoint": "http://fake/v1", "model": "gemma", "revision": "pinned",
            "enable_thinking": False, "max_tokens": 1024, "decoding": "sag"}
    output = tmp_path / "episode"
    suite._worker((task, config, str(output), "student", spec, []))
    episode = json.loads((output / "episode.json").read_text())
    assert actions == ["look"] and episode["success"] is True
    assert episode["steps"][0]["policy_metadata"]["sag"]["selected_sample"] == 1
    assert len(episode["steps"][0]["policy_metadata"]["sag"]["candidates"]) == 8
    assert episode["metrics"]["completion_tokens"]["known_total"] == 80
    with pytest.raises(ValueError, match="batch-invariant"):
        suite.run_suite(config=config, tasks_path="unused", output=tmp_path / "forbidden", spec=spec)
    config["server_provenance"] = {"config": {"server": {"batch_invariant": True}}}
    with pytest.raises(ValueError, match="passing --lora-check"):
        suite.run_suite(config=config, tasks_path="unused", output=tmp_path / "forbidden", spec=spec)
    assert not (tmp_path / "forbidden").exists()
