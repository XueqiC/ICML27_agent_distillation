"""Offline AgentDistill security, real-transition accounting and stage contracts."""
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from baselines.agentdistill import teacher_cap
from baselines.agentdistill.box import build, parse_clusters, load_box
from baselines.agentdistill.collection import collect, checked_pool, tasks_for
from baselines.agentdistill.policy import BoxStudent, Manager, ToolPolicy
from baselines.agentdistill.sandbox import Machine, Rejected, validate
from baselines.agentdistill.teacher import Budget, PromptTeacher, read
from baselines.common.pipeline import prepare
from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.manifest import resolve_task
from sed.runner import run_episode, replay_episode
from sed.storage import digest, file_hash, write_json
from teachers.capped import CappedTeacher, CollectorHalted
from teachers.openai_teacher import TeacherResult
from test_capped_teacher import RawClient
from test_stage2b_seeds import seed_bank, factories


def tool(code=None, name="move", parameters=None):
    return {"name": name, "description": "Perform one H0 command, supplied by command.",
        "parameters": ["command"] if parameters is None else parameters,
        "code": code or f"def {name}(env, command):\n    return env.step(command)"}


def page(text="Room", commands=None):
    return {"observation": text, "admissible_commands": sorted(commands or ["look", "go north"])}


@pytest.mark.parametrize("source", [
    "import os\ndef move(env, command):\n    return env.step(command)",
    "def move(env, command):\n    return open('/tmp/escape', 'w')",
    "def move(env, command):\n    return __import__('os').system(command)",
    "def move(env, command):\n    return env.__class__.__mro__",
    "def move(env, command):\n    return env.step.__globals__",
    "def move(env, command):\n    return getattr(env, command)",
    "def move(env, command):\n    env = command\n    return env.step(command)",
    "def move(env, command):\n    env.step = command\n    return env.step(command)",
    "def move(env, command):\n    return [env.step(command) for x in command]",
    "def move(env, command):\n    return move(env, command)",
    "def move(env, command=open('x')):\n    return env.step(command)",
    "@evil\ndef move(env, command):\n    return env.step(command)",
    "def move(env, command):\n    try:\n        return env.step(command)\n    except:\n        pass",
    "def move(env, command):\n    return command * 1000000000",
    "def move(env, command):\n    return f'{command!r}'",
    "def move(env, command):\n    yield command",
])
def test_sandbox_rejects_host_access_and_unbounded_primitives(source):
    with pytest.raises(Rejected):
        validate(tool(source))


def test_sandbox_dynamic_actions_return_visible_only():
    code = """def move(env, command):
    results = []
    for action in [command, 'go north']:
        if action in env.observe()['admissible_commands']:
            results.append(env.step(action))
    return results
"""
    start = {**page(), "extra.expert_plan": "SECRET", "internal_state": "SECRET"}
    machine = Machine(tool(code), {"command": "look"}, start)
    gen = machine.run()
    assert next(gen) == "look"
    assert gen.send({**page("after look"), "won": "SECRET"}) == "go north"
    with pytest.raises(StopIteration) as done:
        gen.send({**page("after north"), "audit": "SECRET"})
    assert done.value.value == [page("after look"), page("after north")]
    assert "SECRET" not in json.dumps(machine.observations)


def test_sandbox_rechecks_admissibility_each_action_and_rejects_fabricated_return():
    gen = Machine(tool("def move(env, command):\n    a = env.step(command)\n    return env.step('go north')"),
                  {"command": "look"}, page()).run()
    assert next(gen) == "look"
    with pytest.raises(Rejected, match="currently H0"):
        gen.send(page("changed", ["look"]))
    gen = Machine(tool("def move(env, command):\n    env.step(command)\n    return 'success'"),
                  {"command": "look"}, page()).run()
    next(gen)
    with pytest.raises(Rejected, match="actual action observations"):
        gen.send(page())


def test_sandbox_fuel_memory_arguments_and_parameter_limit():
    with pytest.raises(Rejected, match="instruction"):
        next(Machine(tool("def move(env, command):\n    while True:\n        pass"),
                     {"command": "look"}, page(), fuel=30).run())
    with pytest.raises(Rejected, match="value limit"):
        next(Machine(tool("def move(env, command):\n    while True:\n        command = command + command"),
                     {"command": "look"}, page()).run())
    with pytest.raises(Rejected):
        Machine(tool(), {"command": {"__class__": "x"}}, page())
    with pytest.raises(Rejected):
        validate(tool("def move(env, a, b, c, d):\n    return env.step(a)", parameters=["a", "b", "c", "d"]))


class Script:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, history, task, visible):
        self.prompts.append(deepcopy([history, task, visible]))
        raw = self.replies.pop(0)
        return TeacherResult(raw, None, {"prompt_tokens": 5, "completion_tokens": 3,
            "thinking_tokens": 0, "thinking_chars": 0})


def env_factory(root, win_at=99, resets=None):
    def create(config, task):
        if resets is not None:
            resets.append(task["task_id"])
        def backend(path, seed):
            result = FakeBackend(path, seed)
            result.win_at = win_at
            return result
        return Environment(root, max_episode_steps=40, backend_factory=backend)
    return create


def first_task(seed_bank):
    root, manifest, freeze, config = seed_bank
    config["environment"]["max_episode_steps"] = 40
    task = resolve_task(manifest, freeze, next(t["task_id"] for t in manifest["tasks"] if t["source_split"] == "train"))
    task["parent_seed_ids"] = ["trusted-seed"]
    return root, task, config


@pytest.mark.parametrize("win_at,expected", [(99, 40), (3, 3)])
def test_shared_runner_caps_each_tool_action_and_replays(seed_bank, tmp_path, win_at, expected):
    root, task, config = first_task(seed_bank)
    code = "def move(env, command):\n    while True:\n        env.step(command)"
    model = Script(['Thought: repeat\nAction: mcp {"name":"move","arguments":{"command":"look"}}'])
    policy = ToolPolicy(model, [tool(code)], tmp_path / "calls")
    factory = env_factory(root, win_at)
    episode = run_episode(factory(config, task), policy, task, config, tmp_path / "episode",
        policy_condition="agentdistill", parent_seed_ids=task["parent_seed_ids"])
    policy.finish(episode)
    assert len(episode["steps"]) == expected
    assert episode["metrics"]["environment_steps"] == expected
    assert episode["metrics"]["prompt_tokens"]["known_total"] == 5
    assert episode["agentdistill_model_turns"] == 1
    assert len(model.prompts) == 1
    assert episode["success"] is (win_at == 3)
    assert replay_episode(factory(config, task), episode)["matched"]
    assert policy.records[0]["status"] == "stopped_at_episode_boundary"
    assert policy.records[0]["actions"] == expected


def test_student_receives_tool_observations_and_all_tools_without_lineage(tmp_path):
    from baselines.agentdistill.prompts import messages
    tools = [tool(), tool(name="second")]
    source = Script(['Action: mcp {"name":"move","arguments":{"command":"look"}}', 'Action: go north'])
    policy = ToolPolicy(source, tools, tmp_path)
    assert policy.generate([], {}, page()).parsed_action == "look"
    assert policy.generate([], {}, {**page("actual result"), "expert_plan": "SECRET"}).parsed_action == "go north"
    assert "actual result" in json.dumps(source.prompts[-1])
    assert "SECRET" not in json.dumps(source.prompts)
    rendered = messages([], {"task_id": "SECRET"}, {**page(), "audit": "SECRET"},
                        [{**t, "source": "SECRET"} for t in tools])
    assert "SECRET" not in json.dumps(rendered)
    assert all(t["name"] in rendered[0]["content"] for t in tools)


def test_student_cannot_create_tools_and_invalid_calls_use_model_budget(tmp_path):
    source = Script(['Action: create_mcp {"request":"x"}'] * 40)
    policy = ToolPolicy(source, [], tmp_path)
    assert policy.generate([], {}, page()).parsed_action is None
    assert len(source.prompts) == 40
    assert policy.generate([], {}, page()).parsed_action is None
    assert len(source.prompts) == 40


def budget_factory(client):
    def factory(config, *, output, cache_dir, usage_ledger, run_id, cap, teacher_class, sampling):
        return CappedTeacher(counter_path=Path(output) / "counter.json", cap=cap,
            cache_dir=cache_dir, usage_ledger=usage_ledger, run_id=run_id, client=client,
            teacher_class=teacher_class, sampling=sampling, sleep=lambda _: None)
    return factory


def test_capped_teacher_custom_prompts_no_hidden_inputs_and_resume(tmp_path, task):
    task = {**task, "source_split": "train", "parent_seed_ids": ["seed"], "extra.expert_plan": "SECRET"}
    client = RawClient([json.dumps(tool())])
    identity = {"unique_instances": 1}
    config = {"models": {"teacher": {"sampling": {}}}}
    budget = Budget(tmp_path, config, identity, factory=budget_factory(client))
    args = (["abstract", "a"], "abstract", task, "CUSTOM ABSTRACT PROMPT")
    raw = budget.request(*args, visible={**page(), "audit": "SECRET"})
    again = Budget(tmp_path, config, identity, factory=budget_factory(client))
    assert again.request(*args, visible=page()) == raw and len(client.requests) == 1
    assert "CUSTOM ABSTRACT PROMPT" in json.dumps(client.requests)
    assert "SECRET" not in json.dumps(client.requests)
    # Missing slot result resumes through capped cache, without another API call.
    next((tmp_path / "requests/abstract").glob("*/result.json")).unlink()
    assert again.request(*args, visible=page()) == raw and len(client.requests) == 1
    assert read(tmp_path / "counter.json")["attempts"] == 1
    assert read(tmp_path / "counter.json")["identity"]["cap"] == teacher_cap(1)
    with pytest.raises(ValueError):
        again.request(*args[:-1], "changed")
    with pytest.raises(ValueError):
        again.request(["test"], "abstract", {**task, "split": "test", "sealed": True}, "no")


def test_failed_transport_stays_halted_and_sanitized(tmp_path, task):
    task = {**task, "source_split": "train", "parent_seed_ids": ["seed"]}
    client = RawClient(error=TimeoutError("SECRET_SDK_EXCEPTION"))
    config, identity = {"models": {"teacher": {"sampling": {}}}}, {"unique_instances": 1}
    for i in range(2):
        budget = Budget(tmp_path, config, identity, factory=budget_factory(client))
        with pytest.raises((CollectorHalted, RuntimeError)):
            budget.request(["run", i], "manager", task, "act")
    assert len(client.requests) == 1
    assert all("SECRET_SDK_EXCEPTION" not in p.read_text() for p in tmp_path.rglob("*.json*"))


class FakeBudget:
    def __init__(self, *, always_fail=False):
        self.requests = []
        self.always_fail = always_fail

    def request(self, slot, role, task, prompt, history=(), visible=None):
        self.requests.append((slot, role, task, prompt, history, visible))
        if role == "manager":
            return 'Action: create_mcp {"request":"Look once"}' if slot[-1] == 0 else "Action: look"
        if role == "create":
            return json.dumps({"tool": tool(), "arguments": {"command": "look"}})
        if role == "abstract":
            return json.dumps(tool())
        if role == "cluster":
            inputs = json.loads(prompt.split("Input code:\n")[1])
            return json.dumps({"clusters": [{"name": "observe", "members": [r["id"] for r in inputs]}]})
        if role == "consolidate":
            return json.dumps(tool(name="observe"))
        raise AssertionError(role)


@pytest.mark.parametrize("win_at,expected_runs,expected_tools", [(1, 1, 1), (99, 3, 0)])
def test_collection_success_filter_three_run_cap_and_resume(seed_bank, tmp_path, win_at, expected_runs, expected_tools):
    root, task, config = first_task(seed_bank)
    cell, output = tmp_path / "cell", tmp_path / "collected"
    identity = {"unique_instances": 1, "K": 1}
    write_json(cell / "identity.json", identity)
    write_json(cell / "tasks.json", [task["task_id"]])
    budget, resets = FakeBudget(), []
    factory = env_factory(root, win_at, resets)
    collect([task], config, cell, output, identity, budget, env_factory=factory)
    assert len(resets) == expected_runs
    pool = checked_pool(output / "pool.json", [task])
    assert len(pool) == expected_tools
    assert len(read(output / "runs.json")) == expected_runs
    calls = len(budget.requests)
    collect([task], config, cell, tmp_path / "resume", identity, budget, env_factory=factory)
    assert len(resets) == expected_runs and len(budget.requests) == calls
    if pool:
        path = Path(pool[0]["source_directory"]) / "episode/episode.json"
        episode = read(path)
        episode["success"] = False
        write_json(path, episode)
        with pytest.raises(ValueError, match="evidence changed"):
            checked_pool(output / "pool.json", [task])


def test_interrupted_reservations_consume_slots_without_fake_restore(seed_bank, tmp_path):
    root, task, config = first_task(seed_bank)
    cell, identity = tmp_path / "cell", {"unique_instances": 1}
    write_json(cell / "identity.json", identity)
    write_json(cell / "tasks.json", [])
    for index in range(3):
        write_json(cell / "runs" / digest(task["task_id"]) / f"run-{index}/reservation.json",
            {"cell": digest(identity), "task_id": task["task_id"], "run": index, "parent_seed_ids": task["parent_seed_ids"]})
    budget = FakeBudget()
    collect([task], config, cell, tmp_path / "out", identity, budget,
            env_factory=lambda *_: pytest.fail("Must not reset interrupted slots"))
    assert not budget.requests and read(tmp_path / "out/pool.json") == []
    assert len(read(tmp_path / "out/runs.json")) == 3


def test_all_three_box_stages_and_one_tool_per_cluster(seed_bank, tmp_path):
    root, task, config = first_task(seed_bank)
    cell, identity, budget = tmp_path / "cell", {"unique_instances": 1}, FakeBudget()
    write_json(cell / "identity.json", identity)
    write_json(cell / "tasks.json", [])
    collect([task], config, cell, tmp_path / "collect", identity, budget, env_factory=env_factory(root, 1))
    source = tmp_path / "collect"
    for kind in ("abstract", "cluster", "consolidate"):
        dest = tmp_path / kind
        build(kind, source, dest, [task], budget, config)
        source = dest
    box = load_box(source / "box.json", file_hash(source / "box.json"))
    assert len(box["tools"]) == 1 and box["tools"][0]["name"] == "observe"
    assert [r[1] for r in budget.requests] == ["manager", "create", "abstract", "cluster", "consolidate"]
    assert "SECRET" not in json.dumps([(r[3], r[4], r[5]) for r in budget.requests])
    assert box["label"] == "different interface" and box["training_exposure"] == 0


@pytest.mark.parametrize("clusters", [[], [{"name": "x", "members": ["a", "a"]}],
    [{"name": "x", "members": ["b"]}], [{"name": "x", "members": []}]])
def test_bad_cluster_partition_rejected(clusters):
    with pytest.raises(Rejected):
        parse_clusters(json.dumps({"clusters": clusters}), ["a"])


def test_pipeline_stage_isolation_and_cpu_allocation(tmp_path, monkeypatch):
    import baselines.agentdistill.pipeline as p
    args = NS(output=str(tmp_path), through="abstract")
    monkeypatch.setattr(p, "tasks_for", lambda *_: [])
    monkeypatch.setattr(p, "Budget", lambda *_: pytest.fail("No teacher before prerequisites"))
    with pytest.raises(ValueError, match="prerequisite"):
        p.run_stages(args, {}, {})
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    with pytest.raises(ValueError, match="GPU allocation"):
        p.require_cpu()


def test_verified_support_prepare_and_stage_publication(seed_bank, tmp_path, monkeypatch):
    from teachers.collect_seeds import collect as collect_seeds
    from sed.seed_selection import select_seeds
    from sed.seed_evidence import materialize_subsets
    import baselines.agentdistill.pipeline as p
    root, manifest, freeze, config = seed_bank
    env, teacher = factories(root, config, set(), [])
    verified = tmp_path / "verified"
    collect_seeds(select_seeds(manifest, freeze, k=10), manifest, freeze, config, verified,
                  env_factory=env, teacher_factory=teacher)
    materialize_subsets(verified, manifest, freeze, [5])
    verified = verified / "subsets/k5"
    config["environment"]["max_episode_steps"] = 40
    for key, value in (("manifest", manifest), ("split_freeze", freeze)):
        path = tmp_path / (key + ".json")
        write_json(path, value)
        config["paths"][key] = str(path)
    snapshot = tmp_path / "snapshot"
    write_json(snapshot / "config.json", {"model_type": "gemma4_unified"})
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    args = NS(output=str(tmp_path / "cell"), verified=str(verified), snapshot=str(snapshot),
              config=str(config_path), gpus=1, through="collect")
    identity = prepare(args, config, method="agentdistill", recipe=p.RECIPE)
    tasks = tasks_for(args.output, config)
    assert len(tasks) == identity["K"] == 5 and all(t["parent_seed_ids"] for t in tasks)
    budget = FakeBudget()
    monkeypatch.setattr(p, "Budget", lambda *_: budget)
    original_collect = p.collect
    monkeypatch.setattr(p, "collect", lambda *a: original_collect(*a, env_factory=env_factory(root, 1)))
    for stage in p.STAGES:
        args.through = stage
        p.run_stages(args, identity, config)
    models = read(Path(args.output) / "models.json")
    assert models[0]["checkpoint"] == str(snapshot) and models[0]["training_exposure"] == 0
    assert models[0]["label"] == "different interface" and models[0]["equivalence_required"]
    count = len(budget.requests)
    p.run_stages(args, identity, config)
    assert len(budget.requests) == count
    # Evaluation consumes the publication and starts HF -> vLLM gate -> dev140,
    # without constructing a teacher or re-entering generation.
    args.env_python = args.train_python = args.vllm_python = "python"
    args.workers = 1
    commands = []
    @contextmanager
    def served(args, attempt):
        commands.append("server")
        yield "http://127.0.0.1:123/v1"
    def command(argv, log, **kwargs):
        commands.append(argv)
        dest = Path(argv[argv.index("--output") + 1])
        write_json(dest / "stub.json", {"ok": True})
    monkeypatch.setattr(p, "serve_base", served)
    monkeypatch.setattr(p, "run_command", command)
    monkeypatch.setattr(p, "Budget", lambda *_: pytest.fail("Evaluation must not instantiate teacher"))
    p.evaluate(args, config)
    assert "--hf-only" in commands[0] and commands[1] == "server"
    assert "--hf-reference" in commands[2]
    assert "baselines.agentdistill.evaluation" in commands[3]
    assert "data/manifest/dev140.json" in commands[3]
    write_json(models[0]["box"], {})
    with pytest.raises(ValueError, match="artifacts changed"):
        p.run_stages(args, identity, config)


def test_gate_checks_base_weights_server_box_and_actual_scores(tmp_path):
    from baselines.agentdistill.gate import validate_report
    from sed.seed_evidence import artifact_hashes
    from tools.check_full_serving import compare
    snapshot = tmp_path / "snapshot"
    write_json(snapshot / "config.json", {"model_type": "gemma4_unified"})
    box_path = tmp_path / "box.json"
    write_json(box_path, {"label": "different interface", "training_exposure": 0, "tools": [tool()]})
    sha = file_hash(box_path)
    cases = [{"input_ids": [1, 2, 106], "input_sha256": digest([1, 2, 106]), "prompt_tokens": 1,
        "source": {"training_eligible": True, "task_id": "train/a", "turn_index": i},
        "hf_logprobs": [-1., -2.], "vllm_logprobs": [-1., -2.]} for i in range(8)]
    student = {"requested_repo": "google/gemma-4-12B-it", "revision": "707f0a3b"}
    endpoint = "http://127.0.0.1:123/v1"
    identity = {"snapshot": str(snapshot), "snapshot_artifact_sha256": artifact_hashes(snapshot),
        "student": student, "box_sha256": sha, "prompts_sha256": digest([
            {k: v for k, v in row.items() if k not in {"hf_logprobs", "vllm_logprobs"}} for row in cases])}
    server = {"config": {"models": {"student": student}, "server": {
        "endpoint": endpoint, "model_path": str(snapshot), "checkpoint_kind": "base", "merged_model": False,
        "lora_adapters": [], "batch_invariant": True, "served_model_name": student["requested_repo"]}}}
    report = {**compare(cases), "identity": identity, "prompts": cases,
        "server_provenance_sha256": digest(server), "endpoint": endpoint}
    validate_report(report, server, endpoint, box_path, sha)
    bad = deepcopy(report)
    bad["prompts"][0]["vllm_logprobs"] = [-9., -9.]
    with pytest.raises(ValueError, match="gate failed"):
        validate_report(bad, server, endpoint, box_path, sha)
    bad = deepcopy(report)
    bad["prompts"][0]["source"]["task_id"] = "test/a"
    with pytest.raises(ValueError):
        validate_report(bad, server, endpoint, box_path, sha)
    write_json(snapshot / "weights.json", {"changed": True})
    with pytest.raises(ValueError, match="identity changed"):
        validate_report(report, server, endpoint, box_path, sha)


def test_invalid_consolidation_blocks_publication(tmp_path, task):
    source = tmp_path / "cluster"
    row = {"id": "a", "tool": tool(), "task_id": task["task_id"]}
    write_json(source / "abstracted.json", [row])
    write_json(source / "clusters.json", [{"name": "x", "members": ["a"]}])
    budget = NS(request=lambda *_: json.dumps(tool("def move(env, command):\n    return open(command)")))
    with pytest.raises(Rejected, match="publication refused"):
        build("consolidate", source, tmp_path / "output", [task], budget, {})
    assert not (tmp_path / "output/box.json").exists()
    assert read(tmp_path / "output/rejections.json")[0]["reason"] == "invalid_consolidation"


def test_full_train_is_recognized_by_registered_rule(data_root, tmp_path):
    from sed.config import load_config
    from sed.manifest import build_manifest, freeze_splits
    from sed.seed_selection import select_seeds, ALL_TRAIN_RULE
    from teachers.collect_seeds import collect as collect_seeds
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=1)
    env, teacher = factories(data_root, config, set(), [])
    verified = tmp_path / "full"
    result = collect_seeds(select_seeds(manifest, freeze, rule=ALL_TRAIN_RULE), manifest, freeze,
        config, verified, env_factory=env, teacher_factory=teacher)
    assert result["K"] == 4  # Not in registered few-shot K; identified by its rule.
    for key, value in (("manifest", manifest), ("split_freeze", freeze)):
        config["paths"][key] = str(tmp_path / (key + ".json"))
        write_json(config["paths"][key], value)
    snapshot = tmp_path / "snapshot"
    write_json(snapshot / "config.json", {})
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    args = NS(output=str(tmp_path / "cell"), verified=str(verified), snapshot=str(snapshot),
              config=str(config_path), gpus=1)
    identity = prepare(args, config, method="agentdistill", recipe={})
    assert identity["K"] == identity["unique_instances"] == 4 and identity["task_groups"] == 2
    assert len(tasks_for(args.output, config)) == 4


def test_student_sampling_is_frozen_and_thinking_off(tmp_path):
    client = BoxStudent(tools=[tool()], endpoint="http://127.0.0.1:123/v1", model="google/gemma-4-12B-it",
        revision="707f0a3b", enable_thinking=False, log_dir=tmp_path)
    assert client.request_sampling([], {}) == {"temperature": 0, "seed": 0}
    assert client.spec["chat_template_kwargs"] == {"enable_thinking": False}
    assert client.spec["max_tokens"] == 1024
