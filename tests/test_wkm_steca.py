"""Offline algorithm, source-integrity, resume and dual-adapter integration tests."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from baselines.common.conversation import episode_messages
from baselines.common.method_data import read
from baselines.steca.data import deviation, ndtw, build_datasets, reflection_prompt, calibrated_reply, ReflectionTeacher
from baselines.wkm.format import agent_messages, world_prompt, task_synthesis
from baselines.wkm.retrieval import mix, KnowledgeBase
from baselines.wkm.policy import WKMStudent
from sed.storage import digest, write_json
from teachers.openai_teacher import TeacherResult
from train.dataset import encode_turn

def trajectory():
    visible = {"observation": "room", "admissible_commands": ["look"]}
    return {"schema_version": 2, "task_id": "train/a/1", "split": "train", "sealed": False,
        "initial": {"visible": dict(visible, facts="PRIVATE_PDDL")},
        "steps": [{"visible": visible, "event_id": str(i), "action_raw": f"Thought: step {i}\nAction: look",
                   "audit": "SECRET", "action_outcome": "legal"} for i in range(2)]}


@pytest.fixture
def tokenizer():
    pytest.importorskip("transformers")
    from train.dataset import load_tokenizer
    path = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
    if not path.exists():
        pytest.skip("Pinned tokenizer not installed")
    return load_tokenizer(path)

from test_stage2b_seeds import seed_bank, factories


def test_wkm_upstream_prompts_and_h0_allowlist():
    import ast
    from baselines.wkm import prompts
    source = Path("third_party/WKM/src/prompts/knowledge_template.py")
    if source.exists():
        scope = {}
        nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.Assign)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "upstream_constants", "exec"), scope)
        assert prompts.TASK_KNOWLEDGE == scope["TASK_KNOWLEDGE"]
        assert prompts.ALFWORLD_STATE == scope["ALFWORLD_STATE"]
    messages = episode_messages(trajectory())
    changed = agent_messages(messages, "Task Knowledge: inspect first")
    assert changed[0] == messages[0] and changed[2:] == messages[2:]
    assert json.loads(changed[1]["content"])["admissible_commands"] == ["look"]
    assert "PRIVATE_PDDL" not in json.dumps(changed) and "SECRET" not in json.dumps(changed)
    assert "No rejected trajectory" in task_synthesis(messages, None)[0]["content"]
    assert world_prompt(messages[:2])[-1] == messages[1]
    assert world_prompt(messages[:4], "task", ["initial"])[-1]["content"] == messages[2]["content"] + "\n" + messages[3]["content"]


def test_wkm_gamma_full_commands_and_previous_action_filter(tmp_path):
    actions = ["go to table 1", "go to fridge 1"]
    selected, agent, mixed = mix(actions, dict(zip(actions, [math.log(.9), math.log(.1)])),
                                  dict(zip(actions, [0., 1.])))
    assert selected == actions[1] and mixed[actions[1]] == pytest.approx(.64)
    assert mix(actions, dict(zip(actions, [0., -1.])), {})[0] == actions[0]
    class Encoder:
        identity = {"model": "stub"}
        def __call__(self, texts):
            return [[1., 0.] for _ in texts]
    payload = {"triples": [{"state": "s", "pre_action": "look", "action": actions[1]},
                           {"state": "s", "pre_action": "take apple", "action": actions[0]}],
               "vectors": [[1., 0.], [1., 0.]], "embedding": Encoder.identity, "gamma": .4, "top_n": 3000}
    file = tmp_path / "kb.json"
    write_json(file, {**payload, "sha256": digest(payload)})
    kb = KnowledgeBase(file, Encoder())
    probabilities, ids = kb.probabilities("s", actions, "look")
    assert probabilities == dict(zip(actions, [0., 1.])) and len(ids) == 2
    assert kb.probabilities("s", actions, "open fridge")[0] == {}
    payload["vectors"][0] = [0., 1.]
    write_json(file, {**payload, "sha256": "wrong"})
    with pytest.raises(ValueError, match="changed"):
        KnowledgeBase(file, Encoder())


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_two_lora_policy_reuses_task_knowledge_and_shared_h0(tmp_path, finish):
    from baselines.wkm.data import PromptStudent
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        text = ("Task Knowledge: inspect" if body["max_tokens"] == 512 else "State Knowledge: at room"
                if body["model"] == "world" else "Thought: inspect\nAction: look")
        return httpx.Response(200, json={"model": body["model"], "choices": [{
            "message": {"content": text}, "finish_reason": finish if body["model"] == "world" else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4}})
    class KB:
        identity = {"sha256": "stub"}
        def probabilities(self, state, commands, previous):
            return {"look": 0., "go north": 1.}, [0]
    policy = WKMStudent(endpoint="http://fake/v1", model="agent", revision="pin", enable_thinking=False,
        knowledge_model="world", knowledge_base="unused", log_dir=tmp_path, kb=KB(), student_factory=PromptStudent,
        transport=httpx.MockTransport(handler),
        scorer=lambda messages, commands, thought: {"look": math.log(.9), "go north": math.log(.1)})
    visible = {"observation": "room", "admissible_commands": ["look", "go north"], "facts": "ORACLE"}
    task = {"task_id": "train/a", "extra.expert_plan": "ORACLE"}
    result = policy.generate([], task, visible)
    assert result.raw_text.endswith("Action: go north")
    assert result.metadata["prompt_tokens"] == 30 and result.metadata["completion_tokens"] == 12
    policy.generate([{"visible": visible, "reply": result.raw_text}], task, visible)
    # Reusing the world model on another episode must clear its newline stop.
    policy.generate([], {**task, "task_id": "train/b"}, visible)
    policy.close()
    assert [c["model"] for c in calls] == ["world", "world", "agent", "world", "agent", "world", "world", "agent"]
    for c in calls:
        if c["model"] == "world" and c["max_tokens"] != 512:
            assert c["max_tokens"] == 128 and c["stop"] == ["\n"]
        else:
            assert "stop" not in c
            assert c["max_tokens"] == (512 if c["model"] == "world" else 1024)
    assert "ORACLE" not in json.dumps(calls)
    assert calls[2]["messages"][0]["content"] == episode_messages(trajectory())[0]["content"]
    records = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert [len(r["knowledge_calls"]) for r in records] == [2, 1, 2]
    for record in records:
        for call in record["knowledge_calls"]:
            assert call["finish_reason"] == finish
            assert call["prompt_tokens"] == 10 and call["completion_tokens"] == 4


def test_deviation_strict_preceding_reward_and_ndtw():
    assert deviation(.4, .6) and not deviation(.4, .4) and not deviation(.8, .6)
    with pytest.raises(ValueError):
        deviation(None, .6)
    vectors = {"a": [1., 0.], "b": [0., 1.]}
    encode = lambda texts: [vectors[t] for t in texts]
    assert ndtw(["a"], ["b"], encode) == pytest.approx(1/math.sqrt(2))
    assert ndtw(["a", "a"], ["a"], encode) == 0
    with pytest.raises(ValueError):
        ndtw([], ["a"], encode)


def planned_rows():
    messages = episode_messages(trajectory())
    prompt = messages[:4]
    failure = [{"role": "assistant", "content": "Thought: wrong\nAction: go north"}]
    common = {"task_id": "train/a/1", "parent_seed_id": "seed", "episode_id": "explore", "j": 1,
        "prompt": prompt, "expert": messages[4:], "explored": failure, "full_explored": prompt + failure,
        "full_expert": messages, "success": False, "deviated": True}
    success = {**common, "j": 0, "episode_id": "success", "prompt": messages[:2], "success": True,
               "deviated": False, "explored": messages[2:], "full_explored": messages}
    return [common, success]


def test_steca_datasets_rewards_reflection_prompt_and_masks(tokenizer):
    rows = planned_rows()
    raw = "Thought: avoid the mistaken move.\nAction: go north"
    reply = calibrated_reply(raw, "look")
    reflections = [{"task_id": rows[0]["task_id"], "j": 1, "episode_id": "explore", "raw_output": raw, "reply": reply}]
    datasets = build_datasets(rows, reflections, lambda texts: [[1., 0.] if "wrong" not in t else [0., 1.] for t in texts])
    assert {k: len(v) for k, v in datasets.items()} == {"Dc": 1, "Ds": 1, "De": 1}
    assert datasets["Dc"][0]["reward"] == pytest.approx(1 + .01/math.sqrt(2))
    assert datasets["Ds"][0]["reward"] > 1 and datasets["De"][0]["reward"] == 1
    assert datasets["Dc"][0]["target"][0]["content"].endswith("Action: look")
    assert "ground truth action is: look" in reflection_prompt(rows[0])
    from baselines.common.conversation import encode_continuation
    example, _ = encode_continuation(rows[0]["prompt"], datasets["Dc"][0]["target"], tokenizer,
        task_id=rows[0]["task_id"], source="trusted_teacher_seed")
    assert len(example["turns"]) == 1
    turn = example["turns"][0]
    supervised = tokenizer.decode([v for v in turn["labels"] if v != -100])
    assert supervised == reply + "<turn|>" and "step 0" not in supervised
    assert "PRIVATE_PDDL" not in tokenizer.decode(turn["input_ids"])


def test_capped_reflection_is_text_only_one_attempt_and_cached(tmp_path):
    from teachers.capped import CappedTeacher, BudgetExceeded
    calls = []
    class Raw:
        status_code = 200
        def parse(self):
            return {"output_text": "Thought: correct the mistake.", "model": "gpt-6-luna", "status": "completed",
                    "id": "fake", "usage": {"input_tokens": 12, "output_tokens": 5}}
    def create(**kwargs):
        calls.append(kwargs)
        return Raw()
    client = SimpleNamespace(max_retries=0, responses=SimpleNamespace(with_raw_response=SimpleNamespace(create=create)))
    kwargs = dict(counter_path=tmp_path / "counter.json", cap=1, cache_dir=tmp_path / "cache",
        usage_ledger=tmp_path / "usage.jsonl", run_id="cell", teacher_class=ReflectionTeacher, client=client)
    teacher = CappedTeacher(**kwargs)
    task = {"task_id": "train/a", "split": "train", "sealed": False}
    visible = {"observation": reflection_prompt(planned_rows()[0]), "admissible_commands": []}
    teacher.generate([], task, visible)
    CappedTeacher(**kwargs).generate([], task, visible)
    assert len(calls) == 1 and "logprobs" not in calls[0]
    with pytest.raises(BudgetExceeded):
        teacher.generate([], task, {**visible, "observation": "another"})
    assert read(tmp_path / "counter.json")["attempts"] == 1


def test_dual_gate_rejects_missing_world_report(tmp_path, monkeypatch):
    from baselines.wkm.evaluation import validate_dual
    config = {"models": {"student": {"requested_repo": "gemma", "revision": "pin"}},
        "server_provenance": {"config": {"server": {"batch_invariant": True}}},
        "lora_equivalence": {"identity": {"adapter": {"base_model": "gemma", "revision": "pin"}}}}
    spec = {"model": "agent", "knowledge_model": "world", "endpoint": "fake", "knowledge_base": str(tmp_path / "kb")}
    monkeypatch.setattr("baselines.wkm.evaluation.validate_report", lambda *a: None)
    with pytest.raises(ValueError, match="BOTH"):
        validate_dual(config, spec)
    payload = {"triples": [], "vectors": []}
    write_json(spec["knowledge_base"], {**payload, "sha256": digest(payload)})
    config["knowledge_lora_equivalence"] = deepcopy(config["lora_equivalence"])
    validate_dual(config, spec)
    config["knowledge_lora_equivalence"]["identity"]["adapter"]["revision"] = "other"
    with pytest.raises(ValueError, match="backbone"):
        validate_dual(config, spec)


def test_steca_detection_uses_previous_expert_and_first_drop(monkeypatch, tmp_path):
    from baselines.steca import data
    episode = trajectory()
    episode["episode_id"] = "seed"
    episode["steps"].append(deepcopy(episode["steps"][-1]))
    task = {"task_id": episode["task_id"]}
    paths = [tmp_path / name for name in ("explore", "expert", "agent")]
    serving = {"model_path": "sft", "training_provenance_sha256": "hash"}
    for path in paths:
        path.mkdir()
        write_json(path / "provenance.json", {"details": {"verified": "verified", "exploration": str(paths[0])},
            "config": {"server_provenance": {"config": {"server": serving}}}})
    explored = [({"task_id": task["task_id"], "j": j, "success": False},
                 {**episode, "episode_id": "explore-" + str(j)}) for j in range(3)]
    monkeypatch.setattr(data, "seeds", lambda *a: [(task, episode)])
    monkeypatch.setattr(data, "checked_rollouts", lambda path, *a, **k: explored if Path(path) == paths[0] else Path(path).name)
    monkeypatch.setattr(data, "values", lambda kind: {(task["task_id"], j): v for j, v in enumerate(
        [.8, .2, 1.] if kind == "expert" else [.1, .6, 0.])})
    planned = data.plan("verified", *paths, {})
    assert [r["deviated"] for r in planned] == [False, True, False]
    assert planned[1]["preceding_expert_mc"] == .8  # same-index .2 would incorrectly accept
    assert planned[2]["decision"] == "later_deviation"


def test_wkm_stages_resume_commands_and_all_checkpoints(tmp_path, monkeypatch):
    from baselines.wkm import pipeline as p
    commands, rollouts = [], []
    def run(command, log, **kw):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        if "baselines.common.fsdp_train" in command:
            (output / "checkpoint").mkdir()
            write_json(output / "result.json", {"examples_seen": 5})
        else:
            write_json(output / "knowledge_base.json", {})
    def rollout(args, checkpoint, attempt, **kw):
        rollouts.append(kw)
        (attempt / "output").mkdir()
        write_json(attempt / "output/summary.json", {})
    def merge(*args):
        out = args[3]
        out.mkdir()
        for name in ("world", "agent", "data"):
            write_json(out / (name + ".json"), {})
    monkeypatch.setattr(p, "run_command", run)
    monkeypatch.setattr(p, "student_stage", rollout)
    monkeypatch.setattr(p, "merge", merge)
    args = SimpleNamespace(output=str(tmp_path), config="config", verified="verified", snapshot="base", through="all",
        train_python="python", embedding_snapshot="embedding", gpus=1)
    p.train_pipeline(args, {"method": "wkm"}, {})
    train = [c for c in commands if "baselines.common.fsdp_train" in c]
    assert len(train) == 3 and len(rollouts) == 2
    assert all(c[c.index("--snapshot")+1] == "base" for c in train)
    assert all(c[c.index("--adaptation")+1] == "wkm_lora" for c in train)
    models = read(tmp_path / "models.json")
    assert [m["name"] for m in models] == ["experienced", "world", "wkm"]
    assert models[-1]["adapter"] != models[-1]["knowledge_adapter"]
    count = len(commands)
    p.train_pipeline(args, {"method": "wkm"}, {})
    assert len(commands) == count
    write_json(tmp_path / "stages/merge/attempt-000/output/data.json", {"tampered": True})
    with pytest.raises(ValueError, match="artifacts changed"):
        p.train_pipeline(args, {"method": "wkm"}, {})


@pytest.mark.parametrize("empty", [False, True])
def test_steca_stage_split_and_single_epoch(tmp_path, monkeypatch, empty):
    from baselines.steca import pipeline as p
    commands, rollouts, reflections = [], [], []
    def run(command, log, **kw):
        commands.append(command)
        out = Path(command[command.index("--output") + 1])
        out.mkdir(parents=True)
        if "baselines.common.fsdp_train" in command:
            (out / "checkpoint").mkdir()
        else:
            write_json(out / "data.json", [] if empty else [{}])
            write_json(out / "provenance.json", {"details": {"counts": {"Dc": 0}}})
    def rollout(args, checkpoint, attempt, **kw):
        rollouts.append((checkpoint, kw))
        (attempt / "output").mkdir()
    def reflect(planned, config, verified, output, ledger):
        reflections.append(ledger)
        output.mkdir()
        write_json(output / "reflections.json", [])
    monkeypatch.setattr(p, "run_command", run)
    monkeypatch.setattr(p, "rollout_stage", rollout)
    monkeypatch.setattr(p, "plan", lambda *a: [])
    monkeypatch.setattr(p, "reflect", reflect)
    args = SimpleNamespace(output=str(tmp_path), config="config", verified="verified", snapshot="base", through="reflect",
        train_python="python", embedding_snapshot="embedding", gpus=4)
    with pytest.raises(ValueError, match="CPU stage requires"):
        p.train_pipeline(args, {}, {})
    assert not commands
    args.through = "agent-mc"
    p.train_pipeline(args, {}, {})
    assert len(commands) == 1 and len(rollouts) == 3 and not reflections
    args.through = "reflect"
    p.train_pipeline(args, {}, {})
    assert len(commands) == 1 and len(reflections) == 1
    args.through = "all"
    p.train_pipeline(args, {}, {})
    training = [c for c in commands if "baselines.common.fsdp_train" in c]
    assert len(training) == (1 if empty else 2)
    if not empty:
        assert training[-1][training[-1].index("--epochs")+1] == "1"
        assert training[-1][training[-1].index("--objective")+1] == "steca"
    assert [m["name"] for m in read(tmp_path / "models.json")] == ["sft", "reinforced"]
    assert all(m["full_equivalence_required"] for m in read(tmp_path / "models.json"))


@pytest.fixture
def wkm_sources(seed_bank, tmp_path):
    from teachers.collect_seeds import collect
    from sed.seed_selection import select_seeds
    from baselines.common.explore import run_exploration, explore_one
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    from sed.runner import ScriptedPolicy
    root, manifest, freeze, config = seed_bank
    config["paths"].update(manifest=str(tmp_path / "m.json"), split_freeze=str(tmp_path / "f.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    env, teacher = factories(root, config, set(), [])
    verified = tmp_path / "verified"
    collection = collect(select_seeds(manifest, freeze, k=5), manifest, freeze, config, verified,
                         env_factory=env, teacher_factory=teacher)
    config["environment"]["max_episode_steps"] = 40
    ids = [r["task_id"] for r in collection["verified"]]
    from baselines.ipr.rollouts import seeds as checked_seeds
    parents = {t["task_id"]: [e["episode_id"]] for t, e in checked_seeds(verified, config)}
    taskfile = tmp_path / "tasks.json"
    write_json(taskfile, {"task_ids": ids, "list_sha256": digest(ids), "manifest_sha256": digest(manifest),
        "split_freeze_sha256": digest(freeze), "parent_seed_ids": parents})
    def worker(job):
        def backend(path, seed):
            value = FakeBackend(path, seed)
            value.win_at = 99  # known failure, actual 40-turn execution and replay
            return value
        return explore_one(job, env_factory=lambda c,t: Environment(root, max_episode_steps=40, backend_factory=backend),
                           policy_factory=lambda **kw: ScriptedPolicy(["look"]*40))
    explored = tmp_path / "explore"
    run_exploration(config=config, tasks_path=taskfile, output=explored,
        spec={"enable_thinking": False, "max_tokens": 1024}, decoding="greedy", worker_fn=worker)
    return verified, explored, config


def test_wkm_real_fake_environment_merge_and_tamper(wkm_sources, tmp_path, tokenizer):
    from baselines.wkm.data import synthesize, merge, training_examples
    verified, explored, config = wkm_sources
    class Stub:
        spec = {}
        def generate(self, history, task, visible):
            text = "State Knowledge: room" if self.spec["max_tokens"] == 128 else "Task Knowledge: inspect"
            return TeacherResult(text, None, {"finish_reason": "stop", "prompt_tokens": 4, "completion_tokens": 4})
    synthesis = tmp_path / "synthesis"
    knowledge = synthesize(verified, explored, synthesis, config, Stub())
    assert len(knowledge) == 5 and all(r["rejected"] for r in knowledge)
    output = tmp_path / "merged"
    merge(verified, explored, synthesis, output, config)
    for role in ("agent", "world"):
        examples = training_examples(output / (role + ".json"), config, tokenizer, verified=verified)
        assert len(examples) == 5 and all(e["packing"] == "turn" for e in examples)
        assert "SECRET" not in json.dumps(examples)
    rows = read(output / "data.json")
    rows[0]["task_knowledge"] = "tamper"
    write_json(output / "data.json", rows)
    with pytest.raises(ValueError, match="changed"):
        training_examples(output / "agent.json", config, tokenizer, verified=verified)


def test_steca_checked_environment_to_reflection_and_training(seed_bank, tmp_path, tokenizer):
    from baselines.ipr.rollouts import run_rollouts, rollout_one
    from baselines.steca.data import plan, reflect, merge, training_examples
    from teachers.collect_seeds import collect
    from teachers.capped import from_config, CappedTeacher
    from sed.seed_selection import select_seeds
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    from sed.runner import ScriptedPolicy
    root, manifest, freeze, config = seed_bank
    config["paths"].update(manifest=str(tmp_path / "m.json"), split_freeze=str(tmp_path / "f.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    config["environment"]["max_episode_steps"] = 40
    class Backend(FakeBackend):
        def __init__(self, path, seed):
            super().__init__(path, seed)
            self.win_at, self.dead = 3, False
        def state(self):
            return {**super().state(), "lost": self.dead}
        def step(self, action):
            if action == "go north":
                self.dead = True
                return self.state(), 0., True
            return super().step(action)
    def environment(*unused):
        return Environment(root, max_episode_steps=40, backend_factory=Backend)
    _, teacher = factories(root, config, set(), [])
    verified = tmp_path / "verified"
    collect(select_seeds(manifest, freeze, k=5), manifest, freeze, config, verified,
            env_factory=environment, teacher_factory=teacher)
    server = tmp_path / "server.json"
    write_json(server, {"config": {"server": {"batch_invariant": True, "model_path": "sft",
        "training_provenance_sha256": "sft-hash"}}})
    def policy(action):
        value = ScriptedPolicy([action]*40)
        value.spec = {"test": True}
        value.close = lambda: None
        return value
    def worker(job):
        return rollout_one(job, env_factory=environment,
            student_factory=lambda **kw: policy("go north"),
            mc_factory=lambda **kw: policy("look"))
    paths = {kind: tmp_path / kind for kind in ("expert-mc", "explore", "agent-mc")}
    for kind, out in paths.items():
        run_rollouts(config=config, verified=verified, kind=kind, iteration=0 if kind == "expert-mc" else 1,
            output=out, spec={"model": config["models"]["student"]["requested_repo"],
                "revision": config["models"]["student"]["revision"], "enable_thinking": False,
                "max_tokens": 1024 if kind == "explore" else 512},
            server_provenance=server, exploration=paths["explore"] if kind == "agent-mc" else None,
            workers=1, worker_fn=worker)
    planned = plan(verified, paths["explore"], paths["expert-mc"], paths["agent-mc"], config)
    assert len(planned) == 15
    assert sum(r["deviated"] for r in planned) == 5
    assert all(r["j"] == 1 for r in planned if r["deviated"])
    planfile = tmp_path / "plan.json"
    write_json(planfile, planned)
    factory_calls = []
    class FakeTeacher:
        def generate(self, history, task, visible):
            factory_calls.append(visible)
            return TeacherResult("Thought: avoid that direction.\nAction: look", None,
                {"finish_reason": "stop", "call_id": "fake"})
    def factory(*args, **kwargs):
        assert kwargs["cap"] == 5 and kwargs["teacher_class"] is ReflectionTeacher
        return FakeTeacher()
    reflected = tmp_path / "reflection"
    reflect(planned, config, verified, reflected, tmp_path / "ledger", teacher_factory=factory)
    assert len(factory_calls) == 5
    assert "SECRET" not in json.dumps(factory_calls)
    class Encode:
        identity = {"model": "fake-bert"}
        def __call__(self, texts):
            return [[1., 0.] if "go north" in t else [0., 1.] for t in texts]
    output = tmp_path / "dataset"
    sources = [planfile, *[paths[k] / f for k in ("explore", "expert-mc", "agent-mc")
                          for f in ("provenance.json", "receipts.json", "plan.json")]]
    n = merge(planned, reflected / "reflections.json", output, config, verified, sources, Encode())
    assert n == 20 and len(read(output / "Dc.json")) == 5 and len(read(output / "Ds.json")) == 15
    encoded = training_examples(output / "data.json", config, tokenizer, verified=verified)
    assert len(encoded) == 20
    assert {row["dataset_scale"] for row in encoded} == {4., 20/15}
    assert encoded[0]["turns"][0]["label_provenance"]["target_source"] == "teacher_reflection_and_expert_action"
    assert encoded[0]["turns"][0]["label_provenance"]["trusted_seed"] is False
    raw = read(reflected / "reflections.json")
    raw[0]["raw_output"] = "tampered"
    write_json(reflected / "reflections.json", raw)
    with pytest.raises(ValueError, match="source changed"):
        training_examples(output / "data.json", config, tokenizer, verified=verified)


def test_wkm_sequence_scoring_uses_actual_full_command_logps(tmp_path):
    from baselines.wkm.data import PromptStudent
    from baselines.wkm.policy import Scorer
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if request.url.path == "/tokenize":
            if "messages" in body:
                return httpx.Response(200, json={"tokens": [2,3], "count": 2, "max_model_len": 32768})
            return httpx.Response(200, json={"tokens": [7 if "look" in body["prompt"] else 8,106]})
        ids = body["prompt"]
        return httpx.Response(200, json={"model": "agent", "choices": [{"prompt_token_ids": ids,
            "prompt_logprobs": [None] + [{str(i): {"logprob": -.1 if i in (7,106) else -2.}} for i in ids[1:]]}]})
    student = PromptStudent(endpoint="http://fake/v1", model="agent", revision="pin", enable_thinking=False,
        log_dir=tmp_path / "student", transport=httpx.MockTransport(handler))
    scorer = Scorer(student, tmp_path / "scores")
    scores = scorer(episode_messages(trajectory())[:2], ["look", "go north"], "Thought: inspect")
    assert scores == {"look": pytest.approx(-.2), "go north": pytest.approx(-2.1)}
    scoring = [r for r in calls if "prompt_logprobs" in r]
    assert len(scoring) == 2 and all(r["max_tokens"] == 0 for r in scoring)
    ledger = [json.loads(line) for line in (tmp_path / "scores/calls.jsonl").read_text().splitlines()]
    assert sum(r["status"] == "completed" for r in ledger) == len(calls)


def test_full_checkpoint_gate_statistics_and_server_binding():
    from tools.check_full_serving import compare, validate_report
    cases = [{"input_ids": [2,7,106], "prompt_tokens": 1, "input_sha256": digest([2,7,106]),
        "source": {"task_id": "train/a", "turn_index": i, "training_eligible": True},
        "hf_logprobs": [-1.,-.1], "vllm_logprobs": [-1.01,-.1]} for i in range(8)]
    stats = compare(cases)
    assert stats["passed"]
    server = {"config": {"server": {"model_path": "sft", "training_provenance_sha256": "hash",
        "endpoint": "http://fake", "batch_invariant": True, "merged_model": True,
        "checkpoint_kind": "full_parameter", "served_model_name": "student"}}}
    prompts = [{k: v for k, v in row.items() if k not in {"hf_logprobs", "vllm_logprobs"}} for row in cases]
    report = {"prompts": cases, **stats, "server_provenance_sha256": digest(server),
        "identity": {"checkpoint": "sft", "prompts_sha256": digest(prompts), "prompt_count": 8},
        "model_name": "student", "training_provenance_sha256": "hash", "endpoint": "http://fake"}
    validate_report(report, server, "http://fake")
    with pytest.raises(ValueError):
        validate_report(report, server, "http://other")
    server["config"]["server"]["endpoint"] = "http://other"
    report["server_provenance_sha256"] = digest(server)
    with pytest.raises(ValueError):
        validate_report(report, server, "http://fake")
    cases[0]["vllm_logprobs"][0] = -5.
    assert not compare(cases)["passed"]


def test_wkm_shared_suite_worker_executes_only_mixed_action(data_root, tmp_path, monkeypatch):
    import eval.run_suite as suite
    import baselines.wkm.policy as policy_module
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
    monkeypatch.setattr(suite, "_environment", lambda c,t: Environment(data_root, split=t["split"],
        max_episode_steps=40, backend_factory=Backend))
    created = []
    class Policy:
        def __init__(self, **kwargs):
            created.append(kwargs)
        def generate(self, history, task, visible):
            return TeacherResult("Thought: choose\nAction: go north", "go north", {"wkm": {"gamma": .4}})
        def close(self):
            pass
    monkeypatch.setattr(policy_module, "WKMStudent", Policy)
    spec = {"endpoint": "http://fake/v1", "model": "agent", "revision": "pin", "enable_thinking": False,
        "max_tokens": 1024, "decoding": "wkm", "knowledge_model": "world", "knowledge_base": "kb.json"}
    suite._worker((task, config, str(tmp_path / "episode"), "student", spec, []))
    episode = read(tmp_path / "episode/episode.json")
    assert episode["success"] is True and actions == ["go north"]
    assert created[0]["model"] == "agent" and created[0]["knowledge_model"] == "world"
    assert created[0]["knowledge_base"] == "kb.json" and "decoding" not in created[0]


def test_dual_adapter_evaluation_model_record_launch_and_hash_gate(tmp_path, monkeypatch):
    from baselines.eto.pipeline import evaluate_pipeline
    from sed.seed_evidence import artifact_hashes
    calls = []
    checkpoints = []
    for name in ("agent", "world"):
        path = tmp_path / name
        path.mkdir()
        (path / "weights").write_text(name)
        write_json(path / "training_state.json", {"artifact_sha256": artifact_hashes(path)})
        checkpoints.append(path)
    kb = tmp_path / "kb.json"
    write_json(kb, {})
    write_json(tmp_path / "models.json", [{"name": "wkm", "adapter": str(checkpoints[0]),
        "checkpoint": str(checkpoints[0]), "knowledge_adapter": str(checkpoints[1]), "knowledge_base": str(kb)}])
    monkeypatch.setattr("baselines.eto.pipeline.run_command", lambda command, log, **kwargs: calls.append((command, kwargs["env"])))
    args = SimpleNamespace(output=str(tmp_path), config="config", snapshot="base", verified="verified",
        env_python="env", train_python="train", vllm_python="vllm", workers=2, ready_timeout=30)
    evaluate_pipeline(args)
    env = calls[0][1]
    assert env["DECODING"] == "wkm" and env["INCLUDE_BASE"] == "0"
    assert env["DOSE_ADAPTER"].startswith("wkm=") and env["WKM_ADAPTER"].startswith("wkm-knowledge=")
    assert env["TASKS"].endswith("dev140.json") and env["WKM_KNOWLEDGE_BASE"] == str(kb)
    evaluate_pipeline(args)
    assert len(calls) == 1
    (checkpoints[1] / "weights").write_text("tamper")
    with pytest.raises(ValueError, match="Knowledge adapter changed"):
        evaluate_pipeline(args)


def test_reflection_refuses_gpu_job_before_teacher(tmp_path, monkeypatch):
    from baselines.steca.data import reflect
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")  # GPU ID zero IS an allocation
    with pytest.raises(ValueError, match="CPU-only"):
        reflect([], {}, "unused", tmp_path / "out", tmp_path / "ledger")
    assert not (tmp_path / "out").exists()


def test_stage_wrappers_no_implicit_gpu_and_keep_rank_identity():
    for method, gpus in (("wkm", 1), ("steca", 4)):
        text = Path(f"scripts/hpg_{method}.slurm").read_text()
        assert "#SBATCH --gpus" not in text
        assert f"{method.upper()}_THROUGH" in text and f"NGPUS=${{NGPUS:-{gpus}}}" in text
        assert '--gpus "$NGPUS"' in text
    shell = Path("scripts/eval_adapters.sh").read_text()
    assert '--knowledge-lora-check "$OUT/$WKM_KNOWLEDGE_NAME/check/report.json"' in shell
    assert 'if [[ "$DECODING" != wkm ]]' in shell
