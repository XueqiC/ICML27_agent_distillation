"""CoA offline protocol, capped roles, executed collection, stages and serving."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest

from baselines.coa import RECIPE
from baselines.coa.data import collect, datasets, filter_episode, training_examples, checked_record, support
from baselines.coa.protocol import blocks, encode_episode, h0_reply, messages, tool_action
from baselines.coa.teacher import MultiAgentTeacher, RoleTeacher, ROLE_PROMPT, budget, require_cpu
from baselines.coa.rollout import CoAStudent, checked_groups
from baselines.coa.rl import mixed, query_kept
from sed.storage import digest, write_json
from teachers.openai_teacher import TeacherResult
from teachers.capped import BudgetExceeded
from test_capped_teacher import RawClient, capped
from test_had import Tokenizer, page
from test_stage2b_seeds import seed_bank, factories
from train.dataset import TURN_TOKEN


def episode(n=5, reflection=True, correction=False):
    visible = page("VISIBLE_ONLY", ["look"])
    value = {"schema_version": 2, "task_id": "train/a/1", "split": "train", "sealed": False,
        "success": True, "status": "ok", "complete": True, "initial": {"visible": dict(visible, facts="SECRET")}, "steps": []}
    for i in range(n):
        reply = "<think>Inspect progress</think>\n"
        if reflection:
            reply += "<reflection>Review visible feedback</reflection>\n"
        if correction and i == 0:
            reply += "<double_check>Score: 1\nReconsider the plan</double_check>\n"
        reply += "<alfworld>Action: look</alfworld>"
        value["steps"].append({"visible": dict(visible, audit="SECRET"), "event_id": str(i),
            "action_raw": "Action: look", "action_parsed": "look", "action_outcome": "legal",
            "environment_steps": 1, "policy_metadata": {"coa_reply": reply, "audit": "SECRET"}})
    return value


def test_tags_h0_bridge_and_observation_mask():
    tokenizer, ep = Tokenizer(), episode(correction=True)
    encoded = encode_episode(ep, tokenizer)
    history = []
    for turn, step in zip(encoded["turns"], ep["steps"]):
        text = tokenizer.decode(turn["input_ids"])
        target = tokenizer.decode([v for v in turn["labels"] if v != -100])
        assert "SECRET" not in text
        assert "VISIBLE_ONLY" in text and "VISIBLE_ONLY" not in target
        assert "<observation>" not in target and target == step["policy_metadata"]["coa_reply"] + TURN_TOKEN
        assert turn["labels"][:turn["prompt_tokens"]] == [-100] * turn["prompt_tokens"]
        assert "<alfworld>Action: look</alfworld>" in target
    assert tool_action(ep["steps"][0]["policy_metadata"]["coa_reply"], ["look"]) == ("look", "legal")
    assert tool_action("<alfworld>Action: teleport</alfworld>", ["look"]) == ("teleport", "inadmissible")
    ep["split"] = "dev"
    with pytest.raises(ValueError):
        encode_episode(ep, tokenizer)


@pytest.mark.parametrize("reply", ["Action: look", "<alfworld>look</alfworld>",
    "<alfworld>Action: look</alfworld><alfworld>Action: look</alfworld>",
    "<think><alfworld>Action: look</alfworld></think>", "<observation>win</observation>",
    "<alfworld>Action: look</alfworld><answer>done</answer>", "<alfworld>Action: look"])
def test_malformed_tags_do_not_execute(reply):
    assert h0_reply(reply, ["look"]) == ("Invalid CoA tool call", None, "parser_error")


def test_progressive_filter_and_exposures():
    assert filter_episode(episode(4), run_index=0)["reason"] == "fewer_than_five_interactions"
    losing = episode()
    losing["success"] = "unknown"
    assert filter_episode(losing, run_index=0)["copies"] == 0
    assert filter_episode(episode(), run_index=0)["copies"] == 1
    corrected = filter_episode(episode(correction=True), run_index=0)
    assert corrected["error_correction"] and corrected["copies"] == 2
    nonreflective = [filter_episode(episode(reflection=False), run_index=i)["copies"] for i in range(30)]
    assert 0 < sum(nonreflective) < 30
    assert nonreflective == [filter_episode(episode(reflection=False), run_index=i)["copies"] for i in range(30)]
    malformed = episode()
    malformed["steps"][0]["policy_metadata"]["coa_reply"] = "I won"
    assert filter_episode(malformed, run_index=0)["copies"] == 0
    # Role chatter does not count as an executed interaction.
    no_step = episode()
    no_step["steps"][0]["environment_steps"] = 0
    assert filter_episode(no_step, run_index=0)["copies"] == 0


def test_capped_dynamic_roles_no_leaks_cache_and_cap(tmp_path, task, visible):
    texts = []
    for role, output in [("plan", "<plan>Inspect first</plan>"),
                         ("reflection", "<reflection>Review evidence</reflection>"),
                         ("double_check", "<double_check>Score: 2\nNeed evidence</double_check>"),
                         ("alfworld", "<alfworld>Action: look</alfworld>")]:
        texts.extend([json.dumps({"text": "Select " + role, "next": role}), output])
    client = RawClient(texts=texts)
    c = capped(tmp_path, client, cap=8, teacher_class=RoleTeacher, prompt_template=ROLE_PROMPT)
    policy = MultiAgentTeacher(c, task, 0, tmp_path / "agents")
    result = policy.generate([], {"secret": "SECRET"}, dict(visible, facts="SECRET"))
    assert result.parsed_action == "look" and result.metadata["coa_calls"] == 8
    assert [r["agent"] for r in result.metadata["coa_agents"]] == ["think", "plan", "think", "reflection", "think", "double_check", "think", "alfworld"]
    assert "SECRET" not in json.dumps(client.requests)
    assert task["task_id"] not in json.dumps(client.requests)
    assert json.loads((tmp_path / "test.counter.json").read_text())["attempts"] == 8
    # Exact cached replay within the same reserved run does not spend a slot.
    replay = MultiAgentTeacher(c, task, 0, tmp_path / "agents2")
    assert replay.generate([], {}, visible).raw_text == result.raw_text
    assert len(client.requests) == 8
    with pytest.raises(BudgetExceeded):
        MultiAgentTeacher(c, task, 1, tmp_path / "agents3").generate([], {}, visible)


def test_teacher_evaluation_purpose_is_separate_and_dev_only(tmp_path, task, visible):
    dev = dict(task, split="dev", task_id="valid_seen/a/1")
    client = RawClient()
    c = capped(tmp_path, client, evaluation_only=True)
    assert c.generate([], dev, visible).parsed_action == "look"
    assert c.identity["purpose"] == "dev_evaluation_only"
    for forbidden in (task, dict(dev, split="test"), dict(dev, sealed=True)):
        with pytest.raises(ValueError):
            c.generate([], forbidden, visible)
    with pytest.raises(ValueError):
        capped(tmp_path / "train", RawClient()).generate([], dev, visible)
    assert len(client.requests) == 1


@pytest.mark.parametrize("key,value", [("CUDA_VISIBLE_DEVICES", "0"), ("SLURM_JOB_GPUS", "0"),
    ("SLURM_STEP_GPUS", "0"), ("SLURM_GPUS_ON_NODE", "4")])
def test_teacher_never_holds_gpu(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="GPU|CPU"):
        require_cpu()


def test_student_real_http_adapter_uses_tags_and_original_history(tmp_path):
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"model": "student", "choices": [{"message": {"content": "<think>Look</think><alfworld>Action: look</alfworld>"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 10}})
    policy = CoAStudent(task_id="train/a", endpoint="http://127.0.0.1/v1", model="student", revision="rev",
        enable_thinking=False, log_dir=tmp_path, transport=httpx.MockTransport(respond))
    p = page(commands=["look"])
    first = policy.generate([], {}, p)
    assert first.parsed_action == "look" and first.raw_text == "Action: look"
    policy.generate([{"visible": p, "reply": first.raw_text}], {}, p)
    assert "<think>Look</think>" in requests[-1]["messages"][2]["content"]
    assert requests[0]["temperature"] == 0 and requests[0]["seed"] == 0 and requests[0]["max_tokens"] == 1024
    policy.close()


def test_eight_sample_query_and_dynamic_filters():
    all_win = [{"success": True}] * 8
    all_fail = [{"success": False}] * 8
    partial = all_win[:7] + all_fail[:1]
    assert not query_kept(all_win) and query_kept(all_fail) and query_kept(partial)
    assert not mixed(all_win) and not mixed(all_fail) and mixed(partial)
    for bad in (all_win[:7], [{"success": "unknown"}] + all_win[:7]):
        with pytest.raises(ValueError):
            query_kept(bad)


def test_real_collection_filter_loader_resume_tamper(seed_bank, tmp_path, monkeypatch):
    import baselines.coa.data as data
    from harness.alfworld_env import Environment
    from conftest import FakeBackend
    from sed.manifest import resolve_task
    root, manifest, freeze, config = seed_bank
    config["environment"]["max_episode_steps"] = 40
    for name, content in (("manifest", manifest), ("split_freeze", freeze)):
        path = tmp_path / (name + ".json")
        write_json(path, content)
        config["paths"][name] = str(path)
    task = next(resolve_task(manifest, freeze, t["task_id"]) for t in manifest["tasks"] if t["source_split"] == "train")
    listing = {"task_ids": [task["task_id"]], "parent_seed_ids": {task["task_id"]: ["seed"]}}
    def env(config, task):
        def backend(path, seed):
            b = FakeBackend(path, seed)
            b.win_at = 5
            return b
        return Environment(root, max_episode_steps=40, backend_factory=backend)
    class RoleClient(RawClient):
        def create(self, **kwargs):
            state = json.loads(kwargs["input"][-1]["content"])
            role = state["role"]
            if role == "think":
                next_role = "reflection" if "reflection" in state["allowed_next_roles"] else "alfworld"
                raw = json.dumps({"text": "Inspect", "next": next_role})
            else:
                raw = "<reflection>Review observations</reflection>" if role == "reflection" else "<alfworld>Action: look</alfworld>"
            self.texts.insert(0, raw)
            return super().create(**kwargs)
    client = RoleClient()
    cell = tmp_path / "cell"
    c = budget(config, cell / "teacher", {"test": 1}, [task], client=client)
    out = tmp_path / "collected"
    collect(config, listing, [task], out, cell, {"test": 1}, env_factory=env, capped=c)
    records = data.read(out / "receipts.json")
    assert len(records) == 3 and all(r["training_eligible"] for r in records)
    assert all(checked_record(r)["success"] is True for r in records)
    assert len(client.requests) == 3 * 5 * 4
    collect(config, listing, [task], tmp_path / "resumed", cell, {"test": 1}, env_factory=env, capped=c)
    assert len(client.requests) == 60
    monkeypatch.setattr(data, "support", lambda *a: (listing, [task]))
    verified = tmp_path / "verified"
    verified.mkdir()
    write_json(verified / "manifest.json", {})
    dataset = tmp_path / "dataset"
    datasets(out, dataset, config, verified)
    examples = training_examples(dataset / "data.json", config, Tokenizer(), verified)
    assert len(examples) == 3 and len(examples[0]["turns"]) == 5
    assert "<observation>" in (dataset / "alpaca.json").read_text()
    record = records[0]
    (Path(record["directory"]) / "agents.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        training_examples(dataset / "data.json", config, Tokenizer(), verified)


def test_stage_isolation_hashes_and_interrupted_runs(tmp_path):
    from baselines.coa.pipeline import stage
    calls = []
    with pytest.raises(ValueError, match="prerequisite"):
        stage(tmp_path, "teacher", "hash", lambda a: calls.append(a), selected="train-sft")
    assert calls == []
    def action(a):
        (a / "output").mkdir()
        write_json(a / "output/result.json", {"ok": True})
    out = stage(tmp_path, "teacher", "hash", action, selected="teacher")
    assert stage(tmp_path, "teacher", "hash", lambda _: pytest.fail("reran"), selected="data") == out
    write_json(out / "result.json", {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        stage(tmp_path, "teacher", "hash", action, selected="teacher")


def test_pipeline_gpu_stage_cannot_collect_teacher(tmp_path, monkeypatch):
    import baselines.coa.pipeline as p
    monkeypatch.setattr(p, "support", lambda *a: ({"task_ids": ["train/a"]}, [{}]))
    monkeypatch.setattr(p, "collect", lambda *a: pytest.fail("teacher invoked"))
    with pytest.raises(ValueError, match="prerequisite teacher"):
        p.train_pipeline(NS(output=tmp_path, through="train-sft", verified="unused"), {}, {})


def test_schedule_full_support_and_original_recipe():
    from baselines.coa.pipeline import official_sources
    from baselines.coa.rl import query_schedule
    from baselines.common import registered_support
    from sed.seed_selection import ALL_TRAIN_RULE
    assert registered_support({"K": 3553, "selection_rule": ALL_TRAIN_RULE})
    assert all(registered_support({"K": k}) for k in (5, 10, 20, 50, 100))
    ids = [str(i) for i in range(100)]
    schedule = query_schedule(ids)
    assert len(schedule["task_ids"]) == 192
    assert len(schedule["dropped"][0]["task_ids"]) == 4
    assert schedule["sampler_seed"] == 1 and schedule["generation_batch"] == 12
    assert len(official_sources()) == 14
    assert RECIPE["sft"]["original_batch"] == 128 and RECIPE["rl"]["clip_high"] == .28


def test_eval_requires_coa_gate_before_environment(tmp_path):
    from eval.run_suite import run_suite
    with pytest.raises((ValueError, KeyError)):
        run_suite(config={}, tasks_path="missing", output=tmp_path / "eval", spec={"coa": True, "model": "x", "endpoint": "http://localhost"})
    assert not (tmp_path / "eval").exists()


def test_launcher_cpu_allocation_guard(tmp_path):
    import os
    import subprocess
    config = tmp_path / "hpg.env"
    config.write_text("PROJECT_ROOT=" + str(Path.cwd()) + "\n")
    env = {**os.environ, "HPG_CONFIG": str(config), "COA_OUT": str(tmp_path / "cell"),
           "COA_THROUGH": "teacher", "VERIFIED": "fake", "SLURM_JOB_GPUS": "0"}
    result = subprocess.run(["bash", "scripts/hpg_coa.slurm"], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "cannot hold a GPU" in result.stderr


def test_empty_and_length_limited_rl_completions_have_only_sampled_targets():
    ep, tok = episode(n=1), Tokenizer()
    ep["success"] = False
    step = ep["steps"][0]
    step.update(action_raw="Invalid CoA tool call", action_parsed=None, action_outcome="parser_error", environment_steps=0)
    step["policy_metadata"] = {"coa_reply": "", "finish_reason": "stop"}
    row = encode_episode(ep, tok, rl=True)["turns"][0]
    assert [v for v in row["labels"] if v != -100] == [106]
    step["policy_metadata"] = {"coa_reply": "<think>unfinished", "finish_reason": "length"}
    row = encode_episode(ep, tok, rl=True)["turns"][0]
    assert row["labels"][-1] != 106
    assert tok.decode([v for v in row["labels"] if v != -100]) == "<think>unfinished"
    ep["steps"] = []
    assert encode_episode(ep, tok, rl=True)["supervised_tokens"] == 0


def test_interrupted_teacher_run_spends_slot_without_fake_restore(tmp_path, monkeypatch):
    import baselines.coa.data as data
    task = {"task_id": "train/a", "split": "train", "source_split": "train", "sealed": False}
    listing = {"task_ids": ["train/a"], "parent_seed_ids": {"train/a": ["seed"]}}
    identity = {"cell": "test"}
    cell = tmp_path / "cell"
    for index in range(3):
        directory = cell / "teacher/executions" / digest(["train/a", index])
        directory.mkdir(parents=True)
        write_json(directory / "start.json", {"identity": digest(identity), "task_id": "train/a", "run_index": index,
            "evaluation": False, "parent_seed_ids": ["seed"]})
    monkeypatch.setattr(data, "write_provenance", lambda *a, **k: None)
    class NeverCalled:
        def generate(self, *args):
            pytest.fail("Interrupted run was restarted")
    config = {"environment": {"max_episode_steps": 40}, "paths": {"manifest": "unused", "split_freeze": "unused"}}
    out = tmp_path / "out"
    collect(config, listing, [task], out, cell, identity, capped=NeverCalled(),
            env_factory=lambda *a: pytest.fail("Fake restoration/reset"))
    receipts = data.read(out / "receipts.json")
    assert len(receipts) == 3 and all(r["status"] == "interrupted" and not r["training_eligible"] for r in receipts)
    assert all(r["success"] == "unknown" for r in receipts)
    for r in receipts:
        assert checked_record(r) is None


def test_collection_rejects_dev_as_training_before_environment(tmp_path):
    task = {"task_id": "valid_seen/a", "split": "dev", "source_split": "valid_seen", "sealed": False}
    with pytest.raises(ValueError, match="frozen train"):
        collect({}, {}, [task], tmp_path / "out", tmp_path / "cell", {}, capped=object(),
                env_factory=lambda *a: pytest.fail("Environment reset on dev"))
    assert not (tmp_path / "out").exists()


def test_gate_numerical_thresholds_and_corruption():
    from baselines.coa.gate import comparison
    cases = [{"input_ids": [1, 2, 106], "prompt_tokens": 1, "input_sha256": digest([1, 2, 106]),
        "source": {"task_id": "train/a", "turn_index": i, "training_eligible": True},
        "hf_logprobs": [-1., -2.], "vllm_logprobs": [-1.01, -2.01]} for i in range(8)]
    assert comparison(cases)["passed"]
    bad = deepcopy(cases)
    bad[0]["vllm_logprobs"] = [-4., -4.]
    assert not comparison(bad)["passed"]
    bad[0]["input_ids"] = [3, 2, 106]
    with pytest.raises(ValueError):
        comparison(bad)


def test_dynamic_sampling_discards_constant_groups_and_accumulates(tmp_path, monkeypatch):
    import baselines.coa.pipeline as p
    ids = ["train/" + str(i) for i in range(16)]
    listing = {"task_ids": ids, "parent_seed_ids": {i: ["seed"] for i in ids}}
    snapshot, attempt = tmp_path / "checkpoint", tmp_path / "attempt"
    snapshot.mkdir()
    attempt.mkdir()
    write_json(snapshot / "training_state.json", {})
    calls, batches = [], {}
    @contextmanager
    def served(*a):
        yield "endpoint"
    def rollout(args, selected, endpoint, attempt, output, index):
        output.mkdir()
        calls.append(selected["task_ids"])
        # First group all zero, second mixed, next two mixed. Target batch=2.
        batches[str(output)] = [(tid, [{"success": i != 0 and j == 0} for j in range(8)])
            for i, tid in enumerate(selected["task_ids"], start=0 if len(calls) == 1 else 1)]
    monkeypatch.setattr(p, "served", served)
    monkeypatch.setattr(p, "rollout_command", rollout)
    monkeypatch.setattr(p, "checked_groups", lambda root, _: batches[str(root)])
    monkeypatch.setattr(p, "write_provenance", lambda *a, **k: None)
    eligible = {**listing, "query_schedule": {"task_ids": ids * 2}}
    p.explore_update(NS(), {}, listing, eligible, snapshot, attempt, 0, 0)
    plan = json.loads((attempt / "output/plan.json").read_text())
    assert len(calls) == 2 and len(plan["selected"]) == 2 and len(plan["discarded"]) == 1
    assert len(plan["unused_mixed_groups"]) == 1 and plan["next_cursor"] == 4


def test_teacher_dev_evaluation_does_not_prepare_training_data(tmp_path, monkeypatch):
    import baselines.coa.pipeline as p
    from sed.config import load_config
    from sed.provenance import code_hashes
    from baselines.coa import RECIPE
    root = tmp_path / "cell"
    root.mkdir()
    config = load_config()
    args = ["--snapshot", str(tmp_path / "base"), "--verified", str(tmp_path / "verified"),
            "--output", str(root), "--mode", "teacher-evaluate"]
    identity = {"method": "coa", "recipe": {"parameters": RECIPE, "official_source_sha256": {}},
        "config_sha256": digest(config), "snapshot": str(tmp_path / "base"), "verified": str(tmp_path / "verified"),
        "code_sha256": code_hashes(Path.cwd())}
    write_json(root / "identity.json", identity)
    monkeypatch.setattr(p, "official_sources", lambda: {})
    monkeypatch.setattr(p, "prepare", lambda *a, **k: pytest.fail("Evaluation entered prepare"))
    monkeypatch.setattr(p, "load_frozen", lambda *a: ({}, {}))
    monkeypatch.setattr(p, "frozen_tasks", lambda *a, **k: ({"name": "dev140"}, [{"split": "dev", "sealed": False}]))
    called = []
    def collected(*a, **k):
        called.append(k)
        a[3].mkdir()
    monkeypatch.setattr(p, "collect", collected)
    monkeypatch.setattr(p, "update_disclosure", lambda *a, **k: None)
    p.main(args)
    assert called == [{"evaluation": True}]
    assert not (root / "tasks.json").exists()


def test_verified_support_loader_accepts_actual_nested_view(seed_bank, tmp_path):
    from sed.seed_selection import select_seeds
    from sed.seed_evidence import materialize_subsets
    from teachers.collect_seeds import collect as collect_seeds
    root, manifest, freeze, config = seed_bank
    for name, value in (("manifest", manifest), ("split_freeze", freeze)):
        path = tmp_path / (name + ".json")
        write_json(path, value)
        config["paths"][name] = str(path)
    selection = select_seeds(manifest, freeze, k=10)
    environment, teacher = factories(root, config, set(), [])
    verified = tmp_path / "verified"
    collect_seeds(selection, manifest, freeze, config, verified, env_factory=environment, teacher_factory=teacher)
    materialize_subsets(verified, manifest, freeze, [5])
    full, _ = support(config, verified)
    nested, tasks = support(config, verified / "subsets/k5")
    assert len(tasks) == 5 and set(nested["task_ids"]).issubset(full["task_ids"])
    assert all(nested["parent_seed_ids"][t] == full["parent_seed_ids"][t] for t in nested["task_ids"])


from test_full_train import full_bank


def test_verified_support_loader_accepts_actual_full_train(full_bank, tmp_path):
    from test_full_train import factories as full_factories
    from teachers.collect_seeds import collect as collect_seeds
    _, manifest, freeze, selection, config = full_bank
    environment, teacher = full_factories(full_bank, tmp_path / "seed_calls.jsonl")
    verified = tmp_path / "verified"
    collect_seeds(selection, manifest, freeze, config, verified, env_factory=environment, teacher_factory=teacher)
    listing, tasks = support(config, verified)
    assert len(tasks) == len(selection["candidate_order"]) == 4  # Not a registered few-shot K.
    assert len(listing["parent_seed_ids"]) == 4


def test_sampler_schedule_cli_uses_cpu_training_environment(tmp_path):
    import os
    import subprocess
    import sys
    path, out = tmp_path / "tasks.json", tmp_path / "schedule.json"
    write_json(path, {"task_ids": [str(i) for i in range(100)]})
    result = subprocess.run([sys.executable, "-m", "baselines.coa.rl", "--tasks", str(path), "--output", str(out)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "src:."}, capture_output=True, text=True)
    assert result.returncode == 0
    assert len(json.loads(out.read_text())["task_ids"]) == 192
