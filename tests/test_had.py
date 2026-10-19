"""Offline HAD harness, paired capped labels, stage isolation and evaluation."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from baselines.had import RECIPE
from baselines.had.data import DrawTeacher, encode_state, round_tasks, require_cpu_teacher
from baselines.had.harness import records, augment, messages, validity
from baselines.had.objective import round_plan
from sed.react import build_messages
from sed.storage import digest, write_json
from test_capped_teacher import RawClient, capped
from test_stage2b_seeds import seed_bank, factories
from train.dataset import EMPTY_THOUGHT, TURN_TOKEN


class Tokenizer:
    """Native-boundary stub with character offsets; no model download."""
    all_special_ids = [106]
    def encode(self, text, **kwargs):
        parts = text.split(TURN_TOKEN)
        ids = []
        for i, part in enumerate(parts):
            if i:
                ids.append(106)
            ids.extend(ord(c) + 200 for c in part)
        return ids
    def decode(self, ids, **kwargs):
        return "".join(TURN_TOKEN if i == 106 else chr(i - 200) for i in ids)
    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}
    def apply_chat_template(self, turns, *, add_generation_prompt, **kwargs):
        value = "".join(t["role"] + ":" + t["content"].strip() + TURN_TOKEN + "\n" for t in turns)
        return value + "assistant:" + EMPTY_THOUGHT if add_generation_prompt else value


def page(text="You arrive at desk 1.", commands=None):
    return {"observation": text, "admissible_commands": commands or ["look", "take mug 1 from desk 1", "go to sinkbasin 1"]}


def state(**kwargs):
    return {"state_id": "one", "policy_version": 0, "task_id": "train/a/1", "parent_seed_ids": ["seed"],
        "history": [], "visible": page(), "task": {}, "student_reply": "Thought: STUDENT reasoning\nAction: look",
        "with_harness": {"raw_text": "Thought: TEACHER reasoning\nAction: take mug 1 from desk 1"},
        "without_harness": {"raw_text": "Thought: OTHER reasoning\nAction: look"}, **kwargs}


def test_tracking_history_stall_and_no_hidden_fields():
    visible = page()
    polluted = {**visible, "location": "SECRET", "inventory": ["SECRET"], "extra.expert_plan": "SECRET"}
    history = [{"visible": polluted, "reply": "Action: look", "audit": "SECRET"}]
    history += [{"visible": page("Nothing happens."), "reply": "Action: look"} for _ in range(4)]
    record = records(history, page("Nothing happens."))
    assert record["location"] == "desk 1" and record["inventory"] == []
    assert record["step"] == 6 and record["remaining"] == 34
    assert record["stall"] and record["failed_actions"] == [{"action": "look", "attempts": 5}]
    assert validity("look", record) == (False, "repeated_no_effect")
    assert "SECRET" not in json.dumps(messages(history, {"internal_state": "SECRET"}, polluted))
    assert records(history, polluted) == records(history, visible)
    assert augment(history, visible)["admissible_commands"] == sorted(visible["admissible_commands"])
    assert not records([], visible)["stall"] and not records([], visible)["failed_actions"]
    assert records(history[:1], page("Nothing happens."))["failed_actions"] == []
    # Both teacher views keep the same realized student replies and H0 commands.
    plus, minus = messages(history, {}, visible), build_messages(history, {}, visible)
    assert plus[:-1] == minus[:-1] and plus[-1] != minus[-1]


def test_inventory_visits_and_validity_are_pre_action():
    first = page()
    second = page("You pick up the mug 1 from the desk 1.", ["look", "put mug 1 in/on desk 1"])
    third = page("You arrive at sinkbasin 1.", ["clean mug 1 with sinkbasin 1", "look"])
    history = [{"visible": first, "reply": "Action: take mug 1 from desk 1"},
               {"visible": second, "reply": "Action: go to sinkbasin 1"}]
    record = records(history, third)
    assert record["inventory"] == ["mug 1"] and record["location"] == "sinkbasin 1"
    assert record["visits"][0]["left_turn"] == 2 and record["visits"][1]["arrived_turn"] == 2
    assert validity("clean mug 1 with sinkbasin 1", record)[0]
    assert not validity("teleport", record)[0]
    record["inventory"] = []
    assert validity("clean mug 1 with sinkbasin 1", record) == (False, "inventory_contradiction")
    # A novel return state resets stall even if the last action failed earlier.
    assert not record["stall"]


def test_encoding_scores_actions_only_under_student_reasoning():
    tokenizer = Tokenizer()
    row = encode_state(state(), tokenizer)
    assert row["mask"]
    assert tokenizer.decode([v for v in row["response"]["labels"] if v != -100]) == state()["with_harness"]["raw_text"] + TURN_TOKEN
    for side, expected in (("chosen", "take mug 1 from desk 1"), ("rejected", "look")):
        encoded = row[side]
        supervised = tokenizer.decode([v for v in encoded["labels"] if v != -100])
        assert supervised == expected
        prompt = tokenizer.decode(encoded["input_ids"])
        assert "STUDENT reasoning" in prompt and "TEACHER reasoning" not in prompt and "OTHER reasoning" not in prompt
        assert encoded["labels"][-1] == -100  # Action-only preference, no invented EOS target.
    changed = state(with_harness={"raw_text": "Thought: invalid\nAction: teleport"})
    masked = encode_state(changed, tokenizer)
    assert not masked["mask"]
    assert "teleport" in tokenizer.decode([v for v in masked["response"]["labels"] if v != -100])
    assert not encode_state(state(student_reply="unparseable"), tokenizer)["mask"]
    with pytest.raises(ValueError, match="no truncation"):
        encode_state(state(), tokenizer, max_length=5)


@pytest.mark.parametrize("k", [5, 10, 20, 50, 100, 3553])
def test_support_cycles_and_batch_once_staleness(k):
    ids = [str(i) for i in range(k)]
    listing = {"task_ids": ids, "parent_seed_ids": {i: [i] for i in ids}}
    first, second = round_tasks(listing, 0), round_tasks(listing, 1)
    assert len(first["task_ids"]) == min(16, k)
    assert second["task_ids"][0] == str(min(16, k) % k)
    rows = [{"state_id": str(i), "policy_version": 0} for i in range(k)]
    plan = round_plan(rows, version=1, start_update=0)
    indices = [i for b in plan["batches"] for i in b["indices"]]
    assert len(indices) == len(set(indices)) == k
    assert plan["b_prime"] == min(60, max(1, k // 8))
    with pytest.raises(ValueError, match="Stale"):
        round_plan(rows, version=2, start_update=0)
    with pytest.raises(ValueError, match="twice"):
        round_plan(rows + rows[:1], version=0, start_update=0)
    end = round_plan(rows, version=0, start_update=249)
    assert end["end_update"] == 250
    assert len(end["unused_at_terminal_budget"]) + len(end["batches"][0]["indices"]) == k


def test_paired_draw_cache_and_cell_cap(tmp_path, task, visible):
    teacher = capped(tmp_path, RawClient(), teacher_class=DrawTeacher, cap=2)
    teacher.teacher.draw_id = "with"
    a = teacher.generate([], task, augment([], visible))
    teacher.teacher.draw_id = "without"
    b = teacher.generate([], task, visible)
    assert a.metadata["cache_key"] != b.metadata["cache_key"]
    assert teacher.generate([], task, visible).metadata["cache_hit"]
    teacher.teacher.draw_id = "new-round"
    from teachers.capped import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        teacher.generate([], task, visible)
    assert json.loads((tmp_path / "test.counter.json").read_text())["attempts"] == 2


@pytest.mark.parametrize("key,value", [("CUDA_VISIBLE_DEVICES", "0"), ("SLURM_JOB_GPUS", "0"),
    ("SLURM_STEP_GPUS", "1"), ("SLURM_GPUS_ON_NODE", "4"), ("SLURM_GPUS", "1")])
def test_teacher_refuses_gpu_allocation(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="Teacher"):
        require_cpu_teacher()


def test_pipeline_stages_do_not_cross_resource_boundaries(tmp_path, monkeypatch):
    import baselines.had.pipeline as p
    output = tmp_path / "cell"
    output.mkdir()
    listing = {"task_ids": ["train/a"], "parent_seed_ids": {"train/a": ["seed"]}}
    write_json(output / "tasks.json", listing)
    args = NS(output=str(output), snapshot=str(tmp_path / "base"), verified="verified", gpus=1,
        config="config", env_python="env-python", train_python="train-python", vllm_python="vllm-python",
        workers=1, model_name="model", through="train-0")
    calls = []
    @contextmanager
    def served(*args):
        calls.append("serve")
        yield "endpoint"
    def command(cmd, log, **kw):
        destination = Path(cmd[cmd.index("--output") + 1])
        destination.mkdir(parents=True)
        if "baselines.had.rollout" in cmd:
            calls.append("rollout")
        else:
            calls.append("train")
            write_json(destination / "checkpoint/training_state.json", {"final": True})
            write_json(destination / "result.json", {"start_update": 0, "end_update": 1,
                "examples_seen": 5, "unused_at_terminal_budget": []})
    def collect(*args, **kwargs):
        calls.append("teacher")
        Path(args[2]).mkdir(parents=True)
        write_json(Path(args[2]) / "data.json", {})
    monkeypatch.setattr(p, "served", served)
    monkeypatch.setattr(p, "run_command", command)
    monkeypatch.setattr(p, "collect", collect)
    with pytest.raises(ValueError, match="Missing prerequisite explore-0"):
        p.train_pipeline(args, {}, {})
    assert calls == []
    for selected, expected in (("explore-0", ["serve", "rollout"]),
                               ("teacher-0", ["teacher"]), ("train-0", ["train"])):
        args.through = selected
        before = len(calls)
        p.train_pipeline(args, {}, {})
        assert calls[before:] == expected
        p.train_pipeline(args, {}, {})
        assert calls[before:] == expected
    models = json.loads((output / "models.json").read_text())
    assert {m["harness"] for m in models} == {"had", "h0"}
    assert len({m["checkpoint"] for m in models}) == 1
    # Completed artifact tampering fails before resuming downstream stages.
    (output / "stages/explore-0/attempt-000/output/tamper").write_text("changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        p.train_pipeline(args, {}, {})


def test_equivalence_measurements_fail_closed():
    from baselines.had.gate import comparison
    cases = [{"input_ids": [1, 2, 3, 106], "prompt_tokens": 2,
        "input_sha256": digest([1, 2, 3, 106]), "source": {"task_id": f"train/{i}", "training_eligible": True},
        "hf_logprobs": [-.4, -.2], "vllm_logprobs": [-.41, -.21]} for i in range(8)]
    assert comparison(cases)["passed"]
    cases[0]["vllm_logprobs"] = [-4., -2.]
    assert not comparison(cases)["passed"]
    cases[0]["source"]["task_id"] = "valid_seen/leak"
    with pytest.raises(ValueError):
        comparison(cases)


def test_paper_constants_and_training_command():
    from baselines.had.pipeline import training_command
    args = NS(train_python="python", config="config", snapshot="base", verified="support", output="out", gpus=4)
    cmd = training_command(args, data="data", snapshot="previous", output="next", round_index=2, start_update=16)
    assert cmd[cmd.index("--objective") + 1] == "had"
    assert cmd[cmd.index("--start-update") + 1] == "16"
    assert RECIPE["teacher_cap"] == 30000 and RECIPE["beta"] == RECIPE["rho"] == .5
    assert RECIPE["weight_decay"] == .01 and RECIPE["warmup_steps"] == 15


def test_student_requests_harness_only_for_had_and_sampling_is_seeded(tmp_path):
    import httpx
    from baselines.had.rollout import HADStudent, H0Student
    captured = []
    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "model", "id": "stub", "choices": [{
            "message": {"content": "Thought: student\nAction: look"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 6}})
    options = dict(endpoint="http://localhost/v1", model="model", revision="revision", enable_thinking=False,
                   transport=httpx.MockTransport(respond), max_tokens=1024)
    for i, policy in enumerate((HADStudent(task_id="train/a", rollout=True, log_dir=tmp_path / "roll", **options),
        HADStudent(task_id="dev/a", log_dir=tmp_path / "had", **options),
        H0Student(log_dir=tmp_path / "h0", **options))):
        result = policy.generate([], {}, page())
        policy.close()
        assert result.parsed_action == "look"
        if i < 2:
            assert result.metadata["had_records"] == records([], page())
    assert captured[0]["temperature"] == captured[0]["top_p"] == 1 and captured[0]["top_k"] == -1
    assert captured[1]["temperature"] == captured[2]["temperature"] == 0
    assert captured[1]["seed"] == captured[2]["seed"] == 0
    assert "HAD state tracking" in json.dumps(captured[0]["messages"])
    assert "HAD state tracking" in json.dumps(captured[1]["messages"])
    assert captured[2]["messages"] == build_messages([], {}, page())


def test_checked_real_execution_to_paired_labels_with_stub_teacher(seed_bank, tmp_path):
    from baselines.common.explore import explore_one, run_exploration
    from baselines.had.data import collect as label, load_rows, visited_states
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    from sed.seed_evidence import checked_episode
    from sed.seed_selection import select_seeds
    from sed.runner import ScriptedPolicy
    from teachers.collect_seeds import collect as collect_seeds
    root, manifest, freeze, config = seed_bank
    config["paths"].update(manifest=str(tmp_path / "manifest.json"), split_freeze=str(tmp_path / "freeze.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    env, teacher = factories(root, config, set(), [])
    verified = tmp_path / "verified"
    collection = collect_seeds(select_seeds(manifest, freeze, k=1), manifest, freeze, config, verified,
                              env_factory=env, teacher_factory=teacher)
    config["environment"]["max_episode_steps"] = 40
    receipt = collection["verified"][0]
    parent = checked_episode(verified, receipt)["episode_id"]
    tasks = tmp_path / "tasks.json"
    write_json(tasks, {"task_ids": [receipt["task_id"]], "list_sha256": digest([receipt["task_id"]]),
        "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze),
        "parent_seed_ids": {receipt["task_id"]: [parent]}})
    class Policy(ScriptedPolicy):
        def generate(self, history, task, visible_context):
            reply = super().generate(history, task, visible_context)
            return reply._replace(metadata={**reply.metadata, "had_records": records(history, visible_context)})
    def backend(path, seed):
        result = FakeBackend(path, seed)
        result.win_at = 2
        return result
    def worker(job):
        return explore_one(job, policy_factory=lambda **_: Policy(["look"]),
            env_factory=lambda c, t: Environment(root, max_episode_steps=40, backend_factory=backend))
    exploration = tmp_path / "exploration"
    run_exploration(config=config, tasks_path=tasks, output=exploration, spec={"enable_thinking": False,
        "max_tokens": 1024}, decoding="greedy", workers=1, worker_fn=worker)
    states = visited_states(exploration, verified, config, 0)
    assert len(states) == 2 and all(s["parent_seed_ids"] == [parent] for s in states)
    client = RawClient(texts=["unparseable but complete", "Action: go north", "Action: look", "Action: look"])
    collector = capped(tmp_path / "calls", client, cap=4, teacher_class=DrawTeacher,
                       sampling={**config["models"]["teacher"]["sampling"], "temperature": 1.0})
    data = tmp_path / "labels"
    label(exploration, verified, data, config, cell=tmp_path / "cell", identity={}, round_index=0, teacher=collector)
    rows = load_rows(data / "data.json", config, verified=verified, identity={}, round_index=0)
    assert len(client.requests) == 4 and len(rows) == 2
    assert not encode_state(rows[0], Tokenizer())["mask"]
    prompts = [r["input"] for r in client.requests]
    assert "HAD state tracking" in json.dumps(prompts[0]) and "HAD state tracking" not in json.dumps(prompts[1])
    assert "SECRET" not in json.dumps(prompts)
    changed = json.loads((data / "data.json").read_text())
    changed["rows"][0]["with_harness"], changed["rows"][0]["without_harness"] = (
        changed["rows"][0]["without_harness"], changed["rows"][0]["with_harness"])
    write_json(data / "data.json", changed)
    with pytest.raises(ValueError, match="durable capped"):
        load_rows(data / "data.json", config, verified=verified, identity={}, round_index=0)
