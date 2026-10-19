"""Offline GAD data contracts, stage transactions, sampler and launch isolation."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import httpx
import pytest

from baselines.gad import RECIPE
from baselines.gad.data import DrawTeacher, require_cpu_teacher, encode_state
from baselines.gad.sampling import sample_group
from baselines.gad.training import round_plan
from sed.storage import digest, JsonlLedger, write_json
from test_capped_teacher import capped, RawClient
from test_had import Tokenizer
from test_stage2b_seeds import seed_bank, factories


def test_pins_and_training_amount():
    from baselines.gad.pipeline import recipe
    assert recipe()["algorithm"] == RECIPE
    assert RECIPE["group_size"] == 8 and RECIPE["kl_coef"] == .001
    assert RECIPE["warmup_epochs"] == 1 and RECIPE["joint_epochs"] == 2
    for n in (1, 5, 10, 20, 50, 100, 3553, 142120):
        for r in range(3):
            plan = round_plan(n, r)
            pre = [p for p in plan if p["kind"] == "discriminator"]
            assert len(pre) == (10 if r == 0 else 0)
            indices = [i for p in plan if p["kind"] != "discriminator" for i in p["indices"]]
            assert sorted(indices) == list(range(n))
            assert all(p["b_prime"] == min(256, max(1, n // 8)) for p in plan)


def test_sampling_exact_token_ids_group_and_cap(tmp_path):
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "student", "choices": [
            {"index": i, "text": "raw", "token_ids": [i + 2] * (1024 if i == 0 else i + 1),
             "finish_reason": "length" if i == 0 else "stop"} for i in reversed(range(8))],
             "usage": {"prompt_tokens": 2, "completion_tokens": 1059}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        groups = sample_group(client, "http://local/v1", "student", [7, 8], seed=17,
                              ledger=JsonlLedger(tmp_path / "calls.jsonl"))
    assert len(groups) == 8 and groups[0]["input_ids"] == [7, 8] + [2] * 1024
    assert seen[0]["max_tokens"] == 1024 and seen[0]["n"] == 8
    assert seen[0]["temperature"] == .8 and seen[0]["return_token_ids"]
    assert "logprobs" not in seen[0] and seen[0]["seed"] == 17


@pytest.mark.parametrize("change", ["missing_ids", "missing_choice", "wrong_model", "over_cap", "transport"])
def test_sampling_rejects_without_retokenizing_and_sanitizes(tmp_path, change):
    def handler(request):
        if change == "transport":
            raise RuntimeError("SECRET_EXCEPTION")
        raw = {"model": "student", "choices": [
            {"index": i, "text": "raw", "token_ids": [2, 3], "finish_reason": "stop"} for i in range(8)]}
        if change == "missing_ids":
            del raw["choices"][0]["token_ids"]
        if change == "missing_choice":
            raw["choices"].pop()
        if change == "wrong_model":
            raw["model"] = "substitution"
        if change == "over_cap":
            raw["choices"][0]["token_ids"] = [2] * 1025
        return httpx.Response(200, json=raw)
    path = tmp_path / "calls.jsonl"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(RuntimeError, match="sanitized"):
        sample_group(client, "http://local/v1", "student", [7], seed=0, ledger=JsonlLedger(path))
    assert "SECRET_EXCEPTION" not in path.read_text()


def test_native_encoding_whitelist():
    row = {"history": [], "task": {"internal_state": "SECRET"},
        "visible": {"observation": "cup", "admissible_commands": ["look"], "facts": "SECRET"},
        "teacher": {"raw_text": "Thought: visible\nAction: look", "metadata": {"secret": "SECRET"}},
        "extra.expert_plan": "SECRET"}
    tokenizer = Tokenizer()
    encoded = encode_state(row, tokenizer)
    assert "SECRET" not in tokenizer.decode(encoded["input_ids"])
    assert tokenizer.decode([i for i in encoded["labels"] if i != -100]) == row["teacher"]["raw_text"] + "<turn|>"
    assert all(i == -100 for i in encoded["labels"][:encoded["prompt_tokens"]])
    row["teacher"]["raw_text"] = "x" * 1536
    with pytest.raises(ValueError, match="never truncate"):
        encode_state(row, tokenizer)


@pytest.mark.parametrize("key,value", [("CUDA_VISIBLE_DEVICES", "0"), ("SLURM_JOB_GPUS", "0"),
    ("SLURM_STEP_GPUS", "1"), ("SLURM_GPUS_ON_NODE", "4"), ("SLURM_GPUS", "1")])
def test_teacher_cpu_allocation(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="Teacher"):
        require_cpu_teacher()


def test_one_draw_cache_cap_and_no_reasoning_access(tmp_path, task, visible):
    client = RawClient(texts=["unparseable reply", "Action: look"])
    teacher = capped(tmp_path, client, cap=2, teacher_class=DrawTeacher)
    teacher.teacher.draw_id = "round-0-history-1"
    assert teacher.generate([], task, visible).raw_text == "unparseable reply"
    assert teacher.generate([], task, visible).metadata["cache_hit"]
    teacher.teacher.draw_id = "round-1-history-1"
    teacher.generate([], task, visible)
    assert len(client.requests) == 2
    assert not {"logprobs", "include", "reasoning"} & set(client.requests[0])
    teacher.teacher.draw_id = "third"
    from teachers.capped import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        teacher.generate([], task, visible)


def test_executed_histories_teacher_same_context_and_tamper(seed_bank, tmp_path):
    from baselines.common.explore import explore_one, run_exploration
    from baselines.gad.data import collect, load_rows, visited_states
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
    def backend(path, seed):
        result = FakeBackend(path, seed)
        result.win_at = 2
        return result
    def worker(job):
        return explore_one(job, policy_factory=lambda **_: ScriptedPolicy(["look"]),
            env_factory=lambda c, t: Environment(root, max_episode_steps=40, backend_factory=backend))
    exploration = tmp_path / "exploration"
    run_exploration(config=config, tasks_path=tasks, output=exploration, spec={"enable_thinking": False,
        "max_tokens": 1024}, decoding="greedy", workers=1, worker_fn=worker)
    states = visited_states(exploration, verified, config, 0)
    assert len(states) == 2 and states[1]["history"][0]["reply"] == "Action: look"
    identity = {"unique_instances": 1}
    client = RawClient(texts=["complete but unparseable", "Action: go north"])
    capped_teacher = capped(tmp_path / "teacher", client, cap=120, teacher_class=DrawTeacher)
    collect(exploration, verified, tmp_path / "labels", config, cell=tmp_path / "cell",
            identity=identity, round_index=0, teacher=capped_teacher)
    rows = load_rows(tmp_path / "labels/data.json", config, verified=verified, identity=identity, round_index=0)
    assert len(client.requests) == len(rows) == 2
    assert "SECRET" not in json.dumps([r["input"] for r in client.requests])
    assert "Action: look" in json.dumps(client.requests[1]["input"])
    # Completed teacher stage replay consumes no additional capped calls.
    collect(exploration, verified, tmp_path / "labels-again", config, cell=tmp_path / "cell",
            identity=identity, round_index=0, teacher=capped_teacher)
    assert len(client.requests) == 2
    dataset = json.loads((tmp_path / "labels/data.json").read_text())
    dataset["rows"][0]["teacher"]["raw_text"] = "forged"
    write_json(tmp_path / "labels/data.json", dataset)
    with pytest.raises(ValueError, match="capped raw"):
        load_rows(tmp_path / "labels/data.json", config, verified=verified, identity=identity, round_index=0)


def test_pipeline_one_resource_stage_and_resume(tmp_path, monkeypatch):
    import baselines.gad.pipeline as p
    args = NS(output=str(tmp_path / "cell"), snapshot="base", verified="verified", through="train-0",
        config="config", model_name="model", env_python="python", workers=1)
    calls = []
    @contextmanager
    def served(*_):
        calls.append("serve")
        yield "endpoint"
    def command(cmd, log, **kw):
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == ""
        calls.append("explore")
        Path(cmd[cmd.index("--output") + 1]).mkdir(parents=True)
    def collect(*pos, **kw):
        calls.append("teacher")
        write_json(Path(pos[2]) / "data.json", {})
    def train(*pos, **kw):
        calls.append("train")
        checkpoint = tmp_path / f"checkpoint-{kw['round_index']}"
        write_json(Path(kw["output"]) / "result.json", {"checkpoint": str(checkpoint)})
        write_json(Path(args.output) / "models.json", [])
    monkeypatch.setattr(p, "served", served)
    monkeypatch.setattr(p, "run_command", command)
    monkeypatch.setattr(p, "collect", collect)
    monkeypatch.setattr(p, "train_round", train)
    with pytest.raises(ValueError, match="Missing prerequisite explore-0"):
        p.train_pipeline(args, {}, {})
    assert calls == []
    for stage in p.STAGES:
        args.through = stage
        start = len(calls)
        p.train_pipeline(args, {}, {})
        assert calls[start:] == (["serve", "explore"] if stage.startswith("explore") else [stage.split("-")[0]])
        done = len(calls)
        p.train_pipeline(args, {}, {})
        assert len(calls) == done
    write_json(Path(args.output) / "stages/explore-0/attempt-000/output/tamper.json", {})
    with pytest.raises(ValueError, match="artifacts changed"):
        p.train_pipeline(args, {}, {})


def test_batch_resampling_optimizer_chain_and_resume(tmp_path, monkeypatch):
    import baselines.gad.training as t
    args = NS(output=str(tmp_path / "cell"), verified="verified", config="config", gpus=1,
              train_python="python", snapshot="base")
    data = tmp_path / "data.json"
    write_json(data, {})
    monkeypatch.setattr(t, "load_rows", lambda *a, **kw: [None] * 2)
    monkeypatch.setattr(t, "check_history_policy", lambda *a: None)
    sampled, commands = [], []
    @contextmanager
    def served(args, snapshot, attempt):
        sampled.append(str(snapshot))
        yield "endpoint"
    def command(cmd, log, **kw):
        commands.append(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True)
        if "baselines.gad.sampling" in cmd:
            write_json(out / "groups.json", {})
        else:
            if cmd[cmd.index("--role") + 1] == "actor":
                write_json(out / "checkpoint/training_state.json", {})
            else:
                write_json(out / "rewards.json", {})
    monkeypatch.setattr(t, "served", served)
    monkeypatch.setattr(t, "run_command", command)
    previous, snapshot = None, "base"
    for r in range(3):
        out = tmp_path / f"round-{r}"
        t.train_round(args, {}, {}, data=data, snapshot=snapshot, previous=previous, round_index=r, output=out)
        result = json.loads((out / "result.json").read_text())
        before = len(commands)
        t.train_round(args, {}, {}, data=data, snapshot=snapshot, previous=previous, round_index=r,
                      output=tmp_path / f"resume-{r}")
        assert len(commands) == before
        previous, snapshot = result, result["checkpoint"]
    assert len(sampled) == 16  # 10 D-only + 2 CE + 2 + 2 joint batches
    assert sampled[:11] == ["base"] * 11
    assert len(set(sampled[10:])) == 6  # Fresh generator after every actor update.
    models = json.loads((Path(args.output) / "models.json").read_text())
    assert len(models) == 6
    actor = [c for c in commands if "--role" in c and c[c.index("--role") + 1] == "actor"]
    assert "--resume" not in actor[0] and all("--resume" in c for c in actor[1:])
    assert all("--reference" in c for c in actor[2:])
    assert len({c[c.index("--reference") + 1] for c in actor[2:]}) == 1


@pytest.mark.parametrize("gpu", [False, True])
def test_slurm_teacher_resource_guard(tmp_path, gpu):
    fake = tmp_path / "python"
    fake.write_text("#!" + sys.executable + "\nimport os\nassert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n")
    fake.chmod(0o755)
    config = tmp_path / "hpg.env"
    config.write_text("# inherit the explicit test environment\n")
    env = os.environ.copy()
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE"):
        env.pop(key, None)
    env.update(PROJECT_ROOT=str(Path.cwd()), HPG_CONFIG=str(config), ENV_PYTHON=str(fake),
        TRAIN_PYTHON=str(fake), VLLM_PYTHON=str(fake), GAD_THROUGH="teacher-0", GAD_OUT=str(tmp_path / "out"),
        VERIFIED="verified", MODEL_PATH="base", ENABLE_THINKING="off", MAX_TOKENS="1024", MAX_LENGTH="32768",
        WORKERS="1", READY_TIMEOUT="1", NGPUS="1", CUDA_VISIBLE_DEVICES="")
    if gpu:
        env["SLURM_JOB_GPUS"] = "0"
    result = subprocess.run(["bash", "scripts/hpg_gad.slurm"], env=env, capture_output=True, timeout=10)
    assert result.returncode == (2 if gpu else 0), result.stderr
