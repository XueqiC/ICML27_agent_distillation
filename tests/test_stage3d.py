"""LEAP offline fakes: oracle isolation, durable caps, evidence and iteration chain."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from baselines.leap import pipeline
from baselines.leap.data import min_subsample, step_rows, prepare_set, load_set, seeds
from baselines.leap.expert import collect_one
from baselines.leap.protocol import render, trajectory, parse_correction, parse_extraction, template_hashes
from baselines.leap.teacher import LeapTeacher, TeacherBudget, checked_outcome, read
from sed.config import load_config
from sed.storage import digest, write_json
from test_teacher import FakeClient, response
from test_stage2b_seeds import seed_bank, factories


STATE = "**Task:** Find cup\n**Essential Objects:** PRIVATE_OBJECT\n**Critical Locations:** PRIVATE_LOCATION\n**Optimal Action Sequence:** PRIVATE_PLAN"


def episode():
    visible = {"observation": "Find the cup.", "admissible_commands": ["look", "go north"],
               "facts": "SECRET_FACTS", "extra.expert_plan": "SECRET_PLAN"}
    return {"schema_version": 2, "task_id": "train/a/1", "split": "train", "sealed": False,
        "episode_id": "student-original", "privileged_state": STATE,
        "initial": {"visible": visible, "audit": STATE}, "steps": [
            {"event_id": "s0", "action_raw": "Thought: ORIGINAL_REASON_0\nAction: look", "visible": dict(visible, observation="Room two"), "audit": STATE},
            {"event_id": "s1", "action_raw": "Thought: ORIGINAL_REASON_1\nAction: go north", "visible": visible}]}


def correction(length=2, action="inadmissible invented command"):
    return json.dumps({"summary": "SECRET_SUMMARY", "privileged_state": STATE, "trajectory": [
        {"timestep": i, "original_observation": "SECRET_ECHO", "original_reason": "SECRET_ECHO",
         "original_action": "SECRET_ECHO", "candidate_actions": ["SECRET_ECHO"], "is_corrected": False,
         "corrected_reason": f"Check accessible places {i}", "corrected_action": action} for i in range(length)]})


def factory(client):
    return lambda **kw: LeapTeacher(client=client, sleep=lambda _: None, **kw)


def budget(tmp_path, task, client, *, kind="privileged"):
    task = dict(task, parent_seed_ids=["seed-1"])
    config = load_config()
    config["models"]["teacher"]["rate_limit"] = {"requests_per_minute": 10000, "tokens_per_minute": 10000000}
    return TeacherBudget(tmp_path / "budget", tasks=[task], config=config, kind=kind,
        iteration=0 if kind == "extract" else 1, run_identity="test", max_output_tokens=4096,
        teacher_factory=factory(client)), task


def test_official_templates_switch_and_extraction_whitelist():
    source = episode()
    rows = trajectory(source)
    assert "SECRET" not in json.dumps(rows) and "PRIVATE" not in json.dumps(rows)
    privileged = render("privileged", rows, STATE)
    zero = render("zero", rows)
    assert STATE in privileged[-1]["content"]
    assert "PRIVATE" not in json.dumps(zero) and "privileged state" not in json.dumps(zero)
    assert "Check" not in zero[-1]["content"]  # No generated replacement history.
    extraction = render("extract", trajectory(source, expert=True))
    assert "ORIGINAL_REASON" not in json.dumps(extraction)
    assert "SECRET" not in json.dumps(extraction)
    assert parse_extraction(STATE) == STATE
    assert len(template_hashes()) == 3
    with pytest.raises(ValueError, match="forbidden"):
        render("zero", rows, STATE)
    with pytest.raises(ValueError, match="requires"):
        render("privileged", rows)


@pytest.mark.parametrize("fenced", [False, True])
def test_correction_parser_drops_echoes_keeps_all_steps_and_illegal_actions(fenced):
    raw = correction()
    labels = parse_correction("```json\n" + raw + "\n```" if fenced else raw, 2)
    assert len(labels) == 2
    assert all(set(r) == {"corrected_reason", "corrected_action"} for r in labels)
    assert "SECRET" not in json.dumps(labels) and "PRIVATE" not in json.dumps(labels)
    assert labels[0]["corrected_action"] == "inadmissible invented command"


@pytest.mark.parametrize("change", ["length", "order", "type", "multiline", "state_copy"])
def test_bad_corrections_rejected(change):
    value = json.loads(correction())
    if change == "length": value["trajectory"].pop()
    if change == "order": value["trajectory"].reverse()
    if change == "type": value["trajectory"][0]["corrected_reason"] = {}
    if change == "multiline": value["trajectory"][0]["corrected_action"] = "look\nAction: hack"
    if change == "state_copy": value["trajectory"][0]["corrected_reason"] = STATE
    with pytest.raises(ValueError):
        parse_correction(json.dumps(value), 2, privileged_state=STATE)


def test_original_multiturn_history_and_structural_privilege_isolation():
    source = episode()
    original = deepcopy(source)
    labels = parse_correction(correction(), 2)
    rows = step_rows(source, parents=["seed"], corrections=labels)
    assert len(rows) == 2 and source == original
    assert all(r["inadmissible_correction"] for r in rows)
    assert rows[1]["messages"][2] == {"role": "assistant", "content": source["steps"][0]["action_raw"]}
    assert "Check accessible" not in json.dumps(rows[1]["messages"])
    assert rows[1]["target"] == "Thought: Check accessible places 1\nAction: inadmissible invented command"
    content = json.dumps([{k: r[k] for k in ("messages", "target")} for r in rows])
    assert "SECRET" not in content and "PRIVATE" not in content
    assert source["task_id"] not in content
    seed_rows = step_rows(source, parents=["seed"])
    assert [r["target"] for r in seed_rows] == [s["action_raw"] for s in source["steps"]]
    source["split"] = "dev"
    with pytest.raises(ValueError): step_rows(source, parents=["seed"])


def test_min_subsampling_preserves_equal_sets_seed_and_empty_semantics():
    sets = [[{"set": i, "index": j} for j in range(n)] for i, n in enumerate([10, 3, 7, 6])]
    original = deepcopy(sets)
    rows, report = min_subsample(sets)
    assert report["before"] == [10, 3, 7, 6] and report["after"] == [3] * 4
    assert report["N_examples"] == 12 and report["minimum"] == 3
    assert [sum(r["set"] == i for r in rows) for i in range(4)] == [3] * 4
    assert (rows, report) == min_subsample(sets) and rows != min_subsample(sets, seed=1)[0]
    assert sets == original
    empty, report = min_subsample([*sets, []])
    assert empty == [] and report["after"] == [0] * 5


def test_teacher_transport_cache_usage_caps_and_scope(tmp_path, task):
    client = FakeClient([response(correction())])
    limited, task = budget(tmp_path, task, client)
    messages = render("privileged", trajectory(episode()), STATE)
    first = limited.request(task, messages, length=2, privileged_state=STATE)
    assert first["status"] == "accepted" and first["attempts_reserved"] == 1
    assert limited.request(task, messages, length=2, privileged_state=STATE) == first
    assert len(client.requests) == 1
    assert client.requests[0]["temperature"] == .3 and client.requests[0]["top_p"] == 1
    assert client.requests[0]["max_output_tokens"] == 4096
    assert not {"logprobs", "reasoning", "include"} & set(client.requests[0])
    assert first["result"]["metadata"]["returned_model_id"] == "gpt-6-luna"
    assert first["result"]["metadata"]["usage"]["reasoning_tokens"] == 10
    assert read(tmp_path / "sampling.json")["mode"] == "official_accepted"
    assert checked_outcome(first["directory"]) == first
    with pytest.raises(ValueError, match="outside frozen"):
        limited.request(dict(task, task_id="train/new/1"), messages, length=2)
    with pytest.raises(ValueError):
        limited.request(dict(task, split="test", sealed=True), messages, length=2)
    assert len(client.requests) == 1


def test_teacher_cache_covers_privileged_state_and_ablation(tmp_path, task):
    client = FakeClient([response(correction()), response(correction()), response(correction())])
    task = dict(task, parent_seed_ids=["seed"])
    def teacher(messages):
        return LeapTeacher(messages=messages, validator=lambda raw: parse_correction(raw, 2),
            cache_dir=tmp_path / "cache", usage_ledger=tmp_path / "usage.jsonl", run_id="t", client=client)
    context = {"observation": "", "admissible_commands": []}
    messages = render("privileged", trajectory(episode()), STATE)
    a = teacher(messages).generate([], task, context)
    b = teacher(messages).generate([], task, context)
    teacher(render("privileged", trajectory(episode()), STATE + " Changed")).generate([], task, context)
    teacher(render("zero", trajectory(episode()))).generate([], task, context)
    assert not a.metadata["cache_hit"] and b.metadata["cache_hit"] and len(client.requests) == 3


def test_failures_parser_rejections_and_resume_cannot_expand_attempt_cap(tmp_path, task, capsys):
    client = FakeClient([TimeoutError("sk-private-secret"), response("invalid json"), response("still invalid")])
    limited, task = budget(tmp_path, task, client)
    result = limited.request(task, render("privileged", trajectory(episode()), STATE), length=2, privileged_state=STATE)
    assert result["status"] == "exhausted" and result["attempts_reserved"] == 3
    again, task = budget(tmp_path, task, client)
    assert again.request(task, render("privileged", trajectory(episode()), STATE), length=2, privileged_state=STATE) == result
    assert len(client.requests) == 3
    assert "private-secret" not in capsys.readouterr().out
    assert all("private-secret" not in p.read_text() for p in tmp_path.rglob("*") if p.is_file())


@pytest.mark.parametrize("param", ["temperature", "top_p"])
@pytest.mark.parametrize("code", [None, "unsupported_parameter", "unsupported_value"])
@pytest.mark.parametrize("kind", ["extract", "privileged"])
def test_unsupported_sampling_falls_back_within_same_three_attempts(tmp_path, task, param, code, kind):
    class Unsupported(Exception):
        status_code = 400
        body = {"error": {"type": "invalid_request_error", "param": param, "code": code,
                          "message": "PRIVATE_PROVIDER_MESSAGE"}}
    client = FakeClient([Unsupported("PRIVATE_PROVIDER_MESSAGE"), response(STATE if kind == "extract" else correction())])
    limited, task = budget(tmp_path, task, client, kind=kind)
    messages = render(kind, trajectory(episode()), STATE if kind == "privileged" else None)
    kwargs = {"length": 2, "privileged_state": STATE} if kind == "privileged" else {}
    result = limited.request(task, messages, **kwargs)
    assert result["status"] == "accepted" and result["attempts_reserved"] == len(client.requests) == 2
    assert client.requests[0]["temperature"] == .3 and client.requests[0]["top_p"] == 1
    assert "temperature" not in client.requests[1] and "top_p" not in client.requests[1]
    decision = read(tmp_path / "sampling.json")
    assert decision["mode"] == "configured_default" and decision["parameter"] == param
    directory = Path(result["directory"])
    assert decision["evidence"] == str((directory / "attempt-0/usage.jsonl").resolve())
    assert read(directory / "attempt-0/failure.json")["unsupported_sampling"]
    assert read(directory / "attempt-0/start.json")["mode"] == "official_unconfirmed"
    default = dict(limited.config["models"]["teacher"]["sampling"], max_output_tokens=4096)
    start = read(directory / "attempt-1/start.json")
    assert start["mode"] == "configured_default" and start["sampling"] == default
    assert client.requests[1] == {k: v for k, v in client.requests[0].items()
                                  if k not in ("temperature", "top_p")} | default
    assert all("PRIVATE_PROVIDER" not in p.read_text() for p in tmp_path.rglob("*") if p.is_file())


def test_interruption_reserves_a_slot_and_incomplete_response_is_not_cached(tmp_path, task):
    class InterruptClient(FakeClient):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            raise KeyboardInterrupt
    client = InterruptClient()
    limited, task = budget(tmp_path, task, client)
    messages = render("privileged", trajectory(episode()), STATE)
    with pytest.raises(KeyboardInterrupt): limited.request(task, messages, length=2, privileged_state=STATE)
    incomplete = response(correction())
    incomplete.status = "incomplete"
    next_client = FakeClient([incomplete, response(correction())])
    limited, task = budget(tmp_path, task, next_client)
    result = limited.request(task, messages, length=2, privileged_state=STATE)
    assert result["status"] == "accepted" and result["attempts_reserved"] == 3
    assert len(next_client.requests) == 2


@pytest.mark.parametrize("win", [True, False])
def test_expert_executes_in_harness_and_never_exports_oracle_fields(seed_bank, tmp_path, win):
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    from sed.manifest import resolve_task
    root, manifest, freeze, config = seed_bank
    config["environment"]["max_episode_steps"] = 40
    config["paths"].update(manifest=str(tmp_path / "m.json"), split_freeze=str(tmp_path / "f.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    task = dict(resolve_task(manifest, freeze, next(t["task_id"] for t in manifest["tasks"] if t["source_split"] == "train")), parent_seed_ids=["seed"])
    class Backend(FakeBackend):
        plan = ["look"]
        def __init__(self, path, seed):
            super().__init__(path, seed)
            self.win_at = 1 if win else 99
    result = collect_one(task, config, tmp_path / "expert", env_factory=lambda c, t:
        Environment(root, backend_factory=Backend, max_episode_steps=40))
    assert result["available"] is win
    assert result["trusted_trajectories_added"] == 0
    actual = read(tmp_path / "expert/episode/episode.json")
    assert len(actual["steps"]) == (1 if win else 40)
    assert "SECRET" not in json.dumps(trajectory(actual, expert=True))
    assert actual["success"] is win
    with pytest.raises(ValueError):
        collect_one(dict(task, split="dev"), config, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("variant,empty", [("privileged", False), ("zero", False), ("zero", True)])
def test_four_iteration_chain_merged_init_aggregation_resume_and_empty(tmp_path, monkeypatch, variant, empty):
    args = NS(output=str(tmp_path), verified="verified", snapshot="BASE", config="config", variant=variant,
        train_python="TRAIN", env_python="ENV", model_name="gemma", workers=1, through="all")
    commands, oracle_calls = [], []
    monkeypatch.setattr(pipeline, "seeds", lambda *a: [({"task_id": "train/a/1"}, {})])
    def prepare(verified, output, config, *, iteration, **kwargs):
        output.mkdir(parents=True)
        n = 0 if empty and iteration else [10, 5, 7, 3][iteration]
        write_json(output / "examples.json", [{"set": iteration, "i": i} for i in range(n)])
        write_json(output / "report.json", {"inadmissible_corrections": iteration})
    def expert(*args):
        oracle_calls.append("expert")
        args[-1].mkdir()
    def extract(*args):
        oracle_calls.append("extract")
        args[-1].mkdir()
    def correct(*args):
        args[-1].mkdir()
        write_json(args[-1] / "report.json", {"failed_rollouts": 1, "corrections": int(not empty)})
    @contextmanager
    def serve(*args): yield "http://fake/v1"
    def command(cmd, log, **kwargs):
        commands.append(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True)
        write_json(out / "result.json", {"examples_seen": 1, "supervised_tokens_seen": 10})
        if "baselines.leap.merge" in cmd:
            (out / "checkpoint").mkdir()
    monkeypatch.setattr(pipeline, "prepare_set", prepare)
    monkeypatch.setattr(pipeline, "collect_experts", expert)
    monkeypatch.setattr(pipeline, "extract_states", extract)
    monkeypatch.setattr(pipeline, "correct_rollouts", correct)
    monkeypatch.setattr(pipeline, "served", serve)
    monkeypatch.setattr(pipeline, "run_command", command)
    monkeypatch.setattr(pipeline, "update_disclosure", lambda *a: None)
    pipeline.train_pipeline(args, {"K": 5}, {})
    training = [c for c in commands if "baselines.leap.train" in c]
    merges = [c for c in commands if "baselines.leap.merge" in c]
    assert len(training) == len(merges) == (1 if empty else 4)
    for i, cmd in enumerate(training):
        init = cmd[cmd.index("--snapshot") + 1]
        assert init == ("BASE" if i == 0 else str(Path(merges[i - 1][merges[i - 1].index("--output") + 1]) / "checkpoint"))
        assert len(cmd[cmd.index("--sets") + 1:]) == i + 1
        assert cmd[cmd.index("--variant") + 1] == variant
    count = len(commands)
    pipeline.train_pipeline(args, {"K": 5}, {})
    assert len(commands) == count
    assert oracle_calls == (["expert", "extract"] if variant == "privileged" else [])
    result = read(tmp_path / "result.json")
    assert result["status"] == ("stopped_empty_min_aggregate" if empty else "complete")
    if not empty:
        assert read(tmp_path / "iteration-3.json")["aggregation"]["after"] == [3] * 4
    # Hash-bound stages reject changes even when the input path stays the same.
    path = next((tmp_path / "stages/data-0").glob("attempt-*/output/examples.json"))
    write_json(path, [])
    with pytest.raises(ValueError, match="identity/artifacts"):
        pipeline.train_pipeline(args, {"K": 5}, {})


def test_caps_and_launcher():
    caps = pipeline.teacher_caps(20, "privileged")
    assert caps["extraction"] == {"logical_calls": 20, "api_attempts": 60}
    assert list(caps["correction_by_iteration"].values()) == [{"logical_calls": 20, "api_attempts": 60}] * 3
    assert pipeline.teacher_caps(20, "zero")["extraction"]["logical_calls"] == 0
    launch = Path("scripts/hpg_leap.slurm").read_text()
    assert "#SBATCH --gpus=1" in launch and "VLLM_BATCH_INVARIANT=1" in launch
    assert "LEAP_VARIANT" in launch and "LEAP_CORRECTION_TOKENS" in launch


@pytest.mark.parametrize("changed_file", ["baselines/leap/merge.py", "train/merge_lora.py",
    "baselines/leap/DEVIATIONS.md", "tests/test_leap_merge.py", "tests/test_stage3d.py"])
def test_new_merge_code_refuses_frozen_pipeline_resume(tmp_path, monkeypatch, changed_file):
    from baselines.common import pipeline as common
    from sed.seed_evidence import artifact_hashes

    args = NS(output=str(tmp_path / "run"), verified=str(tmp_path / "verified"),
              snapshot=str(tmp_path / "base"), config="config", gpus=1)
    Path(args.snapshot).mkdir()
    write_json(Path(args.snapshot) / "config.json", {"model_type": "gemma4_unified"})
    tasks = [{"task_id": f"train/task-{i}"} for i in range(5)]
    collection = {"K": 5, "unique_instances": 5, "task_groups": 5, "verified": tasks}
    monkeypatch.setattr(common, "load_frozen", lambda *a: ({}, {}))
    monkeypatch.setattr(common, "load_verified_collection", lambda *a:
        (collection, {"K": 5, "seed": 0}, Path(args.verified)))
    monkeypatch.setattr(common, "checked_episode", lambda root, row: {"episode_id": row["task_id"]})
    monkeypatch.setattr(common, "training_tasks", lambda *a: None)
    monkeypatch.setattr(common, "write_provenance", lambda *a, **kw: None)
    config = {"paths": {name: str(tmp_path / name) for name in ("raw", "verified", "manifest", "split_freeze")}}
    recipe = {"variant": "zero"}
    identity = pipeline.prepare(args, config, method="leap", recipe=recipe)
    assert changed_file in identity["code_sha256"]
    root_identity = digest(identity)
    output = Path(args.output)

    def complete(attempt):
        (attempt / "output").mkdir()
        write_json(attempt / "output/result.json", {"complete": True})

    data = pipeline.stage_run(output, "data-0", digest([root_identity, "seeds"]), complete)
    trained = pipeline.stage_run(output, "train-0", digest([root_identity, 0, args.snapshot,
                                str(data / "examples.json")]), complete)

    def fail_merge(attempt):
        raise ValueError("Merged logits exceed tolerance; checkpoint must not be served")

    with pytest.raises(ValueError, match="Merged logits"):
        pipeline.stage_run(output, "merge-0", digest([root_identity, 0, args.snapshot, str(trained)]), fail_merge)
    before = artifact_hashes(output)
    new_hashes = dict(identity["code_sha256"])
    new_hashes[changed_file] = digest("changed code tree")
    monkeypatch.setattr(common, "code_hashes", lambda root: new_hashes)
    # main() calls prepare() before train_pipeline(): it refuses the root
    # identity before even looking up data-0/train-0 or retrying merge-0.
    with pytest.raises(ValueError, match="different frozen artifact: .*identity.json"):
        pipeline.prepare(args, config, method="leap", recipe=recipe)
    assert artifact_hashes(output) == before
    assert not (output / "stages/merge-0/attempt-001").exists()
    # Even bypassing prepare would fail the completed data stage's binding.
    new_identity = dict(identity, code_sha256=new_hashes)
    with pytest.raises(ValueError, match="Completed stage identity/artifacts changed"):
        pipeline.stage_run(output, "data-0", digest([digest(new_identity), "seeds"]), complete)
    assert artifact_hashes(output) == before
    args.output = str(tmp_path / "fresh-run")
    assert pipeline.prepare(args, config, method="leap", recipe=recipe) == new_identity


@pytest.mark.parametrize("variant", ["privileged", "zero"])
def test_checked_seed_rollout_correction_dataset_and_disclosure(seed_bank, tmp_path, monkeypatch, variant):
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    from teachers.collect_seeds import collect
    from sed.seed_selection import select_seeds
    from sed.runner import ScriptedPolicy
    from baselines.common.explore import run_exploration, explore_one
    from sed.storage import file_hash
    root, manifest, freeze, config = seed_bank
    config["paths"].update(manifest=str(tmp_path / "manifest.json"), split_freeze=str(tmp_path / "freeze.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    env, teacher = factories(root, config, set(), [])
    verified = tmp_path / "verified"
    collection = collect(select_seeds(manifest, freeze, k=5), manifest, freeze, config, verified,
        env_factory=env, teacher_factory=teacher)
    config["environment"]["max_episode_steps"] = 40
    bank = seeds(verified, config)
    tasks = [t for t, _ in bank]
    ids = [t["task_id"] for t in tasks]
    listing = tmp_path / "tasks.json"
    write_json(listing, {"task_ids": ids, "list_sha256": digest(ids), "manifest_sha256": digest(manifest),
        "split_freeze_sha256": digest(freeze), "parent_seed_ids": {t["task_id"]: t["parent_seed_ids"] for t in tasks}})
    class FailedBackend(FakeBackend):
        def step(self, action):
            self.position += 1
            state = self.state()
            state.update(won=False, lost=self.position == 2)
            return state, 0., state["lost"]
    def worker(job):
        return explore_one(job, env_factory=lambda c, t: Environment(root, max_episode_steps=40, backend_factory=FailedBackend),
            policy_factory=lambda **kw: ScriptedPolicy(["look"]))
    exploration = tmp_path / "exploration"
    run_exploration(config=config, tasks_path=listing, output=exploration, spec={"enable_thinking": False, "max_tokens": 1024},
        decoding="greedy", worker_fn=worker)
    args = NS(output=str(tmp_path / "run"), variant=variant, correction_tokens=4096)
    states = tmp_path / "states" if variant == "privileged" else None
    if states:
        extraction_client = FakeClient([response(STATE) for _ in range(4)])
        extractor = TeacherBudget(Path(args.output) / "audit/teacher/extraction", tasks=tasks, config=config,
            kind="extract", iteration=0, run_identity="test", max_output_tokens=4096,
            teacher_factory=factory(extraction_client), sampling_root=Path(args.output) / "audit/teacher")
        state_rows = [{"task_id": tasks[0]["task_id"], "state": None, "status": "expert_unavailable"}]
        for t in tasks[1:]:
            result = extractor.request(t, render("extract", trajectory(episode(), expert=True)))
            state_rows.append({"task_id": t["task_id"], "state": result["labels"], "status": result["status"],
                "request_directory": result["directory"], "outcome_sha256": file_hash(Path(result["directory"]) / "outcome.json")})
        write_json(states / "states.json", state_rows)
    expected = 4 if states else 5
    client = FakeClient([response(correction()) for _ in range(expected)])
    monkeypatch.setattr(pipeline, "TeacherBudget", lambda *a, **kw: TeacherBudget(*a, teacher_factory=factory(client), **kw))
    identity = {"K": 5, "unique_instances": 5, "task_groups": 5,
                "recipe": {"variant": variant, "label": pipeline.LABELS[variant]}}
    corrected = tmp_path / "corrections"
    pipeline.correct_rollouts(args, config, identity, tasks, exploration, states, 1, corrected)
    report = read(corrected / "report.json")
    assert report["failed_rollouts"] == 5 and report["corrections"] == expected
    assert report["excluded_no_privileged_state"] == 5 - expected
    prepare_set(verified, tmp_path / "set0", config, iteration=0)
    prepare_set(verified, tmp_path / "set1", config, iteration=1, exploration=exploration, corrections=corrected)
    d0 = load_set(tmp_path / "set0/examples.json", config, expected_iteration=0, verified=verified)
    d1 = load_set(tmp_path / "set1/examples.json", config, expected_iteration=1, verified=verified)
    assert len(d0) == collection["K"] == 5  # One-step fake seeds; ALL K remain.
    assert len(d1) == 2 * expected and all(r["inadmissible_correction"] for r in d1)
    assert "SECRET" not in json.dumps(d1) and "PRIVATE" not in json.dumps(d1)
    assert d1[1]["messages"][2]["content"] == "Action: look"
    assert len(client.requests) == expected
    if states:
        assert all(STATE in r["input"][1]["content"] for r in client.requests)
    else:
        assert all("PRIVATE" not in json.dumps(r["input"]) for r in client.requests)
    pipeline.update_disclosure(args.output, identity)
    totals = read(Path(args.output) / "disclosure.json")["totals"]
    total_calls = expected + (4 if states else 0)
    assert totals["teacher_calls"] == totals["teacher_logical_calls"] == total_calls
    assert totals["teacher_input_tokens"] == 13 * total_calls
    assert totals["teacher_output_tokens"] == 17 * total_calls
    assert totals["teacher_unknown_usage"] == 0
    # Data/source tampering cannot silently enter training.
    d1[0]["target"] = "Thought: injected\nAction: look"
    write_json(tmp_path / "set1/examples.json", d1)
    with pytest.raises(ValueError, match="original source"):
        load_set(tmp_path / "set1/examples.json", config, expected_iteration=1, verified=verified)
    with pytest.raises(ValueError, match="chain"):
        load_set(tmp_path / "set0/examples.json", config, expected_iteration=1, verified=verified)


def test_initialization_rejects_wrong_iteration_variant_or_unmerged_adapter(monkeypatch):
    from baselines.leap import train
    state = {"leap_iteration": 0, "leap_variant": "privileged", "fresh_lora": True, "adaptation": "leap_merged_lora"}
    monkeypatch.setattr(train, "validate_checkpoint", lambda *a: state)
    train.validate_initialization("base", "base", iteration=0, variant="privileged", student={})
    train.validate_initialization("iter0", "base", iteration=1, variant="privileged", student={})
    with pytest.raises(ValueError, match="configured base"):
        train.validate_initialization("iter0", "base", iteration=0, variant="privileged", student={})
    with pytest.raises(ValueError, match="previous merged"):
        train.validate_initialization("iter0", "base", iteration=2, variant="privileged", student={})
    with pytest.raises(ValueError, match="same variant"):
        train.validate_initialization("iter0", "base", iteration=1, variant="zero", student={})
    state["adaptation"] = "leap_lora"
    with pytest.raises(ValueError, match="previous merged"):
        train.validate_initialization("adapter", "base", iteration=1, variant="privileged", student={})


def test_disclosure_counts_failed_rejected_and_interrupted_usage(tmp_path, task):
    client = FakeClient([TimeoutError("private"), response("invalid"), response("invalid")])
    limited, task = budget(tmp_path / "audit/teacher", task, client)
    limited.request(task, render("privileged", trajectory(episode()), STATE), length=2, privileged_state=STATE)
    identity = {"K": 5, "unique_instances": 5, "task_groups": 5,
                "recipe": {"variant": "privileged", "label": pipeline.LABELS["privileged"]}}
    pipeline.update_disclosure(tmp_path, identity)
    report = read(tmp_path / "disclosure.json")["totals"]
    assert report["teacher_calls"] == report["teacher_attempt_slots_reserved"] == 3
    assert report["teacher_logical_calls"] == 1 and report["teacher_failed_attempts"] == 1
    assert report["teacher_returned"] == report["teacher_usage_known"] == 2
    assert report["teacher_unknown_usage"] == 1 and report["teacher_input_tokens"] == 26
