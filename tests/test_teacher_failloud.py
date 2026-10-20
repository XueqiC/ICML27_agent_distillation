"""Failure/rejection/resume contracts at real stage boundaries; transports are stubs."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from baselines.eto.pipeline import stage_run
from sed.config import load_config
from sed.manifest import resolve_task
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, write_json
from teachers.openai_teacher import TeacherCallError
from teachers.recover import recover
from test_capped_teacher import RawClient, capped, http_429
from test_stage2b_seeds import seed_bank
from test_full_train import full_bank
from test_rollout_pool import rollout
from test_stage3d import STATE, correction, episode, factory
from test_teacher import FakeClient, response


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def bank(seed_bank, tmp_path):
    root, manifest, freeze, config = seed_bank
    config["environment"]["max_episode_steps"] = 40
    config["models"]["teacher"]["rate_limit"] = {"requests_per_minute": 100000, "tokens_per_minute": 100000000}
    for name, value in (("manifest", manifest), ("split_freeze", freeze)):
        config["paths"][name] = str(tmp_path / (name + ".json"))
        write_json(config["paths"][name], value)
    task = next(resolve_task(manifest, freeze, t["task_id"]) for t in manifest["tasks"] if t["source_split"] == "train")
    task["parent_seed_ids"] = ["seed"]
    return root, task, config


def transaction(root, name, action):
    return stage_run(root, name, "frozen-test-identity", lambda a: action(a / "output"))


def assert_failed(root, name):
    stage = root / "stages" / name
    assert not (stage / "complete.json").exists()
    report = read(stage / "attempt-000/output/teacher_report.json")
    assert report["status"] == "failed" and report["transport_failures"] == 1
    assert report["rejections"] == 0
    assert read(stage / "attempt-000/failure.json")["status"] == "failed"


@pytest.mark.parametrize("kind", ["extract", "correct"])
@pytest.mark.parametrize("failure", ["429", "timeout", "incomplete", "unknown"])
def test_leap_stage_failure_resume_and_rejection(bank, tmp_path, monkeypatch, kind, failure):
    from baselines.leap import pipeline as p
    from baselines.leap.teacher import TeacherBudget
    _, task, config = bank
    root = tmp_path / "cell"
    args = NS(output=str(root), variant="zero", correction_tokens=4096, extraction_tokens=4096)
    source = tmp_path / "source"
    ep = episode()
    ep.update(task_id=task["task_id"], success=True)
    write_json(source / "episode/episode.json", ep)
    receipt = {"task_id": task["task_id"], "available": True, "artifact_sha256": artifact_hashes(source)}
    write_json(source / "receipt.json", receipt)
    experts = tmp_path / "experts"
    write_json(experts / "experts.json", [{**receipt, "directory": str(source),
        "receipt_sha256": file_hash(source / "receipt.json")}])
    exploration = tmp_path / "exploration"
    write_json(exploration / "receipts.json", [])
    monkeypatch.setattr(p, "checked_rollouts", lambda *a: [({"task_id": task["task_id"], "won": False,
        "parent_seed_ids": task["parent_seed_ids"]}, ep)])
    good = STATE if kind == "extract" else correction(action="look")
    bad = http_429() if failure == "429" else TimeoutError("SECRET_ERROR")
    if failure in {"incomplete", "unknown"}:
        bad = response(good)
        bad.status = "incomplete" if failure == "incomplete" else None
    client = FakeClient([bad, response(good)])
    monkeypatch.setattr(p, "TeacherBudget", lambda *a, **kw: TeacherBudget(*a, teacher_factory=factory(client), **kw))
    def action(output):
        if kind == "extract":
            p.extract_states(args, config, {}, [task], experts, output)
        else:
            p.correct_rollouts(args, config, {}, [task], exploration, None, 1, output)
    with pytest.raises(TeacherCallError, match="NOT complete"):
        transaction(root, kind, action)
    assert_failed(root, kind)
    assert not list((root / "audit/teacher").rglob("outcome.json"))
    result = transaction(root, kind, action)
    assert read(result / "report.json")["transport_failures"] == 0
    assert len(client.requests) == 2
    assert list((root / "audit/teacher").rglob("failure.json"))
    # All three official malformed-output attempts remain rejections.
    args.output = str(tmp_path / "format-cell")
    client.results = [response("malformed output") for _ in range(3)]
    result = transaction(Path(args.output), kind, action)
    report = read(result / "report.json")
    assert report["transport_failures"] == 0 and report["rejections"] == 3
    assert report["extraction_failed" if kind == "extract" else "rejected_corrections"] == 1


@pytest.mark.parametrize("method", ["gad", "had", "steca"])
def test_label_stage_failure_resume_and_format(bank, tmp_path, monkeypatch, method):
    import importlib
    from baselines.had.data import DrawTeacher
    from baselines.steca.data import ReflectionTeacher
    from test_wkm_steca import planned_rows
    m = importlib.import_module("baselines." + method + ".data")
    _, task, config = bank
    root, cell = tmp_path / "transaction", tmp_path / "cell"
    exploration, verified = tmp_path / "exploration", tmp_path / "verified"
    write_json(exploration / "receipts.json", [])
    write_json(verified / "manifest.json", {})
    state = {"state_id": "one", "task_id": task["task_id"], "parent_seed_ids": task["parent_seed_ids"],
        "history": [], "visible": {"observation": "room", "admissible_commands": ["look"]}}
    rows = planned_rows()
    rows[0]["task_id"] = task["task_id"]
    if method == "steca":
        monkeypatch.setattr(m, "seeds", lambda *a: [(task, {})])
    else:
        monkeypatch.setattr(m, "visited_states", lambda *a: [state])
    client = RawClient(error=TimeoutError("SECRET_ERROR"))
    t = capped(cell, client, cap=5, teacher_class=ReflectionTeacher if method == "steca" else DrawTeacher)
    def action(output):
        if method == "steca":
            return m.reflect(rows, config, verified, output, cell, teacher_factory=lambda *a, **kw: t)
        return m.collect(exploration, verified, output, config, cell=cell,
                         identity={"unique_instances": 1}, round_index=0, teacher=t)
    with pytest.raises(TeacherCallError, match="NOT complete"):
        transaction(root, method, action)
    assert_failed(root, method)
    before = t._read_counter()["attempts"]
    recover(t.counter_path, reason="transport_fixed")
    assert t._read_counter()["attempts"] == before
    client.error = None
    client.texts = ["Thought: inspect.\nAction: look"] * 4
    result = transaction(root, method, action)
    assert read(result / "teacher_report.json")["transport_failures"] == 0
    # A fresh run keeps the method's original handling of invalid OUTPUT.
    t = capped(tmp_path / "format-teacher", RawClient(texts=["malformed"] * 4), cap=5,
               teacher_class=ReflectionTeacher if method == "steca" else DrawTeacher)
    if method == "steca":
        t.teacher.client.client.texts = ["Action: go north"]
        with pytest.raises(ValueError):
            transaction(tmp_path / "format", method, action)
        report = read(tmp_path / "format/stages/steca/attempt-000/output/teacher_report.json")
    else:
        out = transaction(tmp_path / "format", method, action)
        report = read(out / "report.json")
        assert len(read(out / "data.json")["rows"]) == 1  # Preserve raw-label/masking algorithm.
    assert report["transport_failures"] == 0 and report["rejections"] > 0


@pytest.mark.parametrize("evaluation", [False, True])
def test_coa_stage_failure_resume_and_format(bank, tmp_path, evaluation):
    from baselines.coa.data import collect
    from baselines.coa.teacher import budget
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    root, task, config = bank
    if evaluation:
        manifest = read(config["paths"]["manifest"])
        freeze = read(config["paths"]["split_freeze"])
        task = next(resolve_task(manifest, freeze, t["task_id"]) for t in manifest["tasks"]
                    if t["source_split"] == "valid_seen")
    listing = {"parent_seed_ids": {task["task_id"]: ["seed"]}}
    def env(*a):
        def backend(path, seed):
            result = FakeBackend(path, seed)
            result.win_at = 1
            return result
        return Environment(root, max_episode_steps=40, backend_factory=backend, split=task["split"])
    class Client(RawClient):
        malformed = False
        def create(self, **kwargs):
            role = json.loads(kwargs["input"][-1]["content"])["role"]
            raw = json.dumps({"text": "Inspect", "next": "alfworld"}) if role == "think" else "<alfworld>Action: look</alfworld>"
            self.texts.insert(0, "malformed" if self.malformed else raw)
            return super().create(**kwargs)
    client = Client(error=TimeoutError("SECRET_ERROR"))
    cell = tmp_path / "cell"
    t = budget(config, cell / "transport", {}, [task], evaluation=evaluation, client=client)
    name = "teacher-evaluate" if evaluation else "teacher"
    def action(output):
        collect(config, listing, [task], output, cell, {}, evaluation=evaluation, env_factory=env, capped=t)
    with pytest.raises(TeacherCallError, match="NOT complete"):
        transaction(cell, name, action)
    assert_failed(cell, name)
    assert not list((cell / ("teacher_eval" if evaluation else "teacher")).rglob("receipt.json"))
    recover(t.counter_path, reason="transport_fixed")
    client.error = None
    result = transaction(cell, name, action)
    assert all(r["status"] == "ok" for r in read(result / "receipts.json"))
    assert list(cell.rglob("transport_failure.json"))
    cell = tmp_path / "format"
    client.malformed = True
    t = budget(config, cell / "transport", {}, [task], evaluation=evaluation, client=client)
    result = transaction(cell, name, action)
    report = read(result / "teacher_report.json")
    assert report["transport_failures"] == 0 and report["rejections"] > 0


@pytest.mark.parametrize("kind", ["collect", "abstract", "cluster", "consolidate"])
def test_agentdistill_stage_failure_resume_and_format(bank, tmp_path, monkeypatch, kind):
    from baselines.agentdistill import collection, box
    from baselines.agentdistill.teacher import Budget
    from test_agentdistill import tool, env_factory, budget_factory
    root, task, config = bank
    cell, identity = tmp_path / "cell", {"unique_instances": 1}
    write_json(cell / "identity.json", identity)
    write_json(cell / "tasks.json", [task["task_id"]])
    source = tmp_path / "source"
    row = {"id": "one", "tool": tool(), "task_id": task["task_id"], "parent_seed_ids": ["seed"],
        "run": 0, "source_directory": "unused", "receipt_sha256": "unused", "raw_tool_sha256": "unused"}
    write_json(source / "pool.json", [row])
    write_json(source / "abstracted.json", [row])
    write_json(source / "clusters.json", [{"name": "move", "members": ["one"]}])
    monkeypatch.setattr(box, "checked_pool", lambda *a: [row])
    good = {"collect": ['Action: create_mcp {"request":"Look once"}', json.dumps({"tool": tool(), "arguments": {"command": "look"}})],
            "abstract": [json.dumps(tool())], "cluster": [json.dumps({"clusters": [{"name": "move", "members": ["one"]}]})],
            "consolidate": [json.dumps(tool())]}[kind]
    client = RawClient(texts=good, error=TimeoutError("SECRET_ERROR"))
    def fast_factory(*args, **kwargs):
        value = budget_factory(client)(*args, **kwargs)
        value.teacher.limiter.limits = {"requests_per_minute": 100000, "tokens_per_minute": 100000000}
        return value
    b = Budget(cell / "teacher", config, identity, factory=fast_factory)
    def action(output):
        if kind == "collect":
            collection.collect([task], config, cell, output, identity, b, env_factory=env_factory(root, 1))
        else:
            box.build(kind, source, output, [task], b, config)
    with pytest.raises(TeacherCallError, match="NOT complete"):
        transaction(cell, kind, action)
    assert_failed(cell, kind)
    recover(b.client.counter_path, reason="transport_fixed")
    client.error = None
    result = transaction(cell, kind, action)
    assert read(result / "teacher_report.json")["transport_failures"] == 0
    cell = tmp_path / "format"
    write_json(cell / "identity.json", identity)
    write_json(cell / "tasks.json", [task["task_id"]])
    client = RawClient(texts=["malformed"] * 125)
    b = Budget(cell / "teacher", config, identity, factory=fast_factory)
    if kind in {"cluster", "consolidate"}:
        with pytest.raises(ValueError):
            transaction(cell, kind, action)
        result = cell / "stages" / kind / "attempt-000/output"
    else:
        result = transaction(cell, kind, action)
    report = read(result / "teacher_report.json")
    assert report["transport_failures"] == 0 and report["rejections"] > 0


@pytest.mark.parametrize("full", [False, True])
def test_seed_collectors_transport_format_and_resume(seed_bank, full_bank, tmp_path, full):
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    from sed.seed_selection import select_seeds, select_full_train
    from teachers.collect_seeds import collect
    from teachers.openai_teacher import Teacher
    root, manifest, freeze, config = seed_bank
    config["environment"]["max_episode_steps"] = 1
    if full:
        root, manifest, freeze, _, config = full_bank
    selection = select_full_train(manifest, freeze) if full else select_seeds(manifest, freeze, k=1)
    def env(task):
        def backend(path, seed):
            result = FakeBackend(path, seed)
            result.win_at = 1
            return result
        return Environment(root, max_episode_steps=1, backend_factory=backend)
    client = FakeClient([TimeoutError("SECRET_ERROR")])
    def teacher(task, cache, ledger):
        return Teacher(cache_dir=cache, usage_ledger=ledger, run_id="test", client=client,
                       max_attempts=1, sleep=lambda _: None)
    output = tmp_path / "collection"
    with pytest.raises(TeacherCallError, match="NOT complete"):
        collect(selection, manifest, freeze, config, output, env_factory=env, teacher_factory=teacher)
    assert read(output / "manifest.json")["status"] == "transport_failed"
    assert read(output / "teacher_report.json")["transport_failures"] == 1
    assert read(output / "teacher_report.json")["rejections"] == 0
    old = list(output.rglob("transport_failure.json"))
    assert len(old) == 1
    client.results = [response()] * (len(manifest["tasks"]) + 1)
    result = collect(selection, manifest, freeze, config, output, resume=True, env_factory=env, teacher_factory=teacher)
    assert result["status"] == "complete" and result["trusted_trajectories"] > 0
    assert result["transport_failures"] == 1 and result["transport_failure_evidence"]
    assert old[0].exists()
    client.results = [response("malformed")] * (3 * len(manifest["tasks"]) + 1)
    rejected = tmp_path / "format"
    collect(selection, manifest, freeze, config, rejected, env_factory=env, teacher_factory=teacher)
    report = read(rejected / "teacher_report.json")
    assert report["transport_failures"] == 0 and report["rejections"] > 0


def test_adftp_transport_resume_and_format(seed_bank, tmp_path):
    from baselines.adftp.collect import collect
    from test_adftp import setup_source, stub_factory
    from sed.seed_evidence import load_verified_collection
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=1)
    task_id = load_verified_collection(source, manifest, freeze)[0]["verified"][0]["task_id"]
    calls, failed = [], {task_id}
    output = tmp_path / "ftp"
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env,
                  teacher_factory=stub_factory(calls, fail=failed))
    with pytest.raises(TeacherCallError):
        collect(source, manifest, freeze, config, output, **kwargs)
    assert not list((output / "stages").rglob("complete.json"))
    assert read(output / "teacher_report.json")["transport_failures"] == 1
    assert not (output / "collection/manifest.json").exists()
    counter = next((output / "work").rglob("counter.json"))
    recover(counter, reason="transport_fixed")
    failed.clear()
    result = collect(source, manifest, freeze, config, output, **kwargs)
    assert read(result / "manifest.json")["fallback_count"] == 0
    assert list(output.rglob("transport_failure.json"))
    kwargs["teacher_factory"] = stub_factory(calls, mismatch={task_id})
    collect(source, manifest, freeze, config, tmp_path / "format", **kwargs)
    report = read(tmp_path / "format/teacher_report.json")
    assert report["transport_failures"] == 0 and report["rejections"] == 3


@pytest.mark.parametrize("error", ["halted", "cap"])
def test_capped_refuses_completion_even_without_a_transport_call(tmp_path, task, visible, error):
    from teachers.capped import CollectorHalted, BudgetExceeded
    from teachers.stage import teacher_stage
    t = capped(tmp_path / "teacher", RawClient(), cap=0 if error == "cap" else 2)
    if error == "halted":
        write_json(t.counter_path, {"identity": t.identity, "attempts": 1, "pending": True, "halted": False})
    @teacher_stage("output", name="stub.collect")
    def action(output):
        t.generate([], task, visible)
    with pytest.raises(CollectorHalted if error == "halted" else BudgetExceeded):
        transaction(tmp_path / "stage", "collect", action)
    assert_failed(tmp_path / "stage", "collect")


def test_recovery_never_refunds_caps_or_accepts_modified_history(tmp_path, task, visible):
    from teachers.capped import BudgetExceeded
    client = RawClient(error=TimeoutError("SECRET_ERROR"))
    t = capped(tmp_path, client, cap=2)
    with pytest.raises(TeacherCallError):
        t.generate([], task, visible)
    recover(t.counter_path, reason="transport_fixed")
    assert t._read_counter()["attempts"] == 1
    client.error = None
    t.generate([], task, visible)
    with pytest.raises(BudgetExceeded):
        recover(t.counter_path, reason="transport_fixed")
    rows = t.teacher.ledger.path.read_text().splitlines()
    t.teacher.ledger.path.write_text("\n".join(rows[1:]) + "\n")
    with pytest.raises(TeacherCallError, match="ledger changed"):
        t.generate([], task, visible)


# Spawn-safe worker also exercises the evaluation worker's broad exception guard.
def failed_evaluation_worker(job, outcome=None):
    import socket
    import eval.run_suite as suite
    from harness.alfworld_env import Environment
    from conftest import FakeBackend
    from teachers.openai_teacher import Teacher
    def forbidden(*args, **kwargs):
        raise AssertionError("No network in stub evaluation worker")
    socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = forbidden
    socket.getaddrinfo = socket.create_connection = forbidden
    suite._environment = lambda config, task: Environment(config["environment"]["data_root"],
        split=task["split"], max_episode_steps=1, backend_factory=FakeBackend)
    suite.Teacher = lambda **kwargs: Teacher(**kwargs, client=FakeClient([
        TimeoutError("SECRET_ERROR") if outcome is None else outcome]))
    return suite._worker(job)


def malformed_evaluation_worker(job):
    from test_teacher import response
    return failed_evaluation_worker(job, response("malformed output"))


def test_teacher_evaluation_transport_fails_and_rerun_completes(data_root, tmp_path):
    from test_suite import fixture_list, fake_teacher_worker
    from eval.run_suite import run_suite
    tasks, manifest, freeze = fixture_list(data_root, tmp_path)
    config = load_config()
    config["models"]["teacher"]["max_attempts"] = 1
    config["environment"].update(data_root=str(data_root), max_episode_steps=1)
    for name, value in (("manifest", manifest), ("split_freeze", freeze)):
        config["paths"][name] = str(tmp_path / (name + ".json"))
        write_json(config["paths"][name], value)
    kwargs = dict(config=config, tasks_path=tasks, policy="teacher", workers=1)
    output = tmp_path / "failed"
    with pytest.raises(TeacherCallError):
        run_suite(**kwargs, output=output, worker_fn=failed_evaluation_worker)
    assert not (output / "summary.json").exists()
    assert read(output / "teacher_report.json")["transport_failures"] >= 1
    result = run_suite(**kwargs, output=tmp_path / "rerun", worker_fn=fake_teacher_worker)
    assert result["completed_episodes"] == 4
    run_suite(**kwargs, output=tmp_path / "format", worker_fn=malformed_evaluation_worker)
    report = read(tmp_path / "format/teacher_report.json")
    assert report["transport_failures"] == 0 and report["rejections"] == 4
    assert read(tmp_path / "format/summary.json")["rejections"] == 4


def test_unknown_partial_execution_ledger_fails_before_fresh_retry(tmp_path):
    from teachers.stage import execution_attempt, teacher_stage
    directory = tmp_path / "executions/original"
    write_json(directory / "started.json", {"status": "started"})
    write_json(directory / "episode/episode.json", {"complete": True, "status": "ok"})
    ledger = directory / "teacher/usage.jsonl"
    ledger.parent.mkdir()
    ledger.write_text('{"status": "started",')
    @teacher_stage("output", name="unknown-execution")
    def action(output):
        return execution_attempt(directory, start="started.json")
    with pytest.raises(TeacherCallError, match="NOT complete"):
        action(tmp_path / "report")
    assert read(tmp_path / "report/teacher_report.json")["transport_failures"] == 1
    assert ledger.read_text() == '{"status": "started",'
    assert action(tmp_path / "report") == directory.parent / "transport_retries/original/attempt-1"


@pytest.mark.parametrize("kind", ["corrections", "rewrites"])
def test_tool_collectors_fail_and_resume(rollout, tmp_path, kind):
    from tools.collect_corrections import collect as corrections, state_ref
    from tools.collect_rewrites import collect as rewrites, RewriteTeacher
    from teachers.openai_teacher import Teacher
    from test_collect_rewrites import rewrite_reply
    manifest, freeze, _, _, _, records = rollout
    ep = read(records[0][1] / "episode.json")
    client = RawClient(error=TimeoutError("SECRET_ERROR"))
    teacher = capped(tmp_path / "teacher", client, cap=3,
                     teacher_class=Teacher if kind == "corrections" else RewriteTeacher)
    def action(output):
        if kind == "corrections":
            corrections(records, [state_ref(records[0][0]["states"][0])], teacher, output, manifest, freeze)
        else:
            rewrites([ep], teacher, output, manifest, freeze)
    root = tmp_path / "stage"
    with pytest.raises(TeacherCallError):
        transaction(root, kind, action)
    assert_failed(root, kind)
    recover(teacher.counter_path, reason="transport_fixed")
    client.error = None
    client.texts = ["Action: look" if kind == "corrections" else rewrite_reply(ep)]
    result = transaction(root, kind, action)
    assert read(result / "teacher_report.json")["transport_failures"] == 0


def test_corrections_resume_keeps_completed_labels_byte_identical(rollout, tmp_path):
    from tools.collect_corrections import collect, state_ref
    from test_capped_teacher import SequenceClient
    manifest, freeze, _, _, _, records = rollout
    states = [state_ref(s) for s in records[0][0]["states"]]
    client = SequenceClient([RawClient().create(), TimeoutError("SECRET_ERROR"), RawClient().create()])
    teacher = capped(tmp_path / "teacher", client, cap=3)
    output = tmp_path / "labels"
    with pytest.raises(TeacherCallError):
        collect(records, states, teacher, output, manifest, freeze)
    previous = {p: p.read_bytes() for p in (output / "episodes").rglob("episode.json")}
    assert len(previous) == 1
    recover(teacher.counter_path, reason="transport_fixed")
    result = collect(records, states, teacher, output, manifest, freeze)
    assert result["episode_count"] == len(states)
    assert all(p.read_bytes() == content for p, content in previous.items())
    assert len(client.requests) == 3


def test_agentdistill_cached_unknown_response_cannot_supply_a_label(tmp_path, task):
    from baselines.agentdistill.teacher import Budget
    from test_agentdistill import budget_factory, tool
    task = {**task, "source_split": "train", "parent_seed_ids": ["seed"]}
    budget = Budget(tmp_path, {"models": {"teacher": {"sampling": {}}}},
                    {"unique_instances": 1}, factory=budget_factory(RawClient(texts=[json.dumps(tool())])))
    args = (["abstract", "one"], "abstract", task, "Abstract tool")
    budget.request(*args)
    path = next(tmp_path.glob("requests/abstract/*/result.json"))
    saved = read(path)
    saved["response"]["metadata"].pop("response_status")
    saved["result_sha256"] = digest(saved["response"])
    write_json(path, saved)
    with pytest.raises(TeacherCallError, match="incomplete/unknown"):
        budget.request(*args)


def test_adftp_action_parse_rejection_is_separate_from_transport(tmp_path, task, visible):
    from baselines.adftp.teacher import FTPPolicy, FTPTeacher
    from teachers.stage import teacher_stage
    task = {**task, "parent_seed_ids": ["seed"]}
    teacher = capped(tmp_path / "teacher", RawClient(texts=["Find the cup.", "Thought: Find the cup."]),
                     cap=2, teacher_class=FTPTeacher)
    @teacher_stage("output", name="adftp-action")
    def action(output):
        return FTPPolicy(teacher, task, output).generate([], task, visible)
    action(tmp_path / "action")
    report = read(tmp_path / "action/teacher_report.json")
    assert report["rejections"] == report["parser_rejections"] == 1
    assert report["transport_failures"] == 0


def test_adftp_keeps_format_rejection_when_later_transport_fails(seed_bank, tmp_path):
    from baselines.adftp.collect import collect
    from test_adftp import setup_source, stub_factory
    from teachers.capped import CappedTeacher
    from baselines.adftp.teacher import FTPTeacher
    from test_capped_teacher import SequenceClient
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=1)
    texts = ["Find the cup.", "Action: look", "Find the cup.",
             "Thought: Find the cup. Continue.\nAction: look"]
    raws = [RawClient(texts=[text]).create() for text in texts]
    client = SequenceClient([raws[0], raws[1], TimeoutError("SECRET_ERROR"), raws[2], raws[3]])
    def factory(config, directory, task, identity, rate_limit_dir):
        return CappedTeacher(counter_path=directory / "counter.json", cap=123,
            cache_dir=directory / "cache", usage_ledger=directory / "usage.jsonl", run_id=task["task_id"],
            teacher_class=FTPTeacher, client=client, sampling={"max_output_tokens": 1024}, sleep=lambda _: None)
    output = tmp_path / "ftp"
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env, teacher_factory=factory)
    with pytest.raises(TeacherCallError):
        collect(source, manifest, freeze, config, output, **kwargs)
    rejection = next((output / "work").glob("*/attempt-1/rejection.json"))
    before = rejection.read_bytes()
    recover(next((output / "work").rglob("counter.json")), reason="transport_fixed")
    result = collect(source, manifest, freeze, config, output, **kwargs)
    receipt = read(result / "manifest.json")["verified"][0]
    assert receipt["ftp_attempts"][0]["rejection"] == "first_reply_prefix_mismatch"
    assert receipt["ftp_attempts"][1]["won"] is True
    assert rejection.read_bytes() == before and len(client.requests) == 5
