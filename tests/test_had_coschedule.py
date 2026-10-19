"""No GPU, sockets, environment backend, or teacher transport is used here."""
from contextlib import contextmanager
import json
import multiprocessing
from pathlib import Path
import random
import shutil
import signal
import time
from types import SimpleNamespace as NS

import pytest

from baselines.had import coschedule as co
from baselines.had import pipeline as p
from sed.storage import JsonlLedger, locked, write_json


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def cell(tmp_path, monkeypatch):
    output = tmp_path / "cell"
    base = tmp_path / "base"
    write_json(base / "training_state.json", {"weight": 0})
    listing = {"task_ids": ["train/a", "train/b"],
               "parent_seed_ids": {"train/a": ["seed-a"], "train/b": ["seed-b"]}}
    identity = {"K": 2, "unique_instances": 2, "task_groups": 1, "method": "had"}
    args = NS(output=str(output), snapshot=str(base), verified="verified", gpus=4,
        config="config", env_python="env-python", train_python="train-python", vllm_python="vllm-python",
        workers=1, model_name="model", through="explore-0", worker="gpu",
        gpu_wait_seconds=5, poll_seconds=.005, poll_max_seconds=.02)
    control = {}

    def checkpoint(name, point):
        owner = "teacher" if name.startswith("teacher-") else "gpu"
        if control.get("kill") == (name, point) and args.worker == owner:
            write_json(output / "scheduling/kill-ready.json", {"name": name, "point": point})
            while True:
                time.sleep(.01)

    def record(name):
        JsonlLedger(output / "scheduling/stub-stages.jsonl").append({"stage": name})

    def prepare(*_, **__):
        write_json(output / "identity.json", identity)
        write_json(output / "tasks.json", listing)
        return identity

    @contextmanager
    def served(args, snapshot, attempt):
        assert args.worker == "gpu"
        write_json(attempt / "server.json", {"snapshot": str(snapshot),
            "weight": read(Path(snapshot) / "training_state.json")["weight"]})
        yield "stub-endpoint"

    def command(cmd, log, **kw):
        assert args.worker == "gpu"
        target = Path(cmd[cmd.index("--output") + 1])
        attempt = target.parent
        name = attempt.parent.name
        checkpoint(name, "before_output")
        record(name)
        if "baselines.had.rollout" in cmd:
            assert kw["env"]["CUDA_VISIBLE_DEVICES"] == ""
            write_json(target / "rollout.json", {"args": cmd, **read(attempt / "server.json")})
        else:
            labels = read(cmd[cmd.index("--had-data") + 1])
            snapshot = cmd[cmd.index("--snapshot") + 1]
            version = int(cmd[cmd.index("--had-round") + 1])
            start = int(cmd[cmd.index("--start-update") + 1])
            weight = read(Path(snapshot) / "training_state.json")["weight"]
            assert labels["weight"] == weight  # Exact current-policy input.
            assert labels["round"] == version
            result = {"start_update": start, "end_update": min(250, start + 100),
                      "examples_seen": 4, "unused_at_terminal_budget": [], "weight": weight + version + 1}
            write_json(target / "checkpoint/training_state.json", result)
            write_json(target / "result.json", result)
            write_json(target / "inputs.json", {"args": cmd, "labels": labels})
        checkpoint(name, "after_output")

    def collect(explored, verified, target, config, **kw):
        assert args.worker == "teacher"
        name = target.parent.parent.name
        checkpoint(name, "before_output")
        record(name)
        rollout = read(explored / "rollout.json")
        write_json(target / "data.json", {"exploration": str(explored), "weight": rollout["weight"],
            "round": kw["round_index"], "verified": verified, "identity": kw["identity"]})
        checkpoint(name, "after_output")

    original_stage = co.stage_run
    def stage_run(root, name, key, action):
        result = original_stage(root, name, key, action)
        checkpoint(name, "after_complete")
        return result

    import baselines.eto.pipeline as transactions
    # Make attempt timestamps deterministic for a byte-for-byte comparison of
    # every existing artifact, not just the model output. Scheduler clocks stay real.
    monkeypatch.setattr(transactions, "time", NS(time=lambda: 1234.0))
    monkeypatch.setattr(p, "prepare", prepare)
    monkeypatch.setattr(p, "served", served)
    monkeypatch.setattr(p, "run_command", command)
    monkeypatch.setattr(p, "collect", collect)
    monkeypatch.setattr(co, "stage_run", stage_run)
    prepare()
    return args, identity, control


def legacy(args, identity):
    for r in range(3):
        for kind in ("explore", "teacher", "train"):
            args.worker = "teacher" if kind == "teacher" else "gpu"
            args.through = f"{kind}-{r}"
            p.train_pipeline(args, identity, {})
            p.update_disclosure(args.output, identity)


def spawn(args, role):
    # Linux fork retains the fake stages and the autouse no-network guard.
    def run():
        args.worker = role
        raise SystemExit(co.run_worker(args, {}))
    process = multiprocessing.get_context("fork").Process(target=run)
    process.start()
    return process


def finish(process):
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("Worker did not finish")
    assert process.exitcode == 0


def artifacts(output):
    return {str(p.relative_to(output)): p.read_bytes() for p in Path(output).rglob("*")
            if p.is_file() and "scheduling" not in p.parts and not p.name.startswith(".")}


def test_pair_matches_every_legacy_stage_input_output_and_artifact(cell):
    args, identity, _ = cell
    legacy(args, identity)
    expected = artifacts(args.output)
    sequence = (Path(args.output) / "scheduling/stub-stages.jsonl").read_bytes()
    assert [json.loads(line)["stage"] for line in sequence.splitlines()] == [
        f"{kind}-{r}" for r in range(3) for kind in ("explore", "teacher", "train")]
    shutil.rmtree(args.output)
    workers = [spawn(args, role) for role in ("teacher", "gpu")]
    try:
        for worker in workers:
            finish(worker)
        assert artifacts(args.output) == expected
        assert (Path(args.output) / "scheduling/stub-stages.jsonl").read_bytes() == sequence
        # No second execution/exposure when either worker restarts a complete cell.
        for role in ("gpu", "teacher"):
            finish(spawn(args, role))
        assert artifacts(args.output) == expected
        assert (Path(args.output) / "scheduling/stub-stages.jsonl").read_bytes() == sequence
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()


@pytest.mark.parametrize("seed", range(9))
def test_sigkill_resume_preserves_completed_transactions(cell, seed):
    args, identity, control = cell
    legacy(args, identity)
    expected_weight = read(Path(args.output) / "stages/train-2/attempt-000/output/result.json")["weight"]
    shutil.rmtree(args.output)
    rng = random.Random(seed)
    stage = f"{('explore', 'teacher', 'train')[seed % 3]}-{rng.randrange(3)}"
    point = ("before_output", "after_output", "after_complete")[seed // 3]
    victim = "teacher" if stage.startswith("teacher") else "gpu"
    control["kill"] = (stage, point)
    workers = {role: spawn(args, role) for role in ("gpu", "teacher")}
    try:
        marker = Path(args.output) / "scheduling/kill-ready.json"
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        workers[victim].kill()
        workers[victim].join(5)
        assert workers[victim].exitcode == -signal.SIGKILL
        completed = {p: p.read_bytes() for p in Path(args.output).glob("stages/*/complete.json")}
        control.clear()
        workers[victim] = spawn(args, victim)
        for worker in workers.values():
            finish(worker)
        assert all(p.read_bytes() == data for p, data in completed.items())
        output = Path(args.output)
        completion = read(output / "stages/train-2/complete.json")
        final = read(output / "stages/train-2" / completion["attempt"] / "output/result.json")
        assert final["weight"] == expected_weight
        assert read(output / "result.json")["updates"] == 250
        attempts = list((output / "stages" / stage).glob("attempt-*"))
        assert len(attempts) == (1 if point == "after_complete" else 2)
        if point != "after_complete":
            assert (attempts[0] / "started.json").exists()  # Failed evidence retained.
    finally:
        for worker in workers.values():
            if worker.is_alive():
                worker.kill()
                worker.join()


def test_wait_backoff_deadline_and_no_teacher_on_gpu(cell):
    args, identity, _ = cell
    now, sleeps = [0.0], []
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    waiter = co.Waiter(args.output, "gpu", limit=1200, initial=7, maximum=60,
                       clock=lambda: now[0], sleep=sleep)
    with pytest.raises(co.WaitExpired):
        co.worker_pipeline(args, identity, {}, waiter)
    assert now[0] == 1200 and sleeps[:4] == [7, 14, 28, 56]
    output = Path(args.output)
    assert (output / "stages/explore-0/complete.json").exists()
    assert not (output / "stages/teacher-0").exists()
    assert not (output / "stages/train-0").exists()
    logs = [json.loads(line) for line in (output / "scheduling/gpu.jsonl").read_text().splitlines()]
    assert logs[-1]["event"] == "wait_finished" and logs[-1]["outcome"] == "resubmit"
    assert len([r for r in logs if r["event"] == "wait_poll"]) == len(sleeps)


def test_idle_worker_returns_resubmit_and_is_resumable(cell):
    args, identity, _ = cell
    args.gpu_wait_seconds = .02
    assert co.run_worker(args, {}) == co.RESUBMIT
    before = (Path(args.output) / "stages/explore-0/complete.json").read_bytes()
    args.gpu_wait_seconds = 5
    workers = [spawn(args, role) for role in ("gpu", "teacher")]
    for worker in workers:
        finish(worker)
    assert (Path(args.output) / "stages/explore-0/complete.json").read_bytes() == before


def test_hash_mismatch_stops_before_new_work(cell):
    args, identity, _ = cell
    legacy(args, identity)
    output = Path(args.output)
    (output / "stages/explore-0/attempt-000/output/rollout.json").write_text("{}")
    with pytest.raises(ValueError, match="artifacts changed"):
        co.run_worker(args, {})
    assert len(list(output.glob("stages/*/attempt-*"))) == 9


def test_same_role_duplicates_share_completed_stages(cell):
    args, _, _ = cell
    workers = [spawn(args, role) for role in ("teacher", "gpu", "gpu", "teacher")]
    try:
        for worker in workers:
            finish(worker)
        assert len(list(Path(args.output).glob("stages/*/attempt-*"))) == 9
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()


def test_legacy_lock_wait_also_obeys_idle_limit(tmp_path):
    now = [0.0]
    def sleep(seconds):
        now[0] += seconds
    waiter = co.Waiter(tmp_path, "gpu", limit=3, initial=1, maximum=2,
                       clock=lambda: now[0], sleep=sleep)
    path = tmp_path / ".pipeline.lock"
    with locked(path), pytest.raises(co.WaitExpired), waiter.lock(path, shared=True):
        pytest.fail("A co-scheduled worker entered a legacy writer's cell")
    assert now[0] == 3


@pytest.mark.parametrize("error", ["BudgetExceeded", "CollectorHalted", "TeacherCallError"])
def test_teacher_halt_does_not_loop_gpu_resubmissions(cell, error):
    args, identity, _ = cell
    p.train_pipeline(args, identity, {})
    write_json(Path(args.output) / "stages/teacher-0/attempt-000/failure.json", {"error_type": error})
    with pytest.raises(co.PeerStageHalted):
        co.run_worker(args, {})
    assert not (Path(args.output) / "stages/train-0").exists()


@pytest.mark.parametrize("limit", [0, 1201, float("inf"), float("nan")])
def test_cannot_extend_gpu_idle_cap(tmp_path, limit):
    with pytest.raises(ValueError):
        co.Waiter(tmp_path, "gpu", limit=limit)
