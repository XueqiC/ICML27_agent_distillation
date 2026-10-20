"""POSIX process shards: fake transport/environment, real durable collection."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.smartad import collect as sm
from sed.storage import JsonlLedger, digest
from teachers.openai_teacher import TeacherCallError
from test_smartad import factory, setup_source
from test_stage2b_seeds import seed_bank


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def process_factory(path, *, barrier=None, unsupported=False, submitted=None, release=None):
    def make(config, directory, task, identity, limiter, sampling_root):
        class Calls:
            first = True

            def append(self, kwargs):
                JsonlLedger(path).append({"task_id": task["task_id"], "slot": directory.parent.name,
                    "pid": os.getpid(), "temperature": kwargs.get("temperature")})
                if self.first:
                    self.first = False
                    if barrier is not None:
                        barrier.wait(timeout=20)
                    if submitted is not None:
                        submitted.set()
                        assert release.wait(timeout=30)
        return factory(Calls(), unsupported=unsupported)(
            config, directory, task, identity, limiter, sampling_root)
    return make


def join(processes):
    try:
        for process in processes:
            process.join(timeout=30)
        assert [p.exitcode for p in processes] == [0] * len(processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=5)


def count_reuse(monkeypatch, path):
    original = sm.reuse_check

    def check(*args, **kwargs):
        JsonlLedger(path).append({"task_id": args[1]["task_id"]})
        return original(*args, **kwargs)

    monkeypatch.setattr(sm, "reuse_check", check)


@pytest.mark.parametrize("default,unsupported", [(False, False), (True, True), (True, False)])
def test_concurrent_slots_reuse_once_and_overlapping_resumes(
        seed_bank, tmp_path, monkeypatch, default, unsupported):
    monkeypatch.setenv("SED_HASH_WORKERS", "1")
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, default=default)
    calls, reuse, output = tmp_path / "calls.jsonl", tmp_path / "reuse.jsonl", tmp_path / "pool"
    count_reuse(monkeypatch, reuse)
    ctx = mp.get_context("fork")  # Inherits pytest's network/API prohibitions.
    # Nine requests must be in flight for ONE task before any can finish.
    # The mismatch case additionally executes slot 1, so needs no barrier.
    barrier = ctx.Barrier(9) if unsupported or not default else None
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env,
        teacher_factory=process_factory(calls, barrier=barrier, unsupported=unsupported))
    processes = []
    # Include both a duplicate shard and an overlapping, differently sized run.
    for index, count in [*((i, 9) for i in range(9)), (0, 9), (0, 1)]:
        process = ctx.Process(target=sm.collect, args=(source, manifest, freeze, config, output),
                             kwargs={**kwargs, "shard_index": index, "shard_count": count})
        process.start()
        processes.append(process)
    join(processes)
    pool = sm.load_pool(output, manifest, freeze)[0]
    expected_slots = 10 if default and not unsupported else 9
    assert len(rows(calls)) == expected_slots + (9 if unsupported else 0)
    successful = [r for r in rows(calls) if not unsupported or r["temperature"] is None]
    assert len({(r["task_id"], r["slot"]) for r in successful}) == len(successful) == expected_slots
    if barrier is not None:
        assert len({r["pid"] for r in successful}) == 9
    assert len(rows(reuse)) == 1
    assert pool["api_attempts"] == len(rows(calls))
    assert pool["tasks"][0]["reuse"] == (expected_slots == 9)
    assert len(list((output / "work").glob("*/candidate-*/candidate.json"))) == 10
    before = (output / "candidates.json").read_bytes()
    sm.collect(source, manifest, freeze, config, output, shard_count=16, shard_index=15, **kwargs)
    assert len(rows(reuse)) == 1 and (output / "candidates.json").read_bytes() == before


def test_partial_shard_cannot_publish_and_identity_ignores_shard_count(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    calls, output = [], tmp_path / "pool"
    kwargs = dict(rate_limit_dir=tmp_path / "limit", env_factory=env, teacher_factory=factory(calls))
    sm.collect(source, manifest, freeze, config, output, shard_count=3, shard_index=0, **kwargs)
    assert len(calls) == 3
    assert not (output / "candidates.json").exists()
    assert not list(output.glob("work/*/reuse.json"))
    assert json.loads((output / "result.json").read_text())["status"] == "collecting"
    identity = (output / "identity.json").read_bytes()
    # A different partition resumes the missing slots, never resampling the first three.
    sm.collect(source, manifest, freeze, config, output, shard_count=2, shard_index=0, **kwargs)
    assert not (output / "candidates.json").exists()
    sm.collect(source, manifest, freeze, config, output, shard_count=2, shard_index=1, **kwargs)
    assert (output / "identity.json").read_bytes() == identity
    assert "shard_count" not in json.loads(identity)
    assert len(calls) == sm.load_pool(output, manifest, freeze)[0]["api_attempts"] == 9
    with pytest.raises(ValueError, match="use --resume"):
        sm.collect(source, manifest, freeze, config, output, resume=False, **kwargs)


def test_conditional_first_slot_has_deterministic_owner(seed_bank, tmp_path, monkeypatch):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, default=True)
    calls, reuse, output = [], tmp_path / "reuse.jsonl", tmp_path / "pool"
    count_reuse(monkeypatch, reuse)
    kwargs = dict(rate_limit_dir=tmp_path / "limit", env_factory=env,
                  teacher_factory=factory(calls), shard_count=3)
    # Slot 1's flattened offset is 9, owned by shard 0. It exits before the
    # other nine slots are final; shard 2 can decide reuse but cannot execute 1.
    for index in (0, 1, 2, 2):
        sm.collect(source, manifest, freeze, config, output, shard_index=index, **kwargs)
    assert len(calls) == 9 and len(rows(reuse)) == 1
    assert not (output / "candidates.json").exists()
    assert not list(output.glob("work/*/candidate-01/candidate.json"))
    sm.collect(source, manifest, freeze, config, output, shard_index=0, **kwargs)
    assert len(calls) == 10 and len(rows(reuse)) == 1
    assert sm.load_pool(output, manifest, freeze)[0]["status"] == "complete"


@pytest.mark.parametrize("boundary", ["episode", "request", "reuse"])
def test_killed_shard_never_executes_a_started_slot_twice(seed_bank, tmp_path, monkeypatch, boundary):
    monkeypatch.setenv("SED_HASH_WORKERS", "1")
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    calls, reuse, output = tmp_path / "calls.jsonl", tmp_path / "reuse.jsonl", tmp_path / "pool"
    count_reuse(monkeypatch, reuse)
    ctx = mp.get_context("fork")
    stopped, release = ctx.Event(), ctx.Event()
    run, frozen_write = sm.run_episode, sm.frozen_write

    def stop_after_episode(*args, **kwargs):
        run(*args, **kwargs)
        stopped.set()
        assert release.wait(timeout=30)

    def stop_after_decision(path, value):
        frozen_write(path, value)
        if path.name == "reuse.json":
            stopped.set()
            assert release.wait(timeout=30)

    if boundary == "episode":
        monkeypatch.setattr(sm, "run_episode", stop_after_episode)
    if boundary == "reuse":
        monkeypatch.setattr(sm, "frozen_write", stop_after_decision)
    kwargs = dict(rate_limit_dir=tmp_path / "limit", env_factory=env,
        teacher_factory=process_factory(calls, submitted=stopped if boundary == "request" else None,
                                        release=release))
    process = ctx.Process(target=sm.collect, args=(source, manifest, freeze, config, output), kwargs=kwargs)
    process.start()
    try:
        assert stopped.wait(timeout=20)
        assert not (output / "candidates.json").exists()
    finally:
        process.kill()
        process.join(timeout=5)
    assert process.exitcode == -9
    monkeypatch.setattr(sm, "run_episode", run)
    monkeypatch.setattr(sm, "frozen_write", frozen_write)
    kwargs["teacher_factory"] = process_factory(calls)
    if boundary == "request":
        with pytest.raises(TeacherCallError):
            sm.collect(source, manifest, freeze, config, output, **kwargs)
        assert len(rows(calls)) == 1
        assert not (output / "candidates.json").exists()
    else:
        for index in range(3):
            sm.collect(source, manifest, freeze, config, output, shard_count=3, shard_index=index, **kwargs)
        assert len(rows(calls)) == len({r["slot"] for r in rows(calls)}) == 9
        assert len(rows(reuse)) == 1
        assert sm.load_pool(output, manifest, freeze)[0]["status"] == "complete"


def test_shards_and_cells_share_one_posix_limiter(seed_bank, tmp_path, monkeypatch):
    monkeypatch.setenv("SED_HASH_WORKERS", "1")
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    calls, limiter = tmp_path / "calls.jsonl", tmp_path / "limit"
    ctx, processes = mp.get_context("fork"), []
    outputs = [tmp_path / "cell-a", tmp_path / "cell-b"]
    for output, count in zip(outputs, (2, 3)):
        for index in range(count):
            process = ctx.Process(target=sm.collect, args=(source, manifest, freeze, config, output),
                kwargs=dict(shard_count=count, shard_index=index, rate_limit_dir=limiter,
                            env_factory=env, teacher_factory=process_factory(calls)))
            process.start()
            processes.append(process)
    join(processes)
    assert len(rows(calls)) == 18
    assert (outputs[0] / "identity.json").read_bytes() == (outputs[1] / "identity.json").read_bytes()
    assert len(list(limiter.glob("*.json"))) == 1
    state = json.loads((limiter / (digest("gpt-6-luna") + ".json")).read_text())
    assert len(state["reservations"]) == 18
    assert all(sm.load_pool(output, manifest, freeze)[0]["status"] == "complete" for output in outputs)


@pytest.mark.parametrize("index,count", [(-1, 2), (2, 2), (0, 0), (0, -1), (True, 2)])
def test_invalid_shards_fail_before_io(tmp_path, index, count):
    with pytest.raises(ValueError, match="shard|Shard"):
        sm.collect(None, None, None, None, tmp_path / "pool", rate_limit_dir=tmp_path / "limit",
                   shard_index=index, shard_count=count)
    assert not (tmp_path / "pool").exists()


def test_cli_forwards_shards_and_resume(seed_bank, tmp_path, monkeypatch):
    from baselines.smartad import pipeline
    source, manifest, freeze, config, _ = setup_source(seed_bank, tmp_path)
    received = []
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline, "load_frozen", lambda *_: (manifest, freeze))
    monkeypatch.setattr(pipeline.signal, "signal", lambda *_: None)
    monkeypatch.setattr(pipeline, "collect", lambda *args, **kwargs: received.append(kwargs))
    pipeline.main(["--verified", str(source), "--output", str(tmp_path / "pool"),
                   "--shard-count", "16", "--shard-index", "7", "--resume"])
    assert received[0]["shard_count"] == 16 and received[0]["shard_index"] == 7
    assert received[0]["resume"] is True


@pytest.mark.parametrize("source_status", ["collecting", "complete", "deferred"])
def test_launcher_concurrent_cells_complete_source_gate_and_resume(tmp_path, source_status):
    fake, ledger = tmp_path / "fake-python", tmp_path / "commands.jsonl"
    fake.write_text("#!" + sys.executable + "\n" + r'''
import fcntl, json, os, sys, time
from pathlib import Path
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
args = sys.argv[1:]
if args[0] == "-c":
    sys.argv = args[1:]
    exec(args[1])
elif args[0] == "-":
    script = sys.stdin.read()
    if "load_verified_collection" not in script:  # Stub only source preflight.
        sys.argv = args
        exec(script)
else:
    ledger = Path(os.environ["FAKE_LEDGER"])
    with ledger.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(args) + "\n")
    deadline = time.monotonic() + 15
    # No worker can finish until ALL cells/shards have been launched.
    while len(ledger.read_text().splitlines()) < int(os.environ["FAKE_EXPECTED"]):
        assert time.monotonic() < deadline, "Launcher serialized cells or shards"
        time.sleep(.02)
    output = Path(args[args.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("identity", {}), ("candidates", {"status": "complete"})):
        if name == "candidates" and os.environ.get("FAKE_DEFER") == "1" and output.name == "k100_s0":
            calls = [json.loads(line) for line in ledger.read_text().splitlines()]
            count = sum(r[r.index("--output") + 1] == str(output) for r in calls)
            if count <= int(os.environ["SMARTAD_SHARDS"]):
                continue
        temporary = output / (name + str(os.getpid()) + ".tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(output / (name + ".json"))
''')
    fake.chmod(0o755)
    sources, output, logs = tmp_path / "sources", tmp_path / "pools", tmp_path / "logs"
    for name in ("seeds_k100_s0", "seeds_k100_s1", "seeds_k100_s2", "seeds_full_train_r2"):
        directory = sources / name
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"status": source_status}))
    shards = 16 if source_status == "collecting" else 2
    cells = 4 if source_status == "complete" else 3
    expected = cells * shards
    first_total = expected + (shards if source_status == "deferred" else 0)
    env = dict(os.environ, PROJECT_ROOT=str(Path.cwd()), ENV_PYTHON=str(fake),
        COLLECTION_ROOT=str(sources), SMARTAD_ROOT=str(output), LOG_ROOT=str(logs),
        SMARTAD_FULL_SOURCE=str(sources / "seeds_full_train_r2"),
        SMARTAD_RATE_LIMIT_DIR=str(tmp_path / "limiter"), FAKE_LEDGER=str(ledger),
        FAKE_EXPECTED=str(expected), FAKE_DEFER=str(int(source_status == "deferred")))
    env.pop("SMARTAD_SHARDS", None)
    if shards != 16:
        env["SMARTAD_SHARDS"] = str(shards)
    command = ["bash", "scripts/rai_collect_smartad.sh"]
    first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=25)
    assert first.returncode == 0, first.stderr
    invocations = rows(ledger)
    assert len(invocations) == len(list(logs.glob("*shard*.log"))) == first_total
    if source_status != "complete":
        assert "Skipping full_train" in first.stdout
        assert not (output / "full_train").exists()
    else:
        assert (output / "full_train/candidates.json").exists()
    if source_status == "deferred":
        assert "completing deferred candidate slots" in first.stdout
    assert {r[r.index("--shard-index") + 1] for r in invocations} == {str(i) for i in range(shards)}
    for args in invocations:
        assert "--resume" in args
        assert args[args.index("--shard-count") + 1] == str(shards)
        assert args[args.index("--rate-limit-dir") + 1] == str(tmp_path / "limiter")
        assert ("--subset-k" in args) == (Path(args[args.index("--output") + 1]).name != "full_train")
    refused = subprocess.run(command, env=env, capture_output=True, text=True, timeout=25)
    assert refused.returncode == 2 and len(rows(ledger)) == first_total
    env["FAKE_EXPECTED"] = str(first_total + expected)
    resumed = subprocess.run([*command, "--resume"], env=env, capture_output=True, text=True, timeout=25)
    assert resumed.returncode == 0, resumed.stderr
    assert len(rows(ledger)) == first_total + expected
