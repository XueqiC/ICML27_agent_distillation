"""Rollout launchers use fake Python executables: no sockets, GPUs or models."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from test_r03_sets import bank


@pytest.fixture
def launcher(tmp_path):
    fake = tmp_path / "fake-python"
    fake.write_text("#!" + sys.executable + "\n" + r'''
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
def record(kind):
    with open(os.environ["FAKE_LEDGER"], "a") as handle:
        handle.write(json.dumps({"kind": kind, "args": args,
            "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "order": os.environ.get("CUDA_DEVICE_ORDER"),
            "offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
            "attention": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
            "sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")}) + "\n")
def value(flag): return args[args.index(flag) + 1]
scenario = os.environ.get("FAKE_SCENARIO", "pass")
if args[0] == "-":
    if len(args) == 2:
        print(json.loads(Path(args[1]).read_text())["endpoint"])
    else:
        record("preflight")
    sys.exit(0)
out = Path(value("--output"))
out.mkdir(parents=True, exist_ok=False)
if args[0] == "harness/serve_student.py":
    kind = "server"
elif args[0] == "tools/check_lora_serving.py":
    kind = "hf" if "--hf-only" in args else "check"
elif args[:2] == ["tools/rollout_pool.py", "run"]:
    kind = "rollout"
else:
    raise AssertionError("Unexpected executable invocation")
def stop(*_):
    record("stopped" if kind == "server" else "stopped_work")
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
record(kind)
if scenario == kind + "_error":
    (out / "failure.json").write_text('{"error_type": "ValueError"}')
    sys.exit(7)
if kind == "server":
    assert value("--port") == "0"
    (out / "provenance.json").write_text("{}")
    (out / "ready.json").write_text(json.dumps({"endpoint": "http://127.0.0.1:34567/v1"}))
    while True: time.sleep(.05)
if scenario == "block_" + kind:
    while True: time.sleep(.05)
if kind in {"hf", "check"}:
    (out / ("reference.json" if kind == "hf" else "report.json")).write_text('{"passed": true}')
''')
    fake.chmod(0o755)
    for name in ("snapshot", "verified", "adapter with spaces"):
        (tmp_path / name).mkdir()
    (tmp_path / "pool.json").write_text("{}")
    env = os.environ.copy()  # Never log the full environment.
    for key in ("ADAPTERS", "MERGED_MODEL_PATH", "INCLUDE_BASE", "BATCH_INVARIANT",
                "LABEL_SAMPLES", "LORA_DTYPE", "LORA_CONVERSION", "ROLLOUT_SUBSET"):
        env.pop(key, None)
    env.update(PROJECT_ROOT=str(Path.cwd()), HPG_CONFIG=str(Path.cwd() / "configs/hpg.env"),
               ENV_PYTHON=str(fake), TRAIN_PYTHON=str(fake), VLLM_PYTHON=str(fake),
               CONFIG="configs/base.yaml", MODEL_PATH=str(tmp_path / "snapshot"),
               ADAPTER="p10=" + str(tmp_path / "adapter with spaces"),
               VERIFIED=str(tmp_path / "verified"), OUT=str(tmp_path / "out"),
               TASKS=str(tmp_path / "pool.json"), WORKERS="3", READY_TIMEOUT="5",
               DECODING="greedy", ENABLE_THINKING="off", MAX_TOKENS="1024",
               FAKE_LEDGER=str(tmp_path / "ledger.jsonl"), VLLM_BATCH_INVARIANT="stale")
    return ["bash", "scripts/hpg_rollout_adapter.slurm"], env


def read_ledger(env):
    path = Path(env["FAKE_LEDGER"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize("scenario", ["pass", "hf_error", "check_error", "server_error", "rollout_error"])
def test_rollout_launcher_lifecycle(launcher, scenario):
    command, env = launcher
    env["FAKE_SCENARIO"] = scenario
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == (0 if scenario == "pass" else 1 if scenario in {
        "hf_error", "check_error"} else 7), result.stderr
    rows = read_ledger(env)
    expected = ["preflight", "hf"]
    if scenario != "hf_error":
        expected += ["server"]
        if scenario != "server_error":
            expected += ["check"] + ([] if scenario == "check_error" else ["rollout"]) + ["stopped"]
    assert [row["kind"] for row in rows] == expected
    out = Path(env["OUT"])
    if scenario in {"hf_error", "check_error"}:
        stage = "hf-reference" if scenario == "hf_error" else "serving-comparison"
        assert (out / "check/FAILED").read_text() == f"{stage} failed (exit 7)\n"
        assert not (out / "rollouts").exists()
    for row in rows:
        assert row["cuda"] == ""
        assert row["batch_invariant"] == "1" and row["order"] == "PCI_BUS_ID"
        assert row["offline"] == row["transformers_offline"] == "1"
        assert row["attention"] == "FLASH_ATTN" and row["sampler"] == "0"
        args = row["args"]
        if row["kind"] in {"hf", "check"}:
            assert args[args.index("--adapter") + 1] == env["ADAPTER"].split("=", 1)[1]
            assert args[args.index("--verified") + 1] == env["VERIFIED"]
        if row["kind"] == "server":
            assert args.count("--lora-adapter") == 1
            assert args[args.index("--lora-adapter") + 1] == env["ADAPTER"]
            assert args[args.index("--python") + 1] == env["VLLM_PYTHON"]
            assert args[args.index("--lora-dtype") + 1] == "bfloat16"
            assert args[args.index("--lora-conversion") + 1] == "none"
        if row["kind"] in {"check", "rollout"}:
            assert args[args.index("--endpoint") + 1] == "http://127.0.0.1:34567/v1"
            assert args[args.index("--server-provenance") + 1] == str(out / "server/provenance.json")
        if row["kind"] == "check":
            assert args[args.index("--hf-reference") + 1] == str(out / "hf/reference.json")
        if row["kind"] == "rollout":
            expected_flags = {"--pool": env["TASKS"], "--subset": "R2", "--model-name": "p10",
                "--label-samples": "0", "--workers": "3", "--output": str(out / "rollouts"),
                "--lora-check": str(out / "check/report.json"), "--config": env["CONFIG"]}
            for flag, value in expected_flags.items():
                assert args[args.index(flag) + 1] == value
    before = Path(env["FAKE_LEDGER"]).read_bytes()
    again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert again.returncode != 0 and "Output already exists" in again.stderr
    assert Path(env["FAKE_LEDGER"]).read_bytes() == before


@pytest.mark.parametrize("subset", ["all", "R1_P2", "R1_P3", "R2"])
def test_rollout_launcher_options(launcher, subset):
    command, env = launcher
    env.update(ROLLOUT_SUBSET=subset, LABEL_SAMPLES="2", BATCH_INVARIANT="0",
               LORA_DTYPE="float16", LORA_CONVERSION="gemma4")
    env.pop("TASKS")  # Must override hpg.env's evaluation default before sourcing it.
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    rows = read_ledger(env)
    assert all(row["batch_invariant"] == "0" for row in rows)
    server = next(row["args"] for row in rows if row["kind"] == "server")
    assert server[server.index("--lora-dtype") + 1] == "float16"
    assert server[server.index("--lora-conversion") + 1] == "gemma4"
    rollout = next(row["args"] for row in rows if row["kind"] == "rollout")
    assert rollout[rollout.index("--label-samples") + 1] == "2"
    assert rollout[rollout.index("--subset") + 1] == subset
    assert rollout[rollout.index("--pool") + 1] == "data/manifest/r03_pool200.json"


@pytest.mark.parametrize("key,value", [("ADAPTER", ""), ("ADAPTERS", "a=/one b=/two"),
    ("MERGED_MODEL_PATH", "/checkpoint"), ("INCLUDE_BASE", "1"), ("ROLLOUT_SUBSET", "dev300"),
    ("ROLLOUT_SUBSET", "sealed"), ("LABEL_SAMPLES", "-1"), ("LABEL_SAMPLES", "1.5"),
    ("WORKERS", "0"), ("READY_TIMEOUT", "0"), ("BATCH_INVARIANT", "true"),
    ("LORA_DTYPE", "float32"), ("ENABLE_THINKING", "on"), ("MAX_TOKENS", "512"),
    ("DECODING", "sag")])
def test_rollout_launcher_refuses_invalid_options_before_work(launcher, key, value):
    command, env = launcher
    env[key] = value
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode != 0 and not Path(env["OUT"]).exists()
    assert not read_ledger(env)


@pytest.mark.parametrize("stage", ["hf", "check", "rollout"])
def test_rollout_launcher_term_cleans_work_and_server(launcher, stage):
    command, env = launcher
    env["FAKE_SCENARIO"] = "block_" + stage
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not any(row["kind"] == stage for row in read_ledger(env)):
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, stderr
        kinds = [row["kind"] for row in read_ledger(env)]
        tail = ["stopped_work"] if stage == "hf" else ["stopped_work", "stopped"]
        assert kinds[-len(tail):] == tail
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


@pytest.mark.parametrize("case", ["pool", "dev40", "dev140", "dev300", "sealed", "wrong_p10", "bad_adapter"])
def test_rollout_launcher_real_cpu_preflight(bank, tmp_path, monkeypatch, case):
    from data.manifest.build_r03_sets import build_sets, listing
    from sed.dev_subset import build_dev40
    from sed.manifest import resolve_task
    from sed.storage import write_json
    manifest, freeze, _, _, dev140 = bank
    sets = build_sets(*bank)
    chosen = {"dev40": build_dev40(manifest, freeze), "dev140": dev140,
              "dev300": sets["dev300"]}.get(case, sets["pool200"])
    if case == "sealed":
        task_id = next(t["task_id"] for t in manifest["tasks"] if t["source_split"] == "valid_unseen")
        # Even a listing named as the pool cannot authorize a sealed task.
        chosen = listing("r03_pool200", [resolve_task(manifest, freeze, task_id)], manifest, freeze)
    tasks = tmp_path / "tasks.json"
    write_json(tasks, chosen)
    monkeypatch.setattr("sed.config.load_config", lambda _: {
        "paths": {"manifest": "unused", "split_freeze": "unused"},
        "models": {"student": {"requested_repo": "google/gemma-4-12B-it"}}})
    monkeypatch.setattr("sed.manifest.load_frozen", lambda *_: (manifest, freeze))
    monkeypatch.setattr("harness.lora_adapters.inspect_adapter", lambda *_: {
        "checkpoint_metadata": {"supervised_tokens": 1 if case == "wrong_p10" else 104070}})
    adapter = "invalid" if case == "bad_adapter" else "p10=/checkpoint with spaces"
    monkeypatch.setattr(sys, "argv", ["-", "unused", "unused", str(tasks), "all", adapter])
    script = Path("scripts/rollout_adapter.sh").read_text()
    preflight = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    if case == "pool":
        exec(compile(preflight, "rollout_adapter preflight", "exec"), {})
    else:
        with pytest.raises(ValueError):
            exec(compile(preflight, "rollout_adapter preflight", "exec"), {})
