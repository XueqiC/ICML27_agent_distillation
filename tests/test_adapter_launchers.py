"""Exercise launcher order/cleanup with fake executables; no sockets or models."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("platform,fail_check", [("rai", False), ("hpg", False), ("rai", True)])
def test_launcher_lifecycle(tmp_path, platform, fail_check):
    fake = tmp_path / "fake-python"
    fake.write_text("#!" + sys.executable + "\n" + r'''
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
ledger = Path(os.environ["FAKE_LEDGER"])
def record(kind, **kw):
    with ledger.open("a") as f:
        f.write(json.dumps({"kind": kind, "args": args,
            "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "order": os.environ.get("CUDA_DEVICE_ORDER"),
            "offline": os.environ.get("HF_HUB_OFFLINE"),
            "attention": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"), **kw}) + "\n")
def value(flag): return args[args.index(flag) + 1]
if args[0] == "-":
    if len(args) == 2:
        print(json.loads(Path(args[1]).read_text())["endpoint"])
    else:
        print("google/gemma-4-12B-it")
    sys.exit(0)
out = Path(value("--output"))
out.mkdir(parents=True, exist_ok=False)
if args[0] == "harness/serve_student.py":
    record("server")
    assert value("--port") == "0"
    def stop(*_):
        record("stopped")
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    (out / "ready.json").write_text(json.dumps({"endpoint": "http://127.0.0.1:34567/v1"}))
    (out / "provenance.json").write_text("{}")
    while True: time.sleep(0.05)
elif args[0] == "tools/check_lora_serving.py":
    kind = "hf" if "--hf-only" in args else "check"
    record(kind)
    if kind == "check" and os.environ.get("FAKE_FAIL_CHECK") == "1": sys.exit(1)
    (out / ("reference.json" if kind == "hf" else "report.json")).write_text("{}")
elif args[0] == "eval/run_suite.py":
    record("eval")
else:
    raise AssertionError("Unexpected fake executable invocation")
''')
    fake.chmod(0o755)
    for folder in ("snapshot", "verified"):
        (tmp_path / folder).mkdir()
    tasks = tmp_path / "dev40.json"
    tasks.write_text("{}")
    ledger = tmp_path / "ledger.jsonl"
    out = tmp_path / "out"
    env = os.environ.copy()  # Never log the complete environment.
    env.update(PROJECT_ROOT=str(Path.cwd()), ENV_PYTHON=str(fake), VLLM_PYTHON=str(fake),
               TRAIN_PYTHON=str(fake), MODEL_PATH=str(tmp_path / "snapshot"),
               VERIFIED=str(tmp_path / "verified"), FAKE_LEDGER=str(ledger),
               FAKE_FAIL_CHECK="1" if fail_check else "0", ENABLE_THINKING="off",
               MAX_TOKENS="1024", READY_TIMEOUT="10", WORKERS="8", INCLUDE_BASE="1")
    if platform == "rai":
        command = ["bash", "scripts/rai_eval_student.sh", "3", str(tasks), str(out),
                   "--base", "--workers", "6", "a=/fake/one", "b=/fake/two"]
    else:
        env.update(HPG_CONFIG=str(Path.cwd() / "configs/hpg.env"), TASKS=str(tasks), OUT=str(out),
                   ADAPTERS="a=/fake/one b=/fake/two", CUDA_VISIBLE_DEVICES="allocated")
        command = ["bash", "scripts/hpg_eval_adapters.slurm"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == int(fail_check), result.stderr
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    expected = ["hf", "hf", "server", "eval", "check"]
    if not fail_check: expected += ["eval", "check", "eval"]
    assert [r["kind"] for r in rows] == expected + ["stopped"]
    for row in rows:
        assert row["order"] == "PCI_BUS_ID" and row["offline"] == "1"
        assert row["attention"] == "FLASH_ATTN" and row["sampler"] == "0"
        assert row["cuda"] == ("" if row["kind"] in {"check", "eval"}
                               else "3" if platform == "rai" else "allocated")
        if row["kind"] == "eval":
            args = row["args"]
            name = args[args.index("--model-name") + 1]
            assert args[args.index("--output") + 1] == str(out / name / "eval")
            assert args[args.index("--workers") + 1] == ("6" if platform == "rai" else "8")
            assert "http://127.0.0.1:34567/v1" in args
            if name in {"a", "b"}: assert "--lora-check" in args
    # Existing output refusal happens before any new process can be launched.
    before = ledger.read_bytes()
    again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert again.returncode != 0 and ledger.read_bytes() == before
