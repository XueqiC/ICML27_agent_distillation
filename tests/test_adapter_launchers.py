"""Exercise launcher order/cleanup with fake executables; no sockets or models."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("platform,scenario,include_base", [
    (platform, scenario, include_base)
    for platform in ("rai", "hpg")
    for scenario, include_base in (("pass", True), ("pass", False), ("check_a", True),
                                   ("check_a", False), ("check_b", True), ("check_all", True),
                                   ("check_error", True), ("hf_a", True))
] + [("hpg_full", "pass", True), ("hpg_sag", "pass", False), ("hpg_sag", "check_a", False)])
def test_launcher_lifecycle(tmp_path, platform, scenario, include_base):
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
            "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
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
    scenario = os.environ["FAKE_SCENARIO"]
    name = value("--model-name")
    failed = ((kind == "hf" and scenario == "hf_a" and name == "a") or
              (kind == "check" and (scenario == "check_all" or scenario == "check_" + name or
                                    (scenario == "check_error" and name == "a"))))
    if failed and (kind == "hf" or scenario == "check_error"):
        (out / "failure.json").write_text(json.dumps({"passed": False, "error_type": "ValueError"}))
    else:
        (out / ("reference.json" if kind == "hf" else "report.json")).write_text(
            json.dumps({"passed": not failed, "model_name": name}))
    if failed: sys.exit(7)
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
    env.pop("LORA_DTYPE", None)
    expected_dtype = "float16" if scenario == "pass" and not include_base else "bfloat16"
    if expected_dtype == "float16":
        env["LORA_DTYPE"] = expected_dtype
    env.pop("BATCH_INVARIANT", None)
    expected_bi = "0" if scenario == "pass" and not include_base and platform != "hpg_sag" else "1"
    if expected_bi == "0" or platform == "hpg_full":
        env["BATCH_INVARIANT"] = expected_bi
    env["VLLM_BATCH_INVARIANT"] = "1" if expected_bi == "0" else "0"
    env.update(PROJECT_ROOT=str(Path.cwd()), ENV_PYTHON=str(fake), VLLM_PYTHON=str(fake),
               TRAIN_PYTHON=str(fake), MODEL_PATH=str(tmp_path / "snapshot"),
               VERIFIED=str(tmp_path / "verified"), FAKE_LEDGER=str(ledger),
               FAKE_SCENARIO=scenario, ENABLE_THINKING="off",
               MAX_TOKENS="1024", READY_TIMEOUT="10", WORKERS="8", INCLUDE_BASE=str(int(include_base)))
    env["DECODING"] = "sag" if platform == "hpg_sag" else "greedy"
    if platform == "rai":
        command = ["bash", "scripts/rai_eval_student.sh", "3", str(tasks), str(out)]
        if include_base:
            command.append("--base")
        command += ["--workers", "6", "a=/fake/one", "b=/fake/two"]
    else:
        env.update(HPG_CONFIG=str(Path.cwd() / "configs/hpg.env"), TASKS=str(tasks), OUT=str(out),
                   ADAPTERS="a=/fake/one b=/fake/two", CUDA_VISIBLE_DEVICES="allocated")
        if platform == "hpg_full":
            env.update(ADAPTERS="", MERGED_MODEL_PATH=str(tmp_path / "snapshot"), FULL_CHECKPOINT_NAME="iter-1")
        command = ["bash", "scripts/hpg_eval_adapters.slurm"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == int(scenario != "pass"), result.stderr
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    for row in rows:
        if row["kind"] == "server":
            assert row["args"][row["args"].index("--lora-dtype") + 1] == expected_dtype
        if row["kind"] == "eval":
            assert row["args"][row["args"].index("--decoding") + 1] == env["DECODING"]
    expected = ["hf", "hf", "server"] + (["eval"] if include_base else [])
    failed_names = ({"a", "b"} if scenario == "check_all" else
                    {"b"} if scenario == "check_b" else {"a"} if scenario != "pass" else set())
    for name in ("a", "b"):
        if scenario != "hf_a" or name != "a":
            expected.append("check")
        if name not in failed_names:
            expected.append("eval")
    if platform == "hpg_full":
        expected = ["server", "eval"]
        assert "--merged-model-path" in rows[0]["args"]
    assert [r["kind"] for r in rows] == expected + ["stopped"]
    if platform != "hpg_full":
        for name in ("a", "b"):
            check = out / name / "check"
            if name in failed_names:
                assert not (out / name / "eval").exists()
                stage = "hf-reference" if scenario == "hf_a" else "serving-comparison"
                assert (check / "FAILED").read_text() == f"{stage} failed (exit 7)\n"
                if scenario in {"hf_a", "check_error"}:
                    failure_dir = out / name / "hf" if scenario == "hf_a" else check
                    assert json.loads((failure_dir / "failure.json").read_text()) == {
                        "passed": False, "error_type": "ValueError"}
                    assert not (check / "report.json").exists()
                else:
                    assert json.loads((check / "report.json").read_text()) == {"passed": False, "model_name": name}
            else:
                assert (out / name / "eval").is_dir()
                assert not (check / "FAILED").exists()
                assert json.loads((check / "report.json").read_text()) == {"passed": True, "model_name": name}
        evaluated = [r["args"][r["args"].index("--model-name") + 1] for r in rows if r["kind"] == "eval"]
        assert evaluated == (["google/gemma-4-12B-it"] if include_base else []) + [
            name for name in ("a", "b") if name not in failed_names]
        assert (out / "google/gemma-4-12B-it/eval").is_dir() is include_base
    for row in rows:
        assert row["batch_invariant"] == expected_bi
        assert row["order"] == "PCI_BUS_ID" and row["offline"] == "1"
        assert row["attention"] == "FLASH_ATTN" and row["sampler"] == "0"
        assert row["cuda"] == ("" if row["kind"] in {"check", "eval"}
                               else "3" if platform == "rai" else "allocated")
        if row["kind"] == "eval":
            args = row["args"]
            name = args[args.index("--model-name") + 1]
            assert args[args.index("--output") + 1] == str(out / ("iter-1" if platform == "hpg_full" else name) / "eval")
            assert args[args.index("--workers") + 1] == ("6" if platform == "rai" else "8")
            assert "http://127.0.0.1:34567/v1" in args
            if name in {"a", "b"}: assert "--lora-check" in args
    # Existing output refusal happens before any new process can be launched.
    before = ledger.read_bytes()
    again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert again.returncode != 0 and ledger.read_bytes() == before


@pytest.mark.parametrize("batch_invariant", [None, "0", "1"])
def test_hpg_student_batch_invariant_env(tmp_path, batch_invariant):
    fake = tmp_path / "fake-python"
    fake.write_text("#!" + sys.executable + "\n" + r'''
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--output") + 1])
out.mkdir(parents=True)
with open(os.environ["FAKE_LEDGER"], "a") as handle:
    handle.write(json.dumps({"command": args[0], "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT")}) + "\n")
if args[0] == "harness/serve_student.py":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    (out / "ready.json").write_text("{}")
    while True: time.sleep(.05)
assert args[0] == "eval/run_suite.py"
''')
    fake.chmod(0o755)
    (tmp_path / "snapshot").mkdir()
    (tmp_path / "tasks.json").write_text("{}")
    env = os.environ.copy()
    env.pop("BATCH_INVARIANT", None)
    if batch_invariant is not None:
        env["BATCH_INVARIANT"] = batch_invariant
    env.update(PROJECT_ROOT=str(Path.cwd()), HPG_CONFIG=str(Path.cwd() / "configs/hpg.env"),
               ENV_PYTHON=str(fake), VLLM_PYTHON=str(fake), MODEL_PATH=str(tmp_path / "snapshot"),
               TASKS=str(tmp_path / "tasks.json"), OUT=str(tmp_path / "out"), SLURM_JOB_ID="1234",
               ENABLE_THINKING="off", MAX_TOKENS="1024", READY_TIMEOUT="5", WORKERS="1",
               FAKE_LEDGER=str(tmp_path / "ledger.jsonl"), VLLM_BATCH_INVARIANT="stale")
    result = subprocess.run(["bash", "scripts/hpg_eval_student.slurm"], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [r["command"] for r in rows] == ["harness/serve_student.py", "eval/run_suite.py"]
    assert all(r["batch_invariant"] == (batch_invariant or "1") for r in rows)


@pytest.mark.parametrize("script", ["eval_adapters.sh", "rai_eval_student.sh", "hpg_eval_adapters.slurm",
                                   "hpg_eval_student.slurm", "rai_mechanism.sh"])
def test_launchers_reject_invalid_batch_invariant_before_work(script):
    env = dict(os.environ, BATCH_INVARIANT="true")
    result = subprocess.run(["bash", "scripts/" + script], env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and "BATCH_INVARIANT must be 0 or 1" in result.stderr
