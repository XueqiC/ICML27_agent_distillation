"""Exercise HAD HF -> serving -> check -> eval with fake executables only."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("harness,fail_gate", [("had", False), ("h0", False), ("had", True)])
def test_full_checkpoint_launcher_enforces_gate(tmp_path, harness, fail_gate):
    fake = tmp_path / "python"
    fake.write_text("#!" + sys.executable + "\n" + r'''
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-':
    if len(args) == 2:
        data = json.loads(Path(args[1]).read_text())
        print(data['endpoint'] if 'endpoint' in data else int(data.get('method') == 'had'))
    else:
        print('google/gemma-4-12B-it')
    raise SystemExit(0)
def val(key): return args[args.index(key)+1]
def record(kind):
    with open(os.environ['HAD_TEST_LEDGER'], 'a') as handle:
        handle.write(json.dumps({'kind': kind, 'args': args, 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')})+'\n')
out = Path(val('--output'))
out.mkdir(parents=True)
if args[:2] == ['-m', 'baselines.had.gate']:
    hf = '--hf-only' in args
    record('hf' if hf else 'check')
    (out / ('reference.json' if hf else 'report.json')).write_text('{}')
    if not hf and os.environ['HAD_TEST_FAIL'] == '1': raise SystemExit(7)
elif args[0] == 'harness/serve_student.py':
    record('serve')
    (out / 'ready.json').write_text(json.dumps({'endpoint':'http://127.0.0.1:34567/v1'}))
    (out / 'provenance.json').write_text('{}')
    def stop(*_):
        record('stopped')
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    while True: time.sleep(.05)
elif args[0] == 'eval/run_suite.py':
    record('eval')
else:
    raise AssertionError('Unexpected executable')
''')
    fake.chmod(0o755)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training_state.json").write_text(json.dumps({"method": "had"}))
    (tmp_path / "tasks.json").write_text("{}")
    config = tmp_path / "hpg.env"
    config.write_text("# Use explicitly supplied test environment.\n")
    env = os.environ.copy()
    env.update(PROJECT_ROOT=str(Path.cwd()), HPG_CONFIG=str(config), ENV_PYTHON=str(fake),
        TRAIN_PYTHON=str(fake), VLLM_PYTHON=str(fake), MODEL_PATH=str(checkpoint),
        MERGED_MODEL_PATH=str(checkpoint), FULL_CHECKPOINT_NAME="iter-1-" + harness,
        VERIFIED=str(checkpoint), HAD_EVAL_HARNESS=harness, WORKERS="1", READY_TIMEOUT="5",
        TASKS=str(tmp_path / "tasks.json"), OUT=str(tmp_path / "out"), ADAPTERS="", INCLUDE_BASE="1",
        HAD_TEST_LEDGER=str(tmp_path / "ledger.jsonl"), HAD_TEST_FAIL=str(int(fail_gate)),
        ENABLE_THINKING="off", MAX_TOKENS="1024", CUDA_VISIBLE_DEVICES="", BATCH_INVARIANT="1", DECODING="greedy")
    result = subprocess.run(["bash", "scripts/hpg_eval_adapters.slurm"], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == (7 if fail_gate else 0), result.stderr
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == (["hf", "serve", "check", "stopped"] if fail_gate else
                                        ["hf", "serve", "check", "eval", "stopped"])
    if not fail_gate:
        args = rows[-2]["args"]
        assert args[args.index("--had-harness") + 1] == harness
        assert "--had-check" in args and rows[-2]["cuda"] == ""
