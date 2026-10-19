"""Test SLURM submission and resubmission using fake executables only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.fixture
def launcher(tmp_path):
    bindir = tmp_path / "fake bin"
    bindir.mkdir()
    sbatch = bindir / "sbatch"
    sbatch.write_text("#!" + sys.executable + "\n" + r'''
import json, os, sys
from pathlib import Path
ledger = Path(os.environ['HAD_TEST_SUBMISSIONS'])
with ledger.open('a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\n')
print(1000 + len(ledger.read_text().splitlines()))
''')
    python = bindir / "python"
    python.write_text("#!" + sys.executable + "\n" + r'''
import json, os, sys, time
from pathlib import Path
Path(os.environ['HAD_TEST_CALLED']).write_text(json.dumps(sys.argv[1:]))
if os.environ.get('HAD_TEST_BLOCK') == '1':
    while True: time.sleep(.01)
raise SystemExit(int(os.environ.get('HAD_TEST_EXIT', '0')))
''')
    sbatch.chmod(0o755)
    python.chmod(0o755)
    config = tmp_path / "hpg.env"
    config.write_text("# All test values are explicitly supplied.\n")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SLURM_"):
            del env[key]
    env.update(PATH=str(bindir) + os.pathsep + env["PATH"], PROJECT_ROOT=str(Path.cwd()),
        HPG_CONFIG=str(config), HAD_OUT=str(tmp_path / "cell with spaces"), VERIFIED="verified",
        ENV_PYTHON=str(python), TRAIN_PYTHON=str(python), VLLM_PYTHON=str(python),
        HAD_TEST_SUBMISSIONS=str(tmp_path / "submissions.jsonl"), HAD_TEST_CALLED=str(tmp_path / "called.json"),
        MODEL_PATH="snapshot", ENABLE_THINKING="off", MAX_TOKENS="1024", MAX_LENGTH="32768",
        WORKERS="8", READY_TIMEOUT="5", CUDA_VISIBLE_DEVICES="")
    return env


def submissions(env):
    path = Path(env["HAD_TEST_SUBMISSIONS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def launch(env, *args):
    return subprocess.run(["bash", "scripts/ops_had_cell.sh", *args], env=env,
                          capture_output=True, text=True, timeout=10)


def test_pair_resources_and_account_arguments(launcher):
    result = launch(launcher, "--account", "test-account", "--qos=test-qos")
    assert result.returncode == 0, result.stderr
    cpu, gpu = submissions(launcher)
    for row in (cpu, gpu):
        assert row[row.index("--account") + 1] == "test-account"
        assert row.count("--qos=test-qos") == 2  # sbatch plus the resubmission argv.
        assert "--time=48:00:00" in row and "--nodes=1" in row and "--ntasks=1" in row
        assert "--signal=B:USR1@120" in row
        assert not any(arg.startswith("--dependency") for arg in row)
    assert {"--partition=hpg-milan", "--gpus=0", "--cpus-per-task=2", "--mem=8G"} <= set(cpu)
    assert {"--partition=hpg-b200", "--gpus=4", "--cpus-per-task=8", "--mem=256G"} <= set(gpu)
    assert cpu[cpu.index("--worker") + 1] == "teacher"
    assert gpu[gpu.index("--worker") + 1] == "gpu"


@pytest.mark.parametrize("status", [0, 1, 75])
def test_only_idle_exit_resubmits_gpu(launcher, status):
    launcher.update(SLURM_JOB_ID="42", HAD_TEST_EXIT=str(status))
    result = launch(launcher, "--worker", "gpu", "-A", "account", "-q", "qos")
    assert result.returncode == (0 if status == 75 else status), result.stderr
    rows = submissions(launcher)
    assert len(rows) == int(status == 75)
    if rows:
        assert "--dependency=afterany:42" in rows[0]
        assert rows[0][-4:] == ["-A", "account", "-q", "qos"]
    called = json.loads(Path(launcher["HAD_TEST_CALLED"]).read_text())
    assert called[called.index("--worker") + 1] == "gpu"
    assert called[called.index("--gpus") + 1] == "4"
    assert called[called.index("--gpu-wait-seconds") + 1] == "1200"


@pytest.mark.parametrize("role", ["gpu", "teacher"])
@pytest.mark.parametrize("signum", [signal.SIGUSR1, signal.SIGTERM])
def test_walltime_rollover_and_explicit_cancellation(launcher, role, signum):
    launcher.update(SLURM_JOB_ID="42", HAD_TEST_BLOCK="1")
    process = subprocess.Popen(["bash", "scripts/ops_had_cell.sh", "--worker", role,
        "--account=account"], env=launcher, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not Path(launcher["HAD_TEST_CALLED"]).exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert Path(launcher["HAD_TEST_CALLED"]).exists()
        process.send_signal(signum)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == (0 if signum == signal.SIGUSR1 else 143), stderr
        rows = submissions(launcher)
        assert len(rows) == int(signum == signal.SIGUSR1)
        if rows:
            assert rows[0][rows[0].index("--worker") + 1] == role
            assert "--dependency=afterany:42" in rows[0]
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def test_resume_only_one_member_and_reject_resource_override(launcher):
    result = launch(launcher, "--resume-role", "gpu", "--account=account")
    assert result.returncode == 0, result.stderr
    assert len(submissions(launcher)) == 1
    result = launch(launcher, "--gpus=1")
    assert result.returncode == 2
    assert len(submissions(launcher)) == 1


def test_cpu_worker_rejects_gpu_allocation(launcher):
    launcher["SLURM_JOB_GPUS"] = "0"
    result = launch(launcher, "--worker", "teacher")
    assert result.returncode == 2
    assert not Path(launcher["HAD_TEST_CALLED"]).exists()
    assert not submissions(launcher)
