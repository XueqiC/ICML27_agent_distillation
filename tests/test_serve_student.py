import io
import json
from types import SimpleNamespace

import pytest

from harness.serve_student import job_port, server_command, wait_ready
from sed.config import load_config


def test_launcher_command_and_job_port(tmp_path):
    args = SimpleNamespace(model_path=str(tmp_path), merged_model_path=None, python="server-python", port=23456)
    spec = load_config()["models"]["student"]
    command = server_command(args, spec, {"gemma4_parser": True})
    assert command[:5] == ["server-python", "-m", "vllm.entrypoints.cli.main", "serve", str(tmp_path)]
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.85"
    assert command[-2:] == ["--reasoning-parser", "gemma4"]
    assert spec["revision"] in command
    assert "--reasoning-parser" not in server_command(args, spec, {"gemma4_parser": False})
    assert job_port("12345678") == job_port("12345678") == 25678
    with pytest.raises(ValueError):
        job_port("")


def test_readiness_and_dead_server():
    class Opener:
        def open(self, url, timeout):
            return io.BytesIO(json.dumps({"data": [{"id": "student"}]}).encode())
    process = SimpleNamespace(poll=lambda: None)
    wait_ready(process, 23456, 1, "student", opener=Opener())
    with pytest.raises(RuntimeError, match="exited"):
        wait_ready(SimpleNamespace(poll=lambda: 1), 23456, 1, "student", opener=Opener())
    with pytest.raises(TimeoutError):
        wait_ready(process, 23456, 0, "student", opener=Opener())
