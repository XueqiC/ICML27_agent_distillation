import io
import json
from types import SimpleNamespace

import pytest

from harness.serve_student import job_port, server_command, wait_ready
from sed.config import load_config


@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_server_records_and_passes_batch_invariant(tmp_path, monkeypatch, setting):
    import harness.serve_student as server
    if setting is None:
        monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    else:
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", setting)
    model = tmp_path / "model"
    model.mkdir()
    registry = tmp_path / "registry.py"
    registry.write_text("# fake runtime metadata")
    monkeypatch.setattr(server, "inspect_runtime", lambda _: {
        "gemma4_parser": False, "parser_registry_path": str(registry)})
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def bind(self, address): pass
    monkeypatch.setattr(server.socket, "socket", lambda *_: Socket())
    monkeypatch.setattr(server.signal, "signal", lambda *_: None)
    captured = {}
    def popen(command, *, env, **kwargs):
        captured["batch_invariant"] = env["VLLM_BATCH_INVARIANT"]
        return SimpleNamespace(pid=123, wait=lambda: 0)
    monkeypatch.setattr(server.subprocess, "Popen", popen)
    # Real provenance writing uses subprocess.run for git, not our fake child.
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="fake-head"))
    monkeypatch.setattr(server, "wait_ready", lambda *a, **k: None)
    monkeypatch.setattr(server, "stop_server", lambda *_: None)
    out = tmp_path / "out"
    assert server.main(["--python", "fake-python", "--model-path", str(model),
                        "--port", "23456", "--output", str(out)]) == 0
    config = json.loads((out / "provenance.json").read_text())["config"]["server"]
    assert config["batch_invariant"] is (setting == "1")
    assert config["environment"]["VLLM_BATCH_INVARIANT"] == captured["batch_invariant"] == (setting or "0")


def test_invalid_batch_invariant_fails_before_server(tmp_path, monkeypatch):
    from harness.serve_student import main
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "true")
    with pytest.raises(SystemExit):
        main(["--python", "unused", "--model-path", str(tmp_path), "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


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
