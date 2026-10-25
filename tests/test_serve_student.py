import io
import json
from pathlib import Path
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


@pytest.mark.parametrize("serving", ["base", "lora", "merged"])
@pytest.mark.parametrize("config_kind", ["gemma4", "gemma4_unified", "other", "Gemma4",
                                        "missing", "unreadable", "malformed", "non-object", "bad-encoding"])
def test_attention_backend_command(tmp_path, monkeypatch, serving, config_kind):
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
    base = tmp_path / "base"
    base.mkdir()
    model = base
    args = SimpleNamespace(model_path=str(base), merged_model_path=None,
                           python="server-python", port=23456, lora_dtype="float16")
    if serving == "merged":
        model = tmp_path / "merged"
        model.mkdir()
        args.merged_model_path = str(model)
        # Selection must use the served checkpoint, even if the base differs.
        (base / "config.json").write_text(json.dumps({
            "model_type": "gemma4_unified" if config_kind == "gemma4" else "gemma4"}))
        (model / "training_state.json").write_text(json.dumps({
            "control": "dose_matched_init", "text_export_dtype": "float32"}))
    config_path = model / "config.json"
    if config_kind != "missing":
        payload = {"malformed": b"{", "non-object": b"[]", "bad-encoding": b"\xff"}.get(
            config_kind, json.dumps({"model_type": config_kind}).encode())
        config_path.write_bytes(payload)
    if config_kind == "unreadable":
        read_text = Path.read_text
        def unreadable(path, *args, **kwargs):
            if path == config_path:
                raise PermissionError("unreadable config")
            return read_text(path, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", unreadable)
    adapters = [{"name": "sft", "rank": 16, "served_path": str(tmp_path / "adapter")}] if serving == "lora" else []
    student = {"requested_repo": "student", "revision": "pinned-revision"}
    # Exact command produced before the Gemma-4 backend override.
    expected = ["server-python", "-m", "vllm.entrypoints.cli.main", "serve", str(model),
                "--host", "127.0.0.1", "--port", "23456", "--served-model-name", "student",
                "--revision", "pinned-revision", "--tokenizer-revision", "pinned-revision",
                "--max-model-len", "32768", "--gpu-memory-utilization", "0.85",
                "--generation-config", "vllm", "--seed", "0"]
    if serving == "merged":
        expected += ["--dtype", "bfloat16"]
    expected += ["--reasoning-parser", "gemma4"]
    if serving == "lora":
        expected += ["--lora-dtype", "float16", "--enable-lora", "--max-lora-rank", "16",
                     "--max-loras", "1", "--lora-modules", "sft=" + str(tmp_path / "adapter")]
    command = server_command(args, student, {"gemma4_parser": True}, adapters)
    if config_kind == "gemma4":
        position = command.index("--seed") + 2
        assert command[position:position + 2] == ["--attention-backend", "TRITON_ATTN"]
        assert command.count("--attention-backend") == command.count("TRITON_ATTN") == 1
        command = command[:position] + command[position + 2:]
    else:
        assert "--attention-backend" not in command
    assert command == expected


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
