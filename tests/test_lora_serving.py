"""Offline CPU coverage; synthetic checkpoints, fake HTTP, no vLLM imports."""
import json
from pathlib import Path
import shutil
import struct
from types import SimpleNamespace

import httpx
import pytest

from harness.lora_adapters import (adapter_identity, convert_adapter, inspect_adapter,
                                   parse_adapters, safetensors_header)
from harness.serve_student import free_port, server_command
from sed.config import load_config
from sed.storage import digest, file_hash, write_json
from tools.check_lora_serving import (compare_prompts, comparison, fixed_prompts,
                                     validate_report, vllm_logprobs)


@pytest.fixture
def adapter_bank(tmp_path):
    student = load_config()["models"]["student"]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    write_json(snapshot / "config.json", {"model_type": "gemma4_unified"})
    adapter = tmp_path / "train" / "checkpoint"
    adapter.mkdir(parents=True)
    module = "model.language_model.layers.0.self_attn.q_proj"
    config = {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "bias": "none",
              "base_model_name_or_path": str(snapshot), "revision": None,
              "target_modules": [module]}
    write_json(adapter / "adapter_config.json", config)
    prefix = "base_model.model." + module
    header = {prefix + ".lora_A.weight": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]},
              prefix + ".lora_B.weight": {"dtype": "F32", "shape": [2, 1], "data_offsets": [8, 16]},
              "__metadata__": {"format": "pt"}}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (adapter / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + struct.pack("<ffff", 1.0, -0.0, 2.0, 3.0))
    provenance = {"config": {"models": {"student": student}, "sft": {"base_snapshot": {
        "path": str(snapshot), "metadata_sha256": {"config.json": file_hash(snapshot / "config.json")}}}}}
    write_json(adapter.parent / "provenance.json", provenance)
    state = {"supervised_tokens": 10501, "passes": 1.05, "packing": "episode",
             "identity_sha256": digest(provenance), "artifact_sha256": {
                 p.name: file_hash(p) for p in adapter.iterdir()}}
    write_json(adapter / "training_state.json", state)
    return adapter, snapshot, student


def test_adapter_identity_and_command(adapter_bank, monkeypatch):
    path, snapshot, student = adapter_bank
    row = inspect_adapter("sft", path, student, snapshot)
    assert row["checkpoint_metadata"] == {"supervised_tokens": 10501, "passes": 1.05, "packing": "episode"}
    assert row["adapter_weights_sha256"] == file_hash(path / "adapter_model.safetensors")
    assert row["adapter_config_sha256"] == file_hash(path / "adapter_config.json")
    args = SimpleNamespace(model_path=snapshot, merged_model_path=None, python="fake", port=23456)
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.42")
    cmd = server_command(args, student, {"gemma4_parser": True}, [row, dict(row, name="later", rank=32)])
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.42"
    assert cmd[cmd.index("--max-lora-rank") + 1] == "32"
    assert cmd[cmd.index("--max-loras") + 1] == "2"
    assert cmd[cmd.index("--lora-modules") + 1:] == [f"sft={path}", f"later={path}"]
    args.merged_model_path = str(snapshot)
    with pytest.raises(ValueError, match="unmerged"):
        server_command(args, student, {"gemma4_parser": False}, [row])
    assert "--enable-lora" not in server_command(args, student, {"gemma4_parser": False})
    for value in ("nan", "inf", "0", "1.1", "bad"):
        monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", value)
        with pytest.raises(ValueError):
            server_command(args, student, {"gemma4_parser": False})


@pytest.mark.parametrize("values", [["base=/tmp"], ["a=/tmp", "a=/tmp/b"],
                                    ["=x"], ["a="], ["a"], ["../x=y"], ["-x=y"]])
def test_names_refused(values):
    with pytest.raises(ValueError):
        parse_adapters(values, "base")


def test_names_paths_with_spaces_and_equals(tmp_path):
    path = tmp_path / "space = path"
    assert parse_adapters(["sft=" + str(path)], "base") == [("sft", path)]


@pytest.mark.parametrize("field,value", [("base_model_name_or_path", "wrong/model"),
                                         ("revision", "wrong"), ("rank_pattern", {"q_proj": 8})])
def test_adapter_mismatch_refused(adapter_bank, field, value):
    path, snapshot, student = adapter_bank
    config = json.loads((path / "adapter_config.json").read_text())
    config[field] = value
    write_json(path / "adapter_config.json", config)
    with pytest.raises(ValueError):
        inspect_adapter("sft", path, student, snapshot)


def test_provenance_and_weights_tamper_refused(adapter_bank):
    path, snapshot, student = adapter_bank
    with (path / "adapter_model.safetensors").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        inspect_adapter("sft", path, student, snapshot)


def test_relocated_snapshot_and_training_revision(adapter_bank, tmp_path):
    path, snapshot, student = adapter_bank
    relocated = tmp_path / "hpg-snapshot"
    shutil.copytree(snapshot, relocated)
    assert inspect_adapter("sft", path, student, relocated)["revision"] == student["revision"]
    provenance_path = path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["config"]["models"]["student"]["revision"] = "different"
    write_json(provenance_path, provenance)
    state = json.loads((path / "training_state.json").read_text())
    state["identity_sha256"] = digest(provenance)
    write_json(path / "training_state.json", state)
    with pytest.raises(ValueError, match="revision"):
        inspect_adapter("sft", path, student, relocated)


@pytest.mark.parametrize("mode", ["targets", "gemma4"])
def test_conversion_pure_rename(adapter_bank, tmp_path, mode):
    path, snapshot, student = adapter_bank
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    original = inspect_adapter("sft", path, student, snapshot)
    dest = tmp_path / "out" / mode
    row = convert_adapter(original, dest, mode)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    assert row["conversion"]["tensor_bytes_identical"] is True
    old_header, old_offset = safetensors_header(path / "adapter_model.safetensors")
    new_header, new_offset = safetensors_header(dest / "adapter_model.safetensors")
    assert before["adapter_model.safetensors"][old_offset:] == (dest / "adapter_model.safetensors").read_bytes()[new_offset:]
    assert list(old_header.values()) == list(new_header.values())
    cfg = json.loads((dest / "adapter_config.json").read_text())
    old_cfg = json.loads(before["adapter_config.json"])
    assert cfg.pop("target_modules") == ["q_proj"]
    old_cfg.pop("target_modules")
    assert cfg == old_cfg  # rank, alpha, scaling, and every other field unchanged
    assert adapter_identity(row) == adapter_identity(original)
    if mode == "targets":
        assert row["conversion"]["adapter_weights_sha256"] == original["adapter_weights_sha256"]
    else:
        assert all("base_model.model.language_model.model." in k
                   for k in new_header if k != "__metadata__")
    with pytest.raises(ValueError, match="outside"):
        convert_adapter(original, path / "conversion", mode)


def test_select_port_by_binding(monkeypatch):
    calls = []
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): calls.append("close")
        def bind(self, address): calls.append(address)
        def getsockname(self): return "127.0.0.1", 34567
    monkeypatch.setattr("harness.serve_student.socket.socket", lambda *a: Socket())
    assert free_port() == 34567
    assert calls == [("127.0.0.1", 0), "close"]


def test_listener_must_belong_to_child_group(tmp_path, monkeypatch):
    from harness.serve_student import owns_listener
    (tmp_path / "net").mkdir()
    (tmp_path / "net/tcp").write_text(
        "header\n0: 0100007F:8707 00000000:0000 0A 0 0 0 0 0 987\n")
    directory = tmp_path / "123/fd"
    directory.mkdir(parents=True)
    (directory / "4").symlink_to("socket:[987]")
    monkeypatch.setattr("harness.serve_student.os.getpgid", lambda pid: 456)
    assert owns_listener(456, 34567, proc=tmp_path)
    assert not owns_listener(999, 34567, proc=tmp_path)
    assert not owns_listener(456, 34568, proc=tmp_path)


def test_server_provenance_and_all_models_readiness(adapter_bank, tmp_path, monkeypatch):
    import harness.serve_student as serving
    path, snapshot, student = adapter_bank
    monkeypatch.setattr(serving, "inspect_runtime", lambda _: {
        "gemma4_parser": True, "vllm": "fake", "parser_registry_path": str(snapshot / "config.json")})
    monkeypatch.setattr(serving, "stop_server", lambda _: None)
    monkeypatch.setattr(serving.signal, "signal", lambda *a: None)
    monkeypatch.setattr(serving.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, wait=lambda: 0))
    seen = []
    monkeypatch.setattr(serving, "wait_ready", lambda process, port, timeout, models, **kw: seen.extend(models))
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): pass
        def getsockname(self): return "127.0.0.1", 34567
    monkeypatch.setattr(serving.socket, "socket", lambda *a: Socket())
    # Provenance uses subprocess.run; avoid globally replacing Popen's internals.
    monkeypatch.setattr(serving, "write_provenance", lambda out, **kw: write_json(out / "provenance.json", kw))
    out = tmp_path / "server"
    assert serving.main(["--python", "fake", "--model-path", str(snapshot), "--port", "0",
                         "--lora-adapter", "sft=" + str(path), "--output", str(out)]) == 0
    record = json.loads((out / "provenance.json").read_text())["config"]["server"]
    assert seen == [student["requested_repo"], "sft"]
    assert record["port"] > 0 and str(record["port"]) in record["endpoint"]
    assert record["lora_adapters"][0] == inspect_adapter("sft", path, student, snapshot)
    assert not (out / "ready.json").exists()


def cases():
    return [{"input_ids": [2, 3, 10, 106], "prompt_tokens": 2,
             "input_sha256": digest([2, 3, 10, 106]), "hf_logprobs": [-1.0, -2.0],
             "source": {"task_id": "train/task/1", "turn_index": i,
                        "training_eligible": True}} for i in range(8)]


def fake_completion(request):
    body = json.loads(request.content)
    assert body["prompt"] == [2, 3, 10, 106]
    assert body["prompt_logprobs"] == 1 and body["return_token_ids"]
    assert body["add_special_tokens"] is False and body["max_tokens"] == 0
    values = [-1.01, -2.01] if body["model"] == "sft" else [-1.3, -2.3]
    return httpx.Response(200, json={"model": body["model"], "choices": [{
        "prompt_token_ids": body["prompt"], "prompt_logprobs": [None, {"3": {"logprob": -9}},
            {"10": {"logprob": values[0]}}, {"106": {"logprob": values[1]}}]}]})


def test_logprob_equivalence_and_report_gate(adapter_bank):
    path, snapshot, student = adapter_bank
    row = inspect_adapter("sft", path, student, snapshot)
    with httpx.Client(transport=httpx.MockTransport(fake_completion)) as client:
        report = compare_prompts(cases(), client, "http://127.0.0.1:34567/v1", "sft", "base")
    assert report["passed"] and len(report["prompts"]) == 8
    assert report["aggregate"]["adapter_vllm_vs_hf"]["mean_abs_diff"] == pytest.approx(0.01)
    server = {"config": {"server": {"lora_adapters": [row], "served_model_name": "base",
                                    "endpoint": "http://127.0.0.1:34567/v1", "merged_model": False}}}
    report.update(model_name="sft", server_provenance_sha256=digest(server),
                  identity={"adapter": adapter_identity(row)})
    validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")
    with pytest.raises(ValueError, match="Endpoint"):
        validate_report(report, server, "sft", "http://127.0.0.1:34568/v1")
    report["server_provenance_sha256"] = "stale"
    with pytest.raises(ValueError, match="passing"):
        validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


def test_thresholds_fail_closed():
    assert comparison([-1], [-1.02], [-2])["passed"]
    assert not comparison([-1], [-1.06], [-2])["passed"]
    assert not comparison([-1], [-1.02], [-1.05])["passed"]
    assert not comparison([-1], [-1], [-1])["passed"]
    for values in (([], [], []), ([-1], [], [-1]), ([float("nan")], [-1], [-2])):
        with pytest.raises(ValueError): comparison(*values)


def test_prompt_ids_must_match():
    def fake(request):
        response = fake_completion(request)
        data = response.json()
        data["choices"][0]["prompt_token_ids"] = [99]
        return httpx.Response(200, json=data)
    with httpx.Client(transport=httpx.MockTransport(fake)) as client:
        with pytest.raises(ValueError, match="alignment"):
            vllm_logprobs(client, "http://127.0.0.1/v1", "sft", cases()[0])


def test_fixed_selection_training_only():
    examples = [{"input_ids": [2, 3, 106], "prompt_tokens": 2, "packing": "turn",
                 "label_provenance": {"task_id": "train/task/1", "turn_index": i,
                                      "training_eligible": True}} for i in reversed(range(10))]
    assert [r["source"]["turn_index"] for r in fixed_prompts(examples)] == list(range(8))
    examples[-1]["label_provenance"]["training_eligible"] = False
    with pytest.raises(ValueError, match="training-eligible"):
        fixed_prompts(examples)
    with pytest.raises(ValueError, match="eight"):
        fixed_prompts(examples[:2])


@pytest.mark.parametrize("passes", [True, False])
def test_check_cli_json_report_and_exit_status(adapter_bank, tmp_path, monkeypatch, passes):
    import tools.check_lora_serving as check
    from sed.provenance import write_provenance
    path, snapshot, student = adapter_bank
    adapter = inspect_adapter("sft", path, student, snapshot)
    verified = tmp_path / "verified"
    verified.mkdir()
    write_json(verified / "manifest.json", {})
    identity = check.source_identity(SimpleNamespace(verified=str(verified), snapshot=str(snapshot)),
                                     load_config(), adapter)
    hf_dir = tmp_path / "hf"
    old = write_provenance(hf_dir, command="check_lora_serving", config={})
    write_json(hf_dir / "reference.json", {"identity": identity, "prompts": cases(),
               "provenance_sha256": digest(old), "template_sha256": "fake-template"})
    server = {"config": {"server": {"lora_adapters": [adapter], "served_model_name": "base",
                                    "endpoint": "http://127.0.0.1:34567/v1", "merged_model": False}}}
    server_file = tmp_path / "server.json"
    write_json(server_file, server)
    def completion(request):
        data = fake_completion(request).json()
        if not passes and data["model"] == "sft":
            data["choices"][0]["prompt_logprobs"][2]["10"]["logprob"] = -1.5
        return httpx.Response(200, json=data)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(completion), **kw))
    output = tmp_path / "check"
    status = check.main(["--snapshot", str(snapshot), "--adapter", str(path),
                        "--verified", str(verified), "--model-name", "sft",
                        "--endpoint", "http://127.0.0.1:34567/v1", "--server-provenance", str(server_file),
                        "--hf-reference", str(hf_dir / "reference.json"), "--output", str(output)])
    assert status == (0 if passes else 1)
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is passes and len(report["prompts"]) == 8
    assert (output / "provenance.json").exists()
