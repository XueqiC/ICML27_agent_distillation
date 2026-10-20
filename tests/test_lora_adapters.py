"""Gemma-4 serving derivations: synthetic safetensors bytes, CPU, no model loads."""
import copy
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from harness.lora_adapters import (GEMMA4_K_EQ_V_RULE, adapter_identity, convert_adapter,
                                   derive_gemma4_value_lora, inspect_adapter, safetensors_header)
from sed.storage import digest, file_hash, write_json
from tools.check_lora_serving import check_server, validate_report
from test_lora_serving import adapter_bank, bound_report


def module(layer, projection):
    return f"model.language_model.layers.{layer}.self_attn.{projection}_proj"


def tensor(layer, projection, part):
    return f"base_model.model.{module(layer, projection)}.lora_{part}.weight"


def save_tensors(path, tensors):
    header, payload = {"__metadata__": {"format": "pt"}}, b""
    for key, (dtype, shape, data) in tensors.items():
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def read_tensors(path):
    header, offset = safetensors_header(path)
    payload = path.read_bytes()[offset:]
    return {key: (spec["dtype"], spec["shape"], payload[slice(*spec["data_offsets"])])
            for key, spec in header.items() if key != "__metadata__"}


def bind_training(path, snapshot):
    """Refresh only synthetic training fixtures after constructing their bytes."""
    provenance_path = path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["config"]["sft"]["base_snapshot"]["metadata_sha256"] = {
        p.name: file_hash(p) for p in snapshot.glob("*.json")}
    write_json(provenance_path, provenance)
    state = json.loads((path / "training_state.json").read_text())
    state.update(identity_sha256=digest(provenance), artifact_sha256={
        name: file_hash(path / name) for name in ("adapter_config.json", "adapter_model.safetensors")})
    write_json(path / "training_state.json", state)


@pytest.fixture
def k_eq_v_bank(adapter_bank):
    path, snapshot, student = adapter_bank
    write_json(snapshot / "config.json", {"model_type": "gemma4_unified", "text_config": {
        "num_hidden_layers": 3, "attention_k_eq_v": True,
        "layer_types": ["sliding_attention", "full_attention", "full_attention"]}})
    write_json(snapshot / "model.safetensors.index.json", {"weight_map": {
        module(i, p) + ".weight": "absent-payload.safetensors"
        for i in range(3) for p in (("k", "v") if i == 0 else ("k",))}})
    tensors = read_tensors(path / "adapter_model.safetensors")
    # Include signed zero and NaN payload bits; numeric equality is insufficient.
    for i, dtype, data in [(1, "BF16", b"\x00\x80\xc1\x7f"),
                           (2, "F32", struct.pack("<II", 0x80000000, 0x7FC00001))]:
        for part, shape in [("A", [1, 2]), ("B", [2, 1])]:
            tensors[tensor(i, "k", part)] = (dtype, shape, data)
    save_tensors(path / "adapter_model.safetensors", tensors)
    config = json.loads((path / "adapter_config.json").read_text())
    config.update(r=1, lora_alpha=2)
    config["target_modules"] += [module(i, "k") for i in (1, 2)]
    write_json(path / "adapter_config.json", config)
    bind_training(path, snapshot)
    return path, snapshot, student


@pytest.mark.parametrize("snapshot_format", ["index", "header"])
@pytest.mark.parametrize("targets", ["full", "suffix"])
def test_value_derivation_bitwise_and_deterministic(k_eq_v_bank, tmp_path, snapshot_format, targets):
    path, snapshot, student = k_eq_v_bank
    if snapshot_format == "header":
        index = snapshot / "model.safetensors.index.json"
        names = json.loads(index.read_text())["weight_map"]
        # Header alone: any attempt to load these base tensors would fail.
        encoded = json.dumps({key: {"dtype": "F32", "shape": [100000, 100000],
                                   "data_offsets": [0, 40000000000]} for key in names}).encode()
        (snapshot / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
        index.unlink()
    if targets == "suffix":
        config = json.loads((path / "adapter_config.json").read_text())
        config["target_modules"] = ["q_proj", "k_proj", "v_proj"]
        write_json(path / "adapter_config.json", config)
    bind_training(path, snapshot)
    original = inspect_adapter("sft", path, student, snapshot)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    derived = derive_gemma4_value_lora(original, snapshot, tmp_path / "server/derived")
    again = derive_gemma4_value_lora(original, snapshot, tmp_path / "server/again")
    dest = Path(derived["served_path"])
    old = read_tensors(path / "adapter_model.safetensors")
    new = read_tensors(dest / "adapter_model.safetensors")
    assert {key: new[key] for key in old} == old
    assert set(new) - set(old) == {tensor(i, "v", p) for i in (1, 2) for p in ("A", "B")}
    for i in (1, 2):
        for part in ("A", "B"):
            assert new[tensor(i, "v", part)] == old[tensor(i, "k", part)]
    old_header, old_offset = safetensors_header(path / "adapter_model.safetensors")
    new_header, new_offset = safetensors_header(dest / "adapter_model.safetensors")
    assert {k: new_header[k] for k in old_header} == old_header
    old_payload = before["adapter_model.safetensors"][old_offset:]
    assert (dest / "adapter_model.safetensors").read_bytes()[new_offset:].startswith(old_payload)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    cfg = json.loads((dest / "adapter_config.json").read_text())
    previous = json.loads(before["adapter_config.json"])
    assert {k: v for k, v in cfg.items() if k != "target_modules"} == {
        k: v for k, v in previous.items() if k != "target_modules"}
    if targets == "full":
        assert cfg["target_modules"] == previous["target_modules"] + [module(i, "v") for i in (1, 2)]
    else:
        assert (dest / "adapter_config.json").read_bytes() == before["adapter_config.json"]
    assert derived["derivation"]["rule_id"] == GEMMA4_K_EQ_V_RULE
    assert derived["derivation"]["affected_layers"] == [1, 2]
    assert adapter_identity(derived) == adapter_identity(original)
    assert derived["path"] == str(path)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        assert (dest / name).read_bytes() == (Path(again["served_path"]) / name).read_bytes()


@pytest.mark.parametrize("scenario", ["base_has_v", "no_k_lora"])
def test_derivation_noop(k_eq_v_bank, tmp_path, scenario):
    path, snapshot, student = k_eq_v_bank
    if scenario == "base_has_v":
        config = json.loads((snapshot / "config.json").read_text())
        config["text_config"]["attention_k_eq_v"] = False
        write_json(snapshot / "config.json", config)
        index = snapshot / "model.safetensors.index.json"
        data = json.loads(index.read_text())
        data["weight_map"].update({module(i, "v") + ".weight": "base.safetensors" for i in (1, 2)})
        write_json(index, data)
    else:
        weights = path / "adapter_model.safetensors"
        save_tensors(weights, {k: v for k, v in read_tensors(weights).items() if ".k_proj." not in k})
    bind_training(path, snapshot)
    original = inspect_adapter("sft", path, student, snapshot)
    dest = tmp_path / "unused"
    assert derive_gemma4_value_lora(original, snapshot, dest) is original
    assert not dest.exists()


@pytest.mark.parametrize("scenario", ["extra_v", "missing_v", "flag_false", "missing_k"])
def test_config_index_disagreement_refused(k_eq_v_bank, tmp_path, scenario):
    path, snapshot, _ = k_eq_v_bank
    index = snapshot / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    if scenario == "extra_v":
        data["weight_map"][module(1, "v") + ".weight"] = "base.safetensors"
    elif scenario == "missing_v":
        del data["weight_map"][module(0, "v") + ".weight"]
    elif scenario == "missing_k":
        del data["weight_map"][module(2, "k") + ".weight"]
    else:
        config = json.loads((snapshot / "config.json").read_text())
        config["text_config"]["attention_k_eq_v"] = False
        write_json(snapshot / "config.json", config)
    write_json(index, data)
    with pytest.raises(ValueError, match="disagreement|Incomplete"):
        derive_gemma4_value_lora({"path": str(path)}, snapshot, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("parts", [("A",), ("B",), ("A", "B")])
def test_preexisting_value_lora_refused(k_eq_v_bank, tmp_path, parts):
    path, snapshot, _ = k_eq_v_bank
    weights = path / "adapter_model.safetensors"
    tensors = read_tensors(weights)
    for part in parts:
        tensors[tensor(1, "v", part)] = tensors[tensor(1, "k", part)]
    save_tensors(weights, tensors)
    with pytest.raises(ValueError, match="Pre-existing v_proj"):
        derive_gemma4_value_lora({"path": str(path)}, snapshot, tmp_path / "derived")


def test_incomplete_key_pair_and_checkpoint_destination_refused(k_eq_v_bank, tmp_path):
    path, snapshot, _ = k_eq_v_bank
    with pytest.raises(ValueError, match="outside checkpoint"):
        derive_gemma4_value_lora({"path": str(path)}, snapshot, path / "derived")
    weights = path / "adapter_model.safetensors"
    tensors = read_tensors(weights)
    del tensors[tensor(1, "k", "B")]
    save_tensors(weights, tensors)
    with pytest.raises(ValueError, match="Incomplete k_proj"):
        derive_gemma4_value_lora({"path": str(path)}, snapshot, tmp_path / "derived")


def server_record(adapter, snapshot):
    return {"config": {"server": {
        "lora_adapters": [adapter], "model_path": str(snapshot),
        "served_model_name": "base", "endpoint": "http://127.0.0.1:34567/v1", "merged_model": False}}}


@pytest.mark.parametrize("mode", ["none", "targets", "gemma4"])
def test_launcher_derivation_provenance_and_binding(k_eq_v_bank, tmp_path, monkeypatch, mode):
    import harness.serve_student as serving
    path, snapshot, student = k_eq_v_bank
    original = inspect_adapter("sft", path, student, snapshot)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    monkeypatch.setattr(serving, "inspect_runtime", lambda _: {
        "gemma4_parser": True, "parser_registry_path": str(snapshot / "config.json")})
    monkeypatch.setattr(serving.signal, "signal", lambda *a: None)
    monkeypatch.setattr(serving, "wait_ready", lambda *a, **k: None)
    monkeypatch.setattr(serving, "stop_server", lambda *a: None)
    monkeypatch.setattr(serving.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, wait=lambda: 0))
    monkeypatch.setattr(serving, "write_provenance", lambda out, **kw: write_json(out / "provenance.json", kw))
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): pass
    monkeypatch.setattr(serving.socket, "socket", lambda *a: Socket())
    output = tmp_path / "server"
    assert serving.main(["--python", "fake", "--model-path", str(snapshot), "--port", "34567",
                         "--lora-adapter", "sft=" + str(path), "--lora-adapter", "other=" + str(path),
                         "--lora-conversion", mode, "--output", str(output)]) == 0
    provenance = json.loads((output / "provenance.json").read_text())
    info = provenance["config"]["server"]
    for row in info["lora_adapters"]:
        derived = row["derivation"]
        assert Path(derived["path"]).is_relative_to(output)
        assert Path(row["served_path"]).is_relative_to(output)
        assert derived["rule_id"] == GEMMA4_K_EQ_V_RULE and derived["affected_layers"] == [1, 2]
        for filename, field in [("adapter_config.json", "adapter_config_sha256"),
                                ("adapter_model.safetensors", "adapter_weights_sha256")]:
            assert derived[field] == file_hash(Path(derived["path"]) / filename)
        assert adapter_identity(row) == adapter_identity(original) and row["path"] == str(path)
        assert row["name"] + "=" + row["served_path"] in info["command"]
        assert (row["conversion"] is None) is (mode == "none")
        all_bytes = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        assert check_server(provenance, info["endpoint"], row["name"], original) == student["requested_repo"]
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == all_bytes
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    with pytest.raises(ValueError, match="outside training checkpoint"):
        serving.main(["--python", "fake", "--model-path", str(snapshot), "--port", "34567",
                      "--lora-adapter", "sft=" + str(path), "--output", str(path / "server")])
    assert not (path / "server").exists()


@pytest.mark.parametrize("mode", ["none", "targets", "gemma4"])
@pytest.mark.parametrize("tamper", ["weights", "config", "rule", "layers", "missing_rule", "served_path",
                                   "original", "conversion"])
def test_binding_recomputes_derivation_and_rejects_tampering(k_eq_v_bank, tmp_path, mode, tamper):
    path, snapshot, student = k_eq_v_bank
    original = inspect_adapter("sft", path, student, snapshot)
    row = derive_gemma4_value_lora(original, snapshot, tmp_path / "derived")
    if mode != "none":
        row = convert_adapter(row, tmp_path / "converted", mode)
    server = server_record(row, snapshot)
    endpoint = server["config"]["server"]["endpoint"]
    assert check_server(server, endpoint, "sft", original) == "base"
    if tamper in {"weights", "config", "original", "conversion"}:
        target = path if tamper == "original" else Path(
            row["served_path"] if tamper == "conversion" else row["derivation"]["path"])
        filename = "adapter_config.json" if tamper == "config" else "adapter_model.safetensors"
        data = bytearray((target / filename).read_bytes())
        data[-2 if tamper == "config" else -1] ^= 1
        (target / filename).write_bytes(data)
        # Forging the recorded hash must not bypass recomputation.
        if tamper != "original":
            record = row["conversion"] if tamper == "conversion" and mode != "none" else row["derivation"]
            field = "adapter_config_sha256" if tamper == "config" else "adapter_weights_sha256"
            record[field] = file_hash(target / filename)
    elif tamper == "rule":
        row["derivation"]["rule_id"] = "unknown_rule"
    elif tamper == "layers":
        row["derivation"]["affected_layers"] = [1]
    elif tamper == "missing_rule":
        row.pop("derivation")
        row["served_path"] = str(path)
    else:
        row["served_path"] = str(path)
    with pytest.raises(ValueError):
        check_server(server, endpoint, "sft", original)


def test_report_gate_binds_original_hf_identity_to_derived_bytes(k_eq_v_bank, tmp_path, bound_report):
    path, snapshot, student = k_eq_v_bank
    original = inspect_adapter("sft", path, student, snapshot)
    row = derive_gemma4_value_lora(original, snapshot, tmp_path / "derived")
    server = server_record(row, snapshot)
    report = copy.deepcopy(bound_report[0])
    report.update(server_provenance_sha256=digest(server), identity={"adapter": adapter_identity(original)})
    endpoint = server["config"]["server"]["endpoint"]
    validate_report(report, server, "sft", endpoint)
    with (Path(row["served_path"]) / "adapter_model.safetensors").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_report(report, server, "sft", endpoint)
