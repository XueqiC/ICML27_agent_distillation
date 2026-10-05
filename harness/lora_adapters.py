"""CPU-only adapter identity checks and optional, byte-preserving compatibility copies."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import struct

from sed.storage import digest, file_hash, write_json
from train.modeling import LORA_TARGETS


def parse_adapters(values, base_name):
    result = []
    names = {base_name}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not path or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("LoRA must be NAME=PATH with a safe, nonempty NAME")
        if name in names:
            raise ValueError("Duplicate LoRA NAME or NAME equals base served-model-name")
        names.add(name)
        result.append((name, Path(path).expanduser().resolve()))
    return result


def inspect_adapter(name, path, student, snapshot):
    path, snapshot = Path(path).resolve(), Path(snapshot).resolve()
    config = json.loads((path / "adapter_config.json").read_text())
    state = json.loads((path / "training_state.json").read_text())
    provenance_path = path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if state["identity_sha256"] != digest(provenance):
        raise ValueError("Checkpoint provenance mismatch")
    trained = provenance["config"]["models"]["student"]
    if any(trained[k] != student[k] for k in ("requested_repo", "revision")):
        raise ValueError("Adapter base_model/revision differs from configured student")
    recorded = provenance["config"]["sft"]["base_snapshot"]
    # Paths may differ between rai and HPG. Compare the recorded metadata bytes.
    if not recorded["metadata_sha256"] or any(
        not (snapshot / key).is_file() or file_hash(snapshot / key) != value
        for key, value in recorded["metadata_sha256"].items()
    ):
        raise ValueError("Serving snapshot differs from adapter training snapshot metadata")
    base = config.get("base_model_name_or_path", config.get("base_model"))
    if base not in {student["requested_repo"], recorded["path"], str(snapshot)}:
        raise ValueError("Adapter base_model differs from configured student")
    if config.get("base_model") not in (None, base, student["requested_repo"]):
        raise ValueError("Conflicting adapter base_model")
    if config.get("revision") not in (None, student["revision"]):
        raise ValueError("Adapter revision differs from configured student")
    if (config.get("peft_type") != "LORA" or type(config.get("r")) is not int
            or config["r"] < 1 or config.get("rank_pattern") or config.get("alpha_pattern")
            or config.get("use_dora") or config.get("use_rslora")
            or config.get("modules_to_save") or config.get("bias", "none") != "none"):
        raise ValueError("Unsupported adapter configuration; no silent LoRA changes")
    weights = path / "adapter_model.safetensors"
    hashes = {p.name: file_hash(p) for p in (path / "adapter_config.json", weights)}
    if any(state["artifact_sha256"].get(k) != v for k, v in hashes.items()):
        raise ValueError("Adapter checkpoint artifacts changed after training")
    metadata = {k: state[k] for k in ("supervised_tokens", "passes", "packing")}
    return {"name": name, "path": str(path), "served_path": str(path), "rank": config["r"],
            "adapter_config_sha256": hashes["adapter_config.json"],
            "adapter_weights_sha256": hashes["adapter_model.safetensors"],
            "training_state_sha256": file_hash(path / "training_state.json"),
            "training_provenance_sha256": file_hash(provenance_path),
            "checkpoint_metadata": metadata, "base_model": student["requested_repo"],
            "revision": student["revision"], "conversion": None}


def safetensors_header(path):
    with Path(path).open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(handle.read(size))
    return header, size + 8


def payload_hash(path, offset):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def convert_adapter(adapter, destination, mode):
    """Explicit fallback only: targets, or targets plus HF->vLLM Gemma prefix.

    Native vLLM 0.27.1 already maps Gemma keys. Never rewrite weights in place,
    transpose, cast, merge, remove tensors, or change scaling/target coverage.
    """
    if mode not in {"targets", "gemma4"}:
        raise ValueError("Unknown LoRA conversion")
    source, destination = Path(adapter["path"]), Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError("Conversion must be outside checkpoint directory")
    config = json.loads((source / "adapter_config.json").read_text())
    targets = config["target_modules"]
    pattern = r"model\.language_model\.layers\.\d+\.(?:self_attn|mlp)\.(?:" + "|".join(LORA_TARGETS) + ")"
    if not isinstance(targets, list) or not targets or not all(re.fullmatch(pattern, t) for t in targets):
        raise ValueError("Conversion requires full text-only Gemma target_modules")
    weights = source / "adapter_model.safetensors"
    header, offset = safetensors_header(weights)
    rename = {}
    modules = set()
    for key in header:
        if key == "__metadata__":
            continue
        match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", key)
        if not match or match[1] not in targets:
            raise ValueError("Unexpected adapter tensor; conversion would change target coverage")
        modules.add(match[1])
        rename[key] = (key.replace("base_model.model.model.language_model.",
                                  "base_model.model.language_model.model.", 1)
                       if mode == "gemma4" else key)
    if modules != set(targets) or len(set(rename.values())) != len(rename):
        raise ValueError("Incomplete or colliding tensor rename")
    converted = {rename.get(k, k): v for k, v in header.items()}
    config["target_modules"] = sorted({t.rsplit(".", 1)[-1] for t in targets})
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "adapter_config.json", config, exclusive=True)
    target = destination / weights.name
    if mode == "targets":
        shutil.copyfile(weights, target)
    else:
        encoded = json.dumps(converted, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with weights.open("rb") as src, target.open("xb") as dst:
            dst.write(struct.pack("<Q", len(encoded)))
            dst.write(encoded)
            src.seek(offset)
            shutil.copyfileobj(src, dst)
    after, after_offset = safetensors_header(target)
    before_hash, after_hash = payload_hash(weights, offset), payload_hash(target, after_offset)
    if after != converted or before_hash != after_hash:
        raise ValueError("Conversion purity check failed")
    record = {"mode": mode, "config_changes": {"target_modules": {
                  "before": targets, "after": config["target_modules"]}},
              "tensor_renames": {k: v for k, v in rename.items() if k != v},
              "tensor_payload_sha256": before_hash, "tensor_bytes_identical": True,
              "adapter_config_sha256": file_hash(destination / "adapter_config.json"),
              "adapter_weights_sha256": file_hash(target)}
    write_json(destination / "conversion.json", record, exclusive=True)
    return {**adapter, "served_path": str(destination), "conversion": record}


def adapter_identity(adapter):
    return {k: adapter[k] for k in ("adapter_config_sha256", "adapter_weights_sha256",
                                   "training_state_sha256", "training_provenance_sha256",
                                   "base_model", "revision")}
