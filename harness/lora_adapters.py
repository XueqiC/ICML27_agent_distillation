"""CPU-only adapter identity checks and deterministic serving compatibility copies."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import struct

from sed.storage import canonical_bytes, digest, file_hash, write_json
from train.modeling import LORA_TARGETS

GEMMA4_K_EQ_V_RULE = "gemma4_k_eq_v_value_lora_v1"


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


def _encoded_header(header):
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded


def _payload_chunks(weights, offset, copies=()):
    """Stream original bytes, then the specified original tensor byte ranges."""
    with Path(weights).open("rb") as handle:
        for start, end in [(0, Path(weights).stat().st_size - offset), *copies]:
            handle.seek(offset + start)
            remaining = end - start
            while remaining:
                chunk = handle.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError("Truncated adapter tensor payload")
                remaining -= len(chunk)
                yield chunk


def _artifact_hashes(config_bytes, prefix, chunks):
    value = hashlib.sha256(prefix)
    for chunk in chunks:
        value.update(chunk)
    return {"adapter_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "adapter_weights_sha256": value.hexdigest()}


def _gemma4_k_eq_v_layers(snapshot):
    """Cross-check every text decoder layer using only config and weight names."""
    snapshot = Path(snapshot)
    config = json.loads((snapshot / "config.json").read_text())
    text = config["text_config"]
    count, types = text["num_hidden_layers"], text["layer_types"]
    if (type(count) is not int or count < 1 or len(types) != count
            or type(text.get("attention_k_eq_v")) is not bool
            or any(t not in {"full_attention", "sliding_attention"} for t in types)):
        raise ValueError("Invalid Gemma-4 k_eq_v layer configuration")
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        names = json.loads(index.read_text())["weight_map"]
    else:
        shards = sorted(snapshot.glob("*.safetensors"))
        if not shards:
            raise ValueError("Gemma-4 snapshot needs a safetensors index or headers")
        names = {key for shard in shards for key in safetensors_header(shard)[0]}
    projections = {}
    for name in names:
        match = re.fullmatch(
            r"(?:model\.)?language_model(?:\.model)?\.layers\.(\d+)\.self_attn\.([kv])_proj\.weight",
            name)
        if match:
            layer, projection = int(match[1]), match[2]
            if projection in projections.setdefault(layer, set()):
                raise ValueError("Ambiguous Gemma-4 snapshot projection names")
            projections[layer].add(projection)
    if set(projections) != set(range(count)) or any("k" not in p for p in projections.values()):
        raise ValueError("Incomplete Gemma-4 decoder layer weights in snapshot index/headers")
    missing = {i for i, p in projections.items() if "v" not in p}
    configured = {i for i, t in enumerate(types) if text["attention_k_eq_v"] and t == "full_attention"}
    if missing != configured:
        raise ValueError("Gemma-4 k_eq_v config/index disagreement: "
                         f"missing v_proj layers {sorted(missing)}, configured {sorted(configured)}")
    return sorted(configured)


def _value_lora_plan(source, snapshot):
    """Deterministic bytes for the serving copy; never load base weight payloads."""
    layers = _gemma4_k_eq_v_layers(snapshot)
    weights = source / "adapter_model.safetensors"
    header, offset = safetensors_header(weights)
    original_size = size = weights.stat().st_size - offset
    copies, modules, affected = [], [], []
    for layer in layers:
        prefix = f"base_model.model.model.language_model.layers.{layer}.self_attn."
        keys = [prefix + f"k_proj.lora_{part}.weight" for part in ("A", "B")]
        values = [key.replace(".k_proj.", ".v_proj.") for key in keys]
        if any(key in header for key in values):
            raise ValueError(f"Pre-existing v_proj LoRA on k_eq_v layer {layer}")
        if not any(key in header for key in keys):
            continue
        if not all(key in header for key in keys):
            raise ValueError(f"Incomplete k_proj LoRA A/B pair on layer {layer}")
        affected.append(layer)
        modules.append(prefix.removeprefix("base_model.model.") + "v_proj")
        for key, value in zip(keys, values):
            start, end = header[key]["data_offsets"]
            if not 0 <= start < end <= original_size:
                raise ValueError("Invalid k_proj LoRA tensor offsets")
            copies.append((start, end))
            header[value] = {**header[key], "data_offsets": [size, size + end - start]}
            size += end - start
    if not affected:
        return None
    config_bytes = (source / "adapter_config.json").read_bytes()
    config = json.loads(config_bytes)
    targets = config["target_modules"]
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise ValueError("k_eq_v derivation requires a target_modules list")
    additions = [module for module in modules
                 if not any(module == t or module.endswith("." + t) for t in targets)]
    if additions:
        config["target_modules"] = targets + additions
        config_bytes = canonical_bytes(config) + b"\n"
    return config_bytes, header, offset, copies, affected


def derive_gemma4_value_lora(adapter, snapshot, destination):
    """Serving equivalence only: duplicate each absent k_eq_v value LoRA from K.

    Run before optional naming conversion. Original tensor payloads, dtypes,
    shapes and scaling remain unchanged; no-op adapters keep their original path.
    """
    source, destination = Path(adapter["path"]).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError("Derivation must be outside checkpoint directory")
    plan = _value_lora_plan(source, snapshot)
    if plan is None:
        return adapter
    config_bytes, header, offset, copies, layers = plan
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "adapter_config.json").open("xb") as handle:
        handle.write(config_bytes)
    with (destination / "adapter_model.safetensors").open("xb") as handle:
        handle.write(_encoded_header(header))
        for chunk in _payload_chunks(source / "adapter_model.safetensors", offset, copies):
            handle.write(chunk)
    record = {"rule_id": GEMMA4_K_EQ_V_RULE, "affected_layers": layers, "path": str(destination),
              "adapter_config_sha256": file_hash(destination / "adapter_config.json"),
              "adapter_weights_sha256": file_hash(destination / "adapter_model.safetensors")}
    return {**adapter, "served_path": str(destination), "derivation": record}


def _conversion_plan(source, mode):
    if mode not in {"targets", "gemma4"}:
        raise ValueError("Unknown LoRA conversion")
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
    record = {"mode": mode, "config_changes": {"target_modules": {
                  "before": targets, "after": config["target_modules"]}},
              "tensor_renames": {k: v for k, v in rename.items() if k != v},
              "tensor_payload_sha256": payload_hash(weights, offset), "tensor_bytes_identical": True}
    return config, converted, offset, record


def convert_adapter(adapter, destination, mode):
    """Explicit naming fallback, after any mandatory serving derivation.

    Native vLLM 0.27.1 already maps Gemma keys. Never rewrite weights in place,
    transpose, cast, merge, remove tensors, or change scaling/target coverage.
    """
    source = Path(adapter.get("served_path", adapter["path"])).resolve()
    destination = Path(destination).resolve()
    if any(destination.is_relative_to(p) for p in (source, Path(adapter["path"]).resolve())):
        raise ValueError("Conversion must be outside checkpoint directory")
    config, converted, offset, record = _conversion_plan(source, mode)
    weights = source / "adapter_model.safetensors"
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "adapter_config.json", config, exclusive=True)
    target = destination / weights.name
    if mode == "targets":
        shutil.copyfile(weights, target)
    else:
        with weights.open("rb") as src, target.open("xb") as dst:
            dst.write(_encoded_header(converted))
            src.seek(offset)
            shutil.copyfileobj(src, dst)
    after, after_offset = safetensors_header(target)
    before_hash, after_hash = payload_hash(weights, offset), payload_hash(target, after_offset)
    if after != converted or before_hash != after_hash:
        raise ValueError("Conversion purity check failed")
    record.update(adapter_config_sha256=file_hash(destination / "adapter_config.json"),
                  adapter_weights_sha256=file_hash(target))
    write_json(destination / "conversion.json", record, exclusive=True)
    return {**adapter, "served_path": str(destination), "conversion": record}


def validate_served_adapter(adapter, snapshot=None):
    """Recompute derivation/conversion hashes from original bytes, without writes."""
    def check_files(path, expected):
        for name, field in (("adapter_config.json", "adapter_config_sha256"),
                            ("adapter_model.safetensors", "adapter_weights_sha256")):
            if file_hash(path / name) != expected[field]:
                raise ValueError("Served adapter artifact hash mismatch")

    source = Path(adapter["path"]).resolve()
    check_files(source, adapter)
    derived = adapter.get("derivation")
    plan = _value_lora_plan(source, snapshot) if snapshot is not None else None
    if derived is not None:
        if plan is None:
            raise ValueError("Unexpected k_eq_v derivation or missing base snapshot")
        config_bytes, header, offset, copies, layers = plan
        expected = {"rule_id": GEMMA4_K_EQ_V_RULE, "affected_layers": layers,
                    "path": derived["path"], **_artifact_hashes(
                        config_bytes, _encoded_header(header),
                        _payload_chunks(source / "adapter_model.safetensors", offset, copies))}
        if derived != expected:
            raise ValueError("Serving derivation differs from documented k_eq_v rule")
        source = Path(derived["path"]).resolve()
        check_files(source, expected)
    elif plan is not None:
        raise ValueError("Missing required k_eq_v serving derivation")
    conversion = adapter.get("conversion")
    served = Path(adapter["served_path"]).resolve()
    if conversion is not None:
        mode = conversion["mode"]
        config, header, offset, expected = _conversion_plan(source, mode)
        weights = source / "adapter_model.safetensors"
        with weights.open("rb") as handle:
            prefix = handle.read(offset) if mode == "targets" else _encoded_header(header)
        expected.update(_artifact_hashes(canonical_bytes(config) + b"\n", prefix,
                                         _payload_chunks(weights, offset)))
        if conversion != expected:
            raise ValueError("Serving conversion differs from documented naming rule")
        check_files(served, expected)
    elif served != source:
        raise ValueError("Undocumented served adapter path")


def adapter_identity(adapter):
    return {k: adapter[k] for k in ("adapter_config_sha256", "adapter_weights_sha256",
                                   "training_state_sha256", "training_provenance_sha256",
                                   "base_model", "revision")}
