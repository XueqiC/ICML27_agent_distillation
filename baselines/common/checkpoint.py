"""Full checkpoint names and metadata; header inspection does not import torch."""
import json
from pathlib import Path
import struct


def checkpoint_layout(path):
    """Read actual tensor headers, also checking any HF shard index against them."""
    path = Path(path)
    tensors = {}
    for shard in sorted(path.glob("*.safetensors")):
        with shard.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size > 16 * 1024 * 1024:
                raise ValueError("Invalid safetensors header length")
            header = json.loads(handle.read(size))
        for key, spec in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError(f"Duplicate checkpoint tensor: {key}")
            tensors[key] = {"file": shard.name, "shape": spec["shape"], "dtype": spec["dtype"]}
    if not tensors:
        raise ValueError("Checkpoint has no safetensors tensors")
    index = path / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        if mapping != {key: spec["file"] for key, spec in tensors.items()}:
            raise ValueError("Checkpoint shard index differs from tensor headers")
    return tensors


def checkpoint_key_map(model, base_keys, *, state_dict=None):
    """Invert Transformers' base-file -> model renamings without touching weights.

    Use the registered rules, even if a previous load only used a subset. The
    original base's stored names determine this export's namespace. Gemma4's
    conversions are renamings only; reject tensor transforms rather than
    pretending that a merge/split can be represented by a key map.
    """
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightRenaming, rename_source_key

    conversions = get_model_conversion_mapping(model, add_legacy=False)
    if any(not isinstance(rule, WeightRenaming) for rule in conversions):
        raise ValueError("Checkpoint export requires one-to-one tensor renamings")
    state_dict = model.state_dict() if state_dict is None else state_dict
    mapping = {}
    for file_name in sorted(base_keys):
        model_name, _ = rename_source_key(file_name, conversions, [])
        if model_name in mapping:
            raise ValueError(f"Checkpoint conversion collision: {model_name}")
        if model_name not in state_dict:
            raise ValueError(f"Base checkpoint tensor missing from model: {file_name}")
        mapping[model_name] = file_name

    tied = getattr(model, "all_tied_weights_keys", {})
    for name in state_dict.keys() - mapping.keys():
        source = tied.get(name)
        if (source not in mapping
                or model.get_parameter(name) is not model.get_parameter(source)):
            raise ValueError(f"Untied model tensor missing from original base snapshot: {name}")
    if set(mapping.values()) != set(base_keys):
        raise ValueError("Exported checkpoint key set differs from original base snapshot")
    return mapping


def validate_checkpoint_layout(path, base_layout, non_text_keys):
    exported = checkpoint_layout(path)
    if exported.keys() != base_layout.keys():
        missing = sorted(base_layout.keys() - exported.keys())
        extra = sorted(exported.keys() - base_layout.keys())
        raise ValueError(f"Exported checkpoint key set differs from original base snapshot: "
                         f"missing={missing}, extra={extra}")
    for key in non_text_keys:
        if any(exported[key][field] != base_layout[key][field] for field in ("shape", "dtype")):
            raise ValueError(f"Non-text checkpoint shape/dtype differs from original base snapshot: {key}")
