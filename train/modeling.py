"""Official HF/PEFT loading shared by SFT and merging; local files only."""
import json
from pathlib import Path
import shutil

from sed.storage import file_hash

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def copy_base_assets(base, destination, *, overwrite=False, exclude=()):
    """Preserve tokenizer/template/processor files without copying model configs.

    Full-FT export opts into its original overwrite behavior; merged exports
    keep files already written by the model and tokenizer.
    """
    destination = Path(destination)
    copied = []
    for path in Path(base).iterdir():
        if path.is_file() and path.suffix in {".json", ".jinja", ".jinja2", ".model", ".txt"} and not (
                path.name in {"config.json", "generation_config.json"} or ".index." in path.name):
            target = destination / path.name
            if path.name in exclude or (not overwrite and target.exists()):
                continue
            shutil.copyfile(path, target)
            copied.append(path.name)
    return sorted(copied)


def snapshot_metadata(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError("Student snapshot must be a local directory")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "gemma4_unified":
        raise ValueError("Expected official Gemma4Unified checkpoint; no model substitution")
    return {"path": str(path), "metadata_sha256": {
                p.name: file_hash(p) for p in sorted(path.iterdir())
                if p.is_file() and p.suffix in {".json", ".jinja", ".jinja2"}},
            "weight_verification": "local snapshot identity and metadata; base weight bytes not certified"}


def load_student(path, *, device="cuda", dtype="bfloat16"):
    import torch
    from transformers import Gemma4UnifiedForConditionalGeneration
    snapshot_metadata(path)
    return Gemma4UnifiedForConditionalGeneration.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, dtype=getattr(torch, dtype),
        attn_implementation="eager", device_map={"": device})


def attach_lora(model, *, rank=16, alpha=32, dropout=0.0, gradient_checkpointing=True):
    from peft import LoraConfig, get_peft_model
    # Restrict to text attention/MLP projections; the multimodal towers are frozen.
    names = [name for name, _ in model.named_modules()
             if ".language_model." in name and name.rsplit(".", 1)[-1] in LORA_TARGETS]
    if {name.rsplit(".", 1)[-1] for name in names} != set(LORA_TARGETS):
        raise ValueError("Missing expected attention/MLP LoRA projections")
    adapted = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                    target_modules=names, bias="none", task_type="CAUSAL_LM"))
    adapted.config.get_text_config().use_cache = False
    if gradient_checkpointing:
        adapted.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        adapted.enable_input_require_grads()
    return adapted
