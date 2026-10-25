"""Official HF/PEFT loading shared by SFT and merging; local files only."""
import json
from pathlib import Path
import shutil

from sed.storage import file_hash

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def text_decoder_name(model_type):
    """Provenance lookup before model allocation or distributed wrapping."""
    names = {"gemma4_unified": "Gemma4UnifiedTextDecoderLayer", "gemma4": "Gemma4TextDecoderLayer"}
    if model_type not in names:
        raise ValueError("Unsupported Gemma text decoder family")
    return names[model_type]


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
    if config.get("model_type") not in {"gemma4_unified", "gemma4"}:
        raise ValueError("Expected official Gemma4Unified or Gemma4 checkpoint; no model substitution")
    return {"path": str(path), "metadata_sha256": {
                p.name: file_hash(p) for p in sorted(path.iterdir())
                if p.is_file() and p.suffix in {".json", ".jinja", ".jinja2"}},
            "weight_verification": "local snapshot identity and metadata; base weight bytes not certified"}


def load_student(path, *, device="cuda", dtype="bfloat16"):
    import torch
    from transformers import Gemma4UnifiedForConditionalGeneration
    snapshot_metadata(path)
    model_class = Gemma4UnifiedForConditionalGeneration
    if json.loads((Path(path) / "config.json").read_text())["model_type"] == "gemma4":
        from transformers import Gemma4ForConditionalGeneration
        model_class = Gemma4ForConditionalGeneration
    return model_class.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, dtype=getattr(torch, dtype),
        attn_implementation="eager", device_map={"": device})


def text_decoder_class(model_type):
    if model_type == "gemma4_unified":
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextDecoderLayer
        return Gemma4UnifiedTextDecoderLayer
    if model_type == "gemma4":
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        return Gemma4TextDecoderLayer
    raise ValueError("Unsupported Gemma text decoder family")


def text_decoder_layers(model):
    """Select only text blocks for FSDP; multimodal towers are never blocks."""
    cls = text_decoder_class(model.config.model_type)
    layers = [module for module in model.modules() if isinstance(module, cls)]
    if len(layers) != model.config.text_config.num_hidden_layers:
        raise ValueError(f"{cls.__name__} wrap count differs from config")
    return layers


def attach_lora(model, *, rank=16, alpha=32, dropout=0.0, gradient_checkpointing=True,
                targets=None):
    from peft import LoraConfig, get_peft_model
    # Restrict to text attention/MLP projections; the multimodal towers are frozen.
    targets = LORA_TARGETS if targets is None else targets
    if not targets or not set(targets).issubset(LORA_TARGETS):
        raise ValueError("Unsupported text LoRA targets")
    names = [name for name, _ in model.named_modules()
             if ".language_model." in name and name.rsplit(".", 1)[-1] in targets]
    if model.config.model_type == "gemma4":
        # Per-layer gates/projections are not part of the existing LoRA recipe.
        import re
        pattern = r"model\.language_model\.layers\.\d+\.(?:self_attn|mlp)\.(?:" + "|".join(targets) + ")"
        names = [name for name in names if re.fullmatch(pattern, name)]
    if {name.rsplit(".", 1)[-1] for name in names} != set(targets):
        raise ValueError("Missing expected attention/MLP LoRA projections")
    adapted = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                    target_modules=names, bias="none", task_type="CAUSAL_LM"))
    if model.config.model_type == "gemma4":
        # PEFT can minimize long lists to suffixes. Preserve the actual E-series
        # text coverage for provenance, reloads and explicit serving conversion.
        adapted.peft_config["default"].target_modules = set(names)
    adapted.config.get_text_config().use_cache = False
    if gradient_checkpointing:
        adapted.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        adapted.enable_input_require_grads()
    return adapted
