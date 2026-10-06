"""Opt-in, evidence-bound Seed-SFT initialization for the dose controls."""
import argparse
import json
import math
import os
from pathlib import Path

from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.seed_evidence import artifact_hashes, load_verified_collection
from sed.storage import digest, file_hash, fsync_dir, write_json
from train.merge_lora import LEAP_MERGE_POLICY, validate_adapter_artifacts
from train.modeling import LORA_TARGETS

CONTROL = "dose_matched_init"
SUFFIX = " (Seed-SFT init)"


def read(path):
    return json.loads(Path(path).read_text())


def validate_init(adapter, snapshot, verified, config, *, collection_sha256):
    """Read train/sft.py's final state and its hash-bound recipe/input receipt.

    Recipe fields live in parent provenance (referenced by training_state), not
    directly in the legacy state. Compare content hashes across moved clusters.
    """
    from harness.lora_adapters import inspect_adapter
    adapter = Path(adapter).expanduser().resolve()
    state = validate_adapter_artifacts(adapter)
    provenance = read(adapter.parent / "provenance.json")
    student = config["models"]["student"]
    inspect_adapter("seed-sft-init", adapter, student, snapshot)
    if (provenance.get("command") != "sft" or state.get("final") is not True
            or not student.get("revision") or state.get("adaptation") is not None):
        raise ValueError("Init must be a final Seed-SFT adapter for the pinned student revision")
    recipe = provenance["config"]["sft"]
    expected = {"rank": 16, "alpha": 32, "dropout": 0.0, "learning_rate": 1e-5,
        "tokens_per_update": 512, "packing": "episode", "optimizer": "AdamW", "weight_decay": 0.0,
        "betas": [0.9, 0.999], "eps": 1e-8, "gradient_clip": 1.0, "scheduler": "constant",
        "dtype": "bfloat16", "loss": "supervised_token_mean_per_update", "attention": "eager",
        "microbatch": 1, "gradient_checkpointing": True, "enable_thinking": False}
    if any(recipe.get(k) != v for k, v in expected.items()) or state.get("packing") != "episode":
        raise ValueError("Init does not use the Seed-SFT dose-control recipe")
    passes = state.get("passes")
    if type(passes) not in (float, int) or not math.isfinite(passes) or passes <= 0:
        raise ValueError("Init must record positive realized passes")
    peft = read(adapter / "adapter_config.json")
    targets = peft.get("target_modules", [])
    if (peft["r"] != 16 or peft.get("lora_alpha") != 32 or peft.get("lora_dropout") != 0
            or not isinstance(targets, list)
            or {t.rsplit(".", 1)[-1] for t in targets} != set(LORA_TARGETS)
            or any(not t.startswith("model.language_model.layers.") for t in targets)):
        raise ValueError("Init adapter differs from Seed-SFT text LoRA r16/a32")
    # The original collection path can be unavailable after a cluster transfer.
    # Its exact manifest file hash is in the state-bound SFT provenance inputs.
    source = str(Path(recipe["verified"]) / "manifest.json")
    if provenance["inputs"].get(source) != file_hash(Path(verified) / "manifest.json"):
        raise ValueError("Init was trained on a different verified collection (collection sha)")
    collection = read(Path(verified) / "manifest.json")
    if state.get("examples_seen", 0) <= 0 or passes != state["examples_seen"] / collection["K"]:
        raise ValueError("Init passes differ from episode exposure on the verified collection")
    if digest(collection) != collection_sha256:
        raise ValueError("Run collection sha changed")
    return {"path": str(adapter), "artifact_sha256": artifact_hashes(adapter),
        "training_state": state, "training_provenance_sha256": file_hash(adapter.parent / "provenance.json"),
        "recipe": recipe, "collection_sha256": collection_sha256,
        "student": student, "passes": passes}


def bind_init(args, config, identity):
    if not getattr(args, "init_checkpoint", None):
        return  # No added default keys, including nulls, in the frozen identity.
    init = validate_init(args.init_checkpoint, args.snapshot, args.verified, config,
                         collection_sha256=identity["collection_sha256"])
    init["merge_policy"] = LEAP_MERGE_POLICY if identity.get("method") == "leap" else "seed_sft_fp32"
    identity.update(control=CONTROL, init=init)


def validate_evaluation_init(args, config):
    """An explicitly supplied init must also match the run being evaluated."""
    if not getattr(args, "init_checkpoint", None):
        return
    identity = read(Path(args.output) / "identity.json")
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, _ = load_verified_collection(args.verified, manifest, freeze)
    requested = {"collection_sha256": digest(collection), "method": identity.get("method")}
    bind_init(args, config, requested)
    if identity.get("init") != requested["init"]:
        raise ValueError("Evaluation init differs from the frozen pipeline identity")


def control_fields(identity):
    return {k: identity[k] for k in ("control", "init") if k in identity}


def model_record(identity, name, checkpoint, *, label=None, **extra):
    row = {"name": name, "checkpoint": str(checkpoint), **extra}
    if label is not None:
        row["label"] = label
    if identity.get("control") == CONTROL:
        row["label"] = (label or name) + SUFFIX
    return row


def materialize_init(args, identity):
    from baselines.eto.pipeline import run_command, stage_run
    command = [args.train_python, "-m", "baselines.common.dose_control", "--config", args.config,
        "--snapshot", args.snapshot, "--verified", args.verified,
        "--identity", str(Path(args.output) / "identity.json")]
    merged = stage_run(args.output, "init", digest([digest(identity), "init"]),
        lambda a: run_command(command + ["--output", str(a / "output")], a / "merge.log"))
    return merged / "checkpoint"


def export_init(snapshot, adapter, output, config, init, *, variant=None, device="cpu"):
    """Full-FT starts at exact fp32 base+delta; LEAP uses its bf16 policy."""
    from baselines.common.fsdp_train import export_checkpoint, validate_checkpoint
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = write_provenance(output, command="dose_matched_init", config=config,
        inputs=[Path(adapter) / "training_state.json", Path(adapter).parent / "provenance.json"],
        details={"control": CONTROL, "init": init, "base_artifact_sha256": artifact_hashes(snapshot)})
    staging = output / ".checkpoint.partial"
    state = {"final": True, "control": CONTROL, "init": init, "merge_policy": init["merge_policy"],
        "provenance_sha256": digest(provenance), "examples_seen": 0, "supervised_tokens_seen": 0,
        "training_exposure": "Reused checkpoint; original exposure recorded under init.training_state"}
    if variant is None:
        from peft import PeftModel
        from train.modeling import load_student
        # CPU fp32 avoids both low-precision base addition and GPU TF32 matmul.
        model = PeftModel.from_pretrained(load_student(snapshot, device="cpu", dtype="float32"),
            adapter, local_files_only=True, is_trainable=False)
        merged = model.merge_and_unload(safe_merge=True)
        state.update(export_checkpoint(merged, staging, snapshot, text_dtype="float32"))
    else:
        from sed.react import build_messages
        from train.dataset import load_tokenizer
        from train.merge_lora import merge_adapter
        tokenizer = load_tokenizer(snapshot)
        ids = tokenizer.apply_chat_template(build_messages([], {}, {"observation": "You are in a room.",
            "admissible_commands": ["look", "inventory"]}), tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        report = merge_adapter(snapshot, adapter, staging, list(ids), device=device, dtype="bfloat16",
            tokenizer=tokenizer, config=config, policy=LEAP_MERGE_POLICY,
            provenance_inputs=[Path(adapter) / "training_state.json", Path(adapter).parent / "provenance.json"])
        state.update(adaptation="leap_merged_lora", checkpoint_kind="merged_lora", leap_iteration=0,
            leap_variant=variant, privileged_teacher=variant == "privileged", fresh_lora=False,
            text_export_dtype="bfloat16", non_text_source=str(Path(snapshot).resolve()),
            policy=report["policy"], merge_passed=report["passed"],
            strict_allclose_passed=report["strict_allclose_passed"], artifact_sha256=artifact_hashes(staging))
    write_json(staging / "training_state.json", state)
    os.rename(staging, output / "checkpoint")
    fsync_dir(output)
    validate_checkpoint(output / "checkpoint", config["models"]["student"], base_snapshot=snapshot)
    write_json(output / "result.json", state)
    return output / "checkpoint"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "snapshot", "verified", "identity", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    config, identity = load_config(args.config), read(args.identity)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, _ = load_verified_collection(args.verified, manifest, freeze)
    init = validate_init(identity["init"]["path"], args.snapshot, args.verified, config,
                         collection_sha256=digest(collection))
    variant = identity["recipe"]["variant"] if identity.get("method") == "leap" else None
    init["merge_policy"] = LEAP_MERGE_POLICY if variant else "seed_sft_fp32"
    if init != identity["init"] or artifact_hashes(args.snapshot) != identity["snapshot_artifact_sha256"]:
        raise ValueError("Init/base changed after pipeline preparation")
    export_init(args.snapshot, init["path"], args.output, config, init, variant=variant, device="cuda")


if __name__ == "__main__":
    main()
