"""Checked LoRA merge, published in the existing standalone-checkpoint schema."""
import argparse
import os
from pathlib import Path

from baselines.leap.teacher import read
from sed.config import load_config
from sed.provenance import write_provenance
from sed.react import build_messages
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, fsync_dir, write_json
from train.dataset import load_tokenizer
from train.merge_lora import merge_adapter, validate_adapter_artifacts
from train.modeling import snapshot_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    for name in ("snapshot", "adapter", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    adapter = Path(args.adapter)
    state = validate_adapter_artifacts(adapter)
    training = read(adapter.parent / "provenance.json")
    if (state["identity_sha256"] != digest(training) or state.get("adaptation") != "leap_lora"
            or training["config"]["sft"]["base_snapshot"] != snapshot_metadata(args.snapshot)
            or training["config"]["models"]["student"] != config["models"]["student"]
            or training["config"]["leap_training"]["snapshot_artifact_sha256"] != artifact_hashes(args.snapshot)):
        raise ValueError("LEAP merge base/adapter provenance changed")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = write_provenance(output, command="leap_checked_merge", config=config,
        inputs=[args.config, adapter / "training_state.json", adapter.parent / "provenance.json"],
        details={"leap_iteration": state["leap_iteration"], "adaptation": "leap_merged_lora",
                 "leap_variant": state["leap_variant"], "privileged_teacher": state["privileged_teacher"],
                 "base_artifact_sha256": artifact_hashes(args.snapshot)})
    tokenizer = load_tokenizer(args.snapshot)
    ids = tokenizer.apply_chat_template(build_messages([], {}, {"observation": "You are in a room.",
        "admissible_commands": ["look", "inventory"]}), tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    staging = output / ".checkpoint.partial"
    report = merge_adapter(args.snapshot, args.adapter, staging, list(ids), tokenizer=tokenizer, config=config,
        provenance_inputs=[adapter / "training_state.json", adapter.parent / "provenance.json"])
    state.update(adaptation="leap_merged_lora", checkpoint_kind="merged_lora", text_export_dtype="bfloat16",
        provenance_sha256=digest(provenance), artifact_sha256=artifact_hashes(staging), merge_passed=report["passed"])
    write_json(staging / "training_state.json", state)
    os.rename(staging, output / "checkpoint")
    fsync_dir(output)
    write_json(output / "result.json", state)


if __name__ == "__main__":
    main()
