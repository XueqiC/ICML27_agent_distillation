"""Merge, save/reload, and compare fixed-input next-token logits before serving."""
import argparse
import gc
import json
from pathlib import Path
import time

from sed.config import load_config
from sed.provenance import write_provenance
from sed.react import build_messages
from sed.storage import digest, write_json
from sed.seed_evidence import artifact_hashes
from train.dataset import load_tokenizer
from train.modeling import load_student, snapshot_metadata


def validate_adapter_artifacts(adapter):
    adapter = Path(adapter)
    state = json.loads((adapter / "training_state.json").read_text())
    hashes = artifact_hashes(adapter)
    hashes.pop("training_state.json")
    if hashes != state["artifact_sha256"]:
        raise ValueError("Adapter checkpoint artifacts changed after training")
    return state


def next_token_logits(model, input_ids):
    import torch
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                     logits_to_keep=1).logits[0, -1].float().cpu()


def merge_adapter(base, adapter, output, input_ids, *, device="cuda", dtype="bfloat16",
                  atol=0.125, rtol=0.01, tokenizer=None, config=None, provenance_inputs=()):
    """Compare base+adapter against both merged and reloaded standalone weights.

    bf16 uses atol=.125, rtol=.01 (rounding from W + BA); CPU float32 tests use
    atol=1e-5, rtol=1e-5. All vocabulary logits must pass, not only argmax.
    No simultaneous second full model is needed during the reload check.
    """
    import torch
    from peft import PeftModel
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_provenance(output, command="merge_lora", config={**(config or {}), "merge": {
        "snapshot": str(base), "adapter": str(adapter), "output": str(output),
        "device": device, "dtype": dtype, "atol": atol, "rtol": rtol}},
        inputs=provenance_inputs,
        details={"base_snapshot": snapshot_metadata(base), "fixed_input_sha256": digest(input_ids)})
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = PeftModel.from_pretrained(load_student(base, device=device, dtype=dtype), adapter,
                                      local_files_only=True, is_trainable=False)
    reference = next_token_logits(model, input_ids)
    merged = model.merge_and_unload(safe_merge=True)
    in_memory = next_token_logits(merged, input_ids)
    merged.save_pretrained(output, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(output)
    del model, merged
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    reloaded = load_student(output, device=device, dtype=dtype)
    standalone = next_token_logits(reloaded, input_ids)
    checks = {}
    for name, logits in (("merged_in_memory", in_memory), ("merged_reloaded", standalone)):
        checks[name] = {"passed": bool(torch.isfinite(reference).all() and torch.isfinite(logits).all()
                                       and torch.allclose(reference, logits, atol=atol, rtol=rtol)),
                        "max_absolute_error": float((reference - logits).abs().max()),
                        "mean_absolute_error": float((reference - logits).abs().mean()),
                        "argmax_equal": int(reference.argmax()) == int(logits.argmax())}
    report = {"passed": all(v["passed"] for v in checks.values()), "checks": checks,
              "input_ids": input_ids, "input_sha256": digest(input_ids), "dtype": dtype,
              "atol": atol, "rtol": rtol, "comparison": "all next-token vocabulary logits",
              "wall_time_seconds": time.monotonic() - started,
              "base": snapshot_metadata(base), "adapter_sha256": artifact_hashes(adapter),
              "merged_sha256": artifact_hashes(output), "enable_thinking": False}
    report["wall_time_seconds"] = time.monotonic() - started
    report["peak_memory_bytes"] = (torch.cuda.max_memory_allocated()
                                   if device.startswith("cuda") else None)
    write_json(output / "consistency.json", report, exclusive=True)
    if not report["passed"]:
        raise ValueError("Merged logits exceed tolerance; checkpoint must not be served")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=0.125)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args(argv)
    import math
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        parser.error("Tolerances must be finite and nonnegative")
    config = load_config(args.config)
    adapter = Path(args.adapter)
    state = validate_adapter_artifacts(adapter)
    training_provenance = json.loads((adapter.parent / "provenance.json").read_text())
    if state["identity_sha256"] != digest(training_provenance):
        raise ValueError("Checkpoint provenance mismatch")
    if training_provenance["config"]["sft"]["base_snapshot"] != snapshot_metadata(args.snapshot):
        raise ValueError("Merge base differs from recorded training snapshot")
    tokenizer = load_tokenizer(args.snapshot)
    messages = build_messages([], {}, {"observation": "You are in a room. Look around.",
                                      "admissible_commands": ["look", "inventory"]})
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False)
    # Transformers 5 may return a BatchEncoding by default.
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    report = merge_adapter(args.snapshot, adapter, args.output, list(ids), tokenizer=tokenizer,
                           atol=args.atol, rtol=args.rtol, config=config,
                           provenance_inputs=[args.config, adapter / "training_state.json",
                                              adapter.parent / "provenance.json"])
    print(json.dumps({"output": args.output, "passed": report["passed"]}))


if __name__ == "__main__":
    main()
