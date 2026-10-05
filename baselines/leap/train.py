"""One-GPU LEAP SFT: fresh text q/v LoRA over the previous MERGED model."""
import argparse
from copy import deepcopy
from functools import partial
import os
from pathlib import Path
import time

from baselines.common.fsdp_train import batch_logps, validate_checkpoint
from baselines.common.schedule import dose, epoch_batches, lr_multiplier
from baselines.leap.data import encode_rows, load_set, min_subsample
from baselines.leap.protocol import LABELS
from baselines.leap.teacher import read
from sed.config import load_config
from sed.provenance import write_provenance
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, fsync_dir, JsonlLedger, write_json
from train.dataset import load_tokenizer
from train.modeling import load_student, snapshot_metadata

RECIPE = {"training_seed": 0, "iterations": [0, 1, 2, 3], "epochs": 1,
    "max_length": 32768, "rank": 128, "alpha": 64, "dropout": .05,
    "targets": ["q_proj", "v_proj"], "learning_rate": 3e-5, "scheduler": "cosine",
    "warmup_steps": 10, "max_grad_norm": .3, "original_batch": 32, "microbatch": 4,
    "optimizer": "adamw_torch_fused", "weight_decay": 0., "betas": [.9, .999], "eps": 1e-8,
    "weights": "bfloat16", "attention": "sdpa", "fresh_lora": True,
    "packing": "turn", "train_split": 1., "eval_save_steps": 200,
    "checkpoint_selection": False, "annotate_reason": "skipped; seed Thought is REASON"}


def attach_fresh_lora(model, *, gradient_checkpointing=True):
    from peft import LoraConfig, get_peft_model
    if getattr(model, "peft_config", None) or any("lora_" in n for n, _ in model.named_parameters()):
        raise ValueError("LEAP requires a merged model before attaching a fresh LoRA")
    names = [name for name, _ in model.named_modules() if ".language_model." in name
             and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}]
    if {n.rsplit(".", 1)[-1] for n in names} != {"q_proj", "v_proj"}:
        raise ValueError("Missing text q/v projections")
    adapted = get_peft_model(model, LoraConfig(r=128, lora_alpha=64, lora_dropout=.05,
        target_modules=names, bias="none", task_type="CAUSAL_LM"))
    adapted.set_attn_implementation("sdpa")
    adapted.config.get_text_config().use_cache = False
    if gradient_checkpointing:
        adapted.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        adapted.enable_input_require_grads()
    return adapted


def prepare_model(snapshot, *, device="cuda"):
    if (Path(snapshot) / "adapter_config.json").exists():
        raise ValueError("Cannot continue the previous adapter; merge it first")
    return attach_fresh_lora(load_student(snapshot, device=device,
        dtype="bfloat16" if str(device).startswith("cuda") else "float32"))


def validate_initialization(snapshot, base_snapshot, *, iteration, variant, student):
    if iteration == 0:
        if Path(snapshot).resolve() != Path(base_snapshot).resolve():
            raise ValueError("Iteration 0 must start from configured base")
        return
    state = validate_checkpoint(snapshot, student)
    if (state.get("leap_iteration") != iteration - 1 or not state.get("fresh_lora")
            or state.get("adaptation") != "leap_merged_lora" or state.get("leap_variant") != variant):
        raise ValueError("LEAP must initialize from the previous merged iterate of the same variant")


def train_loop(model, examples, output, *, device="cpu"):
    """Completion CE per physical microbatch, mean over accumulation slots.

    Retain official microbatch four (or the smaller X5 batch). Partial global
    batches normalize by actual slots; no duplicate examples or truncation.
    CPU uses unfused AdamW solely for offline tests of the same arithmetic.
    """
    import torch
    plan = dose(len(examples), 32, epochs=1)
    if plan["total_steps"] >= 200:
        raise ValueError("Dose reaches official eval/save; registered no-selection regime no longer applies")
    cuda = str(device).startswith("cuda")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-5, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., fused=cuda)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_multiplier,
        scheduler="cosine", total_steps=plan["total_steps"], warmup_ratio=0, warmup_steps=10))
    ledger = JsonlLedger(Path(output) / "updates.jsonl")
    started, exposure, tokens, forward = time.monotonic(), 0, 0, 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for update, batch in enumerate(epoch_batches(len(examples), plan["b_prime"], epochs=1, seed=0), 1):
        optimizer.zero_grad(set_to_none=True)
        indices = batch["indices"]
        chunks = [indices[i:i + 4] for i in range(0, len(indices), 4)]
        mean_loss = 0.
        for chunk in chunks:
            rows = [examples[index] for index in chunk]
            supervised = sum(sum(v != -100 for v in r["labels"]) for r in rows)
            loss = -batch_logps(model, rows, device).sum() / supervised
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite LEAP loss")
            (loss / len(chunks)).backward()
            mean_loss += float(loss.detach()) / len(chunks)
        norm = torch.nn.utils.clip_grad_norm_(parameters, .3, error_if_nonfinite=True)
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        schedule.step()
        exposure += len(indices)
        tokens += sum(sum(v != -100 for v in examples[i]["labels"]) for i in indices)
        forward += sum(len(examples[i]["input_ids"]) for i in indices)
        ledger.append({"update": update, **batch, "lr": lr, "loss": mean_loss,
            "global_batch": len(indices), "microbatches": list(map(len, chunks)), "gradient_norm": float(norm),
            "examples_seen": exposure, "supervised_tokens_seen": tokens,
            "logical_forward_tokens": forward}, unique_key="update")
    return {**plan, "updates": update, "examples_seen": exposure, "supervised_tokens_seen": tokens,
        "logical_forward_tokens": forward, "wall_time_seconds": time.monotonic() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "optimizer_fused": cuda, "gradient_clip": .3, "warmup_steps": 10,
        "loss_reduction": "completion_token_mean_per_microbatch_then_mean_over_actual_accumulation_slots"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    for name in ("snapshot", "base-snapshot", "verified", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--iteration", type=int, choices=range(4), required=True)
    parser.add_argument("--variant", choices=list(LABELS), required=True)
    parser.add_argument("--sets", nargs="+", required=True)
    args = parser.parse_args(argv)
    if len(args.sets) != args.iteration + 1:
        parser.error("Supply exactly the example sets iter0..i")
    config = load_config(args.config)
    if (config["training"]["thinking_mode"] != "off" or config["training"]["student_step_token_cap"] != 1024
            or config["environment"]["max_episode_steps"] != 40):
        raise ValueError("LEAP harness protocol differs")
    validate_initialization(args.snapshot, args.base_snapshot, iteration=args.iteration,
        variant=args.variant, student=config["models"]["student"])
    for path in args.sets[1:]:
        corrections = read(Path(path).parent / "provenance.json")["details"]["corrections"]
        if read(Path(corrections) / "provenance.json")["details"]["variant"] != args.variant:
            raise ValueError("Correction dataset belongs to a different LEAP variant")
    sets = [load_set(path, config, expected_iteration=i, verified=args.verified) for i, path in enumerate(args.sets)]
    rows, aggregation = min_subsample(sets, seed=0)
    if not rows:
        raise ValueError("Official min aggregation is empty; no training")
    tokenizer = load_tokenizer(args.snapshot)
    examples = encode_rows(rows, tokenizer)
    plan = dose(len(examples), 32, epochs=1)
    if plan["total_steps"] >= 200:
        raise ValueError("Registered dose cannot trigger eval/save step 200")
    import torch
    from transformers import set_seed
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Production LEAP training requires one allocated GPU")
    set_seed(0)
    model = prepare_model(args.snapshot)
    model.peft_config["default"].revision = config["models"]["student"]["revision"]
    effective = deepcopy(config)
    effective["sft"] = {"base_snapshot": snapshot_metadata(args.snapshot)}
    effective["leap_training"] = {**vars(args), **RECIPE, **plan, "aggregation": aggregation,
        "label": LABELS[args.variant], "privileged_teacher": args.variant == "privileged",
        "snapshot_artifact_sha256": artifact_hashes(args.snapshot), "examples_sha256": digest(examples),
        "template_sha256": digest(tokenizer.chat_template), "no_truncation": True,
        "adapter_master_dtype": "float32", "frozen_base_dtype": "bfloat16"}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = write_provenance(output, command="leap_sft", config=effective,
        inputs=[args.config, *args.sets, Path(args.verified) / "manifest.json",
                config["paths"]["manifest"], config["paths"]["split_freeze"]])
    write_json(output / "aggregation.json", aggregation)
    write_json(output / "example_sources.json", [e["label_provenance"] for e in examples])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = train_loop(model, examples, output, device="cuda")
    staging = output / ".adapter.partial"
    model.save_pretrained(staging, safe_serialization=True)
    state.update(identity_sha256=digest(provenance), final=True, adaptation="leap_lora",
        leap_variant=args.variant, privileged_teacher=args.variant == "privileged",
        leap_iteration=args.iteration, fresh_lora=True, packing="turn", passes=1,
        supervised_tokens=state["supervised_tokens_seen"], artifact_sha256=artifact_hashes(staging))
    write_json(staging / "training_state.json", state)
    os.rename(staging, output / "adapter")
    fsync_dir(output)
    write_json(output / "result.json", state)


if __name__ == "__main__":
    main()
