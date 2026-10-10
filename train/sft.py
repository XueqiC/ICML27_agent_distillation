"""Offline Seed-SFT: official Gemma4Unified forward, PEFT LoRA, token-mean CE.

Reference pi1.py was read for lessons (native turn closure, token accounting,
complete turns). This independent loop uses no code or custom model forward
from that project.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.storage import digest, write_json, JsonlLedger, fsync_dir
from sed.seed_evidence import artifact_hashes, verified_collection_inputs
from train.dataset import load_tokenizer, build_verified_dataset, mask_report
from train.modeling import attach_lora, load_student, snapshot_metadata
from train.optimizer_state import (adamw, learning_rate_at, save_update_state,
                                   trainable_state, assert_trainable_equal, warmth_statistics)


def exposure_plan(costs, *, token_budget=512, seed=0, supervised_tokens=None,
                  passes=None, checkpoint_tokens=(), packing="turn", turn_counts=None):
    """Seeded exhaustive passes; thresholds round up to a complete example.

    Episode updates close once they reach the budget; episodes are never split.
    Legacy turn updates fit within the budget unless one turn exceeds it.
    Checkpoint thresholds flush on the first complete example reaching them.
    With both limits supplied the first reached wins. Every final state is saved.
    """
    if (not costs or any(type(c) is not int or c < 1 for c in costs)
            or type(token_budget) is not int or token_budget < 1):
        raise ValueError("Positive costs and supervised-token update budget required")
    if packing not in ("turn", "episode"):
        raise ValueError("Packing must be turn or episode")
    if turn_counts is None:
        turn_counts = [1] * len(costs)
    if len(turn_counts) != len(costs) or any(type(c) is not int or c < 1 for c in turn_counts):
        raise ValueError("Positive assistant turn counts required for every example")
    if supervised_tokens is None and passes is None:
        raise ValueError("Specify cumulative supervised tokens and/or passes")
    if ((supervised_tokens is not None and (type(supervised_tokens) is not int or supervised_tokens < 1))
            or (passes is not None and (type(passes) is not int or passes < 1))):
        raise ValueError("Training limits must be positive integers")
    endpoints = sorted(set(checkpoint_tokens))
    if any(type(v) is not int or v < 1 for v in endpoints):
        raise ValueError("Checkpoint token endpoints must be positive integers")
    if supervised_tokens and any(v > supervised_tokens for v in endpoints):
        raise ValueError("Checkpoint endpoint exceeds token limit")
    if passes and any(v > passes * sum(costs) for v in endpoints):
        raise ValueError("Checkpoint endpoint exceeds pass limit")
    rng = random.Random(seed)
    batches, pending, pending_tokens, total, turns, seen, pass_index = [], [], 0, 0, 0, 0, 0
    def flush(reached=(), final=False):
        nonlocal pending, pending_tokens
        if not pending:
            return
        batches.append({"indices": pending, "supervised_tokens": pending_tokens,
                        "cumulative_supervised_tokens": total, "turns_seen": turns,
                        "examples_seen": seen, "packing": packing,
                        "passes": seen / len(costs), "token_equivalent_passes": total / sum(costs),
                        "requested_endpoints": list(reached), "final": final})
        pending, pending_tokens = [], 0
    while True:
        order = list(range(len(costs)))
        rng.shuffle(order)
        for position, index in enumerate(order):
            cost = costs[index]
            if packing == "turn" and pending and pending_tokens + cost > token_budget:
                flush()
            pending.append(index)
            pending_tokens += cost
            total += cost
            turns += turn_counts[index]
            seen += 1
            reached = [v for v in endpoints if v <= total]
            endpoints = [v for v in endpoints if v > total]
            final = ((supervised_tokens is not None and total >= supervised_tokens)
                     or (passes is not None and pass_index + 1 >= passes and position == len(order) - 1))
            if reached or final or pending_tokens >= token_budget:
                flush(reached, final)
            if final:
                return batches
        pass_index += 1


def deterministic_seed(seed):
    # Must be set before the first CUDA context is initialized.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from transformers import set_seed
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def turn_loss_sum(model, row, device):
    """Native forward; only materialize logits that predict supervised tokens."""
    import torch
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor(row["labels"], dtype=torch.long, device=device)
    target_positions = torch.nonzero(labels != -100).flatten()
    if not len(target_positions) or int(target_positions[0]) < 1:
        raise ValueError("A target must have a causal predecessor")
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=target_positions - 1)
    return torch.nn.functional.cross_entropy(output.logits[0].float(), labels[target_positions], reduction="sum")


def token_mean_backward(model, examples, batch):
    """Accumulate the frozen batch exactly as Seed-SFT (caller clears grads)."""
    import torch
    device = next(model.parameters()).device
    total_loss, actual_tokens = 0.0, 0
    for index in batch["indices"]:
        row = examples[index]
        actual_tokens += sum(v != -100 for v in row["labels"])
        loss = turn_loss_sum(model, row, device)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite SFT loss")
        total_loss += float(loss.detach())
        (loss / batch["supervised_tokens"]).backward()
    if actual_tokens != batch["supervised_tokens"]:
        raise ValueError("Realized exposure differs from frozen update plan")
    return total_loss, actual_tokens


def warmup_exposure(examples, plan, steps):
    batches = [plan[i % len(plan)] for i in range(steps)]
    return {"steps": steps, "supervised_tokens": sum(b["supervised_tokens"] for b in batches),
            "examples": sum(len(b["indices"]) for b in batches),
            "forward_tokens": sum(len(examples[i]["input_ids"]) for b in batches for i in b["indices"]),
            "source": "extra gradient exposure on the same frozen seed data; no new data",
            "counted_in_supervised_tokens_seen": False, "real_update_data_cursor": 0}


def moment_warmup(model, optimizer, examples, plan, steps, *, gradient_clip=1.0, ledger=None,
                  log_warmth=False):
    """Advance m/v/step at exactly zero LR; real updates still start at plan[0]."""
    import torch
    before = trainable_state(model)
    rates = [group["lr"] for group in optimizer.param_groups]
    parameters = [p for p in model.parameters() if p.requires_grad]
    model.train()
    try:
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        for step in range(1, steps + 1):
            batch = plan[(step - 1) % len(plan)]
            optimizer.zero_grad(set_to_none=True)
            loss, tokens = token_mean_backward(model, examples, batch)
            norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
            warmth = {"warmth": warmth_statistics(optimizer)} if log_warmth else {}
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ledger is not None:
                ledger.append({"step": step, "plan_index": (step - 1) % len(plan),
                               "indices": batch["indices"], "supervised_tokens": tokens,
                               "loss": loss / tokens, "gradient_norm": float(norm),
                               "clip_active": bool(gradient_clip / (norm + 1e-6) < 1), "learning_rate": 0.0,
                               **warmth},
                              unique_key="step")
        assert_trainable_equal(model, before)
    finally:
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate
    return warmup_exposure(examples, plan, steps)


def checkpoint_metadata(batch, *, updates, wall_seconds, update_seconds, peak_memory,
                        peak_reserved, bank_tokens, identity_sha256):
    return {"schema_version": 1, "supervised_tokens": batch["cumulative_supervised_tokens"],
            "passes": batch["passes"], "token_equivalent_passes": batch["token_equivalent_passes"],
            "turns_seen": batch["turns_seen"], "updates": updates,
            "examples_seen": batch["examples_seen"], "packing": batch["packing"],
            "wall_time_seconds": wall_seconds, "optimizer_compute_seconds": update_seconds,
            "seconds_per_update": update_seconds / updates,
            "effective_supervised_tokens_per_second": batch["cumulative_supervised_tokens"] / max(wall_seconds, 1e-9),
            "peak_memory_bytes": peak_memory, "peak_reserved_memory_bytes": peak_reserved,
            "bank_supervised_tokens": bank_tokens, "identity_sha256": identity_sha256,
            "requested_token_endpoints": batch["requested_endpoints"],
            "endpoint_overshoot_tokens": {str(v): batch["cumulative_supervised_tokens"] - v
                                          for v in batch["requested_endpoints"]},
            "checkpoint_rule": f"first complete {batch['packing']} at/above threshold; final state always saved",
            "final": batch["final"]}


def train_loop(model, examples, plan, output, *, learning_rate=1e-5, tokenizer=None,
               identity_sha256, gradient_clip=1.0, moment_warmup_steps=0,
               lr_warmup_steps=0, save_every_update_through=0, log_warmth=False):
    import torch
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if any(type(n) is not int or n < 0 for n in
           (moment_warmup_steps, lr_warmup_steps, save_every_update_through)):
        raise ValueError("Warmup and state-save counts must be nonnegative integers")
    if any(e.get("label_provenance", {}).get("training_eligible") is False for e in examples):
        raise ValueError("Diagnostic examples cannot enter training")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = adamw(parameters, learning_rate)
    model.train()
    empty_visible_counts = [e.get("empty_visible_turns_masked", 0) for e in examples]
    start, compute_seconds, checkpoints = time.monotonic(), 0.0, []
    ledger = JsonlLedger(output / "updates.jsonl")
    warm = {}
    if moment_warmup_steps:
        warm = {"moment_warmup_exposure": moment_warmup(model, optimizer, examples, plan,
                    moment_warmup_steps, gradient_clip=gradient_clip,
                    ledger=JsonlLedger(output / "moment_warmup.jsonl"), log_warmth=log_warmth)}
    for update, batch in enumerate(plan, 1):
        if cuda:
            torch.cuda.synchronize(device)
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total_loss, actual_tokens = token_mean_backward(model, examples, batch)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
        warmth = {"warmth": warmth_statistics(optimizer)} if log_warmth else {}
        if lr_warmup_steps:
            optimizer.param_groups[0]["lr"] = learning_rate_at(learning_rate, lr_warmup_steps, update)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize(device)
        seconds = time.monotonic() - tick
        compute_seconds += seconds
        ledger.append({"update": update, **batch, "loss": total_loss / actual_tokens,
                       "seconds": seconds, "gradient_norm": float(grad_norm),
                       **warmth,
                       **({"learning_rate": optimizer.param_groups[0]["lr"]} if lr_warmup_steps else {})},
                      unique_key="update")
        if update <= save_every_update_through:
            save_update_state(model, optimizer, output, update=update, batch=batch, plan=plan,
                              learning_rate=learning_rate, lr_warmup_steps=lr_warmup_steps,
                              gradient_clip=gradient_clip, identity_sha256=identity_sha256,
                              moment_warmup_steps=moment_warmup_steps, extra=warm, tokenizer=tokenizer)
        if batch["requested_endpoints"] or batch["final"]:
            metadata = checkpoint_metadata(batch, updates=update, wall_seconds=time.monotonic() - start,
                update_seconds=compute_seconds,
                peak_memory=torch.cuda.max_memory_allocated(device) if cuda else None,
                peak_reserved=torch.cuda.max_memory_reserved(device) if cuda else None,
                bank_tokens=sum(e["supervised_tokens"] for e in examples), identity_sha256=identity_sha256)
            metadata.update(empty_visible_turns_masked=sum(empty_visible_counts),
                            empty_visible_turn_examples=sum(count > 0 for count in empty_visible_counts))
            metadata.update(warm)
            name = f"checkpoint-tokens-{batch['cumulative_supervised_tokens']}"
            staging = output / ("." + name + ".partial")
            staging.mkdir(exist_ok=False)
            model.save_pretrained(staging, safe_serialization=True)
            if tokenizer is not None:
                tokenizer.save_pretrained(staging)
            torch.save({"optimizer": optimizer.state_dict(), "updates": update,
                        "supervised_tokens": batch["cumulative_supervised_tokens"],
                        "torch_rng": torch.get_rng_state()}, staging / "optimizer.pt")
            metadata["artifact_sha256"] = artifact_hashes(staging)
            metadata["wall_time_seconds"] = time.monotonic() - start
            metadata["effective_supervised_tokens_per_second"] = (
                batch["cumulative_supervised_tokens"] / metadata["wall_time_seconds"])
            write_json(staging / "training_state.json", metadata, exclusive=True)
            os.rename(staging, output / name)
            fsync_dir(output)
            checkpoints.append({"path": name, **metadata})
            write_json(output / "profiling_report.json", {"device": str(device), "checkpoints": checkpoints,
                       "scope": "training loop including checkpoint saves; model/tokenizer load reported separately",
                       "cuda_memory_measured": cuda})
    return checkpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--verified", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--supervised-tokens", type=int)
    parser.add_argument("--passes", type=int)
    parser.add_argument("--checkpoint-tokens", type=int, nargs="*", default=[])
    parser.add_argument("--tokens-per-update", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--moment-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--save-every-update-through", type=int, default=0)
    parser.add_argument("--log-warmth", action="store_true")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--packing", choices=("turn", "episode"), default="episode")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.supervised_tokens is None and args.passes is None:
        parser.error("Specify --supervised-tokens and/or --passes")
    if args.rank < 1 or args.alpha < 1 or not 0 <= args.dropout < 1:
        parser.error("Invalid LoRA settings")
    if min(args.moment_warmup_steps, args.lr_warmup_steps, args.save_every_update_through) < 0:
        parser.error("Warmup and state-save counts must be nonnegative")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    preparation_start = time.monotonic()
    config = load_config(args.config)
    if (config["training"]["thinking_mode"] != "off"
            or config["training"]["student_step_token_cap"] != 1024):
        raise ValueError("Stage 2b requires thinking OFF and student step cap 1024")
    metadata = snapshot_metadata(args.snapshot)
    deterministic_seed(args.seed)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Production SFT requires a GPU; CPU plumbing is exercised only by unit tests")
    tokenizer = load_tokenizer(args.snapshot)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    examples = build_verified_dataset(args.verified, tokenizer, manifest, freeze,
                                      max_length=args.max_length, packing=args.packing)
    plan = exposure_plan([e["supervised_tokens"] for e in examples], token_budget=args.tokens_per_update,
                         seed=args.seed, supervised_tokens=args.supervised_tokens, passes=args.passes,
                         checkpoint_tokens=args.checkpoint_tokens, packing=args.packing,
                         turn_counts=[e["turn_count"] for e in examples])
    report = mask_report(examples, tokenizer)
    # Omit disabled additions even from provenance, preserving the legacy config.
    options = {k: v for k, v in vars(args).items() if v or k not in {
        "moment_warmup_steps", "lr_warmup_steps", "save_every_update_through", "log_warmth"}}
    config["sft"] = {**options, "base_snapshot": metadata, "dtype": "bfloat16",
                     "enable_thinking": False, "microbatch": 1, "gradient_checkpointing": True,
                     "loss": "supervised_token_mean_per_update", "attention": "eager",
                     "template_sha256": digest(tokenizer.chat_template), "plan_sha256": digest(plan),
                     "examples_sha256": digest(examples), "optimizer": "AdamW", "weight_decay": 0.0,
                     "betas": [0.9, 0.999], "eps": 1e-8, "gradient_clip": 1.0,
                     "scheduler": "constant", "deterministic_algorithms": True}
    if args.lr_warmup_steps:
        config["sft"]["scheduler"] = "linear_warmup_then_constant"
    if args.moment_warmup_steps:
        config["sft"]["moment_warmup_exposure"] = warmup_exposure(examples, plan, args.moment_warmup_steps)
    provenance = write_provenance(output, command="sft", config=config,
                      inputs=[args.config, *verified_collection_inputs(args.verified, manifest, freeze),
                              config["paths"]["manifest"],
                              config["paths"]["split_freeze"]])
    write_json(output / "mask_report.json", report, exclusive=True)
    write_json(output / "exposure_plan.json", plan, exclusive=True)
    write_json(output / "example_sources.json", [e["label_provenance"] for e in examples], exclusive=True)
    model = attach_lora(load_student(args.snapshot), rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    preparation = {"seconds": time.monotonic() - preparation_start, "packing": args.packing,
                   "examples_per_pass": len(examples),
                   "assistant_turns_per_pass": sum(e["turn_count"] for e in examples),
                   "supervised_tokens_per_pass": report["supervised_tokens_per_pass"],
                   "estimated_forward_tokens_per_pass": report["estimated_forward_tokens_per_pass"],
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(output / "preparation.json", preparation)
    print(json.dumps({"preparation": preparation}), flush=True)
    checkpoints = train_loop(model, examples, plan, output, learning_rate=args.learning_rate,
                             tokenizer=tokenizer, identity_sha256=digest(provenance),
                             moment_warmup_steps=args.moment_warmup_steps,
                             lr_warmup_steps=args.lr_warmup_steps,
                             save_every_update_through=args.save_every_update_through,
                             log_warmth=args.log_warmth)
    print(json.dumps({"checkpoints": [c["path"] for c in checkpoints],
                      "supervised_tokens": checkpoints[-1]["supervised_tokens"]}))


if __name__ == "__main__":
    main()
