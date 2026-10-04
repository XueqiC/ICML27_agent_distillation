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
from sed.seed_evidence import artifact_hashes
from train.dataset import load_tokenizer, build_verified_dataset, mask_report
from train.modeling import attach_lora, load_student, snapshot_metadata


def exposure_plan(costs, *, token_budget=512, seed=0, supervised_tokens=None,
                  passes=None, checkpoint_tokens=()):
    """Seeded exhaustive passes; ceilings rounded up to the next complete turn.

    Updates fit within the budget unless a single turn exceeds it. Checkpoint
    thresholds flush the pending update on the first complete turn reaching them.
    With both limits supplied the first reached wins. Every final state is saved.
    """
    if (not costs or any(type(c) is not int or c < 1 for c in costs)
            or type(token_budget) is not int or token_budget < 1):
        raise ValueError("Positive costs and supervised-token update budget required")
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
    batches, pending, pending_tokens, total, turns, pass_index = [], [], 0, 0, 0, 0
    def flush(reached=(), final=False):
        nonlocal pending, pending_tokens
        if not pending:
            return
        batches.append({"indices": pending, "supervised_tokens": pending_tokens,
                        "cumulative_supervised_tokens": total, "turns_seen": turns,
                        "passes": turns / len(costs), "token_equivalent_passes": total / sum(costs),
                        "requested_endpoints": list(reached), "final": final})
        pending, pending_tokens = [], 0
    while True:
        order = list(range(len(costs)))
        rng.shuffle(order)
        for position, index in enumerate(order):
            cost = costs[index]
            if pending and pending_tokens + cost > token_budget:
                flush()
            pending.append(index)
            pending_tokens += cost
            total += cost
            turns += 1
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


def checkpoint_metadata(batch, *, updates, wall_seconds, update_seconds, peak_memory,
                        peak_reserved, bank_tokens, identity_sha256):
    return {"schema_version": 1, "supervised_tokens": batch["cumulative_supervised_tokens"],
            "passes": batch["passes"], "token_equivalent_passes": batch["token_equivalent_passes"],
            "turns_seen": batch["turns_seen"], "updates": updates,
            "wall_time_seconds": wall_seconds, "optimizer_compute_seconds": update_seconds,
            "seconds_per_update": update_seconds / updates,
            "effective_supervised_tokens_per_second": batch["cumulative_supervised_tokens"] / max(wall_seconds, 1e-9),
            "peak_memory_bytes": peak_memory, "peak_reserved_memory_bytes": peak_reserved,
            "bank_supervised_tokens": bank_tokens, "identity_sha256": identity_sha256,
            "requested_token_endpoints": batch["requested_endpoints"],
            "endpoint_overshoot_tokens": {str(v): batch["cumulative_supervised_tokens"] - v
                                          for v in batch["requested_endpoints"]},
            "checkpoint_rule": "first complete turn at/above threshold; final state always saved",
            "final": batch["final"]}


def train_loop(model, examples, plan, output, *, learning_rate=1e-5, tokenizer=None,
               identity_sha256, gradient_clip=1.0):
    import torch
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if any(e.get("label_provenance", {}).get("training_eligible") is False for e in examples):
        raise ValueError("Diagnostic examples cannot enter training")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0,
                                 betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)
    model.train()
    start, compute_seconds, checkpoints = time.monotonic(), 0.0, []
    ledger = JsonlLedger(output / "updates.jsonl")
    for update, batch in enumerate(plan, 1):
        if cuda:
            torch.cuda.synchronize(device)
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        actual_tokens = 0
        for index in batch["indices"]:  # microbatch = one complete turn, full history
            row = examples[index]
            cost = sum(v != -100 for v in row["labels"])
            actual_tokens += cost
            loss = turn_loss_sum(model, row, device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite SFT loss")
            total_loss += float(loss.detach())
            # Weight by actual supervised tokens, never by number of turns.
            (loss / batch["supervised_tokens"]).backward()
        if actual_tokens != batch["supervised_tokens"]:
            raise ValueError("Realized exposure differs from frozen update plan")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize(device)
        seconds = time.monotonic() - tick
        compute_seconds += seconds
        ledger.append({"update": update, **batch, "loss": total_loss / actual_tokens,
                       "seconds": seconds, "gradient_norm": float(grad_norm)}, unique_key="update")
        if batch["requested_endpoints"] or batch["final"]:
            metadata = checkpoint_metadata(batch, updates=update, wall_seconds=time.monotonic() - start,
                update_seconds=compute_seconds,
                peak_memory=torch.cuda.max_memory_allocated(device) if cuda else None,
                peak_reserved=torch.cuda.max_memory_reserved(device) if cuda else None,
                bank_tokens=sum(e["supervised_tokens"] for e in examples), identity_sha256=identity_sha256)
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
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.supervised_tokens is None and args.passes is None:
        parser.error("Specify --supervised-tokens and/or --passes")
    if args.rank < 1 or args.alpha < 1 or not 0 <= args.dropout < 1:
        parser.error("Invalid LoRA settings")
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
    examples = build_verified_dataset(args.verified, tokenizer, manifest, freeze, max_length=args.max_length)
    plan = exposure_plan([e["supervised_tokens"] for e in examples], token_budget=args.tokens_per_update,
                         seed=args.seed, supervised_tokens=args.supervised_tokens, passes=args.passes,
                         checkpoint_tokens=args.checkpoint_tokens)
    report = mask_report(examples, tokenizer)
    config["sft"] = {**vars(args), "base_snapshot": metadata, "dtype": "bfloat16",
                     "enable_thinking": False, "microbatch": 1, "gradient_checkpointing": True,
                     "loss": "supervised_token_mean_per_update", "attention": "eager",
                     "template_sha256": digest(tokenizer.chat_template), "plan_sha256": digest(plan),
                     "examples_sha256": digest(examples), "optimizer": "AdamW", "weight_decay": 0.0,
                     "betas": [0.9, 0.999], "eps": 1e-8, "gradient_clip": 1.0,
                     "scheduler": "constant", "deterministic_algorithms": True}
    provenance = write_provenance(output, command="sft", config=config,
                      inputs=[args.config, Path(args.verified) / "manifest.json",
                              Path(args.verified) / "provenance.json", config["paths"]["manifest"],
                              config["paths"]["split_freeze"]])
    write_json(output / "mask_report.json", report, exclusive=True)
    write_json(output / "exposure_plan.json", plan, exclusive=True)
    write_json(output / "example_sources.json", [e["label_provenance"] for e in examples], exclusive=True)
    model = attach_lora(load_student(args.snapshot), rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    write_json(output / "preparation.json", {"seconds": time.monotonic() - preparation_start,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    checkpoints = train_loop(model, examples, plan, output, learning_rate=args.learning_rate,
                             tokenizer=tokenizer, identity_sha256=digest(provenance))
    print(json.dumps({"checkpoints": [c["path"] for c in checkpoints],
                      "supervised_tokens": checkpoints[-1]["supervised_tokens"]}))


if __name__ == "__main__":
    main()
