"""Deterministic replacement continuation from a restored Seed-SFT checkpoint."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import time

from sed.seed_evidence import artifact_hashes
from sed.storage import JsonlLedger, digest, fsync_dir, write_json
from tools.crn_tail import (Experiment, add_state_arguments, require_no_dropout, restore,
                            select_examples, trainables, validate_examples)
from train.optimizer_state import capture_state, learning_rate_at, restore_rng, save_update_state
from train.sft import checkpoint_metadata, token_mean_backward


def continuation_plan(reference, original_plan, state, extra=(), *, target_tokens=416280,
                      fraction=0.0, seed=0, schedule=None, token_budget=512):
    """Preserve the prefix; replace absolute update positions, stop on whole examples.

    Fraction scheduling uses floor(k*f)-floor((k-1)*f), k=1 at the first
    remaining update. Explicit schedules use absolute 1-based update numbers,
    as either a list (sample a batch) or {number: extra IDs}. Extra batches use
    independent uniform draws with replacement until >= token_budget tokens.
    """
    validate_examples(reference)
    if extra:
        validate_examples(extra)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1 or type(target_tokens) is not int:
        raise ValueError("Invalid fraction or target budget")
    if not original_plan or token_budget < 1:
        raise ValueError("Nonempty reference plan and positive batch token budget required")
    cursor, seen = state["data_cursor"], state["supervised_tokens"]
    if cursor != state["updates"] or not 0 <= cursor <= len(original_plan):
        raise ValueError("Checkpoint cursor differs from reference plan")
    if cursor and seen != original_plan[cursor - 1]["cumulative_supervised_tokens"]:
        raise ValueError("Checkpoint token count differs from reference plan")
    if target_tokens <= seen:
        raise ValueError("Target budget must exceed checkpoint exposure")
    if schedule is not None and fraction != 0:
        raise ValueError("Use fraction OR explicit schedule")
    if schedule is None:
        explicit = None
    elif isinstance(schedule, list):
        if any(type(x) is not int for x in schedule) or len(set(schedule)) != len(schedule):
            raise ValueError("Schedule positions must be distinct integers")
        explicit = dict.fromkeys(schedule)
    elif isinstance(schedule, dict):
        explicit = {int(k): v for k, v in schedule.items()}
        if len(explicit) != len(schedule):
            raise ValueError("Duplicate schedule positions")
    else:
        raise ValueError("Schedule must be a list or object")
    if explicit and min(explicit) <= cursor:
        raise ValueError("Schedule cannot replace checkpoint prefix updates")
    if (fraction or explicit) and not extra:
        raise ValueError("Replacement requires an extra example pool")
    rng = random.Random(seed)
    plan = deepcopy(original_plan[:cursor])
    examples = list(reference) + list(extra)
    prefix = original_plan[cursor - 1] if cursor else {}
    turns = prefix.get("turns_seen", 0)
    count = prefix.get("examples_seen", 0)
    bank_tokens = sum(e["supervised_tokens"] for e in reference)
    relative, replacement_count = 0, 0
    while seen < target_tokens:
        relative += 1
        update = cursor + relative
        reference_position = update - 1
        reference_index = reference_position % len(original_plan)
        reference_batch = original_plan[reference_index]
        replace = (update in explicit) if explicit is not None else (
            math.floor(relative * fraction + 1e-12) > math.floor((relative - 1) * fraction + 1e-12))
        if replace:
            replacement_count += 1
            requested = explicit[update] if explicit is not None else None
            if requested is not None:
                selected = select_examples(extra, requested)
                # Identity preserves duplicate-equal rows as distinct exposure units.
                lookup = {id(row): i for i, row in enumerate(extra)}
                indices = [len(reference) + lookup[id(e)] for e in selected]
            else:
                indices, tokens = [], 0
                while tokens < token_budget:
                    index = rng.randrange(len(extra))
                    indices.append(len(reference) + index)
                    tokens += extra[index]["supervised_tokens"]
        else:
            indices = reference_batch["indices"]
        used, cost = [], 0
        for index in indices:
            row = examples[index]
            used.append(index)
            cost += row["supervised_tokens"]
            seen += row["supervised_tokens"]
            turns += row.get("turn_count", 1)
            count += 1
            if seen >= target_tokens:
                break
        final = seen >= target_tokens
        plan.append({"indices": used, "supervised_tokens": cost, "cumulative_supervised_tokens": seen,
            "turns_seen": turns, "examples_seen": count, "packing": reference_batch["packing"],
            "passes": count / len(reference), "token_equivalent_passes": seen / bank_tokens,
            "requested_endpoints": [target_tokens] if final else [], "final": final,
            "pool": "extra" if replace else "reference", "reference_position": reference_position,
            "reference_plan_index": reference_index, "reference_cycle": reference_position // len(original_plan),
            "reference_indices_replaced": list(reference_batch["indices"]) if replace else [],
            "relative_update": relative})
    unused = sorted(set(explicit or {}) - set(range(cursor + 1, len(plan) + 1)))
    if unused:
        raise ValueError("Explicit replacement schedule extends beyond the realized target budget")
    return examples, plan, {"target_tokens": target_tokens, "realized_tokens": seen,
        "overshoot_tokens": seen - target_tokens, "remaining_updates": relative,
        "replacement_updates": replacement_count, "reference_cycles": plan[-1]["reference_cycle"],
        "fraction": fraction, "seed": seed, "schedule": schedule,
        "pass_denominator": "reference bank; extra exposures are counted but are not new seed trajectories"}


def continue_training(model, optimizer, state, examples, plan, output, *, identity_sha256,
                      tokenizer=None, save_every_update=False):
    """Resume native updates; no reinitialization of optimizer or data/RNG cursor."""
    import torch
    require_no_dropout(model)
    validate_examples(examples)
    cursor = state["data_cursor"]
    if cursor >= len(plan):
        raise ValueError("Continuation plan has no remaining updates")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "updates.jsonl").exists():
        raise ValueError("Refusing to append to an existing continuation")
    restore(model, optimizer, state)
    parameters = list(trainables(model).values())
    rate = state.get("learning_rate", optimizer.param_groups[0]["lr"])
    warmup = state.get("lr_warmup_steps", 0)
    clipping = state.get("gradient_clip", 1.0)
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    start, compute, checkpoints = time.monotonic(), 0.0, []
    ledger = JsonlLedger(output / "updates.jsonl")
    for position in range(cursor, len(plan)):
        batch, update = plan[position], position + 1
        if cuda:
            torch.cuda.synchronize(device)
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss, tokens = token_mean_backward(model, examples, batch)
        norm = torch.nn.utils.clip_grad_norm_(parameters, clipping, error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(rate, warmup, update)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize(device)
        seconds = time.monotonic() - tick
        compute += seconds
        ledger.append({"update": update, **batch, "loss": loss / tokens, "seconds": seconds,
            "gradient_norm": float(norm), "clip_active": bool(clipping / (norm + 1e-6) < 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "example_sources": [examples[i].get("label_provenance", {}) for i in batch["indices"]]},
            unique_key="update")
        extra = {"parent_identity_sha256": state.get("identity_sha256"),
                 "resume_update": cursor, "reference_position": batch.get("reference_position", position),
                 "moment_warmup_exposure": state.get("moment_warmup_exposure")}
        # Always save the final named state too: full Python/CPU/CUDA RNG and exact trainables.
        if save_every_update or batch["final"]:
            save_update_state(model, optimizer, output, update=update, batch=batch, plan=plan,
                learning_rate=rate, lr_warmup_steps=warmup, gradient_clip=clipping,
                identity_sha256=identity_sha256, moment_warmup_steps=state.get("moment_warmup_steps", 0),
                extra=extra, tokenizer=tokenizer)
        if batch["requested_endpoints"] or batch["final"]:
            metadata = checkpoint_metadata(batch, updates=update, wall_seconds=time.monotonic() - start,
                update_seconds=compute, peak_memory=torch.cuda.max_memory_allocated(device) if cuda else None,
                peak_reserved=torch.cuda.max_memory_reserved(device) if cuda else None,
                bank_tokens=sum(e["supervised_tokens"] for e in examples), identity_sha256=identity_sha256)
            metadata.update(extra)
            metadata.update(continued_updates=update - cursor, continuation_compute_seconds=compute,
                continuation_seconds_per_update=compute / (update - cursor),
                continuation_supervised_tokens=batch["cumulative_supervised_tokens"] - state["supervised_tokens"])
            name = f"checkpoint-tokens-{batch['cumulative_supervised_tokens']}"
            staging = output / ("." + name + ".partial")
            staging.mkdir(exist_ok=False)
            saved = capture_state(model, optimizer, updates=update,
                                  supervised_tokens=batch["cumulative_supervised_tokens"])
            try:
                model.save_pretrained(staging, safe_serialization=True)
                if tokenizer is not None:
                    tokenizer.save_pretrained(staging)
                # Superset of the original token-checkpoint payload, readable by its loader.
                torch.save(saved, staging / "optimizer.pt")
                metadata["artifact_sha256"] = artifact_hashes(staging)
                write_json(staging / "training_state.json", metadata, exclusive=True)
                os.rename(staging, output / name)
                fsync_dir(output)
            finally:
                restore_rng(saved)
            checkpoints.append({"path": name, **metadata})
    report = {"checkpoints": checkpoints, "wall_seconds": time.monotonic() - start,
              "continued_updates": len(plan) - cursor, "optimizer_compute_seconds": compute,
              "seconds_per_update": compute / (len(plan) - cursor)}
    write_json(output / "profiling_report.json", report, exclusive=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_state_arguments(parser, surrogate=False)
    parser.add_argument("--extra", help="Extra verified example dataset root")
    parser.add_argument("--target-tokens", type=int, default=416280)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--fraction", type=float, default=0.0)
    selection.add_argument("--schedule", help="JSON absolute 1-based update positions or position-to-extra-ID-list map")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every-update", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Continuation requires a new output directory")
    experiment = Experiment(args)
    extra = experiment.load_dataset(args.extra) if args.extra else []
    schedule = json.loads(Path(args.schedule).read_text()) if args.schedule else None
    examples, plan, accounting = continuation_plan(experiment.reference, experiment.plan, experiment.state, extra,
        target_tokens=args.target_tokens, fraction=args.fraction, seed=args.seed, schedule=schedule,
        token_budget=experiment.recipe["tokens_per_update"])
    # Provenance reflects the actual modified plan and combined example bank.
    recipe = {**experiment.recipe, "plan_sha256": digest(plan), "examples_sha256": digest(examples),
              "supervised_tokens": args.target_tokens, "output": str(output),
              "continuation": {**accounting, "reference_root": args.verified, "extra_root": args.extra,
                               "parent_checkpoint": str(experiment.checkpoint)}}
    config = {**experiment.provenance["config"], "sft": recipe, "continuation_tool": vars(args)}
    provenance = experiment.write_provenance(output, "continue_from_state", config=config,
                                             extra_inputs=[args.schedule] if args.schedule else [])
    output.mkdir(parents=True, exist_ok=False)
    # Same run layout as train/sft, not a new adapter export convention.
    write_json(output / "provenance.json", provenance, exclusive=True)
    write_json(output / "exposure_plan.json", plan, exclusive=True)
    write_json(output / "example_sources.json", [e.get("label_provenance", {}) for e in examples], exclusive=True)
    write_json(output / "continuation_plan.json", accounting, exclusive=True)
    report = continue_training(experiment.model, experiment.optimizer, experiment.state, examples, plan, output,
        identity_sha256=digest(provenance), tokenizer=experiment.tokenizer, save_every_update=args.save_every_update)
    print(json.dumps({"checkpoints": [c["path"] for c in report["checkpoints"]], **accounting}))


if __name__ == "__main__":
    main()
