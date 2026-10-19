"""Equation (8)-(9), exact global minibatch gradients with sharded storage."""
import math
from pathlib import Path

from baselines.common.fsdp_train import sequence_logps, supervised_count
from baselines.common.schedule import dose, epoch_batches, lr_multiplier, rank_slots
from baselines.had import RECIPE
from sed.storage import JsonlLedger


def action_margin(scores, row):
    if all(row["chosen"][key] == row["rejected"][key] for key in ("input_ids", "labels")):
        # Identical actions have exactly zero derivative (paper A.1). Batched
        # GEMM cancellation can leave roundoff which norm matching amplifies.
        return scores.sum() * 0.0
    return scores[0] / supervised_count(row["chosen"]) - scores[1] / supervised_count(row["rejected"])


def losses(model, row, device, *, beta=RECIPE["beta"]):
    import torch.nn.functional as F
    distillation = -sequence_logps(model, [row["response"]], device)[0] / supervised_count(row["response"])
    scores = sequence_logps(model, [row["chosen"], row["rejected"]], device)
    delta = action_margin(scores, row)
    return distillation, -F.logsigmoid(beta * delta)


def gradient_norm(parameters):
    import torch
    norm = torch.nn.utils.clip_grad_norm_(parameters, math.inf, error_if_nonfinite=True)
    # DTensor reductions include every shard; do not average local shard norms.
    return float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm)


def balanced_backward(model, rows, device="cpu", *, rank=0, world_size=1,
                      beta=RECIPE["beta"], rho=RECIPE["rho"]):
    """Two streamed backward passes; lambda is detached, no higher derivatives.

    Keep one sharded distillation-gradient buffer, not a batch of graphs.
    Every rank executes both passes including zero-weight dummy rows.
    """
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    parameters = [p for p in model.parameters() if p.requires_grad]
    n, valid = len(rows), sum(bool(r["mask"]) for r in rows)
    slots = rank_slots(list(range(n)), world_size)
    values = torch.zeros(2, device=device)
    model.zero_grad(set_to_none=True)
    for slot in slots:
        index = slot[rank]
        row = rows[index if index is not None else 0]
        loss = -sequence_logps(model, [row["response"]], device)[0] / supervised_count(row["response"])
        scale = world_size / n if index is not None else 0.0
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite HAD response loss")
        (loss * scale).backward()
        values[0] += loss.detach() / n if index is not None else 0
    d_norm = gradient_norm(parameters)
    saved = [p.grad.detach().clone() if p.grad is not None else None for p in parameters]
    model.zero_grad(set_to_none=True)
    for slot in slots:
        index = slot[rank]
        row = rows[index if index is not None else 0]
        scores = sequence_logps(model, [row["chosen"], row["rejected"]], device)
        delta = action_margin(scores, row)
        loss = -F.logsigmoid(beta * delta)
        active = index is not None and row["mask"]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite HAD preference loss")
        (loss * (world_size / valid if active else 0.0)).backward()
        values[1] += loss.detach() / valid if active else 0
    p_norm = gradient_norm(parameters)
    weight = rho * d_norm / p_norm if p_norm else 0.0
    with torch.no_grad():
        for parameter, d_grad in zip(parameters, saved):
            if parameter.grad is not None:
                parameter.grad.mul_(weight)
                if d_grad is not None:
                    parameter.grad.add_(d_grad)
            else:
                parameter.grad = d_grad
    del saved
    if world_size > 1:
        dist.all_reduce(values)
    return {"distillation_loss": float(values[0]), "preference_loss": float(values[1]),
            "loss": float(values[0] + weight * values[1]), "lambda": weight,
            "distillation_gradient_norm": d_norm, "preference_gradient_norm": p_norm,
            "valid_states": valid, "batch_states": n}


def round_plan(rows, *, version, start_update):
    if not 0 <= start_update < RECIPE["updates"]:
        raise ValueError("HAD start update outside the 250-update budget")
    if len({row["state_id"] for row in rows}) != len(rows):
        raise ValueError("A HAD state cannot be used twice")
    if any(not 0 <= version - row["policy_version"] < RECIPE["max_policy_versions"] for row in rows):
        raise ValueError("Stale or future HAD policy version")
    plan = dose(len(rows), RECIPE["original_batch"], epochs=1)
    batches = list(epoch_batches(len(rows), plan["b_prime"], epochs=1, seed=version))
    batches = batches[:RECIPE["updates"] - start_update]
    used = {i for batch in batches for i in batch["indices"]}
    return {**plan, "start_update": start_update, "end_update": start_update + len(batches),
            "policy_version": version, "batches": batches,
            "unused_at_terminal_budget": [r["state_id"] for i, r in enumerate(rows) if i not in used]}


def train_round(model, optimizer, examples, output, *, version, start_update,
                rank=0, world_size=1, device="cpu"):
    import torch
    plan = round_plan(examples, version=version, start_update=start_update)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    ledger = JsonlLedger(Path(output) / "updates.jsonl") if rank == 0 else None
    exposure = supervised = physical_tokens = 0
    for offset, batch in enumerate(plan["batches"]):
        update = start_update + offset
        rows = [examples[i] for i in batch["indices"]]
        lr = RECIPE["learning_rate"] * lr_multiplier(update, scheduler="cosine", total_steps=250,
                                                     warmup_ratio=0, warmup_steps=15)
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = balanced_backward(model, rows, device, rank=rank, world_size=world_size)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                              1.0, error_if_nonfinite=True)
        optimizer.step()
        exposure += len(rows)
        supervised += sum(supervised_count(r["response"]) +
            (sum(supervised_count(r[k]) for k in ("chosen", "rejected")) if r["mask"] else 0) for r in rows)
        for slot in rank_slots(list(range(len(rows))), world_size):
            for index in slot:
                row = rows[index if index is not None else 0]
                physical_tokens += len(row["response"]["input_ids"]) + 2 * max(
                    len(row[k]["input_ids"]) for k in ("chosen", "rejected"))
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        if ledger:
            ledger.append({"update": update + 1, "policy_version": version, "lr": lr,
                "state_ids": [r["state_id"] for r in rows], "examples_seen": exposure,
                "supervised_tokens_seen": supervised, "physical_policy_forward_tokens": physical_tokens,
                "gradient_norm": float(norm), **metrics}, unique_key="update")
    return {**plan, "updates": len(plan["batches"]), "examples_seen": exposure,
            "supervised_tokens_seen": supervised, "physical_policy_forward_tokens": physical_tokens}
