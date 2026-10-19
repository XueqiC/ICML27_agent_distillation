"""DAPO math adapted from OPPO AFM/verl at 9a6594d (Apache-2.0).

Sources: verl/trainer/ppo/core_algos.py compute_grpo_outcome_advantage,
compute_policy_loss; ray_trainer.py filter_groups; web-agent DAPO launcher.
No upstream models, rewards, hidden data or distributed runtime are imported.
"""
from pathlib import Path

from baselines.coa import RECIPE
from baselines.common.conversation import sequence_rows
from baselines.common.schedule import rank_slots
from sed.storage import JsonlLedger


def query_schedule(ids):
    """Pinned RandomSampler seed=1 and DataLoader(drop_last=True), two epochs."""
    import torch
    from baselines.common.schedule import dose
    if not ids:
        return {"task_ids": [], "dropped": [], "generation_batch": 0}
    batch = dose(len(ids), RECIPE["rl"]["generation_batch"], epochs=1)["b_prime"]
    generator = torch.Generator().manual_seed(1)
    sampler = torch.utils.data.RandomSampler(ids, generator=generator)
    result, dropped = [], []
    for epoch in range(RECIPE["rl"]["epochs"]):
        order = list(sampler)
        count = len(order) // batch * batch
        result.extend(ids[i] for i in order[:count])
        dropped.append({"epoch": epoch, "task_ids": [ids[i] for i in order[count:]]})
    return {"task_ids": result, "dropped": dropped, "generation_batch": batch, "sampler_seed": 1}


def rewards(episodes):
    if len(episodes) != 8 or any(type(e.get("success")) is not bool for e in episodes):
        raise ValueError("Eight known environment outcomes required")
    return [float(e["success"]) for e in episodes]


def query_kept(episodes):
    return sum(rewards(episodes)) != 8


def mixed(episodes):
    return 0 < sum(rewards(episodes)) < 8


def advantages(values):
    import torch
    scores = torch.as_tensor(values, dtype=torch.float32)
    if scores.numel() != 8 or not torch.isfinite(scores).all() or not ((scores == 0) | (scores == 1)).all():
        raise ValueError("DAPO reward is binary environment success, eight samples")
    # torch.std's default sample correction matches the pinned GRPO function.
    return (scores - scores.mean()) / (scores.std() + 1e-6)


def policy_loss(log_prob, old_log_prob, advantage, *, reduction="mean"):
    import torch
    cfg = RECIPE["rl"]
    ratio = (log_prob - old_log_prob.detach()).clamp(-20, 20).exp()
    a = torch.as_tensor(advantage, dtype=log_prob.dtype, device=log_prob.device)
    raw = -a * ratio
    clipped = -a * ratio.clamp(1 - cfg["clip_low"], 1 + cfg["clip_high"])
    upper = torch.maximum(raw, clipped)
    loss = torch.where(a < 0, torch.minimum(-a * cfg["clip_c"], upper), upper)
    return loss.sum() if reduction == "sum" else loss.mean()


def token_logps(model, row, device):
    import torch
    ids = torch.tensor([row["input_ids"]], device=device)
    labels = torch.tensor(row["labels"], device=device)
    positions = torch.nonzero(labels[1:] != -100).flatten()
    if not len(positions) or labels[0] != -100:
        raise ValueError("No causal response tokens")
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=positions).logits[0]
    values = []
    for start in range(0, len(positions), 128):
        target = labels[positions[start:start + 128] + 1]
        values.append(logits[start:start + 128].float().log_softmax(-1).gather(1, target[:, None]).squeeze(1))
    return torch.cat(values)


def train_update(model, optimizer, groups, output, *, update, rank=0, world_size=1, device="cpu"):
    """One PPO epoch/minibatch; old probabilities frozen before any optimizer step.

    A logical batch consists of whole groups of eight trajectories. Turns are
    only physical microsteps; observations never enter the policy loss.
    """
    import torch
    from contextlib import nullcontext
    if not groups or not 0 <= update < RECIPE["rl"]["max_updates"]:
        raise ValueError("Empty/out-of-budget DAPO update")
    flat = []
    for group in groups:
        a = advantages(group["rewards"])
        if not a.abs().sum() or len(group["examples"]) != 8:
            raise ValueError("DAPO dynamic sampling must remove constant-reward groups")
        for example, value in zip(group["examples"], a.tolist()):
            flat.extend((row, value) for row in sequence_rows(example))
    tokens = sum(sum(v != -100 for v in r["labels"]) for r, _ in flat)
    if not tokens:
        raise ValueError("Mixed outcomes contain no sampled response tokens")
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()
    slots = rank_slots(list(range(len(flat))), world_size)
    old = {}
    with torch.no_grad(), amp():
        for slot in slots:
            index = slot[rank]
            # Identical collective forward counts on every FSDP rank.
            old[index] = token_logps(model, flat[index or 0][0], device).detach().cpu()
    lr = RECIPE["rl"]["learning_rate"] * min(1.0, update / RECIPE["rl"]["warmup_steps"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.
    for slot in slots:
        index = slot[rank]
        row, a = flat[index or 0]
        with amp():
            logp = token_logps(model, row, device)
            loss = policy_loss(logp, old[index].to(device), a, reduction="sum")
            loss = loss * (world_size / tokens if index is not None else 0)
        loss.backward()
        loss_value += float(loss.detach()) / world_size
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    if hasattr(norm, "full_tensor"):
        norm = norm.full_tensor()
    if world_size > 1:
        value = torch.tensor(loss_value, device=device)
        torch.distributed.all_reduce(value)
        loss_value = float(value)
    result = {"start_update": update, "end_update": update + 1, "lr": lr,
        "loss": loss_value, "gradient_norm": float(norm), "groups_seen": len(groups),
        "examples_seen": len(groups) * 8, "supervised_tokens_seen": tokens,
        "logical_forward_tokens": sum(len(r["input_ids"]) for r, _ in flat),
        "old_logprob_forward_tokens": sum(len(r["input_ids"]) for r, _ in flat),
        "loss_reduction": "global_response_token_mean", "teacher_calls": 0}
    if rank == 0:
        JsonlLedger(Path(output) / "updates.jsonl").append({"update": update + 1, **result})
    return result


if __name__ == "__main__":
    import argparse
    from baselines.coa.data import read
    from sed.storage import write_json
    parser = argparse.ArgumentParser(description="Freeze the official CPU query sampler order")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, query_schedule(read(args.tasks)["task_ids"]))
