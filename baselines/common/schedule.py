"""X5 dose and HF step semantics, without a transformers import at preparation."""
import math
import random


def dose(n_examples, original_batch, *, epochs=3):
    if any(type(v) is not int or v < 1 for v in (n_examples, original_batch, epochs)):
        raise ValueError("N, original batch and epochs must be positive integers")
    batch = min(original_batch, max(1, n_examples // 8))
    steps = math.ceil(n_examples / batch)
    return {"b_orig": original_batch, "N_examples": n_examples, "b_prime": batch,
            "steps_per_epoch": steps, "epochs": epochs, "total_steps": steps * epochs,
            "last_batch": n_examples % batch or batch}


def warmup_count(total_steps, warmup_ratio, warmup_steps=None):
    """Ratio uses HF ceil; an explicit integer also permits AgentGym's floor."""
    if total_steps < 1 or not 0 <= warmup_ratio <= 1:
        raise ValueError("Invalid scheduler configuration")
    if warmup_steps is not None:
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        return warmup_steps
    return math.ceil(total_steps * warmup_ratio)


def lr_multiplier(step, *, scheduler, total_steps, warmup_ratio, warmup_steps=None):
    if scheduler not in {"cosine", "constant_with_warmup", "linear"}:
        raise ValueError("Unsupported scheduler")
    if total_steps < 1 or not 0 <= warmup_ratio <= 1 or step < 0:
        raise ValueError("Invalid scheduler configuration")
    # TrainingArguments.get_warmup_steps uses ceil, not floor/round.
    warmup = warmup_count(total_steps, warmup_ratio, warmup_steps)
    if step < warmup:
        return float(step) / max(1, warmup)
    if scheduler == "constant_with_warmup":
        return 1.0
    if scheduler == "linear":
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup))
    progress = float(step - warmup) / max(1, total_steps - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def epoch_batches(n_examples, batch_size, *, epochs=3, seed=0):
    if min(n_examples, batch_size, epochs) < 1:
        raise ValueError("Positive sampler sizes required")
    for epoch in range(epochs):
        order = list(range(n_examples))
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, n_examples, batch_size):
            yield {"epoch": epoch, "indices": order[start:start + batch_size]}


def rank_slots(indices, world_size):
    """Equal collective rounds, exact global batch even when b' < ranks.

    None slots perform a duplicate forward/backward with zero weight. These
    are recorded as physical compute, never extra training examples or tasks.
    """
    if not indices or world_size < 1:
        raise ValueError("Nonempty global batch and positive world size required")
    return [indices[i:i + world_size] + [None] * max(0, i + world_size - len(indices))
            for i in range(0, len(indices), world_size)]
