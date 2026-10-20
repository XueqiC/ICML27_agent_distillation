"""Training-only convergence rule (user decision 2026-10-20)."""
import math

DEFAULT_DELTA = 0.01


def rule(threshold=DEFAULT_DELTA):
    return (f"L_last(prev) - L_last(current) < {threshold:g} nats/supervised-token; "
            "adjacent doubling rungs; token-weighted last complete epoch training loss only")


def compare_losses(previous, current, threshold=DEFAULT_DELTA):
    if not all(math.isfinite(v) for v in (previous, current, threshold)) or threshold <= 0:
        raise ValueError("Finite losses and a positive finite convergence delta required")
    delta = previous - current
    return {"L_last_prev": previous, "L_last": current, "delta": delta,
            "converged": delta < threshold, "rule": rule(threshold)}


def doubling_passes(max_passes):
    if type(max_passes) is not int or max_passes < 2 or max_passes & (max_passes - 1):
        raise ValueError("--max-passes must be a power of two, at least 2")
    return [2 ** i for i in range(max_passes.bit_length())]
