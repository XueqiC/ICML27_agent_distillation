"""Training-only convergence with divergence guard (user decision 2026-10-21)."""
import math

DEFAULT_DELTA = 0.01


def rule(threshold=DEFAULT_DELTA):
    return ("both rungs valid: L_last <= max(1.1 * L_min, L_min + 0.02) within each rung; "
            "current rung also requires L_last(current) <= max(1.1 * L_last(prev), L_last(prev) + 0.02); "
            f"L_last(prev) - L_last(current) < {threshold:g} nats/supervised-token; "
            "adjacent doubling rungs; token-weighted last complete epoch training loss only")


def rung_status(last, minimum):
    """Use the minimum complete-epoch token mean from this same rung."""
    if not all(math.isfinite(v) for v in (last, minimum)) or not 0 <= minimum <= last:
        raise ValueError("Finite nonnegative rung losses with L_min <= L_last required")
    limit = max(1.1 * minimum, minimum + 0.02)
    valid = last <= limit
    return {"L_min": minimum, "L_last": last, "valid": valid, "validity_limit": limit,
            "reason": "Last complete epoch is near the rung minimum" if valid else
                      "Unrecovered divergence: last complete epoch exceeds the rung validity limit"}


def compare_losses(previous, current, threshold=DEFAULT_DELTA, *, previous_min, current_min):
    if not all(math.isfinite(v) for v in (previous, current, threshold)) or threshold <= 0:
        raise ValueError("Finite losses and a positive finite convergence delta required")
    prev, curr = rung_status(previous, previous_min), rung_status(current, current_min)
    delta = previous - current
    cross_rung_limit = max(1.1 * previous, previous + 0.02)
    cross_rung_valid = current <= cross_rung_limit
    failed_checks = [name + "_within_rung" for name, status in (("previous", prev), ("current", curr))
                     if not status["valid"]]
    if not cross_rung_valid:
        failed_checks.append("cross_rung")
    reason = ("Unrecovered divergence; failed checks: " + ", ".join(failed_checks) if failed_checks else
              "Both rungs valid; delta below threshold" if delta < threshold else
              "Both rungs valid; delta at or above threshold")
    reason += f"; cross-rung limit={cross_rung_limit:g}"
    if not cross_rung_valid:
        reason += "; doubling the training ended higher than the half-length rung: unrecovered divergence"
    return {"L_last_prev": previous, "L_last": current, "delta": delta,
            "previous": prev, "current": curr, "reason": reason,
            "cross_rung_limit": cross_rung_limit, "cross_rung_valid": cross_rung_valid,
            "failed_checks": failed_checks,
            "converged": not failed_checks and delta < threshold, "rule": rule(threshold)}


def doubling_passes(max_passes):
    if type(max_passes) is not int or max_passes < 2 or max_passes & (max_passes - 1):
        raise ValueError("--max-passes must be a power of two, at least 2")
    return [2 ** i for i in range(max_passes.bit_length())]
