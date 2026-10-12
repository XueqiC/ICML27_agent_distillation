"""R3 preregistered plug-in mean-field curve from Corollary 7.2(a)."""
import argparse
import json
import math
from pathlib import Path
import time

from sed.storage import digest, write_json
from tools.crn_tail import (Experiment, add_state_arguments, message_backward, require_no_dropout,
                            restore, surrogate_gradient, trainables)


APPROXIMATIONS = [
    "Current bias-corrected m and v estimate stationary tail mean and mean square.",
    "After that plug-in, use Corollary 7.2(a): no evolving bias correction, epsilon=0, weight decay=0.",
    "Replace E[m/sqrt(v)] by E[m]/sqrt(E[v]); tail gradients are independent of theta.",
    "F is linearized in theta at the restored state, with a fixed per-token surrogate gradient.",
    "No cross-coordinate curvature or model feedback; sum coordinate responses including message j=0 and T tail steps.",
    "Scaled gradients use c times the measured gradient (native backward can introduce small rounding differences).",
    "Clipped curve scales c*g then applies the global norm-1 rule, including PyTorch's 1e-6 denominator offset.",
    "m=0,v>0 uses the continuous unnormalized extension (outside the nonzero-mean assumption).",
    "v=0 coordinates are excluded and counted; a sensitive excluded coordinate makes the prediction incomplete.",
    "Peak is a grid argmax; zero crossing is an interpolated sign-change bracket, not a continuous optimizer/root certificate.",
]


def mean_field_response(g, m, v, dF, *, beta1=0.9, beta2=0.999, eta=1e-5, T=10):
    """Exact coordinate mean-field response. Inputs m,v ALREADY bias corrected.

    Unnormalized form is algebraically identical to rho/lambda/alpha in 7.2(a)
    for m != 0, and avoids the undefined orientation at m=0. Requires v > 0.
    """
    import torch
    if type(T) is not int or T < 0 or not 0 <= beta1 < 1 or not 0 < beta2 < 1:
        raise ValueError("Invalid horizon or Adam coefficients")
    g, m, v, dF = [x.double() for x in (g, m, v, dF)]
    if any(not bool(torch.isfinite(x).all()) for x in (g, m, v, dF)) or bool((v <= 0).any()):
        raise ValueError("Finite inputs and strictly positive v required")
    response = torch.zeros_like(g)
    for j in range(T + 1):
        b, c = (1 - beta1) * beta1 ** j, (1 - beta2) * beta2 ** j
        # (1-c)*v+c*g^2 is stable even when g=0 and c is close to one.
        message = (m * (1 - b) + b * g) / ((1 - c) * v + c * g.square()).sqrt()
        zero = m * (1 - b) / ((1 - c) * v).sqrt()
        response.add_(-eta * dF * (message - zero))
    return response


def curve_summary(grid, values):
    best = max(range(len(grid)), key=lambda i: values[i])
    crossings = []
    # Exclude the trivial zero-gradient reference at c=0.
    for a, b, fa, fb in zip(grid, grid[1:], values, values[1:]):
        if a > 0 and fa == 0:
            crossings.append({"bracket": [a, a], "linear_interpolation": a})
        if fa * fb < 0:
            crossings.append({"bracket": [a, b], "linear_interpolation": a - fa * (b - a) / (fb - fa)})
    if grid[-1] > 0 and values[-1] == 0:
        crossings.append({"bracket": [grid[-1], grid[-1]], "linear_interpolation": grid[-1]})
    return {"curve": [{"c": c, "predicted_delta_F": y} for c, y in zip(grid, values)],
            "predicted_peak_c_grid": grid[best], "peak_at_grid_boundary": best in (0, len(grid) - 1),
            "zero_crossings": crossings,
            "first_negative_c_grid": next((c for c, v in zip(grid, values) if c > 0 and v < 0), None)}


def predict_from_gradients(state, raw, clipped, dF, grid, *, T=10, device="cpu", raw_norm=None):
    import torch
    grid = sorted(set(grid))
    if not grid or any(not math.isfinite(c) or c < 0 for c in grid):
        raise ValueError("Multiplier grid must be finite and nonnegative")
    if type(T) is not int or T < 0:
        raise ValueError("T must be a nonnegative integer")
    if raw_norm is None:
        raw_norm = math.sqrt(sum(float(g.double().square().sum()) for g in raw.values() if g is not None))
    curves = {"unclipped": [0.0] * len(grid), "clipped": [0.0] * len(grid)}
    counts = {"coordinates": 0, "v_zero": 0, "sensitive_v_zero": 0, "m_zero": 0, "rho_gt_one": 0}
    rho_max, clipped_difference = 0.0, 0.0
    entries = [(i, group) for group in state["optimizer"]["param_groups"] for i in group["params"]]
    for (name, template), (index, group) in zip(state["trainable"].items(), entries):
        entry = state["optimizer"]["state"].get(index, {})
        beta1, beta2 = group["betas"]
        step = float(entry.get("step", 0))
        if step <= 0:
            raise ValueError("Stationary amplitude prediction requires a warmed optimizer state")
        if group["weight_decay"] or group.get("amsgrad", False) or group.get("maximize", False):
            raise ValueError("Ordinary Adam with no decay required")
        eta = state.get("learning_rate", group["lr"])
        # Parameter-sized scratch on GPU, never all 65M coordinates in float64 on GPU.
        m = entry["exp_avg"].to(device=device, dtype=torch.float64) / (1 - beta1 ** step)
        v = entry["exp_avg_sq"].to(device=device, dtype=torch.float64) / (1 - beta2 ** step)
        g = (torch.zeros_like(template) if raw[name] is None else raw[name]).to(device=device, dtype=torch.float64)
        sensitivity = dF[name].to(device=device, dtype=torch.float64)
        valid = v > 0
        counts["coordinates"] += v.numel()
        counts["v_zero"] += int((~valid).sum())
        counts["sensitive_v_zero"] += int((~valid & (sensitivity != 0) & ((g != 0) | (m != 0))).sum())
        counts["m_zero"] += int((valid & (m == 0)).sum())
        m, v, g, sensitivity = [x[valid] for x in (m, v, g, sensitivity)]
        if not v.numel():
            continue
        rho = m.abs() / v.sqrt()
        counts["rho_gt_one"] += int((rho > 1).sum())
        rho_max = max(rho_max, float(rho.max()))
        if clipped[name] is not None:
            measured = clipped[name].to(device=device, dtype=torch.float64)[valid]
            clipped_difference = max(clipped_difference,
                float((measured - g * min(1.0, 1.0 / (raw_norm + 1e-6))).abs().max()))
        for k, c in enumerate(grid):
            for variant in curves:
                factor = min(1.0, 1.0 / (c * raw_norm + 1e-6)) if variant == "clipped" else 1.0
                curves[variant][k] += float(mean_field_response(c * factor * g, m, v, sensitivity,
                    beta1=beta1, beta2=beta2, eta=eta, T=T).sum())
    complete = counts["sensitive_v_zero"] == 0
    return {"schema_version": 1, "T": T, "prediction_complete": complete,
            "counts": counts, "rho_max": rho_max, "raw_gradient_norm": raw_norm,
            "measured_vs_rescaled_clip_max_abs": clipped_difference,
            "predictions": {k: curve_summary(grid, v) for k, v in curves.items()},
            "normalization": "rho=|mhat|/sqrt(vhat); lambda=c*g*sign(mhat)/sqrt(vhat); alpha_eta=-dF*sign(mhat)*eta",
            "approximations": APPROXIMATIONS}


def amplitude_prediction(model, optimizer, state, examples, surrogate, message, grid, *, datasets=None, T=10):
    require_no_dropout(model)
    start = time.monotonic()
    restore(model, optimizer, state)
    try:
        dF = surrogate_gradient(model, surrogate)
        # Message must define the unscaled base batch; the grid owns multiplication/clipping.
        from tools.crn_tail import resolve_message
        _, multipliers, _, _ = resolve_message(message, examples, datasets)
        if not multipliers or any(c != 1 for c in multipliers):
            raise ValueError("Amplitude base message must be nonzero and unscaled")
        if isinstance(message, dict) and "union" in message:
            raise ValueError("Amplitude base message must be a single episode batch")
        message = {**message, "clip": True}
        raw, stats = message_backward(model, message, examples, datasets)
        clipped = {n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in trainables(model).items()}
        result = predict_from_gradients(state, raw, clipped, dF, grid, T=T,
            device=next(model.parameters()).device, raw_norm=stats["pre_clip_grad_norm"])
        result["message_stats"] = stats
    finally:
        restore(model, optimizer, state)
    result["wall_seconds"] = time.monotonic() - start
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_state_arguments(parser)
    parser.add_argument("--message", required=True)
    parser.add_argument("--grid", type=float, nargs="+", default=[0, 0.5, 1, 2, 4, 8])
    parser.add_argument("--tail-updates", type=int, default=10)
    args = parser.parse_args(argv)
    message = json.loads(Path(args.message).read_text())
    experiment = Experiment(args)
    experiment.load_messages([message])
    provenance = experiment.write_provenance(args.output, "amplitude_prediction", extra_inputs=[args.message])
    result = amplitude_prediction(experiment.model, experiment.optimizer, experiment.state,
        experiment.reference, experiment.surrogate, message, args.grid, datasets=experiment.datasets,
        T=args.tail_updates)
    write_json(args.output, {**result, "provenance_sha256": digest(provenance)}, exclusive=True)


if __name__ == "__main__":
    main()
