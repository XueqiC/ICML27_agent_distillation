"""R0a sensitivity-weighted frozen-denominator error (Definition 5.4/Prop. 5.5)."""
import argparse
import json
import math
from pathlib import Path
import re
import time

from sed.storage import digest, write_json
from tools.crn_tail import (Experiment, add_state_arguments, require_no_dropout, restore,
                            surrogate_gradient)
from tools.grad_stats import gradient_and_step
from train.optimizer_state import _kappa_summary


def coordinate_warmth(g, m, v, step, *, beta1=0.9, beta2=0.999, eps=1e-8):
    """Raw saved moments, consumed gradient, and PRE-message optimizer step."""
    import torch
    g, m, v = g.double(), m.double(), v.double()
    t = step + 1
    a = (beta2 * v / (1 - beta2 ** t)).sqrt()
    n = (beta1 * m + (1 - beta1) * g) / (1 - beta1 ** t)
    kappa = torch.where(v == 0, torch.inf, (1 - beta2) * g.square() / (beta2 * v))
    u = n.abs() / (a + eps)
    return kappa, u


def parameter_groups(name):
    block = "A" if ".lora_A." in name else "B" if ".lora_B." in name else "other"
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return block, match.group(1) if match else "other"


def warmth_report(model, optimizer, state, examples, surrogate, message, *, datasets=None, arrays_path=None):
    import torch
    require_no_dropout(model)
    start = time.monotonic()
    restore(model, optimizer, state)
    try:
        sensitivity = surrogate_gradient(model, surrogate)
        raw, gradients, delta, stats = gradient_and_step(model, optimizer, state, examples, message,
                                                        datasets=datasets)
        del raw
        entries = [(i, group) for group in state["optimizer"]["param_groups"] for i in group["params"]]
        if len(entries) != len(state["trainable"]):
            raise ValueError("Optimizer/trainable layout differs")
        aggregates, top, kappas, zeros, arrays = {}, [], [], [], {}
        share_numerator, share_denominator = 0.0, 0.0
        for (name, template), (index, group) in zip(state["trainable"].items(), entries):
            # None gradients skip native Adam; do not pretend they consumed a zero.
            if gradients[name] is None:
                continue
            g = gradients[name].double()
            entry = state["optimizer"]["state"].get(index, {})
            m = entry.get("exp_avg", torch.zeros_like(g)).double()
            v = entry.get("exp_avg_sq", torch.zeros_like(g)).double()
            beta1, beta2 = group["betas"]
            kappa, u = coordinate_warmth(g, m, v, float(entry.get("step", 0)),
                                        beta1=beta1, beta2=beta2, eps=group["eps"])
            L = sensitivity[name].double().abs()
            rate = state.get("learning_rate", group["lr"])
            term = rate * L * u * torch.minimum(kappa / 2, torch.ones_like(kappa))
            signed = L * delta[name]
            actual_signed = sensitivity[name].double() * delta[name]
            block, layer = parameter_groups(name)
            for key in ("all", "block:" + block, "layer:" + layer, "layer_block:" + layer + ":" + block):
                row = aggregates.setdefault(key, {"E_fd": 0.0, "first_order_L_signed": 0.0,
                    "parameter_first_order_signed": 0.0, "max_term": 0.0, "coordinates": 0})
                row["E_fd"] += float(term.sum())
                row["first_order_L_signed"] += float(signed.sum())
                row["parameter_first_order_signed"] += float(actual_signed.sum())
                row["max_term"] = max(row["max_term"], float(term.max()))
                row["coordinates"] += term.numel()
            values, positions = torch.topk(term.flatten(), min(20, term.numel()))
            for value, position in zip(values.tolist(), positions.tolist()):
                coordinate = list(torch.unravel_index(torch.tensor(position), template.shape))
                top.append({"parameter": name, "flat_index": position, "coordinate": [int(c) for c in coordinate],
                            "term": value, "kappa": float(kappa.flatten()[position]),
                            "u": float(u.flatten()[position]), "L": float(L.flatten()[position])})
            top = sorted(top, key=lambda x: (-x["term"], x["parameter"], x["flat_index"]))[:20]
            kappas.append(kappa.flatten())
            zeros.append((v == 0).flatten())
            share_numerator += float(((1 - beta1) * g).square().sum())
            share_denominator += float((beta1 * m + (1 - beta1) * g).square().sum())
            if arrays_path:
                for key, value in (("kappa", kappa), ("u", u), ("L", L), ("delta_theta", delta[name])):
                    arrays[name + "::" + key] = value.half().numpy()
        if not kappas:
            raise ValueError("No consumed gradients")
        for row in aggregates.values():
            first = abs(row["first_order_L_signed"])
            row["first_order_L_abs"] = first
            row["E_fd_over_first_order"] = row["E_fd"] / first if first else None
            row["preregistered_warm_criterion"] = row["E_fd"] <= 0.1 * first
        quantiles = _kappa_summary(torch.cat(kappas), torch.cat(zeros))
        quantiles["gradient_share"] = math.sqrt(share_numerator / share_denominator) if share_denominator else None
        for row in top:
            if not math.isfinite(row["kappa"]):
                row["kappa"] = "inf"
        if arrays_path:
            import numpy as np
            # Explicit opt-in only; exclusive open avoids replacing another diagnostic.
            with open(arrays_path, "xb") as handle:
                np.savez_compressed(handle, **arrays)
        result = {"schema_version": 1, **aggregates.pop("all"), "breakdown": aggregates,
                "top20": top, "old_quantiles": quantiles, "message_stats": stats,
                "sensitivity": "L=abs(gradient of micro token-mean held-out log likelihood); proxy, not a Lipschitz bound",
                "first_order_convention": "L_signed=sum |dF_i|*realized_delta_i; parameter_signed=sum dF_i*realized_delta_i",
                "arithmetic": "float64 diagnostics; raw moments with next-step bias correction; native realized delta",
                "arrays_npz": str(arrays_path) if arrays_path else None,
                "wall_seconds": time.monotonic() - start}
    finally:
        restore(model, optimizer, state)
    result["wall_seconds"] = time.monotonic() - start
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_state_arguments(parser)
    parser.add_argument("--message", required=True, help="JSON message file")
    parser.add_argument("--arrays-float16-npz", help="Opt in to full per-coordinate arrays (large, lossy)")
    args = parser.parse_args(argv)
    message = json.loads(Path(args.message).read_text())
    experiment = Experiment(args)
    experiment.load_messages([message])
    provenance = experiment.write_provenance(args.output, "warmth_dump", extra_inputs=[args.message])
    result = warmth_report(experiment.model, experiment.optimizer, experiment.state,
        experiment.reference, experiment.surrogate, message, datasets=experiment.datasets,
        arrays_path=args.arrays_float16_npz)
    write_json(args.output, {**result, "provenance_sha256": digest(provenance)}, exclusive=True)


if __name__ == "__main__":
    main()
