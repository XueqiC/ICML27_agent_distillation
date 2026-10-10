"""Offline one-step Adam response diagnostics on the frozen Seed-SFT exposure plan."""
import argparse
import json
from pathlib import Path
import time

from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.storage import digest, file_hash, write_json
from train.dataset import build_verified_dataset, load_tokenizer, mask_report
from train.modeling import attach_lora, load_student, snapshot_metadata
from train.optimizer_state import (adamw, bitwise_equal, capture_state, learning_rate_at,
                                   kappa_statistics, load_checkpoint_state, load_update_state,
                                   restore_rng, restore_state)
from train.sft import deterministic_seed, moment_warmup, token_mean_backward


def parse_state(value):
    if value == "cold":
        return "cold", 0
    parts = value.split(":")
    if len(parts) == 2 and parts[0] in {"warmup", "update", "checkpoint"} and parts[1].isdigit() and int(parts[1]) > 0:
        return parts[0], int(parts[1])
    raise ValueError("States must be cold, warmup:<positive W>, update:<positive k>, or checkpoint:<positive tokens>")


def planned_pairs(plan, cursor, count):
    """Cycle only the recorded plan; choose the first subsequent disjoint B."""
    for _ in range(count):
        a_pos = cursor
        a = plan[a_pos % len(plan)]
        skipped = []
        for b_pos in range(a_pos + 1, a_pos + len(plan) + 1):
            b = plan[b_pos % len(plan)]
            if set(a["indices"]).isdisjoint(b["indices"]):
                break
            skipped.append(b_pos)
        else:
            raise ValueError("Frozen plan has no disjoint batch pair at this cursor")
        yield {"A": a, "B": b, "plan_positions": [a_pos, b_pos],
               "plan_indices": [a_pos % len(plan), b_pos % len(plan)], "skipped_positions": skipped}
        cursor = b_pos + 1


def gradients(model, examples, batch, gradient_clip):
    import torch
    parameters = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    loss, tokens = token_mean_backward(model, examples, batch)
    raw = [None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]
    norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
    clipped = [None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]
    return {"unclipped": raw, "clipped": clipped, "norm": float(norm),
            "clip_active": bool(gradient_clip / (norm + 1e-6) < 1), "loss": loss / tokens}


def flat(values, templates):
    import torch
    return torch.cat([(torch.zeros_like(t) if v is None else v).reshape(-1).double()
                      for v, t in zip(values, templates)])


def ratio(numerator, denominator):
    # Undefined ratios are preserved as null, not silently called zero.
    return float(numerator / denominator) if float(denominator) > 0 else None


def one_step_delta(model, optimizer, state, gradient, *, kind, learning_rate):
    """Execute the native optimizer in the original dtype, then subtract in float64."""
    import torch
    restore_state(model, optimizer, state)
    parameters = [p for p in model.parameters() if p.requires_grad]
    branch = optimizer if kind == "AdamW" else torch.optim.SGD(parameters, lr=learning_rate, foreach=False)
    for group in branch.param_groups:
        group["lr"] = learning_rate
    for parameter, value in zip(parameters, gradient):
        parameter.grad = None if value is None else value.to(parameter.device).clone()
    branch.step()
    return torch.cat([(p.detach().cpu().double() - old.double()).reshape(-1)
                      for p, old in zip(parameters, state["trainable"].values())])


def adam_float64_delta(state, gradient, learning_rate):
    """Analytic AdamW (weight_decay=0), raw moments, bias correction, eps outside sqrt.

    This computes the update, not rounded parameter subtraction. None gradients
    skip a native Adam step; an explicit zero gradient advances/decays moments.
    """
    import torch
    deltas = []
    entries = [(index, group) for group in state["optimizer"]["param_groups"] for index in group["params"]]
    for (index, group), value, template in zip(entries, gradient, state["trainable"].values()):
        if group["weight_decay"] != 0 or group.get("amsgrad", False) or group.get("maximize", False):
            raise ValueError("Analytic reference requires ordinary AdamW with weight_decay=0")
        if value is None:
            deltas.append(torch.zeros_like(template, dtype=torch.float64).flatten())
            continue
        entry = state["optimizer"]["state"].get(index, {})
        g = value.double()
        m = entry.get("exp_avg", torch.zeros_like(g)).double()
        v = entry.get("exp_avg_sq", torch.zeros_like(g)).double()
        t = float(entry.get("step", 0))
        b1, b2 = group["betas"]
        first = (b1 * m + (1 - b1) * g) / (1 - b1 ** (t + 1))
        second = (b2 * v + (1 - b2) * g.square()) / (1 - b2 ** (t + 1))
        deltas.append((-learning_rate * first / (second.sqrt() + group["eps"])).flatten())
    return torch.cat(deltas)


def reference_check(reference, native):
    """Record discrepancies, including native parameter-rounding error; never raise.

    Max relative difference is coordinatewise |ref-native| / max(|ref|, |native|),
    then maximized over A/B/AB/0; two zero coordinates contribute 0. The separate
    pass criterion uses atol=1e-12, rtol=1e-6 against the analytic delta. Native
    low-precision parameter subtraction can fail this check even for correct Adam.
    """
    import torch
    atol, rtol = 1e-12, 1e-6
    maximum, absolute, passed = 0.0, 0.0, True
    for key, ref in reference.items():
        difference = (ref - native[key]).abs()
        scale = torch.maximum(ref.abs(), native[key].abs())
        relative = torch.where(scale == 0, 0.0, difference / scale)
        maximum = max(maximum, float(relative.max()))
        absolute = max(absolute, float(difference.max()))
        passed = passed and bool(torch.all(difference <= atol + rtol * ref.abs()))
    return {"max_relative_difference": maximum, "max_abs_difference": absolute,
            "passed": passed, "atol": atol, "rtol": rtol,
            "criterion": "every coordinate: |reference-native| <= atol + rtol * |reference|",
            "relative_difference_rule": "max over A/B/AB/0 and coordinates; |ref-native| / max(|ref|, |native|); 0/0=0",
            "rounding_note": "native parameter subtraction can fail tolerance; failure is recorded, not raised"}


def response_metrics(deltas, gradient_vectors, weights):
    a, b, ab = (deltas[key] for key in ("A", "B", "AB"))
    zero = deltas["0"]
    residual = ab - weights[0] * a - weights[1] * b
    return {"nu": ratio(residual.norm(), ab.norm()),
            "nu_centered": ratio(residual.norm(), (ab - zero).norm()),
            "nu_spread": ratio(residual.norm(), (a - b).norm()),
            "momentum_share": ratio(zero.norm(), ab.norm()),
            "cos_centered_A_B": ratio((a - zero).dot(b - zero), (a - zero).norm() * (b - zero).norm()),
            "centered_dtheta_AB_norm": float((ab - zero).norm()),
            "affine_residual_norm": float(residual.norm()),
            "cos_dtheta_A_B": ratio(a.dot(b), a.norm() * b.norm()),
            "dtheta_norm": {k: float(v.norm()) for k, v in deltas.items()},
            "sign_agreement_with_negative_gradient": {
                k: float((v.sign() == -gradient_vectors[k].sign()).double().mean()) for k, v in deltas.items()}}


def probe_pair(model, optimizer, state, examples, pair, *, learning_rate, gradient_clip):
    import torch
    restore_state(model, optimizer, state)
    a, b = pair["A"], pair["B"]
    union = {"indices": a["indices"] + b["indices"],
             "supervised_tokens": a["supervised_tokens"] + b["supervised_tokens"]}
    # A then B and replayed AB use the same per-example random draws, even with dropout.
    grads = {"A": gradients(model, examples, a, gradient_clip),
             "B": gradients(model, examples, b, gradient_clip)}
    restore_rng(state)
    grads["AB"] = gradients(model, examples, union, gradient_clip)
    templates = list(state["trainable"].values())
    weights = [a["supervised_tokens"] / union["supervised_tokens"],
               b["supervised_tokens"] / union["supervised_tokens"]]
    raw = {k: flat(v["unclipped"], templates) for k, v in grads.items()}
    residual = raw["AB"] - weights[0] * raw["A"] - weights[1] * raw["B"]
    ids = [i for group in state["optimizer"]["param_groups"] for i in group["params"]]
    second = flat([state["optimizer"]["state"].get(i, {}).get("exp_avg_sq") for i in ids], templates)
    result = {"plan_positions": pair["plan_positions"], "plan_indices": pair["plan_indices"],
              "skipped_positions": pair["skipped_positions"],
              "indices": {"A": a["indices"], "B": b["indices"]},
              "supervised_tokens": {"A": a["supervised_tokens"], "B": b["supervised_tokens"]},
              "token_weights": weights, "learning_rate": learning_rate,
              "gradient_union_check": {"gradient_variant": "unclipped",
                  "relative_residual": ratio(residual.norm(), raw["AB"].norm()),
                  "max_abs_difference": float(residual.abs().max()),
                  "passed": bool(residual.norm() <= 1e-12 + 1e-6 * raw["AB"].norm()),
                  "criterion": "L2 residual <= atol + rtol * ||g_AB||", "atol": 1e-12, "rtol": 1e-6},
              "grad_norms": {k: v["norm"] for k, v in grads.items()},
              "clip_active": {k: v["clip_active"] for k, v in grads.items()},
              "kappa": {}, "responses": {},
              "gradient_exposure_tokens": 2 * union["supervised_tokens"]}
    for clipping in ("unclipped", "clipped"):
        vectors = raw if clipping == "unclipped" else {k: flat(v[clipping], templates) for k, v in grads.items()}
        result["kappa"][clipping] = kappa_statistics(vectors["AB"], second)
        result.setdefault("consumed_grad_norms", {})[clipping] = {k: float(v.norm()) for k, v in vectors.items()}
        difference = vectors["AB"] - weights[0] * vectors["A"] - weights[1] * vectors["B"]
        result.setdefault("gradient_union_residuals", {})[clipping] = {
            "relative_residual": ratio(difference.norm(), vectors["AB"].norm()),
            "max_abs_difference": float(difference.abs().max())}
        # Every coordinate gets an explicit gradient so Adam decays both moments.
        inputs = {k: v[clipping] for k, v in grads.items()}
        inputs["0"] = [torch.zeros_like(t) for t in templates]
        response_vectors = {**vectors, "0": flat(inputs["0"], templates)}
        for kind in ("AdamW", "SGD"):
            deltas = {k: one_step_delta(model, optimizer, state, v, kind=kind,
                                       learning_rate=learning_rate) for k, v in inputs.items()}
            if kind == "SGD":
                assert torch.equal(deltas["0"], torch.zeros_like(deltas["0"])), "SGD zero-gradient delta must be exactly zero"
            else:
                reference = {k: adam_float64_delta(state, v, learning_rate) for k, v in inputs.items()}
                result[f"AdamW_{clipping}_float64_reference"] = {
                    **response_metrics(reference, response_vectors, weights),
                    "native_delta_check": reference_check(reference, deltas)}
            result["responses"][f"{kind}_{clipping}"] = response_metrics(deltas, response_vectors, weights)
            del deltas
    # Native parameter rounding can spoil even SGD's null at tiny LR; expose it separately.
    result["SGD_unclipped_float64_reference"] = response_metrics(
        {k: -learning_rate * v for k, v in {**raw, "0": torch.zeros_like(raw["AB"])}.items()},
        {**raw, "0": torch.zeros_like(raw["AB"])}, weights)
    restore_state(model, optimizer, state)
    return result


def compare_tensors(left, right):
    if left.keys() != right.keys():
        return {"bitwise_equal": False, "max_abs_difference": None, "keys_equal": False}
    same, maximum = True, 0.0
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            return {"bitwise_equal": False, "max_abs_difference": None, "layout_equal": False}
        same = same and bitwise_equal(a, b)
        maximum = max(maximum, float((a.cpu().double() - b.cpu().double()).abs().max()))
    return {"bitwise_equal": same, "max_abs_difference": maximum}


def restore_fidelity(model, optimizer, state, expected, examples, plan):
    import torch
    cursor = state["data_cursor"]
    if expected["data_cursor"] != cursor + 1 or cursor >= len(plan):
        raise ValueError("Fidelity check requires the next saved update in the frozen plan")
    restore_state(model, optimizer, state)
    token_mean_backward(model, examples, plan[cursor])
    parameters = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(parameters, state["gradient_clip"], error_if_nonfinite=True)
    optimizer.param_groups[0]["lr"] = learning_rate_at(state["learning_rate"], state["lr_warmup_steps"], cursor + 1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    actual = capture_state(model, optimizer)
    checks = {"adapter": compare_tensors(actual["trainable"], expected["trainable"])}
    for field in ("exp_avg", "exp_avg_sq", "step"):
        checks[field] = compare_tensors({i: s[field] for i, s in actual["optimizer"]["state"].items()},
                                       {i: s[field] for i, s in expected["optimizer"]["state"].items()})
    checks["torch_rng"] = compare_tensors({"rng": actual["torch_rng"]}, {"rng": expected["torch_rng"]})
    checks["cuda_rng"] = compare_tensors(dict(enumerate(actual["cuda_rng"])), dict(enumerate(expected["cuda_rng"])))
    checks["python_rng"] = {"bitwise_equal": actual["python_rng"] == expected["python_rng"]}
    restore_state(model, optimizer, state)
    return {"status": "checked", "update": cursor + 1, "checks": checks,
            "bitwise_equal": all(c["bitwise_equal"] for c in checks.values()),
            "max_abs_difference": max(c.get("max_abs_difference") or 0.0 for c in checks.values()),
            "gradient_exposure_tokens": plan[cursor]["supervised_tokens"]}


class MemoryLedger:
    def __init__(self):
        self.rows = []

    def append(self, row, **_):
        self.rows.append(row)


def run_probe(model, examples, plan, run, states, *, pairs=4, learning_rate=1e-5,
              lr_warmup_steps=0, gradient_clip=1.0, identity_sha256=None):
    if pairs < 1 or not plan or not states:
        raise ValueError("Positive pairs, a nonempty plan, and states are required")
    if any(e.get("label_provenance", {}).get("training_eligible") is False for e in examples):
        raise ValueError("Diagnostic examples cannot enter the training response probe")
    specs = [(value, *parse_state(value)) for value in states]
    optimizer = adamw([p for p in model.parameters() if p.requires_grad], learning_rate)
    cold = capture_state(model, optimizer, data_cursor=0, updates=0, step=0)
    results = []
    try:
        for label, kind, number in specs:
            restore_state(model, optimizer, cold)
            ledger, warm = MemoryLedger(), None
            if kind in {"update", "checkpoint"}:
                if kind == "checkpoint":
                    saved_path = Path(run) / f"checkpoint-tokens-{number}"
                    provenance = json.loads((Path(run) / "provenance.json").read_text())
                    state = load_checkpoint_state(model, optimizer, saved_path, plan=plan,
                                                  provenance=provenance, identity_sha256=identity_sha256)
                else:
                    saved_path = Path(run) / f"state-update-{number}"
                    state = load_update_state(saved_path, plan=plan, identity_sha256=identity_sha256)
                if kind == "update" and state["updates"] != number:
                    raise ValueError("Saved update number differs from requested state")
                if (state["learning_rate"], state["lr_warmup_steps"], state["gradient_clip"]) != (
                        learning_rate, lr_warmup_steps, gradient_clip):
                    raise ValueError("Saved update recipe differs from probe recipe")
                cursor = state["data_cursor"]
                restore_state(model, optimizer, state)
            else:
                if kind == "warmup":
                    warm = moment_warmup(model, optimizer, examples, plan, number,
                                         gradient_clip=gradient_clip, ledger=ledger)
                state = capture_state(model, optimizer, data_cursor=0, updates=0, step=number)
                # Warmup consumes W gradient batches; real training nevertheless restarts at zero.
                cursor = number
            rate = learning_rate_at(learning_rate, lr_warmup_steps, state["updates"] + 1)
            fidelity = {"status": "unavailable", "reason": "No adjacent saved real update"}
            next_path = Path(run) / f"state-update-{state['updates'] + 1}"
            if kind in {"update", "checkpoint"} and next_path.is_dir():
                expected = load_update_state(next_path, plan=plan, identity_sha256=identity_sha256)
                fidelity = restore_fidelity(model, optimizer, state, expected, examples, plan)
            result = {"state": label, "optimizer_step": state["step"], "data_cursor": state["data_cursor"],
                      "probe_batch_cursor": cursor, "next_learning_rate": rate,
                      "warmup_exposure": warm, "warmup_ledger": ledger.rows,
                      "restore_fidelity": fidelity, "pairs": []}
            if kind in {"update", "checkpoint"}:
                result["saved_state_sha256"] = file_hash(saved_path / "training_state.json")
                result["state_moment_warmup_exposure"] = state.get("moment_warmup_exposure")
                if kind == "checkpoint":
                    result["checkpoint_rng_scope"] = state["checkpoint_rng_scope"]
                if fidelity["status"] == "checked":
                    fidelity["saved_next_state_sha256"] = file_hash(next_path / "training_state.json")
            for pair in planned_pairs(plan, cursor, pairs):
                result["pairs"].append(probe_pair(model, optimizer, state, examples, pair,
                                                 learning_rate=rate, gradient_clip=gradient_clip))
            results.append(result)
    finally:
        restore_state(model, optimizer, cold)
    return {"schema_version": 1, "states": results, "teacher_calls": 0,
            "scope": "training-data optimizer diagnostic; no deployment or scientific finding established",
            "pair_policy": "next A, first subsequent disjoint B; cycle frozen plan, report skips and absolute positions",
            "gradient_rng_coupling": "sequential A then B; AB replays the same per-example random draws",
            "update_arithmetic": "native trainable dtype; realized deltas and metrics in float64",
            "sign_agreement_rule": "all coordinates, including zero == zero",
            "undefined_ratio": None, "infinite_kappa": "inf"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "verified", "config", "run", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--states", nargs="+", required=True)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    for state in args.states:
        parse_state(state)
    output, run = Path(args.output), Path(args.run)
    if output.exists() or output.parent.resolve().is_relative_to(run.resolve()):
        raise ValueError("Use a new probe output outside the immutable training run")
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    config = load_config(args.config)
    if config != {k: v for k, v in provenance["config"].items() if k != "sft"}:
        raise ValueError("Probe config differs from training config")
    recipe = provenance["config"]["sft"]
    if snapshot_metadata(args.snapshot) != recipe["base_snapshot"]:
        raise ValueError("Probe base snapshot differs from training snapshot")
    for path, expected in provenance["inputs"].items():
        if file_hash(path) != expected:
            raise ValueError("Training provenance input changed")
    if (recipe["optimizer"], recipe["betas"], recipe["eps"], recipe["weight_decay"]) != (
            "AdamW", [0.9, 0.999], 1e-8, 0.0):
        raise ValueError("Probe requires the recorded Seed-SFT AdamW recipe")
    deterministic_seed(recipe["seed"])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Production probe requires a GPU; CPU plumbing is exercised only by unit tests")
    tokenizer = load_tokenizer(args.snapshot)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    examples = build_verified_dataset(args.verified, tokenizer, manifest, freeze,
                                      max_length=recipe["max_length"], packing=recipe["packing"])
    plan = json.loads((run / "exposure_plan.json").read_text())
    if digest(plan) != recipe["plan_sha256"] or digest(examples) != recipe["examples_sha256"]:
        raise ValueError("Frozen plan or verified examples differ from training")
    if digest(tokenizer.chat_template) != recipe["template_sha256"]:
        raise ValueError("Tokenizer template differs from training")
    mask_report(examples, tokenizer)  # Match preparation order before seeded LoRA initialization.
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance_dir = output.parent / (output.stem + "-provenance")
    provenance_dir.mkdir(exist_ok=False)
    probe_provenance = write_provenance(provenance_dir, command="adam_response_probe",
        config={**config, "sft": recipe, "adam_probe": vars(args)},
        inputs=[args.config, provenance_path, run / "exposure_plan.json", *provenance["inputs"]],
        details={"training_identity_sha256": digest(provenance), "teacher_calls": 0})
    start = time.monotonic()
    model = attach_lora(load_student(args.snapshot), rank=recipe["rank"], alpha=recipe["alpha"],
                        dropout=recipe["dropout"])
    report = run_probe(model, examples, plan, run, args.states, pairs=args.pairs,
        learning_rate=recipe["learning_rate"], lr_warmup_steps=recipe.get("lr_warmup_steps", 0),
        gradient_clip=recipe["gradient_clip"], identity_sha256=digest(provenance))
    report.update(provenance_sha256=digest(probe_provenance), wall_time_seconds=time.monotonic() - start,
                  training_identity_sha256=digest(provenance), plan_sha256=digest(plan),
                  examples_sha256=digest(examples), trainable_dtypes=sorted({
                      str(p.dtype) for p in model.parameters() if p.requires_grad}))
    write_json(output, report, exclusive=True)
    print(json.dumps({"output": str(output), "states": args.states, "pairs_per_state": args.pairs}), flush=True)


if __name__ == "__main__":
    main()
