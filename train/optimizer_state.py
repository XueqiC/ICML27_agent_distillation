"""Exact trainable/optimizer/RNG snapshots; never copy the frozen base model."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random

from sed.seed_evidence import artifact_hashes
from sed.storage import digest, fsync_dir, write_json


def adamw(parameters, learning_rate):
    import torch
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0,
                             betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)


def learning_rate_at(rate, warmup_steps, update):
    return rate * min(update / warmup_steps, 1.0) if warmup_steps else rate


def kappa_statistics(gradient, second_moment, beta2=0.999):
    import torch
    zero = second_moment == 0
    kappa = torch.where(zero, torch.inf, (1 - beta2) * gradient.square() / (beta2 * second_moment))
    return _kappa_summary(kappa, zero)


def _kappa_summary(kappa, zero):
    import torch
    ordered = torch.sort(kappa).values
    def quantile(q):
        value = float(ordered[max(0, math.ceil(q * len(ordered)) - 1)])
        return value if math.isfinite(value) else "inf"
    return {"q0.5": quantile(0.5), "q0.9": quantile(0.9), "q0.99": quantile(0.99),
            "max": quantile(1), "fraction_gt_1": float((kappa > 1).double().mean()),
            "fraction_v_zero": float(zero.double().mean()), "coordinates": len(kappa),
            "quantile_rule": "inverse empirical CDF; v=0 maps to inf including g=0"}


def warmth_statistics(optimizer):
    """Observe consumed gradients and raw pre-step moments, without mutation/RNG.

    All coordinate arithmetic/reductions stay on the training device in float64;
    only summary scalars leave it. Missing moments/gradients are interpreted as 0.
    """
    import torch
    kappas, zeros, numerator, denominator = [], [], [], []
    with torch.no_grad():
        for group in optimizer.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    continue
                state = optimizer.state.get(parameter, {})
                g = (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).detach().double()
                m = state.get("exp_avg", torch.zeros_like(g)).detach().double()
                v = state.get("exp_avg_sq", torch.zeros_like(g)).detach().double()
                zero = v == 0
                kappas.append(torch.where(zero, torch.inf, (1 - beta2) * g.square() / (beta2 * v)).flatten())
                zeros.append(zero.flatten())
                new_gradient = (1 - beta1) * g
                numerator.append(new_gradient.square().sum())
                denominator.append((beta1 * m + new_gradient).square().sum())
        result = _kappa_summary(torch.cat(kappas), torch.cat(zeros))
        norm = torch.stack(denominator).sum().sqrt()
        result["gradient_share"] = float(torch.stack(numerator).sum().sqrt() / norm) if norm > 0 else None
    return result


def rng_state():
    import torch
    return {"torch_rng": torch.get_rng_state().clone(), "python_rng": random.getstate(),
            "cuda_rng": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}


def restore_rng(state):
    import torch
    torch.set_rng_state(state["torch_rng"].cpu())
    random.setstate(state["python_rng"])
    if state["cuda_rng"]:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs from saved state")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def bitwise_equal(a, b):
    import torch
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                            b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def assert_trainable_equal(model, expected):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not bitwise_equal(parameter, expected[name]):
            raise AssertionError(f"Zero-LR warmup changed trainable tensor: {name}")


def cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return deepcopy(value)


def capture_state(model, optimizer, **metadata):
    return {"schema_version": 1, "trainable": trainable_state(model),
            "optimizer": cpu_copy(optimizer.state_dict()), **rng_state(), **metadata}


def restore_state(model, optimizer, state):
    import torch
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if list(parameters) != list(state["trainable"]):
        raise ValueError("Trainable parameter names/order differ from saved state")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state["trainable"][name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError("Trainable parameter shape/dtype differs from saved state")
            parameter.copy_(value)
    # load_state_dict can alias CPU tensors: each branch must own its moments.
    optimizer.load_state_dict(deepcopy(state["optimizer"]))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    restore_rng(state)


def save_update_state(model, optimizer, output, *, update, batch, plan, learning_rate,
                      lr_warmup_steps, gradient_clip, identity_sha256, moment_warmup_steps,
                      extra=None, tokenizer=None):
    import torch
    name = f"state-update-{update}"
    output = Path(output)
    staging = output / ("." + name + ".partial")
    staging.mkdir(exist_ok=False)
    metadata = {"updates": update, "step": moment_warmup_steps + update, "data_cursor": update,
                "supervised_tokens": batch["cumulative_supervised_tokens"],
                "plan_sha256": digest(plan), "identity_sha256": identity_sha256,
                "learning_rate": learning_rate, "lr_warmup_steps": lr_warmup_steps,
                "scheduler": "linear_warmup_then_constant" if lr_warmup_steps else "constant",
                "gradient_clip": gradient_clip, "moment_warmup_steps": moment_warmup_steps,
                **(extra or {})}
    state = capture_state(model, optimizer, **metadata)
    try:
        model.save_pretrained(staging, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(staging)
        # Exact named trainables also support tiny models and avoid PEFT dtype/key coercion.
        torch.save(state, staging / "optimizer.pt")
        write_json(staging / "training_state.json", {"schema_version": 1, **metadata,
                   "artifact_sha256": artifact_hashes(staging)}, exclusive=True)
        os.rename(staging, output / name)
        fsync_dir(output)
    finally:
        restore_rng(state)


def load_update_state(path, *, plan=None, identity_sha256=None):
    import torch
    path = Path(path)
    metadata = json.loads((path / "training_state.json").read_text())
    hashes = artifact_hashes(path)
    hashes.pop("training_state.json")
    if hashes != metadata["artifact_sha256"]:
        raise ValueError("Saved update artifacts changed")
    state = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    for name, value in metadata.items():
        if name != "artifact_sha256" and state.get(name) != value:
            raise ValueError("Saved update metadata differs from optimizer payload")
    if plan is not None and state["plan_sha256"] != digest(plan):
        raise ValueError("Saved update exposure plan differs")
    if identity_sha256 is not None and state["identity_sha256"] != identity_sha256:
        raise ValueError("Saved update provenance differs")
    if state["data_cursor"] != state["updates"] or state["step"] != state["updates"] + state["moment_warmup_steps"]:
        raise ValueError("Saved update counters disagree")
    return state


def load_checkpoint_state(model, optimizer, path, *, plan, provenance, identity_sha256=None):
    """Restore the older token checkpoint format, which saved only CPU torch RNG.

    PEFT's own export/import helpers define the saved-key mapping. No permissive
    casting, missing trainables, extra tensors, or inferred optimizer steps.
    """
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path(path)
    metadata = json.loads((path / "training_state.json").read_text())
    hashes = artifact_hashes(path)
    hashes.pop("training_state.json")
    if hashes != metadata["artifact_sha256"]:
        raise ValueError("Checkpoint artifacts changed")
    identity = digest(provenance)
    if metadata["identity_sha256"] != identity or identity_sha256 not in (None, identity):
        raise ValueError("Checkpoint provenance differs")
    recipe = provenance["config"]["sft"]
    if (digest(plan) != recipe["plan_sha256"]
            or digest(json.loads((path.parent / "exposure_plan.json").read_text())) != digest(plan)):
        raise ValueError("Checkpoint exposure plan differs")
    if recipe["dropout"] != 0:
        raise ValueError("Checkpoint restore requires dropout == 0; CUDA RNG was not saved")
    updates, tokens = metadata["updates"], metadata["supervised_tokens"]
    if (type(updates) is not int or not 1 <= updates <= len(plan)
            or tokens != plan[updates - 1]["cumulative_supervised_tokens"]
            or path.name != f"checkpoint-tokens-{tokens}"):
        raise ValueError("Checkpoint counters differ from frozen plan")
    payload = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    if payload["updates"] != updates or payload["supervised_tokens"] != tokens:
        raise ValueError("Checkpoint counters differ from optimizer payload")

    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if len(model.peft_config) != 1:
        raise ValueError("Checkpoint restore requires exactly one LoRA adapter")
    adapter_name, = model.peft_config
    if model.peft_config[adapter_name].lora_dropout != 0:
        raise ValueError("Checkpoint model dropout must be zero")
    exported = get_peft_model_state_dict(model, state_dict=parameters,
                                        adapter_name=adapter_name, save_embedding_layers=False)
    names_by_id = {id(p): n for n, p in parameters.items()}
    if (len(exported) != len(parameters) or len({id(p) for p in exported.values()}) != len(parameters)
            or {id(p) for p in exported.values()} != set(names_by_id)):
        raise ValueError("PEFT saved keys do not map exactly once to every trainable")
    saved = load_file(path / "adapter_model.safetensors", device="cpu")
    if saved.keys() != exported.keys():
        raise ValueError("Checkpoint adapter has missing/extra keys")
    for key, parameter in exported.items():
        if saved[key].shape != parameter.shape or saved[key].dtype != parameter.dtype:
            raise ValueError("Checkpoint adapter shape/dtype mismatch")

    stored = payload["optimizer"]
    groups = stored["param_groups"]
    current = optimizer.param_groups
    ids = [i for group in groups for i in group["params"]]
    ordered_parameters = [p for group in current for p in group["params"]]
    if (len(groups) != len(current)
            or any(len(g["params"]) != len(c["params"]) for g, c in zip(groups, current))
            or len(ids) != len(parameters) or len(set(ids)) != len(ids)
            or {id(p) for p in ordered_parameters} != set(names_by_id)
            or list(map(id, ordered_parameters)) != list(map(id, parameters.values()))
            or set(stored["state"]) != set(ids)):
        raise ValueError("Checkpoint optimizer param count/order mismatch")
    warmup = recipe.get("moment_warmup_steps", 0)
    step = updates + warmup
    for group in groups:
        if (tuple(group["betas"]) != tuple(recipe["betas"]) or group["eps"] != recipe["eps"]
                or group["weight_decay"] != 0 or group.get("amsgrad", False)
                or group.get("maximize", False)):
            raise ValueError("Checkpoint optimizer recipe mismatch")
        if group["lr"] != learning_rate_at(recipe["learning_rate"], recipe.get("lr_warmup_steps", 0), updates):
            raise ValueError("Checkpoint optimizer learning rate mismatch")
    for index, parameter in zip(ids, ordered_parameters):
        entry = stored["state"][index]
        if (not isinstance(entry.get("step"), torch.Tensor) or entry["step"].numel() != 1
                or float(entry["step"]) != step):
            raise ValueError("Checkpoint optimizer step differs from updates + moment_warmup_steps")
        for field in ("exp_avg", "exp_avg_sq"):
            value = entry.get(field)
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype):
                raise ValueError("Checkpoint optimizer moment shape/dtype mismatch")

    loaded = set_peft_model_state_dict(model, saved, adapter_name=adapter_name)
    if loaded.unexpected_keys or set(loaded.missing_keys).intersection(parameters):
        raise ValueError("PEFT adapter load has missing/extra keys")
    for key, parameter in exported.items():
        if not bitwise_equal(parameter, saved[key]):
            raise AssertionError(f"Checkpoint trainable differs from file: {names_by_id[id(parameter)]}")
    optimizer.load_state_dict(deepcopy(stored))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.set_rng_state(payload["torch_rng"].cpu())
    return capture_state(model, optimizer, updates=updates, data_cursor=updates, step=step,
                         supervised_tokens=tokens, plan_sha256=digest(plan), identity_sha256=identity,
                         learning_rate=recipe["learning_rate"], lr_warmup_steps=recipe.get("lr_warmup_steps", 0),
                         gradient_clip=recipe["gradient_clip"], moment_warmup_steps=warmup,
                         moment_warmup_exposure=metadata.get("moment_warmup_exposure"),
                         checkpoint_rng_scope="torch CPU only; dropout == 0 required")
