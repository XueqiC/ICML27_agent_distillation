"""Single-GPU acceptance gate for the real student's shared-history scorer.

Run with configs/hpg.env sourced and TRAIN_PYTHON, before production use:
  python -m tools.validate_shared_history_gpu --pairs STAGE/output --output REPORT.json
  python -m tools.validate_shared_history_gpu --sad-prepared SAD_RUN --output REPORT.json

The CLI requires exactly one visible GPU and local bf16 weights. CPU entry points
exist solely for offline tiny-model tests; they cannot certify GPU acceptance.
"""
import argparse
from contextlib import nullcontext
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from baselines.common.conversation import sequence_rows
from baselines.common.fsdp_train import batch_logps, sequence_logps
from baselines.common.shared_history import (add_flex_kernel_options_argument, configure_flex_attention,
    flex_attention_settings, masked_logps, pack_shared_history, shared_history_logps, validate_model)
from sed.storage import digest, file_hash, write_json

MAX_LENGTH = 32768
SIDE_RTOL = 1e-3
GRAD_RTOL = 2e-2
TOKEN_ATOL = 0.05  # Project HF/vLLM gate's base mean-absolute tolerance (nats/token).
SIDES = ("chosen", "rejected")
SCORINGS = ("per_turn", "per_turn_flex", "shared_history")
COMPARISONS = {"kernel_only": ("per_turn", "per_turn_flex"),
               "packing_only": ("per_turn_flex", "shared_history"),
               "combined": ("per_turn", "shared_history")}


def criteria():
    return {"acceptance_comparison": "packing_only",
            "side_sum_relative_difference_max": SIDE_RTOL,
            "loss_relative_difference_max": SIDE_RTOL,
            "gradient_relative_l2_per_tensor_max": GRAD_RTOL,
            "require_all_finite": True, "require_all_selected_gradients": True,
            "relative_denominator": "abs(reference path) or L2(reference path); zero/zero=0, nonzero/zero fails",
            "per_turn_differences": "reported diagnostics; side-sum threshold is the score gate",
            "kernel_only_difference": "diagnostic; numerical magnitude does not gate acceptance",
            "token_mean_absolute_difference": "mean(abs(actual token logp - reference token logp)); diagnostic only",
            "hf_vllm_token_tolerance_nats": TOKEN_ATOL,
            "hf_vllm_token_tolerance_source": "tools/check_lora_serving.py: MEAN_TOLERANCE (base tolerance)"}


def load_pairs(directory, config, tokenizer, count):
    """Use the same evidence loaders and encoders as fsdp_train.prepare_data."""
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["command"] == "eto_pairs":
        from baselines.eto.pairs import load_pair_data
        from baselines.common.conversation import preference_conversation as encode_pair
        method = "eto"
    elif provenance["command"] == "ipr_pairs":
        from baselines.ipr.pairs import load_pair_data, preference_conversation as encode_pair
        method = "ipr"
    else:
        raise ValueError("Expected an ETO or IPR pairs stage output directory")
    pairs = load_pair_data(directory / "pairs.json", config)
    if not 1 <= count <= len(pairs):
        raise ValueError("Requested pair count must be positive and available; no silent undersampling")
    # Stable file order, with no length filtering, retries, or truncation.
    selected = pairs[:count]
    encoded = [encode_pair(pair, tokenizer, max_length=MAX_LENGTH) for pair in selected]
    return encoded, {"method": method, "directory": str(directory.resolve()),
        "available_pairs": len(pairs), "selected_indices": list(range(count)),
        "selection": "first N in verified stage order; no filtering",
        "encoder": f"{encode_pair.__module__}.{encode_pair.__name__}",
        "packing": "turn", "max_length": MAX_LENGTH, "no_truncation": True,
        "selected_pairs_sha256": digest(selected), "encoded_pairs_sha256": digest(encoded),
        "stage_artifact_sha256": {p: file_hash(directory / p)
                                  for p in ("pairs.json", "provenance.json", "report.json")}}


def load_sad_rows(directory, config, tokenizer, count):
    """Resolve a prepared SAD run (or its prepare output) to verified evidence.

    Preparation binds identity.json/tasks.json at the run root. Re-encode the
    original collection with the training encoder/curriculum; never trust an
    arbitrary serialized training row or regenerate teacher data.
    """
    from baselines.sad.data import curriculum, encode_episode, token_plan
    from sed.manifest import load_frozen
    from sed.seed_evidence import checked_episodes, load_verified_collection
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("SAD prepared directory does not exist")
    root = next((p for p in (directory, *directory.parents) if (p / "identity.json").is_file()
                 and (p / "tasks.json").is_file()), None)
    if root is None:
        raise ValueError("Expected prepared SAD identity.json and tasks.json at the run root")
    identity = json.loads((root / "identity.json").read_text())
    listing = json.loads((root / "tasks.json").read_text())
    if (identity["method"] != "sad" or identity["config_sha256"] != digest(config)
            or identity["tasks_sha256"] != digest(listing)):
        raise ValueError("Prepared SAD identity/config/tasks differ")
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, evidence = load_verified_collection(identity["verified"], manifest, freeze)
    if identity["collection_sha256"] != digest(collection):
        raise ValueError("Prepared SAD verified collection changed")
    episodes = list(checked_episodes(evidence, collection["verified"]))
    if (listing["task_ids"] != [e["task_id"] for e in episodes]
            or listing["parent_seed_ids"] != {e["task_id"]: [e["episode_id"]] for e in episodes}
            or listing["manifest_sha256"] != digest(manifest)
            or listing["split_freeze_sha256"] != digest(freeze)):
        raise ValueError("Prepared SAD task/parent/split binding changed")
    if not 1 <= count <= len(episodes):
        raise ValueError("Requested SAD row count must be positive and available; no silent undersampling")
    encoded = curriculum([encode_episode(e, tokenizer, max_length=MAX_LENGTH) for e in episodes])
    rows = encoded[:count]
    return rows, {"method": "sad", "directory": str(directory), "prepared_root": str(root),
        "verified": identity["verified"], "available_rows": len(encoded), "selected_indices": list(range(count)),
        "selection": "first N in stable SAD complexity curriculum; no filtering",
        "encoder": "baselines.sad.data.encode_episode", "packing": "turn", "max_length": MAX_LENGTH,
        "no_truncation": True, "K": identity["K"], "collection_sha256": digest(collection),
        "prepared_scoring": identity.get("recipe", {}).get("scoring", "per_turn"),
        "training_token_plan": token_plan(encoded),
        "encoded_rows_sha256": digest(rows), "encoded_curriculum_sha256": digest(encoded),
        "stage_artifact_sha256": {str(p): file_hash(p) for p in (
            root / "identity.json", root / "tasks.json", root / "provenance.json",
            Path(identity["verified"]) / "manifest.json", Path(identity["verified"]) / "identity.json")}}


def _sides(row):
    return next(((name,) for name in ("sad", "sft", "smartad") if name in row), SIDES)


def load_sft_rows(directory, config, tokenizer, count, objective):
    """Training encoders over checked train receipts; no selection or updates.

    SmartAD here tests Eq. 5 on a common verified collection; it does not claim
    the collection has undergone SmartAD candidate selection.
    """
    from baselines.common.conversation import conversation
    from baselines.smartad.data import encode_episode
    from sed.manifest import load_frozen
    from sed.seed_evidence import load_verified_collection, checked_episodes, artifact_hashes
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, root = load_verified_collection(directory, manifest, freeze)
    if not 0 < count <= len(collection["verified"]):
        raise ValueError("Request positive N within verified collection; no silent undersampling")
    episodes = list(checked_episodes(root, collection["verified"][:count]))
    if objective not in {"sft", "smartad"}:
        raise ValueError("Expected SFT or SmartAD objective")
    rows = [(encode_episode(e, tokenizer, max_length=MAX_LENGTH) if objective == "smartad" else
             conversation(e, tokenizer, source="trusted_teacher_seed", packing="turn", max_length=MAX_LENGTH))
            for e in episodes]
    return rows, {"method": objective, "directory": str(Path(directory).resolve()),
        "available_rows": len(collection["verified"]), "selected_indices": list(range(count)),
        "collection_sha256": digest(collection), "artifact_sha256": artifact_hashes(directory),
        "encoded_rows_sha256": digest(rows), "packing": "turn", "max_length": MAX_LENGTH,
        "selection": "first N verified receipts; diagnostic only, no candidate selection"}


def _sft_loss(model, row, device, scoring, objective, denominator):
    """One production microstep, or its packing-matched Flex kernel control."""
    from baselines.common.fsdp_train import sft_objective, supervised_count
    from baselines.smartad.data import objective as smartad_objective
    import torch
    if scoring != "per_turn_flex":
        if objective == "sft":
            loss, _ = sft_objective(model, row, device, scoring=scoring)
            return loss * supervised_count(row) / denominator
        loss, _ = smartad_objective(model, row, device, scoring=scoring)
        return loss / denominator
    if objective == "sft":
        return -per_turn_flex_logps(model, row, device) / denominator
    logps = per_turn_flex_logps(model, row, device, per_token=True)
    weights = torch.tensor([w for turn in sequence_rows(row)
                            for w, label in zip(turn["token_weights"], turn["labels"]) if label != -100], device=device)
    return -(logps * weights).sum() / denominator


def select_gradients(model):
    """Freeze embeddings/head/towers and every decoder layer except the last two."""
    validate_model(model)
    model.requires_grad_(False)
    decoder = model.model.language_model
    if len(decoder.layers) < 2:
        raise ValueError("At least two decoder layers required")
    for module in (*decoder.layers[-2:], decoder.norm):
        module.requires_grad_(True)
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def _relative(error, reference):
    if not math.isfinite(error) or not math.isfinite(reference):
        return None
    return error / reference if reference else (0.0 if error == 0 else None)


def scalar_difference(reference, actual):
    finite = math.isfinite(reference) and math.isfinite(actual)
    error = abs(actual - reference) if finite else None
    return {"reference": reference if math.isfinite(reference) else None,
            "actual": actual if math.isfinite(actual) else None,
            "signed_difference": actual - reference if finite else None,
            "absolute_difference": error,
            "relative_difference": _relative(error, abs(reference)) if finite else None,
            "finite": finite}


def token_difference(reference, actual):
    if not reference or len(reference) != len(actual):
        raise ValueError("Supervised token counts differ or are empty")
    finite = all(math.isfinite(v) for v in (*reference, *actual))
    total = math.fsum(abs(b - a) for a, b in zip(reference, actual)) if finite else None
    mean = total / len(reference) if finite else None
    return {"token_count": len(reference), "absolute_difference_sum": total,
            "mean_absolute_difference_nats_per_token": mean, "finite": finite,
            "hf_vllm_tolerance_nats_per_token": TOKEN_ATOL,
            "within_hf_vllm_tolerance": mean is not None and mean <= TOKEN_ATOL}


def gradient_difference(reference, actual):
    import torch
    if reference is None or actual is None:
        return {"finite": False, "missing_gradient": True, "relative_l2": None}
    if reference.shape != actual.shape:
        raise ValueError("Gradient tensor shapes differ")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    if not finite:
        return {"finite": False, "missing_gradient": False, "relative_l2": None}
    norm = float(torch.linalg.vector_norm(reference, dtype=torch.float64))
    error = float(torch.linalg.vector_norm(actual - reference, dtype=torch.float64))
    return {"finite": True, "missing_gradient": False, "reference_l2": norm,
            "absolute_l2": error, "relative_l2": _relative(error, norm)}


def _measure(model, device, operation):
    """Synchronized wall time and allocator peak, including mask/compile overhead."""
    import torch
    gpu = torch.device(device).type == "cuda"
    model.zero_grad(set_to_none=True)
    gc.collect()
    if gpu:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    counters = {"forward_tokens": 0, "forward_calls": 0}

    def count_forward(module, args, kwargs):
        counters["forward_tokens"] += kwargs["input_ids"].numel()
        counters["forward_calls"] += 1

    hook = model.register_forward_pre_hook(count_forward, with_kwargs=True)
    started = time.perf_counter()
    try:
        result = operation()
        if gpu:
            torch.cuda.synchronize(device)
        metrics = {**counters, "wall_seconds": time.perf_counter() - started,
                   "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if gpu else None}
        return result, metrics
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def per_turn_flex_logps(model, row, device, *, per_token=False):
    """One unchanged turn per forward; ordinary causal/sliding masks, no packing."""
    import torch
    values = []
    for turn in sequence_rows(row):
        n = len(turn["input_ids"])
        if (not 1 < n <= MAX_LENGTH or len(turn["labels"]) != n or turn["labels"][0] != -100
                or not any(v != -100 for v in turn["labels"])):
            raise ValueError("Invalid untruncated causal training example")
        layout = {"input_ids": turn["input_ids"], "labels": turn["labels"],
                  "position_ids": list(range(n)), "branches": [0] * n}
        values.append(masked_logps(model, [layout], device, per_token=per_token)[0])
    return torch.cat(values) if per_token else sum(values)


def _score(model, row, device, scoring):
    if scoring == "per_turn_flex":
        return per_turn_flex_logps(model, row, device)
    return batch_logps(model, [row], device, scoring=scoring)[0]


def _scores(model, pair, device, scoring, *, per_turn=False):
    import torch
    model.eval()
    with torch.no_grad():
        values, tokens = {}, {}
        for side in _sides(pair):
            row = pair[side]
            if not per_turn:
                values[side] = float(_score(model, row, device, scoring))
            else:
                if scoring == "shared_history":
                    token_values = shared_history_logps(model, [row], device, per_token=True)[0]
                elif scoring == "per_turn_flex":
                    token_values = per_turn_flex_logps(model, row, device, per_token=True)
                else:
                    token_values = torch.cat([sequence_logps(model, [turn], device, per_token=True)[0]
                                              for turn in sequence_rows(row)])
                counts = [sum(v != -100 for v in turn["labels"]) for turn in sequence_rows(row)]
                chunks = token_values.split(counts)
                values[side] = [float(chunk.sum()) for chunk in chunks]
                tokens[side] = [chunk.tolist() for chunk in chunks]
        return (values, tokens) if per_turn else values


def _gradients(model, pair, parameters, device, scoring):
    """Derivative of chosen SUM - rejected SUM, or SAD's negative span SUM.

    Accumulate bf16 unit gradients in fp32, avoiding bf16 rounding on every
    turn's addition. This is a linear sum, with no averaging or DPO weighting.
    No optimizer or model update is performed. Snapshots leave the GPU before
    the next path, so only one model and one path's activations are resident.
    """
    import torch
    model.train()
    accumulated, objective = {}, 0.0
    with torch.enable_grad():
        sides = _sides(pair)
        for side, sign in ([(sides[0], -1.0)] if len(sides) == 1 else [("chosen", 1.0), ("rejected", -1.0)]):
            units = [pair[side]] if scoring == "shared_history" else sequence_rows(pair[side])
            for unit in units:
                value = (_sft_loss(model, unit, device, scoring, side, pair["denominator"])
                         if side in {"sft", "smartad"} else _score(model, unit, device, scoring) * sign)
                objective += float(value.detach())
                value.backward()
                for name, parameter in parameters.items():
                    if parameter.grad is not None:
                        if name not in accumulated:
                            accumulated[name] = parameter.grad.detach().to(dtype=torch.float32, copy=True)
                        else:
                            accumulated[name].add_(parameter.grad.detach())
                    parameter.grad = None
                del value
    return objective, {name: accumulated[name].cpu() if name in accumulated else None for name in parameters}


def forward_counts(pair):
    return {side: {"per_turn": sum(len(t["input_ids"]) for t in sequence_rows(pair[side])),
                   "per_turn_flex": sum(len(t["input_ids"]) for t in sequence_rows(pair[side])),
                   "shared_history": len(pack_shared_history(pair[side])["input_ids"]),
                   "supervised_turns": len(sequence_rows(pair[side])),
                   "supervised_tokens": sum(sum(v != -100 for v in t["labels"])
                                            for t in sequence_rows(pair[side]))} for side in _sides(pair)}


def compare_pair(model, pair, parameters, device, *, progress=None):
    """Shared CPU-test/GPU comparison code; CPU uses the dense SDPA test oracle."""
    import torch
    gpu = torch.device(device).type == "cuda"
    sides = _sides(pair)
    record = {"task_id": pair["task_id"], "forward_tokens_per_pass": forward_counts(pair), "paths": {}}
    snapshots = {}
    for scoring in SCORINGS:
        backend = "flex_attention" if gpu and scoring != "per_turn" else "sdpa"
        path = record["paths"][scoring] = {"attention": backend, "measurements": {}}
        if scoring != "per_turn":
            path["flex_attention"] = configure_flex_attention(model)
        if progress:
            progress(f"{scoring}/no_grad_sums", record)
        model.set_attn_implementation(backend)
        sums, sum_metrics = _measure(model, device, lambda: _scores(model, pair, device, scoring))
        path.update(side_sums=sums)
        if sides == ("sad",):
            path["loss"] = -sums["sad"]
        path["measurements"]["no_grad_sums"] = sum_metrics
        if progress:
            progress(f"{scoring}/no_grad_per_turn", record)
        (turns, tokens), turn_metrics = _measure(model, device, lambda: _scores(model, pair, device, scoring, per_turn=True))
        path.update(turn_logps=turns, token_logps=tokens)
        path["measurements"]["no_grad_per_turn"] = turn_metrics
        if progress:
            progress(f"{scoring}/forward_backward", record)
        (objective, snapshots[scoring]), grad_metrics = _measure(
            model, device, lambda: _gradients(model, pair, parameters, device, scoring))
        path.update(gradient_objective=objective)
        if sides in (("sft",), ("smartad",)):
            path["loss"] = objective
        path["measurements"]["forward_backward"] = grad_metrics
        expected = sum(record["forward_tokens_per_pass"][side][scoring] for side in sides)
        if any(m["forward_tokens"] != expected for m in (sum_metrics, turn_metrics, grad_metrics)):
            raise ValueError("Observed forward token counts differ from encoded layout")
    record["comparisons"] = {}
    for name, (reference, actual) in COMPARISONS.items():
        old, new = (record["paths"][scoring] for scoring in (reference, actual))
        comparison = record["comparisons"][name] = {"reference_path": reference, "actual_path": actual,
            "side_sums": {side: scalar_difference(old["side_sums"][side], new["side_sums"][side]) for side in sides},
            "turns": {}, "tokens": {}}
        if len(sides) == 1:
            comparison["loss_difference"] = scalar_difference(old["loss"], new["loss"])
        for side in sides:
            if len(old["turn_logps"][side]) != len(new["turn_logps"][side]):
                raise ValueError("Per-turn score counts differ")
            comparison["turns"][side] = [
                {**scalar_difference(a, b), "tokens": token_difference(at, bt)}
                for a, b, at, bt in zip(old["turn_logps"][side], new["turn_logps"][side],
                                        old["token_logps"][side], new["token_logps"][side])]
            comparison["tokens"][side] = token_difference(
                [v for turn in old["token_logps"][side] for v in turn],
                [v for turn in new["token_logps"][side] for v in turn])
        comparison["gradients"] = {key: gradient_difference(snapshots[reference][key],
                                                           snapshots[actual][key]) for key in parameters}
        comparison["gradient_objective_difference"] = scalar_difference(old["gradient_objective"], new["gradient_objective"])
    return record


def compare_sad(model, row, parameters, device, *, progress=None):
    return compare_pair(model, {"task_id": row["label_provenance"]["task_id"], "sad": row},
                        parameters, device, progress=progress)


def compare_sft(model, row, parameters, device, *, objective, progress=None):
    from baselines.common.fsdp_train import supervised_count
    from baselines.smartad.data import weight_sum
    denominator = (supervised_count(row) if objective == "sft" else
                   sum(weight_sum(turn) for turn in sequence_rows(row)))
    return compare_pair(model, {"task_id": row["label_provenance"]["task_id"], objective: row,
                               "denominator": denominator}, parameters, device, progress=progress)


def _pooled_token_difference(rows):
    count = sum(row["token_count"] for row in rows)
    finite = bool(count) and all(row["finite"] for row in rows)
    mean = math.fsum(row["absolute_difference_sum"] for row in rows) / count if finite else None
    return {"token_count": count, "mean_absolute_difference_nats_per_token": mean, "finite": finite,
            "hf_vllm_tolerance_nats_per_token": TOKEN_ATOL,
            "within_hf_vllm_tolerance": mean is not None and mean <= TOKEN_ATOL}


def _summarize_comparison(records):
    side_names = tuple(records[0]["side_sums"]) if records else SIDES
    sides = [r["side_sums"][s] for r in records for s in side_names]
    turns = [v for r in records for s in side_names for v in r["turns"][s]]
    grads = [v for r in records for v in r["gradients"].values()]
    tokens = [r["tokens"][s] for r in records for s in side_names]
    finite = bool(records and sides and turns and grads) and all(v["finite"] for v in [*sides, *turns, *grads, *tokens,
        *[r["gradient_objective_difference"] for r in records]])

    def maximum(rows, key):
        values = [r[key] for r in rows]
        return max(values) if values and all(v is not None for v in values) else None

    side_rel, grad_rel = maximum(sides, "relative_difference"), maximum(grads, "relative_l2")
    return {"all_finite_and_present": finite,
        "max_side_absolute_difference": maximum(sides, "absolute_difference"),
        "max_side_relative_difference": side_rel,
        "max_turn_absolute_difference": maximum(turns, "absolute_difference"),
        "max_turn_relative_difference": maximum(turns, "relative_difference"),
        "max_loss_absolute_difference": maximum(
            [r["loss_difference"] for r in records if "loss_difference" in r], "absolute_difference"),
        "max_loss_relative_difference": maximum(
            [r["loss_difference"] for r in records if "loss_difference" in r], "relative_difference"),
        "max_gradient_relative_l2": grad_rel,
        "max_side_relative_difference_by_side": {side: maximum(
            [r["side_sums"][side] for r in records], "relative_difference") for side in side_names},
        "max_gradient_relative_l2_by_tensor": {name: maximum(
            [r["gradients"][name] for r in records], "relative_l2") for name in (records[0]["gradients"] if records else {})},
        "tokens": _pooled_token_difference(tokens),
        "tokens_by_side": {side: _pooled_token_difference([r["tokens"][side] for r in records]) for side in side_names}}


def summarize(records):
    comparisons = {name: {"reference_path": reference, "actual_path": actual,
                         **_summarize_comparison([r["comparisons"][name] for r in records])}
                   for name, (reference, actual) in COMPARISONS.items()}
    packing = comparisons["packing_only"]
    finite = all(c["all_finite_and_present"] for c in comparisons.values())
    side_rel, grad_rel = packing["max_side_relative_difference"], packing["max_gradient_relative_l2"]
    loss_rel = packing["max_loss_relative_difference"]
    loss_ok = not any("loss_difference" in r["comparisons"]["packing_only"] for r in records) or (
        loss_rel is not None and loss_rel <= SIDE_RTOL)
    return {"acceptance_comparison": "packing_only", "all_finite_and_present": finite,
            "comparisons": comparisons,
            "passed": bool(finite and side_rel is not None and side_rel <= SIDE_RTOL
                           and loss_ok and grad_rel is not None and grad_rel <= GRAD_RTOL)}


def _json_safe(value):
    # A failed kernel may yield NaN/Inf. Preserve failure via finite=false/null,
    # never emit invalid JSON NaN literals or hide the failing pair.
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def save_report(path, report):
    write_json(path, _json_safe(report))


def aggregate_measurements(records):
    result = {}
    for scoring in SCORINGS:
        result[scoring] = {}
        for phase in ("no_grad_sums", "no_grad_per_turn", "forward_backward"):
            rows = [r["paths"][scoring]["measurements"][phase] for r in records]
            peaks = [r["max_memory_allocated_bytes"] for r in rows]
            result[scoring][phase] = {
                **{key: sum(r[key] for r in rows) for key in ("wall_seconds", "forward_tokens", "forward_calls")},
                "max_memory_allocated_bytes": max(peaks) if peaks and all(p is not None for p in peaks) else None}
    return result


def _print_pair(index, record):
    unit = "row" if len(record["forward_tokens_per_pass"]) == 1 else "pair"
    for name, comparison in record["comparisons"].items():
        print(f"{unit}={index} {name}: {comparison['reference_path']} vs {comparison['actual_path']}", flush=True)
        if "loss_difference" in comparison:
            print(f"  loss: {comparison['loss_difference']}", flush=True)
        for side in comparison["side_sums"]:
            delta = comparison["side_sums"][side]
            tokens = comparison["tokens"][side]
            print(f"  {side}: signed={delta['signed_difference']} abs={delta['absolute_difference']} "
                  f"rel={delta['relative_difference']} token_MAE={tokens['mean_absolute_difference_nats_per_token']} "
                  f"nats/token (HF/vLLM base tolerance={TOKEN_ATOL:g}, within={tokens['within_hf_vllm_tolerance']})", flush=True)
            for turn, delta in enumerate(comparison["turns"][side]):
                print(f"    turn={turn}: signed={delta['signed_difference']} abs={delta['absolute_difference']} "
                      f"rel={delta['relative_difference']} "
                      f"token_MAE={delta['tokens']['mean_absolute_difference_nats_per_token']}", flush=True)
    print(f"{unit}={index} tokens/pass={record['forward_tokens_per_pass']}", flush=True)
    for scoring, path in record["paths"].items():
        print(f"{unit}={index} {scoring} measurements={json.dumps(path['measurements'])}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", help="ETO/IPR pairs stage output directory (contains pairs.json)")
    source.add_argument("--sad-prepared", help="Prepared SAD run root, or its stages/prepare output directory")
    source.add_argument("--verified", help="Verified collection for SFT/SmartAD numerical acceptance")
    parser.add_argument("--objective", choices=["sft", "smartad"], default="sft",
                        help="With --verified: score the production Kang LoRA objective, dropout disabled explicitly for exactness")
    parser.add_argument("--output", required=True, help="JSON report, also written on numerical/runtime failure")
    parser.add_argument("--num-pairs", "--num-rows", "-n", type=int, default=4)
    parser.add_argument("--snapshot", default=os.environ.get("MODEL_PATH"), help="Defaults to configs/hpg.env MODEL_PATH")
    parser.add_argument("--config", default="configs/base.yaml")
    add_flex_kernel_options_argument(parser)
    args = parser.parse_args(argv)
    records_key = "rows" if args.sad_prepared or args.verified else "pairs"
    unit = "row" if records_key == "rows" else "pair"
    print(f"PASS requires packing-only (per_turn_flex vs shared_history) bf16 side-sum relative difference "
          f"<= {SIDE_RTOL:g}, every gradient tensor relative L2 <= {GRAD_RTOL:g}, and all paths' scores/gradients "
          f"finite and present (no NaN/Inf). Kernel-only differences and token MAE (HF/vLLM base tolerance "
          f"{TOKEN_ATOL:g} nats/token) are diagnostics.", flush=True)
    report = {"status": "running", "passed": False, "gpu_acceptance": False, "criteria": criteria(),
        "scope": "single-GPU real-student bf16 packing acceptance with a matched Flex control; not FSDP/optimizer/scientific validation",
        "settings": {f"num_{records_key}": args.num_pairs, "max_length": MAX_LENGTH, "packing": "turn",
                     "dtype": "bfloat16", "gradient_accumulation_dtype": "float32", "seed": 0,
                     "gradient_checkpointing": "non-reentrant",
                     "objective": args.objective if args.verified else "sad" if args.sad_prepared else "preference",
                     "gradient_objective": ("trajectory token mean" if args.objective == "sft" else "segment weighted token mean")
                         if args.verified else "negative SAD span-token sum" if args.sad_prepared else "chosen sum - rejected sum",
                     **({"adaptation": "kang_lora", "lora_dropout": 0.0,
                         "dropout_scope": "deterministic numerical diagnostic; original training recipe uses 0.05"} if args.verified else {}),
                     "training_updates": 0, "teacher_calls": 0},
        "measurement_notes": "One model; gradients per example; timings include masks, cold compilation and CPU gradient "
            "snapshots, no warmup. Peaks include resident weights and fp32 gradient accumulation. "
            "Root forward tokens exclude decoder checkpoint recomputation. Per-turn/token diagnostics share a separate pass. "
            "CPU tests use dense SDPA mask oracles for both Flex layouts; only CUDA executes the matched Flex kernels.",
        records_key: [], "phase": "setup", "inputs": {
            ("verified_directory" if args.verified else "sad_prepared_directory" if args.sad_prepared else "pairs_directory"):
                str(Path(args.verified or args.sad_prepared or args.pairs).resolve()), "snapshot": args.snapshot,
            "config": args.config, "output": args.output}}
    save_report(args.output, report)
    try:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        import torch
        from sed.config import load_config
        from sed.manifest import load_frozen
        from sed.provenance import code_hashes, utc_now, versions
        from train.dataset import load_tokenizer
        from train.modeling import load_student, snapshot_metadata
        report["settings"]["flex_attention"] = flex_attention_settings(args.flex_kernel_options)
        if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1
                or not torch.cuda.is_available()):
            raise ValueError("Exactly one visible GPU and one process required")
        torch.cuda.set_device(0)
        if not torch.cuda.is_bf16_supported():
            raise ValueError("GPU must support bf16")
        if not args.snapshot or args.num_pairs < 1:
            raise ValueError("Set MODEL_PATH (or --snapshot) and request positive N")
        torch.manual_seed(0)
        config = load_config(args.config)
        if (config["training"]["thinking_mode"] != "off" or config["training"]["student_step_token_cap"] != 1024
                or config["environment"]["max_episode_steps"] != 40):
            raise ValueError("Baseline harness protocol differs")
        metadata = snapshot_metadata(args.snapshot)
        tokenizer = load_tokenizer(args.snapshot)
        report["provenance"] = {"timestamp": utc_now(), "python": sys.version.split()[0],
            "python_executable": sys.executable, "dependencies": versions(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "gpu_count": 1,
            "code_sha256": code_hashes(Path(__file__).resolve().parents[1]),
            "input_sha256": {str(p): file_hash(p) for p in
                (args.config, config["paths"]["manifest"], config["paths"]["split_freeze"])},
            "student": config["models"]["student"], "snapshot": metadata,
            "flex_attention": report["settings"]["flex_attention"],
            "template_sha256": digest(tokenizer.chat_template)}
        report["phase"] = f"checked_{unit}_loading_and_encoding"
        save_report(args.output, report)
        pairs, report["data"] = (load_sft_rows(args.verified, config, tokenizer, args.num_pairs, args.objective)
                                 if args.verified else load_sad_rows(args.sad_prepared, config, tokenizer, args.num_pairs)
                                 if args.sad_prepared else load_pairs(args.pairs, config, tokenizer, args.num_pairs))
        _, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
        task_ids = [p["label_provenance"]["task_id"] if records_key == "rows" else p["task_id"] for p in pairs]
        report["data"]["unique_instances"] = len(set(task_ids))
        report["data"]["task_groups"] = len({freeze["assignments"][t]["group_id"] for t in task_ids})
        report["phase"] = "load_real_student"
        save_report(args.output, report)
        model = load_student(args.snapshot, device="cuda:0", dtype="bfloat16")
        configure_flex_attention(model, report["settings"]["flex_attention"]["kernel_options"])
        if any(p.device != torch.device("cuda:0") or p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError("All parameters must be bf16 on the single GPU")
        model.config.get_text_config().use_cache = False
        if args.verified:
            from train.modeling import attach_lora
            model = attach_lora(model, rank=64, alpha=128, dropout=0.0)
            parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        else:
            parameters = select_gradients(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        report["gradient_parameters"] = {name: list(p.shape) for name, p in parameters.items()}
        report["model_config"] = model.config.to_dict()
        for index, pair in enumerate(pairs):
            def progress(phase, record):
                report["phase"] = f"{unit}_{index}/{phase}"
                report[f"incomplete_{unit}"] = record
                save_report(args.output, report)

            print(f"Comparing {unit} {index + 1}/{len(pairs)}", flush=True)
            compare = compare_sft if args.verified else compare_sad if args.sad_prepared else compare_pair
            # Production LoRA uses bf16 autocast with fp32 adapter masters.
            with torch.autocast("cuda", dtype=torch.bfloat16) if args.verified else nullcontext():
                record = compare(model, pair, parameters, "cuda:0", progress=progress,
                                 **({"objective": args.objective} if args.verified else {}))
            report[records_key].append(record)
            report.pop(f"incomplete_{unit}", None)
            save_report(args.output, report)
            _print_pair(index, record)
        report["summary"] = summarize(report[records_key])
        report[f"measurements_over_all_{records_key}"] = aggregate_measurements(report[records_key])
        report["passed"] = report["gpu_acceptance"] = report["summary"]["passed"]
        report["status"] = "pass" if report["passed"] else "fail"
        report["phase"] = "complete"
        print(json.dumps(report["summary"], allow_nan=False), flush=True)
    except Exception as error:
        # This offline GPU diagnostic needs the full compiler/resource error.
        report["status"] = "error"
        report["error_type"] = type(error).__name__
        report["error_message"] = str(error)
        report["error_repr"] = repr(error)
        report["error_frames"] = [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                                  for frame in traceback.extract_tb(error.__traceback__)]
        print(f"FAIL during {report['phase']} ({type(error).__name__}); see JSON report.", flush=True)
    save_report(args.output, report)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: GPU acceptance; report={args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
