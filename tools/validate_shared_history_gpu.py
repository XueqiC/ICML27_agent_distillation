"""Single-GPU acceptance gate for the real student's shared-history scorer.

Run with configs/hpg.env sourced and TRAIN_PYTHON, before production use:
  python -m tools.validate_shared_history_gpu --pairs STAGE/output --output REPORT.json

The CLI requires exactly one visible GPU and local bf16 weights. CPU entry points
exist solely for offline tiny-model tests; they cannot certify GPU acceptance.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from baselines.common.conversation import sequence_rows
from baselines.common.fsdp_train import batch_logps
from baselines.common.shared_history import (add_flex_kernel_options_argument, configure_flex_attention,
    flex_attention_settings, pack_shared_history, shared_history_logps, validate_model)
from sed.storage import digest, file_hash, write_json

MAX_LENGTH = 32768
SIDE_RTOL = 2e-3
GRAD_RTOL = 2e-2
SIDES = ("chosen", "rejected")
SCORINGS = ("per_turn", "shared_history")


def criteria():
    return {"side_sum_relative_difference_max": SIDE_RTOL,
            "gradient_relative_l2_per_tensor_max": GRAD_RTOL,
            "require_all_finite": True, "require_all_selected_gradients": True,
            "relative_denominator": "abs(per_turn) or L2(per_turn); zero/zero=0, nonzero/zero fails",
            "per_turn_differences": "reported diagnostics; side-sum threshold is the score gate"}


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
    return {"per_turn": reference if math.isfinite(reference) else None,
            "shared_history": actual if math.isfinite(actual) else None,
            "absolute_difference": error,
            "relative_difference": _relative(error, abs(reference)) if finite else None,
            "finite": finite}


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


def _scores(model, pair, device, scoring, *, per_turn=False):
    import torch
    model.eval()
    with torch.no_grad():
        values = {}
        for side in SIDES:
            row = pair[side]
            if not per_turn:
                values[side] = float(batch_logps(model, [row], device, scoring=scoring)[0])
            elif scoring == "shared_history":
                values[side] = shared_history_logps(model, [row], device, per_turn=True)[0].tolist()
            else:
                values[side] = [float(batch_logps(model, [turn], device, scoring="per_turn")[0])
                                for turn in sequence_rows(row)]
        return values


def _gradients(model, pair, parameters, device, scoring):
    """Derivative of chosen SUM - rejected SUM, freeing each turn/side graph.

    Accumulate bf16 unit gradients in fp32, avoiding bf16 rounding on every
    turn's addition. This is a linear sum, with no averaging or DPO weighting.
    No optimizer or model update is performed. Snapshots leave the GPU before
    the next path, so only one model and one path's activations are resident.
    """
    import torch
    model.train()
    accumulated, objective = {}, 0.0
    with torch.enable_grad():
        for side, sign in (("chosen", 1.0), ("rejected", -1.0)):
            units = [pair[side]] if scoring == "shared_history" else sequence_rows(pair[side])
            for unit in units:
                value = batch_logps(model, [unit], device, scoring=scoring)[0] * sign
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
                   "shared_history": len(pack_shared_history(pair[side])["input_ids"]),
                   "supervised_turns": len(sequence_rows(pair[side])),
                   "supervised_tokens": sum(sum(v != -100 for v in t["labels"])
                                            for t in sequence_rows(pair[side]))} for side in SIDES}


def compare_pair(model, pair, parameters, device, *, progress=None):
    """Shared CPU-test/GPU comparison code; CPU uses the dense SDPA test oracle."""
    import torch
    gpu = torch.device(device).type == "cuda"
    record = {"task_id": pair["task_id"], "forward_tokens_per_pass": forward_counts(pair), "paths": {}}
    snapshots = {}
    for scoring in SCORINGS:
        backend = "flex_attention" if gpu and scoring == "shared_history" else "sdpa"
        path = record["paths"][scoring] = {"attention": backend, "measurements": {}}
        if scoring == "shared_history":
            path["flex_attention"] = configure_flex_attention(model)
        if progress:
            progress(f"{scoring}/no_grad_sums", record)
        model.set_attn_implementation(backend)
        sums, sum_metrics = _measure(model, device, lambda: _scores(model, pair, device, scoring))
        path.update(side_sums=sums)
        path["measurements"]["no_grad_sums"] = sum_metrics
        if progress:
            progress(f"{scoring}/no_grad_per_turn", record)
        turns, turn_metrics = _measure(model, device, lambda: _scores(model, pair, device, scoring, per_turn=True))
        path.update(turn_logps=turns)
        path["measurements"]["no_grad_per_turn"] = turn_metrics
        if progress:
            progress(f"{scoring}/forward_backward", record)
        (objective, snapshots[scoring]), grad_metrics = _measure(
            model, device, lambda: _gradients(model, pair, parameters, device, scoring))
        path.update(gradient_objective=objective)
        path["measurements"]["forward_backward"] = grad_metrics
        expected = sum(record["forward_tokens_per_pass"][side][scoring] for side in SIDES)
        if any(m["forward_tokens"] != expected for m in (sum_metrics, turn_metrics, grad_metrics)):
            raise ValueError("Observed forward token counts differ from encoded layout")
    old, new = (record["paths"][scoring] for scoring in SCORINGS)
    record["side_sums"] = {side: scalar_difference(old["side_sums"][side], new["side_sums"][side]) for side in SIDES}
    record["turns"] = {}
    for side in SIDES:
        if len(old["turn_logps"][side]) != len(new["turn_logps"][side]):
            raise ValueError("Per-turn score counts differ")
        record["turns"][side] = [scalar_difference(a, b)
                                  for a, b in zip(old["turn_logps"][side], new["turn_logps"][side])]
    record["gradients"] = {name: gradient_difference(snapshots["per_turn"][name],
                                                   snapshots["shared_history"][name]) for name in parameters}
    record["gradient_objective_difference"] = scalar_difference(old["gradient_objective"], new["gradient_objective"])
    return record


def summarize(records):
    sides = [r["side_sums"][s] for r in records for s in SIDES]
    turns = [v for r in records for s in SIDES for v in r["turns"][s]]
    grads = [v for r in records for v in r["gradients"].values()]
    finite = bool(records and sides and turns and grads) and all(v["finite"] for v in [*sides, *turns, *grads,
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
        "max_gradient_relative_l2": grad_rel,
        "max_side_relative_difference_by_side": {side: maximum(
            [r["side_sums"][side] for r in records], "relative_difference") for side in SIDES},
        "max_gradient_relative_l2_by_tensor": {name: maximum(
            [r["gradients"][name] for r in records], "relative_l2") for name in (records[0]["gradients"] if records else {})},
        "passed": bool(finite and side_rel is not None and side_rel <= SIDE_RTOL
                       and grad_rel is not None and grad_rel <= GRAD_RTOL)}


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
    for side in SIDES:
        delta = record["side_sums"][side]
        print(f"pair={index} {side}: abs={delta['absolute_difference']} rel={delta['relative_difference']}", flush=True)
        for turn, delta in enumerate(record["turns"][side]):
            print(f"  turn={turn}: abs={delta['absolute_difference']} rel={delta['relative_difference']}", flush=True)
    print(f"pair={index} tokens/pass={record['forward_tokens_per_pass']}", flush=True)
    for scoring, path in record["paths"].items():
        print(f"pair={index} {scoring} measurements={json.dumps(path['measurements'])}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="ETO/IPR pairs stage output directory (contains pairs.json)")
    parser.add_argument("--output", required=True, help="JSON report, also written on numerical/runtime failure")
    parser.add_argument("--num-pairs", "-n", type=int, default=4)
    parser.add_argument("--snapshot", default=os.environ.get("MODEL_PATH"), help="Defaults to configs/hpg.env MODEL_PATH")
    parser.add_argument("--config", default="configs/base.yaml")
    add_flex_kernel_options_argument(parser)
    args = parser.parse_args(argv)
    print(f"PASS requires every bf16 side-sum relative difference <= {SIDE_RTOL:g}, every gradient tensor "
          f"relative L2 <= {GRAD_RTOL:g}, and all scores/gradients finite and present (no NaN/Inf).", flush=True)
    report = {"status": "running", "passed": False, "gpu_acceptance": False, "criteria": criteria(),
        "scope": "single-GPU real-student bf16 kernel acceptance; not FSDP/optimizer/scientific validation",
        "settings": {"num_pairs": args.num_pairs, "max_length": MAX_LENGTH, "packing": "turn",
                     "dtype": "bfloat16", "gradient_accumulation_dtype": "float32", "seed": 0,
                     "gradient_checkpointing": "non-reentrant", "gradient_objective": "chosen sum - rejected sum",
                     "training_updates": 0, "teacher_calls": 0},
        "measurement_notes": "One model; per-pair gradients; timings include masks, cold compilation and CPU gradient "
            "snapshots, no warmup. Peaks include resident weights and fp32 gradient accumulation. "
            "Root forward tokens exclude decoder checkpoint recomputation. Per-turn diagnostics are a separate pass.",
        "pairs": [], "phase": "setup", "inputs": {
            "pairs_directory": str(Path(args.pairs).resolve()), "snapshot": args.snapshot,
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
        report["phase"] = "checked_pair_loading_and_encoding"
        save_report(args.output, report)
        pairs, report["data"] = load_pairs(args.pairs, config, tokenizer, args.num_pairs)
        _, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
        report["data"]["unique_instances"] = len({p["task_id"] for p in pairs})
        report["data"]["task_groups"] = len({freeze["assignments"][p["task_id"]]["group_id"] for p in pairs})
        report["phase"] = "load_real_student"
        save_report(args.output, report)
        model = load_student(args.snapshot, device="cuda:0", dtype="bfloat16")
        configure_flex_attention(model, report["settings"]["flex_attention"]["kernel_options"])
        if any(p.device != torch.device("cuda:0") or p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError("All parameters must be bf16 on the single GPU")
        model.config.get_text_config().use_cache = False
        parameters = select_gradients(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        report["gradient_parameters"] = {name: list(p.shape) for name, p in parameters.items()}
        report["model_config"] = model.config.to_dict()
        for index, pair in enumerate(pairs):
            def progress(phase, record):
                report["phase"] = f"pair_{index}/{phase}"
                report["incomplete_pair"] = record
                save_report(args.output, report)

            print(f"Comparing pair {index + 1}/{len(pairs)}", flush=True)
            record = compare_pair(model, pair, parameters, "cuda:0", progress=progress)
            report["pairs"].append(record)
            report.pop("incomplete_pair", None)
            save_report(args.output, report)
            _print_pair(index, record)
        report["summary"] = summarize(report["pairs"])
        report["measurements_over_all_pairs"] = aggregate_measurements(report["pairs"])
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
