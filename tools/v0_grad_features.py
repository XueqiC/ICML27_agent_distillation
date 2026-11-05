"""Frozen-adapter V0 episode gradients, deterministic CountSketch and exact SNR."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]

import numpy as np

from repair import v2_train as v2
from repair.artifacts import provenance, tokenizer_identity
from repair.v2_inputs import ValidationError, require
from sed.storage import digest, file_hash, write_json
from tools.crn_tail import trainables, require_no_dropout
from tools.v0_data import BUCKETS, add_inputs, load_args, provenance_inputs, safe_output

DIMENSION = 2048
HASH_SEED = 17


def coordinate_hash(indices):
    """SplitMix64, modulo 2**64. Hash definition is identical on every device."""
    x = np.asarray(indices, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(HASH_SEED) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def count_sketch(values, *, offset=0, dimension=DIMENSION):
    """Unnormalised linear sketch. CPU float64 reductions avoid CUDA atomics."""
    require(type(dimension) is int and dimension > 0 and offset >= 0, "Invalid sketch shape/offset")
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    values = np.asarray(values).reshape(-1)
    require(np.isfinite(values).all(), "Nonfinite gradient")
    hashed = coordinate_hash(np.arange(offset, offset + values.size, dtype=np.uint64))
    buckets = (hashed % np.uint64(dimension)).astype(np.int64)
    signs = 1.0 - 2.0 * ((hashed >> np.uint64(63)).astype(np.float64))
    return np.bincount(buckets, weights=values.astype(np.float64) * signs, minlength=dimension)


class ModuleMoments:
    """Float32 device accumulators: sum g and sum ||g||², never a sketch."""
    def __init__(self, parameters, eps=1e-12):
        import torch
        require(parameters and eps > 0, "Missing parameters or invalid SNR epsilon")
        self.eps, self.count = eps, 0
        self.sums = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in parameters.items()}
        self.squares = {n: torch.zeros((), device=p.device, dtype=torch.float32) for n, p in parameters.items()}

    def add(self, gradients):
        import torch
        require(gradients.keys() == self.sums.keys(), "Gradient module mismatch")
        with torch.no_grad():
            for name, value in gradients.items():
                if value is None:
                    continue
                value = value.detach().float()
                require(value.shape == self.sums[name].shape and bool(value.isfinite().all()), "Invalid gradient")
                self.sums[name].add_(value)
                self.squares[name].add_(value.square().sum())
        self.count += 1

    def result(self):
        require(self.count > 0, "No episode gradients")
        result = {}
        for name in sorted(self.sums):
            mean_sq = float((self.sums[name] / self.count).square().sum())
            second = float(self.squares[name] / self.count)
            variance = max(second - mean_sq, 0.0)
            result[name] = {"episodes": self.count, "mean_norm": math.sqrt(mean_sq),
                "mean_squared_norm": second, "variance": variance,
                "snr": math.sqrt(mean_sq) / math.sqrt(max(variance, self.eps))}
        return result


def lora_parameters(model):
    parameters = dict(sorted(trainables(model).items()))
    require(parameters and all(any(part in ("lora_A", "lora_B") for part in name.split("."))
                               for name in parameters), "Expected only trainable LoRA A/B tensors")
    return parameters


def parameter_hash(parameters):
    result = hashlib.sha256()
    for name, parameter in sorted(parameters.items()):
        result.update(name.encode())
        result.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def disable_dropout(model):
    """Temporarily disable inherited V2 dropout while keeping checkpointing on."""
    import torch
    changes, seen = [], set()
    objects = list(model.modules()) + list(getattr(model, "peft_config", {}).values())
    config = getattr(model, "config", None)
    if config is not None:
        objects.append(config)
        if hasattr(config, "get_text_config"):
            objects.append(config.get_text_config())
    for obj in objects:
        names = ("p",) if isinstance(obj, torch.nn.modules.dropout._DropoutNd) else (
            "dropout", "attention_dropout", "hidden_dropout", "hidden_dropout_prob", "lora_dropout")
        for name in names:
            value = getattr(obj, name, None)
            if type(value) in (int, float) and value != 0 and (id(obj), name) not in seen:
                seen.add((id(obj), name))
                changes.append((obj, name, value))
                setattr(obj, name, 0.0)
    return changes


def episode_backward(model, examples, *, scoring="shared_history"):
    """V2 per-row token-mean SFT, then a mean over selected episode rows.

    crn_tail's surrogate_gradient is *negative*, micro-token-mean likelihood;
    message_backward also pools tokens. Neither equals V2's row-mean SFT for
    unequal lengths. Reuse their trainables/dropout guard, and V2's exact scorer.
    """
    import torch
    from baselines.common.fsdp_train import sft_objective
    require(examples, "Episode has no supervised rows")
    require_no_dropout(model)
    model.train()  # Keep the inherited non-reentrant gradient checkpointing.
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    loss_value = 0.0
    for example in examples:
        loss, _ = sft_objective(model, example, device, scoring=scoring)
        require(bool(torch.isfinite(loss)), "Nonfinite episode loss")
        loss_value += float(loss.detach()) / len(examples)
        (loss / len(examples)).backward()
    return loss_value


def select_episodes(rows, grouped, *, buckets=("R", "G"), max_episodes=None, allow_no_edit=False):
    """Filter before limiting episodes or touching the model; retain the inventory."""
    buckets = [b.strip() for b in buckets.split(",")] if isinstance(buckets, str) else list(buckets)
    require(buckets and all(b in BUCKETS for b in buckets) and len(set(buckets)) == len(buckets),
            "--buckets must be a nonempty comma-separated subset of R,G,S,D without duplicates")
    buckets = [b for b in BUCKETS if b in buckets]
    require(max_episodes is None or max_episodes > 0, "max-episodes must be positive")
    inventory = [{"episode_id": eid, "bucket": rows[grouped[eid][0]]["v0"]["bucket"]}
                 for eid in sorted(grouped)]
    eligible = [e["episode_id"] for e in inventory if e["bucket"] in buckets]
    ids = eligible[:max_episodes]
    require(ids, "No episodes in selected buckets")
    selected = {}
    for eid in ids:
        indices = grouped[eid]
        edits = [i for i in indices if rows[i]["v0"]["edit_decision"]]
        require(edits or rows[indices[0]]["v0"]["bucket"] not in ("R", "G") or allow_no_edit,
                f"Repair episode {eid} has no edit_decision row; use --allow-no-edit to allow whole-episode loss")
        selected[eid] = edits or indices
    return buckets, inventory, eligible, selected


def collect_features(model, examples, rows, grouped, output, *, metadata, max_episodes=None,
                     scoring="shared_history", buckets=("R", "G"), allow_no_edit=False):
    """No optimizer is created. Only selected episodes contribute gradients/moments."""
    import torch
    buckets, inventory, eligible, selections = select_episodes(rows, grouped, buckets=buckets,
        max_episodes=max_episodes, allow_no_edit=allow_no_edit)
    ids = list(selections)
    no_edit = [eid for eid in ids if not any(rows[i]["v0"]["edit_decision"] for i in grouped[eid])]
    parameters = lora_parameters(model)
    before = parameter_hash(parameters)
    moments = ModuleMoments(parameters)
    sketches, norms, entries = [], {}, []
    device = next(model.parameters()).device
    training = model.training
    dropout_changes = disable_dropout(model)
    started = time.monotonic()
    try:
        require_no_dropout(model)
        for number, eid in enumerate(ids, 1):
            indices = grouped[eid]
            selected = selections[eid]
            with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
                loss = episode_backward(model, [examples[i] for i in selected], scoring=scoring)
            gradients = {n: p.grad for n, p in parameters.items()}
            moments.add(gradients)
            sketch, raw_square, offset = np.zeros(DIMENSION, dtype=np.float64), 0.0, 0
            for name, parameter in parameters.items():
                grad = gradients[name]
                if grad is not None:
                    flat = grad.detach().float().cpu().numpy().reshape(-1)
                    # Fixed-size chunks bound host scratch without changing coordinate IDs.
                    for start in range(0, flat.size, 1 << 20):
                        chunk = flat[start:start + (1 << 20)]
                        sketch += count_sketch(chunk, offset=offset + start)
                        raw_square += float(np.dot(chunk.astype(np.float64), chunk.astype(np.float64)))
                offset += parameter.numel()
            norm = math.sqrt(raw_square)
            projected_norm = float(np.linalg.norm(sketch))
            zero = norm < 1e-12
            if zero:
                sketch.fill(0)
            elif projected_norm:
                sketch /= projected_norm
            norms[eid] = {"raw_l2_norm": norm, "ZERO_GRADIENT": zero,
                          "ZERO_SKETCH": not zero and projected_norm == 0}
            sketches.append(sketch.astype(np.float32))
            entries.append({"episode_id": eid, "bucket": rows[indices[0]]["v0"]["bucket"],
                "row_ids": [rows[i]["example_id"] for i in indices],
                "gradient_row_ids": [rows[i]["example_id"] for i in selected], "loss": loss})
            model.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - started
            print(json.dumps({"episodes": number, "total": len(ids), "episodes_per_second": number / elapsed}), flush=True)
        require(parameter_hash(parameters) == before, "Gradient extraction changed adapter parameters")
    finally:
        model.zero_grad(set_to_none=True)
        model.train(training)
        for obj, name, value in reversed(dropout_changes):
            setattr(obj, name, value)
    elapsed = time.monotonic() - started
    output = Path(output)
    np.save(output / "sketch.npy", np.stack(sketches), allow_pickle=False)
    write_json(output / "norms.json", norms, exclusive=True)
    write_json(output / "modules.json", {"schema": "v0-module-moments/1", **metadata,
        "bucket_filter": buckets, "episode_ids": ids,
        "eps": moments.eps, "accumulator_dtype": "float32", "modules": moments.result()}, exclusive=True)
    index = {"schema": "v0-gradient-features/1", **metadata, "episode_ids": ids, "episodes": entries,
        "bucket_filter": buckets, "dataset_episodes": inventory, "eligible_episodes": len(eligible),
        "allow_no_edit": allow_no_edit, "no_edit_count": len(no_edit), "no_edit_episode_ids": no_edit,
        "parameter_names": list(parameters), "parameter_names_sha256": digest(list(parameters)),
        "parameter_sizes": {n: p.numel() for n, p in parameters.items()}, "parameter_state_sha256": before,
        "dimension": DIMENSION, "seed": HASH_SEED, "hash": "splitmix64-coordinate/1",
        "normalization": "V2 row token mean, then selected episode row mean; unit sketch",
        "max_episodes": max_episodes, "available_episodes": len(grouped), "optimizer_steps": 0,
        "dropout_override": {"mode": "temporarily_zero", "settings_changed": len(dropout_changes)},
        "seconds": elapsed, "episodes_per_second": len(ids) / elapsed,
        "artifact_sha256": {n: file_hash(output / n) for n in ("sketch.npy", "norms.json", "modules.json")}}
    write_json(output / "index.json", index, exclusive=True)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--buckets", default="R,G", help="Comma-separated gradient buckets (default: R,G)")
    parser.add_argument("--allow-no-edit", action="store_true",
                        help="Allow and count R/G episodes without edits, using whole-episode loss")
    args = parser.parse_args(argv)
    try:
        import torch
        require(args.device.startswith("cuda:") and torch.device(args.device).type == "cuda",
                "Production extraction requires --device cuda:N")
        data = load_args(args)
        _, _, _, selections = select_episodes(data["rows"], data["episodes"], buckets=args.buckets,
            max_episodes=args.max_episodes, allow_no_edit=args.allow_no_edit)
        output = safe_output(args.output, data)
        inherited = v2.continuation_recipe(data["theta0"], data["config"], data["snapshot"], 1, seed_k=data["K"])
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        from train.dataset import load_tokenizer
        tokenizer = load_tokenizer(data["snapshot"])
        examples = {i: v2.encode_example(data["rows"][i], tokenizer, max_length=inherited["recipe"]["max_length"])
                    for indices in selections.values() for i in indices}
        model = v2.load_existing_adapter(data["snapshot"], data["theta0"], device, inherited["recipe"])
        require({n: p.numel() for n, p in lora_parameters(model).items()} == inherited["parameters"],
                "Actual trainable tensors differ from theta0")
        output.mkdir(parents=True)
        prov = provenance(output, data["config"], command="v0_grad_features",
            inputs=provenance_inputs(data, [data["inspection"]["files"].relocation.path(args.config)]),
            details={"arguments": vars(args), "inherited": inherited,
            "replay_deduplication": data["inspection"]["replay_deduplication"],
            "tokenizer": tokenizer_identity(data["snapshot"], tokenizer), "teacher_calls": 0})
        result = collect_features(model, examples, data["rows"], data["episodes"], output,
            max_episodes=args.max_episodes, buckets=args.buckets, allow_no_edit=args.allow_no_edit,
            scoring=inherited["recipe"]["scoring"], metadata={
            "adapter_sha256": data["adapter_sha256"], "dataset_sha256": data["dataset_sha256"],
            "provenance_sha256": digest(prov)})
        print(json.dumps({"status": "complete", "episodes": len(result["episode_ids"]),
                          "episodes_per_second": result["episodes_per_second"]}))
        return 0
    except ValidationError as exc:
        print(f"V0 features refused: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"V0 features failed ({type(exc).__name__}); no result marked complete.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
