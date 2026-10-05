"""Full text-parameter SFT/DPO, torchrun FSDP2, exact global example batches.

Production is GPU-only. train_loop/objective functions also support tiny CPU
models for offline verification; no alternate foundation model is selected.
"""
import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from baselines.common.conversation import conversation, preference_conversation
from baselines.common.schedule import dose, epoch_batches, lr_multiplier, rank_slots
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.seed_evidence import checked_episode, load_verified_collection, artifact_hashes
from sed.storage import digest, file_hash, write_json, JsonlLedger, fsync_dir
from train.dataset import load_tokenizer
from train.modeling import load_student, snapshot_metadata


def text_parameter(name):
    return name.startswith("model.language_model.") or name.startswith("lm_head.")


def validate_checkpoint(path, student=None):
    path = Path(path)
    state = json.loads((path / "training_state.json").read_text())
    hashes = artifact_hashes(path)
    hashes.pop("training_state.json")
    provenance = json.loads((path.parent / "provenance.json").read_text())
    if (not state.get("final") or state.get("text_export_dtype") != "bfloat16"
            or hashes != state["artifact_sha256"] or digest(provenance) != state["provenance_sha256"]):
        raise ValueError("Full checkpoint artifacts/provenance changed")
    if student is not None and provenance["config"]["models"]["student"] != student:
        raise ValueError("Full checkpoint differs from configured student identity")
    snapshot_metadata(path)
    return state


def prepare_model(snapshot, *, reference=False):
    """Same class/loader as Seed-SFT, fp32 master weights, SDPA (X6/X7)."""
    model = load_student(snapshot, device="cpu", dtype="float32")
    model.set_attn_implementation("sdpa")
    model.config.get_text_config().use_cache = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not reference and text_parameter(name))
    if reference:
        model.eval()
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def shard_model(model, *, mesh):
    import torch
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextDecoderLayer
    layers = [m for m in model.modules() if isinstance(m, Gemma4UnifiedTextDecoderLayer)]
    if len(layers) != model.config.text_config.num_hidden_layers:
        raise ValueError("Gemma4UnifiedTextDecoderLayer wrap count differs from config")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("FSDP requires fp32-loaded master weights")
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    return model


def batch_logps(model, rows, device):
    """Exact summed assistant log-probs, materializing only target predecessors.

    Selected logits are already causally aligned. Calling the verbatim scorer
    with is_encoder_decoder=True skips its shift, not its log-softmax/mask/sum.
    No sequence or assistant token is truncated. Pair is one example, batch 2.
    """
    import torch
    from baselines.eto.official_loss import OfficialLoss
    length = max(len(r["input_ids"]) for r in rows)
    ids = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        if n > 32768 or n != len(row["labels"]) or row["labels"][0] != -100:
            raise ValueError("Invalid untruncated causal training example")
        ids[i, :n] = torch.tensor(row["input_ids"], device=device)
        labels[i, :n] = torch.tensor(row["labels"], device=device)
        attention[i, :n] = 1
        if not any(v != -100 for v in row["labels"]):
            raise ValueError("No supervised tokens")
    positions = torch.nonzero((labels[:, 1:] != -100).any(dim=0)).flatten()
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False,
                   logits_to_keep=positions).logits
    target = labels[:, positions + 1]
    # Small log-softmax chunks avoid a second full fp32 vocabulary tensor.
    values = [OfficialLoss.get_batch_logps(logits[:, start:start + 128].float(),
                       target[:, start:start + 128].clone(), average_log_prob=False,
                       is_encoder_decoder=True)
              for start in range(0, len(positions), 128)]
    return sum(values)


def sft_objective(model, row, device):
    count = sum(v != -100 for v in row["labels"])
    return -batch_logps(model, [row], device).mean() / count, {}


def dpo_objective(model, reference, pair, device, *, beta):
    import torch
    from baselines.eto.official_loss import OfficialLoss
    if reference is None or reference is model or any(p.requires_grad for p in reference.parameters()):
        raise ValueError("DPO requires a distinct frozen iteration-start reference")
    reference.eval()
    rows = [pair["chosen"], pair["rejected"]]
    with torch.no_grad():
        ref = batch_logps(reference, rows, device)
    policy = batch_logps(model, rows, device)
    settings = SimpleNamespace(beta=beta, loss_type="sigmoid", label_smoothing=0.0)
    losses, chosen, rejected = OfficialLoss.dpo_loss(settings, policy[:1], policy[1:], ref[:1], ref[1:])
    metrics = {"rewards/chosen": chosen.mean(), "rewards/rejected": rejected.mean(),
               "rewards/accuracies": (chosen > rejected).float().mean(),
               "rewards/margins": (chosen - rejected).mean(),
               "logps/chosen": policy[0].detach(), "logps/rejected": policy[1].detach()}
    return losses.mean(), metrics


def train_loop(model, examples, output, *, objective, learning_rate, scheduler,
               warmup_ratio, original_batch, epochs=3, seed=0, beta=0.1,
               reference=None, weight_decay=0.0, rank=0, world_size=1, device="cpu"):
    import torch
    import torch.distributed as dist
    if objective not in {"sft", "dpo"} or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid training objective/learning rate")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("Positive finite beta required")
    for example in examples:
        for row in ([example] if objective == "sft" else [example["chosen"], example["rejected"]]):
            if row.get("label_provenance", {}).get("training_eligible") is False:
                raise ValueError("Diagnostic examples cannot enter training")
    plan = dose(len(examples), original_batch, epochs=epochs)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=weight_decay, foreach=False, fused=False)
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_multiplier,
                  scheduler=scheduler, total_steps=plan["total_steps"], warmup_ratio=warmup_ratio))
    model.train()
    if objective == "dpo":
        # Official DPOMultiTrainer disables dropout in both policy/reference.
        for candidate in (model, reference):
            for module in candidate.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = JsonlLedger(output / "updates.jsonl") if rank == 0 else None
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    started, exposure, supervised, physical, physical_tokens, forward_tokens = time.monotonic(), 0, 0, 0, 0, 0
    for update, batch in enumerate(epoch_batches(len(examples), plan["b_prime"], epochs=epochs, seed=seed), 1):
        optimizer.zero_grad(set_to_none=True)
        metrics_sum = {}
        indices = batch["indices"]
        slots = rank_slots(indices, world_size)
        for slot in slots:
            index = slot[rank]
            active = index is not None
            row = examples[index if active else indices[0]]
            loss, metrics = (sft_objective(model, row, device) if objective == "sft" else
                            dpo_objective(model, reference, row, device, beta=beta))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            # FSDP averages across ranks. This cancels that average and divides
            # by ACTUAL examples, including the final partial batch. Synchronize
            # each microstep to keep gradients sharded during accumulation.
            (loss * (world_size / len(indices) if active else 0.0)).backward()
            for key, value in {"loss": loss.detach(), **metrics}.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value.detach() * int(active)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        lr_schedule.step()
        # Same global accounting on every rank; padding is extra compute only.
        exposure += len(indices)
        def costs(index):
            row = examples[index]
            rows = [row] if objective == "sft" else [row["chosen"], row["rejected"]]
            return (sum(sum(v != -100 for v in r["labels"]) for r in rows),
                    sum(len(r["input_ids"]) for r in rows), len(rows) * max(len(r["input_ids"]) for r in rows))
        supervised += sum(costs(i)[0] for i in indices)
        forward_tokens += sum(costs(i)[1] for i in indices)
        physical += len(slots) * world_size
        physical_tokens += sum(costs(i if i is not None else indices[0])[2] for slot in slots for i in slot)
        keys = sorted(metrics_sum)
        values = torch.stack([metrics_sum[k] for k in keys])
        if world_size > 1:
            dist.all_reduce(values)
        values /= len(indices)
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        if ledger:
            ledger.append({"update": update, **batch, "lr": lr_used, "global_batch": len(indices),
                           "collective_rounds": len(slots), "zero_weight_slots": len(slots) * world_size - len(indices),
                           "gradient_norm": float(norm), "examples_seen": exposure,
                           "supervised_tokens_seen": supervised,
                           **dict(zip(keys, values.cpu().tolist()))}, unique_key="update")
    if cuda:
        torch.cuda.synchronize(device)
    memory = {"rank": rank, "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None}
    all_memory = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(all_memory, memory)
    else:
        all_memory = [memory]
    return {**plan, "updates": update, "seed": seed, "examples_seen": exposure,
            "supervised_tokens_seen": supervised, "logical_forward_tokens": forward_tokens,
            "physical_example_slots": physical, "zero_weight_slots": physical - exposure,
            "physical_policy_forward_tokens": physical_tokens,
            "physical_reference_forward_tokens": physical_tokens if objective == "dpo" else 0,
            "wall_time_seconds": time.monotonic() - started, "memory_per_rank": all_memory,
            "microbatch": 1, "microbatch_unit": "episode" if objective == "sft" else "preference pair",
            "master_dtype": "float32", "compute_dtype": "bfloat16" if cuda else "float32",
            "gradient_clip": 1.0, "weight_decay": weight_decay,
            "loss_reduction": "per_example_mean_then_global_batch_mean" if objective == "sft" else "sigmoid_summed_logps"}


def export_checkpoint(model, destination, base_snapshot, *, rank=0, max_shard_bytes=2_000_000_000):
    """Gather one tensor at a time, bf16 text weights + unchanged base towers.

    Every rank participates in DTensor gathers. Rank zero writes bounded CPU
    safetensors shards, a complete HF index/config and base processor assets.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    destination, base = Path(destination), Path(base_snapshot)
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=False)
    base_map = {}
    for path in sorted(base.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            base_map.update({key: path for key in handle.keys()})
    pending, size, total, mapping, tower_hashes = {}, 0, 0, {}, {}
    def flush():
        nonlocal pending, size
        if not pending:
            return
        name = f"model-{len(set(mapping.values())) + 1:05d}.safetensors"
        save_file(pending, destination / name, metadata={"format": "pt"})
        mapping.update({key: name for key in pending})
        pending, size = {}, 0
    for name, value in model.state_dict().items():
        # Frozen multimodal weights come from the ORIGINAL base, not a partial
        # text-only export or random initialization; no need to gather them.
        if not text_parameter(name):
            if name not in base_map:
                raise ValueError("Frozen checkpoint tensor missing from original base snapshot")
            if rank == 0:
                with safe_open(base_map[name], framework="pt", device="cpu") as handle:
                    tensor = handle.get_tensor(name).contiguous().clone()
                import hashlib
                tower_hashes[name] = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
        else:
            full = value.full_tensor() if hasattr(value, "full_tensor") else value.detach()
            if rank == 0:
                tensor = full.to(device="cpu", dtype=torch.bfloat16).contiguous().clone()
            del full
        if rank == 0:
            nbytes = tensor.numel() * tensor.element_size()
            if size + nbytes > max_shard_bytes:
                flush()
            pending[name] = tensor
            size += nbytes
            total += nbytes
    if rank != 0:
        return None
    flush()
    write_json(destination / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": mapping})
    # Preserve tokenizer/template and multimodal processor assets from the base.
    for path in base.iterdir():
        if path.is_file() and (path.suffix in {".json", ".jinja", ".jinja2", ".model", ".txt"}) and not (
                path.name in {"config.json", "generation_config.json"} or ".index." in path.name):
            shutil.copyfile(path, destination / path.name)
    model.config.get_text_config().use_cache = True
    model.config.dtype = torch.bfloat16
    model.config.get_text_config().dtype = torch.bfloat16
    model.config.save_pretrained(destination)
    if model.generation_config is not None:
        model.generation_config.save_pretrained(destination)
    return {"non_text_tensor_sha256": tower_hashes, "non_text_source": str(base.resolve()),
            "artifact_sha256": artifact_hashes(destination), "text_export_dtype": "bfloat16"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--objective", choices=["sft", "dpo"], required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--base-snapshot", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--verified")
    parser.add_argument("--pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--scheduler", choices=["cosine", "constant_with_warmup"], required=True)
    parser.add_argument("--warmup-ratio", type=float, required=True)
    parser.add_argument("--original-batch", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args(argv)
    preparation_started = time.monotonic()
    if args.max_length != 32768:
        parser.error("Registered no-truncation limit is 32768")
    if args.objective == "dpo" and (not args.reference or Path(args.reference).resolve() != Path(args.snapshot).resolve()):
        parser.error("Reference must be the iteration-start policy snapshot")
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("Production FSDP training needs an allocated GPU; CPU objectives are tested offline")
    local_rank, rank, world = (int(os.environ[k]) for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        set_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        config = load_config(args.config)
        if (config["training"]["thinking_mode"] != "off" or config["training"]["student_step_token_cap"] != 1024
                or config["environment"]["max_episode_steps"] != 40):
            raise ValueError("Baseline harness protocol differs")
        config["training"]["adaptation"] = "full_parameter_text_fsdp2"
        if args.objective == "dpo":
            validate_checkpoint(args.snapshot, config["models"]["student"])
        manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
        tokenizer = load_tokenizer(args.snapshot)
        inputs = [args.config, config["paths"]["manifest"], config["paths"]["split_freeze"]]
        if args.objective == "sft":
            collection, selection, root = load_verified_collection(args.verified, manifest, freeze)
            examples = [conversation(checked_episode(root, receipt), tokenizer, source="trusted_teacher_seed")
                        for receipt in collection["verified"]]
            inputs += [Path(args.verified) / "manifest.json", Path(args.verified) / "identity.json"]
        else:
            from baselines.eto.pairs import load_pair_data
            pairs = load_pair_data(args.pairs, config)
            examples = [preference_conversation(pair, tokenizer) for pair in pairs]
            inputs += [args.pairs, Path(args.pairs).parent / "provenance.json"]
        plan = dose(len(examples), args.original_batch, epochs=args.epochs)
        output = Path(args.output)
        provenance = None
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            config["baseline_training"] = {**vars(args), **plan, "world_size": world,
                "snapshot": snapshot_metadata(args.snapshot), "base_snapshot": snapshot_metadata(args.base_snapshot),
                "snapshot_artifact_sha256": artifact_hashes(args.snapshot),
                "base_artifact_sha256": artifact_hashes(args.base_snapshot),
                "examples_sha256": digest(examples), "template_sha256": digest(tokenizer.chat_template),
                "attention": "sdpa", "master_dtype": "float32", "compute_dtype": "bfloat16",
                "wrap_class": "Gemma4UnifiedTextDecoderLayer", "gradient_checkpointing": True,
                "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
                "no_truncation": True, "packing": "episode", "microbatch": 1,
                "warmup_steps": math.ceil(plan["total_steps"] * args.warmup_ratio),
                "reference_policy": "iteration_start" if args.objective == "dpo" else None,
                "label_smoothing": 0.0, "sft_term_in_dpo": False}
            provenance = write_provenance(output, command="baseline_fsdp_train", config=config, inputs=inputs)
            write_json(output / "example_sources.json", [e.get("label_provenance", {"task_id": e.get("task_id")}) for e in examples])
        dist.barrier()
        mesh = init_device_mesh("cuda", (world,))
        model = shard_model(prepare_model(args.snapshot), mesh=mesh)
        reference = shard_model(prepare_model(args.reference, reference=True), mesh=mesh) if args.objective == "dpo" else None
        started = time.monotonic()
        state = train_loop(model, examples, output, objective=args.objective,
                    learning_rate=args.learning_rate, scheduler=args.scheduler, warmup_ratio=args.warmup_ratio,
                    original_batch=args.original_batch, epochs=args.epochs, seed=args.seed, beta=args.beta,
                    reference=reference, weight_decay=args.weight_decay, rank=rank, world_size=world, device=device)
        # Release reference before gathering the final standalone policy.
        del reference
        torch.cuda.empty_cache()
        staging = output / ".checkpoint.partial"
        saved = export_checkpoint(model, staging, args.base_snapshot, rank=rank)
        export_memory = [None] * world
        dist.all_gather_object(export_memory, {"rank": rank,
              "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        dist.barrier()
        if rank == 0:
            state.update(saved, provenance_sha256=digest(provenance), final=True,
                         wall_time_with_export_seconds=time.monotonic() - started,
                         wall_time_with_preparation_seconds=time.monotonic() - preparation_started,
                         memory_per_rank=export_memory)
            write_json(staging / "training_state.json", state)
            os.rename(staging, output / "checkpoint")
            fsync_dir(output)
            write_json(output / "result.json", {"checkpoint": "checkpoint", **state})
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
