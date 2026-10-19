"""Full text-parameter SFT/DPO, torchrun FSDP2, exact global example batches.

Production is GPU-only. train_loop/objective functions also support tiny CPU
models for offline verification; no alternate foundation model is selected.
"""
import argparse
from datetime import timedelta
from functools import partial
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from baselines.common.checkpoint import checkpoint_key_map, checkpoint_layout, validate_checkpoint_layout
from baselines.common.conversation import conversation, preference_conversation, sequence_rows
from baselines.common.schedule import dose, epoch_batches, lr_multiplier, rank_slots, warmup_count
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.seed_evidence import checked_episodes, load_verified_collection, artifact_hashes
from sed.storage import digest, file_hash, write_json, JsonlLedger, fsync_dir
from train.dataset import load_tokenizer
from train.modeling import copy_base_assets, load_student, snapshot_metadata


def text_parameter(name):
    return name.startswith("model.language_model.") or name.startswith("lm_head.")


def rank_zero_data(rank, world, compute_fn, group, *, broadcast=None):
    """Hand off rank 0's verified objects on the long-timeout CPU group."""
    if world == 1:
        return compute_fn()
    if broadcast is None:
        from torch.distributed import broadcast_object_list as broadcast
    message = [None]
    if rank == 0:
        try:
            value = compute_fn()
        except BaseException:
            # Send no raw exception text; rank 0 retains the original traceback.
            message[0] = (False, None)
            broadcast(message, src=0, group=group)
            raise
        message[0] = (True, value)
    broadcast(message, src=0, group=group)
    ok, value = message[0]
    if not ok:
        raise RuntimeError("Rank 0 startup validation failed; see rank 0 traceback")
    return value


def validate_checkpoint(path, student=None, *, base_snapshot=None):
    path = Path(path)
    state = json.loads((path / "training_state.json").read_text())
    hashes = artifact_hashes(path)
    hashes.pop("training_state.json")
    provenance = json.loads((path.parent / "provenance.json").read_text())
    valid_dtype = state.get("text_export_dtype") == "bfloat16" or (
        state.get("text_export_dtype") == "float32" and state.get("control") == "dose_matched_init"
        and state.get("merge_policy") == "seed_sft_fp32")
    if (not state.get("final") or not valid_dtype
            or hashes != state["artifact_sha256"] or digest(provenance) != state["provenance_sha256"]):
        raise ValueError("Full checkpoint artifacts/provenance changed")
    if student is not None and provenance["config"]["models"]["student"] != student:
        raise ValueError("Full checkpoint differs from configured student identity")
    snapshot_metadata(path)
    base = checkpoint_layout(base_snapshot if base_snapshot is not None else state["non_text_source"])
    validate_checkpoint_layout(path, base, [key for key in base if not text_parameter(key)])
    if state["text_export_dtype"] == "float32":
        exported = checkpoint_layout(path)
        if (provenance.get("command") != "dose_matched_init"
                or provenance.get("details", {}).get("init") != state.get("init")
                or any(exported[key]["dtype"] != "F32" or exported[key]["shape"] != base[key]["shape"]
                       for key in base if text_parameter(key))):
            raise ValueError("Dose-control fp32 checkpoint layout/provenance changed")
    return state


def prepare_model(snapshot, *, reference=False):
    """SDPA with fp32 policy masters or frozen bf16 reference storage (X6/X7)."""
    # FSDP already casts reference parameters to bf16 for every forward.
    # Loading directly in that dtype preserves compute and halves storage.
    model = load_student(snapshot, device="cpu", dtype="bfloat16" if reference else "float32")
    model.set_attn_implementation("sdpa")
    model.config.get_text_config().use_cache = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not reference and text_parameter(name))
    if reference:
        model.eval()
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def shard_model(model, *, mesh, reference=False):
    import torch
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextDecoderLayer
    layers = [m for m in model.modules() if isinstance(m, Gemma4UnifiedTextDecoderLayer)]
    if len(layers) != model.config.text_config.num_hidden_layers:
        raise ValueError("Gemma4UnifiedTextDecoderLayer wrap count differs from config")
    storage_dtype = torch.bfloat16 if reference else torch.float32
    if any(p.dtype != storage_dtype for p in model.parameters()):
        raise ValueError(f"FSDP {'reference' if reference else 'policy'} requires {storage_dtype} storage")
    if reference and (model.training or any(p.requires_grad for p in model.parameters())):
        raise ValueError("FSDP reference must be frozen and in eval mode")
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    return model


def scoring_batches(rows, *, rounds=None):
    """One turn per continuation per forward; zero-weight collective padding."""
    groups = [sequence_rows(row) for row in rows]
    needed = max(map(len, groups))
    rounds = needed if rounds is None else rounds
    if rounds < needed or needed < 1:
        raise ValueError("Scoring rounds cannot omit assistant turns")
    return [[(group[i] if i < len(group) else group[0], i < len(group))
             for group in groups] for i in range(rounds)]


def batch_logps(model, rows, device, *, rounds=None):
    """Sum independent exact-prefix turn scores back into each continuation.

    Explicit rounds align FSDP forwards/backwards across ranks with unequal
    continuation lengths. Dummy turns contribute zero score/gradient. A pair
    forwards at most two sequences together, as in the original scorer.
    """
    import torch
    scores = []
    for batch in scoring_batches(rows, rounds=rounds):
        values = sequence_logps(model, [row for row, _ in batch], device)
        scores.append(values * torch.tensor([active for _, active in batch], device=device))
    return sum(scores)


def sequence_logps(model, rows, device):
    """Exact summed assistant log-probs, materializing only target predecessors.

    Selected logits are already causally aligned. Calling the verbatim scorer
    with is_encoder_decoder=True skips its shift, not its log-softmax/mask/sum.
    No sequence or assistant token is truncated.
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
    # Under no_grad this releases the vocabulary tensor before the next turn
    # or policy forward; only summed log-probs leave this scope.
    del logits
    return sum(values)


def supervised_count(row):
    return sum(sum(v != -100 for v in sequence["labels"]) for sequence in sequence_rows(row))


def sft_microsteps(examples, indices):
    """Expand AFTER trajectory sampling; preserve each trajectory's token mean.

    A turn contributes n_turn/n_trajectory of its trajectory loss. Normalizing
    by turn count would silently overweight short replies and alter the recipe.
    """
    result = []
    for index in indices:
        total = supervised_count(examples[index])
        if total == 0:
            raise ValueError("No supervised tokens")
        result.extend((row, supervised_count(row) / total)
                      for row in sequence_rows(examples[index]))
    return result


def weighted_microsteps(examples, indices):
    """STeCa sums continuation log likelihood; never normalize by token count."""
    result = []
    for index in indices:
        example = examples[index]
        weight = example["reward"] * example["dataset_scale"]
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Invalid STeCa reward/dataset scale")
        result.extend((row, weight) for row in sequence_rows(example))
    return result


def weighted_objective(model, row, device):
    return -batch_logps(model, [row], device).mean(), {}


def sft_objective(model, row, device):
    count = supervised_count(row)
    return -batch_logps(model, [row], device).mean() / count, {}


def dpo_objective(model, reference, pair, device, *, beta, nll_weight=0.0, rounds=None,
                  backward_scale=None):
    """Summed DPO/IPR loss and metrics; optionally backward one turn at a time.

    With backward_scale, the first pass retains only no-grad continuation sums.
    The second pass immediately backpropagates each turn's chain-rule term and
    returns a detached loss (already backpropagated). Dropout must be disabled,
    as in train_loop, so recomputation gives the same scores/derivatives.
    Without backward_scale this remains the original one-graph objective.
    """
    import torch
    from baselines.eto.official_loss import OfficialLoss
    if reference is None or reference is model or any(p.requires_grad for p in reference.parameters()):
        raise ValueError("DPO requires a distinct frozen reference")
    reference.eval()
    rows = [pair["chosen"], pair["rejected"]]
    if backward_scale is not None and any(row.get("packing") != "turn" for row in rows):
        raise ValueError("Two-pass backward requires turn packing")
    with torch.no_grad():
        ref = batch_logps(reference, rows, device, rounds=rounds).detach()
    # Reference activations/logits are out of scope before policy activations
    # are allocated. Only two detached continuation sums remain.
    with torch.set_grad_enabled(torch.is_grad_enabled() and backward_scale is None):
        policy = batch_logps(model, rows, device, rounds=rounds)
    settings = SimpleNamespace(beta=beta, loss_type="sigmoid", label_smoothing=0.0)
    losses, chosen, rejected = OfficialLoss.dpo_loss(settings, policy[:1], policy[1:], ref[:1], ref[1:])
    metrics = {"rewards/chosen": chosen.mean(), "rewards/rejected": rejected.mean(),
               "rewards/accuracies": (chosen > rejected).float().mean(),
               "rewards/margins": (chosen - rejected).mean(),
               "logps/chosen": policy[0].detach(), "logps/rejected": policy[1].detach()}
    # IPR 93335a3 dpo_trainer.py:841-844: chosen NLL is SUMMED, weight 1.
    # ETO's default remains exactly its original sigmoid-only objective.
    if nll_weight:
        metrics.update({"loss/dpo": losses.mean().detach(), "loss/chosen_nll": -policy[0].detach()})
    loss = losses.mean() - nll_weight * policy[0]
    if backward_scale is not None:
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        with torch.no_grad():
            # Match the official loss's subtraction order. Rewards/margins
            # are diagnostics and have a different floating-point grouping.
            delta = (policy[0] - policy[1]) - (ref[0] - ref[1])
            rejected_coefficient = beta * torch.sigmoid(-beta * delta)
            coefficients = torch.stack((-rejected_coefficient - nll_weight,
                                        rejected_coefficient)) * backward_scale
        for batch in scoring_batches(rows, rounds=rounds):
            for side, (turn, active) in enumerate(batch):
                # Even dummy turns/ranks must forward AND backward, in the
                # same order, to align FSDP collectives. Never use no_sync:
                # reduced gradients must stay sharded during accumulation.
                turn_logp = sequence_logps(model, [turn], device)[0]
                (coefficients[side] * active * turn_logp).backward()
                del turn_logp
    return loss, metrics


def train_loop(model, examples, output, *, objective, learning_rate, scheduler,
               warmup_ratio, original_batch, epochs=3, seed=0, beta=0.1,
               reference=None, weight_decay=0.0, rank=0, world_size=1, device="cpu", warmup_steps=None,
               packing=None, processed_token_budget=None, checkpoint_callback=None):
    import torch
    import torch.distributed as dist
    if objective not in {"sft", "sad", "dpo", "ipr", "steca"} or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid training objective/learning rate")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("Positive finite beta required")
    actual_packings = {row.get("packing", "episode") for example in examples
                      for row in ([example] if objective in {"sft", "sad", "steca"} else [example["chosen"], example["rejected"]])}
    if len(actual_packings) != 1 or (packing is not None and actual_packings != {packing}):
        raise ValueError("Training packing differs from encoded examples")
    packing, = actual_packings
    two_pass_turn_backward = objective in {"dpo", "ipr"} and packing == "turn"
    empty_visible_counts = []
    for example in examples:
        groups = [example] if objective in {"sft", "sad", "steca"} else [example["chosen"], example["rejected"]]
        # Count each stage input once, independently of epochs/rank padding.
        empty_visible_counts.append(sum(row.get("empty_visible_turns_masked", 0) for row in groups))
        for row in groups:
            if any(r.get("label_provenance", {}).get("training_eligible") is False
                   for r in [row, *sequence_rows(row)]):
                raise ValueError("Diagnostic examples cannot enter training")
    if objective == "sad":
        from baselines.sad.data import token_plan, ordered_batches
        plan = token_plan(examples, original_batch, processed_token_budget)
        batches = ordered_batches(len(examples), plan["b_prime"], plan["total_steps"])
    else:
        if processed_token_budget is not None or checkpoint_callback is not None:
            raise ValueError("Token budgets/checkpoint callbacks are SAD-only")
        plan = dose(len(examples), original_batch, epochs=epochs)
        batches = epoch_batches(len(examples), plan["b_prime"], epochs=epochs, seed=seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=weight_decay, foreach=False, fused=False)
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_multiplier,
                  scheduler=scheduler, total_steps=plan["total_steps"], warmup_ratio=warmup_ratio,
                  warmup_steps=warmup_steps))
    model.train()
    if objective in {"dpo", "ipr"}:
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
    started = time.monotonic()
    exposure = supervised = physical = physical_tokens = forward_tokens = logical_slots = 0
    recompute_tokens = 0
    for update, batch in enumerate(batches, 1):
        optimizer.zero_grad(set_to_none=True)
        metrics_sum = {}
        indices = batch["indices"]
        microsteps = ([(r, 1.0) for i in indices for r in sequence_rows(examples[i])]
                      if objective == "sad" else weighted_microsteps(examples, indices) if objective == "steca" else
                      sft_microsteps(examples, indices) if objective == "sft" else
                      [(examples[i], 1.0) for i in indices])
        slots = rank_slots(list(range(len(microsteps))), world_size)
        # All ranks have the same logical batch. Derive the same number of
        # preference turn forwards without a separate distributed collective.
        rounds = [1 if objective in {"sft", "sad", "steca"} else max(
            len(sequence_rows(microsteps[i if i is not None else 0][0][side]))
            for i in slot for side in ("chosen", "rejected")) for slot in slots]
        for slot, n_rounds in zip(slots, rounds):
            index = slot[rank]
            active = index is not None
            row, weight = microsteps[index if active else 0]
            # FSDP averages across ranks. Cancel that average and divide by
            # ACTUAL trajectories/pairs, including the final partial batch.
            scale = weight * world_size / len(indices) if active else 0.0
            loss, metrics = ((-batch_logps(model, [row], device).sum(), {}) if objective == "sad" else
                            weighted_objective(model, row, device) if objective == "steca" else
                            sft_objective(model, row, device) if objective == "sft" else
                            dpo_objective(model, reference, row, device, beta=beta,
                                          nll_weight=1.0 if objective == "ipr" else 0.0, rounds=n_rounds,
                                          backward_scale=scale if two_pass_turn_backward else None))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            # Turn weights preserve the per-trajectory assistant-token mean.
            # Synchronize each microstep to keep gradients sharded during accumulation.
            if not two_pass_turn_backward:
                (loss * scale).backward()
            for key, value in {"loss": loss.detach(), **metrics}.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value.detach() * (weight if active else 0.0)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        lr_schedule.step()
        # Same global accounting on every rank; padding is extra compute only.
        exposure += len(indices)
        def costs(microstep, n_rounds=None):
            row, _ = microstep
            groups = [row] if objective in {"sft", "sad", "steca"} else [row["chosen"], row["rejected"]]
            rows = [r for group in groups for r in sequence_rows(group)]
            return (sum(sum(v != -100 for v in r["labels"]) for r in rows),
                    sum(len(r["input_ids"]) for r in rows),
                    sum(len(batch) * max(len(r["input_ids"]) for r, _ in batch)
                        for batch in scoring_batches(groups, rounds=n_rounds)),
                    sum(len(r["input_ids"]) for batch in scoring_batches(groups, rounds=n_rounds)
                        for r, _ in batch) if two_pass_turn_backward else 0)
        supervised += sum(costs(m)[0] for m in microsteps)
        forward_tokens += sum(costs(m)[1] for m in microsteps)
        logical_slots += len(microsteps)
        physical += len(slots) * world_size
        physical_tokens += sum(costs(microsteps[i if i is not None else 0], n_rounds)[2]
                               for slot, n_rounds in zip(slots, rounds) for i in slot)
        recompute_tokens += sum(costs(microsteps[i if i is not None else 0], n_rounds)[3]
                                for slot, n_rounds in zip(slots, rounds) for i in slot)
        keys = sorted(metrics_sum)
        values = torch.stack([metrics_sum[k] for k in keys])
        if world_size > 1:
            dist.all_reduce(values)
        values /= len(indices)
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        if ledger:
            ledger.append({"update": update, **batch, "lr": lr_used, "global_batch": len(indices),
                           "packing": packing, "batch_unit": "trajectory" if objective in {"sft", "sad", "steca"} else "preference pair",
                           "microsteps": len(microsteps), "collective_rounds": sum(rounds),
                           "zero_weight_slots": len(slots) * world_size - len(microsteps),
                           "gradient_norm": float(norm), "examples_seen": exposure,
                           "supervised_tokens_seen": supervised,
                           "logical_forward_tokens": forward_tokens,
                           "two_pass_turn_backward": two_pass_turn_backward,
                           "physical_policy_forward_tokens": physical_tokens + recompute_tokens,
                           "physical_policy_recompute_forward_tokens": recompute_tokens,
                           "physical_reference_forward_tokens": physical_tokens if objective not in {"sft", "sad", "steca"} else 0,
                           **dict(zip(keys, values.cpu().tolist()))}, unique_key="update")
        if checkpoint_callback is not None:
            checkpoint_callback(model, {**plan, "updates": update, "seed": seed,
                "examples_seen": exposure, "supervised_tokens_seen": supervised,
                "logical_forward_tokens": forward_tokens, "processed_token_positions": forward_tokens,
                "physical_policy_forward_tokens": physical_tokens, "packing": packing,
                "loss_normalization": "batch mean of trajectory span-token sums"})
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
            "empty_visible_turns_masked": sum(empty_visible_counts),
            "empty_visible_turn_examples": sum(count > 0 for count in empty_visible_counts),
            "supervised_tokens_seen": supervised, "logical_forward_tokens": forward_tokens,
            "physical_example_slots": physical, "logical_example_slots": logical_slots,
            "zero_weight_slots": physical - logical_slots,
            "two_pass_turn_backward": two_pass_turn_backward,
            "physical_policy_forward_tokens": physical_tokens + recompute_tokens,
            "physical_policy_recompute_forward_tokens": recompute_tokens,
            "physical_reference_forward_tokens": physical_tokens if objective not in {"sft", "sad", "steca"} else 0,
            "wall_time_seconds": time.monotonic() - started, "memory_per_rank": all_memory,
            "packing": packing, "batch_unit": "trajectory" if objective in {"sft", "sad", "steca"} else "preference pair",
            "microbatch": 1, "microbatch_unit": packing if objective in {"sft", "sad", "steca"} else "preference pair",
            "master_dtype": "float32", "compute_dtype": "bfloat16" if cuda else "float32",
            "gradient_clip": 1.0, "weight_decay": weight_decay,
            **({"processed_token_positions": forward_tokens,
                "realized_trajectory_passes": exposure / len(examples)} if objective == "sad" else {}),
            "loss_reduction": ("span_token_sum_then_global_batch_mean" if objective == "sad" else
                               "reward_weighted_summed_logps_dataset_means" if objective == "steca" else
                               "per_example_mean_then_global_batch_mean" if objective == "sft" else
                               "sigmoid_plus_chosen_nll_summed_logps" if objective == "ipr" else "sigmoid_summed_logps")}


def export_checkpoint(model, destination, base_snapshot, *, rank=0, max_shard_bytes=2_000_000_000,
                      text_dtype="bfloat16"):
    """Gather one tensor at a time, text weights + unchanged base towers.

    Default exports remain bf16. Dose-control initialization opts into fp32
    to retain the LoRA delta before FSDP constructs its fp32 policy masters.

    Every rank participates in text DTensor gathers. Rank zero writes bounded
    CPU shards in the original base's namespace, omitting unstored tied aliases.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    if text_dtype not in {"bfloat16", "float32"}:
        raise ValueError("Unsupported text export dtype")
    export_dtype = getattr(torch, text_dtype)
    destination, base = Path(destination), Path(base_snapshot)
    base_layout = checkpoint_layout(base)
    state_dict = model.state_dict()
    key_map = checkpoint_key_map(model, base_layout, state_dict=state_dict)
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=False)
    pending, size, total, mapping, tower_hashes = {}, 0, 0, {}, {}
    def flush():
        nonlocal pending, size
        if not pending:
            return
        name = f"model-{len(set(mapping.values())) + 1:05d}.safetensors"
        save_file(pending, destination / name, metadata={"format": "pt"})
        mapping.update({key: name for key in pending})
        pending, size = {}, 0
    for name, file_name in key_map.items():
        value = state_dict[name]
        spec = base_layout[file_name]
        if list(value.shape) != spec["shape"]:
            raise ValueError(f"Model tensor shape differs from original base snapshot: {name}")
        # Frozen multimodal weights come from the ORIGINAL base, not a partial
        # text-only export or random initialization; no need to gather them.
        if not text_parameter(name):
            if rank == 0:
                with safe_open(base / spec["file"], framework="pt", device="cpu") as handle:
                    tensor = handle.get_tensor(file_name).contiguous().clone()
                import hashlib
                tower_hashes[file_name] = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
        else:
            full = value.full_tensor() if hasattr(value, "full_tensor") else value.detach()
            if rank == 0:
                tensor = full.to(device="cpu", dtype=export_dtype).contiguous().clone()
            del full
        if rank == 0:
            nbytes = tensor.numel() * tensor.element_size()
            if size + nbytes > max_shard_bytes:
                flush()
            pending[file_name] = tensor
            size += nbytes
            total += nbytes
    if rank != 0:
        return None
    flush()
    write_json(destination / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": mapping})
    validate_checkpoint_layout(destination, base_layout, tower_hashes)
    # Preserve tokenizer/template and multimodal processor assets from the base.
    copy_base_assets(base, destination, overwrite=True)
    model.config.get_text_config().use_cache = True
    model.config.dtype = export_dtype
    model.config.get_text_config().dtype = export_dtype
    model.config.save_pretrained(destination)
    if model.generation_config is not None:
        model.generation_config.save_pretrained(destination)
    return {"non_text_tensor_sha256": tower_hashes, "non_text_source": str(base.resolve()),
            "artifact_sha256": artifact_hashes(destination), "text_export_dtype": text_dtype}


def prepare_lora_model(snapshot, *, device="cuda", adaptation="kang_lora"):
    """Registered text-only LoRA recipes; Kang remains the default."""
    from train.modeling import attach_lora
    model = load_student(snapshot, device=device, dtype="bfloat16" if device != "cpu" else "float32")
    model.set_attn_implementation("sdpa")
    if adaptation == "wkm_lora":
        return attach_lora(model, rank=8, alpha=16, dropout=0.0, targets=["q_proj", "v_proj"])
    if adaptation != "kang_lora":
        raise ValueError("Unknown LoRA adaptation")
    return attach_lora(model, rank=64, alpha=128, dropout=0.05)


def export_adapter(model, destination, provenance, state):
    """PEFT artifacts with the existing serve_student/equivalence-gate schema."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(destination, safe_serialization=True)
    state = {**state, "identity_sha256": digest(provenance), "final": True,
             "artifact_sha256": artifact_hashes(destination),
             "passes": state["epochs"], "supervised_tokens": state["supervised_tokens_seen"]}
    write_json(destination / "training_state.json", state)
    return state


def train_lora(args):
    """Single-GPU LoRA branch; the full-parameter FSDP path stays separate."""
    import torch
    from transformers import set_seed
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Kang LoRA production training requires one allocated GPU")
    if Path(args.snapshot).resolve() != Path(args.base_snapshot).resolve():
        raise ValueError("Kang SFT must start from the configured base snapshot")
    started = time.monotonic()
    set_seed(args.seed)
    config = load_config(args.config)
    if (config["training"]["thinking_mode"] != "off" or config["training"]["student_step_token_cap"] != 1024
            or config["environment"]["max_episode_steps"] != 40):
        raise ValueError("Baseline harness protocol differs")
    config["training"]["adaptation"] = args.adaptation
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, root = load_verified_collection(args.verified, manifest, freeze)
    tokenizer = load_tokenizer(args.snapshot)
    examples = [conversation(episode, tokenizer, source="trusted_teacher_seed",
                             packing=args.packing, max_length=args.max_length)
                for episode in checked_episodes(root, collection["verified"])]
    if args.method_data:
        from baselines.wkm.data import training_examples
        examples = training_examples(args.method_data, config, tokenizer, verified=args.verified)
    plan = dose(len(examples), args.original_batch, epochs=args.epochs)
    model = prepare_lora_model(args.snapshot, adaptation=args.adaptation)
    peft_config = model.peft_config["default"]
    peft_config.revision = config["models"]["student"]["revision"]
    base = snapshot_metadata(args.base_snapshot)
    config["sft"] = {"base_snapshot": base, "packing": args.packing, "rank": peft_config.r,
                     "alpha": peft_config.lora_alpha, "dropout": peft_config.lora_dropout, "targets": sorted(peft_config.target_modules)}
    config["baseline_training"] = {**vars(args), **plan, "world_size": 1,
        "base_snapshot": base, "base_artifact_sha256": artifact_hashes(args.base_snapshot),
        "examples_sha256": digest(examples), "template_sha256": digest(tokenizer.chat_template),
        "attention": "sdpa", "compute_dtype": "bfloat16", "adapter_master_dtype": "float32",
        "frozen_base_dtype": "bfloat16", "gradient_checkpointing": True, "no_truncation": True,
        "packing": args.packing, "batch_unit": "trajectory", "microbatch": 1,
        "microbatch_unit": args.packing, "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
        "warmup_steps": warmup_count(plan["total_steps"], args.warmup_ratio, args.warmup_steps),
        "K": collection["K"], "unique_instances": collection["unique_instances"], "task_groups": collection["task_groups"]}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = write_provenance(output, command="baseline_" + args.adaptation + "_train", config=config,
        inputs=[args.config, config["paths"]["manifest"], config["paths"]["split_freeze"],
                Path(args.verified) / "manifest.json", Path(args.verified) / "identity.json",
                *([args.method_data, Path(args.method_data).parent / "provenance.json"] if args.method_data else [])])
    write_json(output / "example_sources.json", [e["label_provenance"] for e in examples])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = train_loop(model, examples, output, objective="sft", learning_rate=args.learning_rate,
            scheduler=args.scheduler, warmup_ratio=args.warmup_ratio, warmup_steps=args.warmup_steps,
            original_batch=args.original_batch, epochs=args.epochs, seed=args.seed,
            weight_decay=args.weight_decay, device="cuda", packing=args.packing)
    state.update(adaptation=args.adaptation, frozen_base_dtype="bfloat16")
    staging = output / ".checkpoint.partial"
    state = export_adapter(model, staging, provenance, state)
    os.rename(staging, output / "checkpoint")
    fsync_dir(output)
    write_json(output / "result.json", {**state, "checkpoint": "checkpoint",
               "wall_time_with_preparation_and_export_seconds": time.monotonic() - started})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--objective", choices=["sft", "sad", "dpo", "ipr", "steca"], required=True)
    parser.add_argument("--adaptation", choices=["full", "kang_lora", "wkm_lora"], default="full")
    parser.add_argument("--packing", choices=["episode", "turn"], default="turn",
                        help="SFT forward packing; X5 batches always count trajectories. Preferences always use turn prefixes.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--base-snapshot", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--verified")
    parser.add_argument("--sft-data", help="Checked AgentGym accumulated dataset; excludes --verified")
    parser.add_argument("--method-data", help="Checked WKM or STeCa stage dataset")
    parser.add_argument("--pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--scheduler", choices=["cosine", "constant_with_warmup", "linear"], required=True)
    parser.add_argument("--warmup-ratio", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, help="Explicit integer, including zero; overrides ratio")
    parser.add_argument("--original-batch", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--processed-token-budget", type=int)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args(argv)
    preparation_started = time.monotonic()
    if args.max_length != 32768:
        parser.error("Registered no-truncation limit is 32768")
    if args.objective == "sft" and bool(args.verified) == bool(args.sft_data):
        parser.error("SFT requires exactly one of --verified or --sft-data")
    if args.objective == "sad":
        if (not args.verified or args.sft_data or args.packing != "turn" or args.adaptation != "full"
                or Path(args.snapshot).resolve() != Path(args.base_snapshot).resolve()):
            parser.error("SAD requires verified-only full tuning from base with turn packing")
        from baselines.sad.data import TOKEN_BUDGET
        if args.processed_token_budget is None:
            args.processed_token_budget = TOKEN_BUDGET
    elif args.processed_token_budget is not None:
        parser.error("Processed-token budgets are SAD-only")
    if args.method_data and not (args.adaptation == "wkm_lora" or args.objective == "steca"):
        parser.error("Method data requires WKM LoRA or STeCa objective")
    if args.objective == "steca" and (not args.method_data or not args.verified or args.adaptation != "full"):
        parser.error("STeCa requires method data, verified support and full fine-tuning")
    if args.adaptation in {"kang_lora", "wkm_lora"}:
        if args.objective != "sft" or not args.verified or args.sft_data:
            parser.error("Kang LoRA requires seed SFT")
        return train_lora(args)
    if args.objective == "dpo" and (not args.reference or Path(args.reference).resolve() != Path(args.snapshot).resolve()):
        parser.error("Reference must be the iteration-start policy snapshot")
    if args.objective == "ipr" and (not args.reference or not args.pairs):
        parser.error("IPR requires current-iteration pairs and the fixed SFT reference")
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
        verification_group = dist.new_group(backend="gloo", timeout=timedelta(hours=4))
        set_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        def prepare_data():
            config = load_config(args.config)
            if (config["training"]["thinking_mode"] != "off" or config["training"]["student_step_token_cap"] != 1024
                    or config["environment"]["max_episode_steps"] != 40):
                raise ValueError("Baseline harness protocol differs")
            config["training"]["adaptation"] = "full_parameter_text_fsdp2"
            if args.objective in {"dpo", "ipr"}:
                validate_checkpoint(args.snapshot, config["models"]["student"], base_snapshot=args.base_snapshot)
                validate_checkpoint(args.reference, config["models"]["student"], base_snapshot=args.base_snapshot)
            manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
            tokenizer = load_tokenizer(args.snapshot)
            inputs = [args.config, config["paths"]["manifest"], config["paths"]["split_freeze"]]
            if args.objective == "steca":
                from baselines.steca.data import training_examples
                validate_checkpoint(args.snapshot, config["models"]["student"], base_snapshot=args.base_snapshot)
                details = json.loads((Path(args.method_data).parent / "provenance.json").read_text())["details"]
                if (Path(details["warmup_checkpoint"]).resolve() != Path(args.snapshot).resolve()
                        or file_hash(Path(args.snapshot).parent / "provenance.json") != details["warmup_provenance_sha256"]):
                    raise ValueError("STeCa reinforced initialization differs from fixed exploration/MC model")
                examples = training_examples(args.method_data, config, tokenizer, verified=args.verified)
                inputs += [args.method_data, Path(args.method_data).parent / "provenance.json"]
            elif args.objective in {"sft", "sad"}:
                if args.sft_data:
                    from baselines.agentgym.data import training_examples
                    examples = training_examples(args.sft_data, config, tokenizer,
                                                 packing=args.packing, max_length=args.max_length)
                    inputs += [args.sft_data, Path(args.sft_data).parent / "provenance.json"]
                else:
                    collection, selection, root = load_verified_collection(args.verified, manifest, freeze)
                    if args.objective == "sad":
                        from baselines.sad.data import curriculum, encode_episode
                        examples = curriculum([encode_episode(e, tokenizer, max_length=args.max_length)
                            for e in checked_episodes(root, collection["verified"])])
                    else:
                        examples = [conversation(episode, tokenizer, source="trusted_teacher_seed",
                                                 packing=args.packing, max_length=args.max_length)
                                    for episode in checked_episodes(root, collection["verified"])]
                    inputs += [Path(args.verified) / "manifest.json", Path(args.verified) / "identity.json"]
            else:
                if args.objective == "ipr":
                    from baselines.ipr.pairs import load_pair_data, preference_conversation as encode_pair
                    details = json.loads((Path(args.pairs).parent / "provenance.json").read_text())["details"]
                    for source, checkpoint in (("expert_mc", args.reference), ("exploration", args.snapshot)):
                        serving = json.loads((Path(details[source]) / "provenance.json").read_text())
                        server = serving["config"]["server_provenance"]["config"]["server"]
                        if (Path(server["model_path"]).resolve() != Path(checkpoint).resolve()
                                or server["training_provenance_sha256"] != file_hash(Path(checkpoint).parent / "provenance.json")):
                            raise ValueError("IPR init/reference differs from exploration/fixed SFT MC model")
                else:
                    from baselines.eto.pairs import load_pair_data
                    encode_pair = preference_conversation
                pairs = load_pair_data(args.pairs, config)
                examples = [encode_pair(pair, tokenizer) for pair in pairs]
                inputs += [args.pairs, Path(args.pairs).parent / "provenance.json"]
                inputs += [Path(args.reference) / "training_state.json", Path(args.reference).parent / "provenance.json"]
            return examples, inputs, config, tokenizer.chat_template
        examples, inputs, config, chat_template = rank_zero_data(rank, world, prepare_data, verification_group)
        if args.objective == "sad":
            from baselines.sad.data import token_plan
            plan = token_plan(examples, args.original_batch, args.processed_token_budget)
        else:
            plan = dose(len(examples), args.original_batch, epochs=args.epochs)
        output = Path(args.output)
        provenance = None
        def record_provenance():
            nonlocal provenance
            output.mkdir(parents=True, exist_ok=False)
            config["baseline_training"] = {**vars(args), **plan, "world_size": world,
                "snapshot": snapshot_metadata(args.snapshot), "base_snapshot": snapshot_metadata(args.base_snapshot),
                "snapshot_artifact_sha256": artifact_hashes(args.snapshot),
                "base_artifact_sha256": artifact_hashes(args.base_snapshot),
                "examples_sha256": digest(examples), "template_sha256": digest(chat_template),
                "attention": "sdpa", "master_dtype": "float32", "compute_dtype": "bfloat16",
                "wrap_class": "Gemma4UnifiedTextDecoderLayer", "gradient_checkpointing": True,
                "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
                "no_truncation": True, "packing": args.packing if args.objective in {"sft", "sad", "steca"} else "turn", "microbatch": 1,
                "batch_unit": "trajectory" if args.objective in {"sft", "sad", "steca"} else "preference pair",
                "microbatch_unit": args.packing if args.objective in {"sft", "sad", "steca"} else "preference pair",
                "warmup_steps": warmup_count(plan["total_steps"], args.warmup_ratio, args.warmup_steps),
                "reference_policy": ("fixed_sft" if args.objective == "ipr" else
                                     "fixed_seed_sft_init" if args.objective == "dpo" and
                                     json.loads((Path(args.reference) / "training_state.json").read_text()).get("control") == "dose_matched_init" else
                                     "iteration_start" if args.objective == "dpo" else None),
                "reference_storage_dtype": "bfloat16" if args.objective in {"dpo", "ipr"} else None,
                "two_pass_turn_backward": args.objective in {"dpo", "ipr"},
                "reference_artifact_sha256": artifact_hashes(args.reference) if args.reference else None,
                "label_smoothing": 0.0, "sft_term_in_dpo": args.objective == "ipr",
                "chosen_nll_weight": 1 if args.objective == "ipr" else 0}
            provenance = write_provenance(output, command="baseline_fsdp_train", config=config, inputs=inputs)
            write_json(output / "example_sources.json", [e.get("label_provenance", {"task_id": e.get("task_id")}) for e in examples])
        rank_zero_data(rank, world, record_provenance, verification_group)
        dist.barrier()
        mesh = init_device_mesh("cuda", (world,))
        model = shard_model(prepare_model(args.snapshot), mesh=mesh)
        reference = (shard_model(prepare_model(args.reference, reference=True), mesh=mesh, reference=True)
                     if args.objective not in {"sft", "sad", "steca"} else None)
        started = time.monotonic()
        checkpoint_callback = None
        if args.objective == "sad":
            from baselines.sad.checkpoints import publisher
            checkpoint_callback = publisher(output, args.base_snapshot, provenance, plan, rank=rank)
        state = train_loop(model, examples, output, objective=args.objective,
                    learning_rate=args.learning_rate, scheduler=args.scheduler, warmup_ratio=args.warmup_ratio,
                    original_batch=args.original_batch, epochs=args.epochs, seed=args.seed, beta=args.beta,
                    reference=reference, weight_decay=args.weight_decay, rank=rank, world_size=world, device=device,
                    warmup_steps=args.warmup_steps, packing=args.packing if args.objective in {"sft", "sad", "steca"} else "turn",
                    processed_token_budget=args.processed_token_budget, checkpoint_callback=checkpoint_callback)
        if args.objective == "sad":
            if rank == 0:
                write_json(output / "result.json", {**state, "models": json.loads((output / "models.json").read_text())})
            dist.barrier()
            return
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
