"""V0 continuation: V2 encoding, optimizer, precision, training loop and export."""
import argparse
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]

from repair import v2_train as v2
from repair.artifacts import provenance, read_json, tokenizer_identity
from repair.v2_inputs import ValidationError, require
from sed.hash_receipts import cache_provenance
from sed.raw_hash_cache import cached_inputs, raw_input_cache
from sed.storage import active_file_hash_cache, digest, file_hash, write_json
from tools.v0_data import (BUCKETS, GENERATORS, add_inputs, episodes, expansion_units, load_args,
                           provenance_inputs, safe_output, sealed_input, verified_parent)
from tools.v0_groups import episode_weights

DEFAULT_MIXTURE = {"R": .50, "G": .25, "S": .15, "D": .10}


def episode_filter(rows, args):
    """Select whole episodes without changing the verified rows or their indices."""
    excluded = getattr(args, "exclude_generators", None)
    fraction = getattr(args, "episode_fraction", None)
    if excluded is None and fraction is None:
        return None
    seed = getattr(args, "subsample_seed", None)
    grouped = episodes(rows)
    buckets = {b: [] for b in BUCKETS}
    for eid, indices in grouped.items():
        identities = {(rows[i]["v0"]["bucket"], rows[i]["v0"]["generator"]) for i in indices}
        require(len(identities) == 1, "Episode crosses buckets or generators")
        bucket, _ = identities.pop()
        buckets[bucket].append(eid)
    kept, dropped = {}, {}
    rng = random.Random(seed)
    for bucket, ids in buckets.items():
        chosen = set(ids)
        if bucket != "D":
            if excluded is not None:
                chosen = {eid for eid in ids if rows[grouped[eid][0]]["v0"]["generator"] not in excluded}
            elif ids:
                shuffled = sorted(ids)
                rng.shuffle(shuffled)
                chosen = set(shuffled[:max(1, round(fraction * len(ids)))])
        kept[bucket] = sorted(chosen)
        dropped[bucket] = sorted(set(ids) - chosen)
    bucket_counts = {b: {"rows": 0, "episodes": 0} for b in BUCKETS}
    generator_counts = {g: {"rows": 0, "episodes": 0} for g in GENERATORS}
    for bucket, ids in kept.items():
        for eid in ids:
            indices = grouped[eid]
            generator = rows[indices[0]]["v0"]["generator"]
            for counts in (bucket_counts[bucket], generator_counts[generator]):
                counts["rows"] += len(indices)
                counts["episodes"] += 1
    kept_rows = sum(counts["rows"] for counts in bucket_counts.values())
    return {"exclude_generators": excluded, "episode_fraction": fraction, "subsample_seed": seed,
            "kept_episode_ids": kept, "dropped_episode_ids": dropped,
            "buckets": bucket_counts, "generators": generator_counts, "kept_rows": kept_rows,
            "updates_for_4_passes": math.ceil(4 * kept_rows / 16)}


def batch_schedule(rows, *, updates, seed, weights=None, mixture=None, filter_record=None):
    """Global B=16: bucket -> weighted episode -> uniform row, with replacement."""
    require(type(updates) is int and updates > 0 and type(seed) is int and seed >= 0, "Invalid updates/seed")
    weights = {} if weights is None else weights
    mixture = DEFAULT_MIXTURE if mixture is None else mixture
    require(set(mixture) == set(BUCKETS) and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                              for v in mixture.values()), "Invalid bucket mixture")
    grouped = episodes(rows)
    if filter_record is not None:
        kept = {eid for ids in filter_record["kept_episode_ids"].values() for eid in ids}
        grouped = {eid: indices for eid, indices in grouped.items() if eid in kept}
    require(set(weights) <= set(grouped) and all(type(w) in (int, float) and math.isfinite(w) and w >= 0
                                               for w in weights.values()), "Invalid episode weights")
    buckets = {b: [eid for eid, indices in grouped.items() if rows[indices[0]]["v0"]["bucket"] == b] for b in BUCKETS}
    if filter_record is not None:
        for bucket in BUCKETS:
            require(mixture[bucket] == 0 or buckets[bucket],
                    f"Episode filter leaves bucket {bucket} empty with positive mixture probability")
    nonempty = [b for b in BUCKETS if buckets[b]]
    total = sum(mixture[b] for b in nonempty)
    require(total > 0, "No nonempty bucket has positive mass")
    probabilities = {b: mixture[b] / total for b in nonempty}
    within = {b: [weights.get(eid, 1.0) for eid in buckets[b]] for b in nonempty}
    require(all(sum(within[b]) > 0 for b in nonempty), "Nonempty bucket has zero episode weight")
    rng = random.Random(seed)
    batches = []
    for update in range(1, updates + 1):
        selections = []
        for _ in range(16):
            bucket = rng.choices(nonempty, weights=[probabilities[b] for b in nonempty], k=1)[0]
            eid = rng.choices(buckets[bucket], weights=within[bucket], k=1)[0]
            index = rng.choice(grouped[eid])
            selections.append({"bucket": bucket, "episode_id": eid, "index": index})
        batches.append({"update": update, "epoch": 0, "indices": [s["index"] for s in selections],
                        "selections": selections})
    return batches, probabilities


def exposure_counts(batches):
    bucket, episode = Counter(), Counter()
    result = {0: {"bucket_exposures": {}, "episode_exposures": {}}}
    for batch in batches:
        for item in batch["selections"]:
            bucket[item["bucket"]] += 1
            episode[item["episode_id"]] += 1
        result[batch["update"]] = {"bucket_exposures": dict(bucket), "episode_exposures": dict(episode)}
    return result


def mask_modules(model, selected):
    parameters = dict(model.named_parameters())
    require(isinstance(selected, list) and selected and all(isinstance(n, str) for n in selected)
            and len(set(selected)) == len(selected), "Module selection must be nonempty and unique")
    require(all(n in parameters for n in selected), "Selected module name does not exist")
    require(all(parameters[n].requires_grad and any(p in ("lora_A", "lora_B") for p in n.split("."))
                for n in selected), "Selection must contain only original trainable LoRA tensors")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in selected)
        parameter.grad = None
    return {n: p.numel() for n, p in parameters.items() if p.requires_grad}


def subset_optimizer_state(saved, original_names, selected_names):
    """Preserve inherited moments, dropping frozen tensors before optimizer construction."""
    if saved is None:
        return None
    saved = deepcopy(saved)
    if "optimizer" in saved:
        if "trainable" in saved:
            require(list(saved["trainable"]) == original_names, "Optimizer parameter order differs")
        saved = saved["optimizer"]
    groups = saved["param_groups"]
    require(len(groups) == 1 and len(groups[0]["params"]) == len(original_names), "Unsupported optimizer parameter mapping")
    require(selected_names and set(selected_names) <= set(original_names)
            and [n for n in original_names if n in selected_names] == selected_names, "Optimizer selection order differs")
    kept = [pid for name, pid in zip(original_names, groups[0]["params"]) if name in selected_names]
    groups[0]["params"] = kept
    saved["state"] = {pid: saved["state"][pid] for pid in kept if pid in saved["state"]}
    return saved


def episode_entries(data):
    return [{"episode_id": eid, "bucket": data["rows"][indices[0]]["v0"]["bucket"]}
            for eid, indices in data["episodes"].items()]


def inherit_expansion_weights(data, parent, document, path):
    """Use this round's parent gradients; new G gradients enter next round."""
    group_path = sealed_input(data, path.with_name("groups.json"))
    groups = read_json(group_path)
    require(groups.get("schema") == "v0-groups/1" and not groups.get("smoke_subset", False)
            and all(groups.get(k) == document.get(k) for k in
                    ("dataset_sha256", "adapter_sha256", "parameter_names_sha256", "features_index_sha256")),
            "Expansion group/weight source identity mismatch")
    require(isinstance(groups.get("groups"), list) and isinstance(groups.get("assignment"), dict),
            "Invalid expansion groups")
    consistency = {g["group_id"]: g["consistency"] for g in groups["groups"]}
    require(len(consistency) == len(groups["groups"]) and all(
        type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in consistency.values()),
        "Invalid expansion group consistency")
    require(set(groups["assignment"]) <= set(parent["episodes"])
            and all(g in consistency for g in groups["assignment"].values()), "Invalid parent group assignment")
    expected = episode_weights(groups, episode_entries(parent))
    require(set(document["weights"]) == set(expected) and all(
        type(document["weights"][eid]) in (int, float) and math.isclose(document["weights"][eid], value,
        rel_tol=1e-12, abs_tol=1e-12) for eid, value in expected.items()),
        "Parent weights differ from group consistency")
    mapping, paths = expansion_units(data, set(data["episodes"]) - set(parent["episodes"]))
    expanded = dict(groups, assignment=dict(groups["assignment"]))
    for eid, item in mapping.items():
        source, gid = item["source_episode_id"], item["group"]
        require(gid in consistency and source in parent["episodes"]
                and parent["rows"][parent["episodes"][source][0]]["v0"]["bucket"] == "R"
                and groups["assignment"].get(source) == gid,
                f"Unresolved G episode expansion group/source: {eid}")
        expanded["assignment"][eid] = gid
        item["consistency"] = consistency[gid]
    weights = episode_weights(expanded, episode_entries(data))
    for eid, item in mapping.items():
        item["weight"] = weights[eid]
    data["weight_inheritance"] = {"normalization": "R/G episode mean 1; S/D 1; excluded gradients neutral before normalization",
        "gradient_policy": "parent gradients only; new G gradients enter next round",
        "groups_sha256": file_hash(group_path), "G_episode_mapping": mapping}
    return weights, [group_path, *paths]


@cached_inputs
def load_choices(data, inherited, *, weights=None, modules=None):
    result_weights, selected, paths = {}, None, []
    parent = None
    for path, schema in ((weights, "v0-episode-weights/1"), (modules, "v0-trainable-modules/1")):
        if not path:
            continue
        path = data["inspection"]["files"].relocation.path(path)
        document = read_json(path)
        require(document.get("schema") == schema, "Invalid weights/modules schema")
        require(document.get("adapter_sha256") == data["adapter_sha256"],
                "Weights/modules dataset or adapter hash mismatch")
        parent_bound = document.get("dataset_sha256") != data["dataset_sha256"]
        if parent_bound:
            if parent is None:
                parent = verified_parent(data)
                paths.extend(provenance_inputs(parent))
            require(document.get("dataset_sha256") == parent["dataset_sha256"],
                    "Weights/modules dataset or adapter hash mismatch")
        require(not document.get("smoke_subset", False), "Smoke-subset features cannot drive production training")
        require(document.get("parameter_names_sha256") == digest(sorted(inherited["parameters"])),
                "Weights/modules parameter-name hash mismatch")
        if schema == "v0-episode-weights/1":
            require(document.get("status", "valid") == "valid" and isinstance(document.get("weights"), dict),
                    "Weights artifact is invalid")
            result_weights = document["weights"]
            if parent_bound:
                result_weights, inherited_paths = inherit_expansion_weights(data, parent, document, path)
                paths.extend(inherited_paths)
            require(set(result_weights) == set(data["episodes"]), "Weights must cover all dataset episodes")
            require(all(result_weights[eid] == 1 for eid, indices in data["episodes"].items()
                        if data["rows"][indices[0]]["v0"]["bucket"] in ("S", "D")), "S/D weights must be 1")
        else:
            selected = document["selected"]
            require(isinstance(selected, list) and selected and len(set(selected)) == len(selected)
                    and all(n in inherited["parameters"] for n in selected), "Selected module name does not exist or is empty")
        paths.append(path)
    return result_weights, selected, paths


def bucket_summary(data, weights):
    result = {}
    for bucket in BUCKETS:
        entries = [e for e in episode_entries(data) if e["bucket"] == bucket]
        values = [weights.get(e["episode_id"], 1.0) for e in entries]
        result[bucket] = {"rows": sum(r["v0"]["bucket"] == bucket for r in data["rows"]),
            "episodes": len(entries), "weight_min": min(values) if values else None,
            "weight_max": max(values) if values else None,
            "weight_mean": sum(values) / len(values) if values else None}
    return result


def validate_inputs(args, *, rank=0, world=1, group=None, broadcast=None):
    """Broadcast rank 0's complete verified objects; peers never open input files."""
    def validate():
        data = load_args(args)
        if hasattr(args, "exclude_generators") or hasattr(args, "episode_fraction"):
            data["filter"] = episode_filter(data["rows"], args)
        inherited = v2.continuation_recipe(data["theta0"], data["config"], data["snapshot"],
                                           args.updates if args.command == "run" else 1, seed_k=data["K"])
        weights, selected, extra = load_choices(data, inherited, weights=args.weights, modules=args.trainable_modules)
        extra.append(data["inspection"]["files"].relocation.path(args.config))
        data["input_validation"] = {"rank": 0, "dataset_sha256": data["dataset_sha256"],
                                    "hash_cache": cache_provenance()}
        if active_file_hash_cache() is not None:
            # Publish before peers enter model loading's metadata checks and
            # before a long training job can be interrupted.
            active_file_hash_cache().flush()
        return data, inherited, weights, selected, extra
    if world == 1:
        return validate()
    from baselines.common.fsdp_train import rank_zero_data
    return rank_zero_data(rank, world, validate, group, broadcast=broadcast)


def run(args):
    import torch
    import torch.distributed as dist
    from baselines.common.fsdp_train import train_loop, rank_zero_data
    from train.dataset import load_tokenizer
    from transformers import set_seed
    require(args.batch_examples == 16, "V0 requires B=16")
    require(args.updates > 0 and all(0 <= c <= args.updates for c in args.checkpoints), "Invalid updates/checkpoints")
    checkpoints = sorted({0, args.updates, *args.checkpoints})
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                          (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    require(world == (1 if args.dry_run else args.gpus), "Launch with the requested GPU rank count")
    if not args.dry_run:
        require(torch.cuda.is_available(), "Production requires allocated GPUs; use --dry-run on CPU")
        torch.cuda.set_device(local)
    device = torch.device("cpu" if args.dry_run else f"cuda:{local}")
    group = None
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    try:
        if world > 1:
            group = dist.new_group(backend="gloo", timeout=timedelta(hours=4))
        data, inherited, weights, selected, extra = validate_inputs(args, rank=rank, world=world, group=group)
        batches, mixture = batch_schedule(data["rows"], updates=args.updates, seed=args.seed, weights=weights,
                                          mixture=dict(zip(BUCKETS, args.bucket_probabilities)),
                                          filter_record=data.get("filter"))
        output = safe_output(args.output, data)
        byte_tokenizer = args.dry_run and args.dry_run_tokenizer == "bytes"
        tokenizer = v2.FakeTokenizer() if byte_tokenizer else load_tokenizer(data["snapshot"])
        examples = [v2.encode_example(row, tokenizer, max_length=inherited["recipe"]["max_length"]) for row in data["rows"]]
        scoring, vocabulary = v2.fake_scoring_examples(examples) if args.dry_run and not byte_tokenizer else (examples, None)
        counts = v2.accounting(examples, batches, world_size=world)
        exposures = exposure_counts(batches)
        for update in counts:
            counts[update].update(exposures[update])
        set_seed(args.seed + rank)
        model = v2.fake_model(len(vocabulary) if vocabulary else 512) if args.dry_run else v2.load_existing_adapter(
            data["snapshot"], data["theta0"], device, inherited["recipe"])
        original = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
        real_inherited = deepcopy(inherited)
        if args.dry_run:
            # Explicit synthetic mapping by A/B side; never claim these are real tensors.
            synthetic_selected = [n for n in original if selected is None or any(n in s.split(".") for s in selected)]
            actual = mask_modules(model, synthetic_selected)
            inherited.update(parameters=actual, trainable_modules=["fake_projection"],
                optimizer_state={"mode": "synthetic_fresh_AdamW", "path": None, "sha256": None})
        else:
            require(original == inherited["parameters"], "Trainable adapter tensors differ from theta0")
            actual = mask_modules(model, list(original) if selected is None else selected)
            inherited.update(parameters=actual, trainable_modules=sorted({n.split(".lora_", 1)[0] for n in actual}))
            if world > 1:
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        dist.broadcast(parameter.data, src=0)
        opt = inherited["optimizer_state"]
        saved = torch.load(opt["path"], map_location="cpu", weights_only=True) if opt["path"] else None
        optimizer_state = subset_optimizer_state(saved, list(original), list(actual))

        def record():
            output.mkdir(parents=True, exist_ok=False)
            effective = deepcopy(data["config"])
            effective["sft"] = deepcopy(inherited["sft"])
            effective["v0_training"] = {"arguments": vars(args), "inherited": real_inherited,
                "effective_bucket_probabilities": mixture, "schedule_sha256": digest(batches),
                "selected_real_tensors": selected, "actual_trainable_tensors": actual,
                "loss_normalization": "V2 per-row token mean; uniform row within weighted episode",
                "synthetic_mask_mapping": "selected real A/B sides -> fake A/B" if args.dry_run else None,
                "optimizer_subset": list(actual)}
            choices = {"parent_binding": data.get("parent_binding"),
                       "weight_inheritance": data.get("weight_inheritance"),
                       "input_validation": data["input_validation"],
                       "episode_weights_sha256": digest(weights), "buckets": bucket_summary(data, weights)}
            if "filter" in data:
                choices["filter"] = data["filter"]
            effective["v0_training"].update(choices)
            if args.dry_run:
                fake_base = output / "fake-base"
                fake_base.mkdir()
                write_json(fake_base / "config.json", {"synthetic": True})
                effective["models"]["student"] = {"requested_repo": "synthetic/v2-tiny-fake", "revision": "fake-only"}
                effective["sft"] = {"rank": 1, "alpha": 1, "dropout": 0, "targets": ["fake_projection"],
                    "packing": "turn", "base_snapshot": {"path": str(fake_base),
                    "metadata_sha256": {"config.json": file_hash(fake_base / "config.json")}}}
            prov = provenance(output, effective, command="v0_train", inputs=provenance_inputs(data, extra),
                details={"relocation": data["inspection"]["files"].report(),
                         "replay_deduplication": data["inspection"]["replay_deduplication"], **choices})
            manifest = {"schema": "v0-training-run/1", "status": "running", "dry_run": args.dry_run,
                **choices,
                "dataset_sha256": data["dataset_sha256"], "adapter_sha256": data["adapter_sha256"],
                "input_manifest": data["manifest"], "provenance_sha256": digest(prov),
                "schedule_sha256": digest(batches), "effective_bucket_probabilities": mixture,
                "encoded_examples_sha256": digest(examples), "tokenizer": {"synthetic": True,
                    "chat_template_sha256": digest(tokenizer.chat_template)} if byte_tokenizer
                    else {"synthetic": False, **tokenizer_identity(data["snapshot"], tokenizer)},
                "fake_scoring_vocabulary": vocabulary, "optimizer_state": opt,
                "scheduler_policy": inherited["scheduler_policy"], "trainable_parameter_count": sum(actual.values()),
                "actual_trainable_parameter_names": sorted(actual), "selected_real_tensors": selected,
                "unique_instances": len({r["task_id"] for r in data["rows"]}),
                "task_groups": len({r["group_id"] for r in data["rows"]}), "episodes": len(data["episodes"]),
                "K_trusted_seed_trajectories": data["K"], "seed_unique_instances": len(data["seed_task_ids"]),
                "teacher_calls_this_run": 0, "environment_transitions_this_run": 0,
                "training_exposure": 0, "planned_accounting": counts, "planned_checkpoints": checkpoints,
                "checkpoints": [{"update": 0, "status": "theta0_reference", "path": str(data["theta0"]), **counts[0]}]}
            write_json(output / "batches.json", batches, exclusive=True)
            write_json(output / "example_sources.json", [{"file": origin, **{k: row[k] for k in
                ("example_id", "source", "task_id", "group_id", "v0", "evidence")}}
                for origin, row in zip(data["origins"], data["rows"])], exclusive=True)
            write_json(output / "manifest.json", manifest, exclusive=True)
            return prov, manifest
        prov, manifest = rank_zero_data(rank, world, record, group)

        def save(current_model, state):
            update = state["updates"]
            require(state["examples_seen"] == counts[update]["examples_seen"] and
                    state["supervised_tokens_seen"] == counts[update]["supervised_target_tokens"] and
                    state["physical_policy_forward_tokens"] == counts[update]["processed_context_tokens"],
                    "Actual exposure differs from scheduled accounting")
            def publish():
                if update in checkpoints:
                    with torch.random.fork_rng(devices=[] if args.dry_run else [local]):
                        item = v2.publish_checkpoint(current_model, output, prov, state, counts[update], inherited,
                                                     dry_run=args.dry_run)
                    manifest["checkpoints"].append(item)
                manifest["realised_exposure"] = counts[update]
                manifest["synthetic_training_exposure" if args.dry_run else "training_exposure"] = state["examples_seen"]
                write_json(output / "manifest.json", manifest)
            rank_zero_data(rank, world, publish, group)

        recipe = inherited["recipe"]
        with nullcontext() if args.dry_run else torch.autocast("cuda", dtype=torch.bfloat16):
            state = train_loop(model, scoring, output, objective="sft", learning_rate=recipe["learning_rate"],
                scheduler=recipe["scheduler"], warmup_ratio=recipe["warmup_ratio"], warmup_steps=recipe["warmup_steps"],
                original_batch=16, seed=args.seed, weight_decay=recipe["weight_decay"], packing=recipe["packing"],
                scoring=recipe["scoring"], flex_kernel_options=recipe.get("flex_kernel_options"),
                rank=rank, world_size=world, device=device, replicated_lora=True,
                fixed_batches=batches, optimizer_state=optimizer_state, checkpoint_callback=save)
        if rank == 0:
            manifest.update(status="synthetic_complete" if args.dry_run else "complete", trainer_state=state,
                            accounting=counts[args.updates])
            manifest["artifacts_sha256"] = {n: file_hash(output / n) for n in
                ("provenance.json", "batches.json", "example_sources.json", "updates.jsonl")}
            write_json(output / "manifest.json", manifest)
            print(json.dumps({"status": manifest["status"], "updates": args.updates,
                              "trainable_parameter_count": sum(actual.values())}))
            return manifest
    finally:
        if world > 1:
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inputs", help="CPU-only verification without loading a model")
    inventory.add_argument("--check", required=True, action="store_true")
    runner = sub.add_parser("run")
    for command in (inventory, runner):
        add_inputs(command)
        command.add_argument("--weights")
        command.add_argument("--trainable-modules")
        # Suppressed defaults preserve the exact pre-filter provenance arguments.
        filters = command.add_mutually_exclusive_group()
        filters.add_argument("--exclude-generators", choices=GENERATORS[:-1], nargs="+", default=argparse.SUPPRESS,
                             help="Exclude non-D episodes from these generators")
        filters.add_argument("--episode-fraction", type=float, default=argparse.SUPPRESS,
                             help="Keep round(F * episodes), at least one, within each of R/G/S")
        command.add_argument("--subsample-seed", type=int, default=argparse.SUPPRESS)
        command.add_argument("--output", required=command is runner)
    runner.add_argument("--updates", type=int, required=True)
    runner.add_argument("--checkpoints", type=int, nargs="+", required=True)
    runner.add_argument("--batch-examples", type=int, default=16)
    runner.add_argument("--bucket-probabilities", type=float, nargs=4, default=list(DEFAULT_MIXTURE.values()),
                        metavar=("R", "G", "S", "D"))
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument("--gpus", type=int, choices=(1, 2), default=1)
    runner.add_argument("--dry-run", action="store_true")
    runner.add_argument("--dry-run-tokenizer", choices=("bytes", "snapshot"), default="bytes")
    values = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(values)
    if args.command == "run" and not args.dry_run and args.gpus == 2 and "RANK" not in os.environ:
        return subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                               str(Path(__file__).resolve()), *values], check=False).returncode
    try:
        with raw_input_cache(full_rehash=args.full_rehash, workers=args.hash_workers):
            return execute(args)
    except ValidationError as exc:
        print(f"V0 training refused: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"V0 training failed ({type(exc).__name__}); no result marked complete.", file=sys.stderr)
    return 1


def execute(args):
    # The scope covers provenance as well as checking. Distributed runs
    # establish their CPU broadcast group before rank 0 reads any inputs.
    filtered = hasattr(args, "exclude_generators") or hasattr(args, "episode_fraction")
    require(not filtered or (args.weights is None and args.trainable_modules is None),
            "Episode filters require uniform weights and all tensors; omit --weights and --trainable-modules")
    if hasattr(args, "episode_fraction"):
        require(0 < args.episode_fraction < 1, "--episode-fraction must be strictly between 0 and 1")
        require(hasattr(args, "subsample_seed"), "--episode-fraction requires --subsample-seed")
    else:
        require(not hasattr(args, "subsample_seed"), "--subsample-seed requires --episode-fraction")
    if args.command == "inputs":
        data, inherited, weights, selected, extra = validate_inputs(args)
        # Also validate numeric weight values without allocating a model.
        batch_schedule(data["rows"], updates=1, seed=0, weights=weights, filter_record=data.get("filter"))
        result = {"schema": "v0-training-inputs/1", "status": "verified", "rows": len(data["rows"]),
            "hash_cache": cache_provenance(),
            "buckets": bucket_summary(data, weights), "parent_binding": data.get("parent_binding"),
            "weight_inheritance": data.get("weight_inheritance"), "episode_weights_sha256": digest(weights),
            "replay_deduplication": data["inspection"]["replay_deduplication"],
            "episodes": len(data["episodes"]), "dataset_sha256": data["dataset_sha256"],
            "K_trusted_seed_trajectories": data["K"], "seed_unique_instances": len(data["seed_task_ids"]),
            "trainable_parameter_count": sum(inherited["parameters"][n] for n in
                                             (selected if selected is not None else inherited["parameters"]))}
        if "filter" in data:
            result["filter"] = {k: v for k, v in data["filter"].items()
                                if k not in ("kept_episode_ids", "dropped_episode_ids")}
        if args.output:
            destination = safe_output(args.output, data)
            write_json(destination, result, exclusive=True)
        else:
            print(json.dumps(result, indent=2))
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
