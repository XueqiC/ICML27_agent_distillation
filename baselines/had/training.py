"""HAD objective driver using the existing FSDP2 model/scorer/export machinery."""
from datetime import timedelta
import json
import os
from pathlib import Path
import time

from baselines.common import fsdp_train as shared
from baselines.had import RECIPE
from baselines.had.data import encode_state, load_rows
from baselines.had.objective import train_round
from sed.config import load_config
from sed.provenance import write_provenance
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, fsync_dir, write_json
from train.dataset import load_tokenizer


def optimizer_for(model):
    import torch
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False, fused=False)


def resume_state(model, optimizer, directory, *, save=False):
    """DCP preserves fp32 masters and Adam moments, without bf16 round trips."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
    options = StateDictOptions(ignore_frozen_params=True, strict=False)
    weights, moments = get_state_dict(model, optimizer, options=options)
    state = {"model": weights, "optimizer": moments}
    if save:
        dcp.save(state, checkpoint_id=str(directory))
    else:
        dcp.load(state, checkpoint_id=str(directory))
        set_state_dict(model, optimizer, model_state_dict=state["model"],
                       optim_state_dict=state["optimizer"], options=options)


def run(args):
    started = time.monotonic()
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    if not torch.cuda.is_available():
        raise RuntimeError("Production FSDP requires allocated GPUs; test train_round with CPU stubs")
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.cuda.reset_peak_memory_stats(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(hours=4))
        torch.manual_seed(0)
        def prepare():
            config = load_config(args.config)
            identity = json.loads(Path(args.had_identity).read_text())
            if (identity["method"] != "had" or identity["recipe"] != RECIPE
                    or identity["config_sha256"] != digest(config)
                    or identity["gpus"] != world):
                raise ValueError("HAD training identity/config/world changed")
            if args.had_round:
                shared.validate_checkpoint(args.snapshot, config["models"]["student"], base_snapshot=args.base_snapshot)
                previous = json.loads((Path(args.snapshot).parent / "result.json").read_text())
                resume = Path(args.snapshot).parent / "resume"
                if (previous["end_update"] != args.start_update or previous["policy_version"] != args.had_round - 1
                        or artifact_hashes(resume) != previous["resume_artifact_sha256"]):
                    raise ValueError("HAD optimizer resume identity changed")
            elif args.start_update != 0 or Path(args.snapshot).resolve() != Path(args.base_snapshot).resolve():
                raise ValueError("HAD round zero must start from configured base")
            tokenizer = load_tokenizer(args.snapshot)
            labels_provenance = json.loads((Path(args.had_data).parent / "provenance.json").read_text())
            rollout = Path(labels_provenance["details"]["exploration"])
            serving = json.loads((rollout / "provenance.json").read_text())["config"]["server_provenance"]
            server = serving["config"]["server"]
            if (Path(server["model_path"]).resolve() != Path(args.snapshot).resolve()
                    or server["batch_invariant"] is not True
                    or (args.had_round and server["training_provenance_sha256"] !=
                        file_hash(Path(args.snapshot).parent / "provenance.json"))):
                raise ValueError("HAD labels came from a different rollout policy")
            rows = load_rows(args.had_data, config, verified=args.verified,
                             identity=identity, round_index=args.had_round)
            examples = [encode_state(row, tokenizer) for row in rows]
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=False)
            effective = {**config, "baseline_training": {**RECIPE, "world_size": world,
                "master_dtype": "float32", "compute_dtype": "bfloat16", "attention": "sdpa",
                "resume": "DCP fp32 masters and Adam moments", "examples_sha256": digest(examples),
                "snapshot_artifact_sha256": artifact_hashes(args.snapshot), "args": vars(args)}}
            provenance = write_provenance(output, command="had_fsdp_train", config=effective,
                inputs=[args.config, args.had_identity, args.had_data])
            return examples, provenance
        examples, provenance = shared.rank_zero_data(rank, world, prepare, group)
        model = shared.shard_model(shared.prepare_model(args.snapshot), mesh=init_device_mesh("cuda", (world,)))
        optimizer = optimizer_for(model)
        if args.had_round:
            resume_state(model, optimizer, Path(args.snapshot).parent / "resume")
        output = Path(args.output)
        result = train_round(model, optimizer, examples, output, version=args.had_round,
            start_update=args.start_update, rank=rank, world_size=world, device=device)
        resume_state(model, optimizer, output / "resume", save=True)
        saved = shared.export_checkpoint(model, output / ".checkpoint.partial", args.base_snapshot, rank=rank)
        torch.cuda.synchronize(device)
        measurements = [None] * world
        dist.all_gather_object(measurements, {"rank": rank, "wall_time_seconds": time.monotonic() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        dist.barrier()
        if rank == 0:
            result.update(saved, resume_artifact_sha256=artifact_hashes(output / "resume"),
                          provenance_sha256=digest(provenance), final=True, method="had",
                          measurements_per_rank=measurements,
                          wall_time_seconds=max(r["wall_time_seconds"] for r in measurements))
            write_json(output / ".checkpoint.partial/training_state.json", result)
            os.rename(output / ".checkpoint.partial", output / "checkpoint")
            fsync_dir(output)
            write_json(output / "result.json", {"checkpoint": "checkpoint", **result})
        dist.barrier()
    finally:
        dist.destroy_process_group()
