"""CoA full fine-tuning using shared FSDP/export infrastructure."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

from baselines.coa import RECIPE
from baselines.coa.data import read, support, training_examples
from baselines.coa.protocol import encode_episode
from baselines.coa.rl import rewards, train_update
from baselines.coa.rollout import checked_groups
from baselines.common import fsdp_train as shared
from sed.config import load_config
from sed.provenance import write_provenance
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, fsync_dir, write_json
from train.dataset import load_tokenizer


def resume_state(model, optimizer, directory, *, save=False):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
    options = StateDictOptions(ignore_frozen_params=True, strict=False)
    weights, moments = get_state_dict(model, optimizer, options=options)
    state = {"model": weights, "optimizer": moments}
    if save:
        dcp.save(state, checkpoint_id=str(directory))
    else:
        dcp.load(state, checkpoint_id=str(directory))
        set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"], options=options)


def rl_examples(path, config, tokenizer, verified, snapshot):
    plan = read(path)
    listing, _ = support(config, verified)
    if plan["support_sha256"] != digest(listing) or plan["snapshot"] != str(Path(snapshot).resolve()):
        raise ValueError("DAPO rollout support/policy changed")
    if plan["checkpoint_state_sha256"] != file_hash(Path(snapshot) / "training_state.json"):
        raise ValueError("DAPO behavior checkpoint changed")
    loaded, result = {}, []
    for source in plan["selected"]:
        root = source["root"]
        if root not in loaded:
            if artifact_hashes(root) != plan["artifacts"][root]:
                raise ValueError("DAPO rollout batch changed")
            server = read(Path(root) / "provenance.json")["config"]["server_provenance"]["config"]["server"]
            if (Path(server["model_path"]).resolve() != Path(snapshot).resolve()
                    or server["batch_invariant"] is not True
                    or server["training_provenance_sha256"] != file_hash(Path(snapshot).parent / "provenance.json")):
                raise ValueError("DAPO sampled a different behavior model")
            loaded[root] = checked_groups(root, listing)
        task_id, episodes = loaded[root][source["group"]]
        result.append({"task_id": task_id, "rewards": rewards(episodes),
                       "examples": [encode_episode(e, tokenizer, rl=True) for e in episodes]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "snapshot", "base-snapshot", "verified", "identity", "data", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--kind", choices=("sft", "rl"), required=True)
    parser.add_argument("--update", type=int, default=0)
    args = parser.parse_args(argv)
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    if not torch.cuda.is_available():
        raise RuntimeError("Production CoA training requires allocated GPUs; unit tests use CPU models")
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    started = time.monotonic()
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(hours=4))
        torch.manual_seed(0)
        def prepare():
            config, identity = load_config(args.config), read(args.identity)
            if (identity["method"] != "coa" or identity["recipe"]["parameters"] != RECIPE
                    or identity["gpus"] != world or identity["config_sha256"] != digest(config)
                    or identity["verified"] != str(Path(args.verified).resolve())
                    or identity["snapshot"] != str(Path(args.base_snapshot).resolve())
                    or identity["snapshot_artifact_sha256"] != artifact_hashes(args.base_snapshot)):
                raise ValueError("CoA training identity changed")
            tokenizer = load_tokenizer(args.snapshot)
            if args.kind == "sft":
                if Path(args.snapshot).resolve() != Path(args.base_snapshot).resolve() or args.update:
                    raise ValueError("CoA SFT must initialize from the configured base")
                examples = training_examples(args.data, config, tokenizer, args.verified)
            else:
                state = shared.validate_checkpoint(args.snapshot, config["models"]["student"], base_snapshot=args.base_snapshot)
                if state.get("method") != "coa" or (args.update == 0 and state.get("kind") != "sft"):
                    raise ValueError("DAPO must initialize from CoA SFT")
                if args.update and (state.get("kind") != "rl" or state["end_update"] != args.update
                        or artifact_hashes(Path(args.snapshot).parent / "resume") != state["resume_artifact_sha256"]):
                    raise ValueError("DAPO optimizer resume changed")
                examples = rl_examples(args.data, config, tokenizer, args.verified, args.snapshot)
            if not examples:
                raise ValueError("Empty CoA training data; no synthetic fallback")
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=False)
            effective = {**config, "baseline_training": {"method": "coa", "recipe": RECIPE,
                "args": vars(args), "examples_sha256": digest(examples), "world_size": world,
                "snapshot_artifact_sha256": artifact_hashes(args.snapshot), "master_dtype": "float32",
                "compute_dtype": "bfloat16", "packing": "turn", "observation_mask": "all user observations and tags"}}
            provenance = write_provenance(output, command="coa_fsdp_train", config=effective,
                inputs=[args.config, args.identity, args.data, Path(args.verified) / "manifest.json"])
            return examples, provenance
        examples, provenance = shared.rank_zero_data(rank, world, prepare, group)
        model = shared.shard_model(shared.prepare_model(args.snapshot), mesh=init_device_mesh("cuda", (world,)))
        output = Path(args.output)
        if args.kind == "sft":
            state = shared.train_loop(model, examples, output, objective="sft", **RECIPE["sft"],
                packing="turn", seed=0, rank=rank, world_size=world, device=device)
        else:
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6,
                betas=(.9, .999), eps=1e-8, weight_decay=.1, foreach=False, fused=False)
            if args.update:
                resume_state(model, optimizer, Path(args.snapshot).parent / "resume")
            # Old logprobs must use exactly the exported bf16 behavior weights.
            # Loading DCP restores fp32 masters; score their bf16 casts as used
            # in serving. Keep moments across rounds, never restart Adam.
            state = train_update(model, optimizer, examples, output, update=args.update,
                                 rank=rank, world_size=world, device=device)
            resume_state(model, optimizer, output / "resume", save=True)
        saved = shared.export_checkpoint(model, output / ".checkpoint.partial", args.base_snapshot, rank=rank)
        dist.barrier()
        if rank == 0:
            state.update(saved, method="coa", kind=args.kind, final=True,
                provenance_sha256=digest(provenance), wall_time_seconds=time.monotonic() - started,
                resume_artifact_sha256=artifact_hashes(output / "resume") if args.kind == "rl" else None)
            write_json(output / ".checkpoint.partial/training_state.json", state)
            os.rename(output / ".checkpoint.partial", output / "checkpoint")
            fsync_dir(output)
            write_json(output / "result.json", {"checkpoint": "checkpoint", **state})
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
