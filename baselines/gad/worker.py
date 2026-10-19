"""One exact global GAD optimizer update using existing FSDP2 machinery."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import time

from baselines.common import fsdp_train as shared
from baselines.common.method_data import read
from baselines.common.schedule import rank_slots
from baselines.gad import RECIPE
from baselines.gad.data import encode_state, load_rows
from baselines.gad.training import round_plan
from baselines.had.training import resume_state
from sed.config import load_config
from sed.provenance import write_provenance
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, fsync_dir, JsonlLedger, write_json
from train.dataset import load_tokenizer


def checked_inputs(args, config, world):
    identity, plan = read(Path(args.cell) / "identity.json"), read(args.plan)
    if (identity["method"] != "gad" or identity["recipe"]["algorithm"] != RECIPE
            or identity["config_sha256"] != digest(config) or identity["gpus"] != world):
        raise ValueError("GAD training cell/config/world changed")
    rows = load_rows(args.data, config, verified=identity["verified"], identity=identity, round_index=plan["round"])
    if plan != round_plan(len(rows), plan["round"])[plan["step"]]:
        raise ValueError("GAD optimizer batch changed")
    path = Path(args.groups)
    sampled, provenance = read(path), read(path.parent / "provenance.json")
    detail = provenance["details"]
    if (sampled["provenance_sha256"] != digest(provenance) or detail["plan"] != plan
            or detail["identity_sha256"] != digest(identity)
            or Path(detail["snapshot"]).resolve() != Path(args.snapshot).resolve()
            or digest(sampled["groups"]) != sampled["groups_sha256"]):
        raise ValueError("Stale/changed GAD samples")
    for source, expected in provenance["inputs"].items():
        if file_hash(source) != expected:
            raise ValueError("GAD sampling source changed")
    groups = sampled["groups"]
    if len(groups) != len(plan["indices"]):
        raise ValueError("GAD sample batch size changed")
    tokenizer = load_tokenizer(identity["snapshot"])
    for index, group in zip(plan["indices"], groups):
        teacher = encode_state(rows[index], tokenizer)
        if (group["state_id"] != rows[index]["state_id"] or group["teacher"] != teacher
                or len(group["samples"]) != 8):
            raise ValueError("GAD teacher/prompt/group changed")
        n = teacher["prompt_tokens"]
        for sample in group["samples"]:
            if (sample["prompt_tokens"] != n or sample["input_ids"][:n] != teacher["input_ids"][:n]
                    or not 1 <= len(sample["input_ids"]) - n <= 1024
                    or len(sample["input_ids"]) > 32768):
                raise ValueError("GAD generated action token boundary changed")
    if Path(args.snapshot).resolve() != Path(identity["snapshot"]).resolve():
        shared.validate_checkpoint(args.snapshot, config["models"]["student"], base_snapshot=identity["snapshot"])
    previous = None
    if args.resume:
        previous = read(Path(args.resume) / "result.json")
        if (previous["role"] != args.role or previous["identity_sha256"] != digest(identity)
                or artifact_hashes(Path(args.resume) / "resume") != previous["resume_artifact_sha256"]):
            raise ValueError("GAD optimizer resume changed")
        if args.role == "actor" and Path(args.snapshot).resolve() != (Path(args.resume) / "checkpoint").resolve():
            raise ValueError("Actor optimizer and sampled policy differ")
        previous_plan = previous["plan"]
        if plan["step"] > (10 if args.role == "actor" and plan["round"] == 0 else 0):
            expected_previous = round_plan(len(rows), plan["round"])[plan["step"] - 1]
        else:
            expected_previous = round_plan(previous_plan["N"], plan["round"] - 1)[-1]
        if previous_plan != expected_previous:
            raise ValueError("GAD optimizer must resume the immediately preceding update")
    elif args.role == "actor" and (plan["round"] != 0 or plan["step"] != 10):
        raise ValueError("Actor optimizer reset is forbidden")
    elif args.role == "critic" and (plan["round"] != 0 or plan["step"] != 0):
        raise ValueError("Discriminator optimizer reset is forbidden")
    if not args.resume and Path(args.snapshot).resolve() != Path(identity["snapshot"]).resolve():
        raise ValueError("Initial GAD update must use the configured base generator")
    if plan["kind"] == "grpo" and args.role == "actor":
        expected = read(Path(args.cell) / "stages/train-0/complete.json")
        warm = Path(args.cell) / "stages/train-0" / expected["attempt"] / "output/result.json"
        if not args.reference or str(Path(args.reference).resolve()) != read(warm)["reference"]:
            raise ValueError("GRPO reference must remain the frozen warmup generator")
        shared.validate_checkpoint(args.reference, config["models"]["student"], base_snapshot=identity["snapshot"])
    return identity, plan, groups, previous


def update(model, optimizer, groups, *, role, kind, device, rank=0, world=1, reference=None, rewards=None):
    """CPU-testable update, exact global scaling even with b' smaller than ranks."""
    import torch
    from baselines.gad.objective import (bradley_terry, group_advantages, grpo_terms,
                                         score, token_logps, optimizer_step)
    for candidate in (model, reference):
        if candidate is not None:
            for module in candidate.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
    model.train()
    if reference is not None:
        reference.eval()
        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError("GRPO reference must be frozen")
    optimizer.zero_grad(set_to_none=True)
    pairs = [(group["teacher"], sample) for group in groups for sample in group["samples"]]
    targets = [g["teacher"] for g in groups] if kind == "ce" and role == "actor" else [s for _, s in pairs]
    total_tokens = sum(len(r["input_ids"]) - r["prompt_tokens"] for r in targets)
    values = torch.zeros(len(pairs), device=device)
    advantage = None
    if role == "actor" and kind == "grpo":
        if reference is None:
            raise ValueError("GRPO requires a frozen reference")
        advantage = group_advantages(torch.tensor(rewards, device=device)).flatten()
        if advantage.numel() != len(pairs):
            raise ValueError("Reward group size changed")
    losses = torch.zeros((), device=device)
    physical = torch.zeros(2, dtype=torch.long, device=device)
    for slot in rank_slots(list(range(len(pairs) if role == "critic" else len(targets))), world):
        active = slot[rank] is not None
        index = slot[rank] if active else 0
        if role == "critic":
            teacher, student = pairs[index]
            physical += torch.tensor([2, len(teacher["input_ids"]) + len(student["input_ids"])], device=device)
            student_score = score(model, student, device)
            teacher_score = score(model, teacher, device)
            if active:
                values[index] = student_score.detach()
            loss = bradley_terry(teacher_score, student_score) / len(pairs)
        elif kind == "ce":
            physical += torch.tensor([1, len(targets[index]["input_ids"])], device=device)
            loss = -token_logps(model, targets[index], device, temperature=1.0).sum() / total_tokens
        elif kind == "grpo":
            physical += torch.tensor([3, 3 * len(targets[index]["input_ids"])], device=device)
            with torch.no_grad():
                old = token_logps(model, targets[index], device)
                ref = token_logps(reference, targets[index], device)
            logp = token_logps(model, targets[index], device)
            loss = grpo_terms(logp, old, ref, advantage[index]).sum() / total_tokens
        else:
            raise ValueError("Invalid GAD actor update")
        (loss * (world if active else 0)).backward()
        if active:
            losses += loss.detach()
    if world > 1:
        import torch.distributed as dist
        dist.all_reduce(values)
        dist.all_reduce(losses)
        dist.all_reduce(physical)
    # Rewards were measured before this update, as in ray_trainer.py.
    grad_norm = optimizer_step(model, optimizer)
    return {"loss": float(losses), "grad_norm": grad_norm, "examples_seen": len(targets),
        "logical_forward_tokens": sum(len(t["input_ids"]) + len(s["input_ids"]) for t, s in pairs)
            if role == "critic" else sum(len(r["input_ids"]) for r in targets) * (3 if kind == "grpo" else 1),
        "physical_forwards": int(physical[0]), "physical_forward_tokens": int(physical[1]),
        "forward_accounting": "Includes old/reference scoring and rank padding; excludes activation recomputation",
        "supervised_tokens_seen": total_tokens if role == "actor" else sum(
            len(t["input_ids"]) - t["prompt_tokens"] + len(s["input_ids"]) - s["prompt_tokens"] for t, s in pairs),
        "rewards": values.reshape(len(groups), 8).cpu().tolist() if role == "critic" else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("actor", "critic"), required=True)
    for key in ("config", "cell", "data", "plan", "groups", "snapshot", "output"):
        parser.add_argument("--" + key, required=True)
    for key in ("resume", "reference", "rewards"):
        parser.add_argument("--" + key)
    args = parser.parse_args(argv)
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from baselines.gad.objective import Discriminator, optimizer_for
    if not torch.cuda.is_available():
        raise RuntimeError("Production GAD uses allocated GPUs; CPU tests call update with tiny models")
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    started = time.monotonic()
    try:
        config = load_config(args.config)
        group = dist.new_group(backend="gloo", timeout=timedelta(hours=4))
        identity, plan, groups, previous = shared.rank_zero_data(rank, world,
            lambda: checked_inputs(args, config, world), group)
        output = Path(args.output)
        def prepare():
            output.mkdir(parents=True, exist_ok=False)
            return write_provenance(output, command="gad_" + args.role, config={**config, "gad": RECIPE},
                inputs=[args.config, args.plan, args.data, args.groups, Path(args.cell) / "identity.json",
                        *([args.rewards] if args.rewards else [])],
                details={"snapshot": args.snapshot, "resume": args.resume, "reference": args.reference,
                    "plan": plan, "master_dtype": "float32", "compute_dtype": "bfloat16",
                    "world_size": world, "identity_sha256": digest(identity)})
        provenance = shared.rank_zero_data(rank, world, prepare, group)
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats(device)
        mesh = init_device_mesh("cuda", (world,))
        model = shared.prepare_model(identity["snapshot"] if args.role == "critic" else args.snapshot)
        if args.role == "critic":
            model = Discriminator(model, model.config.get_text_config().hidden_size)
        model = shared.shard_model(model, mesh=mesh)
        optimizer = optimizer_for(model)
        if args.resume:
            resume_state(model, optimizer, Path(args.resume) / "resume")
        reference, rewards = None, None
        if args.role == "actor" and plan["kind"] == "grpo":
            reference = shared.shard_model(shared.prepare_model(args.reference, reference=True), mesh=mesh, reference=True)
            record = read(args.rewards)
            critic_state = read(Path(args.rewards).parent / "result.json")
            if (record["groups_sha256"] != file_hash(args.groups) or record["plan"] != plan
                    or record["timing"] != "before_discriminator_update"
                    or critic_state["identity_sha256"] != digest(identity) or critic_state["plan"] != plan):
                raise ValueError("Discriminator rewards differ from current groups")
            rewards = record["rewards"]
        result = update(model, optimizer, groups, role=args.role, kind=plan["kind"], device=device,
                        rank=rank, world=world, reference=reference, rewards=rewards)
        if rank == 0:
            JsonlLedger(output / "updates.jsonl").append({**result, "plan": plan})
        resume_state(model, optimizer, output / "resume", save=True)
        saved = shared.export_checkpoint(model, output / ".checkpoint.partial", identity["snapshot"], rank=rank) if args.role == "actor" else {}
        measurements = [None] * world
        dist.all_gather_object(measurements, {"rank": rank, "wall_time_seconds": time.monotonic() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        dist.barrier()
        if rank == 0:
            result.update(saved, role=args.role, method="gad", plan=plan, final=True,
                identity_sha256=digest(identity), provenance_sha256=digest(provenance),
                resume_artifact_sha256=artifact_hashes(output / "resume"),
                optimizer_steps=(previous["optimizer_steps"] if previous else 0) + 1,
                measurements_per_rank=measurements,
                wall_time_seconds=time.monotonic() - started)
            if args.role == "actor":
                write_json(output / ".checkpoint.partial/training_state.json", result)
                os.rename(output / ".checkpoint.partial", output / "checkpoint")
                fsync_dir(output)
            else:
                write_json(output / "rewards.json", {"groups_sha256": file_hash(args.groups),
                    "plan": plan, "rewards": result.pop("rewards"), "timing": "before_discriminator_update"})
            write_json(output / "result.json", result)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
