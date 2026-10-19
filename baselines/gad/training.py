"""Serialized vLLM sampling / FSDP2 updates, with persistent optimizer state.

One batch is resampled from its current generator before any G/D update.
Separate workers avoid holding vLLM, generator, discriminator and reference
on the GPU simultaneously. All attempts remain hash-bound and auditable.
"""
import os
from pathlib import Path

from baselines.common.method_data import read
from baselines.common.pipeline import stage_run, served, run_command, publish_models
from baselines.common.schedule import dose, epoch_batches
from baselines.gad.data import load_rows
from sed.manifest import frozen_write
from sed.storage import digest, file_hash, write_json


def round_plan(n, round_index):
    if round_index not in range(3):
        raise ValueError("GAD requires rounds 0, 1, 2")
    plan = dose(n, 256, epochs=1)
    batches = list(epoch_batches(n, plan["b_prime"], epochs=1, seed=round_index))
    jobs = []
    if round_index == 0:
        jobs += [{"kind": "discriminator", "indices": batches[i % len(batches)]["indices"]} for i in range(10)]
    jobs += [{"kind": "ce" if round_index == 0 else "grpo", "indices": b["indices"]} for b in batches]
    for index, job in enumerate(jobs):
        job.update(round=round_index, step=index, seed=round_index * 1_000_000 + index * 256,
                   N=n, b_prime=plan["b_prime"])
    return jobs


def worker_command(args, *, role, data, plan, groups, snapshot, resume, reference, rewards, output):
    command = [args.train_python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node",
        str(args.gpus), "-m", "baselines.gad.worker", "--role", role, "--config", args.config,
        "--cell", args.output, "--data", str(data), "--plan", str(plan), "--groups", str(groups),
        "--snapshot", str(snapshot), "--output", str(output)]
    for key, value in (("resume", resume), ("reference", reference), ("rewards", rewards)):
        if value is not None:
            command += ["--" + key, str(value)]
    return command


def check_history_policy(data, snapshot):
    source = Path(read(Path(data).parent / "provenance.json")["details"]["exploration"])
    provenance = read(source / "provenance.json")
    serving = provenance["config"]["server_provenance"]["config"]["server"]
    exploration = provenance["config"]["exploration"]
    if (Path(serving["model_path"]).resolve() != Path(snapshot).resolve() or not serving["batch_invariant"]
            or exploration["decoding"] != "greedy" or exploration["student"]["enable_thinking"] is not False
            or exploration["student"]["max_tokens"] != 1024):
        raise ValueError("GAD histories were visited by a different round policy/harness")


def train_round(args, identity, config, *, data, snapshot, previous, round_index, output):
    rows = load_rows(data, config, verified=args.verified, identity=identity, round_index=round_index)
    check_history_policy(data, snapshot)
    root = Path(args.output) / "updates" / f"round-{round_index}"
    actor = previous["actor"] if previous else None
    critic = previous["critic"] if previous else None
    reference = previous["reference"] if previous else None
    current = Path(snapshot)
    for job in round_plan(len(rows), round_index):
        name = f"{job['step']:05d}"
        plan_path = root / f"plan-{name}.json"
        frozen_write(plan_path, job)
        key = digest([identity, job, file_hash(data), str(current), actor, critic, reference])
        def sample(attempt):
            with served(args, current, attempt) as endpoint:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""
                run_command([args.train_python, "-m", "baselines.gad.sampling", "--config", args.config,
                    "--cell", args.output, "--data", str(data), "--plan", str(plan_path),
                    "--snapshot", str(current), "--endpoint", endpoint,
                    "--server-provenance", str(attempt / "server/provenance.json"),
                    "--output", str(attempt / "output")], attempt / "sample.log", env=env)
        sampled = stage_run(root, "sample-" + name, key, sample)
        groups = sampled / "groups.json"
        trained_d = stage_run(root, "critic-" + name, digest([key, str(sampled)]),
            lambda a: run_command(worker_command(args, role="critic", data=data, plan=plan_path,
                groups=groups, snapshot=current, resume=critic, reference=None, rewards=None,
                output=a / "output"), a / "train.log"))
        critic = str(trained_d)
        if job["kind"] != "discriminator":
            trained_g = stage_run(root, "actor-" + name, digest([key, str(trained_d)]),
                lambda a: run_command(worker_command(args, role="actor", data=data, plan=plan_path,
                    groups=groups, snapshot=current, resume=actor, reference=reference,
                    rewards=trained_d / "rewards.json", output=a / "output"), a / "train.log"))
            actor, current = str(trained_g), trained_g / "checkpoint"
            publish_models(args.output, [{"name": f"round-{round_index}-step-{job['step']}",
                "checkpoint": str(current), "method": "gad", "round": round_index,
                "objective": job["kind"], "checkpoint_state_sha256": file_hash(current / "training_state.json")}])
    if round_index == 0:
        reference = str(current)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "result.json", {"round": round_index, "actor": actor, "critic": critic,
        "checkpoint": str(current), "reference": reference, "N_histories": len(rows),
        "plan": round_plan(len(rows), round_index), "selection": None})
