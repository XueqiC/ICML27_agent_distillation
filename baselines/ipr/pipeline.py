"""Resumable SFT -> expert MC -> (explore -> agent MC -> pairs -> DPO+NLL) x6."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys

from baselines.common.dose_control import validate_evaluation_init, materialize_init, model_record
from baselines.common.pipeline import batch_invariant, prepare, update_disclosure as common_disclosure
from baselines.common.schedule import dose
from baselines.eto.pipeline import stage_run, run_command, served, publish_models, evaluate_pipeline
from baselines.ipr.pairs import prepare_pairs
from baselines.ipr.rollouts import read
from sed.config import load_config
from sed.provenance import versions
from sed.runner import episode_metrics
from sed.storage import digest, locked, write_json

RECIPE = {"training_seed": 0, "sampling_seed": 0, "epochs": 3, "max_length": 32768,
    "adaptation": "full", "iterations": 6, "batch_invariant": True,
    "sft": {"learning_rate": 2e-5, "scheduler": "cosine", "warmup_ratio": .03,
            "original_batch": 48, "weight_decay": 0.0, "packing": "turn"},
    "ipr": {"learning_rate": 3e-6, "scheduler": "constant_with_warmup", "warmup_ratio": .1,
            "original_batch": 48, "weight_decay": 0.0, "beta": .1, "scoring": "shared_history"},
    "reference": "fixed_sft", "nll_weight": 1, "mc_samples": 5,
    "mc": {"temperature": 1, "top_p": 1, "top_k": -1, "max_tokens": 512},
    "exploration": {"temperature": 0, "max_tokens": 1024},
    "pairs": {"global_traj": True, "local_traj": True, "step_action": False,
              "traj_threshold": .01, "step_threshold": .01}}


def training_command(args, *, objective, snapshot, output, reference=None, pairs=None):
    command = [args.train_python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.gpus),
        "-m", "baselines.common.fsdp_train", "--config", args.config, "--objective", objective,
        "--snapshot", str(snapshot), "--base-snapshot", args.snapshot, "--output", str(output),
        "--epochs", "3", "--seed", "0", "--max-length", "32768"]
    for key, value in RECIPE[objective].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    if objective == "sft":
        command += ["--verified", args.verified]
    else:
        command += ["--reference", str(reference), "--pairs", str(pairs)]
    return command


def rollout_stage(args, snapshot, attempt, *, kind, iteration, exploration=None):
    with served(args, snapshot, attempt) as endpoint:
        command = [args.env_python, "-m", "baselines.ipr.rollouts", "--config", args.config,
            "--verified", args.verified, "--kind", kind, "--iteration", str(iteration), "--sampling-seed", "0",
            "--endpoint", endpoint, "--model-name", args.model_name, "--workers", str(args.workers),
            "--server-provenance", str(attempt / "server/provenance.json"), "--output", str(attempt / "output")]
        if exploration:
            command += ["--exploration", str(exploration)]
        env = os.environ.copy()  # Inherited privately, never serialized.
        env["CUDA_VISIBLE_DEVICES"] = ""
        run_command(command, attempt / "rollouts.log", env=env)


def train_pipeline(args, identity, config):
    output, root_identity = Path(args.output), digest(identity)
    def train(name, snapshot, objective, reference=None, pairs=None):
        return stage_run(output, name, digest([root_identity, name, str(snapshot), str(reference), str(pairs)]),
            lambda a: run_command(training_command(args, objective=objective, snapshot=snapshot,
                reference=reference, pairs=pairs, output=a / "output"), a / "train.log"))
    sft = (materialize_init(args, identity) if "init" in identity
           else train("train-sft", args.snapshot, "sft") / "checkpoint")
    models = [model_record(identity, "sft", sft)]
    publish_models(output, models)
    if args.through == "sft":
        return
    expert_mc = stage_run(output, "expert-mc", digest([root_identity, str(sft)]),
        lambda a: rollout_stage(args, sft, a, kind="expert-mc", iteration=0))
    if args.through == "expert-mc":
        return
    current = sft
    for iteration in range(1, 7):
        exploration = stage_run(output, f"explore-{iteration}", digest([root_identity, iteration, str(current)]),
            lambda a: rollout_stage(args, current, a, kind="explore", iteration=iteration))
        if args.through == f"explore-{iteration}":
            return
        agent_mc = stage_run(output, f"agent-mc-{iteration}", digest([root_identity, iteration, str(sft), str(exploration)]),
            lambda a: rollout_stage(args, sft, a, kind="agent-mc", iteration=iteration, exploration=exploration))
        if args.through == f"agent-mc-{iteration}":
            return
        pairs = stage_run(output, f"pairs-{iteration}", digest([root_identity, iteration, str(expert_mc), str(agent_mc)]),
            lambda a: prepare_pairs(args.verified, exploration, expert_mc, agent_mc, a / "output", config))
        report = read(pairs / "report.json")
        record = {"iteration": iteration, **report, "init": str(current), "reference": str(sft),
            "dose": dose(report["P"], 48) if report["P"] else None,
            "carry_forward": report["P"] == 0}
        write_json(output / f"iteration-{iteration}.json", record)
        print(json.dumps({"iteration": iteration, "P": report["P"], "P_by_kind": report["P_by_kind"]}), flush=True)
        if args.through == f"pairs-{iteration}":
            return
        if report["P"]:
            current = train(f"train-dpo-{iteration}", current, "ipr", sft, pairs / "pairs.json") / "checkpoint"
        else:
            def carry(attempt):
                (attempt / "output").mkdir()
                write_json(attempt / "output/result.json", {"status": "carried_forward", "P": 0,
                    "checkpoint": str(current), "examples_seen": 0, "supervised_tokens_seen": 0,
                    "logical_forward_tokens": 0})
            stage_run(output, f"train-dpo-{iteration}", digest([root_identity, iteration, str(current), "P=0"]), carry)
        models.append(model_record(identity, f"iter-{iteration}", current, carried_forward=report["P"] == 0))
        publish_models(output, models)
        write_json(output / f"iteration-{iteration}.json", {**record, "checkpoint": str(current)})
        update_disclosure(output, identity)
        if args.through == f"dpo-{iteration}":
            return
    write_json(output / "result.json", {"status": "complete", "final_checkpoint": str(current), "models": models})


def update_disclosure(output, identity):
    common_disclosure(output, identity)
    output = Path(output)
    report = read(output / "disclosure.json")
    attempts = []
    for stage in sorted((output / "stages").iterdir()) if (output / "stages").exists() else []:
        if not stage.name.startswith(("explore-", "expert-mc", "agent-mc-")):
            continue
        for attempt in sorted(stage.glob("attempt-*")):
            episodes = [read(p) for p in sorted((attempt / "output/rollouts").glob("*/episode/episode.json"))]
            metrics = [episode_metrics(e) for e in episodes]
            calls = []
            malformed = False
            for path in sorted((attempt / "output/rollouts").glob("*/calls/calls.jsonl")):
                for line in path.read_text().splitlines():
                    try:
                        calls.append(json.loads(line))
                    except json.JSONDecodeError:
                        malformed = True
            returned = [c for c in calls if c["status"] == "completed"]
            attempted = [c for c in calls if c["status"] == "started"]
            prefixes = sum(sum(s["environment_steps"] for s in e["steps"][:e.get("ipr_branch", {}).get("prefix", 0)])
                           for e in episodes)
            attempts.append({"path": str(attempt.relative_to(output)), "kind": stage.name,
                "complete": (stage / "complete.json").exists() and read(stage / "complete.json")["attempt"] == attempt.name,
                "environment_episodes": len(episodes), "environment_resets_confirmed": sum("initial" in e for e in episodes),
                "environment_steps": sum(m["environment_steps"] for m in metrics), "prefix_environment_steps": prefixes,
                "student_calls": len(attempted), "returned_student_calls": len(returned),
                "unknown_call_usage": len(attempted) - len(returned),
                "unknown_costs_possible": malformed or not (attempt / "output/summary.json").exists(),
                **{key + "_unknown_calls": len(attempted) - sum(c.get(key) is not None for c in returned)
                   for key in ("prompt_tokens", "completion_tokens")},
                **{key: sum(c.get(key) or 0 for c in returned) for key in ("prompt_tokens", "completion_tokens")}})
    report.update(source="ipr_exploration_and_sft_mc", attempts=attempts,
        accounting="All attempts retained. Environment steps INCLUDE reset-prefix replay; prefix steps also reported as a subset. "
                   "Unknowns never become failures or training labels. MC and retries add no trusted seeds or new tasks.")
    for key in ("environment_episodes", "environment_resets_confirmed", "environment_steps", "prefix_environment_steps",
                "student_calls", "returned_student_calls", "unknown_call_usage", "prompt_tokens", "completion_tokens",
                "prompt_tokens_unknown_calls", "completion_tokens_unknown_calls"):
        report["totals"][key] = sum(a[key] for a in attempts)
    report["totals"].pop("replay_environment_episodes", None)
    report["totals"].pop("replay_environment_steps", None)
    write_json(output / "disclosure.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--init-checkpoint", help="Final same-cell Seed-SFT LoRA adapter")
    for name in ("snapshot", "verified", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=["prepare", "train", "evaluate"], default="train")
    parser.add_argument("--through", choices=["all", "sft", "expert-mc"] + [f"{s}-{i}" for i in range(1, 7)
        for s in ("explore", "agent-mc", "pairs", "dpo")], default="all")
    parser.add_argument("--gpus", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=900)
    for name in ("env", "train", "vllm"):
        parser.add_argument(f"--{name}-python", default=os.environ.get(name.upper() + "_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("IPR requires positive rollout workers")
    batch_invariant()
    for name in ("snapshot", "verified", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    config = load_config(args.config)
    args.model_name = config["models"]["student"]["requested_repo"]
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"):
        if args.mode == "evaluate":
            validate_evaluation_init(args, config)
            os.environ["DECODING"] = "greedy"
            evaluate_pipeline(args)
            return
        recipe = deepcopy(RECIPE)
        recipe["runtime_dependencies"] = versions()
        identity = prepare(args, config, method="ipr", recipe=recipe)
        try:
            if args.mode == "train":
                train_pipeline(args, identity, config)
        finally:
            update_disclosure(args.output, identity)


if __name__ == "__main__":
    main()
