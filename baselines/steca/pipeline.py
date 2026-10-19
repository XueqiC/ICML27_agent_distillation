"""STeCa warm-up, fixed-model IPR MC, capped reflection and weighted full FT."""
import os
import json
from pathlib import Path

from baselines.common.method_cli import main as cli, require_completed
from baselines.common.method_data import read
from baselines.ipr.pipeline import rollout_stage, update_disclosure as ipr_disclosure
from baselines.eto.pipeline import stage_run, run_command, publish_models
from baselines.steca.data import plan, reflect
from sed.storage import digest, write_json

RECIPE = {"training_seed": 0, "adaptation": "full", "mc_samples": 5, "delta": 0, "eta": .01,
    "sft": {"learning_rate": 3e-6, "epochs": 3, "original_batch": 16, "scheduler": "cosine", "warmup_ratio": .03},
    "reinforced": {"learning_rate": 3e-6, "epochs": 1, "original_batch": 15,
                   "scheduler": "constant_with_warmup", "warmup_ratio": .1},
    "packing": "turn", "max_length": 32768, "teacher": "gpt-6-luna", "calls_per_deviated_trajectory": 1,
    "mc": {"temperature": 1, "top_p": 1, "top_k": -1, "max_tokens": 512},
    "exploration": {"temperature": 0, "max_tokens": 1024}, "distance_encoder": "google-bert/bert-base-uncased"}
STAGES = ("sft", "expert-mc", "explore", "agent-mc", "detect", "reflect", "datasets", "reinforced")


def training_command(args, output, *, snapshot=None, data=None):
    recipe = RECIPE["reinforced" if data else "sft"]
    command = [args.train_python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.gpus),
        "-m", "baselines.common.fsdp_train", "--config", args.config, "--objective", "steca" if data else "sft",
        "--snapshot", str(snapshot or args.snapshot), "--base-snapshot", args.snapshot, "--verified", args.verified,
        "--output", str(output), "--seed", "0", "--packing", "turn", "--max-length", "32768"]
    for key, value in recipe.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    if data:
        command += ["--method-data", str(data)]
    return command


def train_pipeline(args, identity, config):
    output = Path(args.output)
    if args.through == "reinforced":
        require_completed(output, ["reflect", "datasets"])
    if args.through in {"detect", "reflect", "datasets"}:
        require_completed(output, ["train-sft", "expert-mc", "explore-1", "agent-mc-1"])
    def stage(name, action, *sources):
        return stage_run(output, name, digest([identity, name, *map(str, sources)]), action)
    sft = stage("train-sft", lambda a: run_command(training_command(args, a / "output"), a / "train.log")) / "checkpoint"
    publish_models(output, [{"name": "sft", "checkpoint": str(sft), "full_equivalence_required": True}])
    if args.through == "sft":
        return
    expert_mc = stage("expert-mc", lambda a: rollout_stage(args, sft, a, kind="expert-mc", iteration=0), sft)
    if args.through == "expert-mc":
        return
    exploration = stage("explore-1", lambda a: rollout_stage(args, sft, a, kind="explore", iteration=1), sft)
    if args.through == "explore":
        return
    agent_mc = stage("agent-mc-1", lambda a: rollout_stage(args, sft, a, kind="agent-mc", iteration=1,
        exploration=exploration), sft, exploration)
    if args.through == "agent-mc":
        return
    def detect(attempt):
        planned = plan(args.verified, exploration, expert_mc, agent_mc, config)
        (attempt / "output").mkdir()
        write_json(attempt / "output/plan.json", planned)
    detected = stage("detect", detect, exploration, expert_mc, agent_mc)
    if args.through == "detect":
        return
    planned = read(detected / "plan.json")
    reflected = stage("reflect", lambda a: reflect(planned, config, args.verified, a / "output",
        output / "teacher"), detected)
    if args.through == "reflect":
        return
    def construct(attempt):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        run_command([args.train_python, "-m", "baselines.common.method_worker", "steca", "--config", args.config,
            "--verified", args.verified, "--embedding-snapshot", args.embedding_snapshot,
            "--data", str(detected / "plan.json"), "--reflections", str(reflected / "reflections.json"),
            "--output", str(attempt / "output"), "--sources", *[str(root / file)
                for root in (exploration, expert_mc, agent_mc) for file in ("provenance.json", "receipts.json", "plan.json")]],
            attempt / "datasets.log", env=env)
    dataset = stage("datasets", construct, detected, reflected)
    if args.through == "datasets":
        return
    if read(dataset / "data.json"):
        final = stage("train-reinforced", lambda a: run_command(training_command(args, a / "output", snapshot=sft,
            data=dataset / "data.json"), a / "train.log"), sft, dataset) / "checkpoint"
    else:
        def carry(attempt):
            (attempt / "output").mkdir()
            write_json(attempt / "output/result.json", {"status": "carried_forward", "checkpoint": str(sft),
                "examples_seen": 0, "supervised_tokens_seen": 0, "logical_forward_tokens": 0})
        stage("train-reinforced", carry, sft, dataset)
        final = sft
    publish_models(output, [{"name": "reinforced", "checkpoint": str(final), "carried_forward": final == sft, "full_equivalence_required": True}])
    write_json(output / "result.json", {"status": "complete", "checkpoint": str(final),
        "counts": read(dataset / "provenance.json")["details"]["counts"]})


def update_disclosure(output, identity):
    ipr_disclosure(output, identity)
    output = Path(output)
    report = read(output / "disclosure.json")
    counter = read(output / "teacher/counter.json") if (output / "teacher/counter.json").exists() else {}
    report["totals"]["teacher_calls"] = counter.get("attempts", 0)
    ledger_path = output / "teacher/usage.jsonl"
    usage, unknown = [], False
    for line in ledger_path.read_text().splitlines() if ledger_path.exists() else []:
        try:
            row = json.loads(line)
            if row.get("status") in {"success", "rejected"}:
                usage.append(row.get("usage"))
        except json.JSONDecodeError:
            unknown = True
    report["totals"].update({"teacher_" + key: sum((u or {}).get(key, 0) or 0 for u in usage)
        for key in ("input_tokens", "output_tokens", "reasoning_tokens")})
    report["teacher_unknown_usage_calls"] = counter.get("attempts", 0) - sum(u is not None for u in usage)
    report["teacher_partial_ledger"] = unknown
    report.update(method="steca", teacher_counter=counter, teacher_usage_ledger=str(output / "teacher/usage.jsonl"),
        calibration_verification="Reflection text is teacher-generated; action/observations remain verified expert evidence.")
    write_json(output / "disclosure.json", report)


def main(argv=None):
    cli("steca", RECIPE, STAGES, train_pipeline, update_disclosure, argv)


if __name__ == "__main__":
    main()
