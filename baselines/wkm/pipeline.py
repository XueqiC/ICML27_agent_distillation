"""Experienced LoRA -> rejected rollouts -> self-knowledge -> two independent LoRAs."""
import os
import json
from pathlib import Path

from baselines.common.method_cli import main as cli, require_completed
from baselines.common.pipeline import update_disclosure as common_disclosure
from baselines.common.method_data import read
from baselines.eto.pipeline import stage_run, served, run_command, publish_models
from baselines.wkm.data import merge
from sed.storage import digest, write_json

RECIPE = {"training_seed": 0, "epochs": 3, "rank": 8, "alpha": 16, "dropout": 0,
    "targets": ["q_proj", "v_proj"], "learning_rate": 1e-4, "scheduler": "cosine", "warmup_ratio": 0,
    "original_batch": 32, "packing": "turn", "max_length": 32768, "gamma": .4, "top_n": 3000,
    "teacher_calls": 0, "exploration": "one_greedy_per_support_task", "agent_retraining_init": "base",
    "synthesis": {"temperature": .5, "top_p": 1, "top_k": -1, "seed": 0, "task_tokens": 512, "state_tokens": 128}}
STAGES = ("experienced", "explore", "synthesize", "merge", "knowledge-base", "world", "agent")


def training_command(args, output, data=None):
    command = [args.train_python, "-m", "baselines.common.fsdp_train", "--config", args.config,
        "--objective", "sft", "--adaptation", "wkm_lora", "--snapshot", args.snapshot,
        "--base-snapshot", args.snapshot, "--verified", args.verified, "--output", str(output),
        "--epochs", "3", "--seed", "0", "--max-length", "32768", "--learning-rate", "1e-4",
        "--scheduler", "cosine", "--warmup-ratio", "0", "--original-batch", "32", "--packing", "turn"]
    if data:
        command += ["--method-data", str(data)]
    return command


def student_stage(args, checkpoint, attempt, *, synthesis=False, exploration=None):
    with served(args, checkpoint, attempt) as endpoint:
        command = [args.env_python, "-m", "baselines.wkm.data" if synthesis else "baselines.common.explore",
            "--config", args.config, "--output", str(attempt / "output"), "--endpoint", endpoint,
            "--model-name", "dose-policy", "--server-provenance", str(attempt / "server/provenance.json")]
        if synthesis:
            command += ["--verified", args.verified, "--exploration", str(exploration)]
        else:
            command += ["--tasks", str(Path(args.output) / "tasks.json"), "--decoding", "greedy",
                        "--workers", str(args.workers)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        run_command(command, attempt / "student.log", env=env)


def train_pipeline(args, identity, config):
    output = Path(args.output)
    if args.through in {"merge", "knowledge-base"}:
        require_completed(output, ["train-experienced", "explore-1", "synthesize"])
    def stage(name, action, *sources):
        return stage_run(output, name, digest([identity, name, *map(str, sources)]), action)
    experienced = stage("train-experienced", lambda a: run_command(
        training_command(args, a / "output"), a / "train.log")) / "checkpoint"
    publish_models(output, [{"name": "experienced", "checkpoint": str(experienced), "adapter": str(experienced)}])
    if args.through == "experienced":
        return
    exploration = stage("explore-1", lambda a: student_stage(args, experienced, a), experienced)
    if args.through == "explore":
        return
    synthesis = stage("synthesize", lambda a: student_stage(args, experienced, a, synthesis=True,
        exploration=exploration), experienced, exploration)
    if args.through == "synthesize":
        return
    data = stage("merge", lambda a: merge(args.verified, exploration, synthesis, a / "output", config),
                 exploration, synthesis)
    if args.through == "merge":
        return
    def embed(attempt):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        run_command([args.train_python, "-m", "baselines.common.method_worker", "wkm", "--config", args.config,
            "--verified", args.verified, "--embedding-snapshot", args.embedding_snapshot,
            "--data", str(data / "data.json"), "--output", str(attempt / "output")], attempt / "embed.log", env=env)
    knowledge = stage("knowledge-base", embed, data)
    if args.through == "knowledge-base":
        return
    world = stage("train-world", lambda a: run_command(training_command(args, a / "output", data / "world.json"),
        a / "train.log"), data) / "checkpoint"
    def model(name, agent):
        return {"name": name, "checkpoint": str(agent), "adapter": str(agent),
            "knowledge_adapter": str(world), "knowledge_base": str(knowledge / "knowledge_base.json"),
            "decoding": "wkm", "gamma": .4}
    publish_models(output, [model("world", experienced)])
    if args.through == "world":
        return
    agent = stage("train-agent", lambda a: run_command(training_command(args, a / "output", data / "agent.json"),
        a / "train.log"), data) / "checkpoint"
    publish_models(output, [model("wkm", agent)])
    write_json(output / "result.json", {"status": "complete", "agent": str(agent), "world": str(world),
        "knowledge_base": str(knowledge / "knowledge_base.json"), "teacher_calls": 0})


def update_disclosure(output, identity):
    common_disclosure(output, identity)
    path = Path(output) / "disclosure.json"
    report = read(path)
    synthesis = []
    for attempt in sorted((Path(output) / "stages/synthesize").glob("attempt-*")):
        calls_path = attempt / "synthesis_calls/calls.jsonl"
        calls, partial = [], False
        for line in calls_path.read_text().splitlines() if calls_path.exists() else []:
            try:
                calls.append(json.loads(line))
            except json.JSONDecodeError:
                partial = True
        starts = [r for r in calls if r["status"] == "started"]
        ends = [r for r in calls if r["status"] == "completed"]
        synthesis.append({"path": str(attempt), "student_calls": len(starts), "returned": len(ends),
            "unknown_usage": len(starts)-len(ends), "partial": partial,
            **{k: sum(r.get(k) or 0 for r in ends) for k in ("prompt_tokens", "completion_tokens")}})
    exploration_calls = []
    for path_ in sorted((Path(output) / "stages").glob("explore-*/attempt-*/output/episodes/*/student_calls/calls.jsonl")):
        for line in path_.read_text().splitlines():
            try:
                exploration_calls.append(json.loads(line))
            except json.JSONDecodeError:
                report["unknown_costs_possible"] = True
    returned = [r for r in exploration_calls if r["status"] == "completed"]
    report["totals"].update(student_calls=sum(r["status"] == "started" for r in exploration_calls)
        + sum(r["student_calls"] for r in synthesis),
        **{key: sum(r.get(key) or 0 for r in returned) + sum(r[key] for r in synthesis)
           for key in ("prompt_tokens", "completion_tokens")})
    report.update(method="wkm", source="wkm_self_synthesis", synthesis_attempts=synthesis,
        knowledge_verification="Self-generated text is not environment ground truth; only chosen actions were verified.")
    report["totals"]["teacher_calls"] = 0
    write_json(path, report)


def main(argv=None):
    cli("wkm", RECIPE, STAGES, train_pipeline, update_disclosure, argv)


if __name__ == "__main__":
    main()
