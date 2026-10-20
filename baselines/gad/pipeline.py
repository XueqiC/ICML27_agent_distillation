"""GAD: select exactly one explore-r (GPU), teacher-r (CPU), or train-r (GPU)."""
import argparse
import os
from pathlib import Path
from teachers.stage import ledger_counts
import signal
import sys

from baselines.common.evaluation import evaluate_models
from baselines.common.method_data import read
from baselines.common.pipeline import (batch_invariant, prepare, stage_run, run_command,
    served, update_disclosure as shared_disclosure)
from baselines.gad import RECIPE
from baselines.gad.data import collect, require_cpu_teacher
from baselines.gad.training import train_round
from sed.config import load_config
from sed.provenance import versions
from sed.storage import digest, file_hash, locked, write_json

STAGES = tuple(f"{stage}-{r}" for r in range(3) for stage in ("explore", "teacher", "train"))


def recipe():
    sources = read(Path(__file__).with_name("sources.json"))
    for path, expected in sources.items():
        if file_hash(path) != expected:
            raise ValueError("Pinned GAD upstream implementation/specification changed")
    return {"algorithm": RECIPE, "sources": sources, "runtime_dependencies": versions()}


def selected_stage(root, name, key, action, selected):
    if name != selected and not (Path(root) / "stages" / name / "complete.json").exists():
        raise ValueError("Missing prerequisite " + name + "; run it in its own resource stage job")
    return stage_run(root, name, key, action)


def train_pipeline(args, identity, config):
    if args.through not in STAGES:
        raise ValueError("GAD_THROUGH must select one explore/teacher/train round 0..2")
    if args.through.startswith("teacher-"):
        require_cpu_teacher()
    output, snapshot, previous = Path(args.output), Path(args.snapshot), None
    for r in range(3):
        key = digest([identity, r, str(snapshot), previous])
        def explore(attempt):
            with served(args, snapshot, attempt) as endpoint:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""
                run_command([args.env_python, "-m", "baselines.common.explore", "--config", args.config,
                    "--tasks", str(output / "tasks.json"), "--endpoint", endpoint, "--model-name", args.model_name,
                    "--decoding", "greedy", "--workers", str(args.workers),
                    "--server-provenance", str(attempt / "server/provenance.json"),
                    "--output", str(attempt / "output")], attempt / "explore.log", env=env)
        explored = selected_stage(output, f"explore-{r}", key, explore, args.through)
        if args.through == f"explore-{r}":
            return
        labelled = selected_stage(output, f"teacher-{r}", digest([key, str(explored)]),
            lambda a: collect(explored, args.verified, a / "output", config, cell=output,
                identity=identity, round_index=r), args.through)
        if args.through == f"teacher-{r}":
            return
        trained = selected_stage(output, f"train-{r}", digest([key, str(labelled)]),
            lambda a: train_round(args, identity, config, data=labelled / "data.json", snapshot=snapshot,
                previous=previous, round_index=r, output=a / "output"), args.through)
        previous = read(trained / "result.json")
        snapshot = Path(previous["checkpoint"])
        write_json(output / f"iteration-{r}.json", previous)
        if r == 2:
            write_json(output / "result.json", {"status": "complete", "checkpoint": str(snapshot),
                "models": read(output / "models.json"), "selection": None})
        if args.through == f"train-{r}":
            return


def ledger_rows(path):
    import json
    rows, partial = [], False
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                partial = True
    return rows, partial


def update_disclosure(output, identity):
    output = Path(output)
    shared_disclosure(output, identity)
    report = read(output / "disclosure.json")
    usage, partial = ledger_rows(output / "teacher/usage.jsonl")
    counter = read(output / "teacher/counter.json") if (output / "teacher/counter.json").exists() else {}
    billed = [r for r in usage if r.get("status") in {"success", "rejected", "incomplete"}]
    training = []
    for attempt in sorted((output / "updates").glob("round-*/stages/*/attempt-*")):
        if attempt.parent.name.startswith("sample-"):
            continue
        updates, broken = ledger_rows(attempt / "output/updates.jsonl")
        training.append({"path": str(attempt.relative_to(output)),
            "role": attempt.parent.name.split("-")[0], "updates": updates,
            "unknown_exposure_possible": broken or not (attempt / "output/result.json").exists()})
    sampling = []
    for path in sorted((output / "updates").glob("round-*/stages/sample-*/attempt-*/output/calls.jsonl")):
        calls, broken = ledger_rows(path)
        returned = [r["response"] for r in calls if r["status"] == "returned"]
        sampling.append({"path": str(path.relative_to(output)), "requests": sum(r["status"] == "started" for r in calls),
            "responses": sum(len(r.get("choices", [])) for r in returned),
            "usage": [r.get("usage") for r in returned],
            "unknown_usage_possible": broken or sum(r["status"] == "started" for r in calls) != len(returned)
                or any(r.get("usage") is None for r in returned)})
    report.update(method="gad", teacher_cap=3 * 40 * identity["unique_instances"], teacher_counter=counter,
        teacher_usage_ledger=str(output / "teacher/usage.jsonl"), training_attempts=training,
        sampling_attempts=sampling, teacher_verification="unknown", selection=None,
        unknown_teacher_usage_possible=partial or counter.get("pending", False) or counter.get("halted", False),
        token_accounting="All group IDs/text and usage, teacher raw replies/IDs/timestamps, real rollout/replay "
            "transitions and failed attempts are retained. Completed-update exposure excludes unknown partial work.")
    report["totals"].update(teacher_calls=counter.get("attempts", 0),
        teacher_unknown_usage_calls=counter.get("attempts", 0) - sum(r.get("usage") is not None for r in billed),
        group_responses=sum(s["responses"] for s in sampling),
        **{"group_" + k: sum((u or {}).get(k, 0) or 0 for s in sampling for u in s["usage"])
           for k in ("prompt_tokens", "completion_tokens")},
        **{"teacher_" + k: sum((r.get("usage") or {}).get(k, 0) or 0 for r in billed)
           for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens")},
        **{role + "_" + k: sum(u.get(k, 0) for t in training if t["role"] == role for u in t["updates"])
           for role in ("actor", "critic") for k in ("examples_seen", "supervised_tokens_seen",
                                                      "logical_forward_tokens", "physical_forward_tokens")})
    report["teacher_outcomes"] = ledger_counts(output / "teacher/usage.jsonl")
    write_json(output / "disclosure.json", report)


def evaluate(args, config):
    identity = read(Path(args.output) / "identity.json")
    if (identity["method"] != "gad" or identity["config_sha256"] != digest(config)
            or identity["snapshot"] != args.snapshot or identity["verified"] != args.verified):
        raise ValueError("GAD evaluation cell changed")
    for model in read(Path(args.output) / "models.json"):
        if file_hash(Path(model["checkpoint"]) / "training_state.json") != model["checkpoint_state_sha256"]:
            raise ValueError("Published GAD checkpoint changed")
    # The common evaluator runs HF/vLLM equivalence before each dev140 run.
    evaluate_models(args, decodings=("greedy",))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    for key in ("snapshot", "verified", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate"), default="train")
    parser.add_argument("--through", choices=STAGES, default="explore-0")
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=900)
    for key in ("env", "train", "vllm"):
        parser.add_argument("--" + key + "-python", default=os.environ.get(key.upper() + "_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("Positive workers required")
    for key in ("snapshot", "verified", "output"):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    if Path(args.output).is_relative_to(Path(args.verified)):
        parser.error("GAD output must be outside VERIFIED support storage")
    config = load_config(args.config)
    if (config["training"]["thinking_mode"] != "off" or config["training"]["student_step_token_cap"] != 1024
            or config["environment"]["max_episode_steps"] != 40
            or config["models"]["student"]["requested_repo"] != "google/gemma-4-12B-it"
            or not config["models"]["student"]["revision"].startswith("707f0a3b")
            or config["models"]["teacher"]["provider_model_id"] != "gpt-6-luna"):
        raise ValueError("GAD requires the configured paper-1 models and H0")
    args.model_name = config["models"]["student"]["requested_repo"]
    batch_invariant()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"):
        if args.mode == "evaluate":
            return evaluate(args, config)
        identity = prepare(args, config, method="gad", recipe=recipe())
        try:
            if args.mode == "train":
                train_pipeline(args, identity, config)
        finally:
            update_disclosure(args.output, identity)


if __name__ == "__main__":
    main()
