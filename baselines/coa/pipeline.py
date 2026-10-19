"""Hash-bound, one-stage-per-job CoA collection, SFT, query filter and DAPO."""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from baselines.coa import PIN, RECIPE
from baselines.coa.data import collect, datasets, read, support
from baselines.coa.rl import mixed, query_kept
from baselines.coa.rollout import checked_groups
from baselines.coa.teacher import require_cpu
from baselines.common.pipeline import batch_invariant, prepare, stage_run, run_command, served, publish_models
from baselines.common.schedule import dose
from eval.run_suite import frozen_tasks
from sed.config import load_config
from sed.manifest import frozen_write, load_frozen
from sed.provenance import code_hashes, write_provenance
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, locked, write_json

OFFICIAL_FILES = (
    "AFM/train/web_agent/sft/sft_qwen2.5_7b.sh", "AFM/train/code_agent/sft/sft_qwen2.5_7b.sh",
    "AFM/train/web_agent/rl/train_dapo_web_agent.sh", "AFM/train/code_agent/rl/train_dapo_code_agent.sh",
    "AFM/data/sys_prompts.py", "LLaMA-Factory/data/dataset_info.json",
    "LLaMA-Factory/src/llamafactory/data/processor/supervised.py",
    "verl/verl/trainer/ppo/core_algos.py", "verl/verl/trainer/ppo/ray_trainer.py",
    "verl/verl/trainer/config/ppo_trainer.yaml",
    "verl/verl/trainer/main_ppo.py", "verl/verl/workers/fsdp_workers.py",
    "verl/verl/utils/torch_functional.py",
    "LLaMA-Factory/src/llamafactory/hparams/training_args.py",
)


def official_sources():
    root = Path("third_party/Agent_Foundation_Models")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True).stdout.decode().strip()
    if head != PIN:
        raise ValueError("Official AFM revision changed")
    result = {}
    for name in OFFICIAL_FILES:
        original = subprocess.run(["git", "-C", str(root), "show", PIN + ":" + name], check=True, capture_output=True).stdout
        if hashlib.sha256(original).hexdigest() != file_hash(root / name):
            raise ValueError("Pinned official source changed")
        result[name] = file_hash(root / name)
    return result


def stage(root, name, identity, action, *, selected):
    if name != selected and not (Path(root) / "stages" / name / "complete.json").exists():
        raise ValueError(f"Missing prerequisite {name}; run it as a separate job")
    return stage_run(root, name, identity, action)


def training_command(args, kind, snapshot, data, output, update=0):
    return [args.train_python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.gpus),
        "-m", "baselines.coa.training", "--config", args.config, "--snapshot", str(snapshot),
        "--base-snapshot", args.snapshot, "--verified", args.verified, "--identity", str(Path(args.output) / "identity.json"),
        "--kind", kind, "--update", str(update), "--data", str(data), "--output", str(output)]


def publish(output, name, checkpoint):
    publish_models(output, [{"name": name, "checkpoint": str(checkpoint), "method": "coa", "harness": "coa",
        "full_equivalence_required": True, "checkpoint_state_sha256": file_hash(checkpoint / "training_state.json")}])


def rollout_command(args, listing, endpoint, attempt, output, round_index):
    task_path = Path(output).parent / (Path(output).name + "-tasks.json")
    frozen_write(task_path, listing)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    run_command([args.env_python, "-m", "baselines.coa.rollout", "--config", args.config,
        "--tasks", str(task_path), "--endpoint", endpoint, "--model-name", args.model_name,
        "--server-provenance", str(attempt / "server/provenance.json"), "--round", str(round_index),
        "--output", str(output)], Path(output).parent / (Path(output).name + ".log"), env=env)


def subset(listing, ids):
    return {**listing, "task_ids": ids, "list_sha256": digest(ids),
            "parent_seed_ids": {task_id: listing["parent_seed_ids"][task_id] for task_id in ids}}


def filter_queries(args, config, listing, snapshot, attempt):
    output = attempt / "output"
    output.mkdir()
    with served(args, snapshot, attempt) as endpoint:
        rollout_command(args, listing, endpoint, attempt, output / "rollouts", -1)
    groups = checked_groups(output / "rollouts", listing)
    if [task_id for task_id, _ in groups] != listing["task_ids"]:
        raise ValueError("Query filter must cover every support TRAIN task exactly once")
    decisions = [{"task_id": tid, "successes": sum(e["success"] for e in eps), "keep": query_kept(eps)} for tid, eps in groups]
    write_json(output / "filter.json", decisions)
    write_json(output / "tasks.json", subset(listing, [d["task_id"] for d in decisions if d["keep"]]))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    run_command([args.train_python, "-m", "baselines.coa.rl", "--tasks", str(output / "tasks.json"),
        "--output", str(output / "schedule.json")], attempt / "schedule.log", env=env)
    write_provenance(output, command="coa_sft_query_filter", config=config,
        inputs=[Path(snapshot) / "training_state.json", output / "rollouts/groups.json"],
        details={"samples": 8, "discard_rule": "8/8 environment successes", "training_exposure": 0})


def explore_update(args, config, listing, eligible, snapshot, attempt, update, cursor):
    output = attempt / "output"
    output.mkdir()
    schedule = eligible["query_schedule"]["task_ids"]
    n = len(eligible["task_ids"])
    batch = dose(n, RECIPE["rl"]["original_batch"], epochs=1)["b_prime"]
    gen_batch = dose(n, RECIPE["rl"]["generation_batch"], epochs=1)["b_prime"]
    selected, artifacts, discarded, starts = [], {}, [], cursor
    with served(args, snapshot, attempt) as endpoint:
        batch_index = 0
        while len(selected) < batch and cursor < len(schedule):
            # Upstream drop_last makes both epoch lengths divisible by gen_batch.
            ids = schedule[cursor:cursor + gen_batch]
            root = output / f"batch-{batch_index:04d}"
            rollout_command(args, subset(listing, ids), endpoint, attempt, root, update * 100000 + cursor)
            groups = checked_groups(root, listing)
            if [tid for tid, _ in groups] != ids:
                raise ValueError("Dynamic rollout query schedule changed")
            artifacts[str(root)] = artifact_hashes(root)
            for i, (tid, episodes) in enumerate(groups):
                row = {"root": str(root), "group": i, "task_id": tid}
                if mixed(episodes):
                    selected.append(row)
                else:
                    discarded.append({**row, "reason": "constant_reward", "successes": sum(e["success"] for e in episodes)})
            cursor += len(ids)
            batch_index += 1
    enough = len(selected) >= batch
    plan = {"support_sha256": digest(listing), "snapshot": str(snapshot.resolve()),
        "checkpoint_state_sha256": file_hash(snapshot / "training_state.json"),
        "start_cursor": starts, "next_cursor": cursor, "schedule_size": len(schedule),
        "schedule_sha256": digest(eligible["query_schedule"]),
        "update": update, "batch_size": batch, "generation_batch": gen_batch,
        "selected": selected[:batch] if enough else [],
        "unused_mixed_groups": selected[batch:] if enough else selected,
        "discarded": discarded, "artifacts": artifacts,
        "status": "ready" if enough else "epoch_budget_exhausted"}
    write_json(output / "plan.json", plan)
    write_provenance(output, command="coa_dynamic_group_filter", config=config,
        inputs=[snapshot / "training_state.json"], details=plan)


def train_pipeline(args, identity, config):
    selected, output = args.through, Path(args.output)
    if selected == "teacher":
        require_cpu()
    if selected not in {"teacher", "data", "train-sft", "query-filter"} and not re.fullmatch(r"(explore|train-rl)-([0-9]|[1-5][0-9])", selected):
        raise ValueError("Select teacher, data, train-sft, query-filter, explore-r or train-rl-r (r < 60)")
    listing, tasks = support(config, args.verified)
    def run(name, action, *sources):
        return stage(output, name, digest([identity, name, *map(str, sources)]), action, selected=selected)
    collected = run("teacher", lambda a: collect(config, listing, tasks, a / "output", output, identity))
    if selected == "teacher":
        return
    data = run("data", lambda a: datasets(collected, a / "output", config, args.verified), collected)
    if not read(data / "data.json"):
        write_json(output / "result.json", {"status": "stopped_empty_sft", "training_exposure": 0,
            "reason": "No trajectories passed registered CoA filters; no H0 seed fallback"})
        return
    if selected == "data":
        return
    trained = run("train-sft", lambda a: run_command(training_command(args, "sft", Path(args.snapshot),
        data / "data.json", a / "output"), a / "train.log"), data)
    snapshot = trained / "checkpoint"
    publish(output, "sft", snapshot)
    if selected == "train-sft":
        return
    filtered = run("query-filter", lambda a: filter_queries(args, config, listing, snapshot, a), snapshot)
    eligible = read(filtered / "tasks.json")
    eligible["query_schedule"] = read(filtered / "schedule.json")
    if not eligible["task_ids"]:
        write_json(output / "result.json", {"status": "complete_all_queries_solved_8_of_8", "checkpoint": str(snapshot)})
        return
    if selected == "query-filter":
        return
    cursor = 0
    for update in range(RECIPE["rl"]["max_updates"]):
        explored = run(f"explore-{update}", lambda a: explore_update(args, config, listing, eligible, snapshot,
            a, update, cursor), filtered, snapshot, cursor)
        plan = read(explored / "plan.json")
        if selected == f"explore-{update}":
            return
        if not plan["selected"]:
            write_json(output / "result.json", {"status": "complete_epoch_budget_exhausted",
                "checkpoint": str(snapshot), "updates": update, "next_cursor": plan["next_cursor"]})
            return
        trained = run(f"train-rl-{update}", lambda a: run_command(training_command(args, "rl", snapshot,
            explored / "plan.json", a / "output", update), a / "train.log"), snapshot, explored)
        snapshot, cursor = trained / "checkpoint", plan["next_cursor"]
        publish(output, f"rl-{update + 1}", snapshot)
        if cursor == plan["schedule_size"] or update + 1 == RECIPE["rl"]["max_updates"]:
            write_json(output / "result.json", {"status": "complete", "checkpoint": str(snapshot), "updates": update + 1})
            return
        if selected == f"train-rl-{update}":
            return


def evaluate_pipeline(args, identity):
    for model in read(Path(args.output) / "models.json"):
        checkpoint = Path(model["checkpoint"])
        if model["checkpoint_state_sha256"] != file_hash(checkpoint / "training_state.json"):
            raise ValueError("Published checkpoint changed")
        def evaluate(attempt):
            check = [args.train_python, "-m", "baselines.coa.gate", "--config", args.config,
                "--snapshot", args.snapshot, "--checkpoint", str(checkpoint), "--verified", args.verified]
            env = os.environ.copy()
            if env.get("CUDA_VISIBLE_DEVICES"):
                env["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"].split(",")[0]
            run_command(check + ["--hf-only", "--output", str(attempt / "hf")], attempt / "hf.log", env=env)
            with served(args, checkpoint, attempt) as endpoint:
                server = attempt / "server/provenance.json"
                env["CUDA_VISIBLE_DEVICES"] = ""
                run_command([args.env_python, *check[1:], "--hf-reference", str(attempt / "hf/reference.json"),
                    "--endpoint", endpoint, "--server-provenance", str(server), "--output", str(attempt / "check")],
                    attempt / "check.log", env=env)
                run_command([args.env_python, "eval/run_suite.py", "--config", args.config, "--policy", "student",
                    "--endpoint", endpoint, "--model-name", args.model_name, "--tasks", "data/manifest/dev140.json",
                    "--workers", str(args.workers), "--server-provenance", str(server),
                    "--coa-check", str(attempt / "check/report.json"), "--output", str(attempt / "output")],
                    attempt / "eval.log", env=env)
        stage_run(Path(args.output) / "evaluation", model["name"], digest([identity, model,
            file_hash("data/manifest/dev140.json")]), evaluate)


def update_disclosure(output, identity, *, evaluation=False):
    output = Path(output)
    namespace = "teacher_eval" if evaluation else "teacher"
    usage_path, counter_path = output / namespace / "usage.jsonl", output / namespace / "counter.json"
    rows, partial = [], False
    if usage_path.exists():
        for line in usage_path.read_text().splitlines():
            try:
                import json
                rows.append(json.loads(line))
            except ValueError:
                partial = True
    counter = read(counter_path) if counter_path.exists() else {}
    usage = [r.get("usage") for r in rows if r.get("status") in {"success", "rejected"}]
    episodes = []
    roots = [output / namespace] if evaluation else [output / namespace, output / "stages"]
    for root in roots:
        for path in root.rglob("episode.json"):
            episode = read(path)
            episodes.append({"path": str(path), "task_id": episode["task_id"], "success": episode.get("success", "unknown"),
                             "metrics": episode.get("metrics", {}), "status": episode["status"]})
    exposure, tokens, unknown, training = 0, 0, False, []
    if not evaluation:
        for attempt in (output / "stages").glob("train-*/attempt-*"):
            result_path, ledger = attempt / "output/result.json", attempt / "output/updates.jsonl"
            last = {}
            if ledger.exists():
                for line in ledger.read_text().splitlines():
                    try:
                        import json
                        last = json.loads(line)
                    except ValueError:
                        unknown = True
            exposure += last.get("examples_seen", 0)
            tokens += last.get("supervised_tokens_seen", 0)
            unknown |= not result_path.exists()
            training.append({"attempt": str(attempt), "last_completed_update": last, "complete": result_path.exists()})
    replay_steps = replay_resets = 0
    unknown_replay_costs = False
    for root in roots:
        for path in root.rglob("replay.json"):
            row = read(path)
            replay_steps += row.get("environment_steps", 0)
            replay_resets += row.get("environment_episodes", 0)
            unknown_replay_costs |= bool(row.get("environment_steps_unknown", 0))
    for path in (output / namespace / "executions").glob("*/receipt.json"):
        replay = read(path)["replay"]
        replay_steps += replay.get("environment_steps", 0)
        replay_resets += replay.get("environment_episodes", 0)
        unknown_replay_costs |= replay.get("unknown_costs_possible", False)
    report = {"method": "coa", "evaluation_only": evaluation, "K": identity["K"],
        "unique_instances": identity["unique_instances"], "task_groups": identity["task_groups"],
        "teacher_counter": counter, "teacher_usage_ledger": str(usage_path),
        "teacher_cap": counter.get("identity", {}).get("cap", (140 if evaluation else identity["unique_instances"])
            * RECIPE["teacher_eval_runs" if evaluation else "teacher_runs"] * 40 * RECIPE["max_calls_per_step"]),
        "unknown_teacher_usage_possible": partial or counter.get("halted", False) or counter.get("pending", False),
        "teacher_calls": counter.get("attempts", 0), "teacher_usage_unknown_calls": counter.get("attempts", 0) - sum(u is not None for u in usage),
        "teacher_tokens": {k: sum((u or {}).get(k, 0) or 0 for u in usage) for k in ("input_tokens", "output_tokens", "reasoning_tokens")},
        "episodes": episodes, "training_attempts": training, "examples_seen": exposure,
        "supervised_tokens_seen": tokens, "unknown_training_exposure_possible": unknown,
        "replay_environment_steps": replay_steps, "replay_environment_resets": replay_resets,
        "unknown_replay_costs_possible": unknown_replay_costs,
        "student_tokens": {key: {
            "known_total": sum(e["metrics"].get(key, {}).get("known_total", 0) for e in episodes
                               if "/teacher/" not in e["path"] and "/teacher_eval/" not in e["path"]),
            "unknown_calls": sum(e["metrics"].get(key, {}).get("unknown_calls", 0) for e in episodes
                                 if "/teacher/" not in e["path"] and "/teacher_eval/" not in e["path"])}
            for key in ("prompt_tokens", "completion_tokens", "thinking_tokens")},
        "environment_steps": sum(e["metrics"].get("environment_steps", 0) for e in episodes),
        "trusted_trajectories_added": 0, "new_unique_instances": 0,
        "accounting": "All failures, raw responses, per-call usage, rejection decisions and duplicate exposures retained. K counts original trusted seeds only."}
    write_json(output / (namespace + "/disclosure.json" if evaluation else "disclosure.json"), report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    for key in ("snapshot", "verified", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate", "teacher-evaluate"), default="train")
    parser.add_argument("--through", default="teacher")
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
    config = load_config(args.config)
    for root in ("data/verified", "reports/runs", config["paths"]["raw"], config["paths"]["verified"]):
        if Path(args.output).is_relative_to(Path(root).resolve()):
            raise ValueError("CoA output must be outside protected data/run storage")
    if (file_hash("docs/baselines/NEW_METHOD_SPECS_1019.md") != RECIPE["spec_sha256"]
            or config["models"]["student"]["requested_repo"] != "google/gemma-4-12B-it"
            or not config["models"]["student"]["revision"].startswith("707f0a3b")
            or config["models"]["teacher"]["provider_model_id"] != "gpt-6-luna"
            or config["environment"]["max_episode_steps"] != 40
            or config["training"]["thinking_mode"] != "off" or config["training"]["student_step_token_cap"] != 1024):
        raise ValueError("Pinned CoA spec/models/H0 configuration changed")
    args.model_name = config["models"]["student"]["requested_repo"]
    batch_invariant()
    recipe = {"parameters": deepcopy(RECIPE), "official_source_sha256": official_sources()}
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"):
        if args.mode in {"evaluate", "teacher-evaluate"}:
            identity = read(Path(args.output) / "identity.json")
            if (identity["method"] != "coa" or identity["config_sha256"] != digest(config) or identity["recipe"] != recipe
                    or identity["snapshot"] != args.snapshot or identity["verified"] != args.verified
                    or identity["code_sha256"] != code_hashes(Path.cwd())):
                raise ValueError("CoA evaluation cell identity changed")
            if args.mode == "evaluate":
                evaluate_pipeline(args, identity)
            else:
                require_cpu()
                manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
                listing, tasks = frozen_tasks("data/manifest/dev140.json", manifest, freeze, policy="teacher")
                if listing["name"] != "dev140" or any(t["split"] != "dev" or t.get("sealed") for t in tasks):
                    raise ValueError("Teacher evaluation requires frozen dev140")
                try:
                    stage_run(Path(args.output) / "evaluation", "teacher", digest([identity, listing]),
                        lambda a: collect(config, listing, tasks, a / "output", args.output, identity, evaluation=True))
                finally:
                    update_disclosure(args.output, identity, evaluation=True)
            return
        identity = prepare(args, config, method="coa", recipe=recipe)
        try:
            if args.mode == "train":
                train_pipeline(args, identity, config)
        finally:
            update_disclosure(args.output, identity)


if __name__ == "__main__":
    main()
