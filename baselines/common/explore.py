"""Frozen TRAIN exploration with the unchanged student/harness and replay checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import time

from eval.run_suite import task_result, summarize
from sed.config import load_config
from sed.manifest import authorize_task, load_frozen, resolve_task, verify_task_files
from sed.provenance import write_provenance
from sed.runner import _environment, run_episode, replay_episode
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, write_json
from students.vllm_student import Student, context_options


def training_tasks(listing, manifest, freeze):
    ids = listing["task_ids"]
    if (not ids or len(set(ids)) != len(ids) or listing["list_sha256"] != digest(ids)
            or listing["manifest_sha256"] != digest(manifest)
            or listing["split_freeze_sha256"] != digest(freeze)):
        raise ValueError("TRAIN task list differs from frozen manifest/splits")
    tasks = [resolve_task(manifest, freeze, task_id) for task_id in ids]
    for task in tasks:
        authorize_task(task, "generation")
        if task["source_split"] != "train":
            raise ValueError("Exploration requires original TRAIN tasks")
    return tasks


def explore_one(job, *, env_factory=_environment, policy_factory=Student):
    task, config, directory, spec, parents, inputs = job
    directory = Path(directory)
    replay = {"matched": False, "status": "not_started", "environment_episodes": 0,
              "environment_steps": 0, "environment_steps_unknown": 0}
    try:
        verify_task_files(config["environment"]["data_root"], task)
        policy = policy_factory(log_dir=directory / "student_calls", **spec)
        episode = run_episode(env_factory(config, task), policy, task, config, directory,
                        policy_condition="student_exploration", purpose="generation",
                        parent_seed_ids=parents, input_paths=inputs)
        # Count actual replay calls even if consistency fails partway through.
        class CountingEnvironment:
            def __init__(self, env):
                self.env = env
            def __getattr__(self, name):
                return getattr(self.env, name)
            def reset(self, task_id):
                replay["environment_episodes"] += 1
                return self.env.reset(task_id)
            def step(self, action):
                replay["environment_steps"] += 1
                try:
                    return self.env.step(action)
                except Exception:
                    replay["environment_steps_unknown"] += 1
                    raise
        replay.update(replay_episode(CountingEnvironment(env_factory(config, task)), episode))
        replay["status"] = "ok"
    except Exception as exc:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "failure.json", {"task_id": task["task_id"], "status": "error",
                   "error_type": type(exc).__name__})
        if replay["environment_episodes"]:
            replay["status"] = "error"
    write_json(directory / "replay.json", replay)
    row = task_result(task, directory)
    episode_path = directory / "episode.json"
    episode = json.loads(episode_path.read_text()) if episode_path.exists() else {}
    eligible = row["status"] == "ok" and type(row["success"]) is bool and replay["matched"]
    receipt = {**row, "won": row["success"], "parent_seed_ids": parents,
               "source": "student_exploration", "expansion": True, "trusted_seed": False,
               "trusted_trajectories_added": 0, "training_eligible": eligible,
               "environment_episodes": int(bool(episode)),
               "environment_resets_confirmed": int("initial" in episode),
               "replay": replay, "episode_hashes": artifact_hashes(directory)}
    return receipt


def run_exploration(*, config, tasks_path, output, spec, decoding, workers=1,
                    inputs=(), worker_fn=explore_one):
    spec = {**spec, **context_options(config)}
    if decoding != "greedy":
        raise ValueError("This port supports explicit greedy exploration only")
    if (spec["enable_thinking"] is not False or spec["max_tokens"] != 1024
            or config["environment"]["max_episode_steps"] != 40):
        raise ValueError("Exploration fixes OFF, 1024 tokens and 40 policy turns")
    if workers < 1:
        raise ValueError("Positive workers required")
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    listing = json.loads(Path(tasks_path).read_text())
    tasks = training_tasks(listing, manifest, freeze)  # Before directories/clients/environment.
    output = Path(output).resolve()
    if output.is_relative_to(Path(config["paths"]["verified"]).resolve()):
        raise ValueError("Student expansion cannot be written into trusted seed storage")
    output.mkdir(parents=True, exist_ok=False)
    effective = deepcopy(config)
    effective["exploration"] = {"source": "student_exploration", "expansion": True,
        "training_eligible": True, "trusted_seed": False, "trusted_trajectories_added": 0,
        "decoding": decoding, "student": spec, "workers": workers,
        "task_list_sha256": listing["list_sha256"], "replay_required": True,
        "eligibility_rule": "complete known environment outcome and successful replay"}
    inputs = [*inputs, tasks_path, config["paths"]["manifest"], config["paths"]["split_freeze"]]
    write_provenance(output, command="student_exploration", config=effective, inputs=inputs)
    write_json(output / "tasks.json", listing)
    jobs = [(task, effective, str(output / "episodes" / f"{i:03d}-{digest(task['task_id'])[:12]}"),
             spec, listing.get("parent_seed_ids", {}).get(task["task_id"], []), inputs)
            for i, task in enumerate(tasks)]
    started = time.monotonic()
    if workers == 1:
        receipts = [worker_fn(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            receipts = list(pool.map(worker_fn, jobs))
    rows = [task_result(task, job[2]) for task, job in zip(tasks, jobs)]
    summary = summarize(rows, time.monotonic() - started)
    for receipt in receipts:
        receipt["episode_dir"] = str(Path(receipt["episode_dir"]).relative_to(output))
    summary.update(source="student_exploration", expansion=True, training_eligible=True,
                   trusted_seed=False, trusted_trajectories_added=0,
                   training_eligible_episodes=sum(r["training_eligible"] for r in receipts),
                   task_list_sha256=listing["list_sha256"],
                   environment_episodes=sum(r["environment_episodes"] for r in receipts),
                   environment_resets_confirmed=sum(r["environment_resets_confirmed"] for r in receipts),
                   replay_errors=sum(r["replay"]["status"] == "error" for r in receipts),
                   environment_steps=summary["totals"]["environment_steps"],
                   replay_environment_episodes=sum(r["replay"]["environment_episodes"] for r in receipts),
                   replay_environment_steps=sum(r["replay"]["environment_steps"] for r in receipts),
                   replay_environment_steps_unknown=sum(r["replay"]["environment_steps_unknown"] for r in receipts),
                   teacher_calls=0, training_exposure=0)
    write_json(output / "receipts.json", receipts)
    write_json(output / "summary.json", summary)
    return summary


def checked_rollouts(root, manifest, freeze):
    root = Path(root).resolve()
    listing = json.loads((root / "tasks.json").read_text())
    tasks = training_tasks(listing, manifest, freeze)
    receipts = json.loads((root / "receipts.json").read_text())
    if [r["task_id"] for r in receipts] != [t["task_id"] for t in tasks]:
        raise ValueError("Exploration receipts differ from frozen task list")
    result = []
    for receipt, task in zip(receipts, tasks):
        directory = (root / receipt["episode_dir"]).resolve()
        if not directory.is_relative_to(root) or artifact_hashes(directory) != receipt["episode_hashes"]:
            raise ValueError("Exploration evidence changed")
        if (receipt.get("source") != "student_exploration" or receipt.get("trusted_seed") is not False
                or not receipt.get("training_eligible") or type(receipt["won"]) is not bool):
            raise ValueError("Unknown/incomplete exploration cannot become a negative pair")
        episode = json.loads((directory / "episode.json").read_text())
        replay = json.loads((directory / "replay.json").read_text())
        if (episode["task_id"] != task["task_id"] or episode["split"] != "train"
                or episode["game_sha256"] != task["game_sha256"] or not episode["complete"]
                or episode["status"] != "ok" or episode["success"] != receipt["won"]
                or not replay["matched"] or replay != receipt["replay"]):
            raise ValueError("Exploration receipt disagrees with real execution/replay")
        result.append((receipt, episode))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--decoding", choices=["greedy"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-provenance", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["server_provenance"] = json.loads(Path(args.server_provenance).read_text())
    spec = dict(endpoint=args.endpoint, model=args.model_name,
                revision=config["models"]["student"]["revision"], enable_thinking=False, max_tokens=1024)
    result = run_exploration(config=config, tasks_path=args.tasks, output=args.output, spec=spec,
                decoding=args.decoding, workers=args.workers, inputs=[args.config, args.server_provenance])
    return int(result["training_eligible_episodes"] != result["episodes"])


if __name__ == "__main__":
    raise SystemExit(main())
