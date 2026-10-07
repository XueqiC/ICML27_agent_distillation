"""Parallel IPR exploration/MC, with hash-checked TRAIN evidence and no teacher."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

from baselines.ipr.branch import run_branch
from baselines.ipr.sampling import MCStudent, rollout_seed
from sed.config import load_config
from sed.manifest import authorize_task, load_frozen, resolve_task, verify_task_files
from sed.provenance import write_provenance
from sed.runner import _environment, episode_metrics
from sed.seed_evidence import artifact_hashes, checked_episode, load_verified_collection
from sed.storage import digest, file_hash, write_json
from students.vllm_student import Student, context_options


def read(path):
    return json.loads(Path(path).read_text())


def seeds(verified, config):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, root = load_verified_collection(verified, manifest, freeze)
    result = []
    for receipt in collection["verified"]:
        episode = checked_episode(root, receipt)
        task = resolve_task(manifest, freeze, episode["task_id"])
        authorize_task(task, "generation")
        if task["source_split"] != "train":
            raise ValueError("Only seed TRAIN games are eligible")
        result.append((task, episode))
    return result


def step_values(receipts):
    """Preserve all five outcomes, including unknowns; never average a subset."""
    grouped = {}
    for r in receipts:
        key = (r["task_id"], r["j"])
        grouped.setdefault(key, []).append(r)
    result = []
    for (task_id, j), rows in grouped.items():
        outcomes = [r["success"] for r in rows]
        known = len(rows) == 5 and all(type(v) is bool for v in outcomes)
        result.append({"task_id": task_id, "j": j, "prefix": j + 1,
            "outcomes": outcomes, "rollout_seeds": [r["rollout_seed"] for r in rows],
            "mean_final_success": sum(outcomes) / 5 if known else None})
    return result


def plan_rollouts(experts, *, kind, iteration=0, sampling_seed=0, exploration=None):
    if kind not in {"expert-mc", "explore", "agent-mc"} or (
            iteration != 0 if kind == "expert-mc" else iteration not in range(1, 7)):
        raise ValueError("Invalid IPR stage/iteration")
    explored = {(r["task_id"], r["j"]): e for r, e in (exploration or [])}
    jobs = []
    for task, expert in experts:
        for j in range(len(expert["steps"])):
            parent = explored[(task["task_id"], j)] if kind == "agent-mc" else expert
            prefix = j if kind == "explore" else j + 1
            for sample in range(1 if kind == "explore" else 5):
                lineage = {"task_id": task["task_id"], "parent_seed_id": expert["episode_id"],
                    "kind": kind, "iteration": iteration, "j": j, "sample": sample,
                    "sampling_seed": sampling_seed,
                    "rollout_seed": rollout_seed(sampling_seed, expert["episode_id"], task["task_id"],
                                                  kind, iteration, j, sample),
                    "parent_episode_sha256": digest(parent), "prefix": prefix}
                jobs.append((task, parent, lineage))
    return jobs


def rollout_one(job, *, env_factory=_environment, student_factory=Student, mc_factory=MCStudent):
    task, parent, lineage, config, spec, directory, inputs = job
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    episode = {}
    try:
        verify_task_files(config["environment"]["data_root"], task)
        policy = (student_factory(log_dir=directory / "calls", **spec) if lineage["kind"] == "explore"
                  else mc_factory(log_dir=directory / "calls", sampling_seed=lineage["rollout_seed"], **spec))
        episode = run_branch(env_factory(config, task), policy, task, config, parent, lineage["prefix"],
            directory / "episode", lineage=lineage, inputs=inputs)
    except Exception as exc:
        write_json(directory / "failure.json", {"status": "error", "error_type": type(exc).__name__})
        if (directory / "episode/episode.json").exists():
            episode = read(directory / "episode/episode.json")
    known = (episode.get("complete") is True and episode.get("status") == "ok"
             and type(episode.get("success")) is bool
             and episode.get("ipr_branch", {}).get("replay_matched") is True)
    metrics = episode_metrics(episode) if episode else {}
    prefix_steps = episode.get("steps", [])[:lineage["prefix"]]
    receipt = {**lineage, "group_id": task["group_id"], "training_eligible": known,
        "success": episode.get("success", "unknown") if known else "unknown",
        "source": "ipr_real_environment_branch", "trusted_seed": False, "trusted_trajectories_added": 0,
        "environment_episodes": int(bool(episode)), "environment_resets_confirmed": int("initial" in episode),
        "environment_steps": metrics.get("environment_steps", 0),
        "prefix_environment_steps": sum(s["environment_steps"] for s in prefix_steps),
        "metrics": metrics, "artifact_sha256": artifact_hashes(directory)}
    write_json(directory / "receipt.json", receipt)
    return {**receipt, "directory": str(directory)}


def dispatch(jobs, workers, worker_fn=rollout_one):
    if type(workers) is not int or workers < 1:
        raise ValueError("Positive workers required")
    if workers == 1:
        return [worker_fn(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(worker_fn, jobs))  # Frozen plan order, not completion order.


def checked_rollouts(root, config, *, expected_kind=None, expected_iteration=None):
    root = Path(root).resolve()
    provenance = read(root / "provenance.json")
    details = provenance["details"]
    if ((expected_kind is not None and details["kind"] != expected_kind)
            or (expected_iteration is not None and details["iteration"] != expected_iteration)):
        raise ValueError("IPR rollout stage identity differs")
    for path, expected in provenance["inputs"].items():
        if file_hash(path) != expected:
            raise ValueError("IPR rollout source evidence changed")
    experts = seeds(details["verified"], config)
    exploration = (checked_rollouts(details["exploration"], config, expected_kind="explore",
                   expected_iteration=details["iteration"]) if details["kind"] == "agent-mc" else None)
    plan = plan_rollouts(experts, kind=details["kind"], iteration=details["iteration"],
                         sampling_seed=details["sampling_seed"], exploration=exploration)
    receipts = read(root / "receipts.json")
    if len(receipts) != len(plan) or read(root / "plan.json") != [p[2] for p in plan]:
        raise ValueError("IPR rollouts differ from exhaustive seed-prefix plan")
    if details["kind"] != "explore" and read(root / "values.json") != step_values(receipts):
        raise ValueError("IPR step values differ from all five outcomes")
    checked = []
    for receipt, (task, parent, lineage) in zip(receipts, plan):
        directory = (root / receipt["directory"]).resolve()
        if not directory.is_relative_to(root / "rollouts"):
            raise ValueError("Rollout path escapes evidence root")
        stored = read(directory / "receipt.json")
        hashes = artifact_hashes(directory)
        hashes.pop("receipt.json")
        if (stored != {k: v for k, v in receipt.items() if k != "directory"}
                or hashes != receipt["artifact_sha256"] or any(receipt[k] != v for k, v in lineage.items())):
            raise ValueError("IPR rollout evidence changed")
        if not receipt["training_eligible"] or type(receipt["success"]) is not bool:
            raise ValueError("Unknown/incomplete IPR execution cannot become training evidence")
        episode = read(directory / "episode/episode.json")
        branch = episode["ipr_branch"]
        if (episode["task_id"] != task["task_id"] or episode["game_sha256"] != task["game_sha256"]
                or episode["split"] != "train" or not episode["complete"] or episode["status"] != "ok"
                or episode["success"] != receipt["success"] or not branch["replay_matched"]
                or any(branch[k] != v for k, v in lineage.items())):
            raise ValueError("IPR receipt disagrees with real execution")
        checked.append((receipt, episode))
    return checked


def run_rollouts(*, config, verified, kind, iteration, output, spec, server_provenance,
                 exploration=None, workers=8, sampling_seed=0, inputs=(), worker_fn=rollout_one):
    if (config["environment"]["max_episode_steps"] != 40 or spec["enable_thinking"] is not False
            or spec["max_tokens"] != (1024 if kind == "explore" else 512)):
        raise ValueError("IPR fixes 40 turns, OFF, exploration 1024 / MC 512 tokens")
    student = config["models"]["student"]
    if spec["model"] != student["requested_repo"] or spec["revision"] != student["revision"]:
        raise ValueError("IPR cannot substitute the configured student")
    server = read(server_provenance)
    spec = {**spec, **context_options({"server_provenance": server})}
    if server["config"]["server"]["batch_invariant"] is not True:
        raise ValueError("IPR requires batch-invariant server provenance")
    experts = seeds(verified, config)
    explored = (checked_rollouts(exploration, config, expected_kind="explore", expected_iteration=iteration)
                if kind == "agent-mc" else None)
    plan = plan_rollouts(experts, kind=kind, iteration=iteration, sampling_seed=sampling_seed, exploration=explored)
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(config["paths"][p]).resolve()) for p in ("raw", "verified")):
        raise ValueError("IPR outputs must be outside trusted seed storage")
    output.mkdir(parents=True, exist_ok=False)
    effective = deepcopy(config)
    effective["server_provenance"] = server
    effective["ipr_rollouts"] = {"kind": kind, "iteration": iteration, "student": spec, "workers": workers,
        "sampling_seed": sampling_seed, "batch_invariant": True, "training_exposure": 0,
        "expert_value_index": "prefix j+1 after zero-based expert action j", "mc_samples": 5}
    inputs = [*inputs, server_provenance, Path(verified) / "manifest.json",
              config["paths"]["manifest"], config["paths"]["split_freeze"]]
    if exploration:
        inputs += [Path(exploration) / p for p in ("receipts.json", "provenance.json", "plan.json")]
    write_provenance(output, command="ipr_" + kind, config=effective, inputs=inputs, details={
        "verified": str(Path(verified).resolve()), "kind": kind, "iteration": iteration,
        "exploration": str(Path(exploration).resolve()) if exploration else None, "sampling_seed": sampling_seed})
    write_json(output / "plan.json", [p[2] for p in plan])
    jobs = [(task, parent, lineage, effective, spec, str(output / "rollouts" / f"{i:06d}"), inputs)
            for i, (task, parent, lineage) in enumerate(plan)]
    receipts = dispatch(jobs, workers, worker_fn)
    for receipt in receipts:
        receipt["directory"] = str(Path(receipt["directory"]).relative_to(output))
    write_json(output / "receipts.json", receipts)
    if kind != "explore":
        write_json(output / "values.json", step_values(receipts))
    summary = {"kind": kind, "iteration": iteration, "rollouts": len(receipts),
        "context_overflow_episodes": sum(r["metrics"].get("context_overflow_episodes", 0) for r in receipts),
        "eligible": sum(r["training_eligible"] for r in receipts), "K": len(experts),
        "unique_instances": len({t["task_id"] for t, _ in experts}),
        "task_groups": len({t["group_id"] for t, _ in experts}), "teacher_calls": 0,
        "trusted_trajectories_added": 0, "training_exposure": 0,
        **{key: sum(r[key] for r in receipts) for key in ("environment_episodes",
            "environment_resets_confirmed", "environment_steps", "prefix_environment_steps")}}
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    for name in ("verified", "endpoint", "model-name", "output", "server-provenance"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--kind", choices=["expert-mc", "explore", "agent-mc"], required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exploration")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    spec = {"endpoint": args.endpoint, "model": args.model_name, "revision": config["models"]["student"]["revision"],
            "enable_thinking": False, "max_tokens": 1024 if args.kind == "explore" else 512}
    summary = run_rollouts(config=config, verified=args.verified, kind=args.kind, iteration=args.iteration,
        output=args.output, spec=spec, server_provenance=args.server_provenance, exploration=args.exploration,
        workers=args.workers, sampling_seed=args.sampling_seed, inputs=[args.config])
    return int(summary["eligible"] != summary["rollouts"])


if __name__ == "__main__":
    raise SystemExit(main())
