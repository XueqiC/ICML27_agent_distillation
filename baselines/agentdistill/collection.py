"""At most three actually reserved teacher runs per verified support instance."""
from pathlib import Path

from baselines.common.explore import training_tasks
from baselines.agentdistill.policy import Manager, ToolPolicy
from baselines.agentdistill.teacher import read
from sed.manifest import frozen_write, load_frozen, verify_task_files
from sed.provenance import write_provenance
from sed.runner import _environment, run_episode, episode_metrics
from sed.seed_evidence import artifact_hashes, environment_won
from sed.storage import digest, write_json


def tasks_for(cell, config):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    listing = read(Path(cell) / "tasks.json")
    return [dict(task, parent_seed_ids=listing["parent_seed_ids"][task["task_id"]])
            for task in training_tasks(listing, manifest, freeze)]


def checked_run(directory):
    directory = Path(directory)
    receipt = read(directory / "receipt.json")
    hashes = artifact_hashes(directory)
    hashes.pop("receipt.json")
    if receipt["artifact_sha256"] != hashes:
        raise ValueError("Teacher run evidence changed")
    return receipt


def collect(tasks, config, cell, output, identity, budget, *, env_factory=_environment):
    output, cell = Path(output), Path(cell)
    output.mkdir(parents=True, exist_ok=False)
    pool, runs = [], []
    for task in tasks:
        for index in range(3):
            directory = cell / "runs" / digest(task["task_id"]) / f"run-{index}"
            reservation = directory / "reservation.json"
            binding = {"cell": digest(identity), "task_id": task["task_id"], "run": index,
                       "parent_seed_ids": task["parent_seed_ids"]}
            if (directory / "receipt.json").exists():
                frozen_write(reservation, binding)
                receipt = checked_run(directory)
            else:
                interrupted = reservation.exists()
                frozen_write(reservation, binding)  # durable BEFORE reset or teacher request
                status = "interrupted_unknown" if interrupted else "started"
                if not interrupted:
                    policy = ToolPolicy(Manager(budget, task, index), [], directory / "calls")
                    try:
                        verify_task_files(config["environment"]["data_root"], task)
                        episode = run_episode(env_factory(config, task), policy, task, config, directory / "episode",
                            policy_condition="agentdistill_teacher_different_interface", purpose="generation",
                            parent_seed_ids=task["parent_seed_ids"], input_paths=[cell / "identity.json"])
                        policy.finish(episode)
                        write_json(directory / "episode/episode.json", episode)
                        status = "ok"
                    except BaseException as exc:
                        write_json(directory / "failure.json", {"error_type": type(exc).__name__})
                        # Never free a reservation. Transport failures remain halted in
                        # the cell-wide CappedTeacher; interrupts retain an unknown run.
                        raise
                episode_path = directory / "episode/episode.json"
                episode = read(episode_path) if episode_path.exists() else {}
                won = status == "ok" and environment_won(directory / "episode", episode)
                accepted = []
                if won:
                    for row in policy.records:
                        if row["status"] == "completed" and row["actions"] > 0:
                            tool = row["tool"]
                            if tool not in accepted:
                                accepted.append(tool)
                receipt = {**binding, "status": status, "success": episode.get("success", "unknown"),
                    "accepted_tools": accepted, "trusted_trajectories_added": 0,
                    "source": "agentdistill_teacher_support_run", "expansion": True,
                    "environment_steps": episode_metrics(episode)["environment_steps"] if episode else 0,
                    "artifact_sha256": artifact_hashes(directory)}
                write_json(directory / "receipt.json", receipt, exclusive=True)
            runs.append({"directory": str(directory), "receipt_sha256": digest(receipt)})
            for tool in receipt["accepted_tools"]:
                pool.append({"id": digest([binding, tool]), "tool": tool, "task_id": task["task_id"],
                    "parent_seed_ids": task["parent_seed_ids"], "run": index,
                    "source_directory": str(directory), "receipt_sha256": digest(receipt)})
            if receipt["status"] == "ok" and receipt["success"] is True:
                break
    write_provenance(output, command="agentdistill_collect", config=config,
        inputs=[cell / "identity.json", cell / "tasks.json"], details={"cell": digest(identity)})
    write_json(output / "pool.json", pool)
    write_json(output / "runs.json", runs)
    write_json(output / "result.json", {"runs": len(runs), "tools": len(pool), "training_exposure": 0})


def checked_pool(path, tasks):
    pool = read(path)
    task_map = {t["task_id"]: t for t in tasks}
    if len({r["id"] for r in pool}) != len(pool):
        raise ValueError("Duplicate MCP source")
    for row in pool:
        receipt = checked_run(row["source_directory"])
        episode_dir = Path(row["source_directory"]) / "episode"
        episode = read(episode_dir / "episode.json")
        if (row["task_id"] not in task_map or row["parent_seed_ids"] != task_map[row["task_id"]]["parent_seed_ids"]
                or digest(receipt) != row["receipt_sha256"] or receipt["task_id"] != row["task_id"]
                or receipt["status"] != "ok" or receipt["success"] is not True
                or not environment_won(episode_dir, episode)
                or row["tool"] not in receipt["accepted_tools"]):
            raise ValueError("Only executed successful support runs supply MCPs")
    return pool
