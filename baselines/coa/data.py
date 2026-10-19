"""Executed teacher traces, deterministic filtering and hash-checked TRAIN loaders."""
import json
from pathlib import Path
import re

from baselines.coa import RECIPE
from baselines.coa.protocol import alpaca_record, blocks, encode_episode, episode_turns
from baselines.coa.teacher import MultiAgentTeacher, budget, require_cpu
from baselines.common import registered_support
from baselines.common.explore import training_tasks
from eval.run_suite import task_result, summarize
from sed.manifest import authorize_task, load_frozen, verify_task_files
from sed.provenance import write_provenance
from sed.runner import _environment, run_episode, replay_episode
from sed.seed_evidence import artifact_hashes, checked_episode, load_verified_collection
from sed.storage import digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def support(config, verified):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, selection, root = load_verified_collection(verified, manifest, freeze)
    if not registered_support(selection):
        raise ValueError("Unregistered support collection")
    parents = {r["task_id"]: [checked_episode(root, r)["episode_id"]] for r in collection["verified"]}
    listing = {"task_ids": list(parents), "parent_seed_ids": parents, "list_sha256": digest(list(parents)),
               "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze)}
    return listing, training_tasks(listing, manifest, freeze)


def checked_record(record):
    directory = Path(record["directory"])
    hashes = artifact_hashes(directory)
    hashes.pop("receipt.json", None)
    if hashes != record["artifacts"] or read(directory / "receipt.json") != record:
        raise ValueError("CoA execution evidence changed")
    episode = read(directory / "episode/episode.json") if (directory / "episode/episode.json").exists() else None
    if episode is not None and (episode["task_id"] != record["task_id"]
            or episode.get("sealed") or episode["split"] != ("dev" if record["evaluation"] else "train")):
        raise ValueError("CoA execution task/split differs from its receipt")
    if record["training_eligible"] and (episode is None or episode.get("status") != "ok"
            or not episode.get("complete") or episode.get("success") != record["success"]
            or not record["replay"]["matched"]):
        raise ValueError("CoA receipt lacks complete executed/replayed evidence")
    return episode


def collect(config, listing, tasks, output, cell, identity, *, evaluation=False,
            env_factory=_environment, capped=None):
    require_cpu()
    output, cell = Path(output), Path(cell)
    for task in tasks:
        authorize_task(task, "evaluation" if evaluation else "generation")
        if (evaluation and (task["split"] != "dev" or task.get("sealed"))) or (
                not evaluation and task["source_split"] != "train"):
            raise ValueError("CoA task outside the declared TRAIN/dev scope")
    if config["environment"]["max_episode_steps"] != 40:
        raise ValueError("CoA teacher cap requires the shared 40-turn H0 limit")
    namespace = "teacher_eval" if evaluation else "teacher"
    capped = capped or budget(config, cell / namespace, identity, tasks, evaluation=evaluation)
    runs = RECIPE["teacher_eval_runs" if evaluation else "teacher_runs"]
    records, rows = [], []
    for task in tasks:
        for index in range(runs):
            directory = cell / namespace / "executions" / digest([task["task_id"], index])
            parents = [] if evaluation else listing["parent_seed_ids"][task["task_id"]]
            binding = {"identity": digest(identity), "task_id": task["task_id"], "run_index": index,
                       "evaluation": evaluation, "parent_seed_ids": parents}
            if (directory / "receipt.json").exists():
                record = read(directory / "receipt.json")
                checked_record(record)
                if read(directory / "start.json") != binding:
                    raise ValueError("System run binding changed")
                records.append(record)
                if evaluation:
                    rows.append(task_result(task, directory / "episode"))
                continue
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "start.json").exists():
                # No reset/replay/cache recombination that pretends to be the
                # interrupted run. Its reserved slot remains spent.
                if read(directory / "start.json") != binding:
                    raise ValueError("Interrupted system run binding changed")
                write_json(directory / "interrupted.json", {"status": "interrupted",
                    "verification": "unknown", "run_slot_spent": True})
                record = {**binding, "status": "interrupted", "directory": str(directory.resolve()),
                    "success": "unknown", "replay": {"matched": False, "environment_episodes": 0,
                    "environment_steps": 0, "unknown_costs_possible": True},
                    "source": "coa_multi_agent_teacher", "trusted_seed": False,
                    "trusted_trajectories_added": 0, "training_eligible": False,
                    "artifacts": artifact_hashes(directory)}
                write_json(directory / "receipt.json", record, exclusive=True)
                records.append(record)
                if evaluation:
                    rows.append(task_result(task, directory / "episode"))
                continue
            write_json(directory / "start.json", binding, exclusive=True)
            replay = {"matched": False, "environment_episodes": 0, "environment_steps": 0}
            status, episode = "error", None
            try:
                verify_task_files(config["environment"]["data_root"], task)
                policy = MultiAgentTeacher(capped, task, index, directory)
                episode = run_episode(env_factory(config, task), policy, task, config, directory / "episode",
                    policy_condition="coa_multi_agent_teacher", purpose="evaluation" if evaluation else "generation",
                    parent_seed_ids=parents, input_paths=[config["paths"]["manifest"], config["paths"]["split_freeze"]])
                if not evaluation:
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
                            return self.env.step(action)
                    replay.update(replay_episode(CountingEnvironment(env_factory(config, task)), episode))
                status = "ok"
            except Exception as exc:
                write_json(directory / "failure.json", {"error_type": type(exc).__name__})
            record = {**binding, "status": status, "directory": str(directory.resolve()),
                "success": episode.get("success", "unknown") if episode else "unknown",
                "replay": replay, "source": "coa_multi_agent_teacher", "trusted_seed": False,
                "trusted_trajectories_added": 0, "training_eligible": not evaluation and status == "ok" and replay["matched"],
                "artifacts": artifact_hashes(directory)}
            write_json(directory / "receipt.json", record, exclusive=True)
            records.append(record)
            if evaluation:
                rows.append(task_result(task, directory / "episode"))
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="coa_teacher_evaluation" if evaluation else "coa_teacher_collection",
        config=config, inputs=[config["paths"]["manifest"], config["paths"]["split_freeze"]],
        details={"identity": digest(identity), "evaluation_only": evaluation, "runs_per_task": runs,
                 "ledger": str(cell / namespace / "usage.jsonl"), "listing_sha256": digest(listing)})
    write_json(output / "receipts.json", records)
    if evaluation:
        write_json(output / "per_task.json", rows)
        write_json(output / "summary.json", {**summarize(rows, 0), "teacher_system": "coa",
            "training_exposure": 0, "teacher_cap": len(tasks) * runs * 40 * 8})


def filter_episode(episode, *, run_index):
    interactions = sum(s.get("environment_steps", 0) for s in episode["steps"])
    result = {"copies": 0, "interactions": interactions, "reflective": False, "error_correction": False}
    if episode.get("success") is not True or episode.get("status") != "ok":
        return {**result, "reason": "not_environment_success"}
    if interactions < RECIPE["min_interactions"]:
        return {**result, "reason": "fewer_than_five_interactions"}
    try:
        turns = episode_turns(episode)
        parsed = [blocks(t["reply"]) for t in turns]
        if any(s.get("action_outcome") == "parser_error" for s in episode["steps"]):
            raise ValueError("Invalid tool call")
    except (ValueError, KeyError):
        return {**result, "reason": "malformed_chain"}
    reflective = any(role == "reflection" for turn in parsed for role, _ in turn)
    flattened = [part for turn in parsed for part in turn]
    low = [i for i, (role, text) in enumerate(flattened)
           if role == "double_check" and re.search(r"\bScore:\s*[12]\b", text)]
    # Public low-credibility assessment followed by later reasoning and a real
    # eventual win; never a GRM model or a teacher success assertion.
    correction = any(any(role in {"think", "plan", "reflection"}
                         for role, _ in flattened[i + 1:]) for i in low)
    keep = (int(digest([0, episode["task_id"], run_index])[:16], 16) / 2**64
            < RECIPE["nonreflection_keep_probability"])
    copies = (RECIPE["error_correction_copies"] if correction else 1) if reflective or keep else 0
    return {**result, "reflective": reflective, "error_correction": correction, "copies": copies,
            "reason": "accepted" if copies else "nonreflective_downsample"}


def datasets(collected, output, config, verified):
    listing, _ = support(config, verified)
    records = read(Path(collected) / "receipts.json")
    selected, decisions, exported = [], [], []
    for record in records:
        if (record["evaluation"] or record["task_id"] not in listing["task_ids"]
                or record["parent_seed_ids"] != listing["parent_seed_ids"][record["task_id"]]):
            raise ValueError("Teacher source outside verified support TRAIN")
        episode = checked_record(record)
        decision = filter_episode(episode, run_index=record["run_index"]) if record["training_eligible"] else {
            "copies": 0, "reason": "incomplete_or_unverified"}
        decisions.append({"task_id": record["task_id"], "run_index": record["run_index"], **decision})
        for copy in range(decision["copies"]):
            selected.append({"record": record, "copy": copy})
            exported.append(alpaca_record(episode))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "data.json", selected)
    write_json(output / "alpaca.json", exported)
    write_json(output / "filter.json", decisions)
    write_provenance(output, command="coa_agentic_sft_dataset", config=config,
        inputs=[Path(collected) / "receipts.json", Path(verified) / "manifest.json"],
        details={"data_sha256": digest(selected), "listing_sha256": digest(listing),
            "verified": str(Path(verified).resolve()), "accepted_exposures": len(selected),
            "accepted_unique_trajectories": sum(d["copies"] > 0 for d in decisions),
            "trusted_trajectories_added": 0})


def training_examples(path, config, tokenizer, verified):
    rows = read(path)
    details = read(Path(path).parent / "provenance.json")["details"]
    listing, _ = support(config, verified)
    if (details["data_sha256"] != digest(rows) or details["listing_sha256"] != digest(listing)
            or details["verified"] != str(Path(verified).resolve())):
        raise ValueError("SFT data/support binding changed")
    examples = []
    for row in rows:
        record = row["record"]
        if (record["evaluation"] or not record["training_eligible"] or record["task_id"] not in listing["task_ids"]
                or record["parent_seed_ids"] != listing["parent_seed_ids"][record["task_id"]]):
            raise ValueError("Only checked support TRAIN trajectories may train")
        episode = checked_record(record)
        decision = filter_episode(episode, run_index=record["run_index"])
        if not 0 <= row["copy"] < decision["copies"]:
            raise ValueError("SFT resampling changed")
        examples.append(encode_episode(episode, tokenizer))
    return examples
