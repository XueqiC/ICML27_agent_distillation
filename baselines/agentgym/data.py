"""Checked accumulating AgentEvol data; no cross-iteration deduplication.

Rule adapted from AgentGym 3ef9235 agentenv/utils/agentevol_filter.py:40-72.
Only checked visible conversations are serialized as training content.
"""
import json
import math
from pathlib import Path

from baselines.common.conversation import episode_messages, encode_messages
from baselines.common.explore import checked_rollouts
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import digest, write_json


def accumulate(current, rollouts, seed_tasks):
    """Pure official filter on checked receipts. Stable first successful task wins."""
    allowed, seen, accepted, decisions = set(seed_tasks), set(), [], []
    if not current or any(r["task_id"] not in allowed for r in current):
        raise ValueError("Dataset must stay on the seed TRAIN tasks")
    for index, row in enumerate(rollouts):
        if row["task_id"] not in allowed:
            raise ValueError("AgentEvol explores only seed TRAIN tasks")
        reward = row.get("reward")
        if row.get("training_eligible") is not True or type(reward) not in (int, float) or not math.isfinite(reward):
            reason = "unknown_or_unverified"
        elif reward <= 0.99:
            reason = "reward_not_above_0.99"
        elif row["task_id"] in seen:
            reason = "duplicate_task_within_iteration"
        else:
            reason = "accepted"
            seen.add(row["task_id"])
            accepted.append(row)
        decisions.append({"sample": index, "task_id": row["task_id"], "decision": reason})
    return accepted + list(current), {"D_m": len(current), "F_m": len(accepted),
        "D_next": len(current) + len(accepted), "empty_F": not accepted, "decisions": decisions,
        "order": "F_m then D_m; no cross-iteration deduplication", "training_exposure": 0}


def seed_rows(verified, config):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, root = load_verified_collection(verified, manifest, freeze)
    rows = []
    for receipt in collection["verified"]:
        episode = checked_episode(root, receipt)
        rows.append({"task_id": receipt["task_id"], "episode_id": episode["episode_id"],
            "group_id": receipt["group_id"], "parent_seed_ids": [episode["episode_id"]],
            "source": "trusted_teacher_seed", "training_eligible": True,
            "reward": episode["steps"][-1]["reward"], "messages": episode_messages(episode)})
    return rows


def dataset_values(verified, config, *, previous=None, exploration=None, iteration=0):
    seeds = seed_rows(verified, config)
    if iteration == 0:
        if previous or exploration:
            raise ValueError("D_0 contains only the K seeds")
        return seeds, {"iteration": 0, "D_m": len(seeds), "K": len(seeds), "training_exposure": 0}
    if not previous or not exploration or iteration not in range(1, 5):
        raise ValueError("Accumulation requires D_m and its exploration, m=0..3")
    prior = json.loads((Path(previous).parent / "provenance.json").read_text())["details"]
    if prior["iteration"] != iteration - 1 or Path(prior["verified"]).resolve() != Path(verified).resolve():
        raise ValueError("Dataset chain skipped an iteration or changed seeds")
    current = load_dataset(previous, config)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    checked = checked_rollouts(exploration, manifest, freeze)
    parents = {r["task_id"]: r["parent_seed_ids"] for r in seeds}
    if {r["task_id"] for r, _ in checked} != set(parents):
        raise ValueError("AgentEvol requires exactly one rollout on each seed task")
    rollouts = []
    for receipt, episode in checked:
        if receipt["parent_seed_ids"] != parents[receipt["task_id"]]:
            raise ValueError("Exploration parent seed changed")
        rollouts.append({"task_id": receipt["task_id"], "episode_id": episode["episode_id"],
            "group_id": receipt["group_id"], "parent_seed_ids": receipt["parent_seed_ids"],
            "source": "student_exploration", "training_eligible": receipt["training_eligible"],
            "reward": episode["steps"][-1]["reward"], "messages": episode_messages(episode)})
    rows, report = accumulate(current, rollouts, parents)
    return rows, {**report, "iteration": iteration, "exploration_iteration": iteration - 1,
                  "K": len(seeds), "unique_instances": len(parents),
                  "task_groups": len({r["group_id"] for r in seeds})}


def load_dataset(path, config):
    path = Path(path)
    provenance = json.loads((path.parent / "provenance.json").read_text())
    details = provenance["details"]
    # Reconstruct from checked receipts all the way back to D_0. A valid JSON
    # conversation alone cannot promote diagnostics or an unexecuted label.
    for source, expected in provenance["inputs"].items():
        from sed.storage import file_hash
        if file_hash(source) != expected:
            raise ValueError("AgentEvol source evidence changed")
    rows, report = dataset_values(details["verified"], config, previous=details["previous"],
        exploration=details["exploration"], iteration=details["iteration"])
    if (json.loads(path.read_text()) != rows or digest(rows) != details["dataset_sha256"]
            or json.loads((path.parent / "report.json").read_text()) != report):
        raise ValueError("AgentEvol data differs from checked source evidence")
    return rows


def prepare_dataset(verified, output, config, *, previous=None, exploration=None, iteration=0):
    rows, report = dataset_values(verified, config, previous=previous, exploration=exploration, iteration=iteration)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = [Path(verified) / "manifest.json", config["paths"]["manifest"], config["paths"]["split_freeze"]]
    if previous:
        inputs += [previous, Path(previous).parent / "provenance.json"]
    if exploration:
        inputs += [Path(exploration) / name for name in ("receipts.json", "tasks.json", "provenance.json")]
    write_provenance(output, command="agentgym_dataset", config=config, inputs=inputs, details={
        "verified": str(Path(verified).resolve()), "previous": str(Path(previous).resolve()) if previous else None,
        "exploration": str(Path(exploration).resolve()) if exploration else None,
        "iteration": iteration, "dataset_sha256": digest(rows), "rule": "AgentGym-3ef9235-add_original_data-True"})
    write_json(output / f"train_iter_{iteration}.json", rows)
    write_json(output / "report.json", report)
    return report


def training_examples(path, config, tokenizer, *, packing="turn", max_length=32768):
    examples = []
    for index, row in enumerate(load_dataset(path, config)):
        example = encode_messages(row["messages"], tokenizer, task_id=row["task_id"],
                                  packing=packing, max_length=max_length)
        provenance = example["label_provenance"]
        provenance.update(source=row["source"], trusted_seed=row["source"] == "trusted_teacher_seed",
            parent_seed_ids=row["parent_seed_ids"], source_episode_id=row["episode_id"], dataset_index=index)
        for turn in provenance["turns"]:
            turn["source"] = row["source"]
        examples.append(example)
    return examples
