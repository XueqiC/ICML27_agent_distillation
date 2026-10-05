"""Turn-t supervision from ORIGINAL harness history, seeded min aggregation."""
from copy import deepcopy
import json
from pathlib import Path

from baselines.common.explore import checked_rollouts
from baselines.leap.protocol import require_train, parse_correction
from baselines.leap.teacher import checked_outcome, read
from sed.manifest import load_frozen, resolve_task
from sed.provenance import write_provenance
from sed.react import build_messages
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import digest, file_hash, write_json
from sed.thinking import split_thinking
from sed.visibility import visible_observation
from train.dataset import encode_turn


def seeds(verified, config):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, root = load_verified_collection(verified, manifest, freeze)
    result = []
    for receipt in collection["verified"]:
        episode = checked_episode(root, receipt)
        task = resolve_task(manifest, freeze, receipt["task_id"])
        require_train(task)
        task["parent_seed_ids"] = [episode["episode_id"]]
        result.append((task, episode))
    return result


def step_rows(episode, *, parents, corrections=None):
    require_train(episode)
    if episode.get("schema_version") != 2:
        raise ValueError("Harness ReAct episode required")
    if corrections is not None:
        # Re-validate the label-only schema without looking at oracle inputs.
        corrections = parse_correction(json.dumps({"summary": "", "trajectory": corrections}), len(episode["steps"]))
    visible = visible_observation(episode["initial"]["visible"])
    history, rows = [], []
    for index, step in enumerate(episode["steps"]):
        original, thinking = split_thinking(step["action_raw"])
        if thinking:
            raise ValueError("Hidden thought channel in original history")
        target = original if corrections is None else (
            f"Thought: {corrections[index]['corrected_reason']}\nAction: {corrections[index]['corrected_action']}")
        rows.append({"task_id": episode["task_id"], "parent_seed_ids": list(parents),
            "episode_id": episode["episode_id"], "event_id": step["event_id"], "turn_index": index,
            "source": "trusted_teacher_seed" if corrections is None else "leap_teacher_correction",
            "messages": build_messages(history, {}, visible), "target": target,
            "inadmissible_correction": corrections is not None and
                corrections[index]["corrected_action"] not in visible["admissible_commands"]})
        history.append({"visible": visible, "reply": original})  # NEVER target / echoed teacher history.
        visible = visible_observation(step["visible"])
    return rows


def min_subsample(sets, *, seed=0):
    """HF Dataset.shuffle uses np.default_rng(seed).permutation, per directory.

    Port of a3376d8 scripts/train/sft.py:59-72. User fixes seed 0 instead
    of upstream 42; reset the RNG for EACH directory and for merged shuffle.
    Empty sets produce an explicit empty aggregate, never seed-only fallback.
    """
    import numpy as np
    if not sets:
        raise ValueError("Expected iter0..i example sets")
    minimum = min(map(len, sets))
    indices = [np.random.default_rng(seed).permutation(len(rows))[:minimum].tolist() for rows in sets]
    merged = [deepcopy(rows[index]) for rows, selected in zip(sets, indices) for index in selected]
    order = np.random.default_rng(seed).permutation(len(merged)).tolist()
    return [merged[index] for index in order], {"seed": seed, "minimum": minimum,
        "before": list(map(len, sets)), "after": [minimum] * len(sets), "selected_indices": indices,
        "merged_order": order, "N_examples": len(merged)}


def set_values(verified, config, *, iteration, corrections=None, exploration=None):
    bank = seeds(verified, config)
    if iteration == 0:
        if corrections or exploration:
            raise ValueError("Iteration 0 is BC on all K teacher seeds")
        rows = [row for task, ep in bank for row in step_rows(ep, parents=task["parent_seed_ids"])]
        return rows
    if iteration not in {1, 2, 3} or not corrections or not exploration:
        raise ValueError("Iteration correction evidence missing")
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    rollouts = checked_rollouts(exploration, manifest, freeze)
    parents = {t["task_id"]: t["parent_seed_ids"] for t, _ in bank}
    if [r["task_id"] for r, _ in rollouts] != list(parents):
        raise ValueError("LEAP explores exactly the K seed TRAIN tasks")
    records = read(Path(corrections) / "corrections.json")
    if [r["task_id"] for r in records] != list(parents):
        raise ValueError("Correction coverage differs from seed scope")
    rows = []
    for record, (receipt, episode) in zip(records, rollouts):
        if receipt["parent_seed_ids"] != parents[receipt["task_id"]]:
            raise ValueError("Correction parent seed changed")
        if record["episode_sha256"] != digest(episode):
            raise ValueError("Correction belongs to another rollout")
        if record["status"] != "accepted":
            continue
        if receipt["won"] is not False:
            raise ValueError("Only real failed rollouts can be corrected")
        outcome = checked_outcome(record["request_directory"])
        if (outcome["status"] != "accepted" or outcome["task_id"] != receipt["task_id"]
                or outcome["parent_seed_ids"] != receipt["parent_seed_ids"]
                or file_hash(Path(record["request_directory"]) / "outcome.json") != record["outcome_sha256"]):
            raise ValueError("Correction receipt changed")
        labels = parse_correction(outcome["result"]["raw_text"], len(episode["steps"]))
        if labels != outcome["labels"]:
            raise ValueError("Correction labels differ from sampled output")
        rows.extend(step_rows(episode, parents=receipt["parent_seed_ids"], corrections=labels))
    return rows


def prepare_set(verified, output, config, *, iteration, corrections=None, exploration=None):
    rows = set_values(verified, config, iteration=iteration, corrections=corrections, exploration=exploration)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = [Path(verified) / "manifest.json", config["paths"]["manifest"], config["paths"]["split_freeze"]]
    if corrections:
        inputs += [Path(corrections) / "corrections.json", Path(exploration) / "receipts.json"]
    details = {"iteration": iteration, "verified": str(Path(verified).resolve()),
        "corrections": str(Path(corrections).resolve()) if corrections else None,
        "exploration": str(Path(exploration).resolve()) if exploration else None,
        "rows_sha256": digest(rows)}
    write_provenance(output, command="leap_example_set", config=config, inputs=inputs, details=details)
    write_json(output / "examples.json", rows)
    write_json(output / "report.json", {"iteration": iteration, "examples": len(rows),
        "inadmissible_corrections": sum(r["inadmissible_correction"] for r in rows),
        "unique_instances": len({r["task_id"] for r in rows}), "training_exposure": 0,
        "train_split": 1.0, "checkpoint_selection": False})


def load_set(path, config, *, expected_iteration, verified):
    path = Path(path)
    provenance = read(path.parent / "provenance.json")
    details = provenance["details"]
    if details["iteration"] != expected_iteration or Path(details["verified"]).resolve() != Path(verified).resolve():
        raise ValueError("LEAP dataset chain must contain iter0..i on the same seeds")
    if any(file_hash(source) != sha for source, sha in provenance["inputs"].items()):
        raise ValueError("LEAP dataset source changed")
    rows = set_values(verified, config, iteration=expected_iteration,
        corrections=details["corrections"], exploration=details["exploration"])
    if read(path) != rows or digest(rows) != details["rows_sha256"]:
        raise ValueError("LEAP examples differ from original source history/labels")
    return rows


def encode_rows(rows, tokenizer):
    examples = []
    for row in rows:
        encoded = encode_turn(tokenizer, row["messages"], row["target"], max_length=32768)
        encoded["label_provenance"] = {key: row[key] for key in
            ("task_id", "episode_id", "parent_seed_ids", "event_id", "turn_index", "source")}
        encoded["label_provenance"].update(training_eligible=True, verification="unknown_teacher_label"
            if row["source"] == "leap_teacher_correction" else "executed_seed")
        examples.append(encoded)
    return examples
