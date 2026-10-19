"""Hash-bound, allowlisted method datasets; no serialized event-row prompts."""
import json
from pathlib import Path

from baselines.common.conversation import group_turns
from sed.provenance import write_provenance
from sed.storage import digest, file_hash, write_json
from train.dataset import encode_turn


def read(path):
    return json.loads(Path(path).read_text())


def publish(output, rows, *, method, config, verified, inputs, details=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command=method + "_data", config=config, inputs=inputs,
        details={**(details or {}), "method": method, "verified": str(Path(verified).resolve()),
                 "data_sha256": digest(rows)})
    write_json(output / "data.json", rows)


def checked(path, config, *, method, verified):
    path = Path(path)
    provenance = read(path.parent / "provenance.json")
    details = provenance["details"]
    if (details["method"] != method or Path(details["verified"]).resolve() != Path(verified).resolve()
            or provenance["config"]["models"] != config["models"]):
        raise ValueError("Method data identity differs")
    for source, expected in provenance["inputs"].items():
        if file_hash(source) != expected:
            raise ValueError("Method data source changed")
    rows = read(path)
    if digest(rows) != details["data_sha256"]:
        raise ValueError("Method dataset changed")
    return rows, details


def encode_targets(prompts_targets, tokenizer, *, task_id, parent_seed_id, source):
    turns = []
    for prompt, target in prompts_targets:
        # Only explicit role/content dictionaries from a checked method renderer.
        if any(set(m) != {"role", "content"} or m["role"] not in {"system", "user", "assistant"}
               or not isinstance(m["content"], str) for m in prompt):
            raise ValueError("Invalid method prompt")
        row = encode_turn(tokenizer, prompt, target, max_length=32768)
        row["label_provenance"] = {"task_id": task_id, "parent_seed_ids": [parent_seed_id],
            "source": source, "training_eligible": True, "trusted_seed": False}
        turns.append(row)
    return group_turns(turns)
