"""Shared CPU-only validation for sealed V2 training bundles."""
from collections import Counter
import json
from pathlib import Path
import re

from repair.v2_inputs import require
from sed.storage import digest, file_hash

SCHEMA = "v2-train-interchange/1"
ARMS = {
    "JOINT": {"joint_local", "joint_complete"},
    "JOINT_COMPLETE_ONLY": {"joint_complete"},
    "PREFIX_EXPERT": {"prefix_expert"},
    "OFFLINE": {"offline_teacher"},
    "CONTINUE": set(),
}
EXAMPLE_FIELDS = {"example_id", "arm", "source", "kind", "task_id", "group_id",
                  "task_type", "messages", "supervised_assistant_indices", "weight", "evidence"}
EVIDENCE_FIELDS = {"episode_ref", "step", "executor", "local_event", "episode_success",
                   "label_source", "label_check"}
LOCAL_EVENTS = {"target_seen", "target_taken", "hands_freed", "transform_done",
                "placed", "lamp_used_with_target", "task_success"}
EVIDENCE_HASH_FIELDS = {"public_prefix_sha256", "fact_state_sha256",
                        "facts_before_sha256", "facts_after_sha256"}
DECISION_EVIDENCE_FIELDS = EVIDENCE_HASH_FIELDS | {"local_events", "complete_success_eligible",
                                                "history_rewrites", "next_legal_target_take"}
LABEL_EVIDENCE_PAIRS = (
    ("student_thought", "passed"),
    ("teacher_rationale", "passed"),
    ("teacher_reply", "passed"),
    ("action_only_no_thought", "passed"),
    ("action_only_ungrounded_student", "label_instances_not_public"),
)


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("Nonfinite JSON value")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_evidence(row):
    evidence = row["evidence"]
    optional = DECISION_EVIDENCE_FIELDS if row["kind"] == "decision" else set()
    require(isinstance(evidence, dict) and EVIDENCE_FIELDS <= evidence.keys()
            and evidence.keys() <= EVIDENCE_FIELDS | optional, "Invalid evidence schema")
    event, step = evidence["local_event"], evidence["step"]
    source, check = evidence["label_source"], evidence["label_check"]
    action_only = isinstance(source, str) and source.startswith("action_only_")
    valid_label = ((source, check) in LABEL_EVIDENCE_PAIRS or
        (source == "action_only_invalid_label" and isinstance(check, str)
         and check != "passed" and re.fullmatch(r"[a-z][a-z0-9_]*", check) is not None))
    require(isinstance(evidence["episode_ref"], str) and evidence["episode_ref"]
        and (step is None if row["kind"] == "trajectory" else type(step) is int and step >= 0)
        and evidence["executor"] in ("student", "student_after_guidance", "teacher")
        and (event is None or isinstance(event, str) and event in LOCAL_EVENTS)
        and (row["kind"] != "trajectory" or event is None)
        and type(evidence["episode_success"]) in (bool, type(None))
        and valid_label and (not action_only or row["kind"] == "decision"),
        "Invalid or unchecked label evidence")
    if action_only:
        messages, indices = row["messages"], row["supervised_assistant_indices"]
        require(isinstance(messages, list) and isinstance(indices, list) and all(
            type(i) is int and 0 <= i < len(messages) and isinstance(messages[i], dict)
            and isinstance(messages[i].get("content"), str)
            and re.fullmatch(r"Action: [^\n]+", messages[i]["content"]) is not None
            for i in indices), "Action-only label with non-action target")
    for field in evidence.keys() & optional:
        value = evidence[field]
        if field in EVIDENCE_HASH_FIELDS:
            valid = isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
        elif field == "local_events":
            valid = (isinstance(value, list) and all(isinstance(v, str) and v in LOCAL_EVENTS for v in value)
                     and event == (value[0] if value else None))
        elif field == "complete_success_eligible":
            valid = type(value) is bool
        elif field == "history_rewrites":
            valid = isinstance(value, list) and all(isinstance(v, dict) for v in value)
        else:  # next_legal_target_take: int (not bool), or null; no range restriction.
            valid = value is None or type(value) is int
        require(valid, f"Invalid decision evidence: {field}")


def _read_bundle(bundle, theta0, files):
    """Validate byte seals and summaries before any tokenizer/model allocation."""
    root, manifest = bundle["root"], bundle["manifest"]
    fields = {"schema", "arm", "created", "code_revision", "inputs", "counts", "supervised_turns",
              "example_ids_sha256", "splits_sha256", "theta0", "training_exposure_tasks", "excluded"}
    require(isinstance(manifest, dict) and fields <= manifest.keys(), "Incomplete interchange manifest")
    require(manifest["schema"] == SCHEMA and manifest["arm"] in ARMS, "Unknown interchange schema/arm")
    require(all(isinstance(manifest[k], str) and manifest[k] for k in ("created", "code_revision")),
            "Missing interchange provenance")
    identity = manifest["theta0"]
    require(isinstance(identity, dict) and set(identity) == {"adapter_path", "adapter_weights_sha256"}
            and isinstance(identity["adapter_path"], str) and _sha(identity["adapter_weights_sha256"]),
            "Invalid theta0 identity")
    require(file_hash(Path(theta0) / "adapter_model.safetensors") == identity["adapter_weights_sha256"],
            "theta0 adapter hash differs from dataset manifest")
    require(file_hash(root / "examples.jsonl") == bundle["examples_sha256"], "examples.jsonl sha256 mismatch")
    rows = [_json(line) for line in (root / "examples.jsonl").read_text().splitlines()]
    require(all(isinstance(row, dict) and set(row) == EXAMPLE_FIELDS for row in rows),
            "Example schema must exactly match the interchange whitelist")
    ids = [row["example_id"] for row in rows]
    require(all(_sha(i) for i in ids) and ids == sorted(set(ids)), "Example IDs must be sorted and unique")
    require(manifest["example_ids_sha256"] == digest(ids), "Example ID seal mismatch")
    required_counts = {"by_" + k: dict(Counter(row[k] for row in rows))
                       for k in ("source", "kind", "task_type")}
    for row in rows:
        _validate_evidence(row)
    expected_counts = {**required_counts,
        "by_local_event": dict(Counter(event for row in rows if row["kind"] == "decision"
            for event in row["evidence"].get("local_events", [row["evidence"]["local_event"]])
            if event is not None)),
        "examples": len(rows), "task_groups": len({row["group_id"] for row in rows}),
        "unique_instances": len({row["task_id"] for row in rows})}
    counts = manifest["counts"]
    require(isinstance(counts, dict) and required_counts.keys() <= counts.keys()
            and counts.keys() <= expected_counts.keys(), "Invalid manifest counts keys")
    for name, value in counts.items():
        expected = expected_counts[name]
        valid_type = (isinstance(value, dict) and all(type(n) is int for n in value.values())
                      if isinstance(expected, dict) else type(value) is int)
        require(valid_type and value == expected, f"Manifest counts mismatch: {name}")
    require(all(isinstance(row["supervised_assistant_indices"], list) for row in rows),
            "Supervised indices must be a list")
    require(type(manifest["supervised_turns"]) is int and manifest["supervised_turns"] == sum(
        len(row["supervised_assistant_indices"]) for row in rows), "Supervised turn count mismatch")
    require(manifest["training_exposure_tasks"] == sorted({row["group_id"] for row in rows}),
            "Training exposure groups mismatch")
    require(isinstance(manifest["excluded"], list) and all(isinstance(item, dict)
        and set(item) == {"reason", "count"} and isinstance(item["reason"], str)
        and type(item["count"]) is int and item["count"] >= 0 for item in manifest["excluded"]),
        "Invalid excluded-attempt accounting")
    inputs = manifest["inputs"]
    require(isinstance(inputs, dict) and inputs, "Missing hashed interchange inputs")
    paths = []
    for name in inputs:
        # The shared preflight already verified every input in both bundles.
        path = files.relocation.path(name, root=root)
        paths.append(path.resolve())
    return {"root": root, "manifest": manifest, "rows": rows,
            "inputs": [root / "manifest.json", root / "examples.sha256", root / "examples.jsonl", *paths]}
