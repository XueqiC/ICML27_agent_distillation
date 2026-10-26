"""Strict, versioned teacher schemas. Semantic patch errors stay proposal feedback."""

from dataclasses import dataclass
import json

from repair.patch import Patch, PatchBlock
from repair.records import JsonRecord

SCHEMA_VERSION = "repair-v01.1"


class SchemaError(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError("duplicate_key")
        result[key] = value
    return result


def decode(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_object)
    except (ValueError, TypeError):
        raise SchemaError("invalid_json") from None
    return value


def keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise SchemaError("unexpected_fields")


def text(value):
    if type(value) is not str or not value.strip():
        raise SchemaError("expected_nonempty_text")
    return value


def items(value):
    if type(value) is not list:
        raise SchemaError("expected_list")
    return value


def step_ids(value, n=None):
    values = items(value)
    if any(type(i) is not int or i < -1 or (n is not None and i >= n) for i in values):
        raise SchemaError("invalid_evidence_steps")
    if len(set(values)) != len(values):
        raise SchemaError("duplicate_evidence_steps")
    return values


@dataclass(frozen=True)
class DiagnosisRecord(JsonRecord):
    id_field = "diagnosis_id"
    original_trajectory_id: str
    expressed_plan: list
    preservable_choices: list
    unmet_requirements: list
    candidate_joint_changes: list
    unknowns: list

    def prompt_view(self):
        return {key: getattr(self, key) for key in (
            "expressed_plan", "preservable_choices", "unmet_requirements",
            "candidate_joint_changes", "unknowns")}


def parse_diagnosis(raw, original_trajectory_id, action_count):
    value = decode(raw)
    fields = ("expressed_plan", "preservable_choices", "unmet_requirements",
              "candidate_joint_changes", "unknowns")
    keys(value, fields)
    for field, description, status, literal in (
        ("expressed_plan", "subgoal", "interpretation_status", "hypothesis"),
        ("preservable_choices", "choice", "status", "candidate_until_execution"),
        ("unmet_requirements", "requirement", "certainty", "supported_or_unknown"),
    ):
        for row in items(value[field]):
            keys(row, (description, "evidence_steps", status))
            text(row[description])
            step_ids(row["evidence_steps"], action_count)
            allowed = {"supported", "unknown", literal} if status == "certainty" else {literal}
            if type(row[status]) is not str or row[status] not in allowed:
                raise SchemaError("invalid_diagnosis_status")
    for row in items(value["candidate_joint_changes"]):
        keys(row, ("block_ids", "explanation"))
        text(row["explanation"])
        ids = [text(i) for i in items(row["block_ids"])]
        if not ids or len(set(ids)) != len(ids):
            raise SchemaError("invalid_block_ids")
    for unknown in items(value["unknowns"]):
        text(unknown)
    return DiagnosisRecord(original_trajectory_id, **value)


def parse_candidates(raw, original_trajectory_id):
    value = decode(raw)
    keys(value, ("candidates",))
    candidates = items(value["candidates"])
    if not 1 <= len(candidates) <= 3:
        raise SchemaError("candidate_count")
    result = []
    for candidate in candidates:
        keys(candidate, ("original_trajectory_id", "blocks"))
        if candidate["original_trajectory_id"] != original_trajectory_id:
            raise SchemaError("original_trajectory_mismatch")
        blocks = items(candidate["blocks"])
        if not 1 <= len(blocks) <= 4:
            raise SchemaError("extended_repair" if len(blocks) > 4 else "empty_patch")
        parsed = []
        for block in blocks:
            if (type(block) is not dict or type(block.get("op")) is not str or
                    block["op"] not in {"insert_before", "replace", "delete"}):
                raise SchemaError("invalid_operation")
            indices = ("index",) if block["op"] == "insert_before" else ("start", "end_exclusive")
            keys(block, ("block_id", "op", "actions", *indices))
            text(block["block_id"])
            if any(type(block[i]) is not int for i in indices):
                raise SchemaError("invalid_index_type")
            actions = [text(a) for a in items(block["actions"])]
            if any("\n" in a or "\r" in a or a.startswith(("Action:", "Thought:")) for a in actions):
                raise SchemaError("expected_environment_command")
            if (block["op"] == "delete") != (not actions):
                raise SchemaError("invalid_operation_actions")
            parsed.append(PatchBlock(**block))
        result.append(Patch(original_trajectory_id, parsed))
    # Bounds/overlap are checked by compile_patch, not guessed or format-retried.
    return result


def token_count(value, tokenizer=None):
    return len(tokenizer.encode(value, add_special_tokens=False)) if tokenizer else len(value.split())


@dataclass(frozen=True)
class ReasoningLabel:
    reasoning: str
    evidence_steps: list[int]
    unsupported: bool


def parse_label(raw, tokenizer=None):
    value = decode(raw)
    keys(value, ("reasoning", "evidence_steps", "unsupported"))
    if type(value["unsupported"]) is not bool or type(value["reasoning"]) is not str:
        raise SchemaError("invalid_reasoning_type")
    if not value["unsupported"]:
        text(value["reasoning"])
    if token_count(value["reasoning"], tokenizer) > 64:
        raise SchemaError("reasoning_token_cap")
    from sed.thinking import split_thinking
    if (split_thinking(value["reasoning"])[1] or any(s in value["reasoning"] for s in
            ("Action:", "Thought:", "<|", "<turn|>", "<eos>", "\n", "\r"))):
        raise SchemaError("reasoning_control_text")
    step_ids(value["evidence_steps"])
    return ReasoningLabel(**value)
