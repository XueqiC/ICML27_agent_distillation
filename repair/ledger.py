"""Append-only budget events; reservations are never charged as settled usage."""

from copy import deepcopy
import json
import math
from pathlib import Path
import uuid

from sed.provenance import utc_now
from sed.storage import JsonlLedger


BUDGETS = {
    "supervision": {"initial_teacher_demonstrations": 0,
                    "additional_verified_teacher_trajectories": 0,
                    "additional_teacher_labeled_decisions": 0,
                    "teacher_calls": {k: 0 for k in
                                      ("diagnosis", "proposal", "rationale", "task_generation", "retry")},
                    "teacher_input_tokens": 0, "teacher_output_tokens": 0,
                    "teacher_cached_tokens": 0, "teacher_billed_cost": 0},
    "interaction": {k: 0 for k in ("student_environment_steps", "candidate_replay_steps",
                                   "edit_ablation_steps", "task_validation_steps",
                                   "handback_diagnostic_steps", "evaluation_steps")},
    "optimization": {k: 0 for k in ("optimizer_updates", "processed_context_tokens",
                                    "supervised_target_tokens", "trainable_parameters")},
}


def _budget(values, schema):
    if not isinstance(values, dict) or set(values) - set(schema):
        raise ValueError("Unknown budget fields")
    result = deepcopy(schema)
    for key, value in values.items():
        if isinstance(schema[key], dict):
            result[key] = _budget(value, schema[key])
        elif value is None:
            result[key] = None  # Unreported usage is unknown, never a fabricated zero.
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Budget quantities must be finite and nonnegative")
        else:
            result[key] = value
    return result


class BudgetLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.ledger = JsonlLedger(path)

    def append(self, event, *, settled=True, attempt_id=None, parent_attempt_id=None,
               task_id=None, candidate_id=None, reason=None, supervision=None,
               interaction=None, optimization=None, details=None):
        if type(settled) is not bool:
            raise ValueError("settled must be boolean")
        attempt_id = attempt_id or str(uuid.uuid4())
        row = {"schema_version": 1, "event": event, "settled": settled,
               "attempt_id": attempt_id, "parent_attempt_id": parent_attempt_id,
               "entry_id": attempt_id + (":settled" if settled else ":" + str(uuid.uuid4())),
               "timestamp": utc_now(), "task_id": task_id, "candidate_id": candidate_id,
               "reason": reason, "details": details or {},
               "supervision": _budget(supervision or {}, BUDGETS["supervision"]),
               "interaction": _budget(interaction or {}, BUDGETS["interaction"]),
               "optimization": _budget(optimization or {}, BUDGETS["optimization"])}
        self.ledger.append(row, unique_key="entry_id")
        return row

    def rows(self):
        if not self.path.exists():
            return []
        content = self.path.read_bytes()
        if content and not content.endswith(b"\n"):
            raise ValueError("Damaged JSONL tail")
        return [json.loads(line) for line in content.splitlines()]

    def summary(self):
        return summarize(self.rows())


def summarize(rows):
    """Sum settled rows once; count unknown quantities separately from known sums."""
    result = deepcopy(BUDGETS)
    unknown, seen, count = {}, set(), 0

    def add(target, source, prefix):
        for key, value in source.items():
            name = prefix + "." + key
            if isinstance(value, dict):
                add(target[key], value, name)
            elif value is None:
                unknown[name] = unknown.get(name, 0) + 1
            else:
                target[key] += value

    for row in rows:
        if row.get("settled") is not True:
            continue
        if row["attempt_id"] in seen:
            raise ValueError("Attempt settled more than once")
        seen.add(row["attempt_id"])
        count += 1
        for family, schema in BUDGETS.items():
            add(result[family], _budget(row.get(family, {}), schema), family)
    return dict(result, settled_rows=count, unknown_quantities=unknown)
