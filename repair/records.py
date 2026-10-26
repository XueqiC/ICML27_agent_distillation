"""Content-identified evidence records. Only explicit visible views are prompts."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from typing import ClassVar

from sed.storage import canonical_bytes, digest
from sed.visibility import visible_history, visible_observation

from repair.align import Alignment
from repair.patch import Patch, PatchBlock


TRAJECTORY_STATUSES = {"SUCCESS", "FAILED", "STEP_LIMIT", "CONTEXT_LIMIT", "PARSE_FAILURE", "ENV_ERROR"}
REPAIR_STATUSES = {"VERIFIED", "NO_SUCCESS_WITHIN_BUDGET", "NONREPLAYABLE", "INVALID_PATCH",
                   "UNGROUNDED_TARGET", "LABEL_BUDGET_EXHAUSTED"}
MINIMALITY_STATUSES = {"BEST_OBSERVED", "BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS",
                       "ABLATION_BUDGET_EXHAUSTED"}


class JsonRecord:
    id_field: ClassVar[str]

    def _body(self):
        # A detached JSON-compatible object, including tuple -> list conversion.
        return json.loads(canonical_bytes(asdict(self)))

    @property
    def stable_id(self):
        return self.id_field.removesuffix("_id") + "_" + digest(self._body())

    def to_json(self):
        return {self.id_field: self.stable_id, **self._body()}

    @classmethod
    def _decode(cls, value):
        value = json.loads(value) if isinstance(value, str) else deepcopy(value)
        identity = value.pop(cls.id_field, None)
        return value, identity

    @classmethod
    def from_json(cls, value):
        body, identity = cls._decode(value)
        result = cls(**body)
        if identity is not None and identity != result.stable_id:
            raise ValueError("Record ID differs from content")
        return result


@dataclass(frozen=True)
class TaskRecord(JsonRecord):
    id_field: ClassVar[str] = "record_id"
    task_id: str
    fingerprint: str
    split: str
    initial_state_ref: str
    goal: str
    env_seed: int
    parent_task_id: str | None = None
    parent_repair_id: str | None = None
    generation_mode: str = "existing_training_task"

    @property
    def record_id(self):
        return self.stable_id


def _transition(value):
    if value is None:
        return None
    # Audit labels/digests remain siblings, as in sed.runner, never in visible.
    keys = ("action_raw", "action_parsed", "action_outcome", "reward", "terminated",
            "truncated", "audit_sha256", "environment_steps", "feedback_source", "error_type")
    result = {k: deepcopy(value[k]) for k in keys if k in value}
    result["visible"] = (visible_observation(value["visible"])
                         if value.get("visible") is not None else None)
    return result


@dataclass(frozen=True)
class TrajectoryRecord(JsonRecord):
    id_field: ClassVar[str] = "trajectory_id"
    task_id: str
    initial: dict | None
    steps: list[dict]
    success: bool | str
    termination_reason: str
    student_checkpoint: str | None = None
    token_costs: dict = field(default_factory=dict)
    environment_steps: int = 0
    environment_steps_confirmed: int = 0
    environment_steps_unknown: int = 0
    environment_version: str | None = None
    game_sha256: str | None = None
    fingerprint: str | None = None
    env_seed: int | None = None
    max_steps: int = 40
    cache_hit: bool = False
    cache_source_trajectory_id: str | None = None
    error_type: str | None = None

    def __post_init__(self):
        if self.termination_reason not in TRAJECTORY_STATUSES:
            raise ValueError("Unknown trajectory status")
        if type(self.success) is not bool and self.success != "unknown":
            raise ValueError("Success must be boolean or unknown")
        if (self.termination_reason == "SUCCESS") != (self.success is True):
            raise ValueError("Success and termination reason disagree")
        object.__setattr__(self, "initial", _transition(self.initial))
        object.__setattr__(self, "steps", [_transition(s) for s in self.steps])

    @property
    def trajectory_id(self):
        return self.stable_id

    @property
    def executed_actions(self):
        """Confirmed real actions, including inadmissible commands, before stopping."""
        return [s["action_parsed"] for s in self.steps if s.get("action_parsed") is not None
                and s["visible"] is not None and s.get("error_type") is None]

    def decision_context(self, index):
        """Reconstruct h_t from this execution only, excluding all future feedback."""
        if type(index) is not int or not 0 <= index <= len(self.steps) or self.initial is None:
            raise ValueError("Unavailable decision prefix")
        visible, history = self.initial["visible"], []
        for step in self.steps[:index]:
            if visible is None or step["visible"] is None:
                raise ValueError("Unknown observation in prefix")
            history.append({"visible": visible, "reply": step["action_raw"]})
            visible = step["visible"]
        return {"history": visible_history(history), "visible_context": visible_observation(visible)}


@dataclass(frozen=True)
class PatchRecord(JsonRecord):
    id_field: ClassVar[str] = "patch_record_id"
    original_trajectory_id: str
    blocks: tuple[PatchBlock, ...]
    candidate_source: str
    search_round: int

    def __post_init__(self):
        object.__setattr__(self, "blocks", tuple(
            PatchBlock(**b) if isinstance(b, dict) else b for b in self.blocks))

    @property
    def patch(self):
        return Patch(self.original_trajectory_id, self.blocks)

    @property
    def patch_id(self):
        return self.patch.patch_id

    def _body(self):
        return dict(super()._body(), patch_id=self.patch_id)

    @classmethod
    def from_json(cls, value):
        body, identity = cls._decode(value)
        patch_id = body.pop("patch_id", None)
        result = cls(**body)
        if (identity is not None and identity != result.stable_id) or (
                patch_id is not None and patch_id != result.patch_id):
            raise ValueError("Patch record ID differs from content")
        return result


@dataclass(frozen=True)
class RepairRecord(JsonRecord):
    id_field: ClassVar[str] = "repair_id"
    original_trajectory_id: str
    selected_patch: Patch
    executed_trajectory: TrajectoryRecord | None
    alignment: Alignment | None
    cost: tuple | None
    verification_results: list[dict]
    minimality_status: str
    repair_status: str
    joint_necessity: dict
    original_trajectory: TrajectoryRecord | None = None
    replays_used: int = 0
    ablation_replays_used: int = 0
    cache_hits: int = 0

    def __post_init__(self):
        if self.minimality_status not in MINIMALITY_STATUSES or self.repair_status not in REPAIR_STATUSES:
            raise ValueError("Unknown repair/minimality status")
        if self.cost is not None:
            object.__setattr__(self, "cost", tuple(self.cost))

    @property
    def repair_id(self):
        return self.stable_id

    def _body(self):
        value = super()._body()
        value["selected_patch"] = self.selected_patch.to_json()
        for key in ("executed_trajectory", "original_trajectory"):
            trajectory = getattr(self, key)
            value[key] = trajectory.to_json() if trajectory else None
        return value

    @classmethod
    def from_json(cls, value):
        body, identity = cls._decode(value)
        body["selected_patch"] = Patch.from_json(body["selected_patch"])
        for key in ("executed_trajectory", "original_trajectory"):
            if body[key] is not None:
                body[key] = TrajectoryRecord.from_json(body[key])
        if body["alignment"] is not None:
            body["alignment"] = Alignment.from_json(body["alignment"])
        result = cls(**body)
        if identity is not None and identity != result.stable_id:
            raise ValueError("Repair ID differs from content")
        return result
