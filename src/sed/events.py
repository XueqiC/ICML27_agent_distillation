"""Required section-12 event schema; all fields are mandatory."""

from dataclasses import asdict, dataclass
import math
import re

from sed.storage import BlobStore, JsonlLedger


@dataclass(frozen=True)
class Event:
    run_id: str
    episode_id: str
    event_id: str
    split: str
    task_family: str
    task_id: str
    parent_seed_ids: list[str]
    mechanism_id: str
    student_revision: str
    teacher_provider_id: str | None
    environment_revision: str
    policy_condition: str
    history_ref: str
    visible_context_ref: str
    action_raw: str | None
    action_parsed: str | None
    observation_ref: str | None
    verification_ref: str
    feedback_source: str
    new_environment_information: str
    teacher_tokens_in: int
    teacher_tokens_out: int
    environment_steps: int
    latency_seconds: float
    terminated: bool
    truncated: bool
    status: str
    error_type: str | None

    def validate(self):
        row = asdict(self)
        nullable = {"teacher_provider_id", "action_raw", "action_parsed",
                    "observation_ref", "error_type"}
        for key, value in row.items():
            if key in {"parent_seed_ids", "teacher_tokens_in", "teacher_tokens_out",
                       "environment_steps", "latency_seconds", "terminated", "truncated"}:
                continue
            if value is None and key in nullable:
                continue
            if not isinstance(value, str) or (not value and key != "action_raw"):
                raise ValueError(f"Invalid event field: {key}")
            if key.endswith("_ref") and not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                raise ValueError(f"Invalid reference: {key}")
        if type(self.parent_seed_ids) is not list or any(
                not isinstance(v, str) or not v for v in self.parent_seed_ids):
            raise ValueError("Invalid parent seed IDs")
        for key in ("teacher_tokens_in", "teacher_tokens_out", "environment_steps"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"Invalid count: {key}")
        if type(self.latency_seconds) not in (float, int) or not math.isfinite(
                self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("Invalid latency")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("Termination flags must be booleans")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("Invalid logical split")
        if self.status not in {"ok", "error", "rejected"}:
            raise ValueError("Invalid status")
        if self.status != "ok" and not self.error_type:
            raise ValueError("Non-success event needs an error type")
        if self.feedback_source not in {"real_environment", "teacher_inference",
                                        "human", "analytic_verifier", "wm_synthetic", "none"}:
            raise ValueError("Invalid feedback source")
        return row


class EventWriter:
    def __init__(self, directory):
        from pathlib import Path
        self.directory = Path(directory)
        self.blobs = BlobStore(self.directory / "blobs")
        self.ledger = JsonlLedger(self.directory / "events.jsonl")

    def append(self, event):
        row = event.validate()
        for key, value in row.items():
            if key.endswith("_ref") and value is not None:
                self.blobs.get(value)  # Never publish a dangling reference.
        self.ledger.append(row, unique_key="event_id")
