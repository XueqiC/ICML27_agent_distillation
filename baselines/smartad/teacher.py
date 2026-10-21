"""Capped H0 sampling, with LEAP's explicit unsupported-parameter rule."""
import json
from pathlib import Path

from sed.manifest import frozen_write
from sed.storage import digest, locked
from teachers.capped import CappedTeacher
from teachers.openai_teacher import TeacherCallError, rate_limit_options

REQUESTED_SAMPLING = {"temperature": 1.0, "max_output_tokens": 1024}
DEFAULT_SAMPLING = {"max_output_tokens": 1024}
CANDIDATES = 10
CALLS_PER_CANDIDATE = 41
CALLS_PER_TASK = CANDIDATES * CALLS_PER_CANDIDATE


def read(path):
    return json.loads(Path(path).read_text())


def effective_sampling(sampling_root):
    path = Path(sampling_root) / "sampling_fallback.json"
    if not path.exists():
        return dict(REQUESTED_SAMPLING)
    decision = read(path)
    from teachers.chain_recovery import evidence_path
    ledger = evidence_path(sampling_root, decision["ledger"]) if (Path(sampling_root) / "identity.json").exists() else Path(decision["ledger"])
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    evidence = next((r for r in rows if r.get("call_id") == decision["call_id"]
                     and r.get("status") == "failed"), None)
    if (decision.get("mode") != "provider_default" or decision.get("parameter") != "temperature"
            or evidence is None or digest(evidence) != decision["failure_sha256"]
            or evidence.get("http_status") != 400 or evidence.get("unsupported_sampling_parameter") != "temperature"
            or evidence.get("sampling") != REQUESTED_SAMPLING):
        raise ValueError("SmartAD sampling fallback lacks matching rejection evidence")
    return dict(DEFAULT_SAMPLING)


class SmartADTeacher(CappedTeacher):
    """Only an evidenced HTTP400 temperature rejection permits recovery.

    The rejected request spends one of the 41 logical reservations. No probes,
    candidate retries, or budget refunds. Other failures retain the normal halt.
    """
    def __init__(self, *, sampling_root, **kwargs):
        self.sampling_root = Path(sampling_root)
        super().__init__(sampling=REQUESTED_SAMPLING, **kwargs)
        self._sampling_fallback()

    def _sampling_fallback(self):
        ledger = self.teacher.ledger.path
        rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        ends = [r for r in rows if r["status"] != "state_link"]
        last = ends[-1] if ends else {}
        explicit = (last.get("status") == "failed" and last.get("http_status") == 400
                    and last.get("unsupported_sampling_parameter") == "temperature"
                    and last.get("sampling") == REQUESTED_SAMPLING)
        if explicit:
            decision = self.sampling_root / "sampling_fallback.json"
            # Several in-flight shard requests may reject temperature together.
            # Bind the first durable rejection; every other receipt is retained
            # in its own ledger without racing to replace the shared decision.
            with locked(self.sampling_root / ".sampling.lock"):
                if not decision.exists():
                    frozen_write(decision, {"mode": "provider_default", "parameter": "temperature",
                        "reason": "explicit_unsupported_parameter", "ledger": str(ledger.resolve()),
                        "call_id": last["call_id"], "failure_sha256": digest(last)})
            if self._read_counter()["halted"]:
                from teachers.recover import recover
                recover(self.counter_path, reason="transport_fixed")
        self.teacher.sampling = effective_sampling(self.sampling_root)
        return explicit

    def generate(self, history, task, visible_context):
        self.teacher.sampling = effective_sampling(self.sampling_root)
        try:
            result = super().generate(history, task, visible_context)
        except TeacherCallError:
            if not self._sampling_fallback():
                raise
            result = super().generate(history, task, visible_context)
        usage = result.metadata.get("usage") or {}
        return result._replace(metadata={**result.metadata,
            "mechanism_teacher_provider_id": self.teacher.provider_model_id,
            "mechanism_teacher_billed_usage": result.metadata.get("billed_usage"),
            "prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens"),
            "thinking_tokens": usage.get("reasoning_tokens"), "thinking_chars": None})


def default_teacher(config, directory, task, identity, rate_limit_dir, sampling_root):
    spec = config["models"]["teacher"]
    return SmartADTeacher(counter_path=directory / "counter.json", cap=CALLS_PER_CANDIDATE,
        cache_dir=directory / "cache", usage_ledger=directory / "usage.jsonl",
        run_id=digest([identity, task["task_id"], str(directory.resolve())]),
        sampling_root=sampling_root, provider_model_id=spec["provider_model_id"],
        baseline_method="smartad", secrets_file=config["paths"]["secrets_file"],
        **rate_limit_options(spec, rate_limit_dir))


class SmartADPolicy:
    """Runner supplies only visible task fields to wrappers; retain audit ID."""
    def __init__(self, teacher, task):
        self.teacher, self.task = teacher, task
        self.spec = {"method": "smartad", "harness": "H0", "sampling_requested": REQUESTED_SAMPLING}

    def generate(self, history, task, visible_context):
        return self.teacher.generate(history, self.task, visible_context)
