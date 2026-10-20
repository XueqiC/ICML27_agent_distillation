"""Training-only LEAP text requests through the shared Responses transport.

No SDK calls here. Teacher supplies caching, sanitized output/usage, and the
shared limiter. A durable reservation per attempt bounds retries across process
interruptions, parsing failures, unsupported sampling, and stage resumption.
"""
from copy import deepcopy
import json
from pathlib import Path

from baselines.leap.protocol import parse_correction, parse_extraction, require_train
from sed.manifest import frozen_write
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, locked, write_json
from teachers.openai_teacher import Teacher, TeacherCallError, rate_limit_options
from teachers.stage import count, require_completed


def read(path):
    return json.loads(Path(path).read_text())


class LeapTeacher(Teacher):
    """Explicit opt-in oracle subclass; never used by the rollout runner."""
    def __init__(self, *, messages, validator, **kwargs):
        self.messages, self.validator = deepcopy(messages), validator
        super().__init__(prompt_template=json.dumps(messages, ensure_ascii=False), **kwargs)

    def prompt(self, history, task, visible_context):
        require_train(task)
        return deepcopy(self.messages)

    def parse_response(self, raw):
        try:
            self.validator(raw)
        except (ValueError, TypeError, KeyError, RecursionError):
            return None, "LeapFormatError"
        return None, None


def checked_outcome(path):
    path = Path(path)
    value = read(path / "outcome.json")
    hashes = artifact_hashes(path)
    hashes.pop("outcome.json")
    if hashes != value["artifacts"]:
        raise ValueError("LEAP teacher evidence changed")
    if value["status"] != "accepted" and any(
            not (attempt / "result.json").exists() for attempt in path.glob("attempt-*")):
        raise TeacherCallError("LEAP outcome contains transport-failed/unknown attempts, not format rejections")
    return value


class TeacherBudget:
    def __init__(self, root, *, tasks, config, kind, iteration, run_identity,
                 max_output_tokens, sampling_root=None, teacher_factory=LeapTeacher):
        if kind not in {"extract", "privileged", "zero"} or iteration not in range(4):
            raise ValueError("Invalid LEAP teacher stage")
        if (kind == "extract") != (iteration == 0):
            raise ValueError("Extraction is once at iteration 0; corrections are 1..3")
        if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
            raise ValueError("Teacher cap requires distinct seed tasks")
        for task in tasks:
            require_train(task)
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("Positive output cap required")
        self.root, self.config, self.kind = Path(root), config, kind
        self.tasks = {t["task_id"]: deepcopy(t) for t in tasks}
        self.factory, self.max_output_tokens = teacher_factory, max_output_tokens
        self.sampling_root = Path(sampling_root or self.root.parent)
        self.binding = {"kind": kind, "iteration": iteration, "run_identity": run_identity,
            "tasks_sha256": digest(tasks), "logical_call_cap": len(tasks),
            "api_attempt_cap": 3 * len(tasks), "attempts_per_task": 3,
            "max_output_tokens": max_output_tokens, "teacher": config["models"]["teacher"],
            "sampling_policy": "try official .3/1 on actual request; explicit unsupported parameter => configured default"}
        frozen_write(self.root / "identity.json", self.binding)

    def sampling(self):
        path = self.sampling_root / "sampling.json"
        decision = read(path) if path.exists() else {"mode": "official_unconfirmed"}
        default = dict(self.config["models"]["teacher"]["sampling"])
        # Output length is a separately disclosed cap, not a sampling fallback.
        default["max_output_tokens"] = self.max_output_tokens
        if decision["mode"] != "configured_default":
            default.update(temperature=.3, top_p=1.)
        return default, decision["mode"]

    def request(self, task, messages, *, length=None, privileged_state=None):
        require_train(task)
        if task != self.tasks.get(task["task_id"]):
            raise ValueError("Teacher request is outside frozen seed scope/parents")
        if self.kind == "zero" and privileged_state is not None:
            raise ValueError("0% privileged request cannot contain state")
        validator = parse_extraction if self.kind == "extract" else lambda raw: parse_correction(
            raw, length, privileged_state=privileged_state)
        directory = self.root / "requests" / digest(task["task_id"])
        binding = {"budget_sha256": digest(self.binding), "task_id": task["task_id"],
            "parent_seed_ids": task["parent_seed_ids"], "messages_sha256": digest(messages),
            "length": length, "privileged_state_sha256": digest(privileged_state)}
        with locked(directory / ".request.lock"):
            frozen_write(directory / "binding.json", binding)
            if (directory / "outcome.json").exists():
                return checked_outcome(directory)
            spec = self.config["models"]["teacher"]
            accepted, result = None, None
            transport_failures, rejections = 0, 0
            for index in range(3):
                attempt = directory / f"attempt-{index}"
                if attempt.exists():
                    # Recover a completed response without another logical/API call.
                    if (attempt / "result.json").exists():
                        previous = read(attempt / "result.json")
                        if previous["accepted"]:
                            result, accepted = previous["result"], previous["labels"]
                            break
                        rejections += 1
                        count("rejections")
                    else:
                        transport_failures += 1
                    continue  # A durable start with no result consumes its slot.
                attempt.mkdir()
                sampling, mode = self.sampling()
                write_json(attempt / "start.json", {"sampling": sampling, "mode": mode,
                    "task_id": task["task_id"], "attempt": index + 1}, exclusive=True)
                teacher = self.factory(messages=messages, validator=validator,
                    cache_dir=self.sampling_root / "cache", usage_ledger=attempt / "usage.jsonl",
                    run_id=digest(binding), provider_model_id=spec["provider_model_id"], sampling=sampling,
                    max_attempts=1, timeout_seconds=spec["timeout_seconds"], purpose="generation",
                    secrets_file=self.config["paths"]["secrets_file"],
                    **rate_limit_options(spec, self.sampling_root / "rate_limit"))
                try:
                    response = teacher.generate([], task, {"observation": "", "admissible_commands": []})
                    require_completed(response)
                except TeacherCallError:
                    ledger = attempt / "usage.jsonl"
                    outcomes = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
                    unsupported = next((r for r in outcomes if r.get("unsupported_sampling_parameter")), None)
                    if unsupported:
                        write_json(self.sampling_root / "sampling.json", {"mode": "configured_default",
                            "reason": "Responses rejected official sampling", "evidence": str(ledger.resolve()),
                            "parameter": unsupported["unsupported_sampling_parameter"]})
                    write_json(attempt / "failure.json", {"error_type": "TeacherCallError",
                        "usage": "unknown; inspect shared usage ledger", "unsupported_sampling": bool(unsupported)})
                    # Even an unsupported HTTP sampling parameter is a transport
                    # failure. The recorded fallback applies on explicit resume;
                    # it must not hide a failed request in a rejection set.
                    raise TeacherCallError(
                        f"LEAP teacher transport failed; stage NOT complete. Evidence: {attempt}. "
                        "Resume after fixing transport; reserved attempts remain spent."
                    ) from None
                result = {"raw_text": response.raw_text, "metadata": response.metadata}
                try:
                    accepted = validator(response.raw_text)
                    error = None
                except (ValueError, TypeError, KeyError, RecursionError) as exc:
                    accepted, error = None, type(exc).__name__
                    rejections += 1
                    count("rejections")
                write_json(attempt / "result.json", {"accepted": accepted is not None,
                    "labels": accepted, "error_type": error, "result": result})
                if mode != "configured_default":
                    write_json(self.sampling_root / "sampling.json", {"mode": "official_accepted",
                        "evidence": str((attempt / "result.json").resolve()),
                        "returned_model_id": response.metadata.get("returned_model_id"),
                        "timestamp": response.metadata.get("timestamp")})
                if accepted is not None:
                    break
            if accepted is None and transport_failures:
                raise TeacherCallError(
                    f"LEAP teacher attempt budget exhausted with {transport_failures} transport-failed/unknown "
                    f"slots; no rejection outcome published. Retain {directory} and rerun with new provenance.")
            outcome = {"status": "accepted" if accepted is not None else "exhausted",
                "task_id": task["task_id"], "parent_seed_ids": task["parent_seed_ids"],
                "labels": accepted, "result": result,
                "attempts_reserved": len(list(directory.glob("attempt-*"))),
                "transport_failures": transport_failures, "rejections": rejections,
                "artifacts": artifact_hashes(directory), "directory": str(directory.resolve())}
            write_json(directory / "outcome.json", outcome, exclusive=True)
            return outcome
