"""Case budgets and structured stages over the existing capped teacher transport.

No HTTP implementation lives here. Production callers use ``make_teachers``;
tests inject objects with the same generate(history, task, visible_context) API.
"""

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from pathlib import Path
import uuid
from importlib import metadata
import sys

from sed.manifest import authorize_task
from sed.storage import canonical_bytes, digest, file_hash
from teachers.openai_teacher import Teacher, TeacherCallError
from teachers.stage import require_completed
from repair.prompts import TEMPLATES, unpack_request
from repair.schemas import SchemaError

OUTPUT_CAPS = {"diagnosis": 2048, "proposal": 6144, "far": 6144, "rationale": 128}


class CaseStopped(RuntimeError):
    """Known budget/schema stop. Transport failures still propagate as stage failures."""


@dataclass(frozen=True)
class Limits:
    diagnosis_calls: int = 1
    proposal_calls: int = 2
    rationale_calls: int = 50
    format_retry_calls: int = 1
    total_calls: int = 54
    input_tokens: int = 250000
    output_tokens: int = 24000
    replays: int = 24
    ablations: int = 12
    handbacks: int = 3

    def __post_init__(self):
        ceilings = (1, 2, 50, 1, 54, 250000, 24000, 24, 12, 3)
        if any(type(v) is not int or not 0 <= v <= cap
               for v, cap in zip(asdict(self).values(), ceilings)):
            raise ValueError("Limits must be nonnegative and within the v0.1 ceilings")


class RepairTeacher(Teacher):
    def __init__(self, **kwargs):
        super().__init__(prompt_template=canonical_bytes(TEMPLATES).decode(), **kwargs)

    def prompt(self, history, task, visible_context):
        if history:
            raise ValueError("Repair requests use an explicit public prompt envelope")
        return unpack_request(visible_context).messages()

    def parse_response(self, raw):
        # The case parser owns the single shared format retry. Transport still
        # saves all raw responses, including later schema rejections.
        return "structured_repair_response", None


def make_teachers(config, output, *, run_id):
    """Construct lazily connected, 429-aware clients, without querying a teacher.

    One durable collector counter per stage/branch; CaseSession additionally
    enforces the combined per-case ceilings and reserves EACH transport attempt.
    """
    from teachers.capped import from_config
    return {kind: from_config(config, output=Path(output) / kind,
        cache_dir=Path(output) / kind / "cache", usage_ledger=Path(output) / kind / "usage.jsonl",
        run_id=run_id + "/" + kind, cap=54, teacher_class=RepairTeacher,
        sampling={**config["models"]["teacher"]["sampling"], "max_output_tokens": cap})
        for kind, cap in OUTPUT_CAPS.items()}


class _AttemptReservations:
    def __init__(self, session, prior, input_bound, output_cap, parent):
        self.session, self.prior = session, prior
        self.input_bound, self.output_cap, self.parent = input_bound, output_cap, parent
        self.attempts = 0

    def reserve(self, **kwargs):
        # The first request was already reserved before generate. A 429 retry
        # cannot bypass the case ceilings through the underlying client's loop.
        if self.attempts:
            self.session._reserve(self.input_bound, self.output_cap, "transport_retry")
            self.session.log("teacher_transport_retry", parent_attempt_id=self.parent,
                supervision={"teacher_calls": {"retry": 1}, "teacher_input_tokens": None,
                             "teacher_output_tokens": None},
                details={"previous_attempt_usage": "unknown; see provider usage ledger",
                         "request_id": kwargs["request_id"]})
        self.attempts += 1
        self.session.log("transport_reservation", settled=False,
                         parent_attempt_id=self.parent,
                         details={"request_id": kwargs["request_id"], "attempt": self.attempts,
                                  "input_bound": self.input_bound, "output_cap": self.output_cap})
        if self.prior is not None:
            self.prior.reserve(**kwargs)


class CaseSession:
    def __init__(self, *, task, teacher, ledger, config, limits=None, branch="near",
                 shared=None):
        authorize_task(task, "generation")
        if task.get("evaluation_only") or task.get("task_set") in {"D_probe", "D_unseen_quick"}:
            raise ValueError("Evaluation instances cannot drive repair generation")
        self.task, self.teacher, self.ledger, self.config = task, teacher, ledger, config
        self.max_steps = config["environment"]["max_episode_steps"]
        if type(self.max_steps) is not int or self.max_steps != 40:
            raise ValueError("The v0.1 pilot requires configured H=40")
        self.action_syntax = config["environment"]["action_format"]
        self.limits, self.branch = limits or Limits(), branch
        if any(row["event"] == "case_started" and row["task_id"] == task["task_id"] and
               row["details"].get("branch") == branch for row in ledger.rows()):
            raise ValueError("Case ledger already started; retain the existing session/budgets instead of resetting them")
        self.shared = shared if shared is not None else {"format_retries": 0, "stopped": None,
            "replays": 0, "ablations": 0, "handbacks": 0, "cache_hits": 0,
            "teacher_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "diagnosis_calls": 0, "rationale_calls": 0}
        self.calls = {k: 0 for k in ("diagnosis", "proposal", "rationale", "retry", "transport_retry")}
        self.input_used, self.output_used, self.stopped = 0, 0, None
        self.log("case_started", details={"limits": asdict(self.limits), "config": {
            "max_steps": self.max_steps, "action_syntax": self.action_syntax},
            "parent_seed_ids": list(task.get("parent_seed_ids", []))})
        root = Path(__file__).resolve().parents[1]
        dependencies = {}
        for package in ("openai", "alfworld", "textworld", "transformers"):
            try:
                dependencies[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                dependencies[package] = None
        paths = config.get("paths", {})
        public_config = {"teacher": {k: config.get("models", {}).get("teacher", {}).get(k) for k in
            ("requested_name", "provider_model_id", "sampling", "timeout_seconds", "access")},
            "student": {k: config.get("models", {}).get("student", {}).get(k) for k in
                        ("requested_repo", "revision")},
            "environment": {k: config["environment"].get(k) for k in
                ("name", "version", "textworld_version", "seed", "max_episode_steps", "action_format", "domain_randomization")}}
        self.log("case_provenance", details={"config_sha256": digest(config),
            "effective_config": public_config,
            "python": sys.version.split()[0], "dependencies": dependencies,
            "code_sha256": {str(p.relative_to(root)): file_hash(p) for p in
                sorted((root / "repair").glob("*.py"))},
            "frozen_inputs": {k: {"path": paths.get(k), "sha256": file_hash(paths[k])
                if paths.get(k) and Path(paths[k]).is_file() else None}
                for k in ("manifest", "split_freeze")},
            "authorized_task_sha256": digest(task), "split": task["split"]})

    def log(self, event, **kwargs):
        details = dict(kwargs.pop("details", {}) or {}, branch=self.branch)
        return self.ledger.append(event, task_id=self.task["task_id"], details=details, **kwargs)

    def stop(self, reason):
        self.stopped = reason
        self.log("case_stopped", reason=reason)
        raise CaseStopped(reason)

    def _reserve(self, input_bound, output_cap, kind):
        if self.stopped or self.shared["stopped"]:
            raise CaseStopped(self.stopped or self.shared["stopped"])
        if self.shared["teacher_calls"] >= self.limits.total_calls:
            self.stop("total_call_cap")
        if self.shared["input_tokens"] + input_bound > self.limits.input_tokens:
            self.stop("input_token_cap")
        if self.shared["output_tokens"] + output_cap > self.limits.output_tokens:
            self.stop("output_token_cap")
        if kind in {"diagnosis", "proposal", "rationale"} and self.calls[kind] >= getattr(self.limits, kind + "_calls"):
            self.stop(kind + "_call_cap")
        if kind == "rationale" and self.shared["rationale_calls"] >= self.limits.rationale_calls:
            self.stop("rationale_call_cap")
        if kind == "diagnosis" and self.shared["diagnosis_calls"] >= self.limits.diagnosis_calls:
            self.stop("diagnosis_call_cap")
        self.calls[kind] += 1
        self.input_used += input_bound
        self.output_used += output_cap
        self.shared["teacher_calls"] += 1
        self.shared["input_tokens"] += input_bound
        self.shared["output_tokens"] += output_cap
        self.shared["rationale_calls"] += int(kind == "rationale")
        self.shared["diagnosis_calls"] += int(kind == "diagnosis")

    def request(self, prompt, parser):
        """Validate, at most once retry across the entire original failure case."""
        for retry in (False, True):
            if retry:
                if self.shared["format_retries"] >= self.limits.format_retry_calls:
                    self.shared["stopped"] = "format_retry_exhausted"
                    self.stop("format_retry_exhausted")
                self.shared["format_retries"] += 1
                prompt = replace(prompt, format_retry=True)
            kind = "retry" if retry else "proposal" if prompt.kind == "far" else prompt.kind
            cap = OUTPUT_CAPS[prompt.kind]
            # UTF-8 byte bound + framing, deliberately more conservative than
            # the rate limiter's char/4 estimate. This is NOT reported usage.
            bound = len(canonical_bytes(prompt.messages())) + 32
            self._reserve(bound, cap, kind)
            attempt = str(uuid.uuid4())
            self.log("teacher_retry" if retry else "teacher_request", settled=False,
                attempt_id=attempt, details={"request": prompt.to_json(),
                "request_sha256": digest(prompt.messages()), "input_bound": bound, "output_cap": cap})
            client = self.teacher[prompt.kind] if isinstance(self.teacher, dict) else self.teacher
            underlying = getattr(client, "teacher", client)
            previous = getattr(underlying, "reservation_budget", None)
            hook = _AttemptReservations(self, previous, bound, cap, attempt)
            if isinstance(underlying, Teacher):
                if underlying.sampling["max_output_tokens"] != cap:
                    raise ValueError("Teacher stage output cap differs from repair protocol")
                underlying.reservation_budget = hook
            try:
                result = client.generate([], self.task, {"observation": canonical_bytes(prompt.to_json()).decode(),
                                                        "admissible_commands": []})
                require_completed(result)
            except BaseException as exc:
                self.shared["stopped"] = "teacher_stage_failed"
                self.log("teacher_failure", attempt_id=attempt, reason=type(exc).__name__,
                    supervision={"teacher_calls": {"retry" if kind == "transport_retry" else kind: 1},
                                 "teacher_input_tokens": None, "teacher_output_tokens": None})
                # No raw SDK exception text crosses this boundary.
                if isinstance(exc, (KeyboardInterrupt, SystemExit, CaseStopped)):
                    raise
                raise TeacherCallError("Repair teacher stage failed; inspect retained ledgers") from None
            finally:
                if isinstance(underlying, Teacher):
                    underlying.reservation_budget = previous
            usage = result.metadata["usage"]
            billed = result.metadata.get("billed_usage", usage)
            accounted = billed if result.metadata.get("cache_hit") else usage
            # Release only the completed attempt's reservation. Failed/unknown
            # transport reservations remain charged conservatively.
            self.input_used += accounted["input_tokens"] - bound
            self.output_used += accounted["output_tokens"] - cap
            self.shared["input_tokens"] += accounted["input_tokens"] - bound
            self.shared["output_tokens"] += accounted["output_tokens"] - cap
            self.log("teacher_result", attempt_id=attempt,
                supervision={"teacher_calls": {kind: int(not result.metadata.get("cache_hit"))}, "teacher_input_tokens": billed.get("input_tokens"),
                    "teacher_output_tokens": billed.get("output_tokens"),
                    "teacher_cached_tokens": usage.get("cached_input_tokens", 0),
                    "teacher_billed_cost": result.metadata.get("billed_cost")},
                details={"raw_output": result.raw_text, "teacher_metadata": result.metadata,
                         "template_sha256": prompt.template_sha256})
            if result.metadata.get("cache_hit"):
                self.log("teacher_cache_hit", details={"request_sha256": digest(prompt.messages())})
            if (self.shared["input_tokens"] > self.limits.input_tokens or
                    self.shared["output_tokens"] > self.limits.output_tokens):
                self.stop("reported_usage_exceeded_reservation")
            try:
                if result.truncated or usage["output_tokens"] > cap:
                    raise SchemaError("output_cap")
                return parser(result.raw_text)
            except SchemaError as exc:
                self.log("schema_rejection", reason=str(exc), details={"request_attempt": attempt})
        self.shared["stopped"] = "format_retry_exhausted"
        self.stop("format_retry_exhausted")

    def replay_allowed(self, *, reserve_confirmation=False):
        if self.shared["replays"] >= self.limits.replays - int(reserve_confirmation):
            self.log("replay_budget_exhausted", reason="replay_cap")
            return False
        return True

    def cost_record(self):
        return {"branch": self.branch, "calls": dict(self.calls),
                "input_tokens_or_unsettled_bound": self.input_used,
                "output_tokens_or_unsettled_cap": self.output_used,
                "shared": deepcopy(self.shared), "stop_reason": self.stopped}
