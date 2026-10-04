"""Sampled text/actions through Responses; no logprobs or reasoning payloads.

API shape checked against the installed OpenAI 2.8.1 SDK and official guide:
https://developers.openai.com/api/docs/guides/migrate-to-responses
Only output_text and explicitly allowlisted accounting fields are retained.
"""

from copy import deepcopy
from datetime import timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import time
from typing import NamedTuple
import uuid

from sed.react import ActionParseError, parse_action, build_messages, PROMPT_TEMPLATE, PROMPT_VERSION
from sed.manifest import authorize_task
from sed.provenance import utc_now
from sed.storage import BlobStore, JsonlLedger, digest, locked, write_json
from sed.visibility import visible_history, visible_observation, visible_task
from teachers.rate_limit import RateLimiter, reserved_tokens

class TeacherResult(NamedTuple):
    raw_text: str
    parsed_action: str | None
    metadata: dict


class TeacherCallError(RuntimeError):
    pass


def read_api_key(secrets_file=None):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    path = Path(secrets_file or "~/hq/secrets/llm_apis.env").expanduser()
    if path.is_file():
        for line in path.read_text().splitlines():
            match = re.match(r"^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)$", line)
            if match:
                try:
                    parts = shlex.split(match.group(1), comments=True)
                except ValueError:
                    raise TeacherCallError("Malformed API credential assignment") from None
                if len(parts) == 1 and parts[0]:
                    return parts[0]
    raise TeacherCallError("OPENAI_API_KEY is not configured")


def _field(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _usage(response):
    usage = _field(response, "usage")
    if usage is None:
        return None
    details = _field(usage, "output_tokens_details", {})
    input_details = _field(usage, "input_tokens_details", {})
    return {"input_tokens": _field(usage, "input_tokens", 0),
            "output_tokens": _field(usage, "output_tokens", 0),
            "total_tokens": _field(usage, "total_tokens", 0),
            "reasoning_tokens": _field(details, "reasoning_tokens", 0) or 0,
            "cached_input_tokens": _field(input_details, "cached_tokens", 0) or 0}


def _seconds(value, now, *, reset=False):
    """Parse only rate-limit timing values; never retain raw HTTP headers."""
    try:
        value = str(value).strip()
        try:
            seconds = float(value)
            if reset and seconds > 1_000_000_000:
                seconds -= now
        except ValueError:
            if reset:
                parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", value)
                if not parts or "".join(n + u for n, u in parts) != value:
                    return 0.0
                seconds = sum(float(n) * {"ms": .001, "s": 1, "m": 60,
                                         "h": 3600, "d": 86400}[u] for n, u in parts)
            else:
                date = parsedate_to_datetime(value)
                seconds = date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp() - now
        return max(0.0, seconds) if math.isfinite(seconds) else 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0


def _provider_wait(exc, now):
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", {})
    headers = {str(k).lower(): v for k, v in (headers or {}).items()}
    waits = [_seconds(headers.get("retry-after", ""), now)]
    waits.extend(_seconds(v, now, reset=True) for k, v in headers.items()
                 if k.startswith("x-ratelimit-reset-"))
    return max(waits)


def rate_limit_options(spec, directory):
    """One explicit run directory, shared by suite workers and collection attempts."""
    settings = spec.get("rate_limit", {})
    if set(settings) - {"requests_per_minute", "tokens_per_minute"}:
        raise ValueError("Unsupported teacher rate-limit settings")
    return {"rate_limit_dir": directory, **settings}


class Teacher:
    def __init__(self, *, cache_dir, usage_ledger, run_id,
                 provider_model_id="gpt-6-luna", prompt_template=PROMPT_TEMPLATE,
                 sampling=None, max_attempts=10, timeout_seconds=60,
                 secrets_file=None, client=None, sleep=time.sleep,
                 purpose="generation", clock=time.time, jitter=random.uniform,
                 rate_limit_dir=None, requests_per_minute=300, tokens_per_minute=400_000):
        if provider_model_id != "gpt-6-luna":
            raise ValueError("Model substitution requires a new explicit provenance policy")
        self.provider_model_id = provider_model_id
        self.prompt_template = prompt_template
        self.sampling = dict(sampling or {"max_output_tokens": 512})
        if set(self.sampling) - {"max_output_tokens", "temperature", "top_p"}:
            raise ValueError("Unsupported sampling fields; privileged outputs are forbidden")
        self.sampling.setdefault("max_output_tokens", 512)
        if type(self.sampling["max_output_tokens"]) is not int or self.sampling["max_output_tokens"] <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        self.max_attempts, self.timeout_seconds = max_attempts, timeout_seconds
        self.cache_dir = Path(cache_dir)
        self.ledger = JsonlLedger(usage_ledger)
        self.call_blobs = BlobStore(Path(usage_ledger).parent / "teacher_blobs")
        self.run_id, self.secrets_file, self.client = run_id, secrets_file, client
        self.sleep, self.purpose = sleep, purpose
        self.clock, self.jitter = clock, jitter
        self.limiter = RateLimiter(rate_limit_dir or Path(usage_ledger).parent / "rate_limit",
                                  provider_model_id, requests_per_minute=requests_per_minute,
                                  tokens_per_minute=tokens_per_minute, clock=clock, sleep=sleep)
        self._secret = os.environ.get("OPENAI_API_KEY", "")

    def _redact(self, value):
        if isinstance(value, str):
            if self._secret:
                value = value.replace(self._secret, "[REDACTED]")
            return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)
        if isinstance(value, dict):
            return {k: self._redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        return value

    def _client(self):
        if self.client is None:
            from openai import OpenAI
            self._secret = read_api_key(self.secrets_file)
            # SDK retries are disabled so the ledger accounts for every attempt.
            self.client = OpenAI(api_key=self._secret, max_retries=0,
                                 timeout=self.timeout_seconds)
        return self.client

    def prompt(self, history, task, visible_context):
        return build_messages(history, task, visible_context, prompt_template=self.prompt_template)

    def cache_key(self, history, task, visible_context):
        return digest({"schema_version": 2, "prompt_version": PROMPT_VERSION,
                       "provider_model_id": self.provider_model_id,
                       "prompt_template_sha256": digest(self.prompt_template),
                       "task_id": task["task_id"], "task_sha256": digest(visible_task(task)),
                       "history_sha256": digest(visible_history(history)),
                       "visible_context_sha256": digest(visible_observation(visible_context)),
                       "sampling": self.sampling,
                       "prompt_sha256": digest(self.prompt(history, task, visible_context))})

    def generate(self, history, task, visible_context):
        authorize_task(task, self.purpose)
        # Teacher use on sealed evaluation is deliberately unavailable here:
        # evaluate a frozen student/scripted policy; generation stays train/dev.
        if task.get("sealed") or task["split"] == "test":
            raise ValueError("Teacher generation is forbidden on sealed tasks")
        key = self.cache_key(history, task, visible_context)
        path = self.cache_dir / (key + ".json")
        with locked(str(path) + ".lock"):
            if path.exists():
                cached = json.loads(path.read_text())
                if cached["cache_key"] != key or digest(cached["result"]) != cached["result_sha256"]:
                    raise TeacherCallError("Teacher cache integrity check failed")
                result = cached["result"]
                meta = dict(result["metadata"], cache_hit=True, cache_accessed_at=utc_now(),
                            billed_usage={"input_tokens": 0, "output_tokens": 0,
                                          "reasoning_tokens": 0, "total_tokens": 0})
                self.ledger.append({"run_id": self.run_id, "call_id": str(uuid.uuid4()),
                                    "timestamp": utc_now(), "status": "cache_hit",
                                    "cache_key": key, "provider_model_id": self.provider_model_id,
                                    "task_id": task["task_id"], "usage": meta["billed_usage"],
                                    "error_type": None, "wait_seconds": 0,
                                    "origin_request_id": meta["request_id"]})
                return TeacherResult(result["raw_text"], result["parsed_action"], meta)
            return self._generate_uncached(history, task, visible_context, key, path)

    def _generate_uncached(self, history, task, visible_context, key, path):
        prompt = self.prompt(history, task, visible_context)
        for attempt in range(1, self.max_attempts + 1):
            call_id, started, timestamp = str(uuid.uuid4()), time.monotonic(), utc_now()
            identity = {"run_id": self.run_id, "call_id": call_id, "cache_key": key,
                        "provider_model_id": self.provider_model_id, "task_id": task["task_id"],
                        "split": task["split"], "parent_seed_ids": task.get("parent_seed_ids", []),
                        "attempt": attempt, "timestamp": timestamp,
                        "prompt_template_sha256": digest(self.prompt_template),
                        "sampling": self.sampling, "wait_seconds": 0, "error_type": None}
            preparation_error = None
            try:
                client = self._client()
                identity.update(self.limiter.acquire(reserved_tokens(
                    prompt, self.sampling["max_output_tokens"])))
            except Exception as exc:
                preparation_error = type(exc).__name__
            if preparation_error is not None:
                self.ledger.append(dict(identity, status="failed", request_started=False,
                                        error_type=preparation_error, usage=None,
                                        latency_seconds=time.monotonic() - started))
                raise TeacherCallError("Teacher client/rate limiter could not be initialized; see usage ledger") from None
            prompt_ref = self.call_blobs.put(self._redact(prompt))
            identity["prompt_ref"] = prompt_ref
            # Persist the exact sanitized prompt before any billable request.
            # A durable start with no end marks interrupted/unknown usage.
            self.ledger.append(dict(identity, status="started", usage=None))
            response = None
            failure = None
            try:
                response = client.responses.create(model=self.provider_model_id, input=prompt,
                                                   store=False, **self.sampling)
            except Exception as exc:
                failure = {"error_type": type(exc).__name__,
                           "http_status": getattr(exc, "status_code", None),
                           "request_id": self._redact(getattr(exc, "request_id", None))}
                status = failure["http_status"]
                retryable = (status in {408, 409, 429} or isinstance(status, int) and 500 <= status < 600
                             or status is None and (isinstance(exc, (TimeoutError, ConnectionError))
                             or type(exc).__name__ in {"APIConnectionError", "APITimeoutError",
                                                     "RateLimitError", "ReadTimeout", "ConnectTimeout",
                                                     "WriteTimeout", "PoolTimeout"}))
                provider_wait = _provider_wait(exc, self.clock()) if retryable else 0.0
                if provider_wait:
                    self.limiter.defer_until(self.clock() + provider_wait)
            latency = time.monotonic() - started
            if failure is not None:
                retry = retryable and attempt < self.max_attempts
                wait = max(self.jitter(0, min(2 * 2 ** (attempt - 1), 60)), provider_wait) if retry else 0
                self.ledger.append({**identity, **failure, "status": "failed", "usage": None,
                                    "latency_seconds": latency, "retryable": retryable,
                                    "will_retry": retry, "wait_seconds": wait,
                                    "provider_wait_seconds": provider_wait})
                if retryable and attempt < self.max_attempts:
                    self.sleep(wait)
                    continue
                # Never expose SDK exception messages, HTTP headers or bodies.
                raise TeacherCallError("Teacher request failed; see sanitized usage ledger") from None
            raw = self._redact(_field(response, "output_text", ""))
            usage = _usage(response)
            parser_error = None
            try:
                parsed = parse_action(raw)
            except ActionParseError:
                parsed, parser_error = None, "ActionParseError"
            response_status = _field(response, "status", "completed")
            metadata = self._redact({
                "provider_model_id": self.provider_model_id,
                "returned_model_id": _field(response, "model"),
                "request_id": _field(response, "_request_id"), "response_id": _field(response, "id"),
                "usage": usage, "billed_usage": usage, "latency_seconds": latency,
                "timestamp": timestamp, "cache_hit": False, "cache_key": key,
                "call_id": call_id, "run_id": self.run_id, "attempt": attempt,
                "response_status": response_status, "parser_error": parser_error,
                "prompt_ref": prompt_ref,
            })
            result = {"raw_text": raw, "parsed_action": parsed, "metadata": metadata}
            result_ref = self.call_blobs.put(result)
            self.ledger.append(dict(identity, status="success" if parser_error is None else "rejected",
                                    result_ref=result_ref, usage=usage,
                                    returned_model_id=metadata["returned_model_id"],
                                    request_id=metadata["request_id"], response_id=metadata["response_id"],
                                    latency_seconds=latency, error_type=parser_error))
            # Raw rejected responses stay in the ledger/blob store, not reusable cache.
            if parser_error is None:
                write_json(path, {"cache_key": key, "result": result,
                                  "result_sha256": digest(result)}, exclusive=True)
            return TeacherResult(raw, parsed, deepcopy(metadata))
        raise AssertionError("Unreachable")
