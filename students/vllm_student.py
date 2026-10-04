"""Greedy chat-completions client for the explicitly configured local student.

Uses HTTP only, with no OpenAI credentials/SDK, retry, truncation, or fallback.
The optional httpx transport allows a complete in-memory HTTP server in tests.
"""

from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

from sed.provenance import utc_now
from sed.react import action_outcome, build_messages, PROMPT_VERSION
from sed.storage import BlobStore, JsonlLedger
from sed.thinking import split_thinking
from teachers.openai_teacher import TeacherResult


class StudentCallError(RuntimeError):
    pass


def separate_response(message):
    reply, embedded = split_thinking(message.get("content"))
    # vLLM 0.27.1 emits reasoning; older OpenAI-compatible servers may use
    # reasoning_content. Neither is included in the visible policy reply.
    field = "reasoning" if message.get("reasoning") is not None else "reasoning_content"
    structured = message.get(field)
    if structured is not None and type(structured) is not str:
        raise TypeError("Student reasoning must be text")
    blocks = [s for s in (structured, embedded) if s]
    thinking = "\n".join(dict.fromkeys(blocks))
    shape = field if structured is not None else "raw_channels" if embedded else "content_only"
    return reply, thinking, shape


class Student:
    def __init__(self, *, endpoint, model, revision, enable_thinking, log_dir,
                 max_tokens=2048, timeout_seconds=180, transport=None):
        url = urlsplit(endpoint)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or
                url.password or url.query or url.fragment):
            raise ValueError("Endpoint must be an HTTP base URL without credentials/query")
        if (type(enable_thinking) is not bool or type(max_tokens) is not int or max_tokens < 1
                or timeout_seconds <= 0):
            raise ValueError("Explicit boolean thinking and positive max_tokens required")
        self.spec = {"endpoint": endpoint.rstrip("/"), "model": model,
                     "revision": revision, "temperature": 0, "max_tokens": max_tokens,
                     "chat_template_kwargs": {"enable_thinking": enable_thinking},
                     "timeout_seconds": timeout_seconds, "prompt_version": PROMPT_VERSION}
        self.log_dir, self.transport, self.client = Path(log_dir), transport, None

    def generate(self, history, task, visible_context):
        import httpx
        messages = build_messages(history, task, visible_context)
        body = {"model": self.spec["model"], "messages": messages, "temperature": 0,
                "max_tokens": self.spec["max_tokens"], "stream": False,
                "chat_template_kwargs": self.spec["chat_template_kwargs"],
                # Preserve raw thought boundaries if the server has no parser.
                "skip_special_tokens": False}
        blobs = BlobStore(self.log_dir / "blobs")
        ledger = JsonlLedger(self.log_dir / "calls.jsonl")
        identity = {"call_id": str(uuid.uuid4()), "timestamp": utc_now(),
                    "request_ref": blobs.put(body), "student_revision": self.spec["revision"]}
        ledger.append(dict(identity, status="started", usage=None))
        started = time.monotonic()
        try:
            if self.client is None:
                self.client = httpx.Client(timeout=self.spec["timeout_seconds"],
                                           transport=self.transport, trust_env=False)
            response = self.client.post(self.spec["endpoint"] + "/chat/completions", json=body)
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            identity.update(returned_model_id=result.get("model"), response_id=result.get("id"),
                            response_created=result.get("created"),
                            raw_response_ref=blobs.put({k: choice["message"].get(k) for k in
                                ("content", "reasoning", "reasoning_content", "refusal")}))
            reply, thinking, shape = separate_response(choice["message"])
            if result.get("model") != self.spec["model"]:
                raise ValueError("Returned model differs from configured served identity")
            usage = result.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            # Do not estimate reasoning tokens from characters. Many vLLM builds
            # return only total completion tokens, which already include thinking.
            thinking_tokens = details.get("reasoning_tokens")
            if thinking_tokens is None and not thinking and choice.get("finish_reason") == "stop":
                thinking_tokens = 0
            parsed, outcome = action_outcome(reply, visible_context["admissible_commands"])
            metadata = dict(identity,
                            latency_seconds=time.monotonic() - started,
                            finish_reason=choice.get("finish_reason"),
                            prompt_tokens=usage.get("prompt_tokens"),
                            completion_tokens=usage.get("completion_tokens"),
                            thinking_tokens=thinking_tokens, thinking_chars=len(thinking),
                            thinking_return_shape=shape, thinking_ref=blobs.put(thinking),
                            chat_template_kwargs=self.spec["chat_template_kwargs"],
                            temperature=0, max_tokens=self.spec["max_tokens"],
                            parser_error="ActionParseError" if parsed is None else None,
                            action_outcome=outcome, policy="student")
            ledger.append(dict(metadata, status="completed"))
            return TeacherResult(reply, parsed, metadata)
        except Exception as exc:
            # Never persist raw HTTP/SDK exception text or response error bodies.
            ledger.append(dict(identity, status="failed", error_type=type(exc).__name__,
                               usage=None, latency_seconds=time.monotonic() - started))
            raise StudentCallError("Student request failed; inspect sanitized call ledger") from None

    def close(self):
        if self.client is not None:
            self.client.close()
