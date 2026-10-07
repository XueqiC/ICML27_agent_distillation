"""Greedy chat-completions client for the explicitly configured local student.

Uses HTTP only, with no OpenAI credentials/SDK, retry, truncation, or fallback.
The optional httpx transport allows a complete in-memory HTTP server in tests.
"""

from pathlib import Path
import re
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


class ContextOverflow(StudentCallError):
    """Known terminal policy failure; metadata contains no raw HTTP error."""
    def __init__(self, metadata):
        super().__init__("Student context_overflow")
        self.metadata = metadata


def context_options(config):
    """Enable exact prechecks from current or legacy serving provenance.

    Without a recorded limit, retain server-side overflow detection rather than
    guessing a tokenizer or context capacity for an external endpoint.
    """
    server = config.get("server_provenance", {}).get("config", {}).get("server", {})
    limit = server.get("max_model_len")
    command = server.get("command", [])
    if limit is None and "--max-model-len" in command:
        limit = int(command[command.index("--max-model-len") + 1])
    return {"max_model_len": limit} if limit is not None else {}


def context_rejection(response):
    """Recognize vLLM context 400s only; never retain the error body."""
    if response.status_code != 400:
        return None
    text = response.text.lower()
    if not (re.search(r"maximum context length|context length (?:is )?exceeded", text)
            and re.search(r"exceed|too (?:large|long)|however, (?:you requested|your request)", text)):
        return None
    def number(pattern):
        match = re.search(pattern, text)
        return int(match[1]) if match else None
    count = number(r"(\d+) (?:input tokens|tokens in (?:the )?(?:messages|prompt)|in the messages)")
    # Newer vLLM may report a lower bound after bounded tokenization, or
    # reject by character length without ever counting the full prompt.
    if "upper bound" in text and "characters" in text:
        count = None
    return {"rendered_prompt_tokens": count,
            "prompt_token_count_kind": "unknown" if count is None else
                "lower_bound" if re.search(r"at least \d+ input tokens", text) else "exact",
            "max_model_len": number(r"maximum context length is (\d+)")}


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
                 max_tokens=1024, timeout_seconds=180, transport=None, max_model_len=None):
        url = urlsplit(endpoint)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or
                url.password or url.query or url.fragment):
            raise ValueError("Endpoint must be an HTTP base URL without credentials/query")
        if (type(enable_thinking) is not bool or type(max_tokens) is not int or max_tokens < 1
                or timeout_seconds <= 0):
            raise ValueError("Explicit boolean thinking and positive max_tokens required")
        if max_model_len is not None and (type(max_model_len) is not int or max_model_len < 1):
            raise ValueError("max_model_len must be a positive integer")
        self.spec = {"endpoint": endpoint.rstrip("/"), "model": model,
                     "revision": revision, "temperature": 0, "max_tokens": max_tokens,
                     "chat_template_kwargs": {"enable_thinking": enable_thinking},
                     "timeout_seconds": timeout_seconds, "prompt_version": PROMPT_VERSION}
        self.log_dir, self.transport, self.client = Path(log_dir), transport, None
        self.max_model_len = max_model_len
        if max_model_len is not None:
            self.spec.update(max_model_len=max_model_len, prompt_token_counter="server_tokenize")

    def messages(self, history, task, visible_context):
        return build_messages(history, task, visible_context)

    def sampling(self):
        return {"temperature": 0}

    def request_sampling(self, history, task):
        return self.sampling()

    def select_choice(self, result, visible_context, blobs):
        return result["choices"][0], {}

    def generate(self, history, task, visible_context):
        import httpx
        messages = self.messages(history, task, visible_context)
        sampling = self.request_sampling(history, task)
        body = {"model": self.spec["model"], "messages": messages, **sampling,
                "max_tokens": self.spec["max_tokens"], "stream": False,
                "chat_template_kwargs": self.spec["chat_template_kwargs"],
                # Preserve raw thought boundaries if the server has no parser.
                "skip_special_tokens": False}
        blobs = BlobStore(self.log_dir / "blobs")
        ledger = JsonlLedger(self.log_dir / "calls.jsonl")
        identity = {"call_id": str(uuid.uuid4()), "timestamp": utc_now(),
                    "request_ref": blobs.put(body), "student_revision": self.spec["revision"]}
        if "seed" in sampling:
            identity["seed"] = sampling["seed"]
        ledger.append(dict(identity, status="started", usage=None))
        started = time.monotonic()
        prompt_tokens = None
        limit = self.max_model_len
        def reject(source, counts):
            metadata = dict(identity, status="rejected", error_type="context_overflow",
                termination_reason="context_overflow", step_index=len(history),
                rendered_prompt_tokens=counts["rendered_prompt_tokens"],
                prompt_token_count_kind=counts.get("prompt_token_count_kind", "exact"),
                max_model_len=counts["max_model_len"], max_tokens=self.spec["max_tokens"],
                context_overflow_source=source, policy="student", usage=None,
                # Rendered length is separate from inference token usage.
                prompt_tokens=0, completion_tokens=0, thinking_tokens=0, thinking_chars=0,
                latency_seconds=time.monotonic() - started)
            ledger.append(metadata)
            raise ContextOverflow(metadata)
        try:
            if self.client is None:
                self.client = httpx.Client(timeout=self.spec["timeout_seconds"],
                                           transport=self.transport, trust_env=False)
            if limit is not None:
                # vLLM's renderer avoids a second tokenizer dependency in CPU
                # environment workers. These are the unchanged generation inputs.
                token_body = {k: body[k] for k in ("model", "messages", "chat_template_kwargs")}
                token_body.update(add_generation_prompt=True, add_special_tokens=False)
                response = self.client.post(self.spec["endpoint"].removesuffix("/v1") + "/tokenize",
                                            json=token_body)
                overflow = context_rejection(response)
                if overflow is not None:
                    reject("server_tokenize", {**overflow,
                                               "max_model_len": overflow["max_model_len"] or limit})
                response.raise_for_status()
                tokens = response.json()
                prompt_tokens, limit = tokens["count"], tokens["max_model_len"]
                if (type(prompt_tokens) is not int or prompt_tokens < 0
                        or type(limit) is not int or limit < 1):
                    raise ValueError("Invalid server token count/limit")
                if prompt_tokens + self.spec["max_tokens"] > limit:
                    reject("prompt_precheck", {"rendered_prompt_tokens": prompt_tokens, "max_model_len": limit})
            response = self.client.post(self.spec["endpoint"] + "/chat/completions", json=body)
            overflow = context_rejection(response)
            if overflow is not None:
                reject("server_chat", {
                    "rendered_prompt_tokens": prompt_tokens if prompt_tokens is not None
                        else overflow["rendered_prompt_tokens"],
                    "prompt_token_count_kind": "exact" if prompt_tokens is not None
                        else overflow["prompt_token_count_kind"],
                    "max_model_len": overflow["max_model_len"] or limit})
            response.raise_for_status()
            result = response.json()
            choice, selection_metadata = self.select_choice(result, visible_context, blobs)
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
                            **sampling, max_tokens=self.spec["max_tokens"],
                            parser_error="ActionParseError" if parsed is None else None,
                            action_outcome=outcome, policy="student")
            metadata.update(selection_metadata)
            ledger.append(dict(metadata, status="completed"))
            return TeacherResult(reply, parsed, metadata)
        except ContextOverflow:
            raise
        except Exception as exc:
            # Never persist raw HTTP/SDK exception text or response error bodies.
            ledger.append(dict(identity, status="failed", error_type=type(exc).__name__,
                               usage=None, latency_seconds=time.monotonic() - started))
            raise StudentCallError("Student request failed; inspect sanitized call ledger") from None

    def close(self):
        if self.client is not None:
            self.client.close()
