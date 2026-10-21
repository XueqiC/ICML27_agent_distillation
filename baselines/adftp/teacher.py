"""FTP prompting atop the capped Responses transport; no assistant prefill."""
import json

from sed.manifest import authorize_task
from sed.react import build_messages, action_outcome
from sed.storage import write_json
from sed.visibility import visible_observation, visible_task
from teachers.openai_teacher import Teacher
from teachers.stage import count

COT_PROMPT = ("Think step by step about how to solve the task in the initial observation. "
              "Write a visible solution plan without taking actions or using tools.")
FTP_PROMPT_VERSION = "adftp-h0-first-paragraph-v1"
MAX_EPISODE_STEPS = 40
COT_CALLS = 1
ATTEMPTS_PER_CANDIDATE = 3
CALLS_PER_ATTEMPT = COT_CALLS + MAX_EPISODE_STEPS
CALLS_PER_TASK = ATTEMPTS_PER_CANDIDATE * CALLS_PER_ATTEMPT


class FTPRejected(ValueError):
    pass


class FTPTeacher(Teacher):
    mode = "cot"
    prefix = None

    def __init__(self, **kwargs):
        super().__init__(prompt_template=FTP_PROMPT_VERSION, **kwargs)

    def prompt(self, history, task, visible_context):
        authorize_task(task, "generation")
        if self.mode == "cot":
            return [{"role": "system", "content": COT_PROMPT},
                    {"role": "user", "content": json.dumps({"task": visible_task(task),
                         "initial_observation": visible_observation(visible_context)}, ensure_ascii=False)}]
        messages = build_messages(history, task, visible_context)
        if not history:
            messages[0]["content"] += ("\nThe first reply must begin with the following exact text "
                "(continue it, then issue one Action line):\n" + self.prefix)
        return messages

    def parse_response(self, raw):
        if self.mode == "cot":
            return None, None
        return super().parse_response(raw)


class FTPPolicy:
    """One CoT call at the actual reset observation, then normal H0 execution."""
    def __init__(self, teacher, task, output):
        self.teacher, self.task, self.output = teacher, task, output
        self.prefix = None
        self.rejection = None
        self.spec = {"method": FTP_PROMPT_VERSION, "prefix_mode": "required_reply_beginning"}

    def generate(self, history, task, visible_context):
        if self.prefix is None:
            self.teacher.teacher.mode = "cot"
            cot = self.teacher.generate([], self.task, visible_context)
            # Literal extraction from the registered spec; no stripping/rephrasing.
            self.prefix = "Thought: " + cot.raw_text.split("\n\n")[0]
            write_json(self.output / "cot.json", {"raw_text": cot.raw_text, "metadata": cot.metadata,
                       "prefix": self.prefix, "parent_seed_ids": self.task["parent_seed_ids"]})
            if not self.prefix[len("Thought: "):].strip():
                count("rejections")
                self.rejection = "empty_first_paragraph"
                raise FTPRejected(self.rejection)
        self.teacher.teacher.mode = "agent"
        self.teacher.teacher.prefix = self.prefix
        reply = self.teacher.generate(history, self.task, visible_context)
        if not history and not reply.raw_text.startswith(self.prefix):
            count("rejections")
            self.rejection = "first_reply_prefix_mismatch"
            write_json(self.output / "prefix_rejection.json", {"reason": self.rejection,
                       "raw_text": reply.raw_text, "metadata": reply.metadata, "required_prefix": self.prefix})
            raise FTPRejected(self.rejection)
        if action_outcome(reply.raw_text, visible_context["admissible_commands"])[1] in {"parser_error", "inadmissible"}:
            count("rejections")
        usage = reply.metadata.get("usage") or {}
        return reply._replace(metadata={**reply.metadata,
            "mechanism_teacher_provider_id": self.teacher.teacher.provider_model_id,
            "mechanism_teacher_billed_usage": reply.metadata.get("billed_usage"),
            "prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens"),
            "thinking_tokens": usage.get("reasoning_tokens"), "thinking_chars": None})
