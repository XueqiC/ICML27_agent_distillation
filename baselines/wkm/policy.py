"""Two served LoRAs, H0 actions and audited local-student command scoring."""
from pathlib import Path
import uuid

from baselines.wkm.data import PromptStudent
from baselines.wkm.format import agent_messages, world_prompt, visible_messages
from baselines.wkm.retrieval import KnowledgeBase, mix
from sed.provenance import utc_now
from sed.storage import BlobStore, JsonlLedger
from sed.react import parse_action
from teachers.openai_teacher import TeacherResult
from students.vllm_student import StudentCallError, ContextOverflow


class Scorer:
    def __init__(self, student, directory):
        self.student = student
        self.last_usage = []
        self.blobs = BlobStore(Path(directory) / "blobs")
        self.ledger = JsonlLedger(Path(directory) / "calls.jsonl")

    def post(self, url, **kwargs):
        """Audit every tokenization/scoring call, including failed attempts."""
        import httpx
        student = self.student
        if student.client is None:
            student.client = httpx.Client(timeout=student.spec["timeout_seconds"],
                                         transport=student.transport, trust_env=False)
        record = {"call_id": str(uuid.uuid4()), "timestamp": utc_now(),
                  "request_ref": self.blobs.put(kwargs["json"]), "endpoint_kind": url.rsplit("/", 1)[-1]}
        self.ledger.append({**record, "status": "started"})
        try:
            response = student.client.post(url, **kwargs)
            response.raise_for_status()
            result = response.json()
            if url.endswith("/completions"):
                self.last_usage.append(result.get("usage") or {})
            self.ledger.append({**record, "status": "completed", "raw_response_ref": self.blobs.put(result),
                "usage": result.get("usage"), "returned_model_id": result.get("model"),
                "response_created": result.get("created")})
            return response
        except Exception as exc:
            self.ledger.append({**record, "status": "failed", "error_type": type(exc).__name__, "usage": None})
            raise StudentCallError("WKM student scoring failed; inspect sanitized ledger") from None

    def __call__(self, messages, commands, thought):
        from tools.check_lora_serving import vllm_logprobs
        self.last_usage = []
        spec = self.student.spec
        endpoint = spec["endpoint"]
        tokenize = endpoint.removesuffix("/v1") + "/tokenize"
        prefix = self.post(tokenize, json={"model": spec["model"], "messages": messages,
            "add_generation_prompt": True, "add_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False}}).json()
        result = {}
        for command in commands:
            reply = (thought + "\n" if thought else "") + "Action: " + command
            suffix = self.post(tokenize, json={"model": spec["model"], "prompt": reply + "<turn|>",
                "add_special_tokens": False}).json()["tokens"]
            ids = prefix["tokens"] + suffix
            if len(suffix) > 1024 or len(ids) > prefix["max_model_len"]:
                raise ContextOverflow({"termination_reason": "context_overflow", "policy": "student",
                    "rendered_prompt_tokens": len(prefix["tokens"]), "max_model_len": prefix["max_model_len"],
                    "max_tokens": len(suffix)})
            values = vllm_logprobs(self, endpoint, spec["model"],
                {"input_ids": ids, "prompt_tokens": len(prefix["tokens"])})
            result[command] = sum(values)
        return result


class WKMStudent:
    def __init__(self, *, knowledge_model, knowledge_base, log_dir, scorer=None, kb=None,
                 student_factory=PromptStudent, **spec):
        if knowledge_model == spec["model"] or spec["enable_thinking"] or spec.get("max_tokens", 1024) != 1024:
            raise ValueError("WKM requires two distinct LoRAs, OFF and H0 token cap")
        spec.setdefault("max_tokens", 1024)
        self.agent = student_factory(log_dir=Path(log_dir) / "agent", **spec)
        self.world = student_factory(log_dir=Path(log_dir) / "world", **{**spec, "model": knowledge_model})
        self.kb = kb or KnowledgeBase(knowledge_base)
        self.scorer = scorer or Scorer(self.agent, Path(log_dir) / "scores")
        self.ledger = JsonlLedger(Path(log_dir) / "decisions.jsonl")
        self.spec = {**spec, "knowledge_model": knowledge_model, "knowledge_base": self.kb.identity, "gamma": .4}
        self.task_knowledge, self.states, self.task_id = None, [], None

    def _knowledge(self, prompt, history, task, visible, cap, stop=None):
        self.world.prompt_messages = prompt
        self.world.spec.update(max_tokens=cap, stop=stop)
        result = self.world.generate(history, task, visible)
        if not result.raw_text.strip() or result.metadata.get("finish_reason") not in {"stop", "length"}:
            raise StudentCallError("Invalid WKM knowledge generation")
        return result

    def generate(self, history, task, visible_context):
        messages = visible_messages(history, task, visible_context)
        knowledge_calls = []
        if not history:
            self.states, self.task_id = [], task["task_id"]
            result = self._knowledge(world_prompt(messages), history, task, visible_context, 512)
            self.task_knowledge = result.raw_text.strip()
            knowledge_calls.append(result.metadata)
        if self.task_id != task["task_id"] or len(history) != len(self.states):
            raise ValueError("WKM history/knowledge state mismatch")
        state = self._knowledge(world_prompt(messages, self.task_knowledge, self.states),
                                history, task, visible_context, 128, stop=["\n"])
        self.states.append(state.raw_text.strip())
        knowledge_calls.append(state.metadata)
        rendered = agent_messages(messages, self.task_knowledge)
        self.agent.prompt_messages = rendered
        sampled = self.agent.generate(history, task, visible_context)
        thought = sampled.raw_text.split("Action:", 1)[0].strip()
        commands = sorted(set(visible_context["admissible_commands"]))
        logps = self.scorer(rendered, commands, thought)
        previous = parse_action(history[-1]["reply"]) if history else "look"
        knowledge, retrieved = self.kb.probabilities(self.states[-1], commands, previous)
        action, agent, combined = mix(commands, logps, knowledge)
        record = {"task_id": task["task_id"], "step": len(history), "task_knowledge": self.task_knowledge,
            "state_knowledge": self.states[-1], "p_agent": agent, "p_know": knowledge,
            "p_mixed": combined, "retrieved": retrieved, "empty_retrieval_fallback": not bool(knowledge),
            "action": action, "agent_metadata": sampled.metadata, "knowledge_calls": knowledge_calls,
            "score_usage": getattr(self.scorer, "last_usage", [])}
        self.ledger.append(record)
        reply = (thought + "\n" if thought else "") + "Action: " + action
        costs = [sampled.metadata, *knowledge_calls, *getattr(self.scorer, "last_usage", [])]
        totals = {key: sum(r[key] for r in costs) if all(r.get(key) is not None for r in costs) else None
                  for key in ("prompt_tokens", "completion_tokens")}
        return TeacherResult(reply, action, {**sampled.metadata, **totals, "wkm": record,
            "parser_error": None, "action_outcome": "legal"})

    def close(self):
        self.agent.close()
        self.world.close()
