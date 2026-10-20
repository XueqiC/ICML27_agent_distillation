"""Paper-derived dynamic role system; every invocation uses CappedTeacher."""
import json
import os
from pathlib import Path

from baselines.coa import RECIPE
from baselines.coa.protocol import SYSTEM, blocks, h0_reply, messages
from sed.storage import JsonlLedger, digest
from sed.visibility import visible_observation
from teachers.capped import CappedTeacher
from teachers.stage import count, require_completed
from teachers.openai_teacher import Teacher, TeacherResult, rate_limit_options

ROLE_PROMPT = """You are one role of an ALFWorld chain-of-agents teacher. Use only the
provided public observations, admissible commands and visible agent messages.
Return concise task work; no private reasoning or hidden state is requested.
The controller names your role and the allowed next roles. For think, return JSON
{\"text\": \"brief coordination decision\", \"next\": \"one allowed role\"}.
For plan/reflection return only the corresponding XML tag containing your work.
For double_check return <double_check>Score: N\nbrief evidence-based check</double_check>
with N from 1 to 4. A score is your assessment, never environment ground truth.
For alfworld return <alfworld>Action: one command</alfworld>, with no other text.
"""


class RoleTeacher(Teacher):
    def prompt(self, history, task, visible_context):
        # The controller builds this string from a visible-only schema. Neither
        # the task ID nor parent seed IDs enter the model prompt.
        return [{"role": "system", "content": ROLE_PROMPT},
                {"role": "user", "content": visible_observation(visible_context)["observation"]}]

    def parse_response(self, raw):
        return None, None  # Role-specific validation occurs in the controller.


def require_cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise ValueError("CoA teacher stages must not hold a GPU")
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        if os.environ.get(key, ""):
            raise ValueError("CoA teacher stages require a CPU allocation")
    for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE"):
        if os.environ.get(key, "") not in ("", "0"):
            raise ValueError("CoA teacher stages require a CPU allocation")


def budget(config, directory, identity, tasks, *, evaluation=False, client=None):
    require_cpu()
    directory = Path(directory)
    spec = config["models"]["teacher"]
    runs = RECIPE["teacher_eval_runs" if evaluation else "teacher_runs"]
    return CappedTeacher(counter_path=directory / "counter.json", cap=len(tasks) * runs * 40 * 8,
        cache_dir=directory / "cache", usage_ledger=directory / "usage.jsonl",
        run_id=digest([identity, "teacher-dev" if evaluation else "teacher-train"]),
        evaluation_only=evaluation, teacher_class=RoleTeacher, client=client,
        provider_model_id=spec["provider_model_id"], prompt_template=ROLE_PROMPT,
        sampling={"max_output_tokens": RECIPE["teacher_output_tokens"]},
        timeout_seconds=spec["timeout_seconds"], secrets_file=config["paths"]["secrets_file"],
        **rate_limit_options(spec, directory / "rate_limit"))


class MultiAgentTeacher:
    def __init__(self, capped, task, run_index, log_dir):
        self.capped, self.task, self.run_index = capped, task, run_index
        self.turns = []
        self.ledger = JsonlLedger(Path(log_dir) / "agents.jsonl")
        self.spec = {"method": "coa", "teacher": "gpt-6-luna", "roles": 5,
                     "max_invocations_per_step": 8, "run_index": run_index}

    def generate(self, history, task, visible_context):
        if len(history) != len(self.turns):
            raise ValueError("CoA teacher history changed")
        page = visible_observation(visible_context)
        active, outputs, records, calls, usages = set(), [], [], 0, []
        while True:
            choices = [r for r in ("plan", "reflection", "double_check") if r not in active] + ["alfworld"]
            for role in ("think", "selected"):
                role = selected if role == "selected" else role
                context = {"observation": json.dumps({"protocol": SYSTEM, "role": role,
                    "allowed_next_roles": choices, "system_run": self.run_index,
                    "history": messages(self.turns, page), "current_agent_outputs": outputs}, ensure_ascii=False),
                    "admissible_commands": page["admissible_commands"]}
                result = self.capped.generate([], self.task, context)
                require_completed(result)
                calls += 1
                usages.append(result.metadata.get("billed_usage"))
                record = {"step": len(history), "agent": role, "state": context["observation"],
                    "raw_output": result.raw_text, "metadata": result.metadata, "status": "rejected"}
                try:
                    if role == "think":
                        value = json.loads(result.raw_text)
                        selected = value["next"]
                        if selected not in choices or not isinstance(value["text"], str):
                            raise ValueError("Invalid role selection")
                        rendered = "<think>" + value["text"] + "</think>"
                    else:
                        rendered = result.raw_text.strip()
                    parts = blocks(rendered)
                    if len(parts) != 1 or parts[0][0] != role:
                        raise ValueError("Role output uses incorrect tags")
                    if role == "double_check":
                        import re
                        if not re.search(r"\bScore:\s*[1-4]\b", parts[0][1]):
                            raise ValueError("Verification score missing")
                    record.update(status="accepted", output=rendered)
                    outputs.append(rendered)
                    records.append({"agent": role, "output": rendered})
                except (ValueError, TypeError, KeyError):
                    count("rejections")
                    raise
                finally:
                    self.ledger.append(record)
            if selected == "alfworld":
                break
            active.add(selected)
        if calls > RECIPE["max_calls_per_step"]:
            raise AssertionError("Controller exceeded fixed call bound")
        reply = "\n".join(outputs)
        bridge, action, outcome = h0_reply(reply, page["admissible_commands"])
        if outcome in {"parser_error", "inadmissible"}:
            count("rejections")
        self.turns.append({"visible": page, "reply": reply})
        usage = {key: sum(u[key] for u in usages) if all(u is not None and u.get(key) is not None for u in usages)
                 else None for key in ("input_tokens", "output_tokens", "reasoning_tokens")}
        return TeacherResult(bridge, action, {"coa_reply": reply, "coa_agents": records,
            "coa_calls": calls, "action_outcome": outcome, "finish_reason": "stop",
            "prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"],
            "thinking_tokens": usage["reasoning_tokens"],
            "mechanism_teacher_billed_usage": {key: sum((u or {}).get(key, 0) or 0 for u in usages)
                for key in ("input_tokens", "output_tokens", "reasoning_tokens")},
            "teacher_usage_unknown_calls": sum(u is None or u.get("input_tokens") is None
                or u.get("output_tokens") is None for u in usages),
            "event_token_counts": "known lower bounds; unknown counts remain null in policy metrics and ledger",
            "mechanism_teacher_provider_id": "gpt-6-luna"})
