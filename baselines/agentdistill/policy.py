"""Extended ReAct policy lowered to individual actions for the shared H0 runner."""
from copy import deepcopy
import json
from pathlib import Path
from teachers.stage import count

from baselines.agentdistill.prompts import CREATE, messages, policy_template
from baselines.agentdistill.sandbox import Machine, Rejected, validate
from sed.react import ActionParseError, parse_action
from sed.storage import JsonlLedger, digest
from sed.visibility import visible_observation
from students.vllm_student import Student, ContextOverflow
from teachers.openai_teacher import TeacherResult


class BoxStudent(Student):
    def __init__(self, *, tools, **kwargs):
        super().__init__(**kwargs)
        self.tools = tools

    def messages(self, history, task, visible_context):
        return messages(history, task, visible_context, self.tools)

    def request_sampling(self, history, task):
        return {"temperature": 0, "seed": 0}


class Manager:
    def __init__(self, budget, task, run):
        self.budget, self.task, self.run = budget, task, run
        self.calls = 0

    def generate(self, history, task, visible, tools):
        slot = [self.task["task_id"], self.run, self.calls]
        self.calls += 1
        raw = self.budget.request(slot, "manager", self.task, policy_template(tools, manager=True), history, visible)
        return TeacherResult(raw, None, {"teacher_slot": slot})

    def create(self, request, history, visible):
        raw = self.budget.request([self.task["task_id"], self.run, self.calls - 1], "create", self.task,
            CREATE + "\nManager subtask:\n" + request, history, visible)
        row = json.loads(raw)
        if set(row) != {"tool", "arguments"}:
            raise Rejected("Invalid creator schema")
        validate(row["tool"])
        return row


class ToolPolicy:
    """One coroutine yield is one run_episode turn and one real env.step.

The model receives its extended reply and the returned observations, never
internal flattened execution metadata. Raw calls and rejected tools are durable.
"""
    def __init__(self, model, tools, log_dir):
        for tool in tools:
            validate(tool)
        if len({t["name"] for t in tools}) != len(tools):
            raise ValueError("Tool names must be unique")
        self.model, self.tools = model, deepcopy(tools)
        self.history, self.pending, self.active = [], None, None
        self.turns, self.executed = 0, 0
        self.records = []
        self.model_calls = []
        self.log_dir = Path(log_dir)
        self.spec = {"method": "agentdistill", "label": "different interface",
            "box_sha256": digest(tools), "base": getattr(model, "spec", None)}

    def log(self, row):
        JsonlLedger(self.log_dir / "tools.jsonl").append(row)

    def receive(self, visible):
        if not self.active:
            return None
        try:
            action = self.active["generator"].send(visible)
            self.active["actions"] += 1
            return action
        except StopIteration as done:
            self.end_tool("completed", done.value)
        except Rejected as exc:
            count("rejections")
            self.end_tool("rejected", {"error": str(exc)})
        return None

    def end_tool(self, status, value):
        active = self.active
        row = {"turn": active["turn"], "tool": active["tool"], "arguments": active["arguments"],
               "actions": active["actions"], "status": status, "return": value}
        self.records.append(row)
        self.log(row)
        self.active = None
        self.pending["feedback"] = json.dumps({"tool": row["tool"]["name"], "status": status, "observations": value})

    def result(self, action, metadata=None):
        self.executed += 1
        return TeacherResult("Action: " + action, action,
            {"agentdistill_turn": self.turns, "tool_action": bool(self.active),
             "prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0,
             "thinking_chars": 0, **(metadata or {})})

    def generate(self, history, task, visible_context):
        visible = visible_observation(visible_context)
        if self.active:
            action = self.receive(visible)
            if action is not None:
                return self.result(action)
        current = visible
        if self.pending:
            self.history.append({"visible": self.pending["before"], "reply": self.pending["raw"]})
            if self.pending.get("feedback"):
                current = {**visible, "observation": self.pending["feedback"] + "\n" + visible["observation"]}
            self.pending = None
        while self.turns < 40:
            self.turns += 1
            try:
                result = (self.model.generate(self.history, task, current, self.tools) if isinstance(self.model, Manager)
                          else self.model.generate(self.history, task, current))
            except ContextOverflow as exc:
                self.model_calls.append(exc.metadata)
                raise
            self.model_calls.append(result.metadata)
            raw = result.raw_text
            self.log({"turn": self.turns, "status": "model_reply", "raw_text": raw, "metadata": result.metadata})
            self.pending = {"before": current, "raw": raw}
            try:
                action = parse_action(raw)
                if action.startswith("create_mcp "):
                    if not isinstance(self.model, Manager):
                        raise Rejected("Student cannot create tools")
                    request = json.loads(action[len("create_mcp "):])
                    if set(request) != {"request"} or type(request["request"]) is not str:
                        raise Rejected("Invalid creation request")
                    created = self.model.create(request["request"], self.history, visible)
                    tool, arguments = created["tool"], created["arguments"]
                    if tool["name"] in {t["name"] for t in self.tools}:
                        raise Rejected("Tool name already exists")
                    self.tools.append(tool)
                elif action.startswith("mcp "):
                    call = json.loads(action[4:])
                    if set(call) != {"name", "arguments"}:
                        raise Rejected("Invalid tool call")
                    tool = next((t for t in self.tools if t["name"] == call["name"]), None)
                    if tool is None:
                        raise Rejected("Unknown tool")
                    arguments = call["arguments"]
                else:
                    # Preserve shared H0 parsing, including inadmissible actions.
                    return TeacherResult(raw, action, result.metadata)
                machine = Machine(tool, arguments, visible)
                self.active = {"generator": machine.run(), "tool": tool, "arguments": arguments,
                               "actions": 0, "turn": self.turns}
                action = next(self.active["generator"])
                self.active["actions"] = 1
                return self.result(action, result.metadata)
            except (Rejected, ActionParseError, ValueError, TypeError, KeyError, StopIteration) as exc:
                count("rejections")
                feedback = "Rejected tool/reply: use the documented interface and current admissible commands."
                self.log({"turn": self.turns, "status": "rejected", "raw_text": raw,
                          "error_type": type(exc).__name__})
                self.active = None
                self.history.append({"visible": current, "reply": raw})
                self.pending = None
                current = {**visible, "observation": feedback + "\n" + visible["observation"]}
        # No extra model call and no fake environment action after model-turn cap.
        return TeacherResult("AgentDistill manager turn budget exhausted", None,
            {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0, "thinking_chars": 0})

    def finish(self, episode):
        if self.active:
            last = episode["steps"][-1]["visible"] if episode["steps"] else episode["initial"]["visible"]
            action = self.receive(last)
            if action is not None:
                # A requested continuation was NOT executed after termination.
                self.active["actions"] -= 1
                self.end_tool("stopped_at_episode_boundary", {"observations": last})
        # Count actual model requests (including rejected replies), not synthetic
        # low-level coroutine continuations. Keep action counts from the runner.
        from sed.runner import episode_metrics
        episode["policy_calls"] = self.model_calls
        episode["agentdistill_model_turns"] = self.turns
        episode["label"] = "different interface"
        episode["metrics"] = episode_metrics(episode)

    def close(self):
        if hasattr(self.model, "close"):
            self.model.close()
