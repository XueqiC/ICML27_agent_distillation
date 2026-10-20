"""One durable capped transport per cell, with fixed logical request slots."""
import json
from pathlib import Path

from baselines.agentdistill import RECIPE, teacher_cap
from sed.manifest import authorize_task, frozen_write
from sed.storage import digest, write_json
from sed.visibility import visible_history, visible_observation, visible_task
from teachers.capped import from_config, BudgetExceeded
from teachers.openai_teacher import Teacher, TeacherResult
from teachers.stage import require_completed


class PromptTeacher(Teacher):
    """Request-specific visible instructions; slot identity never enters prompts."""
    request = None

    def prompt(self, history, task, visible_context):
        return self.request["messages"]

    def parse_response(self, raw):
        # Transport acceptance only; role-specific syntax is checked and logged
        # downstream. Cache malformed text too, so restart cannot redraw a slot.
        return None, None

    def cache_key(self, history, task, visible_context):
        return digest([super().cache_key(history, task, visible_context), self.request])


def read(path):
    return json.loads(Path(path).read_text())


class Budget:
    def __init__(self, root, config, identity, *, factory=from_config):
        self.root = Path(root)
        self.identity = digest(identity)
        n = identity["unique_instances"]
        self.cap = teacher_cap(n)
        self.limit = {"manager": 120 * n, "create": 120 * n,
                      "abstract": 120 * n, "cluster": 1, "consolidate": 120 * n}
        self.client = factory(config, output=self.root, cache_dir=self.root / "cache",
            usage_ledger=self.root / "usage.jsonl", run_id="agentdistill-" + self.identity,
            cap=self.cap, teacher_class=PromptTeacher,
            sampling={**config["models"]["teacher"]["sampling"], "max_output_tokens": RECIPE["box_output_tokens"]})

    def request(self, slot, role, task, prompt, history=(), visible=None):
        authorize_task(task, "generation")
        if task["source_split"] != "train" or role not in self.limit:
            raise ValueError("Teacher requests require support TRAIN and a registered role")
        self.client.check_health()
        visible = visible_observation(visible or {"observation": "", "admissible_commands": []})
        from sed.react import build_messages
        messages = build_messages(visible_history(history), visible_task(task), visible, prompt_template=prompt)
        record = {"cell": self.identity, "slot": slot, "role": role, "task_id": task["task_id"],
            "parent_seed_ids": task["parent_seed_ids"], "messages": messages}
        directory = self.root / "requests" / role / digest(slot)
        if not (directory / "request.json").exists():
            used = len(list((self.root / "requests" / role).glob("*/request.json")))
            if used >= self.limit[role]:
                raise BudgetExceeded("Teacher logical role cap reached; no label returned")
        frozen_write(directory / "request.json", record)
        result_path = directory / "result.json"
        if result_path.exists():
            result = read(result_path)
            if result["request_sha256"] != digest(record) or result["result_sha256"] != digest(result["response"]):
                raise ValueError("Teacher response identity changed")
            require_completed(TeacherResult(result["response"]["raw_text"], None,
                                            result["response"]["metadata"]))
            return result["response"]["raw_text"]
        self.client.teacher.request = {"slot": slot, "messages": messages}
        result = self.client.generate(visible_history(history), task, visible)
        require_completed(result)
        response = {"raw_text": result.raw_text, "metadata": result.metadata}
        write_json(result_path, {"request_sha256": digest(record), "response": response,
            "result_sha256": digest(response)}, exclusive=True)
        return result.raw_text
