"""Public-only prompts; never serialize evidence records or arbitrary env fields."""

from dataclasses import dataclass
from hashlib import sha256
import json

from sed.storage import canonical_bytes
from sed.thinking import split_thinking
from repair.schemas import SCHEMA_VERSION

DIAGNOSIS = """Analyze this failed interactive agent execution. Keep the task, initial state and
success criterion fixed. A different or longer valid route is not an error. Distinguish
invalid actions, valid but insufficient actions, preservable choices, and unknowns.
Do not invent observations or assume the first divergence is the first error.
Return JSON only: {"expressed_plan":[{"subgoal":"...","evidence_steps":[0],
"interpretation_status":"hypothesis"}],"preservable_choices":[{"choice":"...",
"evidence_steps":[0],"status":"candidate_until_execution"}],"unmet_requirements":[
{"requirement":"...","evidence_steps":[0],"certainty":"unknown"}],
"candidate_joint_changes":[{"block_ids":["p1","p2"],"explanation":"..."}],"unknowns":["..."]}.
Lists may be empty. Certainty is supported or unknown (the design's supported_or_unknown
literal is also accepted). Step IDs are zero-based action indices; -1 denotes the initial observation."""

PROPOSAL = """Propose up to 3 distinct complete patches for the same task and initial state.
Prefer successful plans that minimally change the student's actions; preserve valid choices.
Allow joint noncontiguous changes. Include every exploration, movement and recovery action.
These are hypotheses until executed. Never modify task, initial state or observations.
Return JSON only: {"candidates":[{"original_trajectory_id":"supplied ID","blocks":[
{"block_id":"p1","op":"insert_before","index":0,"actions":["command"]}]}]}.
At most 4 blocks per candidate. Other block forms:
{"block_id":"p2","op":"replace","start":0,"end_exclusive":1,"actions":["command"]},
{"block_id":"p3","op":"delete","start":0,"end_exclusive":1,"actions":[]}.
ALL indices refer to ORIGINAL zero-based actions, never edited offsets. Insert at n appends.
Ranges cannot overlap; one insertion per boundary; no insertion inside a removed range.
Insertions at a range's left boundary run first. Reordering costs delete plus insert.
Commands must be actual single-line environment commands, without Thought/Action wrappers.
If feedback is present, use only its real observations and return FULL NEW patches."""

FAR = PROPOSAL + """
Alternative-route variant: propose a DIFFERENT valid plan fixing the SAME task requirements,
using the SAME target object instances and required operation types as the supplied successful
Near execution. Keep action length comparable. No meaningless actions, loops or padding.
Prefer a valid alternative order or route, not a damaged or artificially lengthened Near plan."""

RATIONALE = """Write a brief PUBLIC decision justification for the supplied current action.
Use ONLY the task and current public prefix. No future observations, outcome, future plan,
or post-hoc diagnosis is available. Exploring an unseen container is allowed; claiming its
unseen contents are known is not. Return JSON only: {"reasoning":"at most 64 tokens",
"evidence_steps":[-1],"unsupported":false}. Cite earlier zero-based action step IDs only;
-1 denotes the initial observation. If the action or justification needs unknown facts,
return unsupported:true. Do not provide hidden reasoning, action lines or new observations."""

TEMPLATES = {"diagnosis": DIAGNOSIS, "proposal": PROPOSAL, "far": FAR, "rationale": RATIONALE}
VERSIONS = {kind: f"{SCHEMA_VERSION}/{kind}/1" for kind in TEMPLATES}


@dataclass(frozen=True)
class Prompt:
    kind: str
    payload: dict
    format_retry: bool = False

    @property
    def template_sha256(self):
        return sha256((VERSIONS[self.kind] + "\n" + TEMPLATES[self.kind]).encode()).hexdigest()

    def to_json(self):
        return {"kind": self.kind, "template_version": VERSIONS[self.kind],
                "template_sha256": self.template_sha256, "format_retry": self.format_retry,
                "payload": self.payload}

    def messages(self):
        retry = "\nFormat retry: the preceding response failed validation. Follow this schema exactly." if self.format_retry else ""
        return [{"role": "system", "content": TEMPLATES[self.kind] + retry},
                {"role": "user", "content": canonical_bytes(self.payload).decode()}]


def observation(visible):
    value = visible["observation"]
    if type(value) is not str:
        raise TypeError("Public observation must be text")
    return value


def public_trace(trajectory):
    """No admissible commands, audit fields, task paths or policy metadata."""
    steps, action_index = [], 0
    for step in trajectory.steps:
        action = step.get("action_parsed")
        steps.append({"action_index": action_index if action is not None else None,
                      "reply": split_thinking(step["action_raw"])[0],
                      "action": action,
                      "observation": observation(step["visible"]) if step["visible"] else None})
        action_index += int(action is not None)
    return {"initial_observation": observation(trajectory.initial["visible"]), "steps": steps,
            "success": trajectory.success, "termination": trajectory.termination_reason}


def task_text(task):
    value = task["instruction"]
    if type(value) is not str:
        raise TypeError("Task text must be a string")
    return value


def diagnosis_prompt(task, trajectory, action_syntax, max_steps):
    return Prompt("diagnosis", {"task": task_text(task), "trace": public_trace(trajectory),
                               "action_syntax": action_syntax, "step_limit": max_steps})


def proposal_prompt(task, trajectory, diagnosis, action_syntax, max_steps, *,
                    executions=(), rejections=(), near=None):
    payload = {**diagnosis_prompt(task, trajectory, action_syntax, max_steps).payload,
               "original_trajectory_id": trajectory.trajectory_id,
               "diagnosis": diagnosis.prompt_view(),
               "replay_feedback": [public_trace(t) for t in executions],
               "patch_rejections": [{"patch_id": p, "reason": r} for p, r in rejections]}
    if near is not None:
        payload["near_execution"] = public_trace(near)
    return Prompt("far" if near is not None else "proposal", payload)


def rationale_prompt(task, initial_observation, prefix, current_action):
    # Each prefix item is built anew from these four scalar public fields.
    return Prompt("rationale", {"task": task_text(task), "initial_observation": initial_observation,
        "prefix": [{k: turn[k] for k in ("action_index", "reply", "action", "observation")}
                   for turn in prefix], "current_action": current_action})


def unpack_request(visible_context):
    value = json.loads(visible_context["observation"])
    if set(value) != {"kind", "template_version", "template_sha256", "format_retry", "payload"}:
        raise ValueError("Invalid repair prompt envelope")
    prompt = Prompt(value["kind"], value["payload"], value["format_retry"])
    if prompt.to_json() != value:
        raise ValueError("Repair prompt version/hash mismatch")
    return prompt
