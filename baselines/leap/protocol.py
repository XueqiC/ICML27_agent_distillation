"""Pinned LEAP prompts and strict sampled-text parsing; no upstream imports."""
import json
import importlib.metadata
from pathlib import Path
import re

from sed.manifest import authorize_task
from sed.react import ActionParseError, parse_action
from sed.storage import file_hash
from sed.thinking import split_thinking
from sed.visibility import visible_observation

PIN = "a3376d89a2eeee88f03555a4174a0bdf7faf8be6"
UPSTREAM = Path(__file__).resolve().parents[2] / "third_party/leap_llm"
TEMPLATES = {
    "extract": "prompts/alfworld/extract_privileged_state_template.j2",
    "privileged": "prompts/alfworld/correction_oracle_template.j2",
    "zero": "prompts/alfworld_realizability_ablation/generation_oracle_0_privileged_template.j2",
}
LABELS = {"privileged": "LEAP (privileged teacher)", "zero": "LEAP (0% privileged ablation)"}
HEADINGS = ("Task", "Essential Objects", "Critical Locations", "Optimal Action Sequence")


def template_hashes():
    return {name: file_hash(UPSTREAM / path) for name, path in TEMPLATES.items()}


def expert_source_hashes():
    package = importlib.metadata.distribution("alfworld")
    if package.version != "0.4.2":
        raise ValueError("LEAP requires alfworld 0.4.2")
    files = ["alfworld/agents/environment/alfred_tw_env.py",
             "alfworld/agents/expert/handcoded_expert.py",
             "alfworld/agents/expert/handcoded_expert_tw.py"]
    return {name: file_hash(package.locate_file(name)) for name in files}


def require_train(task):
    authorize_task(task, "generation")
    if not task["task_id"].startswith("train/") or task.get("source_split", "train") != "train":
        raise ValueError("LEAP requires native frozen seed TRAIN tasks")


def trajectory(episode, *, expert=False):
    """Only executed/visible fields, never audit, facts, success or task path."""
    require_train(episode)
    if episode.get("schema_version") != 2:
        raise ValueError("Harness ReAct episode required")
    visible, rows = visible_observation(episode["initial"]["visible"]), []
    for index, step in enumerate(episode["steps"]):
        reply, thinking = split_thinking(step["action_raw"])
        if thinking:
            raise ValueError("Hidden thought channel cannot enter LEAP data")
        try:
            action = parse_action(reply)
        except ActionParseError:
            action = ""  # Original parser-error turn, not a fabricated action.
        reason = reply.rsplit("\nAction:", 1)[0] if "\nAction:" in reply else reply
        if reason.startswith("Thought:"):
            reason = reason[len("Thought:"):].strip()
        row = {"observation": visible["observation"], "action": action,
               "candidate_actions": visible["admissible_commands"]}
        if not expert:
            row.update(timestep=index, reason=reason)
        rows.append(row)
        visible = visible_observation(step["visible"])
    return rows


def render(kind, rows, privileged_state=None):
    from jinja2 import StrictUndefined, Template
    if kind not in TEMPLATES:
        raise ValueError("Unknown LEAP template")
    if kind == "privileged" and not privileged_state:
        raise ValueError("Privileged correction requires extracted expert state")
    if kind != "privileged" and privileged_state is not None:
        raise ValueError("Privileged state forbidden in extraction/zero template")
    template = Template((UPSTREAM / TEMPLATES[kind]).read_text(), undefined=StrictUndefined)
    payload = {"input": json.dumps(rows, ensure_ascii=False)} if kind == "extract" else {
        "student_trajectory": json.dumps(rows, ensure_ascii=False)}
    if kind == "privileged":
        payload["privileged_state"] = privileged_state
    return [{"role": "system", "content": template.render(system=True)},
            {"role": "user", "content": template.render(system=False, **payload)}]


def parse_extraction(raw):
    if not isinstance(raw, str) or not raw.strip() or split_thinking(raw)[1]:
        raise ValueError("Invalid extraction text")
    if not all(re.search(r"\b" + re.escape(heading) + r"\s*\**:", raw, re.I) for heading in HEADINGS):
        raise ValueError("Extraction must contain all four official headings")
    return raw.strip()


def parse_correction(raw, length, *, privileged_state=None):
    """Read only labels. Echoed observations/history and flags never train.

    Accept fenced or bare JSON without upstream's lossy string rewrites.
    Admissibility is deliberately NOT a filter. Reject malformed alignment,
    model control channels and literal privileged-summary copies.
    """
    match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", raw, re.S)
    value = json.loads(match.group(1) if match else raw)
    if (not isinstance(value, dict) or not isinstance(value.get("summary"), str)
            or not isinstance(value.get("trajectory"), list) or len(value["trajectory"]) != length):
        raise ValueError("Correction must align with every original step")
    result = []
    for index, row in enumerate(value["trajectory"]):
        if not isinstance(row, dict) or ("timestep" in row and
                (type(row["timestep"]) is not int or row["timestep"] != index)):
            raise ValueError("Correction timestep alignment differs")
        reason, action = row.get("corrected_reason"), row.get("corrected_action")
        if not all(isinstance(v, str) and v.strip() for v in (reason, action)):
            raise ValueError("Correction requires nonempty reason/action strings")
        if "\n" in action or "\r" in action:
            raise ValueError("Correction action must be a single command")
        target = f"Thought: {reason.strip()}\nAction: {action.strip()}"
        if (split_thinking(target)[1] or any(h.lower() in target.lower() for h in HEADINGS[1:])
                or privileged_state and privileged_state.strip() in target):
            raise ValueError("Privileged summary/control channel in correction target")
        result.append({"corrected_reason": reason.strip(), "corrected_action": action.strip()})
    return result
