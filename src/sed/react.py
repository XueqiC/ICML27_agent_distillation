"""The shared teacher/student ReAct contract. Only visible fields enter prompts."""

import re

from sed.storage import canonical_bytes
from sed.visibility import visible_history, visible_observation, visible_task

PROMPT_VERSION = "alfworld-react-v2"
PROMPT_TEMPLATE = (
    "You act in an ALFWorld household environment. Complete the task in the "
    "initial observation using the visible conversation and admissible commands. "
    "Each reply should contain a brief visible 'Thought: ...' line (optional), "
    "then end with 'Action: <one admissible command>'. Copy the command exactly. "
    "Issue one action per reply. Observations and admissible_commands are supplied "
    "as JSON in user turns."
)
PARSER_FEEDBACK = "Your reply must end with a line 'Action: <one admissible command>'."


class ActionParseError(ValueError):
    pass


def parse_action(raw):
    if type(raw) is not str:
        raise ActionParseError("Policy reply must be text")
    # Horizontal whitespace only: an empty Action cannot consume the next line.
    lines = re.findall(r"^[ \t]*Action[ \t]*:[ \t]*([^\r\n]*)", raw, re.MULTILINE)
    if not lines or not lines[-1].strip():
        raise ActionParseError("Missing nonempty Action line")
    return lines[-1].strip()


def action_outcome(raw, commands):
    try:
        command = parse_action(raw)
    except ActionParseError:
        return None, "parser_error"
    return command, "legal" if command in commands else "inadmissible"


def build_messages(history, task, visible_context, *, prompt_template=PROMPT_TEMPLATE):
    instruction = visible_task(task)["instruction"]
    messages = [{"role": "system", "content": prompt_template}]
    for turn in visible_history(history):
        messages.append({"role": "user", "content": canonical_bytes(turn["visible"]).decode()})
        # Legacy action-only records are explicitly formatted as ReAct turns.
        reply = turn.get("reply", "Action: " + turn.get("action", ""))
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": canonical_bytes(
        visible_observation(visible_context)).decode()})
    if instruction:
        messages[1]["content"] = instruction + "\n" + messages[1]["content"]
    return messages
