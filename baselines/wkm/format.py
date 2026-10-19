"""H0 bridge for WKM's process_ag/process_wm and synthesis prompts (f1fe28f)."""
from copy import deepcopy
import json

from baselines.wkm.prompts import (TASK_KNOWLEDGE, STATE_KNOWLEDGE, ALFWORLD_EXAMPLES,
                                  ALFWORLD_WM_SYSTEM_PROMPT)
from sed.react import build_messages
from sed.visibility import visible_observation


def trajectory_text(messages):
    # Upstream concatenates conversation values. Keep complete visible observations.
    return "\n".join(m["content"] for m in messages[1:])


def task_synthesis(chosen, rejected):
    return [{"role": "user", "content": TASK_KNOWLEDGE["Instruction"] + TASK_KNOWLEDGE["Input"].format(
        Success_T=trajectory_text(chosen), Failed_T=trajectory_text(rejected) if rejected else
        "[No rejected trajectory: the single executed exploration succeeded.]",
        Example=ALFWORLD_EXAMPLES["pick_and_place"])}]


def state_synthesis(trajectory):
    return [{"role": "user", "content": STATE_KNOWLEDGE["Instruction"].format(task="alfworld") + "\n" +
        STATE_KNOWLEDGE["Example"]["alfworld"] + "\n" + STATE_KNOWLEDGE["Input"].format(Trajectory=trajectory)}]


def agent_messages(messages, knowledge):
    result = deepcopy(messages)
    # process_ag appends task knowledge to the FIRST task observation only.
    # H0's JSON observations/parser and system instruction otherwise stay intact.
    visible = visible_observation(json.loads(result[1]["content"]))
    visible["observation"] += "\n" + knowledge
    result[1]["content"] = json.dumps(visible, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return result


def world_prompt(messages, task_knowledge=None, states=()):
    """process_wm: previous agent action + observation -> knowledge.

    Knowledge replies, rather than agent replies, form the WM's assistant history.
    An extra initial-state request covers the spec's state-at-every-step requirement.
    """
    result = [{"role": "system", "content": ALFWORLD_WM_SYSTEM_PROMPT}, dict(messages[1])]
    if task_knowledge is None:
        return result
    result.append({"role": "assistant", "content": task_knowledge})
    j = (len(messages) - 2) // 2
    if j == 0:
        return result + [{"role": "user", "content": "State Knowledge: summarize the current state."}]
    for i in range(1, j + 1):
        result.append({"role": "user", "content": messages[2*i]["content"] + "\n" + messages[2*i+1]["content"]})
        if i < j:
            result.append({"role": "assistant", "content": states[i]})
    return result


def visible_messages(history, task, visible):
    return build_messages(history, task, visible)
