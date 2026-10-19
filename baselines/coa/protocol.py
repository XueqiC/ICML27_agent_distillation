"""CoA tags over the H0 visible interface; no audit fields become model input."""
import json
import re

from sed.react import action_outcome
from sed.visibility import visible_observation
from train.dataset import EMPTY_THOUGHT, TURN_TOKEN, encode_turn
from baselines.common.conversation import group_turns

ROLES = ("think", "plan", "reflection", "double_check", "alfworld")
SYSTEM = """Solve the visible ALFWorld task using a chain of agents. Use <think> for concise
coordination, <plan> for a plan, <reflection> to critique progress, and <double_check>
to verify it (Score: 1 to 4; scores below 3 require reconsideration). Select roles
as needed. The single tool is <alfworld>Action: one command</alfworld>. Emit exactly
one tool call at the end of each response. Use the supplied admissible commands.
Only the environment supplies <observation>. Never invent observations. Real
execution, not an answer tag or a confidence score, determines completion."""
BLOCK = re.compile(r"<(think|plan|reflection|double_check|alfworld)>(.*?)</\1>", re.S)


def blocks(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty CoA response")
    result, end = [], 0
    for match in BLOCK.finditer(text):
        if text[end:match.start()].strip() or re.search(r"</?[A-Za-z_]+>", match[2]):
            raise ValueError("Invalid or nested CoA tag")
        if not match[2].strip():
            raise ValueError("Empty CoA agent output")
        result.append((match[1], match[2].strip()))
        end = match.end()
    if not result or text[end:].strip():
        raise ValueError("Unclosed/unknown CoA tag")
    return result


def tool_action(text, commands):
    try:
        parts = blocks(text)
        if parts[-1][0] != "alfworld" or sum(r == "alfworld" for r, _ in parts) != 1:
            raise ValueError("Exactly one final tool call required")
        return action_outcome(parts[-1][1], commands)
    except ValueError:
        return None, "parser_error"


def h0_reply(text, commands):
    action, outcome = tool_action(text, commands)
    # The shared runner reparses this bridge. Original output is retained in
    # coa_reply and the raw transport ledger, including malformed replies.
    return ("Action: " + action if action is not None else "Invalid CoA tool call"), action, outcome


def observation(page):
    return "<observation>" + json.dumps(visible_observation(page), ensure_ascii=False, sort_keys=True) + "</observation>\n\n"


def messages(turns, page):
    out = [{"role": "system", "content": SYSTEM}]
    for turn in turns:
        out += [{"role": "user", "content": observation(turn["visible"])},
                {"role": "assistant", "content": turn["reply"]}]
    out.append({"role": "user", "content": observation(page)})
    return out


def episode_turns(episode, *, strict=True):
    from sed.manifest import authorize_task
    authorize_task(episode, "generation")
    page, turns = visible_observation(episode["initial"]["visible"]), []
    for step in episode["steps"]:
        reply = step["policy_metadata"]["coa_reply"]
        bridge, action, _ = h0_reply(reply, page["admissible_commands"])
        if bridge != step["action_raw"] or action != step["action_parsed"]:
            raise ValueError("CoA output differs from executed H0 action")
        if strict:
            blocks(reply)
        turns.append({"visible": page, "reply": reply,
                      "finish_reason": step["policy_metadata"].get("finish_reason", "stop")})
        page = visible_observation(step["visible"])
    return turns


def encode_episode(episode, tokenizer, *, rl=False):
    turns, encoded, history = episode_turns(episode, strict=not rl), [], []
    for index, turn in enumerate(turns):
        prompt_messages = messages(history, turn["visible"])
        if rl and not turn["reply"].strip():
            # Empty stop completions still sampled a turn end. Context overflow
            # before generation has no step and never invents a target token.
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            if not prompt.endswith(EMPTY_THOUGHT):
                raise ValueError("Gemma OFF generation prefix changed")
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            ending = tokenizer.encode(TURN_TOKEN, add_special_tokens=False)
            if ending != [106] or len(prefix) + 1 > 32768:
                raise ValueError("Invalid empty CoA response boundary/context")
            row = {"input_ids": prefix + ending, "labels": [-100] * len(prefix) + ending,
                "attention_mask": [1] * (len(prefix) + 1), "supervised_tokens": 1,
                "empty_visible_turns_masked": 0, "prompt_tokens": len(prefix)}
        else:
            row = encode_turn(tokenizer, prompt_messages, turn["reply"], max_length=32768)
        if rl and turn["finish_reason"] == "length":
            # A length-limited rollout never sampled the chat EOS.
            for key in ("input_ids", "labels", "attention_mask"):
                row[key] = row[key][:-1]
            row["supervised_tokens"] -= 1
        row.update(packing="turn", label_provenance={"task_id": episode["task_id"],
            "turn_index": index, "source": "coa_student" if rl else "coa_multi_agent_teacher",
            "training_eligible": True})
        encoded.append(row)
        history.append(turn)
    if rl and not any(r["supervised_tokens"] for r in encoded):
        return {"turns": encoded, "packing": "turn", "supervised_tokens": 0,
            "label_provenance": {"task_id": episode["task_id"], "source": "coa_student",
                                 "training_eligible": True}}
    return group_turns(encoded)


def alpaca_record(episode):
    """Released dataset_info uses default Alpaca instruction/input/output fields."""
    turns = episode_turns(episode)
    output = ""
    for index, turn in enumerate(turns):
        output += turn["reply"] + "\n\n" + observation(episode["steps"][index]["visible"])
    return {"instruction": SYSTEM + "\n\n" + observation(turns[0]["visible"]),
            "input": "", "output": output}
