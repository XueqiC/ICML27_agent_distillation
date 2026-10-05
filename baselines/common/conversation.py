"""Allowlisted harness episodes and exact train.dataset episode token spans.

No event/audit dictionary is ever serialized into a prompt. Receipt validation
belongs to the evidence loader; this module accepts its checked episode.
"""
from copy import deepcopy

from sed.manifest import authorize_task
from sed.react import build_messages
from sed.thinking import split_thinking
from sed.visibility import visible_observation
from train.dataset import episode_examples


def episode_messages(episode, *, trailing_observation=False):
    authorize_task(episode, "generation")
    if episode.get("schema_version") != 2:
        raise ValueError("Stage 2 event-log episode required")
    history = []
    visible = visible_observation(episode["initial"]["visible"])
    for step in episode["steps"]:
        reply, thinking = split_thinking(step["action_raw"])
        if thinking:
            raise ValueError("Hidden thought channel in visible episode")
        messages = build_messages(history, {}, visible)
        messages.append({"role": "assistant", "content": reply})
        history.append({"visible": visible, "reply": reply})
        visible = visible_observation(step["visible"])
    if not history:
        raise ValueError("Trajectory has no assistant turns")
    return build_messages(history, {}, visible) if trailing_observation else messages


def conversation(episode, tokenizer, *, source, max_length=32768):
    if source not in {"trusted_teacher_seed", "student_exploration"}:
        raise ValueError("Explicit trajectory source required")
    row, = episode_examples(episode, tokenizer, max_length=max_length, packing="episode")
    row = deepcopy(row)
    provenance = row["label_provenance"]
    provenance.update(source=source, trusted_seed=source == "trusted_teacher_seed")
    for turn in provenance["turns"]:
        turn["source"] = source
    row["messages"] = episode_messages(episode)
    return row


def encode_messages(messages, tokenizer, *, task_id, max_length=32768):
    """Round-trip only the visible schema, then use the existing span engine.

    DPO has a shared system+first-user prompt. Observation JSON is parsed and
    allowlisted; canonical build_messages equality rejects injected fields,
    changed instructions or alternate templates. Final outcome is never read.
    """
    import json
    if (len(messages) < 3 or messages[0]["role"] != "system"
            or messages[-1]["role"] != "assistant" or len(messages) % 2 != 1):
        raise ValueError("Conversation must end with an assistant turn")
    observations, replies = [], []
    for index in range(1, len(messages), 2):
        user, assistant = messages[index:index + 2]
        if user["role"] != "user" or assistant["role"] != "assistant":
            raise ValueError("Expected alternating observation/assistant turns")
        observations.append(visible_observation(json.loads(user["content"])))
        replies.append(assistant["content"])
    episode = {"schema_version": 2, "task_id": task_id, "split": "train", "sealed": False,
               "initial": {"visible": observations[0]}, "steps": [
                   {"event_id": f"pair:{i}", "action_raw": reply,
                    "visible": observations[min(i + 1, len(observations) - 1)]}
                   for i, reply in enumerate(replies)]}
    if episode_messages(episode) != messages:
        raise ValueError("Conversation differs from the allowlisted harness rendering")
    return conversation(episode, tokenizer, source="student_exploration", max_length=max_length)


def preference_conversation(pair, tokenizer, *, max_length=32768):
    prompt = pair["prompt"]
    if len(prompt) != 2 or [m["role"] for m in prompt] != ["system", "user"]:
        raise ValueError("DPO prompt must be system plus first user turn")
    rows = {side: encode_messages(prompt + pair[side], tokenizer,
                                 task_id=pair["task_id"], max_length=max_length)
            for side in ("chosen", "rejected")}
    # Only completed prompt turns are shared. Assistant headers/empty thought
    # markers belong to the masked continuation (one-turn and multi-turn differ).
    text = tokenizer.apply_chat_template(prompt, tokenize=False,
                     add_generation_prompt=False, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    for side, row in rows.items():
        if row["input_ids"][:len(ids)] != ids or any(v != -100 for v in row["labels"][:len(ids)]):
            raise ValueError("DPO prompt is not a shared masked token prefix")
        row["label_provenance"]["source"] = ("trusted_teacher_seed" if side == "chosen"
                                               else "student_exploration")
        row["label_provenance"]["trusted_seed"] = side == "chosen"
        for turn in row["label_provenance"]["turns"]:
            turn["source"] = row["label_provenance"]["source"]
        row["continuation_input_ids"] = row["input_ids"][len(ids):]
        row["continuation_labels"] = row["labels"][len(ids):]
    return {"task_id": pair["task_id"], "prompt_input_ids": ids, **rows}
