"""Allowlisted trajectories with episode or exact evaluation-prefix turn rows.

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


def sequence_rows(example):
    """Only target-bearing turns get forwards; empty turns remain in history."""
    return ([row for row in example["turns"] if any(v != -100 for v in row["labels"])]
            if "turns" in example else [example])


def group_turns(turns):
    if not turns:
        raise ValueError("Trajectory/continuation has no assistant turns")
    if not any(t["supervised_tokens"] for t in turns):
        raise ValueError("Nothing to score: all assistant turns have empty visible content")
    first = turns[0]["label_provenance"]
    return {"turns": turns, "packing": "turn", "turn_count": len(turns),
            "supervised_tokens": sum(t["supervised_tokens"] for t in turns),
            "empty_visible_turns_masked": sum(t["empty_visible_turns_masked"] for t in turns),
            "label_provenance": {"task_id": first["task_id"], "source": first["source"],
                "training_eligible": first["training_eligible"],
                "turns": [t["label_provenance"] for t in turns]}}


def set_source(example, source):
    provenance = example["label_provenance"]
    for item in [provenance, *provenance["turns"]]:
        item.update(source=source, trusted_seed=source == "trusted_teacher_seed")


def conversation(episode, tokenizer, *, source, max_length=32768, packing="episode",
                 allow_empty_visible=False):
    """One logical trajectory, regardless of the number of forward sequences.

    Turn rows are exactly train.dataset's turn mode. Keeping them grouped makes
    X5 sampling count trajectories, including repeated instances, never turns.
    Parser-masking methods may opt into empty history; ordinary teacher SFT
    keeps its strict nonempty-target default.
    """
    if source not in {"trusted_teacher_seed", "student_exploration"}:
        raise ValueError("Explicit trajectory source required")
    rows = deepcopy(episode_examples(episode, tokenizer, max_length=max_length, packing=packing,
                                    allow_empty_visible=allow_empty_visible or source == "student_exploration"))
    row = group_turns(rows) if packing == "turn" else rows[0]
    set_source(row, source)
    row["messages"] = episode_messages(episode)
    return row


def encode_messages(messages, tokenizer, *, task_id, source="trusted_teacher_seed",
                    max_length=32768, packing="episode"):
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
    return conversation(episode, tokenizer, source=source, max_length=max_length, packing=packing)


def encode_continuation(prompt, continuation, tokenizer, *, task_id, source, max_length=32768):
    """Score every continuation reply after its own exact generation prefix.

    Prefix assistant turns remain native completed history and are never scored.
    Continuation log-probs are still summed across all assistant target tokens.
    """
    row = encode_messages(prompt + continuation, tokenizer, task_id=task_id,
                          source=source, max_length=max_length, packing="turn")
    prefix_turns = sum(m["role"] == "assistant" for m in prompt)
    row = group_turns(row["turns"][prefix_turns:])
    set_source(row, source)
    text = tokenizer.apply_chat_template(prompt, tokenize=False,
                     add_generation_prompt=False, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    for turn in row["turns"]:
        if turn["input_ids"][:len(ids)] != ids or any(v != -100 for v in turn["labels"][:len(ids)]):
            raise ValueError("Preference prompt is not a shared masked token prefix")
        turn["continuation_input_ids"] = turn["input_ids"][len(ids):]
        turn["continuation_labels"] = turn["labels"][len(ids):]
    return row, ids


def preference_conversation(pair, tokenizer, *, max_length=32768):
    prompt = pair["prompt"]
    if len(prompt) != 2 or [m["role"] for m in prompt] != ["system", "user"]:
        raise ValueError("DPO prompt must be system plus first user turn")
    rows = {}
    for side in ("chosen", "rejected"):
        rows[side], ids = encode_continuation(prompt, pair[side], tokenizer,
            task_id=pair["task_id"], max_length=max_length,
            source="trusted_teacher_seed" if side == "chosen" else "student_exploration")
    return {"task_id": pair["task_id"], "prompt_input_ids": ids,
            "empty_visible_turns_masked": sum(r["empty_visible_turns_masked"] for r in rows.values()), **rows}
