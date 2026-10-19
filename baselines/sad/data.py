"""SAD visible-span masks and a fixed, token-count curriculum."""
import math
import re

from baselines.common.conversation import conversation, sequence_rows
from baselines.common.schedule import dose
from train.dataset import TURN_ID, TURN_TOKEN

TOKEN_BUDGET = 655_000_000
DENOMINATORS = (16, 8, 4, 2, 1)


def spans(reply):
    """Reason paragraphs and action lines; other labelled sections stay masked."""
    headers = list(re.finditer(r"(?m)^[ \t]*([A-Za-z][A-Za-z _-]*):", reply))
    result = []
    for i, match in enumerate(headers):
        kind = match.group(1)
        if kind == "Thought":
            end = headers[i + 1].start() if i + 1 < len(headers) else len(reply)
            result.append((match.start(), end, "reason"))
        elif kind == "Action":
            end = reply.find("\n", match.end())
            result.append((match.start(), len(reply) if end == -1 else end, "action"))
    if not any(kind == "action" for _, _, kind in result):
        raise ValueError("SAD reply requires an Action line")
    return result


def encode_episode(episode, tokenizer, *, max_length=32768):
    example = conversation(episode, tokenizer, source="trusted_teacher_seed",
                           packing="turn", max_length=max_length)
    reason = action = 0
    for row, step in zip(example["turns"], episode["steps"]):
        reply = step["action_raw"].strip()
        target = reply + TURN_TOKEN
        encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
        start = row["prompt_tokens"]
        if encoded["input_ids"] != row["input_ids"][start:]:
            raise ValueError("SAD offsets differ from encode_turn target IDs")
        ranges = spans(reply)
        kinds = []
        for token, (left, right) in zip(encoded["input_ids"], encoded["offset_mapping"]):
            selected = {kind for lo, hi, kind in ranges if left < hi and right > lo}
            if len(selected) > 1:
                raise ValueError("Tokenizer token crosses SAD reason/action boundary")
            kinds.append("action" if token == TURN_ID else next(iter(selected), None))
        if kinds[-1] != "action" or encoded["input_ids"][-1] != TURN_ID:
            raise ValueError("SAD requires Gemma turn-end supervision")
        row["labels"] = [-100] * start + [token if kind else -100
                                      for token, kind in zip(encoded["input_ids"], kinds)]
        row["reason_tokens"] = kinds.count("reason")
        row["action_tokens"] = kinds.count("action")
        row["supervised_tokens"] = row["reason_tokens"] + row["action_tokens"]
        row["sad_spans"] = ranges
        reason += row["reason_tokens"]
        action += row["action_tokens"]
    example.update(reason_tokens=reason, action_tokens=action, complexity=reason + action,
                   supervised_tokens=reason + action)
    example["label_provenance"].update(sad_complexity=reason + action,
        reason_tokens=reason, action_tokens=action, lambda_r=1, lambda_a=1,
        mask="Thought sections, Action lines and turn-end; tokenizer offset overlap")
    return example


def curriculum(examples):
    # Python's stable sort preserves verified collection order for ties.
    return sorted(examples, key=lambda row: row["complexity"])


def token_plan(examples, original_batch=64, token_budget=TOKEN_BUDGET):
    if type(token_budget) is not int or token_budget < 1:
        raise ValueError("Positive integer processed-token budget required")
    plan = dose(len(examples), original_batch, epochs=1)
    costs = [sum(len(r["input_ids"]) for r in sequence_rows(e)) for e in examples]
    if min(costs) < 1:
        raise ValueError("Empty SAD trajectory")
    batches = [list(range(i, min(i + plan["b_prime"], len(examples))))
               for i in range(0, len(examples), plan["b_prime"])]
    batch_costs = [sum(costs[i] for i in indices) for indices in batches]
    complete_epochs, remainder = divmod(token_budget - 1, sum(costs))
    partial_steps, subtotal = 0, 0
    for cost in batch_costs:
        partial_steps += 1
        subtotal += cost
        if subtotal > remainder:
            break
    return {**plan, "epochs": complete_epochs + 1,
            "total_steps": complete_epochs * len(batches) + partial_steps,
            "processed_token_budget": token_budget,
            "planned_processed_tokens": complete_epochs * sum(costs) + subtotal,
            "curriculum": "stable complexity ascending; identical each epoch",
            "checkpoint_tokens": [math.ceil(token_budget / d) for d in DENOMINATORS],
            "token_unit": "logical input positions, including masked repeated history; excludes rank padding"}


def ordered_batches(n, batch_size, total_steps):
    update = epoch = 0
    while update < total_steps:
        for start in range(0, n, batch_size):
            yield {"epoch": epoch, "indices": list(range(start, min(start + batch_size, n)))}
            update += 1
            if update == total_steps:
                return
        epoch += 1
