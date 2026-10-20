"""SmartAD Eq. 3/5: exact-prefix turn NLL and assistant segment weights."""
import math
import re

from baselines.common.conversation import conversation, sequence_rows
from train.dataset import TURN_ID, TURN_TOKEN

WEIGHTS = {"reason": 1.0, "action": 1.5, "final": 2.0}


def segment_spans(reply, *, final):
    """H0 Action line (including its header); format/turn end remain reason."""
    matches = list(re.finditer(r"(?m)^[ \t]*Action[ \t]*:[ \t]*[^\r\n]+", reply))
    if not matches:
        raise ValueError("SmartAD requires an H0 Action span")
    # The H0 parser executes the last Action line. Other text stays reason.
    match = matches[-1]
    return [(match.start(), match.end(), "final" if final else "action")]


def encode_episode(episode, tokenizer, *, max_length=32768):
    example = conversation(episode, tokenizer, source="trusted_teacher_seed",
                           packing="turn", max_length=max_length)
    for index, (row, step) in enumerate(zip(example["turns"], episode["steps"])):
        reply = step["action_raw"].strip()
        encoded = tokenizer(reply + TURN_TOKEN, add_special_tokens=False, return_offsets_mapping=True)
        start = row["prompt_tokens"]
        if encoded["input_ids"] != row["input_ids"][start:]:
            raise ValueError("SmartAD offsets differ from exact-prefix target IDs")
        spans = segment_spans(reply, final=index == len(episode["steps"]) - 1)
        kinds = []
        for token, (left, right) in zip(encoded["input_ids"], encoded["offset_mapping"]):
            kind = next((kind for lo, hi, kind in spans if left < hi and right > lo), "reason")
            kinds.append("reason" if token == TURN_ID else kind)
        row["segment_labels"] = [None] * start + kinds
        row["token_weights"] = [0.0] * start + [WEIGHTS[kind] for kind in kinds]
        row["label_provenance"].update(smartad_spans=spans, segment_weights=WEIGHTS)
    example["label_provenance"]["smartad"] = "H0 Action spans; last turn Action is final; other targets reason"
    return example


def macro_nll(turn_token_losses):
    """Eq. 3: token mean inside each assistant turn, then equal turn mean."""
    if not turn_token_losses or any(not values for values in turn_token_losses):
        raise ValueError("NLL needs nonempty assistant turns")
    means = [sum(values) / len(values) for values in turn_token_losses]
    if not all(math.isfinite(value) for value in means):
        raise ValueError("Nonfinite candidate NLL")
    return sum(means) / len(means)


def score_episode(model, tokenizer, episode, *, device="cuda"):
    import torch
    from baselines.common.fsdp_train import sequence_logps
    example = conversation(episode, tokenizer, source="trusted_teacher_seed", packing="turn")
    model.eval()
    turns = []
    with torch.no_grad():
        for row in sequence_rows(example):
            count = row["supervised_tokens"]
            loss = -sequence_logps(model, [row], device)[0].float().item()
            turns.append({"tokens": count, "nll_sum": loss, "nll_mean": loss / count})
    score = sum(row["nll_mean"] for row in turns) / len(turns)
    if not math.isfinite(score):
        raise ValueError("Nonfinite candidate NLL")
    return {"nll": score, "turns": turns, "equation": "mean_turn(mean_token(NLL))"}


def weight_sum(row):
    values = row["token_weights"]
    if len(values) != len(row["labels"]) or any(
            (weight != 0 if label == -100 else weight not in WEIGHTS.values())
            for weight, label in zip(values, row["labels"])):
        raise ValueError("SmartAD weights must align with assistant label positions")
    total = sum(values)
    if total <= 0:
        raise ValueError("No weighted assistant tokens")
    return total


def microsteps(examples, indices, *, scoring="per_turn"):
    rows = [row for i in indices for row in sequence_rows(examples[i])]
    total = sum(weight_sum(row) for row in rows)
    # Shared train_loop divides by len(indices); cancel it here. Each row's
    # objective is a weighted SUM, so accumulated gradients equal Eq. 5.
    units = [examples[i] for i in indices] if scoring == "shared_history" else rows
    return [(row, len(indices) / total) for row in units]


def weighted_ce(logits, labels, weights):
    """Eq. 5 over a whole batch; logits/labels already causally aligned."""
    import torch
    import torch.nn.functional as F
    active = labels != -100
    if weights.shape != labels.shape or torch.any(weights[~active] != 0) or torch.any(weights[active] <= 0):
        raise ValueError("Invalid label-aligned weights")
    denom = weights.sum()
    if denom <= 0:
        raise ValueError("No weighted labels")
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             labels.reshape(-1), reduction="none", ignore_index=-100).reshape(labels.shape)
    return (losses * weights).sum() / denom


def objective(model, row, device, *, scoring="per_turn"):
    import torch
    from baselines.common.fsdp_train import sequence_logps
    from baselines.common.shared_history import shared_history_logps
    turns = sequence_rows(row)
    for turn in turns:
        weight_sum(turn)
    if scoring == "shared_history":
        logps = shared_history_logps(model, [row], device, per_token=True)[0]
    elif scoring == "per_turn":
        logps = torch.cat([sequence_logps(model, [turn], device, per_token=True)[0] for turn in turns])
    else:
        raise ValueError("Scoring must be per_turn or shared_history")
    weights = torch.tensor([w for turn in turns for w, label in zip(turn["token_weights"], turn["labels"])
                            if label != -100], device=device)
    return -(logps.reshape(-1) * weights).sum(), {}
