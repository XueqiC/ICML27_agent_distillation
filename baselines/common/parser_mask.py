"""Opt-in SAD/SmartAD masking; audit outcomes never enter model inputs."""
from sed.react import ActionParseError, parse_action


def mask_parser_rejected_replies(example, episode):
    """Mask complete targets without editing their native history serialization.

    Removed tokens are counted against the shared assistant-all encoding before
    method-specific spans (undefined for malformed replies), including turn end.
    """
    replies = removed = 0
    for row, step in zip(example["turns"], episode["steps"]):
        try:
            parse_action(step["action_raw"])
            rejected = False
        except ActionParseError:
            rejected = True
        outcome = step.get("action_outcome")
        if outcome is not None:
            recorded_rejection = outcome == "parser_error"
            if recorded_rejection != rejected:
                raise ValueError("Recorded action_outcome disagrees with H0 parse_action rejection")
            rejected = recorded_rejection
        count = sum(label != -100 for label in row["labels"]) if rejected else 0
        row["label_provenance"].update(parser_rejected_reply_masked=rejected,
                                      parser_rejected_supervised_tokens_removed=count)
        if rejected:
            row["labels"] = [-100] * len(row["input_ids"])
            row["supervised_tokens"] = 0
            row["assistant_spans"] = []
            replies += 1
            removed += count
    example["supervised_tokens"] = sum(row["supervised_tokens"] for row in example["turns"])
    example["label_provenance"]["parser_rejection_mask"] = {
        "episodes": int(replies > 0), "replies": replies, "supervised_tokens_removed": removed}


def parser_rejection_counts(examples):
    """Count each training input once, independent of epochs or rank padding."""
    return {key: sum(example.get("label_provenance", {}).get("parser_rejection_mask", {}).get(key, 0)
                     for example in examples)
            for key in ("episodes", "replies", "supervised_tokens_removed")}
