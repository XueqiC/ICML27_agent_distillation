"""Allowlisted ReAct SFT turns with the exact OFF evaluation generation prefix.

One example teaches one complete assistant turn. Earlier assistant turns are
context only; each reply is supervised once per pass. No context is truncated.
The legacy Stage 1 action example API is retained for its callers.
"""

import argparse
import json
from pathlib import Path

from sed.manifest import authorize_task
from sed.storage import canonical_bytes
from sed.visibility import visible_history, visible_observation, visible_task
from sed.react import build_messages, PROMPT_VERSION, PROMPT_TEMPLATE
from sed.storage import digest, file_hash, write_json
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.manifest import load_frozen, resolve_task
from sed.thinking import split_thinking

TURN_TOKEN = "<turn|>"
TURN_ID = 106
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


def build_action_example(record):
    authorize_task(record["task"], "generation")
    action = record["action_parsed"]
    if type(action) is not str or not action:
        raise ValueError("Training action must be parsed text")
    payload = {"task": visible_task(record["task"]),
               "history": visible_history(record["history"]),
               "visible_context": visible_observation(record["visible_context"])}
    # Construct every output field explicitly: never dict(record) or **record.
    return {"messages": [
                {"role": "user", "content": canonical_bytes(payload).decode(), "supervised": False},
                {"role": "assistant", "content": action, "supervised": True}],
            "label_provenance": {"event_id": record["event_id"],
                                 "source": record["label_source"]}}


def load_tokenizer(snapshot):
    from transformers import AutoTokenizer
    if not Path(snapshot).is_dir():
        raise ValueError("Tokenizer requires an existing local student snapshot")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if (tokenizer.convert_tokens_to_ids(TURN_TOKEN) != TURN_ID
            or tokenizer.encode(TURN_TOKEN, add_special_tokens=False) != [TURN_ID]
            or TURN_ID not in tokenizer.all_special_ids):
        raise ValueError("Expected pinned Gemma <turn|> id 106")
    return tokenizer


def encode_turn(tokenizer, messages, reply, *, max_length):
    # These are exactly the messages and generation settings used by evaluation.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
    if not prompt.endswith(EMPTY_THOUGHT):
        raise ValueError("Gemma OFF generation prefix changed")
    # Completed-chat rendering omits the empty thought prefill. Use it only to
    # obtain the native target serialization, retaining the evaluation prefix.
    completed = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": reply}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prefix_without_thought = prompt[:-len(EMPTY_THOUGHT)]
    if not completed.startswith(prefix_without_thought) or not completed.endswith(TURN_TOKEN + "\n"):
        raise ValueError("Gemma chat-template boundary changed")
    target = completed[len(prefix_without_thought):-1]
    if target != reply.strip() + TURN_TOKEN or not reply.strip():
        raise ValueError("Template transformed or removed the visible teacher reply")
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if any(i in tokenizer.all_special_ids for i in target_ids[:-1]):
        raise ValueError("Visible teacher reply contains a model control token")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt + target, add_special_tokens=False)
    if ids != prompt_ids + target_ids:
        raise ValueError("Tokenizer boundary crosses prompt/assistant supervision")
    if len(ids) > max_length:
        raise ValueError(f"Complete trajectory context has {len(ids)} tokens > max_length={max_length}; no truncation")
    labels = [-100] * len(prompt_ids) + target_ids
    assert ids[-1] == labels[-1] == TURN_ID
    assert all(label == -100 for label in labels[:len(prompt_ids)])
    assert tokenizer.decode(target_ids, skip_special_tokens=False) == target
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids),
            "supervised_tokens": len(target_ids), "prompt_tokens": len(prompt_ids)}


def episode_examples(episode, tokenizer, *, max_length=32768, diagnostic=False):
    if diagnostic:
        # Explicit report-only entry point. Sealed artifacts remain forbidden.
        authorize_task(episode, "diagnosis")
    else:
        authorize_task(episode, "generation")
    if episode.get("schema_version") != 2:
        raise ValueError("Stage 2 ReAct trajectory required; no action-only relabeling")
    history, examples = [], []
    visible = visible_observation(episode["initial"]["visible"])
    for index, step in enumerate(episode["steps"]):
        reply, thinking = split_thinking(step["action_raw"])
        if thinking:
            raise ValueError("Teacher seed contains a hidden thought channel")
        messages = build_messages(history, {}, visible)
        encoded = encode_turn(tokenizer, messages, reply, max_length=max_length)
        encoded["label_provenance"] = {"task_id": episode["task_id"], "turn_index": index,
                "event_id": step["event_id"], "source": "teacher_visible_reply_real_environment",
                "training_eligible": not diagnostic, "action_outcome": step.get("action_outcome"),
                "message_sha256": digest(messages)}
        examples.append(encoded)
        history.append({"visible": visible, "reply": reply})
        visible = visible_observation(step["visible"])
    if not examples:
        raise ValueError("Trajectory has no assistant turns")
    return examples


def mask_report(examples, tokenizer, *, first=3):
    rows = []
    for index, example in enumerate(examples):
        ids, labels, start = example["input_ids"], example["labels"], example["prompt_tokens"]
        assert labels[:start] == [-100] * start
        assert labels[start:] == ids[start:] and labels[-1] == TURN_ID
        assert sum(v != -100 for v in labels) == example["supervised_tokens"]
        assert tokenizer.decode(ids[:start], skip_special_tokens=False).endswith(EMPTY_THOUGHT)
        row = {"index": index, "total_tokens": len(ids), "supervised_tokens": example["supervised_tokens"],
               "masked_tokens": start, "assistant_end_label": labels[-1],
               "assertions": {"all_prompt_system_user_observation_tokens_masked": True,
                              "empty_thought_channel_masked": True,
                              "only_current_assistant_and_native_turn_end_supervised": True},
               "label_provenance": example["label_provenance"]}
        if index < first:
            row["decoded_supervised_span"] = tokenizer.decode(labels[start:], skip_special_tokens=False)
        rows.append(row)
    return {"schema_version": 1, "prompt_version": PROMPT_VERSION, "enable_thinking": False,
            "unit": "complete assistant turn with full masked conversation prefix",
            "no_truncation": True, "supervised_tokens_per_pass": sum(r["supervised_tokens"] for r in rows),
            "examples": rows}


def build_verified_dataset(root, tokenizer, manifest, freeze, *, max_length=32768):
    collection, _, evidence_root = load_verified_collection(root, manifest, freeze)
    examples = []
    for receipt in collection["verified"]:
        episode = checked_episode(evidence_root, receipt)
        examples.extend(episode_examples(episode, tokenizer, max_length=max_length))
    return examples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--verified")
    source.add_argument("--diagnostic-episode", help="Report only; never creates a training dataset")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--manifest", default="data/manifest/alfworld_manifest.json")
    parser.add_argument("--freeze", default="data/manifest/alfworld_split_freeze.json")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    tokenizer = load_tokenizer(args.snapshot)
    manifest, freeze = load_frozen(args.manifest, args.freeze)
    if args.diagnostic_episode:
        episode = json.loads(Path(args.diagnostic_episode).read_text())
        task = resolve_task(manifest, freeze, episode["task_id"])
        authorize_task(task, "diagnosis")
        if episode["split"] != task["split"]:
            raise ValueError("Diagnostic episode split differs from freeze")
        examples = episode_examples(episode, tokenizer, max_length=args.max_length, diagnostic=True)
    else:
        examples = build_verified_dataset(args.verified, tokenizer, manifest, freeze, max_length=args.max_length)
    report = mask_report(examples, tokenizer)
    report.update(training_data_created=False, diagnostic_only=bool(args.diagnostic_episode),
                  snapshot=str(Path(args.snapshot).resolve()), template_sha256=digest(tokenizer.chat_template),
                  source=args.diagnostic_episode or args.verified,
                  source_sha256=file_hash(args.diagnostic_episode or Path(args.verified) / "manifest.json"))
    write_json(args.report, report, exclusive=True)
    print(f"Mask report: {args.report}; {len(examples)} complete turns")


if __name__ == "__main__":
    main()
