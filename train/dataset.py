"""Allowlisted ReAct SFT with complete-episode or complete-turn packing.

Episode packing labels every reply in the last turn's full conversation.
Turn packing retains the exact OFF evaluation prefix for each labeled reply.
Each reply is supervised once per pass. No context is truncated.
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


def encode_turn(tokenizer, messages, reply, *, max_length, as_history=False):
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
    native_prefix_ids = tokenizer.encode(prefix_without_thought, add_special_tokens=False)
    if prompt_ids != native_prefix_ids + tokenizer.encode(EMPTY_THOUGHT, add_special_tokens=False):
        raise ValueError("Tokenizer boundary crosses native history/empty thought")
    turn_forward_tokens = len(prompt_ids) + len(target_ids)
    if as_history:
        # Native completed history has no empty thought prefill. Do not modify
        # the template: locate this reply as serialized in subsequent prefixes.
        prompt = prefix_without_thought
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
            "supervised_tokens": len(target_ids), "prompt_tokens": len(prompt_ids),
            "empty_visible_turns_masked": 0,
            "generation_start": len(native_prefix_ids) if not as_history else None,
            "turn_forward_tokens": turn_forward_tokens}


def _encode_empty_history(tokenizer, messages, *, max_length):
    """Retain native completed history without inventing an empty-turn target."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
    completed = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": ""}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if (not prompt.endswith(EMPTY_THOUGHT)
            or completed != prompt[:-len(EMPTY_THOUGHT)] + TURN_TOKEN + "\n"):
        raise ValueError("Gemma empty assistant history boundary changed")
    ids = tokenizer.encode(completed[:-1], add_special_tokens=False)
    if len(ids) > max_length:
        raise ValueError(f"Complete trajectory context has {len(ids)} tokens > max_length={max_length}; no truncation")
    return {"input_ids": ids, "labels": [-100] * len(ids), "attention_mask": [1] * len(ids),
            "supervised_tokens": 0, "prompt_tokens": len(ids), "turn_forward_tokens": 0,
            "empty_visible_turns_masked": 1}


def episode_examples(episode, tokenizer, *, max_length=32768, diagnostic=False, packing="episode",
                     allow_empty_visible=False):
    if packing not in ("turn", "episode"):
        raise ValueError("Packing must be turn or episode")
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
        final = index == len(episode["steps"]) - 1
        if allow_empty_visible and not reply.strip():
            encoded = _encode_empty_history(tokenizer, messages, max_length=max_length)
        else:
            encoded = encode_turn(tokenizer, messages, reply, max_length=max_length,
                                  as_history=packing == "episode" and not final)
        encoded["packing"] = packing
        encoded["turn_count"] = 1
        encoded["assistant_spans"] = ([{"start": encoded["prompt_tokens"],
                                       "stop": len(encoded["input_ids"]), "turn_index": index}]
                                      if encoded["supervised_tokens"] else [])
        encoded["estimated_forward_tokens"] = {"turn": encoded.pop("turn_forward_tokens"),
                                                "episode": len(encoded["input_ids"]) if final else 0}
        encoded["label_provenance"] = {"task_id": episode["task_id"], "turn_index": index,
                "event_id": step["event_id"], "source": "teacher_visible_reply_real_environment",
                "training_eligible": not diagnostic, "action_outcome": step.get("action_outcome"),
                "message_sha256": digest(messages)}
        if encoded["empty_visible_turns_masked"]:
            encoded["label_provenance"]["empty_visible_reply_masked"] = True
        examples.append(encoded)
        history.append({"visible": visible, "reply": reply})
        visible = visible_observation(step["visible"])
    if not examples:
        raise ValueError("Trajectory has no assistant turns")
    if not any(e["supervised_tokens"] for e in examples):
        raise ValueError("Nothing to score: all assistant turns have empty visible content")
    if packing == "episode":
        # A nonempty final reply keeps the exact evaluation generation prefix.
        # Empty replies and earlier replies use native completed history.
        ids = examples[-1]["input_ids"]
        labels = [-100] * len(ids)
        spans = []
        for example in examples:
            if ids[:len(example["input_ids"])] != example["input_ids"]:
                raise ValueError("Completed assistant history differs from the final turn prefix")
            for span in example["assistant_spans"]:
                start, stop = span["start"], span["stop"]
                labels[start:stop] = example["labels"][start:stop]
                spans.append(span)
        return [{"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids),
                 "supervised_tokens": sum(e["supervised_tokens"] for e in examples),
                 "empty_visible_turns_masked": sum(e["empty_visible_turns_masked"] for e in examples),
                 "prompt_tokens": spans[0]["start"], "packing": packing,
                 "turn_count": len(examples), "assistant_spans": spans,
                 "estimated_forward_tokens": {
                     mode: sum(e["estimated_forward_tokens"][mode] for e in examples)
                     for mode in ("turn", "episode")},
                 "label_provenance": {"task_id": episode["task_id"],
                     "source": "teacher_visible_reply_real_environment",
                     "training_eligible": not diagnostic,
                     "turns": [e["label_provenance"] for e in examples]}}]
    return examples


def mask_report(examples, tokenizer, *, first=3):
    rows = []
    packing = examples[0]["packing"]
    for index, example in enumerate(examples):
        assert example["packing"] == packing
        ids, labels = example["input_ids"], example["labels"]
        spans = example["assistant_spans"]
        expected = [-100] * len(ids)
        for span in spans:
            start, stop = span["start"], span["stop"]
            assert 0 < start < stop <= len(ids) and ids[stop - 1] == TURN_ID
            assert all(v == -100 for v in expected[start:stop])
            expected[start:stop] = ids[start:stop]
            prefix = tokenizer.decode(ids[:start], skip_special_tokens=False)
            if packing == "turn" or span["turn_index"] == example["turn_count"] - 1:
                assert prefix.endswith(EMPTY_THOUGHT)
            else:
                assert prefix.endswith("<|turn>model\n")
        assert labels == expected
        assert sum(v != -100 for v in labels) == example["supervised_tokens"]
        row = {"index": index, "total_tokens": len(ids), "supervised_tokens": example["supervised_tokens"],
               "masked_tokens": len(ids) - example["supervised_tokens"], "assistant_end_label": labels[-1],
               "assistant_spans": spans, "turn_count": example["turn_count"],
               "empty_visible_turns_masked": example["empty_visible_turns_masked"],
               "prefix_mismatch_turn_indices": [s["turn_index"] for s in spans
                   if packing == "episode" and s["turn_index"] != example["turn_count"] - 1],
               "assertions": {"all_system_user_observation_tokens_masked": True,
                              "empty_thought_channel_masked": True,
                              "only_assistant_replies_and_native_turn_ends_supervised": True},
               "label_provenance": example["label_provenance"]}
        if index < first:
            row["decoded_supervised_spans"] = [
                tokenizer.decode(labels[s["start"]:s["stop"]], skip_special_tokens=False) for s in spans]
            if packing == "turn":
                row["decoded_supervised_span"] = row["decoded_supervised_spans"][0] if spans else ""
        rows.append(row)
    return {"schema_version": 2, "prompt_version": PROMPT_VERSION, "enable_thinking": False,
            "packing": packing, "unit": "complete " + packing,
            "no_truncation": True, "supervised_tokens_per_pass": sum(r["supervised_tokens"] for r in rows),
            "estimated_forward_tokens_per_pass": {
                mode: sum(e["estimated_forward_tokens"][mode] for e in examples)
                for mode in ("turn", "episode")},
            "evaluation_prefix_mismatch": {
                "affected_turns": sum(len(r["prefix_mismatch_turn_indices"]) for r in rows),
                "generation_prefix": "<|turn>model\n" + EMPTY_THOUGHT,
                "history_prefix": "<|turn>model\n",
                "description": "Generation appends an empty thought channel before the reply; "
                    "completed history renders <|turn>model\\n<reply><turn|> without it. "
                    "Episode packing supervises all non-final turns in that history format; "
                    "the final reply retains the exact evaluation generation prefix. "
                    "Turn packing retains that prefix for every supervised reply. Template unchanged."},
            "examples": rows}


def build_verified_dataset(root, tokenizer, manifest, freeze, *, max_length=32768, packing="episode"):
    collection, _, evidence_root = load_verified_collection(root, manifest, freeze)
    examples = []
    for receipt in collection["verified"]:
        episode = checked_episode(evidence_root, receipt)
        examples.extend(episode_examples(episode, tokenizer, max_length=max_length, packing=packing))
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
    parser.add_argument("--packing", choices=("turn", "episode"), default="episode")
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
        examples = episode_examples(episode, tokenizer, max_length=args.max_length, diagnostic=True,
                                    packing=args.packing)
    else:
        examples = build_verified_dataset(args.verified, tokenizer, manifest, freeze,
                                          max_length=args.max_length, packing=args.packing)
    report = mask_report(examples, tokenizer)
    report.update(training_data_created=False, diagnostic_only=bool(args.diagnostic_episode),
                  snapshot=str(Path(args.snapshot).resolve()), template_sha256=digest(tokenizer.chat_template),
                  source=args.diagnostic_episode or args.verified,
                  source_sha256=file_hash(args.diagnostic_episode or Path(args.verified) / "manifest.json"))
    write_json(args.report, report, exclusive=True)
    print(f"Mask report: {args.report}; {len(examples)} complete {args.packing} examples; "
          f"estimated forward tokens/pass: {report['estimated_forward_tokens_per_pass']}")


if __name__ == "__main__":
    main()
