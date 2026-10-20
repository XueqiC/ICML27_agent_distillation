"""P4: one visible-thought rewrite request per K20 trajectory; preserve action bytes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from teachers.stage import teacher_stage, count, require_completed
import re

from data.manifest.build_r03_sets import dataset_entry, guarded_task, publish_dataset
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.react import parse_action
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import JsonlLedger, canonical_bytes, digest, write_json
from sed.thinking import split_thinking
from sed.visibility import visible_history, visible_observation
from teachers.capped import from_config
from teachers.openai_teacher import Teacher

REWRITE_PROMPT = (
    "Rewrite the brief, visible Thought text in each demonstration turn for clarity. "
    "Do not supply hidden reasoning. Copy each supplied action string EXACTLY, including whitespace. "
    'Return only a JSON object {"turns":[{"thought":"brief visible explanation","action":"exact command"}]}, '
    "one entry per input turn, in the same order. Do not add observations, commands or turns."
)


class RewriteTeacher(Teacher):
    def __init__(self, **kwargs):
        super().__init__(prompt_template=REWRITE_PROMPT, **kwargs)

    def prompt(self, history, task, visible_context):
        # history is explicitly assembled from demonstration-visible fields.
        turns = [{"visible": t["visible"], "reply": t["reply"], "action": parse_action(t["reply"])}
                 for t in visible_history(history)]
        return [{"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": canonical_bytes({"turns": turns}).decode()}]

    def parse_response(self, raw):
        try:
            value = json.loads(raw)
            if set(value) != {"turns"} or not isinstance(value["turns"], list):
                raise ValueError
            for turn in value["turns"]:
                if set(turn) != {"thought", "action"} or any(type(v) is not str for v in turn.values()):
                    raise ValueError
            return "structured_rewrite", None
        except (ValueError, TypeError):
            return None, "RewriteFormatError"


def rewrite_history(episode):
    visible, history = visible_observation(episode["initial"]["visible"]), []
    for step in episode["steps"]:
        reply, hidden = split_thinking(step["action_raw"])
        if hidden:
            raise ValueError("Rewrite requires visible-only demonstration replies")
        history.append({"visible": visible, "reply": reply})
        visible = visible_observation(step["visible"])
    return history, visible


def apply_rewrite(episode, raw):
    value = json.loads(raw)
    if set(value) != {"turns"} or len(value["turns"]) != len(episode["steps"]):
        raise ValueError("Rewrite changed turn count")
    rewritten = deepcopy(episode)
    for original, step, turn in zip(episode["steps"], rewritten["steps"], value["turns"]):
        if set(turn) != {"thought", "action"} or any(type(v) is not str for v in turn.values()):
            raise ValueError("Malformed rewrite turn")
        if turn["action"].encode("utf-8") != original["action_parsed"].encode("utf-8"):
            raise ValueError("Rewrite changed action bytes")
        if parse_action(original["action_raw"]) != original["action_parsed"]:
            raise ValueError("Original action parser mismatch")
        thought = turn["thought"]
        if (not thought.strip() or re.search(r"^\s*Action\s*:", thought, re.MULTILINE)
                or split_thinking(thought)[1]
                or any(token in thought for token in ("<|", "<turn|>", "<eos>", "<channel|>"))):
            raise ValueError("Rewrite thought contains actions/control channels or is empty")
        action_lines = re.findall(r"^[ \t]*Action[ \t]*:[^\r\n]*", original["action_raw"], re.MULTILINE)
        if len(action_lines) != 1:
            raise ValueError("Expected exactly one original Action line")
        # Preserve original Action LINE bytes, not a reserialized command.
        step["action_raw"] = "Thought: " + thought + "\n" + action_lines[0]
        if parse_action(step["action_raw"]) != original["action_parsed"]:
            raise ValueError("Rewritten action parser mismatch")
        step["policy_metadata"] = {"source": "teacher_visible_thought_rewrite"}
    rewritten["episode_id"] = "rewrite-" + digest([episode["episode_id"], raw])
    rewritten["label_verification"] = "actions_identical_to_verified_parent; rewritten_thought_not_verified"
    rewritten["parent_episode_id"] = episode["episode_id"]
    return rewritten


@teacher_stage("output", name="collect_rewrites")
def collect(episodes, teacher, output, manifest, freeze):
    entries, rejected, planned = [], 0, []
    output = Path(output)
    for episode in episodes:
        task = guarded_task(manifest, freeze, episode["task_id"])
        history, visible = rewrite_history(episode)
        audit = {"task_id": task["task_id"], "parent_seed_ids": [episode["episode_id"]],
                 "source": "R1_P4", "new_environment_steps": 0}
        try:
            result = teacher.generate(history, task, visible)
        except Exception as exc:
            JsonlLedger(output / "collection.jsonl").append({**audit, "status": "stopped",
                                                           "error_type": type(exc).__name__})
            raise
        if not teacher.dry_run:
            require_completed(result)
        if teacher.dry_run:
            planned.append(result.metadata)
            continue
        try:
            rewritten = apply_rewrite(episode, result.raw_text)
        except (ValueError, TypeError, KeyError, AttributeError):
            rejected += 1
            count("rejections")
            JsonlLedger(output / "collection.jsonl").append({**audit, "accepted": False,
                "reason": "rewrite_format_or_action_invariance_failed", "raw_output": result.raw_text,
                "teacher_metadata": result.metadata})
            continue
        JsonlLedger(output / "collection.jsonl").append({**audit, "accepted": True,
            "raw_output": result.raw_text, "teacher_metadata": result.metadata,
            "verification": "parent_actions_byte_identical; thought_verification_unknown"})
        entries.append(dataset_entry(output, rewritten, source="teacher_visible_thought_rewrite",
                                     parent_seed_ids=[episode["episode_id"]]))
    if teacher.dry_run:
        return {"dry_run": True, "trajectories": len(episodes),
                "would_call": sum(m["would_call"] for m in planned),
                "within_cap": all(m["within_cap"] for m in planned),
                "halted": any(m["halted"] for m in planned)}
    return publish_dataset(output, entries, manifest, freeze, source="R1_P4", rejected=rejected,
        transport_failures=0, rejections=rejected, new_environment_steps=0, new_trusted_trajectories=0,
        verification="unchanged executed parent actions; thought truth unknown")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--k20", default="data/verified/seeds_k20_s0")
    p.add_argument("--cap", type=int, default=40)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--cache", default="data/raw/r03_teacher_cache")
    p.add_argument("--usage-ledger", default="data/raw/r03_teacher_usage.jsonl")
    p.add_argument("--run-id", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if not 0 <= args.cap <= 40:
        p.error("P4 cap must be at most 40")
    config = load_config(args.config)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, selection, evidence = load_verified_collection(args.k20, manifest, freeze)
    if collection["K"] != 20 or selection["seed"] != 0:
        raise ValueError("P4 requires K20 seed 0")
    episodes = [checked_episode(evidence, r) for r in collection["verified"]]
    sampling = {**config["models"]["teacher"]["sampling"], "max_output_tokens": args.max_output_tokens}
    teacher = from_config(config, output=args.output, cache_dir=args.cache, usage_ledger=args.usage_ledger,
        run_id=args.run_id, cap=args.cap, dry_run=args.dry_run, teacher_class=RewriteTeacher, sampling=sampling)
    if not args.dry_run:
        config["r03_collector"] = {"run_id": args.run_id, "cap": args.cap, "sampling": sampling,
                                  "cache": args.cache, "usage_ledger": args.usage_ledger}
        write_provenance(args.output, command="r03_collect_rewrites", config=config,
            inputs=[args.config, str(Path(args.k20) / "manifest.json"),
                    config["paths"]["manifest"], config["paths"]["split_freeze"]])
    result = collect(episodes, teacher, args.output, manifest, freeze)
    if args.dry_run:
        write_json(Path(args.output) / "dry_run.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
