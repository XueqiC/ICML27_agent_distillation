"""CoA student execution through the shared H0 runner and replay explorer."""
import argparse
from functools import partial
import json
from pathlib import Path

from baselines.coa.protocol import h0_reply, messages
from baselines.common.explore import explore_one, training_tasks
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.storage import JsonlLedger, digest, write_json
from students.vllm_student import Student, context_options


class CoAStudent(Student):
    def __init__(self, *, task_id, sample=0, round_index=0, rollout=False, **kwargs):
        super().__init__(**kwargs)
        self.task_id, self.sample, self.round_index, self.rollout = task_id, sample, round_index, rollout
        self.turns = []
        self.spec.update(method="coa", sample=sample, round_index=round_index, temperature=1.0 if rollout else 0.0)

    def messages(self, history, task, visible_context):
        if len(history) != len(self.turns):
            raise ValueError("CoA student history changed")
        return messages(self.turns, visible_context)

    def request_sampling(self, history, task):
        seed = int(digest([0, self.task_id, self.round_index, self.sample, len(history)])[:8], 16) if self.rollout else 0
        return {"temperature": 1.0 if self.rollout else 0.0, "top_p": 1.0, "top_k": -1, "seed": seed}

    def generate(self, history, task, visible_context):
        result = super().generate(history, task, visible_context)
        bridge, action, outcome = h0_reply(result.raw_text, visible_context["admissible_commands"])
        JsonlLedger(self.log_dir / "coa_bridge.jsonl").append({"call_id": result.metadata["call_id"],
            "coa_reply": result.raw_text, "h0_reply": bridge, "action": action, "outcome": outcome})
        self.turns.append({"visible": visible_context, "reply": result.raw_text})
        return result._replace(raw_text=bridge, parsed_action=action, metadata={**result.metadata,
            "coa_reply": result.raw_text, "action_outcome": outcome,
            "parser_error": "ActionParseError" if action is None else None})


def group_rollouts(config, listing, tasks, output, spec, *, round_index, inputs=(), worker=explore_one):
    """Eight fresh resets per prompt; all samples retained, including all-zero groups."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="coa_student_rollouts", config=config, inputs=inputs,
        details={"round_index": round_index, "samples": 8, "listing_sha256": digest(listing)})
    groups = []
    for occurrence, task in enumerate(tasks):
        rows = []
        for sample in range(8):
            directory = output / "episodes" / f"{occurrence:04d}-{sample}"
            factory = partial(CoAStudent, task_id=task["task_id"], sample=sample,
                              round_index=round_index, rollout=True)
            job = (task, config, directory, spec, listing["parent_seed_ids"][task["task_id"]], inputs)
            rows.append(worker(job, policy_factory=factory))
        groups.append({"task_id": task["task_id"], "samples": rows})
        write_json(output / "groups.json", groups)
    return groups


def checked_groups(root, listing):
    from sed.seed_evidence import artifact_hashes
    from baselines.coa.data import read
    groups = read(Path(root) / "groups.json")
    result = []
    for group in groups:
        if group["task_id"] not in listing["task_ids"] or len(group["samples"]) != 8:
            raise ValueError("RL requires eight samples on support TRAIN")
        episodes = []
        for receipt in group["samples"]:
            directory = Path(receipt["episode_dir"]).resolve()
            if not directory.is_relative_to(Path(root).resolve()) or artifact_hashes(directory) != receipt["episode_hashes"]:
                raise ValueError("Student rollout artifacts changed")
            episode = read(directory / "episode.json")
            if (not receipt["training_eligible"] or not receipt["replay"]["matched"]
                    or receipt["task_id"] != group["task_id"] or episode["task_id"] != group["task_id"]
                    or episode["split"] != "train" or episode.get("sealed")
                    or type(episode["success"]) is not bool or episode["success"] != receipt["won"]
                    or not episode["complete"] or episode["status"] != "ok"
                    or receipt["parent_seed_ids"] != listing["parent_seed_ids"][group["task_id"]]):
                raise ValueError("Unknown/incomplete/out-of-scope RL sample")
            episodes.append(episode)
        result.append((group["task_id"], episodes))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "tasks", "endpoint", "model-name", "server-provenance", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["server_provenance"] = json.loads(Path(args.server_provenance).read_text())
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    listing = json.loads(Path(args.tasks).read_text())
    tasks = training_tasks(listing, manifest, freeze)
    spec = dict(endpoint=args.endpoint, model=args.model_name, revision=config["models"]["student"]["revision"],
                enable_thinking=False, max_tokens=1024, **context_options(config))
    group_rollouts(config, listing, tasks, args.output, spec, round_index=args.round,
        inputs=[args.config, args.tasks, args.server_provenance])
    checked_groups(args.output, listing)


if __name__ == "__main__":
    main()
