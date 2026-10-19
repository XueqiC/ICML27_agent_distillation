"""HAD policy wrapper over the shared H0 runner/exploration machinery."""
import argparse
from functools import partial
import json
from pathlib import Path

from baselines.common.explore import explore_one, run_exploration
from baselines.had.harness import messages, records
from sed.config import load_config
from sed.storage import digest
from students.vllm_student import Student


class HADStudent(Student):
    def __init__(self, *, task_id, rollout=False, round_index=0, **kwargs):
        super().__init__(**kwargs)
        self.task_id, self.rollout, self.round_index = task_id, rollout, round_index
        self.spec.update(harness="had", temperature=1.0 if rollout else 0.0,
                         round_index=round_index, sampling_seed=0)

    def messages(self, history, task, visible_context):
        return messages(history, task, visible_context)

    def request_sampling(self, history, task):
        seed = int(digest([0, self.round_index, self.task_id, len(history)])[:8], 16) if self.rollout else 0
        return {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "seed": seed} if self.rollout else {
            "temperature": 0.0, "seed": seed}

    def generate(self, history, task, visible_context):
        result = super().generate(history, task, visible_context)
        return result._replace(metadata={**result.metadata, "had_records": records(history, visible_context)})


class H0Student(Student):
    def request_sampling(self, history, task):
        return {"temperature": 0, "seed": 0}


def worker(job):
    return explore_one(job, policy_factory=partial(HADStudent, task_id=job[0]["task_id"],
                       rollout=True, round_index=job[1]["had_round"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "tasks", "endpoint", "model-name", "server-provenance", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["server_provenance"] = json.loads(Path(args.server_provenance).read_text())
    config["had_round"] = args.round
    config["had_rollout_sampling"] = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "seed": 0}
    spec = dict(endpoint=args.endpoint, model=args.model_name,
                revision=config["models"]["student"]["revision"], enable_thinking=False, max_tokens=1024)
    summary = run_exploration(config=config, tasks_path=args.tasks, output=args.output, spec=spec,
        decoding="greedy", workers=args.workers, worker_fn=worker,
        inputs=[args.config, args.server_provenance])
    # The shared explorer's decoding argument selects a single reply, not SAG;
    # HADStudent supplies the actual paper temperature to the logged request.
    if summary["training_eligible_episodes"] != summary["episodes"]:
        raise RuntimeError("HAD rollouts incomplete; inspect retained attempts")


if __name__ == "__main__":
    main()
