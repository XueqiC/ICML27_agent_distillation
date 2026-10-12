"""R03 pool rollout, environment-win filtering, and reusable visible state records."""
import argparse
import json
import os
from pathlib import Path

from data.manifest.build_r03_sets import (dataset_entry, guarded_task, publish_dataset)
from sed.config import load_config
from sed.manifest import load_frozen, verify_task_files
from sed.provenance import write_provenance
from sed.react import build_messages
from sed.runner import _environment, run_episode
from sed.seed_evidence import environment_won
from sed.storage import BlobStore, JsonlLedger, digest, file_hash, write_json
from sed.visibility import visible_history, visible_observation
from students.vllm_student import Student, context_options
from teachers.capped import state_hash


class PoolStudent(Student):
    """Greedy driver plus independent, unexecuted labels from the same student."""
    label_sampling = {"temperature": 1, "top_p": 1, "top_k": -1, "min_p": 0,
                      "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0}

    def __init__(self, *, label_samples=0, **kwargs):
        if type(label_samples) is not int or label_samples < 0:
            raise ValueError("label_samples must be a nonnegative integer")
        super().__init__(**kwargs)
        if self.spec["chat_template_kwargs"]["enable_thinking"]:
            raise ValueError("Pool student requires thinking off")
        self.label_samples, self._label_mode = label_samples, False
        self.spec.update(label_samples=label_samples, label_sampling=dict(self.label_sampling))

    def sampling(self):
        return dict(self.label_sampling) if self._label_mode else super().sampling()

    def generate(self, history, task, visible_context):
        greedy = super().generate(history, task, visible_context)
        samples = []
        try:
            self._label_mode = True
            for index in range(self.label_samples):
                try:
                    sample = super().generate(history, task, visible_context)
                    raw = BlobStore(self.log_dir / "blobs").get(sample.metadata["raw_response_ref"])
                    samples.append({"sample_index": index, "status": "completed",
                        "reply": sample.raw_text, "action": sample.parsed_action,
                        "raw_content": raw["content"], "metadata": sample.metadata})
                except Exception as exc:
                    # The client's durable ledger retains attempts and sanitized errors.
                    # A failed diagnostic call never changes the greedy transition.
                    samples.append({"sample_index": index, "status": "failed",
                        "error_type": type(exc).__name__, "metadata": getattr(exc, "metadata", {})})
        finally:
            self._label_mode = False
        return greedy._replace(metadata={**greedy.metadata, "label_samples": samples})


def pool_tasks(path, manifest, freeze, subset="all"):
    pool = json.loads(Path(path).read_text())
    ids = pool["task_ids"]
    if (pool.get("name") != "r03_pool200" or pool["list_sha256"] != digest(ids)
            or len(ids) != len(set(ids)) or pool["manifest_sha256"] != digest(manifest)
            or pool["split_freeze_sha256"] != digest(freeze)):
        raise ValueError("Invalid frozen R03 pool")
    tasks = [guarded_task(manifest, freeze, t) for t in ids]
    if pool["group_ids"] != [t["group_id"] for t in tasks]:
        raise ValueError("Pool group IDs differ from freeze")
    if subset != "all":
        selected = pool["subsets"][subset]
        if not set(selected) <= set(ids) or len(set(selected)) != len(selected):
            raise ValueError("Invalid pool subset")
        tasks = [guarded_task(manifest, freeze, t) for t in selected]
    return pool, tasks


def episode_record(directory, task, *, relative_to):
    """Reconstruct EXACT visible contexts from run_episode's audited evidence."""
    directory = Path(directory)
    ep = json.loads((directory / "episode.json").read_text())
    blobs = BlobStore(directory / "blobs")
    history, states = [], []
    visible = visible_observation(ep["initial"]["visible"])
    for i, step in enumerate(ep["steps"]):
        verification = blobs.get(step["verification_ref"])
        states.append({"episode_id": ep["episode_id"], "step": i,
            "task_id": task["task_id"], "task_type": task["task_type"], "group_id": task["group_id"],
            "state_hash": state_hash(history, task, visible),
            "history": visible_history(history), "visible_context": visible,
            "messages": build_messages(history, task, visible),
            "student_reply": step["action_raw"], "student_action": step["action_parsed"],
            "reward": step["reward"], "success": verification["success"],
            "environment_steps": step["environment_steps"], "action_outcome": step["action_outcome"],
            "terminated": step["terminated"], "truncated": step["truncated"],
            "next_visible": visible_observation(step["visible"]), "event_id": step["event_id"]})
        metadata = step.get("policy_metadata", {})
        if "label_samples" in metadata:
            states[-1]["label_samples"] = metadata["label_samples"]
        history.append({"visible": visible, "reply": step["action_raw"]})
        visible = visible_observation(step["visible"])
    completed_ids = {ep["initial"]["event_id"], *(s["event_id"] for s in ep["steps"])}
    failed_attempts = []
    for line in (directory / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event_id"] in completed_ids:
            continue
        before = visible_observation(blobs.get(event["visible_context_ref"]))
        prefix = visible_history(blobs.get(event["history_ref"]))
        verification = blobs.get(event["verification_ref"])
        failed_attempts.append({"event_id": event["event_id"], "step": len(prefix),
            "history": prefix, "visible_context": before,
            "messages": build_messages(prefix, task, before),
            "student_reply": event["action_raw"], "student_action": event["action_parsed"],
            "reward": verification["reward"] if event["feedback_source"] == "real_environment" else None,
            "success": verification["success"], "status": event["status"],
            "error_type": event["error_type"], "environment_steps_attempted": event["environment_steps"],
            "environment_steps_confirmed": verification["environment_steps_confirmed"]})
        metadata = verification.get("policy_metadata", {})
        if "label_samples" in metadata:
            failed_attempts[-1]["label_samples"] = metadata["label_samples"]
    return {"schema_version": 1, "format": "r03_rollout_v1", "episode_id": ep["episode_id"],
        "task_id": task["task_id"], "task_type": task["task_type"], "group_id": task["group_id"],
        "split": ep["split"], "sealed": ep["sealed"], "seed": ep["seed"],
        "environment_revision": ep["environment_revision"], "max_episode_steps": ep["max_episode_steps"],
        "complete": ep["complete"], "status": ep["status"], "success": ep.get("success", "unknown"),
        "metrics": ep["metrics"], "states": states, "failed_attempts": failed_attempts,
        "episode_dir": os.path.relpath(directory, relative_to),
        "episode_sha256": file_hash(directory / "episode.json"),
        "events_sha256": file_hash(directory / "events.jsonl"),
        "provenance_sha256": file_hash(directory / "provenance.json")}


def read_rollouts(paths, manifest, freeze, pool_path, *, subset="all"):
    _, tasks = pool_tasks(pool_path, manifest, freeze, subset)
    allowed = {t["task_id"]: t for t in tasks}
    result, seen = [], set()
    for filename in paths:
        path = Path(filename)
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["task_id"] not in allowed:
                raise ValueError("Rollout is outside the frozen query pool/subset")
            if record["episode_id"] in seen:
                raise ValueError("Duplicate rollout episode")
            seen.add(record["episode_id"])
            directory = path.parent / record["episode_dir"]
            expected = episode_record(directory, allowed[record["task_id"]], relative_to=path.parent)
            if record != expected:
                raise ValueError("Rollout differs from original environment evidence")
            if record["split"] != "train" or record["sealed"]:
                raise ValueError("Rollout is not an unsealed training game")
            result.append((record, directory))
    return result


def run_pool(tasks, config, output, *, env_factory, policy_factory, inputs=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="r03_rollout_pool", config=config, inputs=inputs)
    ledger = JsonlLedger(output / "rollouts.jsonl")
    failures = 0
    for i, task in enumerate(tasks):
        from sed.manifest import authorize_task
        authorize_task(task, "generation")
        directory = output / "episodes" / f"{i:03d}-{digest(task['task_id'])[:12]}"
        try:
            run_episode(env_factory(task), policy_factory(directory), task, config, directory,
                policy_condition="r03_p10_lora_greedy_thinking_off", purpose="generation", input_paths=inputs)
        except Exception as exc:
            failures += 1
            JsonlLedger(output / "failures.jsonl").append({"task_id": task["task_id"],
                                                        "error_type": type(exc).__name__})
        if (directory / "episode.json").exists():
            ep = json.loads((directory / "episode.json").read_text())
            if "initial" in ep:
                record = episode_record(directory, task, relative_to=output)
                ledger.append(record, unique_key="episode_id")
                per_episode = output / "episode_jsonl" / (digest(task["task_id"]) + ".jsonl")
                # Evidence paths are relative to the JSONL containing them.
                JsonlLedger(per_episode).append(episode_record(directory, task, relative_to=per_episode.parent))
    result = {"games": len(tasks), "failed_episodes": failures, "rollouts": str(ledger.path)}
    write_json(output / "summary.json", result, exclusive=True)
    return result


def filter_successes(records, output, manifest, freeze):
    output = Path(output)
    entries = []
    for record, directory in records:
        task = guarded_task(manifest, freeze, record["task_id"])
        ep = json.loads((directory / "episode.json").read_text())
        won = environment_won(directory, ep)
        JsonlLedger(output / "selection.jsonl").append({"episode_id": ep["episode_id"],
            "task_id": task["task_id"], "accepted": won, "success": ep.get("success", "unknown"),
            "reason": "real_environment_win" if won else "not_a_verified_environment_win",
            "source_episode_sha256": file_hash(directory / "episode.json")})
        if won:
            entry = dataset_entry(output, ep, source="student_real_environment_win")
            entry["source_episode_sha256"] = file_hash(directory / "episode.json")
            entries.append(entry)
    return publish_dataset(output, entries, manifest, freeze, source="R1_P3",
        verification="real_environment", trusted_executed_trajectories=len(entries), new_teacher_calls=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "filter"))
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--pool", default="data/manifest/r03_pool200.json")
    p.add_argument("--subset", choices=("all", "R1_P2", "R1_P3", "R2"), default="R2")
    p.add_argument("--games", help="Optional JSON list, restricted to the chosen subset")
    p.add_argument("--rollouts", nargs="+")
    p.add_argument("--endpoint")
    p.add_argument("--model-name")
    p.add_argument("--server-provenance")
    p.add_argument("--lora-check")
    p.add_argument("--label-samples", type=int, default=0,
                   help="Extra unexecuted temperature-1 student actions per visited state (default: 0)")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if args.label_samples < 0 or (args.mode == "filter" and args.label_samples):
        p.error("--label-samples must be nonnegative and is only used by run")
    config = load_config(args.config)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    inputs = [args.config, args.pool, config["paths"]["manifest"], config["paths"]["split_freeze"]]
    pool, tasks = pool_tasks(args.pool, manifest, freeze, args.subset)
    if args.mode == "filter":
        if not args.rollouts:
            p.error("filter requires --rollouts")
        records = read_rollouts(args.rollouts, manifest, freeze, args.pool)
        chosen = set(pool["subsets"]["R1_P3"])
        records = [(r, d) for r, d in records if r["task_id"] in chosen]
        write_provenance(args.output, command="r03_filter_successes", config=config,
                         inputs=inputs + args.rollouts)
        result = filter_successes(records, args.output, manifest, freeze)
    else:
        if not all((args.endpoint, args.model_name, args.server_provenance, args.lora_check)):
            p.error("run requires endpoint, model-name, server-provenance and lora-check")
        if not config["environment"]["data_root"] or config["environment"]["max_episode_steps"] != 40:
            p.error("Require installed ALFWORLD_DATA_ROOT and evaluation cap of 40")
        from tools.check_lora_serving import validate_report
        server = json.loads(Path(args.server_provenance).read_text())
        report = json.loads(Path(args.lora_check).read_text())
        validate_report(report, server, args.model_name, args.endpoint)
        checked = report["identity"]["adapter"]
        model = config["models"]["student"]
        if checked["base_model"] != model["requested_repo"] or checked["revision"] != model["revision"]:
            raise ValueError("Adapter differs from configured student")
        served = next(a for a in server["config"]["server"]["lora_adapters"]
                      if a["name"] == args.model_name)
        if served.get("checkpoint_metadata", {}).get("supervised_tokens") != 104070:
            raise ValueError("R03 rollouts require the p10 adapter at 104070 supervised tokens")
        config["server_provenance"], config["lora_equivalence"] = server, report
        inputs += [args.server_provenance, args.lora_check]
        if args.games:
            ids = json.loads(Path(args.games).read_text())
            allowed = {t["task_id"]: t for t in tasks}
            if len(ids) != len(set(ids)) or not set(ids) <= allowed.keys():
                raise ValueError("Requested games must be a unique pool subset")
            tasks = [allowed[t] for t in ids]
            inputs.append(args.games)
        config["r03_rollout"] = {"subset": args.subset, "task_ids": [t["task_id"] for t in tasks],
            "checkpoint": "p10", "endpoint": args.endpoint, "model": args.model_name,
            "temperature": 0, "enable_thinking": False, "max_tokens": 1024,
            "label_samples": args.label_samples, "label_sampling": dict(PoolStudent.label_sampling)}
        def environment(task):
            verify_task_files(config["environment"]["data_root"], task)
            return _environment(config, task)
        def policy(directory):
            return PoolStudent(label_samples=args.label_samples,
                endpoint=args.endpoint, model=args.model_name, revision=model["revision"],
                enable_thinking=False, max_tokens=1024, log_dir=directory / "student_calls",
                **context_options(config))
        result = run_pool(tasks, config, args.output, env_factory=environment,
                          policy_factory=policy, inputs=inputs)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
