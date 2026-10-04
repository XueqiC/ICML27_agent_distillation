"""Run exactly one frozen-task episode, preserving every transition for replay."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import uuid

from harness.alfworld import Environment, ReplayMismatch
from sed.actions import ActionParseError, parse_action
from sed.config import load_config
from sed.events import Event, EventWriter
from sed.manifest import authorize_task, load_frozen, resolve_task, verify_task_files
from sed.provenance import versions, write_provenance
from sed.storage import digest, write_json
from sed.visibility import visible_history, visible_observation, visible_task
from teachers.openai_teacher import Teacher, TeacherResult


class ScriptedPolicy:
    """A fixed action sequence, cycling after its end (default: look)."""
    def __init__(self, actions=None):
        self.actions = list(actions or ["look"])
        if not self.actions or any(type(a) is not str for a in self.actions):
            raise ValueError("Script requires text commands")
        self.index = 0

    def generate(self, history, task, visible_context):
        raw = self.actions[self.index % len(self.actions)]
        self.index += 1
        try:
            parsed, error = parse_action(raw), None
        except ActionParseError:
            parsed, error = None, "ActionParseError"
        return TeacherResult(raw, parsed, {"policy": "scripted", "parser_error": error,
                                         "billed_usage": {"input_tokens": 0, "output_tokens": 0}})


def run_episode(env, policy, task, config, output_dir, *, policy_condition,
                purpose="generation", parent_seed_ids=(), input_paths=()):
    authorize_task(task, purpose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id, episode_id = str(uuid.uuid4()), str(uuid.uuid4())
    if isinstance(policy, Teacher):
        if task["split"] == "test":
            raise ValueError("No teacher generation on sealed tasks")
        policy.run_id = run_id
        policy.purpose = purpose
    task = dict(task, parent_seed_ids=list(parent_seed_ids))
    provenance = write_provenance(output_dir, command="episode", config=config,
                                  inputs=input_paths, run_id=run_id,
                                  details={"task_id": task["task_id"], "split": task["split"],
                                           "purpose": purpose, "policy_condition": policy_condition,
                                           "scripted_actions": getattr(policy, "actions", None),
                                           "seed": env.seed, "max_episode_steps": env.max_episode_steps,
                                           "parent_seed_ids": list(parent_seed_ids)})
    writer = EventWriter(output_dir)
    environment_revision = digest({"dependencies": versions(), "game_sha256": task.get("game_sha256"),
                                   "code_sha256": provenance["code_sha256"],
                                   "seed": env.seed, "max_episode_steps": env.max_episode_steps})
    history, steps = [], []
    visible = {"observation": "", "admissible_commands": []}
    episode = {"schema_version": 1, "run_id": run_id, "episode_id": episode_id,
               "task_id": task["task_id"], "split": task["split"], "sealed": task.get("sealed", False),
               "game_sha256": task.get("game_sha256"), "environment_revision": environment_revision,
               "seed": env.seed, "max_episode_steps": env.max_episode_steps,
               "complete": False, "status": "started", "steps": steps}
    write_json(output_dir / "episode.json", episode)
    counter = 0

    def record(before, result, raw, action, metadata, elapsed, *, error=None, env_steps=0):
        nonlocal counter
        observation, reward, terminated, truncated, audit = result
        event_id = f"{episode_id}:{counter:04d}"
        counter += 1
        verification = {"audit": audit, "reward": reward, "teacher_metadata": metadata,
                        "executable": "unknown" if error == "step_error" else error is None,
                        "success": audit.get("won", "unknown"),
                        "fact_source": "real_environment" if error is None else "unknown",
                        "environment_steps_attempted": env_steps,
                        "environment_steps_confirmed": env_steps if error is None else "unknown",
                        "failure_reason": error}
        usage = metadata.get("billed_usage") or {}
        event = Event(
            run_id=run_id, episode_id=episode_id, event_id=event_id, split=task["split"],
            task_family=task.get("task_type", "alfworld"), task_id=task["task_id"],
            parent_seed_ids=list(parent_seed_ids), mechanism_id="audit_only",
            student_revision=config["models"]["student"]["revision"],
            teacher_provider_id=(config["models"]["teacher"]["provider_model_id"]
                                 if policy_condition == "teacher" else None),
            environment_revision=environment_revision, policy_condition=policy_condition,
            history_ref=writer.blobs.put(visible_history(history)),
            visible_context_ref=writer.blobs.put(visible_observation(before)),
            action_raw=raw, action_parsed=action,
            observation_ref=(writer.blobs.put(visible_observation(observation)) if error is None else None),
            verification_ref=writer.blobs.put(verification),
            feedback_source="real_environment" if error is None else "none",
            new_environment_information="unknown_until_audit",
            teacher_tokens_in=usage.get("input_tokens", 0), teacher_tokens_out=usage.get("output_tokens", 0),
            environment_steps=env_steps, latency_seconds=elapsed,
            terminated=terminated, truncated=truncated,
            status="ok" if error is None else "error", error_type=error)
        writer.append(event)
        return {"event_id": event_id, "action_raw": raw, "action_parsed": action,
                "visible": visible_observation(observation), "reward": reward,
                "terminated": terminated, "truncated": truncated,
                "verification_ref": event.verification_ref, "audit_sha256": digest(audit)}

    phase, raw, action, metadata = "reset", None, None, {}
    started = time.monotonic()
    try:
        result = env.reset(task["task_id"])
        episode["initial"] = record(visible, result, None, None, {}, time.monotonic() - started)
        visible, _, terminated, truncated, _ = result
        episode["game_sha256"] = env.game_hash
        write_json(output_dir / "episode.json", episode)
        while not (terminated or truncated):
            phase, raw, action, metadata = "policy", None, None, {}
            started = time.monotonic()
            # Policy never receives task filenames, split labels, hidden audit,
            # verification, reward, or arbitrary event fields as prompt material.
            if isinstance(policy, Teacher):
                result_policy = policy.generate(visible_history(history), task, visible_observation(visible))
            else:
                result_policy = policy.generate(visible_history(history), visible_task(task),
                                                visible_observation(visible))
            raw, action, metadata = result_policy
            if action is None:
                raise ActionParseError("Policy output could not be parsed")
            if parse_action(raw) != action:
                raise ActionParseError("Policy result disagrees with shared parser")
            phase = "step"
            before = deepcopy(visible)
            result = env.step(action)
            step = record(before, result, raw, action, metadata, time.monotonic() - started,
                          env_steps=1)
            steps.append(step)
            history.append({"visible": before, "action": action})
            visible, _, terminated, truncated, audit = result
            write_json(output_dir / "episode.json", episode)
        episode.update(complete=True, status="ok", terminated=terminated, truncated=truncated,
                       success=bool(result[4].get("won", False)))
        write_json(output_dir / "episode.json", episode)
        return episode
    except Exception as exc:
        error = "ActionParseError" if isinstance(exc, ActionParseError) else f"{phase}_error"
        failed = (visible, 0.0, False, False,
                  {"phase": phase, "error_type": type(exc).__name__, "won": "unknown"})
        record(visible, failed, raw, action, metadata, time.monotonic() - started,
               error=error, env_steps=1 if phase == "step" else 0)
        episode.update(status="error", error_type=error, failed_action=action, complete=False)
        write_json(output_dir / "episode.json", episode)
        raise RuntimeError(f"Episode failed ({error}); inspect sanitized run artifacts") from None
    finally:
        env.close()


def replay_episode(env, episode):
    if episode.get("sealed") or episode["split"] == "test":
        raise ValueError("Sealed episodes cannot be used for diagnostic replay")
    if env.seed != episode["seed"] or env.max_episode_steps != episode["max_episode_steps"]:
        raise ReplayMismatch("Replay configuration differs")
    try:
        result = env.reset(episode["task_id"])
        if env.game_hash != episode["game_sha256"]:
            raise ReplayMismatch("Game content differs")
        for index, expected in enumerate([episode["initial"], *episode["steps"]]):
            if index:
                result = env.step(expected["action_parsed"])
            visible, reward, terminated, truncated, audit = result
            if (visible_observation(visible) != expected["visible"] or reward != expected["reward"] or
                    terminated != expected["terminated"] or truncated != expected["truncated"] or
                    digest(audit) != expected["audit_sha256"]):
                raise ReplayMismatch(f"Episode diverges at step {index}")
        return {"matched": True, "steps_replayed": len(episode["steps"]),
                "complete_episode": episode["complete"]}
    finally:
        env.close()


def _args(description, argv, replay=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output", required=True)
    if replay:
        parser.add_argument("--episode", required=True)
    else:
        parser.add_argument("--task-id", required=True)
        parser.add_argument("--policy", choices=("scripted", "teacher"), default="scripted")
        parser.add_argument("--purpose", choices=("generation", "diagnosis", "evaluation"), default="generation")
        parser.add_argument("--actions", help='JSON file with a list of scripted actions; defaults to ["look"]')
        parser.add_argument("--parent-seed-id", action="append", default=[])
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.data_root:
        config["environment"]["data_root"] = args.data_root
    if not config["environment"]["data_root"]:
        parser.error("Set ALFWORLD_DATA_ROOT or --data-root")
    return args, config


def _environment(config, task):
    return Environment(config["environment"]["data_root"], split=task["split"],
                       seed=config["environment"]["seed"],
                       max_episode_steps=config["environment"]["max_episode_steps"])


def main(argv=None):
    args, config = _args(__doc__, argv)
    inputs = [args.config, config["paths"]["manifest"], config["paths"]["split_freeze"]]
    manifest, freeze = load_frozen(*inputs[1:])
    task = resolve_task(manifest, freeze, args.task_id)
    authorize_task(task, args.purpose)
    verify_task_files(config["environment"]["data_root"], task)
    if args.policy == "teacher":
        spec = config["models"]["teacher"]
        policy = Teacher(cache_dir=config["paths"]["teacher_cache"],
                         usage_ledger=config["paths"]["teacher_usage"], run_id="pending",
                         provider_model_id=spec["provider_model_id"], sampling=spec["sampling"],
                         max_attempts=spec["max_attempts"], timeout_seconds=spec["timeout_seconds"],
                         secrets_file=config["paths"]["secrets_file"], purpose=args.purpose)
    else:
        if args.actions:
            inputs.append(args.actions)
        policy = ScriptedPolicy(json.loads(Path(args.actions).read_text()) if args.actions else None)
    episode = run_episode(_environment(config, task), policy, task, config, args.output,
                          policy_condition=args.policy, purpose=args.purpose,
                          parent_seed_ids=args.parent_seed_id, input_paths=inputs)
    print(json.dumps({"run_id": episode["run_id"], "steps": len(episode["steps"]),
                      "success": episode["success"], "terminated": episode["terminated"],
                      "truncated": episode["truncated"], "output": args.output}))


def replay_main(argv=None):
    args, config = _args("Replay an episode and verify every observation", argv, replay=True)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    episode = json.loads(Path(args.episode).read_text())
    task = resolve_task(manifest, freeze, episode["task_id"])
    authorize_task(task, "replay")
    verify_task_files(config["environment"]["data_root"], task)
    write_provenance(args.output, command="replay", config=config,
                     inputs=[args.config, args.episode, config["paths"]["manifest"],
                             config["paths"]["split_freeze"]])
    result = replay_episode(_environment(config, task), episode)
    write_json(Path(args.output) / "result.json", result, exclusive=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
