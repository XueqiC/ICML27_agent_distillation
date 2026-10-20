"""Run exactly one frozen-task episode, preserving every transition for replay."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import uuid

from harness.alfworld_env import Environment, ReplayMismatch
from sed.react import ActionParseError, parse_action, action_outcome, PARSER_FEEDBACK
from sed.config import load_config
from sed.events import Event, EventWriter
from sed.manifest import authorize_task, load_frozen, resolve_task, verify_task_files
from sed.provenance import versions, write_provenance
from sed.storage import digest, write_json
from sed.visibility import visible_history, visible_observation, visible_task
from students.vllm_student import ContextOverflow
from teachers.openai_teacher import Teacher, TeacherResult, TeacherCallError, rate_limit_options


class ScriptedPolicy:
    """A fixed action sequence, cycling after its end (default: look)."""
    def __init__(self, actions=None):
        self.actions = list(actions or ["look"])
        if not self.actions or any(type(a) is not str for a in self.actions):
            raise ValueError("Script requires text commands")
        self.index = 0

    def generate(self, history, task, visible_context):
        command = self.actions[self.index % len(self.actions)]
        raw = "Action: " + command if "\n" not in command else command
        self.index += 1
        try:
            parsed, error = parse_action(raw), None
        except ActionParseError:
            parsed, error = None, "ActionParseError"
        return TeacherResult(raw, parsed, {"policy": "scripted", "parser_error": error,
                                         "billed_usage": {"input_tokens": 0, "output_tokens": 0}})


def episode_metrics(episode):
    steps = episode["steps"]
    calls = episode.get("policy_calls", [s.get("policy_metadata", {}) for s in steps])
    outcomes = episode.get("action_outcomes", [s.get("action_outcome") for s in steps])
    def total(key):
        values = [m.get(key) for m in calls]
        return {"known_total": sum(v for v in values if v is not None),
                "unknown_calls": sum(v is None for v in values)}
    return {"success": episode.get("success", "unknown"),
            "steps": episode.get("policy_turns_attempted", len(steps)),
            "environment_steps": sum(s.get("environment_steps", 1) for s in steps) +
                                 int(episode.get("error_type") == "step_error"),
            "environment_steps_confirmed": sum(s.get("environment_steps", 1) for s in steps),
            "environment_steps_unknown": int(episode.get("error_type") == "step_error"),
            "parser_errors": outcomes.count("parser_error"),
            "inadmissible_actions": outcomes.count("inadmissible"),
            "legal_actions": outcomes.count("legal"),
            "prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
            "thinking_tokens": total("thinking_tokens"), "thinking_chars": total("thinking_chars"),
            "policy_latency_seconds": sum(m.get("latency_seconds", 0) for m in calls),
            "wall_time_seconds": episode.get("wall_time_seconds", 0),
            "context_overflow_episodes": int(episode.get("termination_reason") == "context_overflow"),
            "failed_policy_calls": int(episode.get("error_type") == "policy_error")}


def run_episode(env, policy, task, config, output_dir, *, policy_condition,
                purpose="generation", parent_seed_ids=(), input_paths=()):
    authorize_task(task, purpose)
    if isinstance(policy, Teacher) and (task["split"] == "test" or task.get("sealed")):
        raise ValueError("No teacher generation on sealed tasks")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id, episode_id = str(uuid.uuid4()), str(uuid.uuid4())
    if isinstance(policy, Teacher):
        policy.run_id, policy.purpose = run_id, purpose
    task = dict(task, parent_seed_ids=list(parent_seed_ids))
    provenance = write_provenance(output_dir, command="episode", config=config,
                                  inputs=input_paths, run_id=run_id,
                                  details={"task_id": task["task_id"], "split": task["split"],
                                           "purpose": purpose, "policy_condition": policy_condition,
                                           "policy_spec": getattr(policy, "spec", None),
                                           "scripted_actions": getattr(policy, "actions", None),
                                           "seed": env.seed, "max_episode_steps": env.max_episode_steps,
                                           "parent_seed_ids": list(parent_seed_ids)})
    writer = EventWriter(output_dir)
    environment_revision = digest({"dependencies": versions(), "game_sha256": task.get("game_sha256"),
                                   "code_sha256": provenance["code_sha256"],
                                   "seed": env.seed, "max_episode_steps": env.max_episode_steps})
    history, steps = [], []
    visible = {"observation": "", "admissible_commands": []}
    episode = {"schema_version": 2, "run_id": run_id, "episode_id": episode_id,
               "task_id": task["task_id"], "split": task["split"], "sealed": task.get("sealed", False),
               "game_sha256": task.get("game_sha256"), "environment_revision": environment_revision,
               "seed": env.seed, "max_episode_steps": env.max_episode_steps,
               "complete": False, "status": "started", "steps": steps,
               "policy_calls": [], "policy_turns_attempted": 0, "action_outcomes": []}
    if isinstance(policy, Teacher):
        episode.update(transport_failures=0, rejections=0)
    write_json(output_dir / "episode.json", episode)
    counter = 0
    episode_started = time.monotonic()

    def record(before, result, raw, action, metadata, elapsed, *, error=None,
               env_steps=0, outcome="not_applicable"):
        nonlocal counter
        observation, reward, terminated, truncated, audit = result
        event_id = f"{episode_id}:{counter:04d}"
        counter += 1
        parser_error = outcome == "parser_error"
        policy_rejection = parser_error or error == "context_overflow"
        source = "analytic_verifier" if policy_rejection else "real_environment" if error is None else "none"
        verification = {"audit": audit, "reward": reward, "policy_metadata": metadata,
                        "executable": "unknown" if error == "step_error" else not parser_error and error is None,
                        "success": audit.get("won", "unknown"), "fact_source": source,
                        "environment_steps_attempted": env_steps,
                        "environment_steps_confirmed": env_steps if error is None or policy_rejection else "unknown",
                        "failure_reason": error, "action_outcome": outcome,
                        "inadmissible": outcome == "inadmissible"}
        usage = metadata.get("mechanism_teacher_billed_usage") or (
            (metadata.get("billed_usage") or {}) if isinstance(policy, Teacher) else {})
        event = Event(
            run_id=run_id, episode_id=episode_id, event_id=event_id, split=task["split"],
            task_family=task.get("task_type", "alfworld"), task_id=task["task_id"],
            parent_seed_ids=list(parent_seed_ids), mechanism_id="audit_only",
            student_revision=config["models"]["student"]["revision"],
            teacher_provider_id=(config["models"]["teacher"]["provider_model_id"]
                                 if isinstance(policy, Teacher) else metadata.get("mechanism_teacher_provider_id")),
            environment_revision=environment_revision, policy_condition=policy_condition,
            history_ref=writer.blobs.put(visible_history(history)),
            visible_context_ref=writer.blobs.put(visible_observation(before)),
            action_raw=raw, action_parsed=action,
            observation_ref=(writer.blobs.put(visible_observation(observation))
                             if error is None or parser_error else None),
            verification_ref=writer.blobs.put(verification),
            feedback_source=source, new_environment_information="none" if policy_rejection else "unknown_until_audit",
            teacher_tokens_in=usage.get("input_tokens", 0), teacher_tokens_out=usage.get("output_tokens", 0),
            environment_steps=env_steps, latency_seconds=elapsed,
            terminated=terminated, truncated=truncated,
            status=("incomplete" if error == "budget_exhausted" else
                    "rejected" if policy_rejection else "ok" if error is None else "error"), error_type=error,
            action_outcome=outcome, inadmissible=outcome == "inadmissible",
            policy_metadata_ref=writer.blobs.put(metadata))
        writer.append(event)
        return {"event_id": event_id, "action_raw": raw, "action_parsed": action,
                "visible": visible_observation(observation), "reward": reward,
                "terminated": terminated, "truncated": truncated,
                "verification_ref": event.verification_ref, "audit_sha256": digest(audit),
                "action_outcome": outcome, "inadmissible": outcome == "inadmissible",
                "environment_steps": env_steps, "policy_metadata": metadata}

    phase, raw, action, metadata = "reset", None, None, {}
    outcome = "not_applicable"
    started = time.monotonic()
    try:
        result = env.reset(task["task_id"])
        episode["initial"] = record(visible, result, None, None, {}, time.monotonic() - started)
        visible, _, terminated, truncated, audit = result
        episode["game_sha256"] = env.game_hash
        write_json(output_dir / "episode.json", episode)
        while not (terminated or truncated):
            phase, raw, action, metadata = "policy", None, None, {}
            outcome = "not_applicable"
            started = time.monotonic()
            episode["policy_turns_attempted"] += 1
            # Only Teacher receives a task identity for authorization/cache, never
            # as prompt text. Student/history inputs remain explicitly whitelisted.
            result_policy = policy.generate(visible_history(history),
                                            task if isinstance(policy, Teacher) else visible_task(task),
                                            visible_observation(visible))
            raw, reported_action, metadata = result_policy
            action, outcome = action_outcome(raw, visible["admissible_commands"])
            if isinstance(policy, Teacher) and outcome in {"parser_error", "inadmissible"}:
                from teachers.stage import count
                count("rejections")
                episode["rejections"] += 1
            episode["action_outcomes"].append(outcome)
            metadata = dict(metadata)
            if isinstance(policy, Teacher):
                usage = metadata.get("usage") or {}
                metadata.update(prompt_tokens=usage.get("input_tokens"),
                                completion_tokens=usage.get("output_tokens"),
                                thinking_tokens=usage.get("reasoning_tokens"), thinking_chars=None,
                                finish_reason=metadata.get("response_status"))
            episode["policy_calls"].append(metadata)
            if action != reported_action:
                raise ActionParseError("Policy result disagrees with shared parser")
            before = deepcopy(visible)
            if outcome == "parser_error":
                # Keep the actual environment state/audit unchanged; this is
                # parser feedback, not a fabricated environment transition.
                result = ({"observation": PARSER_FEEDBACK,
                           "admissible_commands": before["admissible_commands"]},
                          0.0, False, False, audit)
            else:
                phase = "step"
                result = env.step(action)  # Inadmissible commands are issued verbatim.
            visible, reward, terminated, truncated, audit = result
            truncated = (truncated or len(steps) + 1 >= env.max_episode_steps) and not terminated
            result = (visible, reward, terminated, truncated, audit)
            step = record(before, result, raw, action, metadata, time.monotonic() - started,
                          env_steps=int(outcome != "parser_error"), outcome=outcome,
                          error="parser_error" if outcome == "parser_error" else None)
            steps.append(step)
            history.append({"visible": before, "reply": raw})
            write_json(output_dir / "episode.json", episode)
        episode.update(complete=True, status="ok", terminated=terminated, truncated=truncated,
                       success=audit.get("won", "unknown"))
        return episode
    except ContextOverflow as exc:
        metadata = exc.metadata
        episode["policy_calls"].append(metadata)
        # This terminates the policy, not the environment. Keep the actual
        # transition prefix intact for reset/replay and record rejection apart.
        terminal = (visible, 0.0, True, False, {"won": False, "termination_reason": "context_overflow"})
        episode["termination"] = record(visible, terminal, None, None, metadata,
            time.monotonic() - started, error="context_overflow")
        episode.update(complete=True, status="ok", terminated=True, truncated=False, success=False,
            termination_reason="context_overflow", termination_step_index=len(steps),
            termination_prompt_tokens=metadata["rendered_prompt_tokens"],
            termination_prompt_token_count_kind=metadata["prompt_token_count_kind"])
        return episode
    except Exception as exc:
        incomplete = getattr(exc, "incomplete", False)
        error = "budget_exhausted" if incomplete else "ActionParseError" if isinstance(exc, ActionParseError) else f"{phase}_error"
        if len(episode["policy_calls"]) < episode["policy_turns_attempted"]:
            episode["policy_calls"].append({"latency_seconds": time.monotonic() - started,
                                            "status": "incomplete" if incomplete else "failed", "usage": None})
        failed = (visible, 0.0, False, False,
                  {"phase": phase, "error_type": type(exc).__name__, "won": "unknown"})
        record(visible, failed, raw, action, metadata, time.monotonic() - started,
               error=error, env_steps=1 if phase == "step" else 0, outcome=outcome)
        episode.update(status="incomplete" if incomplete else "error", error_type=error, failed_action=action, complete=False,
                       success="unknown")
        if isinstance(exc, TeacherCallError):
            episode.update(teacher_transport_failure=True, transport_failures=1)
            # Preserve the typed failure after the runner's finally block saves
            # the actual execution prefix. Collectors must not turn it into loss.
            raise type(exc)("Teacher transport failed; inspect sanitized episode and usage ledger") from None
        status = "incomplete" if incomplete else "failed"
        raise RuntimeError(f"Episode {status} ({error}); inspect sanitized run artifacts") from None
    finally:
        episode["wall_time_seconds"] = time.monotonic() - episode_started
        episode["metrics"] = episode_metrics(episode)
        write_json(output_dir / "episode.json", episode)
        env.close()
        if hasattr(policy, "close"):
            policy.close()


def check_replay_result(result, expected, index, max_episode_steps):
    """Shared exact transition check for diagnostic replay and real branches."""
    visible, reward, terminated, truncated, audit = result
    truncated = (truncated or index >= max_episode_steps) and not terminated
    if (visible_observation(visible) != expected["visible"] or reward != expected["reward"] or
            terminated != expected["terminated"] or truncated != expected["truncated"] or
            digest(audit) != expected["audit_sha256"]):
        raise ReplayMismatch(f"Episode diverges at step {index}")


def replay_episode(env, episode):
    if episode.get("sealed") or episode["split"] == "test":
        raise ValueError("Sealed episodes cannot be used for diagnostic replay")
    if env.seed != episode["seed"] or env.max_episode_steps != episode["max_episode_steps"]:
        raise ReplayMismatch("Replay configuration differs")
    executed = 0
    try:
        result = env.reset(episode["task_id"])
        if env.game_hash != episode["game_sha256"]:
            raise ReplayMismatch("Game content differs")
        for index, expected in enumerate([episode["initial"], *episode["steps"]]):
            if index:
                if expected.get("action_outcome") == "parser_error":
                    if action_outcome(expected["action_raw"], result[0]["admissible_commands"])[1] != "parser_error":
                        raise ReplayMismatch("Recorded parser error is parseable")
                    result = ({"observation": PARSER_FEEDBACK,
                               "admissible_commands": result[0]["admissible_commands"]},
                              0.0, False, False, result[4])
                else:
                    result = env.step(expected["action_parsed"])
                    executed += 1
            check_replay_result(result, expected, index, env.max_episode_steps)
        return {"matched": True, "steps_replayed": executed, "policy_turns": len(episode["steps"]),
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
                         **rate_limit_options(spec, Path(args.output) / "data/teacher_rate_limit"),
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
