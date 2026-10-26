"""Reset-and-replay execution using the harness parser, visibility and won field."""

from copy import deepcopy
from dataclasses import replace
from importlib import metadata
from pathlib import Path
import uuid

from sed.react import action_outcome, PARSER_FEEDBACK
from sed.storage import digest, file_hash
from sed.visibility import visible_observation

from repair.align import action_reply, canonical_action
from repair.ledger import BUDGETS
from repair.records import TrajectoryRecord


def environment_version(env):
    dependencies = {}
    for name in ("alfworld", "textworld", "numpy"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    root = Path(__file__).resolve().parents[1]
    backend = getattr(env, "backend_factory", type(None))
    return digest({"dependencies": dependencies,
                   "adapter": type(env).__module__ + "." + type(env).__qualname__,
                   "backend": getattr(backend, "__module__", type(backend).__module__) + "." +
                              getattr(backend, "__qualname__", type(backend).__qualname__),
                   "code": {p: file_hash(root / p) for p in (
                       "harness/alfworld_env/environment.py", "src/sed/react.py",
                       "src/sed/visibility.py", "repair/align.py", "repair/execute.py")}})


def task_fingerprint(version, game_sha256, initial_visible, seed):
    """Game content binds initial state/goal; public reset text binds randomization."""
    return digest({"environment_version": version, "game_sha256": game_sha256,
                   "initial_visible": visible_observation(initial_visible), "env_seed": seed})


class ResultCache:
    """Explicit, process-local cache of real replay evidence; ENV_ERROR is not cached."""
    def __init__(self):
        self._results = {}
        self.hits = 0

    def get(self, key, scope):
        result = self._results.get((key, scope))
        if result is None:
            return None
        self.hits += 1
        steps = [dict(s, environment_steps=0, feedback_source=(
            "analytic_verifier" if s["action_outcome"] == "parser_error"
            else "cached_real_environment")) for s in result.steps]
        return replace(deepcopy(result), steps=steps, cache_hit=True,
                       cache_source_trajectory_id=result.trajectory_id,
                       environment_steps=0, environment_steps_confirmed=0, environment_steps_unknown=0)

    def put(self, key, scope, result):
        if result.termination_reason != "ENV_ERROR":
            self._results[(key, scope)] = deepcopy(result)


def _transition(result, *, raw=None, action=None, outcome="not_applicable", steps=0,
                feedback_source="real_environment"):
    visible, reward, terminated, truncated, audit = result
    return {"action_raw": raw, "action_parsed": action, "action_outcome": outcome,
            "visible": visible_observation(visible), "reward": reward,
            "terminated": terminated, "truncated": truncated, "audit_sha256": digest(audit),
            "environment_steps": steps, "feedback_source": feedback_source}


def execute_actions(env_factory, task_id, actions, max_steps, *, cache=None, ledger=None,
                    interaction_kind="candidate_replay_steps", version=None, candidate_id=None,
                    retry_of=None, continuation=None, prefix_reference=None):
    """max_steps counts policy turns as in runner; failed step calls are charged.

    Cache lookup follows a real reset (zero action steps), never a snapshot.
    Limits, exact parsed commands and reset audit digest additionally scope the
    required content key, so whitespace-sensitive execution cannot be aliased.
    """
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if interaction_kind not in BUDGETS["interaction"]:
        raise ValueError("Unknown interaction budget")
    if continuation is not None and (cache is not None or prefix_reference is None):
        raise ValueError("Hand-back requires an uncached replay and a measured prefix reference")
    if continuation is not None and (prefix_reference.task_id != task_id or prefix_reference.max_steps != max_steps):
        raise ValueError("Hand-back task/horizon differs from its measured reference")
    actions = list(actions)
    raw_actions = [action_reply(a) for a in actions]
    attempt_id = str(uuid.uuid4())
    if ledger is not None:
        ledger.append("retry" if retry_of else "attempt", settled=False, attempt_id=attempt_id,
                      parent_attempt_id=retry_of, task_id=task_id, candidate_id=candidate_id)
    env, initial, steps, key, scope = None, None, [], None, None
    used, confirmed, unknown = 0, 0, 0
    game_hash, fingerprint, seed, error_type = None, None, None, None
    success, reason, phase, result_record = "unknown", "ENV_ERROR", "factory", None
    try:
        env = env_factory()
        phase = "reset"
        seed = env.seed
        version = version or environment_version(env)
        result = env.reset(task_id)
        initial = _transition(result)
        game_hash = env.game_hash
        fingerprint = task_fingerprint(version, game_hash, initial["visible"], seed)
        key = (version, game_hash, fingerprint, tuple(canonical_action(a) for a in actions), seed)
        scope = (max_steps, env.max_episode_steps, initial["audit_sha256"],
                 tuple(action_outcome(raw, [])[0] for raw in raw_actions),
                 # Invalid replies are distinct evidence even when canonical text matches.
                 tuple(raw if action_outcome(raw, [])[0] is None else None for raw in raw_actions))
        result_record = cache.get(key, scope) if cache is not None else None
        if result_record is not None:
            # The transitions came from the cached execution, but raw replies
            # belong to this attempt (e.g. different public Thought text).
            result_record = replace(result_record, task_id=task_id,
                                    steps=[dict(s, action_raw=raw_actions[i])
                                           for i, s in enumerate(result_record.steps)])
        else:
            visible, _, terminated, truncated, audit = result
            success = audit.get("won", "unknown")
            reason = "FAILED"
            for index in range(max_steps):
                if success is True or terminated or truncated or index >= max_steps:
                    break
                if index < len(raw_actions):
                    raw = raw_actions[index]
                elif continuation is None:
                    break
                else:
                    phase = "handback_prefix_check"
                    reference = prefix_reference
                    keys = ("visible", "audit_sha256", "reward", "terminated", "truncated")
                    if (game_hash != reference.game_sha256 or seed != reference.env_seed or
                            version != reference.environment_version or
                            any(initial[k] != reference.initial[k] for k in keys) or
                            any(any(s[k] != r[k] for k in (*keys, "action_parsed"))
                                for s, r in zip(steps[:len(raw_actions)], reference.steps))):
                        raise ValueError("Handback prefix replay diverged")
                    from sed.visibility import visible_history
                    history, previous = [], initial["visible"]
                    for step in steps:
                        history.append({"visible": previous, "reply": step["action_raw"]})
                        previous = step["visible"]
                    phase = "handback_policy"
                    raw = action_reply(continuation(visible_history(history), visible_observation(visible)))
                action, outcome = action_outcome(raw, visible["admissible_commands"])
                if outcome == "parser_error":
                    result = ({"observation": PARSER_FEEDBACK,
                               "admissible_commands": visible["admissible_commands"]},
                              0.0, False, False, audit)
                else:
                    phase = "step"
                    used += 1
                    try:
                        result = env.step(action)
                    except Exception:
                        unknown += 1
                        steps.append({"action_raw": raw, "action_parsed": action,
                                      "action_outcome": outcome, "visible": None,
                                      "environment_steps": 1, "feedback_source": "none",
                                      "error_type": "step_error"})
                        raise
                    confirmed += 1
                phase = "observation"
                visible, reward, terminated, truncated, audit = result
                # The runner also caps parser-error turns, which do not step the env.
                truncated = (truncated or index + 1 >= min(max_steps, env.max_episode_steps)) and not terminated
                result = (visible, reward, terminated, truncated, audit)
                steps.append(_transition(result, raw=raw, action=action, outcome=outcome,
                                         steps=int(outcome != "parser_error"), feedback_source=(
                                             "analytic_verifier" if outcome == "parser_error"
                                             else "real_environment")))
                if continuation is not None and index < len(raw_actions):
                    phase = "handback_prefix_check"
                    reference = prefix_reference
                    keys = ("visible", "audit_sha256", "reward", "terminated", "truncated")
                    if (index >= len(reference.steps) or game_hash != reference.game_sha256 or
                            seed != reference.env_seed or version != reference.environment_version or
                            any(initial[k] != reference.initial[k] for k in keys) or
                            any(steps[-1][k] != reference.steps[index][k]
                                for k in (*keys, "action_parsed"))):
                        raise ValueError("Handback prefix replay diverged")
                success = audit.get("won", "unknown")
            reason = ("SUCCESS" if success is True else "FAILED" if terminated else
                      "STEP_LIMIT" if truncated or len(steps) >= max_steps else "FAILED")
    except Exception as exc:
        # Exception text can contain credentials or private state. Store only type/phase.
        success, reason, error_type = "unknown", "ENV_ERROR", phase + ":" + type(exc).__name__
        result_record = None
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                success, reason, error_type = "unknown", "ENV_ERROR", "close:" + type(exc).__name__
                result_record = None
    if result_record is None:
        result_record = TrajectoryRecord(task_id, initial, steps, success, reason,
            environment_steps=used, environment_steps_confirmed=confirmed, environment_steps_unknown=unknown,
            environment_version=version, game_sha256=game_hash, fingerprint=fingerprint, env_seed=seed,
            max_steps=max_steps, error_type=error_type)
        if cache is not None and key is not None:
            cache.put(key, scope, result_record)
    if ledger is not None:
        ledger.append("cache_hit" if result_record.cache_hit else
                      "success" if result_record.success is True else "failure",
                      attempt_id=attempt_id, parent_attempt_id=retry_of, task_id=task_id,
                      candidate_id=candidate_id, reason=result_record.termination_reason,
                      interaction={interaction_kind: result_record.environment_steps},
                      details={"trajectory_id": result_record.trajectory_id,
                               "cache_source_trajectory_id": result_record.cache_source_trajectory_id,
                               "environment_steps_confirmed": result_record.environment_steps_confirmed,
                               "environment_steps_unknown": result_record.environment_steps_unknown,
                               "error_type": result_record.error_type})
    return result_record
