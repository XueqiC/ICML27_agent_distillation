"""ALFWorld through official textworld.start and AlfredDemangler APIs.

The single-task selection idea was informed by tc-alignment-vllm's
src/bfas/adapters/alfworld.py:_worker_episode (read-only reference).
This new adapter uses no old runtime, prompts, expert, or split semantics.
"""

from copy import deepcopy
from pathlib import Path
import random

from sed.storage import digest, file_hash
from sed.visibility import visible_observation


class ReplayMismatch(RuntimeError):
    """No branch can be used after replay diverges."""


class TextWorldBackend:
    def __init__(self, game_path, seed):
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
        # Never install AlfredExpert, request policy_commands/facts, or use the
        # AlfredTWEnv train mode (which automatically installs an expert).
        self.env = textworld.start(
            str(game_path),
            request_infos=textworld.EnvInfos(won=True, lost=True, admissible_commands=True),
            wrappers=[lambda env: AlfredDemangler(env, shuffle=False)])
        self.env.seed(seed)
        self.seed = seed

    def reset(self):
        # ALFWorld grammar can use Python/NumPy RNG. Preserve the caller's RNG
        # and carry episode-local RNG state through every transition.
        import numpy as np
        self.random_state = random.Random(self.seed).getstate()
        self.numpy_state = np.random.RandomState(self.seed).get_state()
        return self._call(self.env.reset)

    def _call(self, fn, *args):
        import numpy as np
        outer_random, outer_numpy = random.getstate(), np.random.get_state()
        try:
            random.setstate(self.random_state)
            np.random.set_state(self.numpy_state)
            return fn(*args)
        finally:
            self.random_state, self.numpy_state = random.getstate(), np.random.get_state()
            random.setstate(outer_random)
            np.random.set_state(outer_numpy)

    def step(self, action):
        return self._call(self.env.step, action)

    def close(self):
        self.env.close()


def partition_state(state):
    visible = visible_observation({"observation": state["feedback"],
                                   "admissible_commands": state["admissible_commands"]})
    # Native PDDL states contain hidden _facts even when facts were not
    # requested. Export a narrow audit, never serialize the whole GameState.
    facts = state.get("_facts", state.get("facts", []))
    audit = {"won": bool(state.get("won", False)), "lost": bool(state.get("lost", False)),
             "internal_facts": sorted(str(f) for f in facts),
             "fact_source": "real_environment_internal_state",
             "oracle_fields_discarded": sorted(k for k in state if
                 "expert" in k or "policy_commands" in k or "walkthrough" in k)}
    return visible, audit


class Environment:
    def __init__(self, data_root, *, max_episode_steps=40, seed=0,
                 backend_factory=TextWorldBackend, split="train"):
        if type(max_episode_steps) is not int or max_episode_steps <= 0:
            raise ValueError("Step limit must be a positive integer")
        self.data_root = Path(data_root).resolve()
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.backend_factory = backend_factory
        if split not in {"train", "dev", "test"}:
            raise ValueError("Unknown logical split")
        self.split = split
        self.backend = None
        self.ready = False
        self.task_id = None

    def reset(self, task_id):
        self.close()
        parts = Path(task_id).parts
        if Path(task_id).is_absolute() or ".." in parts or not parts:
            raise ValueError("Task ID must be a relative manifest path")
        if parts[0] == "valid_unseen" and self.split != "test":
            raise ValueError("Sealed task cannot be opened as train/dev")
        if self.split == "train" and parts[0] != "train":
            raise ValueError("Training requires a native train task")
        path = (self.data_root / task_id / "game.tw-pddl").resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError("Task path escapes data root")
        self.game_hash = file_hash(path)
        self.backend = self.backend_factory(path, self.seed)
        self.task_id = task_id
        self.actions, self.states = [], []
        self.ready = True
        try:
            state = self.backend.reset()
            visible, audit = partition_state(state)
            self.terminated = audit["won"] or audit["lost"]
            self.truncated = False
            self.states.append(self._signature(visible, 0.0, audit))
            return deepcopy(visible), 0.0, self.terminated, False, audit
        except BaseException:
            self.close()
            raise

    def _signature(self, visible, reward, audit):
        return {"visible": visible, "reward": float(reward),
                "terminated": self.terminated, "truncated": self.truncated,
                "audit_sha256": digest(audit)}

    def step(self, action):
        if not self.ready or self.terminated or self.truncated:
            raise RuntimeError("Reset required before stepping an inactive episode")
        if type(action) is not str or not action.strip() or "\n" in action or "\r" in action:
            raise ValueError("Expected one text command")
        try:
            state, reward, done = self.backend.step(action)
            visible, audit = partition_state(state)
            self.actions.append(action)
            self.terminated = bool(done)
            self.truncated = len(self.actions) >= self.max_episode_steps and not self.terminated
            self.states.append(self._signature(visible, reward, audit))
            return deepcopy(visible), float(reward), self.terminated, self.truncated, audit
        except BaseException:
            # A failed environment call may have mutated state. Never guess.
            self.close()
            raise

    def snapshot(self):
        if not self.ready:
            raise RuntimeError("No active state to snapshot")
        if self.split not in {"train", "dev"} or self.task_id.startswith("valid_unseen/"):
            raise ValueError("Branching is restricted to train/dev")
        body = {"schema_version": 1, "task_id": self.task_id, "game_sha256": self.game_hash,
                "seed": self.seed, "max_episode_steps": self.max_episode_steps,
                "split": self.split, "actions": deepcopy(self.actions),
                "states": deepcopy(self.states)}
        return dict(body, snapshot_sha256=digest(body))

    def restore(self, snapshot):
        self.close()
        body = {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}
        try:
            if digest(body) != snapshot["snapshot_sha256"]:
                raise ReplayMismatch("Snapshot hash mismatch")
            if self.split not in {"train", "dev"} or body["split"] != self.split:
                raise ReplayMismatch("Branch split mismatch")
            if body["task_id"].startswith("valid_unseen/"):
                raise ReplayMismatch("Sealed task branching is forbidden")
            if body["seed"] != self.seed or body["max_episode_steps"] != self.max_episode_steps:
                raise ReplayMismatch("Replay configuration mismatch")
            if len(body["states"]) != len(body["actions"]) + 1:
                raise ReplayMismatch("Incomplete prefix evidence")
            result = self.reset(body["task_id"])
            if self.game_hash != body["game_sha256"] or self.states[0] != body["states"][0]:
                raise ReplayMismatch("Reset state differs from snapshot")
            for index, action in enumerate(body["actions"], 1):
                result = self.step(action)
                if self.states[-1] != body["states"][index]:
                    raise ReplayMismatch(f"State differs after prefix step {index}")
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        self.ready = False
        if self.backend is not None:
            self.backend.close()
            self.backend = None
