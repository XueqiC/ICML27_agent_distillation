"""Small deterministic backend exercised through the real Environment adapter."""

from functools import partial

from harness.alfworld_env import Environment


TASK_ID = "train/room_a/trial_0"


class RepairBackend:
    instances = []

    def __init__(self, path, seed):
        self.key, self.door, self.won, self.lost = False, False, False, False
        self.actions, self.closed = [], False
        self.instances.append(self)

    def state(self):
        return {"feedback": f"Key={self.key}; door={self.door}; actions={','.join(self.actions)}",
                "admissible_commands": ["look", "inspect", "take key", "open door", "finish", "fail", "boom"],
                "won": self.won, "lost": self.lost,
                "_facts": ["PRIVATE_FACT"], "extra.expert_plan": ["PRIVATE_PLAN"],
                "policy_commands": ["PRIVATE_POLICY"], "hidden_success": "PRIVATE_LABEL"}

    def reset(self):
        return self.state()

    def step(self, action):
        self.actions.append(action)
        if action == "boom":
            raise RuntimeError("PRIVATE_EXCEPTION_SECRET")
        if action == "take key":
            self.key = True
        if action == "open door" and self.key:
            self.door = True
        if action == "finish":
            self.won = self.key and self.door
        if action == "fail":
            self.lost = True
        # A positive reward alone must never imply success.
        return self.state(), 1.0, self.won or self.lost

    def close(self):
        self.closed = True


def factory(data_root, *, backend=RepairBackend, seed=0, limit=40):
    return partial(Environment, data_root, backend_factory=backend, seed=seed, max_episode_steps=limit)
