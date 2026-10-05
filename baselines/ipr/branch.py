"""Real reset/prefix replay through the shared runner; no ICL or restored state.

Small replay-controller adaptation of analysis/mechanism/branch.py. This is
TRAIN generation, not a diagnostic branch: it has its own eligibility/lineage.
"""
from copy import deepcopy
from pathlib import Path

from harness.alfworld_env import ReplayMismatch
from sed.manifest import authorize_task
from sed.react import action_outcome, PARSER_FEEDBACK
from sed.runner import check_replay_result, run_episode
from sed.storage import digest, write_json
from teachers.openai_teacher import TeacherResult


class CheckedEnvironment:
    def __init__(self, env, parent, prefix):
        self.env, self.parent, self.prefix = env, parent, prefix
        self.turn, self.checked, self.result = 0, -1, None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def check(self, result, expected, turn):
        check_replay_result(result, expected, turn, self.max_episode_steps)
        self.checked = turn

    def reset(self, task_id):
        self.result = self.env.reset(task_id)
        if self.env.game_hash != self.parent["game_sha256"]:
            raise ReplayMismatch("Parent game content changed")
        self.check(self.result, self.parent["initial"], 0)
        return self.result

    def step(self, action):
        self.result = self.env.step(action)
        if self.turn <= self.prefix:
            self.check(self.result, self.parent["steps"][self.turn - 1], self.turn)
        return self.result


class PrefixPolicy:
    def __init__(self, checked, continuation):
        self.env, self.continuation = checked, continuation
        self.spec = {"continuation": continuation.spec, "prefix": checked.prefix}

    def generate(self, history, task, visible):
        index = len(history)
        self.env.turn = index + 1
        if index >= self.env.prefix:
            return self.continuation.generate(history, task, visible)
        expected = self.env.parent["steps"][index]
        raw = expected["action_raw"]
        action, outcome = action_outcome(raw, visible["admissible_commands"])
        if action != expected["action_parsed"] or outcome != expected["action_outcome"]:
            raise ReplayMismatch("Parent action/parser outcome differs")
        if outcome == "parser_error":
            self.env.check(({"observation": PARSER_FEEDBACK,
                "admissible_commands": visible["admissible_commands"]}, 0., False, False,
                self.env.result[4]), expected, index + 1)
        return TeacherResult(raw, action, {"policy": "prefix_replay", "parent_step": index + 1,
            "prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0, "thinking_chars": 0})

    def close(self):
        self.continuation.close()


def run_branch(env, continuation, task, config, parent, prefix, output, *, lineage, inputs=()):
    authorize_task(task, "generation")
    authorize_task(parent, "generation")
    if (task["source_split"] != "train" or parent["task_id"] != task["task_id"]
            or parent["split"] != "train" or parent["game_sha256"] != task["game_sha256"]):
        raise ValueError("IPR branches require the same frozen seed TRAIN game")
    if (type(prefix) is not int or not 0 <= prefix <= len(parent["steps"])
            or env.seed != parent["seed"] or env.max_episode_steps != 40
            or parent["max_episode_steps"] != 40 or config["environment"]["max_episode_steps"] != 40):
        raise ValueError("IPR preserves the parent seed, prefix and 40-turn cap")
    branch = {**lineage, "prefix": prefix, "parent_episode_sha256": digest(parent),
        "source": "ipr_real_environment_branch", "trusted_seed": False,
        "trusted_trajectories_added": 0, "training_exposure": 0, "replay_matched": False}
    effective = deepcopy(config)
    effective["ipr_branch"] = branch
    checked = CheckedEnvironment(env, parent, prefix)
    try:
        episode = run_episode(checked, PrefixPolicy(checked, continuation), task, effective, output,
            policy_condition="ipr_" + lineage["kind"], purpose="generation",
            parent_seed_ids=[lineage["parent_seed_id"]], input_paths=inputs)
    finally:
        path = Path(output) / "episode.json"
        if path.exists():
            import json
            episode = json.loads(path.read_text())
            branch["replay_matched"] = checked.checked == prefix
            episode["ipr_branch"] = branch
            write_json(path, episode)
    if not branch["replay_matched"]:
        raise ReplayMismatch("Prefix ended before all requested actions were replayed")
    return episode
