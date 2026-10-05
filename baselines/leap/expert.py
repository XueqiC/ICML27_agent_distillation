"""Audit-only handcoded expert, attached only by the LEAP TRAIN collector.

The shared Environment, runner, visibility partition and termination rules are
unchanged. AlfredExpert asks for facts internally; only its next command is
read here. Neither facts nor expert_plan becomes a teacher/student observation.
"""
from copy import deepcopy
import importlib.metadata
from pathlib import Path

from baselines.leap.protocol import require_train
from baselines.leap.teacher import read
from harness.alfworld_env.environment import Environment, TextWorldBackend
from sed.manifest import verify_task_files
from sed.runner import run_episode
from sed.seed_evidence import artifact_hashes
from sed.storage import write_json
from teachers.openai_teacher import TeacherResult


class ExpertBackend(TextWorldBackend):
    def __init__(self, game_path, seed):
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert
        if importlib.metadata.version("alfworld") != "0.4.2":
            raise ValueError("LEAP expert requires installed alfworld 0.4.2")
        self.env = textworld.start(str(game_path), request_infos=textworld.EnvInfos(
            won=True, lost=True, admissible_commands=True), wrappers=[
                lambda env: AlfredDemangler(env, shuffle=False),
                lambda env: AlfredExpert(env, expert_type="handcoded")])
        self.env.seed(seed)
        self.seed, self.plan = seed, []

    def reset(self):
        state = super().reset()
        self.plan = list(state.get("extra.expert_plan", []))
        return state

    def step(self, action):
        state, reward, done = super().step(action)
        self.plan = list(state.get("extra.expert_plan", []))
        return state, reward, done


class ExpertPolicy:
    spec = {"policy": "alfworld_0.4.2_AlfredExpert_handcoded", "audit_only": True}

    def __init__(self, env):
        self.env = env

    def generate(self, history, task, visible):
        plan = self.env.backend.plan
        if not plan or not isinstance(plan[0], str) or "\n" in plan[0] or "\r" in plan[0]:
            raise ValueError("Handcoded expert has no usable command")
        action = plan[0]
        return TeacherResult("Action: " + action, action, {"policy": self.spec["policy"],
            "prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0, "thinking_chars": 0})


def expert_environment(config, task):
    require_train(task)
    return Environment(config["environment"]["data_root"], split="train", backend_factory=ExpertBackend,
        seed=config["environment"]["seed"], max_episode_steps=40)


def collect_one(task, config, output, *, env_factory=expert_environment):
    require_train(task)
    if config["environment"]["max_episode_steps"] != 40:
        raise ValueError("LEAP expert uses the same 40-turn harness")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episode, error = {}, None
    effective = deepcopy(config)
    effective["leap_expert"] = {"privileged_training_resource": True,
        "expert": "alfworld.agents.environment.alfred_tw_env.AlfredExpert(handcoded)",
        "training_input": False, "trusted_seed": False}
    try:
        verify_task_files(config["environment"]["data_root"], task)
        env = env_factory(config, task)
        episode = run_episode(env, ExpertPolicy(env), task, effective, output / "episode",
            policy_condition="leap_audit_handcoded_expert", parent_seed_ids=task["parent_seed_ids"],
            input_paths=[config["paths"]["manifest"], config["paths"]["split_freeze"]])
    except Exception as exc:
        error = type(exc).__name__  # No raw environment/SDK errors.
        if (output / "episode/episode.json").exists():
            episode = read(output / "episode/episode.json")
    available = (episode.get("complete") is True and episode.get("status") == "ok"
                 and episode.get("success") is True and bool(episode.get("steps")))
    receipt = {"task_id": task["task_id"], "parent_seed_ids": task["parent_seed_ids"],
        "available": available, "success": episode.get("success", "unknown"),
        "error_type": error, "reason": "executed_win" if available else "expert_failed_or_unavailable",
        "episode_id": episode.get("episode_id"), "metrics": episode.get("metrics"),
        "trusted_trajectories_added": 0, "training_exposure": 0, "artifact_sha256": artifact_hashes(output)}
    write_json(output / "receipt.json", receipt)
    return receipt
