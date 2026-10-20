"""Reset, check a real prefix, then continue through the unchanged episode runner."""
import argparse
from copy import deepcopy
from pathlib import Path

from analysis.mechanism.common import (analysis_inputs, authorize_episode, budget_directory, frozen_episode, output_guard,
                                      read_json, new_output)
from analysis.mechanism.examples import load_seeds, render_example, select_example
from analysis.mechanism.policies import (ADVICE_INSTRUCTION, ADVICE_PROMPT_TEMPLATE, BranchStudent, BudgetExceeded, TeacherBudget,
    add_budget_args, add_student_args, make_teacher, student_spec, validate_continuation)
from analysis.mechanism.visible import information_action, restatement
from harness.alfworld_env import ReplayMismatch
from sed.config import load_config
from sed.react import PARSER_FEEDBACK, action_outcome
from sed.runner import _environment, check_replay_result, run_episode
from sed.storage import digest, write_json
from sed.visibility import visible_history, visible_observation, visible_task
from teachers.openai_teacher import Teacher, TeacherResult
from teachers.stage import teacher_stage, count


class CheckedEnvironment:
    """The runner really resets and executes the prefix; no fabricated restoration."""
    def __init__(self, env, parent, t):
        self.env, self.parent, self.t = env, parent, t
        self.turn = 0
        self.result = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task_id):
        self.result = self.env.reset(task_id)
        if self.env.game_hash != self.parent["game_sha256"]:
            raise ReplayMismatch("Parent game content changed")
        check_replay_result(self.result, self.parent["initial"], 0, self.max_episode_steps)
        return self.result

    def step(self, action):
        self.result = self.env.step(action)
        if self.turn <= self.t:
            check_replay_result(self.result, self.parent["steps"][self.turn-1], self.turn, self.max_episode_steps)
        return self.result

    def parser_replay(self, expected, commands):
        result = ({"observation": PARSER_FEEDBACK, "admissible_commands": commands},
                  0.0, False, False, self.result[4])
        check_replay_result(result, expected, self.turn, self.max_episode_steps)


class BranchPolicy:
    def __init__(self, env, parent, task, t, continuation, arm, directory, teacher=None, example=None):
        self.env, self.parent, self.task, self.t = env, parent, task, t
        self.continuation, self.arm, self.directory = continuation, arm, Path(directory)
        self.teacher, self.example = teacher, example
        self.spec = {"continuation": getattr(continuation, "spec", None), "arm": arm}
        self.intervention = {"arm": arm, "user_turn": t, "history_retention": "persists_in_history",
                             "applied": False}
        self.first_action = None
        self.first_action_checked = False
        self.failure = None

    def _call(self, policy, history, current):
        result = policy.generate(visible_history(history),
                                 self.task if isinstance(policy, Teacher) else visible_task(self.task),
                                 visible_observation(current))
        if isinstance(policy, Teacher):
            meta = dict(result.metadata)
            usage = meta.get("usage") or {}
            meta.update(prompt_tokens=usage.get("input_tokens"), completion_tokens=usage.get("output_tokens"),
                        thinking_tokens=usage.get("reasoning_tokens"), thinking_chars=None,
                        mechanism_teacher_billed_usage=meta.get("billed_usage") or {},
                        mechanism_teacher_provider_id=meta.get("provider_model_id"))
            result = TeacherResult(result.raw_text, result.parsed_action, meta)
        return result

    def generate(self, history, task, current):
        try:
            return self._generate(history, current)
        except Exception as exc:
            self.failure = type(exc).__name__
            raise

    def _generate(self, history, current):
        turn = len(history)
        self.env.turn = turn + 1
        if turn < self.t:
            expected = self.parent["steps"][turn]
            raw = expected["action_raw"]
            action, result = action_outcome(raw, current["admissible_commands"])
            if action != expected["action_parsed"] or result != expected["action_outcome"]:
                raise ReplayMismatch("Parent action/parser outcome differs")
            if result == "parser_error":
                self.env.parser_replay(expected, current["admissible_commands"])
            return TeacherResult(raw, action, {"policy": "prefix_replay", "parent_step": turn+1})
        advice_result = None
        if turn == self.t and not self.intervention["applied"]:
            self.intervention["applied"] = True
            suffix = None
            if self.arm == "I1":
                suffix = restatement(history, current)
            elif self.arm == "I2":
                action = information_action(history, current)
                self.intervention.update(information_action=action, environment_step_cost=1,
                                         student_user_turn=turn+1)
                write_json(self.directory / "intervention.json", self.intervention)
                return TeacherResult("Action: " + action, action, {"policy": "I2_information_action"})
            elif self.arm == "I3":
                advice_result = self._call(self.teacher, history, current)
                self.intervention.update(teacher_raw=advice_result.raw_text, teacher_metadata=advice_result.metadata,
                                         prompt_template_sha256=digest(ADVICE_PROMPT_TEMPLATE),
                                         instruction_sha256=digest(ADVICE_INSTRUCTION))
                write_json(self.directory / "intervention.json", self.intervention)
                # Keep a disobedient response as a rejected attempt, never execute it.
                if action_outcome(advice_result.raw_text, current["admissible_commands"])[0] is not None:
                    count("rejections")
                    raise ValueError("I3 teacher advice included an action")
                suffix = "Advice: " + advice_result.raw_text
            elif self.arm == "I4":
                result = self._call(self.teacher, history, current)
                self.intervention.update(teacher_raw=result.raw_text, teacher_metadata=result.metadata)
                write_json(self.directory / "intervention.json", self.intervention)
                return result
            elif self.arm == "I5":
                if self.example is None:
                    raise ValueError("I5 unavailable: no same-task-type TRAIN seed")
                suffix = render_example(self.example)
                self.intervention.update(seed={k: v for k, v in self.example.items() if k != "episode"},
                                         seed_task_id=self.example["episode"]["task_id"])
            if suffix is not None:
                self.continuation.overlay.append(turn, suffix)
                self.intervention["appended_text"] = suffix
            write_json(self.directory / "intervention.json", self.intervention)
        result = self._call(self.continuation, history, current)
        if turn == self.t and self.arm == "I0":
            self.first_action, self.first_action_checked = result.parsed_action, True
        if advice_result:
            result = TeacherResult(result.raw_text, result.parsed_action,
                dict(result.metadata, intervention_teacher=advice_result.metadata,
                     mechanism_teacher_billed_usage=advice_result.metadata.get("billed_usage") or {},
                     mechanism_teacher_provider_id=advice_result.metadata.get("provider_model_id")))
        return result

    def close(self):
        if hasattr(self.continuation, "close"):
            self.continuation.close()


@teacher_stage("output", name="mechanism.branch")
def run_branch(env, continuation, task, config, parent, t, output, *, arm="takeover", sample_index=0,
               sampling_seed=0, teacher=None, example=None, parent_path=None, inputs=(), replay_check=True):
    authorize_episode(parent)
    if task["task_id"] != parent["task_id"] or task["split"] != parent["split"]:
        raise ValueError("Branch task differs from parent")
    if arm not in {"takeover", *[f"I{i}" for i in range(6)]}:
        raise ValueError("Unknown intervention arm")
    if type(t) is not int or not 0 <= t <= len(parent["steps"]):
        raise ValueError("Prefix length outside parent trajectory")
    if arm != "takeover" and t >= len(parent["steps"]):
        raise ValueError("Intervention needs the logged next action")
    if (env.seed != parent["seed"] or env.max_episode_steps != 40 or parent["max_episode_steps"] != 40
            or config["environment"]["max_episode_steps"] != 40):
        raise ValueError("Branches preserve parent environment seed and fixed 40-turn cap")
    output = output_guard(output, inputs=[parent_path] if parent_path else [], config=config)
    if output.exists():
        raise FileExistsError("Branch attempts require a fresh output directory")
    branch = {"parent_episode": parent["episode_id"], "parent_task_id": parent["task_id"],
              "parent_episode_sha256": digest(parent), "parent_path": str(parent_path) if parent_path else None,
              "t": t, "arm": arm, "sample_index": sample_index, "sampling_seed": sampling_seed,
              "history_retention": "intervention user message persists in history",
              "purpose": "diagnosis", "training_exposure": 0, "prefix_replay_required": True,
              "expansion_source": "real_environment_diagnostic_branch",
              "continuation": getattr(continuation, "spec", None), "replay_check_required": replay_check}
    effective = deepcopy(config)
    effective["mechanism_branch"] = branch
    checked = CheckedEnvironment(env, parent, t)
    controller = BranchPolicy(checked, parent, task, t, continuation, arm, output, teacher, example)
    try:
        episode = run_episode(checked, controller, task, effective, output, policy_condition="mechanism_"+arm,
                              purpose="diagnosis",
                              parent_seed_ids=[example["episode"]["episode_id"]] if arm == "I5" and example else [],
                              input_paths=analysis_inputs(config, [*inputs, *([parent_path] if parent_path else [])]))
    except Exception:
        if controller.failure == "BudgetExceeded":
            raise BudgetExceeded("Teacher cap reached; branch remains incomplete") from None
        raise
    finally:
        if (output / "episode.json").exists():
            episode = read_json(output / "episode.json")
            branch["replay_matched"] = (len(episode["steps"]) >= t and "initial" in episode and
                                        controller.failure != "ReplayMismatch" and episode.get("error_type") != "reset_error")
            branch["failure_type"] = controller.failure
            if arm == "I0" and sample_index == 0 and replay_check:
                expected = parent["steps"][t]["action_parsed"]
                branch["logged_next_action"] = expected
                branch["reproduced_next_action"] = (controller.first_action == expected if controller.first_action_checked else None)
                branch["non_deterministic"] = (controller.first_action_checked and controller.first_action != expected
                                               or episode.get("success") is True)
            episode["branch"] = branch
            write_json(output / "episode.json", episode)
    return episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--t", required=True, type=int)
    parser.add_argument("--policy", choices=["teacher", "student"], required=True)
    parser.add_argument("--arm", choices=["takeover", *[f"I{i}" for i in range(6)]], default="takeover")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--verified")
    parser.add_argument("--output", required=True)
    add_student_args(parser, required=False)
    add_budget_args(parser)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    parent, task = frozen_episode(args.episode, config, check_files=True)
    if args.policy == "student" and (not args.endpoint or not args.model_name):
        parser.error("Student continuation requires --endpoint and --model-name")
    spec = student_spec(config, args) if args.policy == "student" else None
    if args.arm != "takeover" and args.policy != "student":
        parser.error("Intervention arms require student continuation")
    if spec:
        validate_continuation(config, spec, args.episode, verified=args.verified if args.arm == "I5" else None)
    seeds = load_seeds(args.verified, config) if args.verified else []
    output_guard(args.output, inputs=[args.episode])
    output = new_output(args.output, config=config, command="branch", inputs=[args.config, args.episode])
    budget = TeacherBudget(budget_directory(args.budget_dir, output)/"budget.json", max_episode=args.max_teacher_calls,
                           max_run=args.max_teacher_calls_run)
    teacher = make_teacher(config, output, budget, parent["episode_id"], namespace=digest([parent["episode_id"], args.t, args.arm]),
                           sample_index=args.sample_index, advice=args.arm == "I3")
    sample_seed = (int(digest([args.sampling_seed, parent["episode_id"], args.t, args.sample_index])[:8], 16)
                   if args.sample_index else args.sampling_seed)
    continuation = teacher if args.policy == "teacher" else BranchStudent(**spec, log_dir=output/"episode/student_calls",
                          sample_index=args.sample_index, sampling_seed=sample_seed)
    return run_branch(_environment(config, task), continuation, task, config, parent, args.t, output/"episode",
                      arm=args.arm, sample_index=args.sample_index, sampling_seed=sample_seed,
                      teacher=teacher, example=select_example(parent["task_id"], seeds) if seeds else None,
                      parent_path=args.episode, inputs=["docs/MECHANISM_PLAN.md", args.config])


if __name__ == "__main__":
    main()
