"""Private CPU validation worker. Never import this module in a training loader.

Uses TextWorld's installed Fast Downward ``PddlState.replan`` (the same
mechanism as AlfredExpert(planner)), only here in the validation process.
Every proposed command is executed through the existing Environment wrapper.
No expert wrapper, fact, or plan is added to the policy observation interface.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import socket

from harness.alfworld_env.environment import Environment
from repair.synth_tasks import digest, file_hash, group_parts, write_json


class ValidationFailure(Exception):
    pass


def task_line(visible):
    lines = [line.strip() for line in visible["observation"].splitlines() if line.startswith("Your task is to: ")]
    return lines[0] if len(lines) == 1 else None


def visible_only(visible):
    return {"observation": visible["observation"], "admissible_commands": list(visible["admissible_commands"])}


class Execution:
    def __init__(self, seed, env_factory=Environment):
        self.seed, self.env_factory = seed, env_factory
        self.env = None
        self.episodes = []

    def reset(self, root, task_id, purpose):
        self.close()
        self.env = self.env_factory(root, seed=self.seed, max_episode_steps=40, split="train")
        self.current = {"purpose": purpose, "task_id": task_id, "steps": []}
        self.episodes.append(self.current)
        self.result = self.env.reset(task_id)
        self.current["initial"] = self.entry(self.result)
        return self.result

    @staticmethod
    def entry(result):
        visible, reward, terminated, truncated, audit = result
        return {"visible": visible_only(visible), "reward": reward, "terminated": terminated,
                "truncated": truncated, "won": bool(audit["won"]), "audit_sha256": digest(audit)}

    def step(self, action):
        row = {"action": action, "status": "started"}
        self.current["steps"].append(row)
        self.result = self.env.step(action)
        row.update(self.entry(self.result), status="executed")
        return self.result

    def replay(self, actions):
        for action in actions:
            if self.result[2] or self.result[3]:
                break
            self.step(action)
        return self.result

    def plan(self):
        # Intentionally privileged, validation-only. Match backend RNG handling.
        backend = self.env.backend
        return backend._call(backend.env._pddl_state.replan, backend.env._entity_infos)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def transitions(self):
        return sum(len(episode["steps"]) for episode in self.episodes)


def same_episode(left, right):
    return left["initial"] == right["initial"] and left["steps"] == right["steps"]


def validate_game(job, out, source_root, execution):
    root = out / "data_root"
    initial = execution.reset(root, job["task_id"], "witness_first_reset")
    if task_line(initial[0]) != job["task_text"]:
        raise ValidationFailure("task_text_mismatch")
    if initial[4]["won"]:
        raise ValidationFailure("already_won_at_reset")
    plan = execution.plan()
    execution.current["proposed_plan"] = plan
    execution.replay(plan[:40])
    first = execution.current
    won_first = bool(execution.result[4]["won"])
    execution.reset(root, job["task_id"], "witness_second_reset")
    execution.replay(plan[:40])
    second = execution.current
    won_second = bool(execution.result[4]["won"])
    deterministic = same_episode(first, second)
    result = {"loads": True, "task_line_matches": True, "initial_task_text": task_line(initial[0]),
              "witness_mechanism": "TextWorld Fast Downward PddlState.replan",
              "witness_length": len(first["steps"]), "proposed_plan_length": len(plan),
              "won_first": won_first, "won_second": won_second,
              "same_initial_observation": first["initial"] == second["initial"],
              "deterministic": deterministic}
    if job["generator"] == "G2":
        execution.reset(root, job["task_id"], "parent_actions_negative_control")
        execution.replay(job["parent_actions"])
        result["parent_actions_win"] = bool(execution.result[4]["won"])
        result["negative_control"] = "coincidentally_wins" if result["parent_actions_win"] else "does_not_win"
    if not won_first or not won_second:
        result["failure_reason"] = "no_witness_within_40_steps"
    elif not deterministic:
        result["failure_reason"] = "nondeterministic_replay"
    return result


def choose_error(error_type, visible, next_action, object_type, recep_type):
    commands = sorted(visible["admissible_commands"])
    if error_type == "refused_action":
        action = "take nonexistentobject 999 from nonexistentreceptacle 999"
        if action in commands:
            raise ValidationFailure("refusal_control_is_admissible")
        return [action]
    if error_type == "wrong_object":
        options = [a for a in commands if a.startswith("take ") and a.split()[1] != object_type.lower()]
        return options[:1]
    if error_type == "wrong_receptacle":
        options = [a for a in commands if a.startswith("move " + object_type.lower() + " ")
                   and a.split(" to ", 1)[-1].split()[0] != recep_type.lower()]
        return options[:1]
    if error_type == "useless_search":
        # Repeated examination of an already-visited location; no correction follows.
        return ["look", "look"] if "look" in commands else []
    raise ValidationFailure("unsupported_error_type")


def validate_prefix(job, out, source_root, execution, parent_episode):
    _, obj_type, _, recep_type, _ = group_parts(job["parent_seed_task_id"])
    execution.reset(source_root, job["task_id"], "controlled_error_prefix")
    if execution.current["initial"] != parent_episode["initial"]:
        raise ValidationFailure("parent_reset_mismatch")
    if task_line(execution.result[0]) != job["task_text"]:
        raise ValidationFailure("task_text_mismatch")
    prefix, error_actions = [], []
    minimum = 2 if job["error_type"] == "useless_search" else job["prefix_index"] // 4
    for index, next_action in enumerate(job["parent_actions"]):
        if index >= minimum:
            error_actions = choose_error(job["error_type"], execution.result[0], next_action, obj_type, recep_type)
        if error_actions:
            break
        if execution.result[2] or execution.result[3] or index + 1 + 2 >= 40:
            break
        execution.step(next_action)
        prefix.append(next_action)
        if execution.current["steps"][-1] != parent_episode["steps"][index]:
            raise ValidationFailure("verified_prefix_replay_mismatch")
    if not error_actions:
        raise ValidationFailure("no_controlled_error_opportunity")
    before = execution.entry(execution.result)
    for action in error_actions:
        execution.step(action)
    first = execution.current
    if execution.result[2] or execution.result[3]:
        raise ValidationFailure("error_prefix_terminal")
    after = execution.entry(execution.result)
    if job["error_type"] == "refused_action":
        if before["audit_sha256"] != after["audit_sha256"] or after["visible"]["observation"] != "Nothing happens.":
            raise ValidationFailure("refused_action_changed_state")
    elif job["error_type"] in {"wrong_object", "wrong_receptacle"}:
        if after["visible"]["observation"] == "Nothing happens." or before["audit_sha256"] == after["audit_sha256"]:
            raise ValidationFailure("controlled_error_not_executed")
    actions = prefix + error_actions
    prefix_path = f"prefixes/{job['id']}.json"
    # Explicitly whitelisted feedback; no state, expert action, or outcome label.
    write_json(out / prefix_path, {"schema_version": 1, "task_id": job["task_id"],
        "parent_seed_task_id": job["parent_seed_task_id"], "generator": "G3", "error_type": job["error_type"],
        "prefix_actions": actions, "error_action_offset": len(prefix),
        "initial_observation": first["initial"]["visible"],
        "transitions": [{"action": s["action"], "observation": s["visible"]} for s in first["steps"]],
        "ends_immediately_after_error": True})
    execution.reset(source_root, job["task_id"], "controlled_error_second_reset")
    execution.replay(actions)
    second = execution.current
    deterministic = same_episode(first, second)
    # Witness starts from the genuinely replayed error state. It is written only
    # to the witness file, never appended to the prefix artifact.
    recovery_plan = execution.plan()
    second["recovery_start_offset"] = len(actions)
    second["proposed_recovery_plan"] = recovery_plan
    execution.replay(recovery_plan[:40 - len(actions)])
    recovery_steps = len(second["steps"]) - len(actions)
    won_first = bool(execution.result[4]["won"])
    execution.reset(source_root, job["task_id"], "controlled_error_recovery_repeat")
    execution.replay(actions + recovery_plan[:40 - len(actions)])
    won_second = bool(execution.result[4]["won"])
    deterministic = deterministic and same_episode(second, execution.current)
    result = {"loads": True, "task_line_matches": True, "deterministic": deterministic,
              "same_initial_observation": first["initial"] == second["initial"],
              "won_first": won_first, "won_second": won_second,
              "witness_mechanism": "TextWorld Fast Downward PddlState.replan from real error state",
              "witness_length": recovery_steps, "prefix_length": len(actions),
              "error_action_offset": len(prefix), "error_executed": True,
              "state_sha256": after["audit_sha256"], "prefix_actions_sha256": digest(actions),
              "prefix_path": prefix_path, "proposed_plan_length": len(recovery_plan)}
    if not won_first or not won_second:
        result["failure_reason"] = "no_recovery_witness_within_40_total_steps"
    elif not deterministic:
        result["failure_reason"] = "nondeterministic_replay"
    return result


def validate_jobs(out, env_factory=Environment, jobs=None):
    out = Path(out)
    request = json.loads((out / "validation_requests.json").read_text())
    seed, root = request["seed"], request["data_root"]
    jobs = request["jobs"] if jobs is None else jobs
    parent_cache, seen_states = {}, set()
    for index, job in enumerate(jobs):
        execution = Execution(seed, env_factory)
        validation, status, rejection = {}, "valid", None
        try:
            parent = job["parent_seed_task_id"]
            if file_hash(Path(root) / parent / "game.tw-pddl") != job["parent_game_sha256"]:
                raise ValidationFailure("parent_game_hash_changed")
            if parent not in parent_cache:
                execution.reset(root, parent, "verify_original_successful_seed_actions")
                execution.replay(job["parent_actions"])
                parent_cache[parent] = {"won": bool(execution.result[4]["won"]),
                                        "episode": execution.current, "witness_owner": job["id"]}
            cached = parent_cache[parent]
            validation["parent_replay_witness_id"] = cached["witness_owner"]
            if not cached["won"]:
                raise ValidationFailure("original_seed_actions_do_not_win")
            if job["generator"] == "G3":
                validation.update(validate_prefix(job, out, root, execution, cached["episode"]))
                state_key = (parent, validation["state_sha256"])
                if state_key in seen_states:
                    validation["failure_reason"] = "duplicate_recovery_state"
                seen_states.add(state_key)
            else:
                if file_hash(out / job["game_path"]) != job["game_sha256"]:
                    raise ValidationFailure("candidate_game_hash_changed")
                validation.update(validate_game(job, out, root, execution))
            if validation.get("failure_reason"):
                status, rejection = "rejected", validation["failure_reason"]
        except ValidationFailure as error:
            status, rejection = "rejected", str(error)
        except Exception as error:
            # Exception text may contain arbitrary runtime data; retain type only.
            status, rejection = "rejected", "validation_exception:" + type(error).__name__
        finally:
            execution.close()
        validation["environment_transitions"] = execution.transitions
        witness_path = f"witnesses/{job['id']}.json"
        write_json(out / witness_path, {"never_for_training": True, "purpose": "solvability validation only",
            "id": job["id"], "mechanism": "installed TextWorld Fast Downward planner",
            "episodes": execution.episodes, "validation": validation, "status": status, "rejection": rejection})
        result = {"status": status, "validation": validation, "witness_path": witness_path,
                  "witness_sha256": file_hash(out / witness_path)}
        if rejection:
            result["rejection"] = rejection
        if validation.get("prefix_path"):
            result["prefix_path"] = validation["prefix_path"]
        write_json(out / "validation" / (job["id"] + ".json"), result)
        print(f"{job['generator']} {job['id']} {status} {rejection or ''}", flush=True)


def offline():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    def forbidden(*args, **kwargs):
        raise RuntimeError("Network disabled in synthetic task validation")
    for name in ("connect", "connect_ex", "sendto"):
        setattr(socket.socket, name, forbidden)
    socket.create_connection = forbidden
    socket.getaddrinfo = forbidden


def validate_parent(args):
    out, jobs = args
    offline()
    validate_jobs(out, jobs=jobs)


def main():
    offline()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out
    request = json.loads((out / "validation_requests.json").read_text())
    grouped = defaultdict(list)
    for job in request["jobs"]:
        grouped[job["parent_seed_task_id"]].append(job)
    if request.get("workers", 1) == 1:
        validate_jobs(out)
    else:
        with ProcessPoolExecutor(max_workers=request["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            list(pool.map(validate_parent, [(out, jobs) for jobs in grouped.values()]))


if __name__ == "__main__":
    main()
