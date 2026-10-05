"""Stage 4a: offline fake policies/environments and frozen real-fact fixtures."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from analysis.mechanism.branch import run_branch
from analysis.mechanism.common import (audit_states, completed_branch, output_guard, seal_branch, workspace)
from analysis.mechanism.examples import render_example, select_example
from analysis.mechanism.fce import IncompleteProbe, dry_plan, locate_fce, probe_points
from analysis.mechanism.goals import Facts, object_types, parse_goal
from analysis.mechanism.interventions import SharedInterventionTeacher, arm_outcome
from analysis.mechanism.policies import (ADVICE_INSTRUCTION, ADVICE_PROMPT_TEMPLATE, BranchStudent, BranchTeacher, BudgetExceeded,
                                       Overlay, TeacherBudget, validate_continuation)
from analysis.mechanism.readout import build_readout, decision_cell, paired_mcnemar, wilson
from analysis.mechanism.taxonomy import LABELS, classify, comparison, shares
from analysis.mechanism.visible import information_action, restatement, summarize_visible
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.react import PARSER_FEEDBACK, build_messages, action_outcome
from sed.runner import ScriptedPolicy, run_episode
from sed.storage import BlobStore, digest, file_hash, write_json
from teachers.openai_teacher import TeacherResult

TASK_ID = "valid_seen/pick_heat_then_place_in_recep-Apple-None-CounterTop-1/trial_0"
TYPES = ["objecttype(a: object, appletype: otype)", "objecttype(m: object, mugtype: otype)",
         "receptacletype(c: receptacle, countertoptype: rtype)",
         "receptacletype(d: receptacle, drawertype: rtype)",
         "receptacletype(w: receptacle, microwavetype: rtype)",
         "objecttype(l: object, desklamptype: otype)"]


def visible(text="You see an apple 1.", commands=None):
    return {"observation": text, "admissible_commands": commands or ["look", "inventory"]}


def episode(actions, observations=None, facts=None, task_id=TASK_ID, success=False):
    observations = observations or [f"You see apple 1, observation {i}." for i in range(len(actions))]
    initial = {"visible": visible("Your task is to: heat apple.\nYou see a drawer 1.")}
    steps = [{"action_parsed": a, "action_raw": "Action: "+a if a else "bad",
              "visible": visible(o), "action_outcome": "legal" if a else "parser_error"}
             for a, o in zip(actions, observations)]
    audits = [{"internal_facts": TYPES + (f or []), "won": success and i == len(actions)}
              for i, f in enumerate(facts or [[] for _ in range(len(actions)+1)])]
    ep = {"task_id": task_id, "split": "dev", "sealed": False, "episode_id": "parent",
          "complete": True, "status": "ok", "success": success, "initial": initial,
          "steps": steps, "truncated": len(steps) == 40, "max_episode_steps": 40}
    return ep, audits


def test_goal_ids_and_complete_real_fact_lists():
    rows = json.loads(Path("tests/fixtures/mechanism_real_facts.json").read_text())
    assert len(rows) == 3
    for row in rows:
        goal = parse_goal(row["task_id"])
        facts = Facts(row["facts"])
        assert goal.base_object in set(facts.types.values())
        assert len(facts.types) > 10
        assert object_types(row["facts"]) == facts.types
        if goal.required:
            assert any(all(facts.has(p, obj) for p in goal.required) for obj in facts.types)
        else:
            assert goal.task_type == "look_at_obj_in_light"
            assert facts.get("istoggled") and facts.held()
    goal = parse_goal("train/pick_and_place_with_movable_recep-AppleSliced-Bowl-CounterTop-7/trial_0")
    assert goal.base_object == "apple" and goal.required == ["issliced"]
    assert goal.movable_receptacle == "Bowl" and goal.target_receptacle == "CounterTop"


@pytest.mark.parametrize("task_id", ["valid_unseen/room/trial", "../train/room", "/train/room"])
def test_goal_refuses_sealed_and_unsafe_paths(task_id):
    with pytest.raises(ValueError):
        parse_goal(task_id)


def test_taxonomy_order_all_rules_and_indices():
    ep, audits = episode(["take mug 1 from drawer 1", "look", "look", "look", None, None, None],
        observations=["You see apple 1."]+["nothing"]*6)
    ep["steps"][0]["action_outcome"] = "inadmissible"
    result = classify(ep, audits)
    assert result["primary"] == "F-format"
    assert result["labels"] == ["F-format", "F-goal", "F-loop"]
    assert result["triggers"]["F-format"] == [1, 5, 6, 7]
    assert result["triggers"]["F-goal"] == [1]
    assert set(result["evidence"]) == set(result["labels"])


def test_search_uses_only_observations_not_goal_actions_commands_or_audit():
    ep, audits = episode(["take apple 1 from drawer 1"], ["Nothing happens."])
    ep["steps"][0]["visible"]["admissible_commands"] = ["take apple 1 from drawer 1"]
    audits[1]["internal_facts"] += ["holds(agent1: agent, a: object)"]
    result = classify(ep, audits)
    assert "F-search" in result["labels"] and result["triggers"]["F-search"] == [0, 1]
    ep["steps"][0]["visible"]["observation"] = "In it, you see a pineapple 1."
    assert "F-search" in classify(ep, audits)["labels"]
    ep["steps"][0]["visible"]["observation"] = "In it, you see an apple 1."
    assert "F-search" not in classify(ep, audits)["labels"]


def test_goal_search_containers_do_not_count_as_goal_commitments():
    ep, audits = episode(["go to drawer 1", "open drawer 1", "examine drawer 1", "heat apple 1 with microwave 1"])
    assert "F-goal" not in classify(ep, audits)["labels"]
    ep["steps"][-1]["action_parsed"] = "move apple 1 to drawer 1"
    assert classify(ep, audits)["primary"] == "F-goal"


def test_preconditions_are_based_on_audit_not_teacher_claims():
    held = ["holds(agent1: agent, a: object)"]
    ep, audits = episode(["take apple 1 from drawer 1", "move apple 1 to countertop 1"],
                         facts=[[], held, ["inreceptacle(a: object, c: receptacle)"]])
    ep["steps"][0]["action_raw"] = "Thought: I heated it.\nAction: take apple 1 from drawer 1"
    result = classify(ep, audits)
    assert result["primary"] == "F-precondition"
    assert result["triggers"]["F-precondition"] == [2]
    for audit in audits[1:]:
        audit["internal_facts"].append("ishot(a: object)")
    assert "F-precondition" not in classify(ep, audits)["labels"]
    ep["steps"][1]["action_parsed"] = "heat apple 1 with drawer 1"
    assert "F-precondition" in classify(ep, audits)["labels"]


def test_look_at_object_toggled_lamp_and_order():
    tid = "valid_seen/look_at_obj_in_light-Apple-None-DeskLamp-1/trial_0"
    held = ["holds(agent1: agent, a: object)"]
    ep, audits = episode(["use desklamp 1", "take apple 1 from drawer 1"],
                         facts=[[], ["istoggled(l: object)"], held], task_id=tid)
    result = classify(ep, audits)
    assert result["primary"] == "F-precondition" and result["triggers"]["F-precondition"] == [1, 2]


def test_cycles_cap_success_unknown_and_format_boundary():
    ep, audits = episode(["look", "inventory"]*3, ["apple 1", "empty"]*3)
    assert "F-loop" in classify(ep, audits)["labels"]
    ep, audits = episode(["look"]*40)
    assert classify(ep, audits)["primary"] == "F-cap"
    ep["success"] = True
    audits[-1]["won"] = True
    assert classify(ep, audits)["labels"] == []
    ep["complete"] = False
    assert classify(ep, audits)["classification_status"] == "unknown"
    ep, audits = episode([None, "look", "inventory", "look"])
    assert classify(ep, audits)["primary"] == "F-format"


def test_gap_and_regression_require_known_paired_outcomes():
    student = {"episodes": [{"task_id": str(i), "success": s} for i, s in enumerate([False, False, "unknown", True])]}
    teacher = {"episodes": [{"task_id": str(i), "success": s} for i, s in enumerate([True, "unknown", True])]}
    result = comparison(student, teacher, "teacher")
    assert result["task_ids"] == ["0"] and result["unknown_pairs"] == 2
    assert result["unpaired_student_tasks"] == ["3"]


def test_visible_summary_latest_observation_held_and_no_audit():
    history = [{"visible": dict(visible(text), audit={"internal_facts": ["SECRET"]}, hidden="SECRET"),
                "reply": "Thought: SECRET false location\nAction: look", "audit": "SECRET"}
               for text in ["You arrive at drawer 1. The drawer 1 is closed.",
                "You open the drawer 1. In it, you see a mug 1, and an apple 1.",
                "You pick up the apple 1 from the drawer 1.",
                "You arrive at countertop 1. On the countertop 1, you see nothing.",
                "You move the apple 1 to the countertop 1."]]
    current = dict(visible("On the drawer 1, you see nothing."), audit="SECRET")
    summary = summarize_visible(history, current)
    assert summary["held"] is None
    assert summary["objects_seen_where"] == {"apple 1": "countertop 1"}
    assert summary["receptacles_visited"] == ["drawer 1", "countertop 1"]
    assert "SECRET" not in restatement(history, current)
    assert restatement(history, current).startswith("Facts so far:")


def test_information_rule_precedence_and_admissibility():
    history = [{"visible": visible("You arrive at drawer 1."), "reply": "Action: look"},
               {"visible": visible("You pick up the apple 1 from the drawer 1."), "reply": "Action: look"}]
    commands = ["look", "inventory", "examine drawer 1"]
    assert information_action(history, visible("Nothing happens.", commands)) == "inventory"
    assert information_action(history, visible("Other feedback.", commands)) == "examine drawer 1"
    assert information_action([], visible("room", commands)) == "look"
    with pytest.raises(ValueError):
        information_action([], visible("room", ["inventory"]))


def test_seed_retrieval_order_and_no_audit_rendering():
    seeds = []
    for i, object_type in enumerate(["Mug", "Apple", "Apple"]):
        ep, _ = episode(["look"], task_id=TASK_ID.replace("valid_seen", "train").replace("Apple", object_type), success=True)
        ep["split"] = "train"
        ep["audit"] = "SECRET"
        seeds.append({"order": i, "episode": ep})
    assert select_example(TASK_ID, seeds)["order"] == 1
    assert "SECRET" not in render_example(seeds[0])
    seeds[0]["episode"]["split"] = "dev"
    with pytest.raises(ValueError):
        select_example(TASK_ID, seeds)


class BranchBackend:
    def __init__(self, path, seed):
        self.n = 0
        self.won = False
    def state(self):
        return {"feedback": f"You arrive at drawer 1. On the drawer 1, you see an apple 1. Visit {self.n}.",
                "admissible_commands": ["look", "inventory", "examine drawer 1", "win"],
                "_facts": TYPES, "won": self.won, "lost": False}
    def reset(self):
        return self.state()
    def step(self, action):
        self.n += 1
        self.won = action == "win"
        return self.state(), float(self.won), self.won
    def close(self):
        pass


@pytest.fixture
def branch_parent(tmp_path):
    root = tmp_path/"dataset"
    directory = root/TASK_ID
    directory.mkdir(parents=True)
    (directory/"game.tw-pddl").write_text("{}")
    task = {"task_id": TASK_ID, "split": "dev", "sealed": False, "task_type": "pick_heat_then_place_in_recep"}
    config = load_config()
    env = lambda: Environment(root, backend_factory=BranchBackend, split="dev")
    parent_path = tmp_path/"parent"
    parent = run_episode(env(), ScriptedPolicy(["look"]), task, config, parent_path,
                         policy_condition="fake_student", purpose="diagnosis")
    return root, env, task, config, parent, parent_path/"episode.json"


class FakeStudent:
    def __init__(self, action="look", **kwargs):
        self.action, self.overlay, self.requests = action, Overlay(), []
        self.spec = {"policy": "fake"}
    def generate(self, history, task, current):
        messages = self.overlay.apply(build_messages(history, task, current))
        self.requests.append(messages)
        return TeacherResult("Action: "+self.action, self.action, {})


def test_real_reset_prefix_replay_and_remaining_cap(branch_parent, tmp_path):
    _, env, task, config, parent, path = branch_parent
    policy = FakeStudent()
    result = run_branch(env(), policy, task, config, parent, 37, tmp_path/"branch", arm="I0", parent_path=path)
    assert result["complete"] and len(result["steps"]) == 40 and len(policy.requests) == 3
    assert result["branch"]["replay_matched"] and result["branch"]["reproduced_next_action"]
    assert not result["branch"]["non_deterministic"]
    assert all(s["policy_metadata"]["policy"] == "prefix_replay" for s in result["steps"][:37])
    assert len((tmp_path/"branch/events.jsonl").read_text().splitlines()) == 41
    assert "SECRET" not in json.dumps(policy.requests)
    with pytest.raises(ValueError):
        run_branch(env(), FakeStudent(), task, config, parent, 37, tmp_path/"branch", parent_path=path)


@pytest.mark.parametrize("mutate", ["visible", "audit_sha256", "game_sha256", "reward"])
def test_mismatch_refuses_continuation(branch_parent, tmp_path, mutate):
    _, env, task, config, parent, path = branch_parent
    parent = deepcopy(parent)
    if mutate == "game_sha256":
        parent[mutate] = "wrong"
    elif mutate == "visible":
        parent["steps"][0][mutate]["observation"] = "wrong"
    else:
        parent["steps"][0][mutate] = "wrong"
    policy = FakeStudent()
    with pytest.raises(RuntimeError):
        run_branch(env(), policy, task, config, parent, 2, tmp_path/"mismatch", parent_path=path)
    assert policy.requests == []
    recorded = json.loads((tmp_path/"mismatch/episode.json").read_text())
    assert not recorded["branch"]["replay_matched"]


def test_parser_prefix_replays_without_real_step(branch_parent, tmp_path):
    _, env, task, config, _, _ = branch_parent
    parent = run_episode(env(), ScriptedPolicy(["bad\nreply", "look"]), task, config, tmp_path/"parser-parent",
                         policy_condition="fake", purpose="diagnosis")
    result = run_branch(env(), FakeStudent(), task, config, parent, 3, tmp_path/"parser-branch")
    assert result["steps"][0]["environment_steps"] == 0
    assert result["steps"][2]["visible"]["observation"] == PARSER_FEEDBACK
    assert result["metrics"]["environment_steps"] == 38


@pytest.mark.parametrize("arm", ["I1", "I2", "I5"])
def test_intervention_once_history_persists_and_information_cost(branch_parent, tmp_path, arm):
    _, env, task, config, parent, path = branch_parent
    seed = deepcopy(parent)
    seed.update(task_id=TASK_ID.replace("valid_seen", "train"), split="train", success=True)
    policy = FakeStudent()
    result = run_branch(env(), policy, task, config, parent, 37, tmp_path/arm,
                        arm=arm, parent_path=path, example={"order": 0, "episode": seed})
    assert len(result["steps"]) == 40
    if arm == "I2":
        assert len(policy.requests) == 2
        assert result["steps"][37]["action_parsed"] == "examine drawer 1"
        assert result["steps"][37]["environment_steps"] == 1
        assert result["steps"][37]["visible"]["observation"] in policy.requests[0][-1]["content"]
    else:
        tag = "Facts so far:" if arm == "I1" else "Example of a solved task:"
        assert len(policy.requests) == 3
        for messages in policy.requests:
            assert sum(tag in m["content"] for m in messages) == 1
            assert tag in messages[1+2*37]["content"]
        assert policy.requests[-1][-1]["content"] == build_messages(
            [{"visible": s["visible"], "reply": "Action: look"} for s in parent["steps"][:39]],
            task, result["steps"][38]["visible"])[-1]["content"]


def test_control_detects_action_and_outcome_non_determinism(branch_parent, tmp_path):
    _, env, task, config, parent, _ = branch_parent
    result = run_branch(env(), FakeStudent("win"), task, config, parent, 4, tmp_path/"noise", arm="I0")
    assert result["success"] is True and result["branch"]["non_deterministic"]
    assert not result["branch"]["reproduced_next_action"]


def test_branch_refuses_sealed_before_output(branch_parent, tmp_path):
    _, env, task, config, parent, _ = branch_parent
    parent = dict(parent, task_id=TASK_ID.replace("valid_seen", "valid_unseen"))
    with pytest.raises(ValueError):
        run_branch(env(), FakeStudent(), task, config, parent, 1, tmp_path/"sealed")
    assert not (tmp_path/"sealed").exists()


def test_sampled_student_uses_exact_template_cap_and_seed(tmp_path):
    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"model": "student", "choices": [{"message": {"content": "Action: look"},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 8, "completion_tokens": 3}})
    for index in (0, 1):
        student = BranchStudent(endpoint="http://fake/v1", model="student", revision="pinned",
            enable_thinking=False, log_dir=tmp_path/str(index), sample_index=index, sampling_seed=17,
            transport=httpx.MockTransport(handler))
        student.generate([], {"instruction": ""}, visible())
        student.close()
    assert bodies[0]["temperature"] == 0 and "seed" not in bodies[0]
    assert bodies[1]["temperature"] == .7 and bodies[1]["seed"] == 17
    for body in bodies:
        assert body["messages"] == build_messages([], {}, visible())
        assert body["max_tokens"] == 1024 and body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "logprobs" not in body


class FakeResponses:
    def __init__(self, raw="Action: look", errors=0):
        self.raw, self.errors, self.calls = raw, errors, []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            self.errors -= 1
            raise TimeoutError("SECRET should never persist")
        return SimpleNamespace(output_text=self.raw, model="gpt-6-luna", id="fake", status="completed",
            usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5), _request_id="request")


def fake_teacher(tmp_path, api, *, sample_index=0, cap=20, advice=False, namespace="probe", episode_key="parent"):
    budget = TeacherBudget(tmp_path/"budget.json", max_episode=cap, max_run=cap)
    return BranchTeacher(namespace=namespace, sample_index=sample_index, budget=budget,
        episode_key=episode_key, advice=advice, cache_dir=tmp_path/"cache", usage_ledger=tmp_path/"usage.jsonl",
        run_id="fake", client=SimpleNamespace(responses=api), sampling={"max_output_tokens": 8},
        sleep=lambda _: None, jitter=lambda *_: 0)


def test_teacher_sample_keys_cache_and_persistent_caps(tmp_path):
    api = FakeResponses()
    task = {"task_id": TASK_ID, "split": "dev"}
    one = fake_teacher(tmp_path, api, sample_index=0, cap=2)
    first = one.generate([], task, visible())
    assert not first.metadata["cache_hit"]
    assert one.generate([], task, visible()).metadata["cache_hit"]
    two = fake_teacher(tmp_path, api, sample_index=1, cap=2)
    assert two.cache_key([], task, visible()) != one.cache_key([], task, visible())
    two.generate([], task, visible())
    assert len(api.calls) == 2
    with pytest.raises(BudgetExceeded):
        fake_teacher(tmp_path, api, sample_index=2, cap=2).generate([], task, visible())
    assert len(api.calls) == 2
    assert all("logprobs" not in c and "SECRET" not in json.dumps(c) for c in api.calls)
    assert json.loads((tmp_path/"budget.json").read_text())["total"] == 2


def test_teacher_retry_counts_towards_caps_and_sanitizes(tmp_path):
    api = FakeResponses(errors=10)
    teacher = fake_teacher(tmp_path, api, cap=2)
    with pytest.raises(BudgetExceeded):
        teacher.generate([], {"task_id": TASK_ID, "split": "dev"}, visible())
    assert len(api.calls) == 2
    assert "SECRET" not in (tmp_path/"usage.jsonl").read_text()


def test_teacher_per_episode_and_per_run_caps(tmp_path):
    budget = TeacherBudget(tmp_path/"budget.json", max_episode=1, max_run=2)
    budget.reserve("a", {})
    with pytest.raises(BudgetExceeded):
        budget.reserve("a", {})
    budget.reserve("b", {})
    with pytest.raises(BudgetExceeded):
        budget.reserve("c", {})
    with pytest.raises(ValueError):
        TeacherBudget(tmp_path/"budget.json", max_episode=2, max_run=2).read()


@pytest.mark.parametrize("arm,raw", [("I3", "Take the object, then heat it."), ("I4", "Action: win")])
def test_advice_action_shared_across_samples_and_no_oracle(branch_parent, tmp_path, arm, raw):
    _, env, task, config, parent, _ = branch_parent
    api = FakeResponses(raw)
    wrapped = SharedInterventionTeacher(fake_teacher(tmp_path/"teacher", api, advice=arm=="I3"),
                                       tmp_path/"shared.json", arm=="I3")
    for index in (0, 1):
        student = FakeStudent()
        result = run_branch(env(), student, task, config, parent, 38, tmp_path/f"{arm}-{index}",
            arm=arm, teacher=wrapped, sample_index=index)
        if arm == "I3":
            assert len(student.requests) == 2 and "Advice:" in student.requests[1][1+2*38]["content"]
        else:
            assert result["success"] is True and len(student.requests) == 0
    assert len(api.calls) == 1
    prompt = json.dumps(api.calls[0]["input"])
    assert TASK_ID not in prompt and "internal_facts" not in prompt
    if arm == "I3":
        assert ADVICE_INSTRUCTION in api.calls[0]["input"][-1]["content"]
        assert api.calls[0]["input"][0]["content"] == ADVICE_PROMPT_TEMPLATE
        assert "end with 'Action:" not in api.calls[0]["input"][0]["content"]
        recorded = json.loads((tmp_path/"I3-0/intervention.json").read_text())
        assert recorded["prompt_template_sha256"] == digest(ADVICE_PROMPT_TEMPLATE)
    events = [json.loads(s) for s in (tmp_path/f"{arm}-0/events.jsonl").read_text().splitlines()]
    assert sum(e["teacher_tokens_in"] for e in events) == 3


def test_advice_action_rejected_and_retained(branch_parent, tmp_path):
    _, env, task, config, parent, _ = branch_parent
    teacher = fake_teacher(tmp_path/"teacher", FakeResponses("Action: win"), advice=True)
    with pytest.raises(RuntimeError):
        run_branch(env(), FakeStudent(), task, config, parent, 1, tmp_path/"bad-advice", arm="I3", teacher=teacher)
    assert (tmp_path/"bad-advice/intervention.json").exists()
    assert (tmp_path/"teacher/usage.jsonl").exists()


def test_fce_scan_refines_first_interval_and_never_guesses_unknown():
    calls = []
    def probe(t):
        calls.append(t)
        return t < 6
    result = locate_fce(19, probe)
    assert result["fce"] == 5 and calls == [0, 4, 8, 5, 6]
    assert result["coarse_interval"] == [4, 8]
    assert locate_fce(8, lambda _: False)["status"] == "teacher_unsolvable_from_start"
    assert locate_fce(7, lambda _: True)["status"] == "no_drop_detected"
    assert locate_fce(8, lambda t: True if t == 0 else None)["status"] == "incomplete"
    assert probe_points(7) == [0, 4, 7]
    assert dry_plan(40)["scan_policy_call_upper_bound"] == 440


def test_immutable_branch_receipt_and_resume_hashes(branch_parent, tmp_path):
    _, env, task, config, parent, _ = branch_parent
    attempt = tmp_path/"probe/attempt-000"
    ep = run_branch(env(), FakeStudent("win"), task, config, parent, 0, attempt/"episode")
    receipt = seal_branch(attempt, ep)
    assert completed_branch(attempt.parent) == receipt
    (attempt/"episode/events.jsonl").write_text("changed")
    with pytest.raises(ValueError):
        completed_branch(attempt.parent)


def test_output_and_workspace_ownership(tmp_path):
    with pytest.raises(ValueError):
        output_guard("data/verified/do-not-create")
    run = tmp_path/"existing"
    run.mkdir()
    write_json(run/"provenance.json", {"command": "suite"})
    with pytest.raises(ValueError):
        output_guard(run/"new-child")
    config = load_config()
    own = tmp_path/"owned"
    workspace(own, config=config, command="test", identity={"cap": 1})
    workspace(own, config=config, command="test", identity={"cap": 1})
    with pytest.raises(ValueError):
        workspace(own, config=config, command="test", identity={"cap": 2})


def test_checkpoint_and_exact_training_collection_binding(tmp_path):
    config = load_config()
    verified = tmp_path/"verified"
    verified.mkdir()
    write_json(verified/"manifest.json", {"K": 5})
    training = tmp_path/"training"
    training.mkdir()
    write_json(training/"provenance.json", {"config": {"sft": {"verified": str(verified)}},
        "inputs": {str(verified/"manifest.json"): file_hash(verified/"manifest.json")}})
    identity = {"training_provenance_sha256": file_hash(training/"provenance.json"), "adapter_weights_sha256": "pinned"}
    config["lora_equivalence"] = {"identity": {"adapter": identity}}
    source = tmp_path/"source"
    source.mkdir()
    write_json(source/"provenance.json", {"config": {"lora_equivalence": config["lora_equivalence"],
        "suite": {"student": {"model": "S"}}, "server_provenance": {"config": {"server": {
            "lora_adapters": [{"name": "S", "path": str(training/"checkpoint")}]
        }}}}})
    validate_continuation(config, {"model": "S"}, source/"episode.json", verified=verified)
    validate_continuation(config, {"model": config["models"]["student"]["requested_repo"]},
                          source/"episode.json", verified=verified, subject="base")
    write_json(verified/"manifest.json", {"K": 20})
    with pytest.raises(ValueError, match="actual training set"):
        validate_continuation(config, {"model": "S"}, source/"episode.json", verified=verified)
    config["lora_equivalence"]["identity"]["adapter"] = {"adapter_weights_sha256": "other"}
    with pytest.raises(ValueError, match="checkpoint differs"):
        validate_continuation(config, {"model": "S"}, source/"episode.json")


def test_rescue_wilson_decision_table_and_paired_comparisons():
    def samples(outcomes):
        return [{"sample_index": i, "success": x} for i, x in enumerate(outcomes)]
    assert arm_outcome(samples([True, True, True, True, False]))["rescue"] is True
    assert arm_outcome(samples([False, True, True, True, True]))["rescue"] is False
    assert arm_outcome(samples([True, True, True, False, False]))["rescue"] is False
    assert arm_outcome(samples([True, True, True, "unknown", False]))["rescue"] is None
    assert wilson(0, 0)["ci95"] == [None, None]
    assert wilson(0, 10)["ci95"][1] == pytest.approx(.2775328)
    for experience, information, expected in [(True, True, "both"), (True, False, "M1_cell"),
            (False, True, "M2_cell"), (False, False, "neither")]:
        arms = dict.fromkeys(ARMS_FOR_TEST, False)
        arms.update(I5=experience, I3=information)
        assert decision_cell(arms) == expected
    assert paired_mcnemar({"a": True, "b": False}, {"a": False, "b": True})["exact_two_sided_p"] == 1


ARMS_FOR_TEST = [f"I{i}" for i in range(6)]


def test_readout_coverage_format_and_unknown_stay_separate():
    ep, audits = episode(["look"])
    row = classify(ep, audits)
    tax = {"G": {"task_ids": [TASK_ID]}, "runs": {"student": {"episodes": [row], "shares": shares([row])}}}
    fce = {"episodes": [{"task_id": TASK_ID, "status": "fce_found", "fce": 0, "episode_length": 1}]}
    arms = {arm: {"samples": [{"sample_index": i, "success": arm == "I1"} for i in range(5)]} for arm in ARMS_FOR_TEST}
    interventions = {"episodes": [{"task_id": TASK_ID, "arms": arms, "taxonomy_labels": [], "task_type_covered": True}]}
    result = build_readout(tax, fce, interventions)
    assert result["episodes"][0]["label"] == "M1_requires_coverage"
    assert result["decision_cells"]["M1_cell"]["count"] == 1
    result = build_readout(tax, fce, interventions, {TASK_ID: {"covered": True, "evidence": "seed take/heat pattern"}})
    assert result["episodes"][0]["label"] == "M1"
    interventions["episodes"][0]["taxonomy_labels"] = ["F-format"]
    tax["runs"]["student"]["episodes"][0]["labels"] = ["F-format"]
    result = build_readout(tax, fce, interventions)
    assert result["episodes"] == [] and result["exclusions"]["F-format_separate"] == 1
