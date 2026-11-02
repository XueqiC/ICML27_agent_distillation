"""Offline CPU tests. Real games are executed by the CLI spike, never unit tests."""

from copy import deepcopy
import json
from pathlib import Path
import random

import pytest

from repair import synth_tasks as synth
from repair.synth_validation import Execution, choose_error, validate_game, validate_jobs, validate_prefix


TEMPLATES = json.loads((Path(__file__).parent / "fixtures/synth_tasks/seed_templates.json").read_text())["templates"]


def small_game(task_type="pick_and_place_simple"):
    template = TEMPLATES[task_type]
    return {"pddl_domain": "fixture-domain", "pddl_problem": """
(define (problem miniature) (:domain fixture)
(:objects mug1 mug2 bowl1 - object table1 table2 fridge1 microwave1 sink1 - receptacle)
(:init
 (objectType mug1 MugType) (objectType mug2 MugType) (objectType bowl1 BowlType)
 (pickupable mug1) (pickupable mug2) (pickupable bowl1)
 (cleanable mug1) (heatable mug1) (coolable mug1)
 (objectType lamp1 DeskLampType) (toggleable lamp1)
 (receptacleType table1 CounterTopType) (receptacleType table2 CounterTopType)
 (receptacleType fridge1 FridgeType) (receptacleType microwave1 MicrowaveType)
 (receptacleType sink1 SinkBasinType)
 (receptacleAtLocation table1 loc1) (receptacleAtLocation table2 loc2)
 (receptacleAtLocation fridge1 loc3) (receptacleAtLocation microwave1 loc4)
 (receptacleAtLocation sink1 loc5)
 (canContain CounterTopType MugType) (canContain FridgeType MugType)
 (inReceptacle mug1 table1) (objectAtLocation mug1 loc1)
 (inReceptacle mug2 table1) (objectAtLocation mug2 loc1)
 (inReceptacle bowl1 table1) (objectAtLocation bowl1 loc1)
 (inReceptacle lamp1 table2) (objectAtLocation lamp1 loc2)
 (openable fridge1) (atLocation agent1 loc1))
""" + template["goal"] + ")",
        "grammar": 'grammar :: """\n{"task": [{"rhs": ' + json.dumps(template["task_text"]) + '}]}\n"""',
        "solvable": True, "walkthrough": ["SECRET_OLD_PLAN"]}


def template_seeds():
    return [synth.Seed(row["source_task_id"], small_game(task), [], {}) for task, row in TEMPLATES.items()]


def test_relocate_consistent_and_immutable():
    game = small_game()
    original = deepcopy(game)
    edited = synth.relocate(game, "mug1", "fridge1")
    before, after = set(synth.init_facts(game)), set(synth.init_facts(edited))
    assert before - after == {("inReceptacle", "mug1", "table1"), ("objectAtLocation", "mug1", "loc1")}
    assert after - before == {("inReceptacle", "mug1", "fridge1"), ("objectAtLocation", "mug1", "loc3")}
    assert ("openable", "fridge1") in after and ("opened", "fridge1") not in after
    assert synth.section(game["pddl_problem"], "goal") == synth.section(edited["pddl_problem"], "goal")
    assert game == original
    assert edited["walkthrough"] == [] and not edited["solvable"]


@pytest.mark.parametrize("fact,reason", [
    ("(inReceptacleObject mug1 bowl1)", "unsupported_object_dependency"),
    ("(inReceptacleObject bowl1 mug1)", "unsupported_object_dependency"),
    ("(wasInReceptacle mug1 table1)", "unsupported_object_dependency"),
    ("(mysteryLocation mug1 loc1)", "unsupported_object_dependency"),
    ("(full fridge1)", "full_receptacle"),
    ("(objectAtLocation mug1 loc2)", "ambiguous_location"),
])
def test_relocate_fails_closed_on_dependencies(fact, reason):
    game = small_game()
    initial = synth.section(game["pddl_problem"], "init")
    game["pddl_problem"] = synth.replace_section(game["pddl_problem"], "init", initial[:-1] + fact + ")")
    with pytest.raises(ValueError, match=reason):
        synth.relocate(game, "mug1", "fridge1")


def test_inconsistent_source_location_rejected():
    game = small_game()
    game["pddl_problem"] = game["pddl_problem"].replace("(objectAtLocation mug1 loc1)", "(objectAtLocation mug1 loc2)")
    with pytest.raises(ValueError, match="inconsistent_parent_location"):
        synth.relocate(game, "mug1", "fridge1")


@pytest.mark.parametrize("task_type", synth.TASK_TYPES)
def test_goal_forms_and_task_language_are_exact_seed_substitutions(task_type):
    parent = synth.Seed("train/pick_and_place_simple-Mug-None-CounterTop-1/trial_0", small_game(), [], {})
    template = next(s for s in template_seeds() if synth.group_parts(s.group)[0] == task_type)
    target = "DeskLamp" if task_type == "look_at_obj_in_light" else "CounterTop"
    game, group = synth.goal_edit(parent, template, "Mug", target)
    assert synth.section(game["pddl_problem"], "init") == synth.section(parent.game["pddl_problem"], "init")
    assert synth.group_parts(group) == (task_type, "Mug", "None", target, "1")
    goal = synth.section(game["pddl_problem"], "goal")
    assert "MugType" in goal and target + "Type" in goal
    assert "mug" in synth.task_text(game) and target.lower() in synth.task_text(game)
    assert game["walkthrough"] == []
    assert template.game["walkthrough"] == ["SECRET_OLD_PLAN"]


def test_templates_and_pairs_do_not_expand_beyond_seeds():
    parent = synth.Seed("train/pick_and_place_simple-Mug-None-CounterTop-1/trial_0", small_game(), [], {})
    pairs = synth.legal_pairs([parent])
    assert ("MugType", "FridgeType") in pairs
    assert ("MugType", "MicrowaveType") not in pairs
    rows = list(synth.generate_candidates(parent, [parent], "G1", random.Random(0)))
    assert rows
    assert all(row["edit"]["destination"] not in {"microwave1", "sink1"} for _, row in rows)
    assert [(synth.problem_hash(g), r["edit"]) for g, r in rows] == [
        (synth.problem_hash(g), r["edit"]) for g, r in synth.generate_candidates(parent, [parent], "G1", random.Random(0))]


def test_g2_filters_missing_lamp_capability_and_two_instances():
    parent = synth.Seed("train/pick_and_place_simple-Mug-None-CounterTop-1/trial_0", small_game(), [], {})
    rows = list(synth.generate_candidates(parent, template_seeds(), "G2", random.Random(1)))
    for game, row in rows:
        parts = row["fingerprint"]["tuple"]
        if parts[0] == "pick_two_obj_and_place":
            assert parts[1] == "Mug"
        if parts[0] in {"pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep"}:
            assert parts[1] == "Mug"
        assert synth.init_facts(game) == synth.init_facts(parent.game)


def test_fingerprint_normalizes_init_order_and_whitespace():
    game = small_game()
    other = deepcopy(game)
    facts = synth.init_facts(game)
    other["pddl_problem"] = synth.replace_section(other["pddl_problem"], "init", "(:init\n" +
        "\n".join("(" + "  ".join(f) + ")" for f in reversed(facts)) + ")")
    assert synth.problem_hash(game) == synth.problem_hash(other)
    assert synth.problem_hash(game) != synth.problem_hash(synth.relocate(game, "mug1", "table2"))


def test_split_and_dedupe_checks_ignore_native_split_prefix():
    group = "pick_and_place_simple-Mug-None-CounterTop-1"
    rec = {"fingerprint": synth.fingerprint("train/" + group + "/trial_0", small_game())}
    assert synth.rejection_reason(rec, {synth.group_name("valid_seen/" + group)}, set(), set()) == "heldout_group"
    key = rec["fingerprint"]["problem_sha256"]
    assert synth.rejection_reason(rec, set(), {key}, set()) == "seed_duplicate"
    assert synth.rejection_reason(rec, set(), set(), {key}) == "synthetic_duplicate"


def test_freeze_uses_identity_lists_without_reading_games(tmp_path):
    root, main, repo = (tmp_path / x for x in ("dataset", "main", "repo"))
    groups = ["pick_and_place_simple-Mug-None-CounterTop-" + str(i) for i in range(1, 8)]
    for split, group in zip(("valid_seen", "valid_unseen"), groups):
        game = root / split / group / "trial" / "game.tw-pddl"
        game.parent.mkdir(parents=True)
        game.write_text("HIDDEN_STATE_MUST_NOT_BE_PARSED")
    synth.write_json(main / "data/manifest/dev140.json", {"task_groups": 1, "task_ids": ["valid_seen/" + groups[0] + "/trial"]})
    synth.write_json(repo / "data/manifest/repair_v01_splits.json", {
        "E": {"group_ids": [groups[6]]}, "sets": {"U_mech": {"group_ids": [groups[2]]}}})
    synth.write_json(main / "reports/runs/method_v1/U_dev_listing.json", {"group_ids": ["train/" + groups[3]]})
    synth.write_json(main / "reports/runs/method_v1/v1a_cg7_spec.json", [{"task_id": "train/" + groups[4] + "/trial"}])
    synth.write_json(main / "reports/runs/method_v1/records/hidden.json", {"task_id": groups[5]})
    synth.write_json(main / "reports/runs/method_v1/eval/tasks/run_manifest.json", {
        "tasks": [{"task_id": groups[4]}], "checkpoint": {"training": {"task_ids": [groups[6]]}}})
    frozen = synth.freeze_heldout(root, main, repo)
    assert set(frozen["groups"]) == set(groups[:5])
    assert frozen == synth.freeze_heldout(root, main, repo)


def test_output_cannot_overlap_inputs_or_overwrite(tmp_path):
    for out in (tmp_path, tmp_path / "main/out", tmp_path / "dataset/out"):
        with pytest.raises(ValueError, match="overlaps"):
            synth.safe_output(out, tmp_path / "dataset", tmp_path / "main")
    out = tmp_path / "output"
    out.mkdir()
    (out / "existing").write_text("keep")
    with pytest.raises(ValueError, match="overwrite"):
        synth.safe_output(out, tmp_path / "dataset", tmp_path / "main")


def test_training_whitelist_cannot_leak_witnesses_or_labels():
    record = {"status": "valid", "task_id": "train/synth/x", "generator": "G1",
              "parent_seed_task_id": "train/parent/trial", "task_text": "Visible task",
              "pddl_diff": "SECRET", "witness_path": "SECRET", "validation": {"won": True}, "facts": "SECRET"}
    assert set(synth.training_task_view(record)) == {"task_id", "task_text"}
    record["status"] = "rejected"
    with pytest.raises(ValueError):
        synth.training_task_view(record)


def test_accounting_does_not_count_seed_groups_or_prefixes_as_new_groups():
    original = "pick_and_place_simple-Mug-None-CounterTop-1"
    changed = "pick_and_place_simple-Mug-None-Fridge-1"
    records = [{"generator": generator, "status": "valid", "fingerprint": {"group": group}}
               for generator, group in (("G1", original), ("G1", original), ("G2", changed), ("G3", original))]
    result = synth.summarize_records(records, ["train/" + original + "/trial"], ["G1", "G2", "G3"])["accounting"]
    assert result["K"] == 1 and result["new_executable_tasks"] == 3
    assert result["synthetic_task_groups"] == 2 and result["new_task_groups"] == 1
    assert result["recovery_prefixes"] == 1


def make_seed_receipt(tmp_path):
    root, collection = tmp_path / "dataset", tmp_path / "collection"
    task = "train/pick_and_place_simple-Mug-None-CounterTop-1/trial_0"
    game_path = root / task / "game.tw-pddl"
    synth.write_json(game_path, small_game())
    episode_dir = collection / "attempts/one/episode"
    synth.write_json(episode_dir / "episode.json", {"complete": True, "success": True, "task_id": task,
                                                   "seed": 0, "game_sha256": synth.file_hash(game_path)})
    (episode_dir / "events.jsonl").write_text(json.dumps({"environment_steps": 1, "action_parsed": "look"}) + "\n")
    row = {"task_id": task, "success": True, "won": True, "episode_dir": "attempts/one/episode",
           "episode_hashes": {name: synth.file_hash(episode_dir / name) for name in ("episode.json", "events.jsonl")}}
    manifest = collection / "subsets/k1/manifest.json"
    synth.write_json(manifest, {"K": 1, "verified": [row], "attempts": [row, {"success": False}],
                                "source_collection": {"path": "../.."}})
    return root, manifest, game_path, episode_dir


def test_seed_loader_counts_verified_trajectories_not_attempts(tmp_path):
    root, manifest, _, _ = make_seed_receipt(tmp_path)
    seeds, receipt = synth.load_seeds(manifest, root)
    assert len(seeds) == 1 and len(receipt["attempts"]) == 2
    assert seeds[0].actions == ["look"]


@pytest.mark.parametrize("tamper", ["game", "events", "K", "split"])
def test_seed_loader_requires_receipt_integrity(tmp_path, tamper):
    root, manifest, game, episode = make_seed_receipt(tmp_path)
    if tamper == "game":
        game.write_text("changed")
    elif tamper == "events":
        (episode / "events.jsonl").write_text("changed")
    else:
        content = json.loads(manifest.read_text())
        if tamper == "K":
            content["K"] = 2
        else:
            content["verified"][0]["task_id"] = content["verified"][0]["task_id"].replace("train/", "valid_unseen/")
        synth.write_json(manifest, content)
    with pytest.raises(ValueError):
        synth.load_seeds(manifest, root)


class FakeEnv:
    """Deterministic state machine, including real refusal/no-state-change behavior."""
    def __init__(self, *args, **kwargs):
        self.backend = self
        self.env = self
        self._pddl_state = self
        self._entity_infos = {}

    def reset(self, task):
        self.n = 0
        self.steps = 0
        return self.result("Room.\nYour task is to: put a mug in countertop.")

    def result(self, text):
        return ({"observation": text, "admissible_commands": ["look", "finish", "go to table 1"]},
                float(self.n == 9), self.n == 9, self.steps >= 40,
                {"won": self.n == 9, "lost": False, "internal_facts": [str(self.n)]})

    def step(self, action):
        self.steps += 1
        if action == "finish":
            self.n = 9
        elif action == "go to table 1":
            self.n += 1
        return self.result("Nothing happens." if action.startswith("take nonexistent") else "Visible " + str(self.n))

    def replan(self, infos):
        return ["finish"]

    def _call(self, fn, *args):
        return fn(*args)

    def close(self):
        pass


def test_validator_replays_witness_and_records_coincidental_parent_win(tmp_path):
    job = {"generator": "G2", "task_id": "train/synth/test", "parent_actions": ["finish"],
           "task_text": "Your task is to: put a mug in countertop."}
    execution = Execution(0, FakeEnv)
    result = validate_game(job, tmp_path, tmp_path, execution)
    assert result["won_first"] and result["won_second"] and result["deterministic"]
    assert result["negative_control"] == "coincidentally_wins"
    assert execution.transitions == 3


def test_validator_does_not_infer_success_from_a_plan(tmp_path):
    class BadPlanner(FakeEnv):
        def replan(self, infos):
            return ["look"]
    execution = Execution(0, BadPlanner)
    result = validate_game({"generator": "G1", "task_id": "train/synth/x",
                           "task_text": "Your task is to: put a mug in countertop."}, tmp_path, tmp_path, execution)
    assert result["failure_reason"] == "no_witness_within_40_steps"
    assert not result["won_first"]


def test_nondeterministic_observation_rejects_even_two_wins(tmp_path):
    class DriftingEnv(FakeEnv):
        resets = 0

        def reset(self, task):
            type(self).resets += 1
            result = super().reset(task)
            result[0]["observation"] = str(self.resets) + " " + result[0]["observation"]
            return result
    result = validate_game({"generator": "G1", "task_id": "train/synth/x",
        "task_text": "Your task is to: put a mug in countertop."}, tmp_path, tmp_path, Execution(0, DriftingEnv))
    assert result["won_first"] and result["won_second"]
    assert result["failure_reason"] == "nondeterministic_replay"


def test_controlled_errors_use_admissible_commands_and_wrong_types():
    visible = {"admissible_commands": ["take mug 1 from table 1", "take book 1 from table 1",
        "move mug 1 to countertop 1", "move mug 1 to fridge 1", "look"]}
    assert choose_error("wrong_object", visible, "", "Mug", "CounterTop") == ["take book 1 from table 1"]
    assert choose_error("wrong_receptacle", visible, "", "Mug", "CounterTop") == ["move mug 1 to fridge 1"]
    assert choose_error("useless_search", visible, "", "Mug", "CounterTop") == ["look", "look"]


@pytest.mark.parametrize("crashes", [False, True])
def test_failed_parent_replay_is_retained_and_never_used(tmp_path, crashes):
    class BrokenEnv(FakeEnv):
        def step(self, action):
            if crashes:
                raise RuntimeError("SECRET_EXCEPTION_TOKEN")
            return super().step(action)
    source = tmp_path / "source"
    task = "train/pick_and_place_simple-Mug-None-CounterTop-1/trial"
    synth.write_json(source / task / "game.tw-pddl", {})
    synth.write_json(tmp_path / "validation_requests.json", {"seed": 0, "data_root": str(source), "jobs": [{
        "id": "failed", "parent_seed_task_id": task, "parent_actions": ["look"], "generator": "G3",
        "parent_game_sha256": synth.file_hash(source / task / "game.tw-pddl")}]})
    validate_jobs(tmp_path, env_factory=BrokenEnv)
    result = json.loads((tmp_path / "validation/failed.json").read_text())
    witness = (tmp_path / result["witness_path"]).read_text()
    assert result["status"] == "rejected"
    assert result["rejection"] == ("validation_exception:RuntimeError" if crashes else "original_seed_actions_do_not_win")
    assert "SECRET_EXCEPTION_TOKEN" not in witness
    assert json.loads(witness)["never_for_training"]


def test_prefix_stops_at_error_and_keeps_recovery_in_witness_only(tmp_path):
    execution = Execution(0, FakeEnv)
    task_id = "train/pick_and_place_simple-Mug-None-CounterTop-1/trial_0"
    actions = ["go to table 1", "finish"]
    execution.reset(tmp_path, task_id, "parent")
    execution.replay(actions)
    parent_episode = execution.current
    result = validate_prefix({"id": "prefix", "task_id": task_id, "parent_seed_task_id": task_id,
        "parent_actions": actions, "prefix_index": 0, "error_type": "refused_action",
        "task_text": "Your task is to: put a mug in countertop."}, tmp_path, tmp_path, execution, parent_episode)
    prefix = json.loads((tmp_path / result["prefix_path"]).read_text())
    assert len(prefix["prefix_actions"]) == 1
    assert "finish" not in prefix["prefix_actions"]
    assert all(row["action"] != "finish" for row in prefix["transitions"])
    assert "internal_facts" not in json.dumps(prefix) and "won" not in json.dumps(prefix)
    assert result["won_first"] and result["won_second"] and result["deterministic"]


def test_nested_numeric_init_is_not_silently_dropped():
    game = small_game()
    game["pddl_problem"] = game["pddl_problem"].replace("(:init", "(:init (= (total-cost) 0)")
    with pytest.raises(ValueError, match="Non-atomic"):
        synth.init_facts(game)
