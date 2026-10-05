"""Offline trait rules on the Stage 4a real typed-fact lists and fake resets."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from analysis import traits
from analysis.mechanism.goals import Facts, parse_goal
from sed.manifest import freeze_splits
from sed.storage import BlobStore, digest, write_json


REAL = json.loads(Path("tests/fixtures/mechanism_real_facts.json").read_text())
OBS = ("You are in the middle of a room. Looking quickly around you, you see a cabinet 2, "
       "a cabinet 1, a countertop 1, and a drawer 1.\n\nYour task is to: put a mug in cabinet 3.")


@pytest.mark.parametrize("index,family,transform,objects", [(0, "look", None, 1), (1, "transform", "heat", 2),
                                                           (2, "transform", "clean", 1)])
def test_real_fact_types_counts_and_transforms(index, family, transform, objects):
    row = REAL[index]
    result = traits.initial_traits(row["task_id"], OBS, row["facts"])
    assert result["family"] == family
    assert result["transformation"] == transform
    assert result["state_change"] == ("yes" if transform else "no")
    assert result["goal_object_count"] == objects
    assert result["distractors"] == objects - 1
    assert result["room_size"] == 4


@pytest.mark.parametrize("kind,family,transform,required", [
    ("pick_and_place_simple", "simple", None, 1), ("look_at_obj_in_light", "look", None, 1),
    ("pick_heat_then_place_in_recep", "transform", "heat", 1),
    ("pick_cool_then_place_in_recep", "transform", "cool", 1),
    ("pick_clean_then_place_in_recep", "transform", "clean", 1),
    ("pick_two_obj_and_place", "multi-object", None, 2)])
def test_all_six_templates_on_real_mug_facts(kind, family, transform, required):
    task = f"valid_seen/{kind}-Mug-None-Cabinet-3/trial_0"
    result = traits.initial_traits(task, OBS, REAL[1]["facts"])
    assert (result["family"], result["transformation"], result["required_object_count"]) == (family, transform, required)
    assert result["distractors"] == 2 - required


def located_mugs():
    # Controlled variants of a full real fact list; only mug containment and
    # receptacle opened predicates change. The original Stage 4a lists are
    # post-action snapshots, used here as rule fixtures, never as dev initial states.
    row = REAL[1]
    parsed = Facts(row["facts"])
    mugs = sorted(o for o, t in parsed.types.items() if t == "mug")
    cabinet = next(r for r, in sorted(parsed.get("openable")) if parsed.types.get(r) == "cabinet")
    countertop = next(r for r, t in sorted(parsed.types.items()) if t == "countertop")
    facts = [text for (name, args), text in parsed.rows
             if not (name == "inreceptacle" and args[0] in mugs)
             and not (name == "opened" and args[0] == cabinet)]
    facts += [f"inreceptacle({o}: o, {cabinet}: r)" for o in mugs]
    return row["task_id"], facts, mugs, cabinet, countertop


def test_hidden_requires_every_goal_object_closed_and_openable():
    task, facts, mugs, cabinet, countertop = located_mugs()
    assert traits.initial_traits(task, OBS, facts)["hidden_target"] == "yes"
    assert traits.initial_traits(task, OBS, facts + [f"opened({cabinet}: r)"])["hidden_target"] == "no"
    nonopenable = [text for (name, args), text in Facts(facts).rows
                   if not (name == "openable" and args[0] == cabinet)]
    assert traits.initial_traits(task, OBS, nonopenable)["hidden_target"] == "no"
    mixed = [text for (name, args), text in Facts(facts).rows
             if not (name == "inreceptacle" and args[0] == mugs[0])]
    mixed += [f"inreceptacle({mugs[0]}: o, {countertop}: r)"]
    assert traits.initial_traits(task, OBS, mixed)["hidden_target"] == "no"
    unlocated = [text for (name, args), text in Facts(facts).rows
                 if not (name == "inreceptacle" and args[0] == mugs[0])]
    assert traits.initial_traits(task, OBS, unlocated)["hidden_target"] == "no"


def test_distractors_include_unplaced_objects_not_type_constants_or_duplicate_facts():
    row = REAL[1]
    facts = row["facts"] + ["objectType(extra_mug: o, MugType: t)"]
    result = traits.initial_traits(row["task_id"], OBS, facts * 2)
    assert result["goal_object_count"] == 3
    assert (result["distractors"], result["distractors_bucket"]) == (2, ">=2")


def test_room_count_is_observation_inventory_only():
    assert len(traits.room_receptacles(OBS + "\nOn countertop 1, you see a mug 1.")) == 4
    assert traits.room_receptacles("Looking quickly around you, you see an ottoman 1, a desk 12, and a desk 12.") == ["desk 12", "ottoman 1"]
    with pytest.raises(ValueError, match="room receptacle"):
        traits.room_receptacles("Your task is to: put mug 1 on desk 1.")


def test_tertiles_ties_boundaries_unknown_and_empty():
    values = [1, 1, 1, 2, 2, 3, None]
    cuts = traits.tertiles(values)
    assert cuts["cutpoints"] == [1, 2]
    assert [traits.tertile_bucket(v, cuts) for v in values] == ["low"] * 3 + ["middle"] * 2 + ["high", "unknown"]
    tied = traits.tertiles([2] * 140)
    assert tied["cutpoints"] == [2, 2]
    assert traits.tertile_bucket(2, tied) == "low"
    assert traits.tertiles([None])["cutpoints"] == []
    assert traits.tertile_bucket(None, traits.tertiles([])) == "unknown"


def test_coverage_counts_trusted_trajectories_not_unique_tasks_or_attempts():
    task = REAL[1]["task_id"].replace("valid_seen/", "train/")
    receipt = {"task_id": task, "won": True, "task_type": parse_goal(task).task_type}
    collection = {"status": "complete", "K": 2, "verified": [receipt, dict(receipt)],
                  "attempts": [receipt] * 20}
    counts = traits.coverage_counts(collection)
    assert counts[receipt["task_type"]] == 2
    assert sum(counts.values()) == 2
    assert [traits.count_bucket(v) for v in (0, 1, 2, 10)] == ["0", "1", ">=2", ">=2"]
    collection["verified"][0] = dict(receipt, won=False)
    with pytest.raises(ValueError, match="trusted"):
        traits.coverage_counts(collection)


@pytest.mark.parametrize("prefix", ["valid_unseen", "test", "train", "../valid_seen", "/valid_seen"])
def test_sealed_and_non_dev_refused(prefix):
    with pytest.raises(ValueError):
        traits.initial_traits(REAL[1]["task_id"].replace("valid_seen", prefix), OBS, REAL[1]["facts"])


def write_episode(root, name, task, length, success, *, complete=True, audit_won=None):
    directory = root / "episodes" / name
    audit = {"won": success if audit_won is None else audit_won}
    verification = {"audit": audit, "fact_source": "real_environment"}
    ref = BlobStore(directory / "blobs").put(verification)
    step = {"verification_ref": ref, "audit_sha256": digest(audit)}
    episode = {"task_id": task, "split": "dev", "sealed": False, "game_sha256": "game", "seed": 0,
               "complete": complete, "status": "ok", "success": success, "steps": [step] * length,
               "initial": step}
    write_json(directory / "episode.json", episode)


def test_teacher_shortest_success_ignores_failures_and_incomplete(tmp_path):
    a, b, c = (r["task_id"] for r in REAL)
    games = {t: {"game_sha256": "game", "environment_seed": 0} for t in (a, b, c)}
    for name, task, length, success, complete in [("long", a, 12, True, True), ("short", a, 8, True, True),
                                                ("fail", a, 1, False, True), ("pending", a, 2, True, False),
                                                ("failed_only", b, 40, False, True)]:
        write_episode(tmp_path, name, task, length, success, complete=complete)
    result = traits.teacher_lengths(tmp_path, games, traits.Inputs())
    assert result[a]["steps"] == 8
    assert result[a]["successful_trajectories"] == 2
    assert result[a]["observed_trajectories"] == 4
    assert result[b]["steps"] is None and result[c]["steps"] is None
    write_episode(tmp_path, "false_claim", a, 1, True, audit_won=False)
    with pytest.raises(ValueError, match="environment evidence"):
        traits.teacher_lengths(tmp_path, games, traits.Inputs())


def test_freeze_hash_tamper_and_strict_overwrite(tmp_path):
    path = tmp_path / "traits.json"
    value = traits.freeze_traits(path, {"schema_version": 1})
    before = path.read_bytes()
    assert value["traits_sha256"] == digest({"schema_version": 1})
    with pytest.raises(FileExistsError):
        traits.freeze_traits(path, {"schema_version": 1})
    assert path.read_bytes() == before
    write_json(path, dict(value, schema_version=2))
    with pytest.raises(ValueError, match="hash mismatch"):
        traits.load_traits(path)


def test_builder_resets_once_no_policy_steps_and_rejects_existing_before_reset(tmp_path, monkeypatch):
    tasks = []
    root = tmp_path / "games"
    for i in range(140):
        task_id = f"valid_seen/pick_and_place_simple-Bowl-None-Desk-{i}/trial_0"
        game = root / task_id / "game.tw-pddl"
        write_json(game, {"solvable": True})
        sha = traits.file_hash(game)
        tasks.append({"task_id": task_id, "group_id": task_id.rsplit("/", 1)[0], "source_split": "valid_seen",
                      "task_type": "pick_and_place_simple", "eligible": True,
                      "game_sha256": sha, "files": {"game.tw-pddl": sha}})
    manifest = {"tasks": tasks}
    freeze = freeze_splits(manifest)
    listing = traits.build_dev140(manifest, freeze)
    for name, value in [("manifest.json", manifest), ("freeze.json", freeze), ("dev.json", listing)]:
        write_json(tmp_path / name, value)
    config = {"paths": {"manifest": str(tmp_path / "manifest.json"), "split_freeze": str(tmp_path / "freeze.json")},
              "environment": {"data_root": str(root), "seed": 0, "max_episode_steps": 40}}
    resets = []

    class FakeEnvironment:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, task_id):
            resets.append(task_id)
            self.game_hash = tasks[0]["game_sha256"]
            return {"observation": OBS}, 0, False, False, {"internal_facts": deepcopy(REAL[0]["facts"])}

        def step(self, *args):
            pytest.fail("Traits must never execute actions")

        def close(self):
            pass

    monkeypatch.setattr(traits, "teacher_lengths", lambda directory, games, inputs:
                        {t: {"steps": None} for t in games})
    monkeypatch.setattr(traits, "seed_coverage", lambda *args: {})
    output = tmp_path / "traits.json"
    kwargs = dict(config=config, tasks_path=tmp_path / "dev.json", teacher=tmp_path / "teacher",
                  verified_root=tmp_path / "seeds", output=output, env_factory=FakeEnvironment)
    result = traits.build_traits(**kwargs)
    assert resets == listing["task_ids"]
    assert traits.load_traits(output) == result
    assert result["provenance"]["settings"]["environment_steps"] == 0
    with pytest.raises(FileExistsError):
        traits.build_traits(**kwargs)
    assert len(resets) == 140
