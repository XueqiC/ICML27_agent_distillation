"""Offline paired statistics, coverage partitions and read-only input guards."""
import json

import pytest

from analysis.k_ladder import wilson
from analysis import stratified
from analysis.traits import Inputs
from sed.storage import write_json


def small_traits():
    tasks = [f"valid_seen/pick_and_place_simple-Mug-None-Desk-{i}/trial_0" for i in range(4)]
    games = {t: {"task_type": "pick_and_place_simple", "family": "simple", "group_id": t.rsplit("/", 1)[0],
                 "game_sha256": "game", "environment_seed": 0, "hidden_target": "no", "room_size_bucket": "low",
                 "distractors_bucket": "0", "minimum_length_bucket": "unknown", "state_change": "no",
                 "seed_coverage": {"k5s0": {"bucket": "0" if i < 2 else ">=2"},
                                   "k5s1": {"bucket": "0" if i % 2 == 0 else "1"}}}
             for i, t in enumerate(tasks)}
    return {"traits_sha256": "traits", "task_ids": tasks, "games": games, "task_groups": 4,
            "coverage": {"k5s0": {}, "k5s1": {}}}


def runs_for(traits):
    vectors = {"teacher": [True, True, True, False], "base": [True, False, False, False],
               "k5s0-t400k": [False, True, True, False], "k5s1-eto": [False, False, True, True],
               "baseline": [True, False, True, False]}
    return {label: {"outcomes": dict(zip(traits["task_ids"], values)), "complete": True,
                    "batch_invariant": None if label == "teacher" else True}
            for label, values in vectors.items()}


def test_wilson_and_exact_mcnemar_known_values():
    tasks = list(range(10))
    row = dict.fromkeys(tasks, True)
    base = dict.fromkeys(tasks, False)
    value = stratified.paired_difference(row, base, tasks)
    assert value["difference"] == 1
    assert value["p"] == pytest.approx(2 / 1024)
    assert value["only_a"] == 10 and value["only_b"] == 0
    assert 0 < value["difference95"][0] < value["difference95"][1] <= 1
    reverse = stratified.paired_difference(base, row, tasks)
    assert reverse["difference95"] == pytest.approx([-value["difference95"][1], -value["difference95"][0]])
    summary = stratified.success_summary({0: True, 1: False, 2: True, 3: False}, list(range(4)))
    assert summary["wilson95"] == pytest.approx(wilson(2, 4))
    assert summary["wilson95"] == pytest.approx([0.1500357088, 0.8499642912])


def test_pairs_use_intersection_not_marginal_denominators_and_preserve_unknown():
    row = {"a": True, "b": False, "c": None, "d": True}
    reference = {"a": False, "b": None, "c": True}
    ids = list("abcde")
    value = stratified.paired_difference(row, reference, ids)
    assert value["n"] == 1 and value["n_unpaired"] == 4
    assert value["task_ids"] == ["a"] and value["difference"] == 1
    summary = stratified.success_summary(row, ids)
    assert (summary["n_games"], summary["n_known"], summary["n_missing"], summary["n_unknown"]) == (5, 3, 1, 1)
    assert summary["rate"] == 2 / 3
    empty = stratified.paired_difference({}, {}, ids)
    assert empty["p"] is None and empty["difference"] is None and empty["difference95"] == [None, None]
    assert stratified.success_summary({}, [])["wilson95"] == [None, None]
    assert stratified.paired_difference(row, row, ["a", "b"])["p"] == 1
    assert stratified.paired_difference(row, row, ["a", "b"])["difference"] == 0


def test_coverage_matched_to_each_row_references_share_exact_same_games():
    traits = small_traits()
    result = stratified.build_readout(traits, runs_for(traits))
    zero = {r["coverage_context"]: r for r in result["buckets"] if r["trait"] == "seed_coverage" and r["bucket"] == "0"}
    assert set(zero) == {"k5s0", "k5s1"}
    assert zero["k5s0"]["task_ids"] == traits["task_ids"][:2]
    assert zero["k5s1"]["task_ids"] == traits["task_ids"][::2]
    assert set(zero["k5s0"]["rows"]) == {"teacher", "base", "k5s0-t400k"}
    assert set(zero["k5s1"]["rows"]) == {"teacher", "base", "k5s1-eto"}
    for bucket in zero.values():
        for row in bucket["rows"].values():
            assert row["vs_base"]["task_ids"] == bucket["task_ids"]
            assert row["vs_teacher"]["task_ids"] == bucket["task_ids"]
            assert row["gap_teacher_minus_row"] == -row["vs_teacher"]["difference"]
    assert zero["k5s0"]["rows"]["k5s0-t400k"]["gap_teacher_minus_row"] == .5
    assert result["rows"]["baseline"]["coverage_context"] is None
    text = stratified.markdown(result)
    assert "12/140" in text and "exact" in text and "NA" in text
    json.dumps(result, allow_nan=False)


def test_partition_totals_empty_buckets_and_unknown_length():
    traits = small_traits()
    result = stratified.build_readout(traits, runs_for(traits))
    for trait in stratified.BUCKETS:
        if trait != "seed_coverage":
            assert sum(r["n_games"] for r in result["buckets"] if r["trait"] == trait) == 4
    bucket = next(r for r in result["buckets"] if r["trait"] == "minimum_length" and r["bucket"] == "unknown")
    assert bucket["n_games"] == 4
    empty = next(r for r in result["buckets"] if r["trait"] == "hidden_target" and r["bucket"] == "yes")
    assert empty["n_games"] == 0
    assert empty["rows"]["base"]["vs_teacher"]["p"] is None


def test_unfrozen_coverage_and_mixed_serving_refused():
    traits = small_traits()
    with pytest.raises(ValueError, match="no frozen coverage"):
        stratified.coverage_context("k10s2-augmented-sft", traits)
    runs = runs_for(traits)
    runs["baseline"]["batch_invariant"] = None
    with pytest.raises(ValueError, match="Mixed batch_invariant"):
        stratified.build_readout(traits, runs)
    assert stratified.build_readout(traits, runs, allow_mixed_serving=True)["mixed_serving"] is True


def eval_episode(traits, index=0, **kwargs):
    return {"task_id": traits["task_ids"][index], "split": "dev", "sealed": False, "game_sha256": "game",
            "seed": 0, "complete": True, "status": "ok", "success": True, "steps": [], **kwargs}


def test_load_eval_reuses_ladder_and_does_not_turn_unknown_into_failure(tmp_path, monkeypatch):
    traits = small_traits()
    for index, kwargs in enumerate([{}, {"success": None}, {"status": "error", "success": False}]):
        write_json(tmp_path / "episodes" / str(index) / "episode.json", eval_episode(traits, index, **kwargs))
    called = []
    original = stratified.load_run

    def load(path):
        called.append(path)
        return original(path)

    monkeypatch.setattr(stratified, "load_run", load)
    inputs = Inputs()
    result = stratified.load_eval(tmp_path, traits, inputs)
    assert called == [tmp_path]
    assert list(result["outcomes"].values()).count(None) == 2
    assert len(result["outcomes"]) == 3
    assert result["outcomes"][traits["task_ids"][0]] is True
    inputs.verify()


@pytest.mark.parametrize("change", [{"sealed": True}, {"split": "test"}, {"seed": 1}, {"game_sha256": "different"},
                                    {"task_id": "valid_unseen/pick_and_place_simple-Mug-None-Desk-1/trial_0"}])
def test_eval_refuses_sealed_or_unmatched_state(tmp_path, change):
    traits = small_traits()
    write_json(tmp_path / "episodes/0/episode.json", eval_episode(traits, **change))
    with pytest.raises(ValueError):
        stratified.load_eval(tmp_path, traits, Inputs())


def test_duplicate_eval_task_refused(tmp_path):
    traits = small_traits()
    for index in range(2):
        write_json(tmp_path / "episodes" / str(index) / "episode.json", eval_episode(traits))
    with pytest.raises(ValueError, match="Duplicate task"):
        stratified.load_eval(tmp_path, traits, Inputs())


def test_input_modification_detected(tmp_path):
    path = tmp_path / "input.json"
    write_json(path, {"a": 1})
    inputs = Inputs()
    inputs.read(path)
    write_json(path, {"a": 2})
    with pytest.raises(ValueError, match="changed"):
        inputs.verify()


def test_cli_rejects_protected_outputs_before_reading_runs(tmp_path):
    for output in ("reports/runs/stage4e-test", "data/verified/stage4e-test"):
        with pytest.raises(ValueError):
            stratified.main(["--row", "base=unused", "--out", output])
