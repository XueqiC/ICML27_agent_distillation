"""Amendment 3: real runner/replay with fake policies, no HTTP or GPU."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.mechanism.cache import copy_cache_seed, prepare_cache_seed
from analysis.mechanism.common import read_json
from analysis.mechanism.fce import IncompleteProbe, dry_plan, locate_closest_approach, run_fce
from analysis.mechanism.interventions import run_interventions
from analysis.mechanism.policies import BranchTeacher
from analysis.mechanism.readout import build_readout, distance_diagnostics, markdown, validate_fce_artifacts
from sed.storage import digest, file_hash, write_json
from test_mechanism import FakeResponses, FakeStudent, TASK_ID
from test_mechanism_pipeline import pipeline


def test_dense_minimum_and_coarse_ties_do_not_resample():
    calls = []
    def probe(t):
        calls.append(t)
        return {0: [9, 7], 4: [6, 5], 8: [3, 2], 9: [3, None],
                10: [2, 1], 11: [1, 1], 12: [1, 1], 16: [1, 1]}[t]
    result = locate_closest_approach(16, probe)
    assert calls == [0, 4, 8, 12, 16, 9, 10, 11]
    assert result["coarse_minimum_prefix"] == 12
    assert result["dense_probe_points"] == [9, 10, 11, 12]
    assert result["D_0"] == 7 and result["D_t_star"] == 1 and result["t_star"] == 10
    assert result["turns_remaining_at_t_star"] == 30
    diagnostic = distance_diagnostics(dict(result, task_id=TASK_ID, episode_length=16))
    assert diagnostic["turns_spent_after_t_star_without_progress"] == 6
    assert result["budget_fce"] is None and result["budget_fce_status"] == "no_drop_detected"


def test_budget_refinement_retained_and_its_minimum_can_win():
    calls = []
    def probe(t):
        calls.append(t)
        return {0: [7, 6], 4: [5, 4], 8: [None, None], 12: [3, 2],
                5: [1, 2], 6: [None, None], 9: [1, 1], 10: [2, 3], 11: [2, 2]}[t]
    result = locate_closest_approach(12, probe)
    assert calls == [0, 4, 8, 12, 5, 6, 9, 10, 11]
    assert result["budget_fce"] == result["fce"] == 5
    assert result["coarse_interval"] == [4, 8]
    assert result["coarse_minimum_prefix"] == 12 and result["t_star"] == 5
    assert result["budget_fce_status"] == "fce_found"


@pytest.mark.parametrize("length,distances,expected,tc,dense", [
    (0, {0: [1, 2]}, 0, 0, [0]),
    (3, {0: [3, 4], 3: [1, 2], 1: [1, 3], 2: [1, 1]}, 1, 3, [0, 1, 2, 3]),
    (2, {0: [None, None], 2: [1, None], 1: [2, 2]}, 2, 2, [0, 1, 2]),
    (4, {0: [None, None], 4: [None, None]}, None, None, []),
])
def test_clipping_endpoints_undefined_and_later_solvable(length, distances, expected, tc, dense):
    result = locate_closest_approach(length, distances.__getitem__)
    assert result["t_star"] == expected and result["coarse_minimum_prefix"] == tc
    assert result["dense_probe_points"] == dense
    assert result["D_0"] == (min(d for d in distances[0] if d is not None) if any(
        d is not None for d in distances[0]) else None)
    if expected is None:
        assert result["status"] == "teacher_unsolvable_from_student_states"
        assert result["D_t_star"] is None and result["turns_remaining_at_t_star"] is None
    else:
        assert result["status"] == "closest_approach_found"


@pytest.mark.parametrize("incomplete_at", [4, 37, 9])
def test_unknowns_at_any_phase_do_not_freeze_a_t_star(incomplete_at):
    def probe(t):
        if t == incomplete_at:
            raise IncompleteProbe("fake cap")
        return [None, None] if t == 40 else [min(40-t, 5 if t < 12 else 1)]*2
    result = locate_closest_approach(40, probe)
    assert result["status"] == "incomplete" and result["t_star"] is None
    assert result["D_0"] == 5


def test_dry_estimate_includes_new_prefixes_and_two_samples():
    result = dry_plan(40, max_attempts=3)
    assert result["scan_policy_call_upper_bound"] == 440
    assert result["refinement_policy_call_upper_bound"] == 228
    assert result["densification_policy_call_upper_bound"] == 228
    assert result["http_attempt_upper_bound_before_caps"] == 3*(440+228+228)
    assert dry_plan(0)["densification_policy_call_upper_bound"] == 0


def distance_teacher(calls):
    def factory(config, workspace, budget, parent, *, namespace, sample_index, advice=False):
        prefixes = {digest([parent, t, "takeover"]): t for t in range(41)}
        prefix = prefixes.get(namespace)
        class Responses(FakeResponses):
            def create(self, **kwargs):
                turn = (len(kwargs["input"])-2)//2
                if prefix is None:
                    self.raw = "Take the apple." if advice else "Action: win"
                else:
                    distance = ({0: (9, 7), 4: (6, 5), 8: (3, 2), 9: (3, 3), 10: (2, 1),
                                 11: (1, 1)}.get(prefix, (1, 1)))[sample_index]
                    self.raw = "Action: win" if turn-prefix+1 >= distance else "Action: inventory"
                calls.append(kwargs)
                return super().create(**kwargs)
        return BranchTeacher(namespace=namespace, sample_index=sample_index, budget=budget,
            episode_key=parent, advice=advice, cache_dir=Path(workspace)/"teacher_cache",
            usage_ledger=Path(workspace)/"teacher_usage.jsonl", run_id="fake",
            client=SimpleNamespace(responses=Responses()), sampling={"max_output_tokens": 8},
            max_attempts=1, sleep=lambda _: None, requests_per_minute=100000, tokens_per_minute=1000000000)
    return factory


def test_t_star_interventions_readout_and_seeded_cache_zero_calls(pipeline, tmp_path):
    config, gap, env, _, _, seeds = pipeline
    calls = []
    teacher = distance_teacher(calls)
    old = tmp_path/"old workspace"
    args = dict(gap_path=gap, config=config, max_episode=1000, max_run=1000,
                env_factory=env, teacher_factory=teacher)
    first = run_fce(**args, output=old/"fce")
    row = first["episodes"][0]
    assert first["complete"] and row["budget_fce"] == 39 and row["t_star"] == 10
    assert row["D_0"] == 7 and row["D_t_star"] == 1 and row["turns_remaining_at_t_star"] == 30
    assert row["probe_samples"]["10"] == [
        {"sample_index": 0, "success": True, "teacher_steps_to_go": 2},
        {"sample_index": 1, "success": True, "teacher_steps_to_go": 1}]
    assert row["D_curve"]["40"] is None
    assert len(calls) == first["budget"]["total"] > 0
    original = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    calls.clear()
    new = tmp_path/"new"
    second = run_fce(**args, output=new, seed_teacher_cache_from=old)
    assert calls == [] and second["budget"]["total"] == 0
    assert second["episodes"] == first["episodes"]
    first_episodes = {str(p.relative_to(old/"fce")): p.read_bytes() for p in (old/"fce").rglob("probe_episode.json")}
    assert first_episodes and first_episodes == {
        str(p.relative_to(new)): p.read_bytes() for p in new.rglob("probe_episode.json")}
    assert original == {str(p.relative_to(old)): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    manifest = read_json(new/"teacher_cache_seed.json")
    provenance = read_json(new/"provenance.json")["details"]["teacher_cache_seed"]
    assert provenance["source_path"] == str(old.resolve())
    assert provenance["entry_count"] == len(list((old/"fce/teacher_cache").glob("*.json")))
    assert provenance["manifest_sha256"] == digest(manifest["entries"])
    assert all(file_hash(new/p) == sha for p, sha in manifest["entries"].items())
    usage = [json.loads(line) for line in (new/"teacher_usage.jsonl").read_text().splitlines()]
    assert len(usage) == provenance["entry_count"] and all(r["status"] == "cache_hit" for r in usage)
    assert all(r["usage"]["total_tokens"] == 0 for r in usage)
    # Resume keeps the frozen seed, even if a live source later gains entries.
    assert run_fce(**args, output=new, seed_teacher_cache_from=old) == second
    validate_fce_artifacts(new/"fce.json", second)
    for key in ("D_0", "t_star", "D_t_star", "budget_fce", "turns_remaining_at_t_star"):
        broken = deepcopy(second)
        broken["episodes"][0][key] = -1
        with pytest.raises(ValueError):
            validate_fce_artifacts(new/"fce.json", broken)
    result = run_interventions(gap_path=gap, fce_path=new/"fce.json", verified=None,
        output=tmp_path/"arms", config=config, spec={}, max_episode=1000, max_run=1000,
        env_factory=env, teacher_factory=teacher, student_factory=FakeStudent, seeds=seeds)
    entry = result["episodes"][0]
    assert entry["t"] == entry["t_star"] == 10 and entry["budget_fce"] == 39
    assert len(calls) == 2  # I3 and I4 still share one teacher output per arm.
    assert all(sample["branch"]["t"] == 10 for arm in entry["arms"].values() for sample in arm["samples"])
    report = build_readout(read_json(gap), second, result)
    diagnostic = report["distance_diagnostics"][0]
    assert diagnostic["D_0"] == 7 and diagnostic["D_t_star"] == 1
    assert diagnostic["turns_remaining_at_t_star"] == diagnostic["turns_spent_after_t_star_without_progress"] == 30
    assert diagnostic["budget_fce"] == 39 and report["fce_positions"]["histogram"] == {39: 1}
    assert report["paired_arm_rescue"]["primary"]["I4"]["paired_excess"]["count"] == 1
    text = markdown(report)
    assert all(s in text for s in ("D(0)", "D(t*)", "Turns remaining at t*", "descriptive", "Budget FCE", "D(t) curves"))
    entry["t"] = 39
    with pytest.raises(ValueError, match="closest-approach"):
        build_readout(read_json(gap), second, result)


def test_cache_manifest_frozen_source_is_read_only_and_collision_refused(tmp_path):
    source, output = tmp_path/"source", tmp_path/"output"
    (source/"teacher_blobs").mkdir(parents=True)
    cached = {"cache_key": "key", "result": {"raw_text": "Action: look"}}
    cached["result_sha256"] = digest(cached["result"])
    write_json(source/"teacher_cache/key.json", cached)
    (source/"teacher_cache/key.json.lock").write_text("ignore")
    seed = prepare_cache_seed(source, output)
    copy_cache_seed(seed, output)
    assert seed["entry_count"] == 1 and not (output/"teacher_cache/key.json.lock").exists()
    write_json(source/"teacher_cache/another.json", dict(cached, cache_key="another"))
    assert prepare_cache_seed(source, output) == seed
    with pytest.raises(ValueError, match="separate"):
        prepare_cache_seed(source, source/"child")
    with pytest.raises(ValueError, match="manifest changed"):
        prepare_cache_seed(tmp_path/"another", output)
    (output/"teacher_cache/key.json").write_text("conflict")
    with pytest.raises(ValueError, match="conflicts"):
        copy_cache_seed(seed, output)
    (source/"teacher_cache/key.json").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        copy_cache_seed(seed, tmp_path/"changed")


def test_unsolvable_readout_keeps_distance_diagnostics(pipeline):
    _, gap, *_ = pipeline
    located = locate_closest_approach(40, lambda _: [None, None])
    located.update(task_id=TASK_ID, episode_length=40)
    report = build_readout(read_json(gap), {"episodes": [located]}, {"episodes": []})
    assert report["exclusions"] == {"teacher_unsolvable_from_student_states": 1}
    assert report["distance_diagnostics"][0]["D_0"] is None
    assert report["distance_diagnostics"][0]["turns_spent_after_t_star_without_progress"] is None
    assert report["paired_arm_rescue"]["primary"]["I4"]["paired_excess"]["n"] == 0
