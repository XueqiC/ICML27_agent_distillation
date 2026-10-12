from collections import Counter
import json
import math

import pytest

from sed.manifest import TASK_TYPES
from sed.storage import digest
from tools.collect_corrections import select_r1_states
from tools import r2_allocate
from tools.r2_allocate import (allocate, cluster_id, cluster_statistics, largest_remainder,
                               predicted_error, read_sigma, water_fill)


def synthetic():
    kinds = sorted(TASK_TYPES)
    records, demos = [], []
    # 120 games with varying length and 24 occupied clusters, no model calls.
    for i in range(120):
        kind = kinds[i % 6]
        states = [{"episode_id": f"e{i}", "step": j, "state_hash": digest([i, j])}
                  for j in range(12 + (i % 5) * 6)]
        records.append({"episode_id": f"e{i}", "task_id": f"train/game-{i}",
                        "task_type": kind, "states": states})
    for i in range(20):
        demos.append((kinds[i % 6], {"steps": [{}] * (8 + i % 4 * 5)}))
    stats, _ = cluster_statistics(records, demos)
    sigma = {"schema_version": 1, "checkpoint": "p10", "clusters": {
        cid: {"sigma2": 0.25 + (i % 7) ** 2, "sample_count": 10} for i, cid in enumerate(stats)}}
    r1 = select_r1_states(records, [r["task_id"] for r in records[:40]])
    return records, demos, sigma, r1


@pytest.mark.parametrize("budget", [1, 100, 400])
def test_water_filling_kkt_and_integer_budget(budget):
    weights = {"a": 1.0, "b": 4.0, "c": 9.0, "d": 0.0}
    coverage = {"a": 40, "b": 2, "c": 0, "d": 0}
    result = water_fill(weights, coverage, budget)
    x, lam = result["continuous"], result["lambda"]
    assert sum(x.values()) == pytest.approx(budget)
    assert sum(result["integer"].values()) == budget
    for s, n in x.items():
        if n > 1e-7:
            assert weights[s] / (coverage[s] + n) ** 2 == pytest.approx(lam)
        elif coverage[s]:
            assert weights[s] / coverage[s] ** 2 <= lam + 1e-8
        assert abs(result["integer"][s] - n) <= 1 + 1e-8


def test_finite_capacity_kkt_zero_weights_and_infeasible():
    w, k, caps = {"a": 100.0, "b": 1.0}, {"a": 0, "b": 0}, {"a": 1, "b": 100}
    out = water_fill(w, k, 10, caps)
    assert out["integer"] == {"a": 1, "b": 9}
    assert out["saturated_clusters"] == ["a"]
    assert w["a"] / 1**2 >= out["lambda"]
    assert w["b"] / 9**2 == pytest.approx(out["lambda"])
    assert sum(water_fill({"a": 0, "b": 0}, k, 10, caps)["integer"].values()) == 10
    with pytest.raises(ValueError, match="exceeds"):
        water_fill(w, k, 102, caps)
    assert predicted_error(w, k, {"a": 0, "b": 1}) is None
    assert largest_remainder({"a": 1.5, "b": 1.5}, 3) == {"a": 2, "b": 1}
    single = water_fill({"a": 9.0}, {"a": 0}, 1)
    assert single["lambda"] == pytest.approx(9.0)


def test_sigma_accepts_grad_stats_checkpoint_path_only_at_p10(tmp_path):
    path = tmp_path / "training_state.json"
    path.write_text(json.dumps({"updates": 115, "supervised_tokens": 104070}))
    value = {"schema_version": 1, "checkpoint": str(tmp_path),
             "clusters": {"c": {"sigma2": 1.0, "sample_count": 10}}}
    assert read_sigma(value, ["c"]) == {"c": 1.0}
    path.write_text(json.dumps({"updates": 116, "supervised_tokens": 104070}))
    with pytest.raises(ValueError, match="p10"):
        read_sigma(value, ["c"])


def test_clusters_boundaries_coverage_and_sigma_validation():
    kind = sorted(TASK_TYPES)[0]
    assert [cluster_id(kind, i).split("::")[1] for i in [0, 4, 5, 9, 10, 19, 20, 39]] == [
        "0-4", "0-4", "5-9", "5-9", "10-19", "10-19", "20+", "20+"]
    records, demos, sigma, _ = synthetic()
    stats, _ = cluster_statistics(records, demos)
    assert len(stats) == 24 and sum(v["d"] for v in stats.values()) == pytest.approx(1)
    assert sum(v["k"] for v in stats.values()) == sum(len(d["steps"]) for _, d in demos)
    cid = next(iter(stats))
    sigma["clusters"][cid]["sigma2"] = math.nan
    with pytest.raises(ValueError, match="sigma"):
        read_sigma(sigma, stats)


@pytest.mark.parametrize("adam_weight", ["raw_sigma", "scale_free"])
def test_crn_maximal_overlap_union_and_determinism(adam_weight):
    records, demos, sigma, r1 = synthetic()
    result = allocate(records, demos, sigma, r1_states=r1["states"], adam_weight=adam_weight)
    assert result == allocate(records, demos, sigma, r1_states=r1["states"], adam_weight=adam_weight)
    if adam_weight == "raw_sigma":
        assert result == allocate(records, demos, sigma, r1_states=r1["states"])
        assert result["union_query_count"] == 593
        assert result["r1_r2_overlap"] == 132
        assert result["r1_r2_union_query_count"] == 1061
        assert result["r2_additional_after_r1"] == 461
    assert r1["omitted_by_cap"] > 0 and result["r1_query_count"] == 600
    refs = {s["state_hash"]: cluster_id(r["task_type"], s["step"]) for r in records for s in r["states"]}
    arms = list(result["arms"].values())
    for arm in arms:
        assert len(arm["states"]) == sum(arm["allocation"].values()) == arm["budget"]
        assert len({s["state_hash"] for s in arm["states"]}) == arm["budget"]
        for other in arms:
            overlap = {s["state_hash"] for s in arm["states"]} & {s["state_hash"] for s in other["states"]}
            counts = Counter(refs[s] for s in overlap)
            for cid in result["clusters"]:
                assert counts[cid] == min(arm["allocation"][cid], other["allocation"][cid])
    assert result["union_query_count"] <= 1500
    union = {s["state_hash"] for a in arms for s in a["states"]}
    r1_ids = {s["state_hash"] for s in r1["states"]}
    assert result["r1_r2_union_query_count"] == len(union | r1_ids)
    assert result["r2_additional_after_r1"] == len(union - r1_ids)
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="exceeds"):
        allocate(records[:1], demos, sigma, budgets=(400,), adam_weight=adam_weight)


def test_adam_weight_switch_raw_sigma_dominates_and_kkt():
    kinds = sorted(TASK_TYPES)[:2]
    records = [{"episode_id": f"e{i}", "task_id": f"train/game-{i}",
                "task_type": kinds[0 if i < 40 else 1],
                "states": [{"episode_id": f"e{i}", "step": 0, "state_hash": digest([i])}]}
               for i in range(200)]
    demos = [(kind, {"steps": [{}]}) for kind in kinds]
    high, low = [cluster_id(kind, 0) for kind in kinds]
    sigma = {"schema_version": 1, "checkpoint": "p10", "clusters": {
        high: {"sigma2": 1e6, "sample_count": 10}, low: {"sigma2": 1, "sample_count": 10}}}
    raw = allocate(records, demos, sigma, budgets=(20, 100))
    scale_free = allocate(records, demos, sigma, budgets=(20, 100), adam_weight="scale_free")
    assert raw["arms"]["A3_n20"]["allocation"] == {high: 20, low: 0}
    assert scale_free["arms"]["A3_n20"]["allocation"] == {high: 6, low: 14}
    assert raw["arms"]["A3_n100"]["water_filling"]["saturated_clusters"] == [high]
    for name, arm in raw["arms"].items():
        if not name.startswith("A3"):
            assert arm == scale_free["arms"][name]
    for mode, result in (("raw_sigma", raw), ("scale_free", scale_free)):
        assert result["adam_weight"] == mode
        for cid, row in result["clusters"].items():
            assert row["W_Adam"] == row["d"] * (row["sigma2"] if mode == "raw_sigma" else 1)
            assert row["W_SGD"] == row["d"] ** 2 * row["sigma2"]
        for name, arm in result["arms"].items():
            for definition in ("raw_sigma", "scale_free"):
                expected = sum(row["d"] * (row["sigma2"] if definition == "raw_sigma" else 1)
                               / (row["k"] + arm["allocation"][cid])
                               for cid, row in result["clusters"].items())
                assert arm["predicted_errors"][definition] == {
                    "predicted_error": pytest.approx(expected), "predicted_error_infinite": False}
            assert arm["predicted_error_weight"] == "raw_sigma"
            assert arm["predicted_error"] == arm["predicted_errors"]["raw_sigma"]["predicted_error"]
            if "water_filling" not in arm:
                continue
            solution = arm["water_filling"]
            lam = solution["lambda"]
            assert sum(solution["continuous"].values()) == pytest.approx(arm["budget"])
            assert sum(arm["allocation"].values()) == arm["budget"]
            for cid, n in solution["continuous"].items():
                row = result["clusters"][cid]
                weight = row["W_SGD"] if name.startswith("A2") else row["W_Adam"]
                marginal = weight / (row["k"] + n) ** 2
                assert 0 <= n <= row["unique_states"]
                assert abs(arm["allocation"][cid] - n) <= 1 + 1e-8
                if n < 1e-7:
                    assert marginal <= lam + 1e-8
                elif abs(n - row["unique_states"]) < 1e-7:
                    assert marginal >= lam - 1e-8
                else:
                    assert marginal == pytest.approx(lam)


@pytest.mark.parametrize("adam_weight", ["raw_sigma", "scale_free"])
def test_dual_predictions_preserve_zero_weight_and_infinite_errors(adam_weight):
    records, _, sigma, _ = synthetic()
    for row in sigma["clusters"].values():
        row["sigma2"] = 0
    result = allocate(records, [], sigma, budgets=(0,), adam_weight=adam_weight)
    for arm in result["arms"].values():
        assert arm["predicted_errors"] == {
            "raw_sigma": {"predicted_error": 0, "predicted_error_infinite": False},
            "scale_free": {"predicted_error": None, "predicted_error_infinite": True}}
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="adam_weight"):
        allocate(records, [], sigma, adam_weight="invalid")


@pytest.mark.parametrize("adam_weight,statistics_only", [
    (None, False), ("raw_sigma", False), ("scale_free", False), ("scale_free", True)])
def test_cli_records_adam_weight_in_outputs_and_provenance(tmp_path, monkeypatch,
                                                         adam_weight, statistics_only):
    records, demos, sigma, r1 = synthetic()
    for name in ("config.json", "manifest.json", "freeze.json", "rollouts.jsonl", "k20/manifest.json"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")
    pool = {"subsets": {"R1_P2": [r["task_id"] for r in records[:40]],
                        "R2": [r["task_id"] for r in records]}}
    (tmp_path / "pool.json").write_text(json.dumps(pool))
    (tmp_path / "sigma.json").write_text(json.dumps(sigma))
    config = {"paths": {"manifest": str(tmp_path / "manifest.json"),
                        "split_freeze": str(tmp_path / "freeze.json")}}
    manifest = {r["task_id"]: r for r in records}
    evidence = {r["task_id"]: demo[1] for r, demo in zip(records, demos)}
    monkeypatch.setattr(r2_allocate, "load_config", lambda path: config)
    monkeypatch.setattr(r2_allocate, "load_frozen", lambda *args: (manifest, {}))
    monkeypatch.setattr(r2_allocate, "read_rollouts", lambda *args, **kwargs: [(r, {}) for r in records])
    monkeypatch.setattr(r2_allocate, "load_verified_collection", lambda *args: (
        {"K": 20, "verified": [{"task_id": tid} for tid in evidence]}, {"seed": 0}, evidence))
    monkeypatch.setattr(r2_allocate, "guarded_task", lambda m, f, tid: m[tid])
    monkeypatch.setattr(r2_allocate, "checked_episode", lambda e, r: e[r["task_id"]])
    output = tmp_path / "output"
    args = ["--config", str(tmp_path / "config.json"), "--pool", str(tmp_path / "pool.json"),
            "--rollouts", str(tmp_path / "rollouts.jsonl"), "--k20", str(tmp_path / "k20"),
            "--output", str(output)]
    if adam_weight is not None:
        args += ["--adam-weight", adam_weight]
    args += ["--statistics-only"] if statistics_only else ["--sigma", str(tmp_path / "sigma.json")]
    r2_allocate.main(args)
    expected_mode = adam_weight or "raw_sigma"
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["config"]["r2_allocation"]["adam_weight"] == expected_mode
    assert json.loads((output / "clusters.json").read_text())["adam_weight"] == expected_mode
    assert json.loads((output / "r1_selection.json").read_text()) == r1
    if statistics_only:
        assert not (output / "allocation.json").exists()
    else:
        result = json.loads((output / "allocation.json").read_text())
        expected = allocate(records, demos, sigma, r1_states=r1["states"], adam_weight=expected_mode)
        assert result == {**expected, "rollout_game_count": len(records), "missing_r2_game_ids": []}
        for name, arm in result["arms"].items():
            assert json.loads((output / (name + ".json")).read_text()) == arm
