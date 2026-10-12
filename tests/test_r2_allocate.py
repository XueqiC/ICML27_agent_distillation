from collections import Counter
import json
import math

import pytest

from sed.manifest import TASK_TYPES
from sed.storage import digest
from tools.collect_corrections import select_r1_states
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


def test_crn_maximal_overlap_union_and_determinism():
    records, demos, sigma, r1 = synthetic()
    result = allocate(records, demos, sigma, r1_states=r1["states"])
    assert result == allocate(records, demos, sigma, r1_states=r1["states"])
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
        allocate(records[:1], demos, sigma, budgets=(400,))
