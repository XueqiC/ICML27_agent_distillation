import pytest

from train.sft import exposure_plan, checkpoint_metadata


def test_complete_turn_budget_passes_determinism():
    costs = [200, 150, 90, 600, 40]
    plan = exposure_plan(costs, passes=2, seed=4)
    assert plan == exposure_plan(costs, passes=2, seed=4)
    flat = [i for b in plan for i in b["indices"]]
    assert sorted(flat[:5]) == sorted(flat[5:]) == list(range(5))
    assert plan[-1]["cumulative_supervised_tokens"] == 2 * sum(costs)
    assert plan[-1]["passes"] == 2 and plan[-1]["final"]
    for b in plan:
        assert b["supervised_tokens"] == sum(costs[i] for i in b["indices"])
        assert b["supervised_tokens"] <= 512 or len(b["indices"]) == 1


def test_endpoints_round_up_only_complete_turn_and_metadata():
    plan = exposure_plan([300], supervised_tokens=650, checkpoint_tokens=[250, 650])
    assert [b["cumulative_supervised_tokens"] for b in plan] == [300, 600, 900]
    assert plan[-1]["requested_endpoints"] == [650]
    state = checkpoint_metadata(plan[-1], updates=3, wall_seconds=6, update_seconds=3,
              peak_memory=123, peak_reserved=200, bank_tokens=300, identity_sha256="identity")
    assert state["endpoint_overshoot_tokens"] == {"650": 250}
    assert state["seconds_per_update"] == 1 and state["passes"] == 3
    assert state["peak_memory_bytes"] == 123
    assert exposure_plan([300], supervised_tokens=10000, passes=1)[-1]["supervised_tokens"] == 300


@pytest.mark.parametrize("kwargs", [{}, {"passes": 0}, {"supervised_tokens": -2},
    {"passes": 1, "checkpoint_tokens": [9999]}, {"passes": 1, "token_budget": 0}])
def test_invalid_exposure_settings(kwargs):
    with pytest.raises(ValueError):
        exposure_plan([3], **kwargs)
