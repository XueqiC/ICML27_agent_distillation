"""Synthetic training ledgers: no models, data services, or evaluation reads."""
import json

import pytest

from sed.storage import file_hash, write_json
from tools.convergence_check import check, last_epoch, main
from train.convergence import compare_losses, doubling_passes, rung_status
from train.sft import exposure_plan


def ledger(root, rows, settings=None, *, example_count=1):
    root.mkdir(parents=True, exist_ok=True)
    (root / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(root / "example_sources.json", [{"source": "fake_train_only"}] * example_count)
    if settings:
        write_json(root / "provenance.json", {"config": {"baseline_training": settings}})


@pytest.mark.parametrize("objective", ["sft", "smartad", "sad"])
def test_baseline_token_weighting_final_epoch_and_input_hashes(tmp_path, objective):
    for name, epochs, offset in (("prev", 1, 0), ("curr", 2, .005)):
        rows, seen = [], 0
        for epoch in range(epochs):
            for indices, tokens, loss in (([0, 1], 1, 1.), ([2, 3], 3, 3.)):
                seen += tokens
                # Huge earlier-epoch losses must not affect the last epoch.
                value = loss - offset if epoch == epochs - 1 else 100
                rows.append({"update": len(rows) + 1, "epoch": epoch, "loss": 999,
                    "indices": indices, "examples_seen": 2 * (len(rows) + 1),
                    "global_batch": 2, "supervised_tokens_seen": seen,
                    "supervised_loss_sum": value * tokens, "objective": objective})
        settings = {"objective": objective, "epochs": epochs, "total_steps": len(rows),
                    "processed_token_budget": epochs * 100, "N_examples": 4}
        ledger(tmp_path / name, rows, settings, example_count=4)
    result = check(tmp_path / "prev", tmp_path / "curr")
    assert result["L_last_prev"] == 2.5
    assert result["L_last"] == pytest.approx(2.495)
    assert result["delta"] == pytest.approx(.005) and result["converged"]
    assert result["current"]["supervised_tokens"] == 4
    assert result["current"]["updates"] == 2
    assert result["current"]["final_epoch"] == result["current"]["selected_epoch"] == 1
    assert result["current"]["final_epoch_complete"]
    assert result["current"]["training_examples"] == 4
    assert result["previous"]["L_min"] == 2.5 and result["previous"]["valid"]
    assert result["current"]["L_min"] == pytest.approx(2.495) and result["current"]["valid"]
    assert [row["loss"] for row in result["current"]["epoch_losses"]] == pytest.approx([100, 2.495])
    for name in ("updates.jsonl", "provenance.json", "example_sources.json"):
        path = tmp_path / "curr" / name
        assert result["input_hashes"]["current"][str(path)] == file_hash(path)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("final_updates", [1, 2, 3])
def test_sad_uses_last_complete_epoch_with_ascending_complexity_losses(tmp_path, legacy, final_updates):
    rows, seen, examples_seen = [], 0, 0
    for epoch, losses in enumerate(((100, 200, 300), (1, 4, 9), (.1, .4, .9)[:final_updates])):
        for (indices, tokens), loss in zip((([0, 1], 3), ([2, 3], 7), ([4], 5)), losses):
            seen += tokens
            examples_seen += len(indices)
            row = {"update": len(rows) + 1, "epoch": epoch, "indices": indices,
                   "global_batch": len(indices), "examples_seen": examples_seen,
                   "supervised_tokens_seen": seen, "loss": loss * tokens / len(indices)}
            if not legacy:
                row["supervised_loss_sum"] = loss * tokens
            rows.append(row)
    # The larger budget reaches an easy prefix or all of the next epoch.
    for name, budget in (("prev", 100), ("curr", 200)):
        rung_rows = rows[:6] if name == "prev" else rows
        ledger(tmp_path / name, rung_rows, {"objective": "sad", "processed_token_budget": budget,
                                           "total_steps": len(rung_rows)}, example_count=5)
    main([str(tmp_path / "prev"), str(tmp_path / "curr")])
    result = json.loads((tmp_path / "curr/convergence.json").read_text())["current"]
    assert result["L_last"] == pytest.approx(76 / 15 * (.1 if final_updates == 3 else 1))
    assert result["final_epoch"] == 2
    assert result["selected_epoch"] == (2 if final_updates == 3 else 1)
    assert result["final_epoch_complete"] is (final_updates == 3)
    assert result["updates"] == 3 and result["supervised_tokens"] == 15
    assert result["training_examples"] == 5 and result["amount"] == 200
    assert len(result["epoch_losses"]) == (3 if final_updates == 3 else 2)
    assert result["L_min"] == result["L_last"] and result["valid"]


def test_sad_refuses_without_a_complete_epoch(tmp_path):
    rows = [{"update": 1, "epoch": 0, "indices": [0], "global_batch": 1,
             "supervised_tokens_seen": 2, "loss": .1}]
    ledger(tmp_path, rows, {"objective": "sad", "processed_token_budget": 2}, example_count=2)
    with pytest.raises(ValueError, match="No complete epoch"):
        last_epoch(tmp_path)


@pytest.mark.parametrize("objective", ["sft", "smartad"])
def test_epoch_based_objectives_refuse_partial_final_epoch(tmp_path, objective):
    rows = [{"update": 1, "epoch": 0, "indices": [1, 0], "global_batch": 2,
             "supervised_tokens_seen": 3, "supervised_loss_sum": 3},
            {"update": 2, "epoch": 1, "indices": [0], "global_batch": 1,
             "supervised_tokens_seen": 4, "supervised_loss_sum": .1}]
    ledger(tmp_path, rows, {"objective": objective, "epochs": 2, "total_steps": 2}, example_count=2)
    with pytest.raises(ValueError, match="final epoch is incomplete"):
        last_epoch(tmp_path)


def test_trajectory_refuses_incomplete_epoch_despite_integral_pass_count(tmp_path):
    rows = [{"update": 1, "passes": 1, "indices": [0, 1], "supervised_tokens": 2, "loss": 1.},
            {"update": 2, "passes": 2, "indices": [0], "supervised_tokens": 1, "loss": .1}]
    ledger(tmp_path, rows, example_count=2)
    with pytest.raises(ValueError, match="final epoch is incomplete"):
        last_epoch(tmp_path)


@pytest.mark.parametrize("change,match", [
    ({"indices": [0]}, "Duplicate"),
    ({"indices": [1, 1]}, "Duplicate"),
    ({"indices": [2]}, "within the training example count"),
    ({"indices": [-1]}, "within the training example count"),
    ({"indices": [True]}, "within the training example count"),
    ({"indices": None}, "Per-update indices"),
    ({"indices": []}, "Per-update indices"),
    ({"global_batch": 2}, "global_batch disagrees"),
    ({"examples_seen": 3}, "examples_seen disagrees"),
])
def test_checker_refuses_invalid_example_coverage(tmp_path, change, match):
    rows = [{"update": 1, "epoch": 0, "indices": [0], "supervised_tokens_seen": 1,
             "supervised_loss_sum": 1},
            {"update": 2, "epoch": 0, "indices": [1], "supervised_tokens_seen": 2,
             "supervised_loss_sum": 1, **change}]
    ledger(tmp_path, rows, {"objective": "sad", "processed_token_budget": 2}, example_count=2)
    with pytest.raises(ValueError, match=match):
        last_epoch(tmp_path)


@pytest.mark.parametrize("count", [None, 0, True, 1.5, 2, 3])
def test_example_count_requires_independent_consistent_metadata(tmp_path, count):
    rows = [{"update": 1, "epoch": 0, "indices": [0, 1], "supervised_tokens_seen": 2,
             "supervised_loss_sum": 2}]
    settings = {"objective": "sft", **({"N_examples": count} if count is not None else {})}
    ledger(tmp_path, rows, settings, example_count=2)
    if count == 3:
        with pytest.raises(ValueError, match="counts .* disagree"):
            last_epoch(tmp_path)
    else:
        (tmp_path / "example_sources.json").unlink()
        if count == 2:
            assert last_epoch(tmp_path)["training_examples"] == 2
        else:
            with pytest.raises(ValueError, match="Positive training example count required"):
                last_epoch(tmp_path)


def test_trajectory_legacy_format_and_cli(tmp_path, capsys):
    for name, passes in (("prev", 1), ("curr", 2)):
        plan = exposure_plan([1, 3], token_budget=1, passes=passes, seed=0, packing="episode")
        rows = [{"update": i, **batch, "loss": (1 if batch["indices"] == [0] else 3) - .001 * passes}
                for i, batch in enumerate(plan, 1)]
        ledger(tmp_path / name, rows, example_count=2)
    main([str(tmp_path / "prev"), str(tmp_path / "curr")])
    result = json.loads(capsys.readouterr().out)
    assert json.loads((tmp_path / "curr/convergence.json").read_text()) == result
    assert result["L_last_prev"] == pytest.approx(2.499)
    assert result["L_last"] == pytest.approx(2.498)
    assert result["converged"]
    assert result["current"]["selected_epoch"] == result["current"]["final_epoch"] == 1
    assert result["current"]["updates"] == 2 and result["current"]["supervised_tokens"] == 4


@pytest.mark.parametrize("previous,current,expected", [(.01, 0, False), (.009, 0, True), (0, 0, True),
                                                     (0, .01, True), (0, 1, False)])
def test_strict_signed_rule(previous, current, expected):
    # Small regressions still pass the signed delta; larger ones are invalid.
    assert compare_losses(previous, current, previous_min=previous, current_min=current)["converged"] is expected


def test_k500_first_epoch_divergence_cannot_count_as_convergence():
    # Supplied complete-epoch means from the 8-epoch run; the 4-epoch run ended at .13.
    means = [1.84, 4.47, 4.35, 4.34, 3.97, 1.51, .99]
    result = compare_losses(.13, means[-1], previous_min=.13, current_min=min(means))
    assert result["previous"]["valid"] and result["current"]["valid"]
    assert result["delta"] == pytest.approx(-.86)
    assert result["cross_rung_limit"] == pytest.approx(.15)
    assert not result["cross_rung_valid"] and not result["converged"]
    assert result["failed_checks"] == ["cross_rung"]
    assert "cross_rung" in result["reason"] and "limit=0.15" in result["reason"]
    assert "doubling the training ended higher than the half-length rung" in result["reason"]


@pytest.mark.parametrize("previous,limit", [(0., .02), (.1, .12), (.25, .275), (1., 1.1)])
def test_cross_rung_limit_is_inclusive(previous, limit):
    result = compare_losses(previous, limit, previous_min=previous, current_min=limit)
    assert result["cross_rung_valid"] and result["converged"] and result["failed_checks"] == []
    result = compare_losses(previous, limit + 1e-8, previous_min=previous, current_min=limit + 1e-8)
    assert not result["cross_rung_valid"] and not result["converged"]
    assert result["failed_checks"] == ["cross_rung"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1])
def test_invalid_delta(bad):
    with pytest.raises(ValueError):
        compare_losses(1, 1, bad, previous_min=1, current_min=1)


@pytest.mark.parametrize("minimum,limit", [(0., .02), (.1, .12), (.25, .275), (1., 1.1)])
def test_rung_validity_inclusive_relative_or_absolute_limit(minimum, limit):
    assert rung_status(limit, minimum)["valid"]
    assert not rung_status(limit + 1e-8, minimum)["valid"]


@pytest.mark.parametrize("objective", ["trajectory", "sft", "smartad", "sad"])
@pytest.mark.parametrize("previous,current,validity", [
    ([.25, 1.], [1.2, 1.1, 1.02, 1.001], (False, True)),
    ([1., 1.], [.25, .4, .9, 1.001], (True, False)),
    ([.3, .25], [.3, .25, 2., .249], (True, True)),  # Recovered spike.
    ([.25, 1.145], [.3, .25, 1.146, 1.143], (False, False)),
    ([.2, .13], [4., 3., 2., .99], (True, True)),  # Only the cross-rung guard fails.
])
def test_divergence_guard_uses_each_rungs_complete_epoch_token_means(tmp_path, capsys, objective, previous, current, validity):
    for name, means in (("prev", previous), ("curr", current)):
        rows, seen = [], 0
        for epoch, mean in enumerate(means):
            for index, tokens, loss in ((0, 1, mean / 2), (1, 3, mean * 7 / 6)):
                seen += tokens
                row = {"update": len(rows) + 1, "epoch": epoch, "indices": [index], "global_batch": 1}
                if objective == "trajectory":
                    row.update(supervised_tokens=tokens, loss=loss, passes=epoch + (index + 1) / 2)
                else:
                    row.update(supervised_tokens_seen=seen, supervised_loss_sum=loss * tokens,
                               objective=objective, loss=999)
                rows.append(row)
        settings = None if objective == "trajectory" else {"objective": objective, "epochs": len(means),
                       "total_steps": len(rows), "processed_token_budget": 100 * len(means)}
        ledger(tmp_path / name, rows, settings, example_count=2)
    main([str(tmp_path / "prev"), str(tmp_path / "curr")])
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((tmp_path / "curr/convergence.json").read_text())
    assert result["delta"] == pytest.approx(previous[-1] - current[-1])
    assert result["delta"] < .01  # Every case would pass the old rule.
    limit = max(1.1 * previous[-1], previous[-1] + .02)
    cross_valid = current[-1] <= limit
    failed = [name + "_within_rung" for name, valid in zip(("previous", "current"), validity) if not valid]
    failed += [] if cross_valid else ["cross_rung"]
    assert result["cross_rung_limit"] == pytest.approx(limit)
    assert result["cross_rung_valid"] is cross_valid
    assert result["failed_checks"] == failed
    assert all(name in result["reason"] for name in failed)
    assert result["converged"] is (not failed)
    assert ("Unrecovered divergence" in result["reason"]) is bool(failed)
    for name, means, valid in zip(("previous", "current"), (previous, current), validity):
        rung = result[name]
        assert rung["L_min"] == pytest.approx(min(means))
        assert rung["L_last"] == pytest.approx(means[-1])
        assert rung["valid"] is valid
        assert ("Unrecovered divergence" in rung["reason"]) is (not valid)
        assert [row["loss"] for row in rung["epoch_losses"]] == pytest.approx(means)


def test_checker_rejects_different_warmup_ratios(tmp_path):
    for name, epochs, ratio in (("prev", 1, 0.), ("curr", 2, .1)):
        rows = [{"update": epoch + 1, "epoch": epoch, "indices": [0], "supervised_loss_sum": 1.,
                 "supervised_tokens_seen": epoch + 1} for epoch in range(epochs)]
        ledger(tmp_path / name, rows, {"objective": "sft", "warmup_ratio": ratio})
    with pytest.raises(ValueError, match="settings differ: warmup_ratio"):
        check(tmp_path / "prev", tmp_path / "curr")


def test_checker_rejects_non_doubling_incomplete_and_ambiguous_logs(tmp_path):
    def rows(passes):
        return [{"update": i, "passes": i, "indices": [0], "supervised_tokens": 2, "loss": 1.}
                for i in range(1, passes + 1)]
    ledger(tmp_path / "a", rows(1))
    ledger(tmp_path / "b", rows(3))
    with pytest.raises(ValueError, match="double"):
        check(tmp_path / "a", tmp_path / "b")
    ledger(tmp_path / "b", [{"update": 1, "passes": 1.5, "supervised_tokens": 3, "loss": 1.}])
    with pytest.raises(ValueError, match="crosses"):
        last_epoch(tmp_path / "b")
    ledger(tmp_path / "b", [{"update": 1, "epoch": 0, "supervised_tokens_seen": 3,
                             "global_batch": 2, "loss": 1, "objective": "sft"}])
    with pytest.raises(ValueError, match="lacks exact"):
        last_epoch(tmp_path / "b")
    write_json(tmp_path / "b/result.json", {"total_steps": 2})
    ledger(tmp_path / "b", [{"update": 1, "epoch": 0, "supervised_tokens_seen": 3,
                             "supervised_loss_sum": 3, "objective": "sft"}])
    with pytest.raises(ValueError, match="planned endpoint"):
        last_epoch(tmp_path / "b")


def test_ladder_plan_is_cap_independent_and_flushes_passes():
    def plan(cap):
        return exposure_plan([1, 2, 4], token_budget=5, passes=cap, packing="episode", seed=7,
                             checkpoint_tokens=[7 * p for p in doubling_passes(cap)], flush_passes=True)
    short, long = plan(4), plan(8)
    assert short[:-1] == long[:len(short) - 1]
    assert {**short[-1], "final": False} == long[len(short) - 1]
    assert [b["passes"] for b in long if b["requested_endpoints"]] == [1, 2, 4, 8]
    for epoch in range(8):
        batches = [b for b in long if b["epoch"] == epoch]
        assert sorted(i for b in batches for i in b["indices"]) == [0, 1, 2]
        assert batches[-1]["end_of_pass"]
        assert sum(b["supervised_tokens"] for b in batches) == 7


@pytest.mark.parametrize("args", [["--passes-ladder"], ["--passes-ladder", "--max-passes", "3"],
    ["--passes-ladder", "--max-passes", "2", "--lr-warmup-steps", "1"],
    ["--passes-ladder", "--max-passes", "2", "--passes", "1"],
    ["--passes-ladder", "--max-passes", "2", "--packing", "turn"], ["--max-passes", "2"]])
def test_ladder_cli_rejects_invalid_options_before_loading(tmp_path, args):
    from train.sft import main as train
    with pytest.raises(SystemExit):
        train(["--snapshot", "unused", "--verified", "unused", "--output", str(tmp_path / "run"), *args])
    assert not (tmp_path / "run").exists()
