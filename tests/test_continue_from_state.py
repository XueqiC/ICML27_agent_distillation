from copy import deepcopy
import json
import time

import pytest
torch = pytest.importorskip("torch")

from test_adam_probe import checkpoint_run, load_checkpoint, toy_rows, toy_plan, single_cpu_thread
from test_crn_tail import assert_state_equal
from sed.storage import digest, write_json
from train.continue_from_state import continuation_plan, continue_training
from train.optimizer_state import capture_state, load_checkpoint_state, load_update_state
from train.merge_lora import validate_adapter_artifacts


@pytest.mark.parametrize("checkpoint_run", [0, 3], indirect=True)
def test_f_zero_midrun_checkpoint_reproduces_tiny_training_bitwise(checkpoint_run, tmp_path, capsys):
    run = checkpoint_run
    expected = load_update_state(run.output / "state-update-4", plan=run.plan)
    state = load_checkpoint(run)
    examples, plan, accounting = continuation_plan(run.rows, run.plan, state,
        target_tokens=run.plan[-1]["cumulative_supervised_tokens"], fraction=0)
    assert [b["indices"] for b in plan] == [b["indices"] for b in run.plan]
    assert accounting["replacement_updates"] == 0
    provenance = deepcopy(run.provenance)
    provenance["config"]["sft"]["plan_sha256"] = digest(plan)
    output = tmp_path / "continued"
    output.mkdir()
    write_json(output / "exposure_plan.json", plan)
    report = continue_training(run.model, run.optimizer, state, examples, plan, output,
                               identity_sha256=digest(provenance), save_every_update=True)
    actual = capture_state(run.model, run.optimizer)
    assert_state_equal(actual, expected)
    saved = load_update_state(output / "state-update-4", plan=plan)
    assert_state_equal(saved, expected)
    adapter = output / report["checkpoints"][-1]["path"]
    assert validate_adapter_artifacts(adapter)["updates"] == 4
    reloaded = load_checkpoint_state(run.model, run.optimizer, adapter, plan=plan, provenance=provenance)
    assert_state_equal(reloaded, expected)
    ledger = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert [r["update"] for r in ledger] == [2, 3, 4]
    assert all(r["pool"] == "reference" for r in ledger)
    assert report["continued_updates"] == 3
    with capsys.disabled():
        print("\nR0R3_CONTINUE_TIMING=" + json.dumps({"moment_warmup": run.warmup,
            "three_updates_and_saves_seconds": report["wall_seconds"],
            "compute_seconds_per_update": report["seconds_per_update"]}))


def test_deterministic_fraction_schedule_and_budget_accounting():
    rows = toy_rows()
    original = toy_plan(rows, passes=6)
    state = {"updates": 1, "data_cursor": 1,
             "supervised_tokens": original[0]["cumulative_supervised_tokens"]}
    extra = [{**rows[0], "supervised_tokens": 2}]
    examples, plan, accounting = continuation_plan(rows, original, state, extra,
        target_tokens=30, fraction=.25, token_budget=3, seed=9)
    assert (examples, plan, accounting) == continuation_plan(rows, original, state, extra,
        target_tokens=30, fraction=.25, token_budget=3, seed=9)
    assert [b["relative_update"] for b in plan[1:] if b["pool"] == "extra"] == [4, 8]
    for b in plan:
        assert b["supervised_tokens"] == sum(examples[i]["supervised_tokens"] for i in b["indices"])
    assert accounting["realized_tokens"] >= 30
    assert accounting["overshoot_tokens"] < 3
    assert plan[-1]["final"]


def test_explicit_schedule_preserves_reference_cursor_and_logs_extension():
    rows = toy_rows()
    original = toy_plan(rows, passes=1)
    state = {"updates": 1, "data_cursor": 1,
             "supervised_tokens": original[0]["cumulative_supervised_tokens"]}
    _, plan, accounting = continuation_plan(rows, original, state, rows,
        target_tokens=12, schedule={"2": [1]}, seed=0)
    assert plan[1]["indices"] == [3] and plan[1]["pool"] == "extra"
    assert plan[2]["reference_plan_index"] == 0 and plan[2]["reference_cycle"] == 1
    assert accounting["reference_cycles"] >= 1


@pytest.mark.parametrize("options", [{"fraction": -.1}, {"fraction": 1.1}, {"schedule": [1]},
                                    {"schedule": [999]}, {"fraction": .25, "schedule": [2]}])
def test_bad_schedule_or_fraction_rejected(options):
    rows = toy_rows()
    original = toy_plan(rows)
    state = {"updates": 1, "data_cursor": 1,
             "supervised_tokens": original[0]["cumulative_supervised_tokens"]}
    with pytest.raises(ValueError):
        continuation_plan(rows, original, state, rows, target_tokens=10, **options)
