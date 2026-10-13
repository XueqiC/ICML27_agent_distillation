from copy import deepcopy
from importlib.util import find_spec
import json
from types import SimpleNamespace

import pytest

has_model_stack = all(find_spec(name) is not None for name in ("torch", "transformers", "peft"))
model_test = pytest.mark.skipif(not has_model_stack, reason="requires torch, transformers and peft")
if has_model_stack:
    from test_adam_probe import checkpoint_run, load_checkpoint, single_cpu_thread
from test_crn_tail import assert_state_equal
from sed.storage import digest, file_hash, write_json
from train.continue_from_state import continuation_plan, continue_training, filter_extra_states
from train.optimizer_state import capture_state, load_checkpoint_state, load_update_state
from train.merge_lora import validate_adapter_artifacts
from train.sft import exposure_plan


@model_test
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


@pytest.fixture
def reference_rows():
    return [{"input_ids": [2, 0, 0], "labels": [-100, 0, 0], "supervised_tokens": 2},
            {"input_ids": [2, 1, 1, 1], "labels": [-100, 1, 1, 1], "supervised_tokens": 3}]


def test_deterministic_fraction_schedule_and_budget_accounting(reference_rows):
    rows = reference_rows
    original = exposure_plan([r["supervised_tokens"] for r in rows], passes=6, token_budget=1)
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


def test_explicit_schedule_preserves_reference_cursor_and_logs_extension(reference_rows):
    rows = reference_rows
    original = exposure_plan([r["supervised_tokens"] for r in rows], passes=1, token_budget=1)
    state = {"updates": 1, "data_cursor": 1,
             "supervised_tokens": original[0]["cumulative_supervised_tokens"]}
    _, plan, accounting = continuation_plan(rows, original, state, rows,
        target_tokens=12, schedule={"2": [1]}, seed=0)
    assert plan[1]["indices"] == [3] and plan[1]["pool"] == "extra"
    assert plan[2]["reference_plan_index"] == 0 and plan[2]["reference_cycle"] == 1
    assert accounting["reference_cycles"] >= 1


@pytest.mark.parametrize("options", [{"fraction": -.1}, {"fraction": 1.1}, {"schedule": [1]},
                                    {"schedule": [999]}, {"fraction": .25, "schedule": [2]}])
def test_bad_schedule_or_fraction_rejected(options, reference_rows):
    rows = reference_rows
    original = exposure_plan([r["supervised_tokens"] for r in rows], passes=3, token_budget=1)
    state = {"updates": 1, "data_cursor": 1,
             "supervised_tokens": original[0]["cumulative_supervised_tokens"]}
    with pytest.raises(ValueError):
        continuation_plan(rows, original, state, rows, target_tokens=10, **options)


@pytest.fixture
def extra_pool(tmp_path):
    from data.manifest.build_r03_sets import dataset_entry, publish_dataset
    from sed.manifest import freeze_splits
    from test_r03_sets import TinyTokenizer
    from tools.crn_tail import Experiment

    task_id = "train/group/trial"
    manifest = {"tasks": [{"task_id": task_id, "group_id": "train/group",
        "source_split": "train", "eligible": True, "task_type": "pick_and_place_simple", "files": {}}]}
    freeze = freeze_splits(manifest)
    root = tmp_path / "r2_union"
    hashes = [digest(name) for name in ("alpha", "bravo", "charlie")]
    rejected = digest("rejected")
    entries, audit = [], []
    visible = {"observation": "visible", "admissible_commands": ["look", "go north"], "facts": "SECRET"}
    for i, state_hash in enumerate(hashes):
        # Same task and step, distinct events. No filename or episode-ID hash convention.
        episode = {"schema_version": 2, "episode_id": f"opaque-{i}", "task_id": task_id,
            "split": "train", "sealed": False, "initial": {"visible": visible},
            "steps": [{"event_id": "prefix", "action_raw": "Action: look", "visible": visible},
                      {"event_id": f"label-{i}", "action_raw": "Action: " + ("look" if i == 0 else "go north"),
                       "visible": visible}]}
        entry = dataset_entry(root, episode, source="teacher_action_at_student_visited_state", supervised_turns=[1])
        entry["state_hash"] = state_hash
        entries.append(entry)
        audit.append({"state_hash": state_hash, "accepted": True})
    # A failed attempt followed by acceptance must not count as a rejected-only state.
    audit.extend([{"state_hash": hashes[0], "accepted": False},
                  {"state_hash": rejected, "accepted": False},
                  {"state_hash": rejected, "accepted": False}])
    (root / "collection.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audit))
    publish_dataset(root, entries, manifest, freeze, source="R2")
    context = Experiment.__new__(Experiment)
    context.datasets, context.dataset_purposes, context.inputs = {}, {}, []
    context.manifest, context.freeze, context.tokenizer = manifest, freeze, TinyTokenizer()
    context.recipe = {"max_length": 32768, "packing": "episode", "tokens_per_update": 3}
    examples = context.load_dataset(str(root))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    yield SimpleNamespace(root=root, hashes=hashes, rejected=rejected, examples=examples, context=context)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("form", ["object", "list"])
def test_extra_states_filter_formats_and_preserve_dataset(extra_pool, tmp_path, form):
    pool = extra_pool
    selected = [pool.hashes[2], pool.hashes[0]]  # Input order must not reorder the dataset.
    value = {"states": [{"episode_id": "ignored", "state_hash": h, "step": 1} for h in selected]}
    path = tmp_path / "states.json"
    write_json(path, value if form == "object" else selected)
    before = deepcopy(pool.examples)
    rows, metadata = filter_extra_states(pool.examples, pool.root, path)
    assert rows == [pool.examples[0], pool.examples[2]]
    assert rows[0] is pool.examples[0] and rows[1] is pool.examples[2]
    assert pool.examples == before
    assert metadata == {"path": str(path), "sha256": file_hash(path),
        "counts": {"requested": 2, "matched": 2, "rejected_labels": 0, "unknown": 0},
        "matched_hashes_sha256": digest(sorted(selected)), "examples": 2,
        "supervised_tokens": sum(row["supervised_tokens"] for row in rows)}
    assert all("SECRET" not in pool.context.tokenizer.decode(row["input_ids"], skip_special_tokens=False)
               for row in rows)


@pytest.mark.parametrize("form", ["object", "list"])
def test_extra_states_duplicate_error(extra_pool, tmp_path, form):
    hashes = [extra_pool.hashes[0]] * 2
    path = tmp_path / "states.json"
    write_json(path, {"states": [{"state_hash": h} for h in hashes]} if form == "object" else hashes)
    with pytest.raises(ValueError, match="Duplicate extra state hashes"):
        filter_extra_states(extra_pool.examples, extra_pool.root, path)


@pytest.mark.parametrize("value", [None, {}, {"states": "hash"}, {"states": ["hash"]},
                                    {"states": [{}]}, [123], [""], [{"state_hash": "hash"}]])
def test_extra_states_malformed_selection_error(extra_pool, tmp_path, value):
    path = tmp_path / "states.json"
    write_json(path, value)
    with pytest.raises(ValueError, match="[Ee]xtra state"):
        filter_extra_states(extra_pool.examples, extra_pool.root, path)


def test_extra_states_unknown_hash_reports_counts(extra_pool, tmp_path):
    pool = extra_pool
    path = tmp_path / "states.json"
    write_json(path, [pool.hashes[0], pool.rejected, digest("unknown")])
    with pytest.raises(ValueError, match="Unknown extra state hashes") as error:
        filter_extra_states(pool.examples, pool.root, path)
    assert json.loads(str(error.value).split(": ", 1)[1]) == {
        "requested": 3, "matched": 1, "rejected_labels": 1, "unknown": 1}


def test_extra_states_rejected_labels_counted_once(extra_pool, tmp_path):
    pool = extra_pool
    path = tmp_path / "states.json"
    write_json(path, [pool.hashes[0], pool.rejected])
    rows, metadata = filter_extra_states(pool.examples, pool.root, path)
    assert rows == pool.examples[:1]
    assert metadata["counts"] == {"requested": 2, "matched": 1, "rejected_labels": 1, "unknown": 0}
    assert metadata["matched_hashes_sha256"] == digest(pool.hashes[:1])
    assert metadata["supervised_tokens"] == rows[0]["supervised_tokens"]


@pytest.mark.parametrize("empty", [False, True])
def test_extra_states_zero_match_error(extra_pool, tmp_path, empty):
    path = tmp_path / "states.json"
    write_json(path, [] if empty else [extra_pool.rejected])
    with pytest.raises(ValueError, match="matched zero states") as error:
        filter_extra_states(extra_pool.examples, extra_pool.root, path)
    assert json.loads(str(error.value).split(": ", 1)[1]) == {
        "requested": int(not empty), "matched": 0, "rejected_labels": int(not empty), "unknown": 0}


def test_extra_states_requires_extra_before_loading_model(tmp_path, monkeypatch, capsys):
    import train.continue_from_state as continuation

    def forbidden(*_):
        pytest.fail("Invalid CLI arguments must be rejected before loading a model")
    monkeypatch.setattr(continuation, "Experiment", forbidden)
    with pytest.raises(SystemExit) as error:
        continuation.main(["--snapshot", "unused", "--checkpoint", "unused", "--verified", "unused",
                           "--output", str(tmp_path / "run"), "--extra-states", "states.json"])
    assert error.value.code == 2
    assert "--extra-states requires --extra" in capsys.readouterr().err


@pytest.mark.parametrize("filtered", [False, True])
def test_extra_states_cli_provenance_and_accounting_without_model(extra_pool, tmp_path, monkeypatch, capsys, filtered):
    import train.continue_from_state as continuation

    pool, output = extra_pool, tmp_path / "continued"
    context = pool.context
    context.reference = pool.examples[1:]
    context.plan = exposure_plan([e["supervised_tokens"] for e in context.reference], passes=3, token_budget=1)
    context.state = {"updates": 1, "data_cursor": 1,
                     "supervised_tokens": context.plan[0]["cumulative_supervised_tokens"]}
    context.run = tmp_path / "source_run"
    context.checkpoint = context.run / "state-update-1"
    context.provenance = {"config": {"sft": context.recipe}}
    context.model = context.optimizer = None
    monkeypatch.setattr(continuation, "Experiment", lambda _: context)
    trained = {}
    def fake_training(model, optimizer, state, examples, plan, output, **kwargs):
        assert model is None and optimizer is None
        trained.update(examples=examples, plan=plan)
        return {"checkpoints": [{"path": "fake-checkpoint"}]}
    monkeypatch.setattr(continuation, "continue_training", fake_training)
    path = tmp_path / "states.json"
    write_json(path, [pool.hashes[0], pool.rejected])
    args = ["--snapshot", "unused", "--checkpoint", str(context.checkpoint), "--verified", "unused",
            "--extra", str(pool.root), "--fraction", "0.25", "--target-tokens", "80", "--output", str(output)]
    if filtered:
        args.extend(["--extra-states", str(path)])
    continuation.main(args)
    provenance = json.loads((output / "provenance.json").read_text())
    accounting = json.loads((output / "continuation_plan.json").read_text())
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"checkpoints": ["fake-checkpoint"], **accounting}
    recipe = provenance["config"]["sft"]
    expected_extra = pool.examples[:1] if filtered else pool.examples
    examples, plan, expected_accounting = continuation_plan(context.reference, context.plan, context.state,
        expected_extra, target_tokens=80, fraction=.25, token_budget=3)
    assert trained == {"examples": examples, "plan": plan}
    assert recipe["examples_sha256"] == digest(examples) and recipe["plan_sha256"] == digest(plan)
    replacement_indices = [i for b in plan[1:] if b["pool"] == "extra" for i in b["indices"]]
    assert replacement_indices and all(len(context.reference) <= i < len(examples) for i in replacement_indices)
    if filtered:
        metadata = accounting["extra_states"]
        assert metadata == recipe["continuation"]["extra_states"]
        assert metadata["path"] == str(path) and metadata["sha256"] == file_hash(path)
        assert metadata["counts"] == {"requested": 2, "matched": 1, "rejected_labels": 1, "unknown": 0}
        assert metadata["matched_hashes_sha256"] == digest(pool.hashes[:1])
        assert metadata["supervised_tokens"] == pool.examples[0]["supervised_tokens"]
        assert provenance["inputs"][str(path)] == file_hash(path)
        ledger = pool.root / "collection.jsonl"
        assert provenance["inputs"][str(ledger)] == file_hash(ledger)
        assert recipe["continuation"]["extra_root"] == str(pool.root)
    else:
        assert accounting == expected_accounting
        assert "extra_states" not in provenance["config"]["continuation_tool"]
        assert str(path) not in provenance["inputs"]
        assert str(pool.root / "collection.jsonl") not in provenance["inputs"]
