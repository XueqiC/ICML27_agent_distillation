"""Offline CPU T1 pilot checks using the existing tiny Gemma/LoRA fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from test_adam_probe import checkpoint_run, load_checkpoint, single_cpu_thread, ToyModel
from test_crn_tail import assert_state_equal
from test_continue_from_state import extra_pool
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, write_json
from tools.crn_tail import Experiment, restore, surrogate_score
from train.continue_from_state import continuation_plan, continue_training
from train.optimizer_state import bitwise_equal, load_checkpoint_state, load_update_state
import train.transplant as transplant


def read(path):
    return json.loads(Path(path).read_text())


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert bitwise_equal(a, b)
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.fixture
def cli_run(checkpoint_run, tmp_path, monkeypatch):
    """Only replace dataset/base initialization; updates, save/load and CLI are real."""
    run = checkpoint_run
    snapshot, verified, extra = (tmp_path / name for name in ("snapshot", "verified", "extra"))
    for directory in (snapshot, verified, extra):
        directory.mkdir()
        write_json(directory / "manifest.json", {"fixture": directory.name})
    (snapshot / "weights.bin").write_bytes(b"tiny fixture base identity")
    extra_rows = [{"input_ids": [2, 7], "labels": [-100, 7], "supervised_tokens": 1}]
    recipe = {**run.provenance["config"]["sft"], "tokens_per_update": 1}
    contexts = []

    def experiment(args):
        checkpoint = Path(args.checkpoint).resolve()
        provenance = read(checkpoint.parent / "provenance.json")
        plan = read(checkpoint.parent / "exposure_plan.json")
        state = load_checkpoint_state(run.model, run.optimizer, checkpoint, plan=plan, provenance=provenance)
        context = SimpleNamespace(args=args, checkpoint=checkpoint, run=checkpoint.parent,
            reference=run.rows, plan=plan, state=state, model=run.model, optimizer=run.optimizer,
            recipe={**recipe, **provenance["config"]["sft"]}, provenance=provenance,
            tokenizer=None, datasets={str(verified): run.rows},
            inputs=[checkpoint.parent / "provenance.json", checkpoint.parent / "exposure_plan.json"])
        if getattr(args, "surrogate", None):
            context.surrogate = run.rows
        def dataset(path):
            assert path == str(extra)
            context.datasets[path] = extra_rows
            return extra_rows
        context.load_dataset = dataset
        contexts.append(context)
        return context

    monkeypatch.setattr(transplant, "Experiment", experiment)
    common = ["--snapshot", str(snapshot), "--verified", str(verified), "--device", "cpu"]
    def block(name="B0", *, alternative=False, updates=1, target=10):
        output = tmp_path / name
        args = ["block", *common, "--checkpoint", str(run.path), "--updates", str(updates),
                "--target-tokens", str(target), "--output", str(output)]
        if alternative:
            args += ["--extra", str(extra), "--fraction", "1", "--seed", "1202"]
        transplant.main(args)
        return output / f"state-update-{1 + updates}"

    def post(state, name):
        output = tmp_path / name
        transplant.main(["post", *common, "--checkpoint", str(state), "--plan-from", str(run.path),
                         "--target-tokens", "10", "--output", str(output)])
        return output

    return SimpleNamespace(run=run, block=block, post=post, common=common, contexts=contexts,
                           snapshot=snapshot, extra=extra, extra_rows=extra_rows)


def test_block_matches_full_plan_rng_draws_and_saved_state(cli_run, tmp_path):
    case = cli_run
    # A long enough plan to exercise multiple independently sampled extra batches.
    state = load_checkpoint(case.run)
    extra = case.run.rows
    expected_examples, expected, _ = continuation_plan(case.run.rows, case.run.plan, state, extra,
        target_tokens=60, fraction=.25, seed=1202, token_budget=5)
    examples, full, block, accounting = transplant.block_plan(case.run.rows, case.run.plan, state, extra,
        updates=8, target_tokens=60, fraction=.25, seed=1202, token_budget=5)
    assert examples == expected_examples and full == expected
    assert block[:-1] == expected[:8]
    assert block[-1] == {**expected[8], "final": True}
    assert accounting["block_replacement_updates"] == 2
    for b, f in zip(block[1:], full[1:]):
        assert (b["indices"], b["pool"]) == (f["indices"], f["pool"])
    output = tmp_path / "draws"
    continue_training(case.run.model, case.run.optimizer, state, examples, block, output, identity_sha256="unit")
    saved = load_update_state(output / "state-update-9", plan=block)
    restore(case.run.model, case.run.optimizer, saved)
    assert saved["updates"] == saved["data_cursor"] == 9
    assert accounting["full_plan_sha256"] == digest(expected)
    checkpoint = case.block()
    audit = read(checkpoint.parent / "block.json")
    assert audit["full_plan_file_sha256"] == file_hash(checkpoint.parent / "full_exposure_plan.json")
    provenance = read(checkpoint.parent / "provenance.json")
    assert provenance["inputs"][str(case.snapshot / "weights.bin")] == file_hash(case.snapshot / "weights.bin")
    assert provenance["details"]["directory_sha256"][str(case.snapshot)]


@pytest.mark.parametrize("updates", [0, -1, 100, 1.5])
def test_block_invalid_length(checkpoint_run, updates):
    run = checkpoint_run
    with pytest.raises(ValueError, match="Block updates"):
        transplant.block_plan(run.rows, run.plan, load_checkpoint(run), updates=updates, target_tokens=10)


@pytest.mark.parametrize("channel", ["m", "v", "both", "identity"])
def test_compose_only_requested_fields_change_and_reload(cli_run, tmp_path, channel):
    a, b = cli_run.block(), cli_run.block("B1", alternative=True)
    A, B = load_update_state(a), load_update_state(b)
    m = b if channel in {"m", "both"} else a
    v = b if channel in {"v", "both"} else a
    output = tmp_path / "composed"
    record = transplant.compose(a, m, v, output)
    actual, expected = load_update_state(output), deepcopy(A)
    for field, donor in (("exp_avg", B if m == b else A), ("exp_avg_sq", B if v == b else A)):
        for index, entry in expected["optimizer"]["state"].items():
            entry[field] = donor["optimizer"]["state"][index][field]
    assert_nested_equal(actual, expected)  # Includes RNG and every non-moment field.
    restore(cli_run.run.model, cli_run.run.optimizer, actual)
    assert (output / "adapter_model.safetensors").read_bytes() == (a / "adapter_model.safetensors").read_bytes()
    assert record["sources"]["theta"]["sha256"] == digest(artifact_hashes(a))
    for field, key, changed in (("exp_avg", "m_difference_norm", m == b),
                                ("exp_avg_sq", "v_difference_norm", v == b)):
        norm = sum(float((B["optimizer"]["state"][i][field].double() - e[field].double()).square().sum())
                   for i, e in A["optimizer"]["state"].items()) ** .5
        assert record[key] == (norm if changed else 0)
        if changed:
            assert norm > 0
    # Theta metadata is preserved except the necessarily updated artifact hashes.
    left, right = read(a / "training_state.json"), read(output / "training_state.json")
    left.pop("artifact_sha256")
    right.pop("artifact_sha256")
    assert left == right


@pytest.mark.parametrize("fault", ["step", "shape", "dtype", "order", "learning_rate", "updates"])
def test_compose_rejects_incompatible_state_before_writing(cli_run, tmp_path, fault):
    a, b = cli_run.block(), cli_run.block("B1", alternative=True)
    payload = load_update_state(b)
    entry = next(iter(payload["optimizer"]["state"].values()))
    if fault == "step":
        entry["step"] += 1
    elif fault == "shape":
        entry["exp_avg"] = entry["exp_avg"].flatten()[:1]
    elif fault == "dtype":
        entry["exp_avg_sq"] = entry["exp_avg_sq"].double()
    elif fault == "order":
        payload["optimizer"]["param_groups"][0]["params"].reverse()
    elif fault == "learning_rate":
        payload["optimizer"]["param_groups"][0]["lr"] *= 2
    else:
        payload["updates"] += 1
    torch.save(payload, b / "optimizer.pt")
    metadata = read(b / "training_state.json")
    metadata["artifact_sha256"] = {k: h for k, h in artifact_hashes(b).items() if k != "training_state.json"}
    write_json(b / "training_state.json", metadata)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        transplant.compose(a, b, a, output)
    assert not output.exists()


@pytest.mark.parametrize("checkpoint_run", [0, 3], indirect=True)
def test_block_then_post_bitwise_uninterrupted_duplicate_and_absolute_lr(cli_run):
    case = cli_run
    # The original fixture is a complete uninterrupted run with LR warmup.
    expected = load_update_state(case.run.output / "state-update-4")
    b0 = case.block()
    first, second = case.post(b0, "T00"), case.post(b0, "T00p")
    actual = load_update_state(first / "state-update-4", plan=read(first / "exposure_plan.json"))
    assert_state_equal(actual, expected)
    assert_state_equal(actual, load_update_state(second / "state-update-4"))
    assert (first / "post_plan.json").read_bytes() == (second / "post_plan.json").read_bytes()
    ledger = [json.loads(line) for line in (first / "updates.jsonl").read_text().splitlines()]
    assert [row["update"] for row in ledger] == [3, 4]
    assert [row["learning_rate"] for row in ledger] == [1e-3, 1e-3]
    report = read(first / "profiling_report.json")
    token_checkpoint = first / report["checkpoints"][-1]["path"]
    # Also reload the exported adapter through the existing token checkpoint loader.
    reloaded = load_checkpoint_state(case.run.model, case.run.optimizer, token_checkpoint,
        plan=read(first / "exposure_plan.json"), provenance=read(first / "provenance.json"))
    assert_state_equal(reloaded, expected)


def test_post_plan_identical_for_composed_and_natural_unequal_token_counts(cli_run, tmp_path):
    a, b = cli_run.block(), cli_run.block("B1", alternative=True)
    assert load_update_state(a)["supervised_tokens"] != load_update_state(b)["supervised_tokens"]
    composed = tmp_path / "S_T11"
    transplant.compose(a, b, b, composed)
    t11, n1 = cli_run.post(composed, "T11"), cli_run.post(b, "N1")
    assert (t11 / "post_plan.json").read_bytes() == (n1 / "post_plan.json").read_bytes()
    ta, na = read(t11 / "post.json"), read(n1 / "post.json")
    assert ta["post_plan_file_sha256"] == na["post_plan_file_sha256"]
    assert na["state_supervised_tokens"] != na["reference_supervised_tokens"]
    assert na["actual_final_supervised_tokens"] != na["reference_final_supervised_tokens"]
    reference = read(t11 / "exposure_plan.json")
    for output in (t11, n1):
        ledger = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
        assert [{key: row[key] for key in batch} for row, batch in zip(ledger, reference[2:])] == reference[2:]


@pytest.mark.parametrize("fault", ["cursor", "end", "learning_rate"])
def test_post_refuses_invalid_prefix_or_learning_recipe(checkpoint_run, fault):
    run = checkpoint_run
    state = load_checkpoint(run)
    changed = deepcopy(state)
    if fault == "cursor":
        changed["data_cursor"] += 1
    elif fault == "end":
        changed["updates"] = changed["data_cursor"] = len(run.plan)
    else:
        changed["learning_rate"] *= 2
    with pytest.raises(ValueError):
        transplant.post_plan(run.rows, run.plan, state, changed, target_tokens=10)


def test_score_cli_uses_surrogate_listing_loader_and_same_numbers(cli_run, tmp_path, monkeypatch):
    from data.manifest.build_r03_sets import dataset_entry, publish_dataset
    from sed.manifest import freeze_splits
    from test_r03_sets import TinyTokenizer
    case = cli_run
    checkpoint = case.block()
    manifest = {"tasks": [{"task_id": "train/surrogate/trial", "group_id": "train/surrogate",
        "source_split": "train", "eligible": True, "task_type": "pick_and_place_simple", "files": {}}]}
    freeze = freeze_splits(manifest)
    root = tmp_path / "surrogate"
    visible = {"observation": "visible", "admissible_commands": ["look"]}
    episode = {"schema_version": 2, "episode_id": "unit", "task_id": "train/surrogate/trial", "split": "train",
        "sealed": False, "initial": {"visible": visible},
        "steps": [{"event_id": "e0", "action_raw": "Action: look", "visible": visible}]}
    entry = dataset_entry(root, episode, source="unit")
    publish_dataset(root, [entry], manifest, freeze, source="unit", purpose="surrogate_only")
    listing = tmp_path / "surrogate.json"
    write_json(listing, {"dataset_root": "surrogate", "dataset_manifest_sha256": file_hash(root / "manifest.json")})
    context = Experiment.__new__(Experiment)
    context.inputs, context.datasets, context.dataset_purposes = [], {}, {}
    context.manifest, context.freeze, context.tokenizer = manifest, freeze, TinyTokenizer()
    context.recipe = {"max_length": 32768, "packing": "episode"}
    model = ToyModel()
    model.theta = torch.nn.Parameter(torch.zeros(2048, dtype=torch.float64))
    # ToyModel's forward fixes vocab=4; use the same logits fixture with a larger vocabulary.
    model.forward = lambda *, logits_to_keep, **_: SimpleNamespace(
        logits=model.theta[None, None, :].expand(1, len(logits_to_keep), len(model.theta)))
    context.model, context.optimizer = model, None
    context.checkpoint, context.run = case.run.path, case.run.output
    context.provenance = case.run.provenance
    calls = []
    def experiment(args):
        calls.append(args.surrogate)
        context.surrogate = context.load_dataset(args.surrogate, purpose="surrogate_only")
        return context
    monkeypatch.setattr(transplant, "Experiment", experiment)
    # The checkpoint-load behavior is covered above; score on the fixture model.
    monkeypatch.setattr(transplant, "restore", lambda *_: None)
    output = tmp_path / "score"
    transplant.main(["score", *case.common, "--checkpoint", str(checkpoint), "--surrogate", str(listing),
                     "--output", str(output)])
    actual = read(output / "score.json")
    expected = surrogate_score(model, context.load_dataset(str(listing), purpose="surrogate_only"))
    assert {k: actual[k] for k in expected} == expected
    assert calls == [str(listing)]
    assert actual["checkpoint_sha256"] == digest(artifact_hashes(checkpoint))
    provenance = read(output / "provenance.json")
    assert provenance["inputs"][str(listing)] == file_hash(listing)
    assert provenance["details"]["directory_sha256"][str(root)]


def test_score_restores_real_tiny_adapter_and_composed_state(cli_run, tmp_path):
    case = cli_run
    b0 = case.block()
    composed = tmp_path / "S_T00"
    transplant.compose(b0, b0, b0, composed)
    post = case.post(composed, "T00")
    final = post / read(post / "profiling_report.json")["checkpoints"][-1]["path"]
    surrogate = tmp_path / "surrogate-fixture"
    surrogate.mkdir()
    write_json(surrogate / "manifest.json", {"tiny_rows_sha256": digest(case.run.rows)})
    for name, checkpoint, state in (("composed", composed, load_update_state(composed)),
            ("final", final, load_update_state(post / "state-update-4"))):
        restore(case.run.model, case.run.optimizer, state)
        expected = surrogate_score(case.run.model, case.run.rows)
        with torch.no_grad():
            next(p for p in case.run.model.parameters() if p.requires_grad).add_(1)
        output = tmp_path / ("score-" + name)
        transplant.main(["score", *case.common, "--checkpoint", str(checkpoint),
            "--surrogate", str(surrogate), "--output", str(output)])
        actual = read(output / "score.json")
        assert {key: actual[key] for key in expected} == expected


def test_filtered_block_cli_reuses_r03_state_selection(extra_pool, tmp_path, monkeypatch):
    from train.sft import exposure_plan
    pool = extra_pool
    context = pool.context
    context.reference = pool.examples[1:]
    context.plan = exposure_plan([e["supervised_tokens"] for e in context.reference], passes=6, token_budget=1)
    context.state = {"updates": 1, "data_cursor": 1,
                     "supervised_tokens": context.plan[0]["cumulative_supervised_tokens"]}
    context.run = tmp_path / "original"
    context.checkpoint = context.run / "state-update-1"
    context.checkpoint.mkdir(parents=True)
    write_json(context.checkpoint / "training_state.json", context.state)
    context.provenance = {"config": {"sft": context.recipe}}
    context.model = context.optimizer = None
    context.tokenizer = None
    monkeypatch.setattr(transplant, "Experiment", lambda _: context)
    executed = {}
    def training(*args, **kwargs):
        executed.update(examples=args[3], plan=args[4])
        return {"checkpoints": []}
    monkeypatch.setattr(transplant, "continue_training", training)
    selected = tmp_path / "A3_n100.json"
    write_json(selected, [pool.hashes[0], pool.rejected])
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    output = tmp_path / "block"
    transplant.main(["block", "--snapshot", str(snapshot), "--verified", str(pool.root),
        "--checkpoint", str(context.checkpoint), "--extra", str(pool.root), "--extra-states", str(selected),
        "--fraction", ".25", "--seed", "1202", "--updates", "4", "--target-tokens", "1000",
        "--device", "cpu", "--output", str(output)])
    assert executed["examples"] == context.reference + pool.examples[:1]
    assert executed["plan"][-1]["pool"] == "extra"
    record = read(output / "block.json")
    assert record["extra_states"]["counts"] == {"requested": 2, "matched": 1, "rejected_labels": 1, "unknown": 0}
    provenance = read(output / "provenance.json")
    for path in (selected, pool.root / "collection.jsonl"):
        assert provenance["inputs"][str(path)] == file_hash(path)


def test_real_experiment_loads_post_checkpoints_and_scores(tmp_path, monkeypatch):
    """Use the real Experiment constructor with local fixture data/tokenizer IO."""
    from test_adam_probe import gemma_rows
    from test_stage2b_torch import tiny_model
    from train.modeling import attach_lora, load_student, snapshot_metadata
    from train.sft import deterministic_seed, exposure_plan, train_loop
    import train.dataset
    import sed.manifest

    deterministic_seed(17)
    snapshot = tmp_path / "base"
    tiny_model().save_pretrained(snapshot)
    tokenizer = SimpleNamespace(chat_template="unit", save_pretrained=lambda output:
                                write_json(Path(output) / "tokenizer.json", {"fixture": True}))
    monkeypatch.setattr(train.dataset, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(train.dataset, "mask_report", lambda *_: {})
    manifest, freeze, verified, surrogate = (tmp_path / name for name in ("manifest.json", "freeze.json", "verified", "surr"))
    write_json(manifest, {})
    write_json(freeze, {})
    for path in (verified, surrogate):
        path.mkdir()
        write_json(path / "manifest.json", {"fixture": True})
    monkeypatch.setattr(sed.manifest, "load_frozen", lambda *_: ({}, {}))
    rows = gemma_rows()
    def dataset(self, root, *, purpose="training"):
        assert root in (str(verified), str(surrogate))
        self.datasets[root] = rows
        return rows
    monkeypatch.setattr(Experiment, "load_dataset", dataset)
    monkeypatch.setattr(Experiment, "check_surrogate_disjoint", lambda *_: None)
    plan = exposure_plan([2, 3], passes=2, token_budget=1, checkpoint_tokens=[2])
    recipe = {"dropout": 0, "optimizer": "AdamW", "betas": [0.9, .999], "eps": 1e-8,
        "weight_decay": 0, "learning_rate": 1e-5, "gradient_clip": 1.0, "seed": 17,
        "base_snapshot": snapshot_metadata(snapshot), "dtype": "float32", "rank": 2, "alpha": 4,
        "max_length": 128, "packing": "episode", "tokens_per_update": 1,
        "plan_sha256": digest(plan), "examples_sha256": digest(rows), "template_sha256": digest("unit")}
    provenance = {"config": {"sft": recipe, "paths": {"manifest": str(manifest), "split_freeze": str(freeze)}},
                  "inputs": {str(path): file_hash(path) for path in (manifest, freeze)}}
    source = tmp_path / "source"
    write_json(source / "provenance.json", provenance)
    write_json(source / "exposure_plan.json", plan)
    model = attach_lora(load_student(snapshot, device="cpu", dtype="float32"), rank=2, alpha=4)
    train_loop(model, rows, plan, source, identity_sha256=digest(provenance), save_every_update_through=4)
    z10 = source / "checkpoint-tokens-2"
    common = ["--snapshot", str(snapshot), "--verified", str(verified), "--device", "cpu"]
    block, composed, post = (tmp_path / name for name in ("block", "S_T00", "post"))
    transplant.main(["block", *common, "--checkpoint", str(z10), "--updates", "1",
                     "--target-tokens", "10", "--output", str(block)])
    state = block / "state-update-2"
    transplant.main(["compose", "--theta", str(state), "--m", str(state), "--v", str(state), "--output", str(composed)])
    transplant.main(["post", *common, "--checkpoint", str(composed), "--plan-from", str(z10),
                     "--target-tokens", "10", "--output", str(post)])
    assert_state_equal(load_update_state(post / "state-update-4"), load_update_state(source / "state-update-4"))
    # Both standard checkpoint conventions must work without adapting Experiment.
    for name in ("state-update-4", "checkpoint-tokens-10"):
        experiment = Experiment(SimpleNamespace(snapshot=str(snapshot), verified=str(verified),
                                checkpoint=str(post / name), device="cpu"))
        assert_state_equal(experiment.state, load_update_state(source / "state-update-4"))
    scores = tmp_path / "scores"
    transplant.main(["score", *common, "--checkpoint", str(post / "checkpoint-tokens-10"),
                     "--surrogate", str(surrogate), "--output", str(scores)])
    assert read(scores / "score.json")["F_tilde_geo_prob"] == surrogate_score(experiment.model, rows)["F_tilde_geo_prob"]


@pytest.mark.parametrize("delta,M,verdict", [(0, .002, "positive"), (0, -.002, "positive"),
    (0, .0001, "negative"), (.0013, .02, "uninterpretable"), (.0005, .0014, "negative")])
def test_preregistered_summary(delta, M, verdict):
    values = {"T00": .4, "T00p": .4 + delta, "T11": .4 + M, "T10": .401,
              "T01": .399, "N1": .403, "A0": .402}
    result = transplant.summarize_scores({a: {"F_tilde_geo_prob": f} for a, f in values.items()})
    assert result["verdict"] == verdict
    assert result["tau"] == max(3 * abs(values["T00"] - values["T00p"]), .0012)
    assert result["I"] == result["M"] - result["M_m"] - result["M_v"]
    assert result["P"] + result["M"] == pytest.approx(result["total_feedback_effect"])
    assert result["A0_minus_T00"] == values["A0"] - values["T00"]


@pytest.mark.parametrize("command", ["block", "post", "score", "compose"])
def test_existing_outputs_refused_before_loading_model(tmp_path, monkeypatch, command):
    output = tmp_path / "exists"
    output.mkdir()
    monkeypatch.setattr(transplant, "Experiment", lambda *_: pytest.fail("Should fail before loading model"))
    args = [command, "--output", str(output)]
    if command == "compose":
        args += ["--theta", "unused", "--m", "unused", "--v", "unused"]
    else:
        args += ["--snapshot", "unused", "--verified", "unused", "--checkpoint", "unused"]
        args += {"block": ["--updates", "1"], "post": ["--plan-from", "unused"],
                 "score": ["--surrogate", "unused"]}[command]
    with pytest.raises(ValueError, match="must not pre-exist"):
        transplant.main(args)


def test_shell_syntax():
    for script in ("scripts/hpg_t1_transplant.slurm", "scripts/hpg_t1_eval.slurm"):
        subprocess.run(["bash", "-n", script], check=True)
