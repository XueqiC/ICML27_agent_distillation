"""Offline deterministic paired replacements on toy and actual tiny Gemma LoRA."""
from copy import deepcopy
import json
import math
import random
import struct

import pytest
torch = pytest.importorskip("torch")

from test_adam_probe import (ToyModel, checkpoint_run, load_checkpoint, toy_rows, toy_plan, single_cpu_thread)
from tools.crn_tail import (batch_for, bernstein_radius, build_items, merge_results, message_backward,
                            require_no_dropout, restore, run_item, select_examples, step_message,
                            surrogate_score, tail_plan, validate_examples)
from train.optimizer_state import adamw, capture_state, bitwise_equal
from train.sft import token_mean_backward


@pytest.fixture
def warm_toy():
    torch.set_num_threads(1)
    model, examples = ToyModel(), toy_rows()
    optimizer = adamw(model.parameters(), 1e-5)
    for index in (0, 1, 0):
        step_message(model, optimizer, {"episodes": [index]}, examples)
    state = capture_state(model, optimizer, data_cursor=3, updates=3, supervised_tokens=7,
                          learning_rate=1e-5, gradient_clip=1.0, lr_warmup_steps=0, moment_warmup_steps=0)
    return model, optimizer, state, examples


def item(name="A", replica=0, message=None):
    return {"item_id": name, "replica": replica, "message": message or {"episodes": [0]},
            "tail_updates": 2, "tail_seed": 100 + replica}


def assert_state_equal(a, b):
    for key in ("trainable",):
        assert a[key].keys() == b[key].keys()
        assert all(bitwise_equal(a[key][n], b[key][n]) for n in a[key])
    for index, values in a["optimizer"]["state"].items():
        assert all(bitwise_equal(v, b["optimizer"]["state"][index][k]) for k, v in values.items())
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    assert bitwise_equal(a["torch_rng"], b["torch_rng"])
    assert a["python_rng"] == b["python_rng"]


def test_tiny_gemma_item_twice_is_bitwise_and_restores_all_state(checkpoint_run):
    run = checkpoint_run
    state = load_checkpoint(run)
    before = deepcopy(state)
    first = run_item(run.model, run.optimizer, state, run.rows, run.rows, item(), token_budget=2,
                     learning_rate=1e-3)
    second = run_item(run.model, run.optimizer, state, run.rows, run.rows, item(), token_budget=2,
                      learning_rate=1e-3)
    for key in ("F_tilde_geo_prob", "F_tilde_per_token", "F_tilde_per_traj"):
        assert struct.pack("d", first[key]) == struct.pack("d", second[key])
    assert first["message"] == second["message"]
    assert_state_equal(capture_state(run.model, run.optimizer), before)
    assert_state_equal(state, before)  # Snapshot never aliased by a branch.


def test_item_timing_breakdown_includes_both_restores(warm_toy, monkeypatch):
    import tools.crn_tail as crn
    model, optimizer, state, rows = warm_toy
    baseline = run_item(model, optimizer, state, rows, rows, item(), token_budget=3)
    clock = iter([0., 2., 5., 10., 17., 18., 21.])
    monkeypatch.setattr(crn.time, "monotonic", lambda: next(clock))
    def forbidden(*_):
        raise AssertionError("CPU timing must not synchronize CUDA")
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    result = run_item(model, optimizer, state, rows, rows, item(), token_budget=3)
    assert result["timing_seconds"] == {"restore": 5., "message_update": 3.,
                                        "tail_updates": 5., "surrogate_scoring": 7.}
    assert result["wall_seconds"] == 21.
    for key in baseline.keys() - {"timing_seconds", "wall_seconds"}:
        assert result[key] == baseline[key]
    assert_state_equal(capture_state(model, optimizer), state)


def test_cuda_timing_synchronizes_model_devices_before_clock(monkeypatch):
    from types import SimpleNamespace
    import tools.crn_tail as crn
    events = []
    devices = [torch.device(name) for name in ("cpu", "cuda:0", "cuda:0", "cuda:1")]
    model = SimpleNamespace(parameters=lambda: iter(SimpleNamespace(device=d) for d in devices))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: events.append(str(d)))
    def clock():
        events.append("clock")
        return 42.
    monkeypatch.setattr(crn.time, "monotonic", clock)
    assert crn.synchronized_time(model) == 42.
    assert events == ["cuda:0", "cuda:1", "clock"]


def test_zero_message_tail_equals_independent_native_step(warm_toy):
    model, optimizer, state, examples = warm_toy
    trial = item("zero", message="zero")
    measured = run_item(model, optimizer, state, examples, examples, trial, token_budget=3)
    restore(model, optimizer, state)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    assert optimizer.state[model.theta]["step"] == 4
    assert torch.equal(optimizer.state[model.theta]["exp_avg"], state["optimizer"]["state"][0]["exp_avg"] * 0.9)
    for batch in tail_plan(examples, 2, trial["tail_seed"], token_budget=3):
        optimizer.zero_grad(set_to_none=True)
        token_mean_backward(model, examples, batch)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    expected = surrogate_score(model, examples)
    assert measured["F_tilde_per_token"] == expected["F_tilde_per_token"]
    assert measured["message"]["update_norm"] > 0


def test_union_is_token_weighted_and_clipped_once(warm_toy):
    model, optimizer, state, examples = warm_toy
    ga, _ = message_backward(model, {"episodes": [0], "clip": False}, examples)
    gb, _ = message_backward(model, {"episodes": [1], "clip": False}, examples)
    union = {"union": [{"episodes": [0]}, {"episodes": [1]}]}
    gab, stats = message_backward(model, union, examples)
    torch.testing.assert_close(gab["theta"], 0.4 * ga["theta"] + 0.6 * gb["theta"], atol=1e-7, rtol=1e-7)
    assert stats["token_weights"] == [0.4, 0.6]
    plain, _ = message_backward(model, {"episodes": [0, 1]}, examples)
    assert torch.equal(gab["theta"], plain["theta"])
    scaled, _ = message_backward(model, {"episodes": [0], "loss_multiplier": 4, "clip": False}, examples)
    torch.testing.assert_close(scaled["theta"], ga["theta"] * 4)
    with pytest.raises(ValueError, match="disjoint"):
        message_backward(model, {"union": [{"episodes": [0]}, {"episodes": [0]}]}, examples)


def test_tail_prefix_and_common_random_stream():
    from train.sft import exposure_plan
    rows = toy_rows()
    expected = exposure_plan([2, 3], token_budget=3, passes=30, seed=19, packing="episode")
    actual = tail_plan(rows, 7, 19, token_budget=3)
    assert [b["indices"] for b in actual] == [b["indices"] for b in expected[:7]]
    pools = {"A": {"ids": list(range(5))}, "B": {"ids": list(range(10, 20))}}
    built = build_items(pools, 4, batch_size=3)
    assert built == build_items(pools, 4, batch_size=3)
    for a, b in zip(built["items"][::2], built["items"][1::2]):
        assert a["tail_seed"] == b["tail_seed"]
        assert a["message"]["episodes"] == [(i - 10) // 2 for i in b["message"]["episodes"]]
        stream = random.Random(a["replica"])
        assert a["message"]["episodes"] == [int(stream.random() * 5) for _ in range(3)]
    token_items = build_items({"A": {"ids": [0, 1], "costs": [2, 3]}}, 2, token_budget=6)
    assert all(sum([2, 3][i] for i in t["message"]["episodes"]) >= 6 for t in token_items["items"])


def merge_rows():
    return [{"item_id": name, "replica": r, "tail_seed": r, "tail_updates": 10, "tail_sha256": str(r),
             "F_tilde_per_token": value, "message": {"token_weights": [0.4, 0.6] if name == "AB" else []}}
            for r in range(3) for name, value in {"0": -4.0, "A": -3.8 + r * 0.1,
                "B": -3.7, "AB": -3.0}.items()]


def test_merge_bernstein_uses_paired_variance_and_theorem_M():
    result = merge_results(merge_rows(), B=4, metric="F_tilde_per_token",
                           unions=[{"zero": "0", "A": "A", "B": "B", "union": "AB"}])
    pair = next(p for p in result["pairs"] if (p["j"], p["k"]) == ("A", "0"))
    assert pair["Z"] == pytest.approx([0.2, 0.3, 0.4])
    assert pair["sample_variance"] == pytest.approx(0.01)
    log = math.log(4 * 3**2 / 0.05)
    assert pair["radius_a_priori"] == pytest.approx(math.sqrt(2 * .01 * log / 3) + 7 * 4 * log / 6)
    assert pair["radius_observed_range_DIAGNOSTIC"] < pair["radius_a_priori"]
    response = result["response_nonadditivity"][0]
    assert response["median_nu_resp"] == pytest.approx(abs(1 - .4 * .3 - .6 * .3))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "tail", "identity", "range"])
def test_merge_rejects_invalid_pairing(fault):
    rows = merge_rows()
    B = 4
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "tail":
        rows[0]["tail_sha256"] = "changed"
    elif fault == "identity":
        rows[0]["experiment_sha256"] = "changed"
    else:
        B = 0.01
    with pytest.raises(ValueError):
        merge_results(rows, B=B, metric="F_tilde_per_token")


def test_surrogate_normalizations_and_input_guards(warm_toy):
    model, _, _, rows = warm_toy
    score = surrogate_score(model, rows)
    assert score["F_tilde_per_traj"] == pytest.approx(score["F_tilde_per_token"] * 2.5)
    assert score["surrogate_trajectories"] == 2
    assert 0 < score["F_tilde_geo_prob"] <= 1
    with pytest.raises(ValueError, match="Diagnostic"):
        validate_examples([{**rows[0], "label_provenance": {"training_eligible": False}}])
    with pytest.raises(ValueError, match="dropout"):
        require_no_dropout(torch.nn.Sequential(torch.nn.Dropout(.1)))
    with pytest.raises(ValueError, match="ambiguous"):
        select_examples([{**r, "example_id": "same"} for r in rows], ["same"])


def test_geometric_probability_averages_trajectories_after_token_normalization(monkeypatch, warm_toy):
    import tools.crn_tail as crn
    model, _, _, rows = warm_toy
    # Two turns form trajectory A; unequal trajectory lengths distinguish all means.
    examples = [{**rows[0], "trajectory_id": "A", "loss": 2.0},
                {**rows[1], "trajectory_id": "A", "loss": 8.0},
                {**rows[0], "trajectory_id": "B", "loss": 1.0}]
    monkeypatch.setattr(crn, "turn_loss_sum", lambda model, row, device: row["loss"])
    model.eval()
    score = surrogate_score(model, examples)
    assert not model.training
    assert score["F_tilde_geo_prob"] == pytest.approx((math.exp(-2) + math.exp(-.5)) / 2)
    assert score["F_tilde_per_token"] == pytest.approx(-11 / 7)
    assert score["F_tilde_per_traj"] == -5.5
    assert score["F_tilde_macro_token_mean"] == -1.25
    for row in examples:
        row["loss"] = 0
    assert surrogate_score(model, examples)["F_tilde_geo_prob"] == 1


def test_bounded_merge_defaults_and_invalid_bounds():
    rows = [{**r, "F_tilde_geo_prob": math.exp(r["F_tilde_per_token"])} for r in merge_rows()]
    out = merge_results(rows)
    assert out["metric"] == "F_tilde_geo_prob" and out["B"] == 2
    pair = next(p for p in out["pairs"] if (p["j"], p["k"]) == ("A", "0"))
    expected = [math.exp(-3.8 + r * .1) - math.exp(-4) for r in range(3)]
    assert pair["Z"] == pytest.approx(expected)
    assert pair["radius_a_priori"] == pytest.approx(bernstein_radius(
        pair["sample_variance"], 3, 4, .05, 2))
    with pytest.raises(ValueError, match="explicit"):
        merge_results(rows, metric="F_tilde_per_token")
    with pytest.raises(ValueError, match="B >= 2"):
        merge_results(rows, B=1)
    for value in (0, -1, 1.01):
        with pytest.raises(ValueError, match="probability"):
            merge_results([{**r, "F_tilde_geo_prob": value} for r in rows])


def test_tiny_model_all_diagnostics_and_timings(checkpoint_run, capsys):
    from tools.grad_stats import gradient_statistics
    from tools.warmth_dump import warmth_report
    from tools.amplitude_prediction import amplitude_prediction
    run = checkpoint_run
    state = load_checkpoint(run)
    message = {"episodes": [0]}
    crn = run_item(run.model, run.optimizer, state, run.rows, run.rows,
                   {**item(), "tail_updates": 10}, token_budget=2, learning_rate=1e-3)
    warmth = warmth_report(run.model, run.optimizer, state, run.rows, run.rows, message)
    gradients = gradient_statistics(run.model, run.optimizer, state, run.rows, run.rows,
        candidates={"batch": message}, example_sets={"cluster": {"episodes": [0, 1]}})
    amplitude = amplitude_prediction(run.model, run.optimizer, state, run.rows, run.rows,
                                     message, [0, .5, 1, 2, 4, 8])
    assert "block:A" in warmth["breakdown"] and "block:B" in warmth["breakdown"]
    assert amplitude["counts"]["coordinates"] == sum(p.numel() for p in run.model.parameters() if p.requires_grad)
    assert_state_equal(capture_state(run.model, run.optimizer), state)
    with capsys.disabled():
        print("\nR0R3_TINY_TIMINGS=" + json.dumps({"crn_T10": crn["wall_seconds"],
            "warmth": warmth["wall_seconds"], "grad_stats": gradients["wall_seconds"],
            "amplitude": amplitude["wall_seconds"]}))


def test_sharded_cli_build_run_merge(warm_toy, tmp_path, monkeypatch):
    import tools.crn_tail as crn
    model, optimizer, state, rows = warm_toy
    loads = []
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training_state.json").write_text("{}")
    class TinyExperiment:
        def __init__(self, args):
            loads.append(args.shard)
            self.model, self.optimizer, self.state = model, optimizer, state
            self.reference, self.surrogate, self.datasets = rows, rows, {}
            self.recipe = {"tokens_per_update": 3, "packing": "episode"}
            self.checkpoint = checkpoint
        def load_messages(self, messages):
            for message in messages:
                crn.resolve_message(message, rows)
        def write_provenance(self, *args, **kwargs):
            return {"details": {"dataset_sha256": {"reference": "tiny"}}, "code_sha256": {}, "dependencies": {}}
    monkeypatch.setattr(crn, "Experiment", TinyExperiment)
    pools, items, merged = (tmp_path / p for p in ("pools.json", "items.json", "merged.json"))
    pools.write_text(json.dumps({"A": {"ids": [0]}, "B": {"ids": [1]}}))
    crn.main(["build-items", "--pools", str(pools), "--replicas", "3", "--tail-updates", "1", "--output", str(items)])
    outputs = []
    for shard in range(2):
        output = tmp_path / f"shard-{shard}.jsonl"
        outputs.append(str(output))
        crn.main(["run", "--snapshot", "unused", "--checkpoint", str(checkpoint), "--verified", "unused",
                  "--surrogate", "unused", "--items", str(items), "--shard", str(shard), "--num-shards", "2",
                  "--output", str(output)])
        for row in map(json.loads, output.read_text().splitlines()):
            timings = row["timing_seconds"]
            assert set(timings) == {"restore", "message_update", "tail_updates", "surrogate_scoring"}
            assert all(math.isfinite(t) and t >= 0 for t in timings.values())
            assert sum(timings.values()) <= row["wall_seconds"] + 1e-9
    assert loads == [0, 1]  # One experiment/base load per process, never per item.
    crn.main(["merge", "--items", str(items), "--results", *outputs, "--output", str(merged)])
    result = json.loads(merged.read_text())
    assert len(result["pairs"]) == 2 and result["replicas"] == [0, 1, 2]
    assert result["metric"] == "F_tilde_geo_prob" and result["B"] == 2
    assert_state_equal(capture_state(model, optimizer), state)


def test_r03_loader_dispatch_surrogate_isolation_and_hashes(tmp_path):
    from tools.crn_tail import Experiment
    from test_r03_sets import TinyTokenizer
    from data.manifest.build_r03_sets import dataset_entry, publish_dataset
    from sed.manifest import freeze_splits
    from sed.storage import write_json, file_hash
    manifest = {"tasks": [{"task_id": f"train/group{i}/trial", "group_id": f"train/group{i}",
        "source_split": "train", "eligible": True, "task_type": "pick_and_place_simple", "files": {}}
        for i in range(2)]}
    freeze = freeze_splits(manifest)
    visible = {"observation": "visible", "admissible_commands": ["look"], "facts": "SECRET"}
    roots = [tmp_path / "train", tmp_path / "surrogate"]
    for i, root in enumerate(roots):
        ep = {"schema_version": 2, "episode_id": str(i), "task_id": f"train/group{i}/trial", "split": "train",
              "sealed": False, "initial": {"visible": visible},
              "steps": [{"event_id": "e0", "action_raw": "Action: look", "visible": visible}]}
        entry = dataset_entry(root, ep, source="unit")
        publish_dataset(root, [entry], manifest, freeze, source="unit",
                        purpose="training" if i == 0 else "surrogate_only")
    listing = tmp_path / "surrogate.json"
    write_json(listing, {"dataset_root": "surrogate", "dataset_manifest_sha256": file_hash(roots[1] / "manifest.json")})
    context = Experiment.__new__(Experiment)
    context.datasets, context.dataset_purposes, context.inputs = {}, {}, []
    context.manifest, context.freeze, context.tokenizer = manifest, freeze, TinyTokenizer()
    context.recipe = {"max_length": 32768, "packing": "episode"}
    reference = context.load_dataset(str(roots[0]))
    context.surrogate = context.load_dataset(str(listing), purpose="surrogate_only")
    context.check_surrogate_disjoint(reference)
    assert context.surrogate[0]["label_provenance"]["training_eligible"] is False
    assert listing in context.inputs and roots[1] / "manifest.json" in context.inputs
    assert "SECRET" not in context.tokenizer.decode(reference[0]["input_ids"], skip_special_tokens=False)
    with pytest.raises(ValueError, match="both"):
        context.load_dataset(str(listing))
    with pytest.raises(ValueError, match="overlaps"):
        context.check_surrogate_disjoint(context.surrogate)


def test_merge_refuses_missing_items_fixed_horizon_and_reused_seeds():
    from sed.storage import digest
    rows = merge_rows()
    with pytest.raises(ValueError, match="exactly cover"):
        merge_results([{**r, "item_sha256": "wrong"} for r in rows], B=4,
                      expected_items=[item("A")])
    for r in rows:
        if r["replica"] == 2:
            r["tail_updates"] = 11
    with pytest.raises(ValueError, match="fixed horizon"):
        merge_results(rows, B=4, metric="F_tilde_per_token")
    rows = merge_rows()
    for r in rows:
        r["tail_seed"] = 0
    with pytest.raises(ValueError, match="fresh tail"):
        merge_results(rows, B=4, metric="F_tilde_per_token")
