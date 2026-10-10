"""Offline CPU checks for zero-LR moments, exact replay, and optimizer response."""
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from sed.seed_evidence import artifact_hashes
from sed.storage import digest, write_json
from train.modeling import attach_lora, load_student
from train.optimizer_state import (adamw, bitwise_equal, capture_state, learning_rate_at,
                                   load_checkpoint_state, load_update_state, restore_state,
                                   trainable_state, warmth_statistics)
from train.sft import deterministic_seed, exposure_plan, moment_warmup, train_loop
from tools.adam_response_probe import (MemoryLedger, adam_float64_delta, gradients, kappa_statistics,
                                       one_step_delta, parse_state, planned_pairs, reference_check,
                                       response_metrics, restore_fidelity, run_probe)
from test_stage2b_torch import tiny_model


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))

    def forward(self, *, input_ids, logits_to_keep, **_):
        return SimpleNamespace(logits=self.theta[None, None, :].expand(1, len(logits_to_keep), 4))

    def save_pretrained(self, output, **_):
        from safetensors.torch import save_file
        save_file({"theta": self.theta.detach()}, output / "model.safetensors")


def toy_rows():
    return [{"input_ids": [2, 0, 0], "labels": [-100, 0, 0], "supervised_tokens": 2},
            {"input_ids": [2, 1, 1, 1], "labels": [-100, 1, 1, 1], "supervised_tokens": 3}]


def toy_plan(rows, passes=3):
    return exposure_plan([r["supervised_tokens"] for r in rows], passes=passes, token_budget=1)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    torch.set_num_threads(1)


def test_moment_warmup_preserves_bits_advances_moments_and_cycles():
    model, rows = ToyModel(), toy_rows()
    plan = toy_plan(rows, passes=1)
    optimizer = adamw(model.parameters(), 1e-5)
    before, ledger = trainable_state(model), MemoryLedger()
    report = moment_warmup(model, optimizer, rows, plan, 7, gradient_clip=0.1, ledger=ledger)
    assert all(bitwise_equal(p, before[n]) for n, p in model.named_parameters())
    assert optimizer.param_groups[0]["lr"] == 1e-5
    state = optimizer.state[model.theta]
    assert int(state["step"]) == 7
    assert state["exp_avg"].count_nonzero() and (state["exp_avg_sq"] > 0).all()
    assert [r["plan_index"] for r in ledger.rows] == [0, 1, 0, 1, 0, 1, 0]
    assert all(r["learning_rate"] == 0 and r["clip_active"] for r in ledger.rows)
    assert report["supervised_tokens"] == sum(r["supervised_tokens"] for r in ledger.rows)
    assert report["real_update_data_cursor"] == 0 and not report["counted_in_supervised_tokens_seen"]


def test_default_options_leave_plan_ledger_checkpoint_bytes_identical(tmp_path, monkeypatch):
    import train.sft as sft
    snapshots = []
    for name, options in (("implicit", {}), ("explicit", {"moment_warmup_steps": 0,
                           "lr_warmup_steps": 0, "save_every_update_through": 0, "log_warmth": False})):
        deterministic_seed(11)
        clock = itertools.count(1)
        monkeypatch.setattr(sft.time, "monotonic", lambda: next(clock))
        rows = toy_rows()
        plan = toy_plan(rows)
        output = tmp_path / name
        output.mkdir()
        write_json(output / "exposure_plan.json", plan)
        train_loop(ToyModel(), rows, plan, output, identity_sha256="unit", **options)
        snapshots.append({str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()})
        assert not (output / "moment_warmup.jsonl").exists()
        assert not list(output.glob("state-update-*"))
    assert snapshots[0] == snapshots[1]


def test_lr_schedule_and_separate_warmup_checkpoint_accounting(tmp_path):
    rows = toy_rows()
    plan = toy_plan(rows)
    checkpoints = train_loop(ToyModel(), rows, plan, tmp_path, identity_sha256="unit",
                             learning_rate=1e-3, lr_warmup_steps=3, moment_warmup_steps=7,
                             save_every_update_through=2)
    ledger = [json.loads(line) for line in (tmp_path / "updates.jsonl").read_text().splitlines()]
    assert [r["learning_rate"] for r in ledger] == [1e-3 / 3, 1e-3 * 2 / 3, 1e-3, 1e-3, 1e-3, 1e-3]
    assert [learning_rate_at(1e-3, 3, i) for i in range(1, 7)] == [r["learning_rate"] for r in ledger]
    final = checkpoints[-1]
    assert final["supervised_tokens"] == 15 and final["updates"] == 6
    assert final["moment_warmup_exposure"]["steps"] == 7
    state = load_update_state(tmp_path / "state-update-2", plan=plan, identity_sha256="unit")
    assert (state["data_cursor"], state["updates"], state["step"]) == (2, 2, 9)
    assert int(next(iter(state["optimizer"]["state"].values()))["step"]) == 9
    assert state["scheduler"] == "linear_warmup_then_constant"
    assert not (tmp_path / "state-update-3").exists()


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_tiny_gemma_lora_saved_states_restore_next_update_bitwise(tmp_path, dropout):
    deterministic_seed(17)
    base = tmp_path / "base"
    tiny_model().save_pretrained(base)
    model = attach_lora(load_student(base, device="cpu", dtype="float32"), rank=2, alpha=4, dropout=dropout)
    rows = [{"input_ids": [2, 8, 9, 10, 106], "labels": [-100, -100, -100, 10, 106], "supervised_tokens": 2},
            {"input_ids": [2, 5, 6, 11, 12, 106], "labels": [-100, -100, -100, 11, 12, 106], "supervised_tokens": 3}]
    plan = toy_plan(rows)
    output = tmp_path / "train"
    initial = trainable_state(model)
    optimizer = adamw([p for p in model.parameters() if p.requires_grad], 1e-3)
    moment_warmup(model, optimizer, rows, plan, 3)
    assert all(bitwise_equal(p, initial[n]) for n, p in model.named_parameters() if p.requires_grad)
    train_loop(model, rows, plan, output, identity_sha256="unit", learning_rate=1e-3,
               moment_warmup_steps=3, lr_warmup_steps=3, save_every_update_through=3)
    first = load_update_state(output / "state-update-1", plan=plan, identity_sha256="unit")
    second = load_update_state(output / "state-update-2", plan=plan, identity_sha256="unit")
    model = attach_lora(load_student(base, device="cpu", dtype="float32"), rank=2, alpha=4, dropout=dropout)
    optimizer = adamw([p for p in model.parameters() if p.requires_grad], 1e-3)
    restore_state(model, optimizer, first)
    assert all(bitwise_equal(p, first["trainable"][n]) for n, p in model.named_parameters() if p.requires_grad)
    assert bitwise_equal(torch.get_rng_state(), first["torch_rng"])
    assert random.getstate() == first["python_rng"]
    frozen_copy = deepcopy(first)
    check = restore_fidelity(model, optimizer, first, second, rows, plan)
    assert check["bitwise_equal"] and check["max_abs_difference"] == 0
    for key, old in frozen_copy["optimizer"]["state"].items():
        assert all(bitwise_equal(tensor, first["optimizer"]["state"][key][name]) for name, tensor in old.items())
    result = run_probe(model, rows, plan, output, ["update:1"], pairs=1,
                       learning_rate=1e-3, lr_warmup_steps=3, identity_sha256="unit")
    assert result["states"][0]["restore_fidelity"]["bitwise_equal"]
    pair = result["states"][0]["pairs"][0]
    assert pair["gradient_union_check"]["passed"], pair["gradient_union_check"]
    assert pair["SGD_unclipped_float64_reference"]["nu"] < 1e-6
    assert (output / "state-update-1" / "adapter_model.safetensors").is_file()


def test_probe_sgd_null_cold_sign_map_and_warm_affinity(tmp_path):
    model, rows = ToyModel(), toy_rows()
    plan = toy_plan(rows)
    optimizer = adamw(model.parameters(), 1e-5)
    state = capture_state(model, optimizer)
    gradient = gradients(model, rows, plan[0], 1.0)["unclipped"]
    delta = one_step_delta(model, optimizer, state, gradient, kind="AdamW", learning_rate=1e-5)
    expected = -1e-5 * gradient[0] / (gradient[0].abs() + 1e-8)
    torch.testing.assert_close(delta, expected, rtol=1e-12, atol=1e-18)
    restore_state(model, optimizer, state)
    result = run_probe(model, rows, plan, tmp_path, ["cold", "warmup:128"], pairs=2)
    cold, warm = result["states"]
    for row in result["states"]:
        for pair in row["pairs"]:
            assert pair["responses"]["SGD_unclipped"]["nu"] < 1e-6
            assert pair["responses"]["SGD_unclipped"]["nu_centered"] < 1e-6
            for clipping in ("unclipped", "clipped"):
                assert pair["responses"][f"SGD_{clipping}"]["dtheta_norm"]["0"] == 0
                assert pair[f"AdamW_{clipping}_float64_reference"]["native_delta_check"]["passed"]
            assert pair["gradient_union_check"]["passed"]
            assert set(pair["indices"]["A"]).isdisjoint(pair["indices"]["B"])
    assert cold["pairs"][0]["kappa"]["unclipped"]["q0.5"] == "inf"
    assert cold["pairs"][0]["kappa"]["unclipped"]["fraction_v_zero"] == 1
    assert warm["pairs"][0]["kappa"]["unclipped"]["fraction_v_zero"] == 0
    assert cold["pairs"][0]["responses"]["AdamW_unclipped"]["nu"] > 0.1
    assert cold["pairs"][0]["responses"]["AdamW_unclipped"]["nu_centered"] > 0.1
    metrics = warm["pairs"][0]["responses"]["AdamW_unclipped"]
    assert metrics["momentum_share"] > 0
    assert metrics["nu_centered"] == pytest.approx(metrics["affine_residual_norm"] / metrics["centered_dtheta_AB_norm"])
    assert metrics["nu_centered"] > metrics["nu"]
    assert warm["pairs"][0]["responses"]["AdamW_unclipped"]["nu"] < 0.01
    assert warm["probe_batch_cursor"] == 128 and warm["data_cursor"] == 0
    assert torch.equal(model.theta, torch.zeros_like(model.theta))
    json.dumps(result, allow_nan=False)


def test_clipping_is_a_separate_nonlinearity(tmp_path):
    rows = toy_rows()
    result = run_probe(ToyModel(), rows, toy_plan(rows), tmp_path, ["cold"], pairs=1, gradient_clip=0.1)
    pair = result["states"][0]["pairs"][0]
    assert all(pair["clip_active"].values())
    assert pair["responses"]["SGD_unclipped"]["nu"] < 1e-6
    assert pair["responses"]["SGD_clipped"]["nu"] > 0.1


def test_kappa_quantiles_include_zero_moments():
    result = kappa_statistics(torch.tensor([0., 1., 2., 3.]).double(), torch.tensor([0., 1., 0., 9.]).double())
    assert result["fraction_v_zero"] == result["fraction_gt_1"] == 0.5
    assert result["q0.5"] == pytest.approx(0.001 / 0.999)
    assert result["q0.9"] == result["q0.99"] == result["max"] == "inf"


def test_disjoint_plan_pairing_reports_skips_and_rejects_impossible_pairs():
    plan = [{"indices": [0]}, {"indices": [0, 1]}, {"indices": [2]}]
    pair, = planned_pairs(plan, 0, 1)
    assert pair["plan_positions"] == [0, 2] and pair["skipped_positions"] == [1]
    with pytest.raises(ValueError, match="no disjoint"):
        list(planned_pairs(plan[:2], 0, 1))


def test_restore_rejects_tampering_plan_and_provenance(tmp_path):
    rows = toy_rows()
    plan = toy_plan(rows)
    train_loop(ToyModel(), rows, plan, tmp_path, identity_sha256="unit", save_every_update_through=1)
    path = tmp_path / "state-update-1"
    with pytest.raises(ValueError, match="plan differs"):
        load_update_state(path, plan=plan[::-1])
    with pytest.raises(ValueError, match="provenance differs"):
        load_update_state(path, identity_sha256="other")
    (path / "extra").write_text("tampered")
    with pytest.raises(ValueError, match="artifacts changed"):
        load_update_state(path)


@pytest.mark.parametrize("option", ["moment_warmup_steps", "lr_warmup_steps", "save_every_update_through"])
def test_negative_options_rejected(tmp_path, option):
    rows = toy_rows()
    with pytest.raises(ValueError, match="nonnegative"):
        train_loop(ToyModel(), rows, toy_plan(rows), tmp_path, identity_sha256="unit", **{option: -1})


@pytest.mark.parametrize("enabled", [False, True])
def test_sft_cli_records_only_enabled_options(tmp_path, monkeypatch, enabled):
    import train.sft as sft
    config = {"training": {"thinking_mode": "off", "student_step_token_cap": 1024},
              "paths": {"manifest": "manifest", "split_freeze": "freeze"}}
    rows = [dict(row, turn_count=1, label_provenance={"training_eligible": True}) for row in toy_rows()]
    tokenizer = SimpleNamespace(chat_template="template")
    captured = {}
    monkeypatch.setattr(sft, "load_config", lambda _: config)
    monkeypatch.setattr(sft, "snapshot_metadata", lambda _: {"path": "base"})
    monkeypatch.setattr(sft, "deterministic_seed", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sft, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(sft, "load_frozen", lambda *_: ({}, {}))
    monkeypatch.setattr(sft, "build_verified_dataset", lambda *_, **__: rows)
    monkeypatch.setattr(sft, "mask_report", lambda *_: {"supervised_tokens_per_pass": 5,
                                                      "estimated_forward_tokens_per_pass": 7})
    monkeypatch.setattr(sft, "verified_collection_inputs", lambda *_: [])
    monkeypatch.setattr(sft, "write_provenance", lambda *_, **kw: {"config": kw["config"]})
    monkeypatch.setattr(sft, "load_student", lambda _: None)
    monkeypatch.setattr(sft, "attach_lora", lambda *_, **__: ToyModel())
    def train(*_, **kwargs):
        captured.update(kwargs)
        return [{"path": "checkpoint", "supervised_tokens": 5}]
    monkeypatch.setattr(sft, "train_loop", train)
    args = ["--snapshot", "base", "--verified", "verified", "--output", str(tmp_path / "run"), "--passes", "1"]
    if enabled:
        args += ["--moment-warmup-steps", "3", "--lr-warmup-steps", "2", "--save-every-update-through", "4", "--log-warmth"]
    sft.main(args)
    recipe = config["sft"]
    if enabled:
        assert recipe["scheduler"] == "linear_warmup_then_constant"
        assert recipe["log_warmth"] is captured["log_warmth"] is True
        for key, value in {"moment_warmup_steps": 3, "lr_warmup_steps": 2, "save_every_update_through": 4}.items():
            assert recipe[key] == captured[key] == value
        assert recipe["moment_warmup_exposure"]["supervised_tokens"] == 15
        assert not recipe["moment_warmup_exposure"]["counted_in_supervised_tokens_seen"]
    else:
        assert recipe["scheduler"] == "constant"
        assert all(key not in recipe for key in ("moment_warmup_steps", "lr_warmup_steps",
                   "save_every_update_through", "moment_warmup_exposure", "log_warmth"))
        assert captured["log_warmth"] is False


def test_zero_gradient_decays_moments_and_reference_matches_nondefault_adam():
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.7, 0.8), eps=0.2,
                                  weight_decay=0, foreach=False, fused=False)
    model.theta.grad = torch.tensor([1., -2., 3., -4.], dtype=torch.float64)
    optimizer.step()
    state = capture_state(model, optimizer)
    zero = torch.zeros_like(model.theta)
    native = {}
    reference = {}
    for key, value in {"A": -zero + 0.3, "B": -zero - 0.1, "AB": zero + 0.1, "0": zero}.items():
        native[key] = one_step_delta(model, optimizer, state, [value], kind="AdamW", learning_rate=0.01)
        reference[key] = adam_float64_delta(state, [value], 0.01)
        torch.testing.assert_close(native[key], reference[key], rtol=1e-12, atol=1e-17)
    assert native["0"].norm() > 0
    before = state["optimizer"]["state"][0]
    after = optimizer.state[model.theta]
    assert after["step"] == before["step"] + 1
    assert torch.equal(after["exp_avg"], before["exp_avg"] * 0.7)
    assert torch.equal(after["exp_avg_sq"], before["exp_avg_sq"] * 0.8)
    assert reference_check(reference, native)["passed"]
    native["A"] = native["A"] + 1
    assert not reference_check(reference, native)["passed"]


def test_response_metrics_keep_undefined_ratios_null():
    zero = torch.zeros(4, dtype=torch.float64)
    deltas = dict.fromkeys(("A", "B", "AB", "0"), zero)
    metrics = response_metrics(deltas, deltas, [0.4, 0.6])
    for key in ("nu", "nu_centered", "nu_spread", "momentum_share", "cos_centered_A_B", "cos_dtheta_A_B"):
        assert metrics[key] is None


def gemma_rows():
    return [{"input_ids": [2, 8, 9, 10, 106], "labels": [-100, -100, -100, 10, 106], "supervised_tokens": 2},
            {"input_ids": [2, 5, 6, 11, 12, 106], "labels": [-100, -100, -100, 11, 12, 106], "supervised_tokens": 3}]


@pytest.fixture
def checkpoint_run(tmp_path, request):
    deterministic_seed(17)
    model = attach_lora(tiny_model(), rank=2, alpha=4)
    rows = gemma_rows()
    plan = exposure_plan([2, 3], passes=2, token_budget=1, checkpoint_tokens=[2])
    warmup = getattr(request, "param", 3)
    recipe = {"dropout": 0.0, "learning_rate": 1e-3, "lr_warmup_steps": 3,
              "gradient_clip": 1.0, "plan_sha256": digest(plan), "betas": [0.9, 0.999], "eps": 1e-8}
    if warmup:
        recipe["moment_warmup_steps"] = warmup
    provenance = {"config": {"sft": recipe}}
    output = tmp_path / "run"
    output.mkdir()
    write_json(output / "provenance.json", provenance)
    write_json(output / "exposure_plan.json", plan)
    train_loop(model, rows, plan, output, identity_sha256=digest(provenance), learning_rate=1e-3,
               moment_warmup_steps=warmup, lr_warmup_steps=3, save_every_update_through=4)
    optimizer = adamw([p for p in model.parameters() if p.requires_grad], 1e-3)
    path = output / f"checkpoint-tokens-{plan[0]['cumulative_supervised_tokens']}"
    return SimpleNamespace(model=model, optimizer=optimizer, rows=rows, plan=plan, path=path,
                           output=output, provenance=provenance, warmup=warmup)


def load_checkpoint(run):
    return load_checkpoint_state(run.model, run.optimizer, run.path, plan=run.plan, provenance=run.provenance)


@pytest.mark.parametrize("checkpoint_run", [0, 3], indirect=True)
def test_checkpoint_load_is_bitwise_and_uses_update_cursor(checkpoint_run):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    run = checkpoint_run
    state = load_checkpoint(run)
    saved = load_file(run.path / "adapter_model.safetensors")
    actual = get_peft_model_state_dict(run.model, save_embedding_layers=False)
    assert actual.keys() == saved.keys()
    assert all(bitwise_equal(actual[key], saved[key]) for key in saved)
    update = load_update_state(run.output / "state-update-1", plan=run.plan)
    assert all(bitwise_equal(state["trainable"][key], value) for key, value in update["trainable"].items())
    assert all(float(entry["step"]) == 1 + run.warmup for entry in run.optimizer.state.values())
    assert state["data_cursor"] == state["updates"] == 1
    assert bitwise_equal(torch.get_rng_state(), state["torch_rng"])
    token = run.plan[0]["cumulative_supervised_tokens"]
    final_token = run.plan[-1]["cumulative_supervised_tokens"]
    assert parse_state(f"checkpoint:{token}") == ("checkpoint", token)
    report = run_probe(run.model, run.rows, run.plan, run.output, [f"checkpoint:{token}", f"checkpoint:{final_token}"],
                       pairs=1, learning_rate=1e-3, lr_warmup_steps=3, identity_sha256=digest(run.provenance))
    first, final = report["states"]
    assert first["restore_fidelity"]["bitwise_equal"], first["restore_fidelity"]
    assert first["probe_batch_cursor"] == first["data_cursor"] == 1
    assert first["pairs"][0]["plan_positions"][0] == 1
    assert final["restore_fidelity"]["status"] == "unavailable"


@pytest.mark.parametrize("fault, message", [
    ("artifact", "artifacts changed"), ("identity", "provenance differs"),
    ("plan", "plan differs"), ("dropout", "dropout == 0"),
    ("missing", "missing/extra keys"), ("extra", "missing/extra keys"),
    ("shape", "shape/dtype mismatch"), ("dtype", "shape/dtype mismatch"),
    ("step", "optimizer step differs"), ("param_count", "param count/order mismatch"),
    ("moment_shape", "moment shape/dtype mismatch")])
def test_checkpoint_rejects_invalid_artifacts_and_state(checkpoint_run, fault, message):
    from safetensors.torch import load_file, save_file
    run = checkpoint_run
    metadata_path = run.path / "training_state.json"
    metadata = json.loads(metadata_path.read_text())
    if fault == "artifact":
        (run.path / "unexpected.txt").write_text("tampered")
    elif fault == "identity":
        metadata["identity_sha256"] = "wrong"
    elif fault == "plan":
        write_json(run.output / "exposure_plan.json", run.plan[::-1])
    elif fault == "dropout":
        run.provenance["config"]["sft"]["dropout"] = 0.2
        metadata["identity_sha256"] = digest(run.provenance)
        write_json(run.output / "provenance.json", run.provenance)
    elif fault in {"step", "param_count", "moment_shape"}:
        payload = torch.load(run.path / "optimizer.pt", weights_only=True)
        if fault == "step":
            next(iter(payload["optimizer"]["state"].values()))["step"] += 1
        elif fault == "moment_shape":
            next(iter(payload["optimizer"]["state"].values()))["exp_avg_sq"] = torch.zeros(1)
        else:
            payload["optimizer"]["param_groups"][0]["params"].pop()
        torch.save(payload, run.path / "optimizer.pt")
    else:
        saved = load_file(run.path / "adapter_model.safetensors")
        key = next(iter(saved))
        if fault == "missing":
            saved.pop(key)
        elif fault == "extra":
            saved["extra.weight"] = torch.zeros(1)
        elif fault == "shape":
            saved[key] = saved[key].flatten()
        elif fault == "dtype":
            saved[key] = saved[key].double()
        save_file(saved, run.path / "adapter_model.safetensors")
    if fault != "artifact":
        metadata["artifact_sha256"] = {k: v for k, v in artifact_hashes(run.path).items() if k != "training_state.json"}
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match=message):
        load_checkpoint(run)


@pytest.mark.parametrize("warmup", [0, 3])
def test_log_warmth_is_bitwise_neutral_for_training_and_checkpoints(tmp_path, monkeypatch, warmup):
    import train.sft as sft
    deterministic_seed(17)
    initial = attach_lora(tiny_model(), rank=2, alpha=4, dropout=0.2)
    rows, snapshots, weights, rngs = gemma_rows(), [], [], []
    plan = toy_plan(rows, passes=1)
    for enabled in (False, True):
        deterministic_seed(11)
        clock = itertools.count(1)
        monkeypatch.setattr(sft.time, "monotonic", lambda: next(clock))
        model = deepcopy(initial)
        output = tmp_path / str(enabled)
        train_loop(model, rows, plan, output, identity_sha256="unit", log_warmth=enabled,
                   gradient_clip=0.01, moment_warmup_steps=warmup, save_every_update_through=2)
        weights.append(trainable_state(model))
        rngs.append(torch.get_rng_state().clone())
        # Includes adapter bytes, optimizer/RNG payloads, hashes and metadata.
        snapshots.append({str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*")
                          if p.is_file() and p.suffix != ".jsonl"})
        ledgers = [output / "updates.jsonl"]
        if warmup:
            ledgers.insert(0, output / "moment_warmup.jsonl")
        all_rows = [json.loads(line) for path in ledgers for line in path.read_text().splitlines()]
        assert all(("warmth" in row) == enabled for row in all_rows)
        if enabled:
            first = all_rows[0]["warmth"]
            assert first["fraction_v_zero"] == first["fraction_gt_1"] == 1
            assert first["q0.5"] == first["q0.9"] == first["q0.99"] == first["max"] == "inf"
            assert first["gradient_share"] == 1
    assert snapshots[0] == snapshots[1]
    assert all(bitwise_equal(weights[0][key], weights[1][key]) for key in weights[0])
    assert bitwise_equal(*rngs)


def test_warmth_uses_consumed_gradient_and_pre_step_raw_moments():
    model = ToyModel()
    optimizer = adamw(model.parameters(), 0.01)
    model.theta.grad = torch.zeros_like(model.theta)
    assert warmth_statistics(optimizer)["gradient_share"] is None
    assert not optimizer.state  # Observation must not lazily initialize Adam state.
    model.theta.grad = torch.ones_like(model.theta)
    optimizer.step()
    model.theta.grad = torch.tensor([4., -3., 2., -1.], dtype=torch.float64)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    before = capture_state(model, optimizer)
    gradient = model.theta.grad.clone()
    result = warmth_statistics(optimizer)
    m, v = optimizer.state[model.theta]["exp_avg"], optimizer.state[model.theta]["exp_avg_sq"]
    assert result == {**kappa_statistics(gradient, v),
                      "gradient_share": pytest.approx(float((0.1 * gradient).norm() / (0.9 * m + 0.1 * gradient).norm()))}
    assert bitwise_equal(model.theta.grad, gradient)
    for key, value in before["optimizer"]["state"][0].items():
        assert bitwise_equal(value, optimizer.state[model.theta][key])
    assert bitwise_equal(torch.get_rng_state(), before["torch_rng"])
    assert random.getstate() == before["python_rng"]


@pytest.mark.parametrize("options", [{}, {"MOMENT_WARMUP_STEPS": "0", "LR_WARMUP_STEPS": "0",
    "SAVE_EVERY_UPDATE_THROUGH": "0", "LOG_WARMTH": "0"}, {"MOMENT_WARMUP_STEPS": "3",
    "LR_WARMUP_STEPS": "2", "SAVE_EVERY_UPDATE_THROUGH": "4", "LOG_WARMTH": "1"}])
def test_slurm_argv_defaults_and_options(tmp_path, options):
    root = Path(__file__).resolve().parents[1]
    model, verified, run = (tmp_path / name for name in ("model", "verified", "run"))
    for path in (model, verified, run):
        path.mkdir()
    for path in (verified / "manifest.json", run / "provenance.json", run / "exposure_plan.json"):
        path.write_text("{}")
    capture = tmp_path / "capture"
    capture.write_text('#!/bin/bash\nprintf "%s\\0" "$@"\n')
    capture.chmod(0o755)
    config = tmp_path / "hpg.env"
    values = {"PROJECT_ROOT": str(root), "ENABLE_THINKING": "off", "MAX_TOKENS": "1024",
              "TRAIN_PYTHON": str(capture), "MODEL_PATH": str(model), "VERIFIED": str(verified),
              "TRAIN_OUT": "output with spaces", "TOKENS_PER_UPDATE": "512", "LEARNING_RATE": "0.00001",
              "TRAIN_SEED": "7", "MAX_LENGTH": "32768", "TRAIN_PASSES": "2", "TRAIN_TOKENS": "100",
              "CHECKPOINT_TOKENS": "20 80"}
    config.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()))
    env = {"PATH": os.defpath, "HPG_CONFIG": str(config), **options}
    script = root / "scripts/hpg_train_sft.slurm"
    subprocess.run(["bash", "-n", str(script)], check=True, env=env)
    result = subprocess.run(["bash", str(script)], check=True, env=env, capture_output=True)
    expected = ["train/sft.py", "--snapshot", str(model), "--verified", str(verified),
                "--output", "output with spaces", "--tokens-per-update", "512", "--learning-rate", "0.00001",
                "--seed", "7", "--max-length", "32768", "--passes", "2", "--supervised-tokens", "100",
                "--checkpoint-tokens", "20", "80"]
    if options.get("LOG_WARMTH") == "1":
        expected += ["--moment-warmup-steps", "3", "--lr-warmup-steps", "2", "--save-every-update-through", "4", "--log-warmth"]
    assert result.stdout == b"".join(value.encode() + b"\0" for value in expected)
    probe = root / "scripts/hpg_adam_probe.slurm"
    env.update(PROBE_RUN=str(run), PROBE_STATES="cold checkpoint:20 update:1")
    subprocess.run(["bash", "-n", str(probe)], check=True, env=env)
    result = subprocess.run(["bash", str(probe)], check=True, env=env, capture_output=True)
    argv = result.stdout.decode().split("\0")
    position = argv.index("--states")
    assert argv[position + 1:position + 4] == ["cold", "checkpoint:20", "update:1"]
