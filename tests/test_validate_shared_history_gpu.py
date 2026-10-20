"""Offline tiny-model smoke of the GPU tool's actual encoding/comparison path.

CPU SDPA implements the packed visibility oracle here; no CPU result certifies
FlexAttention CUDA forward/backward. Only evidence retrieval is faked.
"""
from copy import deepcopy
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from baselines.common.conversation import episode_messages, preference_conversation, sequence_rows
from baselines.common.fsdp_train import batch_logps
from baselines.ipr.pairs import preference_conversation as ipr_encode
from sed.storage import digest
from test_r03_sets import TinyTokenizer
from tools import validate_shared_history_gpu as gate


def tiny_model():
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedTextConfig, Gemma4UnifiedForConditionalGeneration
    torch.set_num_threads(1)
    torch.manual_seed(42)
    text = Gemma4UnifiedTextConfig(vocab_size=1200, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        global_head_dim=8, max_position_embeddings=32768,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"], sliding_window=7)
    config = Gemma4UnifiedConfig(text_config=text, vision_config=None, audio_config=None,
                                image_token_id=1197, audio_token_id=1198, video_token_id=1199)
    config._attn_implementation = "sdpa"
    model = Gemma4UnifiedForConditionalGeneration(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def synthetic_pair(method):
    visible = {"observation": "room", "admissible_commands": ["look", "go north"]}
    episode = {"schema_version": 2, "task_id": "train/synthetic/trial_0", "split": "train", "sealed": False,
        "initial": {"visible": {**visible, "facts": "SECRET"}}, "steps": [
            {"event_id": str(i), "action_raw": reply, "visible": visible}
            for i, reply in enumerate(("Action: look", "Action: go north", "Action: look"))]}
    expert = episode_messages(episode)
    student = deepcopy(expert)
    student[-1]["content"] = "Action: go north"
    if method == "eto":
        # Include an empty visible reply retained as history, without supervision.
        student[4]["content"] = ""
        return {"task_id": episode["task_id"], "prompt": expert[:2],
                "chosen": expert[2:], "rejected": student[2:]}
    return {"task_id": episode["task_id"], "j": 1, "kind": "entire", "prompt": expert[:4],
        "chosen": student[4:5], "rejected": expert[4:], "chosen_source": "student_exploration",
        "rejected_source": "trusted_teacher_seed", "parent_seed_ids": ["synthetic-seed"],
        "agent_episode_id": "synthetic-agent"}


def stage(tmp_path, monkeypatch, method):
    directory = tmp_path / method
    directory.mkdir()
    raw = [synthetic_pair(method)] * 2
    (directory / "pairs.json").write_text(json.dumps(raw))
    (directory / "provenance.json").write_text(json.dumps({"command": method + "_pairs"}))
    (directory / "report.json").write_text("{}")

    def fake_evidence_loader(path, config):
        assert path == directory / "pairs.json"
        assert config == {"synthetic_fixture": True}
        return json.loads(path.read_text())

    monkeypatch.setattr(f"baselines.{method}.pairs.load_pair_data", fake_evidence_loader)
    return directory, raw


@pytest.mark.parametrize("method", ["eto", "ipr"])
def test_encoding_and_numerical_comparison_cpu(tmp_path, monkeypatch, method):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    directory, raw = stage(tmp_path, monkeypatch, method)
    tokenizer = TinyTokenizer()
    pairs, metadata = gate.load_pairs(directory, {"synthetic_fixture": True}, tokenizer, 1)
    expected = (preference_conversation if method == "eto" else ipr_encode)(raw[0], tokenizer, max_length=32768)
    assert pairs == [expected]
    assert metadata["selected_indices"] == [0] and metadata["available_pairs"] == 2
    assert metadata["packing"] == "turn" and metadata["max_length"] == 32768
    assert metadata["encoded_pairs_sha256"] == digest(pairs)
    for side in gate.SIDES:
        for row in sequence_rows(pairs[0][side]):
            assert "SECRET" not in tokenizer.decode(row["input_ids"], skip_special_tokens=False)
    model = tiny_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    parameters = gate.select_gradients(model)
    assert set(parameters) == {name for name, _ in model.named_parameters() if name.startswith((
        "model.language_model.layers.1.", "model.language_model.layers.2.", "model.language_model.norm."))}
    record = gate.compare_pair(model, pairs[0], parameters, "cpu")
    summary = gate.summarize([record])
    assert summary["passed"], summary
    assert summary["acceptance_comparison"] == "packing_only"
    for comparison in summary["comparisons"].values():
        assert comparison["max_side_relative_difference"] < 1e-5
        assert comparison["max_gradient_relative_l2"] < 1e-4
        assert comparison["tokens"]["mean_absolute_difference_nats_per_token"] < 1e-5
    for scoring in gate.SCORINGS:
        path = record["paths"][scoring]
        expected_tokens = sum(record["forward_tokens_per_pass"][s][scoring] for s in gate.SIDES)
        for measurement in path["measurements"].values():
            assert measurement["forward_tokens"] == expected_tokens
            assert measurement["max_memory_allocated_bytes"] is None
            assert measurement["wall_seconds"] > 0
            assert measurement["forward_calls"] == (2 if scoring == "shared_history" else
                sum(len(sequence_rows(pairs[0][s])) for s in gate.SIDES))
        for side in gate.SIDES:
            assert len(path["token_logps"][side]) == len(sequence_rows(pairs[0][side]))
            assert path["side_sums"][side] == pytest.approx(sum(path["turn_logps"][side]), rel=1e-6)
            assert [len(v) for v in path["token_logps"][side]] == [
                sum(v != -100 for v in t["labels"]) for t in sequence_rows(pairs[0][side])]
    assert record["paths"]["per_turn_flex"]["flex_attention"] == record["paths"]["shared_history"]["flex_attention"]
    assert sum(record["forward_tokens_per_pass"][s]["shared_history"] for s in gate.SIDES) < sum(
        record["forward_tokens_per_pass"][s]["per_turn"] for s in gate.SIDES)
    aggregate = gate.aggregate_measurements([record, record])
    for scoring in gate.SCORINGS:
        for phase, measurement in record["paths"][scoring]["measurements"].items():
            assert aggregate[scoring][phase]["forward_tokens"] == 2 * measurement["forward_tokens"]
            assert aggregate[scoring][phase]["max_memory_allocated_bytes"] is None
    if method == "ipr":
        turns = record["comparisons"]["packing_only"]["turns"]
        assert len(turns["chosen"]) == 1 and len(turns["rejected"]) == 2
        assert pairs[0]["chosen"]["label_provenance"]["masked_prefix_turns"] == 1
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]) and p.grad is None

    # Independently verify the incremental gradient is the literal chosen-minus-
    # rejected summed log-prob gradient, including the sign and absence of averaging.
    model.eval()
    objective = batch_logps(model, [pairs[0]["chosen"]], "cpu")[0] - batch_logps(model, [pairs[0]["rejected"]], "cpu")[0]
    expected_grads = torch.autograd.grad(objective, tuple(parameters.values()))
    value, actual = gate._gradients(model, pairs[0], parameters, "cpu", "per_turn")
    assert value == pytest.approx(float(objective.detach()), rel=1e-6)
    for name, reference in zip(parameters, expected_grads):
        torch.testing.assert_close(actual[name], reference, rtol=1e-4, atol=1e-6)
    gate.save_report(tmp_path / "cpu-smoke.json", {"scope": "CPU smoke only", "gpu_acceptance": False,
                                                   "summary": summary, "pairs": [record]})
    assert json.loads((tmp_path / "cpu-smoke.json").read_text())["summary"]["passed"]
    for count in (0, 3):
        with pytest.raises(ValueError, match="no silent undersampling"):
            gate.load_pairs(directory, {"synthetic_fixture": True}, tokenizer, count)


def passing_record():
    comparison = {"side_sums": {side: gate.scalar_difference(-100.0, -100.0) for side in gate.SIDES},
        "turns": {side: [gate.scalar_difference(-100.0, -100.0)] for side in gate.SIDES},
        "tokens": {side: gate.token_difference([-100.0], [-100.0]) for side in gate.SIDES},
        "gradients": {"weight": gate.gradient_difference(torch.ones(2), torch.ones(2))},
        "gradient_objective_difference": gate.scalar_difference(0.0, 0.0)}
    return {"comparisons": {name: deepcopy(comparison) for name in gate.COMPARISONS}}


@pytest.mark.parametrize("failure", ["side", "gradient", "nan_score", "nan_token", "nan_gradient", "missing_gradient", "zero_gradient"])
def test_gate_fails_on_numerical_mismatch_and_nonfinite(tmp_path, failure):
    record = passing_record()
    assert gate.summarize([record])["passed"]
    comparison = record["comparisons"]["packing_only"]
    if failure == "side":
        comparison["side_sums"]["chosen"] = gate.scalar_difference(-100.0, -100.101)
    elif failure == "gradient":
        comparison["gradients"]["weight"] = gate.gradient_difference(torch.ones(2), torch.ones(2) * 1.021)
    elif failure == "nan_score":
        comparison["turns"]["chosen"][0] = gate.scalar_difference(-100.0, float("nan"))
    elif failure == "nan_token":
        comparison["tokens"]["chosen"] = gate.token_difference([-100.0], [float("nan")])
    elif failure == "nan_gradient":
        comparison["gradients"]["weight"] = gate.gradient_difference(torch.ones(2), torch.tensor([float("nan"), 0.]))
    elif failure == "missing_gradient":
        comparison["gradients"]["weight"] = gate.gradient_difference(None, None)
    else:
        comparison["gradients"]["weight"] = gate.gradient_difference(torch.zeros(2), torch.ones(2))
    summary = gate.summarize([record])
    assert not summary["passed"]
    gate.save_report(tmp_path / "failure.json", {"summary": summary, "pairs": [record], "raw_nan": float("nan")})
    text = (tmp_path / "failure.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["raw_nan"] is None


def test_only_packing_magnitude_gates_acceptance():
    record = passing_record()
    for name in ("kernel_only", "combined"):
        comparison = record["comparisons"][name]
        comparison["side_sums"]["chosen"] = gate.scalar_difference(-100.0, -100.37)
        comparison["gradients"]["weight"] = gate.gradient_difference(torch.ones(2), torch.ones(2) * 1.15)
        comparison["tokens"]["chosen"] = gate.token_difference([-1.0], [-1.1])
    summary = gate.summarize([record])
    assert summary["passed"]
    assert not summary["comparisons"]["kernel_only"]["tokens_by_side"]["chosen"]["within_hf_vllm_tolerance"]
    # Even the packing token tolerance is a diagnostic, not an extra gate.
    record["comparisons"]["packing_only"]["tokens"]["chosen"] = gate.token_difference([-1.0], [-1.1])
    assert gate.summarize([record])["passed"]
    record["comparisons"]["kernel_only"]["gradients"]["weight"] = gate.gradient_difference(None, None)
    assert not gate.summarize([record])["passed"]


def test_packing_gate_inclusive_boundaries():
    record = passing_record()
    comparison = record["comparisons"]["packing_only"]
    comparison["side_sums"]["chosen"] = gate.scalar_difference(-1000.0, -1001.0)
    comparison["gradients"]["weight"] = gate.gradient_difference(torch.tensor([1000.]), torch.tensor([1020.]))
    assert gate.summarize([record])["passed"]
    comparison["side_sums"]["chosen"] = gate.scalar_difference(-1000.0, -1001.001)
    assert not gate.summarize([record])["passed"]


def test_token_mae_prevents_cancellation_and_pools_by_token_count():
    row = gate.token_difference([-2.0, -2.0], [-1.75, -2.25])
    assert row["mean_absolute_difference_nats_per_token"] == 0.25
    assert not row["within_hf_vllm_tolerance"]
    pooled = gate._pooled_token_difference([row, gate.token_difference([-2.0] * 8, [-2.0] * 8)])
    assert pooled["token_count"] == 10
    assert pooled["mean_absolute_difference_nats_per_token"] == 0.05
    assert pooled["within_hf_vllm_tolerance"]
    for a, b in (([], []), ([-1.0], [-1.0, -2.0])):
        with pytest.raises(ValueError, match="token counts"):
            gate.token_difference(a, b)
    assert not gate.summarize([])["passed"]


def test_cli_refuses_cpu_and_writes_failure_report(tmp_path, capsys):
    output = tmp_path / "gpu-rejected.json"
    assert gate.main(["--pairs", str(tmp_path / "absent"), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "error" and not report["passed"] and not report["gpu_acceptance"]
    assert report["criteria"]["side_sum_relative_difference_max"] == 1e-3
    assert report["criteria"]["acceptance_comparison"] == "packing_only"
    assert report["criteria"]["hf_vllm_token_tolerance_nats"] == 0.05
    assert report["criteria"]["gradient_relative_l2_per_tensor_max"] == 2e-2
    assert report["error_message"] == "Exactly one visible GPU and one process required"
    assert report["error_repr"] == repr(ValueError(report["error_message"]))
    assert report["settings"]["flex_attention"]["kernel_options"]["BLOCK_M"] == 32
    assert "PASS requires" in capsys.readouterr().out


def test_cli_preserves_full_multiline_exception_and_kernel_overrides(tmp_path, monkeypatch):
    message = ('out of resource: triton_tem_fused_flex_attention_0\n'
               'Required: 102400 Hardware limit: 101376. Reducing block sizes or num_stages may help.')
    error = RuntimeError(message)

    def fail_device_probe():
        raise error

    monkeypatch.setattr(torch.cuda, "device_count", fail_device_probe)
    output = tmp_path / "resource-error.json"
    assert gate.main(["--pairs", str(tmp_path), "--output", str(output),
                      "--flex-kernel-options", '{"BLOCK_M":16,"BLOCK_N":16}']) == 1
    report = json.loads(output.read_text())
    assert report["error_message"] == str(error)
    assert report["error_repr"] == repr(error)
    assert report["error_type"] == "RuntimeError"
    assert report["error_frames"][-1]["function"] == "fail_device_probe"
    assert report["settings"]["flex_attention"]["kernel_options"]["BLOCK_N"] == 16


def test_runtime_failure_preserves_completed_measurements(tmp_path, monkeypatch):
    model = tiny_model()
    pair = preference_conversation(synthetic_pair("eto"), TinyTokenizer())
    parameters = gate.select_gradients(model)
    original = gate._scores

    def fail_turn_diagnostic(*args, **kwargs):
        if kwargs.get("per_turn"):
            raise RuntimeError("synthetic kernel failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(gate, "_scores", fail_turn_diagnostic)
    output = tmp_path / "partial.json"

    def progress(phase, record):
        gate.save_report(output, {"phase": phase, "incomplete_pair": record})

    with pytest.raises(RuntimeError, match="synthetic kernel failure"):
        gate.compare_pair(model, pair, parameters, "cpu", progress=progress)
    report = json.loads(output.read_text())
    assert report["phase"] == "per_turn/no_grad_per_turn"
    assert report["incomplete_pair"]["paths"]["per_turn"]["measurements"]["no_grad_sums"]["forward_tokens"] > 0
    assert not model._forward_pre_hooks
