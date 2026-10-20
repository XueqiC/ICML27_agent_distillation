"""SAD's real span encoder, exact CPU loss/gradients, and logical budget."""
from copy import deepcopy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from baselines.common.conversation import sequence_rows
from baselines.common.fsdp_train import batch_logps, train_loop
from baselines.common.shared_history import pack_shared_history
from baselines.sad.data import curriculum, encode_episode
from test_sad import CharTokenizer, episode
from test_validate_shared_history_gpu import tiny_model
from tools import validate_shared_history_gpu as gate


def sad_rows():
    tokenizer = CharTokenizer()
    rows = []
    for turns in (2, 3):
        source = episode("unused")
        source["task_id"] = f"train/{turns}"
        source["steps"] = [dict(source["steps"][0], event_id=f"e{i}",
            action_raw=f"Intro\nThought: {i}\nNote: MASKED\nAction: look\ntrailer") for i in range(turns)]
        row = encode_episode(source, tokenizer)
        for i, turn in enumerate(row["turns"]):
            assert tokenizer.decode([v for v in turn["labels"] if v != -100]) == (
                f"Thought: {i}\nAction: look<turn|>")
            assert turn["labels"][-1] == 106
            assert -100 in turn["labels"][turn["prompt_tokens"]:]
            assert "HIDDEN" not in tokenizer.decode(turn["input_ids"])
        packed = pack_shared_history(row)
        assert sum(v != -100 for v in packed["labels"]) == row["reason_tokens"] + row["action_tokens"]
        rows.append(row)
    return curriculum(rows)


def assert_fp32_gradient(actual, expected, name):
    # Accept absolute 1e-6, or BOTH tensor-relative L2 and maximum error <=1e-6.
    # As in test_shared_history_torch.assert_gradients, near-zero elements can
    # have large element-relative error from shared-prefix accumulation.
    assert actual is not None and torch.isfinite(actual).all(), name
    error = actual - expected
    if error.abs().max() <= 1e-6:
        return
    assert error.norm() <= 1e-6 * expected.norm(), name
    assert error.abs().max() <= 1e-6 * expected.abs().max(), name


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_sad_span_loss_and_every_parameter_gradient_fp32(backend, checkpointing):
    original = tiny_model()
    original.set_attn_implementation(backend)
    if not checkpointing:
        original.gradient_checkpointing_disable()
    packed = deepcopy(original)
    rows = sad_rows()
    losses = []
    # Match production accumulation: one backward per old turn / packed trajectory,
    # followed by a mean over trajectories, never a mean over tokens or turns.
    for model, scoring in ((original, "per_turn"), (packed, "shared_history")):
        model.train()
        total = 0.0
        for row in rows:
            units = sequence_rows(row) if scoring == "per_turn" else [row]
            for unit in units:
                loss = -batch_logps(model, [unit], "cpu", scoring=scoring).sum() / len(rows)
                total += float(loss.detach())
                loss.backward()
        losses.append(total)
    assert losses[1] == pytest.approx(losses[0], rel=1e-6, abs=1e-6)
    for (name, expected), (actual_name, actual) in zip(original.named_parameters(), packed.named_parameters()):
        assert name == actual_name and expected.dtype == actual.dtype == torch.float32
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            assert actual.grad is not None, name
            assert_fp32_gradient(actual.grad, expected.grad, name)


def test_sad_training_budget_checkpoints_and_actual_physical_positions(tmp_path):
    rows = sad_rows()
    logical = sum(len(t["input_ids"]) for row in rows for t in sequence_rows(row))
    states, checkpoints, ledgers, positions = {}, {}, {}, {}
    for scoring in ("per_turn", "shared_history"):
        model = tiny_model()
        model.gradient_checkpointing_disable()
        callbacks, forwards = [], []
        hook = model.register_forward_pre_hook(
            lambda m, args, kw: forwards.append(kw["input_ids"].numel()), with_kwargs=True)
        states[scoring] = train_loop(model, rows, tmp_path / scoring, objective="sad",
            learning_rate=2e-5, scheduler="cosine", warmup_ratio=.03, original_batch=64,
            processed_token_budget=logical + 1, packing="turn",
            checkpoint_callback=lambda m, s: callbacks.append(s.copy()),
            **({"scoring": scoring} if scoring == "per_turn" else {}))  # Check SAD's default.
        hook.remove()
        checkpoints[scoring] = callbacks
        ledgers[scoring] = [json.loads(line) for line in (tmp_path / scoring / "updates.jsonl").read_text().splitlines()]
        positions[scoring] = forwards
        state = states[scoring]
        assert state["scoring"] == scoring
        assert state["physical_policy_forward_tokens"] == sum(forwards)
        assert state["physical_reference_forward_tokens"] == state["physical_policy_recompute_forward_tokens"] == 0
        assert not state["two_pass_turn_backward"] and not state["two_pass_side_backward"]
        assert callbacks[-1]["physical_policy_forward_tokens"] == sum(forwards)
        assert callbacks[-1]["scoring"] == scoring
        assert ledgers[scoring][-1]["physical_policy_forward_tokens"] == sum(forwards)
    keys = ("processed_token_budget", "planned_processed_tokens", "checkpoint_tokens", "total_steps", "updates",
            "examples_seen", "supervised_tokens_seen", "logical_forward_tokens", "processed_token_positions",
            "realized_trajectory_passes")
    for key in keys:
        assert states["per_turn"][key] == states["shared_history"][key], key
    for old, new in zip(checkpoints["per_turn"], checkpoints["shared_history"], strict=True):
        for key in ("updates", "examples_seen", "supervised_tokens_seen", "logical_forward_tokens",
                    "processed_token_positions", "checkpoint_tokens"):
            assert old[key] == new[key], key
    for old, new in zip(ledgers["per_turn"], ledgers["shared_history"], strict=True):
        for key in ("indices", "epoch", "lr", "logical_forward_tokens", "supervised_tokens_seen", "examples_seen"):
            assert old[key] == new[key], key
        assert new["loss"] == pytest.approx(old["loss"], rel=1e-6, abs=1e-6)
    assert sum(positions["shared_history"]) < sum(positions["per_turn"])
    assert len(positions["shared_history"]) == states["shared_history"]["examples_seen"]


def test_sad_gpu_comparison_path_on_cpu():
    row = sad_rows()[1]
    model = tiny_model()
    parameters = gate.select_gradients(model)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    record = gate.compare_sad(model, row, parameters, "cpu")
    assert gate.summarize([record])["passed"]
    for scoring in gate.SCORINGS:
        path = record["paths"][scoring]
        assert path["loss"] == -path["side_sums"]["sad"]
        assert path["gradient_objective"] == pytest.approx(path["loss"], rel=1e-6)
        assert [len(v) for v in path["token_logps"]["sad"]] == [t["supervised_tokens"] for t in row["turns"]]
        for measurement in path["measurements"].values():
            assert measurement["forward_calls"] == (1 if scoring == "shared_history" else len(row["turns"]))
            assert measurement["forward_tokens"] == record["forward_tokens_per_pass"]["sad"][scoring]
    for comparison in record["comparisons"].values():
        assert comparison["loss_difference"]["relative_difference"] < 1e-6
        assert max(g["relative_l2"] for g in comparison["gradients"].values()) < 1e-6
    # Independent sign/reduction check for the actual SAD scalar differentiated.
    model.eval()
    loss = -batch_logps(model, [row], "cpu").sum()
    expected = torch.autograd.grad(loss, tuple(parameters.values()))
    value, grads = gate._gradients(model, {"sad": row}, parameters, "cpu", "shared_history")
    assert value == pytest.approx(float(loss.detach()), rel=1e-6)
    for name, reference in zip(parameters, expected):
        assert_fp32_gradient(grads[name], reference, name)
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]) and p.grad is None


def test_sad_rank_padding_counts_compute_but_not_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda values: None)
    monkeypatch.setattr(torch.distributed, "all_gather_object",
                        lambda out, value: out.__setitem__(slice(None), [value] * 2))
    row = sad_rows()[1]  # Three turns require padding even on the old path.
    logical = sum(len(t["input_ids"]) for t in sequence_rows(row))
    totals = {}
    for scoring in ("per_turn", "shared_history"):
        states, positions = [], []
        for rank in range(2):
            model = tiny_model()
            model.gradient_checkpointing_disable()
            counts = []
            hook = model.register_forward_pre_hook(
                lambda m, args, kw: counts.append(kw["input_ids"].numel()), with_kwargs=True)
            state = train_loop(model, [row], tmp_path / scoring / str(rank), objective="sad",
                learning_rate=2e-5, scheduler="cosine", warmup_ratio=0, original_batch=64,
                processed_token_budget=1, packing="turn", scoring=scoring, rank=rank, world_size=2)
            hook.remove()
            states.append(state)
            positions.append(sum(counts))
            if scoring == "shared_history" and rank == 1:
                assert all(p.grad is None or not p.grad.count_nonzero() for p in model.parameters())
        for state in states:
            assert state["processed_token_positions"] == state["logical_forward_tokens"] == logical
            assert state["examples_seen"] == 1
            assert state["supervised_tokens_seen"] == row["supervised_tokens"]
            assert state["physical_policy_forward_tokens"] == sum(positions)
            assert state["physical_policy_recompute_forward_tokens"] == 0
            assert state["zero_weight_slots"] == 1
        totals[scoring] = states[0]
    for key in ("checkpoint_tokens", "planned_processed_tokens", "total_steps"):
        assert totals["per_turn"][key] == totals["shared_history"][key]


@pytest.mark.parametrize("objective", ["sft", "steca"])
def test_shared_history_remains_disabled_for_other_training_rows(tmp_path, objective):
    with pytest.raises(ValueError, match="turn-packed DPO/IPR/SAD"):
        train_loop(tiny_model(), sad_rows(), tmp_path, objective=objective, learning_rate=1e-5,
                   scheduler="cosine", warmup_ratio=0, original_batch=1, scoring="shared_history")


@pytest.mark.parametrize("corruption", ["prompt_label", "wrong_token"])
def test_shared_history_rejects_invalid_span_labels(corruption):
    row = sad_rows()[0]
    turn = row["turns"][0]
    index = 0 if corruption == "prompt_label" else -1
    turn["labels"][index] = turn["input_ids"][index] if index == 0 else 107
    with pytest.raises(ValueError, match="exact-prefix"):
        pack_shared_history(row)
