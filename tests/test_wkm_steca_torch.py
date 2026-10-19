"""Tiny CPU gradient, update and two-adapter save/reload checks."""
from copy import deepcopy
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
from baselines.common.fsdp_train import (prepare_lora_model, train_loop, weighted_microsteps,
    weighted_objective, batch_logps)
from test_stage2b_torch import tiny_model


def examples():
    rows = [{"input_ids": [2, 7, 8, 106], "labels": [-100, -100, 8, 106]},
            {"input_ids": [2, 4, 9, 11, 106], "labels": [-100, -100, 9, 11, 106]}]
    return [{"packing": "turn", "turns": rows, "reward": 1.01, "dataset_scale": 2.},
            {"packing": "turn", "turns": rows[:1], "reward": .99, "dataset_scale": 2.}]


def test_weighted_gradient_is_dataset_means_of_summed_logps():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = tiny_model().float().eval()
    data = examples()
    dense = sum(-e["reward"] * e["dataset_scale"] * batch_logps(model, [e], "cpu")[0] / len(data) for e in data)
    dense.backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    for row, weight in weighted_microsteps(data, [0, 1]):
        loss, _ = weighted_objective(model, row, "cpu")
        (weight * loss / len(data)).backward()
    for n, p in model.named_parameters():
        if n in expected:
            torch.testing.assert_close(p.grad, expected[n], atol=2e-5, rtol=2e-5)


def test_weighted_full_training_one_epoch_and_no_reference(tmp_path):
    torch.set_num_threads(1)
    model = tiny_model().float()
    state = train_loop(model, examples(), tmp_path, objective="steca", learning_rate=3e-6,
        scheduler="constant_with_warmup", warmup_ratio=.1, original_batch=15, epochs=1, packing="turn")
    assert state["examples_seen"] == 2 and state["supervised_tokens_seen"] == 7
    assert state["physical_reference_forward_tokens"] == 0 and state["epochs"] == 1
    assert state["loss_reduction"] == "reward_weighted_summed_logps_dataset_means"


def test_wkm_two_fresh_qv_adapters_share_base_and_reload(tmp_path):
    from peft import PeftModel
    from train.modeling import load_student
    torch.set_num_threads(1)
    base = tmp_path / "base"
    tiny_model().save_pretrained(base)
    models = []
    for name in ("agent", "world"):
        model = prepare_lora_model(base, device="cpu", adaptation="wkm_lora")
        config = model.peft_config["default"]
        assert (config.r, config.lora_alpha, config.lora_dropout) == (8, 16, 0)
        assert {n.rsplit(".",1)[-1] for n in config.target_modules} == {"q_proj", "v_proj"}
        frozen = {n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
        train_loop(model, [{"packing":"turn", "turns": examples()[0]["turns"]}], tmp_path / (name + "-train"),
            objective="sft", learning_rate=1e-4, scheduler="cosine", warmup_ratio=0, original_batch=32,
            epochs=3, packing="turn")
        assert all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if n in frozen)
        model.save_pretrained(tmp_path / name)
        loaded = PeftModel.from_pretrained(load_student(base, device="cpu", dtype="float32"), tmp_path / name)
        ids = torch.tensor([[2,7,8]])
        with torch.no_grad():
            torch.testing.assert_close(model.eval()(input_ids=ids).logits, loaded.eval()(input_ids=ids).logits)
        models.append(loaded)
    assert models[0] is not models[1]
