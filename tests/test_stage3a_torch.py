"""Numerical tests use tiny random Gemma, local tokenizer, CPU only."""
import ast
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from test_stage2b_torch import tokenizer, tiny_model, trajectory
from baselines.common.conversation import conversation, episode_messages, preference_conversation
from baselines.common.fsdp_train import (batch_logps, sft_objective, dpo_objective, train_loop,
                                         export_checkpoint, prepare_model, validate_checkpoint)
from baselines.common.schedule import dose, lr_multiplier, rank_slots
from baselines.eto.official_loss import OfficialLoss
from sed.manifest import load_frozen
from sed.react import build_messages
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import digest, write_json
from train.dataset import EMPTY_THOUGHT


def test_every_real_seed_turn_matches_eval_prefix(tokenizer):
    root = Path("data/verified/seeds_k20_s0")
    if not (root / "manifest.json").is_file():
        pytest.skip("No real seed collection installed")
    manifest, freeze = load_frozen("data/manifest/alfworld_manifest.json", "data/manifest/alfworld_split_freeze.json")
    collection, _, evidence = load_verified_collection(root, manifest, freeze)
    episode = checked_episode(evidence, collection["verified"][0])
    row = conversation(episode, tokenizer, source="trusted_teacher_seed")
    history, visible = [], episode["initial"]["visible"]
    for i, (span, step) in enumerate(zip(row["assistant_spans"], episode["steps"])):
        prompt = tokenizer.apply_chat_template(build_messages(history, {}, visible), tokenize=False,
                      add_generation_prompt=True, enable_thinking=False)
        if i < len(episode["steps"]) - 1:
            assert prompt.endswith(EMPTY_THOUGHT)
            prompt = prompt[:-len(EMPTY_THOUGHT)]
        assert row["input_ids"][:span["start"]] == tokenizer.encode(prompt, add_special_tokens=False)
        assert row["labels"][span["stop"] - 1] == 106
        assert tokenizer.decode(row["labels"][span["start"]:span["stop"]]) == step["action_raw"].strip() + "<turn|>"
        history.append({"visible": visible, "reply": step["action_raw"]})
        visible = step["visible"]


def test_dpo_shared_prompt_and_mask_roundtrip(tokenizer):
    chosen = trajectory()
    rejected = deepcopy(chosen)
    rejected["steps"] = rejected["steps"][:1]
    prompt = episode_messages(chosen)[:2]
    pair = {"task_id": chosen["task_id"], "prompt": prompt,
            "chosen": episode_messages(chosen)[2:], "rejected": episode_messages(rejected)[2:]}
    rows = preference_conversation(pair, tokenizer)
    for side, episode in [("chosen", chosen), ("rejected", rejected)]:
        expected = conversation(episode, tokenizer, source="student_exploration")
        actual = rows[side]
        assert actual["input_ids"] == expected["input_ids"] and actual["labels"] == expected["labels"]
        assert rows["prompt_input_ids"] + actual["continuation_input_ids"] == actual["input_ids"]
        assert actual["labels"][-1] == 106
        assert "PRIVATE_PDDL" not in tokenizer.decode(actual["input_ids"])
    assert all(t["source"] == "student_exploration" for t in rows["rejected"]["label_provenance"]["turns"])
    with pytest.raises(ValueError, match="no truncation"):
        preference_conversation(pair, tokenizer, max_length=1)
    pair["prompt"][1]["content"] = '{"observation":"secret", "admissible_commands":[], "facts":"PRIVATE"}'
    with pytest.raises(ValueError, match="allowlisted"):
        preference_conversation(pair, tokenizer)


@pytest.mark.parametrize("kind,ratio,total", [(k, r, n) for k in ("cosine", "constant_with_warmup")
                                             for r in (0, .03, .1, 1) for n in (3, 27)])
def test_schedule_matches_hf_including_step_zero(kind, ratio, total):
    from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    kwargs = {"num_warmup_steps": math.ceil(total * ratio)}
    factory = get_constant_schedule_with_warmup
    if kind == "cosine":
        kwargs["num_training_steps"] = total
        factory = get_cosine_schedule_with_warmup
    schedule = factory(optimizer, **kwargs)
    for step in range(total + 2):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(lr_multiplier(step, scheduler=kind, total_steps=total, warmup_ratio=ratio))
        optimizer.step()
        schedule.step()


def original_methods():
    source = Path("third_party/ETO/fastchat/train/dpo_trainer.py").read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "DPOMultiTrainer")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"get_batch_logps", "dpo_loss"}]
    ours = ast.parse(Path("baselines/eto/official_loss.py").read_text())
    actual = next(n for n in ours.body if isinstance(n, ast.ClassDef))
    assert [ast.dump(n, include_attributes=False) for n in methods] == [ast.dump(n, include_attributes=False) for n in actual.body]
    cls.body = methods
    module = ast.Module(body=[cls], type_ignores=[])
    from typing import Tuple
    scope = {"torch": torch, "F": torch.nn.functional, "Tuple": Tuple}
    # Compile ONLY the two pinned, local method bodies; no old trainer imports.
    cls.bases, cls.decorator_list = [], []
    exec(compile(ast.fix_missing_locations(module), "official_local_methods", "exec"), scope)
    return scope["DPOMultiTrainer"]


def test_official_loss_verbatim_and_random_formula_equivalence():
    torch.manual_seed(1)
    original = original_methods()
    logits = torch.randn(3, 11, 17, requires_grad=True)
    labels = torch.randint(0, 17, (3, 11))
    labels[:, [0, 2, 3, 7]] = -100
    actual = OfficialLoss.get_batch_logps(logits, labels)
    expected = original.get_batch_logps(logits, labels)
    safe = labels[:, 1:].clamp_min(0)
    formula = logits[:, :-1].log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    formula = (formula * (labels[:, 1:] != -100)).sum(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, formula)
    for beta in (.1, .5):
        settings = SimpleNamespace(beta=beta, label_smoothing=0, loss_type="sigmoid")
        values = [torch.randn(7, requires_grad=True) for _ in range(4)]
        result = OfficialLoss.dpo_loss(settings, *values)
        reference = original.dpo_loss(settings, *values)
        formula = -torch.nn.functional.logsigmoid(beta * ((values[0] - values[1]) - (values[2] - values[3])))
        torch.testing.assert_close(result[0], formula)
        for a, b in zip(result, reference):
            torch.testing.assert_close(a, b)
        assert not result[1].requires_grad and not result[2].requires_grad


def rows():
    return [{"input_ids": [2, 5, 106], "labels": [-100, -100, 106]},
            {"input_ids": [2, 5, 106, 8, 9, 7, 106], "labels": [-100, 5, 106, -100, -100, 7, 106]}]


def test_tiny_objectives_match_dense_logits_and_batch_mean():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = tiny_model()
    model.eval()
    examples = rows()
    for row in examples:
        ids = torch.tensor([row["input_ids"]])
        labels = torch.tensor([row["labels"]])
        dense = model(input_ids=ids, labels=labels, use_cache=False)
        sparse = batch_logps(model, [row], "cpu")
        expected = OfficialLoss.get_batch_logps(dense.logits.float(), labels)
        torch.testing.assert_close(sparse, expected)
        torch.testing.assert_close(sft_objective(model, row, "cpu")[0], dense.loss)
    pair = {"chosen": examples[0], "rejected": examples[1]}
    torch.testing.assert_close(batch_logps(model, examples, "cpu"),
                               torch.cat([batch_logps(model, [row], "cpu") for row in examples]))
    reference = deepcopy(model).eval().requires_grad_(False)
    loss, metrics = dpo_objective(model, reference, pair, "cpu", beta=.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert metrics["rewards/accuracies"] == 0 and metrics["rewards/margins"] == 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None for p in reference.parameters())
    # Simulate FSDP's rank average, b'<world and a non-divisible batch.
    for indices in ([0], [0, 1], [0, 1, 0, 1, 0]):
        for world in (2, 4, 8):
            expected = sum(sft_objective(model, examples[i], "cpu")[0] for i in indices) / len(indices)
            values = []
            for slot in rank_slots(indices, world):
                values.extend(sft_objective(model, examples[i if i is not None else indices[0]], "cpu")[0]
                              * (world / len(indices) if i is not None else 0) for i in slot)
            torch.testing.assert_close(sum(values) / world, expected)
            parameter = model.model.language_model.layers[0].self_attn.q_proj.weight
            expected_grad, = torch.autograd.grad(expected, parameter)
            actual_grad, = torch.autograd.grad(sum(values) / world, parameter)
            torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("objective", ["sft", "dpo"])
def test_tiny_full_parameter_training_export_reload_cpu(tmp_path, objective):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    base = tmp_path / "base"
    model = tiny_model()
    # Include a real non-text tensor, absent in text-only tiny configurations.
    model.model.register_parameter("test_frozen_tower", torch.nn.Parameter(torch.randn(3, 4), requires_grad=False))
    model.save_pretrained(base)
    (base / "preprocessor_config.json").write_text('{"test_processor":true}')
    before = {n: p.clone().detach() for n, p in model.named_parameters()}
    reference = deepcopy(model).eval().requires_grad_(False) if objective == "dpo" else None
    examples = rows() if objective == "sft" else [{"chosen": rows()[0], "rejected": rows()[1]}]
    state = train_loop(model, examples, tmp_path / "train", objective=objective, reference=reference,
                learning_rate=.001, scheduler="cosine" if objective == "sft" else "constant_with_warmup",
                warmup_ratio=.1, original_batch=64 if objective == "sft" else 32, epochs=3)
    assert state["updates"] == len(examples) * 3 and state["examples_seen"] == len(examples) * 3
    assert state["memory_per_rank"][0]["peak_memory_bytes"] is None
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert torch.equal(before["model.test_frozen_tower"], model.model.test_frozen_tower)
    updates = [json.loads(line) for line in (tmp_path / "train/updates.jsonl").read_text().splitlines()]
    assert updates[0]["lr"] == 0 and updates[1]["lr"] > 0
    if objective == "dpo":
        assert all(torch.equal(before[n], p) for n, p in reference.named_parameters())
        assert "rewards/margins" in updates[-1] and "rewards/accuracies" in updates[-1]
    report = export_checkpoint(model, tmp_path / "checkpoint", base, max_shard_bytes=2000)
    assert "model.test_frozen_tower" in report["non_text_tensor_sha256"]
    assert (tmp_path / "checkpoint/preprocessor_config.json").exists()
    student = {"requested_repo": "google/gemma-4-12B-it", "revision": "test-only"}
    provenance = {"config": {"models": {"student": student}}}
    write_json(tmp_path / "provenance.json", provenance)
    write_json(tmp_path / "checkpoint/training_state.json", {**report, "final": True,
                                                           "provenance_sha256": digest(provenance)})
    validate_checkpoint(tmp_path / "checkpoint", student)
    with pytest.raises(ValueError, match="student identity"):
        validate_checkpoint(tmp_path / "checkpoint", {**student, "revision": "different"})
    from safetensors import safe_open
    index = json.loads((tmp_path / "checkpoint/model.safetensors.index.json").read_text())
    name = "model.test_frozen_tower"
    with safe_open(tmp_path / "checkpoint" / index["weight_map"][name], framework="pt") as handle:
        assert torch.equal(handle.get_tensor(name), before[name])
    from train.modeling import load_student
    # Reload bf16 FILE weights into fp32 CPU compute. Casting an entire model
    # with .to(bf16) also rounds nonpersistent RoPE caches, unlike HF loading.
    loaded = load_student(tmp_path / "checkpoint", device="cpu", dtype="float32")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("model.language_model.") or name.startswith("lm_head."):
                parameter.copy_(parameter.to(torch.bfloat16).float())
    model.eval()
    loaded.eval()
    ids = torch.tensor([[2, 5, 106]])
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids).logits, loaded(input_ids=ids).logits)


def test_prepare_model_sdpa_fp32_and_checkpointing(tmp_path):
    tiny_model().save_pretrained(tmp_path / "base")
    policy = prepare_model(tmp_path / "base")
    reference = prepare_model(tmp_path / "base", reference=True)
    assert all(p.dtype == torch.float32 for p in policy.parameters())
    assert policy.config.get_text_config()._attn_implementation == "sdpa"
    assert policy.is_gradient_checkpointing and not reference.training
    assert not any(p.requires_grad for p in reference.parameters())
