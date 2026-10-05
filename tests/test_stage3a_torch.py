"""Numerical tests use tiny random Gemma, local tokenizer, CPU only."""
import ast
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from test_stage2b_torch import tokenizer, tiny_model, trajectory
from baselines.common.conversation import conversation, episode_messages, preference_conversation, sequence_rows
from baselines.common.fsdp_train import (batch_logps, sft_objective, dpo_objective, train_loop,
                                         export_checkpoint, prepare_model, shard_model, validate_checkpoint,
                                         sft_microsteps, scoring_batches)
from baselines.common.schedule import dose, lr_multiplier, rank_slots
from baselines.eto.official_loss import OfficialLoss
from sed.manifest import load_frozen
from sed.react import build_messages
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import digest, write_json
from train.dataset import EMPTY_THOUGHT, episode_examples


def test_every_real_seed_turn_matches_eval_prefix(tokenizer):
    root = Path("data/verified/seeds_k20_s0")
    if not (root / "manifest.json").is_file():
        pytest.skip("No real seed collection installed")
    manifest, freeze = load_frozen("data/manifest/alfworld_manifest.json", "data/manifest/alfworld_split_freeze.json")
    collection, _, evidence = load_verified_collection(root, manifest, freeze)
    episode = checked_episode(evidence, collection["verified"][0])
    example = conversation(episode, tokenizer, source="trusted_teacher_seed", packing="turn")
    history, visible = [], episode["initial"]["visible"]
    for row, step in zip(example["turns"], episode["steps"]):
        span, = row["assistant_spans"]
        prompt = tokenizer.apply_chat_template(build_messages(history, {}, visible), tokenize=False,
                      add_generation_prompt=True, enable_thinking=False)
        assert prompt.endswith(EMPTY_THOUGHT)
        assert row["input_ids"][:span["start"]] == tokenizer.encode(prompt, add_special_tokens=False)
        assert row["labels"][span["stop"] - 1] == 106
        assert tokenizer.decode(row["labels"][span["start"]:span["stop"]]) == step["action_raw"].strip() + "<turn|>"
        history.append({"visible": visible, "reply": step["action_raw"]})
        visible = step["visible"]


@pytest.mark.parametrize("turns", [1, 2, 5])
def test_baseline_turn_rows_equal_dataset_and_every_eval_prefix(tokenizer, turns):
    episode = trajectory()
    episode["steps"] = [dict(episode["steps"][0], event_id=f"event-{i}",
        action_raw=f"Thought: step {i} " + "carefully " * i + "\nAction: look") for i in range(turns)]
    actual = conversation(episode, tokenizer, source="trusted_teacher_seed", packing="turn")
    expected = episode_examples(episode, tokenizer, packing="turn")
    messages = episode_messages(episode)
    for i, (row, target) in enumerate(zip(actual["turns"], expected)):
        for field in ("input_ids", "labels", "attention_mask", "assistant_spans"):
            assert row[field] == target[field]
        prompt = tokenizer.apply_chat_template(messages[:2 + i * 2], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert prompt.endswith(EMPTY_THOUGHT)
        assert row["input_ids"][:row["prompt_tokens"]] == ids
        assert row["labels"][:len(ids)] == [-100] * len(ids)
        assert row["labels"][-1] == 106
    packed = conversation(episode, tokenizer, source="trusted_teacher_seed", packing="episode")
    assert actual["supervised_tokens"] == packed["supervised_tokens"]


def test_dpo_shared_prompt_and_mask_roundtrip(tokenizer):
    chosen = trajectory()
    rejected = deepcopy(chosen)
    rejected["steps"] = rejected["steps"][:1]
    prompt = episode_messages(chosen)[:2]
    pair = {"task_id": chosen["task_id"], "prompt": prompt,
            "chosen": episode_messages(chosen)[2:], "rejected": episode_messages(rejected)[2:]}
    rows = preference_conversation(pair, tokenizer)
    for side, episode in [("chosen", chosen), ("rejected", rejected)]:
        expected = conversation(episode, tokenizer, source="student_exploration", packing="turn")
        actual = rows[side]
        for turn, target in zip(actual["turns"], expected["turns"]):
            assert turn["input_ids"] == target["input_ids"] and turn["labels"] == target["labels"]
            assert rows["prompt_input_ids"] + turn["continuation_input_ids"] == turn["input_ids"]
            assert turn["labels"][-1] == 106
            assert "PRIVATE_PDDL" not in tokenizer.decode(turn["input_ids"])
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


def test_prepare_and_shard_policy_fp32_reference_bf16(tmp_path, monkeypatch):
    tiny_model().save_pretrained(tmp_path / "base")
    policy = prepare_model(tmp_path / "base")
    reference = prepare_model(tmp_path / "base", reference=True)
    assert all(p.dtype == torch.float32 for p in policy.parameters())
    assert all(p.dtype == torch.bfloat16 for p in reference.parameters())
    assert policy.config.get_text_config()._attn_implementation == "sdpa"
    assert reference.config.get_text_config()._attn_implementation == "sdpa"
    assert policy.is_gradient_checkpointing and not reference.is_gradient_checkpointing
    assert not reference.training and not reference.config.get_text_config().use_cache
    assert not any(p.requires_grad for p in reference.parameters())
    assert sum(p.numel() * p.element_size() for p in policy.parameters()) == 2 * sum(
        p.numel() * p.element_size() for p in reference.parameters())

    # Inspect the production wrap/precision choices without a process group or GPU.
    import torch.distributed.fsdp as fsdp
    wraps = []
    monkeypatch.setattr(fsdp, "fully_shard", lambda module, **kwargs: wraps.append((module, kwargs)))
    mesh = object()
    for candidate, is_reference in ((policy, False), (reference, True)):
        wraps.clear()
        assert shard_model(candidate, mesh=mesh, reference=is_reference) is candidate
        assert [module for module, _ in wraps] == [*candidate.model.language_model.layers, candidate]
        for _, kwargs in wraps:
            assert kwargs["mesh"] is mesh and kwargs["reshard_after_forward"] is True
            assert kwargs["mp_policy"].param_dtype == torch.bfloat16
            assert kwargs["mp_policy"].reduce_dtype == torch.float32
    with pytest.raises(ValueError, match="policy requires torch.float32"):
        shard_model(reference, mesh=mesh)
    with pytest.raises(ValueError, match="reference requires torch.bfloat16"):
        shard_model(policy, mesh=mesh, reference=True)
    reference.train()
    with pytest.raises(ValueError, match="frozen and in eval mode"):
        shard_model(reference, mesh=mesh, reference=True)
    reference.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen and in eval mode"):
        shard_model(reference, mesh=mesh, reference=True)


@pytest.mark.parametrize("seed", [0, 17])
def test_reference_bf16_storage_has_bitwise_equal_logps_to_fp32_compute_cast(tmp_path, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    tiny_model().save_pretrained(tmp_path / "base")
    # Reconstruct the former fp32-stored reference with the same loader/SDPA.
    fp32 = prepare_model(tmp_path / "base").eval().requires_grad_(False)
    fp32.gradient_checkpointing_disable()
    bf16 = prepare_model(tmp_path / "base", reference=True)

    class Bf16Compute(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, **kwargs):
            # FSDP2 casts parameters to param_dtype before forward; buffers
            # (including fp32 RoPE caches) retain their original dtype/values.
            parameters = {name: p.to(torch.bfloat16) for name, p in self.model.named_parameters()}
            return torch.func.functional_call(self.model, parameters, (), kwargs)

    for name, value in fp32.named_buffers():
        # RoPE caches must stay fp32: a blanket model.to(bf16) rounds them.
        # Other loader buffers (e.g. embed_scale) can differ in storage dtype;
        # keep them native and test the resulting log-probs end to end below.
        if "rotary_emb" in name:
            other = dict(bf16.named_buffers())[name]
            assert value.dtype == other.dtype == torch.float32 and torch.equal(value, other)
    examples = [{"packing": "turn", "turns": rows()},
                {"packing": "turn", "turns": [rows()[0]]}]
    with torch.no_grad():
        expected = batch_logps(Bf16Compute(fp32).eval(), examples, "cpu", rounds=3)
        actual = batch_logps(Bf16Compute(bf16).eval(), examples, "cpu", rounds=3)
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert actual.grad_fn is None and not actual.requires_grad
    assert all(p.dtype == torch.float32 for p in fp32.parameters())
    assert all(p.dtype == torch.bfloat16 for p in bf16.parameters())


@pytest.mark.parametrize("nll_weight", [0.0, 1.0], ids=["dpo", "ipr"])
def test_reference_activations_released_before_policy_forward(tmp_path, monkeypatch, nll_weight):
    torch.set_num_threads(1)
    tiny_model().save_pretrained(tmp_path / "base")
    policy = prepare_model(tmp_path / "base")
    reference = prepare_model(tmp_path / "base", reference=True)
    tensors, scores = [], []

    def record_activation(module, args, output):
        assert not torch.is_grad_enabled()
        tensors.append(weakref.ref(output))

    def record_logits(module, args, output):
        assert output.logits.grad_fn is None and not output.logits.requires_grad
        tensors.append(weakref.ref(output.logits))

    def check_released(module, args):
        assert all(tensor() is None for tensor in tensors)

    def score(model, *args, **kwargs):
        result = batch_logps(model, *args, **kwargs)
        if model is reference:
            assert result.shape == (2,) and result.grad_fn is None and not result.requires_grad
            scores.append(result)
        else:
            assert scores and tensors
        return result

    monkeypatch.setattr("baselines.common.fsdp_train.batch_logps", score)
    handles = [reference.model.language_model.layers[0].mlp.register_forward_hook(record_activation),
               reference.register_forward_hook(record_logits),
               reference.register_forward_pre_hook(check_released),
               policy.register_forward_pre_hook(check_released)]
    pair = {"chosen": {"packing": "turn", "turns": rows()},
            "rejected": {"packing": "turn", "turns": [rows()[0]]}}
    try:
        loss, metrics = dpo_objective(policy, reference, pair, "cpu", beta=.1,
                                      nll_weight=nll_weight, rounds=3)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert len(tensors) == 6  # An activation and logits from each scoring round.
    assert all(tensor() is None for tensor in tensors)
    assert all(not value.requires_grad for value in metrics.values())
    assert all(p.grad is None for p in reference.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())


def packed_examples(packing, n=25):
    examples = []
    for i in range(n):
        count = 1 + i % 3
        if packing == "episode":
            examples.append({"packing": packing, "input_ids": [2] + [5, 6, 106] * count,
                             "labels": [-100] + [-100, 6, 106] * count})
        else:
            examples.append({"packing": packing, "turns": [
                {"input_ids": [2] + [5, 6, 106] * t + [5, 6, 106],
                 "labels": [-100] * (2 + 3 * t) + [6, 106]} for t in range(count)]})
    return examples


def test_training_updates_keep_trajectory_batches_lrs_and_exposure_across_packings(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(2)
    base = tiny_model()
    states, ledgers = {}, {}
    for packing in ("episode", "turn"):
        examples = packed_examples(packing)
        states[packing] = train_loop(deepcopy(base), examples, tmp_path / packing, objective="sft",
            learning_rate=2e-4, scheduler="linear", warmup_ratio=0, original_batch=8,
            epochs=2, seed=7, packing=packing)
        ledgers[packing] = [json.loads(line) for line in (tmp_path / packing / "updates.jsonl").read_text().splitlines()]
        expected_forward = 2 * sum(len(row["input_ids"]) for e in examples for row in sequence_rows(e))
        assert states[packing]["logical_forward_tokens"] == expected_forward
        assert states[packing]["physical_policy_forward_tokens"] == expected_forward
        assert states[packing]["zero_weight_slots"] == 0
        assert ledgers[packing][-1]["logical_forward_tokens"] == expected_forward
    for key in ("updates", "examples_seen", "supervised_tokens_seen", "b_prime", "steps_per_epoch", "last_batch"):
        assert states["turn"][key] == states["episode"][key]
    assert states["turn"]["b_prime"] == 3 and states["turn"]["last_batch"] == 1
    assert states["turn"]["logical_forward_tokens"] > states["episode"]["logical_forward_tokens"]
    for turn, episode in zip(ledgers["turn"], ledgers["episode"]):
        for key in ("update", "epoch", "indices", "lr", "global_batch", "examples_seen", "supervised_tokens_seen"):
            assert turn[key] == episode[key]


@pytest.mark.parametrize("indices", [[0], [0, 1], [2, 0, 1]])
def test_turn_accumulation_gradient_is_trajectory_token_mean_with_rank_padding(indices):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = tiny_model().eval()
    examples = [{"packing": "turn", "turns": [rows()[0]]},
                {"packing": "turn", "turns": rows()},
                {"packing": "turn", "turns": [rows()[1], rows()[1], rows()[0]]}]
    expected = sum(sft_objective(model, examples[i], "cpu")[0] for i in indices) / len(indices)
    microsteps = sft_microsteps(examples, indices)
    actual = 0
    for slot in rank_slots(list(range(len(microsteps))), 4):
        for index in slot:
            row, weight = microsteps[index if index is not None else 0]
            actual += sft_objective(model, row, "cpu")[0] * (weight * 4 / len(indices) if index is not None else 0)
    actual /= 4
    torch.testing.assert_close(actual, expected)
    parameter = model.model.language_model.layers[0].self_attn.q_proj.weight
    torch.testing.assert_close(torch.autograd.grad(actual, parameter)[0], torch.autograd.grad(expected, parameter)[0])


@pytest.mark.parametrize("nll_weight", [0., 1.])
def test_preference_turn_scores_and_gradients_are_sums_not_turn_means(nll_weight):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = tiny_model().eval()
    reference = deepcopy(model).eval().requires_grad_(False)
    with torch.no_grad():
        next(reference.parameters()).add_(.01)
    pair = {"chosen": {"packing": "turn", "turns": rows()},
            "rejected": {"packing": "turn", "turns": [rows()[1]]}}
    def dense_scores(candidate):
        result = []
        for side in ("chosen", "rejected"):
            score = 0
            for row in pair[side]["turns"]:
                logits = candidate(input_ids=torch.tensor([row["input_ids"]]), use_cache=False).logits
                score += OfficialLoss.get_batch_logps(logits.float(), torch.tensor([row["labels"]]))[0]
            result.append(score)
        return torch.stack(result)
    policy, ref = dense_scores(model), dense_scores(reference)
    torch.testing.assert_close(batch_logps(model, [pair["chosen"], pair["rejected"]], "cpu", rounds=4), policy)
    expected = -torch.nn.functional.logsigmoid(.1 * ((policy[0] - policy[1]) - (ref[0] - ref[1]))) - nll_weight * policy[0]
    actual, _ = dpo_objective(model, reference, pair, "cpu", beta=.1, nll_weight=nll_weight, rounds=4)
    torch.testing.assert_close(actual, expected)
    parameter = model.model.language_model.layers[0].self_attn.q_proj.weight
    torch.testing.assert_close(torch.autograd.grad(actual, parameter)[0], torch.autograd.grad(expected, parameter)[0])
    assert all(p.grad is None for p in reference.parameters())


@pytest.mark.parametrize("objective", ["dpo", "ipr"])
def test_preference_turn_training_accounts_for_padding_and_reference(tmp_path, objective):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = tiny_model()
    reference = deepcopy(model).eval().requires_grad_(False)
    pair = {"chosen": {"packing": "turn", "turns": rows()},
            "rejected": {"packing": "turn", "turns": [rows()[1]]}}
    forwards = []
    handle = model.register_forward_pre_hook(lambda module, args, kwargs:
        forwards.append((torch.is_grad_enabled(), kwargs["input_ids"].numel())), with_kwargs=True)
    try:
        state = train_loop(model, [pair], tmp_path, objective=objective, reference=reference,
            learning_rate=1e-5, scheduler="constant_with_warmup", warmup_ratio=0,
            original_batch=48, epochs=1, packing="turn")
    finally:
        handle.remove()
    assert state["updates"] == state["examples_seen"] == 1
    assert state["supervised_tokens_seen"] == 9
    assert state["logical_forward_tokens"] == 17
    assert state["two_pass_turn_backward"] is True
    assert state["physical_reference_forward_tokens"] == 28
    assert state["physical_policy_recompute_forward_tokens"] == 24
    assert state["physical_policy_forward_tokens"] == sum(n for _, n in forwards) == 52
    assert [grad for grad, _ in forwards] == [False] * 2 + [True] * 4
    ledger, = [json.loads(line) for line in (tmp_path / "updates.jsonl").read_text().splitlines()]
    assert ledger["collective_rounds"] == 2 and ledger["global_batch"] == 1
    assert ledger["physical_policy_forward_tokens"] == 52
    assert ledger["physical_policy_recompute_forward_tokens"] == 24
    assert ledger["two_pass_turn_backward"] is True
    with pytest.raises(ValueError, match="cannot omit"):
        scoring_batches([pair["chosen"], pair["rejected"]], rounds=1)


@pytest.mark.parametrize("nll_weight", [0.0, 1.0], ids=["dpo", "ipr"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_two_pass_matches_all_one_graph_parameter_gradients_and_metrics(nll_weight, checkpointing):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    original = tiny_model().train()
    if checkpointing:
        original.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    streamed = deepcopy(original)
    reference = deepcopy(original).eval().requires_grad_(False)
    with torch.no_grad():
        next(reference.parameters()).add_(.03)
    pair = {"chosen": {"packing": "turn", "turns": rows()},
            "rejected": {"packing": "turn", "turns": [rows()[1], rows()[0], rows()[1]]}}
    kwargs = {"beta": .5, "nll_weight": nll_weight, "rounds": 5}
    expected_loss, expected_metrics = dpo_objective(original, reference, pair, "cpu", **kwargs)
    scale = .37  # The global batch/rank factor applies to DPO AND chosen NLL.
    (expected_loss * scale).backward()
    loss, metrics = dpo_objective(streamed, reference, pair, "cpu", backward_scale=scale, **kwargs)
    assert not loss.requires_grad and loss.grad_fn is None
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    assert metrics.keys() == expected_metrics.keys()
    for key in metrics:
        assert not metrics[key].requires_grad
        torch.testing.assert_close(metrics[key], expected_metrics[key], rtol=0, atol=0)
    for (name, expected), (actual_name, actual) in zip(original.named_parameters(), streamed.named_parameters()):
        assert name == actual_name
        if expected.grad is None:
            assert actual.grad is None
        else:
            # Splitting a padded pair into single turns changes fp32 GEMM
            # reduction order (including Gemma's internal fp32 operations).
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-6, atol=3e-6,
                                       msg=lambda detail: f"{name}: {detail}")
    assert all(p.grad is None for p in reference.parameters())


@pytest.mark.parametrize("nll_weight", [0.0, 1.0], ids=["dpo", "ipr"])
@pytest.mark.parametrize("indices", [[0], [0, 1, 0]])
def test_two_pass_rank_and_turn_padding_preserve_global_pair_mean(nll_weight, indices):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = tiny_model().eval()
    reference = deepcopy(base).requires_grad_(False)
    pairs = [{"chosen": {"packing": "turn", "turns": rows()[:count]},
              "rejected": {"packing": "turn", "turns": [rows()[1]] * (3 - count)}} for count in (1, 2)]
    expected = sum(dpo_objective(base, reference, pairs[i], "cpu", beta=.1, nll_weight=nll_weight)[0]
                   for i in indices) / len(indices)
    expected.backward()
    rank_models = [deepcopy(base) for _ in range(2)]
    counts = []
    for rank, model in enumerate(rank_models):
        model.zero_grad(set_to_none=True)
        forwards, backwards = [], []
        handles = [model.register_forward_pre_hook(lambda module, args:
                       forwards.append(torch.is_grad_enabled())),
                   next(model.parameters()).register_hook(lambda grad: backwards.append(True))]
        try:
            for slot in rank_slots(indices, len(rank_models)):
                index = slot[rank]
                rounds = max(len(pairs[i if i is not None else 0][side]["turns"])
                             for i in slot for side in ("chosen", "rejected"))
                dpo_objective(model, reference, pairs[index if index is not None else 0], "cpu",
                    beta=.1, nll_weight=nll_weight, rounds=rounds,
                    backward_scale=len(rank_models) / len(indices) if index is not None else 0.0)
        finally:
            for handle in handles:
                handle.remove()
        counts.append((forwards, len(backwards)))
        if all(slot[rank] is None for slot in rank_slots(indices, len(rank_models))):
            assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())
    assert counts[0] == counts[1]
    assert counts[0][1] == sum(counts[0][0])  # Dummy forwards still backward for every rank.
    for name, parameter in base.named_parameters():
        grads = [dict(model.named_parameters())[name].grad for model in rank_models]
        if parameter.grad is None:
            assert all(grad is None for grad in grads)
        else:
            torch.testing.assert_close(sum(grads) / len(rank_models), parameter.grad, rtol=2e-6, atol=3e-6,
                                       msg=lambda detail: f"{name}: {detail}")


@pytest.mark.parametrize("nll_weight", [0.0, 1.0], ids=["dpo", "ipr"])
def test_two_pass_chain_rule_on_float64_tiny_causal_model(nll_weight):
    class TinyCausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 8)
            self.output = torch.nn.Linear(8, 128)

        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
            hidden = self.embedding(input_ids).cumsum(dim=1).tanh()
            return SimpleNamespace(logits=self.output(hidden[:, logits_to_keep]))

    torch.manual_seed(44)
    original = TinyCausalModel().double()
    streamed = deepcopy(original)
    reference = deepcopy(original).requires_grad_(False)
    with torch.no_grad():
        reference.output.weight.add_(.02 * torch.randn_like(reference.output.weight))
    pair = {"chosen": {"packing": "turn", "turns": rows()},
            "rejected": {"packing": "turn", "turns": [rows()[1], rows()[1], rows()[0]]}}
    loss, _ = dpo_objective(original, reference, pair, "cpu", beta=.5, nll_weight=nll_weight, rounds=4)
    (loss * .5).backward()
    dpo_objective(streamed, reference, pair, "cpu", beta=.5, nll_weight=nll_weight, rounds=4,
                  backward_scale=.5)
    for actual, expected in zip(streamed.parameters(), original.parameters()):
        # The production log-softmax/scorer intentionally remains fp32.
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("nll_weight", [0.0, 1.0], ids=["dpo", "ipr"])
def test_two_pass_retains_only_one_turn_graph_even_for_forty_turns(nll_weight):
    torch.set_num_threads(1)
    torch.manual_seed(43)
    model = tiny_model().eval()
    reference = deepcopy(model).requires_grad_(False)

    def retained_graph_peak(turns, two_pass):
        retained = {"live": 0, "peak": 0}

        class SavedTensor:
            def __init__(self, tensor):
                self.tensor = tensor.detach()
                self.size = tensor.numel() * tensor.element_size()
                retained["live"] += self.size
                retained["peak"] = max(retained["peak"], retained["live"])

            def __del__(self):
                retained["live"] -= self.size

        def before_forward(module, args, kwargs):
            if two_pass:
                # The previous backward must release EVERY saved tensor
                # before allocating the next turn's graph, on either side.
                assert retained["live"] == 0
                if torch.is_grad_enabled():
                    assert kwargs["input_ids"].shape[0] == 1

        pair = {side: {"packing": "turn", "turns": [row] * turns}
                for side, row in zip(("chosen", "rejected"), rows())}
        model.zero_grad(set_to_none=True)
        handle = model.register_forward_pre_hook(before_forward, with_kwargs=True)
        try:
            with torch.autograd.graph.saved_tensors_hooks(SavedTensor, lambda saved: saved.tensor):
                loss, _ = dpo_objective(model, reference, pair, "cpu", beta=.1, nll_weight=nll_weight,
                                        backward_scale=1.0 if two_pass else None)
                if not two_pass:
                    loss.backward()
        finally:
            handle.remove()
        assert retained["live"] == 0
        return retained["peak"]

    single_turn = retained_graph_peak(1, True)
    assert single_turn > 0
    assert retained_graph_peak(40, True) == single_turn
    # Positive control: the old objective really retains all turn graphs.
    assert retained_graph_peak(40, False) > 20 * single_turn
