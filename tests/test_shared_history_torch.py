"""CPU fp32 acceptance: real Gemma4 decoder, both attention types, no downloads."""
from copy import deepcopy
import json
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from test_stage2b_torch import tiny_model, tokenizer, trajectory
from baselines.common.conversation import conversation, sequence_rows
from baselines.common.fsdp_train import batch_logps, dpo_objective, train_loop
from baselines.common.schedule import rank_slots
from baselines.common.shared_history import (pack_shared_history, attention_masks,
    shared_history_logps, _block_metadata, BLOCK_SIZE, configure_flex_attention, flex_attention_settings)


def random_conversation(seed, turns=5, *, prefix=13, empty=False):
    rng = random.Random(seed)
    history, rows = [2] + [rng.randrange(10, 100) for _ in range(prefix)], []
    for t in range(turns):
        native = history + [7] + [rng.randrange(10, 100) for _ in range(rng.randrange(2, 10))] + [8]
        reply = [rng.randrange(10, 100) for _ in range(rng.randrange(1, 9))] + [106]
        prompt = native + [3, 4, 5]  # Three-token EMPTY_THOUGHT stand-in.
        ids = prompt + reply
        row = {"packing": "turn", "input_ids": ids,
               "labels": [-100] * len(prompt) + reply,
               "generation_start": len(native), "prompt_tokens": len(prompt)}
        if empty and t % 3 == 1:
            row = {"packing": "turn", "input_ids": native + [106],
                   "labels": [-100] * (len(native) + 1)}
            reply = [106]
        rows.append(row)
        history = native + reply + [9]
    return {"packing": "turn", "turns": rows}


def model(seed=42, backend="sdpa", checkpointing=False):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    value = tiny_model()
    value.config.get_text_config().sliding_window = 7
    for layer in value.model.language_model.layers:
        if layer.self_attn.is_sliding:
            layer.self_attn.sliding_window = 7
    value.set_attn_implementation(backend)
    if checkpointing:
        value.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return value


def assert_gradients(actual, expected):
    for name, wanted in expected.named_parameters():
        got = dict(actual.named_parameters())[name]
        if wanted.grad is None:
            assert got.grad is None, name
        else:
            assert got.grad is not None, name
            error = (got.grad - wanted.grad).norm()
            assert error <= 1e-5 * wanted.grad.norm() + 1e-10, (name, error, wanted.grad.norm())
            # Near-zero elements suffer cancellation when history gradients
            # accumulate through one node instead of many duplicated nodes.
            # Require BOTH relative L2 and relative max error per parameter.
            assert (got.grad - wanted.grad).abs().max() <= 1e-5 * wanted.grad.abs().max() + 1e-10, name


@pytest.mark.parametrize("seed", [1, 2, 17])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_each_turn_logps_match_with_both_layer_types_and_padding(seed, backend):
    m = model(backend=backend).eval()
    rows = [random_conversation(seed, empty=True), random_conversation(seed + 1, 2, prefix=29)]
    with torch.no_grad():
        actual = shared_history_logps(m, rows, "cpu", per_turn=True)
        for row, got in zip(rows, actual):
            expected = torch.cat([batch_logps(m, [turn], "cpu") for turn in sequence_rows(row)])
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=0)


def test_real_tokenizer_native_history_and_generation_paths(tokenizer):
    episode = trajectory()
    episode["steps"] = [dict(episode["steps"][0], action_raw=reply, event_id=str(i))
                        for i, reply in enumerate(["Action: look", "", "Action: go north", " "])]
    row = conversation(episode, tokenizer, source="student_exploration", packing="turn")
    packed = pack_shared_history(row)
    for branch, turn in enumerate(sequence_rows(row), 1):
        indices = [i for i, b in enumerate(packed["branches"]) if b == branch]
        path = [i for i in range(indices[0]) if packed["branches"][i] == 0] + indices
        assert [packed["input_ids"][i] for i in path] == turn["input_ids"]
        assert [packed["position_ids"][i] for i in path] == list(range(len(path)))
    assert sum(v != -100 for v in packed["labels"]) == row["supervised_tokens"]


@pytest.mark.parametrize("weight", [0., 1.])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_dpo_ipr_loss_metrics_and_every_gradient_match_two_pass(weight, checkpointing):
    original = model(checkpointing=checkpointing).train()
    packed = deepcopy(original)
    reference = model(57).eval().requires_grad_(False)
    pair = {"chosen": random_conversation(1, 3), "rejected": random_conversation(2, 5, empty=True)}
    opts = dict(beta=.5, nll_weight=weight, backward_scale=.37, rounds=7)
    expected, expected_metrics = dpo_objective(original, reference, pair, "cpu", **opts)
    actual, actual_metrics = dpo_objective(packed, reference, pair, "cpu", scoring="shared_history", **opts)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    for key in expected_metrics:
        torch.testing.assert_close(actual_metrics[key], expected_metrics[key], rtol=1e-5, atol=2e-5)
    assert_gradients(packed, original)
    assert all(p.grad is None for p in reference.parameters())


@pytest.mark.parametrize("weight", [0., 1.])
@pytest.mark.parametrize("indices", [[0], [0, 1, 0]])
def test_rank_padding_and_global_pair_normalization(weight, indices):
    original = model().eval()
    reference = model(57).eval().requires_grad_(False)
    pairs = [{"chosen": random_conversation(4, t), "rejected": random_conversation(5, 4-t)} for t in (1, 2)]
    for i in indices:
        dpo_objective(original, reference, pairs[i], "cpu", beta=.1, nll_weight=weight,
                      rounds=6, backward_scale=1 / len(indices))
    ranks = [model().eval() for _ in range(2)]
    counts = []
    for rank, m in enumerate(ranks):
        calls, backs = [], []
        handles = [m.register_forward_pre_hook(lambda *args: calls.append(torch.is_grad_enabled())),
                   next(m.parameters()).register_hook(lambda grad: backs.append(1))]
        for slot in rank_slots(indices, 2):
            i = slot[rank]
            dpo_objective(m, reference, pairs[i if i is not None else 0], "cpu", beta=.1, nll_weight=weight,
                          backward_scale=2 / len(indices) if i is not None else 0., scoring="shared_history")
        for handle in handles:
            handle.remove()
        counts.append((calls, backs))
        if indices == [0] and rank == 1:
            assert all(p.grad is None or not p.grad.count_nonzero() for p in m.parameters())
    assert counts[0] == counts[1]
    averaged = deepcopy(ranks[0])
    for name, param in averaged.named_parameters():
        grads = [dict(m.named_parameters())[name].grad for m in ranks]
        param.grad = None if grads[0] is None else sum(grads) / 2
    assert_gradients(averaged, original)


@pytest.mark.parametrize("objective", ["dpo", "ipr"])
def test_training_counts_actual_forwards_and_preserves_optimizer_inputs(tmp_path, objective):
    pair = {"chosen": random_conversation(1, 2), "rejected": random_conversation(2, 4)}
    original, packed = model(), model()
    reference = model(57).eval().requires_grad_(False)
    opts = dict(objective=objective, reference=reference, learning_rate=1e-5,
                scheduler="constant_with_warmup", warmup_ratio=0, original_batch=48, epochs=1, packing="turn")
    old = train_loop(original, [pair], tmp_path / "old", **opts)
    forwards, ref_forwards = [], []
    handles = [packed.register_forward_pre_hook(lambda m, args, kw: forwards.append(
                   (torch.is_grad_enabled(), kw["input_ids"].numel())), with_kwargs=True),
               reference.register_forward_pre_hook(lambda m, args, kw: ref_forwards.append(
                   kw["input_ids"].numel()), with_kwargs=True)]
    kernel_options = {"BLOCK_M": 16, "BLOCK_N": 16}
    new = train_loop(packed, [pair], tmp_path / "new", scoring="shared_history",
                     flex_kernel_options=kernel_options, **opts)
    for handle in handles:
        handle.remove()
    assert new["physical_policy_forward_tokens"] == sum(n for _, n in forwards)
    assert new["physical_policy_recompute_forward_tokens"] == sum(n for grad, n in forwards if grad)
    assert new["physical_reference_forward_tokens"] == sum(ref_forwards)
    assert new["physical_policy_forward_tokens"] < old["physical_policy_forward_tokens"]
    for key in ("updates", "examples_seen", "supervised_tokens_seen", "logical_forward_tokens"):
        assert new[key] == old[key]
    assert new["scoring"] == "shared_history" and new["two_pass_side_backward"]
    ledger = json.loads((tmp_path / "new/updates.jsonl").read_text())
    assert ledger["scoring"] == new["scoring"]
    assert ledger["flex_attention"] == new["flex_attention"] == flex_attention_settings(kernel_options)
    assert reference.config.get_text_config().shared_history_flex_kernel_options == new["flex_attention"]["kernel_options"]
    assert ledger["physical_policy_forward_tokens"] == new["physical_policy_forward_tokens"]
    assert_gradients(packed, original)
    for name, param in original.named_parameters():
        torch.testing.assert_close(dict(packed.named_parameters())[name], param, rtol=1e-5, atol=1e-7)


def test_block_metadata_matches_dense_predicate_and_long_masks_stay_small():
    m = model()
    rows = [pack_shared_history(random_conversation(7, 20)), pack_shared_history(random_conversation(9, 3))]
    _, masks = attention_masks(rows, m.config.get_text_config(), "cpu", backend="flex_attention")
    _, dense = attention_masks(rows, m.config.get_text_config(), "cpu", backend="sdpa")
    for kind, block in masks.items():
        length = dense[kind].shape[-1]
        partial, full = _block_metadata(rows, length, 7 if kind == "sliding_attention" else None)
        for b in range(len(rows)):
            for q in range(partial.shape[-2]):
                for k in range(partial.shape[-1]):
                    tile = dense[kind][b, 0, q*BLOCK_SIZE:(q+1)*BLOCK_SIZE, k*BLOCK_SIZE:(k+1)*BLOCK_SIZE]
                    if tile.any():
                        assert partial[b, 0, q, k] or full[b, 0, q, k]
                    if full[b, 0, q, k]:
                        assert tile.all()
        assert block.mask_mod is not None
    long = pack_shared_history(random_conversation(7, 20, prefix=30000))
    _, masks = attention_masks([long], m.config.get_text_config(), "cpu", backend="flex_attention")
    tensors = [v for mask in masks.values() for v in vars(mask).values() if isinstance(v, torch.Tensor)]
    assert sum(t.numel() * t.element_size() for t in tensors) < 8_000_000
    assert all(not t.is_floating_point() for t in tensors)
    with pytest.raises(ValueError, match="CPU-test-only"):
        attention_masks([long], m.config.get_text_config(), "cpu", backend="sdpa")


def test_bad_prefix_and_dropout_fail_closed():
    row = random_conversation(7)
    row["turns"][1]["input_ids"][0] += 1
    with pytest.raises(ValueError, match="consistent extension"):
        pack_shared_history(row)
    m = model()
    m.config.get_text_config().attention_dropout = .1
    with pytest.raises(ValueError, match="zero dropout"):
        shared_history_logps(m, [random_conversation(7)], "cpu")


def test_flex_interface_forward_matches_per_turn(monkeypatch):
    # CPU math implementation exercises HF's Flex interface and BlockMask.
    # GPU Triton compilation is deliberately outside this CPU-only acceptance.
    from torch.nn.attention.flex_attention import flex_attention
    monkeypatch.setattr("transformers.integrations.flex_attention.compile_friendly_flex_attention",
        lambda q, k, v, training=False, **kwargs: flex_attention(q, k, v, **kwargs))
    original = model().eval()
    packed = model(backend="flex_attention").eval()
    rows = [random_conversation(1, 3), random_conversation(2, 20)]
    # Torch 2.11/2.13 explicitly refuse CPU Flex backward. All-gradient
    # acceptance above uses the same predicate with CPU SDPA/eager instead.
    with torch.no_grad():
        expected = batch_logps(original, rows, "cpu")
        actual = shared_history_logps(packed, rows, "cpu")
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("options", [None, {"BLOCK_M": 16, "BLOCK_N": 16, "bwd_BLOCK_M1": 32, "bwd_BLOCK_N1": 32}])
def test_large_head_kernel_options_reach_flex_and_checkpoint_recomputation(monkeypatch, checkpointing, options):
    from transformers import Gemma4UnifiedTextConfig, Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
    from torch.nn.attention.flex_attention import BlockMask
    torch.set_num_threads(1)
    torch.manual_seed(42)
    text = Gemma4UnifiedTextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=256, global_head_dim=512, max_position_embeddings=128,
        layer_types=["sliding_attention", "full_attention"], sliding_window=7)
    config = Gemma4UnifiedConfig(text_config=text, vision_config=None, audio_config=None,
        image_token_id=120, audio_token_id=121, video_token_id=122)
    config._attn_implementation = "sdpa"
    original = Gemma4UnifiedForConditionalGeneration(config).train()
    packed = deepcopy(original)
    packed.set_attn_implementation("flex_attention")
    if checkpointing:
        packed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    settings = configure_flex_attention(packed, options)
    configure_flex_attention(packed)  # Repeated scoring must not stack hooks.
    assert all(len(layer.self_attn._forward_pre_hooks) == 1 for layer in packed.model.language_model.layers)
    expected_options = settings["kernel_options"]
    assert expected_options["BLOCK_M"] == expected_options["BLOCK_N"] == (16 if options else 32)
    assert all(expected_options[key] == (32 if not options or key.endswith("1") else 16)
               for key in ("bwd_BLOCK_M1", "bwd_BLOCK_N1", "bwd_BLOCK_M2", "bwd_BLOCK_N2"))
    calls = []

    def flex_boundary(q, k, v, *, block_mask, kernel_options, training=False, **kwargs):
        assert isinstance(block_mask, BlockMask) and block_mask.BLOCK_SIZE == (128, 128)
        calls.append((q.shape[-1], deepcopy(kernel_options)))
        if q.shape[-1] > 256:
            assert kernel_options == expected_options
            assert all(128 % value == 0 for key, value in kernel_options.items() if "BLOCK_" in key)
        else:
            assert kernel_options is None
        # Real HF layers and kwargs; only the compiled Flex boundary is mocked.
        b = torch.arange(q.shape[0])[:, None, None, None]
        qi = torch.arange(q.shape[-2])[None, None, :, None]
        ki = torch.arange(k.shape[-2])[None, None, None, :]
        mask = block_mask.mask_mod(b, 0, qi, ki)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            scale=kwargs["scale"], enable_gqa=kwargs["enable_gqa"])

    monkeypatch.setattr("transformers.integrations.flex_attention.compile_friendly_flex_attention", flex_boundary)
    rows = [random_conversation(1, 2)]
    expected = batch_logps(original, rows, "cpu")
    actual = shared_history_logps(packed, rows, "cpu")
    assert calls == [(256, None), (512, expected_options)]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actual.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in packed.parameters())
    if checkpointing:
        assert len(calls) == 4 and {head for head, _ in calls[2:]} == {256, 512}
    else:
        assert len(calls) == 2


@pytest.mark.parametrize("options", [[], {"BLOCK_M": 256}, {"BLOCK_N": 48}, {"BLOCK_M": 8},
    {"BLOCK_M": True}, {"num_stages": 0}, {"num_warps": 3}, {"BLOCKS_ARE_CONTIGUOUS": True},
    {"bwd_BLOCK_M1": 64, "bwd_BLOCK_N1": 32}, {"bwd_BLOCK_M2": 32, "bwd_BLOCK_N2": 64}])
def test_invalid_flex_options_fail_before_kernel_launch(options):
    with pytest.raises(ValueError, match="Flex"):
        flex_attention_settings(options)


def test_rank_padding_physical_accounting(tmp_path, monkeypatch):
    # The same known pair is executed on one real and one zero-weight rank.
    # Fake only collectives; execute the real objective/backward/accounting.
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda values: None)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda out, value: out.__setitem__(slice(None), [value] * 2))
    pair = {"chosen": random_conversation(1, 2), "rejected": random_conversation(2, 3)}
    counts, states = [], []
    for rank in range(2):
        m, ref = model(), model(57).eval().requires_grad_(False)
        inputs = []
        handle = m.register_forward_pre_hook(lambda m, args, kw: inputs.append(kw["input_ids"].numel()), with_kwargs=True)
        states.append(train_loop(m, [pair], tmp_path / str(rank), objective="ipr", reference=ref,
            learning_rate=1e-5, scheduler="constant_with_warmup", warmup_ratio=0, original_batch=48,
            epochs=1, rank=rank, world_size=2, scoring="shared_history"))
        handle.remove()
        counts.append(sum(inputs))
        if rank == 1:
            assert all(p.grad is None or not p.grad.count_nonzero() for p in m.parameters())
    for state in states:
        assert state["zero_weight_slots"] == 1
        assert state["examples_seen"] == 1
        assert state["physical_policy_forward_tokens"] == sum(counts)
        assert state["physical_reference_forward_tokens"] == sum(counts) // 2
