"""Offline E-series compatibility and c6b15b2 reference-family regression gates."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from baselines.common.conversation import conversation, sequence_rows
from baselines.common.fsdp_train import batch_logps, prepare_model, shard_model, text_parameter
from baselines.common.shared_history import (configure_flex_attention, pack_shared_history,
                                             shared_history_logps)
from harness.lora_adapters import (convert_adapter, derive_gemma4_value_lora,
                                   validate_served_adapter)
from sed.storage import canonical_bytes, file_hash
from train import dataset, modeling
from test_stage2b_torch import trajectory
from test_shared_history_torch import random_conversation

REFERENCE = "c6b15b25f92bc1a92709ba89fcbe3b3f8aa9e8e6"
SNAPSHOTS = {
    "gemma4_unified": ("gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    "gemma4": ("gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2"),
}


def snapshot(family):
    name, revision = SNAPSHOTS[family]
    path = Path.home() / ".cache/huggingface/hub" / f"models--google--{name}" / "snapshots" / revision
    if not path.is_dir():
        pytest.skip(f"Pinned {name} local snapshot is not installed")
    return path


@lru_cache(None)
def golden(path):
    # Execute the committed source without checking out or editing any worktree.
    result = subprocess.run(["git", "show", f"{REFERENCE}:{path}"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, check=True)
    module = types.ModuleType("reference_" + path.replace("/", "_"))
    exec(compile(result.stdout, f"{REFERENCE}:{path}", "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module", params=list(SNAPSHOTS))
def family(request):
    return request.param


@pytest.fixture(scope="module")
def real_tokenizer(family):
    return dataset.load_tokenizer(snapshot(family))


def tiny_model(family, *, towers=False, large_heads=False, vocab_size=128):
    from transformers import AutoConfig, Gemma4ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(snapshot(family), local_files_only=True)
    text = config.text_config
    config.dtype = text.dtype = "float32"
    for name, value in dict(vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            num_global_key_value_heads=1, head_dim=256 if large_heads else 16,
            global_head_dim=512 if large_heads else 32, max_position_embeddings=4096,
            layer_types=["sliding_attention", "full_attention"] * 2, sliding_window=7,
            attention_dropout=0., use_cache=False).items():
        setattr(text, name, value)
    if family == "gemma4":
        text.num_kv_shared_layers = 2
        text.hidden_size_per_layer_input = 8
        text.vocab_size_per_layer_input = vocab_size
        assert text.attention_k_eq_v is False
    if towers:
        # Real multimodal classes remain present for exclusion checks, without
        # loading their weights or invoking their forwards.
        vision = config.vision_config
        for name, value in dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, head_dim=16, global_head_dim=16,
                position_embedding_size=32).items():
            setattr(vision, name, value)
        if config.audio_config is not None:
            audio = config.audio_config
            for name, value in dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                    output_proj_dims=32, subsampling_conv_channels=[8, 4]).items():
                setattr(audio, name, value)
    else:
        config.vision_config = config.audio_config = None
    config.image_token_id, config.audio_token_id, config.video_token_id = 120, 121, 122
    config._attn_implementation = "sdpa"
    cls = Gemma4ForConditionalGeneration if family == "gemma4" else Gemma4UnifiedForConditionalGeneration
    return cls(config).float()


@pytest.mark.parametrize("packing", ["turn", "episode"])
@pytest.mark.parametrize("replies", [
    ["Action: look"],
    ["  Thought: café — inspect\nAction: look  ", "Action: go north", "Action: look"],
    ["Action: look", "", "Action: go north", " "],
])
def test_real_templates_and_12b_byte_golden(family, real_tokenizer, packing, replies):
    episode = trajectory()
    episode["steps"] = [dict(episode["steps"][0], action_raw=reply, event_id=str(i))
                        for i, reply in enumerate(replies)]
    kwargs = dict(packing=packing, allow_empty_visible=True)
    rows = dataset.episode_examples(episode, real_tokenizer, **kwargs)
    report = dataset.mask_report(rows, real_tokenizer)
    expected_prefix = dataset.MODEL_PREFIX + (dataset.EMPTY_THOUGHT if family == "gemma4_unified" else "")
    assert report["evaluation_prefix_mismatch"]["generation_prefix"] == expected_prefix
    if family == "gemma4_unified":
        old = golden("train/dataset.py")
        expected = old.episode_examples(episode, real_tokenizer, **kwargs)
        assert canonical_bytes(rows) == canonical_bytes(expected)
        assert canonical_bytes(report) == canonical_bytes(old.mask_report(expected, real_tokenizer))
    else:
        assert report["evaluation_prefix_mismatch"]["affected_turns"] == 0
        if packing == "turn":
            for row in rows:
                if row["supervised_tokens"]:
                    assert row["generation_start"] == row["prompt_tokens"]
                    assert real_tokenizer.decode(row["input_ids"][:row["prompt_tokens"]]).endswith(expected_prefix)
    if packing == "turn":
        row = {"packing": "turn", "turns": rows}
        packed = pack_shared_history(row)
        if family == "gemma4_unified":
            assert canonical_bytes(packed) == canonical_bytes(golden("baselines/common/shared_history.py").pack_shared_history(row))
        for branch, turn in enumerate(sequence_rows(row), 1):
            indices = [i for i, b in enumerate(packed["branches"]) if b == branch]
            path = [i for i in range(indices[0]) if packed["branches"][i] == 0] + indices
            assert [packed["input_ids"][i] for i in path] == turn["input_ids"]
            assert [packed["position_ids"][i] for i in path] == list(range(len(path)))


@pytest.mark.parametrize("reply", ["", " ", "Action: <|turn>model", "Action: <|think|>"])
def test_reply_guards_remain(family, real_tokenizer, reply):
    with pytest.raises(ValueError, match="transformed|control token"):
        dataset.encode_turn(real_tokenizer, [{"role": "user", "content": "Look"}], reply, max_length=1000)


def test_prefix_and_token_boundary_guards(family, real_tokenizer):
    class Changed:
        def __getattr__(self, name):
            return getattr(real_tokenizer, name)

        def apply_chat_template(self, *args, **kwargs):
            text = real_tokenizer.apply_chat_template(*args, **kwargs)
            return text + "unexpected" if kwargs["add_generation_prompt"] else text

    with pytest.raises(ValueError, match="OFF generation prefix changed"):
        dataset.encode_turn(Changed(), [{"role": "user", "content": "Look"}], "Action: look", max_length=1000)
    with pytest.raises(ValueError, match="no truncation"):
        dataset.encode_turn(real_tokenizer, [{"role": "user", "content": "Look"}], "Action: look", max_length=1)

    class Crossing(Changed):
        def apply_chat_template(self, *args, **kwargs):
            return real_tokenizer.apply_chat_template(*args, **kwargs)

        def encode(self, text, **kwargs):
            ids = real_tokenizer.encode(text, **kwargs)
            return ids + [7] if text.startswith("<bos>") and text.endswith(dataset.TURN_TOKEN) else ids

    with pytest.raises(ValueError, match="boundary crosses prompt/assistant"):
        dataset.encode_turn(Crossing(), [{"role": "user", "content": "Look"}], "Action: look", max_length=1000)


def numeric_rows(family):
    rows = [random_conversation(1, 3), random_conversation(3, 2, prefix=29)]
    if family == "gemma4":
        for row in rows:
            for turn in row["turns"]:
                start, prompt = turn["generation_start"], turn["prompt_tokens"]
                for field in ("input_ids", "labels"):
                    turn[field] = turn[field][:start] + turn[field][prompt:]
                turn["prompt_tokens"] = start
    return rows


def compare_gradients(actual, expected, *, rtol=2e-5):
    for (name, got), (other, want) in zip(actual.named_parameters(), expected.named_parameters(), strict=True):
        assert name == other
        if want.grad is None:
            assert got.grad is None, name
        else:
            assert got.grad is not None and torch.isfinite(got.grad).all(), name
            error = got.grad - want.grad
            assert error.norm() <= rtol * want.grad.norm() + 2e-6, name
            assert error.abs().max() <= rtol * want.grad.abs().max() + 2e-6, name


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("adaptation", ["full", "lora"])
def test_packed_loss_and_every_gradient(family, backend, checkpointing, adaptation):
    original = tiny_model(family).train()
    original.set_attn_implementation(backend)
    if adaptation == "lora":
        original = modeling.attach_lora(original, rank=2, alpha=4, gradient_checkpointing=checkpointing)
        with torch.no_grad():
            for name, param in original.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=.03)
    elif checkpointing:
        original.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    packed = deepcopy(original)
    rows = numeric_rows(family)
    # Nonuniform weights and interior mask holes also exercise SAD-style labels.
    rows[0]["turns"][1]["labels"][-2] = -100
    expected = batch_logps(original, rows, "cpu")
    actual = shared_history_logps(packed, rows, "cpu")
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    weights = torch.tensor([.37, .63])
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    compare_gradients(packed, original)


def test_flex_routes_all_text_layers_and_recomputation(family, monkeypatch):
    from torch.nn.attention.flex_attention import BlockMask
    original = tiny_model(family, large_heads=True).train()
    packed = deepcopy(original)
    packed.set_attn_implementation("flex_attention")
    packed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    settings = configure_flex_attention(packed)
    configure_flex_attention(packed)
    layers = modeling.text_decoder_layers(packed)
    assert all(len(layer.self_attn._forward_pre_hooks) == 1 for layer in layers)
    calls = []

    def boundary(q, k, v, *, block_mask, kernel_options, **kwargs):
        assert isinstance(block_mask, BlockMask)
        assert kernel_options == (settings["kernel_options"] if q.shape[-1] > 256 else None)
        calls.append(q.shape[-1])
        b = torch.arange(q.shape[0])[:, None, None, None]
        qi = torch.arange(q.shape[-2])[None, None, :, None]
        ki = torch.arange(k.shape[-2])[None, None, None, :]
        mask = block_mask.mask_mod(b, 0, qi, ki)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=kwargs["scale"], enable_gqa=kwargs["enable_gqa"])

    monkeypatch.setattr("transformers.integrations.flex_attention.compile_friendly_flex_attention", boundary)
    rows = numeric_rows(family)
    expected, actual = batch_logps(original, rows, "cpu"), shared_history_logps(packed, rows, "cpu")
    assert calls == [256, 512, 256, 512]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    expected.sum().backward()
    actual.sum().backward()
    assert len(calls) == 8  # every text layer also routed during recomputation
    # Large unscaled 512-d heads amplify fp32 reduction-order differences.
    compare_gradients(packed, original, rtol=1e-4)


def test_lora_targets_and_fsdp_exclude_towers(family, monkeypatch):
    base = tiny_model(family, towers=True)
    layers = modeling.text_decoder_layers(base)
    expected_class = "Gemma4TextDecoderLayer" if family == "gemma4" else "Gemma4UnifiedTextDecoderLayer"
    assert len(layers) == 4 and all(type(layer).__name__ == expected_class for layer in layers)
    assert modeling.text_decoder_name(base.config.model_type) == expected_class
    monkeypatch.setattr("baselines.common.fsdp_train.load_student", lambda *a, **kw: base)
    full = prepare_model("unused")
    assert all(p.requires_grad == text_parameter(n) for n, p in full.named_parameters())
    if family == "gemma4":
        assert full.model.audio_tower is not None and full.model.vision_tower is not None
        assert full.model.language_model.per_layer_model_projection.weight.requires_grad
    calls = []
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda m, **kw: calls.append((m, kw)))
    shard_model(full, mesh=None)
    assert [m for m, _ in calls] == layers + [full]
    if family == "gemma4":
        ignored = calls[-1][1]["ignored_params"]
        assert ignored == {p for n, p in full.named_parameters() if not text_parameter(n)}
        assert ignored and not any(p.requires_grad for p in ignored)
    else:
        new_calls = calls[:]
        calls.clear()
        golden("baselines/common/fsdp_train.py").shard_model(full, mesh=None)
        assert calls == new_calls
    configure_flex_attention(base)
    assert all(not getattr(m, "_shared_history_flex_hook", False)
               for n, m in base.named_modules() if "vision" in n or "audio" in n)
    expected_names = [n for n, _ in base.named_modules()
                      if n.startswith("model.language_model.layers.") and n.rsplit(".", 1)[-1] in modeling.LORA_TARGETS]
    old_base = deepcopy(base)
    adapted = modeling.attach_lora(base, rank=2, alpha=4, gradient_checkpointing=False)
    actual_names = [n.removeprefix("base_model.model.") for n, m in adapted.named_modules() if hasattr(m, "lora_A")]
    assert set(actual_names) == set(expected_names)
    if family == "gemma4":
        assert set(adapted.peft_config["default"].target_modules) == set(expected_names)
    assert all(".language_model.layers." in n and "lora_" in n
               for n, p in adapted.named_parameters() if p.requires_grad)
    if family == "gemma4":
        assert len(expected_names) == 2 * 7 + 2 * 5
        assert all(not hasattr(layer.self_attn, "k_proj") and not hasattr(layer.self_attn, "v_proj") for layer in layers[2:])
    else:
        old = golden("train/modeling.py").attach_lora(old_base, rank=2, alpha=4, gradient_checkpointing=False)
        assert old.peft_config["default"].to_dict() == adapted.peft_config["default"].to_dict()


def test_load_dispatch_and_metadata(family, monkeypatch):
    import transformers
    cls = transformers.Gemma4ForConditionalGeneration if family == "gemma4" else transformers.Gemma4UnifiedForConditionalGeneration
    calls = []
    monkeypatch.setattr(cls, "from_pretrained", lambda *a, **kw: calls.append((a, kw)))
    modeling.load_student(snapshot(family), device="cpu", dtype="float32")
    assert len(calls) == 1
    if family == "gemma4_unified":
        expected = modeling.snapshot_metadata(snapshot(family))
        assert canonical_bytes(expected) == canonical_bytes(golden("train/modeling.py").snapshot_metadata(snapshot(family)))
        golden("train/modeling.py").load_student(snapshot(family), device="cpu", dtype="float32")
        assert calls[0] == calls[1]


def test_real_prefix_gradients_smartad_selection_and_hf_reference(family, real_tokenizer):
    from baselines.smartad.data import score_episode
    from tools.check_lora_serving import hf_logprobs
    episode = trajectory()
    original = tiny_model(family, vocab_size=len(real_tokenizer)).eval()
    packed = deepcopy(original)
    row = conversation(episode, real_tokenizer, source="trusted_teacher_seed", packing="turn")
    expected = batch_logps(original, [row], "cpu")
    actual = shared_history_logps(packed, [row], "cpu")
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    expected.sum().backward()
    actual.sum().backward()
    compare_gradients(packed, original)
    selection = score_episode(original, real_tokenizer, episode, device="cpu")
    assert selection["equation"] == "mean_turn(mean_token(NLL))"
    for turn, score in zip(row["turns"], selection["turns"], strict=True):
        values = hf_logprobs(original, turn)
        assert len(values) == score["tokens"]
        assert -sum(values) == pytest.approx(score["nll_sum"], rel=1e-6, abs=1e-5)
    assert selection["nll"] == sum(t["nll_mean"] for t in selection["turns"]) / len(selection["turns"])


@pytest.mark.parametrize("mode", ["targets", "gemma4"])
def test_serving_conversion_real_peft_and_12b_golden(family, tmp_path, mode):
    from safetensors.torch import load_file
    model = modeling.attach_lora(tiny_model(family), rank=2, alpha=4, gradient_checkpointing=False)
    if family == "gemma4_unified":
        # The legacy converter requires full paths. PEFT minimizes this four-
        # layer fixture's target list; expand only the fixture, not the recipe.
        model.peft_config["default"].target_modules = {
            n.removeprefix("base_model.model.") for n, m in model.named_modules() if hasattr(m, "lora_A")}
    source = tmp_path / "adapter"
    model.save_pretrained(source, safe_serialization=True)
    adapter = {"path": str(source), "served_path": str(source),
               "adapter_config_sha256": file_hash(source / "adapter_config.json"),
               "adapter_weights_sha256": file_hash(source / "adapter_model.safetensors")}
    derived = derive_gemma4_value_lora(adapter, snapshot(family), tmp_path / "derived") if family == "gemma4" else adapter
    if family == "gemma4":
        assert derived is adapter and not (tmp_path / "derived").exists()
    actual = convert_adapter(derived, tmp_path / "converted", mode, snapshot=snapshot(family))
    if family == "gemma4":
        validate_served_adapter(actual, snapshot(family))
        assert actual["conversion"]["model_type"] == "gemma4"
    else:
        expected = golden("harness/lora_adapters.py").convert_adapter(adapter, tmp_path / "golden", mode)
        assert canonical_bytes(actual["conversion"]) == canonical_bytes(expected["conversion"])
        for name in ("adapter_config.json", "adapter_model.safetensors", "conversion.json"):
            assert (tmp_path / "converted" / name).read_bytes() == (tmp_path / "golden" / name).read_bytes()
    before, after = load_file(source / "adapter_model.safetensors"), load_file(tmp_path / "converted/adapter_model.safetensors")
    rename = actual["conversion"]["tensor_renames"]
    assert len(before) == len(after)
    for key, tensor in before.items():
        mapped = rename.get(key, key)
        if mode == "gemma4":
            assert mapped.startswith("base_model.model.language_model.model.layers.")
        assert torch.equal(tensor.view(torch.uint8), after[mapped].view(torch.uint8))


def test_12b_value_derivation_byte_golden(tmp_path):
    base = tiny_model("gemma4_unified")
    base_path = tmp_path / "base"
    base.save_pretrained(base_path)
    model = modeling.attach_lora(base, rank=2, alpha=4, gradient_checkpointing=False)
    source = tmp_path / "adapter"
    model.save_pretrained(source, safe_serialization=True)
    adapter = {"path": str(source)}
    actual = derive_gemma4_value_lora(adapter, base_path, tmp_path / "new")
    expected = golden("harness/lora_adapters.py").derive_gemma4_value_lora(adapter, base_path, tmp_path / "old")
    assert actual["derivation"]["affected_layers"] == expected["derivation"]["affected_layers"] == [1, 3]
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        assert (tmp_path / "new" / name).read_bytes() == (tmp_path / "old" / name).read_bytes()


def test_eseries_load_reload_hf_reference_and_dose_control(tmp_path):
    from peft import PeftModel
    from baselines.common.dose_control import validate_init
    from sed.seed_evidence import artifact_hashes
    from sed.storage import digest, write_json
    from tools.check_lora_serving import add_hf_references, hf_logprobs
    base_path = tmp_path / "base"
    tiny_model("gemma4", towers=True).save_pretrained(base_path)
    base = modeling.load_student(base_path, device="cpu", dtype="float32")
    assert type(base).__name__ == "Gemma4ForConditionalGeneration"
    assert all(p.dtype == torch.float32 for p in base.parameters())
    base.set_attn_implementation("sdpa")
    rows = [{"input_ids": [2, 8, 9, 20, 106], "prompt_tokens": 3}]
    expected_base = hf_logprobs(base, rows[0])
    model = modeling.attach_lora(base, gradient_checkpointing=False)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=.03)
    adapter_path = tmp_path / "sft/checkpoint"
    model.save_pretrained(adapter_path)
    add_hf_references(model, rows)
    assert rows[0]["hf_base_logprobs"] == expected_base
    assert rows[0]["hf_logprobs"] != expected_base
    reloaded = PeftModel.from_pretrained(modeling.load_student(base_path, device="cpu", dtype="float32"), adapter_path)
    reloaded.set_attn_implementation("sdpa")
    torch.testing.assert_close(torch.tensor(hf_logprobs(reloaded, rows[0])), torch.tensor(rows[0]["hf_logprobs"]), rtol=0, atol=0)
    assert set(reloaded.peft_config["default"].target_modules) == set(model.peft_config["default"].target_modules)
    # Exercise the actual hash-bound dose-control validator with synthetic
    # receipts. No environment/teacher interaction or scientific training claim.
    student = {"requested_repo": "google/gemma-4-E4B-it", "revision": SNAPSHOTS["gemma4"][1]}
    verified = tmp_path / "verified"
    write_json(verified / "manifest.json", {"K": 1})
    recipe = {"rank": 16, "alpha": 32, "dropout": 0.0, "learning_rate": 1e-5,
        "tokens_per_update": 512, "packing": "episode", "optimizer": "AdamW", "weight_decay": 0.0,
        "betas": [0.9, 0.999], "eps": 1e-8, "gradient_clip": 1.0, "scheduler": "constant",
        "dtype": "bfloat16", "loss": "supervised_token_mean_per_update", "attention": "eager",
        "microbatch": 1, "gradient_checkpointing": True, "enable_thinking": False,
        "verified": str(verified), "base_snapshot": modeling.snapshot_metadata(base_path)}
    provenance = {"command": "sft", "config": {"models": {"student": student}, "sft": recipe},
                  "inputs": {str(verified / "manifest.json"): file_hash(verified / "manifest.json")}}
    write_json(adapter_path.parent / "provenance.json", provenance)
    write_json(adapter_path / "training_state.json", {"final": True, "packing": "episode", "passes": 1,
        "examples_seen": 1, "supervised_tokens": 2, "identity_sha256": digest(provenance),
        "artifact_sha256": artifact_hashes(adapter_path)})
    result = validate_init(adapter_path, base_path, verified, {"models": {"student": student}},
                           collection_sha256=digest({"K": 1}))
    assert result["student"] == student
