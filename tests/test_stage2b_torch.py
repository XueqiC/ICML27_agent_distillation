"""CPU only. Runs in the taskeq venv when project .venv lacks torch."""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from sed.react import build_messages
from train.dataset import load_tokenizer, episode_examples, mask_report, EMPTY_THOUGHT
from train.modeling import attach_lora, load_student
from train.sft import exposure_plan, train_loop, turn_loss_sum, deterministic_seed
from train.merge_lora import merge_adapter, validate_adapter_artifacts
from test_stage2b_seeds import seed_bank, factories


@pytest.fixture(scope="module")
def tokenizer():
    path = Path(os.environ.get("GEMMA_TOKENIZER_PATH", str(Path.home() /
        ".cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")))
    if not path.is_dir():
        pytest.skip("Pinned tokenizer files are not installed")
    return load_tokenizer(path)


def trajectory():
    visible = {"observation": "UNIQUE_OBSERVATION_SENTINEL", "admissible_commands": ["look"]}
    return {"schema_version": 2, "task_id": "train/a/1", "split": "train", "sealed": False,
            "initial": {"visible": dict(visible, facts="PRIVATE_PDDL")},
            "steps": [{"visible": visible, "event_id": f"event-{i}",
                       "action_raw": f"Thought: step {i}\nAction: look", "audit": "SECRET",
                       "action_outcome": "legal"} for i in range(2)]}


def test_real_tokenizer_native_106_mask_and_eval_prefix(tokenizer):
    episode = trajectory()
    examples = episode_examples(episode, tokenizer)
    report = mask_report(examples, tokenizer)
    history = []
    for i, (row, step) in enumerate(zip(examples, episode["steps"])):
        prompt = tokenizer.apply_chat_template(build_messages(history, {}, episode["initial"]["visible"]),
                      tokenize=False, enable_thinking=False, add_generation_prompt=True)
        prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert row["input_ids"][:row["prompt_tokens"]] == prefix_ids
        assert row["labels"][:len(prefix_ids)] == [-100] * len(prefix_ids)
        assert row["labels"][-1] == 106
        supervised = report["examples"][i]["decoded_supervised_span"]
        assert supervised == step["action_raw"] + "<turn|>"
        assert "UNIQUE_OBSERVATION_SENTINEL" not in supervised
        assert "PRIVATE_PDDL" not in tokenizer.decode(row["input_ids"])
        assert prompt.endswith(EMPTY_THOUGHT)
        history.append({"visible": episode["initial"]["visible"], "reply": step["action_raw"]})
    with pytest.raises(ValueError, match="no truncation"):
        episode_examples(episode, tokenizer, max_length=len(examples[0]["input_ids"]))
    episode["split"] = "dev"
    with pytest.raises(ValueError, match="frozen train"):
        episode_examples(episode, tokenizer)
    diagnostic = episode_examples(episode, tokenizer, diagnostic=True)
    assert not diagnostic[0]["label_provenance"]["training_eligible"]
    episode["split"] = "test"
    with pytest.raises(ValueError, match="Sealed"):
        episode_examples(episode, tokenizer, diagnostic=True)


def tiny_model():
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedTextConfig, Gemma4UnifiedForConditionalGeneration
    text = Gemma4UnifiedTextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
             head_dim=8, global_head_dim=8, max_position_embeddings=128,
             layer_types=["sliding_attention", "full_attention"], sliding_window=32)
    config = Gemma4UnifiedConfig(text_config=text, vision_config=None, audio_config=None,
                                image_token_id=120, audio_token_id=121, video_token_id=122)
    config._attn_implementation = "eager"
    return Gemma4UnifiedForConditionalGeneration(config)


def test_official_forward_backward_checkpoint_reload_and_merge(tmp_path):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    torch.set_num_threads(1)
    deterministic_seed(7)
    base = tmp_path / "base"
    tiny_model().save_pretrained(base)
    model = attach_lora(load_student(base, device="cpu", dtype="float32"), rank=2, alpha=4)
    rows = [{"input_ids": [2, 8, 9, 10, 106], "labels": [-100, -100, -100, 10, 106],
             "supervised_tokens": 2},
            {"input_ids": [2, 5, 6, 11, 12, 106], "labels": [-100, -100, -100, 11, 12, 106],
             "supervised_tokens": 3}]
    model.eval()
    row = rows[0]
    with torch.no_grad():
        expected = model(input_ids=torch.tensor([row["input_ids"]]),
                         labels=torch.tensor([row["labels"]]), use_cache=False).loss * 2
        actual = turn_loss_sum(model, row, "cpu")
    torch.testing.assert_close(actual, expected)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    plan = exposure_plan([2, 3], token_budget=5, passes=2, checkpoint_tokens=[5])
    result = train_loop(model, rows, plan, tmp_path / "train", learning_rate=1e-3, identity_sha256="unit")
    assert len(result) == 2 and result[-1]["supervised_tokens"] == 10
    assert result[-1]["updates"] == 2 and result[-1]["passes"] == 2
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)
    adapter = tmp_path / "train" / result[-1]["path"]
    state = json.loads((adapter / "training_state.json").read_text())
    assert state["peak_memory_bytes"] is None and state["seconds_per_update"] > 0
    assert (adapter / "optimizer.pt").is_file()
    assert validate_adapter_artifacts(adapter) == state
    (adapter / "unexpected").write_text("tampered")
    with pytest.raises(ValueError, match="artifacts changed"):
        validate_adapter_artifacts(adapter)
    (adapter / "unexpected").unlink()
    report = merge_adapter(base, adapter, tmp_path / "merged", [2, 8, 9],
                          device="cpu", dtype="float32", atol=1e-5, rtol=1e-5)
    assert report["passed"] and report["checks"]["merged_reloaded"]["argmax_equal"]
    assert (tmp_path / "merged" / "model.safetensors").exists()
    assert not (tmp_path / "merged" / "adapter_config.json").exists()


def test_verified_loader_checks_train_evidence_and_integrity(seed_bank, tmp_path, tokenizer):
    from teachers.collect_seeds import collect
    from sed.seed_selection import select_seeds
    from train.dataset import build_verified_dataset
    root, manifest, freeze, config = seed_bank
    environment, teacher = factories(root, config, set(), [])
    output = tmp_path / "verified"
    result = collect(select_seeds(manifest, freeze, k=1), manifest, freeze, config, output,
                     env_factory=environment, teacher_factory=teacher)
    rows = build_verified_dataset(output, tokenizer, manifest, freeze)
    assert len(rows) == 1 and rows[0]["labels"][-1] == 106
    episode_path = output / result["verified"][0]["episode_dir"] / "episode.json"
    episode = json.loads(episode_path.read_text())
    episode["initial"]["visible"]["observation"] = "changed"
    episode_path.write_text(json.dumps(episode))
    with pytest.raises(ValueError, match="evidence changed"):
        build_verified_dataset(output, tokenizer, manifest, freeze)


def test_accumulation_is_supervised_token_mean(tmp_path):
    torch.set_num_threads(1)
    deterministic_seed(3)
    base = tmp_path / "base"
    tiny_model().save_pretrained(base)
    first = attach_lora(load_student(base, device="cpu", dtype="float32"), rank=2, alpha=4,
                        gradient_checkpointing=False)
    second = attach_lora(load_student(base, device="cpu", dtype="float32"), rank=2, alpha=4,
                         gradient_checkpointing=False)
    second.load_state_dict(first.state_dict())
    rows = [{"input_ids": [2, 5, 106], "labels": [-100, -100, 106], "supervised_tokens": 1},
            {"input_ids": [2, 5, 6, 7, 106], "labels": [-100, 5, 6, 7, 106], "supervised_tokens": 4}]
    plan = exposure_plan([1, 4], token_budget=5, passes=1)
    train_loop(first, rows, plan, tmp_path / "train", learning_rate=1e-3, identity_sha256="unit")
    second.train()
    params = [p for p in second.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0, foreach=False, fused=False)
    loss = sum(turn_loss_sum(second, rows[i], "cpu") for i in plan[0]["indices"]) / 5
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b)
    diagnostic = dict(rows[0], label_provenance={"training_eligible": False})
    with pytest.raises(ValueError, match="Diagnostic"):
        train_loop(first, [diagnostic], plan, tmp_path / "bad", identity_sha256="unit")
