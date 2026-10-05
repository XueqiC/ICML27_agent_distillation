"""Tiny CPU models and installed local tokenizer only; no GPU/network."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from baselines.common.fsdp_train import prepare_lora_model, export_adapter, train_loop, sft_objective
from baselines.common.schedule import lr_multiplier
from harness.lora_adapters import inspect_adapter
from sed.config import load_config
from sed.storage import write_json
from train.modeling import LORA_TARGETS, load_student, snapshot_metadata
from test_stage2b_torch import tiny_model, tokenizer, trajectory


@pytest.mark.parametrize("total,warmup", [(1, 0), (5, 0), (9, 0), (10, 1), (27, 2), (27, 3), (3, 3), (3, 5)])
def test_linear_every_step_matches_transformers(total, warmup):
    from transformers import get_linear_schedule_with_warmup
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total)
    for step in range(total + warmup + 2):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(lr_multiplier(step,
            scheduler="linear", total_steps=total, warmup_ratio=.1, warmup_steps=warmup))
        optimizer.step()
        schedule.step()


def test_kang_text_targets_update_save_reload_and_serving_schema(tmp_path):
    from peft import PeftModel
    torch.set_num_threads(1)
    torch.manual_seed(0)
    base = tmp_path / "base"
    tiny_model().save_pretrained(base)
    model = prepare_lora_model(base, device="cpu")
    config = model.peft_config["default"]
    assert (config.r, config.lora_alpha, config.lora_dropout) == (64, 128, .05)
    assert config.bias == "none" and not config.use_dora and not config.use_rslora
    assert len(config.target_modules) == 14
    assert all(".language_model." in name for name in config.target_modules)
    assert {name.rsplit(".", 1)[-1] for name in config.target_modules} == set(LORA_TARGETS)
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    assert model.config.get_text_config()._attn_implementation == "sdpa"
    assert not model.config.get_text_config().use_cache
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rows = [{"input_ids": [2, 7, 8, 106], "labels": [-100, -100, 8, 106]},
            {"input_ids": [2, 6, 106, 10, 11, 106], "labels": [-100, 6, 106, -100, 11, 106]}]
    model.eval()
    with torch.no_grad():
        row = rows[1]
        native = model(input_ids=torch.tensor([row["input_ids"]]), labels=torch.tensor([row["labels"]]), use_cache=False).loss
        actual, _ = sft_objective(model, row, "cpu")
        torch.testing.assert_close(actual, native)
    output = tmp_path / "train"
    state = train_loop(model, rows, output, objective="sft", learning_rate=2e-4, scheduler="linear",
        warmup_ratio=0, warmup_steps=0, original_batch=8, epochs=2, seed=0)
    assert state["examples_seen"] == 4 and state["supervised_tokens_seen"] == 12
    assert state["b_prime"] == 1 and state["updates"] == 4 and state["gradient_clip"] == 1
    assert state["weight_decay"] == 0
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if not p.requires_grad)
    updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert [u["lr"] for u in updates] == pytest.approx([2e-4, 1.5e-4, 1e-4, .5e-4])
    provenance = {"config": load_config()}
    provenance["config"]["sft"] = {"base_snapshot": snapshot_metadata(base)}
    write_json(output / "provenance.json", provenance)
    saved = export_adapter(model, output / "checkpoint", provenance, state)
    adapter = inspect_adapter("kang", output / "checkpoint", provenance["config"]["models"]["student"], base)
    assert adapter["rank"] == 64 and adapter["checkpoint_metadata"] == {"packing": "episode", "passes": 2, "supervised_tokens": 12}
    assert saved["final"] and (output / "checkpoint/adapter_model.safetensors").is_file()
    loaded = PeftModel.from_pretrained(load_student(base, device="cpu", dtype="float32"), output / "checkpoint")
    loaded.eval()
    model.eval()
    ids = torch.tensor([[2, 7, 8, 106]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(input_ids=ids, use_cache=False).logits, model(input_ids=ids, use_cache=False).logits)


def test_kang_targets_exclude_multimodal_decoys():
    from train.modeling import attach_lora
    model = tiny_model()
    model.model.vision_tower = torch.nn.ModuleDict({name: torch.nn.Linear(32, 32) for name in LORA_TARGETS})
    adapted = attach_lora(model, rank=64, alpha=128, dropout=.05, gradient_checkpointing=False)
    assert all(".language_model." in n for n in adapted.peft_config["default"].target_modules)
    assert all(not p.requires_grad for n, p in adapted.named_parameters() if "vision_tower" in n)


def test_accumulated_conversation_preserves_all_assistant_closures(tokenizer, monkeypatch):
    from baselines.agentgym.data import training_examples
    from baselines.common.conversation import episode_messages, conversation
    episode = trajectory()
    rows = [{"task_id": episode["task_id"], "episode_id": "seed", "parent_seed_ids": ["seed"],
             "source": source, "messages": episode_messages(episode)}
            for source in ["student_exploration", "trusted_teacher_seed"]]
    monkeypatch.setattr("baselines.agentgym.data.load_dataset", lambda *a: rows)
    examples = training_examples("fake", {}, tokenizer)
    expected = conversation(episode, tokenizer, source="trusted_teacher_seed")
    for i, example in enumerate(examples):
        assert example["input_ids"] == expected["input_ids"] and example["labels"] == expected["labels"]
        assert example["label_provenance"]["source"] == rows[i]["source"]
        assert example["label_provenance"]["parent_seed_ids"] == ["seed"]
        assert all(example["labels"][s["stop"] - 1] == 106 for s in example["assistant_spans"])
        assert all(t["source"] == rows[i]["source"] for t in example["label_provenance"]["turns"])


def test_explicit_kang_schedule_matches_installed_library_defaults():
    from transformers import TrainingArguments
    # Upstream pyproject pins TRL 0.19.1, but does NOT pin transformers.
    args = TrainingArguments(output_dir="unused", use_cpu=True, report_to="none")
    assert args.lr_scheduler_type.value == "linear" and args.get_warmup_steps(20) == 0
