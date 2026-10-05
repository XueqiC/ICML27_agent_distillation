"""Tiny CPU Gemma/PEFT plus the installed pinned tokenizer; no GPU/network."""
from copy import deepcopy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from baselines.common.fsdp_train import batch_logps
from baselines.common.schedule import lr_multiplier
from baselines.leap.data import encode_rows, step_rows, min_subsample
from baselines.leap.protocol import parse_correction
from baselines.leap.train import attach_fresh_lora, prepare_model, train_loop
from train.merge_lora import next_token_logits
from train.modeling import load_student
from test_stage2b_torch import tiny_model, tokenizer
from test_stage3d import episode, correction


def test_real_gemma_template_only_turn_t_supervised_original_history_and_no_oracle(tokenizer):
    original = episode()
    rows = step_rows(original, parents=["seed"], corrections=parse_correction(correction(), 2))
    examples = encode_rows(rows, tokenizer)
    for i, (row, example) in enumerate(zip(rows, examples)):
        prefix = tokenizer.apply_chat_template(row["messages"], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        assert example["input_ids"][:len(ids)] == ids
        assert example["labels"][:len(ids)] == [-100] * len(ids)
        assert tokenizer.decode(example["labels"][len(ids):]) == row["target"] + "<turn|>"
        assert example["labels"][-1] == 106
        decoded = tokenizer.decode(example["input_ids"])
        assert "SECRET" not in decoded and "PRIVATE" not in decoded
        assert "ORIGINAL_REASON_0" in decoded if i else "ORIGINAL_REASON_0" not in decoded
        assert example["label_provenance"]["verification"] == "unknown_teacher_label"
    long = deepcopy(rows)
    long[0]["target"] = "Thought: " + "word " * 40000 + "\nAction: look"
    with pytest.raises(ValueError, match="no truncation"):
        encode_rows(long, tokenizer)


def test_qv_only_fresh_lora_after_checked_merge_and_weight_update(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    base = tmp_path / "base"
    raw = tiny_model()
    raw.model.vision_tower = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(32, 32),
                                               "v_proj": torch.nn.Linear(32, 32)})
    decoy = attach_fresh_lora(raw, gradient_checkpointing=False)
    assert all(not p.requires_grad for n, p in decoy.named_parameters() if "vision_tower" in n)
    del raw, decoy
    tiny_model().save_pretrained(base)
    model = prepare_model(base, device="cpu")
    config = model.peft_config["default"]
    assert (config.r, config.lora_alpha, config.lora_dropout) == (128, 64, .05)
    assert len(config.target_modules) == 4
    assert all(".language_model." in n and n.rsplit(".", 1)[-1] in {"q_proj", "v_proj"} for n in config.target_modules)
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    assert model.config.get_text_config()._attn_implementation == "sdpa"
    with pytest.raises(ValueError, match="merged model"):
        attach_fresh_lora(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rows = [{"input_ids": [2, 5, 6, 106], "labels": [-100, -100, 6, 106]} for _ in range(16)]
    state = train_loop(model, rows, tmp_path / "train")
    assert state["b_prime"] == 2 and state["updates"] == 8 and state["examples_seen"] == 16
    assert state["supervised_tokens_seen"] == 32 and state["gradient_clip"] == .3
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if not p.requires_grad)
    updates = [json.loads(line) for line in (tmp_path / "train/updates.jsonl").read_text().splitlines()]
    assert [u["lr"] for u in updates] == pytest.approx([3e-5 * s / 10 for s in range(8)])
    assert all(u["microbatches"] == [2] for u in updates)
    reference = next_token_logits(model, [2, 5, 6])
    merged = model.merge_and_unload(safe_merge=True)
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    torch.testing.assert_close(reference, next_token_logits(merged, [2, 5, 6]), atol=1e-5, rtol=1e-5)
    merged.save_pretrained(tmp_path / "merged")
    next_model = prepare_model(tmp_path / "merged", device="cpu")
    assert next_model.peft_config["default"].r == 128
    assert all(torch.count_nonzero(p) == 0 for n, p in next_model.named_parameters() if "lora_B" in n)
    assert all(torch.count_nonzero(p) > 0 for n, p in next_model.named_parameters() if "lora_A" in n)
    torch.testing.assert_close(reference, next_token_logits(next_model, [2, 5, 6]), atol=1e-5, rtol=1e-5)


def test_microbatch_completion_loss_matches_native_token_ce():
    torch.set_num_threads(1)
    model = tiny_model().eval()
    rows = [{"input_ids": [2, 5, 106], "labels": [-100, -100, 106]},
            {"input_ids": [2, 7, 9, 106], "labels": [-100, 7, 9, 106]}]
    ids = torch.tensor([[2, 5, 106, 0], [2, 7, 9, 106]])
    labels = torch.tensor([[-100, -100, 106, -100], [-100, 7, 9, 106]])
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    with torch.no_grad():
        native = model(input_ids=ids, attention_mask=attention, labels=labels, use_cache=False).loss
        ours = -batch_logps(model, rows, "cpu").sum() / 4
    torch.testing.assert_close(native, ours)


@pytest.mark.parametrize("n", [1, 8, 10, 25, 100])
def test_fixed_ten_step_cosine_matches_installed_transformers(n):
    from transformers import get_cosine_schedule_with_warmup
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=n)
    for step in range(n + 1):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(lr_multiplier(step,
            scheduler="cosine", total_steps=n, warmup_ratio=0, warmup_steps=10))
        optimizer.step()
        scheduler.step()


def test_min_subsampling_matches_official_hf_dataset_shuffle():
    datasets = pytest.importorskip("datasets")
    sets = [[{"id": f"{i}-{j}"} for j in range(n)] for i, n in enumerate([8, 3, 5])]
    sources = [datasets.Dataset.from_list(rows) for rows in sets]
    # Execute the actual released aggregation statements with only seed changed
    # to the user's fixed 0. No upstream module/remote dataset is imported.
    import ast
    from pathlib import Path
    tree = ast.parse(Path("third_party/leap_llm/scripts/train/sft.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assignments = sorted([n for n in ast.walk(main) if isinstance(n, ast.Assign)
        and 59 <= n.lineno <= 83 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in {"N_train", "N_eval", "train_dataset", "eval_dataset"}], key=lambda n: n.lineno)
    class Seed(ast.NodeTransformer):
        def visit_keyword(self, node):
            if node.arg == "seed": node.value = ast.Constant(value=0)
            return node
    block = Seed().visit(ast.Module(body=assignments, type_ignores=[]))
    ast.fix_missing_locations(block)
    scope = {"train_datasets": sources, "eval_datasets": sources,
             "concatenate_datasets": datasets.concatenate_datasets}
    exec(compile(block, "official_leap_min_aggregation", "exec"), scope)
    actual, _ = min_subsample(sets)
    assert list(scope["train_dataset"]) == actual
