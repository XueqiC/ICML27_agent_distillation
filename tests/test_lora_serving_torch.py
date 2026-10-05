"""CPU tensor/reference checks in the taskeq environment; no serving runtime."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from harness.lora_adapters import convert_adapter
from sed.react import build_messages
from sed.seed_selection import select_seeds
from teachers.collect_seeds import collect
from tools.check_lora_serving import fixed_prompts, hf_logprobs
from train.dataset import build_verified_dataset, EMPTY_THOUGHT
from train.modeling import attach_lora
from test_stage2b_torch import tiny_model, tokenizer
from test_stage2b_seeds import seed_bank, factories


def test_hf_logprob_alignment_and_real_peft_conversion(tmp_path):
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    model = attach_lora(tiny_model(), rank=2, alpha=4, gradient_checkpointing=False).eval()
    # Nonzero B exercises the adapter rather than the zero-initialized base.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.fill_(0.03)
    row = {"input_ids": [2, 8, 9, 10, 106], "prompt_tokens": 3}
    ids = torch.tensor([row["input_ids"]])
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
        expected = logits[2:4].log_softmax(-1).gather(1, ids[0, 3:, None]).squeeze(1)
    torch.testing.assert_close(torch.tensor(hf_logprobs(model, row)), expected)
    path = tmp_path / "adapter"
    model.save_pretrained(path, safe_serialization=True)
    before = load_file(path / "adapter_model.safetensors")
    converted = convert_adapter({"path": str(path)}, tmp_path / "converted", "gemma4")
    after = load_file(tmp_path / "converted/adapter_model.safetensors")
    mapping = converted["conversion"]["tensor_renames"]
    assert len(mapping) == len(before) == len(after)
    for key, value in before.items():
        other = after[mapping[key]]
        assert value.dtype == other.dtype and value.shape == other.shape
        assert torch.equal(value.view(torch.uint8), other.view(torch.uint8))
    cfg = json.loads((path / "adapter_config.json").read_text())
    assert all("model.language_model." in t for t in cfg["target_modules"])


def test_eight_verified_turns_exact_evaluation_template(seed_bank, tmp_path, tokenizer):
    root, manifest, freeze, config = seed_bank
    environment, teacher = factories(root, config, set(), [])
    output = tmp_path / "verified"
    collect(select_seeds(manifest, freeze, k=10), manifest, freeze, config, output,
            env_factory=environment, teacher_factory=teacher)
    rows = build_verified_dataset(output, tokenizer, manifest, freeze, packing="turn")
    selected = fixed_prompts(rows)
    assert len(selected) == 8 and all(r["source"]["training_eligible"] for r in selected)
    for row in selected:
        prompt = tokenizer.decode(row["input_ids"][:row["prompt_tokens"]], skip_special_tokens=False)
        assert prompt.endswith(EMPTY_THOUGHT)
        assert "SECRET" not in prompt and "PRIVATE_PDDL" not in prompt
        assert row["input_ids"][-1] == 106
    # The same builder and unchanged OFF template define the actual evaluation prefix.
    visible = {"observation": "Visible room at 0", "admissible_commands": ["look", "go north"]}
    expected = tokenizer.apply_chat_template(build_messages([], {}, visible), tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
    assert tokenizer.decode(selected[0]["input_ids"][:selected[0]["prompt_tokens"]]) == expected
