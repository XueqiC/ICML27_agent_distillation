"""Pinned local Gemma tokenizer and tiny random Gemma; CPU only."""
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from test_stage2b_torch import tokenizer, tiny_model
from test_had import state
from baselines.had.data import encode_state
from baselines.had.objective import balanced_backward


def test_native_command_masks_and_tiny_gemma_gradient(tokenizer):
    torch.set_num_threads(1)
    row = encode_state(state(), tokenizer)
    for key, text in (("chosen", "take mug 1 from desk 1"), ("rejected", "look")):
        actual = tokenizer.decode([i for i in row[key]["labels"] if i != -100])
        assert actual.strip() == text
        assert row[key]["labels"][-1] == -100
    model = tiny_model()
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    metrics = balanced_backward(model, [row])
    assert metrics["lambda"] > 0 and metrics["valid_states"] == 1
