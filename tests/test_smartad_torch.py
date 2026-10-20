"""Independent equations and tiny CPU model gradient/update checks."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")
import torch.nn.functional as F

from baselines.common.fsdp_train import train_loop
from baselines.common.schedule import dose
from baselines.smartad.data import encode_episode, macro_nll, microsteps, objective, score_episode, weighted_ce
from test_sad import CharTokenizer, episode
from train.dataset import TURN_TOKEN


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(512, 5)
        self.head = torch.nn.Linear(5, 512)
    def forward(self, input_ids, attention_mask, logits_to_keep=None, **kwargs):
        hidden = self.embedding(input_ids).cumsum(1) / 100
        if logits_to_keep is not None:
            hidden = hidden[:, logits_to_keep]
        return SimpleNamespace(logits=self.head(hidden))


def trajectory():
    ep = episode("Thought: check\nAction: look")
    ep["steps"].append({**ep["steps"][0], "event_id": "e2", "action_raw": "Thought: done\nAction: look"})
    return ep


def test_labels_exact_prefix_and_no_hidden_fields():
    tokenizer = CharTokenizer()
    example = encode_episode(trajectory(), tokenizer)
    first, last = example["turns"]
    for row, kind, weight in ((first, "action", 1.5), (last, "final", 2.0)):
        assert tokenizer.decode([x for x, k in zip(row["input_ids"], row["segment_labels"]) if k == kind]) == "Action: look"
        assert all(w == weight for w, k in zip(row["token_weights"], row["segment_labels"]) if k == kind)
        assert row["segment_labels"][-1] == "reason" and row["token_weights"][-1] == 1
        assert row["labels"][-1] == 106
        assert all(v == -100 for v in row["labels"][:row["prompt_tokens"]])
        assert all(v == 0 for v in row["token_weights"][:row["prompt_tokens"]])
        assert "HIDDEN" not in tokenizer.decode(row["input_ids"])
    assert "Thought: check" in tokenizer.decode(last["input_ids"][:last["prompt_tokens"]])
    with pytest.raises(ValueError, match="no truncation"):
        encode_episode(trajectory(), tokenizer, max_length=10)


def test_macro_turn_equation_and_exact_model_context():
    assert macro_nll([[1], [3, 3, 3]]) == 2
    assert macro_nll([[1], [3, 3, 3]]) != 2.5  # not the global token mean
    torch.manual_seed(0)
    model, tokenizer = Tiny(), CharTokenizer()
    ep = trajectory()
    example = encode_episode(ep, tokenizer)
    losses = []
    for row in example["turns"]:
        ids = torch.tensor([row["input_ids"]])
        labels = torch.tensor(row["labels"])[1:]
        logits = model(ids, torch.ones_like(ids)).logits[0, :-1]
        values = F.cross_entropy(logits, labels, reduction="none", ignore_index=-100)
        losses.append(values[labels != -100].tolist())
    result = score_episode(model, tokenizer, ep, device="cpu")
    assert result["nll"] == pytest.approx(macro_nll(losses), rel=1e-6)
    assert [r["tokens"] for r in result["turns"]] == [len(x) for x in losses]


def test_eq5_normalization_and_gradient_across_unequal_trajectories(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model, reference = Tiny(), None
    reference = deepcopy(model)
    examples = [encode_episode(trajectory(), CharTokenizer()),
                encode_episode(episode("Action: look"), CharTokenizer())]
    numerator, denominator = 0, 0
    for example in examples:
        for row in example["turns"]:
            ids = torch.tensor([row["input_ids"]])
            labels, weights = torch.tensor([row["labels"]]), torch.tensor([row["token_weights"]])
            logits = reference(ids, torch.ones_like(ids)).logits[:, :-1]
            loss = weighted_ce(logits, labels[:, 1:], weights[:, 1:])
            numerator += loss * weights.sum()
            denominator += weights.sum()
    expected = numerator / denominator
    expected.backward()
    actual = 0
    for row, weight in microsteps(examples, [0, 1]):
        loss, _ = objective(model, row, "cpu")
        scaled = loss * weight / 2
        actual += scaled.detach()
        scaled.backward()
    torch.testing.assert_close(actual, expected.detach())
    for p, ref in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, ref.grad, atol=1e-6, rtol=1e-5)
    before = [p.detach().clone() for p in model.parameters()]
    result = train_loop(model, examples, tmp_path, objective="smartad", learning_rate=2e-4,
        scheduler="linear", warmup_ratio=0, warmup_steps=0, original_batch=4, epochs=2, seed=0,
        packing="turn")
    assert result["b_prime"] == 1 and result["examples_seen"] == 4
    assert result["loss_reduction"] == "segment_weighted_token_mean_over_global_batch"
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


@pytest.mark.parametrize("n,expected", [(1, 1), (5, 1), (10, 1), (20, 2), (31, 3), (50, 4), (100, 4)])
def test_batch_shrink_rule(n, expected):
    assert dose(n, 4, epochs=2)["b_prime"] == expected
