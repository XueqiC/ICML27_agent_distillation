"""Tiny CPU model: exact masked gradient matching, updates and DCP resume."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")

from baselines.had.objective import losses, balanced_backward, train_round
from baselines.had.training import optimizer_for, resume_state


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 5)
        self.head = torch.nn.Linear(5, 17)
    def forward(self, input_ids, attention_mask, logits_to_keep, **kwargs):
        hidden = self.embedding(input_ids).cumsum(1)
        return NS(logits=self.head(hidden[:, logits_to_keep]))


def row(ids, start):
    return {"input_ids": ids, "labels": [-100] * start + ids[start:]}


def examples():
    return [{"state_id": str(i), "policy_version": 0, "mask": i != 1,
        "response": row([1, 2, 3, 4, 5], 2), "chosen": row([1, 6, 7, 8, 9], 3),
        "rejected": row([1, 6, 7, 10, 11, 12], 3)} for i in range(3)]


@pytest.mark.parametrize("masks", [[True, False, True], [False]*3, [True]*3])
def test_streamed_gradient_equals_equations_8_9(masks):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model, rows = Tiny(), examples()
    for r, mask in zip(rows, masks):
        r["mask"] = mask
    params = list(model.parameters())
    pieces = [losses(model, r, "cpu") for r in rows]
    d_loss = sum(d for d, p in pieces) / len(rows)
    if any(masks):
        p_loss = sum(p for (d, p), m in zip(pieces, masks) if m) / sum(masks)
        gd = torch.autograd.grad(d_loss, params, retain_graph=True)
        gp = torch.autograd.grad(p_loss, params, retain_graph=True)
        lam = .5 * sum(g.square().sum() for g in gd).sqrt() / sum(g.square().sum() for g in gp).sqrt()
        expected = torch.autograd.grad(d_loss + lam.detach() * p_loss, params)
    else:
        lam = torch.tensor(0.)
        expected = torch.autograd.grad(d_loss, params)
    metrics = balanced_backward(model, rows)
    assert metrics["valid_states"] == sum(masks)
    assert metrics["lambda"] == pytest.approx(float(lam), rel=1e-5)
    for p, g in zip(params, expected):
        torch.testing.assert_close(p.grad, g, atol=1e-6, rtol=1e-5)


def test_identical_actions_have_zero_preference_gradient():
    model, rows = Tiny(), examples()
    for r in rows:
        r["rejected"] = r["chosen"]
    metrics = balanced_backward(model, rows)
    assert metrics["lambda"] == 0 and metrics["preference_gradient_norm"] == 0
    assert metrics["preference_loss"] == pytest.approx(.69314718)


def test_exact_fp32_adam_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = Tiny()
    optimizer = optimizer_for(model)
    rows = examples()
    first = train_round(model, optimizer, rows, tmp_path / "first", version=0, start_update=0)
    assert first["end_update"] == 3 and first["examples_seen"] == 3
    resume_state(model, optimizer, tmp_path / "resume", save=True)
    restored = Tiny()
    second_optimizer = optimizer_for(restored)
    resume_state(restored, second_optimizer, tmp_path / "resume")
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    rows = [{**r, "policy_version": 1, "state_id": "new-" + r["state_id"]} for r in rows]
    a = train_round(model, optimizer, rows, tmp_path / "a", version=1, start_update=3)
    b = train_round(restored, second_optimizer, rows, tmp_path / "b", version=1, start_update=3)
    assert a == b
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
