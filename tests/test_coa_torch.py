"""CPU numerical checks against pinned upstream math and optimizer continuity."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")

from baselines.coa.rl import advantages, policy_loss, train_update, token_logps
from baselines.coa.training import resume_state
from test_had_torch import Tiny, row


def test_dapo_clipping_matches_pinned_official_function_and_gradient():
    torch.set_num_threads(1)
    path = Path("third_party/Agent_Foundation_Models/verl/verl/trainer/ppo/core_algos.py")
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"compute_policy_loss", "agg_loss"}]
    scope = {"torch": torch, "verl_F": NS(masked_mean=lambda value, mask: (value * mask).sum() / mask.sum())}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    old = torch.tensor([[-2., -2., -2., -2., -2.]])
    new = torch.tensor([[-4., -1.9, -1.5, -.5, 1.]], requires_grad=True)
    a = torch.tensor([[1., 1., -1., -1., -1.]])
    expected = scope["compute_policy_loss"](old, new, a, torch.ones_like(a), cliprange_low=.2,
                                            cliprange_high=.28, clip_ratio_c=10.)[0]
    actual = policy_loss(new, old, a)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, new, retain_graph=True)[0],
                               torch.autograd.grad(expected, new)[0])


def test_grpo_uses_sample_std_not_population_std():
    rewards = [0, 0, 0, 0, 0, 0, 0, 1]
    values = torch.tensor(rewards, dtype=torch.float32)
    torch.testing.assert_close(advantages(rewards), (values - values.mean()) / (values.std() + 1e-6))
    assert advantages([1] * 8).abs().sum() == 0


def groups():
    examples = []
    for i in range(8):
        r = row([1, 2, 3, 4 + i, 5], 2)
        r.update(packing="turn", label_provenance={"training_eligible": True})
        examples.append({"turns": [r] if i % 2 else [r, row([2, 1, 2, 6], 2)]})
    return [{"rewards": [0.] * 4 + [1.] * 4, "examples": examples}]


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=.1, foreach=False, fused=False)


def test_token_mean_gradient_and_update_matches_full_batch(tmp_path):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = Tiny()
    reference = deepcopy(model)
    g = groups()
    opt, ref_opt = optimizer(model), optimizer(reference)
    expected_loss = 0
    count = 0
    adv = advantages(g[0]["rewards"])
    for e, a in zip(g[0]["examples"], adv):
        for r in e["turns"]:
            logp = token_logps(reference, r, "cpu")
            expected_loss += policy_loss(logp, logp.detach(), a, reduction="sum")
            count += len(logp)
    expected_loss /= count
    expected_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    ref_opt.step()
    result = train_update(model, opt, g, tmp_path, update=3)
    assert result["examples_seen"] == 8 and result["supervised_tokens_seen"] == count
    assert result["loss"] == pytest.approx(float(expected_loss.detach()), abs=1e-6)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)


def test_rl_resume_restores_masters_and_adam_moments(tmp_path):
    torch.manual_seed(8)
    torch.set_num_threads(1)
    model = Tiny()
    opt = optimizer(model)
    train_update(model, opt, groups(), tmp_path / "first", update=3)
    resume_state(model, opt, tmp_path / "resume", save=True)
    restored = Tiny()
    restored_opt = optimizer(restored)
    resume_state(restored, restored_opt, tmp_path / "resume")
    a = train_update(model, opt, groups(), tmp_path / "a", update=4)
    b = train_update(restored, restored_opt, groups(), tmp_path / "b", update=4)
    assert a == b
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_zero_variance_cannot_silently_train(tmp_path):
    model, g = Tiny(), groups()
    g[0]["rewards"] = [0.] * 8
    with pytest.raises(ValueError, match="constant-reward"):
        train_update(model, optimizer(model), g, tmp_path, update=0)
