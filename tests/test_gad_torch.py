"""CPU equation parity against pinned upstream and tiny native Gemma updates."""
import ast
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")

from baselines.gad.objective import (Discriminator, bradley_terry, group_advantages,
    grpo_terms, optimizer_for, score, token_logps)
from baselines.gad.worker import update
from baselines.had.training import resume_state
from test_stage2b_torch import tiny_model


def official():
    """Execute just the pinned arithmetic, without importing Ray/verl/network code."""
    path = Path("third_party/verl-gad/verl/trainer/ppo/core_algos.py")
    wanted = {"compute_grpo_outcome_advantage", "agg_loss", "compute_policy_loss",
              "kl_penalty", "compute_discriminator_loss"}
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            nodes.append(node)
    assert len(nodes) == len(wanted)
    context = {"torch": torch, "nn": torch.nn, "defaultdict": defaultdict,
        "np": __import__("numpy"), "verl_F": NS(masked_mean=lambda x, mask, **kw: (x * mask).sum() / mask.sum())}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), context)
    return context


def test_grpo_advantages_match_official_unbiased_std_and_constant_group():
    upstream = official()
    rewards = torch.tensor([[1., 2., 3., -1., 6., 7., 8., -2.], [4.] * 8])
    mask = torch.ones(16, 3)
    token_rewards = torch.zeros_like(mask)
    token_rewards[:, -1] = rewards.flatten()
    expected, _ = upstream["compute_grpo_outcome_advantage"](token_rewards, mask,
                    __import__("numpy").repeat([0, 1], 8))
    actual = group_advantages(rewards)
    torch.testing.assert_close(actual.flatten(), expected[:, 0])
    assert torch.equal(actual[1], torch.zeros(8)) and not actual.requires_grad
    with pytest.raises(ValueError, match="groups of eight"):
        group_advantages(torch.ones(3, 7))


def test_clipped_grpo_kl_loss_and_gradients_match_official():
    upstream = official()
    old = torch.tensor([[-4., -3., -2., -3., -4., -4., -1., -2.]])
    logp = torch.tensor([[-2., -3.5, -1., -6., -30., -1., -2., -1.]], requires_grad=True)
    ref = torch.tensor([[-1., -6., -3., -1., -1., -30., -3., -4.]])
    advantage = torch.tensor([[1., -1., 2., -2., 3., -3., -4., 4.]])
    mask = torch.tensor([[1., 1., 1., 1., 1., 1., 0., 0.]])
    pg, *_ = upstream["compute_policy_loss"](old, logp, advantage, mask, cliprange=.2)
    kl = upstream["kl_penalty"](logp, ref, "low_var_kl")
    expected = pg + .001 * (kl * mask).sum() / mask.sum()
    actual = (grpo_terms(logp, old, ref, advantage) * mask).sum() / mask.sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logp, retain_graph=True)[0],
                               torch.autograd.grad(expected, logp)[0])
    teacher, student = torch.tensor([1., -3.]), torch.tensor([-.1, 4.])
    torch.testing.assert_close(bradley_terry(teacher, student).mean(),
        upstream["compute_discriminator_loss"](student[:, None], teacher[:, None], torch.ones(2, 1), torch.ones(2, 1)))


def groups():
    teacher = {"input_ids": [2, 4, 7, 106], "prompt_tokens": 2}
    return [{"teacher": teacher, "samples": [
        {"input_ids": [2, 4, 10 + i] + ([5, 106] if i % 2 else []), "prompt_tokens": 2}
        for i in range(8)]}]


def test_tiny_native_discriminator_last_token_and_trainable_backbone():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    generator = tiny_model()
    discriminator = Discriminator(deepcopy(generator), 32)
    ids = torch.tensor([[2, 4, 7, 106, 0], [0, 2, 4, 5, 6]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]])
    hidden = discriminator.generator.model(input_ids=ids, attention_mask=mask).last_hidden_state
    expected = discriminator.score(hidden[torch.arange(2), torch.tensor([3, 4])]).squeeze(-1)
    actual = discriminator(ids, mask)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert discriminator.score.weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in discriminator.generator.model.parameters())
    assert all(p.grad is None for p in generator.parameters())


def test_exact_causal_temperature_scores_and_no_unsampled_eos():
    torch.set_num_threads(1)
    model = tiny_model()
    row = groups()[0]["samples"][0]
    ids = torch.tensor([row["input_ids"]])
    logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float() / .8
    full = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    torch.testing.assert_close(token_logps(model, row, "cpu"), full[0, row["prompt_tokens"] - 1:])
    assert token_logps(model, row, "cpu").numel() == 1  # This capped sample has no EOS.


def test_ce_bt_joint_update_preupdate_reward_and_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    actor = tiny_model()
    critic = Discriminator(deepcopy(actor), 32)
    gopt, dopt = optimizer_for(actor), optimizer_for(critic)
    batch = groups()
    with torch.no_grad():
        initial_rewards = [float(score(critic, sample, "cpu")) for sample in batch[0]["samples"]]
    initial_head = critic.score.weight.detach().clone()
    d = update(critic, dopt, batch, role="critic", kind="discriminator", device="cpu")
    assert d["rewards"][0] == pytest.approx(initial_rewards)
    assert not torch.equal(critic.score.weight, initial_head)
    g = update(actor, gopt, batch, role="actor", kind="ce", device="cpu")
    assert g["examples_seen"] == 1 and g["supervised_tokens_seen"] == 2
    reference = deepcopy(actor).eval()
    reference.requires_grad_(False)
    frozen = {k: v.clone() for k, v in reference.state_dict().items()}
    resume_state(actor, gopt, tmp_path / "actor", save=True)
    restored = tiny_model()
    ropt = optimizer_for(restored)
    resume_state(restored, ropt, tmp_path / "actor")
    assert set(ropt.state_dict()["state"]) == set(gopt.state_dict()["state"])
    d = update(critic, dopt, batch, role="critic", kind="grpo", device="cpu")
    a = update(actor, gopt, batch, role="actor", kind="grpo", device="cpu", reference=reference, rewards=d["rewards"])
    b = update(restored, ropt, batch, role="actor", kind="grpo", device="cpu", reference=reference, rewards=d["rewards"])
    assert a["loss"] == pytest.approx(b["loss"])
    assert a["examples_seen"] == 8 and a["supervised_tokens_seen"] == 16
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    for key, value in reference.state_dict().items():
        assert torch.equal(value, frozen[key])
    assert all(p.grad is None for p in reference.parameters())
    resume_state(critic, dopt, tmp_path / "critic", save=True)
    copy_d = Discriminator(tiny_model(), 32)
    copy_opt = optimizer_for(copy_d)
    resume_state(copy_d, copy_opt, tmp_path / "critic")
    torch.testing.assert_close(critic.score.weight, copy_d.score.weight, atol=0, rtol=0)


def test_actor_gradient_matches_dense_global_token_mean():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    actual, expected = tiny_model(), None
    expected = deepcopy(actual)
    ref = deepcopy(actual).eval().requires_grad_(False)
    batch = groups()
    rewards = [[-1., 3., 7., 2., 0., 4., 1., 9.]]
    adv = group_advantages(torch.tensor(rewards))[0]
    rows = batch[0]["samples"]
    denominator = sum(len(row["input_ids"]) - row["prompt_tokens"] for row in rows)
    loss = 0
    for i, row in enumerate(rows):
        lp = token_logps(expected, row, "cpu")
        with torch.no_grad():
            rp = token_logps(ref, row, "cpu")
        loss = loss + grpo_terms(lp, lp.detach(), rp, adv[i]).sum() / denominator
    loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), .2)
    expected_opt = torch.optim.SGD(expected.parameters(), lr=.01)
    expected_opt.step()
    actual_opt = torch.optim.SGD(actual.parameters(), lr=.01)
    update(actual, actual_opt, batch, role="actor", kind="grpo", device="cpu", reference=ref, rewards=rewards)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], atol=1e-7, rtol=1e-6)
