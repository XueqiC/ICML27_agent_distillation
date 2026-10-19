# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Modified for the paper-1 GAD H0/FSDP2 port. Licensed under Apache-2.0;
# see LICENSE.verl and NOTICE.md. Distributed without warranties.
"""Small equation port of verl-gad@707757b (Apache-2.0).

Sources: verl/trainer/ppo/core_algos.py (GRPO, policy loss, low_var_kl,
Bradley-Terry), workers/critic/dp_critic.py (last response token).
No verl import or checkpoint from its example launchers is used.
"""
import torch
from torch import nn
from torch.nn import functional as F


def group_advantages(rewards):
    if rewards.ndim != 2 or rewards.shape[1] != 8 or not torch.isfinite(rewards).all():
        raise ValueError("Expected finite rewards for complete groups of eight")
    rewards = rewards.detach()
    return (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-6)


def grpo_terms(logp, old_logp, reference_logp, advantage):
    """Per-token loss; caller applies the exact global response-token mean."""
    ratio = (logp - old_logp.detach()).clamp(-20, 20).exp()
    advantage = advantage.detach()
    clipped = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.2))
    policy = torch.where(advantage < 0, torch.minimum(-advantage * 3.0, clipped), clipped)
    delta = (reference_logp.detach() - logp).clamp(-20, 20)
    kl = (delta.exp() - delta - 1).clamp(-10, 10)
    return policy + 0.001 * kl


def bradley_terry(teacher, student):
    return -F.logsigmoid(teacher - student)


class Discriminator(nn.Module):
    """Independent generator copy with a new scalar head, not a reward checkpoint."""
    def __init__(self, generator, hidden_size):
        super().__init__()
        self.generator = generator
        self.score = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.score.weight, mean=0.0, std=0.02)
        # Unused LM head stays out of Adam. The text backbone is fully trained.
        for name, parameter in generator.named_parameters():
            if name.startswith("lm_head."):
                parameter.requires_grad_(False)

    @property
    def config(self):
        return self.generator.config

    def forward(self, input_ids, attention_mask):
        # Call the multimodal wrapper's base model, retaining its text plumbing
        # but avoiding vocabulary logits and all-layer hidden-state retention.
        hidden = self.generator.model(input_ids=input_ids, attention_mask=attention_mask,
                                      use_cache=False).last_hidden_state
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        last = positions.masked_fill(~attention_mask.bool(), -1).max(-1).values
        if (last < 0).any():
            raise ValueError("Discriminator cannot score an empty sequence")
        return self.score(hidden[torch.arange(len(last), device=last.device), last]).squeeze(-1)


def token_logps(model, row, device, *, temperature=0.8):
    """Score exactly the generated IDs, including a sampled EOS, never an invented EOS."""
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    prompt = row["prompt_tokens"]
    if not 0 < prompt < ids.shape[1] <= 32768:
        raise ValueError("Invalid complete prompt/response")
    positions = torch.arange(prompt - 1, ids.shape[1] - 1, device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False, logits_to_keep=positions).logits[0]
    target = ids[0, prompt:]
    return torch.cat([-F.cross_entropy(logits[i:i + 64].float() / temperature,
                                      target[i:i + 64], reduction="none")
                      for i in range(0, len(target), 64)])


def score(model, row, device):
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    return model(ids, torch.ones_like(ids))[0].float()


def optimizer_for(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False, fused=False)


def optimizer_step(model, optimizer):
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.2)
    if hasattr(norm, "full_tensor"):
        norm = norm.full_tensor()
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite GAD gradient; no update/checkpoint published")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)
