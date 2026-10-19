# Attribution

GAD: Tianzhu Ye, Li Dong, Zewen Chi, Xun Wu, Shaohan Huang, Furu Wei,
*Black-Box On-Policy Distillation of Large Language Models* (2025),
<https://arxiv.org/abs/2511.10643>.

Read and pinned before implementation:

- microsoft/LMOps `6f53ef41c951de0962bbf27fe3f5819eabc8ce0e`, `gad/README.md`
  and the GPT-5-Chat warmup/adversarial launchers (MIT).
- YTianZHU/verl `707757b9202c466d986e529a4658434d0ac6a7ee`,
  `verl/trainer/ppo/core_algos.py`, `ray_trainer.py`,
  `verl/workers/actor/dp_actor.py`, `critic/dp_critic.py`, and the PPO config
  (Apache-2.0). See `sources.json` for content hashes.

The small equation port in `objective.py` derives GRPO advantage normalization,
dual-clipped PPO, low-variance KL and Bradley–Terry from these sources.
The upstream copyright notices are retained there. The license is reproduced
in `LICENSE.verl`. This repository's H0 data path, serial stage driver and
Gemma scalar-head integration are independently implemented. No upstream
training dataset, model checkpoint, infrastructure tree or launch environment
is imported. Source inspection does not authorize any checkpoint used in the
upstream example scripts.

Shared local infrastructure is reused without changes: common pipeline,
exploration, scheduling, FSDP model loading/sharding/export, evaluation, and
HAD's durable text draw cache, CPU allocation guard and DCP state helper.
