# Remark 9.4 · What the controlled evidence covers, and the planned real-model tests

*Statement*

[← Framework](../README.md) · [Statement](09-04_scope_of_evidence.md) · [中文](09-04_scope_of_evidence_ZH.md)

**In one paragraph.** Every check in this framework is an exact finite computation or runs in a controlled small network. This remark states that scope plainly and lists the real-model tests that are designed to probe the three main results on an actual agent-distillation recipe: Theorem A (how much information a learner needs, and where to acquire it: [Theorems 6.3](06-03_learner_relative_lower_bound.md#s6-3) and [6.5](06-05_achievability.md#s6-5)), Theorem B (Adam's second-moment channel: [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1) and its corollaries) and Theorem C (the CRN paired-replacement certificate: [Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8)). It has no proof and no experiments page of its own.

**Setting used on this page.**

- *Recipe.* ALFWorld ([Shridhar et al., 2021](#ref-shridhar2021alfworld)); the main-table recipe: supervised fine-tuning on $`K=20`$ trusted demonstrations, a LoRA student ([Hu et al., 2022](#ref-hu2022lora); Gemma-4-12B-it, rank 16) trained by AdamW ([Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)) at constant learning rate with global gradient-norm clipping at 1.0.
- *Saved state.* $`z_{10}`$: a complete learner state saved partway through training (adapter, AdamW moments $`m,v`$, step counter and random state), from which every test restarts.
- *Surrogate and evaluation.* $`\tilde F`$: mean teacher-forced log-likelihood of held-out successful teacher trajectories, used only for scoring. Task evaluation is on a held-out development set; a sealed test split is never used. A noise reference (the same adapter evaluated on a second server) sets the size of differences that count.
- *CRN paired-replacement score.* $`\hat S^{jk}`$: from $`z_{10}`$, replace the next reference update by one batch of candidate $`j`$ (or $`k`$), run $`T`$ reference updates in an order set by a shared seed, score with $`\tilde F`$, and average the paired differences over $`r`$ replicas ([Definition 8.6](08-06_crn_paired_replacement_score.md#s8-6)).
- *Per-coordinate quantities.* $`\lambda`$: a message's magnitude relative to the running RMS $`\sqrt v`$; $`\rho`$: the signal-to-noise ratio read from the optimizer state; $`\lambda^\star`$: the value peak of [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2).
- *Learner weights.* $`W_s`$: the learner-relative weight of a state cluster $`s`$; $`d(s)`$ its student visitation rate; $`k_s`$ its teacher coverage; $`\sigma_s^2`$ the second moment of single-sample gradients in the cluster; in the water-filling rule $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ the symbol $`\lambda>0`$ is the multiplier fixed by the query budget, not the magnitude ratio ([Propositions 6.10](06-10_water_filling_allocation.md#s6-10)–[6.11](06-11_learner_weights_sgd_warm_adam.md#s6-11)).

<a id="s9-4"></a>

**Remark 9.4 (What the controlled evidence covers, and the planned real-model tests).**

- **Covered.** [Proposition 9.1](09-01_coupling_width.md#s9-1) is checked by exact finite computations and in a controlled small network; the selection rules of [Remark 9.3](09-03_derivation_design_rules.md#s9-3) are probed on frozen held-out panels of that network ([Check 9.4](../experiments/09-03_derivation_design_rules.md#c9-4), [Check 9.5](../experiments/09-03_derivation_design_rules.md#c9-5)). The checks of Theorems A, B and C are on the experiments pages of the statements they test. All of them are exact finite computations or controlled small networks.
- **Not covered.** No statement of the framework is supported by a real-model or benchmark result. The real-model tests below are designed and pre-registered; no result from them is reported on these pages.
- **Planned real-model tests** (ALFWorld, the recipe above, from the saved state $`z_{10}`$). Each configuration runs with a single seed during exploration; predictions, criteria and compute and teacher budgets are written down before the runs they predict.
  - *R0 (re-analysis of the existing pilot).* At the pilot's probe states, report the sensitivity-weighted frozen-denominator error and its largest term against the first-order term ([Remark 5.11](05-11_checking_warm_state_conditions.md#s5-11)), with a pre-registered threshold below which the warm-Adam description is said to hold. Measure the non-additivity of the **response**, by CRN tails on disjoint batch pairs, alongside the non-additivity of the update map already measured.
  - *R1 (Theorem C: same-state, same-budget ranking).* From $`z_{10}`$, compare four candidate feedback laws: replay of the $`K`$ demonstrations (the reference), teacher corrections at student states (queries, as in DAgger: [Ross et al., 2011](#ref-ross2011reduction)), self-generated trajectories filtered by the environment reward, and teacher rewrites of the demonstrations. Rank them by the CRN paired-replacement score ([Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8)), record the ranking before any continued training, then continue training with each candidate and evaluate.
    - Criteria: rank agreement (Kendall $`\tau`$) of the certified ordering with the realized ordering, and agreement in sign of every certified pair whose realized difference exceeds the noise reference.
    - The calibration $`(\alpha,\varepsilon_{\rm sur})`$ is estimated from the full-tail outcomes ([Remark 8.11](08-11_surrogate_calibration_and_transfer.md#s8-11)); with three candidate pairs its transfer cannot be tested here, which remains the open problem of [Remark 8.12](08-12_open_problems.md#s8-12).
    - Comparison rankers, implemented from their definitions and not presented as baselines: a cosine score in the style of LESS ([Xia et al., 2024](#ref-xia2024less)), the gradient norm, the parameter-only first-order score, and static proxies.
  - *R2 (Theorem A: query allocation).* With a fixed query budget, compare no queries, uniform allocation over student states, water-filling with SGD weights $`W_s\propto d(s)^2\sigma_s^2`$, and learner-relative water-filling with warm-Adam weights $`W_s\propto d(s)\sigma_s^2`$, $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ ([Proposition 6.10](06-10_water_filling_allocation.md#s6-10)). The closed-form weights of [Proposition 6.11](06-11_learner_weights_sgd_warm_adam.md#s6-11) serve as a computable allocation rule; because the real model shares parameters across states, its state-owned assumption does not hold, so the test is whether the rule is useful, not whether the closed form is exact. The predicted order and magnitudes are computed beforehand from the variance formula of Proposition 6.10; the comparison of the two water-filling arms tests the learner separation of [Corollary 6.12](06-12_learner_separations.md#s6-12).
  - *R3 (Theorem B: the second-moment channel).* At $`z_{10}`$, scale one reference batch by multipliers $`c`$, so that cosine scores are identical across $`c`$ and only the magnitude $`\lambda`$ changes, with clipping turned off for that step and, separately, kept on. The predicted response curve is computed beforehand from the mean-field formula of [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2)(a) with the saved optimizer state; with clipping on, the clipping cap of [Remark 7.4](07-04_relation_prior_work_role_clipping.md#s7-4) predicts no difference between multipliers whose clipped gradients coincide. Where the measured value falls with $`c`$, cosine and parameter-only scores misrank the corresponding pairs ([Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3)); the misranked pairs are counted.
  - Baselines are run only from complete official code.

**What it says.** All present evidence is exact or controlled; the real-model tests R0–R3 are pre-registered and target, in order, the warm-state conditions and response non-additivity, the certificate (Theorem C), query allocation (Theorem A) and the second-moment channel (Theorem B).

**Role in the framework.**

- It marks the boundary of the claims: nothing in the framework is claimed for real models until R0–R3 report.
- It connects each planned test to the statement it tests, so a real-model result can be read directly against a prediction written in advance.

**Uses.**

- [Remark 5.11](05-11_checking_warm_state_conditions.md#s5-11): warm-state checks on a real model (R0).
- [Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8), [Remarks 8.11](08-11_surrogate_calibration_and_transfer.md#s8-11)–[8.12](08-12_open_problems.md#s8-12): the certificate, its calibration and its open problems (R1).
- [Propositions 6.10](06-10_water_filling_allocation.md#s6-10)–[6.11](06-11_learner_weights_sgd_warm_adam.md#s6-11), [Corollary 6.12](06-12_learner_separations.md#s6-12): allocation and learner weights (R2).
- [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2), [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3), [Remark 7.4](07-04_relation_prior_work_role_clipping.md#s7-4): amplitude law, ranker inconsistency, clipping (R3).

**Used by.**

- [Remark 7.4](07-04_relation_prior_work_role_clipping.md#s7-4): the real-model clipping test.
- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2) and [Remark 9.3](09-03_derivation_design_rules.md#s9-3): their planned real-model tests.

**Pre-registration.** The full configurations, predictions, criteria and budgets are in [the pre-registration document](../../docs/direction/2026-10-12_real_model_R0-R3_prereg_ZH.md) (in Chinese).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-hu2022lora"></a>

**Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2022).** LoRA: Low-rank adaptation of large language models. *The Tenth International Conference on Learning Representations (ICLR)*. [openreview.net/forum?id=nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9); arXiv [2106.09685](https://arxiv.org/abs/2106.09685).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-ross2011reduction"></a>

**Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (2011).** A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 15:627–635. [proceedings.mlr.press/v15/ross11a](https://proceedings.mlr.press/v15/ross11a.html).

<a id="ref-shridhar2021alfworld"></a>

**Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht (2021).** ALFWorld: Aligning text and embodied environments for interactive learning. *The Ninth International Conference on Learning Representations (ICLR)*. arXiv [2010.03768](https://arxiv.org/abs/2010.03768).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
