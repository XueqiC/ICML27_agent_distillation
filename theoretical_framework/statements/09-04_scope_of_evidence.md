# Remark 9.4 · What the evidence covers: controlled checks and real-model tests

*Statement*

[← Framework](../README.md) · [Statement](09-04_scope_of_evidence.md) · [中文](09-04_scope_of_evidence_ZH.md)

**In one paragraph.** The framework has two kinds of evidence. Most checks are exact finite computations or run in controlled small networks; they test that each statement is correct under its assumptions. A smaller set are real-model tests on an actual agent-distillation recipe, each pre-registered with its prediction and criterion before the run it predicts; they test whether a statement's assumptions and predictions hold for that recipe. This remark lists every real-model test, what it tests, what was pre-registered and what came out, and says plainly which claims the evidence supports and which it does not. So far no real-model test supports a prediction of the framework: one pre-registered criterion failed because it was mis-specified, one restricted-model prediction had the wrong sign, one ranking holds only by a diagnostic radius, and the remaining tests are pending. It has no proof and no experiments page of its own; the numbers are on the experiments pages it links.

**Setting used on this page.**

- *Recipe.* ALFWorld ([Shridhar et al., 2021](#ref-shridhar2021alfworld)); supervised fine-tuning on $`K=20`$ trusted demonstrations of a LoRA student ([Hu et al., 2022](#ref-hu2022lora); Gemma-4-12B-it, rank 16) trained by AdamW ([Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)) at constant learning rate with global gradient-norm clipping at 1.0. The reference protocol continues this training to a fixed token budget, called p40 below; p03 and p10 denote earlier checkpoints of the same run.
- *Saved state.* $`z_{10}`$: the complete learner state at p10 (update 115; adapter, AdamW moments $`m,v`$, step counter and random state), from which every continuation test restarts.
- *Surrogate and evaluation.* $`\tilde F`$: a held-out teacher-forced likelihood score on 40 successful teacher trajectories (geometric-mean token probability, with the per-token log-likelihood as a second metric), used only for scoring. Task evaluation is on a held-out development set of 300 tasks; a sealed test split is never used.
- *Noise reference.* The same adapter evaluated on two identically configured servers solved 241 and 242 of the 300 tasks, with 15 tasks flipping between them; differences of this size are treated as server noise ([Check 8.21](../experiments/08-14_anchor_reuse_drift_certificate.md#c8-21)). One server pair on one checkpoint does not show that this noise is irreducible or independent across runs.
- *CRN paired-replacement score.* $`\hat S^{jk}`$: from a saved state, replace the next reference update by one batch of candidate $`j`$ (or $`k`$), run $`T`$ reference updates in an order set by a shared seed, score with $`\tilde F`$, and average the paired differences over $`r`$ replicas ([Definition 8.6](08-06_crn_paired_replacement_score.md#s8-6)). Its *strict* radius uses the a-priori range of the score; its *diagnostic* radius uses the range observed in the sample.

<a id="s9-4"></a>

**Remark 9.4 (What the evidence covers: controlled checks and real-model tests).**

- **Controlled evidence.** [Proposition 9.1](09-01_coupling_width.md#s9-1) is checked by exact finite computations and in a controlled small network; the selection rules of [Remark 9.3](09-03_derivation_design_rules.md#s9-3) are probed on frozen held-out panels of that network ([Check 9.4](../experiments/09-03_derivation_design_rules.md#c9-4), [Check 9.5](../experiments/09-03_derivation_design_rules.md#c9-5)). The checks of Theorems A, B and C, and of every other numbered result, are on the experiments pages of the statements they test; they are exact finite computations, controlled small networks or synthetic constructions. The results on the response operator, predictive closure, reuse, estimation cost, compilation and optimizer memory ([Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4), [Definition 4.8](04-08_response_operator.md#s4-8) to [Proposition 4.13](04-13_finite_horizon_gramians.md#s4-13), [Propositions 6.14–6.16](06-14_convex_realizability_legal_pool.md#s6-14), [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6), [Propositions 8.12–8.15](08-12_finite_catalog_spectral_certificate.md#s8-12) and [Corollary 8.16](08-16_composed_guarantee.md#s8-16)) are supported only by synthetic checks.
- **Real-model tests.** Each runs on the recipe above, with a single seed per configuration; predictions, criteria and compute and teacher budgets were written down before the runs they predict.
  - *R0a: the warm-state condition ([Remark 5.11](05-11_checking_warm_state_conditions.md#s5-11)).* Pre-registered: at six states from update 1 to p40, the sensitivity-weighted frozen-denominator error must be at most a tenth of the first-order term. Outcome: the criterion fails at all six states, but because it is mis-specified: its denominator multiplies absolute sensitivities by signed parameter changes and largely cancels. Against the signed first-order term, a comparison chosen after seeing the data and labelled post hoc, the ratio is about 30, 2 and 1 at updates 1, 4 and 8, and about 0.1 at p03, p10 and p40 ([Check 5.19](../experiments/05-11_checking_warm_state_conditions.md#c5-19)).
  - *R0b: non-additivity of the response.* Pre-registered: median response non-additivity above 0.2. Outcome: 0.19 at update 8 (not met, borderline) and 0.86 at p10 (met), with only 4 replicas per pair; at p10 the mean union effect is below the per-replica scatter, so the ratio is noisy ([Check 5.20](../experiments/05-11_checking_warm_state_conditions.md#c5-20)).
  - *R1: same-state ranking by the CRN certificate ([Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8)).* Four candidates from $`z_{10}`$: replay of the $`K`$ demonstrations (P1, the reference), teacher corrections at student states (P2), self-generated trajectories filtered by the environment reward (P3) and teacher rewrites of the demonstrations (P4). The ranking is recorded before any continued training, then each candidate's continued training is evaluated. Outcome so far: P1 against P3, with $`r=64`$ replicas and $`T=10`$ tail updates, scores $`+0.0014`$ (57 of 64 replicas positive). That exceeds the diagnostic radius $`0.0012`$, but the strict radius is $`0.325`$, so the ranking P1 above P3 holds by the diagnostic radius only, and the strict certificate is valid but vacuous ([Check 8.17](../experiments/08-08_crn_paired_replacement_certificate.md#c8-17)). The surrogate calibration is not yet estimated. P2 and P4 and the continued-training arms are pending.
  - *R2: learner-relative query allocation (Theorem A).* With teacher-label budgets $`n=100`$ and $`n=400`$, compare no queries (the existing reference run), uniform allocation over student states, water-filling with SGD weights $`W_s\propto d(s)^2\sigma_s^2`$ and learner-relative water-filling with the warm-Adam weight $`W_s\propto d(s)`$ ([Propositions 6.10](06-10_water_filling_allocation.md#s6-10) and [6.11](06-11_learner_weights_sgd_warm_adam.md#s6-11)). The scale-free form of the Adam weight was fixed before any query, after the pre-registered raw-variance form was found to contradict Proposition 6.11(b); the correction is recorded as such. Pre-registered primary criterion: at $`n=100`$, learner-relative minus uniform exceeds the noise reference and a paired McNemar test gives $`p<0.05`$. Also pre-registered: the sign of the warm-Adam minus SGD difference at $`n=400`$ matches the prediction, and across the arms the predicted error and the number of solved tasks have Spearman correlation at most $`-0.6`$. Outcome: pending.
  - *R3: the restricted amplitude model ([Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2)).* At $`z_{10}`$, one reference batch is scaled by $`c=0.5,1,2,4,8`$ with clipping off for that step and, separately, on. Pre-registered prediction: the response curve from Corollary 7.2(a) with the saved optimizer state. Outcome: the measured per-token responses are between $`+0.0002`$ and $`+0.0012`$, all inside their diagnostic radii, while the predictions are between $`-0.019`$ and $`-0.080`$: opposite sign and 28 to 339 times larger in magnitude ([Check 7.6](../experiments/07-02_stationary_amplitude_law.md#c7-6)). With clipping on, the two multipliers whose clipped gradients coincide give equal means, as the clipping cap of [Remark 7.4](07-04_relation_prior_work_role_clipping.md#s7-4) predicts. The most likely failed premise is linear response (tail gradients independent of the parameters); this is a hypothesis, not a finding.
  - *T1: optimizer-state transplant ([Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6)).* At $`z_{10}`$, the first 100 updates of the reference and of one alternative feedback are run; at the same parameters, the alternative's first moment, second moment or both are transplanted into the reference state, and every arm continues with the same reference updates. A duplicate reference arm measures numerical nondeterminism $`\delta_{\rm num}`$; the threshold is $`\tau=\max(3\delta_{\rm num},0.0012)`$ on the surrogate. Pre-registered, not yet run. A negative result will be stated as: at this checkpoint, with this feedback block and the fixed reference tail, the net effect of transplanting both moments did not exceed the threshold. The net effect is the sum of the momentum effect, the second-moment effect and their interaction, so a small net effect can come from cancellation and does not show that each channel is small; channel interpretations beyond the pre-registration are exploratory. The test measures a finite intervention effect, not the adjoint prediction of [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1), and a positive result is not by itself evidence of a deployment gain.
  - Baselines are run only from complete official code.
- **What the evidence supports and what it does not.**

| Evidence | Supports | Does not support |
|---|---|---|
| Controlled checks of every numbered result | Each statement is correct under its stated assumptions, in exact or controlled settings | Any claim about a real model |
| R0a | The warm-state measurement must use the signed first-order term; the pre-registered criterion was mis-specified, and the post-hoc ratio is about 0.1 from p03 on | That the pre-registered criterion passed, or that a post-hoc metric replaces it |
| R0b | Response non-additivity is measurable and present at p10 | A precise estimate (4 replicas) or a pass at update 8 |
| R1 so far | A diagnostic ranking of P1 above P3 on the surrogate | A certified ranking, a task-level gain, or a calibrated surrogate |
| R3 | The restricted amplitude model's prediction fails on the real model at p10 | A quadratic response law, an optimal amplitude or a sign change near twice that amplitude on the real model; a single identified cause of the failure |
| R2, T1 | Nothing yet | Anything until they report |

**What it says.** All evidence for the framework's statements is exact, controlled or synthetic. Of the real-model tests, R0a, R0b, R1 (partly) and R3 have reported: none supports a prediction, R0a's criterion was mis-specified and is reported as failed, R3's prediction had the wrong sign, and R1's ranking holds only by a diagnostic radius. R2 and T1 are pending.

**Role in the framework.**

- It marks the boundary of the claims: nothing in the framework is claimed for real models beyond what these tests report, and negative outcomes are kept, not replaced by post-hoc metrics.
- It connects each real-model test to the statement it tests, so each result can be read directly against a prediction written in advance.

**Uses.**

- [Remark 5.11](05-11_checking_warm_state_conditions.md#s5-11): warm-state checks on a real model (R0a, R0b).
- [Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8), [Remark 8.11](08-11_surrogate_calibration_and_transfer.md#s8-11), [Remark 8.17](08-17_open_problems.md#s8-17): the certificate, its calibration and its open problems (R1).
- [Propositions 6.10](06-10_water_filling_allocation.md#s6-10) and [6.11](06-11_learner_weights_sgd_warm_adam.md#s6-11), [Corollary 6.12](06-12_learner_separations.md#s6-12): allocation and learner weights (R2).
- [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2), [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3), [Remark 7.4](07-04_relation_prior_work_role_clipping.md#s7-4): the restricted amplitude model, ranker inconsistency, clipping (R3).
- [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6): optimizer memory (T1).

**Used by.**

- [Remark 7.4](07-04_relation_prior_work_role_clipping.md#s7-4): the real-model clipping test.
- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2) and [Remark 9.3](09-03_derivation_design_rules.md#s9-3): their real-model tests.
- [Remark 9.5](09-05_assumption_audit.md#s9-5): which assumptions the real-model tests bear on.

**Pre-registration.** The configurations, predictions, criteria and budgets of R0–R3 are in [the R0–R3 pre-registration](../../docs/direction/2026-10-12_real_model_R0-R3_prereg_ZH.md), including the R2 weight correction in its appendix B; those of T1 are in [the T1 pre-registration](../../docs/direction/2026-10-14_T1_optimizer_state_transplant_prereg_ZH.md). Both are in Chinese.

## Related work

- R1 compares the CRN ranking with candidate-generation schemes from imitation and distillation; teacher corrections at the student's own states are the queries of DAgger ([Ross et al., 2011](#ref-ross2011reduction)).

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
