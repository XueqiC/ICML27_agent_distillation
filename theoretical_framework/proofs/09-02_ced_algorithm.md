# Algorithm 9.2 · Consequence-Calibrated Experience Distillation, CED

*Proof*

[← Framework](../README.md) · [Statement](../statements/09-02_ced_algorithm.md) · [Proof](09-02_ced_algorithm.md) · [Experiments](../experiments/09-02_ced_algorithm.md) · [中文](09-02_ced_algorithm_ZH.md)

**What is proved here.** An algorithm is not a theorem, so what this page establishes is twofold. First, for each of the six steps of CED, which earlier statements force that step: a method that skips it loses the guarantee those statements provide. Second, each of the guarantees (G1)–(G3) follows by applying one cited theorem to the output of one step, and (G4) records what is left when the hypotheses of those theorems are not met.

**Setting used on this page.**

- *Student and recipe.* A complete learner state $`z=(\theta,o)`$, parameters plus optimizer state ([Definition 1.3](../README.md#s1-3)); a declared recipe (update $`U_t`$, optimizer, horizon $`R`$, reference tail); declared deployment tasks with score $`F_E`$; $`L_E(S)`$ is the learning value of a provider policy $`S`$, and $`L_E(0)`$ that of always using the reference.
- *Evidence.* $`K`$ trusted demonstrations per task family and a budget of $`n`$ permitted experiments (teacher queries at states the student visits, environment branches, resets) ([Definition 1.2](../README.md#s1-2)). $`D`$ is the transcript of all evidence; $`\mathcal R_{\rm info}(D)`$ is its information regret ([Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)).
- *Candidate feedback laws.* $`P^0,\dots,P^M`$ emit messages $`Y\sim P^j_t(\cdot\mid z)`$; $`P^0`$ is the reference. A message produces the increment $`\Delta^j`$ of the complete state. The one-boundary response of a message is $`g_{t,z}(Y)`$ ([Definition 1.4](../README.md#s1-4)); $`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$, and the advantage over the reference is $`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$: the change in task value from replacing one round of the reference by candidate $`j`$ at $`z`$, followed by the reference tail.
- *Learner-relative weights and coverage.* At agent states $`s`$ the student visits, $`W_s`$ is the learner-relative weight and $`k_s=KH\,d_T(s)`$ the teacher coverage; $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ is the water-filling allocation, with the multiplier $`\lambda>0`$ fixed by $`\sum_sn_s^\star=n`$ ([Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10)).
- *Magnitude quantities.* Per coordinate, $`\lambda`$ (outside the water-filling formula) is a message's amplitude relative to the running RMS $`\sqrt v`$, $`\rho`$ the signal-to-noise ratio read from the optimizer state, and $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$ the value peak ([Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2)).
- *Paired-replacement score.* At an audited state, $`\hat S^{jk}`$ is the mean over $`r`$ independent replicas of the difference of surrogate outcomes $`\tilde F`$ after one round with $`j`$ and with $`k`$, followed by a shared surrogate tail with common random numbers ([Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6)). $`e^{jk}_{\rm stat}`$ is its empirical-Bernstein radius and $`(\alpha,\varepsilon_{\rm sur})`$ the surrogate calibration of [Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7).
- *Confidence level* $`\delta`$; *checkpoints* $`t_1<\dots<t_A`$.

**Statement ([Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2)).**

*Input:* a student $`z_0`$ with its declared recipe and deployment tasks; $`K`$ trusted demonstrations per family and a budget of $`n`$ permitted experiments; candidate generators for $`P^1,\dots,P^M`$ and the reference $`P^0`$; an audit budget (checkpoints, $`r`$ replicas, a surrogate $`\tilde F`$, a few full-tail calibration audits, confidence $`\delta`$). *Output:* the distilled student $`z_R`$; an evidence ledger; at each checkpoint the scores $`\hat S^{jk}`$ with radii, the selected law and whether the selection is certified; a frozen selector with separate method-transfer and policy-transfer reports.

*Procedure.*

1. **Record the complete learner state** $`z_t=(\theta_t,o_t)`$ at every checkpoint, including $`m`$, $`v`$, the step counter, the sampler and the random state, and record warmth ([Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11)).
2. **Acquire before compiling.** Estimate $`W_s`$ and $`k_s`$ by a pilot (closed forms of [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11) where coordinates are state-owned, grouped CRN scores otherwise); allocate the remaining queries by $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$, in general the c-optimal design with target $`c=\sum_tJ_t^*g_t`$; keep a ledger in which only new evidence counts.
3. **Compile candidates** $`P^0,\dots,P^M`$, each specified by the packets it actually emits.
4. **Record each candidate's learner statistic and control its magnitude:** its mean optimizer increment, clipping activation rate, and per-coordinate $`(\lambda,\rho)`$; flag mass on high-$`\rho`$ coordinates beyond $`\lambda^\star`$ or in the large-$`\lambda`$ regime; a rescaled or clipped variant is a new candidate.
5. **Measure consequence at the actual states:** compute $`\hat S^{jk}`$ and $`e^{jk}_{\rm stat}`$ for all pairs at $`z_t`$; calibrate $`(\alpha,\varepsilon_{\rm sur})`$ against a few full-tail paired audits; use first-order scores only as a screen on unflagged pairs whose margin exceeds the second-order remainder; fit any response model under the bounded-coefficient objective.
6. **Select, train, re-audit, freeze, test transfer:** select $`j_t\in\arg\max_j[\hat S^{j0}-e^{j0}_{\rm stat}-\varepsilon_{\rm sur}]`$, keeping the reference if no lower bound is positive; train with $`P^{j_t}`$ to the next checkpoint, extending the selection to unaudited states only with transport terms or a re-audit; freeze everything before the evaluation data and report method transfer and policy transfer separately.

*Guarantees.* (G1) Under Assumption 8.7, with probability at least $`1-\delta`$, simultaneously for all pairs at $`z_t`$: a selection $`j_t\ne0`$ with positive lower bound does not lower the task value of one round followed by the reference tail. (G2) With full-tail paired audits at the actual states and lower-confidence-bound selection, $`L_E({\rm adaptive})-L_E(0)\ge-\delta`$. (G3) No post-processing step lowers the information regret; under the assumptions of [Theorem 6.5](../statements/06-05_achievability.md#s6-5), step 2 attains the learner-relative information bound up to the constants stated there. (G4) Otherwise the claim is empirical. *Known limitation:* with the a-priori range of Assumption 8.7 the radius can exceed typical candidate gaps ([Remark 8.12](../statements/08-12_open_problems.md#s8-12)), and step 6 then falls under (G4).

**Proof.** The argument has two parts: why each step is required, and how each guarantee follows.

*Part 1: why each step is required.*

- **Step 1 (complete state).** The statistic a feedback law must carry depends on the optimizer state, not on $`\theta`$ alone:
  - at a cold Adam step the update is a sign quotient and magnitudes are retained only in $`m`$ ([Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3));
  - warm Adam is first-order affine in the consumed gradient with coefficients set by $`(m,v)`$, while cold steps separate equal-mean feedback ([Propositions 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6) and [5.7](../statements/05-07_cold_steps_separate_equal_mean.md#s5-7));
  - the first-order statistic is learner-specific ([Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)).
  
  Coverage and transport are statements about complete states ([Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2): the transport bound is indexed by the complete state $`z\in\mathcal C_t`$). A checkpoint that keeps only $`\theta`$ cannot be audited.
- **Step 2 (acquire before compiling).**
  - Post-processing cannot reduce information regret ([Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4): $`\mathcal R_{\rm info}(C(D))\ge\mathcal R_{\rm info}(D)`$ for every randomized map $`C`$); under aliasing, replay has zero value ([Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7): two environments with the same demonstration law and different learner-relative functionals cannot be told apart by any distiller).
  - Passive data can leave a gap that only queries close ([Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6): worst-case regret at least $`\Delta`$ without queries, attainable with them).
  - Demonstrations and queries share one budget at a learner-relative exchange rate ([Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11), [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8)).
  - The minimal error is of order $`\mathfrak n_{K,n}`$ ([Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)) and is attained by the c-optimal design ([Theorem 6.5](../statements/06-05_achievability.md#s6-5)), which in the state-coverage model is water-filling ([Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10)).
  
  So queries must be acquired, and allocated by the learner's weights, before any compilation: no later step can recover what the transcript lacks.
- **Step 3 (packets).** Grouping changes the outcome exactly through the non-additive part of the joint-packet response ([Proposition 9.1](../statements/09-01_coupling_width.md#s9-1): the coupling width is zero if and only if the packet response is additive). A candidate is therefore its packet law, not a bag of units.
- **Step 4 (learner statistic and magnitude).**
  - Candidates with the same mean gradient can differ at first order ([Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)).
  - Under warm Adam a larger same-direction message can be worth less ([Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2): the value is concave in $`\lambda`$ with peak near $`\lambda^\star`$).
  - Parameter-only and cosine scores misrank or tie in that region ([Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes.md#s7-3)), and first-order scores are unreliable where the second-moment channel dominates ([Remark 7.5](../statements/07-05_where_first_order_rankers_fail.md#s7-5)).
  - Global-norm clipping caps $`\lambda`$ ([Remark 7.4](../statements/07-04_relation_prior_work_role_clipping.md#s7-4)), so a clipped variant is a genuinely different candidate.
  
  Flagged candidates are therefore ranked only by the finite measurement of step 5.
- **Step 5 (measure at the actual states).**
  - The paired-replacement score is exact in expectation, with no first-order approximation and no curvature constant ([Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8)(a)); under common random numbers its variance is governed by the difference of the two candidates' increments ([Proposition 8.10](../statements/08-10_cost_and_crn.md#s8-10)(c)).
  - The surrogate differs from the task objective, so $`(\alpha,\varepsilon_{\rm sur})`$ must be calibrated against full-tail task audits ([Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7)(A), [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3), [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11)).
  - A first-order score certifies a pair only when its first-order margin exceeds the second-order remainder ([Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9)), hence the restriction of the screen.
  - For a response model on features, the bounded-coefficient objective is the exact certified unresolved discrepancy ([Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1)), so a model with smaller residual but larger coefficients gives a worse certificate.
- **Step 6 (select, re-audit, freeze, transfer).**
  - Selecting by the lower bound is the decision rule of [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8)(c); selecting by point estimates carries no such guarantee.
  - Dropping the transport term is invalid: there are instances where an audit reports zero discrepancy while the learning gap is $`1`$ ([Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)(f)). Hence the transport terms or the re-audit for unaudited states.
  - The audit event of [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)(a) is simultaneous over a candidate set fixed before the audit, so everything must be frozen before the data that evaluates it.
  - Method transfer and policy transfer are different functionals with different test classes ([Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2)(c): transfer holds only through a verified decomposition of the target responses), so they are reported separately.

*Part 2: the guarantees.*

**(G1).** At checkpoint $`z_t`$, step 5 computes $`\hat S^{jk}`$ and $`e^{jk}_{\rm stat}`$ for all ordered pairs under the hypotheses of Assumption 8.7. By [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8)(b), with probability at least $`1-\delta`$, simultaneously for all pairs,

```math
L_F(P^j)-L_F(P^0)\ \ge\ \alpha\big(\hat S^{j0}-e^{j0}_{\rm stat}-\varepsilon_{\rm sur}\big).
```

Step 6 selects $`j_t\ne0`$ only when the bracket is positive, and $`\alpha>0`$, so on this event $`L_F(P^{j_t})-L_F(P^0)>0`$. By [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6), $`L_F(P^j)-L_F(P^0)=A_t(z_t,j)`$, the task-value change of replacing one round of the reference by $`P^j`$ at $`z_t`$ followed by the reference tail. This is (G1). Because the event is simultaneous over all pairs, it holds for the data-dependent choice $`j_t`$.

**(G2).** If the audits of step 5 are full-tail paired audits at the states training actually reaches and step 6 selects by their lower confidence bound (keeping the reference otherwise), the resulting provider policy is the adaptive policy of [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3)(c). Part (c)(ii) of that theorem gives, unconditionally over audits and training, $`L_E({\rm adaptive})-L_E(0)\ge-\delta`$. The mechanism is the signed identity of Theorem 8.3(b), $`L_E(S)-L_E(0)=\sum_t\mathbb E_{d^S_t}A_t(Z_t,J_t)`$: the whole-run gain is the sum of per-round advantages at the states actually reached, each of which the audit controls.

**(G3).** Steps 3–6 take the transcript $`D`$ produced by step 2 and transform it (replay, rewriting, packing, selection), so the student they produce is a randomized post-processing $`C(D)`$. By [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4), $`\mathcal R_{\rm info}(C(D))\ge\mathcal R_{\rm info}(D)`$: none of these steps lowers the information regret. The acquisition of step 2 is the c-optimal design with target $`c=\sum_tJ_t^*g_t`$, which under the assumptions of [Theorem 6.5](../statements/06-05_achievability.md#s6-5) attains the learner-relative information bound up to the constant factors stated there.

**(G4).** Each of the three cases listed in (G4) removes a hypothesis used above: point-estimate selection removes the lower-bound condition used in (G1)–(G2); an unverified calibration removes Assumption 8.7(A); use at unaudited states without transport terms removes the transport hypothesis of Theorem 8.2, and Theorem 8.2(f) shows that the conclusion can then fail. No certificate remains, and the claim is about the mean outcome of the frozen method over a declared population, supported by whole-run randomized comparisons.

**Known limitation.** Under Assumption 8.7(B) the range $`B`$ is a-priori ($`B\le2`$ for $`\tilde F\in[0,1]`$), and the range term $`7B\log(4M^2/\delta)/(3(r-1))`$ of $`e^{jk}_{\rm stat}`$ can exceed the gaps between nearby candidates for any affordable $`r`$ ([Remark 8.12](../statements/08-12_open_problems.md#s8-12)). Then no lower bound is positive, step 6 keeps the reference or reports an uncertified selection, and the run falls under (G4). $`\square`$

## Related work

- **Data and teacher selection.** LESS ([Xia et al., 2024](#ref-xia2024less)) ranks training data by gradient cosine similarity to a target; GRACE ([Panigrahi et al., 2025](#ref-panigrahi2025good)) scores teachers for distillation. Algorithm 9.2 instead ranks candidate feedback laws by their measured consequence at the actual optimizer state, and [Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes.md#s7-3) shows where cosine-type scores are inconsistent.
- **Active acquisition.** DAgger ([Ross et al., 2011](#ref-ross2011reduction)) queries the teacher at the student's states. Step 2 allocates such queries by learner-relative weights rather than uniformly ([Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10)).
- **Influence and metagradients.** TracIn ([Pruthi et al., 2020](#ref-pruthi2020estimating)), influence functions ([Koh & Liang, 2017](#ref-koh2017understanding)), metagradients ([Engstrom et al., 2025](#ref-engstrom2025optimizing)) and In-Run Data Shapley ([Wang et al., 2025](#ref-wang2025data)) are first-order or fixed-state attributions. In Algorithm 9.2 they can serve only as the screen of step 5, under [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-engstrom2025optimizing"></a>

**Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, Aleksander Madry (2025).** Optimizing ML training with metagradient descent. arXiv preprint. arXiv [2503.13751](https://arxiv.org/abs/2503.13751).

<a id="ref-koh2017understanding"></a>

**Pang Wei Koh, Percy Liang (2017).** Understanding black-box predictions via influence functions. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1885–1894. [proceedings.mlr.press/v70/koh17a](https://proceedings.mlr.press/v70/koh17a.html).

<a id="ref-panigrahi2025good"></a>

**Abhishek Panigrahi, Bingbin Liu, Sadhika Malladi, Sham Kakade, Surbhi Goel (2025).** In good GRACEs: Principled teacher selection for knowledge distillation. arXiv preprint. arXiv [2511.02833](https://arxiv.org/abs/2511.02833).

<a id="ref-pruthi2020estimating"></a>

**Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan (2020).** Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 19920–19930. arXiv [2002.08484](https://arxiv.org/abs/2002.08484).

<a id="ref-ross2011reduction"></a>

**Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (2011).** A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 15:627–635. [proceedings.mlr.press/v15/ross11a](https://proceedings.mlr.press/v15/ross11a.html).

<a id="ref-wang2025data"></a>

**Jiachen T. Wang, Prateek Mittal, Dawn Song, Ruoxi Jia (2025).** Data Shapley in one training run. *13th International Conference on Learning Representations (ICLR)*. arXiv [2406.11011](https://arxiv.org/abs/2406.11011).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
