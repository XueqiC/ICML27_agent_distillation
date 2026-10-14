# Remark 9.3 · Derivation of the design rules

*Proof*

[← Framework](../README.md) · [Statement](../statements/09-03_derivation_design_rules.md) · [Proof](09-03_derivation_design_rules.md) · [Experiments](../experiments/09-03_derivation_design_rules.md) · [中文](09-03_derivation_design_rules_ZH.md)

**What is derived here.** For each of the seven design rules behind [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2), the earlier statements that force it. Each derivation either applies a proved bound that the rule respects or exhibits, through an earlier statement, an instance where a method that breaks the rule loses the guarantee. The rules are necessary conditions for the stated guarantees, not a proof that CED beats any baseline.

**Setting used on this page.**

- *Learner and candidates.* A complete learner state $`z=(\theta,o)`$, parameters plus optimizer state ([Definition 1.3](../README.md#s1-3)); candidate feedback laws $`P^0,\dots,P^M`$ emitting messages $`Y\sim P^j_t(\cdot\mid z)`$, with $`P^0`$ the reference; a message produces the increment $`\Delta`$ of the complete state.
- *Responses.* $`g_{t,z}(Y)`$ is the one-boundary response ([Definition 1.4](../README.md#s1-4)); $`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$; $`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$ is the value of replacing one round of the reference by $`j`$ at $`z`$, followed by the reference tail; $`d^S_t`$ is the law of the state at round $`t`$ under protocol $`S`$.
- *Information.* $`D`$ is the transcript; $`\mathcal R_{\rm info}(D)`$ its information regret ([Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)); $`\mathfrak n_{K,n}`$ is the learner-relative information norm with $`K`$ demonstrations and $`n`$ queries, and $`c=\sum_tJ_t^*g_t`$ the learner-relative target vector: the gradient of the learner-relative functional, combining the Jacobian $`J_t`$ of the learner's mean increment with respect to the environment parameter and the task direction $`g_t`$ ([Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2)).
- *Mean increment.* $`\mu_t^P(z)=\mathbb E_P\Delta`$; warmth measures how far an Adam state is from its frozen-denominator description ([Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11)); $`p_{\rm act}`$ is the clipping activation rate and $`c`$ the clipping threshold in [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9).
- *Response models.* Features $`\phi_1,\dots,\phi_d`$ with matrix $`\Phi`$; a response model $`c+b^\top\phi`$ with constant $`c`$ and coefficients $`b`$; tolerances $`\epsilon_j`$; response estimate $`\widehat g`$ with $`\lVert\widehat g-g\rVert_\infty\le\eta`$; $`\mathrm{osc}(h)=\max h-\min h`$.
- *Notation.* As in the cited statements, the letter $`c`$ denotes the target vector in Rule 1, the clipping threshold in Rule 3 and the constant of a response model in Rules 4 and 6; the meaning is fixed by the rule; likewise $`r`$ is the residual of a response model in Rule 6 and, written $`r(\mathcal M,\mathcal K)`$, the information–compilation radius in Rule 7.
- *Transfer.* Method transfer evaluates $`L_{E'}(S)`$ for the frozen protocol $`S`$ rerun on a new family $`E'`$ with its own $`K`$ demonstrations; policy transfer evaluates $`J_{E'}(\pi_z)`$ for the distilled student $`z`$ with no further update.
- *Gaps (Rule 7).* A response interface $`\phi(y)\in\mathbb R^r`$ with coefficient matrix $`C`$ ([Definition 4.8](../statements/04-08_response_operator.md#s4-8)); a confidence set $`\mathcal M`$ for the target feature mean; the set $`\mathcal K`$ of response outputs reachable by mixtures of the finite legal pool; the information–compilation radius $`r(\mathcal M,\mathcal K)=\min_{q\in\mathcal K}\sup_{\mu\in\mathcal M}\lVert C\mu-q\rVert_\omega`$ ([Proposition 6.16](../statements/06-16_information_compilation_radius.md#s6-16)).

**Statement ([Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3)).** Algorithm 9.2 implements seven rules, each a necessary condition for the corresponding guarantee:

- **Rule 1 (Acquire before compiling).** Decision-relevant facts that the transcript does not determine can be supplied only by new permitted experiments; how much to acquire, and where, is set by the learner.
- **Rule 2 (Select by measured consequence).** On a declared family, a selection rule is consistent with the learning-value ordering only if it is a function of the mean responses $`\lbrace\bar g_t(z,j)\rbrace_j`$, or of a statistic proved sufficient for them on that family.
- **Rule 3 (Match the learner's statistic, not the raw gradient).** What a candidate must match, to first order, is the mean increment of the complete learner state in the directions of the continuation-value gradients; responses and their statistics are indexed by complete states and their warmth.
- **Rule 4 (Fit stable response models).** A response model $`c+b^\top\phi`$ on features measured to tolerances $`\epsilon_j`$ is fitted by minimizing $`\mathrm{osc}(\widehat g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$.
- **Rule 5 (Account for coverage; certify by paired replacement; use first-order audits only where the margin dominates).** A guarantee on the learning value of the selected candidate takes one of four forms: (i) the frozen-audit certificate with measured transport and exit probability; (ii) the actual-state lower-confidence-bound guarantee with full-tail paired audits; (iii) the CRN paired-replacement certificate at audited states with a calibrated surrogate, extended along training only by transport terms or by re-auditing; (iv) whole-run randomized comparisons of the frozen method, labelled empirical.
- **Rule 6 (Freeze, then test transfer).** Selectors, features, cells, surrogates and candidates are frozen before the data that evaluates them; transfer is guaranteed only through a verified decomposition of the target responses and is otherwise measured; method transfer and policy transfer are reported separately.
- **Rule 7 (Route each gap to its remedy).** Information gaps are closed by acquisition, realizability gaps by enlarging the legal pool, closure gaps by extending the tests or the recorded state; reuse across checkpoints and two-fidelity estimation are used only when their cost conditions are verified.

The rules do not by themselves show that an instance of CED outperforms a given baseline.

**Derivation.**

**Rule 1 (Acquire before compiling).**

- Let $`D'=C(D)`$ be any randomized post-processing of the transcript: replay, rewriting, paraphrase or synthetic descendants. Then $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$ ([Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4)). By the decomposition of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2) (regret = information term + algorithm term + teacher term), compilation can lower only the algorithm term.
- Passive data can leave a gap even when the teacher is optimal: in the two-world family the worst-case regret is at least $`\Delta`$ ([Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)(a)). More generally, if two environments have the same demonstration law and different learner-relative functionals, every distiller, replay included, has worst error at least half their difference, for every $`K`$ ([Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7)).
- The amount to acquire is bounded below by [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)(b) (active lower bound), [Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10) (few-shot information floor) and [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11) (one shared budget) in the explicit models, and by $`\tfrac14\mathfrak n_{K,n}`$ in general ([Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)). It is attained by the c-optimal design ([Theorem 6.5](../statements/06-05_achievability.md#s6-5)), which in the state-coverage model is water-filling ([Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10)).
- The target vector $`c=\sum_tJ_t^*g_t`$ depends on the optimizer, so the same demonstrations can be sufficient for one learner and insufficient for another ([Corollary 6.12](../statements/06-12_learner_separations.md#s6-12)). Acquisition must therefore be planned for the learner actually used.
- Compiling an acquired fact into the student has its own cost ([Proposition 2.7](../statements/02-07_compilation_cost.md#s2-7)), accounted separately from acquisition.

**Rule 2 (Select by measured consequence).**

- *The response is the invariant.* Two message laws are interchangeable for every one-boundary replacement followed by the declared tail if and only if they have equal means under every response of the declared class ([Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2)(b)). The value of replacing the reference by candidate $`j`$ at $`z`$ is exactly $`A_t(z,j)`$, and the total effect of a candidate policy is $`\sum_t\mathbb E_{d^S_t}A_t`$ ([Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3)(b)).
- *Static value and mean gradients do not identify the ordering.* [Proposition 3.1](../statements/03-01_static_value_not_learning_value.md#s3-1)(a) gives two sources that agree on the teacher value and on the static policy value of every aliased student, yet distill students of different value. A rule that depends only on these statistics makes the same choice for the instance and its relabelling, so it selects the worse source in one of them. [Proposition 3.1](../statements/03-01_static_value_not_learning_value.md#s3-1)(b) gives two packets with equal label marginals and equal mean gradient whose learning-value difference changes sign between two tasks; a rule depending only on these statistics is wrong for one of the tasks.
- *Two proxy classes are inconsistent.* Task-blind statistics cannot rank correctly for every task, and parameter-only first-order scores and cosine scores misrank or tie in the region described by [Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2) ([Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes.md#s7-3)). For other static proxies no inconsistency is proved; the static proxies tested in the controlled learner did not reliably select ([Check 9.4](../experiments/09-03_derivation_design_rules.md#c9-4)).
- *Packet organization is part of the candidate.* Regrouping units with fixed marginals changes the learning value exactly through the non-additive part of the packet response $`H_z`$ ([Proposition 9.1](../statements/09-01_coupling_width.md#s9-1)). A measured response must therefore be the response to the packets actually emitted, not a sum of per-unit scores.
- *No universal small statistic.* For unrestricted learners no fixed family of fewer than $`m-1`$ statistics determines all response means on an alphabet of $`m`$ messages ([Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2)(d)). Small sufficient statistics exist only for restricted families, for example joint moments up to order $`N`$ for affine learners ([Theorem 4.4](../statements/04-04_affine_learners_moment_order.md#s4-4)).

**Rule 3 (Match the learner's statistic, not the raw gradient).**

- The first-order representation expresses the outcome difference through the mean increment of the complete state ([Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1): $`(K_t^P-K_t^Q)V(z)=\langle\nabla V(z),\mu_t^P(z)-\mu_t^Q(z)\rangle+r_1`$). [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10) computes that statistic for each learner:
  - the mean gradient for SGD;
  - the clipped mean under clipping, and the mean direction only up to $`c(1-p_{\rm act})`$ ([Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9));
  - the sign law at a cold Adam step, with magnitudes retained in $`m`$ ([Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3));
  - the mean Adam increment at a warm state, which reduces to $`(\mathbb EG,\mathbb E\,G\odot G)`$ only under the frozen-denominator approximation ([Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5), [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6)).
- At a cold state, laws with equal means separate at first order ([Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean.md#s5-7)). With $`\beta_2`$ fixed, warmth levels off at $`(1-\beta_2)/\beta_2`$ instead of vanishing ([Proposition 5.8](../statements/05-08_evolution_warmth.md#s5-8)), so the frozen-denominator description keeps a per-step error floor and warmth must be measured, not assumed small.
- Under warm Adam, a message also acts through the second-moment channel ([Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1)); its value is concave in the magnitude relative to the running RMS and becomes negative beyond an explicit threshold in the stated regime ([Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2)). The magnitude relative to $`\sqrt v`$ is therefore part of the statistic.
- Two states with identical parameters but different optimizer moments emit different next states and can have different responses, so transport conditions are stated on complete states ([Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)).

**Rule 4 (Fit stable response models).**

- [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1)(a) identifies this objective as the exact certified unresolved discrepancy: the largest response difference between two laws whose feature means differ by at most $`\epsilon_j`$ equals $`\min_b[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert]`$. With a response estimate within $`\eta`$ in sup norm, the certified bound is its value at $`\widehat g`$ plus $`2\eta`$ (Theorem 8.1(d)). For each fixed $`b`$ the bound is the bracket evaluated at $`b`$, so the minimizing $`b`$ gives the best certificate.
- A model with a smaller residual but larger $`\sum_j\epsilon_j\lvert b_j\rvert`$ gives a strictly worse certificate whenever the sum is larger. A zero residual can coexist with a vacuous certificate when the features are small multiples of the response ([Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1)(c): with $`\phi_1=\alpha g`$, the discrepancy is $`\min\lbrace1,\epsilon/\alpha\rbrace`$).

**Rule 5 (Account for coverage; certify by paired replacement; use first-order audits only where the margin dominates).** The four forms are:

- (i) the frozen-audit certificate with measured transport and exit probability ([Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)(c));
- (ii) the actual-state lower-confidence-bound guarantee with full-tail paired audits ([Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3)(c));
- (iii) the CRN paired-replacement certificate at audited states with a calibrated surrogate ([Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8)), extended along training only by transport terms or by re-auditing ([Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11));
- (iv) whole-run randomized comparisons of the frozen method, labelled empirical.

Why these and only these:

- Dropping the transport term is invalid: an audit can report zero discrepancy while the learning gap is $`1`$ ([Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)(f)).
- Full-tail audits at the actual states guarantee a loss of at most $`\delta`$, but cost on the order of a thousand paired replicas per candidate ([Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3)(c)–(d): 1,016 replicas for $`M=4`$, $`\delta=0.05`$, radius $`0.1`$).
- The paired-replacement score is exact in expectation, needs no continuation-value gradient and no curvature, and its radius has only a statistical term and a calibration term ([Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8)). Under common random numbers its variance is governed by the difference of the two candidates' increments ([Proposition 8.10](../statements/08-10_cost_and_crn.md#s8-10)).
- A first-order score certifies a pair only if the pair's first-order term exceeds the second-order remainder bound. Pairs whose mean increments agree along the surrogate gradient are therefore never certified at first order, whatever their true gap ([Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9)); Theorem 8.8 can certify them.
- Point-estimate selection and transport-free extrapolation carry none of the guarantees (i)–(iii).

**Rule 6 (Freeze, then test transfer).**

*Freezing.* The audit event of [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)(a) is simultaneous over a candidate set fixed before the audit. Reusing audit samples to construct or train a candidate changes its law, and the certificate then needs a new independent audit. The same holds for any selector tuned on its evaluation data.

*Transfer.* If every target-family response satisfies $`g=c+b^\top\phi+r`$ with $`\lVert b\rVert_1\le B`$ and $`\lVert r\rVert_\infty\le\tau`$, then, by [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2)(c),

```math
\sup_g\lvert(P-Q)g\rvert\le B\,\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau .
```

Conversely, if some target response lies outside the span of the features, there are feature-matched laws that it separates ([Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2)(a)), and no audit on the training family alone certifies the target.

*Two kinds of transfer.* Method transfer evaluates $`L_{E'}(S)`$ for the frozen protocol $`S`$ applied to a new family $`E'`$ with its own $`K`$ demonstrations; its test class is the response class of $`E'`$, defined through updates. Policy transfer evaluates $`J_{E'}(\pi_z)`$ for the distilled student $`z`$ with no further update; its test functions are the deployment scores of $`E'`$. These are different functionals with different test classes, so a bound on one does not bound the other, and they are reported separately.

**Rule 7 (Route each gap to its remedy).** The four remedies are not interchangeable; each is forced by a statement that shows the others do not help.

*Information gaps call for acquisition, and only along the visible directions.* For a fixed interface, [Proposition 6.16](../statements/06-16_information_compilation_radius.md#s6-16) gives

```math
r(\mathcal M,\mathcal K)\ \ge\ \max\Big\lbrace\tfrac12\,\mathrm{diam}_\omega(C\mathcal M),\ \sup_{\mu\in\mathcal M}\mathrm{dist}_\omega(C\mu,\mathcal K)\Big\rbrace ,
```

with $`r(\mathcal M,\mathcal K)=0`$ only when $`C\mathcal M`$ is a single point inside $`\mathcal K`$. Two consequences follow. Shrinking $`\mathcal M`$ along $`\ker C`$ leaves $`C\mathcal M`$, and so both lower bounds, unchanged: queries that only resolve such directions cannot lower the error on any declared test, so acquisition is planned for the directions the tests see (Rule 1 gives where and how much). When the diameter term dominates, the transcript leaves the declared tests undetermined, and only new evidence can shrink it ([Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4)).

*Realizability gaps call for a larger legal pool.* A target feature mean is reached by a mixture of the pool exactly when it lies in the convex hull of the pool's feature vectors ([Proposition 6.14](../statements/06-14_convex_realizability_legal_pool.md#s6-14)). Acquiring more information about the same target shrinks $`\mathcal M`$ but does not move $`\mathcal K`$. On the confidence event the true target mean $`\mu_P`$ lies in every confidence set, so for every $`q\in\mathcal K`$ the worst case over $`\mathcal M`$ is at least $`\lVert C\mu_P-q\rVert_\omega\ge\mathrm{dist}_\omega(C\mu_P,\mathcal K)`$; the radius never falls below $`\mathrm{dist}_\omega(C\mu_P,\mathcal K)`$, however much information is acquired. Only enlarging the pool, and so $`\mathcal K`$, can lower it. Negative weights or unverified generated messages would enlarge $`\mathcal K`$ only by leaving the legal feedback class.

*Closure gaps call for more tests or more state.* A feature set can predict every current test exactly and still miss a future test by $`1`$, because the next update moves the state along a direction the features do not span ([Proposition 4.11](../statements/04-11_static_not_dynamic_closure_error.md#s4-11)). No query about the current tests detects this, since by construction the current tests are already predicted. The remedy is to add the missing future tests, or the state coordinate that carries them, until the test space is closed under the declared updates.

*Estimation-cost gaps call for reuse or two-fidelity estimation, only under their conditions.* Reusing an anchor's interface at a new checkpoint costs the anchor's estimation error plus the measured operator drift, and the drift itself must be certified by paired continuations ([Proposition 8.14](../statements/08-14_anchor_reuse_drift_certificate.md#s8-14)); reuse saves compute only when these continuations cost less than refitting. The two-fidelity estimator is cheaper than full continuations exactly under its cost condition, which involves variances and a correlation that must be supported by an independent pilot ([Proposition 8.15](../statements/08-15_two_fidelity_response_estimation.md#s8-15)). Without the verified condition, each device can cost more than direct measurement.

Applying a remedy to the wrong gap therefore spends budget without lowering the error of any declared test, which is why the gap is classified first. $`\square`$

## Related work

- **Data and teacher selection.** LESS ([Xia et al., 2024](#ref-xia2024less)) and GRACE ([Panigrahi et al., 2025](#ref-panigrahi2025good)) select data or teachers by static or gradient-based scores; Rule 2 says when such scores can be consistent with the learning-value ordering.
- **Influence and metagradients.** TracIn ([Pruthi et al., 2020](#ref-pruthi2020estimating)), influence functions ([Koh & Liang, 2017](#ref-koh2017understanding)), metagradients ([Engstrom et al., 2025](#ref-engstrom2025optimizing)) and In-Run Data Shapley ([Wang et al., 2025](#ref-wang2025data)) are first-order or fixed-state attributions; under Rule 5 they certify only above the margin of [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9).

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

<a id="ref-wang2025data"></a>

**Jiachen T. Wang, Prateek Mittal, Dawn Song, Ruoxi Jia (2025).** Data Shapley in one training run. *13th International Conference on Learning Representations (ICLR)*. arXiv [2406.11011](https://arxiv.org/abs/2406.11011).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
