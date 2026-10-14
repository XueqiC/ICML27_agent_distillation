# Remark 9.3 · Derivation of the design rules

*Statement*

[← Framework](../README.md) · [Statement](09-03_derivation_design_rules.md) · [Proof](../proofs/09-03_derivation_design_rules.md) · [Experiments](../experiments/09-03_derivation_design_rules.md) · [中文](09-03_derivation_design_rules_ZH.md)

**In one paragraph.** [Algorithm 9.2](09-02_ced_algorithm.md#s9-2) (CED) is the distillation method of the framework. This remark extracts the seven design rules behind it and says, for each, which earlier statements force it. A rule here is a necessary condition: a method that claims the corresponding guarantee must follow it. The remark does not claim that following the rules makes an instance of CED beat a given baseline; that is an empirical question.

**Setting used on this page.**

- *Learner and candidates.* A complete learner state $`z=(\theta,o)`$, parameters plus optimizer state ([Definition 1.3](../README.md#s1-3)); candidate feedback laws $`P^0,\dots,P^M`$ emitting messages $`Y\sim P^j_t(\cdot\mid z)`$ at round $`t`$, with $`P^0`$ the reference.
- *Responses.* The one-boundary response of a message is $`g_{t,z}(Y)`$: the expected final task value after the update on $`Y`$ at $`z`$ followed by the declared tail ([Definition 1.4](../README.md#s1-4)). Its mean under candidate $`j`$ is $`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$, and $`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$ is the value of replacing one round of the reference by $`j`$.
- *Transcript and information regret.* $`D`$ is the transcript of all evidence (demonstrations and permitted experiments); its information regret $`\mathcal R_{\rm info}(D)`$ is the part of the regret no algorithm can remove from $`D`$ ([Theorem 2.2](02-02_regret_decomposition_decision_ambiguity.md#s2-2)).
- *Mean increment.* $`\mu_t^P(z)=\mathbb E_P\Delta`$ is the mean increment of the complete state under candidate $`P`$; warmth measures how far an Adam state is from its frozen-denominator description ([Remark 5.11](05-11_checking_warm_state_conditions.md#s5-11)).
- *Response models.* Features $`\phi_1,\dots,\phi_d`$ of a message with matrix $`\Phi`$; a response model is $`c+b^\top\phi`$; feature means are measured to tolerances $`\epsilon_j`$; $`\widehat g`$ is a response estimate; $`\mathrm{osc}(h)=\max h-\min h`$.
- *Transfer.* $`L_E(S)`$ is the learning value of protocol $`S`$ on task family $`E`$; $`J_E(\pi_z)`$ the deployment value of the student $`z`$. *Method transfer* reruns the frozen protocol on a new family $`E'`$ with its own $`K`$ demonstrations and evaluates $`L_{E'}(S)`$; *policy transfer* deploys the distilled student on $`E'`$ with no further update and evaluates $`J_{E'}(\pi_z)`$.

<a id="s9-3"></a>

**Remark 9.3 (Derivation of the design rules).** Algorithm 9.2 implements seven rules. Each rule states what a method that claims the corresponding guarantee must do; the [proof page](../proofs/09-03_derivation_design_rules.md) gives the statements that force it. The rules are necessary conditions for those guarantees; they do not by themselves show that an instance of CED outperforms a given baseline.

- **Rule 1 (Acquire before compiling).** Decision-relevant facts that the transcript does not determine can be supplied only by new permitted experiments; how much to acquire, and where, is set by the learner.
- **Rule 2 (Select by measured consequence).** On a declared family, a selection rule is consistent with the learning-value ordering only if it is a function of the mean responses $`\lbrace\bar g_t(z,j)\rbrace_j`$, or of a statistic proved sufficient for them on that family.
- **Rule 3 (Match the learner's statistic, not the raw gradient).** What a candidate must match, to first order, is the mean increment of the complete learner state in the directions of the continuation-value gradients. Responses and their statistics are therefore indexed by complete states and their warmth.
- **Rule 4 (Fit stable response models).** A response model $`c+b^\top\phi`$ on features measured to tolerances $`\epsilon_j`$ is fitted by minimizing $`\mathrm{osc}(\widehat g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$.
- **Rule 5 (Account for coverage; certify by paired replacement; use first-order audits only where the margin dominates).** A guarantee on the learning value of the selected candidate takes one of four forms:
  - (i) the frozen-audit certificate with measured transport and exit probability;
  - (ii) the actual-state lower-confidence-bound guarantee with full-tail paired audits;
  - (iii) the CRN paired-replacement certificate at audited states with a calibrated surrogate, extended along training only by transport terms or by re-auditing;
  - (iv) whole-run randomized comparisons of the frozen method, labelled empirical.
- **Rule 6 (Freeze, then test transfer).** Selectors, features, cells, surrogates and candidates are frozen before the data that evaluates them; transfer is guaranteed only through a verified decomposition of the target responses and is otherwise measured. Method transfer and policy transfer are reported separately.
- **Rule 7 (Route each gap to its remedy).** Before spending budget, classify what separates the available feedback from a sufficient one, and pay for the matching remedy only:
  - an *information gap* (the declared tests cannot yet be predicted because the transcript leaves the target uncertain in directions the tests see) is closed by acquisition;
  - a *realizability gap* (the target is determined but no mixture of the legal pool reaches it) is closed by enlarging the legal pool, not by more information;
  - a *closure gap* (a summary that predicts the current tests does not predict them after the declared further updates) is closed by extending the tests or the recorded state;
  - an *estimation-cost gap* (the response cannot be measured within budget) may be reduced by reusing an interface across checkpoints or by two-fidelity estimation, but only when the corresponding cost condition has been verified.

**What it says.**

- Acquire missing information with new experiments, planned for this learner; no rewriting of old data can replace it.
- Judge candidates by their measured consequence for this learner, including how the units are packed.
- What must be matched is the optimizer's statistic at the complete state, including the message magnitude relative to the running RMS under warm Adam, not the raw gradient.
- Fit response models with small coefficients on noisy features; certify by paired replacement or full-tail audits; account for coverage when moving to new states.
- Freeze everything before the data that evaluates it, and report the two kinds of transfer separately.
- Decide what kind of gap remains before buying anything: information, realizability, closure or estimation cost each have their own remedy, and the wrong remedy buys nothing.

**Intuition.** Every shortcut a method might take (rewriting instead of acquiring, ranking by a static score, matching gradients, trusting an audit at states it never saw, tuning on the evaluation data, querying for a target that no legal feedback can reach) corresponds to an earlier statement that exhibits a case where the shortcut fails.

**Role in the framework.**

- It is the bridge from the theory of §2–§8 to the method: each step of [Algorithm 9.2](09-02_ced_algorithm.md#s9-2) implements one or more rules.
- Its necessity claims are proved; Rules 1–6 are probed in the controlled learner ([experiments page](../experiments/09-03_derivation_design_rules.md)), and the real-model tests are described in [Remark 9.4](09-04_scope_of_evidence.md#s9-4). Rule 7 has no check of its own; its parts are checked on the pages of the statements that force it.

**Derivation outline.**

- **Rule 1.** Post-processing cannot lower the information regret, passive data can leave a gap even with an optimal teacher, and the amount and placement of acquisition are set by the learner ([Proposition 2.4](02-04_post_processing_cannot_reduce_information.md#s2-4), [Theorem 2.6](02-06_passive_impossibility_active_attainability.md#s2-6), [Theorems 6.3](06-03_learner_relative_lower_bound.md#s6-3) and [6.5](06-05_achievability.md#s6-5), [Proposition 6.10](06-10_water_filling_allocation.md#s6-10)).
- **Rule 2.** The response is the invariant ([Theorem 3.2](03-02_continuation_response_characterization.md#s3-2)); static value and mean gradients do not identify the ordering ([Proposition 3.1](03-01_static_value_not_learning_value.md#s3-1)); task-blind statistics and parameter-only or cosine scores are inconsistent ([Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3)); packet organization is part of the candidate ([Proposition 9.1](09-01_coupling_width.md#s9-1)); no universal small statistic exists ([Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2)(d)).
- **Rule 3.** The first-order statistic is the mean increment of the complete state ([Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1), [Theorem 5.10](05-10_first_order_statistic_by_learner.md#s5-10)); under warm Adam the magnitude relative to the running RMS is part of it ([Theorem 7.1](07-01_state_channel_decomposition.md#s7-1), [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2)).
- **Rule 4.** Measurement error on a feature costs the size of its coefficient ([Theorem 8.1](08-01_stability_approximation_tradeoff.md#s8-1)).
- **Rule 5.** Dropping transport is invalid, full-tail audits are expensive, paired replacement is exact in expectation, and first-order scores certify only above their margin ([Theorems 8.2](08-02_frozen_finite_interface_selection_transport.md#s8-2), [8.3](08-03_actual_state_adaptive_audit_fidelity.md#s8-3), [8.8](08-08_crn_paired_replacement_certificate.md#s8-8), [Corollary 8.9](08-09_first_order_version_margin_criterion.md#s8-9)).
- **Rule 6.** Audit events are simultaneous only over candidates fixed beforehand; transfer is guaranteed only through a verified decomposition of target responses ([Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2)(c)); method transfer and policy transfer are different functionals.
- **Rule 7.** The information–compilation radius is bounded below separately by the visible uncertainty and by the distance to the legal outputs, and unknowns in the kernel of the interface cost nothing ([Proposition 6.16](06-16_information_compilation_radius.md#s6-16)); a target is realizable only inside the convex hull of the pool ([Proposition 6.14](06-14_convex_realizability_legal_pool.md#s6-14)); current sufficiency does not imply sufficiency after further updates ([Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11)); reuse and two-fidelity estimation save compute only under their stated cost conditions ([Propositions 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14) and [8.15](08-15_two_fidelity_response_estimation.md#s8-15)).

**Uses.**

- [Proposition 2.4](02-04_post_processing_cannot_reduce_information.md#s2-4), [Theorem 2.6](02-06_passive_impossibility_active_attainability.md#s2-6), [Theorems 6.3](06-03_learner_relative_lower_bound.md#s6-3), [6.5](06-05_achievability.md#s6-5): acquisition is necessary and learner-set (Rule 1).
- [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2), [Proposition 3.1](03-01_static_value_not_learning_value.md#s3-1), [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3), [Proposition 9.1](09-01_coupling_width.md#s9-1): select by response (Rule 2).
- [Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1), [Theorem 5.10](05-10_first_order_statistic_by_learner.md#s5-10), [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1): the learner's first-order statistic (Rule 3).
- [Theorem 8.1](08-01_stability_approximation_tradeoff.md#s8-1): stable response models (Rule 4).
- [Theorems 8.2](08-02_frozen_finite_interface_selection_transport.md#s8-2), [8.3](08-03_actual_state_adaptive_audit_fidelity.md#s8-3), [8.8](08-08_crn_paired_replacement_certificate.md#s8-8), [Corollary 8.9](08-09_first_order_version_margin_criterion.md#s8-9): the four forms of guarantee (Rule 5).
- [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2): transfer needs a verified decomposition (Rule 6).
- [Propositions 6.14](06-14_convex_realizability_legal_pool.md#s6-14) and [6.16](06-16_information_compilation_radius.md#s6-16), [Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11), [Propositions 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14) and [8.15](08-15_two_fidelity_response_estimation.md#s8-15): routing gaps to remedies (Rule 7).

**Used by.**

- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2): each step implements these rules.
- [Remark 9.4](09-04_scope_of_evidence.md#s9-4): the real-model tests.
- [Remark 9.5](09-05_assumption_audit.md#s9-5): the assumption audit.

## Related work

- Rule 1 rests on the easy direction of Blackwell's comparison of experiments: a garbling of the transcript cannot add information ([Blackwell, 1953](#ref-blackwell1953equivalent)). Acquiring labels at the student's own states is the move of DAgger ([Ross et al., 2011](#ref-ross2011reduction)).
- Rule 2 and Rule 3 are stated against first-order data scores: influence functions ([Koh & Liang, 2017](#ref-koh2017understanding)), TracIn ([Pruthi et al., 2020](#ref-pruthi2020estimating)) and LESS ([Xia et al., 2024](#ref-xia2024less)) rank training data by gradient statistics rather than by measured consequence at the complete learner state.
- Rule 6 separates the data that tunes a method from the data that tests it, as the held-out safety test of high-confidence policy improvement does ([Thomas et al., 2015](#ref-thomas2015high)).
- The realizability part of Rule 7 is the convex-hull criterion of convex analysis, with Carathéodory's bound on the support ([Rockafellar, 1970](#ref-rockafellar1970convex)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-blackwell1953equivalent"></a>

**David Blackwell (1953).** Equivalent comparisons of experiments. *The Annals of Mathematical Statistics* 24, no. 2, 265–272. DOI [10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032).

<a id="ref-koh2017understanding"></a>

**Pang Wei Koh, Percy Liang (2017).** Understanding black-box predictions via influence functions. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1885–1894. [proceedings.mlr.press/v70/koh17a](https://proceedings.mlr.press/v70/koh17a.html).

<a id="ref-pruthi2020estimating"></a>

**Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan (2020).** Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 19920–19930. arXiv [2002.08484](https://arxiv.org/abs/2002.08484).

<a id="ref-rockafellar1970convex"></a>

**R. Tyrrell Rockafellar (1970).** Convex Analysis. Princeton University Press, Princeton, NJ (Princeton Mathematical Series 28). DOI [10.1515/9781400873173](https://doi.org/10.1515/9781400873173).

<a id="ref-ross2011reduction"></a>

**Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (2011).** A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 15:627–635. [proceedings.mlr.press/v15/ross11a](https://proceedings.mlr.press/v15/ross11a.html).

<a id="ref-thomas2015high"></a>

**Philip S. Thomas, Georgios Theocharous, Mohammad Ghavamzadeh (2015).** High confidence policy improvement. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37:2380–2388. [proceedings.mlr.press/v37/thomas15](https://proceedings.mlr.press/v37/thomas15.html).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
