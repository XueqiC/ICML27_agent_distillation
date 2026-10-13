# Algorithm 9.2 · Consequence-Calibrated Experience Distillation, CED

*Statement*

[← Framework](../README.md) · [Statement](09-02_ced_algorithm.md) · [Proof](../proofs/09-02_ced_algorithm.md) · [Experiments](../experiments/09-02_ced_algorithm.md) · [中文](09-02_ced_algorithm_ZH.md)

**In one paragraph.** This is the distillation method that the framework derives. Instead of judging feedback by how good it looks (teacher quality, similarity to a target, a gradient statistic), CED judges each candidate feedback law by its measured consequence for this learner, in its current optimizer state. It first acquires missing information where the learner needs it most, then compiles candidate feedback laws, records what each candidate makes the optimizer do, measures each candidate's effect at the actual training state by paired replacement with common random numbers, selects by a lower confidence bound, and finally freezes everything before testing transfer. Every step cites the statement that requires it, and every output carries either a stated guarantee or the label "empirical".

**Setting used on this page.**

- *Student and recipe.* A complete learner state $`z=(\theta,o)`$, parameters plus optimizer state ([Definition 1.3](../README.md#s1-3)); a declared recipe (update $`U_t`$, optimizer, horizon $`R`$, reference tail); declared deployment tasks with score $`F_E`$ and learning value $`L_E(S)`$ of a protocol $`S`$.
- *Evidence.* $`K`$ trusted demonstrations per task family and a budget of $`n`$ permitted experiments (teacher queries at states the student visits, environment branches, resets) ([Definition 1.2](../README.md#s1-2)).
- *Candidate feedback laws (providers).* $`P^1,\dots,P^M`$ emit messages $`Y\sim P^j_t(\cdot\mid z)`$ (training units, packets or targets); $`P^0`$ is the reference, for example the raw demonstrations. The one-boundary response of a message is $`g_{t,z}(Y)`$ ([Definition 1.4](../README.md#s1-4)); a candidate's mean response is $`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$ and its signed advantage over the reference is $`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$.
- *Learner-relative weights and coverage.* At agent states $`s`$ the student visits, $`W_s`$ is the learner-relative weight and $`k_s=KH\,d_T(s)`$ the teacher coverage of [Proposition 6.10](06-10_water_filling_allocation.md#s6-10); water-filling allocates $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ queries, where in this formula $`\lambda>0`$ is the multiplier fixed by the budget, $`\sum_sn_s^\star=n`$.
- *Magnitude quantities.* Per coordinate, $`\lambda`$ (outside the water-filling formula) is a message's amplitude relative to the running RMS $`\sqrt v`$, $`\rho`$ the signal-to-noise ratio read from the optimizer state, and $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$ the value peak of [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2).
- *Paired-replacement score.* $`\hat S^{jk}`$ is the CRN paired-replacement score of candidates $`j,k`$ at an audited state, with statistical radius $`e^{jk}_{\rm stat}`$, surrogate objective $`\tilde F`$, and calibration $`(\alpha,\varepsilon_{\rm sur})`$ ([Definition 8.6](08-06_crn_paired_replacement_score.md#s8-6), [Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8)).
- *Confidence level* $`\delta`$; *checkpoints* $`t_1<\dots<t_A`$; $`r`$ CRN replicas per candidate and checkpoint.

<a id="s9-2"></a>

**Algorithm 9.2 (Consequence-Calibrated Experience Distillation, CED).**

*Input.*

- A student: initial complete learner state $`z_0=(\theta_0,o_0)`$; a declared recipe (update $`U_t`$, optimizer, horizon $`R`$, reference tail); declared deployment tasks with score $`F_E`$.
- Evidence: $`K`$ trusted demonstrations per task family, and a budget of $`n`$ permitted experiments (teacher queries at states the student visits, environment branches, resets).
- Candidate generators that produce feedback laws $`P^1,\dots,P^M`$ (for example replay of the demonstrations, teacher corrections at student states, self-generated trajectories filtered by the environment reward, teacher rewrites of the demonstrations), and the reference law $`P^0`$.
- An audit budget: checkpoints $`t_1<\dots<t_A`$, $`r`$ replicas per candidate and checkpoint, a surrogate objective $`\tilde F`$, a small number of full-tail task-evaluation paired audits for calibration, and a confidence level $`\delta`$.

*Output.*

- The distilled student $`z_R`$.
- An evidence ledger that separates new evidence from replays, rewrites and synthetic descendants.
- At each checkpoint: the paired-replacement scores $`\hat S^{jk}`$ with their radii, the selected law, and whether the selection is certified.
- A frozen selector, and separate method-transfer and policy-transfer reports.

*Procedure.*

1. **Record the complete learner state.** At every checkpoint keep a restorable complete state $`z_t=(\theta_t,o_t)`$, including $`m`$, $`v`$, the step counter, the sampler and the random state. Record warmth as in [Remark 5.11](05-11_checking_warm_state_conditions.md#s5-11): the sensitivity-weighted warmth and its maximum term.
2. **Acquire before compiling.**
   - Run a pilot to estimate, at the states the student visits, the learner-relative weights $`W_s`$ and the teacher coverage $`k_s=KH\,d_T(s)`$. Use the closed forms of [Proposition 6.11](06-11_learner_weights_sgd_warm_adam.md#s6-11) when parameter coordinates are state-owned; otherwise measure $`W_s`$ on state clusters with grouped CRN replacement scores ([Definition 8.6](08-06_crn_paired_replacement_score.md#s8-6)).
   - Allocate the remaining queries by water-filling, $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ ([Proposition 6.10](06-10_water_filling_allocation.md#s6-10)); in the general model this is the c-optimal design with target vector $`c=\sum_tJ_t^*g_t`$ ([Theorem 6.5](06-05_achievability.md#s6-5)).
   - Keep the ledger. Only new evidence counts toward the information budget.
3. **Compile candidates.** Turn the evidence into a finite menu of candidate feedback laws $`P^0,\dots,P^M`$. Each candidate is specified by the packets it actually emits: unit content and grouping.
4. **Record each candidate's learner statistic and control its magnitude.**
   - At $`z_t`$, record for each candidate the quantities that determine its first-order effect under this optimizer: the mean optimizer increment $`\mathbb E\,\Delta`$ ([Theorem 5.10](05-10_first_order_statistic_by_learner.md#s5-10); for AdamW the mean Adam increment of [Proposition 5.5](05-05_exact_first_order_statistic_adam.md#s5-5)), the clipping activation rate ([Proposition 5.9](05-09_partially_active_clipping.md#s5-9)), and per coordinate the magnitude ratio $`\lambda`$ relative to the running RMS and the signal-to-noise ratio $`\rho`$ read from the optimizer state ([Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2)).
   - Flag a candidate whose mass on high-$`\rho`$ coordinates lies beyond the value peak $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$, or in the large-$`\lambda`$ regime of Corollary 7.2(c). A rescaled or clipped variant of a candidate is a new candidate and enters the menu of step 5.
5. **Measure consequence at the actual states.**
   - At $`z_t`$, draw $`r`$ replicas with common random numbers and compute the paired-replacement scores $`\hat S^{jk}`$ for all pairs, including every candidate against the reference ([Definition 8.6](08-06_crn_paired_replacement_score.md#s8-6)), with radii $`e_{\rm stat}^{jk}`$ ([Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8)).
   - Calibrate $`(\alpha,\varepsilon_{\rm sur})`$ on a few candidate pairs against full-tail task-evaluation paired audits ([Theorem 8.3](08-03_actual_state_adaptive_audit_fidelity.md#s8-3), [Remark 8.11](08-11_surrogate_calibration_and_transfer.md#s8-11)).
   - First-order scores may serve only as a screen, on pairs whose first-order margin exceeds the second-order remainder ([Corollary 8.9](08-09_first_order_version_margin_criterion.md#s8-9)) and that step 4 did not flag.
   - If a response model on features is fitted, fit it under the bounded-coefficient objective of [Theorem 8.1](08-01_stability_approximation_tradeoff.md#s8-1).
6. **Select, train, re-audit, freeze, test transfer.**
   - Select $`j_t\in\arg\max_j\big[\hat S^{j0}-e_{\rm stat}^{j0}-\varepsilon_{\rm sur}\big]`$. If no candidate has a positive lower bound against the reference, keep the reference; a selection made from point estimates is reported as uncertified.
   - Train with $`P^{j_t}`$ until the next checkpoint. Using the selection at states that were not audited requires the transport and exit terms of [Theorem 8.2](08-02_frozen_finite_interface_selection_transport.md#s8-2), or a re-audit at the next checkpoint that checks whether the ordering still holds ([Remark 8.11](08-11_surrogate_calibration_and_transfer.md#s8-11)).
   - Freeze the candidate menu, features, cells, surrogate, calibration and selector before the evaluation data. Evaluate method transfer (the frozen protocol rerun on a new task family with its own $`K`$ demonstrations) and policy transfer (the distilled student deployed on the new family with no further update) separately, on held-out families, recipes and states.

*Guarantees.* Each guarantee is the cited theorem applied to the output of the corresponding step.

- **(G1) Per checkpoint.** Under [Assumption 8.7](08-07_certificate_assumptions.md#s8-7), with probability at least $`1-\delta`$, simultaneously for all pairs at $`z_t`$: if step 6 selects $`j_t\ne0`$ because its lower bound is positive, then replacing one round of the reference by $`P^{j_t}`$ at $`z_t`$, followed by the reference tail, does not lower the task value ([Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8)).
- **(G2) Whole run.** If the audits of step 5 are full-tail paired audits at the actual states and step 6 selects by their lower confidence bound, then $`L_E({\rm adaptive})-L_E(0)\ge-\delta`$ ([Theorem 8.3](08-03_actual_state_adaptive_audit_fidelity.md#s8-3)).
- **(G3) Information.** No step that only post-processes the transcript lowers the information regret ([Proposition 2.4](02-04_post_processing_cannot_reduce_information.md#s2-4)). Under the assumptions of [Theorem 6.5](06-05_achievability.md#s6-5), the acquisition of step 2 attains the learner-relative information bound up to the constant factors stated there.
- **(G4) Otherwise empirical.** A selection made from point estimates, a calibration whose transfer is not verified, or a selection used at states that were not audited without transport terms carries no certificate. The claim is then about the mean outcome of the frozen method over a declared population of tasks and seeds. It is supported by whole-run randomized comparisons and is labelled empirical.
- **Known limitation.** With the a-priori range of [Assumption 8.7](08-07_certificate_assumptions.md#s8-7), the radius of Theorem 8.8 can exceed the typical gaps between candidates; this is an open problem ([Remark 8.12](08-12_open_problems.md#s8-12)). In that regime step 6 falls under (G4).

**What it says.** Distill by: keeping the full optimizer state; acquiring information where the learner-relative weight is high and teacher coverage is low; compiling candidates as packet laws; recording each candidate's optimizer statistic and flagging oversized messages; measuring each candidate's consequence at the actual state by CRN paired replacement, calibrated against a few full-tail audits; selecting by lower confidence bound; and freezing before testing transfer.

**Intuition.** Choose feedback by what happens after this learner, in its current state, consumes it, not by how good the feedback looks on its own.

**Role in the framework.**

- It is where the framework lands: every step is required by a statement of §2–§8, and [Remark 9.3](09-03_derivation_design_rules.md#s9-3) derives the six design rules behind the steps.
- The planned real-model tests R0–R3 test its steps ([Remark 9.4](09-04_scope_of_evidence.md#s9-4)).

**Derivation outline.** The [proof page](../proofs/09-02_ced_algorithm.md) gives, for each step, the statements that force it, and derives (G1)–(G3) by applying Theorem 8.8, Theorem 8.3, Proposition 2.4 and Theorem 6.5 to the outputs of steps 5, 6 and 2.

**Illustration.** Suppose the reference replays the $`K`$ demonstrations and a second candidate adds teacher corrections at long-tail student states. For states visited often enough to be in the frequent regime of [Proposition 6.11](06-11_learner_weights_sgd_warm_adam.md#s6-11), the learner-relative weight under warm Adam scales like the visitation rate $`d(s)`$, not like $`d(s)^2`$ as under SGD. So, relative to the common states, the long-tail states weigh more, and step 2 spends more queries there than an SGD-style allocation would. Step 5 then decides by measured consequence whether the corrections, compiled as packets, actually beat replay at the current state.

**Uses.** Each step cites the statements it applies; the main ones are

- [Proposition 2.4](02-04_post_processing_cannot_reduce_information.md#s2-4), [Theorem 6.5](06-05_achievability.md#s6-5), [Propositions 6.10–6.11](06-10_water_filling_allocation.md#s6-10): acquisition (step 2);
- [Proposition 9.1](09-01_coupling_width.md#s9-1): candidates are packet laws (step 3);
- [Theorem 5.10](05-10_first_order_statistic_by_learner.md#s5-10), [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2), [Remark 7.5](07-05_where_first_order_rankers_fail.md#s7-5): learner statistic and magnitude control (step 4);
- [Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8), [Theorem 8.3](08-03_actual_state_adaptive_audit_fidelity.md#s8-3), [Corollary 8.9](08-09_first_order_version_margin_criterion.md#s8-9): measurement and certification (steps 5–6);
- [Theorem 8.2](08-02_frozen_finite_interface_selection_transport.md#s8-2): transport for unaudited states (step 6).

**Used by.**

- [Remark 9.3](09-03_derivation_design_rules.md#s9-3): the design rules the steps implement.
- [Remark 9.4](09-04_scope_of_evidence.md#s9-4): the planned real-model tests.

## Related work

- Step 2 queries the teacher at the student's own states, as DAgger does ([Ross et al., 2011](#ref-ross2011reduction)); here the queries are allocated by learner-relative weight.
- Data-selection and teacher-selection scores such as LESS ([Xia et al., 2024](#ref-xia2024less)) and GRACE ([Panigrahi et al., 2025](#ref-panigrahi2025good)) rank candidates by a statistic of gradients; CED ranks them by their measured consequence for the learner.

## References

- [Ross et al., 2011](../references.md#ross2011reduction): DAgger, queries at the student's states (step 2 allocates such queries by learner-relative weight).
- [Xia et al., 2024](../references.md#xia2024less) and [Panigrahi et al., 2025](../references.md#panigrahi2025good): data and teacher selection scores (CED ranks by measured consequence instead).
