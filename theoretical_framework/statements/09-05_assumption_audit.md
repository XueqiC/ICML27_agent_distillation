# Remark 9.5 · Assumption audit

*Statement*

[← Framework](../README.md) · [Statement](09-05_assumption_audit.md) · [中文](09-05_assumption_audit_ZH.md)

**In one paragraph.** A framework that holds for any measurable learner says little about a particular one unless it is clear which further conditions each useful conclusion needs. This remark separates three things. First, the conditions the main line needs, which are few and part of the problem itself. Second, the stronger conditions that some results use, each listed with the results that use it and what the framework does when the condition fails. Third, what cannot be had at once, whatever the assumptions. It has no proof and no experiments page of its own: each row points to the statement that proves or uses it.

**Setting used on this page.**

- *Main line.* The exact consequence identity of [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2), its version with an approximate value ([Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4)), the response operator and its spectrum ([Definition 4.8](04-08_response_operator.md#s4-8), [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9)), the learner-relative information bound (Theorem A) and the finite certificates of §8.
- *Restricted model.* A model with conditions beyond the main line, under which a statement is proved exactly or to leading order; its conclusions are mechanism hypotheses for the real system, to be tested.

<a id="s9-5"></a>

**Remark 9.5 (Assumption audit).**

**(a) What the main line needs.** A finite training budget, a measurable true conditional kernel for every declared protocol (non-linear updates, clipping, cold-start Adam, non-stationary learning rates and state-dependent randomness included), bounded task scores, and a declared comparison scope (environment, initial state law, evaluator and training mechanism shared, or their differences made explicit). Token independence, gradient stationarity, small step sizes, low rank, global contraction, Fisher regularity and known hidden states are not among them.

**(b) Stronger conditions, the results that use them, and what happens when they fail.**

| Condition | Judgement | How the framework handles it |
|---|---|---|
| The parameter vector alone is the learner state | False for optimizers with memory | The complete state includes $`m`$, $`v`$, the clock, buffers and the history the update uses ([Definition 1.3](../README.md#s1-3)); equal parameters can have different continuations ([Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6)) |
| Raw observations are Markov | Unnecessary | Use the full history or a predictive state ([Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10)) |
| Gradients, tokens or repeated data are independent | Usually too strong in practice | The main line does not use it; independence is assumed only across experimental replicates ([Assumption 8.7](08-07_certificate_assumptions.md#s8-7), [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12)) |
| Affine averaging learners, independent experience blocks, finitely many decisions | Restricted-model conditions | They give the exact moment order and the cumulant rates of [Theorems 4.4–4.5](04-04_affine_learners_moment_order.md#s4-4) as computable examples; they are not extended to arbitrary optimizers |
| A small learning rate makes the complete-state Taylor remainder second order | Not in general: the optimizer state can move on other scales | The main line uses exact kernels; local expansions report their own remainders ([Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1)) |
| The continuation gradient is frozen and does not depend on the environment | Valid for the first-order target, possibly insufficient for the derivative of the true gain | Theorem A's target is the first-order functional of [Definition 6.2](06-02_learner_relative_functional_information_norm.md#s6-2); the derivative of the true gain also differentiates the response mechanism |
| Every design has positive-definite information | Too strong for a linear target | Singular designs are allowed; only the target direction must be identifiable ([Lemma 6.4](06-04_design_minimax_equality.md#s6-4)) |
| An efficient two-stage estimator exists | A substantive condition, not implied by c-optimal design | Proved in the linear-Gaussian case ([Proposition 6.15](06-15_linear_gaussian_target.md#s6-15)); in general an assumption of [Theorem 6.5](06-05_achievability.md#s6-5) |
| The oracle law at the estimate can be generated | Strong for real distillation | Stated as an assumption of Theorem 6.5; with a finite legal pool, convex realizability and its residual replace it ([Proposition 6.14](06-14_convex_realizability_legal_pool.md#s6-14)) |
| Independent visits, independent coordinates, frequent-visit regime, mean field | Restricted-model conditions | The learner weights of [Proposition 6.11](06-11_learner_weights_sgd_warm_adam.md#s6-11) hold to leading order in those regimes and are used as a computable allocation rule whose usefulness R2 tests ([Remark 9.4](09-04_scope_of_evidence.md#s9-4)) |
| Adam at a stationary point $`m=\mathbb Eg`$, $`v=\mathbb Eg^2`$; tail gradients independent of the parameters; affine terminal score; $`\mathbb E[m/\sqrt v]`$ replaced by $`\mathbb Em/\sqrt{\mathbb Ev}`$; no bias correction, $`\epsilon=0`$, no decay | Restricted mean-field model; the replacement of the ratio's mean is not an identity | [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2) is proved in that model; its real-model prediction had the opposite sign to the measurement ([Check 7.6](../experiments/07-02_stationary_amplitude_law.md#c7-6)); the general mechanism is the exact decomposition of [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1) |
| Quantiles of warmth control the error | Insufficient for a task-weighted error | Report the sensitivity-weighted error and its largest terms ([Remark 5.11](05-11_checking_warm_state_conditions.md#s5-11), [Check 5.19](../experiments/05-11_checking_warm_state_conditions.md#c5-19)) |
| Low rank, fast spectral decay, a finite-dimensional predictive state | Not universal facts | Measured properties or conditional compression; incompressible cases are kept ([Theorem 4.9](04-09_spectral_sufficiency.md#s4-9), [Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10)) |
| A global bound on density ratios | Possibly too strong | For a finite catalog, use its mixture as reference measure ([Definition 4.8](04-08_response_operator.md#s4-8)); outside the catalog the coverage cost is paid again |
| A small average error makes every task and state accurate | False | Per-test residuals with their weights, and coverage of the actual occupancy ([Theorem 4.9](04-09_spectral_sufficiency.md#s4-9), [Proposition 8.13](08-13_shared_features_across_checkpoints.md#s8-13)) |
| All dynamics are stable or contractive | Unnecessary and possibly false | Finite-horizon gains or Bellman residuals; infinite-horizon linear time-invariant bounds are not imported ([Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11), [Proposition 4.13](04-13_finite_horizon_gramians.md#s4-13)) |
| Every observation has probability bounded away from zero | Too strong for rich observation spaces | Conditioning only on events of positive probability; rare branches are charged separately ([Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10)) |
| Spectral or predictive-state outputs can be emitted as legal messages | False | Positive-weight compilation over the legal pool with a realizability check ([Proposition 6.14](06-14_convex_realizability_legal_pool.md#s6-14)) |
| Frozen candidates, independent replicates, a known population range | The true conditions of the finite-sample certificates | Kept and implemented in the protocols ([Assumption 8.7](08-07_certificate_assumptions.md#s8-7), [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12)); adaptive generation of candidates needs fresh confirmation data or a sequentially valid analysis |
| The observed sample range is the population range | False | A radius built from it is diagnostic only and is never reported as strict ([Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8), [Check 8.17](../experiments/08-08_crn_paired_replacement_certificate.md#c8-17)) |
| A surrogate calibration holds at every new state | Strong; not implied by finite calibration residuals | An exchangeability assumption ([Remark 8.11](08-11_surrogate_calibration_and_transfer.md#s8-11)); a testable transfer condition is open ([Remark 8.17](08-17_open_problems.md#s8-17)) |
| Checkpoints can be restored, futures forked and states queried for free | False for real systems | Only declared interventions are used, and every fork, pilot, fit, drift check and evaluation is charged ([Algorithm 9.2](09-02_ced_algorithm.md#s9-2), [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12)) |

**(c) What cannot be had at once.** These obstacles do not disappear under a change of vocabulary.

- *No universal strong compression with per-task fidelity.* With $`m`$ messages and the $`m`$ indicator responses, the centred response operator has rank $`m-1`$, so every feedback difference matters for some task, and no fixed family of fewer than $`m-1`$ statistics determines all response means ([Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2)(d)). This does not contradict the approximate interface bounds for a fixed density budget and an average error.
- *Current sufficiency is not future sufficiency.* A summary can predict every current test and miss a future one by the full score range ([Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11)).
- *No identification without information.* When two environments produce the same transcript law, an error of half their difference cannot be removed by any distiller, generation included ([Theorem 6.3](06-03_learner_relative_lower_bound.md#s6-3), [Corollary 6.7](06-07_aliasing_zero_value_replay.md#s6-7)).
- *A fully general surrogate does not preserve rankings.* If the surrogate is one minus the true score, every ranking is reversed; regions a finite calibration set does not cover can behave the same way, which is why calibration transfer is an assumption ([Remark 8.11](08-11_surrogate_calibration_and_transfer.md#s8-11)).
- *A zero measured response does not show that a direction is absent.* Without excitation and coverage, an unmeasured direction looks like a zero mode; a low-rank observed submatrix only bounds the rank from below ([Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10), [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12)).

The framework therefore does not claim uniformly low-dimensional, cheap, all-task-optimal distillation without structure. What it offers is a theory in which structure can be measured, the cost of a mismatch can be computed and a failure can be recognized.

**(d) Conditions of the results on the response operator, closure, reuse, estimation cost, compilation and memory.**

| Result | What it needs | Part of the main line? | If it fails |
|---|---|---|---|
| Finite-dimensional factorization and linear predictive realization ([Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12), [Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10)) | A finite-dimensional declared future test space, spanned by the chosen tests | No; an exact finite representation, possibly strong | Use the residual versions; enlarge the tests or the state; make no general low-dimension claim |
| Sufficiency for adaptive future protocols | Closure of the conditioned operators over observation branches | Only for adaptive protocols | Narrow the protocol class or pay for the omitted branches |
| Shared message features across checkpoints ([Proposition 8.13](08-13_shared_features_across_checkpoints.md#s8-13)) | A common message measure, tests with the same meaning, bounded density differences at each state | Not given in every environment | Work on a common legal pool, or build separate interfaces per cluster of states |
| Anchor transfer ([Proposition 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14)) | A controlled response drift; restorable forks; independent paired replicates | An operating condition of that certificate | Build a new anchor or report the transfer as uncertified; no global smoothness is assumed |
| The memory construction ([Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6)) | A specified reachable history, realizable gradients, non-zero denominators | Conditions of a constructed instance | It shows existence and mechanism; the size of the effect on a real model is measured separately (T1 in [Remark 9.4](09-04_scope_of_evidence.md#s9-4)) |
| The complete one-step adjoint ([Theorem 7.1](07-01_state_channel_decomposition.md#s7-1)) | Local differentiability of the update and of the continuation value | No; an explanatory derivative | Use exact responses where the update is not differentiable; finite changes keep their remainders |
| Unbiased two-fidelity estimation ([Proposition 8.15](08-15_two_fidelity_response_estimation.md#s8-15)) | Two independent groups; equal marginals of the cheap output; estimator and allocation frozen | Yes, for that estimator; ordinary but must be implemented | Confirm on independent data, or prove the properties of an adaptive estimator separately |
| Two-fidelity estimation is cheaper | Its cost condition, enough reuse, and the pilot and fitting charged | No; a condition to be tested | Use direct full continuations and make no cheapness claim |
| The information–compilation radius ([Proposition 6.16](06-16_information_compilation_radius.md#s6-16)) | A valid confidence set, a compact legal feasible set, a fixed or calibrated response interface | The scope of that conditional model | Use a joint uncertainty set over interface and target, and pay the calibration error |
| Extension to new tasks, candidates or checkpoints | Coverage, or a valid transfer control | Never follows from passing the old catalog | Add confirmation data or narrow the scope of the guarantee |

None of these results adds global gradient stationarity, a zero Hessian, contraction of all dynamics, default low rank, free exact responses, universal calibration transfer or generation of arbitrary oracle laws. The stronger conditions of the restricted models are kept as such and are not renamed into general facts.

**What it says.** The main line needs only a finite budget, a measurable true kernel, bounded scores and a declared comparison scope. Every stronger condition is attached to the results that use it, with the action the framework takes when it fails, and some combinations of wishes cannot be satisfied at all.

**Role in the framework.**

- It is the reader's map of what each conclusion costs in assumptions, and it prevents the generality of the main line from being read as a claim that the method needs no structure.
- It is the counterpart of [Remark 9.4](09-04_scope_of_evidence.md#s9-4): that remark says which predictions the evidence supports, this one says which assumptions the predictions rest on.

**Uses.** Each row cites the statement that proves or uses it; the main ones are [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2), [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9), [Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11), [Theorem 6.5](06-05_achievability.md#s6-5), [Proposition 6.14](06-14_convex_realizability_legal_pool.md#s6-14), [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2), [Assumption 8.7](08-07_certificate_assumptions.md#s8-7) and [Proposition 8.15](08-15_two_fidelity_response_estimation.md#s8-15).

**Used by.**

- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2): the assumptions behind each step.
- [Remark 9.6](09-06_contribution_units_prior_work.md#s9-6): the scope of each contribution claim.

## Related work

- The main-line identities belong to the performance-difference family ([Kakade & Langford, 2002](#ref-kakade2002approximately)); they organize errors but do not by themselves make them small.
- Predictive-state representations ([Littman, Sutton & Singh, 2001](#ref-littman2001predictive)) use a finite-dimensional test space as a modelling condition; here the same condition is kept as a condition, not a default.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).

<a id="ref-littman2001predictive"></a>

**Michael L. Littman, Richard S. Sutton, Satinder Singh (2001).** Predictive representations of state. *Advances in Neural Information Processing Systems 14 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).
