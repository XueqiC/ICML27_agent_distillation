# Corollary 6.7 · Aliasing and the zero value of replay

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-07_aliasing_zero_value_replay.md) · [Proof](../proofs/06-07_aliasing_zero_value_replay.md) · [Experiments](../experiments/06-07_aliasing_zero_value_replay.md) · [中文](06-07_aliasing_zero_value_replay_ZH.md)

**In one paragraph.** If two environments produce the same demonstration law but differ in the learner-relative functional, then no number of demonstrations, and no replay, rewriting or synthesis of them, can bring the worst-case error below half the difference.

**Setting used on this page.**

- **Environments.** A family $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown ([Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1)).
- **Oracle feedback** $`P^*_\theta`$: the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data.** $`K`$ i.i.d. demonstrations with law $`P^D_\theta`$, and $`n`$ queries that may be adaptive: a query point $`X_i`$ in the query space $`\mathcal X_Q`$ has outcome $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$. The law of the whole transcript is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from the transcript and private randomness (whose law does not depend on $`\theta`$) to a feedback law $`Q`$. Replay, rewriting and synthesis are distillers ([Definition 1.2](../README.md#s1-2)).
- **Audited states** $`z_1,\dots,z_T`$: fixed complete learner states, i.e. parameters together with the optimizer state ([Definition 1.3](../README.md#s1-3)), for example the states reference training reaches at checkpoints.
- **Mean increment** $`\mu^P_t(z)`$: the expected change of the complete learner state when one message from the law $`P`$ is consumed at $`z`$ ([Definition 1.3](../README.md#s1-3)). For the oracle, $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$, and the **learner Jacobian** is $`J_t=D_\theta\mu^*_t`$.
- **Task directions** $`g_t=\nabla V_{t+1}(z_t)`$: gradients of the continuation value of the declared reference tail ([Definition 1.4](../README.md#s1-4)); they do not depend on $`\theta`$.
- **Learner-relative functional** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$, with gradient $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$, and **first-order fidelity error** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$ ([Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2)). By [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) this is the first-order part of the gap in final task score between the oracle feedback and $`Q`$ at the audited states.

<a id="s6-7"></a>

**Corollary 6.7 (Aliasing and the zero value of replay).**

- If $`\theta_0\neq\theta_1`$ satisfy $`P^D_{\theta_0}=P^D_{\theta_1}`$ and $`\Psi(\theta_0)\neq\Psi(\theta_1)`$, and either $`n=0`$ or $`P^Q_{\theta_0}(\cdot\mid x)=P^Q_{\theta_1}(\cdot\mid x)`$ for every $`x`$, then for every $`K`$ every distiller has $`\max_i\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$.
- Replay, rewriting and synthesis are functions of the transcript and private randomness, so they are covered by "every distiller" and cannot lower this bound.
- The bound is learner-relative. Whether demonstrations suffice depends on whether the directions that the demonstrations cannot see have a component along **this learner's** $`c`$; the same demonstrations can suffice for one learner and not for another ([Corollary 6.12](../statements/06-12_learner_separations.md#s6-12)(a)).
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)(a) is the instance in which the two worlds have identical demonstration laws and different optimal student policies; [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4) is the statement that post-processing cannot help.

**What it says.**

- If two environments give identical demonstration laws and the learner-relative functional differs between them, every distiller errs by at least half the difference, for every $`K`$; replay and rewriting cannot help.

**Intuition.**

- More copies of the same evidence never separate worlds that the evidence cannot separate.

**Role in the framework.**

- It is the learner-relative form of passive impossibility ([Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)(a)) and of the fact that post-processing cannot add information ([Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4)): whether demonstrations suffice depends on the learner.

**Derivation outline.**

- Identical transcript laws give total variation zero in [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)(a).

**Uses.**

- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3): two-point bound with total variation zero.
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6): the special case of two aliased worlds.
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4): post-processing cannot reduce information regret.

**Used by.**

- [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10): irreducible part (d).
- [Corollary 6.12](../statements/06-12_learner_separations.md#s6-12): the SGD side of the separation.
- [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3): design rule: acquire before compiling.

**Status.** proved.

## Related work

- Replay, rewriting and synthesis are garblings of the transcript, and a garbling is never more informative than the experiment it comes from ([Blackwell, 1953](#ref-blackwell1953equivalent)); the corollary is the quantitative, learner-relative form of this fact for aliased environments.
- The remedy, querying the teacher where the learner goes instead of replaying demonstrations, is the idea of interactive imitation learning ([Ross et al., 2011](#ref-ross2011reduction)); the corollary adds that whether a direction needs queries depends on the learner's $`c`$.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-blackwell1953equivalent"></a>

**David Blackwell (1953).** Equivalent comparisons of experiments. *The Annals of Mathematical Statistics* 24, no. 2, 265–272. DOI [10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032).

<a id="ref-ross2011reduction"></a>

**Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (2011).** A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 15:627–635. [proceedings.mlr.press/v15/ross11a](https://proceedings.mlr.press/v15/ross11a.html).
