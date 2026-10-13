# Corollary 6.7 · Aliasing and the zero value of replay

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-07_aliasing_zero_value_replay.md) · [Proof](../proofs/06-07_aliasing_zero_value_replay.md) · [Experiments](../experiments/06-07_aliasing_zero_value_replay.md) · [中文](06-07_aliasing_zero_value_replay_ZH.md)

**What is proved here.** Identical transcript laws in two environments force a worst-case error of at least half the functional gap, for every $`K`$.

**Setting used on this page.**

- **Environments.** A family $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown ([Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1)).
- **Oracle feedback** $`P^*_\theta`$: the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data.** $`K`$ i.i.d. demonstrations with law $`P^D_\theta`$, and $`n`$ queries that may be adaptive: a query point $`X_i`$ in the query space $`\mathcal X_Q`$ has outcome $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$. The law of the whole transcript is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from the transcript and private randomness (whose law does not depend on $`\theta`$) to a feedback law $`Q`$. Replay, rewriting and synthesis are distillers ([Definition 1.2](../README.md#s1-2)).
- **Audited states** $`z_1,\dots,z_T`$: fixed complete learner states, i.e. parameters together with the optimizer state ([Definition 1.3](../README.md#s1-3)), for example the states reference training reaches at checkpoints.
- **Mean increment** $`\mu^P_t(z)`$: the expected change of the complete learner state when one message from the law $`P`$ is consumed at $`z`$ ([Definition 1.3](../README.md#s1-3)). For the oracle, $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$, and the **learner Jacobian** is $`J_t=D_\theta\mu^*_t`$.
- **Task directions** $`g_t=\nabla V_{t+1}(z_t)`$: gradients of the continuation value of the declared reference tail ([Definition 1.4](../README.md#s1-4)); they do not depend on $`\theta`$.
- **Learner-relative functional** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$, with gradient $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$, and **first-order fidelity error** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$ ([Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2)). By [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) this is the first-order part of the gap in final task score between the oracle feedback and $`Q`$ at the audited states.

**Statement ([Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7)).**

- If $`\theta_0\neq\theta_1`$ satisfy $`P^D_{\theta_0}=P^D_{\theta_1}`$ and $`\Psi(\theta_0)\neq\Psi(\theta_1)`$, and either $`n=0`$ or $`P^Q_{\theta_0}(\cdot\mid x)=P^Q_{\theta_1}(\cdot\mid x)`$ for every $`x`$, then for every $`K`$ every distiller has $`\max_i\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$.
- Replay, rewriting and synthesis are functions of the transcript and private randomness, so they are covered by "every distiller" and cannot lower this bound.
- The bound is learner-relative. Whether demonstrations suffice depends on whether the directions that the demonstrations cannot see have a component along **this learner's** $`c`$; the same demonstrations can suffice for one learner and not for another ([Corollary 6.12](../statements/06-12_learner_separations.md#s6-12)(a)).
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)(a) is the instance in which the two worlds have identical demonstration laws and different optimal student policies; [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4) is the statement that post-processing cannot help.

**Results used in the proof.**

- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3): part (a), the two-point bound, with total variation zero.

**Classical tools.** Le Cam's two-point method ([Le Cam, 1973](#ref-lecam1973convergence)), through [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)(a): when the two transcript laws coincide, no test separates the two worlds.

**Proof.** The transcript laws coincide, so $`\mathrm{TV}=0`$ and [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)(a) applies with the square-root term equal to zero. $`\square`$

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).
