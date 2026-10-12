# Theorem 6.3 · Learner-relative lower bound

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-03_learner_relative_lower_bound.md) · [Proof](../proofs/06-03_learner_relative_lower_bound.md) · [Experiments](../experiments/06-03_learner_relative_lower_bound.md) · [中文](06-03_learner_relative_lower_bound_ZH.md)

**In one paragraph.** No distiller, replay, rewriting and synthesis included, can reproduce the oracle feedback's first-order effect on the declared task better than a quarter of the learner-relative information norm. This is the lower half of Theorem A.

**Setting used on this page.**

- **Environments.** A family $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown ([Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1)).
- **Oracle feedback** $`P^*_\theta`$: the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data.** $`K`$ i.i.d. demonstrations with law $`P^D_\theta`$, and $`n`$ queries that may be adaptive: a query point $`X_i`$ in the query space $`\mathcal X_Q`$ has outcome $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$. The law of the whole transcript is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from the transcript and private randomness (whose law does not depend on $`\theta`$) to a feedback law $`Q`$. Replay, rewriting and synthesis are distillers ([Definition 1.2](../README.md#s1-2)).
- **Audited states** $`z_1,\dots,z_T`$: fixed complete learner states, i.e. parameters together with the optimizer state ([Definition 1.3](../README.md#s1-3)), for example the states reference training reaches at checkpoints.
- **Mean increment** $`\mu^P_t(z)`$: the expected change of the complete learner state when one message from the law $`P`$ is consumed at $`z`$ ([Definition 1.3](../README.md#s1-3)). For the oracle, $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$, and the **learner Jacobian** is $`J_t=D_\theta\mu^*_t`$.
- **Task directions** $`g_t=\nabla V_{t+1}(z_t)`$: gradients of the continuation value of the declared reference tail ([Definition 1.4](../README.md#s1-4)); they do not depend on $`\theta`$.
- **Learner-relative functional** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$, with gradient $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$, and **first-order fidelity error** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$ ([Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2)). By [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) this is the first-order part of the gap in final task score between the oracle feedback and $`Q`$ at the audited states.
- **Fisher informations** $`I_D(\theta)`$ of one demonstration and $`I_Q(\theta;x)`$ of one query at $`x\in\mathcal X_Q`$. For a direction $`h\in\mathbb R^p`$, $`\mathbb I_{K,n}(\theta;h)=K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h`$.
- **Learner-relative information norm** $`\mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\lvert c(\theta)^\top h\rvert/\sqrt{\mathbb I_{K,n}(\theta;h)}`$, with $`0/0=0`$ and $`a/0=\infty`$ for $`a>0`$.
- **Growth condition (G)**: there are $`N\to\infty`$ and $`0<a\le b<\infty`$ with $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$ for every unit $`h`$.

<a id="s6-3"></a>

**Theorem 6.3 (Learner-relative lower bound).**

- **(a) Exact two-point bound.** For all $`\theta_0,\theta_1\in\Theta`$ and every distiller,

```math
\max_{i\in\lbrace0,1\rbrace}\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\big\lvert\Psi(\theta_1)-\Psi(\theta_0)\big\rvert\Big(1-\sqrt{\tfrac12\big(K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)\big)}\Big),
```

  where $`\mathrm{KL}_D=\mathrm{KL}(P^D_{\theta_1}\Vert P^D_{\theta_0})`$ and $`\mathrm{KL}_Q(x)=\mathrm{KL}(P^Q_{\theta_1}(\cdot\mid x)\Vert P^Q_{\theta_0}(\cdot\mid x))`$.

- **(b) Local bound.** Assume that each $`\mu^*_t`$ is $`C^1`$ near $`\theta_0`$, that both experiments are differentiable in quadratic mean at $`\theta_0`$, and that, as $`\tau\to0`$,

```math
\mathrm{KL}(P^D_{\theta_0+\tau h}\Vert P^D_{\theta_0})\le\tfrac12\tau^2h^\top I_D(\theta_0)h\,(1+o(1)),\qquad \mathrm{KL}_Q(x)\le\tfrac12\tau^2h^\top I_Q(\theta_0;x)h\,(1+o(1)),
```

  uniformly in $`x\in\mathcal X_Q`$ and in unit $`h`$. Let $`h`$ be a unit direction with $`c(\theta_0)^\top h\neq0`$ and $`\mathbb I_{K,n}(\theta_0;h)\to\infty`$, and put $`\tau=\mathbb I_{K,n}(\theta_0;h)^{-1/2}`$. Then

```math
\inf_Q\max_{\theta\in\lbrace\theta_0,\,\theta_0+\tau h\rbrace}\mathbb E_\theta\mathrm{Err}_\theta(Q)\ge\frac{\lvert c(\theta_0)^\top h\rvert}{4\sqrt{\mathbb I_{K,n}(\theta_0;h)}}\,\big(1-o(1)\big).
```

- **(c) The norm as the local lower bound.** Under (G), taking for $`h`$ a maximizer of the ratio in [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2), the local minimax lower bound is $`\tfrac14\mathfrak n_{K,n}(\theta_0)\,(1-o(1))`$.

**What it says.**

- No distiller, replay, rewriting and synthesis included, can match the oracle feedback's first-order effect on the declared task better than a quarter of the learner-relative information norm $`\mathfrak n_{K,n}`$. The norm measures how much the $`K`$ demonstrations and $`n`$ queries reveal about the environment in the one direction $`c`$ that the learner and the task care about.

**Intuition.**

- Two environments that the data cannot tell apart, but that this learner would treat differently, force an error on one of them; what counts is the learner's sensitivity $`c`$, not the size of the environment difference.

**Role in the framework.**

- It is the lower half of Theorem A and the quantitative core of "acquire before compiling"; aliasing ([Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7)) and the exchange rate ([Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8)) follow from it.

**Derivation outline.**

- The distiller's own contribution $`\sum_t\langle g_t,\mu^Q_t(z_t)\rangle`$ does not depend on $`\theta`$, so by the triangle inequality the errors in the two worlds sum to at least $`\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$.
- Le Cam's two-point argument: on the overlap of the two transcript laws no distiller can be right in both worlds, so the sum of expected errors is at least the gap times $`1-\mathrm{TV}`$.
- Pinsker's inequality and the chain rule for KL along the adaptive transcript: demonstrations contribute $`K\,\mathrm{KL}_D`$, each query at most $`\sup_x\mathrm{KL}_Q(x)`$, and the query choices contribute nothing because they follow the same rule in both worlds.
- Locally, KL is quadratic in the displacement with the Fisher informations, and the choice $`\tau=\mathbb I^{-1/2}`$ keeps the total variation at most $`1/2`$, which yields the constant $`1/4`$.

**Illustration.** With one parameter, $`\Psi(\theta)=c\theta`$, demonstrations of Fisher information $`i_D`$ each and queries of information $`i_Q`$ each, $`\mathfrak n_{K,n}=\lvert c\rvert/\sqrt{Ki_D+ni_Q}`$. A learner with a larger sensitivity $`\lvert c\rvert`$ needs proportionally more information for the same error.

**Uses.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2): the functional, the error and the norm.
- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1): Err is the first-order outcome gap.

**Used by.**

- [Theorem 6.5](../statements/06-05_achievability.md#s6-5): matching upper bound.
- [Remark 6.6](../statements/06-06_class_uniform_fidelity.md#s6-6): class-uniform version.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7): aliasing: total variation zero.
- [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8): exchange rate.
- [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10): explicit state-coverage bound.
- [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2): acquisition step of CED.

**Status.** proved; (b) and (c) in the local asymptotic minimax sense.

## References

- [Le Cam, 1973](../references.md#lecam1973convergence)
- [van der Vaart, 1998](../references.md#vandervaart1998asymptotic)
