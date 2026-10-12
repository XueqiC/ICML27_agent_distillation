# Corollary 6.8 · Exchange rate between demonstrations and queries

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-08_exchange_rate_between_demonstrations_queries.md) · [Proof](../proofs/06-08_exchange_rate_between_demonstrations_queries.md) · [中文](06-08_exchange_rate_between_demonstrations_queries_ZH.md)

**In one paragraph.** Demonstrations and queries draw on one information budget, direction by direction. At the binding direction one demonstration is worth a definite number of queries, and that number is set by the learner.

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

<a id="s6-8"></a>

**Corollary 6.8 (Exchange rate between demonstrations and queries).** For a direction $`h`$ write $`e_D(h)=h^\top I_D(\theta_0)h`$ and $`e_Q(h)=\sup_xh^\top I_Q(\theta_0;x)h`$. Under the conditions of [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)(b), a distiller whose local worst-case expected first-order error is at most $`\varepsilon`$ must satisfy, for every $`h`$,

```math
K\,e_D(h)+n\,e_Q(h)\ge\frac{(c(\theta_0)^\top h)^2}{16\,\varepsilon^2}\,\big(1-o(1)\big).
```

- The feasible set of $`(K,n)`$ is an intersection of half-planes. At a boundary point the binding direction $`h^\star`$ fixes the local exchange rate: one demonstration is worth $`e_D(h^\star)/e_Q(h^\star)`$ queries.
- $`h^\star`$ is determined by the learner's $`c`$. The exchange rate is therefore learner-relative: the same demonstrations are worth different numbers of queries to different learners.
- [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11) is the analogous linear budget for testing between two worlds; here the budget holds direction by direction, weighted by the learner.

**What it says.**

- Demonstrations and queries enter one linear information budget direction by direction; at the binding direction one demonstration is worth a definite number of queries, and that number depends on the learner.

**Intuition.**

- A demonstration and a query are both evidence; their relative value is how much each reveals in the direction this learner cares about.

**Role in the framework.**

- It quantifies the trade-off between collecting more demonstrations and spending queries; it is the learner-relative form of [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11).

**Derivation outline.**

- Rearrange [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)(b) direction by direction.

**Uses.**

- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3): local lower bound, direction by direction.
- [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11): the two-world information budget.

**Used by.**

- [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3): design rule: acquire before compiling.

**Status.** proved.
