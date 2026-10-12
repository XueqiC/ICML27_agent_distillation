# Assumption 6.9 · State-coverage model

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-09_state_coverage_model.md) · [中文](06-09_state_coverage_model_ZH.md)

**In one paragraph.** This assumption specializes Theorem A to the structure that dominates agent distillation: demonstrations cover the states the teacher visits, while training and deployment happen at the states the student visits. Each agent state has its own oracle message law, and information about it comes only from teacher visits and queries at that state.

**Setting used on this page.**

- **Environments.** A family $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown ([Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1)).
- **Oracle feedback** $`P^*_\theta`$: the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data.** $`K`$ i.i.d. demonstrations with law $`P^D_\theta`$, and $`n`$ queries that may be adaptive: a query point $`X_i`$ in the query space $`\mathcal X_Q`$ has outcome $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$. The law of the whole transcript is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from the transcript and private randomness (whose law does not depend on $`\theta`$) to a feedback law $`Q`$. Replay, rewriting and synthesis are distillers ([Definition 1.2](../README.md#s1-2)).
- **Audited states** $`z_1,\dots,z_T`$: fixed complete learner states, i.e. parameters together with the optimizer state ([Definition 1.3](../README.md#s1-3)), for example the states reference training reaches at checkpoints.
- **Mean increment** $`\mu^P_t(z)`$: the expected change of the complete learner state when one message from the law $`P`$ is consumed at $`z`$ ([Definition 1.3](../README.md#s1-3)). For the oracle, $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$, and the **learner Jacobian** is $`J_t=D_\theta\mu^*_t`$.
- **Task directions** $`g_t=\nabla V_{t+1}(z_t)`$: gradients of the continuation value of the declared reference tail ([Definition 1.4](../README.md#s1-4)); they do not depend on $`\theta`$.
- **Learner-relative functional** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$, with gradient $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$, and **first-order fidelity error** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$ ([Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2)). By [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) this is the first-order part of the gap in final task score between the oracle feedback and $`Q`$ at the audited states.

<a id="s6-9"></a>

**Assumption 6.9 (State-coverage model).**

- **States and laws.** A finite set $`\mathcal S`$ of agent states. At state $`s`$ the oracle message is $`Y\sim P^*_s`$, for example the teacher's action or correction at $`s`$, on a finite message alphabet. The laws $`(P^*_s)_{s\in\mathcal S}`$ vary independently across states: no generalization across states is assumed.
- **Functional.** The learner-relative functional $`\Psi`$ is a Hadamard-differentiable functional of $`(P^*_s)_{s\in\mathcal S}`$. Its influence function at $`s`$ is $`f_s(y)=\frac{\mathrm d}{\mathrm d\epsilon}\Psi\big(P^*_s+\epsilon(\delta_y-P^*_s)\big)\big\vert_{\epsilon=0}`$, so $`\mathbb E_{P^*_s}f_s=0`$.
- **Counts.** State $`s`$ receives $`N_s=k_s+n_s`$ oracle samples: $`k_s`$ from demonstrations (in expectation $`KH\,d_T(s)`$, with $`H`$ the trajectory length and $`d_T`$ the teacher's state-visitation distribution) and $`n_s`$ from queries, $`\sum_sn_s=n`$. Bounds are stated for given counts; with random counts they hold conditionally on them.

**What it is.**

- No generalization between states is assumed: a sample at one state says nothing about another.
- The learner-relative functional depends smoothly on the per-state laws; its influence function $`f_s`$ measures how one more sample at $`s`$ moves it.
- Demonstration samples arrive at the teacher's visitation rate, query samples where the distiller chooses.

**Role in the framework.**

- It makes the information norm explicit ([Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10)) and lets the learner weights be computed ([Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11)).

**Uses.**

- [Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1): environments and data.
- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2): the learner-relative functional.

**Used by.**

- [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10): explicit bound and water-filling.
- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11): learner weights.
- [Example 6.13](../statements/06-13_long_tail_example.md#s6-13): long-tail example.

**Status.** assumption.
