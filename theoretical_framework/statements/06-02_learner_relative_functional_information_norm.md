# Definition 6.2 · Learner-relative functional and information norm

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-02_learner_relative_functional_information_norm.md) · [中文](06-02_learner_relative_functional_information_norm_ZH.md)

**In one paragraph.** This definition turns "how well does a distiller reproduce the oracle feedback, for this learner and this task" into one scalar functional $`\Psi`$, and measures how much the data reveal about it by the learner-relative information norm $`\mathfrak n_{K,n}`$.

**Setting used on this page.**

- **Environments.** A family $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown ([Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1)).
- **Oracle feedback** $`P^*_\theta`$: the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data.** $`K`$ i.i.d. demonstrations with law $`P^D_\theta`$, and $`n`$ queries that may be adaptive: a query point $`X_i`$ in the query space $`\mathcal X_Q`$ has outcome $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$. The law of the whole transcript is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from the transcript and private randomness (whose law does not depend on $`\theta`$) to a feedback law $`Q`$. Replay, rewriting and synthesis are distillers ([Definition 1.2](../README.md#s1-2)).
- **Mean increment** $`\mu^P_t(z)`$: the expected change of the complete learner state $`z`$ (parameters and optimizer state) when one message from the law $`P`$ is consumed ([Definition 1.3](../README.md#s1-3)).
- **Continuation value** $`V_{t+1}`$: the expected final task score when training continues from a state under the declared reference tail ([Definition 1.4](../README.md#s1-4)).
- **First-order representation** ([Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1)): to first order, the effect of one round of feedback on the final score is the inner product of $`\nabla V_{t+1}`$ with the mean increment.

<a id="s6-2"></a>

**Definition 6.2 (Learner-relative functional and information norm).**

- **Learner Jacobian.** The oracle mean increment at an audited state is $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$. The learner Jacobian $`J_t(\theta)=D_\theta\mu^*_t(\theta)`$ is a linear map from $`\mathbb R^p`$ to the state space $`\mathcal Z`$. It depends on the environment, through $`P^*_\theta`$, and on the learner, through the increment map $`\Delta_t`$.
- **Learner-relative functional.** The declared task fixes the directions $`g_t=\nabla V_{t+1}(z_t)\in\mathcal Z`$, the continuation-value gradients of the reference tail at the audited states; they do not depend on $`\theta`$. Define

```math
\Psi(\theta)=\sum_{t=1}^{T}\big\langle g_t,\,\mu^*_t(\theta)\big\rangle,\qquad c(\theta)=\nabla_\theta\Psi(\theta)=\sum_{t=1}^{T}J_t(\theta)^{*}g_t\in\mathbb R^p .
```

- **First-order fidelity error.** For the output $`Q`$ of a distiller,

```math
\mathrm{Err}_\theta(Q)=\Big\lvert\Psi(\theta)-\sum_{t=1}^{T}\big\langle g_t,\,\mu^Q_t(z_t)\big\rangle\Big\rvert .
```

- **Remarks on the definition.**
  - The consumed gradient $`G_t(z,y)`$ and the increment map $`\Delta_t`$ depend only on the learner state and the message, so $`\mu^Q_t(z_t)`$ does not depend on $`\theta`$; the environment enters only through which messages the oracle law emits.
  - By [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1)(a) with $`k=1`$, $`\langle g_t,\mu^*_t(\theta)-\mu^Q_t(z_t)\rangle`$ is the first-order part of the effect of replacing one round of $`Q`$-feedback by oracle feedback at $`z_t`$, followed by the reference tail. $`\mathrm{Err}_\theta`$ adds these first-order effects over the audited states.
- **Information form and norm.** Let $`I_D(\theta)`$ be the Fisher information of one demonstration and $`I_Q(\theta;x)`$ that of one query at $`x\in\mathcal X_Q`$. For a direction $`h\in\mathbb R^p`$,

```math
\mathbb I_{K,n}(\theta;h)=K\,h^\top I_D(\theta)h+n\sup_{x\in\mathcal X_Q}h^\top I_Q(\theta;x)h,\qquad \mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\frac{\lvert c(\theta)^\top h\rvert}{\sqrt{\mathbb I_{K,n}(\theta;h)}},
```

  with the conventions $`0/0=0`$ and $`a/0=\infty`$ for $`a>0`$. $`\mathfrak n_{K,n}`$ is the **learner-relative information norm**: the dual norm of the learner-relative gradient $`c`$ with respect to the information gauge $`\sqrt{\mathbb I_{K,n}}`$.

- **Growth condition (G).** Some of the asymptotic statements below assume that the information grows non-degenerately: there are $`N\to\infty`$ and constants $`0<a\le b<\infty`$ with $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$ for all unit $`h`$. This holds, for example, when $`K`$ and $`n`$ grow proportionally and $`I_D+\sup_xI_Q(x)`$ is positive definite in every direction.

**What it is.**

- $`\Psi`$ adds, over the audited states, the inner product of the task's continuation-value gradient with the oracle mean increment: the first-order effect of the oracle feedback on the final task score.
- $`c=\nabla\Psi=\sum_tJ_t^*g_t`$ combines how the learner's mean increment moves with $`\theta`$ (the Jacobian $`J_t`$) with what the task rewards ($`g_t`$).
- $`\mathfrak n_{K,n}`$ is the dual norm of $`c`$ in the information gauge: the uncertainty the data leave about $`\Psi`$ along its least-informed direction.

**Role in the framework.**

- Every bound in §6 is stated through $`\mathfrak n_{K,n}`$; the growth condition (G) is the regularity under which the bounds are asymptotically tight.

**Uses.**

- [Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1): environments, data, distillers, audited states.
- [Definition 1.3](../README.md#s1-3): complete learner state and mean increment.
- [Definition 1.4](../README.md#s1-4): continuation value.
- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1): Err is the first-order outcome gap.

**Used by.**

- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3): lower bound in terms of the norm.
- [Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4): design value equals the squared norm.
- [Theorem 6.5](../statements/06-05_achievability.md#s6-5): achievability.
- [Remark 6.6](../statements/06-06_class_uniform_fidelity.md#s6-6): class-uniform version.
- [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8): exchange rate.
- [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9): state-coverage model.

**Status.** definition.

## Related work

- For $`n=0`$, $`\mathfrak n_{K,0}^2=c^\top(K\,I_D)^{-1}c`$ is the classical local asymptotic minimax bound on the variance of regular estimators of the smooth functional $`\Psi`$ ([van der Vaart, 1998](#ref-vandervaart1998asymptotic)); with queries the best design enters ([Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4)).
- $`\Psi`$ adds, over audited checkpoints, inner products of a task gradient with the learner's update, the structure of gradient-tracing influence estimates such as TracIn ([Pruthi et al., 2020](#ref-pruthi2020estimating)); here it is the quantity a distiller must reproduce, not a score for ranking data.
