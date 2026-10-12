# Corollary 6.12 · Learner separations

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-12_learner_separations.md) · [Proof](../proofs/06-12_learner_separations.md) · [Experiments](../experiments/06-12_learner_separations.md) · [中文](06-12_learner_separations_ZH.md)

**In one paragraph.** The same demonstrations can be sufficient for one learner and insufficient for another. Exactly, when one learner ignores a scale that the demonstrations do not reveal; quantitatively, when a long tail of rarely covered states matters more to warm Adam than to SGD.

**Setting used on this page.**

- **Environments.** A family $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown ([Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1)).
- **Oracle feedback** $`P^*_\theta`$: the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data.** $`K`$ i.i.d. demonstrations with law $`P^D_\theta`$, and $`n`$ queries that may be adaptive: a query point $`X_i`$ in the query space $`\mathcal X_Q`$ has outcome $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$. The law of the whole transcript is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from the transcript and private randomness (whose law does not depend on $`\theta`$) to a feedback law $`Q`$. Replay, rewriting and synthesis are distillers ([Definition 1.2](../README.md#s1-2)).
- **Audited states** $`z_1,\dots,z_T`$: fixed complete learner states, i.e. parameters together with the optimizer state ([Definition 1.3](../README.md#s1-3)), for example the states reference training reaches at checkpoints.
- **Mean increment** $`\mu^P_t(z)`$: the expected change of the complete learner state when one message from the law $`P`$ is consumed at $`z`$ ([Definition 1.3](../README.md#s1-3)). For the oracle, $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$, and the **learner Jacobian** is $`J_t=D_\theta\mu^*_t`$.
- **Task directions** $`g_t=\nabla V_{t+1}(z_t)`$: gradients of the continuation value of the declared reference tail ([Definition 1.4](../README.md#s1-4)); they do not depend on $`\theta`$.
- **Learner-relative functional** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$, with gradient $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$, and **first-order fidelity error** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$ ([Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2)). By [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) this is the first-order part of the gap in final task score between the oracle feedback and $`Q`$ at the audited states.
- **Agent states** $`s\in\mathcal S`$, a finite set; on this page $`s`$ denotes an agent state ([Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9)).
- **Per-state oracle laws.** At state $`s`$ the oracle message is $`Y\sim P^*_s`$ (for example the teacher's action or correction there), on a finite alphabet; the laws vary independently across states, so nothing is assumed to generalize between states.
- **Influence function** $`f_s(y)`$ of $`\Psi`$ at $`s`$: the derivative of $`\Psi`$ when $`P^*_s`$ is moved towards the point mass at $`y`$; it has mean zero under $`P^*_s`$.
- **Learner-relative weight** $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$: how much one more oracle sample at $`s`$ is worth for estimating $`\Psi`$.
- **Counts.** State $`s`$ receives $`N_s=k_s+n_s`$ oracle samples: $`k_s`$ from demonstrations (in expectation $`KH\,d_T(s)`$, with $`H`$ the trajectory length and $`d_T`$ the teacher's state-visitation distribution) and $`n_s`$ from queries, $`\sum_sn_s=n`$.

<a id="s6-12"></a>

**Corollary 6.12 (Learner separations).**

- **(a) Exact separation by scale invariance.** Let $`\theta=(\alpha,b)`$ with $`b>0`$, let the oracle consumed gradient be $`b\,G_\alpha(Y)`$ (for example a feedback weight or advantage scale), and let the demonstration law depend on $`\alpha`$ only.
  - For SGD with always-active norm clipping at level $`c_{\rm clip}`$ ($`\lVert bG_\alpha(Y)\rVert>c_{\rm clip}`$ almost surely), and for normalized SGD, the oracle mean increment does not depend on $`b`$, so $`\Psi`$ depends on $`\alpha`$ only and $`c(\theta)`$ has no component along $`b`$. If the demonstrations' Fisher information about $`\alpha`$ is non-degenerate, the error tends to zero as $`K\to\infty`$ ([Theorem 6.5](../statements/06-05_achievability.md#s6-5) with $`n=0`$, applied to the model in $`\alpha`$; realizability holds with any scale that keeps clipping active).
  - For plain SGD, $`c(\theta)`$ has the $`b`$-component $`-\eta\sum_t\langle g_t,\mathbb E\,G_\alpha\rangle`$; when it is nonzero, the demonstrations carry no information about $`b`$ and, by [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7), no $`K`$ suffices.
  - The same demonstrations are therefore sufficient for one learner and insufficient for the other. When clipping is active only with probability $`p_{\rm act}<1`$, [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9) puts each clipped mean consumed gradient within $`c_{\rm clip}(1-p_{\rm act})`$ of the $`b`$-free vector $`c_{\rm clip}\,\mathbb E[G_\alpha/\lVert G_\alpha\rVert]`$, so the separation degrades gracefully.
- **(b) Quantitative separation by long-tail coverage.** By [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10) and [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11)(d): when the student's state distribution has many low-frequency states, all in the frequent regime of warm Adam, and the teacher rarely reaches them, the share of the information requirement carried by those states is larger under warm Adam than under SGD, and warm Adam needs correspondingly more demonstrations or queries there for the same relative accuracy ([Example 6.13](../statements/06-13_long_tail_example.md#s6-13)).

**What it says.**

- Demonstrations that reveal a feedback direction but not its scale are sufficient for an always-clipped learner and insufficient, for every $`K`$, for SGD. Quantitatively, a long tail of rarely covered states costs warm Adam far more information than SGD.

**Intuition.**

- What a learner ignores, it does not need to be told; what it amplifies, it needs to be told precisely.

**Role in the framework.**

- It is the formal content of "the same demonstrations can be sufficient for one learner and not for another".

**Derivation outline.**

- (a) Always-active clipping removes the scale, so $`c`$ has no component along $`b`$; for SGD the scale moves $`\Psi`$ while the demonstration law stays fixed, which is [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7). With clipping active only with probability $`p_{\rm act}`$, [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9) bounds the remaining scale dependence by $`c_{\rm clip}(1-p_{\rm act})`$. (b) Combine [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10) with the weights of [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11).

**Uses.**

- [Theorem 6.5](../statements/06-05_achievability.md#s6-5): achievability for the clipped learner.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7): aliasing for SGD.
- [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10): explicit information formula.
- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11): learner weights.
- [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9): partially active clipping.

**Used by.**

- [Example 6.13](../statements/06-13_long_tail_example.md#s6-13): worked example of (b).
- [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3): design rule: match the learner.

**Status.** proved; (b) to leading order in the stated regime.
