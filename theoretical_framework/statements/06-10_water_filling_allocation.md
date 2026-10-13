# Proposition 6.10 · Explicit information bound and optimal query allocation by water-filling

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-10_water_filling_allocation.md) · [Proof](../proofs/06-10_water_filling_allocation.md) · [Experiments](../experiments/06-10_water_filling_allocation.md) · [中文](06-10_water_filling_allocation_ZH.md)

**In one paragraph.** In the state-coverage model the learner-relative information norm has an explicit form, $`(\sum_sW_s/N_s)^{1/2}`$ up to a factor between $`1/4`$ and $`1`$, and the optimal way to spend queries is water-filling in the learner-relative weights.

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

<a id="s6-10"></a>

**Proposition 6.10 (Explicit information bound and optimal query allocation by water-filling).** Under [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9) let $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$, the **learner-relative weight** of state $`s`$.

- **(a) Lower bound.** Let the counts grow proportionally ($`N_s=a_sN`$, $`N\to\infty`$) on the states with $`W_s>0`$. For every distiller the local minimax first-order error is at least

```math
\tfrac14\Big(\sum_{s}\frac{W_s}{N_s}\Big)^{1/2}\big(1-o(1)\big).
```

- **(b) Upper bound.** Assume that $`\Psi`$ is twice continuously differentiable in the probability vectors $`(P_s)_s`$. The per-state empirical law, that is, replaying the acquired messages at each state, achieves

```math
\mathbb E\lvert\hat\Psi-\Psi\rvert\le\Big(\sum_s\frac{W_s}{N_s}\Big)^{1/2}\big(1+o(1)\big).
```

- **(c) Optimal queries (water-filling).** The minimizer of $`\sum_sW_s/(k_s+n_s)`$ over $`n_s\ge0`$ with $`\sum_sn_s=n`$ (continuous relaxation) is

```math
n_s^\star=\Big(\sqrt{W_s/\lambda}-k_s\Big)_+,\qquad \text{with }\lambda>0\text{ fixed by }\textstyle\sum_sn_s^\star=n .
```

  Queries flow to states whose learner-relative weight is high and whose teacher coverage is low.

- **(d) Irreducible part.** A state with $`W_s>0`$ and $`k_s=n_s=0`$, one the student reaches but the teacher does not and that is never queried, leaves an error floor that no $`K`$ removes ([Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7)).

**What it says.**

- When demonstrations cover teacher states and training happens at student states, the error is governed by $`\sum_sW_s/N_s`$, with $`W_s`$ the learner-relative weight of state $`s`$ and $`N_s`$ its sample count. The optimal query allocation is water-filling, $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$; a state with positive weight and no coverage leaves an irreducible error.

**Intuition.**

- Queries should go where the learner is sensitive and the teacher has not been.

**Role in the framework.**

- It makes Theorem A operational for agent distillation and gives the acquisition step of the method its allocation rule ([Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2)).

**Derivation outline.**

- (a) A tilting submodel $`(1+\tau h_sf_s)P^*_s`$ has derivative $`\sum_sh_sW_s`$ and Fisher information $`\sum_sN_sh_s^2W_s`$; [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3) and Cauchy–Schwarz, with the optimal tilt $`h_s=1/N_s`$, give the bound.
- (b) The plug-in error is a sum of independent per-state sample means of the influence function plus a remainder of order $`1/N`$.
- (c) The objective $`\sum_sW_s/(k_s+n_s)`$ is convex; the KKT conditions equalize the marginal gain $`W_s/(k_s+n_s)^2`$ across queried states.
- (d) Nothing observed depends on $`P^*_s`$, so [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7) applies.

**Illustration.** Two states with equal teacher coverage, one with weight four times the other: water-filling gives the heavier state twice as many total samples ($`\sqrt4=2`$), not four times as many.

**Uses.**

- [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9): state-coverage model.
- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3): two-point argument along a tilting submodel.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7): irreducible part.

**Used by.**

- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11): weights to insert into the formula.
- [Corollary 6.12](../statements/06-12_learner_separations.md#s6-12): long-tail separation.
- [Example 6.13](../statements/06-13_long_tail_example.md#s6-13): worked example.
- [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2): acquisition step of CED.

**Status.** proved.

## Related work

- Without teacher coverage ($`k_s=0`$) the optimum $`n_s\propto\sqrt{W_s}`$ is Neyman's allocation for stratified sampling ([Neyman, 1934](#ref-neyman1934two)). With coverage it becomes water-filling, the solution shape of power allocation over parallel Gaussian channels ([Cover & Thomas, 2006](#ref-cover2006elements)).
- DAgger queries the teacher at the states the student visits, in proportion to student visitation ([Ross et al., 2011](#ref-ross2011reduction)); the information-optimal allocation follows the learner-relative weights and the existing coverage instead.

## References

- [Ross, Gordon & Bagnell, 2011](../references.md#ross2011reduction)
