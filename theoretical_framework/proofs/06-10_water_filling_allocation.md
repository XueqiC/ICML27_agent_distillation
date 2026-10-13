# Proposition 6.10 · Explicit information bound and optimal query allocation by water-filling

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-10_water_filling_allocation.md) · [Proof](../proofs/06-10_water_filling_allocation.md) · [Experiments](../experiments/06-10_water_filling_allocation.md) · [中文](06-10_water_filling_allocation_ZH.md)

**What is proved here.** The lower bound $`\tfrac14(\sum_sW_s/N_s)^{1/2}`$, the matching upper bound of the per-state empirical law, the water-filling allocation and the irreducible part.

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

**Statement ([Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10)).** Under [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9) let $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$, the **learner-relative weight** of state $`s`$.

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

**Results used in the proof.**

- [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9): the state-coverage model.
- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3): the two-point argument, applied along a one-dimensional tilting submodel.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7): for part (d).

**Classical tools.** Cauchy–Schwarz, the KKT conditions of a convex program, and one-dimensional submodels through the influence function, the standard device for information bounds on functionals ([van der Vaart, 1998](#ref-vandervaart1998asymptotic)).

**Proof.** (a) Fix $`h\in\mathbb R^{\mathcal S}`$ and use the tilting submodel $`P_{s,\tau}=(1+\tau h_sf_s)\,P^*_s`$, a probability law for $`\lvert\tau\rvert`$ small because $`f_s`$ is bounded on a finite alphabet and has mean zero.

- Along the submodel, $`\frac{\mathrm d}{\mathrm d\tau}\Psi=\sum_s\int f_s\,h_sf_s\,\mathrm dP^*_s=\sum_sh_sW_s`$, by the definition of the influence function.
- One sample at $`s`$ has score $`h_sf_s`$ and Fisher information $`h_s^2W_s`$, so the transcript has information $`\sum_sN_sh_s^2W_s`$ for the scalar parameter $`\tau`$; the KL expansion is uniform because the scores are bounded.

The proof of [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3) uses only the chain rule for KL, which applies to fixed per-state counts. It gives the lower bound $`\tfrac14\lvert\sum_sh_sW_s\rvert/(\sum_sN_sh_s^2W_s)^{1/2}(1-o(1))`$. By Cauchy–Schwarz,

```math
\Big(\sum_sh_sW_s\Big)^2=\Big(\sum_s h_s\sqrt{N_sW_s}\cdot\sqrt{W_s/N_s}\Big)^2\le\Big(\sum_sN_sh_s^2W_s\Big)\Big(\sum_sW_s/N_s\Big),
```

with equality at $`h_s=1/N_s`$, which gives (a).

(b) Let $`\hat P_s`$ be the empirical law of the $`N_s`$ samples at $`s`$. A second-order expansion gives $`\Psi(\hat P)-\Psi(P^*)=\sum_s\frac1{N_s}\sum_{i\le N_s}f_s(Y_{s,i})+\mathrm{Rem}`$ with $`\mathbb E\lvert\mathrm{Rem}\rvert\le C\sum_s\mathbb E\lVert\hat P_s-P^*_s\rVert^2=O(\sum_s1/N_s)=O(1/N)`$. The linear term has mean zero and variance $`\sum_sW_s/N_s`$, so its expected absolute value is at most the square root of that by Jensen's inequality, and $`O(1/N)=o\big((\sum_sW_s/N_s)^{1/2}\big)`$ under proportional growth. The output $`Q=\hat P`$ is realizable by replay.

(c) The objective is convex on $`n_s\ge0`$ and strictly convex in the coordinates with $`W_s>0`$. The KKT conditions are $`W_s/(k_s+n_s)^2=\lambda`$ when $`n_s>0`$ and $`W_s/k_s^2\le\lambda`$ when $`n_s=0`$, which is the stated formula; the left side of the budget equation is continuous and strictly decreasing in $`\lambda`$ on the active range, so $`\lambda`$ is unique.

(d) No observation depends on $`P^*_s`$, while $`\Psi`$ does ($`W_s>0`$). Two oracle laws that differ only at $`s`$ have identical transcript laws and different $`\Psi`$, so [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7) applies. $`\square`$

## Related work

- **Active acquisition in imitation learning.** DAgger ([Ross, Gordon & Bagnell, 2011](../references.md#ross2011reduction)) queries the teacher at the states the student visits; in the terms of this proposition it allocates queries in proportion to student visitation. The information-optimal allocation is water-filling in the learner-relative weights, and these weights change with the optimizer ([Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11)).
- **Allocation rule.** Without teacher coverage the optimum $`n_s\propto\sqrt{W_s}`$ is Neyman's allocation for stratified sampling ([Neyman, 1934](#ref-neyman1934two)); with coverage $`k_s`$ it is water-filling, the solution shape of power allocation over parallel Gaussian channels ([Cover & Thomas, 2006](#ref-cover2006elements)).

## References

- [Ross, Gordon & Bagnell, 2011](../references.md#ross2011reduction)
