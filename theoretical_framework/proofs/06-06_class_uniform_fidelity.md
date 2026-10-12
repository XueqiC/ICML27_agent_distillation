# Remark 6.6 · Class-uniform fidelity

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-06_class_uniform_fidelity.md) · [Proof](../proofs/06-06_class_uniform_fidelity.md) · [中文](06-06_class_uniform_fidelity_ZH.md)

**What is proved here.** For class-uniform fidelity the lower bound is $`\tfrac14\sup_V\mathfrak n_{K,n}(c_V)`$ and a mixture of c-optimal designs achieves $`d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)(1+o(1))`$.

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
- **Designs.** An approximate design $`\xi`$ is a probability measure on the query space; it spreads the $`n`$ queries over query points. $`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$ is the average query information and $`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$ the total information of demonstrations plus queries.

**Statement ([Remark 6.6](../statements/06-06_class_uniform_fidelity.md#s6-6)).** When fidelity is required uniformly over a declared class $`\mathcal V`$ of continuation values, each $`V\in\mathcal V`$ gives directions $`g^V_t=\nabla V_{t+1}(z_t)`$, a functional $`\Psi_V`$ and a gradient $`c_V`$; the error is $`\mathrm{Err}^{\mathcal V}_\theta(Q)=\sup_{V\in\mathcal V}\lvert\Psi_V(\theta)-\sum_t\langle g^V_t,\mu^Q_t(z_t)\rangle\rvert`$.

- **Lower bound.** The proof of [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3) goes through with $`\lvert c^\top h\rvert`$ replaced by $`\sigma(h)=\sup_V\lvert c_V^\top h\rvert`$, since $`\sup_V\lvert a_V\rvert+\sup_V\lvert b_V\rvert\ge\sup_V\lvert a_V-b_V\rvert`$. Exchanging the two suprema, the local lower bound is $`\tfrac14\sup_V\mathfrak n_{K,n}(c_V)`$.
- **Upper bound.** Let $`d`$ be the dimension of $`\mathrm{span}\lbrace c_V\rbrace`$ ($`d\le\dim\mathrm{span}\,\mathcal G`$). Under the analogue of (E) for $`d`$ linear functionals, the plug-in distiller with a suitable mixture design has error at most $`d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)\,(1+o(1))`$.
- So the lower and upper bounds differ by at most the factor $`4d^{3/2}`$, which depends only on the dimension of the declared class. Sharper constants are not claimed.

**Results used in the proof.**

- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3): the two-point argument, which goes through for a supremum of absolute values.
- [Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4): c-optimal designs for each axis $`\ell_k`$.
- [Theorem 6.5](../statements/06-05_achievability.md#s6-5): the second-order term.

**Classical tools.** John's theorem on maximal-volume ellipsoids ([John, 1948](../references.md#john1948extremum)).

**Proof of the upper bound.** Let $`\mathcal K`$ be the closed convex hull of $`\lbrace\pm c_V\rbrace`$, a symmetric convex body in a $`d`$-dimensional subspace. By John's theorem there is an ellipsoid $`\mathcal E=\lbrace\sum_kw_k\ell_k:\lVert w\rVert\le1\rbrace\subseteq\mathcal K\subseteq\sqrt d\,\mathcal E`$, so $`\sigma(u)=\sup_{v\in\mathcal K}v^\top u\le\sqrt d\,\big(\sum_{k=1}^d(\ell_k^\top u)^2\big)^{1/2}`$. Let $`\xi_k`$ be c-optimal for $`\ell_k`$ and $`\bar\xi=\tfrac1d\sum_k\xi_k`$. Then $`A(\bar\xi)\succeq\tfrac1dA(\xi_k)`$ for each $`k`$, so $`\ell_k^\top A(\bar\xi)^{-1}\ell_k\le d\,\mathfrak n_{K,n}(\ell_k)^2`$. Since $`\ell_k\in\mathcal K`$ and $`\mathfrak n_{K,n}(\cdot)`$ is a seminorm of its vector argument, $`\mathfrak n_{K,n}(\ell_k)\le\sup_V\mathfrak n_{K,n}(c_V)`$. By Jensen's inequality,

```math
\mathbb E\,\sigma(\hat\theta-\theta)\le\sqrt d\,\Big(\sum_{k=1}^d\ell_k^\top A(\bar\xi)^{-1}\ell_k\Big)^{1/2}(1+o(1))\le d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)\,(1+o(1)),
```

and the second-order term is handled as in [Theorem 6.5](../statements/06-05_achievability.md#s6-5). $`\square`$

## References

- [John, 1948](../references.md#john1948extremum)
