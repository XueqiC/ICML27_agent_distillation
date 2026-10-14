# Lemma 6.4 · Design minimax equality

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-04_design_minimax_equality.md) · [Proof](../proofs/06-04_design_minimax_equality.md) · [Experiments](../experiments/06-04_design_minimax_equality.md) · [中文](06-04_design_minimax_equality_ZH.md)

**In one paragraph.** The best variance that any query design can achieve for the learner-relative functional equals the squared information norm, and an optimal design exists: the classical c-optimal (Elfving) design, with a target vector set by the learner and the task. No design has to identify every parameter direction. The equality holds with a pseudo-inverse whenever the directions that no demonstration and no query informs do not affect the target, and the value is infinite exactly when one of them does.

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
- **Singular information.** $`I_D(\theta)`$ and every $`I_Q(\theta;x)`$ are positive semidefinite, so $`A(\xi)`$ is positive semidefinite and may be singular. For a positive semidefinite $`A`$, $`A^+`$ is its Moore–Penrose pseudo-inverse and $`\mathrm{range}(A)=(\ker A)^\perp`$.

<a id="s6-4"></a>

**Lemma 6.4 (Design minimax equality).** Assume that $`\mathcal X_Q`$ is a compact metric space and that $`x\mapsto I_Q(\theta;x)`$ is continuous. Let $`\xi`$ range over the Borel probability measures on $`\mathcal X_Q`$ (approximate designs), put $`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$ and $`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$; no design is required to make $`A(\xi)`$ invertible. For a positive semidefinite matrix $`A`$ put

```math
\Phi_c(A)=\begin{cases}c^\top A^+c, & c\in\mathrm{range}(A),\\ +\infty, & c\notin\mathrm{range}(A).\end{cases}
```

Then, with the conventions $`0/0=0`$ and $`a/0=+\infty`$ for $`a>0`$,

```math
\inf_\xi\Phi_c\big(A(\xi)\big)=\sup_{h\neq0}\frac{(c^\top h)^2}{K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h}=\mathfrak n_{K,n}^2 ,
```

and the infimum is attained by some design $`\xi^*`$, the **c-optimal design** for the target vector $`c=\sum_tJ_t^*g_t`$. Moreover:

- **(a) Invertible designs.** If $`A(\xi)\succ0`$ for every $`\xi`$, then $`\Phi_c(A(\xi))=c^\top A(\xi)^{-1}c`$, and the identity is the classical c-optimal design equality.
- **(b) When the value is finite.** Call $`h`$ *uninformed* if $`K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h=0`$. The common value is finite if and only if $`c^\top h=0`$ for every uninformed $`h`$. In that case $`c\in\mathrm{range}\,A(\xi^*)`$, and $`\xi^*`$ may leave every uninformed direction unidentified.
- **(c) No queries.** For $`n=0`$ and $`K>0`$ the value is $`c^\top(KI_D)^+c`$ when $`c\in\mathrm{range}(I_D)`$, and $`+\infty`$ otherwise.

**What it says.**

- The best variance any query design can achieve for estimating the learner-relative functional equals the square of the information norm, and an optimal design exists. It is the c-optimal (Elfving) design whose target vector is the learner Jacobian applied to the continuation-value gradients.
- The designs may be singular. Directions of $`\theta`$ that the learner's target ignores ($`c^\top h=0`$) never need to be identified; the value becomes infinite only when the data cannot inform some direction that the target does use.

**Intuition.**

- The best way to spend queries is the one that is most informative about the single direction the learner will use; this is a classical design problem with a learner-specific target.
- Paying to identify nuisance directions is wasted for a linear target: the pseudo-inverse simply ignores them.

**Role in the framework.**

- It turns the lower bound into an acquisition rule and makes the upper bound computable ([Theorem 6.5](../statements/06-05_achievability.md#s6-5)).
- It is the scalar-target case of the exact linear-Gaussian design calculus of [Proposition 6.15](../statements/06-15_linear_gaussian_target.md#s6-15).
- The relaxation concerns the linear functional only. [Theorem 6.5](../statements/06-05_achievability.md#s6-5) still regularizes the design so that every direction is eventually identified, because its plug-in error has a second-order remainder in all directions; that condition is not removed here.

**Derivation outline.**

- For every positive semidefinite $`A`$, $`\Phi_c(A)=\sup_h\big(2c^\top h-h^\top Ah\big)`$; optimizing the scale of $`h`$ turns this into the supremum of $`(c^\top h)^2/h^\top Ah`$ with the stated conventions. No inverse is needed.
- The function $`2c^\top h-h^\top A(\xi)h`$ is affine and weakly continuous in the design and concave in the direction, and the designs form a weakly compact convex set, so Sion's minimax theorem exchanges the two optimizations.
- For a fixed direction, the best design puts all its mass on the most informative query point, which produces the $`\sup_x`$ term. The value $`\xi\mapsto\Phi_c(A(\xi))`$ is a supremum of continuous functions, hence lower semicontinuous, so it attains its infimum on the compact design set.

**Uses.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2): definition of the norm.

**Used by.**

- [Theorem 6.5](../statements/06-05_achievability.md#s6-5): the design that attains the norm.
- [Remark 6.6](../statements/06-06_class_uniform_fidelity.md#s6-6): mixtures of c-optimal designs.
- [Proposition 6.15](../statements/06-15_linear_gaussian_target.md#s6-15): the scalar-target case of the linear-Gaussian risk.

**Status.** proved.

## Related work

- c-optimal designs and their geometric characterization go back to [Elfving, 1952](#ref-elfving1952optimum); the general theory, including designs with prior information such as the demonstrations' term $`K\,I_D`$ and singular information matrices with the estimability condition $`c\in\mathrm{range}\,A`$, is in [Pukelsheim, 1993](#ref-pukelsheim1993optimal). The exchange of the two optimizations is Sion's minimax theorem ([Sion, 1958](#ref-sion1958general)). Allowing singular designs is therefore a direct extension of classical design duality, not a new design theory.
- What is specific here is the target vector $`c=\sum_tJ_t^*g_t`$: it is set by the learner and the task.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).

<a id="ref-sion1958general"></a>

**Maurice Sion (1958).** On general minimax theorems. *Pacific Journal of Mathematics* 8, no. 1, 171–176. DOI [10.2140/pjm.1958.8.171](https://doi.org/10.2140/pjm.1958.8.171).
