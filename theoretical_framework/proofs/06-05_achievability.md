# Theorem 6.5 · Achievability; tight to constants in the scalar case

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-05_achievability.md) · [Proof](../proofs/06-05_achievability.md) · [Experiments](../experiments/06-05_achievability.md) · [中文](06-05_achievability_ZH.md)

**What is proved here.** The plug-in distiller with a regularized c-optimal design has expected first-order error at most $`\mathfrak n_{K,n}(1+o(1))+o(N^{-1/2})`$.

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

**Statement ([Theorem 6.5](../statements/06-05_achievability.md#s6-5)).** Assume (G), the regularity of [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)(b), and:

- **(R) Realizability.** For every $`\theta'\in\Theta`$ the distiller can emit the oracle law $`P^*_{\theta'}`$ at the audited states, for example by running a simulator or the teacher model at $`\theta'`$, or by reweighting messages obtained from queries.
- **(I) Identifiability.** Some design $`\xi_0`$ has $`\lambda_{\min}(M(\xi_0))=\lambda_0>0`$.
- **(S) Smoothness.** $`\Psi\in C^2`$ near $`\theta_0`$ with $`\lVert\nabla^2\Psi\rVert\le H_\Psi`$.
- **Design.** Use the regularized c-optimal design $`\xi_n=(1-\varepsilon_n)\xi^*+\varepsilon_n\xi_0`$, with $`\xi^*`$ from [Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4), $`\varepsilon_n\to0`$ and $`\varepsilon_n\sqrt n\to\infty`$ (for example $`\varepsilon_n=n^{-1/4}`$).
- **(E) Efficient two-stage estimation.** $`\xi_n`$ is computed at a consistent pilot estimate obtained from $`o(n)`$ queries and rounded to integer counts. The final estimator $`\hat\theta`$ satisfies, uniformly over $`\theta`$ in a shrinking neighbourhood of $`\theta_0`$ and with $`\xi_n`$ evaluated at the true $`\theta`$,

```math
\mathbb E_\theta\big[(c(\theta)^\top(\hat\theta-\theta))^2\big]\le c(\theta)^\top A(\xi_n)^{-1}c(\theta)\,(1+o(1)),\qquad \mathbb E_\theta\lVert\hat\theta-\theta\rVert^2\le\frac{C}{\lambda_{\min}(A(\xi_n))} .
```

Let the distiller output $`Q=P^*_{\hat\theta}`$. Then

```math
\mathbb E_\theta\mathrm{Err}_\theta(Q)\le\mathfrak n_{K,n}(\theta)\,\big(1+o(1)\big)+o\big(N^{-1/2}\big),
```

and the last term is $`o(\mathfrak n_{K,n}(\theta))`$ whenever $`c(\theta)\neq0`$. Together with [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)(c), the smallest achievable first-order error is, in the local minimax sense, between $`\tfrac14\mathfrak n_{K,n}`$ and $`\mathfrak n_{K,n}`$ up to factors $`1\pm o(1)`$. The optimal queries follow the c-optimal design whose target vector is $`c=\sum_tJ_t^*g_t`$.

**Results used in the proof.**

- [Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4): $`c^\top A(\xi^*)^{-1}c=\mathfrak n_{K,n}^2`$ and the existence of $`\xi^*`$.
- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3): part (c), the matching lower bound.
- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2): the growth condition (G).

**Classical tools.** Jensen's inequality and a second-order Taylor bound.

**Proof.** By (R) the output $`P^*_{\hat\theta}`$ is available, and $`\mathrm{Err}_\theta(P^*_{\hat\theta})=\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert`$. By (S),

```math
\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert\le\lvert c(\theta)^\top(\hat\theta-\theta)\rvert+\tfrac12H_\Psi\lVert\hat\theta-\theta\rVert^2 .
```

- *Linear term.* $`A(\xi_n)=KI_D+n(1-\varepsilon_n)M(\xi^*)+n\varepsilon_nM(\xi_0)\succeq(1-\varepsilon_n)A(\xi^*)`$, hence $`c^\top A(\xi_n)^{-1}c\le c^\top A(\xi^*)^{-1}c/(1-\varepsilon_n)=\mathfrak n^2_{K,n}(1+o(1))`$ by [Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4). By Jensen's inequality and (E), $`\mathbb E\lvert c^\top(\hat\theta-\theta)\rvert\le\mathfrak n_{K,n}(1+o(1))`$.
- *Quadratic term.* By (E) it is at most $`\tfrac12H_\Psi C/\lambda_{\min}(A(\xi_n))`$, and $`\lambda_{\min}(A(\xi_n))\ge\max\lbrace K\lambda_{\min}(I_D),\,n\varepsilon_n\lambda_0\rbrace`$. This is $`\gg\sqrt N`$:
  - if $`n\ge\delta N`$ for some $`\delta>0`$, then $`n\varepsilon_n\gg\sqrt n\ge\sqrt{\delta N}`$;
  - otherwise $`n=o(N)`$ along a subsequence, so for $`h\in\ker I_D`$ we would have $`\mathbb I_{K,n}(h)\le n\sup_xh^\top I_Qh=o(N)`$, contradicting (G); hence $`I_D\succ0`$, $`K`$ grows like $`N`$, and $`K\lambda_{\min}(I_D)\asymp N`$.

  So the quadratic term is $`o(N^{-1/2})`$.
- *Comparison.* If $`c(\theta)\neq0`$, (G) gives $`\mathfrak n_{K,n}\ge\lVert c\rVert/\sqrt{bN}`$, so $`o(N^{-1/2})=o(\mathfrak n_{K,n})`$. $`\square`$

**Comments.**

- The regularization $`\varepsilon_n`$ matters: c-optimal designs are often supported on few query points, so $`A(\xi^*)`$ can be singular in directions that $`c`$ does not see, while the second-order term of $`\Psi`$ still needs $`\hat\theta`$ to be consistent in every direction.
- In a Gaussian linear experiment, $`\Psi`$ is linear, the efficient estimator is exactly Gaussian, and the plug-in error is exactly $`\sqrt{2/\pi}\,\sqrt{c^\top A^{-1}c}`$, about $`0.80\,\mathfrak n_{K,n}`$ at the optimal design.

## Related work

- **Regularized designs and two-stage estimation** are standard in optimal design ([Pukelsheim, 1993](#ref-pukelsheim1993optimal)) and in local asymptotic theory ([van der Vaart, 1998](#ref-vandervaart1998asymptotic)); the regularization is needed here because the learner-relative c-optimal design is typically singular.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).

<a id="ref-vandervaart1998asymptotic"></a>

**A. W. van der Vaart (1998).** Asymptotic Statistics. Cambridge University Press, Cambridge (Cambridge Series in Statistical and Probabilistic Mathematics 3). DOI [10.1017/CBO9780511802256](https://doi.org/10.1017/CBO9780511802256).
