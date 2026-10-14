# Lemma 6.4 · Design minimax equality

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-04_design_minimax_equality.md) · [Proof](../proofs/06-04_design_minimax_equality.md) · [Experiments](../experiments/06-04_design_minimax_equality.md) · [中文](06-04_design_minimax_equality_ZH.md)

**What is proved here.** The c-optimal design value $`\inf_\xi\Phi_c(A(\xi))`$ equals the squared information norm, without assuming that any design makes $`A(\xi)`$ invertible, and the infimum is attained. The proof replaces the inverse by a variational formula valid for every positive semidefinite matrix, exchanges the two optimizations by Sion's theorem, and then evaluates the inner problem in closed form.

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

**Statement ([Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4)).** Assume that $`\mathcal X_Q`$ is a compact metric space and that $`x\mapsto I_Q(\theta;x)`$ is continuous. Let $`\xi`$ range over the Borel probability measures on $`\mathcal X_Q`$ (approximate designs), put $`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$ and $`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$; no design is required to make $`A(\xi)`$ invertible. For a positive semidefinite matrix $`A`$ put

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

**Results used in the proof.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2): definition of the information norm $`\mathfrak n_{K,n}`$.

**Classical tools.** Prokhorov's theorem and Sion's minimax theorem ([Sion, 1958](#ref-sion1958general)); the Moore–Penrose pseudo-inverse of a symmetric positive semidefinite matrix.

**Proof.**

**Step 1 (a variational formula for every positive semidefinite matrix).** Let $`A\succeq0`$ and $`\ell_A(h)=2c^\top h-h^\top Ah`$. We show

```math
\Phi_c(A)=\sup_{h\in\mathbb R^p}\ell_A(h)=\sup_{h\neq0}\frac{(c^\top h)^2}{h^\top Ah}\qquad(0/0=0,\ a/0=+\infty\text{ for }a>0).
```

If $`c\in\mathrm{range}(A)`$, write $`c=Aw`$. Then $`c^\top A^+c=w^\top AA^+Aw=w^\top Aw`$ and

```math
\ell_A(h)=2w^\top Ah-h^\top Ah=w^\top Aw-(h-w)^\top A(h-w)\le w^\top Aw ,
```

with equality at $`h=w`$; so $`\sup_h\ell_A=c^\top A^+c`$. If $`c\notin\mathrm{range}(A)=(\ker A)^\perp`$, there is $`v\in\ker A`$ with $`c^\top v>0`$, and $`\ell_A(sv)=2s\,c^\top v\to+\infty`$ as $`s\to\infty`$. This proves the first equality. For the second, fix $`h\neq0`$ and maximize over the scale $`s\in\mathbb R`$: $`\sup_s\ell_A(sh)=\sup_s\big(2s\,c^\top h-s^2h^\top Ah\big)`$ equals $`(c^\top h)^2/h^\top Ah`$ when $`h^\top Ah>0`$, $`+\infty`$ when $`h^\top Ah=0`$ and $`c^\top h\neq0`$, and $`0`$ when both vanish. Taking the supremum over directions gives the ratio form with the stated conventions.

**Step 2 (exchange of the two optimizations).** Let $`\mathcal P`$ be the set of Borel probability measures on $`\mathcal X_Q`$ and $`L(\xi,h)=\ell_{A(\xi)}(h)=2c^\top h-K\,h^\top I_Dh-n\int h^\top I_Q(x)h\,\xi(\mathrm dx)`$. By Step 1 the left side of the lemma is $`\inf_{\xi\in\mathcal P}\sup_hL(\xi,h)`$.

- $`\mathcal P`$ is convex, and compact in the weak topology by Prokhorov's theorem, because $`\mathcal X_Q`$ is a compact metric space; it sits in the locally convex space of finite signed measures.
- For fixed $`h`$, $`\xi\mapsto L(\xi,h)`$ is affine and weakly continuous, because $`x\mapsto h^\top I_Q(x)h`$ is continuous and bounded on the compact set $`\mathcal X_Q`$.
- For fixed $`\xi`$, $`h\mapsto L(\xi,h)`$ is concave and continuous on the convex set $`\mathbb R^p`$.

Sion's minimax theorem, with $`\xi`$ as the minimizing variable on the compact convex set $`\mathcal P`$ (lower semicontinuous and quasi-convex in $`\xi`$) and $`h`$ as the maximizing variable (upper semicontinuous and quasi-concave in $`h`$), gives

```math
\inf_{\xi\in\mathcal P}\sup_hL(\xi,h)=\sup_h\inf_{\xi\in\mathcal P}L(\xi,h).
```

**Step 3 (attainment).** The map $`\xi\mapsto\Phi_c(A(\xi))=\sup_hL(\xi,h)`$ is a pointwise supremum of weakly continuous functions, so it is weakly lower semicontinuous with values in $`(-\infty,+\infty]`$. A lower semicontinuous function on a compact set attains its infimum; a minimizer is $`\xi^*`$. (If the value is $`+\infty`$, every design attains it.)

**Step 4 (the inner problem).** For fixed $`h`$, $`\sup_{\xi\in\mathcal P}\int h^\top I_Q(x)h\,\xi(\mathrm dx)=\max_xh^\top I_Q(x)h`$: the integral never exceeds the maximum of the continuous integrand, and a point mass at a maximizer attains it. Hence, with $`q(h)=K\,h^\top I_Dh+n\max_xh^\top I_Q(x)h`$,

```math
\sup_h\inf_{\xi\in\mathcal P}L(\xi,h)=\sup_h\big(2c^\top h-q(h)\big).
```

The function $`q`$ is nonnegative and positively homogeneous of degree 2, so the scale optimization of Step 1 applies verbatim and gives $`\sup_{h\neq0}(c^\top h)^2/q(h)`$ with the same conventions. By [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2) this is $`\mathfrak n_{K,n}^2`$. Steps 1–4 prove the displayed identity and the attainment.

**Step 5 (the three special cases).**

- (a) If $`A(\xi)\succ0`$, then $`\mathrm{range}\,A(\xi)=\mathbb R^p`$ and $`A(\xi)^+=A(\xi)^{-1}`$.
- (b) Let $`\mathcal N=\lbrace h:q(h)=0\rbrace`$ be the set of uninformed directions. Because every information matrix is positive semidefinite, $`q(h)=0`$ means $`I_Dh=0`$ (when $`K>0`$) and $`I_Q(x)h=0`$ for every $`x`$ (when $`n>0`$), so $`\mathcal N`$ is a linear subspace and $`q(h+u)=q(h)`$ for every $`u\in\mathcal N`$. If some $`u\in\mathcal N`$ has $`c^\top u\neq0`$, the ratio is $`+\infty`$ at $`u`$. Otherwise $`c\perp\mathcal N`$; writing $`h=h_\perp+u`$ with $`h_\perp\in\mathcal N^\perp`$, $`u\in\mathcal N`$, the ratio equals $`(c^\top h_\perp)^2/q(h_\perp)`$. The function $`q`$ is continuous (the maximum over a compact set of a jointly continuous function) and strictly positive on the unit sphere of $`\mathcal N^\perp`$, so it has a positive minimum $`q_{\min}`$ there, and the ratio is at most $`\lVert c\rVert^2/q_{\min}<\infty`$. When the value is finite, $`\Phi_c(A(\xi^*))<\infty`$, which by definition means $`c\in\mathrm{range}\,A(\xi^*)`$; nothing forces $`A(\xi^*)`$ to be invertible.
- (c) For $`n=0`$ every design gives $`A=K\,I_D`$, and $`\Phi_c(KI_D)`$ is $`c^\top(KI_D)^+c`$ if $`c\in\mathrm{range}(I_D)`$ and $`+\infty`$ otherwise. $`\square`$

**Comments.**

- For rank-one query information $`I_Q(x)=f_xf_x^\top`$ and $`K=0`$, $`\xi^*`$ is characterized geometrically by Elfving's theorem. The term $`K\,I_D`$ makes it a c-optimal design with prior information.
- The same identity can be reached by applying the invertible case to $`A(\xi)+\varepsilon I`$ and letting $`\varepsilon\downarrow0`$; the variational formula of Step 1 avoids exchanging that limit with the infimum over designs.

## Related work

- **Optimal design.** c-optimal designs and their geometry go back to [Elfving, 1952](#ref-elfving1952optimum); the minimax exchange is Sion's theorem ([Sion, 1958](#ref-sion1958general)); designs with prior information and singular information matrices, with estimability of $`c^\top\theta`$ meaning $`c\in\mathrm{range}\,A`$, are standard ([Pukelsheim, 1993](#ref-pukelsheim1993optimal)). What is specific here is the target vector $`c=\sum_tJ_t^*g_t`$: it is set by the optimizer's increment map and the declared task.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).

<a id="ref-sion1958general"></a>

**Maurice Sion (1958).** On general minimax theorems. *Pacific Journal of Mathematics* 8, no. 1, 171–176. DOI [10.2140/pjm.1958.8.171](https://doi.org/10.2140/pjm.1958.8.171).
