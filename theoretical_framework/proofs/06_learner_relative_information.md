# §6 Learner-relative information bound (Theorem A) — Proofs

[← Framework](../README.md) · [Experiments](../experiments/06_learner_relative_information.md) · [中文](06_learner_relative_information_ZH.md)

**Summary.**

- The section asks how much a distiller can learn about the oracle feedback from $`K`$ demonstrations and $`n`$ permitted queries, measured in the direction that matters for the actual learner and the declared task.
- **Theorem A** is the pair [Theorem 6.3](#s6-3) and [Theorem 6.5](#s6-5):
  - every distiller, replay, rewriting and synthesis included, has first-order fidelity error at least $`\tfrac14\mathfrak n_{K,n}`$, where $`\mathfrak n_{K,n}`$ is the learner-relative information norm of [Definition 6.2](#s6-2);
  - a plug-in distiller fed by a c-optimal (Elfving) query design reaches $`\mathfrak n_{K,n}(1+o(1))`$; the target vector of the design is the learner Jacobian applied to the continuation-value gradients, so the optimizer decides where information must be collected ([Lemma 6.4](#s6-4)).
- Consequences: aliasing and the zero value of replay ([Corollary 6.7](#s6-7)); an exchange rate between demonstrations and queries ([Corollary 6.8](#s6-8)); explicit formulas and a water-filling query allocation for the state-coverage structure of agent distillation ([Proposition 6.10](#s6-10)); learner weights for SGD and warm Adam ([Proposition 6.11](#s6-11)); separations between learners ([Corollary 6.12](#s6-12), [Example 6.13](#s6-13)).
- Uses: the complete learner state, increment map and continuation value ([Definitions 1.3–1.4](../README.md#s1-3)); the first-order representation ([Proposition 5.1](05_optimizer_state.md#s5-1)); the Adam increment ([Proposition 5.5](05_warm_adam.md#s5-5), [Theorem 5.10](05_learner_statistics.md#s5-10)); partially active clipping ([Proposition 5.9](05_learner_statistics.md#s5-9)). [Corollary 6.7](#s6-7) and [Corollary 6.8](#s6-8) are learner-relative versions of [Theorem 2.6](02_feasibility_and_acquisition.md#s2-6)(a), [Proposition 2.4](02_feasibility_and_acquisition.md#s2-4) and [Proposition 2.11](02_feasibility_and_acquisition.md#s2-11).

## Setting

<a id="s6-1"></a>

**Definition 6.1 (Environment family, data experiment, distiller, audited states).**

- **Environment family.** $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown.
- **Oracle feedback.** $`P^*_\theta=(P^*_{\theta,t})_t`$ is the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data experiment.**
  - $`K`$ demonstrations $`O_1,\dots,O_K`$ are i.i.d. with law $`P^D_\theta`$.
  - $`n`$ queries may be adaptive: the query point $`X_i`$ in the query space $`\mathcal X_Q`$ is a function of the history and of private randomness, and the outcome is $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$.
  - The law of the whole transcript $`D`$ is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from $`(D,U)`$ to a feedback law $`Q=(Q_t)_t`$, where the private randomness $`U`$ has a law that does not depend on $`\theta`$.
  - Replay, rewriting and synthesis are distillers, as long as they acquire no further $`\theta`$-dependent information.
  - The pretrained knowledge of a teacher model is common initial information, the same for every $`\theta`$. A new teacher call that observes a new state is a query.
- **Audited states.** Fixed complete learner states $`z_1,\dots,z_T`$ ([Definition 1.3](../README.md#s1-3)), for example the states the reference training reaches at checkpoints. They do not depend on $`\theta`$.

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
  - By [Proposition 5.1](05_optimizer_state.md#s5-1)(a) with $`k=1`$, $`\langle g_t,\mu^*_t(\theta)-\mu^Q_t(z_t)\rangle`$ is the first-order part of the effect of replacing one round of $`Q`$-feedback by oracle feedback at $`z_t`$, followed by the reference tail. $`\mathrm{Err}_\theta`$ adds these first-order effects over the audited states.
- **Information form and norm.** Let $`I_D(\theta)`$ be the Fisher information of one demonstration and $`I_Q(\theta;x)`$ that of one query at $`x\in\mathcal X_Q`$. For a direction $`h\in\mathbb R^p`$,

```math
\mathbb I_{K,n}(\theta;h)=K\,h^\top I_D(\theta)h+n\sup_{x\in\mathcal X_Q}h^\top I_Q(\theta;x)h,\qquad \mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\frac{\lvert c(\theta)^\top h\rvert}{\sqrt{\mathbb I_{K,n}(\theta;h)}},
```

  with the conventions $`0/0=0`$ and $`a/0=\infty`$ for $`a>0`$. $`\mathfrak n_{K,n}`$ is the **learner-relative information norm**: the dual norm of the learner-relative gradient $`c`$ with respect to the information gauge $`\sqrt{\mathbb I_{K,n}}`$.

- **Growth condition (G).** Some of the asymptotic statements below assume that the information grows non-degenerately: there are $`N\to\infty`$ and constants $`0<a\le b<\infty`$ with $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$ for all unit $`h`$. This holds, for example, when $`K`$ and $`n`$ grow proportionally and $`I_D+\sup_xI_Q(x)`$ is positive definite in every direction.

## Lower bound and achievability

<a id="s6-3"></a>

**Theorem 6.3 (Learner-relative lower bound).**

- **(a) Exact two-point bound.** For all $`\theta_0,\theta_1\in\Theta`$ and every distiller,

```math
\max_{i\in\lbrace0,1\rbrace}\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\big\lvert\Psi(\theta_1)-\Psi(\theta_0)\big\rvert\Big(1-\sqrt{\tfrac12\big(K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)\big)}\Big),
```

  where $`\mathrm{KL}_D=\mathrm{KL}(P^D_{\theta_1}\Vert P^D_{\theta_0})`$ and $`\mathrm{KL}_Q(x)=\mathrm{KL}(P^Q_{\theta_1}(\cdot\mid x)\Vert P^Q_{\theta_0}(\cdot\mid x))`$.

- **(b) Local bound.** Assume that each $`\mu^*_t`$ is $`C^1`$ near $`\theta_0`$, that both experiments are differentiable in quadratic mean at $`\theta_0`$, and that, as $`\tau\to0`$,

```math
\mathrm{KL}(P^D_{\theta_0+\tau h}\Vert P^D_{\theta_0})\le\tfrac12\tau^2h^\top I_D(\theta_0)h\,(1+o(1)),\qquad \mathrm{KL}_Q(x)\le\tfrac12\tau^2h^\top I_Q(\theta_0;x)h\,(1+o(1)),
```

  uniformly in $`x\in\mathcal X_Q`$ and in unit $`h`$. Let $`h`$ be a unit direction with $`c(\theta_0)^\top h\neq0`$ and $`\mathbb I_{K,n}(\theta_0;h)\to\infty`$, and put $`\tau=\mathbb I_{K,n}(\theta_0;h)^{-1/2}`$. Then

```math
\inf_Q\max_{\theta\in\lbrace\theta_0,\,\theta_0+\tau h\rbrace}\mathbb E_\theta\mathrm{Err}_\theta(Q)\ge\frac{\lvert c(\theta_0)^\top h\rvert}{4\sqrt{\mathbb I_{K,n}(\theta_0;h)}}\,\big(1-o(1)\big).
```

- **(c) The norm as the local lower bound.** Under (G), taking for $`h`$ a maximizer of the ratio in [Definition 6.2](#s6-2), the local minimax lower bound is $`\tfrac14\mathfrak n_{K,n}(\theta_0)\,(1-o(1))`$.

*Proof.* (a) Write $`s=\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$ and $`a(Q)=\sum_t\langle g_t,\mu^Q_t(z_t)\rangle`$, which does not depend on $`\theta`$ ([Definition 6.2](#s6-2)). By the triangle inequality,

```math
\mathrm{Err}_{\theta_0}(Q)+\mathrm{Err}_{\theta_1}(Q)=\lvert\Psi(\theta_0)-a(Q)\rvert+\lvert\Psi(\theta_1)-a(Q)\rvert\ge s .
```

$`Q`$ is a function of the transcript $`D`$ and of $`U`$, whose law is the same in both worlds; append $`U`$ to $`D`$, which does not change the total variation distance between the two transcript laws. Let $`p_i`$ be the density of $`\mathsf P^{K,n}_{\theta_i}`$ with respect to $`p_0+p_1`$. Then

```math
\mathbb E_{\theta_0}\mathrm{Err}_{\theta_0}+\mathbb E_{\theta_1}\mathrm{Err}_{\theta_1}\ge\int\min(p_0,p_1)\big(\mathrm{Err}_{\theta_0}+\mathrm{Err}_{\theta_1}\big)\ge s\big(1-\mathrm{TV}(\mathsf P^{K,n}_{\theta_0},\mathsf P^{K,n}_{\theta_1})\big).
```

By Pinsker's inequality, $`\mathrm{TV}\le\sqrt{\mathrm{KL}(\mathsf P^{K,n}_{\theta_1}\Vert\mathsf P^{K,n}_{\theta_0})/2}`$. By the chain rule for KL along the transcript:

- the demonstrations contribute $`K\,\mathrm{KL}_D`$;
- the $`i`$-th query outcome, given the history, contributes $`\mathbb E\,\mathrm{KL}_Q(X_i)\le\sup_x\mathrm{KL}_Q(x)`$;
- the choice of $`X_i`$ and of the private randomness has the same conditional law in both worlds and contributes zero.

So the KL is at most $`K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)`$. A maximum is at least an average, which gives (a).

(b) Since $`\mu^*_t\in C^1`$ and $`g_t`$ is fixed, $`\Psi\in C^1`$ and $`\lvert\Psi(\theta_0+\tau h)-\Psi(\theta_0)\rvert=\tau\lvert c(\theta_0)^\top h\rvert+o(\tau)`$. By the KL assumptions,

```math
K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)\le\tfrac12\tau^2\,\mathbb I_{K,n}(\theta_0;h)\,(1+o(1)).
```

With $`\tau=\mathbb I_{K,n}(\theta_0;h)^{-1/2}\to0`$ the square root in (a) is at most $`\tfrac12(1+o(1))`$. Hence the bound in (a) is at least $`\tfrac12\cdot\tau\lvert c^\top h\rvert(1+o(1))\cdot\tfrac12(1-o(1))`$, which is the claim; the relative form uses $`c(\theta_0)^\top h\neq0`$.

(c) Under (G) the ratio $`\lvert c^\top h\rvert/\sqrt{\mathbb I_{K,n}(\theta_0;h)}`$ is continuous on the unit sphere and attains its supremum at some $`h^\star`$. If $`c(\theta_0)\neq0`$, comparing $`h^\star`$ with $`c/\lVert c\rVert`$ gives $`\lvert c^\top h^\star\rvert\ge\lVert c\rVert\sqrt{a/b}>0`$, so the $`o(1)`$ terms of (b) are uniform along $`h^\star=h^\star_{K,n}`$. Apply (b) with $`h=h^\star`$. If $`c(\theta_0)=0`$ the claim is trivial. $`\square`$

<a id="s6-4"></a>

**Lemma 6.4 (Design minimax equality).** Assume that $`\mathcal X_Q`$ is a compact metric space and that $`x\mapsto I_Q(\theta;x)`$ is continuous. Let $`\xi`$ range over the Borel probability measures on $`\mathcal X_Q`$ (approximate designs), put $`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$ and $`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$, and assume $`A(\xi)\succ0`$ for every $`\xi`$. Then

```math
\inf_\xi c^\top A(\xi)^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h}=\mathfrak n_{K,n}^2 ,
```

and the infimum is attained by some design $`\xi^*`$, the **c-optimal design** for the target vector $`c=\sum_tJ_t^*g_t`$. For $`n=0`$ and singular $`I_D`$, the identity holds with $`c^\top(KI_D)^+c`$ when $`c\in\mathrm{range}(I_D)`$, and with $`+\infty`$ on both sides otherwise.

*Proof.* If $`c=0`$ both sides vanish; let $`c\neq0`$. For $`A\succ0`$, Cauchy–Schwarz gives $`(c^\top h)^2\le(c^\top A^{-1}c)(h^\top Ah)`$ with equality at $`h=A^{-1}c`$, so

```math
c^\top A^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{h^\top Ah}=\frac{1}{\inf_{h\in H}h^\top Ah},\qquad H=\lbrace h:c^\top h=1\rbrace .
```

Let $`f(\xi,h)=h^\top A(\xi)h`$. Then the left side of the lemma is $`1/\sup_\xi\inf_{h\in H}f(\xi,h)`$. For fixed $`h`$, $`\xi\mapsto h^\top M(\xi)h=\int h^\top I_Q(x)h\,\xi(\mathrm dx)`$ is maximized by a point mass at a maximizer of the continuous function $`x\mapsto h^\top I_Q(x)h`$ on the compact set $`\mathcal X_Q`$, so $`\sup_\xi f(\xi,h)=\mathbb I_{K,n}(h)`$ and the right side is $`1/\inf_{h\in H}\sup_\xi f(\xi,h)`$. It remains to exchange $`\sup_\xi`$ and $`\inf_h`$:

- the set of designs is convex and compact in the weak topology (Prokhorov's theorem), inside the locally convex space of finite signed measures;
- $`f(\cdot,h)`$ is affine and weakly continuous, because its integrand is continuous and bounded;
- $`f(\xi,\cdot)`$ is convex and continuous on the convex set $`H`$.

Sion's minimax theorem applied to $`-f`$, with the compact convex set of designs as the minimizing variable, gives $`\max_\xi\inf_{h\in H}f=\inf_{h\in H}\sup_\xi f`$. The function $`\xi\mapsto\inf_hf(\xi,h)`$ is an infimum of continuous functions, hence upper semicontinuous, and attains its maximum on the compact design set; this is $`\xi^*`$. For $`n=0`$, the pseudo-inverse form of Cauchy–Schwarz gives $`\sup_h(c^\top h)^2/(K\,h^\top I_Dh)=c^\top(KI_D)^+c`$ when $`c\perp\ker I_D`$ (directions in the kernel contribute $`0/0=0`$); otherwise some $`h\in\ker I_D`$ has $`c^\top h\neq0`$ and the ratio is $`+\infty`$. $`\square`$

- For rank-one query information $`I_Q(x)=f_xf_x^\top`$ and $`K=0`$, $`\xi^*`$ is characterized geometrically by Elfving's theorem. The term $`K\,I_D`$ makes it a c-optimal design with prior information.

<a id="s6-5"></a>

**Theorem 6.5 (Achievability; tight to constants in the scalar case).** Assume (G), the regularity of [Theorem 6.3](#s6-3)(b), and:

- **(R) Realizability.** For every $`\theta'\in\Theta`$ the distiller can emit the oracle law $`P^*_{\theta'}`$ at the audited states, for example by running a simulator or the teacher model at $`\theta'`$, or by reweighting messages obtained from queries.
- **(I) Identifiability.** Some design $`\xi_0`$ has $`\lambda_{\min}(M(\xi_0))=\lambda_0>0`$.
- **(S) Smoothness.** $`\Psi\in C^2`$ near $`\theta_0`$ with $`\lVert\nabla^2\Psi\rVert\le H_\Psi`$.
- **Design.** Use the regularized c-optimal design $`\xi_n=(1-\varepsilon_n)\xi^*+\varepsilon_n\xi_0`$, with $`\xi^*`$ from [Lemma 6.4](#s6-4), $`\varepsilon_n\to0`$ and $`\varepsilon_n\sqrt n\to\infty`$ (for example $`\varepsilon_n=n^{-1/4}`$).
- **(E) Efficient two-stage estimation.** $`\xi_n`$ is computed at a consistent pilot estimate obtained from $`o(n)`$ queries and rounded to integer counts. The final estimator $`\hat\theta`$ satisfies, uniformly over $`\theta`$ in a shrinking neighbourhood of $`\theta_0`$ and with $`\xi_n`$ evaluated at the true $`\theta`$,

```math
\mathbb E_\theta\big[(c(\theta)^\top(\hat\theta-\theta))^2\big]\le c(\theta)^\top A(\xi_n)^{-1}c(\theta)\,(1+o(1)),\qquad \mathbb E_\theta\lVert\hat\theta-\theta\rVert^2\le\frac{C}{\lambda_{\min}(A(\xi_n))} .
```

Let the distiller output $`Q=P^*_{\hat\theta}`$. Then

```math
\mathbb E_\theta\mathrm{Err}_\theta(Q)\le\mathfrak n_{K,n}(\theta)\,\big(1+o(1)\big)+o\big(N^{-1/2}\big),
```

and the last term is $`o(\mathfrak n_{K,n}(\theta))`$ whenever $`c(\theta)\neq0`$. Together with [Theorem 6.3](#s6-3)(c), the smallest achievable first-order error is, in the local minimax sense, between $`\tfrac14\mathfrak n_{K,n}`$ and $`\mathfrak n_{K,n}`$ up to factors $`1\pm o(1)`$. The optimal queries follow the c-optimal design whose target vector is $`c=\sum_tJ_t^*g_t`$.

*Proof.* By (R) the output $`P^*_{\hat\theta}`$ is available, and $`\mathrm{Err}_\theta(P^*_{\hat\theta})=\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert`$. By (S),

```math
\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert\le\lvert c(\theta)^\top(\hat\theta-\theta)\rvert+\tfrac12H_\Psi\lVert\hat\theta-\theta\rVert^2 .
```

- *Linear term.* $`A(\xi_n)=KI_D+n(1-\varepsilon_n)M(\xi^*)+n\varepsilon_nM(\xi_0)\succeq(1-\varepsilon_n)A(\xi^*)`$, hence $`c^\top A(\xi_n)^{-1}c\le c^\top A(\xi^*)^{-1}c/(1-\varepsilon_n)=\mathfrak n^2_{K,n}(1+o(1))`$ by [Lemma 6.4](#s6-4). By Jensen's inequality and (E), $`\mathbb E\lvert c^\top(\hat\theta-\theta)\rvert\le\mathfrak n_{K,n}(1+o(1))`$.
- *Quadratic term.* By (E) it is at most $`\tfrac12H_\Psi C/\lambda_{\min}(A(\xi_n))`$, and $`\lambda_{\min}(A(\xi_n))\ge\max\lbrace K\lambda_{\min}(I_D),\,n\varepsilon_n\lambda_0\rbrace`$. This is $`\gg\sqrt N`$:
  - if $`n\ge\delta N`$ for some $`\delta>0`$, then $`n\varepsilon_n\gg\sqrt n\ge\sqrt{\delta N}`$;
  - otherwise $`n=o(N)`$ along a subsequence, so for $`h\in\ker I_D`$ we would have $`\mathbb I_{K,n}(h)\le n\sup_xh^\top I_Qh=o(N)`$, contradicting (G); hence $`I_D\succ0`$, $`K`$ grows like $`N`$, and $`K\lambda_{\min}(I_D)\asymp N`$.

  So the quadratic term is $`o(N^{-1/2})`$.
- *Comparison.* If $`c(\theta)\neq0`$, (G) gives $`\mathfrak n_{K,n}\ge\lVert c\rVert/\sqrt{bN}`$, so $`o(N^{-1/2})=o(\mathfrak n_{K,n})`$. $`\square`$

- The regularization $`\varepsilon_n`$ matters: c-optimal designs are often supported on few query points, so $`A(\xi^*)`$ can be singular in directions that $`c`$ does not see, while the second-order term of $`\Psi`$ still needs $`\hat\theta`$ to be consistent in every direction.
- In a Gaussian linear experiment, $`\Psi`$ is linear, the efficient estimator is exactly Gaussian, and the plug-in error is exactly $`\sqrt{2/\pi}\,\sqrt{c^\top A^{-1}c}`$, about $`0.80\,\mathfrak n_{K,n}`$ at the optimal design.

<a id="s6-6"></a>

**Remark 6.6 (Class-uniform fidelity).** When fidelity is required uniformly over a declared class $`\mathcal V`$ of continuation values, each $`V\in\mathcal V`$ gives directions $`g^V_t=\nabla V_{t+1}(z_t)`$, a functional $`\Psi_V`$ and a gradient $`c_V`$; the error is $`\mathrm{Err}^{\mathcal V}_\theta(Q)=\sup_{V\in\mathcal V}\lvert\Psi_V(\theta)-\sum_t\langle g^V_t,\mu^Q_t(z_t)\rangle\rvert`$.

- **Lower bound.** The proof of [Theorem 6.3](#s6-3) goes through with $`\lvert c^\top h\rvert`$ replaced by $`\sigma(h)=\sup_V\lvert c_V^\top h\rvert`$, since $`\sup_V\lvert a_V\rvert+\sup_V\lvert b_V\rvert\ge\sup_V\lvert a_V-b_V\rvert`$. Exchanging the two suprema, the local lower bound is $`\tfrac14\sup_V\mathfrak n_{K,n}(c_V)`$.
- **Upper bound.** Let $`d`$ be the dimension of $`\mathrm{span}\lbrace c_V\rbrace`$ ($`d\le\dim\mathrm{span}\,\mathcal G`$). Under the analogue of (E) for $`d`$ linear functionals, the plug-in distiller with a suitable mixture design has error at most $`d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)\,(1+o(1))`$.
- So the lower and upper bounds differ by at most the factor $`4d^{3/2}`$, which depends only on the dimension of the declared class. Sharper constants are not claimed.

*Proof of the upper bound.* Let $`\mathcal K`$ be the closed convex hull of $`\lbrace\pm c_V\rbrace`$, a symmetric convex body in a $`d`$-dimensional subspace. By John's theorem there is an ellipsoid $`\mathcal E=\lbrace\sum_kw_k\ell_k:\lVert w\rVert\le1\rbrace\subseteq\mathcal K\subseteq\sqrt d\,\mathcal E`$, so $`\sigma(u)=\sup_{v\in\mathcal K}v^\top u\le\sqrt d\,\big(\sum_{k=1}^d(\ell_k^\top u)^2\big)^{1/2}`$. Let $`\xi_k`$ be c-optimal for $`\ell_k`$ and $`\bar\xi=\tfrac1d\sum_k\xi_k`$. Then $`A(\bar\xi)\succeq\tfrac1dA(\xi_k)`$ for each $`k`$, so $`\ell_k^\top A(\bar\xi)^{-1}\ell_k\le d\,\mathfrak n_{K,n}(\ell_k)^2`$. Since $`\ell_k\in\mathcal K`$ and $`\mathfrak n_{K,n}(\cdot)`$ is a seminorm of its vector argument, $`\mathfrak n_{K,n}(\ell_k)\le\sup_V\mathfrak n_{K,n}(c_V)`$. By Jensen's inequality,

```math
\mathbb E\,\sigma(\hat\theta-\theta)\le\sqrt d\,\Big(\sum_{k=1}^d\ell_k^\top A(\bar\xi)^{-1}\ell_k\Big)^{1/2}(1+o(1))\le d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)\,(1+o(1)),
```

and the second-order term is handled as in [Theorem 6.5](#s6-5). $`\square`$

## Consequences

<a id="s6-7"></a>

**Corollary 6.7 (Aliasing and the zero value of replay).**

- If $`\theta_0\neq\theta_1`$ satisfy $`P^D_{\theta_0}=P^D_{\theta_1}`$ and $`\Psi(\theta_0)\neq\Psi(\theta_1)`$, and either $`n=0`$ or $`P^Q_{\theta_0}(\cdot\mid x)=P^Q_{\theta_1}(\cdot\mid x)`$ for every $`x`$, then for every $`K`$ every distiller has $`\max_i\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$.
- Replay, rewriting and synthesis are functions of the transcript and private randomness, so they are covered by "every distiller" and cannot lower this bound.
- The bound is learner-relative. Whether demonstrations suffice depends on whether the directions that the demonstrations cannot see have a component along **this learner's** $`c`$; the same demonstrations can suffice for one learner and not for another ([Corollary 6.12](#s6-12)(a)).
- [Theorem 2.6](02_feasibility_and_acquisition.md#s2-6)(a) is the instance in which the two worlds have identical demonstration laws and different optimal student policies; [Proposition 2.4](02_feasibility_and_acquisition.md#s2-4) is the statement that post-processing cannot help.

*Proof.* The transcript laws coincide, so $`\mathrm{TV}=0`$ and [Theorem 6.3](#s6-3)(a) applies with the square-root term equal to zero. $`\square`$

<a id="s6-8"></a>

**Corollary 6.8 (Exchange rate between demonstrations and queries).** For a direction $`h`$ write $`e_D(h)=h^\top I_D(\theta_0)h`$ and $`e_Q(h)=\sup_xh^\top I_Q(\theta_0;x)h`$. Under the conditions of [Theorem 6.3](#s6-3)(b), a distiller whose local worst-case expected first-order error is at most $`\varepsilon`$ must satisfy, for every $`h`$,

```math
K\,e_D(h)+n\,e_Q(h)\ge\frac{(c(\theta_0)^\top h)^2}{16\,\varepsilon^2}\,\big(1-o(1)\big).
```

- The feasible set of $`(K,n)`$ is an intersection of half-planes. At a boundary point the binding direction $`h^\star`$ fixes the local exchange rate: one demonstration is worth $`e_D(h^\star)/e_Q(h^\star)`$ queries.
- $`h^\star`$ is determined by the learner's $`c`$. The exchange rate is therefore learner-relative: the same demonstrations are worth different numbers of queries to different learners.
- [Proposition 2.11](02_feasibility_and_acquisition.md#s2-11) is the analogous linear budget for testing between two worlds; here the budget holds direction by direction, weighted by the learner.

*Proof.* By [Theorem 6.3](#s6-3)(b), $`\varepsilon\ge\lvert c^\top h\rvert/(4\sqrt{\mathbb I_{K,n}(\theta_0;h)})\,(1-o(1))`$ with $`\mathbb I_{K,n}(\theta_0;h)=K\,e_D(h)+n\,e_Q(h)`$. Rearrange; directions with $`c^\top h=0`$ give no constraint. $`\square`$

## The state-coverage model: explicit formulas

This part specializes Theorem A to the structure that dominates agent distillation: demonstrations cover the states the teacher visits, while training and deployment happen at the states the student visits. In this part $`s`$ denotes an agent state.

<a id="s6-9"></a>

**Assumption 6.9 (State-coverage model).**

- **States and laws.** A finite set $`\mathcal S`$ of agent states. At state $`s`$ the oracle message is $`Y\sim P^*_s`$, for example the teacher's action or correction at $`s`$, on a finite message alphabet. The laws $`(P^*_s)_{s\in\mathcal S}`$ vary independently across states: no generalization across states is assumed.
- **Functional.** The learner-relative functional $`\Psi`$ is a Hadamard-differentiable functional of $`(P^*_s)_{s\in\mathcal S}`$. Its influence function at $`s`$ is $`f_s(y)=\frac{\mathrm d}{\mathrm d\epsilon}\Psi\big(P^*_s+\epsilon(\delta_y-P^*_s)\big)\big\vert_{\epsilon=0}`$, so $`\mathbb E_{P^*_s}f_s=0`$.
- **Counts.** State $`s`$ receives $`N_s=k_s+n_s`$ oracle samples: $`k_s`$ from demonstrations (in expectation $`KH\,d_T(s)`$, with $`H`$ the trajectory length and $`d_T`$ the teacher's state-visitation distribution) and $`n_s`$ from queries, $`\sum_sn_s=n`$. Bounds are stated for given counts; with random counts they hold conditionally on them.

<a id="s6-10"></a>

**Proposition 6.10 (Explicit information bound and optimal query allocation by water-filling).** Under [Assumption 6.9](#s6-9) let $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$, the **learner-relative weight** of state $`s`$.

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

- **(d) Irreducible part.** A state with $`W_s>0`$ and $`k_s=n_s=0`$, one the student reaches but the teacher does not and that is never queried, leaves an error floor that no $`K`$ removes ([Corollary 6.7](#s6-7)).

*Proof.* (a) Fix $`h\in\mathbb R^{\mathcal S}`$ and use the tilting submodel $`P_{s,\tau}=(1+\tau h_sf_s)\,P^*_s`$, a probability law for $`\lvert\tau\rvert`$ small because $`f_s`$ is bounded on a finite alphabet and has mean zero.

- Along the submodel, $`\frac{\mathrm d}{\mathrm d\tau}\Psi=\sum_s\int f_s\,h_sf_s\,\mathrm dP^*_s=\sum_sh_sW_s`$, by the definition of the influence function.
- One sample at $`s`$ has score $`h_sf_s`$ and Fisher information $`h_s^2W_s`$, so the transcript has information $`\sum_sN_sh_s^2W_s`$ for the scalar parameter $`\tau`$; the KL expansion is uniform because the scores are bounded.

The proof of [Theorem 6.3](#s6-3) uses only the chain rule for KL, which applies to fixed per-state counts. It gives the lower bound $`\tfrac14\lvert\sum_sh_sW_s\rvert/(\sum_sN_sh_s^2W_s)^{1/2}(1-o(1))`$. By Cauchy–Schwarz,

```math
\Big(\sum_sh_sW_s\Big)^2=\Big(\sum_s h_s\sqrt{N_sW_s}\cdot\sqrt{W_s/N_s}\Big)^2\le\Big(\sum_sN_sh_s^2W_s\Big)\Big(\sum_sW_s/N_s\Big),
```

with equality at $`h_s=1/N_s`$, which gives (a).

(b) Let $`\hat P_s`$ be the empirical law of the $`N_s`$ samples at $`s`$. A second-order expansion gives $`\Psi(\hat P)-\Psi(P^*)=\sum_s\frac1{N_s}\sum_{i\le N_s}f_s(Y_{s,i})+\mathrm{Rem}`$ with $`\mathbb E\lvert\mathrm{Rem}\rvert\le C\sum_s\mathbb E\lVert\hat P_s-P^*_s\rVert^2=O(\sum_s1/N_s)=O(1/N)`$. The linear term has mean zero and variance $`\sum_sW_s/N_s`$, so its expected absolute value is at most the square root of that by Jensen's inequality, and $`O(1/N)=o\big((\sum_sW_s/N_s)^{1/2}\big)`$ under proportional growth. The output $`Q=\hat P`$ is realizable by replay.

(c) The objective is convex on $`n_s\ge0`$ and strictly convex in the coordinates with $`W_s>0`$. The KKT conditions are $`W_s/(k_s+n_s)^2=\lambda`$ when $`n_s>0`$ and $`W_s/k_s^2\le\lambda`$ when $`n_s=0`$, which is the stated formula; the left side of the budget equation is continuous and strictly decreasing in $`\lambda`$ on the active range, so $`\lambda`$ is unique.

(d) No observation depends on $`P^*_s`$, while $`\Psi`$ does ($`W_s>0`$). Two oracle laws that differ only at $`s`$ have identical transcript laws and different $`\Psi`$, so [Corollary 6.7](#s6-7) applies. $`\square`$

<a id="s6-11"></a>

**Proposition 6.11 (Learner weights of SGD and warm Adam).** Besides [Assumption 6.9](#s6-9), assume:

- every agent state owns its own parameter coordinate (state-owned coordinates);
- each training step draws one state $`s_k\sim d`$ i.i.d. and one oracle message there;
- the consumed gradient does not depend on the parameters (the linear-response regime): $`g=G(s,Y)`$; write $`m_1=\mathbb Eg`$ and $`m_2=\mathbb Eg^2`$ at state $`s`$;
- the outcome is $`F=\langle\zeta,\theta_R\rangle`$, and $`\Psi`$ is the expected outcome change $`\mathbb E F(\theta_R)-F(\theta_0)`$, a functional of $`(P^*_s)_s`$;
- Adam has $`\epsilon`$ negligible relative to $`\sqrt v`$ and $`\beta_1<\sqrt{\beta_2}`$.

Then, to leading order as $`R\to\infty`$ in each stated regime:

- **(a) SGD.** $`f_s^{\rm SGD}(y)=-\eta R\,d(s)\,\zeta_s\,(g-m_1)`$, so $`W_s^{\rm SGD}=\eta^2R^2\zeta_s^2\,d(s)^2\,\mathrm{Var}(g)`$.
- **(b) Warm Adam, frequent regime.** Assume many visits within the effective averaging window of the second moment: $`d(s)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$ with bias correction, and in addition $`t\gg1/(1-\beta_2)`$ without it. Then

```math
f_s^{\rm Adam}(y)=-\eta R\sqrt{d(s)}\,\zeta_s\Big[\frac{g-m_1}{\sqrt{m_2}}-\frac{m_1(g^2-m_2)}{2m_2^{3/2}}\Big],\qquad W_s^{\rm Adam}=\eta^2R^2\zeta_s^2\,d(s)\,\mathrm{Var}\Big(\frac{g}{\sqrt{m_2}}-\frac{m_1g^2}{2m_2^{3/2}}\Big).
```

  The second term in the bracket is the second-moment channel: a message that raises $`m_2`$ lowers every later step size.

- **(c) Warm Adam, isolated regime.** Assume $`d(s)\ll\min\lbrace1-\beta_1,1-\beta_2\rbrace`$, and either no bias correction or $`t\gg1/(1-\beta_2)`$. Each visit then meets $`m\approx0`$ and $`v\approx0`$ and moves the coordinate by $`-\eta\,\kappa_{\rm iso}\,\mathrm{sign}(g)`$, with

```math
\kappa_{\rm iso}=\frac{1-\beta_1}{\sqrt{1-\beta_2}\,\big(1-\beta_1/\sqrt{\beta_2}\big)} .
```

  Hence $`f_s(y)=-\eta R\,d(s)\,\zeta_s\,\kappa_{\rm iso}\,(\mathrm{sign}\,g-\mathbb E\,\mathrm{sign}\,g)`$ and $`W_s\propto d(s)^2\,\mathrm{Var}(\mathrm{sign}\,g)`$: the same scaling in $`d`$ as SGD, but the message enters only through its sign. With bias correction, a visit at step $`t\gg1/(1-\beta_1)`$ to a coordinate whose $`m`$ and $`v`$ are still zero (for example its first visit) moves it by $`-\eta\,\kappa_{\rm iso}\sqrt{1-\beta_2^{\,t}}\,\mathrm{sign}(g)`$; the constant changes, the scaling in $`d`$ does not.

- **(d) Consequence.** Let $`s`$ be a frequent and $`s'`$ a rarer state, $`d(s)>d(s')`$, both in the frequent regime and with the same variance factors. The weight ratio $`W_{s'}/W_s`$ is $`(d(s')/d(s))^2`$ under SGD and $`d(s')/d(s)`$ under warm Adam. Relative to the frequent state, warm Adam therefore weights the rarer state $`d(s)/d(s')`$ times more heavily than SGD does. Demonstrations that cover the frequent states and miss the student's long tail can suffice for SGD and not for warm Adam.

*Proof.* Coordinates do not interact, so $`\Psi=\sum_s\zeta_s\,\mathbb E[\theta_{s,R}-\theta_{s,0}]`$, and it suffices to compute, for each $`s`$, the Gateaux derivative of the expected displacement with respect to $`P^*_s`$. Write $`\delta m_1(y)=g(y)-m_1`$ and $`\delta m_2(y)=g(y)^2-m_2`$ for the derivatives of $`m_1,m_2`$ along $`\delta_y-P^*_s`$.

(a) The SGD displacement is $`-\eta\sum_k1\lbrace s_k=s\rbrace g_k`$ with mean $`-\eta Rd(s)m_1`$, linear in $`P^*_s`$; its derivative is $`-\eta Rd(s)\,\delta m_1`$.

(b) Visits form a Bernoulli($`d(s)`$) sequence. The second moment is a weighted average of the per-step squared gradients $`g_k^21\lbrace s_k=s\rbrace`$ over an effective window of $`\min\lbrace t,1/(1-\beta_2)\rbrace`$ steps (after bias correction). In the frequent regime the window contains many visits, so by the law of large numbers $`v_k=d(s)m_2(1+o(1))`$; a single visit changes $`v`$ by the relative amount $`(1-\beta_2)g^2/(d(s)m_2)=o(1)`$, so the correlation between $`m_k`$ and $`v_k`$ is of lower order. The momentum is linear in the gradients and $`\sum_km_k=\sum_jg_j1\lbrace s_j=s\rbrace(1-\beta_1^{R-j})`$, whose mean is $`Rd(s)m_1(1+o(1))`$. Hence the expected displacement is

```math
\Phi(m_1,m_2)=-\eta R\sqrt{d(s)}\,\frac{m_1}{\sqrt{m_2}}\,\big(1+o(1)\big).
```

$`\Phi`$ is a smooth function of $`(m_1,m_2)`$, so by the chain rule (delta method) its influence function is $`\partial_{m_1}\Phi\,\delta m_1+\partial_{m_2}\Phi\,\delta m_2`$ with $`\partial_{m_1}\Phi=-\eta R\sqrt{d}/\sqrt{m_2}`$ and $`\partial_{m_2}\Phi=\tfrac12\eta R\sqrt d\,m_1/m_2^{3/2}`$. Multiplying by $`\zeta_s`$ gives the stated $`f_s`$; the term from $`\partial_{m_2}\Phi`$ is the second-moment channel.

(c) Between visits, $`m`$ and $`v`$ are multiplied by $`\beta_1`$ and $`\beta_2`$ per step; a gap is about $`1/d(s)`$ steps, so $`\beta_1^{1/d}`$ and $`\beta_2^{1/d}`$ are negligible and each visit meets $`m\approx0`$, $`v\approx0`$. At the $`j`$-th step after a visit with gradient $`g`$, $`m=(1-\beta_1)\beta_1^{\,j}g`$ and $`v=(1-\beta_2)\beta_2^{\,j}g^2`$, so the displacement is

```math
-\eta\,\frac{(1-\beta_1)\beta_1^{\,j}g}{\sqrt{(1-\beta_2)\beta_2^{\,j}}\,\lvert g\rvert}=-\frac{\eta(1-\beta_1)}{\sqrt{1-\beta_2}}\Big(\frac{\beta_1}{\sqrt{\beta_2}}\Big)^{j}\mathrm{sign}(g).
```

The geometric series converges because $`\beta_1<\sqrt{\beta_2}`$, and its terms decay within $`O(1/(1-\beta_1/\sqrt{\beta_2}))`$ steps, much less than the gap; summing over $`j\ge0`$ gives $`-\eta\kappa_{\rm iso}\mathrm{sign}(g)`$. The total displacement is $`-\eta\kappa_{\rm iso}\sum_k1\lbrace s_k=s\rbrace\mathrm{sign}(g_k)`$, with mean $`-\eta Rd(s)\kappa_{\rm iso}\mathbb E\,\mathrm{sign}(g)`$, linear in $`P^*_s`$, which gives $`f_s`$. With bias correction, the $`j`$-th step after a visit at step $`t`$ is divided by $`(1-\beta_1^{\,t+j})`$ and multiplied by $`\sqrt{1-\beta_2^{\,t+j}}`$; for $`t\gg1/(1-\beta_1)`$ and $`j`$ within the decay range, these factors are $`1+o(1)`$ and $`\sqrt{1-\beta_2^{\,t}}\,(1+o(1))`$.

(d) Divide the weights of (a) and of (b), then use [Proposition 6.10](#s6-10)(c), (d). $`\square`$

- The two regimes meet near $`d(s)\approx1-\beta_2`$, where a visit's effect on $`v`$ neither averages out nor decays completely. Neither closed form applies there.

<a id="s6-12"></a>

**Corollary 6.12 (Learner separations).**

- **(a) Exact separation by scale invariance.** Let $`\theta=(\alpha,b)`$ with $`b>0`$, let the oracle consumed gradient be $`b\,G_\alpha(Y)`$ (for example a feedback weight or advantage scale), and let the demonstration law depend on $`\alpha`$ only.
  - For SGD with always-active norm clipping at level $`c_{\rm clip}`$ ($`\lVert bG_\alpha(Y)\rVert>c_{\rm clip}`$ almost surely), and for normalized SGD, the oracle mean increment does not depend on $`b`$, so $`\Psi`$ depends on $`\alpha`$ only and $`c(\theta)`$ has no component along $`b`$. If the demonstrations' Fisher information about $`\alpha`$ is non-degenerate, the error tends to zero as $`K\to\infty`$ ([Theorem 6.5](#s6-5) with $`n=0`$, applied to the model in $`\alpha`$; realizability holds with any scale that keeps clipping active).
  - For plain SGD, $`c(\theta)`$ has the $`b`$-component $`-\eta\sum_t\langle g_t,\mathbb E\,G_\alpha\rangle`$; when it is nonzero, the demonstrations carry no information about $`b`$ and, by [Corollary 6.7](#s6-7), no $`K`$ suffices.
  - The same demonstrations are therefore sufficient for one learner and insufficient for the other. When clipping is active only with probability $`p_{\rm act}<1`$, [Proposition 5.9](05_learner_statistics.md#s5-9) puts each clipped mean consumed gradient within $`c_{\rm clip}(1-p_{\rm act})`$ of the $`b`$-free vector $`c_{\rm clip}\,\mathbb E[G_\alpha/\lVert G_\alpha\rVert]`$, so the separation degrades gracefully.
- **(b) Quantitative separation by long-tail coverage.** By [Proposition 6.10](#s6-10) and [Proposition 6.11](#s6-11)(d): when the student's state distribution has many low-frequency states, all in the frequent regime of warm Adam, and the teacher rarely reaches them, the share of the information requirement carried by those states is larger under warm Adam than under SGD, and warm Adam needs correspondingly more demonstrations or queries there for the same relative accuracy ([Example 6.13](#s6-13)).

*Proof.* (a) With always-active clipping, $`\mathrm{clip}_{c_{\rm clip}}(bG)=c_{\rm clip}\,G/\lVert G\rVert`$ does not depend on $`b>0`$, so $`\partial_b\mu^*_t=0`$; the rest follows from [Theorem 6.5](#s6-5) and [Corollary 6.7](#s6-7), with $`\theta_0=(\alpha,b)`$ and $`\theta_1=(\alpha,b')`$ having identical demonstration laws and $`\Psi(\theta_1)-\Psi(\theta_0)=-\eta(b'-b)\sum_t\langle g_t,\mathbb E\,G_\alpha\rangle\neq0`$ for SGD. (b) Combine [Proposition 6.10](#s6-10)(a)–(b) with the weights of [Proposition 6.11](#s6-11). $`\square`$

<a id="s6-13"></a>

**Example 6.13 (One common state and $`N`$ long-tail states).** The student visits a common state with probability $`1-q`$ and each of $`N`$ tail states with probability $`q/N`$. The teacher covers the common state with probability $`1-q_T`$ and each tail state with probability $`q_T/N`$, so $`N_s\propto KH\,d_T(s)`$. All states have the same variance factor and deployment weight, and the tail states are in the frequent regime of warm Adam, $`(q/N)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$.

- **Contributions to $`\sum_sW_s/N_s`$.** Up to the common factor:
  - tail states: $`N(q/N)^2/(q_T/N)=q^2/q_T`$ under SGD, and $`N(q/N)/(q_T/N)=Nq/q_T`$ under warm Adam;
  - common state: $`(1-q)^2/(1-q_T)`$ under SGD, and $`(1-q)/(1-q_T)`$ under warm Adam.
- **Amplification.** Relative to the common state, the tail term is larger under warm Adam than under SGD by the factor $`N(1-q)/q`$. For instance, $`N=20`$ and $`q=0.2`$ give a factor $`80`$; with $`\beta_2=0.999`$ each tail state then has $`(q/N)/(1-\beta_2)=10`$ expected visits per second-moment window, inside the frequent regime.
- **Allocation.** The water-filling rule of [Proposition 6.10](#s6-10)(c) therefore sends more of the query budget to the tail states under warm Adam than under SGD.

## Related work

- **Lower-bound technique.** The two-point method is Le Cam's (Le Cam, 1973); the local form with Fisher information and differentiability in quadratic mean follows the standard local asymptotic minimax theory (van der Vaart, 1998). Influence functions of statistical functionals are classical; here they are taken of the learner-relative functional.
- **Optimal design.** c-optimal designs and their geometry go back to Elfving (1952); the minimax exchange in [Lemma 6.4](#s6-4) is Sion's theorem (Sion, 1958); designs with prior information and regularized designs are standard (Pukelsheim, 1993). What is specific here is the target vector $`c=\sum_tJ_t^*g_t`$: it is set by the optimizer's increment map and the declared task.
- **Active acquisition in imitation learning.** DAgger (Ross, Gordon and Bagnell, 2011) queries the teacher at the states the student visits. In the language of [Proposition 6.10](#s6-10) it allocates queries in proportion to student visitation; the information-optimal allocation is water-filling in the learner-relative weights, and these weights change with the optimizer ([Proposition 6.11](#s6-11)).
- **Data and teacher selection scores.** LESS (Xia et al., 2024) and GRACE (Panigrahi et al., 2025) score candidate data or teachers. Theorem A is not another score: it is a lower bound that no distiller can beat, together with the acquisition rule that attains it.

**References.** Elfving, G. (1952), *Ann. Math. Statist.* 23, no. 2. Le Cam, L. (1973), *Ann. Statist.* 1, no. 1. Panigrahi, A., et al. (2025), arXiv:2511.02833. Pukelsheim, F. (1993), *Optimal Design of Experiments*, Wiley. Ross, S., Gordon, G., Bagnell, D. (2011), AISTATS. Sion, M. (1958), *Pacific J. Math.* 8, no. 1. van der Vaart, A. W. (1998), *Asymptotic Statistics*, Cambridge University Press. Xia, M., et al. (2024), ICML.
