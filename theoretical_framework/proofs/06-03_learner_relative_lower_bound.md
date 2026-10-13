# Theorem 6.3 · Learner-relative lower bound

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-03_learner_relative_lower_bound.md) · [Proof](../proofs/06-03_learner_relative_lower_bound.md) · [Experiments](../experiments/06-03_learner_relative_lower_bound.md) · [中文](06-03_learner_relative_lower_bound_ZH.md)

**What is proved here.** Every distiller's worst expected first-order error over two environments is at least half the gap of the learner-relative functional times one minus the total variation between their transcript laws; locally this gives a quarter of the information norm.

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

**Statement ([Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)).**

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

- **(c) The norm as the local lower bound.** Under (G), taking for $`h`$ a maximizer of the ratio in [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2), the local minimax lower bound is $`\tfrac14\mathfrak n_{K,n}(\theta_0)\,(1-o(1))`$.

**Results used in the proof.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2): the distiller's term $`\sum_t\langle g_t,\mu^Q_t(z_t)\rangle`$ does not depend on $`\theta`$; definition of $`\mathbb I_{K,n}`$, of $`\mathfrak n_{K,n}`$ and of (G).

**Classical tools.** Le Cam's two-point method ([Le Cam, 1973](../references.md#lecam1973convergence)); Pinsker's inequality and the chain rule for KL divergence ([Cover & Thomas, 2006](#ref-cover2006elements)); local asymptotics under differentiability in quadratic mean ([van der Vaart, 1998](../references.md#vandervaart1998asymptotic)).

**Proof.** (a) Write $`s=\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$ and $`a(Q)=\sum_t\langle g_t,\mu^Q_t(z_t)\rangle`$, which does not depend on $`\theta`$ ([Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2)). By the triangle inequality,

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

## Related work

- **Lower-bound technique.** The two-point method is Le Cam's ([Le Cam, 1973](../references.md#lecam1973convergence)); the local form with Fisher information and differentiability in quadratic mean follows the standard local asymptotic minimax theory ([van der Vaart, 1998](../references.md#vandervaart1998asymptotic)). What is specific here is the functional: it is set by the learner's increment map and the declared task.
- **Data and teacher selection scores.** LESS ([Xia et al., 2024](../references.md#xia2024less)) and GRACE ([Panigrahi et al., 2025](../references.md#panigrahi2025good)) score candidate data or teachers. Theorem A is not another score: it is a lower bound that no distiller can beat, together with the acquisition rule that attains it.

## References

- [Le Cam, 1973](../references.md#lecam1973convergence)
- [Panigrahi et al., 2025](../references.md#panigrahi2025good)
- [van der Vaart, 1998](../references.md#vandervaart1998asymptotic)
- [Xia et al., 2024](../references.md#xia2024less)
