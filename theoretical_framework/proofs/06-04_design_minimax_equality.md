# Lemma 6.4 · Design minimax equality

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-04_design_minimax_equality.md) · [Proof](../proofs/06-04_design_minimax_equality.md) · [Experiments](../experiments/06-04_design_minimax_equality.md) · [中文](06-04_design_minimax_equality_ZH.md)

**What is proved here.** The c-optimal design value $`\inf_\xi c^\top A(\xi)^{-1}c`$ equals the squared information norm, and the infimum is attained.

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

**Statement ([Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4)).** Assume that $`\mathcal X_Q`$ is a compact metric space and that $`x\mapsto I_Q(\theta;x)`$ is continuous. Let $`\xi`$ range over the Borel probability measures on $`\mathcal X_Q`$ (approximate designs), put $`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$ and $`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$, and assume $`A(\xi)\succ0`$ for every $`\xi`$. Then

```math
\inf_\xi c^\top A(\xi)^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h}=\mathfrak n_{K,n}^2 ,
```

and the infimum is attained by some design $`\xi^*`$, the **c-optimal design** for the target vector $`c=\sum_tJ_t^*g_t`$. For $`n=0`$ and singular $`I_D`$, the identity holds with $`c^\top(KI_D)^+c`$ when $`c\in\mathrm{range}(I_D)`$, and with $`+\infty`$ on both sides otherwise.

**Results used in the proof.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2): definition of the information norm $`\mathfrak n_{K,n}`$.

**Classical tools.** Cauchy–Schwarz, Prokhorov's theorem and Sion's minimax theorem ([Sion, 1958](../references.md#sion1958general)).

**Proof.** If $`c=0`$ both sides vanish; let $`c\neq0`$. For $`A\succ0`$, Cauchy–Schwarz gives $`(c^\top h)^2\le(c^\top A^{-1}c)(h^\top Ah)`$ with equality at $`h=A^{-1}c`$, so

```math
c^\top A^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{h^\top Ah}=\frac{1}{\inf_{h\in H}h^\top Ah},\qquad H=\lbrace h:c^\top h=1\rbrace .
```

Let $`f(\xi,h)=h^\top A(\xi)h`$. Then the left side of the lemma is $`1/\sup_\xi\inf_{h\in H}f(\xi,h)`$. For fixed $`h`$, $`\xi\mapsto h^\top M(\xi)h=\int h^\top I_Q(x)h\,\xi(\mathrm dx)`$ is maximized by a point mass at a maximizer of the continuous function $`x\mapsto h^\top I_Q(x)h`$ on the compact set $`\mathcal X_Q`$, so $`\sup_\xi f(\xi,h)=\mathbb I_{K,n}(h)`$ and the right side is $`1/\inf_{h\in H}\sup_\xi f(\xi,h)`$. It remains to exchange $`\sup_\xi`$ and $`\inf_h`$:

- the set of designs is convex and compact in the weak topology (Prokhorov's theorem), inside the locally convex space of finite signed measures;
- $`f(\cdot,h)`$ is affine and weakly continuous, because its integrand is continuous and bounded;
- $`f(\xi,\cdot)`$ is convex and continuous on the convex set $`H`$.

Sion's minimax theorem applied to $`-f`$, with the compact convex set of designs as the minimizing variable, gives $`\max_\xi\inf_{h\in H}f=\inf_{h\in H}\sup_\xi f`$. The function $`\xi\mapsto\inf_hf(\xi,h)`$ is an infimum of continuous functions, hence upper semicontinuous, and attains its maximum on the compact design set; this is $`\xi^*`$. For $`n=0`$, the pseudo-inverse form of Cauchy–Schwarz gives $`\sup_h(c^\top h)^2/(K\,h^\top I_Dh)=c^\top(KI_D)^+c`$ when $`c\perp\ker I_D`$ (directions in the kernel contribute $`0/0=0`$); otherwise some $`h\in\ker I_D`$ has $`c^\top h\neq0`$ and the ratio is $`+\infty`$. $`\square`$

**Comments.**

- For rank-one query information $`I_Q(x)=f_xf_x^\top`$ and $`K=0`$, $`\xi^*`$ is characterized geometrically by Elfving's theorem. The term $`K\,I_D`$ makes it a c-optimal design with prior information.

## Related work

- **Optimal design.** c-optimal designs and their geometry go back to [Elfving, 1952](../references.md#elfving1952optimum); the minimax exchange is Sion's theorem ([Sion, 1958](../references.md#sion1958general)); designs with prior information are standard ([Pukelsheim, 1993](../references.md#pukelsheim1993optimal)). What is specific here is the target vector $`c=\sum_tJ_t^*g_t`$: it is set by the optimizer's increment map and the declared task.

## References

- [Elfving, 1952](../references.md#elfving1952optimum)
- [Pukelsheim, 1993](../references.md#pukelsheim1993optimal)
- [Sion, 1958](../references.md#sion1958general)
