# Proposition 5.5 · The exact first-order statistic of Adam and the frozen-denominator error

*Statement*

[← Framework](../README.md) · [Statement](../statements/05-05_exact_first_order_statistic_adam.md) · [Proof](../proofs/05-05_exact_first_order_statistic_adam.md) · [Experiments](../experiments/05-05_exact_first_order_statistic_adam.md) · [中文](05-05_exact_first_order_statistic_adam_ZH.md)

**In one paragraph.** For Adam, [Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1) says that the first-order statistic of a feedback law is the mean Adam increment. This proposition makes that precise. The mean increment is a nonlinear functional of the gradient law, not determined by its mean and second moment. Freezing the denominator turns the parameter step into preconditioned momentum SGD, with an explicit per-coordinate error governed by the warmth. The common statement that warm Adam needs only $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ therefore rests on two separate approximations.

**Setting used on this page.**

The objects below are restated so that this page can be read on its own; the general notation is that of [Definitions 1.3–1.4](../README.md#s1-3).

- **State.** The complete learner state is $`z=(\theta,o)\in\mathcal Z\subseteq\mathbb R^p\times\mathbb R^q`$. Here $`\theta`$ are the trainable parameters, for example LoRA adapters, and $`o`$ is the optimizer state, for example Adam's moments $`m,v`$ and the step counter. When the counter matters it is written explicitly.
- **Feedback and gradients.** At round $`t`$ a feedback law $`P_t(\cdot\mid z)`$ emits a message $`Y`$. The learner computes a consumed gradient $`g=G_t(z,Y,\xi)\in\mathbb R^p`$ after loss normalization and accumulation, with $`\xi`$ the declared update randomness. Clipping, if any, is part of the optimizer.
- **Optimizer.** The optimizer is a measurable increment map $`z^+=z+\Delta_t(z,g)`$. The compared feedback laws share the optimizer and the gradient map and differ only in the message law. A change of optimizer is a different $`\Delta`$, compared in the same way.
- **Letters reused locally.** In this section $`s`$ counts gradient evaluations absorbed by Adam's second moment (Proposition 5.8). It is unrelated to the agent states $`s`$ of Section 6. The clipping threshold is $`c`$; the weight-decay coefficient is $`\lambda_{\rm wd}`$.

The learning kernel of a law $`P`$ is

```math
(K_t^Pf)(z)=\mathbb E_{Y\sim P_t(\cdot\mid z),\,\xi}\,f\big(z+\Delta_t(z,G_t(z,Y,\xi))\big).
```

For a law $`P`$ at $`(t,z)`$, the message-level mean increment is $`D_t(z,y)=\mathbb E_\xi\Delta_t(z,G_t(z,y,\xi))`$, and

```math
\mu_t^P(z)=\mathbb E_{Y\sim P}D_t(z,Y),\qquad m_{k,t}^P(z)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(z,G_t(z,Y,\xi))^{\otimes j}.
```

- **Continuation.** $`V^P_{t+1}:\mathcal Z\to[0,1]`$ is the continuation value of protocol $`P`$: the later updates run under $`P`$, then the student is deployed and scored by $`F_E`$ ([Definition 1.4](../README.md#s1-4)). $`L(P)=\mathbb E\,F_E(Z_R^P)`$, and $`d_t^Q`$ is the law of $`Z_t`$ under protocol $`Q`$.
- **Norm.** A norm $`\lVert\cdot\rVert`$ on $`\mathbb R^{p+q}`$ is fixed, with dual norm $`\lVert\cdot\rVert_*`$.

**Adam and AdamW.** Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) is bias-corrected, with $`\epsilon`$ outside the square root, optional decoupled weight decay (AdamW; [Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)) and no AMSGrad. An update with counter $`t\ge1`$ acts on $`z=(\theta,m,v,t-1)`$ and a consumed gradient $`G\in\mathbb R^p`$. Coordinatewise,

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- The hyperparameters satisfy $`\beta_1\in[0,1)`$, $`\beta_2\in(0,1)`$, $`\eta>0`$, $`\lambda_{\rm wd}\ge0`$, $`\epsilon\ge0`$. Arithmetic is exact real arithmetic.
- The next complete state is $`z^+=(\theta^+,m^+,v^+,t)`$.
- A *cold* state has $`m=v=0`$ and $`t=1`$. A *warm* state has $`v_i>0`$ for every coordinate $`i`$.

*Warm-state notation ([Definition 5.4](../statements/05-04_warmth_frozen_denominator_surrogate.md#s5-4)).*

Consider an Adam update with counter $`t\ge1`$ at $`z=(\theta,m,v,t-1)`$, and assume $`v_i>0`$, or $`\epsilon>0`$ whenever some $`v_i=0`$.

- *Retained denominator, warmth, relative step.* For a consumed gradient $`G`$,

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^{\,t}}},\qquad\kappa_i(G)=\frac{(1-\beta_2)G_i^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(G)=\infty\ \ (v_i=0),\qquad u_i(G)=\frac{\lvert n_i(G)\rvert}{a_i+\epsilon}.
```

- When $`v_i>0`$, $`d_i=a_i\sqrt{1+\kappa_i}`$ and $`v_i^+=\beta_2v_i\,(1+\kappa_i)`$. The warmth $`\kappa_i`$ is the relative perturbation that the new gradient makes to the retained second moment.
- *Frozen-denominator surrogate.* With the fixed diagonal preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$,

```math
\tilde\theta^+(G)=(1-\eta\lambda_{\rm wd})\theta-\eta\,\Phi\,n(G),\qquad\tilde\Delta(z,G)=\big(\tilde\theta^+(G)-\theta,\;m^+(G)-m,\;v^+(G)-v\big),\qquad\tilde z^+(G)=\big(\tilde\theta^+(G),\,m^+(G),\,\beta_2v,\,t\big).
```

- $`\tilde\theta^+`$ is one momentum-SGD step with the fixed preconditioner $`\Phi`$. $`\tilde\Delta`$ is the *frozen-denominator increment*: its $`(\theta,m)`$ blocks are affine in $`G`$ and its $`v`$ block is affine in $`G^{\odot2}`$. $`\tilde z^+`$ additionally freezes the second moment at $`\beta_2v`$; it is the comparison state of Proposition 5.6. For $`g\in\mathbb R^p`$, $`\tilde z^+(g)`$ is the same map evaluated at consumed gradient $`g`$.

<a id="s5-5"></a>

**Proposition 5.5 (The exact first-order statistic of Adam and the frozen-denominator error).** In the setting of Definition 5.4:

- **(a) Exact statistic.** The Adam increment is $`\Delta=(\Delta\theta,\Delta m,\Delta v,1)`$ with $`\Delta\theta=-\eta\lambda_{\rm wd}\theta-\eta\,n(G)/(d(G)+\epsilon)`$, $`\Delta m=(1-\beta_1)(G-m)`$ and $`\Delta v=(1-\beta_2)(G^{\odot2}-v)`$. The first-order statistic of Proposition 5.1 is the exact mean $`\mu=\mathbb E\,\Delta(z,G)`$. Its parameter block is a nonlinear functional of the law of $`G`$: if $`\beta_1<1`$ and $`v_i>0`$ or $`\epsilon>0`$ for some coordinate $`i`$, there are two finitely supported laws with equal $`\mathbb E\,G`$ and equal $`\mathbb E\,G^{\odot2}`$ whose parameter blocks $`\mu_\theta`$ differ.
- **(b) Frozen-denominator error.** For every coordinate and every $`G`$,

```math
\lvert\theta_i^+(G)-\tilde\theta_i^+(G)\rvert\le\eta\,u_i(G)\,\min\lbrace \kappa_i(G)/2,\;1\rbrace .
```

- **(c) Two approximations, kept apart.** The frozen-denominator mean increment $`\tilde\mu=\mathbb E\,\tilde\Delta(z,G)`$ is determined exactly by $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$, and $`\lvert\mu_{\theta,i}-\tilde\mu_{\theta,i}\rvert\le\eta\,\mathbb E[u_i\min\lbrace\kappa_i/2,1\rbrace]`$ while the $`m`$ and $`v`$ blocks agree exactly. The statement "warm Adam needs only $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$" therefore rests on two separate approximations:
  - the first-order approximation of Proposition 5.1, whose remainder is controlled by the second moments of the increment, with the block structure described in Remark 5.2;
  - the frozen-denominator approximation, whose error is controlled by (b) and is small in the warmth.

**What it says.** For Adam the first-order statistic is the exact mean Adam increment, which is not determined by the mean and second moment of the gradient. Freezing the denominator at its retained value turns the step into preconditioned momentum SGD, with a per-coordinate error at most $`\eta u_i\min\lbrace\kappa_i/2,1\rbrace`$, where $`\kappa_i`$ is the warmth.

**Intuition.** Adam divides by a running root-mean-square that the new gradient itself perturbs. When that perturbation is small (small warmth) the step is nearly linear in the gradient.

**Role in the framework.** Separates the two approximations hidden in "warm Adam only needs $`(\mathbb E G,\mathbb E G^{\odot2})`$": the first-order expansion and the frozen denominator.

**Derivation outline.**

- With $`v^+=\beta_2v(1+\kappa)`$ the true denominator is $`a\sqrt{1+\kappa}`$; comparing $`1/(a\sqrt{1+\kappa}+\epsilon)`$ with $`1/(a+\epsilon)`$ gives the factor $`1-(1+\kappa)^{-1/2}\le\min\lbrace\kappa/2,1\rbrace`$.
- The Adam step is a bounded, nonconstant function of each gradient coordinate, hence not a quadratic polynomial; a kernel vector of the moment map on four points gives two laws with equal first two moments and different mean step.

**Uses.**

- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) — Optimizer-transformed expansion

**Used by.**

- [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6) — Warm Adam is first-order affine in the consumed gradient
- [Proposition 5.8](../statements/05-08_evolution_warmth.md#s5-8) — Evolution of warmth
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10) — The first-order statistic by learner
- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11) — Learner weights of SGD and warm Adam

## Related work

- [Balles & Hennig, 2018](#ref-balles2018dissecting) read Adam's direction as the sign of the mean gradient scaled by a variance adaptation factor, so that it depends on the gradient law through $`\mathbb E\,G`$ and $`\mathbb E\,G^{\odot2}`$. Proposition 5.5 shows that this description of a warm step rests on two separate approximations, the first-order expansion and the frozen denominator, and bounds the second by the warmth.
