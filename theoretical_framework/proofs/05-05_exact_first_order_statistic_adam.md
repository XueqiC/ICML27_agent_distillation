# Proposition 5.5 · The exact first-order statistic of Adam and the frozen-denominator error

*Proof*

[← Framework](../README.md) · [Statement](../statements/05-05_exact_first_order_statistic_adam.md) · [Proof](../proofs/05-05_exact_first_order_statistic_adam.md) · [Experiments](../experiments/05-05_exact_first_order_statistic_adam.md) · [中文](05-05_exact_first_order_statistic_adam_ZH.md)

**What is proved here.** (a) The Adam increment, and a constructive pair of laws with equal mean and equal second moment whose parameter blocks differ; (b) the coordinatewise frozen-denominator error bound; (c) the affine dependence of the frozen-denominator mean increment on $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ and the resulting split into two approximations.

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

**Adam and AdamW.** Adam is bias-corrected, with $`\epsilon`$ outside the square root, optional decoupled weight decay (AdamW) and no AMSGrad. An update with counter $`t\ge1`$ acts on $`z=(\theta,m,v,t-1)`$ and a consumed gradient $`G\in\mathbb R^p`$. Coordinatewise,

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

**Statement ([Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5)).** In the setting of Definition 5.4:

- **(a) Exact statistic.** The Adam increment is $`\Delta=(\Delta\theta,\Delta m,\Delta v,1)`$ with $`\Delta\theta=-\eta\lambda_{\rm wd}\theta-\eta\,n(G)/(d(G)+\epsilon)`$, $`\Delta m=(1-\beta_1)(G-m)`$ and $`\Delta v=(1-\beta_2)(G^{\odot2}-v)`$. The first-order statistic of Proposition 5.1 is the exact mean $`\mu=\mathbb E\,\Delta(z,G)`$. Its parameter block is a nonlinear functional of the law of $`G`$: if $`\beta_1<1`$ and $`v_i>0`$ or $`\epsilon>0`$ for some coordinate $`i`$, there are two finitely supported laws with equal $`\mathbb E\,G`$ and equal $`\mathbb E\,G^{\odot2}`$ whose parameter blocks $`\mu_\theta`$ differ.
- **(b) Frozen-denominator error.** For every coordinate and every $`G`$,

```math
\lvert\theta_i^+(G)-\tilde\theta_i^+(G)\rvert\le\eta\,u_i(G)\,\min\lbrace \kappa_i(G)/2,\;1\rbrace .
```

- **(c) Two approximations, kept apart.** The frozen-denominator mean increment $`\tilde\mu=\mathbb E\,\tilde\Delta(z,G)`$ is determined exactly by $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$, and $`\lvert\mu_{\theta,i}-\tilde\mu_{\theta,i}\rvert\le\eta\,\mathbb E[u_i\min\lbrace\kappa_i/2,1\rbrace]`$ while the $`m`$ and $`v`$ blocks agree exactly. The statement "warm Adam needs only $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$" therefore rests on two separate approximations:
  - the first-order approximation of Proposition 5.1, whose remainder is controlled by the second moments of the increment, with the block structure described in Remark 5.2;
  - the frozen-denominator approximation, whose error is controlled by (b) and is small in the warmth.

*Proof of (a).* The increment is the definition of AdamW. For the second claim fix the coordinate $`i`$ and let $`G=g\,e_i`$ with scalar $`g`$, all other coordinates zero. The parameter block of coordinate $`i`$ is then $`\mathbb E f(g)`$ with

```math
f(g)=-\eta\,\frac{\big(\beta_1m_i+(1-\beta_1)g\big)/(1-\beta_1^{\,t})}{\sqrt{\big(\beta_2v_i+(1-\beta_2)g^2\big)/(1-\beta_2^{\,t})}+\epsilon}
```

- $`f`$ is bounded on $`\mathbb R`$: the denominator is bounded below by a positive constant when $`v_i>0`$ or $`\epsilon>0`$, and as $`g\to\pm\infty`$, $`f(g)`$ tends to $`\mp\eta(1-\beta_1)\sqrt{1-\beta_2^{\,t}}/\big((1-\beta_1^{\,t})\sqrt{1-\beta_2}\big)`$. The two limits differ because $`\beta_1<1`$, so $`f`$ is not constant.
- A polynomial of degree at most two that is bounded on $`\mathbb R`$ is constant. Hence $`f`$ is not such a polynomial, and there are four points $`g_1<g_2<g_3<g_4`$ on which $`f`$ does not agree with any polynomial of degree at most two (the quadratic through three points of the graph misses some fourth point).
- The linear map $`w\mapsto(\sum_kw_k,\sum_kw_kg_k,\sum_kw_kg_k^2)`$ from $`\mathbb R^4`$ to $`\mathbb R^3`$ has a nonzero kernel vector $`w^*`$. Its row space is the restriction of $`\mathrm{span}\lbrace1,g,g^2\rbrace`$ to the four points, so $`\sum_kw^*_kf(g_k)\ne0`$, for otherwise the restriction of $`f`$ would lie in that span.
- For small $`\tau>0`$, $`P=\tfrac14\mathbf 1+\tau w^*`$ and $`Q=\tfrac14\mathbf 1-\tau w^*`$ are probability vectors on the four points (because $`\sum_kw^*_k=0`$). They have equal $`\mathbb E\,g`$ and $`\mathbb E\,g^2`$, hence equal $`\mathbb E\,G`$ and $`\mathbb E\,G^{\odot2}`$, while $`\mathbb E_Pf-\mathbb E_Qf=2\tau\sum_kw^*_kf(g_k)\ne0`$. $`\square`$

*Proof of (b).*

- *Case $`v_i>0`$.* Then $`\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\lvert n_i\rvert\,\big\lvert\tfrac1{a_i\sqrt{1+\kappa_i}+\epsilon}-\tfrac1{a_i+\epsilon}\big\rvert`$. Bringing the reciprocals to a common denominator gives the identity below. Since $`\epsilon\ge0`$, the last factor is at most $`(\sqrt{1+\kappa_i}-1)/\sqrt{1+\kappa_i}=1-(1+\kappa_i)^{-1/2}`$. This is below $`1`$, and below $`\kappa_i/2`$ because $`\sqrt{1+\kappa}-1\le\kappa/2`$ and $`\sqrt{1+\kappa}\ge1`$. Hence the factor is at most $`\min\lbrace \kappa_i/2,1\rbrace`$.

```math
\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\,u_i\cdot\frac{a_i(\sqrt{1+\kappa_i}-1)}{a_i\sqrt{1+\kappa_i}+\epsilon}.
```

- *Case $`v_i=0`$ with $`\epsilon>0`$.* Then $`a_i=0`$, $`\kappa_i=\infty`$ and $`u_i=\lvert n_i\rvert/\epsilon`$. Both reciprocals $`1/(d_i+\epsilon)`$ and $`1/\epsilon`$ lie in $`[0,1/\epsilon]`$, so their difference is at most $`1/\epsilon`$, and the left side is at most $`\eta u_i=\eta u_i\min\lbrace \infty,1\rbrace`$. $`\square`$

*Proof of (c).* By Definition 5.4, $`\tilde\theta^+-\theta=-\eta\lambda_{\rm wd}\theta-\eta\Phi\beta_1m/(1-\beta_1^{\,t})-\eta\Phi(1-\beta_1)G/(1-\beta_1^{\,t})`$ and $`m^+-m=(1-\beta_1)(G-m)`$ are affine in $`G`$ with coefficients fixed by the state, and $`v^+-v=(1-\beta_2)(G^{\odot2}-v)`$ is affine in $`G^{\odot2}`$. Taking expectations, $`\tilde\mu`$ is an affine function of $`(\mathbb E\,G,\mathbb E\,G^{\odot2})`$. The $`m`$ and $`v`$ blocks of $`\Delta`$ and $`\tilde\Delta`$ coincide, and the $`\theta`$ blocks differ by $`\theta^+-\tilde\theta^+`$, whose expectation is bounded coordinatewise by (b). $`\square`$
