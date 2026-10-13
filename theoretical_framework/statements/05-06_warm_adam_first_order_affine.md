# Proposition 5.6 · Warm Adam is first-order affine in the consumed gradient

*Statement*

[← Framework](../README.md) · [Statement](../statements/05-06_warm_adam_first_order_affine.md) · [Proof](../proofs/05-06_warm_adam_first_order_affine.md) · [Experiments](../experiments/05-06_warm_adam_first_order_affine.md) · [中文](05-06_warm_adam_first_order_affine_ZH.md)

**In one paragraph.** At a warm Adam state, where every coordinate has a positive second moment, the one-boundary response is the expected final score after one update with message $`y`$ followed by the rest of training. This proposition shows that the response is an affine function of the consumed gradient up to an explicit residual with three parts: the frozen-denominator error, a second-moment sensitivity term and a curvature term. Two feedback laws with the same mean consumed gradient therefore differ only by these residuals.

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

<a id="s5-6"></a>

**Proposition 5.6 (Warm Adam is first-order affine in the consumed gradient).** Fix a round $`t\ge1`$, a state $`z=(\theta,m,v,t-1)`$ and a finite message set $`Y`$ with consumed gradients $`G(y)`$. Let $`V:\mathcal Z\to[0,1]`$ be the continuation value after the update (the declared reference tail followed by deployment, with tail randomness integrated), $`\psi(y)=V(z^+(y))`$ the one-boundary response, and $`F(g)=V(\tilde z^+(g))`$. Assume, with all states generated from $`z`$ by messages in $`Y`$ and the counter fixed at $`t`$:

- **(W)** $`v_i>0`$ for every coordinate (warm state);
- **(L)** $`\lvert V(\vartheta,m,v)-V(\vartheta',m,v)\rvert\le\sum_iL_i\lvert\vartheta_i-\vartheta_i'\rvert`$ on the pairs $`\big(\theta^+(y),\tilde\theta^+(y)\big)`$ at $`(m,v)=(m^+(y),v^+(y))`$;
- **$`(\mathrm S_v)`$** there are $`L_v\ge0`$ and a cap $`\bar k\in(0,\infty]`$ such that, at $`(\vartheta,m)=(\tilde\theta^+(y),m^+(y))`$, $`\lvert V(\vartheta,m,\beta_2v\odot(1+k))-V(\vartheta,m,\beta_2v)\rvert\le L_v\max_i\min\lbrace k_i,\bar k\rbrace`$ for $`k=\kappa(y)`$ (since $`V\in[0,1]`$ this always holds with $`L_v\bar k\ge1`$; the content lies in small $`L_v`$);
- **(H)** $`F`$ is twice continuously differentiable on the convex hull $`\mathcal C`$ of $`\lbrace 0\rbrace \cup\lbrace G(y):y\in Y\rbrace`$, and $`\lvert s^\top\nabla^2F(g)\,s\rvert\le H_G\lVert s\rVert_\infty^2`$ for all $`g\in\mathcal C`$ and $`s\in\mathbb R^p`$.

Let $`\psi_0=F(0)`$ and $`b=\nabla F(0)`$; both are independent of $`y`$. Then for every $`y\in Y`$

```math
\big\lvert\psi(y)-\psi_0-b^\top G(y)\big\rvert\le\eta\sum_iL_iu_i(y)\min\lbrace \tfrac{\kappa_i(y)}2,1\rbrace +L_v\max_i\min\lbrace \kappa_i(y),\bar k\rbrace +\tfrac12H_G\lVert G(y)\rVert_\infty^2=:r(y),
```

and for any two message laws $`P_0,P_1`$ on $`Y`$

```math
\big\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\big\rvert\le\min\Big\lbrace 1,\;\big\lvert b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)\big\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r\Big\rbrace .
```

- In particular, laws with equal mean consumed gradient differ in response by at most $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$.
- If $`V`$ is differentiable in $`(\theta,m)`$, then the first-order statistic is the mean preconditioned consumed gradient, with weight

```math
b=(1-\beta_1)\Big[-\frac{\eta}{1-\beta_1^{\,t}}\,\Phi\,\nabla_\theta V+\nabla_mV\Big]\Big|_{\tilde z^+(0)}.
```

**What it says.** At a warm state, the one-boundary response is an affine function of the consumed gradient plus an explicit residual: a frozen-denominator term, a second-moment sensitivity term, and a curvature term. Laws with equal mean consumed gradient differ by at most the expected residuals.

**Intuition.** Once the second moment has accumulated, one Adam step behaves like one preconditioned momentum-SGD step.

**Role in the framework.** Says from which point of training the affine and first-order theory applies to Adam, with a computable error.

**Derivation outline.**

- Split the response into three differences: true versus frozen parameter step (bounded by Proposition 5.5(b)), true versus frozen second moment (bounded by $`(\mathrm S_v)`$), and the Taylor remainder of $`F`$ (bounded by (H)).

**Uses.**

- [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5) — The exact first-order statistic of Adam and the frozen-denominator error

**Used by.**

- [Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean.md#s5-7) — Cold steps separate equal-mean feedback at first order
- [Proposition 5.8](../statements/05-08_evolution_warmth.md#s5-8) — Evolution of warmth
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10) — The first-order statistic by learner

## Related work

- LESS ([Xia et al., 2024](#ref-xia2024less)) uses Adam-preconditioned gradients, computed with the optimizer state of a short LoRA warm-up run, as a first-order proxy for the effect of a training example. Proposition 5.6 states from which point of training such an affine description of a warm Adam step holds, with an explicit residual.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
