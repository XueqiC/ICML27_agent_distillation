# Proposition 5.8 · Evolution of warmth

*Statement*

[← Framework](../README.md) · [Statement](../statements/05-08_evolution_warmth.md) · [Proof](../proofs/05-08_evolution_warmth.md) · [Experiments](../experiments/05-08_evolution_warmth.md) · [中文](05-08_evolution_warmth_ZH.md)

**In one paragraph.** Adam's second moment is an exponential moving average with a fixed $`\beta_2`$. This proposition computes how the warmth of a typical gradient evolves as the second moment absorbs gradient evaluations. In the stationary mean-field case the law is exact, with two-sided bounds: the warmth falls like $`1/(\beta_2s)`$ only while $`s(1-\beta_2)\ll1`$ and then levels off at $`(1-\beta_2)/\beta_2`$. The proposition also shows that bias correction does not change the warmth, bounds the warmth-controlled error accumulated over a run, and states when the hypothesis of that bound holds, including after a learning-rate-zero warm-up.

**Setting used on this page.**

The objects below are restated so that this page can be read on its own; the general notation is that of [Definitions 1.3–1.4](../README.md#s1-3).

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

<a id="s5-8"></a>

**Proposition 5.8 (Evolution of warmth).** Let the second moment start at $`v_0=0`$ and absorb gradient evaluations $`G_1,G_2,\dots`$ by $`v_s=\beta_2v_{s-1}+(1-\beta_2)G_s^{\odot2}`$, coordinatewise. Write $`\gamma=1-\beta_2`$.

**(a) Stationary second moment: exact mean-field warmth.** Suppose $`\mathbb E\,G_{k,i}^2=\sigma_i^2>0`$ for all $`k`$. Define the mean-field warmth $`\bar\kappa_i(s)=(1-\beta_2)\sigma_i^2/(\beta_2\,\mathbb E\,v_{s,i})`$. Then

```math
\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2^{\,s}),\qquad \bar\kappa_i(s)=\frac{1-\beta_2}{\beta_2(1-\beta_2^{\,s})},\qquad \frac{\max\lbrace 1/s,\;1-\beta_2\rbrace}{\beta_2}\le\bar\kappa_i(s)\le\frac{1/s+(1-\beta_2)}{\beta_2}.
```

- $`\bar\kappa_i(s)`$ is decreasing in $`s`$ and converges to the plateau $`\kappa_\infty=(1-\beta_2)/\beta_2>0`$; it does not tend to zero.
- In the early phase it behaves like $`1/(\beta_2s)`$: precisely, $`1\le\beta_2s\,\bar\kappa_i(s)\le1+s(1-\beta_2)`$, so the approximation $`\bar\kappa_i(s)\approx1/(\beta_2s)`$ has relative error at most $`s(1-\beta_2)`$ and is accurate only when $`s(1-\beta_2)\ll1`$.
- If the gradient is constant, $`G_k=g`$, the formula holds for the warmth $`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ itself, not only for its mean-field version.

**(b) Bias correction does not change warmth.** With bias-corrected denominators $`d_i=\sqrt{v_i^+/(1-\beta_2^{\,t})}`$ and $`a_i=\sqrt{\beta_2v_i/(1-\beta_2^{\,t})}`$, the ratio $`d_i/a_i=\sqrt{1+\kappa_i}`$ is the same as without bias correction. Warmth, and the bound of Proposition 5.5(b), are therefore unaffected by bias correction.

**(c) Accumulation along a run.** Compare a reference protocol $`0`$ and an alternative protocol $`S`$ over $`R`$ real updates $`s=1,\dots,R`$ preceded by $`W`$ warm-up evaluations, through the telescoping identity of [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization.md#s3-2). Let $`\mathcal S_{\rm warm}\subseteq\lbrace 1,\dots,R\rbrace`$ be the updates at which, at every visited state, the hypotheses of Proposition 5.6 hold with constants indexed by $`s`$, the two laws have equal mean consumed gradient, and, almost surely, the condition below holds for some constant $`c>0`$.

```math
\max_i\kappa_i\le c\Big(\frac1{W+s}+(1-\beta_2)\Big).
```

Then

```math
\lvert L(0)-L(S)\rvert\le\big\lvert\lbrace s\le R:s\notin\mathcal S_{\rm warm}\rbrace\big\rvert+\bar A\sum_{s\in\mathcal S_{\rm warm}}\min\Big\lbrace 1,\frac{c}{W+s}+c(1-\beta_2)\Big\rbrace +\sum_{s\in\mathcal S_{\rm warm}}\bar B_s,
```

where $`\bar A=\sup_s2\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)`$, $`\bar u_{i,s}`$ bounds $`u_i`$ at update $`s`$, and $`\bar B_s=\sup H_{G,s}\lVert G\rVert_\infty^2`$. For $`W\ge1`$,

```math
\sum_{s=1}^{R}\min\Big\lbrace1,\;\frac{c}{W+s}+c(1-\beta_2)\Big\rbrace\le c\log\Big(1+\frac RW\Big)+c(1-\beta_2)R .
```

- The warmth-controlled contribution therefore has a per-update floor of order $`c(1-\beta_2)`$ that does not vanish as training lengthens. The logarithmic term dominates only when $`(1-\beta_2)R\lesssim\log(1+R/W)`$.
- With $`W=0`$ the first real update starts from $`v=0`$ and violates (W): it is the cold step of Proposition 5.3, charged at most $`1`$, and by Proposition 5.7 its contribution can be of first order in $`\eta`$. If $`\mathcal S_{\rm warm}=\lbrace 2,\dots,R\rbrace`$, the warmth-controlled sum is at most $`c\log R+c(1-\beta_2)(R-1)`$.

**(d) Warm-up, and when the hypothesis of (c) holds.** A *moment warm-up* of length $`W`$ performs $`W`$ optimizer steps with learning rate exactly $`0`$ on gradients of the declared data. It leaves $`\theta`$ unchanged and advances $`m`$, $`v`$ and the counter by the same recursions as real steps; every gradient absorbed into $`v`$, from warm-up or from a real update, counts as one evaluation. Real update $`s`$ therefore acts on a second moment that has absorbed $`W+s-1`$ evaluations. If the realized $`v`$ is replaced by its mean (the mean-field approximation) and the gradients satisfy $`G_{k,i}^2\le\bar\gamma\,\sigma_i^2`$, then for $`W+s\ge2`$ the hypothesis of (c) holds with $`c=2\bar\gamma/\beta_2`$. The replacement of $`v`$ by its mean is an approximation; it needs concentration of the exponential moving average and a stationary gradient scale.

**What it says.** With stationary gradients the mean-field warmth is exactly $`(1-\beta_2)/(\beta_2(1-\beta_2^s))`$ after $`s`$ absorbed evaluations. It falls like $`1/(\beta_2s)`$ only while $`s(1-\beta_2)\ll1`$, then levels off at $`(1-\beta_2)/\beta_2`$. Accumulated over a run, the warmth-controlled error is at most $`c\log(1+R/W)+c(1-\beta_2)R`$; bias correction does not change warmth; learning-rate-zero warm-up advances the second moment exactly like real steps.

**Intuition.** A fixed $`\beta_2`$ means a fixed averaging window, so every new gradient keeps perturbing the denominator by a fixed relative amount.

**Role in the framework.** Fixes how far into training the warm description is accurate, and shows it never becomes exact under a fixed $`\beta_2`$.

**Derivation outline.**

- Unroll the exponential moving average and take expectations; the bounds follow from Bernoulli's inequality and from $`e^{-y}(1+y)\le1`$.
- For (c), bound each warm term of the telescoping identity by Proposition 5.6 and use $`\min\lbrace1,a+b\rbrace\le\min\lbrace1,a\rbrace+b`$ and $`\sum_s1/(W+s)\le\log(1+R/W)`$.

**Illustration.** With $`\beta_2=0.999`$, the warmth of a typical gradient is about $`1/s`$ for the first hundred evaluations, but it never falls much below $`0.001`$: after a few thousand steps every new gradient still perturbs the second moment by about a tenth of a percent, and that floor adds a term linear in the number of updates.

**Uses.**

- [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6) — Warm Adam is first-order affine in the consumed gradient
- [Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean.md#s5-7) — Cold steps separate equal-mean feedback at first order
- [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2) — Continuation-response characterization

**Used by.**

- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10) — The first-order statistic by learner
- [Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11) — Checking the warm-state conditions on a real model

## Related work

- The bias correction is that of [Kingma & Ba, 2015](#ref-kingma2015adam); Proposition 5.8 shows that it does not change the warmth.
- Learning-rate warm-up for Adam, and the large variance of the adaptive learning rate that motivates it, are analysed by [Liu et al., 2020](#ref-liu2020variance). Proposition 5.8 adds that a warm-up with learning rate zero advances the second moment exactly like real steps, and that under a fixed $`\beta_2`$ the warmth levels off at $`(1-\beta_2)/\beta_2`$ instead of vanishing.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-liu2020variance"></a>

**Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, Jiawei Han (2020).** On the variance of the adaptive learning rate and beyond. *8th International Conference on Learning Representations (ICLR)*. arXiv [1908.03265](https://arxiv.org/abs/1908.03265).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).
