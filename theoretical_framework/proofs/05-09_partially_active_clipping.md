# Proposition 5.9 · Partially active clipping

*Proof*

[← Framework](../README.md) · [Statement](../statements/05-09_partially_active_clipping.md) · [Proof](../proofs/05-09_partially_active_clipping.md) · [Experiments](../experiments/05-09_partially_active_clipping.md) · [中文](05-09_partially_active_clipping_ZH.md)

**What is proved here.** The split identity on the event that clipping is inactive, the norm bound, and the two-law version by the triangle inequality. The clipping map is global-norm clipping as introduced by [Pascanu et al., 2013](#ref-pascanu2013difficulty).

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

**Statement ([Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9)).** Let $`\mathrm{clip}_c(g)=g\min\lbrace1,c/\lVert g\rVert\rbrace`$ with $`c>0`$, set $`G/\lVert G\rVert=0`$ when $`G=0`$, and let $`p_{\rm act}=\Pr(\lVert G\rVert>c)`$ be the activation rate under a law of $`G`$. Then

```math
\mathbb E\,\mathrm{clip}_c(G)=\mathbb E\big[G\,1\lbrace\lVert G\rVert\le c\rbrace\big]+c\,\mathbb E\Big[\frac{G}{\lVert G\rVert}1\lbrace\lVert G\rVert>c\rbrace\Big],\qquad \Big\lVert\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E\frac{G}{\lVert G\rVert}\Big\rVert\le c\,(1-p_{\rm act}).
```

- For two laws, the clipped means differ from the scaled mean directions by at most $`c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$ in total. For instance, at an activation rate of $`0.9`$ under both laws, replacing the clipped mean by the mean direction errs by at most $`0.2c`$ in the difference.
- The always-active case $`p_{\rm act}=1`$ is the only one in which the mean direction is exactly the clipped mean.

*Proof.* The first identity splits $`\mathbb E\,\mathrm{clip}_c(G)`$ on the event $`\lbrace\lVert G\rVert\le c\rbrace`$. For the second, $`\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E[G/\lVert G\rVert]=\mathbb E\big[(G-cG/\lVert G\rVert)1\lbrace\lVert G\rVert\le c\rbrace\big]`$, and $`\lVert G-cG/\lVert G\rVert\rVert=\lvert\lVert G\rVert-c\rvert=c-\lVert G\rVert\le c`$ on that event (including $`G=0`$). Hence the norm is at most $`\mathbb E[(c-\lVert G\rVert)1\lbrace\lVert G\rVert\le c\rbrace]\le c\Pr(\lVert G\rVert\le c)=c(1-p_{\rm act})`$. The two-law statement follows by the triangle inequality. $`\square`$

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-pascanu2013difficulty"></a>

**Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013).** On the difficulty of training recurrent neural networks. *Proceedings of the 30th International Conference on Machine Learning (ICML)*, PMLR 28, no. 3, 1310–1318. [proceedings.mlr.press/v28/pascanu13](https://proceedings.mlr.press/v28/pascanu13.html).
