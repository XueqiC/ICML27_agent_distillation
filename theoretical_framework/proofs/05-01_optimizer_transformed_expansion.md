# Proposition 5.1 · Optimizer-transformed expansion

*Proof*

[← Framework](../README.md) · [Statement](../statements/05-01_optimizer_transformed_expansion.md) · [Proof](../proofs/05-01_optimizer_transformed_expansion.md) · [Experiments](../experiments/05-01_optimizer_transformed_expansion.md) · [中文](05-01_optimizer_transformed_expansion_ZH.md)

**What is proved here.** All four parts are proved: (a) the one-boundary Taylor expansion of the effect of a feedback law, with an explicit remainder; (b) its insertion into the telescoping identity over a whole run, with a dual-norm certificate and a stopped version; (c) first-order sufficiency, necessity and the minimal number of first-order statistics; (d) the moment hierarchy.

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

**Condition $`(\mathrm S_k)`$ (smooth continuation on the reachable segment).** Let $`\mathcal S_{t,z}`$ be the convex hull of $`z`$ and of all points $`z+\Delta_t(z,g)`$, with $`g`$ in the support of the consumed gradients under the compared laws. Then:

- $`V=V_{t+1}`$ is $`C^{k+1}`$ on $`\mathcal S_{t,z}`$;
- $`\lvert\nabla^{k+1}V(w)[h,\dots,h]\rvert\le M_{k+1}\lVert h\rVert^{k+1}`$ for every $`w\in\mathcal S_{t,z}`$ and every $`h`$;
- the moments $`m^P_{k+1,t}(z)`$ and $`m^Q_{k+1,t}(z)`$ are finite.

**Scope of $`(\mathrm S_k)`$.**

- Under clipping or sign updates, a multi-step continuation is generally not smooth at states where a later gradient crosses the clip threshold or a later update sign flips. $`(\mathrm S_k)`$ then holds only on regions that avoid these kinks, for example where clipping is always active or momentum dominates every sign.
- The mean increments of these optimizers are computed exactly in Theorem 5.10, without any smoothness assumption. The remainder bounds below are claimed only where $`(\mathrm S_k)`$ holds.

**Declared family.** For Proposition 5.1(c), fix a boundary $`(t,z)`$ and a set $`\mathcal V`$ of continuations, one per deployment task and reference tail of interest. Each satisfies $`(\mathrm S_1)`$ on $`\mathcal S_{t,z}`$ with the common Hessian bound $`M_2`$. Let

```math
U=\mathrm{span}\lbrace \nabla V(z):V\in\mathcal V\rbrace ,\qquad \mathcal G^{(1)}_{t,z}=\lbrace\,y\mapsto\langle\nabla V(z),D_t(z,y)\rangle:\ V\in\mathcal V\,\rbrace .
```

$`\mathcal G^{(1)}_{t,z}`$ is the first-order response class.

**Statement ([Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1)).**

**(a) One boundary, order $`k`$.** Under $`(\mathrm S_k)`$,

```math
(K_t^P-K_t^Q)V(z)=\sum_{j=1}^{k}\frac1{j!}\Big\langle\nabla^jV(z),\;\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\Big\rangle+r_k,\qquad\lvert r_k\rvert\le\frac{M_{k+1}}{(k+1)!}\big(m_{k+1}^P+m_{k+1}^Q\big).
```

In particular, for $`k=1`$,

```math
(K_t^P-K_t^Q)V(z)=\big\langle\nabla V(z),\mu_t^P(z)-\mu_t^Q(z)\big\rangle+r_1,\qquad\lvert r_1\rvert\le\tfrac{M_2}2\big(m_2^P+m_2^Q\big).
```

**(b) Whole run.** Compare protocols $`P`$ and $`Q`$ over $`R`$ rounds from a common initial law. Assume $`(\mathrm S_1)`$ at every state visited by $`Q`$, with constants $`M_{2,t}(z)`$ for the continuation $`V^P_{t+1}`$. Then

```math
L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\Big\langle\nabla V^P_{t+1}(Z_t),\;\mu_t^P(Z_t)-\mu_t^Q(Z_t)\Big\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\sum_t\mathbb E_{d_t^Q}\Big[\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big].
```

Consequently, with $`\Lambda_t(z)=\lVert\nabla V^P_{t+1}(z)\rVert_*`$,

```math
\lvert L(P)-L(Q)\rvert\le\min\Big\lbrace 1,\ \sum_t\mathbb E_{d_t^Q}\Big[\Lambda_t\,\lVert\mu_t^P-\mu_t^Q\rVert+\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big]\Big\rbrace .
```

*Stopped version.* Suppose $`(\mathrm S_1)`$ holds only on certified sets $`C_t\subseteq\mathcal Z`$. Let $`\tau=\min\lbrace t<R:Z_t^Q\notin C_t\rbrace`$, with $`\tau=R`$ if there is no such $`t`$, and $`\rho=\Pr(\tau<R)`$. Then

```math
\lvert L(P)-L(Q)\rvert\le\mathbb E_Q\sum_{t<\tau}\Big[\Lambda_t\,\lVert\mu_t^P-\mu_t^Q\rVert+\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big]\,(Z_t)+\rho.
```

**(c) First-order sufficiency, necessity and minimality.** Use the Euclidean pairing, and let $`\Pi_U`$ be the orthogonal projection onto $`U`$.

- *Sufficiency.* If $`\Pi_U(\mu^P-\mu^Q)=0`$, then $`\lvert(K^P-K^Q)V(z)\rvert\le\tfrac{M_2}2(m_2^P+m_2^Q)`$ for every $`V\in\mathcal V`$.
- *Necessity.* If $`\Pi_U(\mu^P-\mu^Q)\ne0`$, then some $`V\in\mathcal V`$ has $`\langle\nabla V(z),\mu^P-\mu^Q\rangle\ne0`$, and for every such $`V`$ the display below holds. The mismatch is therefore detected by that task whenever the first-order term exceeds the remainder bound.

```math
\lvert(K^P-K^Q)V(z)\rvert\ge\big\lvert\langle\nabla V(z),\mu^P-\mu^Q\rangle\big\rvert-\tfrac{M_2}2(m_2^P+m_2^Q).
```

- *Minimality.* On a finite message alphabet of size $`m`$, the minimal number of linear expectation statistics that determine every first-order term $`\langle\nabla V(z),\mu^P-\mu^Q\rangle`$, $`V\in\mathcal V`$, for all law pairs is given below.

```math
d_0^{(1)}=\dim\mathrm{span}\big(\lbrace 1\rbrace \cup\mathcal G^{(1)}_{t,z}\big)-1\le\min\lbrace \dim U,\ m-1\rbrace .
```

**(d) Moment hierarchy.** Under $`(\mathrm S_k)`$, suppose two laws agree on $`\mathbb E\Delta^{\otimes j}`$ for $`j=1,\dots,k`$ at a state. Then their one-boundary discrepancy there is at most $`\tfrac{M_{k+1}}{(k+1)!}(m_{k+1}^P+m_{k+1}^Q)`$. Summed over the visited states, as in (b), this bounds the run-level gap.

- For SGD, $`\Delta=-\eta G`$, so the matched quantities are the gradient moments $`\mathbb E G^{\otimes j}`$, and the bound is $`O(\eta^{k+1})`$ per round.
- For cold Adam, the parameter block matches the moments of $`h_\epsilon(G)`$ (Proposition 5.3). As $`\epsilon\to0`$ these are the joint sign moments.
- Under always-active clipping, the parameter block involves only the moments of the direction $`G/\lVert G\rVert`$.

*Proof of (a).* Fix a realized increment $`\Delta=\Delta_t(z,g)`$ with $`g`$ in the support. The segment $`\lbrace z+u\Delta:u\in[0,1]\rbrace`$ lies in $`\mathcal S_{t,z}`$, because $`\mathcal S_{t,z}`$ is convex and contains both endpoints. On this segment $`\varphi(u)=V(z+u\Delta)`$ is $`C^{k+1}`$, with $`\varphi^{(j)}(u)=\nabla^jV(z+u\Delta)[\Delta^{\otimes j}]`$. Taylor's theorem with Lagrange remainder gives some $`u^*\in(0,1)`$ such that

```math
V(z+\Delta)=V(z)+\sum_{j=1}^{k}\frac1{j!}\nabla^jV(z)[\Delta^{\otimes j}]+R(\Delta),\qquad R(\Delta)=\frac1{(k+1)!}\nabla^{k+1}V(z+u^*\Delta)[\Delta^{\otimes(k+1)}].
```

By $`(\mathrm S_k)`$, $`\lvert R(\Delta)\rvert\le\tfrac{M_{k+1}}{(k+1)!}\lVert\Delta\rVert^{k+1}`$.

Integrate this identity over $`Y\sim P`$ and $`\xi`$, and separately over $`Y\sim Q`$ and $`\xi`$. Each term is integrable: $`V`$ takes values in $`[0,1]`$, $`\lvert R\rvert`$ is dominated by a multiple of $`\lVert\Delta\rVert^{k+1}`$ with finite expectation, and each $`\nabla^jV(z)[\Delta^{\otimes j}]`$, $`j\le k`$, is a polynomial in $`\Delta`$ whose degree-$`j`$ moments are finite by Lyapunov's inequality. Subtract the two integrals.

- The constant $`V(z)`$ cancels, because both laws are probability measures.
- The map $`\Delta^{\otimes j}\mapsto\nabla^jV(z)[\Delta^{\otimes j}]`$ is linear, so the expectation passes inside: $`\mathbb E_P\nabla^jV(z)[\Delta^{\otimes j}]=\langle\nabla^jV(z),\mathbb E_P\Delta^{\otimes j}\rangle`$.
- The remainders give $`r_k=\mathbb E_PR-\mathbb E_QR`$, and $`\lvert r_k\rvert\le\mathbb E_P\lvert R\rvert+\mathbb E_Q\lvert R\rvert\le\tfrac{M_{k+1}}{(k+1)!}(m^P_{k+1}+m^Q_{k+1})`$.

For $`k=1`$, $`\mathbb E_P\Delta=\mathbb E_{Y\sim P}D_t(z,Y)=\mu_t^P(z)`$ by the tower property over $`\xi`$. $`\square`$

*Proof of (b).* The telescoping identity of [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization.md#s3-2) holds for any two protocols from a common initial law, whichever is labelled the reference. With $`P`$ as the reference and $`Q`$ as the alternative,

```math
L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\big[(K_t^P-K_t^Q)V^P_{t+1}(Z_t)\big].
```

At each visited state $`Z_t=z`$, $`(\mathrm S_1)`$ holds for $`V^P_{t+1}`$, and (a) with $`k=1`$ gives

```math
(K_t^P-K_t^Q)V^P_{t+1}(z)=\langle\nabla V^P_{t+1}(z),\mu_t^P(z)-\mu_t^Q(z)\rangle+r_{1,t}(z),\qquad\lvert r_{1,t}(z)\rvert\le\tfrac{M_{2,t}(z)}2\big(m^P_{2,t}(z)+m^Q_{2,t}(z)\big).
```

- The maps $`z\mapsto\mu_t(z)`$, $`m_{2,t}(z)`$ and $`\nabla V^P_{t+1}(z)`$ are measurable. Integrating against $`d_t^Q`$ and summing over $`t`$ gives the representation, with $`\mathcal E=\sum_t\mathbb E_{d_t^Q}r_{1,t}`$ and the stated bound on $`\lvert\mathcal E\rvert`$.
- For the certificate, apply $`\lvert\langle a,b\rangle\rvert\le\lVert a\rVert_*\lVert b\rVert`$ to each first-order term and the triangle inequality to the sum. The cap at $`1`$ holds because $`L(P),L(Q)\in[0,1]`$.
- For the stopped version, define $`\varepsilon_t(z)=\Lambda_t(z)\lVert\mu^P_t(z)-\mu^Q_t(z)\rVert+\tfrac{M_{2,t}(z)}2\big(m^P_{2,t}(z)+m^Q_{2,t}(z)\big)`$ on $`C_t`$. By (a) and the dual-norm inequality, $`\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\rvert\le\varepsilon_t(z)`$ for $`z\in C_t`$. The stopped composition of [Theorem 3.2(c)](../statements/03-02_continuation_response_characterization.md#s3-2), applied with $`P`$ as the reference, gives $`\lvert L(P)-L(Q)\rvert\le\mathbb E_Q\sum_{t<\tau}\varepsilon_t(Z_t)+\rho`$. $`\square`$

*Proof of (c).* Write $`w=\mu^P-\mu^Q`$.

- *Sufficiency.* For $`V\in\mathcal V`$, $`\nabla V(z)\in U`$. Hence $`\langle\nabla V(z),w\rangle=\langle\nabla V(z),\Pi_Uw\rangle=0`$, and (a) with $`k=1`$ leaves only $`r_1`$.
- *Necessity.* Suppose $`\langle\nabla V(z),w\rangle=0`$ for every $`V\in\mathcal V`$. Then $`w`$ is orthogonal to a spanning set of $`U`$, so $`\Pi_Uw=0`$, a contradiction. For a $`V`$ with $`\langle\nabla V(z),w\rangle\ne0`$, (a) gives $`(K^P-K^Q)V(z)=\langle\nabla V(z),w\rangle+r_1`$; the reverse triangle inequality and the bound on $`\lvert r_1\rvert`$ give the display.
- *Minimality.* For $`V\in\mathcal V`$ the first-order term is a linear functional of the message law,

```math
\langle\nabla V(z),\mu^P-\mu^Q\rangle=\sum_y\big(P(y)-Q(y)\big)\,g_V(y),\qquad g_V(y)=\langle\nabla V(z),D_t(z,y)\rangle\in\mathcal G^{(1)}_{t,z}.
```

- Determining every first-order term from the means of features $`\phi_1,\dots,\phi_d`$, for all laws, is therefore the exact-sufficiency question of [Theorem 4.2(a)](../statements/04-02_sufficiency_duality_transfer.md#s4-2) for the response class $`\mathcal G^{(1)}_{t,z}`$. That theorem gives the minimal number $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)}_{t,z})-1`$.
- For the upper bound, each $`g_V`$ has the form $`y\mapsto\langle u,D_t(z,y)\rangle`$ with $`u\in U`$. The map $`u\mapsto\langle u,D_t(z,\cdot)\rangle`$ is linear from $`U`$ into functions on the alphabet, so its image has dimension at most $`\dim U`$. Hence $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)})\le\dim U+1`$. All functions on an $`m`$-point alphabet form an $`m`$-dimensional space containing the constants, so the same quantity is also at most $`m`$. Subtracting $`1`$ gives both bounds. $`\square`$

*Proof of (d).* The hypothesis makes every term $`\langle\nabla^jV(z),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle`$, $`j\le k`$, of (a) vanish, leaving $`r_k`$.

- For SGD, $`\Delta^{\otimes j}=(-\eta)^jG^{\otimes j}`$, so matching increment moments is matching gradient moments, and $`m_{k+1}=\eta^{k+1}\mathbb E\lVert G\rVert^{k+1}`$.
- For cold Adam and for always-active clipping, the message-dependent part of the parameter block of $`\Delta`$ is $`-\eta h_\epsilon(G)`$ (Proposition 5.3(a)), respectively $`-\eta c\,G/\lVert G\rVert`$. Its tensor powers are the stated moments. $`\square`$

## Related work

- Proposition 5.1 is of the same kind as stochastic modified equations ([Li, Tai & E, 2019](#ref-li2019stochastic)), TracIn ([Pruthi et al., 2020](#ref-pruthi2020estimating)), influence functions ([Koh & Liang, 2017](#ref-koh2017understanding)), metagradients ([Engstrom et al., 2025](#ref-engstrom2025optimizing)) and In-Run Data Shapley ([Wang et al., 2025](#ref-wang2025data)); see Remark 5.2.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-engstrom2025optimizing"></a>

**Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, Aleksander Madry (2025).** Optimizing ML training with metagradient descent. arXiv preprint. arXiv [2503.13751](https://arxiv.org/abs/2503.13751).

<a id="ref-koh2017understanding"></a>

**Pang Wei Koh, Percy Liang (2017).** Understanding black-box predictions via influence functions. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1885–1894. [proceedings.mlr.press/v70/koh17a](https://proceedings.mlr.press/v70/koh17a.html).

<a id="ref-li2019stochastic"></a>

**Qianxiao Li, Cheng Tai, Weinan E (2019).** Stochastic modified equations and dynamics of stochastic gradient algorithms I: Mathematical foundations. *Journal of Machine Learning Research* 20, no. 40, 1–47. [jmlr.org/papers/v20/17-526](https://jmlr.org/papers/v20/17-526.html).

<a id="ref-pruthi2020estimating"></a>

**Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan (2020).** Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 19920–19930. arXiv [2002.08484](https://arxiv.org/abs/2002.08484).

<a id="ref-wang2025data"></a>

**Jiachen T. Wang, Prateek Mittal, Dawn Song, Ruoxi Jia (2025).** Data Shapley in one training run. *13th International Conference on Learning Representations (ICLR)*. arXiv [2406.11011](https://arxiv.org/abs/2406.11011).
