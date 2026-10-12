# §5 Optimizer state and the first-order representation — Proofs

[← Framework](../README.md) · [Experiments](../experiments/05_optimizer_state.md) · [中文](05_optimizer_state_ZH.md)

This page retains Proposition 5.1 through Proposition 5.3 and the setting; the proofs continue in [Warm Adam](05_warm_adam.md) and [learner statistics](05_learner_statistics.md).

**Summary.**

- A stateful first-order optimizer is a map from a consumed gradient to an increment of the *complete* learner state $`z=(\theta,o)`$. Proposition 5.1 expands the effect of a feedback law in the moments of that increment. To first order, what a feedback law must carry is the mean increment $`\mu`$ in the directions of the continuation-value gradients. This is a tool: the calculation is of the same kind as stochastic modified equations, TracIn and metagradients (Remark 5.2).
- For Adam, the statistic depends on the optimizer state.
  - At a cold state the parameter step sees only the sign law of the gradient, while the moments retain its magnitude (Proposition 5.3).
  - At a warm state the exact statistic is the mean Adam increment. The pair $`(\mathbb EG,\mathbb E\,G\odot G)`$ determines it only under the frozen-denominator approximation, whose error is controlled by the warmth (Propositions 5.5–5.7).
  - Under a fixed $`\beta_2`$ the warmth does not vanish: after an early phase in which it falls like $`1/(\beta_2s)`$, it levels off at $`(1-\beta_2)/\beta_2`$ (Proposition 5.8).
- Under clipping the statistic is the clipped mean; the mean direction alone suffices only up to an error $`c(1-p_{\rm act})`$ (Proposition 5.9).
- Theorem 5.10 lists the first-order statistic for SGD, momentum, warm and cold Adam, clipping, Lion and LoRA. Remark 5.11 states what must be measured to check the warm-state conditions on a real model.
- Uses: the telescoping identity and the stopped composition of [Theorem 3.2](03_learning_consequence.md#s3-2); the sufficiency duality of [Theorem 4.2](04_complexity_of_sufficiency.md#s4-2); the moment-matched laws of [Theorem 4.4](04_complexity_of_sufficiency.md#s4-4); the SGD expansion of [Theorem 4.7](04_rates_and_composition.md#s4-7).
- Used by: the learner Jacobian of Section 6 ([Definition 6.2](06_learner_relative_information.md#s6-2), [Proposition 6.11](06_learner_relative_information.md#s6-11)); the state-channel decomposition of Section 7 ([Theorem 7.1](07_second_moment_channel.md#s7-1)); the measurement results of Section 8 ([Proposition 8.4](08_certificate.md#s8-4), [Corollary 8.9](08_certificate.md#s8-9)); the method of Section 9.

## Setting

The notation of [Definitions 1.3–1.4](../README.md#s1-3) is used throughout.

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

## The first-order representation

<a id="s5-1"></a>

**Proposition 5.1 (Optimizer-transformed expansion).**

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

*Proof of (b).* The telescoping identity of [Theorem 3.2(a)](03_learning_consequence.md#s3-2) holds for any two protocols from a common initial law, whichever is labelled the reference. With $`P`$ as the reference and $`Q`$ as the alternative,

```math
L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\big[(K_t^P-K_t^Q)V^P_{t+1}(Z_t)\big].
```

At each visited state $`Z_t=z`$, $`(\mathrm S_1)`$ holds for $`V^P_{t+1}`$, and (a) with $`k=1`$ gives

```math
(K_t^P-K_t^Q)V^P_{t+1}(z)=\langle\nabla V^P_{t+1}(z),\mu_t^P(z)-\mu_t^Q(z)\rangle+r_{1,t}(z),\qquad\lvert r_{1,t}(z)\rvert\le\tfrac{M_{2,t}(z)}2\big(m^P_{2,t}(z)+m^Q_{2,t}(z)\big).
```

- The maps $`z\mapsto\mu_t(z)`$, $`m_{2,t}(z)`$ and $`\nabla V^P_{t+1}(z)`$ are measurable. Integrating against $`d_t^Q`$ and summing over $`t`$ gives the representation, with $`\mathcal E=\sum_t\mathbb E_{d_t^Q}r_{1,t}`$ and the stated bound on $`\lvert\mathcal E\rvert`$.
- For the certificate, apply $`\lvert\langle a,b\rangle\rvert\le\lVert a\rVert_*\lVert b\rVert`$ to each first-order term and the triangle inequality to the sum. The cap at $`1`$ holds because $`L(P),L(Q)\in[0,1]`$.
- For the stopped version, define $`\varepsilon_t(z)=\Lambda_t(z)\lVert\mu^P_t(z)-\mu^Q_t(z)\rVert+\tfrac{M_{2,t}(z)}2\big(m^P_{2,t}(z)+m^Q_{2,t}(z)\big)`$ on $`C_t`$. By (a) and the dual-norm inequality, $`\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\rvert\le\varepsilon_t(z)`$ for $`z\in C_t`$. The stopped composition of [Theorem 3.2(c)](03_learning_consequence.md#s3-2), applied with $`P`$ as the reference, gives $`\lvert L(P)-L(Q)\rvert\le\mathbb E_Q\sum_{t<\tau}\varepsilon_t(Z_t)+\rho`$. $`\square`$

*Proof of (c).* Write $`w=\mu^P-\mu^Q`$.

- *Sufficiency.* For $`V\in\mathcal V`$, $`\nabla V(z)\in U`$. Hence $`\langle\nabla V(z),w\rangle=\langle\nabla V(z),\Pi_Uw\rangle=0`$, and (a) with $`k=1`$ leaves only $`r_1`$.
- *Necessity.* Suppose $`\langle\nabla V(z),w\rangle=0`$ for every $`V\in\mathcal V`$. Then $`w`$ is orthogonal to a spanning set of $`U`$, so $`\Pi_Uw=0`$, a contradiction. For a $`V`$ with $`\langle\nabla V(z),w\rangle\ne0`$, (a) gives $`(K^P-K^Q)V(z)=\langle\nabla V(z),w\rangle+r_1`$; the reverse triangle inequality and the bound on $`\lvert r_1\rvert`$ give the display.
- *Minimality.* For $`V\in\mathcal V`$ the first-order term is a linear functional of the message law,

```math
\langle\nabla V(z),\mu^P-\mu^Q\rangle=\sum_y\big(P(y)-Q(y)\big)\,g_V(y),\qquad g_V(y)=\langle\nabla V(z),D_t(z,y)\rangle\in\mathcal G^{(1)}_{t,z}.
```

- Determining every first-order term from the means of features $`\phi_1,\dots,\phi_d`$, for all laws, is therefore the exact-sufficiency question of [Theorem 4.2(a)](04_complexity_of_sufficiency.md#s4-2) for the response class $`\mathcal G^{(1)}_{t,z}`$. That theorem gives the minimal number $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)}_{t,z})-1`$.
- For the upper bound, each $`g_V`$ has the form $`y\mapsto\langle u,D_t(z,y)\rangle`$ with $`u\in U`$. The map $`u\mapsto\langle u,D_t(z,\cdot)\rangle`$ is linear from $`U`$ into functions on the alphabet, so its image has dimension at most $`\dim U`$. Hence $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)})\le\dim U+1`$. All functions on an $`m`$-point alphabet form an $`m`$-dimensional space containing the constants, so the same quantity is also at most $`m`$. Subtracting $`1`$ gives both bounds. $`\square`$

*Proof of (d).* The hypothesis makes every term $`\langle\nabla^jV(z),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle`$, $`j\le k`$, of (a) vanish, leaving $`r_k`$.

- For SGD, $`\Delta^{\otimes j}=(-\eta)^jG^{\otimes j}`$, so matching increment moments is matching gradient moments, and $`m_{k+1}=\eta^{k+1}\mathbb E\lVert G\rVert^{k+1}`$.
- For cold Adam and for always-active clipping, the message-dependent part of the parameter block of $`\Delta`$ is $`-\eta h_\epsilon(G)`$ (Proposition 5.3(a)), respectively $`-\eta c\,G/\lVert G\rVert`$. Its tensor powers are the stated moments. $`\square`$

<a id="s5-2"></a>

**Remark 5.2 (Relation to stochastic modified equations, TracIn, influence functions and metagradients).**

- Proposition 5.1 is a Taylor expansion of the complete-state increment, inserted into a telescoping sum over rounds. It is the same kind of calculation as weak approximation and stochastic modified equations (Li, Tai & E, 2019), TracIn (Pruthi et al., 2020), influence functions (Koh & Liang, 2017), metagradients (Engstrom et al., 2025) and In-Run Data Shapley (Wang et al., 2025). No novelty is claimed for it; it is the tool from which the later results are derived.
- What the bookkeeping makes explicit:
  - *The object.* The first-order statistic is the $`U`$-projection of the mean increment of the complete state, including the optimizer moments $`m`$ and $`v`$. It is neither the mean gradient nor the loss. The optimizer map $`\Delta`$ alone decides which features of the gradient law enter (Theorem 5.10).
  - *Special cases.* For SGD, (a) with $`k=2`$ is the local SGD expansion of [Theorem 4.7](04_rates_and_composition.md#s4-7). (c) is [Theorem 4.2(a)–(c)](04_complexity_of_sufficiency.md#s4-2) applied to the linearized response class. (b) is the telescoping identity of [Theorem 3.2](03_learning_consequence.md#s3-2) with a local expansion inserted at each round.
- *What the remainder measures (order-of-magnitude heuristic; the bound itself is (b) with $`M_2`$).* Split the increment into blocks.
  - *Parameter block.* $`\Delta^\theta=O(\eta)`$, so its second-order contribution $`\tfrac12\nabla^2_{\theta\theta}V[\Delta^\theta,\Delta^\theta]`$ is $`O(\eta^2)`$.
  - *First-moment block (momentum $`m`$).* $`\Delta^m=O(1-\beta_1)`$. The continuation depends on $`m`$ through later parameter increments, linearly to leading order, with coefficient $`O(\eta/(1-\beta_1))`$. So $`\nabla^2_{mm}V=O(\eta^2/(1-\beta_1)^2)`$, and this block also contributes $`O(\eta^2)`$.
  - *Second-moment block (Adam's $`v`$).* $`\Delta^v=(1-\beta_2)(G^{\odot2}-v)`$. The continuation depends on $`v`$ through the denominators of later steps, which carry a factor $`\eta`$ and no further power of it. The block's second-order contribution is therefore first order in $`\eta`$. Its size relative to the block's first-order term is set by $`\Delta v_i/v_i^{\rm next}`$, where $`v^{\rm next}`$ is the second moment used by the next step: about the warmth $`\kappa_i`$ (Definition 5.4) at a warm state, and of order one at the cold state.
  - Hence, wherever $`(\mathrm S_1)`$ holds, the remainder of (b) is $`O(\eta^2)`$ per round for SGD, momentum and sign-type optimizers. For Adam-type optimizers over more than one round it is first order in $`\eta`$, with relative size of the order of the warmth. At $`R=1`$ the continuation after the boundary is $`F_E(\theta)`$, which does not depend on $`o`$, and the remainder is $`O(\eta^2)`$ for every optimizer.
- *Necessity in practice.* (c) detects a mismatch only when the first-order term exceeds the remainder bound. Where the remainder is $`O(\eta^2)`$, any fixed mismatch is detected for small enough $`\eta`$. For Adam-type multi-round responses the remainder is also first order in $`\eta`$, so detection requires the margin condition of [Corollary 8.9](08_certificate.md#s8-9), and where the second-moment channel dominates a finite measurement is needed ([Theorem 8.8](08_certificate.md#s8-8), [Remark 7.5](07_second_moment_channel.md#s7-5)).
- *Cost.* $`\mu_t^P(z)`$ needs only consumed-gradient evaluations at the state $`z`$; no training tail is needed ([Proposition 8.4](08_certificate.md#s8-4)). The constants $`\Lambda_t`$ and $`M_{2,t}`$ are properties of the continuation and are not estimated by sampling messages.

## Cold-start Adam

**Sign map.** For $`\epsilon>0`$, $`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$. For $`\epsilon=0`$, the *ideal sign update* is $`h_0(g)_i=\mathrm{sign}(g_i)`$, with $`\mathrm{sign}(0)=0`$. This convention is part of the definition; a literal floating-point Adam with $`\epsilon=0`$ would compute $`0/0`$.

**Utilities.** An *immediate-parameter utility* is a bounded measurable function of $`\theta_1`$. A *complete-state utility* is a bounded measurable function of the next complete state $`(\theta_1,m_1,v_1,1)`$. Consumed-gradient laws $`P,Q`$ are probability laws on $`\mathbb R^p`$. In the finite statements they are supported on a finite alphabet $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$. Put $`\bar\theta=(1-\eta\lambda_{\rm wd})\theta_0`$.

<a id="s5-3"></a>

**Proposition 5.3 (Cold-start Adam: sign quotient and optimizer memory).** Start Adam from a cold state $`(\theta_0,0,0,0)`$.

- **(a) First step.** $`\theta_1=\bar\theta-\eta\,h_\epsilon(G)`$.
- **(b) $`\epsilon=0`$: the joint sign law.** Two laws give equal expected values for all immediate-parameter utilities if and only if their joint sign-vector laws coincide. On a finite alphabet with $`s`$ attainable sign patterns, the minimum number of linear expectation statistics that suffices for all such utilities, over unrestricted laws on $`\mathcal A`$, is $`s-1`$; here $`s\le3^p`$, or $`s\le2^p`$ without zero coordinates. Coordinatewise sign marginals do not suffice: for $`p=2`$, let $`P`$ be uniform on $`\lbrace (1,1),(-1,-1)\rbrace`$ and $`Q`$ uniform on $`\lbrace (1,-1),(-1,1)\rbrace`$. Every coordinate has the same sign marginal and the mean step is $`0`$ under both laws, yet the utility $`F(\theta)=1\lbrace (\theta_1-\bar\theta_1)(\theta_2-\bar\theta_2)>0\rbrace`$ has expectation $`1`$ under $`P`$ and $`0`$ under $`Q`$.
- **(c) $`\epsilon>0`$: the full law.** $`h_\epsilon`$ is a coordinatewise bijection $`\mathbb R\to(-1,1)`$, with the inverse and derivative below. Equality for all immediate-parameter utilities is therefore equivalent to equality of the full consumed-gradient laws. On $`\mathcal A`$ the minimum number of statistics is $`m-1`$.

```math
h_\epsilon^{-1}(u)_i=\frac{\epsilon u_i}{1-\lvert u_i\rvert},\qquad\frac{d}{du}h_\epsilon^{-1}(u)=\frac{\epsilon}{(1-\lvert u\rvert)^2}.
```

**(d) Margin-sensitive approximation.** Let $`F:\mathbb R^p\to[0,1]`$ satisfy $`\lvert F(u)-F(w)\rvert\le\sum_iL_i\lvert u_i-w_i\rvert`$ for all pairs $`u=\bar\theta-\eta h_\epsilon(g)`$, $`w=\bar\theta-\eta h_0(g)`$ with $`g`$ in the supports. For $`a>0`$ put $`\beta_{P,i}(a)=P(0<\lvert G_i\rvert<a)`$, and let $`\Delta_{\rm sign}(F)=\lvert\mathbb E_PF(\bar\theta-\eta h_0(G))-\mathbb E_QF(\bar\theta-\eta h_0(G))\rvert`$. Then $`\Delta_{\rm sign}(F)\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$ and

```math
\big\lvert\mathbb E_PF(\bar\theta-\eta h_\epsilon(G))-\mathbb E_QF(\bar\theta-\eta h_\epsilon(G))\big\rvert\le\Delta_{\rm sign}(F)+\eta\sum_iL_i\Big[\beta_{P,i}(a)+\beta_{Q,i}(a)+\frac{2\epsilon}{a+\epsilon}\Big].
```

**(e) Retained optimizer state.** For $`\beta_1<1`$, the first complete state contains $`m_1=(1-\beta_1)G`$, an injective function of $`G`$.

- *Witness.* Take one coordinate, $`\beta_1=\beta_2=\tfrac12`$, $`\lambda_{\rm wd}=0`$, $`\epsilon=0`$, $`\eta>0`$ and a cold start. Compare the gradient histories $`(G_1,G_2)=(1,-\tfrac34)`$ and $`(G_1',G_2')=(2,-\tfrac34)`$. Both give $`\theta_1=\theta_1'=\theta_0-\eta`$ and the same sign history $`(+,-)`$. At step 2, the bias-corrected moments are $`\widehat m_2=-\tfrac16`$ and $`\widehat m_2'=+\tfrac16`$, and $`\widehat v_2=\tfrac{17}{24}`$ and $`\widehat v_2'=\tfrac{41}{24}`$, so the second steps below move in opposite directions. The bounded test $`F(\theta)=1\lbrace \theta>\theta_1\rbrace`$ separates the histories by $`1`$. The histories are realizable by squared losses $`\tfrac12(\theta-a)^2`$ at $`\theta_0=0`$: first targets $`a=-1`$ and $`a=-2`$, then the common second target $`a=\theta_1+\tfrac34`$.

```math
\theta_2=\theta_1+\frac{\eta}{6\sqrt{17/24}},\qquad\theta_2'=\theta_1-\frac{\eta}{6\sqrt{41/24}}.
```

- *Momentum carries the memory.* If $`\beta_1=0`$, any two histories with the same sign of $`G_2`$ have second steps in the same direction, so no witness of this type exists. For every $`\beta_1\in(0,1)`$ one exists.

*Proof of (a).* Cold moments give $`m_1=(1-\beta_1)G`$ and $`v_1=(1-\beta_2)G^{\odot2}`$. Dividing by the bias corrections at $`t=1`$ gives $`n=G`$ and $`d^2=G^{\odot2}`$, so $`d=\lvert G\rvert`$ coordinatewise. Substituting gives $`\theta_1=\bar\theta-\eta\,G/(\lvert G\rvert+\epsilon)`$. At $`\epsilon=0`$ the declared convention gives $`h_0=\mathrm{sign}`$. $`\square`$

*Proof of (b).* The map $`\sigma\mapsto\bar\theta-\eta\sigma`$ from $`\lbrace -1,0,1\rbrace ^p`$ to $`\mathbb R^p`$ is injective, because $`\eta>0`$. Hence $`\theta_1`$ is an injective function of the sign vector $`\sigma(G)`$.

- *Sufficiency.* If the joint sign laws coincide, the laws of $`\theta_1`$ coincide, and so do all expected utilities.
- *Necessity.* For each attainable pattern $`\sigma^\star`$, the indicator $`F=1\lbrace \theta_1=\bar\theta-\eta\sigma^\star\rbrace`$ is a bounded utility with expectation $`\Pr(\sigma(G)=\sigma^\star)`$. Equal expectations for all such $`F`$ force equal pattern probabilities.
- *Dimension.* View every utility as a function on $`\mathcal A`$, namely $`u_F(g)=F(\bar\theta-\eta h_0(g))`$. The family $`\mathcal U=\lbrace u_F\rbrace`$ consists exactly of the functions on $`\mathcal A`$ that are constant on the fibres of $`\sigma`$: it contains them because $`F`$ is arbitrary on the $`s`$ distinct output points, and nothing else because $`u_F`$ depends on $`g`$ only through $`\sigma(g)`$. So $`\mathrm{span}(\lbrace 1\rbrace \cup\mathcal U)`$ is spanned by the $`s`$ fibre indicators and has dimension $`s`$. By [Theorem 4.2(a)](04_complexity_of_sufficiency.md#s4-2), the minimum number of linear statistics whose equal means imply equal means of every $`u\in\mathcal U`$, for all laws on $`\mathcal A`$, is $`s-1`$; the indicators of $`s-1`$ patterns attain it. Each coordinate sign takes at most three values, so $`s\le3^p`$, and $`s\le2^p`$ when $`0`$ is excluded.
- *Marginals.* Under both laws of the witness each coordinate equals $`\pm1`$ with probability $`1/2`$, and $`\mathbb E h_0(G)=0`$. Under $`P`$ the two coordinates of $`\theta_1-\bar\theta=-\eta\sigma`$ always share a sign, so $`F=1`$; under $`Q`$ they never do, so $`F=0`$. $`\square`$

*Proof of (c).* Fix $`\epsilon>0`$. On $`g\ge0`$, $`h_\epsilon(g)=g/(g+\epsilon)`$ is continuous and strictly increasing from $`0`$ to $`1`$. By oddness, $`h_\epsilon`$ is a bijection $`\mathbb R\to(-1,1)`$.

- Solve $`u=g/(\lvert g\rvert+\epsilon)`$. Then $`\mathrm{sign}u=\mathrm{sign}g`$ and $`\lvert u\rvert(\lvert g\rvert+\epsilon)=\lvert g\rvert`$, so $`\lvert g\rvert=\epsilon\lvert u\rvert/(1-\lvert u\rvert)`$ and $`g=\epsilon u/(1-\lvert u\rvert)`$. For $`u>0`$ the derivative is $`\epsilon/(1-u)+\epsilon u/(1-u)^2=\epsilon/(1-u)^2`$, and by symmetry the same formula holds for $`u<0`$.
- Since $`g\mapsto\bar\theta-\eta h_\epsilon(g)`$ is injective, the argument of (b) applies with fibres equal to singletons. Equality for all bounded utilities is then equivalent to equality of the laws of $`G`$, and on $`\mathcal A`$ the dimension is $`m-1`$. $`\square`$

*Proof of (d).* Fix a coordinate $`i`$ and a gradient value $`g`$.

- If $`g_i=0`$, then $`h_\epsilon(g)_i=0=h_0(g)_i`$.
- If $`g_i\ne0`$, then $`\lvert h_\epsilon(g_i)-\mathrm{sign}(g_i)\rvert=\lvert g_i\rvert\cdot\big\lvert\tfrac1{\lvert g_i\rvert+\epsilon}-\tfrac1{\lvert g_i\rvert}\big\rvert=\tfrac{\epsilon}{\lvert g_i\rvert+\epsilon}`$. This is at most $`1`$ always, and at most $`\epsilon/(a+\epsilon)`$ when $`\lvert g_i\rvert\ge a`$.

Hence $`\mathbb E_P\lvert h_\epsilon(G_i)-h_0(G_i)\rvert\le\beta_{P,i}(a)+\epsilon/(a+\epsilon)`$. By the Lipschitz hypothesis applied pointwise to the pair $`\big(\bar\theta-\eta h_\epsilon(G),\,\bar\theta-\eta h_0(G)\big)`$,

```math
\big\lvert\mathbb E_PF(\bar\theta-\eta h_\epsilon(G))-\mathbb E_PF(\bar\theta-\eta h_0(G))\big\rvert\le\eta\sum_iL_i\Big[\beta_{P,i}(a)+\frac{\epsilon}{a+\epsilon}\Big],
```

and the same holds under $`Q`$. The triangle inequality through the two ideal-sign expectations gives the bound. For the total-variation statement, $`F(\bar\theta-\eta h_0(G))=f(\sigma(G))`$ for some $`f:\lbrace -1,0,1\rbrace ^p\to[0,1]`$, and any such $`f`$ satisfies $`\lvert\mathbb E_Pf-\mathbb E_Qf\rvert\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$. $`\square`$

*Proof of (e).* Injectivity of $`m_1=(1-\beta_1)G`$ is immediate for $`\beta_1<1`$. For the witness, $`1-\beta_1=1-\beta_2=\tfrac12`$, and the bias corrections are $`\tfrac12`$ at $`t=1`$ and $`\tfrac34`$ at $`t=2`$.

- *Step 1.* $`m_1=G_1/2`$ and $`v_1=G_1^2/2`$, so $`\widehat m_1=G_1`$ and $`\widehat v_1=G_1^2`$. Since $`G_1>0`$ in both histories, $`\theta_1=\theta_0-\eta`$.
- *Step 2.* $`m_2=\tfrac12m_1+\tfrac12G_2=\tfrac{G_1}4-\tfrac38`$, which is $`-\tfrac18`$ for $`G_1=1`$ and $`+\tfrac18`$ for $`G_1=2`$; then $`\widehat m_2=\tfrac43m_2=\mp\tfrac16`$. Similarly $`v_2=\tfrac12v_1+\tfrac12G_2^2=\tfrac{G_1^2}4+\tfrac9{32}`$, which is $`\tfrac{17}{32}`$ or $`\tfrac{41}{32}`$, so $`\widehat v_2=\tfrac{17}{24}`$ or $`\tfrac{41}{24}`$.
- With $`\epsilon=0`$, $`\theta_2=\theta_1-\eta\widehat m_2/\sqrt{\widehat v_2}`$, which is the display. The denominators are positive and the numerators have opposite signs, so the steps are opposite, and $`F(\theta_2)=1`$, $`F(\theta_2')=0`$.
- *Realizability.* The gradient of $`\tfrac12(\theta-a)^2`$ is $`\theta-a`$. At $`\theta_0=0`$ the targets $`a=-1,-2`$ give $`G_1=1,2`$. Both first steps land at $`\theta_1=-\eta`$, so the common target $`a=\theta_1+\tfrac34`$ gives $`G_2=-\tfrac34`$ in both histories. No branch-dependent future gradient is needed.
- *Positive $`\epsilon`$.* For the same gradient sequences the direction of the second step is $`-\mathrm{sign}(\widehat m_2)`$ for every $`\epsilon\ge0`$, because the denominator $`\sqrt{\widehat v_2}+\epsilon`$ is positive; the two directions stay opposite.
- *Momentum.* If $`\beta_1=0`$ then $`m_2=G_2`$, and the second step has the sign of $`-G_2`$ whatever the first gradient was. If $`\beta_1>0`$, then $`m_2=\beta_1(1-\beta_1)G_1+(1-\beta_1)G_2`$. For $`0<G_1<G_1'`$, choosing $`G_2<0`$ with $`\lvert G_2\rvert`$ strictly between $`\beta_1G_1`$ and $`\beta_1G_1'`$ gives opposite signs of $`m_2`$, while the first steps $`-\eta\,\mathrm{sign}(G_1)`$ coincide at $`\epsilon=0`$. $`\square`$

**Discussion of Proposition 5.3.**

- *Three levels of sufficiency.* Immediate-parameter sufficiency of a cold step needs the joint sign law at $`\epsilon=0`$, and the full law at $`\epsilon>0`$. Complete-state sufficiency needs the full consumed-gradient law, because $`m_1`$ is injective. Task sufficiency needs only the response class of the actual continuation ([Theorem 3.2](03_learning_consequence.md#s3-2)), which may be much smaller. The exact statistic counts concern unrestricted laws; a restricted task family can need fewer.
- *The discontinuity at $`\epsilon=0`$ is exact, not operational.* For arbitrary measurable utilities, any $`\epsilon>0`$ makes magnitudes identifiable. For Lipschitz utilities with little gradient mass near zero, (d) shows that the sign approximation is accurate. The inverse in (c) is ill-conditioned as $`\lvert u\rvert\to1`$.
- *Warm states move the threshold.* At a warm state the numerator of coordinate $`i`$ changes sign at $`G_i=-\beta_1m_i/(1-\beta_1)`$, and the denominator depends on $`v_i`$ and $`G_i^2`$. For a scalar cross-entropy gradient $`G=p-Z`$, the cold sign probabilities are the masses of $`Z`$ below, at and above the current prediction $`p`$. Allowing all thresholds identifies the distribution function of $`Z`$; no fixed threshold or marginal sign feature is sufficient across changing states.
- *Consequence for the affine theory.* Two label laws can match all moments below order $`N`$ ([Theorem 4.4](04_complexity_of_sufficiency.md#s4-4)) and still differ in their sign probabilities about the current prediction. A cold Adam step separates them at first order in $`\eta`$ (Proposition 5.7).
