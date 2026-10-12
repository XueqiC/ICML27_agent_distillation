# §5 Optimizer state and the first-order representation — Proofs

[← Framework](../README.md) · [Experiments](../experiments/05_optimizer_state.md) · [中文](05_optimizer_state_ZH.md)

**Summary.**

- A stateful first-order optimizer is a map from a consumed gradient to an increment of the *complete* learner state $`z=(\theta,o)`$. Proposition 5.1 expands the effect of a feedback law in the moments of that increment. To first order, what a feedback law must carry is the mean increment $`\mu`$ in the directions of the continuation-value gradients. This is a tool: the calculation is of the same kind as stochastic modified equations, TracIn and metagradients (Remark 5.2).
- For Adam, the statistic depends on the optimizer state.
  - At a cold state the parameter step sees only the sign law of the gradient, while the moments retain its magnitude (Proposition 5.3).
  - At a warm state the exact statistic is the mean Adam increment. The pair $`(\mathbb EG,\mathbb E\,G\odot G)`$ determines it only under the frozen-denominator approximation, whose error is controlled by the warmth (Propositions 5.5–5.7).
  - Under a fixed $`\beta_2`$ the warmth does not vanish: after an early phase in which it falls like $`1/(\beta_2s)`$, it levels off at $`(1-\beta_2)/\beta_2`$ (Proposition 5.8).
- Under clipping the statistic is the clipped mean; the mean direction alone suffices only up to an error $`c(1-p_{\rm act})`$ (Proposition 5.9).
- Theorem 5.10 lists the first-order statistic for SGD, momentum, warm and cold Adam, clipping, Lion and LoRA. Remark 5.11 states what must be measured to check the warm-state conditions on a real model.
- Uses: the telescoping identity and the stopped composition of [Theorem 3.2](03_learning_consequence.md#s3-2); the sufficiency duality of [Theorem 4.2](04_complexity_of_sufficiency.md#s4-2); the moment-matched laws of [Theorem 4.4](04_complexity_of_sufficiency.md#s4-4); the SGD expansion of [Theorem 4.7](04_complexity_of_sufficiency.md#s4-7).
- Used by: the learner Jacobian of Section 6 ([Definition 6.2](06_learner_relative_information.md#s6-2), [Proposition 6.11](06_learner_relative_information.md#s6-11)); the state-channel decomposition of Section 7 ([Theorem 7.1](07_second_moment_channel.md#s7-1)); the measurement results of Section 8 ([Proposition 8.4](08_verification.md#s8-4), [Corollary 8.9](08_verification.md#s8-9)); the method of Section 9.

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
  - *Special cases.* For SGD, (a) with $`k=2`$ is the local SGD expansion of [Theorem 4.7](04_complexity_of_sufficiency.md#s4-7). (c) is [Theorem 4.2(a)–(c)](04_complexity_of_sufficiency.md#s4-2) applied to the linearized response class. (b) is the telescoping identity of [Theorem 3.2](03_learning_consequence.md#s3-2) with a local expansion inserted at each round.
- *What the remainder measures (order-of-magnitude heuristic; the bound itself is (b) with $`M_2`$).* Split the increment into blocks.
  - *Parameter block.* $`\Delta^\theta=O(\eta)`$, so its second-order contribution $`\tfrac12\nabla^2_{\theta\theta}V[\Delta^\theta,\Delta^\theta]`$ is $`O(\eta^2)`$.
  - *First-moment block (momentum $`m`$).* $`\Delta^m=O(1-\beta_1)`$. The continuation depends on $`m`$ through later parameter increments, linearly to leading order, with coefficient $`O(\eta/(1-\beta_1))`$. So $`\nabla^2_{mm}V=O(\eta^2/(1-\beta_1)^2)`$, and this block also contributes $`O(\eta^2)`$.
  - *Second-moment block (Adam's $`v`$).* $`\Delta^v=(1-\beta_2)(G^{\odot2}-v)`$. The continuation depends on $`v`$ through the denominators of later steps, which carry a factor $`\eta`$ and no further power of it. The block's second-order contribution is therefore first order in $`\eta`$. Its size relative to the block's first-order term is set by $`\Delta v_i/v_i^{\rm next}`$, where $`v^{\rm next}`$ is the second moment used by the next step: about the warmth $`\kappa_i`$ (Definition 5.4) at a warm state, and of order one at the cold state.
  - Hence, wherever $`(\mathrm S_1)`$ holds, the remainder of (b) is $`O(\eta^2)`$ per round for SGD, momentum and sign-type optimizers. For Adam-type optimizers over more than one round it is first order in $`\eta`$, with relative size of the order of the warmth. At $`R=1`$ the continuation after the boundary is $`F_E(\theta)`$, which does not depend on $`o`$, and the remainder is $`O(\eta^2)`$ for every optimizer.
- *Necessity in practice.* (c) detects a mismatch only when the first-order term exceeds the remainder bound. Where the remainder is $`O(\eta^2)`$, any fixed mismatch is detected for small enough $`\eta`$. For Adam-type multi-round responses the remainder is also first order in $`\eta`$, so detection requires the margin condition of [Corollary 8.9](08_verification.md#s8-9), and where the second-moment channel dominates a finite measurement is needed ([Theorem 8.8](08_verification.md#s8-8), [Remark 7.5](07_second_moment_channel.md#s7-5)).
- *Cost.* $`\mu_t^P(z)`$ needs only consumed-gradient evaluations at the state $`z`$; no training tail is needed ([Proposition 8.4](08_verification.md#s8-4)). The constants $`\Lambda_t`$ and $`M_{2,t}`$ are properties of the continuation and are not estimated by sampling messages.

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

## Warm Adam

<a id="s5-4"></a>

**Definition 5.4 (Warmth and the frozen-denominator surrogate).** Consider an Adam update with counter $`t\ge1`$ at $`z=(\theta,m,v,t-1)`$, and assume $`v_i>0`$, or $`\epsilon>0`$ whenever some $`v_i=0`$.

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

*Proof.* Fix $`y`$ and drop it from the notation. Since $`V(\tilde\theta^+,m^+,\beta_2v)=V(\tilde z^+(G))=F(G)`$,

```math
\psi-\psi_0-b^\top G=\underbrace{\big[V(z^+)-V(\tilde\theta^+,m^+,v^+)\big]}_{(i)}+\underbrace{\big[V(\tilde\theta^+,m^+,v^+)-V(\tilde\theta^+,m^+,\beta_2v)\big]}_{(ii)}+\underbrace{\big[F(G)-F(0)-\nabla F(0)^\top G\big]}_{(iii)}.
```

- *(i)* By (L) and Proposition 5.5(b), $`\lvert(i)\rvert\le\sum_iL_i\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\sum_iL_iu_i\min\lbrace \kappa_i/2,1\rbrace`$.
- *(ii)* By (W), $`v^+=\beta_2v\odot(1+\kappa)`$ with finite $`\kappa\ge0`$. By $`(\mathrm S_v)`$ at $`(\tilde\theta^+,m^+)`$, $`\lvert(ii)\rvert\le L_v\max_i\min\lbrace \kappa_i,\bar k\rbrace`$.
- *(iii)* The segment $`\lbrace \tau G:\tau\in[0,1]\rbrace`$ lies in $`\mathcal C`$. Taylor's theorem with integral remainder gives $`F(G)-F(0)-\nabla F(0)^\top G=\int_0^1(1-\tau)\,G^\top\nabla^2F(\tau G)\,G\,d\tau`$. By (H) the integrand is at most $`(1-\tau)H_G\lVert G\rVert_\infty^2`$ in absolute value, so $`\lvert(iii)\rvert\le\tfrac12H_G\lVert G\rVert_\infty^2`$.

Adding the three bounds proves the pointwise statement. For two laws, write $`\psi=\psi_0+b^\top G+e`$ with $`\lvert e\rvert\le r`$. Then $`\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi=b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)+\mathbb E_{P_0}e-\mathbb E_{P_1}e`$; the constant $`\psi_0`$ cancels because both laws are probability measures, and $`Y`$ is finite, so all expectations exist. The triangle inequality gives the bound, and $`\psi\in[0,1]`$ gives the cap at $`1`$. For $`b`$, the chain rule applies with $`\partial\tilde\theta^+/\partial g=-\eta(1-\beta_1)\Phi/(1-\beta_1^{\,t})`$, $`\partial m^+/\partial g=(1-\beta_1)I`$, and the $`v`$-component of $`\tilde z^+`$ independent of $`g`$. $`\square`$

<a id="s5-7"></a>

**Proposition 5.7 (Cold steps separate equal-mean feedback at first order).** Take a cold state ($`m=v=0`$, $`t=1`$), the ideal sign map $`h_0`$ (Proposition 5.3), a scalar parameter and the immediate utility $`V(\theta)=\tfrac12+w\theta`$, restricted to a neighbourhood where $`V\in[0,1]`$.

- **(a) Equal means.** Let $`P_0(G=1)=\tfrac14`$, $`P_0(G=-\tfrac13)=\tfrac34`$ and $`P_1(G=0)=1`$. Both laws have mean $`0`$, and $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$.
- **(b) Equal moments to any fixed order.** For $`N\ge1`$, let $`P_N(Z=i/N)=\binom Ni/2^{N-1}`$ for even $`i`$ and $`Q_N(Z=i/N)=\binom Ni/2^{N-1}`$ for odd $`i`$, $`0\le i\le N`$ (the laws of [Theorem 4.4(c)](04_complexity_of_sufficiency.md#s4-4)), and $`G=q-Z`$ with $`q\in(0,1/N)`$. The two gradient laws agree on all moments of order below $`N`$, and $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$.
- In both cases the separation is first order in $`\eta`$. At a warm state with the same utility, Proposition 5.6 bounds the gap of equal-mean laws by $`\eta\lvert w\rvert\big(\mathbb E_{P_0}[u\min\lbrace \kappa/2,1\rbrace ]+\mathbb E_{P_1}[u\min\lbrace \kappa/2,1\rbrace ]\big)`$.

*Proof.*

- *(a)* By Proposition 5.3(a) with $`h_0=\mathrm{sign}`$, $`\theta_1=\theta_0-\eta\,\mathrm{sign}(G)`$. Under $`P_0`$, $`\mathbb E\,\mathrm{sign}\,G=\tfrac14-\tfrac34=-\tfrac12`$, so $`\mathbb E\theta_1=\theta_0+\eta/2`$. Under $`P_1`$, $`\mathrm{sign}\,0=0`$, so $`\mathbb E\theta_1=\theta_0`$. The utility is affine, so the response gap is $`\lvert w\rvert\eta/2`$. The means agree: $`\tfrac14\cdot1+\tfrac34\cdot(-\tfrac13)=0`$.
- *(b)* Both $`P_N`$ and $`Q_N`$ are probability laws because $`\sum_{i\ \rm even}\binom Ni=\sum_{i\ \rm odd}\binom Ni=2^{N-1}`$. For $`\ell<N`$, $`\sum_i(-1)^i\binom Ni\,i^\ell=0`$, so their moments of order below $`N`$ agree, and so do those of the affine image $`G=q-Z`$. With $`q\in(0,1/N)`$: under $`P_N`$, $`G>0`$ exactly when $`i=0`$, which has probability $`2^{1-N}`$, and otherwise $`Z\ge2/N>q`$ and $`G<0`$, so $`\mathbb E_{P_N}\mathrm{sign}\,G=2^{2-N}-1`$. Under $`Q_N`$, $`Z\ge1/N>q`$ always, so $`\mathbb E_{Q_N}\mathrm{sign}\,G=-1`$. The expected first steps differ by $`\eta\,2^{2-N}`$, and the affine utility gives the gap $`\lvert w\rvert\eta\,2^{2-N}`$.
- *Warm counterpart.* For $`V(\theta)=\tfrac12+w\theta`$ without a tail, $`F`$ is affine in $`g`$, so $`H_G=0`$; $`V`$ does not depend on $`(m,v)`$, so $`L_v=0`$; and $`L=\lvert w\rvert`$. With equal means, Proposition 5.6 gives the displayed bound. $`\square`$

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

**(c) Accumulation along a run.** Compare a reference protocol $`0`$ and an alternative protocol $`S`$ over $`R`$ real updates $`s=1,\dots,R`$ preceded by $`W`$ warm-up evaluations, through the telescoping identity of [Theorem 3.2(a)](03_learning_consequence.md#s3-2). Let $`\mathcal S_{\rm warm}\subseteq\lbrace 1,\dots,R\rbrace`$ be the updates at which, at every visited state, the hypotheses of Proposition 5.6 hold with constants indexed by $`s`$, the two laws have equal mean consumed gradient, and, almost surely, the condition below holds for some constant $`c>0`$.

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

*Proof of (a).* Unrolling the recursion, $`v_{s,i}=(1-\beta_2)\sum_{r=1}^s\beta_2^{\,s-r}G_{r,i}^2`$, so $`\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2)\sum_{j<s}\beta_2^{\,j}=\sigma_i^2(1-\beta_2^{\,s})`$. Substituting into the definition gives $`\bar\kappa_i(s)`$. For the bounds:

- By Bernoulli's inequality, $`1-(1-\gamma)^s\le s\gamma`$, so $`\bar\kappa\ge1/(\beta_2s)`$. Since $`1-(1-\gamma)^s\le1`$, also $`\bar\kappa\ge\gamma/\beta_2`$.
- The upper bound is equivalent to $`s\gamma\le(1+s\gamma)\big(1-(1-\gamma)^s\big)`$, that is, to $`(1-\gamma)^s(1+s\gamma)\le1`$. This follows from $`(1-\gamma)^s\le e^{-s\gamma}`$ and $`e^{-y}(1+y)\le1`$ for $`y\ge0`$.
- Monotonicity holds because $`1-\beta_2^{\,s}`$ increases in $`s`$, and the limit is immediate. Multiplying the two-sided bound by $`\beta_2s`$ gives $`1\le\beta_2s\bar\kappa\le1+s\gamma`$.
- For a constant gradient, $`v_{s,i}=g_i^2(1-\beta_2^{\,s})`$ deterministically, and $`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ equals the formula. $`\square`$

*Proof of (b).* $`d_i^2/a_i^2=v_i^+/(\beta_2v_i)=1+\kappa_i`$: the factor $`1/(1-\beta_2^{\,t})`$ divides numerator and denominator alike. $`\square`$

*Proof of (c).* [Theorem 3.2(a)](03_learning_consequence.md#s3-2) gives

```math
L(0)-L(S)=\sum_{s=1}^R\mathbb E_{d_s^S}\big[(K_s^0-K_s^S)V_{s+1}^0(Z_s)\big],
```

and at each visited state $`(K_s^0-K_s^S)V_{s+1}^0=\mathbb E_{P_0}\psi_s-\mathbb E_{P_1}\psi_s`$, a difference of two $`[0,1]`$-valued expectations.

- For $`s\notin\mathcal S_{\rm warm}`$, bound its absolute value by $`1`$.
- For $`s\in\mathcal S_{\rm warm}`$, Proposition 5.6 with equal means bounds it by $`\mathbb E_{P_0}r_s+\mathbb E_{P_1}r_s`$. Use $`\min\lbrace \kappa_i/2,1\rbrace \le\min\lbrace \kappa_{\max},1\rbrace`$ and $`\min\lbrace \kappa,\bar k\rbrace \le\max\lbrace 1,\bar k\rbrace \min\lbrace 1,\kappa\rbrace`$, which holds whether $`\kappa\le1`$ or $`\kappa>1`$. With the hypothesis on $`\kappa_{\max}`$, the first two terms of $`r_s`$ are at most $`\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)\min\lbrace 1,c/(W+s)+c(1-\beta_2)\rbrace`$, and the curvature term is at most $`\tfrac12\bar B_s`$. Summing over the two laws gives the factor $`2`$.

Summing over $`s`$ proves the first display. For the second, $`\min\lbrace1,a+b\rbrace\le\min\lbrace1,a\rbrace+b`$ for $`a,b\ge0`$, so the sum is at most $`\sum_{s=1}^R c/(W+s)+c(1-\beta_2)R`$; and $`c/(W+s)\le\int_{s-1}^{s}c\,dz/(W+z)`$, so $`\sum_{s=1}^Rc/(W+s)\le c\log(1+R/W)`$ for $`W\ge1`$. For $`W=0`$ and $`s`$ running from $`2`$, $`\sum_{s=2}^Rc/s\le c\log R`$ in the same way. $`\square`$

*Proof of (d).* With learning rate $`0`$ the parameter update is $`\theta^+=(1-0\cdot\lambda_{\rm wd})\theta-0=\theta`$, while $`m`$, $`v`$ and the counter follow the same recursions as in a real step. Hence after $`W`$ warm-up evaluations and $`s-1`$ real updates, $`v`$ has absorbed $`W+s-1`$ evaluations. Under the mean-field replacement, $`v_i=\sigma_i^2(1-\beta_2^{\,k})`$ with $`k=W+s-1\ge1`$, so $`\kappa_i=(G_i^2/\sigma_i^2)\,\bar\kappa_i(k)\le\bar\gamma\,(1/k+1-\beta_2)/\beta_2`$ by (a). Since $`k\ge(W+s)/2`$ when $`W+s\ge2`$, $`1/k\le2/(W+s)`$, and therefore $`\kappa_i\le(2\bar\gamma/\beta_2)\big(1/(W+s)+(1-\beta_2)\big)`$. $`\square`$

**Discussion of Propositions 5.5–5.8.**

- *The switch.* A feedback source must match a different statistic depending on the optimizer state.
  - At a cold Adam state, it must match the joint sign law of the consumed gradient, together with the magnitudes retained in $`m`$ and $`v`$ (Proposition 5.3).
  - At a warm Adam state, the exact statistic is the mean Adam increment; to first order in the warmth it is the mean preconditioned consumed gradient together with $`\mathbb E\,G^{\odot2}`$ in the $`v`$ block (Propositions 5.5–5.6).
  - Under SGD, it must match the mean consumed gradient up to $`O(\eta^2)`$ (the local SGD expansion of [Theorem 4.7](04_complexity_of_sufficiency.md#s4-7)).
- *The warmth floor.* Under a fixed $`\beta_2`$ the warmth of a typical gradient stays near $`(1-\beta_2)/\beta_2`$ for the rest of training (Proposition 5.8(a)). The frozen-denominator description is therefore accurate only to that relative order, at every step, and the accumulated non-affine contribution has a term linear in $`R`$ (Proposition 5.8(c)).
- *Order of the curvature term.* For a smooth tail, $`F`$ depends on $`g`$ through $`\tilde\theta^+`$, which is scaled by $`\eta`$, and through $`m^+`$, which enters later updates scaled by $`\eta`$. One therefore expects $`H_G=O(\eta^2)`$. This is an expectation about the tail, not part of (H).
- *What warmth does not remove.* The curvature term $`\tfrac12H_G\lVert G\rVert_\infty^2`$ does not depend on $`\kappa`$ and accumulates linearly in $`R`$. Over long horizons the residual difference between equal-mean sources is governed by curvature, which SGD shares. Warm-up controls only the cold window.
- *Messages versus gradients.* Proposition 5.6 is affine in the *consumed* gradient. When $`G(y)`$ is itself nonlinear in $`y`$, the response is not affine in the message. Examples are sequence-level teacher forcing, where the prefix changes the Jacobian, and active global-norm clipping.
- *Locality.* Proposition 5.6 is a one-boundary statement whose constants are properties of the continuation value. A certificate along training also needs the transport and coverage terms of [Theorem 8.2](08_verification.md#s8-2).

## Clipping

<a id="s5-9"></a>

**Proposition 5.9 (Partially active clipping).** Let $`\mathrm{clip}_c(g)=g\min\lbrace1,c/\lVert g\rVert\rbrace`$ with $`c>0`$, set $`G/\lVert G\rVert=0`$ when $`G=0`$, and let $`p_{\rm act}=\Pr(\lVert G\rVert>c)`$ be the activation rate under a law of $`G`$. Then

```math
\mathbb E\,\mathrm{clip}_c(G)=\mathbb E\big[G\,1\lbrace\lVert G\rVert\le c\rbrace\big]+c\,\mathbb E\Big[\frac{G}{\lVert G\rVert}1\lbrace\lVert G\rVert>c\rbrace\Big],\qquad \Big\lVert\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E\frac{G}{\lVert G\rVert}\Big\rVert\le c\,(1-p_{\rm act}).
```

- For two laws, the clipped means differ from the scaled mean directions by at most $`c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$ in total. For instance, at an activation rate of $`0.9`$ under both laws, replacing the clipped mean by the mean direction errs by at most $`0.2c`$ in the difference.
- The always-active case $`p_{\rm act}=1`$ is the only one in which the mean direction is exactly the clipped mean.

*Proof.* The first identity splits $`\mathbb E\,\mathrm{clip}_c(G)`$ on the event $`\lbrace\lVert G\rVert\le c\rbrace`$. For the second, $`\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E[G/\lVert G\rVert]=\mathbb E\big[(G-cG/\lVert G\rVert)1\lbrace\lVert G\rVert\le c\rbrace\big]`$, and $`\lVert G-cG/\lVert G\rVert\rVert=\lvert\lVert G\rVert-c\rvert=c-\lVert G\rVert\le c`$ on that event (including $`G=0`$). Hence the norm is at most $`\mathbb E[(c-\lVert G\rVert)1\lbrace\lVert G\rVert\le c\rbrace]\le c\Pr(\lVert G\rVert\le c)=c(1-p_{\rm act})`$. The two-law statement follows by the triangle inequality. $`\square`$

## The statistic by learner

**Optimizers.** All operations are coordinatewise. Terms of $`\Delta`$ that do not depend on the message cancel in $`\mu^P-\mu^Q`$: weight decay, momentum decay and bias-correction constants applied to the old state.

- *SGD:* $`\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta G`$.
- *Heavy-ball:* $`m^+=\beta m+G`$, $`\theta^+=\theta-\eta m^+`$. *Nesterov:* $`m^+=\beta m+G`$, $`\theta^+=\theta-\eta(G+\beta m^+)`$.
- *Adam and AdamW:* as in the Setting, with the warm-Adam notation of Definition 5.4 and $`c_1=(1-\beta_1)/(1-\beta_1^{\,t})`$.
- *Global-norm clipping:* $`\mathrm{clip}_c(G)`$, applied before the base optimizer.
- *Lion form of sign optimizers:* $`u=\mathrm{sign}(\beta_1m+(1-\beta_1)G)`$, $`\theta^+=\theta-\eta u`$, $`m^+=\beta_2m+(1-\beta_2)G`$.
- *LoRA:* an adapted weight $`W=W_0+\alpha BA`$ with $`A\in\mathbb R^{r\times k}`$, $`B\in\mathbb R^{d\times r}`$ and full-weight gradient $`\Gamma=\nabla_W\ell`$. The trainable parameters are $`(A,B)`$, with the standard initialization $`B_0=0`$ and $`A_0`$ random.

For two laws $`P,Q`$ at a state write $`\delta_1=\mathbb E_PG-\mathbb E_QG`$ and $`\delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}`$. By Proposition 5.1(c), the projection of $`\mu^P-\mu^Q`$ onto the span $`U`$ of the continuation gradients is what two feedback laws must agree on to first order.

<a id="s5-10"></a>

**Theorem 5.10 (The first-order statistic by learner).**

**(a) SGD.** $`\mu^P-\mu^Q=-\eta\,\delta_1`$. The first-order statistic is the mean consumed gradient; the second-order statistic is $`\mathbb E\,GG^\top`$.

**(b) Heavy-ball and Nesterov momentum.** The increments are affine in $`G`$: heavy-ball has $`\Delta^\theta=-\eta(\beta m+G)`$ and $`\Delta^m=(\beta-1)m+G`$; Nesterov has $`\Delta^\theta=-\eta\big((1+\beta)G+\beta^2m\big)`$ and $`\Delta^m=(\beta-1)m+G`$. Hence the first-order term is as below. The statistic is the mean consumed gradient; momentum changes its weight, not the statistic.

```math
\langle\nabla V,\mu^P-\mu^Q\rangle=\big\langle-\eta\gamma\nabla_\theta V+\nabla_mV,\ \delta_1\big\rangle,\qquad\gamma=1\ \text{(heavy-ball)},\quad\gamma=1+\beta\ \text{(Nesterov)}.
```

**(c) Adam and AdamW at a warm state.** The exact first-order statistic is the mean Adam increment (Proposition 5.5(a)). In the blocks $`(\theta,m,v)`$, with $`e(G)=\theta^+-\tilde\theta^+`$ and $`\lvert e_i\rvert\le\eta u_i\min\lbrace \kappa_i/2,1\rbrace`$ (Proposition 5.5(b)),

```math
\mu^P-\mu^Q=\Big(-\eta c_1\Phi\,\delta_1+\big(\mathbb E_Pe-\mathbb E_Qe\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).
```

- The pair $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ determines $`\mu^P-\mu^Q`$ exactly under the frozen-denominator approximation, and in general up to the frozen-denominator error, at most $`\eta\,(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace \kappa_i/2,1\rbrace]`$ per coordinate. Relative to the step $`\eta u_i`$ the error is at most $`\kappa_i/2`$, and by Proposition 5.8 the warmth of a typical gradient does not vanish along training under a fixed $`\beta_2`$; it levels off near $`(1-\beta_2)/\beta_2`$.
- The coordinatewise second moment enters at first order through $`v`$, with weight $`(1-\beta_2)\nabla_vV`$. If $`\delta_1=0`$, the first-order term equals $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle+\langle\nabla_\theta V,\mathbb E_Pe-\mathbb E_Qe\rangle`$, where the second summand is at most $`\eta\sum_i\lvert\partial_{\theta_i}V\rvert(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace\kappa_i/2,1\rbrace]`$ in absolute value. Equal-mean laws can therefore differ at first order through the second-moment channel, by an amount that is not small in the warmth. [Theorem 7.1](07_second_moment_channel.md#s7-1) computes $`\nabla_vV`$ by an adjoint recursion.

**(d) Adam at the cold state.** With $`h_\epsilon`$ as in Proposition 5.3,

```math
\mu^P-\mu^Q=\Big(-\eta\big(\mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).
```

The parameter block sees only $`\mathbb E\,h_\epsilon(G)`$, which as $`\epsilon\to0`$ is the sign law. The moment blocks retain $`\mathbb E\,G`$ and $`\mathbb E\,G^{\odot2}`$. If the run ends after this update ($`R=1`$), the continuation is $`F_E(\theta)`$, and the first-order term is $`-\eta\,\langle\nabla_\theta V,\ \mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\rangle`$.

**(e) Gradient clipping.** With clipped SGD, $`\mu^P-\mu^Q=-\eta\big(\mathbb E_P\mathrm{clip}_c(G)-\mathbb E_Q\mathrm{clip}_c(G)\big)`$ exactly, so the statistic is the clipped mean.

- *Witness.* Take $`c=1`$, $`P(G=4e_1)=\tfrac14`$, $`P(G=-\tfrac43e_1)=\tfrac34`$ and $`Q=\delta_0`$. Then $`\delta_1=0`$ but $`\mathbb E_P\mathrm{clip}_1G=-\tfrac12e_1`$, and the first-order term is $`\tfrac\eta2\,\partial_{\theta_1}V`$.
- *Mean direction.* By Proposition 5.9, the clipped mean equals $`c\,\mathbb E[G/\lVert G\rVert]`$ up to $`c(1-p_{\rm act})`$ in norm under each law. Replacing the clipped means by the scaled mean directions therefore changes $`\mu^P_\theta-\mu^Q_\theta`$ by at most $`\eta c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$ in norm.
- *Always-active clipping.* If $`\lVert G\rVert>c`$ almost surely under both laws, the statistic is exactly $`c\,\mathbb E[G/\lVert G\rVert]`$. If in addition the two laws induce the same law of $`G/\lVert G\rVert`$, the increment laws coincide and the laws are equivalent at every order.
- *Clipped base optimizers.* For any base optimizer in (a)–(d) and (f), the statistics are those of the clipped gradient $`\mathrm{clip}_c(G)`$.

**(f) Sign optimizers with momentum (Lion form).** For $`m_i\ne0`$, with a zero counted as half a flip,

```math
\mathbb E\,\Delta^\theta_i=-\eta\,\mathrm{sign}(m_i)\,(1-2\pi_i),\qquad\pi_i=\Pr\lbrace \mathrm{sign}(\beta_1m_i+(1-\beta_1)G_i)\ne\mathrm{sign}(m_i)\rbrace ,
```

and $`\Delta^m=(1-\beta_2)(G-m)`$. The parameter block's statistic is the flip-probability vector $`\pi`$.

- If momentum dominates, i.e. $`(1-\beta_1)\lvert G_i\rvert<\beta_1\lvert m_i\rvert`$ almost surely under both laws, then $`\pi_i=0`$, and coordinate $`i`$'s parameter increment does not depend on the message.
- The $`m`$ block still carries $`(1-\beta_2)\delta_1`$. Its first-order effect $`\langle\nabla_mV,(1-\beta_2)\delta_1\rangle`$ vanishes wherever the continuation is locally constant in $`m`$, that is, where no later update sign changes under small changes of $`m`$.

**(g) LoRA at initialization.** The adapter gradients are $`G_A=\alpha B^\top\Gamma`$ and $`G_B=\alpha\Gamma A^\top`$. At $`B=B_0=0`$:

- $`G_A=0`$ for every message. Under SGD, under momentum started from $`m=0`$, and under cold Adam, the first update leaves $`A`$ unchanged for every law, exactly.
- The $`B`$-block statistic is $`\alpha\,\mathbb E[\Gamma]A_0^\top`$ for SGD and momentum, and $`\mathbb E\,h_\epsilon(\alpha\Gamma A_0^\top)`$ for cold Adam.
- If two laws have mean full-weight gradients that differ only by $`\Delta\Gamma`$ with $`\Delta\Gamma A_0^\top=0`$, they agree at first order in step 1 under SGD and momentum. Under cold Adam, laws whose sketches $`\Gamma A_0^\top`$ have the same sign law give identical first updates as $`\epsilon\to0`$.
- After step 1, $`B\ne0`$ and $`G_A=\alpha B^\top\Gamma`$ carries a second sketch, so the dimension of the statistic grows with training.

**Summary.**

| Part | Learner and state | First-order statistic of the feedback law | Enters through |
|---|---|---|---|
| **(a)** | SGD | $`\mathbb E\,G`$ | $`\theta`$ |
| **(b)** | heavy-ball, Nesterov | $`\mathbb E\,G`$ | $`\theta`$ and $`m`$, with weight $`-\eta\gamma\nabla_\theta V+\nabla_mV`$ |
| **(c)** | Adam/AdamW, warm | exactly the mean Adam increment; $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ under the frozen-denominator approximation, error $`O(\eta\,u\,\kappa)`$ | $`\theta`$ (preconditioned by $`\Phi`$), $`m`$, $`v`$ |
| **(d)** | Adam/AdamW, cold | $`\mathbb E\,h_\epsilon(G)`$ (the sign law as $`\epsilon\to0`$); $`\mathbb E\,G`$ and $`\mathbb E\,G^{\odot2}`$ | $`\theta`$; $`m`$ and $`v`$ |
| **(e)** | clipped SGD | $`\mathbb E\,\mathrm{clip}_c(G)`$ exactly; the mean unit direction up to $`c(1-p_{\rm act})`$, exactly when clipping is always active | $`\theta`$ |
| **(e)** | clipped base optimizer | the statistic of the base optimizer, taken of $`\mathrm{clip}_c(G)`$ | as for the base optimizer |
| **(f)** | Lion / sign with momentum | flip probabilities $`\pi`$; none in $`\theta`$ when momentum dominates | $`\theta`$; $`m`$ |
| **(g)** | LoRA at $`B=0`$, step 1 | the rank-$`r`$ sketch $`\mathbb E[\Gamma]A_0^\top`$ (SGD), or its sign law (cold Adam); $`A`$ is fixed | $`B`$ only |

*Proof.* Throughout, $`\mu^P-\mu^Q=\mathbb E_PD-\mathbb E_QD`$, where $`D(y)=\mathbb E_\xi\Delta(z,G(y,\xi))`$. Any term of $`\Delta`$ that is a fixed function of the state $`z`$ has equal expectation under both laws and cancels. Each case computes $`\Delta`$ from its update rule and takes the difference of expectations.

- *(a)* $`\Delta^\theta=-\eta\lambda_{\rm wd}\theta-\eta G`$. The first term cancels, and linearity gives $`-\eta\delta_1`$. The second-order statistic follows from Proposition 5.1(a) with $`k=2`$: $`\mathbb E\,\Delta^{\otimes2}=\eta^2\mathbb E\,GG^\top`$ up to state-only terms.
- *(b)* For heavy-ball, $`\theta^+-\theta=-\eta(\beta m+G)`$ and $`m^+-m=(\beta-1)m+G`$. For Nesterov, $`\theta^+-\theta=-\eta(G+\beta m^+)=-\eta(G+\beta^2m+\beta G)`$. The message-dependent parts are $`-\eta\gamma G`$ in $`\theta`$ and $`G`$ in $`m`$, so $`\mu^P-\mu^Q=(-\eta\gamma\delta_1,\ \delta_1)`$; pairing with $`(\nabla_\theta V,\nabla_mV)`$ gives the display.
- *(c)* For the moment blocks, $`m^+-m=(1-\beta_1)(G-m)`$ and $`v^+-v=(1-\beta_2)(G^{\odot2}-v)`$ give $`(1-\beta_1)\delta_1`$ and $`(1-\beta_2)\delta_2`$. For the parameter block, $`\tilde\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\Phi\beta_1m/(1-\beta_1^{\,t})-\eta c_1\Phi G`$ by Definition 5.4, and $`\theta^+=\tilde\theta^++e(G)`$. Taking the difference of expectations gives the $`\theta`$ block, and $`\lvert\mathbb E_Pe_i-\mathbb E_Qe_i\rvert\le\mathbb E_P\lvert e_i\rvert+\mathbb E_Q\lvert e_i\rvert`$ with Proposition 5.5(b) gives the error bound. If $`\delta_1=0`$, the $`\theta`$ block reduces to $`\mathbb E_Pe-\mathbb E_Qe`$ and the $`m`$ block vanishes, which gives the stated form of the first-order term and its bound. The statement on the size of the error along training is Proposition 5.8(a).
- *(d)* At $`m=v=0`$ and $`t=1`$, $`n=G`$ and $`d=\lvert G\rvert`$, so $`\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta h_\epsilon(G)`$ (Proposition 5.3(a)). The three increments are $`-\eta\lambda_{\rm wd}\theta-\eta h_\epsilon(G)`$, $`(1-\beta_1)G`$ and $`(1-\beta_2)G^{\odot2}`$. As $`\epsilon\to0`$, $`h_\epsilon(g)_i\to\mathrm{sign}(g_i)`$ with $`h_0(0)=0`$. If $`R=1`$ the continuation is $`F_E(\theta)`$, so $`\nabla_mV=\nabla_vV=0`$ and only the parameter block contributes.
- *(e)* With clipped SGD, $`\Delta^\theta=-\eta\,\mathrm{clip}_c(G)`$ up to weight decay. For the witness, $`\mathbb E_PG=\tfrac14\cdot4e_1+\tfrac34\cdot(-\tfrac43e_1)=0=\mathbb E_QG`$, while $`\mathrm{clip}_1(4e_1)=e_1`$ and $`\mathrm{clip}_1(-\tfrac43e_1)=-e_1`$ give $`\mathbb E_P\mathrm{clip}_1G=-\tfrac12e_1`$; hence $`\mu^P-\mu^Q=\tfrac\eta2e_1`$. The mean-direction bound is Proposition 5.9 applied under each law, with the triangle inequality. If $`\lVert G\rVert>c`$ almost surely, $`\mathrm{clip}_c(G)=cG/\lVert G\rVert`$; equal direction laws make the increments $`-\eta cG/\lVert G\rVert`$ equal in law, so every term of Proposition 5.1(a) and the exact response coincide. Clipped base optimizers apply the base update to $`\mathrm{clip}_c(G)`$.
- *(f)* Write $`c_i=\beta_1m_i+(1-\beta_1)G_i`$. For $`m_i\ne0`$: off the flip event, $`\mathrm{sign}(c_i)=\mathrm{sign}(m_i)`$; on a strict flip, $`\mathrm{sign}(c_i)=-\mathrm{sign}(m_i)`$; on $`c_i=0`$, $`\mathrm{sign}(c_i)=0=\mathrm{sign}(m_i)(1-2\cdot\tfrac12)`$. Hence $`\mathbb E\,\mathrm{sign}(c_i)=\mathrm{sign}(m_i)(1-2\pi_i)`$. Under momentum dominance $`c_i`$ has the sign of $`m_i`$ almost surely, so $`\pi_i=0`$. The $`m`$-block increment $`(1-\beta_2)(G-m)`$ gives $`(1-\beta_2)\delta_1`$. Later parameter updates depend on $`m`$ only through the signs of later $`c`$'s; where none of these changes under a small perturbation of $`m`$, the continuation is locally constant in $`m`$ and $`\nabla_mV=0`$.
- *(g)* For $`W=W_0+\alpha BA`$ the chain rule gives $`\partial\ell/\partial B_{jl}=\alpha\sum_q\Gamma_{jq}A_{lq}`$ and $`\partial\ell/\partial A_{lq}=\alpha\sum_jB_{jl}\Gamma_{jq}`$, i.e. $`G_B=\alpha\Gamma A^\top`$ and $`G_A=\alpha B^\top\Gamma`$. At $`B_0=0`$, $`G_A=0`$ identically: SGD moves $`A`$ by $`0`$; momentum from $`m=0`$ gives $`m^+=0`$; cold Adam moves $`A`$ by $`-\eta h_\epsilon(0)=0`$. The $`B`$ increment is $`-\eta\alpha\Gamma A_0^\top`$ under SGD, the same at the first momentum step, and $`-\eta h_\epsilon(\alpha\Gamma A_0^\top)`$ under cold Adam; expectations give the statistics. If $`\Delta\Gamma A_0^\top=0`$, the $`B`$-block means agree and the $`A`$-block means are zero, so $`\mu^P=\mu^Q`$ in step 1; under cold Adam, equal sign laws of the sketches give equal laws of $`h_0(\alpha\Gamma A_0^\top)`$. After one step $`B_1\ne0`$ in general, so $`G_A=\alpha B_1^\top\Gamma`$ no longer vanishes. $`\square`$

**Consequences of Theorem 5.10.**

- *Which statistic, not just which data.* The same pair of feedback laws can be first-order equivalent for one learner and first-order distinct for another. Two label laws with equal means are first-order equivalent for SGD and momentum, but can be distinct for cold Adam, sign optimizers and clipped SGD, whose increments see the sign law or the clipped mean, and for warm Adam through the second-moment channel. Teacher data must therefore be judged under the student's actual optimizer.
- *Few-shot LoRA with AdamW and global-norm clipping.* At initialization, row (g) applies: the first update sees the feedback only through the sign law of the rank-$`r`$ sketch $`\Gamma A_0^\top`$. Once clipping is active on part of the batch, rows (c)–(e) are taken of the clipped gradient, with the mean-direction simplification accurate only to $`c(1-p_{\rm act})`$.
- *Relation to Propositions 5.3 and 5.6.* Row (d) unifies Proposition 5.3(a)–(c) and (e): the parameter block carries the sign law, and the moment blocks carry the magnitudes that can reverse the second step. Row (c) is the linearized form of Proposition 5.6; it refines the residual $`L_v\kappa`$ of Proposition 5.6 into the explicit linear term $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle`$.

<a id="s5-11"></a>

**Remark 5.11 (Checking the warm-state conditions on a real model).**

- The warm-state error of Proposition 5.6 is the sensitivity-weighted sum $`\eta\sum_iL_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ plus the term $`L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace`$. A statement about quantiles of $`\kappa`$, such as "most coordinates have small warmth", does not verify either term: a few coordinates with large $`L_iu_i\kappa_i`$ can dominate the sum, and the second term depends on the maximum.
- What should be reported on a real model is therefore the weighted sum $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ and its largest terms, together with $`\max_i\kappa_i`$. The sensitivities $`L_i`$ are properties of the continuation value and are not observable from the update alone; any substitute $`\hat L_i`$, for example the magnitude of a held-out validation gradient, is a proxy and must be reported as one.
- A measured non-additivity of the update map (how far $`\Delta`$ is from affine in the message) is a property of the optimizer map, not of the task response. Whether the response is affine must be measured on the response itself, for example by the paired-replacement scores of [Definition 8.6](08_verification.md#s8-6).
- By Proposition 5.8(a), the relevant reference value for the warmth of a typical coordinate late in training is the plateau $`(1-\beta_2)/\beta_2`$, not zero.

## Related work

- The bias-corrected first step of Adam and its sign-descent behaviour are standard: Kingma & Ba (2015); Balles & Hennig (2018) on Adam as sign descent with variance adaptation; Liu et al. (RAdam, ICLR 2020) on early-step variance; signSGD (Bernstein et al., 2018) and Lion (Chen et al., 2023). Decoupled weight decay is Loshchilov & Hutter (2019).
- Adam-preconditioned gradient features for data selection after a LoRA warm-up are used by LESS (Xia et al., 2024); they correspond to the parameter block of Theorem 5.10(c). The Adam version of In-Run Data Shapley (Ding et al., 2026) uses a fixed-state approximation, i.e. the current preconditioned step only.
- Proposition 5.1 is of the same kind as stochastic modified equations (Li, Tai & E, 2019), TracIn (Pruthi et al., 2020), influence functions (Koh & Liang, 2017), metagradients (Engstrom et al., 2025) and In-Run Data Shapley (Wang et al., 2025); see Remark 5.2.
- What this section adds is the information statement: which first-order statistic of the feedback law a given optimizer makes relevant, over the complete state including $`m`$ and $`v`$, with explicit residuals as a function of warmth, and the exact long-run behaviour of the warmth under a fixed $`\beta_2`$.
