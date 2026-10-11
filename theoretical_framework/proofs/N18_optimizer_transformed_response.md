# N18 · Optimizer-transformed linear response — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N18_optimizer_transformed_response.md) · [中文](N18_optimizer_transformed_response_ZH.md)

## Setting and assumptions

**State.** The complete learner state is $`x=(\theta,s)\in\mathcal X\subseteq\mathbb R^p\times\mathbb R^q`$. Here $`\theta`$ are the trainable parameters, for example LoRA adapters, and $`s`$ is the optimizer state, for example Adam's moments $`m,v`$. The step counter is absorbed into the round index $`t`$.

**Feedback and gradients.** At round $`t`$ a feedback law $`P_t(\cdot\mid x)`$ emits a message $`Y`$. The learner computes a consumed gradient $`g=G_t(x,Y,\xi)\in\mathbb R^p`$ after loss normalization and accumulation, with $`\xi`$ the declared update randomness. Clipping, if any, is part of the optimizer.

**Optimizer.** The optimizer is a measurable increment map

```math
x^+=x+\Delta_t(x,g).
```

**Kernel.** The learning kernel of law $`P`$ is

```math
(K_t^Pf)(x)=\mathbb E_{Y\sim P_t(\cdot\mid x),\,\xi}\,f\big(x+\Delta_t(x,G_t(x,Y,\xi))\big).
```

The compared interfaces share the optimizer and the gradient map and differ only in the message law. A change of optimizer is a different $`\Delta`$, compared the same way.

**Increment moments.** For a law $`P`$ at $`(t,x)`$, the message-level mean increment is $`D_t(x,y)=\mathbb E_\xi\Delta_t(x,G_t(x,y,\xi))`$. Define

```math
\mu_t^P(x)=\mathbb E_{Y\sim P}D_t(x,Y),\qquad m_{k,t}^P(x)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(x,G_t(x,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(x,G_t(x,Y,\xi))^{\otimes j}.
```

**Continuation.** $`V^P_{t+1}:\mathcal X\to[0,1]`$ is the continuation of protocol $`P`$: the later updates run under $`P`$, then the student is deployed and scored by $`F_E`$ (node [N05](N05_continuation_response.md)). $`L(P)=\mathbb E\,F_E(X_R^P)`$, and $`d_t^Q`$ is the law of $`X_t`$ under protocol $`Q`$. A norm $`\lVert\cdot\rVert`$ on $`\mathbb R^{p+q}`$ is fixed, with dual norm $`\lVert\cdot\rVert_*`$.

**Assumption $`S_k`$ (smooth continuation on the reachable segment).** Let $`\mathcal S_{t,x}`$ be the convex hull of $`x`$ and of all points $`x+\Delta_t(x,g)`$, with $`g`$ in the support of the consumed gradients under the compared laws. Then:

- $`V=V_{t+1}`$ is $`C^{k+1}`$ on $`\mathcal S_{t,x}`$;
- $`\lvert\nabla^{k+1}V(w)[h,\dots,h]\rvert\le M_{k+1}\lVert h\rVert^{k+1}`$ for every $`w\in\mathcal S_{t,x}`$ and every $`h`$;
- the moments $`m^P_{k+1,t}(x)`$ and $`m^Q_{k+1,t}(x)`$ are finite.

**Scope of $`S_k`$.** Under clipping or sign updates, a multi-step continuation is generally not smooth at states where a later gradient crosses the clip threshold or a later update sign flips. $`S_k`$ then holds only on regions that avoid these kinks, for example where clipping is always active or momentum dominates every sign. The increment means of these optimizers are computed exactly in node [N19](N19_learner_statistics.md) without any smoothness assumption. The remainder bounds below are claimed only where $`S_k`$ holds.

**Declared family.** For (c), fix a boundary $`(t,x)`$ and a set $`\mathcal V`$ of continuations, one per deployment task and suffix of interest. Each satisfies $`S_1`$ on $`\mathcal S_{t,x}`$ with the common Hessian bound $`M_2`$. Let

```math
U=\mathrm{span}\lbrace \nabla V(x):V\in\mathcal V\rbrace ,\qquad \mathcal G^{(1)}_{t,x}=\lbrace\,y\mapsto\langle\nabla V(x),D_t(x,y)\rangle:\ V\in\mathcal V\,\rbrace .
```

$`\mathcal G^{(1)}_{t,x}`$ is the first-order response class.

## Statement

**(a)** **Theorem (optimizer-transformed moment expansion).** Under $`S_k`$,

```math
(K_t^P-K_t^Q)V(x)=\sum_{j=1}^{k}\frac1{j!}\Big\langle\nabla^jV(x),\;\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\Big\rangle+r_k,\qquad\lvert r_k\rvert\le\frac{M_{k+1}}{(k+1)!}\big(m_{k+1}^P+m_{k+1}^Q\big).
```

In particular, for $`k=1`$,

```math
(K_t^P-K_t^Q)V(x)=\big\langle\nabla V(x),\mu_t^P(x)-\mu_t^Q(x)\big\rangle+r_1,\qquad\lvert r_1\rvert\le\tfrac{M_2}2\big(m_2^P+m_2^Q\big).
```

**(b)** **Theorem (run-level first-order representation and certificate).** Compare protocols $`P`$ and $`Q`$ over $`R`$ rounds from a common initial law. Assume $`S_1`$ at every state visited by $`Q`$, with constants $`M_{2,t}(x)`$ for the continuation $`V^P_{t+1}`$. Then

```math
L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\Big\langle\nabla V^P_{t+1}(X_t),\;\mu_t^P(X_t)-\mu_t^Q(X_t)\Big\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\sum_t\mathbb E_{d_t^Q}\Big[\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big].
```

Consequently, with $`\Lambda_t(x)=\lVert\nabla V^P_{t+1}(x)\rVert_*`$,

```math
\lvert L(P)-L(Q)\rvert\le\min\Big\lbrace 1,\ \sum_t\mathbb E_{d_t^Q}\Big[\Lambda_t\,\lVert\mu_t^P-\mu_t^Q\rVert+\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big]\Big\rbrace .
```

*Stopped version.* Suppose $`S_1`$ holds only on certified sets $`C_t\subseteq\mathcal X`$. Let $`\tau=\min\lbrace t<R:X_t^Q\notin C_t\rbrace`$, with $`\tau=R`$ if there is no such $`t`$, and $`\rho=\Pr(\tau<R)`$. Then

```math
\lvert L(P)-L(Q)\rvert\le\mathbb E_Q\sum_{t<\tau}\Big[\Lambda_t\,\lVert\mu_t^P-\mu_t^Q\rVert+\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big]\,(X_t)+\rho.
```

**(c)** **Theorem (first-order sufficiency, necessity and minimality).** Use the Euclidean pairing, and let $`\Pi_U`$ be the orthogonal projection onto $`U`$.

**1.** *Sufficiency.* If $`\Pi_U(\mu^P-\mu^Q)=0`$, then $`\lvert(K^P-K^Q)V(x)\rvert\le\tfrac{M_2}2(m_2^P+m_2^Q)`$ for every $`V\in\mathcal V`$.

**2.** *Necessity.* If $`\Pi_U(\mu^P-\mu^Q)\ne0`$, then some $`V\in\mathcal V`$ has $`\langle\nabla V(x),\mu^P-\mu^Q\rangle\ne0`$. For every such $`V`$,

```math
\lvert(K^P-K^Q)V(x)\rvert\ge\big\lvert\langle\nabla V(x),\mu^P-\mu^Q\rangle\big\rvert-\tfrac{M_2}2(m_2^P+m_2^Q).
```

The mismatch is therefore detected by that task whenever the first-order term exceeds the remainder bound.

**3.** *Minimality.* On a finite message alphabet of size $`m`$, the minimal number of linear expectation statistics that determine every first-order term $`\langle\nabla V(x),\mu^P-\mu^Q\rangle`$, $`V\in\mathcal V`$, for all law pairs is

```math
d_0^{(1)}=\dim\mathrm{span}\big(\lbrace 1\rbrace \cup\mathcal G^{(1)}_{t,x}\big)-1\le\min\lbrace \dim U,\ m-1\rbrace .
```

**(d)** **Corollary (optimizer-transformed moment hierarchy).** Under $`S_k`$, suppose two laws agree on $`\mathbb E\Delta^{\otimes j}`$ for $`j=1,\dots,k`$ at a state. Then their one-boundary discrepancy there is at most $`\tfrac{M_{k+1}}{(k+1)!}(m_{k+1}^P+m_{k+1}^Q)`$. Summed over the visited states, as in (b), this bounds the run-level gap.

- For SGD, $`\Delta=-\eta G`$, so the matched quantities are the gradient moments $`\mathbb E G^{\otimes j}`$, and the bound is $`O(\eta^{k+1})`$ per round.
- For cold Adam, the parameter block matches the moments of $`h_\epsilon(G)`$. As $`\epsilon\to0`$ these are the joint sign moments.
- Under always-active clipping, the parameter block involves only the moments of the direction $`G/\lVert G\rVert`$.

## Proof

**Proof of (a).** Fix a realized increment $`\Delta=\Delta_t(x,g)`$ with $`g`$ in the support. The segment $`\lbrace x+u\Delta:u\in[0,1]\rbrace`$ lies in $`\mathcal S_{t,x}`$, because $`\mathcal S_{t,x}`$ is convex and contains both endpoints. On this segment $`\varphi(u)=V(x+u\Delta)`$ is $`C^{k+1}`$, with $`\varphi^{(j)}(u)=\nabla^jV(x+u\Delta)[\Delta^{\otimes j}]`$. Taylor's theorem with Lagrange remainder gives some $`u^*\in(0,1)`$ such that

```math
V(x+\Delta)=V(x)+\sum_{j=1}^{k}\frac1{j!}\nabla^jV(x)[\Delta^{\otimes j}]+R(\Delta),\qquad R(\Delta)=\frac1{(k+1)!}\nabla^{k+1}V(x+u^*\Delta)[\Delta^{\otimes(k+1)}].
```

By $`S_k`$, $`\lvert R(\Delta)\rvert\le\tfrac{M_{k+1}}{(k+1)!}\lVert\Delta\rVert^{k+1}`$.

Integrate this identity over $`Y\sim P`$ and $`\xi`$, and separately over $`Y\sim Q`$ and $`\xi`$. Each term is integrable: $`V`$ takes values in $`[0,1]`$, $`\lvert R\rvert`$ is dominated by a multiple of $`\lVert\Delta\rVert^{k+1}`$ with finite expectation, and each $`\nabla^jV(x)[\Delta^{\otimes j}]`$, $`j\le k`$, is a polynomial in $`\Delta`$ whose degree-$`j`$ moments are finite by Lyapunov's inequality. Subtract the two integrals.

- The constant $`V(x)`$ cancels, because both laws are probability measures.
- The map $`\Delta^{\otimes j}\mapsto\nabla^jV(x)[\Delta^{\otimes j}]`$ is linear, so the expectation passes inside: $`\mathbb E_P\nabla^jV(x)[\Delta^{\otimes j}]=\langle\nabla^jV(x),\mathbb E_P\Delta^{\otimes j}\rangle`$.
- The remainders give $`r_k=\mathbb E_PR-\mathbb E_QR`$, and $`\lvert r_k\rvert\le\mathbb E_P\lvert R\rvert+\mathbb E_Q\lvert R\rvert\le\tfrac{M_{k+1}}{(k+1)!}(m^P_{k+1}+m^Q_{k+1})`$.

For $`k=1`$, $`\mathbb E_P\Delta=\mathbb E_{Y\sim P}D_t(x,Y)=\mu_t^P(x)`$ by the tower property over $`\xi`$. $`\square`$

**Proof of (b).** the telescoping identity [N05(a)](N05_continuation_response.md), holds for any two protocols from a common initial law, whichever is labelled the reference. With $`P`$ as the reference and $`Q`$ as the alternative,

```math
L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\big[(K_t^P-K_t^Q)V^P_{t+1}(X_t)\big].
```

At each visited state $`X_t=x`$, $`S_1`$ holds for $`V^P_{t+1}`$. (a) with $`k=1`$ gives

```math
(K_t^P-K_t^Q)V^P_{t+1}(x)=\langle\nabla V^P_{t+1}(x),\mu_t^P(x)-\mu_t^Q(x)\rangle+r_{1,t}(x),\qquad\lvert r_{1,t}(x)\rvert\le\tfrac{M_{2,t}(x)}2\big(m^P_{2,t}(x)+m^Q_{2,t}(x)\big).
```

The maps $`x\mapsto\mu_t(x)`$, $`m_{2,t}(x)`$ and $`\nabla V^P_{t+1}(x)`$ are measurable. Integrate against $`d_t^Q`$ and sum over $`t`$; this gives the representation, with $`\mathcal E=\sum_t\mathbb E_{d_t^Q}r_{1,t}`$ and the stated bound on $`\lvert\mathcal E\rvert`$.

For the certificate, apply $`\lvert\langle a,b\rangle\rvert\le\lVert a\rVert_*\lVert b\rVert`$ to each first-order term and the triangle inequality to the sum. The cap at $`1`$ holds because $`L(P),L(Q)\in[0,1]`$.

For the stopped version, define $`\varepsilon_t(x)=\Lambda_t(x)\lVert\mu^P_t(x)-\mu^Q_t(x)\rVert+\tfrac{M_{2,t}(x)}2\big(m^P_{2,t}(x)+m^Q_{2,t}(x)\big)`$ on $`C_t`$. By (a) and the dual-norm inequality, $`\lvert(K^P_t-K^Q_t)V^P_{t+1}(x)\rvert\le\varepsilon_t(x)`$ for $`x\in C_t`$. [N05(c)](N05_continuation_response.md) of node N05, applied with $`P`$ as the reference, gives $`\lvert L(P)-L(Q)\rvert\le\mathbb E_Q\sum_{t<\tau}\varepsilon_t(X_t)+\rho`$. $`\square`$

**(c)**

*(1)* For $`V\in\mathcal V`$, $`\nabla V(x)\in U`$. Hence $`\langle\nabla V(x),w\rangle=\langle\nabla V(x),\Pi_Uw\rangle`$ for $`w=\mu^P-\mu^Q`$, which is $`0`$ by hypothesis. (a) with $`k=1`$ leaves only $`r_1`$.

*(2)* Suppose $`\langle\nabla V(x),w\rangle=0`$ for every $`V\in\mathcal V`$. Then $`w`$ is orthogonal to every element of a spanning set of $`U`$, so $`w\perp U`$ and $`\Pi_Uw=0`$, a contradiction. For a $`V`$ with $`\langle\nabla V(x),w\rangle\ne0`$, (a) gives $`(K^P-K^Q)V(x)=\langle\nabla V(x),w\rangle+r_1`$. The reverse triangle inequality and the bound on $`\lvert r_1\rvert`$ give the display.

*(3)* For $`V\in\mathcal V`$, the first-order term is

```math
\langle\nabla V(x),\mu^P-\mu^Q\rangle=\sum_y\big(P(y)-Q(y)\big)\,g_V(y),\qquad g_V(y)=\langle\nabla V(x),D_t(x,y)\rangle\in\mathcal G^{(1)}_{t,x}.
```

Determining every first-order term from the means of features $`\phi_1,\dots,\phi_d`$, for all laws, is therefore the exact-sufficiency question of [N06(a)](N06_sufficiency_duality.md), node [N06](N06_sufficiency_duality.md), for the response class $`\mathcal G^{(1)}_{t,x}`$. That theorem gives the minimal number $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)}_{t,x})-1`$.

For the upper bound, each $`g_V`$ has the form $`y\mapsto\langle u,D_t(x,y)\rangle`$ with $`u\in U`$. The map $`u\mapsto\langle u,D_t(x,\cdot)\rangle`$ is linear from $`U`$ into functions on the alphabet, so its image has dimension at most $`\dim U`$. Hence $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)})\le\dim U+1`$. All functions on an $`m`$-point alphabet form an $`m`$-dimensional space containing the constants, so the same quantity is also at most $`m`$. Subtracting $`1`$ gives both bounds. $`\square`$

**Proof of (d).** The hypothesis makes every term $`\langle\nabla^jV(x),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle`$, $`j\le k`$, of (a) vanish, leaving $`r_k`$.

- For SGD, $`\Delta^{\otimes j}=(-\eta)^jG^{\otimes j}`$, so matching increment moments is matching gradient moments, and $`m_{k+1}=\eta^{k+1}\mathbb E\lVert G\rVert^{k+1}`$.
- For cold Adam and for always-active clipping, the parameter block of $`\Delta`$ is $`-\eta h_\epsilon(G)`$, respectively $`-\eta c\,G/\lVert G\rVert`$. Its tensor powers are the stated moments. $`\square`$

## Remarks

1. **The object.** To first order, what a feedback law must preserve for a given learner and task family is the $`U`$-projection of the mean *optimizer increment* over the complete state, including $`m`$ and $`v`$. It is not the mean gradient, the loss or the teacher's accuracy. The optimizer map $`\Delta`$ alone decides which features of the gradient law enter. Node [N19](N19_learner_statistics.md) computes this statistic for each common learner.

2. **Relation to the earlier nodes.** For SGD, (a) with $`k=2`$ is [N10(c)](N10_update_composition.md). (c) is [N06(a)–(c)](N06_sufficiency_duality.md) of node [N06](N06_sufficiency_duality.md) applied to the linearized response class. (b) is the telescoping identity of node [N05](N05_continuation_response.md) with a local expansion inserted at each round. [N12(W1)](N12_optimizer_warmth.md) of node [N12](N12_optimizer_warmth.md) is the warm-Adam case of (a), with the $`v`$-block written as a residual instead of a linear term.

3. **What the remainder is small in (order-of-magnitude heuristic; the bound itself is (b) with $`M_2`$).** Split the increment into blocks.
   - *Parameter block.* $`\Delta^\theta=O(\eta)`$, so its second-order contribution $`\tfrac12\nabla^2_{\theta\theta}V[\Delta^\theta,\Delta^\theta]`$ is $`O(\eta^2)`$.
   - *First-moment block (momentum $`m`$).* $`\Delta^m=O(1-\beta_1)`$. The continuation depends on $`m`$ through later parameter increments, linearly to leading order, with coefficient $`O(\eta/(1-\beta_1))`$. So $`\nabla^2_{mm}V=O(\eta^2/(1-\beta_1)^2)`$, and this block also contributes $`O(\eta^2)`$.
   - *Second-moment block (Adam's $`v`$).* $`\Delta^v=(1-\beta_2)(G^{\odot2}-v)`$. The continuation depends on $`v`$ through the denominators $`(\hat v)^{-1/2}`$ of later steps, and these denominators carry a factor $`\eta`$ but no further power. The block's second-order contribution is therefore **first order in $`\eta`$**. Its size relative to the block's first-order term is set by $`\Delta v_i/v_i^{\rm next}`$, where $`v^{\rm next}`$ is the second moment used by the next step. At a warm state this ratio is about the warmth $`\kappa_i`$ of node [N12](N12_optimizer_warmth.md). At the cold state it is of order one.

   Hence, wherever $`S_1`$ holds, the (b) remainder is $`O(\eta^2)`$ per round for SGD, momentum and sign-type optimizers. For Adam-type optimizers over more than one round it is first order in $`\eta`$, and its absolute size shrinks with warm-up. At $`R=1`$ the continuation after the boundary is $`F_E(\theta)`$, which does not depend on $`s`$; the remainder is then $`O(\eta^2)`$ for every optimizer.

4. **Necessity in practice.** (c)(2) detects a mismatch only when the first-order term exceeds the remainder bound. Where the remainder is $`O(\eta^2)`$, any fixed mismatch is detected for small enough $`\eta`$. For Adam-type multi-round responses the remainder is also first order in $`\eta`$, so detection requires the margin condition of node [N20](N20_first_order_audit.md) ([N20(b)](N20_first_order_audit.md)).

5. **Cost of the statistic.** $`\mu_t^P(x)`$ needs only consumed-gradient evaluations at the state $`x`$; no training tail is needed. The constants $`\Lambda_t`$ and $`M_{2,t}`$ are properties of the continuation and are not estimated by sampling messages. Node [N20](N20_first_order_audit.md) gives the estimator and states what remains to be calibrated.

## Dependencies and source

- **Uses:** [N05](N05_continuation_response.md) (telescoping identity [N05(a)](N05_continuation_response.md) and stopped composition [N05(c)](N05_continuation_response.md)), [N06](N06_sufficiency_duality.md) ([N06(a)](N06_sufficiency_duality.md)), [N10](N10_update_composition.md) (SGD expansion [N10(c)](N10_update_composition.md)), [N11](N11_adam_cold_start.md) and [N12](N12_optimizer_warmth.md) (Adam maps and warmth).
- **Used by:** [N19](N19_learner_statistics.md) (catalog of statistics), [N20](N20_first_order_audit.md) (estimation and margin), [N17](N17_ced_method.md) (method).

**Source:** [OTLR §2](../../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#2-the-expansion); [OTLR §3](../../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#3-necessity-and-minimality-of-the-first-order-statistic); [OTLR §6](../../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#6-higher-order-the-optimizer-transformed-moment-hierarchy) — Theorems C1–C3; Corollary C1′; §6. [Optimizer_Transformed_Linear_Response_EN.md](../../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md), §1–§3 and §6 (Theorems C1–C3, the scaling paragraph and the moment hierarchy).
