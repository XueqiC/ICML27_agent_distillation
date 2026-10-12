# Proposition 5.1 · Optimizer-transformed expansion

*Statement*

[← Framework](../README.md) · [Statement](../statements/05-01_optimizer_transformed_expansion.md) · [Proof](../proofs/05-01_optimizer_transformed_expansion.md) · [Experiments](../experiments/05-01_optimizer_transformed_expansion.md) · [中文](05-01_optimizer_transformed_expansion_ZH.md)

**In one paragraph.** This proposition expands the effect of replacing one feedback law by another in the moments of the optimizer's increment of the *complete* learner state, parameters and optimizer state together. It is the tool from which the later learner-specific results are computed: to first order, what a feedback law must carry is the mean increment of the complete state, and only in the directions to which the continuation value is sensitive. The calculation is of the same kind as stochastic modified equations, TracIn and metagradients ([Remark 5.2](05-02_relation_to_prior_work.md#s5-2)); no novelty is claimed for it.

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

**What it says.** Write any stateful first-order optimizer, clipping included, as a map from a consumed gradient to an increment of the complete state (parameters, momentum, second moments, counter). The effect of two feedback laws on the training outcome is, to first order, the sum over rounds of the inner product between the continuation-value gradient and the difference of mean state increments; the remainder is controlled by second moments of the increment. Matching the mean increment in the directions of the continuation-value gradients is sufficient to first order, necessary where the first-order term exceeds the remainder, and the number of statistics is the dimension of the linearized response class.

**Intuition.** Feedback affects the student only through the change of state it causes. To first order only the mean of that change matters, and only in the directions the continuation value cares about.

**Role in the framework.** A tool, of the same kind as stochastic modified equations, TracIn and metagradients (Remark 5.2). Every later statement about what a learner makes the feedback carry, and Theorems A–C, are computed from it.

**Derivation outline.**

- Taylor-expand $`V(z+\Delta)`$ along the segment from $`z`$ to $`z+\Delta`$ with a Lagrange remainder, and integrate under each law; constants cancel and expectations pass through the multilinear terms.
- For the run, insert this expansion into each term of the telescoping identity of Theorem 3.2.
- Sufficiency and necessity follow from $`\nabla V(z)\in U`$; minimality is the sufficiency duality of Theorem 4.2 applied to the linearized response class.

**Uses.**

- [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2) — Continuation-response characterization
- [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2) — Sufficiency duality and transfer

**Used by.**

- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10) — The first-order statistic by learner
- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2) — Learner-relative functional and information norm
- [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1) — State-channel decomposition
- [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4) — Measuring the mean increment at the actual states
- [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9) — The first-order version, its margin criterion, and its cap
