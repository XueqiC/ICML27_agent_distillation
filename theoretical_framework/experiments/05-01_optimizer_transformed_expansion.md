# Proposition 5.1 · Optimizer-transformed expansion

*Experiments*

[← Framework](../README.md) · [Statement](../statements/05-01_optimizer_transformed_expansion.md) · [Proof](../proofs/05-01_optimizer_transformed_expansion.md) · [Experiments](../experiments/05-01_optimizer_transformed_expansion.md) · [中文](05-01_optimizer_transformed_expansion_ZH.md)

**What is tested here.** Two checks test this proposition: exact rational computations of the expansion and of the first-order statistic count ([Check 5.1](../experiments/05-01_optimizer_transformed_expansion.md#c5-1)), and the run-level representation in a controlled network across eleven optimizer configurations ([Check 5.2](../experiments/05-01_optimizer_transformed_expansion.md#c5-2)).

**Verdict.**

- **Proposition 5.1 (optimizer-transformed expansion).**
  - The exact checks pass on 3,712 instances with residual 0. The order-$`\deg V`$ expansion equals the gap on 140/140 polynomial continuations, and every truncated remainder stays within its bound (largest ratio 0.275).
  - In the controlled network, every smooth optimizer's exact cells stay within the remainder bound (largest ratio 0.9986; this group is a diagnostic, status INFO).
  - The complete-state first-order term predicts the cold-Adam gap of `v014:C23`: 0.07568 predicted against 0.07691 measured. The raw mean-gradient term is $`-8.9\times10^{-11}`$.

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

**Status labels.** `PASS` means the implemented consistency assertions passed. `INFO` marks an empirical diagnostic: each cell gets its own PASS, FAIL or UNAVAILABLE status, and the group as a whole makes no pass/fail claim. Unless a check says otherwise, every network is the controlled small network of `v014:C23`: one-hot context, 4 tanh units, a sigmoid output, mean cross-entropy, float64. None of these networks is a real model.

**Statement under test ([Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1)).**

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

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.1](../experiments/05-01_optimizer_transformed_expansion.md#c5-1) | [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1)(a),(c) | `v015:V8` | PASS |
| [Check 5.2](../experiments/05-01_optimizer_transformed_expansion.md#c5-2) | [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1)(b),(d); [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(d) | `v015:E10` | INFO (diagnostic); every smooth exact cell within bound |

<a id="c5-1"></a>

## Check 5.1 · Exact finite checks of the expansion and the statistic count (Proposition 5.1(a),(c))

**Verdict.** The expansion is exact in rational arithmetic, and every truncated remainder respects its bound. The first-order statistic count is exact: one statistic fewer always admits a feature-matched pair that differs at first order. 3,712 instances, 4,978 assertions, worst residual 0.

**Why.**

- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1)(a) is a Taylor identity integrated against two laws. A bookkeeping error in the tensor moments, in the cancellation of $`V(z)`$, or in the remainder constant would show up as an instance whose gap differs from the truncated expansion by more than the bound.
- Part (c) claims that the projected mean increment is the exact minimal first-order statistic. A wrong count would show up in one of two ways: a feature-matched pair with a nonzero first-order term at the claimed dimension, or a statistic that is not needed.

**Setup.**

- *Polynomials.*
  - $`D=2,\dots,5`$, with 35 replicates per $`D`$; the degree cycles through $`1,\dots,4`$.
  - Coefficients lie in $`\lbrace -1,-2/3,\dots,1\rbrace`$, with a unit leading monomial.
  - Each instance has a random rational state $`z`$, six rational increments and two random laws $`P,Q`$ on them.
  - All arithmetic is exact (`Fraction`).
- *Remainder constants.* $`M_{k+1}`$ is a coefficient-sum majorant on the box that contains $`z`$ and $`z+\Delta`$. It is a rational upper bound, not a fitted constant.
- *Logistic.*
  - The continuation is $`V=\sigma(q)`$ with $`q(z)=\tfrac12z^\top Az+b^\top z`$ on $`\mathbb R^3`$, with six increments and random laws.
  - Hessian certificate: $`M_2=L^2/(6\sqrt3)+\lVert A\rVert_{\rm op}/4`$, where $`L=\max_{\rm vertices}\lVert Az+b\rVert_2`$.
  - 2,200 random instances, plus 12 restarts × 100 hill-climbing steps.
- *Ranks.*
  - 160 panels with alphabet size $`m\in\lbrace 4,\dots,9\rbrace`$ and increment dimension $`D\in\lbrace 2,\dots,5\rbrace`$.
  - Random integer increments, and random gradient directions that span $`U`$.
  - Ranks are exact (SymPy); witnesses are found by LP (HiGHS).
- Seed `SeedSequence([150150, 8])`.

**Procedure.**

1. For each polynomial, compute the exact gap and the order-$`j`$ terms $`\langle\nabla^jV(z),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle/j!`$.
   - Assert that their sum equals the gap when $`k=\deg V`$.
   - Assert that the remainder is within its bound for every $`k<\deg V`$.
2. For the logistic family, assert $`\lvert\text{gap}-\text{first order}\rvert\le\tfrac{M_2}2(m_2^P+m_2^Q)`$. Then maximize the ratio adversarially.
3. For each rank panel:
   - compute $`d_0^{(1)}=\mathrm{rank}[\mathbf 1,\mathcal G^{(1)}]-1`$ and check it against the centred rank;
   - assert $`d_0^{(1)}\le\min\lbrace\dim U,m-1\rbrace`$;
   - check that a perturbation in the null space of the retained statistics has zero $`U`$-projection and leaves every first-order term unchanged;
   - retain only $`d_0^{(1)}-1`$ statistics, solve an LP for a feature-matched pair, and assert that some first-order term differs.

**Result** (`theory/v015_checks/results_v015_independent.json`, `v015:V8`).

| Quantity | Value |
|---|---:|
| status / instances / assertions / worst residual | PASS / 3,712 / 4,978 / 0 |
| polynomials whose order-$`\deg V`$ expansion is exact | 140 / 140 |
| truncated remainders within bound | 344 / 344; largest remainder/bound 0.275 |
| logistic: largest $`\lvert\text{gap}-\text{first order}\rvert`$/bound | 0.183 (gap $`-0.0231`$, first order $`-0.2307`$, bound 1.137, $`M_2=0.823`$) |
| rank panels with $`d_0^{(1)}\le\min\lbrace\dim U,m-1\rbrace`$ | 160 / 160 |
| panels with an LP witness at $`d_0^{(1)}-1`$ statistics | 158 (the other 2 have $`d_0^{(1)}=0`$); smallest separating objective 0.113 |

**Why it supports Proposition 5.1.**

- Parts (a) and (c) are finite statements, and here they are tested in exact arithmetic.
- The remainder bound holds with room to spare for a smooth non-polynomial continuation.
- The count is minimal: removing one statistic always leaves two laws that agree on every retained statistic but differ at first order.

<a id="c5-2"></a>

## Check 5.2 · The run-level representation in a controlled network, across optimizers (Proposition 5.1(b),(d))

**Verdict.**

- Over a whole run, the gap between two feedback laws equals the first-order term of the *complete-state increment* plus a remainder within its curvature bound. This holds in every exact cell where the continuation is smooth (largest ratio 0.9986).
- In the cold-Adam cell of `v014:C23`, the increment's first-order term accounts for 98.4% of the gap. The raw mean-gradient term is $`-8.9\times10^{-11}`$.
- The remainder has the order the moment hierarchy of part (d) predicts. The exceptions are multi-round Adam, where the second-moment block makes it first order in $`\eta`$, and cells outside the smoothness hypothesis.

**Why.** In a real network the representation can fail in three ways:

- the first-order term misses the gap;
- the remainder exceeds its bound;
- the remainder does not shrink as $`\eta\to0`$.

This check measures all three. It also compares the result with the naive first-order term that uses the raw mean gradient.

**Setup.**

- *Laws.* The `v014:C23` label laws for moment $`N=2,3,4`$ and parity $`N=2,3`$, which match label moments below order $`N`$. Also the clipped-mean witness, realized on a linear-logit model.
- *Optimizers (11).*
  - SGD;
  - heavy-ball and Nesterov ($`\beta=0.9`$);
  - Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) cold ($`\beta=(0.9,0.999)`$, $`\epsilon=10^{-8}`$, bias correction);
  - Adam warm after $`W=5`$ and $`W=20`$ lr = 0 moment warm-up steps;
  - AdamW ([Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled); $`\lambda=0.01`$);
  - clipped SGD and clipped Adam;
  - signSGD ([Bernstein et al., 2018](#ref-bernstein2018signsgd));
  - Lion ([Chen et al., 2023](#ref-chen2023symbolic); $`\beta_1=0.9`$, $`\beta_2=0.99`$).
- *Clip threshold.*
  - Chosen per task to give the clipped mass closest to 50%.
  - Where gradient norms tie, the 30–70% band is not attainable; for moment $`N=3`$ and parity $`N=3`$ the mass is 25%.
- *Grid.* $`\eta\in\lbrace 10^{-3},3\cdot10^{-3},10^{-2},3\cdot10^{-2},10^{-1}\rbrace`$ and $`R\in\lbrace 1,4,40\rbrace`$, 990 cells in all.
- Seed `SeedSequence([150150, 110])`.

**Procedure.** For each cell:

1. Compute the gap $`L(P)-L(Q)`$:
   - for $`R=1,4`$, exactly, by complete history enumeration;
   - for $`R=40`$, from independent runs, with standard errors.
2. Compute the first-order term $`\sum_t\mathbb E_{d_t^Q}\langle\nabla V^P_{t+1}(Z_t),\mu^P_t-\mu^Q_t\rangle`$ over the complete state $`(\theta,m,v)`$.
   - $`\nabla V^P`$ comes from Richardson-extrapolated central differences of the exactly enumerated $`P`$-continuation.
   - The differentiation error is recorded.
3. Compute the naive term, which treats the optimizer as SGD preconditioned by the current Adam denominator and uses the raw mean gradient.
4. Estimate $`M_2`$ from sampled and hill-climbed segment curvature, never padded after a gap is observed, and evaluate the remainder bound.
5. Record $`\lvert\text{gap}-\text{first order}\rvert`$/bound and the log–log slopes in $`\eta`$.

**Result 1 — the bound in the exact cells** (`v015:E10`, `data.optimizer_summary`, `data.cells`).

| Optimizer | exact cells | largest exact ratio | exact cells above bound |
|---|---:|---:|---:|
| SGD | 60 | 0.9986 | 0 |
| heavy-ball | 60 | 0.9984 | 0 |
| Nesterov | 60 | 0.9980 | 0 |
| Adam cold | 60 | 0.9981 | 0 |
| Adam warm, $`W=5`$ | 60 | 0.3339 | 0 |
| Adam warm, $`W=20`$ | 60 | 0.8432 | 0 |
| AdamW | 60 | 0.9817 | 0 |
| clipped Adam | 55 finite (5 with nonfinite derivatives) | 0.9982 | 0 |
| signSGD | 60 | 0.9981 | 0 |
| Lion | 60 | 0.9981 | 0 |
| clipped SGD | 60 | 1.2453 | 1 (multi-step clip kink) |

**Result 2 — the `v014:C23` cold-Adam gap is a first-order effect of the increment** (`data.C23_prediction`; moment $`N=3`$, $`\eta=0.1`$, $`R=1`$).

| Quantity | Value |
|---|---:|
| measured gap $`L(P)-L(Q)`$ | 0.076907 |
| first-order term of the complete-state increment | 0.075683 |
| naive first-order term (raw mean gradient) | $`-8.87\times10^{-11}`$ |
| remainder $`\lvert\text{gap}-\text{first order}\rvert`$ | 0.001225 |
| remainder bound ($`\hat M_2=0.342`$) | 0.04447 |

- The two label laws have equal means, so every mean-gradient statistic is blind to them. The cold-Adam increment $`-\eta h_\epsilon(G)`$ is not ([Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)(a)), and this is the cold-step row (d) of [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10).
- In the same cell, signSGD and Lion give the same gap and the same first-order term (0.07691 and 0.07568), because their first step is the same sign map.
- Clipped SGD gives 0.00303 for both quantities, while its raw mean-gradient term is 0.

**Result 3 — order of the remainder** (log–log slope of $`\lvert\text{gap}-\text{first order}\rvert`$ in $`\eta`$, per task; `data.optimizer_summary[*].slopes`; cells whose remainder sits at the numerical floor have no slope).

| Optimizer | $`R=1`$ | $`R=4`$ |
|---|---|---|
| SGD, heavy-ball, Nesterov | 1.98–2.0 at $`N=2`$ and on the clipped witness; 3.0 at moment and parity $`N=3`$ | 1.95–1.99 at $`N=2`$ and on the clipped witness; 2.91–2.99 at $`N=3`$; Nesterov 3.98 at moment $`N=4`$ |
| clipped SGD, signSGD, Lion | 2.0; 3.0 at $`N=3`$ | 2.0–2.09; 2.71–3.01 at $`N=3`$ |
| Adam cold | 2.0–3.0 | 0.99 (moment $`N=3`$), 1.0 (clipped witness); 1.96–2.98 on the other tasks |
| AdamW | 2.0–3.0 | 0.99 (moment $`N=3`$), 1.31 (moment $`N=4`$), 1.0 (clipped witness); 1.98–2.98 on the other tasks |
| clipped Adam | 2.0–3.0 | 0.91 (moment $`N=3`$), 1.04 (clipped witness); 2.0–2.93 on the other tasks |
| Adam warm, $`W=5`$ | 1.99–2.98 | 1.11–1.17 (moment $`N=2,3,4`$), 1.0 (clipped witness); 2.03 and 2.92 (parity) |
| Adam warm, $`W=20`$ | 2.0–3.0 | 1.0 (moment $`N=3`$), 1.16 (moment $`N=4`$), 1.0 (clipped witness); 3.04 (moment $`N=2`$), 1.99 and 3.0 (parity) |

**Result 4 — how much of the multi-round Adam gap is first order** (moment $`N=3`$, $`R=4`$; `data.cells`).

| Optimizer | $`\eta=10^{-3}`$: gap / first order | $`\eta=10^{-1}`$: gap / first order | absolute remainder at $`\eta=10^{-3}`$ |
|---|---|---|---:|
| Adam cold | $`1.90\times10^{-3}`$ / $`1.24\times10^{-3}`$ | 0.1196 / 0.0810 | $`6.6\times10^{-4}`$ |
| Adam warm, $`W=5`$ | $`4.38\times10^{-4}`$ / $`2.22\times10^{-4}`$ | 0.0663 / 0.0365 | $`2.2\times10^{-4}`$ |
| Adam warm, $`W=20`$ | $`4.44\times10^{-5}`$ / $`2.01\times10^{-5}`$ | 0.0044 / 0.0020 | $`2.4\times10^{-5}`$ |
| clipped SGD | $`1.21\times10^{-4}`$ / $`1.21\times10^{-4}`$ | 0.0115 / 0.0115 | $`2.2\times10^{-11}`$ |

**Result 5 — the boundary of the smoothness hypothesis.**

| Where | What happens |
|---|---|
| clipped SGD, clipped witness, $`\eta=0.1`$, $`R=4`$ | The only exact cell above its bound: ratio 1.245 (gap 0.04476, first order 0.04637, bound 0.00129). Later clipping decisions make the continuation nonsmooth, so the smoothness hypothesis of Proposition 5.1 is not met. |
| $`R=40`$ (330 Monte Carlo cells) | 25 cells are above the point bound before standard errors are taken into account. Two remain above it after four standard errors, and both are sign-type: signSGD on moment $`N=4`$ at $`\eta=0.01`$, and Lion on the clipped witness at $`\eta=10^{-3}`$. |
| clipped Adam, moment $`N=4`$ | 10 cells have nonfinite central differences at cold zero-gradient coordinates and are recorded as unavailable. |

![Figure 5.1: first-order representation across optimizers](../figures/fig5_1_linear_response.png)

*Figure 5.1.*
- (A) The remainder of the first-order representation shrinks as $`\eta^2`$ for SGD and sign optimizers, and as $`\eta^3`$ when two increment moments match. For multi-round Adam it shrinks only as $`\eta`$.
- (B) In the cold-Adam cell, the first-order term of the optimizer increment accounts for the gap, while the raw mean gradient is blind to it.
- (C) In every exact cell where the smoothness hypothesis holds, the gap minus the first-order term stays within the remainder bound. The single exceedance is multi-step clipped SGD.

[PDF](../figures/fig5_1_linear_response.pdf)

**Why it supports Proposition 5.1.**

- Where the continuation is smooth, the gap is the first-order term of the optimizer increment plus a remainder within the stated bound.
- The increment is the right object. For laws with equal means, the raw mean gradient predicts nothing, while the increment predicts 98% of the cold-Adam gap.
- The remainder orders follow the moment hierarchy of part (d):
  - $`O(\eta^2)`$ for SGD, momentum, clipping and sign optimizers;
  - $`O(\eta^3)`$ when two increment moments agree.
- For Adam-type optimizers over several rounds, the remainder is first order in $`\eta`$. This is the second-moment channel of [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1). The first-order term then carries 45–68% of the gap on the moment task. From the cold state to $`W=20`$ warm-up steps, the absolute remainder falls 27-fold.
  - This is why a first-order ranking needs the margin condition of [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9). Where the second-moment channel dominates, finite measurement is needed instead ([Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8)).
- Every cell above its bound lies outside the smoothness hypothesis: multi-step clipped SGD, and Monte Carlo cells of sign optimizers.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)). The linear programs of `v015:V2` and `v015:V8` are solved by `scipy.optimize.linprog` with the HiGHS solver ([Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)).

## Artifacts

- `theory/v015_checks/results_v015_independent.json`:
  - `v015:V8`, and `v015:E10` (per-cell records in `data.cells`; summaries in `data.optimizer_summary`; the cold-Adam cell in `data.C23_prediction`; the design in `data.design`);
  - `v015:V2`, `v015:V3`, `v015:V4`, `v015:E0`, `v015:E1`, `v015:E2`, `v015:E3`;
  - `v015:E8`: the frozen-denominator, warm-affine, cold-separation and warmth-trajectory parts (the last with `per_step` and `fits`), and the 210 `data.gaps` cells;
  - `v015:E11` (`data.clipped_mean`, `data.always_active`, `data.warm_Adam`, `data.Lion`, `data.LoRA`).
- `theory/v015_checks/c23_bundle.json`: exact laws, network, optimizer code and per-history raw steps.
- `theory/v015_checks/raw/E8_trajectories.npz`: per-replica warmth trajectories. It is gitignored and regenerated by the first command; its SHA-256 is recorded as `raw_sha256` in the warmth-trajectory part of `v015:E8`.
- `theory/v016_checks/results_v016.json`: claims `v016:C1.exact`, `v016:C1.moments` and `v016:C1.accumulation`. The full arrays are in `theory/v016_checks/results_v016_raw.json.gz`, which is gitignored; its SHA-256 is in `raw_artifact`.
- `theory/independent_checks/results_independent_v014.json`: `controlled_neural`.
- `docs/direction/v015_revision/results_v015.json`: package checks `pkg:T03`–`pkg:T08`.
- Code:
  - `theory/v015_checks/otlr.py` (`check_V8`, `check_E10`, `check_E11`);
  - `theory/v015_checks/warmth.py` (`check_E8`);
  - `theory/v016_checks/check_v016.py` (`v016:C1`);
  - specifications in `theory/v015_checks/README.md` and `theory/v016_checks/SPEC.md`.
- Figures: `../figures/fig5_1_linear_response`, `fig5_2_cold_adam`, `fig5_3_warmth` and `fig5_4_learner_statistics`, each as `.pdf` and `.png`. The specs are in `../figures/specs/fig5_1.md` to `fig5_4.md`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-bernstein2018signsgd"></a>

**Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, Animashree Anandkumar (2018).** signSGD: Compressed optimisation for non-convex problems. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:560–569. [proceedings.mlr.press/v80/bernstein18a](https://proceedings.mlr.press/v80/bernstein18a.html).

<a id="ref-chen2023symbolic"></a>

**Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le (2023).** Symbolic discovery of optimization algorithms. *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 49205–49233. arXiv [2302.06675](https://arxiv.org/abs/2302.06675).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
