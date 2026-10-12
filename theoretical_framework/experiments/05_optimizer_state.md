# §5 Optimizer state and the first-order representation — Experiments

[← Framework](../README.md) · [Proofs](../proofs/05_optimizer_state.md) · [中文](05_optimizer_state_ZH.md)

This page retains Check 5.1 through Check 5.7 and the verdict table for all §5 checks; Check 5.8 through Check 5.18 continue in [Warm Adam and learner statistics](05_warm_adam_and_learners.md).

**Verdict.**

- **Proposition 5.1 (optimizer-transformed expansion).**
  - The exact checks pass on 3,712 instances with residual 0. The order-$`\deg V`$ expansion equals the gap on 140/140 polynomial continuations, and every truncated remainder stays within its bound (largest ratio 0.275).
  - In the controlled network, every smooth optimizer's exact cells stay within the remainder bound (largest ratio 0.9986; this group is a diagnostic, status INFO).
  - The complete-state first-order term predicts the cold-Adam gap of `v014:C23`: 0.07568 predicted against 0.07691 measured. The raw mean-gradient term is $`-8.9\times10^{-11}`$.
- **Proposition 5.3 (cold-start Adam).** All five parts are confirmed exactly.
  - The quotient dimensions are $`s-1`$ and $`m-1`$ on 300/300 alphabets.
  - The margin bound comes within 0.994 of being attained.
  - The optimizer-memory witness has opposite second steps for every $`\epsilon`$ tested and in every $`(\beta_1,\beta_2)`$ cell with $`\beta_1>0`$.
  - In the controlled network, the Adam gap between laws with equal gradient moments survives a frozen Jacobian and vanishes when sign disagreement is removed.
- **Proposition 5.5(b) (frozen-denominator error).** No violation on 6,944 adversarially searched states. The best ratio is 1.0, so the bound is tight.
- **Proposition 5.6 (warm Adam is first-order affine).** The bound holds on all 2,496 law pairs, with constants fixed in advance. The largest gap/bound is 0.988.
- **Proposition 5.7 (cold steps separate equal-mean feedback).** The cold gap equals $`\lvert w\rvert\eta/2`$ exactly. At warm states, the normalized gap scales like $`\kappa^{0.9995}`$.
- **Proposition 5.8 (evolution of warmth).**
  - The exact recursion matches the closed form to $`7.1\times10^{-13}`$ for $`s`$ up to $`10^6`$.
  - Both bounds hold at every age, and the upper bound is within a factor 1.33 of $`\bar\kappa(s)`$. The plateau $`(1-\beta_2)/\beta_2`$ is reached and does not vanish.
  - Under stationary Gaussian, heavy-tailed and sparse gradients, 75/75 Monte Carlo intervals contain $`\sigma^2(1-\beta_2^{\,s})`$. The accumulation bound holds on 9,000 grid points.
  - Real trajectories cover only the early phase. There the median warmth falls with log–log slope $`-1.06`$, and lr = 0 warm-up and real steps count alike.
- **Proposition 5.9 and Theorem 5.10(e) (clipping).**
  - Equal raw means with different clipped means give a first-order gap. Its remainder has slope 2.0015.
  - With clipping always active ($`p_{\rm act}=1`$), equal mean directions give a first-order term of exactly 0.
  - The partial-activation bound of Proposition 5.9 is a two-line inequality and has no separate numerical check.
- **Theorem 5.10 (the first-order statistic by learner).** The five catalog witnesses pass with 105 assertions and worst residual $`8.0\times10^{-21}`$.
  - Warm Adam: at equal gradient means, the first-order term is carried entirely by the $`v`$ block, as predicted to $`10^{-12}`$.
  - Lion: under momentum dominance, the parameter block is exactly blind to the message.
  - LoRA: at $`B=0`$, null-space differences are second order and range differences are first order.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.1](#c5-1) | [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1)(a),(c) | `v015:V8` | PASS |
| [Check 5.2](#c5-2) | [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1)(b),(d); [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10)(d) | `v015:E10` | INFO (diagnostic); every smooth exact cell within bound |
| [Check 5.3](#c5-3) | [Proposition 5.3](../proofs/05_optimizer_state.md#s5-3)(a)–(c) | `v015:V2`, `pkg:T03`–`pkg:T06` | PASS |
| [Check 5.4](#c5-4) | [Proposition 5.3](../proofs/05_optimizer_state.md#s5-3)(d) | `v015:V3`, `pkg:T07` | PASS |
| [Check 5.5](#c5-5) | [Proposition 5.3](../proofs/05_optimizer_state.md#s5-3)(e) | `v015:V4`, `v015:E3`, `pkg:T08` | PASS |
| [Check 5.6](#c5-6) | [Proposition 5.3](../proofs/05_optimizer_state.md#s5-3)(a)–(c) in a network | `v015:E0`, `v015:E1` | PASS |
| [Check 5.7](#c5-7) | [Proposition 5.3](../proofs/05_optimizer_state.md#s5-3) versus feature learning | `v015:E2`, `v014:C23` (Adam settings) | INFO; gap persists in 39/45 frozen-Jacobian cells |
| [Check 5.8](05_warm_adam_and_learners.md#c5-8) | [Proposition 5.5](../proofs/05_warm_adam.md#s5-5)(b) | `v015:E8` (frozen-denominator part) | PASS |
| [Check 5.9](05_warm_adam_and_learners.md#c5-9) | [Proposition 5.6](../proofs/05_warm_adam.md#s5-6), [Remark 5.11](../proofs/05_learner_statistics.md#s5-11) | `v015:E8` (warm-affine part) | PASS |
| [Check 5.10](05_warm_adam_and_learners.md#c5-10) | [Proposition 5.7](../proofs/05_warm_adam.md#s5-7) | `v015:E8` (cold-separation part) | PASS |
| [Check 5.11](05_warm_adam_and_learners.md#c5-11) | [Proposition 5.8](../proofs/05_warm_adam.md#s5-8)(a)–(c) | `v016:C1.exact`, `v016:C1.moments`, `v016:C1.accumulation` | PASS |
| [Check 5.12](05_warm_adam_and_learners.md#c5-12) | [Proposition 5.8](../proofs/05_warm_adam.md#s5-8)(a) early phase and (d) warm-up, on real trajectories | `v015:E8` (warmth-trajectory part) | PASS |
| [Check 5.13](05_warm_adam_and_learners.md#c5-13) | [Propositions 5.6](../proofs/05_warm_adam.md#s5-6)–[5.8](../proofs/05_warm_adam.md#s5-8) over a training run | `v015:E8` (`data.gaps`) | PASS |
| [Check 5.14](05_warm_adam_and_learners.md#c5-14) | [Proposition 5.9](../proofs/05_learner_statistics.md#s5-9), [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10)(e) | `v015:E11` (clipped mean) | PASS |
| [Check 5.15](05_warm_adam_and_learners.md#c5-15) | [Proposition 5.9](../proofs/05_learner_statistics.md#s5-9) at $`p_{\rm act}=1`$, [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10)(e) | `v015:E11` (always active) | PASS |
| [Check 5.16](05_warm_adam_and_learners.md#c5-16) | [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10)(c), [Proposition 5.5](../proofs/05_warm_adam.md#s5-5)(a) | `v015:E11` (warm Adam) | PASS |
| [Check 5.17](05_warm_adam_and_learners.md#c5-17) | [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10)(f) | `v015:E11` (Lion) | PASS |
| [Check 5.18](05_warm_adam_and_learners.md#c5-18) | [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10)(g) | `v015:E11` (LoRA) | PASS |

**Status labels.** `PASS` means the implemented consistency assertions passed. `INFO` marks an empirical diagnostic: each cell gets its own PASS, FAIL or UNAVAILABLE status, and the group as a whole makes no pass/fail claim. Unless a check says otherwise, every network is the controlled small network of `v014:C23`: one-hot context, 4 tanh units, a sigmoid output, mean cross-entropy, float64. None of these networks is a real model.

<a id="c5-1"></a>

## Check 5.1 · Exact finite checks of the expansion and the statistic count (Proposition 5.1(a),(c))

**Verdict.** The expansion is exact in rational arithmetic, and every truncated remainder respects its bound. The first-order statistic count is exact: one statistic fewer always admits a feature-matched pair that differs at first order. 3,712 instances, 4,978 assertions, worst residual 0.

**Why.**

- [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1)(a) is a Taylor identity integrated against two laws. A bookkeeping error in the tensor moments, in the cancellation of $`V(z)`$, or in the remainder constant would show up as an instance whose gap differs from the truncated expansion by more than the bound.
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
  - Adam cold ($`\beta=(0.9,0.999)`$, $`\epsilon=10^{-8}`$, bias correction);
  - Adam warm after $`W=5`$ and $`W=20`$ lr = 0 moment warm-up steps;
  - AdamW ($`\lambda=0.01`$);
  - clipped SGD and clipped Adam;
  - signSGD;
  - Lion ($`\beta_1=0.9`$, $`\beta_2=0.99`$).
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

- The two label laws have equal means, so every mean-gradient statistic is blind to them. The cold-Adam increment $`-\eta h_\epsilon(G)`$ is not ([Proposition 5.3](../proofs/05_optimizer_state.md#s5-3)(a)), and this is the cold-step row (d) of [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10).
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
- For Adam-type optimizers over several rounds, the remainder is first order in $`\eta`$. This is the second-moment channel of [Theorem 7.1](../proofs/07_second_moment_channel.md#s7-1). The first-order term then carries 45–68% of the gap on the moment task. From the cold state to $`W=20`$ warm-up steps, the absolute remainder falls 27-fold.
  - This is why a first-order ranking needs the margin condition of [Corollary 8.9](../proofs/08_certificate.md#s8-9). Where the second-moment channel dominates, finite measurement is needed instead ([Theorem 8.8](../proofs/08_certificate.md#s8-8)).
- Every cell above its bound lies outside the smoothness hypothesis: multi-step clipped SGD, and Monte Carlo cells of sign optimizers.

<a id="c5-3"></a>

## Check 5.3 · Cold map, exact quotient dimensions, inverse conditioning (Proposition 5.3(a)–(c))

**Verdict.** On 300 random finite gradient alphabets:

- the cold Adam step equals $`c-\eta h_\epsilon(G)`$ to $`4.4\times10^{-16}`$;
- exactly $`s-1`$ statistics are necessary and sufficient at $`\epsilon=0`$, and exactly $`m-1`$ at $`\epsilon>0`$;
- the derivative of the inverse map reaches $`10^{22}`$.

**Why.** [Proposition 5.3](../proofs/05_optimizer_state.md#s5-3)(a)–(c) makes three claims, and each is a finite statement that can be checked exactly:

- an exact formula for the first step;
- the immediate-parameter information is the joint sign law (dimension $`s-1`$) at $`\epsilon=0`$, and the full law (dimension $`m-1`$) at $`\epsilon>0`$;
- coordinate sign marginals are not sufficient.

**Setup.**

- 300 random finite gradient alphabets with $`p=1,\dots,4`$ coordinates, including zero coordinates.
- An independent recurrence for part (a), checked in float64 and, where the square root is exact, in exact rationals.
- Exact ranks and null spaces. When fewer than $`s-1`$ features are used, HiGHS LPs construct separating laws.
- Inverse round-trips on a $`u\to1`$ grid with $`\epsilon=1/100`$.

**Procedure.**

- Compare the first step with $`c-\eta h_\epsilon(G)`$.
- Count the attainable sign patterns $`s`$, and verify that $`s-1`$ pattern indicators suffice.
- For any smaller feature set, exhibit a zero-mass perturbation and the two laws it separates.
- Repeat with $`\epsilon>0`$, which gives dimension $`m-1`$.
- Evaluate the joint-versus-marginal example.

**Result** (`v015:V2`).

| Quantity | Value |
|---|---|
| status, instances, assertions, worst residual | PASS, 300, 3,288, $`4.44\times10^{-16}`$ |
| quotient dimension $`=s-1`$ at $`\epsilon=0`$ | 300/300 alphabets (1 to 14 atoms; $`s`$ from 1 to 14) |
| dimension $`=m-1`$ at $`\epsilon>0`$ | 300/300 |
| example separation (`quotients[0]`) | 4 atoms, 4 patterns, 2 features: $`P=(\tfrac14,\tfrac18,\tfrac38,\tfrac14)`$ and $`Q=(\tfrac14,\tfrac38,\tfrac18,\tfrac14)`$ match both features, but their pattern indicators differ by $`-\tfrac14`$ |
| inverse derivative, $`\epsilon=1/100`$ | $`u=0.9`$: 1; $`u=0.999`$: $`10^4`$; $`u=1-10^{-12}`$: $`10^{22}`$ |
| joint versus marginal | identical marginals; same-sign utility 1 against 0 |

Independent package implementation (`docs/direction/v015_revision/results_v015.json`, `checks`), all PASS:

- `pkg:T03`: 480 cases; update error $`\le2.22\times10^{-16}`$.
- `pkg:T04`: 25 atoms and 9 sign patterns; statistic dimensions 8 and 24.
- `pkg:T05`: equal mean updates; same-sign utility gap 1.0.
- `pkg:T06`: label laws matching orders 0–2. The SGD mean parameter gap is 0, the ideal-sign Adam gap is $`1/10`$, and the finite-$`\epsilon`$ gap is $`250/2703`$.

**Why it supports Proposition 5.3(a)–(c).**

- At $`\epsilon=0`$, the exact information content of a cold Adam step is the joint sign quotient; at $`\epsilon>0`$ it is the full gradient law. The counts are exactly $`s-1`$ and $`m-1`$.
- The inverse at $`\epsilon>0`$ exists but is badly conditioned. In practice, then, the magnitude information is nearly invisible.
- `pkg:T06` shows the same thing at the level of means: two laws that SGD cannot tell apart at first order are told apart by Adam.

<a id="c5-4"></a>

## Check 5.4 · Margin-sensitive approximation (Proposition 5.3(d))

**Verdict.** The margin bound holds on 1,660 exact instances. Adversarial search reaches a gap/bound ratio of 0.994, so the bound is nearly tight.

**Why.** Part (d) says when the sign quotient is a good *approximate* description for regular utilities. It connects the exact statement, that any $`\epsilon>0`$ identifies magnitudes, with practice.

**Setup.**

- Random finite gradient laws $`P,Q`$.
- Coordinatewise-Lipschitz utilities $`F\in[0,1]`$, built as clipped sums of piecewise-linear coordinate functions with known $`L_i`$.
- Random margins $`a`$ and random $`\epsilon`$.
- Exact rational arithmetic throughout.
- Adversarial restarts with hill-climbing over $`(G,P,Q,L,a,\epsilon,\eta)`$.

**Procedure.** On every instance, assert the margin bound and $`\Delta_{\rm sign}\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$. Then maximize gap/bound.

**Result** (`v015:V3`).

| Quantity | Value |
|---|---|
| status, instances, assertions, worst residual | PASS, 1,660, 3,320, 0 |
| largest gap/bound, adversarial | 0.9941 |
| largest gap/bound, random | 0.837 |
| largest $`\Delta_{\rm sign}/\mathrm{TV}`$ found | 0.274 |

`pkg:T07`: 1,500 instances, maximum gap/bound 0.739, no violation.

**Why it supports Proposition 5.3(d).** The bound is valid and nearly attained. For regular utilities the sign quotient is accurate, up to two terms: the gradient mass near zero and the $`\epsilon/(a+\epsilon)`$ terms.

<a id="c5-5"></a>

## Check 5.5 · Magnitude-bearing optimizer memory (Proposition 5.3(e))

**Verdict.** The witness is exact:

- $`\widehat m_2=\mp\tfrac16`$ and $`\widehat v_2=\tfrac{17}{24},\tfrac{41}{24}`$, so the second steps point in opposite directions.
- It persists for every $`\epsilon`$ from 0 to $`10^{-1}`$.
- It exists in every $`(\beta_1,\beta_2)`$ grid cell with $`\beta_1>0`$, and in none with $`\beta_1=0`$.

The same pair runs through the network optimizer, and a network witness exists.

**Why.** Part (e) claims that matching the first-step parameter information is not enough for a learning continuation, because $`m_1`$ keeps magnitudes. For this to count, the counterexample must be exact, realizable by real losses, robust to $`\epsilon`$, and present in the actual optimizer code.

**Setup.**

- *Exact (`v015:V4`).*
  - Fraction and symbolic arithmetic for the gradient histories $`(1,-\tfrac34)`$ and $`(2,-\tfrac34)`$, with $`\beta_1=\beta_2=\tfrac12`$.
  - A squared-loss realization.
  - $`\epsilon\in\lbrace 0,10^{-1},\dots,10^{-12}\rbrace`$.
  - A $`5\times5`$ grid $`\beta_1,\beta_2\in\lbrace 0,\tfrac1{10},\tfrac12,\tfrac9{10},\dots\rbrace`$, with 300 search trials per cell.
- *Network (`v015:E3`).*
  - The same pair through the scalar optimizer of Checks 5.6–5.7, at $`\epsilon=0`$ and $`\epsilon=10^{-8}`$.
  - A soft-label search for same-sign histories with opposite second updates.

**Procedure.**

- Compute every moment and both second steps, and check that the two steps have opposite signs.
- In the network, search for label pairs $`(y_1,y_2)`$ and $`(y_1',y_2)`$ whose first steps coincide but whose second updates are opposite on some coordinate.

**Result** (`v015:V4`, `v015:E3`).

| Quantity | Value |
|---|---|
| `v015:V4` status, instances | PASS, 39 |
| exact moments | $`m_2=-\tfrac18,+\tfrac18`$; $`\widehat m_2=-\tfrac16,+\tfrac16`$; $`\widehat v_2=\tfrac{17}{24},\tfrac{41}{24}`$ |
| second steps at $`\eta=1`$ | $`+\sqrt{102}/51=+0.198`$ and $`-\sqrt{246}/123=-0.128`$ |
| squared-loss realization | first targets $`-1`$ and $`-2`$; common second target $`-\tfrac14`$ |
| opposite directions at positive $`\epsilon`$ | all 14 values from $`0`$ to $`10^{-1}`$ |
| $`\beta`$-grid cells with a witness | 20/25: every cell with $`\beta_1>0`$, no cell with $`\beta_1=0`$ |
| `v015:E3` scalar trace, $`\epsilon=0`$ and $`10^{-8}`$ | second updates $`+0.19803`$ and $`-0.12752`$ |
| `v015:E3` network witnesses | found at $`\epsilon=0`$ and $`10^{-8}`$: labels $`(0,0.7)`$ and $`(0.1,0.7)`$ give identical first steps and opposite second steps on the output bias |

`pkg:T08`: equal first outputs and equal sign histories. The second increments are $`+0.019803`$ and $`-0.012752`$ at $`\eta=0.1`$, and the threshold-task gap is 1.

**Why it supports Proposition 5.3(e).**

- The counterexample is exact, realizable, robust to $`\epsilon`$, and present in a real network.
- The $`\beta`$ grid isolates the mechanism. The first moment $`m`$ carries the magnitude memory, so the effect exists exactly when $`\beta_1>0`$.

<a id="c5-6"></a>

## Check 5.6 · In a network, the Adam gap is the cold-start sign quotient (Proposition 5.3(a)–(c))

**Verdict.**

- Two label laws agree on all moments through order 2 and have identical gradient moments through order 2. Yet their expected first Adam updates differ by $`\eta`$ per coordinate, and Adam's return gap at $`\eta=0.1`$ is 0.0769 ($`R=1`$) and 0.1196 ($`R=4`$).
- Removing sign disagreement removes the gap entirely. Five warm moment steps reduce it to 0.0034 and 0.0098.

**Why.** In the controlled network, Adam separates label laws that agree on all moments through order $`N-1`$, and SGD does not. [Proposition 5.3](../proofs/05_optimizer_state.md#s5-3)(a)–(c) predicts the mechanism: cold Adam responds to the sign law of the gradient $`G=p-Z`$ around the current prediction, not to moments.

- `v015:E0` fixes an exact reproduction of the gap.
- `v015:E1` traces the chain from input moments to gradient moments, then to the normalized update, then to the return. It then ablates one factor at a time.

**Setup.**

- Adam with $`\beta=(0.9,0.999)`$, $`\epsilon=10^{-8}`$ outside the square root, and bias correction on.
- Laws: `law_scalar(N)` matches the label moments of orders $`1,\dots,N-1`$ only.
- Gap: $`\mathbb E_P[p_R^N]-\mathbb E_Q[p_R^N]`$, by complete enumeration of finite histories (standard error 0).
- Ablations (moment $`N=3`$, $`\eta=0.1`$):
  - $`\epsilon`$ inside the square root;
  - $`\epsilon\in\lbrace 10^{-6},10^{-4},10^{-2},10^{-1}\rbrace`$;
  - no bias correction;
  - five warm moment steps on a third law;
  - global clip 0.1;
  - sum reduction;
  - a support shifted so that all first gradients share one sign.

**Procedure.**

- Enumerate the per-support gradients, gradient moments, finite-$`\epsilon`$ updates and their expectations.
- Compare them with the executed network step at relative tolerance $`10^{-12}`$.
- Rerun each ablation from the same snapshot.

**Result** (`v015:E0`, `v015:E1`, `theory/v015_checks/c23_bundle.json`).

| Reproduction (moment $`N=3`$) | $`\eta=0.01`$ | $`\eta=0.1`$ |
|---|---|---|
| Adam gap, $`R=1`$ | 0.00757 | 0.07691 |
| Adam gap, $`R=4`$ | 0.01892 | 0.11961 |
| affine-SGD comparator, $`R=1`$ / $`R=4`$ | $`-6.3\times10^{-9}`$ / $`-2.5\times10^{-8}`$ | $`-6.3\times10^{-6}`$ / $`-2.1\times10^{-5}`$ |

The chain for moment $`N=3`$ at $`\eta=0.1`$ (`data.transitions`):

- Input moment differences of orders 1, 2, 3: $`0,0,-0.0556`$.
- Joint gradient moments: equal through order 2 (to $`3\times10^{-17}`$); order 3 differs by 0.056.
- Expected first update: $`+0.05`$ against $`-0.05`$ per coordinate, a difference of $`\eta`$.
- Return difference: 0.0769.
- Under $`P`$ the label is $`0`$ with probability $`\tfrac14`$ and $`\tfrac23`$ with probability $`\tfrac34`$, so the output-bias gradient $`p-Z`$ is positive with probability $`\tfrac14`$.
- Under $`Q`$ the label is $`\tfrac13`$ with probability $`\tfrac34`$ and $`1`$ with probability $`\tfrac14`$, so the same gradient is positive with probability $`\tfrac34`$.
- The measured map error is $`2.1\times10^{-15}`$ relative.

| Ablation (moment $`N=3`$, $`\eta=0.1`$) | gap $`R=1`$ | gap $`R=4`$ |
|---|---|---|
| baseline | 0.0769 | 0.1196 |
| $`\epsilon`$ inside sqrt; $`\epsilon=10^{-6}`$; sum reduction | 0.0769 | 0.1196 |
| $`\epsilon=10^{-4}`$ | 0.0767 | 0.1197 |
| $`\epsilon=10^{-2}`$ | 0.0610 | 0.1148 |
| $`\epsilon=10^{-1}`$ | 0.0211 | 0.0446 |
| no bias correction | 0.2477 | $`-0.0081`$ |
| five warm moment steps | **0.0034** | **0.0098** |
| global clip 0.1 | 0.0769 | 0.1840 |
| shifted support (one first-gradient sign) | **0.0000** | **0.0000** |

The shifted support keeps the matched label moments: orders 1 and 2 are still equal.

**Why it supports Proposition 5.3.**

- Two laws with identical gradient moments through order 2 give first Adam updates that differ by $`\eta`$ per coordinate. The cause is exactly the sign quotient: the probability that the label lies above the current prediction is $`1/4`$ under one law and $`3/4`$ under the other.
- Removing sign disagreement removes the gap. Warming the moments removes most of it, as [Proposition 5.7](../proofs/05_warm_adam.md#s5-7) and [Proposition 5.6](../proofs/05_warm_adam.md#s5-6) predict.
- The gap is insensitive to the $`\epsilon`$ convention until $`\epsilon`$ becomes comparable to the gradient magnitudes ($`10^{-2}`$), as part (d) predicts.

<a id="c5-7"></a>

## Check 5.7 · Optimizer effect, not feature learning (Proposition 5.3)

**Verdict.** An Adam that cannot learn features, because its Jacobian is frozen, reproduces the gap almost exactly: 0.0757 against 0.0769 at $`R=1`$, and 0.1225 against 0.1196 at $`R=4`$. The frozen-Jacobian gap persists in 39 of 45 cells. Over the declared Adam settings, the tangent affine-SGD comparator misses the gap in 32 of 42 configurations.

**Why.** A gap against an affine-SGD comparator could come from the optimizer, from nonlinear features, or from both. Freezing the network Jacobian, while keeping the sigmoid, the cross-entropy and the same Adam recurrence, isolates the optimizer.

**Setup** (`v015:E2`).

- Five branches from one snapshot, with identical within-source histories:
  - the actual network with Adam;
  - a frozen Jacobian, $`\mathrm{logit}(\theta)=\mathrm{logit}(\theta_0)+J_0(\theta-\theta_0)`$, with Adam and its own moments;
  - the actual network with SGD;
  - the frozen Jacobian with SGD;
  - tangent affine SGD.
- Grid: 5 tasks × $`\eta\in\lbrace 0.01,0.1,1\rbrace`$ × $`R\in\lbrace 1,4,40\rbrace`$ = 45 cells.
- The derivative checks of $`J_0`$ have residuals $`\le2.6\times10^{-11}`$.
- A frozen-Adam gap *persists* if its magnitude exceeds $`\max(10^{-10},3\,\mathrm{SE})`$.

**Result** (`v015:E2`, `data.comparisons`; status INFO).

| Cell | actual Adam | frozen-Jacobian Adam | actual SGD | tangent SGD |
|---|---|---|---|---|
| moment $`N=3`$, $`\eta=0.1`$, $`R=1`$ | 0.0769 | 0.0757 | 0.0000 | 0.0000 |
| moment $`N=3`$, $`\eta=0.1`$, $`R=4`$ | 0.1196 | 0.1225 | 0.0000 | 0.0000 |
| moment $`N=3`$, $`\eta=0.01`$, $`R=4`$ | 0.0189 | 0.0188 | 0.0000 | 0.0000 |
| parity $`N=2`$, $`\eta=0.1`$, $`R=4`$ | 0.0762 | 0.0774 | 0.0034 | 0.0034 |
| moment $`N=4`$, $`\eta=0.1`$, $`R=4`$ | $`-0.0218`$ | $`-0.0223`$ | $`-0.0000`$ | 0.0000 |

- The frozen-Jacobian Adam gap persists in 39 of 45 cells.
- Across all cells with $`\eta\le0.1`$ and $`R\le4`$, the three SGD branches agree with one another to within $`2.8\times10^{-4}`$, and none exceeds 0.0034.

The affine comparator under Adam, from `theory/independent_checks/results_independent_v014.json`, `controlled_neural.details.observed_affine_comparison_cohorts`. The settings are $`\beta=(0.9,0.999)`$, $`\epsilon=10^{-8}`$, bias correction, $`\eta\in\lbrace 0.01,0.1\rbrace`$, $`R\in\lbrace 1,4,40\rbrace`$, over 7 task laws:

| Cohort | Configurations | Agreements | Departures | Largest deviation |
|---|---:|---:|---:|---|
| Adam, all declared settings | 42 | 10 | 32 | 0.1196 (moment $`N=3`$, $`\eta=0.1`$, $`R=4`$) |

![Figure 5.2: cold-start Adam](../figures/fig5_2_cold_adam.png)

*Figure 5.2.*
- (A) Cold Adam separates laws whose first two gradient moments agree. A frozen-Jacobian Adam reproduces the gap, while SGD shows none.
- (B) Removing sign disagreement (shifted support) removes the gap, and warm moments remove most of it.
- (C) Two histories with identical sign sequences take opposite second steps, because $`m`$ retains magnitudes.

[PDF](../figures/fig5_2_cold_adam.pdf)

**Why it supports Proposition 5.3.**

- The gap is an optimizer-state effect, as the proposition states.
- In this network, moment matching ([Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)) controls SGD but not cold Adam. The part that the affine-SGD prediction misses is the cold-start quotient of parts (a)–(c) together with the momentum memory of part (e).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

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
