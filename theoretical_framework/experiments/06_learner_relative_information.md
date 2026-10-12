# §6 Learner-relative information bound (Theorem A) — Experiments

[← Framework](../README.md) · [Proofs](../proofs/06_learner_relative_information.md) · [中文](06_learner_relative_information_ZH.md)

**Verdict.** All six checks pass. They are exact convex programs and Monte Carlo simulations of synthetic Gaussian and tabular models; they test the mathematics of Theorem A, not a real distillation run.

- **Design duality ([Lemma 6.4](../proofs/06_learner_relative_information.md#s6-4)).** On 216 instances the c-optimal design value and the dual information norm agree to a relative gap of at most $`4.5\times10^{-13}`$, including the singular and non-identifiable cases.
- **Lower bound and achievability ([Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3), [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5)).** On 192 identifiable instances the plug-in risk under the integer-rounded c-optimal design lies between $`0.789\,\mathfrak n`$ and $`0.850\,\mathfrak n`$ (median $`0.798\approx\sqrt{2/\pi}`$), inside the predicted band $`[\mathfrak n/4,\mathfrak n]`$.
- **Aliasing ([Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7)).** Along the null space of the demonstrations the data laws coincide exactly for every $`K`$ up to $`10^6`$, and the worst error never falls below half the functional gap.
- **Learner weights ([Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11)).** Measured sensitivity slopes in $`d`$: SGD $`1.00`$, frequent-regime Adam $`0.489`$ (weight slope $`0.98`$), isolated-regime Adam $`0.970`$. The frequent-regime closed form, second-moment term included, is within 0.6–7.1%; without that term it is off by 14–21%. The isolated-regime closed form is within 8.6%. Near $`d\approx1-\beta_2`$ neither closed form applies.
- **Water-filling ([Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10)).** In all 16 random profiles water-filling has the lowest realized error; competing allocations are 3% to 105% worse in mean absolute error, and every predicted ratio matches its realized ratio within $`0.022`$.
- **Separation ([Corollary 6.12](../proofs/06_learner_relative_information.md#s6-12)(a)).** With the same demonstrations, the always-clipped learner's error falls like $`K^{-1/2}`$ (from $`0.091`$ at $`K=1`$ to $`0.00024`$ at $`K=10^5`$), while SGD's worst-case error stays at the floor $`0.45`$ for every $`K`$.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 6.1](#c6-1) | [Lemma 6.4](../proofs/06_learner_relative_information.md#s6-4) | `v016:C2.design` | PASS |
| [Check 6.2](#c6-2) | [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3), [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5) | `v016:C2.risk` | PASS |
| [Check 6.3](#c6-3) | [Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7) | `v016:C2.aliasing` | PASS |
| [Check 6.4](#c6-4) | [Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11) | `v016:C3.sensitivity` | PASS |
| [Check 6.5](#c6-5) | [Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10) | `v016:C3.allocation` | PASS |
| [Check 6.6](#c6-6) | [Corollary 6.12](../proofs/06_learner_relative_information.md#s6-12)(a) | `v016:C3.separation` | PASS |

All checks run on CPU in float64 with seeds `numpy.SeedSequence([160160, group, subcase, ...])` (PCG64). The `v016:C2` group takes 2.3 s and the `v016:C3` group 27.3 s.

<a id="c6-1"></a>

## Check 6.1 · The c-optimal design equals the dual information norm (Lemma 6.4)

**Verdict.** PASS. Primal and dual agree on all 216 instances; the worst relative gap is $`4.5\times10^{-13}`$ against a tolerance of $`10^{-6}`$.

**Why.** [Lemma 6.4](../proofs/06_learner_relative_information.md#s6-4) says that the best approximate design value $`\inf_\xi c^\top A(\xi)^{-1}c`$ equals $`\mathfrak n_{K,n}^2`$, a supremum over directions. The two sides are computed by unrelated convex programs; a gap larger than solver accuracy, or a case where one side is finite and the other infinite, would falsify the lemma or its conventions for singular data.

**Setup.**

- Gaussian linear model: $`\theta\in\mathbb R^p`$ with $`p\in\lbrace2,3,5,8\rbrace`$; a demonstration observes $`A_D\theta+N(0,I)`$, so $`I_D=A_D^\top A_D`$; a query at $`x`$ observes $`f_x^\top\theta+N(0,1)`$, so $`I_Q(x)=f_xf_x^\top`$, with $`\lvert\mathcal X_Q\rvert\in\lbrace4,16\rbrace`$.
- Three demonstration families for each $`(p,\lvert\mathcal X_Q\rvert)`$: full rank; an exact null column with a nonzero component of $`c`$ along it (aliased); an exact null column with $`c`$ in the identifiable range (singular but estimable).
- $`K\in\lbrace1,10,100\rbrace`$, $`n\in\lbrace0,10,100\rbrace`$: 216 instances, 864 assertions.
- Primal: minimize $`c^\top M(w)^{-1}c`$ over the simplex of design weights $`w`$ by SLSQP with analytic gradients, plus a check of the simplex KKT (Elfving) condition.
- Dual: minimize $`K\,h^\top I_Dh+n\,t^2`$ subject to $`c^\top h=1`$ and $`t\ge\pm f_x^\top h`$, a convex quadratic program. Where SLSQP reports a status failure, an independent trust-constr solve with the exact Hessian and an active-face KKT linear solve must certify stationarity, feasibility and nonnegative multipliers to $`10^{-9}`$.
- $`n=0`$: spectral range tests; a non-identifiable $`c`$ gives $`+\infty`$ on both sides, an identifiable singular $`c`$ uses the pseudo-inverse.

**Result.**

| Assertion | Instances | Worst residual | Tolerance |
|---|---|---|---|
| Primal–dual relative gap | 216 | $`4.5\times10^{-13}`$ | $`10^{-6}`$ |
| Simplex KKT (Elfving) relative gap | 216 | $`1.96\times10^{-7}`$ | $`10^{-6}`$ |
| Dual feasibility | 216 | $`1.0\times10^{-13}`$ | $`10^{-8}`$ |
| Solver success | 216 | 0 failures | 0 |

- 136 dual solves succeed directly with SLSQP. In 8 cases SLSQP reports a status failure despite accurate values; trust-constr and the active-face KKT certificate then succeed. The 72 cases with $`n=0`$ use the spectral path.
- The 24 aliased cases with $`n=0`$ give $`+\infty`$ on both sides, as the lemma's convention requires.

**Why it supports Lemma 6.4.** Two independent formulations, one over designs and one over directions, give the same number to near machine precision on full-rank, singular and non-identifiable instances. The lemma's exchange of $`\sup`$ and $`\inf`$ is therefore not an artefact of a particular solver. The check uses rank-one query information; the lemma itself does not need it.

<a id="c6-2"></a>

## Check 6.2 · The achieved risk lies between 𝔫/4 and 𝔫 (Theorems 6.3 and 6.5)

**Verdict.** PASS. On all 192 identifiable instances the Monte Carlo mean absolute error of the efficient plug-in estimator, under the integer-rounded c-optimal design, lies in $`[0.789,\,0.850]\cdot\mathfrak n_{K,n}`$. No two-point lower bound exceeds the achieved risk.

**Why.** [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3) puts every distiller at or above $`\mathfrak n/4`$, and [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5) says the c-optimal plug-in achieves $`\mathfrak n(1+o(1))`$. A ratio outside $`[1/4,1]`$ would falsify the pair. Integer rounding of the design is a finite-$`n`$ effect that the asymptotic theorem does not cover, so it is measured rather than assumed away.

**Setup.**

- The 216 instances of Check 6.1 minus the 24 non-identifiable ones: 192 instances, 960 assertions.
- The approximate c-optimal design is rounded to integer counts by largest remainders without changing $`n`$; the rounding loss is the relative increase of the variance $`c^\top A^{-1}c`$.
- Independent Gaussian noise is generated for the demonstration and query sufficient statistics, and the efficient linear estimator of $`c^\top\theta`$ is computed: 30,000 replicates per instance. This simulates observations rather than sampling the error from its predicted variance.
- Recorded alongside: the exact Gaussian mean absolute error $`\sqrt{2/\pi}\sqrt{c^\top A_{\rm int}^{-1}c}`$; the Pinsker two-point bound of [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3)(a) at the least favourable displacement $`\tau=\mathbb I^{-1/2}`$; the exact Gaussian total-variation two-point bound at the same displacement.

**Result.**

| Quantity | Value |
|---|---|
| Achieved risk $`/\,\mathfrak n_{K,n}`$: min / median / max | $`0.7885`$ / $`0.7982`$ / $`0.8504`$ |
| $`\sqrt{2/\pi}`$ (exact design, no rounding) | $`0.7979`$ |
| Largest rounding loss (the instance with ratio $`0.850`$: $`n=10`$ queries over 16 points) | 12.3% of the variance |
| Smallest margin above $`\mathfrak n/4`$ / below $`\mathfrak n`$ (absolute) | $`0.0204`$ / $`0.0075`$ |
| Pinsker two-point bound $`/\,\mathfrak n`$ | $`0.2500`$ in every instance |
| Exact Gaussian TV two-point bound $`/\,\mathfrak n`$ (same displacement) | $`0.3085=1-\Phi(1/2)`$ in every instance |
| Achieved risk $`/`$ Pinsker bound, minimum | $`3.15`$ |
| Monte Carlo vs exact Gaussian MAE, worst | $`2.91`$ standard errors (99.9% pointwise tolerance $`3.29`$) |

![Figure 6.1: information bound, design duality and aliasing](../figures/fig6_1_information_bound.png)

*Figure 6.1.* (a) Achieved plug-in risk divided by the information norm $`\mathfrak n_{K,n}`$ for all 192 identifiable settings, with the factors $`1/4`$ and $`1`$ of Theorems 6.3 and 6.5 and the Gaussian value $`\sqrt{2/\pi}`$; the few points above $`\sqrt{2/\pi}`$ are settings where integer rounding of a small query budget inflates the variance. (b) The c-optimal design value against the dual information objective (Lemma 6.4) on the diagonal; the 24 settings that are infinite on both sides are counted, not plotted. (c) The worst-error floor $`\lvert c^\top h\rvert/2`$ of eight aliased instances against $`K`$, flat in every instance (Corollary 6.7).

**Why it supports Theorems 6.3 and 6.5.**

- The achieved risk sits near $`\sqrt{2/\pi}\,\mathfrak n`$, the exact Gaussian value, so the upper constant $`1`$ of [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5) holds with room, even after integer rounding.
- The Pinsker two-point bound equals $`\mathfrak n/4`$ exactly at the theorem's displacement, so the constant $`1/4`$ in [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3) is what the proof delivers; the exact TV bound at the same point is somewhat larger ($`0.31\,\mathfrak n`$).
- Limits: in this model $`\Psi`$ is linear and the experiments are Gaussian, so the second-order and efficiency conditions (S) and (E) of [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5) hold exactly. The check does not test nonlinear $`\Psi`$ or non-Gaussian experiments.

<a id="c6-3"></a>

## Check 6.3 · Aliased directions leave a floor that no K removes (Corollary 6.7)

**Verdict.** PASS. For every aliased model and every $`K\in\lbrace1,10,100,10^6\rbrace`$ the data laws at $`\theta`$ and $`\theta+h`$ coincide exactly, and the worst error is at least half the functional gap.

**Why.** [Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7) claims that when $`h`$ lies in the null space of the demonstrations and $`c^\top h\neq0`$, no number of demonstrations helps. A nonzero difference in the data laws, or an estimator that beats $`\lvert\Delta\Psi\rvert/2`$ in both worlds, would falsify it.

**Setup.**

- 8 aliased models (an exact zero column of $`A_D`$, so $`A_Dh=0`$ with no numerical tolerance), $`n=0`$, $`K\in\lbrace1,10,100,10^6\rbrace`$: 32 instances, 3,408 assertions.
- Compare the means and covariances of the demonstration sufficient statistics at $`\theta`$ and $`\theta+h`$; check the deterministic triangle inequality behind the bound, and record the midpoint estimator, which attains it.

**Result.**

- Mean and covariance differences: exactly $`0`$ in all 208 comparisons.
- Triangle inequality: 3,200 checks, worst residual $`5.6\times10^{-17}`$ (tolerance $`2\times10^{-14}`$).
- Example: functional gap $`0.1866`$, minimax floor $`0.0933`$, attained by the midpoint at every $`K`$ (Figure 6.1(c)).

**Why it supports Corollary 6.7.** The laws are identical, not merely close, so the floor is exact and independent of $`K`$; this is the mechanism by which replay or rewriting of demonstrations cannot help.

<a id="c6-4"></a>

## Check 6.4 · Learner weights: SGD scales like d², warm Adam like d (Proposition 6.11)

**Verdict.** PASS. Every asserted slope lies within its bootstrap 95% interval enlarged by the predeclared $`0.1`$. Frequent-regime Adam has sensitivity slope $`0.489`$ in $`d`$, hence weight slope $`0.98`$, against $`1.00`$ (weight slope $`2.0`$) for SGD. The closed forms are accurate inside their regimes and fail, as stated, near $`d\approx1-\beta_2`$.

**Why.** [Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11) predicts how strongly one more oracle sample at a state moves the outcome, for each optimizer. The weights decide where queries should go ([Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10)) and drive the separation between SGD and Adam. Measured sensitivities with the wrong slope, or a closed form that misses the second-moment term, would falsify the proposition.

**Setup.**

- Each target state owns a scalar coordinate; one state is visited per step; teacher label $`y\sim\mathrm{Bernoulli}(\pi)`$ with $`\pi=0.35`$; gradient $`g=c_0-y`$ with $`c_0=0.2`$; $`\zeta=1`$, $`\eta=0.01`$, $`\epsilon=10^{-12}`$.
- Learners: SGD; SGD always clipped to $`0.1`$; Adam with $`\beta_1=0.9`$. They share visits and label randomness.
- Sensitivity $`\mathrm d\,\mathbb E\theta_{s,R}/\mathrm d\pi_s`$ by common-random-number central differences ($`\delta=0.02`$; $`\delta=0.04`$ as a diagnostic), with the perturbation applied after a common burn-in.

| Regime | $`\beta_2`$ | Bias correction | $`d(s)`$ (log-spaced) | $`d/(1-\beta_2)`$ | Burn-in | $`R`$ | Replicates |
|---|---|---|---|---|---|---|---|
| frequent | 0.999 | no | 0.005–0.4, 7 values | 5–400 | 20,000 | 30,000 | 384 |
| isolated | 0.9 | no | 0.0005–0.01, 6 values | 0.005–0.1 | 2,000 | 60,000 | 384 |
| transition | 0.999 | no | 0.0003–0.02, 7 values | 0.3–20 | 20,000 | 30,000 | 384 |
| short horizon | 0.999 | yes | 0.0005–0.01, 6 values | 0.5–10 (0.1–2 visits in $`R`$) | 0 | 200 | 20,000 |

- Asserted slopes: $`1`$ for SGD and clipped SGD in every regime; $`1/2`$ for frequent Adam; $`1`$ for isolated and short-horizon Adam. The transition slope is descriptive.

**Result.**

| Regime | Learner | Slope in $`d`$ [bootstrap 95% CI] | Closed form vs measured |
|---|---|---|---|
| frequent | Adam | $`0.4887`$ $`[0.4827,0.4948]`$ | frequent formula: 0.6–7.1% |
| frequent | SGD / clipped SGD | $`1.0029`$ $`[0.9967,1.0093]`$ | exact linear formula |
| isolated | Adam | $`0.9699`$ $`[0.9443,0.9969]`$ | isolated formula, $`\kappa_{\rm iso}=6.16`$: +0.2% to −8.6% |
| isolated | SGD / clipped SGD | $`0.9970`$ $`[0.9715,1.0234]`$ | exact linear formula |
| transition | Adam | $`0.6252`$ $`[0.5952,0.6610]`$ | see below |
| transition | SGD / clipped SGD | $`1.0185`$ $`[0.9896,1.0518]`$ | exact linear formula |
| short horizon | Adam | $`0.9166`$ $`[0.8599,0.9786]`$ | see below |
| short horizon | SGD / clipped SGD | $`0.9722`$ $`[0.9172,1.0328]`$ | exact linear formula |

- **Second-moment term.** In the frequent regime, dropping the second term of [Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11)(b) overpredicts the measured sensitivity by 14% (at $`d=0.005`$) to 21% (at $`d=0.4`$). The largest error of the full formula, 7.1%, is at the smallest $`d`$, where only 5 visits fall in a second-moment window.
- **Isolated regime.** The error of the isolated formula grows from 0.2% at $`d/(1-\beta_2)=0.005`$ to 8.6% at $`0.1`$, as visits start to overlap.
- **Transition ($`d\approx1-\beta_2`$).** The frequent formula is off by −46% at $`d/(1-\beta_2)=0.3`$ and −17% at $`0.6`$, and within 10% from $`1.2`$ upward. The isolated formula is off by −19% at $`0.3`$ and −12.5% at $`0.6`$, and much worse above (−22% to −81%). The measured slope $`0.625`$ lies between the two regimes. The switch between closed forms sits at $`d/(1-\beta_2)\approx1`$.
- **Short horizon with bias correction.** The slope $`0.917`$ is consistent with $`1`$ within the predeclared tolerance, but its 95% interval misses $`1`$ by $`0.021`$, the worst slope residual of the check. The magnitude is 24–33% of the uncorrected isolated formula. This is what [Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11)(c) predicts for bias correction: a first visit at step $`t`$ carries the factor $`\sqrt{1-\beta_2^{\,t}}`$, whose average over $`t\le200`$ is $`0.29`$ (a back-of-envelope computation, not part of the check).
- **Linear learners.** Monte Carlo means agree with the exact formula within $`1.52`$ standard errors on all 52 comparisons (guard $`4.5`$). All 12 rows have positive sensitivities.
- **Finite-difference step.** Switching $`\delta`$ from $`0.02`$ to $`0.04`$ changes the sensitivities by at most $`1.24`$ (frequent Adam at $`d=0.4`$, 0.4% of the value).

![Figure 6.2: learner weights, query allocation and the scale-invariance separation](../figures/fig6_2_learner_weights.png)

*Figure 6.2.* (a) Measured sensitivity against visitation $`d(s)`$ on log-log axes, with pointwise 95% intervals and fitted against predicted slopes in the legend: SGD and clipped SGD $`1`$, frequent-regime Adam $`1/2`$, isolated-regime and short-horizon bias-corrected Adam $`1`$. Horizons and burn-in differ between regimes, so absolute heights are not comparable across series. (b) Realized mean absolute error per profile for water-filling, uniform, proportional-to-$`d`$ and SGD-weight allocations; water-filling is lowest in every profile. (c) Predicted against realized error ratios with familywise intervals, on the diagonal (Proposition 6.10). (d) Error against $`K`$ for always-clipped SGD, which falls like $`K^{-1/2}`$ below its RMS bound, and SGD's worst case over the two worlds, which stays at the aliasing floor $`0.45`$ (Corollary 6.12(a)).

**Why it supports Proposition 6.11.**

- The slopes separate the learners exactly as predicted: SGD weights scale like $`d^2`$, frequent-regime Adam weights like $`d`$, and isolated-regime Adam returns to $`d^2`$ with a sign-only dependence.
- The closed form with the second-moment term is accurate where its regime holds, and the term is necessary: without it the error rises from 0.6–7.1% to 14–21%.
- The deviations are where the proposition says its closed forms stop: near $`d\approx1-\beta_2`$, and for the constant (not the slope) under bias correction with a short horizon.

<a id="c6-5"></a>

## Check 6.5 · Water-filling minimizes the realized error (Proposition 6.10)

**Verdict.** PASS. In all 16 profiles the water-filling allocation has the smallest realized mean absolute error, and the predicted error ratios of the five competing allocations match the realized ones within Monte Carlo error.

**Why.** [Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10) predicts both the optimal allocation and, through $`\sum_sW_s/N_s`$, how much worse every other allocation is. A competitor with lower realized error, or a predicted ratio outside its simultaneous confidence interval, would falsify it.

**Setup.**

- 16 random profiles with 6–10 states: log-spaced student visitation $`d`$ (shuffled), Dirichlet teacher visitation $`d_T`$, Bernoulli laws with $`\pi\in[0.15,0.85]`$, and learner-relative weights $`W_s=d(s)\,e^{N(0,1.44)}`$ (linear in $`d`$, as for frequent-regime Adam).
- 800 teacher visits and $`n=4{,}000`$ queries. The plug-in functional is exactly linear in the state probabilities, with coefficient $`\sqrt{W_s/(\pi_s(1-\pi_s))}`$, so its variance formula is exact.
- Water-filling by a scalar root for $`\lambda`$, then discretized by marginal-gain exchanges. Competitors: uniform; proportional to $`d`$; water-filling with SGD-type weights $`W_s\,d(s)`$ (the weights one would use for the wrong learner); gradient norm ($`\propto d\sqrt{0.04+0.6\pi}`$); uncertainty ($`\propto\pi(1-\pi)`$).
- Binomial sufficient statistics model the integer visits: 60,000 replicates per arm. Ratio intervals are familywise 95% (Bonferroni over both ratios, all profiles and all five comparisons).
- One extra case: a state with $`W>0`$ and no teacher or query visits.

**Result.**

| Competing allocation | Realized MAE ratio to water-filling: min / median / max | Largest $`\lvert`$predicted − realized$`\rvert`$ |
|---|---|---|
| uniform | $`1.136`$ / $`1.415`$ / $`1.858`$ | $`0.010`$ |
| proportional to $`d`$ | $`1.033`$ / $`1.209`$ / $`1.695`$ | $`0.009`$ |
| water-filling with SGD-type weights | $`1.078`$ / $`1.165`$ / $`1.315`$ | $`0.011`$ |
| gradient norm | $`1.028`$ / $`1.188`$ / $`1.743`$ | $`0.011`$ |
| uncertainty | $`1.146`$ / $`1.441`$ / $`2.054`$ | $`0.022`$ |

- Budget residual at most $`5.9\times10^{-11}`$; KKT residual $`4.4\times10^{-16}`$; the continuous and integer allocations are optimal among the compared ones.
- In every profile the best competitor is 2.8% to 21.9% worse than water-filling in realized mean absolute error.
- Worst standardized deviation between predicted and realized ratios: $`3.23`$, below the familywise critical value $`3.60`$.
- Irreducible case: weights $`(1,2)`$, teacher counts $`(100,0)`$, no queries. The predicted variance is $`+\infty`$; the two candidate laws $`0.25`$ and $`0.75`$ at the unseen state give identical observations; functional gap $`0.5`$, minimax floor $`0.25`$.

**Why it supports Proposition 6.10.**

- The formula $`\sum_sW_s/N_s`$ predicts realized errors to about one percent of the ratio, and the allocation it implies wins in every profile.
- Using another learner's weights costs 8% to 32%: the allocation is learner-relative, not a property of the data alone.
- Limits: the profiles are synthetic and the plug-in functional is linear; the check does not estimate $`W_s`$ from a real learner.

<a id="c6-6"></a>

## Check 6.6 · The same demonstrations suffice for clipped SGD and not for SGD (Corollary 6.12(a))

**Verdict.** PASS. The clipped learner's increment is exactly scale invariant and its error decays like $`K^{-1/2}`$; SGD's worst-case error never drops below half the functional gap.

**Why.** [Corollary 6.12](../proofs/06_learner_relative_information.md#s6-12)(a) claims an exact separation: demonstrations that reveal $`\alpha`$ but not the scale $`b`$ suffice for an always-clipped learner and are useless for SGD. Dependence of the clipped increment on $`b`$, or an SGD error below the floor, would falsify it.

**Setup.**

- $`\alpha=0.65`$; gradient support $`\lbrace0.2,-0.8\rbrace`$ scaled by $`b\in\lbrace1,3\rbrace`$; clipping threshold $`0.1`$, so clipping is always active; $`\eta R=1`$.
- Demonstrations observe labels only, so their law is $`\mathrm{Binomial}(K,\alpha)`$ and does not depend on $`b`$.
- $`K=1,10,\dots,10^5`$; 60,000 replicates. The clipped functional and a midpoint-scale SGD estimator are evaluated in both worlds.

**Result.**

| $`K`$ | Clipped learner MAE | RMS bound | SGD worst-case error | Floor $`\lvert\Delta\Psi\rvert/2`$ |
|---|---|---|---|---|
| 1 | $`0.0910`$ | $`0.0954`$ | $`1.0448`$ | $`0.45`$ |
| 10 | $`0.0247`$ | $`0.0302`$ | $`0.4714`$ | $`0.45`$ |
| 100 | $`0.00762`$ | $`0.00954`$ | $`0.4502`$ | $`0.45`$ |
| 1,000 | $`0.00240`$ | $`0.00302`$ | $`0.4501`$ | $`0.45`$ |
| 10,000 | $`0.000763`$ | $`0.000954`$ | $`0.4500`$ | $`0.45`$ |
| 100,000 | $`0.000241`$ | $`0.000302`$ | $`0.4500`$ | $`0.45`$ |

- Scale invariance of the clipped increment: exact in all 12 comparisons.
- The SGD estimator used here approaches the floor; the floor itself holds for every estimator because the two worlds have identical demonstration laws.

**Why it supports Corollary 6.12(a).** The separation is exact: with the same demonstrations, one learner's error vanishes at the parametric rate and the other's is bounded away from zero for every $`K`$. Which demonstrations are sufficient depends on the learner.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C2,C3 --output /tmp/v016_C2_C3.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

The first command regenerates the full result file (about 104 s); the second runs only the groups of this page into a separate file.

## Artifacts

- `theory/v016_checks/results_v016.json`: claim ids `v016:C2.design`, `v016:C2.risk`, `v016:C2.aliasing`, `v016:C3.sensitivity`, `v016:C3.allocation`, `v016:C3.separation`, each with assertions, worst residuals, failures and data rows.
- `theory/v016_checks/results_v016_raw.json.gz`: complete rows (design inputs, per-arm counts); not tracked by git, its SHA-256 is recorded under `raw_artifact` in the main file.
- `theory/v016_checks/SPEC.md` (the sections for `v016:C2` and `v016:C3`) and `theory/v016_checks/check_v016.py` (functions `c2`, `c3`).
- `figures/fig6_1_information_bound.png`, `figures/fig6_2_learner_weights.png` (PDF versions alongside).
