# Proposition 5.8 · Evolution of warmth

*Experiments*

[← Framework](../README.md) · [Statement](../statements/05-08_evolution_warmth.md) · [Proof](../proofs/05-08_evolution_warmth.md) · [Experiments](../experiments/05-08_evolution_warmth.md) · [中文](05-08_evolution_warmth_ZH.md)

**What is tested here.** [Check 5.11](../experiments/05-08_evolution_warmth.md#c5-11) tests the exact law, the bounds, the plateau and the accumulation bound with exact recursions and Monte Carlo. [Check 5.12](../experiments/05-08_evolution_warmth.md#c5-12) measures warmth on actual training trajectories, which cover only the early phase. [Check 5.13](../experiments/05-08_evolution_warmth.md#c5-13) follows the cold window over a training run.

**Verdict.**

- **Proposition 5.8 (evolution of warmth).**
  - The exact recursion matches the closed form to $`7.1\times10^{-13}`$ for $`s`$ up to $`10^6`$.
  - Both bounds hold at every age, and the upper bound is within a factor 1.33 of $`\bar\kappa(s)`$. The plateau $`(1-\beta_2)/\beta_2`$ is reached and does not vanish.
  - Under stationary Gaussian, heavy-tailed and sparse gradients, 75/75 Monte Carlo intervals contain $`\sigma^2(1-\beta_2^{\,s})`$. The accumulation bound holds on 9,000 grid points.
  - Real trajectories cover only the early phase. There the median warmth falls with log–log slope $`-1.06`$, and lr = 0 warm-up and real steps count alike.

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

**Status labels.** `PASS` means the implemented consistency assertions passed. `INFO` marks an empirical diagnostic: each cell gets its own PASS, FAIL or UNAVAILABLE status, and the group as a whole makes no pass/fail claim. Unless a check says otherwise, every network is the controlled small network of `v014:C23`: one-hot context, 4 tanh units, a sigmoid output, mean cross-entropy, float64. None of these networks is a real model.

**Statement under test ([Proposition 5.8](../statements/05-08_evolution_warmth.md#s5-8)).** Let the second moment start at $`v_0=0`$ and absorb gradient evaluations $`G_1,G_2,\dots`$ by $`v_s=\beta_2v_{s-1}+(1-\beta_2)G_s^{\odot2}`$, coordinatewise. Write $`\gamma=1-\beta_2`$.

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

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.11](../experiments/05-08_evolution_warmth.md#c5-11) | [Proposition 5.8](../statements/05-08_evolution_warmth.md#s5-8)(a)–(c) | `v016:C1.exact`, `v016:C1.moments`, `v016:C1.accumulation` | PASS |
| [Check 5.12](../experiments/05-08_evolution_warmth.md#c5-12) | [Proposition 5.8](../statements/05-08_evolution_warmth.md#s5-8)(a) early phase and (d) warm-up, on real trajectories | `v015:E8` (warmth-trajectory part) | PASS |
| [Check 5.13](../experiments/05-08_evolution_warmth.md#c5-13) | [Propositions 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6)–[5.8](../statements/05-08_evolution_warmth.md#s5-8) over a training run | `v015:E8` (`data.gaps`) | PASS |

<a id="c5-11"></a>

## Check 5.11 · The exact warmth law: bounds, plateau, accumulation (Proposition 5.8(a)–(c))

**Verdict.**

- **Exact recursion.** For constant gradients, the raw second-moment recursion matches the closed form $`\bar\kappa(s)=(1-\beta_2)/(\beta_2(1-\beta_2^{\,s}))`$ to $`7.1\times10^{-13}`$ relative, for five values of $`\beta_2`$ and $`s`$ up to $`10^6`$.
- **Bounds and plateau.** Both bounds of part (a) hold at every one of 10,900 sampled ages, and the upper bound is at most a factor 1.33 above $`\bar\kappa`$. Warmth reaches the plateau $`(1-\beta_2)/\beta_2`$ and stays there.
- **Bias correction.** It leaves the warmth ratio unchanged exactly (part (b)).
- **Stationary random gradients.** 75/75 Monte Carlo 99.9% intervals contain $`\sigma^2(1-\beta_2^{\,s})`$.
- **Accumulation.** The bound of part (c) holds on all 9,000 grid points. Its linear term is needed: no purely reciprocal envelope $`c/(W+s)`$ survives once $`s(1-\beta_2)`$ is of order one.

**Why.**

- Part (a) gives an exact law for the mean-field warmth and two-sided bounds. Its main content is a nonzero plateau: warmth falls like $`1/(\beta_2s)`$ only while $`s(1-\beta_2)\ll1`$, and with $`\beta_2`$ fixed it does not go to zero.
- Part (c) turns this into an accumulation bound with a linear term $`c(1-\beta_2)R`$.
- A single wrong exponent or constant would show up as a violated bound at some age, or as a Monte Carlo mean outside its interval.

**Setup** (`theory/v016_checks`; CPU; fixed `SeedSequence` seeds).

- `v016:C1.exact`:
  - $`\beta_2\in\lbrace 0.9,0.99,0.995,0.999,0.9999\rbrace`$, constant gradient 1.7, raw $`v_0=0`$;
  - one million updates per $`\beta_2`$, run with a causal IIR filter;
  - every $`s=1,\dots,2000`$, plus 180 log-spaced ages up to $`10^6`$ (2,180 ages per $`\beta_2`$);
  - the closed form is evaluated with a stable `expm1`;
  - bias correction is compared in exact rational arithmetic at $`t=1,7,50`$.
- `v016:C1.moments`:
  - 16,000 independent replicates, with checkpoints $`s\in\lbrace 1,10,100,1000,2000\rbrace`$;
  - gradients: unit Gaussian; Student $`t_3`$ rescaled to unit second moment; Bernoulli(0.05) × Gaussian$`/\sqrt{0.05}`$;
  - intervals: Studentized 99.9% intervals for the Gaussian and sparse laws. Squared $`t_3`$ gradients have infinite variance, so for them $`g^2`$ is capped at 100, a bounded-variable Bernstein interval is used, and the analytically known tail expectation is added to its centre.
- `v016:C1.accumulation`:
  - $`W\in\lbrace 1,4,32\rbrace`$, $`c\in\lbrace 1,2,4\rbrace`$, every $`R=1,\dots,100`$ plus a log grid to $`10^5`$;
  - the capped envelope itself is summed, which dominates every admissible warmth sequence;
  - for $`c\in\lbrace 1,1/\beta_2,2\rbrace`$, the first age at which $`\bar\kappa\le c/(W+s)`$ fails is recorded.
- A nonstationary diagnostic, outside the hypothesis and outside the assertions: second moments $`e^{-s/400}`$ with $`\beta_2=0.999`$.

**Result** (`theory/v016_checks/results_v016.json`; claims `v016:C1.exact`, `v016:C1.moments`, `v016:C1.accumulation`).

| Claim id | Status | Instances | Assertions | Worst residual |
|---|---|---:|---:|---|
| `v016:C1.exact` | PASS | 6 | 32,715 | recursion $`7.1\times10^{-13}`$ relative (tolerance $`10^{-10}`$); lower-bound violation at most $`2.2\times10^{-16}`$, i.e. rounding (tolerance $`2\times10^{-13}`$); bias correction exactly 0 |
| `v016:C1.moments` | PASS | 75 | 75 | no target outside its 99.9% interval |
| `v016:C1.accumulation` | PASS | 135 | 9,045 | largest lhs − rhs $`=-4.7\times10^{-4}`$ |

Exact law, by $`\beta_2`$ (`v016:C1.exact`):

| $`\beta_2`$ | plateau $`(1-\beta_2)/\beta_2`$ | largest upper bound / $`\bar\kappa`$ |
|---|---:|---:|
| 0.9 | 0.1111 | 1.324 |
| 0.99 | 0.01010 | 1.301 |
| 0.995 | 0.005025 | 1.300 |
| 0.999 | 0.001001 | 1.299 |
| 0.9999 | 0.0001000 | 1.298 |

The upper bound is loosest in the transition, at $`s(1-\beta_2)\approx1.8`$ (for $`\beta_2=0.999`$, at $`s=1791`$). As $`\beta_2\to1`$ the ratio tends to $`\max_y(1+y)(1-e^{-y})/y\approx1.298`$.

Values at $`\beta_2=0.999`$ (from the raw arrays):

| $`s`$ | 1 | 10 | 100 | 1,000 | $`10^6`$ |
|---|---:|---:|---:|---:|---:|
| $`\bar\kappa(s)`$ | 1.001 | 0.1006 | 0.01051 | 0.001583 | 0.001001 |
| $`1/(\beta_2s)`$ | 1.001 | 0.1001 | 0.01001 | 0.001001 | $`1.0\times10^{-6}`$ |

At $`s=10^6`$ the reciprocal law is a thousand times too small, and $`\bar\kappa`$ equals the plateau.

Stationary random gradients (`v016:C1.moments`; 25 cells per family):

| Family | Interval method | Largest relative deviation of the mean | Targets inside the 99.9% interval |
|---|---|---:|---:|
| Gaussian | Studentized | 0.76% | 25/25 |
| Student $`t_3`$ (unit second moment) | Bernstein on capped $`g^2`$ + analytic tail | 5.6% (at $`s=1`$, $`\beta_2=0.9`$) | 25/25 |
| sparse Gaussian ($`p=0.05`$) | Studentized | 2.4% | 25/25 |

Accumulation (`v016:C1.accumulation`):

- The bound $`c\log(1+R/W)+c(1-\beta_2)R`$ dominates the summed envelope at every grid point. At $`R=10^5`$, the ratio lhs/rhs lies between 0.965 and 0.999998, so the bound is nearly tight.
- **The linear term is necessary.** Take the reciprocal envelope $`\bar\kappa\le c/(W+s)`$ with $`c=2\ge1/\beta_2`$, counted from the actual raw age.
  - It first fails at raw age 14 ($`\beta_2=0.9`$), 158 (0.99), 317 (0.995), 1,592 (0.999) and 15,935 (0.9999), that is, near $`s(1-\beta_2)\approx1.6`$.
  - For $`c<2`$ it fails at the first step after the burn-in.

Nonstationary diagnostic ($`\beta_2=0.999`$, second moment $`e^{-s/400}`$):

- At $`s=2991`$ the exact $`\mathbb E v_s`$ is 0.0331. Substituting the current second moment into the stationary formula gives 0.00054.
- So $`v`$ lags behind the shrinking gradients, and warmth is about 60 times smaller than the stationary law predicts.
- This lies outside the hypothesis of part (a), and it is not asserted.

![Figure 5.3: optimizer warmth](../figures/fig5_3_warmth.png)

*Figure 5.3.*
- (A) A lr = 0 moment warm-up removes the gap between equal-moment sources at short horizons ($`R=1,4`$). At $`R=40`$ it no longer drives the gap to zero, and where the residual is largest the SGD comparator is nonzero as well.
- (B) The exact mean-field warmth $`\bar\kappa(s)`$ for four values of $`\beta_2`$, with its two-sided bounds (band) and the plateau $`(1-\beta_2)/\beta_2`$ (dashed). Warmth falls like $`1/(\beta_2s)`$ only in the early phase $`s(1-\beta_2)\ll1`$, then levels off and does not vanish. The measured median warmth of the network trajectories ([Check 5.12](../experiments/05-08_evolution_warmth.md#c5-12)) lies in that early phase.
- (C) The residual of the warm-Adam affine bound never exceeds its bound $`r`$, and comes within 0.2% of it.

[PDF](../figures/fig5_3_warmth.pdf)

**Why it supports Proposition 5.8.**

- Part (a) holds exactly, including the plateau. The constant-gradient case is the equality case of the mean-field formula.
- The bounds are tight within a factor 1.33 at the transition, and exact in both limits.
- The mean-field law holds for heavy-tailed and sparse stationary gradients as well, which shows that it depends only on the second moment.
- Part (c) holds, and its linear term $`c(1-\beta_2)R`$ cannot be dropped. With $`\beta_2`$ fixed, warm Adam therefore carries a per-step non-affine floor of order $`c(1-\beta_2)`$ relative to the frozen-denominator description.
- Limit of the evidence: for nonstationary gradients the stationary formula does not apply. When the gradients shrink, warmth falls faster than it predicts.

<a id="c5-12"></a>

## Check 5.12 · Warmth on actual trajectories: the early phase, and warm-up counts like real steps (Proposition 5.8(a),(d))

**Verdict.**

- The network trajectories reach at most 129 absorbed evaluations at $`\beta_2=0.999`$, so $`s(1-\beta_2)\le0.13`$ and they cover only the early phase of Proposition 5.8(a).
- There, at $`\eta=0.01`$ and $`W=0`$, the median warmth falls with median log–log slope $`-1.06`$ over 9 fitted task/source series (median $`R^2=0.999`$). This is the $`1/(\beta_2s)`$ behaviour that part (a) predicts for $`s(1-\beta_2)\ll1`$.
- Warmth depends only on the total number of absorbed gradient evaluations, whether those came from lr = 0 warm-up or from real updates, as part (d) states.

**Why.**

- [Check 5.11](../experiments/05-08_evolution_warmth.md#c5-11) tests the mean-field law with synthetic gradients. Here the question is whether the early-phase behaviour shows up in the coordinatewise warmth of actual Adam trajectories, whose gradients are neither stationary nor identically distributed across coordinates.
- A lr = 0 moment warm-up is a cheap way to warm the state. It is only interchangeable with real steps if $`v`$ advances the same way under both.

**Setup** (`v015:E8`, warmth-trajectory part).

- Actual Adam trajectories of the network: moment and parity tasks, $`\eta\in\lbrace 0.01,0.1\rbrace`$, 128 replicas, $`\beta_2=0.999`$.
- Warm-up: $`W\in\lbrace 0,1,2,5,10,20,50\rbrace`$ lr = 0 steps on the reference law. Every warm-up step is asserted to leave the parameter bytes unchanged.
- Recorded: quantiles of the coordinatewise warmth at every step.
- Fit: median $`=c/(W+s)`$ for $`s=1,\dots,79`$ prior updates, with a free log–log slope.

**Result** (`v015:E8`, warmth-trajectory part).

| Setting | Median slope | Median $`R^2`$ |
|---|---|---|
| $`\eta=0.01`$, $`W=0`$ | $`-1.06`$ | 0.999 |
| $`\eta=0.01`$, $`W\in\lbrace 5,10,20,50\rbrace`$ | $`-1.07`$ to $`-1.14`$ | 0.95–0.99 |
| $`\eta=0.1`$, $`W=0`$ | $`-1.18`$ | 0.94 |
| $`\eta=0.1`$, $`W\in\lbrace 10,20,50\rbrace`$ | $`-1.55`$ to $`-1.93`$ | 0.55–0.68 |

Interchangeability (moment $`N=3`$, $`\eta=0.01`$, source $`P`$): the median $`\kappa`$ at an equal number of prior evaluations $`W+s`$.

| $`W+s`$ | $`W=0`$ | $`W=5`$ | $`W=20`$ | $`W=50`$ | mean-field $`\bar\kappa(W+s)`$ |
|---|---|---|---|---|---|
| 51 | 0.00661 | 0.00642 | 0.00619 | 0.00619 | 0.0201 |
| 79 | 0.00405 | 0.00395 | 0.00413 | 0.00379 | 0.0132 |

**Why it supports Proposition 5.8(a),(d).**

- In the early phase, the measured median warmth has the slope that part (a) predicts.
- The median sits a roughly constant factor below $`\bar\kappa`$ (0.33 at $`W+s=51`$, 0.31 at 79). That is expected: $`\bar\kappa`$ is a ratio of means, while the median of the coordinatewise ratio between a squared gradient and its running average is smaller.
- At $`\eta=0.1`$ the consumed gradients shrink as training converges, and warmth falls faster than reciprocally. This matches the nonstationary diagnostic of [Check 5.11](../experiments/05-08_evolution_warmth.md#c5-11).
- Warm-up evaluations and real evaluations are interchangeable for warmth, which is part (d).
- Limits of the evidence:
  - The trajectories are too short to reach the plateau $`(1-\beta_2)/\beta_2=0.001`$, which sets in near $`s\approx1/(1-\beta_2)=1000`$. The plateau is tested in [Check 5.11](../experiments/05-08_evolution_warmth.md#c5-11).
  - Quantiles describe the bulk of the coordinates. The hypotheses of Propositions 5.6 and 5.8(c) concern the maximum or the sensitivity-weighted sum ([Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11)), and [Check 5.9](../experiments/05-06_warm_adam_first_order_affine.md#c5-9) checks the maximum.

<a id="c5-13"></a>

## Check 5.13 · The cold window over a training run (Propositions 5.6–5.8)

**Verdict.**

- At short horizons, the gap between equal-moment sources is a cold-window effect. A lr = 0 moment warm-up of $`W=10`$ removes a median 99% of it at $`R=1`$ and 97.5% at $`R=4`$. For moment $`N=3`$, $`\eta=0.1`$ the gap falls from 0.0769 to 0.0004 at $`R=1`$, and from 0.1196 to 0.0016 at $`R=4`$.
- At $`R=40`$, gaps up to 0.049 remain for $`W\ge10`$. Where the residual is largest, the SGD comparator is nonzero as well (up to 0.011).

**Why.** Propositions 5.6–5.8 make two predictions about a full training run:

- the warmth-controlled part of the gap concentrates in the first updates, and warm-up removes it;
- the curvature part does not depend on warmth, and it accumulates with the horizon.

**Setup** (`v015:E8`, `data.gaps`).

- The branches of [Check 5.7](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-7) (actual Adam, frozen-Jacobian Adam, tangent affine SGD) are reused unchanged.
- Grid: 5 tasks (moment $`N=2,3,4`$; parity $`N=2,3`$) × $`\eta\in\lbrace 0.01,0.1\rbrace`$ × $`R\in\lbrace 1,4,40\rbrace`$ × $`W\in\lbrace 0,1,2,5,10,20,50\rbrace`$ = 210 cells.
- Histories are identical across $`W`$ and across branches. They are enumerated exactly when there are at most 4,096 of them; otherwise each source gets 2,000 independent runs.
- Warm-up uses one declared IID prefix from the reference law.

**Result** (`v015:E8`, `data.gaps`).

| Horizon | largest gap at $`W=0`$ | largest gap at $`W\ge10`$ | $`\lvert\text{gap}(W{=}10)\rvert/\lvert\text{gap}(W{=}0)\rvert`$ | largest SGD comparator |
|---|---|---|---|---|
| $`R=1`$ | 0.0769 | 0.0006 | median 0.009, max 0.034 (5 cells with gap $`>10^{-3}`$) | 0.0034 over $`R\le4`$ |
| $`R=4`$ | 0.1196 | 0.0064 | median 0.025, max 0.226 (7 cells) | (same) |
| $`R=40`$ | 0.0436 | 0.0487 | not driven to zero | 0.0110 |

| Cell | $`W=0`$ | $`W=1`$ | $`W=2`$ | $`W=5`$ | $`W=10`$ | $`W=20`$ | $`W=50`$ | tangent SGD |
|---|---|---|---|---|---|---|---|---|
| moment $`N=3`$, $`\eta=0.1`$, $`R=1`$ | 0.0769 | 0.0460 | 0.0311 | 0.0032 | 0.0004 | 0.0004 | 0.0002 | 0.0000 |
| moment $`N=3`$, $`\eta=0.1`$, $`R=4`$ | 0.1196 | 0.1016 | 0.0881 | 0.0206 | 0.0016 | 0.0037 | 0.0033 | 0.0000 |
| parity $`N=2`$, $`\eta=0.1`$, $`R=40`$ | 0.0436 | 0.0477 | 0.0441 | 0.0488 | 0.0487 | 0.0391 | 0.0391 | 0.0110 |

- With $`W\in\lbrace 1,2\rbrace`$ the warmth is still of order one, so Proposition 5.6 gives no control and the gap can move either way. For example, moment $`N=2`$, $`\eta=0.1`$, $`R=1`$ goes from 0.0175 to $`-0.0194`$ at $`W=1`$.
- For $`W\ge10`$ the short-horizon gaps are uniformly small.

**Why it supports Propositions 5.6–5.8.**

- At short horizons, the gap between equal-moment sources is the cold-step separation of Proposition 5.7, and warming the state removes it, as Propositions 5.6 and 5.8 predict.
- At long horizons, warm-up no longer drives the gap to zero. The largest residuals, parity $`N=2`$ and moment $`N=2`$ at $`\eta=0.1`$, stay at 0.039–0.049 and 0.005–0.036 for every $`W\ge10`$. The SGD comparator is nonzero in those cells (0.011 and 0.0067).
  - This residual is the curvature and trajectory term, accumulated over 40 updates and shared with SGD. It is not a cold-start effect.
- In the first updates, warmth decides which statistic a feedback source must match. Over long horizons, what remains is the nonlinearity of the trajectory.
- Whether warm-up changes the predictive value of cheap proxies is a method question. It is answered in [§9 experiments](09-03_derivation_design_rules.md#c9-5).

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
