# Theorem 8.8 · CRN paired-replacement certificate; Theorem C

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-08_crn_paired_replacement_certificate.md) · [Proof](../proofs/08-08_crn_paired_replacement_certificate.md) · [Experiments](../experiments/08-08_crn_paired_replacement_certificate.md) · [中文](08-08_crn_paired_replacement_certificate_ZH.md)

**What is tested here.**

- **[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) (CRN paired-replacement certificate, Theorem C).** Measured on the frozen controlled panel of 1,728 candidate pairs:
  - As a ranker, the point estimate $`\hat S`$ has the wrong sign on 48 of 1,687 nonzero-gold pairs (2.8%).
  - Its measurement uses 7.0% of the trajectories of a full-tail paired audit at $`r=64`$, and 28.0% at $`r=256`$.
  - Common random numbers cut the variance of the paired difference to 0.12–0.15 of independent tails (ratio of summed variances).
  - The strict certificate with the a-priori range $`B=2`$ certifies 0 pairs with the aligned surrogate, and 8 pairs with the calibrated surrogate at $`r=256`$; none is wrong. This is the gap recorded in [Remark 8.12](../statements/08-12_open_problems.md#s8-12).

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- Rounds are $`t=0,\dots,R-1`$, and $`z`$ is a complete learner state. For a first-order optimizer the actual update is $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$.
- At state $`z`$ in round $`t`$ the reference emits $`Y\sim P^0_{t,z}`$, and candidate $`j\in\lbrace1,\dots,M\rbrace`$ emits $`Y\sim P^j_{t,z}`$. The kernel of candidate $`j`$ is $`K^j_t`$.
- The reference continuation is $`V^0_{t+1}`$, with $`V^0_R=F_E\in[0,1]`$. The response is $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$. Integrating the message gives the kernel identity

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- $`L_E(j)=\mathbb E\,F_E(Z^j_R)`$ for the process driven by candidate $`j`$. All processes start from the same initial law of $`Z_0`$.
- The *local advantage* of candidate $`j`$ at $`(t,z)`$ is

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- It is the effect on the final score of replacing one reference round by one round of candidate $`j`$, followed by the reference tail. By [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2), the whole-run difference is a sum of such terms over the states the alternative process reaches.

*The paired-replacement score* ([Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6)).

- Fix an audited state $`z`$ at round $`t`$ (a complete learner state, including the optimizer state), conditionally on the history that produced it. The candidates are $`j=0,1,\dots,M`$ with $`M\ge1`$; $`j=0`$ is the reference. Candidate $`j`$ has message law $`P^j=P^j_t(\cdot\mid z)`$, and a message $`Y`$ produces the increment $`\Delta^j=\Delta_t(z,G_t(z,Y,\xi))`$, update randomness included.
- A *surrogate tail* $`T_\omega`$ maps a state to a final state by running a declared (possibly truncated) reference tail with randomness $`\omega`$. A *surrogate objective* $`\tilde F`$ scores the final state (for example, the teacher-forced log-likelihood of held-out successful trajectories, mapped to $`[0,1]`$). Write

```math
\varphi_\omega(u)=\tilde F\big(T_\omega(z+u)\big),\qquad\tilde L(P^j)=\mathbb E\,\varphi_\omega(\Delta^j),
```

where $`\omega`$ is independent of $`\Delta^j`$. $`\tilde L(P^j)`$ is the surrogate outcome of one round with candidate $`j`$ followed by the surrogate tail.

- *Replicas* $`\ell=1,\dots,r`$, $`r\ge2`$: draw a fresh $`\omega_\ell`$ and, for every candidate, one message and its increment $`\Delta^j_\ell`$. All candidates in replica $`\ell`$ share $`\omega_\ell`$ (*common random numbers*); their messages may also be coupled through common random numbers, as long as each $`\Delta^j_\ell`$ has its correct marginal law. Replicas are independent of each other.
- If a candidate's message law has finite support, the draw may be replaced by the exact weighted average over the support; the randomness then comes only from $`\omega`$.
- The score of the ordered pair $`(j,k)`$ is

```math
Z^{jk}_\ell=\varphi_{\omega_\ell}(\Delta^j_\ell)-\varphi_{\omega_\ell}(\Delta^k_\ell),\qquad\hat S^{jk}=\frac1r\sum_{\ell=1}^rZ^{jk}_\ell .
```

- On the task side, $`L_F(P^j)=\mathbb E\,F_E\big(T^{\rm full}(z+\Delta^j)\big)`$ is the outcome of one round with candidate $`j`$ followed by the full reference tail and the task objective. In the notation of the setting, $`L_F(P^j)=(K^j_tV^0_{t+1})(z)`$, so $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$.

*Its assumptions* ([Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7)).

- **(I) Independent replicas.** Conditionally on the history that produced $`z`$, the vectors $`(Z^{jk}_\ell)_{j,k}`$, $`\ell=1,\dots,r`$, are i.i.d.; $`r`$ is fixed before the scores are observed.
- **(B) Bounded scores.** For every pair, $`Z^{jk}_\ell`$ lies almost surely in a known interval of length $`B`$. For example $`\tilde F\in[0,1]`$ gives $`B\le2`$.
- **(A) Surrogate alignment.** There are $`\alpha>0`$ and $`\varepsilon_{\rm sur}\ge0`$ such that, for all pairs,

```math
\Big\lvert\big(L_F(P^j)-L_F(P^k)\big)-\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)\Big\rvert\le\alpha\,\varepsilon_{\rm sur}.
```

$`\varepsilon_{\rm sur}`$ absorbs both the difference between the surrogate and the task objective and the truncation of the tail.

**Statement under test ([Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8)).** Under [Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7), let $`\hat V^{jk}=\frac1{r-1}\sum_\ell(Z^{jk}_\ell-\hat S^{jk})^2`$ be the unbiased sample variance and

```math
e^{jk}_{\rm stat}=\sqrt{\frac{2\hat V^{jk}\log(4M^2/\delta)}{r}}+\frac{7B\log(4M^2/\delta)}{3(r-1)} .
```

- **(a) Exactness.** $`\mathbb E\,Z^{jk}_\ell=\tilde L(P^j)-\tilde L(P^k)`$. No first-order approximation is involved.
- **(b) Certificate.** With probability at least $`1-\delta`$, simultaneously for all ordered pairs $`(j,k)`$, $`j\ne k`$,

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big).
```

- **(c) Decision rule.** Declaring "$`j`$ is better than $`k`$" whenever $`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ makes no wrong declaration, over all pairs simultaneously, with probability at least $`1-\delta`$. With $`k=0`$ this certifies $`A_t(z,j)>0`$.
- The certificate needs no continuation gradient $`\nabla V`$, no curvature constant and no second-order remainder. The only constants to calibrate are $`(\alpha,\varepsilon_{\rm sur})`$ ([Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11)).

<a id="c8-14"></a>

## Check 8.14 · Finite CRN paired replacement on the controlled panel (Theorem 8.8, Proposition 8.10, Remarks 8.11–8.12)

**Verdict.**

- **No wrong strict certificate** at either replica count: the assertion of [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) passes.
- **The strict certificate is nearly empty.** With the a-priori range $`B=2`$, the aligned surrogate certifies 0 of 1,728 pairs and the calibrated short surrogate 0 ($`r=64`$) and 8 ($`r=256`$). The range term of the empirical Bernstein radius is 0.156 at $`r=256`$, against a median true gap of 0.0037. The zero-error statement is therefore nearly vacuous; this is the open problem of [Remark 8.12](../statements/08-12_open_problems.md#s8-12).
- **As a ranker the measurement works.** The aligned point estimate $`\hat S`$ has the wrong sign on 48 of 1,687 nonzero-gold pairs (2.8%) at $`r=256`$, and on 65 (3.9%) at $`r=64`$.
- **It is cheap.** Measurement takes 7.0% ($`r=64`$) and 28.0% ($`r=256`$) of the trajectories of a full-tail paired audit of the same panel.
- **CRN pays.** Common random numbers reduce the variance of the paired difference to 0.12–0.15 of independent tails (ratio of summed variances); the median pair ratio is 0.0035.
- **Calibration transfer can fail.** With the observed range in place of $`B`$, which is a diagnostic only, the aligned surrogate certifies 45–57% of pairs with no error, but the calibrated surrogate certifies 23–28% with 22–28 wrong certificates.

**Why.** [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) replaces the oracle continuation gradient by a measurable quantity: the effect of one finite replacement at the audited state, estimated with common random numbers. Its estimand is exact in expectation and needs no smoothness, so its only error terms are statistical ($`e_{\rm stat}`$) and surrogate calibration ($`\varepsilon_{\rm sur}`$). The check asks three questions:

- whether any strict certificate is wrong (it must not be);
- how much the a-priori range costs (Remark 8.12);
- whether the calibration of Remark 8.11 transfers from calibration states to held-out states.

**Setup.**

- **Panel.** The frozen controlled-network provider panel: 40 unique held-out cells, nine provider laws (one reference and $`M=8`$ candidates), 48 split/cell occurrences, 1,728 unordered candidate pairs (1,584 on smooth objectives), and 432 candidate-versus-reference rows (396 smooth, the rows on which the oracle-margin analysis of [Check 8.15](../experiments/08-09_first_order_version_margin_criterion.md#c8-15) was run).
- **Gold.** The exact eight-update outcome of each pair: one candidate boundary followed by all 128 seven-update reference histories. It is recomputed after measurement and agrees with the frozen records to $`1.1\times10^{-16}`$.
- **Replacements.** The actual one-round optimizer increment is executed for each of the ten support atoms. In replica $`\ell`$ one fresh seven-update reference tail $`\omega_\ell`$ is drawn and shared by all candidates; each updated atom state is continued under it, and terminal outcomes are weighted by each provider's exact law. The integration over atoms happens after the nonlinear tail, so the estimand is $`\tilde L(P^j)-\tilde L(P^k)`$ of [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6), not a mean-update approximation.
- **Replicas.** $`r=256`$ independent tails; $`r=64`$ uses the first 64. Streams `SeedSequence([160160,66,warmup_step,recipe_id,0])`.
- **Surrogates.**
  - *Aligned:* the bounded task utility after the full seven-update reference tail; $`\alpha=1`$, $`\varepsilon_{\rm sur}=0`$.
  - *Short:* the same utility after one reference update. Following Remark 8.11, it is calibrated before any held-out measurement on eight separate calibration cells (288 pairs) that share no complete learner state with the held-out cells: a positive zero-intercept least-squares fit of exact gold differences on measured short differences gives $`\alpha`$, and $`\varepsilon_{\rm sur}`$ is the largest calibration residual divided by $`\alpha`$.
- **Radius.** With $`\delta=0.05`$ simultaneous over pairs at one state, $`\log(4M^2/\delta)=8.5409`$ and $`\hat V`$ the unbiased sample variance of $`Z`$,

```math
e_{\rm stat}=\sqrt{\frac{2\hat V\log(4M^2/\delta)}{r}}+\frac{7B\log(4M^2/\delta)}{3(r-1)},\qquad \text{certified if }\ \lvert\hat S\rvert>e_{\rm stat}+\varepsilon_{\rm sur}.
```

- **Range.** All utilities lie in $`[0,1]`$, so every difference lies in $`[-1,1]`$: the a-priori range $`B=2`$ holds for every draw, state and pair. As a diagnostic only, the same formula is also evaluated with the observed range $`B=\max Z-\min Z`$, which is not a bound on the population support.
- **Comparators.** Separately executed independent tails per candidate, for the CRN variance ratio. For cost, the full-tail paired audit at the Hoeffding rate of 1,016 replicas per candidate ($`M=4`$, $`\delta=0.05`$, $`e=0.1`$), scaled to all executed panel candidates: $`40\times9\times1016=365{,}760`$ trajectories. The panel actually has $`M=8`$, whose Hoeffding rate is 1,154, so this comparison favours the audit.
- **Eligibility.** All pairs are eligible, including the 144 threshold-task pairs with discontinuous return, because Theorem 8.8 needs no smoothness.

**Result — calibration** (`theory/v016_checks/results_v016.json`, check `v016:C6b`, `data[0].calibration_by_r`):

| $`r`$ | $`\alpha`$ | $`\varepsilon_{\rm sur}`$ | largest calibration residual |
|---:|---:|---:|---:|
| 64 | 0.881 | 0.00834 | 0.00735 |
| 256 | 0.886 | 0.00830 | 0.00735 |

**Result — certificates** (check `v016:C6b`, `data[1..8]`; 1,728 pairs; "observed" rows are diagnostics):

| $`r`$ | surrogate | range | certified | wrong certified | certified on the 396 smooth reference rows | large-gold pairs missed ($`\lvert{\rm gold}\rvert\ge0.01`$) |
|---:|---|---|---:|---:|---:|---:|
| 64 | aligned | a-priori $`B=2`$ | 0 | 0 | 0 | 374 / 374 |
| 256 | aligned | a-priori $`B=2`$ | 0 | 0 | 0 | 374 / 374 |
| 64 | short | a-priori $`B=2`$ | 0 | 0 | 0 | 374 / 374 |
| 256 | short | a-priori $`B=2`$ | 8 (0.46%) | 0 | 0 | 366 / 374 |
| 64 | aligned | observed (diagnostic) | 784 (45.4%) | 0 | 157 | 282 / 374 |
| 256 | aligned | observed (diagnostic) | 989 (57.2%) | 0 | 206 | 166 / 374 |
| 64 | short | observed (diagnostic) | 398 (23.0%) | 22 | 90 (20 wrong) | 150 / 374 |
| 256 | short | observed (diagnostic) | 488 (28.2%) | 28 | 102 (22 wrong) | 137 / 374 |

For comparison, the oracle first-order margin of [Check 8.15](../experiments/08-09_first_order_version_margin_criterion.md#c8-15) covers 56 of the same 396 rows, and the measured first-order certificate of [Check 8.16](../experiments/08-09_first_order_version_margin_criterion.md#c8-16) covers 0.

**Result — radius components** (medians over the 1,728 pairs, computed from the raw companion `results_v016_raw.json.gz`):

| $`r`$ | surrogate | $`\lvert{\rm gold}\rvert`$ | $`\lvert\hat S\rvert`$ | variance term | range term, $`B=2`$ | range term, observed $`B`$ | $`\varepsilon_{\rm sur}`$ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | aligned | 0.0037 | 0.0037 | 0.0011 | 0.633 | 0.0028 | 0 |
| 256 | aligned | 0.0037 | 0.0036 | 0.00055 | 0.156 | 0.00087 | 0 |
| 64 | short | 0.0037 | 0.0038 | 0.00035 | 0.633 | 0.00043 | 0.0083 |
| 256 | short | 0.0037 | 0.0037 | 0.00017 | 0.156 | 0.00011 | 0.0083 |

**Result — ranking by the point estimate** (check `v016:C5b`, `data[0].overall.paired_replacement`, aligned, $`r=256`$): 48 wrong signs among 1,687 nonzero-gold pairs (2.8%) and no ties. On the same pairs the first-order rankers are wrong 11.4%–16.4% of the time and the LESS-style cosine 28.8%; the breakdown by optimizer configuration and by second-moment-channel share is in [§7 Experiments](../statements/07-05_where_first_order_rankers_fail.md). The short surrogate's point estimate is wrong on 192 (11.4%, $`r=64`$) and 182 (10.8%, $`r=256`$) of the 1,687 pairs.

**Result — variance reduction by CRN** (`variance`):

| $`r`$ | surrogate | Var(CRN)/Var(independent), summed | median pair ratio | pairs with reduced variance |
|---:|---|---:|---:|---:|
| 64 | aligned | 0.120 | 0.0035 | 99.3% of 1,691 |
| 256 | aligned | 0.143 | 0.0036 | 98.8% of 1,691 |
| 64 | short | 0.148 | 0.0154 | 90.3% of 1,584 |
| 256 | short | 0.147 | 0.0148 | 90.0% of 1,584 |

Pairs whose independent-tail variance is exactly zero have no defined ratio (37 aligned, 144 short) and are counted separately.

**Result — cost** (check `v016:C6b`, `data[0].costs.by_r_surrogate`; reference 365,760 full-tail trajectories):

| $`r`$ | surrogate | trajectories | fraction of full-tail audit | fraction in tail updates |
|---:|---|---:|---:|---:|
| 64 | aligned | 25,600 | 7.0% | 7.0% |
| 64 | short (incl. calibration) | 40,960 | 11.2% | 4.0% |
| 256 | aligned | 102,400 | 28.0% | 28.0% |
| 256 | short (incl. calibration) | 133,120 | 36.4% | 7.6% |

Support sharing executes 10 atom trajectories per replica for 9 candidate values (14 without sharing). The independent-tail variance comparator and the held-out gold audits are diagnostics and are excluded. The audit has a different accuracy criterion, so these ratios alone do not establish superiority.

**Why it supports the statement.**

- **Theorem 8.8.** No strict certificate is wrong at either $`r`$, for either surrogate, on unordered pairs or reference rows. The estimand is exact in expectation, by (a): the point estimate ranks 97.2% of nonzero-gold pairs correctly, with no gradient, curvature or smoothness assumption, including on the threshold task.
- **Proposition 8.10(b), (c).**
  - With $`M=8`$ and $`\delta=0.05`$ the range term is $`7B\cdot8.54/(3(r-1))`$: 0.633 at $`r=64`$ and 0.156 at $`r=256`$ for $`B=2`$. This is two orders of magnitude above the typical gap.
  - The variance term is already below the gap at $`r=64`$.
  - CRN reduces the variance by roughly an order of magnitude in aggregate, and by more than two orders of magnitude for the median aligned pair.
- **Remark 8.12, first problem.** The bottleneck of the strict certificate is the a-priori range, not the variance. With the observed range the aligned surrogate would certify 45–57% of pairs without error, and 157–206 of the 396 reference rows against the oracle margin's 56. The observed range is not a proven bound. A provable range of that size, for example from a pathwise Lipschitz bound, is what is missing.
- **Remark 8.12, second problem; Remark 8.11.** The calibrated short surrogate certifies wrongly 22–28 times under the observed range. Its calibration envelope, fitted on calibration states, does not transfer to every held-out state and recipe; half of the wrong certificates fall in the held-out learning-rate split (11 of 22 at $`r=64`$, 14 of 28 at $`r=256`$). Calibration transfer is therefore an assumption that must be tested, as Remark 8.11 states.

The certificate results are panels B–D of [Figure 8.5](../experiments/08-09_first_order_version_margin_criterion.md#c8-16).

![Figure 8.5: Measurable certificates on the controlled panel](../figures/fig8_5_certificate.png)

*Figure 8.5.* First-order certificate ([Check 8.16](../experiments/08-09_first_order_version_margin_criterion.md#c8-16)): the median radius components $`e_{\rm stat}`$, $`e_{\rm fd}`$ and $`e_2`$ against the median true gap; the remainder alone is comparable to the gap, so nothing is certified. Finite CRN paired replacement ([Check 8.14](../experiments/08-08_crn_paired_replacement_certificate.md#c8-14)): the certified fraction under the a-priori range $`B=2`$ (strict) and under the observed range (diagnostic only), with wrong certificates, for both surrogates and $`r\in\lbrace64,256\rbrace`$. The remaining panels show the distribution of the CRN variance ratio and the trajectory cost relative to the full-tail paired audit. Controlled small network. [PDF](../figures/fig8_5_certificate.pdf)

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C6b,C5b --output theory/v016_checks/subset.json
```

## Artifacts

- `theory/v016_checks/results_v016.json`: checks `v016:C6b`, `v016:C5b`; raw companion `results_v016_raw.json.gz` (per-pair scores, radius components, replica outcomes; not committed); specification `theory/v016_checks/SPEC.md`; code `c5c6b.py`.
- Figure: `../figures/fig8_5_certificate` (panels B–D).
