# Theorem 4.5 · A round-independent bound and averaging rates

*Experiments*

[← Framework](../README.md) · [Statement](../statements/04-05_round_independent_bound_averaging_rates.md) · [Proof](../proofs/04-05_round_independent_bound_averaging_rates.md) · [Experiments](04-05_round_independent_bound_averaging_rates.md) · [中文](04-05_round_independent_bound_averaging_rates_ZH.md)

**What is tested here.** The round-independent bound (a) is attacked by adversarial search; the cumulant expansion (b), the exact scalar law (c) and the unequal-means limit (d) are computed exactly as polynomials in $`1/n`$; and the claim that replay and deterministic descendants do not decay (e) is tested against independent blocks.

- **Verdict: PASS.** No adversarial instance exceeds the bound; the closest has gap/bound $`55/57`$, and gap and bound are constant in $`R`$ from 1 to 50 ([Check 4.6](#c4-6)).
- The first unmatched cumulant gives both the exponent and the coefficient of the decay in all six cases, a vanishing contraction removes the leading term, and unequal means converge to $`J(\mu_P)-J(\mu_Q)`$ at rate $`O(1/n)`$ ([Check 4.7](#c4-7)).
- Replay and deterministic descendants leave the gap constant through 256 exposures, while independent blocks shrink it as $`n^{1-N}`$ ([Check 4.8](#c4-8)).

**Setting used on this page.**

- Student $`p\in[0,1]^k`$, deployments with at most $`N`$ decisions and polynomial value $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$; IID blocks $`Z_i\in[0,1]^k`$ from $`P`$ or $`Q`$; convex-affine learner $`p_R=w_0p_0+\sum_iw_iZ_i`$ ([Definition 4.3](../statements/04-03_binary_context_student_affine_learner.md#s4-3)).
- $`m_P(\alpha)=\mathbb E_PZ^\alpha`$; $`\delta_N(P,Q)=\max_{1\le\lvert\alpha\rvert\le N}\lvert m_P(\alpha)-m_Q(\alpha)\rvert`$ is the largest moment mismatch; $`C_J=\sum_\alpha\lvert c_\alpha\rvert\lvert\alpha\rvert`$.
- The *gap* of a learner is $`\mathbb E_PJ(p_R)-\mathbb E_QJ(p_R)`$. The averaging learner is $`\bar p_n=\frac1n\sum_{i=1}^nZ_i`$, with mean $`\mu`$; $`\kappa_r`$ is the order-$`r`$ joint cumulant and $`D^rJ(\mu)`$ the $`r`$-th derivative tensor at $`\mu`$.
- *Scalar witnesses* $`P_N,Q_N`$: binomial weights $`\binom Ni/2^{N-1}`$ on the even versus the odd points of $`\lbrace0,\tfrac1N,\dots,1\rbrace`$; they agree below order $`N`$ and differ at order $`N`$ by $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$ ([Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order.md#s4-4)). *Parity laws*: uniform on the even- versus odd-parity vertices of $`\lbrace0,1\rbrace^N`$.
- Code IDs: `v014:` = `theory/independent_checks` (results `results_independent_v014.json`); `pkg:` = package verifier re-run into `theory/v014_repro/results_v014_rerun.json`.

**Statement under test ([Theorem 4.5](../statements/04-05_round_independent_bound_averaging_rates.md#s4-5)).**

- **(a)** For every convex-affine learner, $`\lvert\mathbb E_{P^R}J(p_R)-\mathbb E_{Q^R}J(p_R)\rvert\le C_J\,\delta_N(P,Q)\le N3^N\delta_N(P,Q)`$, independently of $`R`$.
- **(b)** If $`P,Q`$ share the mean $`\mu`$ and all joint cumulants of order below $`r`$ ($`2\le r\le N`$), then $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=\langle D^rJ(\mu),\kappa_r(P)-\kappa_r(Q)\rangle/(r!\,n^{r-1})+O(n^{-r})`$; if the contraction vanishes the difference is $`O(n^{-r})`$.
- **(c)** For the scalar witnesses and $`J(p)=p^N`$, the gap is exactly $`\Delta_N/n^{N-1}`$ for every $`n\ge1`$.
- **(d)** If $`\mu_P\ne\mu_Q`$, the gap is $`J(\mu_P)-J(\mu_Q)+O(n^{-1})`$.
- **(e)** If the $`n`$ training blocks are $`n`$ copies of $`T(Z_1)`$ for one parent block and a fixed deterministic map $`T`$ (replay is $`T=\mathrm{id}`$), the gap equals $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$ for every $`n`$.

<a id="c4-6"></a>

## Check 4.6 · A discrepancy bound that does not grow with the number of updates (Theorem 4.5(a))

**Verdict.** PASS on 288 adversarially searched instances (582 assertions, worst residual 0). The bound $`C_J\delta_N`$ is never exceeded; the tightest instance has gap/bound $`55/57\approx0.965`$. On a fixed problem the gap and the bound are both constant as $`R`$ grows from 1 to 50.

**Why.** [Theorem 4.5](../statements/04-05_round_independent_bound_averaging_rates.md#s4-5)(a) bounds the learning-value gap of every convex-affine learner by $`C_J\delta_N\le N3^N\delta_N`$, uniformly in the number of updates $`R`$. It is falsified by an instance above the bound, or by a family whose worst ratio grows with $`R`$.

**Setup.**

- 24 search trials, $`k\in\lbrace 1,2\rbrace`$, $`N\in\lbrace 1,2,3\rbrace`$, support $`\lbrace 0,\tfrac12,1\rbrace ^k`$.
- $`R`$ cycles through $`\lbrace 1,2,5,10,25,50\rbrace`$.
- Random laws $`P,Q`$, random convex weights $`w_0,\dots,w_R`$, a random $`p_0`$, and a random task tree whose exact polynomial gives $`C_J`$.
- Seed: `SeedSequence([20261010, 10])`.

**Procedure.**

1. In each trial, run a 12-step rational hill-climb on $`Q`$ (moving mass $`\le\tfrac1{20}`$ between support points) to maximize gap/bound: 288 instances.
2. Compute each learning value exactly by independent-block moment convolution.
3. Assert gap $`\le C_J\delta_N\le N3^N\delta_N`$.
4. *Fixed-problem sweep.* $`P=(\tfrac14,\tfrac34)`$ and $`Q=(\tfrac34,\tfrac14)`$ on $`\lbrace 0,1\rbrace`$, $`J=p^2`$, uniform average of $`R`$ blocks, $`R=1,\dots,50`$.

**Result.**

| Quantity | Value | Source |
|---|---:|---|
| Instances / assertions | 288 / 582 | entry `v014:C10`, `n_instances`, `.n_assertions` |
| Largest gap/bound found | $`55/57\approx0.9649`$ at $`N=1`$, $`R=10`$ | entry `v014:C10`, `details.best_exact_ratio`, `.best_instance` |
| Status | PASS (worst residual 0) | entry `v014:C10`, `status` |

| Fixed problem, $`R`$ | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| exact gap | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| bound $`C_J\delta_N`$ | 1 | 1 | 1 | 1 | 1 | 1 |

Source: entry `v014:C10`, `details.fixed_problem_R_sweep`.

**Why it supports Theorem 4.5(a).**

- No adversarial instance exceeds the bound, and the bound comes within 4% of being attained.
- The number of updates never enters: in the sweep, gap and bound are constant in $`R`$.

<a id="c4-7"></a>

## Check 4.7 · The first unmatched cumulant sets the rate (Theorem 4.5(b)–(d))

**Verdict.** PASS (1,530 instances, 1,797 assertions). The exact scalar law holds in all 210 cells $`N=2,\dots,8`$, $`n=1,\dots,30`$. In all six cumulant cases the exact gap starts at $`n^{1-r}`$ with the predicted coefficient and has no slower term; a vanishing contraction removes the leading term; unequal means converge to $`J(\mu_P)-J(\mu_Q)=13/32`$ at rate $`O(1/n)`$. A cumulant-free implementation reproduces the exact law in 20 cases.

**Why.** [Theorem 4.5](../statements/04-05_round_independent_bound_averaging_rates.md#s4-5)(b) predicts both the exponent $`1-r`$ and the coefficient $`\langle D^rJ(\mu),\Delta\kappa_r\rangle/r!`$ of the decay in the number $`n`$ of independent blocks. It is falsified by a wrong leading coefficient, by a slower term $`n^{-j}`$ with $`j<r-1`$, by a residual that is not $`O(n^{-r})`$, or by a nonzero leading term when the contraction vanishes. Part (c) is the exact law $`\Delta_N/n^{N-1}`$ for the scalar witnesses, and part (d) gives the limit $`J(\mu_P)-J(\mu_Q)`$ when the means differ.

**Setup.**

- *Exact law, part (c).* The scalar witness laws of [Check 4.4](04-04_affine_learners_moment_order.md#c4-4) with $`J=p^N`$, $`N=2,\dots,8`$, $`n=1,\dots,30`$.
- *Cumulant cases*, six of them, $`k\in\lbrace 1,2\rbrace`$, $`r\in\lbrace 2,3,4\rbrace`$:
  - $`k=1`$: the scalar witness laws of order $`r`$;
  - $`k=2`$: the bivariate lift $`(x,(x+b)/2)`$ with $`b\in\lbrace 0,1\rbrace`$ equally likely;
  - task $`J=(\tfrac1k\sum_jp_j)^{r+1}`$.
- *Vanishing contraction.* $`J=(p-\tfrac12)^{r+2}`$ for $`r=2,3,4`$; its $`r`$-th derivative at $`\mu=\tfrac12`$ is zero.
- *Unequal means.* $`P=(\tfrac14,\tfrac34)`$, $`Q=(\tfrac34,\tfrac14)`$ on $`\lbrace 0,1\rbrace`$, $`J=p^3`$.

**Procedure.**

1. Compute $`\mathbb EJ(\bar p_n)`$ exactly as a polynomial in $`1/n`$, by sample-index multinomial expansion.
2. Compute the cumulants independently, from the moments.
3. Assert that lower cumulants match, that the coefficient of $`n^{1-r}`$ equals the predicted contraction, and that all slower coefficients are zero.
4. Assert that $`n\cdot\lvert n^{r-1}\text{gap}-\text{coefficient}\rvert`$ stays below the envelope of higher coefficients for $`n=1,\dots,220`$.
5. *Independent implementation* (`pkg:scalar_exact_decay`): convolve the laws $`n`$ times and evaluate $`\mathbb E\bar p_n^N`$ directly, never using cumulants, for $`N=2,\dots,6`$ and $`n\in\lbrace 1,2,4,8\rbrace`$.

**Result.**

| Case $`(k,r)`$ | predicted coefficient | exact gap as a polynomial in $`1/n`$ |
|---|---:|---|
| (1, 2) | $`3/8`$ | $`\tfrac38\,n^{-1}`$ |
| (1, 3) | $`-1/9`$ | $`-\tfrac19\,n^{-2}`$ |
| (1, 4) | $`15/512`$ | $`\tfrac{15}{512}\,n^{-3}`$ |
| (2, 2) | $`27/128`$ | $`\tfrac{27}{128}\,n^{-1}`$ |
| (2, 3) | $`-3/64`$ | $`-\tfrac3{64}\,n^{-2}`$ |
| (2, 4) | $`1215/131072`$ | $`\tfrac{1215}{131072}\,n^{-3}`$ |
| vanishing, $`r=2`$ | 0 | $`\tfrac3{16}n^{-2}-\tfrac18n^{-3}`$ |
| vanishing, $`r=3`$ | 0 | $`-\tfrac5{108}n^{-3}+\tfrac5{162}n^{-4}`$ |
| vanishing, $`r=4`$ | 0 | $`\tfrac{45}{4096}n^{-4}-\tfrac{15}{2048}n^{-5}`$ |
| unequal means | limit $`13/32`$ | $`\tfrac{13}{32}+\tfrac9{32}n^{-1}-\tfrac3{16}n^{-2}`$ |

- Sources: entry `v014:C11`, `details.cumulant_cases[*].predicted_coefficient`, `.exact_gap_polynomial_in_inverse_n`, `.gap` (vanishing cases), entry `v014:C11`, `details.unequal_means_gap`, `.unequal_means_limit`.
- The exact law of part (c) holds in all $`7\times30=210`$ cells $`(N,n)`$; status PASS.
- Independent implementation: all 20 enumerated gaps equal $`\Delta_N/n^{N-1}`$ exactly, e.g. $`N=4`$ gives $`3/256,\ 3/2048,\ 3/16384,\ 3/131072`$ (`results_v014_rerun.json → scalar_exact_decay`; fields `enumerated_gap_exact` and `predicted_gap_exact` identical in every row).

**Why it supports Theorem 4.5(b)–(d).**

- In every case the gap begins exactly at $`n^{1-r}`$ with the predicted coefficient: exponent and coefficient are both set by the first unmatched cumulant.
- A zero contraction makes the decay faster ($`n^{-r}`$), as the expansion predicts.
- With unequal means the limit $`J(\mu_P)-J(\mu_Q)`$ is approached at $`O(1/n)`$.

<a id="c4-8"></a>

## Check 4.8 · Independent evidence versus replay versus synthetic descendants (Theorem 4.5(e))

**Verdict.** PASS (45 instances, 140 assertions). With independent blocks the gap decays exactly as $`\text{gap}_1/n^{N-1}`$ (log-log slopes $`-1,-2,-3`$ for $`N=2,3,4`$). Replaying one block, or training on deterministic descendants of one parent block, leaves the gap constant through 256 exposures.

**Why.** The rates of [Theorem 4.5](../statements/04-05_round_independent_bound_averaging_rates.md#s4-5)(b)–(c) are in the number of independent blocks. Part (e) predicts that two kinds of exposure without new evidence leave the gap at its one-block value: replaying one block $`n`$ times, and training on $`n`$ deterministic descendants of one parent block. This separates information from exposure.

**Setup.**

- Scalar witness laws with $`N=2,3,4`$, and parity laws with $`N=2,3`$.
- Exposure $`n\in\lbrace 1,2,4,\dots,256\rbrace`$.
- Descendant channel $`T(Z)=(Z+\tfrac12)/2`$, a fixed information-preserving map.
- Seed: `SeedSequence([20261010, 18])`.

**Procedure.** For each family, $`N`$ and $`n`$, compute three gaps exactly:

- independent blocks, $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)`$;
- replay, $`n`$ copies of one block, so $`\bar p_n=Z_1`$;
- descendants, $`n`$ copies of $`T(Z_1)`$, so $`\bar p_n=T(Z_1)`$.

Then assert that the independent gap equals the one-block gap divided by $`n^{N-1}`$, that the replay gap equals the one-block gap, and that the descendant gap equals the one-block gap divided by $`2^N`$ (the contraction of this channel).

**Result.**

| Family, $`N`$ | gap, $`n=1`$ | independent, $`n=256`$ | replay, $`n=256`$ | descendants, $`n=256`$ | independent log-log slope |
|---|---:|---:|---:|---:|---:|
| scalar, 2 | $`1/4`$ | $`1/1024`$ | $`1/4`$ | $`1/16`$ | $`-1`$ |
| scalar, 3 | $`-1/18`$ | $`-1/1179648`$ | $`-1/18`$ | $`-1/144`$ | $`-2`$ |
| scalar, 4 | $`3/256`$ | $`3/4294967296`$ | $`3/256`$ | $`3/4096`$ | $`-3`$ |
| parity, 2 | $`1`$ | $`1/256`$ | $`1`$ | $`1/4`$ | $`-1`$ |
| parity, 3 | $`1`$ | $`1/65536`$ | $`1`$ | $`1/8`$ | $`-2`$ |

Source: entry `v014:C18`, `details.curves`; slopes from the $`n=1`$ and $`n=256`$ rows.

![Figure 4.3: only independent blocks make the discrepancy decay](../figures/fig4_3_rates.png)

*Figure 4.3.* Independent blocks shrink the gap as $`n^{1-N}`$, while replaying one block or training on its descendants leaves it constant (this check). Rescaled by $`n^{r-1}`$, the exact gap converges to the first-unmatched-cumulant coefficient in all six cases, and a zero contraction removes the leading term ([Check 4.7](#c4-7)). The round-independent bound holds for $`R`$ up to 50 and is nearly attained ([Check 4.6](#c4-6)). [PDF](../figures/fig4_3_rates.pdf)

**Why it supports Theorem 4.5(e).**

- Only independent evidence makes the discrepancy decay, at exactly the predicted exponent $`1-N`$.
- Replay and descendants raise training exposure 256-fold and leave the learning-value gap unchanged; descendants rescale it once through the channel and never decay.

**Limits.** The channel is the stated deterministic map. It is a constructed control, not a claim about every possible post-processing.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C10,C11,C18 \
  --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py \
  --output theory/v014_repro/results_v014_rerun.json
```

`--only` preserves each claim's random stream (`SeedSequence([20261010, claim number])`), so a subset run reproduces the full-run numbers.

## Artifacts

- Results: `theory/independent_checks/results_independent_v014.json` (entries `v014:C10`, `v014:C11`, `v014:C18`); `theory/v014_repro/results_v014_rerun.json` (`scalar_exact_decay`).
- Code: `theory/independent_checks/check_v014_independent.py` (`check_C10`, `check_C11`, `check_C18`); `docs/direction/v014_handoff/verify_environment_information_v014.py`; specification `theory/independent_checks/SPEC.md`.
- Existing plot: `theory/independent_checks/figs/C18_evidence_exposure.png`. Figure spec: `theoretical_framework/figures/specs/fig4_3.md`.
