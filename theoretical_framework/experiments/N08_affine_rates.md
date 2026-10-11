# N08 · Rates in the affine theory — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N08_affine_rates.md) · [中文](N08_affine_rates_ZH.md)

**Verdict.** N08 holds; every assertion below passed, all in exact rational arithmetic.

- **(a)** **(C10).** The bound $`C_J\delta_N`$ holds on 288 adversarially searched instances with up to $`R=50`$ updates. The tightest ratio is gap/bound $`=55/57\approx0.965`$. On a fixed problem the bound stays at 1 while $`R`$ grows from 1 to 50.
- **(b)** **(C11).**
  - The averaging identity [N08(c)](../proofs/N08_affine_rates.md) is exact for $`N=2,\dots,8`$ and every $`n=1,\dots,30`$.
  - In six scalar and bivariate cases with first unmatched cumulant of order $`r=2,3,4`$, the exact gap polynomial in $`1/n`$ has leading coefficient equal to the predicted tensor contraction, with no slower term.
  - When the contraction vanishes, the leading term disappears.
  - With unequal means the gap converges to $`J(\mu_P)-J(\mu_Q)=13/32`$ at rate $`O(1/n)`$.
- **(c)** **Package check.** An independent check by direct convolution reproduces [N08(c)](../proofs/N08_affine_rates.md) in 20 cases.
- **Replay and descendants (C18).** With independent blocks the gap decays exactly as $`\text{gap}_1/n^{N-1}`$; the log-log slopes are $`-1,-2,-3`$ for $`N=2,3,4`$. Replaying one block or using one parent's descendants leaves the gap constant through 256 exposures.

![Independent blocks drive the decay; replay does not](../figures/N08_rates.png)

[PDF](../figures/N08_rates.pdf)

### C10 — [N08(a)](../proofs/N08_affine_rates.md): a discrepancy bound that does not grow with R

**Why.** [N08(a)](../proofs/N08_affine_rates.md) claims a bias bound for *every* convex affine learner, uniform in the number of updates $`R`$. It would be falsified by an instance with gap $`>C_J\delta_N`$, or by a family whose worst ratio grows with $`R`$.

**Setup.**

- 24 search trials with $`k\in\lbrace 1,2\rbrace`$ and $`N\in\lbrace 1,2,3\rbrace`$, on the support $`\lbrace 0,\tfrac12,1\rbrace ^k`$.
- $`R`$ cycles through $`\lbrace 1,2,5,10,25,50\rbrace`$.
- Random laws $`P,Q`$, random convex weights $`w_0,\dots,w_R`$, a random $`p_0`$, and a random task tree whose exact polynomial gives $`C_J`$.
- Seed: `SeedSequence([20261010, 10])`.

**Procedure.**

1. In each trial, run a 12-step rational hill-climb on $`Q`$, moving mass $`\le\tfrac1{20}`$ between support points, to *maximize* gap/bound. This gives 288 instances.
2. Compute each learning value exactly by independent-block moment convolution.
3. Assert gap $`\le C_J\delta_N\le N3^N\delta_N`$.
4. *Fixed-problem sweep.* Take $`P=(\tfrac14,\tfrac34)`$ and $`Q=(\tfrac34,\tfrac14)`$ on $`\lbrace 0,1\rbrace`$, $`J=p^2`$, and the uniform average of $`R`$ blocks, for $`R=1,\dots,50`$.

**Result.**

| Quantity | Value | Source |
|---|---:|---|
| Instances / assertions | 288 / 582 | `claims[id=C10].n_instances`, `.n_assertions` |
| Largest gap/bound found | $`55/57\approx0.9649`$ at $`N=1`$, $`R=10`$ | `claims[id=C10].details.best_exact_ratio`, `.best_instance` |
| Status | PASS (worst residual 0) | `claims[id=C10].status` |

| Fixed problem, $`R`$ | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| exact gap | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| bound $`C_J\delta_N`$ | 1 | 1 | 1 | 1 | 1 | 1 |

Source: `claims[id=C10].details.fixed_problem_R_sweep`.

**What this shows.**

- No adversarial search exceeds the bound, and the bound comes within 4% of being attained.
- The bound holds uniformly in $`R`$: the sweep's gap and bound are both constant in $`R`$, so the number of updates never enters.

### C11 — [N08(b)](../proofs/N08_affine_rates.md) and [N08(c)](../proofs/N08_affine_rates.md): the first unmatched cumulant sets the rate

**Why.** [N08(b)](../proofs/N08_affine_rates.md) predicts both the *exponent* $`1-r`$ and the *coefficient* $`\langle D^rJ(\mu),\Delta\kappa_r\rangle/r!`$ of the gap decay in the number of independent blocks. It would be falsified by:

- a wrong leading coefficient;
- a slower term $`n^{-j}`$ with $`j<r-1`$;
- a residual that is not $`O(n^{-r})`$;
- a nonzero leading term when the contraction vanishes.

**Setup.**

- *Exact law [N08(c)](../proofs/N08_affine_rates.md).* The scalar laws [N07(c)](../proofs/N07_affine_moment_order.md) with $`J=p^N`$, for $`N=2,\dots,8`$ and $`n=1,\dots,30`$.
- *Cumulant cases.* Six cases, $`k\in\lbrace 1,2\rbrace`$ and $`r\in\lbrace 2,3,4\rbrace`$:
  - $`k=1`$: the scalar laws of order $`r`$;
  - $`k=2`$: the bivariate lift $`(x,(x+b)/2)`$ with $`b\in\lbrace 0,1\rbrace`$ equally likely;
  - the task is $`J=(\tfrac1k\sum_jp_j)^{r+1}`$.
- *Vanishing contraction.* $`J=(p-\tfrac12)^{r+2}`$ for $`r=2,3,4`$, whose $`r`$-th derivative at $`\mu=\tfrac12`$ is zero.
- *Unequal means.* $`P=(\tfrac14,\tfrac34)`$ and $`Q=(\tfrac34,\tfrac14)`$ on $`\lbrace 0,1\rbrace`$, with $`J=p^3`$.

**Procedure.**

1. Compute $`\mathbb EJ(\bar p_n)`$ *exactly* as a polynomial in $`1/n`$, by sample-index multinomial expansion.
2. Compute the cumulants *independently*, from the moments.
3. Assert that lower cumulants match, that the coefficient of $`n^{1-r}`$ equals the predicted contraction, and that all slower coefficients are zero.
4. Assert that $`n\cdot\lvert n^{r-1}\text{gap}-\text{coefficient}\rvert`$ stays below the envelope of higher coefficients for $`n=1,\dots,220`$.

**Result.**

| Case $`(k,r)`$ | predicted coefficient | exact gap as polynomial in $`1/n`$ |
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

Sources: `claims[id=C11].details.cumulant_cases[*].predicted_coefficient`, `.exact_gap_polynomial_in_inverse_n`, `.gap` (vanishing cases), and `claims[id=C11].details.unequal_means_gap`, `.unequal_means_limit`. Identity [N08(c)](../proofs/N08_affine_rates.md) holds exactly in all $`7\times30=210`$ $`(N,n)`$ cells. In total there are 1,530 instances and 1,797 assertions (`claims[id=C11].n_instances`, `.n_assertions`); status PASS.

**What this shows.**

- In every case, the gap begins exactly at $`n^{1-r}`$ with the predicted coefficient.
- In the vanishing cases, a zero contraction makes the decay faster ($`n^{-r}`$).
- With unequal means, the limit $`J(\mu_P)-J(\mu_Q)`$ is approached at $`O(1/n)`$.
- Exponent and coefficient are both determined by the first unmatched cumulant.

### Package check — exact decay by direct convolution

**Why.** This is an independent implementation that never uses cumulants. It convolves the laws $`n`$ times and evaluates $`\mathbb E\bar p_n^N`$ directly.

**Result.** For $`N=2,\dots,6`$ and $`n\in\lbrace 1,2,4,8\rbrace`$, all 20 enumerated gaps equal the prediction $`\Delta_N/n^{N-1}`$ exactly. For example, $`N=4`$ gives $`3/256,\ 3/2048,\ 3/16384,\ 3/131072`$. Source: `theory/v014_repro/results_v014_rerun.json → scalar_exact_decay`. The fields `enumerated_gap_exact` and `predicted_gap_exact` are identical in every row, and the file is identical to the package's `docs/direction/v014_handoff/results_v014.json`.

### C18 — independent evidence versus replay versus synthetic descendants

**Why.** The rate in [N08(b)](../proofs/N08_affine_rates.md) is in the number of *independent* blocks. [N08(c′)](../proofs/N08_affine_rates.md) predicts that two kinds of training exposure without new evidence leave the gap at its one-block value:

- replaying one block $`n`$ times;
- training on $`n`$ deterministic descendants of one parent block.

This separates information from exposure: more epochs are not more data.

**Setup.**

- Scalar laws [N07(c)](../proofs/N07_affine_moment_order.md) with $`N=2,3,4`$, and parity laws with $`N=2,3`$.
- Exposure $`n\in\lbrace 1,2,4,\dots,256\rbrace`$.
- The descendant channel is $`Z\mapsto(Z+\tfrac12)/2`$, a fixed information-preserving map.
- Seed: `SeedSequence([20261010, 18])`.

**Procedure.** For each family, $`N`$ and $`n`$, compute three gaps exactly:

- independent blocks, $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)`$;
- replay, where $`n`$ copies of one block give $`\bar p_n=Z_1`$;
- descendants, where $`n`$ copies of $`T(Z_1)`$ give $`\bar p_n=T(Z_1)`$.

Then assert three identities:

- the independent gap equals the one-block gap divided by $`n^{N-1}`$;
- the replay gap equals the one-block gap;
- the descendant gap equals the one-block gap divided by $`2^N`$, the contraction of the chosen channel.

**Result.**

| Family, $`N`$ | gap, $`n=1`$ | independent, $`n=256`$ | replay, $`n=256`$ | descendants, $`n=256`$ | independent log-log slope |
|---|---:|---:|---:|---:|---:|
| scalar, 2 | $`1/4`$ | $`1/1024`$ | $`1/4`$ | $`1/16`$ | $`-1`$ |
| scalar, 3 | $`-1/18`$ | $`-1/1179648`$ | $`-1/18`$ | $`-1/144`$ | $`-2`$ |
| scalar, 4 | $`3/256`$ | $`3/4294967296`$ | $`3/256`$ | $`3/4096`$ | $`-3`$ |
| parity, 2 | $`1`$ | $`1/256`$ | $`1`$ | $`1/4`$ | $`-1`$ |
| parity, 3 | $`1`$ | $`1/65536`$ | $`1`$ | $`1/8`$ | $`-2`$ |

Source: `claims[id=C18].details.curves`. The slope is computed from the $`n=1`$ and $`n=256`$ rows. There are 45 instances and 140 assertions; status PASS.

**What this shows.**

- Only independent evidence makes the discrepancy decay, and it decays at exactly the predicted exponent $`1-N`$.
- Replay and descendants increase training exposure $`256`$-fold yet leave the learning-value gap unchanged.
- Descendants rescale the gap once, through the channel, and never decay.
- The scope of the channel result is the stated deterministic channel. It is a constructed control, not a claim about every possible postprocessing.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C10,C11,C18 \
  --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py \
  --output theory/v014_repro/results_v014_rerun.json
```

## Artifacts

- Results:
  - `theory/independent_checks/results_independent_v014.json` (C10, C11, C18)
  - `theory/v014_repro/results_v014_rerun.json` (`scalar_exact_decay`)
- Existing plot: `theory/independent_checks/figs/C18_evidence_exposure.png`
- Code: `theory/independent_checks/check_v014_independent.py` (`check_C10`, `check_C11`, `check_C18`)
- Figure spec: `theoretical_framework/figures/specs/N08.md`
