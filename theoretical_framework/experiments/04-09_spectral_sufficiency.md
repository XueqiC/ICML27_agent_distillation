# Theorem 4.9 · Spectral sufficiency and linear interface complexity

*Experiments*

[← Framework](../README.md) · [Statement](../statements/04-09_spectral_sufficiency.md) · [Proof](../proofs/04-09_spectral_sufficiency.md) · [Experiments](04-09_spectral_sufficiency.md) · [中文](04-09_spectral_sufficiency_ZH.md)

**What is tested here.**

- **(a)–(c), the weighted spectrum.** On a random weighted catalog, the mode identity, the Hilbert–Schmidt bound and the best rank-3 residual all hold to rounding error (**PASS**, [Check 4.14](#c4-14)).
- **(f), probability-feasible lower bound.** For three arbitrary features, two strictly positive laws with equal feature means differ on the tests by $`0.0950`$, above the guaranteed $`0.0804`$ (**PASS**, [Check 4.15](#c4-15)).
- **(d)–(e), what averages and boundedness do not give.** The generic dimension bound demands more modes than the input space has in all 12 settings tried, while a smooth response class needs only 3 modes at the same accuracy; a test with weight $`10^{-6}`$ can carry error 1 under an average error of $`10^{-3}`$; indicator responses on 8 messages have full centred rank 7 (**PASS**, [Check 4.16](#c4-16)).

**Setting used on this page.**

- A finite catalog: $`p`$ tests with weights $`w_i>0`$, $`m`$ messages with reference weights $`q_j>0`$, both summing to 1, and a response matrix $`G\in[0,1]^{p\times m}`$ with $`G_{ij}=g_i(y_j)`$.
- In orthonormal input coordinates $`x=\sqrt q\odot h`$ the response operator of [Definition 4.8](../statements/04-08_response_operator.md#s4-8) is the matrix $`D_{\sqrt w}\,G\,D_{\sqrt q}\,\Pi_q`$ with $`\Pi_q=I-\sqrt q\sqrt q^\top`$; its singular values are the $`\sigma_j`$.
- Mode differences $`\alpha_j=\langle v_j,h\rangle_\rho`$, rank-$`r`$ linear interfaces, and per-test residuals $`e_r(a)`$ as in the statement.

**Statement under test ([Theorem 4.9](../statements/04-09_spectral_sufficiency.md#s4-9)).** (a) $`\lVert\mathcal Rh\rVert_\omega^2=\sum_j\sigma_j^2\alpha_j^2`$; (b)–(c) matching the leading $`r`$ modes leaves at most $`B\sigma_{r+1}`$, and this is optimal among rank-$`r`$ linear interfaces; (d) per-test bounds pay $`e_r(a)`$ and a rare test is not controlled by the average; (e) the generic bound $`r+1\ge B^2/(4\varepsilon^2)`$ has no compression content under the $`J`$-candidate mixture measure; (f) on a finite message set, $`r`$ features leave two probability laws with equal statistics and response gap at least $`a\sigma_{r+1}`$.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 4.14](#c4-14) | (a), (b), (c) | `v017:C2.spectrum` | PASS |
| [Check 4.15](#c4-15) | (f) | `v017:C3.positive` | PASS |
| [Check 4.16](#c4-16) | (d), (e) | `v017:C4.generic`, `v017:C9.rare` | PASS |

<a id="c4-14"></a>

## Check 4.14 · Weighted spectrum, Hilbert–Schmidt bound and best truncation (Theorem 4.9(a)–(c))

**Verdict.** All identities hold to rounding error; the squared Hilbert–Schmidt norm is $`0.044\le\tfrac14`$.

**Why.** The weights $`\omega`$ and $`\rho`$ and the centring enter the operator in a specific way. An error in where $`\sqrt w`$, $`\sqrt q`$ or the projection $`\Pi_q`$ appear would break the mode identity, the density-difference identity of [Definition 4.8](../statements/04-08_response_operator.md#s4-8), or the equality between the best rank-$`r`$ residual and $`\sigma_{r+1}`$.

**Setup.**

- $`p=7`$ tests, $`m=11`$ messages, weights drawn uniformly from $`[0.3,1.3]`$ and normalized, $`G=\sigma(N)`$ with $`N`$ a standard normal matrix and $`\sigma`$ the logistic function.
- A random unit $`x`$ in the centred input space for (a); the truncation residual of the best rank-3 approximation in operator norm for (c); two random laws $`P,Q`$ from a flat Dirichlet law for the density-difference identity.

**Result.**

| Quantity | Value | JSON key (entry `v017:C2.spectrum`) |
|---|---|---|
| error of (4.12) at a random unit input | $`2.8\times10^{-18}`$ | `spectral_identity_error` |
| $`\lvert\lVert\mathcal R-\mathcal R_3\rVert_{\rm op}-\sigma_4\rvert`$ | $`2.8\times10^{-17}`$ | `best_rank_residual_error` |
| $`\lVert\mathcal R\rVert_{\rm HS}^2`$ (bound $`\tfrac14`$) | $`0.04418`$ | `hilbert_schmidt_squared` |
| error of $`(\mathcal Rh)(a)=\mathbb E_Pg_a-\mathbb E_Qg_a`$ | $`1.7\times10^{-17}`$ | `density_difference_error` |
| singular values $`\sigma_1,\dots,\sigma_7`$ | $`0.1179,\ 0.1053,\ 0.0877,\ 0.0805,\ 0.0587,\ 0.0309,\ 0.0249`$ | `singular_values` |

**Why it supports the statement.** The operator built from the weighted, centred catalog has exactly the mode identity of (a), its best rank-3 residual is $`\sigma_4`$ as (b)–(c) state, and applied to a density difference it returns the weighted score differences of the two laws.

<a id="c4-15"></a>

## Check 4.15 · Positive laws that three features cannot separate (Theorem 4.9(f))

**Verdict.** Two strictly positive laws with equal means of three arbitrary features differ on the tests by $`0.0950`$, above the guaranteed $`a\sigma_4=0.0804`$.

**Why.** (c) is proved on a signed ball, which contains differences that no pair of probability laws can produce. (f) claims that the same lower bound is reached by actual laws once the perturbation is scaled to keep them positive.

**Setup.** The catalog of [Check 4.14](#c4-14) regenerated with its own seed; three features with standard normal values; $`v`$ a unit vector in the span of the first four input modes orthogonal to the three features; $`a=1.8/\lVert v\rVert_\infty`$, strictly inside the admissible range $`a\le2/\lVert v\rVert_\infty`$; $`P,Q`$ from (4.17).

**Result.**

| Quantity | Value | JSON key (entry `v017:C3.positive`) |
|---|---|---|
| largest feature-mean difference | $`2.8\times10^{-16}`$ | `feature_match_error` |
| response gap $`\lVert\mathcal R(p-q)\rVert_\omega`$ | $`0.0950`$ | `response_gap` |
| guaranteed $`a\sigma_4`$ | $`0.0804`$ | `guaranteed_gap` |
| smallest probability in $`P,Q`$ | $`0.0109`$ | `smallest_probability` |

**Why it supports the statement.** The construction of (f) yields two genuine probability laws that no interface reading those three features can tell apart, and their response gap exceeds the bound $`a\sigma_{r+1}`$, as stated.

<a id="c4-16"></a>

## Check 4.16 · The generic bound, a structured spectrum, a rare test and full rank (Theorem 4.9(d)–(e))

**Verdict.** In all 12 settings the generic bound asks for more modes than the input space has. A smooth response class reaches error $`0.01`$ with 3 modes where the generic bound asks for 2,500. A rare test carries error 1 under average error $`10^{-3}`$. Indicator responses on 8 messages have centred rank 7.

**Why.**

- (e) states that boundedness alone gives a valid but, under the mixture measure, empty dimension count. The check computes it at the attained worst case $`B^2=2J`$.
- The contrast with a structured class shows that useful compression, when it exists, comes from the measured spectrum.
- (d) states that a per-test guarantee cannot be read from the weighted average, and a full-rank class shows that no compression follows from boundedness and measurability alone.

**Setup.**

- *Generic bound.* $`J\in\lbrace2,5,20,100\rbrace`$ point-mass candidates on distinct messages; $`\rho`$ their uniform mixture; $`h`$ the density difference of two candidates; $`\varepsilon\in\lbrace0.1,0.3,0.7\rbrace`$. Sufficient dimension $`B^2/(4\varepsilon^2)`$ against the available input dimension $`J-1`$.
- *Structured class.* 30 tests and 40 messages on a grid $`y\in[0,1]`$, $`g_k(y)=\sigma(3\sin(2\pi k/30+2y))`$, uniform weights.
- *Rare test and full rank.* A test of weight $`10^{-6}`$ with error 1; 8 messages with indicator responses $`g_a=\mathbf 1\lbrace y=a\rbrace`$.

**Result.**

| Quantity | Value | JSON key |
|---|---|---|
| $`B^2`$ attained | $`2J`$ in every row | `v017:C4.generic` `rows[].B_squared` |
| sufficient dimension / input dimension, $`J=20`$, $`\varepsilon=0.3`$ | $`111.1`$ / $`19`$ | `rows[].sufficient_dim`, `rows[].input_dim` |
| sufficient dimension / input dimension, $`J=100`$, $`\varepsilon=0.7`$ | $`102.0`$ / $`99`$ | same |
| rows where the generic bound exceeds the input dimension | 12 of 12 | `rows[].vacuous` |
| structured class, $`\sigma_1,\dots,\sigma_6`$ | $`0.192,\ 0.0598,\ 0.0176,\ 0.00736,\ 0.00203,\ 0.00094`$ | `v017:C4.generic` `smooth_example_singular_values` |
| structured class, numerical rank (tolerance $`10^{-10}`$) | 15 | `smooth_example_rank_tol_1e-10` |
| weighted RMS error / rare-test error | $`10^{-3}`$ / $`1`$ | `v017:C9.rare` `weighted_rms_error`, `rare_test_error` |
| centred rank of 8 indicator responses | 7 | `v017:C9.rare` `centred_rank` |

**Why it supports the statement.**

- The generic dimension never drops below the number of statistics that are exact anyway, as (e) states.
- In the structured class, $`\sigma_4=0.0074\le0.01<\sigma_3`$, so with $`B=1`$ the linear interface complexity at $`\varepsilon=0.01`$ is 3, while (4.16) asks for $`r+1\ge2{,}500`$. The bound is valid and far from tight when the spectrum decays.
- The rare test and the indicator class are the two failure modes that (d) and the scope of the theorem warn about: an average does not control a rare test, and bounded responses can need $`m-1`$ modes.

## Reproduce

The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## Artifacts

- `theory/v017_checks/results_v017.json`: entries `v017:C2.spectrum`, `v017:C3.positive`, `v017:C4.generic`, `v017:C9.rare`; functions `c_spectrum`, `c_positive`, `c_generic`, `c_rare` in `theory/v017_checks/check_v017.py`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
