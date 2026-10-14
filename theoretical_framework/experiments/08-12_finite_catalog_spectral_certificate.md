# Proposition 8.12 · Finite-catalog spectral certificate

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-12_finite_catalog_spectral_certificate.md) · [Proof](../proofs/08-12_finite_catalog_spectral_certificate.md) · [Experiments](08-12_finite_catalog_spectral_certificate.md) · [中文](08-12_finite_catalog_spectral_certificate_ZH.md)

**What is tested here.**

- **(b)–(c), gap-free residual.** On a random weighted catalog with an entrywise error of at most 0.004, the operator error is 0.00115 and the residual of the estimated rank-3 interface is 0.0666, below the guaranteed $`\sigma_4+2\varepsilon=0.0746`$ (**PASS**).
- **(f) and the radius at affordable budgets.** A catalog of 20 candidates with 4 replicates costs 80 continuation runs when all tests share one continuation protocol and 800 when they need 10. The entrywise Hoeffding radius at $`p=10`$, $`m=20`$, $`\delta=0.05`$ is 1.06, 0.53, 0.26 and 0.13 for 4, 16, 64 and 256 replicates per cell. All exceed 0.1, while $`\sigma_1\le\tfrac12`$: the certificate is valid but, at this scale, vacuous.

**Setting used on this page.**

- One checkpoint; $`p`$ tests with weights $`w`$; $`m`$ frozen candidates with weights $`q`$; cell means $`G_{ij}\in[0,1]`$; $`n`$ independent replicates per cell, averaged into $`\hat G`$; $`p_{\rm train}`$ distinct continuation protocols among the tests.
- $`M=D_{\sqrt w}GD_{\sqrt q}\Pi_q`$ with $`\Pi_q=I-\sqrt q\sqrt q^{\top}`$, singular values $`\sigma_k`$; $`\hat M`$, $`\hat\sigma_k`$ and the projection $`\hat\Pi_r`$ onto its first $`r`$ right singular vectors are built from $`\hat G`$.
- $`\varepsilon_n=\sqrt{\log(2pm/\delta)/(2n)}`$ is the entrywise Hoeffding radius.

**Statement under test ([Proposition 8.12](../statements/08-12_finite_catalog_spectral_certificate.md#s8-12)).** With probability at least $`1-\delta`$, every cell satisfies $`\lvert\hat G_{ij}-G_{ij}\rvert\le\varepsilon_n`$, and on that event

```math
\lVert\hat M-M\rVert_{\rm op}\le\varepsilon_n,\qquad\lvert\hat\sigma_k-\sigma_k\rvert\le\varepsilon_n,\qquad\lVert M(I-\hat\Pi_r)\rVert_{\rm op}\le\sigma_{r+1}+2\varepsilon_n ,
```

with a per-test bound that adds $`2\varepsilon_n`$ for working with estimated cells; a complete catalog costs $`m\,n\,p_{\rm train}`$ continuation runs.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 8.18](#c8-18) | (b), (c), (f) | `v017:C17.catalog` | PASS; radius vacuous at affordable $`n`$ |

<a id="c8-18"></a>

## Check 8.18 · Gap-free truncation residual and the catalog radius (Proposition 8.12)

**Verdict.**

- The operator error of the estimated matrix is below the entrywise bound, and the residual of the estimated interface is below $`\sigma_{r+1}+2\varepsilon`$, with no use of a spectral gap: **PASS**.
- At budgets of 4 to 256 replicates per cell the radius $`\varepsilon_n`$ is 0.13 or more. Since the response operator has $`\sigma_1\le\tfrac12`$, and the real-model candidate difference measured so far is of order $`10^{-3}`$ on the surrogate ([Check 8.17](08-08_crn_paired_replacement_certificate.md#c8-17)), the strict certificate is **valid but vacuous** at these budgets.

**Why.**

- (b)–(c) claim that an entrywise error controls the whole spectrum and the truncated interface with the same constant, because the weights sum to one. A random instance with an error placed at the edge of the bound tests the constants and the absence of any gap condition.
- (f) and the radius table make the cost of a strict catalog explicit before any real catalog is run.

**Setup.**

- $`p=7`$ tests, $`m=11`$ candidates, rank $`r=3`$. Weights $`w_i`$, $`q_j`$ drawn uniformly from $`[0.3,1.3]`$ and normalized. $`G=\mathrm{sigmoid}(Z)`$ with standard normal $`Z`$.
- $`\hat G=\mathrm{clip}(G+U,0,1)`$ with $`U_{ij}`$ uniform on $`[-0.004,0.004]`$: the event of (a) with $`\varepsilon=0.004`$.
- Radius table: $`\varepsilon_n`$ at $`p=10`$, $`m=20`$, $`\delta=0.05`$ for $`n\in\lbrace4,16,64,256\rbrace`$; run counts for $`m=20`$, $`n=4`$ and $`p_{\rm train}\in\lbrace1,10\rbrace`$.
- Seeded with `SeedSequence([20261014, tag])`; NumPy ([Harris et al., 2020](#ref-harris2020array)).

**Result** (`theory/v017_checks/results_v017.json`, entry `v017:C17.catalog`):

| quantity | value |
|---|---:|
| entrywise bound $`\varepsilon`$ | 0.004 |
| $`\lVert\hat M-M\rVert_{\rm op}`$ | 0.00115 |
| $`\lVert M(I-\hat\Pi_3)\rVert_{\rm op}`$ | 0.0666 |
| guaranteed $`\sigma_4+2\varepsilon`$ | 0.0746 |

| replicates per cell $`n`$ | 4 | 16 | 64 | 256 |
|---|---:|---:|---:|---:|
| $`\varepsilon_n`$ ($`p=10`$, $`m=20`$, $`\delta=0.05`$) | 1.06 | 0.53 | 0.26 | 0.13 |

| catalog | continuation runs |
|---|---:|
| $`m=20`$, $`n=4`$, $`p_{\rm train}=1`$ | 80 |
| $`m=20`$, $`n=4`$, $`p_{\rm train}=10`$ | 800 |

**Why it supports the statement.**

- The measured residual 0.0666 respects the gap-free bound 0.0746, and the operator error 0.00115 respects the entrywise bound, as (b)–(c) assert; no gap between $`\sigma_3`$ and $`\sigma_4`$ was assumed or needed.
- The radius table shows what (a) costs: it falls like $`n^{-1/2}`$ and is still 0.13 at 256 replicates per cell. A strict catalog certificate that resolves differences of a few hundredths therefore needs replicate counts far beyond any continuation budget. That is a property of the range-based Hoeffding radius, not a proof that no cheaper certificate exists.
- The run counts show that metrics read off one run are free: the cost multiplies the distinct continuation protocols, not the tests.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The suite uses NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C17.catalog`; code `theory/v017_checks/check_v017.py`, function `c_catalog`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
