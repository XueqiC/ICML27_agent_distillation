# Corollary 8.16 · Composed guarantee from reuse, estimation and compilation

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-16_composed_guarantee.md) · [Proof](../proofs/08-16_composed_guarantee.md) · [Experiments](08-16_composed_guarantee.md) · [中文](08-16_composed_guarantee_ZH.md)

**What is tested here.**

- **(c), size of one step.** With 10 replicates per cell on a 20-candidate catalog ($`\varepsilon_n=0.578`$) and 10 drift pairs at the measured evaluation noise ($`\hat\Delta=0.718`$), the absolute per-step floor $`B(\epsilon_j+\Delta)`$ at $`B=2`$ is 2.59. It exceeds the trivial bound 1 on the whole gap before any sum over steps: the guarantee is **valid but vacuous** at this scale.
- **Signed sum versus per-term certification.** A Monte Carlo of the signed-sum estimator over $`R=20`$ terms with $`N=400`$ queries gives variance 1.011 against the exact $`\sigma^2R^2/N=1`$ (**PASS**). At unit noise and unit target error the signed sum needs 400 queries; certifying each term to $`\varepsilon/R`$ and adding absolute values needs 8,000.

**Setting used on this page.**

- Reference (target) and compiled legal feedback laws $`P^0_t`$, $`P^S_t`$; approximate value $`\widehat V_t`$; local response difference $`d_t`$; Bellman residuals $`r^0_t`$; terminal error $`e_R`$; exit times $`\tau^0`$, $`\tau^S`$ ([Proposition 3.4](../statements/03-04_joint_decomposition_approximate_value.md#s3-4)).
- At a certified state with anchor $`j`$: information–compilation radius $`\mathfrak r_{t,z}`$ and solver error $`\eta_{\rm opt}`$ ([Proposition 6.16](../statements/06-16_information_compilation_radius.md#s6-16)); anchor's first discarded singular value $`\hat\sigma_{j,r_j+1}`$ and estimation error $`\epsilon_j`$ ([Proposition 8.12](../statements/08-12_finite_catalog_spectral_certificate.md#s8-12)); drift $`\Delta_{t,z}`$ ([Proposition 8.14](../statements/08-14_anchor_reuse_drift_certificate.md#s8-14)); density bound $`B_{t,z}`$; weight $`\omega_{a_t}`$ of the test that reads $`\widehat V_{t+1}`$.

**Statement under test ([Corollary 8.16](../statements/08-16_composed_guarantee.md#s8-16)).** At every certified state

```math
\lvert d_t(z)\rvert\le\beta_t(z)=\frac{\mathfrak r_{t,z}+\eta_{\rm opt}+B_{t,z}\big(\hat\sigma_{j,r_j+1}+\epsilon_j+\Delta_{t,z}\big)}{\sqrt{\omega_{a_t}}},
```

and $`\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau^S}\beta_t+\mathcal E_{\rm dyn}+\mathcal E_{\rm terminal}+\mathcal E_{\rm exit}`$; at the measured scale the per-step floor is 2.59 at $`B=2`$, above the trivial bound.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 8.23](#c8-23) | (c) and the cost comparison | `v017:C22.composed` | PASS; valid but vacuous |

<a id="c8-23"></a>

## Check 8.23 · Per-step floor and signed-sum versus per-term certification (Corollary 8.16)

**Verdict.**

- Per-step floor $`B(\epsilon_j+\Delta)=2\times(0.578+0.718)=2.59`$, with every other term of $`\beta_t`$ nonnegative. If the initial state is certified, the composed bound is at least 2.59; if it is not, the exit term alone is 1. Since the gap is at most 1, the bound is **vacuous** at this scale either way. The floor is $`1.296\,B`$: $`B=2`$ is the two-candidate worst case, a 20-candidate catalog allows up to $`\sqrt{40}\approx6.3`$, and a compiled law close to its target may have $`B=0.5`$ (floor 0.65). Even then two or more audited steps sum to more than 1.
- The signed-sum estimator has the predicted variance $`\sigma^2R^2/N`$ (Monte Carlo 1.011 against 1.000): **PASS**.

**Why.** The corollary is valid by construction; the question is whether it can be informative. Its anchor and drift terms come from bounded-range concentration, so their size does not depend on how small one step's effect is. The check puts in the radii of [Check 8.18](../experiments/08-12_finite_catalog_spectral_certificate.md#c8-18) and [Check 8.21](../experiments/08-14_anchor_reuse_drift_certificate.md#c8-21) and compares the certification scheme with direct estimation of the signed sum.

**Setup.**

- *Floor.* One test ($`\omega_{a_t}=1`$), $`B=2`$; $`\epsilon_j=\varepsilon_n=\sqrt{\log(2pm/\delta)/(2n)}`$ with $`p=1`$, $`m=20`$, $`n=10`$, $`\delta=0.05`$; $`\Delta=\hat\Delta`$ of [Proposition 8.14](../statements/08-14_anchor_reuse_drift_certificate.md#s8-14) with $`\bar W=(\sqrt{15}/300)^2`$, 10 pairs, $`M=10`$, $`\delta=0.05`$.
- *Signed sum.* $`R=20`$ terms $`a_t`$ drawn once from a standard normal; $`N=400`$ queries split uniformly, 20 per term, each a noisy observation with $`\sigma=1`$; the estimator is the sum of the per-term means; 20,000 Monte Carlo repetitions; NumPy ([Harris et al., 2020](#ref-harris2020array)).
- *Counts* at $`\sigma=\varepsilon=1`$: signed sum $`\sigma^2R^2/\varepsilon^2`$; per-term certification to $`\varepsilon/R`$ with $`\sigma^2R^2/\varepsilon^2`$ queries per term, times $`R`$ terms (logarithmic union-bound factors omitted).

**Result** (`theory/v017_checks/results_v017.json`, entry `v017:C22.composed`):

| quantity | value |
|---|---:|
| $`\varepsilon_n`$ ($`p=1`$, $`m=20`$, $`n=10`$) | 0.578 |
| $`\hat\Delta`$ (10 pairs, measured noise) | 0.718 |
| per-step floor $`B(\epsilon_j+\Delta)`$ | 2.59 |
| signed-sum variance, Monte Carlo | 1.011 |
| signed-sum variance, exact $`\sigma^2R^2/N`$ | 1.000 |
| queries at unit error, signed sum | 400 |
| queries at unit error, per-term absolute certification | 8,000 |

**Why it supports the statement.**

- (c): the floor is computed exactly from the two radii; it exceeds the trivial bound, so at this scale the composed guarantee is an error budget, not a useful certificate. Summing over $`R`$ steps multiplies the floor by up to $`R`$.
- The comparison shows where the waste comes from: the certification scheme adds $`R`$ absolute radii of size $`\varepsilon/R`$, which costs a factor $`R`$ more queries than estimating the signed sum. This is the cost of the scheme, not a lower bound for the problem, and it distinguishes estimating one protocol pair's gain from certifying every local response.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The suite uses NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C22.composed`; code `theory/v017_checks/check_v017.py`, function `c_composed`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
