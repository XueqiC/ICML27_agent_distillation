# Proposition 8.15 · Two-fidelity response estimation and its cost condition

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-15_two_fidelity_response_estimation.md) · [Proof](../proofs/08-15_two_fidelity_response_estimation.md) · [Experiments](08-15_two_fidelity_response_estimation.md) · [中文](08-15_two_fidelity_response_estimation_ZH.md)

**What is tested here.**

- **(a)–(b), exact.** On a finite joint law the two-fidelity estimator is unbiased, and the optimal allocation meets the error target with the cost $`S^2/\varepsilon^2`$ exactly (**PASS**).
- **(c)–(d), thresholds.** The fixed-coefficient thresholds are 0.7662 (cheap part reused) and 0.7875 (not reused) at a cost ratio of one tenth. The nested estimator with the optimal coefficient has threshold 0.5750 when every cheap output is charged, and a numerical minimization over the coefficient and the allocation reproduces its cost-ratio formula at five correlations (**PASS**). When the cheap output of each paired sample is reused, the same minimization gives 0.3163, which matches $`\sqrt c=0.3162`$. Like for like: 0.7875 against 0.5750 without reuse, 0.7662 against 0.3163 with reuse.
- **Scale.** The synthetic cost ratio 0.0402 assumes a pairing correlation of about 0.995. No short-versus-full correlation has been measured on the real model, so no real saving is claimed. Each cell still needs at least one full continuation: with four full replicates per cell the saving is at most fourfold.

**Setting used on this page.**

- One cell: full-continuation output $`X\in[0,1]^p`$ with target $`f=\mathbb EX`$; cheap output $`Y\in[0,1]^p`$ from a frozen estimator.
- Group 0: $`n_0`$ cheap outputs; group 1: $`n_1`$ coupled pairs $`(X',Y')`$; groups independent. $`v_0`$, $`v_1`$, $`v_H`$ are the weighted total variances of $`Y`$, $`X-Y`$ and $`X`$; $`c_0`$, $`c_1`$, $`c_H`$ the costs of a cheap sample, a paired sample and a full sample; $`\varrho`$ the correlation of $`X`$ and $`Y`$ for a scalar test; $`c=c_L/c_H`$ in the nested estimator.

**Statement under test ([Proposition 8.15](../statements/08-15_two_fidelity_response_estimation.md#s8-15)).** $`\hat f=\bar Y+\overline{X'-Y'}`$ is unbiased with $`\mathbb E\lVert\hat f-f\rVert_\omega^2=v_0/n_0+v_1/n_1`$; the least cost at mean squared error $`\varepsilon^2`$ is $`(\sqrt{v_0c_0}+\sqrt{v_1c_1})^2/\varepsilon^2`$; it saves computation iff

```math
\sqrt{v_0c_0}+\sqrt{v_1c_1}<\sqrt{v_Hc_H} ;
```

the nested estimator with the optimal coefficient has relative cost $`(\sqrt{1-\varrho^2}+\lvert\varrho\rvert\sqrt c)^2`$, below 1 iff $`\lvert\varrho\rvert>2\sqrt c/(1+c)`$; and each cell still needs at least one full continuation.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 8.22](#c8-22) | (a)–(e) | `v017:C21.fidelity` | PASS |

<a id="c8-22"></a>

## Check 8.22 · Unbiasedness, allocation and the cost thresholds (Proposition 8.15)

**Verdict.**

- Unbiasedness and the allocation identities hold exactly on the finite law: **PASS**.
- Thresholds: $`\varrho>0.7662`$ ($`c_1=c_H`$), $`\varrho>0.7875`$ ($`c_1=1.1c_H`$) with coefficient 1; $`\lvert\varrho\rvert>0.5750`$ for the nested estimator with the optimal coefficient at $`c=0.1`$ when every cheap output is charged, and $`\lvert\varrho\rvert>0.3163`$ when the cheap output of each paired sample is reused.
- The formula $`(\sqrt{1-\varrho^2}+\lvert\varrho\rvert\sqrt c)^2`$ agrees with a direct numerical minimization at $`\varrho\in\lbrace0.5,0.6,0.8,0.9,0.99\rbrace`$ to better than $`10^{-6}`$.

**Why.** The proposition replaces expensive full continuations by a cheap predictor plus a correction. Its value hinges on two things the check makes explicit: the exactness of the unbiasedness and allocation algebra, and the correlation a real cheap predictor would need. The nested optimal-coefficient formula is taken from the multifidelity literature ([Peherstorfer, Willcox & Gunzburger, 2016](#ref-peherstorfer2016optimal)); the brute-force minimization checks it independently in this parametrization.

**Setup.**

- *Finite law.* $`(X,Y)\in\lbrace0,1\rbrace^2`$ with $`\Pr(X=Y=0)=\Pr(X=Y=1)=0.5(1-0.0025)`$ and $`\Pr(X\ne Y)=0.0025`$, split equally between the two disagreements. Then $`v_H=v_0=0.25`$, $`v_1=0.0025`$, correlation $`0.995`$. Costs $`c_0=0.01`$, $`c_1=1.01`$, $`c_H=1`$; target mean squared error $`10^{-4}`$.
- *Fixed-coefficient thresholds.* Scalar test, $`v_0=v_H`$, $`c_0=c_H/10`$; root of $`\sqrt{0.1}+\sqrt{2(1-\varrho)c_1/c_H}=1`$ for $`c_1/c_H\in\lbrace1,1.1\rbrace`$ by Brent's method (SciPy, [Virtanen et al., 2020](#ref-virtanen2020scipy)).
- *Nested estimator.* Relative cost times variance $`(1+c\,\kappa)\big(1+(\alpha^2-2\alpha\varrho)(1-1/\kappa)\big)`$ with $`\kappa=n_L/n_H\ge1`$, minimized by Nelder–Mead over $`(\alpha,\log(\kappa-1))`$ and capped at 1 (full samples alone); compared with the closed form, also capped at 1.

**Result** (`theory/v017_checks/results_v017.json`, entry `v017:C21.fidelity`):

| quantity | value |
|---|---:|
| $`v_H`$, $`v_0`$, $`v_1`$ | 0.25, 0.25, 0.0025 |
| optimal $`n_0`$, $`n_1`$ (continuous) | 5,012.5, 49.9 |
| two-fidelity cost / full-only cost | 0.0402 |
| fixed-coefficient threshold, $`c_1=c_H`$ | 0.7662 |
| fixed-coefficient threshold, $`c_1=1.1c_H`$ | 0.7875 |
| nested optimal-coefficient threshold, $`c=0.1`$, every cheap output charged | 0.5750 |
| nested optimal-coefficient threshold, $`c=0.1`$, cheap output of each pair reused | 0.3163 ($`\sqrt c=0.3162`$) |
| maximum saving with four full replicates per cell | 4-fold |

| $`\varrho`$ | 0.50 | 0.60 | 0.80 | 0.90 | 0.99 |
|---|---:|---:|---:|---:|---:|
| relative cost, formula | 1.000 | 0.980 | 0.728 | 0.519 | 0.206 |
| relative cost, numerical minimum | 1.000 | 0.980 | 0.728 | 0.519 | 0.206 |

**Why it supports the statement.**

- (a)–(b): the variance and cost identities hold to machine precision; the 0.0402 ratio is what the formula gives for a nearly perfectly correlated cheap predictor.
- (c)–(d): the thresholds are where the cost condition switches; the agreement between formula and minimization confirms the nested cost ratio. At $`\varrho=0.5`$ there is no saving.
- (e): one full continuation per cell is unavoidable, so with four full replicates per cell the best possible saving is fourfold, and the number of cells is unchanged.
- The real-model premise, a correlation above these thresholds between a short and a full continuation of the same cell, has not been measured; until it is, the proposition gives a decision rule, not a speed-up.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The suite uses NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C21.fidelity`; code `theory/v017_checks/check_v017.py`, function `c_fidelity`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-peherstorfer2016optimal"></a>

**Benjamin Peherstorfer, Karen Willcox, Max Gunzburger (2016).** Optimal model management for multifidelity Monte Carlo estimation. *SIAM Journal on Scientific Computing* 38, no. 5 (section A, Methods and Algorithms for Scientific Computing). DOI [10.1137/15M1046472](https://doi.org/10.1137/15M1046472).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
