# Proposition 6.15 · Linear-Gaussian target: exact risk, whitened response, one-query gain

*Experiments*

[← Framework](../README.md) · [Statement](../statements/06-15_linear_gaussian_target.md) · [Proof](../proofs/06-15_linear_gaussian_target.md) · [Experiments](06-15_linear_gaussian_target.md) · [中文](06-15_linear_gaussian_target_ZH.md)

**What is tested here.** One check, [Check 6.9](#c6-9), on a synthetic linear-Gaussian experiment with a three-dimensional consequence target.

- The trace risk equals the sum of the generalized eigenvalues to $`5.6\times10^{-17}`$ (**PASS**).
- The closed-form one-query gain equals the difference of the two traces to $`9.0\times10^{-17}`$ (**PASS**).
- A Monte Carlo run of generalized least squares reproduces the exact risk within 0.3% (tolerance 2%, **PASS**).

**Setting used on this page.**

- $`y=D\vartheta+\epsilon`$, $`\epsilon\sim\mathcal N(0,\Sigma)`$, $`\Sigma\succ0`$; target $`f=C\vartheta\in\mathbb R^k`$; loss $`\lVert x\rVert_W^2=x^\top Wx`$.
- $`I=D^\top\Sigma^{-1}D`$, $`B=C^\top WC`$; generalized eigenvalues $`Bv_j=\lambda_jIv_j`$; whitened operator $`\mathcal R_I=W^{1/2}CI^{-1/2}`$.
- A unit-noise query $`a`$ adds $`aa^\top`$ to $`I`$, and $`\Delta(a)=a^\top I^{-1}BI^{-1}a/(1+a^\top I^{-1}a)`$.

**Statement under test ([Proposition 6.15](../statements/06-15_linear_gaussian_target.md#s6-15)).** For $`I\succ0`$, generalized least squares has constant risk $`\mathrm{tr}(BI^{-1})=\sum_j\lambda_j`$, the minimax risk (a); the $`\lambda_j`$ are the squared singular values of $`\mathcal R_I`$ (c); one query lowers the minimax risk by exactly $`\Delta(a)`$ (d).

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 6.9](#c6-9) | (a), (c), (d) | `v017:C11.gaussian` | PASS |

<a id="c6-9"></a>

## Check 6.9 · Exact trace risk, eigen-identity and one-query gain (Proposition 6.15)

**Verdict.** PASS. Both identities hold to rounding error, and simulated generalized least squares has the predicted risk.

**Why.** Parts (a) and (d) are the basis of the computable query rule: rank candidate queries by $`\Delta(a)`$. An error in the trace formula, in the identification of its eigenvalues, or in the Sherman–Morrison step would make that ranking wrong. The Monte Carlo run checks the risk formula against an estimator that is actually run, not only against another formula.

**Setup.**

- $`d=4`$ parameters, $`D\in\mathbb R^{9\times4}`$ with i.i.d. standard normal entries (seeded), $`\Sigma=I_9`$.
- Information $`I=D^\top D+0.4\,I_4`$; the added term plays the role of prior information, for example demonstrations, and makes $`I`$ well conditioned.
- Target $`C\in\mathbb R^{3\times4}`$ with standard normal entries, $`W=\mathrm{diag}(0.2,0.3,0.5)`$.
- One random query vector $`a\in\mathbb R^4`$.
- Monte Carlo: a fixed $`\vartheta`$, 200,000 independent draws of $`y=D\vartheta+\mathcal N(0,I_9)`$, the estimator $`\hat\vartheta=(D^\top D)^{-1}D^\top y`$, compared with the exact risk $`\mathrm{tr}(B(D^\top D)^{-1})`$ of that estimator.

**Result.**

| Quantity | Value |
|---|---|
| $`\mathrm{tr}(BI^{-1})`$ | 0.40062 |
| $`\lvert\mathrm{tr}(BI^{-1})-\sum_j\lambda_j\rvert`$ | $`5.6\times10^{-17}`$ |
| one-query gain $`\Delta(a)`$ | 0.05375 |
| $`\lvert\Delta(a)-[\mathrm{tr}(BI^{-1})-\mathrm{tr}(B(I+aa^\top)^{-1})]\rvert`$ | $`9.0\times10^{-17}`$ |
| exact risk of least squares on $`D`$ alone | 0.45909 |
| Monte Carlo risk (200,000 draws) | 0.45762 |

The Monte Carlo risk differs from the exact one by 0.3%, inside the 2% tolerance set for 200,000 draws.

**Why it supports Proposition 6.15.** The eigenvalues computed from $`I^{-1/2}BI^{-1/2}`$ add up to the trace, which is the identity behind (a) and (c); the one-query formula agrees with a direct recomputation of the enlarged experiment, which is (d); and the simulated estimator attains the trace risk, so the risk formula describes an estimator that can actually be run. Minimaxity itself is a proof, not a numerical statement, and is not tested here.

## Reproduce

From the repository root (CPU only, a few seconds):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The suite uses NumPy ([Harris et al., 2020](#ref-harris2020array)) with a fixed seed derived from 20261014 and the check name (`rng_for` in the script).

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C11.gaussian`.
- `theory/v017_checks/check_v017.py`: function `c_gaussian`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
