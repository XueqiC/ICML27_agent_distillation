# Proposition 8.13 · Shared message features across checkpoints

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-13_shared_features_across_checkpoints.md) · [Proof](../proofs/08-13_shared_features_across_checkpoints.md) · [Experiments](08-13_shared_features_across_checkpoints.md) · [中文](08-13_shared_features_across_checkpoints_ZH.md)

**What is tested here.**

- **(c), the counterexample.** With two checkpoints responding to orthogonal directions and state-conditional inputs, the actual RMS response is 1, the common-input bound $`B\sigma_1(\mathcal S)`$ gives 0.707, and the tail-sum bound gives 1 (**PASS**: the wrong bound fails, the right one holds with equality).
- **(a)–(b), the tail identity.** On a random stack of four checkpoint operators, the state-averaged Hilbert–Schmidt residual of the rank-3 shared interface equals the tail sum of squared stacked singular values to $`5.3\times10^{-15}`$ (**PASS**).

**Setting used on this page.**

- One reference measure $`\rho`$ on a common message space, $`\mathcal H=L^2_0(\rho)`$; tests with weights $`\omega`$ of the same meaning at every checkpoint; checkpoint weights $`\nu`$.
- $`\mathcal R_z`$ is the response operator at checkpoint $`z`$; $`(\mathcal Sh)(z,a)=(\mathcal R_zh)(a)`$ is the stacked operator, with singular values $`\sigma_j(\mathcal S)`$ and $`\Pi_r`$ the projection onto its first $`r`$ right singular vectors; $`h_z`$ is the feedback difference at $`z`$, with $`\lVert h_z\rVert_\rho\le B`$.

**Statement under test ([Proposition 8.13](../statements/08-13_shared_features_across_checkpoints.md#s8-13)).**

```math
\mathbb E_{z\sim\nu}\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega^2\le B^2\sum_{j>r}\sigma_j(\mathcal S)^2 ,
```

$`\Pi_r`$ minimizes the state-averaged Hilbert–Schmidt residual among rank-$`r`$ shared interfaces, and the tail sum cannot be replaced by $`B^2\sigma_{r+1}(\mathcal S)^2`$.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 8.19](#c8-19) | (a), (b), (c) | `v017:C18.shared` | PASS |

<a id="c8-19"></a>

## Check 8.19 · Conditional inputs need the tail sum (Proposition 8.13)

**Verdict.**

- In the two-checkpoint construction the true RMS response is $`1.0`$. The common-input operator bound gives $`0.7071`$, which is **wrong**; the Hilbert–Schmidt tail bound gives $`1.0`$, which holds with equality.
- On a random stack the averaged residual of the truncated shared basis equals the tail sum of squared singular values up to rounding ($`5.3\times10^{-15}`$).

**Why.** The tempting shortcut is to stack the response rows of all checkpoints, take one singular value decomposition and quote $`B\sigma_{r+1}(\mathcal S)`$ as the reuse error. That is correct only for a single input shared by all checkpoints. In training the reference and alternative feedback change with the state, so each checkpoint receives its own input. The check shows the shortcut failing on the smallest possible example and the correct quantity, the tail sum, being exact for the shared truncated basis.

**Setup.**

- *Counterexample.* Two checkpoints of weight $`\tfrac12`$, response rows $`(1,0)`$ and $`(0,1)`$ in orthonormal coordinates of a two-dimensional $`\mathcal H`$, inputs $`e_1`$ at the first checkpoint and $`e_2`$ at the second ($`B=1`$), $`r=0`$. The stacked operator is $`I_2/\sqrt2`$. Rescaling the scores by any $`c>0`$, as in the proof, keeps them bounded and leaves the comparison unchanged.
- *Tail identity.* Four checkpoint operators with independent standard normal $`5\times8`$ entries, weight $`\tfrac14`$ each (stacked and divided by 2), shared rank $`r=3`$.
- Seeded with `SeedSequence([20261014, tag])`; NumPy ([Harris et al., 2020](#ref-harris2020array)).

**Result** (`theory/v017_checks/results_v017.json`, entry `v017:C18.shared`):

| quantity | value |
|---|---:|
| actual RMS response, $`r=0`$ | 1.0000 |
| common-input bound $`B\sigma_1(\mathcal S)`$ | 0.7071 |
| tail-sum bound $`B(\sum_j\sigma_j(\mathcal S)^2)^{1/2}`$ | 1.0000 |
| tail identity error, random stack, $`r=3`$ | $`5.3\times10^{-15}`$ |

**Why it supports the statement.**

- (c): the actual value exceeds the first-discarded-singular-value bound, so that bound is invalid for state-conditional inputs; the tail sum is attained.
- (a)–(b): the averaged Hilbert–Schmidt residual of $`\Pi_r`$ equals $`\sum_{j>r}\sigma_j(\mathcal S)^2`$, the quantity that bounds the conditional-input error and that $`\Pi_r`$ minimizes.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The suite uses NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C18.shared`; code `theory/v017_checks/check_v017.py`, function `c_shared`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
