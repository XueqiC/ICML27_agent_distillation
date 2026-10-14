# Proposition 6.14 · Convex realizability over a finite legal pool

*Experiments*

[← Framework](../README.md) · [Statement](../statements/06-14_convex_realizability_legal_pool.md) · [Proof](../proofs/06-14_convex_realizability_legal_pool.md) · [Experiments](06-14_convex_realizability_legal_pool.md) · [中文](06-14_convex_realizability_legal_pool_ZH.md)

**What is tested here.** One check, [Check 6.8](#c6-8), on a synthetic pool of 30 messages with 4 features.

- A target inside the convex hull is matched exactly by a vertex solution that uses 5 messages, the $`r+1`$ of part (b) (**PASS**).
- The sparse solution has a much larger density than the dense mixture that defined the target, as part (d) predicts (**PASS**).
- A target outside the hull is reported infeasible, as part (a) requires (**PASS**).

The check is a synthetic computation; it says nothing about how often real targets fall inside a real pool's hull.

**Setting used on this page.**

- **Legal pool.** A finite set $`\lbrace y_1,\dots,y_m\rbrace`$ of verified legal messages; a legal mixture is $`q\in\Delta_m`$.
- **Features.** $`\phi(y)\in\mathbb R^r`$, $`\Phi=[\phi(y_1),\dots,\phi(y_m)]`$; the feature mean of $`q`$ is $`\Phi q`$.
- **Target.** $`\mu_*\in\mathbb R^r`$.
- **Reference measure.** $`\rho`$ with $`\rho_i>0`$; $`dQ/d\rho=(q_i/\rho_i)_i`$ and $`\lVert f\rVert_\rho^2=\sum_i\rho_if_i^2`$. The quantity $`\lVert dQ/d\rho-1\rVert_\rho`$ is the $`\chi`$-distance of $`q`$ from $`\rho`$; $`\lVert dQ/d\rho\rVert_\rho^2=1+\lVert dQ/d\rho-1\rVert_\rho^2`$.

**Statement under test ([Proposition 6.14](../statements/06-14_convex_realizability_legal_pool.md#s6-14)).** A legal mixture with $`\Phi q=\mu_*`$ exists if and only if $`\mu_*\in\mathrm{conv}\lbrace\phi(y_i)\rbrace`$ (a); if one exists, a vertex solution uses at most $`r+1`$ messages (b); outside the hull the best feature error is the positive distance to the hull (c); a mixture supported on $`S`$ has $`\lVert dQ/d\rho\rVert_\rho^2\ge1/\rho(S)`$ (d); density caps keep compilation convex and force $`\rho(S)\ge1/\kappa`$ (e).

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 6.8](#c6-8) | (a), (b), (d) | `v017:C10.realizable` | PASS |

<a id="c6-8"></a>

## Check 6.8 · Exact compilation with at most r+1 messages, its density price, and infeasibility outside the hull (Proposition 6.14)

**Verdict.** PASS. The vertex solution matches the target's 4 feature means to $`8.3\times10^{-17}`$ with exactly 5 messages; its $`\chi`$-distance from the reference measure is 2.78, against 0.87 for the dense mixture that generated the target; a target placed outside the hull is declared infeasible.

**Why.** Parts (a) and (b) are the claim that compilation into a verified pool is a linear feasibility problem with sparse solutions. Part (d) is the warning attached to that claim: the sparse solution pays in density. A solver returning a feasible point with more than $`r+1`$ messages at a vertex, a feasible point for a target outside the hull, or a sparse solution with a small density would contradict the proposition.

**Setup.**

- Pool of $`m=30`$ messages with $`r=4`$ features $`\phi(y_i)`$ drawn i.i.d. standard normal (seeded); uniform reference measure $`\rho_i=1/30`$.
- Target: $`\mu_*=\Phi w`$ for a random dense mixture $`w`$ (Dirichlet with unit parameters), so $`\mu_*`$ lies inside the hull.
- Compilation: the linear program $`\lbrace q\ge0,\ \Phi q=\mu_*,\ \mathbf 1^\top q=1\rbrace`$ with a zero objective, solved by the HiGHS dual simplex method ([Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)) through SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)); a simplex method returns a vertex.
- Infeasibility: the target is moved to the pool's feature mean plus $`10\max_{i,k}\lvert\phi_k(y_i)\rvert`$ in every coordinate, which is outside the hull.

**Result.**

| Quantity | Value |
|---|---|
| support of the vertex solution | 5 |
| bound $`r+1`$ of part (b) | 5 |
| feature-mean error $`\lVert\Phi q-\mu_*\rVert_\infty`$ | $`8.3\times10^{-17}`$ |
| $`\chi`$-distance of the dense mixture $`w`$ | 0.875 |
| $`\chi`$-distance of the sparse solution $`q`$ | 2.785 |
| target outside the hull | infeasible (solver status 2) |

By part (d), any mixture on 5 of the 30 messages has $`\lVert dQ/d\rho\rVert_\rho^2\ge30/5=6`$, that is a $`\chi`$-distance of at least $`\sqrt5\approx2.236`$. The sparse solution, at 2.785, respects this floor, and it is more than three times as far from the reference measure as the dense mixture with the same feature means.

**Why it supports Proposition 6.14.** The vertex solution attains the support bound of (b) exactly, the infeasible case behaves as (a) requires, and the density of the sparse solution sits above the floor of (d). The comparison with the dense mixture shows the practical meaning of (d): the same feature means can be matched by laws whose densities, and hence whose spectral error bounds in [Theorem 4.9](../statements/04-09_spectral_sufficiency.md#s4-9), differ by a factor of three.

## Reproduce

From the repository root (CPU only, a few seconds):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The suite uses NumPy ([Harris et al., 2020](#ref-harris2020array)) with a fixed seed derived from 20261014 and the check name (`rng_for` in the script), and SciPy's `linprog`.

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C10.realizable`.
- `theory/v017_checks/check_v017.py`: function `c_realizable`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
