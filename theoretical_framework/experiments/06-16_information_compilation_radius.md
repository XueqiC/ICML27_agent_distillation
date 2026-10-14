# Proposition 6.16 · Information–compilation radius

*Experiments*

[← Framework](../README.md) · [Statement](../statements/06-16_information_compilation_radius.md) · [Proof](../proofs/06-16_information_compilation_radius.md) · [Experiments](06-16_information_compilation_radius.md) · [中文](06-16_information_compilation_radius_ZH.md)

**What is tested here.** One check, [Check 6.10](#c6-10), made of three small synthetic cases.

- Uncertainty that lies entirely in $`\ker C`$ contributes nothing, while an unreachable target still leaves an error of 0.1, the distance to the legal set (**PASS**).
- Two allowed targets 1 apart with a legal set that contains both give the radius 0.5, half the diameter (**PASS**).
- For a box uncertainty set the inner supremum is attained at a vertex: 200,000 random points of the box never exceed the vertex maximum (**PASS**).

These are exact computations on constructed cases; no real response interface has been built, because the real-model logs do not yet record per-item responses or a declared target.

**Setting used on this page.**

- Tests with weights $`\omega_a>0`$; interface $`g_a(y)=k_a+c_a^\top\phi(y)+e_a(y)`$ with $`C\in\mathbb R^{p\times r}`$; $`\lVert x\rVert_\omega^2=\sum_a\omega_ax_a^2`$.
- Uncertainty set $`\mathcal M\subset\mathbb R^r`$ for the target feature mean; legal image $`\mathcal K=\lbrace C\mu_Q\rbrace`$; $`\Gamma(q)=\sup_{\mu\in\mathcal M}\lVert C\mu-q\rVert_\omega`$ and $`\mathfrak r=\min_{q\in\mathcal K}\Gamma(q)`$.

**Statement under test ([Proposition 6.16](../statements/06-16_information_compilation_radius.md#s6-16)).** $`\mathfrak r\ge\max\lbrace\tfrac12\mathrm{diam}_\omega(C\mathcal M),\sup_\mu\mathrm{dist}_\omega(C\mu,\mathcal K)\rbrace`$ (b); uncertainty along $`\ker C`$ does not change the radius and a single target outside $`\mathcal K`$ leaves its distance (e); for a box $`\mathcal M`$ the inner supremum is attained at a vertex (f).

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 6.10](#c6-10) | (b), (e), (f) | `v017:C13.radius` | PASS |

<a id="c6-10"></a>

## Check 6.10 · Invisible uncertainty, unreachable targets and vertex maxima (Proposition 6.16)

**Verdict.** PASS. In each case the computed radius equals the value predicted by the lower bounds of (b), which are attained here, and the box case confirms the vertex reduction of (f).

**Why.** The practical claim of the proposition is diagnostic: it tells apart an information gap (half the diameter of $`C\mathcal M`$) from a realizability gap (the distance to $`\mathcal K`$) and says that uncertainty along $`\ker C`$ is free. The first two cases isolate each gap; the third checks the computational reduction used when $`\mathcal M`$ is a box.

**Setup.**

- *Invisible uncertainty, unreachable target.* $`r=2`$, one test, $`C=(1,0)`$. The evidence allows the two scenarios $`\mu=(0.2,-100)`$ and $`\mu=(0.2,100)`$, which differ only along $`\ker C`$, so $`C\mathcal M=\lbrace0.2\rbrace`$. The legal responses fill $`\mathcal K=[0,0.1]`$, searched on a grid of 101 points.
- *Information gap.* One test, $`C\mathcal M=\lbrace0,1\rbrace`$ (two scenarios), $`\mathcal K=[0,1]`$, grid of 101 points.
- *Box.* $`r=4`$, $`p=3`$, $`C`$ with standard normal entries, $`\mathcal M`$ a random box $`\prod_i[l_i,u_i]`$ with $`l_i<0<u_i`$, a random centre $`q`$. Compare $`\max_\mu\lVert C\mu-q\rVert`$ over the 16 vertices with the maximum over 200,000 uniform random points of the box. The check uses unit test weights.

**Result.**

| Case | Quantity | Value | Predicted |
|---|---|---|---|
| invisible uncertainty | $`\mathrm{diam}(C\mathcal M)`$ | 0 | 0 |
| unreachable target | radius $`\mathfrak r`$ | 0.1 | $`\mathrm{dist}(0.2,[0,0.1])=0.1`$ |
| information gap | radius $`\mathfrak r`$ | 0.5 | $`\tfrac12\mathrm{diam}=0.5`$ |
| box | maximum over the 16 vertices | 5.7366 | upper bound for every point |
| box | maximum over 200,000 random points | 5.6436 | at most 5.7366 |

**Why it supports Proposition 6.16.** In the first case the uncertainty spans 200 units, yet the radius is decided entirely by the distance to the legal set: more evidence about the second coordinate would be wasted, and only a larger legal set helps, exactly as in (e). In the second case the legal set is as rich as possible and the radius is still half the diameter, the information floor of (b). The box case shows that the inner problem can be solved by vertex enumeration when $`r`$ is small; it does not test the general hardness statement, which is a reduction, not a computation.

## Reproduce

From the repository root (CPU only, a few seconds):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The suite uses NumPy ([Harris et al., 2020](#ref-harris2020array)) with a fixed seed derived from 20261014 and the check name (`rng_for` in the script).

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C13.radius`.
- `theory/v017_checks/check_v017.py`: function `c_radius`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
