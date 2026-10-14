# Proposition 4.12 · Responses factor through the next predictive state

*Experiments*

[← Framework](../README.md) · [Statement](../statements/04-12_response_predictive_state_factorization.md) · [Proof](../proofs/04-12_response_predictive_state_factorization.md) · [Experiments](04-12_response_predictive_state_factorization.md) · [中文](04-12_response_predictive_state_factorization_ZH.md)

**What is tested here.**

- **(a)–(b), factorization and equal kernels.** On a random catalog with 9 messages, a 3-dimensional predictive basis and 6 tests, the response operator equals $`C\mathcal T_z`$ to $`1.4\times10^{-16}`$ and the two kernels coincide to $`1.4\times10^{-15}`$ (**PASS**).
- **(d), residual version.** With a residual added to every test value, the deviation from $`C\mathcal T_z`$ on a random unit input is $`0.0084`$, below the bound $`\epsilon_{\rm pred}=0.0447`$ (**PASS**).

**Setting used on this page.**

- Finite messages with reference weights $`\rho`$; tests with responses $`g_a(y)=k_a+c_a^\top\Gamma(y)`$ (plus a residual in (d)); $`\mathcal T_zh=\mathbb E_\rho[\Gamma h]`$; the response operator in orthonormal input coordinates $`x=\sqrt\rho\odot h`$.

**Statement under test ([Proposition 4.12](../statements/04-12_response_predictive_state_factorization.md#s4-12)).** (a) $`\mathcal R_z=C\mathcal T_z`$; (b) if $`C`$ has full column rank, $`\ker\mathcal R_z=\ker\mathcal T_z`$; (d) with residuals, $`\lVert\mathcal R_zh-C\mathcal T_zh\rVert_\omega\le\epsilon_{\rm pred}\lVert h\rVert_\rho`$.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 4.19](#c4-19) | (a), (b), (d) | `v017:C7.bridge` | PASS |

<a id="c4-19"></a>

## Check 4.19 · Factorization, equal kernels and the residual bound (Proposition 4.12)

**Verdict.** The factorization and the kernel equality hold to rounding error; the residual version holds with room to spare.

**Why.**

- The factorization depends on the centring: the constants $`k_a`$ must drop out because $`h`$ has $`\rho`$-mean zero. Building $`\mathcal R_z`$ from responses that include random constants tests that step.
- Kernel equality needs full column rank; with $`p=6`$ tests and $`d=3`$ features, $`C`$ has full column rank, and the null spaces are compared in both directions.

**Setup.**

- $`m=9`$ messages, $`\rho`$ from a flat Dirichlet law; $`\Gamma\in\mathbb R^{3\times9}`$, $`C\in\mathbb R^{6\times3}`$ and $`k\in\mathbb R^6`$ standard normal; responses $`g=k\mathbf 1^\top+C\Gamma`$.
- $`\mathcal T_z`$ and $`\mathcal R_z`$ as matrices in the coordinates $`x=\sqrt\rho\odot h`$, after centring under $`\rho`$.
- Kernel comparison: $`\lVert\mathcal R_zN_T\rVert+\lVert\mathcal T_zN_R\rVert`$ with $`N_T,N_R`$ orthonormal bases of the two null spaces.
- Residual version: an additive residual $`e`$ with entries of scale $`0.05`$, uniform test weights $`\omega_a=1/6`$, a random unit input in the centred space; $`\epsilon_{\rm pred}`$ computed from the uncentred residual, which is the larger bound in (4.24).

**Result.**

| Quantity | Value | JSON key (entry `v017:C7.bridge`) |
|---|---|---|
| $`\lVert\mathcal R_z-C\mathcal T_z\rVert`$ (largest entry) | $`1.4\times10^{-16}`$ | `factorization_error` |
| kernel equality error | $`1.4\times10^{-15}`$ | `kernel_equality_error` |
| $`\lVert\mathcal R_zh-C\mathcal T_zh\rVert_\omega`$ with residual, $`\lVert h\rVert_\rho=1`$ | $`0.0084`$ | `residual_lhs` |
| bound $`\epsilon_{\rm pred}`$ | $`0.0447`$ | `residual_bound` |

**Why it supports the statement.** The response operator built from the test responses is exactly $`C\mathcal T_z`$, with the constants removed by centring; the two null spaces coincide when $`C`$ has full column rank; and the residual operator stays within $`\epsilon_{\rm pred}\lVert h\rVert_\rho`$.

## Reproduce

The check uses NumPy ([Harris et al., 2020](#ref-harris2020array)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C7.bridge`; function `c_bridge` in `theory/v017_checks/check_v017.py`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
