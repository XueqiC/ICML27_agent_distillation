# Proposition 4.13 · Finite-horizon Gramians, and what control theory does not transfer

*Experiments*

[← Framework](../README.md) · [Statement](../statements/04-13_finite_horizon_gramians.md) · [Proof](../proofs/04-13_finite_horizon_gramians.md) · [Experiments](04-13_finite_horizon_gramians.md) · [中文](04-13_finite_horizon_gramians_ZH.md)

**What is tested here.** On a time-varying linear system whose one-step gains reach $`1.28`$, the finite-horizon Hankel singular values equal the square roots of the eigenvalues of $`W_c^{1/2}W_oW_c^{1/2}`$ to $`1.0\times10^{-15}`$ (**PASS**). No stability is used.

**Setting used on this page.**

- A finite-horizon time-varying linear system $`\delta z_{t+1}=A_t\delta z_t+B_t\delta u_t`$, $`\delta o_t=C_t\delta z_t`$, split at $`\tau`$; reachability matrix $`\mathscr C_\tau`$, observability matrix $`\mathscr O_\tau`$, Hankel operator $`H_\tau=\mathscr O_\tau\mathscr C_\tau`$, Gramians $`W_c=\mathscr C_\tau\mathscr C_\tau^\top`$ and $`W_o=\mathscr O_\tau^\top\mathscr O_\tau`$.

**Statement under test ([Proposition 4.13](../statements/04-13_finite_horizon_gramians.md#s4-13)).** (a) $`\sigma_j(H_\tau)^2=\lambda_j(W_c^{1/2}W_oW_c^{1/2})`$ for every $`j`$, with no stability or contraction assumption.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 4.20](#c4-20) | (a) | `v017:C8.gramian` | PASS |

<a id="c4-20"></a>

## Check 4.20 · Hankel singular values from finite-horizon Gramians without stability (Proposition 4.13(a))

**Verdict.** The identity holds to rounding error on a system that expands in one direction.

**Why.** The classical statements about Hankel singular values are usually given for stable time-invariant systems. (a) claims the identity over a finite horizon for arbitrary time-varying matrices; a system with one-step gains above 1 checks that no stability is hidden in the argument.

**Setup.**

- Two-dimensional state, horizon $`N=5`$, split $`\tau=2`$.
- $`A_t=\begin{pmatrix}1.15+0.03t&0.12\\0.04&0.75\end{pmatrix}`$, $`B_t=(1,\ 0.4(-1)^t)^\top`$, $`C_t=(1,\ 0.25+0.06t)`$.
- $`\mathscr C_\tau`$ from the inputs at $`t=0,1`$; $`\mathscr O_\tau`$ from the outputs at $`t=2,\dots,5`$; singular values of $`H_\tau`$ against the eigenvalues of $`W_c^{1/2}W_oW_c^{1/2}`$, with $`W_c^{1/2}`$ from a symmetric eigendecomposition.

**Result.**

| Quantity | Value | JSON key (entry `v017:C8.gramian`) |
|---|---|---|
| identity error | $`1.0\times10^{-15}`$ | `identity_error` |
| Hankel singular values | $`4.746,\ 0.0371`$ | `hankel_singular_values` |
| largest one-step gain $`\lVert A_t\rVert_2`$ | $`1.283`$ | `max_one_step_gain` |

**Why it supports the statement.** The Hankel singular values are exactly the Gramian quantities of (a) although the system is not contractive, so the finite-horizon identity holds without the stability that the classical bounds need.

## Reproduce

The check uses NumPy ([Harris et al., 2020](#ref-harris2020array)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C8.gramian`; function `c_gramian` in `theory/v017_checks/check_v017.py`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
