# Proposition 4.11 · Static sufficiency is not dynamic sufficiency; finite-horizon closure error

*Experiments*

[← Framework](../README.md) · [Statement](../statements/04-11_static_not_dynamic_closure_error.md) · [Proof](../proofs/04-11_static_not_dynamic_closure_error.md) · [Experiments](04-11_static_not_dynamic_closure_error.md) · [中文](04-11_static_not_dynamic_closure_error_ZH.md)

**What is tested here.**

- **(a), counterexample.** The best least-squares closure of $`(\mathbf 1,s)`$ leaves a residual of $`0.5`$; adding $`b`$ closes the space (rank 3); two states with equal current value are 1 apart after one update (**PASS**).
- **(b), error recursion.** On a random 6-state chain with three features, five steps and one-step gains up to $`1.13`$ (not a contraction), the exact error identity (4.21) holds to $`2.5\times10^{-16}`$ (**PASS**).

**Setting used on this page.**

- Features $`\psi`$, kernels $`K_t`$, matrices $`A_t`$, residuals $`r_t=K_t\psi-A_t\psi`$, true predictive states $`q_t=\mathbb E\psi(Z_t)`$, model predictions $`\hat q_{t+1}=A_t\hat q_t`$.

**Statement under test ([Proposition 4.11](../statements/04-11_static_not_dynamic_closure_error.md#s4-11)).** (a) A statically sufficient feature set need not be closed, and the future gap at equal current value can be 1. (b) $`q_H-\hat q_H=A_{H-1:0}(q_0-\hat q_0)+\sum_tA_{H-1:t+1}\mathbb Er_t(Z_t)`$ exactly, without contraction.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 4.18](#c4-18) | (a), (b) | `v017:C6.closure` | PASS |

<a id="c4-18"></a>

## Check 4.18 · The closure counterexample and the exact error recursion (Proposition 4.11)

**Verdict.** The counterexample behaves exactly as stated, and the error recursion is exact on a non-contractive model.

**Why.**

- (a) is a claim about all linear closures of $`(\mathbf 1,s)`$; the least-squares fit is the best one in the Euclidean sense, and a residual of $`0.5`$ confirms that no exact closure exists.
- (b) claims an identity, not a bound, for arbitrary fitted matrices. A non-contractive example checks that no contraction is used.

**Setup.**

- (a) The four states $`(s,b)\in\lbrace0,1\rbrace^2`$, $`F=s`$, update $`(s,b)\mapsto(b,b)`$; least-squares fit of $`K\psi`$ on $`\psi=(\mathbf 1,s)`$; rank of $`(\mathbf 1,s,KF)`$.
- (b) 6 states; 5 random stochastic matrices $`K_t`$ (rows from a flat Dirichlet law); features: the constant and two standard normal columns; $`A_t`$ fitted by least squares, so $`r_t`$ contains the fitting error; initial law from a flat Dirichlet law; $`\hat q_0=q_0`$ plus Gaussian noise of scale $`0.01`$.

**Result.**

| Quantity | Value | JSON key (entry `v017:C6.closure`) |
|---|---|---|
| largest closure residual of $`(\mathbf 1,s)`$ | $`0.5`$ | `closure_defect` |
| rank after adding $`KF=b`$ | 3 | `closed_test_rank` |
| next-step score gap at equal current value | $`1`$ | `future_gap_equal_current_value` |
| error of identity (4.21) | $`2.5\times10^{-16}`$ | `recursion_identity_error` |
| largest one-step gain $`\lVert A_t\rVert_2`$ | $`1.126`$ | `max_one_step_gain` |

**Why it supports the statement.** The feature set predicts today's score exactly and misses tomorrow's by the maximal amount, as (a) states; the recursion reproduces the predictive error exactly with fitted, non-contractive maps, as (b) states.

## Reproduce

The check uses NumPy ([Harris et al., 2020](#ref-harris2020array)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C6.closure`; function `c_closure` in `theory/v017_checks/check_v017.py`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
