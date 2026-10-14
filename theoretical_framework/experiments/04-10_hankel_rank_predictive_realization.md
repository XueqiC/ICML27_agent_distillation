# Theorem 4.10 · Hankel rank and linear predictive realization

*Experiments*

[← Framework](../README.md) · [Statement](../statements/04-10_hankel_rank_predictive_realization.md) · [Proof](../proofs/04-10_hankel_rank_predictive_realization.md) · [Experiments](04-10_hankel_rank_predictive_realization.md) · [中文](04-10_hankel_rank_predictive_realization_ZH.md)

**What is tested here.**

- **(b), realization from the Hankel row space.** On a random controlled process with 4 hidden states, the Hankel matrix over prefixes and suffixes of length at most 3 has rank 4 with the constant column. A 4-dimensional realization fitted on prefixes of length at most 2 predicts every test after words of length 4 and 5, outside the fitting range, with error $`4.0\times10^{-15}`$ (**PASS**).
- **(c), conditioning.** In the indicator basis, where the predictive vector is the belief itself, the normalized update equals Bayes' rule exactly (**PASS**).

**Setting used on this page.**

- Controls $`b`$ with Markov kernels $`K_b`$, words $`w`$ with $`K_w=K_{b_1}\cdots K_{b_\ell}`$, bounded outputs $`F_j`$, initial law $`\mu_0`$.
- The task Hankel matrix $`H(p,(w,j))=\mu_0K_pK_wF_j`$ with a constant column; observation operators $`T_{b,o}`$ with $`\sum_oT_{b,o}=K_b`$.

**Statement under test ([Theorem 4.10](../statements/04-10_hankel_rank_predictive_realization.md#s4-10)).** (b) If $`\mathrm{rank}\,H=d`$, there is a $`d`$-dimensional linear realization $`q(pb)=A_bq(p)`$, $`H(p,(w,j))=c_j^\top A_wq(p)`$, and none of smaller dimension. (c) With observations, $`q'=M_{b,o}q/(e_1^\top M_{b,o}q)`$ is the predictive state of the posterior.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 4.17](#c4-17) | (b), (c) | `v017:C5.realization` | PASS |

<a id="c4-17"></a>

## Check 4.17 · Realization from the Hankel row space and the conditioned update (Theorem 4.10(b)–(c))

**Verdict.** The Hankel rank equals the number of hidden states; the row-space realization predicts unseen longer words to rounding error; the conditioned update is Bayes' rule.

**Why.**

- (b) claims more than a low-rank fit: the transition maps obtained from the column shift must predict the expectations after words that were not used to fit them. Predicting words of length 4–5 from a realization fitted on prefixes of length at most 2 tests exactly that.
- (c) is the normalization that turns open-loop predictions into an online update; an error in which row carries the probability would show up immediately.

**Setup.**

- 4 hidden states, 2 controls, 2 observations, 2 outputs. For each control, observation-and-next-state probabilities are drawn uniformly from $`[0.1,1]`$ and normalized, giving $`T_{b,o}`$; $`K_b=\sum_oT_{b,o}`$. Outputs uniform on $`[0,1]`$; $`\mu_0`$ from a flat Dirichlet law.
- Hankel matrix over all prefixes and suffixes of length at most 3, with the constant column.
- Realization: coordinates in an orthonormal basis of the row space; $`A_b`$ obtained by least squares from the pairs $`(q(p),q(pb))`$ with $`\lvert p\rvert\le2`$.
- Prediction test: for every word of length 4 or 5, propagate $`q`$ from the empty prefix with the fitted $`A_b`$ and compare the predicted row with the exact expectations.
- Conditioning: control 1, observation 0, from $`\mu_0`$, in the indicator basis.

**Result.**

| Quantity | Value | JSON key (entry `v017:C5.realization`) |
|---|---|---|
| Hankel rank with constant column / hidden states | 4 / 4 | `hankel_rank_with_constant`, `hidden_states` |
| largest prediction error, words of length 4–5 | $`4.0\times10^{-15}`$ | `max_prediction_error_longer_words` |
| conditioned update versus Bayes posterior | $`0`$ | `conditioning_vs_bayes_error` |

**Why it supports the statement.**

- The realization built only from the Hankel row space and short prefixes reproduces every test expectation after longer words, as (4.19) states: the shift-induced maps are exact transition maps, not a fit.
- The conditioned update reproduces Bayes' rule. The check uses the indicator basis, in which (4.20) reduces to Bayes' rule on the belief; it confirms the normalization and the role of the constant coordinate, not the estimation of $`M_{b,o}`$ in a compressed basis.

## Reproduce

The check uses NumPy ([Harris et al., 2020](#ref-harris2020array)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C5.realization`; function `c_realization` in `theory/v017_checks/check_v017.py`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
