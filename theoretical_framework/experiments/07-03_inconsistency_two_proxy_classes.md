# Proposition 7.3 · Inconsistency of two proxy classes

*Experiments*

[← Framework](../README.md) · [Statement](../statements/07-03_inconsistency_two_proxy_classes.md) · [Proof](../proofs/07-03_inconsistency_two_proxy_classes.md) · [Experiments](07-03_inconsistency_two_proxy_classes.md) · [中文](07-03_inconsistency_two_proxy_classes_ZH.md)

**What is tested here.**

- Part (b), on constructed message pairs in the proposition's own setting: **PASS** in all 24 settings. The parameter-only score ranks the larger message first, the cosine score ties, and both the exact first-order value and the finite response prefer the smaller message ([Check 7.4](#c7-4)).
- Both parts, descriptively, on a trained controlled network where the proposition's assumptions do not hold: every cosine and parameter-only ranker has a wrong-sign rate between 14.7% and 28.8% on 1,687 provider pairs, against 2.8% for the finite paired-replacement score (summary of [Check 7.5](07-05_where_first_order_rankers_fail.md#c7-5), whose full report is on the page of [Remark 7.5](07-05_where_first_order_rankers_fail.md#c7-5)).
- Part (a) is a two-line identity ($`F\mapsto1-F`$) and is not tested separately.

**Setting used on this page.**

- One coordinate of warm Adam without bias correction ($`\epsilon=0`$), tail gradients IID with mean $`\bar g`$ and mean square $`\bar v`$, score affine in the parameter; $`\rho=\lvert\bar g\rvert/\sqrt{\bar v}`$ is the signal-to-noise ratio.
- A message of amplitude $`\lambda`$ is $`g=\lambda\sqrt{v}`$ in the direction of $`\bar g`$; $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$ is the optimum of the quadratic law of [Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2), and $`\lambda_{\rm PO}=(1-\beta_1)\beta_2/(\beta_1\rho(1-\beta_2))`$ the amplitude below which the parameter-only score increases with $`\lambda`$.
- *Parameter-only score*: the continuation gradient restricted to the parameters, paired with the message's own Adam step. *Cosine score*: a function of the direction of the message (LESS-type, [Xia et al., 2024](#ref-xia2024less)). *Exact first-order value* $`\mathrm{Val}`$: all three channels of [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1). *Finite response*: the measured change of the score against a zero-gradient message, with common random numbers ([Glasserman, 2004](#ref-glasserman2004monte)).
- A ranker is *wrong* on a pair when the sign of its score difference is opposite to the sign of the gold difference.

**Statement under test ([Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes.md#s7-3)).** (a) A statistic that depends only on the state and the feedback law, not on the task, ranks some pair wrongly, or fails to separate it, on some task. (b) In the setting of Corollary 7.2 with $`\alpha>0`$ and $`0<\lambda_1<\lambda_2`$: the parameter-only score $`\alpha\eta\,(\beta_1\rho+(1-\beta_1)\lambda)/\sqrt{\beta_2+(1-\beta_2)\lambda^2}`$ increases for $`\lambda<\lambda_{\rm PO}`$, so it prefers $`\lambda_2`$; every cosine score ties the two; the exact value prefers $`\lambda_1`$ whenever $`\Delta F(\lambda_2)<\Delta F(\lambda_1)`$, which holds for $`\lambda_1+\lambda_2>2\lambda^\star`$ up to the error of Corollary 7.2(b).

<a id="c7-4"></a>

## Check 7.4 · Parameter-only and cosine scores on constructed pairs (Proposition 7.3(b))

**Verdict.** PASS in 24/24 settings. The parameter-only score always ranks the larger message first, the cosine score always ties, and both the exact first-order value and the finite response always prefer the smaller message.

**Why.** [Proposition 7.3(b)](../statements/07-03_inconsistency_two_proxy_classes.md#s7-3) predicts that two same-direction messages with $`\lambda_1+\lambda_2>2\lambda^\star`$ are ranked in the wrong order by parameter-only scores and are not separated by cosine scores.

**Setup.**

- The 24 core settings of the amplitude-law check ([Check 7.2](07-02_stationary_amplitude_law.md#c7-2)): exact one-dimensional Adam, $`\beta_1=0.9`$, $`\beta_2\in\lbrace0.99,0.999\rbrace`$, $`\rho\in\lbrace0.3,0.7,0.95\rbrace`$, $`T\in\lbrace50,200,1000,5000\rbrace`$, tail gradients IID $`\mathcal N(\rho,1-\rho^2)`$, 4,096 replicates with common random numbers.
- In each setting, $`\lambda_1=1.25\lambda^\star`$ and $`\lambda_2=1.75\lambda^\star`$, so $`\lambda_1+\lambda_2=3\lambda^\star>2\lambda^\star`$. The amplitudes range from 1.3 to 116.7, and the warmth $`(1-\beta_2)\lambda^2`$ from 0.002 to 13.6.
- Recorded: the parameter-only score and its derivative sign at both endpoints; the cosine score; the exact $`\mathrm{Val}`$; and the finite rollout response, each with 95% CRN intervals over 4,096 replicates.
- Assertions:
  - $`(1-\beta_1)\beta_2-\beta_1\rho(1-\beta_2)\lambda>0`$ at both endpoints, i.e. both amplitudes are below $`\lambda_{\rm PO}`$;
  - the parameter-only score prefers $`\lambda_2`$;
  - the cosines are equal;
  - $`\mathrm{Val}`$ and the rollout prefer $`\lambda_1`$.

**Result.**

| Assertion | Instances | Worst margin | Status |
|---|---:|---:|---|
| both amplitudes below $`\lambda_{\rm PO}`$ | 48 | — | PASS |
| parameter-only prefers $`\lambda_2`$ | 24 | smallest score gap 0.044 | PASS |
| cosine ties | 24 | both cosines 1 | PASS |
| exact $`\mathrm{Val}`$ prefers $`\lambda_1`$ | 24 | smallest gap 0.263 (CI lower end 0.263) | PASS |
| finite response prefers $`\lambda_1`$ | 24 | smallest gap 0.262 (CI lower end 0.261) | PASS |

**Why it supports the statement.** The reversal holds with wide margins in every setting, including settings with warmth well above the quadratic range. So the inconsistency is not an artifact of the small-increment expansion used in the proof.

**Limits.** One coordinate and exogenous gradients, the setting of the proposition; the network test is summarized below.

### [Check 7.5](07-05_where_first_order_rankers_fail.md#c7-5) · Rankers on a controlled network, by second-moment share

- **What it tests for this proposition.** Whether parameter-only and cosine scores rank feedback laws reliably on a trained network with real optimizer states, outside the one-coordinate setting of (b).
- **Setup.** A frozen panel of a small controlled network (one-hot context, 4 tanh units, sigmoid output, float64), four complete states, eight Adam/SGD recipes, nine provider laws on two-label packets: 1,687 provider pairs with nonzero gold, gold being the exact eight-update outcome difference. The cosine is a LESS-style proxy, not the official LESS implementation.
- **Key numbers.** Wrong-sign rates on all 1,687 pairs: LESS-style cosine 28.8% (345 ties), update-oriented cosine 15.6% (345 ties), parameter-only score with the exact continuation gradient 14.7% (120 ties); finite paired-replacement score 2.8%. The parameter-only score's errors concentrate where the second-moment channel matters (14.4% below share 0.5, 27.3% above); the cosine's do not (31.6% below, 24.2% above).
- **Verdict.** Descriptive; consistent with the proposition. The panel does not isolate the amplitude mechanism of (b); [Check 7.4](#c7-4) does.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C4,C5,C5b
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

- Seeds: `numpy.SeedSequence([160160, group, subcase, ...])`, PCG64, float64, single BLAS thread.

The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v016_checks/results_v016.json` — claim ids `v016:C4.ranking` ([Check 7.4](#c7-4)), `v016:C5` and `v016:C5b` ([Check 7.5](07-05_where_first_order_rankers_fail.md#c7-5)).
- `theory/v016_checks/results_v016_raw.json.gz` — raw arrays referenced by JSON pointers. Not committed; SHA-256 `1777069872ea80c09dfd5260d610871b68e41a10721eba2826dcd4d647874f69`.
- `theory/v016_checks/SPEC.md`, `check_v016.py`, `c5c6.py`, `c5c6b.py` and their tests.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-glasserman2004monte"></a>

**Paul Glasserman (2004).** Monte Carlo Methods in Financial Engineering. Springer, New York (Stochastic Modelling and Applied Probability 53). DOI [10.1007/978-0-387-21617-1](https://doi.org/10.1007/978-0-387-21617-1).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
