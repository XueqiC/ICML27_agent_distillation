# Proposition 3.4 · Joint decomposition with an approximate value

*Experiments*

[← Framework](../README.md) · [Statement](../statements/03-04_joint_decomposition_approximate_value.md) · [Proof](../proofs/03-04_joint_decomposition_approximate_value.md) · [Experiments](03-04_joint_decomposition_approximate_value.md) · [中文](03-04_joint_decomposition_approximate_value_ZH.md)

**What is tested here.**

- **(a), exact identity.** On a four-round learner with a nonlinear Adam-like update, the joint decomposition reproduces the exact learning gap $`0.07616`$ with error $`1.4\times10^{-17}`$ (**PASS**).
- **(b), absolute bound.** The bound is $`0.178`$, valid and about 2.3 times the gap (**PASS**).
- **(d), stopped version.** With a certified set that both processes leave with probability $`0.667`$ and $`0.424`$, the stopped bound is $`1.197`$: valid, and here larger than 1, so it carries no information (**PASS**, reported as valid but loose).

**Setting used on this page.**

- Reference and alternative protocols $`K_t^0,K_t^S`$ on complete learner states, from a common initial state; $`L_E(S)=\mathbb E_SF_E(Z_R)`$.
- An approximate value $`\widehat V_t`$ with reference Bellman residual $`r_t^0=K_t^0\widehat V_{t+1}-\widehat V_t`$, terminal error $`e_R=F_E-\widehat V_R`$ and local response difference $`d_t=(K_t^0-K_t^S)\widehat V_{t+1}`$.
- Certified sets $`C_t^j`$ and per-process exit times $`\tau^j`$ for part (d).

**Statement under test ([Proposition 3.4](../statements/03-04_joint_decomposition_approximate_value.md#s3-4)).** (a) $`L_E(0)-L_E(S)=\mathbb E_S\sum_td_t+\mathbb E_0\sum_tr_t^0-\mathbb E_S\sum_tr_t^0+\mathbb E_0e_R-\mathbb E_Se_R`$. (b) Taking absolute values term by term bounds the gap. (d) With $`\widehat V\in[0,1]`$, stopping each process at its own exit from the certified sets bounds the gap by the stopped sums, the terminal errors on the non-exited paths and the two exit probabilities.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 3.5](#c3-5) | (a), (b), (d) | `v017:C1.joint` | PASS |

<a id="c3-5"></a>

## Check 3.5 · Exact joint decomposition, absolute bound and stopped bound (Proposition 3.4)

**Verdict.** The identity holds to rounding error. Both bounds are valid. The absolute bound is 2.3 times the gap; the stopped bound exceeds 1 and is therefore uninformative in this instance.

**Why.**

- (a) is an exact identity with an arbitrary approximate value. A sign or occupancy error (for example, using the reference occupancy for the response term) would leave a nonzero residual on a process whose two chains visit different states.
- The learner is nonlinear and carries its complete optimizer state, so the check exercises the claim that no linearization, stationarity or contraction is needed.
- (d) charges each process's exit once. The check also shows the price: when exits are likely, the bound is valid but useless.

**Setup.**

- Complete state $`z=(\theta,m,v,s)`$ with a step counter $`s`$, starting at $`(0.2,0.03,0.08,2)`$. Messages $`y\in\lbrace-1,+1\rbrace`$.
- Update: gradient $`\theta+0.2\theta^3-y`$; moments $`m'=0.7m+0.3\,\mathrm{grad}`$, $`v'=0.85v+0.15\,\mathrm{grad}^2`$; bias correction with $`0.7^{s+1}`$ and $`0.85^{s+1}`$; step $`\theta'=\theta-0.17\,\hat m/(\sqrt{\hat v}+0.01)`$.
- Feedback laws: $`\Pr(y=+1\mid z)=\sigma(0.6+0.5\theta-0.2m)`$ for the reference and $`\sigma(-0.4+0.5\theta-0.2m)`$ for the alternative, with $`\sigma`$ the logistic function.
- $`R=4`$ rounds; final score $`F_E=\sigma(1.3\theta-0.25\theta^2)`$; approximate value $`\widehat V_t=\sigma(0.15+0.03t+0.4\theta)`$ for $`t<R`$ and $`\widehat V_R=F_E`$ (so $`e_R=0`$).
- Exact enumeration of all $`2^4`$ paths of each process.
- Part (d): the same certified set $`C_t=\lbrace\theta\le0.2\rbrace`$ for both processes; the cut $`0.2`$ is the larger of $`\theta_0`$ and the 60% quantile of the round-3 parameters.

**Result.**

| Quantity | Value | JSON key (entry `v017:C1.joint`) |
|---|---|---|
| exact gap $`L_E(0)-L_E(S)`$ | $`0.076162`$ | `return_gap` |
| error of identity (3.7) | $`1.4\times10^{-17}`$ | `joint_identity_error` |
| signed response term $`\mathbb E_S\sum_td_t`$ | $`0.021068`$ | `signed_response_term` |
| signed residual difference $`\mathbb E_0\sum_tr_t^0-\mathbb E_S\sum_tr_t^0`$ | $`0.055093`$ | `signed_residual_difference` |
| absolute bound (3.8) | $`0.17818`$ | `absolute_bound` |
| stopped bound (3.9) | $`1.1975`$ | `stopped_bound` |
| exit probabilities, reference / alternative | $`0.667`$ / $`0.424`$ | `exit_probability_reference`, `exit_probability_alternative` |

**Why it supports the statement.**

- The two signed pieces add up to the gap, as (3.7) states. Most of the gap (0.055 of 0.076) sits in the residual difference: an approximate value that is poor along the two processes moves the bookkeeping from the response term into the residual terms, without making the identity less exact.
- The absolute bound is valid but more than twice the gap, because it adds residuals that cancel in (3.7). This is why the statement recommends estimating the signed terms.
- The stopped bound is valid and exceeds 1 because both exit probabilities are large; each is paid once, as (3.9) states, but here that is already too much.

## Reproduce

The check uses NumPy ([Harris et al., 2020](#ref-harris2020array)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## Artifacts

- `theory/v017_checks/results_v017.json`: entry `v017:C1.joint`; function `c_joint` in `theory/v017_checks/check_v017.py`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
