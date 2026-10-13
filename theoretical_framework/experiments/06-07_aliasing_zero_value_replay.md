# Corollary 6.7 · Aliasing and the zero value of replay

*Experiments*

[← Framework](../README.md) · [Statement](../statements/06-07_aliasing_zero_value_replay.md) · [Proof](../proofs/06-07_aliasing_zero_value_replay.md) · [Experiments](../experiments/06-07_aliasing_zero_value_replay.md) · [中文](06-07_aliasing_zero_value_replay_ZH.md)

**What is tested here.** Corollary 6.7 is tested by [Check 6.3](../experiments/06-07_aliasing_zero_value_replay.md#c6-3): along the null space of the demonstrations the data laws coincide exactly for every $`K`$ up to $`10^6`$, and the worst error never falls below half the functional gap. Verdict: PASS.

**Setting used on this page.**

- **Environments.** A family $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown ([Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1)).
- **Oracle feedback** $`P^*_\theta`$: the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data.** $`K`$ i.i.d. demonstrations with law $`P^D_\theta`$, and $`n`$ queries that may be adaptive: a query point $`X_i`$ in the query space $`\mathcal X_Q`$ has outcome $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$. The law of the whole transcript is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from the transcript and private randomness (whose law does not depend on $`\theta`$) to a feedback law $`Q`$. Replay, rewriting and synthesis are distillers ([Definition 1.2](../README.md#s1-2)).
- **Audited states** $`z_1,\dots,z_T`$: fixed complete learner states, i.e. parameters together with the optimizer state ([Definition 1.3](../README.md#s1-3)), for example the states reference training reaches at checkpoints.
- **Mean increment** $`\mu^P_t(z)`$: the expected change of the complete learner state when one message from the law $`P`$ is consumed at $`z`$ ([Definition 1.3](../README.md#s1-3)). For the oracle, $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$, and the **learner Jacobian** is $`J_t=D_\theta\mu^*_t`$.
- **Task directions** $`g_t=\nabla V_{t+1}(z_t)`$: gradients of the continuation value of the declared reference tail ([Definition 1.4](../README.md#s1-4)); they do not depend on $`\theta`$.
- **Learner-relative functional** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$, with gradient $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$, and **first-order fidelity error** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$ ([Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2)). By [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) this is the first-order part of the gap in final task score between the oracle feedback and $`Q`$ at the audited states.

**Statement under test ([Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7)).**

- If $`\theta_0\neq\theta_1`$ satisfy $`P^D_{\theta_0}=P^D_{\theta_1}`$ and $`\Psi(\theta_0)\neq\Psi(\theta_1)`$, and either $`n=0`$ or $`P^Q_{\theta_0}(\cdot\mid x)=P^Q_{\theta_1}(\cdot\mid x)`$ for every $`x`$, then for every $`K`$ every distiller has $`\max_i\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$.
- Replay, rewriting and synthesis are functions of the transcript and private randomness, so they are covered by "every distiller" and cannot lower this bound.
- The bound is learner-relative. Whether demonstrations suffice depends on whether the directions that the demonstrations cannot see have a component along **this learner's** $`c`$; the same demonstrations can suffice for one learner and not for another ([Corollary 6.12](../statements/06-12_learner_separations.md#s6-12)(a)).
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)(a) is the instance in which the two worlds have identical demonstration laws and different optimal student policies; [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4) is the statement that post-processing cannot help.

<a id="c6-3"></a>

## Check 6.3 · Aliased directions leave a floor that no K removes (Corollary 6.7)

**Verdict.** PASS. For every aliased model and every $`K\in\lbrace1,10,100,10^6\rbrace`$ the data laws at $`\theta`$ and $`\theta+h`$ coincide exactly, and the worst error is at least half the functional gap.

**Why.** [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7) claims that when $`h`$ lies in the null space of the demonstrations and $`c^\top h\neq0`$, no number of demonstrations helps. A nonzero difference in the data laws, or an estimator that beats $`\lvert\Delta\Psi\rvert/2`$ in both worlds, would falsify it.

**Setup.**

- 8 aliased models (an exact zero column of $`A_D`$, so $`A_Dh=0`$ with no numerical tolerance), $`n=0`$, $`K\in\lbrace1,10,100,10^6\rbrace`$: 32 instances, 3,408 assertions.
- Compare the means and covariances of the demonstration sufficient statistics at $`\theta`$ and $`\theta+h`$; check the deterministic triangle inequality behind the bound, and record the midpoint estimator, which attains it.

**Result.**

- Mean and covariance differences: exactly $`0`$ in all 208 comparisons.
- Triangle inequality: 3,200 checks, worst residual $`5.6\times10^{-17}`$ (tolerance $`2\times10^{-14}`$).
- Example: functional gap $`0.1866`$, minimax floor $`0.0933`$, attained by the midpoint at every $`K`$ (Figure 6.1(c)).

**Why it supports Corollary 6.7.** The laws are identical, not merely close, so the floor is exact and independent of $`K`$; this is the mechanism by which replay or rewriting of demonstrations, garblings of the transcript in the sense of [Blackwell, 1953](#ref-blackwell1953equivalent), cannot help.

![Figure 6.1](../figures/fig6_1_information_bound.png)

*Figure 6.1, panel (c) concerns this page.* The worst-error floor $`\lvert c^\top h\rvert/2`$ of eight aliased instances against $`K`$, flat in every instance (Corollary 6.7). Panels (a) and (b) belong to Theorems 6.3/6.5 and Lemma 6.4.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C2 --output /tmp/v016_C2.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

The first command reruns the check group(s) of this page into a separate file; `theory/v016_checks/check_v016.py` without `--only` regenerates the full result file (about 104 s). All checks run in float64 on CPU with seeds `numpy.SeedSequence([160160, group, subcase, ...])` (PCG64). The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)) for linear algebra and random numbers, and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)) for optimization and statistical functions.

## Artifacts

- `theory/v016_checks/results_v016.json`: claim ids `v016:C2.aliasing`, each with assertions, worst residuals, failures and data rows.
- `theory/v016_checks/results_v016_raw.json.gz`: complete rows; not tracked by git, its SHA-256 is recorded under `raw_artifact` in the main file.
- `theory/v016_checks/SPEC.md` (sections `v016:C2`) and `theory/v016_checks/check_v016.py`.
- `figures/fig6_1_information_bound.png` (PDF alongside)

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-blackwell1953equivalent"></a>

**David Blackwell (1953).** Equivalent comparisons of experiments. *The Annals of Mathematical Statistics* 24, no. 2, 265–272. DOI [10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
