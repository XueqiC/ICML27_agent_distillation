# Proposition 6.10 · Explicit information bound and optimal query allocation by water-filling

*Experiments*

[← Framework](../README.md) · [Statement](../statements/06-10_water_filling_allocation.md) · [Proof](../proofs/06-10_water_filling_allocation.md) · [Experiments](../experiments/06-10_water_filling_allocation.md) · [中文](06-10_water_filling_allocation_ZH.md)

**What is tested here.** Proposition 6.10 is tested by [Check 6.5](../experiments/06-10_water_filling_allocation.md#c6-5): in all 16 random profiles water-filling has the lowest realized error; competing allocations are 3% to 105% worse in mean absolute error, and every predicted error ratio matches the realized ratio within $`0.022`$. Verdict: PASS.

**Setting used on this page.**

- **Environments.** A family $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown ([Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1)).
- **Oracle feedback** $`P^*_\theta`$: the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data.** $`K`$ i.i.d. demonstrations with law $`P^D_\theta`$, and $`n`$ queries that may be adaptive: a query point $`X_i`$ in the query space $`\mathcal X_Q`$ has outcome $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$. The law of the whole transcript is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from the transcript and private randomness (whose law does not depend on $`\theta`$) to a feedback law $`Q`$. Replay, rewriting and synthesis are distillers ([Definition 1.2](../README.md#s1-2)).
- **Audited states** $`z_1,\dots,z_T`$: fixed complete learner states, i.e. parameters together with the optimizer state ([Definition 1.3](../README.md#s1-3)), for example the states reference training reaches at checkpoints.
- **Mean increment** $`\mu^P_t(z)`$: the expected change of the complete learner state when one message from the law $`P`$ is consumed at $`z`$ ([Definition 1.3](../README.md#s1-3)). For the oracle, $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$, and the **learner Jacobian** is $`J_t=D_\theta\mu^*_t`$.
- **Task directions** $`g_t=\nabla V_{t+1}(z_t)`$: gradients of the continuation value of the declared reference tail ([Definition 1.4](../README.md#s1-4)); they do not depend on $`\theta`$.
- **Learner-relative functional** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$, with gradient $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$, and **first-order fidelity error** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$ ([Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2)). By [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) this is the first-order part of the gap in final task score between the oracle feedback and $`Q`$ at the audited states.
- **Agent states** $`s\in\mathcal S`$, a finite set; on this page $`s`$ denotes an agent state ([Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9)).
- **Per-state oracle laws.** At state $`s`$ the oracle message is $`Y\sim P^*_s`$ (for example the teacher's action or correction there), on a finite alphabet; the laws vary independently across states, so nothing is assumed to generalize between states.
- **Influence function** $`f_s(y)`$ of $`\Psi`$ at $`s`$: the derivative of $`\Psi`$ when $`P^*_s`$ is moved towards the point mass at $`y`$; it has mean zero under $`P^*_s`$.
- **Learner-relative weight** $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$: how much one more oracle sample at $`s`$ is worth for estimating $`\Psi`$.
- **Counts.** State $`s`$ receives $`N_s=k_s+n_s`$ oracle samples: $`k_s`$ from demonstrations (in expectation $`KH\,d_T(s)`$, with $`H`$ the trajectory length and $`d_T`$ the teacher's state-visitation distribution) and $`n_s`$ from queries, $`\sum_sn_s=n`$.

**Statement under test ([Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10)).** Under [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9) let $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$, the **learner-relative weight** of state $`s`$.

- **(a) Lower bound.** Let the counts grow proportionally ($`N_s=a_sN`$, $`N\to\infty`$) on the states with $`W_s>0`$. For every distiller the local minimax first-order error is at least

```math
\tfrac14\Big(\sum_{s}\frac{W_s}{N_s}\Big)^{1/2}\big(1-o(1)\big).
```

- **(b) Upper bound.** Assume that $`\Psi`$ is twice continuously differentiable in the probability vectors $`(P_s)_s`$. The per-state empirical law, that is, replaying the acquired messages at each state, achieves

```math
\mathbb E\lvert\hat\Psi-\Psi\rvert\le\Big(\sum_s\frac{W_s}{N_s}\Big)^{1/2}\big(1+o(1)\big).
```

- **(c) Optimal queries (water-filling).** The minimizer of $`\sum_sW_s/(k_s+n_s)`$ over $`n_s\ge0`$ with $`\sum_sn_s=n`$ (continuous relaxation) is

```math
n_s^\star=\Big(\sqrt{W_s/\lambda}-k_s\Big)_+,\qquad \text{with }\lambda>0\text{ fixed by }\textstyle\sum_sn_s^\star=n .
```

  Queries flow to states whose learner-relative weight is high and whose teacher coverage is low.

- **(d) Irreducible part.** A state with $`W_s>0`$ and $`k_s=n_s=0`$, one the student reaches but the teacher does not and that is never queried, leaves an error floor that no $`K`$ removes ([Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7)).

<a id="c6-5"></a>

## Check 6.5 · Water-filling minimizes the realized error (Proposition 6.10)

**Verdict.** PASS. In all 16 profiles the water-filling allocation has the smallest realized mean absolute error, and the predicted error ratios of the five competing allocations match the realized ones within Monte Carlo error.

**Why.** [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10) predicts both the optimal allocation and, through $`\sum_sW_s/N_s`$, how much worse every other allocation is. A competitor with lower realized error, or a predicted ratio outside its simultaneous confidence interval, would falsify it.

**Setup.**

- 16 random profiles with 6–10 states: log-spaced student visitation $`d`$ (shuffled), Dirichlet teacher visitation $`d_T`$, Bernoulli laws with $`\pi\in[0.15,0.85]`$, and learner-relative weights $`W_s=d(s)\,e^{N(0,1.44)}`$ (linear in $`d`$, as for frequent-regime Adam).
- 800 teacher visits and $`n=4{,}000`$ queries. The plug-in functional is exactly linear in the state probabilities, with coefficient $`\sqrt{W_s/(\pi_s(1-\pi_s))}`$, so its variance formula is exact.
- Water-filling by a scalar root for $`\lambda`$, then discretized by marginal-gain exchanges. Competitors: uniform; proportional to $`d`$; water-filling with SGD-type weights $`W_s\,d(s)`$ (the weights one would use for the wrong learner); gradient norm ($`\propto d\sqrt{0.04+0.6\pi}`$); uncertainty ($`\propto\pi(1-\pi)`$).
- Binomial sufficient statistics model the integer visits: 60,000 replicates per arm. Ratio intervals are familywise 95% (Bonferroni over both ratios, all profiles and all five comparisons).
- One extra case: a state with $`W>0`$ and no teacher or query visits.

**Result.**

| Competing allocation | Realized MAE ratio to water-filling: min / median / max | Largest $`\lvert`$predicted − realized$`\rvert`$ |
|---|---|---|
| uniform | $`1.136`$ / $`1.415`$ / $`1.858`$ | $`0.010`$ |
| proportional to $`d`$ | $`1.033`$ / $`1.209`$ / $`1.695`$ | $`0.009`$ |
| water-filling with SGD-type weights | $`1.078`$ / $`1.165`$ / $`1.315`$ | $`0.011`$ |
| gradient norm | $`1.028`$ / $`1.188`$ / $`1.743`$ | $`0.011`$ |
| uncertainty | $`1.146`$ / $`1.441`$ / $`2.054`$ | $`0.022`$ |

- Budget residual at most $`5.9\times10^{-11}`$; KKT residual $`4.4\times10^{-16}`$; the continuous and integer allocations are optimal among the compared ones.
- In every profile the best competitor is 2.8% to 21.9% worse than water-filling in realized mean absolute error.
- Worst standardized deviation between predicted and realized ratios: $`3.23`$, below the familywise critical value $`3.60`$.
- Irreducible case: weights $`(1,2)`$, teacher counts $`(100,0)`$, no queries. The predicted variance is $`+\infty`$; the two candidate laws $`0.25`$ and $`0.75`$ at the unseen state give identical observations; functional gap $`0.5`$, minimax floor $`0.25`$.

**Why it supports Proposition 6.10.**

- The formula $`\sum_sW_s/N_s`$ predicts realized errors to about one percent of the ratio, and the allocation it implies wins in every profile.
- Using another learner's weights costs 8% to 32%: the allocation is learner-relative, not a property of the data alone.
- Limits: the profiles are synthetic and the plug-in functional is linear; the check does not estimate $`W_s`$ from a real learner.

![Figure 6.2](../figures/fig6_2_learner_weights.png)

*Figure 6.2, panels (b) and (c) concern this page.* (b) Realized mean absolute error per profile for water-filling, uniform, proportional-to-$`d`$ and SGD-weight allocations; water-filling is lowest in every profile. (c) Predicted against realized error ratios with familywise intervals, on the diagonal. Panels (a) and (d) belong to Proposition 6.11 and Corollary 6.12(a).

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C3 --output /tmp/v016_C3.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

The first command reruns the check group(s) of this page into a separate file; `theory/v016_checks/check_v016.py` without `--only` regenerates the full result file (about 104 s). All checks run in float64 on CPU with seeds `numpy.SeedSequence([160160, group, subcase, ...])` (PCG64).

## Artifacts

- `theory/v016_checks/results_v016.json`: claim ids `v016:C3.allocation`, each with assertions, worst residuals, failures and data rows.
- `theory/v016_checks/results_v016_raw.json.gz`: complete rows; not tracked by git, its SHA-256 is recorded under `raw_artifact` in the main file.
- `theory/v016_checks/SPEC.md` (sections `v016:C3`) and `theory/v016_checks/check_v016.py`.
- `figures/fig6_2_learner_weights.png` (PDF alongside)
