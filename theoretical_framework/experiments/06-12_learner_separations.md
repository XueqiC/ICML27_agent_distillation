# Corollary 6.12 · Learner separations

*Experiments*

[← Framework](../README.md) · [Statement](../statements/06-12_learner_separations.md) · [Proof](../proofs/06-12_learner_separations.md) · [Experiments](../experiments/06-12_learner_separations.md) · [中文](06-12_learner_separations_ZH.md)

**What is tested here.** Part (a) is tested by [Check 6.6](../experiments/06-12_learner_separations.md#c6-6): with the same demonstrations, the always-clipped learner's error falls like $`K^{-1/2}`$ (from $`0.091`$ at $`K=1`$ to $`0.00024`$ at $`K=10^5`$), while SGD's worst-case error stays at the floor $`0.45`$ for every $`K`$. Part (b) rests on the weights checked in [Check 6.4](../experiments/06-11_learner_weights_sgd_warm_adam.md#c6-4) and the allocation checked in [Check 6.5](../experiments/06-10_water_filling_allocation.md#c6-5). Verdict: PASS.

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

**Statement under test ([Corollary 6.12](../statements/06-12_learner_separations.md#s6-12)).**

- **(a) Exact separation by scale invariance.** Let $`\theta=(\alpha,b)`$ with $`b>0`$, let the oracle consumed gradient be $`b\,G_\alpha(Y)`$ (for example a feedback weight or advantage scale), and let the demonstration law depend on $`\alpha`$ only.
  - For SGD with always-active norm clipping at level $`c_{\rm clip}`$ ($`\lVert bG_\alpha(Y)\rVert>c_{\rm clip}`$ almost surely), and for normalized SGD, the oracle mean increment does not depend on $`b`$, so $`\Psi`$ depends on $`\alpha`$ only and $`c(\theta)`$ has no component along $`b`$. If the demonstrations' Fisher information about $`\alpha`$ is non-degenerate, the error tends to zero as $`K\to\infty`$ ([Theorem 6.5](../statements/06-05_achievability.md#s6-5) with $`n=0`$, applied to the model in $`\alpha`$; realizability holds with any scale that keeps clipping active).
  - For plain SGD, $`c(\theta)`$ has the $`b`$-component $`-\eta\sum_t\langle g_t,\mathbb E\,G_\alpha\rangle`$; when it is nonzero, the demonstrations carry no information about $`b`$ and, by [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7), no $`K`$ suffices.
  - The same demonstrations are therefore sufficient for one learner and insufficient for the other. When clipping is active only with probability $`p_{\rm act}<1`$, [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9) puts each clipped mean consumed gradient within $`c_{\rm clip}(1-p_{\rm act})`$ of the $`b`$-free vector $`c_{\rm clip}\,\mathbb E[G_\alpha/\lVert G_\alpha\rVert]`$, so the separation degrades gracefully.
- **(b) Quantitative separation by long-tail coverage.** By [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10) and [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11)(d): when the student's state distribution has many low-frequency states, all in the frequent regime of warm Adam, and the teacher rarely reaches them, the share of the information requirement carried by those states is larger under warm Adam than under SGD, and warm Adam needs correspondingly more demonstrations or queries there for the same relative accuracy ([Example 6.13](../statements/06-13_long_tail_example.md#s6-13)).

<a id="c6-6"></a>

## Check 6.6 · The same demonstrations suffice for clipped SGD and not for SGD (Corollary 6.12(a))

**Verdict.** PASS. The clipped learner's increment is exactly scale invariant and its error decays like $`K^{-1/2}`$; SGD's worst-case error never drops below half the functional gap.

**Why.** [Corollary 6.12](../statements/06-12_learner_separations.md#s6-12)(a) claims an exact separation: demonstrations that reveal $`\alpha`$ but not the scale $`b`$ suffice for an always-clipped learner and are useless for SGD. Dependence of the clipped increment on $`b`$, or an SGD error below the floor, would falsify it.

**Setup.**

- $`\alpha=0.65`$; gradient support $`\lbrace0.2,-0.8\rbrace`$ scaled by $`b\in\lbrace1,3\rbrace`$; clipping threshold $`0.1`$, so clipping is always active; $`\eta R=1`$.
- Demonstrations observe labels only, so their law is $`\mathrm{Binomial}(K,\alpha)`$ and does not depend on $`b`$.
- $`K=1,10,\dots,10^5`$; 60,000 replicates. The clipped functional and a midpoint-scale SGD estimator are evaluated in both worlds.

**Result.**

| $`K`$ | Clipped learner MAE | RMS bound | SGD worst-case error | Floor $`\lvert\Delta\Psi\rvert/2`$ |
|---|---|---|---|---|
| 1 | $`0.0910`$ | $`0.0954`$ | $`1.0448`$ | $`0.45`$ |
| 10 | $`0.0247`$ | $`0.0302`$ | $`0.4714`$ | $`0.45`$ |
| 100 | $`0.00762`$ | $`0.00954`$ | $`0.4502`$ | $`0.45`$ |
| 1,000 | $`0.00240`$ | $`0.00302`$ | $`0.4501`$ | $`0.45`$ |
| 10,000 | $`0.000763`$ | $`0.000954`$ | $`0.4500`$ | $`0.45`$ |
| 100,000 | $`0.000241`$ | $`0.000302`$ | $`0.4500`$ | $`0.45`$ |

- Scale invariance of the clipped increment: exact in all 12 comparisons.
- The SGD estimator used here approaches the floor; the floor itself holds for every estimator because the two worlds have identical demonstration laws.

**Why it supports Corollary 6.12(a).** The separation is exact: with the same demonstrations, one learner's error vanishes at the parametric rate and the other's is bounded away from zero for every $`K`$. Which demonstrations are sufficient depends on the learner.

![Figure 6.2](../figures/fig6_2_learner_weights.png)

*Figure 6.2, panel (d) concerns this page.* Error against $`K`$ for always-clipped SGD, which falls like $`K^{-1/2}`$ below its RMS bound, and SGD's worst case over the two worlds, which stays at the aliasing floor $`0.45`$. Panels (a)–(c) belong to Propositions 6.10 and 6.11.

### [Check 6.4](../experiments/06-11_learner_weights_sgd_warm_adam.md#c6-4) · Learner weights: SGD scales like d², warm Adam like d

- **What it tests for Corollary 6.12(b).** The learner weights that make the long tail matter more to warm Adam than to SGD.
- **Setup.** State-owned scalar coordinates, Bernoulli teacher labels, common-random-number sensitivities for SGD, always-clipped SGD and Adam ($`\beta_1=0.9`$) across frequent, isolated, transition and short-horizon regimes.
- **Key numbers.** Sensitivity slope in $`d`$: SGD $`1.00`$ (weight slope $`2`$), frequent-regime Adam $`0.489`$ (weight slope $`0.98`$); the frequent-regime closed form is within 0.6–7.1%.
- **Verdict.** PASS: the weights scale as part (b) requires.

### [Check 6.5](../experiments/06-10_water_filling_allocation.md#c6-5) · Water-filling minimizes the realized error

- **What it tests for Corollary 6.12(b).** That allocating queries with the weights of the wrong learner costs real accuracy.
- **Setup.** 16 random profiles with 6–10 states, weights linear in $`d`$ as for frequent-regime Adam, 800 teacher visits and 4,000 queries, 60,000 replicates per allocation.
- **Key numbers.** Water-filling with SGD-type weights is $`1.078`$–$`1.315`$ times worse in realized mean absolute error than water-filling with the correct weights.
- **Verdict.** PASS: the allocation, and hence the information requirement, is learner-relative.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C3 --output /tmp/v016_C3.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

The first command reruns the check group(s) of this page into a separate file; `theory/v016_checks/check_v016.py` without `--only` regenerates the full result file (about 104 s). All checks run in float64 on CPU with seeds `numpy.SeedSequence([160160, group, subcase, ...])` (PCG64).

## Artifacts

- `theory/v016_checks/results_v016.json`: claim ids `v016:C3.separation`, `v016:C3.sensitivity`, `v016:C3.allocation`, each with assertions, worst residuals, failures and data rows.
- `theory/v016_checks/results_v016_raw.json.gz`: complete rows; not tracked by git, its SHA-256 is recorded under `raw_artifact` in the main file.
- `theory/v016_checks/SPEC.md` (sections `v016:C3`) and `theory/v016_checks/check_v016.py`.
- `figures/fig6_2_learner_weights.png` (PDF alongside)
