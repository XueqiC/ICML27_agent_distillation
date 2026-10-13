# Proposition 6.11 · Learner weights of SGD and warm Adam

*Experiments*

[← Framework](../README.md) · [Statement](../statements/06-11_learner_weights_sgd_warm_adam.md) · [Proof](../proofs/06-11_learner_weights_sgd_warm_adam.md) · [Experiments](../experiments/06-11_learner_weights_sgd_warm_adam.md) · [中文](06-11_learner_weights_sgd_warm_adam_ZH.md)

**What is tested here.** Proposition 6.11 is tested by [Check 6.4](../experiments/06-11_learner_weights_sgd_warm_adam.md#c6-4): measured sensitivity slopes in $`d`$ are $`1.00`$ for SGD, $`0.489`$ for frequent-regime Adam (weight slope $`0.98`$) and $`0.970`$ for isolated-regime Adam. The frequent-regime closed form, second-moment term included, is within 0.6–7.1% (14–21% without that term); the isolated-regime form is within 8.6%; near $`d\approx1-\beta_2`$ neither applies, as stated. Verdict: PASS.

**Setting used on this page.**

- **Agent states** $`s\in\mathcal S`$, a finite set; on this page $`s`$ denotes an agent state ([Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9)).
- **Per-state oracle laws.** At state $`s`$ the oracle message is $`Y\sim P^*_s`$ (for example the teacher's action or correction there), on a finite alphabet; the laws vary independently across states, so nothing is assumed to generalize between states.
- **Influence function** $`f_s(y)`$ of $`\Psi`$ at $`s`$: the derivative of $`\Psi`$ when $`P^*_s`$ is moved towards the point mass at $`y`$; it has mean zero under $`P^*_s`$.
- **Learner-relative weight** $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$: how much one more oracle sample at $`s`$ is worth for estimating $`\Psi`$.
- **Counts.** State $`s`$ receives $`N_s=k_s+n_s`$ oracle samples: $`k_s`$ from demonstrations (in expectation $`KH\,d_T(s)`$, with $`H`$ the trajectory length and $`d_T`$ the teacher's state-visitation distribution) and $`n_s`$ from queries, $`\sum_sn_s=n`$.
- **State-owned coordinates.** Each agent state $`s`$ has its own scalar parameter $`\theta_s`$; each training step visits one state $`s_k\sim d`$ i.i.d., where $`d(s)`$ is the student's visitation probability, and consumes one oracle message there.
- **Gradients.** The consumed gradient $`g=G(s,Y)`$ does not depend on the parameters (linear-response regime); at state $`s`$, $`m_1=\mathbb Eg`$ and $`m_2=\mathbb Eg^2`$.
- **Outcome** $`F=\langle\zeta,\theta_R\rangle`$ after $`R`$ steps with learning rate $`\eta`$; $`\Psi`$ is the expected outcome change.
- **Adam** with momentum $`\beta_1`$, second-moment decay $`\beta_2`$, $`\beta_1<\sqrt{\beta_2}`$ and negligible $`\epsilon`$ ([Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5) gives the exact Adam increment).

**Statement under test ([Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11)).** Besides [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9), assume:

- every agent state owns its own parameter coordinate (state-owned coordinates);
- each training step draws one state $`s_k\sim d`$ i.i.d. and one oracle message there;
- the consumed gradient does not depend on the parameters (the linear-response regime): $`g=G(s,Y)`$; write $`m_1=\mathbb Eg`$ and $`m_2=\mathbb Eg^2`$ at state $`s`$;
- the outcome is $`F=\langle\zeta,\theta_R\rangle`$, and $`\Psi`$ is the expected outcome change $`\mathbb E F(\theta_R)-F(\theta_0)`$, a functional of $`(P^*_s)_s`$;
- Adam has $`\epsilon`$ negligible relative to $`\sqrt v`$ and $`\beta_1<\sqrt{\beta_2}`$.

Then, to leading order as $`R\to\infty`$ in each stated regime:

- **(a) SGD.** $`f_s^{\rm SGD}(y)=-\eta R\,d(s)\,\zeta_s\,(g-m_1)`$, so $`W_s^{\rm SGD}=\eta^2R^2\zeta_s^2\,d(s)^2\,\mathrm{Var}(g)`$.
- **(b) Warm Adam, frequent regime.** Assume many visits within the effective averaging window of the second moment: $`d(s)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$ with bias correction, and in addition $`t\gg1/(1-\beta_2)`$ without it. Then

```math
f_s^{\rm Adam}(y)=-\eta R\sqrt{d(s)}\,\zeta_s\Big[\frac{g-m_1}{\sqrt{m_2}}-\frac{m_1(g^2-m_2)}{2m_2^{3/2}}\Big],\qquad W_s^{\rm Adam}=\eta^2R^2\zeta_s^2\,d(s)\,\mathrm{Var}\Big(\frac{g}{\sqrt{m_2}}-\frac{m_1g^2}{2m_2^{3/2}}\Big).
```

  The second term in the bracket is the second-moment channel: a message that raises $`m_2`$ lowers every later step size.

- **(c) Warm Adam, isolated regime.** Assume $`d(s)\ll\min\lbrace1-\beta_1,1-\beta_2\rbrace`$, and either no bias correction or $`t\gg1/(1-\beta_2)`$. Each visit then meets $`m\approx0`$ and $`v\approx0`$ and moves the coordinate by $`-\eta\,\kappa_{\rm iso}\,\mathrm{sign}(g)`$, with

```math
\kappa_{\rm iso}=\frac{1-\beta_1}{\sqrt{1-\beta_2}\,\big(1-\beta_1/\sqrt{\beta_2}\big)} .
```

  Hence $`f_s(y)=-\eta R\,d(s)\,\zeta_s\,\kappa_{\rm iso}\,(\mathrm{sign}\,g-\mathbb E\,\mathrm{sign}\,g)`$ and $`W_s\propto d(s)^2\,\mathrm{Var}(\mathrm{sign}\,g)`$: the same scaling in $`d`$ as SGD, but the message enters only through its sign. With bias correction, a visit at step $`t\gg1/(1-\beta_1)`$ to a coordinate whose $`m`$ and $`v`$ are still zero (for example its first visit) moves it by $`-\eta\,\kappa_{\rm iso}\sqrt{1-\beta_2^{\,t}}\,\mathrm{sign}(g)`$; the constant changes, the scaling in $`d`$ does not.

- **(d) Consequence.** Let $`s`$ be a frequent and $`s'`$ a rarer state, $`d(s)>d(s')`$, both in the frequent regime and with the same variance factors. The weight ratio $`W_{s'}/W_s`$ is $`(d(s')/d(s))^2`$ under SGD and $`d(s')/d(s)`$ under warm Adam. Relative to the frequent state, warm Adam therefore weights the rarer state $`d(s)/d(s')`$ times more heavily than SGD does. Demonstrations that cover the frequent states and miss the student's long tail can suffice for SGD and not for warm Adam.

<a id="c6-4"></a>

## Check 6.4 · Learner weights: SGD scales like d², warm Adam like d (Proposition 6.11)

**Verdict.** PASS. Every asserted slope lies within its bootstrap 95% interval enlarged by the predeclared $`0.1`$. Frequent-regime Adam has sensitivity slope $`0.489`$ in $`d`$, hence weight slope $`0.98`$, against $`1.00`$ (weight slope $`2.0`$) for SGD. The closed forms are accurate inside their regimes and fail, as stated, near $`d\approx1-\beta_2`$.

**Why.** [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11) predicts how strongly one more oracle sample at a state moves the outcome, for each optimizer. The weights decide where queries should go ([Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10)) and drive the separation between SGD and Adam. Measured sensitivities with the wrong slope, or a closed form that misses the second-moment term, would falsify the proposition.

**Setup.**

- Each target state owns a scalar coordinate; one state is visited per step; teacher label $`y\sim\mathrm{Bernoulli}(\pi)`$ with $`\pi=0.35`$; gradient $`g=c_0-y`$ with $`c_0=0.2`$; $`\zeta=1`$, $`\eta=0.01`$, $`\epsilon=10^{-12}`$.
- Learners: SGD; SGD always clipped to $`0.1`$ (norm clipping, [Pascanu et al., 2013](#ref-pascanu2013difficulty)); Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) with $`\beta_1=0.9`$. They share visits and label randomness.
- Sensitivity $`\mathrm d\,\mathbb E\theta_{s,R}/\mathrm d\pi_s`$ by common-random-number central differences ([Glasserman, 2004](#ref-glasserman2004monte); $`\delta=0.02`$; $`\delta=0.04`$ as a diagnostic), with the perturbation applied after a common burn-in.

| Regime | $`\beta_2`$ | Bias correction | $`d(s)`$ (log-spaced) | $`d/(1-\beta_2)`$ | Burn-in | $`R`$ | Replicates |
|---|---|---|---|---|---|---|---|
| frequent | 0.999 | no | 0.005–0.4, 7 values | 5–400 | 20,000 | 30,000 | 384 |
| isolated | 0.9 | no | 0.0005–0.01, 6 values | 0.005–0.1 | 2,000 | 60,000 | 384 |
| transition | 0.999 | no | 0.0003–0.02, 7 values | 0.3–20 | 20,000 | 30,000 | 384 |
| short horizon | 0.999 | yes | 0.0005–0.01, 6 values | 0.5–10 (0.1–2 visits in $`R`$) | 0 | 200 | 20,000 |

- Asserted slopes: $`1`$ for SGD and clipped SGD in every regime; $`1/2`$ for frequent Adam; $`1`$ for isolated and short-horizon Adam. The transition slope is descriptive.

**Result.**

| Regime | Learner | Slope in $`d`$ [bootstrap 95% CI] | Closed form vs measured |
|---|---|---|---|
| frequent | Adam | $`0.4887`$ $`[0.4827,0.4948]`$ | frequent formula: 0.6–7.1% |
| frequent | SGD / clipped SGD | $`1.0029`$ $`[0.9967,1.0093]`$ | exact linear formula |
| isolated | Adam | $`0.9699`$ $`[0.9443,0.9969]`$ | isolated formula, $`\kappa_{\rm iso}=6.16`$: +0.2% to −8.6% |
| isolated | SGD / clipped SGD | $`0.9970`$ $`[0.9715,1.0234]`$ | exact linear formula |
| transition | Adam | $`0.6252`$ $`[0.5952,0.6610]`$ | see below |
| transition | SGD / clipped SGD | $`1.0185`$ $`[0.9896,1.0518]`$ | exact linear formula |
| short horizon | Adam | $`0.9166`$ $`[0.8599,0.9786]`$ | see below |
| short horizon | SGD / clipped SGD | $`0.9722`$ $`[0.9172,1.0328]`$ | exact linear formula |

- **Second-moment term.** In the frequent regime, dropping the second term of [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11)(b) overpredicts the measured sensitivity by 14% (at $`d=0.005`$) to 21% (at $`d=0.4`$). The largest error of the full formula, 7.1%, is at the smallest $`d`$, where only 5 visits fall in a second-moment window.
- **Isolated regime.** The error of the isolated formula grows from 0.2% at $`d/(1-\beta_2)=0.005`$ to 8.6% at $`0.1`$, as visits start to overlap.
- **Transition ($`d\approx1-\beta_2`$).** The frequent formula is off by −46% at $`d/(1-\beta_2)=0.3`$ and −17% at $`0.6`$, and within 10% from $`1.2`$ upward. The isolated formula is off by −19% at $`0.3`$ and −12.5% at $`0.6`$, and much worse above (−22% to −81%). The measured slope $`0.625`$ lies between the two regimes. The switch between closed forms sits at $`d/(1-\beta_2)\approx1`$.
- **Short horizon with bias correction.** The slope $`0.917`$ is consistent with $`1`$ within the predeclared tolerance, but its 95% interval misses $`1`$ by $`0.021`$, the worst slope residual of the check. The magnitude is 24–33% of the uncorrected isolated formula. This is what [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11)(c) predicts for bias correction: a first visit at step $`t`$ carries the factor $`\sqrt{1-\beta_2^{\,t}}`$, whose average over $`t\le200`$ is $`0.29`$ (a back-of-envelope computation, not part of the check).
- **Linear learners.** Monte Carlo means agree with the exact formula within $`1.52`$ standard errors on all 52 comparisons (guard $`4.5`$). All 12 rows have positive sensitivities.
- **Finite-difference step.** Switching $`\delta`$ from $`0.02`$ to $`0.04`$ changes the sensitivities by at most $`1.24`$ (frequent Adam at $`d=0.4`$, 0.4% of the value).

![Figure 6.2: learner weights, query allocation and the scale-invariance separation](../figures/fig6_2_learner_weights.png)

*Figure 6.2.* (a) Measured sensitivity against visitation $`d(s)`$ on log-log axes, with pointwise 95% intervals and fitted against predicted slopes in the legend: SGD and clipped SGD $`1`$, frequent-regime Adam $`1/2`$, isolated-regime and short-horizon bias-corrected Adam $`1`$. Horizons and burn-in differ between regimes, so absolute heights are not comparable across series. (b) Realized mean absolute error per profile for water-filling, uniform, proportional-to-$`d`$ and SGD-weight allocations; water-filling is lowest in every profile. (c) Predicted against realized error ratios with familywise intervals, on the diagonal (Proposition 6.10). (d) Error against $`K`$ for always-clipped SGD, which falls like $`K^{-1/2}`$ below its RMS bound, and SGD's worst case over the two worlds, which stays at the aliasing floor $`0.45`$ (Corollary 6.12(a)).

**Why it supports Proposition 6.11.**

- The slopes separate the learners exactly as predicted: SGD weights scale like $`d^2`$, frequent-regime Adam weights like $`d`$, and isolated-regime Adam returns to $`d^2`$ with a sign-only dependence.
- The closed form with the second-moment term is accurate where its regime holds, and the term is necessary: without it the error rises from 0.6–7.1% to 14–21%.
- The deviations are where the proposition says its closed forms stop: near $`d\approx1-\beta_2`$, and for the constant (not the slope) under bias correction with a short horizon.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C3 --output /tmp/v016_C3.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

The first command reruns the check group(s) of this page into a separate file; `theory/v016_checks/check_v016.py` without `--only` regenerates the full result file (about 104 s). All checks run in float64 on CPU with seeds `numpy.SeedSequence([160160, group, subcase, ...])` (PCG64). The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)) for linear algebra and random numbers, and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)) for optimization and statistical functions.

## Artifacts

- `theory/v016_checks/results_v016.json`: claim ids `v016:C3.sensitivity`, each with assertions, worst residuals, failures and data rows.
- `theory/v016_checks/results_v016_raw.json.gz`: complete rows; not tracked by git, its SHA-256 is recorded under `raw_artifact` in the main file.
- `theory/v016_checks/SPEC.md` (sections `v016:C3`) and `theory/v016_checks/check_v016.py`.
- `figures/fig6_2_learner_weights.png` (PDF alongside)
