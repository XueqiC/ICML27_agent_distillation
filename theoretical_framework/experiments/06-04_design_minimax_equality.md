# Lemma 6.4 · Design minimax equality

*Experiments*

[← Framework](../README.md) · [Statement](../statements/06-04_design_minimax_equality.md) · [Proof](../proofs/06-04_design_minimax_equality.md) · [Experiments](../experiments/06-04_design_minimax_equality.md) · [中文](06-04_design_minimax_equality_ZH.md)

**What is tested here.** Lemma 6.4 is tested by [Check 6.1](../experiments/06-04_design_minimax_equality.md#c6-1): on 216 instances, including singular and non-identifiable ones, the c-optimal design value and the dual information norm agree to a relative gap of at most $`4.5\times10^{-13}`$. Verdict: PASS.

**Setting used on this page.**

- **Environments.** A family $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown ([Definition 6.1](../statements/06-01_environment_family_and_distiller.md#s6-1)).
- **Oracle feedback** $`P^*_\theta`$: the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data.** $`K`$ i.i.d. demonstrations with law $`P^D_\theta`$, and $`n`$ queries that may be adaptive: a query point $`X_i`$ in the query space $`\mathcal X_Q`$ has outcome $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$. The law of the whole transcript is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from the transcript and private randomness (whose law does not depend on $`\theta`$) to a feedback law $`Q`$. Replay, rewriting and synthesis are distillers ([Definition 1.2](../README.md#s1-2)).
- **Audited states** $`z_1,\dots,z_T`$: fixed complete learner states, i.e. parameters together with the optimizer state ([Definition 1.3](../README.md#s1-3)), for example the states reference training reaches at checkpoints.
- **Mean increment** $`\mu^P_t(z)`$: the expected change of the complete learner state when one message from the law $`P`$ is consumed at $`z`$ ([Definition 1.3](../README.md#s1-3)). For the oracle, $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$, and the **learner Jacobian** is $`J_t=D_\theta\mu^*_t`$.
- **Task directions** $`g_t=\nabla V_{t+1}(z_t)`$: gradients of the continuation value of the declared reference tail ([Definition 1.4](../README.md#s1-4)); they do not depend on $`\theta`$.
- **Learner-relative functional** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$, with gradient $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$, and **first-order fidelity error** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$ ([Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2)). By [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) this is the first-order part of the gap in final task score between the oracle feedback and $`Q`$ at the audited states.
- **Fisher informations** $`I_D(\theta)`$ of one demonstration and $`I_Q(\theta;x)`$ of one query at $`x\in\mathcal X_Q`$. For a direction $`h\in\mathbb R^p`$, $`\mathbb I_{K,n}(\theta;h)=K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h`$.
- **Learner-relative information norm** $`\mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\lvert c(\theta)^\top h\rvert/\sqrt{\mathbb I_{K,n}(\theta;h)}`$, with $`0/0=0`$ and $`a/0=\infty`$ for $`a>0`$.
- **Growth condition (G)**: there are $`N\to\infty`$ and $`0<a\le b<\infty`$ with $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$ for every unit $`h`$.
- **Designs.** An approximate design $`\xi`$ is a probability measure on the query space; it spreads the $`n`$ queries over query points. $`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$ is the average query information and $`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$ the total information of demonstrations plus queries.

**Statement under test ([Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4)).** Assume that $`\mathcal X_Q`$ is a compact metric space and that $`x\mapsto I_Q(\theta;x)`$ is continuous. Let $`\xi`$ range over the Borel probability measures on $`\mathcal X_Q`$ (approximate designs), put $`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$ and $`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$, and assume $`A(\xi)\succ0`$ for every $`\xi`$. Then

```math
\inf_\xi c^\top A(\xi)^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h}=\mathfrak n_{K,n}^2 ,
```

and the infimum is attained by some design $`\xi^*`$, the **c-optimal design** for the target vector $`c=\sum_tJ_t^*g_t`$. For $`n=0`$ and singular $`I_D`$, the identity holds with $`c^\top(KI_D)^+c`$ when $`c\in\mathrm{range}(I_D)`$, and with $`+\infty`$ on both sides otherwise.

<a id="c6-1"></a>

## Check 6.1 · The c-optimal design equals the dual information norm (Lemma 6.4)

**Verdict.** PASS. Primal and dual agree on all 216 instances; the worst relative gap is $`4.5\times10^{-13}`$ against a tolerance of $`10^{-6}`$.

**Why.** [Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4) says that the best approximate design value $`\inf_\xi c^\top A(\xi)^{-1}c`$ equals $`\mathfrak n_{K,n}^2`$, a supremum over directions. The two sides are computed by unrelated convex programs; a gap larger than solver accuracy, or a case where one side is finite and the other infinite, would falsify the lemma or its conventions for singular data.

**Setup.**

- Gaussian linear model: $`\theta\in\mathbb R^p`$ with $`p\in\lbrace2,3,5,8\rbrace`$; a demonstration observes $`A_D\theta+N(0,I)`$, so $`I_D=A_D^\top A_D`$; a query at $`x`$ observes $`f_x^\top\theta+N(0,1)`$, so $`I_Q(x)=f_xf_x^\top`$, with $`\lvert\mathcal X_Q\rvert\in\lbrace4,16\rbrace`$.
- Three demonstration families for each $`(p,\lvert\mathcal X_Q\rvert)`$: full rank; an exact null column with a nonzero component of $`c`$ along it (aliased); an exact null column with $`c`$ in the identifiable range (singular but estimable).
- $`K\in\lbrace1,10,100\rbrace`$, $`n\in\lbrace0,10,100\rbrace`$: 216 instances, 864 assertions.
- Primal: minimize $`c^\top M(w)^{-1}c`$ over the simplex of design weights $`w`$ by SLSQP with analytic gradients, plus a check of the simplex KKT (Elfving) condition ([Elfving, 1952](#ref-elfving1952optimum)).
- Dual: minimize $`K\,h^\top I_Dh+n\,t^2`$ subject to $`c^\top h=1`$ and $`t\ge\pm f_x^\top h`$, a convex quadratic program. Where SLSQP reports a status failure, an independent trust-constr solve with the exact Hessian and an active-face KKT linear solve must certify stationarity, feasibility and nonnegative multipliers to $`10^{-9}`$.
- $`n=0`$: spectral range tests; a non-identifiable $`c`$ gives $`+\infty`$ on both sides, an identifiable singular $`c`$ uses the pseudo-inverse.

**Result.**

| Assertion | Instances | Worst residual | Tolerance |
|---|---|---|---|
| Primal–dual relative gap | 216 | $`4.5\times10^{-13}`$ | $`10^{-6}`$ |
| Simplex KKT (Elfving) relative gap | 216 | $`1.96\times10^{-7}`$ | $`10^{-6}`$ |
| Dual feasibility | 216 | $`1.0\times10^{-13}`$ | $`10^{-8}`$ |
| Solver success | 216 | 0 failures | 0 |

- 136 dual solves succeed directly with SLSQP. In 8 cases SLSQP reports a status failure despite accurate values; trust-constr and the active-face KKT certificate then succeed. The 72 cases with $`n=0`$ use the spectral path.
- The 24 aliased cases with $`n=0`$ give $`+\infty`$ on both sides, as the lemma's convention requires.

**Why it supports Lemma 6.4.** Two independent formulations, one over designs and one over directions, give the same number to near machine precision on full-rank, singular and non-identifiable instances. The lemma's exchange of $`\sup`$ and $`\inf`$ is therefore not an artefact of a particular solver. The check uses rank-one query information; the lemma itself does not need it.

![Figure 6.1](../figures/fig6_1_information_bound.png)

*Figure 6.1, panel (b) concerns this page.* The c-optimal design value against the dual information objective (Lemma 6.4), on the diagonal; the 24 settings that are infinite on both sides are counted, not plotted. Panels (a) and (c) belong to Theorems 6.3/6.5 and Corollary 6.7.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C2 --output /tmp/v016_C2.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

The first command reruns the check group(s) of this page into a separate file; `theory/v016_checks/check_v016.py` without `--only` regenerates the full result file (about 104 s). All checks run in float64 on CPU with seeds `numpy.SeedSequence([160160, group, subcase, ...])` (PCG64). The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)) for linear algebra and random numbers, and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)) for optimization and statistical functions.

## Artifacts

- `theory/v016_checks/results_v016.json`: claim ids `v016:C2.design`, each with assertions, worst residuals, failures and data rows.
- `theory/v016_checks/results_v016_raw.json.gz`: complete rows; not tracked by git, its SHA-256 is recorded under `raw_artifact` in the main file.
- `theory/v016_checks/SPEC.md` (sections `v016:C2`) and `theory/v016_checks/check_v016.py`.
- `figures/fig6_1_information_bound.png` (PDF alongside)
