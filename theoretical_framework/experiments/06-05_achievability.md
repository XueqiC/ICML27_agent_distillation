# Theorem 6.5 · Achievability; tight to constants in the scalar case

*Experiments*

[← Framework](../README.md) · [Statement](../statements/06-05_achievability.md) · [Proof](../proofs/06-05_achievability.md) · [Experiments](../experiments/06-05_achievability.md) · [中文](06-05_achievability_ZH.md)

**What is tested here.** Theorem 6.5, with Theorem 6.3, is tested by [Check 6.2](../experiments/06-05_achievability.md#c6-2): on 192 identifiable instances the plug-in risk under the integer-rounded c-optimal design lies between $`0.789\,\mathfrak n`$ and $`0.850\,\mathfrak n`$ (median $`0.798\approx\sqrt{2/\pi}`$), inside the predicted band $`[\mathfrak n/4,\mathfrak n]`$. Verdict: PASS.

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

**Statement under test ([Theorem 6.5](../statements/06-05_achievability.md#s6-5)).** Assume (G), the regularity of [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)(b), and:

- **(R) Realizability.** For every $`\theta'\in\Theta`$ the distiller can emit the oracle law $`P^*_{\theta'}`$ at the audited states, for example by running a simulator or the teacher model at $`\theta'`$, or by reweighting messages obtained from queries.
- **(I) Identifiability.** Some design $`\xi_0`$ has $`\lambda_{\min}(M(\xi_0))=\lambda_0>0`$.
- **(S) Smoothness.** $`\Psi\in C^2`$ on a convex neighbourhood $`U`$ of $`\theta_0`$, with $`\lVert\nabla^2\Psi\rVert\le H_\Psi`$ on $`U`$.
- **(T) Tail control.** Uniformly over $`\theta`$ in the shrinking neighbourhood of (E),

```math
\mathbb E_\theta\big[\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert\,\mathbf 1\lbrace\hat\theta\notin U\rbrace\big]=o\big(N^{-1/2}\big).
```

  The Taylor bound of (S) controls the error only when the segment from $`\theta`$ to $`\hat\theta`$ stays in $`U`$; (T) controls the rest. It holds, for example, when $`\Psi`$ is bounded on $`\Theta`$: the neighbourhood of (E) shrinks to the interior point $`\theta_0`$, so for some $`r>0`$ Chebyshev's inequality and (E) give $`\mathsf P_\theta(\hat\theta\notin U)\le C/(r^2\lambda_{\min}(A(\xi_n)))=o(N^{-1/2})`$ (the proof shows $`\lambda_{\min}(A(\xi_n))\gg\sqrt N`$). Alternatively, use an estimator that takes values in $`U`$ by construction and is verified to satisfy (E).
- **Design.** Use the regularized c-optimal design $`\xi_n=(1-\varepsilon_n)\xi^*+\varepsilon_n\xi_0`$, with $`\xi^*`$ from [Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4), $`\varepsilon_n\to0`$ and $`\varepsilon_n\sqrt n\to\infty`$ (for example $`\varepsilon_n=n^{-1/4}`$).
- **(E) Efficient two-stage estimation.** $`\xi_n`$ is computed at a consistent pilot estimate obtained from $`o(n)`$ queries and rounded to integer counts. The final estimator $`\hat\theta`$ satisfies, uniformly over $`\theta`$ in a shrinking neighbourhood of $`\theta_0`$ and with $`\xi_n`$ evaluated at the true $`\theta`$,

```math
\mathbb E_\theta\big[(c(\theta)^\top(\hat\theta-\theta))^2\big]\le c(\theta)^\top A(\xi_n)^{-1}c(\theta)\,(1+o(1)),\qquad \mathbb E_\theta\lVert\hat\theta-\theta\rVert^2\le\frac{C}{\lambda_{\min}(A(\xi_n))} .
```

Let the distiller output $`Q=P^*_{\hat\theta}`$. Then

```math
\mathbb E_\theta\mathrm{Err}_\theta(Q)\le\mathfrak n_{K,n}(\theta)\,\big(1+o(1)\big)+o\big(N^{-1/2}\big),
```

and the last term is $`o(\mathfrak n_{K,n}(\theta))`$ whenever $`c(\theta)\neq0`$. Together with [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)(c), the smallest achievable first-order error is, in the local minimax sense, between $`\tfrac14\mathfrak n_{K,n}`$ and $`\mathfrak n_{K,n}`$ up to factors $`1\pm o(1)`$. The optimal queries follow the c-optimal design whose target vector is $`c=\sum_tJ_t^*g_t`$.

<a id="c6-2"></a>

## Check 6.2 · The achieved risk lies between 𝔫/4 and 𝔫 (Theorems 6.3 and 6.5)

**Verdict.** PASS. On all 192 identifiable instances the Monte Carlo mean absolute error of the efficient plug-in estimator, under the integer-rounded c-optimal design, lies in $`[0.789,\,0.850]\cdot\mathfrak n_{K,n}`$. No two-point lower bound exceeds the achieved risk.

**Why.** [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3) puts every distiller at or above $`\mathfrak n/4`$, and [Theorem 6.5](../statements/06-05_achievability.md#s6-5) says the c-optimal plug-in achieves $`\mathfrak n(1+o(1))`$. A ratio outside $`[1/4,1]`$ would falsify the pair. Integer rounding of the design is a finite-$`n`$ effect that the asymptotic theorem does not cover, so it is measured rather than assumed away.

**Setup.**

- The 216 instances of [Check 6.1](../experiments/06-04_design_minimax_equality.md#c6-1) minus the 24 non-identifiable ones: 192 instances, 960 assertions.
- The approximate c-optimal design ([Elfving, 1952](#ref-elfving1952optimum)) is rounded to integer counts by largest remainders without changing $`n`$; the rounding loss is the relative increase of the variance $`c^\top A^{-1}c`$.
- Independent Gaussian noise is generated for the demonstration and query sufficient statistics, and the efficient linear estimator of $`c^\top\theta`$ is computed: 30,000 replicates per instance. This simulates observations rather than sampling the error from its predicted variance.
- Recorded alongside: the exact Gaussian mean absolute error $`\sqrt{2/\pi}\sqrt{c^\top A_{\rm int}^{-1}c}`$; the Pinsker two-point bound of [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)(a) at the least favourable displacement $`\tau=\mathbb I^{-1/2}`$; the exact Gaussian total-variation two-point bound at the same displacement.

**Result.**

| Quantity | Value |
|---|---|
| Achieved risk $`/\,\mathfrak n_{K,n}`$: min / median / max | $`0.7885`$ / $`0.7982`$ / $`0.8504`$ |
| $`\sqrt{2/\pi}`$ (exact design, no rounding) | $`0.7979`$ |
| Largest rounding loss (the instance with ratio $`0.850`$: $`n=10`$ queries over 16 points) | 12.3% of the variance |
| Smallest margin above $`\mathfrak n/4`$ / below $`\mathfrak n`$ (absolute) | $`0.0204`$ / $`0.0075`$ |
| Pinsker two-point bound $`/\,\mathfrak n`$ | $`0.2500`$ in every instance |
| Exact Gaussian TV two-point bound $`/\,\mathfrak n`$ (same displacement) | $`0.3085=1-\Phi(1/2)`$ in every instance |
| Achieved risk $`/`$ Pinsker bound, minimum | $`3.15`$ |
| Monte Carlo vs exact Gaussian MAE, worst | $`2.91`$ standard errors (99.9% pointwise tolerance $`3.29`$) |

![Figure 6.1: information bound, design duality and aliasing](../figures/fig6_1_information_bound.png)

*Figure 6.1.* (a) Achieved plug-in risk divided by the information norm $`\mathfrak n_{K,n}`$ for all 192 identifiable settings, with the factors $`1/4`$ and $`1`$ of Theorems 6.3 and 6.5 and the Gaussian value $`\sqrt{2/\pi}`$; the few points above $`\sqrt{2/\pi}`$ are settings where integer rounding of a small query budget inflates the variance. (b) The c-optimal design value against the dual information objective (Lemma 6.4) on the diagonal; the 24 settings that are infinite on both sides are counted, not plotted. (c) The worst-error floor $`\lvert c^\top h\rvert/2`$ of eight aliased instances against $`K`$, flat in every instance (Corollary 6.7).

**Why it supports Theorems 6.3 and 6.5.**

- The achieved risk sits near $`\sqrt{2/\pi}\,\mathfrak n`$, the exact Gaussian value, so the upper constant $`1`$ of [Theorem 6.5](../statements/06-05_achievability.md#s6-5) holds with room, even after integer rounding.
- The Pinsker two-point bound equals $`\mathfrak n/4`$ exactly at the theorem's displacement, so the constant $`1/4`$ in [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3) is what the proof delivers; the exact TV bound at the same point is somewhat larger ($`0.31\,\mathfrak n`$).
- Limits: in this model $`\Psi`$ is linear and the experiments are Gaussian, so the second-order, tail and efficiency conditions (S), (T) and (E) of [Theorem 6.5](../statements/06-05_achievability.md#s6-5) hold exactly. The check does not test nonlinear $`\Psi`$ or non-Gaussian experiments.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C2 --output /tmp/v016_C2.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

The first command reruns the check group(s) of this page into a separate file; `theory/v016_checks/check_v016.py` without `--only` regenerates the full result file (about 104 s). All checks run in float64 on CPU with seeds `numpy.SeedSequence([160160, group, subcase, ...])` (PCG64). The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)) for linear algebra and random numbers, and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)) for optimization and statistical functions.

## Artifacts

- `theory/v016_checks/results_v016.json`: claim ids `v016:C2.risk`, each with assertions, worst residuals, failures and data rows.
- `theory/v016_checks/results_v016_raw.json.gz`: complete rows; not tracked by git, its SHA-256 is recorded under `raw_artifact` in the main file.
- `theory/v016_checks/SPEC.md` (sections `v016:C2`) and `theory/v016_checks/check_v016.py`.
- `figures/fig6_1_information_bound.png` (PDF alongside)

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
