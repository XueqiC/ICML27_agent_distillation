# Theorem 6.3 · Learner-relative lower bound

*Experiments*

[← Framework](../README.md) · [Statement](../statements/06-03_learner_relative_lower_bound.md) · [Proof](../proofs/06-03_learner_relative_lower_bound.md) · [Experiments](../experiments/06-03_learner_relative_lower_bound.md) · [中文](06-03_learner_relative_lower_bound_ZH.md)

**What is tested here.** Theorem 6.3 is tested through [Check 6.2](../experiments/06-05_achievability.md#c6-2), whose primary page is that of [Theorem 6.5](../statements/06-05_achievability.md#s6-5). On 192 identifiable Gaussian instances the Pinsker two-point bound of part (a), evaluated at the theorem's displacement, equals exactly $`\mathfrak n_{K,n}/4`$, and no two-point bound exceeds the achieved risk. Verdict: PASS.

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

**Statement under test ([Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3)).**

- **(a) Exact two-point bound.** For all $`\theta_0,\theta_1\in\Theta`$ and every distiller,

```math
\max_{i\in\lbrace0,1\rbrace}\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\big\lvert\Psi(\theta_1)-\Psi(\theta_0)\big\rvert\Big(1-\sqrt{\tfrac12\big(K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)\big)}\Big),
```

  where $`\mathrm{KL}_D=\mathrm{KL}(P^D_{\theta_1}\Vert P^D_{\theta_0})`$ and $`\mathrm{KL}_Q(x)=\mathrm{KL}(P^Q_{\theta_1}(\cdot\mid x)\Vert P^Q_{\theta_0}(\cdot\mid x))`$.

- **(b) Local bound.** Assume that each $`\mu^*_t`$ is $`C^1`$ near $`\theta_0`$, that both experiments are differentiable in quadratic mean at $`\theta_0`$, and that, as $`\tau\to0`$,

```math
\mathrm{KL}(P^D_{\theta_0+\tau h}\Vert P^D_{\theta_0})\le\tfrac12\tau^2h^\top I_D(\theta_0)h\,(1+o(1)),\qquad \mathrm{KL}_Q(x)\le\tfrac12\tau^2h^\top I_Q(\theta_0;x)h\,(1+o(1)),
```

  uniformly in $`x\in\mathcal X_Q`$ and in unit $`h`$. Let $`h`$ be a unit direction with $`c(\theta_0)^\top h\neq0`$ and $`\mathbb I_{K,n}(\theta_0;h)\to\infty`$, and put $`\tau=\mathbb I_{K,n}(\theta_0;h)^{-1/2}`$. Then

```math
\inf_Q\max_{\theta\in\lbrace\theta_0,\,\theta_0+\tau h\rbrace}\mathbb E_\theta\mathrm{Err}_\theta(Q)\ge\frac{\lvert c(\theta_0)^\top h\rvert}{4\sqrt{\mathbb I_{K,n}(\theta_0;h)}}\,\big(1-o(1)\big).
```

- **(c) The norm as the local lower bound.** Under (G), taking for $`h`$ a maximizer of the ratio in [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2), the local minimax lower bound is $`\tfrac14\mathfrak n_{K,n}(\theta_0)\,(1-o(1))`$.

### [Check 6.2](../experiments/06-05_achievability.md#c6-2) · The achieved risk lies between 𝔫/4 and 𝔫

- **What it tests for Theorem 6.3.** Whether the lower-bound constant $`1/4`$ is what the two-point argument actually delivers, and whether any distiller beats it.
- **Setup.** 192 identifiable Gaussian linear instances ($`p\in\lbrace2,3,5,8\rbrace`$, $`K\in\lbrace1,10,100\rbrace`$, $`n\in\lbrace0,10,100\rbrace`$); efficient plug-in estimator under the integer-rounded c-optimal design, 30,000 replicates per instance; the Pinsker two-point bound of part (a) (Le Cam's method, [Le Cam, 1973](#ref-lecam1973convergence), with Pinsker's inequality, [Cover & Thomas, 2006](#ref-cover2006elements)) and the exact Gaussian total-variation bound, both at the displacement $`\tau=\mathbb I^{-1/2}`$.
- **Key numbers.** The Pinsker bound equals $`0.2500\,\mathfrak n`$ in every instance; the exact total-variation bound equals $`0.3085\,\mathfrak n=(1-\Phi(1/2))\,\mathfrak n`$; the achieved risk is at least $`3.15`$ times the Pinsker bound; achieved risk$`/\mathfrak n`$ lies in $`[0.789,0.850]`$.
- **Verdict.** PASS. No distiller in the check comes close to the lower bound, and the constant $`1/4`$ is exactly what the proof gives at its displacement; the exact bound at the same point is somewhat larger.

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

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-cover2006elements"></a>

**Thomas M. Cover, Joy A. Thomas (2006).** Elements of Information Theory. 2nd edition, Wiley-Interscience, Hoboken, NJ. DOI [10.1002/047174882X](https://doi.org/10.1002/047174882X).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
