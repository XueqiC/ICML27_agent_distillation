# Proposition 2.3 · Robust ambiguity duality

*Experiments*

[← Framework](../README.md) · [Statement](../statements/02-03_robust_ambiguity_duality.md) · [Proof](../proofs/02-03_robust_ambiguity_duality.md) · [Experiments](02-03_robust_ambiguity_duality.md) · [中文](02-03_robust_ambiguity_duality_ZH.md)

**What is tested here.** [Check 2.2](../experiments/02-03_robust_ambiguity_duality.md#c2-2) computes the canonical two-world instance by hand-checkable LPs: primal and dual are both 0.1, with minimax mixture and least favourable prior both $`(0.5,0.5)`$ (PASS). [Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity.md#c2-1) confirms the duality on 220 random instances, solving the primal and dual LPs separately (PASS).

**Setting used on this page.**

- A finite set $`\mathcal E`$ of environments with a prior $`\mu`$, conditional on the common initial information $`I_0`$ (pretrained model, known rules, the $`K`$ trusted demonstrations).
- Finite policy classes $`\Pi\subseteq\Pi_{\rm ref}`$ with values $`J_E(\pi)\in[0,1]`$: $`\Pi`$ is what the student can deploy, $`\Pi_{\rm ref}`$ a reference class (for example, what the teacher can execute).
- The transcript $`D`$: everything the learner observes beyond $`I_0`$, including every experiment it may run; its law given $`E`$ is $`P(D\mid E)`$, and $`\mu(E\mid D)`$ is the posterior.
- A learner: a kernel $`\lambda(\cdot\mid D)`$ that outputs a policy $`\widehat\pi\in\Pi`$; its private randomness is independent of $`E`$ given $`D`$.
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ and $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ are the best values in environment $`E`$; $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ is the Bayes value, the best expected return a student in $`\Pi`$ can reach given $`D`$.
- The three regret terms $`\mathcal R_{\rm rep}`$, $`\mathcal R_{\rm info}`$, $`\mathcal R_{\rm alg}`$ are defined in [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1).
- An ambiguity class $`C\subseteq\mathcal E`$ is a finite set of environments; $`\Delta(C)`$ is the set of priors on it.

**Statement under test ([Proposition 2.3](../statements/02-03_robust_ambiguity_duality.md#s2-3)).** For a finite ambiguity class $`C\subseteq\mathcal E`$ let

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{E\in C}\big[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\big]. \qquad(2.3)
```

Then

```math
r(C)=\max_{\mu\in\Delta(C)}\Big[\sum_E\mu_EJ_E^\Pi-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)\Big]. \qquad(2.4)
```

Moreover:

- **(a)** If every allowed experiment has the same law under every $`E\in C`$, every learner has worst-case expected regret at least $`r(C)`$ on $`C`$. If $`C`$ is known, the minimax mixture attains $`r(C)`$.
- **(b)** $`r(C)=0`$ if and only if some $`\pi\in\Pi`$ is optimal in every $`E\in C`$.

<a id="c2-2"></a>

## Check 2.2 · The canonical two-world ambiguity instance (Proposition 2.3)

**Verdict.** PASS: primal and dual values are both 0.1, with minimax mixture and least-favourable prior both $`(0.5,0.5)`$.

**Why.** A hand-checkable instance of Proposition 2.3 whose optimal mixtures are known in closed form. By von Neumann's minimax theorem the two LP values must coincide ([von Neumann, 1928](#ref-vonneumann1928zur)).

**Setup.** Two environments prefer opposite policies, with regret matrix $`M=\begin{pmatrix}0&0.2\cr 0.2&0\end{pmatrix}`$. Primal and dual LPs are solved with HiGHS ([Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)) through SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)) by the package v0.14 verifier, seed 2026101014.

**Result.**

| Quantity | Value | JSON key (entry of the check) |
|---|---|---|
| Primal $`r(C)`$ | 0.1 | `robust_ambiguity_minimax.primal_regret` |
| Dual value | 0.1 | `robust_ambiguity_minimax.dual_regret` |
| Minimax policy mixture | $`(0.5,0.5)`$ | `.policy_mixture` |
| Least-favourable prior | $`(0.5,0.5)`$ | `.environment_mixture` |

**Why it supports the statement.**

- When the two worlds cannot be told apart, every learner's output law is common to both worlds, and the best achievable worst-case expected regret is $`r(C)=0.1>0`$, half of the 0.2 regret of committing to either policy (Proposition 2.3(a)).
- The value is a *mixture-averaged* regret, as the comments on the statement page note: a sampled policy has regret 0 or 0.2.

### [Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity.md#c2-1) · Regret decomposition, ambiguity criterion, duality and data processing

- *What it tests for this proposition.* The equality of (2.3) and (2.4), and that the value is zero exactly when a common optimum exists.
- *Setup.* 220 of the 500 random finite decision problems of [Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity.md#c2-1) (3–6 environments, 2–5 policies, random rational values); the primal min-over-mixtures LP and the dual max-over-priors LP are solved separately with HiGHS.
- *Result.* The two LPs agree on every instance (worst residual $`1.33\times10^{-15}`$), and the value is 0 exactly on the instances with a common optimum.
- *Verdict.* PASS.

## Reproduce

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

```bash
theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py --output theory/v014_repro/results_v014_rerun.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C1 --output theory/independent_checks/subset.json
```

## Artifacts

- `theory/v014_repro/results_v014_rerun.json`: key `robust_ambiguity_minimax`.
- `theory/independent_checks/results_independent_v014.json`: the entry of `v014:C1`; code `theory/independent_checks/check_v014_independent.py` (`check_C1`).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).

<a id="ref-vonneumann1928zur"></a>

**John von Neumann (1928).** Zur Theorie der Gesellschaftsspiele. *Mathematische Annalen* 100, no. 1, 295–320. DOI [10.1007/BF01448847](https://doi.org/10.1007/BF01448847).
