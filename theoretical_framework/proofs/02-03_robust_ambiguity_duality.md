# Proposition 2.3 · Robust ambiguity duality

*Proof*

[← Framework](../README.md) · [Statement](../statements/02-03_robust_ambiguity_duality.md) · [Proof](02-03_robust_ambiguity_duality.md) · [Experiments](../experiments/02-03_robust_ambiguity_duality.md) · [中文](02-03_robust_ambiguity_duality_ZH.md)

**What is proved here.** Write the robust ambiguity value as the value of a finite zero-sum game between the learner (policy mixtures) and an adversary (priors), and exchange min and max by von Neumann's minimax theorem. Parts (a) and (b) follow from the game form.

**Setting used on this page.**

- A finite set $`\mathcal E`$ of environments with a prior $`\mu`$, conditional on the common initial information $`I_0`$ (pretrained model, known rules, the $`K`$ trusted demonstrations).
- Finite policy classes $`\Pi\subseteq\Pi_{\rm ref}`$ with values $`J_E(\pi)\in[0,1]`$: $`\Pi`$ is what the student can deploy, $`\Pi_{\rm ref}`$ a reference class (for example, what the teacher can execute).
- The transcript $`D`$: everything the learner observes beyond $`I_0`$, including every experiment it may run; its law given $`E`$ is $`P(D\mid E)`$, and $`\mu(E\mid D)`$ is the posterior.
- A learner: a kernel $`\lambda(\cdot\mid D)`$ that outputs a policy $`\widehat\pi\in\Pi`$; its private randomness is independent of $`E`$ given $`D`$.
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ and $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ are the best values in environment $`E`$; $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ is the Bayes value, the best expected return a student in $`\Pi`$ can reach given $`D`$.
- The three regret terms $`\mathcal R_{\rm rep}`$, $`\mathcal R_{\rm info}`$, $`\mathcal R_{\rm alg}`$ are defined in [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1).
- An ambiguity class $`C\subseteq\mathcal E`$ is a finite set of environments; $`\Delta(C)`$ is the set of priors on it.

**Statement ([Proposition 2.3](../statements/02-03_robust_ambiguity_duality.md#s2-3)).** For a finite ambiguity class $`C\subseteq\mathcal E`$ let

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

**Proof.**

Define the regret matrix $`M_{E,\pi}=J_E^\Pi-J_E(\pi)\in[0,1]`$. For a mixture $`\rho`$ the map $`\mu\mapsto\mu^\top M\rho`$ is linear, so its maximum over $`\Delta(C)`$ is attained at a vertex, and

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{\mu\in\Delta(C)}\mu^\top M\rho .
```

This is the value of a finite two-player zero-sum game. Von Neumann's minimax theorem exchanges min and max:

```math
r(C)=\max_{\mu\in\Delta(C)}\min_{\rho\in\Delta(\Pi)}\mu^\top M\rho=\max_\mu\min_{\pi\in\Pi}(\mu^\top M)_\pi .
```

The inner minimum over the simplex is attained at a pure policy, and $`(\mu^\top M)_\pi=\sum_E\mu_EJ_E^\Pi-\sum_E\mu_EJ_E(\pi)`$, which gives (2.4).

- **(a)** If no experiment distinguishes the members of $`C`$, the law of $`D`$ is the same under every $`E\in C`$, and so is the output law $`\bar\rho=\sum_D\Pr(D)\lambda(\cdot\mid D)`$. The learner's expected regret in $`E`$ is $`(M\bar\rho)_E`$, so its worst case over $`C`$ is at least $`\min_\rho\max_E(M\rho)_E=r(C)`$. If $`C`$ is known, output a minimizer $`\rho^\star`$, which exists because $`\Delta(\Pi)`$ is compact and the objective is continuous.
- **(b)** If $`\pi^\circ`$ is optimal in every $`E\in C`$, the point mass at $`\pi^\circ`$ has worst-case regret 0. Conversely, if $`r(C)=0`$ is attained by $`\rho^\star`$, then $`\sum_\pi\rho^\star(\pi)M_{E,\pi}=0`$ for every $`E\in C`$. Since $`M\ge0`$, every $`\pi`$ in the support of $`\rho^\star`$ has $`M_{E,\pi}=0`$ for all $`E\in C`$, so it is a common optimum. $`\square`$

## Related work

- The exchange of min and max is von Neumann's minimax theorem for finite zero-sum games ([von Neumann, 1928](#ref-vonneumann1928zur)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-vonneumann1928zur"></a>

**John von Neumann (1928).** Zur Theorie der Gesellschaftsspiele. *Mathematische Annalen* 100, no. 1, 295–320. DOI [10.1007/BF01448847](https://doi.org/10.1007/BF01448847).
