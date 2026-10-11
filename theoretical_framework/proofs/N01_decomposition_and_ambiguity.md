# N01 · Regret decomposition and decision ambiguity — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N01_decomposition_and_ambiguity.md) · [中文](N01_decomposition_and_ambiguity_ZH.md)

## Setting and assumptions

**Assumption 1.1 (finite decision problem).**

- $`\mathcal E`$ is a finite set of environments with prior $`\mu`$. The prior is conditional on the common initial information $`I_0`$: the pretrained model, the known rules and the $`K`$ demonstrations.
- $`\Pi\subseteq\Pi_{\rm ref}`$ are finite policy classes, with values $`J_E(\pi)\in[0,1]`$. Here $`\Pi`$ is the student's deployment class and $`\Pi_{\rm ref}`$ the reference class.
- The transcript $`D`$ takes values in a finite alphabet $`\mathcal D`$ with law $`P(D\mid E)`$. Every experiment the learner may run, adaptive or not, is part of $`D`$.
- A learner is a Markov kernel $`\lambda(\cdot\mid D)\in\Delta(\Pi)`$, and its output is $`\widehat\pi\sim\lambda(\cdot\mid D)`$. Its private randomness is conditionally independent of $`E`$ given $`D`$.

Write $`\mu(E\mid D)`$ for the posterior. Define

```math
J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi),\qquad J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi),\qquad B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D].
```

All maxima are attained because the classes are finite.

## Statement

**(a)** **Proposition (exact decomposition).** Under Assumption 1.1,

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\underbrace{\mathbb E[J_E^{\rm ref}-J_E^\Pi]}_{\mathcal R_{\rm rep}}+\underbrace{\mathbb E J_E^\Pi-\mathbb E_D B(D)}_{\mathcal R_{\rm info}}+\underbrace{\mathbb E_D\big[B(D)-\mathbb E[J_E(\widehat\pi)\mid D]\big]}_{\mathcal R_{\rm alg}}, \qquad(1)
```

and $`\mathcal R_{\rm rep},\mathcal R_{\rm info},\mathcal R_{\rm alg}\ge0`$.

**(b)** **Proposition (decision ambiguity).** $`\mathcal R_{\rm info}=0`$ if and only if, for every $`D`$ with $`\Pr(D)>0`$,

```math
\bigcap_{E:\,\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing. \qquad(2)
```

**(c)** **Proposition (robust ambiguity and its dual).** For a finite ambiguity class $`C\subseteq\mathcal E`$, let

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{E\in C}\big[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\big]. \qquad(3)
```

Then

```math
r(C)=\max_{\mu\in\Delta(C)}\Big[\sum_E\mu_EJ_E^\Pi-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)\Big]. \qquad(4)
```

Moreover:

- (a) If every allowed experiment has the same law under every $`E\in C`$, then every learner has worst-case expected regret at least $`r(C)`$ on $`C`$. If $`C`$ is known, the minimax mixture attains $`r(C)`$.
- (b) $`r(C)=0`$ if and only if some $`\pi\in\Pi`$ is optimal in every $`E\in C`$.

**(d)** **Proposition (post-processing cannot create facts).** Let $`D'\sim\kappa(\cdot\mid D)`$ be a randomized post-processing whose kernel $`\kappa`$ does not depend on $`E`$, so that $`E\to D\to D'`$ is a Markov chain. Then

```math
\mathbb E\max_{\pi}\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D], \qquad(5)
```

that is, $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$. If $`Z\perp E\mid(D,I_0)`$, then $`B(D,Z)=B(D)`$, so adding $`Z`$ leaves $`\mathcal R_{\rm info}`$ unchanged.

## Proof

**Proof of (a).** The identity (1) follows by adding and subtracting $`\mathbb E J_E^\Pi`$ and $`\mathbb E_DB(D)`$. It remains to show that each term is nonnegative.

$`\mathcal R_{\rm rep}\ge0`$. Since $`\Pi\subseteq\Pi_{\rm ref}`$, we have $`J_E^\Pi\le J_E^{\rm ref}`$ for every $`E`$.

$`\mathcal R_{\rm info}\ge0`$. Fix $`D`$ with $`\Pr(D)>0`$. For every $`\pi\in\Pi`$, $`\mathbb E[J_E(\pi)\mid D]\le\mathbb E[J_E^\Pi\mid D]`$, because $`J_E(\pi)\le J_E^\Pi`$ pointwise. Maximizing over $`\pi`$ gives $`B(D)\le\mathbb E[J_E^\Pi\mid D]`$. Taking expectations over $`D`$ and applying the tower property gives $`\mathbb E_DB(D)\le\mathbb EJ_E^\Pi`$.

$`\mathcal R_{\rm alg}\ge0`$. Conditional independence of $`\widehat\pi`$ and $`E`$ given $`D`$ gives

```math
\mathbb E[J_E(\widehat\pi)\mid D]=\sum_{\pi\in\Pi}\lambda(\pi\mid D)\,\mathbb E[J_E(\pi)\mid D]\le\max_\pi\mathbb E[J_E(\pi)\mid D]=B(D).
```

The inequality holds because a convex combination cannot exceed its largest term. $`\square`$

**Proof of (b).** By the tower property,

```math
\mathcal R_{\rm info}=\sum_{D:\Pr(D)>0}\Pr(D)\,\big(\mathbb E[J_E^\Pi\mid D]-B(D)\big).
```

Each bracket is nonnegative by the proof of (a). Hence $`\mathcal R_{\rm info}=0`$ if and only if every bracket vanishes. Fix such a $`D`$, and let $`\pi_D\in\arg\max_\pi\mathbb E[J_E(\pi)\mid D]`$. Then

```math
\mathbb E[J_E^\Pi\mid D]-B(D)=\sum_{E}\mu(E\mid D)\big(J_E^\Pi-J_E(\pi_D)\big),
```

which is a sum of nonnegative terms.

- *($`\Leftarrow`$)* Suppose $`\pi^\circ`$ lies in the intersection (2). Then $`\mathbb E[J_E(\pi^\circ)\mid D]=\mathbb E[J_E^\Pi\mid D]\ge B(D)\ge\mathbb E[J_E(\pi^\circ)\mid D]`$, so the bracket is zero.
- *($`\Rightarrow`$)* If the bracket is zero, every summand with $`\mu(E\mid D)>0`$ vanishes. Then $`\pi_D`$ is optimal in every environment of positive posterior mass, so it lies in the intersection. $`\square`$

**Proof of (c).** Define the regret matrix $`M_{E,\pi}=J_E^\Pi-J_E(\pi)\in[0,1]`$. For a mixture $`\rho`$, the function $`\mu\mapsto\mu^\top M\rho`$ is linear, so its maximum over $`\Delta(C)`$ is attained at a vertex. Hence

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{\mu\in\Delta(C)}\mu^\top M\rho.
```

This is the value of a finite two-player zero-sum game, and von Neumann's minimax theorem exchanges min and max:

```math
r(C)=\max_{\mu\in\Delta(C)}\min_{\rho\in\Delta(\Pi)}\mu^\top M\rho=\max_\mu\min_{\pi\in\Pi}(\mu^\top M)_\pi.
```

The inner minimum over the simplex is attained at a pure policy. Moreover $`(\mu^\top M)_\pi=\sum_E\mu_EJ_E^\Pi-\sum_E\mu_EJ_E(\pi)`$, so the inner minimum equals $`\sum_E\mu_EJ_E^\Pi-\max_\pi\sum_E\mu_EJ_E(\pi)`$. This is (4).

- *(a)* If no experiment distinguishes the members of $`C`$, the law of $`D`$ is the same under every $`E\in C`$. The output law $`\bar\rho=\sum_D\Pr(D)\lambda(\cdot\mid D)`$ is therefore also common to all $`E\in C`$. The learner's expected regret in $`E`$ is $`(M\bar\rho)_E`$, so its worst case over $`C`$ is at least $`\min_\rho\max_E(M\rho)_E=r(C)`$. If $`C`$ is known, outputting a minimizing $`\rho^\star`$ attains $`r(C)`$. Such a minimizer exists because $`\Delta(\Pi)`$ is compact and the objective is continuous.
- *(b)* If $`\pi^\circ`$ is optimal in every $`E\in C`$, the point mass at $`\pi^\circ`$ gives worst-case regret 0. Conversely, suppose $`r(C)=0`$, attained by $`\rho^\star`$. Then $`\sum_\pi\rho^\star(\pi)M_{E,\pi}=0`$ for every $`E\in C`$. Since $`M\ge0`$, every $`\pi`$ in the support of $`\rho^\star`$ satisfies $`M_{E,\pi}=0`$ for all $`E\in C`$. Any such $`\pi`$ is a common optimum. $`\square`$

**Proof of (d).** Let $`h_\pi(D)=\mathbb E[J_E(\pi)\mid D]`$. The Markov property ($`E\perp D'\mid D`$) and the tower property give

```math
\mathbb E[J_E(\pi)\mid D']=\mathbb E\big[\mathbb E[J_E(\pi)\mid D,D']\,\big|\,D'\big]=\mathbb E[h_\pi(D)\mid D'].
```

A maximum of conditional expectations is at most the conditional expectation of the maximum:

```math
\max_\pi\mathbb E[h_\pi(D)\mid D']\le\mathbb E\big[\max_\pi h_\pi(D)\,\big|\,D'\big]=\mathbb E[B(D)\mid D'].
```

Take expectations over $`D'`$ and apply the tower property to obtain (5). Since $`\mathbb EJ_E^\Pi`$ does not depend on the transcript, (5) is equivalent to $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$.

For the augmentation, $`Z\perp E\mid D`$ gives $`\mu(E\mid D,Z)=\mu(E\mid D)`$. Hence $`\mathbb E[J_E(\pi)\mid D,Z]=h_\pi(D)`$ for every $`\pi`$, and $`B(D,Z)=B(D)`$. $`\square`$

## Remarks

1. **What each term isolates.** $`\mathcal R_{\rm rep}`$ is what no signal can remove, because the student cannot execute the reference optimum. $`\mathcal R_{\rm info}`$ is what remains when the transcript does not determine a common optimal decision. $`\mathcal R_{\rm alg}`$ is the learner's failure to use information it has. One learner's failure is therefore not an information lower bound. Additional information need not help a fixed learner, because it can only lower $`\mathcal R_{\rm info}`$.
2. **Generated data.** By (d), synthetic rewrites, paraphrases and replays of $`D`$ cannot lower $`\mathcal R_{\rm info}`$. They can lower $`\mathcal R_{\rm alg}`$ by making existing information easier to learn. Knowledge already inside a pretrained teacher belongs in $`I_0`$; new external queries enlarge $`D`$.
3. **Probability semantics of (3).** Suppose a confidence set $`C_D`$ contains the true environment with probability $`1-\delta`$, and a mixture has worst-case gap at most $`\varepsilon`$ on $`C_D`$. This bounds the mixture's *expected* regret on that data event. It does not bound the regret of one sampled policy.
4. **Equality is Bayes-additive only.** Maximizing the three terms separately, for instance taking worst cases term by term, does not preserve the identity (1).
5. **Scope.** Finiteness guarantees attained maxima and well-defined conditional expectations. General measurable spaces need measurable-selection and integrability conditions. The decomposition is a classical decision-theoretic identity; its role here is to separate the three error sources that the rest of the framework treats separately.

## Dependencies and source

- **Uses:** none (first node).
- **Used by:** [N02](N02_passive_vs_active_acquisition.md) (an instance where $`\mathcal R_{\rm info}>0`$ for perfect demonstrations), [N03](N03_few_shot_information_floor.md) (exact $`\mathcal R_{\rm info}(K)`$), [N04](N04_static_vs_learning_sufficiency.md).

**Source:** [v0.14.1 §2](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#2-first-principles-feasibility-identification-and-use) — Props. 1–3; eqs. (1)–(5). [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md), §2, Propositions 1–3, eqs. (1)–(5). Audit: [Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md), A1–A4 and D2.
