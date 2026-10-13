# Proposition 2.3 · Robust ambiguity duality

*Statement*

[← Framework](../README.md) · [Statement](02-03_robust_ambiguity_duality.md) · [Proof](../proofs/02-03_robust_ambiguity_duality.md) · [Experiments](../experiments/02-03_robust_ambiguity_duality.md) · [中文](02-03_robust_ambiguity_duality_ZH.md)

**In one paragraph.** When several environments cannot be told apart by any allowed experiment, the learner must hedge across them. This proposition gives the price of hedging in two equivalent forms: the best worst-case regret over policy mixtures, and the largest information regret over priors on those environments.

**Setting used on this page.**

- A finite set $`\mathcal E`$ of environments with a prior $`\mu`$, conditional on the common initial information $`I_0`$ (pretrained model, known rules, the $`K`$ trusted demonstrations).
- Finite policy classes $`\Pi\subseteq\Pi_{\rm ref}`$ with values $`J_E(\pi)\in[0,1]`$: $`\Pi`$ is what the student can deploy, $`\Pi_{\rm ref}`$ a reference class (for example, what the teacher can execute).
- The transcript $`D`$: everything the learner observes beyond $`I_0`$, including every experiment it may run; its law given $`E`$ is $`P(D\mid E)`$, and $`\mu(E\mid D)`$ is the posterior.
- A learner: a kernel $`\lambda(\cdot\mid D)`$ that outputs a policy $`\widehat\pi\in\Pi`$; its private randomness is independent of $`E`$ given $`D`$.
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ and $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ are the best values in environment $`E`$; $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ is the Bayes value, the best expected return a student in $`\Pi`$ can reach given $`D`$.
- The three regret terms $`\mathcal R_{\rm rep}`$, $`\mathcal R_{\rm info}`$, $`\mathcal R_{\rm alg}`$ are defined in [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1).
- An ambiguity class $`C\subseteq\mathcal E`$ is a finite set of environments; $`\Delta(C)`$ is the set of priors on it and $`\mathbb E_{\pi\sim\rho}`$ averages over a policy mixture $`\rho\in\Delta(\Pi)`$.

<a id="s2-3"></a>

**Proposition 2.3 (Robust ambiguity duality).** For a finite ambiguity class $`C\subseteq\mathcal E`$ let

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

**What it says.**

- The minimax regret of hedging over an ambiguity class equals the largest Bayes information regret over priors on the class.
- If the experiments cannot distinguish the members of the class, no learner does better in the worst case, and the minimax mixture attains the value. The value is zero exactly when one policy is optimal in all members.

**Intuition.**

- Indistinguishable worlds force the learner to hedge. The price of hedging is the value of a zero-sum game against the least favourable prior.

**Role in the framework.**

- It gives the information error a computable minimax form and a certificate of its size (a least favourable prior). The two-world family of [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6) is an instance with value $`\Delta`$.

**Derivation outline.**

- The regret matrix $`M_{E,\pi}=J_E^\Pi-J_E(\pi)`$ defines a finite zero-sum game: the learner picks a mixture of policies, the adversary a prior over $`C`$.
- Von Neumann's minimax theorem exchanges min and max; the inner minimum is attained at a pure policy, which gives the dual form.
- When the experiments cannot tell the members of $`C`$ apart, the learner's output law is the same in all of them, so its worst-case regret is at least the game value.

**Illustration.** Two environments prefer opposite policies, with regret matrix $`\begin{pmatrix}0&0.2\cr 0.2&0\end{pmatrix}`$. The minimax mixture is $`(\tfrac12,\tfrac12)`$ and the least favourable prior is $`(\tfrac12,\tfrac12)`$; both sides equal $`0.1`$. This is a mixture-averaged regret: a sampled policy has regret 0 or 0.2.

**Comments.**

- **Probability semantics of (2.3).** Suppose a confidence set $`C_D`$ contains the true environment with probability $`1-\delta`$ and a mixture has worst-case gap at most $`\varepsilon`$ on $`C_D`$. This bounds the mixture's *expected* regret on that data event, not the regret of one sampled policy.

**Uses.**

- [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1): the regret terms.

**Used by.**

- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6): an instance with value Δ.

## Related work

- The equality of (2.3) and (2.4) is von Neumann's minimax theorem for the finite zero-sum game with payoff $`M_{E,\pi}=J_E^\Pi-J_E(\pi)`$ ([von Neumann, 1928](#ref-vonneumann1928zur)); the maximizing prior is the least favourable prior of that game.
