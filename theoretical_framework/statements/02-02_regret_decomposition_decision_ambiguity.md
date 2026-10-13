# Theorem 2.2 · Regret decomposition and decision ambiguity

*Statement*

[← Framework](../README.md) · [Statement](02-02_regret_decomposition_decision_ambiguity.md) · [Proof](../proofs/02-02_regret_decomposition_decision_ambiguity.md) · [Experiments](../experiments/02-02_regret_decomposition_decision_ambiguity.md) · [中文](02-02_regret_decomposition_decision_ambiguity_ZH.md)

**In one paragraph.** Before asking how well a student learns, one must ask whether the data could tell it what to do at all. This theorem splits the student's expected regret exactly into a representation term, an information term and an algorithm term, and characterizes when the information term is zero: exactly when the environments the data cannot rule out share an optimal policy.

**Setting used on this page.**

- A finite set $`\mathcal E`$ of environments with a prior $`\mu`$, conditional on the common initial information $`I_0`$ (pretrained model, known rules, the $`K`$ trusted demonstrations).
- Finite policy classes $`\Pi\subseteq\Pi_{\rm ref}`$ with values $`J_E(\pi)\in[0,1]`$: $`\Pi`$ is what the student can deploy, $`\Pi_{\rm ref}`$ a reference class (for example, what the teacher can execute).
- The transcript $`D`$: everything the learner observes beyond $`I_0`$, including every experiment it may run; its law given $`E`$ is $`P(D\mid E)`$, and $`\mu(E\mid D)`$ is the posterior.
- A learner: a kernel $`\lambda(\cdot\mid D)`$ that outputs a policy $`\widehat\pi\in\Pi`$; its private randomness is independent of $`E`$ given $`D`$.
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ and $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ are the best values in environment $`E`$; $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ is the Bayes value, the best expected return a student in $`\Pi`$ can reach given $`D`$.
- The three regret terms $`\mathcal R_{\rm rep}`$, $`\mathcal R_{\rm info}`$, $`\mathcal R_{\rm alg}`$ are defined in [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1).

<a id="s2-2"></a>

**Theorem 2.2 (Regret decomposition and decision ambiguity).** Under [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1):

**(a) Exact decomposition.** The expected regret against the reference optimum is

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\mathcal R_{\rm rep}+\mathcal R_{\rm info}+\mathcal R_{\rm alg},\qquad \mathcal R_{\rm rep},\ \mathcal R_{\rm info},\ \mathcal R_{\rm alg}\ge0. \qquad(2.1)
```

**(b) Decision ambiguity.** $`\mathcal R_{\rm info}=0`$ if and only if, for every $`D`$ with $`\Pr(D)>0`$,

```math
\bigcap_{E:\,\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing. \qquad(2.2)
```

**What it says.**

- The expected regret against the best reference policy is the sum of three nonnegative terms: representation error (the student class cannot reach the reference optimum), information error (the data cannot tell the environments apart) and algorithm error (the learner does not use the data well).
- The information error is zero if and only if, for every transcript of positive probability, all environments with positive posterior probability share an optimal policy.

**Intuition.**

- The information error is what is lost even by a perfect learner, because the data leave the right decision ambiguous. More data can only lower it; a better learner can only lower the algorithm error.

**Role in the framework.**

- It separates "not enough information" from "learned badly". Every information result of the framework bounds one of these terms; [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6) and [Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10) compute the information term in explicit families.

**Derivation outline.**

- The identity follows by adding and subtracting $`\mathbb EJ_E^\Pi`$ and $`\mathbb E_DB(D)`$.
- The representation term is nonnegative because $`\Pi\subseteq\Pi_{\rm ref}`$. The information term is nonnegative because $`\mathbb E[J_E(\pi)\mid D]\le\mathbb E[J_E^\Pi\mid D]`$ for every policy. The algorithm term is nonnegative because, given $`D`$, the learner's output is a convex combination of policies and cannot beat the best of them.
- For (b), $`\mathcal R_{\rm info}`$ is a sum over transcripts of nonnegative brackets $`\sum_E\mu(E\mid D)(J_E^\Pi-J_E(\pi_D))`$; a bracket vanishes exactly when a posterior-optimal policy is optimal in every environment of positive posterior probability.

**Comments.**

- **What each term isolates.**
  - $`\mathcal R_{\rm rep}`$ is what no signal can remove: the student cannot execute the reference optimum.
  - $`\mathcal R_{\rm info}`$ is what remains when the transcript does not determine a common optimal decision.
  - $`\mathcal R_{\rm alg}`$ is the learner's failure to use information it has. One learner's failure is therefore not an information lower bound, and additional information need not help a fixed learner, because it can only lower $`\mathcal R_{\rm info}`$.
- **Equality is Bayes-additive only.** Maximizing the three terms separately, for instance taking worst cases term by term, does not preserve (2.1).
- **Scope.** Finiteness guarantees attained maxima and well-defined conditional expectations; general measurable spaces need measurable-selection and integrability conditions. The decomposition is a classical decision-theoretic identity. Its role here is to separate the three error sources that the rest of the framework treats separately.

**Uses.**

- [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1): Bayes value and the three terms.

**Used by.**

- [Proposition 2.3](../statements/02-03_robust_ambiguity_duality.md#s2-3): robust version of the information term.
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4): post-processing raises the information term.
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6): an information gap with an optimal teacher.
- [Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10): the information term after K demonstrations.
- [Proposition 3.1](../statements/03-01_static_value_not_learning_value.md#s3-1): learning gaps beyond representation.
- [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3): acquire before compiling.

## Related work

- The bracket $`\mathbb E[J_E^\Pi\mid D]-B(D)`$ in the derivation of (b) is the expected value of perfect information about $`E`$ given $`D`$ ([Howard, 1966](#ref-howard1966information)). Part (b) says that this value is zero for every transcript exactly when the environments the transcript cannot rule out share an optimal policy.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-howard1966information"></a>

**Ronald A. Howard (1966).** Information value theory. *IEEE Transactions on Systems Science and Cybernetics* 2, no. 1, 22–26. DOI [10.1109/TSSC.1966.300074](https://doi.org/10.1109/TSSC.1966.300074).
