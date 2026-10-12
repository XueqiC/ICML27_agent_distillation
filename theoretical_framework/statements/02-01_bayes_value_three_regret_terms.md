# Definition 2.1 · Bayes value and the three regret terms

*Statement*

[← Framework](../README.md) · [Statement](02-01_bayes_value_three_regret_terms.md) · [中文](02-01_bayes_value_three_regret_terms_ZH.md)

**In one paragraph.** This definition fixes the objects with which §2 asks whether the available data can determine what the student should do. It introduces the Bayes value $`B(D)`$, the best expected return reachable from the transcript, and splits the student's shortfall into three regret terms: what the student class cannot represent, what the data cannot determine, and what the learner fails to use.

**Setting used on this page.**

- [Definition 1.1](../README.md#s1-1): environments $`E`$, values $`J_E(\pi)\in[0,1]`$, the student class $`\Pi`$ and the reference class $`\Pi_{\rm ref}`$.
- [Definition 1.2](../README.md#s1-2): the common initial information $`I_0`$ and the transcript $`D`$ of further experiments.
- $`\Delta(\Pi)`$ is the set of probability distributions on $`\Pi`$; a Markov kernel $`\lambda(\cdot\mid D)`$ assigns one to each transcript.

<a id="s2-1"></a>

**Definition 2.1 (Bayes value and the three regret terms).** Consider a finite decision problem.

- $`\mathcal E`$ is a finite set of environments with prior $`\mu`$. The prior is conditional on the common initial information $`I_0`$: the pretrained model, the known rules and the $`K`$ trusted demonstrations.
- $`\Pi\subseteq\Pi_{\rm ref}`$ are finite policy classes with values $`J_E(\pi)\in[0,1]`$. $`\Pi`$ is the student's deployment class and $`\Pi_{\rm ref}`$ the reference class.
- The transcript $`D`$ takes values in a finite alphabet $`\mathcal D`$ with law $`P(D\mid E)`$. Every experiment the learner may run, adaptive or not, is part of $`D`$.
- A learner is a Markov kernel $`\lambda(\cdot\mid D)\in\Delta(\Pi)`$ with output $`\widehat\pi\sim\lambda(\cdot\mid D)`$. Its private randomness is conditionally independent of $`E`$ given $`D`$.
- $`\mu(E\mid D)`$ is the posterior.

Define

```math
J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi),\qquad J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi),\qquad B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D],
```

and the representation, information and algorithm regrets

```math
\mathcal R_{\rm rep}=\mathbb E[J_E^{\rm ref}-J_E^\Pi],\qquad \mathcal R_{\rm info}=\mathbb E J_E^\Pi-\mathbb E_D B(D),\qquad \mathcal R_{\rm alg}=\mathbb E_D\big[B(D)-\mathbb E[J_E(\widehat\pi)\mid D]\big].
```

All maxima are attained because the classes are finite. $`B(D)`$ is the Bayes value: the best expected return a student in $`\Pi`$ can reach given the transcript.

**What it says.**

- $`B(D)`$ is the value of the best decision a Bayesian could make with the transcript, using a policy the student can execute.
- $`\mathcal R_{\rm rep}`$ measures the student class against the reference class, $`\mathcal R_{\rm info}`$ measures the transcript against perfect knowledge of the environment, and $`\mathcal R_{\rm alg}`$ measures the learner against the Bayes value.

**Intuition.**

- Representation: could the student do it at all? Information: does the data say what to do? Algorithm: does the learner do what the data say?

**Role in the framework.**

- The three terms are the bookkeeping for all of §2: [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2) proves they add up exactly, and the later results bound $`\mathcal R_{\rm info}`$ under specific data.

**Uses.**

- none

**Used by.**

- [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2): exact three-term decomposition.
- [Proposition 2.3](../statements/02-03_robust_ambiguity_duality.md#s2-3): robust version of the information term.
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4): data processing of the information term.
