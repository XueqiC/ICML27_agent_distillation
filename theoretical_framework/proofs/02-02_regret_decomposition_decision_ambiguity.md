# Theorem 2.2 · Regret decomposition and decision ambiguity

*Proof*

[← Framework](../README.md) · [Statement](../statements/02-02_regret_decomposition_decision_ambiguity.md) · [Proof](02-02_regret_decomposition_decision_ambiguity.md) · [Experiments](../experiments/02-02_regret_decomposition_decision_ambiguity.md) · [中文](02-02_regret_decomposition_decision_ambiguity_ZH.md)

**What is proved here.** The decomposition is an exact identity, and each of its three terms is shown to be nonnegative. The zero-information criterion follows by writing the information term as a sum of nonnegative posterior brackets.

**Setting used on this page.**

- A finite set $`\mathcal E`$ of environments with a prior $`\mu`$, conditional on the common initial information $`I_0`$ (pretrained model, known rules, the $`K`$ trusted demonstrations).
- Finite policy classes $`\Pi\subseteq\Pi_{\rm ref}`$ with values $`J_E(\pi)\in[0,1]`$: $`\Pi`$ is what the student can deploy, $`\Pi_{\rm ref}`$ a reference class (for example, what the teacher can execute).
- The transcript $`D`$: everything the learner observes beyond $`I_0`$, including every experiment it may run; its law given $`E`$ is $`P(D\mid E)`$, and $`\mu(E\mid D)`$ is the posterior.
- A learner: a kernel $`\lambda(\cdot\mid D)`$ that outputs a policy $`\widehat\pi\in\Pi`$; its private randomness is independent of $`E`$ given $`D`$.
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ and $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ are the best values in environment $`E`$; $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ is the Bayes value, the best expected return a student in $`\Pi`$ can reach given $`D`$.
- The three regret terms $`\mathcal R_{\rm rep}`$, $`\mathcal R_{\rm info}`$, $`\mathcal R_{\rm alg}`$ are defined in [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1).

**Statement ([Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)).** Under [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1):

**(a) Exact decomposition.** The expected regret against the reference optimum is

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\mathcal R_{\rm rep}+\mathcal R_{\rm info}+\mathcal R_{\rm alg},\qquad \mathcal R_{\rm rep},\ \mathcal R_{\rm info},\ \mathcal R_{\rm alg}\ge0. \qquad(2.1)
```

**(b) Decision ambiguity.** $`\mathcal R_{\rm info}=0`$ if and only if, for every $`D`$ with $`\Pr(D)>0`$,

```math
\bigcap_{E:\,\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing. \qquad(2.2)
```

**Proof.**

**(a)** The identity follows by adding and subtracting $`\mathbb EJ_E^\Pi`$ and $`\mathbb E_DB(D)`$. It remains to show that each term is nonnegative.

- $`\mathcal R_{\rm rep}\ge0`$: since $`\Pi\subseteq\Pi_{\rm ref}`$, $`J_E^\Pi\le J_E^{\rm ref}`$ for every $`E`$.
- $`\mathcal R_{\rm info}\ge0`$: fix $`D`$ with $`\Pr(D)>0`$. For every $`\pi\in\Pi`$, $`\mathbb E[J_E(\pi)\mid D]\le\mathbb E[J_E^\Pi\mid D]`$, because $`J_E(\pi)\le J_E^\Pi`$ pointwise. Maximizing over $`\pi`$ gives $`B(D)\le\mathbb E[J_E^\Pi\mid D]`$. Taking expectations over $`D`$ and applying the tower property gives $`\mathbb E_DB(D)\le\mathbb EJ_E^\Pi`$.
- $`\mathcal R_{\rm alg}\ge0`$: conditional independence of $`\widehat\pi`$ and $`E`$ given $`D`$ gives

```math
\mathbb E[J_E(\widehat\pi)\mid D]=\sum_{\pi\in\Pi}\lambda(\pi\mid D)\,\mathbb E[J_E(\pi)\mid D]\le\max_\pi\mathbb E[J_E(\pi)\mid D]=B(D),
```

because a convex combination cannot exceed its largest term.

**(b)** By the tower property,

```math
\mathcal R_{\rm info}=\sum_{D:\Pr(D)>0}\Pr(D)\,\big(\mathbb E[J_E^\Pi\mid D]-B(D)\big).
```

Each bracket is nonnegative by (a), so $`\mathcal R_{\rm info}=0`$ if and only if every bracket vanishes. Fix such a $`D`$ and let $`\pi_D\in\arg\max_\pi\mathbb E[J_E(\pi)\mid D]`$. Then

```math
\mathbb E[J_E^\Pi\mid D]-B(D)=\sum_{E}\mu(E\mid D)\big(J_E^\Pi-J_E(\pi_D)\big),
```

a sum of nonnegative terms.

- ($`\Leftarrow`$) If $`\pi^\circ`$ lies in the intersection (2.2), then $`\mathbb E[J_E(\pi^\circ)\mid D]=\mathbb E[J_E^\Pi\mid D]\ge B(D)\ge\mathbb E[J_E(\pi^\circ)\mid D]`$, so the bracket is zero.
- ($`\Rightarrow`$) If the bracket is zero, every summand with $`\mu(E\mid D)>0`$ vanishes. Then $`\pi_D`$ is optimal in every environment of positive posterior mass, so it lies in the intersection. $`\square`$
