# Proposition 2.4 · Post-processing cannot reduce information regret

*Proof*

[← Framework](../README.md) · [Statement](../statements/02-04_post_processing_cannot_reduce_information.md) · [Proof](02-04_post_processing_cannot_reduce_information.md) · [Experiments](../experiments/02-04_post_processing_cannot_reduce_information.md) · [中文](02-04_post_processing_cannot_reduce_information_ZH.md)

**What is proved here.** The Bayes value after an environment-independent post-processing is the expectation of a maximum of conditional expectations, which is at most the expectation of the original Bayes value. Augmenting the transcript by independent noise leaves every posterior, and hence the Bayes value, unchanged.

**Setting used on this page.**

- A finite set $`\mathcal E`$ of environments with a prior $`\mu`$, conditional on the common initial information $`I_0`$ (pretrained model, known rules, the $`K`$ trusted demonstrations).
- Finite policy classes $`\Pi\subseteq\Pi_{\rm ref}`$ with values $`J_E(\pi)\in[0,1]`$: $`\Pi`$ is what the student can deploy, $`\Pi_{\rm ref}`$ a reference class (for example, what the teacher can execute).
- The transcript $`D`$: everything the learner observes beyond $`I_0`$, including every experiment it may run; its law given $`E`$ is $`P(D\mid E)`$, and $`\mu(E\mid D)`$ is the posterior.
- A learner: a kernel $`\lambda(\cdot\mid D)`$ that outputs a policy $`\widehat\pi\in\Pi`$; its private randomness is independent of $`E`$ given $`D`$.
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ and $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ are the best values in environment $`E`$; $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ is the Bayes value, the best expected return a student in $`\Pi`$ can reach given $`D`$.
- The three regret terms $`\mathcal R_{\rm rep}`$, $`\mathcal R_{\rm info}`$, $`\mathcal R_{\rm alg}`$ are defined in [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1).
- A post-processing is a kernel $`\kappa(\cdot\mid D)`$ that does not depend on $`E`$, so $`E\to D\to D'`$ is a Markov chain.

**Statement ([Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4)).** Let $`D'\sim\kappa(\cdot\mid D)`$ be a randomized post-processing whose kernel $`\kappa`$ does not depend on $`E`$, so that $`E\to D\to D'`$ is a Markov chain. Then

```math
\mathbb E\max_{\pi}\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D], \qquad(2.5)
```

that is, $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$. If $`Z\perp E\mid(D,I_0)`$, then $`B(D,Z)=B(D)`$, so adding $`Z`$ to the transcript leaves $`\mathcal R_{\rm info}`$ unchanged.

**Proof.**

Let $`h_\pi(D)=\mathbb E[J_E(\pi)\mid D]`$. The Markov property ($`E\perp D'\mid D`$) and the tower property give

```math
\mathbb E[J_E(\pi)\mid D']=\mathbb E\big[\mathbb E[J_E(\pi)\mid D,D']\,\big|\,D'\big]=\mathbb E[h_\pi(D)\mid D'] .
```

A maximum of conditional expectations is at most the conditional expectation of the maximum:

```math
\max_\pi\mathbb E[h_\pi(D)\mid D']\le\mathbb E\big[\max_\pi h_\pi(D)\,\big|\,D'\big]=\mathbb E[B(D)\mid D'] .
```

Taking expectations over $`D'`$ gives (2.5). Since $`\mathbb EJ_E^\Pi`$ does not depend on the transcript, (2.5) is equivalent to $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$. For the augmentation, $`Z\perp E\mid D`$ gives $`\mu(E\mid D,Z)=\mu(E\mid D)`$, hence $`\mathbb E[J_E(\pi)\mid D,Z]=h_\pi(D)`$ for every $`\pi`$ and $`B(D,Z)=B(D)`$. $`\square`$

## Related work

- That garbling a transcript can only lower the Bayes value of every decision problem is the easy direction of Blackwell's comparison of experiments ([Blackwell, 1953](#ref-blackwell1953equivalent)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-blackwell1953equivalent"></a>

**David Blackwell (1953).** Equivalent comparisons of experiments. *The Annals of Mathematical Statistics* 24, no. 2, 265–272. DOI [10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032).
