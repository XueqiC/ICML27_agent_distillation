# Proposition 2.4 · Post-processing cannot reduce information regret

*Statement*

[← Framework](../README.md) · [Statement](02-04_post_processing_cannot_reduce_information.md) · [Proof](../proofs/02-04_post_processing_cannot_reduce_information.md) · [Experiments](../experiments/02-04_post_processing_cannot_reduce_information.md) · [中文](02-04_post_processing_cannot_reduce_information_ZH.md)

**In one paragraph.** Replay, paraphrase, rewriting and synthetic generation are all transformations of data the learner already has. This proposition shows that no such transformation can reduce the information term of the regret, because it cannot add evidence about the environment; it can only make existing evidence easier to learn.

**Setting used on this page.**

- A finite set $`\mathcal E`$ of environments with a prior $`\mu`$, conditional on the common initial information $`I_0`$ (pretrained model, known rules, the $`K`$ trusted demonstrations).
- Finite policy classes $`\Pi\subseteq\Pi_{\rm ref}`$ with values $`J_E(\pi)\in[0,1]`$: $`\Pi`$ is what the student can deploy, $`\Pi_{\rm ref}`$ a reference class (for example, what the teacher can execute).
- The transcript $`D`$: everything the learner observes beyond $`I_0`$, including every experiment it may run; its law given $`E`$ is $`P(D\mid E)`$, and $`\mu(E\mid D)`$ is the posterior.
- A learner: a kernel $`\lambda(\cdot\mid D)`$ that outputs a policy $`\widehat\pi\in\Pi`$; its private randomness is independent of $`E`$ given $`D`$.
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ and $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ are the best values in environment $`E`$; $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ is the Bayes value, the best expected return a student in $`\Pi`$ can reach given $`D`$.
- The three regret terms $`\mathcal R_{\rm rep}`$, $`\mathcal R_{\rm info}`$, $`\mathcal R_{\rm alg}`$ are defined in [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1).
- A post-processing is a kernel $`\kappa(\cdot\mid D)`$ that maps the transcript to a new transcript $`D'`$ without looking at the environment, so that $`E\to D\to D'`$ is a Markov chain. $`\mathcal R_{\rm info}(D)`$ is the information term computed from transcript $`D`$.

<a id="s2-4"></a>

**Proposition 2.4 (Post-processing cannot reduce information regret).** Let $`D'\sim\kappa(\cdot\mid D)`$ be a randomized post-processing whose kernel $`\kappa`$ does not depend on $`E`$, so that $`E\to D\to D'`$ is a Markov chain. Then

```math
\mathbb E\max_{\pi}\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D], \qquad(2.5)
```

that is, $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$. If $`Z\perp E\mid(D,I_0)`$, then $`B(D,Z)=B(D)`$, so adding $`Z`$ to the transcript leaves $`\mathcal R_{\rm info}`$ unchanged.

**What it says.**

- Any randomized transformation of the transcript that does not depend on the environment can only raise the information term or leave it unchanged.
- Adding data that is independent of the environment given the transcript leaves the Bayes value and the information term unchanged.

**Intuition.**

- Processing data can make existing information easier to learn, which lowers the algorithm error, but it cannot create a missing fact.

**Role in the framework.**

- It rules out generated data as a substitute for new experiments. [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7) states the same for the learner-relative first-order effect, for every distiller.

**Derivation outline.**

- Because $`E\to D\to D'`$ is a Markov chain, $`\mathbb E[J_E(\pi)\mid D']=\mathbb E\big[\mathbb E[J_E(\pi)\mid D]\,\big|\,D'\big]`$.
- A maximum of conditional expectations is at most the conditional expectation of the maximum, so the Bayes value under $`D'`$ is at most that under $`D`$.

**Illustration.** Replaying, paraphrasing or rewriting the $`K`$ demonstrations by an environment-independent procedure is such a $`\kappa`$. It can make the student fit the existing information better (lower $`\mathcal R_{\rm alg}`$) but cannot lower $`\mathcal R_{\rm info}`$.

**Comments.**

- **Generated data.** By Proposition 2.4, synthetic rewrites, paraphrases and replays of $`D`$ cannot lower $`\mathcal R_{\rm info}`$. They can lower $`\mathcal R_{\rm alg}`$ by making existing information easier to learn. Knowledge already inside a pretrained teacher belongs to $`I_0`$; new external queries enlarge $`D`$.

**Uses.**

- [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1): the Bayes value and the information term.

**Used by.**

- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6): passive learners cannot recover the sign.
- [Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10): replay adds nothing to the information floor.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7): learner-relative version for every distiller.
- [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3): generated data cannot replace acquisition.

## Related work

- That an environment-independent garbling of the transcript cannot raise the Bayes value of any decision problem is the easy direction of Blackwell's comparison of experiments ([Blackwell, 1953](#ref-blackwell1953equivalent)).
- (2.5) is the decision-theoretic counterpart of the data-processing inequality for the Markov chain $`E\to D\to D'`$ ([Cover & Thomas, 2006](#ref-cover2006elements)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-blackwell1953equivalent"></a>

**David Blackwell (1953).** Equivalent comparisons of experiments. *The Annals of Mathematical Statistics* 24, no. 2, 265–272. DOI [10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032).

<a id="ref-cover2006elements"></a>

**Thomas M. Cover, Joy A. Thomas (2006).** Elements of Information Theory. 2nd edition, Wiley-Interscience, Hoboken, NJ. DOI [10.1002/047174882X](https://doi.org/10.1002/047174882X).
