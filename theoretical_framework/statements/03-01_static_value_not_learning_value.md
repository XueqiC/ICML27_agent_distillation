# Proposition 3.1 · Static value is not learning value

*Statement*

[← Framework](../README.md) · [Statement](03-01_static_value_not_learning_value.md) · [Proof](../proofs/03-01_static_value_not_learning_value.md) · [Experiments](../experiments/03-01_static_value_not_learning_value.md) · [中文](03-01_static_value_not_learning_value_ZH.md)

**In one paragraph.** Is it enough to judge a feedback source by how good it looks: a good teacher, a high policy value, the right mean gradient? This proposition gives two explicit counterexamples. In both, two sources agree on every such static quantity, yet the students trained on them differ, and in the second the sign of the difference flips with the task.

**Setting used on this page.**

- *Aliased student:* one context, so the student is a single number $`p\in[0,1]`$, the probability of action $`A`$; $`\pi_p`$ is the corresponding policy and $`J(\pi_p)`$ its value ([Definition 1.1](../README.md#s1-1)).
- *Frequency distillation:* from $`m`$ teacher demonstrations, $`\widehat p`$ is the fraction of $`A`$ actions.
- *One SGD step:* $`p^+=p_0-\eta\,\partial_p\ell`$ on a squared loss, with step $`0<\eta\le1`$.
- The *learning value* of a source is the expected deployment value of the student it trains.

<a id="s3-1"></a>

**Proposition 3.1 (Static value is not learning value).** *Construction (a): four-state sources.*

- States $`u,v,w,x`$ share the reward rules below; action $`B`$ always succeeds with probability $`\tfrac12`$.
- The reference source draws the state from $`\rho_0=(\tfrac34,\tfrac14,0,0)`$, the alternative source from $`\rho_1=(0,0,\tfrac14,\tfrac34)`$.
- An optimal teacher observes the state and plays its better action.
- The student is aliased: it has one context and plays $`A`$ with probability $`p`$, giving policy $`\pi_p`$.
- The learner is frequency distillation: from $`m\ge1`$ independent teacher demonstrations, $`\widehat p`$ is the fraction of $`A`$ actions (also the maximizer of empirical cross-entropy).

| State | $`\rho_0`$ | $`\rho_1`$ | $`r(A)`$ | $`r(B)`$ |
|---|---:|---:|---:|---:|
| $`u`$ | $`3/4`$ | $`0`$ | $`19/30`$ | $`1/2`$ |
| $`v`$ | $`1/4`$ | $`0`$ | $`3/10`$ | $`1/2`$ |
| $`w`$ | $`0`$ | $`1/4`$ | $`9/10`$ | $`1/2`$ |
| $`x`$ | $`0`$ | $`3/4`$ | $`13/30`$ | $`1/2`$ |

*Construction (b): two label packets.* A packet is a pair of binary labels $`(Y_1,Y_2)`$.

- Under the *independent* source, $`Y_1,Y_2`$ are IID Bernoulli $`(\tfrac12)`$. Under the *paired* source, $`(Y_1,Y_2)`$ is uniform on $`\lbrace (0,1),(1,0)\rbrace`$.
- The learner takes one gradient step of size $`0<\eta\le1`$ from $`p_0=\tfrac12`$ on the squared loss $`\tfrac14\sum_{i=1}^2(p-Y_i)^2`$:

```math
p^+=p_0-\eta\big(p_0-\tfrac{Y_1+Y_2}2\big)=\frac{1-\eta}2+\eta\,\frac{Y_1+Y_2}2 .
```

- Deployment is two independent decisions that both must choose action 1, so $`J(p)=p^2`$. The alternative task is $`J'(p)=1-p^2`$.

Then:

- **(a) Identical static values, different students.** In Construction (a), both sources give every aliased student the same value $`J(\pi_p)=\tfrac12+p/20`$ and give the teacher the same value $`\tfrac35`$. The teacher's $`A`$-frequencies are $`\tfrac34`$ and $`\tfrac14`$, and frequency distillation on $`m\ge1`$ demonstrations yields expected returns $`\tfrac{43}{80}`$ (reference) and $`\tfrac{41}{80}`$ (alternative): a gap of $`\tfrac1{40}`$ for every $`m`$.
- **(b) Identical mean updates, different learning values.** In Construction (b), the two sources have the same label marginals, the same mean gradient and the same mean update $`\mathbb Ep^+=\tfrac12`$, yet

```math
\mathbb E_{\rm indep}J(p^+)-\mathbb E_{\rm paired}J(p^+)=\frac{\eta^2}8, \qquad(3.1)
```

and under $`J'=1-p^2`$ the difference is $`-\eta^2/8`$.

**What it says.**

- Two data sources can give every student in the class the same policy value and give the teacher the same value, yet distill students with returns $`\tfrac{43}{80}`$ and $`\tfrac{41}{80}`$.
- Two label packets can have the same marginals and the same mean gradient, yet learning values that differ by $`\eta^2/8`$, with the sign reversed on another task.

**Intuition.**

- What is learned depends on how the data act on the student *through the update*, and on what the deployment rewards; how good the data look by themselves does not decide it.

**Role in the framework.**

- It rules out teacher quality, policy value and the mean gradient as sufficiency criteria and motivates the continuation response of [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2).

**Derivation outline.**

- (a) Both state laws give the action $`A`$ the same average reward $`\tfrac{11}{20}`$, so every aliased student has the same value. The teacher, however, plays $`A`$ with frequency $`\tfrac34`$ under one law and $`\tfrac14`$ under the other, and frequency distillation copies that frequency in expectation.
- (b) The two packets have equal first moments but different variances of the mean label ($`\tfrac18`$ against $`0`$). The deployment value $`p^2`$ reads the variance of the updated student, so $`\mathbb E(p^+)^2`$ differs by $`\eta^2/8`$; for $`1-p^2`$ the sign flips.

**Illustration.** Static evaluation of the data (how good the teacher is, how good each student looks on the data's states, the mean gradient) can be identical while learning reads a different statistic: the action frequency in (a), the second moment in (b). Which statistic matters is decided by the composition of the update with the deployment objective.

**Comments.**

- **What is equal and what is not.**
  - In (a) the static evaluation agrees on the whole restricted policy class $`\Pi=\lbrace \pi_p\rbrace`$ and on the teacher. Learning, however, consumes the teacher's *action frequencies*, which the values do not determine.
  - In (b) the label marginals and the mean gradient agree. The deployment's curvature, however, reads the packet's *second moment*.
  - In both cases the quantity that decides the outcome is the composition of the actual update with the deployment objective. [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2) makes this composition the governing object.
- **Fidelity is not improvement.** The sign of (3.1) depends on the task: the same variance helps under $`p^2`$ and hurts under $`1-p^2`$. Lower gradient variance, higher fidelity and better task performance are distinct properties.
- **Scope.** (a) concerns the stated aliased class; it does not contradict full Bellman value equivalence over unrestricted policy and value tests. The gap is specific to the learning algorithm: a learner that ignores both datasets and outputs $`p=1`$ has no gap. That is the point: sufficiency must be defined relative to the declared learner.

**Uses.**

- [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2): the gap is not representation or information.

**Used by.**

- [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2): the right invariant: continuation response.
- [Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes.md#s7-3): task-blind statistics cannot rank.
- [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3): select by measured consequence.

## Related work

- First-order influence estimates score a training example by an inner product with its gradient: influence functions through an inverse-Hessian-weighted inner product ([Koh & Liang, 2017](#ref-koh2017understanding)), TracIn through a sum of inner products along the training run ([Pruthi et al., 2020](#ref-pruthi2020estimating)). Averaged over a source, such a score depends on the source only through its mean gradient, so it cannot separate the two packets of Construction (b), whose mean gradients at $`p_0`$ agree.
- The comparison in (a) is value equivalence restricted to the aliased class. Value equivalence pins down the true model only as the sets of tested policies and functions grow ([Grimm et al., 2020](#ref-grimm2020value)), which is why (a) does not contradict full Bellman value equivalence.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-grimm2020value"></a>

**Christopher Grimm, André Barreto, Satinder Singh, David Silver (2020).** The value equivalence principle for model-based reinforcement learning. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 5541–5552. arXiv [2011.03506](https://arxiv.org/abs/2011.03506).

<a id="ref-koh2017understanding"></a>

**Pang Wei Koh, Percy Liang (2017).** Understanding black-box predictions via influence functions. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1885–1894. [proceedings.mlr.press/v70/koh17a](https://proceedings.mlr.press/v70/koh17a.html).

<a id="ref-pruthi2020estimating"></a>

**Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan (2020).** Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 19920–19930. arXiv [2002.08484](https://arxiv.org/abs/2002.08484).
