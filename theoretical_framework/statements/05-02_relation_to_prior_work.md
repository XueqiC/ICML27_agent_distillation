# Remark 5.2 · Relation to stochastic modified equations, TracIn, influence functions and metagradients

*Statement*

[← Framework](../README.md) · [Statement](../statements/05-02_relation_to_prior_work.md) · [中文](05-02_relation_to_prior_work_ZH.md)

**In one paragraph.** This remark places [Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1) among existing first-order analyses of training, states what its bookkeeping over the complete learner state makes explicit, and explains why, for Adam-type optimizers, the second-order remainder is of first order in the learning rate.

**Setting used on this page.**

The objects below are restated so that this page can be read on its own; the general notation is that of [Definitions 1.3–1.4](../README.md#s1-3).

- **State.** The complete learner state is $`z=(\theta,o)\in\mathcal Z\subseteq\mathbb R^p\times\mathbb R^q`$. Here $`\theta`$ are the trainable parameters, for example LoRA adapters, and $`o`$ is the optimizer state, for example Adam's moments $`m,v`$ and the step counter. When the counter matters it is written explicitly.
- **Feedback and gradients.** At round $`t`$ a feedback law $`P_t(\cdot\mid z)`$ emits a message $`Y`$. The learner computes a consumed gradient $`g=G_t(z,Y,\xi)\in\mathbb R^p`$ after loss normalization and accumulation, with $`\xi`$ the declared update randomness. Clipping, if any, is part of the optimizer.
- **Optimizer.** The optimizer is a measurable increment map $`z^+=z+\Delta_t(z,g)`$. The compared feedback laws share the optimizer and the gradient map and differ only in the message law. A change of optimizer is a different $`\Delta`$, compared in the same way.
- **Letters reused locally.** In this section $`s`$ counts gradient evaluations absorbed by Adam's second moment (Proposition 5.8). It is unrelated to the agent states $`s`$ of Section 6. The clipping threshold is $`c`$; the weight-decay coefficient is $`\lambda_{\rm wd}`$.

The learning kernel of a law $`P`$ is

```math
(K_t^Pf)(z)=\mathbb E_{Y\sim P_t(\cdot\mid z),\,\xi}\,f\big(z+\Delta_t(z,G_t(z,Y,\xi))\big).
```

For a law $`P`$ at $`(t,z)`$, the message-level mean increment is $`D_t(z,y)=\mathbb E_\xi\Delta_t(z,G_t(z,y,\xi))`$, and

```math
\mu_t^P(z)=\mathbb E_{Y\sim P}D_t(z,Y),\qquad m_{k,t}^P(z)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(z,G_t(z,Y,\xi))^{\otimes j}.
```

- **Continuation.** $`V^P_{t+1}:\mathcal Z\to[0,1]`$ is the continuation value of protocol $`P`$: the later updates run under $`P`$, then the student is deployed and scored by $`F_E`$ ([Definition 1.4](../README.md#s1-4)). $`L(P)=\mathbb E\,F_E(Z_R^P)`$, and $`d_t^Q`$ is the law of $`Z_t`$ under protocol $`Q`$.
- **Norm.** A norm $`\lVert\cdot\rVert`$ on $`\mathbb R^{p+q}`$ is fixed, with dual norm $`\lVert\cdot\rVert_*`$.

<a id="s5-2"></a>

**Remark 5.2 (Relation to stochastic modified equations, TracIn, influence functions and metagradients).**

- Proposition 5.1 is a Taylor expansion of the complete-state increment, inserted into a telescoping sum over rounds. It is the same kind of calculation as weak approximation and stochastic modified equations ([Li, Tai & E, 2019](../references.md#li2019stochastic)), TracIn ([Pruthi et al., 2020](../references.md#pruthi2020estimating)), influence functions ([Koh & Liang, 2017](../references.md#koh2017understanding)), metagradients ([Engstrom et al., 2025](../references.md#engstrom2025optimizing)) and In-Run Data Shapley ([Wang et al., 2025](../references.md#wang2025data)). No novelty is claimed for it; it is the tool from which the later results are derived.
- What the bookkeeping makes explicit:
  - *The object.* The first-order statistic is the $`U`$-projection of the mean increment of the complete state, including the optimizer moments $`m`$ and $`v`$. It is neither the mean gradient nor the loss. The optimizer map $`\Delta`$ alone decides which features of the gradient law enter (Theorem 5.10).
  - *Special cases.* For SGD, (a) with $`k=2`$ is the local SGD expansion of [Theorem 4.7](../statements/04-07_update_composition_amplifies_information_requirement.md#s4-7). (c) is [Theorem 4.2(a)–(c)](../statements/04-02_sufficiency_duality_transfer.md#s4-2) applied to the linearized response class. (b) is the telescoping identity of [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2) with a local expansion inserted at each round.
- *What the remainder measures (order-of-magnitude heuristic; the bound itself is (b) with $`M_2`$).* Split the increment into blocks.
  - *Parameter block.* $`\Delta^\theta=O(\eta)`$, so its second-order contribution $`\tfrac12\nabla^2_{\theta\theta}V[\Delta^\theta,\Delta^\theta]`$ is $`O(\eta^2)`$.
  - *First-moment block (momentum $`m`$).* $`\Delta^m=O(1-\beta_1)`$. The continuation depends on $`m`$ through later parameter increments, linearly to leading order, with coefficient $`O(\eta/(1-\beta_1))`$. So $`\nabla^2_{mm}V=O(\eta^2/(1-\beta_1)^2)`$, and this block also contributes $`O(\eta^2)`$.
  - *Second-moment block (Adam's $`v`$).* $`\Delta^v=(1-\beta_2)(G^{\odot2}-v)`$. The continuation depends on $`v`$ through the denominators of later steps, which carry a factor $`\eta`$ and no further power of it. The block's second-order contribution is therefore first order in $`\eta`$. Its size relative to the block's first-order term is set by $`\Delta v_i/v_i^{\rm next}`$, where $`v^{\rm next}`$ is the second moment used by the next step: about the warmth $`\kappa_i`$ (Definition 5.4) at a warm state, and of order one at the cold state.
  - Hence, wherever $`(\mathrm S_1)`$ holds, the remainder of (b) is $`O(\eta^2)`$ per round for SGD, momentum and sign-type optimizers. For Adam-type optimizers over more than one round it is first order in $`\eta`$, with relative size of the order of the warmth. At $`R=1`$ the continuation after the boundary is $`F_E(\theta)`$, which does not depend on $`o`$, and the remainder is $`O(\eta^2)`$ for every optimizer.
- *Necessity in practice.* (c) detects a mismatch only when the first-order term exceeds the remainder bound. Where the remainder is $`O(\eta^2)`$, any fixed mismatch is detected for small enough $`\eta`$. For Adam-type multi-round responses the remainder is also first order in $`\eta`$, so detection requires the margin condition of [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9), and where the second-moment channel dominates a finite measurement is needed ([Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8), [Remark 7.5](../statements/07-05_where_first_order_rankers_fail.md#s7-5)).
- *Cost.* $`\mu_t^P(z)`$ needs only consumed-gradient evaluations at the state $`z`$; no training tail is needed ([Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4)). The constants $`\Lambda_t`$ and $`M_{2,t}`$ are properties of the continuation and are not estimated by sampling messages.

**What it says.** Proposition 5.1 is a Taylor expansion inserted into a telescoping sum, the same kind of calculation as stochastic modified equations, TracIn, influence functions, metagradients and In-Run Data Shapley. What its bookkeeping adds is the complete state: Adam's second-moment block makes the remainder first order in the learning rate.

**Role in the framework.** Fixes the novelty claim (none for Proposition 5.1) and explains when first-order detection needs a margin condition.

## References

- [Li, Q., Tai, C., & E, W. (2019). Stochastic modified equations and dynamics of stochastic gradient algorithms I: Mathematical foundations. *Journal of Machine Learning Research*, vol. 20, no. 40.](../references.md#li2019stochastic)
- [Pruthi, G., Liu, F., Kale, S., & Sundararajan, M. (2020). Estimating training data influence by tracing gradient descent. *NeurIPS*.](../references.md#pruthi2020estimating)
- [Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. *ICML*.](../references.md#koh2017understanding)
- [Engstrom, L., Ilyas, A., Chen, B., Feldmann, A., Moses, W., & Madry, A. (2025). Optimizing ML training with metagradient descent. arXiv:2503.13751.](../references.md#engstrom2025optimizing)
- [Wang, J. T., Mittal, P., Song, D., & Jia, R. (2025). Data Shapley in one training run. *ICLR*.](../references.md#wang2025data)
