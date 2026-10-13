# Proposition 7.3 · Inconsistency of two proxy classes

*Statement*

[← Framework](../README.md) · [Statement](07-03_inconsistency_two_proxy_classes.md) · [Proof](../proofs/07-03_inconsistency_two_proxy_classes.md) · [Experiments](../experiments/07-03_inconsistency_two_proxy_classes.md) · [中文](07-03_inconsistency_two_proxy_classes_ZH.md)

**In one paragraph.** Data-selection and data-attribution methods often rank training examples by a cheap score computed at the current state: a statistic of the data alone, the predicted change of the loss through the parameter step, or the cosine between a training gradient and a target gradient. This proposition proves that two such classes are not consistent rankers. A statistic that ignores the deployment task must rank some pair of feedback laws wrongly on some task. And under warm Adam, in the regime of the amplitude law of [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2), parameter-only scores rank two same-direction messages in the wrong order, while cosine scores cannot tell them apart.

**Setting used on this page.**

- *Feedback laws and learning value.* A feedback law $`P`$ (message law) is applied by the actual learner at a fixed state; $`L_F(P)`$ is the resulting expected final score for a deployment task with score $`F\in[0,1]`$ ([Definitions 1.3–1.4](../README.md#s1-3)). "Ranks correctly" means ordering laws by $`L_F`$.
- *Task-blind statistic.* A real-valued $`\mathcal T(P)`$ that depends only on the current state and the feedback law, not on the task, used to rank laws by its value.
- *Adam regime of Corollary 7.2.* One coordinate $`i`$ of warm Adam without bias correction, $`\epsilon=0`$, no decay, tail gradients independent of $`\theta`$ with mean $`\bar g_i\ne0`$ and mean square $`\bar v_i`$, score affine in $`\theta_i`$, and $`T`$ tail updates after the message.
  - $`\alpha_i=-(\nabla F)_i\,\mathrm{sign}(\bar g_i)`$; $`\lambda=g_i\,\mathrm{sign}(\bar g_i)/\sqrt{\bar v_i}`$ is the message amplitude in units of the running RMS; $`\rho_i=\lvert\bar g_i\rvert/\sqrt{\bar v_i}`$ is the signal-to-noise ratio.
  - $`\Delta F_i(\lambda)`$ is the mean-field response of Corollary 7.2(a); $`Q_i`$, $`S_i`$, $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho_i\omega_T)`$ and the error bound are those of Corollary 7.2(b), with $`\omega_T=1-\beta_2^{\,T+1}`$.
- *Parameter-only score.* $`S_{\rm PO}(g)=\langle\chi^\theta,\Delta\theta(g)\rangle`$, the parameter channel of [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1) alone: the continuation gradient restricted to the parameters, paired with the message's own Adam step. TracIn-style scores ([Pruthi et al., 2020](#ref-pruthi2020estimating)) computed with Adam steps and the fixed-state In-Run Data Shapley for Adam ([Ding et al., 2026](#ref-ding2026inrun)) belong to this class.
- *Cosine score.* Any score that is a function of the direction of the message gradient only (or, in one coordinate, of the direction of the parameter step). LESS ([Xia et al., 2024](#ref-xia2024less)) belongs to this class.

<a id="s7-3"></a>

**Proposition 7.3 (Inconsistency of two proxy classes).**

- **(a) Task-blind statistics.** Let $`\mathcal T`$ be any statistic that depends only on the current state and the feedback law, not on the deployment task, and that ranks laws by its value. Then there are two tasks and two laws $`P,Q`$ at the same learner and state such that the true order of $`P,Q`$ is opposite on the two tasks while $`\mathcal T(P),\mathcal T(Q)`$ are the same on both. Hence $`\mathcal T`$ ranks wrongly, or fails to separate a nonzero difference, on at least one task.
- **(b) Parameter-only and cosine scores.** In the setting of [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2) with $`\alpha_i>0`$, take two same-direction messages with amplitudes $`0<\lambda_1<\lambda_2`$.
  - The parameter-only score satisfies $`S_{\rm PO}=\alpha_i\eta\,h(\lambda)`$ with $`h(\lambda)=(\beta_1\rho_i+(1-\beta_1)\lambda)/\sqrt{\beta_2+(1-\beta_2)\lambda^2}`$. The function $`h`$ is strictly increasing for $`\lambda<\lambda_{\rm PO}:=(1-\beta_1)\beta_2/(\beta_1\rho_i(1-\beta_2))`$. This class contains TracIn-style scores with Adam steps and fixed-state Adam data attribution. If $`\lambda_2<\lambda_{\rm PO}`$, it ranks the larger message first.
  - A cosine score is a function of the direction of the message gradient only, or, in one coordinate, of the direction of the parameter step. This class contains LESS-type scores. Every such score gives the two messages the same value.
  - The exact value prefers the smaller message whenever $`\Delta F_i(\lambda_2)<\Delta F_i(\lambda_1)`$. By [Corollary 7.2(b)](07-02_stationary_amplitude_law.md#s7-2) this holds when $`\lambda_1+\lambda_2>2\lambda^\star`$ and $`Q_i(\lambda_1)-Q_i(\lambda_2)>11(1-\beta_2)\max\lbrace1,\lambda_2^2\rbrace\big(S_i(\lambda_1)+S_i(\lambda_2)\big)`$. In general it can be read from [Corollary 7.2(a)](07-02_stationary_amplitude_law.md#s7-2).
  - In this region the parameter-only score ranks the pair in the wrong order and the cosine score cannot separate it.

**What it says.**

- No ranking rule that ignores the deployment task can be right for every task.
- Under warm Adam there is an explicit range of message amplitudes where the larger of two same-direction messages is worse, yet a parameter-only score prefers it and a cosine score ties the two.

**Intuition.**

- The parameter-only score sees the bigger immediate step and misses the damage the big message does through $`v`$ to every later step.
- The cosine sees only the direction, and the two messages point the same way.

**Role in the framework.**

- It is a proof, not only an empirical observation, that two widely used families of influence scores are not consistent rankers under Adam.
- It is one reason the method selects feedback by measured consequence ([Remark 9.3](09-03_derivation_design_rules.md#s9-3), Rule 2) and flags large messages before ranking them ([Algorithm 9.2](09-02_ced_algorithm.md#s9-2), step 4).
- Proved inconsistency is limited to these two classes; for other static proxies the framework reports only that the proxies tested did not reliably select.

**Derivation outline.**

- (a): reversing the task ($`F\mapsto1-F`$) reverses every true order but no task-blind statistic. A natural pair is the two label packets of [Proposition 3.1](03-01_static_value_not_learning_value.md#s3-1)(b).
- (b): the parameter-only score is the message's own step, whose derivative in $`\lambda`$ has the sign of $`(1-\beta_1)\beta_2-\beta_1\rho_i(1-\beta_2)\lambda`$. The exact value follows the concave $`Q_i`$, which decreases past $`\lambda^\star`$; positive multiples of one gradient have the same direction, so cosines tie.

**Illustration (analytic).** For a long remaining horizon $`2\lambda^\star\approx2/\rho_i`$, while $`\lambda_{\rm PO}\approx(1-\beta_1)/(\beta_1\rho_i(1-\beta_2))`$, about $`11/\rho_i`$ at $`\beta_1=0.9`$, $`\beta_2=0.99`$. So the range where the parameter-only score prefers the larger message ($`\lambda_2<\lambda_{\rm PO}`$) overlaps the range where the exact value prefers the smaller one ($`\lambda_1+\lambda_2>2\lambda^\star`$, within the error of Corollary 7.2(b)). Every pair in the overlap is misranked by the parameter-only score and tied by every cosine score.

**Uses.**

- [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2): the exact and quadratic amplitude laws.
- [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1): the parameter channel that defines $`S_{\rm PO}`$.
- [Proposition 3.1](03-01_static_value_not_learning_value.md#s3-1): two packets whose learning-value difference changes sign between tasks.

**Used by.**

- [Remark 7.4](07-04_relation_prior_work_role_clipping.md#s7-4): placing TracIn-style, fixed-state Shapley and LESS scores.
- [Remark 7.5](07-05_where_first_order_rankers_fail.md#s7-5): where first-order rankers fail.
- [Remark 9.3](09-03_derivation_design_rules.md#s9-3) and [Algorithm 9.2](09-02_ced_algorithm.md#s9-2): selection by measured consequence.

## References

- [Pruthi et al., 2020](../references.md#pruthi2020estimating): TracIn.
- [Ding et al., 2026](../references.md#ding2026inrun): In-Run Data Shapley for Adam (fixed-state).
- [Xia et al., 2024](../references.md#xia2024less): LESS.
