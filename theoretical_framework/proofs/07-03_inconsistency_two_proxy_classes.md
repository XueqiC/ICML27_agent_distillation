# Proposition 7.3 · Inconsistency of two proxy classes

*Proof*

[← Framework](../README.md) · [Statement](../statements/07-03_inconsistency_two_proxy_classes.md) · [Proof](07-03_inconsistency_two_proxy_classes.md) · [Experiments](../experiments/07-03_inconsistency_two_proxy_classes.md) · [中文](07-03_inconsistency_two_proxy_classes_ZH.md)

**What is proved here.** (a) A ranking statistic that does not look at the deployment task is wrong, or fails to separate, on some task, because reversing the task reverses every true order but not the statistic. (b) In the warm, stationary Adam regime of the amplitude law, the parameter-only score of a message is an explicit increasing function of its amplitude over a stated range, cosine scores cannot distinguish two positive multiples of one gradient, and the exact value decreases past the optimal amplitude; so on an explicit set of pairs the first ranks wrongly and the second ties.

**Setting used on this page.**

- *Learning value.* For a feedback law $`P`$ applied by the actual learner at a fixed state, $`L_F(P)`$ is the expected final score for a task with score $`F\in[0,1]`$ ([Definitions 1.3–1.4](../README.md#s1-3)).
- *Task-blind statistic.* $`\mathcal T(P)`$ depends only on the state and the law, not on the task, and ranks laws by its value.
- *Adam regime.* One coordinate $`i`$ of warm Adam without bias correction ($`\epsilon=0`$, no decay); tail gradients independent of $`\theta`$, stationary with mean $`\bar g_i\ne0`$ and mean square $`\bar v_i`$; score affine in $`\theta_i`$; $`T`$ tail updates after the message; warm state $`m_t=\bar g_i`$, $`v_t=\bar v_i`$.
- $`\alpha_i=-(\nabla F)_i\,\mathrm{sign}(\bar g_i)`$, $`\lambda=g_i\,\mathrm{sign}(\bar g_i)/\sqrt{\bar v_i}`$, $`\rho_i=\lvert\bar g_i\rvert/\sqrt{\bar v_i}`$, $`\omega_T=1-\beta_2^{\,T+1}`$, $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho_i\omega_T)`$.
- $`\Delta F_i(\lambda)`$, $`Q_i(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho_i\omega_T\lambda^2`$, $`S_i(\lambda)=\lvert\lambda\rvert(1-\beta_1^{\,T+1})+\tfrac12\rho_i\omega_T\lambda^2`$ and the bound $`\lvert\Delta F_i-\alpha_i\eta Q_i\rvert\le11(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\lvert\alpha_i\rvert\eta S_i`$ are those of [Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2).
- $`S_{\rm PO}(g)=\langle\chi^\theta,\Delta\theta(g)\rangle`$ is the parameter channel of [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1); TracIn-style scores ([Pruthi et al., 2020](#ref-pruthi2020estimating)) with Adam steps and the fixed-state In-Run Data Shapley for Adam ([Ding et al., 2026](#ref-ding2026inrun)) are of this form. A cosine score is a function of the direction of the message gradient (in one coordinate, of the direction of the parameter step), as in LESS ([Xia et al., 2024](#ref-xia2024less)).

**Statement ([Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes.md#s7-3)).**

- **(a)** For any task-blind ranking statistic $`\mathcal T`$ there are two tasks and two laws $`P,Q`$ at the same learner and state whose true order is opposite on the two tasks while $`\mathcal T(P),\mathcal T(Q)`$ are the same on both; so $`\mathcal T`$ ranks wrongly, or fails to separate a nonzero difference, on at least one task.
- **(b)** In the Adam regime with $`\alpha_i>0`$ and same-direction amplitudes $`0<\lambda_1<\lambda_2`$: $`S_{\rm PO}=\alpha_i\eta\,h(\lambda)`$ with $`h(\lambda)=(\beta_1\rho_i+(1-\beta_1)\lambda)/\sqrt{\beta_2+(1-\beta_2)\lambda^2}`$, strictly increasing for $`\lambda<\lambda_{\rm PO}=(1-\beta_1)\beta_2/(\beta_1\rho_i(1-\beta_2))`$, so it ranks the larger message first when $`\lambda_2<\lambda_{\rm PO}`$; every cosine score ties the two messages; the exact value prefers the smaller message whenever $`\Delta F_i(\lambda_2)<\Delta F_i(\lambda_1)`$, which holds when $`\lambda_1+\lambda_2>2\lambda^\star`$ and $`Q_i(\lambda_1)-Q_i(\lambda_2)>11(1-\beta_2)\max\lbrace1,\lambda_2^2\rbrace\big(S_i(\lambda_1)+S_i(\lambda_2)\big)`$.

**Proof.**

*(a)* Take any task with score $`F`$ and two laws with $`L_F(P)\ne L_F(Q)`$, and the task with score $`F'=1-F\in[0,1]`$. Since $`L_{F'}=1-L_F`$, we get $`L_{F'}(P)-L_{F'}(Q)=-(L_F(P)-L_F(Q))`$, while $`\mathcal T`$, which does not see the task, is unchanged. If $`\mathcal T(P)\ne\mathcal T(Q)`$ it orders $`P,Q`$ the same way on both tasks and is wrong on one; if $`\mathcal T(P)=\mathcal T(Q)`$ it fails to separate a nonzero difference.

A natural pair of tasks with the same property is given by [Proposition 3.1](../statements/03-01_static_value_not_learning_value.md#s3-1)(b) (equal label marginals and mean gradient do not fix the learning value): two label packets with equal marginals and equal mean gradient have learning values whose difference changes sign between $`J(p)=p^2`$ and $`J(p)=1-p^2`$.

*(b)* In the linear-response regime $`\chi^\theta=\nabla F`$ ([Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1)(d): the parameter costate is constant when the tail gradients do not depend on $`\theta`$). From the warm state, the message's own step is

```math
\Delta\theta_i(g)=-\eta\,\frac{\beta_1\bar g_i+(1-\beta_1)g_i}{\sqrt{\beta_2\bar v_i+(1-\beta_2)g_i^2}}=-\eta\,\mathrm{sign}(\bar g_i)\,h(\lambda),
```

so $`S_{\rm PO}=(\nabla F)_i\Delta\theta_i=\alpha_i\eta\,h(\lambda)`$. Differentiating,

```math
h'(\lambda)=\frac{(1-\beta_1)\beta_2-\beta_1\rho_i(1-\beta_2)\lambda}{\big(\beta_2+(1-\beta_2)\lambda^2\big)^{3/2}},
```

which is positive for $`\lambda<\lambda_{\rm PO}`$. With $`\alpha_i>0`$ and $`\lambda_1<\lambda_2<\lambda_{\rm PO}`$, $`S_{\rm PO}(\lambda_2)>S_{\rm PO}(\lambda_1)`$.

- Positive multiples of one gradient have the same direction. In one coordinate, both parameter steps point along $`-\mathrm{sign}(\bar g_i)`$. So any cosine score ties the two messages.
- For the exact value, $`Q_i(\lambda_2)-Q_i(\lambda_1)=(\lambda_2-\lambda_1)\big[(1-\beta_1^{\,T+1})-\tfrac12\rho_i\omega_T(\lambda_1+\lambda_2)\big]`$, which is negative iff $`\lambda_1+\lambda_2>2\lambda^\star`$. By [Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2)(b) (the quadratic law with its explicit error bound), $`\Delta F_i(\lambda)`$ lies within $`11(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\lvert\alpha_i\rvert\eta S_i(\lambda)`$ of $`\alpha_i\eta Q_i(\lambda)`$ at each amplitude, and $`\max\lbrace1,\lambda_1^2\rbrace\le\max\lbrace1,\lambda_2^2\rbrace`$. So the stated margin implies $`\Delta F_i(\lambda_2)<\Delta F_i(\lambda_1)`$. $`\square`$

**Remarks on the proof.**

- (b) is a statement about small increments, through the quadratic law of Corollary 7.2(b); for large increments the response must be read from Corollary 7.2(a), and the conclusion holds whenever that formula gives $`\Delta F_i(\lambda_2)<\Delta F_i(\lambda_1)`$.
- The interval $`2\lambda^\star<\lambda_1+\lambda_2`$ with $`\lambda_2<\lambda_{\rm PO}`$ is nonempty in typical settings. For a long horizon $`2\lambda^\star\approx2/\rho_i`$, while $`\lambda_{\rm PO}\approx(1-\beta_1)/(\beta_1\rho_i(1-\beta_2))`$, about $`11/\rho_i`$ at $`\beta_1=0.9`$, $`\beta_2=0.99`$.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-ding2026inrun"></a>

**Meng Ding, Zeqing Zhang, Di Wang, Lijie Hu (2026).** In-Run Data Shapley for Adam optimizer. arXiv preprint. arXiv [2602.00329](https://arxiv.org/abs/2602.00329).

<a id="ref-pruthi2020estimating"></a>

**Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan (2020).** Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 19920–19930. arXiv [2002.08484](https://arxiv.org/abs/2002.08484).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
