# Proposition 7.6 · Equal immediate parameters do not fix the continuation

*Statement*

[← Framework](../README.md) · [Statement](07-06_equal_immediate_parameters_different_continuation.md) · [Proof](../proofs/07-06_equal_immediate_parameters_different_continuation.md) · [Experiments](../experiments/07-06_equal_immediate_parameters_different_continuation.md) · [中文](07-06_equal_immediate_parameters_different_continuation_ZH.md)

**In one paragraph.** A learner without optimizer memory forgets how it reached its parameters: two messages that leave the same parameters lead to the same future, whatever training follows. Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) does not forget. This proposition shows it exactly, with standard hyperparameters and a state reached by one ordinary step. Two different gradients produce the same parameter update, because once the momentum is nonzero Adam's step is a two-to-one function of the gradient; they leave different moments behind, and the next update already differs. The construction survives $`\epsilon>0`$. It is a mechanism illustration near a cold start, not a measurement of how large the memory effect is in warm training; part (e) gives the exact warm-state identities that say what the memory carries there. Together with [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1) it forms **Theorem B** of the framework.

**Setting used on this page.**

- *Complete learner state, continuation value, response.* $`z=(\theta,o)`$ with parameters $`\theta`$ and optimizer state $`o`$ ([Definition 1.3](../README.md#s1-3)); the reference continuation value $`V^0_{t+1}`$ and the one-boundary response $`g_{t,z}(y)=\mathbb E_{\xi_t}V^0_{t+1}(U_t(z,y,\xi_t))`$ ([Definition 1.4](../README.md#s1-4)).
- *Memoryless learner.* The reference kernels $`K^0_k`$, $`k>t`$, map functions of $`(\theta,k)`$ to functions of $`(\theta,k)`$, and the deployed score depends on the parameters only, $`F_E(z)=F_E(\theta)`$. Plain SGD with a fixed learning-rate schedule and no momentum, buffer or sampler memory is memoryless. SGD with momentum is not: its momentum buffer is optimizer state.
- *Adam, one coordinate.* Hyperparameters $`\beta_1\in(0,1)`$, $`\beta_2\in(0,1)`$, learning rate $`\eta>0`$, $`\epsilon\ge0`$, bias correction, no weight decay. At step counter $`s`$ a consumed gradient $`g`$ gives

```math
m'=\beta_1m+(1-\beta_1)g,\qquad v'=\beta_2v+(1-\beta_2)g^2,\qquad \theta'=\theta-\eta\,q_{1,s}\,\frac{m'}{\sqrt{q_{2,s}v'}+\epsilon},
```

with $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$ and $`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$, as in [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1).
- *Step magnitude.* For $`\epsilon=0`$ and a fixed state $`(m,v,s)`$ put $`a=\beta_1m`$, $`b=1-\beta_1`$, $`c=\beta_2v`$, $`d=1-\beta_2`$ and

```math
S(g)=\frac{a+bg}{\sqrt{c+dg^2}},\qquad\text{so that}\qquad \theta'=\theta-\eta\,\frac{q_{1,s}}{\sqrt{q_{2,s}}}\,S(g).
```

- *Realizable gradients.* Every consumed gradient used below is the gradient of a one-dimensional squared loss $`\tfrac12(\theta-y)^2`$ at the current parameter, with the label $`y`$ chosen accordingly. Every such loss has Hessian $`1`$.

<a id="s7-6"></a>

**Proposition 7.6 (Equal immediate parameters do not fix the continuation).**

**(a) Memoryless learners.** For a memoryless learner, $`V^0_{t+1}(z)`$ depends on $`z`$ only through $`\theta`$. Hence two messages $`y_A,y_B`$ at round $`t`$ whose updates give the same law of the next parameters have equal one-boundary responses, $`g_{t,z}(y_A)=g_{t,z}(y_B)`$, for every declared task and every memoryless continuation.

**(b) Adam's step is two-to-one in the gradient.** Let $`\epsilon=0`$, $`v>0`$ and $`m>0`$ (the case $`m<0`$ is symmetric). On $`g>-a/b`$ the step magnitude $`S`$ is positive, strictly increasing on $`(-a/b,g^\star]`$ and strictly decreasing on $`[g^\star,\infty)`$, where

```math
g^\star=\frac{bc}{ad}=\frac{(1-\beta_1)\beta_2v}{\beta_1(1-\beta_2)m},\qquad S(g^\star)=\sqrt{\frac{a^2}{c}+\frac{b^2}{d}},\qquad \lim_{g\to\infty}S(g)=\frac{b}{\sqrt d}.
```

Every value strictly between $`b/\sqrt d`$ and $`S(g^\star)`$ is taken by exactly two gradients $`g_A<g^\star<g_B`$. The two messages give the same next parameter $`\theta'`$ and different first moments, $`m'_A\ne m'_B`$.

**(c) An exact instance with standard hyperparameters.** Take $`\beta_1=0.9`$, $`\beta_2=0.999`$, $`\eta=1`$, $`\epsilon=0`$, bias correction and no weight decay. From $`(\theta,m,v)=(0,0,0)`$ one step with consumed gradient $`1`$ reaches the checkpoint $`z_1=(-1,\tfrac1{10},\tfrac1{1000})`$ at step counter $`1`$. At $`z_1`$ the second-step gradients $`g_A=\tfrac12`$ and $`g_B=\tfrac{2103}{790}`$ are the two roots of $`711g^2-2248.2\,g+946.35=0`$, that is, of

```math
\frac{(0.9+g)^2}{0.999+g^2}=\frac{1960}{1249}.
```

Therefore:

- the immediate parameters agree exactly, $`\theta_2\approx-1.932179639`$ on both sides;
- the moments differ: $`(m_2,v_2)=(\tfrac7{50},\tfrac{1249}{10^6})`$ after $`g_A`$ and $`(\tfrac{1407}{3950},\tfrac{50460849}{6241\cdot10^6})`$ after $`g_B`$;
- the next update differs: with one common third-step loss whose gradient at $`\theta_2`$ is $`1`$, the squared third-step magnitudes $`m_3^2/v_3`$ are the distinct rationals $`\tfrac{51076000}{2247751}`$ and $`\tfrac{1103967076000}{56651388151}`$, and $`\theta_3^A-\theta_3^B\approx-0.0712027`$. For the bounded score $`F(\theta)=(1+e^{-\theta})^{-1}`$ the difference is $`\approx-0.00364994`$.

For any $`\eta>0`$, with the same consumed gradients, the immediate parameters again agree and the parameter difference after the third step is $`\eta`$ times the one above.

**(d) Positive $`\epsilon`$.** For every sufficiently small $`\epsilon>0`$ there is a gradient $`g_B(\epsilon)`$, continuous in $`\epsilon`$ with $`g_B(0)=\tfrac{2103}{790}`$, such that the instance of (c) run with this $`\epsilon`$ in every step again gives equal immediate parameters and a nonzero third-step difference. At $`\epsilon=10^{-8}`$, $`g_B(\epsilon)\approx2.662025475`$ and $`\theta_3^A-\theta_3^B\approx-0.0712027`$.

**(e) What the memory carries in a warm state.** At any state with $`v>0`$, a consumed gradient $`g`$ changes the second moment by the exact relative amount

```math
\frac{v'-v}{v}=(1-\beta_2)\Big(\frac{g^2}{v}-1\Big).
```

With $`v`$ held fixed, no bias correction, $`\epsilon=0`$ and tail gradients that do not depend on $`\theta`$, a perturbation $`\delta g`$ of the message moves the parameter at lag $`k=0,1,\dots`$ by $`-\eta(1-\beta_1)\beta_1^{\,k}\,\delta g/\sqrt v`$. The message's own update carries the fraction $`1-\beta_1`$ of the total first-order displacement $`-\eta\,\delta g/\sqrt v`$, and the first $`K`$ updates carry $`1-\beta_1^{\,K}`$.

**What it says.**

- Without optimizer memory, the parameters after a message summarize everything the message does for later training. With Adam they do not, even when the message's whole immediate effect is known exactly.
- The equality in (c) is not a coincidence. Once the momentum is nonzero, Adam's step is a two-to-one function of the gradient around its peak $`g^\star`$, and the two gradients leave different moments.
- In the language of [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1)(b), the two messages of (c) have identical parameter increments and differ only through the momentum and second-moment channels.
- In a warm state the memory acts in two ways. The momentum delivers most of a gradient's first-order displacement after the message's own step: $`1-\beta_1=0.1`$ of it at once, about $`0.65`$ within ten updates for $`\beta_1=0.9`$. The second moment can change by far more than $`1-\beta_2`$ when the message's gradient is large relative to $`v`$.
- *Scope.*
  - (c) is a construction near a cold start. At step counter $`2`$ the bias correction of $`v`$ is $`1/(1-\beta_2^{\,2})\approx500`$, so a single gradient dominates $`v`$.
  - It separates interfaces that read only the immediate parameter change; in one dimension every direction (cosine) score also ties, trivially. It is not a separation from LESS ([Xia et al., 2024](#ref-xia2024less)), TRAK ([Park et al., 2023](#ref-park2023trak)) or datamodels ([Ilyas et al., 2022](#ref-ilyas2022datamodels)), which read other information.
  - Under the premises of (e) the total first-order displacement is a constant multiple of the immediate one, so rankings by immediate displacement are preserved there. They can change through curvature (even with a fixed Hessian, momentum's response to an impulse depends on the curvature of each direction) and through $`v`$.

**Intuition.**

- Adam divides the momentum by a running RMS that the new gradient also enlarges. A small gradient adds little to both, and a very large gradient adds a lot to both, so the ratio saturates. In between the step peaks, and a moderate and a large gradient can produce the same step.
- The parameters cannot tell the two messages apart; the moments can. The large gradient leaves a larger momentum and a larger $`v`$, and the next update reads both.

**Role in the framework.**

- With [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1) it forms Theorem B. Theorem 7.1 computes the exact first-order value of a message through the parameters, the momentum and the second moment; this proposition shows by an exact example that the momentum and second-moment channels cannot be dropped from an interface. Neither result uses the stationarity, linear-response or mean-field premises of [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2).
- It is the reason the complete learner state ([Definition 1.3](../README.md#s1-3)), not the parameters alone, is the state the method records and conditions on ([Algorithm 9.2](09-02_ced_algorithm.md#s9-2)).
- Its real-model counterpart is the pre-registered state-transplant test ([pre-registration](../../docs/direction/2026-10-14_T1_optimizer_state_transplant_prereg_ZH.md), in Chinese). At a warm checkpoint the same parameters are continued with the momentum and second moment of a reference branch or of an alternative branch, in a 2×2 design. Its result is not yet available.
- For a parameterization $`\theta=f(q)`$, such as LoRA, the locally reachable parameter directions are the range of the Jacobian of $`f`$. This is a local statement; it does not cap the rank of the response over many updates.

**Derivation outline.**

- (a) Backward induction: $`V^0_R=F_E`$ depends on $`\theta`$ only, and a kernel that reads $`(\theta,k)`$ only maps such functions to such functions.
- (b) The sign of $`S'(g)`$ is the sign of $`bc-adg`$; the Cauchy–Schwarz inequality gives the peak value and the point where it is attained.
- (c) Exact rational arithmetic: the squared second steps $`m_2^2/v_2`$ coincide, the squared third steps $`m_3^2/v_3`$ do not.
- (d) The implicit function theorem at $`(g_B,0)`$, where $`S'(g_B)<0`$, followed by continuity of the third-step difference.
- (e) One line of algebra for $`v`$; the momentum recursion is linear.

**Uses.**

- [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1): the Adam/AdamW update with bias correction, and the three channels.
- [Proposition 5.3](05-03_cold_start_adam_sign_quotient.md#s5-3): the cold-first-step witness, which this proposition extends to later steps and to $`\epsilon>0`$.

**Used by.**

- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2): the complete learner state as the state of record.

## Related work

- [Proposition 5.3](05-03_cold_start_adam_sign_quotient.md#s5-3)(e) gives a witness at the cold first step with $`\epsilon=0`$ and $`\beta_1=\beta_2=\tfrac12`$: there Adam's step is a sign function and only the momentum keeps the magnitude. With $`\epsilon>0`$ the cold first step is a bijection of the gradient, so that witness needs $`\epsilon=0`$. This proposition shows the same failure at a later step, with standard hyperparameters, bias correction and $`\epsilon>0`$, because at any state with nonzero momentum Adam's step is two-to-one in the gradient.
- The step magnitude $`S`$ of (b) is, in units of the running RMS, the parameter-only score $`h(\lambda)`$ of [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3)(b), and its peak $`g^\star`$ is the point $`\lambda_{\rm PO}`$ of that proposition. Proposition 7.3 uses the increase of the score below $`\lambda_{\rm PO}`$ to show a misranking inside the model of Corollary 7.2; here the rise and fall around the peak give an exact tie in the immediate parameters, with no model.
- Scores that value a training example only by the update it causes under the current Adam state read exactly the quantity that (c) holds fixed. In-Run Data Shapley for Adam ([Ding et al., 2026](#ref-ding2026inrun)) scores only the current preconditioned update ([Remark 7.4](07-04_relation_prior_work_role_clipping.md#s7-4)), so it gives the two messages of (c) the same score although their continuations differ. LESS ([Xia et al., 2024](#ref-xia2024less)) compares directions of Adam-preconditioned gradients, which in one dimension carry no information at all; in many dimensions it reads more than the immediate update, and (c) says nothing about it there.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-ding2026inrun"></a>

**Meng Ding, Zeqing Zhang, Di Wang, Lijie Hu (2026).** In-Run Data Shapley for Adam optimizer. arXiv preprint. arXiv [2602.00329](https://arxiv.org/abs/2602.00329).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-park2023trak"></a>

**Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Madry (2023).** TRAK: Attributing model behavior at scale. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202:27074–27113. [proceedings.mlr.press/v202/park23c](https://proceedings.mlr.press/v202/park23c.html).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
