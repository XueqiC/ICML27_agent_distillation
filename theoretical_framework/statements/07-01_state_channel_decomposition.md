# Theorem 7.1 · State-channel decomposition

*Statement*

[← Framework](../README.md) · [Statement](07-01_state_channel_decomposition.md) · [Proof](../proofs/07-01_state_channel_decomposition.md) · [Experiments](../experiments/07-01_state_channel_decomposition.md) · [中文](07-01_state_channel_decomposition_ZH.md)

**In one paragraph.** Under Adam, one training message changes more than the parameters: it also changes the running moments $`m`$ and $`v`$, and through them every later step of training. This theorem computes, for Adam/AdamW with bias correction, decoupled weight decay, $`\epsilon`$ and global-norm clipping, the exact gradient of the eventual outcome with respect to the whole optimizer state, by a backward (costate) recursion along the rest of training. With that gradient, the first-order value of a message splits exactly into three parts: what it does through the parameters, through the momentum, and through the second moment. Together with [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6) it forms **Theorem B** of the framework.

**Setting used on this page.**

- *Complete learner state.* $`z=(\theta,o)`$ with parameters $`\theta`$ and optimizer state $`o=(m,v,s)`$: Adam's first moment $`m`$, second moment $`v`$ (both coordinatewise) and the absolute step counter $`s`$ ([Definition 1.3](../README.md#s1-3)).
- *AdamW update* (Adam of [Kingma & Ba, 2015](#ref-kingma2015adam) with the decoupled weight decay of [Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)). Hyperparameters $`\beta_1\in[0,1)`$, $`\beta_2\in(0,1)`$, learning rate $`\eta>0`$, $`\epsilon\ge0`$ and decoupled weight decay $`\lambda_{\rm wd}\ge0`$. A consumed gradient $`g`$ at step $`s`$ produces

```math
m'=\beta_1m+(1-\beta_1)g,\qquad v'=\beta_2v+(1-\beta_2)\,g\odot g,\qquad \theta'=(1-\eta\lambda_{\rm wd})\,\theta-\eta\,q_{1,s}\,\frac{m'}{\sqrt{q_{2,s}v'}+\epsilon},
```

- $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$ and $`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$ are the bias-correction factors ($`q_{1,s}=q_{2,s}=1`$ without bias correction). Vector operations are coordinatewise.
- *Increment.* $`\Delta(z,g)=(\theta'-\theta,\;m'-m,\;v'-v)`$, the Adam increment map of [Proposition 5.5](05-05_exact_first_order_statistic_adam.md#s5-5). The step counter advances by one and is never differentiated.
- *Message and reference tail.* At round $`t`$ the learner is at $`z_t=(\theta_t,m_t,v_t,s_t)`$. A message with consumed gradient $`g`$ moves it to $`z_t+\Delta_t(z_t,g)`$. The reference tail then runs updates $`k=t+1,\dots,R-1`$; update $`k`$ uses the consumed gradient $`g_k(\theta)`$ of its own data, which is the loss gradient $`\nabla\ell_k(\theta)`$ or its global-norm clipped version $`\mathrm{clip}_c(\nabla\ell_k(\theta))`$ ([Pascanu et al., 2013](#ref-pascanu2013difficulty)). The final score is $`F(\theta_R)`$.
- *Continuation value.* $`V_{t+1}(z)`$ is the final score reached when the tail starts from $`z`$ ([Definition 1.4](../README.md#s1-4)). With tail randomness $`\omega`$ (data order, sampling), $`V_{t+1}=\mathbb E_\omega V_{t+1,\omega}`$.
- *First-order value of a message.* $`\mathrm{Val}(g)=\langle\nabla V_{t+1}(z_t),\Delta_t(z_t,g)\rangle`$, the first-order term of $`V_{t+1}(z_t+\Delta_t(z_t,g))-V_{t+1}(z_t)`$ in the expansion of [Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1). For a message law $`P`$, $`\mathrm{Val}(P)=\mathbb E_P\mathrm{Val}(G)=\langle\nabla V_{t+1}(z_t),\mu^P_t(z_t)\rangle`$, with $`\mu^P_t`$ the mean increment.

<a id="s7-1"></a>

**Theorem 7.1 (State-channel decomposition).** Fix a realization of the tail randomness. Assume:

**(D1)** $`F`$ is $`C^1`$, and each consumed gradient $`g_k`$ is $`C^1`$ in $`\theta`$ near the tail trajectory started at $`z_t`$, with Jacobian $`J_k=\partial g_k/\partial\theta`$. Without clipping, $`J_k=H_k=\nabla^2\ell_k(\theta_k)`$. With global-norm clipping active at threshold $`c`$, and $`\lVert\nabla\ell_k(\theta_k)\rVert\ne c`$ along the trajectory,

```math
J_k=\frac{c}{\lVert r_k\rVert}\Big(I-\frac{r_kr_k^\top}{\lVert r_k\rVert^2}\Big)H_k,\qquad r_k=\nabla\ell_k(\theta_k).
```

**(D2)** At every tail update $`v'_k>0`$ coordinatewise.

Along the tail started at $`z_t`$, write $`m'_k,v'_k`$ for the moments after update $`k`$, and

```math
D_1^{(k)}=\mathrm{diag}\Big(\frac{q_{1,s_k}}{\sqrt{q_{2,s_k}v'_k}+\epsilon}\Big),\qquad D_2^{(k)}=\mathrm{diag}\Big(\frac{q_{1,s_k}\,q_{2,s_k}\,m'_k}{2\sqrt{q_{2,s_k}v'_k}\,\big(\sqrt{q_{2,s_k}v'_k}+\epsilon\big)^2}\Big).
```

Define costates $`\chi_k=(\chi^\theta_k,\chi^m_k,\chi^v_k)`$ by $`\chi_R=(\nabla F(\theta_R),0,0)`$ and, for $`k=R-1,\dots,t+1`$,

```math
\begin{aligned}
\chi^m_k&=\beta_1\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big),\qquad \chi^v_k=\beta_2\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big),\\
\chi^\theta_k&=(1-\eta\lambda_{\rm wd})\chi^\theta_{k+1}+J_k^\top\Big[(1-\beta_1)\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big)+2(1-\beta_2)\,\mathrm{diag}(g_k)\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big)\Big].
\end{aligned}
```

Then:

**(a) Costate identity.** $`V_{t+1}`$ is differentiable at $`z_t`$ in $`(\theta,m,v)`$, and $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$.

**(b) Three channels.** For every message $`g`$,

```math
\mathrm{Val}(g)=\underbrace{\big\langle\chi^\theta_{t+1},\Delta\theta(g)\big\rangle}_{\text{parameter channel}}+\underbrace{\big\langle\chi^m_{t+1},(1-\beta_1)(g-m_t)\big\rangle}_{\text{momentum channel}}+\underbrace{\big\langle\chi^v_{t+1},(1-\beta_2)(g\odot g-v_t)\big\rangle}_{\text{second-moment channel}} .
```

**(c) What each channel reads.**

- The second-moment channel depends on $`g`$ only through $`g\odot g`$. It is the effect of the message on all later step sizes, through the value of $`v`$ it leaves behind.
- The momentum channel is affine in $`g`$.
- The parameter channel contains the message's own step, which is a nonlinear function of $`g`$ because $`g`$ also enters its own denominator.
- For a law $`P`$, the second-moment channel depends on $`P`$ only through $`\mathbb E_P\,G\odot G`$.

**(d) Linear-response closed form.** If $`J_k=0`$ for all tail updates (the tail gradients do not depend on $`\theta`$) and $`\lambda_{\rm wd}=0`$, then $`\chi^\theta_k\equiv\nabla F`$ and

```math
\chi^m_{t+1}=-\eta\sum_{k=t+1}^{R-1}\beta_1^{\,k-t}D_1^{(k)}\nabla F,\qquad \chi^v_{t+1}=\eta\sum_{k=t+1}^{R-1}\beta_2^{\,k-t}D_2^{(k)}\nabla F .
```

**(e) Exogenous random tails.** Suppose the law of the tail randomness $`\omega`$ does not depend on the learner state, as for a fixed data order or sampling that does not look at the learner. If the tail has finitely many random branches, each satisfying (D1)–(D2), or more generally if the branchwise costates are dominated so that differentiation and expectation commute, then $`\nabla V_{t+1}(z_t)=\mathbb E_\omega\chi_{t+1,\omega}`$, and (b) holds with $`\mathbb E_\omega\chi_{t+1,\omega}`$.

**(f) State-dependent sampling.** Suppose instead that the tail has finitely many branches $`\omega`$ whose probabilities $`p_\omega(z)`$ depend on the state the tail starts from, as when later feedback is chosen from the current student's own rollouts. Assume each branch satisfies (D1)–(D2) and each $`p_\omega`$ is $`C^1`$ at $`z_t`$; this holds, for example, when each sampling step draws $`\omega_k`$ with a probability $`\pi_k(\omega_k\mid z_k)`$ that is $`C^1`$ in the state, so that $`p_\omega(z)=\prod_k\pi_k(\omega_k\mid z_k)`$ along the branch. Then

```math
\nabla V_{t+1}(z_t)=\sum_\omega p_\omega(z_t)\,\chi_{t+1,\omega}+\sum_\omega V_{t+1,\omega}(z_t)\,\nabla p_\omega(z_t).
```

- The second term does not vanish in general, also with finitely many branches. Because $`\sum_\omega\nabla p_\omega=0`$, it equals $`\sum_\omega\big(V_{t+1,\omega}(z_t)-b\big)\nabla p_\omega(z_t)`$ for every constant $`b`$, and, when every $`p_\omega(z_t)>0`$, $`\mathbb E_\omega\big[V_{t+1,\omega}(z_t)\nabla\log p_\omega(z_t)\big]`$.
- The three-channel split (b) holds with this vector in place of $`\chi_{t+1}`$: the channels are its $`\theta`$-, $`m`$- and $`v`$-blocks, and the sampling term enters every block on which the branch probabilities $`p_\omega`$ depend. Through the tail dynamics this can include all three: a sampling step that looks only at later parameters $`\theta_k`$ still makes $`p_\omega`$ depend on $`m_t`$ and $`v_t`$. What changes is the continuation gradient that weights the channels, not the decomposition.
- If the state dependence can be reparametrized, (e) applies instead: the tail is driven by exogenous noise $`u`$ whose law does not depend on the state, and for each $`u`$ every consumed gradient $`g_k`$ is a $`C^1`$ function of $`\theta_k`$ alone, with $`J_k`$ its total derivative, which includes the dependence of the sampled data on $`\theta_k`$. Discrete sampling of trajectories or of feedback does not satisfy this, and (f) is then the version to use.

**What it says.**

- The gradient of the final outcome with respect to the complete Adam state at the time of a message is computable by one backward pass along the rest of training, with the explicit recursion above.
- A message's first-order value is the sum of a parameter term, a momentum term and a second-moment term; nothing else enters.
- The second-moment term sees only the squared gradient $`g\odot g`$: two messages with the same squared gradients act identically through $`v`$.

**Intuition.**

- A message changes the current step, but it also leaves a different $`m`$ and a different $`v`$ behind. A larger $`v`$ shrinks every later step until it decays.
- The costates $`\chi^m`$ and $`\chi^v`$ are the exchange rates between "a little more momentum" or "a little more second moment" now and the final outcome.

**Role in the framework.**

- It makes the optimizer-state part of the first-order representation of [Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1) computable for Adam.
- Its scope with respect to sampling: the costate average of (e) is the continuation gradient only when the tail randomness is exogenous. When later feedback is chosen from the student's own rollouts, the continuation gradient carries the sampling term of (f). The transplant test of [Remark 9.4](09-04_scope_of_evidence.md#s9-4) uses a tail with fixed batches, to which (e) with a single branch applies.
- With [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6) it forms Theorem B: this theorem gives the exact first-order value of a message through the three channels, and Proposition 7.6 shows by an exact example that the momentum and second-moment channels cannot be dropped. Neither needs stationarity, linear response or a mean-field substitution.
- [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2) evaluates it in closed form in a restricted mean-field model, whose real-model prediction was not supported; [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3) uses it inside that model to show that parameter-only and cosine influence scores are inconsistent; [Remark 7.5](07-05_where_first_order_rankers_fail.md#s7-5) states where its first-order use fails.

**Derivation outline.**

- One Adam update is a $`C^1`$ map of $`(\theta,m,v)`$. Its Jacobian has the blocks $`\partial m'/\partial m=\beta_1I`$, $`\partial v'/\partial v=\beta_2I`$, $`\partial\theta'/\partial m'=-\eta D_1`$ and $`\partial\theta'/\partial v'=\eta D_2`$, plus the $`J_k`$ terms through $`g_k(\theta)`$.
- The continuation value is $`F`$ composed with the tail maps. The chain rule, accumulated from the end (an adjoint pass), gives $`\chi_k=D\Phi_k^\top\chi_{k+1}`$; multiplying the transposed blocks gives the recursion.
- The clipping Jacobian is the derivative of $`c\,r/\lVert r\rVert`$.

**Uses.**

- [Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1): defines the first-order value as a pairing with the mean increment.
- [Proposition 5.5](05-05_exact_first_order_statistic_adam.md#s5-5): the exact Adam increment map.

**Used by.**

- [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2): closed-form amplitude law in a restricted mean-field model.
- [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6): the channels in which two messages with equal parameter increments differ.
- [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3): inconsistency of two proxy classes.
- [Remark 7.5](07-05_where_first_order_rankers_fail.md#s7-5): where first-order rankers fail.
- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2) and [Remark 9.3](09-03_derivation_design_rules.md#s9-3): magnitude control in the method.

## Related work

- The costate recursion is reverse-mode differentiation through training, written out in closed form for the Adam state. The same principle computes exact gradients through SGD with momentum ([Maclaurin et al., 2015](#ref-maclaurin2015gradient)) and metagradients through large-scale training ([Engstrom et al., 2025](#ref-engstrom2025optimizing)), with respect to hyperparameters or data weights.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-engstrom2025optimizing"></a>

**Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, Aleksander Madry (2025).** Optimizing ML training with metagradient descent. arXiv preprint. arXiv [2503.13751](https://arxiv.org/abs/2503.13751).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-maclaurin2015gradient"></a>

**Dougal Maclaurin, David Duvenaud, Ryan Adams (2015).** Gradient-based hyperparameter optimization through reversible learning. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37:2113–2122. [proceedings.mlr.press/v37/maclaurin15](https://proceedings.mlr.press/v37/maclaurin15.html).

<a id="ref-pascanu2013difficulty"></a>

**Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013).** On the difficulty of training recurrent neural networks. *Proceedings of the 30th International Conference on Machine Learning (ICML)*, PMLR 28, no. 3, 1310–1318. [proceedings.mlr.press/v28/pascanu13](https://proceedings.mlr.press/v28/pascanu13.html).
