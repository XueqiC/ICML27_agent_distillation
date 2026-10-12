# §7 Theorem B: Adam's second-moment channel — Statements

[← Framework](../README.md) · [Proofs](../proofs/07_second_moment_channel.md) · [Experiments](../experiments/07_second_moment_channel.md) · [中文](07_second_moment_channel_ZH.md)

### §7 · Theorem B: Adam's second-moment channel — exact statements

[Proofs](../proofs/07_second_moment_channel.md) · [Experiments](../experiments/07_second_moment_channel.md)

**What this section does.**

- Computes the continuation-value gradient of Adam/AdamW by a reverse costate recursion, and splits the first-order value of a message into parameter, momentum and second-moment channels (Theorem 7.1).
- Solves the warm, stationary, linear-response case exactly for every message amplitude: an optimal amplitude, a sign change, and a large-amplitude criterion (Corollary 7.2).
- Derives from it that parameter-only and cosine influence scores are not consistent rankers under Adam (Proposition 7.3), and states where every first-order ranker is unreliable (Remark 7.5).
- "Theorem B" denotes Theorem 7.1 together with Corollary 7.2.

**Setting.** The state is $`z_t=(\theta_t,m_t,v_t,s_t)`$ and AdamW is the increment map of [Proposition 5.5(a)](../proofs/05_warm_adam.md#s5-5), with bias-correction factors $`q_{1,s},q_{2,s}`$ (both 1 without correction), decoupled decay $`\lambda_{\rm wd}`$ and consumed gradients $`g_k(\theta)`$ (possibly global-norm clipped, Jacobian $`J_k`$). A message $`g`$ at round $`t`$ is followed by the reference tail $`k=t+1,\dots,R-1`$ and the score $`F(\theta_R)`$. Its first-order value is $`\mathrm{Val}(g)=\langle\nabla V_{t+1}(z_t),\Delta_t(z_t,g)\rangle`$, the first-order term of [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1).

#### [Theorem 7.1](../proofs/07_second_moment_channel.md#s7-1) · State-channel decomposition

Assume $`F\in C^1`$, each $`g_k`$ is $`C^1`$ in $`\theta`$ near the tail trajectory (clipping away from its threshold), and $`v'_k>0`$ along the tail. Put $`D_1^{(k)}=\mathrm{diag}\big(q_{1,s_k}/(\sqrt{q_{2,s_k}v'_k}+\epsilon)\big)`$ and $`D_2^{(k)}=\mathrm{diag}\big(q_{1,s_k}q_{2,s_k}m'_k/(2\sqrt{q_{2,s_k}v'_k}(\sqrt{q_{2,s_k}v'_k}+\epsilon)^2)\big)`$. Run the costates backward from $`\chi_R=(\nabla F,0,0)`$:

```math
\begin{aligned}
\chi^m_k&=\beta_1\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big),\qquad \chi^v_k=\beta_2\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big),\\
\chi^\theta_k&=(1-\eta\lambda_{\rm wd})\chi^\theta_{k+1}+J_k^\top\Big[(1-\beta_1)\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big)+2(1-\beta_2)\,\mathrm{diag}(g_k)\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big)\Big].
\end{aligned}
```

- **(a)** $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$.
- **(b)** $`\mathrm{Val}(g)=\langle\chi^\theta_{t+1},\Delta\theta(g)\rangle+\langle\chi^m_{t+1},(1-\beta_1)(g-m_t)\rangle+\langle\chi^v_{t+1},(1-\beta_2)(g\odot g-v_t)\rangle`$: parameter, momentum and second-moment channels.
- **(c)** The second-moment channel depends on $`g`$ only through $`g\odot g`$, and on a law only through $`\mathbb E\,G\odot G`$; it is the message's effect on all later step sizes.
- **(d)** If the tail gradients do not depend on $`\theta`$ and $`\lambda_{\rm wd}=0`$: $`\chi^\theta\equiv\nabla F`$, $`\chi^m_{t+1}=-\eta\sum_{k>t}\beta_1^{\,k-t}D_1^{(k)}\nabla F`$, $`\chi^v_{t+1}=\eta\sum_{k>t}\beta_2^{\,k-t}D_2^{(k)}\nabla F`$.
- **(e)** For random tails, the expected costate, when differentiation and expectation commute.

*Derivation outline.*

- One Adam update is a $`C^1`$ map of $`(\theta,m,v)`$. Its Jacobian has the blocks $`\partial m'/\partial m=\beta_1I`$, $`\partial v'/\partial v=\beta_2I`$, $`\partial\theta'/\partial m'=-\eta D_1`$ and $`\partial\theta'/\partial v'=\eta D_2`$, plus the $`J_k`$ terms through $`g_k(\theta)`$.
- The continuation value is $`F`$ composed with the tail maps. The chain rule, accumulated from the end (an adjoint pass), gives $`\chi_k=D\Phi_k^\top\chi_{k+1}`$; multiplying the transposed blocks gives the recursion.
- The clipping Jacobian is the derivative of $`c\,r/\lVert r\rVert`$.

*Uses:* [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1), [Proposition 5.5](../proofs/05_warm_adam.md#s5-5). *Used by:* Corollary 7.2, Proposition 7.3, Remark 7.5, [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/07_second_moment_channel.md#s7-1) · [Experiments](../experiments/07_second_moment_channel.md#c7-1) · proved

#### [Corollary 7.2](../proofs/07_second_moment_channel.md#s7-2) · Stationary amplitude law

For one coordinate, take tail gradients that do not depend on $`\theta`$, stationary with mean $`\bar g\ne0`$ and mean square $`\bar v>0`$; $`F`$ affine in $`\theta_i`$; $`\epsilon=0`$, no bias correction, no decay; a warm state $`m_t=\bar g`$, $`v_t=\bar v`$; and $`T`$ tail updates after the message. Let

- $`\alpha=-(\nabla F)_i\,\mathrm{sign}(\bar g)`$ and $`\lambda=g\,\mathrm{sign}(\bar g)/\sqrt{\bar v}`$ (the message amplitude in units of the running RMS);
- $`\rho=\lvert\bar g\rvert/\sqrt{\bar v}`$ (the signal-to-noise ratio of the coordinate);
- $`b_j=(1-\beta_1)\beta_1^{\,j}`$, $`c_j=(1-\beta_2)\beta_2^{\,j}`$, $`\omega_T=1-\beta_2^{\,T+1}`$.

**(a)** The mean-field response of replacing a zero-gradient message by $`g`$ (the step $`m/\sqrt v`$ replaced by $`\mathbb Em/\sqrt{\mathbb Ev}`$) is, for every $`\lambda`$,

```math
\Delta F(\lambda)=\alpha\eta\sum_{j=0}^{T}\Big[\frac{\rho+b_j(\lambda-\rho)}{\sqrt{1+c_j(\lambda^2-1)}}-\frac{\rho(1-b_j)}{\sqrt{1-c_j}}\Big].
```

- **(b)** With $`Q(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho\,\omega_T\lambda^2`$ and $`S(\lambda)=\lvert\lambda\rvert(1-\beta_1^{\,T+1})+\tfrac12\rho\,\omega_T\lambda^2`$: if $`\beta_2\ge\tfrac12`$, then $`\lvert\Delta F-\alpha\eta Q\rvert\le11(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\,\lvert\alpha\rvert\eta S`$, and the same holds for the first-order value. So $`Q`$ is concave with maximum at $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$. The sign change lies within relative distance $`4u`$ of $`2\lambda^\star`$ when $`u=11(1-\beta_2)\max\lbrace1,9\lambda^{\star2}\rbrace\le1/8`$; the threshold $`2\lambda^\star`$ is usable only when $`(1-\beta_2)(2\lambda^\star)^2\ll1`$.
- **(c)** As $`\lambda\to\infty`$, $`\Delta F\to\alpha\eta[\kappa_{\rm iso}(T)-\rho\Lambda(T)]`$ with $`\kappa_{\rm iso}(T)=\sum_jb_j/\sqrt{c_j}`$ and $`\Lambda(T)=\sum_j(1-b_j)/\sqrt{1-c_j}\approx T+1`$. For $`\alpha>0`$, all large same-direction messages are worse than a zero message if $`\rho>\kappa_{\rm iso}(T)/\Lambda(T)`$, and better if $`\rho<\kappa_{\rm iso}(T)/\Lambda(T)`$.
- **(d)** Under SGD the response is linear in $`\lambda`$.

*Derivation outline.*

- Under the mean-field substitution the message's deviation decays geometrically: $`\mathbb Em_{t+j}=\bar g+b_j(g-\bar g)`$ and $`\mathbb Ev_{t+j}=\bar v(1+c_j(g^2/\bar v-1))`$. Summing the $`T+1`$ steps $`\mathbb Em/\sqrt{\mathbb Ev}`$, with and without the message, gives (a).
- (b) expands $`(1+c_j(\lambda^2-1))^{-1/2}`$ with explicit mean-value bounds on the first- and second-order remainders. The cross terms $`\sum_jb_jc_j`$ are absorbed into the constant.
- (c) is the termwise limit: the first fraction tends to $`b_j/\sqrt{c_j}`$. A negative limit together with $`\Delta F'(0)>0`$ forces a positive zero.

*Illustration (analytic, from the formula).* Take $`\beta_1=0.9`$, $`\beta_2=0.99`$ and $`\rho=0.5`$.

- With $`T=1000`$ remaining updates, $`\lambda^\star=2`$. The exact formula peaks at $`\lambda\approx1.96`$ and changes sign at $`\lambda\approx3.98`$, next to $`2\lambda^\star=4`$. A message twice the running RMS is best; one four times larger than the RMS is worth less than no message.
- With $`T=20`$, $`2\lambda^\star\approx18.7`$ but the exact sign change is at $`\lambda\approx112`$, where the quadratic law no longer applies. The large-amplitude limit is negative, but only slightly ($`\rho\Lambda\approx10.1`$ against $`\kappa_{\rm iso}\approx9.2`$).

*Uses:* Theorem 7.1, [Proposition 6.11(c)](../proofs/06_learner_relative_information.md#s6-11) (the constant $`\kappa_{\rm iso}`$). *Used by:* Proposition 7.3, Remarks 7.4–7.5, [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/07_second_moment_channel.md#s7-2) · [Experiments](../experiments/07_second_moment_channel.md#c7-2) · proved (the mean-field substitution is part of the definition of $`\Delta F`$; its accuracy is a measured question)

#### [Proposition 7.3](../proofs/07_second_moment_channel.md#s7-3) · Inconsistency of two proxy classes

- **(a)** Any statistic that depends only on the state and the feedback law, not on the task, ranks some pair of laws wrongly, or fails to separate them, on some task: the score $`F`$ and the score $`1-F`$ demand opposite orders.
- **(b)** In the setting of Corollary 7.2 with $`\alpha>0`$ and $`0<\lambda_1<\lambda_2`$:
  - the parameter-only score $`\langle\chi^\theta,\Delta\theta(g)\rangle=\alpha\eta\,(\beta_1\rho+(1-\beta_1)\lambda)/\sqrt{\beta_2+(1-\beta_2)\lambda^2}`$ increases for $`\lambda<(1-\beta_1)\beta_2/(\beta_1\rho(1-\beta_2))`$, so it prefers $`\lambda_2`$;
  - any cosine score ties the two messages;
  - the exact value prefers $`\lambda_1`$ whenever $`\Delta F(\lambda_2)<\Delta F(\lambda_1)`$, which holds for $`\lambda_1+\lambda_2>2\lambda^\star`$ up to the error of Corollary 7.2(b).

*Derivation outline.*

- (a): reversing the task reverses every true order but no task-blind statistic. A natural pair is the label packets of [Proposition 3.1(b)](../proofs/03_learning_consequence.md#s3-1).
- (b): the parameter-only score is the message's own step, whose derivative in $`\lambda`$ has the sign of $`(1-\beta_1)\beta_2-\beta_1\rho(1-\beta_2)\lambda`$. The exact value follows the concave $`Q`$, which decreases past $`\lambda^\star`$.

*Uses:* Corollary 7.2, [Proposition 3.1](../proofs/03_learning_consequence.md#s3-1). *Used by:* Remarks 7.4–7.5, [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/07_second_moment_channel.md#s7-3) · [Experiments](../experiments/07_second_moment_channel.md#c7-4) · proved

#### [Remark 7.4](../proofs/07_second_moment_channel.md#s7-4) · Relation to prior work; the role of clipping

- Liu and Li (2026) analyze short-horizon propagation through the AdamW states, without an amplitude law or threshold.
- In-Run Data Shapley for Adam (Ding et al., 2026) and TracIn-style scores with Adam steps are parameter-only; LESS (Xia et al., 2024) is a cosine score. Both classes are covered by Proposition 7.3(b).
- Global-norm clipping bounds $`\lambda`$ and hence the second-moment loss. This is a testable mechanism, not a claimed result.

[Proof](../proofs/07_second_moment_channel.md#s7-4) · remark

#### [Remark 7.5](../proofs/07_second_moment_channel.md#s7-5) · Where first-order rankers fail

- Where the second-moment share of a pair's increment contrast is high, the message is large relative to $`v`$. That is the nonlinear, saturating regime of Corollary 7.2(a),(c), where the linearization overstates the $`v`$-channel harm.
- Any first-order score is unreliable there, including the full-state score with the exact continuation gradient.
- Ranking there should use finite-replacement measurement ([Theorem 8.8](../proofs/08_certificate.md#s8-8)) or the nonlinear formula.

[Proof](../proofs/07_second_moment_channel.md#s7-5) · [Experiments](../experiments/07_second_moment_channel.md#c7-5) · remark
