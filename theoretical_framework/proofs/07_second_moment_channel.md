# §7 Adam second-moment channel (Theorem B) — Proofs

[← Framework](../README.md) · [Experiments](../experiments/07_second_moment_channel.md) · [中文](07_second_moment_channel_ZH.md)

**Summary.**

- [Theorem 7.1](#s7-1) computes the gradient of the continuation value for Adam/AdamW by a reverse (costate) recursion. The first-order value of a message then splits exactly into a parameter channel, a momentum channel and a **second-moment channel**. The second-moment channel depends on the message only through $`g\odot g`$.
- [Corollary 7.2](#s7-2) solves the warm, stationary, linear-response case in closed form. For an arbitrary message amplitude $`\lambda`$ (the message gradient measured in units of the running RMS) the mean-field response is an explicit sum.
  - For small increments the value is a concave quadratic in $`\lambda`$, with an optimal amplitude $`\lambda^\star`$ and a sign change near $`2\lambda^\star`$.
  - For very large amplitudes the direct effect saturates. Whether a very large same-direction message ends up worse than a zero message is decided by an explicit criterion.
- [Proposition 7.3](#s7-3) derives two consequences for influence scores.
  - Task-blind statistics cannot rank feedback laws correctly for every task.
  - In the regime of Corollary 7.2, parameter-only first-order scores rank two same-direction messages in the wrong order, and cosine scores cannot separate them.
- [Remark 7.5](#s7-5) states where any first-order ranker is unreliable.
- "Theorem B" denotes [Theorem 7.1](#s7-1) together with [Corollary 7.2](#s7-2).
- Uses: the complete learner state and continuation value ([Definitions 1.3–1.4](../README.md#s1-3)); the first-order representation ([Proposition 5.1](05_optimizer_state.md#s5-1)); the Adam increment map ([Proposition 5.5(a)](05_warm_adam.md#s5-5)); the isolated-visit constant $`\kappa_{\rm iso}`$ ([Proposition 6.11(c)](06_learner_relative_information.md#s6-11)); [Proposition 3.1(b)](03_learning_consequence.md#s3-1).

## Setting

**State and update.** The complete learner state is $`z=(\theta,o)`$ with optimizer state $`o=(m,v,s)`$; $`s`$ is the absolute step counter. AdamW with hyperparameters $`\beta_1\in[0,1)`$, $`\beta_2\in(0,1)`$, $`\eta>0`$, $`\epsilon\ge0`$ and decoupled weight decay $`\lambda_{\rm wd}\ge0`$ maps a consumed gradient $`g`$ at step $`s`$ to

```math
m'=\beta_1m+(1-\beta_1)g,\qquad v'=\beta_2v+(1-\beta_2)\,g\odot g,\qquad \theta'=(1-\eta\lambda_{\rm wd})\,\theta-\eta\,q_{1,s}\,\frac{m'}{\sqrt{q_{2,s}v'}+\epsilon},
```

with bias-correction factors $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$ and $`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$, or $`q_{1,s}=q_{2,s}=1`$ without bias correction. Operations on vectors are coordinatewise. The increment is $`\Delta(z,g)=(\theta'-\theta,\;m'-m,\;v'-v)`$, the map of [Proposition 5.5(a)](05_warm_adam.md#s5-5); the counter advances by one and is never differentiated.

**Message and reference tail.**

- At round $`t`$ the learner is at $`z_t=(\theta_t,m_t,v_t,s_t)`$ and a message with consumed gradient $`g`$ moves it to $`z_t+\Delta_t(z_t,g)`$.
- The reference tail then runs the updates $`k=t+1,\dots,R-1`$. Update $`k`$ uses the consumed gradient $`g_k(\theta)`$ of its own data. This is the loss gradient $`\nabla\ell_k(\theta)`$, or its global-norm clipped version $`\mathrm{clip}_c(\nabla\ell_k(\theta))`$.
- The final score is $`F(\theta_R)`$. The continuation value $`V_{t+1}(z)`$ is the score reached when the tail starts from $`z`$ ([Definition 1.4](../README.md#s1-4)); with tail randomness $`\omega`$ (data order, sampling), $`V_{t+1}=\mathbb E_\omega V_{t+1,\omega}`$.

**First-order value of a message.** Following [Proposition 5.1](05_optimizer_state.md#s5-1), the first-order value of a message $`g`$ at $`z_t`$ is

```math
\mathrm{Val}(g)=\big\langle\nabla V_{t+1}(z_t),\,\Delta_t(z_t,g)\big\rangle ,
```

the first-order term of $`V_{t+1}(z_t+\Delta_t(z_t,g))-V_{t+1}(z_t)`$. For a message law $`P`$, $`\mathrm{Val}(P)=\mathbb E_P\mathrm{Val}(G)=\langle\nabla V_{t+1}(z_t),\mu^P_t(z_t)\rangle`$, the term that [Proposition 5.1](05_optimizer_state.md#s5-1) sums over rounds. Only differences $`\mathrm{Val}(g)-\mathrm{Val}(g')`$ at the same state enter a comparison.

## The state-channel decomposition

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

**(e) Random tails.** If the tail has finitely many random branches, each satisfying (D1)–(D2), or more generally if the branchwise costates are dominated so that differentiation and expectation commute, then $`\nabla V_{t+1}(z_t)=\mathbb E_\omega\chi_{t+1,\omega}`$, and (b) holds with $`\mathbb E_\omega\chi_{t+1,\omega}`$.

*Proof.* (a) One tail update is the map $`\Phi_k:(\theta,m,v)\mapsto(\theta',m',v')`$ with $`g=g_k(\theta)`$. Under (D1)–(D2) it is $`C^1`$ near the trajectory. Its Jacobian blocks, with $`D_1=D_1^{(k)}`$, $`D_2=D_2^{(k)}`$, $`J=J_k`$, are as follows.

- $`\partial m'/\partial m=\beta_1I`$, $`\partial m'/\partial\theta=(1-\beta_1)J`$, $`\partial m'/\partial v=0`$.
- $`\partial v'/\partial v=\beta_2I`$, $`\partial v'/\partial\theta=2(1-\beta_2)\,\mathrm{diag}(g)\,J`$, $`\partial v'/\partial m=0`$.
- $`\partial\theta'/\partial m'=-\eta D_1`$. The derivative of $`(\sqrt{q_2v'}+\epsilon)^{-1}`$ with respect to $`v'`$ is $`-q_2/\big(2\sqrt{q_2v'}(\sqrt{q_2v'}+\epsilon)^2\big)`$, so $`\partial\theta'/\partial v'=\eta D_2`$.
- Hence $`\partial\theta'/\partial m=-\eta\beta_1D_1`$, $`\partial\theta'/\partial v=\eta\beta_2D_2`$, and $`\partial\theta'/\partial\theta=(1-\eta\lambda_{\rm wd})I-\eta(1-\beta_1)D_1J+2\eta(1-\beta_2)D_2\,\mathrm{diag}(g)\,J`$.

$`V_{t+1}(z)=F(\pi_\theta\,\Phi_{R-1}\circ\cdots\circ\Phi_{t+1}(z))`$, where $`\pi_\theta`$ extracts the parameters. By the chain rule, $`\nabla V_{t+1}(z_t)=D\Phi_{t+1}^\top\cdots D\Phi_{R-1}^\top(\nabla F,0,0)`$. Accumulating from the right gives $`\chi_k=D\Phi_k^\top\chi_{k+1}`$ with $`\chi_R=(\nabla F,0,0)`$. Multiplying the transposed blocks (all $`D`$'s are diagonal) gives the stated recursion: the $`m`$-row collects $`-\eta\beta_1D_1\chi^\theta+\beta_1\chi^m`$, the $`v`$-row collects $`\eta\beta_2D_2\chi^\theta+\beta_2\chi^v`$, and the $`\theta`$-row collects the remaining terms. When clipping is active, $`g=c\,r/\lVert r\rVert`$, and the derivative of $`r\mapsto r/\lVert r\rVert`$ is $`(I-rr^\top/\lVert r\rVert^2)/\lVert r\rVert`$, which gives the stated $`J_k`$. The recursion uses $`J_k^\top`$, which need not be symmetric.

(b) is the definition of $`\mathrm{Val}`$ with $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$ and the increments $`\Delta m=(1-\beta_1)(g-m_t)`$, $`\Delta v=(1-\beta_2)(g\odot g-v_t)`$ of [Proposition 5.5(a)](05_warm_adam.md#s5-5).

(c) is read off from (b); for a law, take expectations in (b).

(d) With $`J_k=0`$ and $`\lambda_{\rm wd}=0`$, the $`\theta`$-row gives $`\chi^\theta_k=\chi^\theta_{k+1}=\nabla F`$. Then $`\chi^m_k=\beta_1\chi^m_{k+1}-\eta\beta_1D_1^{(k)}\nabla F`$ with $`\chi^m_R=0`$. Unrolling gives $`\chi^m_{t+1}=-\eta\sum_{k=t+1}^{R-1}\beta_1^{\,k-t}D_1^{(k)}\nabla F`$; the $`v`$-row is identical with $`+\eta\beta_2D_2^{(k)}`$.

(e) follows from (a) applied branchwise and the interchange of $`\nabla`$ and $`\mathbb E_\omega`$. $`\square`$

- The step counter is held fixed during differentiation; decoupled decay multiplies $`\theta`$ by $`1-\eta\lambda_{\rm wd}`$ with the unmodified learning rate.
- At a clipping threshold the map is not differentiable and no derivative is claimed.

## The stationary amplitude law

The next result evaluates [Theorem 7.1](#s7-1) in the simplest regime where the second-moment channel can be computed in closed form: one coordinate, a warm stationary state, gradients of the tail that do not depend on $`\theta`$.

<a id="s7-2"></a>

**Corollary 7.2 (Stationary amplitude law).** Assume, for a coordinate $`i`$:

- the tail gradients $`g_{k,i}`$ do not depend on $`\theta`$ (linear-response regime, $`J_k=0`$), are stationary with mean $`\bar g_i\ne0`$ and mean square $`\bar v_i>0`$, and $`F`$ is affine in $`\theta_i`$;
- $`\epsilon=0`$, no bias correction, $`\lambda_{\rm wd}=0`$;
- the message arrives at a warm state $`m_{t,i}=\bar g_i`$, $`v_{t,i}=\bar v_i`$; there are $`T=R-1-t`$ tail updates after the message update.

Put

- $`\alpha_i=-(\nabla F)_i\,\mathrm{sign}(\bar g_i)`$, so $`\alpha_i>0`$ when moving against $`\bar g_i`$ improves the outcome;
- $`\lambda=g_i\,\mathrm{sign}(\bar g_i)/\sqrt{\bar v_i}`$, the same-direction amplitude of the message relative to the running RMS;
- $`\rho_i=\lvert\bar g_i\rvert/\sqrt{\bar v_i}\in(0,1]`$, the signal-to-noise ratio of the coordinate;
- $`b_j=(1-\beta_1)\beta_1^{\,j}`$, $`c_j=(1-\beta_2)\beta_2^{\,j}`$ for $`j=0,\dots,T`$ ($`j=0`$ is the message update), and $`\omega_T=1-\beta_2^{\,T+1}=\sum_{j\le T}c_j`$.

The **mean-field response** $`\Delta F_i(\lambda)`$ is the change of $`F`$ when a message of amplitude $`\lambda`$ replaces a zero-gradient message, computed with the step $`m_k/\sqrt{v_k}`$ replaced by $`\mathbb Em_k/\sqrt{\mathbb Ev_k}`$, the expectations being over the tail gradients.

**(a) Exact mean-field response, every $`\lambda`$.**

```math
\Delta F_i(\lambda)=\alpha_i\eta\sum_{j=0}^{T}\Big[\frac{\rho_i+b_j(\lambda-\rho_i)}{\sqrt{1+c_j(\lambda^2-1)}}-\frac{\rho_i(1-b_j)}{\sqrt{1-c_j}}\Big].
```

**(b) Small increments: the quadratic law.** Let $`Q_i(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho_i\omega_T\lambda^2`$ and $`S_i(\lambda)=\lvert\lambda\rvert(1-\beta_1^{\,T+1})+\tfrac12\rho_i\omega_T\lambda^2`$. If $`\beta_2\ge\tfrac12`$, then for every $`\lambda`$

```math
\big\lvert\Delta F_i(\lambda)-\alpha_i\eta\,Q_i(\lambda)\big\rvert\le11\,(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\;\lvert\alpha_i\rvert\eta\,S_i(\lambda).
```

The same bound holds for the mean-field first-order value difference $`\mathrm{Val}_i(\lambda)-\mathrm{Val}_i(0)`$ of [Theorem 7.1](#s7-1). Consequently:

- $`Q_i`$ is concave and is maximized at $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho_i\omega_T)`$.
- Let $`u=11(1-\beta_2)\max\lbrace1,9\lambda^{\star2}\rbrace`$ and $`\alpha_i>0`$. If $`u\le1/8`$, then $`\Delta F_i>0`$ on $`[\lambda^\star,\,2\lambda^\star(1-4u)]`$ and $`\Delta F_i<0`$ on $`[2\lambda^\star(1+4u),\,3\lambda^\star]`$. So the sign change lies within relative distance $`4u`$ of $`2\lambda^\star`$, and a same-direction message just above it is worse than a zero-gradient message.
- The threshold $`2\lambda^\star`$ is therefore usable only when $`(1-\beta_2)(2\lambda^\star)^2\ll1`$. Otherwise the sign change, if any, must be computed from (a) and (c).
- Short remaining horizon, $`T(1-\beta_2)\ll1`$. Here $`\omega_T\approx(T+1)(1-\beta_2)`$, so $`\lambda^\star`$ is large and $`(1-\beta_2)(2\lambda^\star)^2\approx4(1-\beta_1^{\,T+1})^2/\big(\rho_i^2(T+1)^2(1-\beta_2)\big)`$. Unless this is small, $`2\lambda^\star`$ lies outside the range of (b).
- Long remaining horizon, $`T(1-\beta_2)\gg1`$. Here $`\omega_T\to1`$, $`\lambda^\star\to(1-\beta_1^{\,T+1})/\rho_i\approx1/\rho_i`$, and $`(1-\beta_2)(2\lambda^\star)^2\approx4(1-\beta_2)/\rho_i^2`$, which is small unless the coordinate is very noisy.

**(c) Large amplitudes.** As $`\lambda\to\infty`$,

```math
\Delta F_i(\lambda)\to\alpha_i\eta\big[\kappa_{\rm iso}(T)-\rho_i\,\Lambda(T)\big],\qquad \kappa_{\rm iso}(T)=\sum_{j=0}^{T}\frac{b_j}{\sqrt{c_j}},\qquad \Lambda(T)=\sum_{j=0}^{T}\frac{1-b_j}{\sqrt{1-c_j}} .
```

- Let $`\alpha_i>0`$. If $`\rho_i\Lambda(T)>\kappa_{\rm iso}(T)`$, every sufficiently large same-direction message is worse than a zero-gradient message, and $`\Delta F_i`$ has a positive zero. If $`\rho_i\Lambda(T)<\kappa_{\rm iso}(T)`$, every sufficiently large same-direction message is better. The exact critical signal-to-noise ratio is $`\kappa_{\rm iso}(T)/\Lambda(T)`$.
- Since $`\Lambda(T)=(T+1)\big(1+O(\tfrac1{T+1}+(1-\beta_2))\big)`$, the criterion is approximately $`\rho_i(T+1)>\kappa_{\rm iso}(T)`$.
- $`\kappa_{\rm iso}(T)`$ is the finite-horizon version of the isolated-visit constant $`\kappa_{\rm iso}=(1-\beta_1)/\big(\sqrt{1-\beta_2}\,(1-\beta_1/\sqrt{\beta_2})\big)`$ of [Proposition 6.11(c)](06_learner_relative_information.md#s6-11), to which it converges as $`T\to\infty`$ when $`\beta_1<\sqrt{\beta_2}`$.
- Reading. A very large message sets its own denominator, so its direct effect saturates near $`\eta\kappa_{\rm iso}(T)`$. Meanwhile it depresses every later step: at lag $`j`$ the comparator's step is about $`\rho_i`$ and the message's step is about $`\rho_i/\sqrt{1+c_j\lambda^2}`$. This lasts while $`c_j\lambda^2\gtrsim1`$, i.e. for about $`\log((1-\beta_2)\lambda^2)/(1-\beta_2)`$ steps, capped at $`T+1`$.

**(d) Comparison with SGD.** For SGD in the same regime, $`\Delta F_i(\lambda)=\alpha_i\eta\sqrt{\bar v_i}\,\lambda`$ is linear: a larger same-direction message is always better.

*Proof.* (a) Under the mean-field substitution the tail gradients are replaced by their moments, so the message's deviation decays geometrically. For $`j\ge0`$,

```math
\mathbb Em_{t+j,i}=\bar g_i+b_j(g_i-\bar g_i),\qquad \mathbb Ev_{t+j,i}=\bar v_i\big(1+c_j(g_i^2/\bar v_i-1)\big),
```

where index $`t+j`$ denotes the moments used by the $`j`$-th update after the message ($`j=0`$ is the message update). The $`j=0`$ case is the message update from $`m_t=\bar g`$, $`v_t=\bar v`$. Each later update multiplies the deviation of $`\mathbb Em`$ by $`\beta_1`$ and that of $`\mathbb Ev`$ by $`\beta_2`$, because the tail gradients have mean $`\bar g_i`$ and mean square $`\bar v_i`$. With $`F`$ affine in $`\theta_i`$, the change of outcome is $`-(\nabla F)_i\,\eta\sum_{j=0}^{T}\big[\mathbb Em_{t+j}(g)/\sqrt{\mathbb Ev_{t+j}(g)}-\mathbb Em_{t+j}(0)/\sqrt{\mathbb Ev_{t+j}(0)}\big]`$.

Divide numerator and denominator by $`\sqrt{\bar v_i}`$ and use $`\bar g_i/\sqrt{\bar v_i}=\mathrm{sign}(\bar g_i)\rho_i`$, $`g_i/\sqrt{\bar v_i}=\mathrm{sign}(\bar g_i)\lambda`$ and $`-(\nabla F)_i\,\mathrm{sign}(\bar g_i)=\alpha_i`$. All denominators are positive, since $`c_j\le1-\beta_2<1`$.

(b) Write $`\gamma=1-\beta_2\le\tfrac12`$ and $`f_j(x)=(1+c_j(x-1))^{-1/2}`$. The bracket in (a) equals $`\rho_i(1-b_j)\big(f_j(\lambda^2)-f_j(0)\big)+b_j\lambda f_j(\lambda^2)`$, so

```math
\frac{\Delta F_i}{\alpha_i\eta}-Q_i=\underbrace{\lambda\sum_jb_j\big(f_j(\lambda^2)-1\big)}_{E_1}+\underbrace{\rho_i\sum_j(1-b_j)\big(f_j(\lambda^2)-f_j(0)+\tfrac12c_j\lambda^2\big)}_{E_2}+\underbrace{\tfrac12\rho_i\lambda^2\sum_jb_jc_j}_{E_3}.
```

Two elementary bounds hold for $`w\ge-\gamma\ge-\tfrac12`$:

- $`\lvert(1+w)^{-1/2}-1\rvert\le\tfrac12(1-\gamma)^{-3/2}\lvert w\rvert\le1.42\lvert w\rvert`$;
- $`\lvert1-(1+w)^{-3/2}\rvert\le\tfrac32(1-\gamma)^{-5/2}\lvert w\rvert\le8.49\lvert w\rvert`$.

Both follow from the mean value theorem, since $`1+w\ge1-\gamma`$. The arguments are $`w=c_j(x-1)`$ with $`x\in[0,\lambda^2]`$, so $`\lvert w\rvert\le c_j\max\lbrace1,\lambda^2\rbrace`$. Also $`\sum_jb_jc_j\le\gamma\sum_jb_j=\gamma(1-\beta_1^{\,T+1})`$ and $`\sum_jc_j^2\le\gamma\omega_T`$.

- $`\lvert E_1\rvert\le1.42\,\gamma\max\lbrace1,\lambda^2\rbrace\,\lvert\lambda\rvert(1-\beta_1^{\,T+1})`$.
- $`f_j(x)-f_j(0)+\tfrac12c_jx=\int_0^x\tfrac12c_j\big(1-(1+c_j(y-1))^{-3/2}\big)\mathrm dy`$, so $`\lvert E_2\rvert\le\rho_i\sum_j\tfrac12c_j\lambda^2\cdot8.49\,c_j\max\lbrace1,\lambda^2\rbrace\le8.49\,\gamma\max\lbrace1,\lambda^2\rbrace\cdot\tfrac12\rho_i\omega_T\lambda^2`$.
- $`\lvert E_3\rvert\le\tfrac12\rho_i\lambda^2\gamma(1-\beta_1^{\,T+1})\le\tfrac12\gamma\max\lbrace1,\lambda^2\rbrace\lvert\lambda\rvert(1-\beta_1^{\,T+1})`$, using $`\rho_i\le1`$ and $`\lvert\lambda\rvert\le\max\lbrace1,\lambda^2\rbrace`$.

Summing, $`\lvert\Delta F_i/(\alpha_i\eta)-Q_i\rvert\le(1.42+8.49+0.5)\,\gamma\max\lbrace1,\lambda^2\rbrace S_i(\lambda)`$.

For the first-order value, the mean-field costates of [Theorem 7.1(d)](#s7-1) at the pre-message state are $`\chi^m=-\eta(\nabla F)_i\sum_{j=1}^T\beta_1^{\,j}/\sqrt{\bar v_i}`$ and $`\chi^v=\tfrac12\eta(\nabla F)_i\bar g_i\sum_{j=1}^T\beta_2^{\,j}/\bar v_i^{3/2}`$ (here $`D_1=\bar v_i^{-1/2}`$, $`D_2=\bar g_i/(2\bar v_i^{3/2})`$). In normalized units, $`\mathrm{Val}_i(\lambda)-\mathrm{Val}_i(0)`$ is the exact $`j=0`$ bracket of (a) plus $`\sum_{j=1}^T\big(b_j\lambda-\tfrac12\rho_ic_j\lambda^2\big)`$. Its difference from $`Q_i`$ is the $`j=0`$ part of $`E_1+E_2+E_3`$, which obeys the same bound.

The consequences:

- $`Q_i''=-\rho_i\omega_T<0`$, and $`Q_i'(\lambda^\star)=0`$.
- For the sign change, let $`\alpha_i>0`$ and $`\lambda=2\lambda^\star(1+\delta)`$ with $`\lvert\delta\rvert\le\tfrac12`$. Then $`\lambda^\star\le\lambda\le3\lambda^\star`$, so the error is at most $`u\,S_i(\lambda)`$. Also $`Q_i(\lambda)=-(1-\beta_1^{\,T+1})\lambda\delta`$ and $`S_i(\lambda)=(1-\beta_1^{\,T+1})\lambda(2+\delta)`$.
  - For $`\delta\in[4u,\tfrac12]`$: $`\Delta F_i/(\alpha_i\eta)\le(1-\beta_1^{\,T+1})\lambda\,[-\delta+u(2+\delta)]<0`$, because $`\delta(1-u)\ge4u(1-u)>2u`$ when $`u<\tfrac12`$.
  - For $`\delta\in[-\tfrac12,-4u]`$: $`\Delta F_i/(\alpha_i\eta)\ge(1-\beta_1^{\,T+1})\lambda\,[\lvert\delta\rvert-u(2-\lvert\delta\rvert)]>0`$, because $`\lvert\delta\rvert(1+u)>2u`$.
  - $`u\le1/8`$ makes $`4u\le\tfrac12`$, so both intervals are nonempty.
- The horizon statements follow from $`\omega_T=1-(1-\gamma)^{T+1}`$.

(c) As $`\lambda\to\infty`$, the $`j`$-th first fraction behaves like $`b_j\lambda/(\sqrt{c_j}\lambda)\to b_j/\sqrt{c_j}`$ (it tends to $`0=b_j/\sqrt{c_j}`$ when $`b_j=0`$). The second fraction does not depend on $`\lambda`$. This gives the limit.

- If the limit is negative and $`\alpha_i>0`$, then $`\Delta F_i(\lambda)<0`$ for all large $`\lambda`$. Also $`\Delta F_i(0)=0`$ and $`\Delta F_i'(0)=\alpha_i\eta\sum_jb_j/\sqrt{1-c_j}>0`$ (differentiate (a); $`b_0=1-\beta_1>0`$), so $`\Delta F_i>0`$ for small $`\lambda>0`$ and there is a positive zero. The positive case is symmetric.
- $`\Lambda(T)=\sum_j(1-b_j)(1+O(c_j))=(T+1)-(1-\beta_1^{\,T+1})+O(\omega_T)`$.
- $`\kappa_{\rm iso}(T)=\frac{1-\beta_1}{\sqrt{1-\beta_2}}\sum_{j\le T}(\beta_1/\sqrt{\beta_2})^j`$ converges to $`\kappa_{\rm iso}`$ when $`\beta_1<\sqrt{\beta_2}`$.
- The reading follows from (a): for $`\lambda`$ large and lag $`j`$ with $`b_j\lambda`$ negligible, the message's term is $`\rho_i(1-b_j)/\sqrt{1+c_j(\lambda^2-1)}`$ against the comparator's $`\rho_i(1-b_j)/\sqrt{1-c_j}`$.

(d) Plain SGD moves $`\theta_i`$ by $`-\eta g_i`$ and does not change later steps, so $`\Delta F_i=-(\nabla F)_i\eta g_i=\alpha_i\eta\sqrt{\bar v_i}\lambda`$. $`\square`$

- The mean-field substitution ignores the fluctuation of $`m_k`$ and $`v_k`$ around their means. It is the only approximation in (a); its accuracy is measured in [Check 7.2](../experiments/07_second_moment_channel.md#c7-2).
- The constant 11 in (b) is a valid but conservative constant; the measured agreement of $`2\lambda^\star`$ with the actual sign change is far better ([Check 7.3](../experiments/07_second_moment_channel.md#c7-3)).
- The positive-zero statement in (c) is a one-way implication. That no zero exists when the limit is positive is not claimed. In the checked settings the two always agreed ([Check 7.2](../experiments/07_second_moment_channel.md#c7-2)).

## Consequences for influence scores

<a id="s7-3"></a>

**Proposition 7.3 (Inconsistency of two proxy classes).**

- **(a) Task-blind statistics.** Let $`\mathcal T`$ be any statistic that depends only on the current state and the feedback law, not on the deployment task, and that ranks laws by its value. Then there are two tasks and two laws $`P,Q`$ at the same learner and state such that the true order of $`P,Q`$ is opposite on the two tasks while $`\mathcal T(P),\mathcal T(Q)`$ are the same on both. Hence $`\mathcal T`$ ranks wrongly, or fails to separate a nonzero difference, on at least one task.
- **(b) Parameter-only and cosine scores.** In the setting of [Corollary 7.2](#s7-2) with $`\alpha_i>0`$, take two same-direction messages with amplitudes $`0<\lambda_1<\lambda_2`$.
  - The parameter-only score $`S_{\rm PO}(g)=\langle\chi^\theta,\Delta\theta(g)\rangle`$ satisfies $`S_{\rm PO}=\alpha_i\eta\,h(\lambda)`$ with $`h(\lambda)=(\beta_1\rho_i+(1-\beta_1)\lambda)/\sqrt{\beta_2+(1-\beta_2)\lambda^2}`$. The function $`h`$ is strictly increasing for $`\lambda<\lambda_{\rm PO}:=(1-\beta_1)\beta_2/(\beta_1\rho_i(1-\beta_2))`$. This class contains TracIn-style scores with Adam steps and fixed-state Adam data attribution. If $`\lambda_2<\lambda_{\rm PO}`$, it ranks the larger message first.
  - A cosine score is a function of the direction of the message gradient only, or, in one coordinate, of the direction of the parameter step. This class contains LESS-type scores. Every such score gives the two messages the same value.
  - The exact value prefers the smaller message whenever $`\Delta F_i(\lambda_2)<\Delta F_i(\lambda_1)`$. By [Corollary 7.2(b)](#s7-2) this holds when $`\lambda_1+\lambda_2>2\lambda^\star`$ and $`Q_i(\lambda_1)-Q_i(\lambda_2)>11(1-\beta_2)\max\lbrace1,\lambda_2^2\rbrace\big(S_i(\lambda_1)+S_i(\lambda_2)\big)`$. In general it can be read from [Corollary 7.2(a)](#s7-2).
  - In this region the parameter-only score ranks the pair in the wrong order and the cosine score cannot separate it.

*Proof.* (a) Take any task with score $`F`$ and laws with $`L_F(P)\ne L_F(Q)`$, and the task with score $`F'=1-F\in[0,1]`$. Then $`L_{F'}(P)-L_{F'}(Q)=-(L_F(P)-L_F(Q))`$, while $`\mathcal T`$ is unchanged. A natural pair of tasks with the same property is given by [Proposition 3.1(b)](03_learning_consequence.md#s3-1). Two label packets with equal marginals and equal mean gradient have learning values whose difference changes sign between $`J(p)=p^2`$ and $`J(p)=1-p^2`$.

(b) Here $`\chi^\theta=\nabla F`$, and the message's own step is

```math
\Delta\theta_i(g)=-\eta\,\frac{\beta_1\bar g_i+(1-\beta_1)g_i}{\sqrt{\beta_2\bar v_i+(1-\beta_2)g_i^2}}=-\eta\,\mathrm{sign}(\bar g_i)\,h(\lambda),
```

so $`S_{\rm PO}=(\nabla F)_i\Delta\theta_i=\alpha_i\eta\,h(\lambda)`$. Differentiating,

```math
h'(\lambda)=\frac{(1-\beta_1)\beta_2-\beta_1\rho_i(1-\beta_2)\lambda}{\big(\beta_2+(1-\beta_2)\lambda^2\big)^{3/2}},
```

which is positive for $`\lambda<\lambda_{\rm PO}`$.

- Positive multiples of one gradient have the same direction. In one coordinate, both parameter steps point along $`-\mathrm{sign}(\bar g_i)`$. So any cosine score ties the two messages.
- For the exact value, $`Q_i(\lambda_2)-Q_i(\lambda_1)=(\lambda_2-\lambda_1)\big[(1-\beta_1^{\,T+1})-\tfrac12\rho_i\omega_T(\lambda_1+\lambda_2)\big]`$, which is negative iff $`\lambda_1+\lambda_2>2\lambda^\star`$. The stated margin absorbs the error bound of [Corollary 7.2(b)](#s7-2) at both amplitudes. $`\square`$

- (b) is a statement about small increments, through the quadratic law of [Corollary 7.2(b)](#s7-2); for large increments the response must be read from [Corollary 7.2(a)](#s7-2).
- The interval $`2\lambda^\star<\lambda_1+\lambda_2`$ with $`\lambda_2<\lambda_{\rm PO}`$ is nonempty in typical settings. For a long horizon $`2\lambda^\star\approx2/\rho_i`$, while $`\lambda_{\rm PO}\approx(1-\beta_1)/(\beta_1\rho_i(1-\beta_2))`$, about $`11/\rho_i`$ at $`\beta_1=0.9`$, $`\beta_2=0.99`$.

<a id="s7-4"></a>

**Remark 7.4 (Relation to prior work; the role of clipping).**

- Liu and Li (2026) write AdamW as an input–state–output system and analyze how a minibatch perturbation propagates through the momentum and second-moment states over short horizons (8–32 steps). They give no amplitude law, optimal amplitude, threshold, or implication for data selection; [Corollary 7.2](#s7-2) supplies these for the long-horizon stationary case.
- In-Run Data Shapley for Adam (Ding et al., 2026) uses a fixed-state approximation: only the current preconditioned update is scored. It therefore belongs to the parameter-only class of [Proposition 7.3(b)](#s7-3). So do TracIn-style scores (Pruthi et al., 2020) evaluated with Adam steps. LESS (Xia et al., 2024) ranks by the cosine between Adam-preconditioned training gradients and validation gradients, the cosine class of [Proposition 7.3(b)](#s7-3).
- Global-norm clipping bounds the amplitude $`\lambda`$ of any single message. It therefore bounds the loss that the second-moment channel can cause ([Corollary 7.2(a),(c)](#s7-2)). This is a testable mechanism for why clipping helps; it is not claimed as a result here.

<a id="s7-5"></a>

**Remark 7.5 (Where first-order rankers fail).**

- Call the second-moment share of a candidate pair the absolute $`v`$-channel contraction of the pair's increment contrast, divided by the sum of the absolute $`\theta`$-, $`m`$- and $`v`$-channel contractions.
- A high share typically means that the increment is large relative to $`v`$. That is the nonlinear, saturating regime of [Corollary 7.2(a),(c)](#s7-2), where the first-order value keeps following the quadratic law while the actual response does not. There the linearization overstates the harm done through the $`v`$-channel.
- The decomposition of [Theorem 7.1](#s7-1) is exact, but its first-order use is least reliable exactly where the $`v`$-channel matters.
- Where the second-moment share is high, a ranking should use a finite-replacement measurement ([Theorem 8.8](08_certificate.md#s8-8)) or a nonlinear formula such as [Corollary 7.2(a)](#s7-2), not any first-order score. Neither a parameter-only score nor a full-state score with the exact continuation gradient is sufficient.
- [Proposition 7.3(b)](#s7-3) is a statement inside the small-increment regime of [Corollary 7.2(b)](#s7-2).
- The controlled-panel evidence for this remark is in [Check 7.5](../experiments/07_second_moment_channel.md#c7-5).

## Related work

- Ding, M., Zhang, Z., Wang, D., Hu, L. (2026). In-Run Data Shapley for Adam optimizer. arXiv:2602.00329.
- Kingma, D. P., Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.
- Liu, K., Li, S. (2026). Finite-horizon input–output dynamics of minibatch perturbations in AdamW. arXiv:2608.19762.
- Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization. ICLR.
- Pruthi, G., Liu, F., Kale, S., Sundararajan, M. (2020). Estimating training data influence by tracing gradient descent. NeurIPS.
- Wang, J. T., Mittal, P., Song, D., Jia, R. (2025). Data Shapley in one training run. ICLR.
- Xia, M., Malladi, S., Gururangan, S., Arora, S., Chen, D. (2024). LESS: Selecting influential data for targeted instruction tuning. ICML.
