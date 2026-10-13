# Theorem 5.10 · The first-order statistic by learner

*Proof*

[← Framework](../README.md) · [Statement](../statements/05-10_first_order_statistic_by_learner.md) · [Proof](../proofs/05-10_first_order_statistic_by_learner.md) · [Experiments](../experiments/05-10_first_order_statistic_by_learner.md) · [中文](05-10_first_order_statistic_by_learner_ZH.md)

**What is proved here.** Each of (a)–(g) is proved by computing the increment of the update rule, cancelling the terms that depend only on the state, and taking the difference of expectations.

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

**Adam and AdamW.** Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) is bias-corrected, with $`\epsilon`$ outside the square root, optional decoupled weight decay (AdamW; [Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)) and no AMSGrad. An update with counter $`t\ge1`$ acts on $`z=(\theta,m,v,t-1)`$ and a consumed gradient $`G\in\mathbb R^p`$. Coordinatewise,

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- The hyperparameters satisfy $`\beta_1\in[0,1)`$, $`\beta_2\in(0,1)`$, $`\eta>0`$, $`\lambda_{\rm wd}\ge0`$, $`\epsilon\ge0`$. Arithmetic is exact real arithmetic.
- The next complete state is $`z^+=(\theta^+,m^+,v^+,t)`$.
- A *cold* state has $`m=v=0`$ and $`t=1`$. A *warm* state has $`v_i>0`$ for every coordinate $`i`$.

*Warm-state notation ([Definition 5.4](../statements/05-04_warmth_frozen_denominator_surrogate.md#s5-4)).*

Consider an Adam update with counter $`t\ge1`$ at $`z=(\theta,m,v,t-1)`$, and assume $`v_i>0`$, or $`\epsilon>0`$ whenever some $`v_i=0`$.

- *Retained denominator, warmth, relative step.* For a consumed gradient $`G`$,

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^{\,t}}},\qquad\kappa_i(G)=\frac{(1-\beta_2)G_i^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(G)=\infty\ \ (v_i=0),\qquad u_i(G)=\frac{\lvert n_i(G)\rvert}{a_i+\epsilon}.
```

- When $`v_i>0`$, $`d_i=a_i\sqrt{1+\kappa_i}`$ and $`v_i^+=\beta_2v_i\,(1+\kappa_i)`$. The warmth $`\kappa_i`$ is the relative perturbation that the new gradient makes to the retained second moment.
- *Frozen-denominator surrogate.* With the fixed diagonal preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$,

```math
\tilde\theta^+(G)=(1-\eta\lambda_{\rm wd})\theta-\eta\,\Phi\,n(G),\qquad\tilde\Delta(z,G)=\big(\tilde\theta^+(G)-\theta,\;m^+(G)-m,\;v^+(G)-v\big),\qquad\tilde z^+(G)=\big(\tilde\theta^+(G),\,m^+(G),\,\beta_2v,\,t\big).
```

- $`\tilde\theta^+`$ is one momentum-SGD step with the fixed preconditioner $`\Phi`$. $`\tilde\Delta`$ is the *frozen-denominator increment*: its $`(\theta,m)`$ blocks are affine in $`G`$ and its $`v`$ block is affine in $`G^{\odot2}`$. $`\tilde z^+`$ additionally freezes the second moment at $`\beta_2v`$; it is the comparison state of Proposition 5.6. For $`g\in\mathbb R^p`$, $`\tilde z^+(g)`$ is the same map evaluated at consumed gradient $`g`$.

**Optimizers.** All operations are coordinatewise. Terms of $`\Delta`$ that do not depend on the message cancel in $`\mu^P-\mu^Q`$: weight decay, momentum decay and bias-correction constants applied to the old state.

- *SGD:* $`\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta G`$.
- *Heavy-ball:* $`m^+=\beta m+G`$, $`\theta^+=\theta-\eta m^+`$. *Nesterov:* $`m^+=\beta m+G`$, $`\theta^+=\theta-\eta(G+\beta m^+)`$.
- *Adam and AdamW:* as in the Setting, with the warm-Adam notation of Definition 5.4 and $`c_1=(1-\beta_1)/(1-\beta_1^{\,t})`$.
- *Global-norm clipping:* $`\mathrm{clip}_c(G)`$, applied before the base optimizer.
- *Lion form of sign optimizers:* $`u=\mathrm{sign}(\beta_1m+(1-\beta_1)G)`$, $`\theta^+=\theta-\eta u`$, $`m^+=\beta_2m+(1-\beta_2)G`$.
- *LoRA:* an adapted weight $`W=W_0+\alpha BA`$ with $`A\in\mathbb R^{r\times k}`$, $`B\in\mathbb R^{d\times r}`$ and full-weight gradient $`\Gamma=\nabla_W\ell`$. The trainable parameters are $`(A,B)`$, with the standard initialization $`B_0=0`$ and $`A_0`$ random.

For two laws $`P,Q`$ at a state write $`\delta_1=\mathbb E_PG-\mathbb E_QG`$ and $`\delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}`$. By Proposition 5.1(c), the projection of $`\mu^P-\mu^Q`$ onto the span $`U`$ of the continuation gradients is what two feedback laws must agree on to first order.

**Statement ([Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)).**

**(a) SGD.** $`\mu^P-\mu^Q=-\eta\,\delta_1`$. The first-order statistic is the mean consumed gradient; the second-order statistic is $`\mathbb E\,GG^\top`$.

**(b) Heavy-ball and Nesterov momentum.** The increments are affine in $`G`$: heavy-ball has $`\Delta^\theta=-\eta(\beta m+G)`$ and $`\Delta^m=(\beta-1)m+G`$; Nesterov has $`\Delta^\theta=-\eta\big((1+\beta)G+\beta^2m\big)`$ and $`\Delta^m=(\beta-1)m+G`$. Hence the first-order term is as below. The statistic is the mean consumed gradient; momentum changes its weight, not the statistic.

```math
\langle\nabla V,\mu^P-\mu^Q\rangle=\big\langle-\eta\gamma\nabla_\theta V+\nabla_mV,\ \delta_1\big\rangle,\qquad\gamma=1\ \text{(heavy-ball)},\quad\gamma=1+\beta\ \text{(Nesterov)}.
```

**(c) Adam and AdamW at a warm state.** The exact first-order statistic is the mean Adam increment (Proposition 5.5(a)). In the blocks $`(\theta,m,v)`$, with $`e(G)=\theta^+-\tilde\theta^+`$ and $`\lvert e_i\rvert\le\eta u_i\min\lbrace \kappa_i/2,1\rbrace`$ (Proposition 5.5(b)),

```math
\mu^P-\mu^Q=\Big(-\eta c_1\Phi\,\delta_1+\big(\mathbb E_Pe-\mathbb E_Qe\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).
```

- The pair $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ determines $`\mu^P-\mu^Q`$ exactly under the frozen-denominator approximation, and in general up to the frozen-denominator error, at most $`\eta\,(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace \kappa_i/2,1\rbrace]`$ per coordinate. Relative to the step $`\eta u_i`$ the error is at most $`\kappa_i/2`$, and by Proposition 5.8 the warmth of a typical gradient does not vanish along training under a fixed $`\beta_2`$; it levels off near $`(1-\beta_2)/\beta_2`$.
- The coordinatewise second moment enters at first order through $`v`$, with weight $`(1-\beta_2)\nabla_vV`$. If $`\delta_1=0`$, the first-order term equals $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle+\langle\nabla_\theta V,\mathbb E_Pe-\mathbb E_Qe\rangle`$, where the second summand is at most $`\eta\sum_i\lvert\partial_{\theta_i}V\rvert(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace\kappa_i/2,1\rbrace]`$ in absolute value. Equal-mean laws can therefore differ at first order through the second-moment channel, by an amount that is not small in the warmth. [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1) computes $`\nabla_vV`$ by an adjoint recursion.

**(d) Adam at the cold state.** With $`h_\epsilon`$ as in Proposition 5.3,

```math
\mu^P-\mu^Q=\Big(-\eta\big(\mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).
```

The parameter block sees only $`\mathbb E\,h_\epsilon(G)`$, which as $`\epsilon\to0`$ is the sign law. The moment blocks retain $`\mathbb E\,G`$ and $`\mathbb E\,G^{\odot2}`$. If the run ends after this update ($`R=1`$), the continuation is $`F_E(\theta)`$, and the first-order term is $`-\eta\,\langle\nabla_\theta V,\ \mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\rangle`$.

**(e) Gradient clipping.** With clipped SGD, $`\mu^P-\mu^Q=-\eta\big(\mathbb E_P\mathrm{clip}_c(G)-\mathbb E_Q\mathrm{clip}_c(G)\big)`$ exactly, so the statistic is the clipped mean.

- *Witness.* Take $`c=1`$, $`P(G=4e_1)=\tfrac14`$, $`P(G=-\tfrac43e_1)=\tfrac34`$ and $`Q=\delta_0`$. Then $`\delta_1=0`$ but $`\mathbb E_P\mathrm{clip}_1G=-\tfrac12e_1`$, and the first-order term is $`\tfrac\eta2\,\partial_{\theta_1}V`$.
- *Mean direction.* By Proposition 5.9, the clipped mean equals $`c\,\mathbb E[G/\lVert G\rVert]`$ up to $`c(1-p_{\rm act})`$ in norm under each law. Replacing the clipped means by the scaled mean directions therefore changes $`\mu^P_\theta-\mu^Q_\theta`$ by at most $`\eta c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$ in norm.
- *Always-active clipping.* If $`\lVert G\rVert>c`$ almost surely under both laws, the statistic is exactly $`c\,\mathbb E[G/\lVert G\rVert]`$. If in addition the two laws induce the same law of $`G/\lVert G\rVert`$, the increment laws coincide and the laws are equivalent at every order.
- *Clipped base optimizers.* For any base optimizer in (a)–(d) and (f), the statistics are those of the clipped gradient $`\mathrm{clip}_c(G)`$.

**(f) Sign optimizers with momentum (Lion form).** For $`m_i\ne0`$, with a zero counted as half a flip,

```math
\mathbb E\,\Delta^\theta_i=-\eta\,\mathrm{sign}(m_i)\,(1-2\pi_i),\qquad\pi_i=\Pr\lbrace \mathrm{sign}(\beta_1m_i+(1-\beta_1)G_i)\ne\mathrm{sign}(m_i)\rbrace ,
```

and $`\Delta^m=(1-\beta_2)(G-m)`$. The parameter block's statistic is the flip-probability vector $`\pi`$.

- If momentum dominates, i.e. $`(1-\beta_1)\lvert G_i\rvert<\beta_1\lvert m_i\rvert`$ almost surely under both laws, then $`\pi_i=0`$, and coordinate $`i`$'s parameter increment does not depend on the message.
- The $`m`$ block still carries $`(1-\beta_2)\delta_1`$. Its first-order effect $`\langle\nabla_mV,(1-\beta_2)\delta_1\rangle`$ vanishes wherever the continuation is locally constant in $`m`$, that is, where no later update sign changes under small changes of $`m`$.

**(g) LoRA at initialization.** The adapter gradients are $`G_A=\alpha B^\top\Gamma`$ and $`G_B=\alpha\Gamma A^\top`$. At $`B=B_0=0`$:

- $`G_A=0`$ for every message. Under SGD, under momentum started from $`m=0`$, and under cold Adam, the first update leaves $`A`$ unchanged for every law, exactly.
- The $`B`$-block statistic is $`\alpha\,\mathbb E[\Gamma]A_0^\top`$ for SGD and momentum, and $`\mathbb E\,h_\epsilon(\alpha\Gamma A_0^\top)`$ for cold Adam.
- If two laws have mean full-weight gradients that differ only by $`\Delta\Gamma`$ with $`\Delta\Gamma A_0^\top=0`$, they agree at first order in step 1 under SGD and momentum. Under cold Adam, laws whose sketches $`\Gamma A_0^\top`$ have the same sign law give identical first updates as $`\epsilon\to0`$.
- After step 1, $`B\ne0`$ and $`G_A=\alpha B^\top\Gamma`$ carries a second sketch, so the dimension of the statistic grows with training.

**Summary.**

| Part | Learner and state | First-order statistic of the feedback law | Enters through |
|---|---|---|---|
| **(a)** | SGD | $`\mathbb E\,G`$ | $`\theta`$ |
| **(b)** | heavy-ball, Nesterov | $`\mathbb E\,G`$ | $`\theta`$ and $`m`$, with weight $`-\eta\gamma\nabla_\theta V+\nabla_mV`$ |
| **(c)** | Adam/AdamW, warm | exactly the mean Adam increment; $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ under the frozen-denominator approximation, error $`O(\eta\,u\,\kappa)`$ | $`\theta`$ (preconditioned by $`\Phi`$), $`m`$, $`v`$ |
| **(d)** | Adam/AdamW, cold | $`\mathbb E\,h_\epsilon(G)`$ (the sign law as $`\epsilon\to0`$); $`\mathbb E\,G`$ and $`\mathbb E\,G^{\odot2}`$ | $`\theta`$; $`m`$ and $`v`$ |
| **(e)** | clipped SGD | $`\mathbb E\,\mathrm{clip}_c(G)`$ exactly; the mean unit direction up to $`c(1-p_{\rm act})`$, exactly when clipping is always active | $`\theta`$ |
| **(e)** | clipped base optimizer | the statistic of the base optimizer, taken of $`\mathrm{clip}_c(G)`$ | as for the base optimizer |
| **(f)** | Lion / sign with momentum | flip probabilities $`\pi`$; none in $`\theta`$ when momentum dominates | $`\theta`$; $`m`$ |
| **(g)** | LoRA at $`B=0`$, step 1 | the rank-$`r`$ sketch $`\mathbb E[\Gamma]A_0^\top`$ (SGD), or its sign law (cold Adam); $`A`$ is fixed | $`B`$ only |

*Proof.* Throughout, $`\mu^P-\mu^Q=\mathbb E_PD-\mathbb E_QD`$, where $`D(y)=\mathbb E_\xi\Delta(z,G(y,\xi))`$. Any term of $`\Delta`$ that is a fixed function of the state $`z`$ has equal expectation under both laws and cancels. Each case computes $`\Delta`$ from its update rule and takes the difference of expectations.

- *(a)* $`\Delta^\theta=-\eta\lambda_{\rm wd}\theta-\eta G`$. The first term cancels, and linearity gives $`-\eta\delta_1`$. The second-order statistic follows from Proposition 5.1(a) with $`k=2`$: $`\mathbb E\,\Delta^{\otimes2}=\eta^2\mathbb E\,GG^\top`$ up to state-only terms.
- *(b)* For heavy-ball, $`\theta^+-\theta=-\eta(\beta m+G)`$ and $`m^+-m=(\beta-1)m+G`$. For Nesterov, $`\theta^+-\theta=-\eta(G+\beta m^+)=-\eta(G+\beta^2m+\beta G)`$. The message-dependent parts are $`-\eta\gamma G`$ in $`\theta`$ and $`G`$ in $`m`$, so $`\mu^P-\mu^Q=(-\eta\gamma\delta_1,\ \delta_1)`$; pairing with $`(\nabla_\theta V,\nabla_mV)`$ gives the display.
- *(c)* For the moment blocks, $`m^+-m=(1-\beta_1)(G-m)`$ and $`v^+-v=(1-\beta_2)(G^{\odot2}-v)`$ give $`(1-\beta_1)\delta_1`$ and $`(1-\beta_2)\delta_2`$. For the parameter block, $`\tilde\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\Phi\beta_1m/(1-\beta_1^{\,t})-\eta c_1\Phi G`$ by Definition 5.4, and $`\theta^+=\tilde\theta^++e(G)`$. Taking the difference of expectations gives the $`\theta`$ block, and $`\lvert\mathbb E_Pe_i-\mathbb E_Qe_i\rvert\le\mathbb E_P\lvert e_i\rvert+\mathbb E_Q\lvert e_i\rvert`$ with Proposition 5.5(b) gives the error bound. If $`\delta_1=0`$, the $`\theta`$ block reduces to $`\mathbb E_Pe-\mathbb E_Qe`$ and the $`m`$ block vanishes, which gives the stated form of the first-order term and its bound. The statement on the size of the error along training is Proposition 5.8(a).
- *(d)* At $`m=v=0`$ and $`t=1`$, $`n=G`$ and $`d=\lvert G\rvert`$, so $`\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta h_\epsilon(G)`$ (Proposition 5.3(a)). The three increments are $`-\eta\lambda_{\rm wd}\theta-\eta h_\epsilon(G)`$, $`(1-\beta_1)G`$ and $`(1-\beta_2)G^{\odot2}`$. As $`\epsilon\to0`$, $`h_\epsilon(g)_i\to\mathrm{sign}(g_i)`$ with $`h_0(0)=0`$. If $`R=1`$ the continuation is $`F_E(\theta)`$, so $`\nabla_mV=\nabla_vV=0`$ and only the parameter block contributes.
- *(e)* With clipped SGD, $`\Delta^\theta=-\eta\,\mathrm{clip}_c(G)`$ up to weight decay. For the witness, $`\mathbb E_PG=\tfrac14\cdot4e_1+\tfrac34\cdot(-\tfrac43e_1)=0=\mathbb E_QG`$, while $`\mathrm{clip}_1(4e_1)=e_1`$ and $`\mathrm{clip}_1(-\tfrac43e_1)=-e_1`$ give $`\mathbb E_P\mathrm{clip}_1G=-\tfrac12e_1`$; hence $`\mu^P-\mu^Q=\tfrac\eta2e_1`$. The mean-direction bound is Proposition 5.9 applied under each law, with the triangle inequality. If $`\lVert G\rVert>c`$ almost surely, $`\mathrm{clip}_c(G)=cG/\lVert G\rVert`$; equal direction laws make the increments $`-\eta cG/\lVert G\rVert`$ equal in law, so every term of Proposition 5.1(a) and the exact response coincide. Clipped base optimizers apply the base update to $`\mathrm{clip}_c(G)`$.
- *(f)* Write $`c_i=\beta_1m_i+(1-\beta_1)G_i`$. For $`m_i\ne0`$: off the flip event, $`\mathrm{sign}(c_i)=\mathrm{sign}(m_i)`$; on a strict flip, $`\mathrm{sign}(c_i)=-\mathrm{sign}(m_i)`$; on $`c_i=0`$, $`\mathrm{sign}(c_i)=0=\mathrm{sign}(m_i)(1-2\cdot\tfrac12)`$. Hence $`\mathbb E\,\mathrm{sign}(c_i)=\mathrm{sign}(m_i)(1-2\pi_i)`$. Under momentum dominance $`c_i`$ has the sign of $`m_i`$ almost surely, so $`\pi_i=0`$. The $`m`$-block increment $`(1-\beta_2)(G-m)`$ gives $`(1-\beta_2)\delta_1`$. Later parameter updates depend on $`m`$ only through the signs of later $`c`$'s; where none of these changes under a small perturbation of $`m`$, the continuation is locally constant in $`m`$ and $`\nabla_mV=0`$.
- *(g)* For $`W=W_0+\alpha BA`$ the chain rule gives $`\partial\ell/\partial B_{jl}=\alpha\sum_q\Gamma_{jq}A_{lq}`$ and $`\partial\ell/\partial A_{lq}=\alpha\sum_jB_{jl}\Gamma_{jq}`$, i.e. $`G_B=\alpha\Gamma A^\top`$ and $`G_A=\alpha B^\top\Gamma`$. At $`B_0=0`$, $`G_A=0`$ identically: SGD moves $`A`$ by $`0`$; momentum from $`m=0`$ gives $`m^+=0`$; cold Adam moves $`A`$ by $`-\eta h_\epsilon(0)=0`$. The $`B`$ increment is $`-\eta\alpha\Gamma A_0^\top`$ under SGD, the same at the first momentum step, and $`-\eta h_\epsilon(\alpha\Gamma A_0^\top)`$ under cold Adam; expectations give the statistics. If $`\Delta\Gamma A_0^\top=0`$, the $`B`$-block means agree and the $`A`$-block means are zero, so $`\mu^P=\mu^Q`$ in step 1; under cold Adam, equal sign laws of the sketches give equal laws of $`h_0(\alpha\Gamma A_0^\top)`$. After one step $`B_1\ne0`$ in general, so $`G_A=\alpha B_1^\top\Gamma`$ no longer vanishes. $`\square`$

**Consequences of Theorem 5.10.**

- *Which statistic, not just which data.* The same pair of feedback laws can be first-order equivalent for one learner and first-order distinct for another. Two label laws with equal means are first-order equivalent for SGD and momentum, but can be distinct for cold Adam, sign optimizers and clipped SGD, whose increments see the sign law or the clipped mean, and for warm Adam through the second-moment channel. Teacher data must therefore be judged under the student's actual optimizer.
- *Few-shot LoRA with AdamW and global-norm clipping.* At initialization, row (g) applies: the first update sees the feedback only through the sign law of the rank-$`r`$ sketch $`\Gamma A_0^\top`$. Once clipping is active on part of the batch, rows (c)–(e) are taken of the clipped gradient, with the mean-direction simplification accurate only to $`c(1-p_{\rm act})`$.
- *Relation to Propositions 5.3 and 5.6.* Row (d) unifies Proposition 5.3(a)–(c) and (e): the parameter block carries the sign law, and the moment blocks carry the magnitudes that can reverse the second step. Row (c) is the linearized form of Proposition 5.6; it refines the residual $`L_v\kappa`$ of Proposition 5.6 into the explicit linear term $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle`$.

## Related work

- Adam-preconditioned gradient features for data selection after a LoRA warm-up are used by LESS ([Xia et al., 2024](../references.md#xia2024less)); they correspond to the parameter block of Theorem 5.10(c). The Adam version of In-Run Data Shapley ([Ding et al., 2026](../references.md#ding2026inrun)) uses a fixed-state approximation, i.e. the current preconditioned step only.

## References

- [Xia, M., Malladi, S., Gururangan, S., Arora, S., & Chen, D. (2024). LESS: Selecting influential data for targeted instruction tuning. *ICML*.](../references.md#xia2024less)
- [Ding, M., Zhang, Z., Wang, D., & Hu, L. (2026). In-Run Data Shapley for Adam optimizer. arXiv:2602.00329.](../references.md#ding2026inrun)
