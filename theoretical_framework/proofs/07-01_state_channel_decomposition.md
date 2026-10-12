# Theorem 7.1 · State-channel decomposition

*Proof*

[← Framework](../README.md) · [Statement](../statements/07-01_state_channel_decomposition.md) · [Proof](07-01_state_channel_decomposition.md) · [Experiments](../experiments/07-01_state_channel_decomposition.md) · [中文](07-01_state_channel_decomposition_ZH.md)

**What is proved here.** The gradient of the continuation value with respect to the complete Adam/AdamW state $`(\theta,m,v)`$ equals a costate computed by a backward recursion along the reference tail. Pairing it with the Adam increment of a message splits the message's first-order value exactly into a parameter channel, a momentum channel and a second-moment channel. The proof is the chain rule applied to the composition of tail updates, with every Jacobian block written out.

**Setting used on this page.**

- *Complete learner state.* $`z=(\theta,o)`$ with optimizer state $`o=(m,v,s)`$: first moment $`m`$, second moment $`v`$, absolute step counter $`s`$ ([Definition 1.3](../README.md#s1-3)).
- *AdamW update* with $`\beta_1\in[0,1)`$, $`\beta_2\in(0,1)`$, $`\eta>0`$, $`\epsilon\ge0`$, decoupled weight decay $`\lambda_{\rm wd}\ge0`$, applied to a consumed gradient $`g`$ at step $`s`$:

```math
m'=\beta_1m+(1-\beta_1)g,\qquad v'=\beta_2v+(1-\beta_2)\,g\odot g,\qquad \theta'=(1-\eta\lambda_{\rm wd})\,\theta-\eta\,q_{1,s}\,\frac{m'}{\sqrt{q_{2,s}v'}+\epsilon},
```

- with bias-correction factors $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$, $`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$ (or both $`1`$), all operations coordinatewise. The increment is $`\Delta(z,g)=(\theta'-\theta,\;m'-m,\;v'-v)`$ ([Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5)); the counter advances and is not differentiated.
- *Tail.* After a message at round $`t`$, updates $`k=t+1,\dots,R-1`$ use consumed gradients $`g_k(\theta)`$ (loss gradients $`\nabla\ell_k(\theta)`$, possibly global-norm clipped at threshold $`c`$). The score is $`F(\theta_R)`$, and $`V_{t+1}(z)`$ is the score reached when the tail starts from $`z`$ ([Definition 1.4](../README.md#s1-4)).
- *First-order value.* $`\mathrm{Val}(g)=\langle\nabla V_{t+1}(z_t),\Delta_t(z_t,g)\rangle`$ ([Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1)).

**Statement ([Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1)).** Fix a realization of the tail randomness and assume:

**(D1)** $`F`$ is $`C^1`$, and each $`g_k`$ is $`C^1`$ in $`\theta`$ near the tail trajectory started at $`z_t`$, with Jacobian $`J_k=\partial g_k/\partial\theta`$; without clipping $`J_k=H_k=\nabla^2\ell_k(\theta_k)`$, and with clipping active at threshold $`c`$ and $`\lVert\nabla\ell_k(\theta_k)\rVert\ne c`$ along the trajectory,

```math
J_k=\frac{c}{\lVert r_k\rVert}\Big(I-\frac{r_kr_k^\top}{\lVert r_k\rVert^2}\Big)H_k,\qquad r_k=\nabla\ell_k(\theta_k).
```

**(D2)** $`v'_k>0`$ coordinatewise at every tail update.

With $`m'_k,v'_k`$ the moments after update $`k`$ and

```math
D_1^{(k)}=\mathrm{diag}\Big(\frac{q_{1,s_k}}{\sqrt{q_{2,s_k}v'_k}+\epsilon}\Big),\qquad D_2^{(k)}=\mathrm{diag}\Big(\frac{q_{1,s_k}\,q_{2,s_k}\,m'_k}{2\sqrt{q_{2,s_k}v'_k}\,\big(\sqrt{q_{2,s_k}v'_k}+\epsilon\big)^2}\Big),
```

define $`\chi_R=(\nabla F(\theta_R),0,0)`$ and, for $`k=R-1,\dots,t+1`$,

```math
\begin{aligned}
\chi^m_k&=\beta_1\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big),\qquad \chi^v_k=\beta_2\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big),\\
\chi^\theta_k&=(1-\eta\lambda_{\rm wd})\chi^\theta_{k+1}+J_k^\top\Big[(1-\beta_1)\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big)+2(1-\beta_2)\,\mathrm{diag}(g_k)\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big)\Big].
\end{aligned}
```

Then (a) $`V_{t+1}`$ is differentiable at $`z_t`$ and $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$; (b) for every message $`g`$,

```math
\mathrm{Val}(g)=\big\langle\chi^\theta_{t+1},\Delta\theta(g)\big\rangle+\big\langle\chi^m_{t+1},(1-\beta_1)(g-m_t)\big\rangle+\big\langle\chi^v_{t+1},(1-\beta_2)(g\odot g-v_t)\big\rangle ,
```

the parameter, momentum and second-moment channels; (c) the second-moment channel depends on $`g`$ only through $`g\odot g`$ (on a law only through $`\mathbb E_P\,G\odot G`$), the momentum channel is affine in $`g`$, and the parameter channel is nonlinear in $`g`$; (d) if $`J_k=0`$ for all tail updates and $`\lambda_{\rm wd}=0`$, then $`\chi^\theta_k\equiv\nabla F`$ and

```math
\chi^m_{t+1}=-\eta\sum_{k=t+1}^{R-1}\beta_1^{\,k-t}D_1^{(k)}\nabla F,\qquad \chi^v_{t+1}=\eta\sum_{k=t+1}^{R-1}\beta_2^{\,k-t}D_2^{(k)}\nabla F ;
```

(e) for random tails whose branchwise costates are dominated, $`\nabla V_{t+1}(z_t)=\mathbb E_\omega\chi_{t+1,\omega}`$ and (b) holds with $`\mathbb E_\omega\chi_{t+1,\omega}`$.

**Proof.**

*(a)* One tail update is the map $`\Phi_k:(\theta,m,v)\mapsto(\theta',m',v')`$ with $`g=g_k(\theta)`$. Under (D1)–(D2) it is $`C^1`$ near the trajectory: (D1) makes $`g_k`$ differentiable, and (D2) keeps the square root away from $`0`$. Write $`D_1=D_1^{(k)}`$, $`D_2=D_2^{(k)}`$, $`J=J_k`$. Its Jacobian blocks are:

- $`\partial m'/\partial m=\beta_1I`$, $`\partial m'/\partial\theta=(1-\beta_1)J`$, $`\partial m'/\partial v=0`$.
- $`\partial v'/\partial v=\beta_2I`$, $`\partial v'/\partial\theta=2(1-\beta_2)\,\mathrm{diag}(g)\,J`$, $`\partial v'/\partial m=0`$.
- $`\partial\theta'/\partial m'=-\eta D_1`$. The derivative of $`(\sqrt{q_2v'}+\epsilon)^{-1}`$ with respect to $`v'`$ is $`-q_2/\big(2\sqrt{q_2v'}(\sqrt{q_2v'}+\epsilon)^2\big)`$, so $`\partial\theta'/\partial v'=\eta D_2`$.
- Hence $`\partial\theta'/\partial m=-\eta\beta_1D_1`$, $`\partial\theta'/\partial v=\eta\beta_2D_2`$, and $`\partial\theta'/\partial\theta=(1-\eta\lambda_{\rm wd})I-\eta(1-\beta_1)D_1J+2\eta(1-\beta_2)D_2\,\mathrm{diag}(g)\,J`$.

$`V_{t+1}(z)=F(\pi_\theta\,\Phi_{R-1}\circ\cdots\circ\Phi_{t+1}(z))`$, where $`\pi_\theta`$ extracts the parameters. By the chain rule,

```math
\nabla V_{t+1}(z_t)=D\Phi_{t+1}^\top\cdots D\Phi_{R-1}^\top\,(\nabla F,0,0).
```

Accumulating from the right gives $`\chi_k=D\Phi_k^\top\chi_{k+1}`$ with $`\chi_R=(\nabla F,0,0)`$. Multiplying the transposed blocks (all $`D`$'s are diagonal):

- the $`m`$-row collects $`-\eta\beta_1D_1\chi^\theta_{k+1}+\beta_1\chi^m_{k+1}`$;
- the $`v`$-row collects $`\eta\beta_2D_2\chi^\theta_{k+1}+\beta_2\chi^v_{k+1}`$;
- the $`\theta`$-row collects $`(1-\eta\lambda_{\rm wd})\chi^\theta_{k+1}`$ plus $`J^\top`$ applied to $`(1-\beta_1)\chi^m_{k+1}-\eta(1-\beta_1)D_1\chi^\theta_{k+1}+2(1-\beta_2)\mathrm{diag}(g)\big(\chi^v_{k+1}+\eta D_2\chi^\theta_{k+1}\big)`$.

This is the stated recursion. When clipping is active, $`g=c\,r/\lVert r\rVert`$ with $`r=\nabla\ell_k(\theta)`$, and the derivative of $`r\mapsto r/\lVert r\rVert`$ is $`(I-rr^\top/\lVert r\rVert^2)/\lVert r\rVert`$; composing with $`\partial r/\partial\theta=H_k`$ gives the stated $`J_k`$. The recursion uses $`J_k^\top`$, which need not be symmetric.

*(b)* By definition $`\mathrm{Val}(g)=\langle\nabla V_{t+1}(z_t),\Delta_t(z_t,g)\rangle`$. Insert $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$ from (a) and the increments $`\Delta m=(1-\beta_1)(g-m_t)`$ and $`\Delta v=(1-\beta_2)(g\odot g-v_t)`$ of the Adam increment map ([Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5): the exact mean Adam increment is the first-order statistic), and split the inner product by blocks.

*(c)* Read off from (b): the $`v`$-block of the increment depends on $`g`$ only through $`g\odot g`$, the $`m`$-block is affine in $`g`$, and $`\Delta\theta(g)`$ contains $`g`$ in both numerator and denominator. For a law, take expectations in (b).

*(d)* With $`J_k=0`$ and $`\lambda_{\rm wd}=0`$, the $`\theta`$-row gives $`\chi^\theta_k=\chi^\theta_{k+1}=\nabla F`$ for every $`k`$. Then $`\chi^m_k=\beta_1\chi^m_{k+1}-\eta\beta_1D_1^{(k)}\nabla F`$ with $`\chi^m_R=0`$, and unrolling gives $`\chi^m_{t+1}=-\eta\sum_{k=t+1}^{R-1}\beta_1^{\,k-t}D_1^{(k)}\nabla F`$. The $`v`$-row is identical with coefficient $`+\eta\beta_2D_2^{(k)}`$.

*(e)* Apply (a) on each branch and interchange $`\nabla`$ and $`\mathbb E_\omega`$, which is allowed for finitely many branches and, more generally, under domination of the branchwise costates. $`\square`$

**Remarks on the proof.**

- The step counter is held fixed during differentiation; decoupled decay multiplies $`\theta`$ by $`1-\eta\lambda_{\rm wd}`$ with the unmodified learning rate.
- At a clipping threshold the map is not differentiable, and no derivative is claimed there.

## References

- [Kingma & Ba, 2015](../references.md#kingma2015adam): the Adam optimizer.
- [Loshchilov & Hutter, 2019](../references.md#loshchilov2019decoupled): decoupled weight decay (AdamW).
