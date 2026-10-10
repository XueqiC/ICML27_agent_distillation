# N19 · What each learner makes the feedback carry — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N19_learner_statistics.md) · [中文](N19_learner_statistics_ZH.md)

## Setting and assumptions

Use the setting of node [N18](N18_optimizer_transformed_response.md).

- The complete state is $x=(\theta,s)$, the consumed gradient of message $y$ is $G=G(y)$, and the optimizer increment is $\Delta(x,G)$.
- For two laws $P,Q$ at a state, the first-order statistic is the difference of mean increments $\mu^P-\mu^Q$. By Theorem C3 of node N18, its projection onto the span $U$ of the continuation gradients is exactly what two feedback laws must agree on to first order.
- Write $\delta_1=\mathbb E_PG-\mathbb E_QG$ and $\delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}$.
- Every term of $\Delta$ that does not depend on the message cancels in $\mu^P-\mu^Q$ and is omitted: weight decay, momentum decay, and bias-correction constants applied to the old state.

**Optimizers.** All operations are coordinatewise.

- *SGD:* $\theta^+=(1-\eta\lambda)\theta-\eta G$.
- *Heavy-ball:* $m^+=\beta m+G$, $\theta^+=\theta-\eta m^+$. *Nesterov:* $m^+=\beta m+G$, $\theta^+=\theta-\eta(G+\beta m^+)$.
- *Adam and AdamW* (bias-corrected, $\epsilon$ outside the square root, step counter $t$): $m^+=\beta_1m+(1-\beta_1)G$, $v^+=\beta_2v+(1-\beta_2)G^{\odot2}$, $\theta^+=(1-\eta\lambda)\theta-\eta\,n(G)/(d(G)+\epsilon)$, with $n=m^+/(1-\beta_1^t)$ and $d=\sqrt{v^+/(1-\beta_2^t)}$. A *warm* state has $v_i>0$ for all $i$. The *cold* state is $m=v=0$ at $t=1$.
- *Global-norm clipping:* $\operatorname{clip}_c(G)=G\min\lbrace 1,c/\lVert G\rVert\rbrace$, applied before the base optimizer.
- *Lion form of sign optimizers:* $u=\operatorname{sign}(\beta_1m+(1-\beta_1)G)$, $\theta^+=\theta-\eta u$, $m^+=\beta_2m+(1-\beta_2)G$.
- *LoRA:* an adapted weight $W=W_0+\alpha BA$ with $A\in\mathbb R^{r\times k}$, $B\in\mathbb R^{d\times r}$ and full-weight gradient $\Gamma=\nabla_W\ell$. The trainable parameters are $(A,B)$. The standard initialization is $B_0=0$ with $A_0$ random.

**Warm-Adam notation** (node [N12](N12_optimizer_warmth.md)): $a_i=\sqrt{\beta_2v_i/(1-\beta_2^t)}$, $P=\operatorname{diag}(1/(a_i+\epsilon))$, $c_1=(1-\beta_1)/(1-\beta_1^t)$, $\kappa_i=(1-\beta_2)G_i^2/(\beta_2v_i)$ and $u_i=\lvert n_i\rvert/(a_i+\epsilon)$.

## Statement

**Theorem C4 (the first-order statistic of each learner).**

**(a) SGD.** $\mu^P-\mu^Q=(-\eta\,\delta_1)$. The first-order statistic is the mean consumed gradient. The second-order statistic is $\mathbb E\,GG^\top$.

**(b) Heavy-ball and Nesterov momentum.** The increments are affine in $G$:

- heavy-ball: $\Delta^\theta=-\eta(\beta m+G)$ and $\Delta^m=(\beta-1)m+G$;
- Nesterov: $\Delta^\theta=-\eta\big((1+\beta)G+\beta^2m\big)$ and $\Delta^m=(\beta-1)m+G$.

Hence the first-order term is
$$\langle\nabla V,\mu^P-\mu^Q\rangle=\big\langle-\eta\gamma\nabla_\theta V+\nabla_mV,\ \delta_1\big\rangle,\qquad\gamma=1\ \text{(heavy-ball)},\quad\gamma=1+\beta\ \text{(Nesterov)}.$$
The statistic is the mean consumed gradient. Momentum changes its weight, not the statistic.

**(c) Adam and AdamW at a warm state.** There is a message-dependent $e(G)$ with $\lvert e_i\rvert\le\eta u_i\min\lbrace \kappa_i/2,1\rbrace$ such that
$$\mu^P-\mu^Q=\Big(-\eta c_1P\,\delta_1+\big(\mathbb E_Pe-\mathbb E_Qe\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big)$$
in the blocks $(\theta,m,v)$. The first-order statistic is the pair $(\mathbb E\,G,\ \mathbb E\,G^{\odot2})$, up to the warmth error, which is at most $\eta(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace \kappa_i/2,1\rbrace]$ per coordinate. The coordinatewise second moment enters at first order through $v$, weighted by $(1-\beta_2)\nabla_vV$. Equal-mean laws are therefore not first-order equivalent for warm Adam unless $\nabla_vV\perp\delta_2$.

**(d) Adam at the cold state.** With $h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)$,
$$\mu^P-\mu^Q=\Big(-\eta\big(\mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).$$
The parameter block sees only $\mathbb E\,h_\epsilon(G)$, which as $\epsilon\to0$ is the sign law (node [N11](N11_adam_cold_start.md)). The moment blocks retain $\mathbb E\,G$ and $\mathbb E\,G^{\odot2}$. If the run ends after this update ($R=1$), the continuation is $F_E(\theta)$, and the first-order term is
$$-\eta\,\big\langle\nabla_\theta V,\ \mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\big\rangle.$$

**(e) Gradient clipping.** With clipped SGD, $\mu^P-\mu^Q=-\eta\big(\mathbb E_P\operatorname{clip}_c(G)-\mathbb E_Q\operatorname{clip}_c(G)\big)$, so the statistic is the clipped mean.

- *Witness.* Take $c=1$, $P(G=4e_1)=\tfrac14$, $P(G=-\tfrac43e_1)=\tfrac34$ and $Q=\delta_0$. Then $\delta_1=0$ but $\mathbb E_P\operatorname{clip}_1G=-\tfrac12e_1$, and the first-order term is $\tfrac\eta2\,\partial_{\theta_1}V$.
- *Always-active clipping.* If $\lVert G\rVert>c$ almost surely under both laws, the statistic is $c\,\mathbb E[G/\lVert G\rVert]$, the mean unit direction. If in addition the two laws induce the same law of $G/\lVert G\rVert$, the increment laws coincide and the laws are equivalent at every order.
- *Clipped base optimizers.* For any base optimizer in (a)–(d) and (f), the statistics are those of the clipped gradient $\operatorname{clip}_c(G)$.

**(f) Sign optimizers with momentum (Lion form).** For $m_i\ne0$,
$$\mathbb E\,\Delta^\theta_i=-\eta\operatorname{sign}(m_i)\,(1-2\pi_i),\qquad\pi_i=\Pr\lbrace \operatorname{sign}(\beta_1m_i+(1-\beta_1)G_i)\ne\operatorname{sign}(m_i)\rbrace ,$$
with a zero counted as half a flip, and $\Delta^m=(1-\beta_2)(G-m)$. The parameter block's statistic is the flip-probability vector $\pi$.

- If momentum dominates, i.e. $(1-\beta_1)\lvert G_i\rvert<\beta_1\lvert m_i\rvert$ almost surely under both laws, then $\pi_i=0$, and coordinate $i$'s parameter increment does not depend on the message.
- The $m$-block still carries $(1-\beta_2)\delta_1$. Its first-order effect is $\langle\nabla_mV,(1-\beta_2)\delta_1\rangle$, which vanishes wherever the continuation is locally constant in $m$, that is, where no later update sign changes under small changes of $m$.

**(g) LoRA at initialization.** The adapter gradients are $G_A=\alpha B^\top\Gamma$ and $G_B=\alpha\Gamma A^\top$. At $B=B_0=0$:

1. $G_A=0$ for every message. Under SGD, under momentum started from $m=0$, and under cold Adam, the first update leaves $A$ unchanged for every law, bitwise in exact arithmetic.
2. The $B$-block statistic is $\alpha\,\mathbb E[\Gamma]A_0^\top$ for SGD and momentum, and $\mathbb E\,h_\epsilon(\alpha\Gamma A_0^\top)$ for cold Adam.
3. If two laws have mean full-weight gradients that differ only by $\Delta\Gamma$ with $\Delta\Gamma A_0^\top=0$, they agree at first order in step 1 under SGD and momentum. Under cold Adam, laws whose sketches $\Gamma A_0^\top$ have the same sign law give identical first updates as $\epsilon\to0$.
4. After step 1, $B\ne0$ and $G_A=\alpha B^\top\Gamma$ carries a second sketch, so the dimension of the statistic grows with training.

**Summary.** Each learner's first-order statistic of the feedback:

| Learner and state | First-order statistic of the feedback law | Enters through |
|---|---|---|
| SGD | $\mathbb E\,G$ | $\theta$ |
| heavy-ball, Nesterov | $\mathbb E\,G$ | $\theta$ and $m$, with weight $-\eta\gamma\nabla_\theta V+\nabla_mV$ |
| Adam/AdamW, warm | $\mathbb E\,G$ and $\mathbb E\,G^{\odot2}$, up to a warmth error $O(\eta\,u\,\kappa)$ | $\theta$ (preconditioned by $P$), $m$, $v$ |
| Adam/AdamW, cold | $\mathbb E\,h_\epsilon(G)$ (the sign law as $\epsilon\to0$); $\mathbb E\,G$ and $\mathbb E\,G^{\odot2}$ | $\theta$; $m$ and $v$ |
| clipped SGD | $\mathbb E\operatorname{clip}_c(G)$; the mean unit direction when clipping is always active | $\theta$ |
| clipped base optimizer | the statistic of the base optimizer, taken of $\operatorname{clip}_c(G)$ | as for the base optimizer |
| Lion / sign with momentum | flip probabilities $\pi$; none in $\theta$ when momentum dominates | $\theta$; $m$ |
| LoRA at $B=0$, step 1 | the rank-$r$ sketch $\mathbb E[\Gamma]A_0^\top$ (SGD), or its sign law (cold Adam); $A$ is fixed | $B$ only |

## Proof

Throughout, $\mu^P-\mu^Q=\mathbb E_PD-\mathbb E_QD$, where $D(y)=\mathbb E_\xi\Delta(x,G(y,\xi))$. Any term of $\Delta$ that is a fixed function of the state $x$ has equal expectation under both laws and cancels. Each case computes $\Delta$ from its update rule and takes the difference of expectations.

**(a)** $\Delta^\theta=-\eta\lambda\theta-\eta G$. The first term cancels, and linearity of expectation gives $-\eta\delta_1$. The second-order statistic follows from Theorem C1 with $k=2$: $\mathbb E\,\Delta^{\otimes2}=\eta^2\mathbb E\,GG^\top$ up to state-only terms.

**(b)** For heavy-ball, $\theta^+-\theta=-\eta m^+=-\eta(\beta m+G)$ and $m^+-m=(\beta-1)m+G$. For Nesterov, $\theta^+-\theta=-\eta(G+\beta m^+)=-\eta(G+\beta^2m+\beta G)$. In both cases the state-only terms cancel. The message-dependent parts are $-\eta\gamma G$ in $\theta$ and $G$ in $m$. Hence $\mu^P-\mu^Q=(-\eta\gamma\delta_1,\ \delta_1)$, and pairing with $(\nabla_\theta V,\nabla_mV)$ gives the display.

**(c)** For the moment blocks, $m^+-m=(1-\beta_1)(G-m)$ and $v^+-v=(1-\beta_2)(G^{\odot2}-v)$, which give $(1-\beta_1)\delta_1$ and $(1-\beta_2)\delta_2$.

For the parameter block, write the frozen-denominator surrogate of node [N12](N12_optimizer_warmth.md):
$$\tilde\theta^+=(1-\eta\lambda)\theta-\eta P\,n(G)=(1-\eta\lambda)\theta-\eta P\frac{\beta_1m}{1-\beta_1^t}-\eta c_1P\,G .$$
Lemma W0 gives $\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta u_i\min\lbrace \kappa_i/2,1\rbrace$. Define $e(G)=\theta^+-\tilde\theta^+$. Then $\Delta^\theta=(\text{state-only terms})-\eta c_1PG+e(G)$, and taking the difference of expectations gives the $\theta$-block. The bound on $\mathbb E_Pe_i-\mathbb E_Qe_i$ follows from $\lvert\mathbb E_Pe_i-\mathbb E_Qe_i\rvert\le\mathbb E_P\lvert e_i\rvert+\mathbb E_Q\lvert e_i\rvert$.

The last claim: if $\delta_1=0$, the first-order term reduces to $\langle\nabla_\theta V,\mathbb E_Pe-\mathbb E_Qe\rangle+(1-\beta_2)\langle\nabla_vV,\delta_2\rangle$. Its second summand is nonzero whenever $\nabla_vV\not\perp\delta_2$.

**(d)** At $m=v=0$ and $t=1$, $m^+=(1-\beta_1)G$ and $1-\beta_1^1=1-\beta_1$, so $n=G$. Similarly $v^+=(1-\beta_2)G^{\odot2}$ and $d=\lvert G\rvert$. Hence $\theta^+=(1-\eta\lambda)\theta-\eta G/(\lvert G\rvert+\epsilon)=(1-\eta\lambda)\theta-\eta h_\epsilon(G)$. The three increments are $-\eta\lambda\theta-\eta h_\epsilon(G)$, $(1-\beta_1)G$ and $(1-\beta_2)G^{\odot2}$, which give the display. As $\epsilon\to0$, $h_\epsilon(g)_i\to\operatorname{sign}(g_i)$, with $h_0(0)=0$ (Prop. A2, node N11).

If $R=1$, the continuation after the update is $F_E(\theta)$. Then $\nabla_mV=\nabla_vV=0$, and the first-order term is $\langle\nabla_\theta V,\mu^P_\theta-\mu^Q_\theta\rangle=-\eta\langle\nabla_\theta V,\mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\rangle$.

**(e)** With clipped SGD, $\Delta^\theta=-\eta\operatorname{clip}_c(G)$ up to weight decay, and the difference of expectations is as stated.

- *Witness.* $\mathbb E_PG=\tfrac14\cdot4e_1+\tfrac34\cdot(-\tfrac43e_1)=0=\mathbb E_QG$. Also $\operatorname{clip}_1(4e_1)=e_1$ and $\operatorname{clip}_1(-\tfrac43e_1)=-e_1$, so $\mathbb E_P\operatorname{clip}_1G=(\tfrac14-\tfrac34)e_1=-\tfrac12e_1$, while $\mathbb E_Q\operatorname{clip}_1G=0$. Hence $\mu^P-\mu^Q=-\eta(-\tfrac12e_1)=\tfrac\eta2e_1$, and the first-order term is $\tfrac\eta2\,\partial_{\theta_1}V$.
- *Always-active clipping.* If $\lVert G\rVert>c$, then $\operatorname{clip}_c(G)=cG/\lVert G\rVert$, so $\mathbb E\operatorname{clip}_cG=c\,\mathbb E[G/\lVert G\rVert]$. If the two laws induce the same law of $G/\lVert G\rVert$, the increments $-\eta cG/\lVert G\rVert$ have the same law, so every term of Theorem C1 and the exact response coincide.
- *Clipped base optimizers.* They apply the base update to $\operatorname{clip}_c(G)$; replace $G$ by $\operatorname{clip}_c(G)$ in the base computations.

**(f)** Write $c_i=\beta_1m_i+(1-\beta_1)G_i$. For $m_i\ne0$:

- off the flip event, $\operatorname{sign}(c_i)=\operatorname{sign}(m_i)$;
- on a strict flip, $\operatorname{sign}(c_i)=-\operatorname{sign}(m_i)$;
- on $c_i=0$, $\operatorname{sign}(c_i)=0=\operatorname{sign}(m_i)(1-2\cdot\tfrac12)$.

Hence $\mathbb E\operatorname{sign}(c_i)=\operatorname{sign}(m_i)(1-2\pi_i)$ with zeros counted as half a flip, and $\mathbb E\Delta^\theta_i=-\eta\operatorname{sign}(m_i)(1-2\pi_i)$.

If $(1-\beta_1)\lvert G_i\rvert<\beta_1\lvert m_i\rvert$ almost surely, then $c_i$ has the sign of $m_i$ almost surely, so $\pi_i=0$ and $\Delta^\theta_i=-\eta\operatorname{sign}(m_i)$ for every message.

The $m$-block increment is $(1-\beta_2)(G-m)$, giving $(1-\beta_2)\delta_1$. Later parameter updates depend on $m$ only through the signs of later $c$'s. Where none of these signs changes under a small perturbation of $m$, the continuation is locally constant in $m$, so $\nabla_mV=0$ there.

**(g)** For $W=W_0+\alpha BA$, the chain rule gives $\partial\ell/\partial B_{jl}=\alpha\sum_q\Gamma_{jq}A_{lq}$, i.e. $G_B=\alpha\Gamma A^\top$. Likewise $\partial\ell/\partial A_{lq}=\alpha\sum_jB_{jl}\Gamma_{jq}$, i.e. $G_A=\alpha B^\top\Gamma$.

1. At $B_0=0$, $G_A=0$ identically. SGD moves $A$ by $-\eta\cdot0=0$. Momentum from $m=0$ gives $m^+=0$, hence no move. Cold Adam moves $A$ by $-\eta h_\epsilon(0)=0$, since $h_\epsilon(0)=0/(0+\epsilon)=0$ for $\epsilon>0$. All three hold for every message, so for every law.
2. The $B$-increment is $-\eta\alpha\Gamma A_0^\top$ under SGD; under momentum from $m=0$ its first step equals SGD's; under cold Adam it is $-\eta h_\epsilon(\alpha\Gamma A_0^\top)$. Taking expectations gives the statistics.
3. If $\mathbb E_P\Gamma-\mathbb E_Q\Gamma=\Delta\Gamma$ with $\Delta\Gamma A_0^\top=0$, the $B$-block means agree under SGD and momentum, and the $A$-block means are both zero. So $\mu^P=\mu^Q$ in step 1. Under cold Adam, if the sketches have the same sign law, the limits $h_0(\alpha\Gamma A_0^\top)$ have the same law, so the first updates agree as $\epsilon\to0$.
4. After one step $B_1\ne0$ in general, so $G_A=\alpha B_1^\top\Gamma$ no longer vanishes. $\square$

## Remarks

1. **Which statistic, not just which data.** The same pair of feedback laws can be equivalent for one learner and far apart for another:
   - the C23 moment laws have equal means, so they are first-order equivalent for SGD and momentum;
   - they are first-order distinct for cold Adam, sign optimizers and clipped SGD, whose increments see the sign law or the clipped mean.

   Teacher data must therefore be judged under the student's actual optimizer.
2. **Few-shot LoRA recipes.** Our Seed-SFT recipe trains LoRA with AdamW from a cold state and clips at global norm 1. The main-table K20 run recorded a pre-clip norm of 5.28 at its final update, so clipping is active and the statistics of rows (c)–(e) are taken of the clipped gradient. At initialization, row (g) applies: the first update sees the feedback only through the sign law of the rank-$r$ sketch $\Gamma A_0^\top$.
3. **Relation to earlier nodes.**
   - Row (d) unifies Props. A2 and A4 of node [N11](N11_adam_cold_start.md): the parameter block carries the sign law, and the moment blocks carry the magnitudes that A4 shows can reverse the second step.
   - Row (c) is the linearized form of Prop. W1 of node [N12](N12_optimizer_warmth.md). It refines W1's $L_v\kappa$ residual into an explicit linear term in $\mathbb E\,G^{\odot2}$.
4. **Precedents.** The cold first step of Adam (Kingma and Ba, 2015), Adam as sign descent with variance adaptation (Balles and Hennig, 2018), signSGD (Bernstein et al., 2018) and Lion (Chen et al., 2023) are established. Adam-preconditioned gradient features for data selection after a LoRA warmup are used by LESS (Xia et al., 2024); they correspond to the parameter block of row (c). The contribution here is the complete-state catalog, with the moment blocks included, as the minimal first-order statistic of Theorem C3.

## Dependencies and source

- **Uses:** [N18](N18_optimizer_transformed_response.md) (Theorems C1 and C3), [N11](N11_adam_cold_start.md) (cold Adam map (A7)), [N12](N12_optimizer_warmth.md) (Lemma W0 and warm-Adam notation).
- **Used by:** [N20](N20_first_order_audit.md), [N17](N17_ced_method.md).
- **Source:** `docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md`, §4 (Theorem C4(a)–(g)).
