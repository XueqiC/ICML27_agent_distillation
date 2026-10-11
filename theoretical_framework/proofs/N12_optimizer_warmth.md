# N12 · Optimizer warmth switches the first-order feedback statistic — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N12_optimizer_warmth.md) · [中文](N12_optimizer_warmth_ZH.md)

## Setting and assumptions

**State and update.** Bias-corrected Adam/AdamW acts as in [N11(a)](N11_adam_cold_start.md) (node [N11](N11_adam_cold_start.md)). The hyperparameters satisfy $`\beta_1\in[0,1)`$, $`\beta_2\in(0,1)`$, $`\eta>0`$, $`\lambda\ge0`$ and $`\epsilon\ge0`$. Consider round $`t\ge1`$ from a complete learner state $`z=(\theta,m,v,t-1)`$. A feedback message $`y`$ in a finite set $`Y`$ yields a consumed gradient $`G(y)\in\mathbb R^p`$, defined after accumulation, normalization and clipping. Coordinatewise,

```math
n_i(y)=\frac{\beta_1m_i+(1-\beta_1)G_i(y)}{1-\beta_1^t},\qquad d_i(y)=\sqrt{\frac{\beta_2v_i+(1-\beta_2)G_i(y)^2}{1-\beta_2^t}},
```

```math
\theta_i^+(y)=(1-\eta\lambda)\theta_i-\eta\frac{n_i(y)}{d_i(y)+\epsilon}.
```

The next complete state is $`z^+(y)=(\theta^+(y),m^+(y),v^+(y),t)`$, with $`m^+=\beta_1m+(1-\beta_1)G`$ and $`v^+=\beta_2v+(1-\beta_2)G^{\odot2}`$.

**Definition (warmth).** Assume $`v_i>0`$, or $`\epsilon>0`$ whenever some $`v_i=0`$. Put

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^t}},\qquad\kappa_i(y)=\frac{(1-\beta_2)G_i(y)^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(y)=\infty\ \ (v_i=0),
```

```math
u_i(y)=\frac{\lvert n_i(y)\rvert}{a_i+\epsilon}.
```

When $`v_i>0`$, $`d_i(y)=a_i\sqrt{1+\kappa_i(y)}`$ and $`v_i^+(y)=\beta_2v_i\,(1+\kappa_i(y))`$. Thus $`\kappa_i`$ is the relative perturbation that the new gradient makes to the retained second moment.

**Definition (frozen-denominator surrogate).** Let $`P=\mathrm{diag}(1/(a_i+\epsilon))`$ and

```math
\tilde\theta^+(y)=(1-\eta\lambda)\theta-\eta P\,n(y),\qquad\tilde z^+(y)=(\tilde\theta^+(y),\,m^+(y),\,\beta_2v,\,t).
```

$`\tilde z^+`$ is an affine function of $`G(y)`$. It is one momentum-SGD step with fixed diagonal preconditioner $`P`$, followed by a second moment that ignores the new gradient. For $`g\in\mathbb R^p`$ write $`\tilde z^+(g)`$ for the same map evaluated at consumed gradient $`g`$.

**Continuation and response.** $`V`$ maps complete states to $`[0,1]`$. It is the declared reference learning suffix followed by deployment, with any suffix randomness integrated (node [N05](N05_continuation_response.md)). The one-boundary response of message $`y`$ is $`\psi(y)=V(z^+(y))`$; this is [N05](N05_continuation_response.md) with the update randomness inside $`V`$. Define $`F(g)=V(\tilde z^+(g))`$.

**Assumptions for (W1).** All states below are the ones generated from $`z`$ by messages in $`Y`$; the counter is fixed at $`t`$.

- **(W)** $`v_i>0`$ for every coordinate (warm state).
- **(L)** For all such $`m,v`$: $`\lvert V(\vartheta,m,v)-V(\vartheta',m,v)\rvert\le\sum_iL_i\lvert\vartheta_i-\vartheta_i'\rvert`$ on the pairs $`\big(\theta^+(y),\tilde\theta^+(y)\big)`$ at $`(m,v)=(m^+(y),v^+(y))`$.
- **(S)** There are $`L_v\ge0`$ and a cap $`\bar k\in(0,\infty]`$ such that, at $`(\vartheta,m)=(\tilde\theta^+(y),m^+(y))`$, $`\lvert V(\vartheta,m,\beta_2v\odot(1+k))-V(\vartheta,m,\beta_2v)\rvert\le L_v\max_i\min\lbrace k_i,\bar k\rbrace`$ for $`k=\kappa(y)`$. Since $`V\in[0,1]`$, (S) always holds with $`L_v\bar k\ge1`$; the content lies in small $`L_v`$.
- **(H)** $`F`$ is twice continuously differentiable on the convex hull $`\mathcal C`$ of $`\lbrace 0\rbrace \cup\lbrace G(y):y\in Y\rbrace`$, and $`\lvert s^\top\nabla^2F(g)\,s\rvert\le H_G\lVert s\rVert_\infty^2`$ for all $`g\in\mathcal C`$ and $`s\in\mathbb R^p`$.

**Warm-up.** A *moment warm-up* of length $`W`$ performs $`W`$ optimizer steps with learning rate exactly $`0`$, using gradients of the declared data. It leaves $`\theta`$ unchanged and advances $`m`$, $`v`$ and the counter. Every gradient absorbed into $`v`$, whether from warm-up or from a real update, is a *gradient evaluation*.

## Statement

**(W0)** **Lemma.** For every coordinate,

```math
\lvert\theta_i^+(y)-\tilde\theta_i^+(y)\rvert\le\eta\,u_i(y)\,\min\lbrace \kappa_i(y)/2,\;1\rbrace .
```

**(W1)** **Proposition (warm Adam is first-order affine in the consumed gradient).** Under (W), (L), (S) and (H), let $`c=F(0)`$ and $`b=\nabla F(0)`$. Both are independent of $`y`$. Then for every $`y\in Y`$,

```math
\big\lvert\psi(y)-c-b^\top G(y)\big\rvert\le\eta\sum_iL_iu_i(y)\min\lbrace \tfrac{\kappa_i(y)}2,1\rbrace +L_v\max_i\min\lbrace \kappa_i(y),\bar k\rbrace +\tfrac12H_G\lVert G(y)\rVert_\infty^2=:r(y).
```

Consequently, for any two message laws $`P_0,P_1`$ on $`Y`$,

```math
\big\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\big\rvert\le\min\Big\lbrace 1,\;\big\lvert b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)\big\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r\Big\rbrace .
```

In particular, laws with equal mean consumed gradient differ in response by at most $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$. If $`V`$ is differentiable in $`(\theta,m)`$, then

```math
b=(1-\beta_1)\Big[-\frac{\eta}{1-\beta_1^t}P\,\nabla_\theta V+\nabla_mV\Big]\Big|_{\tilde z^+(0)}.
```

So the first-order statistic is the mean preconditioned consumed gradient.

**(W2)** **Proposition (cold first-order separation).** Take a cold state ($`m=v=0`$, $`t=1`$), the ideal sign map ($`\epsilon\to0`$, node [N11](N11_adam_cold_start.md)), a scalar parameter and the immediate utility $`V(\theta)=\tfrac12+w\theta`$, restricted to a neighbourhood where $`V\in[0,1]`$.

- (a) *Equal means.* Let $`P_0(G=1)=\tfrac14`$, $`P_0(G=-\tfrac13)=\tfrac34`$ and $`P_1(G=0)=1`$. Both laws have mean $`0`$, and $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$.
- (b) *Equal moments to any fixed order.* For each $`N\ge1`$, let $`P_N,Q_N`$ be the laws [N07(c)](N07_affine_moment_order.md) on $`\lbrace i/N\rbrace`$ (node [N07](N07_affine_moment_order.md)), and $`G=q-Z`$ with $`q\in(0,1/N)`$. The two gradient laws agree on all moments of order below $`N`$, and $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$.

In both cases the separation is first order in $`\eta`$. At a warm state with the same utility, W1 bounds the gap of equal-mean laws by $`\eta\lvert w\rvert(\mathbb E_{P_0}u\min\lbrace \kappa/2,1\rbrace +\mathbb E_{P_1}u\min\lbrace \kappa/2,1\rbrace )`$.

**(W3)** **Corollary (cold window).**

- (a) *Mean-field warmth.* Suppose $`v`$ starts at $`0`$ and has absorbed $`s\ge1`$ gradient evaluations whose coordinate second moments are all $`\sigma_i^2>0`$. Then $`\mathbb E v_i=(1-\beta_2^s)\sigma_i^2`$. The mean-field warmth $`\bar\kappa_i(s):=(1-\beta_2)\sigma_i^2/(\beta_2\,\mathbb E v_i)`$ satisfies

  ```math
  \frac1{\beta_2\,s}\le\bar\kappa_i(s)\le\frac1{\beta_2^{\,s}\,s}.
  ```

  In particular $`\bar\kappa_i(s)\le e^{1/\beta_2}/s`$ for $`s\le(1-\beta_2)^{-1}`$. After $`W`$ warm-up evaluations, real update $`s`$ has $`W+s-1`$ prior evaluations.
- (b) *Accumulation.* Compare the reference and an alternative provider over $`R`$ real updates through the telescoping identity of [N05(a)–(c)](N05_continuation_response.md) (node [N05](N05_continuation_response.md)). Let $`\mathcal S_{\rm warm}\subseteq\lbrace 1,\dots,R\rbrace`$ be the updates at which, at every visited state, (W), (L), (S), (H) hold with constants indexed by $`s`$, the two laws have equal mean consumed gradient, and $`\max_i\kappa_i\le c/(W+s)`$ almost surely for some constant $`c>0`$, which is unrelated to $`c=F(0)`$ in W1. Then

  ```math
  \lvert L(0)-L(S)\rvert\le\lvert\lbrace s\notin\mathcal S_{\rm warm}\rbrace\rvert+\bar A\sum_{s\in\mathcal S_{\rm warm}}\min\Big\lbrace 1,\frac{c}{W+s}\Big\rbrace +\sum_{s\in\mathcal S_{\rm warm}}\bar B_s,
  ```

  where $`\bar A=\sup_s2\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)`$, $`\bar B_s=\sup H_{G,s}\lVert G\rVert_\infty^2`$, and $`\bar u_{i,s}`$ bounds $`u_i`$.
  - With $`W\ge1`$ warm-up evaluations, typically $`\mathcal S_{\rm warm}=\lbrace 1,\dots,R\rbrace`$, and the warmth-controlled sum is at most $`c\log(1+R/W)`$.
  - With $`W=0`$, the first real update starts from $`v=0`$ and violates (W). It is the cold step of node [N11](N11_adam_cold_start.md), and it contributes at most $`1`$; by W2 its contribution can be of first order in $`\eta`$. If $`\mathcal S_{\rm warm}=\lbrace 2,\dots,R\rbrace`$, the warmth-controlled sum is at most $`c\log R`$.

*What is exact and what is approximate.* Part (a) is exact for the *mean* of $`v`$. Part (b) is exact given its hypothesis $`\max_i\kappa_i\le c/(W+s)`$ on $`\mathcal S_{\rm warm}`$. The step from (a) to that hypothesis replaces the realized $`v_i`$ by its mean and $`G_i^2/\sigma_i^2`$ by a bounded constant. This step is an approximation: it needs concentration of the exponential moving average and stationary gradient scale. Check E8-W3 tests it on actual trajectories.

## Proof

**(W0)**

- *Case $`v_i>0`$.* Then $`\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\lvert n_i\rvert\,\lvert\tfrac1{a_i\sqrt{1+\kappa_i}+\epsilon}-\tfrac1{a_i+\epsilon}\rvert`$. Bringing the reciprocals to a common denominator,

  ```math
  \lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\,u_i\cdot\frac{a_i(\sqrt{1+\kappa_i}-1)}{a_i\sqrt{1+\kappa_i}+\epsilon}.
  ```

  Since $`\epsilon\ge0`$, the last factor is at most $`(\sqrt{1+\kappa_i}-1)/\sqrt{1+\kappa_i}=1-(1+\kappa_i)^{-1/2}`$. This is below $`1`$, and below $`\kappa_i/2`$ because $`\sqrt{1+\kappa}-1\le\kappa/2`$ and $`\sqrt{1+\kappa}\ge1`$. Hence the factor is at most $`\min\lbrace \kappa_i/2,1\rbrace`$.
- *Case $`v_i=0`$ with $`\epsilon>0`$.* Then $`a_i=0`$, $`\kappa_i=\infty`$ and $`u_i=\lvert n_i\rvert/\epsilon`$. Both reciprocals $`1/(d_i+\epsilon)`$ and $`1/\epsilon`$ lie in $`[0,1/\epsilon]`$, so their difference is at most $`1/\epsilon`$, and the left side is at most $`\eta u_i=\eta u_i\min\lbrace \infty,1\rbrace`$. $`\square`$

**Proof of (W1).** Fix $`y`$ and drop it from notation. Split

```math
\psi-c-b^\top G=\underbrace{\big[V(z^+)-V(\tilde\theta^+,m^+,v^+)\big]}_{(i)}+\underbrace{\big[V(\tilde\theta^+,m^+,v^+)-V(\tilde\theta^+,m^+,\beta_2v)\big]}_{(ii)}+\underbrace{\big[F(G)-F(0)-\nabla F(0)^\top G\big]}_{(iii)}.
```

The identity holds because $`V(\tilde\theta^+,m^+,\beta_2v)=V(\tilde z^+(G))=F(G)`$.

- *(i)* By (L) and (W0), $`\lvert(i)\rvert\le\sum_iL_i\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\sum_iL_iu_i\min\lbrace \kappa_i/2,1\rbrace`$.
- *(ii)* By (W), $`v^+=\beta_2v\odot(1+\kappa)`$ with finite $`\kappa\ge0`$. By (S) at $`(\tilde\theta^+,m^+)`$, $`\lvert(ii)\rvert\le L_v\max_i\min\lbrace \kappa_i,\bar k\rbrace`$.
- *(iii)* The segment $`\lbrace \tau G:\tau\in[0,1]\rbrace`$ lies in $`\mathcal C`$. Taylor's theorem with integral remainder gives

  ```math
  F(G)-F(0)-\nabla F(0)^\top G=\int_0^1(1-\tau)\,G^\top\nabla^2F(\tau G)\,G\,d\tau.
  ```

  By (H) the integrand is at most $`(1-\tau)H_G\lVert G\rVert_\infty^2`$ in absolute value. So $`\lvert(iii)\rvert\le\tfrac12H_G\lVert G\rVert_\infty^2`$.

Adding the three bounds proves the pointwise statement.

For the two-law statement, write $`\psi=c+b^\top G+e`$ with $`\lvert e\rvert\le r`$. Then

```math
\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi=b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)+\mathbb E_{P_0}e-\mathbb E_{P_1}e.
```

The constant $`c`$ cancels because both laws are probability measures; $`Y`$ is finite, so all expectations exist. The triangle inequality gives the bound, and $`\psi\in[0,1]`$ gives the cap at $`1`$.

For the formula for $`b`$, the chain rule applies with $`\partial\tilde\theta^+/\partial g=-\eta(1-\beta_1)P/(1-\beta_1^t)`$, $`\partial m^+/\partial g=(1-\beta_1)I`$, and the $`v`$-component independent of $`g`$. $`\square`$

**(W2)**

*(a)* By [N11(a)](N11_adam_cold_start.md) with $`h_0=\mathrm{sign}`$, $`\theta_1=\theta_0-\eta\mathrm{sign}(G)`$. Under $`P_0`$, $`\mathbb E\mathrm{sign}G=\tfrac14-\tfrac34=-\tfrac12`$, so $`\mathbb E\theta_1=\theta_0+\eta/2`$. Under $`P_1`$, $`\mathrm{sign}0=0`$, so $`\mathbb E\theta_1=\theta_0`$. The utility is affine, so the response gap is $`\lvert w\rvert\eta/2`$. The means agree: $`\tfrac14\cdot1+\tfrac34\cdot(-\tfrac13)=0`$.

*(b)* The law $`P_N`$ is supported on even $`i`$ and $`Q_N`$ on odd $`i`$. Their moments of order $`\ell<N`$ agree ([N07(c)](N07_affine_moment_order.md)). Hence the moments of the affine image $`G=q-Z`$ also agree below order $`N`$. With $`q\in(0,1/N)`$:

- Under $`P_N`$, $`G>0`$ exactly when $`i=0`$, which has probability $`\binom N0/2^{N-1}=2^{1-N}`$; otherwise $`Z\ge2/N>q`$ and $`G<0`$. So $`\mathbb E_{P_N}\mathrm{sign}G=2^{2-N}-1`$.
- Under $`Q_N`$, $`Z\ge1/N>q`$ always, so $`\mathbb E_{Q_N}\mathrm{sign}G=-1`$.

The expected first steps differ by $`\eta\,2^{2-N}`$, and the affine utility gives the gap $`\lvert w\rvert\eta2^{2-N}`$.

*Warm counterpart.* For $`V(\theta)=\tfrac12+w\theta`$ without suffix, $`F`$ is affine in $`g`$, so $`H_G=0`$. $`V`$ does not depend on $`(m,v)`$, so $`L_v=0`$, and $`L=\lvert w\rvert`$. With equal means, W1 gives the displayed bound. $`\square`$

**(W3)**

*(a)* $`v_i=(1-\beta_2)\sum_{r=1}^s\beta_2^{s-r}(G_i^{(r)})^2`$, so $`\mathbb E v_i=(1-\beta_2)\sigma_i^2\sum_{r=0}^{s-1}\beta_2^r=(1-\beta_2^s)\sigma_i^2`$. Hence $`\bar\kappa_i(s)=(1-\beta_2)/(\beta_2(1-\beta_2^s))`$. Since $`1-\beta^s=(1-\beta)\sum_{r=0}^{s-1}\beta^r`$ and $`\beta^{s-1}\le\beta^r\le1`$, we have $`(1-\beta)s\beta^{s-1}\le1-\beta^s\le(1-\beta)s`$. Substituting gives $`1/(\beta_2s)\le\bar\kappa_i\le1/(\beta_2^ss)`$. For the last claim, $`\log\beta\ge-(1-\beta)/\beta`$ gives $`\beta^{s}\ge\beta^{1/(1-\beta)}\ge e^{-1/\beta}`$ when $`s\le1/(1-\beta)`$.

*(b)* [N05(a)–(c)](N05_continuation_response.md) gives

```math
L(0)-L(S)=\sum_{s=1}^R\mathbb E_{d_s^S}\big[(K_s^0-K_s^S)V_{s+1}^0(Z_s)\big],
```

and at each visited state $`(K_s^0-K_s^S)V_{s+1}^0=\mathbb E_{P_0}\psi_s-\mathbb E_{P_1}\psi_s`$, a difference of two $`[0,1]`$-valued expectations.

- For $`s\notin\mathcal S_{\rm warm}`$, bound its absolute value by $`1`$.
- For $`s\in\mathcal S_{\rm warm}`$, W1 with equal means bounds it by $`\mathbb E_{P_0}r_s+\mathbb E_{P_1}r_s`$. Use $`\min\lbrace \kappa_i/2,1\rbrace \le\min\lbrace \kappa_{\max},1\rbrace`$, and $`\min\lbrace \kappa,\bar k\rbrace \le\max\lbrace 1,\bar k\rbrace \min\lbrace 1,\kappa\rbrace`$, which holds whether $`\kappa\le1`$ or $`\kappa>1`$. Then the first two terms of $`r_s`$ are at most $`\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)\min\lbrace 1,c/(W+s)\rbrace`$, and the curvature term is at most $`\tfrac12\bar B_s`$. Summing over the two laws gives the factor $`2`$.

Summing over $`s`$ proves the display.

For the sums: if $`W\ge1`$, then $`\sum_{s=1}^Rc/(W+s)\le c\int_W^{W+R}dx/x=c\log(1+R/W)`$. If $`W=0`$ and $`s`$ runs from $`2`$, then $`\sum_{s=2}^Rc/s\le c\int_1^Rdx/x=c\log R`$. $`\square`$

## Remarks

1. **The switch.** A feedback source must match a different statistic depending on the optimizer state.
   - At a cold Adam state, it must match the joint sign law of the consumed gradient, together with magnitudes retained in $`m`$ (node [N11](N11_adam_cold_start.md)).
   - At a warm Adam state ($`\kappa\ll1`$), it must match the mean preconditioned consumed gradient, up to $`O(\eta\,u\,\kappa+L_v\kappa+H_G\lVert G\rVert^2)`$.
   - Under SGD, it must match the mean consumed gradient up to $`O(\eta^2)`$ ([N10(c)](N10_update_composition.md)).
2. **Order of the residual.** For a smooth suffix, $`F`$ depends on $`g`$ through $`\tilde\theta^+`$, which is scaled by $`\eta`$, and through $`m^+`$, which enters later updates scaled by $`\eta`$. One therefore expects $`H_G=O(\eta^2)`$. This is an expectation about the suffix, not part of (H). The measured $`H_G`$ in check E8-W1 grows roughly like $`\eta^2`$ between $`\eta=0.01`$ and $`0.1`$.
3. **What warmth does not remove.** The curvature term $`\tfrac12H_G\lVert G\rVert_\infty^2`$ does not depend on $`\kappa`$. In (W3)(b) it accumulates linearly in $`R`$, while the warmth-controlled part accumulates logarithmically. Over long horizons the residual difference between equal-mean sources is therefore governed by curvature (trajectory nonlinearity), which SGD shares. Warm-up controls only the cold window.
4. **Messages versus gradients.** W1 is affine in the *consumed* gradient. When $`G(y)`$ is itself nonlinear in $`y`$, the response is not affine in the message. Examples are sequence-level teacher forcing, where the prefix changes the Jacobian, and active global-norm clipping.
5. **Locality.** W1 is a one-boundary statement. Its constants are properties of the continuation value. A certificate along training would also need the transport and coverage terms of [N14(b)](N14_frozen_audit_transport.md) (node [N14](N14_frozen_audit_transport.md)).
6. **Precedents.** The bias-corrected first step and the sign-descent view of Adam are standard: Kingma and Ba (2015), Balles and Hennig (2018), and Liu et al. (RAdam, ICLR 2020) on early-step variance. The contribution here is the information statement: which first-order statistic a feedback law must control, as a function of optimizer warmth, with an explicit residual.

## Dependencies and source

- Uses: [N05](N05_continuation_response.md) (response [N05](N05_continuation_response.md)/[N05](N05_continuation_response.md), telescoping identity [N05(a)](N05_continuation_response.md)), [N07](N07_affine_moment_order.md) (laws [N07(c)](N07_affine_moment_order.md)), [N10](N10_update_composition.md) (SGD expansion [N10(c)](N10_update_composition.md)), [N11](N11_adam_cold_start.md) (cold map [N11(a)](N11_adam_cold_start.md)).
- Used by: [N17](N17_ced_method.md).

This page makes four things explicit:
  - assumption (W) in W1;
  - the $`v_i=0`$ case of W0;
  - the moment-matched family in W2(b);
  - the exact mean-field bounds and the exact/approximate split in W3.

**Source:** [warm-Adam §2](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md#2-warm-one-step-approximation); [warm-Adam §3](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md#3-cold-start-separates-equal-mean-feedback-at-first-order); [warm-Adam §4](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md#4-how-warmth-evolves-and-a-free-way-to-warm-the-state) — Lemma W0; Props. W1–W2; Cor. W3. [Warm_Adam_Response_EN.md](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md) (Lemma W0, Props. W1–W2, Cor. W3).
