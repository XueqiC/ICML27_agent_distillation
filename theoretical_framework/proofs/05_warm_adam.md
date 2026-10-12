# §5 Warm Adam — Proofs

[← Framework](../README.md) · [Proofs](05_optimizer_state.md) · [Experiments](../experiments/05_warm_adam_and_learners.md) · [中文](05_warm_adam_ZH.md)

This page contains Definition 5.4 through Proposition 5.8 and uses the [original setting](05_optimizer_state.md#setting); see the [preceding page](05_optimizer_state.md) for the expansion and cold-start proofs, and [learner statistics](05_learner_statistics.md) for clipping and the learner catalog.

## Warm Adam

<a id="s5-4"></a>

**Definition 5.4 (Warmth and the frozen-denominator surrogate).** Consider an Adam update with counter $`t\ge1`$ at $`z=(\theta,m,v,t-1)`$, and assume $`v_i>0`$, or $`\epsilon>0`$ whenever some $`v_i=0`$.

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

<a id="s5-5"></a>

**Proposition 5.5 (The exact first-order statistic of Adam and the frozen-denominator error).** In the setting of Definition 5.4:

- **(a) Exact statistic.** The Adam increment is $`\Delta=(\Delta\theta,\Delta m,\Delta v,1)`$ with $`\Delta\theta=-\eta\lambda_{\rm wd}\theta-\eta\,n(G)/(d(G)+\epsilon)`$, $`\Delta m=(1-\beta_1)(G-m)`$ and $`\Delta v=(1-\beta_2)(G^{\odot2}-v)`$. The first-order statistic of Proposition 5.1 is the exact mean $`\mu=\mathbb E\,\Delta(z,G)`$. Its parameter block is a nonlinear functional of the law of $`G`$: if $`\beta_1<1`$ and $`v_i>0`$ or $`\epsilon>0`$ for some coordinate $`i`$, there are two finitely supported laws with equal $`\mathbb E\,G`$ and equal $`\mathbb E\,G^{\odot2}`$ whose parameter blocks $`\mu_\theta`$ differ.
- **(b) Frozen-denominator error.** For every coordinate and every $`G`$,

```math
\lvert\theta_i^+(G)-\tilde\theta_i^+(G)\rvert\le\eta\,u_i(G)\,\min\lbrace \kappa_i(G)/2,\;1\rbrace .
```

- **(c) Two approximations, kept apart.** The frozen-denominator mean increment $`\tilde\mu=\mathbb E\,\tilde\Delta(z,G)`$ is determined exactly by $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$, and $`\lvert\mu_{\theta,i}-\tilde\mu_{\theta,i}\rvert\le\eta\,\mathbb E[u_i\min\lbrace\kappa_i/2,1\rbrace]`$ while the $`m`$ and $`v`$ blocks agree exactly. The statement "warm Adam needs only $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$" therefore rests on two separate approximations:
  - the first-order approximation of Proposition 5.1, whose remainder is controlled by the second moments of the increment, with the block structure described in Remark 5.2;
  - the frozen-denominator approximation, whose error is controlled by (b) and is small in the warmth.

*Proof of (a).* The increment is the definition of AdamW. For the second claim fix the coordinate $`i`$ and let $`G=g\,e_i`$ with scalar $`g`$, all other coordinates zero. The parameter block of coordinate $`i`$ is then $`\mathbb E f(g)`$ with

```math
f(g)=-\eta\,\frac{\big(\beta_1m_i+(1-\beta_1)g\big)/(1-\beta_1^{\,t})}{\sqrt{\big(\beta_2v_i+(1-\beta_2)g^2\big)/(1-\beta_2^{\,t})}+\epsilon}
```

- $`f`$ is bounded on $`\mathbb R`$: the denominator is bounded below by a positive constant when $`v_i>0`$ or $`\epsilon>0`$, and as $`g\to\pm\infty`$, $`f(g)`$ tends to $`\mp\eta(1-\beta_1)\sqrt{1-\beta_2^{\,t}}/\big((1-\beta_1^{\,t})\sqrt{1-\beta_2}\big)`$. The two limits differ because $`\beta_1<1`$, so $`f`$ is not constant.
- A polynomial of degree at most two that is bounded on $`\mathbb R`$ is constant. Hence $`f`$ is not such a polynomial, and there are four points $`g_1<g_2<g_3<g_4`$ on which $`f`$ does not agree with any polynomial of degree at most two (the quadratic through three points of the graph misses some fourth point).
- The linear map $`w\mapsto(\sum_kw_k,\sum_kw_kg_k,\sum_kw_kg_k^2)`$ from $`\mathbb R^4`$ to $`\mathbb R^3`$ has a nonzero kernel vector $`w^*`$. Its row space is the restriction of $`\mathrm{span}\lbrace1,g,g^2\rbrace`$ to the four points, so $`\sum_kw^*_kf(g_k)\ne0`$, for otherwise the restriction of $`f`$ would lie in that span.
- For small $`\tau>0`$, $`P=\tfrac14\mathbf 1+\tau w^*`$ and $`Q=\tfrac14\mathbf 1-\tau w^*`$ are probability vectors on the four points (because $`\sum_kw^*_k=0`$). They have equal $`\mathbb E\,g`$ and $`\mathbb E\,g^2`$, hence equal $`\mathbb E\,G`$ and $`\mathbb E\,G^{\odot2}`$, while $`\mathbb E_Pf-\mathbb E_Qf=2\tau\sum_kw^*_kf(g_k)\ne0`$. $`\square`$

*Proof of (b).*

- *Case $`v_i>0`$.* Then $`\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\lvert n_i\rvert\,\big\lvert\tfrac1{a_i\sqrt{1+\kappa_i}+\epsilon}-\tfrac1{a_i+\epsilon}\big\rvert`$. Bringing the reciprocals to a common denominator gives the identity below. Since $`\epsilon\ge0`$, the last factor is at most $`(\sqrt{1+\kappa_i}-1)/\sqrt{1+\kappa_i}=1-(1+\kappa_i)^{-1/2}`$. This is below $`1`$, and below $`\kappa_i/2`$ because $`\sqrt{1+\kappa}-1\le\kappa/2`$ and $`\sqrt{1+\kappa}\ge1`$. Hence the factor is at most $`\min\lbrace \kappa_i/2,1\rbrace`$.

```math
\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\,u_i\cdot\frac{a_i(\sqrt{1+\kappa_i}-1)}{a_i\sqrt{1+\kappa_i}+\epsilon}.
```

- *Case $`v_i=0`$ with $`\epsilon>0`$.* Then $`a_i=0`$, $`\kappa_i=\infty`$ and $`u_i=\lvert n_i\rvert/\epsilon`$. Both reciprocals $`1/(d_i+\epsilon)`$ and $`1/\epsilon`$ lie in $`[0,1/\epsilon]`$, so their difference is at most $`1/\epsilon`$, and the left side is at most $`\eta u_i=\eta u_i\min\lbrace \infty,1\rbrace`$. $`\square`$

*Proof of (c).* By Definition 5.4, $`\tilde\theta^+-\theta=-\eta\lambda_{\rm wd}\theta-\eta\Phi\beta_1m/(1-\beta_1^{\,t})-\eta\Phi(1-\beta_1)G/(1-\beta_1^{\,t})`$ and $`m^+-m=(1-\beta_1)(G-m)`$ are affine in $`G`$ with coefficients fixed by the state, and $`v^+-v=(1-\beta_2)(G^{\odot2}-v)`$ is affine in $`G^{\odot2}`$. Taking expectations, $`\tilde\mu`$ is an affine function of $`(\mathbb E\,G,\mathbb E\,G^{\odot2})`$. The $`m`$ and $`v`$ blocks of $`\Delta`$ and $`\tilde\Delta`$ coincide, and the $`\theta`$ blocks differ by $`\theta^+-\tilde\theta^+`$, whose expectation is bounded coordinatewise by (b). $`\square`$

<a id="s5-6"></a>

**Proposition 5.6 (Warm Adam is first-order affine in the consumed gradient).** Fix a round $`t\ge1`$, a state $`z=(\theta,m,v,t-1)`$ and a finite message set $`Y`$ with consumed gradients $`G(y)`$. Let $`V:\mathcal Z\to[0,1]`$ be the continuation value after the update (the declared reference tail followed by deployment, with tail randomness integrated), $`\psi(y)=V(z^+(y))`$ the one-boundary response, and $`F(g)=V(\tilde z^+(g))`$. Assume, with all states generated from $`z`$ by messages in $`Y`$ and the counter fixed at $`t`$:

- **(W)** $`v_i>0`$ for every coordinate (warm state);
- **(L)** $`\lvert V(\vartheta,m,v)-V(\vartheta',m,v)\rvert\le\sum_iL_i\lvert\vartheta_i-\vartheta_i'\rvert`$ on the pairs $`\big(\theta^+(y),\tilde\theta^+(y)\big)`$ at $`(m,v)=(m^+(y),v^+(y))`$;
- **$`(\mathrm S_v)`$** there are $`L_v\ge0`$ and a cap $`\bar k\in(0,\infty]`$ such that, at $`(\vartheta,m)=(\tilde\theta^+(y),m^+(y))`$, $`\lvert V(\vartheta,m,\beta_2v\odot(1+k))-V(\vartheta,m,\beta_2v)\rvert\le L_v\max_i\min\lbrace k_i,\bar k\rbrace`$ for $`k=\kappa(y)`$ (since $`V\in[0,1]`$ this always holds with $`L_v\bar k\ge1`$; the content lies in small $`L_v`$);
- **(H)** $`F`$ is twice continuously differentiable on the convex hull $`\mathcal C`$ of $`\lbrace 0\rbrace \cup\lbrace G(y):y\in Y\rbrace`$, and $`\lvert s^\top\nabla^2F(g)\,s\rvert\le H_G\lVert s\rVert_\infty^2`$ for all $`g\in\mathcal C`$ and $`s\in\mathbb R^p`$.

Let $`\psi_0=F(0)`$ and $`b=\nabla F(0)`$; both are independent of $`y`$. Then for every $`y\in Y`$

```math
\big\lvert\psi(y)-\psi_0-b^\top G(y)\big\rvert\le\eta\sum_iL_iu_i(y)\min\lbrace \tfrac{\kappa_i(y)}2,1\rbrace +L_v\max_i\min\lbrace \kappa_i(y),\bar k\rbrace +\tfrac12H_G\lVert G(y)\rVert_\infty^2=:r(y),
```

and for any two message laws $`P_0,P_1`$ on $`Y`$

```math
\big\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\big\rvert\le\min\Big\lbrace 1,\;\big\lvert b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)\big\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r\Big\rbrace .
```

- In particular, laws with equal mean consumed gradient differ in response by at most $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$.
- If $`V`$ is differentiable in $`(\theta,m)`$, then the first-order statistic is the mean preconditioned consumed gradient, with weight

```math
b=(1-\beta_1)\Big[-\frac{\eta}{1-\beta_1^{\,t}}\,\Phi\,\nabla_\theta V+\nabla_mV\Big]\Big|_{\tilde z^+(0)}.
```

*Proof.* Fix $`y`$ and drop it from the notation. Since $`V(\tilde\theta^+,m^+,\beta_2v)=V(\tilde z^+(G))=F(G)`$,

```math
\psi-\psi_0-b^\top G=\underbrace{\big[V(z^+)-V(\tilde\theta^+,m^+,v^+)\big]}_{(i)}+\underbrace{\big[V(\tilde\theta^+,m^+,v^+)-V(\tilde\theta^+,m^+,\beta_2v)\big]}_{(ii)}+\underbrace{\big[F(G)-F(0)-\nabla F(0)^\top G\big]}_{(iii)}.
```

- *(i)* By (L) and Proposition 5.5(b), $`\lvert(i)\rvert\le\sum_iL_i\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\sum_iL_iu_i\min\lbrace \kappa_i/2,1\rbrace`$.
- *(ii)* By (W), $`v^+=\beta_2v\odot(1+\kappa)`$ with finite $`\kappa\ge0`$. By $`(\mathrm S_v)`$ at $`(\tilde\theta^+,m^+)`$, $`\lvert(ii)\rvert\le L_v\max_i\min\lbrace \kappa_i,\bar k\rbrace`$.
- *(iii)* The segment $`\lbrace \tau G:\tau\in[0,1]\rbrace`$ lies in $`\mathcal C`$. Taylor's theorem with integral remainder gives $`F(G)-F(0)-\nabla F(0)^\top G=\int_0^1(1-\tau)\,G^\top\nabla^2F(\tau G)\,G\,d\tau`$. By (H) the integrand is at most $`(1-\tau)H_G\lVert G\rVert_\infty^2`$ in absolute value, so $`\lvert(iii)\rvert\le\tfrac12H_G\lVert G\rVert_\infty^2`$.

Adding the three bounds proves the pointwise statement. For two laws, write $`\psi=\psi_0+b^\top G+e`$ with $`\lvert e\rvert\le r`$. Then $`\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi=b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)+\mathbb E_{P_0}e-\mathbb E_{P_1}e`$; the constant $`\psi_0`$ cancels because both laws are probability measures, and $`Y`$ is finite, so all expectations exist. The triangle inequality gives the bound, and $`\psi\in[0,1]`$ gives the cap at $`1`$. For $`b`$, the chain rule applies with $`\partial\tilde\theta^+/\partial g=-\eta(1-\beta_1)\Phi/(1-\beta_1^{\,t})`$, $`\partial m^+/\partial g=(1-\beta_1)I`$, and the $`v`$-component of $`\tilde z^+`$ independent of $`g`$. $`\square`$

<a id="s5-7"></a>

**Proposition 5.7 (Cold steps separate equal-mean feedback at first order).** Take a cold state ($`m=v=0`$, $`t=1`$), the ideal sign map $`h_0`$ (Proposition 5.3), a scalar parameter and the immediate utility $`V(\theta)=\tfrac12+w\theta`$, restricted to a neighbourhood where $`V\in[0,1]`$.

- **(a) Equal means.** Let $`P_0(G=1)=\tfrac14`$, $`P_0(G=-\tfrac13)=\tfrac34`$ and $`P_1(G=0)=1`$. Both laws have mean $`0`$, and $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$.
- **(b) Equal moments to any fixed order.** For $`N\ge1`$, let $`P_N(Z=i/N)=\binom Ni/2^{N-1}`$ for even $`i`$ and $`Q_N(Z=i/N)=\binom Ni/2^{N-1}`$ for odd $`i`$, $`0\le i\le N`$ (the laws of [Theorem 4.4(c)](04_complexity_of_sufficiency.md#s4-4)), and $`G=q-Z`$ with $`q\in(0,1/N)`$. The two gradient laws agree on all moments of order below $`N`$, and $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$.
- In both cases the separation is first order in $`\eta`$. At a warm state with the same utility, Proposition 5.6 bounds the gap of equal-mean laws by $`\eta\lvert w\rvert\big(\mathbb E_{P_0}[u\min\lbrace \kappa/2,1\rbrace ]+\mathbb E_{P_1}[u\min\lbrace \kappa/2,1\rbrace ]\big)`$.

*Proof.*

- *(a)* By Proposition 5.3(a) with $`h_0=\mathrm{sign}`$, $`\theta_1=\theta_0-\eta\,\mathrm{sign}(G)`$. Under $`P_0`$, $`\mathbb E\,\mathrm{sign}\,G=\tfrac14-\tfrac34=-\tfrac12`$, so $`\mathbb E\theta_1=\theta_0+\eta/2`$. Under $`P_1`$, $`\mathrm{sign}\,0=0`$, so $`\mathbb E\theta_1=\theta_0`$. The utility is affine, so the response gap is $`\lvert w\rvert\eta/2`$. The means agree: $`\tfrac14\cdot1+\tfrac34\cdot(-\tfrac13)=0`$.
- *(b)* Both $`P_N`$ and $`Q_N`$ are probability laws because $`\sum_{i\ \rm even}\binom Ni=\sum_{i\ \rm odd}\binom Ni=2^{N-1}`$. For $`\ell<N`$, $`\sum_i(-1)^i\binom Ni\,i^\ell=0`$, so their moments of order below $`N`$ agree, and so do those of the affine image $`G=q-Z`$. With $`q\in(0,1/N)`$: under $`P_N`$, $`G>0`$ exactly when $`i=0`$, which has probability $`2^{1-N}`$, and otherwise $`Z\ge2/N>q`$ and $`G<0`$, so $`\mathbb E_{P_N}\mathrm{sign}\,G=2^{2-N}-1`$. Under $`Q_N`$, $`Z\ge1/N>q`$ always, so $`\mathbb E_{Q_N}\mathrm{sign}\,G=-1`$. The expected first steps differ by $`\eta\,2^{2-N}`$, and the affine utility gives the gap $`\lvert w\rvert\eta\,2^{2-N}`$.
- *Warm counterpart.* For $`V(\theta)=\tfrac12+w\theta`$ without a tail, $`F`$ is affine in $`g`$, so $`H_G=0`$; $`V`$ does not depend on $`(m,v)`$, so $`L_v=0`$; and $`L=\lvert w\rvert`$. With equal means, Proposition 5.6 gives the displayed bound. $`\square`$

<a id="s5-8"></a>

**Proposition 5.8 (Evolution of warmth).** Let the second moment start at $`v_0=0`$ and absorb gradient evaluations $`G_1,G_2,\dots`$ by $`v_s=\beta_2v_{s-1}+(1-\beta_2)G_s^{\odot2}`$, coordinatewise. Write $`\gamma=1-\beta_2`$.

**(a) Stationary second moment: exact mean-field warmth.** Suppose $`\mathbb E\,G_{k,i}^2=\sigma_i^2>0`$ for all $`k`$. Define the mean-field warmth $`\bar\kappa_i(s)=(1-\beta_2)\sigma_i^2/(\beta_2\,\mathbb E\,v_{s,i})`$. Then

```math
\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2^{\,s}),\qquad \bar\kappa_i(s)=\frac{1-\beta_2}{\beta_2(1-\beta_2^{\,s})},\qquad \frac{\max\lbrace 1/s,\;1-\beta_2\rbrace}{\beta_2}\le\bar\kappa_i(s)\le\frac{1/s+(1-\beta_2)}{\beta_2}.
```

- $`\bar\kappa_i(s)`$ is decreasing in $`s`$ and converges to the plateau $`\kappa_\infty=(1-\beta_2)/\beta_2>0`$; it does not tend to zero.
- In the early phase it behaves like $`1/(\beta_2s)`$: precisely, $`1\le\beta_2s\,\bar\kappa_i(s)\le1+s(1-\beta_2)`$, so the approximation $`\bar\kappa_i(s)\approx1/(\beta_2s)`$ has relative error at most $`s(1-\beta_2)`$ and is accurate only when $`s(1-\beta_2)\ll1`$.
- If the gradient is constant, $`G_k=g`$, the formula holds for the warmth $`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ itself, not only for its mean-field version.

**(b) Bias correction does not change warmth.** With bias-corrected denominators $`d_i=\sqrt{v_i^+/(1-\beta_2^{\,t})}`$ and $`a_i=\sqrt{\beta_2v_i/(1-\beta_2^{\,t})}`$, the ratio $`d_i/a_i=\sqrt{1+\kappa_i}`$ is the same as without bias correction. Warmth, and the bound of Proposition 5.5(b), are therefore unaffected by bias correction.

**(c) Accumulation along a run.** Compare a reference protocol $`0`$ and an alternative protocol $`S`$ over $`R`$ real updates $`s=1,\dots,R`$ preceded by $`W`$ warm-up evaluations, through the telescoping identity of [Theorem 3.2(a)](03_learning_consequence.md#s3-2). Let $`\mathcal S_{\rm warm}\subseteq\lbrace 1,\dots,R\rbrace`$ be the updates at which, at every visited state, the hypotheses of Proposition 5.6 hold with constants indexed by $`s`$, the two laws have equal mean consumed gradient, and, almost surely, the condition below holds for some constant $`c>0`$.

```math
\max_i\kappa_i\le c\Big(\frac1{W+s}+(1-\beta_2)\Big).
```

Then

```math
\lvert L(0)-L(S)\rvert\le\big\lvert\lbrace s\le R:s\notin\mathcal S_{\rm warm}\rbrace\big\rvert+\bar A\sum_{s\in\mathcal S_{\rm warm}}\min\Big\lbrace 1,\frac{c}{W+s}+c(1-\beta_2)\Big\rbrace +\sum_{s\in\mathcal S_{\rm warm}}\bar B_s,
```

where $`\bar A=\sup_s2\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)`$, $`\bar u_{i,s}`$ bounds $`u_i`$ at update $`s`$, and $`\bar B_s=\sup H_{G,s}\lVert G\rVert_\infty^2`$. For $`W\ge1`$,

```math
\sum_{s=1}^{R}\min\Big\lbrace1,\;\frac{c}{W+s}+c(1-\beta_2)\Big\rbrace\le c\log\Big(1+\frac RW\Big)+c(1-\beta_2)R .
```

- The warmth-controlled contribution therefore has a per-update floor of order $`c(1-\beta_2)`$ that does not vanish as training lengthens. The logarithmic term dominates only when $`(1-\beta_2)R\lesssim\log(1+R/W)`$.
- With $`W=0`$ the first real update starts from $`v=0`$ and violates (W): it is the cold step of Proposition 5.3, charged at most $`1`$, and by Proposition 5.7 its contribution can be of first order in $`\eta`$. If $`\mathcal S_{\rm warm}=\lbrace 2,\dots,R\rbrace`$, the warmth-controlled sum is at most $`c\log R+c(1-\beta_2)(R-1)`$.

**(d) Warm-up, and when the hypothesis of (c) holds.** A *moment warm-up* of length $`W`$ performs $`W`$ optimizer steps with learning rate exactly $`0`$ on gradients of the declared data. It leaves $`\theta`$ unchanged and advances $`m`$, $`v`$ and the counter by the same recursions as real steps; every gradient absorbed into $`v`$, from warm-up or from a real update, counts as one evaluation. Real update $`s`$ therefore acts on a second moment that has absorbed $`W+s-1`$ evaluations. If the realized $`v`$ is replaced by its mean (the mean-field approximation) and the gradients satisfy $`G_{k,i}^2\le\bar\gamma\,\sigma_i^2`$, then for $`W+s\ge2`$ the hypothesis of (c) holds with $`c=2\bar\gamma/\beta_2`$. The replacement of $`v`$ by its mean is an approximation; it needs concentration of the exponential moving average and a stationary gradient scale.

*Proof of (a).* Unrolling the recursion, $`v_{s,i}=(1-\beta_2)\sum_{r=1}^s\beta_2^{\,s-r}G_{r,i}^2`$, so $`\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2)\sum_{j<s}\beta_2^{\,j}=\sigma_i^2(1-\beta_2^{\,s})`$. Substituting into the definition gives $`\bar\kappa_i(s)`$. For the bounds:

- By Bernoulli's inequality, $`1-(1-\gamma)^s\le s\gamma`$, so $`\bar\kappa\ge1/(\beta_2s)`$. Since $`1-(1-\gamma)^s\le1`$, also $`\bar\kappa\ge\gamma/\beta_2`$.
- The upper bound is equivalent to $`s\gamma\le(1+s\gamma)\big(1-(1-\gamma)^s\big)`$, that is, to $`(1-\gamma)^s(1+s\gamma)\le1`$. This follows from $`(1-\gamma)^s\le e^{-s\gamma}`$ and $`e^{-y}(1+y)\le1`$ for $`y\ge0`$.
- Monotonicity holds because $`1-\beta_2^{\,s}`$ increases in $`s`$, and the limit is immediate. Multiplying the two-sided bound by $`\beta_2s`$ gives $`1\le\beta_2s\bar\kappa\le1+s\gamma`$.
- For a constant gradient, $`v_{s,i}=g_i^2(1-\beta_2^{\,s})`$ deterministically, and $`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ equals the formula. $`\square`$

*Proof of (b).* $`d_i^2/a_i^2=v_i^+/(\beta_2v_i)=1+\kappa_i`$: the factor $`1/(1-\beta_2^{\,t})`$ divides numerator and denominator alike. $`\square`$

*Proof of (c).* [Theorem 3.2(a)](03_learning_consequence.md#s3-2) gives

```math
L(0)-L(S)=\sum_{s=1}^R\mathbb E_{d_s^S}\big[(K_s^0-K_s^S)V_{s+1}^0(Z_s)\big],
```

and at each visited state $`(K_s^0-K_s^S)V_{s+1}^0=\mathbb E_{P_0}\psi_s-\mathbb E_{P_1}\psi_s`$, a difference of two $`[0,1]`$-valued expectations.

- For $`s\notin\mathcal S_{\rm warm}`$, bound its absolute value by $`1`$.
- For $`s\in\mathcal S_{\rm warm}`$, Proposition 5.6 with equal means bounds it by $`\mathbb E_{P_0}r_s+\mathbb E_{P_1}r_s`$. Use $`\min\lbrace \kappa_i/2,1\rbrace \le\min\lbrace \kappa_{\max},1\rbrace`$ and $`\min\lbrace \kappa,\bar k\rbrace \le\max\lbrace 1,\bar k\rbrace \min\lbrace 1,\kappa\rbrace`$, which holds whether $`\kappa\le1`$ or $`\kappa>1`$. With the hypothesis on $`\kappa_{\max}`$, the first two terms of $`r_s`$ are at most $`\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)\min\lbrace 1,c/(W+s)+c(1-\beta_2)\rbrace`$, and the curvature term is at most $`\tfrac12\bar B_s`$. Summing over the two laws gives the factor $`2`$.

Summing over $`s`$ proves the first display. For the second, $`\min\lbrace1,a+b\rbrace\le\min\lbrace1,a\rbrace+b`$ for $`a,b\ge0`$, so the sum is at most $`\sum_{s=1}^R c/(W+s)+c(1-\beta_2)R`$; and $`c/(W+s)\le\int_{s-1}^{s}c\,dz/(W+z)`$, so $`\sum_{s=1}^Rc/(W+s)\le c\log(1+R/W)`$ for $`W\ge1`$. For $`W=0`$ and $`s`$ running from $`2`$, $`\sum_{s=2}^Rc/s\le c\log R`$ in the same way. $`\square`$

*Proof of (d).* With learning rate $`0`$ the parameter update is $`\theta^+=(1-0\cdot\lambda_{\rm wd})\theta-0=\theta`$, while $`m`$, $`v`$ and the counter follow the same recursions as in a real step. Hence after $`W`$ warm-up evaluations and $`s-1`$ real updates, $`v`$ has absorbed $`W+s-1`$ evaluations. Under the mean-field replacement, $`v_i=\sigma_i^2(1-\beta_2^{\,k})`$ with $`k=W+s-1\ge1`$, so $`\kappa_i=(G_i^2/\sigma_i^2)\,\bar\kappa_i(k)\le\bar\gamma\,(1/k+1-\beta_2)/\beta_2`$ by (a). Since $`k\ge(W+s)/2`$ when $`W+s\ge2`$, $`1/k\le2/(W+s)`$, and therefore $`\kappa_i\le(2\bar\gamma/\beta_2)\big(1/(W+s)+(1-\beta_2)\big)`$. $`\square`$

**Discussion of Propositions 5.5–5.8.**

- *The switch.* A feedback source must match a different statistic depending on the optimizer state.
  - At a cold Adam state, it must match the joint sign law of the consumed gradient, together with the magnitudes retained in $`m`$ and $`v`$ (Proposition 5.3).
  - At a warm Adam state, the exact statistic is the mean Adam increment; to first order in the warmth it is the mean preconditioned consumed gradient together with $`\mathbb E\,G^{\odot2}`$ in the $`v`$ block (Propositions 5.5–5.6).
  - Under SGD, it must match the mean consumed gradient up to $`O(\eta^2)`$ (the local SGD expansion of [Theorem 4.7](04_rates_and_composition.md#s4-7)).
- *The warmth floor.* Under a fixed $`\beta_2`$ the warmth of a typical gradient stays near $`(1-\beta_2)/\beta_2`$ for the rest of training (Proposition 5.8(a)). The frozen-denominator description is therefore accurate only to that relative order, at every step, and the accumulated non-affine contribution has a term linear in $`R`$ (Proposition 5.8(c)).
- *Order of the curvature term.* For a smooth tail, $`F`$ depends on $`g`$ through $`\tilde\theta^+`$, which is scaled by $`\eta`$, and through $`m^+`$, which enters later updates scaled by $`\eta`$. One therefore expects $`H_G=O(\eta^2)`$. This is an expectation about the tail, not part of (H).
- *What warmth does not remove.* The curvature term $`\tfrac12H_G\lVert G\rVert_\infty^2`$ does not depend on $`\kappa`$ and accumulates linearly in $`R`$. Over long horizons the residual difference between equal-mean sources is governed by curvature, which SGD shares. Warm-up controls only the cold window.
- *Messages versus gradients.* Proposition 5.6 is affine in the *consumed* gradient. When $`G(y)`$ is itself nonlinear in $`y`$, the response is not affine in the message. Examples are sequence-level teacher forcing, where the prefix changes the Jacobian, and active global-norm clipping.
- *Locality.* Proposition 5.6 is a one-boundary statement whose constants are properties of the continuation value. A certificate along training also needs the transport and coverage terms of [Theorem 8.2](08_verification.md#s8-2).
