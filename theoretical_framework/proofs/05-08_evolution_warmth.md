# Proposition 5.8 · Evolution of warmth

*Proof*

[← Framework](../README.md) · [Statement](../statements/05-08_evolution_warmth.md) · [Proof](../proofs/05-08_evolution_warmth.md) · [Experiments](../experiments/05-08_evolution_warmth.md) · [中文](05-08_evolution_warmth_ZH.md)

**What is proved here.** (a) The exact mean-field warmth and its bounds; (b) invariance under bias correction; (c) the accumulation bound through the telescoping identity; (d) the warm-up equivalence and the sufficient condition for (c).

**Setting used on this page.**

The objects below are restated so that this page can be read on its own; the general notation is that of [Definitions 1.3–1.4](../README.md#s1-3).

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

**Statement ([Proposition 5.8](../statements/05-08_evolution_warmth.md#s5-8)).** Let the second moment start at $`v_0=0`$ and absorb gradient evaluations $`G_1,G_2,\dots`$ by $`v_s=\beta_2v_{s-1}+(1-\beta_2)G_s^{\odot2}`$, coordinatewise. Write $`\gamma=1-\beta_2`$.

**(a) Stationary second moment: exact mean-field warmth.** Suppose $`\mathbb E\,G_{k,i}^2=\sigma_i^2>0`$ for all $`k`$. Define the mean-field warmth $`\bar\kappa_i(s)=(1-\beta_2)\sigma_i^2/(\beta_2\,\mathbb E\,v_{s,i})`$. Then

```math
\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2^{\,s}),\qquad \bar\kappa_i(s)=\frac{1-\beta_2}{\beta_2(1-\beta_2^{\,s})},\qquad \frac{\max\lbrace 1/s,\;1-\beta_2\rbrace}{\beta_2}\le\bar\kappa_i(s)\le\frac{1/s+(1-\beta_2)}{\beta_2}.
```

- $`\bar\kappa_i(s)`$ is decreasing in $`s`$ and converges to the plateau $`\kappa_\infty=(1-\beta_2)/\beta_2>0`$; it does not tend to zero.
- In the early phase it behaves like $`1/(\beta_2s)`$: precisely, $`1\le\beta_2s\,\bar\kappa_i(s)\le1+s(1-\beta_2)`$, so the approximation $`\bar\kappa_i(s)\approx1/(\beta_2s)`$ has relative error at most $`s(1-\beta_2)`$ and is accurate only when $`s(1-\beta_2)\ll1`$.
- If the gradient is constant, $`G_k=g`$, the formula holds for the warmth $`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ itself, not only for its mean-field version.

**(b) Bias correction does not change warmth.** With bias-corrected denominators $`d_i=\sqrt{v_i^+/(1-\beta_2^{\,t})}`$ and $`a_i=\sqrt{\beta_2v_i/(1-\beta_2^{\,t})}`$, the ratio $`d_i/a_i=\sqrt{1+\kappa_i}`$ is the same as without bias correction. Warmth, and the bound of Proposition 5.5(b), are therefore unaffected by bias correction.

**(c) Accumulation along a run.** Compare a reference protocol $`0`$ and an alternative protocol $`S`$ over $`R`$ real updates $`s=1,\dots,R`$ preceded by $`W`$ warm-up evaluations, through the telescoping identity of [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization.md#s3-2). Let $`\mathcal S_{\rm warm}\subseteq\lbrace 1,\dots,R\rbrace`$ be the updates at which, at every visited state, the hypotheses of Proposition 5.6 hold with constants indexed by $`s`$, the two laws have equal mean consumed gradient, and, almost surely, the condition below holds for some constant $`c>0`$.

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

*Proof of (c).* [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization.md#s3-2) gives

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
  - Under SGD, it must match the mean consumed gradient up to $`O(\eta^2)`$ (the local SGD expansion of [Theorem 4.7](../statements/04-07_update_composition_amplifies_information_requirement.md#s4-7)).
- *The warmth floor.* Under a fixed $`\beta_2`$ the warmth of a typical gradient stays near $`(1-\beta_2)/\beta_2`$ for the rest of training (Proposition 5.8(a)). The frozen-denominator description is therefore accurate only to that relative order, at every step, and the accumulated non-affine contribution has a term linear in $`R`$ (Proposition 5.8(c)).
- *Order of the curvature term.* For a smooth tail, $`F`$ depends on $`g`$ through $`\tilde\theta^+`$, which is scaled by $`\eta`$, and through $`m^+`$, which enters later updates scaled by $`\eta`$. One therefore expects $`H_G=O(\eta^2)`$. This is an expectation about the tail, not part of (H).
- *What warmth does not remove.* The curvature term $`\tfrac12H_G\lVert G\rVert_\infty^2`$ does not depend on $`\kappa`$ and accumulates linearly in $`R`$. Over long horizons the residual difference between equal-mean sources is governed by curvature, which SGD shares. Warm-up controls only the cold window.
- *Messages versus gradients.* Proposition 5.6 is affine in the *consumed* gradient. When $`G(y)`$ is itself nonlinear in $`y`$, the response is not affine in the message. Examples are sequence-level teacher forcing, where the prefix changes the Jacobian, and active global-norm clipping.
- *Locality.* Proposition 5.6 is a one-boundary statement whose constants are properties of the continuation value. A certificate along training also needs the transport and coverage terms of [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2).
