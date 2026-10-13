# Corollary 7.2 · Stationary amplitude law

*Proof*

[← Framework](../README.md) · [Statement](../statements/07-02_stationary_amplitude_law.md) · [Proof](07-02_stationary_amplitude_law.md) · [Experiments](../experiments/07-02_stationary_amplitude_law.md) · [中文](07-02_stationary_amplitude_law_ZH.md)

**What is proved here.** In a warm, stationary, linear-response regime of Adam, the change of the final outcome caused by one message of amplitude $`\lambda`$ (in units of the running RMS) is an explicit finite sum. For small increments it is a concave quadratic in $`\lambda`$ with an explicit error bound, an optimum $`\lambda^\star`$ and a sign change near $`2\lambda^\star`$; for very large amplitudes it has an explicit limit whose sign decides whether huge messages hurt. The proof substitutes the moments of the tail gradients into the Adam recursion, sums the resulting steps, and bounds the Taylor remainders of $`(1+x)^{-1/2}`$.

**Setting used on this page.**

- *Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) without bias correction in one coordinate $`i`$*: $`m'=\beta_1m+(1-\beta_1)g`$, $`v'=\beta_2v+(1-\beta_2)g^2`$, parameter step $`-\eta m'/\sqrt{v'}`$, with $`\epsilon=0`$ and no weight decay.
- *Message and tail.* A message with consumed gradient $`g_i`$ at round $`t`$, followed by $`T=R-1-t`$ tail updates with gradients $`g_{k,i}`$ that do not depend on $`\theta`$, are stationary with mean $`\bar g_i\ne0`$ and mean square $`\bar v_i>0`$. The final score $`F`$ is affine in $`\theta_i`$, with derivative $`(\nabla F)_i`$.
- *Warm state.* $`m_{t,i}=\bar g_i`$, $`v_{t,i}=\bar v_i`$ when the message arrives.
- *Normalized quantities.* $`\alpha_i=-(\nabla F)_i\,\mathrm{sign}(\bar g_i)`$; $`\lambda=g_i\,\mathrm{sign}(\bar g_i)/\sqrt{\bar v_i}`$; $`\rho_i=\lvert\bar g_i\rvert/\sqrt{\bar v_i}\in(0,1]`$; $`b_j=(1-\beta_1)\beta_1^{\,j}`$, $`c_j=(1-\beta_2)\beta_2^{\,j}`$ for $`j=0,\dots,T`$; $`\omega_T=1-\beta_2^{\,T+1}=\sum_{j\le T}c_j`$.
- *Mean-field response* $`\Delta F_i(\lambda)`$: the change of $`F`$ when a message of amplitude $`\lambda`$ replaces a zero-gradient message, with each step $`m_k/\sqrt{v_k}`$ replaced by $`\mathbb Em_k/\sqrt{\mathbb Ev_k}`$.
- *First-order value* $`\mathrm{Val}_i`$ and the linear-response costates $`\chi^m,\chi^v`$ come from [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1)(d).
- $`\kappa_{\rm iso}=(1-\beta_1)/\big(\sqrt{1-\beta_2}\,(1-\beta_1/\sqrt{\beta_2})\big)`$ is the isolated-visit constant of [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11)(c).

**Statement ([Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2)).** In this setting:

(a) for every $`\lambda`$,

```math
\Delta F_i(\lambda)=\alpha_i\eta\sum_{j=0}^{T}\Big[\frac{\rho_i+b_j(\lambda-\rho_i)}{\sqrt{1+c_j(\lambda^2-1)}}-\frac{\rho_i(1-b_j)}{\sqrt{1-c_j}}\Big];
```

(b) with $`Q_i(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho_i\omega_T\lambda^2`$ and $`S_i(\lambda)=\lvert\lambda\rvert(1-\beta_1^{\,T+1})+\tfrac12\rho_i\omega_T\lambda^2`$, if $`\beta_2\ge\tfrac12`$ then for every $`\lambda`$

```math
\big\lvert\Delta F_i(\lambda)-\alpha_i\eta\,Q_i(\lambda)\big\rvert\le11\,(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\;\lvert\alpha_i\rvert\eta\,S_i(\lambda),
```

and the same bound holds for $`\mathrm{Val}_i(\lambda)-\mathrm{Val}_i(0)`$; $`Q_i`$ is concave with maximum at $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho_i\omega_T)`$; with $`u=11(1-\beta_2)\max\lbrace1,9\lambda^{\star2}\rbrace`$, $`\alpha_i>0`$ and $`u\le1/8`$, $`\Delta F_i>0`$ on $`[\lambda^\star,2\lambda^\star(1-4u)]`$ and $`\Delta F_i<0`$ on $`[2\lambda^\star(1+4u),3\lambda^\star]`$; the threshold $`2\lambda^\star`$ is usable only when $`(1-\beta_2)(2\lambda^\star)^2\ll1`$, with the stated short- and long-horizon forms;

(c) as $`\lambda\to\infty`$,

```math
\Delta F_i(\lambda)\to\alpha_i\eta\big[\kappa_{\rm iso}(T)-\rho_i\,\Lambda(T)\big],\qquad \kappa_{\rm iso}(T)=\sum_{j=0}^{T}\frac{b_j}{\sqrt{c_j}},\qquad \Lambda(T)=\sum_{j=0}^{T}\frac{1-b_j}{\sqrt{1-c_j}},
```

so for $`\alpha_i>0`$ all large same-direction messages are worse than a zero-gradient message (and $`\Delta F_i`$ has a positive zero) if $`\rho_i\Lambda(T)>\kappa_{\rm iso}(T)`$, and better if $`\rho_i\Lambda(T)<\kappa_{\rm iso}(T)`$; $`\Lambda(T)\approx T+1`$, and $`\kappa_{\rm iso}(T)\to\kappa_{\rm iso}`$ when $`\beta_1<\sqrt{\beta_2}`$;

(d) under SGD, $`\Delta F_i(\lambda)=\alpha_i\eta\sqrt{\bar v_i}\,\lambda`$.

**Proof.**

*(a)* Under the mean-field substitution the tail gradients are replaced by their moments, so the message's deviation decays geometrically. For $`j\ge0`$,

```math
\mathbb Em_{t+j,i}=\bar g_i+b_j(g_i-\bar g_i),\qquad \mathbb Ev_{t+j,i}=\bar v_i\big(1+c_j(g_i^2/\bar v_i-1)\big),
```

where index $`t+j`$ denotes the moments used by the $`j`$-th update after the message ($`j=0`$ is the message update).

- The $`j=0`$ case is the message update from $`m_t=\bar g`$, $`v_t=\bar v`$: $`m'=\bar g+(1-\beta_1)(g-\bar g)`$ and $`v'=\bar v+(1-\beta_2)(g^2-\bar v)`$.
- Each later update multiplies the deviation of $`\mathbb Em`$ by $`\beta_1`$ and that of $`\mathbb Ev`$ by $`\beta_2`$, because the tail gradients have mean $`\bar g_i`$ and mean square $`\bar v_i`$.
- With $`F`$ affine in $`\theta_i`$, the change of outcome is $`-(\nabla F)_i\,\eta\sum_{j=0}^{T}\big[\mathbb Em_{t+j}(g)/\sqrt{\mathbb Ev_{t+j}(g)}-\mathbb Em_{t+j}(0)/\sqrt{\mathbb Ev_{t+j}(0)}\big]`$.

Divide numerator and denominator by $`\sqrt{\bar v_i}`$ and use $`\bar g_i/\sqrt{\bar v_i}=\mathrm{sign}(\bar g_i)\rho_i`$, $`g_i/\sqrt{\bar v_i}=\mathrm{sign}(\bar g_i)\lambda`$ and $`-(\nabla F)_i\,\mathrm{sign}(\bar g_i)=\alpha_i`$. All denominators are positive, since $`c_j\le1-\beta_2<1`$.

*(b)* Write $`\gamma=1-\beta_2\le\tfrac12`$ and $`f_j(x)=(1+c_j(x-1))^{-1/2}`$. The bracket in (a) equals $`\rho_i(1-b_j)\big(f_j(\lambda^2)-f_j(0)\big)+b_j\lambda f_j(\lambda^2)`$, so

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

Summing, $`\lvert\Delta F_i/(\alpha_i\eta)-Q_i\rvert\le(1.42+8.49+0.5)\,\gamma\max\lbrace1,\lambda^2\rbrace S_i(\lambda)\le11\,\gamma\max\lbrace1,\lambda^2\rbrace S_i(\lambda)`$.

For the first-order value, the mean-field costates of [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1)(d) (the linear-response closed form) at the pre-message state are $`\chi^m=-\eta(\nabla F)_i\sum_{j=1}^T\beta_1^{\,j}/\sqrt{\bar v_i}`$ and $`\chi^v=\tfrac12\eta(\nabla F)_i\bar g_i\sum_{j=1}^T\beta_2^{\,j}/\bar v_i^{3/2}`$; here $`D_1=\bar v_i^{-1/2}`$ and $`D_2=\bar g_i/(2\bar v_i^{3/2})`$. In normalized units, $`\mathrm{Val}_i(\lambda)-\mathrm{Val}_i(0)`$ is the exact $`j=0`$ bracket of (a) plus $`\sum_{j=1}^T\big(b_j\lambda-\tfrac12\rho_ic_j\lambda^2\big)`$. Its difference from $`Q_i`$ is the $`j=0`$ part of $`E_1+E_2+E_3`$, which obeys the same bound.

The consequences:

- $`Q_i''=-\rho_i\omega_T<0`$, and $`Q_i'(\lambda^\star)=0`$.
- For the sign change, let $`\alpha_i>0`$ and $`\lambda=2\lambda^\star(1+\delta)`$ with $`\lvert\delta\rvert\le\tfrac12`$. Then $`\lambda^\star\le\lambda\le3\lambda^\star`$, so the error is at most $`u\,S_i(\lambda)`$. Also $`Q_i(\lambda)=-(1-\beta_1^{\,T+1})\lambda\delta`$ and $`S_i(\lambda)=(1-\beta_1^{\,T+1})\lambda(2+\delta)`$.
  - For $`\delta\in[4u,\tfrac12]`$: $`\Delta F_i/(\alpha_i\eta)\le(1-\beta_1^{\,T+1})\lambda\,[-\delta+u(2+\delta)]<0`$, because $`\delta(1-u)\ge4u(1-u)>2u`$ when $`u<\tfrac12`$.
  - For $`\delta\in[-\tfrac12,-4u]`$: $`\Delta F_i/(\alpha_i\eta)\ge(1-\beta_1^{\,T+1})\lambda\,[\lvert\delta\rvert-u(2-\lvert\delta\rvert)]>0`$, because $`\lvert\delta\rvert(1+u)>2u`$.
  - $`u\le1/8`$ makes $`4u\le\tfrac12`$, so both intervals are nonempty.
- The horizon statements follow from $`\omega_T=1-(1-\gamma)^{T+1}`$.

*(c)* As $`\lambda\to\infty`$, the $`j`$-th first fraction behaves like $`b_j\lambda/(\sqrt{c_j}\lambda)\to b_j/\sqrt{c_j}`$ (it tends to $`0=b_j/\sqrt{c_j}`$ when $`b_j=0`$). The second fraction does not depend on $`\lambda`$. This gives the limit.

- If the limit is negative and $`\alpha_i>0`$, then $`\Delta F_i(\lambda)<0`$ for all large $`\lambda`$. Also $`\Delta F_i(0)=0`$ and $`\Delta F_i'(0)=\alpha_i\eta\sum_jb_j/\sqrt{1-c_j}>0`$ (differentiate (a); $`b_0=1-\beta_1>0`$), so $`\Delta F_i>0`$ for small $`\lambda>0`$ and there is a positive zero. The positive case is symmetric.
- $`\Lambda(T)=\sum_j(1-b_j)(1+O(c_j))=(T+1)-(1-\beta_1^{\,T+1})+O(\omega_T)`$.
- $`\kappa_{\rm iso}(T)=\frac{1-\beta_1}{\sqrt{1-\beta_2}}\sum_{j\le T}(\beta_1/\sqrt{\beta_2})^j`$ converges to $`\kappa_{\rm iso}`$ when $`\beta_1<\sqrt{\beta_2}`$.
- The reading follows from (a): for $`\lambda`$ large and lag $`j`$ with $`b_j\lambda`$ negligible, the message's term is $`\rho_i(1-b_j)/\sqrt{1+c_j(\lambda^2-1)}`$ against the comparator's $`\rho_i(1-b_j)/\sqrt{1-c_j}`$.

*(d)* Plain SGD moves $`\theta_i`$ by $`-\eta g_i`$ and does not change later steps, so $`\Delta F_i=-(\nabla F)_i\eta g_i=\alpha_i\eta\sqrt{\bar v_i}\lambda`$. $`\square`$

**Remarks on the proof.**

- The mean-field substitution ignores the fluctuation of $`m_k`$ and $`v_k`$ around their means. It is part of the definition of $`\Delta F_i`$ and the only approximation in (a); how accurate it is on actual stochastic Adam trajectories is measured on the [experiments page](../experiments/07-02_stationary_amplitude_law.md#c7-2).
- The constant 11 in (b) is valid but conservative; the measured agreement of $`2\lambda^\star`$ with the actual sign change is far better ([Check 7.3](../experiments/07-02_stationary_amplitude_law.md#c7-3)).
- The positive-zero statement in (c) is a one-way implication. That no zero exists when the limit is positive is not claimed. In the checked settings the two always agreed ([Check 7.2](../experiments/07-02_stationary_amplitude_law.md#c7-2)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).
