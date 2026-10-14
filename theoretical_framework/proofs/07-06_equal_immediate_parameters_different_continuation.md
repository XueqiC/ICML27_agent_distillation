# Proposition 7.6 · Equal immediate parameters do not fix the continuation

*Proof*

[← Framework](../README.md) · [Statement](../statements/07-06_equal_immediate_parameters_different_continuation.md) · [Proof](07-06_equal_immediate_parameters_different_continuation.md) · [Experiments](../experiments/07-06_equal_immediate_parameters_different_continuation.md) · [中文](07-06_equal_immediate_parameters_different_continuation_ZH.md)

**What is proved here.** Part (a) by backward induction on the reference kernels. Part (b) from the sign of the derivative of the step magnitude and the Cauchy–Schwarz inequality. Part (c) by exact rational arithmetic. Part (d) by the implicit function theorem. Part (e) by direct algebra on the moment recursions.

**Setting used on this page.**

- $`z=(\theta,o)`$ is the complete learner state ([Definition 1.3](../README.md#s1-3)); $`V^0_{t+1}`$ is the reference continuation value and $`g_{t,z}(y)=\mathbb E_{\xi_t}V^0_{t+1}(U_t(z,y,\xi_t))`$ the one-boundary response ([Definition 1.4](../README.md#s1-4)).
- A learner is *memoryless* if the reference kernels $`K^0_k`$, $`k>t`$, map functions of $`(\theta,k)`$ to functions of $`(\theta,k)`$ and $`F_E(z)=F_E(\theta)`$.
- One coordinate of Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) with bias correction and no weight decay: at step counter $`s`$, $`m'=\beta_1m+(1-\beta_1)g`$, $`v'=\beta_2v+(1-\beta_2)g^2`$, $`\theta'=\theta-\eta q_{1,s}m'/(\sqrt{q_{2,s}v'}+\epsilon)`$, with $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$ and $`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$.
- For $`\epsilon=0`$ and a fixed state: $`a=\beta_1m`$, $`b=1-\beta_1`$, $`c=\beta_2v`$, $`d=1-\beta_2`$, $`S(g)=(a+bg)/\sqrt{c+dg^2}`$, and $`\theta'=\theta-\eta q_{1,s}q_{2,s}^{-1/2}S(g)`$.
- Consumed gradients are realized by squared losses $`\tfrac12(\theta-y)^2`$: at the current parameter $`\theta`$ the label $`y=\theta-g`$ gives the gradient $`g`$, and the Hessian is $`1`$.

**Statement ([Proposition 7.6](../statements/07-06_equal_immediate_parameters_different_continuation.md#s7-6)).** (a) For a memoryless learner, messages whose updates give the same law of the next parameters have equal one-boundary responses for every declared task and memoryless continuation. (b) For Adam with $`\epsilon=0`$, $`v>0`$, $`m>0`$, the step magnitude $`S`$ is strictly increasing on $`(-a/b,g^\star]`$ and strictly decreasing on $`[g^\star,\infty)`$, with $`g^\star=bc/(ad)`$, $`S(g^\star)=\sqrt{a^2/c+b^2/d}`$ and limit $`b/\sqrt d`$; every value strictly between $`b/\sqrt d`$ and $`S(g^\star)`$ is taken by exactly two gradients, which give the same next parameter and different first moments. (c) With $`\beta_1=0.9`$, $`\beta_2=0.999`$, $`\eta=1`$, $`\epsilon=0`$ and bias correction, the gradients $`\tfrac12`$ and $`\tfrac{2103}{790}`$ at $`z_1=(-1,\tfrac1{10},\tfrac1{1000})`$ give equal second-step parameters and different third-step parameters after a common third-step loss. (d) The same holds for every sufficiently small $`\epsilon>0`$ with a gradient $`g_B(\epsilon)`$ continuous in $`\epsilon`$. (e) $`(v'-v)/v=(1-\beta_2)(g^2/v-1)`$; with $`v`$ fixed and tail gradients independent of $`\theta`$, the lag-$`k`$ displacement of a perturbation $`\delta g`$ is $`-\eta(1-\beta_1)\beta_1^{\,k}\delta g/\sqrt v`$.

**Proof of (a).**

1. $`V^0_R=F_E`$ is a function of $`\theta`$ alone.
2. Suppose $`V^0_{k+1}(z)=\phi_{k+1}(\theta)`$ for some $`k>t`$. Since $`K^0_k`$ maps functions of $`(\theta,k)`$ to functions of $`(\theta,k)`$, $`V^0_k=K^0_kV^0_{k+1}`$ is again a function of $`\theta`$ alone. By induction down to $`k=t+1`$, $`V^0_{t+1}(z)=\phi_{t+1}(\theta)`$.
3. Hence $`g_{t,z}(y)=\mathbb E\,\phi_{t+1}(\theta_{t+1})`$, where $`\theta_{t+1}`$ is the parameter component of $`U_t(z,y,\xi_t)`$. This depends on $`y`$ only through the law of $`\theta_{t+1}`$. The argument applies to each declared task and each memoryless continuation separately. $`\square`$

**Proof of (b).**

**1.** For $`g>-a/b`$ the numerator $`a+bg`$ is positive, so $`S(g)>0`$, and

```math
S'(g)=\frac{b(c+dg^2)-dg(a+bg)}{(c+dg^2)^{3/2}}=\frac{bc-adg}{(c+dg^2)^{3/2}}.
```

**2.** Because $`a,b,c,d>0`$, $`S'>0`$ for $`g<g^\star=bc/(ad)`$ and $`S'<0`$ for $`g>g^\star`$. This gives the two strict monotonicity statements.

**3.** *Peak value.* Write $`a+bg=(a/\sqrt c)\sqrt c+(b/\sqrt d)(\sqrt d\,g)`$. By Cauchy–Schwarz, $`a+bg\le\sqrt{a^2/c+b^2/d}\,\sqrt{c+dg^2}`$, with equality exactly when $`(\sqrt c,\sqrt d\,g)`$ is a positive multiple of $`(a/\sqrt c,b/\sqrt d)`$, that is, when $`g=bc/(ad)=g^\star`$. So $`S(g^\star)=\sqrt{a^2/c+b^2/d}`$.

**4.** *Ends of the range.* $`S(g)\to0`$ as $`g\downarrow-a/b`$, and $`S(g)\to b/\sqrt d`$ as $`g\to\infty`$. On $`[g^\star,\infty)`$, $`S`$ decreases strictly towards its limit, so $`S(g)>b/\sqrt d`$ there.

**5.** By continuity and strict monotonicity, every $`u`$ with $`b/\sqrt d<u<S(g^\star)`$ has exactly one preimage $`g_A\in(-a/b,g^\star)`$, because $`S`$ increases from $`0`$ to $`S(g^\star)`$ there, and exactly one preimage $`g_B\in(g^\star,\infty)`$, because $`S`$ decreases from $`S(g^\star)`$ to $`b/\sqrt d`$ without reaching it.

**6.** Equal $`S`$ at the same state $`(\theta,s)`$ gives equal $`\theta'=\theta-\eta q_{1,s}q_{2,s}^{-1/2}S(g)`$. The first moment $`m'=a+bg`$ is strictly increasing in $`g`$, so $`g_A<g_B`$ gives $`m'_A<m'_B`$. $`\square`$

**Proof of (c).**

**1.** *First step*, $`s=1`$, from $`(0,0,0)`$ with gradient $`1`$: $`m_1=\tfrac1{10}`$, $`v_1=\tfrac1{1000}`$, $`q_{1,1}=10`$, $`q_{2,1}=1000`$, so $`\theta_1=-10\cdot\tfrac1{10}/\sqrt{1000\cdot\tfrac1{1000}}=-1`$. This is $`z_1`$.

**2.** *Second step*, $`s=2`$: $`q_{1,2}=\tfrac{100}{19}`$, $`q_{2,2}=\tfrac{10^6}{1999}`$, and

```math
m_2(g)=\frac{0.9+g}{10},\qquad v_2(g)=\frac{0.999+g^2}{1000},\qquad \frac{m_2(g)^2}{v_2(g)}=10\,\frac{(0.9+g)^2}{0.999+g^2}.
```

For $`g>-0.9`$ the step is $`\eta q_{1,2}q_{2,2}^{-1/2}\sqrt{m_2^2/v_2}`$, so two such gradients give equal $`\theta_2`$ if and only if they give equal $`m_2^2/v_2`$.

**3.** *The two gradients.* For $`g_A=\tfrac12`$: $`(0.9+g_A)^2=\tfrac{49}{25}`$ and $`0.999+g_A^2=\tfrac{1249}{1000}`$, so the ratio is $`\tfrac{1960}{1249}`$. For $`g_B=\tfrac{2103}{790}`$: $`0.9+g_B=\tfrac{1407}{395}`$, $`(0.9+g_B)^2=\tfrac{1979649}{156025}`$ and $`0.999+g_B^2=\tfrac{50460849}{6241000}`$. Since $`6241000=40\cdot156025`$, $`50460849=1249\cdot40401`$ and $`1979649\cdot40=79185960=1960\cdot40401`$, the ratio is again $`\tfrac{1960}{1249}`$. Both sides therefore have $`m_2^2/v_2=\tfrac{19600}{1249}`$ and the same $`\theta_2`$.

**4.** *The quadratic.* Clearing denominators in $`(0.9+g)^2/(0.999+g^2)=\tfrac{1960}{1249}`$ gives $`1249(g^2+1.8g+0.81)=1960(g^2+0.999)`$, that is, $`711g^2-2248.2\,g+946.35=0`$. Its roots have sum $`\tfrac{2248.2}{711}=\tfrac{1249}{395}=\tfrac12+\tfrac{2103}{790}`$ and product $`\tfrac{946.35}{711}=\tfrac{2103}{1580}=\tfrac12\cdot\tfrac{2103}{790}`$, so they are exactly $`g_A`$ and $`g_B`$. In the notation of (b), $`a=0.09`$, $`b=0.1`$, $`c=0.000999`$, $`d=0.001`$, and $`g^\star=1.11`$ lies between them.

**5.** *Moments after the second step.* $`m_2^A=\tfrac7{50}`$, $`v_2^A=\tfrac{1249}{10^6}`$; $`m_2^B=\tfrac{1407}{3950}`$, $`v_2^B=\tfrac{50460849}{6241\cdot10^6}`$. They differ.

**6.** *Third step*, $`s=3`$, with one common loss whose gradient at the common $`\theta_2`$ is $`1`$: $`m_3=0.9m_2+0.1`$ and $`v_3=0.999v_2+0.001`$, both positive, and $`\theta_3=\theta_2-\eta q_{1,3}q_{2,3}^{-1/2}\sqrt{m_3^2/v_3}`$. Exactly,

```math
\Big(\frac{m_3^2}{v_3}\Big)^A=\frac{0.226^2}{0.002247751}=\frac{51076000}{2247751},\qquad \Big(\frac{m_3^2}{v_3}\Big)^B=\frac{1103967076000}{56651388151}.
```

These rationals differ (the first is about $`22.72`$, the second about $`19.49`$), so $`\theta_3^A\ne\theta_3^B`$ and $`\theta_3^A<\theta_3^B`$. Evaluated in floating point, $`\theta_3^A-\theta_3^B\approx-0.0712027`$, and the sigmoid score differs by $`\approx-0.00364994`$ ([Check 7.8](../experiments/07-06_equal_immediate_parameters_different_continuation.md#c7-8)).

**7.** *General $`\eta`$.* The moments do not depend on $`\eta`$, and every parameter step is $`\eta`$ times an $`\eta`$-free quantity. With the same consumed gradients (labels $`y=\theta-g`$ at the current parameter), the second-step parameters again agree and the third-step parameter difference is $`\eta`$ times the value above. $`\square`$

**Proof of (d).**

1. Run the whole instance of (c), including the first step, with a common $`\epsilon`$, and let $`\theta_2(g,\epsilon)`$ and $`\theta_3(g,\epsilon)`$ be the second- and third-step parameters when the second-step gradient is $`g`$. For $`\lvert\epsilon\rvert`$ small every denominator $`\sqrt{q_{2,s}v'}+\epsilon`$ is positive, because all $`v'`$ are positive, so both maps are $`C^1`$ on an open set containing $`(g_B,0)`$.
2. Let $`\Phi(g,\epsilon)=\theta_2(g,\epsilon)-\theta_2(\tfrac12,\epsilon)`$. By (c), $`\Phi(g_B,0)=0`$.
3. At $`\epsilon=0`$, $`\partial_g\theta_2=-\eta q_{1,2}q_{2,2}^{-1/2}S'(g)`$, and $`S'(g_B)<0`$ by (b) because $`g_B>g^\star=1.11`$. So $`\partial_g\Phi(g_B,0)\ne0`$.
4. By the implicit function theorem there are $`\epsilon_0>0`$ and a $`C^1`$ function $`g_B(\epsilon)`$ on $`(-\epsilon_0,\epsilon_0)`$ with $`g_B(0)=g_B`$ and $`\Phi(g_B(\epsilon),\epsilon)=0`$: the immediate parameters agree for every such $`\epsilon`$.
5. $`\Psi(\epsilon)=\theta_3(\tfrac12,\epsilon)-\theta_3(g_B(\epsilon),\epsilon)`$ is continuous, and $`\Psi(0)\ne0`$ by (c). Hence $`\Psi(\epsilon)\ne0`$ for all sufficiently small $`\epsilon`$. At $`\epsilon=10^{-8}`$ a bracketing root search gives $`g_B(\epsilon)\approx2.662025475`$ and $`\Psi(\epsilon)\approx-0.0712027`$ ([Check 7.8](../experiments/07-06_equal_immediate_parameters_different_continuation.md#c7-8)). $`\square`$

**Proof of (e).**

1. $`v'-v=(\beta_2-1)v+(1-\beta_2)g^2=(1-\beta_2)(g^2-v)`$; divide by $`v>0`$.
2. Compare two runs that differ only by $`\delta g`$ in the message's consumed gradient. With tail gradients that do not depend on $`\theta`$, both runs consume the same tail gradients, and the linear recursion $`m_{k+1}=\beta_1m_k+(1-\beta_1)g_{k+1}`$ gives the momentum difference $`\delta m_k=(1-\beta_1)\beta_1^{\,k}\delta g`$ at lag $`k`$ ($`k=0`$ is the message's own update).
3. With $`v`$ held fixed, no bias correction and $`\epsilon=0`$, the parameter steps differ by $`-\eta\,\delta m_k/\sqrt v=-\eta(1-\beta_1)\beta_1^{\,k}\delta g/\sqrt v`$.
4. Summing, $`\sum_{k<K}(1-\beta_1)\beta_1^{\,k}=1-\beta_1^{\,K}`$, which tends to $`1`$; the total first-order displacement is $`-\eta\,\delta g/\sqrt v`$, of which the message's own update carries $`1-\beta_1`$. $`\square`$

## Related work

- In terms of the costates of [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1), the two messages of (c) have equal parameter-channel increments, so the difference of their first-order values is $`\langle\chi^m_{t+1},(1-\beta_1)(g_A-g_B)\rangle+\langle\chi^v_{t+1},(1-\beta_2)(g_A^2-g_B^2)\rangle`$. The construction is an instance where only the memory channels of Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) separate two messages.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).
