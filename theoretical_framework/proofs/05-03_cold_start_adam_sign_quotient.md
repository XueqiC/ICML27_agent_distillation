# Proposition 5.3 · Cold-start Adam: sign quotient and optimizer memory

*Proof*

[← Framework](../README.md) · [Statement](../statements/05-03_cold_start_adam_sign_quotient.md) · [Proof](../proofs/05-03_cold_start_adam_sign_quotient.md) · [Experiments](../experiments/05-03_cold_start_adam_sign_quotient.md) · [中文](05-03_cold_start_adam_sign_quotient_ZH.md)

**What is proved here.** All five parts are proved: (a) the first-step formula; (b) the sign-law characterization, its exact statistic count, and a witness that coordinatewise sign marginals do not suffice; (c) the full-law case for $`\epsilon>0`$; (d) the margin-sensitive approximation bound; (e) the optimizer-memory witness, in which two histories with the same signs take second steps in opposite directions.

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

**Adam and AdamW.** Adam is bias-corrected, with $`\epsilon`$ outside the square root, optional decoupled weight decay (AdamW) and no AMSGrad. An update with counter $`t\ge1`$ acts on $`z=(\theta,m,v,t-1)`$ and a consumed gradient $`G\in\mathbb R^p`$. Coordinatewise,

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- The hyperparameters satisfy $`\beta_1\in[0,1)`$, $`\beta_2\in(0,1)`$, $`\eta>0`$, $`\lambda_{\rm wd}\ge0`$, $`\epsilon\ge0`$. Arithmetic is exact real arithmetic.
- The next complete state is $`z^+=(\theta^+,m^+,v^+,t)`$.
- A *cold* state has $`m=v=0`$ and $`t=1`$. A *warm* state has $`v_i>0`$ for every coordinate $`i`$.

**Sign map.** For $`\epsilon>0`$, $`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$. For $`\epsilon=0`$, the *ideal sign update* is $`h_0(g)_i=\mathrm{sign}(g_i)`$, with $`\mathrm{sign}(0)=0`$. This convention is part of the definition; a literal floating-point Adam with $`\epsilon=0`$ would compute $`0/0`$.

**Utilities.** An *immediate-parameter utility* is a bounded measurable function of $`\theta_1`$. A *complete-state utility* is a bounded measurable function of the next complete state $`(\theta_1,m_1,v_1,1)`$. Consumed-gradient laws $`P,Q`$ are probability laws on $`\mathbb R^p`$. In the finite statements they are supported on a finite alphabet $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$. Put $`\bar\theta=(1-\eta\lambda_{\rm wd})\theta_0`$.

**Statement ([Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)).** Start Adam from a cold state $`(\theta_0,0,0,0)`$.

- **(a) First step.** $`\theta_1=\bar\theta-\eta\,h_\epsilon(G)`$.
- **(b) $`\epsilon=0`$: the joint sign law.** Two laws give equal expected values for all immediate-parameter utilities if and only if their joint sign-vector laws coincide. On a finite alphabet with $`s`$ attainable sign patterns, the minimum number of linear expectation statistics that suffices for all such utilities, over unrestricted laws on $`\mathcal A`$, is $`s-1`$; here $`s\le3^p`$, or $`s\le2^p`$ without zero coordinates. Coordinatewise sign marginals do not suffice: for $`p=2`$, let $`P`$ be uniform on $`\lbrace (1,1),(-1,-1)\rbrace`$ and $`Q`$ uniform on $`\lbrace (1,-1),(-1,1)\rbrace`$. Every coordinate has the same sign marginal and the mean step is $`0`$ under both laws, yet the utility $`F(\theta)=1\lbrace (\theta_1-\bar\theta_1)(\theta_2-\bar\theta_2)>0\rbrace`$ has expectation $`1`$ under $`P`$ and $`0`$ under $`Q`$.
- **(c) $`\epsilon>0`$: the full law.** $`h_\epsilon`$ is a coordinatewise bijection $`\mathbb R\to(-1,1)`$, with the inverse and derivative below. Equality for all immediate-parameter utilities is therefore equivalent to equality of the full consumed-gradient laws. On $`\mathcal A`$ the minimum number of statistics is $`m-1`$.

```math
h_\epsilon^{-1}(u)_i=\frac{\epsilon u_i}{1-\lvert u_i\rvert},\qquad\frac{d}{du}h_\epsilon^{-1}(u)=\frac{\epsilon}{(1-\lvert u\rvert)^2}.
```

**(d) Margin-sensitive approximation.** Let $`F:\mathbb R^p\to[0,1]`$ satisfy $`\lvert F(u)-F(w)\rvert\le\sum_iL_i\lvert u_i-w_i\rvert`$ for all pairs $`u=\bar\theta-\eta h_\epsilon(g)`$, $`w=\bar\theta-\eta h_0(g)`$ with $`g`$ in the supports. For $`a>0`$ put $`\beta_{P,i}(a)=P(0<\lvert G_i\rvert<a)`$, and let $`\Delta_{\rm sign}(F)=\lvert\mathbb E_PF(\bar\theta-\eta h_0(G))-\mathbb E_QF(\bar\theta-\eta h_0(G))\rvert`$. Then $`\Delta_{\rm sign}(F)\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$ and

```math
\big\lvert\mathbb E_PF(\bar\theta-\eta h_\epsilon(G))-\mathbb E_QF(\bar\theta-\eta h_\epsilon(G))\big\rvert\le\Delta_{\rm sign}(F)+\eta\sum_iL_i\Big[\beta_{P,i}(a)+\beta_{Q,i}(a)+\frac{2\epsilon}{a+\epsilon}\Big].
```

**(e) Retained optimizer state.** For $`\beta_1<1`$, the first complete state contains $`m_1=(1-\beta_1)G`$, an injective function of $`G`$.

- *Witness.* Take one coordinate, $`\beta_1=\beta_2=\tfrac12`$, $`\lambda_{\rm wd}=0`$, $`\epsilon=0`$, $`\eta>0`$ and a cold start. Compare the gradient histories $`(G_1,G_2)=(1,-\tfrac34)`$ and $`(G_1',G_2')=(2,-\tfrac34)`$. Both give $`\theta_1=\theta_1'=\theta_0-\eta`$ and the same sign history $`(+,-)`$. At step 2, the bias-corrected moments are $`\widehat m_2=-\tfrac16`$ and $`\widehat m_2'=+\tfrac16`$, and $`\widehat v_2=\tfrac{17}{24}`$ and $`\widehat v_2'=\tfrac{41}{24}`$, so the second steps below move in opposite directions. The bounded test $`F(\theta)=1\lbrace \theta>\theta_1\rbrace`$ separates the histories by $`1`$. The histories are realizable by squared losses $`\tfrac12(\theta-a)^2`$ at $`\theta_0=0`$: first targets $`a=-1`$ and $`a=-2`$, then the common second target $`a=\theta_1+\tfrac34`$.

```math
\theta_2=\theta_1+\frac{\eta}{6\sqrt{17/24}},\qquad\theta_2'=\theta_1-\frac{\eta}{6\sqrt{41/24}}.
```

- *Momentum carries the memory.* If $`\beta_1=0`$, any two histories with the same sign of $`G_2`$ have second steps in the same direction, so no witness of this type exists. For every $`\beta_1\in(0,1)`$ one exists.

*Proof of (a).* Cold moments give $`m_1=(1-\beta_1)G`$ and $`v_1=(1-\beta_2)G^{\odot2}`$. Dividing by the bias corrections at $`t=1`$ gives $`n=G`$ and $`d^2=G^{\odot2}`$, so $`d=\lvert G\rvert`$ coordinatewise. Substituting gives $`\theta_1=\bar\theta-\eta\,G/(\lvert G\rvert+\epsilon)`$. At $`\epsilon=0`$ the declared convention gives $`h_0=\mathrm{sign}`$. $`\square`$

*Proof of (b).* The map $`\sigma\mapsto\bar\theta-\eta\sigma`$ from $`\lbrace -1,0,1\rbrace ^p`$ to $`\mathbb R^p`$ is injective, because $`\eta>0`$. Hence $`\theta_1`$ is an injective function of the sign vector $`\sigma(G)`$.

- *Sufficiency.* If the joint sign laws coincide, the laws of $`\theta_1`$ coincide, and so do all expected utilities.
- *Necessity.* For each attainable pattern $`\sigma^\star`$, the indicator $`F=1\lbrace \theta_1=\bar\theta-\eta\sigma^\star\rbrace`$ is a bounded utility with expectation $`\Pr(\sigma(G)=\sigma^\star)`$. Equal expectations for all such $`F`$ force equal pattern probabilities.
- *Dimension.* View every utility as a function on $`\mathcal A`$, namely $`u_F(g)=F(\bar\theta-\eta h_0(g))`$. The family $`\mathcal U=\lbrace u_F\rbrace`$ consists exactly of the functions on $`\mathcal A`$ that are constant on the fibres of $`\sigma`$: it contains them because $`F`$ is arbitrary on the $`s`$ distinct output points, and nothing else because $`u_F`$ depends on $`g`$ only through $`\sigma(g)`$. So $`\mathrm{span}(\lbrace 1\rbrace \cup\mathcal U)`$ is spanned by the $`s`$ fibre indicators and has dimension $`s`$. By [Theorem 4.2(a)](../statements/04-02_sufficiency_duality_transfer.md#s4-2), the minimum number of linear statistics whose equal means imply equal means of every $`u\in\mathcal U`$, for all laws on $`\mathcal A`$, is $`s-1`$; the indicators of $`s-1`$ patterns attain it. Each coordinate sign takes at most three values, so $`s\le3^p`$, and $`s\le2^p`$ when $`0`$ is excluded.
- *Marginals.* Under both laws of the witness each coordinate equals $`\pm1`$ with probability $`1/2`$, and $`\mathbb E h_0(G)=0`$. Under $`P`$ the two coordinates of $`\theta_1-\bar\theta=-\eta\sigma`$ always share a sign, so $`F=1`$; under $`Q`$ they never do, so $`F=0`$. $`\square`$

*Proof of (c).* Fix $`\epsilon>0`$. On $`g\ge0`$, $`h_\epsilon(g)=g/(g+\epsilon)`$ is continuous and strictly increasing from $`0`$ to $`1`$. By oddness, $`h_\epsilon`$ is a bijection $`\mathbb R\to(-1,1)`$.

- Solve $`u=g/(\lvert g\rvert+\epsilon)`$. Then $`\mathrm{sign}u=\mathrm{sign}g`$ and $`\lvert u\rvert(\lvert g\rvert+\epsilon)=\lvert g\rvert`$, so $`\lvert g\rvert=\epsilon\lvert u\rvert/(1-\lvert u\rvert)`$ and $`g=\epsilon u/(1-\lvert u\rvert)`$. For $`u>0`$ the derivative is $`\epsilon/(1-u)+\epsilon u/(1-u)^2=\epsilon/(1-u)^2`$, and by symmetry the same formula holds for $`u<0`$.
- Since $`g\mapsto\bar\theta-\eta h_\epsilon(g)`$ is injective, the argument of (b) applies with fibres equal to singletons. Equality for all bounded utilities is then equivalent to equality of the laws of $`G`$, and on $`\mathcal A`$ the dimension is $`m-1`$. $`\square`$

*Proof of (d).* Fix a coordinate $`i`$ and a gradient value $`g`$.

- If $`g_i=0`$, then $`h_\epsilon(g)_i=0=h_0(g)_i`$.
- If $`g_i\ne0`$, then $`\lvert h_\epsilon(g_i)-\mathrm{sign}(g_i)\rvert=\lvert g_i\rvert\cdot\big\lvert\tfrac1{\lvert g_i\rvert+\epsilon}-\tfrac1{\lvert g_i\rvert}\big\rvert=\tfrac{\epsilon}{\lvert g_i\rvert+\epsilon}`$. This is at most $`1`$ always, and at most $`\epsilon/(a+\epsilon)`$ when $`\lvert g_i\rvert\ge a`$.

Hence $`\mathbb E_P\lvert h_\epsilon(G_i)-h_0(G_i)\rvert\le\beta_{P,i}(a)+\epsilon/(a+\epsilon)`$. By the Lipschitz hypothesis applied pointwise to the pair $`\big(\bar\theta-\eta h_\epsilon(G),\,\bar\theta-\eta h_0(G)\big)`$,

```math
\big\lvert\mathbb E_PF(\bar\theta-\eta h_\epsilon(G))-\mathbb E_PF(\bar\theta-\eta h_0(G))\big\rvert\le\eta\sum_iL_i\Big[\beta_{P,i}(a)+\frac{\epsilon}{a+\epsilon}\Big],
```

and the same holds under $`Q`$. The triangle inequality through the two ideal-sign expectations gives the bound. For the total-variation statement, $`F(\bar\theta-\eta h_0(G))=f(\sigma(G))`$ for some $`f:\lbrace -1,0,1\rbrace ^p\to[0,1]`$, and any such $`f`$ satisfies $`\lvert\mathbb E_Pf-\mathbb E_Qf\rvert\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$. $`\square`$

*Proof of (e).* Injectivity of $`m_1=(1-\beta_1)G`$ is immediate for $`\beta_1<1`$. For the witness, $`1-\beta_1=1-\beta_2=\tfrac12`$, and the bias corrections are $`\tfrac12`$ at $`t=1`$ and $`\tfrac34`$ at $`t=2`$.

- *Step 1.* $`m_1=G_1/2`$ and $`v_1=G_1^2/2`$, so $`\widehat m_1=G_1`$ and $`\widehat v_1=G_1^2`$. Since $`G_1>0`$ in both histories, $`\theta_1=\theta_0-\eta`$.
- *Step 2.* $`m_2=\tfrac12m_1+\tfrac12G_2=\tfrac{G_1}4-\tfrac38`$, which is $`-\tfrac18`$ for $`G_1=1`$ and $`+\tfrac18`$ for $`G_1=2`$; then $`\widehat m_2=\tfrac43m_2=\mp\tfrac16`$. Similarly $`v_2=\tfrac12v_1+\tfrac12G_2^2=\tfrac{G_1^2}4+\tfrac9{32}`$, which is $`\tfrac{17}{32}`$ or $`\tfrac{41}{32}`$, so $`\widehat v_2=\tfrac{17}{24}`$ or $`\tfrac{41}{24}`$.
- With $`\epsilon=0`$, $`\theta_2=\theta_1-\eta\widehat m_2/\sqrt{\widehat v_2}`$, which is the display. The denominators are positive and the numerators have opposite signs, so the steps are opposite, and $`F(\theta_2)=1`$, $`F(\theta_2')=0`$.
- *Realizability.* The gradient of $`\tfrac12(\theta-a)^2`$ is $`\theta-a`$. At $`\theta_0=0`$ the targets $`a=-1,-2`$ give $`G_1=1,2`$. Both first steps land at $`\theta_1=-\eta`$, so the common target $`a=\theta_1+\tfrac34`$ gives $`G_2=-\tfrac34`$ in both histories. No branch-dependent future gradient is needed.
- *Positive $`\epsilon`$.* For the same gradient sequences the direction of the second step is $`-\mathrm{sign}(\widehat m_2)`$ for every $`\epsilon\ge0`$, because the denominator $`\sqrt{\widehat v_2}+\epsilon`$ is positive; the two directions stay opposite.
- *Momentum.* If $`\beta_1=0`$ then $`m_2=G_2`$, and the second step has the sign of $`-G_2`$ whatever the first gradient was. If $`\beta_1>0`$, then $`m_2=\beta_1(1-\beta_1)G_1+(1-\beta_1)G_2`$. For $`0<G_1<G_1'`$, choosing $`G_2<0`$ with $`\lvert G_2\rvert`$ strictly between $`\beta_1G_1`$ and $`\beta_1G_1'`$ gives opposite signs of $`m_2`$, while the first steps $`-\eta\,\mathrm{sign}(G_1)`$ coincide at $`\epsilon=0`$. $`\square`$

**Discussion of Proposition 5.3.**

- *Three levels of sufficiency.* Immediate-parameter sufficiency of a cold step needs the joint sign law at $`\epsilon=0`$, and the full law at $`\epsilon>0`$. Complete-state sufficiency needs the full consumed-gradient law, because $`m_1`$ is injective. Task sufficiency needs only the response class of the actual continuation ([Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2)), which may be much smaller. The exact statistic counts concern unrestricted laws; a restricted task family can need fewer.
- *The discontinuity at $`\epsilon=0`$ is exact, not operational.* For arbitrary measurable utilities, any $`\epsilon>0`$ makes magnitudes identifiable. For Lipschitz utilities with little gradient mass near zero, (d) shows that the sign approximation is accurate. The inverse in (c) is ill-conditioned as $`\lvert u\rvert\to1`$.
- *Warm states move the threshold.* At a warm state the numerator of coordinate $`i`$ changes sign at $`G_i=-\beta_1m_i/(1-\beta_1)`$, and the denominator depends on $`v_i`$ and $`G_i^2`$. For a scalar cross-entropy gradient $`G=p-Z`$, the cold sign probabilities are the masses of $`Z`$ below, at and above the current prediction $`p`$. Allowing all thresholds identifies the distribution function of $`Z`$; no fixed threshold or marginal sign feature is sufficient across changing states.
- *Consequence for the affine theory.* Two label laws can match all moments below order $`N`$ ([Theorem 4.4](../statements/04-04_affine_learners_moment_order.md#s4-4)) and still differ in their sign probabilities about the current prediction. A cold Adam step separates them at first order in $`\eta`$ (Proposition 5.7).

## Related work

- The bias-corrected first step of Adam and its sign-descent behaviour are standard: [Kingma & Ba (2015)](../references.md#kingma2015adam); [Balles & Hennig (2018)](../references.md#balles2018dissecting) on Adam as sign descent with variance adaptation; [Liu et al. (RAdam, ICLR 2020)](../references.md#liu2020variance) on early-step variance; signSGD ([Bernstein et al., 2018](../references.md#bernstein2018signsgd)) and Lion ([Chen et al., 2023](../references.md#chen2023symbolic)). Decoupled weight decay is [Loshchilov & Hutter (2019)](../references.md#loshchilov2019decoupled).

## References

- [Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.](../references.md#kingma2015adam)
- [Balles, L., & Hennig, P. (2018). Dissecting Adam: The sign, magnitude and variance of stochastic gradients. *ICML*.](../references.md#balles2018dissecting)
- [Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., & Han, J. (2020). On the variance of the adaptive learning rate and beyond. *ICLR*.](../references.md#liu2020variance)
- [Bernstein, J., Wang, Y.-X., Azizzadenesheli, K., & Anandkumar, A. (2018). signSGD: Compressed optimisation for non-convex problems. *ICML*.](../references.md#bernstein2018signsgd)
- [Chen, X., Liang, C., Huang, D., Real, E., et al. (2023). Symbolic discovery of optimization algorithms. *NeurIPS*.](../references.md#chen2023symbolic)
- [Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.](../references.md#loshchilov2019decoupled)
