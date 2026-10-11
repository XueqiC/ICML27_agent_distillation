# N11 · Cold-start Adam quotient and optimizer memory — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N11_adam_cold_start.md) · [中文](N11_adam_cold_start_ZH.md)

## Setting and assumptions

**Optimizer.** We use bias-corrected Adam, with optional decoupled weight decay (AdamW), $`\epsilon`$ outside the square root and no AMSGrad. The update consumes the gradient $`G_t\in\mathbb R^p`$ *after* any declared accumulation, loss normalization, rescaling or clipping. With step counter $`t\ge1`$:

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)G_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)G_t^{\odot2},
```

```math
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\,\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}. \qquad(A6)
```

Division and square root act coordinatewise. The parameters satisfy $`\beta_1,\beta_2\in[0,1)`$, $`\eta>0`$, $`\lambda\ge0`$ and $`\epsilon\ge0`$. Arithmetic is exact real arithmetic, with no projection or quantization.

**Cold state.** A cold state is $`m_0=v_0=0`$ with counter $`0`$. Put $`c=(1-\eta\lambda)\theta_0`$.

**Sign map.** For $`\epsilon>0`$, define $`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$. For $`\epsilon=0`$, define the *ideal sign update* $`h_0(g)_i=\mathrm{sign}(g_i)`$, with $`\mathrm{sign}(0)=0`$. This convention is part of the definition; a literal floating-point Adam with $`\epsilon=0`$ would compute $`0/0`$.

**Utilities.** An *immediate-parameter utility* is a bounded measurable function $`F`$ of $`\theta_1`$. A *complete-state utility* is a bounded measurable function of the next complete state $`(\theta_1,m_1,v_1,1)`$. Consumed-gradient laws $`P,Q`$ are probability laws on $`\mathbb R^p`$. In the finite statements they are supported on a finite alphabet $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$.

## Statement

**(a)–(c)** **(cold quotient).**

**1.** **(a)** From a cold state, $`\theta_1=c-\eta\,h_\epsilon(G)`$.

**2.** **(b)** ($`\epsilon=0`$.) Two laws give equal expected values for all immediate-parameter utilities if and only if their joint sign-vector laws coincide. On a finite alphabet with $`s`$ attainable sign patterns, the minimum number of linear expectation statistics that suffices for all such utilities, over unrestricted laws on $`\mathcal A`$, is $`s-1`$. Here $`s\le3^p`$, or $`s\le2^p`$ without zero coordinates.

**3.** **(c)** ($`\epsilon>0`$.) $`h_\epsilon`$ is a coordinatewise bijection $`\mathbb R\to(-1,1)`$ with

```math
h_\epsilon^{-1}(u)_i=\frac{\epsilon u_i}{1-\lvert u_i\rvert},\qquad\frac{d}{du}h_\epsilon^{-1}(u)=\frac{\epsilon}{(1-\lvert u\rvert)^2}. \qquad(A8)
```

Equality for all immediate-parameter utilities is therefore equivalent to equality of the full consumed-gradient laws. On $`\mathcal A`$ the minimum number of statistics is $`m-1`$.

**4.** (Joint versus marginal.) Coordinatewise sign marginals do not determine the joint sign law. Take $`p=2`$, $`P`$ uniform on $`\lbrace (1,1),(-1,-1)\rbrace`$ and $`Q`$ uniform on $`\lbrace (1,-1),(-1,1)\rbrace`$. Every coordinate has the same sign marginal and the mean update is $`0`$ under both laws. The utility $`F(\theta)=1\lbrace (\theta_1-c_1)(\theta_2-c_2)>0\rbrace`$ has expectation $`1`$ under $`P`$ and $`0`$ under $`Q`$.

**(d)** **Proposition (margin-sensitive approximation).** Let $`F:\mathbb R^p\to[0,1]`$ satisfy $`\lvert F(u)-F(w)\rvert\le\sum_iL_i\lvert u_i-w_i\rvert`$ for all pairs $`u=c-\eta h_\epsilon(g)`$, $`w=c-\eta h_0(g)`$ with $`g`$ in the supports. For $`a>0`$ put $`\beta_{P,i}(a)=P(0<\lvert G_i\rvert<a)`$, and let $`\Delta_{\rm sign}(F)=\lvert\mathbb E_PF(c-\eta h_0(G))-\mathbb E_QF(c-\eta h_0(G))\rvert`$. Then

```math
\big\lvert\mathbb E_PF(c-\eta h_\epsilon(G))-\mathbb E_QF(c-\eta h_\epsilon(G))\big\rvert\le\Delta_{\rm sign}(F)+\eta\sum_iL_i\Big[\beta_{P,i}(a)+\beta_{Q,i}(a)+\frac{2\epsilon}{a+\epsilon}\Big], \qquad(A9)
```

and $`\Delta_{\rm sign}(F)\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$.

**(e)** **Proposition (retained optimizer state).** For $`\beta_1<1`$, the first complete state contains $`m_1=(1-\beta_1)G`$, an injective function of $`G`$.

*Counterexample.* Take one coordinate, $`\beta_1=\beta_2=\tfrac12`$, $`\lambda=0`$, $`\epsilon=0`$, $`\eta>0`$ and a cold start. Compare the gradient histories $`(G_1,G_2)=(1,-\tfrac34)`$ and $`(G_1',G_2')=(2,-\tfrac34)`$.

Both give $`\theta_1=\theta_1'=\theta_0-\eta`$, and both have the same sign history $`(+,-)`$.

At step 2, the bias-corrected moments are $`\widehat m_2=-\tfrac16`$ and $`\widehat m_2'=+\tfrac16`$, and $`\widehat v_2=\tfrac{17}{24}`$ and $`\widehat v_2'=\tfrac{41}{24}`$. Hence

```math
\theta_2=\theta_1+\frac{\eta}{6\sqrt{17/24}},\qquad\theta_2'=\theta_1-\frac{\eta}{6\sqrt{41/24}}. \qquad(A10)
```

The two second steps move in opposite directions. The bounded test $`F(\theta)=1\lbrace \theta>\theta_1\rbrace`$ separates the histories by $`1`$.

The histories are realizable by squared losses $`\tfrac12(\theta-a)^2`$ at $`\theta_0=0`$: first targets $`a=-1`$ and $`a=-2`$, then the common second target $`a=\theta_1+\tfrac34`$.

**(e′)** **Corollary (momentum carries the memory).** If $`\beta_1=0`$, any two histories with the same sign of $`G_2`$ have second steps in the same direction, so no witness of the (e) type exists. For every $`\beta_1\in(0,1)`$, the (e) mechanism is available.

## Proof

**Proof of (a).** Cold moments give $`m_1=(1-\beta_1)G`$ and $`v_1=(1-\beta_2)G^{\odot2}`$. Dividing by the bias corrections at $`t=1`$ gives $`\widehat m_1=G`$ and $`\widehat v_1=G^{\odot2}`$, so $`\sqrt{\widehat v_1}=\lvert G\rvert`$ coordinatewise. Substituting into (A6) gives $`\theta_1=c-\eta\,G/(\lvert G\rvert+\epsilon)`$. At $`\epsilon=0`$ the declared convention gives $`h_0=\mathrm{sign}`$.

**Item 2.** The map $`\sigma\mapsto c-\eta\sigma`$ from $`\lbrace -1,0,1\rbrace ^p`$ to $`\mathbb R^p`$ is injective, because $`\eta>0`$. Hence $`\theta_1`$ is an injective function of the sign vector $`\sigma(G)`$.

- *Sufficiency.* If the joint sign laws coincide, the laws of $`\theta_1`$ coincide, and so do all expected utilities.
- *Necessity.* For each attainable pattern $`\sigma^\star`$, the indicator $`F=1\lbrace \theta_1=c-\eta\sigma^\star\rbrace`$ is a bounded utility with expectation $`\Pr(\sigma(G)=\sigma^\star)`$. Equal expectations for all such $`F`$ force equal pattern probabilities.

*Dimension.* View every utility as a function on $`\mathcal A`$, namely $`u_F(g)=F(c-\eta h_0(g))`$. The family $`\mathcal U=\lbrace u_F\rbrace`$ consists exactly of the functions on $`\mathcal A`$ that are constant on the fibres of $`\sigma`$. It contains these functions because $`F`$ is arbitrary on the $`s`$ distinct output points, and contains nothing else because $`u_F`$ depends on $`g`$ only through $`\sigma(g)`$. So $`\mathrm{span}(\lbrace 1\rbrace \cup\mathcal U)`$ is spanned by the $`s`$ fibre indicators and has dimension $`s`$.

By [N06(a)](N06_sufficiency_duality.md) (node [N06](N06_sufficiency_duality.md)), the minimum number of linear statistics whose equal means imply equal means of every $`u\in\mathcal U`$, for all laws on $`\mathcal A`$, is $`\dim\mathrm{span}(\lbrace 1\rbrace \cup\mathcal U)-1=s-1`$. The indicators of $`s-1`$ patterns attain it. Each coordinate sign takes at most three values, so $`s\le3^p`$, and $`s\le2^p`$ when $`0`$ is excluded.

**Item 3.** Fix $`\epsilon>0`$. On $`g\ge0`$, $`h_\epsilon(g)=g/(g+\epsilon)`$ is continuous and strictly increasing from $`0`$ to $`1`$. By oddness, $`h_\epsilon`$ is a bijection $`\mathbb R\to(-1,1)`$.

Solve $`u=g/(\lvert g\rvert+\epsilon)`$. Then $`\mathrm{sign}u=\mathrm{sign}g`$ and $`\lvert u\rvert(\lvert g\rvert+\epsilon)=\lvert g\rvert`$, so $`\lvert g\rvert=\epsilon\lvert u\rvert/(1-\lvert u\rvert)`$. Hence $`g=\epsilon u/(1-\lvert u\rvert)`$. For $`u>0`$, the derivative is $`\epsilon/(1-u)+\epsilon u/(1-u)^2=\epsilon/(1-u)^2`$, and by symmetry the same formula holds for $`u<0`$.

Since $`g\mapsto c-\eta h_\epsilon(g)`$ is injective, the argument of item 2 applies with fibres equal to singletons. Equality for all bounded utilities is then equivalent to equality of the laws of $`G`$, and on $`\mathcal A`$ the dimension is $`m-1`$.

**Item 4.** Under both laws each coordinate equals $`\pm1`$ with probability $`1/2`$, and $`\mathbb E h_0(G)=0`$. Under $`P`$ the two coordinates of $`\theta_1-c=-\eta\sigma`$ always share a sign, so $`F=1`$. Under $`Q`$ they never do, so $`F=0`$. $`\square`$

**Proof of (d).** Fix a coordinate $`i`$ and a gradient value $`g`$.

- If $`g_i=0`$, then $`h_\epsilon(g)_i=0=h_0(g)_i`$.
- If $`g_i\ne0`$, then $`\lvert h_\epsilon(g_i)-\mathrm{sign}(g_i)\rvert=\lvert g_i\rvert\cdot\big\lvert\tfrac1{\lvert g_i\rvert+\epsilon}-\tfrac1{\lvert g_i\rvert}\big\rvert=\tfrac{\epsilon}{\lvert g_i\rvert+\epsilon}`$.

The last quantity is at most $`1`$ always, and at most $`\epsilon/(a+\epsilon)`$ when $`\lvert g_i\rvert\ge a`$. Hence

```math
\mathbb E_P\lvert h_\epsilon(G_i)-h_0(G_i)\rvert\le\beta_{P,i}(a)+\frac{\epsilon}{a+\epsilon}.
```

By the Lipschitz hypothesis applied pointwise to the pair $`\big(c-\eta h_\epsilon(G),\,c-\eta h_0(G)\big)`$,

```math
\big\lvert\mathbb E_PF(c-\eta h_\epsilon(G))-\mathbb E_PF(c-\eta h_0(G))\big\rvert\le\eta\sum_iL_i\Big[\beta_{P,i}(a)+\frac{\epsilon}{a+\epsilon}\Big],
```

and the same holds under $`Q`$. The triangle inequality through the two ideal-sign expectations gives (A9).

For the TV statement, note that $`F(c-\eta h_0(G))=f(\sigma(G))`$ for some $`f:\lbrace -1,0,1\rbrace ^p\to[0,1]`$. Any such $`f`$ satisfies $`\lvert\mathbb E_Pf-\mathbb E_Qf\rvert\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$. $`\square`$

**Proof of (e).** Injectivity of $`m_1=(1-\beta_1)G`$ is immediate for $`\beta_1<1`$.

For the counterexample, $`1-\beta_1=1-\beta_2=\tfrac12`$, and the bias corrections are $`\tfrac12`$ at $`t=1`$ and $`\tfrac34`$ at $`t=2`$.

- *Step 1.* $`m_1=G_1/2`$ and $`v_1=G_1^2/2`$, so $`\widehat m_1=G_1`$ and $`\widehat v_1=G_1^2`$. Since $`G_1>0`$ in both histories, $`\theta_1=\theta_0-\eta`$.
- *Step 2.* $`m_2=\tfrac12m_1+\tfrac12G_2=\tfrac{G_1}4-\tfrac38`$, which is $`-\tfrac18`$ for $`G_1=1`$ and $`+\tfrac18`$ for $`G_1=2`$. Then $`\widehat m_2=\tfrac43m_2=\mp\tfrac16`$.
- Similarly $`v_2=\tfrac12v_1+\tfrac12G_2^2=\tfrac{G_1^2}4+\tfrac9{32}`$, which is $`\tfrac{17}{32}`$ or $`\tfrac{41}{32}`$. So $`\widehat v_2=\tfrac43v_2=\tfrac{17}{24}`$ or $`\tfrac{41}{24}`$.
- With $`\epsilon=0`$, $`\theta_2=\theta_1-\eta\widehat m_2/\sqrt{\widehat v_2}`$, which is (A10). The denominators are positive and the numerators have opposite signs, so the steps are opposite. Then $`F(\theta_2)=1`$ and $`F(\theta_2')=0`$.

*Realizability.* The gradient of $`\tfrac12(\theta-a)^2`$ is $`\theta-a`$. At $`\theta_0=0`$, the targets $`a=-1,-2`$ give $`G_1=1,2`$. Both first steps land at the same $`\theta_1=-\eta`$, so the common target $`a=\theta_1+\tfrac34`$ gives $`G_2=-\tfrac34`$ in both histories. No branch-dependent future gradient is needed.

*Positive $`\epsilon`$.* For the same gradient sequences, the direction of the second step is $`-\mathrm{sign}(\widehat m_2)`$ for every $`\epsilon\ge0`$, because the denominator $`\sqrt{\widehat v_2}+\epsilon`$ is positive. The two directions therefore stay opposite. $`\square`$

**Proof of (e′).** If $`\beta_1=0`$ then $`m_2=G_2`$. The second step is $`-\eta\,G_2/(1-\beta_1^2)\big/(\sqrt{\widehat v_2}+\epsilon)`$ with a positive denominator, so its sign is $`-\mathrm{sign}(G_2)`$ whatever the first gradient was. If $`\beta_1>0`$, then $`m_2=\beta_1(1-\beta_1)G_1+(1-\beta_1)G_2`$. Choosing $`G_2<0`$ with $`\lvert G_2\rvert`$ strictly between $`\beta_1G_1`$ and $`\beta_1G_1'`$ (for $`0<G_1<G_1'`$) gives opposite signs of $`m_2`$, while the first steps $`-\eta\mathrm{sign}(G_1)`$ coincide at $`\epsilon=0`$. $`\square`$

## Remarks

1. **Three levels of sufficiency.** Immediate-parameter sufficiency of a cold step needs the joint sign law at $`\epsilon=0`$, and the full law at $`\epsilon>0`$. Complete-state sufficiency needs the full consumed-gradient law, because $`m_1`$ is injective. Task sufficiency needs only the response class of the actual continuation (node [N05](N05_continuation_response.md)), which may be much smaller. The exact statistic counts concern unrestricted laws; a restricted task family can need fewer.
2. **Discontinuity at $`\epsilon=0`$ is exact, not operational.** For arbitrary measurable utilities, any $`\epsilon>0`$ makes magnitudes identifiable. For Lipschitz utilities with little gradient mass near zero, (A9) shows that the sign approximation is accurate. The inverse (A8) is ill-conditioned as $`\lvert u\rvert\to1`$.
3. **Warm states move the threshold.** At a warm state the numerator of coordinate $`i`$ changes sign at $`G_i=-\beta_1m_i/(1-\beta_1)`$, and the denominator depends on $`v_i`$ and $`G_i^2`$. For a scalar cross-entropy gradient $`G=p-Z`$, the cold sign probabilities are the masses of $`Z`$ below, at and above the current prediction $`p`$. Allowing all thresholds identifies the CDF of $`Z`$. No fixed threshold or marginal sign feature is sufficient across changing states.
4. **Consequence for the affine theory.** Two label laws can match all moments below order $`N`$ (node [N07](N07_affine_moment_order.md)) and still differ in their sign probabilities about the current prediction. A cold Adam step separates them at first order in $`\eta`$. Node [N12](N12_optimizer_warmth.md) quantifies when this effect is present and how it fades as the optimizer warms.

## Dependencies and source

- Uses: [N06](N06_sufficiency_duality.md) ([N06(a)](N06_sufficiency_duality.md) dimension count), [N10](N10_update_composition.md) (learner-dependence of the information requirement).
- Used by: [N12](N12_optimizer_warmth.md).

**Source:** [v0.15 §3](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#3-what-adam-requires-three-distinct-levels-of-sufficiency) — Props. A2–A4; eqs. (A6)–(A10). v0.15 addendum [Optimizer_State_and_Few_Shot_Theory_v015_EN.md](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md) §3, Propositions A2–A4, eqs. (A6)–(A10); [Proof_Review_v015_EN.md](../../docs/direction/v015_revision/Proof_Review_v015_EN.md) §V15.2. Corollary A4′ is added here; it explains the $`\beta`$-grid result of check V4.
