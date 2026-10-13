# Proposition 5.3 · Cold-start Adam: sign quotient and optimizer memory

*Experiments*

[← Framework](../README.md) · [Statement](../statements/05-03_cold_start_adam_sign_quotient.md) · [Proof](../proofs/05-03_cold_start_adam_sign_quotient.md) · [Experiments](../experiments/05-03_cold_start_adam_sign_quotient.md) · [中文](05-03_cold_start_adam_sign_quotient_ZH.md)

**What is tested here.** Five checks test this proposition: the cold map and exact quotient dimensions on random alphabets ([Check 5.3](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-3)), the margin bound ([Check 5.4](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-4)), the optimizer-memory witness ([Check 5.5](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-5)), and two controlled-network checks showing that the Adam gap between moment-matched laws is a cold-start sign effect rather than feature learning ([Checks 5.6](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-6) and [5.7](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-7)).

**Verdict.**

- **Proposition 5.3 (cold-start Adam).** All five parts are confirmed exactly.
  - The quotient dimensions are $`s-1`$ and $`m-1`$ on 300/300 alphabets.
  - The margin bound comes within 0.994 of being attained.
  - The optimizer-memory witness has opposite second steps for every $`\epsilon`$ tested and in every $`(\beta_1,\beta_2)`$ cell with $`\beta_1>0`$.
  - In the controlled network, the Adam gap between laws with equal gradient moments survives a frozen Jacobian and vanishes when sign disagreement is removed.

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

**Sign map.** For $`\epsilon>0`$, $`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$. For $`\epsilon=0`$, the *ideal sign update* is $`h_0(g)_i=\mathrm{sign}(g_i)`$, with $`\mathrm{sign}(0)=0`$. This convention is part of the definition; a literal floating-point Adam with $`\epsilon=0`$ would compute $`0/0`$.

**Utilities.** An *immediate-parameter utility* is a bounded measurable function of $`\theta_1`$. A *complete-state utility* is a bounded measurable function of the next complete state $`(\theta_1,m_1,v_1,1)`$. Consumed-gradient laws $`P,Q`$ are probability laws on $`\mathbb R^p`$. In the finite statements they are supported on a finite alphabet $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$. Put $`\bar\theta=(1-\eta\lambda_{\rm wd})\theta_0`$.

**Status labels.** `PASS` means the implemented consistency assertions passed. `INFO` marks an empirical diagnostic: each cell gets its own PASS, FAIL or UNAVAILABLE status, and the group as a whole makes no pass/fail claim. Unless a check says otherwise, every network is the controlled small network of `v014:C23`: one-hot context, 4 tanh units, a sigmoid output, mean cross-entropy, float64. None of these networks is a real model.

**Statement under test ([Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)).** Start Adam from a cold state $`(\theta_0,0,0,0)`$.

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

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.3](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-3) | [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)(a)–(c) | `v015:V2`, `pkg:T03`–`pkg:T06` | PASS |
| [Check 5.4](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-4) | [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)(d) | `v015:V3`, `pkg:T07` | PASS |
| [Check 5.5](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-5) | [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)(e) | `v015:V4`, `v015:E3`, `pkg:T08` | PASS |
| [Check 5.6](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-6) | [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)(a)–(c) in a network | `v015:E0`, `v015:E1` | PASS |
| [Check 5.7](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-7) | [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3) versus feature learning | `v015:E2`, `v014:C23` (Adam settings) | INFO; gap persists in 39/45 frozen-Jacobian cells |

<a id="c5-3"></a>

## Check 5.3 · Cold map, exact quotient dimensions, inverse conditioning (Proposition 5.3(a)–(c))

**Verdict.** On 300 random finite gradient alphabets:

- the cold Adam step equals $`c-\eta h_\epsilon(G)`$ to $`4.4\times10^{-16}`$;
- exactly $`s-1`$ statistics are necessary and sufficient at $`\epsilon=0`$, and exactly $`m-1`$ at $`\epsilon>0`$;
- the derivative of the inverse map reaches $`10^{22}`$.

**Why.** [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)(a)–(c) makes three claims, and each is a finite statement that can be checked exactly:

- an exact formula for the first step;
- the immediate-parameter information is the joint sign law (dimension $`s-1`$) at $`\epsilon=0`$, and the full law (dimension $`m-1`$) at $`\epsilon>0`$;
- coordinate sign marginals are not sufficient.

**Setup.**

- 300 random finite gradient alphabets with $`p=1,\dots,4`$ coordinates, including zero coordinates.
- An independent recurrence for part (a), checked in float64 and, where the square root is exact, in exact rationals.
- Exact ranks and null spaces. When fewer than $`s-1`$ features are used, HiGHS LPs construct separating laws.
- Inverse round-trips on a $`u\to1`$ grid with $`\epsilon=1/100`$.

**Procedure.**

- Compare the first step with $`c-\eta h_\epsilon(G)`$.
- Count the attainable sign patterns $`s`$, and verify that $`s-1`$ pattern indicators suffice.
- For any smaller feature set, exhibit a zero-mass perturbation and the two laws it separates.
- Repeat with $`\epsilon>0`$, which gives dimension $`m-1`$.
- Evaluate the joint-versus-marginal example.

**Result** (`v015:V2`).

| Quantity | Value |
|---|---|
| status, instances, assertions, worst residual | PASS, 300, 3,288, $`4.44\times10^{-16}`$ |
| quotient dimension $`=s-1`$ at $`\epsilon=0`$ | 300/300 alphabets (1 to 14 atoms; $`s`$ from 1 to 14) |
| dimension $`=m-1`$ at $`\epsilon>0`$ | 300/300 |
| example separation (`quotients[0]`) | 4 atoms, 4 patterns, 2 features: $`P=(\tfrac14,\tfrac18,\tfrac38,\tfrac14)`$ and $`Q=(\tfrac14,\tfrac38,\tfrac18,\tfrac14)`$ match both features, but their pattern indicators differ by $`-\tfrac14`$ |
| inverse derivative, $`\epsilon=1/100`$ | $`u=0.9`$: 1; $`u=0.999`$: $`10^4`$; $`u=1-10^{-12}`$: $`10^{22}`$ |
| joint versus marginal | identical marginals; same-sign utility 1 against 0 |

Independent package implementation (`docs/direction/v015_revision/results_v015.json`, `checks`), all PASS:

- `pkg:T03`: 480 cases; update error $`\le2.22\times10^{-16}`$.
- `pkg:T04`: 25 atoms and 9 sign patterns; statistic dimensions 8 and 24.
- `pkg:T05`: equal mean updates; same-sign utility gap 1.0.
- `pkg:T06`: label laws matching orders 0–2. The SGD mean parameter gap is 0, the ideal-sign Adam gap is $`1/10`$, and the finite-$`\epsilon`$ gap is $`250/2703`$.

**Why it supports Proposition 5.3(a)–(c).**

- At $`\epsilon=0`$, the exact information content of a cold Adam step is the joint sign quotient; at $`\epsilon>0`$ it is the full gradient law. The counts are exactly $`s-1`$ and $`m-1`$.
- The inverse at $`\epsilon>0`$ exists but is badly conditioned. In practice, then, the magnitude information is nearly invisible.
- `pkg:T06` shows the same thing at the level of means: two laws that SGD cannot tell apart at first order are told apart by Adam.

<a id="c5-4"></a>

## Check 5.4 · Margin-sensitive approximation (Proposition 5.3(d))

**Verdict.** The margin bound holds on 1,660 exact instances. Adversarial search reaches a gap/bound ratio of 0.994, so the bound is nearly tight.

**Why.** Part (d) says when the sign quotient is a good *approximate* description for regular utilities. It connects the exact statement, that any $`\epsilon>0`$ identifies magnitudes, with practice.

**Setup.**

- Random finite gradient laws $`P,Q`$.
- Coordinatewise-Lipschitz utilities $`F\in[0,1]`$, built as clipped sums of piecewise-linear coordinate functions with known $`L_i`$.
- Random margins $`a`$ and random $`\epsilon`$.
- Exact rational arithmetic throughout.
- Adversarial restarts with hill-climbing over $`(G,P,Q,L,a,\epsilon,\eta)`$.

**Procedure.** On every instance, assert the margin bound and $`\Delta_{\rm sign}\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$. Then maximize gap/bound.

**Result** (`v015:V3`).

| Quantity | Value |
|---|---|
| status, instances, assertions, worst residual | PASS, 1,660, 3,320, 0 |
| largest gap/bound, adversarial | 0.9941 |
| largest gap/bound, random | 0.837 |
| largest $`\Delta_{\rm sign}/\mathrm{TV}`$ found | 0.274 |

`pkg:T07`: 1,500 instances, maximum gap/bound 0.739, no violation.

**Why it supports Proposition 5.3(d).** The bound is valid and nearly attained. For regular utilities the sign quotient is accurate, up to two terms: the gradient mass near zero and the $`\epsilon/(a+\epsilon)`$ terms.

<a id="c5-5"></a>

## Check 5.5 · Magnitude-bearing optimizer memory (Proposition 5.3(e))

**Verdict.** The witness is exact:

- $`\widehat m_2=\mp\tfrac16`$ and $`\widehat v_2=\tfrac{17}{24},\tfrac{41}{24}`$, so the second steps point in opposite directions.
- It persists for every $`\epsilon`$ from 0 to $`10^{-1}`$.
- It exists in every $`(\beta_1,\beta_2)`$ grid cell with $`\beta_1>0`$, and in none with $`\beta_1=0`$.

The same pair runs through the network optimizer, and a network witness exists.

**Why.** Part (e) claims that matching the first-step parameter information is not enough for a learning continuation, because $`m_1`$ keeps magnitudes. For this to count, the counterexample must be exact, realizable by real losses, robust to $`\epsilon`$, and present in the actual optimizer code.

**Setup.**

- *Exact (`v015:V4`).*
  - Fraction and symbolic arithmetic for the gradient histories $`(1,-\tfrac34)`$ and $`(2,-\tfrac34)`$, with $`\beta_1=\beta_2=\tfrac12`$.
  - A squared-loss realization.
  - $`\epsilon\in\lbrace 0,10^{-1},\dots,10^{-12}\rbrace`$.
  - A $`5\times5`$ grid $`\beta_1,\beta_2\in\lbrace 0,\tfrac1{10},\tfrac12,\tfrac9{10},\dots\rbrace`$, with 300 search trials per cell.
- *Network (`v015:E3`).*
  - The same pair through the scalar optimizer of [Checks 5.6](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-6)–[5.7](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-7), at $`\epsilon=0`$ and $`\epsilon=10^{-8}`$.
  - A soft-label search for same-sign histories with opposite second updates.

**Procedure.**

- Compute every moment and both second steps, and check that the two steps have opposite signs.
- In the network, search for label pairs $`(y_1,y_2)`$ and $`(y_1',y_2)`$ whose first steps coincide but whose second updates are opposite on some coordinate.

**Result** (`v015:V4`, `v015:E3`).

| Quantity | Value |
|---|---|
| `v015:V4` status, instances | PASS, 39 |
| exact moments | $`m_2=-\tfrac18,+\tfrac18`$; $`\widehat m_2=-\tfrac16,+\tfrac16`$; $`\widehat v_2=\tfrac{17}{24},\tfrac{41}{24}`$ |
| second steps at $`\eta=1`$ | $`+\sqrt{102}/51=+0.198`$ and $`-\sqrt{246}/123=-0.128`$ |
| squared-loss realization | first targets $`-1`$ and $`-2`$; common second target $`-\tfrac14`$ |
| opposite directions at positive $`\epsilon`$ | all 14 values from $`0`$ to $`10^{-1}`$ |
| $`\beta`$-grid cells with a witness | 20/25: every cell with $`\beta_1>0`$, no cell with $`\beta_1=0`$ |
| `v015:E3` scalar trace, $`\epsilon=0`$ and $`10^{-8}`$ | second updates $`+0.19803`$ and $`-0.12752`$ |
| `v015:E3` network witnesses | found at $`\epsilon=0`$ and $`10^{-8}`$: labels $`(0,0.7)`$ and $`(0.1,0.7)`$ give identical first steps and opposite second steps on the output bias |

`pkg:T08`: equal first outputs and equal sign histories. The second increments are $`+0.019803`$ and $`-0.012752`$ at $`\eta=0.1`$, and the threshold-task gap is 1.

**Why it supports Proposition 5.3(e).**

- The counterexample is exact, realizable, robust to $`\epsilon`$, and present in a real network.
- The $`\beta`$ grid isolates the mechanism. The first moment $`m`$ carries the magnitude memory, so the effect exists exactly when $`\beta_1>0`$.

<a id="c5-6"></a>

## Check 5.6 · In a network, the Adam gap is the cold-start sign quotient (Proposition 5.3(a)–(c))

**Verdict.**

- Two label laws agree on all moments through order 2 and have identical gradient moments through order 2. Yet their expected first Adam updates differ by $`\eta`$ per coordinate, and Adam's return gap at $`\eta=0.1`$ is 0.0769 ($`R=1`$) and 0.1196 ($`R=4`$).
- Removing sign disagreement removes the gap entirely. Five warm moment steps reduce it to 0.0034 and 0.0098.

**Why.** In the controlled network, Adam separates label laws that agree on all moments through order $`N-1`$, and SGD does not. [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)(a)–(c) predicts the mechanism: cold Adam responds to the sign law of the gradient $`G=p-Z`$ around the current prediction, not to moments.

- `v015:E0` fixes an exact reproduction of the gap.
- `v015:E1` traces the chain from input moments to gradient moments, then to the normalized update, then to the return. It then ablates one factor at a time.

**Setup.**

- Adam with $`\beta=(0.9,0.999)`$, $`\epsilon=10^{-8}`$ outside the square root, and bias correction on.
- Laws: `law_scalar(N)` matches the label moments of orders $`1,\dots,N-1`$ only.
- Gap: $`\mathbb E_P[p_R^N]-\mathbb E_Q[p_R^N]`$, by complete enumeration of finite histories (standard error 0).
- Ablations (moment $`N=3`$, $`\eta=0.1`$):
  - $`\epsilon`$ inside the square root;
  - $`\epsilon\in\lbrace 10^{-6},10^{-4},10^{-2},10^{-1}\rbrace`$;
  - no bias correction;
  - five warm moment steps on a third law;
  - global clip 0.1;
  - sum reduction;
  - a support shifted so that all first gradients share one sign.

**Procedure.**

- Enumerate the per-support gradients, gradient moments, finite-$`\epsilon`$ updates and their expectations.
- Compare them with the executed network step at relative tolerance $`10^{-12}`$.
- Rerun each ablation from the same snapshot.

**Result** (`v015:E0`, `v015:E1`, `theory/v015_checks/c23_bundle.json`).

| Reproduction (moment $`N=3`$) | $`\eta=0.01`$ | $`\eta=0.1`$ |
|---|---|---|
| Adam gap, $`R=1`$ | 0.00757 | 0.07691 |
| Adam gap, $`R=4`$ | 0.01892 | 0.11961 |
| affine-SGD comparator, $`R=1`$ / $`R=4`$ | $`-6.3\times10^{-9}`$ / $`-2.5\times10^{-8}`$ | $`-6.3\times10^{-6}`$ / $`-2.1\times10^{-5}`$ |

The chain for moment $`N=3`$ at $`\eta=0.1`$ (`data.transitions`):

- Input moment differences of orders 1, 2, 3: $`0,0,-0.0556`$.
- Joint gradient moments: equal through order 2 (to $`3\times10^{-17}`$); order 3 differs by 0.056.
- Expected first update: $`+0.05`$ against $`-0.05`$ per coordinate, a difference of $`\eta`$.
- Return difference: 0.0769.
- Under $`P`$ the label is $`0`$ with probability $`\tfrac14`$ and $`\tfrac23`$ with probability $`\tfrac34`$, so the output-bias gradient $`p-Z`$ is positive with probability $`\tfrac14`$.
- Under $`Q`$ the label is $`\tfrac13`$ with probability $`\tfrac34`$ and $`1`$ with probability $`\tfrac14`$, so the same gradient is positive with probability $`\tfrac34`$.
- The measured map error is $`2.1\times10^{-15}`$ relative.

| Ablation (moment $`N=3`$, $`\eta=0.1`$) | gap $`R=1`$ | gap $`R=4`$ |
|---|---|---|
| baseline | 0.0769 | 0.1196 |
| $`\epsilon`$ inside sqrt; $`\epsilon=10^{-6}`$; sum reduction | 0.0769 | 0.1196 |
| $`\epsilon=10^{-4}`$ | 0.0767 | 0.1197 |
| $`\epsilon=10^{-2}`$ | 0.0610 | 0.1148 |
| $`\epsilon=10^{-1}`$ | 0.0211 | 0.0446 |
| no bias correction | 0.2477 | $`-0.0081`$ |
| five warm moment steps | **0.0034** | **0.0098** |
| global clip 0.1 | 0.0769 | 0.1840 |
| shifted support (one first-gradient sign) | **0.0000** | **0.0000** |

The shifted support keeps the matched label moments: orders 1 and 2 are still equal.

**Why it supports Proposition 5.3.**

- Two laws with identical gradient moments through order 2 give first Adam updates that differ by $`\eta`$ per coordinate. The cause is exactly the sign quotient: the probability that the label lies above the current prediction is $`1/4`$ under one law and $`3/4`$ under the other.
- Removing sign disagreement removes the gap. Warming the moments removes most of it, as [Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean.md#s5-7) and [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6) predict.
- The gap is insensitive to the $`\epsilon`$ convention until $`\epsilon`$ becomes comparable to the gradient magnitudes ($`10^{-2}`$), as part (d) predicts.

<a id="c5-7"></a>

## Check 5.7 · Optimizer effect, not feature learning (Proposition 5.3)

**Verdict.** An Adam that cannot learn features, because its Jacobian is frozen, reproduces the gap almost exactly: 0.0757 against 0.0769 at $`R=1`$, and 0.1225 against 0.1196 at $`R=4`$. The frozen-Jacobian gap persists in 39 of 45 cells. Over the declared Adam settings, the tangent affine-SGD comparator misses the gap in 32 of 42 configurations.

**Why.** A gap against an affine-SGD comparator could come from the optimizer, from nonlinear features, or from both. Freezing the network Jacobian, while keeping the sigmoid, the cross-entropy and the same Adam recurrence, isolates the optimizer.

**Setup** (`v015:E2`).

- Five branches from one snapshot, with identical within-source histories:
  - the actual network with Adam;
  - a frozen Jacobian, $`\mathrm{logit}(\theta)=\mathrm{logit}(\theta_0)+J_0(\theta-\theta_0)`$, with Adam and its own moments;
  - the actual network with SGD;
  - the frozen Jacobian with SGD;
  - tangent affine SGD.
- Grid: 5 tasks × $`\eta\in\lbrace 0.01,0.1,1\rbrace`$ × $`R\in\lbrace 1,4,40\rbrace`$ = 45 cells.
- The derivative checks of $`J_0`$ have residuals $`\le2.6\times10^{-11}`$.
- A frozen-Adam gap *persists* if its magnitude exceeds $`\max(10^{-10},3\,\mathrm{SE})`$.

**Result** (`v015:E2`, `data.comparisons`; status INFO).

| Cell | actual Adam | frozen-Jacobian Adam | actual SGD | tangent SGD |
|---|---|---|---|---|
| moment $`N=3`$, $`\eta=0.1`$, $`R=1`$ | 0.0769 | 0.0757 | 0.0000 | 0.0000 |
| moment $`N=3`$, $`\eta=0.1`$, $`R=4`$ | 0.1196 | 0.1225 | 0.0000 | 0.0000 |
| moment $`N=3`$, $`\eta=0.01`$, $`R=4`$ | 0.0189 | 0.0188 | 0.0000 | 0.0000 |
| parity $`N=2`$, $`\eta=0.1`$, $`R=4`$ | 0.0762 | 0.0774 | 0.0034 | 0.0034 |
| moment $`N=4`$, $`\eta=0.1`$, $`R=4`$ | $`-0.0218`$ | $`-0.0223`$ | $`-0.0000`$ | 0.0000 |

- The frozen-Jacobian Adam gap persists in 39 of 45 cells.
- Across all cells with $`\eta\le0.1`$ and $`R\le4`$, the three SGD branches agree with one another to within $`2.8\times10^{-4}`$, and none exceeds 0.0034.

The affine comparator under Adam, from `theory/independent_checks/results_independent_v014.json`, `controlled_neural.details.observed_affine_comparison_cohorts`. The settings are $`\beta=(0.9,0.999)`$, $`\epsilon=10^{-8}`$, bias correction, $`\eta\in\lbrace 0.01,0.1\rbrace`$, $`R\in\lbrace 1,4,40\rbrace`$, over 7 task laws:

| Cohort | Configurations | Agreements | Departures | Largest deviation |
|---|---:|---:|---:|---|
| Adam, all declared settings | 42 | 10 | 32 | 0.1196 (moment $`N=3`$, $`\eta=0.1`$, $`R=4`$) |

![Figure 5.2: cold-start Adam](../figures/fig5_2_cold_adam.png)

*Figure 5.2.*
- (A) Cold Adam separates laws whose first two gradient moments agree. A frozen-Jacobian Adam reproduces the gap, while SGD shows none.
- (B) Removing sign disagreement (shifted support) removes the gap, and warm moments remove most of it.
- (C) Two histories with identical sign sequences take opposite second steps, because $`m`$ retains magnitudes.

[PDF](../figures/fig5_2_cold_adam.pdf)

**Why it supports Proposition 5.3.**

- The gap is an optimizer-state effect, as the proposition states.
- In this network, moment matching ([Theorem 4.4](../statements/04-04_affine_learners_moment_order.md#s4-4)) controls SGD but not cold Adam. The part that the affine-SGD prediction misses is the cold-start quotient of parts (a)–(c) together with the momentum memory of part (e).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)). The linear programs of `v015:V2` and `v015:V8` are solved by `scipy.optimize.linprog` with the HiGHS solver ([Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)).

## Artifacts

- `theory/v015_checks/results_v015_independent.json`:
  - `v015:V8`, and `v015:E10` (per-cell records in `data.cells`; summaries in `data.optimizer_summary`; the cold-Adam cell in `data.C23_prediction`; the design in `data.design`);
  - `v015:V2`, `v015:V3`, `v015:V4`, `v015:E0`, `v015:E1`, `v015:E2`, `v015:E3`;
  - `v015:E8`: the frozen-denominator, warm-affine, cold-separation and warmth-trajectory parts (the last with `per_step` and `fits`), and the 210 `data.gaps` cells;
  - `v015:E11` (`data.clipped_mean`, `data.always_active`, `data.warm_Adam`, `data.Lion`, `data.LoRA`).
- `theory/v015_checks/c23_bundle.json`: exact laws, network, optimizer code and per-history raw steps.
- `theory/v015_checks/raw/E8_trajectories.npz`: per-replica warmth trajectories. It is gitignored and regenerated by the first command; its SHA-256 is recorded as `raw_sha256` in the warmth-trajectory part of `v015:E8`.
- `theory/v016_checks/results_v016.json`: claims `v016:C1.exact`, `v016:C1.moments` and `v016:C1.accumulation`. The full arrays are in `theory/v016_checks/results_v016_raw.json.gz`, which is gitignored; its SHA-256 is in `raw_artifact`.
- `theory/independent_checks/results_independent_v014.json`: `controlled_neural`.
- `docs/direction/v015_revision/results_v015.json`: package checks `pkg:T03`–`pkg:T08`.
- Code:
  - `theory/v015_checks/otlr.py` (`check_V8`, `check_E10`, `check_E11`);
  - `theory/v015_checks/warmth.py` (`check_E8`);
  - `theory/v016_checks/check_v016.py` (`v016:C1`);
  - specifications in `theory/v015_checks/README.md` and `theory/v016_checks/SPEC.md`.
- Figures: `../figures/fig5_1_linear_response`, `fig5_2_cold_adam`, `fig5_3_warmth` and `fig5_4_learner_statistics`, each as `.pdf` and `.png`. The specs are in `../figures/specs/fig5_1.md` to `fig5_4.md`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
