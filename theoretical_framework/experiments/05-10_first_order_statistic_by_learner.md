# Theorem 5.10 · The first-order statistic by learner

*Experiments*

[← Framework](../README.md) · [Statement](../statements/05-10_first_order_statistic_by_learner.md) · [Proof](../proofs/05-10_first_order_statistic_by_learner.md) · [Experiments](../experiments/05-10_first_order_statistic_by_learner.md) · [中文](05-10_first_order_statistic_by_learner_ZH.md)

**What is tested here.** [Checks 5.16](../experiments/05-10_first_order_statistic_by_learner.md#c5-16)–[5.18](../experiments/05-10_first_order_statistic_by_learner.md#c5-18) are catalog witnesses for warm Adam, Lion and LoRA. [Checks 5.14](../experiments/05-09_partially_active_clipping.md#c5-14) and [5.15](../experiments/05-09_partially_active_clipping.md#c5-15), whose full descriptions are on the page of Proposition 5.9, cover clipping. [Check 5.2](../experiments/05-01_optimizer_transformed_expansion.md#c5-2), on the page of Proposition 5.1, covers the cold-Adam row in a controlled network.

**Verdict.**

- **Proposition 5.9 and Theorem 5.10(e) (clipping).**
  - Equal raw means with different clipped means give a first-order gap. Its remainder has slope 2.0015.
  - With clipping always active ($`p_{\rm act}=1`$), equal mean directions give a first-order term of exactly 0.
  - The partial-activation bound of Proposition 5.9 is a two-line inequality and has no separate numerical check.
- **Theorem 5.10 (the first-order statistic by learner).** The five catalog witnesses pass with 105 assertions and worst residual $`8.0\times10^{-21}`$.
  - Warm Adam: at equal gradient means, the first-order term is carried entirely by the $`v`$ block, as predicted to $`10^{-12}`$.
  - Lion: under momentum dominance, the parameter block is exactly blind to the message.
  - LoRA: at $`B=0`$, null-space differences are second order and range differences are first order.

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

**Optimizers.** All operations are coordinatewise. Terms of $`\Delta`$ that do not depend on the message cancel in $`\mu^P-\mu^Q`$: weight decay, momentum decay and bias-correction constants applied to the old state.

- *SGD:* $`\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta G`$.
- *Heavy-ball:* $`m^+=\beta m+G`$, $`\theta^+=\theta-\eta m^+`$. *Nesterov:* $`m^+=\beta m+G`$, $`\theta^+=\theta-\eta(G+\beta m^+)`$.
- *Adam and AdamW:* as in the Setting, with the warm-Adam notation of Definition 5.4 and $`c_1=(1-\beta_1)/(1-\beta_1^{\,t})`$.
- *Global-norm clipping:* $`\mathrm{clip}_c(G)`$, applied before the base optimizer.
- *Lion form of sign optimizers:* $`u=\mathrm{sign}(\beta_1m+(1-\beta_1)G)`$, $`\theta^+=\theta-\eta u`$, $`m^+=\beta_2m+(1-\beta_2)G`$.
- *LoRA:* an adapted weight $`W=W_0+\alpha BA`$ with $`A\in\mathbb R^{r\times k}`$, $`B\in\mathbb R^{d\times r}`$ and full-weight gradient $`\Gamma=\nabla_W\ell`$. The trainable parameters are $`(A,B)`$, with the standard initialization $`B_0=0`$ and $`A_0`$ random.

For two laws $`P,Q`$ at a state write $`\delta_1=\mathbb E_PG-\mathbb E_QG`$ and $`\delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}`$. By Proposition 5.1(c), the projection of $`\mu^P-\mu^Q`$ onto the span $`U`$ of the continuation gradients is what two feedback laws must agree on to first order.

**Status labels.** `PASS` means the implemented consistency assertions passed. `INFO` marks an empirical diagnostic: each cell gets its own PASS, FAIL or UNAVAILABLE status, and the group as a whole makes no pass/fail claim. Unless a check says otherwise, every network is the controlled small network of `v014:C23`: one-hot context, 4 tanh units, a sigmoid output, mean cross-entropy, float64. None of these networks is a real model.

**Statement under test ([Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)).**

**(a) SGD.** $`\mu^P-\mu^Q=-\eta\,\delta_1`$. The first-order statistic is the mean consumed gradient; the second-order statistic is $`\mathbb E\,GG^\top`$.

**(b) Heavy-ball and Nesterov momentum.** The increments are affine in $`G`$: heavy-ball has $`\Delta^\theta=-\eta(\beta m+G)`$ and $`\Delta^m=(\beta-1)m+G`$; Nesterov has $`\Delta^\theta=-\eta\big((1+\beta)G+\beta^2m\big)`$ and $`\Delta^m=(\beta-1)m+G`$. Hence the first-order term is as below. The statistic is the mean consumed gradient; momentum changes its weight, not the statistic.

```math
\langle\nabla V,\mu^P-\mu^Q\rangle=\big\langle-\eta\gamma\nabla_\theta V+\nabla_mV,\ \delta_1\big\rangle,\qquad\gamma=1\ \text{(heavy-ball)},\quad\gamma=1+\beta\ \text{(Nesterov)}.
```

**(c) Adam and AdamW at a warm state.** The exact first-order statistic is the mean Adam increment (Proposition 5.5(a)). In the blocks $`(\theta,m,v)`$, with $`e(G)=\theta^+-\tilde\theta^+`$ and $`\lvert e_i\rvert\le\eta u_i\min\lbrace \kappa_i/2,1\rbrace`$ (Proposition 5.5(b)),

```math
\mu^P-\mu^Q=\Big(-\eta c_1\Phi\,\delta_1+\big(\mathbb E_Pe-\mathbb E_Qe\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).
```

- The pair $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ determines $`\mu^P-\mu^Q`$ exactly under the frozen-denominator approximation, and in general up to the frozen-denominator error, at most $`\eta\,(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace \kappa_i/2,1\rbrace]`$ per coordinate. Relative to the step $`\eta u_i`$ the error is at most $`\kappa_i/2`$, and by Proposition 5.8 the warmth of a typical gradient does not vanish along training under a fixed $`\beta_2`$; it levels off near $`(1-\beta_2)/\beta_2`$.
- The coordinatewise second moment enters at first order through $`v`$, with weight $`(1-\beta_2)\nabla_vV`$. If $`\delta_1=0`$, the first-order term equals $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle+\langle\nabla_\theta V,\mathbb E_Pe-\mathbb E_Qe\rangle`$, where the second summand is at most $`\eta\sum_i\lvert\partial_{\theta_i}V\rvert(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace\kappa_i/2,1\rbrace]`$ in absolute value. Equal-mean laws can therefore differ at first order through the second-moment channel, by an amount that is not small in the warmth. [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1) computes $`\nabla_vV`$ by an adjoint recursion.

**(d) Adam at the cold state.** With $`h_\epsilon`$ as in Proposition 5.3,

```math
\mu^P-\mu^Q=\Big(-\eta\big(\mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).
```

The parameter block sees only $`\mathbb E\,h_\epsilon(G)`$, which as $`\epsilon\to0`$ is the sign law. The moment blocks retain $`\mathbb E\,G`$ and $`\mathbb E\,G^{\odot2}`$. If the run ends after this update ($`R=1`$), the continuation is $`F_E(\theta)`$, and the first-order term is $`-\eta\,\langle\nabla_\theta V,\ \mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\rangle`$.

**(e) Gradient clipping.** With clipped SGD, $`\mu^P-\mu^Q=-\eta\big(\mathbb E_P\mathrm{clip}_c(G)-\mathbb E_Q\mathrm{clip}_c(G)\big)`$ exactly, so the statistic is the clipped mean.

- *Witness.* Take $`c=1`$, $`P(G=4e_1)=\tfrac14`$, $`P(G=-\tfrac43e_1)=\tfrac34`$ and $`Q=\delta_0`$. Then $`\delta_1=0`$ but $`\mathbb E_P\mathrm{clip}_1G=-\tfrac12e_1`$, and the first-order term is $`\tfrac\eta2\,\partial_{\theta_1}V`$.
- *Mean direction.* By Proposition 5.9, the clipped mean equals $`c\,\mathbb E[G/\lVert G\rVert]`$ up to $`c(1-p_{\rm act})`$ in norm under each law. Replacing the clipped means by the scaled mean directions therefore changes $`\mu^P_\theta-\mu^Q_\theta`$ by at most $`\eta c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$ in norm.
- *Always-active clipping.* If $`\lVert G\rVert>c`$ almost surely under both laws, the statistic is exactly $`c\,\mathbb E[G/\lVert G\rVert]`$. If in addition the two laws induce the same law of $`G/\lVert G\rVert`$, the increment laws coincide and the laws are equivalent at every order.
- *Clipped base optimizers.* For any base optimizer in (a)–(d) and (f), the statistics are those of the clipped gradient $`\mathrm{clip}_c(G)`$.

**(f) Sign optimizers with momentum (Lion form).** For $`m_i\ne0`$, with a zero counted as half a flip,

```math
\mathbb E\,\Delta^\theta_i=-\eta\,\mathrm{sign}(m_i)\,(1-2\pi_i),\qquad\pi_i=\Pr\lbrace \mathrm{sign}(\beta_1m_i+(1-\beta_1)G_i)\ne\mathrm{sign}(m_i)\rbrace ,
```

and $`\Delta^m=(1-\beta_2)(G-m)`$. The parameter block's statistic is the flip-probability vector $`\pi`$.

- If momentum dominates, i.e. $`(1-\beta_1)\lvert G_i\rvert<\beta_1\lvert m_i\rvert`$ almost surely under both laws, then $`\pi_i=0`$, and coordinate $`i`$'s parameter increment does not depend on the message.
- The $`m`$ block still carries $`(1-\beta_2)\delta_1`$. Its first-order effect $`\langle\nabla_mV,(1-\beta_2)\delta_1\rangle`$ vanishes wherever the continuation is locally constant in $`m`$, that is, where no later update sign changes under small changes of $`m`$.

**(g) LoRA at initialization.** The adapter gradients are $`G_A=\alpha B^\top\Gamma`$ and $`G_B=\alpha\Gamma A^\top`$. At $`B=B_0=0`$:

- $`G_A=0`$ for every message. Under SGD, under momentum started from $`m=0`$, and under cold Adam, the first update leaves $`A`$ unchanged for every law, exactly.
- The $`B`$-block statistic is $`\alpha\,\mathbb E[\Gamma]A_0^\top`$ for SGD and momentum, and $`\mathbb E\,h_\epsilon(\alpha\Gamma A_0^\top)`$ for cold Adam.
- If two laws have mean full-weight gradients that differ only by $`\Delta\Gamma`$ with $`\Delta\Gamma A_0^\top=0`$, they agree at first order in step 1 under SGD and momentum. Under cold Adam, laws whose sketches $`\Gamma A_0^\top`$ have the same sign law give identical first updates as $`\epsilon\to0`$.
- After step 1, $`B\ne0`$ and $`G_A=\alpha B^\top\Gamma`$ carries a second sketch, so the dimension of the statistic grows with training.

**Summary.**

| Part | Learner and state | First-order statistic of the feedback law | Enters through |
|---|---|---|---|
| **(a)** | SGD | $`\mathbb E\,G`$ | $`\theta`$ |
| **(b)** | heavy-ball, Nesterov | $`\mathbb E\,G`$ | $`\theta`$ and $`m`$, with weight $`-\eta\gamma\nabla_\theta V+\nabla_mV`$ |
| **(c)** | Adam/AdamW, warm | exactly the mean Adam increment; $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ under the frozen-denominator approximation, error $`O(\eta\,u\,\kappa)`$ | $`\theta`$ (preconditioned by $`\Phi`$), $`m`$, $`v`$ |
| **(d)** | Adam/AdamW, cold | $`\mathbb E\,h_\epsilon(G)`$ (the sign law as $`\epsilon\to0`$); $`\mathbb E\,G`$ and $`\mathbb E\,G^{\odot2}`$ | $`\theta`$; $`m`$ and $`v`$ |
| **(e)** | clipped SGD | $`\mathbb E\,\mathrm{clip}_c(G)`$ exactly; the mean unit direction up to $`c(1-p_{\rm act})`$, exactly when clipping is always active | $`\theta`$ |
| **(e)** | clipped base optimizer | the statistic of the base optimizer, taken of $`\mathrm{clip}_c(G)`$ | as for the base optimizer |
| **(f)** | Lion / sign with momentum | flip probabilities $`\pi`$; none in $`\theta`$ when momentum dominates | $`\theta`$; $`m`$ |
| **(g)** | LoRA at $`B=0`$, step 1 | the rank-$`r`$ sketch $`\mathbb E[\Gamma]A_0^\top`$ (SGD), or its sign law (cold Adam); $`A`$ is fixed | $`B`$ only |

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.2](../experiments/05-01_optimizer_transformed_expansion.md#c5-2) | [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1)(b),(d); [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(d) | `v015:E10` | INFO (diagnostic); every smooth exact cell within bound |
| [Check 5.14](../experiments/05-09_partially_active_clipping.md#c5-14) | [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9), [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(e) | `v015:E11` (clipped mean) | PASS |
| [Check 5.15](../experiments/05-09_partially_active_clipping.md#c5-15) | [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9) at $`p_{\rm act}=1`$, [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(e) | `v015:E11` (always active) | PASS |
| [Check 5.16](../experiments/05-10_first_order_statistic_by_learner.md#c5-16) | [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(c), [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5)(a) | `v015:E11` (warm Adam) | PASS |
| [Check 5.17](../experiments/05-10_first_order_statistic_by_learner.md#c5-17) | [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(f) | `v015:E11` (Lion) | PASS |
| [Check 5.18](../experiments/05-10_first_order_statistic_by_learner.md#c5-18) | [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(g) | `v015:E11` (LoRA) | PASS |

<a id="c5-16"></a>

## Check 5.16 · Warm Adam: the second moment enters at first order through the optimizer state (Theorem 5.10(c), Proposition 5.5(a))

**Verdict.**

- With equal gradient means, the parameter block and the $`m`$ block contribute exactly 0. The $`v`$ block contributes $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle=2.504\times10^{-6}`$, as predicted to $`10^{-12}`$.
- So laws with equal means are not first-order equivalent for warm Adam.

**Why.**

- [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5)(a) says that the exact first-order statistic of Adam is the mean Adam increment over the complete state. It is a nonlinear functional of the gradient law. Through $`\Delta v=(1-\beta_2)(G\odot G-v)`$ it depends on $`\mathbb E\,G\odot G`$.
- Two laws with equal means and unequal second moments should therefore have a nonzero first-order term, carried by the $`v`$ block with weight $`(1-\beta_2)\nabla_vV`$.
- In this witness the parameter block vanishes exactly, and so does the frozen-denominator error. The check therefore isolates the $`v`$ channel of [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(c).

**Setup** (`v015:E11`, warm Adam).

- The network with $`k=1`$ context and $`p=13`$ parameters.
- A warm state with $`m=0`$, $`v=0.02`$ in every coordinate and step counter 20.
  - It is an admissible warm state, not a draw from an IID warm-up.
  - The symmetry of the label laws is what makes the parameter term vanish.
- Adam with $`\beta=(0.9,0.999)`$ and $`\eta=0.03`$.
- Labels: $`P`$ is uniform on $`\lbrace0,1\rbrace`$ and $`Q`$ is uniform on $`\lbrace\tfrac14,\tfrac34\rbrace`$. Both have mean $`\tfrac12`$, and their second moments differ.
- The continuation is two later updates with label $`\tfrac14`$, then the moment $`N=3`$ deployment.
- The full-state gradient of the continuation comes from central differences.

**Procedure.**

1. Compute $`\delta_1=\mathbb E_PG-\mathbb E_QG`$, $`\delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}`$, and the three block contributions $`\langle\nabla_\theta V,\mu^P_\theta-\mu^Q_\theta\rangle`$, $`\langle\nabla_mV,\cdot\rangle`$ and $`\langle\nabla_vV,\cdot\rangle`$.
2. Assert that the $`v`$ block equals $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle`$.
3. Compare with the measured gap, and with the $`L_v\kappa`$ term of [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6).

**Result** (`v015:E11`, `data.warm_Adam`).

| Quantity | Value |
|---|---:|
| $`\max_i\lvert\delta_{1,i}\rvert`$ (equal gradient means) | 0 |
| $`\max_i\delta_{2,i}`$ | 0.1875 |
| first-order blocks $`(\theta,\ m,\ v)`$ | $`(0,\ 0,\ 2.5043\times10^{-6})`$ |
| predicted $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle`$ | $`2.5043\times10^{-6}`$ (agreement $`<10^{-12}`$) |
| measured gap | $`8.753\times10^{-6}`$ |
| warm-Adam bound $`L_v\max\kappa`$ on the linear $`v`$ term | $`7.127\times10^{-6}`$ ($`\max\kappa=0.0125`$) |

**Why it supports Theorem 5.10(c) and Proposition 5.5(a).**

- At a warm Adam state, laws with equal mean gradients are separated at first order by the $`v`$ channel, with exactly the predicted coefficient.
- The momentum block vanishes because $`\delta_1=0`$.
- The parameter block also vanishes, for two reasons:
  - its mean term is proportional to $`\delta_1`$, which is 0;
  - its warmth error, the gap between the exact and the frozen-denominator parameter increment (Proposition 5.5(b)), is exactly zero here, because the label laws are symmetric and $`m=0`$.
- The rest of the measured gap is the remainder of [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1).
- The residual bound of Proposition 5.6 covers the linear $`v`$ term. [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(c) replaces that residual by the explicit statistic $`\mathbb E\,G^{\odot2}`$.
- Limit of the evidence: this witness exercises only the case where the frozen-denominator description is exact for the parameter block. In general, the exact first-order object is the mean Adam increment, and $`(\mathbb E\,G,\mathbb E\,G^{\odot2})`$ determines it only up to the frozen-denominator error of Proposition 5.5(b).
- How the $`v`$ channel acts over a long tail is the subject of [§7](../statements/07-01_state_channel_decomposition.md).

<a id="c5-17"></a>

## Check 5.17 · Lion: under momentum dominance the parameter block is blind to the message (Theorem 5.10(f))

**Verdict.** When momentum dominates, the parameter-block contrast between two very different gradient laws is exactly 0, with no sign flips. The $`m`$ block carries exactly $`0.01\,\delta_1`$.

**Why.** [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(f) says that a momentum-dominated sign update does not depend on the message, so even very different gradient laws give the same parameter increment. Only the $`m`$ block records the difference.

**Setup** (`v015:E11`, Lion).

- The network, with Lion ([Chen et al., 2023](#ref-chen2023symbolic); $`\beta_1=0.9`$, $`\beta_2=0.99`$, $`\eta=0.03`$).
- Two messages with labels 0 and 1, whose gradients differ by $`\delta_1`$, with $`\max\lvert\delta_1\rvert=1`$.
- Momentum-dominated state: $`m=0.2`$ in every coordinate, so that $`0.1\lvert G_i\rvert<0.9\cdot0.2`$ for every message.
- Control: $`m=0.005`$.

**Procedure.**

- Compute the per-coordinate flip indicator and the increment difference in each block.
- Assert that the parameter contrast is exactly zero in the dominated case and nonzero in the control.
- Assert that the $`m`$ block equals $`(1-\beta_2)\delta_1=0.01\,\delta_1`$.

**Result** (`v015:E11`, `data.Lion`).

| State | parameter-block contrast $`\max_i\lvert\cdot\rvert`$ | flip fraction | $`m`$-block contrast $`\max_i\lvert\cdot\rvert`$ |
|---|---:|---:|---:|
| momentum-dominated, $`m=0.2`$ | 0 (exactly) | 0 | 0.0100 $`=0.01\max\lvert\delta_1\rvert`$ |
| control, $`m=0.005`$ | 0.0600 $`=2\eta`$ | 0.5 | 0.0100 |

**Why it supports Theorem 5.10(f).**

- Under momentum dominance, the parameter increment is the same for both messages. Their first-order statistic in $`\theta`$ is the flip probability, here 0.
- Where a sign flips, the contrast is a full $`2\eta`$ step.
- The $`m`$ block always carries $`(1-\beta_2)\delta_1`$. When no later sign changes, the continuation is locally constant in $`m`$, so this channel also has no local first-order effect.

<a id="c5-18"></a>

## Check 5.18 · LoRA: the first update sees only a low-rank sketch (Theorem 5.10(g))

**Verdict.**

- At $`B=0`$, $`A`$ is bitwise unchanged after the first step under SGD, heavy-ball and cold Adam, for every law.
- Laws that differ only in the null space of $`A_0^\top`$ have a first-order gap below $`10^{-20}`$ and a second-order gap (slope 2.000).
- For laws that differ in the range, the first-order term matches the measured gap to within 0.3% at every $`\eta`$, and to within $`3\times10^{-5}`$ relative at $`\eta=10^{-3}`$.

**Why.** [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(g) makes two claims:

- At $`B=0`$, the adapter $`A`$ cannot move in step 1, so the feedback enters only through $`\Gamma A_0^\top`$.
- Consequently, a difference in the mean full-weight gradient that lies in the null space of $`A_0^\top`$ is invisible at first order, while a difference in the range is not.

**Setup** (`v015:E11`, LoRA).

- A bilinear sigmoid model $`W=W_0+\alpha BA`$ with $`d=k=4`$, $`W_0=0.2`$, $`\alpha=1`$, $`r\in\lbrace1,2\rbrace`$, $`B_0=0`$ and a random $`A_0`$. This is the LoRA parameterization and initialization ([Hu et al., 2022](#ref-hu2022lora)).
- Messages are soft-label mean cross-entropy targets that realize prescribed full-weight gradients $`\Gamma`$ exactly.
- Three law pairs:
  - *null:* the mean gradients differ by $`\Delta\Gamma`$ with $`\Delta\Gamma A_0^\top=0`$;
  - *range:* the difference has the form $`HA_0`$;
  - *sign:* the sketches $`\Gamma A_0^\top`$ have the same sign law but different magnitudes.
- Optimizers: SGD, heavy-ball and cold Adam. The bitwise audit uses $`\eta=0.03`$; the gaps use the full $`\eta`$ grid.

**Procedure.**

1. Assert that $`A`$ is bitwise unchanged after step 1, for every law, message and optimizer.
2. For the null and range pairs, compute the exact step-1 gap and the first-order term, and fit the slope of the null-pair gap.
3. For the sign pair under cold Adam, assert that the update equals $`-\eta h_\epsilon(\alpha\Gamma A_0^\top)`$, and record the finite-$`\epsilon`$ difference.

**Result** (`v015:E11`, `data.LoRA`).

| Quantity | rank 1 | rank 2 |
|---|---:|---:|
| bitwise checks that $`A`$ is unchanged (SGD, heavy-ball, cold Adam × all messages) | 21 / 21 | 21 / 21 |
| null-space residual $`\max\lvert\Delta\Gamma A_0^\top\rvert`$ | $`2.6\times10^{-18}`$ | $`4.8\times10^{-19}`$ |
| null pair: first-order term at $`\eta=0.1`$ | $`-1.1\times10^{-20}`$ | $`2.5\times10^{-21}`$ |
| null pair: gap at $`\eta=0.1`$; slope in $`\eta`$ | $`-3.66\times10^{-9}`$; 1.99994 | $`-4.83\times10^{-8}`$; 2.0000017 |
| range pair at $`\eta=10^{-3}`$: first-order term vs gap | $`6.06362\times10^{-7}`$ vs $`6.06362\times10^{-7}`$ | $`1.86514\times10^{-7}`$ vs $`1.86509\times10^{-7}`$ |
| sign pair, cold Adam: update difference at $`\epsilon=10^{-8}`$ | $`7.1\times10^{-8}`$ | $`7.8\times10^{-8}`$ |

**Why it supports Theorem 5.10(g).**

- At LoRA initialization, the first update cannot use any part of the feedback outside the rank-$`r`$ sketch $`\Gamma A_0^\top`$.
- Differences in the null space are second order. Differences in the range are first order, with the predicted coefficient.
- Under cold Adam only the sign law of the sketch matters as $`\epsilon\to0`$. At $`\epsilon=10^{-8}`$ the two updates differ by at most $`7.8\times10^{-8}`$, which is the finite-$`\epsilon`$ correction of $`h_\epsilon`$.

### [Check 5.2](../experiments/05-01_optimizer_transformed_expansion.md#c5-2) · The run-level representation in a controlled network, across optimizers (Proposition 5.1(b),(d))

**What it tests here.** This check is described in full on the page of Proposition 5.1. For Theorem 5.10 it tests: [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1)(b),(d); [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(d).

**Verdict.**

- Over a whole run, the gap between two feedback laws equals the first-order term of the *complete-state increment* plus a remainder within its curvature bound. This holds in every exact cell where the continuation is smooth (largest ratio 0.9986).
- In the cold-Adam cell of `v014:C23`, the increment's first-order term accounts for 98.4% of the gap. The raw mean-gradient term is $`-8.9\times10^{-11}`$.
- The remainder has the order the moment hierarchy of part (d) predicts. The exceptions are multi-round Adam, where the second-moment block makes it first order in $`\eta`$, and cells outside the smoothness hypothesis.

**Setup in brief.**

- *Laws.* The `v014:C23` label laws for moment $`N=2,3,4`$ and parity $`N=2,3`$, which match label moments below order $`N`$. Also the clipped-mean witness, realized on a linear-logit model.
- *Optimizers (11).*
  - SGD;
  - heavy-ball and Nesterov ($`\beta=0.9`$);
  - Adam cold ($`\beta=(0.9,0.999)`$, $`\epsilon=10^{-8}`$, bias correction);
  - Adam warm after $`W=5`$ and $`W=20`$ lr = 0 moment warm-up steps;
  - AdamW ($`\lambda=0.01`$);
  - clipped SGD and clipped Adam;
  - signSGD;
  - Lion ($`\beta_1=0.9`$, $`\beta_2=0.99`$).
- *Clip threshold.*
  - Chosen per task to give the clipped mass closest to 50%.
  - Where gradient norms tie, the 30–70% band is not attainable; for moment $`N=3`$ and parity $`N=3`$ the mass is 25%.

### [Check 5.14](../experiments/05-09_partially_active_clipping.md#c5-14) · Clipping: the statistic is the clipped mean (Proposition 5.9, Theorem 5.10(e))

**What it tests here.** This check is described in full on the page of Proposition 5.9. For Theorem 5.10 it tests: [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9), [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(e).

**Verdict.** Two gradient laws have equal raw means but clipped means that differ by $`-\tfrac12`$. They separate at first order, and the predicted gap $`\tfrac\eta2\sigma'(0.4)`$ matches the measured gap with a remainder of log–log slope 2.0015.

**Setup in brief.**

- A sigmoid with a linear logit at $`\theta=0.4`$.
- $`P(G=4)=\tfrac14`$ and $`P(G=-\tfrac43)=\tfrac34`$; $`Q=\delta_0`$.
- Clip at $`c=1`$; clipped SGD.

### [Check 5.15](../experiments/05-09_partially_active_clipping.md#c5-15) · Always-active clipping: only the direction matters (Proposition 5.9 with full activation, Theorem 5.10(e))

**What it tests here.** This check is described in full on the page of Proposition 5.9. For Theorem 5.10 it tests: [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9) at $`p_{\rm act}=1`$, [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(e).

**Verdict.** When every gradient is clipped and the two laws have the same mean unit direction, the first-order term is exactly 0 and the gap is second order (log–log slope 1.99997).

**Setup in brief.**

- Two-dimensional gradients with $`\lVert G\rVert\ge2>c=1`$.
- $`P`$ is uniform on $`\lbrace(\pm2,0)\rbrace`$ and $`Q`$ is uniform on $`\lbrace(0,\pm4)\rbrace`$. Both mean unit directions are 0, and their second direction moments differ.
- The same linear-logit model and $`\eta`$ grid as [Check 5.14](../experiments/05-09_partially_active_clipping.md#c5-14).

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
