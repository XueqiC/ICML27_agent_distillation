# §5 Optimizer state and the first-order representation — Statements

[← Framework](../README.md) · [Proofs](../proofs/05_optimizer_state.md) · [Experiments](../experiments/05_optimizer_state.md) · [中文](05_optimizer_state_ZH.md)

### §5 · Optimizer state and the first-order representation — exact statements

[Proofs](../proofs/05_optimizer_state.md) · [Experiments](../experiments/05_optimizer_state.md)

**What this section does.**

- Writes every stateful first-order optimizer as an increment map on the complete learner state and expands a feedback law's effect in the moments of that increment (Proposition 5.1, a tool).
- Computes, for Adam, how the first-order statistic depends on the optimizer state: the sign law at a cold state, the exact mean Adam increment at a warm state, and the frozen-denominator approximation with its error (Propositions 5.3–5.7).
- Gives the exact evolution of warmth under a fixed $`\beta_2`$ (Proposition 5.8) and the exact clipping statistic at any activation rate (Proposition 5.9).
- Lists the first-order statistic of each common learner (Theorem 5.10).

**Setting.** The complete learner state is $`z=(\theta,o)`$, with $`o`$ the optimizer state. An optimizer is an increment map $`z^+=z+\Delta_t(z,g)`$ of the consumed gradient $`g`$, clipping included. For a feedback law $`P`$, $`\mu^P_t(z)=\mathbb E_P\Delta_t(z,G)`$ is the mean increment and $`m^P_{k,t}`$ the $`k`$-th absolute moment of the increment. $`V^P_{t+1}`$ is the continuation value, $`U=\mathrm{span}\lbrace\nabla V(z):V\in\mathcal V\rbrace`$ for a declared family $`\mathcal V`$, and condition $`(\mathrm S_k)`$ asks for a $`C^{k+1}`$ continuation on the reachable segment with $`\lvert\nabla^{k+1}V[h,\dots,h]\rvert\le M_{k+1}\lVert h\rVert^{k+1}`$. For Adam with counter $`t`$, the warmth of coordinate $`i`$ is $`\kappa_i=(1-\beta_2)G_i^2/(\beta_2v_i)`$, the retained denominator is $`a_i=\sqrt{\beta_2v_i/(1-\beta_2^t)}`$, $`u_i=\lvert n_i\rvert/(a_i+\epsilon)`$ with $`n`$ the bias-corrected momentum, and the frozen-denominator surrogate replaces the denominator by $`a_i+\epsilon`$, i.e. one momentum-SGD step with the fixed preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$ (Definition 5.4).

#### [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1) · Optimizer-transformed expansion

- **(a)** Under $`(\mathrm S_k)`$, $`(K_t^P-K_t^Q)V(z)=\sum_{j=1}^{k}\tfrac1{j!}\langle\nabla^jV(z),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle+r_k`$ with $`\lvert r_k\rvert\le\tfrac{M_{k+1}}{(k+1)!}(m^P_{k+1}+m^Q_{k+1})`$.
- **(b)** Over a whole run, with $`d_t^Q`$ the state law under $`Q`$, the gap is the display below; a dual-norm certificate and a stopped version with an added exit probability $`\rho`$ follow.

```math
L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\big\langle\nabla V^P_{t+1}(Z_t),\mu_t^P(Z_t)-\mu_t^Q(Z_t)\big\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\sum_t\mathbb E_{d_t^Q}\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big).
```

- **(c)** Matching $`\Pi_U\mu`$ is sufficient to first order; a mismatch is detected by some $`V\in\mathcal V`$ whenever its first-order term exceeds $`\tfrac{M_2}2(m_2^P+m_2^Q)`$; the minimal number of first-order statistics is $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)})-1\le\min\lbrace\dim U,m-1\rbrace`$.
- **(d)** Agreement of $`\mathbb E\Delta^{\otimes j}`$ for $`j\le k`$ leaves a per-round gap at most $`\tfrac{M_{k+1}}{(k+1)!}(m^P_{k+1}+m^Q_{k+1})`$, which is $`O(\eta^{k+1})`$ for SGD.

*Derivation outline.*

- Taylor-expand $`V(z+\Delta)`$ along the segment from $`z`$ to $`z+\Delta`$ with a Lagrange remainder, and integrate under each law; constants cancel and expectations pass through the multilinear terms.
- For the run, insert this expansion into each term of the telescoping identity of Theorem 3.2.
- Sufficiency and necessity follow from $`\nabla V(z)\in U`$; minimality is the sufficiency duality of Theorem 4.2 applied to the linearized response class.

*Uses:* [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2), [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2). *Used by:* Theorem 5.10, [Definition 6.2](../proofs/06_learner_relative_information.md#s6-2), [Theorem 7.1](../proofs/07_second_moment_channel.md#s7-1), [Proposition 8.4](../proofs/08_certificate.md#s8-4), [Corollary 8.9](../proofs/08_certificate.md#s8-9).

[Proof](../proofs/05_optimizer_state.md#s5-1) · [Experiments](../experiments/05_optimizer_state.md) · proved

#### [Remark 5.2](../proofs/05_optimizer_state.md#s5-2) · Relation to stochastic modified equations, TracIn, influence functions and metagradients

- Proposition 5.1 is the same kind of calculation as stochastic modified equations, TracIn, influence functions, metagradients and In-Run Data Shapley; no novelty is claimed for it.
- What it makes explicit is the complete-state object, including $`m`$ and $`v`$, and the block structure of its remainder: the parameter and momentum blocks contribute at second order in $`\eta`$, while Adam's second-moment block contributes at first order in $`\eta`$, with relative size of the order of the warmth.

[Proof](../proofs/05_optimizer_state.md#s5-2) · remark

#### [Proposition 5.3](../proofs/05_optimizer_state.md#s5-3) · Cold-start Adam: sign quotient and optimizer memory

From a cold state:

- **(a)** $`\theta_1=\bar\theta-\eta\,h_\epsilon(G)`$, with $`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$ and $`h_0=\mathrm{sign}`$.
- **(b)** At $`\epsilon=0`$, equality of all bounded immediate-parameter utilities is equivalent to equality of the joint sign law; $`s-1`$ statistics are necessary and sufficient for $`s`$ attainable sign patterns; coordinate sign marginals do not suffice.
- **(c)** At $`\epsilon>0`$, $`h_\epsilon`$ is a bijection onto $`(-1,1)`$, so the full gradient law is required ($`m-1`$ statistics on an $`m`$-point alphabet).
- **(d)** For Lipschitz utilities, replacing $`h_\epsilon`$ by the sign map costs at most $`\eta\sum_iL_i[\beta_{P,i}(a)+\beta_{Q,i}(a)+2\epsilon/(a+\epsilon)]`$, where $`\beta_{P,i}(a)`$ is the gradient mass in $`0<\lvert G_i\rvert<a`$.
- **(e)** The first state retains $`m_1=(1-\beta_1)G`$; the histories $`(1,-\tfrac34)`$ and $`(2,-\tfrac34)`$ have identical first steps and signs, yet their second steps go in opposite directions. With $`\beta_1=0`$ no such pair exists.

*Derivation outline.*

- At a cold state the bias-corrected moments are $`G`$ and $`G^{\odot2}`$, so the step is $`G/(\lvert G\rvert+\epsilon)`$.
- The first step is an injective function of the sign vector ($`\epsilon=0`$) or of $`G`$ ($`\epsilon>0`$); the statistic count is the dimension of the functions constant on the fibres, by Theorem 4.2.
- The margin bound uses $`\lvert h_\epsilon(g_i)-\mathrm{sign}(g_i)\rvert=\epsilon/(\lvert g_i\rvert+\epsilon)`$.
- The memory witness is an exact two-step computation; the momentum term $`\beta_1(1-\beta_1)G_1`$ flips the sign of $`m_2`$.

*Illustration.* Two coordinates whose gradients are $`(1,1)`$ or $`(-1,-1)`$ with equal probability, versus $`(1,-1)`$ or $`(-1,1)`$: each coordinate has the same sign marginal and the same mean step $`0`$, but in the first case the two parameters always move together and in the second they never do.

*Uses:* [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2). *Used by:* Proposition 5.7, Theorem 5.10.

[Proof](../proofs/05_optimizer_state.md#s5-3) · [Experiments](../experiments/05_optimizer_state.md) · proved

#### [Proposition 5.5](../proofs/05_warm_adam.md#s5-5) · The exact first-order statistic of Adam and the frozen-denominator error

- **(a)** The first-order statistic of Adam is the exact mean increment $`\mu=\mathbb E\,\Delta_{\rm Adam}`$. Its parameter block is a nonlinear functional of the gradient law: two laws with equal $`\mathbb EG`$ and $`\mathbb E\,G^{\odot2}`$ can have different $`\mu_\theta`$.
- **(b)** $`\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\,u_i\min\lbrace\kappa_i/2,1\rbrace`$ for every coordinate.
- **(c)** The frozen-denominator mean increment is determined exactly by $`(\mathbb EG,\mathbb E\,G^{\odot2})`$. The claim that warm Adam needs only these two moments therefore rests on two separate approximations: the first-order expansion of Proposition 5.1, and the frozen denominator, whose error is (b).

*Derivation outline.*

- With $`v^+=\beta_2v(1+\kappa)`$ the true denominator is $`a\sqrt{1+\kappa}`$; comparing $`1/(a\sqrt{1+\kappa}+\epsilon)`$ with $`1/(a+\epsilon)`$ gives the factor $`1-(1+\kappa)^{-1/2}\le\min\lbrace\kappa/2,1\rbrace`$.
- The Adam step is a bounded, nonconstant function of each gradient coordinate, hence not a quadratic polynomial; a kernel vector of the moment map on four points gives two laws with equal first two moments and different mean step.

*Uses:* Proposition 5.1. *Used by:* Propositions 5.6, 5.8, Theorem 5.10, [Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11).

[Proof](../proofs/05_warm_adam.md#s5-5) · [Experiments](../experiments/05_optimizer_state.md) · proved

#### [Proposition 5.6](../proofs/05_warm_adam.md#s5-6) · Warm Adam is first-order affine in the consumed gradient

At a warm state, under a Lipschitz condition (L) in the parameters, a second-moment sensitivity condition $`(\mathrm S_v)`$ and a curvature bound (H) on $`F(g)=V(\tilde z^+(g))`$,

```math
\big\lvert\psi(y)-\psi_0-b^\top G(y)\big\rvert\le\eta\sum_iL_iu_i\min\lbrace\tfrac{\kappa_i}2,1\rbrace+L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace+\tfrac12H_G\lVert G\rVert_\infty^2=:r(y),
```

with $`b=(1-\beta_1)[-\eta\Phi\nabla_\theta V/(1-\beta_1^t)+\nabla_mV]`$. Laws with equal mean consumed gradient differ in response by at most $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$.

*Derivation outline.*

- Split the response into three differences: true versus frozen parameter step (bounded by Proposition 5.5(b)), true versus frozen second moment (bounded by $`(\mathrm S_v)`$), and the Taylor remainder of $`F`$ (bounded by (H)).

*Uses:* Proposition 5.5. *Used by:* Propositions 5.7, 5.8, Theorem 5.10.

[Proof](../proofs/05_warm_adam.md#s5-6) · [Experiments](../experiments/05_optimizer_state.md) · proved

#### [Proposition 5.7](../proofs/05_warm_adam.md#s5-7) · Cold steps separate equal-mean feedback at first order

At a cold state with the sign map and an affine utility $`V(\theta)=\tfrac12+w\theta`$:

- **(a)** $`P_0(G=1)=\tfrac14`$, $`P_0(G=-\tfrac13)=\tfrac34`$ and $`P_1=\delta_0`$ have equal means and responses that differ by $`\lvert w\rvert\eta/2`$.
- **(b)** For every $`N`$, two gradient laws that agree on all moments below order $`N`$ give responses that differ by $`\lvert w\rvert\eta\,2^{2-N}`$.

*Derivation outline.* The cold step is $`-\eta\,\mathrm{sign}(G)`$, so the response is affine in $`\mathbb E\,\mathrm{sign}(G)`$; the binomial even/odd laws of Theorem 4.4 put different mass on the single point where $`G>0`$.

*Uses:* Proposition 5.3, [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4). *Used by:* Proposition 5.8(c).

[Proof](../proofs/05_warm_adam.md#s5-7) · [Experiments](../experiments/05_optimizer_state.md) · proved

#### [Proposition 5.8](../proofs/05_warm_adam.md#s5-8) · Evolution of warmth

- **(a)** If the gradient second moment is stationary, $`\mathbb E v_s=\sigma^2(1-\beta_2^s)`$ and the mean-field warmth is given below. It decreases to the plateau $`(1-\beta_2)/\beta_2>0`$; the approximation $`1/(\beta_2s)`$ has relative error at most $`s(1-\beta_2)`$.

```math
\bar\kappa(s)=\frac{1-\beta_2}{\beta_2(1-\beta_2^{\,s})},\qquad\frac{\max\lbrace1/s,\;1-\beta_2\rbrace}{\beta_2}\le\bar\kappa(s)\le\frac{1/s+(1-\beta_2)}{\beta_2}.
```

- **(b)** Bias correction does not change warmth.
- **(c)** If $`\max_i\kappa_i\le c\big(1/(W+s)+(1-\beta_2)\big)`$ at the warm updates of a run with $`W\ge1`$ warm-up evaluations, the warmth-controlled part of the gap between equal-mean sources is at most $`\bar A\,[c\log(1+R/W)+c(1-\beta_2)R]`$, plus one unit per cold update and the curvature terms.
- **(d)** A learning-rate-zero warm-up advances $`m`$, $`v`$ and the counter exactly as real steps do; under the mean-field approximation with $`G_i^2\le\bar\gamma\sigma_i^2`$, the hypothesis of (c) holds with $`c=2\bar\gamma/\beta_2`$.

*Derivation outline.*

- Unroll the exponential moving average and take expectations; the bounds follow from Bernoulli's inequality and from $`e^{-y}(1+y)\le1`$.
- For (c), bound each warm term of the telescoping identity by Proposition 5.6 and use $`\min\lbrace1,a+b\rbrace\le\min\lbrace1,a\rbrace+b`$ and $`\sum_s1/(W+s)\le\log(1+R/W)`$.

*Illustration.* With $`\beta_2=0.999`$, the warmth of a typical gradient is about $`1/s`$ for the first hundred evaluations, but it never falls much below $`0.001`$: after a few thousand steps every new gradient still perturbs the second moment by about a tenth of a percent, and that floor adds a term linear in the number of updates.

*Uses:* Propositions 5.6, 5.7, [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2). *Used by:* Theorem 5.10, Remark 5.11.

[Proof](../proofs/05_warm_adam.md#s5-8) · [Experiments](../experiments/05_optimizer_state.md) · proved; the sufficient condition in (d) holds under the stated mean-field approximation

#### [Proposition 5.9](../proofs/05_learner_statistics.md#s5-9) · Partially active clipping

With $`p_{\rm act}=\Pr(\lVert G\rVert>c)`$,

```math
\mathbb E\,\mathrm{clip}_c(G)=\mathbb E\big[G\,1\lbrace\lVert G\rVert\le c\rbrace\big]+c\,\mathbb E\Big[\frac{G}{\lVert G\rVert}1\lbrace\lVert G\rVert>c\rbrace\Big],\qquad\Big\lVert\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E\frac{G}{\lVert G\rVert}\Big\rVert\le c(1-p_{\rm act}).
```

*Derivation outline.* Split on the event $`\lbrace\lVert G\rVert\le c\rbrace`$; on it, $`\lVert G-cG/\lVert G\rVert\rVert=c-\lVert G\rVert\le c`$.

*Used by:* Theorem 5.10(e).

[Proof](../proofs/05_learner_statistics.md#s5-9) · [Experiments](../experiments/05_optimizer_state.md) · proved

#### [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10) · The first-order statistic by learner

| Part | Learner | Statistic the feedback must match to first order |
|---|---|---|
| **(a)** | SGD | $`\mathbb EG`$ |
| **(b)** | heavy-ball, Nesterov | $`\mathbb EG`$, entering through $`\theta`$ and $`m`$ |
| **(c)** | Adam/AdamW, warm | exactly the mean Adam increment; $`(\mathbb EG,\ \mathbb E\,G^{\odot2})`$ under the frozen-denominator approximation, with error at most $`\eta(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace\kappa_i/2,1\rbrace]`$; the second moment enters through $`v`$ with weight $`(1-\beta_2)\nabla_vV`$ |
| **(d)** | Adam, cold step | the sign law $`\mathbb E\,h_\epsilon(G)`$ in $`\theta`$; $`\mathbb EG`$ and $`\mathbb E\,G^{\odot2}`$ retained in $`m,v`$ |
| **(e)** | clipping at norm $`c`$ | the clipped mean exactly; the mean unit direction up to $`c(1-p_{\rm act})`$ per law, exactly when clipping is always active |
| **(f)** | Lion / sign with momentum | flip probabilities; no dependence in $`\theta`$ when momentum dominates |
| **(g)** | LoRA at $`B=0`$ | step 1 leaves $`A`$ unchanged; the feedback enters only through the rank-$`r`$ sketch $`\Gamma A_0^\top`$, or its sign law under cold Adam |

*Derivation outline.* For each update rule, compute the increment, drop the terms that depend only on the state, and take the difference of expectations. Row (c) uses Proposition 5.5; row (e) uses Proposition 5.9; row (g) uses $`G_A=\alpha B^\top\Gamma=0`$ at $`B=0`$.

*Illustration.* Two label laws with the same mean gradient, one concentrated and one spread out, are interchangeable for SGD to first order. Under clipping the spread-out law loses its large gradients to the threshold; under warm Adam its larger second moment raises $`v`$ and shrinks every later step. Both effects are first order.

*Uses:* Propositions 5.1, 5.3, 5.5, 5.9. *Used by:* [Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11), [Corollary 6.12](../proofs/06_learner_relative_information.md#s6-12), [Theorem 7.1](../proofs/07_second_moment_channel.md#s7-1), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/05_learner_statistics.md#s5-10) · [Experiments](../experiments/05_optimizer_state.md) · proved

#### [Remark 5.11](../proofs/05_learner_statistics.md#s5-11) · Checking the warm-state conditions on a real model

- The error of Proposition 5.6 is a sensitivity-weighted sum plus a maximum, so quantiles of the warmth do not verify it. Report the weighted sum $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$, its largest terms and $`\max_i\kappa_i`$, and label any substitute $`\hat L_i`$ as a proxy.
- Non-additivity of the update map is a property of the optimizer, not of the task response; the response must be measured directly, for example by the paired-replacement scores of Definition 8.6.

[Proof](../proofs/05_learner_statistics.md#s5-11) · remark
