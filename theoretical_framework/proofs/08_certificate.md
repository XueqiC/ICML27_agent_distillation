# §8 Verification certificate (Theorem C) — Proofs

[← Framework](../README.md) · [Proofs](08_verification.md) · [Experiments](../experiments/08_verification.md) · [中文](08_certificate_ZH.md)

This page contains Proposition 8.4 through Remark 8.12, including Theorem C; the [sibling page](08_verification.md) supplies the preceding measurement, transport and adaptive-audit proofs and the shared [setting](08_verification.md#setting).

## Measuring the mean increment

Use the objects of [Proposition 5.1](05_optimizer_state.md#s5-1): consumed gradient $`G_t(z,Y,\xi)`$, increment $`\Delta_t(z,g)`$, mean increment $`\mu^P_t(z)`$, second increment moment $`m^P_{2,t}(z)=\mathbb E_P\lVert\Delta_t\rVert^2`$ and continuation $`V^P_{t+1}`$. The norm is the Hilbert norm of the state space (Euclidean in finite dimension).

- **Bounded increments (B).** There is $`b<\infty`$ with $`\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert\le b`$ almost surely, for the compared laws at the audited states. Examples: clipped SGD on the parameter block, $`b=\eta c`$; sign-type updates on $`p`$ parameters, $`b=\eta\sqrt p`$; SGD with gradients bounded by $`\bar G`$, $`b=\eta\bar G`$.
- **Curvature (S).** At the audited state, $`V:=V^P_{t+1}`$ is twice differentiable along the segments $`\lbrace z+s\Delta:s\in[0,1]\rbrace`$ reached by the increments, with $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$; write $`\Lambda=\lVert\nabla V(z)\rVert`$.
- **Audit samples.** At an audited state $`z`$, draw $`n_P`$ messages $`Y\sim P_t(\cdot\mid z)`$ and $`n_Q`$ messages $`Y\sim Q_t(\cdot\mid z)`$, independently and with independent update randomness, and compute the increments $`X=\Delta_t(z,G_t(z,Y,\xi))`$. Their empirical means are $`\bar X_P,\bar X_Q`$ and their empirical second moments $`\hat m^P_2,\hat m^Q_2`$. This takes $`n_P+n_Q`$ consumed-gradient evaluations at $`z`$ and no training tail.

<a id="s8-4"></a>

**Proposition 8.4 (Measuring the mean increment at the actual states).**

**(a) Hilbert-space concentration.** Let $`X_1,\dots,X_n`$ be i.i.d. in a real Hilbert space with $`\lVert X\rVert\le b`$ almost surely and mean $`\mu`$. With probability at least $`1-\delta`$,

```math
\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big).
```

For two independent samples of sizes $`n_P,n_Q`$, with probability at least $`1-\delta`$,

```math
\big\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\big\rVert\le b\Big(1+\sqrt{2\log(2/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big).
```

**(b) First-order audit at an actual state.** Fix an audited state $`z`$, conditionally on the history that produced it, and assume (B) and (S) there. With probability at least $`1-\delta`$ over the audit samples,

```math
\big\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\big\rvert\le\Lambda\big(\lVert\bar X_P-\bar X_Q\rVert+e_1\big)+\tfrac{M_2}{2}\big(\hat m^P_2+\hat m^Q_2+e_2\big),
```

where

```math
e_1=b\Big(1+\sqrt{2\log(4/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big),\qquad e_2=b^2\Big(\sqrt{\tfrac{\log(4/\delta)}{2n_P}}+\sqrt{\tfrac{\log(4/\delta)}{2n_Q}}\Big).
```

**(c) Run-level audit along actual runs.** Draw $`J`$ independent runs of protocol $`Q`$. At every round $`t`$ of every run, apply (b) at the visited state with confidence $`\delta_{\rm a}/(JR)`$, and let $`\hat\varepsilon_{j,t}`$ be the right side of (b), capped at $`1`$. Suppose (B) and (S) hold at every visited state with the stated constants, and audit samples are drawn without changing the run. Then with probability at least $`1-\delta_{\rm a}-\delta_{\rm mc}`$,

```math
\lvert L(P)-L(Q)\rvert\le\frac1J\sum_{j=1}^J\sum_{t=0}^{R-1}\hat\varepsilon_{j,t}+R\sqrt{\frac{\log(1/\delta_{\rm mc})}{2J}}.
```

**What is measured and what is not.**

- The data-dependent parts are $`\bar X_P-\bar X_Q`$ and $`\hat m_2`$. Each needs only consumed-gradient evaluations at the visited state.
- $`\Lambda`$ and $`M_2`$ are properties of the continuation value, not of the messages, and the samples do not estimate them. They must be bounded analytically, or calibrated against full-tail audits ([Theorem 8.3](08_verification.md#s8-3)) with the calibration error added to the bound.

*Proof.*

**(a)** Let $`f(X_1,\dots,X_n)=\lVert\bar X-\mu\rVert`$.

- *Mean.* By Jensen's inequality and independence, $`\mathbb Ef\le(\mathbb Ef^2)^{1/2}=(\mathbb E\lVert X-\mu\rVert^2/n)^{1/2}`$. The cross terms $`\mathbb E\langle X_i-\mu,X_l-\mu\rangle`$, $`i\ne l`$, vanish by independence; this uses the inner product. In a Hilbert space $`\mathbb E\lVert X-\mu\rVert^2=\mathbb E\lVert X\rVert^2-\lVert\mu\rVert^2\le b^2`$, so $`\mathbb Ef\le b/\sqrt n`$.
- *Deviation.* Replacing $`X_i`$ by $`X_i'`$ changes $`f`$ by at most $`\lVert X_i-X_i'\rVert/n\le2b/n`$. McDiarmid's inequality with $`c_i=2b/n`$ gives $`\Pr\lbrace f-\mathbb Ef\ge s\rbrace\le\exp(-2s^2/\sum_ic_i^2)=\exp(-ns^2/(2b^2))`$. Setting this to $`\delta`$ gives $`s=b\sqrt{2\log(1/\delta)/n}`$.
- *Two samples.* Apply the one-sample bound to each sample with $`\delta/2`$, use $`\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\rVert\le\lVert\bar X_P-\mu^P\rVert+\lVert\bar X_Q-\mu^Q\rVert`$, and take a union bound.

**(b)** Allocate $`\delta/2`$ to the two-sample bound of (a), which gives $`e_1`$ since $`\log(2/(\delta/2))=\log(4/\delta)`$. Allocate $`\delta/4`$ to each second moment: $`\lVert X\rVert^2\in[0,b^2]`$, so one-sided Hoeffding gives $`m^P_2\le\hat m^P_2+b^2\sqrt{\log(4/\delta)/(2n_P)}`$ with probability at least $`1-\delta/4`$, and likewise for $`Q`$. On the intersection of the three events, $`\lVert\mu^P-\mu^Q\rVert\le\lVert\bar X_P-\bar X_Q\rVert+e_1`$ and $`m^P_2+m^Q_2\le\hat m^P_2+\hat m^Q_2+e_2`$.

- By (S) and Taylor's theorem with integral remainder, $`V(z+\Delta)=V(z)+\langle\nabla V(z),\Delta\rangle+\rho(\Delta)`$ with $`\lvert\rho(\Delta)\rvert\le\tfrac{M_2}2\lVert\Delta\rVert^2`$. Since $`(K^P_tV)(z)=\mathbb E_PV(z+\Delta)`$, this is the one-boundary case of [Proposition 5.1](05_optimizer_state.md#s5-1):

```math
\big\lvert(K^P_t-K^Q_t)V(z)\big\rvert\le\big\lvert\langle\nabla V(z),\mu^P-\mu^Q\rangle\big\rvert+\tfrac{M_2}{2}\big(m^P_2+m^Q_2\big).
```

- Cauchy–Schwarz gives $`\lvert\langle\nabla V(z),\mu^P-\mu^Q\rangle\rvert\le\Lambda\lVert\mu^P-\mu^Q\rVert`$; substitute the two inequalities. The samples are drawn after the state is fixed, so the argument holds conditionally on the history.

**(c)** Let $`\varepsilon_t(z)=\min\lbrace1,\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\rvert\rbrace`$.

- *Audit event.* Each audit $`(j,t)`$ succeeds with conditional probability at least $`1-\delta_{\rm a}/(JR)`$; the conditional failure probabilities sum to at most $`\delta_{\rm a}`$. On the event that all succeed, $`\varepsilon_t(Z_{j,t})\le\hat\varepsilon_{j,t}`$ for all $`j,t`$, using $`\varepsilon_t\le1`$.
- *Telescoping.* [Theorem 3.2(a)](03_learning_consequence.md#s3-2) with $`P`$ as the reference and the triangle inequality inside the expectation give $`\lvert L(P)-L(Q)\rvert\le\sum_t\mathbb E_{d^Q_t}\varepsilon_t(Z_t)`$. The cap at $`1`$ is valid because each term is a difference of two $`[0,1]`$-valued expectations.
- *Monte Carlo.* The run totals $`S_j=\sum_t\varepsilon_t(Z_{j,t})`$ are i.i.d. in $`[0,R]`$ with mean $`\sum_t\mathbb E_{d^Q_t}\varepsilon_t`$. One-sided Hoeffding gives $`\sum_t\mathbb E_{d^Q_t}\varepsilon_t\le\tfrac1J\sum_jS_j+R\sqrt{\log(1/\delta_{\rm mc})/(2J)}`$ with probability at least $`1-\delta_{\rm mc}`$. On both events $`\tfrac1J\sum_jS_j\le\tfrac1J\sum_{j,t}\hat\varepsilon_{j,t}`$; a union bound completes the proof. $`\square`$

<a id="s8-5"></a>

**Remark 8.5 (Norms and constants).**

- The concentration bound of [Proposition 8.4(a)](#s8-4) holds in the Hilbert norm of the state space; its proof uses the inner product (vanishing cross terms). It does not extend to arbitrary norms with the same constant; for example, in the sup norm on $`\mathbb R^p`$ the analogous bound carries a $`\sqrt{\log p}`$ factor.
- The bound contains no explicit dimension, but $`b`$ itself can grow with the dimension. Under cold-start Adam every coordinate moves by about $`\eta`$ ([Proposition 5.3](05_optimizer_state.md#s5-3)), so $`\lVert\Delta\theta\rVert`$ can reach $`\eta\sqrt p`$.
- A ranking decision needs only the scalar projection $`\langle\nabla V(z),\Delta\rangle`$, whose fluctuation can be far smaller than $`\lVert\nabla V(z)\rVert\,b`$. [Theorem 8.8](#s8-8) goes further: it measures the finite replacement effect directly and needs no bound $`b`$.
- **Cost comparison.** A full-tail audit at one state needs the replica count of [Theorem 8.3(d)](08_verification.md#s8-3) per candidate; (b) needs $`n_P+n_Q`$ gradient evaluations per candidate. What (b) cannot do alone is supply $`\Lambda`$ and $`M_2`$.

## The CRN paired-replacement certificate (Theorem C)

<a id="s8-6"></a>

**Definition 8.6 (CRN paired-replacement score).**

- Fix an audited state $`z`$ at round $`t`$ (a complete learner state, including the optimizer state), conditionally on the history that produced it. The candidates are $`j=0,1,\dots,M`$ with $`M\ge1`$; $`j=0`$ is the reference. Candidate $`j`$ has message law $`P^j=P^j_t(\cdot\mid z)`$, and a message $`Y`$ produces the increment $`\Delta^j=\Delta_t(z,G_t(z,Y,\xi))`$, update randomness included.
- A *surrogate tail* $`T_\omega`$ maps a state to a final state by running a declared (possibly truncated) reference tail with randomness $`\omega`$. A *surrogate objective* $`\tilde F`$ scores the final state (for example, the teacher-forced log-likelihood of held-out successful trajectories, mapped to $`[0,1]`$). Write

```math
\varphi_\omega(u)=\tilde F\big(T_\omega(z+u)\big),\qquad\tilde L(P^j)=\mathbb E\,\varphi_\omega(\Delta^j),
```

where $`\omega`$ is independent of $`\Delta^j`$. $`\tilde L(P^j)`$ is the surrogate outcome of one round with candidate $`j`$ followed by the surrogate tail.

- *Replicas* $`\ell=1,\dots,r`$, $`r\ge2`$: draw a fresh $`\omega_\ell`$ and, for every candidate, one message and its increment $`\Delta^j_\ell`$. All candidates in replica $`\ell`$ share $`\omega_\ell`$ (*common random numbers*); their messages may also be coupled through common random numbers, as long as each $`\Delta^j_\ell`$ has its correct marginal law. Replicas are independent of each other.
- If a candidate's message law has finite support, the draw may be replaced by the exact weighted average over the support; the randomness then comes only from $`\omega`$.
- The score of the ordered pair $`(j,k)`$ is

```math
Z^{jk}_\ell=\varphi_{\omega_\ell}(\Delta^j_\ell)-\varphi_{\omega_\ell}(\Delta^k_\ell),\qquad\hat S^{jk}=\frac1r\sum_{\ell=1}^rZ^{jk}_\ell .
```

- On the task side, $`L_F(P^j)=\mathbb E\,F_E\big(T^{\rm full}(z+\Delta^j)\big)`$ is the outcome of one round with candidate $`j`$ followed by the full reference tail and the task objective. In the notation of the setting, $`L_F(P^j)=(K^j_tV^0_{t+1})(z)`$, so $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$.

<a id="s8-7"></a>

**Assumption 8.7 (Independent replicas, bounded scores, surrogate alignment).**

- **(I) Independent replicas.** Conditionally on the history that produced $`z`$, the vectors $`(Z^{jk}_\ell)_{j,k}`$, $`\ell=1,\dots,r`$, are i.i.d.; $`r`$ is fixed before the scores are observed.
- **(B) Bounded scores.** For every pair, $`Z^{jk}_\ell`$ lies almost surely in a known interval of length $`B`$. For example $`\tilde F\in[0,1]`$ gives $`B\le2`$.
- **(A) Surrogate alignment.** There are $`\alpha>0`$ and $`\varepsilon_{\rm sur}\ge0`$ such that, for all pairs,

```math
\Big\lvert\big(L_F(P^j)-L_F(P^k)\big)-\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)\Big\rvert\le\alpha\,\varepsilon_{\rm sur}.
```

$`\varepsilon_{\rm sur}`$ absorbs both the difference between the surrogate and the task objective and the truncation of the tail.

<a id="s8-8"></a>

**Theorem 8.8 (CRN paired-replacement certificate; Theorem C).** Under [Assumption 8.7](#s8-7), let $`\hat V^{jk}=\frac1{r-1}\sum_\ell(Z^{jk}_\ell-\hat S^{jk})^2`$ be the unbiased sample variance and

```math
e^{jk}_{\rm stat}=\sqrt{\frac{2\hat V^{jk}\log(4M^2/\delta)}{r}}+\frac{7B\log(4M^2/\delta)}{3(r-1)} .
```

- **(a) Exactness.** $`\mathbb E\,Z^{jk}_\ell=\tilde L(P^j)-\tilde L(P^k)`$. No first-order approximation is involved.
- **(b) Certificate.** With probability at least $`1-\delta`$, simultaneously for all ordered pairs $`(j,k)`$, $`j\ne k`$,

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big).
```

- **(c) Decision rule.** Declaring "$`j`$ is better than $`k`$" whenever $`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ makes no wrong declaration, over all pairs simultaneously, with probability at least $`1-\delta`$. With $`k=0`$ this certifies $`A_t(z,j)>0`$.
- The certificate needs no continuation gradient $`\nabla V`$, no curvature constant and no second-order remainder. The only constants to calibrate are $`(\alpha,\varepsilon_{\rm sur})`$ ([Remark 8.11](#s8-11)).

*Proof.*

**(a)** By linearity of expectation, $`\mathbb EZ^{jk}_\ell=\mathbb E\varphi_{\omega_\ell}(\Delta^j_\ell)-\mathbb E\varphi_{\omega_\ell}(\Delta^k_\ell)`$. Each term depends only on the joint law of $`(\omega_\ell,\Delta^j_\ell)`$, which is the product of the law of $`\omega`$ and the marginal law of $`\Delta^j`$, whatever coupling is used across candidates. Hence each term equals $`\tilde L(P^j)`$, respectively $`\tilde L(P^k)`$. With exact enumeration over a finite support the same holds with the inner average in place of the draw.

**(b)** *Empirical Bernstein bound.* Maurer and Pontil (2009) prove the following empirical Bernstein bound. For $`n\ge2`$ i.i.d. variables $`X_i\in[0,1]`$ with unbiased sample variance $`V_n`$, with probability at least $`1-\delta'`$,

```math
\mathbb EX-\frac1n\sum_iX_i\le\sqrt{\frac{2V_n\log(2/\delta')}{n}}+\frac{7\log(2/\delta')}{3(n-1)} .
```

- If $`Z`$ takes values in an interval $`[a,a+B]`$, apply this to $`X=(Z-a)/B\in[0,1]`$, whose sample variance is $`V_n(Z)/B^2`$, and multiply by $`B`$. This gives the same bound for $`Z`$ with $`\sqrt{2V_n(Z)\log(2/\delta')/n}+7B\log(2/\delta')/(3(n-1))`$.
- *Events.* For each ordered pair $`(j,k)`$, apply the bound to $`X=-Z^{jk}`$ (values in an interval of length $`B`$, same sample variance) to obtain the lower tail $`\tilde L(P^j)-\tilde L(P^k)\ge\hat S^{jk}-e^{jk}_{\rm stat}`$. There are $`(M+1)M`$ ordered pairs, and $`(M+1)M\le2M^2`$ for $`M\ge1`$. Use $`\delta'=\delta/(2M^2)`$ for each, so $`\log(2/\delta')=\log(4M^2/\delta)`$. A union bound gives probability at least $`1-\delta`$ that all hold. The event for $`(k,j)`$ is the upper tail for $`(j,k)`$, so the bound is two-sided for every unordered pair.
- *Transfer to the task.* On this event, (A) gives $`L_F(P^j)-L_F(P^k)\ge\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)-\alpha\varepsilon_{\rm sur}\ge\alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big)`$.

**(c)** On the event of (b), $`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ implies $`L_F(P^j)-L_F(P^k)>0`$, because $`\alpha>0`$. With $`k=0`$, $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$. $`\square`$

*Comments on Theorem 8.8.*

- **What is estimated.** The estimand is the finite one-round replacement contrast of [Theorem 3.2](03_learning_consequence.md#s3-2), measured on a surrogate tail. Nothing is linearized, so the second-moment channel of §7, which is nonlinear and saturating in the message magnitude ([Corollary 7.2](07_second_moment_channel.md#s7-2)), is measured as it is.
- **Where the assumptions bite.** (I) requires fresh $`\omega_\ell`$ and fresh messages per replica at a fixed state. (B) needs a range that is known in advance: the observed range of the scores is not a valid $`B`$. (A) is a statement about population differences and has to be calibrated ([Remark 8.11](#s8-11)).

<a id="s8-9"></a>

**Corollary 8.9 (The first-order version, its margin criterion, and its cap).** Use the setting of [Definition 8.6](#s8-6) and let $`\mu^j=\mathbb E\Delta^j`$ and $`m^j_2=\mathbb E\lVert\Delta^j\rVert^2`$.

- **(a) Margin criterion.** Let $`V`$ be a continuation that is twice differentiable along the reached segments with $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$. If $`\lvert\langle\nabla V(z),\mu^j-\mu^k\rangle\rvert>\tfrac{M_2}{2}(m^j_2+m^k_2)`$, then $`(K^j_t-K^k_t)V(z)`$ has the sign of $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$. At run level, let $`T_{jk}=\sum_t\mathbb E_{d^k_t}\langle\nabla V^j_{t+1}(Z_t),\mu^j_t-\mu^k_t\rangle`$ and $`B_{jk}=\sum_t\mathbb E_{d^k_t}\tfrac{M_{2,t}}2(m^j_{2,t}+m^k_{2,t})`$ be the first-order term and the remainder bound of [Proposition 5.1](05_optimizer_state.md#s5-1) with $`P_j`$ in the role of $`P`$; here $`d^k_t`$ is the state law at round $`t`$ of the process driven by $`P_k`$. If $`\lvert T_{jk}\rvert>B_{jk}`$, then $`L(P_j)-L(P_k)`$ has the sign of $`T_{jk}`$.
- **(b) Measured first-order certificate.** Let $`\tilde V(u)=\mathbb E_\omega\varphi_\omega(u)`$ be twice differentiable along $`\lbrace su:s\in[0,1]\rbrace`$ for the reached increments $`u`$, with $`\lvert D^2\tilde V(su)[u,u]\rvert\le\tilde M\lVert u\rVert^2`$, and let $`\tilde s^{jk}=\langle\nabla\tilde V(0),\mu^j-\mu^k\rangle`$. For a step $`h\in(0,1]`$ define $`Z^{jk}_\ell(h)=\big(\varphi_{\omega_\ell}(h\Delta^j_\ell)-\varphi_{\omega_\ell}(h\Delta^k_\ell)\big)/h`$ with mean $`\hat S^{jk}(h)`$, and assume (I), (A), and (B) with range length $`B_h`$. Let $`e^{jk}_{\rm stat}(h)`$ be the radius of [Theorem 8.8](#s8-8) with $`B_h`$ in place of $`B`$, and

```math
e^{jk}_{\rm fd}(h)=\tfrac12h\tilde M\big(m^j_2+m^k_2\big),\qquad\tilde e^{jk}_2=\tfrac12\tilde M\big(m^j_2+m^k_2\big).
```

Then with probability at least $`1-\delta`$, simultaneously for all ordered pairs,

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)-\tilde e^{jk}_2-\varepsilon_{\rm sur}\big).
```

- **(c) Cap.** On the event of (b), a pair is certified by (b) only if $`\tilde s^{jk}>\tilde e^{jk}_2+\varepsilon_{\rm sur}`$. A pair whose first-order term does not exceed the second-order remainder bound is never certified at first order, whatever its true gap. In particular this holds for pairs whose mean increments agree along $`\nabla\tilde V(0)`$, which differ only through second and higher moments of the increment; [Theorem 8.8](#s8-8) can certify such pairs. In population terms, (b) certifies the lower bound $`\tilde s^{jk}-\tilde e^{jk}_2\le\tilde L(P^j)-\tilde L(P^k)`$, the quantity that Theorem 8.8 estimates directly.
- **(d) Oracle scores are not certificates.** A score $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$ computed with the exact continuation gradient of the task, as in controlled experiments that differentiate through the whole reference tail, satisfies (a) wherever the margin holds. It is not a measurable certificate: $`\nabla V(z)`$ and $`M_2`$ are not available from feedback samples, and checking the margin needs both. A cheap score ranks correctly by (a) only if (i) its difference equals the first-order term with the optimizer-transformed mean increment of the actual learner ([Theorem 5.10](05_learner_statistics.md#s5-10)) at the actual state, and (ii) the margin holds. Scores built from the raw mean gradient, or with coefficients shared across states and tasks, violate (i); any score used where the margin fails violates (ii).

*Proof.*

**(a)** *One boundary.* By the Taylor bound used in the proof of [Proposition 8.4(b)](#s8-4), $`(K^j_t-K^k_t)V(z)=s+\rho`$ with $`s=\langle\nabla V(z),\mu^j-\mu^k\rangle`$ and $`\lvert\rho\rvert\le\tfrac{M_2}2(m^j_2+m^k_2)<\lvert s\rvert`$. If $`s>0`$ then $`s+\rho>s-\lvert s\rvert=0`$; if $`s<0`$ then $`s+\rho<0`$. *Run level.* [Proposition 5.1(b)](05_optimizer_state.md#s5-1) with $`P_j`$ as $`P`$ and $`P_k`$ as $`Q`$ gives $`L(P_j)-L(P_k)=T_{jk}+\mathcal E`$ with $`\lvert\mathcal E\rvert\le B_{jk}<\lvert T_{jk}\rvert`$; the same sign argument applies.

**(b)** Taylor's theorem with integral remainder along $`s\mapsto\tilde V(shu)`$ gives, for every reached increment $`u`$,

```math
\Big\lvert\tilde V(hu)-\tilde V(0)-h\langle\nabla\tilde V(0),u\rangle\Big\rvert\le\tfrac12h^2\tilde M\lVert u\rVert^2 .
```

- *Finite-difference bias.* As in [Theorem 8.8(a)](#s8-8), $`\mathbb EZ^{jk}_\ell(h)=\big(\mathbb E\tilde V(h\Delta^j)-\mathbb E\tilde V(h\Delta^k)\big)/h`$. Taking expectations of the display for $`u=\Delta^j`$ and $`u=\Delta^k`$ and dividing by $`h`$ gives $`\lvert\mathbb EZ^{jk}_\ell(h)-\tilde s^{jk}\rvert\le e^{jk}_{\rm fd}(h)`$.
- *Remainder.* The display with $`h=1`$ gives $`\tilde L(P^j)-\tilde L(P^k)=\tilde s^{jk}+\tilde\rho`$ with $`\lvert\tilde\rho\rvert\le\tilde e^{jk}_2`$.
- *Concentration.* The argument of [Theorem 8.8(b)](#s8-8), applied to $`Z^{jk}_\ell(h)`$ with range $`B_h`$, gives an event of probability at least $`1-\delta`$ on which $`\mathbb EZ^{jk}_\ell(h)\ge\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)`$ for all ordered pairs.
- On that event, $`\tilde L(P^j)-\tilde L(P^k)\ge\tilde s^{jk}-\tilde e^{jk}_2\ge\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)-\tilde e^{jk}_2`$, and (A) transfers this to the task as in Theorem 8.8.

**(c)** Certification by (b) means $`\hat S^{jk}(h)>e^{jk}_{\rm stat}(h)+e^{jk}_{\rm fd}(h)+\tilde e^{jk}_2+\varepsilon_{\rm sur}`$. On the event of (b), $`\tilde s^{jk}\ge\mathbb EZ^{jk}_\ell(h)-e^{jk}_{\rm fd}(h)\ge\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)>\tilde e^{jk}_2+\varepsilon_{\rm sur}`$. If the mean increments agree along $`\nabla\tilde V(0)`$, then $`\tilde s^{jk}=0`$ and no certification is possible. The last sentence of (c) is the remainder step of (b).

**(d)** The first claim is (a). For (i): if the score difference equals $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$ and the margin holds, (a) gives the sign. If the score differs from this term, (a) does not control the difference; if the margin fails, the remainder can reverse the sign. $`\square`$

*Comments on Corollary 8.9.*

- **Why the first-order version is weak.** Two effects add. The a-priori range of $`Z(h)`$ grows like $`1/h`$ (for $`\tilde F\in[0,1]`$, $`B_h\le2/h`$), so the range term of $`e_{\rm stat}(h)`$ grows as $`h`$ shrinks, while $`e_{\rm fd}(h)`$ grows with $`h`$. And the remainder $`\tilde e_2`$ does not shrink with $`r`$ at all.
- **Where the margin holds.** When every coordinate on which the continuation is curved moves by $`O(\eta)`$ (SGD, momentum, sign updates on the parameter block), the first-order term is $`O(\eta)`$ and the remainder $`O(\eta^2)`$ per round, so the margin holds at small step sizes. Under Adam the moment increments $`\Delta m=(1-\beta_1)(G-m)`$ and $`\Delta v=(1-\beta_2)(G\odot G-v)`$ are $`O(1)`$, while the continuation depends on $`(m,v)`$ only through steps of size $`\eta`$. The first-order term and the remainder are then both $`O(\eta)`$, and a small step size does not imply the margin. This is the regime of the second-moment channel in §7 ([Theorem 7.1](07_second_moment_channel.md#s7-1)).

<a id="s8-10"></a>

**Proposition 8.10 (Cost, and what common random numbers buy).**

- **(a) Cost.** One audited state needs $`(M+1)r`$ surrogate tails, plus the task-evaluated tails used for calibration ([Remark 8.11](#s8-11)). A full-tail paired audit with the radius of [Theorem 8.3(d)](08_verification.md#s8-3) needs $`2n_t`$ full tails per candidate.
- **(b) Replicas for the range term.** The range term of $`e_{\rm stat}`$ is at most $`\zeta B`$ if and only if

```math
r\ \ge\ 1+\frac{7\log(4M^2/\delta)}{3\zeta}.
```

For example, $`M=8`$, $`\delta=0.05`$ and $`\zeta=0.2`$ give $`\log(4M^2/\delta)=\log5120\approx8.54`$ and $`r\ge101`$. With fewer replicas the range term dominates the radius.

- **(c) Variance under common random numbers.** Suppose $`\varphi_\omega`$ is pathwise Lipschitz on the reached increments, with constant $`\lambda_\omega`$: $`\lvert\varphi_\omega(u)-\varphi_\omega(u')\rvert\le\lambda_\omega\lVert u-u'\rVert`$. Then

```math
\lvert Z^{jk}_\ell\rvert\le\lambda_{\omega_\ell}\,\lVert\Delta^j_\ell-\Delta^k_\ell\rVert,\qquad\mathrm{Var}\,Z^{jk}_\ell\le\mathbb E\big[\lambda_\omega^2\,\lVert\Delta^j-\Delta^k\rVert^2\big].
```

With independent tails $`\omega,\omega'`$ for the two arms (and independent messages), the variance of $`\varphi_\omega(\Delta^j)-\varphi_{\omega'}(\Delta^k)`$ is $`\mathrm{Var}\,\varphi_\omega(\Delta^j)+\mathrm{Var}\,\varphi_{\omega'}(\Delta^k)\ge\mathbb E\,\mathrm{Var}(\varphi_\omega(\Delta^j)\mid\Delta^j)+\mathbb E\,\mathrm{Var}(\varphi_{\omega'}(\Delta^k)\mid\Delta^k)`$, which contains the tail noise and does not vanish as the candidates approach each other. If moreover $`\sup_\omega\lambda_\omega\le\bar\lambda`$ and $`\lVert\Delta^j-\Delta^k\rVert\le b_{jk}`$ almost surely under the coupling, then $`B=2\bar\lambda b_{jk}`$ is a valid range in [Assumption 8.7(B)](#s8-7).

*Proof.* (a) is counting: each replica evaluates each of the $`M+1`$ candidates once. (b) The range term $`7B\log(4M^2/\delta)/(3(r-1))\le\zeta B`$ rearranges to the display; for the example, $`7\cdot8.541/(3\cdot0.2)\approx99.6`$, so $`r\ge100.6`$. (c) The first inequality is the Lipschitz condition at $`u=\Delta^j_\ell`$, $`u'=\Delta^k_\ell`$. Then $`\mathrm{Var}\,Z\le\mathbb EZ^2`$. With independent arms the variance of a difference is the sum of variances, and the law of total variance gives $`\mathrm{Var}\,\varphi_\omega(\Delta^j)\ge\mathbb E\,\mathrm{Var}(\varphi_\omega(\Delta^j)\mid\Delta^j)`$. Under the two uniform bounds, $`Z^{jk}_\ell\in[-\bar\lambda b_{jk},\bar\lambda b_{jk}]`$. $`\square`$

<a id="s8-11"></a>

**Remark 8.11 (Calibrating the surrogate; transfer along training).**

- **Calibration.** On a set of calibration pairs, estimate $`L_F(P^j)-L_F(P^k)`$ by paired full-tail task evaluations ([Theorem 8.3](08_verification.md#s8-3)), regress them on $`\hat S^{jk}`$, and read off $`\hat\alpha`$ and a residual bound $`\hat\varepsilon_{\rm sur}`$. Applying $`(\hat\alpha,\hat\varepsilon_{\rm sur})`$ to other pairs requires an exchangeability assumption: the calibration pairs and the new pairs come from the same family. This is an assumption, not a theorem, and the calibration's own statistical error must be added.
- **Use inside the adaptive audit.** At an audited round, replace the full-tail lower confidence bound of [Theorem 8.3](08_verification.md#s8-3) by $`\Gamma_t=\min\lbrace1,\alpha(\hat S^{J_t0}-e^{J_t0}_{\rm stat}-\varepsilon_{\rm sur})\rbrace`$ for the selected candidate, with $`\Gamma_t=0`$ when no candidate is certified against the reference. If (A) holds at every audited state with the declared constants and the confidence allocations sum to $`\delta`$, Theorem 8.3(c) holds unchanged. Its proof uses only $`A_t\ge\Gamma_t`$ on the audit event and $`0\le\Gamma_t\le1`$, and [Theorem 8.8(b)](#s8-8) supplies the first, since $`A_t(z,j)=L_F(P^j)-L_F(P^0)`$.
- **Transfer along training.** The certificate holds at the audited state. Using the selected candidate at later states needs the transport term of [Theorem 8.2](08_verification.md#s8-2), or a re-audit at the next checkpoint that checks whether the ranking persists (an a-posteriori check).

<a id="s8-12"></a>

**Remark 8.12 (Open problems).**

- **A provable range.** [Theorem 8.8](#s8-8) holds with the a-priori range ($`B\le2`$ for $`\tilde F\in[0,1]`$). Its range term is then of order $`7B\log(4M^2/\delta)/(3r)`$, which can far exceed the gaps between nearby candidates for any affordable $`r`$. The observed range of the scores is not a valid substitute. What is missing is a provable range much smaller than the a-priori one, for example the pathwise Lipschitz bound of [Proposition 8.10(c)](#s8-10) with a certified $`\bar\lambda`$ for real training tails.
- **Calibration transfer.** A testable condition under which $`(\alpha,\varepsilon_{\rm sur})`$, fitted on calibration pairs, remain valid on new pairs and at later states.
- Until both are settled, Theorem 8.8 serves in practice as an unbiased, low-variance ranking score, and as a rigorous certificate only when $`B`$ and $`(\alpha,\varepsilon_{\rm sur})`$ are justified independently.

## Related work

- [Theorem 8.1](08_verification.md#s8-1) is linear-programming and norm duality applied to the operational discrepancy; no new duality principle is claimed.
- [Theorems 8.2](08_verification.md#s8-2)–[8.3](08_verification.md#s8-3) and [Proposition 8.4](#s8-4) use Hoeffding's inequality, union bounds and McDiarmid's inequality. Selecting by a lower confidence bound to guarantee no degradation is the logic of high-confidence policy improvement (Thomas et al., 2015).
- [Theorem 8.8](#s8-8) combines an empirical Bernstein bound (Maurer & Pontil, 2009; Audibert et al., 2009) with common random numbers, a classical variance-reduction device in simulation (Glasserman, 2004). Its content is the estimand: the finite one-round replacement contrast at the actual complete learner state, measured on a calibrated surrogate tail.
- First-order data-attribution scores, such as influence functions (Koh & Liang, 2017), TracIn (Pruthi et al., 2020), In-Run Data Shapley (Wang et al., 2025) and metagradients (Engstrom et al., 2025), use gradients of an objective with respect to training data or hyperparameters. [Corollary 8.9](#s8-9) states when such first-order rankings are guaranteed and why they cannot certify pairs below the remainder. Datamodels (Ilyas et al., 2022) estimate finite retraining effects by regression over many training runs; Theorem 8.8 measures one-round finite effects at a fixed state with paired tails.

**References.**

- Audibert, J.-Y., Munos, R., Szepesvári, C. (2009). Exploration–exploitation tradeoff using variance estimates in multi-armed bandits. *Theoretical Computer Science* 410, no. 19.
- Engstrom, L., Ilyas, A., Chen, B., Feldmann, A., Moses, W., Madry, A. (2025). Optimizing ML training with metagradient descent. arXiv:2503.13751.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Ilyas, A., Park, S. M., Engstrom, L., Leclerc, G., Madry, A. (2022). Datamodels: Predicting predictions from training data. ICML.
- Koh, P. W., Liang, P. (2017). Understanding black-box predictions via influence functions. ICML.
- Maurer, A., Pontil, M. (2009). Empirical Bernstein bounds and sample variance penalization. COLT.
- Pruthi, G., Liu, F., Kale, S., Sundararajan, M. (2020). Estimating training data influence by tracing gradient descent. NeurIPS.
- Thomas, P., Theocharous, G., Ghavamzadeh, M. (2015). High confidence policy improvement. ICML.
- Wang, J. T., Mittal, P., Song, D., Jia, R. (2025). Data Shapley in one training run. ICLR.
