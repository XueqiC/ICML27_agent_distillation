# Corollary 8.9 · The first-order version, its margin criterion, and its cap

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-09_first_order_version_margin_criterion.md) · [Proof](../proofs/08-09_first_order_version_margin_criterion.md) · [Experiments](../experiments/08-09_first_order_version_margin_criterion.md) · [中文](08-09_first_order_version_margin_criterion_ZH.md)

**What is proved here.** The margin criterion from a second-order Taylor bound, the measured first-order certificate from Taylor's theorem along scaled increments plus the empirical Bernstein argument of Theorem 8.8, the cap by chaining these on the confidence event, and the oracle-score statement.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- Rounds are $`t=0,\dots,R-1`$, and $`z`$ is a complete learner state. For a first-order optimizer the actual update is $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$.
- At state $`z`$ in round $`t`$ the reference emits $`Y\sim P^0_{t,z}`$, and candidate $`j\in\lbrace1,\dots,M\rbrace`$ emits $`Y\sim P^j_{t,z}`$. The kernel of candidate $`j`$ is $`K^j_t`$.
- The reference continuation is $`V^0_{t+1}`$, with $`V^0_R=F_E\in[0,1]`$. The response is $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$. Integrating the message gives the kernel identity

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- $`L_E(j)=\mathbb E\,F_E(Z^j_R)`$ for the process driven by candidate $`j`$. All processes start from the same initial law of $`Z_0`$.
- The *local advantage* of candidate $`j`$ at $`(t,z)`$ is

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- It is the effect on the final score of replacing one reference round by one round of candidate $`j`$, followed by the reference tail. By [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2), the whole-run difference is a sum of such terms over the states the alternative process reaches.

*The paired-replacement score* ([Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6)).

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

*Its assumptions* ([Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7)).

- **(I) Independent replicas.** Conditionally on the history that produced $`z`$, the vectors $`(Z^{jk}_\ell)_{j,k}`$, $`\ell=1,\dots,r`$, are i.i.d.; $`r`$ is fixed before the scores are observed.
- **(B) Bounded scores.** For every pair, $`Z^{jk}_\ell`$ lies almost surely in a known interval of length $`B`$. For example $`\tilde F\in[0,1]`$ gives $`B\le2`$.
- **(A) Surrogate alignment.** There are $`\alpha>0`$ and $`\varepsilon_{\rm sur}\ge0`$ such that, for all pairs,

```math
\Big\lvert\big(L_F(P^j)-L_F(P^k)\big)-\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)\Big\rvert\le\alpha\,\varepsilon_{\rm sur}.
```

$`\varepsilon_{\rm sur}`$ absorbs both the difference between the surrogate and the task objective and the truncation of the tail.

**Statement ([Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9)).** Use the setting of [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6) and let $`\mu^j=\mathbb E\Delta^j`$ and $`m^j_2=\mathbb E\lVert\Delta^j\rVert^2`$.

- **(a) Margin criterion.** Let $`V`$ be a continuation that is twice differentiable along the reached segments with $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$. If $`\lvert\langle\nabla V(z),\mu^j-\mu^k\rangle\rvert>\tfrac{M_2}{2}(m^j_2+m^k_2)`$, then $`(K^j_t-K^k_t)V(z)`$ has the sign of $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$. At run level, let $`T_{jk}=\sum_t\mathbb E_{d^k_t}\langle\nabla V^j_{t+1}(Z_t),\mu^j_t-\mu^k_t\rangle`$ and $`B_{jk}=\sum_t\mathbb E_{d^k_t}\tfrac{M_{2,t}}2(m^j_{2,t}+m^k_{2,t})`$ be the first-order term and the remainder bound of [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) with $`P_j`$ in the role of $`P`$; here $`d^k_t`$ is the state law at round $`t`$ of the process driven by $`P_k`$. If $`\lvert T_{jk}\rvert>B_{jk}`$, then $`L(P_j)-L(P_k)`$ has the sign of $`T_{jk}`$.
- **(b) Measured first-order certificate.** Let $`\tilde V(u)=\mathbb E_\omega\varphi_\omega(u)`$ be twice differentiable along $`\lbrace su:s\in[0,1]\rbrace`$ for the reached increments $`u`$, with $`\lvert D^2\tilde V(su)[u,u]\rvert\le\tilde M\lVert u\rVert^2`$, and let $`\tilde s^{jk}=\langle\nabla\tilde V(0),\mu^j-\mu^k\rangle`$. For a step $`h\in(0,1]`$ define $`Z^{jk}_\ell(h)=\big(\varphi_{\omega_\ell}(h\Delta^j_\ell)-\varphi_{\omega_\ell}(h\Delta^k_\ell)\big)/h`$ with mean $`\hat S^{jk}(h)`$, and assume (I), (A), and (B) with range length $`B_h`$. Let $`e^{jk}_{\rm stat}(h)`$ be the radius of [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) with $`B_h`$ in place of $`B`$, and

```math
e^{jk}_{\rm fd}(h)=\tfrac12h\tilde M\big(m^j_2+m^k_2\big),\qquad\tilde e^{jk}_2=\tfrac12\tilde M\big(m^j_2+m^k_2\big).
```

Then with probability at least $`1-\delta`$, simultaneously for all ordered pairs,

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)-\tilde e^{jk}_2-\varepsilon_{\rm sur}\big).
```

- **(c) Cap.** On the event of (b), a pair is certified by (b) only if $`\tilde s^{jk}>\tilde e^{jk}_2+\varepsilon_{\rm sur}`$. A pair whose first-order term does not exceed the second-order remainder bound is never certified at first order, whatever its true gap. In particular this holds for pairs whose mean increments agree along $`\nabla\tilde V(0)`$, which differ only through second and higher moments of the increment; [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) can certify such pairs. In population terms, (b) certifies the lower bound $`\tilde s^{jk}-\tilde e^{jk}_2\le\tilde L(P^j)-\tilde L(P^k)`$, the quantity that Theorem 8.8 estimates directly.
- **(d) Oracle scores are not certificates.** A score $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$ computed with the exact continuation gradient of the task, as in controlled experiments that differentiate through the whole reference tail, satisfies (a) wherever the margin holds. It is not a measurable certificate: $`\nabla V(z)`$ and $`M_2`$ are not available from feedback samples, and checking the margin needs both. A cheap score ranks correctly by (a) only if (i) its difference equals the first-order term with the optimizer-transformed mean increment of the actual learner ([Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)) at the actual state, and (ii) the margin holds. Scores built from the raw mean gradient, or with coefficients shared across states and tasks, violate (i); any score used where the margin fails violates (ii).

*Proof.*

**(a)** *One boundary.* By the Taylor bound used in the proof of [Proposition 8.4(b)](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4), $`(K^j_t-K^k_t)V(z)=s+\rho`$ with $`s=\langle\nabla V(z),\mu^j-\mu^k\rangle`$ and $`\lvert\rho\rvert\le\tfrac{M_2}2(m^j_2+m^k_2)<\lvert s\rvert`$. If $`s>0`$ then $`s+\rho>s-\lvert s\rvert=0`$; if $`s<0`$ then $`s+\rho<0`$. *Run level.* [Proposition 5.1(b)](../statements/05-01_optimizer_transformed_expansion.md#s5-1) with $`P_j`$ as $`P`$ and $`P_k`$ as $`Q`$ gives $`L(P_j)-L(P_k)=T_{jk}+\mathcal E`$ with $`\lvert\mathcal E\rvert\le B_{jk}<\lvert T_{jk}\rvert`$; the same sign argument applies.

**(b)** Taylor's theorem with integral remainder along $`s\mapsto\tilde V(shu)`$ gives, for every reached increment $`u`$,

```math
\Big\lvert\tilde V(hu)-\tilde V(0)-h\langle\nabla\tilde V(0),u\rangle\Big\rvert\le\tfrac12h^2\tilde M\lVert u\rVert^2 .
```

- *Finite-difference bias.* As in [Theorem 8.8(a)](../statements/08-08_crn_paired_replacement_certificate.md#s8-8), $`\mathbb EZ^{jk}_\ell(h)=\big(\mathbb E\tilde V(h\Delta^j)-\mathbb E\tilde V(h\Delta^k)\big)/h`$. Taking expectations of the display for $`u=\Delta^j`$ and $`u=\Delta^k`$ and dividing by $`h`$ gives $`\lvert\mathbb EZ^{jk}_\ell(h)-\tilde s^{jk}\rvert\le e^{jk}_{\rm fd}(h)`$.
- *Remainder.* The display with $`h=1`$ gives $`\tilde L(P^j)-\tilde L(P^k)=\tilde s^{jk}+\tilde\rho`$ with $`\lvert\tilde\rho\rvert\le\tilde e^{jk}_2`$.
- *Concentration.* The argument of [Theorem 8.8(b)](../statements/08-08_crn_paired_replacement_certificate.md#s8-8), applied to $`Z^{jk}_\ell(h)`$ with range $`B_h`$, gives an event of probability at least $`1-\delta`$ on which $`\mathbb EZ^{jk}_\ell(h)\ge\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)`$ for all ordered pairs.
- On that event, $`\tilde L(P^j)-\tilde L(P^k)\ge\tilde s^{jk}-\tilde e^{jk}_2\ge\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)-\tilde e^{jk}_2`$, and (A) transfers this to the task as in Theorem 8.8.

**(c)** Certification by (b) means $`\hat S^{jk}(h)>e^{jk}_{\rm stat}(h)+e^{jk}_{\rm fd}(h)+\tilde e^{jk}_2+\varepsilon_{\rm sur}`$. On the event of (b), $`\tilde s^{jk}\ge\mathbb EZ^{jk}_\ell(h)-e^{jk}_{\rm fd}(h)\ge\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)>\tilde e^{jk}_2+\varepsilon_{\rm sur}`$. If the mean increments agree along $`\nabla\tilde V(0)`$, then $`\tilde s^{jk}=0`$ and no certification is possible. The last sentence of (c) is the remainder step of (b).

**(d)** The first claim is (a). For (i): if the score difference equals $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$ and the margin holds, (a) gives the sign. If the score differs from this term, (a) does not control the difference; if the margin fails, the remainder can reverse the sign. $`\square`$
