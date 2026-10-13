# Theorem 8.8 · CRN paired-replacement certificate; Theorem C

*Statement*

[← Framework](../README.md) · [Statement](../statements/08-08_crn_paired_replacement_certificate.md) · [Proof](../proofs/08-08_crn_paired_replacement_certificate.md) · [Experiments](../experiments/08-08_crn_paired_replacement_certificate.md) · [中文](08-08_crn_paired_replacement_certificate_ZH.md)

**In one paragraph.** Theorem C turns the paired-replacement score into a certificate. At an audited state it measures the finite effect of replacing one training round by one round of each candidate, on a surrogate tail shared through common random numbers, and certifies with an empirical Bernstein radius which candidates are better than which. Nothing is linearized: the certificate needs no continuation gradient, no curvature constant and no second-order remainder.

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

<a id="s8-8"></a>

**Theorem 8.8 (CRN paired-replacement certificate; Theorem C).** Under [Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7), let $`\hat V^{jk}=\frac1{r-1}\sum_\ell(Z^{jk}_\ell-\hat S^{jk})^2`$ be the unbiased sample variance and

```math
e^{jk}_{\rm stat}=\sqrt{\frac{2\hat V^{jk}\log(4M^2/\delta)}{r}}+\frac{7B\log(4M^2/\delta)}{3(r-1)} .
```

- **(a) Exactness.** $`\mathbb E\,Z^{jk}_\ell=\tilde L(P^j)-\tilde L(P^k)`$. No first-order approximation is involved.
- **(b) Certificate.** With probability at least $`1-\delta`$, simultaneously for all ordered pairs $`(j,k)`$, $`j\ne k`$,

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big).
```

- **(c) Decision rule.** Declaring "$`j`$ is better than $`k`$" whenever $`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ makes no wrong declaration, over all pairs simultaneously, with probability at least $`1-\delta`$. With $`k=0`$ this certifies $`A_t(z,j)>0`$.
- The certificate needs no continuation gradient $`\nabla V`$, no curvature constant and no second-order remainder. The only constants to calibrate are $`(\alpha,\varepsilon_{\rm sur})`$ ([Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11)).

**What it says.** At an audited state, give each candidate one round and then run the same surrogate tail with the same random numbers for every candidate. The mean paired score difference estimates the finite effect of the replacement without bias, and an empirical Bernstein radius plus a surrogate-calibration term turns it into a certificate that holds for all candidate pairs at once.

**Intuition.** Instead of linearizing and then bounding the error of the linearization, measure the finite effect directly; sharing the random numbers cancels the tail noise that two independent runs would add.

**Role in the framework.** It replaces the oracle continuation gradient by a measurement, and it is the selection step of the method. Its rigor rests on a known score range and a calibrated surrogate, which are the open problems of Remark 8.12.

**Comments on the statement.**

- **What is estimated.** The estimand is the finite one-round replacement contrast of [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2), measured on a surrogate tail. Nothing is linearized, so the second-moment channel of §7, which is nonlinear and saturating in the message magnitude ([Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2)), is measured as it is.
- **Where the assumptions bite.** (I) requires fresh $`\omega_\ell`$ and fresh messages per replica at a fixed state. (B) needs a range that is known in advance: the observed range of the scores is not a valid $`B`$. (A) is a statement about population differences and has to be calibrated ([Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11)).

**Derivation outline.**

- Exactness is linearity of expectation: each term depends only on the marginal law of one candidate's increment and the independent tail randomness, so coupling the candidates changes nothing in the mean.
- Maurer and Pontil's empirical Bernstein bound for $`[0,1]`$ variables, rescaled to an interval of length $`B`$, gives each one-sided bound; there are at most $`2M^2`$ ordered pairs, which yields $`\log(4M^2/\delta)`$ after a union bound.
- The alignment assumption transfers the surrogate bound to the task.

**Illustration.** Two candidates whose mean increments coincide but whose second moments differ have zero first-order difference; Theorem 8.8 still sees their finite difference, because nothing is linearized.

**Uses.**

- [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6) — the paired-replacement score
- [Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7) — independence, range and alignment
- [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2) — one-round replacement contrasts
- [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) — full-tail audits for calibration

**Used by.**

- [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9) — first-order variant and its cap
- [Proposition 8.10](../statements/08-10_cost_and_crn.md#s8-10) — cost and replica count
- [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11) — use inside the adaptive audit
- [Remark 7.5](../statements/07-05_where_first_order_rankers_fail.md#s7-5) — where first-order rankers fail
- [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2) — selection step of the method

## Related work

- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) combines an empirical Bernstein bound ([Maurer & Pontil, 2009](#ref-maurer2009empirical); [Audibert et al., 2009](#ref-audibert2009exploration)) with common random numbers, a classical variance-reduction device in simulation ([Glasserman, 2004](#ref-glasserman2004monte)). Its content is the estimand: the finite one-round replacement contrast at the actual complete learner state, measured on a calibrated surrogate tail.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-audibert2009exploration"></a>

**Jean-Yves Audibert, Rémi Munos, Csaba Szepesvári (2009).** Exploration–exploitation tradeoff using variance estimates in multi-armed bandits. *Theoretical Computer Science* 410, no. 19, 1876–1902. DOI [10.1016/j.tcs.2009.01.016](https://doi.org/10.1016/j.tcs.2009.01.016).

<a id="ref-glasserman2004monte"></a>

**Paul Glasserman (2004).** Monte Carlo Methods in Financial Engineering. Springer, New York (Stochastic Modelling and Applied Probability 53). DOI [10.1007/978-0-387-21617-1](https://doi.org/10.1007/978-0-387-21617-1).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).
