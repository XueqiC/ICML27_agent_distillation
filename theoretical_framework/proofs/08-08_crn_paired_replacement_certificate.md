# Theorem 8.8 · CRN paired-replacement certificate; Theorem C

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-08_crn_paired_replacement_certificate.md) · [Proof](../proofs/08-08_crn_paired_replacement_certificate.md) · [Experiments](../experiments/08-08_crn_paired_replacement_certificate.md) · [中文](08-08_crn_paired_replacement_certificate_ZH.md)

**What is proved here.** Exactness in expectation by linearity (coupling does not change marginals), a rescaled empirical Bernstein bound for every ordered pair with a union bound over at most 2M² pairs, and the transfer to the task through surrogate alignment.

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

**Statement ([Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8)).** Under [Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7), let $`\hat V^{jk}=\frac1{r-1}\sum_\ell(Z^{jk}_\ell-\hat S^{jk})^2`$ be the unbiased sample variance and

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

*Proof.*

**(a)** By linearity of expectation, $`\mathbb EZ^{jk}_\ell=\mathbb E\varphi_{\omega_\ell}(\Delta^j_\ell)-\mathbb E\varphi_{\omega_\ell}(\Delta^k_\ell)`$. Each term depends only on the joint law of $`(\omega_\ell,\Delta^j_\ell)`$, which is the product of the law of $`\omega`$ and the marginal law of $`\Delta^j`$, whatever coupling is used across candidates. Hence each term equals $`\tilde L(P^j)`$, respectively $`\tilde L(P^k)`$. With exact enumeration over a finite support the same holds with the inner average in place of the draw.

**(b)** *Empirical Bernstein bound.* Maurer and Pontil ([2009](#ref-maurer2009empirical)) prove the following empirical Bernstein bound. For $`n\ge2`$ i.i.d. variables $`X_i\in[0,1]`$ with unbiased sample variance $`V_n`$, with probability at least $`1-\delta'`$,

```math
\mathbb EX-\frac1n\sum_iX_i\le\sqrt{\frac{2V_n\log(2/\delta')}{n}}+\frac{7\log(2/\delta')}{3(n-1)} .
```

- If $`Z`$ takes values in an interval $`[a,a+B]`$, apply this to $`X=(Z-a)/B\in[0,1]`$, whose sample variance is $`V_n(Z)/B^2`$, and multiply by $`B`$. This gives the same bound for $`Z`$ with $`\sqrt{2V_n(Z)\log(2/\delta')/n}+7B\log(2/\delta')/(3(n-1))`$.
- *Events.* For each ordered pair $`(j,k)`$, apply the bound to $`X=-Z^{jk}`$ (values in an interval of length $`B`$, same sample variance) to obtain the lower tail $`\tilde L(P^j)-\tilde L(P^k)\ge\hat S^{jk}-e^{jk}_{\rm stat}`$. There are $`(M+1)M`$ ordered pairs, and $`(M+1)M\le2M^2`$ for $`M\ge1`$. Use $`\delta'=\delta/(2M^2)`$ for each, so $`\log(2/\delta')=\log(4M^2/\delta)`$. A union bound gives probability at least $`1-\delta`$ that all hold. The event for $`(k,j)`$ is the upper tail for $`(j,k)`$, so the bound is two-sided for every unordered pair.
- *Transfer to the task.* On this event, (A) gives $`L_F(P^j)-L_F(P^k)\ge\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)-\alpha\varepsilon_{\rm sur}\ge\alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big)`$.

**(c)** On the event of (b), $`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ implies $`L_F(P^j)-L_F(P^k)>0`$, because $`\alpha>0`$. With $`k=0`$, $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$. $`\square`$

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).
