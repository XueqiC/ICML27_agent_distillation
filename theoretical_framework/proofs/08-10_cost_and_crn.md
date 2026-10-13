# Proposition 8.10 · Cost, and what common random numbers buy

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-10_cost_and_crn.md) · [Proof](../proofs/08-10_cost_and_crn.md) · [Experiments](../experiments/08-10_cost_and_crn.md) · [中文](08-10_cost_and_crn_ZH.md)

**What is proved here.** Counting, rearranging the range term of the radius, and the pathwise Lipschitz bound with the law of total variance.

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

**Statement ([Proposition 8.10](../statements/08-10_cost_and_crn.md#s8-10)).**

- **(a) Cost.** One audited state needs $`(M+1)r`$ surrogate tails, plus the task-evaluated tails used for calibration ([Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11)). A full-tail paired audit with the radius of [Theorem 8.3(d)](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) needs $`2n_t`$ full tails per candidate.
- **(b) Replicas for the range term.** The range term of $`e_{\rm stat}`$ is at most $`\zeta B`$ if and only if

```math
r\ \ge\ 1+\frac{7\log(4M^2/\delta)}{3\zeta}.
```

For example, $`M=8`$, $`\delta=0.05`$ and $`\zeta=0.2`$ give $`\log(4M^2/\delta)=\log5120\approx8.54`$ and $`r\ge101`$. With fewer replicas the range term dominates the radius.

- **(c) Variance under common random numbers.** Suppose $`\varphi_\omega`$ is pathwise Lipschitz on the reached increments, with constant $`\lambda_\omega`$: $`\lvert\varphi_\omega(u)-\varphi_\omega(u')\rvert\le\lambda_\omega\lVert u-u'\rVert`$. Then

```math
\lvert Z^{jk}_\ell\rvert\le\lambda_{\omega_\ell}\,\lVert\Delta^j_\ell-\Delta^k_\ell\rVert,\qquad\mathrm{Var}\,Z^{jk}_\ell\le\mathbb E\big[\lambda_\omega^2\,\lVert\Delta^j-\Delta^k\rVert^2\big].
```

With independent tails $`\omega,\omega'`$ for the two arms (and independent messages), the variance of $`\varphi_\omega(\Delta^j)-\varphi_{\omega'}(\Delta^k)`$ is $`\mathrm{Var}\,\varphi_\omega(\Delta^j)+\mathrm{Var}\,\varphi_{\omega'}(\Delta^k)\ge\mathbb E\,\mathrm{Var}(\varphi_\omega(\Delta^j)\mid\Delta^j)+\mathbb E\,\mathrm{Var}(\varphi_{\omega'}(\Delta^k)\mid\Delta^k)`$, which contains the tail noise and does not vanish as the candidates approach each other. If moreover $`\sup_\omega\lambda_\omega\le\bar\lambda`$ and $`\lVert\Delta^j-\Delta^k\rVert\le b_{jk}`$ almost surely under the coupling, then $`B=2\bar\lambda b_{jk}`$ is a valid range in [Assumption 8.7(B)](../statements/08-07_certificate_assumptions.md#s8-7).

*Proof.* (a) is counting: each replica evaluates each of the $`M+1`$ candidates once. (b) The range term of the empirical Bernstein radius ([Maurer & Pontil, 2009](#ref-maurer2009empirical)), $`7B\log(4M^2/\delta)/(3(r-1))\le\zeta B`$, rearranges to the display; for the example, $`7\cdot8.541/(3\cdot0.2)\approx99.6`$, so $`r\ge100.6`$. (c) The first inequality is the Lipschitz condition at $`u=\Delta^j_\ell`$, $`u'=\Delta^k_\ell`$. Then $`\mathrm{Var}\,Z\le\mathbb EZ^2`$. With independent arms the variance of a difference is the sum of variances, and the law of total variance gives $`\mathrm{Var}\,\varphi_\omega(\Delta^j)\ge\mathbb E\,\mathrm{Var}(\varphi_\omega(\Delta^j)\mid\Delta^j)`$. Under the two uniform bounds, $`Z^{jk}_\ell\in[-\bar\lambda b_{jk},\bar\lambda b_{jk}]`$. $`\square`$

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).
