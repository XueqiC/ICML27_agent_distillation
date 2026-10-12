# Remark 8.11 · Calibrating the surrogate; transfer along training

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-11_surrogate_calibration_and_transfer.md) · [Experiments](../experiments/08-11_surrogate_calibration_and_transfer.md) · [中文](08-11_surrogate_calibration_and_transfer_ZH.md)

**What is tested here.** The checks summarized below also test this item. Each summary states what the check tests for this item, its setup and its key numbers; the full description is on the page linked in its heading.

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

**Adaptive audit.** A *provider policy* $`S`$ chooses at each round a kernel index $`J_t`$ as a function of the history.

- Let $`\mathcal H_t`$ be the history just before the audit of round $`t`$. Conditional on $`\mathcal H_t`$, freeze the actual state $`Z_t`$, a finite set of $`M_t`$ non-reference candidate kernels, a replica count $`n_t`$ and a confidence allocation $`\delta_t`$, with $`\sum_t\delta_t\le\delta`$ almost surely.
- For each candidate, run $`n_t`$ independent paired experiments from $`Z_t`$. Each applies the candidate update or the reference update, then the **full** reference tail and real deployment. Each arm has its correct marginal law, and the paired contrast lies in $`[-1,1]`$.
- Let $`\widehat A_t(j)`$ be the mean contrast and

```math
e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},\qquad\ell_t(j)=\widehat A_t(j)-e_t,\qquad\ell_t(0)=0 .
```

- Select $`J_t\in\arg\max_j\ell_t(j)`$ and set $`\Gamma_t=\ell_t(J_t)\in[0,1]`$. Then apply a **fresh** update from $`K^{J_t}_t`$, discarding the audit replicas. Unaudited rounds use the reference and set $`\Gamma_t=0`$.
- Audits leave the actual learner and environment state unchanged, and the reference ignores audit records. $`\mathcal F_t\supseteq\mathcal H_t`$ is the history after the audit and the selection, before the fresh update.

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

**Statement under test ([Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11)).**

- **Calibration.** On a set of calibration pairs, estimate $`L_F(P^j)-L_F(P^k)`$ by paired full-tail task evaluations ([Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3)), regress them on $`\hat S^{jk}`$, and read off $`\hat\alpha`$ and a residual bound $`\hat\varepsilon_{\rm sur}`$. Applying $`(\hat\alpha,\hat\varepsilon_{\rm sur})`$ to other pairs requires an exchangeability assumption: the calibration pairs and the new pairs come from the same family. This is an assumption, not a theorem, and the calibration's own statistical error must be added.
- **Use inside the adaptive audit.** At an audited round, replace the full-tail lower confidence bound of [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) by $`\Gamma_t=\min\lbrace1,\alpha(\hat S^{J_t0}-e^{J_t0}_{\rm stat}-\varepsilon_{\rm sur})\rbrace`$ for the selected candidate, with $`\Gamma_t=0`$ when no candidate is certified against the reference. If (A) holds at every audited state with the declared constants and the confidence allocations sum to $`\delta`$, Theorem 8.3(c) holds unchanged. Its proof uses only $`A_t\ge\Gamma_t`$ on the audit event and $`0\le\Gamma_t\le1`$, and [Theorem 8.8(b)](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) supplies the first, since $`A_t(z,j)=L_F(P^j)-L_F(P^0)`$.
- **Transfer along training.** The certificate holds at the audited state. Using the selected candidate at later states needs the transport term of [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2), or a re-audit at the next checkpoint that checks whether the ranking persists (an a-posteriori check).

### [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate.md#c8-14) · Finite CRN paired replacement on the controlled panel (Theorem 8.8, Proposition 8.10, Remarks 8.11–8.12)

- **What it tests for Remark 8.11.** Whether surrogate-alignment constants fitted on calibration states transfer to held-out states.
- **Setup.** The short surrogate (one reference update) is calibrated before any held-out measurement on eight separate calibration cells (288 pairs) that share no complete learner state with the held-out cells: a positive zero-intercept least-squares fit gives $`\alpha`$, and $`\varepsilon_{\rm sur}`$ is the largest calibration residual divided by $`\alpha`$. The constants are then applied to 1,728 held-out pairs.
- **Key numbers.** $`\alpha=0.881`$ and $`\varepsilon_{\rm sur}=0.0083`$ at $`r=64`$ ($`0.886`$, $`0.0083`$ at $`r=256`$). With the observed range in place of the a-priori range, a diagnostic only, the calibrated surrogate certifies 23–28% of held-out pairs with 22–28 wrong certificates; half of them fall in the held-out learning-rate split.
- **Verdict.** Calibration transfer can fail, so it must be treated as an assumption to be tested, as the remark states.

## Reproduce

Reproduction commands and result files are listed on the pages of the primary checks: [Theorem 8.8](../experiments/08-08_crn_paired_replacement_certificate.md).
