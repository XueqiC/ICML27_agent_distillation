# Theorem 8.3 · Actual-state adaptive audit; fidelity versus improvement

*Statement*

[← Framework](../README.md) · [Statement](../statements/08-03_actual_state_adaptive_audit_fidelity.md) · [Proof](../proofs/08-03_actual_state_adaptive_audit_fidelity.md) · [Experiments](../experiments/08-03_actual_state_adaptive_audit_fidelity.md) · [中文](08-03_actual_state_adaptive_audit_fidelity_ZH.md)

**In one paragraph.** Instead of auditing frozen cells, one can audit at the states training actually reaches: at each audited round, run paired full-tail experiments for each candidate against the reference, and select by a lower confidence bound. This theorem shows that the resulting training procedure is never worse than the reference by more than the confidence level, keeps the certified improvement, and needs an explicit, large number of paired replicas. It also separates fidelity to the reference from improvement over the initial student.

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

<a id="s8-3"></a>

**Theorem 8.3 (Actual-state adaptive audit; fidelity versus improvement).**

- **(a) Fidelity.** If $`L_E(0)-J_E(\pi_0)\ge\gamma`$ and $`L_E(0)-L_E(S)\le\varepsilon`$, then $`L_E(S)-J_E(\pi_0)\ge\gamma-\varepsilon`$. Here $`J_E(\pi_0)`$ is the value of the initial student; $`\varepsilon`$ can be the certificate of [Theorem 8.2(c)](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2), and $`\gamma`$ the measured gain of reference training from the $`K`$ demonstrations.
- **(b) Signed identity.** For every history-dependent provider policy $`S`$ with the common initial law,

```math
L_E(S)-L_E(0)=\sum_{t=0}^{R-1}\mathbb E_{d^S_t}\,A_t(Z_t,J_t).
```

- **(c) Adaptive audit.** (i) With probability at least $`1-\delta`$, $`\sum_tA_t(Z_t,J_t)\ge\sum_t\Gamma_t`$. (ii) Unconditionally over audits and training,

```math
L_E(\mathrm{adaptive})-L_E(0)\ge-\delta,\qquad L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_t\Gamma_t-2\delta .
```

- **(d) Cost.** Radius $`e_t\le e`$ needs $`n_t\ge\lceil2\log(2M_t/\delta_t)/e^2\rceil`$ paired replicas per candidate. For $`M=4`$, $`\delta=0.05`$, $`e=0.1`$ this is $`\lceil2\log160/0.01\rceil=1016`$. The count is sufficient, not necessary.

**What it says.** At the states training actually reaches, run paired full-tail experiments for each candidate against the reference and select by a lower confidence bound. The resulting training procedure is never worse than the reference by more than the confidence level, and it keeps the certified improvement. The required number of paired replicas is explicit and large.

**Intuition.** It measures directly whether replacing this one round helps, with no extrapolation to unseen states; the price is many full training tails.

**Role in the framework.** It is the reliable but expensive reference audit, used to calibrate cheaper measurements such as the surrogate of Theorem 8.8.

**Comments on the statement.**

- **The lower confidence bound is needed.** Selecting by the point estimate $`\widehat A_t(j)`$ can have strictly negative expected gain even with exact full-tail audits, and a slightly inferior candidate can pass a lower-confidence-bound gate with positive probability. The $`-\delta`$ term in (c)(ii) therefore cannot be removed: an exact no-degradation statement ($`\ge0`$) is false in general. An exact finite instance of the first fact is computed in the [experiments](../experiments/08-03_actual_state_adaptive_audit_fidelity.md).
- **Unconditional, not conditional.** (c)(ii) averages over audit outcomes. Conditioning the terminal value on "all audits succeeded" is invalid in general, because later audit events depend on the states reached; an exact instance is computed in the [experiments](../experiments/08-03_actual_state_adaptive_audit_fidelity.md).
- **Stopped refinement.** If $`\tau`$ is the first failed audit (capped at $`R`$), then $`L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_{t<\tau}\Gamma_t-\delta`$.
- **What is removed and what is not.** Auditing at the actual state removes the transport term of [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2). It does not remove the restoration, full-tail cost or independence requirements. Reusing a winning audit replica as the update, or auditing with a short tail, changes the estimand; a short tail needs an explicit surrogate allowance, which is what [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) supplies.
- **Fidelity is not improvement.** (a) inherits only the reference's measured gain $`\gamma`$. A high-fidelity copy of an ineffective reference does not improve the initial student.


# §8 Verification certificate (Theorem C) — Proofs

[← Framework](../README.md) · Proofs · [Experiments](../experiments/08-03_actual_state_adaptive_audit_fidelity.md) · 中文

This page contains Proposition 8.4 through Remark 8.17, including Theorem C; the sibling page supplies the preceding measurement, transport and adaptive-audit proofs and the shared setting.

**Derivation outline.**

- Conditional on the pre-audit history, the paired contrasts are i.i.d. in $`[-1,1]`$ with mean $`A_t(Z_t,j)`$; Hoeffding and a union bound over candidates give a per-round event, and the failure probabilities add up to at most $`\delta`$ along the run.
- On each round's event the selected advantage is at least $`\Gamma_t\ge0`$.
- The fresh update makes $`A_t`$ the conditional drift of $`V^0`$, so the terminal score telescopes into drifts plus martingale differences; taking expectations gives the unconditional bounds.

**Illustration.** Selecting by the point estimate instead of the lower bound can lose in expectation even when every audit uses the exact full tail: a slightly worse candidate wins the noisy comparison often enough.

**Uses.**

- [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2) — telescoping into one-round advantages
- [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2) — the frozen audit whose transport term this removes

**Used by.**

- [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4) — full-tail calibration of first-order constants
- [Proposition 8.10](../statements/08-10_cost_and_crn.md#s8-10) — cost comparison
- [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11) — certified bound inside the adaptive audit
- [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2) — selection step of the method

## Related work

- [Theorems 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)–[8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) and [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4) use Hoeffding's inequality ([Hoeffding, 1963](#ref-hoeffding1963probability)), union bounds and McDiarmid's inequality ([McDiarmid, 1989](#ref-mcdiarmid1989method)). Selecting by a lower confidence bound to guarantee no degradation is the logic of high-confidence policy improvement ([Thomas et al., 2015](#ref-thomas2015high)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-mcdiarmid1989method"></a>

**Colin McDiarmid (1989).** On the method of bounded differences. In *Surveys in Combinatorics, 1989*, London Mathematical Society Lecture Note Series 141, 148–188, Cambridge University Press. DOI [10.1017/CBO9781107359949.008](https://doi.org/10.1017/CBO9781107359949.008).

<a id="ref-thomas2015high"></a>

**Philip S. Thomas, Georgios Theocharous, Mohammad Ghavamzadeh (2015).** High confidence policy improvement. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37:2380–2388. [proceedings.mlr.press/v37/thomas15](https://proceedings.mlr.press/v37/thomas15.html).
