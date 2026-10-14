# Remark 8.17 · Open problems

*Statement*

[← Framework](../README.md) · [Statement](../statements/08-17_open_problems.md) · [Experiments](../experiments/08-17_open_problems.md) · [中文](08-17_open_problems_ZH.md)

**In one paragraph.** The certificates of §8 are valid as stated, but whether they are decisive in practice depends on questions this framework does not yet answer. Two concern Theorem C: a provable range for the paired scores much smaller than the a-priori one, and a testable condition under which the surrogate calibration transfers. Four concern maintaining a response interface during continued training ([Propositions 8.12–8.15](08-12_finite_catalog_spectral_certificate.md#s8-12), [Corollary 8.16](08-16_composed_guarantee.md#s8-16)): how much independent information one continuation branch provides, how to reuse across cells without a full continuation per cell, how to obtain certificates whose radius scales with the effect, and whether a short continuation predicts a full one on the real learner.

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

<a id="s8-17"></a>

**Remark 8.17 (Open problems).**

- **A provable range.** [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) holds with the a-priori range ($`B\le2`$ for $`\tilde F\in[0,1]`$). Its range term is then of order $`7B\log(4M^2/\delta)/(3r)`$, which can far exceed the gaps between nearby candidates for any affordable $`r`$. The observed range of the scores is not a valid substitute. What is missing is a provable range much smaller than the a-priori one, for example the pathwise Lipschitz bound of [Proposition 8.10(c)](../statements/08-10_cost_and_crn.md#s8-10) with a certified $`\bar\lambda`$ for real training tails.
- **Calibration transfer.** A testable condition under which $`(\alpha,\varepsilon_{\rm sur})`$, fitted on calibration pairs, remain valid on new pairs and at later states.
- Until both are settled, Theorem 8.8 serves in practice as an unbiased, low-variance ranking score, and as a rigorous certificate only when $`B`$ and $`(\alpha,\varepsilon_{\rm sur})`$ are justified independently.
- **Information per continuation branch, and identifying closure.** A closed predictive test space of dimension $`d`$ does not require $`d`$ continuation protocols: one branch can output several metrics, so spanning needs at least $`\lceil d/p\rceil`$ protocols for $`p`$ metrics per branch, and identifying closure needs extension tests ([Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12)). Open: how much independent information one full continuation branch provides about the response operator and its closure, and which interventions identify the closure at least cost.
- **Reuse across cells.** [Proposition 8.15](08-15_two_fidelity_response_estimation.md#s8-15) reduces replicates within a cell but leaves at least one full continuation per cell (candidate × checkpoint × protocol). Sharing corrections or interfaces across cells needs a coverage condition of the kind matrix completion needs; low rank alone does not give it ([Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12)).
- **Certificates that scale with the effect.** The strict radii of [Propositions 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12) and [8.14](08-14_anchor_reuse_drift_certificate.md#s8-14) and of [Corollary 8.16](08-16_composed_guarantee.md#s8-16) are absolute precisions from bounded-range concentration and are vacuous at the measured scale. Open: variance-adaptive or debiased paired certificates whose radius scales with the size of the effect at an affordable number of continuations; the vacuity found here is a property of the stated estimators, not a proof that such certificates cannot exist.
- **Short versus full continuations on the real learner.** The saving condition of [Proposition 8.15](08-15_two_fidelity_response_estimation.md#s8-15) needs, at a cost ratio of one tenth, a pairing correlation of about 0.787 (fixed coefficient) or 0.575 (optimal coefficient) when every cheap output is charged, and 0.766 or about 0.316 when the cheap output of each pair is reused. This correlation has not been measured on the real model.

**What it says.** Six open problems. For Theorem C: a provable score range much smaller than the a-priori range, for example from a certified pathwise Lipschitz constant, and a testable calibration-transfer condition; until both are settled, Theorem 8.8 is in practice an unbiased, low-variance ranking score, and a rigorous certificate only where the range and the calibration are justified independently. For maintaining the interface during continued training: the information one branch provides, reuse across cells, certificates that scale with the effect, and the real short-versus-full correlation.

**Intuition.** The statistics are no longer the bottleneck; the range, the calibration and the number of full continuations are.

**Role in the framework.** It states precisely what remains to make the measurable certificates both rigorous and useful, and it separates what is proved (validity) from what is measured (vacuity at the current scale: [Check 8.17](../experiments/08-08_crn_paired_replacement_certificate.md#c8-17), [Check 8.18](../experiments/08-12_finite_catalog_spectral_certificate.md#c8-18), [Check 8.21](../experiments/08-14_anchor_reuse_drift_certificate.md#c8-21), [Check 8.23](../experiments/08-16_composed_guarantee.md#c8-23)).

**Uses.**

- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) — the certificate
- [Proposition 8.10](../statements/08-10_cost_and_crn.md#s8-10) — pathwise Lipschitz route to a range
- [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11) — calibration and its transfer
- [Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7) — range and alignment assumptions
- [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12) — closure and the number of continuation protocols
- [Propositions 8.12–8.15](08-12_finite_catalog_spectral_certificate.md#s8-12) — catalog, shared features, drift and two-fidelity estimation
- [Corollary 8.16](08-16_composed_guarantee.md#s8-16) — the composed guarantee

**Used by.**

- [Remark 9.4](../statements/09-04_scope_of_evidence.md#s9-4) — scope of the evidence

## Related work

- Variance-adaptive confidence bounds still pay for the a-priori range through a term of order $`B/r`$ ([Audibert et al., 2009](#ref-audibert2009exploration); [Maurer & Pontil, 2009](#ref-maurer2009empirical)). A smaller range therefore has to come from the training problem, for example a certified Lipschitz constant, not from a sharper estimator.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-audibert2009exploration"></a>

**Jean-Yves Audibert, Rémi Munos, Csaba Szepesvári (2009).** Exploration–exploitation tradeoff using variance estimates in multi-armed bandits. *Theoretical Computer Science* 410, no. 19, 1876–1902. DOI [10.1016/j.tcs.2009.01.016](https://doi.org/10.1016/j.tcs.2009.01.016).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).
