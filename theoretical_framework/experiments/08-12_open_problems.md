# Remark 8.12 · Open problems

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-12_open_problems.md) · [Experiments](../experiments/08-12_open_problems.md) · [中文](08-12_open_problems_ZH.md)

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

**Statement under test ([Remark 8.12](../statements/08-12_open_problems.md#s8-12)).**

- **A provable range.** [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) holds with the a-priori range ($`B\le2`$ for $`\tilde F\in[0,1]`$). Its range term is then of order $`7B\log(4M^2/\delta)/(3r)`$, which can far exceed the gaps between nearby candidates for any affordable $`r`$. The observed range of the scores is not a valid substitute. What is missing is a provable range much smaller than the a-priori one, for example the pathwise Lipschitz bound of [Proposition 8.10(c)](../statements/08-10_cost_and_crn.md#s8-10) with a certified $`\bar\lambda`$ for real training tails.
- **Calibration transfer.** A testable condition under which $`(\alpha,\varepsilon_{\rm sur})`$, fitted on calibration pairs, remain valid on new pairs and at later states.
- Until both are settled, Theorem 8.8 serves in practice as an unbiased, low-variance ranking score, and as a rigorous certificate only when $`B`$ and $`(\alpha,\varepsilon_{\rm sur})`$ are justified independently.

### [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate.md#c8-14) · Finite CRN paired replacement on the controlled panel (Theorem 8.8, Proposition 8.10, Remarks 8.11–8.12)

- **What it tests for Remark 8.12.** Whether the a-priori range is the bottleneck of the strict certificate, and whether calibration transfer is a real gap.
- **Setup.** As in the primary check: 1,728 pairs, $`r\in\lbrace64,256\rbrace`$, aligned and calibrated short surrogates, the a-priori range $`B=2`$ (strict) and, as a diagnostic only, the observed range of the scores.
- **Key numbers.** Strict certificate: 0 pairs certified with the aligned surrogate and at most 8 (0.46%) with the calibrated one, none wrong; the range term is 0.156 at $`r=256`$ against a median gap of 0.0037. Observed-range diagnostic: the aligned surrogate certifies 45–57% of pairs with no error, the calibrated surrogate 23–28% with 22–28 wrong certificates.
- **Verdict.** Both open problems are real: a provable range of the size of the observed one would make the certificate decisive, and calibration does not transfer automatically.

### [Check 8.16](../experiments/08-09_first_order_version_margin_criterion.md#c8-16) · The measured first-order certificate and its cap (Corollary 8.9)

- **What it tests for Remark 8.12.** That the failure of first-order certification is a separate cause from the range problem: the first-order certificate fails even before the range enters, because of the second-order remainder.
- **Setup.** The same panel, $`G=8`$ CRN groups, finite-difference steps $`h\in\lbrace10^{-3},3\times10^{-3},10^{-2}\rbrace`$, measured curvature for the finite-difference and remainder terms.
- **Key numbers.** 0 of 1,584 smooth pairs certified; the remainder alone exceeds the measured first-order score on 68.9% of pairs; the empirical Bernstein range term with $`G=8`$ is 2.85 times the observed range.
- **Verdict.** Theorem 8.8 removes the remainder cause entirely; what remains is the range, which is the first open problem.

## Reproduce

Reproduction commands and result files are listed on the pages of the primary checks: [Theorem 8.8](../experiments/08-08_crn_paired_replacement_certificate.md), [Corollary 8.9](../experiments/08-09_first_order_version_margin_criterion.md).
