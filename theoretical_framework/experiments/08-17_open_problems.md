# Remark 8.17 · Open problems

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-17_open_problems.md) · [Experiments](../experiments/08-17_open_problems.md) · [中文](08-17_open_problems_ZH.md)

**What is tested here.** The checks summarized below also test this item. The first two concern the range and calibration problems of Theorem C; the others measure, on the real model and on synthetic instances, how far the strict certificates of the continued-training chain are from being informative, and what the two-fidelity estimator would need. Each summary states what the check tests for this item, its setup and its key numbers; the full description is on the page linked in its heading.

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

**Statement under test ([Remark 8.17](../statements/08-17_open_problems.md#s8-17)).**

- **A provable range.** [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) holds with the a-priori range ($`B\le2`$ for $`\tilde F\in[0,1]`$). Its range term is then of order $`7B\log(4M^2/\delta)/(3r)`$, which can far exceed the gaps between nearby candidates for any affordable $`r`$. The observed range of the scores is not a valid substitute. What is missing is a provable range much smaller than the a-priori one, for example the pathwise Lipschitz bound of [Proposition 8.10(c)](../statements/08-10_cost_and_crn.md#s8-10) with a certified $`\bar\lambda`$ for real training tails.
- **Calibration transfer.** A testable condition under which $`(\alpha,\varepsilon_{\rm sur})`$, fitted on calibration pairs, remain valid on new pairs and at later states.
- Until both are settled, Theorem 8.8 serves in practice as an unbiased, low-variance ranking score, and as a rigorous certificate only when $`B`$ and $`(\alpha,\varepsilon_{\rm sur})`$ are justified independently.
- **Information per continuation branch, and identifying closure.** A closed predictive test space of dimension $`d`$ does not require $`d`$ continuation protocols: one branch can output several metrics, so spanning needs at least $`\lceil d/p\rceil`$ protocols for $`p`$ metrics per branch, and identifying closure needs extension tests ([Proposition 4.12](../statements/04-12_response_predictive_state_factorization.md#s4-12)). Open: how much independent information one full continuation branch provides about the response operator and its closure, and which interventions identify the closure at least cost.
- **Reuse across cells.** [Proposition 8.15](../statements/08-15_two_fidelity_response_estimation.md#s8-15) reduces replicates within a cell but leaves at least one full continuation per cell (candidate × checkpoint × protocol). Sharing corrections or interfaces across cells needs a coverage condition of the kind matrix completion needs; low rank alone does not give it ([Proposition 8.12](../statements/08-12_finite_catalog_spectral_certificate.md#s8-12)).
- **Certificates that scale with the effect.** The strict radii of [Propositions 8.12](../statements/08-12_finite_catalog_spectral_certificate.md#s8-12) and [8.14](../statements/08-14_anchor_reuse_drift_certificate.md#s8-14) and of [Corollary 8.16](../statements/08-16_composed_guarantee.md#s8-16) are absolute precisions from bounded-range concentration and are vacuous at the measured scale. Open: variance-adaptive or debiased paired certificates whose radius scales with the size of the effect at an affordable number of continuations; the vacuity found here is a property of the stated estimators, not a proof that such certificates cannot exist.
- **Short versus full continuations on the real learner.** The saving condition of [Proposition 8.15](../statements/08-15_two_fidelity_response_estimation.md#s8-15) needs, at a cost ratio of one tenth, a pairing correlation of about 0.787 (fixed coefficient) or 0.575 (optimal coefficient) when every cheap output is charged, and 0.766 or about 0.316 when the cheap output of each pair is reused. This correlation has not been measured on the real model.

### [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate.md#c8-14) · Finite CRN paired replacement on the controlled panel (Theorem 8.8, Proposition 8.10, Remarks 8.11 and 8.17)

- **What it tests for Remark 8.17.** Whether the a-priori range is the bottleneck of the strict certificate, and whether calibration transfer is a real gap.
- **Setup.** As in the primary check: 1,728 pairs, $`r\in\lbrace64,256\rbrace`$, aligned and calibrated short surrogates, the a-priori range $`B=2`$ (strict) and, as a diagnostic only, the observed range of the scores.
- **Key numbers.** Strict certificate: 0 pairs certified with the aligned surrogate and at most 8 (0.46%) with the calibrated one, none wrong; the range term of the empirical Bernstein radius ([Maurer & Pontil, 2009](#ref-maurer2009empirical)) is 0.156 at $`r=256`$ against a median gap of 0.0037. Observed-range diagnostic: the aligned surrogate certifies 45–57% of pairs with no error, the calibrated surrogate 23–28% with 22–28 wrong certificates.
- **Verdict.** Both open problems are real: a provable range of the size of the observed one would make the certificate decisive, and calibration does not transfer automatically.

### [Check 8.16](../experiments/08-09_first_order_version_margin_criterion.md#c8-16) · The measured first-order certificate and its cap (Corollary 8.9)

- **What it tests for Remark 8.17.** That the failure of first-order certification is a separate cause from the range problem: the first-order certificate fails even before the range enters, because of the second-order remainder.
- **Setup.** The same panel, $`G=8`$ CRN groups, finite-difference steps $`h\in\lbrace10^{-3},3\times10^{-3},10^{-2}\rbrace`$, measured curvature for the finite-difference and remainder terms.
- **Key numbers.** 0 of 1,584 smooth pairs certified; the remainder alone exceeds the measured first-order score on 68.9% of pairs; the empirical Bernstein range term with $`G=8`$ is 2.85 times the observed range.
- **Verdict.** Theorem 8.8 removes the remainder cause entirely; what remains is the range, which is the first open problem.

### [Check 8.17](../experiments/08-08_crn_paired_replacement_certificate.md#c8-17) · The certificate on the real model: R1, P1 against P3 (Theorem 8.8)

- **What it tests for Remark 8.17.** Whether the range problem persists on the real recipe.
- **Setup.** ALFWorld recipe, audited state $`z_{10}`$, P1 (replay of the $`K=20`$ demonstrations) against P3 (reward-filtered self-generated trajectories), $`r=64`$ replicas, $`T=10`$ shared tail updates, bounded geometric-probability surrogate on 40 held-out teacher trajectories.
- **Key numbers.** $`\hat S=+0.001378`$ (57/64 replicas positive); diagnostic radius 0.001206; strict radius with $`B=2`$: 0.325. Surrogate calibration not estimated.
- **Verdict.** Both open problems of Theorem C are present on the real model: the strict radius is about 240 times the score, and no calibration exists yet.

### [Check 8.18](../experiments/08-12_finite_catalog_spectral_certificate.md#c8-18) · Gap-free truncation residual and the catalog radius (Proposition 8.12)

- **What it tests for Remark 8.17.** The size of a strict catalog certificate at affordable replicate counts, and the cost unit of a catalog.
- **Setup.** Entrywise Hoeffding radius at $`p=10`$ tests, $`m=20`$ candidates, $`\delta=0.05`$; run counts for $`m=20`$, $`n=4`$.
- **Key numbers.** Radius 1.06, 0.53, 0.26, 0.13 at 4, 16, 64, 256 replicates per cell; 80 runs with one continuation protocol, 800 with ten.
- **Verdict.** Certificates that scale with the effect are needed (third new problem); the run count is set by the distinct continuation protocols (first new problem).

### [Check 8.21](../experiments/08-14_anchor_reuse_drift_certificate.md#c8-21) · The drift certificate at the measured evaluation noise (Proposition 8.14(c), (d))

- **What it tests for Remark 8.17.** Whether reuse across checkpoints can be certified cheaply with the stated estimator.
- **Setup.** Noise of the real-model reference (241 and 242 of 300 dev tasks on two identical servers, 15 flips), $`M=10`$ pairs, $`\delta=0.05`$.
- **Key numbers.** Radius 0.718, 0.507, 0.404, 0.227 for 10, 40, 100, 1,000 pairs; about $`4.9\times10^6`$ pairs for 0.03; trivial bound 0.5.
- **Verdict.** Valid but vacuous: a property of this estimator and inequality, which is what the third new problem asks to improve.

### [Check 8.22](../experiments/08-15_two_fidelity_response_estimation.md#c8-22) · Unbiasedness, allocation and the cost thresholds (Proposition 8.15)

- **What it tests for Remark 8.17.** What correlation a short continuation would need to save computation, and the replicate-versus-cell limit.
- **Setup.** Cheap cost one tenth of a full run; fixed coefficient and nested optimal coefficient.
- **Key numbers.** Like-for-like thresholds: 0.787 (fixed coefficient) against 0.575 (optimal coefficient) when every cheap output is charged, 0.766 against about 0.316 when the cheap output of each pair is reused; at most fourfold saving with four full replicates per cell.
- **Verdict.** The fourth new problem is a measurement the framework has not made; the second new problem remains even if it succeeds.

### [Check 8.23](../experiments/08-16_composed_guarantee.md#c8-23) · Per-step floor and signed-sum versus per-term certification (Corollary 8.16)

- **What it tests for Remark 8.17.** Whether the composed guarantee is informative at the measured scale.
- **Setup.** $`B=2`$, one test, $`\varepsilon_n`$ at 10 replicates on 20 candidates, drift radius at 10 pairs; signed-sum Monte Carlo with $`R=20`$.
- **Key numbers.** Per-step floor 2.59; 400 against 8,000 queries for signed-sum estimation against per-term certification at unit noise and error.
- **Verdict.** The composed bound is an error budget at this scale; certificates that scale with the effect are the missing piece.

## Reproduce

Reproduction commands and result files are listed on the pages of the primary checks: [Theorem 8.8](../experiments/08-08_crn_paired_replacement_certificate.md), [Corollary 8.9](../experiments/08-09_first_order_version_margin_criterion.md), [Proposition 8.12](../experiments/08-12_finite_catalog_spectral_certificate.md), [Proposition 8.14](../experiments/08-14_anchor_reuse_drift_certificate.md), [Proposition 8.15](../experiments/08-15_two_fidelity_response_estimation.md), [Corollary 8.16](../experiments/08-16_composed_guarantee.md). The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
