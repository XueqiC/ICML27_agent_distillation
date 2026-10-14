# Proposition 8.14 · Anchor reuse and a paired drift certificate

*Statement*

[← Framework](../README.md) · [Statement](08-14_anchor_reuse_drift_certificate.md) · [Proof](../proofs/08-14_anchor_reuse_drift_certificate.md) · [Experiments](../experiments/08-14_anchor_reuse_drift_certificate.md) · [中文](08-14_anchor_reuse_drift_certificate_ZH.md)

**In one paragraph.** Suppose a response interface was estimated at one anchor checkpoint. Can it be reused at a later checkpoint without estimating the whole operator again? This proposition gives the per-checkpoint answer. The residual of the anchor interface at a new checkpoint is at most the anchor's first discarded singular value, plus the anchor's estimation error, plus the operator drift between the two checkpoints; no spectral gap is needed. The drift can be bounded by a paired experiment that only needs random messages, not the whole catalog: the mean squared difference of paired bounded outputs bounds the squared operator drift. A Hoeffding radius makes this a strict certificate. At the evaluation noise measured on the real model it is valid but vacuous: its radius is above the trivial bound up to tens of pairs, and reaching 0.03 would take about five million pairs.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- A common message space with reference measure $`\rho`$ and $`\mathcal H=L^2_0(\rho)`$; a finite declared test set $`\mathcal A`$ of $`p`$ tests with weights $`\omega_a>0`$, the same at both checkpoints; $`\lVert u\rVert_\omega^2=\sum_a\omega_au_a^2`$.
- Two checkpoints: an *anchor* $`z_0`$ and a *new checkpoint* $`z`$, with response operators $`\mathcal R_0`$ and $`\mathcal R_z`$ ([Definition 4.8](04-08_response_operator.md#s4-8)) and response vectors $`g_0(y)=(g_{0,a}(y))_a`$, $`g_z(y)=(g_{z,a}(y))_a\in[0,1]^p`$.
- *Anchor estimate.* $`\hat{\mathcal R}_0`$ with $`\lVert\hat{\mathcal R}_0-\mathcal R_0\rVert_{\rm op}\le\epsilon_0`$ (for example from [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12)), singular values $`\hat\sigma_k(z_0)`$, and $`\hat\Pi_r`$ the projection onto its first $`r`$ right singular vectors.
- *Paired drift experiment.* Draw $`y_1,\dots,y_n`$ independently from $`\rho`$. For each $`y_i`$, run the declared tests once from $`z`$ and once from $`z_0`$ after the message $`y_i`$, each under its correct marginal protocol, and record bounded output vectors $`X^{(i)}_z,X^{(i)}_0\in[0,1]^p`$. The two runs of one pair may share random numbers; different pairs are independent. Then $`\mathbb E[X^{(i)}_z\mid y_i]=g_z(y_i)`$ and $`\mathbb E[X^{(i)}_0\mid y_i]=g_0(y_i)`$. Write $`D^{(i)}=X^{(i)}_z-X^{(i)}_0`$ and $`W_i=\lVert D^{(i)}\rVert_\omega^2\in[0,1]`$.
- $`M`$ checkpoint pairs are frozen before any pair is run; $`\bar W_z`$ is the average of the $`W_i`$ for pair $`z`$.

<a id="s8-14"></a>

**Proposition 8.14 (Anchor reuse and a paired drift certificate).** In this setting:

**(a) Anchor reuse without a gap.** If $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\Delta_z`$, then

```math
\lVert\mathcal R_z(I-\hat\Pi_r)\rVert_{\rm op}\le\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z .
```

Consequently, for a feedback difference $`h`$ at $`z`$ with $`\lVert h\rVert_\rho\le B_z`$ and $`\hat\Pi_rh=0`$ (the retained feature means match), $`\lVert\mathcal R_zh\rVert_\omega\le B_z\big(\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z\big)`$, and for one test $`a`$ the score difference is at most this divided by $`\sqrt{\omega_a}`$.

**(b) Paired second moment bounds the drift.**

```math
\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}^2\le\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm HS}^2\le\mathbb E_{y\sim\rho}\lVert g_z(y)-g_0(y)\rVert_\omega^2\le\mathbb E\,W_i .
```

**(c) Strict drift certificate.** With probability at least $`1-\delta`$, simultaneously for the $`M`$ frozen pairs,

```math
\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\hat\Delta_z=\sqrt{\min\Big\lbrace1,\ \bar W_z+\sqrt{\frac{\log(M/\delta)}{2n}}\Big\rbrace},
```

so $`\hat\Delta_z`$ can be used as $`\Delta_z`$ in (a). Each pair costs two continuation branches per distinct continuation protocol.

**(d) The estimator floor and its debiased version.** $`\mathbb EW_i=\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2+\mathbb E\,\mathrm{tr}_\omega\mathrm{Cov}(D\mid y)`$, where $`\mathrm{tr}_\omega\mathrm{Cov}(D\mid y)=\mathbb E[\lVert D-\mathbb E[D\mid y]\rVert_\omega^2\mid y]`$. As $`n`$ grows, $`\hat\Delta_z`$ converges to $`(\mathbb EW_i)^{1/2}`$, which is never below the process-noise level $`(\mathbb E\,\mathrm{tr}_\omega\mathrm{Cov}(D\mid y))^{1/2}`$. If each message is run twice, giving conditionally independent pair differences $`D^{(i,1)},D^{(i,2)}`$, then $`U_i=\langle D^{(i,1)},D^{(i,2)}\rangle_\omega\in[-1,1]`$ satisfies $`\mathbb E[U_i\mid y_i]=\lVert g_z(y_i)-g_0(y_i)\rVert_\omega^2`$, and with probability at least $`1-\delta`$, for all $`M`$ pairs,

```math
\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\tilde\Delta_z=\sqrt{\min\Big\lbrace1,\ \max\Big\lbrace0,\ \bar U_z+\sqrt{\frac{2\log(M/\delta)}{n}}\Big\rbrace\Big\rbrace}.
```

This removes the floor at twice the cost, but at zero drift its radius still decays only like $`n^{-1/4}`$.

**What it says.**

- An interface estimated at one checkpoint can be reused at another, paying the measured operator drift; the singular vectors may rotate or cross without harm, because only operator norms enter.
- The drift is measurable by a paired experiment on random messages, with a strict, finite-sample certificate.
- That certificate is valid but, at the measured evaluation noise, vacuous: with the noise of the real-model reference, its radius is 0.718 with 10 pairs, 0.507 with 40, 0.404 with 100 and 0.227 with 1,000 ($`M=10`$, $`\delta=0.05`$), against a trivial bound of $`\tfrac12`$ on the residual itself; a radius of 0.03 needs about $`4.9\times10^6`$ pairs ([Check 8.21](../experiments/08-14_anchor_reuse_drift_certificate.md#c8-21)).

**Intuition.** Reuse is a triangle inequality: whatever the new operator does outside the anchor's retained directions is at most what the anchor does there, plus how wrong the anchor estimate is, plus how far the operator moved. The drift itself is the response operator of a *difference* of scores, so a second moment of paired output differences controls it; but that second moment also contains run-to-run noise, and a bounded-range concentration bound pays for the full range.

**Role in the framework.**

- It is the per-checkpoint reuse step of maintaining a response interface during continued training, complementing the averaged bound of [Proposition 8.13](08-13_shared_features_across_checkpoints.md#s8-13).
- It supplies $`\Delta_{t,z}`$ to [Corollary 8.16](08-16_composed_guarantee.md#s8-16).
- Its vacuity at the measured scale is one of the open problems of [Remark 8.17](08-17_open_problems.md#s8-17): a drift certificate whose radius scales with the effect at an affordable number of pairs.

**Comments on the statement.**

- **What the noise number is.** The real-model reference ran one checkpoint on 300 dev tasks on two identical servers: 241 and 242 successes, with 15 tasks flipping (7 one way, 8 the other). If the flips were symmetric, a single score difference would have standard deviation about $`\sqrt{15}/300\approx0.013`$, giving $`\bar W\approx1.7\times10^{-4}`$ on one test. This covers serving noise only; training-seed noise between two continuations would add to it. One server pair at one checkpoint does not establish irreducible, independent and identically distributed evaluation noise.
- **Why it is vacuous.** The Hoeffding term $`\sqrt{\log(M/\delta)/(2n)}`$ does not shrink with $`\bar W`$. A variance-adaptive bound removes part of this, but the range term remains, and the floor of (d) remains unless the debiased product is used.
- **Several anchors.** One may choose for each checkpoint the anchor with the smallest certified bound; if the choice uses the same audit data, all candidate anchors' events must hold simultaneously.
- **Per-test use.** A guarantee for one test $`a`$ pays $`1/\sqrt{\omega_a}`$, or that row can be audited directly; an RMS bound is not a per-test bound.

**Derivation outline.**

- (a): $`\mathcal R_z=\hat{\mathcal R}_0+(\mathcal R_0-\hat{\mathcal R}_0)+(\mathcal R_z-\mathcal R_0)`$ and $`\lVert I-\hat\Pi_r\rVert_{\rm op}\le1`$.
- (b): $`\mathcal R_z-\mathcal R_0`$ is the response operator of the score difference $`g_z-g_0`$; its Hilbert–Schmidt norm is a weighted variance, below the second moment; conditional Jensen moves from the conditional mean to the paired outputs.
- (c): one-sided Hoeffding inequality for $`W_i\in[0,1]`$ and a union bound over the $`M`$ pairs.
- (d): variance decomposition; conditional independence of the two pair differences; Hoeffding on $`[-1,1]`$.

**Uses.**

- [Definition 4.8](04-08_response_operator.md#s4-8) — the response operator
- [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12) — the anchor estimate and its error

**Used by.**

- [Corollary 8.16](08-16_composed_guarantee.md#s8-16) — drift term of the composed guarantee
- [Remark 8.17](08-17_open_problems.md#s8-17) — certificates whose radius scales with the effect

## Related work

- The drift bound is a perturbation argument in the spirit of the Eckart–Young theorem ([Eckart & Young, 1936](#ref-eckart1936approximation)); the certificate is Hoeffding's inequality ([Hoeffding, 1963](#ref-hoeffding1963probability)). A variance-adaptive alternative is the empirical Bernstein bound ([Maurer & Pontil, 2009](#ref-maurer2009empirical)), which still pays a range term of order $`1/n`$.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).
