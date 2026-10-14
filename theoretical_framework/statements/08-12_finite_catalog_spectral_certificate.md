# Proposition 8.12 · Finite-catalog spectral certificate

*Statement*

[← Framework](../README.md) · [Statement](08-12_finite_catalog_spectral_certificate.md) · [Proof](../proofs/08-12_finite_catalog_spectral_certificate.md) · [Experiments](../experiments/08-12_finite_catalog_spectral_certificate.md) · [中文](08-12_finite_catalog_spectral_certificate_ZH.md)

**In one paragraph.** The response operator of [Definition 4.8](04-08_response_operator.md#s4-8) is a population object. In practice one freezes a finite catalog of tests and candidate feedback laws at one checkpoint, runs every cell a few times, and works with the estimated matrix. This proposition gives what such a catalog certifies: an entrywise Hoeffding radius turns into an operator-norm radius on the weighted, centred matrix, every singular value moves by at most that radius, and the residual of a truncated interface is bounded without any spectral gap. It also fixes the cost unit: the number of continuation runs is the number of candidates times the number of replicates times the number of *distinct continuation protocols*, not times the number of tests. At affordable replicate counts the radius is far above the response scale, so the certificate is valid but, at the measured scale, vacuous.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- One checkpoint $`(t,z)`$ is fixed: a complete learner state, optimizer state included.
- *Tests.* $`p`$ declared tests $`i=1,\dots,p`$ with weights $`w_i>0`$, $`\sum_iw_i=1`$. Each test is a declared task score together with a declared continuation protocol. Tests that share one continuation protocol are read off the same run; $`p_{\rm train}\le p`$ denotes the number of distinct continuation protocols among the tests.
- *Candidates.* $`m`$ frozen candidate feedback laws $`j=1,\dots,m`$ with weights $`q_j>0`$, $`\sum_jq_j=1`$. A candidate may be an executable feedback provider; the catalog then compresses mixtures of providers, not their internal messages.
- *Cell means.* $`G_{ij}\in[0,1]`$ is the expected final score of test $`i`$ after one round with candidate $`j`$ at $`z`$ followed by the test's continuation. This is the response of [Definition 4.8](04-08_response_operator.md#s4-8) with the candidates as messages and $`\rho=q`$.
- *Weighted, centred matrix.* With $`D_a=\mathrm{diag}(a)`$, $`\sqrt q`$ taken entrywise and $`\Pi_q=I-\sqrt q\sqrt q^{\top}`$,

```math
M=D_{\sqrt w}\,G\,D_{\sqrt q}\,\Pi_q .
```

- For a law $`P`$ over the candidates (a mixture with weights $`P_j`$) and another law $`Q`$, the density difference is $`h_j=(P_j-Q_j)/q_j`$ and $`x=D_{\sqrt q}h`$. Then $`\sqrt q^{\top}x=0`$, $`\lVert x\rVert_2=\lVert h\rVert_q`$, and $`(Mx)_i=\sqrt{w_i}\,a_i`$ with $`a_i=\sum_j(P_j-Q_j)G_{ij}`$ the score difference on test $`i`$. So $`\lVert Mx\rVert_2=\lVert a\rVert_\omega`$, where $`\lVert a\rVert_\omega^2=\sum_iw_ia_i^2`$: the matrix $`M`$ is the response operator in orthonormal coordinates. Its singular values are $`\sigma_1\ge\sigma_2\ge\dots`$, and $`\lVert M\rVert_{\rm HS}^2\le\tfrac14`$.
- *Replicates.* Each cell is run $`n`$ times; $`\hat G_{ij}`$ is the cell average. Replicates are independent across replicate index; inside one replicate the candidates may share random numbers, and tests read off the same run are allowed to be dependent. $`\hat M`$ is built from $`\hat G`$ as $`M`$ from $`G`$; $`\hat\sigma_j`$, $`\hat v_j`$ are its singular values and right singular vectors, and $`\hat\Pi_r`$ projects onto $`\hat v_1,\dots,\hat v_r`$.
- *Estimated row coefficients.* For test $`i`$, let $`\hat g_i`$ be row $`i`$ of $`\hat G`$ as a function on the candidates, centred under $`q`$; $`\hat\phi_k=D_{\sqrt q}^{-1}\hat v_k`$ is the $`k`$-th estimated mode as a function on the candidates; $`\hat b_k(i)=\langle\hat g_i,\hat\phi_k\rangle_q`$ and $`\hat e_r(i)=\lVert\hat g_i-\sum_{k\le r}\hat b_k(i)\hat\phi_k\rVert_q`$. The mode-mean differences of $`P`$ and $`Q`$ are $`\alpha_k=\sum_j(P_j-Q_j)\hat\phi_k(j)`$.

<a id="s8-12"></a>

**Proposition 8.12 (Finite-catalog spectral certificate).** In this setting, let

```math
\varepsilon_n=\sqrt{\frac{\log(2pm/\delta)}{2n}} .
```

**(a) Entrywise radius.** With probability at least $`1-\delta`$, simultaneously for every cell, $`\lvert\hat G_{ij}-G_{ij}\rvert\le\varepsilon_n`$.

**(b) Operator radius and spectrum.** On the event of (a),

```math
\lVert\hat M-M\rVert_{\rm op}\le\varepsilon_n,\qquad\lvert\hat\sigma_k-\sigma_k\rvert\le\varepsilon_n\quad\text{for every }k .
```

**(c) Truncated interface without a gap.** On the same event, for every $`r`$,

```math
\lVert M(I-\hat\Pi_r)\rVert_{\rm op}\le\hat\sigma_{r+1}+\varepsilon_n\le\sigma_{r+1}+2\varepsilon_n .
```

**(d) Checkable per-test bound.** On the same event, for every pair of laws $`P,Q`$ over the candidates with $`\lVert h\rVert_q\le B`$ and every test $`i`$,

```math
\Big\lvert\sum_j(P_j-Q_j)G_{ij}\Big\rvert\le\Big\lvert\sum_{k\le r}\hat b_k(i)\,\alpha_k\Big\rvert+B\,\hat e_r(i)+2\varepsilon_n .
```

If only estimates $`\hat\alpha_k`$ with simultaneously valid errors $`\lvert\hat\alpha_k-\alpha_k\rvert\le\eta_k`$ are available, replace $`\alpha_k`$ by $`\hat\alpha_k`$ and add $`\sum_{k\le r}\lvert\hat b_k(i)\rvert\eta_k`$.

**(e) Truncation with frozen modes.** Suppose the exact singular triples $`(\sigma_k,u_k,v_k)`$ of $`M`$ are known. Write $`\phi_k=D_{\sqrt q}^{-1}v_k`$ and $`\alpha_k=\mathbb E_P\phi_k-\mathbb E_Q\phi_k`$, and let $`\hat\alpha_k`$ be the difference of the sample means of $`\phi_k`$ over $`n'`$ independent draws (candidate indices) from each of $`P`$ and $`Q`$. Let $`f=Mx=\sum_k\sigma_k\alpha_ku_k`$ be the true weighted response difference and $`\hat f_r=\sum_{k\le r}\sigma_k\hat\alpha_ku_k`$. Then

```math
\mathbb E\,\lVert\hat f_r-f\rVert_2^2=\sum_{k\le r}\frac{\sigma_k^2}{n'}\big(\mathrm{Var}_P\phi_k+\mathrm{Var}_Q\phi_k\big)+\sum_{k>r}\sigma_k^2\alpha_k^2 .
```

If instead the triples $`(\hat\sigma_k,\hat u_k,\hat v_k)`$ are those of an estimate $`\hat M`$ frozen from data independent of the draws, and $`\hat f_r=\sum_{k\le r}\hat\sigma_k\hat\alpha_k\hat u_k`$ with $`\hat\alpha_k`$ the sample-mean difference of $`\hat\phi_k=D_{\sqrt q}^{-1}\hat v_k`$, the variance term is the same expression with hats, and the bias term becomes $`\lVert(M-\hat M\hat\Pi_r)x\rVert_2^2`$ instead of $`\sum_{k>r}\sigma_k^2\alpha_k^2`$.

**(f) Cost unit.** A complete catalog of $`m`$ candidates with $`n`$ replicates costs $`m\,n\,p_{\rm train}`$ continuation runs plus the evaluations of the $`p`$ tests. It is $`mnp`$ only when every test needs its own continuation.

**What it says.**

- Freezing a finite catalog makes the response operator estimable with a uniform, gap-free guarantee: entrywise errors become an operator-norm error of the same size, and the truncated interface keeps its residual bound.
- The bound is checkable test by test, with an explicit price $`2\varepsilon_n`$ for working with estimated cells.
- The number of continuation runs is set by the distinct continuation protocols, not by the number of metrics read off each run.

**Intuition.** The weights $`w`$ and $`q`$ sum to one, so the weighted Frobenius norm of the error matrix is at most its largest entry; operator norm is below Frobenius norm. A spectral gap matters only if one wants to interpret one estimated singular vector as a stable "mechanism", not for the residual bound.

**Role in the framework.**

- It is the finite-sample version of [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9) at one checkpoint, and the anchor estimate that [Proposition 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14) reuses across checkpoints.
- Its radius $`\varepsilon_n`$ is what [Proposition 8.15](08-15_two_fidelity_response_estimation.md#s8-15) tries to reach more cheaply, and the anchor error $`\epsilon_j`$ of [Corollary 8.16](08-16_composed_guarantee.md#s8-16).
- At $`p=10`$ tests, $`m=20`$ candidates and $`\delta=0.05`$ the radius is 1.06, 0.53, 0.26 and 0.13 for 4, 16, 64 and 256 replicates per cell ([Check 8.18](../experiments/08-12_finite_catalog_spectral_certificate.md#c8-18)). At 256 replicates, $`2\varepsilon_n=0.265`$ is below the a-priori bound $`\tfrac12`$, so the bound is not vacuous against that bound; it is vacuous against the measured effect scale, about $`10^{-3}`$ for the surrogate gap of the real-model pair in [Check 8.17](../experiments/08-08_crn_paired_replacement_certificate.md#c8-17). The strict certificate is valid but, at that scale, uninformative.

**Comments on the statement.**

- **What is not covered.** Unobserved candidates, tests and checkpoints are not protected by (a). A low-rank matrix need not be recoverable from a few cells: an important direction may appear in one rarely sampled cell. Any completion, generalization or off-policy extrapolation needs its own coverage condition.
- **Adaptive catalogs.** The union bound in (a) is over a catalog frozen before the replicates are drawn. Candidates generated from the audit results need fresh confirmation samples or a bound that is valid under adaptive selection.
- **Selection in (e).** If $`r`$ or the modes are chosen from the same data used for $`\hat\alpha`$, the formula does not apply as stated; use sample splitting or cross-fitting.

**Derivation outline.**

- (a) is Hoeffding's inequality ([Hoeffding, 1963](#ref-hoeffding1963probability)) per cell with a union bound over the $`pm`$ cells.
- (b) bounds the operator norm by the weighted Frobenius norm, which is at most $`\varepsilon_n`$ because the weights sum to one; $`\Pi_q`$ is a contraction; Weyl's inequality moves each singular value by at most the operator norm.
- (c) splits $`M=\hat M+(M-\hat M)`$ and uses $`\lVert\hat M(I-\hat\Pi_r)\rVert_{\rm op}=\hat\sigma_{r+1}`$.
- (d) expands the estimated centred row in the estimated modes, applies Cauchy–Schwarz to the remainder, and pays $`2\varepsilon_n`$ for replacing $`G`$ by $`\hat G`$, since $`\sum_j\lvert P_j-Q_j\rvert\le2`$.
- (e) is orthogonality of the output vectors $`u_k`$ and unbiasedness of the mean estimates; no independence between modes is needed.

**Uses.**

- [Definition 4.8](04-08_response_operator.md#s4-8) — the response operator
- [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9) — spectral sufficiency and the truncated interface

**Used by.**

- [Proposition 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14) — anchor estimate reused at other checkpoints
- [Proposition 8.15](08-15_two_fidelity_response_estimation.md#s8-15) — cheaper entrywise radii
- [Corollary 8.16](08-16_composed_guarantee.md#s8-16) — anchor error in the composed guarantee
- [Remark 8.17](08-17_open_problems.md#s8-17) — reuse across cells

## Related work

- (b)–(c) are the matrix-perturbation consequences of Eckart–Young optimality ([Eckart & Young, 1936](#ref-eckart1936approximation)) combined with an entrywise Hoeffding bound ([Hoeffding, 1963](#ref-hoeffding1963probability)). The counterfactual response matrix itself is close to datamodels ([Ilyas et al., 2022](#ref-ilyas2022datamodels)), which regress model outputs on training-subset indicators; here each column is a frozen feedback law at one complete learner state, and each entry is a full continuation.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").
