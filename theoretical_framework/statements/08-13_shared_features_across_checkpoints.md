# Proposition 8.13 · Shared message features across checkpoints

*Statement*

[← Framework](../README.md) · [Statement](08-13_shared_features_across_checkpoints.md) · [Proof](../proofs/08-13_shared_features_across_checkpoints.md) · [Experiments](../experiments/08-13_shared_features_across_checkpoints.md) · [中文](08-13_shared_features_across_checkpoints_ZH.md)

**In one paragraph.** A training run passes through many checkpoints, and estimating a response operator at each one is unaffordable. Stacking the response rows of several checkpoints and learning one set of message features from them is legitimate, provided one keeps track of what is shared. The features are shared; the feedback difference is not, because the reference and the alternative feedback depend on the state. With state-conditional inputs the correct error bound is the tail *sum* of squared singular values of the stacked operator, not the square of the first discarded singular value, and the truncated singular basis is optimal only for the state-averaged objective.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- A common message space with one reference measure $`\rho`$ for all checkpoints considered, and $`\mathcal H=L^2_0(\rho)`$ (centred square-integrable functions).
- A declared test set $`\mathcal A`$ with weights $`\omega`$, with the same meaning at every checkpoint. Different remaining budgets are part of the test label: two rows with different meanings are never treated as one test.
- Checkpoints $`z`$ carry a probability weight $`\nu`$. At each, $`\mathcal R_z:\mathcal H\to L^2(\omega)`$ is the response operator of [Definition 4.8](04-08_response_operator.md#s4-8), so $`\lVert\mathcal R_z\rVert_{\rm HS}^2\le\tfrac14`$.
- The actual feedback difference at $`z`$ is $`h_z=dP_z/d\rho-dQ_z/d\rho\in\mathcal H`$; it depends on $`z`$. If the legal messages depend on the state, a common legal pool or a declared rejection rule is used; a change of support voids the coverage of $`\rho`$.
- The *stacked operator* $`\mathcal S:\mathcal H\to L^2(\nu\otimes\omega)`$ is

```math
(\mathcal Sh)(z,a)=(\mathcal R_zh)(a).
```

It is Hilbert–Schmidt with $`\lVert\mathcal S\rVert_{\rm HS}^2=\mathbb E_{z\sim\nu}\lVert\mathcal R_z\rVert_{\rm HS}^2\le\tfrac14`$; its singular values are $`\sigma_1(\mathcal S)\ge\sigma_2(\mathcal S)\ge\dots`$, and $`\Pi_r`$ projects onto its first $`r`$ right singular vectors. A *shared $`r`$-dimensional linear message interface* is an orthogonal projection $`\Pi`$ of rank $`r`$ on $`\mathcal H`$, i.e. $`r`$ shared message features.

<a id="s8-13"></a>

**Proposition 8.13 (Shared message features across checkpoints).** In this setting:

**(a) Conditional-input bound.** If $`\lVert h_z\rVert_\rho\le B`$ for $`\nu`$-almost every $`z`$, then

```math
\mathbb E_{z\sim\nu}\,\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega^2\le B^2\sum_{j>r}\sigma_j(\mathcal S)^2 .
```

**(b) Optimality for the averaged objective.** Among all shared interfaces of rank $`r`$, $`\Pi_r`$ minimizes $`\mathbb E_{z\sim\nu}\lVert\mathcal R_z(I-\Pi)\rVert_{\rm HS}^2`$, and the minimum is $`\sum_{j>r}\sigma_j(\mathcal S)^2`$. This is optimality for the $`\nu`$-average, not a minimax statement over checkpoint distributions.

**(c) The tail sum cannot be replaced by the first discarded singular value.** There are two equally weighted checkpoints, bounded scores and inputs with $`\lVert h_z\rVert_\rho=B`$ such that, with $`r=0`$ (no feature retained), the left side of (a) equals $`B^2c^2`$ while $`B^2\sigma_1(\mathcal S)^2=B^2c^2/2`$; the tail sum equals $`B^2c^2`$. Bounded scores and probability-valued inputs require $`0<c\le1/(2\lVert e\rVert_\infty)`$ and $`B\le2/\lVert e\rVert_\infty`$, where $`e_1,e_2`$ is the orthonormal basis used in the construction; the factor 2 between the two bounds does not depend on $`c`$ or $`B`$.

**(d) From the weighting to the visited checkpoints.** If the checkpoints actually visited have law $`d`$ with $`d\ll\nu`$ and $`dd/d\nu\le\kappa`$, the left side of (a) under $`d`$ is at most $`\kappa B^2\sum_{j>r}\sigma_j(\mathcal S)^2`$. Without such a bound, the $`\nu`$-average says nothing about the visited checkpoints; use the per-checkpoint guarantee of [Proposition 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14), or pay the exit term of [Proposition 3.4(d)](03-04_joint_decomposition_approximate_value.md#s3-4).

**What it says.**

- Learning one set of message features from many checkpoints is allowed; the feedback difference fed to them may change with the state.
- The price of state-conditional inputs is the full tail of the stacked spectrum. Bounding it by $`\sigma_{r+1}(\mathcal S)`$ alone is wrong, by a factor that can be as large as the number of checkpoints.
- Averaged accuracy does not transfer to the states training actually visits unless their law is dominated by the weighting.

**Intuition.** A common input sees the stacked operator only through one direction at a time, so its largest singular value is the worst case. A state-conditional input can align itself with each checkpoint's own dominant direction; the errors then add up over checkpoints, and only a sum of squared singular values controls them.

**Role in the framework.**

- It fixes the correct norm for reusing a response interface across checkpoints, which is the first step of maintaining sufficiency during continued training.
- [Proposition 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14) gives the per-checkpoint complement, and [Corollary 8.16](08-16_composed_guarantee.md#s8-16) uses the per-checkpoint form.

**Comments on the statement.**

- **What is shared.** Message features $`\phi(y)`$, not feedback laws and not $`h_z`$. Equal feature means at one checkpoint do not imply equal feature means at another.
- **Counterexample in words.** Two checkpoints respond to orthogonal directions, and each checkpoint's feedback difference points along its own direction. Every input has unit norm and every checkpoint has a unit response, but the stacked operator's largest singular value is only $`1/\sqrt2`$ of that ([Check 8.19](../experiments/08-13_shared_features_across_checkpoints.md#c8-19)).

**Derivation outline.**

- (a): per checkpoint, $`\lVert Ah\rVert\le\lVert A\rVert_{\rm HS}\lVert h\rVert`$; integrate over $`\nu`$; the averaged Hilbert–Schmidt norm of $`\mathcal R_z(I-\Pi)`$ is the Hilbert–Schmidt norm of $`\mathcal S(I-\Pi)`$, whose value at $`\Pi_r`$ is the tail sum.
- (b): $`\lVert\mathcal S(I-\Pi)\rVert_{\rm HS}^2=\lVert\mathcal S\rVert_{\rm HS}^2-\mathrm{tr}(\Pi\mathcal S^*\mathcal S\Pi)`$, and Ky Fan's maximum principle maximizes the trace over rank-$`r`$ projections.
- (c): two orthogonal one-dimensional responses realized by bounded scores on three messages.
- (d): change of measure.

**Uses.**

- [Definition 4.8](04-08_response_operator.md#s4-8) — the response operator at one checkpoint
- [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9) — the single-checkpoint truncation

**Used by.**

- [Proposition 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14) — per-checkpoint reuse
- [Corollary 8.16](08-16_composed_guarantee.md#s8-16) — composed guarantee
- [Remark 8.17](08-17_open_problems.md#s8-17) — reuse across cells

## Related work

- (b) is the Eckart–Young theorem ([Eckart & Young, 1936](#ref-eckart1936approximation)) applied to the stacked operator. Datamodels fit one linear model of outputs on training-subset indicators across many test examples ([Ilyas et al., 2022](#ref-ilyas2022datamodels)); here the rows come from different learner states, and the point is that their inputs differ by state.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").
