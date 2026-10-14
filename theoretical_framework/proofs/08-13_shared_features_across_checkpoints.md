# Proposition 8.13 · Shared message features across checkpoints

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-13_shared_features_across_checkpoints.md) · [Proof](08-13_shared_features_across_checkpoints.md) · [Experiments](../experiments/08-13_shared_features_across_checkpoints.md) · [中文](08-13_shared_features_across_checkpoints_ZH.md)

**What is proved here.** (a) integrates a per-checkpoint Hilbert–Schmidt bound and identifies the result with the tail of the stacked spectrum. (b) is Ky Fan's maximum principle for the stacked operator. (c) is an explicit construction with bounded scores. (d) is a change of measure.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- One reference measure $`\rho`$ on a common message space, $`\mathcal H=L^2_0(\rho)`$; tests $`a\in\mathcal A`$ with weights $`\omega`$, of the same meaning at every checkpoint; checkpoint weights $`\nu`$.
- $`\mathcal R_z:\mathcal H\to L^2(\omega)`$ is the response operator at checkpoint $`z`$ ([Definition 4.8](../statements/04-08_response_operator.md#s4-8)); $`h_z\in\mathcal H`$ the actual feedback difference at $`z`$.
- $`(\mathcal Sh)(z,a)=(\mathcal R_zh)(a)`$ is the stacked operator into $`L^2(\nu\otimes\omega)`$, with singular values $`\sigma_j(\mathcal S)`$ and $`\Pi_r`$ the projection onto its first $`r`$ right singular vectors. The map $`(z,h)\mapsto\mathcal R_zh`$ is assumed jointly measurable.

**Statement ([Proposition 8.13](../statements/08-13_shared_features_across_checkpoints.md#s8-13)).** (a) If $`\lVert h_z\rVert_\rho\le B`$ $`\nu`$-a.e., then $`\mathbb E_\nu\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega^2\le B^2\sum_{j>r}\sigma_j(\mathcal S)^2`$. (b) $`\Pi_r`$ minimizes $`\mathbb E_\nu\lVert\mathcal R_z(I-\Pi)\rVert_{\rm HS}^2`$ over rank-$`r`$ projections, with minimum $`\sum_{j>r}\sigma_j(\mathcal S)^2`$. (c) There are two equally weighted checkpoints with bounded scores and inputs $`\lVert h_z\rVert_\rho=B`$ for which, at $`r=0`$, the left side of (a) is $`B^2c^2`$, $`B^2\sigma_1(\mathcal S)^2=B^2c^2/2`$ and the tail sum is $`B^2c^2`$. (d) If the visited law $`d`$ satisfies $`dd/d\nu\le\kappa`$, the left side under $`d`$ is at most $`\kappa B^2\sum_{j>r}\sigma_j(\mathcal S)^2`$.

*Proof.*

**Step 1: the stacked Hilbert–Schmidt norm.** Let $`A`$ be a bounded operator on $`\mathcal H`$ and $`(e_k)`$ an orthonormal basis of $`\mathcal H`$ ($`\mathcal H`$ is separable on a standard measurable space). By the definition of the Hilbert–Schmidt norm and Tonelli's theorem,

```math
\lVert\mathcal SA\rVert_{\rm HS}^2=\sum_k\lVert\mathcal SAe_k\rVert_{L^2(\nu\otimes\omega)}^2=\sum_k\mathbb E_\nu\lVert\mathcal R_zAe_k\rVert_\omega^2=\mathbb E_\nu\lVert\mathcal R_zA\rVert_{\rm HS}^2 .
```

With $`A=I`$ this gives $`\lVert\mathcal S\rVert_{\rm HS}^2=\mathbb E_\nu\lVert\mathcal R_z\rVert_{\rm HS}^2\le\tfrac14`$, so $`\mathcal S`$ is Hilbert–Schmidt and has a singular value decomposition $`\mathcal S=\sum_j\sigma_j(\mathcal S)\,u_j\langle v_j,\cdot\rangle_\rho`$.

**(a)** For each $`z`$, $`\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega\le\lVert\mathcal R_z(I-\Pi_r)\rVert_{\rm op}\lVert h_z\rVert_\rho\le\lVert\mathcal R_z(I-\Pi_r)\rVert_{\rm HS}\,B`$. Square and integrate over $`\nu`$, then apply Step 1 with $`A=I-\Pi_r`$:

```math
\mathbb E_\nu\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega^2\le B^2\,\lVert\mathcal S(I-\Pi_r)\rVert_{\rm HS}^2=B^2\sum_{j>r}\sigma_j(\mathcal S)^2 ,
```

the last equality because $`\mathcal S(I-\Pi_r)=\sum_{j>r}\sigma_j(\mathcal S)u_j\langle v_j,\cdot\rangle_\rho`$.

**(b)** For a rank-$`r`$ orthogonal projection $`\Pi`$, Step 1 gives $`\mathbb E_\nu\lVert\mathcal R_z(I-\Pi)\rVert_{\rm HS}^2=\lVert\mathcal S(I-\Pi)\rVert_{\rm HS}^2`$. Since $`\Pi`$ and $`I-\Pi`$ are complementary orthogonal projections,

```math
\lVert\mathcal S(I-\Pi)\rVert_{\rm HS}^2=\lVert\mathcal S\rVert_{\rm HS}^2-\lVert\mathcal S\Pi\rVert_{\rm HS}^2=\lVert\mathcal S\rVert_{\rm HS}^2-\mathrm{tr}\big(\Pi\,\mathcal S^*\mathcal S\,\Pi\big).
```

$`\mathcal S^*\mathcal S`$ is compact, self-adjoint and positive with eigenvalues $`\sigma_j(\mathcal S)^2`$. By Ky Fan's maximum principle the trace over rank-$`r`$ projections is at most $`\sum_{j\le r}\sigma_j(\mathcal S)^2`$, with equality at $`\Pi_r`$. Hence the minimum is $`\lVert\mathcal S\rVert_{\rm HS}^2-\sum_{j\le r}\sigma_j(\mathcal S)^2=\sum_{j>r}\sigma_j(\mathcal S)^2`$. The objective is the $`\nu`$-average; nothing is claimed about the worst checkpoint.

**(c)** Take three messages with $`\rho`$ uniform, so $`\mathcal H`$ is two-dimensional, and an orthonormal basis $`e_1,e_2`$ of $`\mathcal H`$; let $`\lVert e\rVert_\infty=\max_k\max_y\lvert e_k(y)\rvert`$. Two checkpoints $`z_1,z_2`$ have weight $`\tfrac12`$ each and one test with $`\omega=1`$, with scores $`g_{z_k}(y)=\tfrac12+c\,e_k(y)`$; for $`0<c\le1/(2\lVert e\rVert_\infty)`$ they lie in $`[0,1]`$. Then $`\mathcal R_{z_k}h=\langle g_{z_k}-\tfrac12,h\rangle_\rho=c\langle e_k,h\rangle_\rho`$.

- Inputs: $`P_k=\rho(1+\tfrac B2e_k)`$, $`Q_k=\rho(1-\tfrac B2e_k)`$ are probability laws for $`B\le2/\lVert e\rVert_\infty`$ (they are nonnegative, and $`e_k`$ has $`\rho`$-mean zero), and $`h_{z_k}=Be_k`$ with $`\lVert h_{z_k}\rVert_\rho=B`$.
- With $`r=0`$, $`\Pi_0=0`$, the left side of (a) is $`\tfrac12(c^2B^2+c^2B^2)=B^2c^2`$.
- In the coordinates $`(e_1,e_2)`$ and $`L^2(\nu\otimes\omega)\cong\mathbb R^2`$ (with the weights $`\tfrac12`$ absorbed), $`\mathcal S`$ is the matrix $`\tfrac{c}{\sqrt2}I_2`$, so $`\sigma_1(\mathcal S)^2=\sigma_2(\mathcal S)^2=c^2/2`$: the first discarded singular value gives $`B^2c^2/2`$, the tail sum gives $`B^2c^2`$, which is attained. Within the admissible range $`0<c\le1/(2\lVert e\rVert_\infty)`$, $`B\le2/\lVert e\rVert_\infty`$ the ratio of the two bounds is 2 whatever $`c`$ and $`B`$ are.
- With $`K`$ checkpoints and $`K+1`$ messages the same construction gives $`\sigma_1(\mathcal S)^2=c^2/K`$ against an actual value $`c^2`$: the factor between the wrong and the right bound is $`K`$.

**(d)** For any nonnegative measurable $`f`$, $`\mathbb E_df=\mathbb E_\nu[f\,dd/d\nu]\le\kappa\,\mathbb E_\nu f`$. Let $`f(z)=\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega^2`$ and $`g(z)=B^2\lVert\mathcal R_z(I-\Pi_r)\rVert_{\rm HS}^2`$. As in (a), $`f\le g`$ $`\nu`$-a.e., hence $`d`$-a.e. because $`d\ll\nu`$. Therefore $`\mathbb E_df\le\mathbb E_dg\le\kappa\,\mathbb E_\nu g=\kappa B^2\sum_{j>r}\sigma_j(\mathcal S)^2`$ by Step 1. $`\square`$

## Related work

- (b) is the Eckart–Young theorem ([Eckart & Young, 1936](#ref-eckart1936approximation)) for the stacked Hilbert–Schmidt operator; Ky Fan's principle is its trace form.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).
