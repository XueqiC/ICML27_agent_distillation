# Proposition 8.14 · Anchor reuse and a paired drift certificate

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-14_anchor_reuse_drift_certificate.md) · [Proof](08-14_anchor_reuse_drift_certificate.md) · [Experiments](../experiments/08-14_anchor_reuse_drift_certificate.md) · [中文](08-14_anchor_reuse_drift_certificate_ZH.md)

**What is proved here.** (a) is a three-term triangle inequality. (b) identifies the drift with the response operator of a score difference and bounds it by a second moment, then by paired outputs through conditional Jensen. (c) is a one-sided Hoeffding bound with a union bound. (d) is the conditional variance decomposition and the unbiasedness of a product of two conditionally independent differences.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- Reference measure $`\rho`$, $`\mathcal H=L^2_0(\rho)`$; $`p`$ tests with weights $`\omega_a>0`$, $`\sum_a\omega_a=1`$; $`\lVert u\rVert_\omega^2=\sum_a\omega_au_a^2`$ and $`\langle u,u'\rangle_\omega=\sum_a\omega_au_au'_a`$.
- Anchor $`z_0`$ and new checkpoint $`z`$ with response operators $`\mathcal R_0`$, $`\mathcal R_z`$ and response vectors $`g_0(y),g_z(y)\in[0,1]^p`$, so $`(\mathcal R_zh)(a)=\langle g_{z,a}-\mathbb E_\rho g_{z,a},h\rangle_\rho`$.
- $`\hat{\mathcal R}_0`$ with $`\lVert\hat{\mathcal R}_0-\mathcal R_0\rVert_{\rm op}\le\epsilon_0`$, singular values $`\hat\sigma_k(z_0)`$, projection $`\hat\Pi_r`$ onto its first $`r`$ right singular vectors.
- Paired experiment: $`y_i\sim\rho`$ independent; $`X^{(i)}_z,X^{(i)}_0\in[0,1]^p`$ with $`\mathbb E[X^{(i)}_z\mid y_i]=g_z(y_i)`$ and $`\mathbb E[X^{(i)}_0\mid y_i]=g_0(y_i)`$; pairs independent; $`D^{(i)}=X^{(i)}_z-X^{(i)}_0`$, $`W_i=\lVert D^{(i)}\rVert_\omega^2`$; $`M`$ frozen checkpoint pairs.

**Statement ([Proposition 8.14](../statements/08-14_anchor_reuse_drift_certificate.md#s8-14)).** (a) If $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\Delta_z`$, then $`\lVert\mathcal R_z(I-\hat\Pi_r)\rVert_{\rm op}\le\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z`$, hence $`\lVert\mathcal R_zh\rVert_\omega\le B_z(\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z)`$ when $`\lVert h\rVert_\rho\le B_z`$ and $`\hat\Pi_rh=0`$, and one test pays $`1/\sqrt{\omega_a}`$. (b) $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}^2\le\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm HS}^2\le\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2\le\mathbb EW_i`$. (c) With probability at least $`1-\delta`$, for all $`M`$ pairs, $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\sqrt{\min\lbrace1,\bar W_z+\sqrt{\log(M/\delta)/(2n)}\rbrace}`$. (d) $`\mathbb EW_i=\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2+\mathbb E\,\mathrm{tr}_\omega\mathrm{Cov}(D\mid y)`$; with two conditionally independent pair differences per message, $`U_i=\langle D^{(i,1)},D^{(i,2)}\rangle_\omega\in[-1,1]`$ has $`\mathbb E[U_i\mid y_i]=\lVert g_z(y_i)-g_0(y_i)\rVert_\omega^2`$, and with probability at least $`1-\delta`$, for all $`M`$ pairs, $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\sqrt{\min\lbrace1,\max\lbrace0,\bar U_z+\sqrt{2\log(M/\delta)/n}\rbrace\rbrace}`$.

*Proof.*

**(a)** Write $`\mathcal R_z=\hat{\mathcal R}_0+(\mathcal R_0-\hat{\mathcal R}_0)+(\mathcal R_z-\mathcal R_0)`$. Multiply on the right by $`I-\hat\Pi_r`$, whose operator norm is at most 1, and use the triangle inequality:

```math
\lVert\mathcal R_z(I-\hat\Pi_r)\rVert_{\rm op}\le\lVert\hat{\mathcal R}_0(I-\hat\Pi_r)\rVert_{\rm op}+\lVert\mathcal R_0-\hat{\mathcal R}_0\rVert_{\rm op}+\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z .
```

The first term equals $`\hat\sigma_{r+1}(z_0)`$ by the singular value decomposition of $`\hat{\mathcal R}_0`$. No gap and no identification of individual singular vectors is used, so the vectors may rotate or cross between $`z_0`$ and $`z`$. If $`\hat\Pi_rh=0`$, then $`\mathcal R_zh=\mathcal R_z(I-\hat\Pi_r)h`$, and the operator bound times $`\lVert h\rVert_\rho\le B_z`$ gives the response bound. For one test, $`\omega_a\lvert(\mathcal R_zh)(a)\rvert^2\le\lVert\mathcal R_zh\rVert_\omega^2`$.

**(b)** *Step 1.* The response operator is linear in the score functions: $`((\mathcal R_z-\mathcal R_0)h)(a)=\langle\tilde d_a,h\rangle_\rho`$ with $`d_a=g_{z,a}-g_{0,a}`$ and $`\tilde d_a=d_a-\mathbb E_\rho d_a`$. Its Hilbert–Schmidt norm is computed in an orthonormal basis $`(e_k)`$ of $`\mathcal H`$ by Parseval:

```math
\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm HS}^2=\sum_a\omega_a\sum_k\langle\tilde d_a,e_k\rangle_\rho^2=\sum_a\omega_a\lVert\tilde d_a\rVert_\rho^2=\sum_a\omega_a\mathrm{Var}_\rho d_a\le\sum_a\omega_a\mathbb E_\rho d_a^2=\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2 .
```

(The constants are orthogonal to $`\mathcal H`$, and $`\tilde d_a\in\mathcal H`$.) The operator norm is at most the Hilbert–Schmidt norm.

*Step 2.* Since each output has the correct conditional mean, $`\mathbb E[D^{(i)}\mid y_i]=g_z(y_i)-g_0(y_i)`$, whatever the coupling within the pair. The map $`u\mapsto\lVert u\rVert_\omega^2`$ is convex, so conditional Jensen gives $`\lVert g_z(y_i)-g_0(y_i)\rVert_\omega^2\le\mathbb E[\lVert D^{(i)}\rVert_\omega^2\mid y_i]`$. Taking the expectation over $`y_i\sim\rho`$ yields $`\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2\le\mathbb EW_i`$.

**(c)** Each coordinate of $`D^{(i)}`$ lies in $`[-1,1]`$ and $`\sum_a\omega_a=1`$, so $`W_i\in[0,1]`$; for one checkpoint pair the $`W_i`$ are i.i.d. The one-sided Hoeffding inequality ([Hoeffding, 1963](#ref-hoeffding1963probability)) gives $`\Pr\big(\mathbb EW_i>\bar W_z+s\big)\le e^{-2ns^2}`$. With $`s=\sqrt{\log(M/\delta)/(2n)}`$ the right side is $`\delta/M`$, and a union bound over the $`M`$ frozen pairs gives probability at least $`1-\delta`$ that $`\mathbb EW_i\le\bar W_z+s`$ for all of them. Combine with (b) and with $`\mathbb EW_i\le1`$. The pairs are frozen before they are run; checkpoints chosen after looking at the drift data are not covered.

**(d)** *Decomposition.* Conditionally on $`y`$, $`\mathbb E[\lVert D\rVert_\omega^2\mid y]=\lVert\mathbb E[D\mid y]\rVert_\omega^2+\mathbb E[\lVert D-\mathbb E[D\mid y]\rVert_\omega^2\mid y]`$; take expectations. As $`n\to\infty`$, $`\bar W_z\to\mathbb EW_i`$ almost surely by the law of large numbers, and the Hoeffding term vanishes, so $`\hat\Delta_z\to(\mathbb EW_i)^{1/2}`$, which is at least the square root of the noise term.

*Debiased product.* Let $`D^{(i,1)},D^{(i,2)}`$ be the differences of two pairs run for the same message $`y_i`$, conditionally independent given $`y_i`$, each with conditional mean $`g_z(y_i)-g_0(y_i)`$. Then

```math
\mathbb E[U_i\mid y_i]=\sum_a\omega_a\,\mathbb E[D^{(i,1)}_a\mid y_i]\,\mathbb E[D^{(i,2)}_a\mid y_i]=\lVert g_z(y_i)-g_0(y_i)\rVert_\omega^2 .
```

By Cauchy–Schwarz, $`\lvert U_i\rvert\le\lVert D^{(i,1)}\rVert_\omega\lVert D^{(i,2)}\rVert_\omega\le1`$. The one-sided Hoeffding inequality for variables in an interval of length 2 gives $`\Pr(\mathbb EU_i>\bar U_z+s)\le e^{-ns^2/2}`$; with $`s=\sqrt{2\log(M/\delta)/n}`$ and a union bound, $`\mathbb EU_i\le\bar U_z+s`$ for all $`M`$ pairs with probability at least $`1-\delta`$. By (b), $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}^2\le\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2=\mathbb EU_i`$; the square is nonnegative, and $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\lVert\mathcal R_z\rVert_{\rm HS}+\lVert\mathcal R_0\rVert_{\rm HS}\le1`$, which justifies the clipping. At zero drift the bound is $`(2\log(M/\delta)/n)^{1/4}`$. $`\square`$

## Related work

- The certificate uses Hoeffding's inequality ([Hoeffding, 1963](#ref-hoeffding1963probability)); the reuse step is a perturbation argument of Eckart–Young type ([Eckart & Young, 1936](#ref-eckart1936approximation)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).
