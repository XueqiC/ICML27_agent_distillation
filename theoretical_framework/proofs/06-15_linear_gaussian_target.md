# Proposition 6.15 · Linear-Gaussian target: exact risk, whitened response, one-query gain

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-15_linear_gaussian_target.md) · [Proof](06-15_linear_gaussian_target.md) · [Experiments](../experiments/06-15_linear_gaussian_target.md) · [中文](06-15_linear_gaussian_target_ZH.md)

**What is proved here.** The risk of generalized least squares is computed exactly; a family of Gaussian priors shows that no estimator does better in the worst case; the eigen-identity and the one-query formula are matrix identities.

**Setting used on this page.**

- $`y=D\vartheta+\epsilon`$, $`\epsilon\sim\mathcal N(0,\Sigma)`$, $`\Sigma\succ0`$, $`D`$ known, $`\vartheta\in\mathbb R^d`$ unknown and unrestricted.
- Target $`f(\vartheta)=C\vartheta\in\mathbb R^k`$, loss $`\lVert x\rVert_W^2=x^\top Wx`$ with $`W\succeq0`$.
- $`I=D^\top\Sigma^{-1}D`$, $`B=C^\top WC`$, $`I^+`$ the Moore–Penrose pseudo-inverse.
- A query adds the independent observation $`a^\top\vartheta+\varepsilon`$, $`\varepsilon\sim\mathcal N(0,1)`$, and hence $`aa^\top`$ to $`I`$.

**Statement ([Proposition 6.15](../statements/06-15_linear_gaussian_target.md#s6-15)).** (a) If $`I\succ0`$, $`\hat f=CI^{-1}D^\top\Sigma^{-1}y`$ has constant risk $`\mathrm{tr}(BI^{-1})=\sum_j\lambda_j`$ with $`Bv_j=\lambda_jIv_j`$, the minimax risk over $`\mathbb R^d`$. (b) If $`W^{1/2}C`$ vanishes on $`\ker I`$, $`\hat f=CI^+D^\top\Sigma^{-1}y`$ has constant risk $`\mathrm{tr}(BI^+)`$, the minimax risk, and is unbiased when $`C\ker I=\lbrace0\rbrace`$; if $`W^{1/2}Cv\ne0`$ for some $`v\in\ker I`$, the minimax risk is $`+\infty`$, and on a bounded set containing two points that differ along such a $`v`$ it has a positive floor. (c) For $`I\succ0`$, the squared singular values of $`\mathcal R_I=W^{1/2}CI^{-1/2}`$ are the $`\lambda_j`$. (d) For $`I\succ0`$, one query lowers the minimax risk by $`\Delta(a)=a^\top I^{-1}BI^{-1}a/(1+a^\top I^{-1}a)`$. (e) For $`C=c^\top`$, $`W=1`$, the risk is $`c^\top I^{-1}c`$.

**Results used in the proof.**

- [Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4), only for the comparison in (e).

**Classical tools.** Gaussian conjugacy; Bayes risks bound the minimax risk from below; the Sherman–Morrison formula; Le Cam's two-point argument ([Le Cam, 1973](#ref-lecam1973convergence)).

**Proof.**

**Step 1 (risk of generalized least squares).** Let $`I\succ0`$ and $`\hat\vartheta=I^{-1}D^\top\Sigma^{-1}y`$. Then $`\hat\vartheta-\vartheta=I^{-1}D^\top\Sigma^{-1}\epsilon`$ has mean zero and covariance $`I^{-1}D^\top\Sigma^{-1}\Sigma\Sigma^{-1}DI^{-1}=I^{-1}`$. Hence, for every $`\vartheta`$,

```math
\mathbb E\lVert C(\hat\vartheta-\vartheta)\rVert_W^2=\mathrm{tr}\big(WC\,I^{-1}C^\top\big)=\mathrm{tr}\big(C^\top WC\,I^{-1}\big)=\mathrm{tr}(BI^{-1}).
```

**Step 2 (lower bound).** Put the prior $`\vartheta\sim\mathcal N(0,\tau^2I_d)`$. The posterior of $`\vartheta`$ given $`y`$ is Gaussian with covariance $`(I+\tau^{-2}I_d)^{-1}`$, and for the loss $`\lVert\cdot\rVert_W^2`$ on $`C\vartheta`$ the posterior mean $`C\,\mathbb E[\vartheta\mid y]`$ is Bayes, with Bayes risk $`\mathrm{tr}\big(B(I+\tau^{-2}I_d)^{-1}\big)`$. Any estimator's worst-case risk is at least its average risk under any prior, hence at least every Bayes risk:

```math
\inf_{\tilde f}\sup_\vartheta\mathbb E\lVert\tilde f-C\vartheta\rVert_W^2\ \ge\ \sup_{\tau>0}\mathrm{tr}\big(B(I+\tau^{-2}I_d)^{-1}\big).
```

Write $`I=U\,\mathrm{diag}(\iota_1,\dots,\iota_d)\,U^\top`$. Then $`\mathrm{tr}(B(I+\varepsilon I_d)^{-1})=\sum_iu_i^\top Bu_i/(\iota_i+\varepsilon)`$ with $`u_i^\top Bu_i=\lVert W^{1/2}Cu_i\rVert^2\ge0`$. Each term increases as $`\varepsilon=\tau^{-2}\downarrow0`$. If $`I\succ0`$, the limit is $`\sum_iu_i^\top Bu_i/\iota_i=\mathrm{tr}(BI^{-1})`$, which with Step 1 proves that the risk in (a) is minimax.

**Step 3 (the eigen-identity).** For $`I\succ0`$, $`\mathrm{tr}(BI^{-1})=\mathrm{tr}(I^{-1/2}BI^{-1/2})`$ is the sum of the eigenvalues of the symmetric matrix $`I^{-1/2}BI^{-1/2}`$. Now $`I^{-1/2}BI^{-1/2}w=\lambda w`$ if and only if $`Bv=\lambda Iv`$ with $`v=I^{-1/2}w`$, so these eigenvalues are the generalized eigenvalues $`\lambda_j`$ of $`(B,I)`$. This finishes (a). For (c), $`\mathcal R_I^\top\mathcal R_I=I^{-1/2}C^\top WCI^{-1/2}=I^{-1/2}BI^{-1/2}`$, whose eigenvalues are the squared singular values of $`\mathcal R_I`$; and $`\lVert\mathcal R_I\rVert_{\mathrm{HS}}^2`$ is their sum.

**Step 4 (singular information).** Let $`P_0`$ be the orthogonal projector onto $`\ker I`$. Since $`D^\top\Sigma^{-1}Dv=0`$ implies $`\lVert\Sigma^{-1/2}Dv\rVert^2=0`$, every $`v\in\ker I`$ has $`Dv=0`$, so the law of $`y`$ does not depend on $`P_0\vartheta`$.

- Suppose $`W^{1/2}C`$ vanishes on $`\ker I`$. For $`\hat f=CI^+D^\top\Sigma^{-1}y`$ we have $`\mathbb E\hat f=CI^+I\vartheta=C(I_d-P_0)\vartheta`$, so the bias $`-CP_0\vartheta`$ satisfies $`\lVert CP_0\vartheta\rVert_W=\lVert W^{1/2}CP_0\vartheta\rVert=0`$, and it vanishes when $`C\ker I=\lbrace0\rbrace`$. The fluctuation has covariance $`CI^+D^\top\Sigma^{-1}DI^+C^\top=CI^+II^+C^\top=CI^+C^\top`$. So the risk is $`\mathrm{tr}(WCI^+C^\top)=\mathrm{tr}(BI^+)`$ for every $`\vartheta`$. In the sum of Step 2, every eigenvector $`u_i\in\ker I`$ has $`u_i^\top Bu_i=\lVert W^{1/2}Cu_i\rVert^2=0`$, and the remaining terms increase to $`\sum_{\iota_i>0}u_i^\top Bu_i/\iota_i=\mathrm{tr}(BI^+)`$. The risk is therefore minimax.
- Suppose $`W^{1/2}Cv\ne0`$ for a unit $`v\in\ker I`$. Then $`v^\top Bv>0`$ and, because $`(I+\varepsilon I_d)^{-1}`$ has eigenvalue $`1/\varepsilon`$ on $`v`$, $`\mathrm{tr}(B(I+\varepsilon I_d)^{-1})\ge v^\top Bv/\varepsilon\to\infty`$. The minimax risk over $`\mathbb R^d`$ is $`+\infty`$.
- On a bounded parameter set containing $`\vartheta_0`$ and $`\vartheta_1=\vartheta_0+sv`$ with such a $`v`$ and $`s\ne0`$, the two points give the same law of $`y`$. For any estimator $`\tilde f`$, the triangle inequality gives $`\lVert\tilde f-C\vartheta_0\rVert_W+\lVert\tilde f-C\vartheta_1\rVert_W\ge s\lVert W^{1/2}Cv\rVert`$ pointwise, so the larger of the two risks is at least $`s^2\lVert W^{1/2}Cv\rVert^2/4`$: a positive floor. This is Le Cam's two-point argument with zero total-variation distance.

**Step 5 (one query).** With $`I\succ0`$, the Sherman–Morrison formula gives

```math
(I+aa^\top)^{-1}=I^{-1}-\frac{I^{-1}aa^\top I^{-1}}{1+a^\top I^{-1}a},
```

so $`\mathrm{tr}(BI^{-1})-\mathrm{tr}\big(B(I+aa^\top)^{-1}\big)=\mathrm{tr}(BI^{-1}aa^\top I^{-1})/(1+a^\top I^{-1}a)=a^\top I^{-1}BI^{-1}a/(1+a^\top I^{-1}a)`$. By (a) applied to the enlarged experiment, both traces are minimax risks.

**Step 6 (the scalar case).** For $`C=c^\top`$ and $`W=1`$, $`B=cc^\top`$ and $`\mathrm{tr}(BI^{-1})=c^\top I^{-1}c`$. With $`I=A(\xi)`$ this is the criterion minimized over designs in [Lemma 6.4](../statements/06-04_design_minimax_equality.md#s6-4). $`\square`$

**Comments.**

- The floor in Step 4 uses $`\max(R_0,R_1)\ge(R_0+R_1)/2`$ and $`X_0^2+X_1^2\ge(X_0+X_1)^2/2`$ with $`X_j=\lVert\tilde f-C\vartheta_j\rVert_W`$, both expectations taken under the common law of $`y`$; the constant $`1/4`$ is not optimized.

## Related work

- The trace criterion, its A-, L- and c-optimal special cases and the pseudo-inverse treatment of singular information matrices are standard in optimal design ([Pukelsheim, 1993](#ref-pukelsheim1993optimal); [Elfving, 1952](#ref-elfving1952optimum)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).
