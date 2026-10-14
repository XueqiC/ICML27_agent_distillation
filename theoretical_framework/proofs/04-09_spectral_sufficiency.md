# Theorem 4.9 · Spectral sufficiency and linear interface complexity

*Proof*

[← Framework](../README.md) · [Statement](../statements/04-09_spectral_sufficiency.md) · [Proof](04-09_spectral_sufficiency.md) · [Experiments](../experiments/04-09_spectral_sufficiency.md) · [中文](04-09_spectral_sufficiency_ZH.md)

**What is proved here.** Parts (a)–(b) are Parseval and Bessel in the singular bases. Part (c) is the Eckart–Young argument with an arbitrary decoder: $`r`$ linear statistics cannot see some direction of an $`(r+1)`$-dimensional space. Part (d) is Cauchy–Schwarz on the truncated part of one test's response. Part (e) is a counting bound and an explicit example that attains it. Part (f) turns the invisible direction of (c) into two probability laws.

**Setting used on this page.**

- $`H=L_0^2(\rho)`$, the response operator $`\mathcal R:H\to L^2(\omega)`$, $`(\mathcal Rh)(a)=\langle\tilde g_a,h\rangle_\rho`$, with $`\lVert\mathcal R\rVert_{\rm HS}^2\le\tfrac14`$ and singular value decomposition $`\mathcal Rh=\sum_j\sigma_j\langle v_j,h\rangle_\rho u_j`$ ([Definition 4.8](../statements/04-08_response_operator.md#s4-8)).
- $`h=p-q`$ for laws with $`L^2(\rho)`$ densities, $`\lVert h\rVert_\rho\le B`$, $`\alpha_j=\langle v_j,h\rangle_\rho`$.
- Rank-$`r`$ linear interfaces $`h\mapsto D(s(h))`$ with $`s(h)=(\langle w_i,h\rangle_\rho)_{i\le r}`$, $`w_i\in H`$, and an arbitrary measurable decoder $`D`$.
- $`b_j(a)=\langle\tilde g_a,v_j\rangle_\rho`$, $`e_r(a)=\lVert\tilde g_a-\sum_{j\le r}b_j(a)v_j\rVert_\rho`$; $`\Pi_r`$ is the orthogonal projection of $`H`$ onto $`\mathrm{span}(v_1,\dots,v_r)`$.

**Statement ([Theorem 4.9](../statements/04-09_spectral_sufficiency.md#s4-9)).** (a) $`\lVert\mathcal Rh\rVert_\omega^2=\sum_j\sigma_j^2\alpha_j^2`$. (b) $`\lvert\alpha_j\rvert\le\delta_j`$ for $`j\le r`$ implies $`\lVert\mathcal Rh\rVert_\omega^2\le\sum_{j\le r}\sigma_j^2\delta_j^2+B^2\sigma_{r+1}^2`$. (c) The minimax error of rank-$`r`$ linear interfaces over $`\lbrace h\in H:\lVert h\rVert_\rho\le B\rbrace`$ is $`B\sigma_{r+1}`$, attained by the leading modes. (d) $`\lvert\mathbb E_Pg_a-\mathbb E_Qg_a\rvert\le\lvert\sum_{j\le r}b_j(a)\alpha_j\rvert+Be_r(a)`$ and $`\mathbb E_\omega e_r^2=\sum_{j>r}\sigma_j^2`$. (e) $`(r+1)\sigma_{r+1}^2\le\tfrac14`$; under the $`J`$-candidate mixture measure the bound $`B^2=2J`$ is attained and the resulting dimension is at least the trivial one when $`\varepsilon\le1/\sqrt2`$. (f) On a finite message set, the laws (4.17) are probability laws with equal statistics and response gap at least $`a\sigma_{r+1}`$.

**Proof.**

**Preliminaries.** $`\mathcal R`$ is Hilbert–Schmidt, hence compact, and $`H`$ and $`L^2(\omega)`$ are separable. The singular value decomposition of a compact operator gives orthonormal $`(v_j)`$ spanning $`(\ker\mathcal R)^\perp`$, orthonormal $`(u_j)`$, and $`\mathcal Rh=\sum_j\sigma_j\langle v_j,h\rangle_\rho u_j`$ for every $`h\in H`$, the series converging in $`L^2(\omega)`$. If $`\mathcal R`$ has finite rank $`k`$, set $`\sigma_j=0`$ for $`j>k`$ and complete $`(v_j)`$ by an orthonormal basis of $`\ker\mathcal R`$.

**(a)** By orthonormality of the $`u_j`$ (Parseval), $`\lVert\mathcal Rh\rVert_\omega^2=\sum_j\sigma_j^2\langle v_j,h\rangle_\rho^2=\sum_j\sigma_j^2\alpha_j^2`$. Since $`v_j\in H`$ has $`\rho`$-mean zero, $`\alpha_j=\int v_j(p-q)\,d\rho=\mathbb E_Pv_j-\mathbb E_Qv_j`$.

**(b)** Split (a) at $`r`$. The first part is at most $`\sum_{j\le r}\sigma_j^2\delta_j^2`$. For the tail, $`\sigma_j\le\sigma_{r+1}`$ for $`j>r`$ and Bessel's inequality gives $`\sum_{j>r}\alpha_j^2\le\lVert h\rVert_\rho^2\le B^2`$.

**(c)** *Upper bound.* With $`w_j=v_j`$ and $`D(s)=\sum_{j\le r}\sigma_js_ju_j`$, the error is $`\mathcal Rh-\mathcal R\Pi_rh=\sum_{j>r}\sigma_j\alpha_ju_j`$, whose norm is at most $`\sigma_{r+1}\lVert h\rVert_\rho\le B\sigma_{r+1}`$ by the tail estimate of (b).

*Lower bound.* Fix any $`w_1,\dots,w_r\in H`$ and any decoder $`D`$. If $`\sigma_{r+1}=0`$ there is nothing to prove. Otherwise $`V=\mathrm{span}(v_1,\dots,v_{r+1})`$ has dimension $`r+1`$, and the linear map $`V\to\mathbb R^r`$, $`x\mapsto s(x)`$, has a nontrivial kernel. Pick $`v\in V`$ in that kernel with $`\lVert v\rVert_\rho=1`$. By (a), $`\lVert\mathcal Rv\rVert_\omega^2=\sum_{j\le r+1}\sigma_j^2\langle v_j,v\rangle_\rho^2\ge\sigma_{r+1}^2`$. The inputs $`h_\pm=\pm Bv`$ lie in the ball and have $`s(h_\pm)=0`$, so the interface returns the same $`f=D(0)`$ for both. By the triangle inequality,

```math
\lVert\mathcal Rh_+-f\rVert_\omega+\lVert\mathcal Rh_--f\rVert_\omega\ge\lVert\mathcal Rh_+-\mathcal Rh_-\rVert_\omega=2B\lVert\mathcal Rv\rVert_\omega\ge2B\sigma_{r+1},
```

so one of the two errors is at least $`B\sigma_{r+1}`$. The decoder was arbitrary, so the infimum in (4.14) is at least $`B\sigma_{r+1}`$; the upper bound shows it is attained.

**(d)** Since $`\mathbb E_\rho h=0`$, $`\mathbb E_Pg_a-\mathbb E_Qg_a=\langle\tilde g_a,h\rangle_\rho`$. Write $`\tilde g_a=\sum_{j\le r}b_j(a)v_j+\eta_a`$ with $`\eta_a=(I-\Pi_r)\tilde g_a`$, $`\lVert\eta_a\rVert_\rho=e_r(a)`$. Then

```math
\langle\tilde g_a,h\rangle_\rho=\sum_{j\le r}b_j(a)\alpha_j+\langle\eta_a,h\rangle_\rho,\qquad\lvert\langle\eta_a,h\rangle_\rho\rvert\le e_r(a)\,B
```

by Cauchy–Schwarz. For the average: $`(\mathcal R(I-\Pi_r)h)(a)=\langle\tilde g_a,(I-\Pi_r)h\rangle_\rho=\langle\eta_a,h\rangle_\rho`$, so $`\eta_a`$ is the kernel of the operator $`\mathcal R(I-\Pi_r)`$ and $`\mathbb E_\omega e_r(a)^2=\lVert\mathcal R(I-\Pi_r)\rVert_{\rm HS}^2=\sum_{j>r}\sigma_j^2`$. Finally, for a finite $`\mathcal A`$, $`\omega_a(\mathcal Rh)(a)^2\le\sum_{a'}\omega_{a'}(\mathcal Rh)(a')^2=\lVert\mathcal Rh\rVert_\omega^2`$.

**(e)** The singular values are nonincreasing, so $`(r+1)\sigma_{r+1}^2\le\sum_{j\le r+1}\sigma_j^2\le\lVert\mathcal R\rVert_{\rm HS}^2\le\tfrac14`$ by (4.11). Hence $`B\sigma_{r+1}\le B/(2\sqrt{r+1})`$, which is at most $`\varepsilon`$ when $`r+1\ge B^2/(4\varepsilon^2)`$.

*Attainment of $`B^2=2J`$.* Let $`P^j=\delta_{y_j}`$ for distinct messages $`y_1,\dots,y_J`$. Then $`\rho`$ is uniform on $`\lbrace y_1,\dots,y_J\rbrace`$, the density of $`P^j`$ is $`J\mathbf 1\lbrace y=y_j\rbrace`$, and for $`h=p_1-p_2`$, $`\lVert h\rVert_\rho^2=\tfrac1J(J^2+J^2)=2J`$.

*No compression content.* With $`B^2=2J`$, (4.16) requires $`r+1\ge J/(2\varepsilon^2)\ge J`$ when $`\varepsilon\le1/\sqrt2`$, i.e. $`r\ge J-1`$. Every difference of two mixtures $`\sum_j\lambda_jP^j`$ and $`\sum_j\lambda_j'P^j`$ has density $`\sum_j(\lambda_j-\lambda_j')p_j`$ with $`\sum_j(\lambda_j-\lambda_j')=0`$, so it lies in $`W=\mathrm{span}\lbrace p_j-p_1:j\le J\rbrace`$, of dimension at most $`J-1`$. Choose $`w_1,\dots,w_{J-1}`$ spanning $`W`$ (padding with zeros if the dimension is smaller). On $`W`$ the map $`h\mapsto s(h)`$ is injective, so a linear decoder recovers $`h`$ and hence $`\mathcal Rh`$ exactly. A bound that asks for at least $`J-1`$ statistics therefore guarantees nothing that the trivial interface does not.

**(f)** Here $`H=\lbrace h:\sum_y\rho_yh_y=0\rbrace`$ has dimension $`m-1\ge r+1`$, so the construction of (c) applies (completing the $`v_j`$ inside $`H`$ if the rank is below $`r+1`$; then $`\sigma_{r+1}=0`$ and the claim is trivial). Let $`v`$ be the unit vector of (c) on which the $`r`$ statistics vanish. Define $`P,Q`$ by (4.17).

- *Probability laws.* $`P_y,Q_y\ge0`$ because $`a\lvert v_y\rvert\le a\lVert v\rVert_\infty\le2`$. Their total masses are $`1\pm\tfrac a2\sum_y\rho_yv_y=1`$, since $`v\in H`$.
- *Equal statistics.* $`p-q=av`$, so $`s(p-q)=a\,s(v)=0`$. For features $`\phi_i`$ this is $`\mathbb E_P\phi_i=\mathbb E_Q\phi_i`$.
- *Gap.* $`\lVert p-q\rVert_\rho=a`$ and $`\lVert\mathcal R(p-q)\rVert_\omega=a\lVert\mathcal Rv\rVert_\omega\ge a\sigma_{r+1}`$.
- *Uniform choice of $`a`$.* $`1=\lVert v\rVert_\rho^2\ge\rho_yv_y^2`$ for every $`y`$, so $`\lVert v\rVert_\infty\le1/\sqrt{\rho_{\min}}`$ and $`a=2\sqrt{\rho_{\min}}\le2/\lVert v\rVert_\infty`$.
- *Worst case over probability pairs.* For $`B\le2\sqrt{\rho_{\min}}`$ take $`a=B`$. The pairs $`(P,Q)`$ and $`(Q,P)`$ both produce the statistics $`0`$ and the differences $`\pm Bv`$; the argument of (c) gives an error of at least $`B\sigma_{r+1}`$ on one of them, and the upper bound of (c) applies to every pair with $`\lVert p-q\rVert_\rho\le B`$. $`\square`$

## Related work

- The lower bound in (c) is the Eckart–Young argument ([Eckart & Young, 1936](#ref-eckart1936approximation)) with the linear reconstruction replaced by an arbitrary decoder; the decoder does not help because the encoder is linear.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).
