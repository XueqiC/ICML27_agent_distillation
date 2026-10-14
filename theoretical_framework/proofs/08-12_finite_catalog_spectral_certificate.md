# Proposition 8.12 · Finite-catalog spectral certificate

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-12_finite_catalog_spectral_certificate.md) · [Proof](08-12_finite_catalog_spectral_certificate.md) · [Experiments](../experiments/08-12_finite_catalog_spectral_certificate.md) · [中文](08-12_finite_catalog_spectral_certificate_ZH.md)

**What is proved here.** (a) is a per-cell Hoeffding bound with a union bound. (b) uses that the weighted Frobenius norm of the error matrix is at most the largest entry error, and Weyl's inequality. (c) is a three-term split. (d) expands an estimated row in the estimated modes and pays for the estimation error of the cells. (e) is an orthogonal decomposition of a mean-squared error. (f) counts runs.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- One checkpoint $`(t,z)`$; $`p`$ tests with weights $`w_i>0`$ summing to one; $`m`$ frozen candidate feedback laws with weights $`q_j>0`$ summing to one; $`p_{\rm train}`$ distinct continuation protocols among the tests.
- $`G_{ij}\in[0,1]`$ is the expected final score of test $`i`$ after one round with candidate $`j`$ and the test's continuation; $`\hat G_{ij}`$ is the average of $`n`$ replicates, independent across the replicate index.
- $`M=D_{\sqrt w}GD_{\sqrt q}\Pi_q`$ with $`\Pi_q=I-\sqrt q\sqrt q^{\top}`$; $`\hat M`$ likewise from $`\hat G`$; singular values $`\sigma_k`$, $`\hat\sigma_k`$; $`\hat\Pi_r`$ projects onto the first $`r`$ right singular vectors $`\hat v_1,\dots,\hat v_r`$ of $`\hat M`$.
- For laws $`P,Q`$ over the candidates: $`h_j=(P_j-Q_j)/q_j`$, $`x=D_{\sqrt q}h`$, $`a_i=\sum_j(P_j-Q_j)G_{ij}`$, $`\lVert a\rVert_\omega^2=\sum_iw_ia_i^2`$.
- $`\hat g_i`$ is row $`i`$ of $`\hat G`$, centred under $`q`$; $`\hat\phi_k=D_{\sqrt q}^{-1}\hat v_k`$; $`\hat b_k(i)=\langle\hat g_i,\hat\phi_k\rangle_q`$; $`\hat e_r(i)=\lVert\hat g_i-\sum_{k\le r}\hat b_k(i)\hat\phi_k\rVert_q`$; $`\alpha_k=\sum_j(P_j-Q_j)\hat\phi_k(j)`$.

**Statement ([Proposition 8.12](../statements/08-12_finite_catalog_spectral_certificate.md#s8-12)).** With $`\varepsilon_n=\sqrt{\log(2pm/\delta)/(2n)}`$: (a) with probability at least $`1-\delta`$ all cells satisfy $`\lvert\hat G_{ij}-G_{ij}\rvert\le\varepsilon_n`$; on this event (b) $`\lVert\hat M-M\rVert_{\rm op}\le\varepsilon_n`$ and $`\lvert\hat\sigma_k-\sigma_k\rvert\le\varepsilon_n`$; (c) $`\lVert M(I-\hat\Pi_r)\rVert_{\rm op}\le\hat\sigma_{r+1}+\varepsilon_n\le\sigma_{r+1}+2\varepsilon_n`$; (d) for $`\lVert h\rVert_q\le B`$, $`\lvert a_i\rvert\le\lvert\sum_{k\le r}\hat b_k(i)\alpha_k\rvert+B\hat e_r(i)+2\varepsilon_n`$, with $`\sum_k\lvert\hat b_k(i)\rvert\eta_k`$ added when $`\alpha_k`$ is replaced by an estimate with error at most $`\eta_k`$; (e) with the exact singular triples $`(\sigma_k,u_k,v_k)`$ of $`M`$, $`\phi_k=D_{\sqrt q}^{-1}v_k`$, $`\hat\alpha_k`$ the difference of sample means over $`n'`$ independent draws from each of $`P`$ and $`Q`$, $`f=Mx`$ and $`\hat f_r=\sum_{k\le r}\sigma_k\hat\alpha_ku_k`$,

```math
\mathbb E\,\lVert\hat f_r-f\rVert_2^2=\sum_{k\le r}\frac{\sigma_k^2}{n'}\big(\mathrm{Var}_P\phi_k+\mathrm{Var}_Q\phi_k\big)+\sum_{k>r}\sigma_k^2\alpha_k^2 ;
```

with triples of an independent frozen estimate $`\hat M`$ instead, the variance term carries hats and the bias term is $`\lVert(M-\hat M\hat\Pi_r)x\rVert_2^2`$; (f) a complete catalog costs $`m\,n\,p_{\rm train}`$ continuation runs.

*Proof.*

**Step 0: $`M`$ represents the response operator.** For $`x=D_{\sqrt q}h`$ with $`\sum_jq_jh_j=0`$ we have $`\sqrt q^{\top}x=0`$, so $`\Pi_qx=x`$ and

```math
(Mx)_i=\sqrt{w_i}\sum_jG_{ij}\sqrt{q_j}\,x_j=\sqrt{w_i}\sum_jG_{ij}(P_j-Q_j)=\sqrt{w_i}\,a_i .
```

Hence $`\lVert Mx\rVert_2=\lVert a\rVert_\omega`$ and $`\lVert x\rVert_2^2=\sum_jq_jh_j^2=\lVert h\rVert_q^2`$. Also $`\lVert M\rVert_{\rm HS}^2=\sum_iw_i\,\mathrm{Var}_q(G_{i\cdot})\le\tfrac14`$ because each row takes values in $`[0,1]`$.

**(a)** For a fixed cell the $`n`$ replicate outcomes are independent and lie in $`[0,1]`$ (their mean is $`G_{ij}`$, and dependence across candidates or across tests read off one run does not enter a single cell). Hoeffding's inequality ([Hoeffding, 1963](#ref-hoeffding1963probability)) gives $`\Pr(\lvert\hat G_{ij}-G_{ij}\rvert>s)\le2e^{-2ns^2}`$. With $`s=\varepsilon_n`$ this is $`\delta/(pm)`$, and a union bound over the $`pm`$ cells gives (a).

**(b)** Let $`E=\hat G-G`$, so $`\hat M-M=D_{\sqrt w}ED_{\sqrt q}\Pi_q`$. On the event of (a),

```math
\lVert D_{\sqrt w}ED_{\sqrt q}\rVert_{\rm F}^2=\sum_{i,j}w_iq_jE_{ij}^2\le\varepsilon_n^2\sum_iw_i\sum_jq_j=\varepsilon_n^2 .
```

The operator norm is at most the Frobenius norm and $`\lVert\Pi_q\rVert_{\rm op}=1`$, so $`\lVert\hat M-M\rVert_{\rm op}\le\varepsilon_n`$. Weyl's inequality for singular values, $`\lvert\sigma_k(A+E')-\sigma_k(A)\rvert\le\lVert E'\rVert_{\rm op}`$, gives the second claim.

**(c)** Write $`M(I-\hat\Pi_r)=\hat M(I-\hat\Pi_r)+(M-\hat M)(I-\hat\Pi_r)`$. By the singular value decomposition of $`\hat M`$, $`\lVert\hat M(I-\hat\Pi_r)\rVert_{\rm op}=\hat\sigma_{r+1}`$. The second term has norm at most $`\lVert M-\hat M\rVert_{\rm op}\lVert I-\hat\Pi_r\rVert_{\rm op}\le\varepsilon_n`$. Finally $`\hat\sigma_{r+1}\le\sigma_{r+1}+\varepsilon_n`$ by (b). No gap between $`\sigma_r`$ and $`\sigma_{r+1}`$ is used.

**(d)** Let $`\hat a_i=\sum_j(P_j-Q_j)\hat G_{ij}`$. On the event of (a),

```math
\lvert a_i-\hat a_i\rvert\le\sum_j\lvert P_j-Q_j\rvert\,\lvert G_{ij}-\hat G_{ij}\rvert\le2\varepsilon_n ,
```

because $`\sum_j\lvert P_j-Q_j\rvert\le2`$ for two probability vectors. Since $`\sum_jq_jh_j=0`$, centring the row does not change the inner product: $`\hat a_i=\sum_jq_jh_j\hat G_{ij}=\langle\hat g_i,h\rangle_q`$. Decompose $`\hat g_i=\sum_{k\le r}\hat b_k(i)\hat\phi_k+\hat\varrho_i`$ with $`\lVert\hat\varrho_i\rVert_q=\hat e_r(i)`$. Because $`\langle\hat\phi_k,h\rangle_q=\sum_jq_j\hat\phi_k(j)h_j=\sum_j(P_j-Q_j)\hat\phi_k(j)=\alpha_k`$,

```math
\hat a_i=\sum_{k\le r}\hat b_k(i)\alpha_k+\langle\hat\varrho_i,h\rangle_q,\qquad\lvert\langle\hat\varrho_i,h\rangle_q\rvert\le\hat e_r(i)\,B
```

by Cauchy–Schwarz. Combining with the first display gives the bound. If $`\alpha_k`$ is replaced by $`\hat\alpha_k`$ with $`\lvert\hat\alpha_k-\alpha_k\rvert\le\eta_k`$, then $`\lvert\sum_k\hat b_k(i)\alpha_k\rvert\le\lvert\sum_k\hat b_k(i)\hat\alpha_k\rvert+\sum_k\lvert\hat b_k(i)\rvert\eta_k`$.

**(e)** By Step 0 and the singular value decomposition, $`f=Mx=\sum_k\sigma_k\langle v_k,x\rangle u_k`$ and $`\langle v_k,x\rangle=\sum_jv_{kj}\sqrt{q_j}h_j=\sum_j(P_j-Q_j)\phi_k(j)=\alpha_k`$. Then

```math
\hat f_r-f=\sum_{k\le r}\sigma_k(\hat\alpha_k-\alpha_k)u_k-\sum_{k>r}\sigma_k\alpha_ku_k .
```

The $`u_k`$ are orthonormal, so the squared norm is the sum of the squared coefficients; all cross terms vanish, whatever the dependence between the $`\hat\alpha_k`$. Each $`\hat\alpha_k`$ is a difference of two independent sample means of $`\phi_k`$, so it is unbiased with variance $`(\mathrm{Var}_P\phi_k+\mathrm{Var}_Q\phi_k)/n'`$. Taking expectations gives the formula. The modes are fixed independently of the draws; this is what makes $`\hat\alpha_k`$ unbiased for the fixed $`\alpha_k`$.

*Frozen estimated triples.* Let $`(\hat\sigma_k,\hat u_k,\hat v_k)`$ be the singular triples of an estimate $`\hat M`$ fixed independently of the draws, $`\hat\phi_k=D_{\sqrt q}^{-1}\hat v_k`$, $`\hat\alpha_k`$ the sample-mean difference of $`\hat\phi_k`$ and $`\hat f_r=\sum_{k\le r}\hat\sigma_k\hat\alpha_k\hat u_k`$. As above, $`\mathbb E\hat\alpha_k=\langle\hat v_k,x\rangle`$, so $`\mathbb E\hat f_r=\hat M\hat\Pi_rx`$, and $`\hat f_r-\mathbb E\hat f_r=\sum_{k\le r}\hat\sigma_k(\hat\alpha_k-\mathbb E\hat\alpha_k)\hat u_k`$ has mean zero and, by orthonormality of the $`\hat u_k`$, squared norm with expectation $`\sum_{k\le r}\hat\sigma_k^2(\mathrm{Var}_P\hat\phi_k+\mathrm{Var}_Q\hat\phi_k)/n'`$. The cross term with the deterministic bias vanishes in expectation, so the mean squared error is that variance plus $`\lVert\hat M\hat\Pi_rx-Mx\rVert_2^2=\lVert(M-\hat M\hat\Pi_r)x\rVert_2^2`$. It equals the first formula only when $`\hat M=M`$; in general the bias is not $`\sum_{k>r}\sigma_k^2\alpha_k^2`$.

**(f)** One run of candidate $`j`$ under one continuation protocol produces every test read off that protocol, so a replicate of column $`j`$ needs $`p_{\rm train}`$ runs; $`m`$ columns and $`n`$ replicates need $`m\,n\,p_{\rm train}`$. $`\square`$

## Related work

- Steps (b)–(c) are the standard perturbation companions of the Eckart–Young theorem ([Eckart & Young, 1936](#ref-eckart1936approximation)); the entrywise radius is Hoeffding's ([Hoeffding, 1963](#ref-hoeffding1963probability)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).
