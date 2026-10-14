# Theorem 4.9 · Spectral sufficiency and linear interface complexity

*Statement*

[← Framework](../README.md) · [Statement](04-09_spectral_sufficiency.md) · [Proof](../proofs/04-09_spectral_sufficiency.md) · [Experiments](../experiments/04-09_spectral_sufficiency.md) · [中文](04-09_spectral_sufficiency_ZH.md)

**In one paragraph.** At a fixed checkpoint, the response operator of [Definition 4.8](04-08_response_operator.md#s4-8) turns a difference between two feedback laws into score differences on every declared test. Its singular values say how much each direction of feedback difference moves the tests. This theorem makes that precise: matching the leading input modes leaves an error of at most the next singular value times the size of the difference; no interface built from $`r`$ linear statistics does better in the worst case, even with a nonlinear decoder; a per-test bound holds for every individual test; and the same lower bound is reached by actual probability laws. It also records what the theorem does not give: the generic dimension bound that follows from boundedness alone has no compression content under the natural mixture measure, so useful compression is a measured property, not a consequence of the setting.

**Setting used on this page.**

- A checkpoint $`(t,z)`$, declared tests $`\mathcal A`$ with weights $`\omega`$, reference message measure $`\rho`$, responses $`g_a\in[0,1]`$, and the response operator $`\mathcal R:H=L_0^2(\rho)\to L^2(\omega)`$ with singular value decomposition $`\mathcal Rh=\sum_j\sigma_j\langle v_j,h\rangle_\rho u_j`$ ([Definition 4.8](04-08_response_operator.md#s4-8)).
- Two feedback laws $`P,Q`$ with densities $`p,q\in L^2(\rho)`$, their difference $`h=p-q\in H`$ and a bound $`\lVert h\rVert_\rho\le B`$. The *mode differences* are $`\alpha_j=\langle v_j,h\rangle_\rho=\mathbb E_Pv_j-\mathbb E_Qv_j`$.
- A *rank-$`r`$ linear interface* reads $`r`$ linear statistics $`s(h)=(\langle w_1,h\rangle_\rho,\dots,\langle w_r,h\rangle_\rho)`$ with $`w_i\in H`$ (for features $`\phi_i\in L^2(\rho)`$ these are the feature-mean differences $`\mathbb E_P\phi_i-\mathbb E_Q\phi_i`$, with $`w_i=\phi_i-\mathbb E_\rho\phi_i`$) and returns $`D(s(h))`$ for an arbitrary measurable decoder $`D:\mathbb R^r\to L^2(\omega)`$.
- For a test $`a`$: $`b_j(a)=\langle\tilde g_a,v_j\rangle_\rho`$ and $`e_r(a)=\lVert\tilde g_a-\sum_{j\le r}b_j(a)v_j\rVert_\rho`$, the part of test $`a`$'s centred response outside the leading $`r`$ input modes.

<a id="s4-9"></a>

**Theorem 4.9 (Spectral sufficiency and linear interface complexity).** In this setting:

**(a) Mode identity.**

```math
\lVert\mathcal Rh\rVert_\omega^2=\sum_{j\ge1}\sigma_j^2\alpha_j^2 . \qquad(4.12)
```

**(b) Matched leading modes.** If $`\lvert\alpha_j\rvert\le\delta_j`$ for $`j\le r`$, then

```math
\lVert\mathcal Rh\rVert_\omega^2\le\sum_{j\le r}\sigma_j^2\delta_j^2+B^2\sigma_{r+1}^2 ; \qquad(4.13)
```

in particular $`\alpha_1=\dots=\alpha_r=0`$ gives $`\lVert\mathcal Rh\rVert_\omega\le B\sigma_{r+1}`$.

**(c) Optimal linear interface.** Over all rank-$`r`$ linear interfaces and all decoders,

```math
\inf_{w_1,\dots,w_r,\,D}\ \sup_{h\in H,\ \lVert h\rVert_\rho\le B}\ \big\lVert\mathcal Rh-D(s(h))\big\rVert_\omega=B\sigma_{r+1}, \qquad(4.14)
```

attained by $`w_j=v_j`$ and the linear decoder $`D(s)=\sum_{j\le r}\sigma_js_ju_j`$. The *linear interface complexity* is $`r_\varepsilon(B)=\min\lbrace r:B\sigma_{r+1}\le\varepsilon\rbrace`$.

**(d) Every test.** For every test $`a`$,

```math
\lvert\mathbb E_Pg_a-\mathbb E_Qg_a\rvert\le\Big\lvert\sum_{j\le r}b_j(a)\alpha_j\Big\rvert+B\,e_r(a),\qquad \mathbb E_{a\sim\omega}e_r(a)^2=\sum_{j>r}\sigma_j^2 . \qquad(4.15)
```

If $`\mathcal A`$ is finite and $`\omega_a>0`$, then also $`\lvert(\mathcal Rh)(a)\rvert\le\lVert\mathcal Rh\rVert_\omega/\sqrt{\omega_a}`$.

**(e) The generic dimension bound and where it is empty.** Without any assumption on the spectrum, $`(r+1)\sigma_{r+1}^2\le\tfrac14`$, so $`B\sigma_{r+1}\le B/(2\sqrt{r+1})`$, and

```math
r+1\ge\frac{B^2}{4\varepsilon^2}\quad\Longrightarrow\quad B\sigma_{r+1}\le\varepsilon . \qquad(4.16)
```

This dimension is sufficient, not necessary. Under the mixture measure $`\rho=J^{-1}\sum_{j\le J}P^j`$ of $`J`$ candidate laws, the worst-case bound $`B^2=2J`$ of [Definition 4.8](04-08_response_operator.md#s4-8) is attained when all $`J`$ candidates are point masses on distinct messages (then any two of them have $`\lVert h\rVert_\rho^2=2J`$). With that $`B`$, (4.16) asks for $`r+1\ge J/(2\varepsilon^2)`$, which for every $`\varepsilon\le1/\sqrt2`$ means $`r\ge J-1`$. But all differences of mixtures of the $`J`$ candidates lie in a space of dimension at most $`J-1`$, on which $`J-1`$ linear statistics are already exact. The generic bound therefore carries no compression content in this setting.

**(f) The lower bound is reached by probability laws.** Let $`\mathcal Y`$ be finite with $`m`$ messages, $`\rho_y>0`$ for every $`y`$, $`\rho_{\min}=\min_y\rho_y`$, and $`0\le r<m-1`$. For any $`r`$ linear statistics there is a unit $`v\in\mathrm{span}(v_1,\dots,v_{r+1})`$ on which all of them vanish, and for every $`0<a\le2/\lVert v\rVert_\infty`$ the laws

```math
P_y=\rho_y\big(1+\tfrac a2v_y\big),\qquad Q_y=\rho_y\big(1-\tfrac a2v_y\big) \qquad(4.17)
```

are probability laws with equal statistics, $`\lVert p-q\rVert_\rho=a`$ and $`\lVert\mathcal R(p-q)\rVert_\omega\ge a\sigma_{r+1}`$. The choice $`a=2\sqrt{\rho_{\min}}`$ always qualifies. Hence for $`B\le2\sqrt{\rho_{\min}}`$ the value $`B\sigma_{r+1}`$ of (4.14) is also the worst-case error over pairs of probability laws.

**What it says.**

- The singular values of the response operator are the gains of the directions in which feedback laws can differ. Matching the $`r`$ largest-gain directions leaves at most the next gain times the size of the difference.
- Nothing built from $`r`$ linear statistics can beat this in the worst case, even with a nonlinear decoder, and for small enough differences actual probability laws already realize the worst case.
- A bound on the weighted average over tests is not a bound for a rare test; (d) gives the per-test version and its price.
- Boundedness alone does not imply useful compression: the dimension it guarantees is no smaller than the trivial one under the mixture measure.

**Intuition.** Two feedback laws can differ in many directions, but the learner and the declared tests turn most of them into small score changes. The response operator measures exactly how small, and the singular values sort the directions by consequence rather than by how different the messages look.

**Role in the framework.**

- It is the explicit-metric companion of [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2): there, the worst hidden difference is twice a uniform approximation error with no density condition; here, the metric is fixed by $`(\rho,\omega)`$ and the optimal interface is computable by a singular value decomposition.
- It defines what a compressed feedback interface must preserve at one checkpoint. Whether the preserved directions stay meaningful as training continues is the closure question of [Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11); estimating them from finite runs is [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12).

**Derivation outline.**

- (a) is Parseval's identity in the output modes; (b) splits the sum at $`r`$ and uses Bessel's inequality on the tail.
- (c) Upper bound: project on the first $`r`$ input modes. Lower bound: any $`r`$ linear statistics vanish on some unit vector $`v`$ of the $`(r+1)`$-dimensional span of $`v_1,\dots,v_{r+1}`$; the inputs $`\pm Bv`$ produce the same statistics, so some decoder output is at distance at least $`\lVert\mathcal R(Bv)\rVert_\omega\ge B\sigma_{r+1}`$ from one of them.
- (d) is the Cauchy–Schwarz inequality applied to the part of $`\tilde g_a`$ outside the leading modes; its $`\omega`$-average is the Hilbert–Schmidt norm of the truncated operator.
- (e) Since the $`\sigma_j`$ decrease, $`(r+1)\sigma_{r+1}^2\le\sum_j\sigma_j^2\le\tfrac14`$ by (4.11).
- (f) Perturb $`\rho`$ along the $`v`$ of (c); positivity holds as long as $`a\lvert v_y\rvert\le2`$, and $`\lVert v\rVert_\rho=1`$ implies $`\lvert v_y\rvert\le1/\sqrt{\rho_{\min}}`$.

**Illustration (analytic).** Take $`J`$ candidates that are point masses on distinct messages. Then $`\rho`$ is uniform on those $`J`$ messages and the density difference of two candidates has $`\lVert h\rVert_\rho^2=2J`$. With $`\varepsilon=0.3`$ and $`J=20`$, (4.16) asks for $`r+1\ge112`$ modes, while 19 statistics (the candidate weights) are already exact. The useful question is the measured spectrum, not the generic bound.

**Scope.**

- The optimality in (c) is over linear statistics with an arbitrary decoder. It does not say that every compression algorithm needs $`r`$ numbers: nonlinear encoders, finite-precision codes and bit complexity are outside it.
- The signed ball of (c) is larger than the set of differences of probability laws; (f) shows the gap closes for small $`B`$, with the support-coverage factor $`2\sqrt{\rho_{\min}}`$.
- Low rank is a property of $`(\rho,\omega)`$, the declared tests and the checkpoint. With a single deployment score and a single continuation the rank is at most 1, which has no scientific content.

**Uses.**

- [Definition 4.8](04-08_response_operator.md#s4-8): the response operator, the bound $`\sum_j\sigma_j^2\le\tfrac14`$ and the mixture measure.
- [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2): the uniform-norm version.

**Used by.**

- [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12): the operator factors through the next predictive state.
- [Proposition 6.16](06-16_information_compilation_radius.md#s6-16): the retained modes define the response interface of the information–compilation radius.
- [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12): estimating the spectrum from a finite catalog.
- [Corollary 8.16](08-16_composed_guarantee.md#s8-16): the truncation term of the composed guarantee.

## Related work

- (c) is the operator form of the Eckart–Young theorem on best low-rank approximation ([Eckart & Young, 1936](#ref-eckart1936approximation)), applied to the response operator; it is a classical instantiation, not a new approximation result.
- The ordering by $`\sigma_j`$ is by consequence gain, not by frequency. The response kernel $`k(y,y')=\mathbb E_{a\sim\omega}\tilde g_a(y)\tilde g_a(y')`$ defines $`\mathcal R^*\mathcal R`$ and may have negative entries, so it is not a graph adjacency. A frequency interpretation needs a declared perturbation kernel $`N`$ that is reversible with respect to $`\rho`$, with $`L=I-N`$ and Dirichlet form $`\tfrac12\mathbb E_{Y\sim\rho,Y'\sim N(Y)}(\phi(Y)-\phi(Y'))^2`$; a common eigenbasis of $`L`$ and $`\mathcal R^*\mathcal R`$ exists only when they commute. Fourier and noise-stability analyses of datamodels use such Boolean structure ([Saunshi et al., 2023](#ref-saunshi2023understanding); [O'Donnell, 2014](#ref-odonnell2014analysis)); the gain spectrum here is a different object.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-odonnell2014analysis"></a>

**Ryan O'Donnell (2014).** Analysis of Boolean Functions. Cambridge University Press. DOI [10.1017/CBO9781139814782](https://doi.org/10.1017/CBO9781139814782).

<a id="ref-saunshi2023understanding"></a>

**Nikunj Saunshi, Arushi Gupta, Mark Braverman, Sanjeev Arora (2023).** Understanding influence functions and datamodels via harmonic analysis. *11th International Conference on Learning Representations (ICLR)*. [openreview.net/forum?id=cxCEOSF99f](https://openreview.net/forum?id=cxCEOSF99f); arXiv [2210.01072](https://arxiv.org/abs/2210.01072).
