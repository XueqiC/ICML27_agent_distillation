# Proposition 8.15 · Two-fidelity response estimation and its cost condition

*Statement*

[← Framework](../README.md) · [Statement](08-15_two_fidelity_response_estimation.md) · [Proof](../proofs/08-15_two_fidelity_response_estimation.md) · [Experiments](../experiments/08-15_two_fidelity_response_estimation.md) · [中文](08-15_two_fidelity_response_estimation_ZH.md)

**In one paragraph.** Each cell of a response catalog needs full continuations, which are expensive. A cheap predictor of the full outcome, such as a short continuation, does not have to be unbiased to help: a few full runs paired with the cheap predictor estimate its bias, and the corrected estimator is exactly unbiased. This is the classical control-variate and multifidelity idea, applied to responses. The proposition gives the variance, the cost-optimal allocation and the exact condition under which it saves computation. Compared like for like at a cheap run costing a tenth: when every cheap output is charged, the saving requires a correlation of about 0.787 between the cheap and the full outcome with a fixed coefficient and 0.575 with the optimal coefficient of nested multifidelity estimation; when the cheap output of each paired sample is reused, the thresholds are 0.766 and about 0.316. Two limits are structural: the method reduces the replicates *within* a cell, not the number of cells, and no short-versus-full correlation has been measured on the real model.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- One cell is fixed: a checkpoint, a candidate feedback law and a vector of $`p`$ tests with weights $`\omega`$; $`\lVert u\rVert_\omega^2=\sum_a\omega_au_a^2`$.
- $`X\in[0,1]^p`$ is the output of one *full* continuation; the target is $`f=\mathbb EX`$, the cell's response vector.
- $`Y\in[0,1]^p`$ is the output of a *cheap* estimator that is frozen before the data below are drawn: a short continuation, a clipped proxy, or a predictor that uses the learner state. $`Y`$ may be biased.
- *Two independent groups.* Group 0 has $`n_0`$ i.i.d. cheap outputs $`Y_i`$. Group 1 has $`n_1`$ i.i.d. pairs $`(X'_i,Y'_i)`$, where $`X'_i`$ has the law of $`X`$, $`Y'_i`$ has the law of $`Y`$, and the two are coupled (for example, the short run is the prefix of the full run, or they share random numbers).
- *Variances.* $`v_0=\mathbb E\lVert Y-\mathbb EY\rVert_\omega^2`$, $`v_1=\mathbb E\lVert(X-Y)-\mathbb E(X-Y)\rVert_\omega^2`$ (under the coupling of group 1), $`v_H=\mathbb E\lVert X-\mathbb EX\rVert_\omega^2`$.
- *Costs.* $`c_0`$ per cheap sample; $`c_1`$ per paired sample, including its cheap part (after any reuse of computation); $`c_H`$ per full sample alone.

<a id="s8-15"></a>

**Proposition 8.15 (Two-fidelity response estimation and its cost condition).** In this setting let

```math
\hat f=\frac1{n_0}\sum_{i=1}^{n_0}Y_i+\frac1{n_1}\sum_{i=1}^{n_1}\big(X'_i-Y'_i\big).
```

**(a) Unbiased, with additive variance.** $`\mathbb E\hat f=f`$ and $`\mathbb E\lVert\hat f-f\rVert_\omega^2=v_0/n_0+v_1/n_1`$.

**(b) Cost-optimal allocation.** Ignoring integrality, the least cost $`n_0c_0+n_1c_1`$ subject to $`\mathbb E\lVert\hat f-f\rVert_\omega^2\le\varepsilon^2`$ is

```math
C_{\rm MF}=\frac{S^2}{\varepsilon^2},\qquad S=\sqrt{v_0c_0}+\sqrt{v_1c_1},\qquad n_\ell=\frac S{\varepsilon^2}\sqrt{\frac{v_\ell}{c_\ell}} .
```

Rounding each $`n_\ell`$ up to $`\max\lbrace1,\lceil n_\ell\rceil\rbrace`$ adds at most $`c_0+c_1`$.

**(c) When it saves computation.** Against $`n_H`$ full samples alone at the same mean squared error and the same continuous allocation, the two-fidelity estimator is cheaper if and only if

```math
\sqrt{v_0c_0}+\sqrt{v_1c_1}<\sqrt{v_Hc_H} .
```

For a scalar test with $`v_0=v_H`$ and correlation $`\varrho`$ between $`X`$ and $`Y`$, $`v_1=2v_H(1-\varrho)`$. With $`c_0=c_H/10`$ the condition is $`\varrho>0.766`$ when the cheap part of a pair is reused ($`c_1=c_H`$) and $`\varrho>0.787`$ when it is not ($`c_1=1.1c_H`$).

**(d) Nested estimator with the optimal coefficient.** For a scalar test, let the first $`n_H`$ of $`n_L\ge n_H`$ cheap outputs be paired with the $`n_H`$ full outputs, and $`\hat f_\alpha=\bar X_{n_H}+\alpha(\bar Y_{n_L}-\bar Y_{n_H})`$, with costs $`c_H`$ per full and $`c_L=c\,c_H`$ per cheap output. With $`\alpha=\varrho\sqrt{v_H/v_0}`$ and the optimal ratio $`n_L/n_H=\sqrt{\varrho^2/(c(1-\varrho^2))}`$ (feasible when $`\varrho^2\ge c/(1+c)`$), the cost relative to full samples alone at equal mean squared error is

```math
\big(\sqrt{1-\varrho^2}+\lvert\varrho\rvert\sqrt c\big)^2 ,
```

which is below 1 if and only if $`\lvert\varrho\rvert>2\sqrt c/(1+c)`$: 0.575 at $`c=0.1`$. This cost model charges every cheap output. If the cheap output of each paired sample is reused, so that the cost is $`c_Hn_H+c_L(n_L-n_H)`$, the threshold is lower, about $`\sqrt c\approx0.316`$ at $`c=0.1`$ (numerical, [Check 8.22](../experiments/08-15_two_fidelity_response_estimation.md#c8-22)); compare it with the fixed-coefficient threshold 0.766 of the same reuse model, and 0.575 with 0.787. The optimal $`\alpha`$ is unknown in practice; estimating it from the same samples introduces a bias of order $`1/n_H`$, so an exactly unbiased version freezes $`\alpha`$ from a pilot.

**(e) Replicates, not cells.** Every cell needs at least one full continuation ($`n_1\ge1`$, respectively $`n_H\ge1`$), so relative to $`n_H`$ full replicates per cell the saving is at most $`n_H`$-fold. Reducing the *number of cells* that need a full continuation requires sharing the correction $`\mathbb E(X-Y)`$ across cells, which is a transfer assumption of the kind [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12) does not provide.

**(f) Unknown variances.** For a scalar $`Z`$ with known range length $`L`$ and $`k`$ independent pairs $`(Z_{2i-1},Z_{2i})`$ from a pilot, $`\hat v=\frac1k\sum_i(Z_{2i}-Z_{2i-1})^2/2`$ satisfies, with probability at least $`1-\delta_v`$,

```math
\mathrm{Var}\,Z\le\hat v+\frac{L^2}2\sqrt{\frac{\log(1/\delta_v)}{2k}} .
```

Certifying (c) needs upper bounds on its left side, a lower bound on its right side and joint coverage; comparing point estimates does not certify a saving. The pilot is charged to the budget.

**(g) A strict radius without variance knowledge.** For $`N`$ frozen scalar responses, with probability at least $`1-\delta`$, simultaneously,

```math
\lvert\hat f-f\rvert\le\eta=\sqrt{\frac{\log(4N/\delta)}{2n_0}}+2\sqrt{\frac{\log(4N/\delta)}{2n_1}} .
```

Choosing the proxy, the coefficient or the allocation from the same samples and then treating them as frozen voids (a), (f) and (g).

**What it says.**

- A biased but correlated cheap estimate of a full continuation, corrected by a few paired full runs, gives an unbiased response estimate, with an explicit variance and an explicit cost-optimal allocation.
- It saves computation exactly when the cost-weighted standard deviations satisfy (c): this needs strong pairing correlation at a cost ratio of one tenth: like for like, 0.787 with a fixed coefficient and 0.575 with the optimal one when every cheap output is charged, and 0.766 and about 0.316 when the cheap output of each pair is reused.
- It cuts replicates inside a cell, never below one full continuation per cell; the number of cells is untouched.

**Intuition.** The cheap estimate carries most of the sampling; the full runs only estimate how wrong it is. If the two move together, the difference has little variance and few full runs suffice. If they do not, the correction is as noisy as the full outcome itself and nothing is saved.

**Role in the framework.**

- It is the one estimator in the framework whose purpose is to make response catalogs cheaper, with a condition that can be checked before committing a budget.
- With the radius of (g) in place of the single-fidelity $`\varepsilon_n`$, the spectral bounds of [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12) hold unchanged, and the anchor error $`\epsilon_j`$ of [Corollary 8.16](08-16_composed_guarantee.md#s8-16) can be obtained this way.
- Its premise, a strong short-versus-full correlation on the real learner, is untested: that is an open problem of [Remark 8.17](08-17_open_problems.md#s8-17).

**Comments on the statement.**

- **Low rank does not help here.** Low rank describes the mean responses across cells. If the cheap prediction is constant for a fixed message, it does not reduce the conditional run-to-run variance of the full outcome at all.
- **Amortization.** A fitted cheap predictor of cost $`C_{\rm fit}`$ used for $`M_{\rm use}`$ decisions costs $`C_{\rm fit}/M_{\rm use}`$ per decision; report the actual number of uses, not an infinite amortization.
- **Synthetic illustration.** At $`v_H=v_0=0.25`$, $`v_1=0.0025`$ (correlation about 0.995), $`c_0=0.01`$, $`c_1=1.01`$, $`c_H=1`$, the cost ratio is 0.0402 before pilot costs ([Check 8.22](../experiments/08-15_two_fidelity_response_estimation.md#c8-22)). This is an illustration of the formula, not a measured speed-up.

**Derivation outline.**

- (a): $`\mathbb EX=\mathbb EY+\mathbb E(X-Y)`$; the two group means are independent, so the cross term has mean zero.
- (b): Cauchy–Schwarz, $`(\sum_\ell c_\ell n_\ell)(\sum_\ell v_\ell/n_\ell)\ge(\sum_\ell\sqrt{v_\ell c_\ell})^2`$, with equality at $`n_\ell\propto\sqrt{v_\ell/c_\ell}`$.
- (c): compare $`C_{\rm MF}`$ with $`v_Hc_H/\varepsilon^2`$; for the thresholds solve $`\sqrt c+\sqrt{2(1-\varrho)c_1/c_H}=1`$.
- (d): variance of the nested estimator, its minimum over $`\alpha`$, and Cauchy–Schwarz over $`(n_H,n_L)`$.
- (e): $`C_{\rm MF}\ge c_1\ge c_H`$ against $`n_Hc_H`$.
- (f), (g): Hoeffding bounds for the half squared pair differences and for the two group means, with union bounds.

**Uses.**

- [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12) — the catalog whose cells are estimated

**Used by.**

- [Corollary 8.16](08-16_composed_guarantee.md#s8-16) — anchor estimation error
- [Remark 8.17](08-17_open_problems.md#s8-17) — short-versus-full correlation on the real learner

## Related work

- Optimal model management for multifidelity Monte Carlo ([Peherstorfer, Willcox & Gunzburger, 2016](#ref-peherstorfer2016optimal)) and multilevel Monte Carlo ([Giles, 2015](#ref-giles2015multilevel)) give the nested estimator, its optimal coefficient and allocation, and the cost ratio of (d). Control variates are classical in simulation ([Glasserman, 2004](#ref-glasserman2004monte)). What is specific here is only the object: the cell response of a continued-training catalog, with a short continuation as the low-fidelity model.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-giles2015multilevel"></a>

**Michael B. Giles (2015).** Multilevel Monte Carlo methods. *Acta Numerica* 24, 259–328. DOI [10.1017/S096249291500001X](https://doi.org/10.1017/S096249291500001X).

<a id="ref-glasserman2004monte"></a>

**Paul Glasserman (2004).** Monte Carlo Methods in Financial Engineering. Springer, New York (Stochastic Modelling and Applied Probability 53). DOI [10.1007/978-0-387-21617-1](https://doi.org/10.1007/978-0-387-21617-1).

<a id="ref-peherstorfer2016optimal"></a>

**Benjamin Peherstorfer, Karen Willcox, Max Gunzburger (2016).** Optimal model management for multifidelity Monte Carlo estimation. *SIAM Journal on Scientific Computing* 38, no. 5 (section A, Methods and Algorithms for Scientific Computing). DOI [10.1137/15M1046472](https://doi.org/10.1137/15M1046472).
