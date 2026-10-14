# Proposition 8.15 · Two-fidelity response estimation and its cost condition

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-15_two_fidelity_response_estimation.md) · [Proof](08-15_two_fidelity_response_estimation.md) · [Experiments](../experiments/08-15_two_fidelity_response_estimation.md) · [中文](08-15_two_fidelity_response_estimation_ZH.md)

**What is proved here.** (a) is linearity and independence. (b) is a Cauchy–Schwarz allocation. (c) compares two costs and solves for the correlation thresholds. (d) computes the variance of the nested estimator and optimizes it. (e) counts the minimum full runs per cell. (f) and (g) are Hoeffding bounds.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- One cell (checkpoint, candidate, $`p`$ tests with weights $`\omega`$); $`X\in[0,1]^p`$ the full-continuation output, $`f=\mathbb EX`$; $`Y\in[0,1]^p`$ the output of a frozen cheap estimator.
- Group 0: $`Y_1,\dots,Y_{n_0}`$ i.i.d.; group 1: $`(X'_1,Y'_1),\dots,(X'_{n_1},Y'_{n_1})`$ i.i.d. coupled pairs with the marginals of $`X`$ and $`Y`$; the groups are independent.
- $`v_0`$, $`v_1`$, $`v_H`$ are the weighted total variances of $`Y`$, of $`X-Y`$ under the coupling, and of $`X`$; $`c_0`$, $`c_1`$, $`c_H`$ the costs of a cheap sample, a paired sample and a full sample.

**Statement ([Proposition 8.15](../statements/08-15_two_fidelity_response_estimation.md#s8-15)).** For $`\hat f=\bar Y+\overline{X'-Y'}`$: (a) $`\mathbb E\hat f=f`$, $`\mathbb E\lVert\hat f-f\rVert_\omega^2=v_0/n_0+v_1/n_1`$; (b) the least cost at mean squared error $`\varepsilon^2`$ is $`S^2/\varepsilon^2`$ with $`S=\sqrt{v_0c_0}+\sqrt{v_1c_1}`$, attained at $`n_\ell=(S/\varepsilon^2)\sqrt{v_\ell/c_\ell}`$, and rounding adds at most $`c_0+c_1`$; (c) it is cheaper than full samples alone iff $`\sqrt{v_0c_0}+\sqrt{v_1c_1}<\sqrt{v_Hc_H}`$, with thresholds $`\varrho>0.766`$ ($`c_1=c_H`$) and $`\varrho>0.787`$ ($`c_1=1.1c_H`$) for a scalar test with $`v_0=v_H`$ and $`c_0=c_H/10`$; (d) the nested estimator with the optimal coefficient and allocation has relative cost $`(\sqrt{1-\varrho^2}+\lvert\varrho\rvert\sqrt c)^2`$, below 1 iff $`\lvert\varrho\rvert>2\sqrt c/(1+c)`$; (e) the saving against $`n_H`$ full replicates is at most $`n_H`$-fold; (f) $`\mathrm{Var}\,Z\le\hat v+\tfrac{L^2}2\sqrt{\log(1/\delta_v)/(2k)}`$ with probability at least $`1-\delta_v`$; (g) $`\lvert\hat f-f\rvert\le\sqrt{\log(4N/\delta)/(2n_0)}+2\sqrt{\log(4N/\delta)/(2n_1)}`$ for $`N`$ frozen scalar responses, simultaneously, with probability at least $`1-\delta`$.

*Proof.*

**(a)** $`\mathbb E\bar Y=\mathbb EY`$ and $`\mathbb E\,\overline{X'-Y'}=\mathbb EX-\mathbb EY`$ because each pair has the marginals of $`X`$ and $`Y`$; their sum is $`f`$. Write $`\hat f-f=A_0+A_1`$ with $`A_0=\bar Y-\mathbb EY`$ and $`A_1=\overline{X'-Y'}-\mathbb E(X-Y)`$. The groups are independent and both terms have mean zero, so $`\mathbb E\langle A_0,A_1\rangle_\omega=\langle\mathbb EA_0,\mathbb EA_1\rangle_\omega=0`$. Inside each group the summands are i.i.d., so $`\mathbb E\lVert A_0\rVert_\omega^2=v_0/n_0`$ and $`\mathbb E\lVert A_1\rVert_\omega^2=v_1/n_1`$. Coordinates may be arbitrarily correlated; only the weighted total variance enters.

**(b)** For positive $`n_0,n_1`$, Cauchy–Schwarz gives

```math
\Big(\sum_\ell c_\ell n_\ell\Big)\Big(\sum_\ell\frac{v_\ell}{n_\ell}\Big)\ge\Big(\sum_\ell\sqrt{c_\ell n_\ell}\sqrt{\frac{v_\ell}{n_\ell}}\Big)^2=S^2 .
```

If the error constraint $`\sum_\ell v_\ell/n_\ell\le\varepsilon^2`$ holds, the cost is at least $`S^2/\varepsilon^2`$. At $`n_\ell=(S/\varepsilon^2)\sqrt{v_\ell/c_\ell}`$, $`\sum_\ell v_\ell/n_\ell=(\varepsilon^2/S)\sum_\ell\sqrt{v_\ell c_\ell}=\varepsilon^2`$ and $`\sum_\ell c_\ell n_\ell=(S/\varepsilon^2)\sum_\ell\sqrt{v_\ell c_\ell}=S^2/\varepsilon^2`$. Rounding each $`n_\ell`$ up by less than one sample adds less than $`c_\ell`$ and only lowers the error. If $`v_\ell=0`$ the group still needs one sample, because its mean is unknown.

**(c)** Full samples alone reach mean squared error $`\varepsilon^2`$ at cost $`v_Hc_H/\varepsilon^2`$ under the same continuous allocation. Comparing with (b), the two-fidelity cost is smaller iff $`S^2<v_Hc_H`$, i.e. iff $`\sqrt{v_0c_0}+\sqrt{v_1c_1}<\sqrt{v_Hc_H}`$. For a scalar test with $`v_0=v_H`$ and correlation $`\varrho`$, $`v_1=\mathrm{Var}(X-Y)=2v_H(1-\varrho)`$. Dividing by $`\sqrt{v_Hc_H}`$ and setting $`c_0/c_H=0.1`$, the condition is

```math
\sqrt{0.1}+\sqrt{2(1-\varrho)\,c_1/c_H}<1\iff\varrho>1-\frac{(1-\sqrt{0.1})^2}{2\,c_1/c_H} ,
```

which is $`0.7662`$ for $`c_1=c_H`$ and $`0.7875`$ for $`c_1=1.1c_H`$.

**(d)** Let $`\sigma_H^2=v_H`$, $`\sigma_L^2=v_0`$, and let $`\bar Y_{n_H}`$ be the mean of the first $`n_H`$ cheap outputs, paired with the full outputs, and $`\bar Y_{n_L}`$ the mean of all $`n_L`$. The estimator is unbiased for every $`\alpha`$ because $`\mathbb E(\bar Y_{n_L}-\bar Y_{n_H})=0`$. Writing $`\bar Y_{n_L}-\bar Y_{n_H}=\frac{n_L-n_H}{n_L}(\bar Y_{\rm rest}-\bar Y_{n_H})`$ with $`\bar Y_{\rm rest}`$ the mean of the unpaired outputs,

```math
\mathrm{Var}(\bar Y_{n_L}-\bar Y_{n_H})=\sigma_L^2\Big(\frac1{n_H}-\frac1{n_L}\Big),\qquad\mathrm{Cov}(\bar X_{n_H},\bar Y_{n_L}-\bar Y_{n_H})=\varrho\sigma_H\sigma_L\Big(\frac1{n_L}-\frac1{n_H}\Big),
```

so $`\mathrm{Var}\hat f_\alpha=\sigma_H^2/n_H+(\alpha^2\sigma_L^2-2\alpha\varrho\sigma_H\sigma_L)(1/n_H-1/n_L)`$. It is minimized at $`\alpha=\varrho\sigma_H/\sigma_L`$, where

```math
\mathrm{Var}\hat f_\alpha=\sigma_H^2\Big(\frac{1-\varrho^2}{n_H}+\frac{\varrho^2}{n_L}\Big).
```

With cost $`c_Hn_H+c_Ln_L`$, Cauchy–Schwarz as in (b) gives cost times variance at least $`\sigma_H^2\big(\sqrt{c_H(1-\varrho^2)}+\sqrt{c_L\varrho^2}\big)^2`$, with equality at $`n_L/n_H=\sqrt{c_H\varrho^2/(c_L(1-\varrho^2))}`$. This ratio is at least 1, as the nesting requires, iff $`\varrho^2\ge c/(1+c)`$. Dividing by the full-only value $`\sigma_H^2c_H`$ gives the relative cost $`(\sqrt{1-\varrho^2}+\lvert\varrho\rvert\sqrt c)^2`$. It is below 1 iff $`\sqrt{1-\varrho^2}<1-\lvert\varrho\rvert\sqrt c`$, i.e. (both sides positive) iff $`1-\varrho^2<1-2\lvert\varrho\rvert\sqrt c+\varrho^2c`$, i.e. iff $`\lvert\varrho\rvert\big((1+c)\lvert\varrho\rvert-2\sqrt c\big)>0`$, i.e. iff $`\lvert\varrho\rvert>2\sqrt c/(1+c)`$. Since $`(2\sqrt c/(1+c))^2=4c/(1+c)^2\ge c/(1+c)`$ for $`c\le3`$, the optimal ratio is feasible whenever there is a saving. At $`c=0.1`$ the threshold is $`2\sqrt{0.1}/1.1=0.5750`$. If $`\alpha`$ is estimated from the same samples, $`\hat f_{\hat\alpha}`$ is no longer exactly unbiased; freezing $`\alpha`$ from an independent pilot restores unbiasedness.

**(e)** In (a) the constraint $`n_1\ge1`$ forces cost at least $`c_1\ge c_H`$; in (d), $`n_H\ge1`$ forces cost at least $`c_H`$. With $`n_H`$ full replicates per cell the full-only cost is $`n_Hc_H`$, so the ratio is at least $`1/n_H`$.

**(f)** For independent copies, $`\mathbb E(Z_{2i}-Z_{2i-1})^2/2=\mathrm{Var}\,Z`$, and each term lies in $`[0,L^2/2]`$. Hoeffding's inequality ([Hoeffding, 1963](#ref-hoeffding1963probability)) for $`k`$ i.i.d. terms with range $`L^2/2`$ gives $`\Pr\big(\mathrm{Var}\,Z>\hat v+s\big)\le\exp(-2ks^2/(L^2/2)^2)`$; setting the right side to $`\delta_v`$ gives the bound. Applied to the weighted squared norm of a vector, the same argument bounds the weighted total variance.

**(g)** For one response, $`\bar Y`$ averages $`n_0`$ variables in $`[0,1]`$ and $`\overline{X'-Y'}`$ averages $`n_1`$ variables in $`[-1,1]`$. Two-sided Hoeffding bounds at level $`\delta'`$ give radii $`\sqrt{\log(2/\delta')/(2n_0)}`$ and $`2\sqrt{\log(2/\delta')/(2n_1)}`$. There are $`2N`$ such events; with $`\delta'=\delta/(2N)`$, $`\log(2/\delta')=\log(4N/\delta)`$, and the union bound gives probability at least $`1-\delta`$ that all hold. On this event $`\lvert\hat f-f\rvert\le\lvert\bar Y-\mathbb EY\rvert+\lvert\overline{X'-Y'}-\mathbb E(X-Y)\rvert\le\eta`$. All three statements (a), (f), (g) use that the proxy, the coefficient and the allocation are fixed before the samples are drawn. $`\square`$

## Related work

- (d) is the two-model case of optimal multifidelity Monte Carlo ([Peherstorfer, Willcox & Gunzburger, 2016](#ref-peherstorfer2016optimal)); (a)–(c) are the two-level form of multilevel Monte Carlo ([Giles, 2015](#ref-giles2015multilevel)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-giles2015multilevel"></a>

**Michael B. Giles (2015).** Multilevel Monte Carlo methods. *Acta Numerica* 24, 259–328. DOI [10.1017/S096249291500001X](https://doi.org/10.1017/S096249291500001X).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-peherstorfer2016optimal"></a>

**Benjamin Peherstorfer, Karen Willcox, Max Gunzburger (2016).** Optimal model management for multifidelity Monte Carlo estimation. *SIAM Journal on Scientific Computing* 38, no. 5 (section A, Methods and Algorithms for Scientific Computing). DOI [10.1137/15M1046472](https://doi.org/10.1137/15M1046472).
