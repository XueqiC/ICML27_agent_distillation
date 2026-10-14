# Proposition 6.14 · Convex realizability over a finite legal pool

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-14_convex_realizability_legal_pool.md) · [Proof](06-14_convex_realizability_legal_pool.md) · [Experiments](../experiments/06-14_convex_realizability_legal_pool.md) · [中文](06-14_convex_realizability_legal_pool_ZH.md)

**What is proved here.** All five parts. (a) and (c) are convex geometry of a finite point set, (b) is the vertex argument of linear programming, and (d)–(e) are Cauchy–Schwarz and a summation over the support.

**Setting used on this page.**

- **Legal pool.** A finite set $`\lbrace y_1,\dots,y_m\rbrace`$ of verified legal messages. A *legal mixture* is a probability vector $`q\in\Delta_m`$ ($`q_i\ge0`$, $`\sum_iq_i=1`$).
- **Retained features.** $`\phi(y)\in\mathbb R^r`$, $`\Phi=[\phi(y_1),\dots,\phi(y_m)]\in\mathbb R^{r\times m}`$; a mixture $`q`$ has feature mean $`\Phi q`$.
- **Target.** $`\mu_*\in\mathbb R^r`$.
- **Reference measure.** A probability vector $`\rho`$ with every $`\rho_i>0`$; $`dQ/d\rho=(q_i/\rho_i)_i`$, $`\lVert f\rVert_\rho^2=\sum_i\rho_if_i^2`$; for a target law $`P`$ on the pool, $`B=\lVert dP/d\rho-dQ/d\rho\rVert_\rho`$ ([Theorem 4.9](../statements/04-09_spectral_sufficiency.md#s4-9)). For $`S\subseteq\lbrace1,\dots,m\rbrace`$, $`\rho(S)=\sum_{i\in S}\rho_i`$.

**Statement ([Proposition 6.14](../statements/06-14_convex_realizability_legal_pool.md#s6-14)).** In this setting: (a) a legal mixture with $`\Phi q=\mu_*`$ exists if and only if $`\mu_*\in\mathrm{conv}\lbrace\phi(y_i)\rbrace`$; (b) if one exists, one exists with at most $`r+1`$ nonzero weights, and every vertex of $`\lbrace q\ge0:\Phi q=\mu_*,\mathbf 1^\top q=1\rbrace`$ has this property; (c) if $`\mu_*\notin\mathrm{conv}\lbrace\phi(y_i)\rbrace`$, then for every norm $`\min_{q\in\Delta_m}\lVert\Phi q-\mu_*\rVert=\mathrm{dist}(\mu_*,\mathrm{conv}\lbrace\phi(y_i)\rbrace)>0`$, attained; (d) every mixture with support $`S`$ has $`\lVert dQ/d\rho\rVert_\rho^2\ge1/\rho(S)`$ and $`B\ge\rho(S)^{-1/2}-\lVert dP/d\rho\rVert_\rho`$; (e) density caps $`q_i\le\kappa\rho_i`$ or $`\chi^2`$ constraints $`\sum_iq_i^2/\rho_i\le1+\beta^2`$ keep the feasible set convex and compact, and under the cap $`\rho(S)\ge1/\kappa`$.

**Results used in the proof.** None beyond the definitions of the setting.

**Classical tools.** Carathéodory's theorem ([Rockafellar, 1970](#ref-rockafellar1970convex)) in its linear-programming form; the Cauchy–Schwarz inequality.

**Proof.**

**(a)** The convex hull of finitely many points is the set of their convex combinations. A legal mixture is exactly a vector of convex-combination weights, and its feature mean is the corresponding combination $`\Phi q=\sum_iq_i\phi(y_i)`$. So the set of attainable feature means is $`\mathrm{conv}\lbrace\phi(y_i)\rbrace`$.

**(b)** Let $`\mathcal F=\lbrace q\in\mathbb R^m:q\ge0,\ \Phi q=\mu_*,\ \mathbf 1^\top q=1\rbrace`$, nonempty by assumption. It is a polyhedron contained in the simplex, hence a bounded polytope, and a nonempty polytope has at least one vertex. Let $`q`$ be a vertex and $`S`$ its support. Suppose the columns $`a_i=(\phi(y_i),1)\in\mathbb R^{r+1}`$, $`i\in S`$, were linearly dependent: $`\sum_{i\in S}w_ia_i=0`$ with $`w\ne0`$. Extend $`w`$ by zero outside $`S`$. Then $`\Phi w=0`$ and $`\mathbf 1^\top w=0`$, and for $`\varepsilon=\min_{i\in S}q_i/\max_i\lvert w_i\rvert>0`$ both $`q+\varepsilon w`$ and $`q-\varepsilon w`$ lie in $`\mathcal F`$. Then $`q`$ is the midpoint of two distinct points of $`\mathcal F`$, which contradicts its being a vertex. So the columns indexed by $`S`$ are linearly independent vectors of $`\mathbb R^{r+1}`$, and $`\lvert S\rvert\le r+1`$.

**(c)** $`\mathrm{conv}\lbrace\phi(y_i)\rbrace`$ is the image of the compact simplex under the continuous linear map $`q\mapsto\Phi q`$, hence compact and convex. The function $`x\mapsto\lVert x-\mu_*\rVert`$ is continuous, so it attains its minimum on this set, and the minimum is positive because $`\mu_*`$ lies outside the closed set. By (a), minimizing over $`q\in\Delta_m`$ is the same as minimizing over the hull.

**(d)** Let $`S`$ be the support of $`q`$. By Cauchy–Schwarz,

```math
1=\sum_{i\in S}q_i=\sum_{i\in S}\frac{q_i}{\sqrt{\rho_i}}\sqrt{\rho_i}\le\Big(\sum_{i\in S}\frac{q_i^2}{\rho_i}\Big)^{1/2}\Big(\sum_{i\in S}\rho_i\Big)^{1/2}=\Big\lVert\frac{dQ}{d\rho}\Big\rVert_\rho\,\rho(S)^{1/2}.
```

Squaring gives the first inequality. The triangle inequality in $`L^2(\rho)`$ gives $`B\ge\lVert dQ/d\rho\rVert_\rho-\lVert dP/d\rho\rVert_\rho\ge\rho(S)^{-1/2}-\lVert dP/d\rho\rVert_\rho`$. For uniform $`\rho`$ and $`\lvert S\rvert=r+1`$, $`\rho(S)=(r+1)/m`$.

**(e)** The sets $`\lbrace q:q_i\le\kappa\rho_i\ \forall i\rbrace`$ and $`\lbrace q:\sum_iq_i^2/\rho_i\le1+\beta^2\rbrace`$ are closed and convex, so their intersection with the compact convex feasible set of (b) is compact and convex. Linear constraints keep a linear program; the $`\chi^2`$ constraint is a convex quadratic constraint. Under the cap, $`1=\sum_{i\in S}q_i\le\kappa\sum_{i\in S}\rho_i=\kappa\,\rho(S)`$. For uniform $`\rho`$ this reads $`\lvert S\rvert\ge m/\kappa`$. $`\square`$

**Comments.**

- The $`\chi^2`$ constraint bounds $`\lVert dQ/d\rho\rVert_\rho^2\le1+\beta^2`$ directly, so with it the density bound in [Theorem 4.9](../statements/04-09_spectral_sufficiency.md#s4-9) is controlled by $`\beta`$ and $`\lVert dP/d\rho\rVert_\rho`$ through the triangle inequality.
- Feasibility in (a) is decided by one linear program; an infeasibility certificate is a separating hyperplane $`(w,w_0)`$ with $`w^\top\phi(y_i)\le w_0<w^\top\mu_*`$ for every $`i`$ (Farkas' lemma).

## Related work

- Carathéodory's theorem and the separation theorem used in the comments are classical convex analysis ([Rockafellar, 1970](#ref-rockafellar1970convex)).
- Selecting training examples that steer a known learner is the subject of machine teaching ([Zhu, 2015](#ref-zhu2015machine)); here the selection is a mixture over a fixed verified pool with an explicit density price.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-rockafellar1970convex"></a>

**R. Tyrrell Rockafellar (1970).** Convex Analysis. Princeton University Press, Princeton, NJ (Princeton Mathematical Series 28). DOI [10.1515/9781400873173](https://doi.org/10.1515/9781400873173).

<a id="ref-zhu2015machine"></a>

**Xiaojin Zhu (2015).** Machine teaching: An inverse problem to machine learning and an approach toward optimal education. *Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI)* 29, no. 1. DOI [10.1609/aaai.v29i1.9761](https://doi.org/10.1609/aaai.v29i1.9761).
