# Proposition 6.16 · Information–compilation radius

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-16_information_compilation_radius.md) · [Proof](06-16_information_compilation_radius.md) · [Experiments](../experiments/06-16_information_compilation_radius.md) · [中文](06-16_information_compilation_radius_ZH.md)

**What is proved here.** Existence and convexity of the robust radius, its two lower bounds and the characterization of a zero radius, the guarantee for the radius-optimal compilation, and the computational statements about the inner supremum. The three remedies of part (e) are direct readings of (b) and (c) and are justified at the end.

**Setting used on this page.**

**Interface.** Finitely many declared tests $`a`$ with weights $`\omega_a>0`$; $`g_a(y)=k_a+c_a^\top\phi(y)+e_a(y)`$ with $`\phi(y)\in\mathbb R^r`$, $`\phi\in L^2(\rho)`$, $`C\in\mathbb R^{p\times r}`$ with rows $`c_a^\top`$; $`\mu_P=\mathbb E_P\phi`$; $`\lVert x\rVert_\omega^2=\sum_a\omega_ax_a^2`$.

**Operators.** $`(\mathcal Rh)(a)=\langle\tilde g_a,h\rangle_\rho`$ and $`(\mathcal Eh)(a)=\langle\tilde e_a,h\rangle_\rho`$ on $`L^2_0(\rho)`$, with $`\tilde g_a=g_a-\mathbb E_\rho g_a`$, $`\tilde e_a=e_a-\mathbb E_\rho e_a`$, and $`\lVert\mathcal E\rVert_{\rm op}\le\epsilon_{\rm tail}`$ ([Definition 4.8](../statements/04-08_response_operator.md#s4-8)).

**Uncertainty and legal sets.** $`\mathcal M\subset\mathbb R^r`$ nonempty compact with $`\Pr(\mu_P\in\mathcal M)\ge1-\delta`$; $`\mathcal K=\lbrace C\mu_Q:Q\in\mathcal Q_{\rm legal}\rbrace`$ nonempty compact; $`\Gamma(q)=\sup_{\mu\in\mathcal M}\lVert C\mu-q\rVert_\omega`$, $`\mathfrak r=\min_{q\in\mathcal K}\Gamma(q)`$.

**Statement ([Proposition 6.16](../statements/06-16_information_compilation_radius.md#s6-16)).** (a) The minimum exists; $`\Gamma`$ is convex and 1-Lipschitz. (b) $`\mathfrak r\ge\max\lbrace\tfrac12\mathrm{diam}_\omega(C\mathcal M),\sup_{\mu\in\mathcal M}\mathrm{dist}_\omega(C\mu,\mathcal K)\rbrace`$. (c) $`\mathfrak r=0`$ if and only if $`C\mathcal M`$ is a single point of $`\mathcal K`$. (d) For a minimizer $`q^*=C\mu_{Q^*}`$ and $`\lVert dP/d\rho-dQ^*/d\rho\rVert_\rho\le B`$, with probability at least $`1-\delta`$, $`\lVert\mathcal R(dP/d\rho-dQ^*/d\rho)\rVert_\omega\le\mathfrak r+B\epsilon_{\rm tail}`$; an optimization error $`\eta_{\rm opt}`$ adds $`\eta_{\rm opt}`$. (e) The three remedies. (f) The inner supremum is attained at an extreme point of $`\mathcal M`$; finite scenario sets, single ellipsoids and small boxes are tractable, general boxes are hard.

**Results used in the proof.**

- [Definition 4.8](../statements/04-08_response_operator.md#s4-8): the response operator as the exact score difference.

**Classical tools.** Compactness and continuity; the triangle inequality; maximization of convex functions over polytopes; the S-lemma ([Pólik & Terlaky, 2007](#ref-polik2007survey)).

**Proof.**

**(a)** For fixed $`\mu`$, $`q\mapsto\lVert C\mu-q\rVert_\omega`$ is convex and 1-Lipschitz in $`\lVert\cdot\rVert_\omega`$. A pointwise supremum of convex 1-Lipschitz functions is convex and 1-Lipschitz wherever it is finite, and $`\Gamma(q)\le\sup_{\mu\in\mathcal M}\lVert C\mu\rVert_\omega+\lVert q\rVert_\omega<\infty`$ because $`\mathcal M`$ is compact. A continuous function attains its minimum on the nonempty compact set $`\mathcal K`$. If $`\mathcal K`$ is convex, minimizing the convex function $`\Gamma`$ over it is a convex program.

**(b)** Fix $`q\in\mathcal K`$. For $`\mu,\mu'\in\mathcal M`$, $`\lVert C\mu-C\mu'\rVert_\omega\le\lVert C\mu-q\rVert_\omega+\lVert C\mu'-q\rVert_\omega\le2\Gamma(q)`$; taking the supremum over the pair gives $`\Gamma(q)\ge\tfrac12\mathrm{diam}_\omega(C\mathcal M)`$. Also $`\lVert C\mu-q\rVert_\omega\ge\mathrm{dist}_\omega(C\mu,\mathcal K)`$ for each $`\mu`$ because $`q\in\mathcal K`$, so $`\Gamma(q)\ge\sup_\mu\mathrm{dist}_\omega(C\mu,\mathcal K)`$. Both bounds hold for every $`q\in\mathcal K`$, hence for the minimum.

**(c)** If $`C\mathcal M=\lbrace x\rbrace`$ with $`x\in\mathcal K`$, then $`\Gamma(x)=0`$. Conversely, if $`\mathfrak r=0`$, the minimizer $`q^*\in\mathcal K`$ satisfies $`\lVert C\mu-q^*\rVert_\omega=0`$ for every $`\mu\in\mathcal M`$; since $`\omega_a>0`$ for every test, $`\lVert\cdot\rVert_\omega`$ is a norm, so $`C\mu=q^*`$ for every $`\mu`$.

**(d)** Let $`h=dP/d\rho-dQ^*/d\rho\in L^2_0(\rho)`$. Because $`\mathbb E_\rho h=0`$, constants drop out of every inner product with $`h`$, and

```math
(\mathcal Rh)(a)=\langle\tilde g_a,h\rangle_\rho=c_a^\top\langle\phi,h\rangle_\rho+\langle\tilde e_a,h\rangle_\rho=c_a^\top(\mu_P-\mu_{Q^*})+(\mathcal Eh)(a),
```

since $`\langle\phi,h\rangle_\rho=\mathbb E_P\phi-\mathbb E_{Q^*}\phi`$. Hence $`\mathcal Rh=(C\mu_P-q^*)+\mathcal Eh`$. On the event $`\mu_P\in\mathcal M`$, which has probability at least $`1-\delta`$, $`\lVert C\mu_P-q^*\rVert_\omega\le\Gamma(q^*)=\mathfrak r`$. Always $`\lVert\mathcal Eh\rVert_\omega\le\epsilon_{\rm tail}\lVert h\rVert_\rho\le\epsilon_{\rm tail}B`$. The triangle inequality gives the bound. With an approximate minimizer $`\hat q`$, $`\Gamma(\hat q)\le\mathfrak r+\eta_{\rm opt}`$ replaces $`\mathfrak r`$.

**(e)** If $`\mathcal M`$ varies only along $`\ker C`$, then $`C\mathcal M`$ is a point, and by (b)–(c) the radius is the distance of that point to $`\mathcal K`$; shrinking $`\mathcal M`$ along $`\ker C`$ leaves $`C\mathcal M`$, hence $`\Gamma`$ and $`\mathfrak r`$, unchanged. If $`C\mathcal M`$ has diameter $`D`$, (b) gives $`\mathfrak r\ge D/2`$ for every legal set, so only a smaller $`C\mathcal M`$, i.e. evidence that separates the consequence directions, lowers this part. If $`C\mathcal M=\lbrace x\rbrace`$ with $`x\notin\mathcal K`$, then $`\mathfrak r=\mathrm{dist}_\omega(x,\mathcal K)>0`$ by compactness of $`\mathcal K`$; more evidence cannot shrink a point, and only a larger $`\mathcal K`$ lowers the radius.

**(f)** For fixed $`q`$, $`\mu\mapsto\lVert C\mu-q\rVert_\omega`$ is convex, and a convex function on a compact convex set attains its maximum at an extreme point, so the supremum over $`\mathcal M`$ equals the supremum over the extreme points of its convex hull.

- *Finite scenarios* $`\mathcal M=\lbrace\mu_1,\dots,\mu_s\rbrace`$: minimizing $`t`$ subject to $`\lVert C\mu_j-q\rVert_\omega\le t`$ for all $`j`$ and $`q\in\mathcal K`$ is a second-order cone program when $`\mathcal K`$ is a polytope.
- *A single ellipsoid* $`\mathcal M=\lbrace\mu:(\mu-m_0)^\top S^{-1}(\mu-m_0)\le1\rbrace`$: the inner problem maximizes a quadratic subject to one quadratic constraint. By the S-lemma, $`\Gamma(q)^2\le t`$ holds if and only if there is $`\lambda\ge0`$ with $`t-\lVert C\mu-q\rVert_\omega^2-\lambda\big(1-(\mu-m_0)^\top S^{-1}(\mu-m_0)\big)\ge0`$ for every $`\mu`$, a linear matrix inequality in $`(t,\lambda)`$ after a Schur complement in $`q`$; the outer problem is then a semidefinite program.
- *A box*: the maximum is attained at one of its $`2^r`$ vertices, which can be enumerated for small $`r`$. In general the problem is hard: for a graph with Laplacian $`L\succeq0`$, the maximum cut equals $`\max_{x\in\lbrace-1,1\rbrace^n}x^\top Lx/4`$, which is the maximum of the convex quadratic $`x^\top Lx/4`$ over the box $`[-1,1]^n`$ because that maximum is attained at a vertex. Taking $`r=n`$, $`C`$ with $`C^\top\mathrm{diag}(\omega)C=L/4`$, $`\mathcal K=\lbrace0\rbrace`$ and $`\mathcal M=[-1,1]^n`$ makes $`\Gamma(0)^2`$ exactly the maximum-cut value. $`\square`$

**Comments.**

- The bound in (d) is per checkpoint and in the weighted norm; dividing by $`\sqrt{\omega_a}`$ turns it into a bound for one test $`a`$, at the price that rare tests receive large factors.
- When $`C`$ itself is estimated, the same argument applies to the joint uncertainty set of $`(C,\mu)`$, but $`\Gamma`$ is then no longer a function of $`C\mu`$ alone and the convexity in (a) must be rechecked.

## Related work

- Minimizing a worst-case distance over a feasible set is a robust approximation, or Chebyshev-centre, problem; the S-lemma and its use for one quadratic constraint are surveyed in [Pólik & Terlaky, 2007](#ref-polik2007survey).
- The diameter bound in (b) is the deterministic core of the two-point method of [Le Cam, 1973](#ref-lecam1973convergence): two targets that the evidence cannot exclude force an error of half their distance on one of them.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-polik2007survey"></a>

**Imre Pólik, Tamás Terlaky (2007).** A survey of the S-lemma. *SIAM Review* 49, no. 3, 371–418. DOI [10.1137/S003614450444614X](https://doi.org/10.1137/S003614450444614X).
