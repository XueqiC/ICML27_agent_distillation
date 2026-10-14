# Proposition 6.16 · Information–compilation radius

*Statement*

[← Framework](../README.md) · [Statement](06-16_information_compilation_radius.md) · [Proof](../proofs/06-16_information_compilation_radius.md) · [Experiments](../experiments/06-16_information_compilation_radius.md) · [中文](06-16_information_compilation_radius_ZH.md)

**In one paragraph.** Two different things can stop a distiller from reproducing the consequences of the oracle feedback. The evidence may not yet tell which consequences the oracle would have, or the evidence may be sufficient while no legal mixture of the available messages produces them. Theorem A measures the first; [Proposition 6.14](06-14_convex_realizability_legal_pool.md#s6-14) measures the second. This proposition puts both into one number, a robust approximation radius computed on the response interface: how far the best legal feedback can be from every target that the evidence still allows. The radius is zero only when both gaps are closed, and its two lower bounds say which gap is open, so the next unit of budget goes either to acquisition or to enlarging the legal pool.

**Setting used on this page.**

**A calibrated response interface.** Fix a checkpoint and finitely many declared tests $`a`$ with weights $`\omega_a>0`$ (tests of weight zero are removed first). A frozen feature map $`\phi(y)\in\mathbb R^r`$ describes each test's response as

```math
g_a(y)=k_a+c_a^\top\phi(y)+e_a(y),
```

with known coefficients $`c_a`$ collected in $`C\in\mathbb R^{p\times r}`$ (row $`a`$ is $`c_a^\top`$) and a residual $`e_a`$. For a law $`P`$, $`\mu_P=\mathbb E_P\phi(Y)`$. On $`\mathbb R^p`$, $`\lVert x\rVert_\omega^2=\sum_a\omega_ax_a^2`$.

**Response operator.** As in [Definition 4.8](04-08_response_operator.md#s4-8), with reference measure $`\rho`$: $`(\mathcal Rh)(a)=\langle\tilde g_a,h\rangle_\rho`$ for $`h=dP/d\rho-dQ/d\rho`$. The residual operator $`(\mathcal Eh)(a)=\langle\tilde e_a,h\rangle_\rho`$ uses the centred residuals $`\tilde e_a=e_a-\mathbb E_\rho e_a`$; $`\epsilon_{\rm tail}`$ bounds its operator norm from $`L^2_0(\rho)`$ to $`(\mathbb R^p,\lVert\cdot\rVert_\omega)`$, and $`\phi\in L^2(\rho)`$.

**Unknown target.** The target feedback $`P`$ (for example the oracle law in the true environment) is unknown. The transcript of demonstrations and queries yields a nonempty compact set $`\mathcal M\subset\mathbb R^r`$ with $`\Pr(\mu_P\in\mathcal M)\ge1-\delta`$.

**Legal set.** $`\mathcal Q_{\rm legal}`$ is a nonempty set of legal mixtures of a finite verified pool, possibly with density constraints ([Proposition 6.14](06-14_convex_realizability_legal_pool.md#s6-14)), whose set of feature means is compact. Its image on the tests is $`\mathcal K=\lbrace C\mu_Q:Q\in\mathcal Q_{\rm legal}\rbrace`$.

**Radius.** For $`q\in\mathcal K`$, $`\Gamma(q)=\sup_{\mu\in\mathcal M}\lVert C\mu-q\rVert_\omega`$, and

```math
\mathfrak r(\mathcal M,\mathcal K)=\min_{q\in\mathcal K}\Gamma(q).
```

<a id="s6-16"></a>

**Proposition 6.16 (Information–compilation radius).** In this setting:

**(a) Existence and convexity.** The minimum exists. $`\Gamma`$ is convex and 1-Lipschitz, so the outer problem is a convex program whenever $`\mathcal K`$ is convex (for example a polytope of legal mixtures).

**(b) Two lower bounds.**

```math
\mathfrak r(\mathcal M,\mathcal K)\ \ge\ \max\Big\lbrace\tfrac12\,\mathrm{diam}_\omega(C\mathcal M),\ \sup_{\mu\in\mathcal M}\mathrm{dist}_\omega(C\mu,\mathcal K)\Big\rbrace .
```

**(c) When the radius vanishes.** $`\mathfrak r(\mathcal M,\mathcal K)=0`$ if and only if $`C\mathcal M`$ is a single point and that point lies in $`\mathcal K`$.

**(d) Guarantee for the radius-optimal compilation.** Let $`q^*`$ attain the minimum and $`Q^*\in\mathcal Q_{\rm legal}`$ have $`C\mu_{Q^*}=q^*`$. If $`\lVert dP/d\rho-dQ^*/d\rho\rVert_\rho\le B`$, then with probability at least $`1-\delta`$

```math
\Big\lVert\mathcal R\Big(\frac{dP}{d\rho}-\frac{dQ^*}{d\rho}\Big)\Big\rVert_\omega\ \le\ \mathfrak r(\mathcal M,\mathcal K)+B\,\epsilon_{\rm tail}.
```

If the outer problem is solved only to accuracy $`\eta_{\rm opt}`$ ($`\Gamma(\hat q)\le\mathfrak r+\eta_{\rm opt}`$), add $`\eta_{\rm opt}`$.

**(e) Three different remedies.**

- If $`\mathcal M`$ varies only along $`\ker C`$, then $`C\mathcal M`$ is a point: further evidence about those directions cannot lower the radius, and nothing is gained by reconstructing the environment beyond what $`C`$ sees.
- If $`C\mathcal M`$ is large, at least half its diameter remains however rich the pool is; only evidence that separates the consequence directions lowers it. [Proposition 6.15](06-15_linear_gaussian_target.md#s6-15) gives the computable linear-Gaussian case.
- If $`C\mathcal M`$ is a point outside $`\mathcal K`$, more of the same evidence does not help; the error is at least the distance to $`\mathcal K`$, and only a larger pool or a different interface lowers it.

**(f) Computing the inner supremum.** $`\Gamma(q)`$ is the maximum of a convex function of $`\mu`$ over $`\mathcal M`$.

- For a finite set of scenarios it is a maximum of finitely many norms, and the outer problem is a second-order cone program when $`\mathcal K`$ is a polytope (or, more generally, second-order-cone representable).
- For a single ellipsoid the inner maximum of the quadratic $`\lVert C\mu-q\rVert_\omega^2`$ is computed exactly through the S-lemma ([Pólik & Terlaky, 2007](#ref-polik2007survey)), and the outer problem becomes a semidefinite program when $`\mathcal K`$ is a polytope (or, more generally, representable by linear matrix inequalities).
- For a box or a general polytope the maximum is attained at a vertex, but maximizing a convex quadratic over a box is hard in general (it contains maximum cut). For small $`r`$ the $`2^r`$ vertices of a box can be enumerated.

**What it says.**

- One number, the radius, measures how far the best legal feedback can be from the consequences that the oracle feedback might have, given the evidence so far. It is zero only when the relevant consequences are pinned down and legally attainable.
- Its two lower bounds separate an information gap (half the diameter of the consequences still allowed) from a realizability gap (the distance from the allowed consequences to what legal feedback can produce), and they call for different remedies: acquisition, or a larger pool.
- Uncertainty that the interface cannot see costs nothing.

**Intuition.**

- The distiller must commit to one legal feedback before knowing which target is true. Its worst-case error is the radius of the smallest ball, centred in the legal set, that covers every target the evidence allows. If the allowed targets are spread out, no centre is close to all of them; if they sit in one point outside the legal set, the centre cannot reach it.

**Role in the framework.**

- It connects acquisition (§6), compression into an interface (§4) and legal compilation ([Proposition 6.14](06-14_convex_realizability_legal_pool.md#s6-14)) in one objective on the same tests, instead of treating "more data" and "better feedback" as separate modules.
- Its guarantee (d) is a per-checkpoint, weighted-norm bound. Used inside the composed guarantee of [Corollary 8.16](08-16_composed_guarantee.md#s8-16) it must first be turned into a per-test bound, and the dynamics and coverage terms of [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4) still have to be paid.
- It is conditional on a fixed, calibrated $`C`$. In an unknown environment $`C`$, the terminal evaluation and the visitation can also depend on the environment; then the uncertainty set must be built for the joint object, or the calibration error of $`C`$ added to the bound. A confidence set is a frequentist coverage statement, not an uncalibrated sense of confidence, and messages outside the pool are not covered by it.
- Choosing the next query to shrink the expected or worst-case radius net of its cost is a method proposal; for general nonlinear learning systems it is not shown here to be efficiently solvable or optimal. Building $`C`$, $`\mathcal M`$ and $`\mathcal K`$ requires per-item response logs and a declared target, which must be recorded for this purpose.

**Derivation outline.**

- (a): $`\Gamma`$ is a supremum of convex 1-Lipschitz functions of $`q`$ and is finite because $`\mathcal M`$ is compact; a continuous function attains its minimum on the compact set $`\mathcal K`$.
- (b): for any centre $`q`$, two targets are at most $`2\Gamma(q)`$ apart, and each target is at least its distance to $`\mathcal K`$ from $`q`$.
- (c): a zero radius means one legal point equals every allowed target.
- (d): the response difference splits into $`C(\mu_P-\mu_{Q^*})`$, bounded by $`\Gamma(q^*)`$ on the confidence event, and the residual $`\mathcal E h`$, bounded by $`B\epsilon_{\rm tail}`$.
- (f): a convex function on a polytope is maximal at a vertex; maximum cut is the maximum of the positive semidefinite form $`x^\top Lx/4`$ over $`\lbrace-1,1\rbrace^n`$, with $`L`$ the graph Laplacian.

**Uses.**

- [Definition 4.8](04-08_response_operator.md#s4-8): the response operator and its reference measure.
- [Proposition 6.14](06-14_convex_realizability_legal_pool.md#s6-14): the legal set and its feature means.
- [Proposition 6.15](06-15_linear_gaussian_target.md#s6-15): the computable acquisition case.

**Used by.**

- [Corollary 8.16](08-16_composed_guarantee.md#s8-16): the compilation term of the composed guarantee.
- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2): routing the budget between acquisition and pool enlargement.

**Status.** proved, conditional on the fixed calibrated interface stated in the setting.

## Related work

- The radius is a robust (minimax) approximation problem: a Chebyshev centre of the uncertainty set restricted to the feasible set. The lower bounds are the classical radius–diameter and distance arguments, and the two-point idea behind the information gap is that of [Le Cam, 1973](#ref-lecam1973convergence).
- The S-lemma, which makes the single-ellipsoid case exact, is surveyed in [Pólik & Terlaky, 2007](#ref-polik2007survey).
- What is specific here is the coupling: the uncertainty is about learning consequences on declared tests, and the feasible set is the image of a verified legal pool, so the same number decides between acquiring evidence and changing the pool.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-polik2007survey"></a>

**Imre Pólik, Tamás Terlaky (2007).** A survey of the S-lemma. *SIAM Review* 49, no. 3, 371–418. DOI [10.1137/S003614450444614X](https://doi.org/10.1137/S003614450444614X).
