# Definition 4.1 · Feature sufficiency and unresolved discrepancy

*Statement*

[← Framework](../README.md) · [Statement](04-01_feature_sufficiency_unresolved_discrepancy.md) · [中文](04-01_feature_sufficiency_unresolved_discrepancy_ZH.md)

**In one paragraph.** Two feedback laws are interchangeable for learning exactly when they agree on every response in the response class ([Theorem 3.2](03-02_continuation_response_characterization.md#s3-2)). In practice one compares laws through a few summary statistics, such as the means of some features of the message. This definition fixes the vocabulary for asking how many such statistics are needed: a *feature interface*, the pairs of laws it cannot tell apart, and the largest response difference those pairs can still hide.

**Setting used on this page.**

- $`\mathcal Y`$ is a finite message alphabet with $`\lvert\mathcal Y\rvert=m\ge2`$. A message is one unit of feedback consumed by the learner in one round.
- Functions on $`\mathcal Y`$ are vectors in $`\mathbb R^m`$; $`\mathbf 1`$ is the all-ones vector. Probability laws on $`\mathcal Y`$ are the vectors of $`\Delta(\mathcal Y)=\lbrace P\in\mathbb R^m_{\ge0}:\mathbf 1^\top P=1\rbrace`$, and $`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$ is the difference of the expectations of $`g`$ under $`P`$ and $`Q`$.
- $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ is a *response class*. In the framework it collects the one-boundary responses $`g_{t,z}(y)`$: the expected final task return after the learner receives message $`y`$ at state $`z`$, applies its actual update and completes the declared reference training ([Definition 1.4](../README.md#s1-4)). Only the values of the responses on $`\mathcal Y`$ matter here.
- All probability laws on $`\mathcal Y`$ are admissible unless stated otherwise.

<a id="s4-1"></a>

**Definition 4.1 (Feature sufficiency and unresolved discrepancy).**

- A *feature interface* transmits the means of fixed features $`\phi_1,\dots,\phi_d:\mathcal Y\to\mathbb R`$. Write $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace\subseteq\mathbb R^m`$ and $`H_\phi^\perp=\lbrace v\in\mathbb R^m:v^\top h=0\ \forall h\in H_\phi\rbrace`$. Every $`v\in H_\phi^\perp`$ satisfies $`\mathbf 1^\top v=0`$.
- The *feature-matched pairs* are $`\mathcal M_\phi=\lbrace(P,Q)\in\Delta(\mathcal Y)^2:(P-Q)\phi_j=0,\ j=1,\dots,d\rbrace`$.
- The *unresolved discrepancy* of the interface on $`\mathcal G`$ is

```math
\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\ \sup_{g\in\mathcal G}\ \lvert(P-Q)g\rvert .
```

- The *exact dimension* $`d_0(\mathcal G)`$ is the least $`d`$ for which some features $`\phi_1,\dots,\phi_d`$ give $`(P-Q)g=0`$ for all $`(P,Q)\in\mathcal M_\phi`$ and all $`g\in\mathcal G`$. The *approximate dimension* $`d_\varepsilon(\mathcal G)`$ is the least $`d`$ for which some $`d`$ features give $`\mathfrak D_\phi(\mathcal G)\le\varepsilon`$.
- For $`g\in\mathbb R^m`$ and a subspace $`H`$, $`\mathrm{dist}_\infty(g,H)=\min_{h\in H}\lVert g-h\rVert_\infty`$.

**What it says.**

- A feature interface is a summary of a feedback law by finitely many expectations. Two laws with equal summaries form a feature-matched pair.
- The unresolved discrepancy is the worst response difference that a feature-matched pair can still produce: the risk the summary cannot see.
- $`d_0`$ counts how many statistics make that risk exactly zero; $`d_\varepsilon`$ counts how many make it at most $`\varepsilon`$.

**Intuition.** If the summary statistics are equal but some response still differs, the summary has thrown away information that the learner and the task use.

**Role in the framework.**

- These dimensions count fixed linear expectation statistics. They are not bit complexities, sample sizes or query counts; estimating the statistics from finite data is the subject of §8 ([Theorem 8.1](08-01_stability_approximation_tradeoff.md#s8-1)).
- Every count in §4 is a value of $`d_0`$ or $`d_\varepsilon`$ for a particular response class.

**Uses.**

- [Definition 1.4](../README.md#s1-4) — the one-boundary response and the response class.

**Used by.**

- [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2) — computes $`d_0`$ and $`\mathfrak D_\phi`$ exactly.
- [Theorem 4.4](04-04_affine_learners_moment_order.md#s4-4) — exact counts for affine learners.
- [Proposition 4.6](04-06_exact_order_not_approximate_complexity.md#s4-6) — approximate versus exact order.
- [Theorem 8.1](08-01_stability_approximation_tradeoff.md#s8-1) — the cost of measuring features with error.
