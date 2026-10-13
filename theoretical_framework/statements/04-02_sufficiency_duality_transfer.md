# Theorem 4.2 · Sufficiency duality and transfer

*Statement*

[← Framework](../README.md) · [Statement](04-02_sufficiency_duality_transfer.md) · [Proof](../proofs/04-02_sufficiency_duality_transfer.md) · [Experiments](../experiments/04-02_sufficiency_duality_transfer.md) · [中文](04-02_sufficiency_duality_transfer_ZH.md)

**In one paragraph.** Suppose two feedback laws are compared only through the means of a few features of the message. When do equal feature means guarantee equal learning outcomes, and how large can the outcome difference be when they do not? This theorem answers both questions in linear-algebra terms: exact sufficiency is a span condition, the worst hidden difference is twice a best uniform approximation error, a bounded-coefficient decomposition transfers feature-mean errors into response errors, and an unrestricted learner needs the whole distribution. It is the general counting tool behind every information count in §4.

**Setting used on this page.**

- $`\mathcal Y`$ is a finite message alphabet with $`\lvert\mathcal Y\rvert=m\ge2`$. Functions on $`\mathcal Y`$ are vectors in $`\mathbb R^m`$; $`\mathbf 1`$ is the all-ones vector. Laws are vectors of $`\Delta(\mathcal Y)=\lbrace P\ge0:\mathbf 1^\top P=1\rbrace`$, and $`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$.
- $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ is a response class: the one-boundary responses $`g(y)`$ = expected final task return after receiving message $`y`$, applying the actual update and completing the reference training ([Definition 1.4](../README.md#s1-4)). By [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2), two message laws are interchangeable exactly when $`(P-Q)g=0`$ for every $`g\in\mathcal G`$.
- A feature interface transmits the means of $`\phi_1,\dots,\phi_d`$; $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$ and $`H_\phi^\perp`$ is its orthogonal complement ([Definition 4.1](04-01_feature_sufficiency_unresolved_discrepancy.md#s4-1)).
- $`\mathcal M_\phi`$ is the set of feature-matched pairs $`(P,Q)`$, i.e. $`(P-Q)\phi_j=0`$ for all $`j`$. The unresolved discrepancy is $`\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\sup_{g\in\mathcal G}\lvert(P-Q)g\rvert`$, and $`d_0(\mathcal G)`$ is the least number of features with $`\mathfrak D_\phi(\mathcal G)=0`$.
- $`\mathrm{dist}_\infty(g,H)=\min_{h\in H}\lVert g-h\rVert_\infty`$, $`\mathbb E_P\phi=(\mathbb E_P\phi_1,\dots,\mathbb E_P\phi_d)`$, and $`\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$ is the total variation distance.

<a id="s4-2"></a>

**Theorem 4.2 (Sufficiency duality and transfer).**

- **(a) Exact sufficiency.** $`(P-Q)g=0`$ for all $`(P,Q)\in\mathcal M_\phi`$ and all $`g\in\mathcal G`$ if and only if $`\mathrm{span}(\mathcal G)\subseteq H_\phi`$. Consequently

```math
d_0(\mathcal G)=\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1. \qquad(4.1)
```

- **(b) Approximate sufficiency.**

```math
\mathfrak D_\phi(\mathcal G)=2\sup_{g\in\mathcal G}\ \mathrm{dist}_\infty(g,H_\phi). \qquad(4.2)
```

- Hence the smallest unresolved discrepancy achievable with $`d`$ features is $`2\inf_H\sup_{g\in\mathcal G}\mathrm{dist}_\infty(g,H)`$, the infimum ranging over subspaces $`H\ni\mathbf 1`$ with $`\dim H\le d+1`$.
- **(c) Transfer.** Suppose every $`g`$ in a target class can be written $`g=c+\sum_{\ell=1}^db_\ell\phi_\ell+r`$ with $`\lVert b\rVert_1\le B`$ and $`\lVert r\rVert_\infty\le\tau`$. Then for all $`P,Q\in\Delta(\mathcal Y)`$,

```math
\sup_g\lvert(P-Q)g\rvert\le B\,\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau .
```

- **(d) Unrestricted learners.** Fix a task with two policies of values $`a`$ and $`a+\gamma`$, $`\gamma>0`$. Suppose the learner family contains, for every event $`A\subseteq\mathcal Y`$, the one-step rule "deploy the better policy exactly when $`Y\in A`$". Then

```math
\sup_{A\subseteq\mathcal Y}\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert=\gamma\,\mathrm{TV}(P,Q),\qquad d_0(\mathcal G)=m-1 .
```

**What it says.**

- (a) Equal feature means imply equal response means for every pair of laws exactly when every response is a constant plus a combination of the features. The number of statistics needed is the dimension of the span of the responses, not counting the constant.
- (b) If the features do not span the responses, the worst hidden difference equals twice the best uniform error with which the features approximate the worst response. It is an equality, not only a bound.
- (c) If each response is "features with coefficients of total size at most $`B`$, plus a residual of size at most $`\tau`$", a feature-mean error of size $`\epsilon`$ costs at most $`B\epsilon+2\tau`$ in the outcome.
- (d) A learner that can react to any event of the message forces the interface to carry the whole distribution: $`m-1`$ statistics.

**Intuition.** Equal feature means rule out differences only in the directions the features span. Whatever part of a response lies outside that span is a direction along which two laws can differ invisibly.

**Role in the framework.** It turns "how much information must two feedback laws share" into computable quantities, a span dimension or an approximation width. Every later count in §4, the first-order statistics of §5 and the audit bounds of §8 are applications of it.

**Scope.**

- The supremum in (b) ranges over all pairs sharing some common feature means. If the reference law is fixed in advance, or both laws must lie in one prescribed feature-mean fiber, the feasible differences shrink and (4.2) becomes an upper bound.
- If only a family of laws is realizable, part (a) reads: $`(P-Q)g=0`$ on feature-matched pairs if and only if every feature-matched difference within the family annihilates $`g`$.
- A small residual $`\tau`$ with a large coefficient size $`B`$ amplifies feature-estimation error; [Theorem 8.1](08-01_stability_approximation_tradeoff.md#s8-1) makes this tradeoff exact.
- $`d_0=m-1`$ in (d) needs learners that realize every singleton indicator. Practical learners and tasks realize far fewer responses; (d) shows only that no uniformly small interface exists unless the task and learner class are restricted.

**Derivation outline.**

- (a) If $`g\in H_\phi`$, it is a constant plus a combination of features, and both have equal means under matched laws. If $`g\notin H_\phi`$, its component $`v`$ orthogonal to $`H_\phi`$ is a direction along which the uniform law can be tilted without moving any feature mean, while $`(P-Q)g`$ moves in proportion to $`\lVert v\rVert_2^2`$.
- (b) Hölder's inequality gives $`\lvert(P-Q)(g-h)\rvert\le2\lVert g-h\rVert_\infty`$ for every $`h\in H_\phi`$. For attainment, linear-programming duality writes the sup-norm distance as a maximum of $`\sigma^\top g`$ over $`\sigma\perp H_\phi`$ with $`\lVert\sigma\rVert_1\le1`$, and every such $`2\sigma`$ is the difference of two laws.
- (c) The constant cancels, Hölder bounds the feature term by $`B`$ times the feature-mean gap, and the residual contributes at most $`2\tau`$.
- (d) The responses $`a+\gamma\mathbf 1_A`$ realize the event form of total variation, and the singletons span $`\mathbb R^m`$.

**Illustration.** Take $`\mathcal Y=\lbrace1,2,3\rbrace`$ and the single feature $`\phi(y)=y`$. The laws $`P=(\tfrac12,0,\tfrac12)`$ and $`Q=(0,1,0)`$ have the same mean, 2. The response $`g(y)=y^2`$ separates them by 1: $`\mathbb E_Pg=5`$ and $`\mathbb E_Qg=4`$. The feature $`\phi`$ is insufficient because $`y^2\notin\mathrm{span}\lbrace1,y\rbrace`$ on three points.

**Uses.**

- [Definition 1.4](../README.md#s1-4) — responses and the response class.
- [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2) — interchangeability means agreement on every response.
- [Definition 4.1](04-01_feature_sufficiency_unresolved_discrepancy.md#s4-1) — feature interface, matched pairs, discrepancy.

**Used by.**

- [Theorem 4.4](04-04_affine_learners_moment_order.md#s4-4) — the exact moment count for affine learners.
- [Proposition 4.6](04-06_exact_order_not_approximate_complexity.md#s4-6) — parity's approximation error via (b).
- [Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1) — minimality of first-order statistics.
- [Theorem 8.1](08-01_stability_approximation_tradeoff.md#s8-1) — duality with measurement tolerances.

## Related work

- Part (b) is a linear-programming duality ([Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)). Minimized over subspaces $`H\ni\mathbf 1`$ of dimension at most $`d+1`$, the right side of (4.2) becomes twice a Kolmogorov width of $`\mathcal G`$ in the sup norm, taken over subspaces that contain the constants ([Pinkus, 1985](#ref-pinkus1985nwidths)).
- $`\mathfrak D_\phi`$ is an integral probability metric ([Müller, 1997](#ref-muller1997integral)) restricted to feature-matched pairs. The total variation distance in (d) is the integral probability metric generated by the indicators of events.
