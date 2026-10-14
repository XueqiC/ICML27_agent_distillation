# Proposition 6.14 · Convex realizability over a finite legal pool

*Statement*

[← Framework](../README.md) · [Statement](06-14_convex_realizability_legal_pool.md) · [Proof](../proofs/06-14_convex_realizability_legal_pool.md) · [Experiments](../experiments/06-14_convex_realizability_legal_pool.md) · [中文](06-14_convex_realizability_legal_pool_ZH.md)

**In one paragraph.** Knowing which feature means a feedback law should have is not the same as being able to emit such a law. In agent distillation the distiller can only train on messages that have been generated and verified, so what it can actually emit is a positive mixture of a finite legal pool. This proposition states when such a mixture matches a target vector of feature means: exactly when the target lies in the convex hull of the pool's feature vectors, and then with at most $`r+1`$ messages. It also states the price of such sparse mixtures: they put large weight on few messages, which inflates the density bound that controls the spectral error of [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9).

**Setting used on this page.**

- **Legal pool.** A finite set $`\lbrace y_1,\dots,y_m\rbrace`$ of verified legal messages (trajectories, labels or batches that the training recipe may consume). A *legal mixture* is a probability vector $`q\in\Delta_m`$, that is $`q_i\ge0`$ and $`\sum_iq_i=1`$; it emits $`y_i`$ with probability $`q_i`$. Negative weights and unverified generated messages are not legal.
- **Retained features.** A fixed feature map $`\phi(y)=(\phi_1(y),\dots,\phi_r(y))\in\mathbb R^r`$, for example the leading message modes of [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9) or any frozen interface of [Definition 4.1](04-01_feature_sufficiency_unresolved_discrepancy.md#s4-1). Write $`\Phi=[\phi(y_1),\dots,\phi(y_m)]\in\mathbb R^{r\times m}`$, so a mixture $`q`$ has feature mean $`\Phi q`$.
- **Target.** A vector $`\mu_*\in\mathbb R^r`$ of feature means that the compiled feedback should have, for example the feature mean of the oracle law at an audited state, or a point chosen by [Proposition 6.16](06-16_information_compilation_radius.md#s6-16).
- **Reference measure and densities.** A probability vector $`\rho`$ on the pool with every $`\rho_i>0`$ (for example the mixture measure of [Definition 4.8](04-08_response_operator.md#s4-8)). A mixture $`q`$ has density $`dQ/d\rho=(q_i/\rho_i)_i`$ and $`\lVert f\rVert_\rho^2=\sum_i\rho_if_i^2`$. For a target law $`P`$ on the pool, $`B=\lVert dP/d\rho-dQ/d\rho\rVert_\rho`$ is the density bound that multiplies the spectral tail in [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9).
- For $`S\subseteq\lbrace1,\dots,m\rbrace`$, $`\rho(S)=\sum_{i\in S}\rho_i`$, and the support of $`q`$ is $`\lbrace i:q_i>0\rbrace`$.

<a id="s6-14"></a>

**Proposition 6.14 (Convex realizability over a finite legal pool).** In this setting:

**(a) Existence.** A legal mixture with $`\Phi q=\mu_*`$ exists if and only if $`\mu_*\in\mathrm{conv}\lbrace\phi(y_1),\dots,\phi(y_m)\rbrace`$.
**(b) At most $`r+1`$ messages.** If one exists, one exists with at most $`r+1`$ nonzero weights. Every vertex of the polytope $`\lbrace q\ge0:\Phi q=\mu_*,\ \mathbf 1^\top q=1\rbrace`$, for example the basic solution returned by a simplex method, has this property.
**(c) Outside the hull.** If $`\mu_*\notin\mathrm{conv}\lbrace\phi(y_i)\rbrace`$, then for every norm $`\lVert\cdot\rVert`$ on $`\mathbb R^r`$

```math
\min_{q\in\Delta_m}\lVert\Phi q-\mu_*\rVert=\mathrm{dist}\big(\mu_*,\mathrm{conv}\lbrace\phi(y_i)\rbrace\big)>0 ,
```

and the minimum is attained. No legal mixture of this pool does better; only a larger pool, new evidence or a different training interface can reduce this error.

**(d) The density cost of sparse mixtures.** Every mixture $`q`$ with support $`S`$ satisfies

```math
\Big\lVert\frac{dQ}{d\rho}\Big\rVert_\rho^2=\sum_i\frac{q_i^2}{\rho_i}\ \ge\ \frac1{\rho(S)},\qquad\text{hence}\qquad B=\Big\lVert\frac{dP}{d\rho}-\frac{dQ}{d\rho}\Big\rVert_\rho\ \ge\ \rho(S)^{-1/2}-\Big\lVert\frac{dP}{d\rho}\Big\rVert_\rho .
```

With $`\rho`$ uniform and $`\lvert S\rvert=r+1`$ the right side is $`\sqrt{m/(r+1)}-\lVert dP/d\rho\rVert_\rho`$.

**(e) Constrained compilation.** Adding a density cap $`q_i\le\kappa\rho_i`$ or a $`\chi^2`$ constraint $`\sum_iq_i^2/\rho_i\le1+\beta^2`$ keeps the feasible set convex and compact, so compilation remains a linear or convex quadratic program. Under the cap, every feasible mixture has $`\rho(S)\ge1/\kappa`$; with $`\rho`$ uniform it uses at least $`m/\kappa`$ messages, more than $`r+1`$ whenever $`\kappa<m/(r+1)`$.

**What it says.**

- Compiling a target vector of feature means into legal feedback is a convex feasibility problem over the pool. It is solvable exactly when the target lies in the convex hull of the pool's feature vectors, and then $`r+1`$ messages suffice.
- Outside the hull the error has an exact floor, the distance to the hull, and no reweighting of the same pool removes it.
- Sparse solutions are not free: a mixture on few messages has a large density relative to the reference measure. One cannot claim both "only $`r+1`$ messages" and "the spectral error bound of [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9) unchanged" without checking the density.

**Intuition.**

- A positive mixture can only average what the pool already contains, so its feature means are confined to the convex hull of the pool.
- Concentrating the mixture on a few messages is like importance weighting with large weights: the matched features are exact, but every feature that was not matched is now measured against a law far from the reference.

**Role in the framework.**

- It replaces the assumption that the oracle law can be generated ((R) of [Theorem 6.5](06-05_achievability.md#s6-5)) when the distiller can only emit mixtures of a verified pool. That generability assumption is kept only as a conditional special case.
- It supplies the legal set of [Proposition 6.16](06-16_information_compilation_radius.md#s6-16) and the compilation step of [Algorithm 9.2](09-02_ced_algorithm.md#s9-2).
- It guarantees matched feature means, not textual correctness of the messages and not a lossless learning outcome. Three ways to keep the error accounting honest:
  - add a density cap or a $`\chi^2`$ constraint as in (e), and accept a larger support;
  - use the uniform-norm bound of [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2), which needs no density condition: if every response is uniformly approximated by the features to $`\varepsilon`$, two laws with equal feature means differ by at most $`2\varepsilon`$;
  - report the realized $`B`$ and the per-test residuals of the compiled law.
- If the target lies outside the hull, report infeasibility or pay the distance in (c) as feature mismatch. Negative weights or unverified generated messages are not a proof of achievability.

**Derivation outline.**

- (a) is the definition of the convex hull of a finite set.
- (b) is Carathéodory's theorem in linear-programming form: at a vertex the used columns $`(\phi(y_i),1)\in\mathbb R^{r+1}`$ are linearly independent, so there are at most $`r+1`$ of them.
- (c) uses compactness of the hull; (d) is Cauchy–Schwarz on the support; (e) intersects the simplex with convex constraints and sums the cap over the support.

**Uses.**

- [Definition 4.8](04-08_response_operator.md#s4-8): the reference measure.
- [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9): the density bound $`B`$ in the spectral error.
- [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2): the uniform-norm alternative.

**Used by.**

- [Theorem 6.5](06-05_achievability.md#s6-5): scope of the realizability assumption.
- [Proposition 6.16](06-16_information_compilation_radius.md#s6-16): the legal set.
- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2): compilation of candidate feedback laws.

**Status.** proved.

## Related work

- (b) is Carathéodory's theorem on convex hulls in $`\mathbb R^r`$ ([Rockafellar, 1970](#ref-rockafellar1970convex)), here in its linear-programming form.
- Choosing a small training set that steers a known learner is machine teaching ([Zhu, 2015](#ref-zhu2015machine)). Here the candidates are restricted to a fixed verified pool and the target is a vector of feature means, so the question becomes convex feasibility with an explicit density price.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-rockafellar1970convex"></a>

**R. Tyrrell Rockafellar (1970).** Convex Analysis. Princeton University Press, Princeton, NJ (Princeton Mathematical Series 28). DOI [10.1515/9781400873173](https://doi.org/10.1515/9781400873173).

<a id="ref-zhu2015machine"></a>

**Xiaojin Zhu (2015).** Machine teaching: An inverse problem to machine learning and an approach toward optimal education. *Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI)* 29, no. 1. DOI [10.1609/aaai.v29i1.9761](https://doi.org/10.1609/aaai.v29i1.9761).
