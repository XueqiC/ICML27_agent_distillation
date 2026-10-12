# Theorem 8.1 · Stability–approximation tradeoff

*Statement*

[← Framework](../README.md) · [Statement](../statements/08-01_stability_approximation_tradeoff.md) · [Proof](../proofs/08-01_stability_approximation_tradeoff.md) · [Experiments](../experiments/08-01_stability_approximation_tradeoff.md) · [中文](08-01_stability_approximation_tradeoff_ZH.md)

**In one paragraph.** A finite audit never knows feature means exactly; it knows them up to tolerances. This theorem computes the largest response difference that two message laws can still have when their feature means agree up to those tolerances. The answer is an exact tradeoff between how well the features approximate the response and how large the coefficients of that approximation are. It is the first tool of §8 (verification): it says which features and which response models can serve as a certificate.

**Setting used on this page.**

Let $`\mathcal Y`$ be a finite message alphabet with $`\lvert\mathcal Y\rvert=m\ge2`$.

- Functions on $`\mathcal Y`$ are vectors in $`\mathbb R^m`$, and probability laws are vectors in the simplex $`\Delta(\mathcal Y)`$. For $`h\in\mathbb R^m`$ write $`(P-Q)h=\sum_y(P(y)-Q(y))h(y)`$.
- $`g\in\mathbb R^m`$ is a response. $`\phi_1,\dots,\phi_d\in\mathbb R^m`$ are fixed features, $`\Phi\in\mathbb R^{m\times d}`$ has columns $`\phi_j`$, and $`\Phi b=\sum_jb_j\phi_j`$.
- $`\mathbf{\epsilon}=(\epsilon_1,\dots,\epsilon_d)`$ are tolerances with $`\epsilon_j\ge0`$. Zero tolerances and linearly dependent features are allowed.
- The oscillation of $`h`$ is $`\mathrm{osc}(h)=\max_yh(y)-\min_yh(y)`$.

The *tolerant discrepancy* of $`g`$ is

```math
W_{\mathbf{\epsilon}}(g)=\sup\Big\lbrace\lvert(P-Q)g\rvert:\ P,Q\in\Delta(\mathcal Y),\ \lvert(P-Q)\phi_j\rvert\le\epsilon_j\ \ (j=1,\dots,d)\Big\rbrace .
```

It is the largest response difference between two message laws whose feature means differ by at most the declared tolerances. This is the situation after a finite audit that has measured each feature mean to accuracy $`\epsilon_j`$.

<a id="s8-1"></a>

**Theorem 8.1 (Stability–approximation tradeoff).**

**(a) Exact tradeoff.** Both minima below are attained, and

```math
W_{\mathbf{\epsilon}}(g)=\min_{b\in\mathbb R^d,\ c\in\mathbb R}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big]=\min_{b\in\mathbb R^d}\Big[\mathrm{osc}(g-\Phi b)+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big].
```

In particular $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$, which is at most $`1`$ when $`g\in[0,1]^m`$.

- **(b) Exact case.** For $`\mathbf{\epsilon}=0`$, $`W_0(g)=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$ with $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$. This is the single-response form of [Theorem 4.2(b)](../statements/04-02_sufficiency_duality_transfer.md#s4-2).
- **(c) Stability witness.** Let $`g\in\lbrace0,1\rbrace^m`$ be non-constant, $`d=1`$ and $`\phi_1=\alpha g`$ with $`\alpha>0`$. Then $`W_\epsilon(g)=\min\lbrace1,\epsilon/\alpha\rbrace`$.
- **(d) Finite measurement bridge.** Fix a simultaneous event on which every audited feature mean satisfies $`\lvert\widehat\mu_{P,j}-\mu_{P,j}\rvert\le e_{P,j}`$ and $`\lvert\widehat\mu_{Q,j}-\mu_{Q,j}\rvert\le e_{Q,j}`$. Set $`\epsilon_j=\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}`$. Then on that event $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(g)`$. If moreover $`\lVert\widehat g-g\rVert_\infty\le\eta`$, then $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$.

**What it says.** When each feature mean is known only up to a tolerance, the largest response difference that cannot be ruled out equals the best tradeoff between the approximation error of the response by the features and the size of the approximation's coefficients, each coefficient weighted by its feature's tolerance.

**Intuition.** The cost of measuring a feature imprecisely is its coefficient in the response model. A fit with a small residual and huge coefficients proves nothing.

**Role in the framework.** It decides which features an audit should measure and which response models are acceptable as certificates. The frozen audit of Theorem 8.2 uses its linear term, and the fitting rule of the method minimizes its right side.

**Comments on the statement.**

- **Meaning.** $`\mathrm{osc}(g-\Phi b)`$ is the approximation error and $`\sum_j\epsilon_j\lvert b_j\rvert`$ is the price of measurement error. A feature set that represents $`g`$ exactly with large coefficients is a poor certificate: in (c), $`W=\epsilon/\alpha`$ grows as the feature scale $`\alpha`$ shrinks.
- **Sharpness.** (a) is an equality over unrestricted pairs of laws. For a fixed reference, a restricted family of realizable laws, or fixed-marginal couplings ([Proposition 9.1](../statements/09-01_coupling_width.md#s9-1)), the right side remains an upper bound but may not be attained.
- **Use.** [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2) uses the linear term with a coefficient bound $`B_t\ge\lVert b\rVert_1`$. Minimizing the right side of (a) is the fitting objective for response models in the method ([Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3)).

**Derivation outline.**

- Differences of probability laws are exactly the vectors with zero sum and $`\ell_1`$ norm at most $`2`$, and the support function of that set is the oscillation.
- The feature constraints make $`W_{\mathbf{\epsilon}}`$ a linear program; weak duality gives the upper bound for every coefficient vector $`b`$, and Sion's minimax theorem applied to the Lagrangian gives equality.
- The minimum is attained because the dual objective is convex polyhedral and bounded below.

**Illustration.** A feature that equals the response scaled by $`\alpha=0.01`$ represents it exactly, yet with tolerance $`\epsilon=0.005`$ it certifies only $`W=0.5`$: the coefficient $`1/\alpha`$ multiplies the measurement error.

**Uses.**

- [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2) — exact case: twice the uniform approximation error
- [Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity.md#s4-6) — approximation width of response classes

**Used by.**

- [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2) — coefficient bound in the frozen audit
- [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3) — fitting rule for response models

## Related work

- [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1) is linear-programming and norm duality applied to the operational discrepancy; no new duality principle is claimed.
