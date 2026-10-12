# §8 Verification (Theorem C) — Proofs

[← Framework](../README.md) · [Experiments](../experiments/08_verification.md) · [中文](08_verification_ZH.md)

This page retains Theorem 8.1 through Theorem 8.3 and the setting; mean-increment measurement and Theorem C continue on the [certificate page](08_certificate.md).

**Summary.**

- This section asks how finitely many measurements can certify that a feedback law is good enough, or rank several candidate laws. The results differ in what they measure.
  - [Theorem 8.1](#s8-1): when feature means are measured with tolerances, the guaranteed response difference is an exact tradeoff between approximation error and coefficient size.
  - [Theorem 8.2](#s8-2): an audit on cells frozen in advance certifies learning value only through an explicit transport (coverage) term, and that term cannot be dropped.
  - [Theorem 8.3](#s8-3): paired full-tail experiments at the states training actually reaches, with selection by a lower confidence bound, guarantee no loss against the reference beyond $`\delta`$. They are expensive.
  - [Proposition 8.4](08_certificate.md#s8-4) and [Remark 8.5](08_certificate.md#s8-5): the mean increment can be measured with gradient evaluations alone, but turning it into a bound needs constants of the continuation value that the samples do not estimate.
  - **Theorem C** ([Theorem 8.8](08_certificate.md#s8-8)): a common-random-numbers (CRN) paired-replacement score on a surrogate tail estimates the *finite* effect of a one-round replacement without bias. Its radius contains only a statistical term and a surrogate-calibration term. It needs no continuation gradient, no curvature constant and no second-order remainder.
  - [Corollary 8.9](08_certificate.md#s8-9): the first-order version of the same measurement, its margin criterion, and its cap. A pair whose first-order term does not exceed the second-order remainder is never certified at first order.
  - [Proposition 8.10](08_certificate.md#s8-10), [Remark 8.11](08_certificate.md#s8-11) and [Remark 8.12](08_certificate.md#s8-12): cost and the role of CRN; calibration and transfer along training; the two open problems.
- Uses: the telescoping identity of [Theorem 3.2](03_learning_consequence.md#s3-2); the approximation duality of [Theorem 4.2](04_complexity_of_sufficiency.md#s4-2); the affine rates of [Theorem 4.5](04_rates_and_composition.md#s4-5); the first-order expansion of [Proposition 5.1](05_optimizer_state.md#s5-1); the learner statistics of [Theorem 5.10](05_learner_statistics.md#s5-10).

## Setting

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- Rounds are $`t=0,\dots,R-1`$, and $`z`$ is a complete learner state. For a first-order optimizer the actual update is $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$.
- At state $`z`$ in round $`t`$ the reference emits $`Y\sim P^0_{t,z}`$, and candidate $`j\in\lbrace1,\dots,M\rbrace`$ emits $`Y\sim P^j_{t,z}`$. The kernel of candidate $`j`$ is $`K^j_t`$.
- The reference continuation is $`V^0_{t+1}`$, with $`V^0_R=F_E\in[0,1]`$. The response is $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$. Integrating the message gives the kernel identity

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- $`L_E(j)=\mathbb E\,F_E(Z^j_R)`$ for the process driven by candidate $`j`$. All processes start from the same initial law of $`Z_0`$.
- The *local advantage* of candidate $`j`$ at $`(t,z)`$ is

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- It is the effect on the final score of replacing one reference round by one round of candidate $`j`$, followed by the reference tail. By [Theorem 3.2](03_learning_consequence.md#s3-2), the whole-run difference is a sum of such terms over the states the alternative process reaches.

## Measurement error and coefficient stability

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

- **(b) Exact case.** For $`\mathbf{\epsilon}=0`$, $`W_0(g)=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$ with $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$. This is the single-response form of [Theorem 4.2(b)](04_complexity_of_sufficiency.md#s4-2).
- **(c) Stability witness.** Let $`g\in\lbrace0,1\rbrace^m`$ be non-constant, $`d=1`$ and $`\phi_1=\alpha g`$ with $`\alpha>0`$. Then $`W_\epsilon(g)=\min\lbrace1,\epsilon/\alpha\rbrace`$.
- **(d) Finite measurement bridge.** Fix a simultaneous event on which every audited feature mean satisfies $`\lvert\widehat\mu_{P,j}-\mu_{P,j}\rvert\le e_{P,j}`$ and $`\lvert\widehat\mu_{Q,j}-\mu_{Q,j}\rvert\le e_{Q,j}`$. Set $`\epsilon_j=\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}`$. Then on that event $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(g)`$. If moreover $`\lVert\widehat g-g\rVert_\infty\le\eta`$, then $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$.

*Proof.* Three elementary facts come first.

*Claim 1 (difference set).* $`\lbrace P-Q:P,Q\in\Delta(\mathcal Y)\rbrace=\mathcal D:=\lbrace v\in\mathbb R^m:\mathbf 1^\top v=0,\ \lVert v\rVert_1\le2\rbrace`$.

- If $`v=P-Q`$, then $`\mathbf 1^\top v=1-1=0`$ and $`\lVert v\rVert_1\le\lVert P\rVert_1+\lVert Q\rVert_1=2`$.
- Conversely, let $`v\in\mathcal D`$ and write $`v=v^+-v^-`$ with $`v^\pm\ge0`$ of disjoint supports. Since $`\mathbf 1^\top v=0`$, both parts have the same mass $`s=\mathbf 1^\top v^+=\mathbf 1^\top v^-`$, and $`2s=\lVert v\rVert_1\le2`$. Fix any $`q\in\Delta(\mathcal Y)`$ and put $`P=v^++(1-s)q`$, $`Q=v^-+(1-s)q`$. Both are probability vectors and $`P-Q=v`$.

*Claim 2 (support function).* For every $`h\in\mathbb R^m`$, $`\max_{v\in\mathcal D}h^\top v=\mathrm{osc}(h)`$.

- For $`v\in\mathcal D`$ with parts of common mass $`s\le1`$, $`h^\top v=h^\top v^+-h^\top v^-\le s\max h-s\min h\le\mathrm{osc}(h)`$, because $`\mathrm{osc}(h)\ge0`$.
- Equality holds at $`v=e_{y^+}-e_{y^-}`$, where $`y^\pm`$ are a maximizer and a minimizer of $`h`$.

*Claim 3 (oscillation and centred sup-norm).* $`\mathrm{osc}(h)=2\min_{c\in\mathbb R}\lVert h-c\mathbf 1\rVert_\infty`$, attained at $`c^\star=(\max h+\min h)/2`$. Indeed $`\max h-\min h=(\max h-c)+(c-\min h)\le2\lVert h-c\mathbf 1\rVert_\infty`$ for every $`c`$, and at $`c^\star`$ both summands equal $`\mathrm{osc}(h)/2`$.

*Step 1 (reduction to a linear program).* By Claim 1,

```math
W_{\mathbf{\epsilon}}(g)=\sup\big\lbrace\lvert g^\top v\rvert:\ v\in\mathcal D,\ -\mathbf{\epsilon}\le\Phi^\top v\le\mathbf{\epsilon}\big\rbrace .
```

The feasible set $`\mathcal F`$ is invariant under $`v\mapsto-v`$, so the absolute value can be dropped. $`\mathcal F`$ is a nonempty polytope (it contains $`0`$ and is bounded), so the supremum is a maximum.

*Step 2 (weak duality).* Fix $`b\in\mathbb R^d`$. For every $`v\in\mathcal F`$, by Claim 2 and the feature constraints,

```math
g^\top v=(g-\Phi b)^\top v+b^\top\Phi^\top v\le\mathrm{osc}(g-\Phi b)+\sum_j\lvert b_j\rvert\,\lvert\phi_j^\top v\rvert\le\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert .
```

Taking the maximum over $`v`$ and the infimum over $`b`$ gives $`W_{\mathbf{\epsilon}}(g)\le\inf_b\big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\big]`$.

*Step 3 (strong duality).* Introduce multipliers $`u,w\in\mathbb R^d_{\ge0}`$ for $`\Phi^\top v\le\mathbf{\epsilon}`$ and $`-\Phi^\top v\le\mathbf{\epsilon}`$, keeping $`v\in\mathcal D`$ as the domain. The Lagrangian is

```math
\mathcal L(v;u,w)=g^\top v-u^\top(\Phi^\top v-\mathbf{\epsilon})-w^\top(-\Phi^\top v-\mathbf{\epsilon})=\big(g-\Phi(u-w)\big)^\top v+\mathbf{\epsilon}^\top(u+w).
```

- $`\mathcal L`$ is bilinear, $`\mathcal D`$ is compact and convex, and $`\mathbb R^{2d}_{\ge0}`$ is convex. Sion's minimax theorem gives $`\max_{v\in\mathcal D}\inf_{u,w\ge0}\mathcal L=\inf_{u,w\ge0}\max_{v\in\mathcal D}\mathcal L`$.
- The left side equals $`\max_{v\in\mathcal F}g^\top v`$, since the inner infimum is $`g^\top v`$ on $`\mathcal F`$ and $`-\infty`$ off it.
- By Claim 2 the right side is $`\inf_{u,w\ge0}\big[\mathrm{osc}(g-\Phi(u-w))+\mathbf{\epsilon}^\top(u+w)\big]`$. For fixed $`b=u-w`$, $`\mathbf{\epsilon}^\top(u+w)\ge\sum_j\epsilon_j\lvert b_j\rvert`$ because $`u_j+w_j\ge\lvert u_j-w_j\rvert`$, with equality at $`u=b^+`$, $`w=b^-`$.

Hence $`W_{\mathbf{\epsilon}}(g)=\inf_b\big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\big]`$.

*Step 4 (attainment).* The objective $`f(b)=\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$ is a finite maximum of affine functions, hence convex polyhedral, and it is bounded below by $`0`$. Minimizing it is a feasible linear program in $`(b,t,s)`$ with $`t\ge(g-\Phi b)(y)-(g-\Phi b)(y')`$ for all $`y,y'`$ and $`s_j\ge\pm b_j`$. A feasible linear program with a bounded-below objective has a finite optimum, so the minimum is attained.

*Step 5 (first form).* By Claim 3, $`\mathrm{osc}(g-\Phi b)=2\min_c\lVert g-c\mathbf 1-\Phi b\rVert_\infty`$; substituting gives the first expression in (a). Choosing $`b=0`$ gives $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$.

*Proof of (b).* With $`\mathbf{\epsilon}=0`$ the penalty vanishes, and $`c\mathbf 1+\Phi b`$ ranges over $`H_\phi`$.

*Proof of (c).* Here $`g-b\alpha g=(1-\alpha b)g`$, and $`\mathrm{osc}((1-\alpha b)g)=\lvert1-\alpha b\rvert`$ because $`g`$ takes both values $`0`$ and $`1`$. So $`f(b)=\lvert1-\alpha b\rvert+\epsilon\lvert b\rvert`$. It is convex and piecewise linear with kinks at $`0`$ and $`1/\alpha`$, increasing away from $`[0,1/\alpha]`$ and affine on it. Hence $`\min_bf(b)=\min\lbrace f(0),f(1/\alpha)\rbrace=\min\lbrace1,\epsilon/\alpha\rbrace`$.

*Proof of (d).* On the event, the triangle inequality gives $`\lvert(P-Q)\phi_j\rvert\le\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}=\epsilon_j`$, so $`(P,Q)`$ is feasible in the definition of $`W_{\mathbf{\epsilon}}(g)`$. For the second claim, $`\lvert(P-Q)g\rvert\le\lvert(P-Q)\widehat g\rvert+\lvert P(g-\widehat g)\rvert+\lvert Q(g-\widehat g)\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$. The coefficients $`b`$ may be optimized after the measurements are seen, because the event controls every linear combination of the fixed features simultaneously. $`\square`$

*Comments on Theorem 8.1.*

- **Meaning.** $`\mathrm{osc}(g-\Phi b)`$ is the approximation error and $`\sum_j\epsilon_j\lvert b_j\rvert`$ is the price of measurement error. A feature set that represents $`g`$ exactly with large coefficients is a poor certificate: in (c), $`W=\epsilon/\alpha`$ grows as the feature scale $`\alpha`$ shrinks.
- **Sharpness.** (a) is an equality over unrestricted pairs of laws. For a fixed reference, a restricted family of realizable laws, or fixed-marginal couplings ([Proposition 9.1](09_mechanism_and_method.md#s9-1)), the right side remains an upper bound but may not be attained.
- **Use.** [Theorem 8.2](#s8-2) uses the linear term with a coefficient bound $`B_t\ge\lVert b\rVert_1`$. Minimizing the right side of (a) is the fitting objective for response models in the method ([Remark 9.3](09_mechanism_and_method.md#s9-3)).

## Frozen audits need a transport term

**Frozen audit.** Using only independent pilot data, freeze before the audit:

- $`M`$ executable candidate feedback laws;
- a finite set $`\mathcal U`$ of audit cells $`u`$, each specifying a complete learner state, a task, a query, the update and the reference tail (the letter $`u`$ denotes a cell only in this subsection);
- $`d`$ features $`\phi_{u,\ell}:\mathcal Y\to[0,1]`$ per cell.

$`P^0_u`$ and $`P^j_u`$ are the message laws of the reference and of candidate $`j`$ in cell $`u`$. The audit draws $`n_0`$ independent reference messages per cell and $`n_S`$ independent messages per candidate and cell. Let $`\mu_{j,u,\ell}=\mathbb E_{P^j_u}\phi_{u,\ell}`$, with empirical means $`\widehat\mu_{j,u,\ell}`$, and

```math
D_j=\max_{u,\ell}\lvert\mu_{0,u,\ell}-\mu_{j,u,\ell}\rvert,\qquad\widehat D_j=\max_{u,\ell}\lvert\widehat\mu_{0,u,\ell}-\widehat\mu_{j,u,\ell}\rvert,
```

```math
e_0=\sqrt{\frac{\log(4\lvert\mathcal U\rvert d/\delta)}{2n_0}},\qquad e_S=\sqrt{\frac{\log(4M\lvert\mathcal U\rvert d/\delta)}{2n_S}}.
```

The selected candidate $`\widehat j`$ minimizes $`\widehat D_j`$ up to a known optimization error $`\eta_{\rm opt}`$.

**Certified states.** For each round $`t`$ there is a certified set $`\mathcal C_t`$ of complete states. Each $`z\in\mathcal C_t`$ carries the following.

- An audited cell $`u=u(t,z)`$ whose features are expressed in the message coordinates at $`z`$.
- A *transport bound*: for every candidate that can be selected on the audit event,

```math
\Big\lVert(P^0_{t,z}-P^j_{t,z})\phi_u-(P^0_u-P^j_u)\phi_u\Big\rVert_\infty\le a_{t,j}(z).\qquad\text{(T)}
```

- A *response decomposition*

```math
g_{t,z}=c_{t,z}+b_{t,z}^\top\phi_u+r_{t,z},\qquad\lVert b_{t,z}\rVert_1\le B_t,\quad\lVert r_{t,z}\rVert_\infty\le\tau_t.\qquad\text{(D)}
```

Let $`\sigma=\inf\lbrace t<R:Z_t\notin\mathcal C_t\rbrace`$ ($`\sigma=\infty`$ if no such $`t`$), a stopping time of the training process. **Fresh training:** conditional on the audit result, the selected candidate is trained with randomness independent of the audit samples, and $`\Pr_{\widehat j}(\sigma<R\mid\text{audit result})\le\rho`$.

<a id="s8-2"></a>

**Theorem 8.2 (Frozen finite-interface selection with transport).** All bounds below may be capped at $`1`$.

**(a) Audit concentration.** With probability at least $`1-\delta`$, simultaneously for all $`j`$,

```math
\lvert D_j-\widehat D_j\rvert\le e_0+e_S,\qquad\Longrightarrow\qquad D_{\widehat j}\le\min_jD_j+2(e_0+e_S)+\eta_{\rm opt}.
```

**(b) Local transport bound.** For $`z\in\mathcal C_t`$ and $`u=u(t,z)`$,

```math
\big\lvert(K^0_t-K^j_t)V^0_{t+1}(z)\big\rvert\le\min\Big\lbrace1,\ B_t\big(D_j+a_{t,j}(z)\big)+2\tau_t\Big\rbrace .
```

**(c) Learning-value bound.** On the event of (a),

```math
\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_{t=0}^{R-1}\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace\Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace\Big]+\rho .
```

- **(d) Specializations.** With exact transport ($`a\equiv0`$), $`\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_tB_t(\widehat D_{\widehat j}+e_0+e_S)+2\sum_t\tau_t+\rho`$. Under the state-independent IID-block and convex-affine assumptions of [Theorem 4.5(a)](04_rates_and_composition.md#s4-5), with the audited features equal to the moments $`Z^\alpha`$, $`1\le\lvert\alpha\rvert\le N`$, $`\lvert L_E(0)-L_E(\widehat j)\rvert\le C_J(\widehat D_{\widehat j}+e_0+e_S)`$.
- **(e) Feature drift.** If state-dependent features $`\phi_z`$ satisfy $`\sup_y\lVert\phi_z(y)-\phi_u(y)\rVert_\infty\le\alpha_{zu}`$, then $`\lVert(P^0_{t,z}-P^j_{t,z})\phi_z\rVert_\infty\le D_j+a_{t,j}(z)+2\alpha_{zu}`$, and (b)–(c) hold with $`\phi_z`$ in (D) and $`a`$ replaced by $`a+2\alpha_{zu}`$.
- **(f) Necessity of transport.** For every audit size there is an instance with $`D_j=0`$, an exact response decomposition ($`B=1`$, $`\tau=0`$) and learning gap $`1`$. In it the transport-free expression $`B(\widehat D+e_0+e_S)`$ is strictly below the gap, while the bound of (c) remains valid and is tight.

*Proof.*

**(a)** Each $`\phi_{u,\ell}\in[0,1]`$, and the samples behind each empirical mean are independent. Two-sided Hoeffding gives

```math
\Pr\big(\lvert\widehat\mu_{0,u,\ell}-\mu_{0,u,\ell}\rvert>e_0\big)\le2e^{-2n_0e_0^2}=\frac{\delta}{2\lvert\mathcal U\rvert d},\qquad\Pr\big(\lvert\widehat\mu_{j,u,\ell}-\mu_{j,u,\ell}\rvert>e_S\big)\le\frac{\delta}{2M\lvert\mathcal U\rvert d}.
```

- A union bound over the $`\lvert\mathcal U\rvert d`$ reference means and the $`M\lvert\mathcal U\rvert d`$ candidate means gives an event $`\mathcal A`$ with $`\Pr(\mathcal A^c)\le\delta`$. Correlation between features evaluated on a common message does not affect a union bound.
- On $`\mathcal A`$, $`\big\lvert(\widehat\mu_{0,u,\ell}-\widehat\mu_{j,u,\ell})-(\mu_{0,u,\ell}-\mu_{j,u,\ell})\big\rvert\le e_0+e_S`$ for every $`j,u,\ell`$. The map $`w\mapsto\max_k\lvert w_k\rvert`$ is $`1`$-Lipschitz for the sup norm, so $`\lvert\widehat D_j-D_j\rvert\le e_0+e_S`$ for all $`j`$.
- Let $`j^\star\in\arg\min_jD_j`$. On $`\mathcal A`$, $`D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S\le\widehat D_{j^\star}+\eta_{\rm opt}+e_0+e_S\le D_{j^\star}+2(e_0+e_S)+\eta_{\rm opt}`$. Selecting on the audit data is legitimate because $`\mathcal A`$ is simultaneous over the frozen candidate set.

**(b)** By (K), $`(K^0_t-K^j_t)V^0_{t+1}(z)=(P^0_{t,z}-P^j_{t,z})g_{t,z}`$. Insert (D) term by term.

- The constant $`c_{t,z}`$ integrates to zero against a difference of probability laws.
- For the linear term, put $`w=(P^0_{t,z}-P^j_{t,z})\phi_u\in\mathbb R^d`$. Then $`\lvert b^\top w\rvert\le\lVert b\rVert_1\lVert w\rVert_\infty`$, and by (T) and the triangle inequality $`\lVert w\rVert_\infty\le\lVert(P^0_u-P^j_u)\phi_u\rVert_\infty+a_{t,j}(z)\le D_j+a_{t,j}(z)`$.
- For the residual, $`\lvert(P^0_{t,z}-P^j_{t,z})r\rvert\le\lvert P^0_{t,z}r\rvert+\lvert P^j_{t,z}r\rvert\le2\tau_t`$.
- Since $`g\in[0,1]`$, the left side is also at most $`1`$.

**(c)** Work on $`\mathcal A`$, conditional on a fixed audit result, so $`\widehat j`$ is fixed. By fresh training, the conditional law of the training process is that of the process driven by $`\widehat j`$.

- *Switched process.* Let $`S'`$ use $`K^{\widehat j}_t`$ at round $`t`$ if $`t<\sigma`$ and $`K^0_t`$ otherwise. Since $`\lbrace\sigma\le t\rbrace`$ is determined by $`Z_0,\dots,Z_t`$, $`S'`$ is a well-defined history-dependent process; $`V^0`$ does not depend on the switch flag. Coupling $`S'`$ with the $`\widehat j`$-process through common randomness, the two paths agree through $`Z_\sigma`$, and through $`Z_R`$ on $`\lbrace\sigma\ge R\rbrace`$. Terminal values lie in $`[0,1]`$, so $`\lvert L_E(\widehat j)-L_E(S')\rvert\le\Pr_{\widehat j}(\sigma<R)\le\rho`$.
- *Telescoping.* By the tower property, $`\mathbb E[V^0_{t+1}(Z_{t+1})\mid Z_0,\dots,Z_t]=(K^{S'}_tV^0_{t+1})(Z_t)`$. Summing $`\mathbb E_{S'}[V^0_t(Z_t)-V^0_{t+1}(Z_{t+1})]`$ over $`t`$, with the common initial law and $`V^0_R=F_E`$, gives

```math
L_E(0)-L_E(S')=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[(K^0_t-K^{S'}_t)V^0_{t+1}(Z_t)\big]=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[1\lbrace t<\sigma\rbrace(K^0_t-K^{\widehat j}_t)V^0_{t+1}(Z_t)\big],
```

because the kernels coincide for $`t\ge\sigma`$. This is the stopped form of [Theorem 3.2(a)](03_learning_consequence.md#s3-2).

- *Pre-exit occupancy.* Before $`\sigma`$, $`S'`$ and the $`\widehat j`$-process use the same kernels, so $`\mathbb E_{S'}[1\lbrace t<\sigma\rbrace h(Z_t)]=\mathbb E_{\widehat j}[1\lbrace t<\sigma\rbrace h(Z_t)]`$ for bounded $`h`$. On $`\lbrace t<\sigma\rbrace`$, $`Z_t\in\mathcal C_t`$ and (b) applies; on $`\mathcal A`$, $`D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S`$. This bounds $`\lvert L_E(0)-L_E(S')\rvert`$ by the sum in (c).
- The triangle inequality with the switched-process bound gives (c). The bound holds for every audit result in $`\mathcal A`$, so (T), (D) and the bound on $`\rho`$ must hold for every candidate that can be selected on $`\mathcal A`$.

**(d)** With $`a\equiv0`$, each summand of (c) is at most $`B_t(\widehat D_{\widehat j}+e_0+e_S)+2\tau_t`$, since $`1\lbrace t<\sigma\rbrace\le1`$ and the bracket is nonnegative. Under the IID affine assumptions every training round of candidate $`j`$ draws a fresh block from one fixed law $`P^j`$, independent of the state. Take a single cell whose features are the monomials $`Z^\alpha`$, $`1\le\lvert\alpha\rvert\le N`$, which lie in $`[0,1]`$. Then $`D_j=\delta_N(P^0,P^j)`$, [Theorem 4.5(a)](04_rates_and_composition.md#s4-5) gives $`\lvert L_E(0)-L_E(j)\rvert\le C_J\delta_N(P^0,P^j)`$, and on $`\mathcal A`$, $`\delta_N(P^0,P^{\widehat j})\le\widehat D_{\widehat j}+e_0+e_S`$. No transport is needed, because the audited law is the law used at every state; the training blocks must be fresh draws from it.

**(e)** Write $`\phi_z=\phi_u+(\phi_z-\phi_u)`$. The first part contributes at most $`D_j+a_{t,j}(z)`$ as in (b). For each coordinate $`\ell`$, $`\lvert(P^0_{t,z}-P^j_{t,z})(\phi_z-\phi_u)_\ell\rvert\le2\alpha_{zu}`$. Repeating (b)–(c) with (D) stated for $`\phi_z`$ proves the variant. A bound on sampled messages only is not a bound on $`\sup_y`$.

**(f)** Let $`\mathcal Y=\lbrace0,1\rbrace`$, $`\phi(y)=y`$, $`R=1`$, let the update set the final policy parameter to $`p=y`$, and let $`F_E(p)=p`$. Then $`g_{0,z}(y)=y`$ at every state, so (D) holds with $`c=0`$, $`b=1`$, $`r=0`$.

- At the audited state $`z_a`$ both laws emit $`0`$, so $`D_j=0`$.
- Training starts at $`z_b`$, where the reference emits $`0`$ and the candidate emits $`1`$, so $`\lvert L_E(0)-L_E(j)\rvert=1`$.
- Declaring $`z_b`$ certified with cell $`u(z_b)=z_a`$ and no transport term gives the expression $`B(\widehat D+e_0+e_S)=e_0+e_S`$. With $`n_0=n_S=10^4`$, $`\delta=0.05`$ and $`\lvert\mathcal U\rvert=M=d=1`$ this is $`2\sqrt{\log80/20000}\approx0.0296<1`$.
- In (c), certifying $`z_b`$ requires $`a_{0,j}(z_b)\ge\lvert(0-1)-(0-0)\rvert=1`$, so the bound is at least $`1`$. Alternatively $`z_b\notin\mathcal C_0`$, so $`\sigma=0`$, $`\rho=1`$ and the bound is $`1`$.

In both cases (c) equals the gap after capping, so the transport term can be neither dropped nor set to zero by convention. The same construction with a covered state that reaches an uncovered state with probability $`r`$ shows that every visited pre-exit state must contribute its term, or the first visit to an uncovered state must be counted in $`\rho\ge r`$; otherwise the bound fails by $`r`$. $`\square`$

*Comments on Theorem 8.2.*

- **Complete states.** Cells, transport and certification refer to complete learner states. Two states with the same parameters but different optimizer moments produce different next states and can have different responses ([Proposition 5.3](05_optimizer_state.md#s5-3)). Parameter proximity alone does not establish (T).
- **Exact transport is weaker than equality of laws.** $`a=0`$ requires only that the two feature-discrepancy vectors agree, not the four laws.
- **What is certified.** (c) bounds the population expected value of the selected frozen procedure under fresh training, not one training seed's score. Building a new candidate from the audit samples, or training on them, changes the experiment and needs a new independent audit.
- **Further confidence failures.** If transport bounds, coefficients or residuals are themselves estimated, their failure probabilities add to $`\delta`$.

## Auditing at the actual states

**Adaptive audit.** A *provider policy* $`S`$ chooses at each round a kernel index $`J_t`$ as a function of the history.

- Let $`\mathcal H_t`$ be the history just before the audit of round $`t`$. Conditional on $`\mathcal H_t`$, freeze the actual state $`Z_t`$, a finite set of $`M_t`$ non-reference candidate kernels, a replica count $`n_t`$ and a confidence allocation $`\delta_t`$, with $`\sum_t\delta_t\le\delta`$ almost surely.
- For each candidate, run $`n_t`$ independent paired experiments from $`Z_t`$. Each applies the candidate update or the reference update, then the **full** reference tail and real deployment. Each arm has its correct marginal law, and the paired contrast lies in $`[-1,1]`$.
- Let $`\widehat A_t(j)`$ be the mean contrast and

```math
e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},\qquad\ell_t(j)=\widehat A_t(j)-e_t,\qquad\ell_t(0)=0 .
```

- Select $`J_t\in\arg\max_j\ell_t(j)`$ and set $`\Gamma_t=\ell_t(J_t)\in[0,1]`$. Then apply a **fresh** update from $`K^{J_t}_t`$, discarding the audit replicas. Unaudited rounds use the reference and set $`\Gamma_t=0`$.
- Audits leave the actual learner and environment state unchanged, and the reference ignores audit records. $`\mathcal F_t\supseteq\mathcal H_t`$ is the history after the audit and the selection, before the fresh update.

<a id="s8-3"></a>

**Theorem 8.3 (Actual-state adaptive audit; fidelity versus improvement).**

- **(a) Fidelity.** If $`L_E(0)-J_E(\pi_0)\ge\gamma`$ and $`L_E(0)-L_E(S)\le\varepsilon`$, then $`L_E(S)-J_E(\pi_0)\ge\gamma-\varepsilon`$. Here $`J_E(\pi_0)`$ is the value of the initial student; $`\varepsilon`$ can be the certificate of [Theorem 8.2(c)](#s8-2), and $`\gamma`$ the measured gain of reference training from the $`K`$ demonstrations.
- **(b) Signed identity.** For every history-dependent provider policy $`S`$ with the common initial law,

```math
L_E(S)-L_E(0)=\sum_{t=0}^{R-1}\mathbb E_{d^S_t}\,A_t(Z_t,J_t).
```

- **(c) Adaptive audit.** (i) With probability at least $`1-\delta`$, $`\sum_tA_t(Z_t,J_t)\ge\sum_t\Gamma_t`$. (ii) Unconditionally over audits and training,

```math
L_E(\mathrm{adaptive})-L_E(0)\ge-\delta,\qquad L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_t\Gamma_t-2\delta .
```

- **(d) Cost.** Radius $`e_t\le e`$ needs $`n_t\ge\lceil2\log(2M_t/\delta_t)/e^2\rceil`$ paired replicas per candidate. For $`M=4`$, $`\delta=0.05`$, $`e=0.1`$ this is $`\lceil2\log160/0.01\rceil=1016`$. The count is sufficient, not necessary.

*Proof.*

**(a)** $`L_E(S)-J_E(\pi_0)=[L_E(0)-J_E(\pi_0)]-[L_E(0)-L_E(S)]\ge\gamma-\varepsilon`$. Only a one-sided fidelity bound is used, so a candidate that improves on the reference is allowed.

**(b)** Along the $`S`$-process the tower property gives $`\mathbb E[V^0_{t+1}(Z_{t+1})\mid\text{history to }t]=(K^{J_t}_tV^0_{t+1})(Z_t)`$, so $`\mathbb E[V^0_{t+1}(Z_{t+1})-V^0_t(Z_t)]=\mathbb E\,A_t(Z_t,J_t)`$. Summing over $`t`$ telescopes; $`\mathbb EV^0_0(Z_0)=L_E(0)`$ by the common initial law and $`\mathbb EV^0_R(Z_R)=L_E(S)`$ since $`V^0_R=F_E`$.

**(c)** *Audit event.* Conditional on $`\mathcal H_t`$, the $`n_t`$ contrasts of candidate $`j`$ are i.i.d. in $`[-1,1]`$ with mean $`A_t(Z_t,j)`$; this uses the correct marginals, the full tail and exact restoration. Hoeffding's inequality for range $`2`$ gives

```math
\Pr\big(\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert>e_t\ \big\vert\ \mathcal H_t\big)\le2\exp(-n_te_t^2/2)=\delta_t/M_t .
```

- A union bound over the $`M_t`$ candidates gives an event $`\mathcal A_t`$ with $`\Pr(\mathcal A_t^c\mid\mathcal H_t)\le\delta_t`$, on which $`\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert\le e_t`$ for all $`j`$. The radius requires $`n_t`$ to be fixed before the outcomes of the current audit are seen.
- *Selected drift.* Write $`A_t:=A_t(Z_t,J_t)`$. If $`J_t=0`$, then $`A_t=0=\Gamma_t`$. Otherwise $`\Gamma_t=\widehat A_t(J_t)-e_t\ge0`$, and on $`\mathcal A_t`$, $`A_t\ge\widehat A_t(J_t)-e_t=\Gamma_t`$. At unaudited rounds $`A_t=\Gamma_t=0`$.
- *Claim (i).* Conditioning successively on $`\mathcal H_t`$, $`\Pr(\bigcup_t\mathcal A_t^c)\le\mathbb E\sum_t\Pr(\mathcal A_t^c\mid\mathcal H_t)\le\mathbb E\sum_t\delta_t\le\delta`$. On $`\bigcap_t\mathcal A_t`$, $`\sum_tA_t\ge\sum_t\Gamma_t`$. This holds at actually reached, random states, because each audit statement is conditional on its own pre-audit history.
- *Drift identity.* The actual update is a fresh draw from $`K^{J_t}_t`$, independent of the audit replicas given $`\mathcal F_t`$, and audits leave the state unchanged. Hence $`A_t=\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]-V^0_t(Z_t)`$. The differences $`D_{t+1}=V^0_{t+1}(Z_{t+1})-\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]`$ are martingale differences for the interleaved audit/update filtration, and telescoping gives

```math
F_E(Z_R)-V^0_0(Z_0)=\sum_{t=0}^{R-1}A_t+\sum_{t=0}^{R-1}D_{t+1}.
```

- *Claim (ii).* Always $`A_t\in[-1,1]`$ and $`\Gamma_t\in[0,1]`$, so $`A_t\ge-\mathbf 1_{\mathcal A_t^c}`$ and $`A_t\ge\Gamma_t-2\cdot\mathbf 1_{\mathcal A_t^c}`$ (on $`\mathcal A_t^c`$, $`A_t\ge-1\ge\Gamma_t-2`$). Taking expectations of the telescoped identity, with $`\mathbb EV^0_0(Z_0)=L_E(0)`$ and $`\mathbb E\sum_tD_{t+1}=0`$,

```math
L_E(\mathrm{adaptive})-L_E(0)=\mathbb E\sum_tA_t\ge-\mathbb E\sum_t\Pr(\mathcal A_t^c\mid\mathcal H_t)\ge-\delta ,
```

and likewise $`\mathbb E\sum_tA_t\ge\mathbb E\sum_t\Gamma_t-2\delta`$.

**(d)** $`e_t\le e`$ is equivalent to $`n_t\ge2\log(2M_t/\delta_t)/e^2`$. For $`M=4`$ and $`\delta=0.05`$, $`2\log160\approx10.150`$; dividing by $`0.01`$ and rounding up gives $`1016`$. Hoeffding's inequality is one sufficient choice; any valid fixed-$`n`$ confidence bound with a smaller radius can replace it. $`\square`$

*Comments on Theorem 8.3.*

- **The lower confidence bound is needed.** Selecting by the point estimate $`\widehat A_t(j)`$ can have strictly negative expected gain even with exact full-tail audits, and a slightly inferior candidate can pass a lower-confidence-bound gate with positive probability. The $`-\delta`$ term in (c)(ii) therefore cannot be removed: an exact no-degradation statement ($`\ge0`$) is false in general. An exact finite instance of the first fact is computed in the [experiments](../experiments/08_verification.md).
- **Unconditional, not conditional.** (c)(ii) averages over audit outcomes. Conditioning the terminal value on "all audits succeeded" is invalid in general, because later audit events depend on the states reached; an exact instance is computed in the [experiments](../experiments/08_verification.md).
- **Stopped refinement.** If $`\tau`$ is the first failed audit (capped at $`R`$), then $`L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_{t<\tau}\Gamma_t-\delta`$.
- **What is removed and what is not.** Auditing at the actual state removes the transport term of [Theorem 8.2](#s8-2). It does not remove the restoration, full-tail cost or independence requirements. Reusing a winning audit replica as the update, or auditing with a short tail, changes the estimand; a short tail needs an explicit surrogate allowance, which is what [Theorem 8.8](08_certificate.md#s8-8) supplies.
- **Fidelity is not improvement.** (a) inherits only the reference's measured gain $`\gamma`$. A high-fidelity copy of an ineffective reference does not improve the initial student.
