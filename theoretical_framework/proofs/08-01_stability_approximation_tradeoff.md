# Theorem 8.1 · Stability–approximation tradeoff

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-01_stability_approximation_tradeoff.md) · [Proof](../proofs/08-01_stability_approximation_tradeoff.md) · [Experiments](../experiments/08-01_stability_approximation_tradeoff.md) · [中文](08-01_stability_approximation_tradeoff_ZH.md)

**What is proved here.** The tolerant discrepancy is computed exactly as a linear-programming dual; the minimum is attained; the zero-tolerance case, a stability witness and the bridge to measured feature means follow.

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

**Statement ([Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1)).**

**(a) Exact tradeoff.** Both minima below are attained, and

```math
W_{\mathbf{\epsilon}}(g)=\min_{b\in\mathbb R^d,\ c\in\mathbb R}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big]=\min_{b\in\mathbb R^d}\Big[\mathrm{osc}(g-\Phi b)+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big].
```

In particular $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$, which is at most $`1`$ when $`g\in[0,1]^m`$.

- **(b) Exact case.** For $`\mathbf{\epsilon}=0`$, $`W_0(g)=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$ with $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$. This is the single-response form of [Theorem 4.2(b)](../statements/04-02_sufficiency_duality_transfer.md#s4-2).
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

The feasible set $`\mathcal F`$ is invariant under $`v\mapsto-v`$, so the absolute value can be dropped. $`\mathcal F`$ is a nonempty polytope (it contains $`0`$ and is bounded), so the supremum is a maximum. Steps 2–3 establish weak and strong duality for this program; for linear programs these are the classical duality theorems ([Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)).

*Step 2 (weak duality).* Fix $`b\in\mathbb R^d`$. For every $`v\in\mathcal F`$, by Claim 2 and the feature constraints,

```math
g^\top v=(g-\Phi b)^\top v+b^\top\Phi^\top v\le\mathrm{osc}(g-\Phi b)+\sum_j\lvert b_j\rvert\,\lvert\phi_j^\top v\rvert\le\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert .
```

Taking the maximum over $`v`$ and the infimum over $`b`$ gives $`W_{\mathbf{\epsilon}}(g)\le\inf_b\big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\big]`$.

*Step 3 (strong duality).* Introduce multipliers $`u,w\in\mathbb R^d_{\ge0}`$ for $`\Phi^\top v\le\mathbf{\epsilon}`$ and $`-\Phi^\top v\le\mathbf{\epsilon}`$, keeping $`v\in\mathcal D`$ as the domain. The Lagrangian is

```math
\mathcal L(v;u,w)=g^\top v-u^\top(\Phi^\top v-\mathbf{\epsilon})-w^\top(-\Phi^\top v-\mathbf{\epsilon})=\big(g-\Phi(u-w)\big)^\top v+\mathbf{\epsilon}^\top(u+w).
```

- $`\mathcal L`$ is bilinear, $`\mathcal D`$ is compact and convex, and $`\mathbb R^{2d}_{\ge0}`$ is convex. Sion's minimax theorem ([Sion, 1958](../references.md#sion1958general)) gives $`\max_{v\in\mathcal D}\inf_{u,w\ge0}\mathcal L=\inf_{u,w\ge0}\max_{v\in\mathcal D}\mathcal L`$.
- The left side equals $`\max_{v\in\mathcal F}g^\top v`$, since the inner infimum is $`g^\top v`$ on $`\mathcal F`$ and $`-\infty`$ off it.
- By Claim 2 the right side is $`\inf_{u,w\ge0}\big[\mathrm{osc}(g-\Phi(u-w))+\mathbf{\epsilon}^\top(u+w)\big]`$. For fixed $`b=u-w`$, $`\mathbf{\epsilon}^\top(u+w)\ge\sum_j\epsilon_j\lvert b_j\rvert`$ because $`u_j+w_j\ge\lvert u_j-w_j\rvert`$, with equality at $`u=b^+`$, $`w=b^-`$.

Hence $`W_{\mathbf{\epsilon}}(g)=\inf_b\big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\big]`$.

*Step 4 (attainment).* The objective $`f(b)=\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$ is a finite maximum of affine functions, hence convex polyhedral, and it is bounded below by $`0`$. Minimizing it is a feasible linear program in $`(b,t,s)`$ with $`t\ge(g-\Phi b)(y)-(g-\Phi b)(y')`$ for all $`y,y'`$ and $`s_j\ge\pm b_j`$. A feasible linear program with a bounded-below objective has a finite optimum, so the minimum is attained.

*Step 5 (first form).* By Claim 3, $`\mathrm{osc}(g-\Phi b)=2\min_c\lVert g-c\mathbf 1-\Phi b\rVert_\infty`$; substituting gives the first expression in (a). Choosing $`b=0`$ gives $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$.

*Proof of (b).* With $`\mathbf{\epsilon}=0`$ the penalty vanishes, and $`c\mathbf 1+\Phi b`$ ranges over $`H_\phi`$.

*Proof of (c).* Here $`g-b\alpha g=(1-\alpha b)g`$, and $`\mathrm{osc}((1-\alpha b)g)=\lvert1-\alpha b\rvert`$ because $`g`$ takes both values $`0`$ and $`1`$. So $`f(b)=\lvert1-\alpha b\rvert+\epsilon\lvert b\rvert`$. It is convex and piecewise linear with kinks at $`0`$ and $`1/\alpha`$, increasing away from $`[0,1/\alpha]`$ and affine on it. Hence $`\min_bf(b)=\min\lbrace f(0),f(1/\alpha)\rbrace=\min\lbrace1,\epsilon/\alpha\rbrace`$.

*Proof of (d).* On the event, the triangle inequality gives $`\lvert(P-Q)\phi_j\rvert\le\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}=\epsilon_j`$, so $`(P,Q)`$ is feasible in the definition of $`W_{\mathbf{\epsilon}}(g)`$. For the second claim, $`\lvert(P-Q)g\rvert\le\lvert(P-Q)\widehat g\rvert+\lvert P(g-\widehat g)\rvert+\lvert Q(g-\widehat g)\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$. The coefficients $`b`$ may be optimized after the measurements are seen, because the event controls every linear combination of the fixed features simultaneously. $`\square`$

## References

- [Sion, 1958](../references.md#sion1958general). Sion, M. (1958). On general minimax theorems. Pacific Journal of Mathematics 8, no. 1, 171–176.
