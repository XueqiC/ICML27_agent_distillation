# N13 · Stability–approximation tradeoff — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N13_stability_tradeoff.md) · [中文](N13_stability_tradeoff_ZH.md)

## Setting and assumptions

- $\mathcal Y$ is a finite message alphabet with $\lvert\mathcal Y\rvert=m\ge2$. Functions on $\mathcal Y$ are vectors in $\mathbb R^m$, and probability laws are vectors in the simplex $\Delta(\mathcal Y)$.
- $g\in\mathbb R^m$ is a response (N05). $\phi_1,\dots,\phi_d\in\mathbb R^m$ are fixed features. We write $\Phi\in\mathbb R^{m\times d}$ for the matrix with columns $\phi_j$, and $\Phi b=\sum_jb_j\phi_j$.
- $\boldsymbol\epsilon=(\epsilon_1,\dots,\epsilon_d)$ are tolerances with $\epsilon_j\ge0$. Zero tolerances and linearly dependent features are allowed.
- The oscillation of $h\in\mathbb R^m$ is $\operatorname{osc}(h)=\max_yh(y)-\min_yh(y)$.

The **tolerant discrepancy** of $g$ is
$$W_{\boldsymbol\epsilon}(g)=\sup\Big\lbrace \lvert(P-Q)g\rvert:\ P,Q\in\Delta(\mathcal Y),\ \lvert(P-Q)\phi_j\rvert\le\epsilon_j\ \ (j=1,\dots,d)\Big\rbrace ,$$
where $(P-Q)h=\sum_y(P(y)-Q(y))h(y)$. It is the largest response difference between two message laws whose feature means differ by at most the declared tolerances. This is the situation after a finite audit that has measured each feature mean to accuracy $\epsilon_j$.

## Statement

**Theorem (N13; Thm. 18 of the framework).**
$$W_{\boldsymbol\epsilon}(g)=\min_{b\in\mathbb R^d,\ c\in\mathbb R}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big]=\min_{b\in\mathbb R^d}\Big[\operatorname{osc}(g-\Phi b)+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big].\tag{36}$$
Both minima are attained. In particular $W_{\boldsymbol\epsilon}(g)\le\operatorname{osc}(g)$, which is at most $1$ when $g\in[0,1]^m$.

**Corollary 1 (exact case).** For $\boldsymbol\epsilon=0$, $W_0(g)=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty$ with $H_\phi=\operatorname{span}\lbrace \mathbf 1,\phi_1,\dots,\phi_d\rbrace $. This is the single-response form of Thm. 8(b) (N06).

**Corollary 2 (stability witness).** Let $g\in\lbrace 0,1\rbrace ^m$ be non-constant, let $\phi_1=\alpha g$ with $\alpha>0$, and let $d=1$. Then $W_\epsilon(g)=\min\lbrace 1,\epsilon/\alpha\rbrace $.

**Corollary 3 (finite measurement bridge).** Fix a simultaneous event on which every audited feature mean satisfies $\lvert\widehat\mu_{P,j}-\mu_{P,j}\rvert\le e_{P,j}$ and $\lvert\widehat\mu_{Q,j}-\mu_{Q,j}\rvert\le e_{Q,j}$. Set $\epsilon_j=\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}$. Then on that event $\lvert(P-Q)g\rvert\le W_{\boldsymbol\epsilon}(g)$. If moreover $\lVert\widehat g-g\rVert_\infty\le\eta$, then $\lvert(P-Q)g\rvert\le W_{\boldsymbol\epsilon}(\widehat g)+2\eta$.

## Proof

**Lemma 1 (difference set).** $\lbrace P-Q:P,Q\in\Delta(\mathcal Y)\rbrace =V:=\lbrace v\in\mathbb R^m:\mathbf 1^\top v=0,\ \lVert v\rVert_1\le2\rbrace $.

*Proof.* If $v=P-Q$, then $\mathbf 1^\top v=1-1=0$ and $\lVert v\rVert_1\le\lVert P\rVert_1+\lVert Q\rVert_1=2$. Conversely, let $v\in V$ and write $v=v^+-v^-$ with $v^\pm\ge0$ of disjoint supports. Since $\mathbf 1^\top v=0$, both parts have the same mass $s=\mathbf 1^\top v^+=\mathbf 1^\top v^-$. Moreover $2s=\lVert v\rVert_1\le2$, so $s\le1$. Fix any $r\in\Delta(\mathcal Y)$ and put $P=v^++(1-s)r$ and $Q=v^-+(1-s)r$. Both are nonnegative with unit mass, and $P-Q=v$. $\square$

**Lemma 2 (support function of $V$).** For every $h\in\mathbb R^m$, $\max_{v\in V}h^\top v=\operatorname{osc}(h)$.

*Proof.* Take $v\in V$ with parts $v^\pm$ of common mass $s\le1$ as above. Then $h^\top v=h^\top v^+-h^\top v^-\le s\max h-s\min h=s\operatorname{osc}(h)\le\operatorname{osc}(h)$, because $\operatorname{osc}(h)\ge0$. Equality holds at $v=e_{y^+}-e_{y^-}$, where $y^\pm$ are a maximizer and a minimizer of $h$. $\square$

**Lemma 3 (oscillation and centered sup-norm).** $\operatorname{osc}(h)=2\min_{c\in\mathbb R}\lVert h-c\mathbf 1\rVert_\infty$, attained at $c^\star=(\max h+\min h)/2$.

*Proof.* For any $c$, $\max h-\min h=(\max h-c)+(c-\min h)\le2\lVert h-c\mathbf1\rVert_\infty$. At $c^\star$ both summands equal $\operatorname{osc}(h)/2$, so $\lVert h-c^\star\mathbf 1\rVert_\infty=\operatorname{osc}(h)/2$. $\square$

**Step 1 (reduction to a linear program).** By Lemma 1,
$$W_{\boldsymbol\epsilon}(g)=\sup\lbrace \lvert g^\top v\rvert:\ v\in V,\ -\boldsymbol\epsilon\le\Phi^\top v\le\boldsymbol\epsilon\rbrace .$$
The feasible set $F$ is invariant under $v\mapsto-v$, so the absolute value can be dropped: $W_{\boldsymbol\epsilon}(g)=\sup_{v\in F}g^\top v$. The set $F$ is a nonempty polytope, since it contains $v=0$ and is bounded by $\lVert v\rVert_1\le2$. The supremum is therefore a finite maximum.

**Step 2 (weak duality).** Fix $b\in\mathbb R^d$. For every $v\in F$,
$$g^\top v=(g-\Phi b)^\top v+b^\top\Phi^\top v\le\operatorname{osc}(g-\Phi b)+\sum_j\lvert b_j\rvert\,\lvert\phi_j^\top v\rvert\le\operatorname{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert,$$
using Lemma 2 for the first term and the feature constraints for the second. Taking the maximum over $v$ and the infimum over $b$ gives
$$W_{\boldsymbol\epsilon}(g)\le\inf_b\Big[\operatorname{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\Big].$$

**Step 3 (strong duality).** Introduce multipliers $u,w\in\mathbb R^d_{\ge0}$ for the constraints $\Phi^\top v\le\boldsymbol\epsilon$ and $-\Phi^\top v\le\boldsymbol\epsilon$, and keep $v\in V$ as the domain. The Lagrangian is
$$\mathcal L(v;u,w)=g^\top v-u^\top(\Phi^\top v-\boldsymbol\epsilon)-w^\top(-\Phi^\top v-\boldsymbol\epsilon)=(g-\Phi(u-w))^\top v+\boldsymbol\epsilon^\top(u+w).$$
$\mathcal L$ is bilinear, $V$ is compact and convex, and $\mathbb R^{2d}_{\ge0}$ is convex. Sion's minimax theorem therefore gives
$$\max_{v\in V}\ \inf_{u,w\ge0}\mathcal L=\inf_{u,w\ge0}\ \max_{v\in V}\mathcal L.$$
The left side equals $\max_{v\in F}g^\top v$: the inner infimum is $g^\top v$ if $v\in F$ and $-\infty$ otherwise. By Lemma 2 the right side is $\inf_{u,w\ge0}[\operatorname{osc}(g-\Phi(u-w))+\boldsymbol\epsilon^\top(u+w)]$. For fixed $b=u-w$, the representation cost satisfies $\boldsymbol\epsilon^\top(u+w)\ge\sum_j\epsilon_j\lvert b_j\rvert$, because $u_j+w_j\ge\lvert u_j-w_j\rvert$. Equality is attained at $u=b^+$ and $w=b^-$. Hence
$$W_{\boldsymbol\epsilon}(g)=\inf_{b\in\mathbb R^d}\Big[\operatorname{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\Big].$$

**Step 4 (attainment).** The objective $f(b)=\operatorname{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert$ is a finite maximum of affine functions of $b$, so it is a convex polyhedral function. It is bounded below by $W_{\boldsymbol\epsilon}(g)\ge0$ (Step 2). A polyhedral convex function that is bounded below attains its infimum: minimizing it is a feasible linear program in $(b,t,s)$ with $t\ge(g-\Phi b)(y)-(g-\Phi b)(y')$ for all $y,y'$ and $s_j\ge\pm b_j$, and that program has a finite optimum. So the minimum is attained.

**Step 5 (second form).** By Lemma 3, $\operatorname{osc}(g-\Phi b)=2\min_c\lVert g-c\mathbf 1-\Phi b\rVert_\infty$. Substituting this into Step 4 gives the first expression in (36). Choosing $b=0$ gives $W_{\boldsymbol\epsilon}(g)\le\operatorname{osc}(g)$. $\square$

**Proof of Corollary 1.** With $\boldsymbol\epsilon=0$ the penalty vanishes. Then $\min_{b,c}2\lVert g-c\mathbf 1-\Phi b\rVert_\infty=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty$, because $c\mathbf 1+\Phi b$ ranges over $H_\phi$. $\square$

**Proof of Corollary 2.** Here $g-b\alpha g=(1-\alpha b)g$ and $\operatorname{osc}((1-\alpha b)g)=\lvert1-\alpha b\rvert$ because $g$ takes both values $0$ and $1$. So $f(b)=\lvert1-\alpha b\rvert+\epsilon\lvert b\rvert$. This function is convex and piecewise linear with kinks at $b=0$ and $b=1/\alpha$. On $b<0$ it increases as $b$ decreases, and on $b>1/\alpha$ it increases in $b$. On $[0,1/\alpha]$ it is affine. Hence $\min_bf(b)=\min\lbrace f(0),f(1/\alpha)\rbrace =\min\lbrace 1,\epsilon/\alpha\rbrace $. $\square$

**Proof of Corollary 3.** On the event, $\lvert(P-Q)\phi_j\rvert\le\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}=\epsilon_j$ by the triangle inequality. Hence $(P,Q)$ is feasible in the definition of $W_{\boldsymbol\epsilon}(g)$. For the second claim, $\lvert(P-Q)g\rvert\le\lvert(P-Q)\widehat g\rvert+\lvert(P-Q)(g-\widehat g)\rvert\le W_{\boldsymbol\epsilon}(\widehat g)+2\eta$, since each of $\lvert P(g-\widehat g)\rvert$ and $\lvert Q(g-\widehat g)\rvert$ is at most $\eta$. The coefficients $b$ may be optimized after the feature bounds are observed, because the event controls every linear combination of the fixed features simultaneously. $\square$

## Remarks

1. **Meaning.** The term $\operatorname{osc}(g-\Phi b)$ is the approximation error. The term $\sum_j\epsilon_j\lvert b_j\rvert$ is the price of measurement error. A feature set that represents $g$ exactly but needs large coefficients is a poor certificate: Corollary 2 gives $W=\epsilon/\alpha$, which grows as the feature scale $\alpha$ shrinks. The bound (36) is the exact best tradeoff between the two terms.
2. **Sharpness.** (36) is an equality over unrestricted pairs of laws. For a fixed reference $P$, a restricted family of realizable laws, or fixed-marginal couplings (N16), the right side remains an upper bound but may not be attained.
3. **Use downstream.** N14 uses the linear term $B_t(\cdot)$ with $B_t\ge\lVert b\rVert_1$. Optimizing (36) is the finite-LP fitting objective of CED rule 4 (N17).

## Dependencies and source

- Uses N06 (the approximation duality, recovered as Corollary 1) and N09 (approximation width rather than degree).
- Source: `docs/direction/v015_revision/Environment_Information_Theory_EN.md` §10, Thm. 18, eq. (36). Independent review: `docs/direction/v014_handoff/Independent_Proof_Audit_EN.md` §D1.
- Classical LP/norm duality applied to the operational discrepancy. No new duality principle is claimed.
