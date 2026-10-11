# N06 · Sufficiency duality and transfer — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N06_sufficiency_duality.md) · [中文](N06_sufficiency_duality_ZH.md)

## Setting and assumptions

- $`\mathcal Y`$ is a finite message alphabet with $`\lvert\mathcal Y\rvert=m\ge2`$. Functions on $`\mathcal Y`$ are vectors in $`\mathbb R^m`$. Probability laws are vectors in $`\Delta(\mathcal Y)=\lbrace P\in\mathbb R^m_{\ge0}:\mathbf 1^\top P=1\rbrace`$, and $`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$.
- $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ is a response class. In the framework, $`\mathcal G`$ collects the one-boundary responses $`g_{E,f,t,z}`$ of N05. Here only its values on $`\mathcal Y`$ matter.
- An interface transmits the means of fixed features $`\phi_1,\dots,\phi_d:\mathcal Y\to\mathbb R`$, and $`H_\phi=\mathrm{span}\lbrace \mathbf 1,\phi_1,\dots,\phi_d\rbrace \subseteq\mathbb R^m`$.
- $`H_\phi^\perp=\lbrace v\in\mathbb R^m:v^\top h=0\ \forall h\in H_\phi\rbrace`$ is the annihilator. Every $`v\in H_\phi^\perp`$ satisfies $`\mathbf 1^\top v=0`$.
- All probability laws on $`\mathcal Y`$ are admissible. Restricted law families are discussed in the Remarks.

**Definition 1 (feature-matched pairs, unresolved discrepancy).** Let $`\mathcal M_\phi=\lbrace (P,Q)\in\Delta(\mathcal Y)^2:\ (P-Q)\phi_j=0,\ j=1,\dots,d\rbrace`$. Define

```math
\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\ \sup_{g\in\mathcal G}\ \lvert(P-Q)g\rvert .
```

**Definition 2 (exact dimension).** $`d_0(\mathcal G)`$ is the least $`d`$ for which some features $`\phi_1,\dots,\phi_d`$ make $`(P,Q)\in\mathcal M_\phi\Rightarrow (P-Q)g=0`$ for all $`g\in\mathcal G`$.

## Statement

**Theorem.**

**(a) Exact sufficiency.** For all $`(P,Q)\in\mathcal M_\phi`$ and all $`g\in\mathcal G`$ we have $`(P-Q)g=0`$ if and only if $`\mathrm{span}(\mathcal G)\subseteq H_\phi`$. Consequently

```math
d_0(\mathcal G)=\dim\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)-1. \qquad(19)
```

**(b) Approximate sufficiency.**

```math
\mathfrak D_\phi(\mathcal G)=2\sup_{g\in\mathcal G}\ \inf_{h\in H_\phi}\lVert g-h\rVert_\infty. \qquad(20)
```

Hence the smallest unresolved discrepancy achievable with $`d`$ features is $`2\inf_{H}\sup_{g}\mathrm{dist}_\infty(g,H)`$. The infimum ranges over subspaces $`H\ni\mathbf 1`$ with $`\dim H\le d+1`$.

**(c) Transfer.** Suppose every $`g`$ in a target class satisfies $`g=c+\sum_{\ell=1}^db_\ell\phi_\ell+r`$ with $`\lVert b\rVert_1\le B`$ and $`\lVert r\rVert_\infty\le\tau`$. Then for all $`P,Q\in\Delta(\mathcal Y)`$,

```math
\sup_{g}\lvert(P-Q)g\rvert\le B\,\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau.
```

**(d)** **Corollary.** Fix a task with two policies of values $`a`$ and $`a+\gamma`$, where $`\gamma>0`$. Suppose the learner family contains, for every event $`A\subseteq\mathcal Y`$, the one-step rule "deploy the better policy exactly when $`Y\in A`$". Then

```math
\sup_{A\subseteq\mathcal Y}\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert=\gamma\,\mathrm{TV}(P,Q),\qquad d_0(\mathcal G)=m-1.
```

## Proof

### Lemma 1 (law differences)

```math
\lbrace P-Q:\ P,Q\in\Delta(\mathcal Y)\rbrace =\lbrace v\in\mathbb R^m:\ \mathbf 1^\top v=0,\ \lVert v\rVert_1\le2\rbrace .
```

*Proof.* ($`\subseteq`$) $`\mathbf 1^\top(P-Q)=1-1=0`$, and $`\lVert P-Q\rVert_1\le\lVert P\rVert_1+\lVert Q\rVert_1=2`$.

($`\supseteq`$) Let $`\mathbf 1^\top v=0`$ and $`\lVert v\rVert_1\le2`$. Write $`v=v_+-v_-`$ with $`v_\pm\ge0`$ and disjoint supports. Then $`s:=\mathbf 1^\top v_+=\mathbf 1^\top v_-=\tfrac12\lVert v\rVert_1\le1`$. Fix any $`\rho\in\Delta(\mathcal Y)`$ and set $`P=v_++(1-s)\rho`$ and $`Q=v_-+(1-s)\rho`$. Both are nonnegative with unit mass, and $`P-Q=v`$. $`\square`$

### Lemma 2 (distance to a subspace in the sup norm)

The norm is $`\ell_\infty`$.
For every $`g\in\mathbb R^m`$ and every subspace $`H\subseteq\mathbb R^m`$,

```math
\mathrm{dist}_\infty(g,H):=\min_{h\in H}\lVert g-h\rVert_\infty=\max\lbrace \sigma^\top g:\ \sigma\in H^\perp,\ \lVert\sigma\rVert_1\le1\rbrace ,
```

and both extrema are attained.

*Proof.* Let $`A\in\mathbb R^{m\times k}`$ have columns spanning $`H`$.

*Weak duality.* For $`\sigma\in H^\perp`$ with $`\lVert\sigma\rVert_1\le1`$ and any $`h\in H`$, we have $`\sigma^\top g=\sigma^\top(g-h)\le\lVert\sigma\rVert_1\lVert g-h\rVert_\infty\le\lVert g-h\rVert_\infty`$.

*Strong duality.* The primal is the linear program

```math
\min_{t\in\mathbb R,\,c\in\mathbb R^k}\ t\quad\text{s.t.}\quad g-Ac\le t\mathbf 1,\ \ -(g-Ac)\le t\mathbf 1.
```

It is feasible (take $`c=0`$, $`t=\lVert g\rVert_\infty`$) and bounded below by $`0`$. By LP strong duality, its optimum equals the optimum of the dual

```math
\max_{y^+,y^-\ge0}\ (y^+-y^-)^\top g\quad\text{s.t.}\quad A^\top(y^+-y^-)=0,\ \ \mathbf 1^\top(y^++y^-)=1
```

and both optima are attained. Put $`\sigma=y^+-y^-`$. Then $`\sigma\in H^\perp`$ and $`\lVert\sigma\rVert_1\le\mathbf 1^\top(y^++y^-)=1`$. Conversely, let $`\sigma\in H^\perp`$ with $`\lVert\sigma\rVert_1\le1`$. Take $`y^\pm=\sigma_\pm+\tfrac{1-\lVert\sigma\rVert_1}{2}e_1`$. This pair is dual-feasible with the same objective $`\sigma^\top g`$. So the dual value is exactly $`\max\lbrace \sigma^\top g:\sigma\in H^\perp,\lVert\sigma\rVert_1\le1\rbrace`$. $`\square`$

### Proof of (a)

($`\Leftarrow`$) Let $`\mathrm{span}\mathcal G\subseteq H_\phi`$ and $`g\in\mathcal G`$. Write $`g=c\mathbf 1+\sum_jb_j\phi_j`$. For $`(P,Q)\in\mathcal M_\phi`$:

```math
(P-Q)g=c\,\mathbf 1^\top(P-Q)+\sum_jb_j(P-Q)\phi_j=0.
```

($`\Rightarrow`$) Suppose some $`g\in\mathcal G`$ has $`g\notin H_\phi`$. Decompose $`\mathbb R^m=H_\phi\oplus H_\phi^\perp`$ orthogonally and let $`v`$ be the component of $`g`$ in $`H_\phi^\perp`$. Since $`g\notin H_\phi`$, $`v\ne0`$ and $`v^\top g=\lVert v\rVert_2^2>0`$. Let $`u=\mathbf 1/m`$ and set $`a=\tfrac1{2m\lVert v\rVert_\infty}`$. Define $`P=u`$ and $`Q=u+av`$. Then $`Q\ge u-\tfrac1{2m}\mathbf 1\ge0`$, and $`\mathbf 1^\top Q=1`$ because $`\mathbf 1\in H_\phi`$ gives $`\mathbf 1^\top v=0`$. Also $`(P-Q)\phi_j=-a\,v^\top\phi_j=0`$, so $`(P,Q)\in\mathcal M_\phi`$. But $`(P-Q)g=-a\lVert v\rVert_2^2\ne0`$.

(Count (19)) Let $`s=\dim\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)`$.

- *Upper bound.* Extend $`\lbrace \mathbf 1\rbrace`$ to a basis $`\lbrace \mathbf 1,\phi_1,\dots,\phi_{s-1}\rbrace`$ of $`\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)`$. Then $`H_\phi=\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)\supseteq\mathrm{span}\mathcal G`$, so $`s-1`$ features suffice by ($`\Leftarrow`$).
- *Lower bound.* If $`d`$ features suffice, then by ($`\Rightarrow`$) $`\mathrm{span}\mathcal G\subseteq H_\phi`$. Since also $`\mathbf 1\in H_\phi`$, we get $`\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)\subseteq H_\phi`$ and $`s\le\dim H_\phi\le d+1`$.

Hence $`d_0=s-1`$. $`\square`$

### Proof of (b)

Fix $`g\in\mathcal G`$ and write $`e(g)=\mathrm{dist}_\infty(g,H_\phi)`$.

*Upper bound.* Let $`(P,Q)\in\mathcal M_\phi`$. For every $`h\in H_\phi`$ we have $`(P-Q)h=0`$, since $`h`$ is a combination of $`\mathbf 1`$ and the $`\phi_j`$. Therefore

```math
\lvert(P-Q)g\rvert=\lvert(P-Q)(g-h)\rvert\le\lVert P-Q\rVert_1\,\lVert g-h\rVert_\infty\le2\lVert g-h\rVert_\infty .
```

Minimizing over $`h`$ gives $`\lvert(P-Q)g\rvert\le2e(g)`$.

*Lower bound (attainment).* By Lemma 2 there is $`\sigma\in H_\phi^\perp`$ with $`\lVert\sigma\rVert_1\le1`$ and $`\sigma^\top g=e(g)`$. Then $`v=2\sigma`$ satisfies $`\mathbf 1^\top v=0`$ (as $`\mathbf 1\in H_\phi`$) and $`\lVert v\rVert_1\le2`$. By Lemma 1, $`v=P-Q`$ for some $`P,Q\in\Delta(\mathcal Y)`$. Since $`v\perp\phi_j`$, the pair lies in $`\mathcal M_\phi`$, and $`(P-Q)g=2e(g)`$.

So $`\sup_{(P,Q)\in\mathcal M_\phi}\lvert(P-Q)g\rvert=2e(g)`$ for each $`g`$. Exchanging the two suprema in Definition 1 gives (20). The statement about $`d`$ features follows by minimizing (20) over admissible feature sets, that is, over subspaces $`H_\phi\ni\mathbf 1`$ of dimension at most $`d+1`$. $`\square`$

### Proof of (c)

Let $`\Delta\phi=\mathbb E_P\phi-\mathbb E_Q\phi\in\mathbb R^d`$. For $`g=c+b^\top\phi+r`$:

```math
(P-Q)g=\underbrace{c\,\mathbf 1^\top(P-Q)}_{=0}+\sum_\ell b_\ell\,\Delta\phi_\ell+(\mathbb E_Pr-\mathbb E_Qr).
```

By Hölder, $`\lvert\sum_\ell b_\ell\Delta\phi_\ell\rvert\le\lVert b\rVert_1\lVert\Delta\phi\rVert_\infty\le B\lVert\Delta\phi\rVert_\infty`$. Also $`\lvert\mathbb E_Pr-\mathbb E_Qr\rvert\le\lvert\mathbb E_Pr\rvert+\lvert\mathbb E_Qr\rvert\le2\tau`$. Take the supremum over $`g`$. $`\square`$

### Proof of the Corollary

The learner that deploys the better policy exactly on $`A`$ yields final value $`a+\gamma\mathbf 1_A(Y)`$, so $`g_A=a+\gamma\mathbf 1_A\in\mathcal G`$. Then

```math
(P-Q)g_A=\gamma\,(P(A)-Q(A)),
```

and $`\sup_A\lvert P(A)-Q(A)\rvert=\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$. This is the event characterization of total variation, attained at $`A=\lbrace y:P(y)>Q(y)\rbrace`$.

For the dimension count, the singleton events give $`g_{\lbrace y\rbrace }=a\mathbf 1+\gamma e_y`$. Since $`\gamma>0`$,

```math
\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)\supseteq\mathrm{span}\lbrace e_y:y\in\mathcal Y\rbrace =\mathbb R^m.
```

By (19), $`d_0=m-1`$. $`\square`$

## Remarks

1. **The quantifier in (b).** The supremum in (20) ranges over *all* pairs that share some common feature means. If the reference law $`P`$ is fixed in advance, or both laws must lie in one prescribed feature-mean fiber, the feasible set of differences shrinks. The dual is then taken over that smaller set, and (20) becomes an upper bound.
2. **Restricted law families.** If only a family $`\mathcal P\subsetneq\Delta(\mathcal Y)`$ is realizable, replace $`\lbrace P-Q\rbrace`$ in Lemma 1 by the differences of $`\mathcal P`$. Part (a) then reads: $`(P-Q)g=0`$ on feature-matched pairs if and only if $`g`$ is annihilated by every feature-matched difference.
3. **What is counted.** $`d_0`$ and $`d_\varepsilon`$ count fixed linear expectation statistics. They are not bit complexities, sample sizes or query counts. Estimating the statistics is a separate problem, treated in N13–N14.
4. **Coefficient stability.** Part (c) makes the coefficient size $`B`$ explicit. A small residual $`\tau`$ with large $`B`$ amplifies feature-estimation error. This is the reason for the stability–approximation tradeoff (N13).
5. **Necessity of the rich learner family in the Corollary.** $`d_0=m-1`$ needs the learners to realize all singleton indicators. Practical learners and tasks realize far fewer responses. The corollary shows only that no uniformly small interface can exist without restricting the task and learner class.

## Dependencies and source

- **Uses:** the response class $`\mathcal G`$ of [N05](N05_continuation_response.md) (continuation-response characterization).
- **Used by:** [N07](N07_affine_moment_order.md) (where $`\mathcal G`$ is the space of polynomials), [N09](N09_approximate_complexity.md) ($`d_\varepsilon`$ versus degree), [N13](N13_stability_tradeoff.md) (noisy features).

**Source:** [v0.14.1 §6](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#6-exact-and-approximate-information-complexity) — Thm. 8; Cor. 9; eqs. (18)–(22). v0.14.1 main framework, Thm. 8 and Cor. 9, eqs. (18)–(22), [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §6. Independent review: [Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md), A24, A25 and B1. LP duality and the event form of total variation are classical tools. They are credited as such and not claimed as new.
