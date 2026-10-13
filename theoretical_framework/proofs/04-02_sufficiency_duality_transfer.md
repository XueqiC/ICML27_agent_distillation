# Theorem 4.2 · Sufficiency duality and transfer

*Proof*

[← Framework](../README.md) · [Statement](../statements/04-02_sufficiency_duality_transfer.md) · [Proof](04-02_sufficiency_duality_transfer.md) · [Experiments](../experiments/04-02_sufficiency_duality_transfer.md) · [中文](04-02_sufficiency_duality_transfer_ZH.md)

**What is proved here.** If two feedback laws are compared only through the means of some features, they are guaranteed to give equal learning outcomes exactly when the features span every response; otherwise the worst hidden difference is twice a best uniform approximation error. The proof is linear algebra plus linear-programming duality: differences of probability laws are exactly the zero-sum vectors of $`\ell_1`$ norm at most 2, and the sup-norm distance to a subspace has an exact dual.

**Setting used on this page.**

- $`\mathcal Y`$ is a finite message alphabet with $`\lvert\mathcal Y\rvert=m\ge2`$. Functions on $`\mathcal Y`$ are vectors in $`\mathbb R^m`$; $`\mathbf 1`$ is the all-ones vector and $`e_y`$ the indicator of $`y`$. Laws are the vectors of $`\Delta(\mathcal Y)=\lbrace P\ge0:\mathbf 1^\top P=1\rbrace`$, and $`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$.
- $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ is a response class: each $`g\in\mathcal G`$ gives the expected final task return as a function of the message received in one round ([Definition 1.4](../README.md#s1-4)).
- A feature interface transmits the means of $`\phi_1,\dots,\phi_d`$. $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$, $`H_\phi^\perp`$ is its orthogonal complement, $`\mathcal M_\phi`$ is the set of feature-matched pairs, $`\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\sup_{g\in\mathcal G}\lvert(P-Q)g\rvert`$ is the unresolved discrepancy, and $`d_0(\mathcal G)`$ is the least number of features with zero discrepancy ([Definition 4.1](../statements/04-01_feature_sufficiency_unresolved_discrepancy.md#s4-1)).
- $`\mathrm{dist}_\infty(g,H)=\min_{h\in H}\lVert g-h\rVert_\infty`$; $`\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$.

**Statement ([Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2)).**

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

**Proof.** Two claims come first.

*Claim 1 (law differences).* $`\lbrace P-Q:P,Q\in\Delta(\mathcal Y)\rbrace=\lbrace v\in\mathbb R^m:\mathbf 1^\top v=0,\ \lVert v\rVert_1\le2\rbrace`$.

- ($`\subseteq`$) $`\mathbf 1^\top(P-Q)=1-1=0`$ and $`\lVert P-Q\rVert_1\le\lVert P\rVert_1+\lVert Q\rVert_1=2`$.
- ($`\supseteq`$) Let $`\mathbf 1^\top v=0`$ and $`\lVert v\rVert_1\le2`$. Write $`v=v_+-v_-`$ with $`v_\pm\ge0`$ of disjoint supports. Then $`s:=\mathbf 1^\top v_+=\mathbf 1^\top v_-=\tfrac12\lVert v\rVert_1\le1`$. Fix any $`\rho\in\Delta(\mathcal Y)`$ and put $`P=v_++(1-s)\rho`$, $`Q=v_-+(1-s)\rho`$. Both are nonnegative with unit mass, and $`P-Q=v`$.

*Claim 2 (sup-norm distance to a subspace).* For every $`g\in\mathbb R^m`$ and every subspace $`H\subseteq\mathbb R^m`$,

```math
\mathrm{dist}_\infty(g,H)=\max\lbrace\sigma^\top g:\ \sigma\in H^\perp,\ \lVert\sigma\rVert_1\le1\rbrace ,
```

and both extrema are attained. Let $`A\in\mathbb R^{m\times k}`$ have columns spanning $`H`$.

- *Weak duality.* For $`\sigma\in H^\perp`$ with $`\lVert\sigma\rVert_1\le1`$ and any $`h\in H`$, $`\sigma^\top g=\sigma^\top(g-h)\le\lVert\sigma\rVert_1\lVert g-h\rVert_\infty\le\lVert g-h\rVert_\infty`$.
- *Strong duality.* The distance is the value of the linear program

```math
\min_{t\in\mathbb R,\,c\in\mathbb R^k}\ t\quad\text{s.t.}\quad g-Ac\le t\mathbf 1,\ \ -(g-Ac)\le t\mathbf 1 .
```

- It is feasible ($`c=0`$, $`t=\lVert g\rVert_\infty`$) and bounded below by $`0`$. By LP strong duality ([Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)) its value equals that of the dual

```math
\max_{y^+,y^-\ge0}\ (y^+-y^-)^\top g\quad\text{s.t.}\quad A^\top(y^+-y^-)=0,\ \ \mathbf 1^\top(y^++y^-)=1 ,
```

- and both optima are attained. Put $`\sigma=y^+-y^-`$; then $`\sigma\in H^\perp`$ and $`\lVert\sigma\rVert_1\le\mathbf 1^\top(y^++y^-)=1`$. Conversely, for $`\sigma\in H^\perp`$ with $`\lVert\sigma\rVert_1\le1`$, the pair $`y^\pm=\sigma_\pm+\tfrac{1-\lVert\sigma\rVert_1}{2}e_1`$ is dual-feasible with objective $`\sigma^\top g`$. So the dual value is exactly the stated maximum.

*(a).* ($`\Leftarrow`$) Let $`\mathrm{span}\mathcal G\subseteq H_\phi`$ and $`g\in\mathcal G`$, so $`g=c\mathbf 1+\sum_jb_j\phi_j`$. For $`(P,Q)\in\mathcal M_\phi`$, $`(P-Q)g=c\,\mathbf 1^\top(P-Q)+\sum_jb_j(P-Q)\phi_j=0`$.

($`\Rightarrow`$) Suppose some $`g\in\mathcal G`$ has $`g\notin H_\phi`$. Let $`v`$ be the component of $`g`$ in $`H_\phi^\perp`$ under the orthogonal decomposition $`\mathbb R^m=H_\phi\oplus H_\phi^\perp`$.

- Since $`g\notin H_\phi`$, $`v\ne0`$ and $`v^\top g=\lVert v\rVert_2^2>0`$.
- Put $`u=\mathbf 1/m`$, $`a=1/(2m\lVert v\rVert_\infty)`$, $`P=u`$ and $`Q=u+av`$. Then $`Q\ge u-\tfrac1{2m}\mathbf 1\ge0`$, and $`\mathbf 1^\top Q=1`$ because $`\mathbf 1\in H_\phi`$ gives $`\mathbf 1^\top v=0`$.
- $`(P-Q)\phi_j=-a\,v^\top\phi_j=0`$, so $`(P,Q)\in\mathcal M_\phi`$, but $`(P-Q)g=-a\lVert v\rVert_2^2\ne0`$.

For the count (4.1), let $`s=\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)`$.

- *Upper bound.* Extend $`\lbrace\mathbf 1\rbrace`$ to a basis $`\lbrace\mathbf 1,\phi_1,\dots,\phi_{s-1}\rbrace`$ of $`\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)`$. Then $`H_\phi\supseteq\mathrm{span}\mathcal G`$, and $`s-1`$ features suffice by ($`\Leftarrow`$).
- *Lower bound.* If $`d`$ features suffice, ($`\Rightarrow`$) gives $`\mathrm{span}\mathcal G\subseteq H_\phi`$. Since $`\mathbf 1\in H_\phi`$, $`\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)\subseteq H_\phi`$ and $`s\le\dim H_\phi\le d+1`$.

*(b).* Fix $`g\in\mathcal G`$ and write $`e(g)=\mathrm{dist}_\infty(g,H_\phi)`$.

- *Upper bound.* Let $`(P,Q)\in\mathcal M_\phi`$. Every $`h\in H_\phi`$ is a combination of $`\mathbf 1`$ and the $`\phi_j`$, so $`(P-Q)h=0`$. Hence $`\lvert(P-Q)g\rvert=\lvert(P-Q)(g-h)\rvert\le\lVert P-Q\rVert_1\lVert g-h\rVert_\infty\le2\lVert g-h\rVert_\infty`$, and minimizing over $`h`$ gives $`\lvert(P-Q)g\rvert\le2e(g)`$.
- *Attainment.* By Claim 2 there is $`\sigma\in H_\phi^\perp`$ with $`\lVert\sigma\rVert_1\le1`$ and $`\sigma^\top g=e(g)`$. The vector $`v=2\sigma`$ satisfies $`\mathbf 1^\top v=0`$ and $`\lVert v\rVert_1\le2`$, so by Claim 1 $`v=P-Q`$ for some laws $`P,Q`$. Since $`v\perp\phi_j`$, $`(P,Q)\in\mathcal M_\phi`$, and $`(P-Q)g=2e(g)`$.

So $`\sup_{(P,Q)\in\mathcal M_\phi}\lvert(P-Q)g\rvert=2e(g)`$ for each $`g`$, and exchanging the two suprema in the definition of $`\mathfrak D_\phi`$ gives (4.2). The statement about $`d`$ features follows by minimizing (4.2) over subspaces $`H_\phi\ni\mathbf 1`$ of dimension at most $`d+1`$.

*(c).* Let $`\Delta\phi=\mathbb E_P\phi-\mathbb E_Q\phi\in\mathbb R^d`$. For $`g=c+b^\top\phi+r`$,

```math
(P-Q)g=\underbrace{c\,\mathbf 1^\top(P-Q)}_{=0}+\sum_\ell b_\ell\,\Delta\phi_\ell+(\mathbb E_Pr-\mathbb E_Qr).
```

By Hölder, $`\lvert\sum_\ell b_\ell\Delta\phi_\ell\rvert\le\lVert b\rVert_1\lVert\Delta\phi\rVert_\infty\le B\lVert\Delta\phi\rVert_\infty`$, and $`\lvert\mathbb E_Pr-\mathbb E_Qr\rvert\le2\tau`$. Take the supremum over $`g`$.

*(d).* The learner that deploys the better policy exactly on $`A`$ has final value $`a+\gamma\mathbf 1_A(Y)`$, so $`g_A=a+\gamma\mathbf 1_A\in\mathcal G`$ and $`(P-Q)g_A=\gamma(P(A)-Q(A))`$.

- $`\sup_A\lvert P(A)-Q(A)\rvert=\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$, attained at $`A=\lbrace y:P(y)>Q(y)\rbrace`$ (the event form of total variation).
- The singleton events give $`g_{\lbrace y\rbrace}=a\mathbf 1+\gamma e_y`$. Since $`\gamma>0`$, $`\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)\supseteq\mathrm{span}\lbrace e_y:y\in\mathcal Y\rbrace=\mathbb R^m`$, and (4.1) gives $`d_0=m-1`$. $`\square`$

## Remarks

- **The quantifier in (b).** The supremum in (4.2) ranges over all pairs that share some common feature means. If the reference law is fixed in advance, or both laws must lie in one prescribed feature-mean fiber, the feasible set of differences shrinks, the dual is taken over that smaller set, and (4.2) becomes an upper bound.
- **Restricted law families.** If only a family $`\mathcal P\subsetneq\Delta(\mathcal Y)`$ is realizable, replace the set of Claim 1 by the differences of $`\mathcal P`$. Part (a) then reads: $`(P-Q)g=0`$ on feature-matched pairs if and only if every feature-matched difference annihilates $`g`$.
- **Coefficient stability.** Part (c) makes the coefficient size $`B`$ explicit. A small residual $`\tau`$ with a large $`B`$ amplifies feature-estimation error. [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1) (the exact tradeoff between approximation residual and coefficient size under measurement tolerances) makes this precise.
- **Richness in (d).** $`d_0=m-1`$ needs learners that realize every singleton indicator. Practical learners and tasks realize far fewer responses. Part (d) shows only that no uniformly small interface exists unless the task and learner class are restricted.

## Related work

- LP duality and the event form of total variation are classical tools ([Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)); they are credited as such and not claimed as new.
- What is added is their use to count the information a feedback law must carry for a given response class: a span dimension or an approximation width.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-bertsimas1997introduction"></a>

**Dimitris Bertsimas, John N. Tsitsiklis (1997).** Introduction to Linear Optimization. Athena Scientific, Belmont, MA (Athena Scientific Series in Optimization and Neural Computation 6). ISBN 978-1-886529-19-9.
