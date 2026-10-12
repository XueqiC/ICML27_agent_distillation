# §4 Complexity of sufficiency — Proofs

[← Framework](../README.md) · [Experiments](../experiments/04_complexity_of_sufficiency.md) · [中文](04_complexity_of_sufficiency_ZH.md)

**Summary.**

- §3 fixed the object of sufficiency: two feedback laws are interchangeable exactly when they agree on every response in the response class $`\mathcal G`$ ([Definition 1.4](../README.md#s1-4), [Theorem 3.2](03_learning_consequence.md#s3-2)). This section measures how much information that agreement requires.
- [Theorem 4.2](#s4-2) is the general answer. Exact sufficiency of a feature interface is a span condition; its count is $`\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1`$. Approximate sufficiency is twice a best uniform approximation error, and a decomposition with bounded coefficients transfers feature-mean errors into response errors.
- [Theorems 4.4](#s4-4) and [4.5](#s4-5) compute everything for convex-affine learners on binary-context students. The exact information is the joint moments up to the number $`N`$ of deployment decisions, every order is necessary, the gap bound does not depend on the number of training rounds, and the gap decays with the number of independent blocks at a rate fixed by the first unmatched cumulant. Replay and deterministic descendants do not decay.
- [Proposition 4.6](#s4-6) separates exact order from approximate complexity: about $`\sqrt{N\log(1/\varepsilon)}`$ moments resolve $`p^N`$ to accuracy $`\varepsilon`$, while parity cannot be approximated below $`1/2`$ by any degree below $`N`$.
- [Theorem 4.7](#s4-7) shows that the learner sets the order. Composed polynomial updates multiply the degree, a threshold learner breaks first-moment sufficiency with one decision, and one smooth SGD step has the mean gradient as its first-order statistic and the gradient second moment as its second-order statistic. [Proposition 5.1](05_optimizer_state.md#s5-1) extends the last statement to arbitrary stateful optimizers.

## Feature sufficiency

Throughout this section, $`\mathcal Y`$ is a finite message alphabet with $`\lvert\mathcal Y\rvert=m\ge2`$.

- Functions on $`\mathcal Y`$ are vectors in $`\mathbb R^m`$. Probability laws are vectors in $`\Delta(\mathcal Y)=\lbrace P\in\mathbb R^m_{\ge0}:\mathbf 1^\top P=1\rbrace`$, and $`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$.
- $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ is a response class. In the framework it collects the one-boundary responses of [Definition 1.4](../README.md#s1-4); only their values on $`\mathcal Y`$ matter here.
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

These dimensions count fixed linear expectation statistics. They are not bit complexities, sample sizes or query counts; estimating the statistics from finite data is the subject of §8.

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

  Hence the smallest unresolved discrepancy achievable with $`d`$ features is $`2\inf_H\sup_{g\in\mathcal G}\mathrm{dist}_\infty(g,H)`$, the infimum ranging over subspaces $`H\ni\mathbf 1`$ with $`\dim H\le d+1`$.
- **(c) Transfer.** Suppose every $`g`$ in a target class can be written $`g=c+\sum_{\ell=1}^db_\ell\phi_\ell+r`$ with $`\lVert b\rVert_1\le B`$ and $`\lVert r\rVert_\infty\le\tau`$. Then for all $`P,Q\in\Delta(\mathcal Y)`$,

```math
\sup_g\lvert(P-Q)g\rvert\le B\,\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau .
```

- **(d) Unrestricted learners.** Fix a task with two policies of values $`a`$ and $`a+\gamma`$, $`\gamma>0`$. Suppose the learner family contains, for every event $`A\subseteq\mathcal Y`$, the one-step rule "deploy the better policy exactly when $`Y\in A`$". Then

```math
\sup_{A\subseteq\mathcal Y}\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert=\gamma\,\mathrm{TV}(P,Q),\qquad d_0(\mathcal G)=m-1 .
```

*Proof.* Two claims come first.

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

- It is feasible ($`c=0`$, $`t=\lVert g\rVert_\infty`$) and bounded below by $`0`$. By LP strong duality its value equals that of the dual

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

So $`\sup_{(P,Q)\in\mathcal M_\phi}\lvert(P-Q)g\rvert=2e(g)`$ for each $`g`$, and exchanging the two suprema in Definition 4.1 gives (4.2). The statement about $`d`$ features follows by minimizing (4.2) over subspaces $`H_\phi\ni\mathbf 1`$ of dimension at most $`d+1`$.

*(c).* Let $`\Delta\phi=\mathbb E_P\phi-\mathbb E_Q\phi\in\mathbb R^d`$. For $`g=c+b^\top\phi+r`$,

```math
(P-Q)g=\underbrace{c\,\mathbf 1^\top(P-Q)}_{=0}+\sum_\ell b_\ell\,\Delta\phi_\ell+(\mathbb E_Pr-\mathbb E_Qr).
```

By Hölder, $`\lvert\sum_\ell b_\ell\Delta\phi_\ell\rvert\le\lVert b\rVert_1\lVert\Delta\phi\rVert_\infty\le B\lVert\Delta\phi\rVert_\infty`$, and $`\lvert\mathbb E_Pr-\mathbb E_Qr\rvert\le2\tau`$. Take the supremum over $`g`$.

*(d).* The learner that deploys the better policy exactly on $`A`$ has final value $`a+\gamma\mathbf 1_A(Y)`$, so $`g_A=a+\gamma\mathbf 1_A\in\mathcal G`$ and $`(P-Q)g_A=\gamma(P(A)-Q(A))`$.

- $`\sup_A\lvert P(A)-Q(A)\rvert=\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$, attained at $`A=\lbrace y:P(y)>Q(y)\rbrace`$ (the event form of total variation).
- The singleton events give $`g_{\lbrace y\rbrace}=a\mathbf 1+\gamma e_y`$. Since $`\gamma>0`$, $`\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)\supseteq\mathrm{span}\lbrace e_y:y\in\mathcal Y\rbrace=\mathbb R^m`$, and (4.1) gives $`d_0=m-1`$. $`\square`$

*Remarks on Theorem 4.2.*

- **The quantifier in (b).** The supremum in (4.2) ranges over all pairs that share some common feature means. If the reference law is fixed in advance, or both laws must lie in one prescribed feature-mean fiber, the feasible set of differences shrinks, the dual is taken over that smaller set, and (4.2) becomes an upper bound.
- **Restricted law families.** If only a family $`\mathcal P\subsetneq\Delta(\mathcal Y)`$ is realizable, replace the set of Claim 1 by the differences of $`\mathcal P`$. Part (a) then reads: $`(P-Q)g=0`$ on feature-matched pairs if and only if every feature-matched difference annihilates $`g`$.
- **Coefficient stability.** Part (c) makes the coefficient size $`B`$ explicit. A small residual $`\tau`$ with a large $`B`$ amplifies feature-estimation error. [Theorem 8.1](08_verification.md#s8-1) makes this tradeoff exact.
- **Richness in (d).** $`d_0=m-1`$ needs learners that realize every singleton indicator. Practical learners and tasks realize far fewer responses. Part (d) shows only that no uniformly small interface exists unless the task and learner class are restricted.

## Affine learners

<a id="s4-3"></a>

**Definition 4.3 (Binary-context student and convex-affine learner).**

- **Student.** The student has $`k`$ binary-action contexts and a parameter $`p\in[0,1]^k`$. At a decision with context $`j`$ it takes action 1 with conditional probability $`p_j`$, given the entire past.
- **Deployment.** The deployment is a finite tree with three kinds of nodes:
  - *chance nodes*, whose transition probabilities do not depend on $`p`$;
  - *decision nodes*, each labelled by a context $`j`$ that may depend on the history but not on $`p`$ directly;
  - *terminal nodes*, carrying rewards in $`[0,1]`$ that do not depend on $`p`$.
- Every root-to-leaf path contains at most $`N`$ decision nodes, and a context may recur along a path. $`J_E(p)`$ is the expected terminal reward.
- **Experience.** An experience block is $`Z\in[0,1]^k`$, with arbitrarily dependent coordinates. Blocks $`Z_1,\dots,Z_R`$ are IID from $`P`$ (or from $`Q`$).
- **Convex-affine learner.** With $`p_0\in[0,1]^k`$ common and fixed and weights fixed independently of the data,

```math
p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i,\qquad w_i\ge0,\quad\sum_{i=0}^Rw_i=1 .
```

- **Notation.** For $`\alpha\in\mathbb N^k`$, $`p^\alpha=\prod_jp_j^{\alpha_j}`$, $`\lvert\alpha\rvert=\sum_j\alpha_j`$, and $`m_P(\alpha)=\mathbb E_PZ^\alpha`$. For a polynomial $`f=\sum_\alpha c_\alpha p^\alpha`$, $`\lVert f\rVert_{\rm coef}=\sum_\alpha\lvert c_\alpha\rvert`$.

Then $`p_R\in[0,1]^k`$, and the learning value of a source is $`\mathbb E\,J_E(p_R)`$.

<a id="s4-4"></a>

**Theorem 4.4 (Affine learners: moment order is necessary and sufficient).** In the setting of Definition 4.3:

- **(a) Polynomial deployment value.** $`J_E`$ has a unique representation

```math
J_E(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha,\qquad\sum_\alpha\lvert c_\alpha\rvert\le3^N, \qquad(4.3)
```

  and $`J_E(p)\in[0,1]`$ on $`[0,1]^k`$.
- **(b) Necessary and sufficient joint order.** The following are equivalent:
  - $`m_P(\alpha)=m_Q(\alpha)`$ for every $`\lvert\alpha\rvert\le N`$;
  - for every deployment with at most $`N`$ decisions and every convex-affine learner, $`\mathbb E_{P^R}J_E(p_R)=\mathbb E_{Q^R}J_E(p_R)`$.
- **(c) Every order is necessary.**
  - *Scalar witness.* Let $`N\ge1`$ and $`z_i=i/N`$, $`i=0,\dots,N`$. Put $`P_N(z_i)=\binom Ni/2^{N-1}`$ for even $`i`$ (and $`0`$ for odd $`i`$), and $`Q_N(z_i)=\binom Ni/2^{N-1}`$ for odd $`i`$ (and $`0`$ for even $`i`$). Then $`m_{P_N}(\ell)=m_{Q_N}(\ell)`$ for $`0\le\ell<N`$, while

```math
\Delta_N:=m_{P_N}(N)-m_{Q_N}(N)=\frac{(-1)^NN!}{2^{N-1}N^N},
```

  - and the task $`J(p)=p^N`$ with the one-update learner $`p=Z`$ realizes the gap $`\Delta_N`$.
  - *Parity witness.* Let $`k=N`$, let $`P`$ be uniform on the even-parity vectors of $`\lbrace0,1\rbrace^N`$ and $`Q`$ uniform on the odd-parity vectors. All joint moments of total degree below $`N`$ agree. With $`J_{\rm even}(p)=\tfrac12\big(1+\prod_{j=1}^N(1-2p_j)\big)`$, the one-update learner $`p=Z`$ gives $`\mathbb E_PJ_{\rm even}(Z)=1`$ and $`\mathbb E_QJ_{\rm even}(Z)=0`$.
- **(d) Exact statistic counts and the binary barrier.**
  - On the grid $`\mathcal Y=\lbrace0,\tfrac1N,\dots,1\rbrace^k`$, the least number of linear expectation statistics of a block that is sufficient for all tasks and learners of (b) is $`\binom{k+N}{N}-1`$.
  - On $`\mathcal Y=\lbrace0,1\rbrace^k`$, with $`N\ge k`$ and arbitrary terminal rewards on the $`k`$-action vector, every feature interface with fewer than $`2^k-1`$ statistics has unresolved discrepancy exactly $`1`$. Tolerating any discrepancy $`\varepsilon<1`$ therefore does not reduce this worst-case dimension; $`2^k-1`$ affinely independent features give discrepancy $`0`$.

*Proof.* Three claims are used.

*Claim 3 (coefficient norm under elementary operations).* Let $`f,g`$ be polynomials in $`p`$ and $`j\in\lbrace1,\dots,k\rbrace`$.

- $`\lVert p_jf\rVert_{\rm coef}=\lVert f\rVert_{\rm coef}`$, because $`p^\beta\mapsto p^{\beta+e_j}`$ maps distinct monomials to distinct monomials.
- $`\lVert f+g\rVert_{\rm coef}\le\lVert f\rVert_{\rm coef}+\lVert g\rVert_{\rm coef}`$.
- For weights $`\pi_e\ge0`$ with $`\sum_e\pi_e=1`$, $`\lVert\sum_e\pi_ef_e\rVert_{\rm coef}\le\max_e\lVert f_e\rVert_{\rm coef}`$.

*Claim 4 (block-moment expansion of an affine learner).* Let $`\lvert\alpha\rvert=a`$ and list the coordinate factors of $`p_R^\alpha`$ as $`j(1),\dots,j(a)`$, coordinate $`j`$ appearing $`\alpha_j`$ times. Then

```math
p_R^\alpha=\sum_{s:\lbrace1,\dots,a\rbrace\to\lbrace0,\dots,R\rbrace}\Big(\prod_{l=1}^aw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^RZ_i^{\beta_i(s)},\qquad\beta_i(s)=\sum_{l:s(l)=i}e_{j(l)} .
```

- The coefficients $`\prod_lw_{s(l)}`$ are nonnegative and sum to $`\prod_l\sum_iw_i=1`$.
- For each $`s`$, $`\sum_i\lvert\beta_i(s)\rvert\le a`$, so at most $`a`$ blocks have $`\beta_i(s)\ne0`$.
- Under IID blocks,

```math
\mathbb E\,p_R^\alpha=\sum_s\Big(\prod_lw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^Rm(\beta_i(s)). \qquad(4.4)
```

The expansion is distributivity applied to the product of the $`a`$ linear forms $`w_0p_{0,j(l)}+\sum_{i\ge1}w_iZ_{i,j(l)}`$, choosing one source $`s(l)`$ for each factor; independence across blocks factors the expectation of each product of block monomials.

*Claim 5 (grid independence).* On $`\mathcal Y=\lbrace0,\tfrac1N,\dots,1\rbrace^k`$, the monomials $`\lbrace p^\alpha:\max_j\alpha_j\le N\rbrace`$ are linearly independent as functions on $`\mathcal Y`$; in particular so are the $`\binom{k+N}N`$ monomials with $`\lvert\alpha\rvert\le N`$.

- For $`k=1`$, a polynomial of degree at most $`N`$ that vanishes at $`N+1`$ distinct points is zero.
- For $`k>1`$, write $`f=\sum_{a=0}^Np_1^af_a(p_2,\dots,p_k)`$ and suppose $`f`$ vanishes on $`\mathcal Y`$. For each fixed $`(p_2,\dots,p_k)`$ in the subgrid, $`p_1\mapsto f`$ has degree at most $`N`$ and $`N+1`$ zeros, so every $`f_a`$ vanishes on the $`(k-1)`$-dimensional subgrid and, by induction, is the zero polynomial.
- The total-degree monomials form a subset. Their number is $`\binom{k+N}N`$ by stars and bars: $`\alpha\in\mathbb N^k`$ with $`\lvert\alpha\rvert\le N`$ correspond to $`\alpha\in\mathbb N^{k+1}`$ with $`\lvert\alpha\rvert=N`$.

*(a).* For a node $`u`$ of the deployment tree, let $`V_u(p)`$ be the expected terminal reward from $`u`$ and $`n(u)`$ the maximum number of decision nodes on a path from $`u`$ to a leaf. Induct on the height of $`u`$ to show $`\deg V_u\le n(u)`$ and $`\lVert V_u\rVert_{\rm coef}\le3^{n(u)}`$.

- *Leaf.* $`V_u=r\in[0,1]`$ is constant, with norm at most $`1=3^0`$.
- *Chance node.* $`V_u=\sum_e\pi_eV_{u_e}`$ with $`p`$-independent probabilities and $`n(u_e)\le n(u)`$. The degree is at most $`n(u)`$, and by Claim 3 the norm is at most $`\max_e3^{n(u_e)}\le3^{n(u)}`$.
- *Decision node with context $`j`$.* With children $`u_1`$ (action 1) and $`u_0`$ (action 0), $`V_u=p_jV_{u_1}+(1-p_j)V_{u_0}=V_{u_0}+p_jV_{u_1}-p_jV_{u_0}`$. The degree is at most $`1+(n(u)-1)=n(u)`$, and by Claim 3 the norm is at most $`3\cdot3^{n(u)-1}`$.

At the root $`n\le N`$. Since $`V_u`$ averages rewards in $`[0,1]`$, $`J_E\in[0,1]`$. Uniqueness: two polynomials that agree on $`[0,1]^k`$ differ by a polynomial vanishing on a set with nonempty interior, all of whose partial derivatives vanish at an interior point, so all coefficients agree.

*(b).* (First ⇒ second.) By (a), $`\mathbb EJ_E(p_R)=\sum_{\lvert\alpha\rvert\le N}c_\alpha\,\mathbb Ep_R^\alpha`$. Every block moment in (4.4) has $`\lvert\beta_i\rvert\le\lvert\alpha\rvert\le N`$, so it is the same under $`P`$ and $`Q`$.

(Second ⇒ first.) Fix $`\alpha`$ with $`1\le\lvert\alpha\rvert\le N`$ (the case $`\alpha=0`$ is trivial). The deployment that visits context $`j`$ exactly $`\alpha_j`$ times, in a fixed order, and pays 1 exactly when every action is 1 has $`\lvert\alpha\rvert\le N`$ decisions and value $`p^\alpha`$. With $`R=1`$, $`w_0=0`$, $`w_1=1`$ the learning value is $`m_P(\alpha)`$ under $`P`$ and $`m_Q(\alpha)`$ under $`Q`$, so the second statement forces equality.

*Marginals do not suffice.* $`P=\tfrac12\delta_{(0,0)}+\tfrac12\delta_{(1,1)}`$ and $`Q=\tfrac12\delta_{(0,1)}+\tfrac12\delta_{(1,0)}`$ have $`\mathrm{Bernoulli}(\tfrac12)`$ coordinates under both laws, yet $`m_P(1,1)=\tfrac12`$ and $`m_Q(1,1)=0`$. The task "visit both contexts once, pay 1 if both actions are 1" separates the trained students by $`\tfrac12`$.

*(c).* Let $`(\mathsf Ef)(x)=f(x+1)`$ and $`\Delta=\mathsf E-I`$. The binomial theorem for the commuting operators $`\mathsf E`$ and $`I`$ gives $`(\Delta^Nf)(0)=\sum_{i=0}^N(-1)^{N-i}\binom Nif(i)`$. If $`f`$ has degree $`d\ge1`$ and leading coefficient $`c`$, then $`\Delta f`$ has degree $`d-1`$ and leading coefficient $`cd`$; hence $`\Delta^Nf=0`$ when $`\deg f<N`$, and $`\Delta^Nx^N=N!`$.

- *Scalar witness, mass.* For $`N\ge1`$, $`\sum_{i\ \rm even}\binom Ni=\sum_{i\ \rm odd}\binom Ni=2^{N-1}`$, since their difference is $`(1-1)^N=0`$ and their sum is $`2^N`$.
- *Scalar witness, moments.* For $`\ell\ge0`$,

```math
m_{P_N}(\ell)-m_{Q_N}(\ell)=\frac{1}{2^{N-1}N^\ell}\sum_{i=0}^N(-1)^i\binom Ni\,i^\ell=\frac{(-1)^N}{2^{N-1}N^\ell}\big(\Delta^Nx^\ell\big)(0),
```

- which vanishes for $`\ell<N`$ and equals $`(-1)^NN!/(2^{N-1}N^N)`$ for $`\ell=N`$.
- *Scalar witness, realization.* Visiting one context $`N`$ times and paying 1 only if all actions are 1 gives $`J(p)=p^N`$; with $`p=Z`$ the learning gap is $`\Delta_N`$.
- *Parity witness, proper subsets are uniform.* Let $`S\subsetneq\lbrace1,\dots,N\rbrace`$ and $`x_S\in\lbrace0,1\rbrace^S`$. The complement $`S^c\ne\varnothing`$ has $`2^{N-\lvert S\rvert}`$ completions, half of each parity, since flipping a fixed coordinate of $`S^c`$ is a parity-reversing bijection. So $`P(Z_S=x_S)=Q(Z_S=x_S)=2^{-\lvert S\rvert}`$.
- *Parity witness, moments.* For $`\lvert\alpha\rvert<N`$, the support $`S=\lbrace j:\alpha_j>0\rbrace`$ has $`\lvert S\rvert<N`$. On binary vectors $`Z_j^{\alpha_j}=Z_j`$ for $`\alpha_j\ge1`$, so $`Z^\alpha=\prod_{j\in S}Z_j`$ depends only on $`Z_S`$, and its expectations agree.
- *Parity witness, value.* Visit each context once with independent actions $`a_j\sim\mathrm{Bernoulli}(p_j)`$ and pay 1 if $`\sum_ja_j`$ is even. Then $`\Pr(\text{even})=\tfrac12\big(1+\mathbb E(-1)^{\sum_ja_j}\big)=\tfrac12\big(1+\prod_j(1-2p_j)\big)=J_{\rm even}(p)`$. At binary $`p=Z`$ the actions equal $`Z`$, so $`J_{\rm even}(Z)=\mathbf 1\lbrace Z\ \text{even}\rbrace`$, which has expectation $`1`$ under $`P`$ and $`0`$ under $`Q`$.

*(d), grid count.*

- *Upper bound.* By (b), the $`\binom{k+N}N-1`$ nonconstant moments $`\lbrace m(\alpha):1\le\lvert\alpha\rvert\le N\rbrace`$ are sufficient.
- *Lower bound.* The one-update learners $`p=Z`$ combined with the monomial tasks of (b) put every monomial $`p^\alpha`$ with $`\lvert\alpha\rvert\le N`$, restricted to $`\mathcal Y`$, into $`\mathcal G`$. Every law on the finite grid is admissible, so a sufficient interface must satisfy $`\mathrm{span}\mathcal G\subseteq H_\phi`$ by [Theorem 4.2(a)](#s4-2). By Claim 5, $`\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)\ge\binom{k+N}N`$, hence $`d\ge\binom{k+N}N-1`$.

*(d), binary barrier.*

- *Every reward table is a response.* With $`p=Z`$ and a deployment that visits each of the $`k\le N`$ contexts once, the action vector equals $`Z\in\lbrace0,1\rbrace^k`$, so every terminal reward $`f:\lbrace0,1\rbrace^k\to[0,1]`$ is realized as $`g=f`$. Thus $`\mathcal G=[0,1]^{\mathcal Y}`$.
- *Fewer than $`2^k-1`$ features give discrepancy 1.* If $`d<2^k-1`$, then $`\dim H_\phi\le d+1<2^k=\lvert\mathcal Y\rvert`$, so some $`v\in H_\phi^\perp\setminus\lbrace0\rbrace`$ exists. Because $`\mathbf 1\in H_\phi`$, $`v_\pm`$ have equal mass $`s>0`$. The laws $`P=v_+/s`$ and $`Q=v_-/s`$ have disjoint supports and $`(P-Q)\phi_j=v^\top\phi_j/s=0`$. For $`f=\mathbf 1_{\mathrm{supp}\,v_+}\in\mathcal G`$, $`(P-Q)f=1`$. No discrepancy exceeds 1 because responses lie in $`[0,1]`$.
- *With $`2^k-1`$ features the discrepancy is 0.* If $`\lbrace\mathbf 1,\phi_1,\dots,\phi_{2^k-1}\rbrace`$ are linearly independent, then $`H_\phi=\mathbb R^{\mathcal Y}`$ and equal feature means force $`P=Q`$. $`\square`$

*Remarks on Theorem 4.4.*

- **Universal quantifier.** Necessity in (b) is over the whole task and learner family. A fixed task, or a fixed update recipe, may need far fewer moment combinations.
- **What is preserved.** (b) preserves the expected final task return, not the law of the trained parameter, nor tail events absent from the task class.
- **Size of the witnesses.** $`\lvert\Delta_N\rvert=N!/(2^{N-1}N^N)\sim2\sqrt{2\pi N}(2e)^{-N}`$ decays exponentially, so the scalar witness proves exact necessity of every order, not fixed-tolerance hardness. The parity witness gives a constant gap at every order, at the price of $`k=N`$ coordinates. [Proposition 4.6](#s4-6) develops this contrast.
- **The bound $`3^N`$ in (4.3) is conservative.** The experiments page reports how close adversarial deployments come to it.
- **Where the assumptions bind.**
  - The factorization (4.4) needs IID blocks that do not depend on the state; on-policy feedback that depends on the current student breaks it.
  - Convex nonnegative weights fixed in advance are needed for the expansion; data-dependent weights break it.
  - Degree is measured in the probability coordinates $`p`$. A reparameterization such as $`p=\sigma(\theta)`$ generally destroys finite degree.
- **(d), binary barrier, is deliberately broad.** Arbitrary terminal payoffs make every function a response. Structured task families can have far smaller dimension.

<a id="s4-5"></a>

**Theorem 4.5 (A round-independent bound and averaging rates).** In the setting of Definition 4.3, with $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$ as in [Theorem 4.4(a)](#s4-4), write

```math
\delta_N(P,Q)=\max_{1\le\lvert\alpha\rvert\le N}\lvert m_P(\alpha)-m_Q(\alpha)\rvert,\qquad C_J=\sum_\alpha\lvert c_\alpha\rvert\,\lvert\alpha\rvert .
```

For the averaging learner $`\bar p_n=\frac1n\sum_{i=1}^nZ_i`$ write $`\mu=\mathbb EZ`$ and $`W_n=\bar p_n-\mu`$. Let $`\kappa_s(P)`$ be the order-$`s`$ joint cumulant tensor of $`Z\sim P`$, $`D^sJ(\mu)`$ the $`s`$-th derivative tensor, and $`\langle D^sJ(\mu),T\rangle=\sum_{j_1,\dots,j_s}\partial_{j_1\cdots j_s}J(\mu)\,T_{j_1\cdots j_s}`$.

- **(a) Round-independent bound.** For every convex-affine learner,

```math
\big\lvert\mathbb E_{P^R}J(p_R)-\mathbb E_{Q^R}J(p_R)\big\rvert\le C_J\,\delta_N(P,Q)\le N3^N\delta_N(P,Q). \qquad(4.5)
```

  The bound does not depend on $`R`$, and it may be capped at 1.
- **(b) First unmatched cumulant.** Let $`P,Q`$ have the same mean $`\mu`$ and equal joint cumulants of all orders $`s<r`$, where $`2\le r\le N`$. With $`P,Q,k,N,J`$ fixed and $`n\to\infty`$,

```math
\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=\frac{\langle D^rJ(\mu),\kappa_r(P)-\kappa_r(Q)\rangle}{r!\,n^{r-1}}+O(n^{-r}). \qquad(4.6)
```

  If the displayed contraction vanishes the difference is $`O(n^{-r})`$; a $`\Theta(n^{1-r})`$ rate holds exactly when it is nonzero.
- **(c) Exact law for the scalar witnesses.** For the laws $`P_N,Q_N`$ of [Theorem 4.4(c)](#s4-4) and $`J(p)=p^N`$,

```math
\mathbb E_{P_N^n}\bar p_n^N-\mathbb E_{Q_N^n}\bar p_n^N=\frac{\Delta_N}{n^{N-1}}\quad\text{for every }n\ge1 .
```

- **(d) Unequal means.** If $`\mu_P\ne\mu_Q`$, then $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=J(\mu_P)-J(\mu_Q)+O(n^{-1})`$.
- **(e) Replay and deterministic descendants do not decay.** If the $`n`$ training blocks are $`Z^{(1)}=\dots=Z^{(n)}=T(Z_1)`$ for a single parent block $`Z_1`$ and a fixed deterministic map $`T`$ (replay is $`T=\mathrm{id}`$), then $`\bar p_n=T(Z_1)`$ for every $`n`$, and the learning-value difference equals $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$, independent of $`n`$.

*Proof.* Three claims are used.

*Claim 6 (product telescoping).* For $`a_1,\dots,a_q,b_1,\dots,b_q\in[0,1]`$, $`\lvert\prod_ia_i-\prod_ib_i\rvert\le\sum_i\lvert a_i-b_i\rvert`$. Indeed $`\prod_ia_i-\prod_ib_i=\sum_{l=1}^q\big(\prod_{i<l}b_i\big)(a_l-b_l)\big(\prod_{i>l}a_i\big)`$, and each bracketed product lies in $`[0,1]`$.

*Claim 7 (cumulants of the empirical mean).* For $`s\ge2`$, $`\kappa_s(W_n)=n^{1-s}\kappa_s(Z)`$, and $`\kappa_1(W_n)=0`$. Joint cumulants of order $`s\ge2`$ are translation invariant and multilinear, and they are additive over independent summands because every mixed cumulant of independent vectors vanishes. Hence $`\kappa_s\big(\tfrac1n\sum_iZ_i\big)=n^{-s}\sum_{i=1}^n\kappa_s(Z_i)=n^{1-s}\kappa_s(Z)`$; the mean of $`W_n`$ is zero.

*Claim 8 (moment–cumulant formula).* For a random vector $`W`$ with finite moments, $`\mathbb E[W_{j_1}\cdots W_{j_s}]=\sum_{\pi\in\Pi(s)}\prod_{B\in\pi}\kappa(W_{j_b}:b\in B)`$, where $`\Pi(s)`$ is the set of partitions of $`\lbrace1,\dots,s\rbrace`$. This is the defining inversion of cumulants (Leonov–Shiryaev).

*(a).* Fix $`\alpha`$ with $`1\le\lvert\alpha\rvert=a\le N`$. Write (4.4) as $`\mathbb Ep_R^\alpha=\sum_s\omega_s\pi_s\prod_{i=1}^Rm(\beta_i(s))`$ with $`\omega_s=\prod_lw_{s(l)}\ge0`$, $`\sum_s\omega_s=1`$ and $`\pi_s=\prod_{l:s(l)=0}p_{0,j(l)}\in[0,1]`$.

- For fixed $`s`$, at most $`a`$ blocks have $`\beta_i(s)\ne0`$, each with $`1\le\lvert\beta_i(s)\rvert\le N`$, and all block moments lie in $`[0,1]`$ because $`Z\in[0,1]^k`$.
- By Claim 6, $`\big\lvert\pi_s\prod_im_P(\beta_i(s))-\pi_s\prod_im_Q(\beta_i(s))\big\rvert\le\sum_{i:\beta_i(s)\ne0}\lvert m_P(\beta_i(s))-m_Q(\beta_i(s))\rvert\le a\,\delta_N`$.
- Summing with weights $`\omega_s`$ gives $`\lvert\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha\rvert\le\lvert\alpha\rvert\,\delta_N`$. The term $`\alpha=0`$ cancels, so $`\lvert\sum_\alpha c_\alpha(\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha)\rvert\le C_J\delta_N`$.

Finally $`C_J\le N\sum_\alpha\lvert c_\alpha\rvert\le N3^N`$ by (4.3), and both learning values lie in $`[0,1]`$. The bound is uniform in $`R`$ because $`\sum_s\omega_s=1`$ for every $`R`$.

*(b).* $`J`$ is a polynomial of degree at most $`N`$, so its Taylor expansion at $`\mu`$ terminates, and

```math
\mathbb EJ(\bar p_n)=\sum_{s=0}^N\frac1{s!}\langle D^sJ(\mu),\mathbb E\,W_n^{\otimes s}\rangle .
```

$`P`$ and $`Q`$ share $`\mu`$, so $`D^sJ(\mu)`$ is common. The $`s=0`$ term cancels and the $`s=1`$ term vanishes. For $`s\ge2`$ apply Claim 8 to $`W_n`$.

- Partitions with a singleton block vanish because $`\kappa_1(W_n)=0`$. A partition with $`q`$ blocks of sizes $`b_1,\dots,b_q\ge2`$ contributes, by Claim 7, $`\prod_ln^{1-b_l}\kappa_{b_l}(Z)=n^{q-s}\prod_l\kappa_{b_l}(Z)`$.
- Cumulants of order below $`r`$ agree, so a partition term can differ between $`P`$ and $`Q`$ only if some block has size $`b_l\ge r`$. Then $`s=\sum_lb_l\ge r+2(q-1)`$, hence $`q-s\le2-r-q\le1-r`$.
- Equality $`q-s=1-r`$ holds only for $`q=1`$ and $`s=r`$: if $`q=1`$ and $`s>r`$, then $`q-s=1-s\le-r`$; if $`q\ge2`$, then $`q-s\le2-r-q\le-r`$.

So the only differing term of order $`n^{1-r}`$ is the single block at $`s=r`$, which contributes $`\frac1{r!}\langle D^rJ(\mu),n^{1-r}(\kappa_r(P)-\kappa_r(Q))\rangle`$. Every other differing term carries a factor $`n^{-r}`$ or smaller. There are finitely many such terms ($`s\le N`$), each with a bounded coefficient since $`Z\in[0,1]^k`$ has bounded cumulants, so the remainder is $`O(n^{-r})`$ with a constant depending only on $`P,Q,J`$.

*(c).* For $`N=1`$ the difference is $`m_{P_1}(1)-m_{Q_1}(1)=\Delta_1`$. Let $`N\ge2`$.

- $`P_N`$ and $`Q_N`$ agree on raw moments of order below $`N`$, hence on cumulants of order below $`N`$, since $`\kappa_s`$ is a polynomial in moments of order at most $`s`$.
- In the argument of (b) with $`r=N`$, a differing partition must contain a block of size at least $`N`$, and $`s\le\deg J=N`$. The only such partition is the single block at $`s=N`$, so there is no remainder.
- $`D^N(p^N)=N!`$ and $`\kappa_N(P_N)-\kappa_N(Q_N)=m_{P_N}(N)-m_{Q_N}(N)=\Delta_N`$, because $`\kappa_N`$ equals $`m_N`$ minus a polynomial in lower moments, which agree. The difference is exactly $`\Delta_Nn^{1-N}`$.

*(d).* Expand around $`\mu_P`$ and around $`\mu_Q`$ separately. The $`s=0`$ terms give $`J(\mu_P)-J(\mu_Q)`$. For $`s\ge2`$, every partition into blocks of size at least 2 has $`q\le s/2`$, hence $`q-s\le-s/2\le-1`$, so all remaining terms are $`O(n^{-1})`$.

*(e).* $`\bar p_n=\frac1n\sum_{i=1}^nT(Z_1)=T(Z_1)`$, so $`\mathbb EJ(\bar p_n)=\mathbb EJ(T(Z_1))`$ for every $`n`$. $`\square`$

*Remarks on Theorem 4.5.*

- **Why $`R`$ does not appear in (4.5).** The convex weights make the expansion coefficients a probability vector, whatever the number of blocks. Signed or unconstrained weights break this and can amplify discrepancies; data-dependent weights invalidate the factorization.
- **Population, not one run.** Both bounds concern the expected final return of the training procedure, not the score of one trained student.
- **Coordinate dependence.** $`C_J`$ and $`\delta_N`$ are measured in the monomial basis of the probability coordinates. Large $`N`$ can make (4.5) vacuous, which is why [Proposition 4.6](#s4-6) replaces degree by approximation width.
- **What controls the rate.** In (4.6) the variable is the number $`n`$ of independent blocks. Replay, or deterministic descendants of a block (part (e)), increases training exposure but not $`n`$, and the order-$`r`$ discrepancy stays at its one-block value. This is the counting discipline behind [Theorem 2.10](02_feasibility_and_acquisition.md#s2-10) and the method of §9.
- **Fixed laws.** The $`O(n^{-r})`$ constant depends on $`(P,Q,J)`$; the expansion is not uniform over families of laws whose cumulants grow.

## Approximate complexity

<a id="s4-6"></a>

**Proposition 4.6 (Exact order is not approximate complexity).** Let $`T_j`$ be the Chebyshev polynomials of the first kind ($`T_j(\cos\theta)=\cos(j\theta)`$, $`\lvert T_j\rvert\le1`$ on $`[-1,1]`$, $`T_j(1)=1`$) and $`\mathcal P_q`$ the polynomials of total degree at most $`q`$. Consider the scalar task $`J(p)=p^N`$ on $`[0,1]`$ (one context visited $`N`$ times, success if all actions are 1) and the parity task $`J_{\rm even}`$ of [Theorem 4.4(c)](#s4-4) on $`[0,1]^N`$. For $`p\in[0,1]`$,

```math
p^N=4^{-N}\binom{2N}{N}+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}T_j(2p-1). \qquad(4.7)
```

- **(a) Smooth values have low approximate order.** Let $`q<N`$, let $`h_q\in\mathcal P_q`$ be the truncation of (4.7) after order $`q`$, and let $`B\sim\mathrm{Bin}(2N,\tfrac12)`$. Then:
  - $`\lVert p^N-h_q\rVert_{\infty,[0,1]}=\Pr\lbrace\lvert B-N\rvert>q\rbrace\le2e^{-q^2/N}`$;
  - if two laws $`P,Q`$ on $`[0,1]`$ have equal moments of orders $`\ell\le q`$, then $`\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert\le4e^{-q^2/N}`$;
  - the same bound holds for the learning values $`\mathbb E\,p_R^N`$ after any convex-affine learner with IID scalar blocks.

  Consequently, matching moments through order $`q\ge\min\lbrace N,\lceil\sqrt{N\log(4/\varepsilon)}\rceil\rbrace`$ guarantees a gap of at most $`\varepsilon`$, although exact equality for all laws requires order $`N`$ ([Theorem 4.4(c)](#s4-4)).
- **(b) Parity is incompressible below degree $`N`$.** For every $`q<N`$,

```math
\min_{h\in\mathcal P_q}\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N}=\tfrac12 ,
```

  and the constant $`h\equiv\tfrac12`$ attains the minimum.

*Proof.* *Claim 9 (the identity (4.7)).* Put $`2p-1=\cos\theta`$ with $`\theta\in[0,\pi]`$, so $`p=\cos^2(\theta/2)`$ and

```math
p^N=\Big(\frac{e^{i\theta/2}+e^{-i\theta/2}}{2}\Big)^{2N}=4^{-N}\sum_{b=0}^{2N}\binom{2N}{b}e^{i(b-N)\theta}.
```

Grouping $`b=N\pm j`$ for $`j\ge1`$ with $`\binom{2N}{N+j}=\binom{2N}{N-j}`$ gives $`p^N=4^{-N}\binom{2N}N+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}\cos(j\theta)`$, and $`\cos(j\theta)=T_j(2p-1)`$. Both sides are polynomials that agree on $`[0,1]`$, so (4.7) is a polynomial identity.

*(a), first item.* Write $`a_0=4^{-N}\binom{2N}N`$ and $`a_j=2\cdot4^{-N}\binom{2N}{N-j}`$ for $`j\ge1`$. Then $`p^N-h_q(p)=\sum_{j=q+1}^Na_jT_j(2p-1)`$.

- All $`a_j\ge0`$ and $`\lvert T_j\rvert\le1`$, so $`\lvert p^N-h_q(p)\rvert\le\sum_{j>q}a_j`$, with equality at $`p=1`$ where every $`T_j(1)=1`$.
- Pairing $`b=N-j`$ with $`b=N+j`$, $`\sum_{j=q+1}^Na_j=\sum_{b:\lvert b-N\rvert>q}\binom{2N}b4^{-N}=\Pr\lbrace\lvert B-N\rvert>q\rbrace`$.
- $`B`$ is a sum of $`2N`$ independent $`\lbrace0,1\rbrace`$ variables with mean $`N`$, so Hoeffding's inequality gives $`\Pr\lbrace\lvert B-N\rvert\ge q\rbrace\le2\exp(-2q^2/(2N))=2e^{-q^2/N}`$.

*(a), second item.* Since $`h_q\in\mathcal P_q`$ and the laws match moments through $`q`$, $`\mathbb E_Ph_q(Z)=\mathbb E_Qh_q(Z)`$, so $`\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert=\lvert(P-Q)(p^N-h_q)\rvert\le\lVert P-Q\rVert_1\lVert p^N-h_q\rVert_\infty\le4e^{-q^2/N}`$.

*(a), third item.* For $`\ell\le q`$, (4.4) writes $`\mathbb Ep_R^\ell`$ as a fixed combination of products of block moments of order at most $`\ell`$. So the laws of $`p_R\in[0,1]`$ under $`P^R`$ and under $`Q^R`$ match moments through $`q`$, and the second item applies to them.

*Order needed.* $`4e^{-q^2/N}\le\varepsilon`$ if and only if $`q\ge\sqrt{N\log(4/\varepsilon)}`$; for $`q\ge N`$ the gap is exactly 0 by [Theorem 4.4(b)](#s4-4).

*(b).* *Upper bound.* $`J_{\rm even}`$ is a probability, so $`\lVert J_{\rm even}-\tfrac12\rVert_\infty\le\tfrac12`$.

*Lower bound.* Let $`h\in\mathcal P_q`$ with $`q<N`$, and let $`P,Q`$ be uniform on the even- and odd-parity vertices of $`\lbrace0,1\rbrace^N`$.

- On binary vertices $`p_j^a=p_j`$ for $`a\ge1`$, so $`h`$ restricted to $`\lbrace0,1\rbrace^N`$ is a multilinear polynomial whose monomials $`\prod_{j\in S}p_j`$ have $`\lvert S\rvert\le q<N`$.
- Every proper coordinate subset is uniform under both laws (proof of [Theorem 4.4(c)](#s4-4)), so $`\mathbb E_Ph=\mathbb E_Qh`$.
- On the vertices $`J_{\rm even}(Z)=\mathbf 1\lbrace Z\ \text{even}\rbrace`$, so $`\mathbb E_PJ_{\rm even}=1`$ and $`\mathbb E_QJ_{\rm even}=0`$.

Therefore

```math
1=(P-Q)J_{\rm even}=(P-Q)(J_{\rm even}-h)\le\lVert P-Q\rVert_1\,\lVert J_{\rm even}-h\rVert_{\infty,\lbrace0,1\rbrace^N}\le2\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N},
```

so $`\lVert J_{\rm even}-h\rVert_\infty\ge\tfrac12`$. Equivalently, by [Theorem 4.2(b)](#s4-2), the matched pair $`(P,Q)`$ certifies $`\mathrm{dist}_\infty(J_{\rm even},\mathcal P_q)\ge\tfrac12`$. $`\square`$

*Remarks on Proposition 4.6.*

- **The contrast.** $`p^N`$ and $`J_{\rm even}`$ both have exact degree $`N`$, and exact sufficiency needs all moments through $`N`$ for both. Their approximate requirements differ completely:
  - $`p^N`$ is resolved to accuracy $`\varepsilon`$ by order $`O(\sqrt{N\log(1/\varepsilon)})`$;
  - $`J_{\rm even}`$ is not resolved below $`\tfrac12`$ by any order below $`N`$.
- The operative complexity is therefore the approximate dimension $`d_\varepsilon`$ of the response class ([Theorem 4.2(b)](#s4-2)), not its degree.
- **Scope.** (a) is an upper bound on the sufficient order for this scalar task and the convex-affine learners; no matching lower bound on the approximate order is claimed.
- **Stability.**
  - Exact low-order matching is not a finite-sample certificate. If moments are only estimated, the power-basis coefficients of $`h_q`$ multiply the estimation errors, and they can be large.
  - Bounded Chebyshev features avoid power-basis amplification for one scalar message, but pulled back through several affine updates they need their own coefficient bound.
  - In general the relevant pair is $`(B,\tau)`$ of [Theorem 4.2(c)](#s4-2), and [Theorem 8.1](08_verification.md#s8-1) makes the tradeoff exact.
- **Realizability.** Both witness families are realizable finite feedback protocols: a reset experiment can emit a block drawn from the specified law. This shows that the environments exist, not that a given benchmark permits them.

## The learner sets the order

<a id="s4-7"></a>

**Theorem 4.7 (Update composition amplifies the information requirement).** The setting has three parts.

- **Polynomial updates.** Fix $`R\ge1`$ and a common $`p_0\in[0,1]^k`$. Independent blocks $`Z_1,\dots,Z_R`$ take values in a finite set $`\mathcal Y\subset[0,1]^k`$; block $`Z_i`$ has law $`P_i`$ under the reference source and $`Q_i`$ under the alternative. The learner is $`p_i=U_i(p_{i-1},Z_i)`$, where $`U_i`$ maps $`[0,1]^k\times\mathcal Y`$ into $`[0,1]^k`$ and every coordinate of $`U_i`$ is a polynomial with
  - **(D1)** degree at most $`a_i\ge1`$ in $`p_{i-1}`$ (as a polynomial in $`p_{i-1}`$ whose coefficients are polynomials in $`Z_i`$);
  - **(D2)** degree at most $`b_i\ge1`$ in $`Z_i`$ (as a polynomial in $`Z_i`$ whose coefficients are polynomials in $`p_{i-1}`$).

  The deployment value $`J:[0,1]^k\to[0,1]`$ is a polynomial of total degree at most $`N`$, which holds for every deployment of Definition 4.3 with at most $`N`$ decisions ([Theorem 4.4(a)](#s4-4)).
- **Smooth local setting.** $`\theta\in\mathbb R^p`$, $`V:\mathbb R^p\to\mathbb R`$ is a continuation value ([Definition 1.4](../README.md#s1-4)), and one SGD step with consumed-gradient law $`P`$ acts as $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$, with $`\mu_P=\mathbb E_PG`$ and $`M_P=\mathbb E_PGG^\top`$.
  - **(T3)** $`V`$ is $`C^3`$ on every segment $`[\theta,\theta-\eta g]`$ with $`g`$ in the supports of $`P`$ and $`Q`$, and $`\lvert D^3V(\vartheta)[h,h,h]\rvert\le M_3\lVert h\rVert^3`$ there.
- **Error recursion.** A nonnegative sequence satisfies $`e_0=0`$ and $`e_{t+1}\le(1+\eta L)e_t+C\eta^2`$ for constants $`\eta,L,C\ge0`$.

Then:

- **(a) Degree composition.** Under (D1)–(D2), $`J(p_R)`$, as a polynomial in $`(Z_1,\dots,Z_R)`$, has total degree in block $`Z_i`$ at most

```math
D_i=N\,b_i\prod_{t=i+1}^Ra_t. \qquad(4.8)
```

  Consequently, if $`\mathbb E_{P_i}Z^\alpha=\mathbb E_{Q_i}Z^\alpha`$ for every $`i`$ and every $`\lvert\alpha\rvert\le D_i`$, then $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$. If all blocks share one source law, matching through $`\max_iD_i`$ suffices.
- **(b) Attainment.** For every $`R\ge1`$ there are updates in the class and a deployment with $`N=1`$ whose response is $`Z_1^{2^{R-1}}`$. The scalar laws of order $`M=2^{R-1}`$ from [Theorem 4.4(c)](#s4-4) agree on all moments below $`M`$ yet give different learning values, so (4.8) cannot be improved in general.
- **(c) A threshold learner.** For the learner $`p=\mathbf 1\lbrace Z>1/2\rbrace`$ and $`J(p)=p`$ ($`N=1`$), the laws $`P(Z=0)=\tfrac13`$, $`P(Z=\tfrac34)=\tfrac23`$ and $`Q(Z=\tfrac12)=1`$ have $`\mathbb E_PZ=\mathbb E_QZ=\tfrac12`$, while $`\mathbb E_PJ(p)=\tfrac23`$ and $`\mathbb E_QJ(p)=0`$.
- **(d) Local SGD expansion.** Under (T3),

```math
(K^P-K^Q)V(\theta)=-\eta\nabla V(\theta)^\top(\mu_P-\mu_Q)+\frac{\eta^2}{2}\langle\nabla^2V(\theta),M_P-M_Q\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\frac{M_3\eta^3}{6}\big(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3\big). \qquad(4.9)
```

- **(e) Error accumulation.** Under the error recursion,

```math
e_R\le C\eta^2\sum_{s=0}^{R-1}(1+\eta L)^s=\frac{C\eta}{L}\big((1+\eta L)^R-1\big)\quad(L>0),
```

  with equality when the recursion holds with equality. The linear bound $`e_R\le RC\eta^2`$ holds when $`L=0`$; for $`L>0`$ it fails in general: with $`\eta=\tfrac12`$ and $`L=C=1`$ the equality recursion gives $`e_4=\tfrac{65}{32}>1=RC\eta^2`$.

*Proof.* *(a).* Define backward continuation polynomials $`\Phi_R(p)=J(p)`$ and $`\Phi_{t-1}(p;Z_t,\dots,Z_R)=\Phi_t(U_t(p,Z_t);Z_{t+1},\dots,Z_R)`$ for $`t=R,\dots,1`$, so that $`J(p_R)=\Phi_0(p_0;Z_1,\dots,Z_R)`$. We show by backward induction that $`\Phi_t`$ has degree at most $`d_t:=N\prod_{s=t+1}^Ra_s`$ in $`p`$ ($`d_R=N`$) and degree at most $`D_i`$ in $`Z_i`$ for each $`i>t`$.

- *Base case $`t=R`$.* $`\Phi_R=J`$ has degree at most $`N`$ in $`p`$ and contains no blocks.
- *Inductive step.* Write $`\Phi_t=\sum_{\lvert\gamma\rvert\le d_t}c_\gamma(Z_{t+1},\dots,Z_R)\,p^\gamma`$, where each $`c_\gamma`$ has degree at most $`D_i`$ in $`Z_i`$ for $`i>t`$. Substitute $`p=U_t(p',Z_t)`$. The monomial $`p^\gamma`$ becomes a product of $`\lvert\gamma\rvert`$ coordinate polynomials of $`U_t`$, each of degree at most $`a_t`$ in $`p'`$ by (D1) and at most $`b_t`$ in $`Z_t`$ by (D2). Degrees of products add, so the substituted monomial has degree at most $`a_td_t=d_{t-1}`$ in $`p'`$ and at most $`b_td_t=Nb_t\prod_{s>t}a_s=D_t`$ in $`Z_t`$. The coefficients $`c_\gamma`$ involve neither $`p'`$ nor $`Z_t`$, so the degrees in $`Z_i`$, $`i>t`$, are unchanged.

At $`t=0`$, $`p=p_0`$ is a constant, so $`J(p_R)`$ has degree at most $`D_i`$ in each $`Z_i`$. Expand $`J(p_R)=\sum c_{\alpha_1\cdots\alpha_R}\prod_iZ_i^{\alpha_i}`$, a finite sum with $`\lvert\alpha_i\rvert\le D_i`$. Independence gives $`\mathbb E\prod_iZ_i^{\alpha_i}=\prod_i\mathbb E_{P_i}Z_i^{\alpha_i}`$, and under the matching hypothesis every factor is the same under $`P_i`$ and $`Q_i`$. If $`P_i=P`$ and $`Q_i=Q`$ for all $`i`$, matching through $`\max_iD_i`$ implies matching through each $`D_i`$.

*(b).* Let $`U_1(p,z)=z`$ and $`U_i(p,z)=p\odot p`$ for $`i\ge2`$, so $`b_1=1`$ and $`a_i=2`$ for $`i\ge2`$ (the other degrees are bounded by 1). Both maps keep $`[0,1]^k`$ invariant. Deploy once with $`J(p)=p_1`$ ($`N=1`$). By induction $`p_i=Z_1^{2^{i-1}}`$ coordinatewise, so the response is $`Z_{1,1}^{2^{R-1}}`$, while (4.8) gives $`D_1=1\cdot1\cdot2^{R-1}`$. For the laws $`P_M,Q_M`$ with $`M=2^{R-1}`$, moments of order below $`M`$ agree and $`\mathbb E_{P_M}Z^M-\mathbb E_{Q_M}Z^M=\Delta_M\ne0`$. Later blocks are unused, which (D2) permits.

*(c).* $`\mathbb E_PZ=\tfrac13\cdot0+\tfrac23\cdot\tfrac34=\tfrac12=\mathbb E_QZ`$. Under $`P`$, $`\mathbf 1\lbrace Z>1/2\rbrace=1`$ exactly when $`Z=\tfrac34`$, which has probability $`\tfrac23`$. Under $`Q`$, $`Z=\tfrac12`$ is not greater than $`\tfrac12`$, so the value is 0.

*(d).* Fix $`g`$ in either support and put $`\varphi(\tau)=V(\theta-\tau\eta g)`$, $`\tau\in[0,1]`$. By (T3) $`\varphi\in C^3[0,1]`$, and Taylor's theorem with integral remainder gives $`\varphi(1)=\varphi(0)+\varphi'(0)+\tfrac12\varphi''(0)+\int_0^1\tfrac{(1-\tau)^2}{2}\varphi'''(\tau)\,d\tau`$, with

```math
\varphi'(0)=-\eta\nabla V(\theta)^\top g,\qquad\varphi''(0)=\eta^2g^\top\nabla^2V(\theta)g,\qquad\varphi'''(\tau)=-\eta^3D^3V(\theta-\tau\eta g)[g,g,g].
```

By (T3) $`\lvert\varphi'''\rvert\le M_3\eta^3\lVert g\rVert^3`$, and $`\int_0^1(1-\tau)^2/2\,d\tau=1/6`$. Hence $`V(\theta-\eta g)=V(\theta)-\eta\nabla V^\top g+\tfrac{\eta^2}{2}\langle\nabla^2V,gg^\top\rangle+\rho(g)`$ with $`\lvert\rho(g)\rvert\le\tfrac{M_3\eta^3}{6}\lVert g\rVert^3`$. Take $`\mathbb E_P`$ and $`\mathbb E_Q`$ and subtract: $`V(\theta)`$ cancels, the linear and quadratic terms give the two displayed terms, and $`\mathcal E=\mathbb E_P\rho(G)-\mathbb E_Q\rho(G)`$ obeys the stated bound by the triangle inequality.

*(e).* Induction on $`t`$ gives $`e_t\le C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s`$: the case $`t=0`$ is the empty sum, and $`e_{t+1}\le(1+\eta L)C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s+C\eta^2=C\eta^2\sum_{s=0}^t(1+\eta L)^s`$. Each step uses only the recursion, so equality propagates. The closed form is the geometric sum. For the witness, $`e_{t+1}=\tfrac32e_t+\tfrac14`$ gives $`e_1=\tfrac14`$, $`e_2=\tfrac58`$, $`e_3=\tfrac{19}{16}`$, $`e_4=\tfrac{65}{32}`$. $`\square`$

*Remarks on Theorem 4.7.*

- **The information order belongs to the learner composed with the deployment.** With one convex-affine update ($`a_i=b_i=1`$), (4.8) recovers $`D_i=N`$, which is [Theorem 4.4(b)](#s4-4). Each nonlinear later update multiplies the requirement on all earlier blocks.
- **Degree is coordinate-dependent.** An invertible reparameterization can change the polynomial form of an update. The intrinsic object remains the response class and its approximate dimension ([Theorem 4.2](#s4-2)); degree is a tractable certificate in the chosen probability coordinates.
- **Non-polynomial learners.** (c) shows that once the update is discontinuous no moment order is sufficient in general; even one decision imposes more than a first-moment requirement.
- **Scope of (4.9).** The expansion is local and one-step: its first-order statistic is the mean consumed gradient and its second-order statistic the gradient second moment. For a stateful optimizer the increment of the complete learner state replaces the parameter step ([Proposition 5.1](05_optimizer_state.md#s5-1)); cold and warm Adam are treated in [Proposition 5.3](05_optimizer_state.md#s5-3) and [Propositions 5.5–5.6](05_optimizer_state.md#s5-5).
- **Accumulation needs stability.** (e) is sharp for its recursion. A bound linear in $`R`$ requires $`\eta LR=O(1)`$, since $`\sum_{s<R}(1+\eta L)^s\le Re^{\eta LR}`$, or a contractive error map.

## Related work

- LP duality, the event form of total variation, finite differences, parity independence, the moment–cumulant formula (Leonov–Shiryaev), Chebyshev expansions and Hoeffding's inequality are classical tools; they are credited as such and not claimed as new.
- What this section adds is their use to compute the information a feedback law must carry for a given learner and deployment class: a span dimension or approximation width of the response class, the moment order for convex-affine learners, and its amplification by composed updates.
