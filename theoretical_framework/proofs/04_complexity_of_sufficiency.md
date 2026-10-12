# §4 Complexity of sufficiency — Proofs

[← Framework](../README.md) · [Experiments](../experiments/04_complexity_of_sufficiency.md) · [中文](04_complexity_of_sufficiency_ZH.md)

This page retains Definition 4.1 through Theorem 4.4; see the [continuation](04_rates_and_composition.md) for rates and composition.

**Summary.**

- §3 fixed the object of sufficiency: two feedback laws are interchangeable exactly when they agree on every response in the response class $`\mathcal G`$ ([Definition 1.4](../README.md#s1-4), [Theorem 3.2](03_learning_consequence.md#s3-2)). This section measures how much information that agreement requires.
- [Theorem 4.2](#s4-2) is the general answer. Exact sufficiency of a feature interface is a span condition; its count is $`\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1`$. Approximate sufficiency is twice a best uniform approximation error, and a decomposition with bounded coefficients transfers feature-mean errors into response errors.
- [Theorems 4.4](#s4-4) and [4.5](04_rates_and_composition.md#s4-5) compute everything for convex-affine learners on binary-context students. The exact information is the joint moments up to the number $`N`$ of deployment decisions, every order is necessary, the gap bound does not depend on the number of training rounds, and the gap decays with the number of independent blocks at a rate fixed by the first unmatched cumulant. Replay and deterministic descendants do not decay.
- [Proposition 4.6](04_rates_and_composition.md#s4-6) separates exact order from approximate complexity: about $`\sqrt{N\log(1/\varepsilon)}`$ moments resolve $`p^N`$ to accuracy $`\varepsilon`$, while parity cannot be approximated below $`1/2`$ by any degree below $`N`$.
- [Theorem 4.7](04_rates_and_composition.md#s4-7) shows that the learner sets the order. Composed polynomial updates multiply the degree, a threshold learner breaks first-moment sufficiency with one decision, and one smooth SGD step has the mean gradient as its first-order statistic and the gradient second moment as its second-order statistic. [Proposition 5.1](05_optimizer_state.md#s5-1) extends the last statement to arbitrary stateful optimizers.

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
- **Size of the witnesses.** $`\lvert\Delta_N\rvert=N!/(2^{N-1}N^N)\sim2\sqrt{2\pi N}(2e)^{-N}`$ decays exponentially, so the scalar witness proves exact necessity of every order, not fixed-tolerance hardness. The parity witness gives a constant gap at every order, at the price of $`k=N`$ coordinates. [Proposition 4.6](04_rates_and_composition.md#s4-6) develops this contrast.
- **The bound $`3^N`$ in (4.3) is conservative.** The experiments page reports how close adversarial deployments come to it.
- **Where the assumptions bind.**
  - The factorization (4.4) needs IID blocks that do not depend on the state; on-policy feedback that depends on the current student breaks it.
  - Convex nonnegative weights fixed in advance are needed for the expansion; data-dependent weights break it.
  - Degree is measured in the probability coordinates $`p`$. A reparameterization such as $`p=\sigma(\theta)`$ generally destroys finite degree.
- **(d), binary barrier, is deliberately broad.** Arbitrary terminal payoffs make every function a response. Structured task families can have far smaller dimension.
