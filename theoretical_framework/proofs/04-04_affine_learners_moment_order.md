# Theorem 4.4 · Affine learners: moment order is necessary and sufficient

*Proof*

[← Framework](../README.md) · [Statement](../statements/04-04_affine_learners_moment_order.md) · [Proof](04-04_affine_learners_moment_order.md) · [Experiments](../experiments/04-04_affine_learners_moment_order.md) · [中文](04-04_affine_learners_moment_order_ZH.md)

**What is proved here.** For binary-decision students, deployments with at most $`N`$ decisions and learners that take fixed convex averages of independent experience blocks, two sources give the same expected deployment return for every such task and learner exactly when their joint moments agree up to order $`N`$; every order is necessary; and the exact number of statistics is a binomial coefficient on a grid. The proof rests on two facts: the deployment value is a polynomial of degree $`N`$ in the action probabilities, and the expectation of a polynomial of an affine average of IID blocks factors into block moments.

**Setting used on this page** ([Definition 4.3](../statements/04-03_binary_context_student_affine_learner.md#s4-3)).

- Student: $`k`$ binary-action contexts, parameter $`p\in[0,1]^k`$; at a decision with context $`j`$ it takes action 1 with conditional probability $`p_j`$ given the past.
- Deployment: a finite tree of chance nodes ($`p`$-independent transition probabilities), decision nodes (labelled by a context $`j`$ that may depend on the history but not on $`p`$) and terminal nodes (rewards in $`[0,1]`$); every root-to-leaf path has at most $`N`$ decision nodes; $`J_E(p)`$ is the expected terminal reward.
- Experience: blocks $`Z_1,\dots,Z_R\in[0,1]^k`$, IID from $`P`$ (or $`Q`$), with arbitrarily dependent coordinates. Learner: $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$ with fixed $`p_0`$ and fixed weights $`w_i\ge0`$, $`\sum_{i=0}^Rw_i=1`$.
- Notation: $`p^\alpha=\prod_jp_j^{\alpha_j}`$, $`\lvert\alpha\rvert=\sum_j\alpha_j`$, $`m_P(\alpha)=\mathbb E_PZ^\alpha`$, $`\lVert f\rVert_{\rm coef}=\sum_\alpha\lvert c_\alpha\rvert`$ for $`f=\sum_\alpha c_\alpha p^\alpha`$, $`e_j`$ the $`j`$-th unit multi-index.
- Feature interfaces, $`H_\phi`$ and the unresolved discrepancy are as in [Definition 4.1](../statements/04-01_feature_sufficiency_unresolved_discrepancy.md#s4-1); the response class $`\mathcal G`$ collects the functions $`Z\mapsto J_E(p_R)`$ realized by the tasks and learners considered.

**Statement ([Theorem 4.4](../statements/04-04_affine_learners_moment_order.md#s4-4)).** In the setting of Definition 4.3:

- **(a) Polynomial deployment value.** $`J_E`$ has a unique representation

```math
J_E(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha,\qquad\sum_\alpha\lvert c_\alpha\rvert\le3^N, \qquad(4.3)
```

- and $`J_E(p)\in[0,1]`$ on $`[0,1]^k`$.
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

**Proof.** Three claims are used.

*Claim 1 (coefficient norm under elementary operations).* Let $`f,g`$ be polynomials in $`p`$ and $`j\in\lbrace1,\dots,k\rbrace`$.

- $`\lVert p_jf\rVert_{\rm coef}=\lVert f\rVert_{\rm coef}`$, because $`p^\beta\mapsto p^{\beta+e_j}`$ maps distinct monomials to distinct monomials.
- $`\lVert f+g\rVert_{\rm coef}\le\lVert f\rVert_{\rm coef}+\lVert g\rVert_{\rm coef}`$.
- For weights $`\pi_e\ge0`$ with $`\sum_e\pi_e=1`$, $`\lVert\sum_e\pi_ef_e\rVert_{\rm coef}\le\max_e\lVert f_e\rVert_{\rm coef}`$.

*Claim 2 (block-moment expansion of an affine learner).* Let $`\lvert\alpha\rvert=a`$ and list the coordinate factors of $`p_R^\alpha`$ as $`j(1),\dots,j(a)`$, coordinate $`j`$ appearing $`\alpha_j`$ times. Then

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

*Claim 3 (grid independence).* On $`\mathcal Y=\lbrace0,\tfrac1N,\dots,1\rbrace^k`$, the monomials $`\lbrace p^\alpha:\max_j\alpha_j\le N\rbrace`$ are linearly independent as functions on $`\mathcal Y`$; in particular so are the $`\binom{k+N}N`$ monomials with $`\lvert\alpha\rvert\le N`$.

- For $`k=1`$, a polynomial of degree at most $`N`$ that vanishes at $`N+1`$ distinct points is zero.
- For $`k>1`$, write $`f=\sum_{a=0}^Np_1^af_a(p_2,\dots,p_k)`$ and suppose $`f`$ vanishes on $`\mathcal Y`$. For each fixed $`(p_2,\dots,p_k)`$ in the subgrid, $`p_1\mapsto f`$ has degree at most $`N`$ and $`N+1`$ zeros, so every $`f_a`$ vanishes on the $`(k-1)`$-dimensional subgrid and, by induction, is the zero polynomial.
- The total-degree monomials form a subset. Their number is $`\binom{k+N}N`$ by stars and bars: $`\alpha\in\mathbb N^k`$ with $`\lvert\alpha\rvert\le N`$ correspond to $`\alpha\in\mathbb N^{k+1}`$ with $`\lvert\alpha\rvert=N`$.

*(a).* For a node $`u`$ of the deployment tree, let $`V_u(p)`$ be the expected terminal reward from $`u`$ and $`n(u)`$ the maximum number of decision nodes on a path from $`u`$ to a leaf. Induct on the height of $`u`$ to show $`\deg V_u\le n(u)`$ and $`\lVert V_u\rVert_{\rm coef}\le3^{n(u)}`$.

- *Leaf.* $`V_u=r\in[0,1]`$ is constant, with norm at most $`1=3^0`$.
- *Chance node.* $`V_u=\sum_e\pi_eV_{u_e}`$ with $`p`$-independent probabilities and $`n(u_e)\le n(u)`$. The degree is at most $`n(u)`$, and by Claim 1 the norm is at most $`\max_e3^{n(u_e)}\le3^{n(u)}`$.
- *Decision node with context $`j`$.* With children $`u_1`$ (action 1) and $`u_0`$ (action 0), $`V_u=p_jV_{u_1}+(1-p_j)V_{u_0}=V_{u_0}+p_jV_{u_1}-p_jV_{u_0}`$. The degree is at most $`1+(n(u)-1)=n(u)`$, and by Claim 1 the norm is at most $`3\cdot3^{n(u)-1}`$.

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
- *Lower bound.* The one-update learners $`p=Z`$ combined with the monomial tasks of (b) put every monomial $`p^\alpha`$ with $`\lvert\alpha\rvert\le N`$, restricted to $`\mathcal Y`$, into $`\mathcal G`$. Every law on the finite grid is admissible, so by [Theorem 4.2(a)](../statements/04-02_sufficiency_duality_transfer.md#s4-2) (equal feature means imply equal response means for all laws exactly when the responses lie in the feature span) a sufficient interface must satisfy $`\mathrm{span}\mathcal G\subseteq H_\phi`$. By Claim 3, $`\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)\ge\binom{k+N}N`$, hence $`d\ge\binom{k+N}N-1`$.

*(d), binary barrier.*

- *Every reward table is a response.* With $`p=Z`$ and a deployment that visits each of the $`k\le N`$ contexts once, the action vector equals $`Z\in\lbrace0,1\rbrace^k`$, so every terminal reward $`f:\lbrace0,1\rbrace^k\to[0,1]`$ is realized as $`g=f`$. Thus $`\mathcal G=[0,1]^{\mathcal Y}`$.
- *Fewer than $`2^k-1`$ features give discrepancy 1.* If $`d<2^k-1`$, then $`\dim H_\phi\le d+1<2^k=\lvert\mathcal Y\rvert`$, so some $`v\in H_\phi^\perp\setminus\lbrace0\rbrace`$ exists. Because $`\mathbf 1\in H_\phi`$, $`v_\pm`$ have equal mass $`s>0`$. The laws $`P=v_+/s`$ and $`Q=v_-/s`$ have disjoint supports and $`(P-Q)\phi_j=v^\top\phi_j/s=0`$. For $`f=\mathbf 1_{\mathrm{supp}\,v_+}\in\mathcal G`$, $`(P-Q)f=1`$. No discrepancy exceeds 1 because responses lie in $`[0,1]`$.
- *With $`2^k-1`$ features the discrepancy is 0.* If $`\lbrace\mathbf 1,\phi_1,\dots,\phi_{2^k-1}\rbrace`$ are linearly independent, then $`H_\phi=\mathbb R^{\mathcal Y}`$ and equal feature means force $`P=Q`$. $`\square`$

## Remarks

- **Universal quantifier.** Necessity in (b) is over the whole task and learner family. A fixed task, or a fixed update recipe, may need far fewer moment combinations.
- **What is preserved.** (b) preserves the expected final task return, not the law of the trained parameter, nor tail events absent from the task class.
- **Size of the witnesses.** $`\lvert\Delta_N\rvert=N!/(2^{N-1}N^N)\sim2\sqrt{2\pi N}(2e)^{-N}`$ decays exponentially, so the scalar witness proves exact necessity of every order, not fixed-tolerance hardness. The parity witness gives a constant gap at every order, at the price of $`k=N`$ coordinates. [Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity.md#s4-6) (smooth values need only about $`\sqrt N`$ orders approximately, parity needs all $`N`$) develops this contrast.
- **The bound $`3^N`$ in (4.3) is conservative.** The [experiments page](../experiments/04-04_affine_learners_moment_order.md) reports how close adversarial deployments come to it.
- **Where the assumptions bind.**
  - The factorization (4.4) needs IID blocks that do not depend on the state; on-policy feedback that depends on the current student breaks it.
  - Convex nonnegative weights fixed in advance are needed for the expansion; data-dependent weights break it.
  - Degree is measured in the probability coordinates $`p`$. A reparameterization such as $`p=\sigma(\theta)`$ generally destroys finite degree.
- **(d), binary barrier, is deliberately broad.** Arbitrary terminal payoffs make every function a response. Structured task families can have far smaller dimension.

## Related work

- Finite differences, the independence of parity from proper coordinate subsets, and the linear independence of monomials on a grid are classical tools; they are credited as such and not claimed as new.
- What is added is their use to compute, for convex-affine learners and deployments with at most $`N`$ decisions, the exact information a feedback source must carry.
