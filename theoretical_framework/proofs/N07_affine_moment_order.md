# N07 · Affine learners: moment order is necessary and sufficient — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N07_affine_moment_order.md) · [中文](N07_affine_moment_order_ZH.md)

## Setting and assumptions

**Student and deployment (§7.1).**

- The student has $k$ binary-action contexts and a parameter $p\in[0,1]^k$. At a decision whose context is $j$, it takes action 1 with conditional probability $p_j$, given the entire past.
- The deployment process is a finite tree with three kinds of nodes:
  - *chance nodes*, whose transition probabilities do not depend on $p$;
  - *decision nodes*, each labelled by a context $j$ that may depend on the history but not on $p$ directly;
  - *terminal nodes*, carrying rewards in $[0,1]$ that do not depend on $p$.
- Every root-to-leaf path contains at most $N$ decision nodes. A context may recur along a path.
- $J_E(p)$ is the expected terminal reward.

**Experience and learner (23).**

- Each experience block is $Z\in[0,1]^k$. Its coordinates may be arbitrarily dependent within the block.
- Blocks $Z_1,\dots,Z_R$ are IID from $P$ (or from $Q$).
- The learner is the convex affine average
$$p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i,\qquad w_i\ge0,\quad\sum_{i=0}^Rw_i=1,$$
  where $p_0\in[0,1]^k$ is common and fixed, and the weights are fixed independently of the data. Then $p_R\in[0,1]^k$.

**Notation.**

- For a multi-index $\alpha\in\mathbb N^k$, write $p^\alpha=\prod_jp_j^{\alpha_j}$, $\lvert\alpha\rvert=\sum_j\alpha_j$, and $m_P(\alpha)=\mathbb E_PZ^\alpha$.
- For a polynomial $f=\sum_\alpha c_\alpha p^\alpha$, write $\lVert f\rVert_{\rm coef}=\sum_\alpha\lvert c_\alpha\rvert$.

## Statement

**Lemma 10 (polynomial deployment value).** $J_E$ has a unique representation
$$J_E(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha,\qquad \sum_\alpha\lvert c_\alpha\rvert\le3^N, \tag{24}$$
and $J_E(p)\in[0,1]$ on $[0,1]^k$.

**Theorem 11 (necessary and sufficient joint information order).** The following are equivalent.

1. $m_P(\alpha)=m_Q(\alpha)$ for every $\lvert\alpha\rvert\le N$.
2. For every deployment task of §7.1 with at most $N$ decisions, and every affine learner (23), $\ \mathbb E_{P^R}J_E(p_R)=\mathbb E_{Q^R}J_E(p_R)$.

**Theorem 12 (sharp hierarchy; constant-gap witness).**

(i) *Scalar witness.* Let $N\ge1$ and $z_i=i/N$ for $i=0,\dots,N$. Define $P_N(z_i)=\binom Ni/2^{N-1}$ for even $i$ (and $0$ for odd $i$), and $Q_N(z_i)=\binom Ni/2^{N-1}$ for odd $i$ (and $0$ for even $i$). Then
$$m_{P_N}(\ell)=m_{Q_N}(\ell)\ \ (0\le\ell<N),\qquad \Delta_N:=m_{P_N}(N)-m_{Q_N}(N)=\frac{(-1)^NN!}{2^{N-1}N^N}. \tag{26}$$
The task $J(p)=p^N$ with the one-update learner $p=Z$ realizes the gap $\Delta_N$.

(ii) *Parity witness.* Let $k=N$, let $P$ be uniform on even-parity vectors of $\lbrace 0,1\rbrace ^N$, and let $Q$ be uniform on odd-parity vectors. All joint moments of total degree $<N$ agree. With
$$J_{\rm even}(p)=\tfrac12\Big(1+\prod_{j=1}^N(1-2p_j)\Big), \tag{27}$$
the one-update learner $p=Z$ gives $\mathbb E_PJ_{\rm even}(Z)=1$ and $\mathbb E_QJ_{\rm even}(Z)=0$.

**Theorem 13 (sharp statistic counts; binary barrier).**

(i) On the grid $\mathcal Y=\lbrace 0,\tfrac1N,\dots,1\rbrace ^k$, the least number of linear expectation statistics of a block that is sufficient for all tasks and learners of Theorem 11 is $\binom{k+N}{N}-1$.

(ii) On $\mathcal Y=\lbrace 0,1\rbrace ^k$, with $N\ge k$ and arbitrary terminal rewards on the $k$-action vector, every feature interface with fewer than $2^k-1$ statistics has worst-pair discrepancy exactly $1$. Hence tolerating any discrepancy $\varepsilon<1$ does not reduce this worst-case dimension. With $2^k-1$ affinely independent features, the discrepancy is $0$.

## Proof

### Lemma A (coefficient norm under the elementary operations)

Let $f$ and $g$ be polynomials in $p$, and $j\in\lbrace 1,\dots,k\rbrace $.

- $\lVert p_jf\rVert_{\rm coef}=\lVert f\rVert_{\rm coef}$, because $p^\beta\mapsto p^{\beta+e_j}$ maps distinct monomials to distinct monomials.
- $\lVert f+g\rVert_{\rm coef}\le\lVert f\rVert_{\rm coef}+\lVert g\rVert_{\rm coef}$.
- For weights $\pi_e\ge0$ with $\sum_e\pi_e=1$, $\lVert\sum_e\pi_ef_e\rVert_{\rm coef}\le\max_e\lVert f_e\rVert_{\rm coef}$.

### Proof of Lemma 10

For a node $u$ of the deployment tree, let $V_u(p)$ be the expected terminal reward from $u$. Let $n(u)$ be the maximum number of decision nodes on a path from $u$ to a leaf. We show by induction on the height of $u$ that
$$\deg V_u\le n(u),\qquad\lVert V_u\rVert_{\rm coef}\le3^{n(u)}.$$

- *Leaf.* $V_u=r\in[0,1]$ is constant, with norm $\le1=3^0$.
- *Chance node.* $V_u=\sum_e\pi_eV_{u_e}$ with $p$-independent probabilities. Every child has $n(u_e)\le n(u)$, so the degree is at most $n(u)$ and, by Lemma A, the norm is at most $\max_e3^{n(u_e)}\le3^{n(u)}$.
- *Decision node with context $j$.* The children are $u_1$ (action 1) and $u_0$ (action 0), and
$$V_u=p_jV_{u_1}+(1-p_j)V_{u_0}=V_{u_0}+p_jV_{u_1}-p_jV_{u_0}.$$
  The degree is at most $1+\max(\deg V_{u_1},\deg V_{u_0})\le1+(n(u)-1)=n(u)$. By Lemma A the norm is at most $3\cdot3^{n(u)-1}=3^{n(u)}$.

At the root, $n\le N$. Since $V_u$ is a probability-weighted average of rewards in $[0,1]$, $J_E\in[0,1]$.

*Uniqueness.* If two polynomials agree on $[0,1]^k$, their difference vanishes on a set with nonempty interior. All its partial derivatives therefore vanish at an interior point, so every coefficient is zero. $\square$

### Lemma B (block-moment expansion of an affine learner)

For every multi-index $\alpha$ with $\lvert\alpha\rvert=a$, list the coordinate factors of $p_R^\alpha$ as $j(1),\dots,j(a)$, where coordinate $j$ appears $\alpha_j$ times. Then
$$p_R^\alpha=\sum_{s:\lbrace 1,\dots,a\rbrace \to\lbrace 0,\dots,R\rbrace }\ \Big(\prod_{l=1}^aw_{s(l)}\Big)\ \prod_{l:s(l)=0}p_{0,j(l)}\ \prod_{i=1}^R Z_i^{\beta_i(s)}, \tag{B1}$$
where $\beta_i(s)=\sum_{l:s(l)=i}e_{j(l)}$. The coefficients are nonnegative and sum to $\prod_l\sum_iw_i=1$. Moreover:

- $\sum_i\lvert\beta_i(s)\rvert\le a$;
- at most $a$ blocks have $\beta_i(s)\ne0$;
- under IID blocks,
$$\mathbb E\,p_R^\alpha=\sum_s\Big(\prod_lw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^Rm(\beta_i(s)). \tag{B2}$$

*Proof.* Expand the product of the $a$ linear forms $w_0p_{0,j(l)}+\sum_{i\ge1}w_iZ_{i,j(l)}$ by distributivity, choosing one source $s(l)$ for each factor. Independence across blocks factors the expectation of each product of block monomials. $\square$

### Proof of Theorem 11

(1 ⇒ 2) By Lemma 10, $J_E=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha$, so $\mathbb EJ_E(p_R)=\sum_\alpha c_\alpha\,\mathbb Ep_R^\alpha$. In (B2), every block moment that appears has $\lvert\beta_i\rvert\le\lvert\alpha\rvert\le N$, and these are equal under $P$ and $Q$ by (1). Hence the two learning values coincide.

(2 ⇒ 1) Fix $\alpha$ with $1\le\lvert\alpha\rvert\le N$; the case $\alpha=0$ is trivial, since both moments are 1. Consider the deployment that visits context $j$ exactly $\alpha_j$ times, in a fixed order, and pays 1 exactly when every action taken is action 1. It has $\lvert\alpha\rvert\le N$ decisions and value $J(p)=p^\alpha$. Take the learner with $R=1$, $w_0=0$ and $w_1=1$, so $p_1=Z_1$. Its learning value is $m_P(\alpha)$ under $P$ and $m_Q(\alpha)$ under $Q$, so (2) forces equality. $\square$

*Example (marginals do not suffice).* Let $P=\tfrac12\delta_{(0,0)}+\tfrac12\delta_{(1,1)}$ and $Q=\tfrac12\delta_{(0,1)}+\tfrac12\delta_{(1,0)}$. Both coordinates are $\mathrm{Bernoulli}(\tfrac12)$ under both laws. Yet $m_P(1,1)=\tfrac12$ and $m_Q(1,1)=0$. The task "visit both contexts once, pay 1 if both actions are 1" separates the trained students by $\tfrac12$.

### Proof of Theorem 12

**Finite differences.** Let $(\mathsf Ef)(x)=f(x+1)$ and $\Delta=\mathsf E-I$. By the binomial theorem for the commuting operators $\mathsf E$ and $I$,
$$(\Delta^Nf)(0)=\sum_{i=0}^N(-1)^{N-i}\binom Ni f(i).$$
If $f$ is a polynomial of degree $d\ge1$ with leading coefficient $c$, then $\Delta f$ has degree $d-1$ and leading coefficient $cd$. Hence $\Delta^Nf=0$ when $\deg f<N$, and $\Delta^N x^N=N!$.

**(i) Scalar witness.**

- *Mass.* For $N\ge1$, $\sum_{i\text{ even}}\binom Ni=\sum_{i\text{ odd}}\binom Ni=2^{N-1}$, since their difference is $(1-1)^N=0$ and their sum is $2^N$. So $P_N$ and $Q_N$ are probability laws.
- *Moments.* For $\ell\ge0$,
$$m_{P_N}(\ell)-m_{Q_N}(\ell)=\frac{1}{2^{N-1}N^\ell}\sum_{i=0}^N(-1)^i\binom Ni\,i^\ell=\frac{(-1)^N}{2^{N-1}N^\ell}\,\big(\Delta^Nx^\ell\big)(0).$$
  This vanishes for $\ell<N$, and equals $(-1)^NN!/(2^{N-1}N^N)$ for $\ell=N$.
- *Realization.* The deployment that visits one context $N$ times and pays 1 only if all actions are 1 has $J(p)=p^N$. With $p=Z$ its learning gap is $\Delta_N$.

**(ii) Parity witness.**

*Proper subsets are uniform.* Let $S\subsetneq\lbrace 1,\dots,N\rbrace $ and fix $x_S\in\lbrace 0,1\rbrace ^S$. The complement $S^c\ne\varnothing$ has $2^{N-\lvert S\rvert}$ completions, exactly half of each parity: flipping one fixed coordinate of $S^c$ is a parity-reversing bijection. So $P(Z_S=x_S)=Q(Z_S=x_S)=2^{-\lvert S\rvert}$.

*Moments.* For $\lvert\alpha\rvert<N$, the support $S=\lbrace j:\alpha_j>0\rbrace $ has $\lvert S\rvert\le\lvert\alpha\rvert<N$. On binary vectors $Z_j^{\alpha_j}=Z_j$ for $\alpha_j\ge1$, so $Z^\alpha=\prod_{j\in S}Z_j$ depends only on $Z_S$. Its expectations under $P$ and $Q$ therefore agree.

*Value.* Let the deployment visit each context once, with independent actions $a_j\sim\mathrm{Bernoulli}(p_j)$, and pay 1 if $\sum_ja_j$ is even. Then
$$\Pr(\text{even})=\tfrac12\big(1+\mathbb E(-1)^{\sum_ja_j}\big)=\tfrac12\Big(1+\prod_j(1-2p_j)\Big)=J_{\rm even}(p).$$
At binary $p=Z$ the actions equal $Z$ deterministically, so $J_{\rm even}(Z)=\mathbf 1\lbrace Z\text{ even}\rbrace $. This gives values $1$ under $P$ and $0$ under $Q$. $\square$

### Proof of Theorem 13

**Lemma C (grid independence).** On $\mathcal Y=\lbrace 0,\tfrac1N,\dots,1\rbrace ^k$, the monomials $\lbrace p^\alpha:\max_j\alpha_j\le N\rbrace $ are linearly independent as functions on $\mathcal Y$. In particular, so are the $\binom{k+N}N$ monomials with $\lvert\alpha\rvert\le N$.

*Proof.* Induct on $k$.

- For $k=1$, a univariate polynomial of degree $\le N$ that vanishes at $N+1$ distinct points is zero.
- For $k>1$, write $f=\sum_{a=0}^Np_1^af_a(p_2,\dots,p_k)$ and suppose $f$ vanishes on $\mathcal Y$. Fix $(p_2,\dots,p_k)$ in the subgrid. Then $p_1\mapsto f$ has degree $\le N$ and vanishes at $N+1$ points, so every $f_a$ vanishes on the $(k-1)$-dimensional subgrid. By induction every $f_a$ is the zero polynomial.

Total-degree monomials form a subset of these. Their number is $\binom{k+N}{N}$, by stars and bars: $\alpha\in\mathbb N^k$ with $\lvert\alpha\rvert\le N$ correspond to $\alpha\in\mathbb N^{k+1}$ with $\lvert\alpha\rvert=N$. $\square$

**(i) Count.**

- *Upper bound.* By Theorem 11, the $\binom{k+N}N-1$ nonconstant moments $\lbrace m(\alpha):1\le\lvert\alpha\rvert\le N\rbrace $ are sufficient.
- *Lower bound.* The response class $\mathcal G$ of the one-update learners $p=Z$, combined with the monomial tasks of Theorem 11, contains every monomial $p^\alpha$ with $\lvert\alpha\rvert\le N$, restricted to $\mathcal Y$. Every law on the finite grid is admissible. Any sufficient interface must make equal feature means force equal values of all these responses, so by N06(a), $\operatorname{span}\mathcal G\subseteq H_\phi$. By Lemma C, $\dim\operatorname{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)\ge\binom{k+N}N$. Hence $d\ge\binom{k+N}N-1$.

**(ii) Binary barrier.**

*Every reward table is a response.* Take the learner $p=Z$ and a deployment that visits each of the $k\le N$ contexts once. Since $Z\in\lbrace 0,1\rbrace ^k$, the action vector equals $Z$, so any terminal reward $f:\lbrace 0,1\rbrace ^k\to[0,1]$ is realized: $g=f$. Thus $\mathcal G=[0,1]^{\mathcal Y}$.

*Fewer than $2^k-1$ features give discrepancy 1.* Let $d<2^k-1$. Then $\dim H_\phi\le d+1<2^k=\lvert\mathcal Y\rvert$, so there is $v\in H_\phi^\perp\setminus\lbrace 0\rbrace $. Since $\mathbf 1\in H_\phi$, the parts $v_\pm$ have equal mass $s>0$. Set $P=v_+/s$ and $Q=v_-/s$. These are laws with disjoint supports, and $(P-Q)\phi_j=v^\top\phi_j/s=0$. For $f=\mathbf 1_{\operatorname{supp}v_+}\in\mathcal G$,
$$(P-Q)f=1-0=1.$$
Every response lies in $[0,1]$, so no discrepancy exceeds 1. Hence the worst-pair discrepancy equals 1, and every $\varepsilon<1$ requires $d\ge2^k-1$.

*With $2^k-1$ features, the discrepancy is 0.* If $\lbrace \mathbf 1,\phi_1,\dots,\phi_{2^k-1}\rbrace $ are linearly independent, then $H_\phi=\mathbb R^{\mathcal Y}$. Equal feature means then force $P=Q$, so the discrepancy is 0. $\square$

## Remarks

1. **Universal quantifier.** Necessity in Theorem 11 holds over the *whole* task/learner family. A fixed task, or a fixed update recipe, may require far fewer moment combinations.
2. **What is preserved.** Theorem 11 preserves the expected final task return. It does not preserve the law of the trained parameter, nor tail events absent from the task class.
3. **Tightness of (24).** The bound $3^N$ is conservative. The best ratio $\sum\lvert c_\alpha\rvert/3^N$ found by adversarial search is $2/3$ (experiments, C6).
4. **Size of the scalar witness.** $\lvert\Delta_N\rvert=N!/(2^{N-1}N^N)\sim2\sqrt{2\pi N}(2e)^{-N}$ decays exponentially. It proves *exact* necessity of every order, not fixed-tolerance hardness. The parity witness supplies the constant gap at every order, at the price of $k=N$ coordinates. N09 develops this contrast.
5. **Where the assumptions bind.**
   - IID, state-independent blocks are needed for the factorization (B2). On-policy feedback that depends on the current student breaks it.
   - Convex nonnegative weights fixed in advance are needed for the expansion; data-dependent weights break it.
   - Degree is measured in the probability coordinates $p$. Reparameterizing, e.g. $p=\sigma(\theta)$, generally destroys finite degree.
6. **Theorem 13(ii) is deliberately broad.** Arbitrary terminal payoffs make every function a response. Structured task families can have far smaller dimension (N09, C19).

## Dependencies and source

- **Uses:** N06 (the span criterion (18)–(19), applied in Theorem 13).
- **Used by:** N08 (quantitative rates), N10 (polynomial updates).
- **Source:** v0.14.1 main framework, §7, Lemma 10 and Theorems 11–13, eqs. (23)–(28), `docs/direction/v015_revision/Environment_Information_Theory_EN.md`. Independent review: `docs/direction/v014_handoff/Independent_Proof_Audit_EN.md`, A16–A18, B3 and C2. Finite differences and parity independence are classical tools.
