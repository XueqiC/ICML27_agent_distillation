# Theorem 4.4 · Affine learners: moment order is necessary and sufficient

*Statement*

[← Framework](../README.md) · [Statement](04-04_affine_learners_moment_order.md) · [Proof](../proofs/04-04_affine_learners_moment_order.md) · [Experiments](../experiments/04-04_affine_learners_moment_order.md) · [中文](04-04_affine_learners_moment_order_ZH.md)

**In one paragraph.** For a student that makes binary decisions, a deployment with at most $`N`$ decisions, and a learner that averages independent experience blocks, exactly what information must two sources of experience share for every such task and learner to give the same expected result? This theorem answers completely: all joint moments of the blocks up to order $`N`$, no fewer and no more. Every order is necessary, the exact count is a binomial coefficient on a grid, and with arbitrary payoffs on binary blocks nothing short of the full distribution helps. It is the fully computed instance of the general span criterion of [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2).

**Setting used on this page** ([Definition 4.3](04-03_binary_context_student_affine_learner.md#s4-3)).

- The student has $`k`$ binary-action contexts and parameter $`p\in[0,1]^k`$; at a decision with context $`j`$ it takes action 1 with probability $`p_j`$.
- A deployment is a finite tree of chance, decision and terminal nodes with $`p`$-independent probabilities, context labels and rewards in $`[0,1]`$; every path has at most $`N`$ decisions, and $`J_E(p)`$ is the expected reward.
- Experience blocks $`Z_1,\dots,Z_R\in[0,1]^k`$ are IID from a source law $`P`$ (or $`Q`$). The convex-affine learner is $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$ with fixed weights $`w_i\ge0`$, $`\sum_iw_i=1`$, and fixed $`p_0`$.
- $`p^\alpha=\prod_jp_j^{\alpha_j}`$ for a multi-index $`\alpha\in\mathbb N^k`$, $`\lvert\alpha\rvert=\sum_j\alpha_j`$, and $`m_P(\alpha)=\mathbb E_PZ^\alpha`$ is a joint moment of a block. The learning value of a source is $`\mathbb EJ_E(p_R)`$.
- A *linear expectation statistic* of a block is $`\mathbb E\phi(Z)`$ for a fixed function $`\phi`$; an interface of such statistics and its unresolved discrepancy are as in [Definition 4.1](04-01_feature_sufficiency_unresolved_discrepancy.md#s4-1).

<a id="s4-4"></a>

**Theorem 4.4 (Affine learners: moment order is necessary and sufficient).** In the setting of Definition 4.3:

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

**What it says.**

- (a) A deployment with at most $`N`$ decisions has an expected return that is a polynomial of degree at most $`N`$ in the student's action probabilities, with coefficient mass at most $`3^N`$.
- (b) Two sources give the same learning value for every such deployment and every averaging learner exactly when their joint moments agree up to order $`N`$. Marginals are not enough; cross-moments matter.
- (c) No order can be dropped: for every $`N`$ there are sources that agree on all lower moments and still differ. The scalar gap shrinks exponentially in $`N`$, while the parity gap stays at 1.
- (d) On a grid the count is exactly $`\binom{k+N}N-1`$. On binary blocks with arbitrary payoffs, any interface smaller than the full distribution can be fooled completely.

**Intuition.** The deployment value is a degree-$`N`$ polynomial in $`p`$, and averaging independent blocks turns the expectation of that polynomial into a combination of block moments of order at most $`N`$. Conversely, a deployment that asks context $`j`$ exactly $`\alpha_j`$ times and pays only if every answer is 1 reads off the single moment $`m(\alpha)`$.

**Role in the framework.** It shows, in a model where everything can be computed, that the information a feedback source must carry is set by the deployment: its length $`N`$ fixes the moment order. Theorem 4.5 adds rates, Proposition 4.6 separates exact from approximate order, and Theorem 4.7 shows how other learners change the order.

**Scope.**

- Necessity in (b) is over the whole task and learner family. A fixed task, or a fixed update recipe, may need far fewer moment combinations.
- (b) preserves the expected final task return, not the law of the trained parameter, nor tail events absent from the task class.
- The scalar witness proves exact necessity of every order, not fixed-tolerance hardness, because $`\lvert\Delta_N\rvert\sim2\sqrt{2\pi N}(2e)^{-N}`$ decays exponentially. The parity witness gives a constant gap at every order, at the price of $`k=N`$ coordinates.
- Where the assumptions bind: the factorization behind (b) needs IID blocks that do not depend on the state (on-policy feedback breaks it), convex weights fixed in advance (data-dependent weights break it), and degree measured in the probability coordinates $`p`$ (a reparameterization such as $`p=\sigma(\theta)`$ generally destroys finite degree).
- The binary barrier in (d) is deliberately broad: arbitrary terminal payoffs make every function a response. Structured task families can have far smaller dimension.

**Derivation outline.**

- (a) Backward induction on the deployment tree: a decision node multiplies by $`p_j`$ or $`1-p_j`$, raising the degree by one and the coefficient norm by at most a factor 3; chance nodes average.
- (b) Expanding $`p_R^\alpha`$ by distributivity gives a probability-weighted sum of products of block moments of order at most $`\lvert\alpha\rvert`$; block independence factors the expectation. Conversely the task "visit context $`j`$ exactly $`\alpha_j`$ times, pay 1 if all actions are 1" has value $`p^\alpha`$, so the one-step learner $`p=Z`$ reads off every moment.
- (c) The $`N`$-th finite difference annihilates polynomials of degree below $`N`$ and gives $`N!`$ on $`x^N`$; parity laws are uniform on every proper coordinate subset.
- (d) Monomials of degree at most $`N`$ are linearly independent on the grid, so the span criterion of Theorem 4.2(a) gives the count. On binary blocks every reward table is a response, and any interface of dimension below $`2^k`$ has a nonzero annihilating direction that splits into two disjointly supported laws.

**Illustration.** Two coordinates, $`P=\tfrac12\delta_{(0,0)}+\tfrac12\delta_{(1,1)}`$ and $`Q=\tfrac12\delta_{(0,1)}+\tfrac12\delta_{(1,0)}`$. Both have $`\mathrm{Bernoulli}(\tfrac12)`$ marginals, but the two-decision task "pay 1 if both actions are 1" separates the trained students by $`\tfrac12`$, since $`m_P(1,1)=\tfrac12`$ and $`m_Q(1,1)=0`$. The information is not the marginals but the joint moments up to order $`N=2`$.

**Uses.**

- [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2) — the span criterion behind the counts in (d).
- [Definition 4.3](04-03_binary_context_student_affine_learner.md#s4-3) — the student, deployments and the affine learner.

**Used by.**

- [Theorem 4.5](04-05_round_independent_bound_averaging_rates.md#s4-5) — quantitative bounds and rates from (a)–(c).
- [Proposition 4.6](04-06_exact_order_not_approximate_complexity.md#s4-6) — the scalar and parity witnesses.
- [Theorem 4.7](04-07_update_composition_amplifies_information_requirement.md#s4-7) — the base case of degree composition.

## Related work

- (a) evaluates a behaviour strategy in a finite tree with chance moves ([Kuhn, 1953](#ref-kuhn1953extensive)) by backward induction.
- The parity witness and the binary barrier of (d) are Fourier facts on the Boolean cube ([O'Donnell, 2014](#ref-odonnell2014analysis)): the even- and odd-parity laws differ only in the top Fourier coefficient, and the $`2^k`$ parity functions form a basis of all functions on $`\lbrace0,1\rbrace^k`$.
