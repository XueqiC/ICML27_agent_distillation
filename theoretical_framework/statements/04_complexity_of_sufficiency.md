# §4 Complexity of sufficiency — Statements

[← Framework](../README.md) · [Proofs](../proofs/04_complexity_of_sufficiency.md) · [Experiments](../experiments/04_complexity_of_sufficiency.md) · [中文](04_complexity_of_sufficiency_ZH.md)

### §4 · Complexity of sufficiency — exact statements

[Proofs](../proofs/04_complexity_of_sufficiency.md) · [Experiments](../experiments/04_complexity_of_sufficiency.md)

**What this section does.**

- It measures how much information two feedback laws must share to be interchangeable for learning, given that interchangeability means agreement on the response class $`\mathcal G`$ ([Theorem 3.2](../proofs/03_learning_consequence.md#s3-2)).
- In general the answer is a span dimension or an approximation width of $`\mathcal G`$ ([Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2)).
- For convex-affine learners it is computed completely: the joint moments up to the deployment length, with rates in the number of independent blocks ([Theorems 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4) and [4.5](../proofs/04_rates_and_composition.md#s4-5)); approximate complexity can be much smaller than the exact order ([Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6)).
- Nonlinear or composed updates change the required order, so the learner itself must be modelled ([Theorem 4.7](../proofs/04_rates_and_composition.md#s4-7)).

**Setting (Definitions [4.1](../proofs/04_complexity_of_sufficiency.md#s4-1) and [4.3](../proofs/04_complexity_of_sufficiency.md#s4-3)).**

- $`\mathcal Y`$ is a finite message alphabet with $`m`$ elements; $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ is a response class. A feature interface transmits the means of $`\phi_1,\dots,\phi_d`$; $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$.
- $`\mathcal M_\phi`$ is the set of feature-matched pairs of laws, and the unresolved discrepancy is $`\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\sup_{g\in\mathcal G}\lvert(P-Q)g\rvert`$. The exact dimension $`d_0`$ and the approximate dimension $`d_\varepsilon`$ are the least numbers of features giving discrepancy $`0`$ and at most $`\varepsilon`$.
- For §4's computable model: a student $`p\in[0,1]^k`$ takes action 1 in context $`j`$ with probability $`p_j`$; a deployment is a finite tree with at most $`N`$ decisions on every path and value $`J_E(p)`$; IID experience blocks $`Z_i\in[0,1]^k`$; a convex-affine learner $`p_R=w_0p_0+\sum_iw_iZ_i`$ with weights fixed in advance.

#### Theorem 4.2 · Sufficiency duality and transfer

**(a)** Equal feature means imply equal response means for all laws if and only if $`\mathrm{span}\,\mathcal G\subseteq H_\phi`$; hence $`d_0(\mathcal G)=\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1`$.

**(b)** The unresolved discrepancy is exactly twice the worst best-approximation error:

```math
\mathfrak D_\phi(\mathcal G)=2\sup_{g\in\mathcal G}\inf_{h\in H_\phi}\lVert g-h\rVert_\infty .
```

**(c)** If every target response is $`g=c+b^\top\phi+r`$ with $`\lVert b\rVert_1\le B`$ and $`\lVert r\rVert_\infty\le\tau`$, then $`\lvert(P-Q)g\rvert\le B\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau`$.

**(d)** If the learner family can deploy the better of two policies (values $`a`$ and $`a+\gamma`$) on any event $`A\subseteq\mathcal Y`$, the discrepancy is $`\gamma\,\mathrm{TV}(P,Q)`$ and $`d_0=m-1`$.

*Derivation outline.*

- (a) If $`g\in H_\phi`$, it is a constant plus a combination of features, and both parts have equal means. If $`g\notin H_\phi`$, its component $`v`$ orthogonal to $`H_\phi`$ is a direction along which the uniform law can be tilted without moving any feature mean, while $`(P-Q)g`$ moves by $`\propto\lVert v\rVert_2^2`$.
- (b) Hölder's inequality gives $`\lvert(P-Q)(g-h)\rvert\le2\lVert g-h\rVert_\infty`$ for every $`h\in H_\phi`$. For attainment, LP duality writes the sup-norm distance as a maximum of $`\sigma^\top g`$ over $`\sigma\perp H_\phi`$ with $`\lVert\sigma\rVert_1\le1`$, and every such $`2\sigma`$ is a difference of two laws.
- (c) The constant cancels, Hölder bounds the feature term by $`B`$ times the feature-mean gap, and the residual contributes at most $`2\tau`$.
- (d) The responses $`a+\gamma\mathbf 1_A`$ realize the event form of total variation, and the singletons span $`\mathbb R^m`$.

*Illustration.* With $`\mathcal Y=\lbrace1,2,3\rbrace`$ and the single feature $`\phi(y)=y`$, the laws $`P=(\tfrac12,0,\tfrac12)`$ and $`Q=(0,1,0)`$ have the same mean. The response $`g(y)=y^2`$ separates them by 1: $`\mathbb E_Pg=5`$ and $`\mathbb E_Qg=4`$. The feature $`\phi`$ is insufficient because $`y^2\notin\mathrm{span}\lbrace1,y\rbrace`$ on three points.

*Uses* [Definition 1.4](../README.md#s1-4), [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2). *Used by* [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4), [Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6), [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1), [Theorem 8.1](../proofs/08_verification.md#s8-1).

[Proof](../proofs/04_complexity_of_sufficiency.md#s4-2) · [Experiments](../experiments/04_complexity_of_sufficiency.md#c4-1) · proved

#### Theorem 4.4 · Affine learners: moment order is necessary and sufficient

**(a)** $`J_E(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$ uniquely, with $`\sum_\alpha\lvert c_\alpha\rvert\le3^N`$.

**(b)** $`\mathbb E_PZ^\alpha=\mathbb E_QZ^\alpha`$ for all $`\lvert\alpha\rvert\le N`$ if and only if $`\mathbb E_{P^R}J_E(p_R)=\mathbb E_{Q^R}J_E(p_R)`$ for every deployment with at most $`N`$ decisions and every convex-affine learner.

**(c)** Every order is necessary. The scalar laws $`P_N,Q_N`$ on $`\lbrace0,\tfrac1N,\dots,1\rbrace`$ (binomial weights on even versus odd points) agree below order $`N`$ and differ at order $`N`$ by $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$; the even- and odd-parity laws on $`\lbrace0,1\rbrace^N`$ agree below order $`N`$ and give the parity task values $`1`$ and $`0`$.

**(d)** On the grid $`\lbrace0,\tfrac1N,\dots,1\rbrace^k`$ exactly $`\binom{k+N}N-1`$ statistics are needed. On $`\lbrace0,1\rbrace^k`$ with $`N\ge k`$ and arbitrary rewards, fewer than $`2^k-1`$ statistics leave discrepancy $`1`$.

*Derivation outline.*

- (a) Backward induction on the deployment tree: a decision node multiplies by $`p_j`$ or $`1-p_j`$, raising the degree by one and the coefficient norm by at most a factor 3; chance nodes average.
- (b) Expanding $`p_R^\alpha`$ by distributivity gives a probability-weighted sum of products of block moments of order at most $`\lvert\alpha\rvert`$ (block independence factors the expectation). Conversely the task "visit context $`j`$ exactly $`\alpha_j`$ times, pay 1 if all actions are 1" has value $`p^\alpha`$, so the one-step learner $`p=Z`$ reads off every moment.
- (c) The $`N`$-th finite difference annihilates polynomials of degree below $`N`$ and gives $`N!`$ on $`x^N`$; parity laws are uniform on every proper coordinate subset.
- (d) Monomials of degree at most $`N`$ are linearly independent on the grid, so Theorem 4.2(a) gives the count; on binary blocks every reward table is a response, and any interface of dimension below $`2^k`$ has a nonzero annihilating direction that splits into two disjointly supported laws.

*Illustration.* Two coordinates, $`P=\tfrac12\delta_{(0,0)}+\tfrac12\delta_{(1,1)}`$ and $`Q=\tfrac12\delta_{(0,1)}+\tfrac12\delta_{(1,0)}`$: both have $`\mathrm{Bernoulli}(\tfrac12)`$ marginals, but the two-decision task "pay 1 if both actions are 1" separates the trained students by $`\tfrac12`$. Marginals are not the information; joint moments up to order $`N=2`$ are.

*Uses* [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2). *Used by* [Theorem 4.5](../proofs/04_rates_and_composition.md#s4-5), [Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6), [Theorem 4.7](../proofs/04_rates_and_composition.md#s4-7).

[Proof](../proofs/04_complexity_of_sufficiency.md#s4-4) · [Experiments](../experiments/04_complexity_of_sufficiency.md#c4-2) · proved

#### Theorem 4.5 · A round-independent bound and averaging rates

**(a)** For every convex-affine learner, $`\lvert\mathbb E_{P^R}J(p_R)-\mathbb E_{Q^R}J(p_R)\rvert\le C_J\,\delta_N(P,Q)\le N3^N\delta_N(P,Q)`$, where $`\delta_N`$ is the largest moment mismatch up to order $`N`$ and $`C_J=\sum_\alpha\lvert c_\alpha\rvert\lvert\alpha\rvert`$. The bound does not depend on $`R`$.

**(b)** For the average of $`n`$ independent blocks, with equal means and joint cumulants matched below order $`r`$,

```math
\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=\frac{\langle D^rJ(\mu),\kappa_r(P)-\kappa_r(Q)\rangle}{r!\,n^{r-1}}+O(n^{-r}).
```

**(c)** For the scalar witnesses the expansion is exact: the gap is $`\Delta_N/n^{N-1}`$ for every $`n\ge1`$.

**(d)** With unequal means the gap tends to $`J(\mu_P)-J(\mu_Q)`$ at rate $`O(n^{-1})`$.

**(e)** Replaying one block $`n`$ times, or training on $`n`$ deterministic descendants $`T(Z_1)`$ of it, leaves the gap $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$ for every $`n`$.

*Derivation outline.*

- (a) In the block-moment expansion the coefficients form a probability vector for every $`R`$, and a product of at most $`\lvert\alpha\rvert`$ block moments in $`[0,1]`$ changes by at most the sum of their changes (product telescoping).
- (b) Taylor-expand the polynomial $`J`$ around the common mean and write the centred moments of $`\bar p_n`$ through cumulants. A partition with $`q`$ blocks of total size $`s`$ scales as $`n^{q-s}`$, and the only differing term of order $`n^{1-r}`$ is the single block of size $`r`$.
- (c) For $`J=p^N`$ no other differing partition exists. (d) Partitions into blocks of size at least 2 are $`O(n^{-1})`$. (e) The average of identical copies is the copy.

*Illustration.* Two sources with the same mean and variance but different skewness ($`r=3`$) produce, for a deployment value whose third derivative at the mean does not annihilate the skewness difference, a gap that shrinks as $`1/n^2`$ when $`n`$ independent blocks are averaged. Ten epochs over one block leave the one-block gap untouched.

*Uses* [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4). *Used by* [Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/04_rates_and_composition.md#s4-5) · [Experiments](../experiments/04_complexity_of_sufficiency.md#c4-6) · proved

#### Proposition 4.6 · Exact order is not approximate complexity

**(a)** Truncating the Chebyshev expansion of $`p^N`$ at order $`q<N`$ has sup-error exactly $`\Pr\lbrace\lvert B-N\rvert>q\rbrace\le2e^{-q^2/N}`$ with $`B\sim\mathrm{Bin}(2N,\tfrac12)`$. Laws (or convex-affine learning values) matched through order $`q\ge\min\lbrace N,\lceil\sqrt{N\log(4/\varepsilon)}\rceil\rbrace`$ therefore differ by at most $`\varepsilon`$ on $`p^N`$.

**(b)** For the parity value $`J_{\rm even}`$ on $`[0,1]^N`$, $`\min_{h\in\mathcal P_q}\lVert J_{\rm even}-h\rVert_\infty=\tfrac12`$ for every $`q<N`$.

*Derivation outline.*

- (a) With $`2p-1=\cos\theta`$, $`p^N=\cos^{2N}(\theta/2)`$ expands into Chebyshev polynomials with binomial coefficients. The tail of nonnegative coefficients is the sup-error (attained at $`p=1`$) and is a binomial tail, bounded by Hoeffding.
- (b) The even- and odd-parity laws agree on every polynomial of degree below $`N`$ but differ by 1 on $`J_{\rm even}`$, which by Theorem 4.2(b) forces approximation error at least $`\tfrac12`$.

*Illustration.* For $`N=100`$ and $`\varepsilon=0.01`$, about $`\sqrt{100\log400}\approx25`$ matched moments already resolve $`p^{100}`$, whereas parity over 100 contexts needs all 100 orders.

*Uses* [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2), [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4). *Used by* [Theorem 8.1](../proofs/08_verification.md#s8-1).

[Proof](../proofs/04_rates_and_composition.md#s4-6) · [Experiments](../experiments/04_complexity_of_sufficiency.md#c4-9) · proved

#### Theorem 4.7 · Update composition amplifies the information requirement

**(a)** If the update $`p_i=U_i(p_{i-1},Z_i)`$ is polynomial of degree at most $`a_i`$ in $`p_{i-1}`$ and $`b_i`$ in $`Z_i`$, then $`J(p_R)`$ has degree at most $`D_i=Nb_i\prod_{t>i}a_t`$ in block $`Z_i`$, and matching each block's moments through $`D_i`$ suffices.

**(b)** The bound is attained: squaring updates turn $`Z_1`$ into $`Z_1^{2^{R-1}}`$.

**(c)** A threshold learner $`p=\mathbf 1\lbrace Z>\tfrac12\rbrace`$ separates two laws with equal means already for $`N=1`$.

**(d)** For one smooth SGD step $`\theta\mapsto\theta-\eta G`$,

```math
(K^P-K^Q)V=-\eta\nabla V^\top(\mu_P-\mu_Q)+\tfrac{\eta^2}{2}\langle\nabla^2V,M_P-M_Q\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\tfrac{M_3\eta^3}{6}\big(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3\big).
```

**(e)** Per-step errors obeying $`e_{t+1}\le(1+\eta L)e_t+C\eta^2`$ accumulate to $`C\eta^2\sum_{s<R}(1+\eta L)^s`$, which is linear in $`R`$ only when $`\eta LR=O(1)`$ or the error map contracts.

*Derivation outline.*

- (a) Pull $`J`$ backward through the updates: substituting a degree-$`(a_t,b_t)`$ map into a degree-$`d_t`$ polynomial multiplies the degree in $`p`$ by $`a_t`$ and creates degree $`b_td_t`$ in $`Z_t`$; independence then factors the expectation into per-block moments.
- (b) $`U_1(p,z)=z`$ followed by coordinatewise squaring, with the scalar witnesses of Theorem 4.4(c) at order $`2^{R-1}`$.
- (c) A direct computation with $`P(0)=\tfrac13`$, $`P(\tfrac34)=\tfrac23`$ against $`Q(\tfrac12)=1`$.
- (d) Third-order Taylor expansion of $`V(\theta-\eta g)`$ with integral remainder, then expectation.
- (e) Induction and a geometric sum; the recursion $`e_{t+1}=\tfrac32e_t+\tfrac14`$ shows that the linear simplification can fail.

*Illustration.* Feed the same two block laws to an averaging learner and to a learner that squares its parameter after each block. The first needs moments up to the deployment length; the second needs moments up to the deployment length times $`2^{R-1}`$. Information requirement is a property of the learner, not of the data alone.

*Uses* [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4), [Definition 1.4](../README.md#s1-4). *Used by* [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1), [Proposition 5.3](../proofs/05_optimizer_state.md#s5-3).

[Proof](../proofs/04_rates_and_composition.md#s4-7) · [Experiments](../experiments/04_complexity_of_sufficiency.md#c4-11) · proved
