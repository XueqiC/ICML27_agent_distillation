# §4 Rates and composition — Proofs

[← Framework](../README.md) · [Proofs](04_complexity_of_sufficiency.md) · [Experiments](../experiments/04_complexity_of_sufficiency.md) · [中文](04_rates_and_composition_ZH.md)

This page contains Theorem 4.5 through Theorem 4.7; see the [sibling page](04_complexity_of_sufficiency.md) for the preceding proofs, [feature sufficiency](04_complexity_of_sufficiency.md#feature-sufficiency) and the [affine-learner setting](04_complexity_of_sufficiency.md#s4-3).

<a id="s4-5"></a>

**Theorem 4.5 (A round-independent bound and averaging rates).** In the setting of Definition 4.3, with $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$ as in [Theorem 4.4(a)](04_complexity_of_sufficiency.md#s4-4), write

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
- **(c) Exact law for the scalar witnesses.** For the laws $`P_N,Q_N`$ of [Theorem 4.4(c)](04_complexity_of_sufficiency.md#s4-4) and $`J(p)=p^N`$,

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

**Proposition 4.6 (Exact order is not approximate complexity).** Let $`T_j`$ be the Chebyshev polynomials of the first kind ($`T_j(\cos\theta)=\cos(j\theta)`$, $`\lvert T_j\rvert\le1`$ on $`[-1,1]`$, $`T_j(1)=1`$) and $`\mathcal P_q`$ the polynomials of total degree at most $`q`$. Consider the scalar task $`J(p)=p^N`$ on $`[0,1]`$ (one context visited $`N`$ times, success if all actions are 1) and the parity task $`J_{\rm even}`$ of [Theorem 4.4(c)](04_complexity_of_sufficiency.md#s4-4) on $`[0,1]^N`$. For $`p\in[0,1]`$,

```math
p^N=4^{-N}\binom{2N}{N}+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}T_j(2p-1). \qquad(4.7)
```

- **(a) Smooth values have low approximate order.** Let $`q<N`$, let $`h_q\in\mathcal P_q`$ be the truncation of (4.7) after order $`q`$, and let $`B\sim\mathrm{Bin}(2N,\tfrac12)`$. Then:
  - $`\lVert p^N-h_q\rVert_{\infty,[0,1]}=\Pr\lbrace\lvert B-N\rvert>q\rbrace\le2e^{-q^2/N}`$;
  - if two laws $`P,Q`$ on $`[0,1]`$ have equal moments of orders $`\ell\le q`$, then $`\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert\le4e^{-q^2/N}`$;
  - the same bound holds for the learning values $`\mathbb E\,p_R^N`$ after any convex-affine learner with IID scalar blocks.

  Consequently, matching moments through order $`q\ge\min\lbrace N,\lceil\sqrt{N\log(4/\varepsilon)}\rceil\rbrace`$ guarantees a gap of at most $`\varepsilon`$, although exact equality for all laws requires order $`N`$ ([Theorem 4.4(c)](04_complexity_of_sufficiency.md#s4-4)).
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

*Order needed.* $`4e^{-q^2/N}\le\varepsilon`$ if and only if $`q\ge\sqrt{N\log(4/\varepsilon)}`$; for $`q\ge N`$ the gap is exactly 0 by [Theorem 4.4(b)](04_complexity_of_sufficiency.md#s4-4).

*(b).* *Upper bound.* $`J_{\rm even}`$ is a probability, so $`\lVert J_{\rm even}-\tfrac12\rVert_\infty\le\tfrac12`$.

*Lower bound.* Let $`h\in\mathcal P_q`$ with $`q<N`$, and let $`P,Q`$ be uniform on the even- and odd-parity vertices of $`\lbrace0,1\rbrace^N`$.

- On binary vertices $`p_j^a=p_j`$ for $`a\ge1`$, so $`h`$ restricted to $`\lbrace0,1\rbrace^N`$ is a multilinear polynomial whose monomials $`\prod_{j\in S}p_j`$ have $`\lvert S\rvert\le q<N`$.
- Every proper coordinate subset is uniform under both laws (proof of [Theorem 4.4(c)](04_complexity_of_sufficiency.md#s4-4)), so $`\mathbb E_Ph=\mathbb E_Qh`$.
- On the vertices $`J_{\rm even}(Z)=\mathbf 1\lbrace Z\ \text{even}\rbrace`$, so $`\mathbb E_PJ_{\rm even}=1`$ and $`\mathbb E_QJ_{\rm even}=0`$.

Therefore

```math
1=(P-Q)J_{\rm even}=(P-Q)(J_{\rm even}-h)\le\lVert P-Q\rVert_1\,\lVert J_{\rm even}-h\rVert_{\infty,\lbrace0,1\rbrace^N}\le2\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N},
```

so $`\lVert J_{\rm even}-h\rVert_\infty\ge\tfrac12`$. Equivalently, by [Theorem 4.2(b)](04_complexity_of_sufficiency.md#s4-2), the matched pair $`(P,Q)`$ certifies $`\mathrm{dist}_\infty(J_{\rm even},\mathcal P_q)\ge\tfrac12`$. $`\square`$

*Remarks on Proposition 4.6.*

- **The contrast.** $`p^N`$ and $`J_{\rm even}`$ both have exact degree $`N`$, and exact sufficiency needs all moments through $`N`$ for both. Their approximate requirements differ completely:
  - $`p^N`$ is resolved to accuracy $`\varepsilon`$ by order $`O(\sqrt{N\log(1/\varepsilon)})`$;
  - $`J_{\rm even}`$ is not resolved below $`\tfrac12`$ by any order below $`N`$.
- The operative complexity is therefore the approximate dimension $`d_\varepsilon`$ of the response class ([Theorem 4.2(b)](04_complexity_of_sufficiency.md#s4-2)), not its degree.
- **Scope.** (a) is an upper bound on the sufficient order for this scalar task and the convex-affine learners; no matching lower bound on the approximate order is claimed.
- **Stability.**
  - Exact low-order matching is not a finite-sample certificate. If moments are only estimated, the power-basis coefficients of $`h_q`$ multiply the estimation errors, and they can be large.
  - Bounded Chebyshev features avoid power-basis amplification for one scalar message, but pulled back through several affine updates they need their own coefficient bound.
  - In general the relevant pair is $`(B,\tau)`$ of [Theorem 4.2(c)](04_complexity_of_sufficiency.md#s4-2), and [Theorem 8.1](08_verification.md#s8-1) makes the tradeoff exact.
- **Realizability.** Both witness families are realizable finite feedback protocols: a reset experiment can emit a block drawn from the specified law. This shows that the environments exist, not that a given benchmark permits them.

## The learner sets the order

<a id="s4-7"></a>

**Theorem 4.7 (Update composition amplifies the information requirement).** The setting has three parts.

- **Polynomial updates.** Fix $`R\ge1`$ and a common $`p_0\in[0,1]^k`$. Independent blocks $`Z_1,\dots,Z_R`$ take values in a finite set $`\mathcal Y\subset[0,1]^k`$; block $`Z_i`$ has law $`P_i`$ under the reference source and $`Q_i`$ under the alternative. The learner is $`p_i=U_i(p_{i-1},Z_i)`$, where $`U_i`$ maps $`[0,1]^k\times\mathcal Y`$ into $`[0,1]^k`$ and every coordinate of $`U_i`$ is a polynomial with
  - **(D1)** degree at most $`a_i\ge1`$ in $`p_{i-1}`$ (as a polynomial in $`p_{i-1}`$ whose coefficients are polynomials in $`Z_i`$);
  - **(D2)** degree at most $`b_i\ge1`$ in $`Z_i`$ (as a polynomial in $`Z_i`$ whose coefficients are polynomials in $`p_{i-1}`$).

  The deployment value $`J:[0,1]^k\to[0,1]`$ is a polynomial of total degree at most $`N`$, which holds for every deployment of Definition 4.3 with at most $`N`$ decisions ([Theorem 4.4(a)](04_complexity_of_sufficiency.md#s4-4)).
- **Smooth local setting.** $`\theta\in\mathbb R^p`$, $`V:\mathbb R^p\to\mathbb R`$ is a continuation value ([Definition 1.4](../README.md#s1-4)), and one SGD step with consumed-gradient law $`P`$ acts as $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$, with $`\mu_P=\mathbb E_PG`$ and $`M_P=\mathbb E_PGG^\top`$.
  - **(T3)** $`V`$ is $`C^3`$ on every segment $`[\theta,\theta-\eta g]`$ with $`g`$ in the supports of $`P`$ and $`Q`$, and $`\lvert D^3V(\vartheta)[h,h,h]\rvert\le M_3\lVert h\rVert^3`$ there.
- **Error recursion.** A nonnegative sequence satisfies $`e_0=0`$ and $`e_{t+1}\le(1+\eta L)e_t+C\eta^2`$ for constants $`\eta,L,C\ge0`$.

Then:

- **(a) Degree composition.** Under (D1)–(D2), $`J(p_R)`$, as a polynomial in $`(Z_1,\dots,Z_R)`$, has total degree in block $`Z_i`$ at most

```math
D_i=N\,b_i\prod_{t=i+1}^Ra_t. \qquad(4.8)
```

  Consequently, if $`\mathbb E_{P_i}Z^\alpha=\mathbb E_{Q_i}Z^\alpha`$ for every $`i`$ and every $`\lvert\alpha\rvert\le D_i`$, then $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$. If all blocks share one source law, matching through $`\max_iD_i`$ suffices.
- **(b) Attainment.** For every $`R\ge1`$ there are updates in the class and a deployment with $`N=1`$ whose response is $`Z_1^{2^{R-1}}`$. The scalar laws of order $`M=2^{R-1}`$ from [Theorem 4.4(c)](04_complexity_of_sufficiency.md#s4-4) agree on all moments below $`M`$ yet give different learning values, so (4.8) cannot be improved in general.
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

- **The information order belongs to the learner composed with the deployment.** With one convex-affine update ($`a_i=b_i=1`$), (4.8) recovers $`D_i=N`$, which is [Theorem 4.4(b)](04_complexity_of_sufficiency.md#s4-4). Each nonlinear later update multiplies the requirement on all earlier blocks.
- **Degree is coordinate-dependent.** An invertible reparameterization can change the polynomial form of an update. The intrinsic object remains the response class and its approximate dimension ([Theorem 4.2](04_complexity_of_sufficiency.md#s4-2)); degree is a tractable certificate in the chosen probability coordinates.
- **Non-polynomial learners.** (c) shows that once the update is discontinuous no moment order is sufficient in general; even one decision imposes more than a first-moment requirement.
- **Scope of (4.9).** The expansion is local and one-step: its first-order statistic is the mean consumed gradient and its second-order statistic the gradient second moment. For a stateful optimizer the increment of the complete learner state replaces the parameter step ([Proposition 5.1](05_optimizer_state.md#s5-1)); cold and warm Adam are treated in [Proposition 5.3](05_optimizer_state.md#s5-3) and [Propositions 5.5–5.6](05_warm_adam.md#s5-5).
- **Accumulation needs stability.** (e) is sharp for its recursion. A bound linear in $`R`$ requires $`\eta LR=O(1)`$, since $`\sum_{s<R}(1+\eta L)^s\le Re^{\eta LR}`$, or a contractive error map.

## Related work

- LP duality, the event form of total variation, finite differences, parity independence, the moment–cumulant formula (Leonov–Shiryaev), Chebyshev expansions and Hoeffding's inequality are classical tools; they are credited as such and not claimed as new.
- What this section adds is their use to compute the information a feedback law must carry for a given learner and deployment class: a span dimension or approximation width of the response class, the moment order for convex-affine learners, and its amplification by composed updates.
