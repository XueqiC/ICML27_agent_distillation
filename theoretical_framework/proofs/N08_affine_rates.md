# N08 · Rates in the affine theory — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N08_affine_rates.md) · [中文](N08_affine_rates_ZH.md)

## Setting and assumptions

The setting of N07 applies: the student $`p\in[0,1]^k`$, the deployment value $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$ ([N07(a)](N07_affine_moment_order.md)), IID blocks $`Z_i\in[0,1]^k`$ drawn from a fixed law $`P`$ or $`Q`$, and the convex affine learner $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$ with weights fixed in advance.

**Definitions.**

The moment discrepancy and the degree-weighted coefficient mass of $`J`$ are

```math
\delta_N(P,Q)=\max_{1\le\lvert\alpha\rvert\le N}\lvert m_P(\alpha)-m_Q(\alpha)\rvert,\qquad C_J=\sum_\alpha\lvert c_\alpha\rvert\,\lvert\alpha\rvert .
```

For the averaging learner $`\bar p_n=\frac1n\sum_{i=1}^nZ_i`$, write $`\mu=\mathbb E Z`$ and $`W_n=\bar p_n-\mu`$.

$`\kappa_s(P)`$ is the order-$`s`$ joint cumulant tensor of $`Z\sim P`$, with entries $`\kappa(Z_{j_1},\dots,Z_{j_s})`$.

$`D^sJ(\mu)`$ is the $`s`$-th derivative tensor, and $`\langle D^sJ(\mu),T\rangle=\sum_{j_1,\dots,j_s}\partial_{j_1\cdots j_s}J(\mu)\,T_{j_1\cdots j_s}`$.

For (b), $`P,Q,k,N,J`$ are fixed while $`n\to\infty`$.

## Statement

**(a)** **Theorem (round-independent discrepancy bound).** For every affine learner [N07](N07_affine_moment_order.md),

```math
\big\lvert\mathbb E_{P^R}J(p_R)-\mathbb E_{Q^R}J(p_R)\big\rvert\le C_J\,\delta_N(P,Q)\le N3^N\delta_N(P,Q). \qquad(29)
```

The bound does not depend on $`R`$ and may be capped at 1.

**(b)** **Theorem (first unmatched cumulant).** Let $`P,Q`$ have the same mean $`\mu`$, and let their joint cumulants agree for all orders $`s<r`$, where $`2\le r\le N`$. Then

```math
\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=\frac{\langle D^rJ(\mu),\kappa_r(P)-\kappa_r(Q)\rangle}{r!\,n^{r-1}}+O(n^{-r}). \qquad(30)
```

If the displayed contraction vanishes, the difference is $`O(n^{-r})`$. A $`\Theta(n^{1-r})`$ rate holds exactly when the contraction is nonzero.

**(c)** **Corollary (exact law for the scalar witnesses).** For the laws [N07(c)](N07_affine_moment_order.md) and $`J(p)=p^N`$,

```math
\mathbb E_{P_N^n}\bar p_n^N-\mathbb E_{Q_N^n}\bar p_n^N=\frac{\Delta_N}{n^{N-1}}\quad\text{for every }n\ge1. \qquad(31)
```

**(b′)** **Corollary (unequal means).** If $`\mu_P\ne\mu_Q`$, then $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=J(\mu_P)-J(\mu_Q)+O(n^{-1})`$.

**(c′)** **Corollary (replay and descendants do not decay).** Let the $`n`$ training blocks be $`Z^{(1)}=\dots=Z^{(n)}=T(Z_1)`$ for a single parent block $`Z_1`$ and a fixed deterministic map $`T`$; for replay, $`T`$ is the identity. Then $`\bar p_n=T(Z_1)`$ for every $`n`$. The learning-value difference equals $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$, independent of $`n`$.

## Proof

### Lemma A (product telescoping)
For $`a_1,\dots,a_q,b_1,\dots,b_q\in[0,1]`$, $`\ \lvert\prod_ia_i-\prod_ib_i\rvert\le\sum_i\lvert a_i-b_i\rvert`$.

*Proof.* We have $`\prod_ia_i-\prod_ib_i=\sum_{l=1}^q\big(\prod_{i<l}b_i\big)(a_l-b_l)\big(\prod_{i>l}a_i\big)`$, and each bracketed product lies in $`[0,1]`$. $`\square`$

### Proof of
Fix $`\alpha`$ with $`1\le\lvert\alpha\rvert=a\le N`$. Apply the expansion [N07(b)](N07_affine_moment_order.md) of [N07](N07_affine_moment_order.md#lemma-b-block-moment-expansion-of-an-affine-learner) to $`p_R^\alpha`$:

```math
\mathbb E\,p_R^\alpha=\sum_s\omega_s\,\pi_s\prod_{i=1}^Rm(\beta_i(s)),\qquad\omega_s=\prod_lw_{s(l)}\ge0,\quad\sum_s\omega_s=1,\quad\pi_s=\prod_{l:s(l)=0}p_{0,j(l)}\in[0,1].
```

For a fixed assignment $`s`$, at most $`a`$ blocks have $`\beta_i(s)\ne0`$, and each such block has $`1\le\lvert\beta_i(s)\rvert\le a\le N`$. All block moments lie in $`[0,1]`$ because $`Z\in[0,1]^k`$. By Lemma A,

```math
\Big\lvert\pi_s\prod_im_P(\beta_i(s))-\pi_s\prod_im_Q(\beta_i(s))\Big\rvert\le\sum_{i:\beta_i(s)\ne0}\lvert m_P(\beta_i(s))-m_Q(\beta_i(s))\rvert\le a\,\delta_N .
```

Summing over $`s`$ with the weights $`\omega_s`$ gives $`\lvert\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha\rvert\le\lvert\alpha\rvert\,\delta_N`$. The constant term $`\alpha=0`$ cancels. Hence

```math
\Big\lvert\sum_\alpha c_\alpha\big(\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha\big)\Big\rvert\le\sum_\alpha\lvert c_\alpha\rvert\lvert\alpha\rvert\,\delta_N=C_J\delta_N .
```

Finally, $`C_J\le N\sum_\alpha\lvert c_\alpha\rvert\le N3^N`$ by [N07(a)](N07_affine_moment_order.md), and both values lie in $`[0,1]`$, so the difference is at most 1. The bound is uniform in $`R`$ because $`\sum_s\omega_s=1`$ for every $`R`$. $`\square`$

### Lemma B (cumulants of the empirical mean)
For $`s\ge2`$, $`\kappa_s(W_n)=n^{1-s}\kappa_s(Z)`$, and $`\kappa_1(W_n)=0`$.

*Proof.* Joint cumulants of order $`s\ge2`$ are invariant under translation and multilinear. They are additive over independent summands: every mixed cumulant of independent vectors vanishes. Hence

```math
\kappa_s\Big(\tfrac1n\sum_iZ_i\Big)=n^{-s}\sum_{i=1}^n\kappa_s(Z_i)=n^{1-s}\kappa_s(Z).
```

The mean of $`W_n`$ is zero. $`\square`$

### Lemma C (moment–cumulant formula)
For any random vector $`W`$ with finite moments,

```math
\mathbb E[W_{j_1}\cdots W_{j_s}]=\sum_{\pi\in\Pi(s)}\prod_{B\in\pi}\kappa\big(W_{j_b}:b\in B\big),
```

where $`\Pi(s)`$ is the set of partitions of $`\lbrace 1,\dots,s\rbrace`$. This is the defining inversion of cumulants (Leonov–Shiryaev). $`\square`$

### Proof of
*Taylor expansion.* $`J`$ is a polynomial of degree $`\le N`$, so its Taylor expansion at $`\mu`$ terminates:

```math
J(\mu+w)=\sum_{s=0}^N\frac1{s!}\langle D^sJ(\mu),w^{\otimes s}\rangle .
```

Hence

```math
\mathbb EJ(\bar p_n)=\sum_{s=0}^N\frac1{s!}\langle D^sJ(\mu),\mathbb E\,W_n^{\otimes s}\rangle. \qquad(T)
```

$`P`$ and $`Q`$ share $`\mu`$, so $`D^sJ(\mu)`$ is the same for both. The $`s=0`$ term cancels and the $`s=1`$ term vanishes.

*Partition orders.* For $`s\ge2`$, apply Lemma C to $`W_n`$. Partitions containing a singleton block vanish, since $`\kappa_1(W_n)=0`$. For a partition with $`q`$ blocks of sizes $`b_1,\dots,b_q\ge2`$, Lemma B gives the product

```math
\prod_{l=1}^q n^{1-b_l}\kappa_{b_l}(Z)=n^{q-s}\prod_l\kappa_{b_l}(Z).
```

*Which terms can differ.* Under $`P`$ and $`Q`$ the cumulants of order $`<r`$ agree. A partition term can differ between the two laws only if some block has size $`b_l\ge r`$. For such a partition,

```math
s=\sum_lb_l\ge r+2(q-1),\qquad\text{so}\qquad q-s\le2-r-q\le1-r.
```

Equality $`q-s=1-r`$ holds if and only if $`q=1`$ and $`s=r`$:

- if $`q=1`$ and $`s>r`$, then $`q-s=1-s\le-r`$;
- if $`q\ge2`$, then $`q-s\le2-r-q\le-r`$.

So the only differing term of order $`n^{1-r}`$ is the single-block partition at $`s=r`$. It contributes

```math
\frac1{r!}\langle D^rJ(\mu),\,n^{1-r}(\kappa_r(P)-\kappa_r(Q))\rangle .
```

Every other differing term carries a factor $`n^{-r}`$ or smaller. There are finitely many such terms, since $`s\le N`$ and the partitions are finite, and each coefficient is bounded because $`Z\in[0,1]^k`$ has bounded cumulants. Therefore the remainder is $`O(n^{-r})`$. Its constant depends only on $`P,Q,J`$. $`\square`$

### Proof of
For $`N=1`$, (31) reads $`\mathbb E\bar p_n=\mu`$, so the difference is $`m_{P_1}(1)-m_{Q_1}(1)=\Delta_1`$. Let $`N\ge2`$. $`P_N`$ and $`Q_N`$ agree on the raw moments of order $`<N`$, so they agree on all cumulants of order $`<N`$, because $`\kappa_s`$ is a polynomial in the moments of order $`\le s`$. Apply the argument of (b) with $`r=N`$. Since $`s\le\deg J=N`$, a differing partition must contain a block of size $`\ge N`$. The only such partition is the single block at $`s=N`$, so there is no remainder at all. Now $`D^N(p^N)=N!`$, and $`\kappa_N(P_N)-\kappa_N(Q_N)=m_{P_N}(N)-m_{Q_N}(N)=\Delta_N`$, because $`\kappa_N`$ equals $`m_N`$ minus a polynomial in lower moments, which agree. Hence the difference is exactly $`\Delta_N\,n^{1-N}`$. $`\square`$

### Proof of
Expand (T) around $`\mu_P`$ and around $`\mu_Q`$ separately. The $`s=0`$ terms give $`J(\mu_P)-J(\mu_Q)`$. For $`s\ge2`$, every partition into blocks of size $`\ge2`$ has $`q\le s/2`$, hence $`q-s\le-s/2\le-1`$. All remaining terms are therefore $`O(n^{-1})`$. $`\square`$

### Proof of
$`\bar p_n=\frac1n\sum_{i=1}^nT(Z_1)=T(Z_1)`$. The expectation of $`J(\bar p_n)`$ is therefore $`\mathbb EJ(T(Z_1))`$ for every $`n`$. $`\square`$

## Remarks

1. **Why $`R`$ does not appear in (29).** The convex weights make the expansion coefficients a probability vector, whatever the number of blocks. Signed or unconstrained weights break this and can amplify discrepancies. Data-dependent weights also invalidate the factorization.
2. **Population, not one run.** Both theorems concern the expected final return of the training procedure, not the score of an individual trained student.
3. **Coordinate dependence.** $`C_J`$ and $`\delta_N`$ are measured in the monomial basis of the probability coordinates. Large $`N`$ can make (29) vacuous, so N09 replaces degree by approximation width.
4. **What controls the rate.** In (30) the variable is the number $`n`$ of *independent* blocks. Replaying a block, or producing deterministic descendants of it (c′), increases training exposure but not $`n`$. The order-$`r`$ discrepancy then stays at its one-block value.
5. **Fixed laws.** The $`O(n^{-r})`$ constant depends on $`(P,Q,J)`$. The expansion is not uniform over families of laws whose cumulants grow.

## Dependencies and source

- **Uses:** [N07](N07_affine_moment_order.md) ([N07(a)](N07_affine_moment_order.md) and the block expansion [N07(b)](N07_affine_moment_order.md)).
- **Used by:** none directly.  is the quantitative endpoint of the affine branch, and it explains the replay-versus-independent-evidence distinction used in [N03](N03_few_shot_information_floor.md) and [N17](N17_ced_method.md).

**Source:** [v0.14.1 Theorem 14](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-14-a-discrepancy-bound-independent-of-training-rounds); [v0.14.1 Theorem 15](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-15-the-first-unmatched-interaction-controls-averaging-decay) — Thms. 14–15; eqs. (29)–(31). v0.14.1 main framework, Theorems 14–15 and eqs. (29)–(31), §7, [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md); Corollary 15c is the replay/descendant statement of §13.1 and the C18 specification. Independent review: [Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md), A19–A21. Cumulant additivity and the moment–cumulant formula are classical tools.
