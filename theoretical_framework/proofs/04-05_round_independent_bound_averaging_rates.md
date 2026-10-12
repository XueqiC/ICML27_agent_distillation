# Theorem 4.5 · A round-independent bound and averaging rates

*Proof*

[← Framework](../README.md) · [Statement](../statements/04-05_round_independent_bound_averaging_rates.md) · [Proof](04-05_round_independent_bound_averaging_rates.md) · [Experiments](../experiments/04-05_round_independent_bound_averaging_rates.md) · [中文](04-05_round_independent_bound_averaging_rates_ZH.md)

**What is proved here.** For learners that take fixed convex averages of independent experience blocks, the difference in expected deployment return between two sources is bounded by their largest moment mismatch, whatever the number of training rounds; when $`n`$ independent blocks are averaged the difference decays like $`n^{-(r-1)}`$, with $`r`$ the order of the first unmatched cumulant; and replaying one block does not make it decay. The proof uses the block-moment expansion of an affine learner, a telescoping bound for products, and the moment–cumulant formula.

**Setting used on this page** ([Definition 4.3](../statements/04-03_binary_context_student_affine_learner.md#s4-3)).

- Student $`p\in[0,1]^k`$ with action-1 probability $`p_j`$ in context $`j`$; deployments with at most $`N`$ decisions; IID blocks $`Z_i\in[0,1]^k`$ from a source law $`P`$ or $`Q`$; convex-affine learner $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$ with fixed $`p_0`$ and fixed weights $`w_i\ge0`$, $`\sum_{i=0}^Rw_i=1`$.
- $`p^\alpha=\prod_jp_j^{\alpha_j}`$, $`\lvert\alpha\rvert=\sum_j\alpha_j`$, $`m_P(\alpha)=\mathbb E_PZ^\alpha`$; $`e_j`$ is the $`j`$-th unit multi-index.
- By [Theorem 4.4(a)](../statements/04-04_affine_learners_moment_order.md#s4-4) (a deployment value is a bounded polynomial of degree $`N`$), the deployment value is

```math
J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha,\qquad\sum_\alpha\lvert c_\alpha\rvert\le3^N. \qquad(4.3)
```

- *Block-moment expansion* (Claim 2 in the proof of [Theorem 4.4](04-04_affine_learners_moment_order.md)). For $`\lvert\alpha\rvert=a`$, list the coordinate factors of $`p_R^\alpha`$ as $`j(1),\dots,j(a)`$, coordinate $`j`$ appearing $`\alpha_j`$ times, and for a source map $`s:\lbrace1,\dots,a\rbrace\to\lbrace0,\dots,R\rbrace`$ put $`\beta_i(s)=\sum_{l:s(l)=i}e_{j(l)}`$. Distributivity and block independence give

```math
\mathbb E\,p_R^\alpha=\sum_s\Big(\prod_lw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^Rm(\beta_i(s)), \qquad(4.4)
```

- with nonnegative coefficients $`\prod_lw_{s(l)}`$ summing to 1, and at most $`a`$ blocks with $`\beta_i(s)\ne0`$ for each $`s`$.
- Averaging learner $`\bar p_n=\frac1n\sum_{i=1}^nZ_i`$, mean $`\mu=\mathbb EZ`$, centred average $`W_n=\bar p_n-\mu`$. $`\kappa_s(P)`$ is the order-$`s`$ joint cumulant tensor of $`Z\sim P`$; $`D^sJ(\mu)`$ is the $`s`$-th derivative tensor of $`J`$ at $`\mu`$ and $`\langle D^sJ(\mu),T\rangle`$ its full contraction with a tensor $`T`$.
- Scalar witnesses ([Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order.md#s4-4)): $`P_N(i/N)=\binom Ni/2^{N-1}`$ on even $`i`$, $`Q_N(i/N)=\binom Ni/2^{N-1}`$ on odd $`i`$; they agree below order $`N`$ and differ at order $`N`$ by $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$.

**Statement ([Theorem 4.5](../statements/04-05_round_independent_bound_averaging_rates.md#s4-5)).** In the setting of Definition 4.3, with $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$ as in Theorem 4.4(a), write

```math
\delta_N(P,Q)=\max_{1\le\lvert\alpha\rvert\le N}\lvert m_P(\alpha)-m_Q(\alpha)\rvert,\qquad C_J=\sum_\alpha\lvert c_\alpha\rvert\,\lvert\alpha\rvert .
```

- **(a) Round-independent bound.** For every convex-affine learner,

```math
\big\lvert\mathbb E_{P^R}J(p_R)-\mathbb E_{Q^R}J(p_R)\big\rvert\le C_J\,\delta_N(P,Q)\le N3^N\delta_N(P,Q). \qquad(4.5)
```

- The bound does not depend on $`R`$, and it may be capped at 1.
- **(b) First unmatched cumulant.** Let $`P,Q`$ have the same mean $`\mu`$ and equal joint cumulants of all orders $`s<r`$, where $`2\le r\le N`$. With $`P,Q,k,N,J`$ fixed and $`n\to\infty`$,

```math
\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=\frac{\langle D^rJ(\mu),\kappa_r(P)-\kappa_r(Q)\rangle}{r!\,n^{r-1}}+O(n^{-r}). \qquad(4.6)
```

- If the displayed contraction vanishes the difference is $`O(n^{-r})`$; a $`\Theta(n^{1-r})`$ rate holds exactly when it is nonzero.
- **(c) Exact law for the scalar witnesses.** For the laws $`P_N,Q_N`$ of Theorem 4.4(c) and $`J(p)=p^N`$,

```math
\mathbb E_{P_N^n}\bar p_n^N-\mathbb E_{Q_N^n}\bar p_n^N=\frac{\Delta_N}{n^{N-1}}\quad\text{for every }n\ge1 .
```

- **(d) Unequal means.** If $`\mu_P\ne\mu_Q`$, then $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=J(\mu_P)-J(\mu_Q)+O(n^{-1})`$.
- **(e) Replay and deterministic descendants do not decay.** If the $`n`$ training blocks are $`Z^{(1)}=\dots=Z^{(n)}=T(Z_1)`$ for a single parent block $`Z_1`$ and a fixed deterministic map $`T`$ (replay is $`T=\mathrm{id}`$), then $`\bar p_n=T(Z_1)`$ for every $`n`$, and the learning-value difference equals $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$, independent of $`n`$.

**Proof.** Three claims are used.

*Claim 1 (product telescoping).* For $`a_1,\dots,a_q,b_1,\dots,b_q\in[0,1]`$, $`\lvert\prod_ia_i-\prod_ib_i\rvert\le\sum_i\lvert a_i-b_i\rvert`$. Indeed $`\prod_ia_i-\prod_ib_i=\sum_{l=1}^q\big(\prod_{i<l}b_i\big)(a_l-b_l)\big(\prod_{i>l}a_i\big)`$, and each bracketed product lies in $`[0,1]`$.

*Claim 2 (cumulants of the empirical mean).* For $`s\ge2`$, $`\kappa_s(W_n)=n^{1-s}\kappa_s(Z)`$, and $`\kappa_1(W_n)=0`$. Joint cumulants of order $`s\ge2`$ are translation invariant and multilinear, and they are additive over independent summands because every mixed cumulant of independent vectors vanishes. Hence $`\kappa_s\big(\tfrac1n\sum_iZ_i\big)=n^{-s}\sum_{i=1}^n\kappa_s(Z_i)=n^{1-s}\kappa_s(Z)`$; the mean of $`W_n`$ is zero.

*Claim 3 (moment–cumulant formula).* For a random vector $`W`$ with finite moments, $`\mathbb E[W_{j_1}\cdots W_{j_s}]=\sum_{\pi\in\Pi(s)}\prod_{B\in\pi}\kappa(W_{j_b}:b\in B)`$, where $`\Pi(s)`$ is the set of partitions of $`\lbrace1,\dots,s\rbrace`$. This is the defining inversion of cumulants ([Leonov & Shiryaev, 1959](../references.md#leonov1959method)).

*(a).* Fix $`\alpha`$ with $`1\le\lvert\alpha\rvert=a\le N`$. Write (4.4) as $`\mathbb Ep_R^\alpha=\sum_s\omega_s\pi_s\prod_{i=1}^Rm(\beta_i(s))`$ with $`\omega_s=\prod_lw_{s(l)}\ge0`$, $`\sum_s\omega_s=1`$ and $`\pi_s=\prod_{l:s(l)=0}p_{0,j(l)}\in[0,1]`$.

- For fixed $`s`$, at most $`a`$ blocks have $`\beta_i(s)\ne0`$, each with $`1\le\lvert\beta_i(s)\rvert\le N`$, and all block moments lie in $`[0,1]`$ because $`Z\in[0,1]^k`$.
- By Claim 1, $`\big\lvert\pi_s\prod_im_P(\beta_i(s))-\pi_s\prod_im_Q(\beta_i(s))\big\rvert\le\sum_{i:\beta_i(s)\ne0}\lvert m_P(\beta_i(s))-m_Q(\beta_i(s))\rvert\le a\,\delta_N`$.
- Summing with weights $`\omega_s`$ gives $`\lvert\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha\rvert\le\lvert\alpha\rvert\,\delta_N`$. The term $`\alpha=0`$ cancels, so $`\lvert\sum_\alpha c_\alpha(\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha)\rvert\le C_J\delta_N`$.

Finally $`C_J\le N\sum_\alpha\lvert c_\alpha\rvert\le N3^N`$ by (4.3), and both learning values lie in $`[0,1]`$. The bound is uniform in $`R`$ because $`\sum_s\omega_s=1`$ for every $`R`$.

*(b).* $`J`$ is a polynomial of degree at most $`N`$, so its Taylor expansion at $`\mu`$ terminates, and

```math
\mathbb EJ(\bar p_n)=\sum_{s=0}^N\frac1{s!}\langle D^sJ(\mu),\mathbb E\,W_n^{\otimes s}\rangle .
```

$`P`$ and $`Q`$ share $`\mu`$, so $`D^sJ(\mu)`$ is common. The $`s=0`$ term cancels and the $`s=1`$ term vanishes. For $`s\ge2`$ apply Claim 3 to $`W_n`$.

- Partitions with a singleton block vanish because $`\kappa_1(W_n)=0`$. A partition with $`q`$ blocks of sizes $`b_1,\dots,b_q\ge2`$ contributes, by Claim 2, $`\prod_ln^{1-b_l}\kappa_{b_l}(Z)=n^{q-s}\prod_l\kappa_{b_l}(Z)`$.
- Cumulants of order below $`r`$ agree, so a partition term can differ between $`P`$ and $`Q`$ only if some block has size $`b_l\ge r`$. Then $`s=\sum_lb_l\ge r+2(q-1)`$, hence $`q-s\le2-r-q\le1-r`$.
- Equality $`q-s=1-r`$ holds only for $`q=1`$ and $`s=r`$: if $`q=1`$ and $`s>r`$, then $`q-s=1-s\le-r`$; if $`q\ge2`$, then $`q-s\le2-r-q\le-r`$.

So the only differing term of order $`n^{1-r}`$ is the single block at $`s=r`$, which contributes $`\frac1{r!}\langle D^rJ(\mu),n^{1-r}(\kappa_r(P)-\kappa_r(Q))\rangle`$. Every other differing term carries a factor $`n^{-r}`$ or smaller. There are finitely many such terms ($`s\le N`$), each with a bounded coefficient since $`Z\in[0,1]^k`$ has bounded cumulants, so the remainder is $`O(n^{-r})`$ with a constant depending only on $`P,Q,J`$.

*(c).* For $`N=1`$ the difference is $`m_{P_1}(1)-m_{Q_1}(1)=\Delta_1`$. Let $`N\ge2`$.

- $`P_N`$ and $`Q_N`$ agree on raw moments of order below $`N`$, hence on cumulants of order below $`N`$, since $`\kappa_s`$ is a polynomial in moments of order at most $`s`$.
- In the argument of (b) with $`r=N`$, a differing partition must contain a block of size at least $`N`$, and $`s\le\deg J=N`$. The only such partition is the single block at $`s=N`$, so there is no remainder.
- $`D^N(p^N)=N!`$ and $`\kappa_N(P_N)-\kappa_N(Q_N)=m_{P_N}(N)-m_{Q_N}(N)=\Delta_N`$, because $`\kappa_N`$ equals $`m_N`$ minus a polynomial in lower moments, which agree. The difference is exactly $`\Delta_Nn^{1-N}`$.

*(d).* Expand around $`\mu_P`$ and around $`\mu_Q`$ separately. The $`s=0`$ terms give $`J(\mu_P)-J(\mu_Q)`$. For $`s\ge2`$, every partition into blocks of size at least 2 has $`q\le s/2`$, hence $`q-s\le-s/2\le-1`$, so all remaining terms are $`O(n^{-1})`$.

*(e).* $`\bar p_n=\frac1n\sum_{i=1}^nT(Z_1)=T(Z_1)`$, so $`\mathbb EJ(\bar p_n)=\mathbb EJ(T(Z_1))`$ for every $`n`$. $`\square`$

## Remarks

- **Why $`R`$ does not appear in (4.5).** The convex weights make the expansion coefficients a probability vector, whatever the number of blocks. Signed or unconstrained weights break this and can amplify discrepancies; data-dependent weights invalidate the factorization.
- **Population, not one run.** Both bounds concern the expected final return of the training procedure, not the score of one trained student.
- **Coordinate dependence.** $`C_J`$ and $`\delta_N`$ are measured in the monomial basis of the probability coordinates. Large $`N`$ can make (4.5) vacuous, which is why [Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity.md#s4-6) (approximate order can be far below exact order) replaces degree by approximation width.
- **What controls the rate.** In (4.6) the variable is the number $`n`$ of independent blocks. Replay, or deterministic descendants of a block (part (e)), increases training exposure but not $`n`$, and the order-$`r`$ discrepancy stays at its one-block value. This is the counting discipline behind [Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10) (the information floor after $`K`$ demonstrations) and the method of §9.
- **Fixed laws.** The $`O(n^{-r})`$ constant depends on $`(P,Q,J)`$; the expansion is not uniform over families of laws whose cumulants grow.

## Related work

- The moment–cumulant formula ([Leonov & Shiryaev, 1959](../references.md#leonov1959method)) and the additivity of cumulants over independent summands are classical; they are credited as such and not claimed as new.
- What is added is their use to separate independent evidence from exposure in the learning value of a feedback source.

## References

- [Leonov & Shiryaev, 1959](../references.md#leonov1959method) — the moment–cumulant formula.
