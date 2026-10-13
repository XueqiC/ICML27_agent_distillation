# Theorem 4.5 · A round-independent bound and averaging rates

*Statement*

[← Framework](../README.md) · [Statement](04-05_round_independent_bound_averaging_rates.md) · [Proof](../proofs/04-05_round_independent_bound_averaging_rates.md) · [Experiments](../experiments/04-05_round_independent_bound_averaging_rates.md) · [中文](04-05_round_independent_bound_averaging_rates_ZH.md)

**In one paragraph.** [Theorem 4.4](04-04_affine_learners_moment_order.md#s4-4) says which moments two sources of experience must share to be exactly interchangeable. This theorem makes it quantitative. If the moments differ a little, how much can the learning outcomes differ, and does training longer make it worse? And when a learner averages more and more independent blocks, how fast does the remaining difference vanish? The answers: the gap is bounded by a moment mismatch independently of the number of training rounds; it decays with the number of *independent* blocks at a rate fixed by the first unmatched cumulant; and replaying one block, or training on deterministic rewrites of it, does not make it decay at all.

**Setting used on this page** ([Definition 4.3](04-03_binary_context_student_affine_learner.md#s4-3)).

- Student $`p\in[0,1]^k`$ (action-1 probability per context), deployments with at most $`N`$ decisions, IID experience blocks $`Z_i\in[0,1]^k`$ from a source law $`P`$ or $`Q`$, and convex-affine learners $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$ with fixed nonnegative weights summing to 1.
- By [Theorem 4.4(a)](04-04_affine_learners_moment_order.md#s4-4) the deployment value is a polynomial $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$ with $`\sum_\alpha\lvert c_\alpha\rvert\le3^N`$.
- $`m_P(\alpha)=\mathbb E_PZ^\alpha`$ is a joint moment of a block; $`\lvert\alpha\rvert=\sum_j\alpha_j`$.
- The *averaging learner* is $`\bar p_n=\frac1n\sum_{i=1}^nZ_i`$, with $`\mu=\mathbb EZ`$ and centred average $`W_n=\bar p_n-\mu`$.
- The *scalar witnesses* $`P_N,Q_N`$ of Theorem 4.4(c) live on $`\lbrace0,\tfrac1N,\dots,1\rbrace`$: $`P_N(i/N)=\binom Ni/2^{N-1}`$ for even $`i`$, $`Q_N(i/N)=\binom Ni/2^{N-1}`$ for odd $`i`$ (zero elsewhere). They agree on all moments below order $`N`$ and differ at order $`N`$ by $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$.
- $`\kappa_s(P)`$ is the order-$`s`$ joint cumulant tensor of $`Z\sim P`$ (the multivariate cumulants: $`\kappa_1`$ is the mean, $`\kappa_2`$ the covariance, higher orders measure non-Gaussian structure). $`D^sJ(\mu)`$ is the $`s`$-th derivative tensor of $`J`$ at $`\mu`$, and $`\langle D^sJ(\mu),T\rangle=\sum_{j_1,\dots,j_s}\partial_{j_1\cdots j_s}J(\mu)\,T_{j_1\cdots j_s}`$.

<a id="s4-5"></a>

**Theorem 4.5 (A round-independent bound and averaging rates).** In the setting of Definition 4.3, with $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$ as in Theorem 4.4(a), write

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
- **(c) Exact law for the scalar witnesses.** For the laws $`P_N,Q_N`$ of [Theorem 4.4(c)](04-04_affine_learners_moment_order.md#s4-4) and $`J(p)=p^N`$,

```math
\mathbb E_{P_N^n}\bar p_n^N-\mathbb E_{Q_N^n}\bar p_n^N=\frac{\Delta_N}{n^{N-1}}\quad\text{for every }n\ge1 .
```

- **(d) Unequal means.** If $`\mu_P\ne\mu_Q`$, then $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=J(\mu_P)-J(\mu_Q)+O(n^{-1})`$.
- **(e) Replay and deterministic descendants do not decay.** If the $`n`$ training blocks are $`Z^{(1)}=\dots=Z^{(n)}=T(Z_1)`$ for a single parent block $`Z_1`$ and a fixed deterministic map $`T`$ (replay is $`T=\mathrm{id}`$), then $`\bar p_n=T(Z_1)`$ for every $`n`$, and the learning-value difference equals $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$, independent of $`n`$.

**What it says.**

- (a) A small moment mismatch causes a small outcome difference, and the number of training rounds does not enter the bound.
- (b) When $`n`$ independent blocks are averaged, the difference decays like $`n^{-(r-1)}`$, where $`r`$ is the order of the first cumulant on which the sources disagree, with an explicit leading coefficient.
- (c) For the witnesses of Theorem 4.4(c) the decay law is exact for every $`n`$.
- (d) If the means differ, the difference does not vanish: it tends to $`J(\mu_P)-J(\mu_Q)`$.
- (e) Exposure without new evidence (replaying one block, or training on deterministic rewrites of it) leaves the one-block difference untouched.

**Intuition.** Only independent evidence shrinks the gap. Averaging independent blocks makes the higher cumulants of the average shrink like $`n^{1-s}`$; repetition or deterministic synthesis adds exposure, not information.

**Role in the framework.** It separates the number of independent blocks from the number of epochs. This counting discipline is used for demonstrations and queries in §2 ([Theorem 2.10](02-10_few_shot_information_floor.md#s2-10)) and in the method of §9 ([Algorithm 9.2](09-02_ced_algorithm.md#s9-2)).

**Scope.**

- Why $`R`$ does not appear in (4.5): the convex weights make the expansion coefficients a probability vector, whatever the number of blocks. Signed or unconstrained weights break this and can amplify discrepancies; data-dependent weights invalidate the factorization.
- Both bounds concern the expected final return of the training procedure, not the score of one trained student.
- $`C_J`$ and $`\delta_N`$ are measured in the monomial basis of the probability coordinates. Large $`N`$ can make (4.5) vacuous, which is why [Proposition 4.6](04-06_exact_order_not_approximate_complexity.md#s4-6) replaces degree by approximation width.
- The $`O(n^{-r})`$ constant depends on $`(P,Q,J)`$; the expansion is not uniform over families of laws whose cumulants grow.

**Derivation outline.**

- (a) In the block-moment expansion of $`\mathbb Ep_R^\alpha`$ the coefficients form a probability vector for every $`R`$, and a product of at most $`\lvert\alpha\rvert`$ block moments in $`[0,1]`$ changes by at most the sum of their changes (product telescoping).
- (b) Taylor-expand the polynomial $`J`$ around the common mean and write the centred moments of $`\bar p_n`$ through cumulants. A partition with $`q`$ blocks of total size $`s`$ scales as $`n^{q-s}`$, and the only differing term of order $`n^{1-r}`$ is the single block of size $`r`$.
- (c) For $`J=p^N`$ no other differing partition exists. (d) Partitions into blocks of size at least 2 are $`O(n^{-1})`$. (e) The average of identical copies is the copy.

**Illustration.** Two sources with the same mean and variance but different skewness ($`r=3`$), and a deployment value whose third derivative at the mean does not annihilate the skewness difference: averaging $`n`$ independent blocks shrinks the gap as $`1/n^2`$. Ten epochs over one block leave the one-block gap untouched.

**Uses.**

- [Definition 4.3](04-03_binary_context_student_affine_learner.md#s4-3) — the model.
- [Theorem 4.4](04-04_affine_learners_moment_order.md#s4-4) — the polynomial form (a) and the witnesses (c).

**Used by.**

- [Theorem 2.10](02-10_few_shot_information_floor.md#s2-10) — counting independent evidence, not exposure.
- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2) — the method's counting discipline.

## Related work

- The joint cumulants and the moment–cumulant formula behind (b) are classical ([Leonov & Shiryaev, 1959](#ref-leonov1959method)). The rate comes from their scaling under averaging: the order-$`s`$ cumulant of the average of $`n`$ IID blocks is $`n^{1-s}\kappa_s`$.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-leonov1959method"></a>

**V. P. Leonov, A. N. Shiryaev (1959).** On a method of calculation of semi-invariants. *Theory of Probability and Its Applications* 4, no. 3, 319–329. DOI [10.1137/1104031](https://doi.org/10.1137/1104031).
