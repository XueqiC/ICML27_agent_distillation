# Proposition 4.6 · Exact order is not approximate complexity

*Statement*

[← Framework](../README.md) · [Statement](04-06_exact_order_not_approximate_complexity.md) · [Proof](../proofs/04-06_exact_order_not_approximate_complexity.md) · [Experiments](../experiments/04-06_exact_order_not_approximate_complexity.md) · [中文](04-06_exact_order_not_approximate_complexity_ZH.md)

**In one paragraph.** [Theorem 4.4](04-04_affine_learners_moment_order.md#s4-4) shows that exact interchangeability for deployments with $`N`$ decisions requires every moment up to order $`N`$. But exact equality is a very strong demand. If a small outcome difference $`\varepsilon`$ is acceptable, how many moments are needed? This proposition shows that the answer depends on the task, not on its degree: for the smooth value $`p^N`$ about $`\sqrt{N\log(1/\varepsilon)}`$ moments suffice, while for a parity value no order below $`N`$ helps at all. The operative complexity is therefore the approximate dimension of the response class.

**Setting used on this page.**

- Scalar task: one context visited $`N`$ times, success if all actions are 1, so the deployment value of a student with action probability $`p\in[0,1]`$ is $`J(p)=p^N`$.
- Parity task on $`N`$ contexts, each visited once with independent actions $`a_j\sim\mathrm{Bernoulli}(p_j)`$, paying 1 if $`\sum_ja_j`$ is even: $`J_{\rm even}(p)=\tfrac12\big(1+\prod_{j=1}^N(1-2p_j)\big)`$ on $`[0,1]^N`$ ([Theorem 4.4(c)](04-04_affine_learners_moment_order.md#s4-4)).
- Convex-affine learners with IID blocks are as in [Definition 4.3](04-03_binary_context_student_affine_learner.md#s4-3): $`p_R=w_0p_0+\sum_iw_iZ_i`$ with fixed convex weights.
- $`T_j`$ is the Chebyshev polynomial of the first kind ([Rivlin, 1990](../references.md#rivlin1990chebyshev)): $`T_j(\cos\theta)=\cos(j\theta)`$, so $`\lvert T_j\rvert\le1`$ on $`[-1,1]`$ and $`T_j(1)=1`$. $`\mathcal P_q`$ is the set of polynomials of total degree at most $`q`$, and $`\lVert\cdot\rVert_{\infty,S}`$ is the sup norm on a set $`S`$.
- $`B\sim\mathrm{Bin}(2N,\tfrac12)`$ is a binomial random variable.

<a id="s4-6"></a>

**Proposition 4.6 (Exact order is not approximate complexity).** Let $`T_j`$ be the Chebyshev polynomials of the first kind ($`T_j(\cos\theta)=\cos(j\theta)`$, $`\lvert T_j\rvert\le1`$ on $`[-1,1]`$, $`T_j(1)=1`$) and $`\mathcal P_q`$ the polynomials of total degree at most $`q`$. Consider the scalar task $`J(p)=p^N`$ on $`[0,1]`$ (one context visited $`N`$ times, success if all actions are 1) and the parity task $`J_{\rm even}`$ of Theorem 4.4(c) on $`[0,1]^N`$. For $`p\in[0,1]`$,

```math
p^N=4^{-N}\binom{2N}{N}+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}T_j(2p-1). \qquad(4.7)
```

- **(a) Smooth values have low approximate order.** Let $`q<N`$, let $`h_q\in\mathcal P_q`$ be the truncation of (4.7) after order $`q`$, and let $`B\sim\mathrm{Bin}(2N,\tfrac12)`$. Then:
  - $`\lVert p^N-h_q\rVert_{\infty,[0,1]}=\Pr\lbrace\lvert B-N\rvert>q\rbrace\le2e^{-q^2/N}`$;
  - if two laws $`P,Q`$ on $`[0,1]`$ have equal moments of orders $`\ell\le q`$, then $`\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert\le4e^{-q^2/N}`$;
  - the same bound holds for the learning values $`\mathbb E\,p_R^N`$ after any convex-affine learner with IID scalar blocks.
- Consequently, matching moments through order $`q\ge\min\lbrace N,\lceil\sqrt{N\log(4/\varepsilon)}\rceil\rbrace`$ guarantees a gap of at most $`\varepsilon`$, although exact equality for all laws requires order $`N`$ (Theorem 4.4(c)).
- **(b) Parity is incompressible below degree $`N`$.** For every $`q<N`$,

```math
\min_{h\in\mathcal P_q}\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N}=\tfrac12 ,
```

- and the constant $`h\equiv\tfrac12`$ attains the minimum.

**What it says.**

- (a) For the smooth value $`p^N`$, truncating its Chebyshev expansion after order $`q`$ has error exactly a binomial tail, which falls like $`e^{-q^2/N}`$. So about $`\sqrt{N\log(4/\varepsilon)}`$ matched moments already guarantee accuracy $`\varepsilon`$, for laws and for averaging learners alike.
- (b) For parity, no polynomial of degree below $`N`$ approximates better than the trivial constant $`\tfrac12`$, so no order below $`N`$ resolves it to better than $`\tfrac12`$.

**Intuition.** Smooth responses have good low-order approximations; combinatorial responses do not. Both $`p^N`$ and $`J_{\rm even}`$ have exact degree $`N`$, yet their approximate requirements differ completely.

**Role in the framework.** It shows that the operative complexity is the approximate dimension $`d_\varepsilon`$ of the response class ([Theorem 4.2(b)](04-02_sufficiency_duality_transfer.md#s4-2)), not its degree. Approximate matching raises the question of coefficient stability, answered exactly in [Theorem 8.1](08-01_stability_approximation_tradeoff.md#s8-1).

**Scope.**

- (a) is an upper bound on the sufficient order for this scalar task and the convex-affine learners; no matching lower bound on the approximate order is claimed.
- Exact low-order matching is not a finite-sample certificate: if moments are only estimated, the power-basis coefficients of $`h_q`$ multiply the estimation errors, and they can be large. Bounded Chebyshev features avoid this for one scalar message, but pulled back through several affine updates they need their own coefficient bound; the general tradeoff is the pair $`(B,\tau)`$ of Theorem 4.2(c), made exact by Theorem 8.1.
- Both witness families are realizable finite feedback protocols (a reset experiment can emit a block drawn from the specified law). This shows that such environments exist, not that a given benchmark permits them.

**Derivation outline.**

- (a) With $`2p-1=\cos\theta`$, $`p^N=\cos^{2N}(\theta/2)`$ expands into Chebyshev polynomials with binomial coefficients. The tail of these nonnegative coefficients is the sup-error (attained at $`p=1`$) and equals a binomial tail, bounded by Hoeffding's inequality ([Hoeffding, 1963](../references.md#hoeffding1963probability)). Matched moments cancel the truncation, leaving at most twice the sup-error.
- (b) The even- and odd-parity laws agree on every polynomial of degree below $`N`$ but differ by 1 on $`J_{\rm even}`$, which by Theorem 4.2(b) forces approximation error at least $`\tfrac12`$.

**Illustration.** For $`N=100`$ and $`\varepsilon=0.01`$, about $`\sqrt{100\log400}\approx25`$ matched moments already resolve $`p^{100}`$, whereas parity over 100 contexts needs all 100 orders.

**Uses.**

- [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2) — discrepancy equals twice the approximation error.
- [Theorem 4.4](04-04_affine_learners_moment_order.md#s4-4) — exact order and the parity witness.
- [Definition 4.3](04-03_binary_context_student_affine_learner.md#s4-3) — convex-affine learners.

**Used by.**

- [Theorem 8.1](08-01_stability_approximation_tradeoff.md#s8-1) — coefficient stability of approximate interfaces.

## References

- [Hoeffding, 1963](../references.md#hoeffding1963probability) — Hoeffding's inequality for bounded sums.
- [Rivlin, 1990](../references.md#rivlin1990chebyshev) — Chebyshev polynomials.
