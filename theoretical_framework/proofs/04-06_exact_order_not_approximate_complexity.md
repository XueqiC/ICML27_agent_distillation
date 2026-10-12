# Proposition 4.6 · Exact order is not approximate complexity

*Proof*

[← Framework](../README.md) · [Statement](../statements/04-06_exact_order_not_approximate_complexity.md) · [Proof](04-06_exact_order_not_approximate_complexity.md) · [Experiments](../experiments/04-06_exact_order_not_approximate_complexity.md) · [中文](04-06_exact_order_not_approximate_complexity_ZH.md)

**What is proved here.** The smooth deployment value $`p^N`$ and the parity value $`J_{\rm even}`$ both have exact degree $`N`$, so exact interchangeability requires all moments up to order $`N`$ for both. Approximately they behave completely differently: truncating the Chebyshev expansion of $`p^N`$ after order $`q`$ has error equal to a binomial tail, so about $`\sqrt{N\log(1/\varepsilon)}`$ matched moments suffice for accuracy $`\varepsilon`$; parity cannot be approximated below $`\tfrac12`$ by any polynomial of degree below $`N`$.

**Setting used on this page.**

- Scalar task: one context visited $`N`$ times, success if all actions are 1, so a student with action probability $`p\in[0,1]`$ has deployment value $`J(p)=p^N`$.
- Parity task on $`N`$ contexts, each visited once with independent actions $`a_j\sim\mathrm{Bernoulli}(p_j)`$, paying 1 if $`\sum_ja_j`$ is even: $`J_{\rm even}(p)=\tfrac12\big(1+\prod_{j=1}^N(1-2p_j)\big)`$ ([Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order.md#s4-4)).
- $`T_j`$ is the Chebyshev polynomial of the first kind ([Rivlin, 1990](../references.md#rivlin1990chebyshev)): $`T_j(\cos\theta)=\cos(j\theta)`$, so $`\lvert T_j\rvert\le1`$ on $`[-1,1]`$ and $`T_j(1)=1`$. $`\mathcal P_q`$ is the set of polynomials of total degree at most $`q`$.
- Laws $`P,Q`$ on $`[0,1]`$ (or on blocks $`Z\in[0,1]^k`$) and convex-affine learners $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$ with IID blocks are as in [Definition 4.3](../statements/04-03_binary_context_student_affine_learner.md#s4-3). For a multi-index $`\alpha`$ with $`\lvert\alpha\rvert=a`$ (in the scalar case, $`\alpha=\ell`$), the block-moment expansion from the proof of [Theorem 4.4](04-04_affine_learners_moment_order.md) reads

```math
\mathbb E\,p_R^\alpha=\sum_s\Big(\prod_lw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^Rm(\beta_i(s)), \qquad(4.4)
```

- where the sum is over source maps $`s:\lbrace1,\dots,a\rbrace\to\lbrace0,\dots,R\rbrace`$, $`\beta_i(s)`$ collects the factors assigned to block $`i`$ (so $`\lvert\beta_i(s)\rvert\le a`$), and $`m(\beta)=\mathbb EZ^\beta`$ is a block moment.
- The parity laws: $`P`$ uniform on the even-parity vertices of $`\lbrace0,1\rbrace^N`$, $`Q`$ uniform on the odd-parity vertices. Every proper coordinate subset is uniform under both (proof of Theorem 4.4(c)).

**Statement ([Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity.md#s4-6)).** Let $`T_j`$ be the Chebyshev polynomials of the first kind ($`T_j(\cos\theta)=\cos(j\theta)`$, $`\lvert T_j\rvert\le1`$ on $`[-1,1]`$, $`T_j(1)=1`$) and $`\mathcal P_q`$ the polynomials of total degree at most $`q`$. Consider the scalar task $`J(p)=p^N`$ on $`[0,1]`$ (one context visited $`N`$ times, success if all actions are 1) and the parity task $`J_{\rm even}`$ of Theorem 4.4(c) on $`[0,1]^N`$. For $`p\in[0,1]`$,

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

**Proof.** *Claim 1 (the identity (4.7)).* Put $`2p-1=\cos\theta`$ with $`\theta\in[0,\pi]`$, so $`p=\cos^2(\theta/2)`$ and

```math
p^N=\Big(\frac{e^{i\theta/2}+e^{-i\theta/2}}{2}\Big)^{2N}=4^{-N}\sum_{b=0}^{2N}\binom{2N}{b}e^{i(b-N)\theta}.
```

Grouping $`b=N\pm j`$ for $`j\ge1`$ with $`\binom{2N}{N+j}=\binom{2N}{N-j}`$ gives $`p^N=4^{-N}\binom{2N}N+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}\cos(j\theta)`$, and $`\cos(j\theta)=T_j(2p-1)`$. Both sides are polynomials that agree on $`[0,1]`$, so (4.7) is a polynomial identity.

*(a), first item.* Write $`a_0=4^{-N}\binom{2N}N`$ and $`a_j=2\cdot4^{-N}\binom{2N}{N-j}`$ for $`j\ge1`$. Then $`p^N-h_q(p)=\sum_{j=q+1}^Na_jT_j(2p-1)`$.

- All $`a_j\ge0`$ and $`\lvert T_j\rvert\le1`$, so $`\lvert p^N-h_q(p)\rvert\le\sum_{j>q}a_j`$, with equality at $`p=1`$ where every $`T_j(1)=1`$.
- Pairing $`b=N-j`$ with $`b=N+j`$, $`\sum_{j=q+1}^Na_j=\sum_{b:\lvert b-N\rvert>q}\binom{2N}b4^{-N}=\Pr\lbrace\lvert B-N\rvert>q\rbrace`$.
- $`B`$ is a sum of $`2N`$ independent $`\lbrace0,1\rbrace`$ variables with mean $`N`$, so Hoeffding's inequality ([Hoeffding, 1963](../references.md#hoeffding1963probability)) gives $`\Pr\lbrace\lvert B-N\rvert\ge q\rbrace\le2\exp(-2q^2/(2N))=2e^{-q^2/N}`$.

*(a), second item.* Since $`h_q\in\mathcal P_q`$ and the laws match moments through $`q`$, $`\mathbb E_Ph_q(Z)=\mathbb E_Qh_q(Z)`$, so $`\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert=\lvert(P-Q)(p^N-h_q)\rvert\le\lVert P-Q\rVert_1\lVert p^N-h_q\rVert_\infty\le4e^{-q^2/N}`$.

*(a), third item.* For $`\ell\le q`$, (4.4) writes $`\mathbb Ep_R^\ell`$ as a fixed combination of products of block moments of order at most $`\ell`$. So the laws of $`p_R\in[0,1]`$ under $`P^R`$ and under $`Q^R`$ match moments through $`q`$, and the second item applies to them.

*Order needed.* $`4e^{-q^2/N}\le\varepsilon`$ if and only if $`q\ge\sqrt{N\log(4/\varepsilon)}`$; for $`q\ge N`$ the gap is exactly 0 by [Theorem 4.4(b)](../statements/04-04_affine_learners_moment_order.md#s4-4) (moments matched through the deployment length give equal learning values).

*(b).* *Upper bound.* $`J_{\rm even}`$ is a probability, so $`\lVert J_{\rm even}-\tfrac12\rVert_\infty\le\tfrac12`$.

*Lower bound.* Let $`h\in\mathcal P_q`$ with $`q<N`$, and let $`P,Q`$ be uniform on the even- and odd-parity vertices of $`\lbrace0,1\rbrace^N`$.

- On binary vertices $`p_j^a=p_j`$ for $`a\ge1`$, so $`h`$ restricted to $`\lbrace0,1\rbrace^N`$ is a multilinear polynomial whose monomials $`\prod_{j\in S}p_j`$ have $`\lvert S\rvert\le q<N`$.
- Every proper coordinate subset is uniform under both laws, so $`\mathbb E_Ph=\mathbb E_Qh`$.
- On the vertices $`J_{\rm even}(Z)=\mathbf 1\lbrace Z\ \text{even}\rbrace`$, so $`\mathbb E_PJ_{\rm even}=1`$ and $`\mathbb E_QJ_{\rm even}=0`$.

Therefore

```math
1=(P-Q)J_{\rm even}=(P-Q)(J_{\rm even}-h)\le\lVert P-Q\rVert_1\,\lVert J_{\rm even}-h\rVert_{\infty,\lbrace0,1\rbrace^N}\le2\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N},
```

so $`\lVert J_{\rm even}-h\rVert_\infty\ge\tfrac12`$. Equivalently, by [Theorem 4.2(b)](../statements/04-02_sufficiency_duality_transfer.md#s4-2) (the worst matched-pair discrepancy is twice the uniform approximation error), the matched pair $`(P,Q)`$ certifies $`\mathrm{dist}_\infty(J_{\rm even},\mathcal P_q)\ge\tfrac12`$. $`\square`$

## Remarks

- **The contrast.** $`p^N`$ and $`J_{\rm even}`$ both have exact degree $`N`$, and exact sufficiency needs all moments through $`N`$ for both. Their approximate requirements differ completely:
  - $`p^N`$ is resolved to accuracy $`\varepsilon`$ by order $`O(\sqrt{N\log(1/\varepsilon)})`$;
  - $`J_{\rm even}`$ is not resolved below $`\tfrac12`$ by any order below $`N`$.
- The operative complexity is therefore the approximate dimension $`d_\varepsilon`$ of the response class (Theorem 4.2(b)), not its degree.
- **Scope.** (a) is an upper bound on the sufficient order for this scalar task and the convex-affine learners; no matching lower bound on the approximate order is claimed.
- **Stability.**
  - Exact low-order matching is not a finite-sample certificate. If moments are only estimated, the power-basis coefficients of $`h_q`$ multiply the estimation errors, and they can be large.
  - Bounded Chebyshev features avoid power-basis amplification for one scalar message, but pulled back through several affine updates they need their own coefficient bound.
  - In general the relevant pair is $`(B,\tau)`$ of [Theorem 4.2(c)](../statements/04-02_sufficiency_duality_transfer.md#s4-2), and [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1) (the exact stability–approximation tradeoff under measurement tolerances) makes the tradeoff exact.
- **Realizability.** Both witness families are realizable finite feedback protocols: a reset experiment can emit a block drawn from the specified law. This shows that the environments exist, not that a given benchmark permits them.

## Related work

- Chebyshev expansions ([Rivlin, 1990](../references.md#rivlin1990chebyshev)), Hoeffding's inequality ([Hoeffding, 1963](../references.md#hoeffding1963probability)) and the independence of parity from proper coordinate subsets are classical tools; they are credited as such and not claimed as new.
- What is added is their use to separate the exact information order of a response class from its approximate dimension.

## References

- [Hoeffding, 1963](../references.md#hoeffding1963probability) — Hoeffding's inequality.
- [Rivlin, 1990](../references.md#rivlin1990chebyshev) — Chebyshev polynomials.
