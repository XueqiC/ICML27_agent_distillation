# N09 · Exact order is not approximate complexity — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N09_approximate_complexity.md) · [中文](N09_approximate_complexity_ZH.md)

## Setting and assumptions

- **Scalar task.** A single context visited $N$ times, with success iff all actions are 1, has value $J(p)=p^N$ on $[0,1]$ (N07, Theorem 12).
- **Parity task.** With $k=N$ contexts, each visited once, and success iff the action vector has even parity, the value is $J_{\rm even}(p)=\tfrac12(1+\prod_{j=1}^N(1-2p_j))$ on $[0,1]^N$, as in (27).
- **Chebyshev polynomials.** $T_j$ is the Chebyshev polynomial of the first kind, so $T_j(\cos\theta)=\cos(j\theta)$, and $\lvert T_j\rvert\le1$ on $[-1,1]$ with $T_j(1)=1$.
- **Affine learners.** Learners are the convex affine averages (23) of N07, with IID scalar blocks $Z_i\in[0,1]$.
- **Approximation spaces.** $\mathcal P_q$ denotes polynomials of total degree at most $q$.
- **Matched moments.** "$P,Q$ match moments through $q$" means $\mathbb E_PZ^\ell=\mathbb E_QZ^\ell$ for $\ell\le q$ (scalar case), or all joint moments of total degree $\le q$ (vector case).

## Statement

**Lemma (Chebyshev expansion of $p^N$, eq. (32)).** For $p\in[0,1]$,
$$p^N=4^{-N}\binom{2N}{N}+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}T_j(2p-1). \tag{32}$$

**Proposition 16.** Let $q<N$, and let $h_q$ be the truncation of (32) after order $q$, so $h_q\in\mathcal P_q$. Let $B\sim\mathrm{Bin}(2N,\tfrac12)$. Then:

1. $\displaystyle\lVert p^N-h_q\rVert_{\infty,[0,1]}=\Pr\lbrace \lvert B-N\rvert>q\rbrace \le2e^{-q^2/N}$. $\qquad$ (33)
2. If two scalar laws $P,Q$ on $[0,1]$ match moments through $q$, then $\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert\le4e^{-q^2/N}$.
3. The same bound holds for the learning values $\mathbb E\,p_R^N$ after any fixed convex affine learner (23).

Consequently, matching moments through
$$q\ \ge\ \min\Big\lbrace N,\ \Big\lceil\sqrt{N\log(4/\varepsilon)}\Big\rceil\Big\rbrace $$
guarantees a gap of at most $\varepsilon$, although exact equality for all laws requires order $N$ (N07, Theorem 12).

**Proposition (parity is incompressible below degree $N$).** For every $q<N$,
$$\min_{h\in\mathcal P_q}\ \lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N}=\tfrac12,$$
and the constant $h\equiv\tfrac12$ attains the minimum.

## Proof

### Proof of the Lemma

Put $2p-1=\cos\theta$ with $\theta\in[0,\pi]$. Then $p=\tfrac{1+\cos\theta}{2}=\cos^2(\theta/2)$, so
$$p^N=\cos^{2N}(\theta/2)=\Big(\frac{e^{i\theta/2}+e^{-i\theta/2}}2\Big)^{2N}=4^{-N}\sum_{b=0}^{2N}\binom{2N}{b}e^{i(b-N)\theta}.$$
Group $b=N\pm j$ for $j\ge1$ using $\binom{2N}{N+j}=\binom{2N}{N-j}$:
$$p^N=4^{-N}\binom{2N}{N}+4^{-N}\sum_{j=1}^N\binom{2N}{N-j}\big(e^{ij\theta}+e^{-ij\theta}\big)=4^{-N}\binom{2N}{N}+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}\cos(j\theta).$$
Since $\cos(j\theta)=T_j(\cos\theta)=T_j(2p-1)$, (32) holds for every $p\in[0,1]$. Both sides are polynomials, so it is a polynomial identity. $\square$

### Proof of Proposition 16

*(1), equality.* Write $a_j$ for the coefficient of $T_j(2p-1)$ in (32), so $a_0=4^{-N}\binom{2N}N$ and $a_j=2\cdot4^{-N}\binom{2N}{N-j}$ for $j\ge1$. Then
$$p^N-h_q(p)=\sum_{j=q+1}^Na_jT_j(2p-1).$$
All $a_j\ge0$ and $\lvert T_j\rvert\le1$, so $\lvert p^N-h_q(p)\rvert\le\sum_{j>q}a_j$. At $p=1$ every $T_j(1)=1$, so the bound is attained and
$$\lVert p^N-h_q\rVert_\infty=\sum_{j=q+1}^Na_j=\sum_{j=q+1}^N2\cdot4^{-N}\binom{2N}{N-j}=\sum_{b:\lvert b-N\rvert>q}\binom{2N}{b}4^{-N}=\Pr\lbrace \lvert B-N\rvert>q\rbrace .$$
The third equality pairs $b=N-j$ with $b=N+j$.

*(1), inequality.* $B$ is a sum of $2N$ independent $\lbrace 0,1\rbrace $ variables with mean $N$. Hoeffding's inequality gives
$$\Pr\lbrace \lvert B-N\rvert\ge t\rbrace \le2\exp(-2t^2/(2N)).$$
Taking $t=q$ yields $\Pr\lbrace \lvert B-N\rvert>q\rbrace \le2e^{-q^2/N}$.

*(2).* Since $h_q\in\mathcal P_q$ and the laws match moments through $q$, $\mathbb E_Ph_q(Z)=\mathbb E_Qh_q(Z)$. Hence
$$\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert=\lvert(P-Q)(p^N-h_q)\rvert\le\lVert P-Q\rVert_1\,\lVert p^N-h_q\rVert_\infty\le2\cdot2e^{-q^2/N}.$$

*(3).* For $\ell\le q$, the expansion (B2) of N07 writes $\mathbb E\,p_R^\ell$ as a fixed combination of products of block moments of order $\le\ell\le q$. So the laws of $p_R$ under $P^R$ and under $Q^R$ match moments through $q$, and $p_R\in[0,1]$. Apply (2) to the laws of $p_R$.

*Order needed.* $4e^{-q^2/N}\le\varepsilon$ holds if and only if $q\ge\sqrt{N\log(4/\varepsilon)}$. For $q\ge N$ the gap is exactly 0, by N07, Theorem 11. $\square$

### Proof of the parity proposition

*Upper bound.* $J_{\rm even}$ is the probability of an event, so $J_{\rm even}\in[0,1]$ and $\lVert J_{\rm even}-\tfrac12\rVert_\infty\le\tfrac12$.

*Lower bound.* Let $h\in\mathcal P_q$ with $q<N$. Let $P$ and $Q$ be uniform on the even- and odd-parity vertices of $\lbrace 0,1\rbrace ^N$.

1. On binary vertices, $p_j^a=p_j$ for $a\ge1$. So $h$ restricted to $\lbrace 0,1\rbrace ^N$ equals a multilinear polynomial whose monomials $\prod_{j\in S}p_j$ have $\lvert S\rvert\le q<N$.
2. By N07, Theorem 12(ii), every proper coordinate subset is uniform under both laws. Hence $\mathbb E_Ph=\mathbb E_Qh$.
3. On the vertices, $J_{\rm even}(Z)=\mathbf 1\lbrace Z\text{ even}\rbrace $, so $\mathbb E_PJ_{\rm even}=1$ and $\mathbb E_QJ_{\rm even}=0$.

Therefore
$$1=(P-Q)J_{\rm even}=(P-Q)(J_{\rm even}-h)\le\lVert P-Q\rVert_1\,\lVert J_{\rm even}-h\rVert_{\infty,\lbrace 0,1\rbrace ^N}\le2\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N}.$$
So $\lVert J_{\rm even}-h\rVert_\infty\ge\tfrac12$. Equivalently, by N06(b), the matched pair $(P,Q)$ certifies $\operatorname{dist}_\infty(J_{\rm even},\mathcal P_q)\ge\tfrac12$. $\square$

## Remarks

1. **The contrast.** Both $p^N$ and $J_{\rm even}$ have exact degree $N$, and exact sufficiency needs all moments through $N$ for both (N07). Their approximate requirements differ completely:
   - $p^N$ is resolved to accuracy $\varepsilon$ by order $O(\sqrt{N\log(1/\varepsilon)})$;
   - $J_{\rm even}$ is not resolved below 1/2 by any order $<N$.

   The operative complexity is therefore the approximation width $d_\varepsilon$ of the response class (N06(b)), not its degree.
2. **Scope.** Proposition 16 concerns this scalar task and the affine learners. It is an *upper* bound on the sufficient order. No matching lower bound on the approximate order is claimed.
3. **Stability.**
   - Exact low-order matching is not a finite-sample certificate. If the moments are only estimated, the power-basis coefficients of $h_q$ multiply the estimation errors, and these coefficients can be large.
   - Bounded Chebyshev features avoid power-basis amplification for a single scalar message. Pulled back through several affine updates, however, they need their own coefficient bound.
   - In general the relevant pair is $(B,\tau)$ of N06(c), and the tradeoff is made exact by N13 (Thm. 18).
4. **Realizability.** Both witness families are realizable finite feedback protocols: a reset experiment emits a block drawn from the specified law (§14.1 of the source). This shows only that the constructed environments exist. It does not show that a given benchmark permits them.

## Dependencies and source

- **Uses:** N06 (approximation width (20)) and N07 (Theorems 11–12 and the block expansion).
- **Used by:** N13 (stability–approximation tradeoff).
- **Source:** v0.14.1 main framework, §8, eq. (32), Proposition 16 and the parity paragraph, `docs/direction/v015_revision/Environment_Information_Theory_EN.md`. Independent review: `docs/direction/v014_handoff/Independent_Proof_Audit_EN.md`, B2 and B3. The Chebyshev expansion, Hoeffding's inequality and parity independence are classical tools.
