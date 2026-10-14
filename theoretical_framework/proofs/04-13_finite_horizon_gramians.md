# Proposition 4.13 · Finite-horizon Gramians, and what control theory does not transfer

*Proof*

[← Framework](../README.md) · [Statement](../statements/04-13_finite_horizon_gramians.md) · [Proof](04-13_finite_horizon_gramians.md) · [Experiments](../experiments/04-13_finite_horizon_gramians.md) · [中文](04-13_finite_horizon_gramians_ZH.md)

**What is proved here.** (a) moves the factors of $`H^\top H`$ around using the fact that $`XY`$ and $`YX`$ have the same nonzero eigenvalues. (b) is the minimum-norm solution of a linear system and a Rayleigh quotient. No property of the $`A_t`$ beyond finiteness of the horizon is used.

**Setting used on this page.**

- The finite-horizon linearization with reachability matrix $`\mathscr C_\tau`$, observability matrix $`\mathscr O_\tau`$, Hankel operator $`H_\tau=\mathscr O_\tau\mathscr C_\tau`$, and Gramians $`W_c=\mathscr C_\tau\mathscr C_\tau^\top`$, $`W_o=\mathscr O_\tau^\top\mathscr O_\tau`$, all with Euclidean norms after weighting ([Proposition 4.12](../statements/04-12_response_predictive_state_factorization.md#s4-12) gives the one-step case). The subscript $`\tau`$ is dropped below.

**Statement ([Proposition 4.13](../statements/04-13_finite_horizon_gramians.md#s4-13)).** (a) $`\sigma_j(H)^2=\lambda_j(W_c^{1/2}W_oW_c^{1/2})`$ for every $`j`$. (b) For $`x\in\mathrm{range}\,W_c`$ the least energy reaching $`x`$ is $`x^\top W_c^+x`$, every input reaching $`x`$ gives output energy $`x^\top W_ox`$, $`\sigma_1(H)^2=\sup_{x}x^\top W_ox/x^\top W_c^+x`$ over nonzero reachable $`x`$, and $`H=0`$ exactly when $`W_oW_c=0`$.

**Proof.**

**(a)** The squared singular values of $`H`$ are the eigenvalues of $`H^\top H=\mathscr C^\top\mathscr O^\top\mathscr O\mathscr C=\mathscr C^\top W_o\mathscr C`$. For matrices $`X\in\mathbb R^{a\times b}`$, $`Y\in\mathbb R^{b\times a}`$, $`XY`$ and $`YX`$ have the same nonzero eigenvalues with the same algebraic multiplicities. With $`X=\mathscr C^\top`$ and $`Y=W_o\mathscr C`$, the nonzero eigenvalues of $`\mathscr C^\top W_o\mathscr C`$ are those of $`W_o\mathscr C\mathscr C^\top=W_oW_c`$. With $`X=W_c^{1/2}`$ and $`Y=W_oW_c^{1/2}`$, those are the nonzero eigenvalues of $`W_c^{1/2}W_oW_c^{1/2}`$, which is symmetric positive semidefinite. Both $`H^\top H`$ and $`W_c^{1/2}W_oW_c^{1/2}`$ have only nonnegative eigenvalues, so after sorting and padding with zeros the two sequences coincide.

**(b)** *Least energy.* For $`x\in\mathrm{range}\,W_c=\mathrm{range}\,\mathscr C`$, the solutions of $`\mathscr Cu=x`$ form an affine space whose element of least norm is $`u^\star=\mathscr C^+x`$. Its squared norm is

```math
\lVert u^\star\rVert^2=x^\top(\mathscr C^+)^\top\mathscr C^+x=x^\top(\mathscr C\mathscr C^\top)^+x=x^\top W_c^+x ,
```

using the identity $`(\mathscr C^+)^\top\mathscr C^+=(\mathscr C\mathscr C^\top)^+`$ of the Moore–Penrose inverse.

*Output energy.* If $`\mathscr Cu=x`$ then $`Hu=\mathscr Ox`$ and $`\lVert Hu\rVert^2=x^\top W_ox`$, whatever $`u`$ is.

*Rayleigh quotient.* Decompose any input as $`u=u_0+u_1`$ with $`u_0\in\ker\mathscr C`$ and $`u_1\perp\ker\mathscr C`$; then $`Hu=Hu_1`$, $`\lVert u\rVert\ge\lVert u_1\rVert`$, and $`u_1=\mathscr C^+x`$ for $`x=\mathscr Cu`$. Therefore

```math
\sigma_1(H)^2=\sup_{u\ne0}\frac{\lVert Hu\rVert^2}{\lVert u\rVert^2}=\sup_{x\in\mathrm{range}\,W_c,\ x\ne0}\frac{x^\top W_o\,x}{x^\top W_c^+\,x},
```

where inputs with $`\mathscr Cu=0`$ contribute zero and are omitted. (As a consistency check, substituting $`x=W_c^{1/2}y`$ with $`y\in\mathrm{range}\,W_c`$ turns the quotient into $`y^\top W_c^{1/2}W_oW_c^{1/2}y/\lVert y\rVert^2`$, whose supremum is the largest eigenvalue in (a).)

*Zero gain.* $`H=\mathscr O\mathscr C=0`$ if and only if $`\mathrm{range}\,\mathscr C\subseteq\ker\mathscr O`$. Since $`\mathrm{range}\,W_c=\mathrm{range}\,\mathscr C`$ and $`\ker W_o=\ker\mathscr O`$, this is $`\mathrm{range}\,W_c\subseteq\ker W_o`$, i.e. $`W_oW_c=0`$. $`\square`$

**Remark on the proof.** Nothing in (a) or (b) uses $`\lVert A_t\rVert\le1`$, time invariance or an infinite horizon. Those are needed only for the classical stability theory and the a-priori error bounds of balanced truncation, which the framework does not use.

## Related work

- The identity of (a) is the finite-horizon, time-varying form of the relation between Hankel singular values and the product of Gramians used in balanced realization ([Moore, 1981](#ref-moore1981principal)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-moore1981principal"></a>

**Bruce C. Moore (1981).** Principal component analysis in linear systems: Controllability, observability, and model reduction. *IEEE Transactions on Automatic Control* 26, no. 1, 17–32. DOI [10.1109/TAC.1981.1102568](https://doi.org/10.1109/TAC.1981.1102568).
