# Theorem 4.10 · Hankel rank and linear predictive realization

*Proof*

[← Framework](../README.md) · [Statement](../statements/04-10_hankel_rank_predictive_realization.md) · [Proof](04-10_hankel_rank_predictive_realization.md) · [Experiments](../experiments/04-10_hankel_rank_predictive_realization.md) · [中文](04-10_hankel_rank_predictive_realization_ZH.md)

**What is proved here.** (a) is linear algebra on an invariant function space. (b) shows that the rank of the task Hankel matrix bounds every linear realization from below and is attained by the row-space realization, whose transition maps are induced by the shift of test columns. (c) is Bayes' rule written in predictive coordinates.

**Setting used on this page.**

- Controls $`b\in\mathcal B`$ with Markov kernels $`K_b`$ on bounded functions of the complete learning state; words $`w=b_1\cdots b_\ell`$, $`K_w=K_{b_1}\cdots K_{b_\ell}`$; bounded outputs $`F_1,\dots,F_k`$ ([Definition 1.3](../README.md#s1-3)).
- The test space $`\mathcal V=\mathrm{span}\lbrace\mathbf 1,K_wF_j:w\in\mathcal W\rbrace`$; for (b), prefixes $`\mathcal P`$ (closed under extension) and test columns $`\mathcal T`$ (closed under prefixing, containing the constant test and every $`(\varnothing,j)`$), and $`H(p,(w,j))=\mu_0K_pK_wF_j`$.
- Observation operators $`T_{b,o}f(z)=\mathbb E[\mathbf 1\lbrace O=o\rbrace f(Z')\mid Z=z,b]`$.

**Statement ([Theorem 4.10](../statements/04-10_hankel_rank_predictive_realization.md#s4-10)).** (a) If $`\mathcal V`$ has a finite basis $`\psi`$ with $`\psi_1=\mathbf 1`$, contains the $`F_j`$ and is $`K_b`$-invariant, then $`K_b\psi=A_b\psi`$, $`F_j=c_j^\top\psi`$ and (4.18) holds; invariance holds when $`\mathcal W`$ contains the empty word and is closed under one-step prefixing. (b) If $`\mathrm{rank}\,H=d<\infty`$, every linear realization has dimension at least $`d`$, and (4.19) gives one of dimension $`d`$. (c) If $`T_{b,o}\psi=M_{b,o}\psi`$, then $`\Pr(o\mid\mu,b)=e_1^\top M_{b,o}q`$ and the posterior predictive state is (4.20).

**Proof.**

**(a)** For each $`b`$ and $`i`$, $`K_b\psi_i\in\mathcal V`$, so $`K_b\psi_i=\sum_l(A_b)_{il}\psi_l`$ for unique coefficients; these form $`A_b`$, and in vector form $`K_b\psi=A_b\psi`$. Likewise $`F_j\in\mathcal V`$ gives $`c_j`$. For a state law $`\mu`$, integration is linear, so

```math
q(\mu K_b)=\int K_b\psi\,d\mu=A_b\int\psi\,d\mu=A_b\,q(\mu),\qquad\mathbb E_\mu F_j=c_j^\top\int\psi\,d\mu=c_j^\top q(\mu).
```

*Invariance from prefix closure.* $`K_b\mathbf 1=\mathbf 1`$ because $`K_b`$ is a Markov kernel, and $`K_b(K_wF_j)=K_{bw}F_j\in\mathcal V`$ when $`bw\in\mathcal W`$. The generators of $`\mathcal V`$ are mapped into $`\mathcal V`$, so by linearity $`K_b\mathcal V\subseteq\mathcal V`$. The empty word puts every $`F_j`$ in $`\mathcal V`$.

**(b)** *Lower bound.* Let $`q(p)\in\mathbb R^{d'}`$ and $`c_{w,j}`$ realize every column of $`H`$ indexed by $`\mathcal T`$, the constant column included. For each $`i\le d'`$ let $`x_i\in\mathbb R^{\mathcal T}`$ be the vector $`(c_{w,j,i})_{(w,j)\in\mathcal T}`$. Every row is $`H(p,\cdot)=\sum_{i\le d'}q_i(p)x_i`$, so the row space has dimension at most $`d'`$. The rank of $`H`$ is the dimension of its row space, so $`d\le d'`$.

*Shift.* Define $`S_b:\mathbb R^{\mathcal T}\to\mathbb R^{\mathcal T}`$ by $`(S_bx)(w,j)=x(bw,j)`$ and $`(S_bx)(\mathbf 1)=x(\mathbf 1)`$ on the constant column. It is well defined because $`\mathcal T`$ is closed under one-step prefixing, and it is linear. For a row,

```math
(S_bH(p,\cdot))(w,j)=H(p,(bw,j))=\mu_0K_pK_bK_wF_j=H(pb,(w,j)),
```

and on the constant column both sides equal 1. So $`S_bH(p,\cdot)=H(pb,\cdot)`$, which is again a row because $`\mathcal P`$ is closed under extension. Hence $`S_b`$ maps the set of rows into itself, and by linearity it maps the row space $`\mathcal L`$ into itself.

*Realization.* Choose a basis $`x_1,\dots,x_d`$ of $`\mathcal L`$ and let $`q(p)\in\mathbb R^d`$ be the unique coordinates with $`H(p,\cdot)=\sum_iq_i(p)x_i`$. Let $`A_b`$ be the matrix of $`S_b|_{\mathcal L}`$ in this basis: $`S_bx_i=\sum_l(A_b)_{li}x_l`$. Then

```math
H(pb,\cdot)=S_bH(p,\cdot)=\sum_iq_i(p)S_bx_i=\sum_l\big(A_bq(p)\big)_lx_l,
```

so $`q(pb)=A_bq(p)`$ by uniqueness of coordinates. Let $`c_j=(x_1(\varnothing,j),\dots,x_d(\varnothing,j))^\top`$, so that $`H(p,(\varnothing,j))=c_j^\top q(p)`$. For a word $`w=b_1\cdots b_\ell`$, $`pw\in\mathcal P`$ and $`H(p,(w,j))=\mu_0K_{pw}F_j=H(pw,(\varnothing,j))=c_j^\top q(pw)=c_j^\top A_{b_\ell}\cdots A_{b_1}q(p)`$, which is (4.19).

**(c)** Let $`\mu`$ be the current state law. Since $`\psi_1=\mathbf 1`$, the first row of $`T_{b,o}\psi=M_{b,o}\psi`$ reads $`T_{b,o}\mathbf 1=e_1^\top M_{b,o}\psi`$, so

```math
\Pr(O=o\mid\mu,b)=\int T_{b,o}\mathbf 1\,d\mu=e_1^\top M_{b,o}\,q .
```

When this is positive, the posterior law $`\mu'`$ of the next state given $`O=o`$ satisfies $`\int f\,d\mu'=\int T_{b,o}f\,d\mu/\Pr(O=o\mid\mu,b)`$ for bounded $`f`$, by the definition of conditional expectation. Taking $`f=\psi`$ coordinatewise,

```math
q'=\int\psi\,d\mu'=\frac{\int T_{b,o}\psi\,d\mu}{\Pr(O=o\mid\mu,b)}=\frac{M_{b,o}\,q}{e_1^\top M_{b,o}\,q}. \qquad\square
```

**Remarks on the proof.**

- (b) uses only the algebraic rank of the full matrix over the declared prefixes and columns. With a finite observed submatrix the same argument gives a realization of that submatrix only if the observed rows and columns are themselves closed under the shifts; otherwise the observed rank is only a lower bound on $`d`$.
- Time-dependent kernels are covered by putting the clock into the state, which makes the kernels time-homogeneous on the enlarged state; absorption at the budget keeps every word within the declared training budget.

## Related work

- The row-space construction in (b) is the standard realization argument for predictive state representations and observable operator models ([Singh, James & Rudary, 2004](#ref-singh2004predictive)); (c) is the normalized linear update of [Littman, Sutton & Singh (2001)](#ref-littman2001predictive).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-littman2001predictive"></a>

**Michael L. Littman, Richard S. Sutton, Satinder Singh (2001).** Predictive representations of state. *Advances in Neural Information Processing Systems 14 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).

<a id="ref-singh2004predictive"></a>

**Satinder Singh, Michael R. James, Matthew R. Rudary (2004).** Predictive state representations: A new theory for modeling dynamical systems. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI)*, 512–519, AUAI Press. arXiv [1207.4167](https://arxiv.org/abs/1207.4167).
