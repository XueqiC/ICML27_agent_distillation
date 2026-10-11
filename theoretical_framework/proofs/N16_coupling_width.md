# N16 · Coupling width — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N16_coupling_width.md) · [中文](N16_coupling_width_ZH.md)

## Setting and assumptions

- $`\mathcal Y_1=\lbrace 1,\dots,m\rbrace`$ and $`\mathcal Y_2=\lbrace 1,\dots,n\rbrace`$ are finite sets of valid feedback units, for example the units filling two slots of a training packet.
- $`\mu\in\Delta(\mathcal Y_1)`$ and $`\nu\in\Delta(\mathcal Y_2)`$ are the slot marginals, assumed strictly positive. (With zero entries, everything below holds on the positive-support product.)
- Every pair $`(i,j)`$ is a valid packet.
- The **coupling polytope** is

  ```math
  \mathcal C(\mu,\nu)=\lbrace \Gamma\in\mathbb R^{m\times n}_{\ge0}:\ \Gamma\mathbf 1=\mu,\ \Gamma^\top\mathbf 1=\nu\rbrace .
  ```

- At a learner state $`z`$, $`H_z(i,j)`$ is the expected real terminal return after the **actual update on the joint packet** $`(i,j)`$ followed by the reference suffix. It is a joint-packet response in the sense of N05.
- The **coupling width** is

  ```math
  w(H_z)=\max_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle-\min_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle.
  ```

  It is the largest change in expected learning value obtainable by reorganizing which units are packed together while keeping both slot marginals, and hence the per-unit content, fixed.

## Statement

**Proposition (N16; (a)–(b) of the framework).**

1. **(a)** $`w(H_z)=0`$ if and only if $`H_z(i,j)=a_i+b_j`$ for some $`a\in\mathbb R^m`$ and $`b\in\mathbb R^n`$.
2. **(b)** For all $`\Gamma,\Gamma'\in\mathcal C(\mu,\nu)`$,

   ```math
   \lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\big\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\big\rVert_\infty. \qquad(46)
   ```

   The inequality can be strict.
3. **(b′)** **Restricted pairings.** If some pairs are forbidden, so that the feasible couplings form a proper face of $`\mathcal C`$, zero width does not imply additivity.
4. **(c)** **Strict additive null.** One SGD step at a common parameter, on the union of all packets with identical per-example weights, is invariant to how the examples are grouped into packets. Two or more sequential steps are not invariant in general. For quadratic losses the difference is of order $`\eta^2`$, and it is given explicitly below.

## Proof

**Part 1, "if".** For $`H=a\mathbf 1^\top+\mathbf 1b^\top`$,

```math
\langle\Gamma,H\rangle=\sum_ia_i\sum_j\Gamma_{ij}+\sum_jb_j\sum_i\Gamma_{ij}=\mu^\top a+\nu^\top b,
```

which is the same for every $`\Gamma\in\mathcal C(\mu,\nu)`$.

**Part 1, "only if".**

- $`\Gamma_0=\mu\nu^\top`$ lies in $`\mathcal C(\mu,\nu)`$ and is strictly positive.
- Fix rows $`i\ne i'`$ and columns $`j\ne j'`$, and let $`E=e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}`$. $`E`$ has zero row sums and zero column sums, so $`\Gamma_0\pm sE\in\mathcal C(\mu,\nu)`$ for $`0<s\le\min_{k,l}(\Gamma_0)_{kl}`$.
- Zero width means $`\langle\Gamma,H\rangle`$ is constant on $`\mathcal C`$. Hence $`\langle\Gamma_0+sE,H\rangle=\langle\Gamma_0-sE,H\rangle`$, that is $`\langle E,H\rangle=0`$:

  ```math
  H_{ij}+H_{i'j'}-H_{ij'}-H_{i'j}=0\quad\text{for all rectangles}.
  ```

- Put $`a_i=H_{i1}`$ and $`b_j=H_{1j}-H_{11}`$. The rectangle $`(i,1,j,1)`$ gives $`H_{ij}=H_{i1}+H_{1j}-H_{11}=a_i+b_j`$ (trivially when $`i=1`$ or $`j=1`$).

**Part 2.** $`\Delta=\Gamma-\Gamma'`$ has zero row and column sums, so $`\langle\Delta,a\mathbf 1^\top+\mathbf 1b^\top\rangle=0`$ for all $`a,b`$. Moreover $`\lVert\Delta\rVert_1\le\lVert\Gamma\rVert_1+\lVert\Gamma'\rVert_1=2`$. Hence, for every $`a,b`$,

```math
\lvert\langle\Delta,H\rangle\rvert=\lvert\langle\Delta,H-a\mathbf 1^\top-\mathbf 1b^\top\rangle\rvert\le\lVert\Delta\rVert_1\,\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty\le2\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty.
```

Take the infimum over $`(a,b)`$.

*Strictness.* Let $`\mu=\nu=(0.9,0.1)`$ and $`H=e_{11}`$.

- Every coupling has the form $`\Gamma_{11}=x`$, $`\Gamma_{12}=\Gamma_{21}=0.9-x`$ and $`\Gamma_{22}=x-0.8`$, with $`x\in[0.8,0.9]`$. So $`\langle\Gamma,H\rangle=x`$ and the width is $`0.1`$.
- Every additive table has rectangle sum $`0`$, while $`H`$ has rectangle sum $`1`$. So the residual $`R=H-a\mathbf 1^\top-\mathbf 1b^\top`$ has rectangle sum $`1`$, and some entry satisfies $`\lvert R_{kl}\rvert\ge1/4`$.
- The checkerboard $`R=\tfrac14\begin{pmatrix}1&-1\cr -1&1\end{pmatrix}`$ attains this, with $`a=(\tfrac34,\tfrac14)`$ and $`b=(0,-\tfrac12)`$.
- The right side of (46) is therefore $`2\cdot\tfrac14=0.5>0.1`$.

Couplings are a restricted feasible set (compare N06 and N13), so (46) is an upper bound, not a duality equality.

**Part 3.** Let $`\mu=\nu=(\tfrac12,\tfrac12)`$, forbid the pair $`(2,2)`$, and let $`H=e_{22}`$, the table $`\begin{pmatrix}0&0\cr 0&1\end{pmatrix}`$. Every feasible coupling has $`\Gamma_{22}=0`$, so $`\langle\Gamma,H\rangle=0`$ and the feasible width is $`0`$. But $`H`$ has rectangle sum $`1`$ and is not additive.

**Part 4.** Let the examples be $`\lbrace (x_k,w_k)\rbrace _{k\in I}`$, with per-example losses $`\ell_k`$, and let a grouping be a partition of $`I`$ into packets.

*One step at a common parameter.* The update is

```math
\theta^+=\theta-\eta\sum_{k\in I}w_k\nabla\ell_k(\theta).
```

It depends only on the multiset of examples and weights, not on the partition. Regrouping therefore changes nothing, which makes this a strict null: any measured difference indicates an implementation or randomness difference.

*Two sequential steps.* Packet $`P_1`$ is applied first, then $`P_2`$. Take quadratic losses $`\ell_k(\theta)=\tfrac12(x_k^\top\theta-y_k)^2`$ and write $`g_P(\theta)=\sum_{k\in P}w_k\nabla\ell_k(\theta)`$ and $`H_P=\sum_{k\in P}w_kx_kx_k^\top`$. Then

```math
\theta_1=\theta_0-\eta g_{P_1}(\theta_0),\qquad g_{P_2}(\theta_1)=g_{P_2}(\theta_0)-\eta H_{P_2}g_{P_1}(\theta_0),
```

so

```math
\theta_2=\theta_0-\eta\big(g_{P_1}(\theta_0)+g_{P_2}(\theta_0)\big)+\eta^2H_{P_2}\,g_{P_1}(\theta_0).
```

The first-order term depends only on $`P_1\cup P_2`$. The second-order term $`H_{P_2}g_{P_1}(\theta_0)`$ depends on which examples are in which packet, so it generally changes under regrouping. Nonlinear networks and Adam add further interactions. $`\square`$

## Remarks

1. **What a nonzero width means.** Fixed-marginal reorganization of authentic units can change the learning outcome only through the non-additive part of the joint-packet response $`H_z`$. This is a learning-consequence effect beyond per-unit quality, gradient variance or action-level cost. N16 asks whether that part exists.
2. **What $`H_z`$ must be.** It is the full-tail response to the whole packet under the declared learning experiment. A table of token-level losses is not a table of learning consequences.
3. **Scope.** The result is coupling linear algebra; no new duality is claimed. It turns the question "does packet organization matter for this learner?" into a measurable quantity with an exact null.

## Dependencies and source

- Uses [N05](N05_continuation_response.md) (the joint-packet response is a one-boundary response).

**Source:** [v0.14.1 §12](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#12-one-derived-mechanism-coupling-valid-experience-units) — Prop. 21; eq. (46). [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §12, Prop. 21, eq. (46). Review: [Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) §C1.
