# Proposition 9.1 · Coupling width

*Proof*

[← Framework](../README.md) · [Statement](../statements/09-01_coupling_width.md) · [Proof](09-01_coupling_width.md) · [Experiments](../experiments/09-01_coupling_width.md) · [中文](09-01_coupling_width_ZH.md)

**What is proved here.** Reorganizing which units are packed together, with both slot marginals fixed, changes the expected learning value only through the non-additive part of the packet response: the width is zero exactly for additive responses, and in general it is at most twice the distance of the response from additive tables. Forbidding pairings breaks the characterization, and one SGD step at a common parameter is exactly invariant to regrouping while two sequential steps are not. The proof is linear algebra on the coupling polytope plus a two-step expansion for quadratic losses.

**Setting used on this page.**

- $`\mathcal Y_1=\lbrace1,\dots,m\rbrace`$, $`\mathcal Y_2=\lbrace1,\dots,n\rbrace`$ are finite sets of valid units filling two slots of a training packet; every pair $`(i,j)`$ is a valid packet unless stated otherwise.
- $`\mu\in\Delta(\mathcal Y_1)`$, $`\nu\in\Delta(\mathcal Y_2)`$ are strictly positive slot marginals (otherwise work on the positive-support product).
- The coupling polytope is

```math
\mathcal C(\mu,\nu)=\lbrace \Gamma\in\mathbb R^{m\times n}_{\ge0}:\ \Gamma\mathbf 1=\mu,\ \Gamma^\top\mathbf 1=\nu\rbrace .
```

- $`H_z(i,j)`$ is the expected final task return after the actual update on packet $`(i,j)`$ at complete state $`z`$, followed by the reference tail: a one-boundary response with the packet as the message ([Definition 1.4](../README.md#s1-4), [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2)).
- The coupling width is

```math
w(H_z)=\max_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle-\min_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle .
```

- $`e_{ij}`$ is the table with a single 1 in entry $`(i,j)`$; the *rectangle sum* of a table $`H`$ at rows $`i\ne i'`$ and columns $`j\ne j'`$ is $`H_{ij}+H_{i'j'}-H_{ij'}-H_{i'j}`$.

**Statement ([Proposition 9.1](../statements/09-01_coupling_width.md#s9-1)).** (a) $`w(H_z)=0`$ if and only if $`H_z(i,j)=a_i+b_j`$ for some $`a,b`$. (b) For all $`\Gamma,\Gamma'\in\mathcal C(\mu,\nu)`$,

```math
\lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\big\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\big\rVert_\infty ,
```

and the inequality can be strict. (c) With forbidden pairings, zero feasible width does not imply additivity. (d) One SGD step at a common parameter, on the union of all packets with identical per-example weights, is invariant to regrouping; two or more sequential steps are not in general, and for quadratic losses the difference is of order $`\eta^2`$.

**Proof.**

*(a), "if".* For $`H=a\mathbf 1^\top+\mathbf 1b^\top`$,

```math
\langle\Gamma,H\rangle=\sum_ia_i\sum_j\Gamma_{ij}+\sum_jb_j\sum_i\Gamma_{ij}=\mu^\top a+\nu^\top b,
```

which is the same for every $`\Gamma\in\mathcal C(\mu,\nu)`$.

*(a), "only if".*

- $`\Gamma_0=\mu\nu^\top`$ lies in $`\mathcal C(\mu,\nu)`$ and is strictly positive.
- Fix rows $`i\ne i'`$ and columns $`j\ne j'`$, and let $`E=e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}`$. $`E`$ has zero row sums and zero column sums, so $`\Gamma_0\pm sE\in\mathcal C(\mu,\nu)`$ for $`0<s\le\min_{k,l}(\Gamma_0)_{kl}`$.
- Zero width means that $`\langle\Gamma,H\rangle`$ is constant on $`\mathcal C(\mu,\nu)`$. Hence $`\langle\Gamma_0+sE,H\rangle=\langle\Gamma_0-sE,H\rangle`$, that is $`\langle E,H\rangle=0`$:

```math
H_{ij}+H_{i'j'}-H_{ij'}-H_{i'j}=0\quad\text{for all rectangles}.
```

- Put $`a_i=H_{i1}`$ and $`b_j=H_{1j}-H_{11}`$. The rectangle $`(i,1,j,1)`$ gives $`H_{ij}=H_{i1}+H_{1j}-H_{11}=a_i+b_j`$; this is trivial when $`i=1`$ or $`j=1`$.

*(b).* $`\Delta=\Gamma-\Gamma'`$ has zero row and column sums, so $`\langle\Delta,a\mathbf 1^\top+\mathbf 1b^\top\rangle=0`$ for all $`a,b`$. Moreover $`\lVert\Delta\rVert_1\le\lVert\Gamma\rVert_1+\lVert\Gamma'\rVert_1=2`$. Hence, for every $`a,b`$,

```math
\lvert\langle\Delta,H\rangle\rvert=\lvert\langle\Delta,H-a\mathbf 1^\top-\mathbf 1b^\top\rangle\rvert\le\lVert\Delta\rVert_1\,\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty\le2\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty .
```

Take the infimum over $`(a,b)`$.

*Strictness.* Let $`\mu=\nu=(0.9,0.1)`$ and $`H=e_{11}`$.

- Every coupling has the form $`\Gamma_{11}=x`$, $`\Gamma_{12}=\Gamma_{21}=0.9-x`$, $`\Gamma_{22}=x-0.8`$ with $`x\in[0.8,0.9]`$. So $`\langle\Gamma,H\rangle=x`$ and the width is $`0.1`$.
- Every additive table has rectangle sum $`0`$, while $`H`$ has rectangle sum $`1`$. So the residual $`R=H-a\mathbf 1^\top-\mathbf 1b^\top`$ has rectangle sum $`1`$, and some entry satisfies $`\lvert R_{kl}\rvert\ge1/4`$.
- The checkerboard $`R=\tfrac14\begin{pmatrix}1&-1\cr -1&1\end{pmatrix}`$ attains this, with $`a=(\tfrac34,\tfrac14)`$ and $`b=(0,-\tfrac12)`$.
- The right side of (b) is therefore $`2\cdot\tfrac14=0.5>0.1`$.

Couplings form a restricted feasible set, unlike the unrestricted laws of [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2) (where the worst feature-matched discrepancy equals twice a best uniform approximation error), so (b) is an upper bound, not a duality equality.

*(c).* Let $`\mu=\nu=(\tfrac12,\tfrac12)`$, forbid the pair $`(2,2)`$, and let $`H=e_{22}`$, the table $`\begin{pmatrix}0&0\cr 0&1\end{pmatrix}`$. Every feasible coupling has $`\Gamma_{22}=0`$, so $`\langle\Gamma,H\rangle=0`$ and the feasible width is $`0`$. But $`H`$ has rectangle sum $`1`$ and is not additive.

*(d).* Let the examples be $`\lbrace (x_k,w_k)\rbrace _{k\in I}`$ with per-example losses $`\ell_k`$, and let a grouping be a partition of $`I`$ into packets.

*One step at a common parameter.* The update is

```math
\theta^+=\theta-\eta\sum_{k\in I}w_k\nabla\ell_k(\theta).
```

It depends only on the multiset of examples and weights, not on the partition. Regrouping therefore changes nothing, so this is a strict null: any measured difference indicates an implementation or randomness difference.

*Two sequential steps.* Packet $`P_1`$ is applied first, then $`P_2`$. Take quadratic losses $`\ell_k(\theta)=\tfrac12(x_k^\top\theta-y_k)^2`$ and write $`g_P(\theta)=\sum_{k\in P}w_k\nabla\ell_k(\theta)`$ and $`H_P=\sum_{k\in P}w_kx_kx_k^\top`$. Then

```math
\theta_1=\theta_0-\eta\,g_{P_1}(\theta_0),\qquad g_{P_2}(\theta_1)=g_{P_2}(\theta_0)-\eta H_{P_2}g_{P_1}(\theta_0),
```

so

```math
\theta_2=\theta_0-\eta\big(g_{P_1}(\theta_0)+g_{P_2}(\theta_0)\big)+\eta^2H_{P_2}\,g_{P_1}(\theta_0).
```

- The first-order term depends only on $`P_1\cup P_2`$.
- The second-order term $`H_{P_2}g_{P_1}(\theta_0)`$ depends on which examples are in which packet, so it generally changes under regrouping.
- Nonlinear networks and Adam add further interactions. $`\square`$

**Reading the proposition.**

- **What a nonzero width means.** Fixed-marginal reorganization of authentic units can change the learning outcome only through the non-additive part of the joint-packet response $`H_z`$. This is an effect on the learning consequence, separate from per-unit quality, gradient variance or action-level cost.
- **What $`H_z`$ must be.** It is the full-tail response to the whole packet under the declared learning experiment. A table of token-level losses is not a table of learning consequences.
- **Scope.** The result is coupling linear algebra; no new duality is claimed. It turns the question "does packet organization matter for this learner?" into a measurable quantity with an exact null.
