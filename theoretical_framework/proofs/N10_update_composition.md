# N10 · Update composition amplifies the information requirement — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N10_update_composition.md) · [中文](N10_update_composition_ZH.md)

## Setting and assumptions

**Blocks and updates.** Fix a horizon $`R\ge1`$ and a common initial parameter $`p_0\in[0,1]^k`$. Independent experience blocks $`Z_1,\dots,Z_R`$ take values in a finite set $`\mathcal Y\subset[0,1]^k`$. Block $`Z_i`$ has law $`P_i`$ under the reference source and $`Q_i`$ under the alternative source. The learner applies

```math
p_i=U_i(p_{i-1},Z_i),\qquad i=1,\dots,R,
```

where every coordinate of $`U_i:\mathbb R^k\times\mathbb R^k\to\mathbb R^k`$ is a polynomial with the following properties.

- **(D1)** For fixed $`Z_i`$, it has total degree at most $`a_i\ge1`$ in the old parameter vector $`p_{i-1}`$.
- **(D2)** For fixed $`p_{i-1}`$, it has total degree at most $`b_i\ge1`$ in the new block $`Z_i`$.

We assume $`U_i`$ maps $`[0,1]^k\times\mathcal Y`$ into $`[0,1]^k`$, so every iterate is a valid policy parameter. The numbers $`a_i,b_i`$ are upper bounds; a coordinate may have smaller degree.

**Deployment.** $`J:[0,1]^k\to[0,1]`$ is a polynomial of total degree at most $`N`$. By [N07(a)](N07_affine_moment_order.md) (node [N07](N07_affine_moment_order.md)), this holds for every finite binary-context deployment with at most $`N`$ decisions. The learning value of a source is $`\mathbb E\,J(p_R)`$.

**Smooth local setting (for the SGD expansion).** The parameter is $`\theta\in\mathbb R^p`$ and $`V:\mathbb R^p\to\mathbb R`$ is a continuation value (node [N05](N05_continuation_response.md)). A one-step SGD kernel with consumed-gradient law $`P`$ acts as

```math
(K^PV)(\theta)=\mathbb E_{G\sim P}\,V(\theta-\eta G),
```

with step size $`\eta>0`$. We write $`\mu_P=\mathbb E_PG`$ and $`M_P=\mathbb E_PGG^\top`$.

- **(T3)** $`V`$ is three times continuously differentiable on every segment $`[\theta,\theta-\eta g]`$ with $`g`$ in the supports of $`P`$ and $`Q`$. On these segments, $`\lvert D^3V(\vartheta)[h,h,h]\rvert\le M_3\lVert h\rVert^3`$ for all $`h`$.

**Error recursion.** A nonnegative sequence $`(e_t)`$ satisfies $`e_0=0`$ and

```math
e_{t+1}\le(1+\eta L)e_t+C\eta^2,\qquad t\ge0,
```

for constants $`\eta,L,C\ge0`$.

## Statement

**(a)** **Theorem (degree composition).** Under (D1)–(D2), the final response $`J(p_R)`$, viewed as a polynomial in $`(Z_1,\dots,Z_R)`$, has total degree in block $`Z_i`$ at most

```math
D_i=N\,b_i\prod_{t=i+1}^{R}a_t. \qquad(34)
```

Consequently, suppose that for every $`i`$, $`\mathbb E_{P_i}Z^\alpha=\mathbb E_{Q_i}Z^\alpha`$ for all multi-indices with $`\lvert\alpha\rvert\le D_i`$. Then $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$. If all blocks share one source law, matching through $`\max_iD_i`$ suffices.

**(a)** **Proposition (attainment).** For every $`R\ge1`$ there is an update sequence within the class, and a deployment with $`N=1`$, whose response is $`Z_1^{2^{R-1}}`$. The scalar laws of order $`2^{R-1}`$ from [N07(c)](N07_affine_moment_order.md) agree on all moments below $`2^{R-1}`$, yet give different learning values. The degree bound (34) therefore cannot be improved in general.

**(b)** **Proposition (threshold learner).** Consider the non-polynomial learner $`p=1\lbrace Z>1/2\rbrace`$ with $`J(p)=p`$ ($`N=1`$). Take

```math
P(Z=0)=\tfrac13,\quad P(Z=\tfrac34)=\tfrac23,\qquad Q(Z=\tfrac12)=1.
```

Then $`\mathbb E_PZ=\mathbb E_QZ=\tfrac12`$, while $`\mathbb E_PJ(p)=\tfrac23`$ and $`\mathbb E_QJ(p)=0`$.

**(c)** **Proposition (local SGD expansion).** Under (T3),

```math
(K^P-K^Q)V(\theta)=-\eta\nabla V(\theta)^\top(\mu_P-\mu_Q)+\frac{\eta^2}{2}\langle\nabla^2V(\theta),M_P-M_Q\rangle+\mathcal E, \qquad(35)
```

```math
\lvert\mathcal E\rvert\le\frac{M_3\eta^3}{6}\left(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3\right).
```

**(d)** **Proposition (error accumulation).** Under the error recursion,

```math
e_R\le C\eta^2\sum_{s=0}^{R-1}(1+\eta L)^s=\frac{C\eta}{L}\big((1+\eta L)^R-1\big)\quad(L>0),
```

and the bound holds with equality when the recursion holds with equality. The linear simplification $`e_R\le RC\eta^2`$ holds when $`L=0`$. For $`L>0`$ it fails in general: with $`\eta=\tfrac12`$ and $`L=C=1`$, the equality recursion gives $`e_4=65/32>1=RC\eta^2`$.

## Proof

**Proof of (a).** Define backward continuation polynomials

```math
\Phi_R(p)=J(p),\qquad \Phi_{t-1}(p;Z_t,\dots,Z_R)=\Phi_t\big(U_t(p,Z_t);Z_{t+1},\dots,Z_R\big),\quad t=R,\dots,1.
```

Then $`J(p_R)=\Phi_0(p_0;Z_1,\dots,Z_R)`$.

*Claim.* For each $`t\in\lbrace 0,\dots,R\rbrace`$:

- $`\Phi_t`$ has total degree at most $`d_t:=N\prod_{s=t+1}^Ra_s`$ in $`p`$, with $`d_R=N`$;
- for each $`i>t`$, $`\Phi_t`$ has total degree at most $`D_i`$ in $`Z_i`$.

*Base case $`t=R`$.* $`\Phi_R=J`$ has degree at most $`N`$ in $`p`$ and contains no blocks.

*Inductive step.* Assume the claim at $`t`$. Write $`\Phi_t=\sum_{\lvert\gamma\rvert\le d_t}c_\gamma(Z_{t+1},\dots,Z_R)\,p^\gamma`$. By hypothesis, each coefficient $`c_\gamma`$ has degree at most $`D_i`$ in $`Z_i`$ for $`i>t`$.

Substitute $`p=U_t(p',Z_t)`$. The monomial $`p^\gamma`$ becomes a product of $`\lvert\gamma\rvert`$ coordinate polynomials of $`U_t`$. By (D1) each factor has degree at most $`a_t`$ in $`p'`$, and by (D2) at most $`b_t`$ in $`Z_t`$. Degrees of products add. Hence the substituted monomial has degree at most $`a_t\lvert\gamma\rvert\le a_td_t=d_{t-1}`$ in $`p'`$, and at most $`b_t\lvert\gamma\rvert\le b_td_t=Nb_t\prod_{s>t}a_s=D_t`$ in $`Z_t`$. The coefficients $`c_\gamma`$ do not involve $`p'`$ or $`Z_t`$, so the degrees in $`Z_i`$ for $`i>t`$ are unchanged.

This proves the claim at $`t-1`$. At $`t=0`$, $`p=p_0`$ is a constant, so $`J(p_R)`$ has degree at most $`D_i`$ in each block $`Z_i`$.

*Moment consequence.* Expand $`J(p_R)=\sum_{\alpha_1,\dots,\alpha_R}c_{\alpha_1\cdots\alpha_R}\prod_iZ_i^{\alpha_i}`$, a finite sum with $`\lvert\alpha_i\rvert\le D_i`$. The blocks are independent, so

```math
\mathbb E\prod_iZ_i^{\alpha_i}=\prod_i\mathbb E_{P_i}Z_i^{\alpha_i}.
```

Under the matching hypothesis, every factor is the same under $`P_i`$ and $`Q_i`$. By linearity of expectation over the finite expansion, $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$. If $`P_i=P`$ and $`Q_i=Q`$ for all $`i`$, then matching through $`\max_iD_i`$ implies matching through each $`D_i`$. $`\square`$

**Proof of (a).** Set $`U_1(p,z)=z`$ (so $`b_1=1`$) and $`U_i(p,z)=p\odot p`$ for $`i\ge2`$ (so $`a_i=2`$, and $`b_i=1`$ as an upper bound). Both maps keep $`[0,1]^k`$ invariant. Deploy once with $`J(p)=p_1`$ ($`N=1`$), using the first coordinate. By induction, $`p_i=Z_1^{2^{i-1}}`$ coordinatewise, so the response is $`Z_{1,1}^{2^{R-1}}`$. Here (34) gives $`D_1=1\cdot1\cdot2^{R-1}`$, so the bound is attained.

Now let $`P_M,Q_M`$ be the scalar laws of [N07(c)](N07_affine_moment_order.md) with $`M=2^{R-1}`$ (node [N07](N07_affine_moment_order.md), [N07(c)](N07_affine_moment_order.md)). Their moments of order $`\ell<M`$ agree, and $`\mathbb E_{P_M}Z^M-\mathbb E_{Q_M}Z^M=\Delta_M=(-1)^MM!/(2^{M-1}M^M)\ne0`$. So the two sources match every moment below the attained degree and still produce different learning values. Later blocks are unused, which (D2) permits. $`\square`$

**Proof of (b).** $`\mathbb E_PZ=\tfrac13\cdot0+\tfrac23\cdot\tfrac34=\tfrac12=\mathbb E_QZ`$. Under $`P`$, $`1\lbrace Z>1/2\rbrace =1`$ exactly when $`Z=3/4`$, which has probability $`2/3`$. Under $`Q`$, $`Z=1/2`$ is not greater than $`1/2`$, so the value is $`0`$. $`\square`$

**Proof of (c).** Fix $`g`$ in either support and put $`\varphi(\tau)=V(\theta-\tau\eta g)`$ for $`\tau\in[0,1]`$. By (T3), $`\varphi\in C^3[0,1]`$. Taylor's theorem with integral remainder gives

```math
\varphi(1)=\varphi(0)+\varphi'(0)+\tfrac12\varphi''(0)+\int_0^1\frac{(1-\tau)^2}{2}\varphi'''(\tau)\,d\tau.
```

The derivatives are

```math
\varphi'(0)=-\eta\nabla V(\theta)^\top g,\qquad\varphi''(0)=\eta^2g^\top\nabla^2V(\theta)g,\qquad\varphi'''(\tau)=-\eta^3D^3V(\theta-\tau\eta g)[g,g,g].
```

By (T3), $`\lvert\varphi'''(\tau)\rvert\le M_3\eta^3\lVert g\rVert^3`$. Since $`\int_0^1(1-\tau)^2/2\,d\tau=1/6`$, we get

```math
V(\theta-\eta g)=V(\theta)-\eta\nabla V^\top g+\tfrac{\eta^2}{2}\langle\nabla^2V,gg^\top\rangle+\rho(g),\qquad\lvert\rho(g)\rvert\le\tfrac{M_3\eta^3}{6}\lVert g\rVert^3.
```

Take $`\mathbb E_P`$ and $`\mathbb E_Q`$ and subtract. The constant $`V(\theta)`$ cancels. The linear and quadratic terms give $`-\eta\nabla V^\top(\mu_P-\mu_Q)`$ and $`\tfrac{\eta^2}{2}\langle\nabla^2V,M_P-M_Q\rangle`$. The remainder is $`\mathcal E=\mathbb E_P\rho(G)-\mathbb E_Q\rho(G)`$, and the triangle inequality bounds it by $`\tfrac{M_3\eta^3}{6}(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3)`$. $`\square`$

**(d)** **Proposition.** Induction on $`t`$ shows $`e_t\le C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s`$.

- The case $`t=0`$ is the empty sum.
- If the bound holds at $`t`$, then $`e_{t+1}\le(1+\eta L)C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s+C\eta^2=C\eta^2\sum_{s=0}^{t}(1+\eta L)^s`$.

Each step uses only the recursion, so equality propagates when the recursion is an equality. The closed form is the geometric sum.

For the witness, the equality recursion with $`\eta=\tfrac12`$, $`L=C=1`$ is $`e_{t+1}=\tfrac32e_t+\tfrac14`$. It gives $`e_1=\tfrac14`$, $`e_2=\tfrac58`$, $`e_3=\tfrac{19}{16}`$ and $`e_4=\tfrac{65}{32}`$, which exceeds $`RC\eta^2=1`$. $`\square`$

## Remarks

1. **What the theorem says operationally.** The information order a source must match is a property of the *learner composed with the deployment*, not of the deployment alone. With a single convex-affine update ($`a_i=b_i=1`$), (34) recovers $`D_i=N`$, which is [N07(b)](N07_affine_moment_order.md) of node [N07](N07_affine_moment_order.md). Each nonlinear later update multiplies the requirement on all earlier blocks.
2. **Degree is coordinate-dependent.** An invertible reparameterization can change the polynomial form of an update. The intrinsic object remains the response class and its approximation width (node [N06](N06_sufficiency_duality.md)). Degree is a tractable certificate in the chosen probability coordinates.
3. **Non-polynomial learners.** (b) shows that no moment order is sufficient in general once the update is discontinuous. Even one decision does not impose a first-moment requirement on arbitrary learners.
4. **Scope of (35).** The expansion is local and one-step. Its first-order statistic is the mean consumed gradient and its second-order statistic is the gradient second moment. Applying it with $`V`$ replaced by a training loss, or to Adam, needs a separate argument. Node [N11](N11_adam_cold_start.md) gives the cold-Adam replacement and node [N12](N12_optimizer_warmth.md) the warm-Adam replacement.
5. **Accumulation needs stability.** (d) is sharp for its recursion. A bound linear in $`R`$ requires $`\eta LR=O(1)`$, since $`\sum_{s<R}(1+\eta L)^s\le Re^{\eta LR}`$, or a contractive error map.

## Dependencies and source

- Uses: [N07](N07_affine_moment_order.md) ([N07(a)](N07_affine_moment_order.md) polynomial deployment; laws [N07(c)](N07_affine_moment_order.md)), [N05](N05_continuation_response.md) (continuation kernels $`K^P`$), [N06](N06_sufficiency_duality.md) (response-class view).
- Used by: [N11](N11_adam_cold_start.md), [N12](N12_optimizer_warmth.md).

**Source:** [v0.14.1 §9](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#9-learning-dynamics-can-amplify-the-information-requirement); [v0.15 §7](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#7-experiments-that-discriminate-the-theory) — Thm. 17; eqs. (34)–(35); addendum §7 recursion remark. v0.14.1 [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §9, Theorem 17, eqs. (34)–(35); v0.15 addendum [Optimizer_State_and_Few_Shot_Theory_v015_EN.md](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md) §7, recursion remark.
