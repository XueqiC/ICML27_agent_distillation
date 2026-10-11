# N14 · Frozen finite-interface selection with transport — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N14_frozen_audit_transport.md) · [中文](N14_frozen_audit_transport_ZH.md)

## Setting and assumptions

**Learning process (as in N05).**

Rounds are $`t=0,\dots,R-1`$ and $`z`$ is a complete learner state: parameters, optimizer moments and step counter, buffers, sampler state and history.

At state $`z`$ in round $`t`$, the reference provider emits $`Y\sim P^0_{t,z}`$ and candidate $`j\in\lbrace 1,\dots,M\rbrace`$ emits $`Y\sim P^j_{t,z}`$.

The actual update is $`U_t(z,Y,\xi)`$, and the reference continuation is $`V^0_{t+1}`$, with $`V^0_R=F_E\in[0,1]`$.

The response is $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$.

Integrating the message gives the kernel identity

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace 0,1,\dots,M\rbrace . \qquad(K)
```

$`L_E(j)=\mathbb E F_E(Z_R^j)`$ for the process driven by provider $`j`$. All processes start from the same initial law of $`Z_0`$.

**Frozen audit.**

Using only independent pilot data, freeze the following before the audit:

- $`M`$ executable candidate providers;
- $`L`$ query cells $`x`$, each specifying a full learner state, task, query, update and reference suffix;
- $`d`$ features $`\phi_{x,\ell}:\mathcal Y\to[0,1]`$ per cell.

$`P^0_x`$ and $`P^j_x`$ are the message laws of the reference and of candidate $`j`$ in cell $`x`$.

The audit draws $`n_0`$ independent reference messages per cell and $`n_S`$ independent messages per candidate and cell.

Let $`\mu_{j,x,\ell}=\mathbb E_{P^j_x}\phi_{x,\ell}`$, with empirical means $`\widehat\mu_{j,x,\ell}`$. Define

```math
D_j=\max_{x,\ell}\lvert\mu_{0,x,\ell}-\mu_{j,x,\ell}\rvert,\qquad\widehat D_j=\max_{x,\ell}\lvert\widehat\mu_{0,x,\ell}-\widehat\mu_{j,x,\ell}\rvert,
```

```math
e_0=\sqrt{\frac{\log(4Ld/\delta)}{2n_0}},\qquad e_S=\sqrt{\frac{\log(4MLd/\delta)}{2n_S}}.
```

The selected candidate $`\widehat j`$ minimizes $`\widehat D_j`$ up to a known optimization error $`\eta_{\rm opt}`$.

**Certified states.** For each round $`t`$ there is a certified set $`\mathcal C_t`$ of complete states. Each $`z\in\mathcal C_t`$ carries:

**1.** an audited cell $`x=x(t,z)`$ whose features are expressed in the message coordinates at $`z`$;

**2.** a **transport bound**: for every candidate selectable on the audit event,

```math
\Big\lVert(P^0_{t,z}-P^j_{t,z})\phi_x-(P^0_x-P^j_x)\phi_x\Big\rVert_\infty\le a_{t,j}(z); \qquad(A2)
```

**3.** a **response decomposition**

```math
g_{t,z}=c_{t,z}+b_{t,z}^\top\phi_x+r_{t,z},\qquad\lVert b_{t,z}\rVert_1\le B_t,\quad\lVert r_{t,z}\rVert_\infty\le\tau_t. \qquad(A3)
```

Let $`\sigma=\inf\lbrace t<R:Z_t\notin\mathcal C_t\rbrace`$, with $`\sigma=\infty`$ if no such $`t`$ exists. $`\sigma`$ is a stopping time of the training process.

**Fresh training.** Conditional on the frozen audit result, the selected provider is trained with randomness independent of the audit samples, and $`\Pr_{\widehat j}(\sigma<R\mid\text{audit result})\le\rho`$.

## Statement

**Theorem (N14).**

**1.** **(a)** **Audit concentration.** With probability at least $`1-\delta`$, simultaneously for all $`j`$,

```math
\lvert D_j-\widehat D_j\rvert\le e_0+e_S, \qquad(37)
```

and consequently

```math
D_{\widehat j}\le\min_jD_j+2(e_0+e_S)+\eta_{\rm opt}. \qquad(38)
```

**2.** **(b)** **Local transport bound.** For $`z\in\mathcal C_t`$ and $`x=x(t,z)`$,

```math
\big\lvert(K^0_t-K^j_t)V^0_{t+1}(z)\big\rvert\le\min\Big\lbrace 1,\ B_t\big(D_j+a_{t,j}(z)\big)+2\tau_t\Big\rbrace . \qquad(V)
```

**3.** **(b)** **Learning-value bound.** On the event of (37),

```math
\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_{t=0}^{R-1}\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace \Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace \Big]+\rho. \qquad(39a)
```

**4.** **(c)** **Specializations.**

- With exact transport ($`a\equiv0`$), $`\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_tB_t(\widehat D_{\widehat j}+e_0+e_S)+2\sum_t\tau_t+\rho`$ (c).
- Under the state-independent IID block-source and convex-affine-update assumptions of [N08(a)](N08_affine_rates.md) (N08), with the audited features equal to the moments $`Z^\alpha`$, $`1\le\lvert\alpha\rvert\le N`$, $`\lvert L_E(0)-L_E(\widehat j)\rvert\le C_J(\widehat D_{\widehat j}+e_0+e_S)`$ (c).

**5.** **(b′)** **Feature drift.** If state-dependent features $`\phi_z`$ satisfy $`\sup_y\lVert\phi_z(y)-\phi_x(y)\rVert_\infty\le\alpha_{zx}`$, then

```math
\lVert(P^0_{t,z}-P^j_{t,z})\phi_z\rVert_\infty\le D_j+a_{t,j}(z)+2\alpha_{zx},
```

and (V) and (39a) hold with $`\phi_z`$ in (A3) and $`a`$ replaced by $`a+2\alpha_{zx}`$.

**6.** **(d)** **Necessity of transport.** For every audit size there is an instance in which $`D_j=0`$, the response decomposition is exact ($`B=1`$, $`\tau=0`$), and the learning gap is $`1`$. In that instance the transport-free expression $`B(\widehat D+e_0+e_S)`$ is strictly below the gap. (39a) remains valid there and is tight.

All bounds may be capped at $`1`$.

## Proof

**Part 1.** Each $`\phi_{x,\ell}\in[0,1]`$, and the samples behind each empirical mean are independent. Two-sided Hoeffding gives

```math
\Pr(\lvert\widehat\mu_{0,x,\ell}-\mu_{0,x,\ell}\rvert>e_0)\le2e^{-2n_0e_0^2}=\frac{\delta}{2Ld},
```

```math
\Pr(\lvert\widehat\mu_{j,x,\ell}-\mu_{j,x,\ell}\rvert>e_S)\le2e^{-2n_Se_S^2}=\frac{\delta}{2MLd}.
```

A union bound over the $`Ld`$ reference means and the $`MLd`$ candidate means gives an event $`G`$ with $`\Pr(G^c)\le\delta/2+\delta/2=\delta`$. Correlation between features evaluated on a common message does not affect a union bound. On $`G`$, for every $`j,x,\ell`$,

```math
\big\lvert(\widehat\mu_{0,x,\ell}-\widehat\mu_{j,x,\ell})-(\mu_{0,x,\ell}-\mu_{j,x,\ell})\big\rvert\le e_0+e_S.
```

The map $`u\mapsto\max_k\lvert u_k\rvert`$ is $`1`$-Lipschitz for the sup norm, so $`\lvert\widehat D_j-D_j\rvert\le e_0+e_S`$ for all $`j`$ simultaneously, which is (37).

For (38), let $`j^\star\in\arg\min_jD_j`$. On $`G`$,

```math
D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S\le\widehat D_{j^\star}+\eta_{\rm opt}+e_0+e_S\le D_{j^\star}+2(e_0+e_S)+\eta_{\rm opt}.
```

Selection on the audit data is legitimate because $`G`$ is simultaneous over the frozen candidate set.

**Part 2.** By (K), $`(K^0_t-K^j_t)V^0_{t+1}(z)=(P^0_{t,z}-P^j_{t,z})g_{t,z}`$. Insert (A3) term by term.

- The constant $`c_{t,z}`$ integrates to zero against a difference of probability laws.
- For the linear term, write $`\Delta=(P^0_{t,z}-P^j_{t,z})\phi_x\in\mathbb R^d`$. Then $`\lvert b^\top\Delta\rvert\le\lVert b\rVert_1\lVert\Delta\rVert_\infty`$. By (A2) and the triangle inequality, $`\lVert\Delta\rVert_\infty\le\lVert(P^0_x-P^j_x)\phi_x\rVert_\infty+a_{t,j}(z)\le D_j+a_{t,j}(z)`$. The last step uses $`\lVert(P^0_x-P^j_x)\phi_x\rVert_\infty=\max_\ell\lvert\mu_{0,x,\ell}-\mu_{j,x,\ell}\rvert\le D_j`$.
- For the residual, $`\lvert(P^0_{t,z}-P^j_{t,z})r\rvert\le\lvert P^0_{t,z}r\rvert+\lvert P^j_{t,z}r\rvert\le2\tau_t`$.

Since $`g\in[0,1]`$, the left side of (V) is at most $`1`$.

**Part 3 (stopped propagation).** Work on $`G`$, conditional on a fixed audit result, so $`\widehat j`$ is fixed. By the fresh-training assumption, the conditional law of the training process is that of the process driven by $`\widehat j`$.

*Step 3a (switched process).* Let $`S'`$ use kernel $`K^{\widehat j}_t`$ at round $`t`$ if $`t<\sigma`$, and $`K^0_t`$ otherwise. Since $`\lbrace \sigma\le t\rbrace`$ is determined by $`Z_0,\dots,Z_t`$, $`S'`$ is a well-defined history-dependent process. Augment the state with the flag $`1\lbrace t<\sigma\rbrace`$ if a Markov description is desired. $`V^0`$ does not depend on the flag.

Couple $`S'`$ with the $`\widehat j`$-process using the same randomness. The two paths agree up to and including $`Z_\sigma`$. On $`\lbrace \sigma\ge R\rbrace`$ they agree through $`Z_R`$. Terminal values lie in $`[0,1]`$, so

```math
\lvert L_E(\widehat j)-L_E(S')\rvert\le\Pr_{\widehat j}(\sigma<R)\le\rho.
```

*Step 3b (telescoping).* For any history-dependent process with kernels $`K^{S'}_t`$, the tower property gives

```math
\mathbb E\big[V^0_{t+1}(Z_{t+1})\mid Z_0,\dots,Z_t\big]=(K^{S'}_tV^0_{t+1})(Z_t),
```

so

```math
\mathbb E_{S'}\big[V^0_t(Z_t)-V^0_{t+1}(Z_{t+1})\big]=\mathbb E_{S'}\big[(K^0_t-K^{S'}_t)V^0_{t+1}(Z_t)\big].
```

Summing over $`t`$ and using the common initial law and $`V^0_R=F_E`$ gives

```math
L_E(0)-L_E(S')=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[(K^0_t-K^{S'}_t)V^0_{t+1}(Z_t)\big]=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[1\lbrace t<\sigma\rbrace (K^0_t-K^{\widehat j}_t)V^0_{t+1}(Z_t)\big].
```

The second equality holds because the kernels coincide for $`t\ge\sigma`$.

*Step 3c (pre-exit occupancy).* Before $`\sigma`$, $`S'`$ and the $`\widehat j`$-process use the same kernels. Hence $`(Z_0,\dots,Z_t)`$ restricted to $`\lbrace t<\sigma\rbrace`$ has the same law under both, and $`\mathbb E_{S'}[1\lbrace t<\sigma\rbrace h(Z_t)]=\mathbb E_{\widehat j}[1\lbrace t<\sigma\rbrace h(Z_t)]`$ for bounded $`h`$. On $`\lbrace t<\sigma\rbrace`$ we have $`Z_t\in\mathcal C_t`$, so (V) applies. On $`G`$, (37) gives $`D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S`$. Therefore

```math
\lvert L_E(0)-L_E(S')\rvert\le\sum_t\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace \Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace \Big].
```

*Step 3d.* The triangle inequality with Step 3a gives (39a). The bound holds for every audit result in $`G`$. It therefore requires (A2), (A3) and the bound on $`\rho`$ to hold for every candidate that can be selected on $`G`$.

**Part 4.**

- *(c).* With $`a\equiv0`$, each summand of (39a) is at most $`B_t(\widehat D_{\widehat j}+e_0+e_S)+2\tau_t`$, because $`1\lbrace t<\sigma\rbrace \le1`$ and the bracket is nonnegative.
- *(c).* Under the IID affine assumptions, every training round of provider $`j`$ draws a fresh block from one fixed law $`P^j`$, independent of the state. Take a single cell ($`L=1`$) whose features are the monomials $`Z^\alpha`$, $`1\le\lvert\alpha\rvert\le N`$; these lie in $`[0,1]`$. Then $`D_j=\delta_N(P^0,P^j)`$. [N08(a)](N08_affine_rates.md) (N08) gives $`\lvert L_E(0)-L_E(j)\rvert\le C_J\delta_N(P^0,P^j)`$, and on $`G`$, $`\delta_N(P^0,P^{\widehat j})\le\widehat D_{\widehat j}+e_0+e_S`$. No transport is needed, because the audited law is the law used at every state. The training blocks must be fresh draws from that audited law.

**Part 5.** Write $`\phi_z=\phi_x+(\phi_z-\phi_x)`$. The first part contributes at most $`D_j+a_{t,j}(z)`$, as in Part 2. For each coordinate $`\ell`$,

```math
\lvert(P^0_{t,z}-P^j_{t,z})(\phi_z-\phi_x)_\ell\rvert\le\lvert P^0_{t,z}(\phi_z-\phi_x)_\ell\rvert+\lvert P^j_{t,z}(\phi_z-\phi_x)_\ell\rvert\le2\alpha_{zx}.
```

Using the decomposition (A3) with $`\phi_z`$ and repeating Parts 2–3 proves the stated variants. A bound on sampled messages only is not a bound on $`\sup_y`$.

**Part 6 (necessity).** Let $`\mathcal Y=\lbrace 0,1\rbrace`$, $`\phi(y)=y`$ and $`R=1`$. Let the update set the final policy parameter to $`p=y`$, and let $`F_E(p)=p`$. Then $`g_{0,z}(y)=y`$ at every state, so (A3) holds with $`c=0`$, $`b=1`$, $`r=0`$, i.e. $`B=1`$ and $`\tau=0`$.

- At the audited state $`z_a`$, both providers emit $`0`$, so $`D_j=0`$.
- Training starts at $`z_b`$, where the reference emits $`0`$ and the candidate emits $`1`$. Then $`\lvert L_E(0)-L_E(j)\rvert=\lvert0-1\rvert=1`$.
- Declare $`z_b`$ certified with cell $`x(z_b)=z_a`$ but no transport term. The resulting expression is $`B(\widehat D+e_0+e_S)=e_0+e_S`$. For instance, with $`n_0=n_S=10^4`$, $`\delta=0.05`$ and $`L=M=d=1`$, this is $`2\sqrt{\log 80/20000}=0.0296`$, which is less than $`1`$.
- In (39a), certifying $`z_b`$ requires $`a_{0,j}(z_b)\ge\lvert(0-1)-(0-0)\rvert=1`$, giving a bound of at least $`1`$. Alternatively $`z_b\notin\mathcal C_0`$, so $`\sigma=0`$, $`\rho=1`$, and the bound is $`1`$.

In both cases (39a) equals the gap after capping. The transport term therefore cannot be dropped, and it cannot be assigned zero by convention.

The same construction with a covered state that reaches an uncovered state with probability $`r`$ shows a further point. Every visited pre-exit state must contribute its term, or the first visit to an uncovered state must be counted in $`\rho\ge r`$. Otherwise the bound fails by $`r`$. $`\square`$

## Remarks

1. **Complete states.** Cells, transport and certification refer to complete learner states. Two states with identical parameters but different Adam moments emit different next states (N11) and can have different responses. Parameter proximity alone does not establish (A2).
2. **Exact transport is weaker than law equality.** $`a=0`$ only requires the two feature-discrepancy vectors to agree, not the four laws.
3. **What the audit certifies.** (39a) bounds the population expected value of the selected frozen procedure under fresh training. It is not a bound on one training seed's score. Creating a new provider from the audit samples, or reusing them as training data, changes the experiment. It then needs a new independent audit.
4. **Additional confidence failures.** Confidence failures for transport, coefficients or residuals that are estimated statistically add to $`\delta`$.

## Dependencies and source

- Uses [N05](N05_continuation_response.md) (the telescoping identity and its stopped form) and [N13](N13_stability_tradeoff.md) (coefficient-weighted feature discrepancies). (c) uses [N08](N08_affine_rates.md) ([N08(a)](N08_affine_rates.md)).

**Source:** [v0.14.1 Theorem 19](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-19-frozen-finite-interface-selection); [v0.15 §2](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#2-the-corrected-finite-audit-bridge) — Thm. 19; eqs. (37)–(40), (39a); Prop. A1; eqs. (A2)–(A5). [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §10, Thm. 19, eqs. (37)–(40), (39a). [Optimizer_State_and_Few_Shot_Theory_v015_EN.md](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md) §2, Prop. A1, eqs. (A2)–(A5). Review: [Proof_Review_v015_EN.md](../../docs/direction/v015_revision/Proof_Review_v015_EN.md) §V15.1, and [Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) §A27.
