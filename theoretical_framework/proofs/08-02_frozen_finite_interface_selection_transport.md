# Theorem 8.2 · Frozen finite-interface selection with transport

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-02_frozen_finite_interface_selection_transport.md) · [Proof](../proofs/08-02_frozen_finite_interface_selection_transport.md) · [Experiments](../experiments/08-02_frozen_finite_interface_selection_transport.md) · [中文](08-02_frozen_finite_interface_selection_transport_ZH.md)

**What is proved here.** Simultaneous concentration of the audited discrepancies; a local bound at a certified state through the response decomposition and the transport bound; the learning-value bound through a switched process and the stopped telescoping identity; two specializations; feature drift; and a witness showing that the transport term is necessary.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- Rounds are $`t=0,\dots,R-1`$, and $`z`$ is a complete learner state. For a first-order optimizer the actual update is $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$.
- At state $`z`$ in round $`t`$ the reference emits $`Y\sim P^0_{t,z}`$, and candidate $`j\in\lbrace1,\dots,M\rbrace`$ emits $`Y\sim P^j_{t,z}`$. The kernel of candidate $`j`$ is $`K^j_t`$.
- The reference continuation is $`V^0_{t+1}`$, with $`V^0_R=F_E\in[0,1]`$. The response is $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$. Integrating the message gives the kernel identity

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- $`L_E(j)=\mathbb E\,F_E(Z^j_R)`$ for the process driven by candidate $`j`$. All processes start from the same initial law of $`Z_0`$.
- The *local advantage* of candidate $`j`$ at $`(t,z)`$ is

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- It is the effect on the final score of replacing one reference round by one round of candidate $`j`$, followed by the reference tail. By [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2), the whole-run difference is a sum of such terms over the states the alternative process reaches.

**Frozen audit.** Using only independent pilot data, freeze before the audit:

- $`M`$ executable candidate feedback laws;
- a finite set $`\mathcal U`$ of audit cells $`u`$, each specifying a complete learner state, a task, a query, the update and the reference tail (the letter $`u`$ denotes a cell only in this subsection);
- $`d`$ features $`\phi_{u,\ell}:\mathcal Y\to[0,1]`$ per cell.

$`P^0_u`$ and $`P^j_u`$ are the message laws of the reference and of candidate $`j`$ in cell $`u`$. The audit draws $`n_0`$ independent reference messages per cell and $`n_S`$ independent messages per candidate and cell. Let $`\mu_{j,u,\ell}=\mathbb E_{P^j_u}\phi_{u,\ell}`$, with empirical means $`\widehat\mu_{j,u,\ell}`$, and

```math
D_j=\max_{u,\ell}\lvert\mu_{0,u,\ell}-\mu_{j,u,\ell}\rvert,\qquad\widehat D_j=\max_{u,\ell}\lvert\widehat\mu_{0,u,\ell}-\widehat\mu_{j,u,\ell}\rvert,
```

```math
e_0=\sqrt{\frac{\log(4\lvert\mathcal U\rvert d/\delta)}{2n_0}},\qquad e_S=\sqrt{\frac{\log(4M\lvert\mathcal U\rvert d/\delta)}{2n_S}}.
```

The selected candidate $`\widehat j`$ minimizes $`\widehat D_j`$ up to a known optimization error $`\eta_{\rm opt}`$.

**Certified states.** For each round $`t`$ there is a certified set $`\mathcal C_t`$ of complete states. Each $`z\in\mathcal C_t`$ carries the following.

- An audited cell $`u=u(t,z)`$ whose features are expressed in the message coordinates at $`z`$.
- A *transport bound*: for every candidate that can be selected on the audit event,

```math
\Big\lVert(P^0_{t,z}-P^j_{t,z})\phi_u-(P^0_u-P^j_u)\phi_u\Big\rVert_\infty\le a_{t,j}(z).\qquad\text{(T)}
```

- A *response decomposition*

```math
g_{t,z}=c_{t,z}+b_{t,z}^\top\phi_u+r_{t,z},\qquad\lVert b_{t,z}\rVert_1\le B_t,\quad\lVert r_{t,z}\rVert_\infty\le\tau_t.\qquad\text{(D)}
```

Let $`\sigma=\inf\lbrace t<R:Z_t\notin\mathcal C_t\rbrace`$ ($`\sigma=\infty`$ if no such $`t`$), a stopping time of the training process. **Fresh training:** conditional on the audit result, the selected candidate is trained with randomness independent of the audit samples, and $`\Pr_{\widehat j}(\sigma<R\mid\text{audit result})\le\rho`$.

**Statement ([Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)).** All bounds below may be capped at $`1`$.

**(a) Audit concentration.** With probability at least $`1-\delta`$, simultaneously for all $`j`$,

```math
\lvert D_j-\widehat D_j\rvert\le e_0+e_S,\qquad\Longrightarrow\qquad D_{\widehat j}\le\min_jD_j+2(e_0+e_S)+\eta_{\rm opt}.
```

**(b) Local transport bound.** For $`z\in\mathcal C_t`$ and $`u=u(t,z)`$,

```math
\big\lvert(K^0_t-K^j_t)V^0_{t+1}(z)\big\rvert\le\min\Big\lbrace1,\ B_t\big(D_j+a_{t,j}(z)\big)+2\tau_t\Big\rbrace .
```

**(c) Learning-value bound.** On the event of (a),

```math
\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_{t=0}^{R-1}\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace\Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace\Big]+\rho .
```

- **(d) Specializations.** With exact transport ($`a\equiv0`$), $`\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_tB_t(\widehat D_{\widehat j}+e_0+e_S)+2\sum_t\tau_t+\rho`$. Under the state-independent IID-block and convex-affine assumptions of [Theorem 4.5(a)](../statements/04-05_round_independent_bound_averaging_rates.md#s4-5), with the audited features equal to the moments $`Z^\alpha`$, $`1\le\lvert\alpha\rvert\le N`$, $`\lvert L_E(0)-L_E(\widehat j)\rvert\le C_J(\widehat D_{\widehat j}+e_0+e_S)`$.
- **(e) Feature drift.** If state-dependent features $`\phi_z`$ satisfy $`\sup_y\lVert\phi_z(y)-\phi_u(y)\rVert_\infty\le\alpha_{zu}`$, then $`\lVert(P^0_{t,z}-P^j_{t,z})\phi_z\rVert_\infty\le D_j+a_{t,j}(z)+2\alpha_{zu}`$, and (b)–(c) hold with $`\phi_z`$ in (D) and $`a`$ replaced by $`a+2\alpha_{zu}`$.
- **(f) Necessity of transport.** For every audit size there is an instance with $`D_j=0`$, an exact response decomposition ($`B=1`$, $`\tau=0`$) and learning gap $`1`$. In it the transport-free expression $`B(\widehat D+e_0+e_S)`$ is strictly below the gap, while the bound of (c) remains valid and is tight.

*Proof.*

**(a)** Each $`\phi_{u,\ell}\in[0,1]`$, and the samples behind each empirical mean are independent. Two-sided Hoeffding ([Hoeffding, 1963](../references.md#hoeffding1963probability)) gives

```math
\Pr\big(\lvert\widehat\mu_{0,u,\ell}-\mu_{0,u,\ell}\rvert>e_0\big)\le2e^{-2n_0e_0^2}=\frac{\delta}{2\lvert\mathcal U\rvert d},\qquad\Pr\big(\lvert\widehat\mu_{j,u,\ell}-\mu_{j,u,\ell}\rvert>e_S\big)\le\frac{\delta}{2M\lvert\mathcal U\rvert d}.
```

- A union bound over the $`\lvert\mathcal U\rvert d`$ reference means and the $`M\lvert\mathcal U\rvert d`$ candidate means gives an event $`\mathcal A`$ with $`\Pr(\mathcal A^c)\le\delta`$. Correlation between features evaluated on a common message does not affect a union bound.
- On $`\mathcal A`$, $`\big\lvert(\widehat\mu_{0,u,\ell}-\widehat\mu_{j,u,\ell})-(\mu_{0,u,\ell}-\mu_{j,u,\ell})\big\rvert\le e_0+e_S`$ for every $`j,u,\ell`$. The map $`w\mapsto\max_k\lvert w_k\rvert`$ is $`1`$-Lipschitz for the sup norm, so $`\lvert\widehat D_j-D_j\rvert\le e_0+e_S`$ for all $`j`$.
- Let $`j^\star\in\arg\min_jD_j`$. On $`\mathcal A`$, $`D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S\le\widehat D_{j^\star}+\eta_{\rm opt}+e_0+e_S\le D_{j^\star}+2(e_0+e_S)+\eta_{\rm opt}`$. Selecting on the audit data is legitimate because $`\mathcal A`$ is simultaneous over the frozen candidate set.

**(b)** By (K), $`(K^0_t-K^j_t)V^0_{t+1}(z)=(P^0_{t,z}-P^j_{t,z})g_{t,z}`$. Insert (D) term by term.

- The constant $`c_{t,z}`$ integrates to zero against a difference of probability laws.
- For the linear term, put $`w=(P^0_{t,z}-P^j_{t,z})\phi_u\in\mathbb R^d`$. Then $`\lvert b^\top w\rvert\le\lVert b\rVert_1\lVert w\rVert_\infty`$, and by (T) and the triangle inequality $`\lVert w\rVert_\infty\le\lVert(P^0_u-P^j_u)\phi_u\rVert_\infty+a_{t,j}(z)\le D_j+a_{t,j}(z)`$.
- For the residual, $`\lvert(P^0_{t,z}-P^j_{t,z})r\rvert\le\lvert P^0_{t,z}r\rvert+\lvert P^j_{t,z}r\rvert\le2\tau_t`$.
- Since $`g\in[0,1]`$, the left side is also at most $`1`$.

**(c)** Work on $`\mathcal A`$, conditional on a fixed audit result, so $`\widehat j`$ is fixed. By fresh training, the conditional law of the training process is that of the process driven by $`\widehat j`$.

- *Switched process.* Let $`S'`$ use $`K^{\widehat j}_t`$ at round $`t`$ if $`t<\sigma`$ and $`K^0_t`$ otherwise. Since $`\lbrace\sigma\le t\rbrace`$ is determined by $`Z_0,\dots,Z_t`$, $`S'`$ is a well-defined history-dependent process; $`V^0`$ does not depend on the switch flag. Coupling $`S'`$ with the $`\widehat j`$-process through common randomness, the two paths agree through $`Z_\sigma`$, and through $`Z_R`$ on $`\lbrace\sigma\ge R\rbrace`$. Terminal values lie in $`[0,1]`$, so $`\lvert L_E(\widehat j)-L_E(S')\rvert\le\Pr_{\widehat j}(\sigma<R)\le\rho`$.
- *Telescoping.* By the tower property, $`\mathbb E[V^0_{t+1}(Z_{t+1})\mid Z_0,\dots,Z_t]=(K^{S'}_tV^0_{t+1})(Z_t)`$. Summing $`\mathbb E_{S'}[V^0_t(Z_t)-V^0_{t+1}(Z_{t+1})]`$ over $`t`$, with the common initial law and $`V^0_R=F_E`$, gives

```math
L_E(0)-L_E(S')=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[(K^0_t-K^{S'}_t)V^0_{t+1}(Z_t)\big]=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[1\lbrace t<\sigma\rbrace(K^0_t-K^{\widehat j}_t)V^0_{t+1}(Z_t)\big],
```

because the kernels coincide for $`t\ge\sigma`$. This is the stopped form of [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization.md#s3-2).

- *Pre-exit occupancy.* Before $`\sigma`$, $`S'`$ and the $`\widehat j`$-process use the same kernels, so $`\mathbb E_{S'}[1\lbrace t<\sigma\rbrace h(Z_t)]=\mathbb E_{\widehat j}[1\lbrace t<\sigma\rbrace h(Z_t)]`$ for bounded $`h`$. On $`\lbrace t<\sigma\rbrace`$, $`Z_t\in\mathcal C_t`$ and (b) applies; on $`\mathcal A`$, $`D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S`$. This bounds $`\lvert L_E(0)-L_E(S')\rvert`$ by the sum in (c).
- The triangle inequality with the switched-process bound gives (c). The bound holds for every audit result in $`\mathcal A`$, so (T), (D) and the bound on $`\rho`$ must hold for every candidate that can be selected on $`\mathcal A`$.

**(d)** With $`a\equiv0`$, each summand of (c) is at most $`B_t(\widehat D_{\widehat j}+e_0+e_S)+2\tau_t`$, since $`1\lbrace t<\sigma\rbrace\le1`$ and the bracket is nonnegative. Under the IID affine assumptions every training round of candidate $`j`$ draws a fresh block from one fixed law $`P^j`$, independent of the state. Take a single cell whose features are the monomials $`Z^\alpha`$, $`1\le\lvert\alpha\rvert\le N`$, which lie in $`[0,1]`$. Then $`D_j=\delta_N(P^0,P^j)`$, [Theorem 4.5(a)](../statements/04-05_round_independent_bound_averaging_rates.md#s4-5) gives $`\lvert L_E(0)-L_E(j)\rvert\le C_J\delta_N(P^0,P^j)`$, and on $`\mathcal A`$, $`\delta_N(P^0,P^{\widehat j})\le\widehat D_{\widehat j}+e_0+e_S`$. No transport is needed, because the audited law is the law used at every state; the training blocks must be fresh draws from it.

**(e)** Write $`\phi_z=\phi_u+(\phi_z-\phi_u)`$. The first part contributes at most $`D_j+a_{t,j}(z)`$ as in (b). For each coordinate $`\ell`$, $`\lvert(P^0_{t,z}-P^j_{t,z})(\phi_z-\phi_u)_\ell\rvert\le2\alpha_{zu}`$. Repeating (b)–(c) with (D) stated for $`\phi_z`$ proves the variant. A bound on sampled messages only is not a bound on $`\sup_y`$.

**(f)** Let $`\mathcal Y=\lbrace0,1\rbrace`$, $`\phi(y)=y`$, $`R=1`$, let the update set the final policy parameter to $`p=y`$, and let $`F_E(p)=p`$. Then $`g_{0,z}(y)=y`$ at every state, so (D) holds with $`c=0`$, $`b=1`$, $`r=0`$.

- At the audited state $`z_a`$ both laws emit $`0`$, so $`D_j=0`$.
- Training starts at $`z_b`$, where the reference emits $`0`$ and the candidate emits $`1`$, so $`\lvert L_E(0)-L_E(j)\rvert=1`$.
- Declaring $`z_b`$ certified with cell $`u(z_b)=z_a`$ and no transport term gives the expression $`B(\widehat D+e_0+e_S)=e_0+e_S`$. With $`n_0=n_S=10^4`$, $`\delta=0.05`$ and $`\lvert\mathcal U\rvert=M=d=1`$ this is $`2\sqrt{\log80/20000}\approx0.0296<1`$.
- In (c), certifying $`z_b`$ requires $`a_{0,j}(z_b)\ge\lvert(0-1)-(0-0)\rvert=1`$, so the bound is at least $`1`$. Alternatively $`z_b\notin\mathcal C_0`$, so $`\sigma=0`$, $`\rho=1`$ and the bound is $`1`$.

In both cases (c) equals the gap after capping, so the transport term can be neither dropped nor set to zero by convention. The same construction with a covered state that reaches an uncovered state with probability $`r`$ shows that every visited pre-exit state must contribute its term, or the first visit to an uncovered state must be counted in $`\rho\ge r`$; otherwise the bound fails by $`r`$. $`\square`$

## References

- [Hoeffding, 1963](../references.md#hoeffding1963probability). Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association 58, no. 301, 13–30.
