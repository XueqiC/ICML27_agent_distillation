# Theorem 8.3 · Actual-state adaptive audit; fidelity versus improvement

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-03_actual_state_adaptive_audit_fidelity.md) · [Proof](../proofs/08-03_actual_state_adaptive_audit_fidelity.md) · [Experiments](../experiments/08-03_actual_state_adaptive_audit_fidelity.md) · [中文](08-03_actual_state_adaptive_audit_fidelity_ZH.md)

**What is proved here.** Fidelity; the signed telescoping identity for any history-dependent provider policy; a high-probability drift bound and two unconditional expected-value bounds for the adaptive audit, via conditional Hoeffding bounds, a union bound along the run and a martingale decomposition; and the replica count.

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

**Adaptive audit.** A *provider policy* $`S`$ chooses at each round a kernel index $`J_t`$ as a function of the history.

- Let $`\mathcal H_t`$ be the history just before the audit of round $`t`$. Conditional on $`\mathcal H_t`$, freeze the actual state $`Z_t`$, a finite set of $`M_t`$ non-reference candidate kernels, a replica count $`n_t`$ and a confidence allocation $`\delta_t`$, with $`\sum_t\delta_t\le\delta`$ almost surely.
- For each candidate, run $`n_t`$ independent paired experiments from $`Z_t`$. Each applies the candidate update or the reference update, then the **full** reference tail and real deployment. Each arm has its correct marginal law, and the paired contrast lies in $`[-1,1]`$.
- Let $`\widehat A_t(j)`$ be the mean contrast and

```math
e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},\qquad\ell_t(j)=\widehat A_t(j)-e_t,\qquad\ell_t(0)=0 .
```

- Select $`J_t\in\arg\max_j\ell_t(j)`$ and set $`\Gamma_t=\ell_t(J_t)\in[0,1]`$. Then apply a **fresh** update from $`K^{J_t}_t`$, discarding the audit replicas. Unaudited rounds use the reference and set $`\Gamma_t=0`$.
- Audits leave the actual learner and environment state unchanged, and the reference ignores audit records. $`\mathcal F_t\supseteq\mathcal H_t`$ is the history after the audit and the selection, before the fresh update.

**Statement ([Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3)).**

- **(a) Fidelity.** If $`L_E(0)-J_E(\pi_0)\ge\gamma`$ and $`L_E(0)-L_E(S)\le\varepsilon`$, then $`L_E(S)-J_E(\pi_0)\ge\gamma-\varepsilon`$. Here $`J_E(\pi_0)`$ is the value of the initial student; $`\varepsilon`$ can be the certificate of [Theorem 8.2(c)](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2), and $`\gamma`$ the measured gain of reference training from the $`K`$ demonstrations.
- **(b) Signed identity.** For every history-dependent provider policy $`S`$ with the common initial law,

```math
L_E(S)-L_E(0)=\sum_{t=0}^{R-1}\mathbb E_{d^S_t}\,A_t(Z_t,J_t).
```

- **(c) Adaptive audit.** (i) With probability at least $`1-\delta`$, $`\sum_tA_t(Z_t,J_t)\ge\sum_t\Gamma_t`$. (ii) Unconditionally over audits and training,

```math
L_E(\mathrm{adaptive})-L_E(0)\ge-\delta,\qquad L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_t\Gamma_t-2\delta .
```

- **(d) Cost.** Radius $`e_t\le e`$ needs $`n_t\ge\lceil2\log(2M_t/\delta_t)/e^2\rceil`$ paired replicas per candidate. For $`M=4`$, $`\delta=0.05`$, $`e=0.1`$ this is $`\lceil2\log160/0.01\rceil=1016`$. The count is sufficient, not necessary.

*Proof.*

**(a)** $`L_E(S)-J_E(\pi_0)=[L_E(0)-J_E(\pi_0)]-[L_E(0)-L_E(S)]\ge\gamma-\varepsilon`$. Only a one-sided fidelity bound is used, so a candidate that improves on the reference is allowed.

**(b)** Along the $`S`$-process the tower property gives $`\mathbb E[V^0_{t+1}(Z_{t+1})\mid\text{history to }t]=(K^{J_t}_tV^0_{t+1})(Z_t)`$, so $`\mathbb E[V^0_{t+1}(Z_{t+1})-V^0_t(Z_t)]=\mathbb E\,A_t(Z_t,J_t)`$. Summing over $`t`$ telescopes; $`\mathbb EV^0_0(Z_0)=L_E(0)`$ by the common initial law and $`\mathbb EV^0_R(Z_R)=L_E(S)`$ since $`V^0_R=F_E`$.

**(c)** *Audit event.* Conditional on $`\mathcal H_t`$, the $`n_t`$ contrasts of candidate $`j`$ are i.i.d. in $`[-1,1]`$ with mean $`A_t(Z_t,j)`$; this uses the correct marginals, the full tail and exact restoration. Hoeffding's inequality ([Hoeffding, 1963](../references.md#hoeffding1963probability)) for range $`2`$ gives

```math
\Pr\big(\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert>e_t\ \big\vert\ \mathcal H_t\big)\le2\exp(-n_te_t^2/2)=\delta_t/M_t .
```

- A union bound over the $`M_t`$ candidates gives an event $`\mathcal A_t`$ with $`\Pr(\mathcal A_t^c\mid\mathcal H_t)\le\delta_t`$, on which $`\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert\le e_t`$ for all $`j`$. The radius requires $`n_t`$ to be fixed before the outcomes of the current audit are seen.
- *Selected drift.* Write $`A_t:=A_t(Z_t,J_t)`$. If $`J_t=0`$, then $`A_t=0=\Gamma_t`$. Otherwise $`\Gamma_t=\widehat A_t(J_t)-e_t\ge0`$, and on $`\mathcal A_t`$, $`A_t\ge\widehat A_t(J_t)-e_t=\Gamma_t`$. At unaudited rounds $`A_t=\Gamma_t=0`$.
- *Claim (i).* Conditioning successively on $`\mathcal H_t`$, $`\Pr(\bigcup_t\mathcal A_t^c)\le\mathbb E\sum_t\Pr(\mathcal A_t^c\mid\mathcal H_t)\le\mathbb E\sum_t\delta_t\le\delta`$. On $`\bigcap_t\mathcal A_t`$, $`\sum_tA_t\ge\sum_t\Gamma_t`$. This holds at actually reached, random states, because each audit statement is conditional on its own pre-audit history.
- *Drift identity.* The actual update is a fresh draw from $`K^{J_t}_t`$, independent of the audit replicas given $`\mathcal F_t`$, and audits leave the state unchanged. Hence $`A_t=\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]-V^0_t(Z_t)`$. The differences $`D_{t+1}=V^0_{t+1}(Z_{t+1})-\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]`$ are martingale differences for the interleaved audit/update filtration, and telescoping gives

```math
F_E(Z_R)-V^0_0(Z_0)=\sum_{t=0}^{R-1}A_t+\sum_{t=0}^{R-1}D_{t+1}.
```

- *Claim (ii).* Always $`A_t\in[-1,1]`$ and $`\Gamma_t\in[0,1]`$, so $`A_t\ge-\mathbf 1_{\mathcal A_t^c}`$ and $`A_t\ge\Gamma_t-2\cdot\mathbf 1_{\mathcal A_t^c}`$ (on $`\mathcal A_t^c`$, $`A_t\ge-1\ge\Gamma_t-2`$). Taking expectations of the telescoped identity, with $`\mathbb EV^0_0(Z_0)=L_E(0)`$ and $`\mathbb E\sum_tD_{t+1}=0`$,

```math
L_E(\mathrm{adaptive})-L_E(0)=\mathbb E\sum_tA_t\ge-\mathbb E\sum_t\Pr(\mathcal A_t^c\mid\mathcal H_t)\ge-\delta ,
```

and likewise $`\mathbb E\sum_tA_t\ge\mathbb E\sum_t\Gamma_t-2\delta`$.

**(d)** $`e_t\le e`$ is equivalent to $`n_t\ge2\log(2M_t/\delta_t)/e^2`$. For $`M=4`$ and $`\delta=0.05`$, $`2\log160\approx10.150`$; dividing by $`0.01`$ and rounding up gives $`1016`$. Hoeffding's inequality is one sufficient choice; any valid fixed-$`n`$ confidence bound with a smaller radius can replace it. $`\square`$

## References

- [Hoeffding, 1963](../references.md#hoeffding1963probability). Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association 58, no. 301, 13–30.
