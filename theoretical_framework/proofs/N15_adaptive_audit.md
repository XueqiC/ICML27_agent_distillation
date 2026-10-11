# N15 · Actual-state adaptive auditing; fidelity versus improvement — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N15_adaptive_audit.md) · [中文](N15_adaptive_audit_ZH.md)

## Setting and assumptions

**Learning process.** The training process is the one of N05:

- complete states $`Z_t`$ and a finite horizon $`R`$;
- a fixed reference recipe with kernels $`K^0_t`$ and continuation $`V^0_t\in[0,1]`$, where $`V^0_R=F_E`$;
- $`J_E(\pi_0)`$, the value of the initial student;
- $`L_E(0)=\mathbb E F_E(Z^0_R)`$, the value of reference training.

**Local advantage.** For a candidate kernel $`K^j_t`$, define

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

A **provider policy** $`S`$ chooses at each round a kernel index $`J_t`$, as a function of the history.

**Adaptive audit.**

- Let $`\mathcal H_t`$ be the history just before the audit of round $`t`$. Conditional on $`\mathcal H_t`$, freeze the actual state $`Z_t`$, a finite set of $`M_t`$ non-reference candidate kernels, a replica count $`n_t`$, and a confidence allocation $`\delta_t`$, with $`\sum_t\delta_t\le\delta`$ almost surely.
- For each candidate, run $`n_t`$ independent paired experiments from $`Z_t`$. Each experiment applies the candidate update or the reference update, then the **full** reference-learning suffix and real deployment. Each arm must have its correct marginal law, and the paired contrast lies in $`[-1,1]`$.
- Write $`\widehat A_t(j)`$ for the mean contrast, and set

  ```math
  e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},\qquad\ell_t(j)=\widehat A_t(j)-e_t,\qquad\ell_t(0)=0.
  ```

- Select $`J_t\in\arg\max_j\ell_t(j)`$ and set $`B_t=\ell_t(J_t)\in[0,1]`$. Then apply a **fresh** update from $`K^{J_t}_t`$, discarding the audit-trained replicas.
- Unaudited rounds use the reference and set $`B_t=0`$.
- Audits leave the actual learner and environment state unchanged, and the reference ignores audit records.
- $`\mathcal F_t\supseteq\mathcal H_t`$ is the history after the audit and the selection but before the fresh update.

## Statement

**Theorem (N15).**

1. **(a)** **Fidelity.** If $`L_E(0)-J_E(\pi_0)\ge\gamma`$ and $`L_E(0)-L_E(S)\le\varepsilon`$, then

   ```math
   L_E(S)-J_E(\pi_0)\ge\gamma-\varepsilon. \qquad(41)
   ```

   With $`\varepsilon=\varepsilon_K`$ the certificate [N14(b)](N14_frozen_audit_transport.md) of N14 and $`\gamma=\gamma_K`$ the measured gain of reference training from $`K`$ demonstrations, this is (a).
2. **(b)** **Signed identity.** For every history-dependent provider policy $`S`$ with the common initial law,

   ```math
   L_E(S)-L_E(0)=\sum_{t=0}^{R-1}\mathbb E_{d^S_t}\,A_t(Z_t,J_t).
   ```

3. **(c)** **Adaptive audit.**
   - (a) With probability at least $`1-\delta`$, $`\sum_tA_t(Z_t,J_t)\ge\sum_tB_t`$.
   - (b) Unconditionally over audits and training,

     ```math
     L_E(\mathrm{adaptive})-L_E(0)\ge-\delta, \qquad(44)
     ```

     ```math
     L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_tB_t-2\delta. \qquad(45)
     ```

4. **(d)** **Cost.** To reach radius $`e_t\le e`$ the audit needs $`n_t\ge\lceil2\log(2M_t/\delta_t)/e^2\rceil`$. With $`M=4`$, $`\delta=0.05`$ and $`e=0.1`$ this is $`\lceil 2\log160/0.01\rceil=1016`$ paired replicas per candidate. This count is sufficient, not necessary.

## Proof

**Part 1.** Subtract the two hypotheses: $`L_E(S)-J_E(\pi_0)=[L_E(0)-J_E(\pi_0)]-[L_E(0)-L_E(S)]\ge\gamma-\varepsilon`$. Only a one-sided fidelity bound is used, so a candidate that improves on the reference is allowed.

**Part 2.** Along the $`S`$-process, the tower property over the history gives

```math
\mathbb E[V^0_{t+1}(Z_{t+1})\mid\text{history to }t]=(K^{J_t}_tV^0_{t+1})(Z_t).
```

Hence

```math
\mathbb E\big[V^0_{t+1}(Z_{t+1})-V^0_t(Z_t)\big]=\mathbb E\big[(K^{J_t}_t-K^0_t)V^0_{t+1}(Z_t)\big]=\mathbb E\,A_t(Z_t,J_t).
```

Summing over $`t`$ telescopes. The common initial law gives $`\mathbb E V^0_0(Z_0)=L_E(0)`$, and $`V^0_R=F_E`$ gives $`\mathbb E V^0_R(Z_R)=L_E(S)`$.

**Part 3.**

*Audit event.* Conditional on $`\mathcal H_t`$, the $`n_t`$ contrasts for candidate $`j`$ are i.i.d. in $`[-1,1]`$ with mean $`A_t(Z_t,j)`$. This uses the correct marginals, the full suffix and exact restoration. Hoeffding's inequality for range $`2`$ gives

```math
\Pr\big(\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert>e_t\mid\mathcal H_t\big)\le2\exp(-n_te_t^2/2)=\delta_t/M_t.
```

The union bound over the $`M_t`$ candidates gives an event $`G_t`$ with $`\Pr(G_t^c\mid\mathcal H_t)\le\delta_t`$. On $`G_t`$, $`\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert\le e_t`$ for all $`j`$. The radius requires $`n_t`$ to be fixed before the current audit outcomes are observed.

*Selected drift.* Write $`A_t:=A_t(Z_t,J_t)`$. If $`J_t=0`$, then $`A_t=0=B_t`$. Otherwise $`B_t=\widehat A_t(J_t)-e_t\ge0`$, and on $`G_t`$ we have $`A_t\ge\widehat A_t(J_t)-e_t=B_t`$. So on $`G_t`$, $`A_t\ge B_t\ge0`$. At unaudited rounds, $`A_t=B_t=0`$.

*Claim (a).* Conditioning successively on $`\mathcal H_t`$,

```math
\Pr\Big(\bigcup_tG_t^c\Big)\le\mathbb E\sum_t\Pr(G_t^c\mid\mathcal H_t)\le\mathbb E\sum_t\delta_t\le\delta.
```

On $`\bigcap_tG_t`$, $`\sum_tA_t\ge\sum_tB_t`$. This holds at actually reached, random states, because each audit statement is conditional on its own pre-audit history.

*Drift identity.* Because the actual update is a fresh draw from $`K^{J_t}_t`$, independent of the audit replicas given $`\mathcal F_t`$, and audits leave the state unchanged,

```math
A_t=\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]-V^0_t(Z_t).
```

Let $`D_{t+1}=V^0_{t+1}(Z_{t+1})-\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]`$. These are martingale differences for the interleaved audit/update filtration, so $`\mathbb E\sum_tD_{t+1}=0`$. Telescoping gives

```math
F_E(Z_R)-V^0_0(Z_0)=\sum_{t=0}^{R-1}A_t+\sum_{t=0}^{R-1}D_{t+1}.
```

*Claim (b).* Always $`A_t\in[-1,1]`$ and $`B_t\in[0,1]`$. Hence

```math
A_t\ge-\mathbf 1_{G_t^c},\qquad A_t\ge B_t-2\cdot\mathbf 1_{G_t^c}.
```

The first holds because $`A_t\ge0`$ on $`G_t`$. The second holds because on $`G_t^c`$, $`A_t\ge-1\ge B_t-2`$. Take expectations of the telescoped identity, with $`\mathbb E V^0_0(Z_0)=L_E(0)`$ and $`\mathbb E\sum D=0`$:

```math
L_E(\mathrm{adaptive})-L_E(0)=\mathbb E\sum_tA_t\ge-\mathbb E\sum_t\mathbf 1_{G_t^c}=-\mathbb E\sum_t\Pr(G_t^c\mid\mathcal H_t)\ge-\delta,
```

and similarly $`\mathbb E\sum_tA_t\ge\mathbb E\sum_tB_t-2\delta`$.

**Part 4.** $`e_t\le e`$ is equivalent to $`n_t\ge2\log(2M_t/\delta_t)/e^2`$. For $`M=4`$ and $`\delta=0.05`$, $`2\log160=10.1503`$; dividing by $`e^2=0.01`$ and rounding up gives $`1016`$. Hoeffding's inequality is one sufficient choice. Any valid fixed-$`n`$ confidence bound with a smaller radius can replace it. $`\square`$

## Remarks

1. **The lower confidence bound is necessary.** Selecting by the point estimate $`\widehat A_t(j)`$ can have strictly negative expected gain even with exact full-tail audits; an exact finite witness has gain $`-78/15625`$ (experiments, C16). A slightly inferior candidate can also pass even a lower-confidence-bound gate with positive probability. The $`-\delta`$ term in (44) can therefore not be removed: an exact no-degradation statement ($`\ge0`$) is false in general.
2. **Unconditional, not conditional.** (44)–(45) average over audit outcomes. Conditioning the terminal value on "all audits succeeded" is invalid in general, because future audit events depend on the states reached. A zero-drift witness gives a conditional value of $`119/494`$ against the unconditional $`1/4`$ (experiments, C16).
3. **Stopped refinement.** Let $`\tau`$ be the first failed audit, capped at $`R`$. Then $`L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_{t<\tau}B_t-\delta`$.
4. **What is removed and what is not.** Auditing at the actual state removes the transport/coverage term of N14. It does not remove restoration, full-suffix cost or independence requirements. Reusing a winning audit replica as the update, or auditing with a short tail, changes the estimand. The latter needs an explicit proxy-bias allowance.
5. **Fidelity is not improvement.** (41) inherits only the reference's measured gain $`\gamma`$. A high-fidelity candidate of an ineffective reference does not improve the initial student (experiments, C21).

## Dependencies and source

- Uses [N05](N05_continuation_response.md) (the telescoping identity) and [N14](N14_frozen_audit_transport.md) (the certificate $`\varepsilon_K`$ in (a)).

**Source:** [v0.14.1 §11](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#11-fidelity-improvement-and-adaptive-verification); [v0.15 §5](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#5-how-the-new-pieces-compose) — eqs. (41)–(45); Thm. 20; (A17). [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §11, eqs. (41)–(45), Thm. 20. [Optimizer_State_and_Few_Shot_Theory_v015_EN.md](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md) §5 (A17), §7 (cost). Review: [Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) §C3, §A26.
