# Proposition 4.11 · Static sufficiency is not dynamic sufficiency; finite-horizon closure error

*Proof*

[← Framework](../README.md) · [Statement](../statements/04-11_static_not_dynamic_closure_error.md) · [Proof](04-11_static_not_dynamic_closure_error.md) · [Experiments](../experiments/04-11_static_not_dynamic_closure_error.md) · [中文](04-11_static_not_dynamic_closure_error_ZH.md)

**What is proved here.** (a) is a direct computation on four states. (b) is the error recursion of a linear predictor driven by the exact one-step identity $`q_{t+1}=A_tq_t+\mathbb Er_t(Z_t)`$, followed by Hölder's inequality.

**Setting used on this page.**

- Bounded features $`\psi:\mathcal Z\to\mathbb R^d`$, kernels $`K_t`$ of a fixed control sequence, matrices $`A_t`$, residuals $`r_t=K_t\psi-A_t\psi`$, $`F=c^\top\psi+e_F`$, $`q_t=\mathbb E\psi(Z_t)`$, $`\hat q_{t+1}=A_t\hat q_t`$, $`A_{s:u}=A_s\cdots A_u`$ (empty product $`=I`$) ([Theorem 4.10](../statements/04-10_hankel_rank_predictive_realization.md#s4-10)).

**Statement ([Proposition 4.11](../statements/04-11_static_not_dynamic_closure_error.md#s4-11)).** (a) With $`z=(s,b)\in\lbrace0,1\rbrace^2`$, $`F=s`$ and $`U(s,b)=(b,b)`$, the features $`(\mathbf 1,s)`$ are statically sufficient but not closed; $`(0,0)`$ and $`(0,1)`$ share $`q`$ and current score and have next scores $`0`$ and $`1`$; $`\mathrm{span}\lbrace\mathbf 1,s,b\rbrace`$ is closed. (b) (4.21) holds exactly, and (4.22) holds for every norm and its dual.

**Proof.**

**(a)**

- *Static sufficiency.* $`F=s=(0,1)\cdot(1,s)^\top`$.
- *No closure.* $`(KF)(s,b)=F(b,b)=b`$. Suppose $`b=\alpha+\beta s`$ on all four states. At $`(0,0)`$ and $`(0,1)`$ this gives $`0=\alpha`$ and $`1=\alpha`$, a contradiction. So $`KF\notin\mathrm{span}\lbrace\mathbf 1,s\rbrace`$ and no matrix $`A`$ satisfies $`K\psi=A\psi`$.
- *Same predictive state, different future.* For the point masses at $`(0,0)`$ and $`(0,1)`$, $`q=(1,0)`$ in both cases and $`F=0`$ in both cases. After one update the states are $`(0,0)`$ and $`(1,1)`$, with scores $`0`$ and $`1`$. No function of $`q`$ predicts the next score.
- *Closure after adding $`b`$.* $`K\mathbf 1=\mathbf 1`$, $`Ks=b`$ and $`Kb=b`$, so $`\mathrm{span}\lbrace\mathbf 1,s,b\rbrace`$ is $`K`$-invariant.

**(b)** *One step.* By the Markov property, $`q_{t+1}=\mathbb E\psi(Z_{t+1})=\mathbb E(K_t\psi)(Z_t)`$, and by the definition of $`r_t`$,

```math
q_{t+1}=\mathbb E\big[A_t\psi(Z_t)+r_t(Z_t)\big]=A_tq_t+\mathbb Er_t(Z_t).
```

*Recursion.* With $`\Delta_t=q_t-\hat q_t`$ and $`\hat q_{t+1}=A_t\hat q_t`$, subtraction gives $`\Delta_{t+1}=A_t\Delta_t+\mathbb Er_t(Z_t)`$. Induction on $`H`$: the case $`H=0`$ is trivial, and if (4.21) holds for $`H`$, then

```math
\Delta_{H+1}=A_H\Delta_H+\mathbb Er_H(Z_H)=A_{H:0}\Delta_0+\sum_{t<H}A_{H:t+1}\mathbb Er_t(Z_t)+A_{H:H+1}\mathbb Er_H(Z_H),
```

since $`A_HA_{H-1:t+1}=A_{H:t+1}`$ and $`A_{H:H+1}=I`$. This is (4.21) for $`H+1`$.

*Score error.* $`\mathbb EF(Z_H)=c^\top q_H+\mathbb Ee_F(Z_H)`$, so $`\mathbb EF(Z_H)-c^\top\hat q_H=\mathbb Ee_F(Z_H)+c^\top\Delta_H`$. Insert (4.21) and bound each term by Hölder's inequality, $`\lvert c^\top Mx\rvert=\lvert(M^\top c)^\top x\rvert\le\lVert M^\top c\rVert_*\lVert x\rVert`$, together with $`\lVert\mathbb Er_t(Z_t)\rVert\le\mathbb E\lVert r_t(Z_t)\rVert`$ (Jensen, since norms are convex) and $`\lvert\mathbb Ee_F\rvert\le\mathbb E\lvert e_F\rvert`$. This is (4.22). $`\square`$

**Remark on the proof.** The identity (4.21) holds for any matrices $`A_t`$, including least-squares fits; their fitting error is part of $`r_t`$ by definition. Nothing requires $`\lVert A_t\rVert\le1`$.

## Related work

- The error recursion is the finite-horizon propagation of model error used for approximate predictive states and approximate value-directed compression ([Poupart & Boutilier, 2002](#ref-poupart2002value)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-poupart2002value"></a>

**Pascal Poupart, Craig Boutilier (2002).** Value-directed compression of POMDPs. *Advances in Neural Information Processing Systems 15 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d](https://proceedings.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html).
