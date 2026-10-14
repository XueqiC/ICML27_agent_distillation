# Proposition 3.4 · Joint decomposition with an approximate value

*Proof*

[← Framework](../README.md) · [Statement](../statements/03-04_joint_decomposition_approximate_value.md) · [Proof](03-04_joint_decomposition_approximate_value.md) · [Experiments](../experiments/03-04_joint_decomposition_approximate_value.md) · [中文](03-04_joint_decomposition_approximate_value_ZH.md)

**What is proved here.** Parts (a)–(b) telescope the approximate value along each process and subtract. Part (c) is linearity of the kernel. Part (d) stops each telescoping sum at that process's own exit time and charges the remainder at most once.

**Setting used on this page.**

- Complete learner states $`z\in\mathcal Z`$, learning kernels $`K_t^0`$ (reference) and $`K_t^S`$ (alternative) acting on bounded functions by $`(K_tf)(z)=\mathbb E[f(Z_{t+1})\mid Z_t=z]`$, a common initial law, chains $`Z_t^0`$ and $`Z_t^S`$ with expectations $`\mathbb E_0,\mathbb E_S`$ ([Definition 1.3](../README.md#s1-3)).
- $`F_E\in[0,1]`$ and $`L_E(S)=\mathbb E_SF_E(Z_R)`$ ([Definition 1.4](../README.md#s1-4)).
- Bounded measurable $`\widehat V_0,\dots,\widehat V_R`$; $`r_t^0=K_t^0\widehat V_{t+1}-\widehat V_t`$, $`e_R=F_E-\widehat V_R`$, $`d_t=(K_t^0-K_t^S)\widehat V_{t+1}`$.
- $`d_t^S`$ is the law of $`Z_t^S`$ (the alternative's occupancy at round $`t`$, as in Theorem 3.2); it is a measure on states, distinct from the function $`d_t`$.

**Statement ([Proposition 3.4](../statements/03-04_joint_decomposition_approximate_value.md#s3-4)).** (a) The identity (3.7) holds. (b) If $`\lvert d_t\rvert\le\beta_t`$ $`d_t^S`$-almost everywhere, (3.8) holds. (c) If $`\widehat V_t=c_t^\top\psi_t`$ and $`K_t^0\psi_{t+1}=A_t\psi_t+\epsilon_t`$, then $`r_t^0=(A_t^\top c_{t+1}-c_t)^\top\psi_t+c_{t+1}^\top\epsilon_t`$. (d) If $`\widehat V_t\in[0,1]`$ and $`\lvert d_t\rvert\le\beta_t`$ on $`C_t^S`$, (3.9) holds, with each process stopped at its own first exit $`\tau^j`$ from $`C_t^j`$.

**Proof.**

**Step 1 (one process).** Fix $`j\in\lbrace0,S\rbrace`$. All functions involved are bounded, so every expectation below exists. Write $`\mathcal F_t=\sigma(Z_0^j,\dots,Z_t^j)`$. By the Markov property of the chain driven by $`K^j`$,

```math
\mathbb E\big[\widehat V_{t+1}(Z_{t+1}^j)\mid\mathcal F_t\big]=(K_t^j\widehat V_{t+1})(Z_t^j).
```

Telescoping from $`0`$ to $`R`$ and taking expectations,

```math
\mathbb E_jF_E(Z_R)=\mathbb E\widehat V_0(Z_0)+\sum_{t<R}\mathbb E_j\big[(K_t^j\widehat V_{t+1}-\widehat V_t)(Z_t)\big]+\mathbb E_je_R(Z_R),
```

because $`F_E=\widehat V_R+e_R`$. The first term does not depend on $`j`$: both processes start from the same initial law.

**Step 2 (the two bracketed terms).** For $`j=0`$ the bracket is $`r_t^0`$ by definition. For $`j=S`$,

```math
K_t^S\widehat V_{t+1}-\widehat V_t=(K_t^0\widehat V_{t+1}-\widehat V_t)-(K_t^0-K_t^S)\widehat V_{t+1}=r_t^0-d_t .
```

**Step 3 (part (a)).** Subtract the expansion of Step 1 for $`j=S`$ from the one for $`j=0`$ and insert Step 2:

```math
L_E(0)-L_E(S)=\sum_{t<R}\mathbb E_0r_t^0(Z_t)-\sum_{t<R}\mathbb E_S\big(r_t^0-d_t\big)(Z_t)+\mathbb E_0e_R(Z_R)-\mathbb E_Se_R(Z_R),
```

which is (3.7) after regrouping.

**Step 4 (part (b)).** Apply the triangle inequality to each of the five groups of (3.7), and $`\lvert d_t\rvert\le\beta_t`$ inside $`\mathbb E_S`$, which is legitimate because the bound holds $`d_t^S`$-almost everywhere.

**Step 5 (part (c)).** Each $`K_t^0`$ is linear on bounded functions. Applying it to $`\widehat V_{t+1}=c_{t+1}^\top\psi_{t+1}=\sum_ic_{t+1,i}\psi_{t+1,i}`$ gives $`K_t^0\widehat V_{t+1}=c_{t+1}^\top(A_t\psi_t+\epsilon_t)=(A_t^\top c_{t+1})^\top\psi_t+c_{t+1}^\top\epsilon_t`$. Subtracting $`\widehat V_t=c_t^\top\psi_t`$ gives the claim.

**Step 6 (part (d)).** Fix $`j`$. The exit time $`\tau^j`$ is a stopping time: $`\lbrace\tau^j>t\rbrace=\lbrace Z_0^j\in C_0^j,\dots,Z_t^j\in C_t^j\rbrace\in\mathcal F_t`$. Decompose

```math
\mathbb E_jF_E(Z_R)-\mathbb E\widehat V_0(Z_0)=\mathbb E_j\sum_{t<R}\mathbf 1\lbrace t<\tau^j\rbrace\big(\widehat V_{t+1}(Z_{t+1})-\widehat V_t(Z_t)\big)+\mathbb E_j\big[F_E(Z_R)-\widehat V_{\tau^j}(Z_{\tau^j})\big].
```

- *Stopped sum.* Since $`\mathbf 1\lbrace t<\tau^j\rbrace`$ is $`\mathcal F_t`$-measurable, conditioning as in Step 1 turns the $`t`$-th summand into $`\mathbb E_j[\mathbf 1\lbrace t<\tau^j\rbrace(K_t^j\widehat V_{t+1}-\widehat V_t)(Z_t)]`$, which by Step 2 is $`\mathbb E_0[\mathbf 1\lbrace t<\tau^0\rbrace r_t^0(Z_t)]`$ for $`j=0`$ and $`\mathbb E_S[\mathbf 1\lbrace t<\tau^S\rbrace(r_t^0-d_t)(Z_t)]`$ for $`j=S`$. On $`\lbrace t<\tau^S\rbrace`$ we have $`Z_t^S\in C_t^S`$, so $`\lvert d_t\rvert\le\beta_t`$ there.
- *Remainder.* On $`\lbrace\tau^j=R\rbrace`$ the remainder is $`F_E(Z_R)-\widehat V_R(Z_R)=e_R(Z_R)`$. On $`\lbrace\tau^j<R\rbrace`$ both $`F_E(Z_R)`$ and $`\widehat V_{\tau^j}(Z_{\tau^j})`$ lie in $`[0,1]`$, so the remainder lies in $`[-1,1]`$. Hence the remainder is at most $`\mathbb E_j[\lvert e_R\rvert\mathbf 1\lbrace\tau^j=R\rbrace]+\Pr_j(\tau^j<R)`$ in absolute value.

Subtract the two decompositions; the common term $`\mathbb E\widehat V_0(Z_0)`$ cancels. Bound each of the four resulting pieces (two stopped sums, two remainders) by the triangle inequality. This gives (3.9). With $`C_t^0=\mathcal Z`$ we have $`\tau^0=R`$ almost surely, so the reference exit term is zero. $`\square`$

**Remarks on the proof.**

- Nothing in Steps 1–5 uses Lipschitz constants, contraction, stationarity, small learning rates or differentiability. The only structure used is the Markov property of each chain on the complete state and the common initial law.
- The choice of $`\widehat V`$ is free, but it must be fixed before the comparison. If $`\widehat V`$ is fitted on the same runs whose gap it is used to bound, the residual terms must be estimated on independent runs.

## Related work

- Telescoping an approximate value along the evaluated process and paying Bellman residuals is the simulation-lemma argument of reinforcement learning ([Kakade & Langford, 2002](#ref-kakade2002approximately)); doubly robust off-policy evaluation estimates the same kind of signed decomposition ([Jiang & Li, 2016](#ref-jiang2016doubly)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-jiang2016doubly"></a>

**Nan Jiang, Lihong Li (2016).** Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:652–661. [proceedings.mlr.press/v48/jiang16](https://proceedings.mlr.press/v48/jiang16.html).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).
