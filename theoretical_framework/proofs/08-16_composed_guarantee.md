# Corollary 8.16 · Composed guarantee from reuse, estimation and compilation

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-16_composed_guarantee.md) · [Proof](08-16_composed_guarantee.md) · [Experiments](../experiments/08-16_composed_guarantee.md) · [中文](08-16_composed_guarantee_ZH.md)

**What is proved here.** (a) splits the local response operator into the anchor's retained part, its discarded part and the full drift, bounds each, and evaluates the result at the test that reads the approximate value. (b) inserts (a) into the stopped joint decomposition, which is re-derived here in the form used. (c) is arithmetic.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- Reference (target) laws $`P^0_t(\cdot\mid z)`$ and compiled legal laws $`P^S_t(\cdot\mid z)`$ with a common actual update $`U_t`$ and a common initial law; kernels $`K^0_t`$, $`K^S_t`$; final scores $`L_E(0)`$, $`L_E(S)`$.
- Approximate value $`\widehat V_t\in[0,1]`$; $`r^0_t=K^0_t\widehat V_{t+1}-\widehat V_t`$; $`e_R=F_E-\widehat V_R`$; $`\bar g_{t,z}(y)=\mathbb E_{\xi_t}\widehat V_{t+1}(U_t(z,y,\xi_t))`$; $`d_t(z)=\int\bar g_{t,z}\,d(P^0_t-P^S_t)(\cdot\mid z)=(K^0_t-K^S_t)\widehat V_{t+1}(z)`$.
- Reference measure $`\rho`$, $`\mathcal H=L^2_0(\rho)`$, tests $`\mathcal A`$ with weights $`\omega_a>0`$, response operators $`\mathcal R_{t,z}`$; the test $`a_t`$ has response $`\bar g_{t,z}`$.
- Anchor $`j`$: $`\hat{\mathcal R}_j=\sum_k\hat\sigma_{j,k}\hat u_{j,k}\langle\hat v_{j,k},\cdot\rangle_\rho`$, $`\hat\Pi_j`$ the projection onto $`\hat v_{j,1},\dots,\hat v_{j,r_j}`$, $`C_j\mu=\sum_{k\le r_j}\hat\sigma_{j,k}\mu_k\hat u_{j,k}`$, $`\mu^j_P=(\mathbb E_P\hat v_{j,k})_{k\le r_j}`$.
- Certified states $`C^S_t`$, conditions 1–3 and exit times $`\tau^S`$, $`\tau^0`$ as in the statement.

**Statement ([Corollary 8.16](../statements/08-16_composed_guarantee.md#s8-16)).** On $`\mathcal E`$: (a) for $`z\in C^S_t`$, $`\lvert d_t(z)\rvert\le\beta_t(z)=\big(\mathfrak r_{t,z}+\eta_{\rm opt}+B_{t,z}(\hat\sigma_{j,r_j+1}+\epsilon_j+\Delta_{t,z})\big)/\sqrt{\omega_{a_t}}`$; (b) $`\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau^S}\beta_t(Z_t)+\mathcal E_{\rm dyn}+\mathcal E_{\rm terminal}+\mathcal E_{\rm exit}`$; (c) at one test, $`B=2`$, $`\epsilon_j=0.578`$ and $`\Delta=0.718`$ the per-step floor $`B(\epsilon_j+\Delta)`$ is 2.59, while $`\lvert L_E(0)-L_E(S)\rvert\le1`$.

*Proof.*

**(a)** Fix $`z\in C^S_t`$ and write $`h=h_{t,z}`$, $`j=j(t,z)`$. By condition 3, $`h\in\mathcal H`$ and $`\lVert h\rVert_\rho\le B_{t,z}`$.

*Step 1: split.* Since $`\hat{\mathcal R}_j=\hat{\mathcal R}_j\hat\Pi_j+\hat{\mathcal R}_j(I-\hat\Pi_j)`$,

```math
\mathcal R_{t,z}h=\hat{\mathcal R}_j\hat\Pi_jh+\hat{\mathcal R}_j(I-\hat\Pi_j)h+(\mathcal R_{t,z}-\hat{\mathcal R}_j)h .
```

*Step 2: the retained part is a difference of mode means.* $`\hat{\mathcal R}_j\hat\Pi_jh=\sum_{k\le r_j}\hat\sigma_{j,k}\langle\hat v_{j,k},h\rangle_\rho\hat u_{j,k}`$, and $`\langle\hat v_{j,k},h\rangle_\rho=\int\hat v_{j,k}\,d(P^0_t-P^S_t)=(\mu^j_{P^0})_k-(\mu^j_{P^S})_k`$. Hence $`\hat{\mathcal R}_j\hat\Pi_jh=C_j\mu^j_{P^0}-C_j\mu^j_{P^S}`$. By condition 2, $`\mu^j_{P^0}\in\mathcal M_{t,z}`$, so

```math
\lVert\hat{\mathcal R}_j\hat\Pi_jh\rVert_\omega\le\sup_{\mu\in\mathcal M_{t,z}}\lVert C_j\mu-C_j\mu^j_{P^S}\rVert_\omega\le\mathfrak r_{t,z}+\eta_{\rm opt}.
```

*Step 3: the discarded part.* $`\lVert\hat{\mathcal R}_j(I-\hat\Pi_j)\rVert_{\rm op}=\hat\sigma_{j,r_j+1}`$, so this term is at most $`\hat\sigma_{j,r_j+1}B_{t,z}`$.

*Step 4: the full drift.* By condition 1 and the triangle inequality, $`\lVert\mathcal R_{t,z}-\hat{\mathcal R}_j\rVert_{\rm op}\le\lVert\mathcal R_{t,z}-\mathcal R_j\rVert_{\rm op}+\lVert\mathcal R_j-\hat{\mathcal R}_j\rVert_{\rm op}\le\Delta_{t,z}+\epsilon_j`$, so the third term is at most $`(\Delta_{t,z}+\epsilon_j)B_{t,z}`$. This uses the drift of the whole operator; it therefore also pays for changes of the retained modes' coefficients between the anchor and $`z`$.

*Step 5: one test.* Adding Steps 2–4, $`\lVert\mathcal R_{t,z}h\rVert_\omega\le\mathfrak r_{t,z}+\eta_{\rm opt}+B_{t,z}(\hat\sigma_{j,r_j+1}+\epsilon_j+\Delta_{t,z})`$. For the test $`a_t`$,

```math
(\mathcal R_{t,z}h)(a_t)=\langle\bar g_{t,z}-\mathbb E_\rho\bar g_{t,z},h\rangle_\rho=\int\bar g_{t,z}\,d(P^0_t-P^S_t)=d_t(z),
```

because $`h`$ integrates to zero under $`\rho`$. Finally $`\omega_{a_t}\lvert(\mathcal R_{t,z}h)(a_t)\rvert^2\le\lVert\mathcal R_{t,z}h\rVert_\omega^2`$, which gives (a).

**(b)** This is the stopped joint decomposition of [Proposition 3.4(d)](../statements/03-04_joint_decomposition_approximate_value.md#s3-4); the steps used are as follows. For a chain $`j\in\lbrace0,S\rbrace`$ with exit time $`\tau^j`$ (a stopping time, since membership in $`C^j_t`$ is decided by $`Z_t`$), the indicator $`\mathbf 1\lbrace t<\tau^j\rbrace`$ is determined by $`Z_0,\dots,Z_t`$. The Markov property then gives

```math
\mathbb E_j\big[\widehat V_{\tau^j}(Z_{\tau^j})\big]-\mathbb E\widehat V_0(Z_0)=\mathbb E_j\sum_{t<\tau^j}\big(K^j_t\widehat V_{t+1}-\widehat V_t\big)(Z_t).
```

On $`\lbrace\tau^j=R\rbrace`$, $`F_E(Z_R)=\widehat V_R(Z_R)+e_R(Z_R)`$; on $`\lbrace\tau^j<R\rbrace`$, $`F_E(Z_R)-\widehat V_{\tau^j}(Z_{\tau^j})\in[-1,1]`$ because both values lie in $`[0,1]`$. Hence

```math
L_E(j)-\mathbb E\widehat V_0=\mathbb E_j\sum_{t<\tau^j}\big(K^j_t\widehat V_{t+1}-\widehat V_t\big)(Z_t)+\mathbb E_j\big[e_R\mathbf 1\lbrace\tau^j=R\rbrace\big]+\mathbb E_j\big[(F_E(Z_R)-\widehat V_{\tau^j}(Z_{\tau^j}))\mathbf 1\lbrace\tau^j<R\rbrace\big].
```

For $`j=0`$ the summand is $`r^0_t`$; for $`j=S`$ it is $`K^S_t\widehat V_{t+1}-\widehat V_t=r^0_t-d_t`$. Subtract the two identities, take absolute values term by term, and bound each exit term by the exit probability. On $`\lbrace t<\tau^S\rbrace`$ the state is certified, so (a) applies to $`\lvert d_t(Z_t)\rvert`$. This yields

```math
\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau^S}\big(\beta_t+\lvert r^0_t\rvert\big)(Z_t)+\mathbb E_0\sum_{t<\tau^0}\lvert r^0_t(Z_t)\rvert+\mathcal E_{\rm terminal}+\mathcal E_{\rm exit},
```

which is (b) after regrouping the residual terms into $`\mathcal E_{\rm dyn}`$. The inequality holds on $`\mathcal E`$, where conditions 1–3 hold at every certified state and for every adaptive choice made on the audit data.

**(c)** With $`\omega_{a_t}=1`$ and $`B_{t,z}=2`$, the term $`B(\epsilon_j+\Delta_{t,z})`$ equals $`2(0.578+0.718)=2.59`$; the remaining terms of $`\beta_t`$ are nonnegative. Since $`F_E\in[0,1]`$, $`\lvert L_E(0)-L_E(S)\rvert\le1`$. $`\square`$

## Related work

- Step (b) is performance-difference bookkeeping with an approximate value ([Kakade & Langford, 2002](#ref-kakade2002approximately)), the same structure that underlies doubly robust off-policy evaluation ([Jiang & Li, 2016](#ref-jiang2016doubly)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-jiang2016doubly"></a>

**Nan Jiang, Lihong Li (2016).** Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:652–661. [proceedings.mlr.press/v48/jiang16](https://proceedings.mlr.press/v48/jiang16.html).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).
