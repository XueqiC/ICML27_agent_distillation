# Corollary 8.16 · 由复用、估计与编译组合而成的保证

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-16_composed_guarantee_ZH.md) · [证明](08-16_composed_guarantee_ZH.md) · [实验](../experiments/08-16_composed_guarantee_ZH.md) · [English](08-16_composed_guarantee.md)

**本页证明什么.** (a) 把局部 response operator 拆成锚点保留的部分、丢弃的部分与完整漂移，分别界住，再在读出近似价值的测试处取值。(b) 把 (a) 代入带停止的联合分解，并在这里按所用的形式重新推导一遍。(c) 是算术。

**本页用到的设定与记号.**

全文使用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 参考（目标）分布 $`P^0_t(\cdot\mid z)`$ 与编译出的合法分布 $`P^S_t(\cdot\mid z)`$，实际更新 $`U_t`$ 与初始分布相同；kernel 为 $`K^0_t`$、$`K^S_t`$；最终分数 $`L_E(0)`$、$`L_E(S)`$。
- 近似价值 $`\widehat V_t\in[0,1]`$；$`r^0_t=K^0_t\widehat V_{t+1}-\widehat V_t`$；$`e_R=F_E-\widehat V_R`$；$`\bar g_{t,z}(y)=\mathbb E_{\xi_t}\widehat V_{t+1}(U_t(z,y,\xi_t))`$；$`d_t(z)=\int\bar g_{t,z}\,d(P^0_t-P^S_t)(\cdot\mid z)=(K^0_t-K^S_t)\widehat V_{t+1}(z)`$。
- 参考测度 $`\rho`$，$`\mathcal H=L^2_0(\rho)`$，测试 $`\mathcal A`$ 与权重 $`\omega_a>0`$，response operator $`\mathcal R_{t,z}`$；测试 $`a_t`$ 的 response 为 $`\bar g_{t,z}`$。
- 锚点 $`j`$：$`\hat{\mathcal R}_j=\sum_k\hat\sigma_{j,k}\hat u_{j,k}\langle\hat v_{j,k},\cdot\rangle_\rho`$，$`\hat\Pi_j`$ 投影到 $`\hat v_{j,1},\dots,\hat v_{j,r_j}`$ 上，$`C_j\mu=\sum_{k\le r_j}\hat\sigma_{j,k}\mu_k\hat u_{j,k}`$，$`\mu^j_P=(\mathbb E_P\hat v_{j,k})_{k\le r_j}`$。
- 被认证的状态 $`C^S_t`$、条件 1–3 与退出时刻 $`\tau^S`$、$`\tau^0`$ 同陈述页。

**陈述（[Corollary 8.16](../statements/08-16_composed_guarantee_ZH.md#s8-16)）.** 在 $`\mathcal E`$ 上：(a) 对 $`z\in C^S_t`$，$`\lvert d_t(z)\rvert\le\beta_t(z)=\big(\mathfrak r_{t,z}+\eta_{\rm opt}+B_{t,z}(\hat\sigma_{j,r_j+1}+\epsilon_j+\Delta_{t,z})\big)/\sqrt{\omega_{a_t}}`$；(b) $`\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau^S}\beta_t(Z_t)+\mathcal E_{\rm dyn}+\mathcal E_{\rm terminal}+\mathcal E_{\rm exit}`$；(c) 取一个测试、$`B=2`$、$`\epsilon_j=0.578`$、$`\Delta=0.718`$ 时，单步下限 $`B(\epsilon_j+\Delta)`$ 为 2.59，而 $`\lvert L_E(0)-L_E(S)\rvert\le1`$。

*证明.*

**(a)** 固定 $`z\in C^S_t`$，记 $`h=h_{t,z}`$，$`j=j(t,z)`$。由条件 3，$`h\in\mathcal H`$ 且 $`\lVert h\rVert_\rho\le B_{t,z}`$。

*第 1 步：拆分.* 由于 $`\hat{\mathcal R}_j=\hat{\mathcal R}_j\hat\Pi_j+\hat{\mathcal R}_j(I-\hat\Pi_j)`$，

```math
\mathcal R_{t,z}h=\hat{\mathcal R}_j\hat\Pi_jh+\hat{\mathcal R}_j(I-\hat\Pi_j)h+(\mathcal R_{t,z}-\hat{\mathcal R}_j)h .
```

*第 2 步：保留部分是模式均值之差.* $`\hat{\mathcal R}_j\hat\Pi_jh=\sum_{k\le r_j}\hat\sigma_{j,k}\langle\hat v_{j,k},h\rangle_\rho\hat u_{j,k}`$，而 $`\langle\hat v_{j,k},h\rangle_\rho=\int\hat v_{j,k}\,d(P^0_t-P^S_t)=(\mu^j_{P^0})_k-(\mu^j_{P^S})_k`$。所以 $`\hat{\mathcal R}_j\hat\Pi_jh=C_j\mu^j_{P^0}-C_j\mu^j_{P^S}`$。由条件 2，$`\mu^j_{P^0}\in\mathcal M_{t,z}`$，因此

```math
\lVert\hat{\mathcal R}_j\hat\Pi_jh\rVert_\omega\le\sup_{\mu\in\mathcal M_{t,z}}\lVert C_j\mu-C_j\mu^j_{P^S}\rVert_\omega\le\mathfrak r_{t,z}+\eta_{\rm opt}.
```

*第 3 步：丢弃部分.* $`\lVert\hat{\mathcal R}_j(I-\hat\Pi_j)\rVert_{\rm op}=\hat\sigma_{j,r_j+1}`$，所以这一项至多为 $`\hat\sigma_{j,r_j+1}B_{t,z}`$。

*第 4 步：完整漂移.* 由条件 1 与三角不等式，$`\lVert\mathcal R_{t,z}-\hat{\mathcal R}_j\rVert_{\rm op}\le\lVert\mathcal R_{t,z}-\mathcal R_j\rVert_{\rm op}+\lVert\mathcal R_j-\hat{\mathcal R}_j\rVert_{\rm op}\le\Delta_{t,z}+\epsilon_j`$，所以第三项至多为 $`(\Delta_{t,z}+\epsilon_j)B_{t,z}`$。这里用的是整个算子的漂移，因此锚点与 $`z`$ 之间保留模式系数的变化也已付费。

*第 5 步：单个测试.* 把第 2–4 步相加，$`\lVert\mathcal R_{t,z}h\rVert_\omega\le\mathfrak r_{t,z}+\eta_{\rm opt}+B_{t,z}(\hat\sigma_{j,r_j+1}+\epsilon_j+\Delta_{t,z})`$。对测试 $`a_t`$，

```math
(\mathcal R_{t,z}h)(a_t)=\langle\bar g_{t,z}-\mathbb E_\rho\bar g_{t,z},h\rangle_\rho=\int\bar g_{t,z}\,d(P^0_t-P^S_t)=d_t(z),
```

因为 $`h`$ 在 $`\rho`$ 下积分为零。最后 $`\omega_{a_t}\lvert(\mathcal R_{t,z}h)(a_t)\rvert^2\le\lVert\mathcal R_{t,z}h\rVert_\omega^2`$，即得 (a)。

**(b)** 这就是 [Proposition 3.4(d)](../statements/03-04_joint_decomposition_approximate_value_ZH.md#s3-4) 的带停止联合分解；所用的步骤如下。对链 $`j\in\lbrace0,S\rbrace`$ 与它的退出时刻 $`\tau^j`$（是否属于 $`C^j_t`$ 由 $`Z_t`$ 决定，所以它是停时），示性函数 $`\mathbf 1\lbrace t<\tau^j\rbrace`$ 由 $`Z_0,\dots,Z_t`$ 决定。Markov 性质给出

```math
\mathbb E_j\big[\widehat V_{\tau^j}(Z_{\tau^j})\big]-\mathbb E\widehat V_0(Z_0)=\mathbb E_j\sum_{t<\tau^j}\big(K^j_t\widehat V_{t+1}-\widehat V_t\big)(Z_t).
```

在 $`\lbrace\tau^j=R\rbrace`$ 上 $`F_E(Z_R)=\widehat V_R(Z_R)+e_R(Z_R)`$；在 $`\lbrace\tau^j<R\rbrace`$ 上，由于两个值都在 $`[0,1]`$ 中，$`F_E(Z_R)-\widehat V_{\tau^j}(Z_{\tau^j})\in[-1,1]`$。于是

```math
L_E(j)-\mathbb E\widehat V_0=\mathbb E_j\sum_{t<\tau^j}\big(K^j_t\widehat V_{t+1}-\widehat V_t\big)(Z_t)+\mathbb E_j\big[e_R\mathbf 1\lbrace\tau^j=R\rbrace\big]+\mathbb E_j\big[(F_E(Z_R)-\widehat V_{\tau^j}(Z_{\tau^j}))\mathbf 1\lbrace\tau^j<R\rbrace\big].
```

$`j=0`$ 时求和项为 $`r^0_t`$；$`j=S`$ 时为 $`K^S_t\widehat V_{t+1}-\widehat V_t=r^0_t-d_t`$。两式相减，逐项取绝对值，每个退出项用退出概率界住。在 $`\lbrace t<\tau^S\rbrace`$ 上状态是被认证的，所以 (a) 适用于 $`\lvert d_t(Z_t)\rvert`$。由此得

```math
\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau^S}\big(\beta_t+\lvert r^0_t\rvert\big)(Z_t)+\mathbb E_0\sum_{t<\tau^0}\lvert r^0_t(Z_t)\rvert+\mathcal E_{\rm terminal}+\mathcal E_{\rm exit},
```

把残差项归入 $`\mathcal E_{\rm dyn}`$ 就是 (b)。这个不等式在 $`\mathcal E`$ 上成立；在 $`\mathcal E`$ 上，条件 1–3 在每个被认证的状态、对基于审计数据所做的每一种自适应选择都成立。

**(c)** 取 $`\omega_{a_t}=1`$、$`B_{t,z}=2`$，$`B(\epsilon_j+\Delta_{t,z})=2(0.578+0.718)=2.59`$；$`\beta_t`$ 的其余项非负。由于 $`F_E\in[0,1]`$，$`\lvert L_E(0)-L_E(S)\rvert\le1`$。$`\square`$

## 相关工作

- (b) 这一步是带近似价值的 performance-difference 记账（[Kakade & Langford, 2002](#ref-kakade2002approximately)），也是双重稳健离策略评估（[Jiang & Li, 2016](#ref-jiang2016doubly)）背后的结构。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-jiang2016doubly"></a>

**Nan Jiang, Lihong Li (2016).** Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:652–661. [proceedings.mlr.press/v48/jiang16](https://proceedings.mlr.press/v48/jiang16.html).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).
