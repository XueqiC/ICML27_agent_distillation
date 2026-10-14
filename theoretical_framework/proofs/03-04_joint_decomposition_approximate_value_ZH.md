# Proposition 3.4 · 用近似价值做的联合分解

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/03-04_joint_decomposition_approximate_value_ZH.md) · [证明](03-04_joint_decomposition_approximate_value_ZH.md) · [实验](../experiments/03-04_joint_decomposition_approximate_value_ZH.md) · [English](03-04_joint_decomposition_approximate_value.md)

**本页证明什么.** (a)–(b) 沿每条过程 telescope 近似价值，再相减。(c) 是 kernel 的线性性。(d) 在每条过程各自的离开时刻停止 telescoping，剩余部分至多付一次。

**本页用到的设定.**

- complete learner state $`z\in\mathcal Z`$；学习 kernel $`K_t^0`$（参照）与 $`K_t^S`$（备选）按 $`(K_tf)(z)=\mathbb E[f(Z_{t+1})\mid Z_t=z]`$ 作用在有界函数上；两者有公共初始分布；链 $`Z_t^0`$、$`Z_t^S`$ 的期望记为 $`\mathbb E_0,\mathbb E_S`$（[Definition 1.3](../README_ZH.md#s1-3)）。
- $`F_E\in[0,1]`$，$`L_E(S)=\mathbb E_SF_E(Z_R)`$（[Definition 1.4](../README_ZH.md#s1-4)）。
- 有界可测的 $`\widehat V_0,\dots,\widehat V_R`$；$`r_t^0=K_t^0\widehat V_{t+1}-\widehat V_t`$，$`e_R=F_E-\widehat V_R`$，$`d_t=(K_t^0-K_t^S)\widehat V_{t+1}`$。
- $`d_t^S`$ 是 $`Z_t^S`$ 的分布（备选过程在第 $`t`$ 轮的占据分布，与 Theorem 3.2 相同）；它是状态上的测度，与函数 $`d_t`$ 不同。

**陈述（[Proposition 3.4](../statements/03-04_joint_decomposition_approximate_value_ZH.md#s3-4)）.** (a) 恒等式 (3.7) 成立。(b) 若 $`d_t^S`$-几乎处处有 $`\lvert d_t\rvert\le\beta_t`$，则 (3.8) 成立。(c) 若 $`\widehat V_t=c_t^\top\psi_t`$ 且 $`K_t^0\psi_{t+1}=A_t\psi_t+\epsilon_t`$，则 $`r_t^0=(A_t^\top c_{t+1}-c_t)^\top\psi_t+c_{t+1}^\top\epsilon_t`$。(d) 若 $`\widehat V_t\in[0,1]`$ 且在 $`C_t^S`$ 上有 $`\lvert d_t\rvert\le\beta_t`$，则 (3.9) 成立，其中每条过程在它第一次离开 $`C_t^j`$ 的时刻 $`\tau^j`$ 停止。

**证明.**

**第 1 步（单条过程）.** 固定 $`j\in\lbrace0,S\rbrace`$。所涉函数都有界，下面每个期望都存在。记 $`\mathcal F_t=\sigma(Z_0^j,\dots,Z_t^j)`$。由 $`K^j`$ 驱动的链的 Markov 性质，

```math
\mathbb E\big[\widehat V_{t+1}(Z_{t+1}^j)\mid\mathcal F_t\big]=(K_t^j\widehat V_{t+1})(Z_t^j).
```

从 $`0`$ telescope 到 $`R`$ 并取期望，

```math
\mathbb E_jF_E(Z_R)=\mathbb E\widehat V_0(Z_0)+\sum_{t<R}\mathbb E_j\big[(K_t^j\widehat V_{t+1}-\widehat V_t)(Z_t)\big]+\mathbb E_je_R(Z_R),
```

因为 $`F_E=\widehat V_R+e_R`$。第一项与 $`j`$ 无关：两条过程从同一初始分布出发。

**第 2 步（两个方括号项）.** 对 $`j=0`$，方括号按定义就是 $`r_t^0`$。对 $`j=S`$，

```math
K_t^S\widehat V_{t+1}-\widehat V_t=(K_t^0\widehat V_{t+1}-\widehat V_t)-(K_t^0-K_t^S)\widehat V_{t+1}=r_t^0-d_t .
```

**第 3 步（(a)）.** 用 $`j=0`$ 的展开减去 $`j=S`$ 的展开，代入第 2 步：

```math
L_E(0)-L_E(S)=\sum_{t<R}\mathbb E_0r_t^0(Z_t)-\sum_{t<R}\mathbb E_S\big(r_t^0-d_t\big)(Z_t)+\mathbb E_0e_R(Z_R)-\mathbb E_Se_R(Z_R),
```

重新分组即为 (3.7)。

**第 4 步（(b)）.** 对 (3.7) 的五组项分别用三角不等式，并在 $`\mathbb E_S`$ 内部用 $`\lvert d_t\rvert\le\beta_t`$；这一界 $`d_t^S`$-几乎处处成立，所以可以这样用。

**第 5 步（(c)）.** 每个 $`K_t^0`$ 在有界函数上是线性的。把它作用在 $`\widehat V_{t+1}=c_{t+1}^\top\psi_{t+1}=\sum_ic_{t+1,i}\psi_{t+1,i}`$ 上，得到 $`K_t^0\widehat V_{t+1}=c_{t+1}^\top(A_t\psi_t+\epsilon_t)=(A_t^\top c_{t+1})^\top\psi_t+c_{t+1}^\top\epsilon_t`$。减去 $`\widehat V_t=c_t^\top\psi_t`$ 即得。

**第 6 步（(d)）.** 固定 $`j`$。离开时刻 $`\tau^j`$ 是停时：$`\lbrace\tau^j>t\rbrace=\lbrace Z_0^j\in C_0^j,\dots,Z_t^j\in C_t^j\rbrace\in\mathcal F_t`$。分解

```math
\mathbb E_jF_E(Z_R)-\mathbb E\widehat V_0(Z_0)=\mathbb E_j\sum_{t<R}\mathbf 1\lbrace t<\tau^j\rbrace\big(\widehat V_{t+1}(Z_{t+1})-\widehat V_t(Z_t)\big)+\mathbb E_j\big[F_E(Z_R)-\widehat V_{\tau^j}(Z_{\tau^j})\big].
```

- *停止的和.* $`\mathbf 1\lbrace t<\tau^j\rbrace`$ 关于 $`\mathcal F_t`$ 可测，像第 1 步那样取条件期望，第 $`t`$ 项化为 $`\mathbb E_j[\mathbf 1\lbrace t<\tau^j\rbrace(K_t^j\widehat V_{t+1}-\widehat V_t)(Z_t)]`$；由第 2 步，对 $`j=0`$ 它是 $`\mathbb E_0[\mathbf 1\lbrace t<\tau^0\rbrace r_t^0(Z_t)]`$，对 $`j=S`$ 它是 $`\mathbb E_S[\mathbf 1\lbrace t<\tau^S\rbrace(r_t^0-d_t)(Z_t)]`$。在 $`\lbrace t<\tau^S\rbrace`$ 上有 $`Z_t^S\in C_t^S`$，所以那里 $`\lvert d_t\rvert\le\beta_t`$。
- *剩余部分.* 在 $`\lbrace\tau^j=R\rbrace`$ 上，剩余部分为 $`F_E(Z_R)-\widehat V_R(Z_R)=e_R(Z_R)`$。在 $`\lbrace\tau^j<R\rbrace`$ 上，$`F_E(Z_R)`$ 与 $`\widehat V_{\tau^j}(Z_{\tau^j})`$ 都在 $`[0,1]`$ 中，所以剩余部分落在 $`[-1,1]`$。因此剩余部分的绝对值至多为 $`\mathbb E_j[\lvert e_R\rvert\mathbf 1\lbrace\tau^j=R\rbrace]+\Pr_j(\tau^j<R)`$。

两个分解相减，公共项 $`\mathbb E\widehat V_0(Z_0)`$ 抵消。对得到的四部分（两个停止的和、两个剩余部分）分别用三角不等式，即得 (3.9)。取 $`C_t^0=\mathcal Z`$ 时 $`\tau^0=R`$ 几乎必然成立，参照过程的离开项为零。$`\square`$

**关于证明的说明.**

- 第 1–5 步没有用到 Lipschitz 常数、收缩、平稳性、小学习率或可微性。用到的结构只有每条链在完整状态上的 Markov 性质，以及公共初始分布。
- $`\widehat V`$ 可以任意选，但必须在比较之前固定。若 $`\widehat V`$ 是在同一批运行上拟合的，又要用它去界这批运行的差距，残差项就必须在独立的运行上估计。

## 相关工作

- 沿被评估过程 telescope 一个近似价值、并为 Bellman 残差付账，是强化学习中 simulation lemma 的论证 ([Kakade & Langford, 2002](#ref-kakade2002approximately))；doubly robust off-policy evaluation 估计的也是这类带符号的分解 ([Jiang & Li, 2016](#ref-jiang2016doubly))。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-jiang2016doubly"></a>

**Nan Jiang, Lihong Li (2016).** Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:652–661. [proceedings.mlr.press/v48/jiang16](https://proceedings.mlr.press/v48/jiang16.html).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).
