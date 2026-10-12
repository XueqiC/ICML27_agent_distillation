# Theorem 3.2 · continuation response 刻画

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/03-02_continuation_response_characterization_ZH.md) · [证明](03-02_continuation_response_characterization_ZH.md) · [实验](../experiments/03-02_continuation_response_characterization_ZH.md) · [English](03-02_continuation_response_characterization.md)

**本页证明什么.** (a) 是对轮次的 telescoping 和，结合备选链的 Markov 性质。(b) 把单边界替换的值算成 response 的期望。(c) 在首次离开认证集合时停止 telescoping，把其余训练至多记为一个有界的回报差。

**本页用到的设定.**

- complete learner state $`z\in\mathcal Z`$：参数、optimizer state、缓冲区、采样器状态、随机性与历史（[Definition 1.3](../README_ZH.md#s1-3)）。
- 第 $`t<R`$ 轮，feedback law $`P_t(\cdot\mid z)`$ 发出消息 $`Y_t`$，实际更新给出 $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$，$`\xi_t`$ 是更新的随机性。对 $`Y_t`$ 与 $`\xi_t`$ 积分得到学习 kernel $`K_t`$，它作用在有界函数上：$`(K_tf)(z)=\mathbb E\,f(U_t(z,Y,\xi))`$。
- $`K_t^0`$ 是参照协议，$`K_t^S`$ 是备选协议 $`S`$，两者从同一初始分布 $`\nu`$ 出发。$`Z_t^S`$ 是由 $`K^S`$ 驱动的链，其分布 $`d_t^S`$ 是第 $`t`$ 轮的占据分布。
- $`F_E(z)\in[0,1]`$ 是从状态 $`z`$ 部署的学生在真实任务上的回报；$`L_E(S)=\mathbb E\,F_E(Z_R^S)`$ 是协议 $`S`$ 的学习价值，$`L_E(0)`$ 是参照的学习价值。
- 参照 continuation value 为 $`V_R^0=F_E`$，$`V_t^0=K_t^0V_{t+1}^0`$：从某个状态出发、按参照协议继续训练时的期望最终得分（[Definition 1.4](../README_ZH.md#s1-4)）。
- 快照 $`(t,z)`$ 处的单边界 response 为 $`g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))`$：收到消息 $`y`$、执行实际更新、再走完声明的参照尾段之后的期望最终回报。response 类 $`\mathcal G`$ 收集声明的快照族 $`\mathcal I`$ 上所有的 $`g_{t,z}`$。
- 单边界替换 $`I(t,z,P)`$：在第 $`t`$ 轮从状态 $`z`$ 出发，抽 $`Y\sim P`$，执行实际更新，走声明的参照尾段，用 $`F_E`$ 打分。
- 认证集合 $`C_t\subseteq\mathcal Z`$、首次离开时刻 $`\tau`$ 与离开概率 $`\rho=\Pr(\tau<R)`$ 在 (c) 中定义。

**陈述（[Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2)）.** 在上述设定下：

**(a) 精确复合.**

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\mathbb E_{d_t^S}\big[(K_t^0-K_t^S)V_{t+1}^0(Z_t)\big]. \qquad(3.2)
```

因此，若对 $`d_t^S`$-几乎所有 $`z`$ 有 $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$，则

```math
\lvert L_E(0)-L_E(S)\rvert\le\sum_t\mathbb E_{d_t^S}\varepsilon_t(Z_t). \qquad(3.3)
```

**(b) 操作性刻画.** 两个消息分布 $`P,Q`$ 在声明族中的每个单边界替换上价值都相等，当且仅当

```math
\mathbb E_Pg=\mathbb E_Qg\quad\text{对每个 }g\in\mathcal G. \qquad(3.4)
```

**(c) 带停止的复合.** 设 $`C_t\subseteq\mathcal Z`$ 为认证集合，$`\tau=\min\lbrace t<R:Z_t^S\notin C_t\rbrace`$（若不存在这样的 $`t`$ 则 $`\tau=R`$），$`\rho=\Pr(\tau<R)`$。若对所有 $`z\in C_t`$ 有 $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$，则

```math
\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau}\varepsilon_t(Z_t)+\rho. \qquad(3.5)
```

**证明.**

**(a)**

- *端点.* 迭代递推 $`V_t^0=K_t^0V_{t+1}^0`$ 得 $`V_0^0=K_0^0K_1^0\cdots K_{R-1}^0F_E`$，再由共同的初始分布得 $`L_E(0)=\nu V_0^0=\mathbb E\,V_0^0(Z_0^S)`$。另有 $`L_E(S)=\mathbb EF_E(Z_R^S)=\mathbb EV_R^0(Z_R^S)`$。
- *telescoping.*

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\Big(\mathbb EV_t^0(Z_t^S)-\mathbb EV_{t+1}^0(Z_{t+1}^S)\Big).
```

- *每一项.* 由 $`S`$ 链的 Markov 性质，$`\mathbb E[V_{t+1}^0(Z_{t+1}^S)\mid Z_t^S]=(K_t^SV_{t+1}^0)(Z_t^S)`$；由定义 $`V_t^0=K_t^0V_{t+1}^0`$。所以每一项都等于 $`\mathbb E[(K_t^0-K_t^S)V_{t+1}^0(Z_t^S)]`$，即 (3.2)。
- *界.* 在和号内用三角不等式即得 (3.3)。

**(b)** 固定 $`(t,z)\in\mathcal I`$ 与分布 $`P`$。边界之后的状态是 $`Z_{t+1}=U_t(z,Y,\xi_t)`$，其余过程按参照 kernel 进行。由 Markov 性质与 $`V^0`$ 的定义，给定 $`Z_{t+1}`$ 时终端得分的期望是 $`V_{t+1}^0(Z_{t+1})`$。对 $`\xi_t`$ 与 $`Y\sim P`$ 积分，

```math
\mathrm{val}\,I(t,z,P)=\sum_yP(y)\,\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))=\mathbb E_P\,g_{t,z}.
```

所以 $`\mathrm{val}\,I(t,z,P)-\mathrm{val}\,I(t,z,Q)=(P-Q)g_{t,z}`$。

- *充分性.* 若 (3.4) 成立，每个差都为零。
- *必要性.* 若某个 $`g=g_{t,z}\in\mathcal G`$ 使 $`(P-Q)g\ne0`$，则替换 $`I(t,z,\cdot)`$ 本身就是族中一个价值不等的成员。

对依赖状态的分布 $`P_{t,z},Q_{t,z}`$，论证相同。

**(c)** $`\tau`$ 是停时：$`\lbrace \tau>t\rbrace =\lbrace Z_0\in C_0,\dots,Z_t\in C_t\rbrace`$ 关于 $`\sigma(Z_0,\dots,Z_t)`$ 可测。分解

```math
L_E(0)-L_E(S)=\mathbb E\big[V_0^0(Z_0)-V_\tau^0(Z_\tau)\big]+\mathbb E\big[V_\tau^0(Z_\tau)-F_E(Z_R)\big].
```

- *第一项.* 写成 $`\mathbb E\sum_{t=0}^{R-1}1\lbrace t<\tau\rbrace \big(V_t^0(Z_t)-V_{t+1}^0(Z_{t+1})\big)`$。$`1\lbrace t<\tau\rbrace`$ 关于 $`\sigma(Z_{0:t})`$ 可测，以 $`Z_{0:t}`$ 为条件得

```math
\mathbb E\big[1\lbrace t<\tau\rbrace (V_t^0(Z_t)-V_{t+1}^0(Z_{t+1}))\big]=\mathbb E\big[1\lbrace t<\tau\rbrace (K_t^0-K_t^S)V_{t+1}^0(Z_t)\big].
```

在 $`\lbrace t<\tau\rbrace`$ 上 $`Z_t\in C_t`$，所以它的绝对值至多为 $`\mathbb E[1\lbrace t<\tau\rbrace \varepsilon_t(Z_t)]`$。

- *第二项.* 在 $`\lbrace \tau=R\rbrace`$ 上 $`V_R^0(Z_R)-F_E(Z_R)=0`$。在 $`\lbrace \tau<R\rbrace`$ 上，$`V_\tau^0(Z_\tau)`$ 与 $`F_E(Z_R)`$ 都在 $`[0,1]`$ 中，差在 $`[-1,1]`$ 中。所以第二项的绝对值至多为 $`\Pr(\tau<R)=\rho`$。

两个界相加即得 (3.5)。$`\square`$

## 相关工作

- 这个 telescoping 恒等式属于强化学习中 performance difference 一类的分解 ([Kakade & Langford, 2002](../references_ZH.md#kakade2002approximately)).

## 参考文献

- [Kakade & Langford, 2002](../references_ZH.md#kakade2002approximately)
