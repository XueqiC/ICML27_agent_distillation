# N05 · Continuation-response 刻画 — 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/N05_continuation_response_ZH.md)

## 设定与假设

**假设 5.1（学习过程）。**

- 完整学习器状态 $`z\in\mathcal Z`$ 包括参数、优化器矩、缓冲区、采样器状态、相关随机性和交互历史；没有更小的 Markov 状态时，可以用完整历史。
- $`\mathcal Z`$ 有限，或更一般地是可测空间；下面所有函数都有界且可测。
- 在第 $`t\in\lbrace 0,\dots,R-1\rbrace`$ 轮，反馈分布 $`P_t(\cdot\mid z)`$ 发出消息 $`Y_t`$。实际更新为 $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$，其中 $`\xi_t`$ 是声明的更新随机性。
- 对 $`Y_t`$ 与 $`\xi_t`$ 积分，得到 $`\mathcal Z`$ 上的 Markov kernel $`K_t`$，它作用在有界函数上：$`(K_tf)(z)=\mathbb E[f(U_t(z,Y,\xi))]`$，其中 $`Y\sim P_t(\cdot\mid z)`$，$`\xi`$ 服从声明的分布。
- $`K_t^0`$ 是参照协议，$`K_t^S`$ 是备选协议，两个过程从同一初始分布 $`\nu`$ 出发。
- 每个最终学生都用同一个 $`F_E:\mathcal Z\to[0,1]`$ 打分，即部署策略 $`\pi_z`$ 在真实任务上的回报。

**定义。**

- 参照 continuation：$`V_R^0=F_E`$，$`V_t^0=K_t^0V_{t+1}^0`$ 。
- 消息层面的 response：$`g_{E,f,t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))`$ 。
- 学习价值：$`L_E(S)=\mathbb E\,F_E(Z_R^S)`$，其中 $`Z^S`$ 是从 $`Z_0\sim\nu`$ 出发、由 $`K^S`$ 驱动的链。
- 占据分布：$`d_t^S`$ 是 $`Z_t^S`$ 的分布。

**声明的族。** 一个有限（或可测）的快照集合 $`\mathcal I`$，每个快照 $`(t,z)`$ 带有声明的参照后缀。response 类为 $`\mathcal G=\lbrace g_{E,f,t,z}:(t,z)\in\mathcal I\rbrace`$。

## 陈述

**(a)** **定理（精确复合）。**

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\mathbb E_{d_t^S}\big[(K_t^0-K_t^S)V_{t+1}^0(Z_t)\big]. \qquad(14)
```

因此，若对 $`d_t^S`$-几乎所有 $`z`$ 有 $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$，则

```math
\lvert L_E(0)-L_E(S)\rvert\le\sum_t\mathbb E_{d_t^S}\varepsilon_t(Z_t). \qquad(15)
```

**(b)** **定理（操作性刻画）。** *单边界替换* $`I(t,z,P)`$ 指：在第 $`t`$ 轮从状态 $`z`$ 出发，抽取 $`Y\sim P`$，执行实际更新 $`U_t(z,Y,\xi_t)`$，然后按声明的参照后缀走到第 $`R`$ 轮，并用 $`F_E`$ 打分。两个消息分布 $`P,Q`$ 在所声明族中的每个单边界替换上价值都相等，当且仅当

```math
\mathbb E_Pg=\mathbb E_Qg\quad\text{对每个 }g\in\mathcal G. \qquad(16)
```

**(c)** **定理（停止复合）。** 设 $`C_t\subseteq\mathcal Z`$ 为认证集。令 $`\tau=\min\lbrace t<R:Z_t^S\notin C_t\rbrace`$（若不存在这样的 $`t`$ 则 $`\tau=R`$），$`\rho=\Pr(\tau<R)`$。若对所有 $`z\in C_t`$ 有 $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$，则

```math
\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau}\varepsilon_t(Z_t)+\rho. \qquad(17)
```

**(d)** **命题（导出的 response 类）。** $`\mathcal G`$ 可以递归生成：

- 从 $`\mathcal V_R=\lbrace F_E\rbrace`$ 出发；
- 用参照 kernel 作用，$`\mathcal V_t=K_t^0\mathcal V_{t+1}`$；
- 经实际更新拉回，$`g=\mathbb E_\xi\,v\circ U_t(z,\cdot,\xi)`$，其中 $`v\in\mathcal V_{t+1}`$。

扩大所声明的任务、学习器、后缀或快照族，只会扩大 $`\mathcal G`$，从而只会细化等价关系 (16)。

## 证明

**(a)**

*端点。* 迭代  得 $`V_0^0=K_0^0K_1^0\cdots K_{R-1}^0F_E`$，因此 $`L_E(0)=\nu V_0^0=\mathbb E\,V_0^0(Z_0^S)`$，这里用到了公共初始分布。另一方面，$`L_E(S)=\mathbb EF_E(Z_R^S)=\mathbb EV_R^0(Z_R^S)`$。

*望远镜求和。*

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\Big(\mathbb EV_t^0(Z_t^S)-\mathbb EV_{t+1}^0(Z_{t+1}^S)\Big).
```

*逐项化简。* 由 $`S`$ 链的 Markov 性，$`\mathbb E[V_{t+1}^0(Z_{t+1}^S)\mid Z_t^S]=(K_t^SV_{t+1}^0)(Z_t^S)`$；由 ，$`V_t^0=K_t^0V_{t+1}^0`$。所以每一项都等于 $`\mathbb E[(K_t^0-K_t^S)V_{t+1}^0(Z_t^S)]`$，即 (14)。

*界。* 在和式内用三角不等式即得 (15)。$`\square`$

**(b)**

*替换的价值。* 固定 $`(t,z)\in\mathcal I`$ 和分布 $`P`$。边界之后的状态为 $`Z_{t+1}=U_t(z,Y,\xi_t)`$，其余过程服从参照 kernel。由 Markov 性和 ，给定 $`Z_{t+1}`$ 时终端得分的期望为 $`V_{t+1}^0(Z_{t+1})`$。对 $`\xi_t`$ 和 $`Y\sim P`$ 积分，

```math
\mathrm{val}\,I(t,z,P)=\sum_yP(y)\,\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))=\mathbb E_P\,g_{E,f,t,z}.
```

因此 $`\mathrm{val}\,I(t,z,P)-\mathrm{val}\,I(t,z,Q)=(P-Q)g_{E,f,t,z}`$。

- *充分性。* 若 (16) 成立，每个差都为零。
- *必要性。* 若对某个 $`g=g_{E,f,t,z}\in\mathcal G`$ 有 $`(P-Q)g\ne0`$，则替换 $`I(t,z,\cdot)`$ 本身就是族中价值不等的一员。

对依赖状态的分布 $`P_{t,z},Q_{t,z}`$，同样的论证成立。$`\square`$

**(c) 的证明。** $`\tau`$ 是停时：$`\lbrace \tau>t\rbrace =\lbrace Z_0\in C_0,\dots,Z_t\in C_t\rbrace`$ 关于 $`\sigma(Z_0,\dots,Z_t)`$ 可测。分解

```math
L_E(0)-L_E(S)=\mathbb E\big[V_0^0(Z_0)-V_\tau^0(Z_\tau)\big]+\mathbb E\big[V_\tau^0(Z_\tau)-F_E(Z_R)\big].
```

*第一项。* 把它写成 $`\mathbb E\sum_{t=0}^{R-1}1\lbrace t<\tau\rbrace \big(V_t^0(Z_t)-V_{t+1}^0(Z_{t+1})\big)`$。由于 $`1\lbrace t<\tau\rbrace`$ 关于 $`\sigma(Z_{0:t})`$ 可测，以 $`Z_{0:t}`$ 为条件得

```math
\mathbb E\big[1\lbrace t<\tau\rbrace (V_t^0(Z_t)-V_{t+1}^0(Z_{t+1}))\big]=\mathbb E\big[1\lbrace t<\tau\rbrace (K_t^0-K_t^S)V_{t+1}^0(Z_t)\big].
```

在 $`\lbrace t<\tau\rbrace`$ 上 $`Z_t\in C_t`$，所以其绝对值至多为 $`\mathbb E[1\lbrace t<\tau\rbrace \varepsilon_t(Z_t)]`$。

*第二项。* 在 $`\lbrace \tau=R\rbrace`$ 上，$`V_R^0(Z_R)-F_E(Z_R)=0`$；在 $`\lbrace \tau<R\rbrace`$ 上，$`V_\tau^0(Z_\tau)`$ 与 $`F_E(Z_R)`$ 都在 $`[0,1]`$ 中，其差在 $`[-1,1]`$ 中。所以第二项的绝对值至多为 $`\Pr(\tau<R)=\rho`$。

两个界相加即得 (17)。$`\square`$

**(d) 的证明。** 由 –，每个 $`g\in\mathcal G`$ 都具有所述形式；反过来，每个这种形式的函数都是对应快照的 $`g_{E,f,t,z}`$。若 $`\mathcal I\subseteq\mathcal I'`$，则 $`\mathcal G(\mathcal I)\subseteq\mathcal G(\mathcal I')`$，所以在 $`\mathcal G(\mathcal I')`$ 上满足 (16) 的分布对，在 $`\mathcal G(\mathcal I)`$ 上也满足。$`\square`$

## 注记

1. **核心对象。** $`g`$ 是收到 $`y`$、执行*实际*更新、走完*所声明的参照学习尾段*之后，*最终真实任务回报*的期望。它不是训练损失，不是 teacher 置信度，也不是冻结策略下的动作价值。之后的每个节点（N06–N16）研究的都是 $`\mathcal G`$ 的大小、逼近与测量。
2. **方向与符号。** (14) 中的占据分布是*备选*过程的 $`d_t^S`$。每个漂移项都可正可负。两个接口之间若更新规则不同，这一差异必须计入所比较的 kernel。
3. **对单一最终均值而言，逐点匹配是充分的，但不是必要的。** 不同轮次的误差可以相互抵消。例：两个状态，$`K^0=`$ 恒等，$`K^S=`$ 交换，共两轮，$`F=(0,1)`$，初始状态 $`0`$。最终差距为 $`0`$，但 $`\sum_t\mathbb E\lvert\text{漂移}\rvert=2`$。(b) 对*所有*单边界替换取全称量词，所以在那里匹配每个 $`g`$ 才是必要的。
4. **退出惩罚。** 首次退出之后不受控的尾段，最多贡献一个有界回报的差，所以惩罚是 $`\rho`$ 而不是 $`R\rho`$。若 $`Z_0\notin C_0`$，则 $`\tau=0`$，这一事件计入 $`\rho`$。对连续轨迹而言，仅有有限个存储的 checkpoint 并不能说明 $`\rho`$ 很小；N14 给出定量处理。
5. **尾段必须有响应。** 参照后缀必须对该分支自己当前的学生作出反应。跨分支耦合随机种子是允许的；但如果协议本应查询一个已经改变的学生，却 replay 相同的标签或轨迹，测到的就是另一个 continuation。

## 依赖与来源

- **依赖：** [N04](N04_static_vs_learning_sufficiency_ZH.md)（静态不变量与平均更新不变量为什么不够）。
- **被以下节点使用：** N06（$`\mathcal G`$ 的大小）、N14（带 transport 的有限审计）、N15（在实际状态上的审计）、N16（耦合 response）。

**出处：** [v0.14.1 §5.1](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#51-complete-learning-state-and-continuation-response); [v0.14.1 Theorem 7](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-7-exact-composition-and-operational-necessity); [v0.14.1 §5.2](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#52-the-response-class-is-derived-not-chosen-for-convenience); [v0.15 §1](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#1-what-the-validation-changes) — Thm. 7; eqs. (12)–(17); §5.2; (A1). [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §5，定理 7，式 (12)–(17)，§5.2。审查：[Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) A11–A14。
