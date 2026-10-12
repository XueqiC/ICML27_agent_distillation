# §3 学习后果 — 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/03_learning_consequence_ZH.md) · [English](03_learning_consequence.md)

**概要.**

- [Proposition 3.1](#s3-1) 说明静态准则决定不了学生学到什么。
  - 两个数据来源可以有同样好的 teacher，并让类中每个学生得到相同的策略值，却蒸馏出不同的学生。
  - 两个标签包可以有相同的边缘分布和相同的平均梯度，学习价值却不同；换一个任务，差的符号反过来。
- [Theorem 3.2](#s3-2) 给出不变量。两个训练过程的差，精确等于逐轮单边界替换效应之和，每一项都由 continuation response 决定。两个消息分布在每个单边界替换上可互换，当且仅当它们在每个 response 上的期望相同。当保证只在部分状态空间上成立时，离开这部分的代价只付一次。
- [Remark 3.3](#s3-3) 指出 response 类由学习器、参照尾段和任务推导而来，不是挑出来的。
- **用到.** [Definitions 1.3–1.4](../README_ZH.md#s1-3) 的设定（complete learner state、feedback law、学习 kernel、continuation value、response）；[Theorem 2.2](02_feasibility_and_acquisition_ZH.md#s2-2) 的分解（Proposition 3.1 的差距来自学习器提取了什么，而不是表示能力）。
- **被用于.** [Theorem 4.2](04_complexity_of_sufficiency_ZH.md#s4-2)（response 类的大小）、[Proposition 5.1](05_optimizer_state_ZH.md#s5-1)（它的一阶展开）、[Theorem 8.2](08_verification_ZH.md#s8-2) 与 [Theorem 8.3](08_verification_ZH.md#s8-3)（有限 audit 与实际状态上的 audit）、[Proposition 9.1](09_mechanism_and_method_ZH.md#s9-1)（耦合 response）、[Proposition 7.3(a)](07_second_moment_channel_ZH.md#s7-3)（与任务无关的统计量）以及方法（[Remark 9.3](09_mechanism_and_method_ZH.md#s9-3)）。

## 静态评估决定不了学习结果

<a id="s3-1"></a>

**Proposition 3.1（静态价值不等于学习价值）.**

*构造 (a)：四状态来源.*

- 状态 $`u,v,w,x`$ 共用下面的奖励规则；动作 $`B`$ 的成功概率总是 $`\tfrac12`$。
- 参照来源按 $`\rho_0=(\tfrac34,\tfrac14,0,0)`$ 抽取状态，备选来源按 $`\rho_1=(0,0,\tfrac14,\tfrac34)`$ 抽取。
- 最优 teacher 能看到状态，选其中较好的动作。
- 学生是混叠的：只有一个上下文，以概率 $`p`$ 选 $`A`$，记策略为 $`\pi_p`$。
- 学习器是频率蒸馏：由 $`m\ge1`$ 条独立的 teacher 示范，令 $`\widehat p`$ 为其中 $`A`$ 动作的比例（这也是经验交叉熵的最大化者）。

| 状态 | $`\rho_0`$ | $`\rho_1`$ | $`r(A)`$ | $`r(B)`$ |
|---|---:|---:|---:|---:|
| $`u`$ | $`3/4`$ | $`0`$ | $`19/30`$ | $`1/2`$ |
| $`v`$ | $`1/4`$ | $`0`$ | $`3/10`$ | $`1/2`$ |
| $`w`$ | $`0`$ | $`1/4`$ | $`9/10`$ | $`1/2`$ |
| $`x`$ | $`0`$ | $`3/4`$ | $`13/30`$ | $`1/2`$ |

*构造 (b)：两个标签包.* 一个包是一对二元标签 $`(Y_1,Y_2)`$。

- *独立*来源下，$`Y_1,Y_2`$ 独立同分布于 Bernoulli $`(\tfrac12)`$；*配对*来源下，$`(Y_1,Y_2)`$ 在 $`\lbrace (0,1),(1,0)\rbrace`$ 上均匀。
- 学习器从 $`p_0=\tfrac12`$ 出发，在平方损失 $`\tfrac14\sum_{i=1}^2(p-Y_i)^2`$ 上走一步步长为 $`0<\eta\le1`$ 的梯度：

```math
p^+=p_0-\eta\big(p_0-\tfrac{Y_1+Y_2}2\big)=\frac{1-\eta}2+\eta\,\frac{Y_1+Y_2}2 .
```

- 部署是两次独立决策，两次都必须选动作 1，所以 $`J(p)=p^2`$。备选任务为 $`J'(p)=1-p^2`$。

则：

- **(a) 静态值相同，学生不同.** 在构造 (a) 中，两个来源让每个混叠学生得到相同的值 $`J(\pi_p)=\tfrac12+p/20`$，也让 teacher 得到相同的值 $`\tfrac35`$。teacher 的 $`A`$ 频率分别为 $`\tfrac34`$ 与 $`\tfrac14`$，对 $`m\ge1`$ 条示范做频率蒸馏，期望回报分别为 $`\tfrac{43}{80}`$（参照）与 $`\tfrac{41}{80}`$（备选）：对每个 $`m`$，差距都是 $`\tfrac1{40}`$。
- **(b) 平均更新相同，学习价值不同.** 在构造 (b) 中，两个来源的标签边缘分布、平均梯度和平均更新 $`\mathbb Ep^+=\tfrac12`$ 都相同，但

```math
\mathbb E_{\rm indep}J(p^+)-\mathbb E_{\rm paired}J(p^+)=\frac{\eta^2}8, \qquad(3.1)
```

而在 $`J'=1-p^2`$ 下，这个差为 $`-\eta^2/8`$。

*Proof.* **(a)**

- *静态值.* 状态分布为 $`\rho`$ 时，

```math
J_\rho(\pi_p)=\sum_s\rho(s)\big[p\,r_s(A)+(1-p)\tfrac12\big]=\tfrac12+p\Big(\sum_s\rho(s)r_s(A)-\tfrac12\Big).
```

- 对 $`\rho_0`$，$`\sum_s\rho_0(s)r_s(A)=\tfrac34\cdot\tfrac{19}{30}+\tfrac14\cdot\tfrac3{10}=\tfrac{57}{120}+\tfrac9{120}=\tfrac{11}{20}`$；对 $`\rho_1`$，$`\tfrac14\cdot\tfrac9{10}+\tfrac34\cdot\tfrac{13}{30}=\tfrac9{40}+\tfrac{13}{40}=\tfrac{11}{20}`$。所以对所有 $`p\in[0,1]`$ 有 $`J_{\rho_0}(\pi_p)=J_{\rho_1}(\pi_p)=\tfrac12+\tfrac p{20}`$。
- *teacher.* teacher 在 $`r(A)>\tfrac12`$ 处选 $`A`$，即在 $`u`$（$`\tfrac{19}{30}`$）与 $`w`$（$`\tfrac9{10}`$）；在 $`v`$（$`\tfrac3{10}`$）与 $`x`$（$`\tfrac{13}{30}`$）处选 $`B`$。它的值在 $`\rho_0`$ 下为 $`\tfrac34\cdot\tfrac{19}{30}+\tfrac14\cdot\tfrac12=\tfrac35`$，在 $`\rho_1`$ 下为 $`\tfrac14\cdot\tfrac9{10}+\tfrac34\cdot\tfrac12=\tfrac35`$。它的 $`A`$ 频率在 $`\rho_0`$ 下为 $`\rho_0(u)=\tfrac34`$，在 $`\rho_1`$ 下为 $`\rho_1(w)=\tfrac14`$。
- *学到的学生.* $`m\widehat p`$ 服从 Binomial $`(m,f)`$，$`f\in\lbrace \tfrac34,\tfrac14\rbrace`$，所以对每个 $`m\ge1`$ 有 $`\mathbb E\widehat p=f`$。值关于 $`p`$ 是仿射的，且在两个来源下是同一个函数，所以

```math
\mathbb EJ(\pi_{\widehat p})=\tfrac12+\tfrac{\mathbb E\widehat p}{20}=\tfrac12+\tfrac3{80}=\tfrac{43}{80}\quad\text{或}\quad\tfrac12+\tfrac1{80}=\tfrac{41}{80}.
```

**(b)**

- *一阶矩相同.* 两个来源下每个 $`Y_i`$ 都是 Bernoulli $`(\tfrac12)`$，边缘分布相同。$`p_0`$ 处的梯度为 $`p_0-(Y_1+Y_2)/2`$，在两个来源下均值都为 $`0`$。所以两者都有 $`\mathbb Ep^+=\tfrac12`$。
- *方差.* $`\mathrm{Var}\,p^+=\eta^2\mathrm{Var}\frac{Y_1+Y_2}2`$。独立时为 $`\eta^2\cdot\tfrac14(\tfrac14+\tfrac14)=\eta^2/8`$；配对时 $`Y_1+Y_2\equiv1`$，方差为 $`0`$。
- *值.* $`J(p)=p^2`$ 时 $`\mathbb EJ(p^+)=(\mathbb Ep^+)^2+\mathrm{Var}\,p^+`$，所以差为 $`\eta^2/8-0=\eta^2/8`$。$`J'=1-p^2`$ 时 $`\mathbb EJ'(p^+)=1-(\mathbb Ep^+)^2-\mathrm{Var}\,p^+`$，差为 $`-\eta^2/8`$。$`\square`$

**关于 Proposition 3.1 的说明.**

- **哪些相同，哪些不同.**
  - (a) 中，静态评估在整个受限策略类 $`\Pi=\lbrace \pi_p\rbrace`$ 上以及在 teacher 上都一致。但学习消耗的是 teacher 的*动作频率*，而值并不决定动作频率。
  - (b) 中，标签边缘分布与平均梯度一致。但部署目标的曲率读取的是包的*二阶矩*。
  - 两种情形中，决定结果的都是"实际更新与部署目标的复合"。Theorem 3.2 把这个复合作为基本对象。
- **fidelity 不等于改进.** (3.1) 的符号取决于任务：同一个方差在 $`p^2`$ 下有利，在 $`1-p^2`$ 下有害。梯度方差更低、fidelity 更高、任务表现更好，是三个不同的性质。
- **范围.** (a) 针对所述的混叠类，与在无限制策略和值测试上的完整 Bellman 值等价并不矛盾。差距是学习算法特有的：一个忽略两个数据集、直接输出 $`p=1`$ 的学习器没有差距。这正是要点：充分性必须相对于声明的学习器来定义。

## Continuation response

*设定*（取自 [Definitions 1.3–1.4](../README_ZH.md#s1-3)）.

- complete learner state $`z\in\mathcal Z`$ 包括参数、optimizer state、缓冲区、采样器状态、相关的随机性和交互历史；没有更小的 Markov 状态时，可以用完整历史。$`\mathcal Z`$ 有限，或更一般地是可测空间；下面所有函数都有界且可测。
- 第 $`t\in\lbrace 0,\dots,R-1\rbrace`$ 轮，feedback law $`P_t(\cdot\mid z)`$ 发出消息 $`Y_t`$，实际更新为 $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$，其中 $`\xi_t`$ 是声明的更新随机性。
- 对 $`Y_t`$ 与 $`\xi_t`$ 积分，得到 $`\mathcal Z`$ 上的 Markov kernel $`K_t`$，它作用在有界函数上：$`(K_tf)(z)=\mathbb E[f(U_t(z,Y,\xi))]`$，其中 $`Y\sim P_t(\cdot\mid z)`$，$`\xi`$ 服从声明的分布。$`K_t^0`$ 是参照协议，$`K_t^S`$ 是备选协议 $`S`$。两个过程从同一初始分布 $`\nu`$ 出发。
- 每个最终学生都用同一个 $`F_E:\mathcal Z\to[0,1]`$ 打分，即部署策略 $`\pi_z`$ 在真实任务上的回报。
- 参照 continuation value 为 $`V_R^0=F_E`$，$`V_t^0=K_t^0V_{t+1}^0`$。快照 $`(t,z)`$ 处的单边界 response 为

```math
g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0\big(U_t(z,y,\xi_t)\big).
```

- 协议的学习价值为 $`L_E(S)=\mathbb E\,F_E(Z_R^S)`$，其中 $`Z^S`$ 是从 $`Z_0\sim\nu`$ 出发、由 $`K^S`$ 驱动的链；$`L_E(0)`$ 是参照。占据分布 $`d_t^S`$ 是 $`Z_t^S`$ 的分布。
- 声明的族是快照 $`(t,z)`$ 的有限（或可测）集合 $`\mathcal I`$，每个快照带有声明的参照后缀。response 类为 $`\mathcal G=\lbrace g_{t,z}:(t,z)\in\mathcal I\rbrace`$。
- *单边界替换* $`I(t,z,P)`$ 指：在第 $`t`$ 轮从状态 $`z`$ 出发，抽取 $`Y\sim P`$，执行实际更新 $`U_t(z,Y,\xi_t)`$，按声明的参照后缀走到第 $`R`$ 轮，并用 $`F_E`$ 打分。

<a id="s3-2"></a>

**Theorem 3.2（continuation response 刻画）.** 在上述设定下：

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

*Proof.* **(a)**

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

**关于 Theorem 3.2 的说明.**

- **对象.** $`g_{t,z}(y)`$ 是收到 $`y`$、执行*实际*更新、再走完*声明的参照学习尾段*之后，*最终真实任务回报*的期望。它不是训练损失，不是 teacher 置信度，也不是冻结策略下的动作值。§4 研究 $`\mathcal G`$ 的大小与近似，§5–§7 研究它对 optimizer 的依赖，§8 研究如何测量它。
- **方向与符号.** (3.2) 中的占据分布是*备选*过程的 $`d_t^S`$。每个漂移项可正可负。两个协议在更新规则上的任何差别都必须包含在比较的 kernel 中。
- **对单个最终均值而言，逐点匹配是充分的，不是必要的.** 不同轮的误差可以相互抵消。取两个状态，$`K^0`$ 为恒等，$`K^S`$ 为交换，两轮，$`F=(0,1)`$，初始状态为 $`0`$：最终差距为 $`0`$，而 $`\sum_t\mathbb E\lvert\text{漂移}\rvert=2`$。(b) 对*所有*单边界替换取全称量词，所以那里匹配每个 $`g`$ 变成必要的。
- **离开的代价.** 第一次离开之后，不受控的尾段至多贡献一个有界回报的差，所以代价是 $`\rho`$ 而不是 $`R\rho`$。若 $`Z_0\notin C_0`$，则 $`\tau=0`$，这个事件计入 $`\rho`$。有限个存储的检查点本身并不能保证连续轨迹的 $`\rho`$ 很小；[Theorem 8.2](08_verification_ZH.md#s8-2) 把 coverage 定量化。
- **会响应的尾段.** 参照后缀必须对该分支自己当前的学生作出反应。在分支之间耦合随机种子是合法的；在协议本该查询已改变的学生之处重放相同的标签或轨迹，测到的是另一个 continuation。

<a id="s3-3"></a>

**Remark 3.3（response 类是推导出来的，不是挑出来的）.** 类 $`\mathcal G`$ 由声明的对象递归生成：

- 从 $`\mathcal V_R=\lbrace F_E\rbrace`$ 出发；
- 作用参照 kernel，$`\mathcal V_t=K_t^0\mathcal V_{t+1}`$；
- 经实际更新拉回，$`g=\mathbb E_\xi\,v\circ U_t(z,\cdot,\xi)`$，其中 $`v\in\mathcal V_{t+1}`$。

由 $`V^0`$ 与 $`g_{t,z}`$ 的定义，每个 $`g\in\mathcal G`$ 都有这种形式；反过来，每个这种形式的函数都是对应快照的 response。若 $`\mathcal I\subseteq\mathcal I'`$，则 $`\mathcal G(\mathcal I)\subseteq\mathcal G(\mathcal I')`$，所以对 $`\mathcal G(\mathcal I')`$ 满足 (3.4) 的每一对分布也对 $`\mathcal G(\mathcal I)`$ 满足。扩大声明的任务、学习器、后缀或快照族，只会让等价关系 (3.4) 更细。response 类由学习器与任务确定，不是为了建模方便的选择。

## 相关工作

- telescoping 恒等式 (3.2) 是强化学习与 Markov 链分析中 performance-difference 式的分解：两个 kernel 下终端期望之差，写成单步 kernel 差作用在 continuation value 上之和。
- [Proposition 5.1](05_optimizer_state_ZH.md#s5-1) 中 response 的一阶展开把这个对象与 influence function、训练数据归因联系起来；response 本身的定义不依赖任何展开。
