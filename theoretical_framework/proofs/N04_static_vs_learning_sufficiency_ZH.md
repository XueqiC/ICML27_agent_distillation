# N04 · 静态价值不等于学习价值 — 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/N04_static_vs_learning_sufficiency_ZH.md)

## 设定与假设

**构造 4.1（命题 5：四状态数据源）。**

- 状态 $u,v,w,x$ 共用下表的奖励规则；动作 $B$ 的成功概率恒为 $\tfrac12$。
- 参照数据源按 $\rho_0=(\tfrac34,\tfrac14,0,0)$ 抽取状态，另一数据源按 $\rho_1=(0,0,\tfrac14,\tfrac34)$ 抽取。
- 最优 teacher 能看到状态，并选择更好的动作。
- 学生是 aliased 的：只有一个 context，以概率 $p$ 选 $A$，对应策略 $\pi_p$。
- 学习器为频率蒸馏：从 $m\ge1$ 条独立的 teacher 示范中，取 $\widehat p$ 为 $A$ 动作所占的比例，这也是经验 cross-entropy 的最大化者。

| 状态 | $\rho_0$ | $\rho_1$ | $r(A)$ | $r(B)$ |
|---|---:|---:|---:|---:|
| $u$ | $3/4$ | $0$ | $19/30$ | $1/2$ |
| $v$ | $1/4$ | $0$ | $3/10$ | $1/2$ |
| $w$ | $0$ | $1/4$ | $9/10$ | $1/2$ |
| $x$ | $0$ | $3/4$ | $13/30$ | $1/2$ |

**构造 4.2（命题 6：两种标签 packet）。**

- 一个 packet 是一对二元标签 $(Y_1,Y_2)$。
- *独立*数据源下，$Y_1,Y_2$ 独立同分布于 Bernoulli$(\tfrac12)$；*配对*数据源下，$(Y_1,Y_2)$ 在 $\lbrace (0,1),(1,0)\rbrace $ 上均匀分布。
- 学习器从 $p_0=\tfrac12$ 出发，对平方损失 $\tfrac14\sum_{i=1}^2(p-Y_i)^2$ 做一步步长为 $0<\eta\le1$ 的梯度下降：
$$p^+=p_0-\eta\big(p_0-\tfrac{Y_1+Y_2}2\big)=\frac{1-\eta}2+\eta\,\frac{Y_1+Y_2}2.$$
- 部署包含两个独立决策，都必须选动作 1，即 $J(p)=p^2$；另一个任务为 $J'(p)=1-p^2$。

## 陈述

**命题 4.1（静态价值相同，学到的学生不同）。** 在构造 4.1 中：

- 两个数据源对每个 aliased 学生给出相同的价值 $J(\pi_p)=\tfrac12+p/20$；
- 两者给出的 teacher 价值都是 $\tfrac35$；
- teacher 的 $A$ 频率分别为 $\tfrac34$ 和 $\tfrac14$；
- 在 $m\ge1$ 条示范上做频率蒸馏，得到的期望回报为
$$\mathbb E\,J(\pi_{\widehat p})=\tfrac{43}{80}\quad\text{（参照）},\qquad\tfrac{41}{80}\quad\text{（另一数据源）},\tag{10}$$
对每个 $m$，差距都是 $\tfrac1{40}$。

**命题 4.2（平均更新相同，学习价值不同）。** 在构造 4.2 中：

- 两个数据源的标签边缘分布、平均梯度和平均更新 $\mathbb Ep^+=\tfrac12$ 都相同；
- 然而
$$\mathbb E_{\rm indep}J(p^+)-\mathbb E_{\rm paired}J(p^+)=\frac{\eta^2}8,\tag{11}$$
- 对 $J'=1-p^2$，差为 $-\eta^2/8$。

## 证明

**命题 4.1 的证明。**

*静态价值。* 在状态分布 $\rho$ 下，$\pi_p$ 的价值为
$$J_\rho(\pi_p)=\sum_s\rho(s)\big[p\,r_s(A)+(1-p)\tfrac12\big]=\tfrac12+p\Big(\sum_s\rho(s)r_s(A)-\tfrac12\Big).$$
对 $\rho_0$，$\sum_s\rho_0(s)r_s(A)=\tfrac34\cdot\tfrac{19}{30}+\tfrac14\cdot\tfrac3{10}=\tfrac{57}{120}+\tfrac9{120}=\tfrac{11}{20}$；对 $\rho_1$，$\tfrac14\cdot\tfrac9{10}+\tfrac34\cdot\tfrac{13}{30}=\tfrac9{40}+\tfrac{13}{40}=\tfrac{11}{20}$。因此对所有 $p\in[0,1]$，$J_{\rho_0}(\pi_p)=J_{\rho_1}(\pi_p)=\tfrac12+\tfrac p{20}$。

*teacher。* teacher 在 $r(A)>\tfrac12$ 的状态选 $A$，即 $u$（$\tfrac{19}{30}$）与 $w$（$\tfrac9{10}$）；在 $v$（$\tfrac3{10}$）与 $x$（$\tfrac{13}{30}$）选 $B$。其价值为
$$\tfrac34\cdot\tfrac{19}{30}+\tfrac14\cdot\tfrac12=\tfrac35\quad\text{（在 }\rho_0\text{ 下）},\qquad\tfrac14\cdot\tfrac9{10}+\tfrac34\cdot\tfrac12=\tfrac35\quad\text{（在 }\rho_1\text{ 下）}.$$
它的 $A$ 频率在 $\rho_0$ 下为 $\rho_0(u)=\tfrac34$，在 $\rho_1$ 下为 $\rho_1(w)=\tfrac14$。

*学到的学生。* $m\widehat p$ 服从 Binomial$(m,f)$，$f\in\lbrace \tfrac34,\tfrac14\rbrace $，所以对每个 $m\ge1$ 都有 $\mathbb E\widehat p=f$。价值关于 $p$ 是仿射的，并且在两个数据源下是同一个函数，因此
$$\mathbb EJ(\pi_{\widehat p})=\tfrac12+\tfrac{\mathbb E\widehat p}{20}=\tfrac12+\tfrac3{80}=\tfrac{43}{80}\quad\text{或}\quad\tfrac12+\tfrac1{80}=\tfrac{41}{80}.\qquad\square$$

**命题 4.2 的证明。**

*一阶矩相同。* 两个数据源下每个 $Y_i$ 都是 Bernoulli$(\tfrac12)$，所以边缘分布相同。$p_0$ 处的梯度为 $p_0-(Y_1+Y_2)/2$，两种数据源下均值都是 $0$，因此两者都有 $\mathbb Ep^+=\tfrac12$。

*方差。* $\operatorname{Var}p^+=\eta^2\operatorname{Var}\frac{Y_1+Y_2}2$。

- 独立时，$\operatorname{Var}\frac{Y_1+Y_2}2=\tfrac14(\tfrac14+\tfrac14)=\tfrac18$。
- 配对时，$Y_1+Y_2\equiv1$，方差为 $0$。

*价值。* 对 $J(p)=p^2$，$\mathbb EJ(p^+)=(\mathbb Ep^+)^2+\operatorname{Var}p^+$，所以差为 $\eta^2\cdot\tfrac18-0=\eta^2/8$。对 $J'=1-p^2$，$\mathbb EJ'(p^+)=1-(\mathbb Ep^+)^2-\operatorname{Var}p^+$，差为 $-\eta^2/8$。$\square$

## 注记

1. **哪些相同，哪些不同。**
   - 命题 4.1 中，静态评估在整个受限策略类 $\Pi=\lbrace \pi_p\rbrace $ 和 teacher 上都一致。但学习消耗的是 teacher 的*动作频率*，而价值并不决定动作频率。
   - 命题 4.2 中，标签边缘分布和平均梯度都一致。但部署的曲率读取的是 packet 的*二阶矩*。
   - 两种情形下，真正决定结果的都是实际更新与部署目标的复合。N05 把这个复合作为核心对象。
2. **保真不等于改进。** (11) 的符号取决于任务：同样的方差在 $p^2$ 下有利，在 $1-p^2$ 下有害。梯度方差更低、保真度更高、任务表现更好是三件不同的事。
3. **适用范围。** 命题 4.1 针对所述的 aliased 策略类，与无限制策略/价值测试下的完整 Bellman 价值等价并不矛盾。这个差距依赖于学习算法：一个忽略两份数据、直接输出 $p=1$ 的学习器没有差距。这恰恰说明，充分性必须相对于所声明的学习器来定义。

## 依赖与来源

- **依赖：** N01（分解：这些差距落在学习器提取到了什么，而不在表示能力）。
- **被以下节点使用：** N05（说明为什么要以 continuation response 作为不变量）。
- **来源：** `docs/direction/v015_revision/Environment_Information_Theory_EN.md` §4，命题 5–6，式 (10)–(11)。审查：`docs/direction/v014_handoff/Independent_Proof_Audit_EN.md` A9–A10。
