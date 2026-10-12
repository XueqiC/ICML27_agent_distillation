# Proposition 3.1 · 静态价值不等于学习价值

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/03-01_static_value_not_learning_value_ZH.md) · [证明](03-01_static_value_not_learning_value_ZH.md) · [实验](../experiments/03-01_static_value_not_learning_value_ZH.md) · [English](03-01_static_value_not_learning_value.md)

**本页证明什么.** 两部分都是显式计算。(a) 中两个状态分布给动作 $`A`$ 的平均奖励相同，但 teacher 的动作频率不同；(b) 中两个包均值相同、方差不同，而弯曲的部署值读取方差。

**本页用到的设定.**

- *混叠学生* $`\pi_p`$：一个上下文，以概率 $`p`$ 选 $`A`$。*频率蒸馏：* $`\widehat p`$ = $`m`$ 条 teacher 示范中 $`A`$ 的比例。在平方损失上做一步步长为 $`\eta`$ 的 *SGD*。

**陈述（[Proposition 3.1](../statements/03-01_static_value_not_learning_value_ZH.md#s3-1)）.** *构造 (a)：四状态来源.*

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

**证明.**

**(a)**

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
