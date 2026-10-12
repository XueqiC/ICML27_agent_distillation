# Theorem 2.6 · 被动不可能，主动可达

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-06_passive_impossibility_active_attainability_ZH.md) · [证明](02-06_passive_impossibility_active_attainability_ZH.md) · [实验](../experiments/02-06_passive_impossibility_active_attainability_ZH.md) · [English](02-06_passive_impossibility_active_attainability.md)

**本页证明什么.** 被动部分成立，是因为示范在两个世界中分布相同，所以输出也相同，而仿射的值迫使某一个世界的 regret 至少为 $`\Delta`$。查询下界是对停止后的 transcript 做 change of measure（Wald 恒等式加 data processing），上界是 Hoeffding 不等式。

**本页用到的设定.**

- 未知符号 $`v\in\lbrace -1,+1\rbrace`$ 选定两个世界之一；差距 $`0<\Delta\le1/8`$ 已知。
- 隐藏状态 $`s\in\lbrace 0,1\rbrace`$ 均匀分布。teacher 能看到 $`s`$：在 $`s=0`$ 选 $`A`$，在 $`s=1`$ 选 $`B`$，两者都以概率 1 成功，所以 teacher 在两个世界中都最优。
- 学生是混叠的（看不到 $`s`$），以概率 $`p`$ 选 $`A`$。另外两个状态–动作对的成功概率为 $`\tfrac12\mp2v\Delta`$，所以学生的值为 $`J_v(p)=\tfrac34+v\Delta(2p-1)`$，最好值为 $`J_v^\Pi=\tfrac34+\Delta`$：$`v=+1`$ 时在 $`p=1`$ 取到，$`v=-1`$ 时在 $`p=0`$ 取到。
- *被动接口：* 学习器看到 teacher 示范（可能附带 $`s`$）及对它们的后处理。*查询接口：* 它还可以自适应地 reset 到 $`s=1`$、执行 $`A`$，观察 Bernoulli $`(\tfrac12+2v\Delta)`$ 结果，查询次数是停时 $`N`$。
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ 是二元 KL 散度。
- 完整定义：[Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5)。

**陈述（[Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)）.** 在 [Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5) 的族中：

- **(a) 被动不可能.** 在被动接口下，每个学习器都满足 $`\max_{v}\mathbb E\big[J_v^\Pi-J_v(\widehat p)\big]\ge\Delta`$。
- **(b) 主动下界.** 在查询接口下，设 $`0<\delta<1/2`$，并设在每个世界中，学习器以至少 $`1-\delta`$ 的概率输出 regret 不超过 $`\Delta/2`$ 的学生。则对每个 $`v`$，

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}. \qquad(2.6)
```

- **(c) 主动上界.** $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ 次独立查询足以在每个世界中以至少 $`1-\delta`$ 的概率输出最优端点 $`p\in\lbrace 0,1\rbrace`$。

**证明.**

**(a)** 一条示范记录 $`(s,\text{动作},\text{结果})`$，在两个世界中都以各 $`\tfrac12`$ 的概率等于 $`(0,A,1)`$ 或 $`(1,B,1)`$。示范的分布与 $`v`$ 无关，对它的任何随机后处理（包括学习器的输出 $`\widehat p`$）的分布也与 $`v`$ 无关。令 $`m=\mathbb E\widehat p`$，两个世界中相同。$`J_v`$ 关于 $`p`$ 是仿射的，所以 $`\mathbb EJ_v(\widehat p)=J_v(m)`$，两个期望 regret 为

```math
J_{+1}^\Pi-J_{+1}(m)=2\Delta(1-m),\qquad J_{-1}^\Pi-J_{-1}(m)=2\Delta m .
```

两者的最大值不小于平均值 $`\Delta`$。$`m=\tfrac12`$ 时取等，例如 $`\widehat p\equiv\tfrac12`$。

**(b)** *第 1 步（化为二元检验）.* 在世界 $`+1`$ 中，regret $`2\Delta(1-\widehat p)\le\Delta/2`$ 恰好等价于 $`\widehat p\ge3/4`$；在世界 $`-1`$ 中，$`2\Delta\widehat p\le\Delta/2`$ 恰好等价于 $`\widehat p\le1/4`$。令 $`A=\lbrace \widehat p>1/2\rbrace`$，则 $`P_{+1}(A)\ge1-\delta`$，$`P_{-1}(A)\le\delta`$。

*第 2 步（停止后 transcript 的 KL）.* 令 $`T`$ 为完整 transcript：示范、查询结果 $`Y_1,\dots,Y_N`$ 和学习器的私有随机性，其分布记为 $`P_v^T`$。

- 示范与私有随机性在两个世界中分布相同。给定过去，每次查询的结果是 Bernoulli $`(\tfrac12+2v\Delta)`$。记 $`d_\Delta=\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)`$。
- 令 $`\ell(y)=\log\frac{P_{+1}(Y=y)}{P_{-1}(Y=y)}`$ 为单次查询的对数似然比。它有界：$`\lvert\ell\rvert\le\log\frac{1/2+2\Delta}{1/2-2\Delta}=:c`$。
- 示范因子与随机化因子相消，transcript 的对数似然比为 $`\Lambda(T)=\sum_{t=1}^{N}\ell(Y_t)`$。
- 令 $`\mathcal F_{t-1}`$ 为第 $`t`$ 次查询之前的历史。事件 $`\lbrace t\le N\rbrace`$ 是 $`\mathcal F_{t-1}`$ 可测的，并且在该事件上 $`\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]=d_\Delta`$。又 $`\sum_t\mathbb E_{+1}\lvert1\lbrace t\le N\rbrace \ell(Y_t)\rvert\le c\,\mathbb E_{+1}N<\infty`$，所以可用 Fubini 定理（Wald 恒等式）：

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})=\mathbb E_{+1}\Lambda(T)=\sum_{t\ge1}\mathbb E_{+1}\big[1\lbrace t\le N\rbrace \,\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]\big]=\mathbb E_{+1}[N]\,d_\Delta .
```

*第 3 步（data processing）.* 事件 $`A`$ 是 $`T`$ 的函数，所以

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})\ge\mathrm{kl}(P_{+1}(A),P_{-1}(A))\ge\mathrm{kl}(1-\delta,\delta).
```

第二个不等号成立，是因为当 $`y\le x`$ 时 $`\mathrm{kl}(x,y)`$ 关于 $`x`$ 不减、关于 $`y`$ 不增；在矩形 $`x\ge1-\delta>\tfrac12>\delta\ge y`$ 上最小值在 $`(1-\delta,\delta)`$ 处取到。第 2、3 步给出 $`v=+1`$ 时 (2.6) 的第一个不等式。$`v=-1`$ 时论证相同，因为 $`\mathrm{kl}(\tfrac12-2\Delta,\tfrac12+2\Delta)=d_\Delta`$。

*第 4 步（常数）.* 对 $`0<u\le\tfrac14`$，

```math
\mathrm{kl}(\tfrac12+u,\tfrac12-u)=2u\log\frac{1+2u}{1-2u}=4u\,\mathrm{artanh}(2u)=8u^2\sum_{j\ge0}\frac{(4u^2)^j}{2j+1}.
```

级数系数为正，关于 $`u`$ 递增；在 $`u=\tfrac14`$ 处它等于 $`2\,\mathrm{artanh}(\tfrac12)=\log3`$。所以 $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le8\log3\cdot u^2\le16u^2`$。取 $`u=2\Delta\le\tfrac14`$ 得 $`d_\Delta\le64\Delta^2`$，即 (2.6) 的第二个不等式。

**(c)** 抽 $`n`$ 个独立的查询结果：若 $`\bar Y>\tfrac12`$ 令 $`\widehat v=+1`$，否则 $`\widehat v=-1`$（分析中平局算作错误）。输出 $`\widehat p=1\lbrace \widehat v=+1\rbrace`$。在世界 $`+1`$ 中 $`\mathbb E\bar Y=\tfrac12+2\Delta`$，单侧 Hoeffding 不等式给出

```math
P_{+1}(\bar Y\le\tfrac12)\le\exp(-2n(2\Delta)^2)=\exp(-8n\Delta^2)\le\delta
```

只要 $`n\ge\log(1/\delta)/(8\Delta^2)`$。世界 $`-1`$ 对称。在成功事件上输出就是最优端点，regret 为 0。$`\square`$

## 相关工作

- 停止后 transcript 的 KL 散度用带停时求和的 Wald 恒等式计算 ([Wald, 1944](../references_ZH.md#wald1944cumulative)).
- 充分性的界用的是有界独立变量的 Hoeffding 不等式 ([Hoeffding, 1963](../references_ZH.md#hoeffding1963probability)).
- 带停时的自适应采样的 change of measure 下界，是 best-arm identification 中的标准论证 ([Kaufmann, Cappé & Garivier, 2016](../references_ZH.md#kaufmann2016complexity)).

## 参考文献

- [Wald, 1944](../references_ZH.md#wald1944cumulative)
- [Hoeffding, 1963](../references_ZH.md#hoeffding1963probability)
- [Kaufmann, Cappé & Garivier, 2016](../references_ZH.md#kaufmann2016complexity)
