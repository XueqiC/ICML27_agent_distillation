# Proposition 2.11 · 示范与查询共用一份信息预算

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-11_shared_information_budget_ZH.md) · [证明](02-11_shared_information_budget_ZH.md) · [实验](../experiments/02-11_shared_information_budget_ZH.md) · [English](02-11_shared_information_budget.md)

**本页证明什么.** 把停止后的 transcript 补齐到固定长度，写出它在两个世界之间的对数似然比，注意到查询选择与停止 kernel 相消。界住其余各项的期望，再对"决策正确"这一事件用 data processing，就得到预算不等式。

**本页用到的设定.**

- 两个世界 $`v\in\lbrace0,1\rbrace`$；$`K`$ 条独立同分布的示范，每条 KL 至多 $`\kappa_D^{(v)}`$；给定任何历史 $`h`$ 时，每次查询结果的 KL 至多 $`\kappa_X^{(v)}`$；与世界无关的查询–停止规则，$`N\le N_{\max}`$。
- $`\mathrm{kl}(a,b)`$ 是二元 KL 散度；$`H_{t-1}`$ 是第 $`t`$ 次查询之前的历史。

**陈述（[Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11)）.** 考虑两世界的自适应获取。

- 世界为 $`v\in\lbrace 0,1\rbrace`$，两个世界要求不同的最终决策。
- $`K`$ 条独立同分布的示范中，每条满足 $`\mathrm{KL}(P_v^{\rm demo}\Vert P_{1-v}^{\rm demo})\le\kappa_D^{(v)}<\infty`$。
- 一个共同的、依赖历史的规则选择查询，并在 $`N\le N_{\max}<\infty`$ 时停止。该规则的选择、停止与私有随机化 kernel 都不依赖于 $`v`$。
- 对每个允许的查询 $`x`$ 和相关支撑上的每段之前的历史 $`h`$，$`\mathrm{KL}(P_v(\cdot\mid h,x)\Vert P_{1-v}(\cdot\mid h,x))\le\kappa_X^{(v)}<\infty`$。
- 最终决策在每个世界中出错的概率至多为 $`\delta`$，$`0<\delta<1/2`$。

则

```math
K\kappa_D^{(v)}+\kappa_X^{(v)}\,\mathbb E_vN\ge\mathrm{kl}(1-\delta,\delta). \qquad(2.11)
```

若 $`0<\kappa_X^{(v)}<\infty`$，

```math
\mathbb E_vN\ge\frac{[\mathrm{kl}(1-\delta,\delta)-K\kappa_D^{(v)}]_+}{\kappa_X^{(v)}}. \qquad(2.12)
```

若 $`\kappa_X^{(v)}=0`$ 且 $`K\kappa_D^{(v)}<\mathrm{kl}(1-\delta,\delta)`$，则任何查询预算都达不到所述误差。

**证明.**

*补齐.* 停止后用固定的空符号把 transcript 补齐到长度 $`N_{\max}`$，记为 $`T=(O_1,\dots,O_K,(X_1,Y_1),\dots,(X_N,Y_N))`$，连同停止指示与私有随机性。

*似然比.* 选择、停止与随机化 kernel 在两个世界中相同，所以它们相消：

```math
\log\frac{P_v(T)}{P_{1-v}(T)}=\sum_{k=1}^K\log\frac{P^{\rm demo}_v(O_k)}{P^{\rm demo}_{1-v}(O_k)}+\sum_{t=1}^{N_{\max}}1\lbrace t\le N\rbrace \log\frac{P_v(Y_t\mid H_{t-1},X_t)}{P_{1-v}(Y_t\mid H_{t-1},X_t)}.
```

有限的 KL 上界蕴含相关支撑上的绝对连续性，所以每一项几乎必然有限。

*取 $`\mathbb E_v`$.*

- 第一个和的期望为 $`K\,\mathrm{KL}(P^{\rm demo}_v\Vert P^{\rm demo}_{1-v})\le K\kappa_D^{(v)}`$。
- 第二个和中，$`\lbrace t\le N\rbrace`$ 与 $`X_t`$ 关于 $`(H_{t-1},\text{随机性})`$ 可测。逐项取条件期望，每一项的期望为 $`\mathbb E_v[1\lbrace t\le N\rbrace \,\mathrm{KL}(P_v(\cdot\mid H_{t-1},X_t)\Vert P_{1-v}(\cdot\mid H_{t-1},X_t))]\le\kappa_X^{(v)}\Pr_v(t\le N)`$。
- 由于 $`N_{\max}<\infty`$，和有限。对 $`t`$ 求和得 $`\kappa_X^{(v)}\mathbb E_vN`$。

所以 $`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\le K\kappa_D^{(v)}+\kappa_X^{(v)}\mathbb E_vN`$。

*data processing.* 对"最终决策对 $`v`$ 正确"这一指示变量用 data processing。它在 $`v`$ 下的概率 $`x\ge1-\delta`$，在 $`1-v`$ 下的概率 $`y\le\delta`$，所以由二元 kl 在矩形 $`x\ge1-\delta>\tfrac12>\delta\ge y`$ 上的单调性，$`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\ge\mathrm{kl}(x,y)\ge\mathrm{kl}(1-\delta,\delta)`$。两个界合起来即得 (2.11)，移项得 (2.12)。

*$`\kappa_X^{(v)}=0`$ 的情形.* (2.11) 左端等于 $`K\kappa_D^{(v)}`$。若它小于 $`\mathrm{kl}(1-\delta,\delta)`$，则 (2.11) 对任何 $`N`$ 都不成立，这样的算法不存在。$`\square`$

## 相关工作

- 带停时的自适应采样的 change of measure 下界，是 best-arm identification 中的标准论证 ([Kaufmann, Cappé & Garivier, 2016](../references_ZH.md#kaufmann2016complexity)).
- KL 散度的 data processing 不等式是标准结论 ([Cover & Thomas, 2006](../references_ZH.md#cover2006elements)).

## 参考文献

- [Kaufmann, Cappé & Garivier, 2016](../references_ZH.md#kaufmann2016complexity)
- [Cover & Thomas, 2006](../references_ZH.md#cover2006elements)
