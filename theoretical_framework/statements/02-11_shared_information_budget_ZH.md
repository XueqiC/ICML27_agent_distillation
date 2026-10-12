# Proposition 2.11 · 示范与查询共用一份信息预算

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-11_shared_information_budget_ZH.md) · [证明](../proofs/02-11_shared_information_budget_ZH.md) · [实验](../experiments/02-11_shared_information_budget_ZH.md) · [English](02-11_shared_information_budget.md)

**一段话.** 示范与查询都携带"学生处在哪个世界"的信息。这个命题说明它们共用一份预算：要以至多 $`\delta`$ 的误差在两个世界之间作出决定，示范的 KL 信息加上查询的 KL 信息必须达到 $`\mathrm{kl}(1-\delta,\delta)`$；这对每个自适应的查询–停止规则都成立。

**本页用到的设定.**

- 两个世界 $`v\in\lbrace0,1\rbrace`$，要求不同的最终决策。
- $`K`$ 条独立同分布的示范；$`\kappa_D^{(v)}`$ 界住单条示范的 KL 散度 $`\mathrm{KL}(P_v^{\rm demo}\Vert P_{1-v}^{\rm demo})`$。
- 查询规则根据历史 $`H_{t-1}`$ 选择查询 $`X_t`$，观察结果 $`Y_t`$，在 $`N\le N_{\max}`$ 时停止；它的 kernel 在两个世界中相同。$`\kappa_X^{(v)}`$ 界住给定任何历史时单次查询结果的 KL 散度。
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ 是二元 KL 散度；$`\delta<\tfrac12`$ 是每个世界中允许的误差。

<a id="s2-11"></a>

**Proposition 2.11（示范与查询共用一份信息预算）.** 考虑两世界的自适应获取。

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

**说了什么.**

- 要以至多 $`\delta`$ 的误差区分两个世界，$`K`$ 条示范的 KL 信息加上每次查询的 KL 信息乘以期望查询次数，必须达到 $`\mathrm{kl}(1-\delta,\delta)`$。
- 若查询不携带信息、示范又不够，那么任何查询预算都没有用。

**直觉.**

- 示范与查询往同一个账户里存钱：有信息的示范减少所需的查询，replay 什么也不存。

**在框架中的作用.**

- 它给出 $`K`$ 与查询预算之间的兑换率。[Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8) 让这个兑换率依赖于学习器。

**推导要点.**

- 查询选择、停止与随机化 kernel 在两个世界中相同，在似然比中相消，只剩示范项与查询结果项。
- 前者的期望至多为 $`K\kappa_D`$，后者至多为 $`\kappa_X\,\mathbb EN`$；data processing 不等式要求两者之和至少为 $`\mathrm{kl}(1-\delta,\delta)`$。

**说明.**

- **(2.11) 的范围.** 它是二元识别的界。策略学习版本需要互不相交的近似最优输出集合。(2.12) 右端为零并不说明可达。

**用到.**

- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6): change of measure 论证.

**被用于.**

- [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8): 学习器相对的兑换率.
