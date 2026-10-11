# N02 · 被动不可能，主动可达 — 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/N02_passive_vs_active_acquisition_ZH.md)

## 设定与假设

**两世界 aliased 族。**

- 未知符号 $`v\in\lbrace -1,+1\rbrace`$ 决定世界，gap $`0<\Delta\le1/8`$ 已知。
- 隐状态 $`s\in\lbrace 0,1\rbrace`$ 均匀分布。
- 学生只有一个 aliased context，看不到 $`s`$，只做一次决策：以概率 $`p`$ 选动作 $`A`$，否则选 $`B`$。
- 成功概率：

| 状态 | $`A`$ 成功概率 | $`B`$ 成功概率 |
|---|---:|---:|
| $`s=0`$ | $`1`$ | $`\tfrac12-2v\Delta`$ |
| $`s=1`$ | $`\tfrac12+2v\Delta`$ | $`1`$ |

- teacher 能看到 $`s`$，在 $`s=0`$ 时选 $`A`$，在 $`s=1`$ 时选 $`B`$。

**假设 2.1（被动接口）。** 学习器只能看到 teacher 示范（可以暴露 $`s`$），以及对这些示范的随机后处理。

**假设 2.2（查询接口）。** 此外，学习器可以重置到 $`s=1`$，执行 $`A`$，并观察 Bernoulli 结果。查询可以自适应选择；查询次数 $`N`$ 是满足 $`\mathbb E_vN<\infty`$ 的停时。学习器的算法与私有随机性在两个世界中相同。

记 $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$。

## 陈述

**引理 2.0（取值）。** $`J_v(p)=\tfrac34+v\Delta(2p-1)`$，$`J_v^\Pi=\tfrac34+\Delta`$；$`v=+1`$ 时在 $`p=1`$ 取到，$`v=-1`$ 时在 $`p=0`$ 取到。teacher 以概率 1 成功，并在两个世界中都严格最优。

**(a)** **定理（被动不可能）。** 在假设 2.1 下，任何学习器都满足

```math
\max_{v}\mathbb E\big[J_v^\Pi-J_v(\widehat p)\big]\ge\Delta.
```

**(b)** **定理（主动下界）。** 在假设 2.2 下，令 $`0<\delta<1/2`$。若在每个世界中，学习器都以至少 $`1-\delta`$ 的概率输出遗憾不超过 $`\Delta/2`$ 的学生，则对每个 $`v`$，

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}. \qquad(8)
```

**(c)** **定理（主动上界）。** $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ 次独立查询足以在每个世界中以至少 $`1-\delta`$ 的概率输出最优端点 $`p\in\lbrace 0,1\rbrace`$。

**(d)** **命题（编译）。** 估计出符号后，选目标 $`q=7/8`$（若 $`\widehat v=+1`$）或 $`q=1/8`$（否则），生成 $`m`$ 个 Bernoulli $`(q)`$ 目标标签，令学生为它们的经验频率 $`\widehat p`$。若 $`m\ge\lceil32\log(2/\delta_L)\rceil`$，则以至少 $`1-\delta_L`$ 的概率 $`\lvert\widehat p-q\rvert\le1/8`$，此时编译出的学生以至少 $`1-\delta-\delta_L`$ 的概率遗憾不超过 $`\Delta/2`$。

## 证明

**引理 2.0 的证明。** 对 $`s`$ 取平均：

```math
J_v(p)=\tfrac12\big[p+(1-p)(\tfrac12-2v\Delta)\big]+\tfrac12\big[p(\tfrac12+2v\Delta)+(1-p)\big].
```

第一个括号等于 $`\tfrac{1+p}{2}-2v\Delta(1-p)`$，第二个等于 $`1-\tfrac p2+2v\Delta p`$。两者之和为 $`\tfrac32+2v\Delta(2p-1)`$，所以 $`J_v(p)=\tfrac34+v\Delta(2p-1)`$。它关于 $`p`$ 是线性的，在 $`[0,1]`$ 上的最大值 $`\tfrac34+\Delta`$ 在所述端点取到。

teacher 总是选成功概率为 1 的动作。由 $`2\Delta\le\tfrac14`$，其余每个状态–动作的成功概率都等于 $`\tfrac12\pm2\Delta\in[\tfrac14,\tfrac34]`$，所以 teacher 在两个世界中都严格最优。$`\square`$

**(a) 的证明。** 一条示范记录 $`(s,\text{动作},\text{结果})`$；在两个世界中，它都以各 $`\tfrac12`$ 的概率等于 $`(0,A,1)`$ 或 $`(1,B,1)`$。因此示范的分布与 $`v`$ 无关，它的任何随机后处理（包括学习器输出 $`\widehat p`$）的分布也与 $`v`$ 无关。令 $`m=\mathbb E\widehat p`$，在两个世界中相同。由于 $`J_v`$ 关于 $`p`$ 是仿射的，$`\mathbb EJ_v(\widehat p)=J_v(m)`$，两个期望遗憾分别为

```math
J_{+1}^\Pi-J_{+1}(m)=2\Delta(1-m),\qquad J_{-1}^\Pi-J_{-1}(m)=2\Delta m.
```

二者的最大值不小于其平均值 $`\Delta`$。$`m=\tfrac12`$ 时取等号，例如取 $`\widehat p\equiv\tfrac12`$。$`\square`$

**(b)**

*第 1 步（化为二元检验）。* 在世界 $`+1`$ 中，遗憾 $`2\Delta(1-\widehat p)\le\Delta/2`$ 当且仅当 $`\widehat p\ge3/4`$；在世界 $`-1`$ 中，$`2\Delta\widehat p\le\Delta/2`$ 当且仅当 $`\widehat p\le1/4`$。令 $`A=\lbrace \widehat p>1/2\rbrace`$，则 $`P_{+1}(A)\ge1-\delta`$，$`P_{-1}(A)\le\delta`$。

*第 2 步（停止后 transcript 的 KL）。* 令 $`T`$ 为完整 transcript：示范、查询结果 $`Y_1,\dots,Y_N`$ 以及学习器的私有随机性；记其分布为 $`P_v^T`$。示范与私有随机性在两个世界中分布相同；给定过去，每次查询的结果服从 Bernoulli $`(\tfrac12+2v\Delta)`$。记 $`d_\Delta=\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)`$。

令 $`\ell(y)=\log\frac{P_{+1}(Y=y)}{P_{-1}(Y=y)}`$ 为单次查询的对数似然比，它有界：$`\lvert\ell\rvert\le\log\frac{1/2+2\Delta}{1/2-2\Delta}=:c`$。示范与随机性的因子相互抵消，所以 transcript 的对数似然比是 $`\Lambda(T)=\sum_{t=1}^{N}\ell(Y_t)`$。记 $`\mathcal F_{t-1}`$ 为第 $`t`$ 次查询之前的历史。事件 $`\lbrace t\le N\rbrace`$ 关于 $`\mathcal F_{t-1}`$ 可测，且在该事件上 $`\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]=d_\Delta`$。又 $`\sum_t\mathbb E_{+1}\lvert1\lbrace t\le N\rbrace \ell(Y_t)\rvert\le c\,\mathbb E_{+1}N<\infty`$，所以可用 Fubini 定理。这就是 Wald 恒等式：

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})=\mathbb E_{+1}\Lambda(T)=\sum_{t\ge1}\mathbb E_{+1}\big[1\lbrace t\le N\rbrace \,\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]\big]=\mathbb E_{+1}[N]\,d_\Delta.
```

*第 3 步（数据处理）。* 事件 $`A`$ 是 $`T`$ 的函数，所以

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})\ge\mathrm{kl}(P_{+1}(A),P_{-1}(A))\ge\mathrm{kl}(1-\delta,\delta).
```

第二个不等号成立，是因为当 $`y\le x`$ 时 $`\mathrm{kl}(x,y)`$ 关于 $`x`$ 不减、关于 $`y`$ 不增；在矩形 $`x\ge1-\delta>\tfrac12>\delta\ge y`$ 上，最小值在 $`(1-\delta,\delta)`$ 处取到。结合第 2、3 步，得到 $`v=+1`$ 时 (8) 的第一个不等式。$`v=-1`$ 时同理，因为 $`\mathrm{kl}(\tfrac12-2\Delta,\tfrac12+2\Delta)=d_\Delta`$。

*第 4 步（常数）。* 对 $`0<u\le\tfrac14`$，

```math
\mathrm{kl}(\tfrac12+u,\tfrac12-u)=2u\log\frac{1+2u}{1-2u}=4u\,\mathrm{artanh}(2u)=8u^2\sum_{j\ge0}\frac{(4u^2)^j}{2j+1}.
```

级数系数为正，关于 $`u`$ 递增，在 $`u=\tfrac14`$ 处等于 $`2\mathrm{artanh}(\tfrac12)=\log3`$。因此 $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le8\log3\cdot u^2\le16u^2`$。取 $`u=2\Delta\le\tfrac14`$，得 $`d_\Delta\le64\Delta^2`$，这就是 (8) 的第二个不等式。$`\square`$

**(c) 的证明。** 抽 $`n`$ 个独立查询结果；若 $`\bar Y>\tfrac12`$ 则令 $`\widehat v=+1`$，否则 $`\widehat v=-1`$（分析中平局按错误计），输出 $`\widehat p=1\lbrace \widehat v=+1\rbrace`$。在世界 $`+1`$ 中 $`\mathbb E\bar Y=\tfrac12+2\Delta`$，单侧 Hoeffding 不等式给出

```math
P_{+1}(\bar Y\le\tfrac12)\le\exp(-2n(2\Delta)^2)=\exp(-8n\Delta^2)\le\delta
```

只要 $`n\ge\log(1/\delta)/(8\Delta^2)`$。世界 $`-1`$ 对称。在成功事件上，输出就是最优端点，遗憾为 $`0`$。$`\square`$

**(d) 的证明。** 双侧 Hoeffding 不等式给出

```math
\Pr(\lvert\widehat p-q\rvert>\tfrac18)\le2\exp(-2m/64)=2e^{-m/32}\le\delta_L
```

只要 $`m\ge32\log(2/\delta_L)`$。若 $`\widehat v=v=+1`$ 且 $`\lvert\widehat p-q\rvert\le\tfrac18`$，则 $`\widehat p\ge\tfrac34`$，遗憾为 $`2\Delta(1-\widehat p)\le\Delta/2`$。$`v=-1`$ 时对称，此时 $`\widehat p\le\tfrac14`$。对识别失败（概率 $`\le\delta`$）与编译失败（概率 $`\le\delta_L`$）做 union bound，即得所述概率。$`\square`$

## 注记

1. **完美示范也可能信息不足。** teacher 在两个世界中都以概率 1 成功，但任何被动学习器都有 $`\mathcal R_{\rm info}\ge\Delta`$（N01）。即使示范暴露了 $`s`$，缺失的事实（$`v`$ 的符号）也不在示范里。
2. **获取与编译是两笔不同的成本。** 识别需要 $`\Theta(\Delta^{-2}\log(1/\delta))`$ 次真实环境查询，(8) 与 (c) 在常数意义下匹配。编译只需 $`O(\log(1/\delta_L))`$ 个*合成*目标标签。这些标签是编译出来的，不是新的 teacher 执行，必须单独记账。
3. **紧性。** 精确计算（见实验页）表明，SPRT 在 $`(\Delta,\delta)=(\tfrac18,\tfrac1{10})`$ 和 $`(\tfrac18,\tfrac14)`$ 处恰好达到界 (8)，在 $`(\tfrac1{16},0.01)`$ 和 $`(\tfrac1{64},0.01)`$ 处与界相差不到 1%。
4. **用到的假设。** (c) 的停止规则用到了已知的 $`\Delta`$。(b) 需要所声明的查询接口、与世界无关的算法，以及 $`\mathbb E_vN<\infty`$。这个论证是把标准的 change of measure 用在学生–teacher 信息不对称上，并不是新的检验不等式。

## 依赖与来源

- **依赖：** [N01](N01_decomposition_and_ambiguity_ZH.md)（分解与[N01(d)](N01_decomposition_and_ambiguity_ZH.md)：对被动数据做后处理消除不了 $`\mathcal R_{\rm info}`$）。
- **被以下节点使用：** N03（把 $`K`$ 与查询预算显式化）。

**出处：** [v0.14.1 §3](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#3-a-complete-acquisition-example-perfect-demonstrations-can-be-insufficient) — Thm. 4; eqs. (6)–(9); compilation paragraph. [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §3，定理 4，式 (6)–(9) 及编译段落。审查：[Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) A5–A8。
