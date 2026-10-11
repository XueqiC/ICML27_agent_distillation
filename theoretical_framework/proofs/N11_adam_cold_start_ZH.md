# N11 · 冷启动 Adam 的商结构与优化器记忆 — 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/N11_adam_cold_start_ZH.md)

## 设定与假设

**优化器。** 采用带 bias correction 的 Adam（可带解耦 weight decay，即 AdamW）。$`\epsilon`$ 放在平方根外，不用 AMSGrad。更新消耗的梯度 $`G_t\in\mathbb R^p`$ 指经过声明的累积、loss 归一化、缩放或 clipping *之后*的梯度。步计数 $`t\ge1`$ 时：

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)G_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)G_t^{\odot2},
```

```math
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\,\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}. \qquad(A6)
```

除法和平方根逐坐标进行。参数满足 $`\beta_1,\beta_2\in[0,1)`$、$`\eta>0`$、$`\lambda\ge0`$、$`\epsilon\ge0`$。运算按精确实数算术，没有投影或量化。

**冷状态。** 冷状态指 $`m_0=v_0=0`$、步计数为 $`0`$。记 $`c=(1-\eta\lambda)\theta_0`$。

**符号映射。** 当 $`\epsilon>0`$ 时，定义 $`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$。当 $`\epsilon=0`$ 时，定义*理想符号更新* $`h_0(g)_i=\mathrm{sign}(g_i)`$，并约定 $`\mathrm{sign}(0)=0`$。这个约定是定义的一部分；字面意义上 $`\epsilon=0`$ 的浮点 Adam 会算出 $`0/0`$。

**效用。** *即时参数效用*指 $`\theta_1`$ 的有界可测函数。*完整状态效用*指下一个完整状态 $`(\theta_1,m_1,v_1,1)`$ 的有界可测函数。$`P,Q`$ 是 $`\mathbb R^p`$ 上的消耗梯度分布；有限情形下它们支撑在有限字母表 $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$ 上。

## 命题

**(a)–(c)** **（冷启动商）。**

**1.** **(a)** 从冷状态出发，$`\theta_1=c-\eta\,h_\epsilon(G)`$。

**2.** **(b)** （$`\epsilon=0`$）两个分布对所有即时参数效用给出相同期望，当且仅当它们的联合符号向量分布相同。在可达符号模式数为 $`s`$ 的有限字母表上，对 $`\mathcal A`$ 上任意分布、足以确定所有这类效用的线性期望统计量的最少个数是 $`s-1`$。这里 $`s\le3^p`$；若没有零坐标则 $`s\le2^p`$。

**3.** **(c)** （$`\epsilon>0`$）$`h_\epsilon`$ 是逐坐标的双射 $`\mathbb R\to(-1,1)`$，且

```math
h_\epsilon^{-1}(u)_i=\frac{\epsilon u_i}{1-\lvert u_i\rvert},\qquad\frac{d}{du}h_\epsilon^{-1}(u)=\frac{\epsilon}{(1-\lvert u\rvert)^2}. \qquad(A8)
```

因此，所有即时参数效用的期望相等，等价于完整的消耗梯度分布相等。在 $`\mathcal A`$ 上所需统计量的最少个数为 $`m-1`$。

**4.** （联合与边缘）逐坐标的符号边缘分布不能确定联合符号分布。取 $`p=2`$，$`P`$ 在 $`\lbrace (1,1),(-1,-1)\rbrace`$ 上均匀，$`Q`$ 在 $`\lbrace (1,-1),(-1,1)\rbrace`$ 上均匀。两者每个坐标的符号边缘分布相同，平均更新都是 $`0`$；而效用 $`F(\theta)=1\lbrace (\theta_1-c_1)(\theta_2-c_2)>0\rbrace`$ 在 $`P`$ 下期望为 $`1`$，在 $`Q`$ 下为 $`0`$。

**(d)** **Proposition（对 margin 敏感的逼近）。** 设 $`F:\mathbb R^p\to[0,1]`$ 对所有点对 $`u=c-\eta h_\epsilon(g)`$、$`w=c-\eta h_0(g)`$（$`g`$ 取自支撑）满足 $`\lvert F(u)-F(w)\rvert\le\sum_iL_i\lvert u_i-w_i\rvert`$。对 $`a>0`$ 令 $`\beta_{P,i}(a)=P(0<\lvert G_i\rvert<a)`$，再令 $`\Delta_{\rm sign}(F)=\lvert\mathbb E_PF(c-\eta h_0(G))-\mathbb E_QF(c-\eta h_0(G))\rvert`$。则

```math
\big\lvert\mathbb E_PF(c-\eta h_\epsilon(G))-\mathbb E_QF(c-\eta h_\epsilon(G))\big\rvert\le\Delta_{\rm sign}(F)+\eta\sum_iL_i\Big[\beta_{P,i}(a)+\beta_{Q,i}(a)+\frac{2\epsilon}{a+\epsilon}\Big], \qquad(A9)
```

且 $`\Delta_{\rm sign}(F)\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$。

**(e)** **Proposition（保留的优化器状态）。** 当 $`\beta_1<1`$ 时，第一个完整状态包含 $`m_1=(1-\beta_1)G`$，它是 $`G`$ 的单射函数。

*反例：* 一维情形，$`\beta_1=\beta_2=\tfrac12`$、$`\lambda=0`$、$`\epsilon=0`$、$`\eta>0`$，冷启动。比较梯度历史 $`(G_1,G_2)=(1,-\tfrac34)`$ 与 $`(G_1',G_2')=(2,-\tfrac34)`$。

两者都给出 $`\theta_1=\theta_1'=\theta_0-\eta`$，符号历史都是 $`(+,-)`$。

第二步的 bias-corrected 矩分别为 $`\widehat m_2=-\tfrac16`$、$`\widehat m_2'=+\tfrac16`$，以及 $`\widehat v_2=\tfrac{17}{24}`$、$`\widehat v_2'=\tfrac{41}{24}`$。于是

```math
\theta_2=\theta_1+\frac{\eta}{6\sqrt{17/24}},\qquad\theta_2'=\theta_1-\frac{\eta}{6\sqrt{41/24}}. \qquad(A10)
```

两个第二步方向相反，有界检验 $`F(\theta)=1\lbrace \theta>\theta_1\rbrace`$ 把两者区分开，差值为 $`1`$。

这两段历史可以由 $`\theta_0=0`$ 处的平方损失 $`\tfrac12(\theta-a)^2`$ 实现：第一步目标分别为 $`a=-1`$、$`a=-2`$，第二步共用目标 $`a=\theta_1+\tfrac34`$。

**(e′)** **Corollary（记忆由动量携带）。** 若 $`\beta_1=0`$，任意两段 $`G_2`$ 符号相同的历史，其第二步方向也相同，因此不存在 (e) 型的反例。对每个 $`\beta_1\in(0,1)`$，(e) 的机制都存在。

## 证明

**(a) 的证明。** 冷启动的矩为 $`m_1=(1-\beta_1)G`$，$`v_1=(1-\beta_2)G^{\odot2}`$。除以 $`t=1`$ 的 bias correction 得 $`\widehat m_1=G`$、$`\widehat v_1=G^{\odot2}`$，故 $`\sqrt{\widehat v_1}=\lvert G\rvert`$（逐坐标）。代入 (A6) 得 $`\theta_1=c-\eta\,G/(\lvert G\rvert+\epsilon)`$。$`\epsilon=0`$ 时按所述约定 $`h_0=\mathrm{sign}`$。

**第 2 条。** 由于 $`\eta>0`$，从 $`\lbrace -1,0,1\rbrace ^p`$ 到 $`\mathbb R^p`$ 的映射 $`\sigma\mapsto c-\eta\sigma`$ 是单射，所以 $`\theta_1`$ 是符号向量 $`\sigma(G)`$ 的单射函数。

- *充分性：* 联合符号分布相同，则 $`\theta_1`$ 的分布相同，所有期望效用也相同。
- *必要性：* 对每个可达模式 $`\sigma^\star`$，指示函数 $`F=1\lbrace \theta_1=c-\eta\sigma^\star\rbrace`$ 是有界效用，期望为 $`\Pr(\sigma(G)=\sigma^\star)`$。所有这类 $`F`$ 的期望相等，就迫使各模式概率相等。

*维数。* 把每个效用看作 $`\mathcal A`$ 上的函数 $`u_F(g)=F(c-\eta h_0(g))`$。族 $`\mathcal U=\lbrace u_F\rbrace`$ 恰好由 $`\mathcal A`$ 上在 $`\sigma`$ 的各纤维上取常值的函数组成："包含"成立，因为 $`F`$ 在 $`s`$ 个不同的输出点上可以任意取值；"不多于"成立，因为 $`u_F`$ 只通过 $`\sigma(g)`$ 依赖于 $`g`$。所以 $`\mathrm{span}(\lbrace 1\rbrace \cup\mathcal U)`$ 由 $`s`$ 个纤维指示函数张成，维数为 $`s`$。

由 [N06(a)](N06_sufficiency_duality_ZH.md)（节点 [N06](N06_sufficiency_duality_ZH.md)），对 $`\mathcal A`$ 上一切分布，"均值相等即可推出所有 $`u\in\mathcal U`$ 的均值相等"的线性统计量最少需要 $`\dim\mathrm{span}(\lbrace 1\rbrace \cup\mathcal U)-1=s-1`$ 个，取 $`s-1`$ 个模式的指示函数即可达到。每个坐标的符号最多三种取值，故 $`s\le3^p`$；排除 $`0`$ 时 $`s\le2^p`$。

**第 3 条。** 固定 $`\epsilon>0`$。在 $`g\ge0`$ 上，$`h_\epsilon(g)=g/(g+\epsilon)`$ 连续且从 $`0`$ 严格增到 $`1`$；由奇对称性，$`h_\epsilon`$ 是 $`\mathbb R\to(-1,1)`$ 的双射。

解 $`u=g/(\lvert g\rvert+\epsilon)`$：有 $`\mathrm{sign}u=\mathrm{sign}g`$ 和 $`\lvert u\rvert(\lvert g\rvert+\epsilon)=\lvert g\rvert`$，故 $`\lvert g\rvert=\epsilon\lvert u\rvert/(1-\lvert u\rvert)`$，即 $`g=\epsilon u/(1-\lvert u\rvert)`$。$`u>0`$ 时求导得 $`\epsilon/(1-u)+\epsilon u/(1-u)^2=\epsilon/(1-u)^2`$，$`u<0`$ 时由对称性同理。

由于 $`g\mapsto c-\eta h_\epsilon(g)`$ 是单射，第 2 条的论证在"纤维为单点"的情形下照样成立：所有有界效用的期望相等等价于 $`G`$ 的分布相等，在 $`\mathcal A`$ 上维数为 $`m-1`$。

**第 4 条。** 两个分布下每个坐标都以概率 $`1/2`$ 取 $`\pm1`$，且 $`\mathbb E h_0(G)=0`$。在 $`P`$ 下，$`\theta_1-c=-\eta\sigma`$ 的两个坐标总是同号，所以 $`F=1`$；在 $`Q`$ 下总是异号，所以 $`F=0`$。$`\square`$

**(d) 的证明。** 固定坐标 $`i`$ 和梯度值 $`g`$。

- 若 $`g_i=0`$，则 $`h_\epsilon(g)_i=0=h_0(g)_i`$。
- 若 $`g_i\ne0`$，则 $`\lvert h_\epsilon(g_i)-\mathrm{sign}(g_i)\rvert=\lvert g_i\rvert\cdot\big\lvert\tfrac1{\lvert g_i\rvert+\epsilon}-\tfrac1{\lvert g_i\rvert}\big\rvert=\tfrac{\epsilon}{\lvert g_i\rvert+\epsilon}`$。

这个量总不超过 $`1`$，且当 $`\lvert g_i\rvert\ge a`$ 时不超过 $`\epsilon/(a+\epsilon)`$。因此

```math
\mathbb E_P\lvert h_\epsilon(G_i)-h_0(G_i)\rvert\le\beta_{P,i}(a)+\frac{\epsilon}{a+\epsilon}.
```

对点对 $`\big(c-\eta h_\epsilon(G),\,c-\eta h_0(G)\big)`$ 逐点应用 Lipschitz 假设，得

```math
\big\lvert\mathbb E_PF(c-\eta h_\epsilon(G))-\mathbb E_PF(c-\eta h_0(G))\big\rvert\le\eta\sum_iL_i\Big[\beta_{P,i}(a)+\frac{\epsilon}{a+\epsilon}\Big],
```

$`Q`$ 下同理。经由两个理想符号期望用三角不等式，即得 (A9)。

关于 TV 的部分：$`F(c-\eta h_0(G))=f(\sigma(G))`$，其中 $`f:\lbrace -1,0,1\rbrace ^p\to[0,1]`$；任意这样的 $`f`$ 都满足 $`\lvert\mathbb E_Pf-\mathbb E_Qf\rvert\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$。$`\square`$

**(e) 的证明。** $`\beta_1<1`$ 时 $`m_1=(1-\beta_1)G`$ 显然是单射。

对反例，有 $`1-\beta_1=1-\beta_2=\tfrac12`$，bias correction 在 $`t=1`$ 时为 $`\tfrac12`$，在 $`t=2`$ 时为 $`\tfrac34`$。

- *第 1 步：* $`m_1=G_1/2`$，$`v_1=G_1^2/2`$，故 $`\widehat m_1=G_1`$，$`\widehat v_1=G_1^2`$。两段历史中 $`G_1>0`$，所以 $`\theta_1=\theta_0-\eta`$。
- *第 2 步：* $`m_2=\tfrac12m_1+\tfrac12G_2=\tfrac{G_1}4-\tfrac38`$。$`G_1=1`$ 时为 $`-\tfrac18`$，$`G_1=2`$ 时为 $`+\tfrac18`$，从而 $`\widehat m_2=\tfrac43m_2=\mp\tfrac16`$。
- 同理 $`v_2=\tfrac12v_1+\tfrac12G_2^2=\tfrac{G_1^2}4+\tfrac9{32}`$，即 $`\tfrac{17}{32}`$ 或 $`\tfrac{41}{32}`$，故 $`\widehat v_2=\tfrac43v_2=\tfrac{17}{24}`$ 或 $`\tfrac{41}{24}`$。
- 由于 $`\epsilon=0`$，$`\theta_2=\theta_1-\eta\widehat m_2/\sqrt{\widehat v_2}`$，这正是 (A10)。分母为正、分子异号，两步方向相反，于是 $`F(\theta_2)=1`$、$`F(\theta_2')=0`$。

*可实现性：* $`\tfrac12(\theta-a)^2`$ 的梯度是 $`\theta-a`$。在 $`\theta_0=0`$ 处，目标 $`a=-1,-2`$ 给出 $`G_1=1,2`$。两段历史的第一步落在同一个 $`\theta_1=-\eta`$，因此共同目标 $`a=\theta_1+\tfrac34`$ 在两段历史中都给出 $`G_2=-\tfrac34`$，不需要依赖分支的未来梯度。

*正的 $`\epsilon`$：* 对同样的梯度序列，由于分母 $`\sqrt{\widehat v_2}+\epsilon`$ 为正，第二步方向对任意 $`\epsilon\ge0`$ 都是 $`-\mathrm{sign}(\widehat m_2)`$，因而方向保持相反。$`\square`$

**(e′) 的证明。** 若 $`\beta_1=0`$，则 $`m_2=G_2`$。第二步为 $`-\eta\,G_2/(1-\beta_1^2)\big/(\sqrt{\widehat v_2}+\epsilon)`$，分母为正，所以其符号为 $`-\mathrm{sign}(G_2)`$，与第一个梯度无关。若 $`\beta_1>0`$，则 $`m_2=\beta_1(1-\beta_1)G_1+(1-\beta_1)G_2`$。对 $`0<G_1<G_1'`$，选取 $`G_2<0`$ 使 $`\lvert G_2\rvert`$ 严格介于 $`\beta_1G_1`$ 与 $`\beta_1G_1'`$ 之间，$`m_2`$ 就异号；而在 $`\epsilon=0`$ 时两段历史的第一步 $`-\eta\mathrm{sign}(G_1)`$ 相同。$`\square`$

## 注记

1. **三个层次的充分性。**
   - 冷启动一步的即时参数充分性：$`\epsilon=0`$ 时需要联合符号分布，$`\epsilon>0`$ 时需要完整分布。
   - 完整状态的充分性：需要完整的消耗梯度分布，因为 $`m_1`$ 是单射。
   - 任务充分性：只需要实际 continuation 的 response class（节点 [N05](N05_continuation_response_ZH.md)），它可能小得多。

   上面精确的统计量个数是对无限制分布而言的，限定的任务族可能需要更少。
2. **$`\epsilon=0`$ 处的不连续是精确意义上的，不是操作意义上的。** 对任意可测效用，只要 $`\epsilon>0`$，梯度的大小就可辨识。但对于 Lipschitz 效用，如果梯度在零附近的质量很小，(A9) 表明符号逼近是准确的。当 $`\lvert u\rvert\to1`$ 时，逆映射 (A8) 的条件数很差。
3. **warm 状态会移动阈值。** 在 warm 状态下，第 $`i`$ 个坐标的分子在 $`G_i=-\beta_1m_i/(1-\beta_1)`$ 处变号，分母依赖于 $`v_i`$ 和 $`G_i^2`$。对标量 cross-entropy 梯度 $`G=p-Z`$，冷启动的符号概率就是 $`Z`$ 落在当前预测 $`p`$ 之下、之上和恰好等于 $`p`$ 的概率质量；允许所有阈值就能辨识 $`Z`$ 的 CDF。没有任何固定阈值或边缘符号特征能在变化的状态下都充分。
4. **对仿射理论的影响。** 两个标签分布可以在所有低于 $`N`$ 阶的矩上一致（节点 [N07](N07_affine_moment_order_ZH.md)），而在"标签落在当前预测哪一侧"的概率上不同；冷启动的 Adam 一步就在 $`\eta`$ 的一阶上把它们分开。节点 [N12](N12_optimizer_warmth_ZH.md) 定量刻画这个效应何时存在、如何随优化器变暖而消失。

## 依赖与出处

- 依赖：[N06](N06_sufficiency_duality_ZH.md)（[N06(a)](N06_sufficiency_duality_ZH.md) 的维数计数），[N10](N10_update_composition_ZH.md)（信息需求依赖于学习器）。
- 被使用于：[N12](N12_optimizer_warmth_ZH.md)。

**出处：** [v0.15 §3](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#3-what-adam-requires-three-distinct-levels-of-sufficiency) — Props. A2–A4; eqs. (A6)–(A10). v0.15 附录 [Optimizer_State_and_Few_Shot_Theory_v015_EN.md](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md) §3，Propositions A2–A4，式 (A6)–(A10)；[Proof_Review_v015_EN.md](../../docs/direction/v015_revision/Proof_Review_v015_EN.md) §V15.2。Corollary A4′ 是本页新增的，用来解释检验 V4 在 $`\beta`$ 网格上的结果。
