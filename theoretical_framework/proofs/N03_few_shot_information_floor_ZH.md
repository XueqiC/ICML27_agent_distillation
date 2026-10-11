# N03 · Few-shot 信息下限 — 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/N03_few_shot_information_floor_ZH.md)

## 设定与假设

下面所有概率都以一个固定的、事先声明的预训练信息状态 $`I_{\rm pre}`$ 为条件。额外的 teacher 反馈不计入 $`K`$ 条示范。

**假设 3.1（隐比特模型）。**

- 未知比特 $`\Theta=(\Theta_1,\dots,\Theta_d)`$ 相互独立且公平。
- 部署时 context $`j`$ 出现的权重为 $`w_j\ge0`$，$`\sum_jw_j=1`$；context $`j`$ 的正确决策是 $`\Theta_j`$。
- 学生类包含所有 context 到动作的表，所以 $`\mathcal R_{\rm rep}=0`$。效用为 $`J_\Theta(\pi)=\sum_jw_j\Pr_\pi(A_j=\Theta_j)`$，知情最优值为 1。

**假设 3.2（不泄露的示范）。**

- 第 $`k`$ 条轨迹揭示一个随机子集 $`S_k\subseteq[d]`$ 及其标签 $`\Theta_{S_k}`$。
- $`S_1,\dots,S_K`$ 独立同分布（分布为 $`L`$），且与 $`\Theta`$ 独立；同一条轨迹内揭示的坐标之间可以相关。
- transcript 中的其他内容，其条件分布只依赖于已揭示的标签和独立随机性。
- 记 $`q_j=\Pr(j\in S)`$。

**假设 3.3（硬查询预算）。** 示范之后，学习器最多可以做 $`n`$ 次无噪声、单位成本的查询，每次揭示一个自选的比特。选择可以依赖迄今观察到的一切。$`n`$ 是每条 transcript 上的硬预算，不是期望预算。

**假设 3.4（两世界自适应获取）。** 世界为 $`v\in\lbrace 0,1\rbrace`$，两个世界要求不同的最终决策。

- $`K`$ 条示范独立同分布，每条满足 $`\mathrm{KL}(P_v^{\rm demo}\Vert P_{1-v}^{\rm demo})\le\kappa_D^{(v)}<\infty`$。
- 一个公共的、依赖历史的规则选择查询，并在 $`N\le N_{\max}<\infty`$ 时停止。规则的选择、停止与私有随机化 kernel 都不依赖于 $`v`$。
- 对每个允许的查询 $`x`$ 和相关支撑上的每个前置历史 $`h`$，有 $`\mathrm{KL}(P_v(\cdot\mid h,x)\Vert P_{1-v}(\cdot\mid h,x))\le\kappa_X^{(v)}<\infty`$。
- 最终决策在每个世界中的出错概率至多为 $`\delta`$，$`0<\delta<1/2`$。

## 陈述

**(a)** **命题（信息下限）。** 在假设 3.1–3.2 下，$`K`$ 条轨迹之后可达到的最小期望遗憾为

```math
R_{\rm info}(K)=\frac12\sum_{j=1}^dw_j(1-q_j)^K, \qquad(A11)
```

约定 $`(1-q)^0=1`$。多一条轨迹的边际价值为

```math
R_{\rm info}(K)-R_{\rm info}(K+1)=\frac12\sum_jw_jq_j(1-q_j)^K. \qquad(A12)
```

**(b)** **命题（最优查询）。** 令 $`U_K`$ 为 $`K`$ 条轨迹都未揭示的比特集合，其权重按 $`w_{(1)}\ge\dots\ge w_{(\lvert U_K\rvert)}`$ 排序。在假设 3.1–3.3 下，

```math
R_{\rm info}(K,n)=\frac12\,\mathbb E\Big[\sum_{r=n+1}^{\lvert U_K\rvert}w_{(r)}\Big], \qquad(A13)
```

且查询权重最大的 $`n`$ 个未见比特是最优的；根据揭示的值调整查询没有帮助。

均匀情形（$`w_j=1/d`$，每条轨迹揭示一个均匀抽取的 context）下，记 $`M_K=\lvert U_K\rvert`$，则

```math
R_{\rm info}(K,n)=\frac{\mathbb E[(M_K-n)_+]}{2d},\qquad R_{\rm info}(K,0)=\frac12(1-1/d)^K. \qquad(A14)
```

一般地 $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$，且可以严格大于。

**(c)** **命题（示范信息抵扣查询预算）。** 在假设 3.4 下，

```math
K\kappa_D^{(v)}+\kappa_X^{(v)}\,\mathbb E_vN\ge\mathrm{kl}(1-\delta,\delta). \qquad(A15)
```

若 $`0<\kappa_X^{(v)}<\infty`$，则

```math
\mathbb E_vN\ge\frac{[\mathrm{kl}(1-\delta,\delta)-K\kappa_D^{(v)}]_+}{\kappa_X^{(v)}}. \qquad(A16)
```

若 $`\kappa_X^{(v)}=0`$ 且 $`K\kappa_D^{(v)}<\mathrm{kl}(1-\delta,\delta)`$，则任何查询预算都达不到所述误差。

## 证明

**引理 3.0（未见比特保持公平）。** 在假设 3.1–3.2 下，给定完整 transcript $`T`$，未见比特 $`(\Theta_j)_{j\in U_K}`$ 独立同分布且公平，并与 $`T`$ 及学习器的私有随机性独立。

*证明。* 联合分布可分解为

```math
\Pr(S_{1:K},\Theta,X)=\prod_kL(S_k)\cdot2^{-d}\cdot\Pr(X\mid S_{1:K},\Theta_{\cup_kS_k}),
```

其中 $`X`$ 是额外的 transcript 内容。固定观察到的 $`(S_{1:K},\Theta_{\cup S_k},X)`$，唯一依赖于 $`\Theta_{U_K}`$ 的因子是常数 $`2^{-d}`$。所以给定 $`T`$ 时，$`\Theta_{U_K}`$ 在 $`\lbrace 0,1\rbrace ^{\lvert U_K\rvert}`$ 上均匀分布。学习器的随机性与其他一切独立。$`\square`$

**(a)**

- *下界。* 固定任一学习器。对 $`j\in U_K`$，由引理 3.0，任何（可能随机的）动作 $`A_j`$ 都满足 $`\Pr(A_j=\Theta_j\mid T)=\tfrac12`$；对 $`j\notin U_K`$，成功概率至多为 1。因此条件遗憾至少为 $`\tfrac12\sum_{j\in U_K}w_j`$。
- *可达性。* 照抄每个已揭示的标签、其余随机猜测的那张表达到了这个下界，并且它在学生类中。
- *取期望。* 由于 $`S_k`$ 独立同分布，$`\Pr(j\in U_K)=\Pr(j\notin S_1)^K=(1-q_j)^K`$，于是 $`\mathbb E\sum_{j\in U_K}w_j=\sum_jw_j(1-q_j)^K`$，即 (A11)。
- *(A12)。* 相邻两项相减：$`(1-q_j)^K-(1-q_j)^{K+1}=q_j(1-q_j)^K`$。$`\square`$

**(b)**

*条件值。* 以示范 transcript 为条件，记 $`V(U,b)`$ 为未见集合为 $`U`$、剩余预算为 $`b`$ 时的最优条件遗憾。对 $`b`$ 归纳证明

```math
V(U,b)=\tfrac12\Big(\sum_{j\in U}w_j-\mathrm{top}_b(U)\Big),
```

其中 $`\mathrm{top}_b(U)`$ 是 $`U`$ 中最大的 $`\min(b,\lvert U\rvert)`$ 个权重之和。

- *基础情形。* $`b=0`$ 时就是条件形式的(a)。
- *归纳步。* $`b\ge1`$ 时，学习器可以停止、查询一个已知比特，或查询某个 $`j\in U`$。
  - 停止得到 $`V(U,0)`$，查询已知比特得到 $`V(U,b-1)`$。由归纳假设，以及 $`\mathrm{top}`$ 关于 $`b`$ 单调，二者都不小于右端。
  - 查询 $`j\in U`$ 揭示 $`\Theta_j`$。由引理 3.0，剩余未见比特仍独立同分布且公平，并与揭示出的值独立。因此无论观察到什么值，后续问题都是 $`V(U\setminus\lbrace j\rbrace ,b-1)`$，按值分支没有收益。
- *取最小。* 于是

  ```math
  V(U,b)=\min_{j\in U}\tfrac12\Big(\sum_{U\setminus\lbrace j\rbrace }w-\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\Big)=\tfrac12\Big(\sum_Uw-\max_{j\in U}\big[w_j+\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\big]\Big).
  ```

  其中的最大值等于 $`\mathrm{top}_b(U)`$，取权重最大的 $`j`$ 即达到。归纳完成，同时说明贪心规则最优。

对示范取平均 $`V(U_K,n)`$ 即得 (A13)。

*均匀情形。* 权重为 $`1/d`$ 时 $`\sum_{r>n}w_{(r)}=(M_K-n)_+/d`$，即得 (A14)。$`n=0`$ 时由(a)（$`q_j=1/d`$）得 $`R_{\rm info}(K,0)=\tfrac12(1-1/d)^K`$。由于 $`x\mapsto(x-n)_+`$ 是凸函数，Jensen 不等式给出 $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$。

*严格性。* 取 $`d=3`$、$`K=3`$、$`n=1`$：$`\Pr(M_3=0)=6/27`$，$`\Pr(M_3=1)=18/27`$，$`\Pr(M_3=2)=3/27`$，$`\mathbb EM_3=8/9<1`$。因此右端为 $`0`$，而 $`R_{\rm info}(3,1)=\Pr(M_3=2)/(2\cdot3)=1/54`$。$`\square`$

**(c)**

*补齐。* 停止后用固定的空符号把 transcript 补齐到长度 $`N_{\max}`$，记为 $`T=(O_1,\dots,O_K,(X_1,Y_1),\dots,(X_N,Y_N))`$，并附上停止指示与私有随机性。

*似然比。* 选择、停止与随机化 kernel 在两个世界中相同，在似然比中相互抵消：

```math
\log\frac{P_v(T)}{P_{1-v}(T)}=\sum_{k=1}^K\log\frac{P^{\rm demo}_v(O_k)}{P^{\rm demo}_{1-v}(O_k)}+\sum_{t=1}^{N_{\max}}1\lbrace t\le N\rbrace \log\frac{P_v(Y_t\mid H_{t-1},X_t)}{P_{1-v}(Y_t\mid H_{t-1},X_t)}.
```

有限的 KL 上界意味着在相关支撑上绝对连续，所以每一项几乎必然有限。

*在 $`\mathbb E_v`$ 下取期望。*

- 第一个和的期望为 $`K\,\mathrm{KL}(P^{\rm demo}_v\Vert P^{\rm demo}_{1-v})\le K\kappa_D^{(v)}`$。
- 第二个和中，$`\lbrace t\le N\rbrace`$ 与 $`X_t`$ 关于 $`(H_{t-1},\text{随机性})`$ 可测。逐项取条件期望，每一项的期望为 $`\mathbb E_v[1\lbrace t\le N\rbrace \,\mathrm{KL}(P_v(\cdot\mid H_{t-1},X_t)\Vert P_{1-v}(\cdot\mid H_{t-1},X_t))]\le\kappa_X^{(v)}\Pr_v(t\le N)`$。
- 由于 $`N_{\max}<\infty`$，求和是有限的；对 $`t`$ 求和得 $`\kappa_X^{(v)}\mathbb E_vN`$。

因此 $`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\le K\kappa_D^{(v)}+\kappa_X^{(v)}\mathbb E_vN`$。

*数据处理。* 记最终决策为 $`\mathsf{dec}`$，对指示 $`1\lbrace \mathsf{dec}=v\text{ 对应的正确决策}\rbrace`$ 用数据处理不等式。它在 $`v`$ 下的概率为 $`x\ge1-\delta`$，在 $`1-v`$ 下为 $`y\le\delta`$。由二元 kl 在矩形 $`x\ge1-\delta>\tfrac12>\delta\ge y`$ 上的单调性，$`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\ge\mathrm{kl}(x,y)\ge\mathrm{kl}(1-\delta,\delta)`$。两式结合即得 (A15)，移项得 (A16)。

$`\kappa_X^{(v)}=0`$ *的情形。* 此时 (A15) 左端等于 $`K\kappa_D^{(v)}`$。若它小于 $`\mathrm{kl}(1-\delta,\delta)`$，则对任何 $`N`$ 都违反 (A15)，这样的算法不存在。$`\square`$

## 注记

1. **replay 的价值为零。** 重复展示已揭示的标签不改变 $`U_K`$，因此不改变 $`R_{\rm info}`$。这是 N01 [N01(d)](N01_decomposition_and_ambiguity_ZH.md) 在本模型中的特例。不过它仍可能降低学习器自身的误差 $`\mathcal R_{\rm alg}`$。
2. **不存在只依赖 $`K`$ 的通用速率。** $`q_j=0`$ 的坐标留下不可消除的被动误差 $`\tfrac12w_j`$；$`q_j>0`$ 但很小时，收敛可以任意慢。任何关于 $`K`$ 的速率都需要对 $`(q_j)`$ 做覆盖假设。
3. **各假设的作用。** 比特独立保证查询之间互不影响；若比特相关，一次查询会提供关于另一个比特的信息，贪心结论随之改变。硬预算是必需的：只约束期望预算会允许跨 transcript 分配。不泄露的揭示机制也是必需的：按比特取值选择揭示集合，本身就会泄露信息。
4. **(A15) 的适用范围。** 这是一个二元识别下界。要得到策略学习版本，需要互不相交的近最优输出集合。右端为零时，(A16) 并不说明可达。
5. **这是下限，不是实际风险。** $`R_{\rm info}(K)`$ 是 Bayes 信息下限，实际学习器的风险可以更高（见实验页 E5）。

## 依赖与来源

- **依赖：** [N01](N01_decomposition_and_ambiguity_ZH.md)（$`\mathcal R_{\rm info}`$ 项与[N01(d)](N01_decomposition_and_ambiguity_ZH.md)）、[N02](N02_passive_vs_active_acquisition_ZH.md)（change of measure 技术）。
- **被以下节点使用：** N17（设计规则 1：先获取，再编译）。

**出处：** [v0.15 §4](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#4-make-the-number-of-demonstrations-explicit) — Props. A5–A7; eqs. (A11)–(A16). [Optimizer_State_and_Few_Shot_Theory_v015_EN.md](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md) §4，命题 A5–A7，式 (A11)–(A16)。审查：[Proof_Review_v015_EN.md](../../docs/direction/v015_revision/Proof_Review_v015_EN.md) V15.3–V15.4。
