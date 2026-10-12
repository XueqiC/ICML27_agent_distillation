# §5 Optimizer state 与一阶表示 — 实验

[← 框架总图](../README_ZH.md) · [证明](../proofs/05_optimizer_state_ZH.md) · [English](05_optimizer_state.md)

**结论.**

- **Proposition 5.1（optimizer 变换后的展开）.**
  - 精确检验在 3,712 个实例上通过，残差为 0。在 140 个多项式 continuation value 上，阶数取 $`\deg V`$ 的展开全部等于真实差距；每个截断余项都在其界之内，余项与界的最大比值为 0.275。
  - 在受控网络里，每个光滑 optimizer 的精确单元格都落在余项界之内，最大比值 0.9986。这一组是诊断性的，状态为 INFO。
  - 完整状态上的一阶项预测了 `v014:C23` 中 cold Adam 的差距：预测 0.07568，实测 0.07691。原始平均梯度项只有 $`-8.9\times10^{-11}`$。
- **Proposition 5.3（cold-start Adam）.** 五个部分全部被精确确认。
  - 在 300 个字母表上，商空间维数都是 $`s-1`$ 与 $`m-1`$。
  - margin 界被逼近到 0.994。
  - optimizer 记忆的见证对所有测试过的 $`\epsilon`$ 都给出方向相反的第二步；在 $`\beta_1>0`$ 的每个 $`(\beta_1,\beta_2)`$ 格点上都存在。
  - 在受控网络里，梯度矩相同的两个分布之间的 Adam 差距在冻结 Jacobian 后依然存在；消除符号分歧后，差距消失。
- **Proposition 5.5(b)（冻结分母误差）.** 6,944 个经对抗搜索的状态上没有违反。最好的比值是 1.0，界是紧的。
- **Proposition 5.6（warm Adam 一阶仿射）.** 常数事先固定，界在全部 2,496 对分布上成立，差距/界的最大值为 0.988。
- **Proposition 5.7（cold 步分开均值相同的反馈）.** cold 状态下的差距恰为 $`\lvert w\rvert\eta/2`$。在 warm 状态下，归一化后的差距按 $`\kappa^{0.9995}`$ 缩放。
- **Proposition 5.8（warmth 的演化）.**
  - $`s`$ 直到 $`10^6`$ 时，精确递推与闭式之差不超过 $`7.1\times10^{-13}`$。
  - 两侧界在所有采样点上都成立，上界最多比 $`\bar\kappa(s)`$ 大 1.33 倍。warmth 会到达平台 $`(1-\beta_2)/\beta_2`$，并且不再下降。
  - 在平稳的 Gaussian、重尾和稀疏梯度下，75 个 Monte Carlo 区间全部包含 $`\sigma^2(1-\beta_2^{\,s})`$。累积界在 9,000 个格点上成立。
  - 真实轨迹只覆盖早期阶段：那里 warmth 中位数的 log–log 斜率为 $`-1.06`$，lr = 0 的 warm-up 与真实步的作用相同。
- **Proposition 5.9 与 Theorem 5.10(e)（clipping）.**
  - 原始均值相同、clipped mean 不同的两个分布在一阶就分开，余项斜率为 2.0015。
  - 始终裁剪（$`p_{\rm act}=1`$）时，平均方向相同的分布一阶项恰为 0。
  - Proposition 5.9 中部分激活时的误差界只是两行的不等式，没有单独的数值检验。
- **Theorem 5.10（各学习器的一阶统计量）.** 五个见证全部通过，共 105 条断言，最差残差 $`8.0\times10^{-21}`$。
  - warm Adam：梯度均值相同时，一阶项完全由 $`v`$ 块承担，与预测相差小于 $`10^{-12}`$。
  - Lion：momentum 占主导时，参数块对消息完全不敏感。
  - LoRA：$`B=0`$ 时，零空间方向上的差别是二阶的，值域方向上的差别是一阶的。

| Check | 检验对象 | Code ID | 结果 |
|---|---|---|---|
| [Check 5.1](#c5-1) | [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)(a)、(c) | `v015:V8` | PASS |
| [Check 5.2](#c5-2) | [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)(b)、(d)；[Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(d) | `v015:E10` | INFO（诊断）；所有光滑的精确单元格都在界内 |
| [Check 5.3](#c5-3) | [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(a)–(c) | `v015:V2`、`pkg:T03`–`pkg:T06` | PASS |
| [Check 5.4](#c5-4) | [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(d) | `v015:V3`、`pkg:T07` | PASS |
| [Check 5.5](#c5-5) | [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(e) | `v015:V4`、`v015:E3`、`pkg:T08` | PASS |
| [Check 5.6](#c5-6) | 网络中的 [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(a)–(c) | `v015:E0`、`v015:E1` | PASS |
| [Check 5.7](#c5-7) | [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)：optimizer 效应还是特征学习 | `v015:E2`、`v014:C23`（Adam 设定） | INFO；冻结 Jacobian 后，45 个单元格中 39 个差距仍在 |
| [Check 5.8](#c5-8) | [Proposition 5.5](../proofs/05_optimizer_state_ZH.md#s5-5)(b) | `v015:E8`（冻结分母部分） | PASS |
| [Check 5.9](#c5-9) | [Proposition 5.6](../proofs/05_optimizer_state_ZH.md#s5-6)、[Remark 5.11](../proofs/05_optimizer_state_ZH.md#s5-11) | `v015:E8`（warm 仿射部分） | PASS |
| [Check 5.10](#c5-10) | [Proposition 5.7](../proofs/05_optimizer_state_ZH.md#s5-7) | `v015:E8`（cold 分离部分） | PASS |
| [Check 5.11](#c5-11) | [Proposition 5.8](../proofs/05_optimizer_state_ZH.md#s5-8)(a)–(c) | `v016:C1.exact`、`v016:C1.moments`、`v016:C1.accumulation` | PASS |
| [Check 5.12](#c5-12) | 真实轨迹上的 [Proposition 5.8](../proofs/05_optimizer_state_ZH.md#s5-8)(a)（早期阶段）与 (d)（warm-up） | `v015:E8`（warmth 轨迹部分） | PASS |
| [Check 5.13](#c5-13) | 整段训练中的 [Propositions 5.6](../proofs/05_optimizer_state_ZH.md#s5-6)–[5.8](../proofs/05_optimizer_state_ZH.md#s5-8) | `v015:E8`（`data.gaps`） | PASS |
| [Check 5.14](#c5-14) | [Proposition 5.9](../proofs/05_optimizer_state_ZH.md#s5-9)、[Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(e) | `v015:E11`（clipped mean） | PASS |
| [Check 5.15](#c5-15) | $`p_{\rm act}=1`$ 时的 [Proposition 5.9](../proofs/05_optimizer_state_ZH.md#s5-9)、[Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(e) | `v015:E11`（始终裁剪） | PASS |
| [Check 5.16](#c5-16) | [Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(c)、[Proposition 5.5](../proofs/05_optimizer_state_ZH.md#s5-5)(a) | `v015:E11`（warm Adam） | PASS |
| [Check 5.17](#c5-17) | [Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(f) | `v015:E11`（Lion） | PASS |
| [Check 5.18](#c5-18) | [Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(g) | `v015:E11`（LoRA） | PASS |

**状态标签.**

- `PASS` 表示实现的一致性断言全部通过。
- `INFO` 表示经验诊断：每个单元格单独标 PASS、FAIL 或 UNAVAILABLE，整组不作通过与否的判断。

除非另有说明，网络都是 `v014:C23` 的受控小网络：one-hot 上下文、4 个 tanh 单元、sigmoid 输出、平均 cross-entropy、float64。这些都不是真实模型。

<a id="c5-1"></a>

## Check 5.1 · 展开式与统计量个数的精确有限检验（Proposition 5.1(a)、(c)）

**结论.** 展开式在有理数运算下精确成立，每个截断余项都在其界之内。一阶统计量的个数也是精确的：少保留一个统计量，就总能找到一对特征一致、但一阶项不同的分布。共 3,712 个实例、4,978 条断言，最差残差为 0。

**动机.**

- [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)(a) 是对两个分布积分后的 Taylor 恒等式。如果张量矩的记账、$`V(z)`$ 的抵消或余项常数有误，就会出现某个实例，其差距与截断展开之差超过界。
- (c) 断言投影后的平均增量恰好是最小的一阶统计量。如果计数有误，会出现两种情形之一：在所声称的维数下，有特征一致、但一阶项非零的分布对；或者有多余的统计量。

**设置.**

- *多项式.*
  - $`D=2,\dots,5`$，每个 $`D`$ 重复 35 次，次数在 $`1,\dots,4`$ 之间循环。
  - 系数取自 $`\lbrace -1,-2/3,\dots,1\rbrace`$，首项单项式系数为 1。
  - 每个实例有一个随机有理状态 $`z`$、六个有理增量，以及其上的两个随机分布 $`P,Q`$。
  - 全部使用精确运算（`Fraction`）。
- *余项常数.* $`M_{k+1}`$ 取包含 $`z`$ 与 $`z+\Delta`$ 的盒子上的系数和优函数。这是一个有理上界，不是拟合出来的常数。
- *Logistic.*
  - 在 $`\mathbb R^3`$ 上取 $`V=\sigma(q)`$，$`q(z)=\tfrac12z^\top Az+b^\top z`$，配六个增量和随机分布。
  - Hessian certificate 为 $`M_2=L^2/(6\sqrt3)+\lVert A\rVert_{\rm op}/4`$，其中 $`L=\max_{\rm vertices}\lVert Az+b\rVert_2`$。
  - 2,200 个随机实例，另加 12 次重启 × 100 步爬山搜索。
- *秩.*
  - 160 个面板，字母表大小 $`m\in\lbrace 4,\dots,9\rbrace`$，增量维数 $`D\in\lbrace 2,\dots,5\rbrace`$。
  - 增量是随机整数，梯度方向随机并张成 $`U`$。
  - 秩精确计算（SymPy），见证用 LP 求得（HiGHS）。
- 种子 `SeedSequence([150150, 8])`。

**做法.**

1. 对每个多项式，计算精确差距以及 $`j`$ 阶项 $`\langle\nabla^jV(z),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle/j!`$。
   - 断言 $`k=\deg V`$ 时这些项之和等于差距；
   - 断言对每个 $`k<\deg V`$，余项都在界之内。
2. 对 logistic 族，断言 $`\lvert\text{gap}-\text{first order}\rvert\le\tfrac{M_2}2(m_2^P+m_2^Q)`$，再用对抗搜索最大化这一比值。
3. 对每个秩面板：
   - 计算 $`d_0^{(1)}=\mathrm{rank}[\mathbf 1,\mathcal G^{(1)}]-1`$，并与中心化后的秩核对；
   - 断言 $`d_0^{(1)}\le\min\lbrace\dim U,m-1\rbrace`$；
   - 核对：落在已保留统计量零空间中的扰动，其 $`U`$ 投影为零，所有一阶项都不变；
   - 只保留 $`d_0^{(1)}-1`$ 个统计量，用 LP 求一对特征一致的分布，断言某个一阶项不同。

**结果**（`theory/v015_checks/results_v015_independent.json`，`v015:V8`）.

| 量 | 值 |
|---|---:|
| 状态 / 实例 / 断言 / 最差残差 | PASS / 3,712 / 4,978 / 0 |
| $`\deg V`$ 阶展开精确成立的多项式 | 140 / 140 |
| 在界内的截断余项 | 344 / 344；余项/界最大为 0.275 |
| logistic：$`\lvert\text{gap}-\text{first order}\rvert`$/界的最大值 | 0.183（差距 $`-0.0231`$，一阶项 $`-0.2307`$，界 1.137，$`M_2=0.823`$） |
| 满足 $`d_0^{(1)}\le\min\lbrace\dim U,m-1\rbrace`$ 的秩面板 | 160 / 160 |
| 保留 $`d_0^{(1)}-1`$ 个统计量时有 LP 见证的面板 | 158（其余 2 个 $`d_0^{(1)}=0`$）；最小的分离目标值 0.113 |

**为什么支持 Proposition 5.1.**

- (a) 与 (c) 都是有限的陈述，这里在精确运算下检验。
- 对一个光滑的非多项式 continuation value，余项界成立，并且留有余量。
- 统计量个数是最小的：去掉任何一个，都会剩下两个分布，它们在所有保留的统计量上一致，却在一阶上不同。

<a id="c5-2"></a>

## Check 5.2 · 受控网络中、跨 optimizer 的整段训练表示（Proposition 5.1(b)、(d)）

**结论.**

- 在整段训练上，两个反馈律之间的差距等于*完整状态增量*的一阶项，加上一个落在曲率界之内的余项。在 continuation value 光滑的每个精确单元格里都是如此，最大比值 0.9986。
- 在 `v014:C23` 的 cold Adam 单元格中，增量的一阶项解释了 98.4% 的差距；原始平均梯度项只有 $`-8.9\times10^{-11}`$。
- 余项的阶数符合 (d) 的矩层级。例外有两类：多轮 Adam，其二阶矩块使余项变成 $`\eta`$ 的一阶量；以及光滑性假设之外的单元格。

**动机.** 在真实网络里，这一表示可能以三种方式失效：

- 一阶项解释不了差距；
- 余项超过界；
- $`\eta\to0`$ 时余项不缩小。

本检验把三者都测一遍，并与使用原始平均梯度的朴素一阶项对比。

**设置.**

- *分布.* `v014:C23` 的标签分布：moment $`N=2,3,4`$ 与 parity $`N=2,3`$，它们在低于 $`N`$ 阶的标签矩上一致；另加在线性 logit 模型上实现的 clipped mean 见证。
- *11 种 optimizer.*
  - SGD；
  - heavy-ball 与 Nesterov（$`\beta=0.9`$）；
  - cold Adam（$`\beta=(0.9,0.999)`$，$`\epsilon=10^{-8}`$，带 bias correction）；
  - 先做 $`W=5`$ 和 $`W=20`$ 步 lr = 0 moment warm-up 之后的 warm Adam；
  - AdamW（$`\lambda=0.01`$）；
  - clipped SGD 与 clipped Adam；
  - signSGD；
  - Lion（$`\beta_1=0.9`$，$`\beta_2=0.99`$）。
- *裁剪阈值.*
  - 按任务选取，使被裁剪的概率质量最接近 50%。
  - 梯度范数相同时达不到 30%–70% 的区间：moment $`N=3`$、parity $`N=3`$ 为 25%。
- *网格.* $`\eta\in\lbrace 10^{-3},3\cdot10^{-3},10^{-2},3\cdot10^{-2},10^{-1}\rbrace`$，$`R\in\lbrace 1,4,40\rbrace`$，共 990 个单元格。
- 种子 `SeedSequence([150150, 110])`。

**做法.** 对每个单元格：

1. 计算差距 $`L(P)-L(Q)`$：
   - $`R=1,4`$ 时，完整枚举历史，精确计算；
   - $`R=40`$ 时，用独立运行估计，并给出标准误。
2. 在完整状态 $`(\theta,m,v)`$ 上计算一阶项 $`\sum_t\mathbb E_{d_t^Q}\langle\nabla V^P_{t+1}(Z_t),\mu^P_t-\mu^Q_t\rangle`$。
   - $`\nabla V^P`$ 由对精确枚举的 $`P`$-continuation 做 Richardson 外推的中心差分得到；
   - 同时记录微分误差。
3. 计算朴素项：把 optimizer 当作用当前 Adam 分母做预条件的 SGD，使用原始平均梯度。
4. 用采样和爬山搜索得到线段曲率，估计 $`M_2`$（观察到差距之后从不放大），并计算余项界。
5. 记录 $`\lvert\text{gap}-\text{first order}\rvert`$/界，以及关于 $`\eta`$ 的 log–log 斜率。

**结果 1：精确单元格中的界**（`v015:E10`，`data.optimizer_summary`、`data.cells`）.

| Optimizer | 精确单元格 | 精确比值最大值 | 超出界的精确单元格 |
|---|---:|---:|---:|
| SGD | 60 | 0.9986 | 0 |
| heavy-ball | 60 | 0.9984 | 0 |
| Nesterov | 60 | 0.9980 | 0 |
| cold Adam | 60 | 0.9981 | 0 |
| warm Adam，$`W=5`$ | 60 | 0.3339 | 0 |
| warm Adam，$`W=20`$ | 60 | 0.8432 | 0 |
| AdamW | 60 | 0.9817 | 0 |
| clipped Adam | 55 个有限（5 个导数非有限） | 0.9982 | 0 |
| signSGD | 60 | 0.9981 | 0 |
| Lion | 60 | 0.9981 | 0 |
| clipped SGD | 60 | 1.2453 | 1（多步裁剪 kink） |

**结果 2：`v014:C23` 中 cold Adam 的差距是增量的一阶效应**（`data.C23_prediction`；moment $`N=3`$，$`\eta=0.1`$，$`R=1`$）.

| 量 | 值 |
|---|---:|
| 实测差距 $`L(P)-L(Q)`$ | 0.076907 |
| 完整状态增量的一阶项 | 0.075683 |
| 朴素一阶项（原始平均梯度） | $`-8.87\times10^{-11}`$ |
| 余项 $`\lvert\text{gap}-\text{first order}\rvert`$ | 0.001225 |
| 余项界（$`\hat M_2=0.342`$） | 0.04447 |

- 两个标签分布均值相同，所以任何基于平均梯度的统计量都看不出它们的区别。cold Adam 的增量 $`-\eta h_\epsilon(G)`$ 却能区分它们（[Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(a)）。这正是 [Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10) 中 cold 步那一行 (d)。
- 在同一单元格中，signSGD 和 Lion 给出相同的差距与一阶项（0.07691 与 0.07568），因为它们的第一步是同一个符号映射。
- clipped SGD 的两个量都是 0.00303，而它的原始平均梯度项为 0。

**结果 3：余项的阶数**（$`\lvert\text{gap}-\text{first order}\rvert`$ 关于 $`\eta`$ 的 log–log 斜率，按任务列出；`data.optimizer_summary[*].slopes`；余项落在数值下限的单元格没有斜率）.

| Optimizer | $`R=1`$ | $`R=4`$ |
|---|---|---|
| SGD、heavy-ball、Nesterov | $`N=2`$ 与 clipped 见证上为 1.98–2.0；moment、parity $`N=3`$ 上为 3.0 | $`N=2`$ 与 clipped 见证上为 1.95–1.99；$`N=3`$ 上为 2.91–2.99；Nesterov 在 moment $`N=4`$ 上为 3.98 |
| clipped SGD、signSGD、Lion | 2.0；$`N=3`$ 上为 3.0 | 2.0–2.09；$`N=3`$ 上为 2.71–3.01 |
| cold Adam | 2.0–3.0 | 0.99（moment $`N=3`$）、1.0（clipped 见证）；其余任务 1.96–2.98 |
| AdamW | 2.0–3.0 | 0.99（moment $`N=3`$）、1.31（moment $`N=4`$）、1.0（clipped 见证）；其余任务 1.98–2.98 |
| clipped Adam | 2.0–3.0 | 0.91（moment $`N=3`$）、1.04（clipped 见证）；其余任务 2.0–2.93 |
| warm Adam，$`W=5`$ | 1.99–2.98 | 1.11–1.17（moment $`N=2,3,4`$）、1.0（clipped 见证）；2.03 与 2.92（parity） |
| warm Adam，$`W=20`$ | 2.0–3.0 | 1.0（moment $`N=3`$）、1.16（moment $`N=4`$）、1.0（clipped 见证）；3.04（moment $`N=2`$）、1.99 与 3.0（parity） |

**结果 4：多轮 Adam 的差距中一阶项占多少**（moment $`N=3`$，$`R=4`$；`data.cells`）.

| Optimizer | $`\eta=10^{-3}`$：差距 / 一阶项 | $`\eta=10^{-1}`$：差距 / 一阶项 | $`\eta=10^{-3}`$ 时的余项绝对值 |
|---|---|---|---:|
| cold Adam | $`1.90\times10^{-3}`$ / $`1.24\times10^{-3}`$ | 0.1196 / 0.0810 | $`6.6\times10^{-4}`$ |
| warm Adam，$`W=5`$ | $`4.38\times10^{-4}`$ / $`2.22\times10^{-4}`$ | 0.0663 / 0.0365 | $`2.2\times10^{-4}`$ |
| warm Adam，$`W=20`$ | $`4.44\times10^{-5}`$ / $`2.01\times10^{-5}`$ | 0.0044 / 0.0020 | $`2.4\times10^{-5}`$ |
| clipped SGD | $`1.21\times10^{-4}`$ / $`1.21\times10^{-4}`$ | 0.0115 / 0.0115 | $`2.2\times10^{-11}`$ |

**结果 5：光滑性假设的边界.**

| 位置 | 现象 |
|---|---|
| clipped SGD，clipped 见证，$`\eta=0.1`$，$`R=4`$ | 唯一超出界的精确单元格：比值 1.245（差距 0.04476，一阶项 0.04637，界 0.00129）。后续的裁剪判断使 continuation value 不光滑，Proposition 5.1 的光滑性假设不成立。 |
| $`R=40`$（330 个 Monte Carlo 单元格） | 不计标准误时，25 个单元格超出点估计的界；计入四个标准误后仍有 2 个，都是 sign 类：signSGD 在 moment $`N=4`$、$`\eta=0.01`$，以及 Lion 在 clipped 见证、$`\eta=10^{-3}`$。 |
| clipped Adam，moment $`N=4`$ | 10 个单元格在 cold 状态的零梯度坐标上中心差分非有限，记为不可用。 |

![Figure 5.1：各 optimizer 下的一阶表示](../figures/fig5_1_linear_response.png)

*Figure 5.1.*

- (A) 对 SGD 和 sign 类 optimizer，一阶表示的余项按 $`\eta^2`$ 缩小；两阶增量矩一致时按 $`\eta^3`$ 缩小。多轮 Adam 的余项只按 $`\eta`$ 缩小。
- (B) 在 cold Adam 单元格里，optimizer 增量的一阶项解释了差距，原始平均梯度则完全看不出差别。
- (C) 在光滑性假设成立的每个精确单元格中，差距减去一阶项都在余项界之内。唯一的超出是多步 clipped SGD。

[PDF](../figures/fig5_1_linear_response.pdf)

**为什么支持 Proposition 5.1.**

- 只要 continuation value 光滑，差距就等于 optimizer 增量的一阶项，加上一个在所述界之内的余项。
- 增量才是正确的对象。对均值相同的两个分布，原始平均梯度什么也预测不了，增量却预测出 cold Adam 差距的 98%。
- 余项的阶数符合 (d) 的矩层级：
  - SGD、momentum、clipping 与 sign 类 optimizer 为 $`O(\eta^2)`$；
  - 两阶增量矩一致时为 $`O(\eta^3)`$。
- 对多轮的 Adam 类 optimizer，余项是 $`\eta`$ 的一阶量，来源是 [Theorem 7.1](../proofs/07_second_moment_channel_ZH.md#s7-1) 中的二阶矩通道。此时在 moment 任务上，一阶项只占差距的 45%–68%；从 cold 状态到 $`W=20`$ 步 warm-up，余项绝对值下降 27 倍。
  - 所以一阶排序必须满足 [Corollary 8.9](../proofs/08_verification_ZH.md#s8-9) 的 margin 条件；二阶矩通道占主导时，要改用有限替换测量（[Theorem 8.8](../proofs/08_verification_ZH.md#s8-8)）。
- 超出界的单元格全部落在光滑性假设之外：多步 clipped SGD，以及 sign 类 optimizer 的 Monte Carlo 单元格。

<a id="c5-3"></a>

## Check 5.3 · Cold 映射、商空间的精确维数与逆映射的条件数（Proposition 5.3(a)–(c)）

**结论.** 在 300 个随机有限梯度字母表上：

- cold Adam 的一步与 $`c-\eta h_\epsilon(G)`$ 相差不超过 $`4.4\times10^{-16}`$；
- $`\epsilon=0`$ 时恰好需要且只需要 $`s-1`$ 个统计量，$`\epsilon>0`$ 时为 $`m-1`$ 个；
- 逆映射的导数可达 $`10^{22}`$。

**动机.** [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(a)–(c) 给出三个断言，每一个都是可以精确核对的有限陈述：

- 第一步的精确公式；
- $`\epsilon=0`$ 时，立即参数所含的信息是联合符号分布（维数 $`s-1`$）；$`\epsilon>0`$ 时是完整分布（维数 $`m-1`$）；
- 逐坐标的符号边缘分布不够。

**设置.**

- 300 个随机有限梯度字母表，坐标数 $`p=1,\dots,4`$，包含零坐标。
- 对 (a) 使用独立的递推实现，用 float64 核对；平方根精确时也用精确有理数核对。
- 秩与零空间精确计算。特征少于 $`s-1`$ 个时，用 HiGHS LP 构造被分开的分布。
- 在 $`u\to1`$ 的网格上做逆映射往返，$`\epsilon=1/100`$。

**做法.**

- 把第一步与 $`c-\eta h_\epsilon(G)`$ 比较。
- 统计可达的符号模式数 $`s`$，核对 $`s-1`$ 个模式指示量已经足够。
- 对任何更小的特征集，给出一个零质量扰动，以及它分开的两个分布。
- 在 $`\epsilon>0`$ 下重复，维数变为 $`m-1`$。
- 计算联合分布与边缘分布的对照例子。

**结果**（`v015:V2`）.

| 量 | 值 |
|---|---|
| 状态、实例、断言、最差残差 | PASS、300、3,288、$`4.44\times10^{-16}`$ |
| $`\epsilon=0`$ 时商空间维数 $`=s-1`$ | 300/300 个字母表（1 至 14 个原子，$`s`$ 从 1 到 14） |
| $`\epsilon>0`$ 时维数 $`=m-1`$ | 300/300 |
| 分离的例子（`quotients[0]`） | 4 个原子、4 个模式、2 个特征：$`P=(\tfrac14,\tfrac18,\tfrac38,\tfrac14)`$ 与 $`Q=(\tfrac14,\tfrac38,\tfrac18,\tfrac14)`$ 在两个特征上一致，模式指示量却相差 $`-\tfrac14`$ |
| 逆映射的导数，$`\epsilon=1/100`$ | $`u=0.9`$：1；$`u=0.999`$：$`10^4`$；$`u=1-10^{-12}`$：$`10^{22}`$ |
| 联合与边缘 | 边缘分布相同；同号效用 1 对 0 |

另一套独立实现的检验（`docs/direction/v015_revision/results_v015.json`，`checks`），全部 PASS：

- `pkg:T03`：480 个情形，更新误差 $`\le2.22\times10^{-16}`$。
- `pkg:T04`：25 个原子、9 个符号模式，统计量维数 8 与 24。
- `pkg:T05`：平均更新相同，同号效用差 1.0。
- `pkg:T06`：在 0–2 阶上一致的两个标签分布。SGD 的平均参数差为 0，理想符号 Adam 的差为 $`1/10`$，有限 $`\epsilon`$ 时为 $`250/2703`$。

**为什么支持 Proposition 5.3(a)–(c).**

- cold Adam 一步的精确信息内容：$`\epsilon=0`$ 时是联合符号商，$`\epsilon>0`$ 时是完整的梯度分布；个数恰为 $`s-1`$ 与 $`m-1`$。
- $`\epsilon>0`$ 时逆映射存在，但条件数极差，所以实际上幅度信息几乎看不见。
- `pkg:T06` 在均值层面给出同样的结论：SGD 在一阶上分不开的两个分布，Adam 能分开。

<a id="c5-4"></a>

## Check 5.4 · 对 margin 敏感的近似（Proposition 5.3(d)）

**结论.** margin 界在 1,660 个精确实例上成立。对抗搜索得到的差距/界最大为 0.994，界几乎是紧的。

**动机.** (d) 刻画了：对正则的效用，符号商在什么时候是好的*近似*描述。它连接两件事：精确陈述（任何 $`\epsilon>0`$ 都能识别幅度），以及实际情形。

**设置.**

- 随机的有限梯度分布 $`P,Q`$。
- 逐坐标 Lipschitz 的效用 $`F\in[0,1]`$：由已知 $`L_i`$ 的分段线性坐标函数求和再截断得到。
- 随机的 margin $`a`$ 与随机的 $`\epsilon`$。
- 全程精确有理数运算。
- 对 $`(G,P,Q,L,a,\epsilon,\eta)`$ 做带重启的对抗爬山搜索。

**做法.** 在每个实例上断言 margin 界成立，且 $`\Delta_{\rm sign}\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$；然后最大化差距/界。

**结果**（`v015:V3`）.

| 量 | 值 |
|---|---|
| 状态、实例、断言、最差残差 | PASS、1,660、3,320、0 |
| 差距/界最大值，对抗搜索 | 0.9941 |
| 差距/界最大值，随机 | 0.837 |
| 找到的最大 $`\Delta_{\rm sign}/\mathrm{TV}`$ | 0.274 |

`pkg:T07`：1,500 个实例，差距/界最大 0.739，没有违反。

**为什么支持 Proposition 5.3(d).** 界成立，并且几乎被达到。对正则效用，符号商是准确的，误差只来自两项：接近零的梯度质量，以及 $`\epsilon/(a+\epsilon)`$ 项。

<a id="c5-5"></a>

## Check 5.5 · 携带幅度的 optimizer 记忆（Proposition 5.3(e)）

**结论.** 见证是精确的：

- $`\widehat m_2=\mp\tfrac16`$，$`\widehat v_2=\tfrac{17}{24},\tfrac{41}{24}`$，所以两条历史的第二步方向相反。
- $`\epsilon`$ 从 0 到 $`10^{-1}`$ 都保持相反。
- 在 $`\beta_1>0`$ 的每个 $`(\beta_1,\beta_2)`$ 格点上都存在，在 $`\beta_1=0`$ 的格点上都不存在。

同一对历史在网络的 optimizer 代码中复现，网络中也找到了见证。

**动机.** (e) 断言：只匹配第一步的参数信息，对一段学习 continuation 并不够，因为 $`m_1`$ 保留了幅度。要让这个反例有说服力，它必须是精确的、能由真实损失实现、对 $`\epsilon`$ 稳健，并且出现在实际的 optimizer 代码里。

**设置.**

- *精确部分（`v015:V4`）.*
  - 用 Fraction 与符号运算计算梯度历史 $`(1,-\tfrac34)`$ 与 $`(2,-\tfrac34)`$，$`\beta_1=\beta_2=\tfrac12`$。
  - 平方损失的实现。
  - $`\epsilon\in\lbrace 0,10^{-1},\dots,10^{-12}\rbrace`$。
  - $`5\times5`$ 网格 $`\beta_1,\beta_2\in\lbrace 0,\tfrac1{10},\tfrac12,\tfrac9{10},\dots\rbrace`$，每格搜索 300 次。
- *网络部分（`v015:E3`）.*
  - 同一对历史经过 Check 5.6–5.7 所用的标量 optimizer，$`\epsilon=0`$ 与 $`10^{-8}`$。
  - 用软标签搜索：符号历史相同、第二步更新相反的情形。

**做法.**

- 计算所有矩与两个第二步，核对两步符号相反。
- 在网络中搜索标签对 $`(y_1,y_2)`$ 与 $`(y_1',y_2)`$：第一步相同，第二步在某个坐标上相反。

**结果**（`v015:V4`、`v015:E3`）.

| 量 | 值 |
|---|---|
| `v015:V4` 状态、实例 | PASS、39 |
| 精确矩 | $`m_2=-\tfrac18,+\tfrac18`$；$`\widehat m_2=-\tfrac16,+\tfrac16`$；$`\widehat v_2=\tfrac{17}{24},\tfrac{41}{24}`$ |
| $`\eta=1`$ 时的第二步 | $`+\sqrt{102}/51=+0.198`$ 与 $`-\sqrt{246}/123=-0.128`$ |
| 平方损失的实现 | 第一步目标 $`-1`$ 与 $`-2`$；第二步共同目标 $`-\tfrac14`$ |
| 正 $`\epsilon`$ 下方向相反 | 从 $`0`$ 到 $`10^{-1}`$ 的全部 14 个取值 |
| 有见证的 $`\beta`$ 格点 | 20/25：所有 $`\beta_1>0`$ 的格点，$`\beta_1=0`$ 的格点一个也没有 |
| `v015:E3` 标量轨迹，$`\epsilon=0`$ 与 $`10^{-8}`$ | 第二步更新 $`+0.19803`$ 与 $`-0.12752`$ |
| `v015:E3` 网络见证 | 在 $`\epsilon=0`$ 与 $`10^{-8}`$ 下都找到：标签 $`(0,0.7)`$ 与 $`(0.1,0.7)`$ 的第一步相同，第二步在输出偏置上方向相反 |

`pkg:T08`：第一步输出相同，符号历史相同。$`\eta=0.1`$ 时第二步增量为 $`+0.019803`$ 与 $`-0.012752`$，阈值任务的差距为 1。

**为什么支持 Proposition 5.3(e).**

- 反例是精确的、可实现的、对 $`\epsilon`$ 稳健，并且出现在真实网络中。
- $`\beta`$ 网格把机制分离了出来：幅度记忆由一阶矩 $`m`$ 携带，所以这一效应恰好在 $`\beta_1>0`$ 时存在。

<a id="c5-6"></a>

## Check 5.6 · 网络中的 Adam 差距就是 cold-start 的符号商（Proposition 5.3(a)–(c)）

**结论.**

- 两个标签分布的 0–2 阶矩一致，梯度矩也在 0–2 阶上一致；但它们期望的第一步 Adam 更新在每个坐标上相差 $`\eta`$。$`\eta=0.1`$ 时，Adam 的回报差距为 0.0769（$`R=1`$）与 0.1196（$`R=4`$）。
- 消除符号分歧后差距完全消失；做五步 warm 的矩更新后，差距降到 0.0034 与 0.0098。

**动机.** 在受控网络里，对低于 $`N`$ 阶的矩都一致的标签分布，Adam 能分开，SGD 不能。[Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(a)–(c) 预言了机制：cold Adam 响应的是梯度 $`G=p-Z`$ 在当前预测周围的符号分布，而不是矩。

- `v015:E0` 精确复现这一差距。
- `v015:E1` 追踪整条链：输入矩 → 梯度矩 → 归一化后的更新 → 回报；然后逐个消融各因素。

**设置.**

- Adam，$`\beta=(0.9,0.999)`$，$`\epsilon=10^{-8}`$ 放在平方根外，启用 bias correction。
- 分布：`law_scalar(N)` 只在 $`1,\dots,N-1`$ 阶标签矩上一致。
- 差距：$`\mathbb E_P[p_R^N]-\mathbb E_Q[p_R^N]`$，完整枚举有限历史（标准误为 0）。
- 消融（moment $`N=3`$，$`\eta=0.1`$）：
  - $`\epsilon`$ 放进平方根内；
  - $`\epsilon\in\lbrace 10^{-6},10^{-4},10^{-2},10^{-1}\rbrace`$；
  - 不做 bias correction；
  - 先在第三个分布上做五步 warm 的矩更新；
  - 全局裁剪 0.1；
  - sum reduction；
  - 平移支撑，使所有第一步梯度同号。

**做法.**

- 枚举每个支撑点上的梯度、梯度矩、有限 $`\epsilon`$ 下的更新及其期望。
- 与实际执行的网络更新比较，相对容差 $`10^{-12}`$。
- 每个消融都从同一个快照重新运行。

**结果**（`v015:E0`、`v015:E1`、`theory/v015_checks/c23_bundle.json`）.

| 复现（moment $`N=3`$） | $`\eta=0.01`$ | $`\eta=0.1`$ |
|---|---|---|
| Adam 差距，$`R=1`$ | 0.00757 | 0.07691 |
| Adam 差距，$`R=4`$ | 0.01892 | 0.11961 |
| 仿射 SGD 对照，$`R=1`$ / $`R=4`$ | $`-6.3\times10^{-9}`$ / $`-2.5\times10^{-8}`$ | $`-6.3\times10^{-6}`$ / $`-2.1\times10^{-5}`$ |

moment $`N=3`$、$`\eta=0.1`$ 时的链条（`data.transitions`）：

- 1、2、3 阶输入矩之差：$`0,0,-0.0556`$。
- 联合梯度矩：2 阶及以下相同（误差 $`3\times10^{-17}`$）；3 阶相差 0.056。
- 期望的第一步更新：每个坐标 $`+0.05`$ 对 $`-0.05`$，相差 $`\eta`$。
- 回报之差：0.0769。
- 在 $`P`$ 下，标签以概率 $`\tfrac14`$ 取 0、以概率 $`\tfrac34`$ 取 $`\tfrac23`$，所以输出偏置的梯度 $`p-Z`$ 为正的概率是 $`\tfrac14`$。
- 在 $`Q`$ 下，标签以概率 $`\tfrac34`$ 取 $`\tfrac13`$、以概率 $`\tfrac14`$ 取 1，同一梯度为正的概率是 $`\tfrac34`$。
- 实测映射误差：相对 $`2.1\times10^{-15}`$。

| 消融（moment $`N=3`$，$`\eta=0.1`$） | 差距 $`R=1`$ | 差距 $`R=4`$ |
|---|---|---|
| 基线 | 0.0769 | 0.1196 |
| $`\epsilon`$ 在平方根内；$`\epsilon=10^{-6}`$；sum reduction | 0.0769 | 0.1196 |
| $`\epsilon=10^{-4}`$ | 0.0767 | 0.1197 |
| $`\epsilon=10^{-2}`$ | 0.0610 | 0.1148 |
| $`\epsilon=10^{-1}`$ | 0.0211 | 0.0446 |
| 不做 bias correction | 0.2477 | $`-0.0081`$ |
| 五步 warm 的矩更新 | **0.0034** | **0.0098** |
| 全局裁剪 0.1 | 0.0769 | 0.1840 |
| 平移支撑（第一步梯度同号） | **0.0000** | **0.0000** |

平移后的支撑保留了匹配的标签矩：1 阶与 2 阶仍然相等。

**为什么支持 Proposition 5.3.**

- 两个梯度矩在 2 阶及以下完全相同的分布，第一步 Adam 更新却在每个坐标上相差 $`\eta`$。原因正是符号商：标签落在当前预测之上的概率，在一个分布下是 $`1/4`$，在另一个下是 $`3/4`$。
- 去掉符号分歧，差距就消失；把矩预热，差距就大部分消失，与 [Proposition 5.7](../proofs/05_optimizer_state_ZH.md#s5-7) 和 [Proposition 5.6](../proofs/05_optimizer_state_ZH.md#s5-6) 的预言一致。
- 在 $`\epsilon`$ 增大到与梯度幅度相当（$`10^{-2}`$）之前，差距对 $`\epsilon`$ 的取法不敏感，与 (d) 的预言一致。

<a id="c5-7"></a>

## Check 5.7 · 是 optimizer 效应，不是特征学习（Proposition 5.3）

**结论.**

- 冻结 Jacobian 后 Adam 无法学习特征，却几乎精确地复现了差距：$`R=1`$ 时 0.0757 对 0.0769，$`R=4`$ 时 0.1225 对 0.1196。冻结 Jacobian 的差距在 45 个单元格中的 39 个里依然存在。
- 在声明的 Adam 设定中，切线仿射 SGD 对照在 42 个配置里有 32 个没能解释差距。

**动机.** 与仿射 SGD 对照之间的差距，可能来自 optimizer，可能来自非线性特征，也可能两者都有。冻结网络 Jacobian、同时保留 sigmoid、cross-entropy 和同一个 Adam 递推，就能把 optimizer 的作用单独分离出来。

**设置**（`v015:E2`）.

- 从同一个快照出发的五个分支，同一来源内的历史完全相同：
  - 实际网络 + Adam；
  - 冻结 Jacobian，$`\mathrm{logit}(\theta)=\mathrm{logit}(\theta_0)+J_0(\theta-\theta_0)`$，配 Adam 及其自身的矩；
  - 实际网络 + SGD；
  - 冻结 Jacobian + SGD；
  - 切线仿射 SGD。
- 网格：5 个任务 × $`\eta\in\lbrace 0.01,0.1,1\rbrace`$ × $`R\in\lbrace 1,4,40\rbrace`$，共 45 个单元格。
- $`J_0`$ 的导数核对残差 $`\le2.6\times10^{-11}`$。
- 冻结 Adam 的差距绝对值超过 $`\max(10^{-10},3\,\mathrm{SE})`$ 时，记为*依然存在*。

**结果**（`v015:E2`，`data.comparisons`；状态 INFO）.

| 单元格 | 实际 Adam | 冻结 Jacobian 的 Adam | 实际 SGD | 切线 SGD |
|---|---|---|---|---|
| moment $`N=3`$，$`\eta=0.1`$，$`R=1`$ | 0.0769 | 0.0757 | 0.0000 | 0.0000 |
| moment $`N=3`$，$`\eta=0.1`$，$`R=4`$ | 0.1196 | 0.1225 | 0.0000 | 0.0000 |
| moment $`N=3`$，$`\eta=0.01`$，$`R=4`$ | 0.0189 | 0.0188 | 0.0000 | 0.0000 |
| parity $`N=2`$，$`\eta=0.1`$，$`R=4`$ | 0.0762 | 0.0774 | 0.0034 | 0.0034 |
| moment $`N=4`$，$`\eta=0.1`$，$`R=4`$ | $`-0.0218`$ | $`-0.0223`$ | $`-0.0000`$ | 0.0000 |

- 冻结 Jacobian 的 Adam 差距在 45 个单元格中的 39 个里依然存在。
- 在 $`\eta\le0.1`$、$`R\le4`$ 的所有单元格中，三个 SGD 分支彼此相差不超过 $`2.8\times10^{-4}`$，且都不超过 0.0034。

Adam 下的仿射对照，取自 `theory/independent_checks/results_independent_v014.json` 的 `controlled_neural.details.observed_affine_comparison_cohorts`。设定为 $`\beta=(0.9,0.999)`$、$`\epsilon=10^{-8}`$、带 bias correction、$`\eta\in\lbrace 0.01,0.1\rbrace`$、$`R\in\lbrace 1,4,40\rbrace`$，共 7 个任务分布：

| 组 | 配置数 | 一致 | 偏离 | 最大偏差 |
|---|---:|---:|---:|---|
| Adam，全部声明设定 | 42 | 10 | 32 | 0.1196（moment $`N=3`$，$`\eta=0.1`$，$`R=4`$） |

![Figure 5.2：cold-start Adam](../figures/fig5_2_cold_adam.png)

*Figure 5.2.*

- (A) 两个分布的前两阶梯度矩一致，cold Adam 仍把它们分开；冻结 Jacobian 的 Adam 复现了这一差距，SGD 则没有差距。
- (B) 消除符号分歧（平移支撑）后差距消失；矩预热后差距大部分消失。
- (C) 两条符号序列完全相同的历史，第二步方向相反，因为 $`m`$ 保留了幅度。

[PDF](../figures/fig5_2_cold_adam.pdf)

**为什么支持 Proposition 5.3.**

- 这一差距是 optimizer state 的效应，与命题所述一致。
- 在这个网络里，矩匹配（[Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)）能控制 SGD，控制不了 cold Adam。仿射 SGD 预测遗漏的部分，就是 (a)–(c) 中的 cold-start 商，加上 (e) 中的 momentum 记忆。

<a id="c5-8"></a>

## Check 5.8 · 冻结分母误差（Proposition 5.5(b)）

**结论.** 在 6,944 个经对抗搜索的 optimizer 状态上，$`\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\,u_i\min\lbrace\kappa_i/2,1\rbrace`$ 没有被违反。最好的比值是 1.0000000000000002，即在浮点舍入意义下取等号，所以界不能再改进一个常数因子。

**动机.** (b) 是唯一一处把真实的 Adam 分母换成冻结分母的地方。界只要在某处不成立，[Proposition 5.6](../proofs/05_optimizer_state_ZH.md#s5-6) 也就随之不成立。

**设置**（`v015:E8`，冻结分母部分；种子 `[150150,108]`）.

- 随机状态：
  - $`v`$ 在 $`[10^{-12},1]`$ 上 log-uniform；
  - $`m`$、梯度与 $`t\in\lbrace 1,\dots,2000\rbrace`$ 随机；
  - 五组 $`(\beta_1,\beta_2)`$，包括 $`(0.9,0.999)`$。
- 搜索：24 次随机重启 × 80 步爬山，最大化“左边/右边”。
- 差 $`\theta^+-\tilde\theta^+`$ 用有理化的精确运算计算，避免两个舍入后的更新相减时出现抵消；同时保留直接的浮点减法结果。
- 容差：只允许 $`10^{-12}`$ 的相对松弛。

**结果**（`v015:E8`，冻结分母部分）.

| 量 | 值 |
|---|---|
| 核对的状态 | 6,944 |
| 违反 | 0 |
| 最好的“左边/右边” | $`1.0000000000000002`$；见证落在饱和分支 $`\kappa\gg1`$ |

**为什么支持 Proposition 5.5(b).**

- 界成立，其饱和分支能被达到。
- $`\kappa\to0`$ 时第一个分支也渐近地紧，因为 $`(1-(1+\kappa)^{-1/2})/(\kappa/2)\to1`$。

<a id="c5-9"></a>

## Check 5.9 · Warm Adam 关于 consumed gradient 一阶仿射（Proposition 5.6、Remark 5.11）

**结论.** 在受控网络的实际 warm 状态上，常数在评估任何一对分布之前就已固定。界对全部 2,496 对分布成立：差距/界最大为 0.988，逐点残差/界最大为 0.998。

**动机.** [Proposition 5.6](../proofs/05_optimizer_state_ZH.md#s5-6) 在 warm 状态下给出 response 之差的界：

- 平均 consumed gradient 相同的两个消息分布，response 之差不超过 $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$；
- 均值不同时，不超过 $`\lvert b^\top\Delta\mathbb EG\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r`$。

**设置**（`v015:E8`，warm 仿射部分）.

- 状态：网络的实际 warm 状态，$`t`$ 在 21 到 200 之间。
- 可训练坐标：输出头的 1–4 个坐标，其余固定。
- 消息：每个面板 2–6 条。
- 边界与尾段：
  - 一次实际的 Adam 更新作为边界；
  - 之后是长度 $`R\in\lbrace 0,2,6\rbrace`$ 的 IID 参考尾段，标签全 0 或全 1，各占概率 $`1/2`$；
  - 全部 $`2^R`$ 条尾段历史精确枚举。
- $`\eta\in\lbrace 0.01,0.1\rbrace`$；部署取 moment（$`N=3`$）或 parity（$`N=2`$）回报。
- 常数在评估任何一对分布*之前*，用预先声明的中心差分估计，之后从不放大：
  - $`L_i`$ 沿实际状态到 surrogate 的线段估计；
  - $`L_v`$ 沿二阶矩线段估计，$`\bar k=1`$；
  - $`H_G`$ 取径向与凸包样本上 Hessian 元素绝对值之和。
- $`b`$ 由解析的前向敏感度给出，并与独立的有限差分核对。
- 分布对：随机与对抗构造，均值相同与不同的都有。

**结果**（`v015:E8`，warm 仿射部分）.

| 量 | 值 |
|---|---|
| 面板 × 分布对 | 24 × 104 = 2,496 |
| 违反两分布界的对 | 0 |
| 全部分布对上差距/界的最大值 | 0.988 |
| 逐点 $`\lvert\psi-c-b^\top G\rvert/r`$ 的最大值 | 0.998 |
| 解析 $`b`$ 与有限差分 | 每个面板上一致到 $`\le10^{-8}`$ |
| 被测状态的 warmth（对坐标与消息取最大值） | 每个面板 $`\max\kappa\le0.081`$ |
| 曲率常数与步长 | 例如 moment、$`p=1`$、$`R=0`$：$`\eta=0.01`$ 时 $`H_G=2.1\times10^{-6}`$，$`\eta=0.1`$ 时 $`1.3\times10^{-4}`$ |

[Figure 5.3](#c5-11) 的 (C) 画出了逐点残差与其界。

**为什么支持 Proposition 5.6.**

- 在实际网络状态上，跨不同尾段长度与步长，界都成立，并且几乎被达到。
- 曲率常数大致按 $`\eta^2`$ 增长，与界中曲率项的预期一致。
- warmth 条件是按对坐标与消息的最大值核对的，不是按分位数，这正是 [Remark 5.11](../proofs/05_optimizer_state_ZH.md#s5-11) 要求的形式。
- 证据的局限：常数是估计出来的，所以这是经验上的一致，不是 certificate。有效性由证明给出；这一检验确认了所列的残差项就是起作用的那几项。

<a id="c5-10"></a>

## Check 5.10 · Cold 状态的分离与 warm 状态的缩放（Proposition 5.7）

**结论.**

- cold 状态下，均值相同的两个分布之间的差距恰为 $`\lvert w\rvert\eta/2`$。例如 $`\eta=0.01`$、$`w=-0.7`$ 时为 $`7/2000`$，$`\eta=0.1`$、$`w=1`$ 时为 $`1/20`$。
- warm 状态下，按预条件步长尺度归一化后的差距按 $`\kappa^{0.9995}`$ 增长。

**动机.**

- [Proposition 5.7](../proofs/05_optimizer_state_ZH.md#s5-7) 说，cold 状态在 $`\eta`$ 的一阶上就分开均值相同的分布。
- [Proposition 5.6](../proofs/05_optimizer_state_ZH.md#s5-6) 说，在 warm 状态下，同样的分离只有 warmth 项 $`\eta\,u\,\kappa`$ 那么大。

**设置**（`v015:E8`，cold 分离部分）.

- cold：命题中的精确有理分布，$`\eta\in\lbrace 0.01,0.1\rbrace`$，$`w\in\lbrace -0.7,1\rbrace`$，分别用理想符号映射和 $`\epsilon=10^{-8}`$ 计算。
- warm：同样（不缩放）的分布，取 $`t=1000`$、$`m=0`$；扫描 $`v`$，使 $`\kappa_{\max}`$ 在 $`[10^{-6},10^2]`$ 上取 33 个对数等距值。

**结果**（`v015:E8`，cold 分离部分）.

| 量 | 值 |
|---|---|
| cold 差距，$`\eta=0.01`$，$`w=-0.7`$ | 理想值 $`7/2000=0.0035`$；$`\epsilon=10^{-8}`$ 时 $`0.00349999986`$ |
| cold 差距，$`\eta=0.1`$，$`w=1`$ | 理想值 $`1/20`$；$`\epsilon=10^{-8}`$ 时 $`0.0499999980`$ |
| warm：原始差距对 $`\kappa`$ 的 log–log 斜率 | 1.4995 |
| warm：按预条件步长尺度归一化后的斜率 | 0.9995 |

**为什么支持 Proposition 5.7.**

- cold 差距恰为 $`\lvert w\rvert\eta/2`$，是 $`\eta`$ 的一阶量。
- 在 warm 状态下，改变 $`v`$ 会同时改变 $`\kappa`$ 和步长尺度 $`u\propto\kappa^{1/2}`$。所以原始差距按 $`\kappa^{3/2}`$ 缩放，归一化后的差距按 $`\kappa^{1}`$ 缩放，正是 warmth 项 $`\eta\,u\,\kappa`$ 的形式。

<a id="c5-11"></a>

## Check 5.11 · warmth 的精确规律：两侧界、平台与累积（Proposition 5.8(a)–(c)）

**结论.**

- **精确递推.** 梯度为常数时，二阶矩的原始递推与闭式 $`\bar\kappa(s)=(1-\beta_2)/(\beta_2(1-\beta_2^{\,s}))`$ 的相对误差不超过 $`7.1\times10^{-13}`$，覆盖五个 $`\beta_2`$、$`s`$ 直到 $`10^6`$。
- **两侧界与平台.** (a) 的两侧界在全部 10,900 个采样点上成立，上界最多比 $`\bar\kappa`$ 大 1.33 倍。warmth 到达平台 $`(1-\beta_2)/\beta_2`$ 后停在那里。
- **bias correction.** 它精确地不改变 warmth 比值，即 (b)。
- **平稳随机梯度.** 75 个 Monte Carlo 99.9% 区间全部包含 $`\sigma^2(1-\beta_2^{\,s})`$。
- **累积.** (c) 的界在全部 9,000 个格点上成立，其中的线性项不可缺少：一旦 $`s(1-\beta_2)`$ 达到 1 的量级，任何纯倒数形式的包络 $`c/(W+s)`$ 都不再成立。

**动机.**

- (a) 给出 mean-field warmth 的精确规律和两侧界。核心内容是一个非零的平台：只有在 $`s(1-\beta_2)\ll1`$ 时 warmth 才按 $`1/(\beta_2s)`$ 下降；$`\beta_2`$ 固定时，它不会趋于零。
- (c) 把这一点变成带线性项 $`c(1-\beta_2)R`$ 的累积界。
- 指数或常数只要错一处，就会在某个时刻出现违反的界，或出现落在区间之外的 Monte Carlo 均值。

**设置**（`theory/v016_checks`；CPU；固定的 `SeedSequence` 种子）.

- `v016:C1.exact`：
  - $`\beta_2\in\lbrace 0.9,0.99,0.995,0.999,0.9999\rbrace`$，常数梯度 1.7，原始 $`v_0=0`$；
  - 每个 $`\beta_2`$ 用因果 IIR 滤波执行一百万步更新；
  - 采样点为 $`s=1,\dots,2000`$ 的每一步，加上直到 $`10^6`$ 的 180 个对数等距点（每个 $`\beta_2`$ 共 2,180 个）；
  - 闭式用数值稳定的 `expm1` 计算；
  - bias correction 在 $`t=1,7,50`$ 处用精确有理数比较。
- `v016:C1.moments`：
  - 16,000 个独立副本，检查点 $`s\in\lbrace 1,10,100,1000,2000\rbrace`$；
  - 梯度：单位 Gaussian；缩放到单位二阶矩的 Student $`t_3`$；Bernoulli(0.05) × Gaussian$`/\sqrt{0.05}`$；
  - 区间：Gaussian 与稀疏分布用 Studentized 99.9% 区间。$`t_3`$ 梯度的平方方差无穷，因此把 $`g^2`$ 截断在 100，用有界变量的 Bernstein 区间，再在区间中心加上解析已知的尾部期望。
- `v016:C1.accumulation`：
  - $`W\in\lbrace 1,4,32\rbrace`$，$`c\in\lbrace 1,2,4\rbrace`$，$`R=1,\dots,100`$ 的每一步，加上直到 $`10^5`$ 的对数网格；
  - 直接对截断后的包络求和，它对所有容许的 warmth 序列都是上界；
  - 对 $`c\in\lbrace 1,1/\beta_2,2\rbrace`$，记录 $`\bar\kappa\le c/(W+s)`$ 首次不成立的时刻。
- 一个非平稳诊断（在假设之外，不作断言）：二阶矩按 $`e^{-s/400}`$ 衰减，$`\beta_2=0.999`$。

**结果**（`theory/v016_checks/results_v016.json`；claim 为 `v016:C1.exact`、`v016:C1.moments`、`v016:C1.accumulation`）.

| Claim id | 状态 | 实例 | 断言 | 最差残差 |
|---|---|---:|---:|---|
| `v016:C1.exact` | PASS | 6 | 32,715 | 递推相对误差 $`7.1\times10^{-13}`$（容差 $`10^{-10}`$）；下界最多违反 $`2.2\times10^{-16}`$，即舍入误差（容差 $`2\times10^{-13}`$）；bias correction 精确为 0 |
| `v016:C1.moments` | PASS | 75 | 75 | 没有目标值落在 99.9% 区间之外 |
| `v016:C1.accumulation` | PASS | 135 | 9,045 | lhs − rhs 最大为 $`-4.7\times10^{-4}`$ |

按 $`\beta_2`$ 列出的精确规律（`v016:C1.exact`）：

| $`\beta_2`$ | 平台 $`(1-\beta_2)/\beta_2`$ | 上界 / $`\bar\kappa`$ 的最大值 |
|---|---:|---:|
| 0.9 | 0.1111 | 1.324 |
| 0.99 | 0.01010 | 1.301 |
| 0.995 | 0.005025 | 1.300 |
| 0.999 | 0.001001 | 1.299 |
| 0.9999 | 0.0001000 | 1.298 |

上界在过渡区最松，位置在 $`s(1-\beta_2)\approx1.8`$（$`\beta_2=0.999`$ 时为 $`s=1791`$）。$`\beta_2\to1`$ 时这个比值趋于 $`\max_y(1+y)(1-e^{-y})/y\approx1.298`$。

$`\beta_2=0.999`$ 时的取值（来自原始数组）：

| $`s`$ | 1 | 10 | 100 | 1,000 | $`10^6`$ |
|---|---:|---:|---:|---:|---:|
| $`\bar\kappa(s)`$ | 1.001 | 0.1006 | 0.01051 | 0.001583 | 0.001001 |
| $`1/(\beta_2s)`$ | 1.001 | 0.1001 | 0.01001 | 0.001001 | $`1.0\times10^{-6}`$ |

在 $`s=10^6`$ 处，倒数规律小了一千倍，而 $`\bar\kappa`$ 就等于平台值。

平稳随机梯度（`v016:C1.moments`；每个分布族 25 个单元格）：

| 分布族 | 区间方法 | 均值的最大相对偏差 | 落在 99.9% 区间内的目标 |
|---|---|---:|---:|
| Gaussian | Studentized | 0.76% | 25/25 |
| Student $`t_3`$（单位二阶矩） | 截断 $`g^2`$ 的 Bernstein 区间 + 解析尾部 | 5.6%（$`s=1`$，$`\beta_2=0.9`$） | 25/25 |
| 稀疏 Gaussian（$`p=0.05`$） | Studentized | 2.4% | 25/25 |

累积（`v016:C1.accumulation`）：

- 在每个格点上，界 $`c\log(1+R/W)+c(1-\beta_2)R`$ 都不小于包络之和。$`R=10^5`$ 时 lhs/rhs 在 0.965 与 0.999998 之间，界几乎是紧的。
- **线性项不可缺少.** 取倒数包络 $`\bar\kappa\le c/(W+s)`$，$`c=2\ge1/\beta_2`$，从实际的原始时刻起算：
  - 它首次不成立的原始时刻是 14（$`\beta_2=0.9`$）、158（0.99）、317（0.995）、1,592（0.999）、15,935（0.9999），即 $`s(1-\beta_2)\approx1.6`$ 附近；
  - $`c<2`$ 时，它在 burn-in 之后的第一步就不成立。

非平稳诊断（$`\beta_2=0.999`$，二阶矩为 $`e^{-s/400}`$）：

- 在 $`s=2991`$，精确的 $`\mathbb E v_s`$ 为 0.0331；把当前的二阶矩代入平稳公式，得到的是 0.00054。
- 也就是说，$`v`$ 跟不上梯度变小的速度，warmth 比平稳规律预言的小约 60 倍。
- 这在 (a) 的假设之外，不作断言。

![Figure 5.3：optimizer warmth](../figures/fig5_3_warmth.png)

*Figure 5.3.*

- (A) 在短视野（$`R=1,4`$）下，lr = 0 的 moment warm-up 消除了矩相同的两个来源之间的差距。$`R=40`$ 时它不再能把差距压到零；在残差最大的地方，SGD 对照也不为零。
- (B) 四个 $`\beta_2`$ 下的精确 mean-field warmth $`\bar\kappa(s)`$，带两侧界（阴影）与平台 $`(1-\beta_2)/\beta_2`$（虚线）。只有在早期阶段 $`s(1-\beta_2)\ll1`$，warmth 才按 $`1/(\beta_2s)`$ 下降；之后趋于平稳，不会消失。网络轨迹上实测的 warmth 中位数（Check 5.12）落在早期阶段。
- (C) warm Adam 仿射界的残差从不超过界 $`r`$，并逼近到 0.2% 以内。

[PDF](../figures/fig5_3_warmth.pdf)

**为什么支持 Proposition 5.8.**

- (a) 精确成立，包括平台。常数梯度是 mean-field 公式取等号的情形。
- 两侧界在过渡区相差不超过 1.33 倍，在两端的极限下都是精确的。
- 对重尾和稀疏的平稳梯度，mean-field 规律同样成立，说明它只依赖二阶矩。
- (c) 成立，其线性项 $`c(1-\beta_2)R`$ 不能去掉。因此在 $`\beta_2`$ 固定时，warm Adam 相对冻结分母描述，每一步都有 $`c(1-\beta_2)`$ 量级的非仿射下限。
- 证据的局限：梯度非平稳时，平稳公式不适用；梯度变小时，warmth 下降得比它预言的更快。

<a id="c5-12"></a>

## Check 5.12 · 真实轨迹上的 warmth：早期阶段，以及 warm-up 与真实步作用相同（Proposition 5.8(a)、(d)）

**结论.**

- 网络轨迹在 $`\beta_2=0.999`$ 下最多吸收 129 次梯度评估，即 $`s(1-\beta_2)\le0.13`$，所以只覆盖 Proposition 5.8(a) 的早期阶段。
- 在这一阶段，$`\eta=0.01`$、$`W=0`$ 时，9 条拟合的任务/来源序列上 warmth 中位数的 log–log 斜率中位数为 $`-1.06`$（$`R^2`$ 中位数 0.999）。这正是 (a) 在 $`s(1-\beta_2)\ll1`$ 时预言的 $`1/(\beta_2s)`$ 行为。
- warmth 只取决于吸收的梯度评估总数，与这些评估来自 lr = 0 的 warm-up 还是真实更新无关，正如 (d) 所述。

**动机.**

- Check 5.11 用合成梯度检验 mean-field 规律。这里要问的是：在实际 Adam 轨迹的逐坐标 warmth 上，能否看到早期阶段的行为？这些轨迹的梯度既不平稳，在各坐标间也不同分布。
- lr = 0 的 moment warm-up 是让状态变 warm 的低成本办法；只有当 $`v`$ 在两种情形下按同样方式推进时，它才能与真实步互换。

**设置**（`v015:E8`，warmth 轨迹部分）.

- 网络上的实际 Adam 轨迹：moment 与 parity 任务，$`\eta\in\lbrace 0.01,0.1\rbrace`$，128 个副本，$`\beta_2=0.999`$。
- warm-up：在参考分布上做 $`W\in\lbrace 0,1,2,5,10,20,50\rbrace`$ 步 lr = 0 的更新；断言每一步 warm-up 都不改变参数的字节。
- 记录：每一步逐坐标 warmth 的分位数。
- 拟合：对 $`s=1,\dots,79`$ 步之前的更新，拟合中位数 $`=c/(W+s)`$，log–log 斜率不固定。

**结果**（`v015:E8`，warmth 轨迹部分）.

| 设定 | 斜率中位数 | $`R^2`$ 中位数 |
|---|---|---|
| $`\eta=0.01`$，$`W=0`$ | $`-1.06`$ | 0.999 |
| $`\eta=0.01`$，$`W\in\lbrace 5,10,20,50\rbrace`$ | $`-1.07`$ 至 $`-1.14`$ | 0.95–0.99 |
| $`\eta=0.1`$，$`W=0`$ | $`-1.18`$ | 0.94 |
| $`\eta=0.1`$，$`W\in\lbrace 10,20,50\rbrace`$ | $`-1.55`$ 至 $`-1.93`$ | 0.55–0.68 |

可互换性（moment $`N=3`$，$`\eta=0.01`$，来源 $`P`$）：先前评估次数 $`W+s`$ 相同时的 $`\kappa`$ 中位数。

| $`W+s`$ | $`W=0`$ | $`W=5`$ | $`W=20`$ | $`W=50`$ | mean-field $`\bar\kappa(W+s)`$ |
|---|---|---|---|---|---|
| 51 | 0.00661 | 0.00642 | 0.00619 | 0.00619 | 0.0201 |
| 79 | 0.00405 | 0.00395 | 0.00413 | 0.00379 | 0.0132 |

**为什么支持 Proposition 5.8(a)、(d).**

- 在早期阶段，实测 warmth 中位数的斜率与 (a) 的预言一致。
- 中位数比 $`\bar\kappa`$ 低一个大致恒定的因子（$`W+s=51`$ 时为 0.33，79 时为 0.31）。这是预期之中的：$`\bar\kappa`$ 是均值之比，而平方梯度与其滑动平均之比的逐坐标中位数更小。
- $`\eta=0.1`$ 时，consumed gradient 随训练收敛而变小，warmth 下降得比倒数规律更快，与 Check 5.11 的非平稳诊断一致。
- 对 warmth 而言，warm-up 评估与真实评估可以互换，这就是 (d)。
- 证据的局限：
  - 轨迹太短，到不了平台 $`(1-\beta_2)/\beta_2=0.001`$，平台大约从 $`s\approx1/(1-\beta_2)=1000`$ 开始。平台由 Check 5.11 检验。
  - 分位数描述的是大部分坐标的情况。Proposition 5.6 与 5.8(c) 的假设针对的是最大值或敏感度加权和（[Remark 5.11](../proofs/05_optimizer_state_ZH.md#s5-11)），最大值由 Check 5.9 核对。

<a id="c5-13"></a>

## Check 5.13 · 整段训练中的 cold 窗口（Propositions 5.6–5.8）

**结论.**

- 在短视野下，矩相同的两个来源之间的差距是 cold 窗口的效应。$`W=10`$ 的 lr = 0 moment warm-up 消除了这一差距的中位数 99%（$`R=1`$）与 97.5%（$`R=4`$）。moment $`N=3`$、$`\eta=0.1`$ 时，差距在 $`R=1`$ 从 0.0769 降到 0.0004，在 $`R=4`$ 从 0.1196 降到 0.0016。
- $`R=40`$ 时，$`W\ge10`$ 仍留下至多 0.049 的差距；在残差最大的地方，SGD 对照也不为零（至多 0.011）。

**动机.** Propositions 5.6–5.8 对整段训练有两个预言：

- 受 warmth 控制的那部分差距集中在最初几步，warm-up 能把它消除；
- 曲率部分不依赖 warmth，会随视野累积。

**设置**（`v015:E8`，`data.gaps`）.

- 不加改动地复用 Check 5.7 的分支（实际 Adam、冻结 Jacobian 的 Adam、切线仿射 SGD）。
- 网格：5 个任务（moment $`N=2,3,4`$；parity $`N=2,3`$）× $`\eta\in\lbrace 0.01,0.1\rbrace`$ × $`R\in\lbrace 1,4,40\rbrace`$ × $`W\in\lbrace 0,1,2,5,10,20,50\rbrace`$，共 210 个单元格。
- 不同 $`W`$ 与不同分支使用完全相同的历史。历史不超过 4,096 条时精确枚举，否则每个来源独立运行 2,000 次。
- warm-up 使用参考分布上的一段声明好的 IID 前缀。

**结果**（`v015:E8`，`data.gaps`）.

| 视野 | $`W=0`$ 时的最大差距 | $`W\ge10`$ 时的最大差距 | $`\lvert\text{gap}(W{=}10)\rvert/\lvert\text{gap}(W{=}0)\rvert`$ | SGD 对照的最大值 |
|---|---|---|---|---|
| $`R=1`$ | 0.0769 | 0.0006 | 中位数 0.009，最大 0.034（差距 $`>10^{-3}`$ 的 5 个单元格） | 0.0034（$`R\le4`$） |
| $`R=4`$ | 0.1196 | 0.0064 | 中位数 0.025，最大 0.226（7 个单元格） | （同上） |
| $`R=40`$ | 0.0436 | 0.0487 | 没有压到零 | 0.0110 |

| 单元格 | $`W=0`$ | $`W=1`$ | $`W=2`$ | $`W=5`$ | $`W=10`$ | $`W=20`$ | $`W=50`$ | 切线 SGD |
|---|---|---|---|---|---|---|---|---|
| moment $`N=3`$，$`\eta=0.1`$，$`R=1`$ | 0.0769 | 0.0460 | 0.0311 | 0.0032 | 0.0004 | 0.0004 | 0.0002 | 0.0000 |
| moment $`N=3`$，$`\eta=0.1`$，$`R=4`$ | 0.1196 | 0.1016 | 0.0881 | 0.0206 | 0.0016 | 0.0037 | 0.0033 | 0.0000 |
| parity $`N=2`$，$`\eta=0.1`$，$`R=40`$ | 0.0436 | 0.0477 | 0.0441 | 0.0488 | 0.0487 | 0.0391 | 0.0391 | 0.0110 |

- $`W\in\lbrace 1,2\rbrace`$ 时 warmth 仍是 1 的量级，Proposition 5.6 给不出控制，差距可以朝任一方向变化。例如 moment $`N=2`$、$`\eta=0.1`$、$`R=1`$，在 $`W=1`$ 时从 0.0175 变为 $`-0.0194`$。
- $`W\ge10`$ 时，短视野下的差距一致地很小。

**为什么支持 Propositions 5.6–5.8.**

- 在短视野下，矩相同的两个来源之间的差距就是 Proposition 5.7 中的 cold 步分离；把状态预热就能消除它，与 Propositions 5.6、5.8 的预言一致。
- 在长视野下，warm-up 不再能把差距压到零。最大的残差出现在 $`\eta=0.1`$ 的 parity $`N=2`$ 与 moment $`N=2`$，对每个 $`W\ge10`$ 分别停在 0.039–0.049 与 0.005–0.036；这些单元格里 SGD 对照也不为零（0.011 与 0.0067）。
  - 这部分残差是在 40 步更新上累积的曲率与轨迹项，SGD 也有，不是 cold-start 效应。
- 在最初几步里，warmth 决定了反馈来源必须匹配哪个统计量；视野变长之后，剩下的是轨迹本身的非线性。
- warm-up 是否改变低成本 proxy 的预测价值，是方法层面的问题，见 [§9 实验](09_mechanism_and_method_ZH.md)。

<a id="c5-14"></a>

## Check 5.14 · Clipping：统计量是 clipped mean（Proposition 5.9、Theorem 5.10(e)）

**结论.** 两个梯度分布的原始均值相同，clipped mean 相差 $`-\tfrac12`$。它们在一阶上就被分开：预测的差距 $`\tfrac\eta2\sigma'(0.4)`$ 与实测差距吻合，余项的 log–log 斜率为 2.0015。

**动机.**

- [Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(e) 说，有 clipping 时，反馈必须匹配 $`\mathbb E\,\mathrm{clip}_c(G)`$，而不是 $`\mathbb E\,G`$。
- 如果 clipping 只改变幅度，那么原始均值相同的两个分布在一阶上就会一致。所以原始均值相同、一阶上却有差距，就确认了 clipped mean 才是统计量。
- 两个分布的激活情况不同：$`P`$ 总是被裁剪，$`Q`$ 从不被裁剪。因此 [Proposition 5.9](../proofs/05_optimizer_state_ZH.md#s5-9) 中按事件拆分的两个分支都被用到。

**设置**（`v015:E11`，clipped mean）.

- 线性 logit 的 sigmoid，$`\theta=0.4`$。
- $`P(G=4)=\tfrac14`$，$`P(G=-\tfrac43)=\tfrac34`$；$`Q=\delta_0`$。
- 在 $`c=1`$ 处裁剪；clipped SGD。
- $`\eta\in\lbrace 10^{-3},3\cdot10^{-3},10^{-2},3\cdot10^{-2},10^{-1}\rbrace`$。
- 梯度 $`4`$ 与 $`-4/3`$ 在固定网络的初始化处不会出现，所以改用线性 logit 模型来实现。

**做法.**

- 断言两者原始均值相同、clipped mean 相差 $`-\tfrac12`$。
- 计算精确差距 $`(P-Q)\sigma(\theta+\Delta\theta)`$ 与预测的一阶差距 $`\eta\,\sigma'(0.4)/2`$。
- 断言解析的二阶界 $`\lvert\text{gap}-\text{first order}\rvert\le\eta^2/(12\sqrt3)`$。

**结果**（`v015:E11`，`data.clipped_mean`）.

| $`\eta`$ | 实测差距 | 一阶预测 | 余项 |
|---:|---:|---:|---:|
| 0.001 | $`1.20107\times10^{-4}`$ | $`1.20130\times10^{-4}`$ | $`2.37\times10^{-8}`$ |
| 0.01 | $`1.19892\times10^{-3}`$ | $`1.20130\times10^{-3}`$ | $`2.38\times10^{-6}`$ |
| 0.1 | $`1.17675\times10^{-2}`$ | $`1.20130\times10^{-2}`$ | $`2.46\times10^{-4}`$ |

余项的 log–log 斜率为 2.0015。

![Figure 5.4：各学习器的一阶统计量](../figures/fig5_4_learner_statistics.png)

*Figure 5.4.* 学习器特定的一阶统计量的五个见证（Check 5.14–5.18）：

- (A) 原始均值相同、clipped mean 不同，有 clipping 时出现一阶差距。
- (B) 始终裁剪时，平均方向相同的两个分布只有 $`O(\eta^2)`$ 的差距。
- (C) warm Adam、梯度均值相同时，一阶项完全由 $`v`$ 块承担，与预测一致。
- (D) momentum 占主导的 Lion：参数增量与消息无关。
- (E) 初始化处的 LoRA：零空间中的差别是二阶的，值域中的差别是一阶的。

[PDF](../figures/fig5_4_learner_statistics.pdf)

**为什么支持 Theorem 5.10(e).** 原始平均梯度预测的差距为零；clipped mean 则在一阶上预测出实测差距，余项为 $`O(\eta^2)`$。

<a id="c5-15"></a>

## Check 5.15 · 始终裁剪：只有方向起作用（完全激活时的 Proposition 5.9、Theorem 5.10(e)）

**结论.** 每个梯度都被裁剪、且两个分布的平均单位方向相同时，一阶项恰为 0，差距是二阶的（log–log 斜率 1.99997）。

**动机.** 始终裁剪（$`p_{\rm act}=1`$）时，[Proposition 5.9](../proofs/05_optimizer_state_ZH.md#s5-9) 的误差界 $`c(1-p_{\rm act})`$ 为 0，clipped mean 恰好等于 $`c\,\mathbb E[G/\lVert G\rVert]`$。所以平均方向相同的两个分布，不论幅度如何，在一阶上都必须一致。

**设置**（`v015:E11`，始终裁剪）.

- 二维梯度，$`\lVert G\rVert\ge2>c=1`$。
- $`P`$ 在 $`\lbrace(\pm2,0)\rbrace`$ 上均匀，$`Q`$ 在 $`\lbrace(0,\pm4)\rbrace`$ 上均匀。两者的平均单位方向都是 0，方向的二阶矩不同。
- 线性 logit 模型与 $`\eta`$ 网格同 Check 5.14。

**做法.** 断言一阶项恰为零，并拟合精确差距的斜率。

**结果**（`v015:E11`，`data.always_active`）.

| $`\eta`$ | 一阶项 | 实测差距 |
|---:|---:|---:|
| 0.001 | 0 | $`7.11\times10^{-8}`$ |
| 0.01 | 0 | $`7.11\times10^{-6}`$ |
| 0.1 | 0 | $`7.06\times10^{-4}`$ |

差距的 log–log 斜率为 1.99997。

**为什么支持 Proposition 5.9 与 Theorem 5.10(e).**

- 始终裁剪时，幅度在一阶上不起作用；只有方向的二阶矩把这两个分布分开。
- 如果两个分布的方向分布也相同，它们的增量分布就完全一致。
- 证据的局限：$`0<p_{\rm act}<1`$ 时，Proposition 5.9 用 $`c(1-p_{\rm act})`$ 界定“只有方向起作用”这一简化离真实值有多远；这一中间情形没有专门的数值检验。

<a id="c5-16"></a>

## Check 5.16 · Warm Adam：二阶矩经由 optimizer state 在一阶上进入（Theorem 5.10(c)、Proposition 5.5(a)）

**结论.**

- 梯度均值相同时，参数块与 $`m`$ 块的贡献恰为 0。$`v`$ 块贡献 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle=2.504\times10^{-6}`$，与预测相差小于 $`10^{-12}`$。
- 所以对 warm Adam，均值相同的两个分布在一阶上并不等价。

**动机.**

- [Proposition 5.5](../proofs/05_optimizer_state_ZH.md#s5-5)(a) 说，Adam 的精确一阶统计量是完整状态上的平均 Adam 增量。它是梯度分布的非线性泛函，并通过 $`\Delta v=(1-\beta_2)(G\odot G-v)`$ 依赖 $`\mathbb E\,G\odot G`$。
- 因此，均值相同、二阶矩不同的两个分布，应当有非零的一阶项，由 $`v`$ 块承担，权重为 $`(1-\beta_2)\nabla_vV`$。
- 在这个见证里，参数块恰好为零，冻结分母误差也恰好为零，所以检验单独分离出了 [Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(c) 中的 $`v`$ 通道。

**设置**（`v015:E11`，warm Adam）.

- 网络取 $`k=1`$ 个上下文、$`p=13`$ 个参数。
- warm 状态：$`m=0`$，每个坐标 $`v=0.02`$，步数计数 20。
  - 这是一个容许的 warm 状态，不是从 IID warm-up 中抽出来的；
  - 参数项为零依赖于标签分布的对称性。
- Adam，$`\beta=(0.9,0.999)`$，$`\eta=0.03`$。
- 标签：$`P`$ 在 $`\lbrace0,1\rbrace`$ 上均匀，$`Q`$ 在 $`\lbrace\tfrac14,\tfrac34\rbrace`$ 上均匀。两者均值都是 $`\tfrac12`$，二阶矩不同。
- continuation：之后用标签 $`\tfrac14`$ 更新两步，再部署 moment $`N=3`$。
- continuation value 对完整状态的梯度用中心差分计算。

**做法.**

1. 计算 $`\delta_1=\mathbb E_PG-\mathbb E_QG`$、$`\delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}`$，以及三个块的贡献 $`\langle\nabla_\theta V,\mu^P_\theta-\mu^Q_\theta\rangle`$、$`\langle\nabla_mV,\cdot\rangle`$、$`\langle\nabla_vV,\cdot\rangle`$。
2. 断言 $`v`$ 块等于 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle`$。
3. 与实测差距对比，并与 [Proposition 5.6](../proofs/05_optimizer_state_ZH.md#s5-6) 中的 $`L_v\kappa`$ 项对比。

**结果**（`v015:E11`，`data.warm_Adam`）.

| 量 | 值 |
|---|---:|
| $`\max_i\lvert\delta_{1,i}\rvert`$（梯度均值相同） | 0 |
| $`\max_i\delta_{2,i}`$ | 0.1875 |
| 一阶各块 $`(\theta,\ m,\ v)`$ | $`(0,\ 0,\ 2.5043\times10^{-6})`$ |
| 预测的 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle`$ | $`2.5043\times10^{-6}`$（相差 $`<10^{-12}`$） |
| 实测差距 | $`8.753\times10^{-6}`$ |
| warm Adam 界对线性 $`v`$ 项给出的 $`L_v\max\kappa`$ | $`7.127\times10^{-6}`$（$`\max\kappa=0.0125`$） |

**为什么支持 Theorem 5.10(c) 与 Proposition 5.5(a).**

- 在 warm Adam 状态下，平均梯度相同的两个分布在一阶上被 $`v`$ 通道分开，系数与预测完全一致。
- $`\delta_1=0`$，所以 momentum 块为零。
- 参数块也为零，原因有二：
  - 它的均值项正比于 $`\delta_1`$，而 $`\delta_1=0`$；
  - 它的 warmth 误差，即精确参数增量与冻结分母参数增量之差（Proposition 5.5(b)），在这里恰好为零，因为标签分布对称且 $`m=0`$。
- 实测差距中的其余部分是 [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1) 的余项。
- Proposition 5.6 的残差界覆盖了线性的 $`v`$ 项；[Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(c) 用显式的统计量 $`\mathbb E\,G^{\odot2}`$ 代替了这一残差。
- 证据的局限：这个见证只覆盖了冻结分母描述对参数块恰好精确的情形。一般情况下，精确的一阶对象是平均 Adam 增量；$`(\mathbb E\,G,\mathbb E\,G^{\odot2})`$ 只能把它确定到 Proposition 5.5(b) 的冻结分母误差为止。
- $`v`$ 通道在长尾段上如何起作用，见 [§7](../proofs/07_second_moment_channel_ZH.md)。

<a id="c5-17"></a>

## Check 5.17 · Lion：momentum 占主导时参数块对消息不敏感（Theorem 5.10(f)）

**结论.** momentum 占主导时，两个差别很大的梯度分布在参数块上的对比恰为 0，没有符号翻转；$`m`$ 块恰好承载 $`0.01\,\delta_1`$。

**动机.** [Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(f) 说，momentum 占主导的符号更新与消息无关，所以即使梯度分布差别很大，参数增量也相同；只有 $`m`$ 块记录了差别。

**设置**（`v015:E11`，Lion）.

- 网络配 Lion（$`\beta_1=0.9`$，$`\beta_2=0.99`$，$`\eta=0.03`$）。
- 标签分别为 0 和 1 的两条消息，梯度相差 $`\delta_1`$，$`\max\lvert\delta_1\rvert=1`$。
- momentum 占主导的状态：每个坐标 $`m=0.2`$，使每条消息都满足 $`0.1\lvert G_i\rvert<0.9\cdot0.2`$。
- 对照：$`m=0.005`$。

**做法.**

- 计算逐坐标的翻转指示量，以及每个块上的增量之差。
- 断言占主导的情形下参数对比恰为零，对照情形下不为零。
- 断言 $`m`$ 块等于 $`(1-\beta_2)\delta_1=0.01\,\delta_1`$。

**结果**（`v015:E11`，`data.Lion`）.

| 状态 | 参数块对比 $`\max_i\lvert\cdot\rvert`$ | 翻转比例 | $`m`$ 块对比 $`\max_i\lvert\cdot\rvert`$ |
|---|---:|---:|---:|
| momentum 占主导，$`m=0.2`$ | 0（精确） | 0 | 0.0100 $`=0.01\max\lvert\delta_1\rvert`$ |
| 对照，$`m=0.005`$ | 0.0600 $`=2\eta`$ | 0.5 | 0.0100 |

**为什么支持 Theorem 5.10(f).**

- momentum 占主导时，两条消息的参数增量相同；它们在 $`\theta`$ 上的一阶统计量是翻转概率，这里为 0。
- 一旦某个符号翻转，对比就是完整的 $`2\eta`$ 一步。
- $`m`$ 块总是承载 $`(1-\beta_2)\delta_1`$。如果之后没有符号改变，continuation value 在 $`m`$ 方向上局部为常数，这一通道同样没有局部的一阶效应。

<a id="c5-18"></a>

## Check 5.18 · LoRA：第一步只看到一个低秩 sketch（Theorem 5.10(g)）

**结论.**

- $`B=0`$ 时，在 SGD、heavy-ball 与 cold Adam 下，对每个分布，第一步之后 $`A`$ 都逐位不变。
- 只在 $`A_0^\top`$ 零空间中不同的两个分布，一阶差距小于 $`10^{-20}`$，差距是二阶的（斜率 2.000）。
- 在值域中不同的两个分布，一阶项与实测差距在每个 $`\eta`$ 上都相差不到 0.3%；$`\eta=10^{-3}`$ 时相对误差不到 $`3\times10^{-5}`$。

**动机.** [Theorem 5.10](../proofs/05_optimizer_state_ZH.md#s5-10)(g) 有两点断言：

- $`B=0`$ 时 adapter $`A`$ 在第一步不能移动，所以反馈只经由 $`\Gamma A_0^\top`$ 进入；
- 因此，平均全权重梯度的差别若落在 $`A_0^\top`$ 的零空间中，在一阶上看不见；落在值域中则看得见。

**设置**（`v015:E11`，LoRA）.

- 双线性 sigmoid 模型 $`W=W_0+\alpha BA`$，$`d=k=4`$，$`W_0=0.2`$，$`\alpha=1`$，$`r\in\lbrace1,2\rbrace`$，$`B_0=0`$，$`A_0`$ 随机。
- 消息是软标签的平均 cross-entropy 目标，精确实现指定的全权重梯度 $`\Gamma`$。
- 三组分布对：
  - *零空间*：平均梯度相差 $`\Delta\Gamma`$，且 $`\Delta\Gamma A_0^\top=0`$；
  - *值域*：差别形如 $`HA_0`$；
  - *符号*：sketch $`\Gamma A_0^\top`$ 的符号分布相同、幅度不同。
- optimizer：SGD、heavy-ball 与 cold Adam。逐位审计用 $`\eta=0.03`$，差距用完整的 $`\eta`$ 网格。

**做法.**

1. 断言对每个分布、每条消息、每个 optimizer，第一步之后 $`A`$ 都逐位不变。
2. 对零空间与值域两组分布对，计算第一步的精确差距与一阶项，并拟合零空间那组差距的斜率。
3. 对 cold Adam 下的符号那组，断言更新等于 $`-\eta h_\epsilon(\alpha\Gamma A_0^\top)`$，并记录有限 $`\epsilon`$ 带来的差别。

**结果**（`v015:E11`，`data.LoRA`）.

| 量 | rank 1 | rank 2 |
|---|---:|---:|
| $`A`$ 不变的逐位检验（SGD、heavy-ball、cold Adam × 所有消息） | 21 / 21 | 21 / 21 |
| 零空间残差 $`\max\lvert\Delta\Gamma A_0^\top\rvert`$ | $`2.6\times10^{-18}`$ | $`4.8\times10^{-19}`$ |
| 零空间对：$`\eta=0.1`$ 时的一阶项 | $`-1.1\times10^{-20}`$ | $`2.5\times10^{-21}`$ |
| 零空间对：$`\eta=0.1`$ 时的差距；对 $`\eta`$ 的斜率 | $`-3.66\times10^{-9}`$；1.99994 | $`-4.83\times10^{-8}`$；2.0000017 |
| 值域对，$`\eta=10^{-3}`$：一阶项与差距 | $`6.06362\times10^{-7}`$ 与 $`6.06362\times10^{-7}`$ | $`1.86514\times10^{-7}`$ 与 $`1.86509\times10^{-7}`$ |
| 符号对，cold Adam：$`\epsilon=10^{-8}`$ 时的更新之差 | $`7.1\times10^{-8}`$ | $`7.8\times10^{-8}`$ |

**为什么支持 Theorem 5.10(g).**

- 在 LoRA 初始化处，第一步不能利用 rank-$`r`$ sketch $`\Gamma A_0^\top`$ 之外的任何反馈。
- 零空间中的差别是二阶的；值域中的差别是一阶的，系数与预测一致。
- 在 cold Adam 下，$`\epsilon\to0`$ 时只有 sketch 的符号分布起作用。$`\epsilon=10^{-8}`$ 时两次更新最多相差 $`7.8\times10^{-8}`$，正是 $`h_\epsilon`$ 的有限 $`\epsilon`$ 修正。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

## 产物

- `theory/v015_checks/results_v015_independent.json`：
  - `v015:V8`，以及 `v015:E10`（逐单元格记录在 `data.cells`，汇总在 `data.optimizer_summary`，cold Adam 单元格在 `data.C23_prediction`，设计在 `data.design`）；
  - `v015:V2`、`v015:V3`、`v015:V4`、`v015:E0`、`v015:E1`、`v015:E2`、`v015:E3`；
  - `v015:E8`：冻结分母、warm 仿射、cold 分离与 warmth 轨迹四个部分（最后一部分含 `per_step` 与 `fits`），以及 210 个 `data.gaps` 单元格；
  - `v015:E11`（`data.clipped_mean`、`data.always_active`、`data.warm_Adam`、`data.Lion`、`data.LoRA`）。
- `theory/v015_checks/c23_bundle.json`：精确分布、网络、optimizer 代码，以及每条历史的原始更新。
- `theory/v015_checks/raw/E8_trajectories.npz`：逐副本的 warmth 轨迹。它被 gitignore，由上面第一条命令重新生成；SHA-256 记录在 `v015:E8` warmth 轨迹部分的 `raw_sha256` 中。
- `theory/v016_checks/results_v016.json`：claim 为 `v016:C1.exact`、`v016:C1.moments`、`v016:C1.accumulation`。完整数组在 `theory/v016_checks/results_v016_raw.json.gz` 中，该文件被 gitignore，SHA-256 记录在 `raw_artifact`。
- `theory/independent_checks/results_independent_v014.json`：`controlled_neural`。
- `docs/direction/v015_revision/results_v015.json`：package checks `pkg:T03`–`pkg:T08`。
- 代码：
  - `theory/v015_checks/otlr.py`（`check_V8`、`check_E10`、`check_E11`）；
  - `theory/v015_checks/warmth.py`（`check_E8`）；
  - `theory/v016_checks/check_v016.py`（`v016:C1`）；
  - 说明在 `theory/v015_checks/README.md` 与 `theory/v016_checks/SPEC.md`。
- 图：`../figures/fig5_1_linear_response`、`fig5_2_cold_adam`、`fig5_3_warmth`、`fig5_4_learner_statistics`，各有 `.pdf` 与 `.png`；说明在 `../figures/specs/fig5_1.md` 至 `fig5_4.md`。
