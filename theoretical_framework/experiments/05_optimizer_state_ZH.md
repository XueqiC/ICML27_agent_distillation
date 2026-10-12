# §5 Optimizer state 与一阶表示 — 实验

[← 框架总图](../README_ZH.md) · [证明](../proofs/05_optimizer_state_ZH.md) · [English](05_optimizer_state.md)

本页保留 Check 5.1 至 Check 5.7，以及全部 §5 检验的 verdict 表；Check 5.8 至 Check 5.18 见 [Warm Adam 与学习器统计量](05_warm_adam_and_learners_ZH.md)。

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
| [Check 5.2](#c5-2) | [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)(b)、(d)；[Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)(d) | `v015:E10` | INFO（诊断）；所有光滑的精确单元格都在界内 |
| [Check 5.3](#c5-3) | [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(a)–(c) | `v015:V2`、`pkg:T03`–`pkg:T06` | PASS |
| [Check 5.4](#c5-4) | [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(d) | `v015:V3`、`pkg:T07` | PASS |
| [Check 5.5](#c5-5) | [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(e) | `v015:V4`、`v015:E3`、`pkg:T08` | PASS |
| [Check 5.6](#c5-6) | 网络中的 [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(a)–(c) | `v015:E0`、`v015:E1` | PASS |
| [Check 5.7](#c5-7) | [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)：optimizer 效应还是特征学习 | `v015:E2`、`v014:C23`（Adam 设定） | INFO；冻结 Jacobian 后，45 个单元格中 39 个差距仍在 |
| [Check 5.8](05_warm_adam_and_learners_ZH.md#c5-8) | [Proposition 5.5](../proofs/05_warm_adam_ZH.md#s5-5)(b) | `v015:E8`（冻结分母部分） | PASS |
| [Check 5.9](05_warm_adam_and_learners_ZH.md#c5-9) | [Proposition 5.6](../proofs/05_warm_adam_ZH.md#s5-6)、[Remark 5.11](../proofs/05_learner_statistics_ZH.md#s5-11) | `v015:E8`（warm 仿射部分） | PASS |
| [Check 5.10](05_warm_adam_and_learners_ZH.md#c5-10) | [Proposition 5.7](../proofs/05_warm_adam_ZH.md#s5-7) | `v015:E8`（cold 分离部分） | PASS |
| [Check 5.11](05_warm_adam_and_learners_ZH.md#c5-11) | [Proposition 5.8](../proofs/05_warm_adam_ZH.md#s5-8)(a)–(c) | `v016:C1.exact`、`v016:C1.moments`、`v016:C1.accumulation` | PASS |
| [Check 5.12](05_warm_adam_and_learners_ZH.md#c5-12) | 真实轨迹上的 [Proposition 5.8](../proofs/05_warm_adam_ZH.md#s5-8)(a)（早期阶段）与 (d)（warm-up） | `v015:E8`（warmth 轨迹部分） | PASS |
| [Check 5.13](05_warm_adam_and_learners_ZH.md#c5-13) | 整段训练中的 [Propositions 5.6](../proofs/05_warm_adam_ZH.md#s5-6)–[5.8](../proofs/05_warm_adam_ZH.md#s5-8) | `v015:E8`（`data.gaps`） | PASS |
| [Check 5.14](05_warm_adam_and_learners_ZH.md#c5-14) | [Proposition 5.9](../proofs/05_learner_statistics_ZH.md#s5-9)、[Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)(e) | `v015:E11`（clipped mean） | PASS |
| [Check 5.15](05_warm_adam_and_learners_ZH.md#c5-15) | $`p_{\rm act}=1`$ 时的 [Proposition 5.9](../proofs/05_learner_statistics_ZH.md#s5-9)、[Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)(e) | `v015:E11`（始终裁剪） | PASS |
| [Check 5.16](05_warm_adam_and_learners_ZH.md#c5-16) | [Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)(c)、[Proposition 5.5](../proofs/05_warm_adam_ZH.md#s5-5)(a) | `v015:E11`（warm Adam） | PASS |
| [Check 5.17](05_warm_adam_and_learners_ZH.md#c5-17) | [Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)(f) | `v015:E11`（Lion） | PASS |
| [Check 5.18](05_warm_adam_and_learners_ZH.md#c5-18) | [Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)(g) | `v015:E11`（LoRA） | PASS |

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

- 两个标签分布均值相同，所以任何基于平均梯度的统计量都看不出它们的区别。cold Adam 的增量 $`-\eta h_\epsilon(G)`$ 却能区分它们（[Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)(a)）。这正是 [Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10) 中 cold 步那一行 (d)。
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
  - 所以一阶排序必须满足 [Corollary 8.9](../proofs/08_certificate_ZH.md#s8-9) 的 margin 条件；二阶矩通道占主导时，要改用有限替换测量（[Theorem 8.8](../proofs/08_certificate_ZH.md#s8-8)）。
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
- 去掉符号分歧，差距就消失；把矩预热，差距就大部分消失，与 [Proposition 5.7](../proofs/05_warm_adam_ZH.md#s5-7) 和 [Proposition 5.6](../proofs/05_warm_adam_ZH.md#s5-6) 的预言一致。
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
