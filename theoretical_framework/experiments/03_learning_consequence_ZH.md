# §3 学习后果 — 实验

[← 框架总览](../README_ZH.md) · [证明](../proofs/03_learning_consequence_ZH.md) · [English](03_learning_consequence.md)

**结论.**

- [Proposition 3.1](../proofs/03_learning_consequence_ZH.md#s3-1)：通过。两个分离都是精确恒等式（75 条断言，残差 0）：静态值同为 $`\tfrac12+p/20`$，而对每个 $`m=1,\dots,30`$，频率学习器的回报恰为 $`\tfrac{43}{80}`$ 与 $`\tfrac{41}{80}`$；包的均值相同，差距恰为 $`\eta^2/8`$，在 $`1-p^2`$ 下反号为 $`-\eta^2/8`$。
- 同样的分离在训练过的受控网络中也出现：在全部 24 个精确枚举的设定（$`R\le4`$，SGD 与 Adam）中，两个静态等价的来源都训练出不同的学生，符号与命题预测一致。
- [Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)：通过。在 45 个带实际更新映射与更新随机性的随机有限学习过程上（1,867 条断言，残差 0），telescoping 恒等式 (3.2) 精确成立；对认证集合的对抗选择从未打破带停止的界 (3.5)；单边界值之差恰为 $`(P-Q)g`$。
- 局部差距在一个被审计的格子上认证、再 transport 到访问到的状态时，带停止的复合依然有效且不空：包检验中 400/400 个实例，最紧的达到界的 47%。

| 检验 | 检验对象 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 3.1](#c3-1) | [Proposition 3.1](../proofs/03_learning_consequence_ZH.md#s3-1) | `v014:C3` | 通过 |
| [Check 3.2](#c3-2) | 训练网络中的 [Proposition 3.1](../proofs/03_learning_consequence_ZH.md#s3-1) | `v014:C23`（构造 (a)–(b)） | 受控发现 |
| [Check 3.3](#c3-3) | [Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2) | `v014:C4` | 通过 |
| [Check 3.4](#c3-4) | 带 audit transport 的 [Theorem 3.2(c)](../proofs/03_learning_consequence_ZH.md#s3-2) | `pkg:T12` | 通过 |

<a id="c3-1"></a>

## Check 3.1 · 静态值相同、平均更新相同，学习价值不同（Proposition 3.1）

**结论.** 通过：60 个实例，75 条断言，最大残差 0。Proposition 3.1(a) 的差距对每个 $`m=1,\dots,30`$ 都恰为 $`1/40`$；Proposition 3.1(b) 的差距恰为 $`\eta^2/8`$（在 $`1-p^2`$ 下为 $`-\eta^2/8`$）。

**为什么做.**

- Proposition 3.1(a) 断言对所有 $`p`$ 静态值*精确*相等，同时对所有 $`m`$ 学习价值的差距精确成立。表中的算术错误、非最优的 teacher 或依赖 $`m`$ 的差距都能否定它。
- Proposition 3.1(b) 断言均值相同，但存在依赖曲率、符号随任务而变的差距。

**设置.**

- 关于 $`p`$ 与 $`\eta`$ 的 SymPy 符号计算。
- 频率学习器的精确二项期望，用 `fractions.Fraction`，$`m=1,\dots,30`$，两个来源都算。
- 在全部四个状态上逐个核对 teacher 的最优性。
- 种子 `SeedSequence([20261010, 3])`；检验是确定性的。

**做法.**

1. 对两个状态分布展开 $`J_\rho(\pi_p)`$，断言都等于 $`\tfrac12+p/20`$。
2. 断言 teacher 在两个分布下的值都是 $`\tfrac35`$，且它在每个状态上的动作都最优。
3. 对每个 $`m`$，把 $`\mathbb E\,J(\pi_{\widehat p})`$ 精确算成二项和，断言为 $`\tfrac{43}{80}`$ 与 $`\tfrac{41}{80}`$。
4. 枚举四个等概率的标签对；断言 $`\mathbb Ep^+=\tfrac12`$，$`p^2`$ 下差距为 $`\eta^2/8`$，$`1-p^2`$ 下为 $`-\eta^2/8`$。

**结果.**

| 量 | 值 | JSON 键（条目 `v014:C3`） |
|---|---|---|
| 实例数 / 断言数 / 最大残差 | 60 / 75 / 0 | `n_instances`、`n_assertions`、`worst_residual` |
| Proposition 3.1(a) 的差距（每个 $`m=1,\dots,30`$） | $`1/40`$ | `details.prop5_gap` |
| Proposition 3.1(b) 的差距 | $`\eta^2/8`$（$`1-p^2`$ 下为 $`-\eta^2/8`$） | `details.prop6_gap` |
| 状态 | 通过 | `status` |

**为什么支持这个陈述.** 两个分离都是精确恒等式，不是近似。两个来源可以在选择器可能检查的每个静态量上都一致（每个混叠学生的值、teacher 的值、标签边缘分布、平均梯度），却仍然训练出不同的学生。

<a id="c3-2"></a>

## Check 3.2 · 训练网络中的同一分离（Proposition 3.1，受控网络）

**结论.** 受控发现：在全部 24 个精确枚举的设定（$`R\le4`$）中，每个构造的两个静态等价来源都训练出不同的学生，符号与 Proposition 3.1 的预测一致。范围：受控小网络，不是真实模型，也不是 benchmark。

**为什么做.** Proposition 3.1 用的是理想化的学习器（频率蒸馏、一步 SGD）。问题是：用梯度方法训练的非线性网络是否也能区分这两个静态等价的来源。若差距为零，说明这个分离只是理想化学习器的产物。

**设置.**

- 网络：one-hot 上下文 → 4 个 tanh 单元 → sigmoid 动作概率，固定初始化，在标签块上用交叉熵。
- 优化器：SGD，$`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$；Adam（$`\beta=(0.9,0.999)`$，$`\epsilon=10^{-8}`$，bias correction），$`\eta\in\lbrace 0.01,0.1\rbrace`$。
- $`R\in\lbrace 1,4,40\rbrace`$ 次更新，来源分布取自构造 (a) 与 (b)。
- $`R\le4`$ 时精确枚举所有训练历史。$`R=40`$ 时每个来源跑 2,400 次独立训练；SE 按训练次数计。
- 种子 `SeedSequence([20261010, 23])`。

**结果（差距 = (a) 中参照 − 备选，(b) 中独立 − 配对；未给 SE 的为精确枚举）.**

| 构造 | 优化器，$`\eta`$ | $`R=1`$ | $`R=4`$ | $`R=40`$（SE） |
|---|---|---:|---:|---:|
| (a) | SGD 0.01 | 1.21e-4 | 4.81e-4 | 4.38e-3（1.7e-5） |
| (a) | SGD 0.1 | 1.21e-3 | 4.48e-3 | 2.10e-2（9.2e-5） |
| (a) | SGD 1.0 | 1.10e-2 | 1.88e-2 | 2.44e-2（2.0e-4） |
| (a) | Adam 0.01 | 5.04e-4 | 2.00e-3 | 1.54e-2（6.9e-5） |
| (a) | Adam 0.1 | 4.91e-3 | 1.54e-2 | 2.40e-2（1.9e-4） |
| (b) | SGD 0.01 | 3.47e-6 | 1.36e-5 | 3.7e-5（2.0e-4） |
| (b) | SGD 0.1 | 3.46e-4 | 1.15e-3 | 2.0e-3（1.1e-3） |
| (b) | SGD 1.0 | 2.86e-2 | 1.86e-2 | 1.61e-2（2.7e-3） |
| (b) | Adam 0.01 | 9.19e-5 | 6.33e-4 | 1.23e-3（9.1e-4） |
| (b) | Adam 0.1 | 8.75e-3 | 2.76e-2 | 1.23e-2（2.4e-3） |

数值取自 `controlled_neural.details.configurations[task∈{Prop5,Prop6}].neural_gap` 与 `.SE`；$`\eta=3`$ 的行在 JSON 中。SGD 步长较小时，实测差距与网络的局部切线预测一致：(a) 中 SGD $`\eta=0.01`$、$`R=1`$ 时，实测 1.21170e-4，预测 1.21171e-4（`.local_affine_SGD_prediction`）。

![Figure 3.1：评估相同，学到的学生不同](../figures/fig3_1_static_vs_learning.png)

*Figure 3.1.* (a) 构造 (a)：两个来源下每个混叠学生的值都是 $`\tfrac12+p/20`$，teacher 的值都是 $`\tfrac35`$，但频率蒸馏分别得到 $`\tfrac{43}{80}`$ 与 $`\tfrac{41}{80}`$。(b) 构造 (b)：标签边缘分布、平均梯度与平均更新都相同，但学习价值差在 $`J=p^2`$ 下为 $`+\eta^2/8`$，在 $`J=1-p^2`$ 下为 $`-\eta^2/8`$。(c) 在受控 tanh 网络中，静态等价的来源仍训练出不同的学生（精确枚举，$`R\le4`$；对数坐标）。[PDF](../figures/fig3_1_static_vs_learning.pdf)

**为什么支持这个陈述.**

- 与命题一样，在训练网络中，静态等价的反馈来源也得到不同的学生。每个精确枚举的差距都为正，方向与预测一致：(a) 中参照来源更好，(b) 中在 $`p^2`$ 下独立包更好。
- 差距随步长增大，这符合"学习后果"效应的预期；(a) 中当 $`\eta\le1`$ 时差距还随更新次数增大。
- $`R=40`$（Monte Carlo）时，构造 (b) 有三个差距与零的距离在两个标准误以内（SGD $`\eta=0.01`$ 与 $`0.1`$，Adam $`\eta=0.01`$）；证据主要来自精确枚举的行。
- 差距的大小依赖于 optimizer：$`\eta`$ 相同时，Adam 的差距大于 SGD。§5 解释这种依赖（[Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)，cold-start Adam）。

<a id="c3-3"></a>

## Check 3.3 · 精确复合、带停止的复合与单边界刻画（Theorem 3.2）

**结论.** 通过：45 个随机有限学习过程，1,867 条断言，最大残差 0。对认证集合的对抗搜索从未使 $`\lvert\text{差距}\rvert-(\text{离开前的漂移}+\rho)`$ 为正。

**为什么做.**

- (3.2) 是恒等式。方向上的错误（例如用参照过程的占据分布代替 $`d_t^S`$）或 continuation 递推的错误，都会在随机过程上产生非零残差。
- (3.5) 断言离开的代价是 $`\rho`$ 而不是 $`R\rho`$。若某种认证集合的选法使 $`\lvert\text{差距}\rvert`$ 超过界，就否定了它。
- (3.4) 依赖于单边界值之差恰为 $`(P-Q)g`$。

**设置.**

- 45 个随机过程，状态数 $`\lvert\mathcal Z\rvert\in[4,20]`$，轮数 $`R\in[1,8]`$。
- 消息字母表大小 2–4；更新随机性 $`\xi`$ 取两个值，概率为 $`\tfrac13,\tfrac23`$。
- 随机的确定性更新表 $`U_t(z,y,\xi)\in\mathcal Z`$，每个 $`(t,z)`$ 处随机的参照与备选消息分布；kernel $`K_t^0,K_t^S`$ 按定理中的方式由它们构造。
- 有理的终端得分 $`F\in[0,1]`$，有理的初始分布，精确的 `fractions.Fraction` 运算。种子 `SeedSequence([20261010, 4])`。

**做法.**

1. 反向计算 $`V_t^0`$，正向计算 $`d_t^S`$。断言 $`L(0)-L(S)=\sum_t d_t^S(K_t^0-K_t^S)V_{t+1}^0`$ 精确成立，以及 $`\lvert L(0)-L(S)\rvert\le\sum_td_t^S\lvert(K_t^0-K_t^S)V_{t+1}^0\rvert`$。
2. 带停止的界。认证掩码 $`C_t`$ 在偶数号实例上全部认证，在奇数号实例上随机。35 步爬山每次翻转一个 $`(t,z)`$，以*最大化* $`\lvert\text{差距}\rvert-(\text{离开前的漂移}+\rho)`$，其中 $`\rho`$ 是精确的首次离开概率；每一步都断言 $`\lvert\text{差距}\rvert\le\text{离开前的漂移}+\rho`$。
3. 每轮取一个随机快照 $`(t,z)`$，构造实际的 response 向量 $`g(y)=\sum_\xi\Pr(\xi)V_{t+1}^0(U_t(z,y,\xi))`$，断言单边界值之差精确等于 $`(P-Q)g`$。
4. 见证：
   - *可分离的 response.* $`P=(\tfrac12,0,\tfrac12)`$ 与 $`Q=(0,1,0)`$ 在 $`g_1\equiv1`$ 与 $`g_2=(0,\tfrac12,1)`$ 上一致，在 $`g_3=(0,0,1)`$ 上不同。
   - *抵消.* 两个状态，$`K^0`$ 为恒等，$`K^S`$ 为交换，两轮，$`F=(0,1)`$，从状态 0 出发。

**结果.**

| 量 | 值 | JSON 键（条目 `v014:C4`） |
|---|---|---|
| 过程数 / 断言数 / 最大残差 | 45 / 1,867 / 0 | `n_instances`、`n_assertions`、`worst_residual` |
| 对抗搜索找到的最大 $`\lvert\text{差距}\rvert-(\text{离开前的漂移}+\rho)`$ | $`0.0`$（从未为正） | `details.adversarial_exit_best_lhs_minus_rhs` |
| 抵消见证：最终差距 / $`\sum_t\mathbb E\lvert\text{漂移}\rvert`$ | $`0`$ / $`2`$ | `details.cancellation_absolute_drift` |
| 可分离 response 的见证 | $`P,Q`$ 在 $`g_1,g_2`$ 上相等，在 $`g_3`$ 上不等 | 在 `check_C4` 中断言 |
| 状态 | 通过 | `status` |

![Figure 3.2：telescoping、单边界等价与抵消](../figures/fig3_2_response.png)

*Figure 3.2.* (a) 在一个显式的三状态过程上，学习价值差等于各轮按占据分布加权的 response 差距之和 (3.2)，并被它们的绝对值之和界住 (3.3)。(b) 分布 $`P=(\tfrac12,0,\tfrac12)`$ 与 $`Q=(0,1,0)`$ 在两个 response 上一致、在第三个上不同：两个分布对单边界替换可互换，当且仅当它们在每个 response 上一致 (3.4)。(c) 各轮差距可以抵消（最终差距 0，绝对漂移之和 2），所以匹配单个最终均值不要求逐点匹配。[PDF](../figures/fig3_2_response.pdf)

**为什么支持这个陈述.**

- 学习价值差*精确*等于按占据分布加权的单边界 response 差距之和，用的是备选过程的占据分布，与 (3.2) 一致。
- 对手选择认证哪些状态，也打破不了带停止的界：一次离开的代价至多为 $`\rho`$（(3.5)）。
- 单边界值之差精确为 $`(P-Q)g`$，所以 response 类是正确的不变量（(3.4)）。
- 抵消见证确认：对单个最终均值，逐点匹配不是必要的；这正是 Theorem 3.2(b) 对所有单边界替换取全称量词的原因。

<a id="c3-4"></a>

## Check 3.4 · 带 audit transport 的停止复合（Theorem 3.2(c)）

**结论.** 通过：400 个随机 Markov 实例；每个界都有效，每个界都小于 1，差距与界之比最大为 0.470。

**为什么做.** 这是带停止的界的独立实现，形式与 [Theorem 8.2](../proofs/08_verification_ZH.md#s8-2) 中的用法一致：response 写成带常数基线的指示特征，差距在一个被审计的锚状态上测量，再 transport 到访问到的状态。

**设置.**

- 400 个随机的 4 状态、3 轮 Markov 过程。参照行 $`P_t(z,\cdot)`$ 从 $`\mathrm{Dirichlet}(1)`$ 抽取；备选为 $`Q=0.97P+0.03\,\mathrm{Dirichlet}(1)`$；初始状态固定。
- 一半的试验中状态 3 不被认证，到达它的概率质量计入 $`\rho`$。
- 每轮的界为 $`\sum_z\Pr(Z_t=z,\text{仍存活})\,B_t\,(D_t+a_t(z))`$，其中
  - $`b=V_{t+1}(0{:}3)-V_{t+1}(3)`$，$`B_t=\lVert b\rVert_1`$；
  - $`D_t`$ 是锚状态 0 处的差距；
  - $`a_t(z)`$ 是从状态 $`z`$ 到锚状态的 sup 范数 transport 误差。

  总界再加上 $`\rho`$。
- 包种子 15019。

**结果.**

| 量 | 值 | JSON 键（`checks.T12_stopped_occupancy_composition`） |
|---|---|---|
| 实例数 | 400 | `instances` |
| 差距 ≤ 界 | 400 / 400（已断言） | — |
| 界严格小于 1（不空） | 400 / 400 | `bounds_below_one` |
| 差距 / 界 的最大值 | 0.470 | `maximum_gap_over_bound` |

**为什么支持这个陈述.** 当局部差距 $`\varepsilon_t`$ 在被审计的格子上认证、再 transport 到访问到的状态时，带停止的复合 (3.5) 依然有效且不空。冻结 audit 定理（[Theorem 8.2](../proofs/08_verification_ZH.md#s8-2)）正是建立在这个形式之上。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C3,C4,C23 --output theory/independent_checks/subset.json
theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py --out theory/v015_checks/quick/results_v015_package_rerun.json
```

## 产物

- `theory/independent_checks/results_independent_v014.json`：`v014:C3` 与 `v014:C4` 的条目；`controlled_neural.details.configurations`（按 `task` = `Prop5` 或 `Prop6` 筛选）。
- `theory/independent_checks/check_v014_independent.py`：`check_C3`、`check_C4`（含 `process`、`continuations`、`process_stats`）、`check_C23`。
- `docs/direction/v015_revision/results_v015.json`：`checks.T12_stopped_occupancy_composition`；代码 `docs/direction/v015_revision/verify_v015_extensions.py`（`stopped_learning`）。
- 图：`figures/fig3_1_static_vs_learning.{png,pdf}`、`figures/fig3_2_response.{png,pdf}`；规格见 `figures/specs/fig3_1.md`、`fig3_2.md`。
