# N07 · Affine learner：矩阶数既必要又充分 — 实验

[← 框架总图](../README_ZH.md) · [证明](../proofs/N07_affine_moment_order_ZH.md) · [English](N07_affine_moment_order.md)

**结论。** N07 成立。所有陈述都用精确有理数算术检验，全部检验通过。

- **(a)** **（C6）：** 在 1,080 棵随机生成并经对抗变异的部署树上，"次数 $`\le N`$、系数范数 $`\le3^N`$"全部成立。找到的最大范数比值为 $`2/3`$。
- **(b)** **（C7）：** 在 24 个随机任务和仿射 learner（更新次数最多 $`R=5`$）上，矩匹配的分布给出*完全相同*的学习价值。对每个多重指标，只在该矩上不同的分布对都会被相应的单项式任务区分开。
- **(c)** **（C8 与包内检验）：**
  - $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$ 对 $`N=1,\dots,14`$ 精确成立，所有低阶矩相等。
  - 在 $`N=1,\dots,8`$ 上，parity 分布对所有次数 $`<N`$ 的联合多重指标都匹配，学习价值差距恒为 1。
- **(d)** **（C9 与包内检验）：**
  - 18 个配置的网格秩全部等于 $`\binom{k+N}N`$。
  - 237 个非常数特征中删去任意一个，都会留下可分离的分布对。
  - 32 个随机二元接口（特征数 $`<2^k-1`$）的最坏分布对差异全部恰为 1（偏差 $`\le4.4\times10^{-16}`$）。
- **(d)** **Span 维数（C19）：** 三个 3 次 response 族的 span 维数（含常数）分别为 4（标量幂）、8（parity）、4（结构化求和），与 [N07(d)](../proofs/N07_affine_moment_order_ZH.md) 和 [N06(a)](../proofs/N06_sufficiency_duality_ZH.md) 的预测一致。

![每一阶矩都必要；精确统计量个数是二项式系数](../figures/N07_moment_order.png)

[PDF](../figures/N07_moment_order.pdf)

### C6 — [N07(a)](../proofs/N07_affine_moment_order_ZH.md)：部署价值是有界多项式

**为什么做。** 整个仿射理论都建立在一件事上：$`J_E`$ 是 policy 坐标的多项式，次数 $`\le N`$，系数范数 $`\le3^N`$。只要有一棵树的精确展开次数更高或范数更大，引理就被证伪。

**设置。**

- 90 轮搜索，每轮 $`k\in\lbrace 1,2,3\rbrace`$ 个 context，每条路径 $`N\in\lbrace 1,\dots,5\rbrace`$ 个决策。
- context 可在路径上重复，机会节点使用有理概率。
- 每轮评估 12 棵候选树：每第四棵是全新的随机树，其余是对当前最坏比值树做叶子、context 或机会概率的随机变异，共 1,080 棵树。
- 种子：`SeedSequence([20261010, 6])`。

**做法。**

1. 用 SymPy 对每棵树做符号展开，得到精确的有理多项式。
2. 断言总次数 $`\le N`$ 且 $`\sum\lvert c_\alpha\rvert\le3^N`$。
3. 在 3 个随机有理点上断言 $`J\in[0,1]`$。
4. 对比值 $`\sum\lvert c_\alpha\rvert/3^N`$ 做爬山搜索，寻找最坏情形。

**结果。**

| 量 | 值 | 来源 |
|---|---:|---|
| 树数 / 断言数 | 1,080 / 5,400 | `claims[id=C6].n_instances`, `.n_assertions` |
| 最大 $`\sum\lvert c_\alpha\rvert/3^N`$ | $`2/3`$ | `claims[id=C6].details.exact_best_ratio` |
| 见证 | $`N=1`$，$`J=1-p_1`$ | `claims[id=C6].details.witness` |
| 状态 | PASS（最大残差 0） | `claims[id=C6].status` |

**说明了什么。** 次数界和系数界在每棵树上都成立，包括对抗变异的树。$`3^N`$ 界成立，但偏保守。

### C7 — [N07(b)](../proofs/N07_affine_moment_order_ZH.md)：矩匹配既充分又必要

**为什么做。** [N07(b)](../proofs/N07_affine_moment_order_ZH.md) 有两个方向，各有对应的证伪方式：

- 存在一对矩匹配的分布，在某个任务或仿射 learner 下学习价值*不同*（否定充分性）；
- 存在某个矩，它的失配没有任何任务能察觉（否定必要性）。

**设置与做法。**

- *充分性：* 对 24 个配置（$`k\in\lbrace 1,2\rbrace`$，$`N\in\lbrace 1,2,3\rbrace`$，$`R\in\lbrace 1,\dots,5\rbrace`$）：
  1. 在网格 $`\lbrace 0,\tfrac1{N+1},\dots,1\rbrace ^k`$ 上取随机的正分布 $`P`$；
  2. 构造 $`Q=P+av`$，其中 $`v`$ 取在所有次数 $`\le N`$ 的矩约束（含总质量）的精确零空间中；
  3. 随机抽取凸权重、$`p_0`$ 和任务树；
  4. 用精确的块矩卷积计算两个学习价值，断言所有被匹配的矩以及两个价值都相等。
- *必要性：* 对 $`(k,N)\in\lbrace (1,4),(2,3),(3,2)\rbrace`$ 以及每个满足 $`\lvert\alpha\rvert\le N`$ 的非零 $`\alpha`$：
  1. 符号验证"访问 $`j`$ 恰好 $`\alpha_j`$ 次、全为 1 才回报"的任务等于 $`p^\alpha`$；
  2. 构造*只*在第 $`\alpha`$ 个矩上不同的分布对；
  3. 断言一步 learner $`p=Z`$ 能把它们区分开。
- *两 context 例子：* 边缘相同，交叉矩分别为 $`\tfrac12`$ 和 $`0`$。
- 种子：`SeedSequence([20261010, 7])`。

**结果。**

| 量 | 值 | 来源 |
|---|---:|---|
| 实例数 / 断言数 | 46 / 385 | `claims[id=C7].n_instances`, `.n_assertions` |
| 仿射充分性配置 | 24 个，$`R`$ 最大为 5 | `claims[id=C7].details.affine_configurations` |
| 最大残差 | 0（精确） | `claims[id=C7].worst_residual` |
| 状态 | PASS | `claims[id=C7].status` |

**说明了什么。** 匹配全部 $`N`$ 阶以内的联合矩后，学习价值精确相等：差为零，而不只是很小。反过来，$`N`$ 阶以内的每个矩都能被某个任务察觉。只匹配边缘分布是不够的。

### C8 — [N07(c)](../proofs/N07_affine_moment_order_ZH.md)：每一阶都必要；常数差距见证

**为什么做。** [N07(c)](../proofs/N07_affine_moment_order_ZH.md) 说明层级是严格的：

- 对每个 $`N`$，都存在所有低阶矩一致、但 $`N`$ 阶学习价值不同的分布；
- 在 parity 构造中，差距保持为 1，不随 $`N`$ 缩小。

**设置与做法。**

- *标量见证：* 对 $`N=1,\dots,14`$，用精确有理数按 [N07(c)](../proofs/N07_affine_moment_order_ZH.md) 构造 $`P_N,Q_N`$。断言总质量为 1、所有 $`\ell<N`$ 的矩相等，且 $`m_{P_N}(N)-m_{Q_N}(N)=(-1)^NN!/(2^{N-1}N^N)`$。
- *Parity 见证：* 对 $`N=1,\dots,8`$ 枚举 $`\lbrace 0,1\rbrace ^N`$：
  1. 对总次数 $`<N`$ 的*所有*多重指标（不只是子集）断言联合矩相等；
  2. 断言 $`J_{\rm even}`$ 的值为 1 和 0；
  3. 在 4 个随机有理点 $`p`$ 上，断言 [N07(c)](../proofs/N07_affine_moment_order_ZH.md) 等于枚举得到的偶 parity 概率。
- 种子：`SeedSequence([20261010, 8])`。

**结果。**

| $`N`$ | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $`\Delta_N`$（精确） | $`-1`$ | $`1/4`$ | $`-1/18`$ | $`3/256`$ | $`-3/1250`$ | $`5/10368`$ | $`315/16777216`$ | $`567/800000000`$ | $`\approx9.58\times10^{-10}`$ |

$`\Delta_N`$ 来自 `claims[id=C8].details.scalar`；全部 14 个值都等于闭式表达式（已从 JSON 重新验证）。

| 量 | 值 | 来源 |
|---|---:|---|
| 所有低阶多重指标都匹配的 parity 阶数 | $`N=1,\dots,8`$ | `claims[id=C8].details.parity_all_multiindices`（true） |
| 断言数 | 8,983 | `claims[id=C8].n_assertions` |
| 状态 | PASS（最大残差 0） | `claims[id=C8].status` |

**说明了什么。** 对某个标量任务而言，每一阶 $`N`$ 都是必要的，差距公式精确成立。标量差距随 $`N`$ 指数缩小，parity 差距则在每一阶都保持为 1。因此"精确必要性"和"固定精度下的困难性"是两回事（在 N09 中展开）。

### 包内检验 — 常数差距 parity（独立实现）

**为什么做。** 这是理论作者对 parity 见证的独立重新实现。

**做法。** 对 $`N=1,\dots,8`$，检验全部 $`2^N-1`$ 个真坐标子集的子集矩是否相等。由于二元坐标是幂等的，这已覆盖所有次数 $`<N`$ 的原始矩。然后精确计算两个价值。

**结果。** 对每个 $`N=1,\dots,8`$，精确价值差距都是 `"1"`（参考值 1.0，备选值 0.0），匹配的真子集检验数依次为 $`1,3,7,15,31,63,127,255`$。来源：`theory/v014_repro/results_v014_rerun.json → constant_gap_parity`，与包内自带的 `docs/direction/v014_handoff/results_v014.json` 完全一致。

### C9 — [N07(d)](../proofs/N07_affine_moment_order_ZH.md)：精确统计量个数与二元壁垒

**为什么做。** [N07(d)](../proofs/N07_affine_moment_order_ZH.md) 把 [N06(a)](../proofs/N06_sufficiency_duality_ZH.md) 落成具体数字：

- 在网格上恰好需要、也只需要 $`\binom{k+N}N-1`$ 个统计量；
- 在二元消息、任意回报的设定下，少于 $`2^k-1`$ 个统计量就会留下最坏情形误差 1。

**设置与做法。**

1. *网格秩：* 对所有 $`k\le3,N\le5`$ 以及 $`k=4,N\le3`$（共 18 个配置），计算单项式求值矩阵在网格上的精确整数秩，断言其等于 $`\binom{k+N}N`$。
2. *删去特征：* 对每个配置中的每个非常数单项式（共 237 个），删去它后求解最坏分布对 LP，断言存在特征匹配的分布对能区分被删的单项式。
3. *二元壁垒：* 对 $`k=1,\dots,4`$，各抽 8 个特征数 $`d<2^k-1`$ 的随机有理特征表。取 $`[\mathbf 1;\phi]`$ 的一个零空间向量，以其正支撑为事件，断言 LP 差异等于 1。$`k\le3`$ 时还枚举全部事件，断言最优值为 1。
4. *完整接口：* 用 $`2^k-1`$ 个一般位置的特征，对每个示性 response 和一个随机 response 断言差异为 0。

种子：`SeedSequence([20261010, 9])`。

**结果。**

| 量 | 值 | 来源 |
|---|---:|---|
| 网格秩 $`=\binom{k+N}N`$ | 18/18，例如 $`(k,N)=(3,5)`$：56，$`(4,3)`$：35 | `claims[id=C9].details.grid_ranks` |
| 删去后均可分离的非常数特征 | 237 | `claims[id=C9].details.dropped_features` |
| $`d<2^k-1`$ 的二元接口；最坏分布对差异 | 32 个；全部 $`=1`$（最大偏差 $`4.4\times10^{-16}`$） | `claims[id=C9].details.binary_barriers` |
| 完整 $`2^k-1`$ 接口的差异 | 0（断言 `full generic interface discrepancy`） | `claims[id=C9].status` = PASS |
| 断言数 / 最大 LP 残差 | 903 / $`2.3\times10^{-12}`$ | `claims[id=C9].n_assertions`, `.worst_residual` |

**说明了什么。**

- 二项式计数是精确的：每个非常数矩单独都是必要的，合起来又是充分的。
- 二元壁垒被一个实际的分离事件取到，而不只是一个上界。

### 包内检验 — 网格 response 维数（独立实现）

**结果。** 检验了六个配置下 $`\lbrace 0,\tfrac1N,\dots,1\rbrace ^k`$ 上单项式矩阵的数值秩（容差 $`10^{-10}`$）：

| $`(k,N)`$ | $`(1,1)`$ | $`(1,4)`$ | $`(2,2)`$ | $`(2,3)`$ | $`(3,2)`$ | $`(3,3)`$ |
|---|---:|---:|---:|---:|---:|---:|
| 字母表大小 | 2 | 5 | 9 | 16 | 27 | 64 |
| 秩 $`=\binom{k+N}N`$ | 2 | 5 | 6 | 10 | 10 | 20 |
| 最少非常数统计量 | 1 | 4 | 5 | 9 | 9 | 19 |

来源：`theory/v014_repro/results_v014_rerun.json → grid_response_dimensions`。

### C19 — 三个同次数 response 族的 span 维数

**为什么做。** [N07(d)](../proofs/N07_affine_moment_order_ZH.md) 统计的是所有任务上的最坏情形。C19 用精确秩检验：*给定*的 response 族，其 span 维数是否就是 [N06(a)](../proofs/N06_sufficiency_duality_ZH.md) 和 [N07(d)](../proofs/N07_affine_moment_order_ZH.md) 预测的值。结构化的族维数远小于最坏情形。

**设置。**

- 网格 $`\lbrace 0,\tfrac14,\dots,1\rbrace ^3`$（125 条消息），每个族的最高次数都是 3。
- *标量仿射幂：* $`(a+bx_1)^3`$，5 个 response。
- *多元 parity 拉回：* $`J_{\rm even}`$ 复合 8 个仿射映射，再加上原始 parity，共 9 个 response。
- *结构化求和幂：* $`(a+b\bar x)^3`$，5 个 response。
- 种子：`SeedSequence([20261010, 19])`。

**结果（含常数的精确有理秩）。** 标量幂：4；parity：8；结构化求和：4。来源：`claims[id=C19].details.families[*].span_dimension_including_constant`。

**说明了什么。** 同次数的族在维数上可以相差很大：标量族和结构化族只需 $`1+3`$ 维，parity 族需要 $`2^3=8`$ 维。起作用的是精确维数（正如 [N06(a)](../proofs/N06_sufficiency_duality_ZH.md) 所说），不是次数。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C6,C7,C8,C9,C19 \
  --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py \
  --output theory/v014_repro/results_v014_rerun.json
```

## 产物

- 结果：
  - `theory/independent_checks/results_independent_v014.json`（C6–C9、C19）
  - `theory/v014_repro/results_v014_rerun.json`（`constant_gap_parity`、`grid_response_dimensions`）
- 代码：`theory/independent_checks/check_v014_independent.py`、`docs/direction/v014_handoff/verify_environment_information_v014.py`
- 图规格：`theoretical_framework/figures/specs/N07.md`
