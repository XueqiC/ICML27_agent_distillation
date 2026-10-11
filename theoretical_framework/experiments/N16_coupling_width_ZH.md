# N16 · coupling width — 实验

[← 框架总图](../README_ZH.md) · [证明](../proofs/N16_coupling_width_ZH.md)

**结论.** N16 成立。在 320 个随机 packet-response 表上（C17；2,567 条断言；最坏残差 $`9.9\times10^{-16}`$），以下三点得到确认：

- **(a)** 可加表的 coupling width 恰为零，不可加表的 width 为正；
- **(b)** 成立，余量中位数为 0.231；
- **(c)** 受限配对和严格零效应两条陈述都精确成立。

包内检验给出：可加表 width 为 0，二次配对表 width 为 0.25，严格零效应残差为 $`2.8\times10^{-17}`$。

在受控网络中（[C23(e)](N16_coupling_width_ZH.md#c23e--受控网络中的重新分组)（v0.14 受控网络实验，其中 Adam 偏离仿射对照）），公共状态上的一步 SGD 对重新分组精确不变（残差 0）。同样的标签换一种分组、做顺序更新后，差距随步长和时域增大，从 $`10^{-5}`$ 到 $`0.75`$。这正是命题所说的、只可能出现在单步零效应之外的不可加学习后果效应。
![coupling width：精确零效应与顺序重组效应](../figures/N16_coupling.png)

[PDF](../figures/N16_coupling.pdf)

## C17 · coupling 多面体、可加性刻画与严格零效应

**动机.** [N16(a)–(b)](../proofs/N16_coupling_width_ZH.md) 有四条结论：width 为零的"当且仅当"刻画、不等式 [N16(b)](../proofs/N16_coupling_width_ZH.md)、禁止配对时"当且仅当"的失效，以及单步零效应。任何一条错了，都会在随机实例上明显暴露出来。

**设置.**

- $`\lvert\mathcal Y_1\rvert,\lvert\mathcal Y_2\rvert\in[2,5]`$，边际严格为正。
- 随机可加表 $`H=a+b`$，以及随机不可加表。
- 种子 `SeedSequence([20261010, 17])`。

**做法.**

1. 用两个 LP 计算 width（在 coupling 多面体上分别求 $`\langle\Gamma,H\rangle`$ 的最大值和最小值）。
2. 用第三个 LP 计算 $`\inf_{a,b}\lVert H-a-b\rVert_\infty`$，并检验 [N16(b)](../proofs/N16_coupling_width_ZH.md)。
3. 构造禁止配对的例子。
4. 用有理数精确检验零效应：在公共参数上对线性模型做一步 SGD，把同样的样本重新分组。然后做两步带重新分组的顺序更新。

**结果**（`theory/independent_checks/results_independent_v014.json`，`claims[id=C17]`）：

| 量 | 值 |
|---|---:|
| status | PASS |
| 实例 / 断言 | 320 / 2,567 |
| 最坏残差（LP 可行性） | $`9.85\times10^{-16}`$ |
| [N16(b)](../proofs/N16_coupling_width_ZH.md) 的余量：中位数 / 最大值 | 0.231 / 0.712（`details.median_inequality_slack`、`maximum_slack`） |
| 可行 width 为零的禁止配对表 | $`\begin{pmatrix}0&0\cr 0&1\end{pmatrix}`$（`details.forbidden_pairing_response`） |
| 一步更新下的重组差异（线性模型，精确） | 0 |
| 两步顺序更新下的重组差异（线性模型，精确） | $`1/32`$（`details.sequential_linear_parameter_gap`） |

**说明了什么.**

- 对可加表，width 精确为零。
- [N16(b)](../proofs/N16_coupling_width_ZH.md) 是一个有效但一般不紧的界，典型余量约为回报范围的四分之一。
- 单步零效应在有理数算术下精确成立，所以任何测得的单步效应都是实现上的伪影。
- 顺序更新会产生真实的重组效应，即证明第 4 部分中的 $`\eta^2H_{P_2}g_{P_1}`$ 项。

## 包内检验 · coupling 与可加零效应

**动机.** 对同样的有限陈述做一次独立实现。

**结果**（`theory/v014_repro/results_v014_rerun.json`，`coupling_and_additive_null`）：

| 量 | 值 |
|---|---:|
| 可加表的 width | 0.0 |
| 二次配对表：最小 / 最大 / width | 0.25 / 0.50 / 0.25 |
| 严格的冻结梯度可加零效应，最大残差 | $`2.8\times10^{-17}`$ |

**说明了什么.** 可加表的 width 为零，不可加的二次配对表 width 为正，冻结梯度的零效应在机器精度内精确成立。

## C23(e) · 受控网络中的重新分组

**动机.** width 是通过实际的学习后果来定义的。一个实际的网络在同样的标签上训练、只是分组不同，在公共状态上的一步更新时差异应当恰好为零，顺序更新之后则可能不为零。

**设置.**

- 网络：one-hot 上下文、4 个 tanh 单元、sigmoid 输出、float64；标注为 "controlled neural, not Gemma, not an external benchmark"。
- 来源 A 的 packet 为 $`\lbrace (0,0),(1,1)\rbrace`$，来源 B 的 packet 为 $`\lbrace (0,1),(0,1)\rbrace`$。两者包含同样的四个标签，所以边际相同。
- 优化器：SGD，$`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$；Adam（$`\beta=(0.9,0.999)`$，$`\epsilon=10^{-8}`$），$`\eta\in\lbrace 0.01,0.1\rbrace`$。
- 1、4 或 40 个循环，对应 2、8 或 80 次更新。
- 精确枚举所有确定性历史。

**做法.** 计算两种分组之间最终期望回报的差距。另外单独计算在相同权重下、公共状态上一步更新后的参数差异。

**结果**（`theory/independent_checks/results_independent_v014.json`，`controlled_neural.details`）：

- 对公共状态上的单步更新，`additive_null_parameter_residual` = **0.0**；
- 手写梯度与中心差分相比，`gradient_max_residual` = $`8.7\times10^{-11}`$。

回报差距 A − B（`regrouping`）：

| 优化器 | $`\eta`$ | 2 次更新 | 8 次更新 | 80 次更新 |
|---|---:|---:|---:|---:|
| SGD | 0.01 | $`1.0\times10^{-5}`$ | $`3.8\times10^{-5}`$ | $`3.2\times10^{-4}`$ |
| SGD | 0.1 | $`9.2\times10^{-4}`$ | $`3.2\times10^{-3}`$ | $`7.9\times10^{-3}`$ |
| SGD | 1.0 | 0.040 | 0.062 | 0.076 |
| SGD | 3.0 | 0.658 | 0.747 | 0.749 |
| Adam | 0.01 | −0.0094 | −0.0136 | −0.0032 |
| Adam | 0.1 | −0.0729 | −0.0503 | −0.0029 |

**说明了什么.** 在实际的非线性学习者中，边际相同的反馈如何组合，会改变学到的学生。这个效应恰好在命题要求的地方为零（公共状态上的一步），在 SGD 下随步长和时域增大，在 Adam 下符号和动态都不同。所以 coupling width 是一个真实存在、依赖于优化器的学习后果量。packet 的组织方式属于方法必须测量的 response（N17），而不是数据本身的固定属性。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C17,C23 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py   # 包内检验
```

## 产物

- `theory/independent_checks/results_independent_v014.json`：`claims[id=C17]` 与 `controlled_neural.details.regrouping`。
- `theory/v014_repro/results_v014_rerun.json`：`coupling_and_additive_null`。
- 图：`../figures/N16_coupling.pdf` 与 `.png`（规格见 `../figures/specs/N16.md`）。
