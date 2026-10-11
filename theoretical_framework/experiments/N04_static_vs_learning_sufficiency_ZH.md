# N04 · 静态价值不等于学习价值 — 实验

[← 框架总览](../README_ZH.md) · [证明](../proofs/N04_static_vs_learning_sufficiency_ZH.md)

**结论。** N04 成立。检验 C3 精确验证了两个命题，共 75 条断言，残差为零：

- **(a)**：每个 aliased 学生的静态价值都是 $`\tfrac12+p/20`$，两个数据源下的 teacher 价值都是 $`\tfrac35`$，而对每个 $`m=1,\dots,30`$，频率学习器的回报都恰为 $`\tfrac{43}{80}`$ 和 $`\tfrac{41}{80}`$；
- **(b)**：两种 packet 的均值相同，符号计算得到的差距恰为 $`\eta^2/8`$，换成 $`1-p^2`$ 后反号为 $`-\eta^2/8`$。

受控神经网络实验 [C23](N11_adam_cold_start_ZH.md)（v0.14 受控网络实验，其中 Adam 偏离仿射对照） 的 (a)–(b) 部分在真实训练的网络中观察到了同样的分离，标注为 *controlled neural, not Gemma, not an external benchmark*。在所有精确枚举的配置中，无论用 SGD 还是 Adam，两个静态等价的数据源都训练出了不同的学生。方向与理论一致：[N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) 中参照数据源更好，[N04(b)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) 中在 $`p^2`$ 下独立 packet 更好。

![评估相同，学到的学生不同](../figures/N04_static_vs_learning.png)

[PDF](../figures/N04_static_vs_learning.pdf)

### C3 — [N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md)、[N04(b)](../proofs/N04_static_vs_learning_sufficiency_ZH.md)

**动机。**

- [N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) 断言：对所有 $`p`$，静态价值*精确*相等；同时对所有 $`m`$，学习价值存在精确的差距。表中任何算术错误、teacher 不是最优，或者差距随 $`m`$ 变化，都会推翻它。
- [N04(b)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) 断言：均值相同，但差距依赖于曲率，其符号取决于任务。

**设置。**

- 关于 $`p`$ 与 $`\eta`$ 的 SymPy 符号计算。
- 用 `fractions.Fraction` 精确计算频率学习器的二项期望，$`m=1,\dots,30`$，两个数据源都算。
- 在全部四个状态上逐一检查 teacher 的最优性。
- 种子 `SeedSequence([20261010, 3])`；检验完全确定。

**做法。**

1. 对两个状态分布展开 $`J_\rho(\pi_p)`$，断言它等于 $`\tfrac12+p/20`$。
2. 断言两个分布下 teacher 价值都是 $`\tfrac35`$，且 teacher 在每个状态都选了最优动作。
3. 对每个 $`m`$，以二项求和精确计算 $`\mathbb E\,J(\pi_{\widehat p})`$，断言它等于 $`\tfrac{43}{80}`$ 和 $`\tfrac{41}{80}`$。
4. 枚举四种等可能的标签对：断言 $`\mathbb Ep^+=\tfrac12`$，$`p^2`$ 下差距为 $`\eta^2/8`$，$`1-p^2`$ 下差距为 $`-\eta^2/8`$。

**结果。**

| 量 | 值 | JSON 键（`claims[id=C3]`） |
|---|---|---|
| 实例数 / 断言数 / 最大残差 | 60 / 75 / 0 | `n_instances`、`n_assertions`、`worst_residual` |
| [N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) 的差距（$`m=1,\dots,30`$ 每个都是） | $`1/40`$ | `details.prop5_gap` |
| [N04(b)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) 的差距 | $`\eta^2/8`$（$`1-p^2`$ 下为 $`-\eta^2/8`$） | `details.prop6_gap` |
| 状态 | PASS | `status` |

**说明了什么。** 两种分离都是精确恒等式，不是近似。两个数据源可以在选择器可能检查的所有静态量上一致，包括每个 aliased 学生的价值、teacher 的价值、标签边缘分布和平均梯度，却仍然训练出不同的学生。

### C23 (a)–(b) — 训练出的网络中同样存在分离（受控神经网络）

**动机。** 两个命题用的是理想化的频率学习器。这里要回答的是：用梯度方法训练的真实非线性网络，是否也会区分这两个静态等价的数据源。如果差距为零，就说明这种分离只是理想化学习器的产物。

**设置。**

- 网络：one-hot context → 4 个 tanh 单元 → sigmoid 动作概率，初始化固定，在标签块上用 cross-entropy 训练。
- 优化器：普通 SGD，$`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$；Adam（$`\beta=(0.9,0.999)`$，$`\epsilon=10^{-8}`$，带 bias correction），$`\eta\in\lbrace 0.01,0.1\rbrace`$。
- 更新次数 $`R\in\lbrace 1,4,40\rbrace`$，数据源分布取自[N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md)、[N04(b)](../proofs/N04_static_vs_learning_sufficiency_ZH.md)。
- $`R\le4`$ 时精确枚举所有训练历史；$`R=40`$ 时每个数据源做 2,400 次独立训练，SE 按训练次数计算。
- 种子 `SeedSequence([20261010, 23])`。

**结果（除标注 SE 者外均为精确枚举；差距 = 参照 − 另一数据源，或 独立 − 配对）。**

| 数据源对 | 优化器，$`\eta`$ | $`R=1`$ | $`R=4`$ | $`R=40`$（SE） |
|---|---|---:|---:|---:|
| [N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) | SGD 0.01 | 1.21e-4 | 4.81e-4 | 4.38e-3 (1.7e-5) |
| [N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) | SGD 0.1 | 1.21e-3 | 4.48e-3 | 2.10e-2 (9.2e-5) |
| [N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) | SGD 1.0 | 1.10e-2 | 1.88e-2 | 2.44e-2 (2.0e-4) |
| [N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) | Adam 0.01 | 5.04e-4 | 2.00e-3 | 1.54e-2 (6.9e-5) |
| [N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) | Adam 0.1 | 4.91e-3 | 1.54e-2 | 2.40e-2 (1.9e-4) |
| [N04(b)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) | SGD 0.1 | 3.46e-4 | 1.15e-3 | 2.0e-3 (1.1e-3) |
| [N04(b)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) | SGD 1.0 | 2.86e-2 | 1.86e-2 | 1.61e-2 (2.7e-3) |
| [N04(b)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) | Adam 0.1 | 8.75e-3 | 2.76e-2 | 1.23e-2 (2.4e-3) |

数值取自 `controlled_neural.details.configurations[task∈{Prop5,Prop6}].neural_gap` 与 `.SE`。

在小步长 SGD 下，测得的差距与网络的局部切线预测一致。例如[N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md)、SGD $`\eta=0.01`$、$`R=1`$：测得 1.21170e-4，预测 1.21171e-4（`.local_affine_SGD_prediction`）。

**说明了什么。**

- 和定理中一样，真实网络中的静态等价反馈源也会训练出不同的学生。
- 每一行的方向都与理论一致，差距随步长增大，这正是学习后果效应应有的表现；在[N04(a)](../proofs/N04_static_vs_learning_sufficiency_ZH.md) 中，差距还随更新次数增大。
- 差距大小依赖于优化器：同一 $`\eta`$ 下，Adam 的差距大于 SGD。这种依赖关系在 N11–N12 中分析。

适用范围是受控的小网络，不是 Gemma，也不是 benchmark。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C3,C23 --output theory/independent_checks/subset.json
```

## 产物

- `theory/independent_checks/results_independent_v014.json`：`claims[id=C3]`；`controlled_neural.details.configurations`（按 `task` = `Prop5` 或 `Prop6` 过滤）。
- `theory/independent_checks/check_v014_independent.py`：`check_C3`、`check_C23`。
- 已有的受控神经网络图：`theory/independent_checks/figs/C23_controlled_neural.png`。
