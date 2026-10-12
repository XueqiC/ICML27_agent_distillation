# Proposition 2.3 · 稳健歧义的对偶

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-03_robust_ambiguity_duality_ZH.md) · [证明](../proofs/02-03_robust_ambiguity_duality_ZH.md) · [实验](02-03_robust_ambiguity_duality_ZH.md) · [English](02-03_robust_ambiguity_duality.md)

**本页检验什么.** [Check 2.2](../experiments/02-03_robust_ambiguity_duality_ZH.md#c2-2) 用可手算的 LP 计算典型的两世界实例：原始值与对偶值都是 0.1，minimax 混合与最不利先验都是 $`(0.5,0.5)`$（通过）。[Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity_ZH.md#c2-1) 在 220 个随机实例上分别求解原始 LP 与对偶 LP，确认对偶成立（通过）。

**本页用到的设定.**

- 有限的环境集合 $`\mathcal E`$，先验为 $`\mu`$，以公共初始信息 $`I_0`$（预训练模型、已知规则、$`K`$ 条可信示范）为条件。
- 有限策略类 $`\Pi\subseteq\Pi_{\rm ref}`$，取值 $`J_E(\pi)\in[0,1]`$：$`\Pi`$ 是学生能部署的策略，$`\Pi_{\rm ref}`$ 是参照类（例如 teacher 能执行的策略）。
- transcript $`D`$：学习器在 $`I_0`$ 之外观察到的一切，包括它可以做的每个实验；给定 $`E`$ 时其分布为 $`P(D\mid E)`$，$`\mu(E\mid D)`$ 是后验。
- 学习器：一个 kernel $`\lambda(\cdot\mid D)`$，输出策略 $`\widehat\pi\in\Pi`$；给定 $`D`$ 时，它的私有随机性与 $`E`$ 独立。
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ 与 $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ 是环境 $`E`$ 中的最好值；$`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ 是 Bayes 值，即给定 $`D`$ 时 $`\Pi`$ 中的学生能达到的最好期望回报。
- 三项 regret $`\mathcal R_{\rm rep}`$、$`\mathcal R_{\rm info}`$、$`\mathcal R_{\rm alg}`$ 的定义见 [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1)。
- 歧义类 $`C\subseteq\mathcal E`$ 是有限的环境集合；$`\Delta(C)`$ 是它上面的先验集合。

**被检验的陈述（[Proposition 2.3](../statements/02-03_robust_ambiguity_duality_ZH.md#s2-3)）.** 对有限的歧义类 $`C\subseteq\mathcal E`$，令

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{E\in C}\big[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\big]. \qquad(2.3)
```

则

```math
r(C)=\max_{\mu\in\Delta(C)}\Big[\sum_E\mu_EJ_E^\Pi-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)\Big]. \qquad(2.4)
```

此外：

- **(a)** 若每个允许的实验在所有 $`E\in C`$ 下分布相同，则任何学习器在 $`C`$ 上的最坏期望 regret 至少为 $`r(C)`$。若 $`C`$ 已知，minimax 混合策略可以达到 $`r(C)`$。
- **(b)** $`r(C)=0`$ 当且仅当存在某个 $`\pi\in\Pi`$ 在所有 $`E\in C`$ 中都最优。

<a id="c2-2"></a>

## Check 2.2 · 典型的两世界歧义实例（Proposition 2.3）

**结论.** 通过：原始值与对偶值都是 0.1，minimax 混合策略与最不利先验都是 $`(0.5,0.5)`$。

**为什么做.** 一个可以手算的 Proposition 2.3 实例，最优混合有闭式。

**设置.** 两个环境偏好相反的策略，regret 矩阵 $`M=\begin{pmatrix}0&0.2\cr 0.2&0\end{pmatrix}`$。原始与对偶 LP 由 v0.14 包的验证脚本用 SciPy HiGHS 求解，种子 2026101014。

**结果.**

| 量 | 值 | JSON 键（该检验的条目） |
|---|---|---|
| 原始 $`r(C)`$ | 0.1 | `robust_ambiguity_minimax.primal_regret` |
| 对偶值 | 0.1 | `robust_ambiguity_minimax.dual_regret` |
| minimax 策略混合 | $`(0.5,0.5)`$ | `.policy_mixture` |
| 最不利先验 | $`(0.5,0.5)`$ | `.environment_mixture` |

**为什么支持这个陈述.**

- 两个世界无法区分时，每个学习器的输出分布在两个世界中相同，能达到的最好最坏期望 regret 是 $`r(C)=0.1>0`$，是押注任一策略时 regret 0.2 的一半（Proposition 2.3(a)）。
- 这个值是*混合平均*的 regret，正如陈述页上的说明所指出的：单个抽样策略的 regret 是 0 或 0.2。

### [Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity_ZH.md#c2-1) · regret 分解、歧义判据、对偶与 data processing

- *对本命题检验什么.* (2.3) 与 (2.4) 相等，且值为零恰好在存在共同最优时成立。
- *设置.* [Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity_ZH.md#c2-1) 的 500 个随机有限决策问题中的 220 个（3–6 个环境、2–5 个策略、随机有理值）；用 HiGHS 分别求解原始的"对混合取最小" LP 与对偶的"对先验取最大" LP。
- *结果.* 两个 LP 在每个实例上都一致（最大残差 $`1.33\times10^{-15}`$），值为 0 恰好在存在共同最优的实例上成立。
- *结论.* 通过。

## 复现

```bash
theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py --output theory/v014_repro/results_v014_rerun.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C1 --output theory/independent_checks/subset.json
```

## 产物

- `theory/v014_repro/results_v014_rerun.json`：键 `robust_ambiguity_minimax`。
- `theory/independent_checks/results_independent_v014.json`：`v014:C1` 的条目；代码 `theory/independent_checks/check_v014_independent.py`（`check_C1`）。
