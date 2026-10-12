# Proposition 2.4 · 后处理不能降低信息 regret

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-04_post_processing_cannot_reduce_information_ZH.md) · [证明](../proofs/02-04_post_processing_cannot_reduce_information_ZH.md) · [实验](02-04_post_processing_cannot_reduce_information_ZH.md) · [English](02-04_post_processing_cannot_reduce_information.md)

**本页检验什么.** 没有专门只检验这个命题的检查；有两个检查覆盖它。[Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity_ZH.md#c2-1) 让 500 个随机问题的 transcript 经过随机的与对抗选出的信道 garbling：没有任何 garbling 提高 Bayes 值，用独立数据扩充时它精确不变（通过）。[Check 2.6](../experiments/02-10_few_shot_information_floor_ZH.md#c2-6) 在隐藏比特模型中度量 replay：在全部 944 个精确格点上收益恰为 0（通过）。

**本页用到的设定.**

- 有限的环境集合 $`\mathcal E`$，先验为 $`\mu`$，以公共初始信息 $`I_0`$（预训练模型、已知规则、$`K`$ 条可信示范）为条件。
- 有限策略类 $`\Pi\subseteq\Pi_{\rm ref}`$，取值 $`J_E(\pi)\in[0,1]`$：$`\Pi`$ 是学生能部署的策略，$`\Pi_{\rm ref}`$ 是参照类（例如 teacher 能执行的策略）。
- transcript $`D`$：学习器在 $`I_0`$ 之外观察到的一切，包括它可以做的每个实验；给定 $`E`$ 时其分布为 $`P(D\mid E)`$，$`\mu(E\mid D)`$ 是后验。
- 学习器：一个 kernel $`\lambda(\cdot\mid D)`$，输出策略 $`\widehat\pi\in\Pi`$；给定 $`D`$ 时，它的私有随机性与 $`E`$ 独立。
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ 与 $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ 是环境 $`E`$ 中的最好值；$`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ 是 Bayes 值，即给定 $`D`$ 时 $`\Pi`$ 中的学生能达到的最好期望回报。
- 三项 regret $`\mathcal R_{\rm rep}`$、$`\mathcal R_{\rm info}`$、$`\mathcal R_{\rm alg}`$ 的定义见 [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1)。
- 后处理是不依赖于 $`E`$ 的 kernel $`\kappa(\cdot\mid D)`$，所以 $`E\to D\to D'`$ 是 Markov 链。

**被检验的陈述（[Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)）.** 设 $`D'\sim\kappa(\cdot\mid D)`$ 是随机后处理，kernel $`\kappa`$ 不依赖于 $`E`$，即 $`E\to D\to D'`$ 是 Markov 链。则

```math
\mathbb E\max_{\pi}\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D], \qquad(2.5)
```

即 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$。若 $`Z\perp E\mid(D,I_0)`$，则 $`B(D,Z)=B(D)`$：把 $`Z`$ 加进 transcript 不改变 $`\mathcal R_{\rm info}`$。

### [Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity_ZH.md#c2-1) · regret 分解、歧义判据、对偶与 data processing

- *对本命题检验什么.* 不等式 (2.5)：与环境无关的后处理不能提高 Bayes 值；以及扩充条款 $`B(D,Z)=B(D)`$。
- *设置.* 500 个随机有限决策问题，transcript 有 4 个符号。每条 transcript 经过一个随机的 3 符号信道 garbling；前 30 个实例还经过全部 $`3^4=81`$ 个确定性信道（凸目标的极点）。再附加一个独立的 $`Z`$ 检验扩充。精确有理数运算。
- *结果.* 在所有 garbling（包括对抗选出的极点信道）中，最大的 $`B(D')-B(D)`$ 恰为 0；$`B(D,Z)=B(D)`$ 在每个实例上精确成立。
- *结论.* 通过。

### [Check 2.6](../experiments/02-10_few_shot_information_floor_ZH.md#c2-6) · 示范数曲线、replay 与实际学习器

- *对本命题检验什么.* replay 是与环境无关的后处理，所以必须不改变信息下限。
- *设置.* 隐藏比特风险的精确曲线：$`m\in\lbrace4,8,16\rbrace`$ 个上下文，$`K=0,\dots,32`$ 条示范，预算 $`0,1,2,4`$；*replay* 这种访问方式在与额外示范或定向查询相同的预算下，重新展示已见的标签。
- *结果.* 在全部 944 个格点上 replay 收益恰为 0；$`m=8`$ 时 replay 曲线与"只有 $`K`$ 条示范"即 $`\tfrac12(7/8)^K`$ 重合。
- *结论.* 通过。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C1 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E5 --output theory/v015_checks/quick/sec2.json
```

## 产物

- `theory/independent_checks/results_independent_v014.json`：`v014:C1` 的条目；代码 `theory/independent_checks/check_v014_independent.py`（`check_C1`）。
- `theory/v015_checks/results_v015_independent.json`：`v015:E5` 的条目；代码 `theory/v015_checks/experiments.py`（`check_E5`）。
