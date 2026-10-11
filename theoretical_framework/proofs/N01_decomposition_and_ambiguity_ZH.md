# N01 · 遗憾分解与决策歧义 — 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/N01_decomposition_and_ambiguity_ZH.md)

## 设定与假设

**假设 1.1（有限决策问题）。**

- $`\mathcal E`$ 是有限的环境集合，先验为 $`\mu`$。先验以公共初始信息 $`I_0`$ 为条件，$`I_0`$ 包括预训练模型、已知规则和 $`K`$ 条示范。
- $`\Pi\subseteq\Pi_{\rm ref}`$ 是有限策略类，取值 $`J_E(\pi)\in[0,1]`$。其中 $`\Pi`$ 是学生的部署类，$`\Pi_{\rm ref}`$ 是参照类。
- transcript $`D`$ 取值于有限字母表 $`\mathcal D`$，其分布为 $`P(D\mid E)`$。学习器可以做的所有实验，无论是否自适应，都已包含在 $`D`$ 中。
- 学习器是一个 Markov kernel $`\lambda(\cdot\mid D)\in\Delta(\Pi)`$，输出 $`\widehat\pi\sim\lambda(\cdot\mid D)`$。给定 $`D`$ 时，它的私有随机性与 $`E`$ 条件独立。

记后验为 $`\mu(E\mid D)`$。定义

```math
J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi),\qquad J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi),\qquad B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D].
```

策略类有限，所以所有最大值都能取到。

## 陈述

**(a)** **命题（精确分解）。** 在假设 1.1 下，

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\underbrace{\mathbb E[J_E^{\rm ref}-J_E^\Pi]}_{\mathcal R_{\rm rep}}+\underbrace{\mathbb E J_E^\Pi-\mathbb E_D B(D)}_{\mathcal R_{\rm info}}+\underbrace{\mathbb E_D\big[B(D)-\mathbb E[J_E(\widehat\pi)\mid D]\big]}_{\mathcal R_{\rm alg}}, \qquad(1)
```

且 $`\mathcal R_{\rm rep},\mathcal R_{\rm info},\mathcal R_{\rm alg}\ge0`$。

**(b)** **命题（决策歧义）。**$`\mathcal R_{\rm info}=0`$ 当且仅当对每个满足 $`\Pr(D)>0`$ 的 $`D`$，

```math
\bigcap_{E:\,\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing. \qquad(2)
```

**(c)** **命题（鲁棒歧义及其对偶）。** 对有限歧义类 $`C\subseteq\mathcal E`$，令

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{E\in C}\big[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\big]. \qquad(3)
```

则

```math
r(C)=\max_{\mu\in\Delta(C)}\Big[\sum_E\mu_EJ_E^\Pi-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)\Big]. \qquad(4)
```

此外：

- (a) 若每个允许的实验在所有 $`E\in C`$ 下分布相同，则任何学习器在 $`C`$ 上的最坏情况期望遗憾至少为 $`r(C)`$。若 $`C`$ 已知，minimax 混合策略可以达到 $`r(C)`$。
- (b) $`r(C)=0`$ 当且仅当存在某个 $`\pi\in\Pi`$ 在所有 $`E\in C`$ 中都是最优的。

**(d)** **命题（后处理不能创造事实）。** 设 $`D'\sim\kappa(\cdot\mid D)`$ 是一个随机后处理，kernel $`\kappa`$ 不依赖于 $`E`$，即 $`E\to D\to D'`$ 构成 Markov 链。则

```math
\mathbb E\max_{\pi}\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D], \qquad(5)
```

即 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$。若 $`Z\perp E\mid(D,I_0)`$，则 $`B(D,Z)=B(D)`$，即加入 $`Z`$ 不改变 $`\mathcal R_{\rm info}`$。

## 证明

**(a) 的证明。** 加减 $`\mathbb E J_E^\Pi`$ 与 $`\mathbb E_DB(D)`$ 即得恒等式 (1)。下面逐项证明非负性。

- $`\mathcal R_{\rm rep}\ge0`$。 由 $`\Pi\subseteq\Pi_{\rm ref}`$，对每个 $`E`$ 有 $`J_E^\Pi\le J_E^{\rm ref}`$。
- $`\mathcal R_{\rm info}\ge0`$。 固定满足 $`\Pr(D)>0`$ 的 $`D`$。由于逐点有 $`J_E(\pi)\le J_E^\Pi`$，对每个 $`\pi\in\Pi`$ 有 $`\mathbb E[J_E(\pi)\mid D]\le\mathbb E[J_E^\Pi\mid D]`$。对 $`\pi`$ 取最大得 $`B(D)\le\mathbb E[J_E^\Pi\mid D]`$。对 $`D`$ 取期望并用 tower property，得 $`\mathbb E_DB(D)\le\mathbb EJ_E^\Pi`$。
- $`\mathcal R_{\rm alg}\ge0`$。 给定 $`D`$ 时 $`\widehat\pi`$ 与 $`E`$ 条件独立，故

  ```math
  \mathbb E[J_E(\widehat\pi)\mid D]=\sum_{\pi\in\Pi}\lambda(\pi\mid D)\,\mathbb E[J_E(\pi)\mid D]\le\max_\pi\mathbb E[J_E(\pi)\mid D]=B(D).
  ```

  不等号成立，是因为凸组合不会超过其中最大的一项。$`\square`$

**(b) 的证明。** 由 tower property，

```math
\mathcal R_{\rm info}=\sum_{D:\Pr(D)>0}\Pr(D)\,\big(\mathbb E[J_E^\Pi\mid D]-B(D)\big).
```

由(a) 的证明，每个括号都非负。因此 $`\mathcal R_{\rm info}=0`$ 当且仅当每个括号都为零。固定这样一个 $`D`$，取 $`\pi_D\in\arg\max_\pi\mathbb E[J_E(\pi)\mid D]`$，则

```math
\mathbb E[J_E^\Pi\mid D]-B(D)=\sum_{E}\mu(E\mid D)\big(J_E^\Pi-J_E(\pi_D)\big),
```

这是一组非负项之和。

- *($`\Leftarrow`$)* 设 $`\pi^\circ`$ 属于交集 (2)。则 $`\mathbb E[J_E(\pi^\circ)\mid D]=\mathbb E[J_E^\Pi\mid D]\ge B(D)\ge\mathbb E[J_E(\pi^\circ)\mid D]`$，括号为零。
- *($`\Rightarrow`$)* 若括号为零，则所有 $`\mu(E\mid D)>0`$ 的项都为零。于是 $`\pi_D`$ 在每个后验概率为正的环境中都最优，即属于交集。$`\square`$

**(c) 的证明。** 定义遗憾矩阵 $`M_{E,\pi}=J_E^\Pi-J_E(\pi)\in[0,1]`$。对固定的混合 $`\rho`$，函数 $`\mu\mapsto\mu^\top M\rho`$ 是线性的，它在 $`\Delta(C)`$ 上的最大值在顶点处取到。因此

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{\mu\in\Delta(C)}\mu^\top M\rho.
```

这是一个有限两人零和博弈的值，由 von Neumann minimax 定理可以交换 min 与 max：

```math
r(C)=\max_{\mu\in\Delta(C)}\min_{\rho\in\Delta(\Pi)}\mu^\top M\rho=\max_\mu\min_{\pi\in\Pi}(\mu^\top M)_\pi.
```

内层在单纯形上的最小值在纯策略处取到。又因为 $`(\mu^\top M)_\pi=\sum_E\mu_EJ_E^\Pi-\sum_E\mu_EJ_E(\pi)`$，内层最小值等于 $`\sum_E\mu_EJ_E^\Pi-\max_\pi\sum_E\mu_EJ_E(\pi)`$。这就是 (4)。

- *(a)* 若没有实验能区分 $`C`$ 中的环境，则 $`D`$ 的分布在每个 $`E\in C`$ 下都相同，输出分布 $`\bar\rho=\sum_D\Pr(D)\lambda(\cdot\mid D)`$ 也对所有 $`E\in C`$ 相同。学习器在 $`E`$ 中的期望遗憾为 $`(M\bar\rho)_E`$，所以它在 $`C`$ 上的最坏情况至少为 $`\min_\rho\max_E(M\rho)_E=r(C)`$。若 $`C`$ 已知，输出一个极小化者 $`\rho^\star`$ 即可达到 $`r(C)`$。由于 $`\Delta(\Pi)`$ 紧且目标连续，这样的极小化者存在。
- *(b)* 若 $`\pi^\circ`$ 在每个 $`E\in C`$ 中都最优，则 $`\pi^\circ`$ 上的点质量给出最坏情况遗憾 0。反过来，设 $`r(C)=0`$，由 $`\rho^\star`$ 取到，则对每个 $`E\in C`$ 有 $`\sum_\pi\rho^\star(\pi)M_{E,\pi}=0`$。由于 $`M\ge0`$，$`\rho^\star`$ 支撑集中的每个 $`\pi`$ 对所有 $`E\in C`$ 都满足 $`M_{E,\pi}=0`$，任何这样的 $`\pi`$ 都是公共最优策略。$`\square`$

**(d) 的证明。** 令 $`h_\pi(D)=\mathbb E[J_E(\pi)\mid D]`$。由 Markov 性（$`E\perp D'\mid D`$）与 tower property，

```math
\mathbb E[J_E(\pi)\mid D']=\mathbb E\big[\mathbb E[J_E(\pi)\mid D,D']\,\big|\,D'\big]=\mathbb E[h_\pi(D)\mid D'].
```

条件期望的最大值不超过最大值的条件期望：

```math
\max_\pi\mathbb E[h_\pi(D)\mid D']\le\mathbb E\big[\max_\pi h_\pi(D)\,\big|\,D'\big]=\mathbb E[B(D)\mid D'].
```

对 $`D'`$ 取期望，再用 tower property，即得 (5)。由于 $`\mathbb EJ_E^\Pi`$ 不依赖于 transcript，(5) 等价于 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$。

对增广情形，$`Z\perp E\mid D`$ 给出 $`\mu(E\mid D,Z)=\mu(E\mid D)`$。于是对每个 $`\pi`$ 有 $`\mathbb E[J_E(\pi)\mid D,Z]=h_\pi(D)`$，从而 $`B(D,Z)=B(D)`$。$`\square`$

## 注记

1. **三项各自刻画的是什么。**$`\mathcal R_{\rm rep}`$ 是任何信号都消除不了的部分，因为学生执行不了参照最优策略。$`\mathcal R_{\rm info}`$ 是 transcript 无法确定一个公共最优决策时剩下的部分。$`\mathcal R_{\rm alg}`$ 是学习器没能用好已有信息的部分。因此，某个学习器失败并不构成信息下界；对一个固定的学习器，额外信息也未必有用，因为它只能降低 $`\mathcal R_{\rm info}`$。
2. **生成的数据。** 由(d)，对 $`D`$ 做合成改写、复述或 replay 都不能降低 $`\mathcal R_{\rm info}`$。它们可以让已有信息更容易学，从而降低 $`\mathcal R_{\rm alg}`$。预训练 teacher 已有的知识属于 $`I_0`$；新的外部查询则扩大 $`D`$。
3. **(3) 的概率含义。** 设置信集 $`C_D`$ 以概率 $`1-\delta`$ 包含真实环境，且某个混合策略在 $`C_D`$ 上的最坏差距不超过 $`\varepsilon`$。这界定的是该数据事件上混合策略的*期望*遗憾，不是单个抽样策略的遗憾。
4. **等式只在 Bayes 意义下可加。** 分别最大化三项，例如逐项取最坏情况，不能保持恒等式 (1)。
5. **适用范围。** 有限性保证最大值可取到、条件期望有定义；一般可测空间还需要可测选择与可积性条件。这个分解是经典的决策论恒等式，在这里的作用是把三种误差来源分开，框架的其余部分分别处理它们。

## 依赖与来源

- **依赖：** 无（第一个节点）。
- **被以下节点使用：** N02（完美示范下 $`\mathcal R_{\rm info}>0`$ 的实例）、N03（精确的 $`\mathcal R_{\rm info}(K)`$）、N04。

**出处：** [v0.14.1 §2](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#2-first-principles-feasibility-identification-and-use) — Props. 1–3; eqs. (1)–(5). [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §2，命题 1–3，式 (1)–(5)。审查：[Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) A1–A4、D2。
