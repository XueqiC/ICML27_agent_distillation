# Corollary 6.7 · 混叠与 replay 的零价值

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-07_aliasing_zero_value_replay_ZH.md) · [证明](../proofs/06-07_aliasing_zero_value_replay_ZH.md) · [实验](../experiments/06-07_aliasing_zero_value_replay_ZH.md) · [English](06-07_aliasing_zero_value_replay.md)

**本页检验什么.** Corollary 6.7 由 [Check 6.3](../experiments/06-07_aliasing_zero_value_replay_ZH.md#c6-3) 检验：沿示范的零空间，数据分布对直到 $`10^6`$ 的每个 $`K`$ 都精确相同，最坏误差从不低于泛函差距的一半。结论：PASS。

**本页用到的设定与记号.**

- **环境.** 环境族 $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知（[Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)）。
- **oracle 反馈** $`P^*_\theta`$：若已知 $`\theta`$ 就会使用的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据.** $`K`$ 条独立同分布、分布为 $`P^D_\theta`$ 的示范，以及 $`n`$ 次可以自适应的查询：查询点 $`X_i\in\mathcal X_Q`$ 的结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。整条 transcript 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 从 transcript 与私有随机性（其分布与 $`\theta`$ 无关）到 feedback law $`Q`$ 的任何可测映射。replay、改写、合成都是蒸馏器（[Definition 1.2](../README_ZH.md#s1-2)）。
- **被审计的状态** $`z_1,\dots,z_T`$：固定的 complete learner state，即参数连同 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在各 checkpoint 到达的状态。
- **平均增量** $`\mu^P_t(z)`$：在 $`z`$ 上消费一条来自 $`P`$ 的消息时，complete learner state 的期望变化（[Definition 1.3](../README_ZH.md#s1-3)）。对 oracle 记 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$，**学习器 Jacobian** 为 $`J_t=D_\theta\mu^*_t`$。
- **任务方向** $`g_t=\nabla V_{t+1}(z_t)`$：声明的参考尾段下 continuation value 的梯度（[Definition 1.4](../README_ZH.md#s1-4)），与 $`\theta`$ 无关。
- **学习器相对泛函** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$，梯度 $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$；**一阶 fidelity 误差** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。由 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)，它是被审计状态上 oracle 反馈与 $`Q`$ 之间最终任务得分差距的一阶部分。

**被检验的陈述（[Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)）.**

- 若 $`\theta_0\neq\theta_1`$ 满足 $`P^D_{\theta_0}=P^D_{\theta_1}`$、$`\Psi(\theta_0)\neq\Psi(\theta_1)`$，并且 $`n=0`$ 或对每个 $`x`$ 都有 $`P^Q_{\theta_0}(\cdot\mid x)=P^Q_{\theta_1}(\cdot\mid x)`$，则对每个 $`K`$，任何蒸馏器都有 $`\max_i\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$。
- replay、改写、合成都是 transcript 与私有随机性的函数，因此包含在"任何蒸馏器"之中，不能降低这个下界。
- 这个下界是学习器相对的：示范是否充分，取决于示范看不见的方向在**这个学习器**的 $`c`$ 上有没有分量。同一批示范可以对一个学习器充分、对另一个不充分（[Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12)(a)）。
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)(a) 是其中一个实例：两个世界的示范分布相同，学生的最优策略不同；[Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4) 说的是后处理帮不上忙。

<a id="c6-3"></a>

## Check 6.3 · 混叠方向留下任何 K 都去不掉的下限（Corollary 6.7）

**结论.** PASS。对每个混叠模型与每个 $`K\in\lbrace1,10,100,10^6\rbrace`$，$`\theta`$ 与 $`\theta+h`$ 处的数据分布精确相同，最坏误差至少是泛函差距的一半。

**为什么做.** [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7) 断言：若 $`h`$ 在示范的零空间中且 $`c^\top h\neq0`$，示范再多也没有用。数据分布出现非零差别，或某个估计量在两个世界中都好于 $`\lvert\Delta\Psi\rvert/2`$，都会推翻它。

**设置.**

- 8 个混叠模型（$`A_D`$ 有一个精确的零列，$`A_Dh=0`$，不涉及数值容差），$`n=0`$，$`K\in\lbrace1,10,100,10^6\rbrace`$：32 个实例、3,408 条断言。
- 比较 $`\theta`$ 与 $`\theta+h`$ 处示范充分统计量的均值与协方差；检查下界背后的确定性三角不等式，并记录取到下界的中点估计量。

**结果.**

- 均值与协方差之差：208 次比较全部精确为 $`0`$。
- 三角不等式：3,200 次检查，最大残差 $`5.6\times10^{-17}`$（容差 $`2\times10^{-14}`$）。
- 例：泛函差距 $`0.1866`$，minimax 下限 $`0.0933`$，中点估计量在每个 $`K`$ 上都取到它（Figure 6.1(c)）。

**为什么支持 Corollary 6.7.** 两个分布是完全相同，而不仅是接近，所以下限是精确的，且与 $`K`$ 无关；replay 或改写示范（即 [Blackwell, 1953](#ref-blackwell1953equivalent) 意义下 transcript 的 garbling）帮不上忙，正是这个机制。

![Figure 6.1](../figures/fig6_1_information_bound.png)

*Figure 6.1 中与本页有关的是 (c).* 八个混叠实例的最坏误差下限 $`\lvert c^\top h\rvert/2`$ 随 $`K`$ 的变化，在每个实例上都是水平线（Corollary 6.7）。(a)、(b) 分别对应 Theorems 6.3/6.5 与 Lemma 6.4。

## 复现

在仓库根目录（只用 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C2 --output /tmp/v016_C2.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

第一条命令把本页的检验组单独重跑到一个文件；不带 `--only` 运行 `theory/v016_checks/check_v016.py` 会重新生成完整结果文件（约 104 秒）。所有检验都在 CPU 上用 float64 运行，随机种子为 `numpy.SeedSequence([160160, group, subcase, ...])`（PCG64）。检验用 NumPy（[Harris et al., 2020](#ref-harris2020array)）做线性代数和随机数，用 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）做优化和统计函数。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C2.aliasing`，每项都含断言、最大残差、失败记录与数据行。
- `theory/v016_checks/results_v016_raw.json.gz`：完整的数据行；不纳入 git，其 SHA-256 记录在主文件的 `raw_artifact` 下。
- `theory/v016_checks/SPEC.md`（`v016:C2` 两节或一节）与 `theory/v016_checks/check_v016.py`。
- `figures/fig6_1_information_bound.png`（同目录下有 PDF 版本）
