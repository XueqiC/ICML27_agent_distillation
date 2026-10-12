# Corollary 6.12 · 学习器之间的分离

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-12_learner_separations_ZH.md) · [证明](../proofs/06-12_learner_separations_ZH.md) · [实验](../experiments/06-12_learner_separations_ZH.md) · [English](06-12_learner_separations.md)

**本页检验什么.** (a) 由 [Check 6.6](../experiments/06-12_learner_separations_ZH.md#c6-6) 检验：用同一批示范，始终裁剪的学习器误差按 $`K^{-1/2}`$ 下降（从 $`K=1`$ 时的 $`0.091`$ 降到 $`K=10^5`$ 时的 $`0.00024`$），SGD 的最坏误差对每个 $`K`$ 都停在下限 $`0.45`$。(b) 依赖于 [Check 6.4](../experiments/06-11_learner_weights_sgd_warm_adam_ZH.md#c6-4) 检验的权重和 [Check 6.5](../experiments/06-10_water_filling_allocation_ZH.md#c6-5) 检验的分配。结论：PASS。

**本页用到的设定与记号.**

- **环境.** 环境族 $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知（[Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)）。
- **oracle 反馈** $`P^*_\theta`$：若已知 $`\theta`$ 就会使用的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据.** $`K`$ 条独立同分布、分布为 $`P^D_\theta`$ 的示范，以及 $`n`$ 次可以自适应的查询：查询点 $`X_i\in\mathcal X_Q`$ 的结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。整条 transcript 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 从 transcript 与私有随机性（其分布与 $`\theta`$ 无关）到 feedback law $`Q`$ 的任何可测映射。replay、改写、合成都是蒸馏器（[Definition 1.2](../README_ZH.md#s1-2)）。
- **被审计的状态** $`z_1,\dots,z_T`$：固定的 complete learner state，即参数连同 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在各 checkpoint 到达的状态。
- **平均增量** $`\mu^P_t(z)`$：在 $`z`$ 上消费一条来自 $`P`$ 的消息时，complete learner state 的期望变化（[Definition 1.3](../README_ZH.md#s1-3)）。对 oracle 记 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$，**学习器 Jacobian** 为 $`J_t=D_\theta\mu^*_t`$。
- **任务方向** $`g_t=\nabla V_{t+1}(z_t)`$：声明的参考尾段下 continuation value 的梯度（[Definition 1.4](../README_ZH.md#s1-4)），与 $`\theta`$ 无关。
- **学习器相对泛函** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$，梯度 $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$；**一阶 fidelity 误差** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。由 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)，它是被审计状态上 oracle 反馈与 $`Q`$ 之间最终任务得分差距的一阶部分。
- **agent 状态** $`s\in\mathcal S`$，有限集；本页中 $`s`$ 表示 agent 的状态（[Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9)）。
- **逐状态的 oracle 分布.** 在状态 $`s`$ 上 oracle 消息 $`Y\sim P^*_s`$（例如 teacher 在该状态的动作或修正），取值于有限字母表；各状态的分布互不相关，不假设状态之间有泛化。
- **influence function** $`f_s(y)`$：$`\Psi`$ 在 $`s`$ 处、把 $`P^*_s`$ 向 $`y`$ 处点质量移动时的导数；在 $`P^*_s`$ 下均值为零。
- **学习器相对权重** $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$：在 $`s`$ 上多一个 oracle 样本，对估计 $`\Psi`$ 值多少。
- **样本数.** 状态 $`s`$ 共得到 $`N_s=k_s+n_s`$ 个 oracle 样本：$`k_s`$ 来自示范（期望为 $`KH\,d_T(s)`$，$`H`$ 为轨迹长度，$`d_T`$ 为 teacher 的状态访问分布），$`n_s`$ 来自查询，$`\sum_sn_s=n`$。

**被检验的陈述（[Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12)）.**

- **(a) 尺度不变性带来的精确分离.** 设 $`\theta=(\alpha,b)`$，$`b>0`$，oracle 的 consumed gradient 为 $`b\,G_\alpha(Y)`$（例如反馈的权重或 advantage 的尺度），示范的分布只依赖于 $`\alpha`$。
  - 对始终激活的范数裁剪 SGD（裁剪阈值 $`c_{\rm clip}`$，几乎处处 $`\lVert bG_\alpha(Y)\rVert>c_{\rm clip}`$）以及 normalized SGD，oracle 平均增量与 $`b`$ 无关，所以 $`\Psi`$ 只依赖于 $`\alpha`$，$`c(\theta)`$ 在 $`b`$ 方向上没有分量。只要示范关于 $`\alpha`$ 的 Fisher information 非退化，误差随 $`K\to\infty`$ 趋于零（对关于 $`\alpha`$ 的模型用 [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)，$`n=0`$；可实现性对任何使裁剪保持激活的尺度都成立）。
  - 对普通 SGD，$`c(\theta)`$ 的 $`b`$ 分量为 $`-\eta\sum_t\langle g_t,\mathbb E\,G_\alpha\rangle`$；当它非零时，示范不含关于 $`b`$ 的任何信息，由 [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)，任何 $`K`$ 都不够。
  - 所以同一批示范对一个学习器充分，对另一个不充分。若裁剪只以概率 $`p_{\rm act}<1`$ 激活，[Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9) 保证每个裁剪后的平均 consumed gradient 与不含 $`b`$ 的向量 $`c_{\rm clip}\,\mathbb E[G_\alpha/\lVert G_\alpha\rVert]`$ 相差不超过 $`c_{\rm clip}(1-p_{\rm act})`$，分离随之平滑地减弱。
- **(b) 长尾覆盖带来的定量分离.** 由 [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10) 与 [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)(d)：当学生的状态分布有大量低频状态（都处在 warm Adam 的频繁访问区间）、而 teacher 很少到达这些状态时，这些状态承担的信息需求份额在 warm Adam 下比在 SGD 下更大；要达到同样的相对精度，warm Adam 在这些状态上需要相应更多的示范或查询（[Example 6.13](../statements/06-13_long_tail_example_ZH.md#s6-13)）。

<a id="c6-6"></a>

## Check 6.6 · 同一批示范对裁剪 SGD 充分、对 SGD 不充分（Corollary 6.12(a)）

**结论.** PASS。裁剪学习器的增量精确地尺度不变，误差按 $`K^{-1/2}`$ 衰减；SGD 的最坏误差从不低于泛函差距的一半。

**为什么做.** [Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12)(a) 断言一个精确的分离：揭示 $`\alpha`$ 却不揭示尺度 $`b`$ 的示范，对始终裁剪的学习器充分，对 SGD 毫无用处。若裁剪后的增量依赖于 $`b`$，或 SGD 的误差低于下限，就推翻了它。

**设置.**

- $`\alpha=0.65`$；梯度取值 $`\lbrace0.2,-0.8\rbrace`$，乘以 $`b\in\lbrace1,3\rbrace`$；裁剪阈值 $`0.1`$，所以裁剪始终激活；$`\eta R=1`$。
- 示范只观测标签，其分布是 $`\mathrm{Binomial}(K,\alpha)`$，与 $`b`$ 无关。
- $`K=1,10,\dots,10^5`$；60,000 次重复。在两个世界中分别计算裁剪学习器的泛函与一个取中点尺度的 SGD 估计量。

**结果.**

| $`K`$ | 裁剪学习器 MAE | RMS 界 | SGD 最坏误差 | 下限 $`\lvert\Delta\Psi\rvert/2`$ |
|---|---|---|---|---|
| 1 | $`0.0910`$ | $`0.0954`$ | $`1.0448`$ | $`0.45`$ |
| 10 | $`0.0247`$ | $`0.0302`$ | $`0.4714`$ | $`0.45`$ |
| 100 | $`0.00762`$ | $`0.00954`$ | $`0.4502`$ | $`0.45`$ |
| 1,000 | $`0.00240`$ | $`0.00302`$ | $`0.4501`$ | $`0.45`$ |
| 10,000 | $`0.000763`$ | $`0.000954`$ | $`0.4500`$ | $`0.45`$ |
| 100,000 | $`0.000241`$ | $`0.000302`$ | $`0.4500`$ | $`0.45`$ |

- 裁剪增量的尺度不变性：12 次比较全部精确成立。
- 这里用的 SGD 估计量逼近下限；而下限本身对任何估计量成立，因为两个世界的示范分布完全相同。

**为什么支持 Corollary 6.12(a).** 这个分离是精确的：用同一批示范，一个学习器的误差以参数速率消失，另一个对每个 $`K`$ 都离零有固定距离。哪些示范算充分，取决于学习器。

![Figure 6.2](../figures/fig6_2_learner_weights.png)

*Figure 6.2 中与本页有关的是 (d).* 始终裁剪的 SGD 的误差随 $`K`$ 按 $`K^{-1/2}`$ 下降，低于其 RMS 界；SGD 在两个世界上的最坏误差停在混叠下限 $`0.45`$。(a)–(c) 对应 Propositions 6.10 与 6.11。

### [Check 6.4](../experiments/06-11_learner_weights_sgd_warm_adam_ZH.md#c6-4) · 学习器权重：SGD 按 d² 标度，warm Adam 按 d 标度

- **对 Corollary 6.12(b) 检验什么.** 让长尾对 warm Adam 比对 SGD 更重要的那组学习器权重。
- **设置.** state-owned 标量坐标、Bernoulli 的 teacher 标签；在频繁、孤立、过渡、短 horizon 四个区间中，用 common random numbers 测 SGD、始终裁剪的 SGD 与 Adam（$`\beta_1=0.9`$）的敏感度。
- **关键数字.** 敏感度关于 $`d`$ 的斜率：SGD 为 $`1.00`$（权重斜率 $`2`$），频繁区间 Adam 为 $`0.489`$（权重斜率 $`0.98`$）；频繁区间闭式的误差为 0.6–7.1%。
- **结论.** PASS：权重的标度符合 (b) 的需要。

### [Check 6.5](../experiments/06-10_water_filling_allocation_ZH.md#c6-5) · water-filling 使实际误差最小

- **对 Corollary 6.12(b) 检验什么.** 用错误学习器的权重分配查询，是否会损失真实的精度。
- **设置.** 16 个随机 profile，每个 6–10 个状态，权重与 $`d`$ 成正比（与频繁区间的 Adam 相同），800 次 teacher 访问、4,000 次查询，每种分配 60,000 次重复。
- **关键数字.** 用 SGD 型权重做 water-filling，实际平均绝对误差是用正确权重时的 $`1.078`$–$`1.315`$ 倍。
- **结论.** PASS：分配以及信息需求都是学习器相对的。

## 复现

在仓库根目录（只用 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C3 --output /tmp/v016_C3.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

第一条命令把本页的检验组单独重跑到一个文件；不带 `--only` 运行 `theory/v016_checks/check_v016.py` 会重新生成完整结果文件（约 104 秒）。所有检验都在 CPU 上用 float64 运行，随机种子为 `numpy.SeedSequence([160160, group, subcase, ...])`（PCG64）。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C3.separation`、`v016:C3.sensitivity`、`v016:C3.allocation`，每项都含断言、最大残差、失败记录与数据行。
- `theory/v016_checks/results_v016_raw.json.gz`：完整的数据行；不纳入 git，其 SHA-256 记录在主文件的 `raw_artifact` 下。
- `theory/v016_checks/SPEC.md`（`v016:C3` 两节或一节）与 `theory/v016_checks/check_v016.py`。
- `figures/fig6_2_learner_weights.png`（同目录下有 PDF 版本）
