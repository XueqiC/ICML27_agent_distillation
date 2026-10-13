# Theorem 6.3 · 学习器相对的下界

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-03_learner_relative_lower_bound_ZH.md) · [证明](../proofs/06-03_learner_relative_lower_bound_ZH.md) · [实验](../experiments/06-03_learner_relative_lower_bound_ZH.md) · [English](06-03_learner_relative_lower_bound.md)

**本页检验什么.** Theorem 6.3 由 [Check 6.2](../experiments/06-05_achievability_ZH.md#c6-2) 检验，它的主页面是 [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5) 的实验页。在 192 个可辨识的高斯实例上，(a) 的 Pinsker 两点界在定理所取的位移处恰好等于 $`\mathfrak n_{K,n}/4`$，且没有任何两点界超过实际达到的风险。结论：PASS。

**本页用到的设定与记号.**

- **环境.** 环境族 $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知（[Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)）。
- **oracle 反馈** $`P^*_\theta`$：若已知 $`\theta`$ 就会使用的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据.** $`K`$ 条独立同分布、分布为 $`P^D_\theta`$ 的示范，以及 $`n`$ 次可以自适应的查询：查询点 $`X_i\in\mathcal X_Q`$ 的结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。整条 transcript 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 从 transcript 与私有随机性（其分布与 $`\theta`$ 无关）到 feedback law $`Q`$ 的任何可测映射。replay、改写、合成都是蒸馏器（[Definition 1.2](../README_ZH.md#s1-2)）。
- **被审计的状态** $`z_1,\dots,z_T`$：固定的 complete learner state，即参数连同 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在各 checkpoint 到达的状态。
- **平均增量** $`\mu^P_t(z)`$：在 $`z`$ 上消费一条来自 $`P`$ 的消息时，complete learner state 的期望变化（[Definition 1.3](../README_ZH.md#s1-3)）。对 oracle 记 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$，**学习器 Jacobian** 为 $`J_t=D_\theta\mu^*_t`$。
- **任务方向** $`g_t=\nabla V_{t+1}(z_t)`$：声明的参考尾段下 continuation value 的梯度（[Definition 1.4](../README_ZH.md#s1-4)），与 $`\theta`$ 无关。
- **学习器相对泛函** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$，梯度 $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$；**一阶 fidelity 误差** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。由 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)，它是被审计状态上 oracle 反馈与 $`Q`$ 之间最终任务得分差距的一阶部分。
- **Fisher information**：一条示范的 $`I_D(\theta)`$，在 $`x\in\mathcal X_Q`$ 处一次查询的 $`I_Q(\theta;x)`$。对方向 $`h\in\mathbb R^p`$，$`\mathbb I_{K,n}(\theta;h)=K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h`$。
- **learner-relative information norm** $`\mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\lvert c(\theta)^\top h\rvert/\sqrt{\mathbb I_{K,n}(\theta;h)}`$，约定 $`0/0=0`$，$`a>0`$ 时 $`a/0=\infty`$。
- **增长条件 (G)**：存在 $`N\to\infty`$ 与 $`0<a\le b<\infty`$，使对每个单位向量 $`h`$ 有 $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$。

**被检验的陈述（[Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)）.**

- **(a) 精确两点界.** 对任意 $`\theta_0,\theta_1\in\Theta`$ 与任意蒸馏器，

```math
\max_{i\in\lbrace0,1\rbrace}\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\big\lvert\Psi(\theta_1)-\Psi(\theta_0)\big\rvert\Big(1-\sqrt{\tfrac12\big(K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)\big)}\Big),
```

  其中 $`\mathrm{KL}_D=\mathrm{KL}(P^D_{\theta_1}\Vert P^D_{\theta_0})`$，$`\mathrm{KL}_Q(x)=\mathrm{KL}(P^Q_{\theta_1}(\cdot\mid x)\Vert P^Q_{\theta_0}(\cdot\mid x))`$。

- **(b) 局部界.** 设每个 $`\mu^*_t`$ 在 $`\theta_0`$ 附近属于 $`C^1`$，两个实验在 $`\theta_0`$ 处 differentiable in quadratic mean，并且当 $`\tau\to0`$ 时

```math
\mathrm{KL}(P^D_{\theta_0+\tau h}\Vert P^D_{\theta_0})\le\tfrac12\tau^2h^\top I_D(\theta_0)h\,(1+o(1)),\qquad \mathrm{KL}_Q(x)\le\tfrac12\tau^2h^\top I_Q(\theta_0;x)h\,(1+o(1)),
```

  且对 $`x\in\mathcal X_Q`$ 与单位向量 $`h`$ 一致成立。取单位方向 $`h`$，满足 $`c(\theta_0)^\top h\neq0`$ 与 $`\mathbb I_{K,n}(\theta_0;h)\to\infty`$，令 $`\tau=\mathbb I_{K,n}(\theta_0;h)^{-1/2}`$。则

```math
\inf_Q\max_{\theta\in\lbrace\theta_0,\,\theta_0+\tau h\rbrace}\mathbb E_\theta\mathrm{Err}_\theta(Q)\ge\frac{\lvert c(\theta_0)^\top h\rvert}{4\sqrt{\mathbb I_{K,n}(\theta_0;h)}}\,\big(1-o(1)\big).
```

- **(c) 范数就是局部下界.** 在 (G) 下，取 $`h`$ 为 [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2) 中比值的最大点，局部 minimax 下界为 $`\tfrac14\mathfrak n_{K,n}(\theta_0)\,(1-o(1))`$。

### [Check 6.2](../experiments/06-05_achievability_ZH.md#c6-2) · 实际风险介于 𝔫/4 与 𝔫 之间

- **对 Theorem 6.3 检验什么.** 下界常数 $`1/4`$ 是否正是两点论证给出的值，以及是否有蒸馏器能低于它。
- **设置.** 192 个可辨识的高斯线性实例（$`p\in\lbrace2,3,5,8\rbrace`$，$`K\in\lbrace1,10,100\rbrace`$，$`n\in\lbrace0,10,100\rbrace`$）；整数舍入后的 c-optimal design 下的有效 plug-in 估计量，每个实例 30,000 次重复；(a) 的 Pinsker 两点界（Le Cam 方法，[Le Cam, 1973](#ref-lecam1973convergence)，配合 Pinsker 不等式，[Cover & Thomas, 2006](#ref-cover2006elements)）与精确的高斯 TV 两点界，都取在位移 $`\tau=\mathbb I^{-1/2}`$ 处。
- **关键数字.** Pinsker 界在每个实例上都等于 $`0.2500\,\mathfrak n`$；精确 TV 界等于 $`0.3085\,\mathfrak n=(1-\Phi(1/2))\,\mathfrak n`$；实际风险至少是 Pinsker 界的 $`3.15`$ 倍；实际风险$`/\mathfrak n`$ 落在 $`[0.789,0.850]`$ 之内。
- **结论.** PASS。检验中没有蒸馏器接近下界；常数 $`1/4`$ 正是证明在所取位移处给出的值，同一点上的精确界略大。

## 复现

在仓库根目录（只用 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C2 --output /tmp/v016_C2.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

第一条命令把本页的检验组单独重跑到一个文件；不带 `--only` 运行 `theory/v016_checks/check_v016.py` 会重新生成完整结果文件（约 104 秒）。所有检验都在 CPU 上用 float64 运行，随机种子为 `numpy.SeedSequence([160160, group, subcase, ...])`（PCG64）。检验用 NumPy（[Harris et al., 2020](#ref-harris2020array)）做线性代数和随机数，用 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）做优化和统计函数。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C2.risk`，每项都含断言、最大残差、失败记录与数据行。
- `theory/v016_checks/results_v016_raw.json.gz`：完整的数据行；不纳入 git，其 SHA-256 记录在主文件的 `raw_artifact` 下。
- `theory/v016_checks/SPEC.md`（`v016:C2` 两节或一节）与 `theory/v016_checks/check_v016.py`。
