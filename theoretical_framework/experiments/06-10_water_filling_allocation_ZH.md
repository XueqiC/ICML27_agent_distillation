# Proposition 6.10 · 显式信息界与 water-filling 最优查询分配

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-10_water_filling_allocation_ZH.md) · [证明](../proofs/06-10_water_filling_allocation_ZH.md) · [实验](../experiments/06-10_water_filling_allocation_ZH.md) · [English](06-10_water_filling_allocation.md)

**本页检验什么.** Proposition 6.10 由 [Check 6.5](../experiments/06-10_water_filling_allocation_ZH.md#c6-5) 检验：在全部 16 个随机 profile 上，water-filling 的实际误差最低；其他分配方式的平均绝对误差高 3% 到 105%，每个预测误差比值与实际比值相差不超过 $`0.022`$。结论：PASS。

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

**被检验的陈述（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)）.** 在 [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9) 下，记 $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$，称为状态 $`s`$ 的**学习器相对权重**（learner-relative weight）。

- **(a) 下界.** 设在 $`W_s>0`$ 的状态上样本数按比例增长（$`N_s=a_sN`$，$`N\to\infty`$）。任何蒸馏器的局部 minimax 一阶误差至少为

```math
\tfrac14\Big(\sum_{s}\frac{W_s}{N_s}\Big)^{1/2}\big(1-o(1)\big).
```

- **(b) 上界.** 设 $`\Psi`$ 关于概率向量 $`(P_s)_s`$ 二阶连续可微。逐状态的经验分布，即在每个状态上 replay 获取到的消息，达到

```math
\mathbb E\lvert\hat\Psi-\Psi\rvert\le\Big(\sum_s\frac{W_s}{N_s}\Big)^{1/2}\big(1+o(1)\big).
```

- **(c) 最优查询（water-filling）.** 在 $`n_s\ge0`$、$`\sum_sn_s=n`$ 下最小化 $`\sum_sW_s/(k_s+n_s)`$（连续松弛），解为

```math
n_s^\star=\Big(\sqrt{W_s/\lambda}-k_s\Big)_+,\qquad \text{其中 }\lambda>0\text{ 由 }\textstyle\sum_sn_s^\star=n\text{ 确定}.
```

  查询流向学习器相对权重高、teacher 覆盖低的状态。

- **(d) 不可约部分.** 若某个状态 $`W_s>0`$ 而 $`k_s=n_s=0`$（学生会到、teacher 不到、也从不查询），它留下的误差下限不能被任何 $`K`$ 消除（[Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)）。

<a id="c6-5"></a>

## Check 6.5 · water-filling 使实际误差最小（Proposition 6.10）

**结论.** PASS。全部 16 个 profile 上 water-filling 的实际平均绝对误差最小，五种对照分配的预测误差比与实际误差比在 Monte Carlo 误差内一致。

**为什么做.** [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10) 既给出最优分配，也通过 $`\sum_sW_s/N_s`$ 预测其他分配会差多少。若某个对照分配的实际误差更低，或某个预测比值落在联合置信区间之外，就推翻了它。

**设置.**

- 16 个随机 profile，每个 6–10 个状态：学生访问频率 $`d`$ 对数等距后打乱，teacher 访问频率 $`d_T`$ 取 Dirichlet，Bernoulli 分布的 $`\pi\in[0.15,0.85]`$，学习器相对权重 $`W_s=d(s)\,e^{N(0,1.44)}`$（关于 $`d`$ 线性，与频繁访问区间的 Adam 一样）。
- teacher 访问 800 次，查询 $`n=4{,}000`$ 次。plug-in 泛函关于各状态的概率精确线性，系数为 $`\sqrt{W_s/(\pi_s(1-\pi_s))}`$，所以它的方差公式是精确的。
- water-filling 先对 $`\lambda`$ 求标量根，再用边际增益交换做整数化。对照：均匀；正比于 $`d`$；用 SGD 型权重 $`W_s\,d(s)`$ 做 water-filling（相当于用了另一个学习器的权重）；按梯度范数（$`\propto d\sqrt{0.04+0.6\pi}`$）；按不确定性（$`\propto\pi(1-\pi)`$）。
- 用二项分布的充分统计量模拟整数次访问：每种分配 60,000 次重复。比值区间是 familywise 95%（对两种比值、所有 profile、五种对照做 Bonferroni 校正）。
- 另加一个情形：某个状态 $`W>0`$，却既没有 teacher 访问也没有查询。

**结果.**

| 对照分配 | 实际 MAE 与 water-filling 之比：最小 / 中位数 / 最大 | $`\lvert`$预测 − 实际$`\rvert`$ 的最大值 |
|---|---|---|
| 均匀 | $`1.136`$ / $`1.415`$ / $`1.858`$ | $`0.010`$ |
| 正比于 $`d`$ | $`1.033`$ / $`1.209`$ / $`1.695`$ | $`0.009`$ |
| 用 SGD 型权重做 water-filling | $`1.078`$ / $`1.165`$ / $`1.315`$ | $`0.011`$ |
| 梯度范数 | $`1.028`$ / $`1.188`$ / $`1.743`$ | $`0.011`$ |
| 不确定性 | $`1.146`$ / $`1.441`$ / $`2.054`$ | $`0.022`$ |

- 预算残差至多 $`5.9\times10^{-11}`$；KKT 残差 $`4.4\times10^{-16}`$；连续与整数分配在所比较的分配中都是最优的。
- 每个 profile 上，最好的对照分配的实际平均绝对误差比 water-filling 高 2.8% 到 21.9%。
- 预测比值与实际比值之间的最大标准化偏差为 $`3.23`$，低于 familywise 临界值 $`3.60`$。
- 不可约情形：权重 $`(1,2)`$，teacher 次数 $`(100,0)`$，不查询。预测方差为 $`+\infty`$；未覆盖状态上的两个候选分布 $`0.25`$ 与 $`0.75`$ 给出完全相同的观测；泛函差距 $`0.5`$，minimax 下限 $`0.25`$。

**为什么支持 Proposition 6.10.**

- 公式 $`\sum_sW_s/N_s`$ 对实际误差比的预测精确到约百分之一，由它导出的分配在每个 profile 上都胜出。
- 用另一个学习器的权重要多付出 8% 到 32% 的误差：分配是学习器相对的，不是数据本身的性质。
- 局限：profile 是合成的，plug-in 泛函是线性的；本检验没有从真实学习器上估计 $`W_s`$。

![Figure 6.2](../figures/fig6_2_learner_weights.png)

*Figure 6.2 中与本页有关的是 (b)、(c).* (b) 各 profile 上 water-filling、均匀、按 $`d`$ 成比例与按 SGD 权重四种分配的实际平均绝对误差；每个 profile 上 water-filling 都最低。(c) 预测与实际的误差比值，带族错误率区间，落在对角线上。(a)、(d) 分别对应 Proposition 6.11 与 Corollary 6.12(a)。

## 复现

在仓库根目录（只用 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C3 --output /tmp/v016_C3.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

第一条命令把本页的检验组单独重跑到一个文件；不带 `--only` 运行 `theory/v016_checks/check_v016.py` 会重新生成完整结果文件（约 104 秒）。所有检验都在 CPU 上用 float64 运行，随机种子为 `numpy.SeedSequence([160160, group, subcase, ...])`（PCG64）。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C3.allocation`，每项都含断言、最大残差、失败记录与数据行。
- `theory/v016_checks/results_v016_raw.json.gz`：完整的数据行；不纳入 git，其 SHA-256 记录在主文件的 `raw_artifact` 下。
- `theory/v016_checks/SPEC.md`（`v016:C3` 两节或一节）与 `theory/v016_checks/check_v016.py`。
- `figures/fig6_2_learner_weights.png`（同目录下有 PDF 版本）
