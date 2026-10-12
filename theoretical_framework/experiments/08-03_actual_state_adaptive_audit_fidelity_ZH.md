# Theorem 8.3 · 实际状态上的自适应 audit；保真与改进

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md) · [证明](../proofs/08-03_actual_state_adaptive_audit_fidelity_ZH.md) · [实验](../experiments/08-03_actual_state_adaptive_audit_fidelity_ZH.md) · [English](08-03_actual_state_adaptive_audit_fidelity.md)

**本页检验什么.**

- **[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)（实际状态上的自适应 audit）恰好在所述强度下成立.**
  - 5 组设定、每组 100,000 次运行，收益始终高于两条保证。
  - 符号恒等式在 50 个有理数实例上精确成立。
  - 下置信界与无条件形式都是必要的：见证中精确期望收益为 $`-78/15625`$，条件化偏差为 $`1/4\to119/494`$。
  - 充分的 Hoeffding 代价是每个候选 1,016 个配对副本。

**本页用到的设定与记号.**

全节沿用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 轮次 $`t=0,\dots,R-1`$，$`z`$ 是 complete learner state。对一阶 optimizer，实际更新为 $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$。
- 在第 $`t`$ 轮状态 $`z`$ 处，参照发出 $`Y\sim P^0_{t,z}`$，候选 $`j\in\lbrace1,\dots,M\rbrace`$ 发出 $`Y\sim P^j_{t,z}`$。候选 $`j`$ 的 kernel 记为 $`K^j_t`$。
- 参照 continuation 为 $`V^0_{t+1}`$，$`V^0_R=F_E\in[0,1]`$。response 为 $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$。对消息积分得到 kernel 恒等式

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- 对由候选 $`j`$ 驱动的过程，$`L_E(j)=\mathbb E\,F_E(Z^j_R)`$。所有过程从相同的 $`Z_0`$ 初始律出发。
- 候选 $`j`$ 在 $`(t,z)`$ 处的**局部优势**为

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- 它表示：把一轮参照替换成一轮候选 $`j`$，之后走参照尾段，对最终得分的影响。由 [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2)，整次训练的差距就是这些项在替代过程到达的状态上的和。

**自适应 audit.** **provider 策略** $`S`$ 在每一轮根据历史选择一个 kernel 下标 $`J_t`$。

- 记 $`\mathcal H_t`$ 为第 $`t`$ 轮 audit 之前的历史。在 $`\mathcal H_t`$ 条件下，冻结实际状态 $`Z_t`$、$`M_t`$ 个非参照候选 kernel 组成的有限集合、副本数 $`n_t`$ 与置信分配 $`\delta_t`$，几乎处处 $`\sum_t\delta_t\le\delta`$。
- 对每个候选，从 $`Z_t`$ 出发做 $`n_t`$ 次独立的配对实验。每次实验执行候选更新或参照更新，再执行**完整的**参照尾段与真实部署。每一臂都有正确的边缘分布，配对对比落在 $`[-1,1]`$ 中。
- 记 $`\widehat A_t(j)`$ 为平均对比，并令

```math
e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},\qquad\ell_t(j)=\widehat A_t(j)-e_t,\qquad\ell_t(0)=0 .
```

- 选择 $`J_t\in\arg\max_j\ell_t(j)`$，令 $`\Gamma_t=\ell_t(J_t)\in[0,1]`$。然后从 $`K^{J_t}_t`$ 做一次**新鲜**更新，丢弃 audit 副本。未做 audit 的轮次使用参照，令 $`\Gamma_t=0`$。
- audit 不改变实际的学习器状态与环境状态，参照也不读取 audit 记录。$`\mathcal F_t\supseteq\mathcal H_t`$ 是 audit 与选择之后、新鲜更新之前的历史。

**被检验的陈述（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)）.**

- **(a) 保真.** 若 $`L_E(0)-J_E(\pi_0)\ge\gamma`$ 且 $`L_E(0)-L_E(S)\le\varepsilon`$，则 $`L_E(S)-J_E(\pi_0)\ge\gamma-\varepsilon`$。这里 $`J_E(\pi_0)`$ 是初始学生的价值；$`\varepsilon`$ 可以取 [Theorem 8.2(c)](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) 的证书，$`\gamma`$ 可以取用 $`K`$ 条示范做参照训练时测得的提升。
- **(b) 带符号的恒等式.** 对任何依赖历史、初始律相同的 provider 策略 $`S`$，

```math
L_E(S)-L_E(0)=\sum_{t=0}^{R-1}\mathbb E_{d^S_t}\,A_t(Z_t,J_t).
```

- **(c) 自适应 audit.** (i) 以至少 $`1-\delta`$ 的概率，$`\sum_tA_t(Z_t,J_t)\ge\sum_t\Gamma_t`$。(ii) 对 audit 与训练无条件地，

```math
L_E(\mathrm{adaptive})-L_E(0)\ge-\delta,\qquad L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_t\Gamma_t-2\delta .
```

- **(d) 代价.** 要使半径 $`e_t\le e`$，每个候选需要 $`n_t\ge\lceil2\log(2M_t/\delta_t)/e^2\rceil`$ 个配对副本。$`M=4`$、$`\delta=0.05`$、$`e=0.1`$ 时为 $`\lceil2\log160/0.01\rceil=1016`$。这个数是充分的，不是必要的。

<a id="c8-9"></a>

## Check 8.9 · 有限学习过程上的自适应 audit（Theorem 8.3(c)）

**结论.** 5 组设定、每组 100,000 次独立训练，自适应 audit 的实测收益始终高于两条保证 $`-\delta`$ 与 $`\mathbb E\sum_t\Gamma_t-2\delta`$。被选中的漂移超过 $`\sum_t\Gamma_t`$ 的频率为 0.990–0.999，高于每一个名义 $`1-\delta`$。

**动机.** [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)(c) 给出一个高概率的漂移陈述和两条无条件的期望界。若被违反，会表现为实测收益低于 $`-\delta`$ 或低于 $`\mathbb E\sum\Gamma-2\delta`$，或漂移成立的频率低于 $`1-\delta`$。

**设置.**

- 一个小的有限学习过程：4 个状态，$`R\le6`$，每个状态 $`M\in\lbrace 2,3\rbrace`$ 个候选 kernel，真实 advantage 已知。
- audit 从真实模型中抽取有界的配对对比：完整参考尾段加 Bernoulli 部署。
- 按下置信界选择，$`e_t=\sqrt{2\log(2M/\delta_t)/n_t}`$，之后做一次新的更新。
- 压力设定：小 $`n_t`$、大 $`\delta_t\in[0.1,0.4]`$，以及均值略为负、方差很大的候选。
- 种子 `SeedSequence([20261010, 16])`。
- 每组设定跑 100,000 次完整训练（audit 加更新）；估计收益及其标准误，并以 4 个 SE 的余量检验两条界。

**结果**（`theory/independent_checks/results_independent_v014.json`，条目 `v014:C16`，`details.configurations`）：

| $`R`$ | $`M`$ | $`n`$ | $`\delta`$ | 半径 | 收益（SE） | $`\mathbb E\sum\Gamma`$ | 漂移频率 | ≥ $`1-\delta`$ |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 2 | 8 | 0.4 | 0.759 | 0.00084 (0.00091) | 0.00090 | 0.99412 | ✓ |
| 2 | 2 | 12 | 0.4 | 0.707 | −0.00067 (0.00102) | 0.00088 | 0.99002 | ✓ |
| 3 | 3 | 20 | 0.2 | 0.671 | 0.00003 (0.00106) | 0.00054 | 0.99407 | ✓ |
| 6 | 3 | 64 | 0.1 | 0.429 | 0.00115 (0.00108) | 0.00026 | 0.99628 | ✓ |
| 2 | 3 | 1000 | 0.2 | 0.090 | **0.13638** (0.00097) | 0.05153 | 0.99897 | ✓ |

状态 PASS，共 500,450 个实例、468 条断言。对 327 个不同实例做对抗搜索，最佳余量（界 − 收益）为 $`-0.199`$，即界始终低于收益（`adversarial_best_bound_minus_gain`）。

必要性见证（条目 `v014:C16`，`details`）：

| 见证 | 值 | 含义 |
|---|---|---|
| `point_estimate_exact_gain` | $`-78/15625`$ | 按 $`\widehat A`$ 选择而不减 $`e_t`$，期望上可能亏损，所以需要下置信界 |
| `zero_drift_conditioning_witness` | 条件值 $`119/494`$ 对无条件值 $`1/4`$（状态 1 上 audit 失败概率 $`9/128`$） | 以 audit 通过为条件会使终端值有偏，所以保证必须是无条件的 |
| `conditioning_probe_found_45_violation` | false（搜索了 5 组设定） | 尽管条件值有偏，无条件的界没有被违反 |
| `replicas_required_per_candidate` | 1016 | $`M=4,\delta=0.05,e=0.1`$ 时的 Hoeffding 分配 |

**为什么支持该陈述.** 保证恰好在所述强度下成立，不更强：

- 副本少、$`\delta`$ 大时，audit 几乎从不选择，收益停在 $`-\delta`$ 的下限之内；
- 副本足够（$`n=1000`$）时，它选中有益的候选，收益为 $`0.136`$，远高于 $`\mathbb E\sum\Gamma-2\delta`$。

<a id="c8-10"></a>

## Check 8.10 · 符号恒等式，以及 fidelity 与提升（Theorem 8.3(a)、(b)）

**结论.** 符号 telescoping 恒等式在 50 个有理数过程上精确成立（残差 0）；fidelity 界在三个例子中都取等，其中一个例子里，对一个无效参考的忠实复制会损失 $`0.11`$。

**动机.** Theorem 8.3(b) 是对依赖历史的 provider 选择成立的精确恒等式，(a) 把 fidelity 与提升区分开。两者都做精确检验。

**设置.**

- 50 个带状态相关 kernel 选择的有理数有限过程，外加对 $`\mathrm{LHS}-\mathrm{RHS}`$ 的对抗搜索。
- 三个 fidelity 例子：好的参考（$`\gamma>0`$）、无效的参考（$`\gamma<0`$）、中性的参考（$`\gamma=0`$），各配一个 fidelity 为 $`\varepsilon=1/100`$ 的候选。

**结果**（条目 `v014:C21`）：PASS，834 条断言，最坏残差 0，`adversarial_best_lhs_minus_rhs` = 0。

| 初始 | 参考 | 候选 | $`\gamma`$ | $`\varepsilon`$ | 候选收益 | 收益 ≥ $`\gamma-\varepsilon`$ |
|---:|---:|---:|---:|---:|---:|---|
| 1/2 | 4/5 | 79/100 | 3/10 | 1/100 | 29/100 | ✓（取等） |
| 1/2 | 2/5 | 39/100 | −1/10 | 1/100 | −11/100 | ✓（取等） |
| 1/2 | 1/2 | 49/100 | 0 | 1/100 | −1/100 | ✓（取等） |

**为什么支持该陈述.**

- 符号恒等式精确成立。
- 对一个无效参考的高 fidelity 复制并不能提升初始学生：其收益为 $`-0.11`$。fidelity 只能继承参考自身的收益。

<a id="c8-11"></a>

## Check 8.11 · 自适应门控的精确期望（Theorem 8.3(c)）

**结论.** 不用 Monte Carlo 的精确期望满足两条界；一个略差的候选以 0.00458 的概率通过乐观的门控，所以期望变化可以严格为负。

**动机.** 对自适应门控的独立精确计算，并说明保证为什么带 $`-\delta`$ 而不是 $`0`$。

**结果**（`theory/v014_repro/results_v014_rerun.json`，`adaptive_gate_expectations`；每个候选 $`n=400`$，$`\delta=0.05`$，半径 $`0.136`$）：

| 量 | 值 |
|---|---:|
| 无条件期望收益 | 0.200 |
| 被选中下置信界之和的期望 $`\mathbb E\sum`$ | 0.132 |
| 下界 $`-\delta`$ | −0.050 |
| 下界 $`\mathbb E\sum\Gamma-2\delta`$ | 0.032 |
| 失败概率 | $`1.5\times10^{-8}`$ |
| 不存在精确的不退化：误选概率 / 期望收益（$`n=20`$，$`\delta=0.9`$，候选 0.49 对参考 0.50） | 0.00458 / $`-4.6\times10^{-5}`$ |

**为什么支持该陈述.** 两条界在精确期望下都成立；反例表明，"不比参考差"只能在 $`\delta`$ 以内成立。

<a id="c8-12"></a>

## Check 8.12 · audit 的代价（Theorem 8.3(d)）

**结论.** 每个候选 1,016 个配对副本的 Hoeffding 分配（$`M=4`$，$`\delta=0.05`$，$`e=0.1`$）给出半径 0.09995，与陈述一致；它是充分的，不是必要的。在受控网络中，部署噪声占配对方差的 77%–89%，empirical Bernstein 半径*预测*只需 559 个副本。

**动机.** 定理给出的分配有效但昂贵，在任何真实模型上使用之前都必须知道它的代价。这里在受控网络中测量配对对比的方差，看对方差敏感的半径能省多少。

**设置（`v015:E7`）.**

- 受控网络，一个预先声明的 audit 轮，四个候选 provider 对一个共同参考。
- 一个参考副本在各候选间复用，并保留候选之间的协方差。
- 2,400 个独立的配对 pilot 副本，用逆 CDF 耦合。
- 方差被精确分解为 block 抽样、后续训练和部署三部分。种子 `SeedSequence([150150, 107])`。

**结果**（`docs/direction/v015_revision/results_v015.json` 的 `checks.T13_*`；`theory/v015_checks/results_v015_independent.json` 中的组 `v015:E7`，`data`）：

| 量 | 值 |
|---|---|
| `pkg:T13`：充分的 Hoeffding 配对数（$`M=4,\delta=.05,e=.1`$） | 1016；1016 时半径 = 0.09995；`is_necessary_lower_bound` = false |
| `v015:E7` 的 Hoeffding 分配 | 每个候选 1016；共享参考时 5,080 个训练分支（不复用时 8,128） |
| `v015:E7` 的 empirical Bernstein **预测** | 每个候选 559；$`n=559`$ 时半径 0.0999，基于最大 pilot 方差 0.129 |

配对对比的方差分解（`v015:E7`）：

| 候选 | 均值 | block | 后续训练 | 部署 | 总计 | 部署占比 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −0.084 | 0.0133 | 0.0041 | 0.0908 | 0.1082 | 84% |
| 1 | −0.042 | 0.0084 | 0.0020 | 0.0454 | 0.0559 | 81% |
| 2 | −0.143 | 0.0096 | 0.0060 | 0.1202 | 0.1358 | 89% |
| 3 | 0.057 | 0.0096 | 0.0027 | 0.0410 | 0.0533 | 77% |

**为什么支持该陈述.**

- Hoeffding 分配是充分的，与定理一致。
- 配对方差主要来自部署噪声，而不是训练更新。因此一个对方差敏感、固定 $`n`$ 的半径预测能少用约 45% 的副本。
- 559 是预测值。真正的 audit 必须在固定的 $`n`$ 下计算自己的半径，不能 optional stopping。`v015:E7` 还记录了根据实测集群耗时对 12B 模型做的推算（`data.forecast`），那是规划输入，不是测量。

![Figure 8.3：实际状态上的自适应 audit](../figures/fig8_3_adaptive_audit.png)

*Figure 8.3.* 自适应 audit 的实测收益（±2 SE，每组设定 100,000 次训练，[Check 8.9](../experiments/08-03_actual_state_adaptive_audit_fidelity_ZH.md#c8-9)）从未低于 $`-\delta`$ 或 $`\mathbb E\sum\Gamma-2\delta`$，只有副本足够、能够做出选择时收益才大。第二幅是达到半径 $`e`$ 所需的每候选副本数：$`e=0.1`$ 时 Hoeffding 的 1,016 是充分的，empirical Bernstein 曲线是根据 pilot 方差做的预测。第三幅分解受控网络中配对对比的方差（[Check 8.12](../experiments/08-03_actual_state_adaptive_audit_fidelity_ZH.md#c8-12)），部署噪声占主导。[PDF](../figures/fig8_3_adaptive_audit.pdf)

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C16,C21 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E7 --output theory/v015_checks/quick/s8.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py   # adaptive_gate_expectations
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # T13
```

## 产出文件

- `theory/independent_checks/results_independent_v014.json`：条目 `v014:C16`、`v014:C21`。
- `theory/v015_checks/results_v015_independent.json`：组 `v015:E7`。
- `theory/v014_repro/results_v014_rerun.json`：`adaptive_gate_expectations`。
- `docs/direction/v015_revision/results_v015.json`：`checks.T13_*`。
- 图：`../figures/fig8_3_adaptive_audit`。
