# Proposition 5.7 · cold 步在一阶就分开均值相同的反馈

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-07_cold_steps_separate_equal_mean_ZH.md) · [证明](../proofs/05-07_cold_steps_separate_equal_mean_ZH.md) · [实验](../experiments/05-07_cold_steps_separate_equal_mean_ZH.md) · [English](05-07_cold_steps_separate_equal_mean.md)

**本页检验什么.** [Check 5.10](../experiments/05-07_cold_steps_separate_equal_mean_ZH.md#c5-10) 精确核对 cold 差距 $`\lvert w\rvert\eta/2`$，并测量 warm 状态下归一化差距随 warmth 的标度。

**结论.**

- **Proposition 5.7（cold 步分开均值相同的反馈）.** cold 状态下的差距恰为 $`\lvert w\rvert\eta/2`$。在 warm 状态下，归一化后的差距按 $`\kappa^{0.9995}`$ 缩放。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

**Adam 与 AdamW.** Adam 带 bias correction，$`\epsilon`$ 在根号外，可选 decoupled weight decay（AdamW），不用 AMSGrad。计数器为 $`t\ge1`$ 的一次更新作用在 $`z=(\theta,m,v,t-1)`$ 与 consumed gradient $`G\in\mathbb R^p`$ 上。逐坐标地，

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- 超参数满足 $`\beta_1\in[0,1)`$，$`\beta_2\in(0,1)`$，$`\eta>0`$，$`\lambda_{\rm wd}\ge0`$，$`\epsilon\ge0`$。运算都是精确实数运算。
- 下一个完整状态为 $`z^+=(\theta^+,m^+,v^+,t)`$。
- *cold* 状态指 $`m=v=0`$ 且 $`t=1`$；*warm* 状态指对每个坐标 $`i`$ 有 $`v_i>0`$。

**符号映射.** 对 $`\epsilon>0`$，$`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$。对 $`\epsilon=0`$，*理想符号更新*为 $`h_0(g)_i=\mathrm{sign}(g_i)`$，约定 $`\mathrm{sign}(0)=0`$。这个约定是定义的一部分；字面上的浮点 Adam 在 $`\epsilon=0`$ 时会算出 $`0/0`$。

**效用.** *即时参数效用*是 $`\theta_1`$ 的有界可测函数；*完整状态效用*是下一个完整状态 $`(\theta_1,m_1,v_1,1)`$ 的有界可测函数。consumed gradient 的分布 $`P,Q`$ 是 $`\mathbb R^p`$ 上的概率分布；在有限情形的陈述中，它们支撑在有限字母表 $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$ 上。记 $`\bar\theta=(1-\eta\lambda_{\rm wd})\theta_0`$。

*Warm 状态的记号（[Definition 5.4](../statements/05-04_warmth_frozen_denominator_surrogate_ZH.md#s5-4)）.*

考虑一次计数器为 $`t\ge1`$、作用在 $`z=(\theta,m,v,t-1)`$ 上的 Adam 更新，并设 $`v_i>0`$，或在某些 $`v_i=0`$ 时 $`\epsilon>0`$。

- *保留的分母、warmth、相对步长.* 对 consumed gradient $`G`$，

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^{\,t}}},\qquad\kappa_i(G)=\frac{(1-\beta_2)G_i^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(G)=\infty\ \ (v_i=0),\qquad u_i(G)=\frac{\lvert n_i(G)\rvert}{a_i+\epsilon}.
```

- 当 $`v_i>0`$ 时，$`d_i=a_i\sqrt{1+\kappa_i}`$，$`v_i^+=\beta_2v_i\,(1+\kappa_i)`$。warmth $`\kappa_i`$ 就是新梯度对保留下来的二阶矩造成的相对扰动。
- *Frozen-denominator surrogate.* 取固定的对角 preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$，

```math
\tilde\theta^+(G)=(1-\eta\lambda_{\rm wd})\theta-\eta\,\Phi\,n(G),\qquad\tilde\Delta(z,G)=\big(\tilde\theta^+(G)-\theta,\;m^+(G)-m,\;v^+(G)-v\big),\qquad\tilde z^+(G)=\big(\tilde\theta^+(G),\,m^+(G),\,\beta_2v,\,t\big).
```

- $`\tilde\theta^+`$ 是带固定 preconditioner $`\Phi`$ 的一步 momentum SGD。$`\tilde\Delta`$ 是 *frozen-denominator 增量*：它的 $`(\theta,m)`$ 块关于 $`G`$ 仿射，$`v`$ 块关于 $`G^{\odot2}`$ 仿射。$`\tilde z^+`$ 还把二阶矩冻结在 $`\beta_2v`$，是 Proposition 5.6 中用来比较的状态。对 $`g\in\mathbb R^p`$，$`\tilde z^+(g)`$ 表示同一映射在 consumed gradient 为 $`g`$ 时的值。

**状态标签.**

- `PASS` 表示实现的一致性断言全部通过。
- `INFO` 表示经验诊断：每个单元格单独标 PASS、FAIL 或 UNAVAILABLE，整组不作通过与否的判断。

除非另有说明，网络都是 `v014:C23` 的受控小网络：one-hot 上下文、4 个 tanh 单元、sigmoid 输出、平均 cross-entropy、float64。这些都不是真实模型。

**被检验的陈述（[Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean_ZH.md#s5-7)）.** 取 cold 状态（$`m=v=0`$，$`t=1`$）、理想符号映射 $`h_0`$（Proposition 5.3）、标量参数，以及即时效用 $`V(\theta)=\tfrac12+w\theta`$，限制在 $`V\in[0,1]`$ 的邻域内。

- **(a) 均值相同.** 令 $`P_0(G=1)=\tfrac14`$，$`P_0(G=-\tfrac13)=\tfrac34`$，$`P_1(G=0)=1`$。两者均值都是 $`0`$，而 $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$。
- **(b) 任意固定阶以下的矩都相同.** 对 $`N\ge1`$，令 $`P_N(Z=i/N)=\binom Ni/2^{N-1}`$（$`i`$ 为偶数）与 $`Q_N(Z=i/N)=\binom Ni/2^{N-1}`$（$`i`$ 为奇数），$`0\le i\le N`$（即 [Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4) 中的分布），并令 $`G=q-Z`$，$`q\in(0,1/N)`$。两个梯度分布的所有低于 $`N`$ 阶的矩都相同，而 $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$。
- 两种情形下的分离都是 $`\eta`$ 的一阶量。在 warm 状态、同一效用下，Proposition 5.6 把均值相同的分布之间的差距控制在 $`\eta\lvert w\rvert\big(\mathbb E_{P_0}[u\min\lbrace \kappa/2,1\rbrace ]+\mathbb E_{P_1}[u\min\lbrace \kappa/2,1\rbrace ]\big)`$ 以内。

| 检验 | 检验对象 | 代码编号 | 结果 |
|---|---|---|---|
| [Check 5.10](../experiments/05-07_cold_steps_separate_equal_mean_ZH.md#c5-10) | [Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean_ZH.md#s5-7) | `v015:E8`（cold 分离部分） | PASS |
| [Check 5.13](../experiments/05-08_evolution_warmth_ZH.md#c5-13) | 整段训练中的 [Propositions 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)–[5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8) | `v015:E8`（`data.gaps`） | PASS |

<a id="c5-10"></a>

## Check 5.10 · Cold 状态的分离与 warm 状态的缩放（Proposition 5.7）

**结论.**

- cold 状态下，均值相同的两个分布之间的差距恰为 $`\lvert w\rvert\eta/2`$。例如 $`\eta=0.01`$、$`w=-0.7`$ 时为 $`7/2000`$，$`\eta=0.1`$、$`w=1`$ 时为 $`1/20`$。
- warm 状态下，按预条件步长尺度归一化后的差距按 $`\kappa^{0.9995}`$ 增长。

**动机.**

- [Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean_ZH.md#s5-7) 说，cold 状态在 $`\eta`$ 的一阶上就分开均值相同的分布。
- [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6) 说，在 warm 状态下，同样的分离只有 warmth 项 $`\eta\,u\,\kappa`$ 那么大。

**设置**（`v015:E8`，cold 分离部分）.

- cold：命题中的精确有理分布，$`\eta\in\lbrace 0.01,0.1\rbrace`$，$`w\in\lbrace -0.7,1\rbrace`$，分别用理想符号映射和 $`\epsilon=10^{-8}`$ 计算。
- warm：同样（不缩放）的分布，取 $`t=1000`$、$`m=0`$；扫描 $`v`$，使 $`\kappa_{\max}`$ 在 $`[10^{-6},10^2]`$ 上取 33 个对数等距值。

**结果**（`v015:E8`，cold 分离部分）.

| 量 | 值 |
|---|---|
| cold 差距，$`\eta=0.01`$，$`w=-0.7`$ | 理想值 $`7/2000=0.0035`$；$`\epsilon=10^{-8}`$ 时 $`0.00349999986`$ |
| cold 差距，$`\eta=0.1`$，$`w=1`$ | 理想值 $`1/20`$；$`\epsilon=10^{-8}`$ 时 $`0.0499999980`$ |
| warm：原始差距对 $`\kappa`$ 的 log–log 斜率 | 1.4995 |
| warm：按预条件步长尺度归一化后的斜率 | 0.9995 |

**为什么支持 Proposition 5.7.**

- cold 差距恰为 $`\lvert w\rvert\eta/2`$，是 $`\eta`$ 的一阶量。
- 在 warm 状态下，改变 $`v`$ 会同时改变 $`\kappa`$ 和步长尺度 $`u\propto\kappa^{1/2}`$。所以原始差距按 $`\kappa^{3/2}`$ 缩放，归一化后的差距按 $`\kappa^{1}`$ 缩放，正是 warmth 项 $`\eta\,u\,\kappa`$ 的形式。

### [Check 5.13](../experiments/05-08_evolution_warmth_ZH.md#c5-13) · 整段训练中的 cold 窗口（Propositions 5.6–5.8）

**这里检验什么.** 这项检验的完整描述在 Proposition 5.8 的页面上。对 Proposition 5.7，它检验的是：整段训练中的 [Propositions 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)–[5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8)。

**结论.**

- 在短视野下，矩相同的两个来源之间的差距是 cold 窗口的效应。$`W=10`$ 的 lr = 0 moment warm-up 消除了这一差距的中位数 99%（$`R=1`$）与 97.5%（$`R=4`$）。moment $`N=3`$、$`\eta=0.1`$ 时，差距在 $`R=1`$ 从 0.0769 降到 0.0004，在 $`R=4`$ 从 0.1196 降到 0.0016。
- $`R=40`$ 时，$`W\ge10`$ 仍留下至多 0.049 的差距；在残差最大的地方，SGD 对照也不为零（至多 0.011）。

**设置概要.**

- 不加改动地复用 [Check 5.7](../experiments/05-03_cold_start_adam_sign_quotient_ZH.md#c5-7) 的分支（实际 Adam、冻结 Jacobian 的 Adam、切线仿射 SGD）。
- 网格：5 个任务（moment $`N=2,3,4`$；parity $`N=2,3`$）× $`\eta\in\lbrace 0.01,0.1\rbrace`$ × $`R\in\lbrace 1,4,40\rbrace`$ × $`W\in\lbrace 0,1,2,5,10,20,50\rbrace`$，共 210 个单元格。
- 不同 $`W`$ 与不同分支使用完全相同的历史。历史不超过 4,096 条时精确枚举，否则每个来源独立运行 2,000 次。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

## 产物

- `theory/v015_checks/results_v015_independent.json`：
  - `v015:V8`，以及 `v015:E10`（逐单元格记录在 `data.cells`，汇总在 `data.optimizer_summary`，cold Adam 单元格在 `data.C23_prediction`，设计在 `data.design`）；
  - `v015:V2`、`v015:V3`、`v015:V4`、`v015:E0`、`v015:E1`、`v015:E2`、`v015:E3`；
  - `v015:E8`：冻结分母、warm 仿射、cold 分离与 warmth 轨迹四个部分（最后一部分含 `per_step` 与 `fits`），以及 210 个 `data.gaps` 单元格；
  - `v015:E11`（`data.clipped_mean`、`data.always_active`、`data.warm_Adam`、`data.Lion`、`data.LoRA`）。
- `theory/v015_checks/c23_bundle.json`：精确分布、网络、optimizer 代码，以及每条历史的原始更新。
- `theory/v015_checks/raw/E8_trajectories.npz`：逐副本的 warmth 轨迹。它被 gitignore，由上面第一条命令重新生成；SHA-256 记录在 `v015:E8` warmth 轨迹部分的 `raw_sha256` 中。
- `theory/v016_checks/results_v016.json`：claim 为 `v016:C1.exact`、`v016:C1.moments`、`v016:C1.accumulation`。完整数组在 `theory/v016_checks/results_v016_raw.json.gz` 中，该文件被 gitignore，SHA-256 记录在 `raw_artifact`。
- `theory/independent_checks/results_independent_v014.json`：`controlled_neural`。
- `docs/direction/v015_revision/results_v015.json`：package checks `pkg:T03`–`pkg:T08`。
- 代码：
  - `theory/v015_checks/otlr.py`（`check_V8`、`check_E10`、`check_E11`）；
  - `theory/v015_checks/warmth.py`（`check_E8`）；
  - `theory/v016_checks/check_v016.py`（`v016:C1`）；
  - 说明在 `theory/v015_checks/README.md` 与 `theory/v016_checks/SPEC.md`。
- 图：`../figures/fig5_1_linear_response`、`fig5_2_cold_adam`、`fig5_3_warmth`、`fig5_4_learner_statistics`，各有 `.pdf` 与 `.png`；说明在 `../figures/specs/fig5_1.md` 至 `fig5_4.md`。
