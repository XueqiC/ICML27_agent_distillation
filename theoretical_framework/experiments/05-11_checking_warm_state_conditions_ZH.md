# Remark 5.11 · 在真实模型上检验 warm 状态的条件

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-11_checking_warm_state_conditions_ZH.md) · [实验](../experiments/05-11_checking_warm_state_conditions_ZH.md) · [English](05-11_checking_warm_state_conditions.md)

**本页检验什么.** 一项检验（完整描述在 Proposition 5.6 的页面上）在 warm 仿射界旁边记录了 warmth 的分位数；它说明为什么只看分位数核对不了这个条件。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

**Adam 与 AdamW.** Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）带 bias correction，$`\epsilon`$ 在根号外，可选 decoupled weight decay（AdamW；[Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)），不用 AMSGrad。计数器为 $`t\ge1`$ 的一次更新作用在 $`z=(\theta,m,v,t-1)`$ 与 consumed gradient $`G\in\mathbb R^p`$ 上。逐坐标地，

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- 超参数满足 $`\beta_1\in[0,1)`$，$`\beta_2\in(0,1)`$，$`\eta>0`$，$`\lambda_{\rm wd}\ge0`$，$`\epsilon\ge0`$。运算都是精确实数运算。
- 下一个完整状态为 $`z^+=(\theta^+,m^+,v^+,t)`$。
- *cold* 状态指 $`m=v=0`$ 且 $`t=1`$；*warm* 状态指对每个坐标 $`i`$ 有 $`v_i>0`$。

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

**被检验的陈述（[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11)）.**

- Proposition 5.6 的 warm 状态误差，是按敏感度加权的和 $`\eta\sum_iL_iu_i\min\lbrace\kappa_i/2,1\rbrace`$，再加上 $`L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace`$。关于 $`\kappa`$ 分位数的陈述（例如"大多数坐标的 warmth 很小"）对这两项都不构成验证：少数 $`L_iu_i\kappa_i`$ 很大的坐标就能主导这个和，而第二项取决于最大值。
- 所以在真实模型上应当报告加权和 $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ 及其最大的几项，以及 $`\max_i\kappa_i`$。敏感度 $`L_i`$ 是 continuation value 的性质，单看更新是观察不到的；任何替代量 $`\hat L_i`$（例如留出验证梯度的幅度）都是 proxy，必须作为 proxy 报告。
- 测得的更新映射非可加性（$`\Delta`$ 关于消息离仿射有多远）是 optimizer 映射的性质，不是任务 response 的性质。response 是否仿射必须在 response 本身上测，例如用 [Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6) 的配对替换分数。
- 由 Proposition 5.8(a)，训练后期典型坐标 warmth 的参照值是平台 $`(1-\beta_2)/\beta_2`$，而不是零。

| 检验 | 检验对象 | 代码编号 | 结果 |
|---|---|---|---|
| [Check 5.9](../experiments/05-06_warm_adam_first_order_affine_ZH.md#c5-9) | [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)、[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11) | `v015:E8`（warm 仿射部分） | PASS |

### [Check 5.9](../experiments/05-06_warm_adam_first_order_affine_ZH.md#c5-9) · Warm Adam 关于 consumed gradient 一阶仿射（Proposition 5.6、Remark 5.11）

**这里检验什么.** 这项检验的完整描述在 Proposition 5.6 的页面上。对 Remark 5.11，它检验的是：[Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)、[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11)。

**结论.** 在受控网络的实际 warm 状态上，常数在评估任何一对分布之前就已固定。界对全部 2,496 对分布成立：差距/界最大为 0.988，逐点残差/界最大为 0.998。

**设置概要.**

- 状态：网络的实际 warm 状态，$`t`$ 在 21 到 200 之间。
- 可训练坐标：输出头的 1–4 个坐标，其余固定。
- 消息：每个面板 2–6 条。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

这些检查使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。`v015:V2` 与 `v015:V8` 中的线性规划由 `scipy.optimize.linprog` 调用 HiGHS solver（[Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)）求解。

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
