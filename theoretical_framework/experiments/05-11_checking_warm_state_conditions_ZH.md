# Remark 5.11 · 在真实模型上检验 warm 状态的条件

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-11_checking_warm_state_conditions_ZH.md) · [实验](../experiments/05-11_checking_warm_state_conditions_ZH.md) · [English](05-11_checking_warm_state_conditions.md)

**本页检验什么.** 一项受控网络上的检验，加两项真实模型检验。

- [Check 5.9](../experiments/05-06_warm_adam_first_order_affine_ZH.md#c5-9)（完整描述在 Proposition 5.6 的页面上）在 warm 仿射界旁边记录了 warmth 的分位数；它说明为什么只看分位数核对不了这个条件。
- [Check 5.19](#c5-19)，真实模型：预注册的 warm 判据**在全部六个测量状态上都不成立**，因为它的分母会抵消。与带符号的一阶项比较（事后，明确标注），按敏感度加权的冻结分母误差从 p03 checkpoint 起约为一阶项的 10%，训练初期则比一阶项还大。
- [Check 5.20](#c5-20)，真实模型：response 非可加性的预注册判据在 p10 成立（中位数 0.855），在 update 8 以微弱差距不成立（0.188，阈值 0.2）；只有四个副本。

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

**真实模型设定（用于 [Check 5.19](#c5-19) 与 [Check 5.20](#c5-20)）.**

- *配方.* ALFWorld（[Shridhar et al., 2021](#ref-shridhar2021alfworld)）上用 20 条可信示范（seed 0）做 Seed-SFT；学生为 `google/gemma-4-12B-it`，LoRA rank 16、$`\alpha=32`$、dropout 0（可训练坐标共 65,568,768 个）；AdamW，学习率常数 $`10^{-5}`$，$`\epsilon=10^{-8}`$，全局梯度裁剪 1.0；每个 update 至少 512 个监督 token；完整训练共 469 个 update。
- *状态.* p03、p10、p40 分别是 31,221、104,070（第 115 个 update）、416,280 个监督 token 之后的 checkpoint。
- *surrogate.* $`\tilde F`$ 是 40 条留出的 teacher 成功轨迹上的平均 teacher-forced log-likelihood，这些轨迹从不进入训练。预注册写的是 80 条；为降低打分成本实际用了 40 条，这是一处有记录的偏离。
- *敏感度 proxy.* $`\hat L_i=\lvert\partial\tilde F/\partial\theta_i\rvert`$，是敏感度 $`L_i`$ 的 proxy，不是 Lipschitz 界。
- 两项检验及其判据都在任何作业运行之前预注册。它们是探索阶段的单 seed 运行。

**被检验的陈述（[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11)）.**

- Proposition 5.6 的 warm 状态误差，是按敏感度加权的和 $`\eta\sum_iL_iu_i\min\lbrace\kappa_i/2,1\rbrace`$，再加上 $`L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace`$。关于 $`\kappa`$ 分位数的陈述（例如"大多数坐标的 warmth 很小"）对这两项都不构成验证：少数 $`L_iu_i\kappa_i`$ 很大的坐标就能主导这个和，而第二项取决于最大值。
- 所以在真实模型上应当报告加权和 $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ 及其最大的几项，以及 $`\max_i\kappa_i`$。敏感度 $`L_i`$ 是 continuation value 的性质，单看更新是观察不到的；任何替代量 $`\hat L_i`$（例如留出验证梯度的幅度）都是 proxy，必须作为 proxy 报告。
- 测得的更新映射非可加性（$`\Delta`$ 关于消息离仿射有多远）是 optimizer 映射的性质，不是任务 response 的性质。response 是否仿射必须在 response 本身上测，例如用 [Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6) 的配对替换分数。
- 由 Proposition 5.8(a)，训练后期典型坐标 warmth 的参照值是平台 $`(1-\beta_2)/\beta_2`$，而不是零。

| 检验 | 检验对象 | 代码编号 | 结果 |
|---|---|---|---|
| [Check 5.9](../experiments/05-06_warm_adam_first_order_affine_ZH.md#c5-9) | [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)、[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11) | `v015:E8`（warm 仿射部分） | PASS |
| [Check 5.19](#c5-19) | 真实模型上按敏感度加权的误差 | `v017:R1.r0a` | 预注册判据 FAIL；事后比较 INFO |
| [Check 5.20](#c5-20) | 真实模型上 response 的非可加性 | `v017:R2.r0b` | INFO（p10 满足判据，update 8 未满足） |

### [Check 5.9](../experiments/05-06_warm_adam_first_order_affine_ZH.md#c5-9) · Warm Adam 关于 consumed gradient 一阶仿射（Proposition 5.6、Remark 5.11）

**这里检验什么.** 这项检验的完整描述在 Proposition 5.6 的页面上。对 Remark 5.11，它检验的是：[Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)、[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11)。

**结论.** 在受控网络的实际 warm 状态上，常数在评估任何一对分布之前就已固定。界对全部 2,496 对分布成立：差距/界最大为 0.988，逐点残差/界最大为 0.998。

**设置概要.**

- 状态：网络的实际 warm 状态，$`t`$ 在 21 到 200 之间。
- 可训练坐标：输出头的 1–4 个坐标，其余固定。
- 消息：每个面板 2–6 条。

<a id="c5-19"></a>

## Check 5.19 · 真实模型上按敏感度加权的冻结分母误差（Remark 5.11）

**结论.**

- **预注册判据：六个状态全部 FAIL.** 判据是 $`E_{\rm fd}\le0.1\times\lvert\sum_i\hat L_i\Delta\theta_i\rvert`$。它的分母把非负的 proxy $`\hat L_i`$ 乘上带符号的步长 $`\Delta\theta_i`$，因而互相抵消：在 p10 它只有 $`7.2\times10^{-6}`$，比带符号的一阶项小约 200 倍。这个判据本身写错了，它的失败如实报告。
- **事后比较（明确标注）.** 与带符号的一阶项 $`\lvert\sum_i(\partial\tilde F/\partial\theta_i)\Delta\theta_i\rvert`$ 相比，比值在 p03、p10、p40 约为 0.1，在最初几个 update 大于 1。所以在 warm checkpoint 上，冻结分母描述带有约为一阶项 10% 的误差，并不可以忽略。

**为什么做.** [Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11) 指出 warm 状态的误差是一个按敏感度加权的和，warmth 的分位数核对不了它。这是这个和在真实模型上的第一次测量。

**设置.**

- 六个状态：update 1、4、8 之后，以及 checkpoint p03、p10、p40。预注册列了九个；probe 的 cold 与 warm-up 状态当时没有保存，这是一处有记录的偏离。
- 在每个状态上取一个 consumed batch，逐坐标计算 warmth $`\kappa_i`$、相对步长 $`u_i`$ 与实际步长 $`\Delta\theta_i`$，全部用 float64；$`\tilde F`$ 的梯度给出 $`\hat L_i`$。
- $`E_{\rm fd}=\eta\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$，预注册比值 $`E_{\rm fd}/\lvert\sum_i\hat L_i\Delta\theta_i\rvert`$，事后比值 $`E_{\rm fd}/\lvert\sum_i(\partial\tilde F/\partial\theta_i)\Delta\theta_i\rvert`$。

**结果.**

| 状态 | $`E_{\rm fd}`$ | 预注册比值 | 判据（$`\le0.1`$） | 事后比值 |
|---|---|---|---|---|
| update 1 | 1.19 | 35,472 | FAIL | 29.7 |
| update 4 | 0.0340 | 2,548 | FAIL | 2.13 |
| update 8 | 0.104 | 97.4 | FAIL | 0.984 |
| p03 | $`3.38\times10^{-4}`$ | 19.0 | FAIL | 0.0997 |
| p10 | $`1.48\times10^{-4}`$ | 20.5 | FAIL | 0.0972 |
| p40 | $`1.71\times10^{-4}`$ | 4.09 | FAIL | 0.0837 |

同一批运行存下的 warmth 分位数（每条记录的 `old_quantiles` 字段）显示，在 p10 只有 0.014% 的坐标超过 $`\kappa=1`$，warmth 的中位数为 $`3.1\times10^{-4}`$。只看分位数，这个状态完全是 warm 的；加权和却说冻结分母误差仍有一阶效应的十分之一。

**为什么支持 Remark 5.11.** 这个 remark 的核心论点在 12B 模型上得到确认：$`\kappa`$ 的分位数暗示误差可以忽略，而由少数坐标主导的敏感度加权和约为一阶项的 10%。这项检验还记下了预注册遗漏的另一点：用来比较的一阶项必须保留敏感度的符号，否则会抵消。

<a id="c5-20"></a>

## Check 5.20 · 真实模型上 response 的非可加性（Remark 5.11）

**结论.** INFO。用预注册的 per-token surrogate，p10 的 response 非可加性中位数为 0.855，高于预注册阈值 0.2，并落在预测区间 $`0.631\pm0.25`$ 之内；update 8 为 0.188，略低于阈值，也低于预测区间。只有四个副本，估计噪声很大。

**为什么做.** [Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11) 把更新映射的非可加性与任务 response 的非可加性分开，并指出后者必须在 response 本身上测。这项检验在真实状态上用配对替换做了这件事。

**设置.**

- 状态：update 8 与 p10。四对互不相交的 batch $`(A,B)`$，与 pilot probe 相同。
- 对每一对，分支为 $`\lbrace0,A,B,A\cup B\rbrace`$：替换一个 update，再接 $`T=10`$ 个参照 update 的共同随机数尾段，$`r=4`$ 个副本（[Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)）。
- $`\delta\tilde F(X)=\tilde F(X)-\tilde F(0)`$，$`\nu_{\rm resp}=\lvert\delta\tilde F(A\cup B)-w_A\delta\tilde F(A)-w_B\delta\tilde F(B)\rvert/\lvert\delta\tilde F(A\cup B)\rvert`$，$`w`$ 为 token 权重；报告的是四对的副本均值取中位数。
- 预注册：判据为"中位数大于 0.2 即在 response 层面非可加"；预测为"中位数落在更新映射取值的 ±0.25 内"，裁剪 AdamW 在 p10 的更新映射取值为 0.631。

**结果.**

| 状态 | 中位数 $`\nu_{\rm resp}`$，per-token（预注册的 surrogate） | 中位数 $`\nu_{\rm resp}`$，几何平均 token 概率 | 判据 $`>0.2`$ |
|---|---|---|---|
| update 8 | 0.188 | 0.191 | 不满足（处在边界） |
| p10 | 0.855 | 0.886 | 满足 |

在 p10，并集 batch 的平均效应小于配对分数在副本之间的离散，所以 $`\nu_{\rm resp}`$ 是两个很小且带噪声的数之比；四个副本只够说明结论的方向。

**为什么支持 Remark 5.11.** 这种测量是可行的，而且测的是 response 而不是更新映射，正如该 remark 所要求的。它没有给出精确的值：两个状态上关于阈值的判断都落在四个副本所允许的噪声之内。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

两项真实模型检验从存下的运行输出重新计算所报告的数字：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
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
- `theory/v017_checks/results_v017.json`：条目 `v017:R1.r0a` 与 `v017:R2.r0b`；对应 `theory/v017_checks/check_v017.py` 中的函数 `r_r0a` 与 `r_r0b`。存下的运行输出在 `theory/v017_checks/records/r0a/`（每个状态一个 JSON 文件，含分块明细与存下的 warmth 分位数）与 `theory/v017_checks/records/r0b/`（合并后的配对分数，以及逐对、逐副本的非可加性）。
- 代码：
  - `theory/v015_checks/otlr.py`（`check_V8`、`check_E10`、`check_E11`）；
  - `theory/v015_checks/warmth.py`（`check_E8`）；
  - `theory/v016_checks/check_v016.py`（`v016:C1`）；
  - 说明在 `theory/v015_checks/README.md` 与 `theory/v016_checks/SPEC.md`。
- 图：`../figures/fig5_1_linear_response`、`fig5_2_cold_adam`、`fig5_3_warmth`、`fig5_4_learner_statistics`，各有 `.pdf` 与 `.png`；说明在 `../figures/specs/fig5_1.md` 至 `fig5_4.md`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-shridhar2021alfworld"></a>

**Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht (2021).** ALFWorld: Aligning text and embodied environments for interactive learning. *The Ninth International Conference on Learning Representations (ICLR)*. arXiv [2010.03768](https://arxiv.org/abs/2010.03768).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
