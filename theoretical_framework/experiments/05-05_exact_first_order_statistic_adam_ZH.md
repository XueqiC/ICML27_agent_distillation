# Proposition 5.5 · Adam 的精确一阶统计量与冻结分母误差

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-05_exact_first_order_statistic_adam_ZH.md) · [证明](../proofs/05-05_exact_first_order_statistic_adam_ZH.md) · [实验](../experiments/05-05_exact_first_order_statistic_adam_ZH.md) · [English](05-05_exact_first_order_statistic_adam.md)

**本页检验什么.** [Check 5.8](../experiments/05-05_exact_first_order_statistic_adam_ZH.md#c5-8) 用对抗搜索寻找违反冻结分母误差界 (b) 的情形。[Check 5.16](../experiments/05-10_first_order_statistic_by_learner_ZH.md#c5-16) 的完整描述在 Theorem 5.10 的页面上，它说明梯度均值相同时一阶项全部由 $`v`$ 块承载，与 (a)、(c) 一致。

**结论.**

- **Proposition 5.5(b)（冻结分母误差）.** 6,944 个经对抗搜索的状态上没有违反。最好的比值是 1.0，界是紧的。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

- **状态.** 完整 learner state 为 $`z=(\theta,o)\in\mathcal Z\subseteq\mathbb R^p\times\mathbb R^q`$。$`\theta`$ 是可训练参数（例如 LoRA adapter），$`o`$ 是 optimizer state（例如 Adam 的矩 $`m,v`$ 与步数计数器）。需要用到计数器时显式写出。
- **反馈与梯度.** 第 $`t`$ 轮，feedback law $`P_t(\cdot\mid z)`$ 给出消息 $`Y`$。学习器在 loss 归一化与累加之后计算 consumed gradient $`g=G_t(z,Y,\xi)\in\mathbb R^p`$，$`\xi`$ 是声明的更新随机性。若有裁剪，裁剪算作 optimizer 的一部分。
- **Optimizer.** optimizer 是一个可测的增量映射 $`z^+=z+\Delta_t(z,g)`$。被比较的 feedback law 共用同一个 optimizer 与同一个梯度映射，只在消息分布上不同。换一个 optimizer 就是换一个 $`\Delta`$，用同样的方式比较。
- **本节局部复用的字母.** 本节中 $`s`$ 表示 Adam 二阶矩已吸收的梯度计算次数（Proposition 5.8），与第 6 节的 agent 状态 $`s`$ 无关。裁剪阈值记为 $`c`$，weight decay 系数记为 $`\lambda_{\rm wd}`$。

分布 $`P`$ 的学习 kernel 为

```math
(K_t^Pf)(z)=\mathbb E_{Y\sim P_t(\cdot\mid z),\,\xi}\,f\big(z+\Delta_t(z,G_t(z,Y,\xi))\big).
```

对 $`(t,z)`$ 处的分布 $`P`$，消息层面的平均增量为 $`D_t(z,y)=\mathbb E_\xi\Delta_t(z,G_t(z,y,\xi))`$，并记

```math
\mu_t^P(z)=\mathbb E_{Y\sim P}D_t(z,Y),\qquad m_{k,t}^P(z)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(z,G_t(z,Y,\xi))^{\otimes j}.
```

- **Continuation value.** $`V^P_{t+1}:\mathcal Z\to[0,1]`$ 是协议 $`P`$ 的 continuation value：之后的更新都按 $`P`$ 进行，然后部署学生并用 $`F_E`$ 打分（[Definition 1.4](../README_ZH.md#s1-4)）。$`L(P)=\mathbb E\,F_E(Z_R^P)`$，$`d_t^Q`$ 是协议 $`Q`$ 下 $`Z_t`$ 的分布。
- **范数.** 在 $`\mathbb R^{p+q}`$ 上固定一个范数 $`\lVert\cdot\rVert`$，其对偶范数记为 $`\lVert\cdot\rVert_*`$。

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

**被检验的陈述（[Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)）.** 在 Definition 5.4 的设定下：

- **(a) 精确统计量.** Adam 增量为 $`\Delta=(\Delta\theta,\Delta m,\Delta v,1)`$，其中 $`\Delta\theta=-\eta\lambda_{\rm wd}\theta-\eta\,n(G)/(d(G)+\epsilon)`$，$`\Delta m=(1-\beta_1)(G-m)`$，$`\Delta v=(1-\beta_2)(G^{\odot2}-v)`$。Proposition 5.1 中的一阶统计量就是精确均值 $`\mu=\mathbb E\,\Delta(z,G)`$。它的参数块是 $`G`$ 的分布的非线性泛函：若 $`\beta_1<1`$，且对某个坐标 $`i`$ 有 $`v_i>0`$ 或 $`\epsilon>0`$，则存在两个有限支撑的分布，$`\mathbb E\,G`$ 相同、$`\mathbb E\,G^{\odot2}`$ 也相同，但参数块 $`\mu_\theta`$ 不同。
- **(b) 冻结分母误差.** 对每个坐标与每个 $`G`$，

```math
\lvert\theta_i^+(G)-\tilde\theta_i^+(G)\rvert\le\eta\,u_i(G)\,\min\lbrace \kappa_i(G)/2,\;1\rbrace .
```

- **(c) 两种近似要分开.** frozen-denominator 平均增量 $`\tilde\mu=\mathbb E\,\tilde\Delta(z,G)`$ 由 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ 精确决定，且 $`\lvert\mu_{\theta,i}-\tilde\mu_{\theta,i}\rvert\le\eta\,\mathbb E[u_i\min\lbrace\kappa_i/2,1\rbrace]`$，$`m`$ 块与 $`v`$ 块则完全相同。所以"warm Adam 只需要 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$"这一说法依赖两个不同的近似：
  - Proposition 5.1 的一阶近似，余项由增量的二阶矩控制，分块结构见 Remark 5.2；
  - frozen-denominator 近似，误差由 (b) 控制，随 warmth 变小。

| 检验 | 检验对象 | 代码编号 | 结果 |
|---|---|---|---|
| [Check 5.8](../experiments/05-05_exact_first_order_statistic_adam_ZH.md#c5-8) | [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)(b) | `v015:E8`（冻结分母部分） | PASS |
| [Check 5.16](../experiments/05-10_first_order_statistic_by_learner_ZH.md#c5-16) | [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(c)、[Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)(a) | `v015:E11`（warm Adam） | PASS |

<a id="c5-8"></a>

## Check 5.8 · 冻结分母误差（Proposition 5.5(b)）

**结论.** 在 6,944 个经对抗搜索的 optimizer 状态上，$`\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\,u_i\min\lbrace\kappa_i/2,1\rbrace`$ 没有被违反。最好的比值是 1.0000000000000002，即在浮点舍入意义下取等号，所以界不能再改进一个常数因子。

**动机.** (b) 是唯一一处把真实的 Adam 分母换成冻结分母的地方。界只要在某处不成立，[Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6) 也就随之不成立。

**设置**（`v015:E8`，冻结分母部分；种子 `[150150,108]`）.

- 随机状态：
  - $`v`$ 在 $`[10^{-12},1]`$ 上 log-uniform；
  - $`m`$、梯度与 $`t\in\lbrace 1,\dots,2000\rbrace`$ 随机；
  - 五组 $`(\beta_1,\beta_2)`$，包括 $`(0.9,0.999)`$。
- 搜索：24 次随机重启 × 80 步爬山，最大化“左边/右边”。
- 差 $`\theta^+-\tilde\theta^+`$ 用有理化的精确运算计算，避免两个舍入后的更新相减时出现抵消；同时保留直接的浮点减法结果。
- 容差：只允许 $`10^{-12}`$ 的相对松弛。

**结果**（`v015:E8`，冻结分母部分）.

| 量 | 值 |
|---|---|
| 核对的状态 | 6,944 |
| 违反 | 0 |
| 最好的“左边/右边” | $`1.0000000000000002`$；见证落在饱和分支 $`\kappa\gg1`$ |

**为什么支持 Proposition 5.5(b).**

- 界成立，其饱和分支能被达到。
- $`\kappa\to0`$ 时第一个分支也渐近地紧，因为 $`(1-(1+\kappa)^{-1/2})/(\kappa/2)\to1`$。

### [Check 5.16](../experiments/05-10_first_order_statistic_by_learner_ZH.md#c5-16) · Warm Adam：二阶矩经由 optimizer state 在一阶上进入（Theorem 5.10(c)、Proposition 5.5(a)）

**这里检验什么.** 这项检验的完整描述在 Theorem 5.10 的页面上。对 Proposition 5.5，它检验的是：[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(c)、[Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)(a)。

**结论.**

- 梯度均值相同时，参数块与 $`m`$ 块的贡献恰为 0。$`v`$ 块贡献 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle=2.504\times10^{-6}`$，与预测相差小于 $`10^{-12}`$。
- 所以对 warm Adam，均值相同的两个分布在一阶上并不等价。

**设置概要.**

- 网络取 $`k=1`$ 个上下文、$`p=13`$ 个参数。
- warm 状态：$`m=0`$，每个坐标 $`v=0.02`$，步数计数 20。
  - 这是一个容许的 warm 状态，不是从 IID warm-up 中抽出来的；
  - 参数项为零依赖于标签分布的对称性。
- Adam，$`\beta=(0.9,0.999)`$，$`\eta=0.03`$。

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

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
