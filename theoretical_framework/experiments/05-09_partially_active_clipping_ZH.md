# Proposition 5.9 · 部分激活的梯度裁剪

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-09_partially_active_clipping_ZH.md) · [证明](../proofs/05-09_partially_active_clipping_ZH.md) · [实验](../experiments/05-09_partially_active_clipping_ZH.md) · [English](05-09_partially_active_clipping.md)

**本页检验什么.** [Check 5.14](../experiments/05-09_partially_active_clipping_ZH.md#c5-14) 说明原始均值相同但裁剪后均值不同的两个分布在一阶上就有差距，所以统计量是裁剪后的均值。[Check 5.15](../experiments/05-09_partially_active_clipping_ZH.md#c5-15) 检验始终激活的情形 $`p_{\rm act}=1`$：平均方向相同时一阶项恰好为 $`0`$。部分激活的误差界本身没有单独的数值检验。

**结论.**

- **Proposition 5.9 与 Theorem 5.10(e)（clipping）.**
  - 原始均值相同、clipped mean 不同的两个分布在一阶就分开，余项斜率为 2.0015。
  - 始终裁剪（$`p_{\rm act}=1`$）时，平均方向相同的分布一阶项恰为 0。
  - Proposition 5.9 中部分激活时的误差界只是两行的不等式，没有单独的数值检验。

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

**状态标签.**

- `PASS` 表示实现的一致性断言全部通过。
- `INFO` 表示经验诊断：每个单元格单独标 PASS、FAIL 或 UNAVAILABLE，整组不作通过与否的判断。

除非另有说明，网络都是 `v014:C23` 的受控小网络：one-hot 上下文、4 个 tanh 单元、sigmoid 输出、平均 cross-entropy、float64。这些都不是真实模型。

**被检验的陈述（[Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)）.** 令 $`\mathrm{clip}_c(g)=g\min\lbrace1,c/\lVert g\rVert\rbrace`$，$`c>0`$；约定 $`G=0`$ 时 $`G/\lVert G\rVert=0`$；令 $`p_{\rm act}=\Pr(\lVert G\rVert>c)`$ 为某个 $`G`$ 的分布下的激活率。则

```math
\mathbb E\,\mathrm{clip}_c(G)=\mathbb E\big[G\,1\lbrace\lVert G\rVert\le c\rbrace\big]+c\,\mathbb E\Big[\frac{G}{\lVert G\rVert}1\lbrace\lVert G\rVert>c\rbrace\Big],\qquad \Big\lVert\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E\frac{G}{\lVert G\rVert}\Big\rVert\le c\,(1-p_{\rm act}).
```

- 对两个分布，裁剪后均值与缩放后的平均方向之间的总偏差至多为 $`c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$。例如两个分布的激活率都是 $`0.9`$ 时，用平均方向代替裁剪后均值，在差值上的误差至多为 $`0.2c`$。
- 只有在始终激活（$`p_{\rm act}=1`$）时，平均方向才恰好等于裁剪后均值。

| 检验 | 检验对象 | 代码编号 | 结果 |
|---|---|---|---|
| [Check 5.14](../experiments/05-09_partially_active_clipping_ZH.md#c5-14) | [Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)、[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(e) | `v015:E11`（clipped mean） | PASS |
| [Check 5.15](../experiments/05-09_partially_active_clipping_ZH.md#c5-15) | $`p_{\rm act}=1`$ 时的 [Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)、[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(e) | `v015:E11`（始终裁剪） | PASS |

<a id="c5-14"></a>

## Check 5.14 · Clipping：统计量是 clipped mean（Proposition 5.9、Theorem 5.10(e)）

**结论.** 两个梯度分布的原始均值相同，clipped mean 相差 $`-\tfrac12`$。它们在一阶上就被分开：预测的差距 $`\tfrac\eta2\sigma'(0.4)`$ 与实测差距吻合，余项的 log–log 斜率为 2.0015。

**动机.**

- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(e) 说，有 clipping 时，反馈必须匹配 $`\mathbb E\,\mathrm{clip}_c(G)`$，而不是 $`\mathbb E\,G`$。
- 如果 clipping 只改变幅度，那么原始均值相同的两个分布在一阶上就会一致。所以原始均值相同、一阶上却有差距，就确认了 clipped mean 才是统计量。
- 两个分布的激活情况不同：$`P`$ 总是被裁剪，$`Q`$ 从不被裁剪。因此 [Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9) 中按事件拆分的两个分支都被用到。

**设置**（`v015:E11`，clipped mean）.

- 线性 logit 的 sigmoid，$`\theta=0.4`$。
- $`P(G=4)=\tfrac14`$，$`P(G=-\tfrac43)=\tfrac34`$；$`Q=\delta_0`$。
- 在 $`c=1`$ 处裁剪；clipped SGD。
- $`\eta\in\lbrace 10^{-3},3\cdot10^{-3},10^{-2},3\cdot10^{-2},10^{-1}\rbrace`$。
- 梯度 $`4`$ 与 $`-4/3`$ 在固定网络的初始化处不会出现，所以改用线性 logit 模型来实现。

**做法.**

- 断言两者原始均值相同、clipped mean 相差 $`-\tfrac12`$。
- 计算精确差距 $`(P-Q)\sigma(\theta+\Delta\theta)`$ 与预测的一阶差距 $`\eta\,\sigma'(0.4)/2`$。
- 断言解析的二阶界 $`\lvert\text{gap}-\text{first order}\rvert\le\eta^2/(12\sqrt3)`$。

**结果**（`v015:E11`，`data.clipped_mean`）.

| $`\eta`$ | 实测差距 | 一阶预测 | 余项 |
|---:|---:|---:|---:|
| 0.001 | $`1.20107\times10^{-4}`$ | $`1.20130\times10^{-4}`$ | $`2.37\times10^{-8}`$ |
| 0.01 | $`1.19892\times10^{-3}`$ | $`1.20130\times10^{-3}`$ | $`2.38\times10^{-6}`$ |
| 0.1 | $`1.17675\times10^{-2}`$ | $`1.20130\times10^{-2}`$ | $`2.46\times10^{-4}`$ |

余项的 log–log 斜率为 2.0015。

![Figure 5.4：各学习器的一阶统计量](../figures/fig5_4_learner_statistics.png)

*Figure 5.4.* 学习器特定的一阶统计量的五个见证（[Check 5.14](../experiments/05-09_partially_active_clipping_ZH.md#c5-14)–[5.18](../experiments/05-10_first_order_statistic_by_learner_ZH.md#c5-18)）：

- (A) 原始均值相同、clipped mean 不同，有 clipping 时出现一阶差距。
- (B) 始终裁剪时，平均方向相同的两个分布只有 $`O(\eta^2)`$ 的差距。
- (C) warm Adam、梯度均值相同时，一阶项完全由 $`v`$ 块承担，与预测一致。
- (D) momentum 占主导的 Lion：参数增量与消息无关。
- (E) 初始化处的 LoRA：零空间中的差别是二阶的，值域中的差别是一阶的。

[PDF](../figures/fig5_4_learner_statistics.pdf)

**为什么支持 Theorem 5.10(e).** 原始平均梯度预测的差距为零；clipped mean 则在一阶上预测出实测差距，余项为 $`O(\eta^2)`$。

<a id="c5-15"></a>

## Check 5.15 · 始终裁剪：只有方向起作用（完全激活时的 Proposition 5.9、Theorem 5.10(e)）

**结论.** 每个梯度都被裁剪、且两个分布的平均单位方向相同时，一阶项恰为 0，差距是二阶的（log–log 斜率 1.99997）。

**动机.** 始终裁剪（$`p_{\rm act}=1`$）时，[Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9) 的误差界 $`c(1-p_{\rm act})`$ 为 0，clipped mean 恰好等于 $`c\,\mathbb E[G/\lVert G\rVert]`$。所以平均方向相同的两个分布，不论幅度如何，在一阶上都必须一致。

**设置**（`v015:E11`，始终裁剪）.

- 二维梯度，$`\lVert G\rVert\ge2>c=1`$。
- $`P`$ 在 $`\lbrace(\pm2,0)\rbrace`$ 上均匀，$`Q`$ 在 $`\lbrace(0,\pm4)\rbrace`$ 上均匀。两者的平均单位方向都是 0，方向的二阶矩不同。
- 线性 logit 模型与 $`\eta`$ 网格同 [Check 5.14](../experiments/05-09_partially_active_clipping_ZH.md#c5-14)。

**做法.** 断言一阶项恰为零，并拟合精确差距的斜率。

**结果**（`v015:E11`，`data.always_active`）.

| $`\eta`$ | 一阶项 | 实测差距 |
|---:|---:|---:|
| 0.001 | 0 | $`7.11\times10^{-8}`$ |
| 0.01 | 0 | $`7.11\times10^{-6}`$ |
| 0.1 | 0 | $`7.06\times10^{-4}`$ |

差距的 log–log 斜率为 1.99997。

**为什么支持 Proposition 5.9 与 Theorem 5.10(e).**

- 始终裁剪时，幅度在一阶上不起作用；只有方向的二阶矩把这两个分布分开。
- 如果两个分布的方向分布也相同，它们的增量分布就完全一致。
- 证据的局限：$`0<p_{\rm act}<1`$ 时，Proposition 5.9 用 $`c(1-p_{\rm act})`$ 界定“只有方向起作用”这一简化离真实值有多远；这一中间情形没有专门的数值检验。

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
