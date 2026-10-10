# N19 · 每种学习器要求反馈携带什么 — 实验

[← 框架总图](../README_ZH.md) · [证明](../proofs/N19_learner_statistics_ZH.md)

**结论.** N19 成立。检验 E11 的五个清单见证全部通过，共 105 个断言，最坏残差 $8.0\times10^{-21}$。每个见证确认定理 C4 的一行：

- **裁剪均值，C4(e).** 原始均值相同、裁剪均值不同的两个梯度分布在一阶上就会分开。预测的一阶差距 $\tfrac\eta2\sigma'(0.4)$ 与实测差距吻合，余项斜率为 2.001。
- **裁剪恒为 active，C4(e).** 平均单位方向一致时，一阶项恰为 0，差距是二阶的（斜率 1.99997）。
- **warm Adam，C4(c).** 梯度均值相同时，参数块与 $m$ 块的贡献恰为 0，$v$ 块贡献 $(1-\beta_2)\langle\nabla_vV,\delta_2\rangle=2.504\times10^{-6}$，与预测吻合到 $10^{-12}$。因此均值相同的分布对 warm Adam 在一阶上并不等价。
- **Lion，C4(f).** 动量占优时，两个差别很大的梯度分布在参数块上的对比恰为 0，没有符号翻转；$m$ 块恰好携带 $0.01\,\delta_1$。
- **LoRA，C4(g).** 在 $B=0$ 处，SGD、heavy-ball 和冷启动 Adam 的第一步对每个分布都逐位不改变 $A$。只在 $A_0^\top$ 零空间中不同的分布，一阶差距低于 $10^{-20}$，差距是二阶的（斜率 2.000）；在值域中不同的分布，一阶差距与实测差距在每个 $\eta$ 上都相差不到 0.3%，在 $\eta=10^{-3}$ 时相对误差不到 $3\times10^{-5}$。

冷启动 Adam 这一行 C4(d) 在节点 [N18](N18_optimizer_transformed_response_ZH.md) 的检验 E10 中得到确认：在均值相同的 C23 分布上，冷启动 Adam 的一阶项 $-\eta\langle\nabla_\theta V,\mathbb E_Ph_\epsilon-\mathbb E_Qh_\epsilon\rangle=0.07568$ 解释了实测差距 $0.07691$，而原始平均梯度项为 $-8.9\times10^{-11}$。

![五个清单见证](../figures/N19_learner_statistics.png)

[PDF](../figures/N19_learner_statistics.pdf)

## E11(a) · 裁剪 SGD：统计量是裁剪均值

**动机.** C4(e) 说，在裁剪下反馈必须匹配的是 $\mathbb E\operatorname{clip}_c(G)$，而不是 $\mathbb E\,G$。如果裁剪只改变幅度，原始均值相同的两个分布就会在一阶上一致；原始均值相同时出现一阶差距，就确认了统计量是裁剪均值。

**设置.**

- 在 logit 线性、$\theta=0.4$ 的 sigmoid 上做的直接梯度分布实验。
- $P(G=4)=\tfrac14$，$P(G=-\tfrac43)=\tfrac34$，$Q=\delta_0$；裁剪阈值 $c=1$；裁剪 SGD。
- $\eta\in\lbrace 10^{-3},3\cdot10^{-3},10^{-2},3\cdot10^{-2},10^{-1}\rbrace $。

梯度 $4$ 与 $-4/3$ 在初始化处的固定 C23 网络里无法出现，所以用线性 logit 模型来实现（`groups[id=E11].notes`）。

**做法.** 断言原始均值相同、裁剪均值相差 $-\tfrac12$。计算精确差距 $(P-Q)\sigma(\theta+\Delta\theta)$ 与预测的一阶差距 $\eta\,\sigma'(0.4)/2$，并断言解析的二阶界 $\lvert$gap − first order$\rvert\le\eta^2/(12\sqrt3)$。

**结果**（`groups[id=E11].data.clipped_mean`）：

| $\eta$ | 实测差距 | 一阶预测 | 余项 |
|---:|---:|---:|---:|
| 0.001 | $1.20107\times10^{-4}$ | $1.20130\times10^{-4}$ | $2.37\times10^{-8}$ |
| 0.01 | $1.19892\times10^{-3}$ | $1.20130\times10^{-3}$ | $2.38\times10^{-6}$ |
| 0.1 | $1.17675\times10^{-2}$ | $1.20130\times10^{-2}$ | $2.46\times10^{-4}$ |

余项的 log–log 斜率为 2.0015（`remainder_slope`）。

**说明了什么.** 原始平均梯度预测差距为零；裁剪均值在一阶上预测出实测差距，余项为 $O(\eta^2)$。

## E11(b) · 裁剪恒为 active：只有方向起作用

**动机.** 当每个梯度都被裁剪时，C4(e) 把统计量化为平均单位方向 $c\,\mathbb E[G/\lVert G\rVert]$。于是平均方向相同的两个分布，不论幅度如何，在一阶上都必须一致。

**设置.** 二维梯度，$\lVert G\rVert\ge2>c=1$：$P$ 在 $\lbrace(\pm2,0)\rbrace$ 上均匀，$Q$ 在 $\lbrace(0,\pm4)\rbrace$ 上均匀。两者的平均单位方向都是 $0$，二阶方向矩不同。模型与 $\eta$ 网格同 E11(a)。

**做法.** 断言一阶项恰为零，并拟合精确差距的斜率。

**结果**（`data.always_active`）：

| $\eta$ | 一阶项 | 实测差距 |
|---:|---:|---:|
| 0.001 | 0 | $7.11\times10^{-8}$ |
| 0.01 | 0 | $7.11\times10^{-6}$ |
| 0.1 | 0 | $7.06\times10^{-4}$ |

差距的 log–log 斜率为 1.99997（`slope`）。

**说明了什么.** 裁剪恒为 active 时，幅度在一阶上消失，只有二阶方向矩能分开这两个分布。若两个分布的方向分布也相同，它们的增量分布就完全一致（C4(e)）。

## E11(c) · warm Adam：二阶矩通过 $v$ 在一阶上进入

**动机.** C4(c) 预测 warm Adam 的一阶统计量是 $(\mathbb E\,G,\ \mathbb E\,G^{\odot2})$，其中二阶矩通过 $v$ 块进入，权重为 $(1-\beta_2)\nabla_vV$。因此均值相同、二阶矩不同的两个分布，应当有一个完全由 $v$ 承载的非零一阶项。

**设置.**

- C23 网络，$k=1$ 个上下文，$p=13$ 个参数。
- warm 状态：$m=0$，每个坐标 $v=0.02$，步数计数器为 20；Adam，$\beta=(0.9,0.999)$，$\eta=0.03$。
- 标签：$P$ 在 $\lbrace0,1\rbrace$ 上均匀，$Q$ 在 $\lbrace\tfrac14,\tfrac34\rbrace$ 上均匀。两者均值都是 $\tfrac12$，二阶矩不同。
- 延续值是之后两步、标签为 $\tfrac14$ 的更新，然后做 moment $N=3$ 部署。
- 延续值在完整状态上的梯度由中心差分求得。

**做法.**

1. 计算 $\delta_1$、$\delta_2$ 以及三块各自的贡献 $\langle\nabla_\theta V,\mu^P_\theta-\mu^Q_\theta\rangle$、$\langle\nabla_mV,\cdot\rangle$ 与 $\langle\nabla_vV,\cdot\rangle$。
2. 断言 $v$ 块等于 $(1-\beta_2)\langle\nabla_vV,\delta_2\rangle$。
3. 与实测差距以及命题 W1（节点 [N12](N12_optimizer_warmth_ZH.md)）的 $L_v\kappa$ 项比较。

**结果**（`data.warm_Adam`）：

| 量 | 数值 |
|---|---:|
| $\max_i\lvert\delta_{1,i}\rvert$（梯度均值相同） | 0 |
| $\max_i\delta_{2,i}$ | 0.1875 |
| 各块一阶贡献 $(\theta,\ m,\ v)$ | $(0,\ 0,\ 2.5043\times10^{-6})$ |
| 预测值 $(1-\beta_2)\langle\nabla_vV,\delta_2\rangle$ | $2.5043\times10^{-6}$（吻合到 $10^{-12}$ 以内） |
| 实测差距 | $8.753\times10^{-6}$ |
| W1 对线性 $v$ 项的界 $L_v\max\kappa$ | $7.127\times10^{-6}$（$\max\kappa=0.0125$） |

**说明了什么.**

- 在 warm Adam 状态下，梯度均值相同的分布会被 $v$ 通道在一阶上分开，系数与预测完全一致。
- 动量块为零，因为 $\delta_1=0$。参数块的均值项 $-\eta c_1P\delta_1$ 也为零；它的 warmth 误差 $\mathbb E_Pe-\mathbb E_Qe$ 在这里恰为零，是因为标签分布对称且 $m=0$。
- 实测差距的其余部分是定理 C1 的余项。
- W1 的残差界覆盖了线性 $v$ 项；N19 把这个残差换成了一个显式统计量 $\mathbb E\,G^{\odot2}$。

## E11(d) · Lion：动量占优使参数块对反馈不敏感

**动机.** C4(f) 说，动量占优的 sign 更新与消息无关，所以即使梯度分布差别很大，参数增量也相同；只有 $m$ 块记录这种差别。

**设置.**

- C23 网络，Lion，$\beta_1=0.9$，$\beta_2=0.99$，$\eta=0.03$。
- 两条消息，标签分别为 $0$ 和 $1$，梯度相差 $\delta_1$，$\max\lvert\delta_1\rvert=1$。
- 动量占优状态：每个坐标 $m=0.2$，使得对每条消息都有 $0.1\lvert G_i\rvert<0.9\cdot0.2$。
- 对照：$m=0.005$。

**做法.** 计算逐坐标的翻转指示和各块的增量差。断言占优情形的参数对比恰为零、对照情形非零，并且 $m$ 块等于 $(1-\beta_2)\delta_1=0.01\,\delta_1$。

**结果**（`data.Lion`）：

| 状态 | 参数块对比 $\max_i\lvert\cdot\rvert$ | 翻转比例 | $m$ 块对比 $\max_i\lvert\cdot\rvert$ |
|---|---:|---:|---:|
| 动量占优，$m=0.2$ | 0（精确） | 0 | 0.0100 $=0.01\max\lvert\delta_1\rvert$ |
| 对照，$m=0.005$ | 0.0600 $=2\eta$ | 0.5 | 0.0100 |

**说明了什么.** 动量占优时，两条消息的参数增量完全相同，所以它们在 $\theta$ 上的一阶统计量就是翻转概率，这里为 $0$。一旦符号翻转，对比就是完整的 $2\eta$ 一步。$m$ 块总是携带 $(1-\beta_2)\delta_1$；当之后没有任何符号改变时，延续值关于 $m$ 局部为常数，这条通道在局部也没有一阶效应（`data.Lion[*].note`）。

## E11(e) · LoRA：第一次更新只看到秩 $r$ 的 sketch

**动机.** C4(g) 说，在 $B=0$ 处 adapter $A$ 在第 1 步不能移动，反馈只能通过 $\Gamma A_0^\top$ 进入。平均全权重梯度的差若落在 $A_0^\top$ 的零空间中，在一阶上应当不可见；落在值域中则应当可见。

**设置.**

- 双线性 sigmoid 模型 $W=W_0+\alpha BA$，$d=k=4$，$W_0=0.2$，$\alpha=1$，$r\in\lbrace1,2\rbrace$，$B_0=0$，$A_0$ 随机。
- 消息是软标签的平均交叉熵目标，能精确实现指定的全权重梯度 $\Gamma$。
- 三组分布对：
  - *零空间:* 平均梯度相差 $\Delta\Gamma$，且 $\Delta\Gamma A_0^\top=0$；
  - *值域:* 差的形式为 $HA_0$；
  - *符号:* sketch $\Gamma A_0^\top$ 的符号分布相同、幅度不同。
- 优化器：SGD、heavy-ball 和冷启动 Adam；逐位审查用 $\eta=0.03$，差距用完整的 $\eta$ 网格。

**做法.**

1. 断言对每个分布、每条消息、每个优化器，第 1 步之后 $A$ 逐位不变。
2. 对零空间对与值域对，计算第 1 步的精确差距与一阶项，并拟合零空间对差距的斜率。
3. 对冷启动 Adam 下的符号对，断言更新等于 $-\eta h_\epsilon(\alpha\Gamma A_0^\top)$，并记录有限 $\epsilon$ 带来的差异。

**结果**（`data.LoRA`）：

| 量 | 秩 1 | 秩 2 |
|---|---:|---:|
| $A$ 逐位不变的检查（SGD、heavy-ball、冷启动 Adam × 全部消息） | 21 / 21 | 21 / 21 |
| 零空间残差 $\max\lvert\Delta\Gamma A_0^\top\rvert$ | $2.6\times10^{-18}$ | $4.8\times10^{-19}$ |
| 零空间对：$\eta=0.1$ 时的一阶项 | $-1.1\times10^{-20}$ | $2.5\times10^{-21}$ |
| 零空间对：$\eta=0.1$ 时的差距；关于 $\eta$ 的斜率 | $-3.66\times10^{-9}$；1.99994 | $-4.83\times10^{-8}$；2.0000017 |
| 值域对，$\eta=10^{-3}$：一阶项 vs 差距 | $6.06362\times10^{-7}$ vs $6.06362\times10^{-7}$ | $1.86514\times10^{-7}$ vs $1.86509\times10^{-7}$ |
| 符号对，冷启动 Adam：$\epsilon=10^{-8}$ 时的更新差 | $7.1\times10^{-8}$ | $7.8\times10^{-8}$ |

**说明了什么.**

- 在 LoRA 初始化处，第一次更新无法利用反馈中秩 $r$ sketch $\Gamma A_0^\top$ 以外的任何部分。零空间中的差别是二阶的；值域中的差别是一阶的，系数与预测一致。
- 在冷启动 Adam 下，当 $\epsilon\to0$ 时只有 sketch 的符号分布起作用。$\epsilon=10^{-8}$ 时两次更新至多相差 $7.8\times10^{-8}$，这就是 $h_\epsilon$ 的有限 $\epsilon$ 修正。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E10,E11 --output theory/v015_checks/quick/n19.json
```

## 产物

- `theory/v015_checks/results_v015_independent.json`：`groups[id=E11].data`（`clipped_mean`、`always_active`、`warm_Adam`、`Lion`、`LoRA`）与 `groups[id=E10].data.C23_prediction`。
- 代码：`theory/v015_checks/otlr.py`（`check_E11`）。
- 套件图：`theory/v015_checks/figs/E11.png` 与 `.svg`。
- 图：`../figures/N19_learner_statistics.pdf` 与 `.png`（规格 `../figures/specs/N19.md`）。
