# Proposition 8.4 · 在实际状态上测量平均增量

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-04_measuring_mean_increment_actual_states_ZH.md) · [证明](../proofs/08-04_measuring_mean_increment_actual_states_ZH.md) · [实验](../experiments/08-04_measuring_mean_increment_actual_states_ZH.md) · [English](08-04_measuring_mean_increment_actual_states.md)

**本页检验什么.**

- **[Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4)（测量平均增量）在全部 144 个单元格、直到 $`p=10^5`$ 都成立.** 实测偏差分位数至多为界的 0.615，并且在增量界 $`b`$ 固定时不随 $`p`$ 增长，与 [Remark 8.5](../statements/08-05_norms_constants_ZH.md#s8-5) 一致。

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

沿用 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 的对象：consumed gradient $`G_t(z,Y,\xi)`$，增量 $`\Delta_t(z,g)`$，平均增量 $`\mu^P_t(z)`$，增量二阶矩 $`m^P_{2,t}(z)=\mathbb E_P\lVert\Delta_t\rVert^2`$，以及 continuation $`V^P_{t+1}`$。范数取状态空间的 Hilbert 范数（有限维时为 Euclidean 范数）。

- **增量有界 (B).** 存在 $`b<\infty`$，使在被 audit 的状态上、对被比较的律，几乎必然 $`\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert\le b`$。例如：参数块上的裁剪 SGD，$`b=\eta c`$；$`p`$ 个参数上的 sign 类更新，$`b=\eta\sqrt p`$；梯度以 $`\bar G`$ 为界的 SGD，$`b=\eta\bar G`$。
- **曲率 (S).** 在被 audit 的状态上，$`V:=V^P_{t+1}`$ 沿增量到达的线段 $`\lbrace z+s\Delta:s\in[0,1]\rbrace`$ 二阶可微，且 $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$；记 $`\Lambda=\lVert\nabla V(z)\rVert`$。
- **audit 样本.** 在被 audit 的状态 $`z`$ 上，独立抽取 $`n_P`$ 条消息 $`Y\sim P_t(\cdot\mid z)`$ 与 $`n_Q`$ 条消息 $`Y\sim Q_t(\cdot\mid z)`$，每条配独立的更新随机性，计算增量 $`X=\Delta_t(z,G_t(z,Y,\xi))`$。经验均值为 $`\bar X_P,\bar X_Q`$，经验二阶矩为 $`\hat m^P_2,\hat m^Q_2`$。这需要在 $`z`$ 处做 $`n_P+n_Q`$ 次 consumed gradient 计算，不需要训练尾段。

**被检验的陈述（[Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4)）.**

**(a) Hilbert 空间中的集中性.** 设 $`X_1,\dots,X_n`$ 是实 Hilbert 空间中的独立同分布随机元，几乎必然 $`\lVert X\rVert\le b`$，均值为 $`\mu`$。则以至少 $`1-\delta`$ 的概率，

```math
\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big).
```

对大小为 $`n_P,n_Q`$ 的两个独立样本，以至少 $`1-\delta`$ 的概率，

```math
\big\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\big\rVert\le b\Big(1+\sqrt{2\log(2/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big).
```

**(b) 在一个实际状态上的一阶 audit.** 在产生被 audit 状态 $`z`$ 的历史条件下固定 $`z`$，并设 (B) 与 (S) 在该处成立。则在 audit 样本上以至少 $`1-\delta`$ 的概率，

```math
\big\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\big\rvert\le\Lambda\big(\lVert\bar X_P-\bar X_Q\rVert+e_1\big)+\tfrac{M_2}{2}\big(\hat m^P_2+\hat m^Q_2+e_2\big),
```

其中

```math
e_1=b\Big(1+\sqrt{2\log(4/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big),\qquad e_2=b^2\Big(\sqrt{\tfrac{\log(4/\delta)}{2n_P}}+\sqrt{\tfrac{\log(4/\delta)}{2n_Q}}\Big).
```

**(c) 沿实际训练的整次 audit.** 独立运行协议 $`Q`$ 共 $`J`$ 次。在每次运行的每一轮 $`t`$，在访问到的状态上以置信度 $`\delta_{\rm a}/(JR)`$ 应用 (b)，把 (b) 的右端截断到 $`1`$，记为 $`\hat\varepsilon_{j,t}`$。设 (B) 与 (S) 在每个访问到的状态上以所述常数成立，且抽取 audit 样本不改变训练过程。则以至少 $`1-\delta_{\rm a}-\delta_{\rm mc}`$ 的概率，

```math
\lvert L(P)-L(Q)\rvert\le\frac1J\sum_{j=1}^J\sum_{t=0}^{R-1}\hat\varepsilon_{j,t}+R\sqrt{\frac{\log(1/\delta_{\rm mc})}{2J}}.
```

**测了什么，没测什么.**

- 依赖数据的部分是 $`\bar X_P-\bar X_Q`$ 与 $`\hat m_2`$，它们只需要在访问到的状态上计算 consumed gradient。
- $`\Lambda`$ 与 $`M_2`$ 是 continuation value 的性质，与消息无关，样本不估计它们。它们必须解析地给界，或者用完整尾段的 audit 校准（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)），并把校准误差加进界里。

<a id="c8-13"></a>

## Check 8.13 · 平均增量的集中性（Proposition 8.4(a)、Remark 8.5）

**结论.** 在全部 144 个单元格中（每格 20,000 次试验，共 2,880,000 个实例），失败率都远低于 $`\delta`$，实测偏差分位数从不超过界的 0.615。增量界 $`b`$ 固定时，这个比值从 $`p=1`$ 到 $`p=100{,}000`$ 都不增长。

**动机.** [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4)(a) 说，经验平均增量在 Hilbert 范数下以 $`b(1+\sqrt{2\log(1/\delta)})/\sqrt n`$ 的速率集中，界中不显含维数。LoRA adapter 有上千万个参数。若存在隐含的维数依赖，会表现为失败率高于 $`\delta`$，或实测分位数随 $`p`$ 增长而逼近界。

**设置.**

- 维数 $`p\in\lbrace1,10,1000,100000\rbrace`$，样本量 $`n\in\lbrace8,32,128\rbrace`$，置信度 $`\delta\in\lbrace0.05,0.2\rbrace`$，单样本与双样本统计量。
- 三个 $`b=1`$ 的有界分布（`data.laws`）：
  - *sign*：一个有偏的标量符号乘一个固定的单位符号向量，即 $`\eta=1/\sqrt p`$ 的归一化符号型增量，在 $`P`$ 与 $`Q`$ 下均值不同；
  - *clipped Gaussian*：$`N(0,I/p)`$ 在范数 $`1`$ 处截断；
  - *sparse spikes*：均匀选一个坐标、取等概率的符号，范数为 $`1`$。
- 高维合成量利用各向同性与 Beta 角递推精确抽样，不需要构造 $`p`$ 维向量（`data.implicit_sampling`）。
- 每格 20,000 次独立试验；种子 `SeedSequence([150150, 9])`。
- 每格记录偏差超过界的试验比例（断言 ≤ $`\delta`$ 加四个二项标准误），以及 偏差/界 的经验 $`(1-\delta)`$ 分位数。

**结果**（`theory/v015_checks/results_v015_independent.json`，组 `v015:V9`，`data.rows`）：

| $`p`$ | 单元格数 | 最坏失败率，$`\delta=0.05`$ | 最坏失败率，$`\delta=0.2`$ | 最低覆盖率 | 最大 分位数/界 |
|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 0.00095 | 0.00765 | 0.99235 | 0.615 |
| 10 | 36 | 0.00030 | 0.00315 | 0.99685 | 0.574 |
| 1,000 | 36 | 0.00050 | 0.00355 | 0.99645 | 0.574 |
| 100,000 | 36 | 0.00040 | 0.00335 | 0.99665 | 0.574 |

按分布看，最大 分位数/界 为 0.574（sign）、0.408（clipped Gaussian）、0.615（sparse spikes）；总体最大值 `data.max_quantile_over_bound` = 0.615。单样本 clipped Gaussian 在 $`\delta=0.05`$ 时，这个比值随 $`p`$ 下降，在每个 $`n`$ 下都从 0.406 降到 0.291。

**为什么支持该陈述.**

- 界在每个单元格都成立，且至少保守 1.6 倍；在 $`b`$ 固定时，没有任何量随维数变差。
- 如 [Remark 8.5](../statements/08-05_norms_constants_ZH.md#s8-5) 所说，$`b`$ 本身可以随维数增长：对未归一化的符号型增量，$`b=\eta\sqrt p`$。这时一阶差异也是同一尺度，所以相对精度仍由 $`1/\sqrt n`$ 决定（`data.caveat`）。
- 这一检验只覆盖平均增量的估计。排序还需要延续值梯度与余项（[Check 8.15](../experiments/08-09_first_order_version_margin_criterion_ZH.md#c8-15)–[8.16](../experiments/08-09_first_order_version_margin_criterion_ZH.md#c8-16)）。

集中性结果是 [Figure 8.4](../experiments/08-09_first_order_version_margin_criterion_ZH.md#c8-15) 的 A 幅。

![Figure 8.4：一阶测量与一阶 margin](../figures/fig8_4_first_order_audit.png)

*Figure 8.4.* 左：从 1 到 100,000 的每个维数上，平均增量的经验偏差都低于界的 0.62（[Check 8.13](../experiments/08-04_measuring_mean_increment_actual_states_ZH.md#c8-13)）。中：在冻结的 held-out 面板上，用 **oracle** 延续值梯度计算的一阶分数，在 margin 超过余项界的四分之一后不再判错，而静态 proxy 在大 margin 处仍有 42% 判错（[Check 8.15](../experiments/08-09_first_order_version_margin_criterion_ZH.md#c8-15)）。右：margin 内由一阶决定；margin 外只有实测尾段可靠。受控小网络。[PDF](../figures/fig8_4_first_order_audit.pdf)

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V9 --output theory/v015_checks/quick/s8.json
```

## 产出文件

- `theory/v015_checks/results_v015_independent.json`：组 `v015:V9`；规范 `theory/v015_checks/SPEC.md`。
- 图：`../figures/fig8_4_first_order_audit`（面板 A）。
