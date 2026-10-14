# Proposition 8.14 · 锚点复用与配对漂移证书

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-14_anchor_reuse_drift_certificate_ZH.md) · [证明](../proofs/08-14_anchor_reuse_drift_certificate_ZH.md) · [实验](08-14_anchor_reuse_drift_certificate_ZH.md) · [English](08-14_anchor_reuse_drift_certificate.md)

**本页检验什么.**

- **合成实例上的 (a)、(b)、(d).** 漂移后 checkpoint 上的复用残差为 3.375，低于界 3.477；算子漂移的平方为 0.094，低于配对二阶矩 0.330；去偏的配对乘积无偏，误差 $`1.1\times10^{-16}`$（**PASS**）。
- **实测真实模型噪声下的 (c).** 真实模型的噪声参照（一个 checkpoint、300 个 dev 任务、两台配置相同的服务器）给出 241 和 242 个成功，15 个任务翻转。在这个噪声下，严格的漂移半径在 10 对时为 0.718，40 对 0.507，100 对 0.404，1,000 对 0.227；要降到 0.03 约需 $`4.9\times10^6`$ 对。它要界的残差平凡地至多为 0.5。这个证书在该量级下**成立但为空**。

**本页用到的设定与记号.**

- 参考测度 $`\rho`$，$`\mathcal H=L^2_0(\rho)`$；$`p`$ 个测试，权重 $`\omega`$；锚点 $`z_0`$ 处的 response operator $`\mathcal R_0`$ 与新 checkpoint $`z`$ 处的 $`\mathcal R_z`$，response 向量 $`g_0(y),g_z(y)\in[0,1]^p`$。
- $`\hat{\mathcal R}_0`$ 是锚点估计，误差 $`\epsilon_0`$，奇异值 $`\hat\sigma_k(z_0)`$，$`\hat\Pi_r`$ 投影到其前 $`r`$ 个右奇异向量上。
- 配对实验：随机消息 $`y_i\sim\rho`$，条件均值正确的有界输出 $`X^{(i)}_z,X^{(i)}_0`$，$`D^{(i)}=X^{(i)}_z-X^{(i)}_0`$，$`W_i=\lVert D^{(i)}\rVert_\omega^2`$；$`M`$ 个冻结的 checkpoint 对，每对 $`n`$ 条消息。

**被检验的陈述（[Proposition 8.14](../statements/08-14_anchor_reuse_drift_certificate_ZH.md#s8-14)）.** 若 $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\Delta_z`$，

```math
\lVert\mathcal R_z(I-\hat\Pi_r)\rVert_{\rm op}\le\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z ;
```

漂移满足 $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}^2\le\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2\le\mathbb EW_i`$；以至少 $`1-\delta`$ 的概率，对全部 $`M`$ 对，$`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\hat\Delta_z=\sqrt{\min\lbrace1,\bar W_z+\sqrt{\log(M/\delta)/(2n)}\rbrace}`$；两个条件独立的配对差之乘积是 $`\lVert g_z(y)-g_0(y)\rVert_\omega^2`$ 的无偏估计。

| Check | 检验内容 | Code ID | 结果 |
|---|---|---|---|
| [Check 8.20](#c8-20) | (a)、(b)、(d) | `v017:C19.anchor` | PASS |
| [Check 8.21](#c8-21) | 实测噪声下的 (c)、(d) | `v017:C20.scale`、`v017:R5.noise` | PASS；成立但为空 |

<a id="c8-20"></a>

## Check 8.20 · 复用残差、配对漂移界与去偏乘积（Proposition 8.14(a)、(b)、(d)）

**结论.**

- 锚点的秩 3 接口在漂移后 checkpoint 上的残差 3.375，低于 $`\hat\sigma_4(z_0)+\epsilon_0+\Delta_z=3.477`$。
- 算子漂移的平方 0.094，低于配对二阶矩 0.330。
- 去偏乘积 $`\langle D^{(1)},D^{(2)}\rangle_\omega`$ 的条件均值等于 $`\lVert g_z-g_0\rVert_\omega^2`$，误差 $`1.1\times10^{-16}`$。

三项均为 **PASS**。

**动机.** (a) 是带完整算子漂移的复用不等式；(b) 用可测的配对输出二阶矩代替未知的算子差；(d) 是从这个二阶矩中去掉过程噪声的去偏步骤。每一条都是精确的不等式或恒等式，所以用一个漂移非零、噪声非零、可以精确计算的实例就能直接检验。

**设置.**

- *复用.* 锚点 $`A`$ 为元素独立标准正态的 $`7\times10`$ 矩阵，精确已知（$`\epsilon_0=0`$）；新 checkpoint 为 $`A+E`$，$`E`$ 的元素为 $`0.02`$ 倍标准正态；$`r=3`$；$`\Delta_z=\lVert E\rVert_{\rm op}`$。
- *漂移.* 6 个测试、8 条消息；response 值 $`g_0,g_z`$ 在 $`[0,1]`$ 上均匀；$`\rho`$ 与 $`\omega`$ 为参数全为 1 的 Dirichlet 分布。输出为 Bernoulli，同一对内共用一个均匀变量，$`X_z=\mathbf 1\lbrace U\le g_z\rbrace`$，$`X_0=\mathbf 1\lbrace U\le g_0\rbrace`$，因此边缘正确，且 $`\mathbb E(X_z-X_0)^2=\lvert g_z-g_0\rvert`$。
- *去偏乘积.* 每条消息两对相互独立的配对（独立耦合）；对每个测试与消息，通过对每个差的三个取值做精确枚举，算出乘积的条件均值。
- 随机种子 `SeedSequence([20261014, tag])`；NumPy（[Harris et al., 2020](#ref-harris2020array)）。

**结果**（`theory/v017_checks/results_v017.json`，条目 `v017:C19.anchor`）：

| 量 | 值 | 界或目标 |
|---|---:|---:|
| $`\lVert(A+E)(I-\hat\Pi_3)\rVert_{\rm op}`$ | 3.3746 | 3.4765 |
| $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}^2`$ | 0.0942 | 0.3296（配对二阶矩） |
| 去偏乘积，条件均值的最大误差 | $`1.1\times10^{-16}`$ | 0 |

**为什么支持该陈述.**

- 复用残差满足三角不等式给出的界；锚点的奇异向量并没有被假设为稳定。
- 配对二阶矩界住了漂移的平方，正如 (b) 所说；这里它明显更大，因为二阶矩中含有耦合噪声 $`\lvert g_z-g_0\rvert-(g_z-g_0)^2`$。
- 两个条件独立的差之乘积，期望恰好是条件均值的平方，(d) 正是靠这一点去掉噪声。

<a id="c8-21"></a>

## Check 8.21 · 实测评估噪声下的漂移证书（Proposition 8.14(c)、(d)）

**结论.**

- 以真实模型上测得的噪声水平，10、40、100、1,000 对时严格半径 $`\hat\Delta_z`$ 分别为 0.718、0.507、0.404、0.227（$`M=10`$，$`\delta=0.05`$）。要让半径降到 0.03 约需 $`4.9\times10^6`$ 对，也就是约 $`10^7`$ 个续训分支。
- 它要界的量 $`\lVert\mathcal R_z(I-\hat\Pi_r)\rVert_{\rm op}`$ 平凡地至多为 $`\lVert\mathcal R_z\rVert_{\rm HS}\le0.5`$，所以约 40 对以下证书比平凡界还差，40 对以上也远远谈不上有用：**成立但为空**。
- 在这个噪声下 $`\hat\Delta_z`$ 的估计量地板为 $`\sqrt{\bar W}\approx0.013`$；(d) 的去偏乘积能去掉它，但半径仍按 $`n^{-1/4}`$ 下降。

**动机.** Proposition 8.14 让跨 checkpoint 的复用可以用随机消息来检查，而不需要完整目录。这个检查有没有信息量，取决于单次配对运行的噪声和基于取值范围的集中项。真实模型的噪声参照给出前者；后者由 $`n`$、$`M`$、$`\delta`$ 决定。

**设置.**

- *真实模型噪声参照*（`v017:R5.noise`）：参考 cold-start 训练的最终 checkpoint，在两台配置相同的推理服务器上，用 300 个 ALFWorld（[Shridhar et al., 2021](#ref-shridhar2021alfworld)）dev 任务评估；逐任务的成功记录存放在 `theory/v017_checks/records/noise/`。
- 如果 15 次翻转都是对称的 $`\pm1/300`$ 变化，单次分数差的标准差为 $`\sqrt{15}/300=0.0129`$，单个测试上 $`\bar W=(\sqrt{15}/300)^2=1.67\times10^{-4}`$。这只是推理服务的噪声：两次训练随机性不同的续训还会带来自己的噪声，所以漂移实验中真实的 $`\bar W`$ 至少这么大。一对服务器、一个 checkpoint 不能说明评估噪声不可约或独立同分布。
- 半径：$`\hat\Delta=\sqrt{\min\lbrace1,\bar W+\sqrt{\log(M/\delta)/(2n)}\rbrace}`$，$`M=10`$，$`\delta=0.05`$（`v017:C20.scale`）。

**结果**（条目 `v017:R5.noise` 与 `v017:C20.scale`）：

| 噪声参照 | 值 |
|---|---:|
| 任务数 | 300 |
| 成功数，服务器 a / 服务器 b | 241 / 242 |
| 任务翻转（仅 a / 仅 b） | 15（7 / 8） |
| 若翻转对称，单次分数差的标准差 | 0.0129 |

| 配对数 $`n`$ | Hoeffding 项 | $`\hat\Delta`$ |
|---:|---:|---:|
| 10 | 0.515 | 0.718 |
| 40 | 0.257 | 0.507 |
| 100 | 0.163 | 0.404 |
| 1,000 | 0.051 | 0.227 |
| $`\approx4.9\times10^6`$ | — | 0.030 |

**为什么支持该陈述.**

- (c) 的证书按所述形式成立；这个检验显示的是它的大小。Hoeffding 项为 $`W\in[0,1]`$ 的整个取值范围付费，不随实测的小 $`\bar W`$ 缩小，所以早在噪声地板起作用之前，半径就由 $`n^{-1/4}`$ 行为主导。
- 这种"为空"是这个估计量和这个集中不等式的性质。它并不证明跨 checkpoint 的漂移无法被便宜地认证；它标出了尚未解决的问题（[Remark 8.17](../statements/08-17_open_problems_ZH.md#s8-17)）：半径随效应大小缩放的证书。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

检验程序使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C19.anchor`、`v017:C20.scale`、`v017:R5.noise`；代码 `theory/v017_checks/check_v017.py`，函数 `c_anchor`、`c_scale`、`r_noise`。
- 真实模型记录：`theory/v017_checks/records/noise/a/per_task.tsv`、`theory/v017_checks/records/noise/b/per_task.tsv`（从噪声参照在集群上的运行目录复制）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-shridhar2021alfworld"></a>

**Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht (2021).** ALFWorld: Aligning text and embodied environments for interactive learning. *The Ninth International Conference on Learning Representations (ICLR)*. arXiv [2010.03768](https://arxiv.org/abs/2010.03768).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
