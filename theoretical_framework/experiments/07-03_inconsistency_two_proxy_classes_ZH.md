# Proposition 7.3 · 两类 proxy 的不一致性

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/07-03_inconsistency_two_proxy_classes_ZH.md) · [证明](../proofs/07-03_inconsistency_two_proxy_classes_ZH.md) · [实验](07-03_inconsistency_two_proxy_classes_ZH.md) · [English](07-03_inconsistency_two_proxy_classes.md)

**本页检验什么.**

- (b)，在命题自身设定下构造的消息对上：全部 24 组设定**通过**。只看参数的分数把较大的消息排在前面，余弦分数打平，精确一阶价值与有限 response 都偏好较小的消息（[Check 7.4](#c7-4)）。
- 两部分，描述性地，在命题假设不成立的受训受控网络上：在 1,687 个候选对上，每个余弦与只看参数的排序器判错率都在 14.7% 到 28.8% 之间，有限配对替换分数为 2.8%（[Check 7.5](07-05_where_first_order_rankers_fail_ZH.md#c7-5) 的摘要；完整报告在 [Remark 7.5](07-05_where_first_order_rankers_fail_ZH.md#c7-5) 的页面上）。
- (a) 是一个两行的恒等式（$`F\mapsto1-F`$），不单独检验。

**本页用到的设定与记号.**

- warm、不做 bias correction 的 Adam 的单个坐标（$`\epsilon=0`$），尾段梯度独立同分布，均值 $`\bar g`$、平方的均值 $`\bar v`$，得分关于参数仿射；$`\rho=\lvert\bar g\rvert/\sqrt{\bar v}`$ 是信噪比。
- 幅度为 $`\lambda`$ 的消息是沿 $`\bar g`$ 方向的 $`g=\lambda\sqrt{v}`$；$`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$ 是 [Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2) 二次律的最优幅度，$`\lambda_{\rm PO}=(1-\beta_1)\beta_2/(\beta_1\rho(1-\beta_2))`$ 是只看参数的分数关于 $`\lambda`$ 递增的上界。
- *只看参数的分数*：限制在参数上的 continuation 梯度与消息自己的 Adam 步的配对。*余弦分数*：消息方向的函数（LESS 式，[Xia et al., 2024](#ref-xia2024less)）。*精确一阶价值* $`\mathrm{Val}`$：[Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1) 的全部三个通道。*有限 response*：用 common random numbers（[Glasserman, 2004](#ref-glasserman2004monte)）测得的、相对零梯度消息的得分变化。
- 排序器在一对候选上*判错*，指它的分数差与 gold 差符号相反。

**被检验的陈述（[Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)）.** (a) 只依赖于状态与反馈律、不依赖于任务的统计量，会在某个任务上把某一对排错或分不开。(b) 在 Corollary 7.2 的设定下取 $`\alpha>0`$、$`0<\lambda_1<\lambda_2`$：只看参数的分数 $`\alpha\eta\,(\beta_1\rho+(1-\beta_1)\lambda)/\sqrt{\beta_2+(1-\beta_2)\lambda^2}`$ 在 $`\lambda<\lambda_{\rm PO}`$ 时递增，所以偏好 $`\lambda_2`$；任何余弦分数都给两者打平；只要 $`\Delta F(\lambda_2)<\Delta F(\lambda_1)`$，真实价值就偏好 $`\lambda_1`$，在 Corollary 7.2(b) 的误差内，$`\lambda_1+\lambda_2>2\lambda^\star`$ 时这一点成立。

<a id="c7-4"></a>

## Check 7.4 · 构造的消息对上，只看参数的分数与余弦分数（Proposition 7.3(b)）

**结论.** 24/24 组设定**通过**。只看参数的分数总把较大的消息排在前面，余弦分数总是打平，精确一阶价值与有限 response 总是偏好较小的消息。

**动机.** [Proposition 7.3(b)](../statements/07-03_inconsistency_two_proxy_classes_ZH.md#s7-3) 预言：对 $`\lambda_1+\lambda_2>2\lambda^\star`$ 的两条同向消息，只看参数的分数会排反，余弦分数分不开。

**设置.**

- 幅度定律检验（[Check 7.2](07-02_stationary_amplitude_law_ZH.md#c7-2)）的 24 组核心设定：精确的一维 Adam，$`\beta_1=0.9`$，$`\beta_2\in\lbrace0.99,0.999\rbrace`$，$`\rho\in\lbrace0.3,0.7,0.95\rbrace`$，$`T\in\lbrace50,200,1000,5000\rbrace`$，尾段梯度独立同分布于 $`\mathcal N(\rho,1-\rho^2)`$，4,096 个用 common random numbers 的副本。
- 每组设定取 $`\lambda_1=1.25\lambda^\star`$、$`\lambda_2=1.75\lambda^\star`$，于是 $`\lambda_1+\lambda_2=3\lambda^\star>2\lambda^\star`$。幅度范围为 1.3–116.7，warmth $`(1-\beta_2)\lambda^2`$ 为 0.002–13.6。
- 记录的量：只看参数的分数及其在两个端点处导数的符号；余弦分数；精确的 $`\mathrm{Val}`$；有限 rollout response。每个量都给出 4,096 个副本上的 95% CRN 区间。
- 断言：
  - 两个端点处都有 $`(1-\beta_1)\beta_2-\beta_1\rho(1-\beta_2)\lambda>0`$，即两个幅度都低于 $`\lambda_{\rm PO}`$；
  - 只看参数的分数偏好 $`\lambda_2`$；
  - 两个余弦相等；
  - $`\mathrm{Val}`$ 与 rollout 都偏好 $`\lambda_1`$。

**结果.**

| 断言 | 实例数 | 最小 margin | 状态 |
|---|---:|---:|---|
| 两个幅度都低于 $`\lambda_{\rm PO}`$ | 48 | — | 通过 |
| 只看参数的分数偏好 $`\lambda_2`$ | 24 | 分数差最小 0.044 | 通过 |
| 余弦打平 | 24 | 两个余弦都是 1 | 通过 |
| 精确 $`\mathrm{Val}`$ 偏好 $`\lambda_1`$ | 24 | 差最小 0.263（区间下端 0.263） | 通过 |
| 有限 response 偏好 $`\lambda_1`$ | 24 | 差最小 0.262（区间下端 0.261） | 通过 |

**为什么支持该陈述.** 每组设定都以很大的 margin 出现反转，包括 warmth 远超二次律适用范围的设定。所以这种不一致性不是证明里小增量展开的副产品。

**局限.** 单个坐标、外生梯度，即命题本身的设定；在网络上的检验见下面的摘要。

### [Check 7.5](07-05_where_first_order_rankers_fail_ZH.md#c7-5) · 受控网络上各排序器按二阶矩占比的表现

- **对本命题检验什么.** 在带真实 optimizer state 的受训网络上、在 (b) 的单坐标设定之外，只看参数的分数与余弦分数能否可靠地给反馈律排序。
- **设置.** 一个小型受控网络（one-hot 上下文、4 个 tanh 单元、sigmoid 输出、float64）的冻结面板：四个 complete state、八个 Adam/SGD 配方、九个双标签包上的候选律，共 1,687 个 gold 非零的候选对；gold 是精确的八次更新后结果之差。余弦是 LESS 风格的 proxy，不是官方 LESS 实现。
- **关键数字.** 在全部 1,687 对上的判错率：LESS 风格余弦 28.8%（345 个平局）、面向更新的余弦 15.6%（345 个平局）、用精确 continuation 梯度的只看参数的分数 14.7%（120 个平局）；有限配对替换分数 2.8%。只看参数的分数的错误集中在二阶矩通道重要的地方（占比低于 0.5 时 14.4%，高于时 27.3%）；余弦的错误不是（低于时 31.6%，高于时 24.2%）。
- **结论.** 描述性；与命题一致。该面板没有孤立出 (b) 的幅度机制；[Check 7.4](#c7-4) 做到了。

## 复现

在仓库根目录下运行（仅 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C4,C5,C5b
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

- 随机种子：`numpy.SeedSequence([160160, group, subcase, ...])`，PCG64，float64，单线程 BLAS。

这些检验使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C4.ranking`（[Check 7.4](#c7-4)）、`v016:C5` 与 `v016:C5b`（[Check 7.5](07-05_where_first_order_rankers_fail_ZH.md#c7-5)）。
- `theory/v016_checks/results_v016_raw.json.gz`：原始数组，通过 JSON pointer 引用。未提交到 git；SHA-256 为 `1777069872ea80c09dfd5260d610871b68e41a10721eba2826dcd4d647874f69`。
- `theory/v016_checks/SPEC.md`、`check_v016.py`、`c5c6.py`、`c5c6b.py` 及其测试。
