# Remark 7.5 · 一阶排序器在哪里失效

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/07-05_where_first_order_rankers_fail_ZH.md) · [实验](07-05_where_first_order_rankers_fail_ZH.md) · [English](07-05_where_first_order_rankers_fail.md)

**本页检验什么.** 在带真实 optimizer state 的受训受控网络上，一阶排序器是否在二阶矩通道占主导的地方最不可靠（描述性；断言全部通过）。

- 在 1,687 个 provider 对上，每个一阶或余弦排序器的判错率都在 11.4%–28.8%。
- 完整状态一阶分数用精确 continuation gradient 和全部三个通道，在二阶矩占比不低于 0.5 处判错 51%–67%（这样的 161 对上为 59.6%）。
- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 的有限配对替换分数总体判错 2.8%，高占比处 4.3%（161 对中错 7 对，完整状态分数错 96 对）。

**本页用到的设定与记号.**

- *候选与 gold.* 在一个由 Adam 或 SGD 训练的小网络的同一 complete state 上作用的 provider（反馈）律对；一对的 *gold* 差是八次更新后结果的精确差（一轮候选，之后走参考尾段）。
- *判错.* （分数差）×（gold 差）< 0 时，排序器在这一对上判错；完全打平单独计数，不算错。
- *通道缩并与二阶矩占比.* 用 [Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1) 的精确 continuation 梯度 $`(\chi^\theta,\chi^m,\chi^v)`$，一对候选增量对比的 $`\theta`$、$`m`$、$`v`$ 通道缩并之和就是它的一阶差；二阶矩占比为 $`\lvert v\text{ 缩并}\rvert/(\lvert\theta\rvert+\lvert m\rvert+\lvert v\rvert\text{ 缩并})`$。
- *排序器.* 一阶分数（只看参数、用精确梯度的完整状态分数、CRN 测得的方向差）、LESS 式余弦分数，以及有限配对替换分数 $`\hat S`$（把一轮分别替换成各候选，走共同的参考尾段，$`r`$ 个 CRN 副本）。

**被检验的陈述（[Remark 7.5](../statements/07-05_where_first_order_rankers_fail_ZH.md#s7-5)）.** 二阶矩占比高意味着增量相对于 $`v`$ 很大，即 [Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)(a)、(c) 的饱和区域，线性化在那里高估了 $`v`$ 通道的伤害。所以任何一阶分数，包括用精确 continuation 梯度的完整状态分数，在那里都不可靠；那里的排序应当用 Theorem 8.8 的有限替换测量或非线性公式。

<a id="c7-5"></a>

## Check 7.5 · 受控网络上各排序器按二阶矩占比的表现（Proposition 7.3、Remark 7.5）

**结论.** 描述性；它的断言全部通过。

- 在 1,687 个 provider 对上，每个一阶或余弦排序器的判错率都在 11.4%–28.8%。
- 完整状态一阶分数用精确 continuation gradient 和全部三个通道。二阶矩占比不低于 0.5 时，它的判错率为 51%–67%。
- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 的有限配对替换分数总体判错 2.8%，高占比处 4.3%（161 对中错 7 对，完整状态分数错 96 对）。

**动机.**

- [Remark 7.5](../statements/07-05_where_first_order_rankers_fail_ZH.md#s7-5) 预言，一阶排序器在二阶矩通道占主导的地方最不可靠，因为那里消息相对于 $`v`$ 很大。
- [Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes_ZH.md#s7-3) 预言，只看参数的分数与余弦分数都不是一致的排序器。
- 本检验在一个训练过、带真实 optimizer state 的网络上检验这两点。那里 [Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2) 的假设不成立。

**设置.**

- **面板.** 冻结的受控网络 provider 面板，来自 provider 选择实验（[Check 9.4](09-03_derivation_design_rules_ZH.md#c9-4)），只读、不做任何改动。
  - 网络为 one-hot context、4 个 tanh 单元、sigmoid 输出、平均交叉熵，float64。
  - complete state 为 `fresh`、`warm2`、`warm5`、`warm9`。8 个配方：Adam 取 $`\eta\in\lbrace0.01,0.1\rbrace`$，SGD 取 $`\eta\in\lbrace0.1,1\rbrace`$，各配 $`\epsilon\in\lbrace10^{-8},10^{-3}\rbrace`$。
  - 两标签包上的 9 个 provider 律，以及冻结的留出划分。
  - 在每个划分与 cell 上取全部 36 个无序 provider 对：共 1,728 对，其中 1,687 对 gold 非零。
- **gold.** 八次更新之后结果的精确差：一次候选边界，加上全部 128 条七次更新的参考历史。
- **排序器**（一对的分数 = 两个 provider 的分数之差）：
  - LESS 式余弦：provider 的平均参数步与任务目标梯度之间的余弦。SGD 的主版本按设定用正的损失梯度，更新方向版把它翻转到更新方向。这是 LESS 式的 proxy，不是 LESS 官方实现。
  - 只看参数的一阶分数：把精确 continuation gradient 限制在 $`\theta`$ 上，与平均 $`\Delta\theta`$ 缩并。
  - 完整状态一阶分数：$`\theta`$、$`m`$、$`v`$ 三项与精确 continuation gradient 的缩并，即一阶 audit 的 oracle 分数（[Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9)）。
  - CRN 测得的一阶分数，不用梯度 oracle：沿平均增量方向的 common random numbers 方向差分，用与任务一致的 surrogate 或短尾段 surrogate。
  - [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 的有限配对替换分数 $`\hat S`$，与任务一致的 surrogate，$`r=256`$ 个副本。
- **二阶矩占比.** 一对候选的增量之差在 $`v`$ 通道上的缩并的绝对值，除以 $`\theta`$、$`m`$、$`v`$ 三项缩并的绝对值之和，用精确 continuation gradient 计算。
  - 分组固定为 $`[0,.25),[.25,.5),[.5,.75),[.75,1]`$。
  - 非光滑的 threshold 任务上不可用。
  - SGD 的候选对占比为 0。
- **约定.** 分数 × gold < 0 记为判错。完全打平单独计数，不算错。
- **断言.**
  - 完整状态 oracle 分数与 margin 行复现了一阶 margin audit（[Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9)）的冻结记录：396 行中 56 行在 margin 内，且这些行上没有符号错误。
  - 有限替换分数与 `v016:C6b` 的测量完全一致。
  - 冻结文件未被改动。

**结果.** 判错率（判错数 / gold 非零的对数；有打平时在括号中注明）：

| 排序器 | 全部，1,687 | 占比 $`[0,.25)`$，1,318 | $`[.25,.5)`$，84 | $`[.5,.75)`$，76 | $`[.75,1]`$，85 | 占比 $`\ge.5`$，161 | threshold 任务，124 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 余弦（LESS 式） | 28.8%（345 平） | 33.1% | 8.3% | 19.7% | 28.2% | 24.2% | 3.2% |
| 余弦，更新方向 | 15.6%（345 平） | 16.3% | 8.3% | 19.7% | 28.2% | 24.2% | 1.6% |
| 只看参数，精确梯度 | 14.7%（120 平） | 13.8% | 23.8% | 15.8% | 37.6% | 27.3% | 1.6% |
| 完整状态，精确梯度 | 16.4%（118 平） | 13.1% | 7.1% | **51.3%** | **67.1%** | **59.6%** | 1.6% |
| CRN 一阶，任务一致 surrogate | 13.2%（323 平） | 7.8% | 31.0% | 47.4% | 67.1% | 57.8% | 0.0% |
| CRN 一阶，短尾段 surrogate | 11.4%（323 平） | 6.4% | 26.2% | 38.2% | 65.9% | 52.8% | 0.0% |
| 有限配对替换，$`r=256`$ | **2.8%**（0 平） | 2.9% | 2.4% | 7.9% | 1.2% | **4.3%** | 0.8% |

- **高占比.** 占比不低于 0.5 时，有限分数在 161 对中错 7 对。完整状态分数错 96 对，两种 CRN 一阶分数分别错 93 对和 85 对，只看参数的分数错 44 对。有限分数在那里优于每个一阶排序器，但没有消除全部错误。
- **错误集中在哪里.** 只看参数的分数在占比 0.5 以下判错 14.4%，以上判错 27.3%，错误集中在高占比处。主版本余弦在 0.5 以下判错 31.6%，以上 24.2%，错误不集中。
- **按配方**（判错率）：

| 配方 | 对数 | 余弦 | 只看参数 | 完整状态 | CRN 一阶（一致） | 有限替换 |
|---|---:|---:|---:|---:|---:|---:|
| Adam，$`\eta=0.01`$，$`\epsilon=10^{-8}`$ | 395 | 5.6% | 4.1% | 11.4% | 13.9% | 2.3% |
| Adam，$`\eta=0.01`$，$`\epsilon=10^{-3}`$ | 144 | 19.4% | 9.0% | 18.8% | 20.8% | 2.8% |
| Adam，$`\eta=0.1`$，$`\epsilon=10^{-8}`$ | 144 | 13.9% | 27.1% | 13.2% | 29.2% | 2.8% |
| Adam，$`\eta=0.1`$，$`\epsilon=10^{-3}`$ | 144 | 16.0% | 9.0% | 12.5% | 13.2% | 0.7% |
| SGD，$`\eta=0.1`$，$`\epsilon=10^{-8}`$ | 428 | 44.9% | 12.9% | 12.9% | 2.1% | 1.6% |
| SGD，$`\eta=0.1`$，$`\epsilon=10^{-3}`$ | 144 | 49.3% | 18.1% | 18.1% | 6.2% | 2.1% |
| SGD，$`\eta=1`$，$`\epsilon=10^{-8}`$ | 144 | 45.1% | 29.9% | 29.9% | 21.5% | 7.6% |
| SGD，$`\eta=1`$，$`\epsilon=10^{-3}`$ | 144 | 45.1% | 29.9% | 29.9% | 18.8% | 6.2% |

- **按留出划分**，有限替换对比完整状态分数：
  - 留出状态：0.0% 对 9.7%，288 对；
  - 留出任务：1.0% 对 0.0%，103 对，这些对上所有一阶与余弦分数都打平；
  - 留出 $`\epsilon`$：2.4% 对 18.4%，288 对；
  - 留出学习率：5.2% 对 21.5%，288 对；
  - 同时留出 $`\epsilon`$ 与学习率：3.5% 对 21.2%，288 对；
  - fresh 到 warm：0.0% 对 9.7%，288 对；
  - warm 到 fresh：10.4% 对 30.6%，144 对。
- **复现.** 完整状态分数与冻结记录相差 $`2.8\times10^{-17}`$（容差 $`2\times10^{-6}`$）。重算与冻结的 margin 行都是 56/396，这些行上都没有符号错误。
- **成本.** 精确梯度类分数用了 1,044,480 次 oracle 尾段评估，另有 184,320 次用于独立的方向核对。有限分数复用 `v016:C6b` 的测量，不增加尾段成本；它的成本与 [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 的检验一起报告。

![Figure 7.2：各排序器按二阶矩占比的判错率](../figures/fig7_2_rankers.png)

*Figure 7.2.* 受控 provider 面板上各排序器的判错率：总体，以及按候选对的二阶矩占比分组。每个一阶排序器都在二阶矩通道占主导处错得最多，包括用精确 continuation gradient 的完整状态分数。有限配对替换分数在每一组都保持很低。[PDF](../figures/fig7_2_rankers.pdf)

**为什么支持该陈述.**

- 完整状态一阶分数在一阶意义下是精确的，却在多数高占比候选对上判错。这正是 Remark 7.5 预言的失效：在 $`v`$ 通道重要的地方，一阶形式判断错了它。
- CRN 测得的一阶分数在同样的分组上失效。所以失效来自线性化，而不是梯度估计。
- 改测有限替换后，大部分这类错误消失了。这正是 Remark 7.5 给出的补救。
- 余弦与只看参数的分数在各种配方下都不可靠，与 [Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes_ZH.md#s7-3) 一致；不过面板并没有单独分离出 Proposition 7.3(b) 的幅度机制（[Proposition 7.3 实验页](07-03_inconsistency_two_proxy_classes_ZH.md#c7-4) 上构造消息对的 [Check 7.4](07-03_inconsistency_two_proxy_classes_ZH.md#c7-4) 分离了）。

**局限.**

- 只有一个小的受控网络和一个有限面板。
- 判错率是描述性的，不是对独立任务的估计，而且同一个 cell 会出现在多个划分中。
- LESS 式余弦是 LESS 的 proxy，不是对其实现的复现。
- 不对真实模型下任何结论。

## 复现

在仓库根目录下运行（仅 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C5,C5b
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

- 这两组在 rai 上分别约 22 秒（`v016:C5`）和 3 秒（`v016:C5b`）。
- `v016:C5` 与 `v016:C5b` 以只读方式导入冻结的受控网络面板，运行前后都对 pin 住的文件做哈希核对。
- 随机种子：`numpy.SeedSequence([160160, group, subcase, ...])`，PCG64，float64，单线程 BLAS。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C5`、`v016:C5b`，含排序器统计表、占比分组与复现记录。
- `theory/v016_checks/results_v016_raw.json.gz`：原始数组，通过 JSON pointer 引用。未提交到 git；SHA-256 为 `1777069872ea80c09dfd5260d610871b68e41a10721eba2826dcd4d647874f69`。
- `theory/v016_checks/SPEC.md`（`v016:C5`/`v016:C6` 与 `v016:C5b` 各节）、`c5c6.py`、`c5c6b.py` 及其测试。
- `figures/fig7_2_rankers.{png,pdf}` 及其在 `figures/specs/` 中的说明。
