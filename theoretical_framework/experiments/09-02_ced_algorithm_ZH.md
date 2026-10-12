# Algorithm 9.2 · Consequence-Calibrated Experience Distillation，CED

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/09-02_ced_algorithm_ZH.md) · [证明](../proofs/09-02_ced_algorithm_ZH.md) · [实验](09-02_ced_algorithm_ZH.md) · [English](09-02_ced_algorithm.md)

**本页检验什么.** 整个算法还没有在真实模型上运行过；计划中的真实模型检验 R0–R3 列在 [Remark 9.4](../statements/09-04_scope_of_evidence_ZH.md#s9-4)。目前在一个受控小网络中检验的，是第 1 步与第 5 步所做的选择：按在记录下来的完整状态上实测的"更新加尾段"response 给候选排序，而不按静态统计量排序。

- **实测 response 能选对；测试过的静态 proxy 不能.** 在冻结的留出面板上，测试过的五个静态 proxy 在留出的学习器状态上判错率为 0.359–0.547（随机 0.547），Spearman 至多 0.10。实测的 4 步尾段在每一种状态、$`\epsilon`$、学习率与 warmth 的变化下判错率至多 0.031、regret 为零。在一个留出的任务上，短尾段比随机选得差；只有全尾段在每个划分上都精确（[Check 9.4](09-03_derivation_design_rules_ZH.md#c9-4)）。
- **warmth 必须记录，但 warm-up 救不了静态 proxy.** 学习率为 0 的矩 warm-up 压缩了候选之间的差别，但 10 种静态 proxy 与 warm-up 的组合中，没有一种同时改进 regret、判错率与 Spearman（[Check 9.5](09-03_derivation_design_rules_ZH.md#c9-5)）。
- 这两项检验都声明在 [Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3) 的实验页上，因为它们主要检验那里的设计规则；下面是针对本算法的摘要。

**本页用到的设定与记号.**

- *学习器.* 一个受控网络：one-hot 上下文、4 个 tanh 单元、sigmoid 输出、平均交叉熵、float64。它的完整状态（参数加 Adam 矩 $`m,v`$ 与步数计数器）为 `fresh`、`warm2`、`warm5`、`warm9`；配方为 Adam，$`\eta\in\lbrace0.01,0.1\rbrace`$，以及 SGD，$`\eta\in\lbrace0.1,1\rbrace`$，各配 $`\epsilon\in\lbrace10^{-8},10^{-3}\rbrace`$。
- *候选.* 十个双标签 packet 上的九个标签律，其中 `reference_even` 是参考 $`P^0`$。
- *gold.* 候选的精确学习价值：一次候选边界，之后 7 次参考更新（共 8 次），对全部有限历史精确枚举，再减去参考的同一量。这就是 [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2) 中状态 $`z`$ 上的优势 $`A_t(z,j)`$。
- *proxy.* 静态统计量（输入矩误差、平均梯度、只看符号、第一步的 $`\Delta\theta`$、带 $`(m,v)`$ 的 $`\Delta\theta`$）；1 次与 4 次更新之后的实测短尾段 response（[Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6) 意义下的 surrogate）；对照（随机分数、全尾段）。每个 proxy 只在 pilot 上用 ridge 回归拟合，然后冻结。
- *指标.* 选择 regret（最好候选的 gold 值减去 proxy 选中候选的 gold 值）、判错率（预测对比与 gold 符号相反的候选所占比例）、与 gold 对比的 Spearman 相关。

**被检验的陈述（[Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2)）.** 给定带声明配方的学生、$`K`$ 条可信示范与 $`n`$ 次允许的实验预算、候选 feedback law $`P^1,\dots,P^M`$ 与参考 $`P^0`$，CED 在第 1 步记录每个检查点的完整学习器状态 $`z_t=(\theta_t,o_t)`$ 及其 warmth；在第 2 步编译之前按学习器相对 water-filling 获取查询；在第 3 步把候选编译成 packet 律；在第 4 步记录每个候选的 optimizer 统计量并标记过大的消息；在第 5 步在实际状态上用带半径 $`e^{jk}_{\rm stat}`$ 的 CRN 配对替换分数 $`\hat S^{jk}`$ 测量每个候选的后果，用少量全尾段 audit 校准，一阶分数只在 margin 占主导处作筛选；在第 6 步选择 $`j_t\in\arg\max_j[\hat S^{j0}-e^{j0}_{\rm stat}-\varepsilon_{\rm sur}]`$（没有正下界时保留参考），训练、重新 audit、冻结，并分开报告方法迁移与策略迁移。它的保证是 (G1) 单个检查点上的证书、(G2) 用全尾段 audit 时整次训练的保证 $`L_E({\rm adaptive})-L_E(0)\ge-\delta`$、(G3) 后处理不损失信息，以及 (G4) 其余情形为经验结论。

| 检验 | 对本算法检验什么 | Code ID | 结果 |
|---|---|---|---|
| [Check 9.4](09-03_derivation_design_rules_ZH.md#c9-4) | 第 5 步：实测 response 对比静态 proxy；尾段长度 | `v015:E6` | 受控发现 |
| [Check 9.5](09-03_derivation_design_rules_ZH.md#c9-5) | 第 1 步与第 5 步：warmth 与 proxy 迁移 | `v015:E9` | 受控发现 |

其他页面上的相关检验：一阶排序器与 CRN 分数的对比（[Check 7.5](07-05_where_first_order_rankers_fail_ZH.md#c7-5)），以及配对替换证书本身，都随它们所检验的陈述一起报告（[Remark 7.5](../statements/07-05_where_first_order_rankers_fail_ZH.md#s7-5)、[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)）。

### [Check 9.4](09-03_derivation_design_rules_ZH.md#c9-4) · 用冻结 proxy 做留出选择

**对本算法检验什么.** 第 5 步能否换成更便宜的统计量：如果某个静态 proxy 拟合一次、冻结之后，在留出的学习器状态、任务、配方与 warmth 上选得和实测 response 一样好，第 5 步的测量就没有必要。

**设置.**

- proxy 在 72 行的 pilot 上拟合（状态 `fresh`、`warm2`；任务 `moment`、`parity`；两个配方），并在任何留出测量之前冻结到磁盘。
- 七个留出划分（留出状态、只改 $`\epsilon`$、只改学习率、两者都改、fresh 到 warm、warm 到 fresh、留出任务 `threshold`）；432 个留出行；每个划分只评测一次。

**关键数字**（`v015:E6`）.

| 划分 | 静态 proxy 判错率 | 短尾段 4 判错率 | 随机判错率 | 短尾段 1 / 4 的 regret | 随机 regret | 全尾段 regret |
|---|---:|---:|---:|---:|---:|---:|
| 留出状态 | 0.359–0.547 | 0.016 | 0.547 | 0.0000 / 0.0000 | 0.0056 | 0.0000 |
| fresh 到 warm | 0.359–0.562 | 0.016 | 0.578 | 0.0000 / 0.0000 | 0.0045 | 0.0000 |
| 留出任务 | 0.083–0.250 | 0.042 | 0.458 | 0.0439 / 0.0322 | 0.0186 | 0.0000 |

- 在留出状态上，静态 proxy 的 Spearman 在 −0.00 到 0.10 之间；4 步尾段为 0.99。

**结论.** 支持第 5 步的受控发现。实测 response 能选对，测试过的静态 proxy 不能；但在一个任务族上校准的短 surrogate 尾段，到了另一个任务族可能失去排序，所以第 5 步要用全尾段 audit 校准 surrogate，[Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11) 也把校准迁移当作需要核对的事。这一结论只针对测试过的 proxy，而不是所有静态统计量。

### [Check 9.5](09-03_derivation_design_rules_ZH.md#c9-5) · 学习率为 0 的矩 warm-up 下的 proxy 迁移

**对本算法检验什么.** 能否先把 optimizer 的矩 warm 起来，让静态统计量在 warm 状态上变得有预测力，从而省去记录完整状态及其 warmth（第 1 步）。

**设置.**

- 三个臂 `fresh`、`fresh_W5`、`fresh_W20`：以学习率恰为 0 做 $`W\in\lbrace5,20\rbrace`$ 次 Adam 步，所以只有 $`m`$、$`v`$ 与步数计数器前进。
- 候选、gold、冻结的拟合与指标都与 [Check 9.4](09-03_derivation_design_rules_ZH.md#c9-4) 相同；五个配对的 cold/warm 留出划分，918 行。

**关键数字**（`v015:E9`，5 个划分上 warm − cold 的平均）.

- 随机选择的 regret 在 5 个划分中的 4 个下降，平均在 $`W=5`$ 时降 0.0079，$`W=20`$ 时降 0.0100（阈值任务上在 $`W=20`$ 时从 0.043 降到 0.004）。
- 10 种静态 proxy 与 $`W`$ 的组合中，没有一种同时改进 regret、判错率与 Spearman；它们的判错率上升 1–13 个百分点。

**结论.** 支持第 1 步与第 5 步的受控发现。warmth 改变了由哪个统计量携带信息，而测试过的静态 proxy 在两种状态下都不是那个统计量，所以仍然要测量 response，并且要在记录了 warmth 的状态上测量。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E6,E9 --output theory/v015_checks/quick/s9.json
```

## 产物

- `theory/v015_checks/results_v015_independent.json`：条目 `v015:E6`（`reports`、`negative_results`、`sign_pair_audit`）与条目 `v015:E9`（`summary`、`cheap_proxy_results`、`comparisons`、`freeze_events`）。
- 冻结的契约文件：`theory/v015_checks/proxy_split_freeze.json`、`proxy_models_frozen.json`、`proxy_split_freeze_E9.json`、`proxy_models_frozen_E9.json`。
