# Proposal 1 · 实证发现：按静态标准蒸馏会出什么问题

[English](P1_empirical_findings.md) · [Proposal 1（RCD）](P1_response_calibrated_distillation_ZH.md) · [理论框架](../../theoretical_framework/README_ZH.md)

*状态：2026-10-14。本页所有数字都来自已经跑完的真实模型实验，没有为本页新跑任何计算。图和数字可以用 [`analysis/p1_findings/build_data.py`](../../analysis/p1_findings/build_data.py) 与 [`plot.py`](../../analysis/p1_findings/plot.py) 重新生成，数据快照在 [`figs/p1_findings_data.json`](figs/p1_findings_data.json)。每个 finding 都写明证据强度与局限；标为"中期"或"探索性"的数字不是预注册结论。*

---

## 共同设置

- **环境与评测.** ALFWorld。评测集 dev140 是 valid_seen 的全部 140 个任务；封存的 valid_unseen 从未使用。评测用 batch-invariant 的服务，每个 adapter 先过一道 HF/vLLM 等价性检查，没通过又没有有效重评的格子直接丢掉。
- **噪声参照.** 同一个 adapter 换一台配置相同的 server，结果相差约 ±7 个任务（跨配置参照）。下文凡是"超过噪声"，都是相对这个量说的。
- **学生与 teacher.** 学生是 Gemma-4-12B-it，base 在 dev140 上做对 62 个。teacher 是 gpt-6-luna，做对 114 个。
- **可信示范.** 每个任务族 $`K`$ 条真实执行过、验证成功的 teacher 轨迹（下文简称"示范"），按任务类型分层抽样，每个 $`K`$ 取 3 个 support seed。
- **参照配方（Seed-SFT）.** LoRA r16，AdamW，学习率 1e-5（常数），全局梯度裁剪 1.0。
- **baseline.** 都按原论文的方法设计实现，探索只在这 $`K`$ 个任务上进行。

## 总览

| Finding | 对应的缺陷 | 一句话 | 证据强度 |
|---|---|---|---|
| [A](#finding-a) | 缺陷 1 · 价值 | 同一个"只保留成功轨迹"的规则，在 13 个 support set 上让结果变化 −54 到 +24 | 中强：效应远超噪声；但每组只有一个训练 seed |
| [B](#finding-b) | 缺陷 1 + 3 | 同一批正确示范，训练早期让学生变差、之后才变好；拐点由学生训练到哪里决定 | 强：12 个 run 里 11 个出现；窗口是看过数据后划的 |
| [C](#finding-c) | 缺陷 1（+ optimizer） | 静态"学习信号"大 24 倍的数据，实测后果并不更大；optimizer 先把三者的更新拉平了 | 中：只有一个状态、未校准的 surrogate，P2 只有一半副本 |
| [D](#finding-d) | 缺陷 2 · 信息 | 示范里缺的任务类型，重复训练 40 万 token 也补不上；有 1 条示范，曲线就起来了 | cool 类型上强，look-at 类型上弱 |
| [E](#finding-e) | 缺陷 3 · 时间 | 学生不只是参数：72–100% 的更新来自 optimizer 记忆，两个 batch 的作用不可加，只改 optimizer state 就让结果差 13 个任务 | (a)(b) 强；(c) 单 seed，p=0.015 |

三个缺陷的含义见 [Proposal 1 第 3 节](P1_response_calibrated_distillation_ZH.md#3-三个共同缺陷)。

---

<a id="finding-a"></a>

## Finding A · 同一个静态过滤规则，结果的符号不确定

**一句话.** AgentEvol 的规则是"只保留环境验证成功的自生成轨迹"。在 13 个 support set 上，同一条规则让最终结果变化了 −54 到 +24 个任务；有的组提升，有的组崩溃。

**静态方法怎么想.** 被保留的每条轨迹都验证成功，所以都是"好数据"；多迭代几轮，学生应该越来越好。

**实验设置.**
- 方法：AgentGym 论文的 AgentEvol（[Xi et al., 2024](P1_response_calibrated_distillation_ZH.md#ref-agentgym)）。先在 $`K`$ 条示范上做 BC，然后迭代 3 轮：当前策略在 $`K`$ 个任务上探索，按环境奖励只保留成功的轨迹，合并数据后再训练。
- 配方：按官方做全参数微调。
- 规模：13 个 support set，$`K\in\lbrace1,2,5,10,20,50,100\rbrace`$ 取若干 seed，每组一个训练 seed。
- 评测：每一轮的模型都在 dev140 上评测。上下文溢出修复（10-06）之后补跑的直接评测优先使用。

**结果.**

![Finding A](figs/p1_finding_A_agentevol.png)

- 第 3 轮减 BC：5 组下降超过噪声（−54、−38、−32、−18、−11），3 组上升超过噪声（+9、+11、+24），5 组在噪声内。
- 两组最后一轮掉到 25 个以下：k10s1 从 64 到 10，k5s2 从 44 到 12。
- 崩溃和提升都不由起点决定：k5s1 从 88 掉到 50；k20s2 从 23 升到 47；k50s0 从 93 升到 102。
- 另外两个按静态规则选反馈的方法，也有同样的现象：
  - LEAP（teacher 在学生状态上给出修正）第 3 轮减第 0 轮：$`K=2`$ 为 +1，$`K=10`$ 为 −6，$`K=50`$ 为 +29，$`K=100`$ 为 +33。
  - ETO 从训练充分的学生（Seed-SFT p40）开始时，探索只找到 3 个"成功对失败"的对比对；DPO 的 loss 约为 1e-13，梯度约为 1e-9（训练日志，10-07）。这些按结果选出的对比对，对这个学生几乎没有学习信号。

**为什么支持缺陷 1.**
- 13 组用的是同一个筛选标准，保留下来的每条轨迹都验证成功。如果价值是数据自身的属性，结果的符号不会翻转；而它翻转了，幅度也远超噪声。
- 所以后果取决于静态标准看不到的东西：学生当时的状态（BC 学到哪里）、support set，以及训练的随机性。这正是 [Proposition 3.1](../../theoretical_framework/statements/03-01_static_value_not_learning_value_ZH.md#s3-1) 说的情况：静态量相同，学习价值可以不同，甚至符号相反。

**局限.**
- 每个 support set 只有一个训练 seed。Kang 的配方在 K20 上只换训练 seed，结果就是 63/74/86（极差 23），所以一部分差异可能来自训练随机性。不过训练随机性本身也是"学习器与配方"的属性，不是数据的属性；而且 3 组的下降（−32 到 −54）超过这个极差的一半很多。
- 不同 support set 的内容也不同，这不是只改一个变量的受控实验。
- AgentGym 的 BC 本身在各组之间差别很大（23 到 102），反映了官方配方的不稳定。

<a id="finding-b"></a>

## Finding B · 同一批正确示范，先害后益

**一句话.** 用同一批正确的 teacher 示范继续训练，在某一段训练里会让学生变得比 base 还差，过了这一段才开始变好。拐点出现在差不多的训练进度上，与示范是 5 条还是 50 条关系不大。

**静态方法怎么想.** 示范都是正确的，所以每一条都有正的价值；训练多少只是一个超参数。

**实验设置.**
- 配方：Seed-SFT（参照配方）。
- 规模：$`K\in\lbrace5,10,20,50,100\rbrace`$，每个 $`K`$ 有 3 个 support seed，共 15 个 run。
- checkpoint：在 3 到 40 遍（passes）之间，以及 10 万、20 万、40 万 token 处保存，每个都在 dev140 上评测。
- 横轴：学生已经训练过的监督 token 数，也就是学习器状态的一个刻度。

**结果.**

![Finding B](figs/p1_finding_B_dose.png)

- 在 3.6 万到 7.7 万 token 之间，$`K=5\ldots50`$ 的 12 个 run 里有 11 个低于 base，其中 9 个低出超过噪声，最低到 45。唯一的例外是 K5s1。
- 同样的数据训练到 10 万 token 以上，15 个 run 全部 ≥ 65；到 40 万 token，大多在 95 到 118 之间。
- 拐点处的数据很不一样：$`K=5`$ 是 5 条示范重复约 20 遍，$`K=20`$ 是 20 条示范重复约 5 遍。重复程度差 4 倍，学生训练过的 token 数却差不多。
- 同一个 run 内，在拐点之前继续训练是负的：例如 K10s0 在 1.6 万、2.7 万、5.3 万 token 处依次是 59、55、46。

**为什么支持缺陷 1 和 3.**
- 每个训练样本都是验证正确的示范。静态估值在每一个时刻都给它们同样的正价值；实测的边际价值却先负后正，而且决定符号的是学生训练到了哪里。
- 这就是 [Theorem 3.2](../../theoretical_framework/statements/03-02_continuation_response_characterization_ZH.md#s3-2) 的 continuation response：同一条反馈的价值，取决于学生收到它时的状态 $`z_t`$。
- 对 baseline 的含义：在一个固定剂量下得到的"这批数据有没有用"的结论，换一个剂量可能反过来。各个 baseline 的配方不同，拐点的位置也不同，所以这里只能定性地说这一点。

**局限.**
- 窗口是看过数据之后才划的。
- 每个格子只有一个训练 seed。
- 下降的机制（格式漂移、早期遗忘或其他）还没有查清。

<a id="finding-c"></a>

## Finding C · 静态"学习信号"的大小不等于后果

**一句话.** 在同一个学生状态上比较三类数据：按 loss 和梯度这类静态量，teacher 修正（P2）的学习信号是其他两类的 4 到 24 倍。但裁剪先把三者的更新拉成几乎一样大；实测后果 P2 与 replay（P1）相同，而 loss 最低、最"好学"的自生成成功轨迹（P3）反而最差。

**静态方法怎么想.**
- "可学性"派（挑学生最熟悉、loss 最低的数据）选 P3。
- "信息量"派（难例挖掘，或者在学生出错的地方纠正，DAgger/LEAP 的思路）选 P2，并预期它的效果远大于其他两类。

**实验设置.** 这是预注册的 R1（[Remark 9.4](../../theoretical_framework/statements/09-04_scope_of_evidence_ZH.md#s9-4)）。
- 状态：$`z_{10}`$，即 K20 s0 Seed-SFT 的第 115 次更新（p10）的完整状态。
- 测量：把下一次参照更新换成某个候选的一个 batch，再用 common random numbers 跑 10 次参照尾段更新。同一个副本内，各候选共享尾段 seed。然后用 surrogate $`\tilde F`$ 打分：40 条留出的 teacher 成功轨迹的几何平均 token 概率。
- 候选与 baseline 家族的对应：
  - P1 = 重放 $`K=20`$ 条示范（BC/SFT）；
  - P2 = teacher 在学生出错状态上的修正（DAgger/LEAP）；
  - P3 = 环境验证成功的自生成轨迹（AgentEvol/RFT）。
- 副本数：P1、P3 各 64 个。P2 目前只完成 64 个中的 32 个，是中期结果；剩下一半正在排队。
- 每个 batch 都记录三个静态量：学生 loss（token NLL）、裁剪前的梯度范数、裁剪后的参数更新范数。

**结果.**

![Finding C](figs/p1_finding_C_r1.png)

- **静态量**（P3 / P1 / P2）：
  - 学生 loss：0.13 / 0.21 / 2.05；
  - 裁剪前梯度范数：4.2 / 27.2 / 100.6；
  - 裁剪后更新范数：0.0211 / 0.0217 / 0.0206（95–100% 的 batch 触发了裁剪）。
- **实测后果**（相对 P1）：
  - P3 为 −0.0014。64 个副本里只有 7 个高于 P1；超过诊断半径 0.0012，但严格半径是 0.325。
  - P2 为 +0.0002 ± 0.0002（中期，32 个副本，18 个高于 P1），在噪声内。
- **三种静态标准的排序**：
  - "可学性"给出 P3 > P1 > P2，与实测正好相反：P3 最差。
  - "信息量"给出 P2 ≫ P1 > P3。次序部分一致，但量级完全不对：P2 的梯度是 P1 的 3.7 倍，实测却与 P1 没有差别。
  - "来源"（teacher 来源优于学生自生成）在这里与实测一致。但它也是一个静态标准，这次碰巧对了；事先无法知道哪个静态标准会对。

**为什么支持缺陷 1.**
- 静态量描述的是这条数据"本身"有多大的信号，而后果要经过学习器才能确定。
- 第一步，optimizer 就把信号重塑了：裁剪把三种很不一样的梯度变成同样大的更新，Adam 又按坐标归一化。
- 再往后，后果取决于状态与后续训练（[Theorem 3.2](../../theoretical_framework/statements/03-02_continuation_response_characterization_ZH.md#s3-2)）。

**局限.**
- surrogate 没有校准，而且按构造就偏向 teacher 来源的数据，因为它打分的是 teacher 轨迹的似然。所以 P3 的劣势可能有一部分来自 surrogate 的偏向。
- 只有一个状态 $`z_{10}`$。
- P2 是中期结果。
- 只有诊断性的半径。
- 任务层面的续训还没有跑。

<a id="finding-d"></a>

## Finding D · 示范里缺的，replay 补不上

**一句话.** 有一组 $`K=5`$ 的示范里恰好没有 cool 类型。把这 5 条示范反复训练到 40 万 token，cool 任务一直停在 base 水平（25 个里做对 0–3 个）。另外两组各有 1 条 cool 示范，同样的训练能升到 7–13 个。

**静态方法怎么想.** 训练得更多、把已有轨迹改写、加反思、做对比配对，都能让学生学到更多。

**实验设置.**
- 分层抽样时 $`K=5`$ 少于 6 个任务类型，所以每个 K5 support set 都恰好缺一类：s0 缺 cool，s1 缺 pick-and-place，s2 缺 look-at-in-light。这是一个自然实验。
- 训练是 Finding B 的同一批 run，最多到 40 万 token，相当于 5 条示范重复 40–190 遍。
- 在 dev140 上按类型统计成功数：cool 25 个任务，look-at 13 个，pick-and-place 35 个。
- 另外统计每个 $`K`$ 在所有剂量中的最好成绩。

**结果.**

![Finding D](figs/p1_finding_D_information.png)

- **cool**：缺 cool 的 K5s0 在所有剂量上都是 0–3/25（base 是 1/25）。各有 1 条 cool 示范的 K5s1、K5s2 分别升到 11–13 和 7–8。
- **look-at**：缺这一类的 K5s2 在高剂量处是 5–8/13（base 是 7）；另外两组在高剂量处是 8–12。方向一致，但只有 13 个任务，证据弱。
- **pick-and-place**：base 已经做对 31/35，有天花板，看不出差别，不使用。
- **按 $`K`$ 的最好成绩**（3 个 seed 的中位数）：$`K=1`$ 为 63，2 为 71，5 为 82，10 为 95，20 为 110，50 为 116，100 为 115（teacher 114）。剂量决定能不能接近天花板，天花板本身随 $`K`$ 上升。

**为什么支持缺陷 2.**
- replay 是对同一批证据的后处理，后处理不能降低信息误差（[Proposition 2.4](../../theoretical_framework/statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)）。在示范看不见的方向上，replay 的价值为零（[Corollary 6.7](../../theoretical_framework/statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)）。
- 这里，"看不见的方向"就是 cool 任务的做法。40 万 token 的 replay 没有让它动；一条 cool 示范，也就是新的证据，就让它动了。
- 改写、反思、对比配对、知识合成，都是对同一批轨迹的后处理。按框架，它们与"更多 replay"处在同样的位置，补不了这里缺的信息。

**局限.**
- 只有 3 个自然出现的"缺一类"情形，其中 look-at 的样本小。
- 我们还没有直接测改写与反思（P4 改写候选正在排队）。
- 理论上 heat、clean 的示范可以迁移到 cool，这里没有发生；这个事实只对这个学生和配方成立。
- "所有剂量中的最好成绩"是在 dev 上挑最大值，偏乐观。

<a id="finding-e"></a>

## Finding E · 学生不只是它的参数

**一句话.** 在实际的 AdamW + 裁剪配方下：
- 每次参数更新的 72–100% 已经由 optimizer 的记忆决定；
- 两个 batch 一起的作用，有约一半不能由各自的作用相加得到（SGD 下只有约 3%）；
- 只改变 optimizer state、不动参数和数据，最终结果就差了 13 个任务。

**静态方法怎么想.** 学生就是参数；一条数据的作用就是它带来的那一步更新；样本的价值可以逐条打分再相加。

**实验设置.**
- **(a)(b) Adam response probe.**
  - 在 K20 s0 Seed-SFT run 的 9 个保存状态上测：冷启动；8 步与 32 步学习率为 0 的矩预热之后；第 1、4、8 次更新之后；p03、p10、p40 三个 checkpoint。
  - 每个状态取 4 对互不相交的 batch A、B，用 float64 计算一步参数更新：空 batch（记为 0）、A、B、A∪B。分别在 AdamW 加裁剪（实际配方）、AdamW 不裁剪、SGD 不裁剪三种 optimizer 下计算。
  - 记忆份额 = $`\lVert\Delta\theta(0)\rVert/\lVert\Delta\theta(A\cup B)\rVert`$。
  - 非可加性 $`\nu=\lVert\Delta\theta(A\cup B)-w_A\Delta\theta(A)-w_B\Delta\theta(B)\rVert/\lVert\Delta\theta(A\cup B)-\Delta\theta(0)\rVert`$，其中 $`w`$ 是 token 权重。
- **(c) optimizer 预热 pilot（pilot-warm-1010）.**
  - 训练计划和初始参数完全相同。mw32 臂在训练前先做 32 步学习率为 0 的 AdamW 更新：这会填充 $`m`$、$`v`$ 和步数计数器，但不改变参数。
  - 在同一台 server 上做 dev140 评测，与冷启动臂做配对 McNemar 检验。

**结果.**

![Finding E](figs/p1_finding_E_optimizer_state.png)

- **(a)** AdamW 加裁剪下，所有非冷启动状态的记忆份额都在 0.72–1.00 之间（冷启动时按定义为 0）。空 batch 产生的更新，已经是真实 batch 更新的 72–100%。
- **(b)** 非可加性 $`\nu`$ 的中位数：AdamW 加裁剪为 0.45–0.63，SGD 为 0.02–0.04。
- **(c)** 在 p10，结果从 66 升到 77（+11；19:8 个不一致任务，p = 0.052）；在 p40，从 112 降到 99（−13；6:19，p = 0.015）。
- 预注册的 R0b 在 response 层面也测到了非可加：中位数在第 8 次更新时为 0.19，在 p10 时为 0.86。

**为什么支持缺陷 3.**
- **(a)** 说明一个 batch 带来的即时参数变化，只占那次更新的一小部分；它的大部分作用，要通过 $`m`$ 分摊到之后的很多步里（[Theorem 7.1](../../theoretical_framework/statements/07-01_state_channel_decomposition_ZH.md#s7-1) 的动量通道）。只看一个快照上的即时作用来评价反馈，会漏掉这一部分（[Proposition 7.6](../../theoretical_framework/statements/07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)）。
- **(b)** 说明逐条打分再相加，也就是所有静态选择器默认的做法，在 SGD 下是不错的近似，在实际使用的 AdamW 加裁剪下不是。
- **(c)** 是 Proposition 7.6 在真实模型上的对应：数据与参数完全相同，只有 optimizer state 不同，最终结果就显著不同，而且在 p10 和 p40 方向相反。

**局限.**
- (a)(b) 是一步之内、在参数空间里的测量，不是任务结果。
- (c) 只有一个训练 seed。p40 的 p = 0.015 单独看显著；如果对这个 pilot 的 4 个配对检验（两个臂 × 两个 checkpoint）做 Bonferroni 校正（阈值 0.0125），就不显著了。
- 方向翻转的机制还不清楚。预注册的 optimizer state 移植检验 T1 正在排队，它会直接检验这一点。

---

## 还没有被支持的部分，以及怎样补强

| 缺陷的哪一部分 | 现状 | 补强实验 |
|---|---|---|
| 缺陷 2 · 新证据要按学习器分配 | 只有分配计划：在 24 个状态簇上，SGD 权重与 Adam 权重在 $`n=100`$ 和 $`n=400`$ 时分别把 57 个和 211 个状态送到不同的簇；还没有结果 | R2 正在运行（主判据已预注册） |
| 缺陷 2 · 改写、反思补不了信息 | Finding D 只测了 replay | R1 的 P4（teacher 改写示范）正在排队；Proposal 1 的 E2 直接比较改写与等成本的新查询 |
| Finding A 中训练随机性的占比 | 每组一个训练 seed | 对 3 个 support set 各补 2 个训练 seed |
| Finding B 的窗口是事后划的 | 一个训练 seed，checkpoint 稀 | 在 2–10 万 token 之间加密 checkpoint，并预注册窗口 |
| Finding C 的 surrogate 偏向 | 未校准 | 跑完 P2、P4 的另一半副本；用全尾段任务评测校准 surrogate（Proposal 1 的 E1） |
| Finding D 的受控版本 | 只有自然实验 | 固定其余 4 条示范，只增减 1 条 cool 示范；再比较 cool 状态上的 teacher 查询与改写 5 条示范（等成本） |
| Finding E(c) 的稳健性 | 单 seed | T1，以及 mw32 的第二个训练 seed |

## 数据来源

- dev140 评测：hpg 上 `$B/reports/runs/eval/{seedsft-sweep-bi*, baselines}` 的 rai 副本（`reports/runs/eval/hpg-*`）。
- R1：`$B/reports/runs/r03/r1/{p1p3-*.jsonl, p2p4-0.jsonl, merge_p1p3_*.json}`。
- Adam probe：`$B/reports/runs/pilot-warm-1010/probe/cold.json`。
- 只记在 `notes/exp_log.md` 里的数字，在 `build_data.py` 中逐条注明出处：pilot-warm-1010 的配对计数、LEAP $`K=50/100`$ 的逐轮结果，以及 ETO 起点为 p40 时的 loss 与梯度。

其中 $`B`$ = `/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation`。
