# P0：黑盒 teacher 能否读懂白盒学生（预注册草稿）

日期：2026-11-04 CST。状态：**定稿**（17:30 CST，真实 teacher 运行之前）。用户 12:39 CST 同意与 v0 并行推进。
工具：ICML27_wt_repair_eval 854edff（pinned ../ICML27_run_repair_eval_854edff），配对冻结 reports/p0_tooling/extraction_v3/pairs.json（sha256 cb4e1699…）；rai GPU smoke 通过（reports/p0_tooling/gpu_smoke_v3b）。
来源：
- 用户 11-04 的想法：teacher 虽是黑盒，学生是白盒；teacher 可以读取学生在输入后的内部计算，解读学生的思路，再结合 harness 判断这样做行不行、哪里不行、为什么不行。
- blog *Sparks of Weight-Space Programming*（rikiyat.github.io，11-04 重读原文）：
  - 改权重的是 LLM 本身（GPT-6 Astra）。它有一个可读写权重的 Python kernel，没有 GPU，不能训练、不能用梯度，forward pass 预算为 10/25/50 批。
  - 它找到第 15、19 层 MLP 写入的循环方向，在每层 64 个输出列上减去该方向，共 131,072 个权重（0.016%）。
  - 结果：MATH-500 7.9→48.1，GSM8K 13.8→67.1，IFEval 21.0→62.3；在 Gemma 3 270M 上失败。

## 1. 问题

一句话：只给 gpt-6-luna 读取学生内部状态的工具和 harness，它能否给出**可验证**的失败诊断，并据此提出一个能推广到未见失败的权重干预？

- 学生：θ₀（K20 轨迹 SFT seed 0，LoRA 合并后的权重），不训练。
- 全部数据来自训练侧：v0 第一轮 R 桶的修复记录。不使用 dev140、留出题或任何冻结评测集。

## 2. 单位

- 一个**配对**取自一个已验证的修复 episode：原失败轨迹与**最终验证通过的修复执行**之间的第一个分叉步 t*。
  - 状态 s：原轨迹在第 t* 步决策前的完整上下文。t* 之前两条轨迹的动作相同，且逐步核对了可见观测完全一致，所以修复执行在 t* 看到的是同一个状态；
  - 失败动作 a_f：学生原来在第 t* 步的动作；
  - 修复动作 a_r：最终验证执行在第 t* 步的动作，要求它来自 teacher 的 patch（executors[t*] = patch），而不是学生重新解码。
- 草稿原用"teacher 列出的第一个 breaking point"作单位，抽取时发现两处不成立，因此改为上面的定义：
  - teacher 的 patch 常常不从它标的第一个 breaking point 开始改（例如 bp 标在第 1 步，patch 从第 23 步才开始）；
  - 134 个修复中有 27 个经过 2–3 轮 review，最终执行的动作与第 0 轮 patch 不同。
- 类别：取第 0 轮 review 中 step = t* 的 breaking point；没有时取距离最近的，距离相同取较晚的一步。
- 筛选结果：134 个已验证修复中，可用 113 个，覆盖 74 个任务实例。
  - 排除 21 个：18 个分叉动作来自学生重新解码，3 个类别为 other；
  - 类别来源：69 个恰在分叉步，35 个取后面最近的，9 个取前面最近的。
- 划分：按任务实例分组（seed 0，Random(0) 打乱分组；对每个类别依次加入含该类别的分组，直到诊断集该类别至少 4 个配对）。
  - 诊断集 23 个配对（14 个实例），各类别为 goal_misread 4、missing_precondition 5、refused_action 5、search_incomplete_or_loop 4、wrong_object_or_target 5；
  - 测试集 90 个配对（66 个实例），各类别为 goal_misread 23、missing_precondition 32、refused_action 14、search_incomplete_or_loop 8、wrong_object_or_target 13；
  - 两边任务实例和状态都不重叠；有 10 个 parent seed 两边都有，只作报告。teacher 永远看不到测试集。

## 3. 工具（只在诊断集上开放给 teacher）

工具作用在学生的单个决策点上，返回文本或数值摘要。学生每一步都先写 Thought 再写 Action，所以决策点工具（1、2、3、5、6）默认以学生自己在该步的原始 Thought 为前缀（original_thought），也可以选 greedy_thought（θ₀ greedy 生成的 Thought）或 no_thought；所用 framing 记录在每个结果里。

1. **动作打分**：给定状态和候选动作，返回每个动作的完整 log-prob，以及学生的 greedy 动作。
2. **逐层 logit lens**：每层残差经最终 norm 和 unembedding 之后，a_r 与 a_f 首个差异 token 的 log-prob 差。
3. **注意力汇总**：在每层、每个 head 组上，决策位置对各标注片段的注意力质量。片段包括任务目标、最近观测、更早各步观测、可执行动作列表和 Thought。
4. **上下文反事实**：teacher 改写 s_t（删除、插入或重排片段），得到新状态，可再调用 1–3、5。
5. **激活修补**：在指定层和位置，把一次运行的残差替换成另一次运行的残差，返回动作打分的变化。
6. **方向工具**：
   - 按 teacher 指定的两组（状态, 动作）teacher-forcing，在动作 token 位置取均值，计算两组之差作为方向；
   - 返回方向在各层的投影；
   - 应用权重编辑：在选定层的 MLP 或 attention 输出投影（down_proj / o_proj）上做 W ← W − α u uᵀ W，再做动作打分。其中 u = normalize(g ⊙ d)，g 是紧跟该投影的 post-norm 权重（Gemma-4 的输出先经 RMSNorm 再加入残差），这样 α = 1 时该层写入残差的部分在 d 上的分量为 0。
7. **harness rollout**：从 s_t 起，用（可带编辑的）学生 greedy 跑到结束，返回成功与否和动作序列。

预算：每个类别最多 60 次工具调用、10 次 rollout、5 个候选编辑、400 个真实环境步。teacher token 只设失控上限（每类别输入 8M、输出 1M；每轮会重发配对和完整记录，预计每类别约 4–5M 输入），用量照实记录。
运行：teacher 为 gpt-6-luna（OpenAI，与 v0 轮相同的 provider）。五个类别分三条链在 rai GPU 1 上并行诊断：goal_misread→missing_precondition、refused_action→search_incomplete_or_loop、wrong_object_or_target。§5 的测试也全部在 GPU 1 上进行。（17:28 CST 更正：原计划分到 GPU 0/1/3，但 CLI 只允许 GPU 1，GPU 0/3 的两条链在加载模型前就被拒绝，没有调用 teacher。）

## 4. teacher 的产出（每个类别一份，提交后冻结）

- 用文字写出机制假设：学生为什么在这里选 a_f。
- 位置：涉及的层、片段和方向。
- 一个**部署用编辑**：只能是 §3.6 的权重编辑（层、方向的构造方式、α），可以在不同类别之间复用。
- 若干条可检验的预测，写明在测试集上预期的效果，例如"决策翻转率 > 30%，rollout 成功率高于未编辑"。

## 5. 测试（由我们执行，teacher 不参与）

- 在该类别的测试集上比较以下条件：
  - θ₀ 未编辑；
  - teacher 的编辑；
  - 对照一：同样的层、同样的 α，换成随机方向（5 个随机种子取均值）；
  - 对照二：固定规则方向，即同一类别诊断集上 h(s, a_f) − h(s, a_r) 的均值（original_thought 前缀，teacher-forcing 动作 token 位置），层取 D-param 选中的第 20–41 层，α 和投影类型取 teacher 选的值。
- 指标：
  - 主要：从 s_t 起 greedy rollout 的成功率；
  - 次要：决策翻转率，即 greedy 动作不再是 a_f，分母只计 θ₀ greedy 能复现 a_f 的测试配对，同时报告复现率；
  - 附带伤害：从 S 桶学生成功轨迹中抽 200 个决策，统计 greedy 动作被改变的比例；K20 原题的成功率。
- 判定：
  - **P0 成立**：合并五个类别后，teacher 编辑的 rollout 成功率高于未编辑和两个对照，McNemar 精确双侧检验 p < 0.05；随机方向的 5 个种子各做一次检验，取最大 p，并要求每个种子下 teacher 都更好；附带伤害 ≤ 5%（每个类别的编辑都要满足）；K20 原题不下降。
  - **诊断忠实度**：teacher 预测中在测试集上成立的比例，按条目报告。
  - 文字诊断作为定性结果报告，不参与判定。

## 6. 说明

- 修正部件 S 草稿的一个问题：S1 用"决策 token 前最后位置的 h"求 fail/fix 均值差，但同一配对的前缀完全相同，两者的 h 也相同。S1 定稿时改成 §5 对照二的做法（teacher-forcing 动作 token）。
- 若 P0 成立，P1 就是 S-agent：按诊断在数据、优化、权重三层中选择干预，用新合成的留出题验收；固定规则的 S 作为对照。
- 计算：rai 一张 GPU（保留至少一张空闲），HF bf16 前向加 hook，环境用 envs/p0-venv（v0 轮环境包 + torch 2.13/transformers 5.14.1/peft 0.20）。smoke 实测：模型加载 52 s，单次工具调用 0.03–5 s，40 步 rollout 70 s。
- 测试集每个条件 90 次 rollout；条件为未编辑、teacher、随机 ×5、固定规则，合计约 8 × 90 = 720 次，每类别的编辑只在该类别测试集上跑，约 14 GPU 小时。另有附带伤害 200 个决策 × 每个编辑，以及 K20 原题。

## 7. 结果（11-05，测试集评测完成后补记；以上各节不改）

测试集 90 个配对，每个条件从分叉状态起做一次 greedy rollout，按配对做 McNemar 精确双侧检验。

| 类别 | N | 不编辑 | teacher 编辑 | 随机方向（5 seed 均值） | 固定规则方向 |
|---|---:|---:|---:|---:|---:|
| 目标理解错误 | 23 | 13.0% | 21.7% | 13.0% | 21.7% |
| 缺少前置条件 | 32 | 28.1% | 31.2% | 26.9% | 37.5% |
| 动作被拒后继续 | 14 | 14.3% | 28.6% | 14.3% | 35.7% |
| 搜索不完整或循环 | 8 | 0.0% | 12.5% | 0.0% | 12.5% |
| 物体或目标选错 | 13 | 46.2% | 46.2% | 40.0% | 38.5% |
| **合并** | 90 | 22.2% | 28.9% | 20.9% | 31.1% |

- teacher 编辑 vs 不编辑：8 对变好、2 对变差，p = 0.11；vs 随机方向：5 个 seed 中最大 p = 0.15（逐个为 0.09 / 0.002 / 0.11 / 0.065 / 0.15）；vs 固定规则方向：10 对 vs 12 对，p = 0.83。
- 固定规则方向 vs 不编辑：16 对 vs 8 对，p = 0.15。
- 附带伤害：每个类别的 teacher 编辑在 200 个原本做对的决策上改变 0–5%，都在 5% 以内；K20 原题仍全部成功。
- 决策翻转率（分母为 θ₀ 能复现 a_f 的 77 对）：teacher 0–25%，固定规则 0–50%，随机方向几乎为 0。
- teacher 自己写的 24 条预测中 19 条在测试集上成立（79%），但大多是"增益 > 0""附带伤害 ≤ 5%"这类宽松的预测。
- teacher 用量：诊断阶段约 7.75M 输入 token。

**结论**：
- 按预注册判定，P0 不成立：teacher 的编辑没有显著优于不编辑，也没有优于固定规则方向。黑盒 teacher 读取学生内部状态之后提出的方向，并不比"失败动作与修复动作的残差之差"这条简单规则更好。
- 秩一权重编辑是安全的（附带伤害小），但效果也小：在没见过的失败上最多提高几个百分点。
- 这与 D-param 的结果一致：到目前为止，在参数层面做定向干预（只训练选中张量、直接编辑权重）都没有带来收益，收益来自数据（teacher 修复的 on-policy 轨迹与合成练习）。
- 因此按预注册不进入 P1（以权重编辑为核心的 S-agent）。下一步的改进放在数据一侧。

- 评测产物：`../ICML27_run_repair_eval_0410c67/reports/p0_runs/evaluate_v4_min/{report.json,report.md,results.jsonl}`（tree 0410c67 诊断冻结，评测 tree 9b85c8f 续跑与合并）。
