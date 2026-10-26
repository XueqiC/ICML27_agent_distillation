# 保留学生计划的交互式轨迹修复蒸馏：Pipeline v0.1

日期：2026-10-26  
状态：研究设计与实现规格；尚未执行本文实验，数值默认值均为起始配置，不代表已验证的最优值。  
适用设定：黑盒 teacher、少量初始成功示范、可执行并判断任务成功的 agent 环境。  
首个环境：ALFWorld TextWorld。初始示范网格 K = 10/20/30/100。固定 teacher/student 为 GPT-6-luna / Gemma4-E4B-it；运行时记录实际 API model ID、模型 revision 与 checkpoint hash。

> **核心目标：让 teacher 理解学生已经选择的路线，提出可以联合、可以不连续的修改；通过真实环境执行验证修改，再让学生学习这些接近自身行为分布的成功轨迹。**
>
> v0.1 的学生更新仍采用 SFT。研究变化在于监督从哪里来、如何验证，以及如何按学生当前缺陷构造练习。本文不把它描述为一种新的参数优化范式。


**实施入口：** 先执行第 14 节的快速机制实验与已有 baseline 复用方案；第 2–13 节是完整 pipeline 规格，第 15 节是后续扩展顺序。第一次只跑一个 K、两个短训练臂。

## 1. 本版固定的研究问题

### 1.1 四个可以分别证伪的假设

| 编号 | 假设 | 直接检验 |
|---|---|---|
| H1：修复可行性 | 很多失败可以通过保留学生路线的少量联合修改变成成功 | 同一失败轨迹上加入、撤销、恢复修复 blocks，并比较实际环境结果 |
| H2：学习收益 | 在同样的失败案例和训练配置下，接近学生原轨迹的成功监督更容易被学生学会，并迁移到其他任务 | 同起点训练，比较完整重写目标与联合修复目标；先保持 loss 相同 |
| H3：条件迁移 | 从修复中提取行为适用条件，再构造条件保持／条件翻转练习，比重复原错题或随机增加任务更有效 | 同等新任务数和监督预算下比较练习来源 |
| H4：梯度分配 | 当前学生对修复样本的梯度关系，可以改善练习预算分配 | 随机、语义分组、梯度分组采用相同候选任务池与预算 |

**第一步只验证 H1 和 H2。** H1/H2 未成立前，不用任务生成、梯度聚类或多轮训练掩盖问题。完整 pipeline 保留 H3/H4 的接口。

### 1.2 不预设的结论

- 早于最终失败的某次偏离不自动构成错误；与 teacher 示范不同也不构成错误。
- 单个动作改变后仍失败，不证明该动作无关；可能需要多个修改同时成立。
- 找到成功修复，不证明找到了唯一错误原因，也不证明修复全局最小。
- 学生写出的 reasoning 是其表达的计划和假设，不保证等于内部真实决策原因。
- 教师辅助执行成功，不等于训练后的学生能独立成功。
- 小改动不保证更容易学习；这是 H2 要检验的内容。
- 全离线 agent 轨迹训练仍属于 agent distillation。这里研究的是交互式、学生轨迹驱动的蒸馏。

## 2. 实验协议、数据边界与预算

### 2.1 两种协议必须分开命名

| 协议 | 训练任务范围 | 允许追加的监督 | 用途 |
|---|---|---|---|
| P0：固定 K 个任务 | 始终只有原始 K 个训练任务 | 学生重试、teacher 修复、环境反馈 | 第一阶段机制验证 |
| P1：K 个种子示范 + 追加练习 | 原 K 个任务，加从训练环境取得或生成的新任务 | 同上，以及新任务的成功轨迹 | 完整 pipeline |

P1 只能称为“从 K 个初始示范出发”。如果随后取得 80 条修复轨迹，不能称为“总共只使用 30 条监督”。

环境训练反馈是本协议允许的信息来源。与纯离线 SFT 比较时，标记交互预算差异，并加入交互式对照。

### 2.2 数据集合

| 集合 | 默认值 | 权限 |
|---|---|---|
| D_seed | K ∈ {10,20,30,100}，优先复用各 K 的现有任务及成功示范 | 可训练、重试、修复 |
| U_train | 官方训练部分中，排除 D_seed、D_probe 和评估任务后的合法任务池 | P1 可检索并取得新监督；P0 不使用 |
| D_probe | 60 个预先固定的训练侧保留任务 | 只运行独立评估；不送 teacher、不入梯度库、不用于生成练习 |
| D_eval | 现有约 140 个任务，保持已有 harness | 报告与已有实验的可比结果，不参与本 pipeline 数据生成 |
| D_aug,r | 第 r 轮取得并验证的新任务 | 仅 P1 可训练；保存来源和父修复案例 |

划分规则：

1. 优先保留各 K 已有的 D_seed，不为了制造更大的提升重新挑选任务。
2. 没有既有划分时，按环境任务类型分层，固定随机种子抽取；任务类型不足时按剩余可用数量分配，保存最终 ID 清单。
3. D_probe 在首次修复前选定。默认按环境任务类型分层，不根据本方法的评估结果重新挑题。
4. 任务去重键为 **环境版本 + 场景／初始状态文件 hash + 规范化目标 + 随机化配置**；不能只按自然语言 instruction 去重。
5. 仅改写 instruction、未改变语义和环境的样本算同一任务，不算新任务。
6. 如果现有 140 个任务已长期用于选模，将其标记为 development/reference set；本文不把历史用途重命名为 untouched test。第一阶段无需因此扩大实验规模。

### 2.3 三类预算独立记账

~~~yaml
supervision_budget:
  initial_teacher_demonstrations: K
  additional_verified_teacher_trajectories: measured
  additional_teacher_labeled_decisions: measured
  teacher_calls:
    diagnosis: measured
    proposal: measured
    rationale: measured
    task_generation: measured
    retry: measured
  teacher_input_tokens: measured
  teacher_output_tokens: measured
  teacher_cached_tokens: measured_if_reported
  teacher_billed_cost: measured_if_available

interaction_budget:
  student_environment_steps: measured
  candidate_replay_steps: measured
  edit_ablation_steps: measured
  task_validation_steps: measured
  handback_diagnostic_steps: measured
  evaluation_steps: measured_separately

optimization_budget:
  optimizer_updates: measured
  processed_context_tokens: measured
  supervised_target_tokens: measured
  trainable_parameters: measured
~~~

计数规则：

- 失败、解析失败、超时、被丢弃候选和教师重试都计入预算。
- 环境重新 reset 后重放前缀的动作也计入环境步数。
- API 输入缓存降低费用时，同时报告原始输入量与实际费用；不能把缓存 token 当成未使用监督。
- 不用“调用次数相同”替代 token 成本相同，也不用“epoch 相同”替代优化预算相同。

## 3. 启动条件与环境接口

### 3.1 Student 的起点

默认从现有 **仅使用 D_seed 做 SFT 的 checkpoint θ₀** 开始交互修复。这样直接回答：在已有的对应 K 的 SFT 结果上，交互修复能否带来额外收益。

可选起点：

| 选项 | 定义 | 何时使用 |
|---|---|---|
| I0，默认 | θ₀ = SFT(D_seed) | 验证当前 few-shot SFT 的后续瓶颈 |
| I1 | θ₀ = 原始 instruction-tuned student | 检验是否需要 seed SFT；作为独立消融 |
| I2 | θ₀ = 使用固定少量 seed 数据完成格式适配的 checkpoint | 原模型不能稳定输出可解析动作时 |

同一组对照从完全相同的 θ₀ 开始。更换起点即更换实验，不把两组结果当成单变量比较。

### 3.2 必须实现的环境适配器

以下是项目自定义接口，**不是声称 ALFWorld 原生具有同名 API**。

~~~python
class EnvAdapter:
    def reset(self, task_spec, env_seed) -> PublicObservation: ...
    def step(self, action_text) -> StepResult: ...
    def terminal_success(self, step_result) -> bool: ...
    def canonical_action(self, action_text) -> CanonicalAction: ...
    def public_entities(self, public_history) -> set[str]: ...
    def task_fingerprint(self, task_spec) -> str: ...
~~~

契约：

- reset 必须指向指定任务及初始状态，而不是从训练环境随机取下一题。
- step 返回环境原始 observation、是否结束、成功标记和错误状态；保存原文。
- success 读取并核对当前 harness 的官方成功字段。不能把“任何正 reward”“done=True”或 teacher 自评直接当成成功。
- canonical_action 只规范化空白、动作模板和同一实例的确定性别名；不能把 apple 1 和 apple 2 合并。
- public_entities 从当时公开的任务描述／observation／已公开的动作空间提取。它不是隐藏世界状态读取接口。
- 不默认需要环境快照。v0.1 通过 reset 同一任务并重放动作恢复前缀。
- 如果环境提供快照，快照只用于减少计算；重放步数和快照开销分别记录，不能宣称与无快照对照成本完全相同。
- task factory、任务编辑器和场景编译器属于后续独立模块；没有实现时不能把语言模型输出的一道题视为可执行环境任务。

ALFWorld 官方 README 提供 reset/step 的交互示例，但没有保证本文所需的任意轨迹编辑、任务快照或按目标改写环境均可直接调用。[R1]

### 3.3 最小环境验收

正式收集前固定 5 个训练任务，每个执行两遍相同动作序列，检查：

1. reset 后任务 fingerprint 与初始 observation 相同。
2. 每个动作后的规范化公开 observation 与成功结果一致。
3. 原 teacher success 示范在当前版本仍成功。
4. 成功、失败、步数耗尽三种结束原因可以区分。
5. teacher 与 student 看到的环境信息范围相同；额外隐藏状态仅允许留在验证进程，不能进入训练文本。

若重放不确定，先禁用可配置的场景／对象编号随机化并重新验收。仍不确定时转入随机环境 variant：候选重复 3 次，报告成功次数，不再使用“一次重放证明编辑必要性”的表述。v0.1 的主实验要求通过确定性验收。

## 4. 公共运行约定与数据结构

### 4.1 轨迹与历史

对任务 x，定义：

\[
h_t=(x,o_0,r_1,a_1,o_1,\ldots,r_{t-1},a_{t-1},o_{t-1}),\qquad
y_t=(r_t,a_t).
\]

- r_t 是公开记录的 reasoning；没有此通道时为空。
- a_t 是送给环境的动作。
- o_t 必须来自真实环境。
- reasoning 不直接改变环境状态；它可能影响学生后续生成，因此训练历史仍需保持一致。
- 一轮收集期间冻结 θ_r。该轮所有梯度特征、学生 rollouts 和生成任务上的学生尝试都使用这个 checkpoint。
- history、chat template、system prompt、动作解析器对所有实验臂相同。

### 4.2 需要落盘的记录

| 记录 | 必需字段 |
|---|---|
| RunManifest | run_id、代码版本、环境版本、模型版本、checkpoint hash、tokenizer/chat-template hash、任务清单 hash、配置、所有 seed |
| TaskRecord | task_id、fingerprint、split、initial_state_ref、goal、env_seed、parent_task_id、parent_repair_id、generation_mode |
| TrajectoryRecord | trajectory_id、task_id、student_checkpoint、每步 h/r/a/o、success、termination_reason、token/step 成本 |
| DiagnosisRecord | 原轨迹 ID、计划假设、证据 step IDs、待满足目标、候选联合修改、不能确认的内容 |
| PatchRecord | patch_id、原轨迹 ID、基于原动作索引的 edit blocks、候选来源、搜索轮次 |
| RepairRecord | 原轨迹、选中 patch、实际执行轨迹、动作对齐、成本、每次验证结果、最小性检查状态 |
| LearningEpisode | task_id、repair_id、完整有效历史、被监督的 decision IDs、token mask、来源桶、checkpoint hash |
| GradientRecord | episode_id、参数列表 hash、checkpoint hash、loss 配置 hash、投影 seed、向量、原始范数 |
| CurriculumRecord | cluster_id、代表修复、条件规格、候选任务、可执行性验证、接纳／拒绝原因、预算 |
| EvaluationRecord | checkpoint、任务 ID、独立 rollout、成功、环境步数；不得含 teacher 修复 |

### 4.3 状态枚举

~~~text
trajectory_status:
  SUCCESS | FAILED | STEP_LIMIT | CONTEXT_LIMIT | PARSE_FAILURE | ENV_ERROR

repair_status:
  VERIFIED | NO_SUCCESS_WITHIN_BUDGET | NONREPLAYABLE
  | INVALID_PATCH | UNGROUNDED_TARGET | LABEL_BUDGET_EXHAUSTED

minimality_status:
  BEST_OBSERVED
  | BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS
  | ABLATION_BUDGET_EXHAUSTED

task_status:
  EXECUTABLE_VERIFIED | INVALID_TASK | UNSOLVED_WITHIN_BUDGET
  | DUPLICATE | SPLIT_VIOLATION
~~~

“预算内未找到成功”不能编码成“任务不可解”。教师认定的原因只保存在 diagnosis，不能写成已验证的 causal ground truth。

## 5. 每轮步骤 A：学生尝试与案例选择

### 5.1 默认 rollout

- 对当前训练任务池中每个任务运行 2 次：第 1 次 greedy；第 2 次 temperature = 0.7、top_p = 0.95。
- 初始任务池是 D_seed；P1 后续加入已验证的 D_aug。
- 最大环境动作数 H = 50，若已有 baseline 使用不同 H，则在整个对照组统一采用已有 H，并记录该预注册覆盖。
- 每个 assistant turn 最多生成 256 个 token；使用现有动作格式与 parser。
- 非法动作、解析失败均保留记录；parser 的恢复策略必须沿用 baseline，不能只给新方法额外恢复机会。
- 不让 teacher 在这一步接管，也不把原 teacher 示例放入额外的动态提示中。
- 默认上下文上限 16,384 token；如模型或已有 harness 上限不同，运行前统一锁定。溢出标记 CONTEXT_LIMIT，不静默丢弃任务目标或关键观察。

### 5.2 成功池

成功学生轨迹直接保留，不改写成 teacher 的风格。每轮每任务至多保留 1 条：优先较少环境动作，平局按 trajectory_id 排序。保存其他成功轨迹作为诊断记录，不重复放大该任务的训练权重。

### 5.3 失败池

每轮从已有训练任务的失败中选最多 20 条送入 teacher 修复：

1. 按任务类型分层；
2. 每个任务先取 1 条失败，之后才允许第 2 条；
3. 层内用固定 seed 的打乱顺序；
4. 不按 teacher 是否容易修好挑选；
5. 原始失败总数与被选比例都要报告。

快速机制实验需要至少 6 个不同任务的可用失败。不足时，可对同一组 K 个训练任务追加最多 2 次随机 rollout；仍不足按第 14.0 节处理，不故意破坏学生输出或使用评估集失败。

## 6. 每轮步骤 B：理解计划与提出联合修复

### 6.1 Teacher 能看到什么

可见：原任务描述、完整公开学生轨迹、环境最终是否成功、步数限制和动作语法。  
默认不可见：原 teacher 成功路线、隐藏对象位置、内部状态、测试任务及其结果。

完整轨迹用于事后诊断。后文的 student 训练样本仍只提供相应时间点可见的历史。

### 6.2 Diagnosis 的结构

~~~yaml
diagnosis:
  expressed_plan:
    - subgoal: "学生试图完成的子目标"
      evidence_steps: [3, 7]
      interpretation_status: hypothesis
  preservable_choices:
    - choice: "原路线中可能保留的对象／顺序／探索方向"
      evidence_steps: [3]
      status: candidate_until_execution
  unmet_requirements:
    - requirement: "由原任务文本要求的结果"
      evidence_steps: [0, 12]
      certainty: supported_or_unknown
  candidate_joint_changes:
    - block_ids: [p1, p2]
      explanation: "为何需要同时改变这些部分"
  unknowns:
    - "公开轨迹不能确定的信息"
~~~

Teacher 必须区分：

- 动作当时无效；
- 动作有效但不能完成目标；
- 动作有效，且可保留，只是后续缺少必要步骤；
- 单步是否有问题不能判断，需要检验联合修改。

诊断不要求定位“第一个错误”。若整条计划不可保留，可以提出较大修改，但不能把它伪装成局部修复。

### 6.3 Patch 的可执行语法

原动作序列用 0-based 索引 a₀ … aₙ₋₁。每个 block 为以下一种：

~~~yaml
patch:
  original_trajectory_id: trace_001
  blocks:
    - block_id: p1
      op: insert_before
      index: 5
      actions: ["由当前环境 parser 接受的实际动作字符串"]
    - block_id: p2
      op: replace
      start: 9
      end_exclusive: 11
      actions: ["替换动作 1", "替换动作 2"]
    - block_id: p3
      op: delete
      start: 14
      end_exclusive: 16
      actions: []
~~~

编译规则：

1. 所有索引始终引用原轨迹，不引用前一个 edit 后的偏移。
2. replace/delete 区间不能重叠；同一边界最多一个 insert_before。
3. insert_before 的 index = n 表示在原序列尾部追加。
4. 插入点可位于 replace 区间左边界，但不能在被 replace/delete 区间内部。
5. 按原索引从左到右编译；同一边界先插入，再执行 replacement 或原动作。
6. 不支持任意代码、隐藏状态读取或修改 observation。
7. 调换动作顺序用 delete + insert 表示，不能获得免费的 reorder。
8. v0.1 每个候选最多 4 个 blocks；每个 block 可含多个实际环境动作。
9. 超过 4 个 blocks 的方案记录为 extended-repair variant，不混入默认结果。

Teacher 默认一次提出 3 个动作序列不相同的候选。候选重复由编译后的规范化动作序列 hash 判定。

## 7. 每轮步骤 C：有限预算的最小修复搜索

### 7.1 目标的准确含义

固定原任务 x、原初始状态 s₀、原学生轨迹 τ。候选集合 C_B 是预算 B 下实际搜索到的 patches：

\[
\widehat{\Delta}
=\arg\min_{\Delta\in C_B:\,V_E(x,s_0,\Delta)=1}
\left(d_{\mathrm{act}}(\tau,\tau_\Delta),-p(\tau,\tau_\Delta),|\tau_\Delta|\right).
\]

按元组字典序排序：

1. d_act：原动作序列与实际成功执行序列的 Levenshtein 距离；插入、删除、替换各 1 分。
2. p：在选中成功轨迹中仍实现的、diagnosis 预先登记的学生子目标个数；只作为平局项。无法可靠验证时设为 0。
3. 成功轨迹的实际环境动作数。
4. 仍平局按 patch_id 排序。

具体约定：

- 动作距离不包含 reasoning 文本，也不把 1 个 block 当成 1 个动作。
- 成功后的未执行后缀不进入新轨迹；它相对原轨迹产生的删除仍计入距离。
- 规范化距离定义为 d_act / max(1, 原长度, 新长度)，主要用于跨轨迹报告；选择候选时使用原始距离。
- 不因 teacher 说“只改两处”就把新生成的 12 个动作算成 2 次修改。
- 该目标允许保留不最优但有效的探索路径。另报无效动作率和路径长度，不假定最小修改同时最省环境步骤。

### 7.2 候选验证

每个候选：

1. reset 原任务与原随机种子；
2. 执行编译后的动作序列；
3. 每步读取新的真实 observation；
4. success 时立即停止；超过 H 或候选动作耗尽仍未成功，则判失败；
5. 不把后续动作自动改写为更聪明的策略。需要改写时，必须创建并重新计价的新候选；
6. 一旦环境发生无法恢复的错误，记录 ENV_ERROR，不把该例判为修复有效。

每个失败案例最多两轮 proposal，每轮 3 个候选。第二轮 teacher 可见上一轮实际执行记录与失败结果，必须重新输出完整 patch。

**动作之后的环境状态变了，就必须重新生成 observation。禁止把旧 trace 的 observation 拼接到新动作后面。**

### 7.3 最小性检查

从成功候选中按 7.1 选出当前最优者，然后逐个撤销 edit block：

- 撤销表示恢复原轨迹对应部分，其他 blocks 保持不变；
- 从原初始状态重新执行；
- 如果仍成功且排序成本下降，接受撤销，再从第一个剩余 block 开始检查；
- 直到完整一遍没有可撤销 block，或预算耗尽。

默认预算：

| 项目 | 每条原失败轨迹上限 |
|---|---:|
| 原轨迹重放确认 | 1 次 |
| proposal 调用 | 2 次，每次最多 3 个候选 |
| 单 block 撤销测试 | 12 次 |
| 选中成功候选的重复确认 | 1 次 |
| 额外学生接手续跑诊断 | 最多 3 次 |
| 上述环境 replay 合计 | 最多 24 次，每次不超过 H 步 |

重复候选使用缓存，但仍记录 cache hit；不重复计入实际环境步数。所有缓存键包含环境版本、task fingerprint、动作序列与 seed。

只在完整一遍撤销测试完成时标记 BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS。此结论仅针对预先声明的 blocks：一个 block 内可能仍含多余动作。其余情况标记 BEST_OBSERVED 或 ABLATION_BUDGET_EXHAUSTED。

### 7.4 联合性如何记证据

对含两个 block 的成功修复，若：

- 同时采用 p₁、p₂ 时成功；
- 撤销 p₁ 时失败；
- 撤销 p₂ 时失败；

则记为“在这两个具体替换及固定后缀下，两者都必要”。

这不证明不存在另一种单 block 成功修复，也不证明原轨迹存在唯一两个根本原因。不能从该局部测试直接推出普遍的因果交互结论。

### 7.5 任务修复与任务改写严格分开

在本节中，不允许：

- 删除原目标中的 clean/heat/cool 等要求；
- 把目标容器换成学生最后放置的容器；
- 修改初始状态，使原失败轨迹变成功。

上述变化可以在第 10 节作为新练习，但它们不计为原题修复成功。

### 7.6 学生接续诊断

选中 patch 后，取每个 edit block 的最后一个修改动作执行后的状态，至多取前 3 个位置。通过 reset + 重放修复后前缀恢复状态，再让冻结的 θ_r 独立继续到原总步数 H。

记录：

- handback_success：学生能否在接受局部帮助后完成剩余任务；
- completion_steps；
- 新失败类型。

这项结果用于解释“修复可以执行但学生仍不会续做”。默认不据此删除训练样本，否则会系统性排除较难学习的修复。


## 8. 每轮步骤 D：把成功修复转成可学习的数据

### 8.1 三种信息不能混用

| 信息 | 可以看到完整失败轨迹吗 | 进入 student 的普通决策输入吗 |
|---|---|---|
| 事后 diagnosis | 可以 | 不进入 |
| 当前决策的 reasoning/action 标签 | 标签构造可利用修复动作，但事实说明只能来自当前公开前缀 | 作为当前 assistant target |
| 当前决策的输入历史 | 不可以包含未来 observation | 进入 |

不能把“你刚才失败是因为第 17 步……”放进普通决策训练输入，然后在测试时移除它，却称为相同决策任务。

### 8.2 重建 reasoning 的默认方案

动作执行正确不保证原 reasoning 仍与新状态一致。默认：

1. 第一个动作修改之前，保留完全相同的原始历史。
2. 从第一个修改处开始，对选中的成功动作轨迹逐步重建 reasoning。
3. 每次 reasoning 标签调用只输入：原任务、当前修复后的公开前缀、当前已选定动作。
4. **该调用不输入完整失败轨迹、完整未来动作计划、未来 observation 或事后 diagnosis。**
5. 每次最多输出 64 个 reasoning token；要求短的决策依据，不要求复述教师长篇分析。
6. 当前 turn 的 reasoning 生成后加入历史，再处理下一步。可在不同轨迹间并行，不能在同一轨迹内用未来标签构造过去标签。
7. 记录支持事实的历史 step IDs。自动检查引用均早于当前动作，并检查提及的实例 ID 已公开。
8. 不能由结构检查证明自然语言完全无幻觉。第一阶段对入选的全部修复案例做人工前缀检查，记录不支持的陈述；修正或丢弃相应标签后再训练。

这里得到的是“支持该动作的、前缀一致的说明”，不声称恢复了学生或 teacher 的真实内部思考。

如果既有 harness 不输出 reasoning，则直接使用 action-only 标签，不额外引入 reasoning 通道。如果要把已有 reason-action harness 改为 action-only，所有对照必须一起改，并作为独立格式消融。

### 8.3 当前动作也必须可落地

- 使用未出现且不能从任务／公开 observation 确定的实例编号，标记 UNGROUNDED_TARGET。
- 对尚未观察内容的容器进行探索可以是合法动作；不能因此认定 teacher 偷看隐藏状态。
- “去已知的柜子看看”与“我知道未观察的柜子里有 apple 2”不同。后者不能成为当前 reasoning 标签。
- 环境反馈中原样存在的信息可以使用；不额外给 student 暴露隐藏 admissible-command 列表。
- 教师看过未来而选择了一条幸运搜索路线，并不能被自动检测为信息泄漏。通过前缀检查只能降低这个风险；独立新任务评估才检查这种监督是否可学。

### 8.4 监督位置

默认 repair-mask 在以下决策位置计算 loss：

- insert 产生的新动作；
- replace 产生的动作；
- delete 之后，在“删除位置的正确前缀”上输出的第一个保留动作；
- 如果删除发生在尾部且环境已经成功，不创造不存在的终止动作 target。

其余正确历史作为 context。多个 edit blocks 保存在同一个 LearningEpisode 中，后一个修复点必须使用前一个修改已经执行后的真实历史。

**不能只把错误动作 mask 掉而不给出替代行为；不能把第二处修改拼回包含第一处旧错误的历史。**

动作对齐通过动态规划计算。代价相同时优先：保留相等动作、替换、删除、插入；保存对齐路径，确保 mask 可复现。

### 8.5 两种训练视图

| 视图 | 监督范围 | 用途 |
|---|---|---|
| Full-trajectory | 成功修复轨迹上的全部 assistant decisions | 第一阶段比较“完整重写 vs 联合修复”，隔离目标构造差异 |
| Repair-mask，完整 pipeline 默认 | 8.4 指定的决策 | 检查局部监督是否降低无关改写并提高监督利用率 |

两者都不监督 environment observation、user message 或 system prompt。

### 8.6 Teacher 标签预算

每条失败案例的默认上限：

~~~yaml
teacher_per_failure:
  diagnosis_calls: 1
  proposal_calls: 2
  rationale_calls: 50          # 实际不超过所选轨迹需要重建的步数
  format_retry_calls: 1       # 整个案例共用，不是每个调用各一次
  total_calls_cap: 54
  total_input_token_cap: 250000
  total_output_token_cap: 24000
  diagnosis_output_token_cap: 2048
  proposal_output_token_cap: 6144
  rationale_output_token_cap: 128  # 包括短 reasoning 与证据字段
~~~

- 预算同时受 calls、input tokens、output tokens 三项约束；先到者停止。
- 只对最终选中修复重建 reasoning，不为所有未入选候选生成完整标签。
- 若预算不足以完成一致的 LearningEpisode，保留修复诊断结果，但标记 LABEL_BUDGET_EXHAUSTED，不用半成品伪装成完整监督。
- 原子性的完整 episode 可以产生多个决策训练行；训练权重仍按 episode 归一化，不能让长轨迹因行数更多被重复加权。

## 9. 每轮步骤 E：冻结 checkpoint 上的梯度分组

本模块在学生本轮更新**之前**执行。它组织已经有合法监督的修复案例，不把梯度当成额外标签来源。

### 9.1 梯度定义

对修复 episode i，用与第 11 节一致的 loss：

\[
g_i=\nabla_{\phi}\ell_i(\theta_r),
\]

φ 为本次实际更新的参数，继承现有训练的参数化；若使用 LoRA，φ 就是实际 LoRA 参数。不能从 frozen base 参数计算梯度，却把它当成 LoRA 的真实更新方向。全参数训练下允许流式投影完整梯度；若只用最后几层，必须标记为梯度子空间 variant。

- 一条 joint repair 是一个梯度单位。
- dropout 关闭，使用同一 θ_r、同一 token mask、同一 loss normalization。
- 梯度计算后不执行 optimizer.step；清理累积梯度。
- 原始范数小于 10⁻¹² 的案例标记 ZERO_GRADIENT，不参与方向聚类，但仍可作为训练样本。
- LoRA 某些参数初始梯度为零是可能的；判断整个拼接梯度，不据单个矩阵的零梯度丢弃案例。

### 9.2 默认压缩与聚类

1. 按参数完整名称排序、按元素索引展平，保存该顺序的 hash。
2. 用固定 seed = 17 的 CountSketch 投影到 2,048 维：每个参数坐标被确定性映射到一个桶和一个正负号，桶内求和。
3. 保存原始梯度范数；投影向量做 L2 normalization。
4. 距离为 1 − cosine similarity。
5. 有效样本 N < 4 时只建一个组；否则组数为 min(4, floor(N / 2))。
6. 使用确定性 k-medoids：首个 medoid 为原始范数最大的案例；之后依次选与已有 medoids 最近距离最大的案例。
7. 每轮先分配到最近 medoid，再在组内选总距离最小的成员为 medoid，最多 20 轮；所有平局按 episode_id。
8. 最后将少于 2 个成员的小组并入最近 medoid 的其他组，直到不再存在小组或只剩一组。
9. 每组向 teacher 提供 medoid 及距 medoid 最远的另一个案例，最多 2 个；不把原始梯度向量塞进 prompt。

梯度相近的含义是：在当前参数化和 loss 下，局部更新方向较相近。它不保证这些案例具有同一个语义技能或共同错误原因。

### 9.3 语义条件与梯度分工

Teacher 从组内案例提取：

~~~yaml
skill_spec:
  supporting_repair_ids: [r01, r08]
  required_public_conditions:
    - "什么可见条件下需要这个决策"
  corrected_decision:
    - "需要采取的行为类别"
  preserved_student_choices:
    - "哪些路线选择可以保持"
  applicability_boundary:
    - "改变什么条件后，原行为可能也成立"
  confidence_status: supported_by_examples_or_unverified
~~~

只有有具体案例支持的条件才能驱动默认练习生成。组内语义不一致时拆成多个 skill_spec，但保留同一 gradient_cluster_id，不能捏造共同技能。

### 9.4 可替换方案

| 选项 | 定义 | 比较意义 |
|---|---|---|
| G0 | 不分组，随机选择修复案例 | 必备基线 |
| G1 | 按预先定义的 requirement/action 类型分组 | 排除仅靠语义分组即可解释的收益 |
| G2，默认完整版 | 上述当前参数梯度分组 | 检查当前学生的更新结构是否有用 |
| G3 | 用冻结的 Adam 二阶矩作对角预条件后再投影 | 检查优化器状态是否影响分组 |
| G4 | 针对训练侧保留的目标损失计算 influence-style 分数 | 后续方案，不能使用测试梯度 |

G3 使用 g / (sqrt(v̂) + ε)，v̂ 固定为本轮开始时的二阶矩；新 optimizer 状态尚为空时退化为 G2。该近似忽略后续动量与参数变化，不称为精确 Adam 更新收益。LESS 可作为优化器相关梯度选择的参考，而不是本方法有效性的证明。[R4]

## 10. 每轮步骤 F：从修复条件得到新练习

### 10.1 可执行任务的三种来源

| 模式 | 实现 | 默认使用位置 |
|---|---|---|
| T0 | 仅重试原 K 个任务 | 第一阶段 P0 |
| T1 | 在 U_train 中按 skill_spec 检索合法、未用过的任务 | 完整 v0.1 的可运行默认 |
| T2 | 根据结构化规格修改任务／初始状态，经环境编译器生成实际游戏实例 | 在 task factory 通过验收后开启 |

T1 是“取得新的针对性练习”，不能在论文里写成“生成了全新的环境”。T2 才是实际环境任务生成。T2 接口未实现时，pipeline 仍可运行 T1，但不能声称已经验证了条件翻转生成机制。

T2 需要独立实现：

~~~python
class TaskFactory:
    def propose_specs(self, skill_spec, parent_tasks, seed) -> list[TaskSpec]: ...
    def compile(self, spec) -> ExecutableTask: ...
    def validate_instruction_goal_match(self, task) -> ValidationRecord: ...
~~~

这些函数操作训练侧的结构化任务规格。自然语言 instruction 必须从目标谓词／受控模板渲染，不能先随意写 instruction 再假设 reward 与它一致。

### 10.2 三类练习

| 类型 | 改什么 | 保持什么 | 需要验证什么 |
|---|---|---|---|
| C1 条件保持 | 物体、容器或场景实例 | 同一必要条件与决策关系 | 对应要求仍存在；任务可执行 |
| C2 条件翻转 | 一个明确的目标／状态条件 | 尽可能多的其他条件 | 旧行为何时成功，修复行为何时必要 |
| C3 条件组合 | 同时引入两个已验证修复条件 | 每个条件含义 | 需要联合处理，且目标并非不可达 |

默认完整 v0.1 的 T1 先实现 C1。T2 可用后优先做 C2；C3 等 C1/C2 有正向证据后再加，不把三种生成一起加入首轮机制实验。

### 10.3 C2 的严格判定

设 A 为一条基准动作序列，δ 为待研究的一个修复 block；其他已知必要修改保持固定。

构造 x⁺、x⁻ 两个任务，仅允许改变预先声明的一项条件，验证：

\[
V(x^+,A)=0,\quad V(x^+,A+\delta)=1,\quad V(x^-,A)=1.
\]

还要运行 V(x⁻, A+δ)：

- 若也成功，δ 在 x⁻ 上是“可选”，不是错误动作；
- 若失败，可以记录“该修改在 x⁻ 上不合适”，但需要解释执行层面的失败；
- 不因不需要 clean 就把 clean 标为负例；它可能只是多余但有效。

例如“必须清洁”与“无清洁要求”可以构成适用条件的差异，但不能据此强行训练学生在后者拒绝清洁。

多个错误同时存在时，不能只改变其中一个任务要求，却期待原始整条失败轨迹自动成功。上面的 A 应先固定其他必要修复，以隔离被研究的条件。

### 10.4 默认练习分配

每轮至多 4 个梯度组，每组最多接受 2 个新任务，全轮最多 8 个：

1. 各组先分配相同额度，避免最常见错误吞掉全部预算。
2. T1 中，每组按固定 seed 排序其 skill_spec 匹配任务，最多检查 6 个候选。
3. 任务匹配使用公开目标类型、目标物体类型和已记录的修复 requirement 标签；未实现的语义条件不能假装可以由 metadata 精确检索。
4. T2 中每组最多提出 6 个 TaskSpec；编译与验证失败的候选同样计入生成预算。
5. 对候选先让冻结 θ_r 独立运行 1 次。
6. 学生成功时获得可行性 witness，保存自己的成功轨迹。
7. 学生失败时走同一个修复流程；本轮新任务最多修复 8 条，预算独立于原任务的 20 条。
8. 成功 witness 可以来自学生或实际执行成功的 teacher patch；不接受 teacher 文字宣称可行。
9. 未拿到 witness 的合法任务标记 UNSOLVED_WITHIN_BUDGET，保存到困难任务记录，不把它说成非法任务。
10. 若某组没有足够验证通过的任务，保留未用额度；不静默补入随机简单题。

接纳条件：合法 split、非重复、instruction/goal 一致、实际环境可执行、有成功 witness、来源可追溯。所有入选前后的失败数量都必须报告，以便识别课程是否只在选择容易题。

### 10.5 何时计算新题的梯度

梯度无法为没有目标的任务凭空提供正确标签。

默认本轮用已修复原题的梯度指导选题；新题先经过学生执行／teacher 修复取得监督，其梯度在下一轮重新计算时才进入库。若想在本轮用新题梯度筛选，必须先支付这些题的标注成本，并单独记录这个 variant。

## 11. 每轮步骤 G：学生更新

### 11.1 单 episode 的 loss

一个 episode 的监督决策集合为 M_i。对 reason-action 格式：

\[
\ell_i(\theta)=\frac{1}{|M_i|}\sum_{t\in M_i}
\left[0.25\,\overline{\mathrm{CE}}(r_t\mid h_t)
+0.75\,\overline{\mathrm{CE}}(a_t\mid h_t,r_t)\right].
\]

- 两项分别按其 token 数取平均，避免长 reasoning 自动压过短 action。
- 没有 reasoning 通道时，action 项权重为 1。
- 当前 reasoning 是被学习的 target，而不是在推理时额外提供的正确答案提示。
- source、episode、decision、token 四级归一化都必须按此定义实现；不能直接对 packed token 求总平均而宣称等价。
- M_i 为空的 episode 仅作诊断，不进入训练。

### 11.2 四个互斥数据桶

| 桶 | 内容 | 默认权重 |
|---|---|---:|
| R | 原 K 个任务上的成功修复 | 0.50 |
| G | 追加任务的成功学生轨迹或成功修复 | 0.25 |
| S | 原 K 个任务上的学生成功轨迹 | 0.15 |
| D | 原始 teacher seed 示范 | 0.10 |

生成任务上的成功轨迹只进入 G，不再进入 S；生成任务上的修复只进入 G，不再进入 R。

\[
\mathcal L(\theta)=
\sum_{b\in\{R,G,S,D\}}p_b\,
\mathbb E_{\text{task}\sim U(b)}
\mathbb E_{\text{episode}\sim U(b,\text{task})}\ell_i(\theta).
\]

- 非空桶之间按比例重归一化权重。D 总是保留。
- 先均匀选 task，再选该 task 的 episode，避免多次失败的同一任务支配 loss。
- R/G 中 80% 概率抽本轮数据，20% 抽历史数据；其中一方为空时使用另一方。
- 历史缓存每任务最多 2 条，按最近轮次保留；同轮平局优先较少修改，之后按 ID。
- G/S 中成功学生轨迹用 full-trajectory loss；R/G 中修复轨迹默认用 repair-mask。
- 不对未验证的失败动作做 unlikelihood 或负偏好训练。一个失败轨迹中的动作可能仍是合法好动作。

### 11.3 默认优化配置

默认直接继承已有 Gemma4-E4B-it SFT 的训练配置和 loss 实现，两臂仅替换 target 数据。第 11.1 节是完整版的明确 loss 选项；快速机制实验先沿用现有 loss，不同时改变归一化。下面仅是在没有既定配置时可选择的后备配方，不能覆盖已有 baseline：

~~~yaml
optimization:
  method: lora
  frozen_base: shared_sft_k_checkpoint
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.0
  target: all_attention_query_key_value_output_linear_projections
  train_embeddings: false
  train_lm_head: false
  optimizer: AdamW
  learning_rate: 0.00002
  betas: [0.9, 0.999]
  epsilon: 0.00000001
  weight_decay: 0.0
  max_grad_norm: 1.0
  precision: bf16
  optimizer_updates_per_round: 50
  effective_batch_size_episodes: 8
  learning_rate_schedule: constant
  warmup_steps: 0
  packing: false
  keep_optimizer_state_between_rounds: true
~~~

注意：

- target 是模型适配器要解析的语义范围。启动时必须导出实际 module name 清单，并断言非空、覆盖所需投影、未包含 embedding/lm_head；不凭模型名称假定模块路径。
- 现有 θ₀ 使用全参数更新就继续使用全参数；使用 LoRA 就复用相同 adapter 范围。两个臂的可训练参数集合和 optimizer 初始状态完全一致，不为了本实验额外添加一层 adapter。
- 如果比较全参数更新 variant，则所有臂一起改；起始学习率 5×10⁻⁶，其余比较条件保持一致。
- 完整版每轮 50 updates 是工程起点，不保证已收敛。快速机制实验使用第 14 节的最多 20 updates；两个臂使用相同固定预算。
- 训练按 episode 归一化，允许通过多个 decision forward 累积一个 episode 的梯度。
- 超长样本不能静默裁掉修复所需历史；记排除率。若需要历史压缩，先作为所有臂共享的独立改动验证。

## 12. 每轮步骤 H：重新尝试、评估与结束

1. 完成一次学生更新得到 θᵣ₊₁。
2. 从初始状态独立评估 D_probe，不能注入 diagnosis、repair patch、teacher 临时提示或外部记忆。
3. 如果继续，冻结 θᵣ₊₁ 后重新执行 rollout → repair → grouping → practice → update。
4. 上一轮梯度全部失效；可缓存原数据，但不能直接复用其旧梯度作为新 checkpoint 的特征。
5. 默认完整实验最多 3 轮，每轮 50 updates。
6. D_probe 连续两次轮间评估都未增加至少 1 个成功任务时可提前停止；此规则仅用于工程选模，不是统计显著性判断。
7. 成本上限先到则停止。最终选择 D_probe 成功数最多的 checkpoint；平局选更早轮次。
8. 第一阶段机制实验只有 1 轮、最多 20 updates，按第 14 节固定记录 0/1/5/20；不选 best checkpoint。
9. “新生成的题全做对”只触发增加难度或降低该类采样比例，**不作为训练结束充分条件**。

需要同时保留三种成功率：

- **修复执行成功率**：teacher patch 由环境执行后的结果；
- **接续成功率**：获得部分修复后 student 能否继续；
- **独立成功率**：训练后的 student 从任务初始状态独立执行。

最终方法收益以第三种为主。

## 13. 可选 variant 总表

每轮实验只改变需要验证的一项，避免一次加入所有选项。

| 部分 | 默认 | 可选方案 | 首次比较时固定什么 |
|---|---|---|---|
| 起点 | seed SFT checkpoint | 原 student、格式适配 checkpoint | 任务、teacher、后续训练配置 |
| 修复范围 | 最多 4 个不连续 blocks | 单 block、完整重写、扩大 block 上限 | 原失败轨迹、搜索预算 |
| 成本 | 动作编辑距离 | 子目标编辑距离、编辑距离 + 路径长度加权和 | 成功判定与候选池 |
| 搜索 | 2 轮 × 3 候选 | 1×1、1×3、更多候选 | 分别报告 token/环境成本 |
| 后缀执行 | 固定动作序列真实重放 | student 自主接续、teacher 闭环执行 | 后两者生成的新动作全部计入新轨迹和成本 |
| 标签 | 前缀一致的 reason-action | action-only、只学修复动作不学新 reasoning | 所有臂统一输入输出格式 |
| loss 位置 | repair-mask | full-trajectory、修复位置加权的全轨迹 loss | 相同已验证目标轨迹 |
| 参数 | 继承已有 baseline | 全参数与 LoRA 的独立比较 | 各实验臂同一种参数化 |
| 分组 | 梯度 G2 | 随机 G0、语义 G1、预条件梯度 G3 | 同候选池、同监督预算 |
| 练习来源 | 合法任务检索 T1 | 固定题 T0、编译生成 T2 | 区分额外训练任务数 |
| 练习关系 | C1 | C2、C3 | 实际环境验证规则 |
| 更新目标扩展 | SFT | 同前缀动作偏好、在线 RL | 在核心修复机制验证后另立实验 |

若添加偏好训练，正负动作必须在**同一个有效历史前缀**下有可比较的环境证据；不能把不同状态下的两个动作直接配成 DPO pair。若添加 RL，环境奖励和 teacher 修复信号的用途须另写目标，不把它当成无需说明的替换。


## 14. 第一步机制验证：用直接干预检验因果链

本节是优先执行的方案，覆盖前文关于“第一阶段”的一般描述。

**固定模型：teacher = GPT-6-luna；student = Gemma4-E4B-it。** 优先使用已有的对应 K 的 seed-SFT checkpoint θ₀；原训练题没有足够失败时，按 14.0 的预注册规则使用已有 K=0 checkpoint 验证学习机制。

第一步不是比较“完整 pipeline 是否在榜单上更高”，而是检验下面这条链：

\[
\text{修改特定决策}
\longrightarrow
\text{环境结果由失败变成功}
\]

\[
\text{以这种修复作为监督}
\longrightarrow
\text{学生在自己的失败状态上改变决策}
\longrightarrow
\text{学生脱离 teacher 后更容易完成任务}.
\]

三个箭头要分别有观测。只有最后一张成功率表，无法说明中间机制；只有 target loss 下降，也无法说明学生学会执行。

### 14.0 复用已有 baseline：在现有实验上做增量验证

本节不要求重建 ALFWorld harness 或重跑已有完整 baseline。先整理现有实验资产，再添加修复模块。

**已有结果的用途分成三层：**

| 层次 | 复用内容 | 用途 |
|---|---|---|
| 机制起点 | Gemma4-E4B-it 的 K=0、对应 K 的 SFT checkpoints 及训练任务 rollouts | 找到 student 当前实际失败的状态，建立 E1/E2 |
| 效果参照 | 同 teacher/student 的 SFT、Kang/AD、SAD、SmartAD 在 K=30、K=100 等配置下的已有结果 | 保留已有坐标系，判断增量收益的量级 |
| 增量确认 | 已有 同一 K 的最强 baseline 的 checkpoint | 检验修复是否能改善已有蒸馏方法，而不仅补足普通 SFT |

本文没有填入未读取的原始分数、checkpoint 路径或具体 commit。它们通过下面的 manifest 从现有实验导入；不重新估计已有曲线数值。

#### K=10/20/30/100 的统一设计

K 表示初始 teacher success demonstrations 数；不表示总监督量。实验网格固定为 {10,20,30,100}。

| K | 在机制实验中的角色 | 默认追加修复训练对 | 修复阶段保留 probe | 初次执行优先级 |
|---:|---|---:|---:|---|
| 10 | 最少示范条件，检验修复是否仍可学 | 最多 4 对 | 2 个任务 | 快速主实验有方向后优先扩展 |
| 20 | 少样本主验证条件 | 最多 4 对 | 2 个任务 | 首选；尚无 checkpoint 时先复用 K=30 |
| 30 | 与现有详细结果直接对应 | 最多 4 对 | 2 个任务 | 复用现有资产跑通，或作重复验证 |
| 100 | 较充分示范参照，检查修复信号和剩余提升空间 | 最多 4 对 | 2 个任务 | 最后确认，不预设一定没有收益 |

这里将快速实验缩为 **4 个训练配对 + 2 个修复 probe**，以便 K=10 也能采用同一规模。独立快速评估仍用所有 K 共用的 24 个任务。数据量较小，主要用于判断是否值得继续；有方向后再扩充 probe 和案例。

执行规则：

1. 第一次只跑一个 K：优先已有 K=20 checkpoint；若它尚未产出，就先跑现有 K=30，不等待全部 baseline 完成。
2. 首次通过后原样扩展到 K=10，再扩展 K=30/100 中尚未跑的条件。不要在首次实验前铺开四个 K 的所有 variants。
3. 同一 K 的 N/F 臂使用相同起点和配对任务；不同 K 的有效修复任务可以不同，但必须报告任务覆盖与难度分布。
4. 跨 K 固定追加 teacher token 上限、候选数上限、环境步数上限及最多 20 次更新；实际使用量另报。不把 4 对 target 数相同当成 token 成本完全相同。
5. 既有 checkpoint 的初始训练成本是已发生的成本；报告增量预算时单列，同时保留累计预算。
6. 若为新建 seed sets，优先固定一个 100 题集合，并构造 10⊂20⊂30⊂100 的嵌套子集。已有实验不是嵌套时保留原集合，不重做全部 baseline；报告 overlap。
7. 24 个独立评估任务及完整 D_probe 必须排除四个 K 的 seed task **并集**，而不仅排除当前 K 的训练题。
8. 四个 K 都使用同一个 GPT-6-luna teacher 配置和 Gemma4-E4B-it student revision。已有数据版本不一致时明确分组，不能拼成单一受控 K 曲线。
9. 不要求增益随 K 单调下降。低 K 可能缺少学习修复所需的基础能力，高 K 可能缺少剩余错误；这都是待测现象。

**失败覆盖不足时，不混合协议：**

- P0 在某个 K 上没有足够训练失败，就报告“当前固定 K 题没有足够修复信号”，而不是填补为机制负结果。
- K=0 checkpoint 的补充实验只验证学习机制，不放进“在 SFT-K 上的增量收益曲线”。
- 如果希望四个 K 都在同一种条件下检验增量效果，可固定一个共同的训练侧任务池 U_mech，默认 20 个合法任务，排除全部 seed/probe/eval IDs。四个 K **一起**使用该池采集失败，统一转为 P1。
- U_mech 不需要新任务生成器，直接复用现有训练环境加载器；每个 K 仍最多使用 4 个修复训练对。任务池中的失败 teacher 查询与 probe 标注均计预算。
- 不能只给 K=100 新任务而让 K=10 仅重试原题，然后把差异解释成 K 的作用。

跨 K 的结果表至少包括：

~~~text
K_seed
starting_baseline
starting_checkpoint_hash
protocol_P0_or_P1
available_failed_training_tasks
accepted_target_pairs
repair_probe_count
additional_teacher_input_output_tokens
additional_environment_steps
optimizer_updates
baseline_reference_success
near_target_success
far_target_success
delta_from_own_starting_checkpoint
repair_probe_action_mass_change
repair_probe_continuation_success_change
~~~

绝对成功率用于对接原 baseline 表；相对各自起点的增益用于比较增量效果。两者同时报告，不把起点更强误认为本次修复学得更多。



#### 必须复用的部分

- 当前 K 个训练 task IDs、teacher demonstrations 和数据预处理；
- 当前 teacher 的实际 model ID，student 的 revision、tokenizer、chat template；
- 环境版本、game files、动作 parser、agent prompt、history 处理；
- 环境步数上限、推理采样参数和 evaluation task IDs；
- 原训练代码、可训练参数范围、optimizer、学习率和 loss normalization；
- 原日志格式和成功率统计脚本。

**快速 E2 中不为了本方法新建 LoRA、不改变全参数／LoRA 选择，也不默认引入第 11.1 节的 reasoning/action 加权。** 两个臂首先直接沿用已有 SFT loss。否则优化器、参数化和 loss 同时变化，无法把收益归因于 target。

快速实验只做两个明确的训练预算覆盖：最多 20 updates、effective batch size = 4；两臂共同采用，不分别调参。其余参数继承原训练设置。

若 checkpoint 未保存 optimizer 状态，可在两个臂统一重建 optimizer；记录这是一次共同重启，不能一边恢复旧状态、一边重新初始化。

#### 先检查训练失败覆盖，再固定起点

现有对应 K 的模型可能已经把当前 K 个训练任务全部学会，虽然在另外约 140 题上仍有明显不足。这两件事不矛盾，也会直接影响本方法能否取得新信号。

按以下规则，在产生 repair targets 前决定起点：

1. 优先使用现有 对应 K 的 SFT checkpoint，并复用同配置下已保存的**训练任务** rollouts。
2. 如果已有训练任务失败覆盖足够，直接进行 E1/E2，不重复收集相同轨迹。
3. 如果不足 6 个不同失败任务，按第 5 节的有限重试规则补充；仍不足则不强行满足原题修复的前提。
4. 此时快速机制实验改用已有 K=0 的 Gemma4-E4B-it checkpoint，在同一组 K 个训练任务上进行；不更换 student，不新增环境任务。
5. 这个分支回答“修复监督是否更可学”，**不**回答“能否继续改善已经收敛的 对应 K 的 SFT checkpoint”。两个结论分开报告。
6. 要继续改善一个在当前 K 个训练任务上已全部成功的 checkpoint，需切换 P1：使用训练侧的额外合法任务／条件变体取得失败。新增任务和监督单独记账，不能从约 140 题评估集直接抽失败回训。

这个检查本身会帮助定位瓶颈：原训练题已经成功而其他题失败时，仅重复原题的 failure-driven loop 可能没有信号，课程覆盖才成为必要条件。

#### 旧 trajectory 与旧 teacher target 如何复用

- 与选中起点 checkpoint、prompt、parser 和环境版本一致的训练 rollout 可以直接复用。
- 另一个 baseline 的失败轨迹可以用于分析错误类型，但不能伪装成当前 student 的 on-policy rollout。
- 现有约 140 题的逐任务日志可用于一次性的错误分布分析和结果对照；不把具体评估实例的轨迹、teacher 修复或生成规格回灌训练。
- 如果 E2 从 K=0 起点开始，旧 teacher success trajectory 尚未用于该起点的本实验 seed-SFT，可以作为 F 候选；仍需通过重放和配对检查。
- 如果 E2 从 SFT-K 起点开始，原 teacher target 已被该模型训练过。它可以作为“继续原 SFT”的对照，但不能成为唯一的 Far target；否则比较的是新监督与旧监督重复，而不只是路线距离。
- 因此 SFT 起点的主 E2-F 使用新的、实际成功的替代路线，并检查与原训练 target 不完全重复；旧 teacher 数据继续用于 replay/control。

#### 接到现有代码时只新增的模块

1. 失败记录到 DiagnosisRecord/PatchRecord 的转换；
2. patch 编译与原环境 replay；
3. 成功 target 与 existing trainer 格式之间的适配；
4. 原失败前缀上的决策／接续 probe；
5. 成本与版本 manifest。

不新增第二套动作解析器、第二套成功判定或第二套 leaderboard 统计。

#### 建议的导入 manifest

~~~yaml
existing_experiment_assets:
  teacher_display_name: GPT-6-luna
  student_display_name: Gemma4-E4B-it
  K_grid: [10, 20, 30, 100]
  first_K: 20_if_available_else_existing_30
  primary_candidate: existing_sft_selected_k
  coverage_fallback: existing_student_k0
  confirmation_candidate: strongest_existing_baseline_at_selected_k
  checkpoint_selection_time: before_new_method_results
  seed_task_manifest: reuse_existing
  teacher_demo_manifest: reuse_existing
  train_rollouts: reuse_if_exact_checkpoint_and_harness_match
  optimizer_state: restore_for_both_or_reset_for_both
  update_parameterization: inherit_existing
  training_loss: inherit_existing_sft
  evaluation_harness: inherit_existing
  existing_results: import_exact_values_without_reconstruction
~~~

在原有结果表旁增加这些列，而不是另做无法对应的汇总：

| 起点 | 原方法分数 | +20 updates 的 N 分支 | +20 updates 的 F 分支 | 额外 teacher tokens | 额外环境 steps | 新增训练任务数 |
|---|---|---|---|---|---|---|
| 对应 K 的 SFT，失败覆盖足够时 | 复用已验证记录 | 新测 | 新测 | 实测 | 实测 | P0 为 0 |
| K=0，覆盖不足的机制分支 | 复用已验证记录 | 新测 | 新测 | 实测 | 实测 | P0 为 0 |
| 同一 K 的现有最强 baseline，后续确认 | 复用已验证记录 | 新测 | 新测／继续训练对照 | 实测 | 实测 | 按 P0/P1 标注 |

只有 checkpoint 和评估配置完全一致时才复用原分数；否则只补跑必要的对齐评估，不重训所有 baseline。

已有 K=100 结果可以作为实用参照：“从 30 个种子出发，增加多少反馈后接近了 100 示范的水平”。因为额外监督和交互不同，不能直接把这种接近写成“只用 30 个总监督样本达到 100 个的效果”。



### 14.1 第一性原理：本方法究竟假设了什么

对一次小参数更新 δθ，某个有效行为监督的损失近似变化为：

\[
\ell_{\mathrm{probe}}(\theta+\delta\theta)
-\ell_{\mathrm{probe}}(\theta)
\approx
\nabla_\theta\ell_{\mathrm{probe}}(\theta)^\top\delta\theta.
\]

因此“改得少”本身没有充分理论保证。真正需要发生的是：

1. 修复确实改变了导致任务失败的执行结果；
2. 由这些修复产生的更新，能提高学生在实际遇到的历史下执行有效行为的概率；
3. 这种变化能够影响无辅助的任务完成，而不仅是更容易复述训练 target。

本实验直接检查 1–3。梯度聚类不参与第一阶段；上式用于说明可检验的机制，不要求先实现 influence function 或梯度库。

### 14.2 两个顺序执行的小实验

| 实验 | 是否训练 | 核心干预 | 要回答的问题 |
|---|---|---|---|
| E1：执行干预 | 否 | 加入、撤销、恢复修复 blocks | 修改是否真的使原任务成立；是否确实需要联合修改 |
| E2：学习干预 | 两个臂，各最多 20 updates | 同起点分别学习“近学生路线”与“有效替代路线” | 近学生路线的监督是否更快改变学生决策并提高独立完成率 |

E1 失败就不运行 E2。E2 有清楚方向后，才增加 repair-mask、普通 SFT、完整课程等后续对照。

### 14.3 共享数据：先固定，再看修复结果

1. 在当前 K 个训练任务上，让 θ₀ 按第 5 节运行。
2. 依照预先固定的分层和 seed，选择最多 20 个**不同任务**的自然失败，每任务取 1 条。不要人工破坏成功轨迹制造失败。
3. 在 teacher 产生标签前，从这批任务中固定 2 个作为 D_repair_probe，其余为候选训练任务。
4. D_repair_probe 的任务在 θ₀ 的原 seed-SFT 中可能出现过；它们只是在**本次修复更新阶段保留**，不能称为全新未见任务。
5. 候选训练任务中，最多接纳 4 个满足 14.6 配对条件的任务，构成 D_repair_train。
6. 从既有 D_probe 中，预先固定 24 个独立任务作为 D_unseen_quick；它们不送 teacher，不用于调 prompt、挑 targets 或训练。
7. D_repair_probe 与 D_repair_train 在任务 ID 上不相交；D_unseen_quick 与全部 K 设置的 seed task 并集不相交。
8. 缺少 2 个有效 probe 或 4 个合格训练对时，报告实际数量和缺失原因；不通过评估结果重新分配 probe，也不无限请求 teacher 凑样本。

D_repair_probe 用于检测“会不会处理另一个学生失败状态”；D_unseen_quick 用于检测“会不会从初始状态完成另一个任务”。两者不能互相替代。

### 14.4 E1：不更新 student，验证修复本身

对每条原失败轨迹 τ：

1. 从同一个初始状态重放 τ，确认失败。
2. teacher 按第 6–7 节产生联合修复 Δ。
3. 从同一个初始状态执行 τ+Δ，读取环境是否成功。
4. 对每个 block δ_j，执行 τ+(Δ\{δ_j})。
5. 完成撤销检查后，恢复完整 patch 并重复确认成功版本一次；所有检查受第 7.3 节的 24 次总 replay 预算约束，未完成的检查记 unknown。
6. 所有执行使用同一个任务、初始状态、H 和环境版本；不修改目标来制造成功。

对一个两 block 案例，最低证据表为：

| 执行条件 | 必须由环境取得的结果 | 含义 |
|---|---|---|
| 原轨迹 τ | 失败 | 确实有待修复的失败 |
| 只有 δ₁ | 实际测量 | 不预设单处修改足够 |
| 只有 δ₂ | 实际测量 | 不预设单处修改足够 |
| δ₁ + δ₂ | 成功 | 联合修复足以完成原任务 |
| 撤销再恢复后的 δ₁ + δ₂ | 成功 | 排除偶然执行／恢复错误 |

如果两项单独都失败、联合成功，就有该具体案例的联合修复证据。它不证明其他任何单 block 方案都不可能成功。

**E1 的直接回答：** 哪个实际修改集合使固定任务从失败变成成功。教师解释是否漂亮不进入判定。

E1 输出同时保留原路线中被执行成功的原动作。这样可以展示“原来的某些选择能保留”，而不是只展示新答案正确。

### 14.5 E1 的通过条件与范围

进入 E2 的最低工程条件：

- 至少 4 个候选训练任务与 2 个 probe 有环境验证的成功修复；若不足，可做更小的探索性实验，但不补造案例。
- 所有入选修复都满足原目标、原初始状态、相同 H。
- 至少出现 2 个来自不同任务的多 block 联合必要性案例，才在这次试验中检验“联合修复”部分的主张。
- 如果全是单点错误，仍可检验路径保留，但明确本次没有验证联合性。
- 在所有被检查的原失败上报告修复成功率，不能只报告入选的 6 个成功例子。

单个确定性干预表能够直接验证该案例。需要多少案例来支持“普遍有效”是后续规模化问题，第一步不把两者混为一谈。

### 14.6 E2 的关键对照：同题、同正确性，不同目标路线

每个候选训练任务构造两个**实际执行成功**的 target：

| Target | 定义 |
|---|---|
| N：Near | 按第 7 节选出的联合最小修复，尽量保留学生原来的有效路线 |
| F：Far | teacher 提供的一条有效替代路线，允许改变原来有效的选择，但仍修复相同任务要求 |

为了让比较回答“目标路线接近性”，而不是“一个 target 更短或更容易”，配对需满足：

1. 同任务、同初始状态、同成功标准、同 H。
2. 两条 target 都有实际成功 witness；配对审核排除包含环境明确拒绝动作的 target。只读检查或合法探索没有改变物理状态，不因此算非法。环境没有可靠错误标记时，对这几个配对人工核对公开反馈并保存判断；不能把无效动作率差异隐藏在“路线距离”中。
3. 完成目标所用的对象实例相同，必要的任务要求相同。
4. 两条路线都有同样的目标相关操作类型，例如该洗就洗、该放入指定容器就放入；不能给 F 一个缺少必要操作的坏 target。
5. 环境动作长度比在 [0.8, 1.25]。
6. 经相同标签构造后的 supervised target token 数之比在 [0.8, 1.25]。
7. F 相对原学生轨迹的动作编辑距离，至少比 N 大 max(2, ceil(0.15 × max(L_N, L_F)))，L 是成功 target 的动作长度。使用 target 长度设置最小差距，避免原失败中的很长循环后缀放大阈值。
8. F 是有效的替代计划，不通过填充重复话语、故意绕圈或添加无意义动作来拉远距离。
9. 所有 reasoning 标签均按相同的前缀一致性规则重建；不让 N 使用长详细讲解而 F 只有简短答案。

目标不是找到“最差的 Far”，而是得到两个质量与规模接近、路线距离确实不同的正确目标。

N、F 每个分支各遵守 2×3 候选搜索上限，合计最多 12 个候选；可共用 diagnosis，但两分支成本分别记录。旧 target 可复用时按 14.0 处理。未找到合格配对则记 PAIR_NOT_FOUND；保留 E1 结果，不无限生成。

这里干预的是 target 路线构造规则，编辑距离是量化指标；即使完成长度匹配，也不声称只改变了一个独立的数值 d 而所有语义因素都不变。

**若大多数自然失败无法得到上述配对，就不应把普通 N vs F 的差异严格归因于距离。** 此时可以报告方法整体收益，但机制证据应降级为“多个因素共同改变”。

### 14.7 配对选择的确定规则

在所有环境成功的候选中：

1. 先固定 N 为第 7.1 节定义的 best observed repair，不为了满足配对而更换一个更差的 N。
2. 若 N 不满足 14.6 的基本质量要求，整题退出机制配对，不换一个更有利的 N；其原始修复成功结果仍报告。再从 F 候选中筛选满足 14.6 的方案。
3. 若有多个 F，选与 N 的动作长度比例最接近 1 的；再选 supervised token 数最接近的；再按 ID。
4. 对照臂使用完全相同的合格任务清单。
5. 按预先固定的任务顺序取前 4 对，不按更新后的表现挑选。
6. 记录每条失败因覆盖、长度、token 数、距离或标签质量被排除的原因。

这是一项控制变量的机制实验，其结论首先适用于这些合格配对。之后再用不筛配对的完整失败池检验实用覆盖率。

### 14.8 两个短训练臂

从同一个 θ₀、相同可训练参数和相同 optimizer 状态开始：

| 臂 | 修复目标 | Loss |
|---|---|---|
| E2-N | D_repair_train 的 N targets | 全轨迹 assistant-decision SFT |
| E2-F | 同一任务的 F targets | 完全相同的全轨迹 assistant-decision SFT |

固定配置：

~~~yaml
quick_mechanism_training:
  teacher: GPT-6-luna
  student: Gemma4-E4B-it
  training_task_pairs_cap: 4
  repair_probe_tasks: 2
  independent_quick_eval_tasks: 24
  optimizer_updates_cap: 20
  checkpoints: [0, 1, 5, 20]
  effective_batch_size_episodes: 4
  repair_weight: 0.75
  seed_replay_weight: 0.25
  seed_replay_task_scope: only_D_repair_train_tasks
  loss_mask: full_trajectory
  optimizer_and_parameterization: inherit_existing_sft
  loss_normalization: inherit_existing_sft
  training_seed: 1
  checkpoint_selection: none_report_all_fixed_points
~~~

只重放 D_repair_train 对应任务的 seed 示范；不在这次更新中重放 D_repair_probe 的 seed 示范。两臂使用相同的 task/bucket 采样顺序、相同优化器和同样的 20 updates。

不先用 repair-mask：否则目标距离和监督位置一起变化，无法直接回答机制。也不把三个 seed、多个学生和完整课程一起铺开。

### 14.9 直接观测 1：在学生失败状态上的决策变化

对每个 D_repair_probe 案例：

1. 取原学生轨迹第一次需要修改的动作之前的公开历史 h_i。
2. 这是原学生真实到达的历史，不是任意构造的 teacher 专属状态。
3. 仅从保持 h_i 之前所有动作不变、且从该前缀实际执行成功的候选中，汇总下一动作形成 A_i⁺。更早已经改过状态的候选不能把自己的下一动作加入这个集合。
4. A_i⁺ 仅表示“已有成功 witness 的动作集合”，不表示穷尽所有正确动作。
5. 对 checkpoint 0、1、5、20，在相同 h_i 上采样 8 个 assistant turns；temperature = 0.7、top_p = 0.95，使用相同 seed 清单。
6. 每个 turn 自行生成 reasoning 和 action，不提供正确 reasoning。
7. 规范化动作后，计算落入 A_i⁺ 的比例。

\[
\widehat m_i(\theta)=\frac{1}{8}
\sum_{k=1}^{8}\mathbf 1[a_{ik}\in A_i^+].
\]

比较：

\[
\Delta m_N-\Delta m_F
=
[m(\theta_N)-m(\theta_0)]
-
[m(\theta_F)-m(\theta_0)].
\]

这是一个对有效行为概率质量的直接但有覆盖限制的观察。未命中 A_i⁺ 的动作不能自动判错，因为学生可能找到另一个有效路线。

可另外记录固定有效 decision target 的 NLL 以观察微小变化；NLL 只作辅助，不能代替上面的真实生成或下面的环境完成。

### 14.10 直接观测 2：改变的决策能否带来完成

对 14.9 每个 probe、每个 checkpoint 的前 2 个固定采样 seed：

1. reset 原任务并真实重放到 h_i；
2. 从该点开始让当前 student 自主执行到结束；
3. 保持原任务总预算 H，重放过的前缀动作占用该预算；
4. 不插入后续修复 blocks，不提供 teacher 讲解；
5. 记录最终 success 与新失败位置。

这样可以区分：

- 更常输出某个被教过的动作，但后续仍失败；
- 学会了该决策，并且能够与后续行为配合完成任务；
- 走了另一条 A_i⁺ 未覆盖的成功路线。

该指标叫“原失败状态上的无辅助接续成功率”，不能与从初始状态独立完成混算。

对存在多个必要 blocks 的案例，尤其检查学生是否在执行第一处修复后，还能自行处理后面的问题。只有在后续也实际完成时，才有联合行为被学会的执行证据。

### 14.11 直接观测 3：从新任务初始状态独立完成

只在 checkpoint 0 和两个臂的 checkpoint 20 上运行 D_unseen_quick：

- 24 个固定任务；
- greedy；
- 相同 H；
- 无 teacher、无修复前缀、无 diagnosis；
- 保存逐任务配对结果。

这一步排除“只有给定一个正确前缀才能表现好”。24 题只用于快速方向判断；出现方向后再在完整 60 题 D_probe 上确认，不把 24 题上的 1–2 个任务差异当成定论。

### 14.12 哪些结果正面支持哪个机制

| 结果模式 | 能支持什么 | 不能支持什么 |
|---|---|---|
| 原轨迹失败，联合 patch 成功，撤销必要 blocks 后失败 | 该具体修改集合对固定任务的执行结果有作用 | 不能证明所有别的修复都无效 |
| N 相对 F 在 repair-probe 上更快增加有效动作命中，并更快提高无辅助接续成功 | 近学生路线的监督更快影响了学生真实失败状态上的有效行为 | 还不能单凭这个说新任务独立能力更好 |
| 上述方向成立，且独立新任务也改善 | 支持完整的“修复监督 → 行为改变 → 独立执行”链 | 仍需后续更多任务和重复来确定普遍性 |
| 只在 D_repair_train 上变好，repair-probe 不变 | 训练记忆或局部拟合 | 不支持修复技能迁移 |
| 有效动作命中提高，但接续仍失败 | 学会局部选择，尚未学会完整执行 | 不支持联合能力已经学会 |
| N/F 都提升，但速度与终点接近 | 两种正确监督都有效 | 不支持“更接近学生路线更有利”的区别性假设 |
| F 更快或更好 | 当前数据直接反对近路线优势假设 | 不应通过增加模块掩盖此结果 |
| N 只在不匹配长度／信息量的比较中占优 | 方法可能有实用价值，但机制混杂 | 不能把差异唯一归因于目标路线距离 |
| A_i⁺ 命中没增加，但出现更多其他成功路线 | 学生可能学会了集合外的有效行为 | 不能根据动作匹配率宣布失败 |

第一阶段不需要设定一个方便达标的显著性门槛。判断时同时看：固定 checkpoints 上的变化方向、逐案例是否一致、环境完成是否跟着变化、两个 target 是否真的控制住了差异。

“机制不确定”也是允许的结果；实验应当能让当前假设被反驳。

### 14.13 快速执行顺序与预计工作量

| 顺序 | 内容 | 默认上限 |
|---|---|---|
| 1 | 确认已有 E4B 的 θ₀、当前 K 题与 replay 接口 | 复用已有资产，不重新大规模训练 |
| 2 | 收集自然失败并冻结 split | 最多 20 个任务 |
| 3 | E1 执行干预与 target 配对 | 最多 4 训练对 + 2 repair probes；候选预算固定 |
| 4 | 两个臂的小步更新 | 合计最多 40 optimizer updates |
| 5 | 决策分布与接续检查 | 2 probes，固定 checkpoints 与 seed |
| 6 | 新任务独立执行 | 共享 θ₀ + 两个最终 checkpoint，各 24 题 |
| 7 | 写机制判断 | 按 14.12 对应到明确结论 |

通常主要等待来自 teacher 标签和环境 rollout；不预先承诺具体分钟数。该计划不依赖任务生成器、梯度聚类、奖励模型、完整 SCoRe 复现或大规模训练。

checkpoint 0 的评估两个臂共享，只算一次。中间 checkpoints 只做小 probe；独立 24 题只跑起点与终点，避免在每个训练步重跑完整评估。

### 14.14 有正向证据后再做的确认实验

依次增加，不全部同时开展：

1. 用另一个训练 seed 复核 E2-N/E2-F 的方向；随后从已有 同一 K 的最强 baseline checkpoint 做一次增量确认。若其原题已无失败，明确转为 P1 训练侧新增任务协议，不能拿评估集失败回训。
2. 在相同 N targets 上加入 repair-mask，对比 full-trajectory loss，检验监督位置是否有额外作用。
3. 加入继续 seed-SFT + 公共学生成功池的对照，回答追加交互监督相对单纯延长训练的收益。
4. 在不筛近／远匹配对的全部失败池上跑完整 J/F，报告覆盖率、实际费用与独立成功率，检验实用性。
5. 扩展到完整 D_probe 和已有约 140 题的 reference 评估。
6. 再进入第 15 节的条件课程与梯度分组。

这里把“控制变量的机制实验”和“真实使用条件下的效果实验”分开：前者用于识别原因，后者用于确认方法值得使用。

### 14.15 第一阶段最终交付

- 逐案例的 patch 加入／撤销／恢复结果表；
- N/F 配对是否满足同题、同对象、同正确性、相近长度及不同编辑距离的审核表；
- checkpoint 0/1/5/20 的 repair-probe 有效动作命中与接续成功；
- checkpoint 0/20 的独立新任务结果；
- 所有排除案例、失败候选、教师 token 与环境步数；
- 一段能被数据支持的判断：执行机制成立到哪一层、学习机制成立到哪一层、还缺哪一个箭头。

**第一步最关键的正面证据不是“曲线更高”，而是：同样正确、规模相近的两种监督中，保留学生路线的监督更早改变了学生在自身失败状态下的行为，而这个行为改变随后转化成真实完成。**



## 15. 核心成立后的验证顺序

### 15.1 第二阶段：练习是否传递条件

固定已经有效的修复与 loss 方案，比较：

| 臂 | 新练习 |
|---|---|
| C0 | 不加新题，只重采样已修复题 |
| C_random | 随机合法新任务 |
| C_preserve | C1 条件保持任务 |
| C_boundary | C1 加环境验证的 C2 条件对 |

C_random、C_preserve、C_boundary 使用相同新任务数上限、teacher token 上限和更新步数；记录实际支出。C0 用于检验增加任务本身的收益，不假装它消耗了相同的 teacher 信息。

优先一次增加 8 个任务，只跑 1 轮。C2 的两侧各算一个任务，不能用 8 对与另一个臂的 8 个任务直接比较。

### 15.2 第三阶段：梯度有没有独立贡献

先固定同一批已修复案例和共同候选任务池，再比较：

- G0：随机选代表案例／分配练习；
- G1：按修复语义条件分组；
- G2：按当前 student 梯度分组。

所有臂使用相同任务接纳规则、相同候选池、相同每组／全轮预算和相同训练目标。若任一臂为了计算候选梯度取得了额外标签，这笔标注计入预算，其他臂需获得同等可用数据预算。

只有 G2 在相同预算下改善迁移或成本，才保留“梯度驱动课程”作为贡献。否则采用更简单的 G1，不把聚类图漂亮当成有效性证据。

### 15.3 第四阶段：完整闭环

最后再开 3 轮，并检查：

- 新一轮是否确实出现不同错误；
- 旧 repair 是否仍适合当前 student；
- 新题是否逐渐只剩容易题；
- 单位 teacher token／环境 step 的独立成功率增益是否下降。

完整方法需要与同预算的交互式修复／完整重写对照比较，而不仅与固定 K 条离线 SFT 比较。

## 16. 完整 pipeline 伪代码

以下为算法规格，不是已经运行过的实现。

~~~python
theta = load_shared_seed_sft_checkpoint()
optimizer = init_shared_optimizer(theta)
task_pool = D_seed.tasks
history_buffer = empty_buffer()

for round_id in range(max_rounds):
    frozen = freeze_for_collection(theta)

    # 所有臂共享同一任务边界与公开信息范围。
    rollouts = student_rollouts(frozen, task_pool, attempts=2)
    successes = retain_successes(rollouts, max_per_task=1)
    failures = stratified_select_failures(rollouts, cap=20)

    seed_or_existing_repairs = []
    for failed_trace in failures:
        diagnosis = teacher_diagnose(failed_trace)
        candidates = propose_replay_refine(failed_trace, diagnosis)
        repaired = choose_and_ablate_success(candidates)
        if repaired is not None:
            episode = build_prefix_consistent_episode(repaired)
            if episode.is_valid:
                seed_or_existing_repairs.append(episode)
        log_all_costs_and_failures()

    new_task_episodes = []
    if protocol == "P1" and seed_or_existing_repairs:
        gradients = repair_gradients(
            frozen, seed_or_existing_repairs, actual_training_loss
        )
        groups = group_with_selected_variant(gradients, diagnostics)
        specs = extract_supported_conditions(groups)
        candidates = task_factory_or_retrieval(specs, U_train)

        for task in validate_and_deduplicate(candidates):
            # 每轮最多接纳 8 个新任务；失败也计预算。
            attempted = student_rollout(frozen, task, attempts=1)
            if attempted.success:
                episode = success_episode(attempted)
            else:
                episode = bounded_repair_and_label(attempted)

            if episode is not None and episode.is_valid:
                new_task_episodes.append(episode)
                task_pool.add(task)
            else:
                log_unsolved_or_rejected(task)

            if new_task_budget_exhausted():
                break

    buckets = build_disjoint_buckets(
        repairs=seed_or_existing_repairs,
        generated=new_task_episodes,
        successes=successes,
        seeds=D_seed,
        history=history_buffer,
    )

    theta, optimizer = sft(
        theta, optimizer, buckets,
        updates=50, loss=declared_loss
    )

    history_buffer = update_bounded_history(history_buffer, buckets)
    metrics = evaluate_independently(theta, D_probe)
    save_checkpoint_and_manifest(theta, metrics)

    if fixed_budget_reached() or declared_stopping_rule(metrics):
        break
~~~

实现时必须在两个地方切分数据：

1. 来自原 K 题的 episode 与追加任务的 episode，用于 R/G/S 分类；
2. 原任务失败修复与新任务修复的 20/8 条预算，用于避免在失败候选上无限消耗调用。

伪代码中的每个预算检查都应发生在新的 API 调用或环境 rollout **之前**，并在请求结束后补记实际 token；若 API 返回量超过预估，停止后续工作，不丢弃已发生的成本。

## 17. Teacher prompt 模板

运行前把方括号字段替换为实际内容。模板中的 JSON schema 使用实现中固定的版本，模型输出后由 parser 校验。以下四类调用用途不同，不复用一个“万能反思 prompt”。

### 17.1 计划理解

~~~text
你在分析一个交互式 agent 的失败执行。
任务、初始环境、成功条件都不能修改。
与参考路线不同不等于错误；有效但不最短的路线可以保留。

输入：
[原任务]
[学生公开 reasoning-action-observation trace]
[环境最终 success/termination 信息]
[动作语法与总步数上限]

请按固定 DiagnosisRecord schema 返回：
1. 学生表达出的子目标、假设，以及支持这些解释的具体 step IDs。
2. 哪些选择可能保留；未执行验证前只能称为候选。
3. 原任务哪些要求没有满足；不能确定的地方写 unknown。
4. 哪几处需要联合修改；允许不连续修改。
5. 不要将第一次路线分歧自动认定为第一个错误。
不要输出新的环境 observation。
~~~

### 17.2 联合修复候选

~~~text
为同一个原任务和同一个初始状态提出最多 3 个不同的 patch。
目标是找到环境能执行成功、且尽量少改变学生原动作序列的方案。
允许修改多个不连续位置；不要为了保留明显不可行的计划强行限制修改。

输入：
[原任务与原公开 trace]
[DiagnosisRecord]
[此前候选的真实重放反馈，如有]
[PatchRecord schema、索引规则、动作语法、预算]

每个 patch 最多 4 个 blocks。
所有索引引用原 trace；不能修改任务、初始状态或 observation。
所有新探索、移动、恢复动作都必须显式包含在 patch 中。
你提出的是候选，不得声称未经环境执行的候选已经验证成功。
只返回符合 schema 的结构化输出。
~~~

### 17.3 前缀一致的 reasoning 标签

~~~text
你正在为当前一步动作写简短依据。
你只能依据下面提供的任务与当前公开历史，不知道未来执行结果。

输入：
[原任务]
[当前有效公开历史 h_t]
[当前要执行的动作 a_t]

输出：
- 不超过 64 token 的 reasoning；
- reasoning 所依赖的历史 step IDs；
- 如果动作或其依据要求当前未知事实，返回 unsupported。

可以说“需要检查某容器”；不能声称尚未观察到的内容已经确定。
不要复述事后错误分析，不要提及未来成功或未来 observation。
~~~

### 17.4 条件练习规格

~~~text
请根据已由环境验证的修复案例，提出针对性练习的结构化规格。
你不是直接判定任务成功的 evaluator。

输入：
[最多两个代表 RepairRecords]
[允许的任务模板、目标谓词、对象类型与场景编辑规则]
[禁止使用的任务 fingerprints]
[指定练习类型 C1/C2/C3 与数量]

输出 TaskSpec：
- 保持不变的条件；
- 唯一或明确列出的变化条件；
- 对应的修复技能及支持案例；
- 可执行任务模板所需参数；
- 若为 C2，说明将运行哪三／四次实际验证来判断条件差异。

不要仅改写自然语言；不要改变 instruction 而不改变对应的成功谓词。
没有受支持的变换时返回 no_valid_spec，不强行生成。
~~~

## 18. 实现顺序、验收与失败处理

### 18.1 推荐实现顺序

| 顺序 | 模块 | 完成标准 |
|---|---|---|
| 1 | 环境 reset/replay + 轨迹日志 | 通过第 3.3 节的 5 题确定性验收 |
| 2 | Patch parser/compiler + 动作 alignment | 插入、删除、替换、非连续编辑均能正确编译；冲突 patch 被拒绝 |
| 3 | Teacher diagnosis/proposal + 执行验证 | 每个成功修复都有真实 replay；所有失败都有成本记录 |
| 4 | 前缀标签 + loss mask | 第 14 节入训案例逐条通过审核 |
| 5 | E1 执行干预 + E2-N/E2-F 小步更新 | 产出第 14 节机制证据与判断 |
| 6 | 条件练习检索／task factory | 任务与 reward 一致，重复和 split 检查通过 |
| 7 | 梯度分组 | 同 checkpoint 可复现；与随机／语义基线比较 |
| 8 | 三轮调度与停止 | 无跨轮旧梯度复用，无评估数据回流 |

### 18.2 必须覆盖的实现检查

这些检查验证语义正确性，不用人为制造“模型一定会更好”的测试。

- **索引检查**：两个不连续 replace 按原索引执行，不因前一个插入发生漂移。
- **观察检查**：修改动作后，后缀 observation 均来自新 replay。
- **删除检查**：中间删除会在相应前缀提供新的下一动作 target。
- **成功检查**：done 但失败的例子不会进入 success pool。
- **信息检查**：未来 observation 与全轨迹 diagnosis 不出现在某个过去决策的输入中。
- **归一化检查**：把一个 episode 拆成多条训练行不会改变其总权重。
- **梯度检查**：特征收集前后参数 hash 不变，optimizer step 数不变。
- **分桶检查**：每个 LearningEpisode 只属于一个来源桶。
- **任务检查**：instruction 变化对应目标／状态变化；同义改写不算新任务。
- **预算检查**：被拒绝的候选和格式重试仍出现在账本中。
- **评估检查**：从初始状态运行 student 时没有 teacher patch 或训练专属提示。

### 18.3 异常处理

| 情况 | 默认处理 |
|---|---|
| Teacher JSON 不合 schema | 消耗共用的 1 次格式重试；仍失败则保留失败记录并停止该例 |
| Patch 越界／重叠 | 不执行；反馈到剩余 proposal 调用，不自动猜测索引 |
| 原轨迹无法重放 | 标记 NONREPLAYABLE，排查环境；不能用其做 edit necessity 分析 |
| 6 个候选均失败 | 记录 NO_SUCCESS_WITHIN_BUDGET；不把原示范偷偷替代为最小修复 |
| 存在成功但撤销预算耗尽 | 可训练，标记 BEST_OBSERVED/ABLATION_BUDGET_EXHAUSTED |
| 标签包含不支持的事实 | 修正或剔除；不能仅因为动作最后成功就接受该 reasoning |
| 修复需要改变原目标 | 拒绝作为 repair；可另建新练习并单独验证 |
| 新任务合法但暂时无人解出 | 记 UNSOLVED_WITHIN_BUDGET；不当作非法任务 |
| 生成候选碰到 probe/eval fingerprint | 拒绝，记录 split violation |
| Teacher 服务中断 | 从 checkpoint/账本续跑；未完成数据不进入训练 |
| 所有修复失败、只剩 seed 数据 | 该轮无新修复信号；停止并报告，不把继续 SFT 当成本方法成功闭环 |

### 18.4 计划保留平局项的可实现定义

第 7.1 节的 p 只允许以下可程序化检查的谓词：

- used_object(instance_id)：成功执行轨迹确实操作了该具体实例；
- visited_location(location_id)：成功执行轨迹包含对该地点的有效到达；
- ordered_actions([a,b,...])：规范化动作序列包含该有序子序列。

这些谓词必须在候选产生前由 diagnosis 登记，并通过真实执行轨迹检查。无法映射到上述谓词的自然语言“子目标”不进入 p，仅作解释。p 不涉及读取隐藏世界状态，且只用于动作编辑距离相等时的排序。

## 19. 首版运行配置摘要

下面是上述默认值的集中视图。运行时把实际模型、路径和版本写入 manifest；缺失这些必填字段时程序应拒绝启动，不猜测服务端模型或 checkpoint。

~~~yaml
version: 0.1
protocol: P0_for_pilot_P1_for_full_pipeline
teacher_display_name: GPT-6-luna
student_display_name: Gemma4-E4B-it

data:
  K_grid: [10, 20, 30, 100]
  first_K: 20_if_available_else_existing_30
  seed_tasks: selected_K
  probe_tasks: 60
  reuse_existing_seed_task_ids: true
  evaluation_task_ids: existing_reference_manifest

collection:
  attempts_per_existing_task: 2
  greedy_first_attempt: true
  sampled_temperature: 0.7
  sampled_top_p: 0.95
  max_environment_actions: 50
  max_assistant_turn_tokens: 256
  max_context_tokens: 16384
  existing_failure_repair_cap: 20
  new_task_repair_cap: 8

repair:
  candidate_batches: 2
  candidates_per_batch: 3
  max_blocks_per_candidate: 4
  ablation_replays_cap: 12
  total_replays_cap_per_failure: 24
  candidate_order: action_edit_distance_then_preserved_predicates_then_length
  require_real_environment_success: true
  mutate_goal_or_initial_state: false

labels:
  format: inherit_existing_reason_action_format
  rationale_max_tokens: 64
  regenerate_after_first_action_change: true
  supervised_positions: repaired_decisions_and_delete_successors
  episode_weighting: uniform_over_supervised_decisions

gradient:
  enabled_in_pilot: false
  full_pipeline_variant: G2
  projection: CountSketch
  projection_dimension: 2048
  projection_seed: 17
  max_clusters: 4
  min_cluster_size: 2

curriculum:
  enabled_in_pilot: false
  full_pipeline_source: T1
  generation_T2_requires_compiler_validation: true
  max_candidates_per_cluster: 6
  max_accepted_tasks_per_cluster: 2
  max_accepted_tasks_per_round: 8

training:
  pilot_rounds: 1
  full_pipeline_max_rounds: 3
  full_updates_per_round: 50
  full_effective_batch_size_episodes: 8
  pilot_updates_cap: 20
  pilot_effective_batch_size_episodes: 4
  pilot_parameterization_and_loss: inherit_existing_sft
  pilot_fixed_checkpoints: [0, 1, 5, 20]
  pilot_checkpoint_selection: none
  full_checkpoint_selection: best_probe_success_tie_earliest

required_runtime_manifest_fields:
  - teacher_actual_api_model_id_and_version
  - student_revision_and_checkpoint_hash
  - tokenizer_and_chat_template_hash
  - environment_version_and_game_file_hashes
  - task_split_manifests
  - actual_trainable_module_names
  - optimizer_state_and_trainable_parameterization
  - training_and_initialization_seeds
  - code_revision
  - inherited_harness_overrides
~~~

若 teacher API 不支持某个采样或长度参数，记录真实可用参数与响应限制，所有臂采用一致策略；不声称一个未生效的 temperature 配置已经固定了 teacher 随机性。

## 20. 相关工作与本项目主张边界

以下仅用于定位设计，不是完整 novelty survey；本方法的效果需由上述实验建立。

- **[R1] ALFWorld 官方仓库与交互接口**：用于环境和 reset/step 契约核对。本文新增的 replay 管理、patch compiler、task factory 需要自行实现。  
  [ALFWorld README](https://github.com/alfworld/alfworld/blob/master/README.md)；[基础配置](https://github.com/alfworld/alfworld/blob/master/configs/base_config.yaml)。

- **[R2] SCoRe — Student-Centered Distillation Narrows the Agentic Gap Between Small and Large LLMs**：该工作已经读取完整学生轨迹进行全局判断，然后定位最早错误、干预并进一步训练。不能把“看完整轨迹”本身列为本项目创新。  
  [论文 v3](https://arxiv.org/html/2509.14257v3)。

- **[R3] Projected Distillation: On Learning to Solve Hard Problems from Close, Correct Targets**：已经研究将失败学生输出最小修改为正确目标、在线刷新与局部对比更新。因此“让 teacher 少改学生输出”本身不是足够的新颖性主张。  
  [论文 v1](https://arxiv.org/html/2610.27467v1)。

- **[R4] LESS: Selecting Influential Data for Targeted Instruction Tuning**：提供优化器相关梯度特征及数据选择的先例；梯度分组在本项目中仍需要独立消融。  
  [ICML 2024 论文页面](https://proceedings.mlr.press/v235/xia24c.html)。

- **[R5] CausalFlow: Causal Attribution and Counterfactual Repair for LLM Agent Failures**：涉及轨迹干预、反事实修复和训练可用的修复对。其验证方式包含重新执行或结果预测；本文的 ALFWorld 主设置坚持实际环境执行。不能将“反事实修复”宽泛地宣称为首次提出。  
  [论文 v1](https://arxiv.org/html/2605.25338v1)。

本项目优先检验的组合是：**黑盒 teacher、少量种子任务、尊重学生有效路线的联合动作修复、实际环境验证、从修复条件得到可迁移练习，以及按当前学生更新结构分配有限监督。**

论文层面的贡献应随实验结果收敛：核心修复、条件课程和梯度组织必须分别有证据；未带来收益的模块可以删除。第一份结果应先回答第 14 节，而不是先为整个复杂 pipeline 取一个新名字。



