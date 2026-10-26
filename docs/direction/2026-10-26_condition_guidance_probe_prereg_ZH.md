# 条件提示小实验（CG probe）预注册

日期：2026-10-26（CDT）。设计来自用户 13:57 CDT 的消息，原文存档在 `docs/direction/2026-10-26_condition_guidance_user_ZH.md`。本文件在任何 teacher 调用和 rollout 之前写定。如有改动，追加在文末的"修订"一节，并注明时间和原因。

## 1. 问题

明确指出"当前计划中哪个前提还没有观察支持"，能否在不提供目标位置、也不给出下一步动作的情况下，改变学生的执行？

## 2. 案例

案例来自 E1 U_mech 分支：θ₀ 在原 greedy rollout 中失败，且整条轨迹从未在 observation 里见到目标物体实例的任务，共 7 个。

来源：`reports/runs/repair_pilot/cov_theta0_umech/coverage/failures.json`（attempt-01-greedy）。

| 任务（train/ 下） | 目标物体 |
|---|---|
| look_at_obj_in_light-Newspaper-None-DeskLamp-224 | newspaper |
| pick_and_place_simple-Kettle-None-Shelf-1 | kettle |
| pick_clean_then_place_in_recep-Kettle-None-DiningTable-7 | kettle |
| pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18 | lettuce |
| pick_cool_then_place_in_recep-Potato-None-SideTable-21 | potato |
| pick_cool_then_place_in_recep-Apple-None-Microwave-19 | apple |
| pick_cool_then_place_in_recep-Egg-None-Microwave-5 | egg |

这 7 个都是 U_mech（TRAIN 侧）任务，不涉及 dev140，也不涉及 valid_unseen。另外 3 个"看到但没拿"的失败不在本实验范围内。

## 3. 分支点与预算

- **前缀**：取原轨迹的前 10 个动作（zero-based 0–9），包括学生原来的完整回复（Thought + Action）。
- **重放**：每次 rollout 都做真实的 reset，再重放这 10 个动作。每一步的 transition（observation、admissible commands、reward、终止标志）必须与原记录完全一致，否则这次 rollout 作废并记录原因。
- **续跑**：学生从动作 10 开始自主执行。环境上限保持原来的 H=40 步，前缀已占 10 步，剩 30 步。非法动作也计入步数，与原协议相同。
- **模型与服务**：
  - 4 个分支共用同一个 checkpoint，即 θ₀ = `e4b-tsft-k20-e32-lr5e-5`（LoRA）。
  - 共用同一个 vLLM 服务，在 rai GPU 1（Blackwell 96GB）上运行；该卡与原失败 rollout 是同一张。
  - 服务启动时先做 HF/vLLM LoRA 一致性检查，通过后才开始 rollout。
  - 解码设置与 coverage 阶段相同（1024 token 单轮上限，enable_thinking=false）。

## 4. 四个分支

**提示的注入方式**：
- 注入位置是分支点那一轮的 user turn，也就是动作 9 之后的 observation。在这个 observation 字符串末尾追加 `"\n\nGuidance: " + 提示文本`。
- 只注入这一次。之后各轮的历史都保留这条带提示的 observation，但不再追加新的提示。
- 系统 prompt、其他轮的 observation、admissible commands 都不变。
- 环境记录里保存的是真实 observation。实际发给学生的 prompt 另由学生调用日志保存。

| 分支 | 提示文本 |
|---|---|
| A 自主继续 | 无（不追加） |
| B 一般提醒 | 固定文本，7 个案例相同：`Review your progress so far, then continue to complete the task.` |
| C 条件提示 | teacher 生成的 `condition_hint` |
| D 条件提示加行动指导 | `condition_hint + " A next step that can gather this information: " + command + " (" + reason + ")"` |

C 和 D 来自 teacher 的同一次调用，因此 D 的文本严格包含 C 的文本。

## 5. Teacher 提示的生成

- **模型**：gpt-6-luna（黑盒），每个案例调用一次。
- **只读公开信息**：
  - 任务文本、初始 observation；
  - 前 10 步每一步的学生回复（Thought + Action）和对应 observation；
  - 分支点的 admissible commands；
  - 已用步数（10）和步数上限（40）。
- **不读**：目标位置等隐藏状态、任务文件路径、该任务的 teacher 示范、E1 的修复记录。
- **输出 JSON 包含四项**：
  - `student_plan`：从学生自己的 Thought 归纳出的当前计划和子目标；
  - `conditions`：计划成立所需的条件。每条标注状态（`undetermined` 尚未确定 / `known_unsatisfied` 已知但尚未满足 / `refuted` 已被观察否定 / `satisfied` 已经满足），并给出证据步号（-1 表示初始 observation）；
  - `condition_hint`：写给学生的话，最多 70 个英文词。要说明三点：学生当前的子目标；计划中哪个前提还没有任何观察支持；到目前为止的观察已经排除了什么。
  - `action_suggestion`：
    - `command`：一条信息获取命令，必须从当前 admissible commands 原样复制，动词限 go to / open / examine / look；
    - `reason`：最多 30 个词，只能用公开信息和一般家居常识。

### 自动校验

不通过就带上原因做 format retry，最多 2 次。仍不通过，该案例的 C/D 记为缺失，不手工改文本。

- **C 的文本不得**：
  - 包含分支点的任何 admissible command 字符串；
  - 提到前缀中学生没有访问过的容器实例或容器类型。"访问过"指在前缀中作为合法 go to / open / examine 动作的对象；
  - 超过 70 个词。
- **D 的命令**必须在分支点的 admissible commands 里，动词在允许范围内；`reason` 不超过 30 个词。

### 人工审核

- 我审核全部 7 条 C 和 D，检查两点：
  - C 是否暗示了位置或下一步，包括用常识暗示"去哪找"；
  - 条件状态是否与公开历史一致。
- 审核只能"通过"或"否决"。否决后重新生成一次并记录，仍不通过则该案例记为缺失。
- 通过后写审批文件（含 hints 文件的 sha256），rollout 工具只接受已审批的 hints。
- 全部 7 条提示随报告一起发给用户抽查。

## 6. 解码与样本量

- 每个"案例 × 分支"跑 1 次 greedy，再跑 8 次采样（T=0.7，top_p=0.95，sampling seed 0–7）。
- 每轮的 request seed = sha256([seed, task_id, 轮次])，与分支无关。因此同一 seed 在 4 个分支的同一轮用的是同一个随机数（common random numbers）。
- 共 7 × 4 × 9 = 252 次 rollout。
- **主分析用 8 次采样**；greedy 单独列出，不并入主分析。

## 7. 指标

以下指标都只统计分支点之后（动作 10–39）的执行。

- **主指标：找到目标**。分支点之后的任一 observation 中出现目标物体类型的实例，按正则 `\b<目标类型> \d+\b` 判断，不区分大小写。
- 找到目标所需的步数：从分支点算起，第一个出现目标的动作序号，取值 1–30。只在找到的 rollout 里统计中位数，同时报告找到率。
- 拿到目标：合法地执行了 `take <目标类型> N from ...`。
- 任务成功：环境判定成功。
- 诊断指标（只描述，不作判定）：
  - D 中学生第一步是否执行了 teacher 建议的命令（遵从率）；
  - 执行了建议命令的 rollout 中，下一个 observation 是否直接出现目标，即 teacher 的建议本身是否命中；
  - 分支点后第一步动作与 A 的差异；
  - 回到已经确认为空的位置的次数；
  - 非法动作数；
  - 分支点后第一条 Thought 原文，只记录是否承认位置未知（辅助记录）。

## 8. 分析与判定规则

- **找到率**：每个分支的找到率 = 7 个案例各自"8 次采样中找到的比例"的平均。
- **成对差**：报告 C−B、D−C、B−A、C−A、D−A，95% CI 用按案例的 cluster bootstrap（10,000 次，seed 0）。逐案例的表格全部报告。
- **"X 优于 Y"的标准**：两条同时满足——找到率之差 ≥ 0.15；7 个案例中 X>Y 的案例数比 X<Y 的至少多 2 个。否则记为"无明确差别"。n=7，只看方向，不做显著性结论。
- **结论映射**（来自用户的设计）：
  - C 优于 B：支持"指出缺失条件能帮助学生利用已有能力"。
  - D 优于 C，且 C 不优于 B：学生还需要学习怎样选择信息获取行动，课程应重点训练探索决策。
  - C、D 都不优于 B：下一步检查 teacher 闭环执行在同一前缀、剩余 30 步内能否完成，判断这类情境是否存在可获得、可传授的策略优势。这一步不在本次范围内，先报告，等用户确认后再做。
  - B 优于 A 单独报告：说明一般提醒或"多出一段文字"本身就有作用，C 的效果应相对 B 来看。
- 次要指标（拿到目标、任务成功）用同样的规则报告，但不改变上面的结论映射。

## 9. 预算与执行

- **Teacher**：本实验上限 200,000 input / 60,000 output token（含隐藏推理），计入机制试验的总上限 5M / 0.5M。在此之前已用 449,937 / 156,801。
- **算力**：rai GPU 1 单卡，rollout 8 个 worker 并行。不训练，也不使用 hpg。
- **代码**：在 `repair-v01-eval` 分支新增文件，由 Codex 实现、我审核。实验从固定 commit 的独立 worktree 运行。
- **结果目录**：`reports/runs/repair_pilot/cg_probe/`。

## 10. 已知局限

- C 的文本比 B 长，存在长度混杂。
- 提示只注入一次。用户方案里的"交替进行"（teacher 随新观察重新判断条件）不在本次范围内。
- 7 个案例都是搜索失败，其中多数是学生在逐个开柜子。结论不能直接推广到其他失败类型。
- vLLM greedy 不完全可复现，greedy 结果只作参考。

## 修订

（暂无）
