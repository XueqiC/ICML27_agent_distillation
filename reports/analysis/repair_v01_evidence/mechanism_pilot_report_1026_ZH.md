# 新方法第一步机制验证（pipeline v0.1 · 第 14 节）：完整报告

日期：2026-10-26（CDT）
范围：设计文档 `docs/direction/2026-10-26_agent_distillation_pipeline_v0_1_ZH.md` 第 14 节的 E1（执行干预）；E2（学习干预）因配对数不足没有运行，原因见 §5.6。
预注册：`docs/direction/2026-10-26_pipeline_v01_mechanism_prereg_ZH.md`。§5 记录用户 10-26 10:40 CDT 的决定，§6 记录看任何修复结果之前写下的分支细节。
运行地点：rai（GPU1，RTX PRO 6000 Blackwell）。teacher = gpt-6-luna（黑盒），student = gemma-4-E4B-it @ ee0ef602。

## 0. 结论先行

1. **H1（失败能用少量联合修改修好）在"程序性失败"上成立，在"搜索失败"上不成立。**
   - 两个分支合计 26 个失败，按原轨迹形态分成两类：
     - 走到一半的失败：放了但没完成、跳过必需处理、拿到但没完成；
     - 搜索失败：从没见到目标物体，或者看到了但没拿。
   - 走到一半的失败修好 **10/14（71%）**；搜索失败只修好 **1/12（8%）**。
   - 修好的 11 个里有 10 个**只改了一个连续片段**（replace），替换动作数中位数 2.5，范围 1–15。
2. **真正"联合、不连续"的修复很少**：11 个成功修复中只有 1 个（pick_two Candle）是两个块、而且两块缺一不可（环境实测：只加任一块都失败，两块一起成功，撤销后恢复仍成功）。文档 14.5 要求至少 2 个来自不同任务的案例，所以本轮**没有检验"联合修复"这一主张**。
3. **SFT 之后剩下的失败以搜索为主，修复法当前形态覆盖不到**：
   - θ₀（SFT-K20）在 20 个新任务（U_mech）上失败了 15 个，其中 10 个是搜索失败，最后只修好 3 个（20%）。
   - base 学生（K0 分支）的失败多是原地打转或差最后几步，11 个修好 8 个（73%）。
4. **H2（离学生近的监督更好学）这一轮没能检验**：两个分支都没凑出合格的 N/F 配对（0/4），E2 没有运行。卡住的原因：
   - Near 保留了学生原路线里被环境拒绝的无效动作，文档 14.6 规定这样的 target 不能参与配对；
   - 可修的案例本来就少；
   - 一个已修复的工程 bug（下文 §5.6）。
5. **只重复原 K 题没有失败信号**：θ₀ 在自己的 20 个训练题上 80 个 episode 全部成功。新方法必须从新任务（P1）取得失败。这与文档 14.0 的预判一致。

## 1. 这一步在验证什么

设计文档 14.1 的因果链：修改特定决策 → 环境结果由失败变成功（**E1**）→ 以这种修复作为监督 → 学生在自己的失败状态上改变决策 → 脱离 teacher 后更容易完成任务（**E2**）。第一步只验证 H1（修复可行性）和 H2（近路线目标更好学），不碰 H3（条件课程）、H4（梯度分组）。

## 2. 实验设置

| 项 | 取值 |
|---|---|
| 环境 | ALFWorld 0.4.2 / TextWorld 1.7.0，项目 harness H0（Thought/Action 格式，thinking off） |
| 步数上限 H | 40（文档默认 50；按"沿用已有 harness"的规定改为 40，已登记） |
| 学生单轮输出上限 | 1024 token（文档默认 256，同上） |
| θ₀（主分支） | e4b-tsft-k20-e32-lr5e-5：轨迹SFT K20，rule-A rung e32，lr 5e-5，seed 0，在 LONI 训练。拷到 rai 后 19 个文件的 sha256 与 LONI 一致 |
| θ₀（K0 分支） | base gemma-4-E4B-it，无 adapter |
| 评测硬件 | rai GPU1（Blackwell，CUDA_DEVICE_ORDER=PCI_BUS_ID）。pilot 内所有比较都在同一种卡上；主表数字仍全部来自 hpg B200，不混用 |
| 软件版本 | vLLM 0.27.1、transformers 5.14.1、torch 2.13，与 hpg 一致 |
| 代码 | eval 侧分支 repair-v01-eval（= 主网格 eval 树 46b8527 + repair 包）；train 侧分支 repair-v01-train（= 训出 θ₀ 的树 2506fa9 + repair 包）。**所有真实运行都从固定在某个 commit 的独立 worktree 启动** |
| teacher 预算 | 全实验硬上限 5M 输入 / 0.5M 输出 token。每条失败的上限：诊断 1 次、提案 2 轮 × 3 个候选、标签调用按需；含推理的输出上限 64k |

## 3. 实验步骤（逐步怎么做的）

### 3.1 M0：工程（不用 GPU、不调 teacher）

- 新增 `repair/` 包，由 Codex 实现、我逐模块审过：
  - **patch 编译**：严格按文档 6.3。索引一律指原轨迹；replace/delete 区间不能重叠；每个边界最多一个 insert；最多 4 个块，超过的另记为 extended variant；insert 和 replace 不能为空。
  - **动作对齐与编辑距离**：动态规划，平局顺序是 保留 > 替换 > 删除 > 插入；监督位置按文档 8.4 计算。
  - **执行器**：reset 原任务后真实执行编译出的动作序列，每一步读真实 observation，成功就停。缓存命中不计环境步数。
  - **最小性检查**（7.3）：逐块恢复原动作并重新执行；仍成功且代价下降就接受这次撤销，然后重新开始一轮；预算内跑完整一遍才标 BLOCK_IRREDUCIBLE。
  - **联合必要性**（7.4）：对 2–4 个块，分别记录"去掉某一块"的真实执行结果。
  - **三类预算账本**（2.3）。
  - **teacher prompt**（17.1–17.3，外加 Far 变体）：输入只有公开字段，即任务文本、学生的 Thought/Action/observation、结束原因、动作语法、步数上限；不含 admissible 列表、audit 字段、原 teacher 示范和其他任务。有测试专门检查泄漏。
  - **严格 schema**：每个案例共用一次格式重试。
  - **前缀一致的 reasoning 标签**（8.2）：逐步只输入任务、当前修复后的公开前缀和当前动作；自动检查引用的 step 是否在当前动作之前、提到的实例编号是否已经公开出现过。
  - **N/F 配对**：严格按 14.6 第 1–9 条和 14.7 的选择规则。
  - **三个 probe**（14.9–14.11）和 E2 训练臂：在 2506fa9 的 trainer 上接着训 θ₀ 的 LoRA，20 次 update，在第 1、5、20 步存 checkpoint。
- **确定性验收（文档 3.3）**：选 5 个训练任务，覆盖 5 类任务；teacher 原始成功示范各真实重放 2 遍，另外构造一个 FAILED 和一个 STEP_LIMIT 案例。fingerprint、逐步 observation、成功判定全部一致，三种结束原因可以区分，共 410 个环境步。**通过。**
- 测试：eval 侧 357 个、train 侧 406 个 CPU 测试全部通过。现有 trainer 测试中有 7 个失败，在未改动的 2506fa9 上同样失败，是原本就有的。

### 3.2 M2：冻结任务划分（在任何 teacher 调用之前）

- **范围**：ALFWorld TRAIN 共 3,553 个实例、1,465 个 task group。
- **排除**：所有 baseline **实际训练过**的任务所在的 group，即 K500 链及其所有 view 里被选中的任务、旧网格 seeds_k*_s* 里被选中的任务、r03 各集合；再加上 dev140 和 sealed。合计排除 1,009 个 group，剩 593 个 group 可用。
- **抽取**：
  - D_probe：60 个，每类任务 10 个；
  - D_unseen_quick：24 个，从 D_probe 中每类取 4 个；
  - U_mech：20 个，与 D_probe 按 group 不相交。
- 按 sha256([seed, task_id]) 排序取用，固定 seed，没有缺额。
- 第一版排除规则写得太宽，连候选名单都排除了，结果把 TRAIN 全排光。Codex 照规则执行后发现并报告了这个问题，没有自行放宽，规则由我改正。

### 3.3 M1：覆盖检查（决定走哪个分支）

- **做法**：在 rai GPU1 上用 vLLM serve 待测模型。adapter 先过 LoRA 等价检查（HF 参考 vs vLLM）。然后在指定任务上每题跑若干次：第 1 次 greedy，之后用 T=0.7、top_p=0.95 采样（每次请求的 seed 由 sampling_seed、任务和回合确定）。H 都是 40。
- **预注册的分支规则**：θ₀ 在自己的 K20 训练题上失败的不同任务 ≥ 6 个就做 P0；否则补 2 次采样 rollout；仍不足 6 个就改用 K=0 起点。
- **用户追加**：U_mech 分支（P1）。θ₀ 在 20 个新任务上采集失败，回答"能否在 SFT-K20 上继续改善"。

### 3.4 E1：执行干预（每条失败的流程）

1. **原轨迹确认**：从同一初始状态重放 coverage 里记录的失败轨迹，逐步核对，确认确实失败。
2. **D_repair_probe**：在任何 teacher 调用之前，从失败任务里按 sha256([20261029, task_id]) 最小取 2 个固定下来。它们只用于 probe，不进入训练。
3. **诊断**：调用 1 次 teacher，按 DiagnosisRecord schema 输出学生的计划、可保留的选择、未满足的要求、建议的联合修改、未知项。
4. **提案**：第 1 轮请 teacher 提至多 3 个 patch，每个至多 4 个块，索引一律指原轨迹，按编译后的规范化动作序列去重。没有一个成功时进入第 2 轮：teacher 能看到第 1 轮候选的**真实执行记录**，再提完整的新 patch。
5. **执行验证**：每个候选都 reset 后真实执行。成功即停；动作用完仍没成功，或撞到 40 步上限，都判失败。不自动补后续动作。
6. **选择**：在成功的候选里按 (动作编辑距离, −保留的子目标数, 成功轨迹长度, patch_id) 字典序选最优。保留子目标数这一项目前恒为 0，因为诊断里的自然语言子目标不算已登记的可执行谓词。
7. **最小性和联合必要性**：逐块撤销测试，最多 12 次；最后做一次不走缓存的"撤销后恢复"确认。每条失败的 replay 总数不超过 24 次。
8. **标签**：只对选中的修复重建 reasoning，每步一次调用，每条至多 64 token。动作改动之前的原学生回复逐字保留。
9. **配对（为 E2 准备）**：Near 先过文档 14.6 的基本质量要求：真实成功、不含被环境拒绝的动作、目标物体实例和必需操作能从公开记录核对。过了之后才去请求 Far（teacher 另一条有效路线）；Far 要满足长度比、token 比和编辑距离差的条件。Near 不合格的题整题退出配对，不换用一个更差的 Near（14.7 第 2 条）。

## 4. 结果

### 4.1 M1 覆盖检查

| 运行 | 起点 | 任务 | episode | 失败 episode | 失败的不同任务 | 分支判定 |
|---|---|---|---:|---:|---:|---|
| cov_theta0_k20 | θ₀ | 20 个 K20 训练题 | 40（greedy + 1 次采样） | 0 | 0 | 补 rollout |
| cov_theta0_k20_extra2 | θ₀ | 同上 | 80（再加 2 次采样） | 0 | 0 | **转 K0 分支** |
| cov_base_k20 | base E4B | 同上 | 40 | 19 | **11** | K0 分支可做 |
| cov_theta0_umech | θ₀ | 20 个 U_mech 新任务 | 40 | 29 | **15**（6 类任务都有） | U_mech 分支可做 |

θ₀ 在 rai 上的 LoRA 等价检查两次都通过：adapter 误差 5.0e-6，base 噪声 0.038。

### 4.2 E1 逐案例结果

各列含义：
- **原失败类型**：按 coverage 失败轨迹的动作和公开 observation 判断，规则与主网格分析的 failure_modes.py 相同。
- **原轨迹重复率**：最近 4 步里出现过的重复动作所占比例，越高越接近原地打转。
- **成功步数**：修复后轨迹在第几步成功。
- **编辑距离**：相对原 40 步轨迹，成功之后没执行的那段按文档 7.1 记为删除。


### K0 分支（θ₀ = base E4B，K20 训练题）

| 任务 | 角色 | 原失败类型 | 原轨迹重复率 | 修复结果 | 选中的修复块 | 成功步数 | 编辑距离 | 联合必要 | 候选执行结果 | 标签 | 配对 |
|---|---|---|---:|---|---|---:|---:|---|---|---|---|
| look_at_obj_in_light-Bowl-None-DeskLamp | probe | 拿到但没完成 | 0.4 | ✅ 修复成功 | replace 2 个动作 | 17 | 24 | 单块（不适用） | 动作用完仍未成功×2, 成功×1 | COMPLETE | PAIR_NOT_FOUND（N:environment_rejected_action） |
| pick_clean_then_place_in_recep-Cloth-None-CounterTop | probe | 跳过必需处理 | 0.8 | ✅ 修复成功 | replace 3 个动作 | 8 | 36 | 单块（不适用） | 成功×1 | COMPLETE | PAIR_NOT_FOUND（N:public_goal_coverage_unknown） |
| pick_cool_then_place_in_recep-Mug-None-CoffeeMachine | 候选训练 | 从没见到目标物体 | 0.0 | ❌ 预算内未修好 | — | — | — | — | 动作用完仍未成功×6 | — | — |
| pick_cool_then_place_in_recep-Tomato-None-Microwave | 候选训练 | 从没见到目标物体 | 0.23 | ✅ 修复成功 | replace 15 个动作 | 40 | 15 | 单块（不适用） | 动作用完仍未成功×3, 成功×3 | UNGROUNDED_TARGET | — |
| pick_heat_then_place_in_recep-Mug-None-CoffeeMachine | 候选训练 | 跳过必需处理 | 0.07 | ❌ 预算内未修好 | — | — | — | — | 动作用完仍未成功×4 | — | — |
| pick_heat_then_place_in_recep-Apple-None-CounterTop | 候选训练 | 放了但没完成 | 0.7 | ❌ 预算内未修好 | — | — | — | — | 撞到 40 步上限×1, 动作用完仍未成功×2 | — | — |
| pick_two_obj_and_place-Candle-None-Drawer | 候选训练 | 放了但没完成 | 0.1 | ✅ 修复成功 | replace 1 个动作, insert_before 3 个动作 | 36 | 8 | 是 | 成功×1 | COMPLETE | PAIR_NOT_FOUND（N:environment_rejected_action; N:public_goal_coverage_unknown） |
| pick_two_obj_and_place-AlarmClock-None-Shelf | 候选训练 | 放了但没完成 | 0.62 | ✅ 修复成功 | replace 4 个动作 | 15 | 29 | 单块（不适用） | 成功×1, 动作用完仍未成功×1 | COMPLETE | PAIR_NOT_FOUND（N:environment_rejected_action; N:public_goal_coverage_unknown） |
| pick_two_obj_and_place-CD-None-GarbageCan | 候选训练 | 放了但没完成 | 0.55 | ✅ 修复成功 | replace 1 个动作 | 17 | 24 | 单块（不适用） | 成功×1 | UNGROUNDED_TARGET | — |
| pick_two_obj_and_place-Statue-None-CoffeeTable | 候选训练 | 放了但没完成 | 0.7 | ✅ 修复成功 | replace 7 个动作 | 15 | 30 | 单块（不适用） | 成功×1, 未执行/被拒×1 | COMPLETE | PAIR_NOT_FOUND（N:public_goal_coverage_unknown） |
| pick_two_obj_and_place-Laptop-None-Bed | 候选训练 | 放了但没完成 | 0.5 | ✅ 修复成功 | replace 1 个动作 | 21 | 20 | 单块（不适用） | 成功×1 | COMPLETE | PAIR_NOT_FOUND（N:environment_rejected_action; N:public_goal_coverage_unknown） |

### U_mech 分支（θ₀ = SFT-K20，20 个 U_mech 新任务）

| 任务 | 角色 | 原失败类型 | 原轨迹重复率 | 修复结果 | 选中的修复块 | 成功步数 | 编辑距离 | 联合必要 | 候选执行结果 | 标签 | 配对 |
|---|---|---|---:|---|---|---:|---:|---|---|---|---|
| look_at_obj_in_light-Newspaper-None-DeskLamp | probe | 从没见到目标物体 | 0.0 | ❌ 预算内未修好 | — | — | — | — | 撞到 40 步上限×2, 动作用完仍未成功×2 | — | — |
| look_at_obj_in_light-Bowl-None-DeskLamp | 候选训练 | 拿到但没完成 | 0.7 | ❌ 预算内未修好 | — | — | — | — | 动作用完仍未成功×5 | — | — |
| pick_and_place_simple-Kettle-None-Shelf | 候选训练 | 从没见到目标物体 | 0.0 | ❌ 预算内未修好 | — | — | — | — | 撞到 40 步上限×3 | — | — |
| pick_clean_then_place_in_recep-Bowl-None-Shelf | 候选训练 | 看到但没拿 | 0.0 | ❌ 预算内未修好 | — | — | — | — | 动作用完仍未成功×3 | — | — |
| pick_clean_then_place_in_recep-Kettle-None-DiningTable | 候选训练 | 从没见到目标物体 | 0.0 | ❌ 预算内未修好 | — | — | — | — | 动作用完仍未成功×4 | — | — |
| pick_clean_then_place_in_recep-Lettuce-None-CounterTop | 候选训练 | 从没见到目标物体 | 0.0 | ❌ 预算内未修好 | — | — | — | — | 撞到 40 步上限×2, 动作用完仍未成功×3 | — | — |
| pick_cool_then_place_in_recep-Potato-None-SideTable | probe | 从没见到目标物体 | 0.0 | ❌ 预算内未修好 | — | — | — | — | 动作用完仍未成功×4 | — | — |
| pick_cool_then_place_in_recep-Apple-None-Microwave | 候选训练 | 从没见到目标物体 | 0.0 | ❌ 预算内未修好 | — | — | — | — | 动作用完仍未成功×4 | — | — |
| pick_cool_then_place_in_recep-Egg-None-Microwave | 候选训练 | 从没见到目标物体 | 0.0 | ❌ 预算内未修好 | — | — | — | — | 动作用完仍未成功×4 | — | — |
| pick_heat_then_place_in_recep-Egg-None-CounterTop | 候选训练 | 拿到但没完成 | 0.1 | ✅ 修复成功 | replace 6 个动作 | 40 | 6 | 单块（不适用） | 成功×1 | COMPLETE | PAIR_NOT_FOUND（N:environment_rejected_action） |
| pick_heat_then_place_in_recep-Mug-None-CoffeeMachine | 候选训练 | 看到但没拿 | 0.1 | ❌ 预算内未修好 | — | — | — | — | 动作用完仍未成功×4 | — | — |
| pick_heat_then_place_in_recep-Plate-None-Fridge | 候选训练 | 看到但没拿 | 0.0 | ❌ 预算内未修好 | — | — | — | — | 撞到 40 步上限×3, 动作用完仍未成功×1 | — | — |
| pick_two_obj_and_place-SaltShaker-None-Drawer | 候选训练 | 拿到但没完成 | 0.03 | ❌ 预算内未修好 | — | — | — | — | 动作用完仍未成功×2 | — | — |
| pick_two_obj_and_place-Book-None-Bed | 候选训练 | 放了但没完成 | 0.35 | ✅ 修复成功 | replace 2 个动作 | 9 | 32 | 单块（不适用） | 动作用完仍未成功×1, 成功×1 | COMPLETE | PAIR_NOT_FOUND（action_distance_gap; target_object_instances） |
| pick_two_obj_and_place-SoapBar-None-Drawer | 候选训练 | 放了但没完成 | 0.03 | ✅ 修复成功 | replace 2 个动作 | 29 | 12 | 单块（不适用） | 动作用完仍未成功×2, 成功×1 | UNGROUNDED_TARGET | — |

（K0 分支跑的是修 bug 之前的代码：当时配对检查只认 "put X in Y"，凡是被标为 coverage_unknown 的，都是这个 bug 造成的。U_mech 分支用的是修好后的代码。）

### 4.3 按失败类型汇总（两个分支合计 26 个失败）

| 原失败类型 | K0 分支 | U_mech 分支 | 合计修好 |
|---|---|---|---|
| 走到一半（放了但没完成 / 跳过必需处理 / 拿到但没完成） | 9 个，修好 7 | 5 个，修好 3 | **10/14（71%）** |
| 搜索失败（从没见到目标物体 / 看到但没拿） | 2 个，修好 1 | 10 个，修好 0 | **1/12（8%）** |

- **base 学生的失败多数是走到一半**（9/11）。原轨迹重复率很高，0.4–0.8，基本是在原地打转。teacher 只要换掉其中一小段，学生的路线就能在第 8–21 步成功。
- **θ₀ 的失败多数是搜索失败**（10/15）。重复率接近 0，说明它不是在打转，而是一直在找、但找错了地方。
  - teacher 只看得到公开 observation，也不知道物体在哪。
  - 在学生原路线上插入或替换一些探索动作后，候选大多"动作用完仍未成功"，或者撞到 40 步上限。
  - 例如：在有 21 个抽屉的房间里找报纸。

### 4.4 修改的规模

- 11 个成功修复中，10 个只有一个 replace 块：
  - K0 分支替换的动作数分别是 2、3、15、4、1、7、1；
  - U_mech 分支是 6、2、2；
  - 中位数 2.5。
- 剩下 1 个（Candle）是一个 replace（1 个动作）加一个 insert（3 个动作）。
- 编辑距离在 6–36 之间，偏大的那部分主要来自"成功后没执行的原后缀"：base 学生原本打转到第 40 步，修复后提前成功，后面那段计为删除。所以编辑距离高估了 teacher 实际改动的多少。报告里应该同时给"修复块中的动作数"。

### 4.5 联合必要性

- 唯一的多块修复是 pick_two Candle（K0 分支）。环境实测：
  - 原轨迹：失败；
  - 只加 p1（replace 1 个动作）：失败；
  - 只加 p2（在第 33 步前插入 3 个动作）：失败；
  - p1 + p2：成功；
  - 撤销后恢复：成功。
- 这个具体案例证明两处修改缺一不可。它不证明不存在别的单块修复，也只是 1 例，不满足 14.5 要求的"至少 2 例、来自不同任务"。**本轮不主张"联合修复"。**

### 4.6 标签和配对

- **reasoning 标签**：11 个成功修复里，8 个标签完整通过了自动前缀检查。3 个被标为 UNGROUNDED_TARGET（K0 的 Tomato、CD，U_mech 的 SoapBar），原因是 teacher 的说明提到了当时还没公开出现过的实例编号。按规定，这类案例不进入训练。
- **N/F 配对：两个分支都是 0 对**（E2 需要 4 对）。
  - K0 分支：
    - Near 含被环境拒绝的动作：Bowl、Candle、AlarmClock、Laptop；
    - 覆盖检查 bug：Cloth、Statue（Cloth 是 probe，本来就不进入训练对）；
    - 标签不合格：Tomato、CD。
    - Near 不合格的题按 14.7 第 2 条整题退出配对，所以一个 Far 都没有请求。
  - U_mech 分支：
    - Egg：Near 含被环境拒绝的动作；
    - Book：找到了 1 个 Far，但它与 Near 的编辑距离差不够，而且用的是不同的目标物体实例，不符合 14.6 第 3、7 条；
    - SoapBar：标签不合格。
  - "Near 含被环境拒绝的动作"的根本原因：Near 按最小修改原则保留学生原动作，学生原路线里的无效动作（环境回复 "Nothing happens"）也一起保留了；而 14.6 第 2 条要求配对的 target 不能含这类动作。两条规则在学生无效动作多的时候冲突。

### 4.7 teacher 用量

| 运行 | 输入 token | 输出 token（其中隐藏推理） |
|---|---:|---:|
| canary（2 次，各 1 次诊断调用） | 7,598 | 2,498（2,031） |
| E1 K0 分支（11 个失败） | 216,590 | 58,260（45,320） |
| E1 U_mech 分支（15 个失败） | 225,749 | 96,043（77,786） |
| **合计** | **449,937** | **156,801（125,137）** |

占总预算的 9% / 31%。输出里约 80% 是 gpt-6-luna 的隐藏推理：诊断一次约 2.4k，其中推理约 1.9k。

## 5. 过程中的工程问题（都已修复并记录）

1. **实验与 Codex 共用同一个 worktree**：第二次覆盖检查的 LoRA 检查报错，因为同一 worktree 里的代码在运行中被 Codex 改了。这次运行作废，已归档到 `_aborted/`。之后所有运行都从固定 commit 的独立 worktree 启动。
2. **teacher 输出上限没算推理**：gpt-6-luna 的隐藏推理也计入 max_output_tokens。第一次 canary 的 128 token 被推理全部用完，返回了空文本。改为含推理的上限（诊断 8192、提案 16384、标签 1024；每条失败 64k），可见文本的长度仍由 schema 限制。
3. **配对检查只认 put**：harness 放置物体用的是 "move X to Y"，导致每条成功轨迹都被判为覆盖未知。已修复（91237e3），K0 分支的数据是修复前跑的。
4. **P1 读不到 θ₀ 的支持集路径**：θ₀ 是轨迹SFT pipeline 训出来的，路径记在 baseline_training 字段下，代码只查了 sft 字段，启动时就报错，没有调用 teacher。已修复（0abb6c8）。Codex 顺手改了两处无关调用，我撤回了。
5. **rai 上 collector 需要 tokenizer**：在项目 venv 里追加安装了 transformers 5.14.1（只新增包，不改动已有的包）。

## 6. 对新方法的含义

1. **修复法的适用对象是程序性错误，不是信息不足的搜索。**
   - 走到一半的失败中 71% 能靠一小段局部修改修好，这是 H1 在这类失败上的直接证据。
   - 搜索失败几乎修不了：teacher 也不知道物体在哪，补上信息要靠探索，而探索不是"少量修改"。
2. **起点越强，可修的失败越少。**
   - 对 base 学生，大多数失败可修（8/11）。
   - 对 SFT-K20，新任务上剩下的失败以搜索为主（10/15），只修好 3 个。
   - 这和上午的证据分析一致：SFT 后剩下的失败中 37–47% 是"没找到"，teacher 自己的失败里 73% 也是这种。
   - 所以如果要在 SFT 之上继续提升，新方法必须解决搜索问题，例如：
     - 让 teacher 根据真实执行结果多轮修改，相当于让 teacher 借学生的环境交互去探索；
     - 或者把"该去哪里找"做成条件练习（H3）。
3. **"联合、不连续修改"这一卖点目前证据很弱**（11 个成功修复里只有 1 例）。大多数可修的失败只需要改一处。
4. **配对规则与最小修改原则在学生无效动作多时冲突**，这是设计上需要明确的一点（见 §7 选项 B）。
5. **H2 还没测**：要检验"近路线目标更好学"，需要足够的合格配对。

## 7. 下一步选项（需要你选）

- **A. 按原规则收尾**：本轮以 E1 的结论为止，H2 留到条件更合适时再测。
- **B. 统一清理无效动作后补做配对和 E2（K0 分支）**：在 Near 和 F 两边删掉被环境拒绝的 no-op 动作，重新真实执行确认仍成功；这些删除另列一栏，不算修复改动。
  - K0 分支有 8 个修复成功：Bowl 和 Cloth 是 probe，不进训练；Tomato 和 CD 的标签不合格；剩下 Candle、AlarmClock、Laptop、Statue 4 个训练候选，可能刚好凑够 4 对（还要每个都找到合格的 Far）。
  - 需要：实现和 review 约 1 小时，补 Far 的 teacher 调用约 20 分钟，E2 两臂训练加 probe 约 1.5–2 小时。
- **C. 扩展 E1 到搜索失败**：允许 teacher 多轮"提修复 → 看真实执行 → 再提"，例如 5 轮；或者用文档第 13 节的"teacher 闭环执行"变体。
  - 这超出预注册的 2 × 3 预算，要作为新的 variant 单独登记。
  - 它回答的是"修复法能否扩展到搜索失败"，会改变"少量修改"的定义。
- **D. 加大 U_mech 样本**：U_mech 只有 3 个修复成功，再从训练侧取一批新任务采失败。teacher 预算还剩约 4.5M / 0.34M。

我的建议：先做 **B**，用 K0 分支检验 H2，这是这一轮唯一能直接回答 H2 的路径；同时登记 **C** 作为下一步，因为它对应 SFT 之后真正的瓶颈。

## 附：文件和版本

- 代码：eval 侧 0abb6c8（M0f）；train 侧 425945c（M0c）。主网格树 46b8527 / 2506fa9 未改动。
- 运行目录（rai）：`reports/runs/repair_pilot/`
  - 覆盖检查：`cov_theta0_k20`、`cov_theta0_k20_extra2`、`cov_base_k20`、`cov_theta0_umech`
  - E1：`e1_k0`、`e1_umech`
  - canary：`e1_k0_canary*`
  - 作废的运行：`_aborted/`
- 每个 E1 运行目录都有：
  - collection.json、cases.jsonl：逐案例的全部候选、执行记录、E1 表、最小性检查、标签；
  - human_review.tsv：标签人工审核表；
  - review_requests.json；
  - global_teacher_budget.jsonl：逐次 teacher 调用的账本；
  - run_manifest.json：版本、哈希、配置。
- 划分：`data/manifest/repair_v01_splits.json`（eval 分支）。
- 预注册：`docs/direction/2026-10-26_pipeline_v01_mechanism_prereg_ZH.md` §1–6。
