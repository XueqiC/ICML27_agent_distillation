# E1（K=0 分支）结果：执行干预

起点 θ₀ = base gemma-4-E4B-it（K=0 分支，按预注册规则，因为 SFT-K20 在自己的训练题上 0 失败）。失败来自 base 在 20 个 K20 训练题上的 rollout：11 个不同任务。teacher = gpt-6-luna。全部修复都由真实 ALFWorld 环境执行验证。运行时间：2026-10-26 12:32–12:54 CDT（rai，固定 worktree 1a12f78）。

## 逐案例执行干预表

（"只加其中一个块时"：只应用那一个修复块、其余位置保持学生原动作的执行结果；单块修复就是完整修复本身。）

| 任务 | 角色 | 修复结果 | 修复块 | 原轨迹步数 → 修复后成功步数 | 编辑距离 | 只加其中一个块时 | 联合必要 | Near 中被环境拒绝的动作 | 标签 |
|---|---|---|---|---|---:|---|---|---:|---|
| look_at_obj_in_light-Bowl-None-DeskLamp | probe | ✅ 修复成功 | replace×2 | 40 → 17 | 24 | 成功 | unknown | 1 | COMPLETE |
| pick_clean_then_place_in_recep-Cloth-None-CounterTop | probe | ✅ 修复成功 | replace×3 | 40 → 8 | 36 | 成功 | unknown | 0 | COMPLETE |
| pick_cool_then_place_in_recep-Mug-None-CoffeeMachine | 候选训练 | ❌ 预算内未修好 | — | 40 → — | — | — | — | — | — |
| pick_cool_then_place_in_recep-Tomato-None-Microwave | 候选训练 | ✅ 修复成功 | replace×15 | 40 → 40 | 15 | 成功 | unknown | 4 | UNGROUNDED_TARGET |
| pick_heat_then_place_in_recep-Mug-None-CoffeeMachine | 候选训练 | ❌ 预算内未修好 | — | 40 → — | — | — | — | — | — |
| pick_heat_then_place_in_recep-Apple-None-CounterTop | 候选训练 | ❌ 预算内未修好 | — | 40 → — | — | — | — | — | — |
| pick_two_obj_and_place-Candle-None-Drawer | 候选训练 | ✅ 修复成功 | replace×1, insert_before×3 | 40 → 36 | 8 | 失败, 失败 | True | 5 | COMPLETE |
| pick_two_obj_and_place-AlarmClock-None-Shelf | 候选训练 | ✅ 修复成功 | replace×4 | 40 → 15 | 29 | 成功 | unknown | 1 | COMPLETE |
| pick_two_obj_and_place-CD-None-GarbageCan | 候选训练 | ✅ 修复成功 | replace×1 | 40 → 17 | 24 | 成功 | unknown | 2 | UNGROUNDED_TARGET |
| pick_two_obj_and_place-Statue-None-CoffeeTable | 候选训练 | ✅ 修复成功 | replace×7 | 40 → 15 | 30 | 成功 | unknown | 0 | COMPLETE |
| pick_two_obj_and_place-Laptop-None-Bed | 候选训练 | ✅ 修复成功 | replace×1 | 40 → 21 | 20 | 成功 | unknown | 1 | COMPLETE |

## 汇总

- **修复成功率 8/11（73%）**：
  - 3 个失败都是 heat/cool 类（Mug ×2、Apple），在 2 轮 × 3 个候选内没有找到成功修复。
  - 2 个 probe 都修复成功。
- **修改的局部性**：8 个成功修复中，7 个只有一个 replace 块，替换 1–15 个动作，中位数 3 个。
  - 编辑距离（8–36）偏大，主要不是改动多：base 的原轨迹在后面循环到 40 步上限，修复后在第 8–21 步就成功了，没执行到的后缀按文档 7.1 记为删除，计入距离。
- **联合必要性**：只有 1 个案例（pick_two Candle）是两个块、而且两块缺一不可。原轨迹失败；只加 p1 失败；只加 p2 失败；两块一起成功；撤销后恢复仍成功。
  - 文档 14.5 要求至少 2 个来自不同任务的联合必要性案例，这里不足，所以本分支**不检验联合修复的主张**。
- **标签**：8 个修复中 6 个的前缀一致 reasoning 标签通过了自动检查。2 个被标为 UNGROUNDED_TARGET，原因是 teacher 的说明提到了当时还没公开出现过的实例编号。
- **N/F 配对：0 对**。这一步卡住，原因有两个：
  1. **工程 bug（已在修）**：配对时的目标覆盖检查只认 "put X in/on Y"，而本 harness 放物体用的是 "move X to Y"，所以每条成功轨迹都被判为 public_goal_coverage_unknown。
  2. **规则冲突（需要你决定）**：base 学生原路线里有不少被环境拒绝的无效动作（"Nothing happens"）。Near 按最小修改保留学生原动作，这些无效动作也被保留下来；而文档 14.6 第 2 条要求配对的 target 不能包含被环境拒绝的动作。
     - 8 个修复里有 6 个的 Near 含 1–5 个这样的动作。
     - 即使修好 bug，K0 分支也凑不出 4 对，最多只有 1 对训练对（Statue）。
- **teacher 用量**：输入 216,590 token，输出 58,260 token，其中隐藏推理 45,320。K0 分支预算是 2.5M / 0.25M。

## 需要决定的问题

Near 里被环境拒绝的无效动作怎么处理：
- **A. 按文档原样**：这类 Near 不参与配对。K0 分支只报告 E1，不做 E2；E2 只在 U_mech 分支做（θ₀ 是 SFT 后的模型，无效动作少得多）。
- **B. 统一清理**：在 Near 和 F 两边都删掉被环境拒绝的无效动作，然后重新真实执行确认仍然成功。这些删除不算修复改动，编辑距离另报一列。
  - 好处：不让学生学无效动作。
  - 代价：Near 不再是"严格最小修改"。
