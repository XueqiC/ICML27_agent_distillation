
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
