# teacher 能做、学生做不到的任务：原因，以及增加 K 补上了什么

日期：2026-10-26（CDT）。回答用户 14:03 CDT 的问题。

## 数据与定义

- **学生**：E4B 主网格的全部已评测格子，包括 4 种方法（轨迹SFT、AD、SmartAD、SAD）× K10/K20/K30/K100 × 训练 seed 0–2。
  - SAD K10 已经崩溃（8/140，低于 base 的 43），不计入，所以共 33 个格子。
  - 每个 K 的格子数：K10 8 个，K20 6 个，K30 10 个，K100 9 个。
- **teacher**：gpt-6-luna 单次运行，dev140 上做对 114 题。
- **dev140**：全部来自 valid_seen，按长期惯例用作 development 集。本分析只读评测日志，不回灌训练。
- **读取的内容**：逐题、逐格读 `episode.json` 里的动作、observation 和学生的 Thought。

### 失败形态（按 episode 中走到的最远一步分类）

| 代号 | 名称 | 判定 |
|---|---|---|
| A | 没找到 | 整个 episode 的 observation 里从未出现目标物体实例 |
| B | 看到没拿 | 看到了目标物体，但没有合法的 `take` |
| C | 拿了没处理 | heat/cool/clean 类任务里拿到了物体，却没有合法的变换动作；look 类任务里拿了但没用台灯 |
| D | 跳过处理 | 没有做变换，就把物体放到了目标位置 |
| E | 处理了没放 | 做了变换，但没放到目标位置 |
| F | 两物体未完成 | 两物体任务没完成 |

"合法"指环境没有返回 `Nothing happens.`。

## 结论先行

1. **"teacher 能做、所有学生 baseline 都做不到"的任务只在固定 K 下存在，而且随 K 增大迅速消失。**
   - K10 有 7 题，K20 有 5 题，K30 只剩 1 题，K100 为 0。
   - 不分 K 合起来看是 0 题：teacher 做对的 114 题中，每一题都至少被某个学生格子做对过。
   - 所以问题不在于学生"学不会"，而在于小 K 时"没学到"。
   - 换一个宽一点的口径，即该 K 下最多 1/3 的格子做对：K10 24 题，K20 25 题，K30 8 题，K100 5 题。
2. **做不到的直接原因可以分成四类。前三类随 K 增大明显减少，第四类基本不减少。**
   - (a) 不会这一类任务的操作规则。例如 heat/cool/clean 必须把物体拿在手里、在电器前执行变换，手里同时只能拿一件东西。
   - (b) 没见过的目标措辞。例如 `put a hot X in Y`，在 K≤30 的训练集里一次都没出现过。
   - (c) 环境拒绝了动作（返回 `Nothing happens.`）以后，学生不更新判断，继续当作动作已经生效。
   - (d) 搜索策略固定：按编号逐个穷举同一类容器，不根据物体类型或已有观察选择位置。
3. **K 的作用主要是补齐每类任务的示范数和措辞覆盖，而不是让学生记住具体房间的布局。**
   - 从 K10 到 K100，在 teacher 做对的任务上：
     - 操作类失败（C+D+E+F）占全部 episode 的比例从 19% 降到 6%；
     - "没找到"只从 10% 降到 5%；
     - 失败 episode 中出现"环境拒绝了对目标物体的动作"的比例从 38% 降到 12%。
   - 被 K 救回来的 7 题里，没有一题在 K100 训练集中有"同一房间、同一物体"的示范。
4. **K 解决不了搜索策略。** K100 剩下的失败中，"没找到"占 41%，是最大的一类。
   - 有的题无论 K 多大，学生都把 21 个抽屉逐个开完，没有时间查别处。
   - 有的题 K 越大反而越差：K10/K20 先查台面能做对，K100 改成先逐个开柜子，于是做不对了。

## 1. 严格口径：该 K 下所有学生格子都做不对、teacher 做对的任务

| K | 题数 | 任务（目标原文） | 主要失败形态（该 K 所有格子合计） |
|---|---:|---|---|
| 10 | 7 | heat-Egg-Fridge-14（put a hot egg in fridge） | 拿了没处理 4，看到没拿 2 |
| | | heat-Plate-DiningTable-28（put a hot plate in diningtable） | 看到没拿 8/8 |
| | | heat-Potato-Fridge-12（put a hot potato in fridge） | 没找到 4，跳过处理 2 |
| | | heat-Tomato-DiningTable-16（heat some tomato and put it in diningtable） | 没找到 8/8 |
| | | heat-Tomato-Fridge-23（heat some tomato and put it in fridge） | 拿了没处理 5 |
| | | two-Pan-CounterTop-14（find two pan and put them in countertop） | 两物体未完成 5 |
| | | two-Spatula-Drawer-13（put two spatula in drawer） | 没找到 8/8 |
| 20 | 5 | clean-ButterKnife-DiningTable-16（put a clean butterknife in diningtable） | 没找到 4 |
| | | heat-Mug-SideTable-21（put a hot mug in sidetable） | 看到没拿 3，没找到 3 |
| | | heat-Plate-DiningTable-28 | 看到没拿 6/6 |
| | | heat-Potato-Fridge-12 | 拿了没处理 3，没找到 2 |
| | | two-Pan-CounterTop-14 | 两物体未完成 6/6 |
| 30 | 1 | heat-Plate-DiningTable-28 | 看到没拿 7，拿了没处理 3 |
| 100 | 0 | — | — |

- 13 个条目中有 9 个是 heat 任务。
- K10 时 heat 只有 1 条示范，K20 时有 2 条（见第 3 节）。

## 2. 四类原因，每类附具体轨迹

### (a) 不知道这一类任务的操作规则

**heat：物体放在微波炉里时，heat 命令不可用。**
- 例子：heat-Apple-DiningTable-26，AD K20，第 27–39 步。学生拿到苹果后：
  1. `move apple 1 to microwave 1`，然后 `close`，再 `open`；
  2. 接着执行 `heat apple 1 with microwave 1`，环境返回 `Nothing happens.`；
  3. 学生 Thought："Heating didn't change the state, so I'll take the apple out and reheat it"，然后又把苹果放回微波炉，循环直到步数用完。
- 正确做法是拿着物体站在微波炉前，直接执行 heat。
- 这个"先放进去、关上、打开"的绕路来自 teacher 示范本身：
  - K20 的 2 条 heat 示范里，有 1 条先把物体放进微波炉、开关几次，再拿出来加热。teacher 最后总会"拿出来再 heat"，所以示范是成功的。
  - 只有 1–2 条示范时，学生学到的是绕路的表面动作，没学到"必须拿在手里"这个关键条件。
  - K100 的 11 条 heat 示范里，7 条是直接拿着加热，学生也随之学会了。

**cool：把"放进微波炉"误当成变换。**
- 例子：cool-Apple-Microwave-5（cool some apple and put it in microwave），K10 的失败。
- 学生拿到苹果，直接去微波炉，打开，放进去，再尝试 `heat apple`（Thought："I'll open it before heating the apple"）。
- K10 只有 1 条 cool 示范、1 条 heat 示范。学生把"微波炉"和"加热"绑在了一起，没区分"目标位置是微波炉"和"变换用的电器是冰箱"。
- 这题在 K10 的成功率是 1/8，K100 是 8/9。

**两物体：手里同时只能拿一件。**
- 例子：two-Pan-CounterTop-14，轨迹SFT K20 与 SmartAD K20 的失败，第 16–23 步：
  1. 拿起 pan 1 以后，又执行 `take pan 2 from stoveburner 4`，环境返回 `Nothing happens.`；
  2. 学生 Thought："I have both pans"；
  3. 放下 pan 1 后，又以为 pan 2 也已经放好（"Both pans are on countertop 1; the task is complete"），接着到处找，直到步数用完。
- SmartAD K100 的成功轨迹：
  1. 看到 pan 2 后先记住位置；
  2. 回到台面放下 pan 1；
  3. 再回去拿 pan 2。
- 这题在 K10/K20 的 14 个格子里全部失败，K30 有 6/10 成功，K100 有 6/9。

### (b) 没见过的目标措辞

训练集中 heat/cool 两种措辞的示范数：

| 措辞 | K10 | K20 | K30 | K100 |
|---|---:|---:|---:|---:|
| `heat some X and put it in Y` | 1 | 2 | 3 | 7 |
| `put a hot X in Y` | **0** | **0** | **0** | 4 |
| `cool some X and put it in Y` | 1 | 1 | 2 | 7 |
| `put a cool X in Y` | **0** | 2 | 2 | 6 |

dev140 上按措辞分组的学生成功率（所有任务、所有格子的平均）：

| 类型 | 措辞 | 题数 | teacher | K10 | K20 | K30 | K100 |
|---|---|---:|---:|---:|---:|---:|---:|
| heat | heat some X | 6 | 5/6 | 0.21 | 0.44 | 0.52 | 0.81 |
| heat | **put a hot X** | 10 | 7/10 | **0.10** | **0.17** | **0.30** | 0.47 |
| cool | cool some X | 11 | 7/11 | 0.41 | 0.38 | 0.48 | 0.61 |
| cool | **put a cool X** | 14 | 8/14 | **0.18** | 0.39 | 0.46 | 0.62 |
| clean | clean some X | 12 | 11/12 | 0.94 | 0.88 | 0.84 | 0.82 |
| clean | put a clean X | 15 | 8/15 | 0.47 | 0.53 | 0.52 | 0.62 |
| two | put two X | 16 | 15/16 | 0.50 | 0.68 | 0.76 | 0.90 |
| two | find two X（训练集中 0 条） | 8 | 6/8 | 0.42 | 0.58 | 0.60 | 0.81 |

- **训练集中没出现过的措辞，成功率一直偏低**。
  - `put a cool X` 在 K10 时 0 条示范，成功率 0.18；K20 补到 2 条后升到 0.39。
- **最典型的例子是 heat-Plate-DiningTable-28（put a hot plate in diningtable）**。
  - 学生把 "hot plate" 读成一种物体（电热板），而不是"要加热的盘子"。
  - 轨迹SFT K100：在第 9 步打开 cabinet 5，看到 `plate 3`；Thought 是 "Cabinet 5 contains a plate, not the hot plate; I'll continue searching"。之后又先后看到 plate 4、plate 2、plate 1，一个都没拿，一直找到第 40 步。
  - 这题在 K10/K20/K30 的 24 个格子里全部失败，K100 只做对 1/9。是唯一一题直到 K30 仍然所有格子都失败。
  - teacher 第 9 步看到 plate 3，第 10 步就拿了。
- clean 例外：`clean some X` 在 K10 时只有 1 条示范，成功率却已经是 0.94，K 增大后没有变化。

### (c) 环境拒绝动作后不更新判断

- 统计范围：在 teacher 做对的任务上，学生失败的 episode 里，至少有一次"对目标物体的动作被环境拒绝"。这里的动作指 take/move/heat/cool/clean/use 目标物体。

| K | 失败 episode | 其中出现过对目标物体的动作被拒绝 | 被拒绝后下一条 Thought 仍声称已生效（粗略匹配，是下界） |
|---|---:|---:|---:|
| 10 | 330 | 127（38%） | ≥51 |
| 20 | 191 | 56（29%） | ≥17 |
| 30 | 296 | 75（25%） | ≥21 |
| 100 | 128 | 15（12%） | ≥1 |

- 上面的 heat 和两物体例子都属于这一类：observation 已经否定了学生的假设（"heat 生效了"，"我拿着两个"），学生仍按原计划继续。
- K 增大后，这个比例下降，主要是因为学生更少做出会被拒绝的动作，即规则学会了；并不是学会了"被拒绝之后怎么办"。
- 现有数据没法把这两种情况分开。

### (d) 搜索：固定的穷举顺序

- **按物体类型选择位置，这种先验会随 K 学到一部分**。
  - 例子：heat-Tomato-DiningTable-16。
  - K10 的 8 个格子都没去过冰箱，其中 7 个从 cabinet 1 开始逐个开柜子。番茄就在冰箱里，结果"没找到"8/8。
  - K20 的 6 个格子中有 5 个第一步就去冰箱；K100 做对 8/9。
  - K20 比 K10 多了 2 条 cool 示范，示范里都要去冰箱。
- **"同一类容器从 1 号开始逐个查"这个习惯，K 改变不了**。
  - 例子：pick-Box-Dresser-224。
    - 这个房间有 21 个抽屉，盒子在 sidetable 2 或 sofa 1 上。
    - teacher 先后查了 cabinet、armchair、sidetable、两个抽屉和 sofa，第 7 个动作就在 sofa 上找到了。
    - 学生无论 K 多大，都从 drawer 1 开始逐个查抽屉，前 14 次移动都在抽屉之间。这题在 K100 只做对 2/9。
  - U_mech 里 θ₀ 的报纸案例也在同一个户型（FloorPlan 224），同样是逐个开抽屉。
- **K 增大也可能把顺序改坏**。
  - 例子：clean-Fork-Drawer-8。
  - K10/K20 的格子先查台面，叉子在 countertop 3 上，所以做对 7/8 和 6/6。
  - K100 的多数格子改为从 cabinet 1 开始逐个开柜子，只做对 2/9。
  - 学到的只是一个平均意义上的"先查哪里"，与当前房间里的观察无关。

## 3. 增加 K 具体补上了什么

| 证据 | 数字 |
|---|---|
| 每类任务的示范数（K10 → K100） | heat 1 → 11，cool 1 → 13，clean 2 → 17，两物体 2 → 24，look 1 → 10，simple 3 → 25 |
| 操作类失败 C+D+E+F 占全部 episode 的比例（K10 → K100） | 19% → 6% |
| "没找到"占全部 episode 的比例（K10 → K100） | 10% → 5% |
| 失败中"对目标物体的动作被拒绝"的比例（K10 → K100） | 38% → 12% |
| 被 K 救回的 7 题中，K100 训练集里有"同一房间、同一物体"示范的 | 0/7；其他 teacher 能做的题是 11/102 |
| 被 K 救回的 7 题中，K100 训练集里有同一房间示范的 | 3/7；其他 teacher 能做的题是 63/102，两者相近 |
| 被 K 救回的 7 题的任务类型 | heat 2、cool 2、两物体 2、simple 1 |

"被 K 救回"的定义：在 K10 和 K20 下最多 1/3 的格子做对，在 K100 下至少 2/3 的格子做对。共 7 题：Newspaper-Sofa-224、cool-Apple-Microwave-5、cool-Tomato-Microwave-18、heat-Potato-Fridge-11、heat-Tomato-Fridge-15、two-Pan-CounterTop-14、two-SprayBottle-Cabinet-407。

**解释**：
- 增加 K 带来的主要是每类任务更多"关键步骤做对了"的示范，例如拿出来再加热、放下第一件再拿第二件、cool 去冰箱，以及更多种目标措辞。学生由此学会了这一类任务的规则。
- 增加 K 带来的不是对具体房间的记忆。救回的题几乎没有同房间、同物体的示范。
- 搜索方面，K 只带来一个平均意义上的"先查哪里"的先验。它帮助了一些房间，也损害了另一些房间（Fork-8）。学生没有学会根据已经排除的位置和当前观察来选择下一个位置。

## 4. 对新方法的含义

- **(a)–(c) 正是用户新设计中的"条件"失败**，分别对应三种条件状态：
  - 前提尚未满足：没把物体拿在手里；
  - 前提已被观察否定：动作返回 `Nothing happens.`；
  - 目标理解有误："hot plate" 被当成一种物体。
- 主网格的数据表明，靠增加示范来修这些问题，需要把 K 提高 5–10 倍。这几类失败形态都很固定，有希望用少数失败案例加上条件提示或针对性练习来补。
- **(d) 搜索是 K 修不好的那部分。**
  - 它需要学生根据"哪些位置已经排除"选择下一次观察，这正是今天条件提示小实验的 D 分支要测的内容。
  - 小实验的 7 个案例中，有 1 个与 Box-224 是同一个 21 抽屉的户型。
- **注意**：
  - 失败形态由动作字符串和 observation 用规则判定。"Thought 仍声称已生效"用的是粗略的正则匹配，只能作为下界。
  - teacher 只跑了一次。
  - 每个 K 的格子数不同（6–10 个），成功率是对格子的平均，不是独立样本。
