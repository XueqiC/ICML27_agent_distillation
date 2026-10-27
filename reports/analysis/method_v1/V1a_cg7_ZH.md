# V1a 第一批结果：teacher 从 CG 的 7 个前缀接管

日期：2026-10-27（CDT）。
- 预注册：`docs/direction/2026-10-27_V1a_teacher_takeover_prereg_ZH.md`（118563b）。
- 代码：repair-v01-eval 分支 95e3947。
- 数据：`reports/runs/method_v1/v1a_cg7`（接管）和 `v1a_cg7_fromstart`（起点对照）。
- teacher 用量：接管 4.80M input / 50k output，起点对照 1.57M / 11k；V1a+V1b 累计 6.40M / 63k，上限为 70M / 8M。

## 结果

| 前缀 | CG 中学生 A 分支（找到 / 完成，共 8 次） | teacher 接管（成功 / 4） | 发现目标所需的后续步数 | 起点对照（成功 / 2） | 判定 |
|---|---|---:|---|---:|---|
| Newspaper | 6 / 0 | **3** | 2, 2, 5, 11 | — | 可行 |
| Apple | 4 / 0 | **4** | 3, 4, 5, 7 | — | 稳定可行 |
| Potato | 8 / 8 | 4 | 7, 7, 7, 7 | — | 稳定可行（天花板） |
| Kettle-Shelf | 1 / 0 | 2 | 27, 27, 30, 30 | — | 可行（靠穷举到最后几步） |
| Kettle-DiningTable | 0 / 0 | 0 | 未找到 | 0 | 不可行，起点也失败 |
| Lettuce | 0 / 0 | 0 | 未找到 | 0 | 不可行，起点也失败 |
| Egg | 0 / 0 | 0 | 未找到 | 0 | 不可行，起点也失败 |

28 + 6 次运行全部有效，前缀核对全部通过。

## 观察

1. **在学生失败的两个"中间"前缀上，teacher 能完成。** 这些前缀存在可教的后续行为。
   - **Newspaper**：3 次成功里，teacher 都先把第 2 步拿起的 creditcard 放到 dresser 上（`move creditcard 1 to dresser 1`），再拿报纸，然后去台灯处。canary 中它的 Thought 是 "the credit card may be occupying my inventory space"。学生在同一前缀上从未这样做（CG 的 D 分支 95 个非法动作，只成功 1/8）。
   - **Apple**：teacher 打开 1–3 个柜子后就转向台面，第 3–7 个后续动作在 countertop 2 找到苹果，4/4 完成冷却和放置。学生的 A 分支要到第 28 步才找到，0/8 完成。
2. **三个地板案例对这个 teacher 也是地板。**
   - Kettle-DiningTable、Lettuce、Egg 无论从前缀接管还是从起点开始，teacher 都在 40 步内没见到目标。
   - 这三个房间有 13–20 个柜子。teacher 同样按编号逐个开柜子，从前缀接管时还直接沿用了学生的开柜顺序。
   - 这类案例检验的是 teacher 自身的搜索能力，不能用来检验"教学"，按预注册不进入 V1b 的主比较。
3. **Kettle-Shelf 的成功来自穷举到最后。** 2 次成功都是第 27–30 步才找到，属于边缘可行。

## 对下一步的含义

- 可以进入 V1b 的前缀只有 Newspaper、Apple、Kettle-Shelf，外加 Potato 作天花板检查，案例数不够。需要先建 U_dev 诊断池，按预注册的决策点和 A 分支预筛扩充前缀，再统一跑 V1b。
- 搜索类困难里有一部分是 teacher 也不会的（大量同类容器、没有线索）。持续指导对这类前缀的价值有限，方法验证应以"teacher 可行、学生不可行"的前缀为主。
