# Thinking 开 / 关 pilot（2026-10-04）

**设置**：base 学生 google/gemma-4-12B-it @ 707f0a3b，ReAct 格式（`Thought: … / Action: <命令>`），贪心解码，冻结的 dev40
（valid_seen，6 类分层，seed 0），每局最多 40 步，单步最多 4,096 token，hpg B200 单卡、8 并发。
两个条件只改 `enable_thinking`。作业 44733714（关）/ 44733715（开）。结果目录：hpg `reports/runs/pilot-base-dev40-{off,on}`。

## 结果

| | thinking 关 | thinking 开 |
|---|---|---|
| 成功 | **19 / 40（47.5%）** | **21 / 40（52.5%）** |
| 逐局配对 | 两个都成功 17，只有关成功 2，只有开成功 4，都失败 17 | |
| McNemar 精确检验（2 vs 4） | p = 0.69 | |
| 总步数 / 合法 / 不在列表 / 解析错误 | 1,147 / 1,072 / 66 / 9 | 1,173 / 1,093 / 49 / 31 |
| 出现解析错误的局数 | 4 | 5 |
| 成功局的平均步数 | 16.2 | 19.7 |
| 生成 token（可见 + thinking） | 73,217 | 364,080（约 5 倍） |
| thinking 文本字符数 | 73,682（模型自发进入 thought 通道） | 963,711 |
| 每局平均推理延迟 | 20.5 s | 91.1 s |
| 整体墙钟时间 | 3 分钟 | 15.5 分钟 |

按类型（关 → 开）：pick_and_place 7/7 → 7/7；look_at_obj 5/7 → 4/7；clean 3/6 → 2/6；cool 2/7 → 3/7；**two_obj 2/6 → 5/6**；heat 0/7 → 0/7。

## 观察

1. **开 thinking 没有明显更好**：只多 2 局，配对的差异是 4 比 2，p = 0.69，远在噪声之内；代价是约 5 倍 token、4.5 倍时间、3.4 倍解析错误。
2. **thinking 里是完整的计划**，可见的 Thought 只是一句摘要。例子（two_obj 任务第 1 步）：
   > thinking："The user wants me to put two toilet papers in a drawer. … Plan: 1. Go to toiletpaperhanger 1. 2. Pick up the toilet paper.
   > 3. Go to a drawer … 5. Repeat for the second toilet paper."
   > 可见回复："Thought: I need to find the toilet paper first. … Action: go to toiletpaperhanger 1"

   two_obj 上开 thinking 的提升（2 → 5）最大，看起来与"多步计划"有关，但每类只有 6–7 局，不能下结论。这一点与论文一的机制 2（学生需要更明确的决策信息）有关，值得之后作为受控对照再看。
3. **关着 thinking 时，模型也会自发进入 thought 通道**：关的那组 9 次解析错误全是这种情况。可见回复为空，模型一直写到 4,096 token 上限，全部出在 look_at_obj 类任务里，像是卡住时陷入长推理。开的那组 31 次解析错误也主要是写满上限、没有给出 Action。这是 harness 层面的问题，与方法无关。

## 结论（按事先约定的规则）

差异不明显 → **定为 ReAct + 关 thinking**，所有方法统一使用。

需要定一个 harness 规则来处理"自发 thinking 写满上限"：
- 建议：保持现在的处理（算解析错误、不推进环境、计入步数，并单独统计），同时把单步上限降到 1,024 token，减少时间浪费；每次评测都报告这类事件的比例。
- 备选：遇到写满上限且可见回复为空时，截断后再提示一次。但这会改变策略本身，不建议默认使用。
