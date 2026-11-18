# C2b：teacher 用语言写规则单，放进 system prompt

状态：定稿（11-18 01:45 CST）。预注册 §18（C2b，我自己加的做法）。数据：`data/c2b/`（三版规则单原文、每版的调用记录、用量台账）。

## 1. 要回答的问题

"示范放进 prompt"没有效果（见 `N_small_k_ZH.md`）。另一种不改权重的做法是：**让 teacher 读学生的失败，把该注意的事写成一份规则单，接在 system prompt 后面。** 三个问题：

1. 规则单本身有没有用？
2. 给 teacher 看上一版的效果，让它改，会不会更好？
3. 效果够不够大，值不值得把它编译进权重？（门槛：dev140 ≥ 59）

## 2. 名词

- **规则单**：一段不超过 400 个英文词的文字，接在 system prompt 原文之后。其余不变，不改权重。
- **dossier**：给 teacher 读的材料，25,476 个字符，只含训练侧的内容：原始学生在 20 个训练题上的整条轨迹、同题的 teacher 轨迹、失败统计。
- **进度分**：见 `B4_multi_student_ZH.md` §2。
- **teacher**：gpt-6-luna，黑盒调用。

## 3. 怎么做的

| 步 | 输入 | 做法 | 输出 |
|---|---|---|---|
| 写 v1 | dossier | teacher 一次调用 | 规则单 v1（316 个词） |
| 评 v1 | 原始 E4B + v1 | 20 个训练题整题 rollout；dev140 | 训练题逐题成败、进度分、五个行为统计 |
| 写 v2 | dossier + v1 在训练题上的结果（逐题成败、进度分、行为统计、失败题第一处分歧） | teacher 一次调用 | 规则单 v2（324 个词） |
| 写 v3 | 同上，换成 v2 的结果 | teacher 一次调用 | 规则单 v3（330 个词） |
| 选一版 | 三版在训练题上的进度分 | 进度分最高的是选中版 | v1 |
| 对照臂 | v1 + 每题 2 条同类型示范 | dev140 一次 | 看两者是否叠加 |

teacher 用量：3 次调用，输入合计 35,071 token（上限是 3 次、30 万 token）。评测在 LONI 的 A100 上。

**规则单 v1 的前四条**（原文，全文在 `data/c2b/rulesheet_v1.txt`）：

```
- Parse the goal into the required object, quantity, condition or treatment, and destination. Track which parts are complete.
- Choose the next action that directly advances an incomplete part of the goal. Prefer moving to a relevant place over
  using look, help, or inventory when the observation already gives enough information.
- Copy exactly one command from the current admissible_commands list. Never invent, paraphrase, or combine commands.
- Use the object names and indices shown in observations. Do not guess an index or assume an object is present, carried,
  or inside a container.
```

## 4. 结果

| | dev140 | 训练题（20）做对 | 训练题进度分 |
|---|---|---|---|
| 原始模型 | 42 / 43（30%） | 10 | 0.811 |
| 规则单 v1（选中版，评两次） | **51 / 55（38%）** | 10 | 0.785 |
| 规则单 v2 | 44 | 8 | 0.745 |
| 规则单 v3 | 47 | 8 | 0.777 |
| 规则单 v1 + 每题 2 条同类型示范 | 41 | | |

## 5. 结论

1. **v1 有效。** 两次平均 53 对 42.5，约多 10 题，超出同一模型评两次的波动（0–6 题）。这是所有"不改权重"的做法里唯一确实有效的。
2. **让 teacher 看反馈再改，没有变好。** v2 是 44，v3 是 47，都低于 v1。
3. **再加示范，增益被抵消。** v1 加同类型示范是 41，回到原始模型的水平。
4. **没到门槛。** 最好的 55 低于 59，按预注册**不做编译**。规则单的增益约是 SFT 增益的五分之一（10 对 47）。
5. **训练题上的信号没有预测出 dev140 上的效果。** 三版在训练题上的进度分都**低于**原始模型（0.785、0.745、0.777 对 0.811），做对题数也不多于原始模型；而 v1 在 dev140 上多 10 题。按训练题选出 v1 是选对了（三版里它最高），但如果用"训练题上要高于原始模型才采用"这种验收规则，三版都会被拒掉。这对所有"用 20 个训练题做验收"的做法都是一个提醒，teacher 会话也在其中。

对新方法的含义：**teacher 的价值在"提炼"**。同样是 K20 的信息，原样放进 prompt 没用，提炼成一份三百来词的规则单就有约 10 题。这和数据侧（v0）里"teacher 给最小修改"最有用是一致的。规则单可以作为数据侧方法的一个部件（例如作为生成修复数据的依据），而不是作为取代 SFT 的方法。

## 6. 没做的和局限

- v2、v3 和对照臂各只评了一次 dev140。v2 的 44 和原始模型的 42 / 43 在噪声内；v3 的 47 介于两者之间，不能说它比原始模型好或比 v1 差得很确定。
- 只做了三版，teacher 每次只看上一版的结果。没有试"同时看三版的结果再写"，也没有试更长的规则单。
- 规则单是对着原始 E4B 写的。没有试"SFT 之后的学生 + 规则单"。
- 编译这一步没有做，所以"规则能不能进权重"这一轮没有答案。
