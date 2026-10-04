# Agent 蒸馏 baseline 调研（2026-10-04，按"只跑有官方代码的方法"规则）

检查了 30 多个方法：读了每个仓库的文件树、README 和训练脚本，关键处对照了论文。
BB = 只需要教师采样出来的文本或轨迹，能用黑盒教师。

## 可用（有官方代码）

| 方法（会议） | 官方仓库 | 原生环境 | 方法一句话 | 需要什么 | 训练框架 / 换成 Gemma 4 的难度 | 备注 |
|---|---|---|---|---|---|---|
| **ETO**（ACL 2024） | Yifan-Song793/ETO | ALFWorld、WebShop、SciWorld | 先 SFT，再让学生探索，用"失败 vs 成功"轨迹对做 DPO，迭代 | 专家轨迹 + 环境 rollout，BB ✓ | FastChat 分支，中等难度（torch 2.1 被钉死；label mask 只适配 Llama-2；FSDP 层类要改） | 它的 SFT 阶段就是标准的 BC 基线 |
| **IPR**（EMNLP 2024） | WeiminXiong/IPR | ALFWorld、WebShop、InterCodeSQL | 沿专家轨迹用 Monte-Carlo 估计步级奖励，构造对比步，DPO+SFT 混合，迭代 | 专家轨迹 + 大量学生 rollout，BB ✓ | 和 ETO 同一套框架，中等难度 | rollout 开销大 |
| **LEAP**（ICLR 2025） | sanjibanc/leap_llm | ALFWorld、WebShop、InterCode | 学生 rollout，**能看到特权状态的教师**改写学生的推理和动作，再 SFT 或 DPO，迭代 | API 教师的文字纠正 + 环境 rollout，BB ✓ | TRL，**最容易换** | 和我们的设定最接近；但教师用了特权状态，训练时额外用了信息，必须披露 |
| **AgentGym：BC / AgentEvol**（ACL 2025） | WooooDyy/AgentGym | ALFWorld、SciWorld、WebShop 等 14 个 | BC，再做"探索—学习"式自我进化 | 轨迹 + 环境 rollout，BB ✓ | 自写的 Accelerate trainer，容易（加一个 chat template 类即可） | 唯一还在持续维护的候选 |
| **WKM**（NeurIPS 2024） | zjunlp/WKM | ALFWorld、WebShop、SciWorld | 从专家轨迹和采样轨迹对比中总结任务和状态知识，训练一个单独的知识模型来指导 agent | 专家轨迹 + 选中 / 拒绝的轨迹对，BB ✓ | 老版 LLaMA-Factory LoRA，容易到中等（论文测过 Gemma-7B） | 推理时要两个模型；**和我们的机制 1"知识"直接相关** |
| QLASS（ICML 2025） | Rafa-zy/QLASS | ALFWorld、SciWorld、WebShop | SFT，再训一个 Q 网络，测试时按 Q 值逐步搜索 | 专家数据 + rollout，BB ✓ | DeepSpeed + sglang，中等 | 测试时多花算力，还要第二个模型，算力要单独报 |
| STeCa（Findings ACL 2025） | WangHanLinHenry/STeCa | ALFWorld、VirtualHome | 用 MC 步级奖励找出偏离的步，GPT-4o 反思后改写轨迹，再训练 | 专家轨迹 + rollout + 反思文本，BB ✓ | IPR 那套框架 | **代码不完整**：缺 SFT 脚本和反思脚本，忠实运行存疑 |
| Prefix-GRPO（arXiv 2607.19395） | HappynessI/Prefix_GRPO | ALFWorld、TextCraft、BabyAI | 在环境里重放教师前缀，对学生的续写做 GRPO | 教师轨迹，BB ✓ | verl + AgentGym，中等 | RL 类 baseline；会议未核实 |
| Kang Agent Distillation（NeurIPS 2025） | Nardien/agent-distillation | **只有 QA / 数学** | 在教师的代码 agent 轨迹上 SFT + 首思考前缀 + 自洽动作生成 | 教师轨迹，BB ✓ | smolagents + TRL，容易 | 按你的要求改造到 ALFWorld，所有改动逐条登记 |

## 不可用

- **没有训练代码**：AgentTuning（只有数据和权重）、Agent-FLAN、TinyAgent、Agent-R。
- **没有代码，或只有占位仓库**：SmartAD、AgentRefine、EEF、OVCSD、ATLaS、SAND。
- **需要教师 logits**：SAD（论文的损失是 KL(p_T‖p_S)，黑盒教师下没法忠实复现，且 v5 仍写着"录用后发布代码"）、Guided-OPD、ATOD、DGPO、SOD。
- **不是蒸馏或不合适**：
  - KnowAgent、AutoAct：自训练，没有教师。
  - MPO：只有数据构造和评测，没有训练代码。
  - NAT、FireAct、Lumos、SCoRe：不是 ALFWorld，依赖外部搜索 API。
  - Distilling step-by-step：只支持 T5，不是 agent。
  - SwiftSage：测试时仍要调用大模型。
  - KnowSelf：发布的流程要调用 DeepSeek-V3，与"不用中国模型"冲突。

## 共同的工程问题

没有一个仓库能原样在 gemma-4-12B 上跑：2024 年的仓库都钉死了老版本的 torch / transformers，加载不了 Gemma 4，也跑不了 B200（需要 cu128）。
做法是保留它们的算法代码，只改环境依赖、chat template、label mask 和 FSDP 层类，每一处改动登记为工程改动。
它们公开的 ALFWorld 数据是 planner 动作加 GPT-4 的推理，统一换成我们教师的数据。

## 推荐（按优先级）

1. BC / SFT：通过 ETO 的 SFT 阶段或 AgentGym BC 来跑，作为必需的参照行。
2. ETO
3. IPR
4. LEAP（训练时教师看特权状态，单独标注）
5. AgentGym AgentEvol
6. WKM
7. Kang AD（改造到 ALFWorld）
8. 可选：QLASS（测试时算力单独报）

再加上我们自己的对照组：Seed-SFT、grounded Augmented-SFT、DAgger 式纠正。
