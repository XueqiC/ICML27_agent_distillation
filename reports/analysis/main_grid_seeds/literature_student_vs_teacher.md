# 文献核对：学生是否超过 teacher(2026-10-24,逐篇读原文)

来源：arXiv PDF 原文(pdftotext 后逐表核对),SmartAD 用 docs/baselines/smartad_paper_text.txt。
"teacher" 指生成训练数据、或被当作参照的那个模型;成功率均为论文原表数字。

## A. 我们主表的三个 baseline 原论文：学生都低于 teacher(或没报 teacher)

| 论文 | 任务 | teacher → 学生 | 训练数据 | teacher 分数 | 最好学生 | 出处 |
|---|---|---|---|---|---|---|
| AD (Kang et al., 2505.17612) | QA + 数学，最多 5 步 | Qwen2.5-32B-Instruct → Qwen2.5 0.5B/1.5B/3B/7B | 每题采 1 条 teacher 轨迹,**过滤掉错误的**,约 2,000 条;LoRA r64,2 epoch | Agent Prompting 平均 46.00 | 7B + ftp + sag 平均 42.68(低于 teacher) | §5 Training details;Table 2 |
| SAD (Liu et al., 2505.13820) | ALFWorld / WebShop / HotPotQA | GPT-2-1.5B → 120M/340M/760M;OPT-13B → 1.3B–6.7B;LLaMA-13B → 7B;Orca2-13B → 7B | teacher 低温采样(T=0.2),每题多条;原文没写按成功过滤;损失里用 teacher 分布(KL) | ALFWorld 71.2 / 76.5 / 75.3 / 78.1 | ALFWorld 64.8 / 67.1 / 68.0 / 70.5(全部低于 teacher) | Table 3、Table 4;附录 E.2 |
| SmartAD (Tang & Zhao) | QA + 数学 | Qwen2.5-32B-Instruct → Qwen2.5 1.5B/3B | 每题 10 条 T=1.0 候选，先**剔除错误的**,再选学生 NLL 最低的;LoRA 2 epoch | 主表 Table 2 **没有报告 teacher 分数** | 3B 平均 34.0 | §3.2;Table 2;附录 A |

共同点：学生比 teacher 小 2–64 倍，学生低于 teacher,论文也不和 teacher 比，只和其他蒸馏方法比。

## B. ALFWorld 上的 agent 训练论文：微调后的 7–8B 学生超过被提示的 GPT-4 类模型很常见

| 论文 | 参照的大模型(提示) | 学生 | 学生的训练数据 | 出处 |
|---|---|---|---|---|
| ETO (2403.02502) | GPT-4:seen 42.9 / unseen 38.1 | Llama-2-7B SFT 60.0 / 67.2;ETO 68.6 / 72.4 | 附录 A:ALFWorld 用数据集自带的轨迹,GPT-4 只补写每步的 rationale | Table 2;附录 A |
| IPR (2406.11176) | GPT-4:42.9 / 38.1(与 ETO 同数) | Llama-2-7B SFT 60.0 / 67.2;IPR 更高 | §4.1:"prompt GPT-4 to interact with the environment … filter … to retain only the correct trajectories" | §4.1;Table 2 |
| STeCa (2502.14276) | GPT-4:42.9 / 38.1(沿用) | Llama-2-7B SFT 60.0 / 67.2;STeCa 74.3 / 76.1 | 同上一系的专家轨迹 | Table 2 |
| WKM (2405.14205) | GPT-4:44.29 / 38.05 | Mistral-7B WKM 73.57 / 76.87(NAT、ETO 也都超过 GPT-4) | 训练集专家轨迹 | Table 2 |
| AgentGym (2406.04151) | GPT-4-Turbo:67.5 | Llama-2-7B BC_base 77.5;BC_large 83.0;AgentEvol 88.0 | ALFWorld:400 条 SOTA 模型(如 GPT-4-Turbo)轨迹 + 100 条人工，**按 reward 过滤** | Table 3;§3、附录 C |
| LEAP (2410.05434) | ReAct gpt-4o:in-dist 54.3 / OOD 65.7 | Llama3-8B BC π0 68.6 / 65.7;加特权 teacher 纠错后 π1–π3 90.0–96.4 / 83.6–91.8 | BC:数据集 3,257 个游戏的观测-动作演示,gpt-4o 只补 reason;之后对**失败 rollout** 用带特权状态的 gpt-4o 纠错 | Table 1;附录 |
| HAD (2610.02858) | 同族 teacher:Qwen3-8B unseen 51.5 / seen 50.0;Gemma-4-12B 51.5 / 49.3 | Qwen3-1.7B HAD 63.4 / 51.4(超过 teacher);其它 OPD 基线 31.3–47.0 / 35.7–41.4;Gemma-4-E2B 全部低于 teacher | on-policy 蒸馏，不用成功标签 | Table 2、Table 5 |

## 结论

1. **主表三个 baseline 的论文**:设定是大 teacher 配小 2–64 倍的学生，学生全部低于 teacher;SmartAD 甚至没报 teacher 分数。在这些论文的设定里，学生超过 teacher 不会出现，也没人讨论。
2. **ALFWorld 文献**:微调的 7–8B 学生超过提示下的 GPT-4 / GPT-4-Turbo / gpt-4o 很常见，差距通常比我们大(SFT/BC 学生比提示的大模型高 0 到 +29 个百分点,LEAP OOD 持平、ETO unseen +29;我们 K≥300 各格是 −2 到 +9)。
   - 要注意，这些论文多数的 SFT 数据不是那个大模型自己跑出来的:ETO 和 LEAP 的动作来自数据集的专家轨迹,GPT-4 只补写理由。
   - 和我们最接近的是 IPR(GPT-4 rollout,只留正确的)和 AgentGym(SOTA 模型 + 人工轨迹，按 reward 过滤)。两者都是只学成功轨迹的学生超过了生成数据的模型。
   - 它们的 teacher 分数是各自提示方式下的结果，往往不是同一 harness、同样步数。我们的 teacher 和学生在同一个 harness H0、同样 40 步下评测，比较更干净。
3. **明确以"超过 teacher"为卖点的两篇**:
   - LEAP:靠特权 teacher 对学生失败 rollout 的纠错;
   - HAD:靠 harness 对比的偏好信号。
   - 两者都是在只学成功轨迹之外加了额外信号。只做 BC / OPD 的基线，在 HAD 里低于 teacher,在 LEAP 里与 teacher 持平或略高。
4. **对我们论文**:
   - 在 ALFWorld 上，"学生超过 teacher"本身不是新发现，审稿人会认为它主要来自成功过滤和领域内专门化;
   - 我们要报 teacher 在同一 harness 下的 pass@1 / pass@3,作为模仿上限的参照;
   - 新方法的增益要建立在"超过收敛的轨迹 SFT"上。
