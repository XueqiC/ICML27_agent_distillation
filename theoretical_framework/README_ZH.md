# 面向 Agent 蒸馏的学习器相对经验充分性：理论框架

[English](README.md)

**核心问题.** 给定声明的部署任务、允许的实验和实际使用的学习过程，反馈来源之间的哪些差别会改变蒸馏后学生的最终任务表现？

**中心结果.**

- **Theorem A（§6，learner-relative information bound）.** 对任何蒸馏器（包括 replay、改写、合成），最小可达的一阶误差由示范与查询关于一个学习器相关泛函的 Fisher information 决定。这个泛函的梯度 $`c=\sum_tJ_t^{*}g_t`$ 把学习器的 Jacobian 与任务的 continuation 梯度合在一起。在可辨识性与增长条件下，这个界在常数因子内可以由正则化的 Elfving c-optimal 查询设计达到。因此，同一批示范可以对一个学习器充分、对另一个不充分。
- **Theorem B（§7，Adam 的二阶矩通道）.** 在 warm Adam 下，一条消息的一阶价值精确分解为参数通道、动量通道与二阶矩通道。平稳情形下，相对幅度为 $`\lambda`$ 的消息的 response 有精确的 mean-field 形式：小增量时它是凹的二次函数，在显式的 $`\lambda^*`$ 处最大，在 $`2\lambda^*`$ 附近变负。只看参数步的分数和只看方向的分数因此都是不一致的排序器。
- **Theorem C（§8，可审计的证书）.** 对一轮训练做 common random numbers（CRN）配对替换，可以直接测得一个反馈律的有限效应。由此得到的证书既不需要 continuation value 的梯度，也不需要曲率常数，半径只含统计项与 surrogate 校准项。

这三条建立在 §2–§5 的基础上：
- regret 的精确分解，以及获取缺失信息的代价；
- 以 continuation response 作为充分性的对象；
- 充分性的复杂度；
- 每种 optimizer 让反馈必须携带的统计量。

§9 由以上全部结果推出方法 CED。

**怎么读这一页.** 本页按"先整体、后局部"的顺序展开，类似广度优先搜索。

| 层 | 内容 | 读完能回答 |
|---|---|---|
| [第 0 层](#第-0-层--一段话) | 一段话说清整个框架 | 框架要回答什么问题，分几步回答 |
| [第 1 层](#第-1-层--八个部分) | 八个部分及其衔接 | 每一部分回答什么问题、得到什么结论、交给下一部分什么 |
| [第 2 层](#第-2-层--主要结果) | 每个主要结果的白话解释 | 每个结果说了什么、直觉是什么、在链条里起什么作用 |
| [第 3 层](#第-3-层--各节的精确陈述) | 精确陈述与推导要点 | 确切的假设与公式，以及它们是怎样证明的 |

记号统一在 [§1](#1-设定与记号) 中给出。§2–§9 各有一个**精确陈述**页（陈述与推导要点）、若干**证明**页和**实验**页（较长的节分成几页，以保证每页都能在 GitHub 上完整渲染）：
- 证明页给出完整推导；
- 实验页给出检验各陈述的检查，写明做这个检查的原因、设置、结果，以及结果为什么支持该陈述。

所有测量结果都在实验页上；本页只有陈述和说明性的例子。

**范围.**
- 本页每个陈述都在其所列假设下有证明。少数陈述只在明确给出的渐近区间内证明到首项，本页在相应位置注明。
- 证明都是有限维的，或者带有明确的正则性假设。
- 实验页上的检查是精确有限计算或受控小网络。
- Theorem A–C 在真实模型上的检验已经设计好（[Remark 9.4](proofs/09_mechanism_and_method_ZH.md#s9-4)），尚未运行。

---

## 第 0 层 · 一段话

蒸馏一个 agent 时，手里有：
- 一个预训练学生；
- 每个任务族 $`K`$ 条可信的 teacher 示范；
- 一份允许做的额外实验预算（teacher 调用、环境分支、reset）；
- 一个固定的训练配方（optimizer、学习率、步数）。

可以喂给学生的反馈来源很多：示范本身、replay、合成改写、新的查询、同一批单元的不同打包方式。框架要回答的是：**这些反馈来源之间的哪些差别，会真正改变蒸馏后学生的最终任务表现？** 回答分八步。

1. **数据能不能确定该怎么做（§2）.**
   - 学生的 regret 精确拆成表示、信息、算法三项。
   - 信息缺口只能靠新的实验补上，replay 和改写补不了；即使 teacher 处处最优，也可能留下信息缺口。
   - 补上缺口大约需要 $`\log(1/\delta)/\Delta^2`$ 次查询；示范与查询共用一份信息预算。
2. **反馈对学习器做了什么（§3）.**
   - teacher 质量、策略价值、梯度均值都不能决定学生学到什么。
   - 不变量是 *continuation response*：收到一条消息、执行实际更新、再把训练走完之后的期望最终回报。
3. **两个来源需要在多少统计量上一致（§4）.**
   - 恰好是 response 类张成的空间。对平均化学习器，这是一个由部署长度决定的矩阶。
   - 近似复杂度可以远低于精确阶；非线性更新会把阶数相乘。
4. **实际的 optimizer 让反馈必须携带什么（§5）.**
   - 在一阶意义下，起作用的是*完整学习器状态*（参数加 optimizer state）的平均增量。这个一阶表示是工具，不作为新结论。
   - 对 SGD、momentum、cold 与 warm Adam、裁剪、Lion、LoRA，这个统计量都可以显式写出。warm Adam 的精确统计量是 Adam 增量的均值，常见的"均值加二阶矩"描述只在冻结分母近似下成立，误差显式。
   - Adam 坐标的 warmth 只在训练早期按 $`1/(\beta_2s)`$ 下降，之后停在 $`(1-\beta_2)/\beta_2`$。
5. **需要多少信息、投在哪个方向（Theorem A，§6）.**
   - 学习器决定数据必须在哪个方向 $`c`$ 上有信息。任何蒸馏器的 minimax 误差，在常数因子内等于 learner-relative information norm。
   - 推论：面对混叠时 replay 价值为零；示范与查询按一个学习器相对的兑换率互换；查询按 water-filling 分配；对同一任务，SGD 与 Adam 需要的数据可以差别很大。
6. **一条消息如何通过 Adam 的二阶矩起作用（Theorem B，§7）.**
   - 一条消息还会通过 $`v`$ 改变此后所有步长。
   - 由此得到幅度定律：存在最优幅度；小增量时，在约两倍最优幅度处变号。只看参数的分数与余弦分数在一个显式区域内排错。
7. **怎样用有限测量为选择出具证书（Theorem C 与 §8）.**
   - 有限 audit 需要覆盖（transport）项；在实际状态上做全尾段 audit 是安全的，但代价高。
   - 一阶分数需要 continuation 梯度；只有一阶项超过二阶余项时才能认证一个候选对，平均增量沿梯度方向相同的候选对永远认证不了。
   - CRN 配对替换直接测有限效应。还剩两个开放问题：一个远小于先验区间的可证明区间界，以及一个可检验的 surrogate 校准迁移条件。
8. **方法（§9）.**
   - 一个有严格零假设的机制检验（coupling width）。
   - Consequence-Calibrated Experience Distillation（CED）：按学习器相对的设计获取信息，按经过认证的实测后果选择反馈，匹配学习器的统计量，计入覆盖，冻结之后再检验迁移。

---

## 第 1 层 · 八个部分

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 28, "rankSpacing": 36, "padding": 10, "wrappingWidth": 260}}}%%
flowchart TB
  subgraph P1["第一部分 · 数据与反馈决定了什么"]
    direction LR
    S2["<b>§2 可行性<br/>与信息获取</b><br/>数据能否确定<br/>该怎么做？"]:::b1
    S3["<b>§3 学习后果</b><br/>反馈让学生<br/>学到什么？"]:::b2
    S4["<b>§4 充分性的复杂度</b><br/>需要匹配<br/>多少个统计量？"]:::b3
    S2 --> S3 --> S4
  end
  subgraph P2["第二部分 · 实际学习器需要什么"]
    direction LR
    S5["<b>§5 Optimizer state</b><br/>对真实 optimizer，<br/>反馈必须携带什么？"]:::b4
    S6["<b>§6 Theorem A</b><br/>需要多少信息，<br/>投在哪个方向？"]:::b4
    S7["<b>§7 Theorem B</b><br/>消息如何通过<br/>Adam 的 v 起作用？"]:::b4
    S5 --> S6
    S5 --> S7
  end
  subgraph P3["第三部分 · 如何验证与使用"]
    direction LR
    S8["<b>§8 验证</b><br/>Theorem C：用有限<br/>测量给出证书"]:::b5
    S9["<b>§9 机制与方法</b><br/>CED"]:::b6
    S8 --> S9
  end
  P1 --> P2 --> P3
  style P1 fill:#fafafa,stroke:#999
  style P2 fill:#fafafa,stroke:#999
  style P3 fill:#fafafa,stroke:#999
  classDef b1 fill:#e8f1fb,stroke:#2f6db5,color:#111
  classDef b2 fill:#eaf6ec,stroke:#2e8540,color:#111
  classDef b3 fill:#fdf3e1,stroke:#b8741a,color:#111
  classDef b4 fill:#f6e9f4,stroke:#9c3d8f,color:#111
  classDef b5 fill:#e9f5f6,stroke:#1f7f8c,color:#111
  classDef b6 fill:#fbeaea,stroke:#b53b3b,color:#111
```

箭头表示论证的主线。其他依赖：
- §2 给出 Theorem A（§6）所用的获取预算。
- §3 给出 §8 通篇使用的 continuation 分解。
- §4 给出 §8 使用的近似宽度与系数稳定性。
- §5 给出 §8 要测量的统计量。

### §2 · 可行性与信息获取

**问题.** 在学生能执行的策略里，现有数据能否确定该怎么做？补上缺失的信息要花多少代价？

**结论.**
- 学生相对最优参考策略的 regret 精确拆成表示误差、信息误差、算法误差三项，三项都非负，彼此不能替代。
- 信息误差为零，当且仅当与数据相容的所有环境有一个共同的最优策略。任何与环境无关的数据后处理（replay、改写、合成）都不能降低它。
- 即使 teacher 在每个环境里都最优，也可能留下信息缺口。reset 查询这类主动实验可以补上它，代价是 $`\log(1/\delta)/\Delta^2`$ 量级的查询次数，这个量级既充分又必要。把获得的事实编译进学生是另一项、小得多的代价。
- 在隐藏事实模型中，$`K`$ 条示范之后剩下的信息损失有显式公式；硬预算下的最优查询是按权重贪心；示范与自适应查询满足同一个线性的信息预算不等式。

**交给下一部分.** 缺失的事实必须去获取。学习器从信息里得到什么，是 §3 的问题；某个*具体*学习器需要多少信息，由 §6 的 Theorem A 回答。

### §3 · 学习后果

**问题.** 给定一个反馈来源，学生实际会学到什么？比较两个来源时，应该比较什么？

**结论.**
- 静态指标不够。两个来源可以有同样好的 teacher，给每个学生同样的策略价值，却蒸馏出不同的学生。
- 两个标签包可以有相同的边缘分布和相同的梯度均值，学习价值却不同，而且换一个任务，差的符号反过来。
- 正确的对象是 continuation response：收到消息 $`y`$、执行实际更新、再按参考配方把训练走完之后的期望最终任务回报。
- 整个训练过程的差距精确等于逐轮的单边界替换效应之和。两个消息律可以互换，当且仅当它们在每个 response 下的期望都相同。
- 当保证只在部分状态空间上成立时，离开这部分的概率只付一次，而不是每轮付一次。

**交给下一部分.** 充分性的对象是 response 类 $`\mathcal G`$。§4 度量它的复杂度，§5–§7 对真实 optimizer 计算它，§8 用有限数据检验它。

### §4 · 充分性的复杂度

**问题.** 两个反馈来源要在多少个统计量上一致，才能对学习等价？

**结论.**
- 若用一组特征的均值判断等价，精确充分当且仅当每个 response 都落在这组特征张成的空间里；所需统计量的个数是 response 类张成空间的维数。
- 近似地，最坏情形误差是 response 被这组特征一致逼近的最佳误差的两倍。不受限的学习器需要整个分布。
- 对平均独立经验块的学习器、部署至多 $`N`$ 个决策的任务，恰好需要直到 $`N`$ 阶的全部矩，而且每一阶都必要。
- 差距随*独立*块的个数衰减，速率由第一个不匹配的累积量决定，与训练轮数无关；replay 与确定性合成不会让它衰减。
- 近似复杂度可以远低于精确阶（光滑的值约为 $`\sqrt N`$），奇偶性类型的值是例外。
- 非线性更新的复合会把所需的阶相乘。

**交给下一部分.** 需要匹配什么由学习器本身决定，所以必须对实际 optimizer 建模（§5）；在近似中，系数的稳定性很关键（§8）。

### §5 · Optimizer state 与一阶表示

**问题.** 对实际使用的学习器（SGD、momentum、Adam/AdamW、裁剪、Lion、LoRA），反馈必须携带什么？

**结论.**
- 把任何带状态的一阶 optimizer 写成从 consumed gradient 到完整学习器状态增量的映射。在一阶意义下，一个反馈律对训练结果的影响，等于 continuation value 的梯度与平均完整状态增量的内积。
- 更高阶对应增量的更高阶矩。这个展开是标准的 Taylor 展开加 telescoping，与 stochastic modified equations、TracIn、metagradient 属于同一类计算，这里只当作工具。
- 各 optimizer 的不同之处在于增量：
  - cold-start Adam 第一步只看梯度符号，而它的状态保留了幅度。
  - warm Adam 的精确一阶统计量是 Adam 增量的均值。在冻结分母近似下，它化为梯度的均值与二阶矩，误差逐坐标由该坐标的 warmth 控制。
  - warmth 只在 $`s(1-\beta_2)\ll1`$ 时按 $`1/(\beta_2s)`$ 下降，之后停在 $`(1-\beta_2)/\beta_2`$，所以冻结分母误差每步有一个下限。
  - 裁剪使统计量变成裁剪后的均值。裁剪以概率 $`p_{\rm act}`$ 激活时，只有平均方向起作用，误差不超过 $`c(1-p_{\rm act})`$。

**交给下一部分.** 增量映射告诉我们反馈律必须携带*什么*。Theorem A（§6）问数据关于它必须有多少信息，Theorem B（§7）追踪一条消息如何经过 Adam 的二阶矩，§8 测量它。

### §6 · Theorem A：学习器相对的信息界

**问题.** 给定 $`K`$ 条示范和 $`n`$ 次允许的查询，*任何*蒸馏器能把 oracle 反馈的一阶效应复现到什么程度？查询应该怎么花？

**结论.**
- 令 $`c=\sum_tJ_t^*g_t`$。它把学习器的 Jacobian $`J_t`$（平均增量随未知环境参数怎样变化）与任务的 continuation 梯度 $`g_t`$ 合在一起。
- 任何蒸馏器的 minimax 误差，在 $`1/4`$ 到 $`1`$ 的常数因子内，等于 learner-relative information norm：$`c^\top h`$ 与示范、查询在方向 $`h`$ 上的 Fisher information 之比的最大值。
- 渐近最优的查询服从以 $`c`$ 为目标的 Elfving c-optimal design，并加以正则化，使每个方向最终都可辨识。
- 推论：
  - 示范看不见的方向，使任何数量的 replay 都没有价值；
  - 一条示范值多少次查询，由学习器决定；
  - 在状态覆盖模型中，查询按 water-filling 流向学习器相对权重高、teacher 覆盖低的状态；
  - SGD 按访问频率的平方给状态加权，频繁访问区间的 warm Adam 按访问频率本身加权，所以 Adam 放大学生的长尾；
  - 裁剪使学习器对梯度尺度不敏感。

**交给下一部分.** Theorem A 说明信息需要投在哪个方向。对 Adam，这个方向有一部分经过二阶矩，§7 分析它。

### §7 · Theorem B：Adam 的二阶矩通道

**问题.** 在 warm Adam 下，一条消息怎样作用于最终结果？梯度什么时候越大反而越糟？

**结论.**
- 一条消息的一阶价值精确分解为参数、动量、二阶矩三个通道，各通道的 costate 由一个显式的反向递推给出。
- 在平稳的 mean-field 情形下，对相对幅度为 $`\lambda`$ 的消息，response 有精确闭式：
  - 小增量时是凹的二次函数，相对误差有一个已证明的 $`(1-\beta_2)\max\lbrace1,\lambda^2\rbrace`$ 量级的界；最优幅度 $`\lambda^*=(1-\beta_1^{T+1})/(\rho\,\omega_T)`$，在 $`2\lambda^*`$ 附近变号；
  - $`\lambda`$ 很大时直接效应饱和，而对后续步长的压低持续累积；极限有显式表达式，极限为负时 response 有一个正的零点。
- 因此，只看参数的分数和余弦分数会在一个显式区域内把消息对排错。在 SGD 下，价值只是 $`\lambda`$ 的线性函数。

**交给下一部分.** 二阶矩通道占主导的地方，增量相对 $`v`$ 很大，一阶排序会失效；那里必须测量有限替换（§8）。

### §8 · 验证与 Theorem C

**问题.** 只有有限次测量时，怎样证明一个反馈来源优于另一个，或者从若干候选中选出好的？

**结论.**
- 测量一个特征的误差会被该特征的系数放大；残差小而系数巨大的拟合不是证明。
- 按固定 audit 单元上测得的差距做选择，需要一个显式的覆盖（transport）项。训练到达未覆盖的状态时，audit 可以读出 0，而学习差距是 1。
- 在实际状态上做配对全尾段实验、按置信下界选择，相对参考的损失不会超过 $`\delta`$，但每个候选需要很多配对副本。
- 平均增量本身很容易估计。但一阶分数需要 continuation 梯度和余项常数；只有当候选对的一阶项超过二阶余项时才能认证它。所以平均增量沿梯度方向相同的候选对，无论真实差距多大，一阶分数都认证不了。
- Theorem C 用一轮的 CRN 配对替换代替梯度：期望意义下精确，半径只含统计项与校准项，每个被审计状态需要 $`(M+1)r`$ 个 surrogate 尾段。
- 开放问题：一个远小于先验区间的可证明区间界，以及一个可检验的校准迁移条件。没有它们，严格证书是成立的，但很少能下结论。

**交给下一部分.** §9 把获取、选择与认证组装成一个方法。

### §9 · 机制与方法

**问题.** 由此推出什么方法？哪个机制值得单独检验？

**结论.**
- *Coupling width*：固定每个单元的边缘分布，只改变单元如何组成联合训练包；结果不变当且仅当包的回报在单元之间可加。一步 SGD 是严格的零假设。
- *CED（Consequence-Calibrated Experience Distillation）*：记录完整学习器状态；在编译之前按学习器相对的设计获取缺失信息；用 CRN 配对替换分数选择候选反馈律，并用全尾段 audit 校准、计入覆盖；匹配学习器的统计量，并把消息幅度控制在相对 running 二阶矩合适的范围；冻结选择器，再检验迁移。
- 每条设计规则都是某个已证明陈述的推论。

---

## 第 2 层 · 主要结果

每个主要结果用三行说明：说了什么、直觉、在链条中的作用。精确陈述见[第 3 层](#第-3-层--各节的精确陈述)。图中箭头 A → B 表示 B 的陈述或证明用到了 A。

### §2 · 可行性与信息获取：主要结果

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  T2["<b>Theorem 2.2</b><br/>regret 分解"]:::b1
  P3["<b>Prop. 2.3</b><br/>稳健歧义的对偶"]:::b1
  P4["<b>Prop. 2.4</b><br/>后处理无济于事"]:::b1
  T6["<b>Theorem 2.6</b><br/>被动不可能，<br/>主动可达"]:::b1
  P7["<b>Prop. 2.7</b><br/>编译代价"]:::b1
  L9["<b>Lemma 2.9</b><br/>示范不泄露<br/>未揭示比特"]:::b1
  T10["<b>Theorem 2.10</b><br/>少样本信息下限"]:::b1
  P11["<b>Prop. 2.11</b><br/>同一份信息预算"]:::b1
  T2 --> P3
  T2 --> P4
  T2 --> T6
  T6 --> P7
  T6 --> T10
  L9 --> T10
  T10 --> P11
  classDef b1 fill:#e8f1fb,stroke:#2f6db5,color:#111
```

**[Theorem 2.2](proofs/02_feasibility_and_acquisition_ZH.md#s2-2) · regret 分解与决策歧义**
- *说了什么.* 学生的期望 regret 精确等于三项之和：表示误差（学生类达不到参照最优）、信息误差（数据分不清可能的环境）、算法误差（学习器没有用好数据），三项都非负。信息误差为零，当且仅当对几乎每条 transcript，与它相容的所有环境有共同的最优策略。
- *直觉.* 信息误差是即使学得完美也会损失的部分，因为数据让正确的决策悬而未决。
- *作用.* 它把"信息不够"与"学得不好"分开；后面每个信息论结果都从这里出发。

**[Proposition 2.3](proofs/02_feasibility_and_acquisition_ZH.md#s2-3) · 稳健歧义的对偶**
- *说了什么.* 在一组无法区分的环境上，对混合策略最小化的最坏 regret，等于在这组环境的先验上最大化的 Bayes 信息 regret。它为零，当且仅当有一个策略在所有这些环境中都最优。
- *直觉.* 无法区分的世界迫使学习器对冲；对冲的代价是对抗最不利先验的零和博弈的值。
- *作用.* 它给信息误差一个可计算的 minimax 形式，并给出衡量其大小的证书（一个先验）。

**[Proposition 2.4](proofs/02_feasibility_and_acquisition_ZH.md#s2-4) · 后处理不能降低信息 regret**
- *说了什么.* 任何不依赖于环境的 transcript 随机变换，例如 replay、改写、转述或合成，都只会提高信息误差或使它不变；加入与环境无关的数据不改变它。
- *直觉.* 加工数据可以让已有信息更容易学到，从而降低算法误差，但造不出缺失的事实。
- *作用.* 它排除了用生成数据代替新实验的可能；§6 把它推广成学习器相对的陈述。

**[Theorem 2.6](proofs/02_feasibility_and_acquisition_ZH.md#s2-6) · 被动不可能，主动可达**
- *说了什么.* 在一个两世界族中，teacher 在两个世界里都最优，它的示范在两个世界里分布相同，但学生的最优策略相反。只看示范的学习器总会在某一个世界里至少损失 $`\Delta`$。允许 reset 查询时，$`\log(1/\delta)/\Delta^2`$ 量级的查询既必要又充分（差一个常数因子）。
- *直觉.* 完美的示范展示的是 teacher 做了什么，而不是能力不同的学生应该怎么做；学生得自己去试。
- *作用.* 它说明方法为什么需要主动获取这一步，也说明示范质量不等于信息充分。

**[Proposition 2.7](proofs/02_feasibility_and_acquisition_ZH.md#s2-7) · 编译代价**
- *说了什么.* 获取到缺失的事实之后，把它教给学生还需要 $`O(\log(1/\delta_L))`$ 个编译标签，与差距 $`\Delta`$ 无关。
- *直觉.* 弄清该做什么，与教会学生去做，是两个代价不同的问题。
- *作用.* 在方法的记账中，它把获取预算（真实查询）与编译预算（合成标签）分开。

**[Lemma 2.9](proofs/02_feasibility_and_acquisition_ZH.md#s2-9) · 示范不泄露未揭示比特**
- *说了什么.* 在隐藏比特模型中，给定观察到的一切，任何示范都没揭示的比特仍是独立的均匀硬币。
- *直觉.* 揭示机制与事实本身无关时，示范揭示了一部分事实，却不提供关于其余事实的证据。
- *作用.* 它是精确信息下限与贪心查询最优性的关键一步。

**[Theorem 2.10](proofs/02_feasibility_and_acquisition_ZH.md#s2-10) · 少样本信息下限与硬预算下的最优查询**
- *说了什么.* $`K`$ 条示范之后，信息下限为 $`\tfrac12\sum_jw_j(1-q_j)^K`$，其中 $`w_j`$ 是上下文 $`j`$ 在部署中的权重，$`q_j`$ 是一条示范揭示它的概率。硬预算为 $`n`$ 次查询时，先查权重最大的未见事实是最优的，按揭示出的值调整查询没有帮助。
- *直觉.* 每条示范以一定概率揭示一个事实；没揭示的只能去查，而且先查最值钱的缺口。
- *作用.* 它把 $`K`$ 写进理论，也是 §6 中查询分配的模型情形。

**[Proposition 2.11](proofs/02_feasibility_and_acquisition_ZH.md#s2-11) · 示范与查询共用一份信息预算**
- *说了什么.* 要以至多 $`\delta`$ 的误差区分两个世界，$`K`$ 条示范的 KL 信息加上每次查询的 KL 信息乘以期望查询次数，必须达到 $`\mathrm{kl}(1-\delta,\delta)`$；这对每个自适应的查询–停止规则都成立。
- *直觉.* 示范与查询往同一个账户里存钱：有信息的示范减少所需的查询，replay 什么也不存。
- *作用.* 它给出 $`K`$ 与查询预算之间的兑换率；§6 让这个兑换率依赖于学习器。

### §3 · 学习后果：主要结果

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  P1["<b>Prop. 3.1</b><br/>静态价值 ≠<br/>学习价值"]:::b2
  T2["<b>Theorem 3.2</b><br/>continuation response<br/>刻画"]:::b2
  P1 --> T2
  classDef b2 fill:#eaf6ec,stroke:#2e8540,color:#111
```

**[Proposition 3.1](proofs/03_learning_consequence_ZH.md#s3-1) · 静态价值不等于学习价值**
- *说了什么.* 两个数据来源可以让类中每个学生的策略值相同、teacher 的值也相同，蒸馏出的学生回报却分别是 $`\tfrac{43}{80}`$ 与 $`\tfrac{41}{80}`$。两个标签包可以边缘分布相同、平均梯度相同，学习价值却相差 $`\eta^2/8`$，换一个任务，符号反过来。
- *直觉.* 学到什么，取决于数据*经由更新*如何作用在学生身上，以及部署奖励什么；数据本身看起来有多好并不决定这一点。
- *作用.* 它排除了把 teacher 质量、策略值和平均梯度当作充分性判据，并引出 continuation response。

**[Theorem 3.2](proofs/03_learning_consequence_ZH.md#s3-2) · continuation response 刻画**
- *说了什么.* 两个训练过程之间的差，精确等于逐轮"把这一轮的参照步换成备选步"的效应之和，效应用备选过程实际访问的状态上的 continuation value 度量。两个消息分布在每个单边界替换上可互换，当且仅当它们在每个 response 上的期望相同。当保证只在部分状态空间上成立时，离开它的概率只付一次，而不是每轮付一次。
- *直觉.* response $`g(y)`$ 回答的是："如果这一轮喂给学生 $`y`$，之后按声明的方式继续训练，最终得分是多少"；它把更新规则、其余训练和部署任务都包含在内。
- *作用.* 它确定了充分性的对象，即 response 类 $`\mathcal G`$；§4 度量它的大小，§5–§7 研究它对 optimizer 的依赖，§8 研究如何用有限数据检验它。

### §4 · 充分性的复杂度：主要结果

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  T2["<b>Theorem 4.2</b><br/>充分性对偶与迁移"]:::b3
  T4["<b>Theorem 4.4</b><br/>仿射学习器：矩阶"]:::b3
  T5["<b>Theorem 4.5</b><br/>与轮数无关的界，<br/>平均化速率"]:::b3
  P6["<b>Prop. 4.6</b><br/>精确阶 ≠<br/>近似复杂度"]:::b3
  T7["<b>Theorem 4.7</b><br/>更新复合"]:::b3
  T2 --> T4
  T4 --> T5
  T2 --> P6
  T4 --> T7
  classDef b3 fill:#fdf3e1,stroke:#b8741a,color:#111
```

**[Theorem 4.2](proofs/04_complexity_of_sufficiency_ZH.md#s4-2) · 充分性对偶与迁移**
- *说了什么.* 用一组固定特征的均值来判断两个 feedback law。特征均值相等能推出 response 均值相等，当且仅当每个 response 都落在特征张成的空间里，所以所需统计量个数为 $`\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1`$。近似地，最坏的未分辨差距等于用特征一致逼近 response 的最佳误差的两倍；若 response 写成"特征的线性组合加小残差"，特征均值的误差按系数大小传递过来。不受限的 learner 需要整个分布。
- *直觉.* 特征均值相等只排除了特征所张方向上的差别；response 落在这个空间之外的部分，是接口看不见的风险。
- *作用.* 它把"需要匹配多少信息"变成可计算的量（span 维数或逼近宽度），§4、§5、§8 后面的计数都用到它。

**[Theorem 4.4](proofs/04_complexity_of_sufficiency_ZH.md#s4-4) · 仿射 learner：矩阶充分且必要**
- *说了什么.* 学生有若干二元动作 context，部署至多 $`N`$ 次决策，learner 对独立的经验块取凸平均。此时两个数据源对所有这样的任务与 learner 等价，当且仅当它们直到 $`N`$ 阶的联合矩都相等。每一阶都必要；在网格上精确个数为 $`\binom{k+N}N-1`$；在二元块上，少于 $`2^k-1`$ 个统计量可能留下大小为 1 的差距。
- *直觉.* 部署价值是学生动作概率的 $`N`$ 次多项式，平均化把它的期望变成不超过 $`N`$ 阶块矩的组合。
- *作用.* 这是 Theorem 4.2 一个可以完全算出来的实例，说明信息阶数由部署长度决定。

**[Theorem 4.5](proofs/04_rates_and_composition_ZH.md#s4-5) · 与轮数无关的界与平均化速率**
- *说了什么.* 对凸仿射 learner，学习差距不超过"系数质量 × 最大矩差"，与训练轮数无关。对 $`n`$ 个独立块取平均时，差距按 $`n^{-(r-1)}`$ 衰减，$`r`$ 是第一个不匹配的 cumulant 的阶数。重放同一个块，或用它的确定性改写来训练，差距完全不衰减。
- *直觉.* 只有独立的证据才能缩小差距；重复和确定性合成增加的是暴露量，不是信息。
- *作用.* 它把独立块数与 epoch 数分开。§2 与 §9 的方法都用这条计数规则。

**[Proposition 4.6](proofs/04_rates_and_composition_ZH.md#s4-6) · 精确阶不等于近似复杂度**
- *说了什么.* 对价值 $`p^N`$，匹配到大约 $`\sqrt{N\log(1/\varepsilon)}`$ 阶的矩，差距就小于 $`\varepsilon`$，尽管精确相等需要 $`N`$ 阶。对 parity，任何次数低于 $`N`$ 的多项式的逼近误差都不小于 $`1/2`$。
- *直觉.* 光滑的 response 有很好的低阶近似；组合型的 response 没有。
- *作用.* 它说明真正起作用的复杂度是逼近宽度而不是次数，并引出 §8 回答的系数稳定性问题。

**[Theorem 4.7](proofs/04_rates_and_composition_ZH.md#s4-7) · 更新复合放大信息需求**
- *说了什么.* 复合的多项式更新会把 response 关于早期块的次数，乘上所有后续更新的次数，且这个界可以达到。阈值 learner 只用一次决策就破坏了一阶矩充分性。对光滑 SGD 的一步，两个 feedback law 之差等于梯度均值项加梯度二阶矩项再加三阶余项；逐步误差只有在稳定性条件下才线性累加。
- *直觉.* 同样的数据喂给不同的 learner，需要的信息量不同；需要匹配什么，是"learner 与任务复合"的性质。
- *作用.* 它说明必须对实际的 optimizer 建模，这正是 §5 的内容。

### §5 · Optimizer state 与一阶表示：主要结果

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart TB
  P3["<b>Prop. 5.3</b><br/>cold-start Adam"]:::b4
  P7["<b>Prop. 5.7</b><br/>cold 步分开<br/>均值相同的反馈"]:::b4
  P1["<b>Prop. 5.1</b><br/>一阶表示（工具）"]:::b4
  T10["<b>Theorem 5.10</b><br/>各学习器的<br/>一阶统计量"]:::b4
  P9["<b>Prop. 5.9</b><br/>部分激活的裁剪"]:::b4
  P5["<b>Prop. 5.5</b><br/>Adam 精确统计量，<br/>冻结分母误差"]:::b4
  P6["<b>Prop. 5.6</b><br/>warm Adam<br/>一阶仿射"]:::b4
  P8["<b>Prop. 5.8</b><br/>warmth 的演化"]:::b4
  P3 --> P7
  P1 --> T10
  P3 --> T10
  P9 --> T10
  P5 --> T10
  P5 --> P6
  P6 --> P8
  classDef b4 fill:#f6e9f4,stroke:#9c3d8f,color:#111
```

**[Proposition 5.1](proofs/05_optimizer_state_ZH.md#s5-1) · optimizer 变换后的展开**
- *说了什么.* 任何带状态的一阶 optimizer（包括裁剪）都可以写成一个映射：把 consumed gradient 映成完整状态（参数、momentum、二阶矩、计数器）的增量。两个 feedback law 造成的训练结果之差，在一阶意义下等于各轮 continuation value 梯度与平均状态增量之差的内积之和，余项由增量的二阶矩控制。在 continuation value 梯度方向上匹配平均增量，一阶充分；在一阶项超过余项处也必要；统计量的个数是线性化 response 类的维数。
- *直觉.* 反馈只通过它造成的状态变化影响学生。一阶意义下只有这个变化的均值起作用，而且只在 continuation value 关心的方向上起作用。
- *作用.* 一个工具，与 stochastic modified equations、TracIn、metagradient 是同类计算（Remark 5.2）。后面所有"学习器要求反馈携带什么"的结论，以及 Theorem A–C，都从它算出。

**[Proposition 5.3](proofs/05_optimizer_state_ZH.md#s5-3) · Cold-start Adam：符号商与 optimizer 记忆**
- *说了什么.* 从零 optimizer state 出发，Adam 的第一步只依赖 $`g_i/(\lvert g_i\rvert+\epsilon)`$。$`\epsilon=0`$ 时必须匹配联合符号分布（只匹配符号边缘分布不够）；$`\epsilon>0`$ 时必须匹配整个梯度分布。momentum 保留了第一个梯度的幅度，所以符号相同的两段历史，第二步可以方向相反。
- *直觉.* Adam 的第一步几乎就是 sign descent：幅度从步长里被除掉了，却保留在 optimizer state 中。
- *作用.* 说明矩充分性不能搬到 Adam 上，要匹配什么取决于 optimizer state。

**[Proposition 5.5](proofs/05_warm_adam_ZH.md#s5-5) · Adam 的精确一阶统计量与冻结分母误差**
- *说了什么.* 对 Adam，一阶统计量是 Adam 增量的精确均值，它不由梯度的均值与二阶矩决定。把分母冻结在保留值上，一步 Adam 就变成带 preconditioner 的 momentum SGD，每个坐标的误差至多 $`\eta u_i\min\lbrace\kappa_i/2,1\rbrace`$，$`\kappa_i`$ 是 warmth。
- *直觉.* Adam 除以一个 running RMS，而新梯度本身会扰动这个 RMS。扰动小（warmth 小）时，这一步对梯度近似线性。
- *作用.* 把"warm Adam 只需要 $`(\mathbb E G,\mathbb E G^{\odot2})`$"里藏着的两个近似分开：一阶展开与冻结分母。

**[Proposition 5.6](proofs/05_warm_adam_ZH.md#s5-6) · warm Adam 关于 consumed gradient 一阶仿射**
- *说了什么.* warm 状态下，单边界 response 等于 consumed gradient 的一个仿射函数加显式残差：冻结分母项、二阶矩敏感度项与曲率项。平均 consumed gradient 相同的分布，response 之差至多为残差的期望。
- *直觉.* 二阶矩积累起来以后，一步 Adam 就像一步带 preconditioner 的 momentum SGD。
- *作用.* 说明从训练的哪个阶段起，仿射与一阶理论适用于 Adam，并给出可计算的误差。

**[Proposition 5.7](proofs/05_warm_adam_ZH.md#s5-7) · cold 步在一阶就分开均值相同的反馈**
- *说了什么.* cold 状态下，两个均值相同的梯度分布——甚至低于任意固定阶的矩全都相同——给出的 response 在学习率的一阶上就不同。
- *直觉.* cold 步看的是相对于零的符号，而符号概率不是低阶矩的函数。
- *作用.* 证明 cold 窗口是真正的一阶效应，不是高阶修正。

**[Proposition 5.8](proofs/05_warm_adam_ZH.md#s5-8) · warmth 的演化**
- *说了什么.* 梯度平稳时，吸收 $`s`$ 次计算后的 mean-field warmth 恰为 $`(1-\beta_2)/(\beta_2(1-\beta_2^s))`$。只有在 $`s(1-\beta_2)\ll1`$ 时它才按 $`1/(\beta_2s)`$ 下降，之后停在 $`(1-\beta_2)/\beta_2`$。沿整个训练累积，由 warmth 控制的误差至多 $`c\log(1+R/W)+c(1-\beta_2)R`$；bias correction 不改变 warmth；学习率为零的 warm-up 推进二阶矩的方式与真实步完全相同。
- *直觉.* $`\beta_2`$ 固定就意味着平均窗口固定，所以每个新梯度都会按固定的相对幅度扰动分母。
- *作用.* 确定 warm 描述在训练中准确到什么程度，并说明 $`\beta_2`$ 固定时它永远不会变成精确的。

**[Proposition 5.9](proofs/05_learner_statistics_ZH.md#s5-9) · 部分激活的梯度裁剪**
- *说了什么.* 裁剪后梯度的均值可以精确拆成未裁剪部分与裁剪方向部分；用 $`c`$ 乘平均方向代替它，误差至多 $`c(1-p_{\rm act})`$，$`p_{\rm act}`$ 是裁剪激活的概率。
- *直觉.* 只有低于阈值的梯度保留幅度，其余只贡献方向。
- *作用.* 对任意激活率给出精确的裁剪统计量，并量化"只有方向起作用"这一简化。

**[Theorem 5.10](proofs/05_learner_statistics_ZH.md#s5-10) · 各学习器的一阶统计量**
- *说了什么.* Proposition 5.1 的统计量在常见学习器下的具体形式：SGD 与 momentum 是平均梯度；warm Adam 是 Adam 增量的精确均值，只在冻结分母近似下等于 $`(\mathbb EG,\mathbb EG^{\odot2})`$，二阶矩经由 $`v`$ 进入；cold Adam 的一步在参数上是符号分布；裁剪是裁剪后均值；Lion 是翻转概率；$`B=0`$ 的 LoRA 是 rank-$`r`$ sketch，第一步 $`A`$ 不动。
- *直觉.* 平均梯度相同的两个反馈来源，在 cold Adam、裁剪，或经由二阶矩通道的 warm Adam 下，仍然可以在一阶上不同。
- *作用.* 告诉方法在每种训练配方下要匹配、要测量什么，并为 Theorem A 的 learner Jacobian 和 Theorem B 的状态通道提供输入。

### §6 · Theorem A：学习器相对的信息界：主要结果

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  T3["<b>Theorem 6.3</b><br/>学习器相对的下界"]:::b4
  L4["<b>Lemma 6.4</b><br/>设计的 minimax 等式"]:::b4
  T5["<b>Theorem 6.5</b><br/>c-optimal design<br/>可达"]:::b4
  C7["<b>Cor. 6.7</b><br/>混叠；replay<br/>价值为零"]:::b4
  C8["<b>Cor. 6.8</b><br/>示范与查询的<br/>兑换率"]:::b4
  P10["<b>Prop. 6.10</b><br/>water-filling 分配"]:::b4
  P11["<b>Prop. 6.11</b><br/>学习器权重：<br/>SGD 与 warm Adam"]:::b4
  C12["<b>Cor. 6.12</b><br/>学习器分离"]:::b4
  T3 --> T5
  L4 --> T5
  T3 --> C7
  T3 --> C8
  T5 --> P10
  P10 --> P11
  P11 --> C12
  C7 --> C12
  classDef b4 fill:#f6e9f4,stroke:#9c3d8f,color:#111
```

**[Theorem 6.3](proofs/06_learner_relative_information_ZH.md#s6-3) · 学习器相对的下界**
- *说了什么.* 任何蒸馏器（包括 replay、改写、合成）在声明任务上复现 oracle 反馈的一阶效应时，误差都不低于 learner-relative information norm $`\mathfrak n_{K,n}`$ 的四分之一。这个范数度量的是：$`K`$ 条示范与 $`n`$ 次查询，在学习器与任务真正在意的那个方向 $`c`$ 上，揭示了多少关于环境的信息。
- *直觉.* 数据分不开、而这个学习器会区别对待的两个环境，必然让蒸馏器在其中一个上出错；要紧的是学习器的敏感方向 $`c`$，而不是环境之间差得有多大。
- *作用.* 它是 Theorem A 的下半部分，也是"先获取、再编译"的定量核心；混叠与兑换率两个推论都由它得到。

**[Lemma 6.4](proofs/06_learner_relative_information_ZH.md#s6-4) · 设计的 minimax 等式**
- *说了什么.* 在所有查询 design 中，估计学习器相对泛函所能达到的最小方差，恰好等于信息范数的平方，并且最优 design 存在。它就是以"学习器 Jacobian 作用在 continuation value 梯度上"为目标向量的 c-optimal（Elfving）design。
- *直觉.* 查询最好花在对学习器唯一用到的那个方向最有信息量的地方；这是一个经典的设计问题，只是目标由学习器决定。
- *作用.* 它把下界变成一条获取规则，并让上界可以计算。

**[Theorem 6.5](proofs/06_learner_relative_information_ZH.md#s6-5) · 可达性**
- *说了什么.* 用（略加正则化的）c-optimal 查询 design 估计环境，再在估计值处发出 oracle 律，误差可以达到 $`\mathfrak n_{K,n}(1+o(1))`$。与下界合起来，最优一阶误差被夹在 $`\mathfrak n_{K,n}/4`$ 与 $`\mathfrak n_{K,n}`$ 之间。
- *直觉.* 信息一旦按正确的方向收集到，plug-in 蒸馏器最多只损失一个常数因子。
- *作用.* 它是 Theorem A 的上半部分：信息范数就是衡量"一批经验对给定学习器是否充分"的正确尺度。

**[Corollary 6.7](proofs/06_learner_relative_information_ZH.md#s6-7) · 混叠与 replay 的零价值**
- *说了什么.* 若两个环境给出相同的示范分布，而学习器相对泛函在两者之间不同，那么对每个 $`K`$，任何蒸馏器的误差都至少是差值的一半；replay 与改写都帮不上忙。
- *直觉.* 同一份证据复制再多次，也分不开这份证据本来就分不开的两个世界。
- *作用.* 它是被动不可能性的学习器相对形式：示范够不够，取决于学习器。

**[Corollary 6.8](proofs/06_learner_relative_information_ZH.md#s6-8) · 示范与查询的兑换率**
- *说了什么.* 示范与查询逐方向地进入同一个线性信息预算；在起约束作用的方向上，一条示范值确定数量的查询，而这个数量取决于学习器。
- *直觉.* 示范与查询都是证据，它们的相对价值在于各自在这个学习器关心的方向上揭示了多少。
- *作用.* 它给出"多收集示范"与"花查询"之间的取舍。

**[Proposition 6.10](proofs/06_learner_relative_information_ZH.md#s6-10) · 显式信息界与 water-filling**
- *说了什么.* 当示范覆盖 teacher 的状态、训练发生在学生的状态上时，误差由 $`\sum_sW_s/N_s`$ 决定，其中 $`W_s`$ 是状态 $`s`$ 的学习器相对权重，$`N_s`$ 是它的样本数。最优查询分配是 water-filling，$`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$；权重为正而完全未被覆盖的状态留下不可约的误差。
- *直觉.* 查询应当流向学习器敏感、而 teacher 没去过的状态。
- *作用.* 它让 Theorem A 在 agent 蒸馏中可以直接使用，并给方法的获取步骤提供分配规则。

**[Proposition 6.11](proofs/06_learner_relative_information_ZH.md#s6-11) · SGD 与 warm Adam 的学习器权重**
- *说了什么.* SGD 下一个状态的权重随访问频率的平方增长；warm Adam 在频繁访问区间只随访问频率线性增长，并带有一个二阶矩项；对极少访问的状态，Adam 又回到平方标度，但只看反馈的符号。
- *直觉.* Adam 按 running second moment 归一化，少见状态的步长和常见状态一样大，所以这些状态上的反馈比在 SGD 下更重要。
- *作用.* 它说明 optimizer 改变了信息该在哪里收集，并给出 §7 还会用到的常数 $`\kappa_{\rm iso}`$。

**[Corollary 6.12](proofs/06_learner_relative_information_ZH.md#s6-12) · 学习器之间的分离**
- *说了什么.* 揭示了反馈方向、却不揭示其尺度的示范，对始终裁剪的学习器充分，对 SGD 则对每个 $`K`$ 都不充分。定量地看，一条很少被覆盖的长尾状态，让 warm Adam 付出的信息代价远高于 SGD。
- *直觉.* 学习器忽略的东西不必告诉它；学习器放大的东西必须告诉得很准。
- *作用.* 它把"同一批示范可以对一个学习器充分、对另一个不充分"变成了形式化的结论。

### §7 · Theorem B：Adam 的二阶矩通道：主要结果

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  T1["<b>Theorem 7.1</b><br/>状态通道分解"]:::b4
  C2["<b>Cor. 7.2</b><br/>平稳情形的<br/>幅度定律"]:::b4
  P3["<b>Prop. 7.3</b><br/>两类 proxy<br/>不一致"]:::b4
  T1 --> C2
  C2 --> P3
  classDef b4 fill:#f6e9f4,stroke:#9c3d8f,color:#111
```

**[Theorem 7.1](proofs/07_second_moment_channel_ZH.md#s7-1) · 状态通道分解**
- *说了什么.* 对 Adam/AdamW（含 bias correction、decoupled weight decay、$`\epsilon`$ 与全局范数裁剪），continuation value 关于 $`(\theta,m,v)`$ 的梯度由一个显式的反向 costate 递推给出。由此，一条消息的一阶价值精确分解为参数通道、动量通道和二阶矩通道三项，最后一项只通过 $`g\odot g`$ 依赖于消息。
- *直觉.* 一条消息不只改变当前这一步，还改变 running second moment，进而改变此后每一步的步长。二阶矩通道就是这种改变带来的代价或收益。
- *作用.* 让 [Proposition 5.1](proofs/05_optimizer_state_ZH.md#s5-1) 中 optimizer state 那一部分在 Adam 下可以算出来。幅度定律（Corollary 7.2）和只看参数的影响分数为何失效（Proposition 7.3），都建立在它上面。

**[Corollary 7.2](proofs/07_second_moment_channel_ZH.md#s7-2) · 平稳情形的幅度定律**
- *说了什么.* 在 warm 的平稳状态上，对幅度为 $`\lambda`$（以 running RMS 为单位）的同向消息，mean-field response 是一个对任意 $`\lambda`$ 都成立的显式有限和。
  - 增量小时，它是凹二次函数，最优幅度为 $`\lambda^\star=(1-\beta_1^{T+1})/(\rho\,\omega_T)`$，在 $`2\lambda^\star`$ 附近变号，并有显式误差界。
  - 幅度很大时，直接效应饱和。大消息最终劣于不发消息，当且仅当信噪比 $`\rho`$ 超过 $`\kappa_{\rm iso}(T)/\Lambda(T)`$。
- *直觉.* 大梯度会把 $`v`$ 抬高，从而把此后许多步的步长都压小；超过一定幅度，压小步长的损失就超过大步带来的收益。SGD 没有这种效应，越大越好。
- *作用.* 第一个定量说明 Adam 下消息价值关于幅度非单调的结果。它要求方法控制消息幅度相对于 running second moment 的大小，并预言了线性化分数会在哪里误导。

**[Proposition 7.3](proofs/07_second_moment_channel_ZH.md#s7-3) · 两类 proxy 的不一致性**
- *说了什么.* 不看部署任务的统计量，不可能对所有任务都把反馈律排对。在 Corollary 7.2 的区间里，只看参数的一阶分数会把两条同向消息排反，例如用 Adam 步长的 TracIn 式分数和 fixed-state 的 Adam data Shapley。余弦分数（LESS 式）则根本分不开这两条消息。
- *直觉.* 只看参数的分数看到的是更大的当前步，漏掉了经由 $`v`$ 造成的损失；余弦只看方向，而两条消息方向相同。
- *作用.* 用证明而不只是经验观察说明：两类常用的影响分数在 Adam 下都不是一致的排序器。这是方法按实测后果做选择的理由之一。

### §8 · 验证与 Theorem C：主要结果

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  T1["<b>Theorem 8.1</b><br/>稳定性–近似权衡"]:::b5
  T2["<b>Theorem 8.2</b><br/>带 transport 的<br/>冻结选择"]:::b5
  T3["<b>Theorem 8.3</b><br/>实际状态上的<br/>自适应 audit"]:::b5
  P4["<b>Prop. 8.4</b><br/>测量平均增量"]:::b5
  T8["<b>Theorem 8.8</b><br/>CRN 配对替换证书"]:::b5
  C9["<b>Cor. 8.9</b><br/>一阶版本及其上限"]:::b5
  P10["<b>Prop. 8.10</b><br/>代价与 CRN"]:::b5
  T1 --> T2
  T2 --> T3
  P4 --> C9
  T8 --> C9
  T8 --> P10
  T3 --> T8
  classDef b5 fill:#e9f5f6,stroke:#1f7f8c,color:#111
```

**[Theorem 8.1](proofs/08_verification_ZH.md#s8-1) · 稳定性–近似的权衡**
- *说了什么.* 每个特征均值只知道到某个容差时，无法排除的最大 response 差距，恰好等于两件事的最优权衡：特征对 response 近似得多好，以及这个近似的系数有多大（每个系数按其特征的容差加权）。
- *直觉.* 测不准一个特征的代价，就是它在 response 模型里的系数。残差小但系数巨大的拟合什么也证明不了。
- *作用.* 它决定 audit 应该测哪些特征、哪些 response 模型可以充当证书；Theorem 8.2 的冻结 audit 与方法中的拟合规则都用到它。

**[Theorem 8.2](proofs/08_verification_ZH.md#s8-2) · 带 transport 的冻结有限接口选择**
- *说了什么.* 事先固定 audit cell 与特征，测量每个候选相对参照的特征差距，选差距最小的。被选候选的学习差距，不超过测得的差距、统计误差、训练实际访问状态上的显式 transport 项，再加上离开认证状态集的概率。transport 项去不掉。
- *直觉.* audit 只看得见它覆盖的状态。训练一旦走到别处，audit 可以报告零差距，而学习差距却是最大的。
- *作用.* 它给出任何有限、冻结的 audit 的正确形式，并指出它的盲区；Theorem 8.3 与 8.8 在实际状态上做 audit，避开了这个盲区。

**[Theorem 8.3](proofs/08_verification_ZH.md#s8-3) · 实际状态上的自适应 audit；保真与改进**
- *说了什么.* 在训练实际到达的状态上，对每个候选与参照做配对的完整尾段实验，按置信下界选择。这样得到的训练流程相对参照的损失不超过置信水平，并保留被认证的改进。所需的配对副本数是显式的，而且很大。
- *直觉.* 直接测"替换这一轮有没有帮助"，不外推到没见过的状态；代价是大量完整的训练尾段。
- *作用.* 它是可靠但昂贵的参照 audit，用来校准更便宜的测量，例如 Theorem 8.8 的 surrogate。

**[Proposition 8.4](proofs/08_certificate_ZH.md#s8-4) · 在实际状态上测量平均增量**
- *说了什么.* 一个 feedback law 的 optimizer 平均增量只需梯度计算就能估计，在 Hilbert 范数下误差按样本量的平方根倒数下降。配上 continuation 的梯度范数与曲率，就能界定一轮的效应，沿抽样的训练运行还能界定整次训练的差距。
- *直觉.* 一阶证书中数据那一侧很便宜；continuation 那一侧（梯度与曲率）这些样本根本不测。
- *作用.* 它是一阶 audit 中便宜的那一半，也说明了一阶 audit 为什么仍需要校准常数。

**[Theorem 8.8](proofs/08_certificate_ZH.md#s8-8) · CRN 配对替换证书（Theorem C）**
- *说了什么.* 在被 audit 的状态上，让每个候选各走一轮，然后用同一组随机数为每个候选跑同一个 surrogate 尾段。配对分数差的均值无偏地估计这次替换的有限效应；empirical Bernstein 半径加上 surrogate 校准项，就把它变成对所有候选对同时成立的证书。
- *直觉.* 不先线性化再去界线性化的误差，而是直接测有限效应；共用随机数抵消了两条独立尾段会额外带来的噪声。
- *作用.* 它用测量代替了 oracle continuation 梯度，是方法中的选择步骤。它的严格性依赖于已知的分数范围与校准过的 surrogate，这正是 Remark 8.12 中的未解决问题。

**[Corollary 8.9](proofs/08_certificate_ZH.md#s8-9) · 一阶版本、它的 margin 判据与上限**
- *说了什么.* 只要两个候选的一阶差超过二阶余项，一阶分数就能把它们排对。用有限差分测量时，一阶证书在统计半径之外还要付有限差分项和余项；一阶项低于余项的候选对，无论真实差距多大，它都无法认证。
- *直觉.* 主要靠增量的二阶矩来区分的候选（Adam 下正是如此），任何一阶证书都看不见。
- *作用.* 它划出基于梯度的分数可信的范围，说明 Theorem 8.8 为什么改测有限替换；也把 oracle 诊断与可测量的证书区分开。

**[Proposition 8.10](proofs/08_certificate_ZH.md#s8-10) · 代价，以及 common random numbers 带来的好处**
- *说了什么.* 证书对每个候选、每个副本需要一条 surrogate 尾段。要让范围项变小，大约需要一百个副本；common random numbers 把每个配对分数的方差压到两个候选增量之差的量级。
- *直觉.* 共用随机数的配对尾段之所以不同，只是因为候选不同；独立的尾段还会因为训练本身的噪声而不同。
- *作用.* 它确定方法的副本预算，并指出 pathwise Lipschitz 界是得到可证明范围的途径。

### §9 · 机制与方法：主要结果

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  P1["<b>Prop. 9.1</b><br/>coupling width"]:::b6
  A2["<b>Algorithm 9.2</b><br/>CED"]:::b6
  P1 --> A2
  classDef b6 fill:#fbeaea,stroke:#b53b3b,color:#111
```

**[Proposition 9.1](proofs/09_mechanism_and_method_ZH.md#s9-1) · 耦合宽度**
- *说了什么.* 固定每个反馈单元的边缘分布，只改变单元怎样组合成联合训练 packet。所有这样的重组都不改变学习结果，当且仅当联合 packet 的 response 是各单元贡献之和；否则变化不超过 response 到可加表距离的两倍。同一参数处的一步 SGD 对重组严格不变，连续多步则不然。
- *直觉.* 如果学习器只是逐个"看"单元，分组就无关紧要。分组只通过单元之间的交互起作用，而交互在一次更新影响下一次更新时才出现。
- *作用.* 一个带严格零假设的机制检验；也说明了为什么候选 feedback law 要由它发出的 packet 来确定，而不是一袋单元。

**[Algorithm 9.2](proofs/09_mechanism_and_method_ZH.md#s9-2) · Consequence-Calibrated Experience Distillation（CED）**
- *说了什么.* 记录完整的 optimizer state；先获取缺失的信息，把查询花在学习器相对权重高、teacher 覆盖低的状态上；编译候选 feedback law；记录每个候选的学习器统计量，标记相对 running 二阶矩过大的消息；在实际状态上用 CRN 配对替换测量每个候选的后果，并用少量全尾段 audit 校准；按下置信界选择，重新 audit，冻结，再检验迁移。每项输出都标明它带有的保证，或标为经验结论。
- *直觉.* 按"这个学习器在当前状态下吃下这条反馈之后会发生什么"来选反馈，而不是按反馈本身看起来好不好。
- *作用.* 框架的落脚点：每一步都是 §2–§8 某个结论的推论，计划中的真实模型检验 R0–R3 检验的就是这些步骤。

**[Remark 9.3](proofs/09_mechanism_and_method_ZH.md#s9-3) · 六条设计规则**
- *说了什么.* 先获取，再编译；按实测后果选择；匹配学习器的统计量，而不是原始梯度；拟合系数稳定的 response 模型；计入覆盖，用配对替换做认证，一阶 audit 只用在 margin 占优处；先冻结，再检验迁移。每条规则都是相应保证的必要条件，并由所引结论推出。
- *直觉.* 每条规则堵住方法"声称多于所知"的一种途径：凭空造信息、相信 proxy、忽略 optimizer、过拟合 response 模型、外推 audit、在评测数据上调参。
- *作用.* 它解释了 Algorithm 9.2 为什么有这些步骤；方法的消融实验也应当一次去掉一条规则。

---

## 1. 设定与记号

下面的对象在每一节都会用到；各节自己的定义在该节内编号。

<a id="s1-1"></a>

**Definition 1.1（环境、学生策略类、参考策略类）.**

- 环境 $`E`$ 包括任务分布、状态转移、反馈语义、评分方式和固定的执行 harness。所有回报都在 $`[0,1]`$ 内，$`J_E(\pi)`$ 是策略 $`\pi`$ 的期望回报。
- 学生可部署的策略构成类 $`\Pi\subseteq\Pi_{\rm ref}`$，其中 $`\Pi_{\rm ref}`$ 是参考类（例如 teacher 能执行的策略）。记

```math
J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi),\qquad J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi).
```

<a id="s1-2"></a>

**Definition 1.2（信息与蒸馏器）.**

- $`I_0`$ 是公共初始信息：预训练学生、已知规则，以及每个任务族 $`K`$ 条可信示范。teacher 模型的预训练知识属于 $`I_0`$，在同一族的每个环境中都相同。
- $`D`$ 是从允许的实验中得到的全部后续信息的 transcript：在新状态上调用 teacher、环境分支、reset。观察新状态的 teacher 调用算作一次实验，即属于 $`D`$。
- *蒸馏器*是从 $`(I_0,D)`$ 与私有随机性（与环境无关）到反馈协议（Definition 1.3）的任意可测映射。replay、改写、合成都是蒸馏器；除 $`(I_0,D)`$ 外，它们不获取任何依赖于环境的信息。

<a id="s1-3"></a>

**Definition 1.3（完整学习器状态、反馈律、更新）.**

- *完整学习器状态*（complete learner state）为 $`z=(\theta,o)\in\mathcal Z`$。其中 $`\theta`$ 是参数，$`o`$ 是 optimizer state（对 Adam 是矩 $`m,v`$ 与步数），连同更新所用的缓冲区、采样器状态和历史。$`\mathcal Z`$ 是实 Hilbert 空间，内积为 $`\langle\cdot,\cdot\rangle`$，有限维时即 Euclid 空间。
- 第 $`t`$ 轮，*反馈律*（feedback law）$`P_t(\cdot\mid z)`$ 发出消息 $`Y_t`$。消息给出 *consumed gradient* $`G_t(z,Y_t)`$，它在梯度累积、归一化以及所述 optimizer 包含的裁剪之后定义。实际更新为 $`Z_{t+1}=U_t(z,Y_t,\xi_t)`$，$`\xi_t`$ 是更新的随机性。
- 对一阶 optimizer，$`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$。*增量映射* $`\Delta_t`$ 包含学习率调度、动量、二阶矩、weight decay 与裁剪。反馈律 $`P`$ 的*平均增量*为

```math
\mu^P_t(z)=\mathbb E_{Y\sim P_t(\cdot\mid z)}\,\Delta_t\big(z,G_t(z,Y)\big).
```

- *协议* $`S`$ 是反馈律的序列 $`(P^S_t)_{t<R}`$。对 $`\xi_t`$ 积分得到它的 learning kernel $`K^S_t`$；参考协议记为 $`K^0_t`$。所有协议从同一个初始状态分布出发。
- 最终学生的得分为 $`F_E(z)=J_E(\pi_z)\in[0,1]`$，$`L_E(S)=\mathbb E\,F_E(Z^S_R)`$。每一轮都用同一个反馈律 $`P`$ 的协议记作 $`L(P)`$。

<a id="s1-4"></a>

**Definition 1.4（continuation value、response、声明类）.**

- 参考尾段下的 *continuation value* 为

```math
V^0_R=F_E,\qquad V^0_t=K^0_tV^0_{t+1}\quad(t<R).
```

- $`V^0_{t+1}(z)`$ 是在第 $`t`$ 轮之后从状态 $`z`$ 出发、按参考协议继续训练所得的期望最终得分。尾段从另一个协议 $`P`$ 开始时记作 $`V^P_{t+1}`$。
- 消息 $`y`$ 在第 $`t`$ 轮、状态 $`z`$ 上的*单边界 response* 为

```math
g_{t,z}(y)=\mathbb E_{\xi_t}\,V^0_{t+1}\big(U_t(z,y,\xi_t)\big).
```

- 它是收到 $`y`$、执行实际更新、再走完声明的参考尾段之后，最终真实任务回报的期望。
- *声明类*（declared class）$`\mathcal V`$ 收集所声明的部署任务与参考尾段对应的 continuation value；*response 类* $`\mathcal G`$ 收集相应的 response。两者都在选择任何反馈之前固定。

<a id="s1-5"></a>

**Definition 1.5（计数）.** 下列计数始终分开，不混用。

| 记号 | 含义 |
|---|---|
| $`K`$ | 每个任务族的可信示范条数 |
| $`n_{\rm query}`$，§6 中记为 $`n`$ | 额外允许的实验（查询）次数 |
| §4 中的 $`n`$ | 独立经验块的个数 |
| $`R`$ | 训练更新次数 |
| $`N`$ | 一次部署中的决策个数 |
| $`d`$ | 线性统计量（特征）的个数 |
| §8 中的 $`r`$、$`M`$ | 每个被审计状态的 CRN 副本数、候选反馈律的个数 |

把一个块 replay $`n`$ 次，不等于 $`n`$ 个独立块。

**约定.**

- 陈述按节编号，每节共用一个计数器（Definition 5.4 之后是 Proposition 5.5）。
- *Theorem A* 指 Theorem 6.3 与 6.5，*Theorem B* 指 Theorem 7.1 与 Corollary 7.2，*Theorem C* 指 Theorem 8.8。
- 某个字母在局部被重用时，所在的节会在第一次使用时说明；例如 $`s`$ 在 §5 中是已吸收的梯度评估次数，在 §6 中是 agent 的状态。

---

## 第 3 层 · 各节的精确陈述

每个陈述带有证明页上的编号、精确形式、推导要点、依赖关系，以及指向证明和检验它的实验的链接。

**§2 可行性与信息获取** — [精确陈述](statements/02_feasibility_and_acquisition_ZH.md) · [证明](proofs/02_feasibility_and_acquisition_ZH.md) · [实验](experiments/02_feasibility_and_acquisition_ZH.md)

**本节做什么.**

- 它问的是：现有数据能否确定学生该怎么做，并把这个问题与学生学得好不好分开（[Theorem 2.2](proofs/02_feasibility_and_acquisition_ZH.md#s2-2)）。
- 它说明任何对现有数据的加工都补不上缺失的事实（[Proposition 2.4](proofs/02_feasibility_and_acquisition_ZH.md#s2-4)），而即使是最优 teacher 的示范也可能留下这样的缺口（[Theorem 2.6](proofs/02_feasibility_and_acquisition_ZH.md#s2-6)）。
- 它给缺失信息定价：主动查询需要 $`\Delta^{-2}\log(1/\delta)`$ 量级，编译另算（[Proposition 2.7](proofs/02_feasibility_and_acquisition_ZH.md#s2-7)）。
- 它把示范数 $`K`$ 显式写进结论，并说明示范与查询共用一份信息预算（[Theorem 2.10](proofs/02_feasibility_and_acquisition_ZH.md#s2-10)、[Proposition 2.11](proofs/02_feasibility_and_acquisition_ZH.md#s2-11)）。

**§3 学习后果** — [精确陈述](statements/03_learning_consequence_ZH.md) · [证明](proofs/03_learning_consequence_ZH.md) · [实验](experiments/03_learning_consequence_ZH.md)

**本节做什么.**

- 它说明反馈的静态度量（teacher 质量、策略值、标签边缘分布、平均梯度）决定不了学生学到什么（[Proposition 3.1](proofs/03_learning_consequence_ZH.md#s3-1)）。
- 它找出正确的对象：continuation response，即一条消息、实际更新、再加上声明的其余训练之后，最终任务回报的期望（[Theorem 3.2](proofs/03_learning_consequence_ZH.md#s3-2)）。
- 它说明两个消息分布可互换，恰好当它们在每个 response 上一致；coverage 的失败只付一次代价（[Theorem 3.2(b),(c)](proofs/03_learning_consequence_ZH.md#s3-2)）。
- 它把 response 类 $`\mathcal G`$ 确定为 §4–§8 要度量的对象（[Remark 3.3](proofs/03_learning_consequence_ZH.md#s3-3)）。

**§4 充分性的复杂度** — [精确陈述](statements/04_complexity_of_sufficiency_ZH.md) · [证明 4.1–4.4](proofs/04_complexity_of_sufficiency_ZH.md)、[证明 4.5–4.7](proofs/04_rates_and_composition_ZH.md) · [实验](experiments/04_complexity_of_sufficiency_ZH.md)

**本节做什么.**

- 两个 feedback law 可以在学习上互换，意思是它们在 response 类 $`\mathcal G`$ 上一致（[Theorem 3.2](proofs/03_learning_consequence_ZH.md#s3-2)）。本节度量这种一致需要它们共享多少信息。
- 一般的回答是 $`\mathcal G`$ 的 span 维数或逼近宽度（[Theorem 4.2](proofs/04_complexity_of_sufficiency_ZH.md#s4-2)）。
- 对凸仿射 learner 可以完全算出来：直到部署长度的联合矩，以及随独立块数变化的速率（[Theorem 4.4](proofs/04_complexity_of_sufficiency_ZH.md#s4-4) 与 [4.5](proofs/04_rates_and_composition_ZH.md#s4-5)）；近似复杂度可以远低于精确阶数（[Proposition 4.6](proofs/04_rates_and_composition_ZH.md#s4-6)）。
- 非线性或复合的更新会改变所需的阶数，所以必须对 learner 本身建模（[Theorem 4.7](proofs/04_rates_and_composition_ZH.md#s4-7)）。

**§5 Optimizer state 与一阶表示** — [精确陈述](statements/05_optimizer_state_ZH.md) · [证明 5.1–5.3](proofs/05_optimizer_state_ZH.md)、[5.4–5.8](proofs/05_warm_adam_ZH.md)、[5.9–5.11](proofs/05_learner_statistics_ZH.md) · [实验 5.1–5.7](experiments/05_optimizer_state_ZH.md)、[5.8–5.18](experiments/05_warm_adam_and_learners_ZH.md)

**本节做什么.**

- 把每个带状态的一阶 optimizer 写成完整 learner state 上的增量映射，并按增量的各阶矩展开一个 feedback law 的效应（Proposition 5.1，工具）。
- 对 Adam 算出一阶统计量如何依赖 optimizer state：cold 状态下是符号分布，warm 状态下是 Adam 增量的精确均值，以及冻结分母近似和它的误差（Proposition 5.3–5.7）。
- 给出 $`\beta_2`$ 固定时 warmth 的精确演化（Proposition 5.8），以及任意激活率下裁剪的精确统计量（Proposition 5.9）。
- 列出各种常见学习器的一阶统计量（Theorem 5.10）。

**§6 Theorem A：学习器相对的信息界** — [精确陈述](statements/06_learner_relative_information_ZH.md) · [证明](proofs/06_learner_relative_information_ZH.md) · [实验](experiments/06_learner_relative_information_ZH.md)

**本节做什么.**

- 度量 $`K`$ 条示范与 $`n`$ 次查询在"实际学习器与声明任务所在意的那个方向"上揭示了多少关于 oracle 反馈的信息，并证明这个 learner-relative information norm 决定了任何蒸馏器所能达到的最优一阶 fidelity，相差一个介于 $`1/4`$ 与 $`1`$ 之间的因子（Theorem A：[Theorem 6.3](proofs/06_learner_relative_information_ZH.md#s6-3) 与 [Theorem 6.5](proofs/06_learner_relative_information_ZH.md#s6-5)）。
- 指出最优查询是一个 c-optimal design，其目标向量由 optimizer 决定（[Lemma 6.4](proofs/06_learner_relative_information_ZH.md#s6-4)）。
- 以学习器相对的形式推出混叠、replay 的零价值，以及示范与查询的兑换率（[Corollary 6.7](proofs/06_learner_relative_information_ZH.md#s6-7)、[Corollary 6.8](proofs/06_learner_relative_information_ZH.md#s6-8)）。
- 针对 agent 蒸馏的状态覆盖结构，给出显式公式、water-filling 查询分配与 SGD、warm Adam 的学习器权重，并说明同一批示范可以对一个学习器充分、对另一个不充分（[Proposition 6.10](proofs/06_learner_relative_information_ZH.md#s6-10)–[Corollary 6.12](proofs/06_learner_relative_information_ZH.md#s6-12)）。

**§7 Theorem B：Adam 的二阶矩通道** — [精确陈述](statements/07_second_moment_channel_ZH.md) · [证明](proofs/07_second_moment_channel_ZH.md) · [实验](experiments/07_second_moment_channel_ZH.md)

**这一节做什么.**

- 用反向 costate 递推计算 Adam/AdamW 的 continuation value 梯度，并把一条消息的一阶价值拆成参数、动量、二阶矩三个通道（Theorem 7.1）。
- 在 warm、平稳、线性 response 的情形下，对任意消息幅度精确求解：最优幅度、变号点，以及大幅度判据（Corollary 7.2）。
- 由此推出，只看参数的影响分数和余弦分数在 Adam 下都不是一致的排序器（Proposition 7.3），并说明任何一阶排序器在哪里不可靠（Remark 7.5）。
- "Theorem B"指 Theorem 7.1 与 Corollary 7.2 的合称。

**§8 验证与 Theorem C** — [精确陈述](statements/08_verification_ZH.md) · [证明 8.1–8.3](proofs/08_verification_ZH.md)、[证明 8.4–8.12，Theorem C](proofs/08_certificate_ZH.md) · [实验](experiments/08_verification_ZH.md)

**本节做什么.**

- 本节问：只用有限次测量，怎样认证一个 feedback law 足够好，或者给候选 law 排序。它比较四类测量：冻结 cell 上的特征均值、实际状态上完整尾段的配对对比、optimizer 平均增量，以及 surrogate 尾段上的 CRN 配对替换分数。
- 它说明冻结的 audit 需要显式的 transport 项，而在实际状态上用完整尾段做 audit 虽然可靠，代价却很高。
- 主要结果 **Theorem C**（[Theorem 8.8](proofs/08_certificate_ZH.md#s8-8)）直接测量一轮替换的有限效应，所以证书不需要 continuation 梯度，也没有二阶余项。
- 它精确地说明一阶分数在哪里可信（[Corollary 8.9](proofs/08_certificate_ZH.md#s8-9)），以及还有哪些问题没有解决（[Remark 8.12](proofs/08_certificate_ZH.md#s8-12)）。

**§9 机制与方法** — [精确陈述](statements/09_mechanism_and_method_ZH.md) · [证明](proofs/09_mechanism_and_method_ZH.md) · [实验](experiments/09_mechanism_and_method_ZH.md)

**本节做什么.**

- 分离出一个带精确零假设的学习后果机制：边缘分布固定的反馈单元怎样联合组织（[Proposition 9.1](proofs/09_mechanism_and_method_ZH.md#s9-1)）。
- 把 §2–§8 推出的方法 Consequence-Calibrated Experience Distillation（CED）写成编号的步骤：每一步都引用支撑它的结论，每项输出都带有明确的保证，或标为经验结论（[Algorithm 9.2](proofs/09_mechanism_and_method_ZH.md#s9-2)）。
- 推导这一流程背后的六条设计规则（[Remark 9.3](proofs/09_mechanism_and_method_ZH.md#s9-3)），并把受控证据与计划中的真实模型检验分开（[Remark 9.4](proofs/09_mechanism_and_method_ZH.md#s9-4)）。

## 结果的状态

| 节 | 陈述 | 状态 |
|---|---|---|
| §2 | [Theorem 2.2](proofs/02_feasibility_and_acquisition_ZH.md#s2-2), [Propositions 2.3–2.4](proofs/02_feasibility_and_acquisition_ZH.md#s2-3), [Theorem 2.6](proofs/02_feasibility_and_acquisition_ZH.md#s2-6), [Proposition 2.7](proofs/02_feasibility_and_acquisition_ZH.md#s2-7), [Lemma 2.9](proofs/02_feasibility_and_acquisition_ZH.md#s2-9), [Theorem 2.10](proofs/02_feasibility_and_acquisition_ZH.md#s2-10), [Proposition 2.11](proofs/02_feasibility_and_acquisition_ZH.md#s2-11) | 已证明 |
| §3 | [Proposition 3.1](proofs/03_learning_consequence_ZH.md#s3-1), [Theorem 3.2](proofs/03_learning_consequence_ZH.md#s3-2) | 已证明 |
| §4 | [Theorem 4.2](proofs/04_complexity_of_sufficiency_ZH.md#s4-2), [4.4](proofs/04_complexity_of_sufficiency_ZH.md#s4-4), [4.5](proofs/04_rates_and_composition_ZH.md#s4-5), [Proposition 4.6](proofs/04_rates_and_composition_ZH.md#s4-6), [Theorem 4.7](proofs/04_rates_and_composition_ZH.md#s4-7) | 已证明 |
| §5 | [Proposition 5.1](proofs/05_optimizer_state_ZH.md#s5-1), [5.3](proofs/05_optimizer_state_ZH.md#s5-3), [5.5–5.9](proofs/05_warm_adam_ZH.md#s5-5), [Theorem 5.10](proofs/05_learner_statistics_ZH.md#s5-10) | 已证明（有限维，假设明确）；Proposition 5.1 是标准工具 |
| §6（Theorem A） | [Theorem 6.3](proofs/06_learner_relative_information_ZH.md#s6-3), [Lemma 6.4](proofs/06_learner_relative_information_ZH.md#s6-4), [Theorem 6.5](proofs/06_learner_relative_information_ZH.md#s6-5), [Corollaries 6.7–6.8](proofs/06_learner_relative_information_ZH.md#s6-7), [Proposition 6.10](proofs/06_learner_relative_information_ZH.md#s6-10), [Corollary 6.12](proofs/06_learner_relative_information_ZH.md#s6-12) | 已证明（6.3(b) 与 6.5 为局部渐近 minimax 意义） |
| §6 | [Proposition 6.11](proofs/06_learner_relative_information_ZH.md#s6-11) | 在所述访问区间内证明到首项 |
| §7（Theorem B） | [Theorem 7.1](proofs/07_second_moment_channel_ZH.md#s7-1), [Proposition 7.3](proofs/07_second_moment_channel_ZH.md#s7-3) | 已证明 |
| §7 | [Corollary 7.2](proofs/07_second_moment_channel_ZH.md#s7-2) | 在平稳 mean-field 模型中已证明：(a) 精确公式，(b) 带显式误差界的二次律，(c) 显式极限 |
| §8（Theorem C） | [Theorems 8.1–8.3](proofs/08_verification_ZH.md#s8-1), [Proposition 8.4](proofs/08_certificate_ZH.md#s8-4), [Theorem 8.8](proofs/08_certificate_ZH.md#s8-8), [Corollary 8.9](proofs/08_certificate_ZH.md#s8-9), [Proposition 8.10](proofs/08_certificate_ZH.md#s8-10) | 已证明；[Remark 8.11](proofs/08_certificate_ZH.md#s8-11) 中 surrogate 的校准是一个假设 |
| §9 | [Proposition 9.1](proofs/09_mechanism_and_method_ZH.md#s9-1); [Algorithm 9.2](proofs/09_mechanism_and_method_ZH.md#s9-2) 及其设计规则 | 已证明；设计规则为推导所得 |

**开放问题.**

- **Theorem C 的紧的可证明区间界.** 用 surrogate 的先验区间，证书是成立的；要让它能下结论，需要一个远小于先验区间的可证明区间界，例如来自参考尾段的 pathwise Lipschitz 界（[Remark 8.12](proofs/08_certificate_ZH.md#s8-12)）。
- **surrogate 校准的迁移.** Theorem C 用在部分候选对上拟合的校准 $`(\alpha,\varepsilon_{\rm sur})`$。把它推广到新的候选对，需要一个可检验的可交换性条件，目前还没有（[Remarks 8.11–8.12](proofs/08_certificate_ZH.md#s8-11)）。
- **真实模型上的证据.** 实验页上是精确有限计算与受控小网络。Theorem A–C 在真实 agent 蒸馏配方上的检验已在 [Remark 9.4](proofs/09_mechanism_and_method_ZH.md#s9-4) 中设计，尚未运行。

**检验.** 每个实验页开头都有结论表。有一项检验没有达到预设容差：Corollary 7.2(a) 的一组边界设定，记录在 [§7 实验页](experiments/07_second_moment_channel_ZH.md)。

## 复现检验

在仓库根目录运行，只用 CPU：

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --output theory/independent_checks/results_independent_v014.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --output theory/v015_checks/results_v015_independent.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theoretical_framework/figures/make_figures.py
```

实验页上的检查代码编号由套件前缀和检查名组成：

| 前缀 | 套件 | 位置 |
|---|---|---|
| `v014:` | 独立检验 `C1`–`C23` | `theory/independent_checks` |
| `v015:` | 精确有限检验 `V1`–`V9` 与受控网络实验 `E0`–`E12` | `theory/v015_checks` |
| `pkg:` | 修订包自带的检验（`T01`–`T13` 及具名的包内检验） | `docs/direction/v014_handoff`, `docs/direction/v015_revision` |
| `v016:` | §5–§8 新增内容的检验（`C1`–`C6b`） | `theory/v016_checks` |

页面渲染检查工具见 [tools/README.md](tools/README.md)；图的数据来源见 [figures/README.md](figures/README.md)。

## 附录 · 出处

各陈述整理自下列原始文档，链接指向对应的章节。

| 本框架 | 出处 |
|---|---|
| [Theorem 2.2](proofs/02_feasibility_and_acquisition_ZH.md#s2-2), [Propositions 2.3–2.4](proofs/02_feasibility_and_acquisition_ZH.md#s2-3) | [v0.14.1 §2](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#2-first-principles-feasibility-identification-and-use) |
| [Theorem 2.6](proofs/02_feasibility_and_acquisition_ZH.md#s2-6), [Proposition 2.7](proofs/02_feasibility_and_acquisition_ZH.md#s2-7) | [v0.14.1 §3](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#3-a-complete-acquisition-example-perfect-demonstrations-can-be-insufficient) |
| [Lemma 2.9](proofs/02_feasibility_and_acquisition_ZH.md#s2-9), [Theorem 2.10](proofs/02_feasibility_and_acquisition_ZH.md#s2-10), [Proposition 2.11](proofs/02_feasibility_and_acquisition_ZH.md#s2-11) | [v0.15 §4](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#4-make-the-number-of-demonstrations-explicit) |
| [Proposition 3.1](proofs/03_learning_consequence_ZH.md#s3-1) | [v0.14.1 §4](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#4-why-static-evaluation-is-not-learning-sufficiency) |
| [Theorem 3.2](proofs/03_learning_consequence_ZH.md#s3-2), [Remark 3.3](proofs/03_learning_consequence_ZH.md#s3-3) | [v0.14.1 §5](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#51-complete-learning-state-and-continuation-response); [v0.15 §1](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#1-what-the-validation-changes) |
| [Theorem 4.2](proofs/04_complexity_of_sufficiency_ZH.md#s4-2) | [v0.14.1 §6](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#6-exact-and-approximate-information-complexity) |
| [Theorem 4.4](proofs/04_complexity_of_sufficiency_ZH.md#s4-4) | [v0.14.1 §7](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#7-an-explicit-closed-theory-deployment-dependence-composed-with-learning) |
| [Theorem 4.5](proofs/04_rates_and_composition_ZH.md#s4-5) | [v0.14.1](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-14-a-discrepancy-bound-independent-of-training-rounds) |
| [Proposition 4.6](proofs/04_rates_and_composition_ZH.md#s4-6) | [v0.14.1 §8](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#8-exact-order-is-not-approximate-information-complexity) |
| [Theorem 4.7](proofs/04_rates_and_composition_ZH.md#s4-7) | [v0.14.1 §9](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#9-learning-dynamics-can-amplify-the-information-requirement); [v0.15 §7](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#7-experiments-that-discriminate-the-theory) |
| [Proposition 5.1](proofs/05_optimizer_state_ZH.md#s5-1), [Remark 5.2](proofs/05_optimizer_state_ZH.md#s5-2) | [OTLR §2–§3, §6](../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#2-the-expansion); [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Proposition 5.3](proofs/05_optimizer_state_ZH.md#s5-3) | [v0.15 §3](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#3-what-adam-requires-three-distinct-levels-of-sufficiency) |
| [Definition 5.4](proofs/05_warm_adam_ZH.md#s5-4), [Propositions 5.5–5.7](proofs/05_warm_adam_ZH.md#s5-5) | [warm-Adam §2–§3](../docs/direction/v015_extension/Warm_Adam_Response_EN.md#2-warm-one-step-approximation); [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Proposition 5.8](proofs/05_warm_adam_ZH.md#s5-8) | [warm-Adam §4](../docs/direction/v015_extension/Warm_Adam_Response_EN.md#4-how-warmth-evolves-and-a-free-way-to-warm-the-state); [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Proposition 5.9](proofs/05_learner_statistics_ZH.md#s5-9), [Remark 5.11](proofs/05_learner_statistics_ZH.md#s5-11) | [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Theorem 5.10](proofs/05_learner_statistics_ZH.md#s5-10) | [OTLR §4](../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#4-what-each-optimizer-makes-the-feedback-carry); [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [§6](proofs/06_learner_relative_information_ZH.md), Definitions 6.1 – Example 6.13 | [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [§7](proofs/07_second_moment_channel_ZH.md), Theorem 7.1 – Remark 7.5 | [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Theorem 8.1](proofs/08_verification_ZH.md#s8-1) | [v0.14.1](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-18-an-exact-stabilityapproximation-tradeoff) |
| [Theorem 8.2](proofs/08_verification_ZH.md#s8-2) | [v0.14.1](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-19-frozen-finite-interface-selection); [v0.15 §2](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#2-the-corrected-finite-audit-bridge) |
| [Theorem 8.3](proofs/08_verification_ZH.md#s8-3) | [v0.14.1 §11](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#11-fidelity-improvement-and-adaptive-verification); [v0.15 §5](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#5-how-the-new-pieces-compose) |
| [Proposition 8.4](proofs/08_certificate_ZH.md#s8-4), [Remark 8.5](proofs/08_certificate_ZH.md#s8-5) | [OTLR §5](../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#5-using-the-statistic-at-the-actual-states); [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Corollary 8.9](proofs/08_certificate_ZH.md#s8-9) | [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Definition 8.6 – Remark 8.12](proofs/08_certificate_ZH.md#s8-6) | [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Proposition 9.1](proofs/09_mechanism_and_method_ZH.md#s9-1) | [v0.14.1 §12](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#12-one-derived-mechanism-coupling-valid-experience-units) |
| [Algorithm 9.2](proofs/09_mechanism_and_method_ZH.md#s9-2), [Remarks 9.3–9.4](proofs/09_mechanism_and_method_ZH.md#s9-3) | [v0.14.1 §13](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#13-the-derived-few-shot-method-consequence-calibrated-experience-distillation); [v0.15 §6](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#6-method-consequence-an-optimizer-aware-instance-of-ced); [v0.16 修正稿](../docs/direction/2026-10-11_v016_revision_ZH.md) |
