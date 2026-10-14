# 学习响应谱统一框架：理论修订、完整推导与假设审计

**版本：2026-10-13 独立修订稿。** 本文依据 [ICML27_agent_distillation 的固定快照](https://github.com/XueqiC/ICML27_agent_distillation/tree/86586984a4121514fc2e38e20ee772f52245a7bf/theoretical_framework)及该快照的 `PROJECT_STATE.md`。这是一份可评审、可拆分回仓库的完整提案，尚未合并进原仓库。数值检查是本次实际运行的合成数学检查；没有把它们算作真实模型实验。

## 0. 修订结论与贡献定位

建议把整个框架的中心对象定为**学习响应算子**：某种反馈改变了完整学习状态后，在指定后续训练与真实任务评价下，会改变哪些结果、改变多少。谱分解服务于这个对象，而不以参数梯度、文本相似度或环境重建误差代替它。

这个选择可以统一三类问题，但必须保持它们的区别：

| 问题 | 数学对象 | 可以回答什么 |
|---|---|---|
| 当前应保存哪些反馈差异？ | 精确响应算子及其奇异值 | 线性统计接口的压缩误差与重要方向 |
| 压缩后还能连续更新吗？ | 受控学习过程的预测测试空间、Hankel 矩阵与闭合残差 | 预测状态的线性可实现性，以及多轮误差传播 |
| 哪些重要差异还无法从已有数据识别？ | 响应目标相对于观测信息的几何 | 查询必要性；在线性高斯或正则局部模型中的查询设计 |

**本次最重要的补充是第 6 节的联合保证：响应压缩误差、动态预测误差、终点评价误差和状态覆盖误差必须一起支付。** 静态低秩本身不足以保证长期训练收益。

创新性需要重新划界。SVD 最优逼近、可控性/可观性、PSR、价值导向压缩、Bellman 残差和最优实验设计均有成熟先例。本文给出它们在“完整优化器状态—反馈干预—后续训练—任务结果”系统中的一致推导，并明确有限样本与可实现性条件；不把这些经典定理重新命名为原创发现。潜在论文贡献应来自这个系统上的新结构性结果、可计算方法及被验证的预测，不能仅来自理论拼接。

本文中的“谱”也**不是天然的振荡频率**。奇异值衡量反馈方向的后果增益；没有额外的图、距离或时间平移结构，就没有唯一的“高频/低频”。使用“学习响应谱”比把它直接叫作“频率”更准确。

### 0.1 与原框架的关系

| 原位置 | 本稿处理 |
|---|---|
| §2 信息下界与获取 | 保留精确 transcript 的不可区分性；不以 Fisher 正则性支撑总论 |
| §3 精确 continuation response | 升为全文基础，保留完整状态与停止时刻覆盖界 |
| §4 特征充分性与逼近宽度 | 增加明确测度和范数下的响应谱；保留统一范数版本；不混淆统计量个数与位复杂度 |
| §5–7 优化器、一阶信息范数、Adam | 完整优化器进入精确核；一阶与平稳均值场结论降为有条件的解释性分支 |
| §8 CRN 与证书 | 继续用于响应估计；严格区分排序、概率证书与 surrogate 到真实任务的校准 |
| 新增 | 动态闭合、学习过程 PSR、有限时域控制连接、正概率反馈编译、联合误差定理 |

全文标识：**精确**表示在列明基本条件下成立；**条件性**表示额外结构必须检查；**经典实例化**表示证明工具和一般结论已有先例；**研究假说**表示本稿不声称已经证明或实证支持。

## 1. 从第一性原理确定系统边界

固定环境与评价机制 \(E\)、训练预算 \(R<\infty\)、任务结果 \(F_E\in[0,1]\)。结果可以是多任务得分的预先固定加权平均。若原始收益无界，需要先另行处理尾部或归一化，不能直接沿用本文的范围型证书。

完整状态记为

\[
Z_t=(\theta_t,m_t,v_t,t,\text{buffer}_t,\text{sampler}_t,\text{必要交互历史}).
\]

这不是假设原始观察满足 Markov 性。必要时使用完整历史，使条件分布有定义；这个数学状态可能很大，实际算法是否能压缩它是后面要解决的问题。隐藏环境状态不要求被算法直接读取。

反馈消息 \(Y_t\) 可以是一条轨迹、标签、批次或其他实际允许的训练输入。完整更新

\[
Z_{t+1}=U_t(Z_t,Y_t,\Xi_t)
\]

只要求可测。允许非线性、裁剪、离散操作、冷启动 Adam、非平稳学习率以及状态依赖随机性。随机种子状态若影响未来，须纳入状态；以下的条件更新核已包含这些相关性。

反馈协议 \(P\) 通过 \(P_t(dy\mid z)\) 产生消息，诱导核

\[
K_t^P f(z)=\mathbb E[f(U_t(z,Y_t,\Xi_t))\mid Z_t=z;P].
\]

若随机性与消息耦合，使用真实条件核 \(\Xi_t\mid z,y\)，不强行假设独立。目标为

\[
L(P)=\mathbb E_P F_E(Z_R).
\]

比较协议时，环境、起点分布、评价器以及已声明的训练机制相同。若它们不同，差异必须显式加入输入或核。比较模型预训练知识时，同样必须固定双方可用的先验信息 \(I_0\)；“后处理不增加环境信息”是条件于 \(I_0\) 的陈述，不表示预训练模型没有外部知识。

**主线最低条件仅为：有限预算、可测的真实条件核、有界任务结果、明确的比较范围。** 文本 token 独立、梯度平稳、小步长、低秩、全局收缩、Fisher 非退化和已知隐藏状态均不在其中。

### 定理 1：精确学习后果恒等式〔精确；原 §3 的保留核心〕

设参考协议为 \(P^0\)，替代协议为 \(P^S\)。定义

\[
V_R^0=F_E,\qquad V_t^0=K_t^0V_{t+1}^0,
\]

以及固定检查点 \((t,z)\) 的精确消息响应

\[
g_{t,z}(y)=\mathbb E[V_{t+1}^0(U_t(z,y,\Xi_t))\mid z,y].
\tag{1}
\]

则

\[
\boxed{L(P^0)-L(P^S)
=\sum_{t=0}^{R-1}\mathbb E_{Z_t\sim d_t^S}
\int g_{t,Z_t}(y)\,[P_t^0-P_t^S](dy\mid Z_t).}
\tag{2}
\]

**证明。** 在替代协议轨迹上对 \(V_t^0(Z_t)\) 作望远镜求和，再使用 \(V_t^0=K_t^0V_{t+1}^0\)。条件期望将每一项化为 \((K_t^0-K_t^S)V_{t+1}^0\)。没有 Taylor 展开。□

若仅在认证状态集 \(\mathcal C_t\) 中有局部误差界 \(\beta_t(z)\)，令 \(\tau\) 为替代轨迹首次离开这些集合的时刻，未离开则 \(\tau=R\)。停止上述求和可得

\[
|L(P^0)-L(P^S)|\le
\mathbb E_S\sum_{t<\tau}\beta_t(Z_t)+\Pr_S(\tau<R).
\tag{3}
\]

停止点后的两个条件价值都在 \([0,1]\)，所以退出代价只支付一次。原协议的访问分布不能直接替代式 (2) 中的 \(d_t^S\)。这正是反馈改变训练轨迹后会出现的覆盖问题。

## 2. 学习响应算子及其谱

### 2.1 为什么先固定检查点

在一个固定 \((t,z)\) 上，令 \(a\in\mathcal A\) 索引事先声明的“任务结果＋后续训练协议”测试。记

\[
g_a(y)=\mathbb E[\text{测试 }a\text{ 的最终真实得分}\mid z,\text{本轮输入 }y].
\]

这里可改变后续训练时长、任务和合法训练计划。若只放一个标量测试，则算子秩至多为 1；这个低秩几乎没有科学内容。想讨论一般性，需要预先声明有意义的多测试集合。

选择消息参考概率测度 \(\rho\) 和测试权重概率测度 \(\omega\)。在标准可测空间上定义

\[
\mathcal H=L_0^2(\rho)=\{h:\mathbb E_\rho h=0,\ \mathbb E_\rho h^2<\infty\},
\quad \widetilde g_a=g_a-\mathbb E_\rho g_a,
\]

\[
\boxed{(\mathcal R h)(a)=\langle \widetilde g_a,h\rangle_\rho.}
\tag{4}
\]

当反馈法律 \(P,Q\) 对 \(\rho\) 的密度属于 \(L^2\)，取 \(h=dP/d\rho-dQ/d\rho\)，式 (4) 就是两种反馈在所有测试上的精确得分差。

**线性来自概率混合的期望运算，不来自学习器的线性化。** \(U\) 可以完全非线性。不同状态的条件反馈差异通常不同；不能把多个检查点的响应行堆在一起，再偷偷假设它们共享一个与状态无关的 \(h\)。跨状态分析应逐状态应用式 (4)，或者显式使用 \(h(z,y)\) 的条件算子。

由于 \(0\le g_a\le1\)，有

\[
\|\mathcal R\|_{\mathrm{HS}}^2
=\mathbb E_{a\sim\omega}\operatorname{Var}_{y\sim\rho}g_a(y)
\le\frac14.
\tag{5}
\]

因此 \(\mathcal R\) 是 Hilbert–Schmidt 算子，存在奇异分解

\[
\mathcal R h=\sum_{j\ge1}\sigma_j\langle v_j,h\rangle_\rho u_j,
\qquad \sigma_1\ge\sigma_2\ge\cdots\ge0.
\tag{6}
\]

这里的紧性由有界得分和概率测度推出；没有假设有限秩。但谱衰减是否足够快、是否允许有用的压缩，仍是待测的结构事实。

### 定理 2：谱充分性与线性接口复杂度〔精确；经典逼近理论的实例化〕

令模式差异 \(\alpha_j=\mathbb E_Pv_j-\mathbb E_Qv_j\)。则

\[
\|\mathcal R h\|_{L^2(\omega)}^2
=\sum_j\sigma_j^2\alpha_j^2.
\tag{7}
\]

若前 \(r\) 个模式均匹配，且 \(\|h\|_\rho\le B\)，则

\[
\|\mathcal R h\|_{L^2(\omega)}\le B\sigma_{r+1}.
\tag{8}
\]

若只满足 \(|\alpha_j|\le\delta_j\)，则

\[
\|\mathcal R h\|_{L^2(\omega)}^2
\le\sum_{j\le r}\sigma_j^2\delta_j^2+B^2\sigma_{r+1}^2.
\tag{9}
\]

在放宽的有符号输入球 \(\{h\in\mathcal H:\|h\|\le B\}\) 上，最优秩 \(r\) 线性预测接口的最坏重建误差恰为 \(B\sigma_{r+1}\)。因此可定义相对于 \((\rho,\omega,B)\) 的线性接口复杂度

\[
r_\varepsilon=\min\{r:B\sigma_{r+1}\le\varepsilon\}.
\tag{10}
\]

**证明。** 式 (7) 是输出奇异向量的正交性；式 (8)–(9) 是尾部特征值上界。上界由截断 SVD 实现。任意秩不超过 \(r\) 的线性预测算子，在 \(\operatorname{span}(v_1,\ldots,v_{r+1})\) 上都有非零零空间向量；对其归一化，真实输出范数至少为 \(\sigma_{r+1}\)。即使允许非线性解码器，只要编码端是 \(r\) 个线性统计量，利用对称的两个零空间输入也得到相同最坏情形下界。□

这个最优性不能直接变成“任何概率分布、任何压缩算法都至少需要 \(r\) 个数字”。概率密度必须非负；一般非线性编码、无限精度实数编码与有限位复杂度不在上述结论内。

一个无需额外谱衰减假设的推论是：由式 (5)，

\[
(r+1)\sigma_{r+1}^2\le\sum_j\sigma_j^2\le\tfrac14,
\qquad B\sigma_{r+1}\le\frac{B}{2\sqrt{r+1}}.
\]

因此，**在固定密度差预算 \(B\) 与加权均方后果度量下**，取 \(r+1\ge B^2/(4\varepsilon^2)\) 就有线性接口误差不超过 \(\varepsilon\) 的存在性保证。这是有界响应的直接结论，不要求先假定低秩。它不保证逐任务保真、特征可低成本学习、反馈可实现或跨状态复用；当 \(B\) 随目录规模增大时，维数界也会增大。快速实测谱衰减的意义是比这个通用上界允许更小的接口。

### 命题 2.1：概率可行的局部下界

若消息集合有 \(m\) 个元素，\(\rho_y>0\)，则对任意 \(r\) 个线性消息特征，存在单位 \(v\in\operatorname{span}(v_1,\ldots,v_{r+1})\)，使其与这些特征及常数正交。对

\[
P_y=\rho_y(1+av_y/2),\quad Q_y=\rho_y(1-av_y/2),
\quad 0<a\le 2/\|v\|_\infty,
\tag{11}
\]

\(P,Q\) 是概率法律、特征均值相同，而

\[
\|\mathcal R(dP/d\rho-dQ/d\rho)\|\ge a\sigma_{r+1}.
\]

统一选择 \(a\le2\sqrt{\rho_{\min}}\) 足以确保非负。这个比例常数揭示了支持覆盖的成本，不能省掉后宣称全局概率最优性。

### 2.2 不应隐藏的测度选择

对有限候选反馈法律 \(P^1,\ldots,P^J\)，可取

\[
\rho=J^{-1}\sum_{j=1}^J P^j.
\tag{12}
\]

于是每个候选及其凸混合的密度不超过 \(J\)，任意两者的差满足 \(\|h\|_\rho^2\le2J\)。这是由有限候选集构造出来的条件，不必假设现实世界有全局有界密度比。对只有两个法律的等权混合，有 \(\|h\|^2\le4\)。新消息、新候选或新的支持出现时，须重建覆盖或另付误差。

谱依赖 \(\rho\)、\(\omega\)、任务、优化器状态和未来协议；它不是模型的一组永恒常数。改变权重后比较谱，需要报告权重变化。参数坐标的可逆重参数化若保留相同实际过程和测试，不改变这个以可观察后果定义的算子；改变输入或输出的度量则会改变奇异值。

### 2.3 平均小误差与逐任务保证

令

\[
b_j(a)=\langle\widetilde g_a,v_j\rangle_\rho,\quad
e_r(a)=\left\|\widetilde g_a-\sum_{j\le r}b_j(a)v_j\right\|_\rho.
\]

逐测试有

\[
\boxed{|\mathbb E_P g_a-\mathbb E_Qg_a|
\le\left|\sum_{j\le r}b_j(a)\alpha_j\right|+B e_r(a).}
\tag{13}
\]

这是后续训练保证应使用的版本。有限测试集若 \(\omega_a>0\)，也可以把平均误差除以 \(\sqrt{\omega_a}\) 得到该测试的界；稀有测试会付出大代价。连续测试集的平均界不能直接控制任意单点。一个权重为 \(10^{-6}\)、误差为 1 的任务，只贡献 \(10^{-3}\) 的均方根误差。

原 §4 的统一范数结论因此仍有价值：若每个响应可由特征统一逼近到 \(\epsilon\)，匹配特征的任意两概率法律，其得分差不超过 \(2\epsilon\)，无需 \(L^2\) 密度条件。SVD 是一个有明确度量的可计算选择，不取代所有其他逼近范数。

### 2.4 它与“频率”的准确关系

响应核

\[
k(y,y')=\mathbb E_{a\sim\omega}[\widetilde g_a(y)\widetilde g_a(y')]
\]

定义 \(\mathcal R^*\mathcal R\)，其特征值是 \(\sigma_j^2\)。两个消息在许多后续测试中产生相近后果，就会在该核中相近。这提供内生的“后果几何”。但核的非对角项可以为负，它不自动是图邻接矩阵；\(\sigma_j\) 也不自动是图 Laplacian 频率。

| 经典谱概念 | 排序的含义 | 在本框架中的地位 |
|---|---|---|
| 图 Laplacian 特征值 | 相邻节点间的振荡能量 | 只有先有合理图结构时才可额外引入 |
| \(\mathcal R\) 的奇异值 | 单位反馈差异造成的任务后果增益 | 贯穿全文的主对象 |
| 控制系统 Hankel 奇异值 | 既能被输入激发、又能被未来输出看见的方向 | 第 4–5 节中的动态对应物 |
| 信息白化后的响应奇异值 | 后果重要性相对于可识别信息的大小 | 第 8 节中的查询难度；不等于原始 \(\sigma_j\) |

因此应保留“大响应增益模式”，而不能不加论证地说“保留低频”。原 §4.6 中的 Chebyshev 多项式和 Boolean Fourier 度数可以作为具有额外代数结构的例子；多项式次数高也不等于需要很多任意特征，例如 parity 本身就可以作为一个特征。

## 3. 如何估计谱，以及统计代价在哪里

### 3.1 冻结有限测试与反馈目录的证书

固定 \(p\) 个测试、\(m\) 个合法反馈输入，权重 \(w_i,q_j>0\) 且各自和为 1。令 \(G_{ij}\) 是完整运行后的期望得分。一次反馈列也可以是一种可执行反馈提供者；此时是对提供者的混合权重做压缩，不能同时声称识别了其内部所有消息模式。

使用每格 \(n\) 个独立复现实验估计 \(G_{ij}\)。同一复现内允许不同列共享随机数；独立性要求是跨复现，而非假设 token 独立。对 \(\Pi_q=I-\sqrt q\sqrt q^\top\)，加权中心化矩阵为

\[
M=D_{\sqrt w}G D_{\sqrt q}\Pi_q.
\tag{14}
\]

以至少 \(1-\delta\) 的概率，所有元素同时满足

\[
|\widehat G_{ij}-G_{ij}|\le
\epsilon_n=\sqrt{\frac{\log(2pm/\delta)}{2n}}.
\tag{15}
\]

因为权重归一化，加权 Frobenius 误差不超过 \(\epsilon_n\)，中心化投影不扩张范数，所以

\[
\|\widehat M-M\|_{\mathrm{op}}\le\epsilon_n,
\quad |\widehat\sigma_j-\sigma_j|\le\epsilon_n.
\tag{16}
\]

令 \(\widehat\Pi_r\) 投影到估计的前 \(r\) 个右奇异向量，则

\[
\|M(I-\widehat\Pi_r)\|_{\mathrm{op}}
\le\widehat\sigma_{r+1}+\epsilon_n
\le\sigma_{r+1}+2\epsilon_n.
\tag{17}
\]

**证明。** 第一项用逐格 Hoeffding 与并合界；矩阵不等式用 Frobenius 范数和三角不等式；末式再用奇异值扰动界。□

这不需要谱间隙。只有把某个估计方向解释成唯一稳定的“第 \(j\) 个机制”，才需要间隙或对子空间整体做解释。

若从 \(\widehat G\) 构造逐行系数和残差，则真实响应差还须加 \(2\epsilon_n\)：两概率法律对逐点估计误差的积分差不超过该数。因此有限目录中的可检查式为

\[
\beta_i\le\left|\sum_{j\le r}\widehat b_j(i)\alpha_j\right|
+B\widehat e_r(i)+2\epsilon_n.
\tag{18}
\]

式 (18) 中的 \(\alpha_j\) 是真实特征均值差。若只知道估计值 \(\widehat\alpha_j\) 与同时有效的误差界 \(\eta_j\)，右侧还需加上 \(\sum_{j\le r}|\widehat b_j(i)|\eta_j\)，并将第一项中的 \(\alpha_j\) 换成 \(\widehat\alpha_j\)。匹配样本均值不等于已经匹配真实均值。

未观测的新列、新任务、新状态不受式 (15) 自动保护。低秩也不自动保证少量随机格子即可完成矩阵：某个重要方向可能只出现于一个极少采样的格子。任何矩阵补全、泛化或离策略外推都需要另外的覆盖条件。

### 3.2 有限样本下的谱截断选择

若算子和特征已知或已用独立样本冻结，从 \(P,Q\) 各抽取 \(n\) 个独立消息估计 \(\alpha_j\)，定义

\[
\widehat f_r=\sum_{j\le r}\sigma_j\widehat\alpha_j u_j.
\]

则

\[
\mathbb E\|\widehat f_r-\mathcal Rh\|^2
=\sum_{j\le r}\frac{\sigma_j^2}{n}
\left(\operatorname{Var}_P v_j+\operatorname{Var}_Q v_j\right)
+\sum_{j>r}\sigma_j^2\alpha_j^2.
\tag{19}
\]

证明只用均值估计的无偏性与输出方向的正交性，不要求不同模式的噪声独立。式 (19) 给出截断偏差与采样方差的实际权衡。若谱和模式从同一数据中选择，不能原样套用固定特征公式，须样本分割、交叉拟合或另做统一控制。

这个表达式不包含学习 \(g_a\) 自身的成本。获得响应矩阵可能需要大量真实后续训练；“接口维数低”不能被宣传成“端到端样本成本必然低”。特别是只需评估单个已知测试时，直接估计其得分可能更便宜。

### 3.3 CRN、选择与 surrogate 的位置

共享随机数通常有助于比较，但方差下降不是无条件定理：配对差方差取决于协方差。每一侧的边缘分布必须仍然对应真实实验；强行让状态依赖过程消费相同数据、从而改变边缘分布，会改变估计目标。

先冻结候选与审计设计，再使用同时置信界选择，可避免在同一批有限候选中择优造成的遗漏。若不断根据审计数据生成新候选，要么使用新的确认样本，要么采用适合自适应过程的置信序列/统一复杂度分析；原先的固定候选并合界不足。

短尾或 log-likelihood surrogate 只能估计相应的 surrogate 响应。到真实任务的转移需要独立成立的校准误差界。把校准集上的最大残差当作所有未来状态的确定性上界，或用观测样本极差代替总体支持范围，均不能产生原声称的严格证书。

## 4. 控制理论与预测状态：压缩必须能更新自己

### 4.1 控制输入应是什么

把学习过程本身看作受控系统。控制 \(b\in\mathcal B\) 是可执行的反馈提供者、查询方式或训练干预，状态是完整学习/交互状态，输出是预先声明的部署结果。改变学习率、优化器或任务时，应当明确它们是控制的一部分、测试的一部分，还是另一个系统；不能事后随意混用。

令 \(K_b\) 为相应真实状态转移核。将时钟放入状态，达到预算后吸收，即可处理有限时域、非平稳系统。以下不存在无限时间训练的稳定性假设。

对合法控制词 \(w=b_1\cdots b_\ell\)，定义 \(K_w=K_{b_1}\cdots K_{b_\ell}\)。任务输出为 \(F_1,\ldots,F_k\)。最小预测测试空间是

\[
\mathcal V=\operatorname{span}\{1,K_wF_j:\ w\text{ 为所声明的合法后续测试}\}.
\tag{20}
\]

测试集合需在合法前置一步后封闭；有限时域可用按剩余时长分层的空间表达。如果额外允许任意长控制词，必须通过吸收状态维持原训练预算，而不是偷偷放宽任务。

若 \(\mathcal V\) 有有限维基底 \(\psi=(1,\psi_2,\ldots,\psi_d)^\top\)，则存在矩阵 \(A_b\) 与向量 \(c_j\)，使

\[
K_b\psi=A_b\psi,\qquad F_j=c_j^\top\psi.
\tag{21}
\]

对当前状态分布 \(\mu\)，令 \(q(\mu)=\mathbb E_\mu\psi\)。则

\[
q'=A_bq,\qquad \mathbb E_\mu F_j=c_j^\top q.
\tag{22}
\]

**闭合的含义是这些预测统计量足以继续预测下一步的同类统计量。** 状态转移可以非线性；线性的是函数空间中的期望更新。有限维闭合存在时给出精确表示，但一般系统未必有低维闭合。不能把“存在小 \(d\)”列为适用于所有 agent 的默认前提。

这个不变子空间/Krylov 构造与 Poupart–Boutilier 的价值导向 POMDP 压缩有直接先例 [R2]。把输出改成“训练后的真实任务结果”、把状态扩展到完整优化器，形成本文的问题实例；不能据此宣称首次发现价值充分的动态状态。

### 定理 3：学习过程 Hankel 秩与线性预测实现〔精确；经典实现理论的实例化〕

固定初始分布 \(\mu_0\)，定义任务 Hankel 矩阵

\[
H(p,(w,j))=\mu_0 K_pK_wF_j,
\tag{23}
\]

行是控制前缀 \(p\)，列是后续测试 \((w,j)\)，并加入常数输出。测试和前缀都包括合法一步延伸；时钟与吸收机制处理预算。

实现定理要求核在所用控制目录上定义一致，并且所讨论的矩阵包含所需延伸。状态依赖的合法性须通过显式拒绝/吸收机制或分层空间处理；若只有零散合法实验形成的部分矩阵，其秩通常只提供下界，不能直接使用下面的完整实现结论。

若这整个矩阵的代数秩为有限数 \(d\)，则存在 \(d\) 维线性受控预测实现

\[
q(pb)=A_bq(p),\quad H(p,(w,j))=c_j^\top A_wq(p),
\tag{24}
\]

其中 \(A_w=A_{b_\ell}\cdots A_{b_1}\)。任何实现全部这些输入输出期望的线性表示，其维数至少为 \(d\)。

**证明。** 任意 \(d'\) 维表示将 \(H\) 分解成“前缀状态坐标”乘“未来测试系数”，所以 \(\operatorname{rank}H\le d'\)。反向取 \(H\) 的行空间为预测状态空间。前缀延伸 \(p\mapsto pb\) 诱导线性映射：若 \(\sum_i a_iH(p_i,\cdot)=0\)，则对每个 \((w,j)\)，延伸后的和为 \(\sum_i a_iH(p_i,(bw,j))=0\)。因 \((bw,j)\) 仍是合法列，该映射定义良好。选一组基底即得式 (24)，输出由空后缀列读取。□

四个边界非常重要：

1. 这是对给定初始分布和合法实验所张成的**可达预测空间**的最小性，不是对所有隐藏状态或任意非线性表示的最小性。对所有单点状态建立式 (21) 的空间可能更大。
2. 少数已观测行列组成的低秩子矩阵，不证明完整 \(H\) 低秩。需要后缀延伸检验与保留集预测；缺少激励时还会把未观测方向误判成零模式。
3. 当时间已知，应报告每个时间层的预测维数或说明时钟开销。不要把跨时间拼接造成的秩增加全解释成环境复杂度。
4. 上述开环控制词预测的是干预后的期望。要根据新观察在线更新状态，需要下一节的条件化闭合。

固定检查点的 \(g_a(y)\) 可以视为此类 Hankel 对象中的一个局部输入—后缀块：先执行当前消息，再执行未来测试。第 2 节对这个块中心化并选定权重。**局部响应谱与动态 Hankel 谱有结构联系，但一般并不是同一组奇异值。**

### 4.2 真正的 PSR：观察后也能更新

设执行控制 \(b\) 后观察到 \(o\)。离散观察情形定义次概率算子

\[
T_{b,o}f(z)=\mathbb E[\mathbf1\{O=o\}f(Z')\mid z,b].
\]

如果测试空间在所有 \(T_{b,o}\) 下闭合，且包含常数函数，则有 \(T_{b,o}\psi=M_{b,o}\psi\)。设第一坐标对应常数，\(e_1=(1,0,\ldots)^\top\)，则

\[
\boxed{q'=\frac{M_{b,o}q}{e_1^\top M_{b,o}q},\qquad
\Pr(o\mid q,b)=e_1^\top M_{b,o}q.}
\tag{25}
\]

**证明。** 分子是观察事件与下一步特征的联合未归一化期望，分母是该事件概率；按条件期望归一化即可。□

这正是预测状态表示的经典结构 [R3–R4]。对历史—未来的联合观察测试可构造对应 Hankel 矩阵；对正概率历史进行行归一化不改变其秩。若只保留开环奖励测试，可能失去式 (25) 的闭合，不能因此宣称支持任意观察自适应策略。有限观察、有限时域下，可以把策略的所有观察分支纳入测试类；一般观察空间需要相应可测积分版本与另外的估计分析。

式 (25) 的概率解释针对真实可达的预测状态 \(q\)，不表示任意 \(q\in\mathbb R^d\) 都定义合法概率分布。

不把 \(\Pr(o\mid q,b)\ge p_{\min}>0\) 列为全局基本假设。罕见观察确实可能放大条件化误差，应显式支付相关代价，或使用未归一化量进行整体期望分析。随意除以很小的估计概率再声称稳定，是不可接受的步骤。

精确表示的输出来自真实概率过程，因而概率合法；一般的截断 SVD 或任意估计矩阵不自动保正、不自动归一化，也不保证可生成真实反馈。PSR 的预测实现和消息生成器的实现是不同问题，第 7 节单独处理。

### 4.3 静态充分但动态不充分：一个必要反例

令状态 \(z=(s,b)\in\{0,1\}^2\)，当前部署得分 \(F(z)=s\)，下一次更新为 \(U(s,b)=(b,b)\)。两个状态 \((0,0)\)、\((0,1)\) 当前得分相同，但更新后得分分别为 0、1。

只保留 \(\psi=(1,s)\) 能准确预测当前结果，却不能预测下一步的同一特征，因为 \(K s=b\notin\operatorname{span}(1,s)\)。加入 \(b\) 后空间闭合。该例不依赖近似、优化器异常或分布尾部，说明动态闭合检查无法被当前低秩替代。

### 命题 3.1：近似闭合的有限时域误差〔条件性；不要求收缩〕

若对某个固定合法控制序列，有

\[
K_{b_t}\psi=A_t\psi+r_t,\qquad F=c^\top\psi+e_F,
\]

定义 \(q_t=\mathbb E\psi(Z_t)\)，模型预测 \(\widehat q_{t+1}=A_t\widehat q_t\)。则精确地

\[
q_H-\widehat q_H=A_{H-1:0}(q_0-\widehat q_0)
+\sum_{t=0}^{H-1}A_{H-1:t+1}\mathbb E r_t(Z_t),
\]

其中空矩阵乘积为单位阵。因此

\[
|\mathbb EF(Z_H)-c^\top\widehat q_H|
\le\mathbb E|e_F(Z_H)|
+\|c^\top A_{H-1:0}\|\,\|q_0-\widehat q_0\|
+\sum_t\|c^\top A_{H-1:t+1}\|\,\mathbb E\|r_t(Z_t)\|.
\tag{26}
\]

这直接来自线性递推展开。不假设所有增益小于 1；若增益巨大，界就可能无用，须如实报告。矩阵拟合误差也应算进 \(r_t\)。观察自适应情况下，应按分支/条件状态处理，或使用第 6 节适用于一般协议的 Bellman 残差式，不能无解释地搬用固定开环序列公式。

## 5. 控制理论究竟增加了什么

### 5.1 可控而不可观、可观而不可控

可选地，在确有局部可微性的路径附近写完整状态扰动

\[
\delta z_{t+1}=A_t\delta z_t+B_t\delta u_t+\epsilon_t,
\quad\delta o_t=C_t\delta z_t+\zeta_t.
\tag{27}
\]

\(\epsilon_t,\zeta_t\) 是未被忽略的非线性余项。把训练序列切分为过去与未来，在零初始扰动且先忽略余项的线性系统中，有

\[
\delta z_\tau=\mathscr C_\tau\delta u_{\mathrm{past}},\quad
\delta o_{\mathrm{future}}=\mathscr O_\tau\delta z_\tau,
\quad H_\tau=\mathscr O_\tau\mathscr C_\tau.
\]

过去输入使用声明的能量范数，未来输出使用声明的加权范数。定义有限时域 Gramians

\[
W_c=\mathscr C_\tau\mathscr C_\tau^\top,
\qquad W_o=\mathscr O_\tau^\top\mathscr O_\tau.
\]

则非零 Hankel 奇异值满足

\[
\sigma_j(H_\tau)^2
=\lambda_j\!\left(W_c^{1/2}W_oW_c^{1/2}\right).
\tag{28}
\]

**证明。** \(H_\tau^\top H_\tau=\mathscr C_\tau^\top W_o\mathscr C_\tau\)；使用 \(AB\) 与 \(BA\) 的非零特征值一致性即可。□

这解释了为什么环境里很复杂的一个方向仍可能对蒸馏不重要：合法反馈激发不了它，或者未来任务看不见它；两者必须同时满足才产生大的输入—输出增益。相反，\(m,v\) 中的一个变化可能对当前参数影响很小，却强烈影响未来输出。这是完整状态的重要性，而不局限于 Adam 的某个平稳公式。

### 5.2 哪些控制理论结论不能直接搬过来

有限时域式 (28) 不需要渐近稳定。经典平衡截断的 \(H_\infty\) 误差界 \(2\sum_{j>r}\sigma_j\) 属于相应稳定线性时不变系统及其实现条件 [R5]；不能把它写成一般非线性 Adam 训练的现成保证。

对一个 Hankel 块做最优低秩逼近，也不一定得到具有相同时间移位结构的因果动态模型。必须同时检查第 4 节的闭合残差。式 (27) 对真实训练是否准确，要测余项或限制为明确的局部分析；本稿主定理始终使用精确核，不依赖这个分支。

R3 中的振幅律失败不能由“我们换了一个谱解释”自动修复。新框架容纳这个失败：响应可以非二次、非平稳、受到后续梯度变化和数值扰动影响。究竟哪个机制导致失败，仍需分离实验，不能直接认定唯一原因是 Hessian。

## 6. 联合主定理：从响应压缩到真实训练收益

本节给出建议作为论文中心保证的表达式。它连接静态谱压缩与动态预测，避免把多轮性能建立在未验证的平稳近似上。证明使用标准望远镜恒等式；其作为一般数学工具的原创性不应被夸大。

设 \(\widehat V_t\) 是任意固定的近似参考续训价值函数，可以由预测状态模型给出，也可以由其他方法给出。定义参考 Bellman 残差、终点误差及近似消息响应：

\[
r_t^0(z)=K_t^0\widehat V_{t+1}(z)-\widehat V_t(z),\quad
e_R(z)=F_E(z)-\widehat V_R(z),
\]

\[
\overline g_{t,z}(y)=\mathbb E[\widehat V_{t+1}(U_t(z,y,\Xi_t))\mid z,y],
\quad d_t(z)=\int\overline g_{t,z}\,d(P_t^0-P_t^S).
\tag{29}
\]

注意：\(\overline g\) 中本轮更新仍是实际更新。若连更新核也由模型近似，相关误差必须纳入响应估计或 Bellman 残差，不能视为零。

使用第 2–3 节的 \([0,1]\) 范围常数时，应使 \(\widehat V_t\in[0,1]\)，例如明确裁剪；若采用其他已知范围则相应缩放常数。下面的代数恒等式本身只要求相关期望存在。

### 定理 4：精确联合分解与认证界〔精确〕

两协议起点相同，且相关期望存在时，

\[
\boxed{
L(P^0)-L(P^S)
=\mathbb E_S\sum_{t=0}^{R-1}d_t(Z_t)
+\mathbb E_0\sum_t r_t^0(Z_t)
-\mathbb E_S\sum_t r_t^0(Z_t)
+\mathbb E_0 e_R(Z_R)-\mathbb E_S e_R(Z_R).
}
\tag{30}
\]

若局部响应差由谱/特征/统计误差界 \(|d_t(z)|\le\beta_t(z)\) 控制，则

\[
\boxed{
|L(P^0)-L(P^S)|
\le\underbrace{\sum_t\mathbb E_S\beta_t(Z_t)}_{\text{反馈响应压缩与估计}}
+\underbrace{\sum_t(\mathbb E_0|r_t^0|+\mathbb E_S|r_t^0|)}_{\text{动态预测与闭合}}
+\underbrace{\mathbb E_0|e_R|+\mathbb E_S|e_R|}_{\text{真实终点评价}}.
}
\tag{31}
\]

例如 \(\beta_t\) 可取式 (13)，或在冻结有限目录、同时置信事件上取式 (18)。谱必须针对式 (29) 的响应或有误差控制的估计，而不能用一个无校准关系的梯度矩阵替换。

**证明。** 对任意协议 \(j\in\{0,S\}\)，

\[
L(P^j)-\mathbb E\widehat V_0(Z_0)
=\mathbb E_j e_R(Z_R)
+\sum_t\mathbb E_j[K_t^j\widehat V_{t+1}-\widehat V_t](Z_t).
\]

对 \(j=S\)，中括号为 \(r_t^0-d_t\)；对 \(j=0\)，中括号为 \(r_t^0\)。相减得到式 (30)，再取绝对值得到式 (31)。□

这个分解比逐轮先把 \(\widehat V\) 换成精确 \(V\)、再为每轮重复累计全部未来误差更直接，避免了那种证明方式可能产生的多余重复计数。它仍可能因误差项大而失去实用性，没有无条件非空证书的承诺。

### 6.1 闭合残差如何进入主定理

若 \(\widehat V_t=c_t^\top\psi_t\)，且真实参考核满足

\[
K_t^0\psi_{t+1}=A_t\psi_t+\epsilon_t,
\]

则

\[
r_t^0=(c_{t+1}^\top A_t-c_t^\top)\psi_t+c_{t+1}^\top\epsilon_t.
\tag{32}
\]

第一项是模型内部的值递推误差，第二项是动态特征闭合误差。即使训练出低维特征，若 \(\|c_{t+1}\|\) 很大，小的特征残差仍可能成为大的值误差；不能把它藏进一个未检查的常数。

### 6.2 有限覆盖的版本

令 \(\tau^0,\tau^S\) 分别是两协议首次离开认证状态集的时刻。若 \(\widehat V_t\in[0,1]\)，停止式 (30) 的两个望远镜和可得

\[
\begin{aligned}
|L(P^0)-L(P^S)|\le{}&
\mathbb E_S\sum_{t<\tau^S}\beta_t
+\mathbb E_0\sum_{t<\tau^0}|r_t^0|
+\mathbb E_S\sum_{t<\tau^S}|r_t^0|\\
&+\mathbb E_0[|e_R|\mathbf1\{\tau^0=R\}]
+\mathbb E_S[|e_R|\mathbf1\{\tau^S=R\}]\\
&+\Pr_0(\tau^0<R)+\Pr_S(\tau^S<R).
\end{aligned}
\tag{33}
\]

每条轨迹在退出时的真实剩余价值和近似值均处于 \([0,1]\)，故每个退出概率只付一次。相较式 (3)，这里因参考值也在近似，需要控制两个访问分布；不能默默只保留替代协议的一项。若使用精确 \(V^0\)，应直接回到更紧的式 (3)。对预测值裁剪到 \([0,1]\) 后，要以裁剪后的函数重新计算残差，不能假装裁剪仍保持原线性递推。

若残差只在审计分布 \(\nu_t\) 上测得平均值，必须另外控制实际占用与 \(\nu_t\) 的运输/密度比或退出概率。有限审计点上的最大误差，不自动等于整个认证区域的最大误差。

### 6.3 为什么这比引入更强假设更一般

式 (30) 不需要全局 Lipschitz、收缩、平稳梯度、优化器线性、低维真实环境、可逆观察或可微策略。困难被保留为可明确量化的误差项，而没有通过强假设删除。

代价是：一般系统可能无法以小成本估计这些项，或者界可能很松。这是理论诚实的边界。它提供“什么时候压缩有效”的判据，不提供“所有 agent 必然能被便宜压缩”的定理。

## 7. 从预测特征编译回合法反馈

找到 \(r\) 个模式不等于已经找到可执行的合成消息，更不等于已经拥有完整 world model。

给定经核验的有限消息池 \(\{y_1,\ldots,y_m\}\)，记保留特征为 \(\phi(y)=(v_1(y),\ldots,v_r(y))\)，期望目标为 \(\mu_*\)。一个合法混合反馈必须满足

\[
q_i\ge0,\quad\sum_iq_i=1,\quad\sum_iq_i\phi(y_i)=\mu_*.
\tag{34}
\]

它存在当且仅当 \(\mu_*\in\operatorname{conv}\{\phi(y_i)\}\)。由 Carathéodory 定理，若存在，则有支持不超过 \(r+1\) 的解。该结论是经典凸几何；它保证的是特征均值匹配，不是文本意义正确或整个学习过程无损。

稀疏混合可能在 \(\rho\) 很小的消息上放置很大质量，使第 2 节的 \(B\) 变大。因此不能同时无条件声称“只需 \(r+1\) 条消息”和“原先的谱误差界保持不变”。可行的处理是：

- 在优化中增加密度比上界或 \(\chi^2\) 约束，并允许更多支持；
- 使用第 2.3 节的逐点统一逼近界；
- 显式报告所选混合的实际 \(B\) 和逐测试残差。

若目标落在凸包外，应报告不可实现或把投影误差计入模式不匹配；要改善它，需要扩充合法消息池、查询新证据或改变训练接口。不能靠负权重或未经核验的生成消息证明“achievability”。

这个处理替代原 §6.5 中直接发出 \(P^*_{\widehat\theta}\) 的一般性叙述。原叙述可以保留为**当 oracle 法律确实可生成时**的条件性定理，但不能成为现实 agent 蒸馏的默认上界。

## 8. 查询与信息：哪些后果方向值得补数据

### 8.1 精确、非参数的信息障碍

让环境 \(E\) 未知，算法只能看到允许的 transcript \(T\)，其中包括共同先验信息、演示、查询结果和私有随机性。对两个环境 \(E_0,E_1\)，设 transcript 法律为 \(\mathbb P_0^T,\mathbb P_1^T\)，需要估计的后果向量为 \(f_0,f_1\)。任意估计器满足

\[
\inf_{\widehat f}\max_{i=0,1}\mathbb E_i\|\widehat f-f_i\|
\ge\frac{\|f_1-f_0\|}{2}
\left(1-\operatorname{TV}(\mathbb P_0^T,\mathbb P_1^T)\right).
\tag{35}
\]

**证明。** 在两分布的公共部分上，三角不等式给出两种误差之和至少为 \(\|f_1-f_0\|\)。公共部分的质量是 \(1-\mathrm{TV}\)，最大风险至少为两风险和的一半。□

若问题是从两个候选反馈中选择，两个环境的最优候选相反，且选错的真实训练收益损失均至少为 \(\Delta\)，相同二点检验论证给出

\[
\inf_{\widehat b}\max_i\mathbb E_i[\operatorname{regret}(\widehat b)]
\ge\frac\Delta2(1-\operatorname{TV}).
\tag{36}
\]

必须明确这种决策结构；一个目标估计误差下界不能自动当成任意策略类的任务 regret 下界，因为可能有第三种行动在两个环境中都好。

对自适应查询，KL 链式法则沿算法实际观察到的历史分解：

\[
\operatorname{KL}(\mathbb P_0^T\|\mathbb P_1^T)
=\operatorname{KL}(\text{初始信息}_0\|\text{初始信息}_1)
+\sum_\ell\mathbb E_0\operatorname{KL}
\left(Q_0(\cdot\mid H_\ell,X_\ell)\|Q_1(\cdot\mid H_\ell,X_\ell)\right).
\tag{37}
\]

查询策略在两环境下使用同一算法，故策略自身不另贡献信息。配合 Pinsker 可得到具体预算下界。在 iid 演示、状态无关查询模型中才可进一步简写成 \(K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_{Q_x}\)；一般情形需保留历史条件或对历史和查询共同取上确界。

式 (35)–(37) 不要求参数化、Fisher 信息、梯度平稳或高效估计器。对 transcript 的环境无关随机后处理不会减小不可区分性障碍。生成内容仍可能让有限能力学生更容易学习，但那是表示/算法收益，不是新环境证据。

### 8.2 可完整证明的查询设计特例：线性高斯目标

为了得到可计算设计规则，单独声明模型

\[
y=D\vartheta+\epsilon,\quad\epsilon\sim N(0,\Sigma),\qquad
f(\vartheta)=C\vartheta,
\]

其中 \(D,\Sigma,C\) 已知，\(\Sigma\succ0\)，\(\vartheta\in\mathbb R^d\)。\(f\) 是学习后果目标，不要求恢复所有环境参数。给定输出损失权重 \(W\succeq0\)，令

\[
I=D^\top\Sigma^{-1}D,\qquad B=C^\top WC.
\tag{38}
\]

若 \(I\succ0\)，广义最小二乘给出精确恒定风险

\[
\mathbb E\|\widehat f-f\|_W^2
=\operatorname{tr}(BI^{-1})
=\sum_j\lambda_j,
\quad Bv_j=\lambda_jIv_j.
\tag{39}
\]

而且这是无界参数空间上的 minimax 平方风险。

**证明。** \(\widehat\vartheta=I^{-1}D^\top\Sigma^{-1}y\) 的误差协方差为 \(I^{-1}\)，所以风险为迹。取先验 \(\vartheta\sim N(0,\tau^2I_d)\)，Bayes 风险为 \(\operatorname{tr}[B(I+\tau^{-2}I_d)^{-1}]\)，当 \(\tau\to\infty\) 时逼近该迹；任何 minimax 风险都不小于这些 Bayes 风险。谱等式来自 \(I^{-1/2}BI^{-1/2}\) 的特征分解。□

如果 \(I\) 奇异，只需识别影响目标的方向：当 \(C\ker I=0\) 时可在可识别商空间上用伪逆；若存在 \(v\in\ker I\) 而 \(Cv\ne0\)，在这里的无界参数空间上 minimax 风险为无穷。若参数有界，则是不可消除的误差底，不应继续写成无穷。

定义信息白化后的后果算子

\[
\mathcal R_I=W^{1/2}CI^{-1/2}.
\tag{40}
\]

它的奇异值平方就是式 (39) 的 \(\lambda_j\)。这给出清晰区分：原响应谱表示“会造成多大后果”，白化谱表示“这个后果方向相对于已有证据有多难识别”。高增益但已有大量信息的方向，未必值得继续查询。

若一次单位噪声查询的向量为 \(a\)，新增精度 \(aa^\top\)，则由 Sherman–Morrison 恒等式，目标风险下降量精确为

\[
\boxed{\Delta(a)=
\frac{a^\top I^{-1}BI^{-1}a}{1+a^\top I^{-1}a}.}
\tag{41}
\]

有成本时可以比较 \(\Delta(a)/\operatorname{cost}(a)\)，但这种逐步贪心选择不自动是整体预算下的最优策略。标量目标 \(C=c^\top\) 时，\(B=cc^\top\)，退化为原 §6 的目标定向信息几何；多目标对应加权 A/L-optimal design，而非直接把标量 c-optimal 结论扩展成任意非线性任务的最优性。

### 8.3 从特例回到真实学习系统时不能跳过的步骤

在一般正则参数模型中，可用局部信息近似 \(I\)、用**完整精确后果函数**的导数 \(C=\partial f/\partial\vartheta\) 获得局部版本，但必须单独证明局部近似、可识别性和估计器性质。这不是主定理的前提。

如果 \(f(\vartheta)=\int g_\vartheta(y)P_\vartheta(dy)\)，则在可交换微分时

\[
\partial f=\int g_\vartheta\,\partial P_\vartheta
+\int(\partial g_\vartheta)\,dP_\vartheta.
\tag{42}
\]

环境变化还可能改变后续训练、占用分布和测试结果，均须进入 \(\partial g\)。只有响应机制固定时，才可把导数简单写成 \(\mathcal RJ\)。冻结 continuation 梯度后推导的原一阶目标是一个有效的局部目标，但不能自动等同于真实完整训练收益的导数。

这也说明“贯穿同一个响应对象”不意味着强行把所有谱合并成同一组数字：压缩、更长的动态测试和观测信息白化改变了问题的输入输出空间或度量，应分别报告。

### 8.4 对原设计引理的一项直接放宽：允许奇异设计

原 Lemma 6.4 要求每个设计的 \(A(\xi)\succ0\)。对一个固定线性目标，这比必要条件强。保留查询空间紧、\(I_Q(x)\) 连续且半正定，令

\[
A(\xi)=KI_D+n\int I_Q(x)\,\xi(dx),
\quad
\Phi_c(A)=
\begin{cases}
c^\top A^+c,&c\in\operatorname{range}A,\\
+\infty,&c\notin\operatorname{range}A.
\end{cases}
\]

则不要求每个 \(A(\xi)\) 可逆，仍有

\[
\boxed{\inf_\xi\Phi_c(A(\xi))
=\sup_{h\ne0}\frac{(c^\top h)^2}
{K h^\top I_Dh+n\sup_x h^\top I_Q(x)h}.}
\tag{43}
\]

使用 \(0/0=0\)、正数除以 0 为无穷的约定；若右侧有限，存在达到最优值的设计。

**证明。** 对任意半正定 \(A\)，谱分解直接给出
\(\Phi_c(A)=\sup_h(2c^\top h-h^\top Ah)\)。设计空间紧凸，括号内对设计仿射、对 \(h\) 凹，因此 Sion 极小极大交换 [R15] 给出

\[
\inf_\xi\Phi_c(A(\xi))
=\sup_h\{2c^\top h-Kh^\top I_Dh-n\sup_xh^\top I_Q(x)h\}.
\]

沿每个方向对尺度最大化即得到比值式 (43)。也可先加入 \(\varepsilon I\) 使用原正定引理，再令 \(\varepsilon\downarrow0\)：紧性保证最优设计有聚点，而变分表达式保证下半连续，从而给出相同极限。\(\Phi_c\) 是连续函数族的上确界，故在紧设计空间上下半连续，有限最优值可以达到。□

这个放宽是经典最优设计对偶性的直接延伸，不声称为新的设计理论。它允许不识别对线性目标无用的方向。但原 Theorem 6.5 还要控制非线性 \(\Psi(\widehat\theta)\) 的二阶误差；不能因为式 (43) 成立，就把它所需的估计条件一并删除。

## 9. 一套具体可执行的研究方法

建议保留 CED 的获取—编译—审计结构，改为以下过程。它是由上述定理约束的方法提案，尚不是已经取得真实基准优势的算法。

1. **定义比较范围。** 固定任务、剩余训练预算、可用完整检查点、评价机制与合法反馈/查询目录。列出所有预算：真实环境调用、teacher 调用、checkpoint 恢复、后续训练、终点评测及额外模型拟合。
2. **测量当前后果。** 在可恢复的完整状态上做合法反馈替换，执行真实后续训练或明确标记的短尾测试，获得响应矩阵。若使用 CRN，核验每侧的边缘协议。所有候选选择前冻结确认集。
3. **拟合响应接口。** 明确 \(\rho,\omega\)，估计带置信误差的谱。按式 (13)、(18)、(19) 选维数，报告逐任务残差。将 token、轨迹或提供者作为列时分别说明统计单位。
4. **检查能否继续更新。** 扩展后缀和新检查点，拟合预测状态递推；在独立轨迹上测 Bellman/特征闭合残差。若含观察条件化，检查分母、概率合法性与分支覆盖。必要时增加模式或缩短认证的训练时域。
5. **只查询能解决的缺口。** 区分“未被激发”“影响结果但信息不足”“预测模型错误”“反馈不可实现”。在线性高斯/验证过的局部模型中用式 (41) 定向查询；否则把该分数当待比较的设计启发式，使用真实查询结果验证其收益。
6. **编译合法反馈。** 在已核验消息或提供者的凸包中匹配保留特征，控制密度比和编译残差。无法实现时扩充证据，不使用负权重当作真实采样法律。
7. **先认证再比较。** 用独立或同时有效的估计把式 (31)/(33) 各项填入；对候选的估计优势减去其有效误差半径后再决定是否替换参考协议。若界不能区分候选，报告“未能认证”，仍可单独报告点估计排序。

这套方法目前的成本上界可以很朴素：完整有限目录审计需要 \(pmn\) 次相应的后续评估，另加闭合检查与真实评价。没有额外结构时不承诺优于直接训练。真正有价值的算法进展，是在保持覆盖与认证的同时显著降低这个成本；仅报告矩阵秩低不够。

### 9.1 可以被否证的关键预测

| 预测 | 实验对照 | 会否定什么 |
|---|---|---|
| 响应保留方向比文本/梯度压缩更能保持后续结果 | 等反馈与审计总成本下比较，使用保留任务与消息 | 若只在拟合矩阵上成立，否定可迁移接口的实用性 |
| 完整优化器状态改变未来可见的响应方向 | 固定参数，构造合法但不同 \(m,v\) 状态；同一反馈与后续任务 | 若差异始终可忽略，削弱特定场景中的优化器状态贡献 |
| 动态闭合误差解释静态压缩的长期失败 | 同秩、相近当前重建误差，比较是否加入后缀闭合训练 | 若闭合残差不预测失败，需要修正测量范围或核心方法 |
| 查询要匹配后果与信息，而非只看梯度/不确定性 | 比较均匀、纯信息、原 SGD/Adam 权重与式 (41) 设计 | 若额外拟合/审计成本抵消收益，不支持经济优势 |
| 压缩有效时应能获得非空且正确的真实收益证书 | 预注册评价任务、候选与范围；报告误判和覆盖率 | 只有排序提升而无非空证书，不能声称已解决认证问题 |

扩大测试类时，**精确充分空间的维数不会下降**，因为旧空间包含于新空间。若保留旧测试的绝对权重并新增测试，\(\mathcal R^*\mathcal R\) 增加一个半正定项。可是重新归一化权重，或把“短尾测试”直接替换为“长尾单一测试”，并不保证奇异值单调。这是预测实验应避免的另一处误用。

## 10. 假设审计：哪些保留，哪些退出主线

本次核查覆盖原 README 全部主线、A/B/C 关键陈述与证明、充分性对偶、最优设计、Adam 振幅律和配对替换证据，并对新增推导逐项检查。它不是对仓库每一页证明、全部训练代码的形式化验证。因此结论是“修订主线不再依赖下列过强条件”，而不是保证所有历史版本已无任何漏洞。

| 条件或隐含跳步 | 判断 | 修订后的处理 |
|---|---|---|
| 有限预算、结果有界、真实核可测 | 常规且与问题直接对应 | 主线保留；范围与归一化公开 |
| 只用参数 \(\theta\) 就是完整学习状态 | 对带记忆优化器一般不成立 | 纳入 \(m,v\)、时钟、缓冲区与必要历史 |
| 原始观察必须 Markov | 不必要 | 使用完整历史或预测状态；不要求观察本身 Markov |
| 梯度/token/重复数据彼此独立 | 在实际训练中通常过强 | 主线不需要；只在明确的独立实验复现处使用独立性 |
| 仿射平均学习器、独立经验块、有限决策次数 | 原 §4 的受限模型条件 | 保留为说明阶数与 cumulant 的例子；不推广到任意优化器 |
| 小学习率就保证完整状态 Taylor 余项为 \(O(\eta^2)\) | 一般不成立；优化器状态改变可有不同尺度 | 主线使用精确核；局部近似单独量化完整状态余项 |
| 冻结 continuation 梯度且与环境参数无关 | 对原一阶目标合法，对真实收益导数可能不足 | 明确目标定义；需要完整导数时使用式 (42) |
| DQM 与对所有查询统一的 KL 二次展开 | 正则局部模型常用，但不是任意黑盒环境事实 | 留在 Fisher 分支；DQM 本身不自动替代额外的统一 KL 条件 |
| 每种设计 \(A(\xi)\succ0\) | 对线性目标过强 | 用式 (43) 的伪逆及目标可识别性替换 |
| 所有参数方向信息都按 \(N\) 增长 | 对全参数估计方便，对线性后果可能多余 | 主下界不用；非线性 achievability 需要什么则单独证明 |
| 高效两阶段估计器存在 | 本身是实质条件，不是由 c-optimal 设计自动推出 | 高斯特例完整证明；一般情形不宣称自动达到下界 |
| 可生成任意 \(P^*_{\widehat\theta}\) | 对真实蒸馏明显强 | 改为有限合法池的凸可实现性与显式编译误差 |
| Adam 恰好处于 \(m=\mathbb Eg,v=\mathbb Eg^2\) | 严格的平稳理想化 | 退出主定理，保留为解释性特例 |
| 后续梯度不依赖 \(\theta\)，终点得分仿射 | 对长程神经网络训练过强 | 不用于谱、动态或收益主定理 |
| \(\mathbb E[m/\sqrt v]=\mathbb Em/\sqrt{\mathbb Ev}\) | 一般错误 | 原均值场定义应明确为替代模型；实际误差另测 |
| 无 bias correction、无 weight decay、\(\epsilon=0\) | 特例优化器设定 | 原振幅公式适用范围写明；真实响应核包含实际实现 |
| 冷/暖状态由少数 warmth 分位数决定 | 不足以控制任务加权误差 | 使用任务加权误差或实际响应审计 |
| 低秩、快速谱衰减、有限维 PSR | 不是普遍事实 | 作为测量结果或条件性压缩性质；保留不能压缩的可能 |
| 全局有界密度比 | 可能过强 | 对有限目录用混合参考测度构造；目录外重新付覆盖代价 |
| 低平均谱误差意味着所有任务/状态都准确 | 错误 | 逐测试残差、权重代价及实际占用覆盖 |
| 所有动态矩阵都稳定/收缩 | 不必要且可能不真实 | 有限时域显式增益或 Bellman 残差；不导入 LTI 的无限时域界 |
| 每个观察概率有统一正下界 | 对丰富观察空间过强 | 条件化仅在正概率事件定义；稀有分支另计误差 |
| 全环境可由稀疏因果图/可分 DBN 表示 | 可选建模结构，不是当前任务必需 | 不加入主线；使用时单独比较结构失配 |
| SVD/PSR 预测可以直接生成合法消息 | 不成立 | 第 7 节的正概率编译与可实现性检查 |
| 冻结候选、独立复现、已知总体范围 | 有限样本证书的真实条件 | 保留并按实验协议落实；自适应选择须另处理 |
| 样本极差就是总体范围 | 错误 | 只能作为诊断，不用于声称原严格置信保证 |
| surrogate 校准在所有新状态都成立 | 强且不可从有限校准残差直接推出 | 显式转移误差、保留集验证；无依据则只报告 surrogate 排序 |
| 可随意恢复检查点、分叉未来、查询任意状态且免费 | 对真实系统不成立 | 限定可执行干预集合，把分叉和评估成本全部计入预算 |

### 10.1 无法同时满足的更强愿望

以下障碍不能通过换一套术语消除：

- **任意反馈法律的精确/逐任务保真不能普遍强压缩。** 取 \(m\) 条消息、\(m\) 个指示函数响应，中心化响应秩为 \(m-1\)。任意反馈差异都可能影响某个任务。这不否定第 2 节在固定 \(B\) 和平均度量下的近似接口上界。
- **当前充分性与任意未来充分性不可等同。** 第 4.3 节给出最大得分差为 1 的反例。
- **没有信息的方向不能靠生成自动辨认。** 两环境 transcript 完全相同时，式 (35) 给出不可消除的误差。
- **完全一般的 surrogate 不保证真实结果排序。** 令真实结果为 \(F\)、surrogate 为 \(1-F\)，排序直接相反。有限校准集没有覆盖的区域可存在同样现象。
- **没有激励与覆盖，就不能判断某个谱方向不存在。** 观测到的零响应可能只是没有测到相关干预。

因此应主动放弃“无需任何结构就能得到统一低维、低成本、全任务最优蒸馏”的表述。更可靠的主张是：给出结构可被测量、失配代价可被计算、失败可被识别的理论。

## 11. 现有证据必须怎样重写

仓库 README 的范围说明仍表示真实模型测试结果尚未上理论页；同一快照的 `PROJECT_STATE.md` 已记录 R0/R1/R3 的更新。论文与理论页应同步这些事实，尤其是负结果。

| 已有证据 | 能支持的陈述 | 不支持的陈述 |
|---|---|---|
| 控制面板 aligned CRN 点估计：\(r=256\) 时 1,687 个非零真实差异中 48 个错误符号，约 2.8% | 在该冻结面板上，有限替换测量的排序优于已比较的一阶/方向代理 | 已获得普遍真实任务证书或低成本算法优势 |
| 同面板严格总体范围 \(B=2\) 时，aligned 证书为 0/1,728 | 当前严格界在该预算下基本为空 | 57.2% 的诊断覆盖等于严格认证覆盖 |
| 使用观测极差的 aligned 诊断：989/1,728，57.2% | 值得研究更紧、可证明的范围或序贯界 | 观测极差已经证明了总体尾部被控制 |
| R3 在真实模型 p10：\(c=0.5,1,2,4,8\) 的实测 surrogate 增量约为 +0.0005/+0.0012/+0.0011/+0.0007/+0.0002；均落在诊断半径 0.004–0.007 内 | 原振幅律的该次真实模型预测未获支持 | 已验证统一的二次曲线、最优振幅或 \(2\lambda^*\) 阈值 |
| R3 原预测约 −0.019 到 −0.080，较测量大 15–300 倍且符号相反 | 必须下调平稳线性响应特例的主线地位 | 已确定唯一失败机制；记录中的 Hessian 解释仍是假说 |
| R1 的部分真实模型 surrogate 排序 P1>P3：点估计约 +0.00138，严格半径约 0.325 | 当前 surrogate 的初步排序结果 | 真实任务收益已认证；其 surrogate alignment 尚未校准 |
| R0 预注册 warmth 检验失败；事后修正分母后的误差另有记录 | 测量定义需要修正，并保留“事后”标签 | 将事后指标悄悄替换预注册指标并宣称通过 |
| R2 在此快照未提供最终完成的支持性结果 | 保留为待完成验证 | 查询分配已在真实模型证明有效 |

本稿没有运行上述真实模型实验，也没有改变这些已有记录。新框架需要自己的谱、闭合、编译和成本实验；原来的排序优势不能自动算作这些新主张的验证。

## 12. 与最接近工作的区别，以及论文该主张什么

文献核对截至 2026-10-13；这是针对近邻方向的检索，不是穷尽性新颖性证明。

| 近邻工作 | 已有核心内容 | 本项目需要真正新增并验证的部分 |
|---|---|---|
| Value-Directed Compression of POMDPs，2002 [R2] | 保留价值预测，并要求压缩状态可继续更新 | 完整学习器而非固定环境控制状态；学习后果的可测误差与蒸馏实现 |
| PSR，2001/2004 [R3–R4]；PSTD，2010 [R6] | 以未来实验预测表示状态；谱方法可构建价值相关预测特征 | 反馈干预改变优化器后，跨后续训练测试的预测与认证 |
| Value Equivalence，2020 [R7] | 模型相对于指定策略和函数类保留相关更新 | 将相对性扩展到实际训练更新、优化器记忆和反馈获取预算 |
| LESS，2024 [R9]；GRACE，2025 [R10]；优化训练数据的 metagradient 工作 [R12] | 训练数据价值可依赖学习器、梯度或训练过程 | 精确有限替换、完整状态、动态闭合与不可识别性应带来可验证优势 |
| Adam-aware in-run Data Shapley，2026 [R11] | 将 Adam 机制纳入数据归因 | 不能再以“考虑 Adam 的数据价值”作为单独原创点 |
| MIST-WM，2026 [R8] | 主动探索与任务充分 world model 的联合建模 | 聚焦“使指定学习过程充分”的经验，而不只预测环境控制状态；必须与其任务充分性叙述区分 |

更合适的中心问题是：**对一个指定的学习器及任务集合，哪些可获取、可编译、可动态维持的反馈差异足以保持训练后的结果？**

本稿已完成的理论工作有明确边界：

1. 把精确后果算子的谱、逐任务残差、动态闭合与信息几何放在相容的空间和范数中，给出完整证明。
2. 推导式 (30)–(33) 的联合分解，避免从静态低秩直接跳到长期性能。
3. 放宽原设计引理的全设计正定要求，并分离线性目标与一般非线性 achievability。
4. 用反例阻止不成立的泛化，明确概率编译、覆盖、统计识别与真实评价各自的成本。

这些足以形成更扎实的理论骨架，**还不足以仅凭此认定 ICML best paper 级别**。最值得争取的进一步贡献是：在可执行、计成本的设置下，证明并验证比现有价值压缩/PSR 方法更强的“学习后果压缩—获取—认证”结果。若最终只是把经典算子理论换到 optimizer state 上而没有新结构或实证发现，创新性仍会不足。

### 12.1 建议的论文叙述

> 本文研究相对于学习过程的经验充分性。我们以反馈干预对完整学习状态及后续任务结果的精确影响定义学习响应算子，其谱刻画线性反馈接口在指定测试类中的逼近复杂度。为了支持连续训练，我们将后续测试扩展为受控预测系统，并通过动态闭合残差连接局部压缩与最终训练收益。该框架保留状态覆盖、反馈可实现性及数据不可识别性所造成的误差，不依赖梯度平稳或优化器线性。在线性高斯观测模型中，查询设计进一步由信息白化后的响应谱决定。一般系统上的谱衰减、估计成本及真实任务收益需要实验验证。

这个版本的叙述避免把尚未验证的低秩、均值场或 surrogate 校准放进摘要的无条件结论。

### 12.2 建议回填原仓库的具体位置

| 文件/模块 | 需要的改动 |
|---|---|
| `theoretical_framework/README.md` | 主线改为“精确响应—谱充分性—动态闭合—信息与编译—认证”；加入真实模型负结果与证书空缺 |
| `statements/04-02_sufficiency_duality_transfer.md` 及证明 | 保留统一范数结果；增加本稿第 2–3 节的带测度谱版本与概率可行边界 |
| 新的 dynamic sufficiency 陈述与证明 | 放入第 4 节的可达预测空间、条件观察 PSR 与闭合反例；明确经典来源 |
| `proofs/06-04_design_minimax_equality.md` | 增加式 (43) 的半正定版本，区分信息不可识别与无用参数方向 |
| `proofs/06-05_achievability.md` | 保留原有条件，不以可生成 oracle 法律作为默认现实假设；链接第 7 节编译条件 |
| `statements/07-02_stationary_amplitude_law.md` 及实验页 | 标题与摘要明确“stationary mean-field model”；加入 R3 未支持预测的结果 |
| §8 陈述与证明 | 新增第 6 节联合保证；范围、独立确认、覆盖和 surrogate 误差逐项列明 |
| `statements/09-04_scope_of_evidence.md` | 与固定快照 `PROJECT_STATE.md` 同步，明确新谱/PSR 主张尚无真实模型验证 |

## 13. 本次数值核验与可复现性

实际运行环境：Python、NumPy 2.3.5；固定随机种子 20261013。以下检查用于发现符号、矩阵权重、概率可行性及有限时域递推错误；数学证明在正文中，数值通过不替代证明，也不代表真实模型上的低秩或样本效率。

| 检查 | 实际结果 |
|---|---|
| 加权中心化响应谱恒等式 | 绝对误差 \(5.20\times10^{-18}\) |
| 最优秩截断残差等于 \(\sigma_{r+1}\) | 绝对误差 \(1.39\times10^{-17}\) |
| 任意 3 个特征的概率可行下界构造 | 均值匹配误差 \(6.94\times10^{-17}\)；真实后果差 0.07066，大于下界 0.05740 |
| 估计谱的无间隙残差界 | 实际残差 0.05795，小于界 0.06594 |
| 四步非线性 Adam-like 完整状态精确收益分解 | 收益差 0.07616；恒等式误差 \(9.71\times10^{-17}\) |
| 式 (30) 的联合分解 | 恒等式误差 \(2.78\times10^{-17}\)；真实绝对差 0.07616，小于式 (31) 界 0.17818 |
| 相同参数、不同优化器状态 | 下一步任务响应相差 0.01282 |
| 静态充分但不闭合 | 特征空间由 2 维增至 3 维；原闭合残差 0.5，未来得分差 1 |
| 非收缩有限时域 Hankel/Gramian 恒等式 | 最大单步增益 1.28282；谱恒等式误差 \(1.10\times10^{-15}\) |
| 高斯目标风险与单次查询收益 | 两个恒等式误差分别 \(1.22\times10^{-15}\)、\(2.78\times10^{-17}\) |
| 奇异设计的原始/对偶最优值 | 所有设计均有一个不可识别的无用方向；两侧最优值同为 2.25，差为 0 |
| 稀有任务反例 | 平均误差 0.001，稀有任务误差 1 |
| 不存在普遍强压缩的反例 | 8 条消息的中心化响应秩为 7 |

状态为 **PASS**。附录 A 包含本次运行的完整脚本，保存为 `verify_response_framework.py` 后运行 `python3 verify_response_framework.py` 即可重现并输出 `verification_results.json`。不需要网络、模型权重或 GPU。

## 参考资料

所有文献用于明确已有理论与近邻贡献，没有将其当作本项目的实验结果。

- **[G0] 项目理论快照。** [theoretical_framework，86586984a4121514fc2e38e20ee772f52245a7bf](https://github.com/XueqiC/ICML27_agent_distillation/tree/86586984a4121514fc2e38e20ee772f52245a7bf/theoretical_framework)。重点核对 §3、§4.2、§6.2–6.6、§6.11、§7.1–7.2、§8.8、§8.12、§9.4 及相应证明/实验。
- **[G1] 项目实验状态。** [PROJECT_STATE.md，同一快照](https://github.com/XueqiC/ICML27_agent_distillation/blob/86586984a4121514fc2e38e20ee772f52245a7bf/PROJECT_STATE.md)。第 11 节的数据取自该记录与原 §8.8 实验页。
- **[R1] Eckart, C.; Young, G. (1936).** The Approximation of One Matrix by Another of Lower Rank. *Psychometrika* 1, 211–218. [原论文](https://doi.org/10.1007/BF02288367)。
- **[R2] Poupart, P.; Boutilier, C. (2002).** Value-Directed Compression of POMDPs. *NIPS 15*. [原论文](https://papers.neurips.cc/paper/2192-value-directed-compression-of-pomdps.pdf)。
- **[R3] Littman, M.; Sutton, R.; Singh, S. (2001).** Predictive Representations of State. *NIPS 14*. [会议论文页](https://papers.nips.cc/paper/1983-predictive-representations-of-state)。
- **[R4] Singh, S.; James, M.; Rudary, M. (2004).** Predictive State Representations: A New Theory for Modeling Dynamical Systems. *UAI 2004*. [作者论文存档](https://arxiv.org/abs/1207.4167)；2012 是该存档上传年份，不是首次发表年份。
- **[R5] Moore, B. C. (1981).** Principal Component Analysis in Linear Systems: Controllability, Observability, and Model Reduction. *IEEE Transactions on Automatic Control*. [DOI](https://doi.org/10.1109/TAC.1981.1102568)。另见 [Stanford Lall 课程中的 balanced truncation 说明](https://lall.stanford.edu/engr210a/lectures/lecture11_2001_11_07_01.pdf)。
- **[R6] Boots, B.; Gordon, G. (2010).** Predictive State Temporal Difference Learning. *NIPS 23*. [会议论文页](https://proceedings.neurips.cc/paper/2010/hash/3473decccb0509fb264818a7512a8b9b-Abstract.html)。
- **[R7] Grimm, C.; Barreto, A.; Singh, S.; Silver, D. (2020).** The Value Equivalence Principle for Model-Based Reinforcement Learning. *NeurIPS 2020*. [原论文页](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)。
- **[R8] Feng, F. 等 (2026).** Learning Task-Sufficient World Models by Synergizing Agentic Exploration and Structured Modeling. [arXiv:2607.04409v1](https://arxiv.org/html/2607.04409v1)。这里按预印本引用，不根据页面关键词推断会议录用状态。
- **[R9] Xia, M. 等 (2024).** LESS: Selecting Influential Data for Targeted Instruction Tuning. *ICML 2024*. [PMLR 原论文页](https://proceedings.mlr.press/v235/xia24c.html)。
- **[R10] Panigrahi, A. 等 (2025).** In Good GRACEs: Principled Teacher Selection for Knowledge Distillation. [arXiv:2511.02833v1](https://arxiv.org/html/2511.02833v1)。
- **[R11] Ding, M.; Zhang, Z.; Wang, D.; Hu, L. (2026).** In-Run Data Shapley for Adam Optimizer. [arXiv:2602.00329v1](https://arxiv.org/html/2602.00329v1)。
- **[R12] Engstrom, L. 等 (2025).** Optimizing ML Training with Metagradient Descent. [arXiv:2503.13751v1](https://arxiv.org/html/2503.13751v1)。
- **[R13] Elfving, G. (1952).** Optimum Allocation in Linear Regression Theory. *Annals of Mathematical Statistics*. [DOI](https://doi.org/10.1214/aoms/1177729442)。另见原框架引用的 Pukelsheim, *Optimal Design of Experiments*，及其[出版版本](https://doi.org/10.1137/1.9780898719109)。
- **[R14] Shuman, D. 等 (2013).** The Emerging Field of Signal Processing on Graphs. [作者论文存档](https://arxiv.org/abs/1211.0053)。用于区分图频率与本文的响应增益谱。
- **[R15] Sion, M. (1958).** On General Minimax Theorems. *Pacific Journal of Mathematics* 8, 171–176. [原论文](https://msp.org/pjm/1958/8-1/pjm-v8-n1-p14-p.pdf)。

## 附录 A：完整复现脚本

下面代码与本次运行的脚本一致。

```python
"""Deterministic checks of the revised framework; synthetic, not LLM evidence.
Run with Python >=3.10 and NumPy. No network, model downloads, or GPU required.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def check_close(a, b, tol=2e-11):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert error < tol, (error, a, b)
    return error


def main():
    rng = np.random.default_rng(20261013)
    out = {"seed": 20261013, "numpy": np.__version__,
           "scope": "synthetic mathematical checks only"}

    # Weighted, centered response operator. Euclidean x = sqrt(q) * h.
    p, m, r = 7, 11, 3
    w = rng.uniform(.3, 1.3, p); w /= w.sum()
    q = rng.uniform(.3, 1.3, m); q /= q.sum()
    G = sigmoid(rng.normal(size=(p, m)))
    Pi = np.eye(m) - np.outer(np.sqrt(q), np.sqrt(q))
    R = np.sqrt(w)[:, None] * G * np.sqrt(q)[None, :] @ Pi
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    x = Pi @ rng.normal(size=m); x /= np.linalg.norm(x)
    spectral_error = check_close(np.linalg.norm(R @ x)**2,
                                np.sum(s**2 * (Vt @ x)**2))
    Rr = (U[:, :r] * s[:r]) @ Vt[:r]
    truncation_error = check_close(np.linalg.norm(R - Rr, 2), s[r])
    hs = float(np.linalg.norm(R, 'fro')**2)
    assert hs <= .25 + 1e-14
    out["weighted_spectrum"] = {
        "spectral_identity_error": spectral_error,
        "best_rank_residual_error": truncation_error,
        "hilbert_schmidt_squared": hs,
        "singular_values": s.tolist()}

    # Arbitrary r features: construct two positive laws with matching features.
    features = rng.normal(size=(r, m))
    top = Vt[:r+1].T
    B = (features * np.sqrt(q)[None, :]) @ top
    _, _, Bt = np.linalg.svd(B, full_matrices=True)
    v = top @ Bt[-1]
    h = v / np.sqrt(q)
    alpha = 1.8 / np.max(np.abs(h))
    P = q * (1 + alpha*h/2)
    Q = q * (1 - alpha*h/2)
    assert min(P.min(), Q.min()) > 0
    check_close([P.sum(), Q.sum()], [1, 1])
    feature_error = check_close(features @ P, features @ Q)
    gap = float(np.linalg.norm(np.sqrt(w) * (G @ (P-Q))))
    lower = float(alpha * s[r])
    assert gap >= lower - 1e-13
    out["positive_law_lower_bound"] = {
        "feature_match_error": feature_error, "actual_response_gap": gap,
        "guaranteed_gap": lower, "smallest_probability": float(min(P.min(), Q.min()))}

    # Estimated singular subspaces: an eigengap is unnecessary for error control.
    eps = .004
    Ghat = np.clip(G + rng.uniform(-eps, eps, G.shape), 0, 1)
    Rhat = np.sqrt(w)[:, None] * Ghat * np.sqrt(q)[None, :] @ Pi
    Uh, sh, Vht = np.linalg.svd(Rhat, full_matrices=False)
    Phat = Vht[:r].T @ Vht[:r]
    noise = float(np.linalg.norm(R-Rhat, 2))
    residual = float(np.linalg.norm(R @ (np.eye(m)-Phat), 2))
    assert noise <= eps + 1e-13
    assert residual <= s[r] + 2*eps + 1e-13
    out["estimated_spectrum"] = {"operator_noise": noise,
        "entrywise_bound": eps, "projection_residual": residual,
        "guaranteed_residual": float(s[r] + 2*eps)}

    # Exact nonlinear Adam-like learning dynamics, including m, v, counter.
    # Gradients depend nonlinearly on theta; no zero-Hessian or stationarity premise.
    horizon = 4
    z0 = (.2, .03, .08, 2)

    def update(z, y):
        theta, mom, vel, count = z
        grad = theta + .2*theta**3 - y
        mom = .7*mom + .3*grad
        vel = .85*vel + .15*grad**2
        count += 1
        mh = mom / (1-.7**count)
        vh = vel / (1-.85**count)
        theta = theta - .17*mh/(np.sqrt(vh)+.01)
        return tuple(map(float, (theta, mom, vel))) + (count,)

    def prob(z, kind):
        return float(sigmoid((.6 if kind == 0 else -.4) + .5*z[0]-.2*z[1]))

    def terminal(z):
        return float(sigmoid(1.3*z[0]-.25*z[0]**2))

    @lru_cache(None)
    def V(t, z, kind):
        if t == horizon:
            return terminal(z)
        py = prob(z, kind)
        return ((1-py)*V(t+1, update(z, -1), kind)
                + py*V(t+1, update(z, 1), kind))

    def expectation(t, z, kind, fun):
        py = prob(z, kind)
        return (1-py)*fun(t+1, update(z, -1)) + py*fun(t+1, update(z, 1))

    def occupancies(kind):
        levels = [[(z0, 1.)]]
        for t in range(horizon):
            nxt = []
            for z, mass in levels[-1]:
                py = prob(z, kind)
                nxt.extend([(update(z, -1), mass*(1-py)), (update(z, 1), mass*py)])
            levels.append(nxt)
        return levels

    d0, d1 = occupancies(0), occupancies(1)
    gap = V(0, z0, 0)-V(0, z0, 1)
    telescope = sum(mass * (expectation(t,z,0,lambda tt,zz: V(tt,zz,0))
                            - expectation(t,z,1,lambda tt,zz: V(tt,zz,0)))
                    for t in range(horizon) for z,mass in d1[t])

    def Vhat(t,z):
        return terminal(z) if t == horizon else float(sigmoid(.15+t*.03+z[0]*.4))

    def residual0(t,z):
        return expectation(t,z,0,Vhat)-Vhat(t,z)

    response_term = sum(mass*(expectation(t,z,0,Vhat)-expectation(t,z,1,Vhat))
                        for t in range(horizon) for z,mass in d1[t])
    r0 = sum(mass*residual0(t,z) for t in range(horizon) for z,mass in d0[t])
    r1 = sum(mass*residual0(t,z) for t in range(horizon) for z,mass in d1[t])
    joint = response_term + r0-r1
    absolute_bound = sum(mass*abs(expectation(t,z,0,Vhat)-expectation(t,z,1,Vhat))
                         for t in range(horizon) for z,mass in d1[t])
    absolute_bound += sum(mass*abs(residual0(t,z)) for d in (d0,d1)
                          for t in range(horizon) for z,mass in d[t])
    assert abs(gap) <= absolute_bound + 1e-13
    state_a, state_b = (.2,.1,.04,3), (.2,.1,1.,3)
    same_theta_gap = abs(terminal(update(state_a,1))-terminal(update(state_b,1)))
    assert same_theta_gap > 1e-3
    out["nonlinear_full_state_learning"] = {
        "return_gap": float(gap), "telescope_error": check_close(gap,telescope),
        "joint_decomposition_error": check_close(gap,joint),
        "joint_absolute_bound": float(absolute_bound),
        "same_parameter_different_optimizer_response": float(same_theta_gap)}

    # Immediate value cannot reveal a hidden channel that becomes relevant later.
    states = [(s0,b) for s0 in (0,1) for b in (0,1)]
    K = np.zeros((4,4))
    for i,(s0,b) in enumerate(states):
        K[i,states.index((b,b))] = 1
    F = np.array([s0 for s0,b in states], dtype=float)
    psi = np.column_stack([np.ones(4),F])
    coeff = np.linalg.lstsq(psi,K@psi,rcond=None)[0]
    defect = float(np.max(np.abs(K@psi-psi@coeff)))
    immediate_rank = int(np.linalg.matrix_rank(psi))
    closed_rank = int(np.linalg.matrix_rank(np.column_stack([psi,K@F])))
    assert immediate_rank == 2 and closed_rank == 3
    check_close(F[0],F[1]); check_close((K@F)[1]-(K@F)[0],1.)
    out["dynamic_closure_counterexample"] = {
        "immediate_feature_rank": immediate_rank, "closed_test_rank": closed_rank,
        "closure_defect": defect, "future_gap_for_identical_immediate_value": 1.0}

    # Noncontractive finite-horizon control: past reachability and future observation.
    N, split = 5, 2
    A = [np.array([[1.15+.03*t,.12],[.04,.75]]) for t in range(N)]
    B = [np.array([[1.],[(-1)**t*.4]]) for t in range(N)]
    C = [np.array([[1.,.25+.06*t]]) for t in range(N+1)]
    columns = []
    for j in range(split):
        col = B[j]
        for k in range(j+1,split): col = A[k] @ col
        columns.append(col)
    reach = np.hstack(columns)
    rows, transition = [], np.eye(2)
    for t in range(split,N+1):
        rows.append(C[t]@transition)
        if t < N: transition = A[t]@transition
    observe = np.vstack(rows)
    H = observe@reach
    Wc, Wo = reach@reach.T, observe.T@observe
    ev, vec = np.linalg.eigh(Wc)
    root = (vec*np.sqrt(np.maximum(ev,0)))@vec.T
    gram_sv = np.sqrt(np.maximum(np.linalg.eigvalsh(root@Wo@root),0))[::-1]
    hsv = np.linalg.svd(H,compute_uv=False)
    assert max(np.linalg.norm(a,2) for a in A) > 1
    out["finite_horizon_control"] = {
        "hankel_gramian_identity_error": check_close(hsv,gram_sv),
        "hankel_singular_values": hsv.tolist(),
        "max_one_step_gain": float(max(np.linalg.norm(a,2) for a in A))}

    # Exact Gaussian target-information geometry and one-query reduction.
    design = rng.normal(size=(9,4))
    I = design.T@design+.4*np.eye(4)
    target = rng.normal(size=(3,4))
    W = np.diag([.2,.3,.5]); T = target.T@W@target
    inv = np.linalg.inv(I)
    e,v = np.linalg.eigh(I)
    invroot = (v*(1/np.sqrt(e)))@v.T
    lambdas = np.linalg.eigvalsh(invroot@T@invroot)
    risk = float(np.trace(T@inv))
    a = rng.normal(size=4)
    reduction = float((a@inv@T@inv@a)/(1+a@inv@a))
    actual = risk-float(np.trace(T@np.linalg.inv(I+np.outer(a,a))))
    out["gaussian_target_information"] = {
        "risk_spectral_identity_error": check_close(risk,lambdas.sum()),
        "query_reduction_error": check_close(reduction,actual),
        "risk_before": risk, "risk_reduction": reduction}

    # Semidefinite c-optimal design: every design misses a nuisance direction.
    # I(p)=diag(p,4(1-p),0), c=(1,1,0), optimum p=2/3.
    c = np.array([1.,1.,0.])
    design_weight = 2/3
    singular_I = np.diag([design_weight,4*(1-design_weight),0.])
    optimum = float(c@np.linalg.pinv(singular_I)@c)
    direction = np.array([2.,1.,7.])
    dual = float((c@direction)**2/max(direction[0]**2,4*direction[1]**2))
    grid = np.linspace(.001,.999,10001)
    grid_min = float(np.min(1/grid + 1/(4*(1-grid))))
    assert grid_min >= optimum-1e-12
    assert np.linalg.matrix_rank(singular_I) == 2
    out["singular_design_duality"] = {
        "primal_dual_error": check_close(optimum,dual),
        "exact_optimum": optimum, "grid_minimum": grid_min,
        "unidentified_nuisance_dimensions": 1}

    # Uniform guarantee does not follow from an average over rare tasks.
    rare_probability = 1e-6
    averaged = float(np.sqrt(rare_probability))
    assert averaged == .001
    out["rare_task_counterexample"] = {
        "weighted_rms_error": averaged, "rare_task_error": 1.0}
    # No finite rank bound follows from measurability and boundedness alone.
    universal_G = np.eye(8)
    universal_R = universal_G@(np.eye(8)-np.ones((8,8))/8)
    assert np.linalg.matrix_rank(universal_R) == 7
    out["no_universal_compression"] = {"messages": 8, "centered_rank": 7}
    out["status"] = "PASS"
    destination = Path(__file__).with_name("verification_results.json")
    destination.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(out,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
```

## 附录 B：完整运行结果

```json
{
  "seed": 20261013,
  "numpy": "2.3.5",
  "scope": "synthetic mathematical checks only",
  "weighted_spectrum": {
    "spectral_identity_error": 5.204170427930421e-18,
    "best_rank_residual_error": 1.3877787807814457e-17,
    "hilbert_schmidt_squared": 0.03726674417895432,
    "singular_values": [
      0.12012897982964488,
      0.0917570661633554,
      0.07871960367653781,
      0.057943823269333714,
      0.04950365857920686,
      0.042703736286633716,
      0.024247251999200085
    ]
  },
  "positive_law_lower_bound": {
    "feature_match_error": 6.938893903907228e-17,
    "actual_response_gap": 0.07066006517856635,
    "guaranteed_gap": 0.05740498760182184,
    "smallest_probability": 0.005339107084100807
  },
  "estimated_spectrum": {
    "operator_noise": 0.0015551855302188935,
    "entrywise_bound": 0.004,
    "projection_residual": 0.05794774104832839,
    "guaranteed_residual": 0.0659438232693337
  },
  "nonlinear_full_state_learning": {
    "return_gap": 0.07616161014299472,
    "telescope_error": 9.71445146547012e-17,
    "joint_decomposition_error": 2.7755575615628914e-17,
    "joint_absolute_bound": 0.17817790561575503,
    "same_parameter_different_optimizer_response": 0.012822821477190272
  },
  "dynamic_closure_counterexample": {
    "immediate_feature_rank": 2,
    "closed_test_rank": 3,
    "closure_defect": 0.5000000000000002,
    "future_gap_for_identical_immediate_value": 1.0
  },
  "finite_horizon_control": {
    "hankel_gramian_identity_error": 1.096345236817342e-15,
    "hankel_singular_values": [
      4.746231177365882,
      0.03707361279708037
    ],
    "max_one_step_gain": 1.282821179037119
  },
  "gaussian_target_information": {
    "risk_spectral_identity_error": 1.2212453270876722e-15,
    "query_reduction_error": 2.7755575615628914e-17,
    "risk_before": 0.6799661895948446,
    "risk_reduction": 0.14013534793730256
  },
  "singular_design_duality": {
    "primal_dual_error": 0.0,
    "exact_optimum": 2.25,
    "grid_minimum": 2.2500000000045,
    "unidentified_nuisance_dimensions": 1
  },
  "rare_task_counterexample": {
    "weighted_rms_error": 0.001,
    "rare_task_error": 1.0
  },
  "no_universal_compression": {
    "messages": 8,
    "centered_rank": 7
  },
  "status": "PASS"
}
```
