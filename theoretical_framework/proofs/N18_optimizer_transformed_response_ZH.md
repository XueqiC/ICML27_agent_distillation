# N18 · 经优化器变换的线性 response — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N18_optimizer_transformed_response_ZH.md)

## 设定与假设

**状态.** 完整学习器状态为 $x=(\theta,s)\in\mathcal X\subseteq\mathbb R^p\times\mathbb R^q$。其中 $\theta$ 是可训练参数（例如 LoRA adapter），$s$ 是优化器状态（例如 Adam 的矩 $m,v$）。步数计数器并入轮次下标 $t$。

**反馈与梯度.** 第 $t$ 轮，反馈分布 $P_t(\cdot\mid x)$ 给出消息 $Y$。学习器在 loss 归一化和累加之后计算 consumed gradient $g=G_t(x,Y,\xi)\in\mathbb R^p$，$\xi$ 是声明的更新随机性。若有裁剪，裁剪算作优化器的一部分。

**优化器.** 优化器是一个可测的增量映射
$$x^+=x+\Delta_t(x,g).$$

**Kernel.** 分布 $P$ 的学习 kernel 为
$$(K_t^Pf)(x)=\mathbb E_{Y\sim P_t(\cdot\mid x),\,\xi}\,f\big(x+\Delta_t(x,G_t(x,Y,\xi))\big).$$
被比较的接口使用同一个优化器和同一个梯度映射，只在消息分布上不同。换一个优化器就是换一个 $\Delta$，可以用同样的方式比较。

**增量矩.** 对 $(t,x)$ 处的分布 $P$，消息层面的平均增量为 $D_t(x,y)=\mathbb E_\xi\Delta_t(x,G_t(x,y,\xi))$。定义
$$\mu_t^P(x)=\mathbb E_{Y\sim P}D_t(x,Y),\qquad m_{k,t}^P(x)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(x,G_t(x,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(x,G_t(x,Y,\xi))^{\otimes j}.$$

**延续值.** $V^P_{t+1}:\mathcal X\to[0,1]$ 是协议 $P$ 的延续值：之后的更新都按 $P$ 进行，然后部署学生并用 $F_E$ 打分（节点 [N05](N05_continuation_response_ZH.md)）。$L(P)=\mathbb E\,F_E(X_R^P)$，$d_t^Q$ 是协议 $Q$ 下 $X_t$ 的分布。在 $\mathbb R^{p+q}$ 上固定一个范数 $\lVert\cdot\rVert$，其对偶范数为 $\lVert\cdot\rVert_*$。

**假设 S$_k$（可达线段上的光滑延续值）.** 令 $\mathcal S_{t,x}$ 为 $x$ 与所有点 $x+\Delta_t(x,g)$ 的凸包，其中 $g$ 取遍被比较分布下 consumed gradient 的支撑。要求：

- $V=V_{t+1}$ 在 $\mathcal S_{t,x}$ 上是 $C^{k+1}$ 的；
- 对每个 $w\in\mathcal S_{t,x}$ 和每个 $h$，$\lvert\nabla^{k+1}V(w)[h,\dots,h]\rvert\le M_{k+1}\lVert h\rVert^{k+1}$；
- 矩 $m^P_{k+1,t}(x)$ 与 $m^Q_{k+1,t}(x)$ 有限。

**S$_k$ 的适用范围.** 在裁剪或 sign 更新下，多步延续值一般在这样的状态处不光滑：之后某步的梯度跨过裁剪阈值，或之后某步的更新符号发生翻转。此时 S$_k$ 只在避开这些 kink 的区域成立，例如裁剪恒为 active，或动量主导每个符号的区域。这些优化器的平均增量在节点 [N19](N19_learner_statistics_ZH.md) 中精确算出，不需要任何光滑性假设；下面的余项界只在 S$_k$ 成立处才声称成立。

**声明的族.** 对定理 C3，固定一个边界 $(t,x)$ 和一个延续值集合 $\mathcal V$，每个关心的部署任务和后缀对应其中一个；每个都在 $\mathcal S_{t,x}$ 上满足 S$_1$，且共用 Hessian 界 $M_2$。令
$$U=\operatorname{span}\lbrace \nabla V(x):V\in\mathcal V\rbrace ,\qquad \mathcal G^{(1)}_{t,x}=\lbrace\,y\mapsto\langle\nabla V(x),D_t(x,y)\rangle:\ V\in\mathcal V\,\rbrace .$$
$\mathcal G^{(1)}_{t,x}$ 称为一阶 response 类。

## 陈述

**定理 C1（经优化器变换的矩展开）.** 在 S$_k$ 下，
$$(K_t^P-K_t^Q)V(x)=\sum_{j=1}^{k}\frac1{j!}\Big\langle\nabla^jV(x),\;\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\Big\rangle+r_k,\qquad\lvert r_k\rvert\le\frac{M_{k+1}}{(k+1)!}\big(m_{k+1}^P+m_{k+1}^Q\big).$$
特别地，当 $k=1$ 时，
$$(K_t^P-K_t^Q)V(x)=\big\langle\nabla V(x),\mu_t^P(x)-\mu_t^Q(x)\big\rangle+r_1,\qquad\lvert r_1\rvert\le\tfrac{M_2}2\big(m_2^P+m_2^Q\big).$$

**定理 C2（整次训练的一阶表示与 certificate）.** 从同一初始分布出发，在 $R$ 轮上比较协议 $P$ 与 $Q$。假设在 $Q$ 访问到的每个状态上 S$_1$ 成立，延续值 $V^P_{t+1}$ 的常数为 $M_{2,t}(x)$。则
$$L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\Big\langle\nabla V^P_{t+1}(X_t),\;\mu_t^P(X_t)-\mu_t^Q(X_t)\Big\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\sum_t\mathbb E_{d_t^Q}\Big[\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big].$$
因此，记 $\Lambda_t(x)=\lVert\nabla V^P_{t+1}(x)\rVert_*$，
$$\lvert L(P)-L(Q)\rvert\le\min\Big\lbrace 1,\ \sum_t\mathbb E_{d_t^Q}\Big[\Lambda_t\,\lVert\mu_t^P-\mu_t^Q\rVert+\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big]\Big\rbrace .$$
*停止形式.* 设 S$_1$ 只在认证集 $C_t\subseteq\mathcal X$ 上成立。令 $\tau=\min\lbrace t<R:X_t^Q\notin C_t\rbrace$（若不存在这样的 $t$ 则 $\tau=R$），$\rho=\Pr(\tau<R)$。则
$$\lvert L(P)-L(Q)\rvert\le\mathbb E_Q\sum_{t<\tau}\Big[\Lambda_t\,\lVert\mu_t^P-\mu_t^Q\rVert+\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big]\,(X_t)+\rho.$$

**定理 C3（一阶的充分性、必要性与极小性）.** 使用 Euclidean 内积，令 $\Pi_U$ 为到 $U$ 的正交投影。

1. *充分性.* 若 $\Pi_U(\mu^P-\mu^Q)=0$，则对每个 $V\in\mathcal V$，$\lvert(K^P-K^Q)V(x)\rvert\le\tfrac{M_2}2(m_2^P+m_2^Q)$。
2. *必要性.* 若 $\Pi_U(\mu^P-\mu^Q)\ne0$，则存在 $V\in\mathcal V$ 使 $\langle\nabla V(x),\mu^P-\mu^Q\rangle\ne0$。对每个这样的 $V$，
$$\lvert(K^P-K^Q)V(x)\rvert\ge\big\lvert\langle\nabla V(x),\mu^P-\mu^Q\rangle\big\rvert-\tfrac{M_2}2(m_2^P+m_2^Q).$$
   因此，只要一阶项超过余项界，该任务就能检测出这个不一致。
3. *极小性.* 在大小为 $m$ 的有限消息字母表上，能对所有分布对决定每个一阶项 $\langle\nabla V(x),\mu^P-\mu^Q\rangle$（$V\in\mathcal V$）的线性期望统计量，其最小个数为
$$d_0^{(1)}=\dim\operatorname{span}\big(\lbrace 1\rbrace \cup\mathcal G^{(1)}_{t,x}\big)-1\le\min\lbrace \dim U,\ m-1\rbrace .$$

**推论 C1′（经优化器变换的矩层级）.** 在 S$_k$ 下，设两个分布在某状态处对 $j=1,\dots,k$ 都有相同的 $\mathbb E\Delta^{\otimes j}$。则它们在该处的单边界差异至多为 $\tfrac{M_{k+1}}{(k+1)!}(m_{k+1}^P+m_{k+1}^Q)$；像 C2 那样对访问到的状态求和，就得到整次训练差距的界。

- 对 SGD，$\Delta=-\eta G$，所以需要匹配的是梯度矩 $\mathbb E G^{\otimes j}$，每轮的界是 $O(\eta^{k+1})$。
- 对冷启动 Adam，参数块需要匹配的是 $h_\epsilon(G)$ 的矩；当 $\epsilon\to0$ 时就是联合符号矩。
- 在裁剪恒为 active 时，参数块只涉及方向 $G/\lVert G\rVert$ 的矩。

## 证明

**定理 C1 的证明.** 固定一个实现的增量 $\Delta=\Delta_t(x,g)$，$g$ 在支撑中。因为 $\mathcal S_{t,x}$ 是凸的且包含两个端点，线段 $\lbrace x+u\Delta:u\in[0,1]\rbrace$ 落在 $\mathcal S_{t,x}$ 内。在这条线段上，$\varphi(u)=V(x+u\Delta)$ 是 $C^{k+1}$ 的，且 $\varphi^{(j)}(u)=\nabla^jV(x+u\Delta)[\Delta^{\otimes j}]$。带 Lagrange 余项的 Taylor 定理给出某个 $u^*\in(0,1)$，使得
$$V(x+\Delta)=V(x)+\sum_{j=1}^{k}\frac1{j!}\nabla^jV(x)[\Delta^{\otimes j}]+R(\Delta),\qquad R(\Delta)=\frac1{(k+1)!}\nabla^{k+1}V(x+u^*\Delta)[\Delta^{\otimes(k+1)}].$$
由 S$_k$，$\lvert R(\Delta)\rvert\le\tfrac{M_{k+1}}{(k+1)!}\lVert\Delta\rVert^{k+1}$。

把这个恒等式分别对 $Y\sim P$、$\xi$ 和 $Y\sim Q$、$\xi$ 积分。每一项都可积：$V$ 取值于 $[0,1]$；$\lvert R\rvert$ 被期望有限的 $\lVert\Delta\rVert^{k+1}$ 的常数倍控制；每个 $\nabla^jV(x)[\Delta^{\otimes j}]$（$j\le k$）是 $\Delta$ 的多项式，其 $j$ 阶矩由 Lyapunov 不等式有限。两式相减：

- 常数 $V(x)$ 抵消，因为两个分布都是概率测度。
- 映射 $\Delta^{\otimes j}\mapsto\nabla^jV(x)[\Delta^{\otimes j}]$ 是线性的，所以期望可以移进去：$\mathbb E_P\nabla^jV(x)[\Delta^{\otimes j}]=\langle\nabla^jV(x),\mathbb E_P\Delta^{\otimes j}\rangle$。
- 余项给出 $r_k=\mathbb E_PR-\mathbb E_QR$，且 $\lvert r_k\rvert\le\mathbb E_P\lvert R\rvert+\mathbb E_Q\lvert R\rvert\le\tfrac{M_{k+1}}{(k+1)!}(m^P_{k+1}+m^Q_{k+1})$。

当 $k=1$ 时，对 $\xi$ 用塔性质得 $\mathbb E_P\Delta=\mathbb E_{Y\sim P}D_t(x,Y)=\mu_t^P(x)$。$\square$

**定理 C2 的证明.** 节点 [N05](N05_continuation_response_ZH.md) 的定理 5.1，即望远镜恒等式 (14)，对任意两个从同一初始分布出发的协议都成立，与哪个被称为参照无关。以 $P$ 为参照、$Q$ 为备选，
$$L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\big[(K_t^P-K_t^Q)V^P_{t+1}(X_t)\big].$$
在每个访问到的状态 $X_t=x$ 上，$V^P_{t+1}$ 满足 S$_1$。取 $k=1$ 的定理 C1 给出
$$(K_t^P-K_t^Q)V^P_{t+1}(x)=\langle\nabla V^P_{t+1}(x),\mu_t^P(x)-\mu_t^Q(x)\rangle+r_{1,t}(x),\qquad\lvert r_{1,t}(x)\rvert\le\tfrac{M_{2,t}(x)}2\big(m^P_{2,t}(x)+m^Q_{2,t}(x)\big).$$
映射 $x\mapsto\mu_t(x)$、$m_{2,t}(x)$ 和 $\nabla V^P_{t+1}(x)$ 都可测。对 $d_t^Q$ 积分并对 $t$ 求和，就得到表示式，其中 $\mathcal E=\sum_t\mathbb E_{d_t^Q}r_{1,t}$，$\lvert\mathcal E\rvert$ 的界如上。

对 certificate，对每个一阶项用 $\lvert\langle a,b\rangle\rvert\le\lVert a\rVert_*\lVert b\rVert$，对求和用三角不等式。截断到 $1$ 成立，因为 $L(P),L(Q)\in[0,1]$。

对停止形式，在 $C_t$ 上定义 $\varepsilon_t(x)=\Lambda_t(x)\lVert\mu^P_t(x)-\mu^Q_t(x)\rVert+\tfrac{M_{2,t}(x)}2\big(m^P_{2,t}(x)+m^Q_{2,t}(x)\big)$。由 C1 和对偶范数不等式，对 $x\in C_t$ 有 $\lvert(K^P_t-K^Q_t)V^P_{t+1}(x)\rvert\le\varepsilon_t(x)$。以 $P$ 为参照应用节点 N05 的定理 5.3，得 $\lvert L(P)-L(Q)\rvert\le\mathbb E_Q\sum_{t<\tau}\varepsilon_t(X_t)+\rho$。$\square$

**定理 C3 的证明.**

*(1)* 对 $V\in\mathcal V$，$\nabla V(x)\in U$。因此对 $w=\mu^P-\mu^Q$，$\langle\nabla V(x),w\rangle=\langle\nabla V(x),\Pi_Uw\rangle$，由假设它等于 $0$。取 $k=1$ 的 C1 后只剩下 $r_1$。

*(2)* 设对每个 $V\in\mathcal V$ 都有 $\langle\nabla V(x),w\rangle=0$。那么 $w$ 与 $U$ 的一个张成集中的每个元素都正交，于是 $w\perp U$，$\Pi_Uw=0$，矛盾。对满足 $\langle\nabla V(x),w\rangle\ne0$ 的 $V$，C1 给出 $(K^P-K^Q)V(x)=\langle\nabla V(x),w\rangle+r_1$；反向三角不等式和 $\lvert r_1\rvert$ 的界给出所示不等式。

*(3)* 对 $V\in\mathcal V$，一阶项为
$$\langle\nabla V(x),\mu^P-\mu^Q\rangle=\sum_y\big(P(y)-Q(y)\big)\,g_V(y),\qquad g_V(y)=\langle\nabla V(x),D_t(x,y)\rangle\in\mathcal G^{(1)}_{t,x}.$$
因此"由特征 $\phi_1,\dots,\phi_d$ 的均值、对所有分布决定每个一阶项"，正是节点 [N06](N06_sufficiency_duality_ZH.md) 的定理 8(a) 对 response 类 $\mathcal G^{(1)}_{t,x}$ 提出的精确充分性问题。该定理给出最小个数 $\dim\operatorname{span}(\lbrace1\rbrace\cup\mathcal G^{(1)}_{t,x})-1$。

对上界：每个 $g_V$ 都形如 $y\mapsto\langle u,D_t(x,y)\rangle$，$u\in U$。映射 $u\mapsto\langle u,D_t(x,\cdot)\rangle$ 是从 $U$ 到字母表上函数空间的线性映射，像的维数至多为 $\dim U$。因此 $\dim\operatorname{span}(\lbrace1\rbrace\cup\mathcal G^{(1)})\le\dim U+1$。$m$ 点字母表上的全部函数构成包含常数的 $m$ 维空间，所以同一个量也至多为 $m$。减去 $1$ 就得到两个界。$\square$

**推论 C1′ 的证明.** 假设使 C1 中 $j\le k$ 的每一项 $\langle\nabla^jV(x),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle$ 都为零，只剩下 $r_k$。

- 对 SGD，$\Delta^{\otimes j}=(-\eta)^jG^{\otimes j}$，所以匹配增量矩就是匹配梯度矩，且 $m_{k+1}=\eta^{k+1}\mathbb E\lVert G\rVert^{k+1}$。
- 对冷启动 Adam 和裁剪恒为 active 的情形，$\Delta$ 的参数块分别是 $-\eta h_\epsilon(G)$ 和 $-\eta c\,G/\lVert G\rVert$，其张量幂就是所述的矩。$\square$

## 注记

1. **对象.** 一阶意义下，对给定学习器和任务族，反馈分布必须保持的是完整状态（包括 $m$ 和 $v$）上平均*优化器增量*的 $U$-投影。它不是平均梯度，不是 loss，也不是 teacher 的准确率。梯度分布的哪些特征会进入，完全由优化器映射 $\Delta$ 决定。节点 [N19](N19_learner_statistics_ZH.md) 对每种常见学习器算出这个统计量。

2. **与前面节点的关系.** 对 SGD，取 $k=2$ 的 C1 就是节点 [N10](N10_update_composition_ZH.md) 的式 (35)。C3 是把节点 [N06](N06_sufficiency_duality_ZH.md) 的定理 8 用到线性化的 response 类上。C2 是在节点 [N05](N05_continuation_response_ZH.md) 的望远镜恒等式的每一轮插入一个局部展开。节点 [N12](N12_optimizer_warmth_ZH.md) 的命题 W1 是 C1 在 warm Adam 下的情形，只是把 $v$ 块写成残差而不是线性项。

3. **余项在什么意义下小（数量级上的启发式；界本身是带 $M_2$ 的 C2）.** 把增量分块：
   - *参数块.* $\Delta^\theta=O(\eta)$，所以其二阶贡献 $\tfrac12\nabla^2_{\theta\theta}V[\Delta^\theta,\Delta^\theta]$ 是 $O(\eta^2)$。
   - *一阶矩块（动量 $m$）.* $\Delta^m=O(1-\beta_1)$。延续值通过之后的参数增量依赖于 $m$，主阶是线性的，系数为 $O(\eta/(1-\beta_1))$。所以 $\nabla^2_{mm}V=O(\eta^2/(1-\beta_1)^2)$，这一块的贡献也是 $O(\eta^2)$。
   - *二阶矩块（Adam 的 $v$）.* $\Delta^v=(1-\beta_2)(G^{\odot2}-v)$。延续值通过之后各步的分母 $(\hat v)^{-1/2}$ 依赖于 $v$，这些分母只带一个 $\eta$ 因子，没有更高次幂。因此这一块的二阶贡献是 **$\eta$ 的一阶量**。它相对于本块一阶项的大小由 $\Delta v_i/v_i^{\rm next}$ 决定，其中 $v^{\rm next}$ 是下一步使用的二阶矩。在 warm 状态下这个比值约等于节点 [N12](N12_optimizer_warmth_ZH.md) 的 warmth $\kappa_i$；在冷状态下它是 1 的量级。

   因此，在 S$_1$ 成立处，对 SGD、动量和 sign 类优化器，C2 的余项每轮是 $O(\eta^2)$；对 Adam 类优化器，多于一轮时余项是 $\eta$ 的一阶量，其绝对大小随 warm-up 减小。当 $R=1$ 时，边界之后的延续值是 $F_E(\theta)$，与 $s$ 无关，此时对每种优化器余项都是 $O(\eta^2)$。

4. **实践中的必要性.** C3(2) 只有在一阶项超过余项界时才能检测出不一致。余项为 $O(\eta^2)$ 时，任何固定的不一致在 $\eta$ 足够小时都会被检测出。对 Adam 类的多轮 response，余项也是 $\eta$ 的一阶量，因此检测需要节点 [N20](N20_first_order_audit_ZH.md) 的 margin 条件（定理 C6）。

5. **统计量的代价.** $\mu_t^P(x)$ 只需要在状态 $x$ 处计算 consumed gradient，不需要训练尾段。常数 $\Lambda_t$ 和 $M_{2,t}$ 是延续值的性质，无法通过对消息采样来估计。节点 [N20](N20_first_order_audit_ZH.md) 给出估计量，并说明还有哪些量需要校准。

## 依赖与出处

- **依赖：** [N05](N05_continuation_response_ZH.md)（望远镜恒等式 (14) 与停止复合 (17)）、[N06](N06_sufficiency_duality_ZH.md)（定理 8(a)）、[N10](N10_update_composition_ZH.md)（SGD 展开 (35)）、[N11](N11_adam_cold_start_ZH.md) 与 [N12](N12_optimizer_warmth_ZH.md)（Adam 映射与 warmth）。
- **被用于：** [N19](N19_learner_statistics_ZH.md)（统计量清单）、[N20](N20_first_order_audit_ZH.md)（估计与 margin）、[N17](N17_ced_method_ZH.md)（方法）。
- **出处：** `docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md`，§1–§3 与 §6（定理 C1–C3、缩放段落与矩层级）。
