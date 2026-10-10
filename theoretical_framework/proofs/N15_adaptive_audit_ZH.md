# N15 · 实际状态上的自适应 audit；保真与改进 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N15_adaptive_audit_ZH.md)

## 设定与假设

**学习过程.** 训练过程同 N05：

- 完整状态 $Z_t$，有限时域 $R$；
- 固定的参照配方，kernel 为 $K^0_t$，continuation 为 $V^0_t\in[0,1]$，且 $V^0_R=F_E$；
- $J_E(\pi_0)$ 是初始学生的价值；
- $L_E(0)=\mathbb E F_E(Z^0_R)$ 是参照训练的价值。

**局部优势.** 对候选 kernel $K^j_t$，定义
$$A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].\tag{42}$$
**provider 策略** $S$ 在每一轮根据历史选择一个 kernel 下标 $J_t$。

**自适应 audit（Thm. 20）.**

- 记 $\mathcal H_t$ 为第 $t$ 轮 audit 之前的历史。在 $\mathcal H_t$ 条件下，冻结实际状态 $Z_t$、由 $M_t$ 个非参照候选 kernel 组成的有限集合、副本数 $n_t$ 和置信分配 $\delta_t$，并几乎处处满足 $\sum_t\delta_t\le\delta$。
- 对每个候选，从 $Z_t$ 出发做 $n_t$ 次独立的配对实验。每次实验执行候选更新或参照更新，然后执行**完整的**参照学习 suffix 和真实部署。每一臂都必须有正确的边际律，配对对比落在 $[-1,1]$ 中。
- 记 $\widehat A_t(j)$ 为平均对比，令
$$e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},\qquad\ell_t(j)=\widehat A_t(j)-e_t,\qquad\ell_t(0)=0.$$
- 选择 $J_t\in\arg\max_j\ell_t(j)$，令 $B_t=\ell_t(J_t)\in[0,1]$。然后从 $K^{J_t}_t$ 做一次**新鲜**更新，丢弃 audit 中训练出的副本。
- 未做 audit 的轮次使用参照，并令 $B_t=0$。
- audit 不改变实际的学习者状态和环境状态，参照也不读取 audit 记录。
- $\mathcal F_t\supseteq\mathcal H_t$ 是 audit 与选择之后、新鲜更新之前的历史。

## 陈述

**定理（N15）.**

1. **保真.** 若 $L_E(0)-J_E(\pi_0)\ge\gamma$ 且 $L_E(0)-L_E(S)\le\varepsilon$，则
$$L_E(S)-J_E(\pi_0)\ge\gamma-\varepsilon.\tag{41}$$
   取 $\varepsilon=\varepsilon_K$ 为 N14 的 certificate (39a)，$\gamma=\gamma_K$ 为基于 $K$ 条示范的参照训练的实测增益，即得 (A17)。
2. **带号恒等式.** 对每个具有公共初始律、依赖历史的 provider 策略 $S$，
$$L_E(S)-L_E(0)=\sum_{t=0}^{R-1}\mathbb E_{d^S_t}\,A_t(Z_t,J_t).\tag{43}$$
3. **自适应 audit.**
   - (a) 以至少 $1-\delta$ 的概率，$\sum_tA_t(Z_t,J_t)\ge\sum_tB_t$。
   - (b) 对 audit 与训练无条件地取期望，
$$L_E(\mathrm{adaptive})-L_E(0)\ge-\delta,\tag{44}$$
$$L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_tB_t-2\delta.\tag{45}$$
4. **代价.** 要使半径 $e_t\le e$，audit 需要 $n_t\ge\lceil2\log(2M_t/\delta_t)/e^2\rceil$。$M=4$、$\delta=0.05$、$e=0.1$ 时，每个候选需要 $\lceil 2\log160/0.01\rceil=1016$ 个配对副本。这个数量是充分的，但不是必要的。

## 证明

**第 1 部分.** 两个假设相减：$L_E(S)-J_E(\pi_0)=[L_E(0)-J_E(\pi_0)]-[L_E(0)-L_E(S)]\ge\gamma-\varepsilon$。这里只用了单侧的保真界，所以允许候选优于参照。

**第 2 部分.** 沿 $S$ 过程，对历史用塔性质，
$$\mathbb E[V^0_{t+1}(Z_{t+1})\mid\text{history to }t]=(K^{J_t}_tV^0_{t+1})(Z_t).$$
于是
$$\mathbb E\big[V^0_{t+1}(Z_{t+1})-V^0_t(Z_t)\big]=\mathbb E\big[(K^{J_t}_t-K^0_t)V^0_{t+1}(Z_t)\big]=\mathbb E\,A_t(Z_t,J_t).$$
对 $t$ 求和即望远镜相消。公共初始律给出 $\mathbb E V^0_0(Z_0)=L_E(0)$，$V^0_R=F_E$ 给出 $\mathbb E V^0_R(Z_R)=L_E(S)$。

**第 3 部分.**

*audit 事件.* 在 $\mathcal H_t$ 条件下，候选 $j$ 的 $n_t$ 个对比在 $[-1,1]$ 中独立同分布，均值为 $A_t(Z_t,j)$。这一点用到正确的边际、完整 suffix 和精确恢复。对取值范围为 $2$ 的变量用 Hoeffding 不等式，
$$\Pr\big(\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert>e_t\mid\mathcal H_t\big)\le2\exp(-n_te_t^2/2)=\delta_t/M_t.$$
对 $M_t$ 个候选做 union bound，得到事件 $G_t$，满足 $\Pr(G_t^c\mid\mathcal H_t)\le\delta_t$；在 $G_t$ 上，对所有 $j$ 有 $\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert\le e_t$。这个半径要求 $n_t$ 在观察到本次 audit 的结果之前就已确定。

*被选中的漂移.* 记 $A_t:=A_t(Z_t,J_t)$。若 $J_t=0$，则 $A_t=0=B_t$。否则 $B_t=\widehat A_t(J_t)-e_t\ge0$，且在 $G_t$ 上 $A_t\ge\widehat A_t(J_t)-e_t=B_t$。所以在 $G_t$ 上 $A_t\ge B_t\ge0$。未做 audit 的轮次有 $A_t=B_t=0$。

*结论 (a).* 依次对 $\mathcal H_t$ 取条件，
$$\Pr\Big(\bigcup_tG_t^c\Big)\le\mathbb E\sum_t\Pr(G_t^c\mid\mathcal H_t)\le\mathbb E\sum_t\delta_t\le\delta.$$
在 $\bigcap_tG_t$ 上有 $\sum_tA_t\ge\sum_tB_t$。由于每个 audit 陈述都以它自己的 audit 前历史为条件，这对实际到达的随机状态也成立。

*漂移恒等式.* 实际更新是从 $K^{J_t}_t$ 新鲜抽取的，在 $\mathcal F_t$ 条件下与 audit 副本独立，而且 audit 不改变状态，所以
$$A_t=\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]-V^0_t(Z_t).$$
令 $D_{t+1}=V^0_{t+1}(Z_{t+1})-\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]$。它们是交错的 audit/更新滤子下的鞅差，因此 $\mathbb E\sum_tD_{t+1}=0$。望远镜求和得
$$F_E(Z_R)-V^0_0(Z_0)=\sum_{t=0}^{R-1}A_t+\sum_{t=0}^{R-1}D_{t+1}.$$

*结论 (b).* 总有 $A_t\in[-1,1]$、$B_t\in[0,1]$。因此
$$A_t\ge-\mathbf 1_{G_t^c},\qquad A_t\ge B_t-2\cdot\mathbf 1_{G_t^c}.$$
第一式成立，是因为在 $G_t$ 上 $A_t\ge0$；第二式成立，是因为在 $G_t^c$ 上 $A_t\ge-1\ge B_t-2$。对望远镜恒等式取期望，并用 $\mathbb E V^0_0(Z_0)=L_E(0)$ 和 $\mathbb E\sum D=0$：
$$L_E(\mathrm{adaptive})-L_E(0)=\mathbb E\sum_tA_t\ge-\mathbb E\sum_t\mathbf 1_{G_t^c}=-\mathbb E\sum_t\Pr(G_t^c\mid\mathcal H_t)\ge-\delta,$$
同理 $\mathbb E\sum_tA_t\ge\mathbb E\sum_tB_t-2\delta$。

**第 4 部分.** $e_t\le e$ 等价于 $n_t\ge2\log(2M_t/\delta_t)/e^2$。$M=4$、$\delta=0.05$ 时 $2\log160=10.1503$，除以 $e^2=0.01$ 再向上取整得 $1016$。Hoeffding 不等式只是一种充分的选择，任何半径更小的有效固定 $n$ 置信界都可以替代它。$\square$

## 注记

1. **下置信界是必需的.** 即使 audit 用的是精确的完整尾部，按点估计 $\widehat A_t(j)$ 选择也可能有严格为负的期望增益；一个精确的有限见证给出增益 $-78/15625$（见实验页 C16）。即便用下置信界把关，一个略差的候选也有正概率通过。因此 (44) 中的 $-\delta$ 不能去掉：一般而言，精确的"不退化"（$\ge0$）陈述是错的。
2. **无条件，而非有条件.** (44)–(45) 对 audit 结果取平均。一般来说，以"所有 audit 都成功"为条件来看终端价值是不合法的，因为后续的 audit 事件依赖于到达的状态。一个零漂移见证给出条件价值 $119/494$，而无条件价值是 $1/4$（见实验页 C16）。
3. **停止形式的细化.** 令 $\tau$ 为第一次 audit 失败的轮次，截断在 $R$。则 $L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_{t<\tau}B_t-\delta$。
4. **去掉了什么、没去掉什么.** 在实际状态上 audit，去掉了 N14 中的 transport/覆盖项，但没有去掉恢复、完整 suffix 的代价和独立性要求。把 audit 中胜出的副本直接当作更新，或者只用短尾部做 audit，都会改变被估计的量；后者需要显式的 proxy 偏差余量。
5. **保真不等于改进.** (41) 只继承参照的实测增益 $\gamma$。一个无效参照的高保真候选，并不会改进初始学生（见实验页 C21）。

## 依赖与来源

- 依赖 N05（望远镜恒等式）和 N14（(A17) 中的 certificate $\varepsilon_K$）。
- 来源：`docs/direction/v015_revision/Environment_Information_Theory_EN.md` §11，式 (41)–(45)，Thm. 20；`docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md` §5（A17）、§7（代价）。复核：`docs/direction/v014_handoff/Independent_Proof_Audit_EN.md` §C3、§A26。
