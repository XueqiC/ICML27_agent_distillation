# N14 · 带 transport 的冻结有限接口选择 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N14_frozen_audit_transport_ZH.md)

## 设定与假设

**学习过程（同 N05）.**

- 轮次为 $t=0,\dots,R-1$。$z$ 是完整的学习者状态，包括参数、优化器矩与步数计数、缓冲区、采样器状态和历史。
- 在第 $t$ 轮状态 $z$ 处，参照 provider 发出 $Y\sim P^0_{t,z}$，候选 $j\in\lbrace 1,\dots,M\rbrace $ 发出 $Y\sim P^j_{t,z}$。
- 实际更新为 $U_t(z,Y,\xi)$，参照 continuation 为 $V^0_{t+1}$，且 $V^0_R=F_E\in[0,1]$。
- response 为 $g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]$。
- 对消息积分，得到 kernel 恒等式
$$(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace 0,1,\dots,M\rbrace .\tag{K}$$
- 对由 provider $j$ 驱动的过程，记 $L_E(j)=\mathbb E F_E(Z_R^j)$。所有过程从相同的 $Z_0$ 初始律出发。

**冻结的 audit.**

- 只用独立的 pilot 数据，在 audit 之前冻结以下内容：
  - $M$ 个可执行的候选 provider；
  - $L$ 个查询 cell $x$，每个 cell 指定完整的学习者状态、任务、查询、更新和参照 suffix；
  - 每个 cell 上的 $d$ 个特征 $\phi_{x,\ell}:\mathcal Y\to[0,1]$。
- $P^0_x$ 和 $P^j_x$ 分别是 cell $x$ 中参照与候选 $j$ 的消息律。
- audit 在每个 cell 抽取 $n_0$ 条独立的参照消息，在每个（候选，cell）上抽取 $n_S$ 条独立消息。
- 记 $\mu_{j,x,\ell}=\mathbb E_{P^j_x}\phi_{x,\ell}$，其经验均值为 $\widehat\mu_{j,x,\ell}$。定义
$$D_j=\max_{x,\ell}\lvert\mu_{0,x,\ell}-\mu_{j,x,\ell}\rvert,\qquad\widehat D_j=\max_{x,\ell}\lvert\widehat\mu_{0,x,\ell}-\widehat\mu_{j,x,\ell}\rvert,$$
$$e_0=\sqrt{\frac{\log(4Ld/\delta)}{2n_0}},\qquad e_S=\sqrt{\frac{\log(4MLd/\delta)}{2n_S}}.$$
- 被选中的候选 $\widehat j$ 在已知优化误差 $\eta_{\rm opt}$ 之内最小化 $\widehat D_j$。

**被认证的状态.** 每一轮 $t$ 有一个由完整状态组成的认证集 $\mathcal C_t$。每个 $z\in\mathcal C_t$ 附带：

1. 一个被 audit 的 cell $x=x(t,z)$，其特征用 $z$ 处的消息坐标表示；
2. 一个 **transport 界**：对 audit 事件上每个可能被选中的候选，
$$\Big\lVert(P^0_{t,z}-P^j_{t,z})\phi_x-(P^0_x-P^j_x)\phi_x\Big\rVert_\infty\le a_{t,j}(z);\tag{A2}$$
3. 一个 **response 分解**
$$g_{t,z}=c_{t,z}+b_{t,z}^\top\phi_x+r_{t,z},\qquad\lVert b_{t,z}\rVert_1\le B_t,\quad\lVert r_{t,z}\rVert_\infty\le\tau_t.\tag{A3}$$

令 $\sigma=\inf\lbrace t<R:Z_t\notin\mathcal C_t\rbrace $，若不存在这样的 $t$ 则 $\sigma=\infty$。$\sigma$ 是训练过程的停时。

**新鲜训练.** 在冻结的 audit 结果条件下，被选中的 provider 用独立于 audit 样本的随机性来训练，并且 $\Pr_{\widehat j}(\sigma<R\mid\text{audit result})\le\rho$。

## 陈述

**定理（N14）.**

1. **Audit 集中性.** 以至少 $1-\delta$ 的概率，对所有 $j$ 同时有
$$\lvert D_j-\widehat D_j\rvert\le e_0+e_S,\tag{37}$$
由此
$$D_{\widehat j}\le\min_jD_j+2(e_0+e_S)+\eta_{\rm opt}.\tag{38}$$
2. **局部 transport 界.** 对 $z\in\mathcal C_t$ 与 $x=x(t,z)$，
$$\big\lvert(K^0_t-K^j_t)V^0_{t+1}(z)\big\rvert\le\min\Big\lbrace 1,\ B_t\big(D_j+a_{t,j}(z)\big)+2\tau_t\Big\rbrace .\tag{V}$$
3. **学习价值界.** 在 (37) 的事件上，
$$\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_{t=0}^{R-1}\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace \Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace \Big]+\rho.\tag{39a}$$
4. **特殊情形.**
   - transport 精确（$a\equiv0$）时，$\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_tB_t(\widehat D_{\widehat j}+e_0+e_S)+2\sum_t\tau_t+\rho$（式 (39)）。
   - 在 Thm. 14（N08）的状态无关 IID 块来源和凸仿射更新假设下，若被 audit 的特征就是矩 $Z^\alpha$（$1\le\lvert\alpha\rvert\le N$），则 $\lvert L_E(0)-L_E(\widehat j)\rvert\le C_J(\widehat D_{\widehat j}+e_0+e_S)$（式 (40)）。
5. **特征漂移.** 若状态相关特征 $\phi_z$ 满足 $\sup_y\lVert\phi_z(y)-\phi_x(y)\rVert_\infty\le\alpha_{zx}$，则
$$\lVert(P^0_{t,z}-P^j_{t,z})\phi_z\rVert_\infty\le D_j+a_{t,j}(z)+2\alpha_{zx},\tag{A5}$$
   并且在 (A3) 中改用 $\phi_z$、把 $a$ 换成 $a+2\alpha_{zx}$ 后，(V) 与 (39a) 仍然成立。
6. **transport 的必要性.** 对任意 audit 规模，都存在一个实例：$D_j=0$，response 分解精确（$B=1$，$\tau=0$），而学习差距为 $1$。在该实例中，不含 transport 的表达式 $B(\widehat D+e_0+e_S)$ 严格小于差距；(39a) 在那里依然成立，而且是紧的。

所有界都可以截断在 $1$。

## 证明

**第 1 部分.** 每个 $\phi_{x,\ell}\in[0,1]$，且每个经验均值背后的样本相互独立。由双侧 Hoeffding，
$$\Pr(\lvert\widehat\mu_{0,x,\ell}-\mu_{0,x,\ell}\rvert>e_0)\le2e^{-2n_0e_0^2}=\frac{\delta}{2Ld},$$
$$\Pr(\lvert\widehat\mu_{j,x,\ell}-\mu_{j,x,\ell}\rvert>e_S)\le2e^{-2n_Se_S^2}=\frac{\delta}{2MLd}.$$
对 $Ld$ 个参照均值和 $MLd$ 个候选均值做 union bound，得到事件 $G$，满足 $\Pr(G^c)\le\delta/2+\delta/2=\delta$。同一条消息上算出的不同特征之间的相关性，不影响 union bound。在 $G$ 上，对每个 $j,x,\ell$，
$$\big\lvert(\widehat\mu_{0,x,\ell}-\widehat\mu_{j,x,\ell})-(\mu_{0,x,\ell}-\mu_{j,x,\ell})\big\rvert\le e_0+e_S.$$
映射 $u\mapsto\max_k\lvert u_k\rvert$ 关于 sup 范数是 $1$-Lipschitz 的，所以对所有 $j$ 同时有 $\lvert\widehat D_j-D_j\rvert\le e_0+e_S$，即 (37)。

关于 (38)，取 $j^\star\in\arg\min_jD_j$。在 $G$ 上，
$$D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S\le\widehat D_{j^\star}+\eta_{\rm opt}+e_0+e_S\le D_{j^\star}+2(e_0+e_S)+\eta_{\rm opt}.$$
因为 $G$ 对冻结的候选集是同时成立的，所以用 audit 数据做选择是合法的。

**第 2 部分.** 由 (K)，$(K^0_t-K^j_t)V^0_{t+1}(z)=(P^0_{t,z}-P^j_{t,z})g_{t,z}$。把 (A3) 逐项代入：

- 常数项 $c_{t,z}$ 对概率律之差积分为零。
- 线性项：记 $\Delta=(P^0_{t,z}-P^j_{t,z})\phi_x\in\mathbb R^d$，则 $\lvert b^\top\Delta\rvert\le\lVert b\rVert_1\lVert\Delta\rVert_\infty$。由 (A2) 和三角不等式，$\lVert\Delta\rVert_\infty\le\lVert(P^0_x-P^j_x)\phi_x\rVert_\infty+a_{t,j}(z)\le D_j+a_{t,j}(z)$。最后一步用到 $\lVert(P^0_x-P^j_x)\phi_x\rVert_\infty=\max_\ell\lvert\mu_{0,x,\ell}-\mu_{j,x,\ell}\rvert\le D_j$。
- 残差项：$\lvert(P^0_{t,z}-P^j_{t,z})r\rvert\le\lvert P^0_{t,z}r\rvert+\lvert P^j_{t,z}r\rvert\le2\tau_t$。

由于 $g\in[0,1]$，(V) 的左边不超过 $1$。

**第 3 部分（停止后的传播）.** 在 $G$ 上、以一个固定的 audit 结果为条件来讨论，于是 $\widehat j$ 是固定的。由新鲜训练假设，训练过程的条件律就是由 $\widehat j$ 驱动的过程的律。

*第 3a 步（切换过程）.* 令 $S'$ 在第 $t$ 轮当 $t<\sigma$ 时使用 kernel $K^{\widehat j}_t$，否则使用 $K^0_t$。由于 $\lbrace \sigma\le t\rbrace $ 由 $Z_0,\dots,Z_t$ 决定，$S'$ 是良定义的、依赖历史的过程。如果需要马尔可夫描述，可以把标志 $1\lbrace t<\sigma\rbrace $ 并入状态；$V^0$ 与这个标志无关。

用同一组随机性把 $S'$ 与 $\widehat j$ 过程耦合。两条路径在 $Z_\sigma$ 之前（含 $Z_\sigma$）完全相同；在 $\lbrace \sigma\ge R\rbrace $ 上，它们一直相同到 $Z_R$。终端值在 $[0,1]$ 中，所以
$$\lvert L_E(\widehat j)-L_E(S')\rvert\le\Pr_{\widehat j}(\sigma<R)\le\rho.$$

*第 3b 步（望远镜求和）.* 对任何 kernel 为 $K^{S'}_t$ 的依赖历史的过程，由塔性质，
$$\mathbb E\big[V^0_{t+1}(Z_{t+1})\mid Z_0,\dots,Z_t\big]=(K^{S'}_tV^0_{t+1})(Z_t),$$
因此
$$\mathbb E_{S'}\big[V^0_t(Z_t)-V^0_{t+1}(Z_{t+1})\big]=\mathbb E_{S'}\big[(K^0_t-K^{S'}_t)V^0_{t+1}(Z_t)\big].$$
对 $t$ 求和，利用公共初始律和 $V^0_R=F_E$，得到
$$L_E(0)-L_E(S')=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[(K^0_t-K^{S'}_t)V^0_{t+1}(Z_t)\big]=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[1\lbrace t<\sigma\rbrace (K^0_t-K^{\widehat j}_t)V^0_{t+1}(Z_t)\big].$$
第二个等号成立，是因为 $t\ge\sigma$ 时两个 kernel 相同。

*第 3c 步（退出前的占据）.* 在 $\sigma$ 之前，$S'$ 与 $\widehat j$ 过程使用相同的 kernel。所以限制在 $\lbrace t<\sigma\rbrace $ 上，$(Z_0,\dots,Z_t)$ 在两者下同律；对有界的 $h$，有 $\mathbb E_{S'}[1\lbrace t<\sigma\rbrace h(Z_t)]=\mathbb E_{\widehat j}[1\lbrace t<\sigma\rbrace h(Z_t)]$。在 $\lbrace t<\sigma\rbrace $ 上 $Z_t\in\mathcal C_t$，所以 (V) 适用。在 $G$ 上，(37) 给出 $D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S$。因此
$$\lvert L_E(0)-L_E(S')\rvert\le\sum_t\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace \Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace \Big].$$

*第 3d 步.* 结合第 3a 步用三角不等式，即得 (39a)。这个界对 $G$ 中的每个 audit 结果都成立，所以要求 (A2)、(A3) 以及 $\rho$ 的界对 $G$ 上每个可能被选中的候选都成立。

**第 4 部分.**

- *(39).* 当 $a\equiv0$ 时，因为 $1\lbrace t<\sigma\rbrace \le1$ 且括号内非负，(39a) 的每个求和项不超过 $B_t(\widehat D_{\widehat j}+e_0+e_S)+2\tau_t$。
- *(40).* 在 IID 仿射假设下，provider $j$ 的每一轮训练都从一个固定、与状态无关的律 $P^j$ 中新抽一个块。取单个 cell（$L=1$），其特征为单项式 $Z^\alpha$（$1\le\lvert\alpha\rvert\le N$），它们都在 $[0,1]$ 中，于是 $D_j=\delta_N(P^0,P^j)$。Thm. 14（N08）给出 $\lvert L_E(0)-L_E(j)\rvert\le C_J\delta_N(P^0,P^j)$；在 $G$ 上，$\delta_N(P^0,P^{\widehat j})\le\widehat D_{\widehat j}+e_0+e_S$。这里不需要 transport，因为被 audit 的律就是每个状态上实际使用的律。训练块必须从这个被 audit 的律中新鲜抽取。

**第 5 部分.** 写 $\phi_z=\phi_x+(\phi_z-\phi_x)$。如第 2 部分，第一部分的贡献不超过 $D_j+a_{t,j}(z)$。对每个坐标 $\ell$，
$$\lvert(P^0_{t,z}-P^j_{t,z})(\phi_z-\phi_x)_\ell\rvert\le\lvert P^0_{t,z}(\phi_z-\phi_x)_\ell\rvert+\lvert P^j_{t,z}(\phi_z-\phi_x)_\ell\rvert\le2\alpha_{zx}.$$
在 (A3) 中使用 $\phi_z$ 的分解并重复第 2、3 部分，即证得所述变体。只在抽样到的消息上成立的界，不是关于 $\sup_y$ 的界。

**第 6 部分（必要性）.** 令 $\mathcal Y=\lbrace 0,1\rbrace $，$\phi(y)=y$，$R=1$。让更新把最终策略参数设为 $p=y$，并令 $F_E(p)=p$。于是在每个状态上 $g_{0,z}(y)=y$，(A3) 以 $c=0$、$b=1$、$r=0$ 成立，即 $B=1$，$\tau=0$。

- 在被 audit 的状态 $z_a$ 上，两个 provider 都发出 $0$，所以 $D_j=0$。
- 训练从 $z_b$ 开始，在那里参照发出 $0$，候选发出 $1$。于是 $\lvert L_E(0)-L_E(j)\rvert=\lvert0-1\rvert=1$。
- 若宣称 $z_b$ 被认证、对应 cell $x(z_b)=z_a$，却不计 transport 项，得到的表达式是 $B(\widehat D+e_0+e_S)=e_0+e_S$。例如 $n_0=n_S=10^4$、$\delta=0.05$、$L=M=d=1$ 时，它等于 $2\sqrt{\log 80/20000}=0.0296$，小于 $1$。
- 在 (39a) 中，认证 $z_b$ 要求 $a_{0,j}(z_b)\ge\lvert(0-1)-(0-0)\rvert=1$，界至少为 $1$。或者 $z_b\notin\mathcal C_0$，此时 $\sigma=0$，$\rho=1$，界为 $1$。

两种情况下，截断后的 (39a) 都等于差距。因此 transport 项不能去掉，也不能按惯例记为零。

把同一构造改成"一个被覆盖的状态以概率 $r$ 到达一个未被覆盖的状态"，还能看出：每个被访问的退出前状态都必须贡献自己的项，或者把对未覆盖状态的首次访问计入 $\rho\ge r$。否则界会差 $r$。$\square$

## 注记

1. **完整状态.** cell、transport 和认证都针对完整的学习者状态。参数相同但 Adam 矩不同的两个状态，会产生不同的下一状态（N11），response 也可能不同。仅凭参数接近，不能推出 (A2)。
2. **精确 transport 比律相等更弱.** $a=0$ 只要求两个特征差异向量一致，并不要求四个律分别相等。
3. **audit 认证的是什么.** (39a) 界定的是被选中、冻结的流程在新鲜训练下的总体期望值，而不是某一个训练 seed 的得分。如果用 audit 样本构造新的 provider，或者把它们拿来当训练数据，实验本身就变了，需要一次新的独立 audit。
4. **额外的置信失败.** 如果 transport、系数或残差是统计估计的，它们的置信失败概率要加到 $\delta$ 上。

## 依赖与来源

- 依赖 N05（望远镜恒等式及其停止形式）和 N13（按系数加权的特征差异）；(40) 依赖 N08（Thm. 14）。
- 来源：`docs/direction/v015_revision/Environment_Information_Theory_EN.md` §10，Thm. 19，式 (37)–(40)、(39a)；`docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md` §2，Prop. A1，式 (A2)–(A5)。复核：`docs/direction/v015_revision/Proof_Review_v015_EN.md` §V15.1，以及 `docs/direction/v014_handoff/Independent_Proof_Audit_EN.md` §A27。
