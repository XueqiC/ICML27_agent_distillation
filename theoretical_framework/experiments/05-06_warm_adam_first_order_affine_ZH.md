# Proposition 5.6 · warm Adam 关于 consumed gradient 一阶仿射

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-06_warm_adam_first_order_affine_ZH.md) · [证明](../proofs/05-06_warm_adam_first_order_affine_ZH.md) · [实验](../experiments/05-06_warm_adam_first_order_affine_ZH.md) · [English](05-06_warm_adam_first_order_affine.md)

**本页检验什么.** [Check 5.9](../experiments/05-06_warm_adam_first_order_affine_ZH.md#c5-9) 在 2,496 对分布上检验这个界，常数事先固定。[Check 5.13](../experiments/05-08_evolution_warmth_ZH.md#c5-13) 的完整描述在 Proposition 5.8 的页面上，它追踪矩相同的来源之间的差距在整个训练过程中的变化。

**结论.**

- **Proposition 5.6（warm Adam 一阶仿射）.** 常数事先固定，界在全部 2,496 对分布上成立，差距/界的最大值为 0.988。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

- **状态.** 完整 learner state 为 $`z=(\theta,o)\in\mathcal Z\subseteq\mathbb R^p\times\mathbb R^q`$。$`\theta`$ 是可训练参数（例如 LoRA adapter），$`o`$ 是 optimizer state（例如 Adam 的矩 $`m,v`$ 与步数计数器）。需要用到计数器时显式写出。
- **反馈与梯度.** 第 $`t`$ 轮，feedback law $`P_t(\cdot\mid z)`$ 给出消息 $`Y`$。学习器在 loss 归一化与累加之后计算 consumed gradient $`g=G_t(z,Y,\xi)\in\mathbb R^p`$，$`\xi`$ 是声明的更新随机性。若有裁剪，裁剪算作 optimizer 的一部分。
- **Optimizer.** optimizer 是一个可测的增量映射 $`z^+=z+\Delta_t(z,g)`$。被比较的 feedback law 共用同一个 optimizer 与同一个梯度映射，只在消息分布上不同。换一个 optimizer 就是换一个 $`\Delta`$，用同样的方式比较。
- **本节局部复用的字母.** 本节中 $`s`$ 表示 Adam 二阶矩已吸收的梯度计算次数（Proposition 5.8），与第 6 节的 agent 状态 $`s`$ 无关。裁剪阈值记为 $`c`$，weight decay 系数记为 $`\lambda_{\rm wd}`$。

分布 $`P`$ 的学习 kernel 为

```math
(K_t^Pf)(z)=\mathbb E_{Y\sim P_t(\cdot\mid z),\,\xi}\,f\big(z+\Delta_t(z,G_t(z,Y,\xi))\big).
```

对 $`(t,z)`$ 处的分布 $`P`$，消息层面的平均增量为 $`D_t(z,y)=\mathbb E_\xi\Delta_t(z,G_t(z,y,\xi))`$，并记

```math
\mu_t^P(z)=\mathbb E_{Y\sim P}D_t(z,Y),\qquad m_{k,t}^P(z)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(z,G_t(z,Y,\xi))^{\otimes j}.
```

- **Continuation value.** $`V^P_{t+1}:\mathcal Z\to[0,1]`$ 是协议 $`P`$ 的 continuation value：之后的更新都按 $`P`$ 进行，然后部署学生并用 $`F_E`$ 打分（[Definition 1.4](../README_ZH.md#s1-4)）。$`L(P)=\mathbb E\,F_E(Z_R^P)`$，$`d_t^Q`$ 是协议 $`Q`$ 下 $`Z_t`$ 的分布。
- **范数.** 在 $`\mathbb R^{p+q}`$ 上固定一个范数 $`\lVert\cdot\rVert`$，其对偶范数记为 $`\lVert\cdot\rVert_*`$。

**Adam 与 AdamW.** Adam 带 bias correction，$`\epsilon`$ 在根号外，可选 decoupled weight decay（AdamW），不用 AMSGrad。计数器为 $`t\ge1`$ 的一次更新作用在 $`z=(\theta,m,v,t-1)`$ 与 consumed gradient $`G\in\mathbb R^p`$ 上。逐坐标地，

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- 超参数满足 $`\beta_1\in[0,1)`$，$`\beta_2\in(0,1)`$，$`\eta>0`$，$`\lambda_{\rm wd}\ge0`$，$`\epsilon\ge0`$。运算都是精确实数运算。
- 下一个完整状态为 $`z^+=(\theta^+,m^+,v^+,t)`$。
- *cold* 状态指 $`m=v=0`$ 且 $`t=1`$；*warm* 状态指对每个坐标 $`i`$ 有 $`v_i>0`$。

*Warm 状态的记号（[Definition 5.4](../statements/05-04_warmth_frozen_denominator_surrogate_ZH.md#s5-4)）.*

考虑一次计数器为 $`t\ge1`$、作用在 $`z=(\theta,m,v,t-1)`$ 上的 Adam 更新，并设 $`v_i>0`$，或在某些 $`v_i=0`$ 时 $`\epsilon>0`$。

- *保留的分母、warmth、相对步长.* 对 consumed gradient $`G`$，

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^{\,t}}},\qquad\kappa_i(G)=\frac{(1-\beta_2)G_i^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(G)=\infty\ \ (v_i=0),\qquad u_i(G)=\frac{\lvert n_i(G)\rvert}{a_i+\epsilon}.
```

- 当 $`v_i>0`$ 时，$`d_i=a_i\sqrt{1+\kappa_i}`$，$`v_i^+=\beta_2v_i\,(1+\kappa_i)`$。warmth $`\kappa_i`$ 就是新梯度对保留下来的二阶矩造成的相对扰动。
- *Frozen-denominator surrogate.* 取固定的对角 preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$，

```math
\tilde\theta^+(G)=(1-\eta\lambda_{\rm wd})\theta-\eta\,\Phi\,n(G),\qquad\tilde\Delta(z,G)=\big(\tilde\theta^+(G)-\theta,\;m^+(G)-m,\;v^+(G)-v\big),\qquad\tilde z^+(G)=\big(\tilde\theta^+(G),\,m^+(G),\,\beta_2v,\,t\big).
```

- $`\tilde\theta^+`$ 是带固定 preconditioner $`\Phi`$ 的一步 momentum SGD。$`\tilde\Delta`$ 是 *frozen-denominator 增量*：它的 $`(\theta,m)`$ 块关于 $`G`$ 仿射，$`v`$ 块关于 $`G^{\odot2}`$ 仿射。$`\tilde z^+`$ 还把二阶矩冻结在 $`\beta_2v`$，是 Proposition 5.6 中用来比较的状态。对 $`g\in\mathbb R^p`$，$`\tilde z^+(g)`$ 表示同一映射在 consumed gradient 为 $`g`$ 时的值。

**状态标签.**

- `PASS` 表示实现的一致性断言全部通过。
- `INFO` 表示经验诊断：每个单元格单独标 PASS、FAIL 或 UNAVAILABLE，整组不作通过与否的判断。

除非另有说明，网络都是 `v014:C23` 的受控小网络：one-hot 上下文、4 个 tanh 单元、sigmoid 输出、平均 cross-entropy、float64。这些都不是真实模型。

**被检验的陈述（[Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)）.** 固定一轮 $`t\ge1`$、状态 $`z=(\theta,m,v,t-1)`$，以及有限消息集 $`Y`$，其 consumed gradient 为 $`G(y)`$。令 $`V:\mathcal Z\to[0,1]`$ 为更新之后的 continuation value（声明的参考尾段加部署，尾段随机性已积分），$`\psi(y)=V(z^+(y))`$ 为单边界 response，$`F(g)=V(\tilde z^+(g))`$。设下列条件成立，其中所有状态都由 $`Y`$ 中的消息从 $`z`$ 生成，计数器固定为 $`t`$：

- **(W)** 对每个坐标 $`v_i>0`$（warm 状态）；
- **(L)** 在 $`(m,v)=(m^+(y),v^+(y))`$ 处的点对 $`\big(\theta^+(y),\tilde\theta^+(y)\big)`$ 上，$`\lvert V(\vartheta,m,v)-V(\vartheta',m,v)\rvert\le\sum_iL_i\lvert\vartheta_i-\vartheta_i'\rvert`$；
- **$`(\mathrm S_v)`$** 存在 $`L_v\ge0`$ 与上限 $`\bar k\in(0,\infty]`$，使得在 $`(\vartheta,m)=(\tilde\theta^+(y),m^+(y))`$ 处，对 $`k=\kappa(y)`$ 有 $`\lvert V(\vartheta,m,\beta_2v\odot(1+k))-V(\vartheta,m,\beta_2v)\rvert\le L_v\max_i\min\lbrace k_i,\bar k\rbrace`$（由于 $`V\in[0,1]`$，取 $`L_v\bar k\ge1`$ 时它总成立；有内容的是 $`L_v`$ 小的情形）；
- **(H)** $`F`$ 在 $`\lbrace 0\rbrace \cup\lbrace G(y):y\in Y\rbrace`$ 的凸包 $`\mathcal C`$ 上二阶连续可微，且对所有 $`g\in\mathcal C`$、$`s\in\mathbb R^p`$ 有 $`\lvert s^\top\nabla^2F(g)\,s\rvert\le H_G\lVert s\rVert_\infty^2`$。

令 $`\psi_0=F(0)`$，$`b=\nabla F(0)`$，二者都与 $`y`$ 无关。则对每个 $`y\in Y`$，

```math
\big\lvert\psi(y)-\psi_0-b^\top G(y)\big\rvert\le\eta\sum_iL_iu_i(y)\min\lbrace \tfrac{\kappa_i(y)}2,1\rbrace +L_v\max_i\min\lbrace \kappa_i(y),\bar k\rbrace +\tfrac12H_G\lVert G(y)\rVert_\infty^2=:r(y),
```

并且对 $`Y`$ 上任意两个消息分布 $`P_0,P_1`$，

```math
\big\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\big\rvert\le\min\Big\lbrace 1,\;\big\lvert b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)\big\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r\Big\rbrace .
```

- 特别地，平均 consumed gradient 相同的两个分布，response 之差至多为 $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$。
- 若 $`V`$ 关于 $`(\theta,m)`$ 可微，一阶统计量就是经过 preconditioner 的平均 consumed gradient，权重为

```math
b=(1-\beta_1)\Big[-\frac{\eta}{1-\beta_1^{\,t}}\,\Phi\,\nabla_\theta V+\nabla_mV\Big]\Big|_{\tilde z^+(0)}.
```

| 检验 | 检验对象 | 代码编号 | 结果 |
|---|---|---|---|
| [Check 5.9](../experiments/05-06_warm_adam_first_order_affine_ZH.md#c5-9) | [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)、[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11) | `v015:E8`（warm 仿射部分） | PASS |
| [Check 5.13](../experiments/05-08_evolution_warmth_ZH.md#c5-13) | 整段训练中的 [Propositions 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)–[5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8) | `v015:E8`（`data.gaps`） | PASS |

<a id="c5-9"></a>

## Check 5.9 · Warm Adam 关于 consumed gradient 一阶仿射（Proposition 5.6、Remark 5.11）

**结论.** 在受控网络的实际 warm 状态上，常数在评估任何一对分布之前就已固定。界对全部 2,496 对分布成立：差距/界最大为 0.988，逐点残差/界最大为 0.998。

**动机.** [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6) 在 warm 状态下给出 response 之差的界：

- 平均 consumed gradient 相同的两个消息分布，response 之差不超过 $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$；
- 均值不同时，不超过 $`\lvert b^\top\Delta\mathbb EG\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r`$。

**设置**（`v015:E8`，warm 仿射部分）.

- 状态：网络的实际 warm 状态，$`t`$ 在 21 到 200 之间。
- 可训练坐标：输出头的 1–4 个坐标，其余固定。
- 消息：每个面板 2–6 条。
- 边界与尾段：
  - 一次实际的 Adam 更新作为边界；
  - 之后是长度 $`R\in\lbrace 0,2,6\rbrace`$ 的 IID 参考尾段，标签全 0 或全 1，各占概率 $`1/2`$；
  - 全部 $`2^R`$ 条尾段历史精确枚举。
- $`\eta\in\lbrace 0.01,0.1\rbrace`$；部署取 moment（$`N=3`$）或 parity（$`N=2`$）回报。
- 常数在评估任何一对分布*之前*，用预先声明的中心差分估计，之后从不放大：
  - $`L_i`$ 沿实际状态到 surrogate 的线段估计；
  - $`L_v`$ 沿二阶矩线段估计，$`\bar k=1`$；
  - $`H_G`$ 取径向与凸包样本上 Hessian 元素绝对值之和。
- $`b`$ 由解析的前向敏感度给出，并与独立的有限差分核对。
- 分布对：随机与对抗构造，均值相同与不同的都有。

**结果**（`v015:E8`，warm 仿射部分）.

| 量 | 值 |
|---|---|
| 面板 × 分布对 | 24 × 104 = 2,496 |
| 违反两分布界的对 | 0 |
| 全部分布对上差距/界的最大值 | 0.988 |
| 逐点 $`\lvert\psi-c-b^\top G\rvert/r`$ 的最大值 | 0.998 |
| 解析 $`b`$ 与有限差分 | 每个面板上一致到 $`\le10^{-8}`$ |
| 被测状态的 warmth（对坐标与消息取最大值） | 每个面板 $`\max\kappa\le0.081`$ |
| 曲率常数与步长 | 例如 moment、$`p=1`$、$`R=0`$：$`\eta=0.01`$ 时 $`H_G=2.1\times10^{-6}`$，$`\eta=0.1`$ 时 $`1.3\times10^{-4}`$ |

[Figure 5.3](../experiments/05-08_evolution_warmth_ZH.md#c5-11) 的 (C) 画出了逐点残差与其界。

**为什么支持 Proposition 5.6.**

- 在实际网络状态上，跨不同尾段长度与步长，界都成立，并且几乎被达到。
- 曲率常数大致按 $`\eta^2`$ 增长，与界中曲率项的预期一致。
- warmth 条件是按对坐标与消息的最大值核对的，不是按分位数，这正是 [Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11) 要求的形式。
- 证据的局限：常数是估计出来的，所以这是经验上的一致，不是 certificate。有效性由证明给出；这一检验确认了所列的残差项就是起作用的那几项。

### [Check 5.13](../experiments/05-08_evolution_warmth_ZH.md#c5-13) · 整段训练中的 cold 窗口（Propositions 5.6–5.8）

**这里检验什么.** 这项检验的完整描述在 Proposition 5.8 的页面上。对 Proposition 5.6，它检验的是：整段训练中的 [Propositions 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)–[5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8)。

**结论.**

- 在短视野下，矩相同的两个来源之间的差距是 cold 窗口的效应。$`W=10`$ 的 lr = 0 moment warm-up 消除了这一差距的中位数 99%（$`R=1`$）与 97.5%（$`R=4`$）。moment $`N=3`$、$`\eta=0.1`$ 时，差距在 $`R=1`$ 从 0.0769 降到 0.0004，在 $`R=4`$ 从 0.1196 降到 0.0016。
- $`R=40`$ 时，$`W\ge10`$ 仍留下至多 0.049 的差距；在残差最大的地方，SGD 对照也不为零（至多 0.011）。

**设置概要.**

- 不加改动地复用 [Check 5.7](../experiments/05-03_cold_start_adam_sign_quotient_ZH.md#c5-7) 的分支（实际 Adam、冻结 Jacobian 的 Adam、切线仿射 SGD）。
- 网格：5 个任务（moment $`N=2,3,4`$；parity $`N=2,3`$）× $`\eta\in\lbrace 0.01,0.1\rbrace`$ × $`R\in\lbrace 1,4,40\rbrace`$ × $`W\in\lbrace 0,1,2,5,10,20,50\rbrace`$，共 210 个单元格。
- 不同 $`W`$ 与不同分支使用完全相同的历史。历史不超过 4,096 条时精确枚举，否则每个来源独立运行 2,000 次。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

## 产物

- `theory/v015_checks/results_v015_independent.json`：
  - `v015:V8`，以及 `v015:E10`（逐单元格记录在 `data.cells`，汇总在 `data.optimizer_summary`，cold Adam 单元格在 `data.C23_prediction`，设计在 `data.design`）；
  - `v015:V2`、`v015:V3`、`v015:V4`、`v015:E0`、`v015:E1`、`v015:E2`、`v015:E3`；
  - `v015:E8`：冻结分母、warm 仿射、cold 分离与 warmth 轨迹四个部分（最后一部分含 `per_step` 与 `fits`），以及 210 个 `data.gaps` 单元格；
  - `v015:E11`（`data.clipped_mean`、`data.always_active`、`data.warm_Adam`、`data.Lion`、`data.LoRA`）。
- `theory/v015_checks/c23_bundle.json`：精确分布、网络、optimizer 代码，以及每条历史的原始更新。
- `theory/v015_checks/raw/E8_trajectories.npz`：逐副本的 warmth 轨迹。它被 gitignore，由上面第一条命令重新生成；SHA-256 记录在 `v015:E8` warmth 轨迹部分的 `raw_sha256` 中。
- `theory/v016_checks/results_v016.json`：claim 为 `v016:C1.exact`、`v016:C1.moments`、`v016:C1.accumulation`。完整数组在 `theory/v016_checks/results_v016_raw.json.gz` 中，该文件被 gitignore，SHA-256 记录在 `raw_artifact`。
- `theory/independent_checks/results_independent_v014.json`：`controlled_neural`。
- `docs/direction/v015_revision/results_v015.json`：package checks `pkg:T03`–`pkg:T08`。
- 代码：
  - `theory/v015_checks/otlr.py`（`check_V8`、`check_E10`、`check_E11`）；
  - `theory/v015_checks/warmth.py`（`check_E8`）；
  - `theory/v016_checks/check_v016.py`（`v016:C1`）；
  - 说明在 `theory/v015_checks/README.md` 与 `theory/v016_checks/SPEC.md`。
- 图：`../figures/fig5_1_linear_response`、`fig5_2_cold_adam`、`fig5_3_warmth`、`fig5_4_learner_statistics`，各有 `.pdf` 与 `.png`；说明在 `../figures/specs/fig5_1.md` 至 `fig5_4.md`。
