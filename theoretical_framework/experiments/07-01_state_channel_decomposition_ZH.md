# Theorem 7.1 · 状态通道分解

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/07-01_state_channel_decomposition_ZH.md) · [证明](../proofs/07-01_state_channel_decomposition_ZH.md) · [实验](07-01_state_channel_decomposition_ZH.md) · [English](07-01_state_channel_decomposition.md)

**本页检验什么.** 定理的 (a)：反向 costate 递推给出 continuation value 关于完整 Adam/AdamW 状态 $`(\theta,m,v)`$ 的精确梯度。(b) 的通道拆分与 (d) 的闭式都是 (a) 的代数推论。**结论：通过.** 在全部 1,024 次方向比较中，adjoint 梯度都与中心差分一致，最大相对误差 $`2.8\times10^{-7}`$（容差 $`10^{-6}`$）。这些比较覆盖 256 组配置，包括 bias correction、decoupled weight decay、$`\epsilon>0`$ 与生效的全局范数裁剪。

**本页用到的设定与记号.**

- $`z=(\theta,m,v,s)`$ 是完整 Adam 状态；超参数为 $`\beta_1,\beta_2,\eta,\epsilon`$、decoupled weight decay $`\lambda_{\rm wd}`$、bias correction 因子 $`q_{1,s},q_{2,s}`$ 的 AdamW（[Kingma & Ba, 2015](#ref-kingma2015adam)；[Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)）把 consumed gradient $`g`$ 映为增量 $`\Delta(z,g)`$（[Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)）。
- 第 $`t`$ 轮的消息之后，参考尾段用 consumed gradient $`g_k(\theta)`$（可能经全局范数裁剪）做 $`R-1-t`$ 次更新；$`V_{t+1}(z)`$ 是从 $`z`$ 出发得到的最终得分 $`F(\theta_R)`$。
- costate $`\chi_{t+1}=(\chi^\theta_{t+1},\chi^m_{t+1},\chi^v_{t+1})`$ 是定理中反向递推的输出；下文的"adjoint 梯度"指 $`\chi_{t+1}`$。

**被检验的陈述（[Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1)）.** 若 $`F`$ 属于 $`C^1`$，每个尾段梯度 $`g_k`$ 在尾段轨迹附近关于 $`\theta`$ 属于 $`C^1`$（裁剪不落在阈值上），且尾段上 $`v'_k>0`$，则 $`V_{t+1}`$ 在 $`z_t`$ 处关于 $`(\theta,m,v)`$ 可微，且 $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$，其中 $`\chi`$ 从 $`\chi_R=(\nabla F,0,0)`$ 出发满足

```math
\begin{aligned}
\chi^m_k&=\beta_1\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big),\qquad \chi^v_k=\beta_2\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big),\\
\chi^\theta_k&=(1-\eta\lambda_{\rm wd})\chi^\theta_{k+1}+J_k^\top\Big[(1-\beta_1)\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big)+2(1-\beta_2)\,\mathrm{diag}(g_k)\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big)\Big],
\end{aligned}
```

其中 $`D_1^{(k)}=\mathrm{diag}\big(q_{1,s_k}/(\sqrt{q_{2,s_k}v'_k}+\epsilon)\big)`$，$`D_2^{(k)}=\mathrm{diag}\big(q_{1,s_k}q_{2,s_k}m'_k/(2\sqrt{q_{2,s_k}v'_k}(\sqrt{q_{2,s_k}v'_k}+\epsilon)^2)\big)`$，$`J_k=\partial g_k/\partial\theta`$（裁剪生效时为裁剪的 Jacobian）。于是一条消息的一阶价值拆成参数、动量、二阶矩三个通道。

<a id="c7-1"></a>

## Check 7.1 · costate 递推与有限差分（Theorem 7.1）

**结论.** 通过。1,024 次比较中，方向导数的最大相对误差为 $`2.82\times10^{-7}`$（容差 $`10^{-6}`$）。

**动机.** [Theorem 7.1(a)](../statements/07-01_state_channel_decomposition_ZH.md#s7-1) 声称反向递推给出 continuation value 关于 $`(\theta,m,v)`$ 的精确梯度。这个梯度含 bias correction 因子、decoupled weight decay、$`\epsilon`$ 以及全局范数裁剪的 Jacobian。递推中任何一个块写错，都会在某个方向上表现为不一致。

**设置.**

- float64 下的精确 AdamW，损失为二次或 logistic，参数维数 $`p\in\lbrace5,20\rbrace`$，最终得分为 $`\mathrm{sigmoid}(a^\top\theta)`$，$`\eta=0.007`$。
- 全因子设计：bias correction 关或开；$`\epsilon\in\lbrace10^{-8},10^{-3}\rbrace`$；decoupled weight decay 为 $`0`$ 或 $`0.03`$；$`\beta_1\in\lbrace0,0.9\rbrace`$；$`\beta_2\in\lbrace0.99,0.999\rbrace`$；全局范数裁剪关或阈值 $`0.15`$。共 256 组配置。
- 尾段长度为 20、60 或 200 次更新，初始状态的 $`v`$ 为正。
- 每组配置沿随机的 $`\theta`$、$`m`$、$`v`$ 单独方向和联合方向，对 $`V_{t+1}`$ 做中心差分。步长取 $`h=2\times10^{-4}`$ 与 $`h/2`$，再做 Richardson 外推。
- 若扰动后的轨迹改变了任何一次裁剪决定，就把差分步长减半。这一判断只看前向的裁剪模式，从不参考与 costate 是否一致。
- 容差：相对误差 $`\le10^{-6}`$，分母为 $`\max(\lvert\text{adjoint}\rvert,\lvert\text{FD}\rvert,10^{-10})`$。不使用自动微分。

**结果.**

- 1,024/1,024 次方向比较在容差内，最大相对误差 $`2.82\times10^{-7}`$。
- 1,024 个被采用的差分步都与未扰动轨迹处在同一裁剪分支上。另有两次尝试的步长跨过了裁剪阈值，已减半。
- 比较中包括 $`\beta_1=0`$ 时恰好为零的动量导数。

**为什么支持该陈述.**

- 检验覆盖了递推的每一项，包括超出普通 Adam 步的那些扩展：bias correction、decoupled decay、$`\epsilon`$，以及不对称的裁剪 Jacobian。
- 它检验的是 (a) 中的恒等式。(b) 的通道拆分与 (d) 的闭式都是代数推论。

**局限.** 损失光滑、维数低；按设计不在裁剪阈值上求导，与定理一致。

## 复现

在仓库根目录下运行（仅 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C4
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

- costate 这一组（`v016:C4.costate`）在 rai 上约 16 秒；不加 `--only` 时整套约 104 秒。
- 随机种子：`numpy.SeedSequence([160160, group, subcase, ...])`，PCG64，float64，单线程 BLAS。

本检验使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C4.costate`，含断言与逐配置误差。
- `theory/v016_checks/results_v016_raw.json.gz`：原始数组，通过 JSON pointer 引用。未提交到 git；SHA-256 为 `1777069872ea80c09dfd5260d610871b68e41a10721eba2826dcd4d647874f69`。
- `theory/v016_checks/SPEC.md`（`v016:C4` 一节）、`check_v016.py` 及其测试。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
