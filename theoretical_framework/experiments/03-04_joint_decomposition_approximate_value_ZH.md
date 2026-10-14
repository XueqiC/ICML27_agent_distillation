# Proposition 3.4 · 用近似价值做的联合分解

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/03-04_joint_decomposition_approximate_value_ZH.md) · [证明](../proofs/03-04_joint_decomposition_approximate_value_ZH.md) · [实验](03-04_joint_decomposition_approximate_value_ZH.md) · [English](03-04_joint_decomposition_approximate_value.md)

**本页检验什么.**

- **(a) 精确恒等式.** 在一个四轮、带非线性 Adam 式更新的 learner 上，联合分解以 $`1.4\times10^{-17}`$ 的误差重现精确学习差距 $`0.07616`$（**通过**）。
- **(b) 绝对值界.** 界为 $`0.178`$，有效，约为差距的 2.3 倍（**通过**）。
- **(d) 带停止的版本.** 认证集合被两条过程分别以 $`0.667`$ 和 $`0.424`$ 的概率离开，停止界为 $`1.197`$：有效，但这里大于 1，因此不提供信息（**通过**，记为有效但偏松）。

**本页用到的设定.**

- 在 complete learner state 上的参照协议与备选协议 $`K_t^0,K_t^S`$，公共初始状态；$`L_E(S)=\mathbb E_SF_E(Z_R)`$。
- 近似价值 $`\widehat V_t`$，参照 Bellman 残差 $`r_t^0=K_t^0\widehat V_{t+1}-\widehat V_t`$，终点误差 $`e_R=F_E-\widehat V_R`$，局部 response 差 $`d_t=(K_t^0-K_t^S)\widehat V_{t+1}`$。
- (d) 所用的认证集合 $`C_t^j`$ 与每条过程各自的离开时刻 $`\tau^j`$。

**被检验的陈述（[Proposition 3.4](../statements/03-04_joint_decomposition_approximate_value_ZH.md#s3-4)）.** (a) $`L_E(0)-L_E(S)=\mathbb E_S\sum_td_t+\mathbb E_0\sum_tr_t^0-\mathbb E_S\sum_tr_t^0+\mathbb E_0e_R-\mathbb E_Se_R`$。(b) 逐项取绝对值即可界住差距。(d) 当 $`\widehat V\in[0,1]`$ 时，让每条过程在它离开认证集合时停止，差距被停止的和、未离开路径上的终点误差以及两个离开概率界住。

| Check | 检验内容 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 3.5](#c3-5) | (a)、(b)、(d) | `v017:C1.joint` | 通过 |

<a id="c3-5"></a>

## Check 3.5 · 精确联合分解、绝对值界与停止界（Proposition 3.4）

**结论.** 恒等式在舍入误差内成立。两个界都有效。绝对值界是差距的 2.3 倍；停止界超过 1，在这个例子里不提供信息。

**为什么做.**

- (a) 是对任意近似价值都成立的精确恒等式。符号或占据分布上的错误（例如 response 项误用参照过程的占据分布），在两条链访问不同状态的过程上会留下非零残差。
- 这个 learner 是非线性的，并带着完整的 optimizer state，所以这个检验也在验证"不需要线性化、平稳性或收缩"这一点。
- (d) 让每条过程的离开各付一次。这个检验也展示了代价：离开概率大时，界有效但没用。

**设置.**

- 完整状态 $`z=(\theta,m,v,s)`$，$`s`$ 是步数计数器，初值 $`(0.2,0.03,0.08,2)`$。消息 $`y\in\lbrace-1,+1\rbrace`$。
- 更新：梯度 $`\theta+0.2\theta^3-y`$；动量 $`m'=0.7m+0.3\,\mathrm{grad}`$、$`v'=0.85v+0.15\,\mathrm{grad}^2`$；用 $`0.7^{s+1}`$ 与 $`0.85^{s+1}`$ 做 bias correction；步长 $`\theta'=\theta-0.17\,\hat m/(\sqrt{\hat v}+0.01)`$。
- feedback law：参照为 $`\Pr(y=+1\mid z)=\sigma(0.6+0.5\theta-0.2m)`$，备选为 $`\sigma(-0.4+0.5\theta-0.2m)`$，$`\sigma`$ 是 logistic 函数。
- $`R=4`$ 轮；最终得分 $`F_E=\sigma(1.3\theta-0.25\theta^2)`$；近似价值 $`t<R`$ 时为 $`\widehat V_t=\sigma(0.15+0.03t+0.4\theta)`$，且 $`\widehat V_R=F_E`$（因此 $`e_R=0`$）。
- 精确枚举每条过程的全部 $`2^4`$ 条路径。
- (d)：两条过程用同一个认证集合 $`C_t=\lbrace\theta\le0.2\rbrace`$；阈值 $`0.2`$ 取 $`\theta_0`$ 与第 3 轮参数 60% 分位数中较大者。

**结果.**

| 量 | 值 | JSON 键（条目 `v017:C1.joint`） |
|---|---|---|
| 精确差距 $`L_E(0)-L_E(S)`$ | $`0.076162`$ | `return_gap` |
| 恒等式 (3.7) 的误差 | $`1.4\times10^{-17}`$ | `joint_identity_error` |
| 带符号的 response 项 $`\mathbb E_S\sum_td_t`$ | $`0.021068`$ | `signed_response_term` |
| 带符号的残差之差 $`\mathbb E_0\sum_tr_t^0-\mathbb E_S\sum_tr_t^0`$ | $`0.055093`$ | `signed_residual_difference` |
| 绝对值界 (3.8) | $`0.17818`$ | `absolute_bound` |
| 停止界 (3.9) | $`1.1975`$ | `stopped_bound` |
| 离开概率，参照 / 备选 | $`0.667`$ / $`0.424`$ | `exit_probability_reference`、`exit_probability_alternative` |

**为什么支持这个陈述.**

- 两个带符号的部分加起来正好是差距，与 (3.7) 一致。差距的大部分（0.076 中的 0.055）落在残差之差上：近似价值在两条过程上不准时，记账从 response 项移到残差项，恒等式本身照样精确。
- 绝对值界有效，但超过差距的两倍，因为它把在 (3.7) 中互相抵消的残差加在了一起。这就是陈述建议估计带符号各项的原因。
- 停止界有效，但两个离开概率都大，所以它超过 1；每个离开概率只付一次，与 (3.9) 一致，但在这里一次就已经太多了。

## 复现

检验用到 NumPy ([Harris et al., 2020](#ref-harris2020array))。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C1.joint`；`theory/v017_checks/check_v017.py` 中的函数 `c_joint`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
