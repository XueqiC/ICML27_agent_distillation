# Remark 3.3 · response 类是推导出来的，不是挑出来的

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/03-03_response_class_derived_not_chosen_ZH.md) · [证明](03-03_response_class_derived_not_chosen_ZH.md) · [English](03-03_response_class_derived_not_chosen.md)

**本页证明什么.** 由 continuation value 与 response 的定义，每个 response 都有这种生成形式；反过来每个这种形式的函数都是一个 response。族扩大时 response 类随之包含，因此等价关系变细。

**本页用到的设定.**

- complete learner state $`z\in\mathcal Z`$：参数、optimizer state、缓冲区、采样器状态、随机性与历史（[Definition 1.3](../README_ZH.md#s1-3)）。
- 第 $`t<R`$ 轮，feedback law $`P_t(\cdot\mid z)`$ 发出消息 $`Y_t`$，实际更新给出 $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$，$`\xi_t`$ 是更新的随机性。对 $`Y_t`$ 与 $`\xi_t`$ 积分得到学习 kernel $`K_t`$，它作用在有界函数上：$`(K_tf)(z)=\mathbb E\,f(U_t(z,Y,\xi))`$。
- $`K_t^0`$ 是参照协议，$`K_t^S`$ 是备选协议 $`S`$，两者从同一初始分布 $`\nu`$ 出发。$`Z_t^S`$ 是由 $`K^S`$ 驱动的链，其分布 $`d_t^S`$ 是第 $`t`$ 轮的占据分布。
- $`F_E(z)\in[0,1]`$ 是从状态 $`z`$ 部署的学生在真实任务上的回报；$`L_E(S)=\mathbb E\,F_E(Z_R^S)`$ 是协议 $`S`$ 的学习价值，$`L_E(0)`$ 是参照的学习价值。
- 参照 continuation value 为 $`V_R^0=F_E`$，$`V_t^0=K_t^0V_{t+1}^0`$：从某个状态出发、按参照协议继续训练时的期望最终得分（[Definition 1.4](../README_ZH.md#s1-4)）。
- 快照 $`(t,z)`$ 处的单边界 response 为 $`g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))`$：收到消息 $`y`$、执行实际更新、再走完声明的参照尾段之后的期望最终回报。response 类 $`\mathcal G`$ 收集声明的快照族 $`\mathcal I`$ 上所有的 $`g_{t,z}`$。

**陈述（[Remark 3.3](../statements/03-03_response_class_derived_not_chosen_ZH.md#s3-3)）.** 类 $`\mathcal G`$ 由声明的对象递归生成：

- 从 $`\mathcal V_R=\lbrace F_E\rbrace`$ 出发；
- 作用参照 kernel，$`\mathcal V_t=K_t^0\mathcal V_{t+1}`$；
- 经实际更新拉回，$`g=\mathbb E_\xi\,v\circ U_t(z,\cdot,\xi)`$，其中 $`v\in\mathcal V_{t+1}`$。

因此，扩大声明的任务、学习器、后缀或快照族，只会让 [Theorem 3.2(b)](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) 的等价关系 (3.4) 更细；response 类由学习器与任务确定。

**证明.**

由 $`V^0`$ 与 $`g_{t,z}`$ 的定义，每个 $`g\in\mathcal G`$ 都有这种形式；反过来，每个这种形式的函数都是对应快照的 response。若 $`\mathcal I\subseteq\mathcal I'`$，则 $`\mathcal G(\mathcal I)\subseteq\mathcal G(\mathcal I')`$，所以对 $`\mathcal G(\mathcal I')`$ 满足 (3.4) 的每一对分布也对 $`\mathcal G(\mathcal I)`$ 满足。扩大声明的任务、学习器、后缀或快照族，只会让等价关系 (3.4) 更细。response 类由学习器与任务确定，不是为了建模方便的选择。

## 相关工作

- 这里的包含关系论证，与 model-based 强化学习中 value-equivalence 类的单调性背后的论证相同 ([Grimm et al., 2020](#ref-grimm2020value), Property 3)。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-grimm2020value"></a>

**Christopher Grimm, André Barreto, Satinder Singh, David Silver (2020).** The value equivalence principle for model-based reinforcement learning. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 5541–5552. arXiv [2011.03506](https://arxiv.org/abs/2011.03506).
