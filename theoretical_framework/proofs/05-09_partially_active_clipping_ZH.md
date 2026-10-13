# Proposition 5.9 · 部分激活的梯度裁剪

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-09_partially_active_clipping_ZH.md) · [证明](../proofs/05-09_partially_active_clipping_ZH.md) · [实验](../experiments/05-09_partially_active_clipping_ZH.md) · [English](05-09_partially_active_clipping.md)

**本页证明什么.** 在"裁剪不生效"的事件上拆分得到恒等式，再给出范数界；两个分布的情形由三角不等式得到。这里的裁剪映射就是 [Pascanu et al., 2013](#ref-pascanu2013difficulty) 提出的全局范数裁剪。

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

**陈述（[Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)）.** 令 $`\mathrm{clip}_c(g)=g\min\lbrace1,c/\lVert g\rVert\rbrace`$，$`c>0`$；约定 $`G=0`$ 时 $`G/\lVert G\rVert=0`$；令 $`p_{\rm act}=\Pr(\lVert G\rVert>c)`$ 为某个 $`G`$ 的分布下的激活率。则

```math
\mathbb E\,\mathrm{clip}_c(G)=\mathbb E\big[G\,1\lbrace\lVert G\rVert\le c\rbrace\big]+c\,\mathbb E\Big[\frac{G}{\lVert G\rVert}1\lbrace\lVert G\rVert>c\rbrace\Big],\qquad \Big\lVert\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E\frac{G}{\lVert G\rVert}\Big\rVert\le c\,(1-p_{\rm act}).
```

- 对两个分布，裁剪后均值与缩放后的平均方向之间的总偏差至多为 $`c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$。例如两个分布的激活率都是 $`0.9`$ 时，用平均方向代替裁剪后均值，在差值上的误差至多为 $`0.2c`$。
- 只有在始终激活（$`p_{\rm act}=1`$）时，平均方向才恰好等于裁剪后均值。

*证明.* 第一式是按事件 $`\lbrace\lVert G\rVert\le c\rbrace`$ 拆分 $`\mathbb E\,\mathrm{clip}_c(G)`$。第二式：$`\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E[G/\lVert G\rVert]=\mathbb E\big[(G-cG/\lVert G\rVert)1\lbrace\lVert G\rVert\le c\rbrace\big]`$，在该事件上（包括 $`G=0`$）$`\lVert G-cG/\lVert G\rVert\rVert=\lvert\lVert G\rVert-c\rvert=c-\lVert G\rVert\le c`$。所以范数至多为 $`\mathbb E[(c-\lVert G\rVert)1\lbrace\lVert G\rVert\le c\rbrace]\le c\Pr(\lVert G\rVert\le c)=c(1-p_{\rm act})`$。两个分布的结论由三角不等式得到。$`\square`$

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-pascanu2013difficulty"></a>

**Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013).** On the difficulty of training recurrent neural networks. *Proceedings of the 30th International Conference on Machine Learning (ICML)*, PMLR 28, no. 3, 1310–1318. [proceedings.mlr.press/v28/pascanu13](https://proceedings.mlr.press/v28/pascanu13.html).
