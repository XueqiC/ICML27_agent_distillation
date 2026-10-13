# Figure spec — fig5_1 · Optimizer-transformed linear response

**Output:** `theoretical_framework/figures/fig5_1_linear_response.pdf` (vector) and `fig5_1_linear_response.png` (300 dpi). Layout: full-width rows, at most two subpanels per row, 10 inches wide. English labels. Drawn by `figures/make_figures.py` with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)).

**Message:** The first-order representation is a tool. The first-order term of the *optimizer increment* explains the gap between feedback laws: the C23 cold-Adam gap is first order while the raw mean gradient sees nothing. The remainder has the predicted order — $`\eta^2`$ (or $`\eta^3`$ when two moments match) for SGD and sign optimizers, $`\eta^1`$ for multi-round Adam — and stays below its bound in every smooth exact cell.

All data come from `theory/v015_checks/results_v015_independent.json`, `groups[id=E10].data`. Values stored as strings (`"nan"`) are missing; skip them and never plot them as zero.

**Optimizer colors** (fixed categorical order, all panels): SGD, heavy_ball, Nesterov, Adam_cold, Adam_warm5, Adam_warm20, AdamW, clipped_SGD, clipped_Adam, signSGD, Lion. Display names: "SGD", "heavy-ball", "Nesterov", "Adam cold", "Adam warm W=5", "Adam warm W=20", "AdamW", "clipped SGD", "clipped Adam", "signSGD", "Lion". The optimizers follow their sources: Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)), AdamW ([Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)), signSGD ([Bernstein et al., 2018](#ref-bernstein2018signsgd)) and Lion ([Chen et al., 2023](#ref-chen2023symbolic)); clipping is global-norm clipping ([Pascanu et al., 2013](#ref-pascanu2013difficulty)).

## Panel A — "Remainder order in η (moment N=3)"

- **Data:** `cells` with `task=="moment"`, `N==3`, `R in {1,4}`, and `optimizer in {SGD, Adam_cold, Adam_warm20, signSGD}`. Use x = `eta` and y = `remainder` (= |gap − first_order|).
- **Plot:** log–log lines with markers. Color by optimizer; solid line for R=1, dashed for R=4. Omit points with `remainder <= 1e-15` (numerical floor).
- **Slope guides:** thin grey dotted guides with slopes 1, 2 and 3, anchored at the largest-η point of the Adam cold R=4 line (slope 1), the signSGD R=1 line (slope 2) and the SGD R=1 line (slope 3). Label them "η¹", "η²", "η³".
- **Annotation:** read from `optimizer_summary[*].slopes` (entries with task moment, N=3) and print the fitted slopes next to each line end, e.g. "Adam cold R=4: 0.99".
- **Title:** "Remainder order in η (moment N=3)".

## Panel B — "C23 cold-Adam gap: increment vs raw gradient"

- **Data:** `C23_prediction`, fields `gap`, `first_order`, `naive_first_order`, `remainder`, `bound`.
- **Plot:** horizontal bars on a log x-axis from 1e-12 to 1. Bars in this order: "measured gap", "first order (increment)", "first order (raw mean gradient)", "remainder |gap − first order|", "remainder bound". Use absolute values; mark the naive bar with its sign in the label ("−8.9e−11"). Print each value at the bar end in text ink.
- **Annotation:** "first order explains 98.4% of the gap" computed as `first_order / gap`.
- **Title:** "C23 cold-Adam gap: increment vs raw gradient".

## Panel C — "Bound check in exact cells (R = 1, 4)"

- **Data:** for each optimizer, the maximum of numeric `ratio` over `cells` with `R in {1,4}`. Reproduction check: it must equal `optimizer_summary[optimizer].max_exact_ratio` to 1e-12; if not, stop and report.
- **Plot:** horizontal bars, one per optimizer in the fixed order, with a vertical line at ratio 1 labelled "remainder bound". Hatch the clipped SGD bar, which is above 1, and label it "multi-step clipped SGD: S1 not guaranteed".
- **Annotation:** a small text "R=40 Monte Carlo: 2 of 330 cells above bound after 4 SE (sign-type)" computed from `cells` with `R==40` and positive numeric `excess_over_bound_after_four_SE`.
- **Title:** "Bound check in exact cells (R = 1, 4)".

**Caption.** The first-order representation is a tool. (A) Remainder orders for moment N=3; fitted slopes are read from E10. The slope-2 line is a reference guide, while the selected signSGD series is approximately cubic. (B) The optimizer-increment first-order term explains 98.4% of the Cold-Adam gap; the raw mean gradient is nearly blind. (C) All available exact-cell ratios are at most one except multi-step clipped SGD. Curvature bounds are empirical, and S1 is not certified. Controlled small network; not a real-model result.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../../references.md).

<a id="ref-bernstein2018signsgd"></a>

**Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, Animashree Anandkumar (2018).** signSGD: Compressed optimisation for non-convex problems. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:560–569. [proceedings.mlr.press/v80/bernstein18a](https://proceedings.mlr.press/v80/bernstein18a.html).

<a id="ref-chen2023symbolic"></a>

**Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le (2023).** Symbolic discovery of optimization algorithms. *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 49205–49233. arXiv [2302.06675](https://arxiv.org/abs/2302.06675).

<a id="ref-hunter2007matplotlib"></a>

**John D. Hunter (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering* 9, no. 3, 90–95. DOI [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-pascanu2013difficulty"></a>

**Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013).** On the difficulty of training recurrent neural networks. *Proceedings of the 30th International Conference on Machine Learning (ICML)*, PMLR 28, no. 3, 1310–1318. [proceedings.mlr.press/v28/pascanu13](https://proceedings.mlr.press/v28/pascanu13.html).
