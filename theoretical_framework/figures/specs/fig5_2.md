# Figure spec — fig5_2 (`figures/fig5_2_cold_adam.pdf`, `figures/fig5_2_cold_adam.png`)

**Message.** Two label laws with identical gradient moments through order 2 are separated by cold Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)), not by SGD. The separation is an optimizer effect, since it survives a frozen Jacobian. It disappears when sign disagreement is removed and nearly disappears with warm moments. Optimizer memory makes same-sign histories diverge.

**Layout.** Full-width rows, at most two subpanels per row, 10 inches wide. PDF vector, PNG 300 dpi. Base/labels 14 pt; titles 15 pt; ticks, legends and annotations at least 12 pt (15 px at 900 px width). Drawn by `figures/make_figures.py` with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)).

**Panel A — "Same moments, different Adam response".**
- Data: `theory/v015_checks/results_v015_independent.json → groups[id=E2].data.comparisons`.
- Cells: (moment N=3, η=0.1, R=1), (moment N=3, η=0.1, R=4), (moment N=3, η=0.01, R=4), (parity N=2, η=0.1, R=4), (moment N=4, η=0.1, R=4).
- Grouped bars per cell for `gaps.actual_Adam`, `gaps.frozen_Adam`, `gaps.actual_SGD`, `gaps.tangent_affine_SGD`. Use the same hue order in every group, a zero line, and value labels on the Adam bars.
- y label: "final return gap E_P − E_Q".

**Panel B — "One-factor ablations (moment N=3, η=0.1)".**
- Data: `groups[id=E1].data.ablations` rows with `task=="moment"` and `N==3`.
- Horizontal bars of `gap` for R=1 and R=4 side by side, for variants in this order: baseline, inside_sqrt, epsilon_1e-06, epsilon_0.0001, epsilon_0.01, epsilon_0.1, no_bias_correction, global_clip_0.1, sum_reduction, warm_5, shift_support.
- Highlight warm_5 and shift_support.

**Panel C — "Same signs, different memory (Prop. 5.3(e))".**
- Data: `groups[id=V4].data.exact` (two branches). Use θ0 = 0 and η = 1.
- Plot θ for steps t = 0, 1, 2. Branch G1=1: θ1 = −1, θ2 = −1 + `delta2` (=+√102/51). Branch G1=2: θ2 = −1 − √246/123.
- Annotate $`\widehat m_2=\mp1/6`$ and $`\widehat v_2=17/24, 41/24`$.
- Inset text: "β-grid witnesses: 20/25 cells, all with β1>0" from `data.grid_fraction`/`data.grid`.

**Caption.** Left: cold Adam separates laws whose first two gradient moments agree, and a frozen-Jacobian Adam reproduces the gap while SGD shows small residual gaps. Middle: removing sign disagreement (shifted support) nearly removes the gap, and warm moments remove most of it. Right: two histories with identical sign sequences take opposite second steps because m retains magnitudes.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../../references.md).

<a id="ref-hunter2007matplotlib"></a>

**John D. Hunter (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering* 9, no. 3, 90–95. DOI [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).
