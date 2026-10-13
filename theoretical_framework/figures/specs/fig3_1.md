# Figure spec — fig3_1 (output `fig3_1_static_vs_learning`)

**Layout.** Full-width rows, 10 inches wide, at most two subpanels per row. Base/labels 14 pt, titles 15 pt, all other text at least 12 pt.

**Rendering.** Drawn with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)) by `figures/make_figures.py`.

**Message.** Two sources look identical to every static evaluation, yet the students trained on them differ. The same holds for two packets with identical means. A real network shows the same separation.

**Data.**

- Panels (a) and (b) are analytic: use the formulas below and no data.
- Panel (c): `theory/independent_checks/results_independent_v014.json` → `controlled_neural.details.configurations`, filtered to `task` in {"Prop5", "Prop6"}. Fields: `optimizer`, `step_size`, `updates`, `neural_gap`, `SE`.

**Panel (a): Static equivalence, different trained students.**

- Plot $`J(\pi_p)=\tfrac12+p/20`$ for $`p\in[0,1]`$ as one black line, labelled "reference source = alternative source (identical for every p)".
- Mark the two distilled students:
  - $`p=\tfrac34`$, $`J=\tfrac{43}{80}`$: blue dot, labelled "trained on reference";
  - $`p=\tfrac14`$, $`J=\tfrac{41}{80}`$: orange dot, labelled "trained on alternative".
- Add a vertical bracket showing the gap $`1/40`$.
- Horizontal dashed line at the teacher value $`\tfrac35`$, labelled "teacher (both sources)".
- Axes: "student action-A probability p", "deployment return".

**Panel (b): Equal mean updates, different learning values.**

- Plot the gap $`\mathbb E_{\rm indep}J(p^+)-\mathbb E_{\rm paired}J(p^+)`$ for $`\eta\in[0,1]`$: $`+\eta^2/8`$ for $`J=p^2`$ (blue) and $`-\eta^2/8`$ for $`J=1-p^2`$ (orange).
- Zero line in grey.
- Annotation: "equal label marginals, equal mean gradient, equal mean update".
- Axes: "step size η", "learning-value gap".

**Panel (c): controlled neural network.**

- Grouped bar chart, log y axis, of `neural_gap` for configurations with `updates` in {1, 4}.
- Groups: (Prop5, Prop6) × optimizer and step size, i.e. SGD 0.01, SGD 0.1, SGD 1.0, Adam 0.01, Adam 0.1 (Adam: [Kingma & Ba, 2015](#ref-kingma2015adam)).
- Within each group, two bars for R=1 and R=4.
- All values are positive.
- Label the panel "controlled neural, not Gemma, not an external benchmark".
- Axes: "configuration", "trained-student return gap (log)".

**Caption.** (a) Identical static values for every aliased student, but frequency distillation yields 43/80 vs 41/80. (b) Identical mean updates, but a curvature-dependent gap ±η²/8. (c) In a real tanh network, the statically equivalent sources still give different trained students (exact enumeration, R ≤ 4).
