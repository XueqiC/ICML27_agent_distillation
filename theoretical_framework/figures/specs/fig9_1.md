# Figure spec — fig9_1 · Coupling width

**Output:** `theoretical_framework/figures/fig9_1_coupling.pdf` (vector) and `fig9_1_coupling.png` (300 dpi). Layout: full-width rows, at most two subpanels per row, 10 inches wide. English labels. Drawn with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)) by `figures/make_figures.py`.

**Message:** regrouping identical-marginal feedback has an exactly zero effect at one common-state step. After sequential updates it has a real effect that grows with step size and horizon and depends on the optimizer.

## Panel A — C23(e): regrouping gap vs number of updates

- **Data:** `theory/independent_checks/results_independent_v014.json`, `controlled_neural.details.regrouping`. There are 18 rows with fields `optimizer`, `step_size`, `updates` and `neural_gap`.
- **Plot:**
  - x = updates (2, 8, 80) on a log axis. y = |neural_gap| on a log axis, from 1e-6 to 1.
  - One line plus markers per (optimizer, η). SGD η ∈ {0.01, 0.1, 1, 3} as solid lines in a sequential single-hue ramp, light to dark with η. Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) η ∈ {0.01, 0.1} as dashed lines in a second hue.
  - Mark sign: filled markers where gap > 0 (SGD), open markers where gap < 0 (Adam).
  - Direct labels at the right end, e.g. "SGD η=3", "Adam η=0.1".
- **Inset or text box:** "single common-state step: parameter difference = 0 (exact null)". Read the value from `controlled_neural.details.additive_null_parameter_residual`.
- **Title:** "Same labels, different packets: learning-consequence gap".

## Panel B — Width is zero exactly for additive responses

- **Data:** `theory/v014_repro/results_v014_rerun.json`, `coupling_and_additive_null`: `additive_width` (0.0), `quadratic_pair_min` (0.25), `quadratic_pair_max` (0.5) and `quadratic_pair_width` (0.25). Also `theory/independent_checks/results_independent_v014.json`, `claims[id=C17].details`: `median_inequality_slack` (0.2306) and `maximum_slack` (0.7115).
- **Plot:** two horizontal range bars [min, max] of ⟨Γ, H⟩ over the coupling polytope, the feasible set of a transportation problem ([Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)).
  - "additive H": a degenerate bar with a point marker; draw a single point if min/max are absent. Width 0.
  - "quadratic pair H": the bar from 0.25 to 0.50. Width 0.25.
- **Text box:** "C17: 320 random tables; (46) slack median 0.23, max 0.71".
- **Title:** "Coupling width: 0 iff additive".

**Caption.** (A) In a controlled network (C23(e)), grouping the same four labels as {(0,0),(1,1)} versus {(0,1),(0,1)} changes the final return. The gap is exactly zero for one common-state step and grows with step size and horizon under SGD; under Adam it has the opposite sign and other dynamics. (B) The range of expected packet response over the coupling polytope: a single point for an additive table, positive width for a non-additive one. Only the additive width is stored, so its single point has schematic position. Exact finite computations and a controlled small network; not a real-model result.
