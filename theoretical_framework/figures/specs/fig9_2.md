# Figure spec — fig9_2 · Which measurements select feedback correctly

**Output:** `theoretical_framework/figures/fig9_2_selection.pdf` (vector) and `fig9_2_selection.png` (300 dpi). Layout: full-width rows, at most two subpanels per row, 10 inches wide. English labels.

**Message:** the tested static proxies did not reliably select; measured responses are more reliable on the tested held-out states. The threshold task needs the full tail. Warm-up shrinks the spread between providers but does not rescue static statistics.

All data come from `theory/v015_checks/results_v015_independent.json`.

**Proxy groups and colors** (fixed categorical order, used in all panels):

- *static*: `input_moment_error`, `mean_gradient`, `sign_only`, `first_delta`, `delta_plus_m_v`;
- *measured*: `short_tail_1`, `short_tail_4`;
- *controls*: `random`, `full_tail_gold`.

Display names: "input moment", "mean grad", "sign", "Δθ", "Δθ+m,v", "short tail 1", "short tail 4", "random", "full tail (gold)".

## Panel A — E6 wrong-sign rate by proxy and held-out split

- **Data:** `groups[id=E6].data.reports.<split>.metrics.<proxy>.wrong_sign_rate`. Splits: `held_out_states`, `fresh_to_warm`, `held_out_epsilon` and `held_out_lr`.
- **Plot:** a heat-map table with rows = the 9 proxies (grouped and separated by thin rules) and columns = the 4 splits.
  - Use a single-hue sequential ramp from 0 to 0.6. Do not use a diverging palette, because the quantity has no midpoint.
  - Print each value as text in the cell, using text ink, not the series color.
  - Draw a dashed outline around the static block. Add a side label "chance ≈ random row".
- **Title:** "E6: tested static proxies did not reliably select".

## Panel B — E6 selection regret: held-out states vs threshold task

- **Data:** `...reports.held_out_states.metrics.<proxy>.selection_regret` and `...reports.held_out_task.metrics.<proxy>.selection_regret`. Use `regret_bootstrap95` for whiskers, if present.
- **Plot:** grouped horizontal bars per proxy, with two bars ("held-out states" and "held-out threshold task") and a 2px gap between them.
  - Draw a vertical reference line at the random proxy's regret for each split, in a matching light style.
  - Directly label the short-tail bars on the threshold task with "0.044" and "0.032", and the random line with "random 0.019".
- **Title:** "Short tails fail on the tested threshold task".

## Panel C — E9: warm-up minus cold (mean over 5 splits)

- **Data:** `groups[id=E9].data.summary`. For each entry, read `proxy`, `W` and `metrics.selection_regret.mean_warm_minus_cold`, plus `metrics.wrong_sign_rate.mean_warm_minus_cold`.
- **Plot:** two side-by-side dot plots sharing the y-axis.
  - Rows: input moment, mean grad, Δθ, Δθ+m,v, sign, short tail 1, short tail 4, random.
  - Left x: Δ regret. Right x: Δ wrong-sign.
  - Markers for W=5 (circle) and W=20 (square), with a vertical zero line.
  - Shade the "improvement" side (negative x) very lightly.
- **Annotations:**
  - "random regret falls 4/5 splits ⇒ providers closer after warm-up";
  - "0/8 static proxy/W pairs improve all three metrics". Count this from `groups[id=E9].data.cheap_proxy_results[*].all_three_improve`.
- **Title:** "E9: warm-up compresses differences, static proxies stay non-predictive".

**Caption.** Controlled-learner evidence for CED’s selection rules. (A) Wrong-sign rate of frozen proxies on held-out learner states, warmth shift, ε and learning-rate shifts (E6); static update statistics are at the random level, measured short-tail responses are near zero. (B) Selection regret: short tails select near-optimally on held-out states but worse than random on the held-out threshold task; only the full tail is exact everywhere. (C) Effect of an lr = 0 moment warm-up (E9): the spread between providers shrinks (random regret falls) while static proxies do not improve all metrics. Exact finite-history computations in a controlled small network; not a real-model result.
