# Figure spec — fig2_2 (output `fig2_2_queries`)

**Layout.** Full-width rows, 10 inches wide, at most two subpanels per row. Base/labels 14 pt, titles 15 pt, all other text at least 12 pt.

**Message.** No strategy beats the query lower bound (8). An optimal SPRT sits essentially on it, the sufficient fixed $`n`$ of (9) lies above, and a passive learner is stuck at regret $`\Delta`$.

**Data.** `theory/independent_checks/results_independent_v014.json` → the entry of `claims` with `id == "C2"` → `details.configurations`. Each row has `Delta`, `delta` (rational strings), `bound8`, `min_fixed_n`, `sprt_expected_n`, `n9`, `fixed_ratio` and `sprt_ratio`.

**Panel (a): log–log plot of the number of queries against $`1/\Delta^2`$, for $`\delta=0.05`$.**

- The x values are $`1/\Delta^2`$, i.e. 64, 100, 256, 1024, 4096.
- Four series:
  - bound (8) (`bound8`), black solid;
  - SPRT expected queries (`sprt_expected_n`), blue markers and line;
  - smallest successful fixed-$`n`$ test (`min_fixed_n`), orange markers;
  - sufficient $`n_9`$ (`n9`), grey dashed.
- Inset text: "δ = 0.05".

**Panel (b): ratios to the bound for all 25 configurations.**

- Scatter of `sprt_ratio` (blue circles) and `fixed_ratio` (orange squares) against configuration index. Group the index by $`\delta`$, with vertical separators labelled δ=0.01 … 0.4, and order by $`\Delta`$ within each group.
- Horizontal line at 1, labelled "lower bound (8)".
- Highlight the points with `sprt_ratio == 1.0`, labelled "SPRT attains the bound".
- Y axis: "queries / bound (8)", log scale from 0.9 to 8.

**Panel (c): passive impossibility for $`\Delta=1/8`$.**

- Plot the two regrets $`2\Delta(1-m)`$ (world $`+1`$) and $`2\Delta m`$ (world $`-1`$) against $`m=\mathbb E\widehat p\in[0,1]`$, and their maximum as a thick line.
- Horizontal dashed line at $`\Delta`$, with the minimum at $`m=\tfrac12`$ marked.
- Axes: "mean output E p̂", "expected regret".

**Caption.** (a)–(b) Exact query counts: the SPRT meets the KL lower bound (8), fixed-sample tests and the sufficient n of (9) lie above it, and no strategy falls below. (c) Any passive learner has worst-case regret at least Δ.
