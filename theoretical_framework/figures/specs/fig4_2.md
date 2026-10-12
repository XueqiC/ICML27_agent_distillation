# Figure spec — fig4_2 (affine learners: moment order)

**Output:** `theoretical_framework/figures/fig4_2_moment_order.pdf` (vector) and `fig4_2_moment_order.png` (300 dpi). Size 13 × 4.2 in, three panels. English labels.

**Layout.** Full-width rows, 10 inches wide, at most two subpanels per row. Base/labels 14 pt, titles 15 pt, all other text at least 12 pt.

**Message:** every moment order is necessary, the parity witness keeps a constant gap, and the exact number of statistics is a binomial coefficient.

## Panel (a) — "Every order is necessary" (log scale)

Data:

- `theory/independent_checks/results_independent_v014.json`, claim `id == "C8"`, `details.scalar`: a list of `{N, delta}`, where `delta` is a rational string such as `"-1/18"`. Parse with `fractions.Fraction` and plot $`\lvert\Delta_N\rvert`$ for $`N=1,\dots,14`$.
- `theory/v014_repro/results_v014_rerun.json`, `constant_gap_parity`: a list of `{N, exact_value_gap}`, with `exact_value_gap == "1"` for $`N=1,\dots,8`$.

Plot:

- Semilog-$`y`$. Series 1 is $`\lvert\Delta_N\rvert`$ with markers, labelled "scalar witness (26): lower moments equal".
- Series 2 is the constant 1 at $`N=1,\dots,8`$, with markers, labelled "parity witness (27): all degree < N moments equal".
- Overlay the closed form $`N!/(2^{N-1}N^N)`$ as a thin dashed curve on the same $`N`$ grid, computed in the script, to show the exact match.
- Axes: $`x`$ "order N", $`y`$ "learning-value gap |E_P J − E_Q J|". Title: "(a) Each order is needed".

## Panel (b) — "Exact statistic count = C(k+N, N)"

Data: `results_independent_v014.json`, claim `C9`, `details.grid_ranks`. This is a list of `{k, N, rank}` with 18 entries; compute $`\binom{k+N}{N}`$ in the script. Optionally add the 6 package points from `results_v014_rerun.json → grid_response_dimensions` (`numerical_response_rank`), drawn as hollow markers.

Plot:

- Scatter of $`x=\binom{k+N}N`$ against $`y=`$ rank, on log-log axes, colored by $`k`$ (1–4).
- Identity line $`y=x`$, dashed.
- Annotation: "18/18 equal (exact integer rank)".
- Axes: $`x`$ "C(k+N, N)", $`y`$ "rank of monomial matrix on grid". Title: "(b) Moment count is exact".

## Panel (c) — "Binary barrier"

Data: claim `C9`, `details.binary_barriers`. This is a list of `{k, d, width}` with 32 entries, all widths ≈ 1.

Plot:

- For each $`k=1,\dots,4`$ (one color each), plot the points $`(d,\ \text{width})`$.
- Add one extra point per $`k`$ at $`d=2^k-1`$, $`\text{width}=0`$, drawn with a distinct marker (open circle). Label it "full interface (C9 assertion, PASS)", because the zero values are asserted in C9 but not stored in the JSON.
- Draw the step function $`\text{width}=1`$ for $`d<2^k-1`$ and $`0`$ at $`d=2^k-1`$, as thin lines per $`k`$.
- Use a linear $`x`$-axis from 0 to 15.
- Axes: $`x`$ "number of nonconstant features d", $`y`$ "worst-pair discrepancy". Title: "(c) Fewer than 2^k − 1 features: error 1".

**Caption.** Theorems 11–13 in exact arithmetic. (a) scalar witnesses separate at every order with exponentially small gaps (C8, matching the closed form); parity laws keep gap 1 (package check). (b) the monomial evaluation rank on the grid equals C(k+N, N) in all 18 configurations (C9). (c) on binary messages with arbitrary payoffs, any interface with fewer than 2ᵏ − 1 features leaves worst-case error 1 (C9).
