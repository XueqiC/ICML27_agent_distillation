# Figure spec — fig4_1 (sufficiency duality)

**Output:** `theoretical_framework/figures/fig4_1_duality.pdf` (vector) and `fig4_1_duality.png` (300 dpi). Size 11 × 4.2 in, two panels side by side. English labels.

**Layout.** Full-width rows, 10 inches wide, at most two subpanels per row. Base/labels 14 pt, titles 15 pt, all other text at least 12 pt.

**Message:** the worst feature-matched pair of laws separates a response by *exactly* twice its uniform distance to the feature span.

## Panel (a) — "Thm 8(b) on a 5-point alphabet" (exact, recomputed in the script)

Data, computed exactly with `fractions.Fraction`; no external file is needed:

- Alphabet $`y\in\lbrace 0,\tfrac14,\tfrac12,\tfrac34,1\rbrace`$, feature $`\phi(y)=y`$ (so $`H_\phi=\mathrm{span}\lbrace 1,y\rbrace`$), response $`g(y)=y^2`$.
- Best uniform approximation $`h^*(y)=y-\tfrac18`$, with error $`e=\tfrac18`$.
  - Verify in the script that $`\max_y\lvert g-h^*\rvert=\tfrac18`$.
  - Verify the sign pattern of $`g-h^*`$ at $`y=0,\tfrac12,1`$: $`+\tfrac18,-\tfrac18,+\tfrac18`$.
- Extremal pair: $`P=\tfrac12\delta_0+\tfrac12\delta_1`$, $`Q=\delta_{1/2}`$.
  - Verify $`\mathbb E_P y=\mathbb E_Q y=\tfrac12`$.
  - Verify $`(P-Q)g=\tfrac14=2e`$.

Plot:

- A dense curve of $`g(y)=y^2`$ on $`[0,1]`$ (thin, dark), and the 5 grid points as dots.
- The line $`h^*(y)=y-\tfrac18`$ (second color), with the band $`h^*\pm\tfrac18`$ shaded lightly.
- Stems (lollipops) at the support points, drawn below the axis or as an inset bar strip: $`P`$ masses (0.5 at $`y=0`$ and $`y=1`$) in one color and $`Q`$ mass (1.0 at $`y=\tfrac12`$) in another.
- Annotation box: "$`\mathbb E_P\phi=\mathbb E_Q\phi=1/2`$; $`(P-Q)g=1/4=2\cdot\mathrm{dist}_\infty(g,H_\phi)`$".
- Axes: $`x`$ "message $`y`$", $`y`$ "value". Title: "(a) Worst matched pair attains $`2\cdot`$dist".

## Panel (b) — "Primal LP vs 2 × approximation error" (from committed results)

Data: `theory/independent_checks/results_independent_v014.json`. Use the claim with `id == "C19"`, at `details.families[*].curve[*]`. Each curve entry gives:

- $`x`$ = `2 * uniform_error`
- $`y`$ = `discrepancy`

There are 21 points: 3 families × 7 feature budgets.

Plot:

- A scatter with one marker shape per family (`family` field: "scalar affine powers", "multivariate parity pullbacks", "structured sum powers"). Categorical colors are in fixed order.
- The identity line $`y=x`$ (dashed gray), with axes from $`0`$ to $`0.55`$.
- Annotation: "max |discrepancy − 2·error| = …", computed in the script from the same entries (expected $`1.6\times10^{-15}`$).
- Second annotation from `claims[id=C5].details.max_duality_residual`: "C5: 300 random instances, max residual 2.2e-14".
- Axes: $`x`$ "2 × best uniform approximation error (LP)", $`y`$ "worst-pair discrepancy (primal LP)". Title: "(b) Equality (20) across tested subspaces".

**Caption.** Thm 8(b): for feature-matched laws, the largest response gap equals twice the uniform distance of the response to the feature span. (a) exact 5-point example; (b) independent primal/dual LPs on all C19 subspaces (and C5’s 300 random instances).
