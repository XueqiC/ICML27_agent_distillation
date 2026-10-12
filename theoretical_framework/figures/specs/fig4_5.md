# Figure spec — fig4_5 (`figures/fig4_5_regime.pdf`, `figures/fig4_5_regime.png`)

**Message.** The local SGD description holds exactly where theory says (small $`\eta R`$) and fails as $`\eta R`$ grows. The degree-composition and accumulation statements are exact.

**Layout.** Full-width rows, at most two subpanels per row, 10 inches wide. PDF vector, PNG 300 dpi. Base/labels 14 pt; titles 15 pt; ticks, legends and annotations at least 12 pt (15 px at 900 px width).

**Panel A — "SGD: actual gap vs tangent-affine prediction".**
- Data: `theory/independent_checks/results_independent_v014.json → controlled_neural.details.configurations`, rows with `optimizer == "SGD"` (84 rows).
- Heatmap with rows $`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$ and columns $`R\in\lbrace 1,4,40\rbrace`$.
- Cell color: $`\log_{10}`$ of the max over the 7 task laws of $`\lvert`$`deviation_from_local_affine`$`\rvert`$.
- Cell text: "k/7 agree" from `local_affine_agreement`.
- Colorbar label: "log10 max |actual − affine| gap deviation".
- Annotate: "agreement margin = 1e-4 + 0.1|pred| + 4 SE".

**Panel B — "Expansion (35): remainder vs η³ bound".**
- Data: `claims[id=C13].details.logistic` (60 rows: `eta`, `remainder`, `bound`).
- Log-log scatter, x = `bound`, y = `remainder`, colored by `eta`.
- Dashed line y = x; text "max remainder/bound = 0.274" (`claims[id=C13].best_ratio`).

**Panel C — "Error accumulation needs stability".**
- Data: inputs from `theory/v015_checks/results_v015_independent.json → groups[id=V7].data.linear_failure.inputs` ($`\eta=1/2`$, $`L=1`$, $`C=1`$). Recompute exactly for $`R=1..12`$ with `fractions`:
  - the equality recursion $`e_{t+1}=(1+\eta L)e_t+C\eta^2`$;
  - the linear line $`RC\eta^2`$.
- Plot both on a linear y-axis; mark $`R=4`$ with "e_4 = 65/32 > 1".
- Inset text: "squaring witness: gaps −1, 1/4, 3/256, 315/16777216 at R=1..4" from `claims[id=C13].details.amplification`.

**Caption.** Left: the tangent affine-SGD prediction matches the actual controlled network for η≤0.1 and short horizons, and fails at η≥1. Middle: the third-order remainder of (35) stays below its bound. Right: per-step errors accumulate geometrically; the linear-in-R reading fails without a stability condition.
