# Figure spec — fig8_1 · Stability–approximation tradeoff

**Output:** `theoretical_framework/figures/fig8_1_stability.pdf` (vector) and `fig8_1_stability.png` (300 dpi). Layout: full-width rows, at most two subpanels per row, 10 inches wide. English labels.

**Message:** the tolerant discrepancy equals the dual formula (36). A perfectly representing but small-scale feature certifies only $`\epsilon/\alpha`$.

## Panel A — C22 stability control

- **Data:** `theory/independent_checks/results_independent_v014.json`, entry of `claims` with `id == "C22"`, field `details.configurations`. There are 28 rows with fields `alpha`, `epsilon` (rational strings such as `"1/1000"`; parse with `fractions.Fraction`), `discrepancy` (the LP value, a float) and `formula` (a rational string).
- **Plot:** x = $`\epsilon`$ on a symlog axis, with linear threshold $`10^{-4}`$ so that $`\epsilon=0`$ is shown. y = $`W_\epsilon(g)`$, linear from 0 to 1.05.
  - One series per $`\alpha\in\lbrace 1,1/2,1/10,1/100\rbrace`$.
  - LP values (`discrepancy`) as markers of size 8 or more.
  - The closed form $`\min\lbrace 1,\epsilon/\alpha\rbrace`$ as a thin line over a dense $`\epsilon`$ grid in $`[0,2]`$.
  - Use categorical colors in fixed order, a legend, and direct labels "α = 1", "α = 1/2", "α = 1/10" and "α = 1/100" at the right end.
- **Annotation:** "markers: LP over law pairs; lines: min{1, ε/α}".
- **Title:** "C22: exact representation, small scale ⇒ weak certificate".

## Panel B — Package explicit example and C14 summary

- **Data:** `theory/v014_repro/results_v014_rerun.json`, key `noisy_feature_duality.explicit_example`. There are 6 rows with `tolerance`, `primal`, `dual`, `closed_form` and `coefficient`.
- **Plot:**
  - x = tolerance, from 0 to 1.6. y = discrepancy.
  - Primal as filled circles, dual as open squares (they overlap), and the closed form $`\min\lbrace 1,1/2+\epsilon\rbrace`$ as a line.
  - Shade the region where the optimal coefficient is 1 (tolerance < 0.5) and label it "uses feature (b=1)". Label the rest "ignores feature (b=0)".
- **Text box (lower right):** "C14: 220 random instances, max primal–dual gap 8.9e−16". Read the number from `claims[id=C14].details.max_duality_residual` and format it with 2 significant digits.
- **Title:** "Primal = dual; the optimizer drops the feature when ε outweighs its gain".

**Caption.** Left: tolerant discrepancy for φ = αg (C22). LP values coincide with min{1, ε/α}. Right: the explicit example Y = {0, ½, 1}, φ(y) = y, g = 1{y = 1}; primal, dual and closed form coincide. Exact finite computations, not neural results.
