# Figure spec — fig4_4 (exact order vs approximate complexity)

**Output:** `theoretical_framework/figures/fig4_4_approximation.pdf` (vector) and `fig4_4_approximation.png` (300 dpi). Size 13 × 4.2 in, three panels. English labels.

**Layout.** Full-width rows, 10 inches wide, at most two subpanels per row. Base/labels 14 pt, titles 15 pt, all other text at least 12 pt.

**Message:** $`p^N`$ and parity have the same exact degree $`N`$. A few Chebyshev orders already capture $`p^N`$, while parity stays at error 1/2 until order $`N`$.

## Panel (a) — "p^16 and its low-order Chebyshev truncations" (recomputed exactly)

Data: compute (32) in the script for $`N=16`$. Coefficients are $`a_0=4^{-N}\binom{2N}{N}`$ and $`a_j=2\cdot4^{-N}\binom{2N}{N-j}`$ for $`j\ge1`$. The truncation is $`h_q(p)=\sum_{j\le q}a_jT_j(2p-1)`$; use `numpy.polynomial.chebyshev.chebval`.

Plot:

- $`p^{16}`$ on $`[0,1]`$: a thick dark curve over 2,001 points.
- $`h_q`$ for $`q\in\lbrace 4,6,8\rbrace`$: thinner lines in sequential shades of one hue.
- A horizontal dashed line at $`0.5`$, labelled "best degree-<N approximation of parity is the constant 1/2 (error 1/2)".
- In the legend, give each $`h_q`$ its sup error, read from `theory/v014_repro/results_v014_rerun.json → scalar_approximation` (rows with `N == 16`, `q` in {4, 6, 8}, field `Chebyshev_truncation_error`). The expected values are 0.110, 0.0201 and 0.0021.
- Axes: $`x`$ "p", $`y`$ "value". Title: "(a) p^16 ≈ order-8 polynomial".

## Panel (b) — "Truncation error vs order" (semilog-y)

Data: `results_v014_rerun.json → scalar_approximation`. Rows are `{N, q, Chebyshev_truncation_error, Hoeffding_upper, parity_best_degree_q_error}` for $`N\in\lbrace 4,8,16,32\rbrace`$ and $`q=0,\dots,N`$.

Plot:

- For each $`N`$, one color in sequential order:
  - `Chebyshev_truncation_error` against $`q`$, solid with markers, for $`q<N`$. Drop the $`q=N`$ point, where the error is 0, from the log plot.
  - `Hoeffding_upper` = $`\min(1,2e^{-q^2/N})`$ against $`q`$, dashed, same color.
  - `parity_best_degree_q_error` = 0.5 for $`q<N`$, as a thin gray dotted line, drawn once and labelled "parity (all N)".
- A horizontal reference at $`10^{-3}`$, with markers at the first $`q`$ where each solid curve drops below it. These are computed in the script; the expected values are $`q=4,6,9,13`$.
- Annotate C12: "C12 (N ≤ 16): error = exact binomial tail; max tail/bound = 0.430" (from `claims[id=C12].details.exhaustive_tail_best_ratio` in `theory/independent_checks/results_independent_v014.json`).
- Axes: $`x`$ "approximation order q", $`y`$ "sup-norm error". Title: "(b) Needed order grows like √N".

## Panel (c) — "Same max degree 3, different feature budgets" (C19)

Data: `theory/independent_checks/results_independent_v014.json`, claim `C19`, `details.families[*]`. Use fields `family`, `span_dimension_including_constant`, and `curve[*].feature_budget`, `curve[*].uniform_error`, `curve[*].discrepancy`, `curve[*].optimality`.

Plot:

- `discrepancy` against `feature_budget` ($`d=1,\dots,7`$), one line per family. Use categorical colors in fixed order, matching the fig4_1 figure.
- Points whose `optimality` starts with "upper bound" get hollow markers; exact zeros get filled markers.
- A vertical tick at $`d=\text{span\_dimension}-1`$ for each family (3, 3 and 7), labelled "exact span".
- Linear $`y`$-axis from 0 to 0.55.
- Axes: $`x`$ "number of nonconstant features d", $`y`$ "worst-pair discrepancy". Title: "(c) Parity needs all 2^3 − 1 directions".

**Caption.** Exact degree does not measure approximate information complexity. (a)–(b) Chebyshev truncation of pᴺ has error equal to a binomial tail ≤ 2 exp(−q²/N); error 0.001 needs only q = 4, 6, 9, 13 for N = 4, 8, 16, 32 (package check; C12 for N ≤ 16). Parity keeps error 0.5 below degree N. (c) On the same grid and degree, scalar and structured families are exactly resolved by 3 features, parity only by 7 (C19; hollow markers = upper bounds from the best subspace found). Zero-error endpoint threshold crossings are marked at the threshold, outside the log-error curves.

**Sources.** Drawn by `figures/make_figures.py` with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)). Panel (a) evaluates the Chebyshev truncations ([Rivlin, 1990](#ref-rivlin1990chebyshev)) with NumPy's `chebval` ([Harris et al., 2020](#ref-harris2020array)); the dashed curves of panel (b) are the bound $`2e^{-q^2/N}`$ that Hoeffding's inequality ([Hoeffding, 1963](#ref-hoeffding1963probability)) gives for the binomial tail in [Proposition 4.6(a)](../../statements/04-06_exact_order_not_approximate_complexity.md#s4-6).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-hunter2007matplotlib"></a>

**John D. Hunter (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering* 9, no. 3, 90–95. DOI [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

<a id="ref-rivlin1990chebyshev"></a>

**Theodore J. Rivlin (1990).** Chebyshev Polynomials: From Approximation Theory to Algebra and Number Theory. 2nd edition, Wiley, New York. ISBN 978-0-471-62896-5.
