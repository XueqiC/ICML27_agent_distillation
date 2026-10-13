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

**Sources.** Drawn by `figures/make_figures.py` with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)). The equality in panel (b) is the linear-programming duality of [Theorem 4.2(b)](../../statements/04-02_sufficiency_duality_transfer.md#s4-2) ([Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)); both linear programs behind the stored results were solved by SciPy's `linprog` ([Virtanen et al., 2020](#ref-virtanen2020scipy)) with HiGHS ([Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../../references.md).

<a id="ref-bertsimas1997introduction"></a>

**Dimitris Bertsimas, John N. Tsitsiklis (1997).** Introduction to Linear Optimization. Athena Scientific, Belmont, MA (Athena Scientific Series in Optimization and Neural Computation 6). ISBN 978-1-886529-19-9.

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-hunter2007matplotlib"></a>

**John D. Hunter (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering* 9, no. 3, 90–95. DOI [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
