# Figure spec — fig2_2 (output `fig2_2_queries`)

**Layout.** Full-width rows, 10 inches wide, at most two subpanels per row. Base/labels 14 pt, titles 15 pt, all other text at least 12 pt.

**Rendering.** Drawn with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)) by `figures/make_figures.py`.

**Message.** No strategy beats the query lower bound (2.6) of Theorem 2.6(b), a change-of-measure bound ([Kaufmann et al., 2016](#ref-kaufmann2016complexity)). An optimal SPRT ([Wald, 1945](#ref-wald1945sequential)) sits essentially on it, the sufficient fixed $`n`$ of Theorem 2.6(c) lies above, and a passive learner is stuck at regret $`\Delta`$.

**Data.** `theory/independent_checks/results_independent_v014.json` → the entry of `claims` with `id == "C2"` → `details.configurations`. Each row has `Delta`, `delta` (rational strings), `bound8`, `min_fixed_n`, `sprt_expected_n`, `n9`, `fixed_ratio` and `sprt_ratio`.

**Panel (a): log–log plot of the number of queries against $`1/\Delta^2`$, for $`\delta=0.05`$.**

- The x values are $`1/\Delta^2`$, i.e. 64, 100, 256, 1024, 4096.
- Four series:
  - bound (2.6) (`bound8`), black solid;
  - SPRT expected queries (`sprt_expected_n`), blue markers and line;
  - smallest successful fixed-$`n`$ test (`min_fixed_n`), orange markers;
  - sufficient $`n_9`$ (`n9`), grey dashed.
- Inset text: "δ = 0.05".

**Panel (b): ratios to the bound for all 25 configurations.**

- Scatter of `sprt_ratio` (blue circles) and `fixed_ratio` (orange squares) against configuration index. Group the index by $`\delta`$, with vertical separators labelled δ=0.01 … 0.4, and order by $`\Delta`$ within each group.
- Horizontal line at 1, labelled "lower bound (2.6)".
- Highlight the points with `sprt_ratio == 1.0`, labelled "SPRT attains the bound".
- Y axis: "queries / bound (2.6)", log scale from 0.9 to 8.

**Panel (c): passive impossibility for $`\Delta=1/8`$.**

- Plot the two regrets $`2\Delta(1-m)`$ (world $`+1`$) and $`2\Delta m`$ (world $`-1`$) against $`m=\mathbb E\widehat p\in[0,1]`$, and their maximum as a thick line.
- Horizontal dashed line at $`\Delta`$, with the minimum at $`m=\tfrac12`$ marked.
- Axes: "mean output E p̂", "expected regret".

**Caption.** (a)–(b) Exact query counts: the SPRT meets the KL lower bound (2.6) of Theorem 2.6(b), fixed-sample tests and the sufficient n of Theorem 2.6(c) lie above it, and no strategy falls below. (c) Any passive learner has worst-case regret at least Δ.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../../references.md).

<a id="ref-hunter2007matplotlib"></a>

**John D. Hunter (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering* 9, no. 3, 90–95. DOI [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

<a id="ref-kaufmann2016complexity"></a>

**Emilie Kaufmann, Olivier Cappé, Aurélien Garivier (2016).** On the complexity of best-arm identification in multi-armed bandit models. *Journal of Machine Learning Research* 17, no. 1, 1–42. [jmlr.org/papers/v17/kaufman16a](https://jmlr.org/papers/v17/kaufman16a.html).

<a id="ref-wald1945sequential"></a>

**Abraham Wald (1945).** Sequential tests of statistical hypotheses. *The Annals of Mathematical Statistics* 16, no. 2, 117–186. DOI [10.1214/aoms/1177731118](https://doi.org/10.1214/aoms/1177731118).
