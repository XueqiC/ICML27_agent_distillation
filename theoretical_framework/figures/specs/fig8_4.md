# Figure spec — fig8_4 · Measuring the first-order statistic at the actual states

**Output:** `theoretical_framework/figures/fig8_4_first_order_audit.pdf` (vector) and `fig8_4_first_order_audit.png` (300 dpi). Layout: full-width rows, at most two subpanels per row, 10 inches wide. English labels. Drawn with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)) by `figures/make_figures.py`.

**Message:** E12 uses the oracle continuation gradient. The mean increment can be estimated cheaply and at fixed norm bound b (which can grow with dimension). Where the first-order margin exceeds the remainder, the oracle first-order score never ranks wrongly, while static proxies, which lack $`\nabla V`$, are still wrong. Outside the margin, measured tails are more reliable here but retain nonzero error.

All data come from `theory/v015_checks/results_v015_independent.json`.

## Panel A — "Concentration at fixed b; b can grow with dimension"

- **Data:** `groups[id=V9].data.rows[*]`, fields `p`, `law`, `n`, `delta`, `sample`, `quantile_over_bound`.
- **Plot:** for each law (`sign`, `clipped_Gaussian`, `sparse_spikes`; fixed colors), plot the maximum of `quantile_over_bound` over `n`, `delta` and `sample` against `p` on a log x-axis (1 to 1e5). Draw a horizontal line at 1 labelled "concentration bound" (the bound of Proposition 8.4(a), whose deviation part is McDiarmid's inequality: [McDiarmid, 1989](#ref-mcdiarmid1989method)).
- **Annotation:** "worst failure rate: " + max `failure_rate` at δ=0.05 and at δ=0.2. Report `max_quantile_over_bound` (0.615).
- **Title:** "Concentration at fixed b; b can grow with dimension".

## Panel B — "Wrong-sign rate by first-order margin"

- **Data:** `groups[id=E12].data.bins[*]`, fields `low`, `high`, `n`, `oracle`, `proxies.mean_gradient`, `proxies.delta_plus_m_v`, `proxies.short_tail_4`, `proxies.random`.
- **Plot:** grouped vertical bars per bin, in bin order. The bars are oracle, mean gradient, Δθ+m,v, short tail 4 and random, with fixed colors and a 2px gap between bars. Label each bin "[low, high)" with n underneath.
- Draw a vertical dashed separator between the bins below and above ratio 1, labelled "margin > remainder bound". Print "0" in text ink above zero-height oracle bars.
- **Title:** "Wrong-sign rate by first-order margin".

## Panel C — "Inside vs outside the margin"

- **Data:** `groups[id=E12].data.summary` entry with `split=="overall"`:
  - `oracle_wrong_sign_margin`;
  - `oracle_wrong_sign_eligible`;
  - `proxies.<name>.wrong_sign_margin` and `proxies.<name>.wrong_sign_outside_margin`, for `mean_gradient`, `delta_plus_m_v`, `short_tail_4` and `random`.
  - The outside-margin oracle rate is not stored. Compute it from `bins`: the row-weighted mean of `oracle` over bins with `high <= 1`. As a reproduction check, its combination with the inside value must match `oracle_wrong_sign_eligible` within 0.02 under the same weighting; otherwise report the discrepancy and plot only the stored fields.
- **Plot:** a dot plot with one row per method (oracle, mean gradient, Δθ+m,v, short tail 4, random) and two markers per row: inside the margin (filled) and outside (open).
- **Annotation:** "margin rows: 56 of 396 eligible", from `margin_rows` and `eligible_rows`.
- **Title:** "Inside vs outside the margin". The footer states that E12 uses the oracle continuation gradient.

**Caption.** The bound constant b can grow with dimension. E12 uses the oracle continuation gradient. (A) The empirical deviation of the mean increment stays below 0.62 of the concentration bound at every tested dimension from 1 to 100,000. (B) On the frozen E6 panels, the oracle has no wrong signs in bins at or above one quarter of the remainder bound; this empirical observation is distinct from the C6 margin > bound condition. Static proxies remain wrong 42% of the time in the largest-margin bin. (C) Inside the margin first order decides; outside, measured short tails are more reliable here but retain nonzero error. The outside-margin oracle rate is reconstructed with the specified bin-row weighting. Controlled small network; not a real-model result.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../../references.md).

<a id="ref-hunter2007matplotlib"></a>

**John D. Hunter (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering* 9, no. 3, 90–95. DOI [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

<a id="ref-mcdiarmid1989method"></a>

**Colin McDiarmid (1989).** On the method of bounded differences. In *Surveys in Combinatorics, 1989*, London Mathematical Society Lecture Note Series 141, 148–188, Cambridge University Press. DOI [10.1017/CBO9781107359949.008](https://doi.org/10.1017/CBO9781107359949.008).
