# Figure spec — fig8_3 · Actual-state adaptive auditing

**Output:** `theoretical_framework/figures/fig8_3_adaptive_audit.pdf` (vector) and `fig8_3_adaptive_audit.png` (300 dpi). Layout: full-width rows, at most two subpanels per row, 10 inches wide. English labels. Drawn with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)) by `figures/make_figures.py`.

**Message:** the measured adaptive-audit gain is above both guarantees in every configuration. The cost of the guarantee is large. In the controlled network, deployment noise dominates the paired variance.

## Panel A — C16: measured gain vs the two lower bounds

- **Data:** `theory/independent_checks/results_independent_v014.json`, the `claims` entry with `id == "C16"`, field `details.configurations`. There are 5 rows with fields `R`, `M`, `n`, `delta`, `gain`, `SE` and `expected_sum_B`.
- **Plot:** x = configuration index, with tick labels "R=1,M=2,n=8,δ=.4" and so on.
  - Gain as a point with a ±2·SE error bar.
  - Bound (44), y = −δ: short horizontal dashes.
  - Bound (45), y = expected_sum_B − 2δ: short horizontal dashes in a second style.
  - A thin y = 0 line. Use a linear y-axis; the range covers −0.85 to 0.2, so the bounds are visible.
  - The last configuration (n=1000) gets a direct label "selects the good candidate: +0.136".
- **Title:** "C16: gain ≥ −δ and ≥ E ΣB − 2δ (100k runs each)".

## Panel B — Replicas per candidate vs radius

- **Data (analytic plus committed):**
  - Hoeffding curve ([Hoeffding, 1963](#ref-hoeffding1963probability)) n(e) = 2·log(2M/δ)/e², with M=4, δ=0.05, for e ∈ [0.05, 0.3].
  - Empirical-Bernstein curve ([Maurer & Pontil, 2009](#ref-maurer2009empirical)): solve for the smallest n with sqrt(2·S²·log(4M/δ)/n) + 14·log(4M/δ)/(3·(n−1)) ≤ e. Take S² = 0.12925 from `groups[id=E7].data.empirical_Bernstein.max_pilot_sample_variance` in `theory/v015_checks/results_v015_independent.json`; take the formula string from `...empirical_Bernstein.formula`.
- **Plot:** log-y n against e. Mark the points (0.1, 1016) as "Hoeffding (sufficient)" and (0.1, 559) as "empirical Bernstein (forecast from pilot variance)". Use a dashed line for the forecast curve.
- **Title:** "Audit cost at one state, M = 4, δ = 0.05".

## Panel C — E7 variance decomposition

- **Data:** `groups[id=E7].data.variance_decomposition`. There are 4 rows with `candidate`, `block`, `suffix`, `deployment` and `total`.
- **Plot:** stacked horizontal bars per candidate (block, suffix, deployment) with a 2px gap between segments. Annotate the deployment share as a percentage at the bar end (84%, 81%, 89%, 77%).
- **Title:** "Paired-contrast variance: deployment dominates (controlled network)".

**Caption.** (A) Adaptive actual-state auditing on a finite process (C16): the measured gain (±2 SE, 100,000 trainings per configuration) never falls below −δ or E ΣB − 2δ. (B) Replicas per candidate needed for radius e. Hoeffding’s 1016 at e = 0.1 is sufficient; the empirical-Bernstein curve is a forecast from pilot variance. (C) Exact variance decomposition of paired contrasts in the controlled network (E7). Finite computations and a controlled small network; not a real-model result.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../../references.md).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-hunter2007matplotlib"></a>

**John D. Hunter (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering* 9, no. 3, 90–95. DOI [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).
