# Figure spec — fig2_3 (output `fig2_3_floor`)

**Layout.** Full-width rows, 10 inches wide, at most two subpanels per row. Base/labels 14 pt, titles 15 pt, all other text at least 12 pt.

**Rendering.** Drawn with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)) by `figures/make_figures.py`.

**Message.** The information floor decays as $`(1-q)^K`$. Targeted queries buy far more than extra passive demonstrations, and replay buys nothing. A real learner cannot beat the floor. Demonstration KL and query KL add to cover $`\mathrm{kl}(1-\delta,\delta)`$.

**Data.**

- `theory/v015_checks/results_v015_independent.json`: the entry of `groups` with `id == "E5"`, keys `data.curves` and `data.actual_learners`.
- `docs/direction/v015_revision/results_v015.json`: `checks.T11_adaptive_query_KL_enumeration.rows`.
- Rational strings must be parsed with `fractions.Fraction`.

**Panel (a): information risk against $`K`$, log y axis.**

- Filter: `m == 8`, `q_profile == "uniform"`, `w_profile == "uniform"`, $`K=0,\dots,32`$.
- Series:
  - `active` with `queries == 4`: blue solid, labelled "+4 targeted queries";
  - `passive` with `queries == 4`: orange dashed, labelled "+4 extra demonstrations";
  - `replay` with `queries == 4`: grey dotted, labelled "+4 replays (gain 0)", which coincides with the next series;
  - `active` with `queries == 0`: black solid, labelled "K demonstrations only, ½(7/8)^K".
- Use a clipping floor of $`10^{-6}`$ for zero values.
- Axes: "K (trusted demonstrations)", "information-limited risk".

**Panel (b): actual learner risk against the information floor.**

- Scatter of all `data.actual_learners` rows: x = `information_floor`, y = `risk`.
- Colour by optimizer (Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) blue, SGD orange) and use the marker shape for `access` (active, passive, replay).
- Diagonal $`y=x`$, labelled "information floor".
- Annotation: "no point below the diagonal (min risk − floor = 0)".
- Axes: "information floor", "actual learner risk (12 updates)".

**Panel (c): KL budget, from T11.**

- For $`K=0,\dots,4`$ and direction 0, grouped bars for three quantities:
  - `binary_KL_lower`, i.e. $`\mathrm{kl}(1-\delta,\delta)`$;
  - `transcript_KL`;
  - `KL_upper`, i.e. $`K\kappa_D+\kappa_X\mathbb EN`$.
- The bars must read lower ≤ exact ≤ upper.
- Optionally hatch direction 1 next to them.
- Axes: "K", "nats".

**Caption.** (a) Exact hidden-bit risk (m=8, uniform): targeted queries beat equal numbers of extra demonstrations; replay adds nothing. (b) Adam/SGD learners never go below the information floor. (c) Demonstration plus query KL always covers kl(1−δ, δ) (A15).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../../references.md).

<a id="ref-hunter2007matplotlib"></a>

**John D. Hunter (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering* 9, no. 3, 90–95. DOI [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).
