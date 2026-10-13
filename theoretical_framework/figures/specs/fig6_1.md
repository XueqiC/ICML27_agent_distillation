# Figure spec — information bound (output `fig6_1_information_bound`)

**Message.** Achieved Gaussian plug-in risk is within the information factors; independent primal/dual design objectives agree; aliasing leaves a fixed floor.

**Layout.** Full-width rows, at most two subpanels per row, 10 inches wide. PDF vector, PNG 300 dpi. Base/labels 14 pt; titles 15 pt; ticks, legends and annotations at least 12 pt (15 px at 900 px width). Drawn by `figures/make_figures.py` with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)).

**Data.** Git HEAD `theory/v016_checks/results_v016.json`; no raw companion or rerun needed.

**Panel A — achieved risk / information norm.** `checks[claim_id=C2.risk].data[*].risk_over_frak_n` against stored setting index. Use the plain-text y-axis label “achieved risk / information norm”, matching the caption's learner-relative information norm. Color by the `p` of `checks[claim_id=C2.design].data[design_case]`. Draw references 1/4 (the two-point lower bound, Le Cam's method: [Le Cam, 1973](#ref-lecam1973convergence)), 1, and sqrt(2/pi). All 192 risk settings are retained. Integer query rounding can move the ratio above the unrounded Gaussian factor.

**Panel B — design duality.** The c-optimal design value ([Elfving, 1952](#ref-elfving1952optimum)) against the dual information objective. `checks[claim_id=C2.design].data[*].{primal,dual,kind}`. Log-log scatter on y=x, grouped by kind. Draw only numeric finite values; count the 24 infinite/infinite settings in the panel annotation (192 finite designs). Never replace infinity with zero.

**Panel C — aliasing.** `checks[claim_id=C2.aliasing].data[*].{model_id,K,minimax_lower_bound}`: one curve per model, K=1,10,100,1e6. The ordinate is |cᵀh|/2; verify it remains flat within every model. All eight models and 32 points retained.

**Caption.** Gaussian plug-in risk divided by the learner-relative information norm lies between 1/4 and 1 in the stored settings. Integer allocation explains departures from sqrt(2/pi). Independent primal and dual designs agree; infinite nonidentifiable cases are counted separately. Aliasing leaves the same worst-error floor for every demonstration count. Synthetic checks, not agent-distillation evidence.
