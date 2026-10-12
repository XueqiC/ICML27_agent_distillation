# Figure spec — warmth (output `fig5_3_warmth`)

**Message.** Stationary mean-field warmth has an early reciprocal phase and a nonzero plateau. Warm-up and real updates both advance the second moment. The stored affine residuals satisfy their bound.

**Layout.** Full-width rows, at most two subpanels per row, 10 inches wide. PDF vector, PNG 300 dpi. Base/labels 14 pt; titles 15 pt; ticks, legends and annotations at least 12 pt (15 px at 900 px width).

**Data.** Summary files are Git HEAD blobs: V = `theory/v015_checks/results_v015_independent.json`; U = `theory/v016_checks/results_v016.json`. Panel B reads the working-tree `theory/v016_checks/results_v016_raw.json.gz` only after verifying byte count, SHA-256 and schema against U:`raw_artifact`. No checks are rerun.

**Panel A — zero-learning-rate warm-up gaps.** V:`groups[id=E8].data.gaps`, selecting `eta=0.1`, `(task,N)=(moment,3),(parity,2)` and `R=1,4,40`. Plot `W` vs `gaps.actual_Adam`, with the signed `gaps.tangent_affine_SGD` comparator at `W=0`. Symlog x. Annotate “W = 0 → W = 10”, reading both values explicitly from the stored moment rows: 0.076907458481 → 0.000447102849 (R=1) and 0.119610951546 → 0.001591703495 (R=4), displayed to four decimals as 0.0769 → 0.0004 and 0.1196 → 0.0016. Keep the full plotted W grid through W=50; the annotation compares W=0 with W=10.

**Panel B — exact warmth and bounds.** U:`checks[claim_id=C1.exact].data` entries with `plateau` and `beta2=0.9,0.99,0.999,0.9999`. Their `raw_ref` pointers resolve to `/checks/C1.exact/0`, `/checks/C1.exact/1`, `/checks/C1.exact/3`, `/checks/C1.exact/4`. Plot raw `s` vs `kbar`; shade `lower`–`upper`; dashed `plateau`. Cross-check against the exact closed form `(1-beta2)/(beta2*(1-beta2**s))`, evaluated with `expm1`. The bounds are `max(1/s,1-beta2)/beta2` and `(1/s+1-beta2)/beta2`. Use log-log axes through s=1e6. Exclude the separately stored nonstationary diagnostic from the stationary selection.

Overlay V:`groups[id=E8].data.W3.per_step` with `task=moment,N=3,source=P,eta=0.01,W=0`: x=`prior_gradient_evaluations`, y=`quantiles[2]`. Omit infinite/nonpositive values, without smoothing. Label “early phase, s(1−β₂) ≪ 1”. This measured median is not an estimate of the stationary mean-field expectation. Related U claims `C1.moments` and `C1.accumulation` provide stochastic and accumulated-bound context; they are not additional plotted series.

**Panel C — warm-Adam affine residual vs bound.** V:`groups[id=E8].data.W1.panels[*].{r,pointwise_residual}`. Log-log scatter with all messages, color by `eta`, marker by `task`; diagonal y=x. Distinguish max pointwise residual/r (0.998487) from law-pair max `groups[id=E8].data.W1.best_ratio` (0.988341), with `n_law_pairs=2496`.

**Caption.** A zero-learning-rate moment warm-up reduces short-horizon gaps; long-horizon residuals remain. Exact stationary mean-field warmth has early-phase 1/(beta2 s) behavior and a nonvanishing plateau (1-beta2)/beta2. Shading gives the two-sided bounds; dashed lines give the plateaus. The measured median is an early-phase diagnostic, not the mean-field expectation. Warm-up and real steps advance v alike. The warm-Adam affine bound holds on the stored residuals; max pointwise residual/bound 0.998487, law-pair max 0.988341. This is empirical consistency, not a uniform neural certificate.
