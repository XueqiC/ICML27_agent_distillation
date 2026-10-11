# v0.16 independent numerical verification — specification

Target: `docs/direction/2026-10-11_v016_revision_ZH.md`, §§2–4 and §7.1 C1–C4.
The explicit request also includes Proposition 4.3(b), otherwise listed under C5.
House style: `theory/v015_checks/SPEC.md`. No existing implementation or result
is imported. All computations below are independently derived from the draft.

## Ground rules and deliverables

- Deliverables: `SPEC.md`, `check_v016.py`, `test_check_v016.py`,
  `results_v016.json`, and `results_v016_raw.json.gz`. No other file changes,
  commits, model checkpoints,
  network, teacher calls, torch, or new dependencies. NumPy 2.3.5 and SciPy
  1.17.0 in `theory/.venv` perform all computation; tests use installed pytest.
- Float64, CPU, single BLAS thread. All invocations set
  `CUDA_VISIBLE_DEVICES=''`; the script also enforces it before numerical imports.
  Use `PYTHONDONTWRITEBYTECODE=1` and disable pytest's cache provider to avoid
  extra repository files. All randomness uses
  `numpy.SeedSequence([160160, group, subcase, ...])` with PCG64.
- FAIL is a result, never an excuse to drop a case or change a tolerance.
  Every assertion is counted; every failure records its residual, threshold,
  and the data-row index. Execution exceptions fail their group and subsequent
  groups continue. The CLI exits 1 if any check fails and still writes JSON.
  Unit tests validate implementation, not the truth of every draft claim.
- Synthetic demonstrations, gradient samples, and optimizer rollouts only.
  These do not establish findings on actual agent distillation. No real task,
  seed trajectory, teacher model, or student revision is involved.
- Full runtime target: under 30 minutes on this machine. `--quick` reduces
  sample counts, horizons, and grid sizes for implementation tests; it does
  not change tolerances or hide failures. Quick data are not scientific evidence.

Commands from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks/test_check_v016.py
```

The optional `--only C1,C2,C3,C4` or `--only C4.costate,C4.amplitude` selects groups.
`--output` selects a result path and its sibling `<stem>_raw.json.gz`.
The default full output is `results_v016.json`;
the default quick output is `/tmp/v016_quick_results.json`, so a test cannot
replace the full scientific results.

## JSON contract

Top level: `schema_version`, `status`, `checks`, `runtime_seconds`,
`group_runtime_seconds`, `config`, `versions`, `source_sha256`, `timestamp_utc`,
`scope`, `raw_artifact`. Hashes cover the draft and the three source/specification files.
The configuration records the master seed, mode, selection, device, and lack
of network/model use; it never serializes the process environment.

Every check contains:

- `claim_id`, `asserted`: stable identifier and explicit assertion.
- `n_instances`: number of data records, unchanged by the raw/summary split.
  Every elementwise comparison still counts in `n_assertions`.
- `n_assertions`, `assertions`: per-assertion count, worst residual, tolerance.
- `worst_residual`: maximum nonnegative raw residual across assertion families;
  families can have different units, so inspect their named residuals as well.
- `status`: PASS or FAIL, exactly consistent with `failures`.
- `failures`: all failing cases, with data indices and numeric residuals.
- `data`: settings, summaries, confidence intervals, slopes, C4 maps, and
  `raw_ref` JSON pointers into the decompressed companion.

An inequality residual is LHS minus RHS; a relative error is nonnegative.
No NaN/Infinity JSON tokens are permitted. Infinite mathematical information
bounds are the strings `+infinity`/`-infinity`; unexpected NaN is `NaN` and its
associated assertion fails. No plot files are generated. C4 maps remain in the main file; larger C1
plotting grids and costate inputs are in the companion. Each confidence
interval states its level and replicate count.
Except where specified, MC intervals are pointwise, not simultaneous.


The main file is compact, strict JSON and must be **less than 1,000,000
bytes**; the writer rejects an oversized summary. The gzip companion has
schema `v016-checks-raw-1` and `checks[claim_id][row_index]` containing every
complete data row. It preserves per-replicate C4 response, Val, ranking and
costate arrays, warm m/v, and stochastic limit samples; detailed C1 grids;
C2 feature/design inputs; and C4.costate loss/state/direction arrays and all
finite-difference attempts. No assertion, tolerance, RNG stream, numerical
precision, CI, or slope is changed by serialization. The other checks'
computations and assertions are unchanged.

Main-file costate rows keep derivative projections, errors, selected step
sizes, and counts of attempted/rejected branches; their full ladders remain
in raw rows. Main-file C1 rows keep grid ranges/counts and numerical error or
bound summaries. Each main row's `raw_ref` (for example,
`/checks/C4.amplitude/0`) resolves from the companion's root.

`raw_artifact` records the companion filename, byte count, SHA-256, schema,
and `gzip_mtime=0`. Gzip also omits the original filename. The raw artifact
contains no runtime or timestamp, so identical numerical data produce
identical compressed bytes. Main provenance retains runtime and timestamp.
The raw companion has no 1 MB limit. The CLI writes both outputs even for
scientific FAIL results; failures must remain visible in the main file.

## C1 — Proposition 2.5: warmth and corrected W3

**Motivation.** Independently check the raw second-moment recursion and its
nonzero plateau, and expose failure of the old purely reciprocal hypothesis.

**Setting.** Beta2 is 0.9, 0.99, 0.995, 0.999, 0.9999; constant gradient 1.7;
raw v0=0. Evaluate every s=1..2000 plus 180 log-spaced integer ages to 1e6.

**Procedure and assertions.**

- `C1.exact`: execute all one million updates per beta with a causal IIR
  filter. Compare `(1-beta)*g²/(beta*v_s)` with the stable `expm1` closed form,
  relative tolerance 1e-10. Check both stated bounds at every sampled age,
  allowing 2e-13 relative floating-point error. Record `max_upper_over_kbar`.
  Compare ratios before/after common bias correction with exact rational
  arithmetic, at t=1,7,50; required residual exactly zero.
- Include a nonstationary diagnostic with second moments `exp(-s/400)` and
  beta=.999. Report the exact convolution versus the incorrectly substituted
  stationary expression. It is outside the stationary assertion.
- `C1.moments`: 16,000 independent replicates (quick: 1,500), s up to 2,000,
  checkpoints 1,10,100,1000,2000. Gradients: unit Gaussian; t3/sqrt(3); and
  Bernoulli(.05) times Gaussian/sqrt(.05). Common draws across betas are
  deliberate; replicate paths are independent. For Gaussian/sparse laws,
  use asymptotic Studentized 99.9% intervals.
- **Heavy-tail qualification:** squared t3 gradients have infinite variance.
  A Studentized interval for their raw mean is not justified. Cap g² at 100,
  calculate its first two moments by quadrature, and use a bounded-variable
  Bernstein interval for each replicate's weighted sum. Add the analytically
  known tail expectation to its center. Record the untruncated MC mean too.
  The assertion concerns the corrected 99.9% expectation CI; it makes no
  finite-fourth-moment assumption. Its analytic tail correction is explicit,
  rather than an undisclosed claim of distribution-free estimation.
- `C1.accumulation`: W=1,4,32; c=1,2,4; every R=1..100 plus a log grid to
  1e5. Sum the capped envelope itself, which dominates every admissible kappa
  sequence. Verify the requested logarithmic-plus-linear upper bound with
  1e-9 absolute tolerance. For c=1,1/beta,2, search the first s where the
  literal requested OLD condition `kbar(s)<=c/(W+s)` fails. The age is s,
  not W+s; this distinction matters for failures already at s=1. Separately
  record the natural actual-burn-in interpretation `kbar(W+s)<=c/(W+s)`,
  which exposes later plateau-driven failures when c=2.

**JSON keys.** `s`, `raw_v`, `observed`, `kbar`, `lower`, `upper`, `plateau`,
`max_upper_over_kbar`; `family`, `target`, `method`, `mean`, `se`, `ci_low`,
`ci_high`, tail correction; `R`, `lhs`, `rhs`, `first_failure_s`,
`kbar_at_failure`, `old_rhs`. These suffice for warmth/CI/cumulative plots.

## C2 — Theorems 3.3/3.5, Lemma 3.4, Corollary 3.7

**Motivation.** Check the information-norm equality with independent convex
programs and test attainable absolute risk under actual integer query counts.

**Setting.** Gaussian linear observations, p=2,3,5,8, |X|=4,16;
K=1,10,100; n=0,10,100. For each dimensional pair use three random families:
full-rank demonstrations, an exact null column with nonzero c component,
and an exact null column with c in the identifiable range. All features and
functional coefficients are saved. Quick mode keeps p=2,3 and |X|=4.

**Procedure and assertions.**

- `C2.design`: minimize `c' M(w)^-1 c` over the simplex by analytic-gradient
  SLSQP. Independently minimize `K h' I_D h+n t²`, subject to `c'h=1` and
  `t>=+/-f_x'h`. This linear-epigraph form is equivalent to the requested
  squared epigraph and is a convex quadratic program. Both use the uniform
  design only to condition coordinates, not each other's solutions. No
  ridge, hidden pseudo-observations, or perturbation changes the objective.
- Assert primal/dual relative gap <=1e-6; relative primal simplex KKT gap
  `(w'grad-min(grad))/objective` <=1e-6; dual feasibility <=1e-8; solver
  success. An unsuccessful SLSQP dual solve triggers an independent
  trust-constr solve with exact quadratic Hessian and an active-face KKT
  linear solve. The latter must certify stationarity, feasibility, and
  nonnegative inequality multipliers to 1e-9 before it is accepted. Every
  attempt's status, iteration count, objective, and certificate remain recorded. The preliminary full run
  had eight SLSQP status failures despite accurate gaps; they are retained
  under `dual_attempts`, rather than reclassified as successful. For n=0 use spectral range tests and an
  independent dual direction. Nonidentifiable c gives infinity on both
  sides; identifiable singular c uses the generalized inverse.
- `C2.risk`: largest-remainder integer rounding, **without changing n**.
  Generate independent Gaussian demonstration and query sufficient-statistic
  noises and compute the efficient linear functional estimator; 30,000
  replicates (quick: 3,000). This simulates observations, rather than sampling
  a scalar error from its purported variance. Record the exact Gaussian MAE
  `sqrt(2/pi)*sqrt(c'M_integer^-1 c)` and the variance rounding loss.
- Assert MC MAE lies in `[frak_n/4,frak_n]`, and check the exact Gaussian
  upper bound separately. A finite-n rounding violation is a recorded FAIL,
  even though the theorem is asymptotic. Check MC vs exact MAE at pointwise
  99.9% confidence. Evaluate the least favourable normalized displacement,
  both the theorem's Pinsker/KL lower bound and the stronger exact Gaussian
  total-variation two-point bound. Neither may exceed achieved risk.
- `C2.aliasing`: the null direction is exactly a zero demonstration column;
  A_D h is zero without numerical tolerance. At K=1,10,100,1e6, data means
  and covariances coincide. Check the deterministic triangle inequality
  behind the universal minimax bound and record its midpoint equality case.

**JSON keys.** `A_D`, `features`, `c`, `K`, `n`, `w`, `h`, `primal`, `dual`,
`kkt_residual`, `dual_constraint_residual`, solver statuses; `counts`,
`frak_n`, `rounding_loss`, `rounded_variance`, `exact_MAE`, `risk_over_frak_n`,
`least_favourable_delta`, `KL_D`, `KL_Q_max`, `pinsker_two_point`,
`exact_TV_two_point`, MC intervals; aliasing arrays and lower bound.

## C3 — Propositions 3.10/3.11, Corollary 3.12(a)

**Motivation.** Measure the optimizer's actual learner weights, including
regime transitions and second-moment influence, and validate query allocation.

**Setting and sensitivity procedure.** Each target state owns a scalar
coordinate, c0=.2, pi=.35, zeta=1, eta=.01. One categorical state is visited
per step. A background state receives the remaining probability; its own
coordinate can be omitted because the coordinate dynamics are independent.
All stored inactive coordinates have zero input gradient but their Adam
moments and parameters still update. Epsilon outside sqrt is 1e-12. SGD,
Adam beta1=.9, and SGD always clipped to .1 share visits and label uniforms.

Full sensitivity settings:

| Regime | beta2 | target visitation profile | burn-in | measured R | replicates |
|---|---:|---|---:|---:|---:|
| frequent | .999 | 7 log-spaced .005–.4 | 20,000 | 30,000 | 384 |
| isolated | .9 | 6 log-spaced .0005–.01 | 2,000 | 60,000 | 384 |
| transition | .999 | 7 log-spaced .0003–.02 | 20,000 | 30,000 | 384 |
| short, bias-corrected | .999 | 6 log-spaced .0005–.01 | 0 | 200 | 20,000 |

Common burn-in at pi precedes branching; perturbation starts only afterward.
Central differences at delta=.02 are primary, delta=.04 provides a paired
finite-difference diagnostic. Simultaneously perturbing target-state laws is
equivalent to coordinatewise perturbation because coordinates do not interact.
Report replicate-level 95% t intervals and 1,000 replicate-bootstrap slope
intervals. Quick caps burn-in at 2,000, R at 1,500, and uses 48 replicates
(800 for short runs), with 250 bootstrap draws.

**Sensitivity assertions.** `C3.sensitivity` checks positive sensitivities and
the expected slope: 1 for SGD/clipped SGD, 1/2 for frequent Adam, and 1 for
isolated/short sparse Adam. A stated target must lie within the bootstrap 95%
interval enlarged by a predeclared absolute slope tolerance .1. The transition
slope is descriptive. Linear-learner means are checked against their exact
formula with a 4.5-SE guard. No universal finite-regime relative-error claim
is imposed on the asymptotic Adam influence function. Instead report:

```text
m1 = c0-pi; m2 = c0²+(1-2c0)*pi
frequent derivative = eta R sqrt(d) [1/sqrt(m2)+(1-2c0)m1/(2 m2^(3/2))]
isolated derivative = 2 eta R d kappa_iso
W = pi(1-pi) (d E theta / d pi)²
```

The first formula includes the second-moment term; its omission is separately
reported. Finite response horizons include a startup transient after the
perturbation. Rare-state visits at d/(1-beta2) near one are not called an
asymptotic frequent regime. Record that ratio and the effective visit count.

**Allocation procedure and assertions.** `C3.allocation` uses 16 independent
profiles (quick: 2), 6–10 states, randomized student/teacher visitation,
Bernoulli laws, arbitrary positive W, 800 teacher visits, and n=4,000 queries.
The plug-in functional is exactly linear in state probabilities with
coefficient `sqrt(W/[pi(1-pi)])`, so its variance formula is exact. Solve
water-filling by a scalar multiplier root; discretize by marginal-gain
exchanges until discrete optimality. Compare uniform, proportional-d,
SGD-weight water-filling (W*d), gradient-norm, and uncertainty allocations.
Independent binomial sufficient statistics model the actual integer visits.
Use 60,000 replicates (quick: 5,000). Assertions:

- budget/KKT residual <=1e-8, continuous and discrete predicted optimality;
- water-filling has the smallest **observed** MAE, without suppressing MC
  ranking reversals;
- variance ratios match observed MSE ratios; square roots of variance ratios
  match observed MAE ratios. Delta-method intervals use familywise 95%
  Bonferroni coverage over both ratios, all profiles, and all five comparisons.
  Raw arm CIs are pointwise 95%. This distinction prevents comparing a
  variance ratio directly with an absolute-error ratio.
- An unseen state with positive W and no teacher/query visits gives infinity;
  record indistinguishable laws and its nonzero minimax floor.

**Scale separation.** `C3.separation`: alpha=.65, gradient support {.2,-.8},
b=1 or 3, clipping threshold .1, eta R=1. Demonstrations observe labels only.
For K=1,10,...,100000 estimate alpha using binomial counts, and evaluate the
clipped functional and a midpoint-scale SGD estimator in both worlds. Assert
exact scale invariance, clipped error below its shrinking RMS bound allowing
4 Monte Carlo SEs, and SGD worst risk at least half the functional separation.
The universal SGD claim follows from identical laws; it is not inferred just
from the selected estimator's risk curve.

**JSON keys.** Sensitivities: settings, visits, means/CIs, slope/CIs,
`finite_difference_shift`, both influence predictions and relative errors,
`without_second_moment_term`, `kappa_iso`, `inferred_W`, `weight_slope`.
Allocations: `d`, `d_T`, `W`, `pi`, `functional_coefficients`, `k`,
`continuous_allocation`, per-arm integer counts, predicted variance and MC
MSE/MAE, ratio intervals. Separation: K, world risks, clipped errors/bounds,
and identical-law witness.

## C4 — Theorem 4.1, Corollary 4.2, Proposition 4.3(b)

**Motivation.** Distinguish the exact derivative identity from mean-field
amplitude approximations and from the actual finite message's rollout effect.
The draft's `Val` is a first-order state-channel inner product; even with
exact costates it need not equal the finite-injection response.

**Costate setting.** `C4.costate` crosses quadratic/logistic losses, p=5,20,
bias correction off/on, epsilon=1e-8/1e-3, decay=0/.03, beta1=0/.9,
beta2=.99/.999, clipping off/on (.15). Tails cycle through 20,60,200 steps.
Initial v coordinates are positive; terminal score is sigmoid(a'theta).
Fixed input arrays and initial states are saved. Quick selects every 13th
configuration at p=5 and an 8-step tail.

**Exact extended recursion.** At absolute step j, define

```text
a_j = 1/(1-beta1^j), b_j = 1/(1-beta2^j)   [both 1 without correction]
m' = beta1 m+(1-beta1)g; v' = beta2 v+(1-beta2)g²
theta' = (1-eta wd)theta - eta a_j m'/(sqrt(b_j v')+eps)
D1 = a_j/(sqrt(b_j v')+eps)
D2 = a_j m' b_j/[2 sqrt(b_j v') (sqrt(b_j v')+eps)²]
u_m = chi'_m-eta D1 chi'_theta
u_v = chi'_v+eta D2 chi'_theta
chi_m = beta1 u_m; chi_v = beta2 u_v
chi_theta = (1-eta wd)chi'_theta
          + J_g'[(1-beta1)u_m+2(1-beta2)g*u_v]
```

Absolute step is held fixed during differentiation. Decoupled decay uses the
unmodified learning rate, not a bias-corrected effective learning rate.
Without clipping, J_g is the loss Hessian. With active global-norm clipping,
`J_g = clip/||g_raw|| (I-g_raw g_raw'/||g_raw||²) H_raw`.
This need not be symmetric; the reverse recursion uses its transpose.
No derivative is asserted at a clipping threshold.

**Costate assertions.** Central differences in independent random theta-only,
m-only, v-only, and joint directions. Preserve a diagnostic at 2e-5 and start
the primary ladder at h=2e-4 and h/2. If either perturbed trajectory changes
any clipping decision relative to the unperturbed trajectory, halve h until
both remain on the same smooth branch. This decision uses only forward
clipping patterns, never agreement with the costate. Assert branch consistency
as well as the derivative tolerance. The assertion compares with the
Richardson combination `[4 D(h/2)-D(h)]/3`. Save every attempted step and
clipping-pattern check. Preserve the original 2e-5 difference and its
relative error: the preliminary run had one 1.061e-6 relative discrepancy
on a 2.425e-6 derivative (absolute discrepancy 2.57e-12). The ladder makes
roundoff/truncation distinguishable from a recursion defect. A subsequent
fixed-ladder trial crossed clipping thresholds in two cases (relative errors
1.506e-5 and 8.389e-5); forward branch checks now identify and record those
rejected finite-difference steps. Relative error denominator
is max(|adjoint|,|FD|,1e-10); tolerance 1e-6. Exactly zero m derivatives when
beta1=0 are included. No automatic differentiation library is used.

**Amplitude setting/procedure.** `C4.amplitude` uses exact 1D raw Adam,
beta1=.9, beta2=.99/.999, epsilon=0, eta=alpha=1 (linear response scales
exactly). Gradients are iid N(rho,1-rho²), rho=.3,.7,.95. Starting at zero,
execute `ceil(12/(1-beta2))` burn-in steps. All 4,096 independent replicates
share their own future gradients across messages; quick uses 128 and only
beta2=.999,rho=.7 for the core grid. Evaluate T=50,200,1000,5000,
lambda=-2,.5,1,2,4,8,16, with injected g=lambda sqrt(v) at each actual warmed
state. The noisy warm state is not silently replaced with its mean.

Add six predeclared criterion cases: beta2=.99, T=20,50,100, and
rho=(.8 or 1.2)*kappa_iso(T)/(T+1). These straddle both the approximate and
exact critical rho, remain in (0,1], and use the same lambda grid, burn-in,
and replicate count. They are independent of the core random streams.
Quick mode retains all six criterion cases.

Compute both measured quantities independently of the mean-field formula:

1. **Exact first-order Val difference:** use stochastic reference-tail
   costates at the pre-message state, with exact message increments. For
   F=-theta, `chi_theta=-1`,
   `chi_m=sum_{j=1}^T beta1^j/sqrt(v_j)`,
   `chi_v=-sum_{j=1}^T beta2^j m_j/(2 v_j^(3/2))`.
2. **Exact finite rollout response difference:** inject the message (including
   the zero comparator), then execute all T common-random-number tail steps.

The reference costate tail starts at the pre-message state. Val is retained
as a separate first-order diagnostic; part (a) predicts the finite response.

**Corollary 4.2(a): exact mean-field formula.** Including the message step
j=0, set `b_j=(1-beta1)*beta1^j`, `c_j=(1-beta2)*beta2^j` and evaluate

```text
Delta F(lambda) = alpha*eta*sum_{j=0}^T [
    (rho+b_j*(lambda-rho))/sqrt(1+c_j*(lambda²-1))
    - rho*(1-b_j)/sqrt(1-c_j)]
```

For every lambda on the entire grid, assert
`abs(mean_rollout(lambda)-Delta F(lambda))/max_grid(abs(mean_rollout)) <= .05`.
The denominator is shared by all grid points within a setting (floor 1e-14
only for numerical zero). Save the pointwise errors and their maximum.
No CI allowance modifies the tolerance. Report 95% rollout and Val CIs.

For positive nonzero roots, start at 2 lambda-star and double the upper
endpoint up to 17 evaluations. The lower endpoint is 1e-6, excluding the
trivial root at zero. Use Brent's method once a positive-to-negative bracket
is found. Core measured roots use the first 512 of the same independent CRN
paths (all paths if fewer). The six short-horizon criterion cases use all
4,096 existing paths, since roots near the criterion are sensitive to noise.
Preserve their first-512 estimates and SEs as diagnostics: an initial subset
run gave 18.5% and 20.6% crossing errors at T=20 and T=100, respectively,
on the high-rho side. This precision change does not generate new samples,
change tolerances, or remove settings. Quick and boundary settings use all
their existing paths. Report root MC standard errors by implicit
differentiation. Save the zero-message m/v tail during execution; at step j,
add `.1*beta1^j*lambda*sqrt(v_initial)` to m and
`(1-beta2)*beta2^j*lambda²*v_initial` to v. This is an exact identity for the
executed exogenous-gradient branches, independent of the mean-field formula.

If the measured crossing is bracketed, assert
`abs(mean_field_root/measured_rollout_root-1) <= .10`. The formula uses its
own doubling bracket, so a small root difference cannot be mistaken for
nonexistence. If the measured crossing is unbracketed, the formula must also
have no crossing on its search **up to exactly the same bound**. Assert both
non-bracketing and equality of the bounds. Retain every search endpoint,
value, and status. Unbracketed roots/errors are JSON null; do not label a
bounded empirical search a global proof of nonexistence.

**Corollary 4.2(b): local quadratic law.** Retain
`predicted=(1-beta1^(T+1))*lambda-.5*rho*(1-beta2^(T+1))*lambda²`, the measured
Val and rollout errors, and their CIs. Assert each relative error <=5% only
where `(1-beta2)*lambda²<=.02` and `abs(lambda)<=4`. These relative errors use
`max(abs(predicted),1e-14)` as before. Ineligible points remain in the maps.

Find the exact mean Val root independently, with up to 21 doubling
evaluations, as in the original check. Define
`lambda_star=(1-beta1^(T+1))/(rho*(1-beta2^(T+1)))`. Assert both measured Val
and measured rollout crossings are within 10% of `2*lambda_star` **only**
when `(1-beta2)*(2*lambda_star)²<=.1`. Both warmth boundaries allow 1e-15
floating-point slack so mathematical equality is included. Preserve the original quadratic crossing
error convention `abs(measured_root/(2*lambda_star)-1)`. Outside this region,
report the errors without asserting them. Save the eligibility flag and
threshold warmth for every setting. Missing roots remain null and fail only
an applicable asserted root comparison, never an unconditional bracket rule.

**Corollary 4.2(c): limit and crossing criterion.** For every amplitude
setting (including the boundary settings below), compute

```text
kappa_iso(T) = sum_j b_j/sqrt(c_j)
A(T) = sum_j (1-b_j)/sqrt(1-c_j)
limit = alpha*eta*(kappa_iso(T)-rho*A(T))
exact critical rho = kappa_iso(T)/A(T)
approximate critical rho = kappa_iso(T)/(T+1)
```

The **exact limit's sign** predicts crossing: assert agreement with the
measured root's bracketing and the exact formula's bracketing in every case.
Also compute the exact asymptotic limit of each saved stochastic branch,
`sum_j[dm_j/sqrt(dv_j)-zero_m_j/sqrt(zero_v_j)]`; record its mean/95% CI and
assert its mean has the same negative/nonnegative classification. Check
convergence of the finite formula at `lambda=1e8/sqrt(c_T)` with tolerance
1e-5 relative to `max(abs(limit),1)`. This choice puts every denominator,
including j=T, in the large-amplitude regime. Also preserve the diagnostic
at `1e6/sqrt(c_T)`; its largest original discrepancy was 1.0153e-5, just
above the final-probe tolerance. The approximate rho threshold
is descriptive and does not substitute for the exact criterion.

Retain the boundary settings beta2=.99999, rho=.05, lambda=20, T=0,1,5,
with the same 12-window burn-in (1.2 million executed steps) and 128
replicates (quick:16). Quadratic errors there are diagnostic only. The new
exact formula, crossing comparison and limit assertions cover them too.

**Ranking procedure/assertions.** `C4.ranking` preserves the original pairs
lambda1=1.25 lambda-star and lambda2=1.75 lambda-star on the 24 core settings
(quick:4). Their sum exceeds 2 lambda-star. Record exact parameter-only score
differences, cosine scores, exact Val and exact rollout pair differences
with 95% CRN intervals. Preserve all original assertions: parameter-only
favors the larger message, cosine ties, and both exact Val and rollout favor
the smaller. Additionally record and strictly assert, at **both endpoints**,

```text
(1-beta1)*beta2 - beta1*rho*(1-beta2)*lambda > 0.
```

This expression decreases with positive lambda, so the endpoint checks also
cover the interval between them. Warmth is reported explicitly; failures
are retained. The six criterion cases do not introduce new ranking pairs.

**Main JSON maps.** Each amplitude row retains `lambdas`, `mean_field`,
`rollout` and `val` means/CIs; `mean_field_rollout_relative_error` and its
maximum; quadratic `predicted`, `relative_error`, `rollout_relative_error`,
`eligible`, `warmth`; formula, rollout and Val crossings, all crossing errors,
`quadratic_crossing_warmth`, `quadratic_crossing_eligible`, all `root_search`
records and root SE; `large_lambda_limit` (including exact/approximate rho
thresholds and stochastic limit CI); and warmed-state summaries. Ranking
rows retain both lambdas, lambda-star, derivative signs, scores and paired
CIs. All C4 error maps can be reconstructed from the main file alone.
