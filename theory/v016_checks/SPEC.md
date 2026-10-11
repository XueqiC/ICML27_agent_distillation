# v0.16 independent numerical verification — specification

Target: `docs/direction/2026-10-11_v016_revision_ZH.md`, §§2–5 and §7.1 C1–C6b.
C1–C4 retain their independent implementations and numerical logic unchanged.
C5/C6 import the explicitly requested frozen controlled-network helpers from
`theory/v015_checks` read-only; the new implementation lives in `c5c6.py`.
The v0.15 implementation, manifest, E6/E12 records and validation receipt are
hashed before and after. No v0.15 check function or output writer is invoked.

## Ground rules and deliverables

- Deliverables: `SPEC.md`, `check_v016.py`, `test_check_v016.py`, `c5c6.py`, `test_c5c6.py`,
  `c5c6b.py`, `test_c5c6b.py`,
  `results_v016.json`, and `results_v016_raw.json.gz`. No other file changes,
  commits, model checkpoints,
  network, teacher calls, torch, or new dependencies. NumPy 2.3.5 and SciPy
  1.17.0 in `theory/.venv` perform all computation; tests use installed pytest.
  Frozen imports also load the already installed SymPy/Matplotlib dependencies.
  Matplotlib is initialized with a temporary config directory before importing
  v0.15; a Python audit hook rejects all writes under the pinned v0.15 tree.
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
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

The optional `--only C1,C2,C3,C4,C5,C6,C5b,C6b` or `--only C4.costate,C4.amplitude` selects groups.
`--output` selects a result path and its sibling `<stem>_raw.json.gz`.
The default full output is `results_v016.json`;
the default quick output is `/tmp/v016_quick_results.json`, so a test cannot
replace the full scientific results.

## JSON contract

Top level: `schema_version`, `status`, `checks`, `runtime_seconds`,
`group_runtime_seconds`, `config`, `versions`, `source_sha256`, `timestamp_utc`,
`scope`, `raw_artifact`. Hashes cover the draft and all source/specification/test files.
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

Main-file costate rows keep the maximum direction error, direction blocks,
and counts of attempted/rejected branches. Per-direction projections, errors,
step sizes and full ladders are in raw rows to leave room for C5/C6 under 1 MB.
This is a serialization change only. Main-file C1 rows keep grid ranges/counts and numerical error or
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

## C5/C6 — frozen controlled-network provider panel

Targets: Proposition 4.3, Corollary 5.4, Remark 5.7 and §7.1 C5/C6. This extension
imports `provider_menu`, `snapshot`, `state_vector`, `increment_panel`,
`continuation`, `terminal`, `curvature_search`, `full_gradient`, and `richardson`
from the frozen v0.15 modules. `full_gradient` and `richardson` are used **only
by C5**, after the C6 measurement/calibration results have been fixed. The
old `check_E12` is read to reproduce its protocol; it is never executed,
because it writes artifacts in the pinned tree. Attribution of the neural
primitives remains `v015/common.py` → `independent_checks/check_v014_independent.py`.
No existing network implementation is copied or changed.

**Panel and denominators.** The manifest, nine provider laws and all E6 splits
are unchanged. There are 432 provider-versus-reference rows, including 48
reference self-comparisons, on 48 split/cell combinations and 40 unique cells.
The 36 threshold rows are nonsmooth; E12's denominator is the remaining **396**
rows, with **56** oracle-margin rows. Report that exact denominator, including
self-comparisons, so comparison with E12 is direct. Separately evaluate all
36 unordered distinct provider pairs at every split/cell: **1728** pairs,
**1584** on smooth objectives. Repeated cells across transfer splits remain
in reported rates, but each unique cell is executed only once. All rates are
finite-panel descriptive rates, not estimates from independent task samples.

A gold pair j,k is the difference of their exact eight-update outcomes: one
candidate boundary followed by all 128 seven-update reference histories.
Recompute and compare every provider/reference gold with the frozen records
at absolute tolerance 1e-14. Pair order is j minus k. Nonzero gold means
`abs(gold)>1e-10`; wrong sign means `score*gold<0`. Exactly zero predictions are
reported as ties, not counted as wrong (E12's convention). Unavailable scores
are separately counted. Threshold finite-difference scores are formal
nonsmooth diagnostics; those cells can never receive a C6 certificate.

**Calibration states.** Every original frozen snapshot occurs in an E12
holdout somewhere, including the fresh pilot snapshots in `warm_to_fresh`.
Consequently, reusing the original pilot would not give state-disjoint
calibration. Execute two separate, fixed calibration snapshots with the
unchanged `snapshot(2,3)` and `snapshot(2,7)` helper. It performs 3 or 7 Adam
updates at eta=.03 on deterministic label .25 in both contexts. This adds
10 executed calibration-prefix updates; their labels and complete states
are recorded in the raw artifact. Cross these states with only the original
pilot recipes 0/4 and smooth tasks moment/parity, giving eight calibration
cells and 288 unordered pairs per h. Verify that no calibration complete
state equals any held-out complete state. These local calibration additions
never modify the frozen manifest or evaluation rows. All three h values,
calibration snapshots, bins and the large-gold threshold are fixed in code;
no coverage result changes them.

### C5: ranker comparisons and oracle-only diagnostics

For each provider, enumerate the exact mean optimizer increment from the
actual update map. Compute the following scores, then subtract provider
scores for each pair:

1. `less_cosine`: cosine between the provider mean parameter step and the
   current task-objective parameter gradient. For Adam the direction is
   actual delta-theta (including bias correction and epsilon), not a frozen
   denominator approximation. The request literally specifies the loss
   gradient for SGD, so the primary SGD direction is **positive** mean loss
   gradient. Since the task utility is higher-is-better, this has the opposite
   orientation to the SGD update. `less_utility_orientation` additionally
   reports its negative for SGD, leaving Adam unchanged; both are shown to
   avoid interpreting this convention as an optimizer mechanism. Zero-norm
   cosine is defined as zero and tie counts remain visible. The task gradient
   is a coordinate finite difference of the terminal objective at the state,
   without continuation. This is a LESS-style proxy, not the official LESS
   algorithm or a reproduction of its implementation.
2. `parameter_only`: oracle continuation gradient restricted to theta,
   contracted with mean delta-theta.
3. `full_state`: the sum of the theta, m and v contractions. Use E12's exact
   reference continuation, relative coordinate perturbations, Richardson
   gradient and an independent directional Richardson check (tolerance 2e-6).
4. `measured_aligned`: the chosen-h C6 group score, with no gradient oracle.
   Also report `measured_short` using the separately calibrated short surrogate.
   These are ranking scores on every row, regardless of certificate abstention.

For each pair define v-share as the absolute v **contrast** contraction divided
by the sum of the absolute theta/m/v contrast contractions. The fixed bins are
[0,.25), [.25,.5), [.5,.75), [.75,1]. If the denominator is at most 1e-20 or
the objective is threshold, report v-share unavailable. Report tables overall,
for each of the eight optimizer/eta/epsilon configurations, for each original
split and in each bin. Compare wrong-sign rates at v-share <.5 and >=.5 for
cosine and parameter-only, without asserting enrichment or superiority.

Reproduce E12's reference-row eligibility, scores and margin membership.
For full mode, assert exactly 56/396 and assert no wrong sign on both frozen
and recomputed margin rows. Quick mode uses the shorter curvature search,
so only frozen membership and zero errors are required; no fixed quick
recomputed coverage is imposed. Never change C1–C4 assertions or tolerances.

### C6: measurement path, radius construction and selection

**Groups and randomness.** Requested n=64, G=8. Each frozen provider has only
one or two support atoms, so use the explicitly allowed exact-enumeration
branch. Each group's mean is the probability-weighted mean over the entire
law, repeated across eight groups. In particular, .25/.75 laws are not
incorrectly treated as .5/.5. G groups do not pretend to be eight independent
increment samples; increment uncertainty is eliminated by finite integration.
Each group draws one fresh seven-update reference tail. The same omega is
used for every candidate, for all h values, for both surrogate variants
(the short one uses its first update), and for curvature differences. Streams
are `SeedSequence([160160,6,warmup_step,recipe_id])`; task cells sharing state
and recipe share tails. Independent group draws remain independent, even
if some of the finite-law histories coincide by chance.

**Surrogates and calibration.** Aligned means the task objective after the
full seven-update reference continuation, alpha=1, epsilon-sur=0. Short means
the same objective after one reference update. At each h, fit one pooled
positive zero-intercept least-squares alpha from the 288 calibration short
scores to full-tail gold pair contrasts. Set epsilon-sur to the maximum
absolute calibration residual divided by alpha, and freeze both before any
held-out measurements. Nonpositive/nonfinite alpha invalidates the short
certificate and fails the calibration assertion. Per the explicit request,
the target is actual gold, not an unbiased first-order oracle response:
calibration residuals also include Taylor error and finite-group noise.
Transfer of this observed envelope to held-out states/recipes is an assumption,
not an established consequence of Corollary 5.4 (the former first-order Theorem 5.3).

For h in {.001,.003,.01}, evaluate the theorem's
`Z_g = [F(T_omega(x+h*mu_j))-F(T_omega(x+h*mu_k))]/h` and its mean S-hat.
Store all group scores, all radius components and all h options in the raw
companion. The main file reports all-h coverage, selected-h coverage and the
histogram of selected h. Selection is per pair and surrogate: minimize the
**task-unit radius** `alpha*(e_stat+e_fd+epsilon_sur)+e_2`, breaking ties by
smaller h, never consulting gold or whether a choice certifies.

- `e_stat`: the Bernstein term inherited by Corollary 5.4, with unbiased sample variance (ddof=1), G=8,
  M=8 nonreference candidates and delta=.05; `log(4*M*M/delta)` is 8.540909718.
  B is the observed `max(Z)-min(Z)` for that pair and h across the eight
  groups. **This observed range is not a proven bound on population support.**
  Delta is nominal for a single state/h, not a guarantee simultaneous over
  held-out states or the adaptive three-h selection. No stronger confidence
  statement is claimed.
- `e_fd`: measured ray curvature of the actual CRN continuation. Use five
  values on each mean-increment ray, at fractions 0,1/4,1/2,3/4,1 of h.
  Calculate three central second differences (spacing h/4) and the center's
  Richardson extrapolation from spacings h/2 and h/4. The largest absolute
  normalized value across rays/groups supplies M-tilde; also retain the
  coarse/fine discrepancy. Reuse the endpoint values for the scores. These
  positive-ray probes avoid negative-v perturbations at cold Adam states.
  Then `e_fd=.5*h*Mtilde*max_g(norm(mu_j)^2+norm(mu_k)^2)`.
- `e_2`: E12's same measured `curvature_search` on all ten actual support
  increment rays and the exact 128-history full reference continuation. It
  samples five segment locations, followed by four hill-climb iterations
  (one in quick mode), without using a gradient oracle. Keep all measurement
  error/invalid-probe metadata. Use `.5*M2*(m2_j+m2_k)`, with exact law-weighted
  second moments. Nonfinite or invalid curvature makes the cell ineligible.
- Both curvature terms consistently use E12's scaled norm
  `norm((delta_theta, eta*delta_m, eta*delta_v))`. Scores themselves retain the
  invariant full-state contraction. The FD and remainder curvature estimates
  are empirical, unpadded estimates on their respective rays; neither is a
  proof of a supremum. They are fixed before measuring the held-out gold.

Certification requires a smooth eligible cell and strict
`abs(S-hat)>e_stat+e_fd+epsilon_sur+e_2/alpha`. Assert zero certified wrong
signs for the aligned surrogate **at every h**, which includes the selected
one. Report short-surrogate mistakes without making zero errors an assertion.
When no aligned pair certifies, state that the zero-error check is vacuous;
PASS means the implementation/regression checks passed, not useful coverage.

Missed opportunities use the fixed threshold `abs(gold)>=.01` (one percentage
point of the bounded task utility). Report how many eligible large-gold pairs
are uncertified, both among all unordered pairs and on the E12 reference rows.
Threshold rows are ineligible, not silently discarded or certified.

**Cost accounting.** Count individual tail trajectory executions, not Python
calls or vectorized batches; also count executed tail updates, so a short
one-update trajectory is distinguishable from a seven-update full tail.
One h needs 9*8=72 score endpoints per unique cell/surrogate. All three h use
216 endpoints; the five-point FD stencil uses 1080 trajectories including
these endpoints. Reference-tail curvature uses 13*5*10*128=83,200 trajectories
per smooth cell in full mode (7*5*10*128=44,800 quick). Threshold cells skip
curvature. Recomputing gold uses 10*128=1280 full tails per cell. Calibration
uses only short scores/FD and gold, no oracle or unnecessary curvature.
State-prefix updates are separately disclosed. Shared curvature is counted
once in combined totals and charged explicitly to each stand-alone surrogate
cost; held-out gold audits and C5 gradient-oracle costs are separate.

Compare with the requested N15/T13 1016 replicas per candidate at M=4,
delta=.05,e=.1. Scaling that reference rate to all executed panel candidates
(including the shared reference) gives 40*9*1016=365,760 trajectories. This
uses the requested M=4 reference rate although the actual panel has M=8;
also report the M=8 Hoeffding rate, ceil(2*log(16/.05)/.1^2)=1154. Report both
endpoint-only and complete measurement costs, including curvature/calibration;
do not claim a speed advantage by dropping measured-radius overhead.

**Artifacts/tests.** C5 has one summary row with all requested tables; raw
pair scores, contrast components and oracle diagnostics are in its raw row.
C6 has a protocol/cost row and one row per surrogate. The raw protocol row
contains all executed states, increments, omega indices, finite-difference
stencils, measured curvature diagnostics and calibration states. Each
surrogate's raw row contains all pair/reference certificates at every h.
Tests cover known-quadratic CRN curvature, nonuniform finite-law integration,
Bernstein constants, calibration units and invalid alpha, radius composition,
h selection independent of score, ties/unavailable rates, accounting,
state separation and the read-only guard. A measurement-path test disables
both gradient-oracle helpers. The CLI regression runs all checks, checks
all denominators and raw arrays, validates selection, preserves scientific
FAILs, and enforces strict JSON, deterministic gzip and the 1 MB limit.

## C6b — Theorem 5.3: finite CRN paired replacements

The revised Theorem 5.3 is the finite-replacement certificate; C6 is now
Corollary 5.4. Its numerical implementation, tolerances and selection logic
remain unchanged (only its theorem label is corrected). New code is isolated
in `c5c6b.py`; it imports the read-only panel adapter and numerical helpers
from `c5c6.py`. C1–C6 retain their computations and RNG streams.

**Panel and gold.** Use exactly the same 40 unique held-out cells, nine
providers (M=8), 48 split/cell occurrences, 1,728 unordered pairs and 432
candidate-versus-reference rows, including self rows and split duplicates.
Recompute the exact 128-history, seven-update gold after measurements and
assert agreement with frozen E6 to 1e-14. Reuse C6's eight separate warm3/warm7
calibration cells, recipes 0/4 and tasks moment/parity; assert disjoint keys
and state vectors. Freeze calibration before held-out measurements. No
held-out outcome, oracle gradient, curvature, mean-increment derivative,
or C5 v-share bin enters measurement, calibration, seeds or radius choices.

**Actual replacements and randomness.** Execute the actual one-round
optimizer increment separately for each of the ten support atoms. For each
replica l, draw one fresh seven-update reference tail omega_l from the frozen
reference law. Continue each updated atom state under that same tail, then
weight the terminal outcomes by each provider's exact law. Thus

```text
Y_lj = sum_a P_j(a) F(T_omega_l(x + Delta_a))
Z_ljk = Y_lj - Y_lk; S_hat_jk = mean_l Z_ljk
```

This is **not** `F(T(x + sum_a P_j(a) Delta_a))`. Integrating before the
nonlinear tail would change the estimand. The .25/.75 laws retain their
weights. Only omega is random. Evaluate r=256; r=64 is its first 64 IID
replicas. Reuse omega across all candidates, both surrogate lengths and
tasks with identical state/recipe. Streams are
`SeedSequence([160160,66,warmup_step,recipe_id,0])`. Quick mode also retains
both requested r values and the entire panel.

**Surrogates and calibration.** Aligned is the original bounded task utility
after seven reference updates, with alpha=1 and epsilon-sur=0. Short is the
same utility after one reference update. For each r separately, fit positive
zero-intercept alpha on all 288 calibration finite pair contrasts, regressing
exact full-tail gold differences on measured short finite differences.
Set epsilon-sur to max absolute residual / alpha. Invalid alpha fails the
check and makes the short certificate ineligible. Raw calibration scores,
gold targets, alpha and residuals are retained. This implements Remark 5.6;
short calibration includes truncation and finite-r error. Its transfer to
held-out states/recipes is an assumption, not a proven uniform envelope.

**Range and certificate.** Moment utility is a power of a probability,
parity utility is a probability of even parity, and threshold utility is an
indicator; all lie in [0,1]. Exact convex support averages also lie in [0,1].
Consequently every difference lies in [-1,1], giving the fixed **a-priori
B=2** independently of all draws, states and gold. For each pair use

```text
V_hat = var(Z, ddof=1)
e_stat = sqrt(2 V_hat log(4 M^2/.05)/r) + 7 B log(4 M^2/.05)/(3(r-1))
radius = e_stat + epsilon_sur
certified = abs(S_hat) > radius; task_radius = alpha * radius
```

There are no h, finite-difference or second-order remainder terms. Also
report the same formula with B=max(Z)-min(Z), explicitly marked an
**observed-range diagnostic** without a population-range guarantee. A zero
observed variance does not erase the a-priori range term. Delta=.05 is
simultaneous over candidate pairs at one fixed state, r and surrogate;
there is no claimed joint 95% guarantee over all states or both r values.

All bounded finite rows, including discontinuous threshold objectives, are
eligible: this theorem requires no smoothness. Report both the primary
1,728-pair denominator and the matched C6 smooth subset of 1,584. Report
all 432 reference rows and the matched E12 396 smooth rows for comparison
with its 56/396 oracle margin. Keep self rows in that historical denominator.
Report coverage, wrong-sign certified pairs (gold tolerance 1e-10), certified
gold ties separately, and missed eligible pairs with abs(gold)>=.01. Assert
zero aligned a-priori wrong signs at both r values on unordered and reference
rows. Do not assert positive coverage, observed-range validity, or short
surrogate correctness; zero coverage makes zero wrong signs vacuous.

**Independent-tail variance comparator.** Execute additional tails from
`SeedSequence([160160,66,warmup_step,recipe_id,1,candidate_id])`, separately
for all nine candidates. Atoms of each candidate share its own omega before
integration. Form differences between these separately executed candidate
outcomes. Save both Z arrays and unbiased sample variances for every pair,
and report Var(Z_CRN)/Var(Z_independent) as pair quantiles and the ratio of
summed variances (using the same split multiplicities). Zero independent
variance yields an unavailable ratio with an explicit count, never an
infinite/NaN number or an omitted pair. These extra executions are diagnostics,
not part of producing a CRN certificate.

**Costs.** Count each executed trajectory and each tail update. Shared CRN
integration uses 10*r actual atom trajectories per unique cell/surrogate,
producing 9*r candidate values. Without support sharing it would use 14*r.
The independent comparator executes 14*r per cell/surrogate. Short calibration
uses 8*10*r trajectories plus 8*10*128 full-tail gold trajectories; these
costs are included in its standalone measurement total. Aligned needs no
calibration. Held-out gold (40*10*128), independent-tail variance diagnostics,
and C5 oracle costs are disclosed separately. State calibration warmups
execute ten prefix updates. Tables give standalone costs for each r, while
physical totals charge nested r=64/256 runs only once at r=256. Compare with
C6's existing full-tail audit count 40*9*1016=365,760 and seven updates per
tail; retain its M=4 versus panel M=8 caveat and the latter's 1154 rate.
The audit has a different prescribed accuracy criterion: cost ratios alone
do not establish a superiority claim.

**Artifacts and tests.** One protocol/cost row and eight rows crossing r,
surrogate and range variant. Raw arrays include initial states/increments,
all CRN and independent omega indices, support-atom outcomes, candidate
outcomes, calibration inputs and pair/reference statistics. Observed-range
rows reuse the corresponding a-priori row's raw pairs, which contain both
radii. Tests distinguish nonlinear support integration from a mean update,
check independent streams, exact gold, Bernstein constants, alpha units,
finite-difference calibration, threshold eligibility, degenerate variance,
trajectory/update accounting and forbidden derivative/curvature calls.

## C5b — fifth ranker: finite replacement point estimate

Append aligned r=256 `paired_replacement` S-hat to C5's four primary rankers
(LESS cosine, parameter-only, full-state and measured aligned first-order),
also retaining C5's measured-short and utility-oriented cosine diagnostics.
Reuse the original C5 rows and oracle contrast v-share bins **for reporting
only**; assert gold agreement. Do not modify C5 or replace any score with
gold. All signs, ties, unavailable counts and denominators follow C5.

Report overall, all eight optimizer/eta/epsilon configurations, all splits,
the four frozen v-share bins, and unavailable v-share rows. For v-share>=.5,
report whether the new wrong-sign rate improves each first-order baseline
and whether it eliminates errors entirely. These are descriptive outcomes,
not asserted superiority. Ranking accuracy does not imply certification.
The runner reuses its completed C5 result; `--only C5b` computes that same
baseline internally. C5b reuses C6b measurements at no additional tail cost.
The raw companion retains every extended C5 row; the main JSON keeps tables
and raw references below 1,000,000 bytes. The full CLI regression checks all
16 check statuses, old score-table equality, denominators, raw array shapes
and point-score reconstruction, with scientific failures preserved.
