# Independent numerical verification of the v0.14 theory — specification

Target document: `docs/direction/v014_handoff/Environment_Information_Theory_EN.md`
(SHA-256 9271ae05…). Every statement number below refers to that file.

## Ground rules

- **Independence.** Derive every check from the statement in the theory document.
  Do NOT import, copy, or read `verify_environment_information_v014.py`,
  `verify_environment_information_v013.py`, or their results JSON. Use a different
  formulation from the one used in the paper's own proof wherever possible (e.g. solve
  the primal LP directly in (P, Q) variables instead of the proof's v-reformulation).
- **Never weaken a check to make it pass.** If an assertion fails, keep it failing,
  record the exact instance (seed, arrays as lists, rationals as strings) in the
  results JSON under `counterexamples`, set the claim status to `FAIL`, and continue
  with the other claims. A failing claim is a valuable result.
- Exact arithmetic (`fractions.Fraction` or `sympy.Rational`) for identities on finite
  rational instances; floats only for LPs (scipy `linprog`, method `highs`) and Monte
  Carlo. LP tolerances: report residuals, assert at 1e-8.
- Every randomized family: fixed seed from a master seed, report the number of
  instances and the worst residual / worst ratio found.
- **Adversarial search** where a claim is an inequality: in addition to random
  instances, run a simple optimizer (random restarts + local perturbation hill-climb,
  or an LP when the inner problem is linear) that tries to maximize
  `LHS - RHS` (or `LHS / RHS`). Report the best value found.
- Environment: `theory/.venv/bin/python` (numpy 2.3.5, scipy 1.17.0, sympy 1.14.0,
  matplotlib 3.10.8). Run with `CUDA_VISIBLE_DEVICES=''`, no network. Total runtime
  target ≤ 15 min on CPU (multiprocessing allowed, ≤ 32 workers).
- Write ONLY under `theory/independent_checks/`. Never modify `docs/`.

## Deliverables

1. `theory/independent_checks/check_v014_independent.py` — one entry point,
   `--output results_independent_v014.json`, `--quick` (small instance counts for a
   smoke run), `--only C5,C14` (subset). Code organized as one function per claim
   group below, each returning a dict.
2. `theory/independent_checks/results_independent_v014.json` from a full run:
   per claim `{id, statement, method, kind: exact|random|adversarial|monte_carlo|witness,
   n_instances, worst_residual or best_ratio, status: PASS|FAIL|INFO, notes}`,
   plus top-level `counterexamples`, runtime, package versions, master seed.
3. `theory/independent_checks/README.md` — how to run; one table row per claim with
   status and the key number.
4. `tests/test_theory_independent_quick.py` — runs `--quick` and asserts no FAIL.

## Claims and required checks

**C1 — Prop 1, Prop 2, eqs (3)–(4), Prop 3.**
Random finite instances: 3–6 environments with a random prior; finite Π ⊂ Π_ref with
random values J_E(π) ∈ [0,1]; random transcript channel P(D | E) on a finite
alphabet; random learner = stochastic map D → Δ(Π). Exact rationals.
- (1): identity and nonnegativity of R_rep, R_info, R_alg (≥ 500 instances).
- Prop 2: build instances with and without a common posterior-optimal policy
  (including ties); check R_info = 0 ⇔ condition (2).
- (3) = (4): solve the primal min over ρ and the dual max over μ as two separate LPs;
  check equality and that r(C) = 0 ⇔ a common optimal policy exists (≥ 200).
- Prop 3: random garbling D' = channel(D): inequality (5) exactly; adding Z ⊥ E | D
  leaves B unchanged.

**C2 — Theorem 4 and the compilation paragraph.**
For Δ ∈ {1/8, 1/10, 1/16, 1/32, 1/64} and δ ∈ {0.01, 0.05, 0.1, 0.25, 0.4}:
- Symbolic J_v(p) from the table equals (6); passive minimax over m gives exactly Δ.
- kl(1/2+u, 1/2−u) ≤ 16u² on (0, 1/4] (dense grid plus analytic sup of the ratio).
- Lower bound (8) is respected by actual strategies: compute the exact smallest fixed n
  for which the threshold test succeeds w.p. ≥ 1−δ in both worlds (exact binomial,
  ties counted as errors), and the expected stopping time of a two-sided SPRT tuned to
  error δ (exact via a finite random-walk recursion or ≥ 10⁵ Monte Carlo runs with SE).
  Both must be ≥ the bound in (8); report the ratios.
- (9): exact binomial error of the threshold test at n = ⌈log(1/δ)/(8Δ²)⌉ ≤ δ.
- Compilation: exact binomial P(|p̂ − q| > 1/8) ≤ δ_L at m = ⌈32 log(2/δ_L)⌉ for
  q ∈ {1/8, 7/8}; resulting regret ≤ Δ/2 whenever |p̂ − q| ≤ 1/8.

**C3 — Propositions 5 and 6.**
- Prop 5: symbolic J(π_p) = 1/2 + p/20 for both sources and all p; teacher values 3/5;
  teacher optimality per state; exact expected return of the frequency learner for
  m = 1…30 (binomial expectation) equals 43/80 and 41/80 for every m.
- Prop 6: exact enumeration with symbolic η: equal means, gap η²/8, reversal for 1−p².

**C4 — Theorem 7, eqs (14)–(17).**
Random finite learning processes: |Z| ∈ [4, 20], R ∈ [1, 8], message alphabets, actual
deterministic updates U_t(z, y, ξ) with finite ξ, reference and alternative message
laws, F ∈ [0,1], common initial law. Exact rationals.
- (14) exactly; (15) with ε_t(z) = |(K⁰−K^S)V(z)|.
- (17): random certified subsets; ρ = exact first-exit probability before R; check
  the bound, plus adversarial search for a violation.
- One-boundary characterization (16): P, Q equal on all g of a declared family ⇒ equal
  one-boundary values; a nonzero difference on some g ⇒ the constructed intervention
  separates them.
- Witness that pointwise matching is not necessary for one fixed final mean
  (cancellation across rounds).

**C5 — Theorem 8, Corollary 9.**
Random m ∈ [4, 12], response sets G ⊂ [0,1]^m, features with d < m−1 (≥ 300).
- (a): span criterion vs an LP that searches feature-matched P, Q differing on g;
  d₀ via exact rank.
- (b): D_φ(G) by the primal LP over (P, Q) directly vs 2·min_h ‖g−h‖∞ by a separate LP.
  Include g ∈ H_φ (expect 0).
- (c): random decompositions g = c + bφ + r; maximize LHS − RHS of (21) over (P, Q) by
  LP; expect ≤ 0.
- Cor 9: for m ≤ 8 enumerate all events: sup_A |(P−Q)(a+γ1_A)| = γ·TV; d₀ = m−1.

**C6 — Lemma 10.**
Random finite deployment trees: ≤ N decisions per path, contexts j ∈ [k] reused,
history-dependent context choice, chance nodes, terminal rewards ∈ [0,1].
Compute J(p) as an exact sympy polynomial: total degree ≤ N, Σ|c_α| ≤ 3^N,
J ∈ [0,1] on random p. Adversarially maximize Σ|c_α| / 3^N and report it (the bound is
not claimed tight).

**C7 — Theorem 11.**
- Sufficiency: P, Q on finite supports in [0,1]^k matching all joint moments with
  |α| ≤ N (P random positive, Q = P + a·v, v in the exact null space of the moment
  constraints including total mass, a small enough to keep Q ≥ 0); random affine
  learners (23) with R ∈ [1, 5]; random tasks from the C6 generator → exactly equal
  learning values.
- Necessity: for every α with |α| ≤ N, the task "visit context j α_j times, succeed
  iff all actions are 1" has value p^α (check symbolically through the C6 tree
  evaluator); P, Q differing only on E Z^α give different learning values under p = Z.
- The two-context example (equal marginals, cross moments 1/2 vs 0).

**C8 — Theorem 12.**
- Scalar laws (26) for N = 1…14 in exact rationals: masses sum to 1, moments ℓ < N
  equal, Δ_N formula.
- Parity: N = 1…8, enumerate {0,1}^N; every joint moment with |α| < N equal (enumerate
  all α, not only subsets); J_even values 1 and 0 under p = Z; (27) equals the even-parity
  probability for random p.

**C9 — Theorem 13.**
- (28): exact rank of the monomial evaluation matrix on the grid for all (k, N) with
  k ≤ 3, N ≤ 5 (and k = 4, N ≤ 3) equals C(k+N, N); dropping any one monomial feature
  leaves a separable pair (LP).
- Binary barrier: k ≤ 4, random feature sets with d < 2^k − 1: by LP over events A,
  max over feature-matched (P, Q) of (P−Q)1_A equals 1; with d = 2^k − 1 generic
  features it equals 0.

**C10 — Theorem 14.**
Unmatched random P, Q on finite supports, random affine learners (R up to 50), random
tasks with exact polynomial J: check |gap| ≤ C_J δ_N ≤ N 3^N δ_N. Adversarial search
over (P, Q) maximizing |gap| / (C_J δ_N); report the best ratio (must be ≤ 1). Show the
bound does not grow with R.

**C11 — Theorem 15 and (31).**
- (31) exactly in rationals for N = 2…8, n = 1…30.
- General (30): scalar and bivariate lattice laws P, Q with equal joint cumulants below
  order r ∈ {2, 3, 4} and different order-r cumulants (construct by solving for
  matching moments). Compute E_{P^n} J(p̄_n) exactly (convolution on the lattice or
  multinomial expansion) for n up to ≥ 200; verify n^{r−1}·gap → the predicted
  coefficient ⟨D^rJ(μ), κ_r(P)−κ_r(Q)⟩ / r!, with residual × n bounded. Include a case
  where the contraction vanishes (faster rate).
- Unequal means: gap → J(μ_P) − J(μ_Q) with O(1/n) correction.

**C12 — Equation (32) and Proposition 16.**
- Chebyshev identity (32) exactly (sympy) for N = 1…16.
- Truncation error for q < N: sup over [0,1] (dense grid + p = 1) equals the coefficient
  tail, equals P(|B−N| > q) for B ~ Bin(2N, 1/2) exactly, ≤ 2e^{−q²/N}.
- Parity: for N ≤ 5, LP for the best uniform approximation of (27) on the cube vertices
  by polynomials of degree < N gives exactly 1/2; the constant 1/2 attains it on [0,1]^N.

**C13 — Theorem 17 and (35).**
- Random polynomial updates with declared (a_i, b_i); exact sympy pull-back; per-block
  degree ≤ D_i from (34); blockwise moment matching through D_i preserves the value
  exactly. Witness Z₁^{2^{R−1}} with the (26) laws. Threshold counterexample 2/3 vs 0.
- (35): smooth V on R³ (random cubic polynomial plus a non-polynomial case such as a
  logistic), discrete gradient laws P, Q, update θ − ηG: compare the exact
  (K^P−K^Q)V(θ) with the first two terms; remainder ≤ (M₃η³/6)(E_P‖G‖³ + E_Q‖G‖³) with
  M₃ computed as a bound on the third-derivative operator norm on the segments. Several η.

**C14 — Theorem 18.**
Random finite instances (≥ 200, m ∈ [3, 12], d ∈ [1, m−1], ε_j random including 0):
primal LP over (P, Q) directly vs the dual formula min_b osc(g − Σb_jφ_j) + Σε_j|b_j|
solved as its own LP; also the 2‖·‖∞ form with c. Check W ≤ osc(g), the ε = 0 reduction
to Theorem 8(b), and W → osc(g) for large ε.

**C15 — Theorem 19, eqs (37)–(40).**
- Monte Carlo (≥ 2000 trials per configuration, several (M, L, d, n₀, n_S, δ)): frequency
  of event (37) ≥ 1 − δ (report with SE); (38) holds on the event.
- Finite learning-process instance where every response is exactly c + bᵀφ_x (τ = 0):
  check (39)/(40) on the audit event.
- **Coverage-condition probe.** Construct a state-dependent provider that matches the
  reference at the audited cells but not at a visited unaudited state, with the
  response decomposition valid at both states. Report whether (39) fails when the
  certified set contains the unaudited state. (This tests whether the theorem needs an
  explicit "certified states ⊆ audited cells / provider law at a visited state equals
  that at its audited cell" condition.)

**C16 — Theorem 20.**
Small finite learning process (|Z| ≤ 6, R ≤ 6), M candidate kernels per state with
known true advantages A_t(z, j), audits that sample bounded paired contrasts from the
true model (full reference tail + Bernoulli deployment), LCB selection with
e_t = sqrt(2 log(2M/δ_t)/n_t), fresh update. Stress with small n_t and large δ_t
(0.1–0.4) and candidates with slightly negative mean and high variance.
- Estimate E[L_adaptive] − L₀ by Monte Carlo (≥ 10⁵ runs, report SE) and check (44)
  and (45) with a 4·SE margin; also check claim 1 (frequency that Σ selected true drifts
  ≥ Σ B_t is ≥ 1 − δ).
- Point-estimate selection without the LCB: demonstrate an instance with strictly
  negative expected gain.
- Conditioning probe: search for an instance where E[L − L₀ | all audits passed] is
  below the analogue of the (45) bound, illustrating why the paper refuses that
  conditional statement; report whether found.
- Recompute ⌈2 log(160)/0.01⌉ = 1016.

**C17 — Proposition 21 and the additive null.**
Random |Y₁|, |Y₂| ∈ [2, 5], strictly positive marginals, H random:
- width by two LPs over the coupling polytope; width = 0 for additive H (random a, b);
  width > 0 for random nonadditive H (≥ 300); (46) holds with the inf over (a, b)
  solved by LP; report the typical slack.
- Forbidden pairings: construct an example with zero width on the feasible couplings
  but nonadditive H.
- Strict null: one SGD step of a small linear/logistic model at a common parameter,
  the same examples regrouped into different packets with identical weights → identical
  update (exact in rationals for the linear case); then two sequential updates with
  regrouping → a nonzero difference.

## Additional finite experiments required by the execution plan (§9, §13.1, §13.7)

**C18 — Independent blocks versus replay versus synthetic descendants (§13.1, Thm 15).**
Using the (26) laws and the parity laws: compare the learning-value gap for (a) n
independent blocks, (b) one block replayed n times (same training exposure),
(c) n "synthetic descendants" generated by a postprocessing channel of one block.
Show (a) decays as Δ_N/n^{N−1} while (b) and (c) do not decay with exposure. Plot
both an exposure panel and an independent-evidence panel.

**C19 — Same maximum degree, different response-space complexity (§13.7).**
On a fully enumerable message grid, compare three response families of the same
maximum degree N: scalar powers p^N (and its affine pull-backs), multivariate parity
(27), and a structured low-complexity multivariate control (e.g. responses depending
only on Σ_j p_j). Report (i) exact span dimension including the constant (rational
rank), (ii) the optimal uniform approximation error at feature budgets d = 1…D using
the best d-dimensional subspace found (state clearly whether this is exact or an
upper bound; for the scalar family the Chebyshev subspace gives an upper bound),
(iii) Thm 8(b) discrepancy for that subspace by LP. SVD/Frobenius numbers may be
shown only as labelled diagnostics.

**C20 — Finite calibration: correct features, omitted features, coverage failure.**
A finite learning process with known responses: (i) a correctly specified feature set
— the certificate (39)/(40) holds; (ii) one decision-relevant feature omitted — two
providers match every audited feature yet differ in learning value (quantify);
(iii) coverage failure — training reaches states outside the audited set and the
certificate without ρ fails, with ρ it holds. This complements the C15 probe.

**C21 — Signed local continuation identity (43) and fidelity versus improvement (41).**
Finite processes: (43) exactly with state-dependent provider choices. Fidelity vs
improvement: a good reference (γ > 0) and an ineffective reference (γ ≤ 0), each with
a high-fidelity candidate (small ε): (41) holds and the high-fidelity candidate of the
ineffective reference does not improve the initial student.

**C22 — Stability control for Thm 18.**
Binary response g, feature φ = αg, tolerance ε: the worst-case discrepancy equals
min{1, ε/α} (primal LP and dual formula, α ∈ {1, 0.5, 0.1, 0.01}, several ε).

**C23 — Controlled small neural experiment (§9 last paragraph).**
After C1–C22. Clearly separated in code and results under `controlled_neural`. Train
an actual small neural policy (numpy or a tiny autograd you write; no torch needed)
on the same finite environments: a 1-hidden-layer tanh network mapping one-hot
context to the action-1 probability through a sigmoid, trained by cross-entropy on
blocks of labels, with plain SGD at several step sizes and with Adam. For each of
(a) Prop 5 sources, (b) Prop 6 independent vs paired packets, (c) the (26) moment
laws at N = 2, 3, 4 with J = p^N, (d) the parity laws at N = 2, 3 with (27), (e) a
coupling regrouping of the same labels into different packets: measure the actual
final expected return gap between the two sources (exact over the finite block law
where enumerable, otherwise ≥ 2000 independent training replicas with SE; replicas
are independent training runs, not repeated deployments). Report where the affine
predictions hold approximately (small step, few updates) and where they fail
(large step, Adam, many updates), and the additive-SGD null (identical update for a
single common-state step with identical weights). Label every number "controlled
neural, not Gemma, not an external benchmark".
