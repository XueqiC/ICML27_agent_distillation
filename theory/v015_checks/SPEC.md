# v0.15 independent checks and controlled optimizer experiments — specification

Targets (read both fully before coding):
- `docs/direction/v015_revision/Environment_Information_Theory_EN.md` (v0.14.1, SHA-256 7fd67fa9…), Theorem 19 / eqs. (37)–(40), (39a).
- `docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md` (addendum, SHA-256 c65ecc61…), Props A1–A7, eqs. (A1)–(A17).
- Experimental obligations: `docs/direction/v015_revision/Coding_Agent_Followup_v015_EN.md` §1–§7.
- Existing controlled-neural code to reuse for the experiments: `theory/independent_checks/check_v014_independent.py`
  (functions `law_scalar`, `law_parity`, `neural_init`, `neural_forward`, `neural_gradient`, `neural_step`,
  `tangent_matrix`, `neural_return`, `training_sequences`, `train_sequence`, `check_C23`). Import them; do not copy-edit
  that file and do not change it.

## Ground rules (same as theory/independent_checks/SPEC.md)
- **Independence.** Do NOT import, copy, or read `docs/direction/v015_revision/verify_v015_extensions.py` or
  `results_v015.json`. Derive every check from the statements. Prefer a formulation different from the proof's.
- **Never weaken a check to make it pass.** A failing assertion stays failing: record the exact instance (seed, arrays,
  rationals as strings) under `counterexamples`, status `FAIL`, continue. Expected-failure regressions (e.g. the old (39)
  on its counterexample) must be asserted as "old bound < actual gap", status PASS for the regression.
- Exact arithmetic (`fractions.Fraction` / sympy) for finite rational instances; floats for LPs (scipy highs, assert at
  1e-8), neural arithmetic (float64) and Monte Carlo. Every randomized family: fixed seed from master seed 150150 via
  `SeedSequence([master, group_number])`; report n_instances and worst residual/ratio.
- Adversarial search for every inequality (random restarts + hill-climb maximizing LHS−RHS or LHS/RHS); report best.
- Env: `theory/.venv/bin/python` (numpy 2.3.5, scipy 1.17.0, sympy 1.14.0, matplotlib 3.10.8, pytest 8.4.2).
  `CUDA_VISIBLE_DEVICES=''`, no network, `MPLCONFIGDIR` inside theory/v015_checks/.mplconfig (gitignored).
  Full run ≤ 20 min CPU (≤ 32 processes).
- Write ONLY under `theory/v015_checks/`. Do not touch tests/, baselines/, src/, docs/, configs/, or
  theory/independent_checks/. Do not run the repository test suite (another agent is editing baselines/ and src/).
  Do not commit.

## Deliverables
1. `theory/v015_checks/check_v015.py`: entry point `--output`, `--quick`, `--only V3,E2`; one function per group.
2. `results_v015_independent.json`: per group `{id, statement, method, kind, n_instances, worst_residual|best_ratio,
   status PASS|FAIL|INFO, notes, data}`, top-level counterexamples, runtime, versions, master seed, source SHA-256s.
3. `README.md` (what each group checks, how, result table), `full_run.log`, `figs/*.png|svg` (one per experiment group;
   labelled "controlled neural, not Gemma" where applicable), `tests/test_v015_quick.py` running `--quick` and asserting
   no unexpected FAIL (it must fail if any group FAILs). `.gitignore` for .mplconfig/ and quick outputs.

## Part V — independent checks of the new mathematics
- **V1 Thm 19 repair / Prop A1 (A2)–(A5), (39a).** Finite Markov training processes (2–4 certified states + exit state,
  message alphabets 2–4, random laws, random bounded responses with explicit decomposition c + bᵀφ + r). Compute the
  exact gap |L_E(0)−L_E(j)| by the stopped telescoping/occupancy calculation, the exact a_{t,j}(z), ρ, and (A4);
  assert gap ≤ bound in ≥ 2000 random + adversarial instances. Include (A5) with state-dependent features and
  α_zx. Regressions: (a) the v0.14 counterexample (old (39) value with n0=nS=10,000, δ=0.05, L=M=d=1 vs gap 1);
  (b) the same instance under (39a) with a=1, and with the state counted as exit (ρ=1); (c) omitting a reachable state
  and setting its term to zero must produce a violation for some instance (search for one); (d) a case where
  parameters are identical but optimizer state differs and the response changes (parameter-only coverage misses it).
  Also verify that (40) holds under its state-independent IID affine premise on random instances.
- **V2 Prop A2 (A7)–(A8).** Cold Adam/AdamW first step equals c − η g/(|g|+ε) for random vectors (float64 and exact
  rational where sqrt is exact), against an independent implementation of (A6) and against `torch`-free reference
  arithmetic. Inverse (A8) round-trip; report condition growth near |u|→1. Ideal sign quotient: on random finite
  gradient alphabets (p=1..4 coordinates, with zeros), count reachable sign patterns s and verify by an LP / rank
  computation that s−1 linear statistics are necessary and sufficient for equality of all bounded utilities of θ₁
  (construct the separating zero-mass perturbation when fewer features are used). Joint-vs-marginal sign example.
- **V3 Prop A3 (A9).** Random finite gradient laws P,Q, random coordinatewise-Lipschitz F (e.g. clipped sums of
  piecewise-linear functions with known L_i, ranges in [0,1]), random a, ε. Assert (A9) and Δ_sign ≤ TV(P_sign,Q_sign);
  adversarial search over a, ε, F.
- **V4 Prop A4 (A10).** The exact rational two-step counterexample (moments −1/8 vs +1/8, v̂₂ = 17/24 vs 41/24,
  opposite second steps), its quadratic-loss realization, and the positive-ε variant (ε = 1e-8 and a log grid):
  compute both first updates exactly and confirm the second-step directions stay opposite. Random search for more
  same-sign-history / opposite-second-step pairs across β₁, β₂ grids (report the fraction of the grid where they exist).
- **V5 Props A5–A6 (A11)–(A14).** Exact enumeration of the hidden-bit model: brute-force the Bayes-optimal policy over
  all transcripts for small d (≤ 5), K (≤ 4), budgets n (≤ 3), random w, random reveal-set laws with within-trajectory
  dependence (not only singletons); compare with (A11), (A13), and the greedy largest-weight rule. Include a decision-
  tree search over adaptive query policies (label-dependent) to confirm adaptivity does not help. Verify
  E[(M_K−n)+] ≥ (E M_K − n)+ with strictness witnesses. Verify the occupancy distribution of C_K by two independent
  methods (recursion and inclusion–exclusion).
- **V6 Prop A7 (A15)–(A16).** Two-world adaptive query/stop processes on small finite alphabets with history-dependent
  query rules and stopping (N_max ≤ 4): compute exact transcript KL, E_v N, decision error; assert (A15) for every
  instance; adversarial search for tightness ratio. Also the κ_X = 0 impossibility case.
- **V7 Sharp recursion remark (§7 of addendum).** e_{t+1} ≤ (1+ηL)e_t + Cη² ⇒ the geometric-sum bound; a witness
  where the linear-in-R simplification fails without a stability condition.

## Part E — controlled optimizer experiments (Followup §1–§5)
Use the C23 network/laws from the v0.14 suite (one-hot context → 4 tanh units → sigmoid, mean CE, init as in
`neural_init`; Adam β=(0.9,0.999), ε=1e-8 outside sqrt, bias correction; no clipping/decay unless an ablation says so).
- **E0 C23 reproducibility bundle** (Followup §1): write `c23_bundle.json` with: exact laws for every task (support,
  probabilities as rationals, and which label moments match — note `law_scalar(N)` matches label moments of order
  1..N−1 only), network/init/dtype/loss/reduction, optimizer code and hyperparameters, and for task `moment` N=3
  with Adam η∈{0.01,0.1}, R∈{1,4}: raw per-support (per-history) gradients at each step, parameter updates, m, v,
  final probabilities and returns, plus the exact definitions and recomputed values of the reported gaps (expected
  ≈0.0769 for R=1 and ≈0.1196 for R=4 at η=0.1; affine-SGD comparator ≈0). Also the C15 counterexample and the
  exact Theorem 19 wording it tested (v0.14 SHA 9271ae05…).
- **E1 Actual Adam map audit** (§2.1–2.2): for every support point/history, enumerate the exact expected gradient,
  gradient moments, P(g=0), P(|g|≤ε), the enumerated finite-ε Adam update and its expectation; compare with the
  update actually executed by `neural_step` (predeclared tolerance 1e-12 relative, float64); separately the sign-limit
  diagnostic. Report the three transitions input-moments → gradient-moments → normalized update → task utility for
  the moment N=2,3,4 and parity N=2,3 laws. Ablation matrix, one factor at a time vs the original baseline:
  ε outside vs inside sqrt; ε ∈ {1e-8,1e-6,1e-4,1e-2,1e-1}; bias correction on/off; fresh vs genuinely warm moments
  (warm = after a declared pre-training phase on a third, common law); clipping off vs global-norm clip at a declared
  value; mean vs sum loss reduction; original support vs shifted support (labels shifted so all first gradients share
  one sign; recheck which moment equalities survive the shift). Table: final return gap and its change per factor.
- **E2 Optimizer-matched frozen-Jacobian comparator** (§2.3): from a common snapshot (θ₀,m₀,v₀), five branches:
  actual net + Adam; frozen-Jacobian model (linearize the pre-sigmoid logit in θ at θ₀, keep sigmoid + CE) + the same
  Adam with its own moments; actual net + SGD; frozen-Jacobian + SGD; the existing tangent affine-SGD construction.
  Same laws, data order (enumerated histories where feasible), task returns, η ∈ {0.01, 0.1, 1}, R ∈ {1,4,40}. Log
  ‖J_θt − J_θ0‖_F and the logit linearization residual along training. Report which gaps persist under the frozen
  Jacobian (⇒ evolving features unnecessary for that gap).
- **E3 History regression** (§2.4): (1,−3/4) vs (2,−3/4) in the actual scalar optimizer code used by E1/E2 (exact
  rational at ε=0 where possible, float64 at ε=1e-8), plus the same pair realized as real gradients of the controlled
  network (find labels/contexts producing a same-sign two-step history with opposite second updates, or report that
  none exists on a declared search grid).
- **E4 Coverage diagnostics in the controlled learner** (§3 item 6): audit anchors = a few actual learner states
  (after 0, 2, 5 updates under the reference law); measure D_x, then at fresh actually reached states z estimate
  a_{t,j}(z) (feature discrepancy transport) and the response residual; compare anchored predictions of the
  provider contrast with fresh full-tail measurements. Report residual distributions and failures; label as
  empirical diagnostics, not certificates.
- **E5 Hidden-bit K curves** (§4): exact curves R_K and R_{K,Q} for m ∈ {4, 8, 16}, uniform and nonuniform w, q;
  passive extra demonstrations vs active unseen-context queries vs replay of seen information (replay = zero gain);
  plus an actual-learner curve: a small learner (tabular logits per context trained by Adam and by SGD for a fixed
  number of steps on the revealed bits) to separate information availability from the learner's use of it.
  Figure with K on the x-axis and queries as separate series.
- **E6 Proxy validation G5 in the controlled learner** (§5): build a provider menu (≥ 6 label laws per state,
  including moment-matched pairs and sign-matched pairs), learner states (fresh, warm), recipes (Adam η∈{0.01,0.1},
  SGD η∈{0.1,1}, ε∈{1e-8,1e-3}), task responses (p^N, parity, threshold). Gold = full-tail expected return contrast
  vs the reference provider (enumerated or ≥ 2000 independent runs with SE). Proxies frozen on a pilot split: input-
  moment error, mean gradient, sign-only features, actual first-step Δθ, Δθ + (m, v), short-tail (1 and 4 updates)
  measured response, random score. Held-out splits kept separate: held-out states, held-out task responses, held-out
  recipes (fresh↔warm, changed ε, held-out lr). Metrics: selection regret, wrong-sign rate, Spearman, calibration,
  abstention. Report negative results plainly.
- **E7 Gold-audit cost forecast** (§6): in the controlled learner, paired-variance pilot with correctly coupled
  marginals; decompose variance into block sampling / suffix training / deployment; replicates needed under the
  Hoeffding allocation (M=4, δ=0.05, e=0.1 ⇒ 1016) vs an empirical-Bernstein (Maurer–Pontil) allocation using the
  measured variance; then a Gemma cost projection as a clearly labelled forecast using these measured hpg costs
  (inputs, not results): 4×B200 full-FT SFT on K=20 ≈ 17 min; 1-GPU explore/MC segment ≈ 3.3 h; LoRA SFT K=50 1×B200
  ≈ 1.5 h; 140-game eval ≈ 0.7–1.6 h per checkpoint.

Write a final summary listing each group, status, key numbers, and anything that could not be implemented as specified.
