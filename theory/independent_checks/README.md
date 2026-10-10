# Independent v0.14 numerical checks

Derived from `SPEC.md`, `Environment_Information_Theory_EN.md`, and execution-plan sections 9, 13.1, and 13.7. No original verification scripts or prior results were read, imported, or copied. The program reads only its own source and the three stated documents for provenance hashes.

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --output theory/independent_checks/results_independent_v014.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --quick --output theory/independent_checks/quick.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C5,C14 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python tests/test_theory_independent_quick.py
```

No installation, network, API, GPU, external checkpoint, or repository training module is used. The executable blocks socket connections and DNS, fixes CUDA visibility, and limits numerical library threads to one. All artifacts and caches stay in this directory. A custom output path must also be inside this directory; a bare output filename is resolved here. Quick-run figures go in `figs/quick/`, preserving the full-run plots.

A failing scientific assertion is recorded with its seed, values, and complete witness under `counterexamples`; the run continues and returns exit code 1 after writing JSON. Solver failures and unexpected implementation errors are separately labelled execution errors. No assertion is relaxed to obtain a green run.

Full run: C1–C22 36.24 s; C23 2.41 s. Master seed: 20261010. Each claim uses NumPy SeedSequence([master seed, claim number]); `--only` preserves the selected claim's stream.

| Claim | Status | Key number |
|---|---|---|
| C1 | PASS | max assertion residual 1.33e-15 |
| C2 | PASS | minimum SPRT/lower-bound ratio 1 |
| C3 | PASS | Prop 5 gap 1/40; Prop 6 gap eta²/8 |
| C4 | PASS | max assertion residual 0 |
| C5 | PASS | max assertion residual 5.76e-14 |
| C6 | PASS | largest coefficient ratio 0.666667 |
| C7 | PASS | max assertion residual 0 |
| C8 | PASS | max assertion residual 0 |
| C9 | PASS | max assertion residual 2.3e-12 |
| C10 | PASS | largest gap/bound ratio 0.964912 |
| C11 | PASS | max assertion residual 0 |
| C12 | PASS | max assertion residual 1.04e-17 |
| C13 | PASS | max assertion residual 0 |
| C14 | PASS | max assertion residual 9.8e-13 |
| C15 | FAIL | coverage gap 1 > bound 0.0296041 |
| C16 | PASS | point-estimate gain -78/15625; audit count 1016 |
| C17 | PASS | median width-bound slack 0.230568 |
| C18 | PASS | independent n^(1−N); replay/descendants constant |
| C19 | PASS | span dimensions 4, 8, 4 |
| C20 | PASS | omitted-feature gap 1/2; coverage gap = rho = 1 |
| C21 | PASS | max assertion residual 0 |
| C22 | PASS | W = min(1, epsilon/alpha) |
| C23 | PASS | 57 local agreements / 69 departures; null 0 |

## Failure and test interpretation

C15 preserves the requested coverage-condition probe as a FAIL. Both audited message laws are deterministically zero at cell 0. Training starts at state 1, where the reference always emits 0 and the candidate always emits 1. The same feature and response `g(y)=y` are valid at both states, which are both declared certified; rho is zero. The audit event occurs with probability one, yet the final expected-return gap is 1 and the claimed bound is approximately 0.0296. The connection from (37) to (39) requires explicit visited-state/provider-law coverage by audited cells. C20 independently confirms that a correctly charged first-exit penalty repairs its coverage control.

The quick test intentionally asserts that there are no FAIL statuses, including C15. It therefore fails while this counterexample remains in the theory; it is not skipped, xfailed, or given an exception. A scientific FAIL is different from a process crash. The full JSON is produced in either case.

## Methods and limits

- Finite rational identities use Fraction or SymPy. LPs use independent direct (P,Q), approximation, and oscillation formulations with HiGHS. Feasibility and comparison assertions use 1e-8; worst feasibility residuals are reported separately.
- Binomial probabilities and sequential random-walk recursion are exact rational calculations. Logarithmic/transcendental bounds and dense-grid diagnostics are evaluated numerically. Cumulant rates are checked by exact finite sample-index expansions through n=220, including scalar and bivariate laws and a vanishing leading contraction.
- Inequality searches use LP extrema, deterministic vertex/event enumeration, local perturbations, or bounded scalar optimization. These searches are numerical evidence, not replacements for proofs. The k=4 binary barrier is globally certified by a feasible event achieving the universal upper bound 1; it does not run 65536 redundant event LPs.
- C18 uses an explicit descendant channel `(Z+1/2)/2`; it does not claim every possible postprocessing channel must preserve a nonzero gap. Independent evidence and training exposure are separately recorded.
- C19 reports exact rational span dimensions including constants. Positive approximation errors are upper bounds from the best found dictionary/Chebyshev subspace, not globally optimal subspaces. The uniform error is evaluated on the complete finite grid, held-out responses are labelled separately, and the discrepancy for each selected subspace is independently optimized by LP.
- C16 reports 100000 independent runs per configuration and four-SE comparisons, as requested. The conditional analogue of (45) is searched without asserting a violation when none is found; an additional exact witness demonstrates conditioning bias.
- C23 is **controlled neural, not Gemma, not an external benchmark**. It uses a real one-hidden-layer tanh policy, manual cross-entropy gradients, SGD at four step sizes, and Adam at two step sizes. Complete finite-history enumeration has no sampling SE; otherwise 2400 independent training runs per source are used. Float64 nonlinear arithmetic remains numerical even when histories are enumerated.
- The neural affine comparator is the fixed initial-tangent SGD linearization, explicitly distinguished from the scalar convex-affine theorem and from Adam. Agreement thresholds are fixed in code before execution. Departures are empirical outcomes, not failed affine-theorem assertions.

## Figures

- [C18: evidence and exposure](figs/C18_evidence_exposure.png)
- [C19: response complexity](figs/C19_response_complexity.png)
- [C23: controlled neural](figs/C23_controlled_neural.png)

SVG versions are available alongside the PNGs. Full per-configuration numbers, seeds, provenance hashes, package versions, searches, and counterexamples are in [results_independent_v014.json](results_independent_v014.json).

## Observed neural comparison

controlled neural, not Gemma, not an external benchmark. All numbers below have this scope.

| Settings | Agreements | Departures | Largest absolute gap deviation |
|---|---:|---:|---:|
| SGD, step <= 0.1, at most four updates | 28 | 0 | 0.000252011 |
| SGD, step >= 1 or forty updates | 19 | 37 | 0.525786 |
| Adam, all declared settings | 10 | 32 | 0.119631 |

These compare the actual neural gap with the fixed initial-tangent SGD comparator using the stated descriptive threshold. They do not assert the convex-affine theorem for the neural optimizer. Adam is compared to SGD linearization to expose optimizer dependence.
