# Stage 4e validation — 2026-10-05

Engineering validation of `docs/MECHANISM_PLAN.md` Addendum A. The generated readout contains numerical results without findings or mechanism interpretation.

## Execution

- CPU only: `CUDA_VISIBLE_DEVICES=''`.
- ALFWORLD_DATA_ROOT: `/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1`.
- ALFWorld 0.4.2; TextWorld 1.7.0; NumPy 2.4.6.
- 140 environment resets; 0 actions; 0 model/API calls; 0 new training exposures.
- 0 GPU/server launches, scheduled/cluster jobs, network requests, commits or pushes.
- Source evaluation directories and verified collections were read only.

Exact build/readout commands and source directory labels are in [README](../README.md#stage-4e-frozen-environment-traits-and-stratified-readout).

## Frozen artifact

Artifact: [`dev140_traits.json`](../data/manifest/dev140_traits.json).

| field | value |
|---|---|
| games / task groups | 140 / 137 |
| traits content SHA-256 (canonical body, excluding traits_sha256) | `e6f5bde648744d57271e42cd451093c929cd1133ccf3c74f8edf2d35a11e01a8` |
| file SHA-256 | `a9718cd20dd0129a647519b2ff45fb549397f31b872f1b7e1c023acef0fa26e6` |
| aggregate code SHA-256 | `9fb990b2c0d9be815d2dd4e32fb908299a91059b56394d2a7f55a513d52c7ae9` |
| hashed input files | 708 |
| environment seed | 0 |
| room-size cutoff values | 16 / 30 |
| teacher-length cutoff values | 10 / 19 |
| teacher length known / unknown | 114 / 26 |
| coverage contexts | 9 |

Cutoffs use empirical inverse-CDF ceiling ranks; ties stay in the same bucket. Teacher length counts logged steps of the shortest verified successful trajectory. The other environment traits use only each real reset. Coverage uses validated seed receipts; K counts trajectories.

| trait | bucket | games |
|---|---|---|
| distractors | 0 | 52 |
| distractors | 1 | 33 |
| distractors | >=2 | 55 |
| family | simple | 35 |
| family | transform | 68 |
| family | multi-object | 24 |
| family | look | 13 |
| hidden_target | no | 137 |
| hidden_target | yes | 3 |
| minimum_length | low | 42 |
| minimum_length | middle | 34 |
| minimum_length | high | 38 |
| minimum_length | unknown | 26 |
| room_size | low | 49 |
| room_size | middle | 45 |
| room_size | high | 46 |
| state_change | no | 72 |
| state_change | yes | 68 |
| task_type | look_at_obj_in_light | 13 |
| task_type | pick_and_place_simple | 35 |
| task_type | pick_clean_then_place_in_recep | 27 |
| task_type | pick_cool_then_place_in_recep | 25 |
| task_type | pick_heat_then_place_in_recep | 16 |
| task_type | pick_two_obj_and_place | 24 |

| context | K | support seed | unique instances | task groups | pick | heat | cool | clean | two | look |
|---|---|---|---|---|---|---|---|---|---|---|
| k10s0 | 10 | 0 | 10 | 10 | 2 | 2 | 1 | 2 | 1 | 2 |
| k10s1 | 10 | 1 | 10 | 10 | 1 | 1 | 2 | 2 | 2 | 2 |
| k10s2 | 10 | 2 | 10 | 10 | 2 | 2 | 2 | 2 | 1 | 1 |
| k20s0 | 20 | 0 | 20 | 20 | 3 | 4 | 3 | 4 | 3 | 3 |
| k20s1 | 20 | 1 | 20 | 20 | 3 | 3 | 4 | 4 | 3 | 3 |
| k20s2 | 20 | 2 | 20 | 20 | 3 | 3 | 4 | 4 | 3 | 3 |
| k5s0 | 5 | 0 | 5 | 5 | 1 | 1 | 0 | 1 | 1 | 1 |
| k5s1 | 5 | 1 | 5 | 5 | 0 | 1 | 1 | 1 | 1 | 1 |
| k5s2 | 5 | 2 | 5 | 5 | 1 | 1 | 1 | 1 | 1 | 0 |

## Numerical readout

Outputs: [`stratified.json`](figs/stratified-seedsft/stratified.json) and [`stratified.md`](figs/stratified-seedsft/stratified.md).

36 bucket records; 180 row entries. Coverage contexts in this readout: k5s0, k5s1, k10s0, k20s0. Empty buckets remain explicit. All six rows have 140 known outcomes, 0 missing and 0 unknown. Rates and intervals below are proportions.

| row | success / games | rate | Wilson 95% | batch_invariant |
|---|---|---|---|---|
| teacher | 114 / 140 | 0.814286 | [0.741789, 0.869995] | n/a |
| base | 62 / 140 | 0.442857 | [0.363193, 0.525574] | 1 |
| k20s0-p40 | 115 / 140 | 0.821429 | [0.749668, 0.876020] | 1 |
| k10s0-t400k | 95 / 140 | 0.678571 | [0.597331, 0.750274] | 1 |
| k5s1-t400k | 101 / 140 | 0.721429 | [0.642015, 0.789015] | 1 |
| k5s0-t400k | 72 / 140 | 0.514286 | [0.432225, 0.595583] | 1 |

Paired contrasts are row minus base/teacher on common known games. Gap is teacher minus row on the same teacher/row pairs. McNemar p-values are exact two-sided values, unadjusted. Difference intervals subtract simultaneous Bonferroni-Wilson 97.5% bounds for the two discordant probabilities (nominal conservative 95%). The JSON retains all matched task IDs, discordant counts, row-specific denominators and the historical non-invariant serving-noise reference. Tests and intervals use game instances; task-group counts are separate.

| artifact | file SHA-256 |
|---|---|
| `reports/figs/stratified-seedsft/stratified.json` | `a20f08ded737b6416d1d77062fa9f6f5929d8c6501b334e5c7bd80e5674f0064` |
| `reports/figs/stratified-seedsft/stratified.md` | `de8462eb2dbfd6b66143d7bb271099d096195085cec915194806f99f5af83b41` |

## Checks

- Trait file content hash validated; current code and all consumed input hashes match both artifacts.
- Every environment-trait partition sums to 140 games; success counts sum to the corresponding overall count for each row.
- Every comparison in this readout pairs exactly the bucket task IDs; paired denominators equal bucket game counts.
- Every teacher-minus-row gap equals the negative row-minus-teacher paired difference.
- A second build attempt at the frozen path raises `FileExistsError` before resetting any environment; the file hash stays unchanged.
- `git diff --check`: pass.

Focused offline tests: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/test_traits.py tests/test_stratified.py` → **36 passed in 3.81s**.

The rule tests use the Stage 4a real typed-fact fixtures plus explicit controlled variants for containment/opened states. These existing fixtures are post-action snapshots used only for unit rules; the real table uses fresh initial resets. Fake environments assert one reset per game and forbid stepping. Statistical tests cover known exact p-values, Wilson bounds, empty buckets, unknown/missing outcomes, paired denominators, coverage contexts, serving guards, sealed tasks and duplicates.

Full required offline command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q
```

**543 passed, 7 skipped in 534.95s (0:08:54); 0 failures.** Six optional Torch test modules were skipped because Torch is not installed in this environment. The opt-in ALFWorld integration test was skipped because this test command does not set `ALFWORLD_DATA_ROOT`; the separate real-data traits build above did set it and completed all 140 resets. Tests use the existing network/API blocking fixtures.

Markdown values are rounded to six decimal places; the JSON retains full floating-point values.

## Source inventory check

The six requested evaluation directories and all of `data/verified/` were inventoried before and after the work using file sizes and modification timestamps. Every consumed input was also rechecked by its content hash.

| field | value |
|---|---|
| before_files | `44539` |
| after_files | `44539` |
| changed_files | `0` |
| before_sha256 | `896f10d72004e5ce202fa55b73f268b6416c5bf99e7b35bf83632c94797cce69` |
| after_sha256 | `896f10d72004e5ce202fa55b73f268b6416c5bf99e7b35bf83632c94797cce69` |

A separate whole-tree inventory of `reports/runs/` plus `data/verified/` changed concurrently: 436,005 → 439,850 files, with 3,870 differing size/timestamp entries. The sampled changed paths were in the unrelated `reports/runs/mechanism/k20s0-p20-bi/run2/` workspace. This is not an unchanged-whole-tree claim. That workspace was not used or modified by Stage 4e; the requested input inventory above has zero changes.
