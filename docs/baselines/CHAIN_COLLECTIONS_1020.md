# Teacher collections for the nested support chain (2026-10-20)

Collect once on `data/verified/seeds_full_train_r2/subsets/k500`, seed 0.
Supply K30/K100/K300 support directories explicitly. Sampling and the
`stratified_nested_v2` verified loader belong to `wt/kchain`; integrate that
work before using real v2 inputs. This change does not generate support sets
or modify training.

K counts trusted executed seed trajectories. Each plan also reports unique
instances and task groups. These methods produce additional teacher data on
those instances; candidate counts and repeated training exposure are not K.

## RAI commands

Run from the deployed project root with its existing environment and frozen
manifest/config. Both collectors must use the **same** `TEACHER_RATE_LIMIT_DIR`
as all other concurrently running teacher jobs. If jobs are already running,
use their existing shared directory; do not create a second limiter for them.
The path below is the common default for new explicit cells.

```bash
export TEACHER_RATE_LIMIT_DIR="$PWD/data/teacher_rate_limit"
support="$PWD/data/verified/seeds_full_train_r2/subsets"

bash scripts/rai_collect_adftp.sh \
  --cell chain_k500_s0 --source-view "$support/k500" \
  --output-root reports/runs/adftp-chain --shards 32 \
  --subset-view "30=$support/k30" \
  --subset-view "100=$support/k100" \
  --subset-view "300=$support/k300"

bash scripts/rai_collect_smartad.sh \
  --cell chain_k500_s0 --source-view "$support/k500" \
  --output-root reports/runs/smartad-chain --shards 32 \
  --subset-view "30=$support/k30" \
  --subset-view "100=$support/k100" \
  --subset-view "300=$support/k300"
```

Append `--dry-run` to either command to validate sources and print the cell,
counts, support bindings and all worker commands without creating an output,
limiter, lock, or teacher client. Append `--resume` for an existing output.
`--config`, `--log-root`, and `--rate-limit-dir` are also supported; the shared
`TEACHER_RATE_LIMIT_DIR` overrides the latter, matching the limiter itself.
To plan multiple explicit cells, repeat `--cell` and `--source-view` in matching
order. Every source and existing cell is preflighted before workers start.

The wrappers retain their old no-argument and `--resume` paths unchanged:
AD-FTP runs K100 s0/s1/s2 and full TRAIN; SmartAD runs K100 s0/s1/s2 and adds
full TRAIN only when its source is complete. Their original output names,
source paths, K5/K10/K20/K50 views, limiter argument defaults and SmartAD
16-shard default remain available. Explicit cells default to 32 shards;
`ADFTP_SHARDS`/`SMARTAD_SHARDS` and `ADFTP_ROOT`/`SMARTAD_ROOT` supply defaults
for the new argument path. Do not point a new cell at an old cell's output.

## HPG: eight selection GPUs, then a checked CPU merge

Copy the immutable candidate pool **and its source evidence** to HPG. For v2,
this includes the full-TRAIN collection and its support directories: a support
view refers to the full collection. Retain the published files and absolute
paths in JSON; use `SMARTAD_PATH_MAP` to resolve them on HPG. The frozen task
manifest and split freeze must also be present and match. No teacher stages
run during selection or merge.

For a collection launched from this worktree, these are the launch commands
on HPG after deploying the code and evidence:

```bash
source configs/hpg.env
export VERIFIED="$PROJECT_ROOT/reports/runs/smartad-chain/chain_k500_s0"
export SMARTAD_OUT="$PROJECT_ROOT/reports/runs/smartad-chain-selected/chain_k500_s0"
export SMARTAD_THROUGH=select SMARTAD_SELECTION_SHARDS=8
export SMARTAD_PATH_MAP="/home/xueqi/hq/projects/ICML27_wt_chaincollect=$PROJECT_ROOT"
export SMARTAD_MERGE_SCORES=0

score_job=$(sbatch --parsable --array=0-7%8 \
  --partition=hpg-b200 --gpus=1 --cpus-per-task=8 --mem=96G \
  --time=03:00:00 --output='slurm-smartad-score-%A_%a.out' \
  --export=ALL scripts/hpg_smartad.slurm)

SMARTAD_MERGE_SCORES=1 sbatch --dependency="afterok:${score_job%%;*}" \
  --partition=hpg-milan --gpus=0 --cpus-per-task=8 --mem=96G \
  --time=02:00:00 --output='slurm-smartad-merge-%j.out' \
  --export=ALL scripts/hpg_smartad.slurm
```

If collection used another RAI project path, change the left side of the map
to that exact prefix. Multiple disjoint maps are supported, as before.
`MODEL_PATH` must be the unchanged initial student snapshot, not an adapter.
`SMARTAD_SELECTION_SHARD_INDEX` overrides `SLURM_ARRAY_TASK_ID` for individual
retries. Rerunning an array reuses completed scores. To use four GPUs, set
`SMARTAD_SELECTION_SHARDS=4` and `--array=0-3%4` in a fresh or resumed run;
merge with the same shard count as the completed array.

Each worker deterministically takes every Nth entry of the frozen flattened
(task order, candidate order) list. One independent initial student runs on
one GPU. Incorrect candidates get rejection records without model scoring.
The NLL definition and candidate-ID tie break are unchanged. Per-candidate
locks prevent overlapping workers from scoring the same completed candidate
again. Workers write immutable score files and shard receipts. They do not
publish a selected collection. Merge needs every declared shard, checks its
identity, candidate-list digest and each score-file SHA256, then applies the
same per-task choice as serial selection. Merge never loads a model.

At the supplied estimate of 4,000 correct candidates × 9 seconds, eight GPUs
need about 75 minutes of scoring, plus validation/copy/queue overhead. Four
need about 150 minutes before overhead. Use eight for the 1–2 hour target;
this is arithmetic from the supplied estimate, not a measured GPU result.

## Published views and integrity

- AD-FTP: `adftp-chain/chain_k500_s0/collection`, with
  `collection/subsets/k{30,100,300}`.
- SmartAD candidates: `smartad-chain/chain_k500_s0/candidates.json`, with
  `subsets/k{30,100,300}/candidates.json`. `load_pool()` resolves these views
  against the immutable largest pool. Selection must consume the largest pool.
- SmartAD selected: `smartad-chain-selected/chain_k500_s0/collection`, with
  `collection/subsets/k{30,100,300}`. A view contains exactly the parent
  collection's per-task selected receipts for its supplied support task set.

`--subset-view K=PATH` checks actual verified `task_id` sets, K, seed, and
nesting. It never reruns stratification. Bindings retain canonical manifest
and selection SHA256, exact file SHA256, and the task-ID list. Changed support
artifacts, missing/outside tasks, non-nested views, duplicate K labels, or
mixed `--subset-k`/`--subset-view` requests are refused. V2 does not accept the
legacy `--subset-k` sampler.

Explicit selected/FTP views are ordinary verified publication directories.
They contain private, hash-checked copies of their parent receipt bundles so
the normal loader's path containment checks remain intact and editing a view
cannot change its parent. This uses extra disk space. Their identities bind
the support view and parent collection; their accounting records zero new
calls and transitions. Copies do not create new executions or billable work.
Do not sum overlapping view costs as independent collections. Candidate
views only index parent candidates and do not duplicate their evidence.

AD-FTP has per-task locks and deterministic task sharding; publication waits
for all receipts. Its 123-call task cap and three-attempt/fallback behavior
are unchanged. SmartAD keeps ten slots, 41 calls per candidate / 410 per task,
its existing candidate-01 reuse decision and deferred-slot second pass.
Every failed attempt, rejection and unknown outcome remains in the existing
hashed evidence and ledgers. No new retry/reset mechanism is introduced.

Collector resume keeps the original identity and durable receipt bindings.
All source/config/teacher/sampling/limiter/view fields must still match.
Code/dependency-only upgrades are disclosed under `resume_versions/` without
rewriting old identities, receipts, or usage. Selection uses the same rule for
its score identity; model artifacts, scoring settings and relocation mapping
remain frozen. Original provenance stays intact alongside the new hashes.

## Offline verification

```bash
CUDA_VISIBLE_DEVICES='' SED_HASH_WORKERS=1 .venv/bin/python -m pytest -q \
  tests/test_chain_collections.py tests/test_adftp.py tests/test_smartad.py \
  tests/test_smartad_shards.py tests/test_smartad_relocation.py \
  tests/test_adftp_sad_launchers.py
```

Tests use fake clients/environments, prohibit network/OpenAI access, and cover
explicit task equality/refusal, launcher plans, old cell arguments, overlapping
AD-FTP shards, unchanged candidate-01 behavior, receipt reuse across upgrades,
serial/sharded selection equality, incomplete/tampered score merges and HPG
array/merge wiring. The v2 integration test runs when the K-chain loader is
available; in the isolated worktrees it can be tested read-only with:

```bash
CUDA_VISIBLE_DEVICES='' SED_HASH_WORKERS=1 .venv/bin/python -m pytest -q \
  -o pythonpath='/home/xueqi/hq/projects/ICML27_wt_kchain/src .' \
  tests/test_chain_collections.py::test_v2_support_interface_when_kchain_is_integrated
```

No real teacher collection, GPU scoring, training, commit, or push is performed
as part of these checks. Synthetic plans are implementation checks, not
scientific results.

### Checks completed in this worktree

- Regression suite: **85 passed, 1 skipped** (the isolated branch lacks the
  K-chain loader).
- Final teacher-chain tests after the last view/score checks: **10 passed,
  1 skipped**; separate integration against `wt/kchain`: **1 passed**. That
  integration includes relocation of source/pool, four selection shards,
  CPU merge, and both candidate/selected view validation.
- Shell syntax, Python compilation, and `git diff --check` passed.
- Real support directories were absent here. A synthetic, fake-environment
  full source with 504 successful fixtures supplied a genuine loader-validated
  v2 K500 view, plus K30/K100/K300 views. Both RAI wrappers completed dry runs:

  | Method | Cell | Source K / instances / groups | Seed | Shards | Explicit views |
  |---|---|---|---|---|---|
  | AD-FTP | `chain_k500_s0` | 500 / 500 / 500 | 0 | 32 | 30, 100, 300 |
  | SmartAD | `chain_k500_s0` | 500 / 500 / 500 | 0 | 32 | 30, 100, 300 |

  Plans are retained locally at
  `/tmp/chaincollect-k500-dry/adftp-plan.json` and
  `/tmp/chaincollect-k500-dry/smartad-plan.json`. They include every planned
  worker command and exact support hashes/task IDs. Both plans used the same
  limiter path, and neither created its output or limiter directory. Fixture
  setup used fake responses with socket access blocked; no real API/model
  calls occurred. These fixtures remain under `/tmp`, outside research data.
