# Full-TRAIN support views, 2026-10-20

Registered rule: `stratified_nested_v2`, support seed **0**. The initial main
grid is K=30/100/300/500; K=1000/2000 later uses the same order. K=`all` is the
source collection itself, not a new view: `data/verified/seeds_full_train_r2`.
It is complete, with `all_train_v1` selection, 3,553 candidates, 3,098 trusted
executed trajectories / unique instances, 1,383 groups and six types. K counts
trusted trajectories; instances and groups remain separate counts.

## Frozen selection rule

Only `task_id`, `group_id`, and `task_type` from the source's **verified** receipts
determine the order. Failed candidates, episode content, teacher responses and
dev/valid/test data do not influence selection. The frozen TRAIN manifest and
split assignments are checked by the existing evidence loader.

Type order is lexicographic: `look_at_obj_in_light`, `pick_and_place_simple`,
`pick_clean_then_place_in_recep`, `pick_cool_then_place_in_recep`,
`pick_heat_then_place_in_recep`, `pick_two_obj_and_place`. At position n, choose
the nonempty type maximizing `n*N_t - N*count_t`; this is the specified deficit
multiplied by N, avoiding floating-point ties. Exact ties use the type order.

Within each type, sort group IDs, then shuffle with
`random.Random(digest([seed, "groups", type]))`. Sort each group's task IDs, then
shuffle with `random.Random(digest([seed, "instances", type, group_id]))`.
Traverse these groups in rounds: first trial of every group, then second trial
of every group with one, etc. Breadth-first coverage is **within each type**;
proportional scheduling can repeat a group before all groups of another type
have appeared. Every K view is the ordered prefix `pi[:K]`.

`selection.json` records the rule, seed, K, fixed type order, ordered task IDs,
full-order SHA256, per-type counts, distinct and multi-trial group counts, and
source manifest/selection SHA256. These use the repository's canonical JSON
SHA256 (`sed.storage.digest`); the order hash is over the full task-ID list.
Receipt and episode-file references retain the existing byte-hash checks.
The loader reconstructs the entire order from the validated source and rejects
any differing prefix or metadata, even with recomputed local hashes.

There are only 1,383 source groups. K>1,383 necessarily includes multiple trials
from some groups. These are distinct executed instances, not retries counted
as new tasks. The views include all source attempts for their selected
instances; the source retains all failed candidates and costs.

## Read-only dry run

This worktree contains no local verified data. The matching source is available
read-only at `/home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2`.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/make_full_train_views.py \
  --source /home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2 \
  --ks 30 100 300 500 --dry-run
```

Columns below use the fixed type order above (Light, Place, Clean, Cool, Heat,
Two). No real views were written.

| K | Light | Place | Clean | Cool | Heat | Two | Unique instances | Groups | Multi-trial groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 3 | 8 | 5 | 4 | 3 | 7 | 30 | 30 | 0 |
| 100 | 10 | 25 | 17 | 13 | 11 | 24 | 100 | 100 | 0 |
| 300 | 29 | 75 | 51 | 39 | 33 | 73 | 300 | 300 | 0 |
| 500 | 48 | 126 | 86 | 64 | 55 | 121 | 500 | 500 | 0 |
| all (source) | 298 | 777 | 530 | 399 | 342 | 752 | 3098 | 1383 | 1108 |

Full-order SHA256:
`c0b4377842ac5cbbcd6ed5095b4323fe345c1ee2dcc288907c884db5b47bebfa`.
Source manifest SHA256:
`ace9397a65cdbbb6b1725ee9c1de0ec18edbf65c434ab90c9577dc9ca2b09c7a`.
Source selection SHA256:
`5750e358b141d5050d67ff1fc3abc0f3e83ac3bd15d288ba5f68770838e2015c`.
The CLI completed full source receipt/episode validation successfully.
Checking every prefix of this source gives a maximum absolute type discrepancy
of 0.816011 trajectories (<1). Future-prefix calculations give K1000: 1,000
groups / 0 multi-trial groups; K2000: 1,383 groups / 617 multi-trial groups.

After review, run the same CLI without `--dry-run` against the desired source
path. It only creates `subsets/kK/{selection,identity,manifest}.json`, rejects
an incomplete/non-full source, invalid K, and **any existing target directory**,
and makes zero teacher calls or environment steps. Later request only new Ks
(`--ks 1000 2000`); including an existing K is an error. Dry run creates no files
or locks. Existing source files are never opened for writing.

## Consumers and boundaries

| Consumer | New view support |
| --- | --- |
| Kang, AD-FTP, SmartAD, SAD pipelines / shared FSDP loader | Shared validation accepts the rule and sibling trials; rule-based support registration has no legacy K100 ceiling. |
| `train/sft.py` | Uses the existing whitelisted trajectory dataset and validated view evidence root. |
| Trajectory SFT dose grid | Use `configs/seedsft_dose_grid_full_train_views.json` with `scripts/prepare_seedsft_dose_grid.py`; seed 0, K30/100/300/500/1000/2000 are registered. The pass/token schedule and legacy grid stay unchanged. `support_selection_rule` distinguishes new K100 from old K100. |
| FTP collector / SmartAD collector task lists | Exactly the ordered view instances, with original parent episode IDs. Published new executions preserve selection identity and bind the original view in `identity.support_view`; loading validates both the new evidence and original source prefix. Keep that relative source-view path available when relocating derived collections. |

SmartAD training still requires its collect/select stages and initial-student
binding; a raw seed view alone is not a selected SmartAD collection. For v2,
collector `--subset-k` is refused: create each support view from the full source,
then collect/select on that view. The old K100 `--subset-k` workflow remains
unchanged. Fixed cell launchers (`rai_collect_adftp.sh`,
`rai_collect_smartad.sh`, `hpg_baseline_cells.slurm`) still enumerate the legacy
cells; use the Python pipelines' explicit `--verified` argument for this grid.
These launchers are not generic view-selection interfaces.

No new rule is applied retroactively to `stratified_groups_v1`, its views, or
`all_train_v1`. View accounting is reuse only (`reused_verified_nested_collection`,
zero new teacher calls/environment steps/training exposure); never sum shared
source costs across views. This is support registration and engineering
validation, not a scientific result or a baseline reproduction claim.

## Validation

414 distinct tests selected: **413 passed, 1 skipped**, including reruns.
The new view and dose-grid files passed all 59 focused tests. Existing coverage:
`test_stage2b_seeds`, `test_stage2c_seeds`, `test_stage2f`, `test_full_train`,
`test_full_train_recovery`, `test_adftp`, `test_smartad`, `test_smartad_relocation`,
`test_smartad_shards`, `test_baseline_k_ladder`, `test_sad`, `test_stage3a_torch`,
`test_stage3d`, `test_hash_parallel`, and `test_adftp_sad_launchers`.

All runs used `CUDA_VISIBLE_DEVICES=''` and the test suite's network/API block.
The project `.venv` lacks torch; SAD/numerical checks used the existing
`tc-alignment-taskeq/.venv` (torch 2.13.0, transformers 5.14.1, peft 0.20.0).
LEAP used the main checkout's local upstream templates at the registered
`a3376d89a2eeee88f03555a4174a0bdf7faf8be6` pin; the ETO numerical comparison used
a temporary link to its local upstream reference file, removed afterward.
No downloads or dependency changes. One extension test initially encountered
the expected code-hash resume guard during editing and passed with stable files.
The sole final skip requires `data/verified/seeds_k20_s0`, absent in this worktree.
