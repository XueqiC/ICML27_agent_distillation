# V0 expanded-dataset parent binding validation

Worktree: `/home/xueqi/hq/projects/ICML27_wt_repair_train`, HEAD `39a4b6c`
plus the unstaged changes accompanying this report. No commits, staging,
GPU execution, teacher calls, or real-model updates.

The expanded dataset seals its parent's manifest and examples in `inputs`.
The parent manifest's raw SHA256 is not the consumer's composite
`dataset_sha256`. The reader verifies the sealed parent with the V0 reader,
recomputes that composite identity, and checks unchanged inherited rows.
Weights and module masks must match the current adapter and parameter names.

D-data uses the parent `groups.json`, checks the original weight formula,
and resolves new G episodes through their task's allocated unit to its
gradient group. The unit's source repair must belong to that parent group.
Allocation, execution, generated-task, and synthetic-manifest records must
agree. The existing `episode_weights` formula normalizes R+G together;
S/D remain at 1. No G gradients are computed this round. Parent-bound
D-param selections do not require G-weight mappings.

## Validation

All tests use fake clients/environments and the existing test fixture's
network/API prohibition. The interpreter is
`/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python`.
Environment: `CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH=src:.`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`PYTHONDONTWRITEBYTECODE=1`; test runs also set
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.

V0 suite: **112 passed**, including 21 new tests. The suite covers parent
weights/modules, unrelated bindings, adapter/parameter/smoke guards, missing
or tampered parents, changed inherited rows, unresolved/conflicting G units
and groups, per-episode normalization with unequal trajectory lengths,
neutral excluded gradients, relocation, and persisted check/run provenance.

```bash
python -m pytest -q tests/test_v0_training.py \
  tests/test_v0_producer_contract.py tests/test_v0_expansion_weights.py
python -m pytest -q
```

V0 log: `logs/v0_parent_binding/v0_tests.log`.
The 57 cached pre-run failure IDs were preserved in
`logs/v0_parent_binding/pre_full_lastfailed.json` for comparison with the full
suite.

Full CPU suite: **3,249 passed, 57 failed, 6 skipped**, 180 warnings,
2,778.29 seconds (46:18), exit 1. Compared with the supplied
**3,228 passed / 57 failed** baseline, this adds **21 passing tests**.
The final 57 failing test IDs exactly match the saved pre-run list:
**no new or resolved failures**. These existing failures were not modified.
The V0 suite completed in 109.03 seconds.

Full log: `logs/v0_parent_binding/full_cpu_tests.log`.
Exact failure IDs, per-file counts, and comparison:
`logs/v0_parent_binding/failure_comparison.json`.
The logs/metadata are ignored local artifacts; source, tests, and this report
remain unstaged.

## Real CPU check

**Passed, exit 0, status `verified`.** Output including all 166 G inheritance
records: `logs/v0_parent_binding/real_inputs.json`; stderr is empty.

| Bucket | Rows | Episodes | Unique instances | Task groups |
| --- | ---: | ---: | ---: | ---: |
| R | 2,915 | 134 | 82 | 82 |
| G | 3,221 | 166 | 85 | 85 |
| S | 2,616 | 175 | 175 | 175 |
| D | 20 | 20 | 20 | 20 |
| Total / deduplicated union | 8,772 | 495 | 308 | 308 |

Row/episode counts come from the verified reader. The separately recomputed
identity inventory is saved in `logs/v0_parent_binding/independent_inventory.json`.
R/S identities overlap; their per-bucket unique counts must not be summed.
Replay validation deduplicates 40 input rows to the 20 original seed rows.

G weight minimum / mean / maximum:
`0.010375125529134995 / 1.013197700683633 / 1.0701588712931984`.
R mean: `0.9836506096008724`; combined R+G episode mean: **1.0**.
S and D weights are all **1.0**. Independent calculations from the original
group and allocation records agree within floating-point rounding.

| Inherited gradient group | G episodes | Normalized weight |
| --- | ---: | ---: |
| group-01 | 5 | 0.02563683377074564 |
| group-02 | 157 | 1.0701588712931984 |
| group-03 | 3 | 0.010375125529134995 |
| group-04 | 1 | 0.016565975009806364 |

Verified identities:

- Parent manifest: `08aea691d4229f9278529e28e67643cf7d64140c87fbcfdc48707af79cdc37aa`
- Parent dataset: `0bacaee984f071eeba09997209e5bc02f48ffc4b780b740a02f1e978f5335d2e`
- Expanded dataset: `2a37f5e77bad4bc4283254523a7304c296cfb0a19709cd000a6b2a53e875fa07`
- Adapter: `6a06a2321747fd3707bd20c8d1232b18fa754d8736b6983ddd062ee25dde354a`
- Effective weights: `546bd510ddb65b924a6264bfe5b0c9e982a107849a8ed5eee01cfe0771e55231`

Code hashes and dependency versions are saved in
`logs/v0_parent_binding/validation_metadata.json` (Python 3.12.13,
pytest 7.4.0, torch 2.13.0, transformers 5.14.1, peft 0.20.0,
numpy 2.5.2, safetensors 0.8.0).

The requested command uses these read-only inputs:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
SED_HASH_CACHE_DIR=/home/xueqi/hq/projects/ICML27_v0_runs/sed-hash-receipts \
/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python \
  tools/train_v0_arm.py inputs --check \
  --dataset /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1/expand-D-data/dataset \
  --theta0 /home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint \
  --snapshot /home/xueqi/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --config configs/base_e4b.yaml \
  --synthetic-manifest /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1/expand-D-data/synth/manifest.json \
  --relocate /ddnB/work/xueqic/hq/ICML27_data=/home/xueqi/hq/projects/ICML27_agent_distillation/data \
  --weights /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1/grad_groups/weights.json
```

This validates hashes, identities, row contracts, and sampling inputs. It does
not replay environments, establish a scientific result, or reproduce a
baseline's performance. K counts the 20 trusted executed seed trajectories;
expanded episodes, unique task instances, and task groups are separate counts.
