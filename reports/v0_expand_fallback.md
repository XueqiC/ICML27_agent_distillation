# V0 expansion exhaustion repair

Worktree: `/home/xueqi/hq/projects/ICML27_wt_repair_eval`, HEAD `7467640`.
Changes are unstaged; no git metadata writes, GPU or live teacher service is used.

The original read-only round-1 D-param expansion requested 85 units and filled
83. Units 68 (`search_incomplete_or_loop`) and 82 (`wrong_object_or_target`)
exhausted the G2/mid/`look_at_obj_in_light` edit kind after excluding the initial
training items and probes. D-data's original 85/85 result is not changed.

The allocation still assigns exactly the same units to the same categories or
groups. The plan now includes every eligible verified repair in each group,
including repairs not initially assigned a unit. After searching all original
K20 seeds for the assigned edit kind, generation tries the other repairs in
episode-ID order, using each repair's own edit kind and preferred seed.
`allocation_execution.json` records the original unit, `fallback_from`,
`fallback_to`, and each source's ordered seed search. Generated items separately
record their effective `expansion_source`; their parents remain original seeds.
The existing fingerprint exclusion set covers all initial items, probes, and
items already generated in this expansion.

If all repairs in the category/group are exhausted, the unit is recorded as
`unfilled_after_fallback` and subsequent units continue. `--strict-fill` fails
after preserving generation/exhaustion evidence when a shortfall remains. It
defaults off and is bound into phase identity and worker requests. Summary
`fallback_count` counts successful fallback fills; `fallback_attempted_count`
also includes unsuccessful fallback searches. `unfilled_count` reports the
remaining shortfall.

`--dry-run` runs actual CPU generation and environment validation, stopping
before student attempts, review, labeling, or dataset construction. Fill policy
and dry-run mode are recorded, so the incident output cannot be silently resumed
under changed code or policy.

## Validation

Interpreter: `/home/xueqi/hq/projects/ICML27_agent_distillation/.venv/bin/python`.
Tests use `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH=src:.`, with the repository's network/API-blocking fake fixtures.

- Targeted expansion/pipeline tests: **14 passed**.
- V0 suite (`-m pytest -q tests/test_v0*`): **135 passed** in 56.69 seconds.
- Full CPU suite (`-m pytest -q`): **2,790 passed / 51 failed / 102 skipped**
  in 2,209.59 seconds (36m49s). Compared with **2,780 passed / 51 failed**,
  this adds **10 passing tests**. All 51 failure IDs exactly match the saved
  baseline; there are no new or resolved failures. The full-suite exit status
  is 1 because those baseline failures remain.

The failure-ID comparison uses the baseline referenced by
`reports/v0_label_tokenizer_fix.md`; its machine-readable result is
[`full_cpu_comparison.json`](../spike_out/v0_expand_fallback/full_cpu_comparison.json).
`git diff --check` passed. HEAD remains `7467640`, and the index is empty.

The ten additional cases cover deterministic same-category fallback, exhausting
all seeds before changing source, initial/probe/generated fingerprint exclusion,
all-exhausted continuation, strict CLI failure with durable evidence, both
allocation variants' complete eligible-source lists, CLI defaults, and a
zero-generated-item round that builds the inherited dataset without attempting
the original tasks again.

## Real allocation check

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /home/xueqi/hq/projects/ICML27_agent_distillation/.venv/bin/python \
  -m tools.v0_round expand \
  --output spike_out/v0_expand_fallback/real_dry_run \
  --dataset /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1/dataset \
  --attempts /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1/attempt \
  --synthetic-manifest /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1/synth/manifest.json \
  --seeds /home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20/manifest.json \
  --data-root /home/xueqi/hq/projects/tc-alignment/envs/alfworld/data/json_2.1.1 \
  --main-checkout /home/xueqi/hq/projects/ICML27_wt_repair_eval \
  --variant D-param --dry-run
```

Result: **85 requested, 85 generated, 2 successful fallbacks, 0 unfilled**.
All original allocation fields, including all 85 units, were reproduced exactly.
All 83 previously filled units also retained their original fingerprints. The
new plan adds the eligible repair lists and fallback rule only. This is an
allocation/generation check, not student evaluation or evidence of a scientific
finding.

| Unit | Unchanged category | Fresh generated task |
| --- | --- | --- |
| 68 | `search_incomplete_or_loop` | `train/synth/75aa6a3e5fd47fcb79550327` |
| 82 | `wrong_object_or_target` | `train/synth/e7de7d36c32bd9a2d553a28f` |

Both exhausted source restrictions were G2/mid/`look_at_obj_in_light`. Both
fallbacks selected `repair-001669413c984c80e852fc67165f01f8`, which is eligible
in each respective category, using its own G2/mid/`pick_clean_then_place_in_recep`
edit kind. The original source episodes and restrictions remain in each unit's
`fallback_from`; the replacement is explicit in `fallback_to`.
Both actual parents are the original K20 seed
`train/pick_clean_then_place_in_recep-ButterKnife-None-Drawer-2/trial_T20190908_044255_995925`.
Each new candidate won both validation executions with matching deterministic
traces; their recorded validation costs are 28 and 17 transitions respectively.

The final audit passed: 277 initial/probe fingerprints excluded, 85 unique
generated fingerprints, original K20 seed parents only, unchanged input and
incident artifact hashes, and no student-attempt/review/label/dataset phase or
teacher ledger. The generated items span 76 distinct problem hashes and 66 task
groups; these are separate from the 20 trusted seed trajectories. The new CPU
validation journals, including the preserved partial initial run, record 1,049
started and 1,049 completed transitions. Student calls, teacher calls and training
exposure are all zero.

Machine-readable checks and full fallback entries:
[`dry_check.json`](../spike_out/v0_expand_fallback/dry_check.json).
Recheck with `PYTHONPATH=src:. /home/xueqi/hq/projects/ICML27_agent_distillation/.venv/bin/python
spike_out/v0_expand_fallback/check_result.py`.

The initial uncached replay was interrupted to avoid repeating all historical
environment validations. Its partial transition receipts are preserved under
`synth/interrupted_validation/`. The check then reused 451 exact, hash-verified
candidate receipts from the original incident in the scratch cache, including
all 83 previously accepted candidates. It omitted 18 cached duplicate-state
rejections whose validity depended on the old expansion's generated set.
Frozen initial/probe duplicate rejections remain valid. No receipt was edited,
and new or omitted candidates use actual CPU environment validation. Validation
code hashes and dependency versions match the incident; only the expansion and
round CLI files differ. Cache
preparation is reproducible with `spike_out/v0_expand_fallback/prepare_cache.py`;
the same expansion command then resumes with `--resume`. `synth/cache_receipt.json`
records every copied hash and omitted rejection.

Logs and generated audit artifacts: `spike_out/v0_expand_fallback/`.

## Unstaged changes

- `repair/v0_expand.py`
- `tools/v0_round.py`
- `tests/test_v0_expand.py` (new)
- `tests/test_v0_pipeline.py`
- `repair/README.md`
- `reports/v0_expand_fallback.md` (new)

The ignored scratch directory `spike_out/v0_expand_fallback/` contains the
generation outputs, validation receipts, audit/reproduction scripts and test
logs. No original incident files were changed, and nothing was staged or
committed.
