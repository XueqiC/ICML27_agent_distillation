# V0 round implementation validation — 2026-11-02

Worktree: `/home/xueqi/hq/projects/ICML27_wt_repair_eval`, base `dbfca9d`.
Only new files were added. No existing module hooks, git staging, commits,
pushes, GPU execution, vLLM launch, or real teacher/student calls were used.
Read-only source checkouts and environment games were not modified.

## CPU tests

All runs used `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=src:.` and
`/home/xueqi/hq/projects/ICML27_agent_distillation/.venv/bin/python`.
Unit tests use fake environments/students/teacher transports, including the
existing capped transport with `FakeReviewClient`; no network or OpenAI API.

| Run | Passed | Failed | Skipped | Duration |
| --- | ---: | ---: | ---: | ---: |
| Previous full-suite log | 2,650 | 51 | 102 | 1,872.36 s |
| Full suite during this task | 2,699 | 51 | 102 | 2,097.38 s |
| Final focused suite | 206 | 0 | 0 | 24.52 s |
| Final V0 tests alone | 46 | 0 | 0 | 7.20 s |

The full-suite failure-ID sets are identical: **zero new failures, zero
resolved failures**. The full-suite command exits 1 because those 51 existing
failures remain. The final focused suite covers both V0 files plus
`test_synth_tasks.py`, `test_joint.py`, and `test_supervision.py`. Two native
validation timeout/journal tests were added after full-suite collection;
both pass in the final focused runs. Full-suite and focused totals describe
their actual separate runs, not a combined synthetic pytest result.

Logs:

- Previous: `spike_out/full_cpu_tests.log`
- Full: `spike_out/v0_work_logs/full_cpu_tests.log`
- Exact failure-set comparison: `spike_out/v0_work_logs/full_cpu_comparison.json`
- Final focused: `spike_out/v0_work_logs/final_focused_cpu_tests.log`
- Final V0: `spike_out/v0_work_logs/v0_validation_tests.log`

Covered behaviors include bin quotas and fallback, probe separation, seed-only
parents, G5 executed validation, manifest authorization, G3 prefix replay and
masking, strict whole-review JSON and one retry, suffix-before-handback order,
three-round limit, actual won-only success, private-state prompt exclusion,
prefix-only rationale labeling, refused-action exclusions, executor-derived
edit flags, exact probe intervals, full fake pipeline/resume, immutable-input
expansion, same-kind fresh seed edits, and fail-loud shared ledger caps.

## Completed real-environment synthesis

The final run completed successfully in `spike_out/v0_round_dryrun` using:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /home/xueqi/hq/projects/ICML27_agent_distillation/.venv/bin/python \
  -m tools.v0_round synth \
  --seeds /home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20/manifest.json \
  --data-root /home/xueqi/hq/projects/tc-alignment/envs/alfworld/data/json_2.1.1 \
  --main-checkout /home/xueqi/hq/projects/ICML27_agent_distillation \
  --output spike_out/v0_round_dryrun --workers 4
```

All 20 original seed families completed. There are **206 training items and
71 held-out probes**:

| Bin | Training by generator | Training total | Probes by generator | Probe total |
| --- | --- | ---: | --- | ---: |
| near | G1: 28; G3: 38 | 66 | G1: 10; G3: 1 | 11 |
| mid | G2: 80 | 80 | G2: 20 | 20 |
| far | G2: 40 | 40 | G2: 20 | 20 |
| very_far | G5: 20 | 20 | G5: 20 | 20 |
| Total | | 206 | | 71 |

Across both splits: G1=38, G2=160, G3=39, G5=40. Six near slots used the
recorded alternate-generator fallback. The frozen native held-out group rule
excludes near candidates for nine seed families, leaving **54 training slots
and 9 probe slots unfilled**. The requested 80-probe target was therefore not
achievable under that freeze. Every other bin filled its allocations. Missing
probe parents are enumerated in `real_synth_audit.json`; no substitute parents
or overlapping probes were introduced.

K remains **20 trusted executed seed trajectories**, with 20 unique seed
instances and 20 seed task groups. The output has 277 distinct item/recovery
fingerprints: **238 new executable games and 39 recovery situations**. It has
249 distinct initial-problem fingerprints and 187 native semantic task groups.
The 277 synthetic `group_id` values describe effective state/goal or recovery
identities, not 277 distinct native semantic task groups.

There were 1,333 candidate records, of which 1,056 were rejected:

| Rejection reason | Count |
| --- | ---: |
| `heldout_group` | 879 |
| `already_won_at_reset` | 120 |
| `ambiguous_location` | 35 |
| `validation_timeout_unknown_feasibility` | 11 |
| `duplicate_effective_state` | 5 |
| `validation_exception:KeyError` | 3 |
| `no_witness_within_40_steps` | 2 |
| `no_controlled_error_opportunity` | 1 |

The durable journals charge **6,415 environment transitions**: 6,414 confirmed
and one with unknown outcome. The 11 timeouts are unknown feasibility, not
unsolvability claims. Teacher calls/tokens, student calls and training exposure
are all zero. Accepted candidates passed the actual witness and repeatability
checks; this smoke run establishes software behavior, not a scientific result.

The synthetic manifest is `spike_out/v0_round_dryrun/synth/manifest.json`, with
raw-byte SHA256:

```text
40d2db154f91033d853ca8749793f85e4e6e205770261de946f784bc683d77e0
```

Its environment root is `spike_out/v0_round_dryrun/synth/data_root`.
The completed phase seal, every published game/prefix/witness hash, original
K20 parent membership, quotas, diagnostic distances, and training/probe
fingerprint/group separation were checked. Repeating the exact command with
`--resume` exited 0 and preserved the manifest, completion seal, artifact
hashes, statistics and environment-step counts unchanged.

Artifacts and logs:

- `spike_out/v0_round_dryrun/synth/{manifest,stats,allocation,attempts,completion}.json`
- `spike_out/v0_round_dryrun/synth/parents/` — private candidate, witness and transition receipts.
- `spike_out/v0_work_logs/real_synth_bounded.log`
- `spike_out/v0_work_logs/real_synth_resume.log`
- `spike_out/v0_work_logs/real_synth_audit.json`

## Interrupted first real-environment attempt

The first run reached 18 completed seed families before two native planner
searches stalled. It was interrupted, retained under
`spike_out/v0_round_unbounded_attempt`, and marked `never_for_training` in
`interruption.json`. Its 1,317 completed candidate records account for 6,278
environment transitions. Additional in-flight validation cost is unknown:
that earlier implementation did not have the durable transition journal.
These are repeated executions of the same K20 seeds, not an increase in K.

The final implementation runs each native validation in a separate killable
process with a 120-second limit, recording actions before execution and their
outcomes afterward. Timed-out candidates retain their private partial journals
and are classified as unknown feasibility. The fresh final run uses this
implementation; it does not import training items from the interrupted run.
Candidate order, identities and allocations are seeded. A wall-time limit can
change candidate availability under different machine load; bitwise identical
accepted manifests across such runtime differences are not claimed.

## New files

- `tools/v0_round.py` — all seven CLI subcommands.
- `tools/v0_round_rai.sh` — served-adapter launcher example.
- `tools/V0_ROUND.md` — protocol, usage, and interchange details.
- `tools/V0_ROUND_VALIDATION.md` — this validation report.
- `repair/v0_common.py` — phase seals, provenance, authorization.
- `repair/v0_synth.py` — stratified seed-only synthesis and G5.
- `repair/v0_validation.py` — bounded native validation and transition journals.
- `repair/v0_execution.py` — shared-executor attempts and probe statistics.
- `repair/v0_review.py` — capped teacher transport and closed-loop repair.
- `repair/v0_data.py` — rationale decisions, buckets, and V2 replay publication.
- `repair/v0_expand.py` — allocation, original-seed expansion, new datasets.
- `tests/test_v0_round.py` — component and contract tests.
- `tests/test_v0_pipeline.py` — fake end-to-end and expansion tests.

## Interchange limitation

The inspected sibling train contract requires exactly five `v0` fields and
only G1/G2/G3. The explicit current request requires `bin` and G5. This producer
follows that newer request and preserves the V2 replay bundle unchanged;
the older sibling reader must be updated before consuming these datasets.
No compatibility with that older reader is claimed, and no sibling files
were changed. See `V0_ROUND.md` for the precise disk contract.
