V0 round orchestration now accepts K=10/20/30 from the verified seed manifest. K is checked against the trajectory inventory and recorded in the round, synthesis, and dataset manifests. Serving checks bind K to the registered theta0 cell, final training state, and verified seed view. The launcher honors `MODEL_NAME`, with the original K20 default.

Per seed, allocation remains near/mid/far/very-far = 6/4/2/1 training slots plus one held-out probe per bin. Attempt decoding, H40, review limits, token caps, exclusions, labels, and expansion allocation rules are unchanged. Expansion requests one item per distinct initially failed training item. Replay contains exactly K verified trajectories; `k20_replay` remains the legacy V2 interchange source tag for compatibility. K20 still uses its original V2 replay builder; K10/K30 use the same trajectory renderer and checked seed receipts without depending on the K20-only V2 training pool.

The latest results across the documented test runs are **946 passing checks: 210 V0, 311 P0, and 425 related regressions**. Tests used `/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python`, `CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH=src:.`, and offline fixtures that block network/API use. See [test_summary.json](test_summary.json) and the XML/log files. The initial run's two obsolete CLI fixtures were fixed; a P0 code-hash race during concurrent edits was rerun successfully. All earlier failure logs remain available. Shared adapter/task evaluation was also rerun after concurrent changes: 88 passed.

The broad test command was:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest \
  tests/test_v0_*.py tests/test_p0_*.py tests/test_synth_tasks.py \
  tests/test_supervision.py tests/test_build_v2_label_*.py \
  tests/test_subset_v2_dataset.py tests/test_v2_task_eval.py \
  tests/test_repair_launchers.py tests/test_joint_adapter_eval.py tests/test_lora_serving.py
```

K20 compatibility was checked against the read-only real round:

- All 18 active phase seals and all three datasets, their replay bundles, and synthetic assets validate. See [legacy_validation.json](legacy_validation.json).
- Re-running the candidate factory, allocation, and publication against the real seed inputs and copied validation receipts reproduced `manifest.json`, `attempts.json`, `allocation.json`, and `split_freeze.json` byte for byte. No new K20 planner executions or model calls were made. The manifest SHA-256 remains **`4df53e36906111105035aa0bfdcecf153de766b65efc7cae60fb00b1763cfd00`**.
- Round identity and synthesis scientific plan fields match. Code hashes, creation times, and dependency provenance remain truthful for new runs. The historical D-data expansion predates fallback metadata already present before this task; all its recorded allocation fields match, as does the current D-param plan.
- This worktree and the original K20 checkout contain identical exclusion inputs and the same 923 excluded groups. Eight source paths differ because the worktree directory name changed. For exact K20 reproduction across checkouts, supply `--split-freeze /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1/synth/split_freeze.json`. The option checks the complete exclusion set, every source content hash, and the original referenced files before retaining the original paths. It cannot weaken exclusions. See [freeze_comparison.json](freeze_comparison.json).

Both real CPU synthesis runs completed with four workers, native Fast Downward witnesses, and actual reset/replay validation. Every accepted asset hash and phase seal validates; no accepted task overlaps the frozen excluded groups. Counts are:

| Bin | K10 training | K10 probes | K30 training | K30 probes |
|---|---:|---:|---:|---:|
| near | 36 | 6 | 108 | 18 |
| mid | 40 | 10 | 120 | 30 |
| far | 20 | 10 | 60 | 30 |
| very_far | 10 | 10 | 30 | 30 |
| **Total** | **106** | **36** | **318** | **108** |

The 28 K10 and 84 K30 unfilled slots all arise from the unchanged frozen-group rule: four and twelve seed groups respectively cannot supply near-bin slots, each with six training slots and one probe. Slots were not reassigned to other bins. K20's existing reference remains 206 training items and 71 probes.

| Accounting | K10 | K30 |
|---|---:|---:|
| Trusted executed seed trajectories (K) | 10 | 30 |
| Unique seed instances | 10 | 30 |
| Seed task groups | 10 | 30 |
| Accepted new executable games | 118 | 366 |
| Accepted recovery situations | 24 | 60 |
| Environment transitions during synthesis | 3,280 | 10,387 |
| Teacher/student calls, teacher tokens, training exposure | 0 | 0 |

All rejection reasons are retained below. Counts are rejection occurrences, not independent tasks. Planner timeouts remain unknown feasibility.

| Rejection reason | K10 | K30 |
|---|---:|---:|
| `already_won_at_reset` | 78 | 134 |
| `ambiguous_location` | 32 | 50 |
| `duplicate_effective_state` | 5 | 5 |
| `heldout_group` | 332 | 1,079 |
| `no_controlled_error_opportunity` | 1 | 1 |
| `no_witness_within_40_steps` | 2 | 4 |
| `validation_exception:KeyError` | 2 | 4 |
| `validation_timeout_unknown_feasibility` | 0 | 16 |

The completed outputs are [K10](../../spike_out/v0_k_generalization/_dryrun_k10/) and [K30](../../spike_out/v0_k_generalization/_dryrun_k30/). Their manifests have SHA-256 `cfaaaa1034a71247941d50862964aeae552c85280b69b2a588e1997874e4d0c7` and `41725e1145d0e9e88f45c0d2e85452268d0ec0e461b2d62bd50a0894e244231c`, respectively. Full verification and accounting: [dryrun_summary.json](dryrun_summary.json), [K10 validation](dryrun_k10_validation.json), [K30 validation](dryrun_k30_validation.json). Candidate receipts, rejected attempts, witnesses, allocations, dependency versions, seed/source identities, code hashes, and frozen exclusions remain in each run directory.

The requested `/home/xueqi/hq/projects/ICML27_v0_runs/_dryrun_k{10,30}` directories are outside this session's writable mount. Creating `_dryrun_k10` returned `Read-only file system`; the real dry-runs therefore use `spike_out/v0_k_generalization/_dryrun_k{10,30}` in this worktree.

The specified test interpreter has no `alfworld` or `textworld`. Its initial K10 launch was interrupted and retained under `spike_out/v0_k_generalization/_dryrun_k10_missing_runtime`, with its original log in [synth_k10_missing_runtime.log](synth_k10_missing_runtime.log). No completed synthesis was published from that launch. Its 1,982 retained candidate receipts comprise 1,787 missing-module errors, 157 frozen-group exclusions, and 38 ambiguous-location rejections, with zero environment steps; see [runtime_failure_summary.json](runtime_failure_summary.json). These setup failures are separate from the completed native runs. Real synthesis uses `/home/xueqi/hq/projects/ICML27_agent_distillation/.venv/bin/python`; all recorded dependency versions match K20 exactly, including ALFWorld 0.4.2 and TextWorld 1.7.0. A separate offline native-planner check executed three successful steps on a K10 seed; its full evidence is in [native_runtime_check.json](native_runtime_check.json). See [runtime_versions.json](runtime_versions.json).

Exact rai launch commands are implemented in [launch_rai.sh](launch_rai.sh). They use these roots and adapters:

| K | Round root | Theta0 cell |
|---|---|---|
| 10 | `/home/xueqi/hq/projects/ICML27_v0_runs/v0_k10_r1` | `e4b-tsft-k10-e32-lr5e-5` |
| 30 | `/home/xueqi/hq/projects/ICML27_v0_runs/v0_k30_r1` | `e4b-tsft-k30-e64-lr5e-5` |

Both adapters resolve under `/home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/baselines_hpg/<cell>/stages/train-sft/attempt-000/output/checkpoint`. These commands were **not launched**. The script reuses the existing rai environment/cache configuration, defaults serving to GPU 1, uses 8 rollout/review workers and 16 label workers, and supplies the pinned local tokenizer to labels and expansion. Synthesis uses four CPU workers. Set `GPU`, `WORKERS`, and `LABEL_WORKERS` explicitly to override scheduling.

From this worktree, run each stage when its inputs are ready:

```bash
# S: fresh native CPU synthesis, with the original frozen exclusions.
bash reports/v0_k_generalization/launch_rai.sh 10 synth
bash reports/v0_k_generalization/launch_rai.sh 30 synth

# A: four attempts per training item.
bash reports/v0_k_generalization/launch_rai.sh 10 attempt
bash reports/v0_k_generalization/launch_rai.sh 30 attempt

# Held-out probes, excluded from review/labels/training.
bash reports/v0_k_generalization/launch_rai.sh 10 probe
bash reports/v0_k_generalization/launch_rai.sh 30 probe

# Optional request-size/budget inspection; no teacher calls.
bash reports/v0_k_generalization/launch_rai.sh 10 review-dryrun
bash reports/v0_k_generalization/launch_rai.sh 30 review-dryrun

# B: actual review and verified repair.
bash reports/v0_k_generalization/launch_rai.sh 10 review
bash reports/v0_k_generalization/launch_rai.sh 30 review

# Prefix-consistent rationale labels, then R/S/D dataset construction.
bash reports/v0_k_generalization/launch_rai.sh 10 label
bash reports/v0_k_generalization/launch_rai.sh 30 label
bash reports/v0_k_generalization/launch_rai.sh 10 build
bash reports/v0_k_generalization/launch_rai.sh 30 build

# Run the existing gradient/grouping stage before D-data expansion:
# <round>/grad_groups/groups.json and weights.json must bind <round>/dataset.
bash reports/v0_k_generalization/launch_rai.sh 10 expand-D-data
bash reports/v0_k_generalization/launch_rai.sh 30 expand-D-data
bash reports/v0_k_generalization/launch_rai.sh 10 expand-D-param
bash reports/v0_k_generalization/launch_rai.sh 30 expand-D-param

# To repeat the requested CPU-only checks at their original destinations
# from a session that can write ICML27_v0_runs:
bash reports/v0_k_generalization/launch_rai.sh 10 synth-dryrun
bash reports/v0_k_generalization/launch_rai.sh 30 synth-dryrun
```

**Live-launch prerequisite:** both supplied K10/K30 checkpoint copies lack the sibling `output/provenance.json` required by the existing LoRA serving gate. Restore the original file for each checkpoint before attempt/probe/review/expansion; its digest must match `training_state.json.identity_sha256`. No provenance was fabricated and no serving check was bypassed. The checkpoint K/seed/epoch identities and weight/config hashes were verified on CPU; see [adapter_preflight.json](adapter_preflight.json).

No GPU, live teacher/API, network, training, staging, or commit was used for this work. Concurrent adapter-evaluation/P0 changes were left intact. These checks establish implementation and execution integrity, not a scientific performance result.
