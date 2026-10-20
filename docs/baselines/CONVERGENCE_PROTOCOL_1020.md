# Main-table convergence protocol — 2026-10-20

User decision: a cell is converged when doubling its training amount lowers
its last-complete-epoch mean training loss by **less than 0.01 nats per supervised
token**. For adjacent rungs, compute

```
L_last(r) = sum(last-complete-epoch supervised loss numerators) / sum(supervised tokens)
delta = L_last(r-1) - L_last(r)
converged = delta < 0.01
```

Report the **first converged rung r's endpoint**, not the previous rung, a
minimum-loss checkpoint, or the best evaluation checkpoint. The inequality
is strict; zero or negative improvement qualifies under this literal rule.
Do not take an absolute value. A cap reached without convergence is recorded
as unconverged. Training loss is measured during the actual updates, before
each optimizer step; it is not a new evaluation pass over training data.
**Dev, valid, and test data never enter this decision.** Evaluation is separate
and cannot select a rung. No scientific convergence claim follows from the
CPU plumbing tests or a smoke run.

| Method | Training amount / CLI | Rungs | Paper setting | Slurm override |
| --- | --- | --- | --- | --- |
| AD (Kang) | `--epochs E` | 1, 2, 4, 8, 16, 32, … | 2 epochs | `KANG_EPOCHS` |
| AD-FTP | `--epochs E` | 1, 2, 4, 8, 16, 32, … | 2 epochs | `ADFTP_EPOCHS` |
| SmartAD | `--epochs E` | 1, 2, 4, 8, 16, 32, … | 2 epochs | `SMARTAD_EPOCHS` |
| SAD | `--budget-multiplier m` | 0.5, 1, 2, 4, … | 655,000,000 processed tokens × 1 | `SAD_BUDGET_MULT` |
| Trajectory SFT | `--passes-ladder --max-passes M` | 1, 2, 4, 8, 16, 32, … | Existing fixed-pass recipe remains available | `TRAIN_PASSES_LADDER=1`, `TRAIN_MAX_PASSES` |

AD, AD-FTP and SmartAD start **fresh from the same configured base** for each
rung. Their linear schedule uses the rung's total steps; warmup remains the
paper's explicit zero steps. SAD also restarts each rung: cosine decay and
the 0.03 warmup ratio use its new total steps (warmup rounds up). Its
1/16, 1/8, 1/4, 1/2 and 1/1 checkpoints are fractions of the **new** budget.
Only the full-budget endpoint is the rung endpoint. Budget rounding to a
complete optimizer batch and any overshoot remain disclosed. SAD's final
epoch may be partial. Use the last **complete** epoch for `L_last`: the final
epoch if complete, otherwise the previous epoch. A partial epoch contains only
the simplest trajectories under SAD's ascending-complexity curriculum and
biases the loss downward. Refuse the comparison if no complete epoch exists.
SFT and SmartAD must finish a complete final epoch; otherwise refuse.

All other settings, verified collection/selection, seed, and student snapshot
stay fixed within a cell. Changing epoch count changes the frozen recipe;
changing SAD's multiplier changes its frozen processed-token budget. Use a
separate output tree per rung. Defaults keep the previous recipes and
optimization behavior; they do not automatically enable convergence runs.
Retries, extra passes, and rung restarts are training exposure, not new K,
unique instances, or task groups. Preserve failed attempts and all provenance.

## Loss accounting and comparison

The baseline trainer logs `supervised_loss_sum` before the optimizer's
trajectory/weight normalization. SFT uses summed supervised CE; SAD uses
summed span CE; SmartAD uses **sum(weight × CE)** divided by the count of
supervised tokens for this convergence statistic (not the sum of weights).
The training objective and gradients are unchanged. Distributed zero-weight
padding is excluded. Per-update counts come from differences of cumulative
`supervised_tokens_seen`. Weighting by tokens avoids an unweighted average of
update losses or a last-update-only approximation.

Completeness requires each training example index exactly once in the epoch.
The checker verifies per-update `indices` and available `global_batch` and
`examples_seen` counts against `N_examples` in provenance and/or the length of
`example_sources.json` (both must agree when present). Missing coverage evidence,
duplicate/out-of-range indices, or incomplete nonfinal epochs are refused.

Run the checker on the actual trainer output directories containing
`updates.jsonl`, not their pipeline parents:

```bash
python -m tools.convergence_check \
  reports/runs/ad-e1/stages/train-sft/attempt-000/output \
  reports/runs/ad-e2/stages/train-sft/attempt-000/output
```

It prints JSON and writes `convergence.json` in the second directory (override
with `--output PATH`). Fields include `L_last_prev`, `L_last`, `delta`,
`converged`, `rule`, and hashes of every input ledger/metadata file read. It
checks that amounts double and rejects known incompatible settings. Compare
rungs in ascending order and stop at the first success. SAD requires its
processed-budget metadata in `provenance.json` or `result.json`.
Each rung records the zero-based `final_epoch` and `selected_epoch`, whether the
final epoch is complete, and the selected epoch's `updates` and `supervised_tokens`.

Legacy SAD logs can recover the numerator as `loss × global_batch`. Legacy
SFT logs are exact when the global batch has one trajectory. Larger legacy
SFT batches and legacy SmartAD logs need the new numerator: the checker
rejects insufficient logs instead of treating a trajectory/weight mean as
a supervised-token mean. Legacy trajectory logs are supported when their
updates do not straddle pass boundaries; otherwise exact last-pass attribution
is unavailable and the checker fails explicitly. Never replace missing values
with dev/valid loss or fabricate a convergence result.

## One trajectory-SFT cell

The default trajectory recipe is constant LR `1e-5`, LoRA rank 16, and episode
packing. Ladder mode flushes the pending optimizer batch at **every** pass
boundary and saves at 1, 2, 4, 8, … passes. This boundary rule is independent
of the cap: a longer ladder has the identical update prefix of a shorter one.
No episode is split. This opt-in boundary rule leaves the legacy non-ladder
batch plan unchanged. No scheduler is restarted at a rung; optimizer state and
the deterministic data stream continue. Warmup options and separate token/pass
limits are rejected in ladder mode. The cap must be a power of two, at least 2.

At 2P, compare only the last pass at P and the last pass at 2P, not the whole
P-to-2P interval. Save and report 2P when converged. Decisions and both losses
are in checkpoint `training_state.json`, run `result.json`, and
`convergence_provenance.json`. The outcome provenance binds the frozen run
identity and update-ledger hash without rewriting the original provenance.
The frozen exposure plan describes the cap; the ledger records actual exposure.
When the cap is reached unconverged, `reported_checkpoint` is null; saved
checkpoints remain available but are not advertised as converged.

```bash
PYTHONPATH=src:. "$TRAIN_PYTHON" train/sft.py \
  --snapshot "$MODEL_PATH" \
  --verified data/verified/seeds_k20_s0 \
  --output reports/runs/seedsft-k20s0-convergence \
  --packing episode --learning-rate 1e-5 --rank 16 --seed 0 \
  --passes-ladder --converge-delta 0.01 --max-passes 32

sbatch --export=ALL,VERIFIED=data/verified/seeds_k20_s0,TRAIN_OUT=reports/runs/seedsft-k20s0-convergence,TRAIN_PASSES_LADDER=1,TRAIN_MAX_PASSES=32,TRAIN_CONVERGE_DELTA=0.01 \
  scripts/hpg_train_sft.slurm
```

The wrapper's ladder mode replaces legacy `TRAIN_PASSES`, `TRAIN_TOKENS`, and
`CHECKPOINT_TOKENS`; `VERIFIED` selects the frozen verified training view and
`TRAIN_OUT` must be a new output directory. No jobs are submitted by the checker.
