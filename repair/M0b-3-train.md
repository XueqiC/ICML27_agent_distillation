# M0b-3 training implementation

Implemented on `repair-v01-train`, training tree `2506fa9`, for continuation
of `e4b-tsft-k20-e32-lr5e-5`. No teacher request, GPU job, real pilot training,
or evaluation was performed. Offline tests establish software behavior,
not a scientific finding.

The supplied evaluation-side format is preserved in [E2-interchange.md](E2-interchange.md).
No evaluation-branch Python code is imported. The draft's new trainer, artifact
helpers, launcher outline and CPU tests provided starting material; none of
its shared-file/E-series changes was applied. The draft is removed after reuse.

## Existing recipe and explicit continuation choices

Theta0 used `scripts/hpg_kang.slurm` → `baselines.kang.pipeline.training_command`
→ `baselines.common.fsdp_train.train_lora/train_loop`. This implementation uses
that tree's encoder, shared-history scorer, loss, optimizer, and exporter.
Kang's registered `RECIPE`, `train.modeling.LORA_TARGETS`, and theta0's sealed
provenance/training state supply the parameters. Their source hashes are saved.
The user-specified K20/e32/lr5e-5 run choices are checked explicitly.

The inherited LoRA recipe is r64/alpha128/dropout0.05, text-only
q/k/v/o/gate/up/down with the original explicit E-series layer coverage.
`PeftModel.from_pretrained(..., is_trainable=True, local_files_only=True)`
continues theta0; there is no additional adapter. Actual trainable names and
element counts must exactly match its safetensors header. Frozen base storage
is bf16, adapter masters fp32, compute bf16, with non-reentrant checkpointing
and the original flex-attention settings.

AdamW remains the existing trainer's implementation: betas(.9,.999), eps1e-8,
weight_decay0, foreach=False, fused=False, clip1.0 after replicated LoRA
gradient averaging. Loss remains each episode's assistant-token mean, then the
actual global episode-batch mean. Turn packing/shared-history scoring retain
the exact native OFF prefix. No local repair mask or new normalization is used.

Overrides: 20 updates, effective batch4 episodes, seed1 (rank seeds1/2),
three repair episodes and one trusted replay per update. `random.Random(1)`
shuffles bucket slots and samples tasks/demonstration slots with replacement.
N/F have identical ordered tasks and seed references; target length cannot
change the common 80-slot order. `batches.json` records every selection;
`example_sources.json` binds indices to episode/repair/receipt identities.

**Schedule:** theta0's linear schedule ended. Restart the same linear shape
with LR5e-5 and zero warmup over 20 continuation updates in both arms. The
last update uses LR2.5e-6; the scheduler then reaches zero. Do not restore the
exhausted scheduler counter.

**Optimizer state:** restore `theta0/optimizer.pt` if present, validate group
settings/moment shapes, retain moments, and restart the schedule above. Wrapped
snapshots with named trainables also check their order and exact values. If
absent, both arms initialize AdamW identically. Unknown optimizer-named files
are refused. The normal grid exporter does not save optimizer state, so its
ordinary checkpoint uses identical reinitialization. Dry-run hashes the file
without deserializing optimizer tensors.

## Evidence and model-input boundary

The reader checks format/version, canonical content seal, adjacent source
file byte hash/seal, real/non-canary/ready status, zero training exposure,
ordered four training tasks/two probes, N/F membership, frozen TRAIN identity,
train/probe group separation, independent D_probe exclusions, and the original
seeded repair-probe split. Source/export metadata must agree.

Every target requires explicit human approval, matching episode/trajectory
IDs, review evidence hash, and unchanged public episode hash. Repair episodes
must exactly match the schema-2 whitelist. Seed episodes are projected to the
same whitelist after evidence validation. Native token arrays and arbitrary
entry/receipt/audit dictionaries never reach the encoder. Inputs are freshly
encoded by this tree's unchanged `train.dataset.episode_examples`.

Seed references must match theta0's **original support manifest hash** in its
training provenance. `load_verified_collection` and `checked_episode` validate
original execution-win receipts, retained attempts, and nested K views. Paths
cannot escape the reader's evidence root. Only D_repair_train demonstrations
are encoded. K counts trusted trajectories; unique instances, task groups,
and repeated training exposures are reported separately.

`inspect_adapter` verifies theta0's configured base repo/revision and snapshot
metadata; every checkpoint file hash must match the export. Tokenizer metadata
files, identity hash and chat-template hash must also match. As in the grid
serving gate, base revision/metadata checks do not independently certify all
frozen base weight bytes. Repeated `--path-map OLD_ROOT=LOCAL_ROOT` arguments
relocate evidence without changing hashes. Local manifest, split-freeze, and
repair-split bytes must match collection inputs. Other collector input hashes
are retained as provenance without importing its code.

## Outputs and invocation

A fresh output directory contains `provenance.json`, `manifest.json` (format
`repair_e2_training_run`), `batches.json`, and `example_sources.json`. Actual
training also writes the existing append-only `updates.jsonl`. At updates1/5/20:

```
update-01/manifest.json
update-01/provenance.json
update-01/checkpoint/adapter_config.json
update-01/checkpoint/adapter_model.safetensors
update-01/checkpoint/training_state.json
```

The grid's unchanged `export_adapter` writes each checkpoint. Per-checkpoint
manifests contain every checkpoint file hash, updates/context-token/target-token
counts, trainable parameter count/names, and optimizer/scheduler decisions.
Context tokens count actual shared-history forward positions including masked
context; target tokens count supervision once per sampled episode. Batch4 on
two ranks has no padding. Actual counts must match dry-run plans at publication.
Ranks synchronize publication, which preserves training RNG state. Checkpoint0
is the unchanged theta0 object in the receipt, not another export. All fixed
checkpoints are reported; probes never select a checkpoint.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$ENV_PYTHON" -m tools.repair_train_arm \
  --config configs/base_e4b.yaml --collection "$E2_EXPORT" \
  --theta0 "$THETA0" --snapshot "$MODEL_PATH" --arm N \
  --dry-run --output /new/planned-N

# Caller exports grid runtimes and OFF_HOME caches; each arm needs a new root.
ARM=N REPAIR_COLLECTION="$E2_EXPORT" REPAIR_OUT=/new/arm-N \
  sbatch scripts/hpg_repair_train_arm.slurm
ARM=F REPAIR_COLLECTION="$E2_EXPORT" REPAIR_OUT=/new/arm-F \
  sbatch scripts/hpg_repair_train_arm.slurm
```

`E2_EXPORT` is `e2_training_collection.json` beside finalized `collection.json`.
The launcher optionally forwards `REPAIR_PATH_MAP=OLD_ROOT=LOCAL_ROOT`; direct
CLI use supports multiple mappings. It requests B200×2/CPU8/RAM64G, preserves
caller OFF_HOME caches, follows the grid's offline runtime settings, and
refuses an existing output or symlink. It does not launch evaluation.

Dry-run loads tokenizer assets and safetensors headers, with streaming file
hashes but no model/optimizer tensor loading. It writes exact planned batches
and counts, never saved-checkpoint or actual-exposure claims. Synthetic/canary
exports are refused even for dry-run.

## Necessary shared-file exception

Only `baselines/common/fsdp_train.py` differs under `train/ baselines/ harness/
src/` relative to `2506fa9`: optional `fixed_batches=None` and
`optimizer_state=None` arguments, and correct SFT callback loss metadata for
that new route. These hooks are necessary: the original `dose` reduces small
batch sizes, checkpoint callbacks were SAD-only, and optimizer restoration
was unavailable. Repeatedly calling the old loop would reset optimizer and
scheduler state and change the recipe.

With both options omitted, the original branches and callback metadata are
unchanged. No existing objective, scorer, gradient reduction, optimizer
construction, clipping, scheduling, model preparation or exporter changed.
`train/`, `harness/` and `src/` remain unchanged.

Regression tests execute `2506fa9`'s trainer alongside this version with mock
model scores. All15 valid existing objective/packing/scoring paths produce
bitwise-equal parameters and identical update logs, returned accounting
(excluding elapsed time), and SAD callbacks. Additional tests exercise every
reader rejection, real nested evidence readers with fake executions, path
relocation, identical ordering, exact3:1 composition, analytic AdamW updates,
restoration after exhausted LR, actual PEFT tensor continuation, fixed
checkpoints, serving inspection, OFF_HOME preservation, and direct reads of
this tree's registered recipe.

## Validation in this worktree

- Required command: `CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests/test_repair_*.py` — **304 passed**. One PEFT warning is from saving the randomly initialized tiny test model without a base config path; no real model is substituted.
- Existing `test_sft_shared_history_options.py` and `test_sft_shared_history_torch.py::test_actual_final_partial_batch_and_accounting` — **34 passed** (CPU only).
- `git diff --check` and `bash -n scripts/hpg_repair_train_arm.slurm` passed.
- No commit, push, GPU training, or live API call was made.

Requested shared-code comparison:

```text
$ git diff --stat 2506fa9 -- train baselines harness src
 baselines/common/fsdp_train.py | 44 +++++++++++++++++++++++++++++++++++++++---
 1 file changed, 41 insertions(+), 3 deletions(-)
```

This is the optional-hook exception explained and regression-tested above;
there are no E-series support changes in this implementation.
