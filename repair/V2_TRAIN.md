# E0.5: V2 training arm

Implemented as new files on `repair-v01-train`, based on `425945c`.
The input contract is
[`V2_interchange_spec.md`](/home/xueqi/hq/projects/ICML27_agent_distillation/docs/method/V2_interchange_spec.md),
and the protocol is experiment plan v1 §1 E0.5 and §4. No evaluation-branch
code is imported. Existing modules, including `fsdp_train.py`, are unchanged.

## Invocation

```bash
PYTHONPATH=src:. /path/to/training/python tools/train_v2_arm.py run \
  --dataset /path/to/interchange --theta0 /path/to/tsft/checkpoint \
  --config configs/base_e4b.yaml --snapshot /path/to/pinned/E4B \
  --updates 20 --batch-examples 4 --replay-ratio 0.25 \
  --checkpoints 1 5 20 --seed 1 --gpus 1 --output /new/v2-run
```

Use `--gpus 2` for the main grid's `torch.distributed.run --standalone
--nproc_per_node 2` launcher. The LoRA route in the shared FSDP trainer uses
replicated bf16 bases and fp32 adapters, averaging adapter gradients before
clipping. It does not shard these LoRA models. One GPU runs the same loop at
world size one, without creating a process group. Existing torchrun workers
are recognized and never recursively launch. No GPU allocation is requested
by this tool; the caller supplies the devices and the grid runtime.

Add `--dry-run` for one CPU process with a tiny fake model. The default
`--dry-run-tokenizer bytes` keeps the explicit byte tokenizer and 513-parameter
model. `--dry-run-tokenizer snapshot` loads only the real tokenizer from the
local `--snapshot`, using `local_files_only=True` and `trust_remote_code=False`.
It uses the unchanged production encoder and inherited `max_length`, refusing
overlong contexts without truncation. After validation, a bijective vocabulary
compaction makes the synthetic projection small enough for CPU scoring; it
preserves every token distinction, length, boundary and mask. The original
encoded-example hash, scoring-example hash, vocabulary mapping and tokenizer
file/template hashes are recorded. Only the fake projection's vocabulary size
changes; no pretrained model is loaded.

Both modes execute the shared scorer, backward, AdamW, scheduler, batch
accounting and checkpoint exporter. Neither loads real base, adapter, or
optimizer tensors. Theta0 metadata and hashes are still validated. Fake
checkpoints have a `synthetic/v2-tiny-fake`
identity, `synthetic: true`, and `synthetic_saved` status; they are not E4B
adapters. Their artifact structure passes the serving identity inspector,
but they cannot be served as E4B. Real training exposure stays zero; fake
exposure is recorded separately. Byte-mode counts are synthetic; snapshot-mode
lengths use the real tokenizer, while losses and updates remain synthetic.
Dry-run with `--gpus 2` still
uses one CPU; production padding estimates are available through `accounting`.

Outputs must be new directories outside the sealed dataset, theta0 and base
snapshot. Existing outputs and symlinks are refused.
No teacher, environment rollout, evaluation or checkpoint selection runs here.

## Running on another host

Relocation is opt-in on `run` and `inputs`. For example, append these flags
to the invocation above after copying the sealed bundle and its dependencies:

```bash
  --relocate /home/xueqi/hq/projects/ICML27_agent_distillation=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation \
  --relocate /home/xueqi/.cache/huggingface=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
```

Each rule replaces one absolute directory prefix, preserving the relative tail.
Matching uses whole components, ordered longest first; duplicate/overlapping
OLD prefixes, relative prefixes, `..` tails, file prefixes and NEW inside the
sealed dataset are refused. Symlinks may point elsewhere within resolved NEW
(as HF snapshot links do), but cannot escape it. There is no fallback to OLD
when a mapped file is absent or has different bytes. Add rules for **every**
other unavailable input prefix too: the real OFFLINE seal also references
`/home/xueqi/hq/projects/ICML27_run_repair_eval_7748da5`.

On the source host, create an inventory before copying:

```bash
CUDA_VISIBLE_DEVICES='' python tools/train_v2_arm.py inputs \
  --dataset /path/to/OFFLINE --config configs/base_e4b.yaml \
  --snapshot /path/to/pinned/E4B --list --output /tmp/v2-inputs.json
```

The JSON contains de-duplicated local filenames, source references, prefix-relative
names, byte sizes and recorded SHA256 seals, grouped by destination prefix
(unmapped paths use the dataset root or `/`). Use the relative names as rsync
file lists for each prefix; preserve HF symlink targets or copy with `rsync -L`.
`--output` exclusively creates a new file. Listing does not certify hashes;
theta0 metadata must be local, so list on the source host before transfer.
Theta0 defaults to the sealed `adapter_path`. Snapshot defaults to the sealed
tokenizer path, then theta0's recorded base path. Config defaults to the same
`configs/base_e4b.yaml` as `run`; supply the run's exact overrides for a matching
inventory. `--theta0`, `--snapshot`, `--config`, `--splits`, `--repair-splits`,
and `--dev140` accept relocation. Diagnostic split sources are included in the
inventory with their method-freeze SHA256 seals.

On the destination, use `inputs --dataset /local/OFFLINE --check` with the same
config/snapshot overrides and relocation flags. Both interchange bundles,
theta0's adapter directory, every snapshot file, configs, frozen manifests and
repair/dev140 split sources are fully byte-hashed on CPU. `run` performs the
same check before loading its tokenizer or model. The training reader's schema,
split, example-evidence and recipe checks still apply separately.

When theta0's original K20 view supplies replay membership, the view's manifest
must first match theta0's recorded byte hash. The unchanged
`sed.seed_evidence.load_verified_collection` then validates that view and its
parent collection, including failed attempts and trajectories outside K20.
Its semantic checks and comparisons with the collection's own recorded hashes
always run. Only dependencies covered by successful collection validation skip
the second byte-hash pass in `v2_inputs` and the run's provenance writer.
Unrelated files under a collection directory still receive byte hashing.
Collection metadata with explicit raw-byte seals retains those checks,
including theta0's view-manifest binding; semantic hashes cannot replace a
recorded raw-byte seal.
Missing files, altered recorded hashes and relocation escapes remain refusals;
sealed internal collection paths must resolve to the files the loader validates.

This uses the main grid's `sed.hash_receipts` implementation and honours
`SED_HASH_CACHE` and `SED_HASH_CACHE_DIR`. Receipts match size, `mtime_ns` and
inode, and are disposable accelerators rather than evidence. As on the main
grid, a same-size edit with a deliberately restored timestamp and unchanged
inode requires full rehashing to detect. Add `--full-rehash` to `run` or
`inputs --check` for an audit: it applies `SED_HASH_CACHE=0` throughout validation
and provenance, bypasses both receipt caches, and byte-hashes every input.
Setting `SED_HASH_CACHE=0` directly has the same effect. The flag is invalid
with `inputs --list`.

On HiPerGator, set an explicit persistent **/blue** cache path, never `$HOME`
or `/tmp`, and warm it with the CPU check before allocating a GPU:

```bash
export SED_HASH_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/cache/sed-hash-receipts
CUDA_VISIBLE_DEVICES='' python tools/train_v2_arm.py inputs \
  --dataset /local/OFFLINE --check \
  --config configs/base_e4b.yaml --theta0 /local/tsft/checkpoint \
  --snapshot /local/pinned/E4B
```

Supply the same relocation flags as the training run. Cold means no receipts;
warm checks still enumerate/stat evidence and validate collection semantics.
`inputs --check` reports end-to-end `prep_seconds` without loading a model.

Sealed manifests/examples and original hash keys remain unchanged. Evidence
episode references remain audit strings, not filesystem reads or model inputs.
When the reader needs theta0's original K20 training view, its `verified` path
is relocated and checked against the hash under the original provenance key.
Historical training paths that are not opened remain historical metadata.
Run `manifest.json` records `relocation`; `provenance.json` records the same
object in `details.relocation`: rules, distinct relocated-reference counts per
OLD prefix (including directories, counted once across bundles), verified-file
count, and `all_hashes_verified`. It also records `byte_hashed_files`
(`verified_hashed_files` is its compatibility alias), `collection_covered_files`,
`collection_validation_roots`, and canonical JSON SHA256 anchors for each
validated collection in `collection_validation_anchors`. File counts use
distinct resolved paths and exclude byte-hashed files from collection coverage.
`full_rehash` and `hash_cache` record the effective policy and
`hash_receipts.cache_provenance()`; run provenance also retains its top-level
`hash_cache`. Collection evidence remains hash-bound through theta0's view
manifest and the validated collection anchors. Checkpoint manifest structure
is unchanged.

## Input checks and replay layout

Each bundle has the contract's `examples.jsonl`, `examples.sha256`, and
`manifest.json`. Example rows and message/evidence objects use explicit field
whitelists. JSON duplicate keys and nonfinite values are refused. The reader
checks example byte hashes, sorted unique canonical IDs, counts, supervised
turn totals, exposure groups, hashed input files and theta0's weight hash.
Under spec rule 6, `counts.by_source`, `by_kind`, and `by_task_type` are required
and must equal recomputed counts. Optional `by_local_event` counts every element
of `evidence.local_events` over decisions only, including repeated elements.
When the list is absent it counts `evidence.local_event`, omitting null.
`examples` counts rows,
`task_groups` distinct group IDs, and `unique_instances` distinct task IDs.
Each is checked when present, separately for root and replay. Other count keys
are refused.
All hash references must be available locally. Relative `inputs` paths are
relative to their bundle; absolute paths are used directly. No path is
silently relocated. List/object seals use `sed.storage.digest` (sorted-key,
compact UTF-8 JSON); file seals, including `splits_sha256`, hash raw bytes.

The consumer accepts these explicit replay representations:

- A separately sealed interchange bundle at `dataset/replay/`, containing
  only `k20_replay` rows (normally arm `CONTINUE`). Its split and theta0 hashes
  must agree with the main bundle. Root manifest counts describe root rows;
  replay manifest counts describe replay rows.
- Contract-format replay rows directly in root `examples.jsonl`, with the
  root arm. They are partitioned into the replay pool by source.

Under rule 7, CONTINUE root and `replay/` may be byte-identical K20 pools.
The arm share is drawn from root and the replay share from `replay/`, retaining
both file origins and independent epoch streams. The replay draw uses the same
seeded sequence as every other arm. For other arms a row in both pools is
refused. Repeated content within either sampling pool is refused. Replay
sampling order ignores the arm-dependent example ID.

The frozen method split file is taken from the hashed input named
`method_v1_splits.json`, or explicitly supplied with `--splits`. It must still
be a hashed input of every bundle. The real method freeze contains `V2_pool`
and `U_dev`, each with `task_ids` and `group_ids`. `U_mech`, `D_probe`, and
`D_unseen_quick` are read from `repair_v01_splits.json`; dev140 task IDs come
from `dev140.json`, with groups resolved through the configured frozen manifest.
Both source files must match the SHA256 recorded by the method freeze's
`exclusion_sources` and paths in `rebuild_arguments`. Relative rebuild paths
are resolved from the producer project using `rebuild_arguments.output`, not
from the consumer's working directory. `--repair-splits` and `--dev140` can
specify local copies on both `run` and `inputs`, but cannot override the seals.
Missing sources, missing sets, or changed bytes are refused; none are treated
as empty. Previously supported self-contained method freezes with every set
inline remain valid; recorded external sources always require verification.
Replay membership comes from `sets.K20` when present; otherwise the reader
loads theta0's original hash-bound verified K20 view through the existing
local evidence reader. Every task's group/type and TRAIN eligibility are also
checked against the configured ALFWorld manifest and reproducible split
freeze. V2/K20 overlap and diagnostic task/group overlap are refused. This
implementation does not generate or modify a split file.

Messages must exactly round-trip through `sed.react.build_messages`; user
JSON can contain only observation and admissible commands. No `Guidance:`
text is accepted in any message. Every assistant turn must parse. Decision
indices must identify exactly the last assistant message; its action must be
admissible in the preceding user turn, and its label instance references must
already occur in the prefix. Only `label_check: passed` labels are accepted.
Evidence requires `episode_ref`, `step`, `executor`, `local_event`,
`episode_success`, `label_source`, and `label_check`. A trajectory requires null
`step` and `local_event`; a decision requires a nonnegative integer `step`.
The spec's executor, local-event and label-source enums apply to both kinds.

Rule 8 permits optional decision evidence: `local_events` is a list of non-null
local-event enum values; `complete_success_eligible` is a bool;
`public_prefix_sha256`, `fact_state_sha256`, `facts_before_sha256`, and
`facts_after_sha256` are 64-hex strings; `history_rewrites` is a list of objects;
`next_legal_target_take` is an int or null (bool is refused). If `local_events`
is present, `local_event` must equal its first element, or null for an empty
list. These fields remain audit-only and never enter encoded model inputs.
Any other extra evidence field is refused, as are optional decision fields
on trajectories.

Under the 2026-10-27 spec revision (rules 2, 4 and 5), trajectory indices must
identify every assistant message, and OFFLINE, CONTINUE and K20 replay
trajectories must reproduce the main-grid trajectory-SFT rendering byte for
byte. Their inadmissible actions and not-yet-seen instance references are
retained unchanged and fully supervised, as in the baselines. These two
conditions produce informational diagnostics, not trajectory rejections.
All other validation, including split eligibility, public-field whitelists,
prompt equality, action parsing, guidance exclusion and evidence checks,
still applies to both kinds.

The consumer preserves evidence references and all input hashes. The contract
does not define the referenced raw episode-record schema, so this reader does
not claim to reexecute an episode, independently certify actual observation
history, or prove a rationale's semantic entailment. Those checks belong to
E0.4's execution evidence and label audit; `episode_success` is never used as
a model input or treated as a newly measured environment outcome.

## Continuation, masks and sampling

`repair.train_arms.inherited_recipe` verifies the exact theta0 recipe, metadata
and safetensors trainable inventory. `load_existing_adapter` loads the existing
default LoRA trainably. No new adapter is attached. Explicit text-module
coverage, including absent K/V projections and k_eq_v handling, remains owned
by the existing loader/model/serving path. Serving-only K=V conversion is
never written back into a training adapter.

The inherited r64/alpha128/dropout0.05, targets, bf16 compute/base storage,
fp32 adapter masters, shared-history scoring, AdamW betas/epsilon/weight decay,
clip1.0 and optimizer restoration validation are retained. `optimizer.pt`,
when present, restores moments and validates group settings, shapes and any
named trainable snapshot. When absent, every arm initializes AdamW identically
and records that choice. The exhausted theta0 scheduler is not restored:
the same inherited shape restarts at the inherited LR over the requested N
updates, exactly as the existing continuation path does for 20 updates.

Encoding uses `baselines.common.conversation.encode_messages`, which calls
the main-grid `train.dataset` renderer. Trajectories retain every assistant
target; decisions retain only the final turn row, with the whole native
history masked in its prefix. Supervised targets include native `<turn|>`
token **106**; system/user text, model headers and empty thought prefill are
masked. No context is truncated. Loss is summed assistant-token CE divided
by supervised tokens within each example, then averaged over B examples.
It is not a mean over turns or one token mean over the entire batch.
Inherited SFT has no sample-weight hook, so every weight must equal 1.0.

Every optimizer update has exactly B examples. `B * R` must be integral;
fractional quotas are refused rather than silently rounded. Independent RNG
streams shuffle replay epochs, new-example epochs and batch slot positions.
Each pool is exhausted without replacement, then reshuffled. A batch may
cross an epoch boundary. For equal replay content, seed, B and R, replay slot
positions and draws are identical across arms, regardless of the new pool's
length or token lengths. `batches.json` records every index, pool, epoch and
example ID; its canonical SHA256 is in provenance and manifests.

CONTINUE has only K20 examples. Its non-replay slots use a separate K20 epoch
stream, preserving the shared replay draws even at the same R as the other
arms. Thus its actual K20 source fraction is 1.0. Both streams are recorded,
including possible repeats across their independent epochs. Set R=1 when
only one K20 stream is desired.

## Artifacts and accounting

The run root contains `provenance.json`, `manifest.json`, `batches.json`,
`example_sources.json`, and `updates.jsonl`. The last is the shared trainer's
per-update loss/LR/gradient and token-accounting curve. Provenance includes
effective config, code hashes/revision, source manifests, dependency versions,
student identity, environment configuration, seeds, and inherited optimizer
and scheduler decisions. Teacher/environment calls during training are zero;
source collection provenance remains hash-referenced through the interchange.
The run manifest's `trajectory_diagnostics` is a list of per-example records:

```json
{"example_id": "<trajectory example ID>", "inadmissible_supervised_turns": 1,
 "not_yet_seen_instance_supervised_turns": 0}
```

Counts are recomputed from the supplied public messages for every trajectory
in both pools, including records with zero counts. Each supervised turn counts
at most once per condition, even if it cites multiple unseen instance IDs;
the same turn may count under both conditions. Each prefix ends immediately
before that assistant turn, so later observations cannot ground earlier
references. Decision examples are excluded from these diagnostics and retain
their rejection checks. These are dataset counts, independent of replay
sampling or repeated training exposure. Sealed input manifests and examples
are preserved; diagnostics never enter model inputs or affect loss masks.

New/replay pool inventories distinguish example records, episode references,
unique instances and groups. The inherited K=20 describes theta0's trusted
seed trajectories; it is not the number of training examples or tasks.

Checkpoint 0 always references theta0's original directory and file hashes;
it does not export another adapter. Each requested positive update writes:

```text
update-NN/provenance.json
update-NN/manifest.json
update-NN/checkpoint/adapter_config.json
update-NN/checkpoint/adapter_model.safetensors
update-NN/checkpoint/training_state.json
```

The unchanged `export_adapter` supplies the layout and identity fields read
by `tools/check_lora_serving.py`. Each checkpoint manifest hashes every file
in its adapter directory and its adjacent provenance, and records parameter
counts/names/modules, optimizer updates, examples by source/kind, supervised
targets, processed context, and unique instance/group exposure. Physical
context includes rank-padding forwards; supervised tokens and example counts
exclude padding. Logical context and zero-weight slots are reported separately.
Counts are checked against actual trainer callbacks at every update. Saving
preserves torch RNG state and synchronizes ranks. Requested checkpoints alone
are exported; the final update's accounting/loss is recorded even when it is
not requested as a checkpoint. No optimizer-export hook is added.

## Verification

CPU tests cover token-level masks and native turn end, strict input validation,
decision rejection versus trajectory diagnostics, byte-identical main-grid
rendering for OFFLINE/CONTINUE/inline/separate replay, manifest persistence,
theta0 mismatch refusal, replay composition/epochs/common draws, deterministic
schedules, padding-aware accounting, inherited recipe, synthetic backward and
checkpoint hashes, serving identity structure, and CLI launcher routing.
Tests use the repository's network/OpenAI-blocking fixture and no real model.
Real one-update E4B/Blackwell canary and HF/vLLM numerical equivalence remain
runtime acceptance checks; a CPU dry-run establishes no scientific finding.

Final required command:

```bash
CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python \
  -m pytest -q tests/test_v2_train.py tests/test_repair_*.py
```

Earlier relocation verification: **501 passed** in the combined suite, with the same
four existing PEFT warnings. The final targeted invocation of
`CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python
-m pytest -q tests/test_v2_train.py` passed **95 tests**. This includes 20 new
cases for prefix rules, same-size/same-mtime hash tampering, missing files,
symlink confinement, replay/theta0/training-view relocation, CPU inventory
commands, exclusive output and provenance. The original 75 tests are unchanged.

The no-relocation `inputs --check` on
`/home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/method_v1/v2_data/OFFLINE`
passed: **18,212 file paths present; 18,196 distinct files verified against
recorded SHA256 seals**, including replay and the default run config inputs.
This was a file-integrity check, not training-schema acceptance. At that
revision, the reader rejected the producer's informational count fields,
external diagnostic split sources, and null trajectory steps. The current
reader fixes those three mismatches as described above.

Earlier rule 6 / split-source / evidence verification: **171 passed** with
`CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python
-m pytest -q tests/test_v2_train.py`. Coverage includes optional count fields
in both bundles, unknown/missing/wrong counts, null trajectory versus integer
decision steps, undefined evidence fields/enums, missing or altered split
sources, missing sets, diagnostic group exclusion, explicit source overrides,
relocation, input inventories and a synthetic CLI update using external sources.

The 2026-10-30 rule 6/7/8 revision resolves the former plural-event,
CONTINUE-layout and optional-evidence blockers. Snapshot-tokenizer dry runs
address the byte-tokenizer context overflow without truncation or a limit
change. Theta0's original K20 view needs
the explicit local mapping
`--relocate /ddnB/work/xueqic/hq/ICML27_data=/home/xueqi/hq/projects/ICML27_agent_distillation/data`.

Earlier targeted verification: **226 passed** with the same CUDA-disabled
pytest command above. New cases cover every plural event (including repeats),
singular fallback, empty lists, CONTINUE root/replay origins and shared seeded
draws, overlap refusal in all other arms, every optional evidence field's type,
local-event pairing, unknown fields, trajectory null requirements, and offline
snapshot-tokenizer dry runs without model/optimizer loading. Byte mode remains
the default and retains its 513-parameter fake model.

Real snapshot-tokenizer length profiling uses the longest production turn
context per trajectory, with nearest-rank p95. All contexts fit the inherited
32,768 limit; no context was truncated.

| Pool | Trajectories | Max tokens | p95 tokens |
|---|---:|---:|---:|
| OFFLINE root | 200 | 15,967 | 11,309 |
| K20 replay / CONTINUE root | 20 | 16,759 | 9,405 |
| OFFLINE root plus replay | 220 | 16,759 | 11,309 |

CONTINUE's two files contain the same 20 trajectories, unique instances and
task groups; 40 stored rows do not create 40 independent examples. The
per-trajectory measurements and acceptance artifacts are under
`/tmp/claude-1010/-home-xueqi-hq-projects-tc-alignment/035c9823-fbb5-436e-b37c-069b2060bb62/scratchpad/e05_dry/reader_validation_20261030_02/`.

Both real-bundle CLI acceptance runs passed with `--dry-run-tokenizer snapshot`,
2 updates, B=16, replay ratio 0.25, checkpoints 1 and 2, seed 0, and `--gpus 1`.
The sibling outputs are `OFFLINE_reader_snapshot_01/` and
`CONTINUE_reader_snapshot_01/`. Each run records 32 synthetic exposures and
zero real training exposure. The replay draws match across arms, and CONTINUE
uses 12 root selections plus 4 `replay/` selections per batch. The report
records commands, byte hashes, update logs, checkpoint checks and provenance;
these remain synthetic smoke results, with no pretrained model weights loaded.
Nothing was committed.

The E0.5 receipt-verification revision passes **239 tests** with:

```bash
CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python \
  -m pytest -q tests/test_v2_train.py
```

The added tests use the real collection validator on fake execution receipts.
They cover cold and disk-warm reuse, direct interchange references to collection
files, changed/missing evidence outside the selected K20 view, teacher and
receipt tampering, selection semantics, conflicting interchange seals, ordinary
input byte hashes, full-rehash cache bypass and same-metadata tampering,
environment restoration, CLI flags, and persisted run/provenance coverage.
All existing relocation and refusal tests remain enabled.

Real OFFLINE input-check measurements on 2026-10-30 used the requested theta0,
relocation and receipt directory. The execution shell reports hostname
`sn4622122543`; `rai` does not resolve here and no rai SSH alias is configured.
Its rai identity could not be confirmed, so these are measurements from the
available execution environment, not confirmed bare-host rai or HiPerGator
timings.

| Receipt state | CLI `prep_seconds` | Whole process wall seconds |
|---|---:|---:|
| Cold | 587.794 (9m48s) | 590.713 |
| Warm | 477.084 (7m57s) | 480.039 |

Both checks passed with **899,316 listed paths**, **899,276 distinct files
covered by collection validation**, and **31 distinct byte-hashed files**.
The latter include four collection metadata anchors with explicit raw-byte
seals and 27 other inputs; nine alias paths explain the difference between
listed-path and distinct-file totals. Both the K20 view and
`seeds_full_train_r2` appear in `collection_validation_roots`.

Cold means an empty receipt directory and a fresh process; OS page caches were
not flushed. Warm uses a second fresh process with those disk receipts, so it
cannot inherit the first process's memo. Tests did not overlap this final pair.
Metadata inventory and semantic validation still leave substantial prep time;
these results do not establish HiPerGator idle-GPU acceptance. Warm the cache on
CPU before allocation using the `/blue` setting above.

The command for each pass was:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
SED_HASH_CACHE=1 \
SED_HASH_CACHE_DIR=/tmp/claude-1010/-home-xueqi-hq-projects-tc-alignment/035c9823-fbb5-436e-b37c-069b2060bb62/scratchpad/hashcache_e05 \
/home/xueqi/hq/projects/tc-alignment/.venv/bin/python tools/train_v2_arm.py inputs --check \
  --dataset /home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/method_v1/v2_data/OFFLINE \
  --theta0 /home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint \
  --relocate /ddnB/work/xueqic/hq/ICML27_data=/home/xueqi/hq/projects/ICML27_agent_distillation/data \
  --output /tmp/e05-input-verification-final-20261030/cold.json
```

The warm pass used `warm.json` as its exclusive output. Reports, commands,
settings and code hashes are preserved under
[`reports/runs/e05_input_verification_20261030/`](../reports/runs/e05_input_verification_20261030/)
in this worktree (`reports/runs/` is git-ignored). The requested `/tmp` cache
applies only to this measurement; use `/blue` on HiPerGator. No model was loaded
and nothing was committed.

Rules 2/4/5 revision verification: **481 passed**, including 75 V2 tests;
four existing PEFT warnings concern missing base-config paths for synthetic
tiny models. Only `repair/v2_train.py`, `tests/test_v2_train.py`, and this
document were changed. No commit was made.

Before this revision, the same command passed **462 tests**, including 56 V2
tests, with the same four warnings.

The broader run also included `test_stage2b_torch.py`,
`test_stage3a_torch.py`, `test_baseline_packing.py`,
`test_sft_shared_history_options.py`, `test_sft_shared_history_torch.py`,
`test_shared_history_torch.py`, `test_continue_from_state.py`,
`test_lora_adapters.py`, `test_lora_serving.py`, and
`test_lora_serving_torch.py`: **860 passed, 1 skipped, 2 failed**.
That earlier run preceded six V2 test additions, which passed in the then-final
required run. The failures are existing tests with unavailable prerequisites:

- `test_official_loss_verbatim_and_random_formula_equivalence` requires absent
  `third_party/ETO/fastchat/train/dpo_trainer.py`; reproduced in isolation.
- `test_two_process_gloo_matches_one_process_lora` cannot initialize Gloo's
  TCP transport in this sandbox (`Operation not permitted`).

The skip requires a real seed collection absent here. No existing test or
module was changed to bypass these failures. No real GPU run, live API call,
commit or push was performed.
