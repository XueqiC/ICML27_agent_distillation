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

Add `--dry-run` for one CPU process with an explicit byte tokenizer and a
513-parameter fake model. This executes the shared scorer, backward, AdamW,
scheduler, batch accounting and checkpoint exporter. It never loads the real
base, adapter tensors, tokenizer, or optimizer tensors. Theta0 metadata and
hashes are still validated. Fake checkpoints have a `synthetic/v2-tiny-fake`
identity, `synthetic: true`, and `synthetic_saved` status; they are not E4B
adapters. Their artifact structure passes the serving identity inspector,
but they cannot be served as E4B. Real training exposure stays zero; fake
exposure is recorded separately. Dry-run token counts use the fake tokenizer
and must not be reported as E4B token counts. Dry-run with `--gpus 2` still
uses one CPU; production padding estimates are available through `accounting`.

Outputs must be new directories outside the sealed dataset, theta0 and base
snapshot. Existing outputs and symlinks are refused.
No teacher, environment rollout, evaluation or checkpoint selection runs here.

## Input checks and replay layout

Each bundle has the contract's `examples.jsonl`, `examples.sha256`, and
`manifest.json`. Example rows and message/evidence objects use explicit field
whitelists. JSON duplicate keys and nonfinite values are refused. The reader
checks example byte hashes, sorted unique canonical IDs, counts, supervised
turn totals, exposure groups, hashed input files and theta0's weight hash.
All hash references must be available locally. Relative `inputs` paths are
relative to their bundle; absolute paths are used directly. No path is
silently relocated. List/object seals use `sed.storage.digest` (sorted-key,
compact UTF-8 JSON); file seals, including `splits_sha256`, hash raw bytes.

The spec lists `k20_replay` but does not specify a `replay/` layout. This
consumer accepts either of these explicit representations:

- A separately sealed interchange bundle at `dataset/replay/`, containing
  only `k20_replay` rows (normally arm `CONTINUE`). Its split and theta0 hashes
  must agree with the main bundle. Root manifest counts describe root rows;
  replay manifest counts describe replay rows.
- Contract-format replay rows directly in root `examples.jsonl`, with the
  root arm. They are partitioned into the replay pool by source.

Mixing both representations is refused. Repeated replay content is refused
by the sampler. Replay sampling order ignores the arm-dependent example ID.

The frozen method split file is taken from the hashed input named
`method_v1_splits.json`, or explicitly supplied with `--splits`. It must still
be a hashed input of every bundle. Its `sets` uses this repository's existing
split representation: each set has `task_ids` and `group_ids`. Required sets
are `V2_pool`, `U_dev`, `U_mech`, `D_probe`, `D_unseen_quick`, and `dev140`.
Replay membership comes from `sets.K20` when present; otherwise the reader
loads theta0's original hash-bound verified K20 view through the existing
local evidence reader. Every task's group/type and TRAIN eligibility are also
checked against the configured ALFWorld manifest and reproducible split
freeze. V2/K20 overlap and diagnostic task/group overlap are refused. This
implementation does not generate or modify a split file.

Messages must exactly round-trip through `sed.react.build_messages`; user
JSON can contain only observation and admissible commands. No `Guidance:`
text is accepted in any message. Every assistant turn must parse, and every
supervised action must be admissible in its preceding user turn. Decision
indices must identify exactly the last assistant message; trajectory indices
must identify every assistant message. Label instance references must already
occur in the prefix. Only `label_check: passed` labels are accepted.

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

Result: **462 passed**, including 56 V2 tests; four existing PEFT warnings
concern missing base-config paths for synthetic tiny models.

The broader run also included `test_stage2b_torch.py`,
`test_stage3a_torch.py`, `test_baseline_packing.py`,
`test_sft_shared_history_options.py`, `test_sft_shared_history_torch.py`,
`test_shared_history_torch.py`, `test_continue_from_state.py`,
`test_lora_adapters.py`, `test_lora_serving.py`, and
`test_lora_serving_torch.py`: **860 passed, 1 skipped, 2 failed**.
That run preceded the last six V2 test additions, which passed in the final
required run. The failures are existing tests with unavailable prerequisites:

- `test_official_loss_verbatim_and_random_formula_equivalence` requires absent
  `third_party/ETO/fastchat/train/dpo_trainer.py`; reproduced in isolation.
- `test_two_process_gloo_matches_one_process_lora` cannot initialize Gloo's
  TCP transport in this sandbox (`Operation not permitted`).

The skip requires a real seed collection absent here. No existing test or
module was changed to bypass these failures. No real GPU run, live API call,
commit or push was performed. The four new files have no trailing whitespace;
tracked-file and staged diffs are empty, and HEAD remains `425945c`.
