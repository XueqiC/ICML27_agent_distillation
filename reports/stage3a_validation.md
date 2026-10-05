# Stage 3a engineering validation — 2026-10-04

Implemented shared baseline infrastructure and an ETO algorithm port at
`a2fc5da38f8d00cfaf3f9b6370d586eebaf72904`. This report describes offline engineering
checks, not ALFWorld performance, convergence or a faithful experimental reproduction.

No GPUs were used, no model/teacher servers were started, no HTTP/OpenAI calls
were made, no jobs were submitted, and no commits/pushes were made. Existing
`train/sft.py`, `train/dataset.py`, prompts, parser, decoding defaults and token/turn
caps were retained. `third_party/`, `data/verified/` and `reports/runs/` were not
modified by this work. Real seed evidence and installed tokenizer/source files
were read only. Fake launcher subprocesses write local ledgers and never listen
on a socket.

## Implementation and fidelity

| Component | Implemented behavior |
| --- | --- |
| `baselines/common/conversation.py` | Calls existing `episode_examples(..., packing="episode")`; no alternative mask engine. Shared DPO prompt is system+first-user, every assistant reply ends in 106, all observations and headers masked. Round-trip validation rejects extra fields, altered prompts and model control tokens. |
| `baselines/common/explore.py` | Resolves IDs against frozen manifest/splits, calls `authorize_task(...,"generation")` before side effects, rejects dev/sealed (including train-derived dev), uses existing Student/runner/event store and real reset/replay. Retains unknown/error episodes and sanitized error classes. |
| `baselines/common/schedule.py` | X5 batch, exhaustive epoch permutations (seed+epoch), ceil warmup steps, HF cosine/constant schedule step semantics, explicit zero-weight rank slots. |
| `baselines/common/fsdp_train.py` | torchrun NCCL/FSDP2 production entry; fp32-loaded full text policy, bf16 compute/fp32 reduction, decoder+root sharding, SDPA, checkpointing, AdamW defaults, clip 1.0. Episode-mean SFT and sigmoid summed-logp DPO. |
| `baselines/eto/official_loss.py` | Two verbatim official methods, Apache-2.0 notice, independent AST/formula comparison. |
| `baselines/eto/pairs.py` | Golden 1, won 1/else known loss 0; tie skipped, expert chosen/full rollout rejected, trailing observation dropped, non-assistant ends rejected, no truncation. P/win/tie/lose/rejection decisions retained. Data rechecked against original receipts before training. |
| `baselines/eto/pipeline.py` | SFT then same K tasks → pairs → DPO up to 3 times, exact requested hyperparameters, seed 0, explicit no-pairs stop, immutable stage completion hashes, retries preserved, dev140 evaluation separate. |
| Serving/evaluation scripts | `MERGED_MODEL_PATH` full-checkpoint mode, artifact/provenance/identity checks, base served model name, distinct full-parameter policy condition, unchanged evaluation prompts/decoding. |
| `scripts/hpg_eto.slurm` | Four B200s, one node/task, 32 CPUs, 480G, requested department account/partition/qos, no job submission during validation. |

Complete upstream-to-port change registry: [DEVIATIONS](../baselines/eto/DEVIATIONS.md).
All six repository URLs and commits: [PINS](../baselines/PINS.md).

Local source inspection confirmed transformers **5.14.1**, class
`Gemma4UnifiedTextDecoderLayer` at installed `modeling_gemma4_unified.py:496`,
and `Gemma4UnifiedForConditionalGeneration`, matching `train.modeling.load_student`.
The unified architecture calls its non-text components `embed_vision` and
`embed_audio`; all non-text state keys are copied from the original snapshot.
No Llama/Gemma4 non-unified wrap class or Chinese-origin checkpoint is selected.

## Offline validation

Required commands (from repository root):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  tests/test_stage2b_torch.py tests/test_lora_serving_torch.py tests/test_stage3a_torch.py
bash -n scripts/eval_adapters.sh scripts/hpg_eval_adapters.slurm scripts/hpg_eto.slurm
git diff --check
```

Final full pass: **190 passed, 4 skipped (32.53s)** in the project Python 3.11
environment; **36 passed (28.45s)** in the torch Python 3.12 environment.
The focused launcher/protocol/suite check also passed **32 tests (11.50s)**.
After adding padded-pair log-prob and checkpoint identity checks, the three affected
tiny-objective/export cases passed again (**3 passed, 9.89s**).
Project skips are torch-dependent modules and the optional real-ALFWorld integration
without `ALFWORLD_DATA_ROOT`. Torch emitted one existing PEFT warning in the LoRA
serving test. Local torch is **2.13.0**; HPG's configured cu128 runtime is **2.11.0**.
Tests enforce no CUDA, DNS, TCP/UDP or live OpenAI calls with fake clients/environments.

Covered:

- Every assistant turn of the first checked real K20/s0 seed matches the evaluation
  generation prefix after removing only the documented empty-thought marker for
  non-final turns. The final prefix is exact. Synthetic one-/multi-turn DPO branches
  share the prompt and preserve all assistant masks and 106 closures.
- TRAIN-only exploration authorization precedes environment/client creation;
  real runner with fake backend executes and replays a 40-step failed rollout.
  Costs distinguish rollout versus replay; the output adds zero trusted seeds.
  Unknown service failures are retained and rejected by pair loading.
- X5 cases K=5/10/20 and arbitrary P, epoch completeness/repeatability, rank padding
  for b'<ranks and partial batches, mean loss/gradient scaling equivalence.
- HF scheduler values at every step, including step 0, zero/positive/all-step warmup,
  and the end boundary. Warmup uses ceil, not floor.
- Official method AST equality and independent numerical formulas on random
  tensors. Sparse causal log-probs match the original dense-logit calculation.
- Tiny random Gemma full-parameter SFT and DPO optimizer runs on CPU. Parameters
  update, references stay frozen, scheduler starts correctly, and reward metrics
  are logged. Streamed bf16 checkpoint export/reload preserves logits (fp32 CPU
  execution of bf16 file weights) and exactly copies an injected frozen non-text
  tensor plus processor metadata. No foundation-model weights are loaded by tests.
- Fake shell lifecycle checks preserve existing adapter ordering/cleanup and
  exercise full-checkpoint serving without HF LoRA checks. Pipeline tests cover
  stage reuse, changed-artifact refusal, failed-attempt preservation, exact DPO
  hyperparameters/reference path, and P=0 stopping before DPO.

A single-process CPU gloo/FSDP feasibility probe was attempted with an in-memory
HashStore. Gloo initialization returned `RuntimeError` in this sandbox before
FSDP forward/backward. No successful distributed run is claimed. Per the task's
fallback, the training loop and objectives are tested on tiny CPU models without
FSDP; multi-rank behavior is checked algebraically, not on hardware.

## Artifacts, dose and accounting

Each (K,support-seed) output has `identity.json`, pipeline `provenance.json`,
`tasks.json`, `models.json`, per-iteration pair reports and `disclosure.json`.
A stage lives under `stages/<name>/attempt-NNN/`; successful stages publish a
`complete.json` with hashes. Failed and interrupted attempts remain immutable.
Resume validates a completed stage's code/data/input identity and all artifact
bytes. Changing code, configuration or seed contents requires a new output tree.
An incomplete stage restarts in a new attempt; repeated interactions are additional
cost, not additional K or unique tasks. There is no within-epoch optimizer resume.

Exploration writes `receipts.json`, `summary.json`, per-task `events.jsonl`, blobs,
`episode.json`, `replay.json`, and student request/response/token ledgers. Receipts
include parent seed episode IDs, source=`student_exploration`, expansion=true,
trusted_seed=false, training eligibility, won/steps/legal/inadmissible/parser-error
counts and content hashes. Replays count as additional environment episodes and
steps. Incomplete-attempt costs may be unknown and remain flagged.

Training provenance includes effective arguments, input hashes, code/dependency
versions, frozen manifest and seed collection binding, original and iteration-start
snapshot hashes, student revision, tokenizer template hash, exact data hash, dose,
precision/attention/FSDP class, optimizer and reference definition. The source seed
receipts retain teacher alias, returned ID/timestamp and all teacher usage metadata;
ETO itself has no teacher calls. Each `updates.jsonl` logs actual examples, supervised
tokens, learning rate, loss, gradient norm and DPO rewards/logps. Each stage saves
only its final policy. `checkpoint/training_state.json` and `result.json` include
exposures, logical tokens, physical padded/zero-weight compute, wall time and peak
allocated/reserved GPU bytes per rank (null in CPU tests).

For K20 SFT, b'=2 → 10 optimizer steps/epoch → 30 steps. K5/K10 SFT use b'=1
→ 15/30 total steps. DPO uses N=P for the same rule (P≤K). Every epoch includes
every example exactly once; a partial batch divides by its actual size. Four ranks
can participate with b'=1/2 without increasing the learning batch: unused ranks
run a zero-weight duplicate solely to participate in FSDP collectives. The duplicate
forward/backward counts are disclosed separately; no extra training exposure or
new task is claimed.

Checkpoint export gathers one text tensor at a time, casts it to bf16, and writes
bounded safetensors shards plus the HF index/config and original processor/tokenizer
assets. Non-text tensors are read directly from the original base snapshot, with
byte-value hashes. Exports can be served by `--merged-model-path` without merging
LoRA. Full checkpoint mode verifies training provenance and artifact hashes first.

## Exact HPG commands (not executed)

Training K20/support-seed 0, training seed 0:

```bash
source configs/hpg.env
cd "$PROJECT_ROOT"
export VERIFIED="$PROJECT_ROOT/data/verified/seeds_k20_s0"
export ETO_OUT="$PROJECT_ROOT/reports/runs/eto/k20-s0"
export ETO_MODE=train ETO_THROUGH=all
sbatch --export=ALL scripts/hpg_eto.slurm
```

For K5/support-seed 1:

```bash
export VERIFIED="$PROJECT_ROOT/data/verified/seeds_k20_s1/subsets/k5"
export ETO_OUT="$PROJECT_ROOT/reports/runs/eto/k5-s1"
sbatch --export=ALL,ETO_MODE=train,ETO_THROUGH=all scripts/hpg_eto.slurm
```

Use the same output tree to resume; set `ETO_THROUGH=sft`, `pairs-1`, `dpo-1`,
etc. for an intentional stage boundary. `ETO_MODE=prepare` freezes inputs without
launching training or serving. The default resource request is four GPUs, 32 CPUs,
480G. The job does not fetch dependencies or checkpoint weights.

After training, evaluate every available checkpoint on dev140 in a separate job
(the allocation is reduced because stages are served sequentially):

```bash
sbatch --gpus=1 --cpus-per-task=8 --mem=48G \
  --export=ALL,ETO_MODE=evaluate scripts/hpg_eto.slurm
```

One full checkpoint via the existing adapter launcher:

```bash
export MERGED_MODEL_PATH="$ETO_OUT/stages/sft/attempt-000/output/checkpoint"
export FULL_CHECKPOINT_NAME=sft
export TASKS="$PROJECT_ROOT/data/manifest/dev140.json"
export OUT="$PROJECT_ROOT/reports/runs/eto-dev140/k20-s0-sft"
unset ADAPTERS
sbatch --export=ALL,INCLUDE_BASE=1 scripts/hpg_eval_adapters.slurm
```

`models.json` is authoritative for checkpoint paths if stages needed retries.
Neither training nor these evaluation modes requests sealed data. Evaluation is
not used to decide whether/what to generate or to choose a stage checkpoint.

For an already served policy, explicit TRAIN exploration only:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$ENV_PYTHON" -m baselines.common.explore \
  --tasks "$ETO_OUT/tasks.json" --endpoint http://127.0.0.1:PORT/v1 \
  --model-name google/gemma-4-12B-it --decoding greedy --workers 8 \
  --server-provenance /path/to/server/provenance.json \
  --output /path/to/fresh/student-exploration
```

The `PORT` and server provenance must be taken from the supervisor's actual ready
receipt. This standalone command is an additional exploration run and incurs
additional disclosed interaction cost.

## Remaining acceptance boundary

No 12B training, CUDA/FSDP2 collective execution, B200 memory/throughput measurement,
long-episode GPU run or vLLM reload/serving of a trained full checkpoint was performed.
CPU correctness checks and inspected APIs do not establish those results. A later
allocated run must exercise SFT and DPO plus standalone serving and check recorded
memory/timing/artifact receipts. This engineering deliverable makes no scientific
claim about ETO, Seed-SFT or agent distillation performance.
