# Stage 2b validation — 2026-10-04

Stage 2b implements the Seed-SFT pipeline. No GPU computation, network access,
OpenAI calls, package installation, SLURM submission, commits or pushes were
performed. The required repository documents and governing handoff were read
first, including sections 6, 8.1, 10 and 12 and the section 7.3 profiling rules.
The reference `tc-alignment-taskeq/src/bfas/rtd/baselines/pi1.py` was read for
lessons about native turn boundaries, complete-turn exposure and token-mean
loss. No implementation was copied from that trainer.

## Files and behavior

| Files | Delivered behavior |
| --- | --- |
| `data/seeds/select_seeds.py`, `src/sed/seed_selection.py`, `data/manifest/seeds_k10_s0.json` | Parameterized K/seed/versioned rule, TRAIN-only group sampling, ordered same-type replacements, canonical manifest/freeze hashes, separate requested trajectories and selected groups/instances. Immutable, repeatable output. |
| `teachers/collect_seeds.py`, `src/sed/seed_evidence.py` | Existing Teacher and episode runner; three attempts per candidate by default; first actual environment win per slot; immediate same-type replacement after failure; durable attempt receipts, raw episode/events/blobs, API ledger, provenance, interrupted/unknown accounting, resumable collection. |
| `train/dataset.py` | Shared `build_messages`, offline snapshot tokenizer, exact OFF evaluation generation prefix, assistant reply plus native `<turn\|>`/106 labels, strict prompt masking and length checks, per-example mask report, separate diagnostic-only report path. |
| `train/modeling.py`, `train/sft.py` | Official Gemma4Unified model, PEFT LoRA, deterministic complete-turn token budgeting, token-mean CE, checkpoint artifacts and hashes, exposure ledger, profiling and provenance. |
| `train/merge_lora.py` | Checks adapter hashes/base provenance, merges to standalone safetensors, compares all fixed-input next-token logits with base+adapter before and after standalone reload, records consistency/hashes/profiling. |
| `configs/hpg.env`, `scripts/hpg_train_sft.slurm`, `scripts/hpg_eval_merged.slurm` | Installed training/reference Python paths; one B200, account/qos `fsu-compsci-dept`, four CPUs, 32 GB RAM, eight hours. Merged evaluation reuses the Stage 2a launcher on frozen dev40. |
| `configs/base.yaml`, `eval/run_suite.py`, `students/vllm_student.py`, `scripts/hpg_eval_student.slurm` | Current CLI/launcher contract fixes thinking OFF and a 1,024-token student step cap. Merged-LoRA evaluations receive a distinct condition label. Low-level historical parser/client tests still exercise both thinking response shapes. |
| `src/sed/provenance.py`, `pyproject.toml`, `requirements-notes.md`, README/state docs | Training dependency provenance/pins, CLI entries, workflow and runtime documentation. |
| `tests/test_stage2b_{seeds,schedule,torch}.py` | Determinism, split leakage, replacements, crash recovery, unknown costs, actual tokenizer masks, real native token 106, exposure planning, weighted backward/update, checkpoint integrity, official tiny Gemma merge/save/reload. |

## Seed protocol

`stratified_groups_v1` starts with sorted task types and shuffles them with
`random.Random(seed)`. Quotas are `K // 6`, with one extra for the first `K % 6`
types. Only eligible source-TRAIN instances assigned to frozen logical TRAIN
and listed in `seed_pool_candidates` qualify. Groups containing any non-TRAIN
assignment are excluded. Within each group, the trial with the lowest canonical
SHA-256 of `[seed, "trial", task_id]` is chosen. Each type's sorted groups are
shuffled with RNG seed `sha256([seed, "groups", type])`. Sibling trials never
become independent candidates or replacements. A changed rule needs a new
version/artifact; existing frozen output is never overwritten with different data.

The generated **K=10, seed=0** selection is:

| Shuffled type order | Initial groups |
| --- | ---: |
| pick_heat_then_place_in_recep | 2 |
| pick_clean_then_place_in_recep | 2 |
| pick_and_place_simple | 2 |
| look_at_obj_in_light | 2 |
| pick_two_obj_and_place | 1 |
| pick_cool_then_place_in_recep | 1 |

There are 10 initial instances/groups and 1,455 ordered replacement groups.
**Verified trajectories remain zero**: selection is not collection. K denotes
the requested trusted executed trajectories, with group/instance counts separate.
Selection identity: `8069c4c5dcdf3464d8e4aad66a3fedb8806fc3a636cc56db95557b850e8c125e`.
Manifest and split freeze were not changed, and no sealed environment was opened.

Collection uses a separate teacher cache per logical episode attempt. Failed
attempts can therefore obtain fresh samples without repeatedly replaying the
first failed cached reply. Resuming completed attempts reads and validates their
artifacts without another teacher call. A crash after completed episode output
but before its receipt is recovered without re-execution. An incomplete episode
is retained as an interrupted attempt with unknown outcome/cost where applicable;
it consumes an attempt and the next attempt starts with a real environment reset.
No restored state is fabricated. Three episode attempts are distinct from the
Teacher client's separately logged HTTP retries.

A successful receipt requires the runner's completed successful episode and
the final real-environment verification blob with `audit.won is True`, an
executable transition and an actual environment step. Teacher success prose
cannot qualify it. Each receipt identifies the actual parent task and original
selection slot, API usage including unknown calls, transitions, rejection and
hashes of episode/teacher artifacts. Replacement exhaustion produces a partial
`exhausted` manifest, never an invented K. Resume checks config, selection,
code, dependency identity and saved evidence. Changed identities require a new
output directory. Training loads only a complete manifest's verified records,
rechecks hashes, frozen TRAIN membership, distinct groups and type quotas.

## Masking and real ReAct diagnostic

One example is one **complete assistant turn** with the entire preceding
conversation as masked context. Across a pass, every assistant turn is taught
once, including its own closing token 106. Previous assistant replies are not
repeatedly supervised in later prefixes. No context or target is truncated;
any example exceeding configured `--max-length` fails.

The installed Gemma template injects `<|channel>thought\n<channel|>` only with
`add_generation_prompt=True` and `enable_thinking=False`; rendering a completed
conversation alone omits that prefill. We use the exact evaluation generation
prefix, then derive the visible target/native closure from the completed-chat
template. We assert prompt/target token boundaries, mask **every** prefix token
(including system, JSON user observations, earlier replies and empty thought
prefill), and label the current visible reply plus `<turn|>`/106. The trailing
template newline is outside the generated target. Masks are positional: shared
words in an observation and a legitimate assistant reply are not confused.

The report [stage2b_mask_sample.json](stage2b_mask_sample.json) uses the existing
successful real ALFWorld ReAct episode:
`reports/runs/teacher-dev40/episodes/000-41f40a30d930/episode.json`.
It is a **dev diagnostic only**: no training dataset was created, every example
is marked `training_eligible: false`, and the training loop rejects such rows.
The Stage 1 action-only smoke was not relabeled or used. No new episode ran to
produce this report.

The sample has 16 complete turns, **473 supervised tokens** including 16 native
turn-end tokens, and a largest complete context of **4,182 tokens**.

| Example | Total tokens | Masked tokens | Supervised tokens | Last label |
| --- | ---: | ---: | ---: | ---: |
| 0 | 446 | 422 | 24 | 106 |
| 1 | 662 | 635 | 27 | 106 |
| 2 | 903 | 874 | 29 | 106 |

First decoded supervised span (the report includes the first three):

```text
Thought: I should check the toilet paper hanger for toilet paper.
Action: go to toiletpaperhanger 1<turn|>
```

All 16 examples pass the prompt/user/observation and empty-thought-channel mask
assertions. Sample report SHA-256:
`4999164f236a9caf1d73c1cf565d7f616d98ed97b06577cb6a2616ba61e22731`.

## Training and merge semantics

Defaults: bf16 base, rank 16/alpha 32/dropout 0 LoRA on all text attention and
MLP `q/k/v/o/gate/up/down_proj` modules, gradient checkpointing (non-reentrant),
microbatch one complete turn, AdamW lr 1e-5, betas .9/.999, epsilon 1e-8,
weight decay zero, gradient clipping 1, constant LR, deterministic algorithms,
TF32 disabled and eager attention. Model forward is the official
`Gemma4UnifiedForConditionalGeneration`; its supported `logits_to_keep` selects
only positions predicting supervised tokens to avoid a full context-by-vocabulary
logit tensor. There is no custom forward or teacher-logit access.

Updates accumulate whole turns within the default 512 supervised-token budget;
an individual longer turn runs alone. Sum CE is divided by the actual token
count of that update, so short and long turns are weighted by supervised tokens.
Each pass is a seeded exhaustive shuffle. `--passes` counts complete dataset
passes; `--supervised-tokens` is a cumulative token threshold. If both are given,
training stops at the first reached limit. Checkpoint endpoints and the final
token threshold round **up to the first complete turn**, flushing accumulation.
The actual count and overshoot are recorded; exact arbitrary token equality
cannot be promised while preserving whole turns. All endpoint values must lie
within the specified limits. Final state is always saved.

`training_state.json`, `updates.jsonl`, `exposure_plan.json`, `example_sources.json`
and `profiling_report.json` account for actual supervised tokens, row exposures,
passes (turn fraction and token-equivalent fraction), optimizer updates,
seconds/update, effective supervised tokens/s, wall time and CUDA allocated/
reserved peaks. CPU tests report CUDA memory as null. Checkpoints include adapter,
tokenizer, optimizer state and content hashes; fresh output is required for SFT.
`preparation.json` reports loading/preparation separately. Training-loop wall
time includes checkpoint saves and hashes; update timing covers forward/backward/
optimizer computation. Collection/API and evaluation timing remain separate.

Merge compares **all vocabulary next-token logits** on a fixed synthetic visible
input, first in memory and again after standalone reload. GPU bf16 defaults are
`atol=0.125, rtol=0.01`, explicitly allowing bf16 rounding when adding LoRA into
base weights. Absolute/mean errors and argmax agreement are reported. CPU float32
tests use `atol=1e-5, rtol=1e-5`. Failed or nonfinite comparisons block the merge
CLI/evaluation launcher. These bf16 tolerances remain to be measured on the real
trained checkpoint; passing CPU plumbing does not establish full-model fidelity.
Merge runtime, peak CUDA allocation and output hashes are recorded separately.
Base identity records the configured revision, local snapshot path and metadata
hashes; base weight bytes are not independently certified against Hugging Face.

## Offline validation

The project `.venv` is Python 3.11 with ALFWorld/TextWorld and no torch. The RAI
reference `.venv` is Python 3.12, torch 2.13.0+cu130, Transformers 5.14.1,
PEFT 0.20.0 and Accelerate 1.14.0. Real tokenizer files only were loaded from
the requested pinned local snapshot; no 12B weights were loaded here. Torch
checks instantiate a tiny random official Gemma model solely for CPU plumbing.
Tests hide CUDA and block DNS, TCP/UDP and live OpenAI calls.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:. \
ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1 \
.venv/bin/python -m pytest -q

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:. \
/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python \
  -m pytest -q tests/test_stage2b_torch.py
```

Results: **106 passed, one skipped** in the project venv (torch-only module
skipped), and **four passed** in the reference venv. Existing tests remain green,
including real ALFWorld reset/step/prefix replay. CPU tests exercise official
forward loss equivalence, real backward/update, token-weighted accumulation,
checkpoint metadata/integrity, nonzero adapter changes, adapter reload, merge and
standalone reload. Six CLI `--help` checks, SLURM/config `bash -n`, and
`git diff --check` pass.

Exact sample-report command (use a fresh report path when repeating):

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:. \
/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python train/dataset.py \
  --snapshot /home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --diagnostic-episode reports/runs/teacher-dev40/episodes/000-41f40a30d930/episode.json \
  --report reports/stage2b_mask_sample.json
```

## Exact HPG commands for later execution

These commands were **not executed**. The configured student snapshot, TRAIN
game files, dev40 files and installed venvs must already be present. Teacher
collection makes real network/API calls and requires a later authorized run.
The training and merge jobs load all model/tokenizer files locally.

```bash
cd /blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
source configs/hpg.env
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

# Repeatable selection; the checked-in K=10/seed=0 artifact must agree.
CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" data/seeds/select_seeds.py --k 10 --seed 0

# Later authorized teacher collection; repeating this exact command resumes it.
CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" teachers/collect_seeds.py \
  --selection data/manifest/seeds_k10_s0.json --attempts 3 --output "$VERIFIED"

# Optional offline preflight of the verified TRAIN collection.
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$TRAIN_PYTHON" train/dataset.py --snapshot "$MODEL_PATH" --verified "$VERIFIED" \
  --max-length 32768 --report reports/seed-sft-k10-s0-mask.json

# Concrete initial profiling run: token-only, checkpoints at these thresholds.
export TRAIN_OUT="$PROJECT_ROOT/reports/runs/seed-sft-k10-s0-t20000"
export TRAIN_PASSES=0 TRAIN_TOKENS=20000 CHECKPOINT_TOKENS='5000 10000 20000'
export ENABLE_THINKING=off MAX_TOKENS=1024
sbatch --export=ALL scripts/hpg_train_sft.slurm
```

After that job completes, in a shell with the same config and `TRAIN_OUT`:

```bash
# Select the final checkpoint using its recorded actual count (may exceed 20000).
export ADAPTER
ADAPTER=$("$ENV_PYTHON" -c 'import json, os; from pathlib import Path; p=Path(os.environ["TRAIN_OUT"]); print(p/json.loads((p/"profiling_report.json").read_text())["checkpoints"][-1]["path"])')
export MERGED_MODEL_PATH="$PROJECT_ROOT/reports/runs/seed-sft-k10-s0-t20000-merged"
export OUT="$PROJECT_ROOT/reports/runs/seed-sft-k10-s0-t20000-dev40"
export WORKERS=4 TASKS="$PROJECT_ROOT/data/manifest/dev40.json"
sbatch --export=ALL scripts/hpg_eval_merged.slurm
```

For pass-based training instead, choose a fresh `TRAIN_OUT`, set
`TRAIN_PASSES=3`, and clear `TRAIN_TOKENS` and `CHECKPOINT_TOKENS`. HPG training
Python is `/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/.venv/bin/python`
(user-provided torch 2.11.0+cu128 / Transformers 5.14.1 / PEFT 0.20.0 /
Accelerate 1.14.0 inventory); paths are configuration, not Python constants.

## Remaining experimental acceptance

Actual teacher seed collection, full Gemma bf16 backward/update, B200 memory fit
and throughput, full-model merge tolerance and vLLM serving, dev40 evaluation,
and evidence that the student learns remain unmeasured. The 20,000-token command
is a configurable profiling example, not a selected scientific training budget.
This implementation and tiny CPU tests do not establish a trained baseline,
faithful baseline reproduction, end-to-end GPU acceptance or scientific finding.

## Episode packing follow-up — 2026-10-04

`train/sft.py` and the mask-report CLI now default to `--packing episode`.
`--packing turn` retains the original complete-turn examples and update packing
rule described above. This engineering follow-up used no GPU, network, teacher
API, new environment interactions, package installs, commits or pushes. The
template, visibility whitelist, verified-collection checks and frozen splits
were not changed.

An episode example uses exactly the input IDs of its **last turn** in turn
mode: `apply_chat_template` over the complete preceding conversation with
`enable_thinking=False, add_generation_prompt=True`, followed by the visible
final reply and its native `<turn|>`/106. The final target serialization is
checked against `apply_chat_template` over the completed full conversation;
the trailing template newline is outside the final target, as before. Earlier
assistant replies are located in that native rendered history by checked token
prefixes. Labels cover every visible assistant reply plus its own closing 106.
All other positions, including inter-turn newlines, are masked. Each reply is
supervised exactly once per pass; supervised-token exposure totals are identical
between modes. Every turn retains its event/source provenance.

There is one known prefix difference. At generation, the unmodified template
appends `<|turn>model\n<|channel>thought\n<channel|>` before a reply. Completed
history instead renders `<|turn>model\n<reply><turn|>` without the empty thought
channel. A real-tokenizer two-turn test explicitly verifies both renderings.
Episode packing supervises **all non-final turns** in this history format;
the final turn still uses the exact evaluation generation prefix. Consequently,
the affected count is total assistant turns minus episodes. Turn packing has
zero affected supervised turns. The schema-v2 mask report records this
description, both prefix strings, the affected count and per-example turn
indices, as well as every assistant span and the prompt/observation masks.

The microbatch is one whole episode. An episode-mode optimizer update closes
once accumulated **supervised tokens** reach the requested budget; it may
overshoot by the final whole episode. Checkpoint endpoints and the final token
limit flush at the first complete episode reaching or crossing them. If one
episode crosses multiple endpoints, a single checkpoint records all endpoints
and their individual overshoots. Pass limits still stop at the first reached
limit. No long episode is split or truncated: an overlength sequence fails with
the existing explicit error. `turns_seen` continues to count assistant turns;
`examples_seen` counts episodes in episode mode, and `passes` uses that example
fraction. Token-equivalent passes and all exposure/checkpoint budgets remain
based on supervised tokens. The loss remains the token mean per actual update.

`preparation.json` and its printed JSON now contain the selected mode, examples
and assistant turns per pass, supervised tokens per pass, and
`estimated_forward_tokens_per_pass` for **both** modes. These estimates sum
input sequence lengths; they do not predict attention FLOPs, memory or runtime.
The mask report includes the same estimates.

Offline tokenizer checks on the existing dev diagnostic episode
`reports/runs/teacher-dev40/episodes/000-41f40a30d930/episode.json` give:

| Mode | Sequences/pass | Supervised tokens/pass | Forward input tokens/pass | Affected turns |
| --- | ---: | ---: | ---: | ---: |
| turn | 16 | 473 | 36,648 | 0 |
| episode | 1 | 473 | 4,182 | 15 |

This remains diagnostic-only, with `training_eligible: false`; it was not
trained or added to a training collection. The episode sequence matches the
last turn's 4,182 input IDs exactly.

The existing K=20 RAI mask report at
`reports/runs/train/seedsft-k20-s0-rai/mask_report.json` contains 382 assistant
turns in 20 episodes and 10,407 supervised tokens per pass. Summing all recorded
turn lengths gives **1,424,355** input tokens; summing each episode's final-turn
length gives **101,653** episode-mode input tokens, about **14.01× fewer**.
The corresponding prefix mismatch count is 362. These are estimates derived
from the saved report, not a new verified K=20 load or a GPU throughput result.
The recorded first 512-token update in `updates.jsonl` took 72.7568 seconds;
no episode-mode speedup or full-model memory fit was measured here.

Validation: **142 passed, one skipped** in the project venv (torch unavailable),
and **11 passed** in the reference venv's `tests/test_stage2b_torch.py`.
The tests cover exact per-turn label equality and token multisets with the real
tokenizer (one, two and five turns, including Unicode/outer whitespace), the
two-turn history/generation difference, visibility masks, unchanged template,
length rejection without truncation, deterministic episode budget overshoot,
legacy turn packing, endpoint coalescing/rounding and actual CPU checkpoint
saves. Tiny official Gemma CPU tests also verify disjoint assistant-span loss
against native masked CE and supervised-token-weighted backward accumulation.
CUDA is hidden and test fixtures forbid DNS, sockets and live OpenAI transport.

The offline test commands in the earlier validation section still apply. Both
`train/dataset.py --help` and `train/sft.py --help` expose `--packing`, and
`git diff --check` passes. To reproduce the earlier turn-only mask sample, add
`--packing turn` to its command; omit it or use `--packing episode` for the new
default, with a fresh report path. Existing saved run artifacts are unchanged.
