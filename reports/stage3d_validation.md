# Stage 3d — LEAP engineering validation (2026-10-05)

Implemented the pinned LEAP SFT algorithm with a privileged training teacher,
plus **LEAP (0% privileged ablation)**. No real expert rollout, teacher request,
student server, GPU execution, job submission, commit or push was performed.
`third_party/`, `reports/runs/`, `data/` and the default evaluation protocol were
not changed. This report is engineering evidence, not a learning result or a
claim of end-to-end reproduction of the paper's experiments.

## Source and fixed protocol

Official source: `third_party/leap_llm`, commit
`a3376d89a2eeee88f03555a4174a0bdf7faf8be6`. Read the requested dataproc scripts,
all ALFWorld templates and the 0% ablation template, training script and config;
also inspected the upstream parser, OpenAI helper, PEFT config and merge helper.
No upstream module is imported or executed by production code. Templates are
rendered directly from the pinned local files with their content hashes bound
into run identity. Missing upstream workflow components are not recreated.
Changes and exact upstream references are in
[DEVIATIONS.md](../baselines/leap/DEVIATIONS.md).

Local installed expert: **alfworld 0.4.2**, `alfworld/agents/environment/alfred_tw_env.py`,
`AlfredExpert` lines 64–115. Its handcoded implementation asks for facts internally,
provides `extra.expert_plan`, and uses `HandCodedTWAgent(max_steps=200)` internally.
The enclosing shared harness still enforces **40 turns**. LEAP installs the expert
only inside `baselines/leap/expert.py`; the ordinary environment backend is unchanged.
The installed environment/expert source files are hashed in LEAP run identity.
Availability is determined by an **executed environment win**, never by a planned
route or teacher claim. Failures, exceptions and unavailable experts remain logged.

K remains the count of trusted seed trajectories. Expert runs and student rollouts
add zero trusted seeds. Every generation task comes from checked seed receipts and
the frozen TRAIN manifest, with parent episode IDs. Unique instances and task
groups are reported separately. Expert availability does not filter iter0 BC or
student rollouts; it filters privileged correction only. The 0% ablation skips
unused expert/extraction work and does not use that availability filter.

Iteration 0 uses all K, skips reason annotation, and creates one example per
original step. Later labels come from one whole-trajectory correction request.
Only final-turn `Thought: ...\nAction: ...` plus native 106 is supervised; all
system/user/history tokens are masked. Past thoughts are visible original
student replies. Echoed teacher observations/actions, summaries and audit fields
cannot become context. Inadmissible corrected commands remain labels and are
counted. Their execution verification stays **unknown**.

Each training loads iter0..i sets, subsamples every independently seeded shuffle
to `min(size_0,...,size_i)`, and shuffles the merged rows. A test executes the
released aggregation AST using installed HF Datasets, changing only seed 42→0.
An empty minimum produces an explicit stop; no seed-only replacement training
or extra teacher calls are invented.

Training: r128, alpha64, dropout .05; exact full q/v names under the text model;
fresh adapter on the prior merged checkpoint; bf16 frozen weights, fp32 adapter
masters, SDPA, gradient checkpointing, fused AdamW .9/.999/1e-8, wd0, lr3e-5,
cosine, fixed 10 warmup steps, clip .3, one epoch, seed0. X5 uses step-example
count N and original batch32. Microbatch4 is preserved when it fits; partial
batches/accumulation are explicit. No truncation, length32768.

At K≤20 and ≤40 steps per episode, each set has ≤800 examples. Four equally
subsampled sets have ≤3200 examples, hence ≤100 updates at the largest dose;
smaller X5 batches also stay below 200. The trainer checks this condition before
training. Thus upstream eval-loss selection at step200 cannot fire. All K and
all valid corrections are available for training, with no 85/15 holdout; only
the final epoch checkpoint is published. Dev140 never drives selection.

## Offline validation

The project conftest hides CUDA and blocks DNS, TCP/UDP and real OpenAI transport.
Fake clients/environments supply all interactions. New LEAP tests cover:

- Privileged/audit/fact/plan isolation from student contexts and decoded training
  tokens; official prompt switch; literal privileged-summary rejection in labels.
- Fenced/bare JSON, complete step alignment, malformed replies, preserved
  inadmissible targets and dropped teacher echoes/flags.
- Original-history reconstruction and checked seed/rollout/teacher receipts;
  tampering and wrong iteration chains rejected.
- Expert success/failure recording and TRAIN-only authorization.
- Transport/cache/ledger/limiter integration, unsupported-sampling fallback,
  interrupted reservations, incomplete responses and three-attempt caps.
- All four merged-model initializations, min aggregation, empty aggregate,
  unchanged models on stage resume and hash mismatch rejection.
- Tiny Gemma CPU forward/backward, q/v-only adapter updates with frozen decoy
  vision projections, merge/reload logits, zero fresh LoRA B matrices, completion
  loss, fixed ten-step cosine and real Gemma OFF/106 masks.

Commands (all offline):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  tests/test_stage2b_torch.py tests/test_stage3a_torch.py \
  tests/test_stage3b_torch.py tests/test_stage3c_torch.py \
  tests/test_stage3d_torch.py tests/test_lora_serving_torch.py

bash -n scripts/hpg_leap.slurm
git diff --check
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m baselines.leap.pipeline --help
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m baselines.leap.train --help
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m baselines.leap.merge --help
```

Results:

| Check | Observed result |
| --- | --- |
| Complete project `.venv` suite, final rerun | **507 passed, 7 skipped**, 415.17s |
| All six torch test modules in the separate CPU runtime | **60 passed**, 34.72s; one existing PEFT warning about a tiny test model's absent config path |
| Launcher syntax, all three CLI help paths, diff whitespace | Passed |

The seven project-runtime skips are six torch modules (that venv has no torch;
all were run in the separate CPU environment) and the opt-in real ALFWorld
integration test (no data-root run was requested). The project suite includes
25 LEAP tests; the nine new LEAP torch tests are included in the 60 torch passes.

Local collection runtime: Python3.11, ALFWorld0.4.2, TextWorld1.7.0,
OpenAI2.8.1, Jinja2 3.1.6, NumPy2.4.6, pytest7.4.0. The separate CPU torch
runtime uses Python3.12.13, torch2.13.0, transformers5.14.1, PEFT0.20.0 and
Datasets5.0.1. Installed SDK
Responses signatures accept `temperature` and `top_p`; that does **not** prove
model/account acceptance. No network documentation, canary or pricing lookup was
performed. The OpenAI Docs skill was constrained to local SDK inspection by the
explicit no-network instruction.

## Exact HPG commands for later authorized execution

These commands are a runbook and **were not executed**. Use a checkout separate
from the active mechanism worktree. Paths below follow the repository's HPG
defaults; outputs are on `/blue`, outside original seed storage. Jinja2 3.1.6
and NumPy2.4.6 are the declared `leap` extra; install/validate dependencies ahead
of time rather than downloading anything inside a job.

```bash
source configs/hpg.env
cd "$PROJECT_ROOT"
export VERIFIED="$PROJECT_ROOT/data/verified/seeds_k20_s0"
export BATCH_INVARIANT=1
export LEAP_VARIANT=privileged
export LEAP_OUT="$PROJECT_ROOT/reports/runs/leap/privileged/k20-s0"
export LEAP_EXTRACTION_TOKENS=4096 LEAP_CORRECTION_TOKENS=16384

# Input freeze only: no environment, teacher, server or GPU work.
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$ENV_PYTHON" -m baselines.leap.pipeline \
  --config configs/base.yaml --snapshot "$MODEL_PATH" --verified "$VERIFIED" \
  --variant "$LEAP_VARIANT" --output "$LEAP_OUT" --mode prepare

# One B200, eight CPUs, 128G host RAM, 12-hour allocation.
# First job: expert/extraction, BC and checked iter0 merge.
job0=$(sbatch --parsable --export=ALL,LEAP_THROUGH=merge-0 scripts/hpg_leap.slurm)
job1=$(sbatch --parsable --dependency="afterok:$job0" \
  --export=ALL,LEAP_THROUGH=merge-1 scripts/hpg_leap.slurm)
job2=$(sbatch --parsable --dependency="afterok:$job1" \
  --export=ALL,LEAP_THROUGH=merge-2 scripts/hpg_leap.slurm)
job3=$(sbatch --parsable --dependency="afterok:$job2" \
  --export=ALL,LEAP_THROUGH=all scripts/hpg_leap.slurm)

# Separate frozen dev140 evaluation, published iter0..3, greedy/OFF/1024.
sbatch --dependency="afterok:$job3" --mem=48G \
  --export=ALL,LEAP_MODE=evaluate scripts/hpg_leap.slurm
```

Repeat the same learning pipeline in a different output tree for the ablation:

```bash
export LEAP_VARIANT=zero
export LEAP_OUT="$PROJECT_ROOT/reports/runs/leap/zero/k20-s0"
zero0=$(sbatch --parsable --export=ALL,LEAP_THROUGH=merge-0 scripts/hpg_leap.slurm)
zero1=$(sbatch --parsable --dependency="afterok:$zero0" \
  --export=ALL,LEAP_THROUGH=merge-1 scripts/hpg_leap.slurm)
zero2=$(sbatch --parsable --dependency="afterok:$zero1" \
  --export=ALL,LEAP_THROUGH=merge-2 scripts/hpg_leap.slurm)
zero3=$(sbatch --parsable --dependency="afterok:$zero2" \
  --export=ALL,LEAP_THROUGH=all scripts/hpg_leap.slurm)
sbatch --dependency="afterok:$zero3" --mem=48G \
  --export=ALL,LEAP_MODE=evaluate scripts/hpg_leap.slurm
```

For K5/10 set `VERIFIED="$PROJECT_ROOT/data/verified/seeds_k20_s0/subsets/k5"`
or `/subsets/k10`, plus a new corresponding `LEAP_OUT`. `s0` identifies the
support collection seed; training and subsampling seed remain zero. Reissue the
same stage command to resume. A completed stage is hash checked; an interrupted
training/rollout stage restarts in a retained new attempt. Completed expert tasks
and teacher logical requests resume individually. A reserved interrupted teacher
attempt consumes its slot, including unknown potential usage; no cap reset.

The low-level train and merge commands are generated by `training_command` and
the pipeline. For an existing prepared iteration, the exact form is:

```bash
PYTHONPATH=src:. "$TRAIN_PYTHON" -m baselines.leap.train \
  --config configs/base.yaml --snapshot "$PREVIOUS_MERGED" --base-snapshot "$MODEL_PATH" \
  --verified "$VERIFIED" --variant "$LEAP_VARIANT" --iteration 1 --sets "$ITER0_EXAMPLES" "$ITER1_EXAMPLES" \
  --output "$NEW_TRAIN_OUTPUT"
PYTHONPATH=src:. "$TRAIN_PYTHON" -m baselines.leap.merge \
  --config configs/base.yaml --snapshot "$PREVIOUS_MERGED" \
  --adapter "$NEW_TRAIN_OUTPUT/adapter" --output "$NEW_MERGE_OUTPUT"
```

Here the uppercase paths must refer to the pipeline's recorded example sets and
prior checkpoint, with new output directories. Prefer the launcher to retain
the complete stage graph. All training is from the configured Gemma revision;
there is no model download or fallback. Merges compare all next-token vocabulary
logits before and after merge and reload (bf16 atol .125, rtol .01); failed merges
cannot be published for serving. The existing standalone validation path accepts
the resulting checkpoint without changing any default evaluation request.

## Artifacts, accounting and limits

`identity.json`, `provenance.json`, `tasks.json`, `teacher_caps.json`, `models.json`
and `iteration-0..3.json` bind protocol, code/template/expert-source hashes,
dependencies, config, split freeze, models, parent seeds, example sets and counts.
`audit/experts/` holds full executed expert episodes/events, availability, errors
and provenance. `audit/stages/extract/` holds teacher-extracted states;
`audit/teacher/` holds per-task bindings, durable attempt starts, exact sanitized
prompts/responses, usage ledgers, cache, limiter and sampling acceptance/fallback.
Raw oracle state and expert artifacts never populate student prompts or training
contexts. Checked teacher receipts provide only sampled correction reason/action
labels to the training loader.

`stages/explore-i/` retains shared harness student logs, real replay and token
accounting. `stages/correct-i/` holds hash-bound outcome references and correction
counts. `stages/data-i/` holds whitelist-only examples and inadmissibility counts.
The training loader reconstructs examples from original receipts rather than
trusting serialized prompts; only correction reason/action labels enter targets.
`stages/train-i/` contains selected indices/order, original and subsampled set
sizes, source exposure, per-update ledger, native adapter and training provenance.
`stages/merge-i/` holds standalone checkpoints, consistency checks and provenance.
`disclosure.json` includes all attempts, unknown costs and failed/rejected outputs.

Privileged labels are intentionally teacher-informed. Tests prove the raw state,
oracle fields and echoed histories are not inserted into model inputs. The
official template instructs the teacher to use general reasons, and a literal
summary/heading guard rejects obvious copies. **This is not a semantic guarantee**
that sampled reasons never disclose a hidden fact; no unrequested verifier model
or hidden-state supervision was added.

## Cost estimate and hard caps

Let E≤K be successful expert extractions and F_i≤K known failed student episodes.
The privileged variant makes ≤K extraction plus ≤K logical correction requests
per iteration; available state can reduce correction requests further. Three API
attempts at most belong to each logical request; unsupported-sampling errors and
malformed responses count against the same cap. Attempts are not extra tasks.

| K | Privileged logical requests (≤4K) | Privileged API attempts (≤12K) | 0% logical requests (≤3K) | 0% API attempts (≤9K) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 20 | 60 | 15 | 45 |
| 10 | 40 | 120 | 30 | 90 |
| 20 | 80 | 240 | 60 | 180 |

Using the specification's **illustrative, unmeasured** planning assumptions of
3.5k input/0.3k output tokens per extraction and 13.5k input/9k output per full
40-step correction, K20 with every rollout failed costs approximately **0.88M
input + 0.546M output tokens** for the privileged variant, before retries,
reasoning-token overhead and cache effects. At a 50% failure rate, approximately
0.475M input + 0.276M output. Applying the same conservative correction size to
the 0% variant gives 0.81M input + 0.54M output in the all-failed case; its prompt
is shorter by the omitted state. These estimates are not tokenizer measurements.

If the applicable account rates are `p_in` and `p_out` dollars per million tokens,
the privileged K20 all-failed estimate is **`0.88*p_in + 0.546*p_out` dollars**,
or up to three times that illustrative volume with full retries. No current
gpt-6-luna dollar price was supplied or verified offline, so none is invented.
Actual billed and cached token counts, including reasoning-token accounting,
come from the shared ledger; failed/interrupted unknown usage stays unknown.

The output caps give a harder bound: per logical-attempt sweep, privileged K20
≤`20*(4096+3*16384)` = **1,064,960 output tokens**, or ≤3,194,880 across all three
attempt slots. 0% K20 ≤983,040, or ≤2,949,120 with all retries. These API caps do
not imply that incomplete or malformed responses become accepted training data.

Each variant uses 3K student rollouts plus 3K validation replays; the privileged
variant additionally uses K handcoded expert episodes. At forty transitions each,
that is ≤280K transitions privileged (5,600 at K20) and ≤240K for 0% (4,800), before
interrupted-stage retries. Four-checkpoint dev140 evaluation adds 560 episodes
per variant in separate jobs. All of these tasks are distinct from K accounting.

One B200 is allocated for r128 LoRA on the 12B model. Four bf16 merges require
roughly **96 GB per variant**, **192 GB for both**, plus adapters, logs and retained
partial/failed attempts; reserve additional headroom on `/blue`. Nothing deletes
previous checkpoints. GPU hours have not been measured. Estimate later as the
sum of measured training, merge/reload, rollout and evaluation durations on that
one GPU. The 12-hour request is a scheduling limit, not a measured runtime claim.
