# Sufficient Experience for Black-Box Agent Distillation

This is a new research codebase for Paper 1. We study what experience lets a
student develop an independently successful agent policy, beginning with K
trusted, executed seed trajectories. **K counts trajectories, not tasks.** New
tasks, sampled teacher outputs, and real interactions are allowed and must be
accounted for. No scientific results or trained models are claimed here.

Stage 1 provides a CPU ALFWorld harness, deterministic data manifest and frozen
splits, durable event/blob storage, a black-box teacher adapter, a single-episode
runner, and replay verification. Teacher inputs and training examples use an
explicit visible-field whitelist. Hidden success labels, facts, expert plans,
task filenames, and audit records never enter prompts. The teacher adapter uses
sampled text and accounting metadata through the [official Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses);
it does not request logprobs or hidden reasoning. No API call was made during
Stage 1 implementation or validation.

The student is `google/gemma-4-12B-it` at revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. The teacher is `gpt-6-luna`.
The configuration records the user-reported canary at 2026-10-04 15:00 CDT,
returned ID `gpt-6-luna`, with no dated snapshot. Each future response records
its returned model ID and time; an alias can drift. The student checkpoint has
not been downloaded or trained here. Only post-training with permitted model
origins is in scope; see [AGENTS.md](AGENTS.md) for the hard rules and contract.

Stage 2a adds the shared ReAct teacher/student interface, a greedy vLLM student
client, frozen dev40, process-based suite evaluation, and a HiPerGator launcher.
The completed pilot fixes ReAct, thinking OFF, and a 1,024-token student step
cap for subsequent methods. Stage 2b adds TRAIN seed selection/collection,
assistant-only token masks, LoRA SFT, and checked merge/export. No Stage 2b
teacher collection or full-model GPU training has been run.
Stage 2c adds shared teacher rate limiting, frozen dev140, and nested K=5/10/20
collections that reuse identical verified trajectories. This remains engineering
validation; no new live teacher run or scientific finding is claimed.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/base.yaml` | Fixed model identities, protocol, environment-variable path overrides |
| `src/sed/` | Configuration, durable storage, events, provenance, manifest, runner |
| `harness/alfworld_env/` | Official ALFWorld/TextWorld adapter with verified prefix replay |
| `teachers/openai_teacher.py` | Sampled-action teacher, cache, retry and usage ledger |
| `data/manifest/` | Builder, content hashes, immutable split freeze |
| `data/raw/`, `data/verified/` | Separate unverified and verified experience destinations |
| `data/seeds/select_seeds.py`, `teachers/collect_seeds.py` | Frozen TRAIN groups, same-type replacements, resumable environment-verified collection |
| `train/dataset.py`, `train/sft.py`, `train/merge_lora.py` | Exact Gemma OFF prompts, assistant token masks, LoRA training, checked standalone export |
| `students/vllm_student.py` | Greedy chat client, separate Gemma thinking, durable call accounting |
| `eval/` | Frozen suite and replay CLIs; no online training or training-pool writes |
| `scripts/hpg_eval_student.slurm`, `configs/hpg.env` | B200 pilot job and overridable cluster defaults |
| `analysis/`, `reports/` | Future mechanism analysis and recorded validation/provenance |
| `tests/` | Fake-based unit tests and one optional real-environment integration test |

## Run locally

Python 3.11 is the tested runtime. Direct dependency pins and limitations are
in [requirements-notes.md](requirements-notes.md). From the checkout, set
`PYTHONPATH=src:.`; installation is optional. Paths resolve from the working
directory. Supply your own installed dataset root; code has no cluster paths.

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=src:.
export ALFWORLD_DATA_ROOT=/path/to/alfworld/data/json_2.1.1
python data/manifest/build.py
python -m pytest -q
```

Choose an eligible ID from `seed_pool_candidates` in the split freeze. Use a
Python environment containing ALFWorld and TextWorld for real episodes:

```bash
python harness/run_episode.py --task-id 'train/TASK/TRIAL' \
  --policy scripted --output reports/runs/my-scripted-episode
python eval/replay_episode.py \
  --episode reports/runs/my-scripted-episode/episode.json \
  --output reports/runs/my-replay
```

The default scripted policy repeats `look` until termination or the 40-step
limit; this tests the complete episode lifecycle, not task-solving ability.
`--actions actions.json` accepts a JSON string list, cycling if exhausted.
`--purpose diagnosis` allows dev runs; default `generation` requires train.
Sealed tasks allow frozen-policy evaluation only, never teacher generation,
branching, or diagnostic replay. The runner never promotes raw samples into
verified training data. Trusted seed selection remains empty until executed
trajectories and required intermediate facts are actually verified.

`--policy teacher` is available for later authorized API runs. It reads
`OPENAI_API_KEY`, falling back to `~/hq/secrets/llm_apis.env` without sourcing
shell commands. Do not enable it in tests. `SED_SECRETS_FILE` overrides the
credential file. The client is lazy, SDK retries are disabled, and the wrapper
attempts at most three calls by default. It stores only allowlisted text and
metadata, never raw SDK exceptions or secrets. Cache keys cover the model ID,
template, task, history, visible intervention context, and sampling settings.
Cache hits retain original response provenance and record zero new billed
tokens. Usage unknown after failed/interrupted requests remains unknown.

## Evidence and split rules

`alfworld_manifest.json` hashes every file in each discovered trial directory
under train, valid_seen, and valid_unseen. Playable eligibility follows the
installed ALFWorld task-type, movable/sliced, and solvability rules. Excluded
trials retain reasons. `valid_train` is outside this Stage 1 protocol. The
default freeze assigns eligible train games to the expandable training pool,
valid_seen to dev, and valid_unseen to sealed test. Train-derived dev is also
supported: set `splits.dev_source: train` and an explicit `dev_count` measured
in task groups, keeping sibling trials together. A changed manifest or split
requires a new explicit destination; existing frozen files are not overwritten.

Every episode writes `provenance.json`, `events.jsonl`, content-addressed
`blobs/`, and `episode.json`. A reset event costs zero environment steps; each
issued command costs one. Under ReAct, an unparseable reply consumes one of
40 policy turns but zero environment steps; it produces visible parser feedback.
A parseable inadmissible command is still executed verbatim. Events and summaries
distinguish `legal`, `inadmissible`, and `parser_error`. The last exact `Action:`
line wins; only surrounding whitespace is removed. Failed step calls are flagged as errors, and their
state outcome is unknown. Teacher usage includes reasoning-token counts only,
with request attempts in a separate append-only ledger. A durable `started`
entry without a completion is an interrupted attempt with unknown cost.

JSONL appends preserve the exact old byte prefix, use a writer lock, and
publish through fsync plus atomic rename. This provides crash-safe logical
append-only semantics on a local POSIX filesystem; it costs O(log size) per
append and is intended for Stage 1. Blobs are immutable SHA-256-addressed JSON
and are durable before events reference them. A damaged log is rejected,
never silently truncated. Large-scale or distributed logging needs a separate
storage design. Runtime data and run directories are git-ignored by default.

The harness uses `textworld.start` and official `AlfredDemangler`, with no
expert wrapper. Reset and step return `(visible, reward, terminated, truncated,
audit)`. `visible` has exactly `observation` and `admissible_commands`.
Snapshot restoration performs a real reset and every prefix action, checking
visible text, admissible sets, reward, terminal flags, and hidden audit hashes
at every point. Divergence invalidates the environment; it cannot create a
counterfactual branch. Replay records environment seed and game hash.

Real trained baselines and later SmartAD/DART-SD comparisons remain future
experiments. `train/dataset.py` tokenizes complete ReAct episodes (default) or
individual turns using the student's local chat template, supervising visible
assistant replies plus their native `<turn|>` token 106 and masking everything
else. The episode/history prefix difference is documented under Seed-SFT below.
The old adapter and support-rollout code were read for interface reference;
no old implementation or experimental numbers were copied wholesale.

See [reports/stage1_validation.md](reports/stage1_validation.md) for measured
local counts, exact runtime, test results, episode artifacts, and remaining
limits.


## Shared evaluation on rai and HiPerGator

The shared template/parser is `src/sed/react.py`: optional visible `Thought:`
followed by `Action: <command>`. Both teacher and student receive system
instructions and alternating observation/user and visible reply/assistant turns.
Gemma's separate thinking never enters the ReAct parser or later history.
The teacher cache uses a new protocol/version key; older cached commands are
not reused. Scripted action lists still contain bare commands; the scripted
policy formats them as ReAct replies.

`data/manifest/dev40.json` freezes 40 `valid_seen` dev instances (40 task groups),
seed 0, with quotas 7/7/6/7/7/6 in alphabetic task-type order. Rebuild/check it
without reading environment data:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python data/manifest/build_dev40.py
```

Cluster defaults live in [configs/hpg.env](configs/hpg.env). `ENV_PYTHON` defaults
to the project's `.venv/bin/python` and must contain ALFWorld, TextWorld, PyYAML,
and httpx. `VLLM_PYTHON` points to the separate serving venv. `MODEL_PATH` defaults
to the already-installed HF snapshot at the pinned revision; no download occurs.
Override any default through exported environment variables. From the HPG checkout:

```bash
source configs/hpg.env
cd "$PROJECT_ROOT"
sbatch --export="ALL,ENABLE_THINKING=off,OUT=$PROJECT_ROOT/reports/runs/stage2a-base-dev40-off" scripts/hpg_eval_student.slurm
```

Each job requests account/qos `fsu-compsci-dept`, partition `hpg-b200`, one GPU,
8 CPUs, 48 GB RAM, and 3 hours. It uses eight environment worker processes by
default (`WORKERS`), a 40-turn limit, and greedy decoding with `MAX_TOKENS=1024`
per reply, including any spontaneous thinking. Current launchers require
`ENABLE_THINKING=off`; historical pilot results are in the reports. The server
uses `FLASH_ATTN`, disables the FlashInfer sampler, sets context length 32768 and
GPU-memory utilization 0.85, and enables `--reasoning-parser gemma4` if registered.
It binds loopback on `20000 + SLURM_JOB_ID % 20000`, checks readiness with a timeout,
and shuts down on completion or cancellation. Workers have CUDA hidden.
`harness/serve_student.py --merged-model-path PATH` supports an explicitly merged
LoRA checkpoint for later work; the pilot uses the base snapshot.

Outputs are `$OUT/server/{provenance.json,server.log}` and
`$OUT/eval/{summary.json,per_task.tsv,provenance.json,episodes/}`. Existing output
directories are refused. Per-episode artifacts include events, replay records,
and durable student request/response/thinking blobs. Total completion tokens
include thinking; thinking-token counts are `null` when the server omits the
breakdown, with unknown counts carried into summaries. Thinking characters and
finish reasons are always recorded for returned completions. No character-based
token estimate is used. Server provenance records the local checkpoint path and
metadata hashes; it does not certify local weight bytes against Hugging Face.

For an already running student server, use the suite directly:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$ENV_PYTHON" eval/run_suite.py \
  --policy student --endpoint http://127.0.0.1:PORT/v1 --enable-thinking off \
  --tasks data/manifest/dev40.json --workers 4 --max-tokens 1024 \
  --output reports/runs/base-dev40-direct
```

`--policy teacher` uses the same ReAct interface and puts its cache/usage ledger
inside episode output directories, outside training data. Sealed task lists
require `--sealed-final`; teacher generation remains forbidden on sealed tasks.
No truncation of conversation history, command substitution, or online learning
occurs. Context/service failures are retained as error episodes with unknown
success, and remain in the suite denominator. `steps` counts policy attempts;
`environment_steps` counts attempted actual calls and separately reports
confirmed/unknown transitions.

See [Stage 2a validation](reports/stage2a_validation.md) for the installed vLLM
response schema, test evidence, and limitations.

### Stage 2e: evaluate adapters without merging

`harness/serve_student.py --lora-adapter NAME=PATH` is repeatable. One server
serves the configured base name (`google/gemma-4-12B-it`) and each adapter NAME.
It sets `--enable-lora`, the maximum adapter rank, the adapter count, and
`--lora-modules`. Names must be unique, differ from the base name, and contain
only letters, digits, dots, underscores, or hyphens (starting with a letter or
digit). `--merged-model-path` still serves an already merged checkpoint; it
cannot be combined with adapters. `VLLM_GPU_MEMORY_UTILIZATION` overrides 0.85
for shared cards. No adapter path causes a base model download or merged copy.

Each original checkpoint must contain `adapter_config.json`,
`adapter_model.safetensors`, and `training_state.json`, with its training
`provenance.json` in the parent directory. Copy these together to HPG. The
launcher checks the configured student repo/revision, the training provenance
binding, checkpoint artifact hashes, and the serving snapshot's metadata
against training. PEFT's null revision is resolved through that bound training
provenance. Snapshot paths may differ between hosts. Server provenance records
every original/served adapter path, config and weight SHA-256, supervised tokens,
passes, packing, and runtime source hashes. Local base weight bytes remain
uncertified, as in base serving.

Before an adapter evaluation, `tools/check_lora_serving.py` must pass. It uses
the first eight verified TRAIN turns sorted by `(task_id, turn_index)` from
`--verified`, through the existing `build_messages` and unchanged Gemma OFF
template. It scores every teacher-reply token plus native `<turn|>` closure;
teacher APIs and teacher logprobs are never used. HF+PEFT uses bf16, eager
attention, and GPU inference. The vLLM `/v1/completions` requests carry the exact
token IDs and request `prompt_logprobs`; returned prompt IDs must match.
`report.json` retains per-token values and per-prompt/pooled maximum and mean
absolute differences in nats. Every prompt and the pooled tokens must have
HF–adapter mean difference at most 0.05, and adapter–base mean difference at
least five times that error and strictly positive. A failed check exits nonzero.
This is an engineering equivalence check, not a measure of task success.

The new launchers perform HF reference passes **before** starting vLLM, using
`--hf-only`, then compare with `--hf-reference` from a CPU process while the
server is running. This avoids simultaneously holding HF and vLLM on one GPU.
They start one server, evaluate the base if requested, check/evaluate each
adapter sequentially, and stop the server even on failure or cancellation.
They keep thinking OFF, greedy decoding, 1,024-token replies, the existing
template/parser/40-turn cap, `FLASH_ATTN`, and the disabled FlashInfer sampler.
Environment/evaluation workers have CUDA hidden. No existing output is reused.

For a later authorized rai run (GPU ID is required):

```bash
export VERIFIED=/path/to/verified/seeds_k20_s0
export VLLM_GPU_MEMORY_UTILIZATION=0.55  # choose for available shared-card memory
bash scripts/rai_eval_student.sh 1 data/manifest/dev40.json \
  reports/runs/stage2e-rai-dev40 --base --workers 8 \
  early=/path/to/train/checkpoint-tokens-10407 \
  later=/path/to/train/checkpoint-tokens-20814
```

With no adapters, rai evaluates the base. With adapters, `--base` also requests
the base. `TASKS` can be frozen dev40 or dev140. Rai defaults to `.venv/bin/python`
for environments, the installed `tc-alignment-taskeq` Python for HF, the installed
`tc-alignment-vllm` serving Python, and the pinned local Gemma snapshot/ALFWorld
data paths. `ENV_PYTHON`, `TRAIN_PYTHON`, `VLLM_PYTHON`, `MODEL_PATH`, and
`ALFWORLD_DATA_ROOT` can be overridden. `CUDA_DEVICE_ORDER=PCI_BUS_ID` and offline
HF mode are always exported; workers default to eight.

For a later authorized HPG submission:

```bash
export ADAPTERS='early=/path/to/train/checkpoint-tokens-10407 later=/path/to/train/checkpoint-tokens-20814'
export VERIFIED=/path/to/verified/seeds_k20_s0
export TASKS="$PWD/data/manifest/dev140.json"
export OUT="$PWD/reports/runs/stage2e-hpg-dev140"
export INCLUDE_BASE=1 WORKERS=8
sbatch --export=ALL scripts/hpg_eval_adapters.slurm
```

HPG uses `configs/hpg.env`; `ADAPTERS` is whitespace-delimited and does not support
spaces inside paths. Rai accepts quoted `NAME=PATH` arguments with spaces.
Both new launchers use `--port 0`: a loopback bind asks the kernel for a free
port, recorded permanently in `$OUT/server/provenance.json` and while running
in `ready.json`. Readiness requires all requested model IDs and a listening
socket owned by the child process group, preventing another job's server from
being mistaken for this one. Output is `$OUT/<NAME>/eval`, with the base under
`$OUT/google/gemma-4-12B-it/eval`; adapter HF references and checks are under
`$OUT/<NAME>/{hf,check}`. Server artifacts stay under `$OUT/server`.

For manual serving/evaluation, use repeated `--lora-adapter NAME=PATH`, then run
the checker with `--snapshot PATH --adapter ORIGINAL_PATH --verified DIR
--model-name NAME --endpoint URL --server-provenance FILE --output NEW_DIR`.
The default checker computes HF references on GPU and contacts the existing
server; use the split `--hf-only`/`--hf-reference FILE` modes when memory is
shared. Pass `--model-name NAME --lora-check NEW_DIR/report.json
--server-provenance FILE` to `eval/run_suite.py`. The suite refuses adapter
evaluation without a passing report bound to that adapter and server run.
Omitting `--model-name` retains the base request name.

Source inspection of the installed vLLM 0.27.1 shows native support for the
trainer's full text projection names and Gemma prefix mapping; native loading
is the default, pending a real GPU equivalence check. If a runtime rejects the
names, preserve its failed output and retry into a new output with
`--lora-conversion targets` (only shorten config target names) or `gemma4`
(also rename `base_model.model.model.language_model.*` keys to
`base_model.model.language_model.model.*`). The launchers expose the same choice
as `LORA_CONVERSION`. These are explicit compatibility fallbacks, not automatic
retries for arbitrary server errors. Copies live under `server/adapters/NAME`;
the source checkpoint is never edited. Every rename/config change is recorded,
and tensor payload bytes, shapes, dtypes, and offsets must remain identical.
The HF reference always uses the original adapter; converted serving still
requires the same equivalence gate.

See [Stage 2e validation](reports/stage2e_validation.md) for CPU evidence and
the GPU checks still required on each host.

## Stage 2b: Seed-SFT

Select groups without opening environments or contacting a teacher:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python \
  data/seeds/select_seeds.py --k 10 --seed 0
```

The versioned selection rule shuffles the six types, assigns balanced quotas,
selects one trial per distinct TRAIN group, and freezes ordered replacements.
Selection records zero verified trajectories. `teachers/collect_seeds.py`
later runs up to three real teacher episodes per candidate, retains every
attempt, and promotes only real environment wins. Re-running the same command
resumes the collection; changed code/config/dependencies require a new output.

`train/sft.py` and `train/dataset.py` default to `--packing episode`: one sequence
per verified episode, using exactly the last turn's full conversation and final
generation prefix with `enable_thinking=False`. Every assistant reply and its
closing `<turn|>`/106 receive labels once per pass; system/user observations,
role headers, inter-turn newlines and the empty thought prefill remain masked.
`--packing turn` preserves the previous mode: one reply per sequence, with all
preceding conversation masked. Both modes have identical supervised tokens.
No episode is split or truncated; exceeding `--max-length` fails explicitly.

The unchanged Gemma template appends
`<|turn>model\n<|channel>thought\n<channel|>` before a generated reply, but renders
completed history as `<|turn>model\n<reply><turn|>` without the empty thought
channel. Episode packing therefore supervises all **non-final** replies with
that prefix difference; the final reply retains the evaluation generation
prefix. `mask_report.json` records the affected count (turns minus episodes),
turn indices, and both prefixes. Turn packing has zero affected supervised turns.

Training shuffles complete examples deterministically and accumulates a
supervised-token mean loss. In episode mode, an update closes after its whole
episodes reach the token budget; the last episode may overshoot it. Turn mode
retains the previous rule of fitting turns within the budget, with an oversized
turn alone. In both modes, checkpoint endpoints and the final token limit round
up to the first complete example reaching them, flushing the update. Requested
and actual token counts, overshoot, example/assistant-turn exposures, passes,
updates, timing and peak CUDA memory are recorded. `preparation.json` and the
printed preparation report include both modes' estimated forward tokens per
pass (input token counts, not runtime predictions), and supervised tokens per
pass. `mask_report.json` includes the same token estimates.

The trainer uses official `Gemma4UnifiedForConditionalGeneration` and PEFT LoRA,
bf16, rank 16/alpha 32/dropout 0, all text attention/MLP projections, gradient
checkpointing, microbatch 1, AdamW at 1e-5, and 512 supervised tokens per update.
Token and pass limits are configurable; when both are set, the first reached
stops training. Output directories must be fresh. Merging validates all
next-token logits on a fixed input before and after standalone reload.

See [Stage 2b validation](reports/stage2b_validation.md) for offline tests,
the real ReAct diagnostic mask report, complete HPG commands and remaining
GPU acceptance checks. Cluster paths and the separate training/reference
Python environments live in [configs/hpg.env](configs/hpg.env).

## Stage 2c: teacher reliability and nested seeds

Teacher HTTP attempts use full-jitter exponential backoff (base 2 seconds, cap
60 seconds, at most 10 attempts). `Retry-After` and `x-ratelimit-reset-*` can
extend that delay. Auth/400 errors fail immediately. Every attempt keeps its
sanitized ledger record, including error type, retry wait, limiter wait and token
reservation. Cache hits make no HTTP request and consume no limiter capacity.
When a response carries them, success, rejected-output and HTTP-failure attempt
rows include `rate_limit_headers` with the integer counters
`x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`,
`x-ratelimit-remaining-requests` and `x-ratelimit-remaining-tokens`.
Only these counters are retained; other headers and malformed values are omitted.

`models.teacher.rate_limit` in `configs/base.yaml` defaults to 450 requests/minute
and 180,000 tokens/minute **across all workers of one run** (also the Python
defaults). These are 90% of the provider limits reported for this key's
`gpt-6-luna` responses on 2026-10-04: `x-ratelimit-limit-requests: 500` and
`x-ratelimit-limit-tokens: 200000` (user-reported header observation).
A file-locked rolling minute ledger, keyed by provider model under
`<run>/data/teacher_rate_limit`,
reserves `ceil(prompt_characters / 4) + 256 * (message_count + 1) + max_output_tokens`.
Characters are counted in the canonical JSON prompt as Unicode characters,
not UTF-8 bytes. This follows OpenAI's character-based TPM estimate (about four
characters per token) plus the output cap, plus 8 framing tokens per message.
Reservations are separate from returned billed usage; failed requests keep their
reservations. The existing 429/`Retry-After` backoff remains the safety net, and
provider cooldowns are shared across workers.
Independent runs have independent limits; the commands below collect seeds
sequentially. Keep the limiter state on a filesystem with POSIX lock support.

`dev140.json` contains every frozen `valid_seen` dev game: **140 instances / 137
task groups**. The original manifest, split freeze, dev40 and sealed games are
unchanged. All nine `seeds_k{5,10,20}_s{0,1,2}.json` selections are frozen and
contain zero collected trajectories. At a fixed seed, smaller selections are
per-type prefixes of the same shuffled lists under `stratified_groups_v1`.

Offline, repeatable builders:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python data/manifest/build_dev140.py
for seed in 0 1 2; do
  for k in 5 10 20; do
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python \
      data/seeds/select_seeds.py --k "$k" --seed "$seed"
  done
done
```

Exact commands for a **later live run on rai** (not executed during Stage 2c):

```bash
cd /home/xueqi/hq/projects/ICML27_agent_distillation
export ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python eval/run_suite.py \
  --config configs/base.yaml --policy teacher --tasks data/manifest/dev140.json \
  --workers 8 --output reports/runs/teacher-dev140-stage2c

for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python teachers/collect_seeds.py \
    --config configs/base.yaml --selection "data/manifest/seeds_k20_s${seed}.json" \
    --attempts 3 --subset-k 5 10 --output "data/verified/seeds_k20_s${seed}" || break
done
```

The suite requires a fresh output directory. Repeating a collection command
resumes its saved attempts. Once K=20 completes, `--subset-k 5 10` writes
`data/verified/seeds_k20_sN/subsets/k5/manifest.json` and `subsets/k10/manifest.json`.
Smaller-K sets **reuse the same executed trajectories and receipt files**, with
no new calls, interactions or copied episodes. They inherit successes by original
selection slot, including its same-type replacement if necessary. This differs
from independently collecting each K, whose replacement boundary can differ.
Failed attempts for those slots remain in each view; the source manifest retains
all K=20 attempts and costs. Do not count the views as independent collections or
sum their overlapping costs. K remains the number of verified trajectories made
available to training, while the larger collection's total experience is disclosed.

For K=5 training, use `--verified data/verified/seeds_k20_s0/subsets/k5`; use the
analogous K=10 path or the collection root for K=20. Keep the parent collection
with its subsets: loading checks source hashes, receipt identity, actual wins,
TRAIN membership, distinct groups and quotas before constructing model inputs.

Required offline checks:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1 \
.venv/bin/python -m pytest -q
```

See [Stage 2c validation](reports/stage2c_validation.md) for evidence and limits.

## Baselines: Stage 3a ETO engineering port

`baselines/common/` supplies allowlisted episode conversations, frozen TRAIN
student exploration with replay, X5 batch shrinking and HF schedules, and
full-parameter FSDP2 SFT/DPO. `baselines/eto/` ports the official ALFWorld pair
rule and three-iteration loop. Source commits are in [PINS](baselines/PINS.md);
every port change is registered in [DEVIATIONS](baselines/eto/DEVIATIONS.md).
The existing LoRA `train/sft.py` remains a separate Seed-SFT baseline.

ETO uses K verified teacher seed trajectories (including nested subsets), then
explores those same K tasks once per iteration with the current policy, greedy,
thinking OFF, 40 turns and 1,024 reply tokens. Successful rollouts are skipped
ties; complete verified failures form expert-vs-full-rollout pairs. Unknown or
incomplete execution stops the stage. P=0 stops with an explicit receipt.
Student exploration is training-eligible expansion, never additional trusted
seeds. Replay episodes/steps and failed attempts are disclosed separately.

SFT retains ETO's 2e-5/cosine/.03 warmup/3 epochs; DPO retains iteration 1
1e-6/beta .1, iterations 2–3 5e-7/beta .5, constant-with-warmup/.1/3 epochs.
Weight decay is zero. Only the global batch shrinks:
`b' = min(b_orig, max(1, floor(N_examples / 8)))`, from 64 for SFT and 32 for
DPO. Microbatch is one episode or pair; each pair contains two continuations.
Whole episodes use the exact existing Gemma OFF spans and a 32768 limit with
no truncation. SFT averages each episode's assistant-token CE, then averages
episodes; DPO uses summed assistant log-probs and a frozen iteration-start
reference. Full text parameters have fp32 master weights and bf16 FSDP2 compute;
non-text base components are copied unchanged into the standalone checkpoint.

On HPG, for a **later authorized run** (these commands were not submitted):

```bash
source configs/hpg.env
cd "$PROJECT_ROOT"
export VERIFIED="$PROJECT_ROOT/data/verified/seeds_k20_s0"
export ETO_OUT="$PROJECT_ROOT/reports/runs/eto/k20-s0"
sbatch --export=ALL scripts/hpg_eto.slurm
```

The job requests four B200s, 32 CPUs and 480G on account/partition/qos
`fsu-compsci-dept` / `hpg-b200` / `fsu-compsci-dept`. Training seed is always 0;
`s0` in the tree is the **support collection seed**. For K=5/10, use
`seeds_k20_sN/subsets/k5` or `subsets/k10`, and a distinct `ETO_OUT` per (K,sN).
The same command resumes hash-checked completed stages. Incomplete attempts
remain preserved; a retry restarts that stage and adds its actual costs to the
disclosure. `ETO_THROUGH=sft` (or `explore-1`, `pairs-1`, `dpo-1`, etc.) ends at
a stage boundary. `ETO_MODE=prepare` only freezes the input identity/task list.

Evaluate every available stage on dev140 in a **separate one-GPU job**:

```bash
sbatch --gpus=1 --cpus-per-task=8 --mem=48G \
  --export=ALL,ETO_MODE=evaluate scripts/hpg_eto.slurm
```

Evaluation reuses `scripts/eval_adapters.sh`, serving each standalone checkpoint
through `--merged-model-path`. `models.json` lists SFT and completed iter-1/2/3
checkpoints; there is no sealed evaluation, dev-driven generation or checkpoint
selection. A single full checkpoint can also use `scripts/hpg_eval_adapters.slurm`
with `MERGED_MODEL_PATH`, empty `ADAPTERS`, `INCLUDE_BASE=1` and frozen dev140.

See [Stage 3a validation](reports/stage3a_validation.md) for exact CPU checks,
standalone commands, artifacts and the remaining GPU acceptance boundary.

## Stages 4a–4b: mechanism analysis tooling

`analysis/mechanism/` implements the fixed measurements and I0–I5 arms in
[MECHANISM_PLAN.md](docs/MECHANISM_PLAN.md), including Amendment 1 (taxonomy,
commit `862eaed`) and Amendment 2 (I0 control and paired excess rescues,
commit `07af904`). Stage 4b also records batch-invariant serving as an explicit
protocol setting; it does not re-run any evaluation or branch experiment.
These are diagnostic tools, with
`purpose=diagnosis`, zero training exposure, and no sealed-task access. The
Stage 4a validation uses offline fakes and reads existing dev artifacts; it
does not run a live FCE or intervention experiment.

Start by freezing a taxonomy snapshot into a **new** output directory. `--run`
is S; `--compare` is a teacher or base reference (role inferred from provenance,
or specified by `--compare-role`). Supplying both references emits both G and R:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m analysis.mechanism.taxonomy \
  --run reports/runs/eval/k20s0-rai-dev140/k20s0-p03/eval \
  --teacher-run reports/runs/teacher-dev140 \
  --base-run reports/runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval \
  --output reports/runs/my-mechanism-taxonomy
```

This writes `taxonomy.json`, `shares.md`, `review_samples.json` (up to ten
seed-0 samples per matching label per run, with triggering steps/evidence), and
provenance. An unfinished reference remains an explicitly partial snapshot:
unknown/missing outcomes do not become failures or enter G/R. A later snapshot
needs a new output. Shares use known failed episodes as their denominator;
all matching labels and primary labels are reported separately. Successes
remain unlabelled. Indices are reset=0 and actions=1..N.

Primary labels follow this order: F-format, F-inadmissible, F-goal, F-search,
F-precondition, F-loop, F-cap. F-format requires at least three trailing parser
errors or parser errors on at least 25% of steps. F-inadmissible separately
requires inadmissible commands on at least 25% of steps; neither fraction pools
the two outcomes. Only F-format enters the separate format-confound cohort.

The classifier uses typed PDDL facts only for labels. F-search searches visible
observation text, excluding the `Your task is to:` instruction, which names the
goal by construction. Admissible commands, actions and audit facts are not
sightings. Navigation/open/examine of other receptacles are search operations;
F-goal counts legal executed actions: moving a goal-type object to the wrong
destination, or heat/cool/clean/use on the wrong object type (use/toggle selects
the target lamp for light tasks). Placement into the task's own transformation
appliance is excluded: microwave for heat, fridge for cool, sinkbasin for clean.
These exclusions do not exempt wrong-type transformation/use actions. Target
and required movable receptacles are also allowed destinations. Taking an
object alone and inadmissible attempts do not trigger F-goal.
F-precondition checks actual holds, placements, transformation predicates and
lamp toggles. Evidence records both missing predicates and the relevant state.
An unsuccessful episode matching no rule below the cap remains unclassified.

Inspect a goal/fact mapping without opening an environment:

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.mechanism.goals \
  --task-id 'valid_seen/pick_heat_then_place_in_recep-Apple-None-CounterTop-1/trial_0'
```

Plan FCE probes without constructing clients or environments:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m analysis.mechanism.fce \
  --gap-set reports/runs/my-mechanism-taxonomy/taxonomy.json \
  --output reports/runs/my-mechanism-fce --dry-run \
  --max-teacher-calls 400 --max-teacher-calls-run 12000
```

The dry run reports coarse points (including the episode endpoint), adaptive
refinement and worst-case call estimates. Live use of the same command without
`--dry-run` is a separate experiment. Each probe executes two teacher samples;
their sample indices enter the cache key. A completed probe is true if either
sample wins. Incomplete probes stay unknown. The scan refines the first coarse
1→0 interval; it does not claim to find unprobed transient drops between two
successful coarse points. T(0)=0 is reported separately.

Every teacher HTTP attempt, including retries and interrupted reservations,
counts against explicit episode/run caps. Cache hits consume no new attempt.
The existing teacher cache, usage ledger and shared rate limiter remain in use.
Caps are fixed on resume; exhaustion is recorded as incomplete, never failure.
Use `--budget-dir OWNED_MECHANISM_WORKSPACE` to share caps across FCE and arms.

Stage 4c adds `--workers N` to both `fce` and `interventions` (default **1**).
FCE uses an episode process pool; with N > 1, each episode can run its two
teacher samples concurrently in separate processes. Its adaptive probe and
refinement order remains sequential. Interventions pool individual
(state, arm, sample) continuations. Every branch creates its own environment;
spawned processes isolate ALFWorld's Python/NumPy RNG state. I3/I4 sample zero
produces the shared output before that arm's other samples start, preserving
which sample records the billed usage; a file lock also protects the output.

Seeds, cache keys, sample indices and attempt paths do not use N. Summaries
are written in frozen episode/arm/sample order after workers finish. Their
budget reservation lists are sorted; the durable budget and usage ledgers
retain actual arrival order. HTTP-attempt reservation is atomic across
processes, including retries. Conservative scheduling headroom preserves
serial budget priority: if concurrent units could hit a cap, dispatch narrows
until earlier units finish. This can reduce concurrency with small caps or
large retry limits. Scheduling headroom is never billed. Denied units retain
unknown outcomes and `incomplete.json`, and remain incomplete in episode and
teacher usage records. Completed attempts are validated and skipped on resume.

Worker count is excluded from both workspace and resume identities. An exact
hash-pinned compatibility entry in
`analysis/mechanism/resume_compatibility.json` permits the Stage 4b code at
`3045b60` to resume with this engineering implementation, including checkout
relocation of content-identical inputs. Config, dependencies, caps, model and
data identities still must match. Original identity/provenance and completed
receipts remain intact; `resume_compatibility/` records the transition and new
attempts record current source hashes. Unlisted code changes still refuse
resume. A workspace frozen under Amendment 2 cannot resume against Amendment 3's
different plan hash; this engineering compatibility entry does not authorize a
protocol change. Offline fake-pipeline tests compare byte-identical receipts and
summaries for N=1/4 with fixed clocks/IDs and test interrupted-workspace resume
and contended caps. Live API samples, timestamps and latency retain their real
values; this engineering validation makes no new model reproducibility claim.

`branch` accepts `--episode FILE --t N --policy teacher|student --arm takeover|I0|…|I5`.
Student continuation requires `--endpoint URL --model-name NAME`; adapters also
require `--server-provenance FILE --lora-check FILE`. `--sample-index 0` is
greedy; positive indices use temperature 0.7, with distinct seeds derived from
`--sampling-seed N`, the parent, prefix and sample index. Caps and
a fresh `--output` are required. Every branch resets and executes the prefix,
using the shared replay checks for observations, admissible commands, rewards,
termination and audit hashes. Parser errors consume turns without environment
steps, exactly as in evaluation. Prefix turns and I2's extra real information
action count toward the unchanged 40-turn limit. The output contains normal
`episode.json`, events, blobs and provenance plus the branch parent/t/arm/sample.

For a later authorized run against an already checked server:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m analysis.mechanism.interventions \
  --gap-set reports/runs/my-mechanism-taxonomy/taxonomy.json \
  --fce reports/runs/my-mechanism-fce/fce.json \
  --verified data/verified/seeds_k20_s0 --output reports/runs/my-mechanism-arms \
  --endpoint http://127.0.0.1:PORT/v1 --model-name k20s0-p03 \
  --server-provenance SERVER/provenance.json --lora-check CHECK/report.json \
  --max-teacher-calls 400 --max-teacher-calls-run 12000
```

Every arm, including I0, has greedy + four samples at T=0.7. Sample seeds are
derived from the common sampling seed, task ID and sample index, identically
across arms. I0 is the unmodified control. It still records whether S reproduces
its logged next action and failure, but divergence does not exclude a state or
skip I1–I5. Real environment prefix replay and consistency checks remain mandatory.
I1 restates only visible feedback, with the latest explicit contents observation
winning. I2 prioritizes inventory after `Nothing happens.` when visible feedback
shows a held object; otherwise it examines the current receptacle when legal,
or uses look. Its real action/feedback become an ordinary additional turn.
I3 uses an advice-only teacher system template, appends `Advice:` to S's user
message, and records both template and instruction hashes; action-bearing
advice is rejected and retained. I4 inserts the teacher action. One I3/I4 output
is fixed across that arm's five student continuations, with reuse explicitly
recorded and billed once. I5 uses the same task type, then the greatest number
of shared object/movable/target types, then verified seed order. It renders only
observation/action text. The exact seed collection is checked against S's
training provenance; absent task types are reported separately. The supplied
adapter must match the evaluated S. Add `--subject base` with the configured
base model to repeat the arms on the same S states and seeds; base I0 is a
control, without requiring the base to reproduce S's action.

I1/I3/I5 append text to just the intervened user message. The exact appended
message remains in subsequent request histories. I2 adds its actual information
turn, and I4 replaces just the next assistant action. Shared evaluation prompts,
the parser, OFF template, 1,024-token cap and default greedy decoding are unchanged.
All prompts retain the visible-field whitelist. Teacher text is advice, never
verification. Usage, raw responses, rejected attempts and failures are retained.

The resumable launcher runs one checked vLLM server for S and optional base:

```bash
# Later live execution only; not part of Stage 4a validation.
MAX_TEACHER_CALLS=400 MAX_TEACHER_CALLS_RUN=12000 INCLUDE_BASE=1 WORKERS=8 \
  bash scripts/rai_mechanism.sh 1 reports/runs/my-mechanism-taxonomy/taxonomy.json \
  data/verified/seeds_k20_s0 reports/runs/my-mechanism \
  k20s0-p03=/path/to/checkpoint-tokens-31221
```

It validates the gap/adapter/seed identity before GPU work, generates HF
references before serving, allocates a free port via `serve_student.py --port 0`,
and requires the existing equivalence gate before FCE → arms → readout. Runtime
and model-path overrides match `rai_eval_student.sh`. `WORKERS` defaults to
**8** and is passed to FCE and both student/base intervention runs; set it to
1 for serial execution. Every restart uses new
server/check directories. Only owned, hash-bound analysis workspaces resume;
completed branch attempts are checked and skipped, interrupted attempts are
preserved and retried in new directories. Original evaluations and seed data
are never edited. The launcher stops its server/workers on exit or gate failure.

Read an existing experiment without model/environment calls:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m analysis.mechanism.readout \
  --gap-set reports/runs/my-mechanism-taxonomy/taxonomy.json \
  --fce reports/runs/my-mechanism-fce/fce.json \
  --interventions reports/runs/my-mechanism-arms/interventions.json \
  --output reports/runs/my-mechanism-readout
```

The JSON and Markdown give |G|, taxonomy shares, FCE positions, rescue rates and
95% Wilson intervals, and the fixed 2×2 table. Raw rescue requires greedy success
**and ≥3/4 sampled successes**, for I0 as well as I1–I5. Each arm is shown beside
I0 on the same FCE states with complete results for both. A **paired excess
rescue** counts a state only when the arm rescues and I0 does not. Wilson
intervals use this binary state count, not a subtraction of marginal rates.
Section 6 cells and the I4 execution/capacity label use paired excess rescues.
F-format and missing seed task types retain separate cohorts. I0 replay noise
is retained as audit metadata. Missing controls never count as control failures.
Cells use |G| as denominator; paired intervals disclose unknown pairs and their
matched task IDs. JSON retains all available raw rates under `arm_rescue`,
arm/I0/excess rates on matched states under `paired_arm_rescue`, and raw and
paired per-state outcomes under `arms_by_task` and `paired_excess_by_task`.
Optional `--compare-interventions FILE` adds exact paired McNemar tests on excess
rescues, alongside the historical 12/140 cross-server base-vs-base discordance
reference (non-batch-invariant serving, not an estimate for the new mode). Optional
`--coverage FILE` accepts `{task_id: {"covered": true, "evidence": ...}}` audit
labels. Task-type overlap alone never establishes needed-step coverage, and
M1 stays conditional without it. Section-4 knowledge probes, training fit and
earlier/later checkpoint controls are still required before interpretation;
this engineering stage makes no mechanism claim.

See [Stage 4a validation](reports/stage4a_validation.md) for the offline test
results and real-run taxonomy sanity check.

### Stage 4b serving protocol

The 2026-10-05 rai pilot reported identical base trajectories on all 40 dev40
instances across two vLLM servers with `VLLM_BATCH_INVARIANT=1`, at approximately
2.8× serving time. Its read-only artifacts are
`reports/runs/eval/bi-pilot-dev40-{1,2}`. This supports the protocol switch recorded
under Amendment 2; it does not establish reproducibility for every adapter or
hardware configuration.

`eval_adapters.sh`, `rai_eval_student.sh`, `hpg_eval_adapters.slurm`,
`hpg_eval_student.slurm` and `rai_mechanism.sh` accept `BATCH_INVARIANT=0|1`,
default **1**, and export `VLLM_BATCH_INVARIANT` explicitly. For a manual
`harness/serve_student.py` invocation, set `VLLM_BATCH_INVARIANT=1`; an unset
variable means off and is recorded as such. Server provenance contains
`config.server.batch_invariant` (boolean) and the exact child environment flag.
Suites copy the server's value into `config.suite.batch_invariant` and
`summary.json`; they never infer it from the evaluation worker environment.
Missing legacy provenance stays null/unknown. Teacher suites use null because
they do not use this vLLM server. Mechanism resume identities bind the setting,
so changing it requires a new workspace.

`analysis/k_ladder.py` refuses to combine different student/base settings,
including known versus unknown, before writing a table. For an explicitly
mixed diagnostic table, pass `--allow-mixed-serving`; JSON, Markdown and plot
labels identify each setting. The black-box teacher is marked not applicable
and does not participate in this guard. Historical artifacts are unchanged;
affected paper evaluations and branch replays must be re-run under the new
protocol in separate experiment worktrees.
