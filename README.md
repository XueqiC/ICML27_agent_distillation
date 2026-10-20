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

Stage 3d adds an explicitly flagged exception for the **LEAP training teacher**:
an executed handcoded-expert trajectory supplies an extracted privileged summary
to the correction teacher only. Student prompts and training contexts retain the
visible-field whitelist. The separate 0%-privileged ablation uses no such summary.

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

Before naming conversion, unmerged adapters automatically receive the
[Gemma-4 k_eq_v serving-equivalence fix](docs/LORA_SERVING_K_EQ_V.md).
It copies K LoRA tensors into missing V slots under the server output directory,
with recomputed derivation hashes bound to the unchanged original HF adapter.

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
When remaining requests are zero or remaining tokens cannot cover another
reservation of the same size, the limiter pauses until the corresponding reset
or `Retry-After` deadline (60 seconds if neither supplies a positive delay).

`CappedTeacher` retries only rate-limit HTTP 429s, up to 20 transport attempts in
one logical-call slot. It honors `Retry-After`; otherwise it uses full-jitter
exponential backoff with a 5-second base and 300-second ceiling. Quota, billing
and credit errors halt immediately, as do all other transport failures. Every
retry keeps its ledger entry and non-refundable token reservations. A recovered
429 does not halt the next call or a restart; an interrupted call still does.

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
Independent runs have independent limits by default. Set
`TEACHER_RATE_LIMIT_DIR=/shared/cluster/teacher_rate_limit` in every run that
should share one limiter; it overrides run-specific directories, including
explicit Python directory arguments. Use matching RPM/TPM settings and a
filesystem with POSIX lock support. The effective directory is recorded in each
attempt's usage ledger. The commands below collect seeds sequentially.

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

## Stage 2f: extend the nested seed ladder to K=100

The engineering ladder covers **K in {1, 2, 5, 10, 20, 50, 100}** for support
seeds 0/1/2. The three frozen `data/manifest/seeds_k100_sN.json` selections use
the unchanged `stratified_groups_v1`: in seeded type order, quota i is
`K // 6 + (i < K % 6)`. Each type's selected-plus-replacements list is identical
to its K=20 list; only the boundary moves. These files claim zero new verified
trajectories. Rebuild and check them offline with:

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python data/seeds/select_seeds.py \
    --k 100 --seed "$seed" --extend-selection "data/manifest/seeds_k20_s${seed}.json"
done
```

**Ordering clarification:** existing K=5/K=10 views use original selection slots,
including a slot's replacement. They are not sorted by successful candidate rank:
for example, s0's first type retains successful candidate ranks 4, 5, 2, 3 for
slots 0, 1, 2, 3. Sorting those successes by candidate rank would change the old
K=5/K=10 receipts. Stage 2f prioritizes the requested hard receipt-identity check:
preserve old slots, then fill new slots with unattempted candidates in the frozen
per-type order. K=1/K=2 inherit the first one/two types' first K=5 receipts.

The extension command reads the completed K=20 source, verifies its K=5/K=10
views, and copies every attempt's receipt, trajectory, raw teacher evidence and
cache to the fresh K=100 output. Copies are atomic and independent files; the
source tree is never modified. All attempted source instances, including failures,
are skipped. Each new candidate has at most three attempts; its first real win
fills the next slot, and exhausted failures advance to the next same-type
candidate. Interrupted requests remain disclosed. Resume checks source hashes,
copied evidence, config, code and dependencies before continuing. Sampling,
teacher/student identity and environment protocol must match the source; host
paths and rate limits may change and both configs remain recorded.

Commands for a **later live run on rai** (not run during this engineering stage):

```bash
export ALFWORLD_DATA_ROOT=/path/to/alfworld/data/json_2.1.1
bash scripts/rai_collect_k100.sh
# After an interruption, with the same frozen code/config:
bash scripts/rai_collect_k100.sh --resume
```

The launcher runs s0, s1, s2 sequentially with CUDA disabled and separate logs
under `logs/`. It checks all three outputs before making any call and refuses
existing outputs unless `--resume` is supplied. `PROJECT_ROOT`, `ENV_PYTHON`,
`COLLECT_CONFIG`, `COLLECTION_ROOT` and `LOG_ROOT` are optional overrides.
The equivalent single-seed command is:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python teachers/collect_seeds.py \
  --config configs/base.yaml --selection data/manifest/seeds_k100_s0.json \
  --extend-from data/verified/seeds_k20_s0 --output data/verified/seeds_k100_s0 \
  --attempts 3 --subset-k 1 2 5 10 20 50
```

A completed extension creates views at `seeds_k100_sN/subsets/k{1,2,5,10,20,50}`.
Before publishing completion or materializing/loading a view, validation requires
exact original K=20/K=10/K=5 verified receipts (including task IDs, attempt numbers
and trajectory hashes), and preserves their failure receipts too. Mismatches
raise errors. Training/evaluation consumers use these paths via the existing
verified loader; K=100 uses the collection root. Keep the original K=20 tree and
its subsets beside K=100 when copying to another host: the extension's relative
source reference binds original provenance and the old identity anchors.

The extension manifest separates inherited and new attempts/trajectories and
retains all original token usage, failures and real transitions. Views create no
new calls or interactions. Overlapping collections/views are not independent
costs; disclose the complete K=100 collection's experience even for small-K runs.

`configs/seedsft_dose_grid.json` remains v1. The v2 template,
`configs/seedsft_dose_grid_v2.json`, adds twelve runs: K=1/2/50/100 x s0/1/2.
Its counts and endpoints intentionally remain unresolved until the collections
exist. At submission time, measure them on CPU using the pinned local student
tokenizer and the actual training loader (assistant labels including closure,
episode packing, max length 32768, no truncation):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python scripts/prepare_seedsft_dose_grid.py \
  --snapshot /path/to/pinned/local/gemma-4-12B-it \
  --data-root /path/to/tree/containing/data \
  --output /path/to/fresh/seedsft_dose_grid_v2_final.json
```

This helper loads local tokenizer files only; it performs no model training or
job submission. It records data/tokenizer/code hashes and versions. Checkpoints
are passes {3,5,10,20,30,40} union tokens {100k,200k,400k}; a token point within
5% of its token budget from a pass point merges into that pass point. Merged
labels are disclosed, and run length is the largest remaining endpoint.
Both HPG sweep scripts accept `DOSE_GRID=/path/to/seedsft_dose_grid_v2_final.json`
and reject the unresolved template. For a later v2 array submission, override
the v1 default with `--array=0-11%4`; analysis accepts the same file via `--grid`.
Neither the collections nor final measured grid are produced by this engineering
change. Required validation is the offline full test suite shown above.

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

## Baselines: Stage 3c IPR and deterministic SAG sampling

`baselines/ipr/` implements the pinned IPR algorithm on the K verified seed
trajectories. See [deviations](baselines/ipr/DEVIATIONS.md) and
[offline validation, HPG commands and compute estimates](reports/stage3c_validation.md).
This engineering port makes no learning or baseline-reproduction claim.

SFT uses the shared ETO training machinery with IPR's original global batch 48,
lr 2e-5, cosine, .03 warmup and three epochs. A one-time stage measures five
SFT-model MC continuations after every expert action using actual reset/replay.
Each of six iterations explores all expert branch points greedily with the
current iterate, measures five SFT-model MC continuations after each first agent
action, then constructs official trajectory and entire-continuation pairs.
Both thresholds are .01 and the ALFWorld shorter-continuation flag is preserved.
The disabled upstream expert-MC block is explicitly activated; the byte-identical
WebShop/ALFWorld sampler's erroneous ICL cut is fixed by replaying real episode
actions only.

DPO uses summed assistant log-probs plus summed chosen NLL (weight 1), beta .1,
lr 3e-6, constant_with_warmup, .1 warmup and three epochs. Initialization is the
current iterate; the reference remains SFT across all six iterations. Only the
current iteration's pairs train. P and counts by kind are recorded before
training; P=0 carries the policy forward unchanged and continues the loop.
Both stages apply X5 batch shrinking from 48 and training seed 0. Full text
parameters use four-GPU FSDP2, SDPA and 32768 context without truncation.

Every stage uses the common harness, OFF and a 40-turn total cap, including
replayed turns. Exploration/evaluation use the 1024-token cap; MC uses the
specified 512-token cap, T=1/top_p=1. `--workers` parallelizes independent CPU
environments sharing one vLLM server per stage. Batch-invariant serving is
required. Rollout and request seeds exclude worker count. Hash-bound completed
stages resume; partial attempts are preserved and retried in new directories.
Raw calls, rejected/unknown outcomes, real replay costs and training exposure
remain auditable in `disclosure.json` and stage artifacts.

For a later authorized HPG run, set `VERIFIED`, `IPR_OUT`, `BATCH_INVARIANT=1`
and submit `scripts/hpg_ipr.slurm`. `IPR_MODE=prepare` freezes inputs only;
`IPR_THROUGH=sft`, `expert-mc`, `explore-1`, `agent-mc-1`, `pairs-1`, `dpo-1`,
etc. stop at resumable stage boundaries. The linked validation report provides
a staged job chain to fit the 12-hour wall limit. `IPR_MODE=evaluate` uses the
existing dev140 full-checkpoint path with one GPU, separately from training.

SAG now derives a per-request seed from `(sampling_seed, task_id, turn_index)`
and records it in the request/call/step metadata. Sampling seed defaults to 0;
`eval.run_suite --decoding sag --sag-sampling-seed N` can set it explicitly.
Task identity is kept out of prompts. Default greedy evaluation requests and
mechanism sampling seeds are unchanged; AgentEvol exploration is greedy.

## Baselines: Stage 3b AgentGym BC / AgentEvol and Kang SFT + SAG

`baselines/agentgym/` and `baselines/kang/` use the Stage 3a infrastructure.
These are engineering ports, with offline validation and no learning results.
See [AgentGym deviations](baselines/agentgym/DEVIATIONS.md),
[Kang deviations](baselines/kang/DEVIATIONS.md), and
[Stage 3b validation and exact HPG commands](reports/stage3b_validation.md).

AgentGym BC is iteration 0. AgentEvol trains **from the base model** on D0..D3,
one epoch each, lr 1e-5, linear decay, `int(.1 * total_steps)` warmup, wd 0,
clip 1, full text-parameter FSDP2 on four B200s. The X5 batch shrinks from 32
using the current dataset's trajectory count. After each training, one greedy
rollout runs on each of the **K seed TRAIN tasks only**. Filter reward >.99,
keep the first success per task within that round, prepend Fm to all of Dm,
and preserve duplicate tasks/trajectories across rounds. Empty Fm still runs
the next round. D4 is produced and never trained. Evaluation is separate.

Kang trains a PEFT LoRA adapter from K seeds: r64/alpha128/dropout .05 on
q/k/v/o/gate/up/down text projections, lr 2e-4, two epochs, X5 batch from 8,
linear decay and zero warmup. One full trajectory is one sequence. The shared
trainer keeps every assistant reply plus native 106 supervised, uses SDPA,
training seed 0, and rejects lengths above 32768 without truncation. There is
**no first-thought prefix**: missing upstream generation components are neither
run nor recreated. The official repo pins TRL 0.19.1 but only bounds Transformers
`>=4.0.0`; the deviation registry records this dependency-verification limit.

SAG is opt-in: `eval/run_suite.py --policy student --decoding sag`. It requests
eight samples per step at temperature .4, top_p .8, max1024. Candidates pass
the unchanged parser plus the current admissible-action check (Thought remains
optional). The largest action-string group wins, ties favor the earliest group,
all singletons choose the first passing sample, and no passing samples choose
sample 0. Only the winner reaches the environment. Step metadata retains every
candidate, parsing decision and vote; total usage includes all samples. Default
greedy requests, prompt, parser, template and caps are unchanged.

Both pipelines use hash-bound resumable stages and preserve failed attempts.
`disclosure.json` includes exposure across retries; K stays the original trusted
trajectory count. AgentGym additionally writes `datasets.json` (D0..D4 paths),
`models.json`, and per-iteration Dm/Fm/exposure records. A partial stage restarts
in a new attempt; completed stages are hash checked and reused. Unknown or
incomplete exploration stops its stage, retaining evidence and unknown costs.

For later HPG execution, with paths adjusted to the desired K/support seed:

```bash
source configs/hpg.env
cd "$PROJECT_ROOT"
export VERIFIED="$PROJECT_ROOT/data/verified/seeds_k20_s0"
export BATCH_INVARIANT=1
export AGENTGYM_OUT="$PROJECT_ROOT/reports/runs/agentgym/k20-s0"
sbatch --export=ALL scripts/hpg_agentgym.slurm
# BC only: export AGENTGYM_THROUGH=train-0 before submission.
# Separate evaluation of every completed full checkpoint:
sbatch --gpus=1 --cpus-per-task=8 --mem=48G \
  --export=ALL,AGENTGYM_MODE=evaluate scripts/hpg_agentgym.slurm

export KANG_OUT="$PROJECT_ROOT/reports/runs/kang/k20-s0"
sbatch --export=ALL scripts/hpg_kang.slurm
```

Kang evaluates through `scripts/hpg_eval_adapters.slurm` with `DECODING=sag`,
`BATCH_INVARIANT=1`, and `ADAPTERS=kang-sft=CHECKPOINT_PATH` from `models.json`.
The existing HF+PEFT/vLLM equivalence gate remains mandatory. Both new launchers
enforce `VLLM_BATCH_INVARIANT=1` through `BATCH_INVARIANT=1`; no jobs were submitted
during implementation. Full commands, including adapter-path extraction and
resume boundaries, are in the validation report.

## Baselines: Stage 3d LEAP engineering port

`baselines/leap/` implements **LEAP (privileged teacher)** and **LEAP (0% privileged
ablation)** from `a3376d8`. See [deviations](baselines/leap/DEVIATIONS.md) and
[validation, exact HPG commands and cost estimates](reports/stage3d_validation.md).
This is offline engineering validation, with no learning result claimed.

Iteration 0 trains on all K verified teacher seeds, using their existing visible
Thoughts. Each step is a separate Gemma OFF conversation, with only its final
assistant reply and native token 106 supervised. No `annotate_reason` call or
85/15 holdout is used in this registered dose regime; eval/save step 200 cannot
be reached. Every later iteration explores only the seeds' TRAIN tasks, once
each, through the unchanged greedy 40-turn/1024-token harness and replay checks.

The privileged variant first executes ALFWorld 0.4.2's `AlfredExpert(handcoded)`
through a LEAP-only backend, then requests one extracted summary per successful
expert episode. Availability and failures are recorded under `audit/`. Tasks
without an executed expert win/extracted state are excluded from privileged
correction, while remaining in BC and student exploration. The 0% variant skips
unused oracle collection and corrects all known failed seed-task rollouts with
the official no-privileged template. Its learning pipeline is otherwise the same.

One logical teacher request corrects every step of a failed trajectory. Training
context always comes from the **original student history**; teacher echoes,
summaries, state fields and audit data are excluded. Corrected actions remain
unfiltered by admissibility, with counts recorded. These are sampled labels with
unknown execution verification. The official correction prompt prohibits hidden
facts in reasons; structural isolation and literal-summary rejection do not
establish semantic absence of privileged knowledge from generated labels.

Iterations 1–3 load example sets 0..i, independently shuffle each with seed 0,
subsample each to the smallest set, then shuffle the aggregate. An empty set
stops with an explicit empty-aggregate result. Each iteration attaches a fresh
LoRA to the previous **merged** model: r128/alpha64/dropout .05, text q/v only,
lr 3e-5, cosine, warmup 10, clip .3, one epoch, fused AdamW, bf16 and SDPA.
Global batch shrinks from 32 by X5; the official microbatch four shrinks only
when needed to fit that batch. Full context is retained up to 32768; longer
examples fail. Every merge passes an all-vocabulary logit/reload check before
publication and serving through `--merged-model-path`.

The shared teacher adapter supplies cache, sanitized raw text, usage ledger and
limiter. Each task has at most three durably reserved API attempts per extraction
or correction stage, including parse failures and interruptions. Official
T=.3/top_p=1 is tried on an actual request; explicit unsupported-parameter errors
switch subsequent requests to configured default sampling. Live acceptance has
not been tested. Extraction/correction output caps default to 4096/16384 tokens,
separate from the unchanged 1024-token student cap.

For later authorized execution on one B200, use `scripts/hpg_leap.slurm` with
`VERIFIED`, `LEAP_OUT` on `/blue`, and `LEAP_VARIANT=privileged` or `zero` in
distinct output trees. Budget roughly 24 GB per merged checkpoint, four per
variant, plus retained adapters/attempts. `LEAP_MODE=prepare` freezes inputs only;
`LEAP_THROUGH=merge-0`, `explore-1`, `correct-1`, `data-1`, `train-1`, etc. stop at
resumable stage boundaries. `LEAP_MODE=evaluate` evaluates published iter0..3
on dev140 separately, with batch-invariant serving and no training updates.
No jobs, servers, GPU work or API calls were performed for this port.

## Stages 4a–4d: mechanism analysis tooling

`analysis/mechanism/` implements the fixed measurements and I0–I5 arms in
[MECHANISM_PLAN.md](docs/MECHANISM_PLAN.md), including Amendment 1 (taxonomy,
commit `862eaed`) and Amendment 2 (I0 control and paired excess rescues,
commit `07af904`) and Amendment 3 (closest-approach intervention state,
commit `510d002`). Stage 4b also records batch-invariant serving as an explicit
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

The dry run reports coarse points (including the episode endpoint), budget-drop
refinement, closest-approach densification and worst-case call estimates. Live
use of the same command without `--dry-run` is a separate experiment. Each probe executes two teacher samples;
their sample indices enter the cache key. A completed probe is true if either
sample wins. Incomplete probes stay unknown. Stage 4d completes the unchanged
coarse grid (0, 4, 8, …, endpoint), then retains the original first coarse
1→0 interval refinement as the **budget FCE** (`budget_fce`, with `fce` retained
as an alias). It does not claim to find unprobed transient budget drops.

Each probe records `probe_samples`: success and `teacher_steps_to_go` for each
sample. Steps-to-go count policy turns after the replayed prefix, including
parser errors and inadmissible actions. `D_curve` stores D(t), the minimum of
successful sample lengths, or null if neither succeeded. Around the earliest
coarse prefix t_c with minimum D, probe every prefix in [max(0, t_c−3), t_c],
reusing the same two samples at already probed prefixes. After this densification,
`t_star` is the earliest prefix attaining minimum D across **all** probes,
including budget refinement. `D_0`, `D_t_star` and `turns_remaining_at_t_star`
(40−t*) are recorded. I0–I5 now run at t*, including when no budget drop exists.
T(0)=0 alone no longer excludes a state if a later takeover succeeds. If no
probe succeeds, status is `teacher_unsolvable_from_student_states`. Interrupted
or capped scans retain observed distances but leave t* unset and stay incomplete.

Every teacher HTTP attempt, including retries and interrupted reservations,
counts against explicit episode/run caps. Cache hits consume no new attempt.
The existing teacher cache, usage ledger and shared rate limiter remain in use.
Caps are fixed on resume; exhaustion is recorded as incomplete, never failure.
Use `--budget-dir OWNED_MECHANISM_WORKSPACE` to share caps across FCE and arms.

To reuse teacher outputs across the protocol change, use a **new** workspace
and pass `--seed-teacher-cache-from OLD_WORKSPACE` to `fce`, or set
`SEED_TEACHER_CACHE_FROM=OLD_WORKSPACE` for `rai_mechanism.sh`. The source may be
a direct FCE workspace or its launcher root; the latter resolves `fce/`.
The intended existing source is `reports/runs/mechanism/k20s0-p20-bi/run`.
This option copies `teacher_cache/` and `teacher_blobs/` only. It never writes
or locks the source, and does not import old budgets, ledgers or branch receipts.
Provenance records the source path, cache/blob entry counts and manifest hash;
`teacher_cache_seed.json` freezes the relative file paths and SHA-256 hashes.
Resume reuses that frozen manifest even if the live source grows. Conflicting
destination entries or changed source content fail instead of being overwritten.
Dry runs do not copy anything. Identical prompts/namespaces/sample indices replay
identical cached teacher outputs with zero new HTTP attempts; new dense probes
can still require calls, subject to the unchanged caps and limiter.

Every completed probe also writes `episode/probe_episode.json`, a deterministic
projection of the executed trajectory (raw/parsed actions, visible observations,
rewards, turn outcomes and audit hashes). Offline seeded-cache tests compare these
files byte for byte. Full `episode.json`, events and provenance retain fresh run
IDs, timestamps and truthful cache-hit/zero-billing metadata, so those accounting
artifacts are intentionally different across separate executions.

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

The JSON and Markdown give |G|, taxonomy shares, budget FCE positions, rescue rates and
95% Wilson intervals, and the fixed 2×2 table. Raw rescue requires greedy success
**and ≥3/4 sampled successes**, for I0 as well as I1–I5. Each arm is shown beside
I0 on the same t* states with complete results for both. A **paired excess
rescue** counts a state only when the arm rescues and I0 does not. Wilson
intervals use this binary state count, not a subtraction of marginal rates.
Section 6 cells and the I4 execution/capacity label use paired excess rescues.
F-format and missing seed task types retain separate cohorts. I0 replay noise
is retained as audit metadata. Missing controls never count as control failures.
Cells use |G| as denominator; paired intervals disclose unknown pairs and their
matched task IDs. JSON retains all available raw rates under `arm_rescue`,
arm/I0/excess rates on matched states under `paired_arm_rescue`, and raw and
paired per-state outcomes under `arms_by_task` and `paired_excess_by_task`.
`eligible_t_star_episodes` names the arm denominator; `eligible_fce_episodes`
is retained as a compatibility alias for the same count.
`distance_diagnostics` includes every gap episode, even exclusions and pending
interventions: D(t) curve, t*, D(0), D(t*), remaining turns, budget FCE, and
`turns_spent_after_t_star_without_progress` (parent episode length−t*). The last
field is descriptive: no later **probed** state has a smaller successful teacher
distance; it does not assert that every unprobed turn made no progress.
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

## Stage 4e: frozen environment traits and stratified readout

`analysis/traits.py` implements [Addendum A](docs/MECHANISM_PLAN.md) for the
frozen dev140 population. It resets each game once, executes zero actions, and
reads the initial observation and audit facts. The audit-only output is
[`data/manifest/dev140_traits.json`](data/manifest/dev140_traits.json), with a
canonical content hash, code hashes, dependency versions, environment seed,
game/input hashes and the original split/list identity. Existing output is
refused, including an identical file. This file is not a training input.

Task type has six values and family has four. Hidden target requires **every**
goal-type object to be directly in a receptacle marked `openable` and not
`opened` in the initial facts. Room size counts distinct named receptacles in
the initial room inventory sentence, excluding the task instruction. Distractors
count all goal-type objects minus one required object, or minus two for
`pick_two_obj_and_place`. State change is yes/no, with heat/cool/clean retained
separately. Minimum length alone uses policy outcomes: the fewest logged steps
in a successful teacher dev trajectory, verified against its final environment
win audit; failed, incomplete or absent successes remain unknown. This is a
teacher trajectory length, not an environment-optimal path length.

Room size uses all 140 values for tertiles; minimum length uses known values
only. Cutoffs are the empirical quantiles at ceiling ranks `n/3` and `2n/3`:
low `<= q1`, middle `q1 < value <= q2`, high `> q2`. Equal values stay together,
so bucket sizes need not match; empty and unknown buckets are retained.
Coverage counts verified executed trajectories of the same task type for
all nine K=5/10/20, support-seed=0/1/2 collections. Nested subsets reference
their original evidence. Counts, unique instances and task groups are separate.

The initial build command (refuses to run again at the frozen destination):

```bash
CUDA_VISIBLE_DEVICES='' \
ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1 \
PYTHONPATH=src:. .venv/bin/python -m analysis.traits
```

`analysis/stratified.py` accepts repeatable `--row LABEL=EVAL_DIR` arguments for
teacher, base, students or baselines. Labels beginning `k{K}s{S}-...` select
their frozen coverage context. Each context contains matching method rows and
the base/teacher references on exactly that context's games. Labels without
K/support seed participate in the other traits; no coverage is inferred for
them. `--base-label` and `--teacher-label` can name alternative reference labels.

JSON and Markdown contain per-bucket game/group counts, success counts and
rates with Wilson 95% intervals, paired row-minus-reference differences against
base and teacher, exact two-sided McNemar p-values, and teacher-minus-row gaps.
Paired difference intervals subtract Bonferroni-Wilson 97.5% intervals for the
two discordant probabilities (nominal conservative 95%). Rates are proportions;
p-values are unadjusted. Statistics use game instances and also disclose group
counts. Missing/incomplete/unknown outcomes are counted separately and omitted
from known denominators; comparisons use their own known paired denominator.
The reader reuses `analysis.k_ladder` and records episode success as the outcome
source; sweep copies need not contain audit blobs. Duplicate tasks, sealed tasks,
game-hash/seed mismatches and unsupported coverage contexts are refused. Mixed
serving settings require `--allow-mixed-serving`; the historical non-invariant
12/140 serving-noise reference accompanies the paired tests.

The real-data sanity readout uses these explicit, read-only inputs:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m analysis.stratified \
  --traits data/manifest/dev140_traits.json \
  --row teacher=reports/runs/teacher-dev140 \
  --row base=reports/runs/eval/hpg-seedsft-sweep-bi/k5s0/google/gemma-4-12B-it/eval \
  --row k20s0-p40=reports/runs/eval/hpg-seedsft-sweep-bi/k20s0/k20s0-p40/eval \
  --row k10s0-t400k=reports/runs/eval/hpg-seedsft-sweep-bi/k10s0/k10s0-t400k/eval \
  --row k5s1-t400k=reports/runs/eval/hpg-seedsft-sweep-bi/k5s1/k5s1-t400k/eval \
  --row k5s0-t400k=reports/runs/eval/hpg-seedsft-sweep-bi/k5s0/k5s0-t400k/eval \
  --out reports/figs/stratified-seedsft
```

Readout destinations also refuse overwrites. Use a new destination for a later
readout of the same frozen traits. Neither command writes to `reports/runs/`
or `data/verified/`. Numerical artifacts and offline test results are recorded
in [Stage 4e validation](reports/stage4e_validation.md).
