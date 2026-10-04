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
experiments. `train/dataset.py` now tokenizes complete ReAct turns using the
student's local chat template, masking the entire evaluation prompt and
supervising the visible assistant reply plus native `<turn|>` token 106.
The old adapter and support-rollout code were read for interface reference;
no old implementation or experimental numbers were copied wholesale.

See [reports/stage1_validation.md](reports/stage1_validation.md) for measured
local counts, exact runtime, test results, episode artifacts, and remaining
limits.


## Shared evaluation on HiPerGator

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

Training examples each supervise one complete assistant turn with its exact
evaluation prefix; earlier replies remain masked context and are supervised
once in their own examples. No observations or empty thought-prefill tokens
enter the loss. Overlong contexts fail explicitly. Training shuffles complete
passes deterministically, accumulates a supervised-token mean loss, and saves
checkpoints at the first complete turn reaching each requested endpoint.
Requested and actual token counts, overshoot, passes, updates, timing and peak
CUDA memory are recorded. A turn exceeding the update budget runs alone.

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

`models.teacher.rate_limit` in `configs/base.yaml` defaults to 300 requests/minute
and 400,000 tokens/minute **across all workers of one run**. A file-locked rolling
minute ledger, keyed by provider model under `<run>/data/teacher_rate_limit`,
reserves UTF-8 prompt bytes plus framing allowance and the output cap. This is
conservative admission accounting, separate from returned billed usage; failed
requests keep their reservations. Provider cooldowns are shared across workers.
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
