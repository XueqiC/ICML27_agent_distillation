# Stage 2c validation — 2026-10-04

Small engineering stage only. Read `AGENTS.md`, `PROJECT_STATE.md`, the Stage 2b
validation report and the governing research handoff before implementation.
No GPU, network, live OpenAI calls, package installation, commits or pushes.
No seed collection or teacher evaluation was executed. No scientific result or
faithful baseline reproduction is claimed.

## Teacher retries and run-wide admission control

`teachers/openai_teacher.py` now defaults to **10 total HTTP attempts** per call.
429/`RateLimitError`, transient 5xx, timeouts and connection failures retry;
existing 408/409 retry behavior is retained. Auth, 400 and other nonretryable
errors stop after one attempt. SDK retries remain disabled.

Retry number i (after attempt i) draws `uniform(0, min(2 * 2**(i-1), 60))` seconds.
The actual retry wait is the maximum of that draw and recognized provider timing
headers. `Retry-After` accepts seconds or an HTTP date; `x-ratelimit-reset-*`
accepts duration strings such as `1m2.5s`/`250ms`, seconds or epoch seconds.
Malformed/nonfinite timing values are ignored. Provider lower bounds may exceed
the 60-second exponential cap. No wait follows a final/nonretryable attempt.
Provider cooldowns are also written to the shared limiter, including after the
final failed attempt, so other workers honor them.

`teachers/rate_limit.py` implements a POSIX file-lock protected **rolling 60-second
window**, keyed by the configured provider model under
`<run>/data/teacher_rate_limit/<model-hash>.json`. All suite workers use the same
absolute path, recorded in effective suite provenance. All collection attempts
share their collection root's limiter, outside per-attempt evidence directories.
Standalone episode runs also use their run directory. Direct Teacher callers
default to a limiter next to their usage ledger and can pass a shared directory.
Locks are released before sleeping; budgets are checked again after waking.

Configuration in `configs/base.yaml`:

```yaml
models:
  teacher:
    max_attempts: 10
    timeout_seconds: 60
    rate_limit:
      requests_per_minute: 300
      tokens_per_minute: 400000
```

Every actual HTTP attempt, including retries, reserves one request and
`len(canonical_prompt_UTF8_bytes) + 256 * (message_count + 1) + max_output_tokens`.
This deliberately conservative text-token estimate requires no model access or
network tokenizer. Failed/unknown requests are never refunded. Reservations over
the entire TPM allowance fail before a request; inconsistent limits in the same
shared state fail closed. Cache hits consume no quota. Provider accounting and
other concurrent runs/account users can still cause 429s: this limiter bounds
this run's reservations, not a provider's undocumented accounting algorithm or
aggregate account traffic. Independent runs use independent state.

Each attempted request retains a durable start and terminal ledger row with
status and sanitized error type; terminal failures add `retryable`, `will_retry`,
`wait_seconds` and `provider_wait_seconds`. Successful/started rows have zero
retry wait. `limiter_wait_seconds`, `reserved_tokens` and reservation timestamp
record admission separately. Failed billed usage remains unknown. Raw exception
bodies and headers are not persisted. Prompt whitelist, sampled-output-only
requests, secret redaction, rejected-output retention and cache hashes remain.

## Frozen full development list

`data/manifest/build_dev140.py` builds `dev140.json` solely from frozen metadata:
all eligible source `valid_seen` instances assigned to logical `dev`, excluding
sealed assignments. Exactly 140 distinct IDs are required. Ordering uses the
existing seed-0 SHA-256 order key; schema, manifest/freeze hashes, list hash and
group/type counts follow dev40. Suite loading recomputes and checks the full list.

- Instances: **140**; task groups: **137** (do not treat all games as independent groups).
- Type counts: look 13, simple 35, clean 27, cool 25, heat 16, two-object 24.
- List SHA-256: `8fab06e9c27a5cca09ca3a5729b1c9762029bff71304aa297bc87a04dbc3dade`.
- Original manifest SHA-256: `71339051bb6d53e343ae4a27dd97bcc08680b461d75753a665dfd5a840e11a34`.
- Split freeze SHA-256: `1e704d8993b9643fa32e45404ffee0d642bc8f8194f7977c416a9a6ec015c12a`.

The original manifest, split freeze and dev40 were not changed. The sealed split
was not modified and no sealed game files or environments were opened.

## Nested seed selection and collection

`stratified_groups_v1` is unchanged. The type order, per-type group shuffle and
chosen trial are independent of K. Each type's quota is nondecreasing in K;
therefore K=5 groups are a strict subset of K=10, which is a strict subset of
K=20, at each fixed seed. Tests compare both selected prefixes and the entire
selected-plus-replacement order for seeds 0, 1 and 2. All nine frozen selection
files are present. The existing `seeds_k10_s0.json` is byte-for-byte unchanged.
These are **selections, with zero verified trajectories**, not collected seeds.

`teachers/collect_seeds.py --selection ...k20... --subset-k 5 10` runs the existing
resumable real-win collector once at K=20. Only after completion does it create
immutable subset views at `<collection>/subsets/k5` and `subsets/k10`. They select
the source's verified receipt for each smaller selection's original slot. If a
slot needed replacements, all its failed attempts remain referenced and its
accepted replacement is reused identically. Independently collecting at a smaller
K could choose a different replacement because the replacement boundary depends
on K; these artifacts explicitly disclose largest-K collection reuse.

Subset manifests embed the exact original receipt values and reference the same
on-disk receipt files with content hashes. Episodes are not copied or regenerated.
They record reused trajectory count, zero new teacher calls/transitions, source
collection and selection hashes, and overlapping-cost disclosure. Source K=20
keeps every attempt, negative result, unknown usage and full provenance. Smaller
K describes the trusted trajectories exposed to training, not all experience
acquired while producing the source. Do not sum subset costs or count reuse as
new trajectories/tasks. Sampling seeds are separate collections, not guaranteed
disjoint task sets.

The shared evidence loader verifies complete K, source/selection identity, exact
receipt files, all retained attempt artifacts, real wins, frozen TRAIN membership,
distinct groups/slots and type quotas. Training still constructs inputs through
its explicit visible-field whitelist. Subsets must remain alongside their parent
collection. Interrupted/exhausted sources cannot publish complete smaller-K views.

## Exact commands

Repeatable offline builders, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python data/manifest/build_dev140.py
for seed in 0 1 2; do
  for k in 5 10 20; do
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python \
      data/seeds/select_seeds.py --k "$k" --seed "$seed"
  done
done
```

The following commands are for **later live execution on rai**, and were not run
in this stage. The configured teacher remains `gpt-6-luna`, output cap 2,048;
the ReAct protocol, thinking OFF and student cap 1,024 are unchanged.

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

The suite requires a fresh output directory. The seed loop is sequential, stops
on incomplete collection, and can be rerun to resume with identical code/config/
dependency identity. Three episode attempts per candidate are separate from ten
HTTP attempts per policy call. Train on K=5 with
`--verified data/verified/seeds_k20_s0/subsets/k5`, K=10 with `subsets/k10`, and
K=20 with `data/verified/seeds_k20_s0`; substitute the sampling seed as needed.

## Offline validation

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1 \
.venv/bin/python -m pytest -q
```

Results: **137 passed, 2 skipped** without the data override (20.74 seconds), and
**138 passed, 1 skipped** with installed ALFWorld data (30.35 seconds), including
real TRAIN reset, three steps and checked prefix replay. The project venv has no
torch, so its existing torch-only module is skipped in both passes; the other
skip is the optional real-environment test without `ALFWORLD_DATA_ROOT`.
The collection, full-dev and suite CLI help checks and `git diff --check` pass.
The original manifest, split freeze, dev40 and K=10/seed=0 selection also pass an
explicit unchanged-file diff check.

The tests block network/OpenAI transport and hide CUDA. New tests use fake clients and
clocks for retry bounds, 429 headers/missing headers, transient versus fatal
errors, shared cooldowns, cache/secret behavior, and a real four-process limiter
with shared virtual time testing both RPM and TPM. A spawned teacher suite checks
the shared run path and config overrides. Manifest tests check full-dev identity,
determinism and tamper refusal; nested collection tests include failed attempts,
replacement reuse, resume, exact receipts, loader integration and altered evidence.

Live 8-worker provider behavior remains unmeasured. A successful offline test is
not evidence that the previous rate-limited dev40 run is scientifically valid.
Its 27 aborted episodes remain recorded as failures/unknowns; dev140 requires a
new evaluation run and no results are claimed here.
