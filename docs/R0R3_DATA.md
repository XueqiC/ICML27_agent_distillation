# R0–R3 data and teacher interfaces

Implementation follows `docs/direction/2026-10-12_real_model_R0-R3_prereg_ZH.md`.
Later user-requested clarifications fix R2's gradient labels to temperature-1
student samples and the strict certificate to the bounded `F_tilde_geo_prob`
metric with a-priori B=2; see `R0R3_TOOLS.md`. The preregistration is unchanged.
No model/environment run or teacher request was made during implementation.

## Interface for the concurrent training/statistics implementation

Import `load_surrogate_examples(path, tokenizer, manifest, freeze, max_length=32768,
packing="episode")` from `data.manifest.build_r03_sets`. `path` is
`data/manifest/r03_surrogate.json`. Returned rows have `input_ids`, `labels`,
`supervised_tokens`, `assistant_spans`, and the existing dataset metadata; pass a
row directly to `train.sft.turn_loss_sum`. The surrogate loader requires
`purpose="surrogate_only"` internally and reports `training_eligible=False`.

For P2, P3, P4 and R2 datasets import `load_r03_dataset(root, tokenizer, manifest,
freeze, max_length=32768, packing="episode", purpose="training")` from the same
module. Do not use `train.dataset.build_verified_dataset` on these roots: its
seed-selection validator cannot represent held-out subsets or unexecuted
correction labels. The wrapper uses the existing encoding functions and never
modifies `train/`. Correction rows always use turn packing, with only the final
teacher `Action:` line and native turn-end supervised; all earlier student
replies, observations, and prompts are masked. Corrections have **unknown**
action verification and zero new environment transitions.

`tools.r2_allocate.cluster_id(task_type, step)` defines cluster keys as
`TASK_TYPE::0-4`, `TASK_TYPE::5-9`, `TASK_TYPE::10-19`, `TASK_TYPE::20+`.
Steps are zero-based policy turns (including parser-feedback turns), the same
index used by rollout selection. `--statistics-only` writes `clusters.json`
with visits, normalized `d`, K20 coverage `k`, and up to ten deterministic
`gradient_sample_states` per cluster.

The gradient statistics JSON accepted by `--sigma` is:

```json
{
  "schema_version": 1,
  "checkpoint": "p10",
  "clusters": {
    "pick_and_place_simple::0-4": {"sigma2": 0.125, "sample_count": 10}
  }
}
```

Include every visited cluster. **sigma_s² = mean over sampled states, then over
sampled student labels, of ||grad[-log p(sampled action | state)]||² at the
restored student checkpoint.** Use the full supervised reply token sum and
the **unclipped** gradient in the trainable parameter space (production LoRA
coordinates). Temperature-1 labels from that same student give an unbiased
estimate of the Fisher information trace at each state. Teacher labels, greedy
labels, token-mean loss and clipped gradients do not have that interpretation.
`sigma2` is this nonnegative finite uncentered second moment. Additional
provenance fields (checkpoint hashes, parameter space, estimator and sample
hashes) are allowed and should be supplied by `tools/grad_stats.py`. Missing
values are errors, never silently imputed. Per-arm `allocation.json` entries
contain the predicted Adam error and sampled state references. `null` error
plus `predicted_error_infinite=true` represents a positive-weight cluster with
zero total coverage.

`tools/grad_stats.py` currently writes the real checkpoint path in `checkpoint`.
That output is accepted directly when `CHECKPOINT/training_state.json` is
readable and records `updates=115`, `supervised_tokens=104070`. The literal
`"p10"` form above is an explicit caller assertion for portable exports; keep the
original checkpoint hashes/provenance in the export. No path-name heuristic
silently certifies an arbitrary checkpoint as p10.

## Frozen sets and evidence

The builder fully validates both verified collections, including failed-attempt
receipts and real-environment win evidence, before selecting anything. K20 must
be the identical executed trajectories nested in K100, with support seed 0.

| Set | Trajectories/games | Unique instances | Distinct groups |
| --- | ---: | ---: | ---: |
| Surrogate (trusted trajectories, scoring only) | 80 | 80 | 80 |
| Rollout pool | 200 | 200 | 200 |
| dev300 | 300 | 300 | 297 |

| Task type | Surrogate | Pool | Added dev train games | Full dev300 |
| --- | ---: | ---: | ---: | ---: |
| look_at_obj_in_light | 14 | 34 | 27 | 40 |
| pick_and_place_simple | 14 | 33 | 27 | 62 |
| pick_clean_then_place_in_recep | 13 | 33 | 26 | 53 |
| pick_cool_then_place_in_recep | 13 | 34 | 27 | 52 |
| pick_heat_then_place_in_recep | 13 | 33 | 27 | 43 |
| pick_two_obj_and_place | 13 | 33 | 26 | 50 |

`K100 ∩ pool`, `K100 ∩ dev300`, and `pool ∩ dev300` contain **zero groups**.
The first 140 dev IDs retain the exact dev140 order; its 137 groups explain why
dev300 has 297 groups. The base manifest and split freeze are unchanged. Every
selected task passes `sed.manifest` authorization; sealed `valid_unseen` tasks
are never used. Frozen metadata marks games playable; no new environment was
started to build these lists.

`build_verified_dataset` cannot filter this collection. Therefore the builder
writes **filtered copies of the 80 schema-2 episode JSON files** beneath
`data/verified/r03_surrogate/episodes/`, with a new manifest and provenance.
Original receipts, observations, verification blobs, and failed-attempt evidence
remain in the unchanged source collection. The copied episode JSON may contain
audit references; only the existing encoder's explicit visible-field whitelist
enters model inputs. The loader verifies copied-file hashes before encoding.

Pool ordering is deterministic seed-0 hash selection, one game per group, with
round-robin task types. Frozen nested prefixes are R1 P2=40, R1 P3=80, R2=120.
Run the 120-game prefix once and reuse its smaller prefixes. No game is sampled
from dev300 for rollouts, labels, gradients or training.

## Rollout and dataset semantics

`run` uses the **same `sed.runner.run_episode` implementation invoked by
`harness/run_episode.py`**, and `students.vllm_student.Student`. It checks the
existing adapter/server equivalence report and the recorded p10 token count.
Episode-driving generation is greedy, thinking OFF, 1024 generated tokens per step, 40 policy
turns; parsing, admissible commands, parser feedback, context-overflow handling,
and termination are unchanged. Start the vLLM server separately with the
existing serving/equivalence workflow. No server is started by the new tool.

`run --label-samples k` (default **0**) adds k independent calls to the same
served student at each visited decision state, with **temperature 1**, thinking
OFF, and the identical prompt and caps. Sampling explicitly disables top-k,
top-p/min-p truncation and repetition/presence/frequency penalties so server
generation defaults do not alter the student distribution. These labels never
enter the environment or future episode history. The step's `label_samples`
list stores sample index, status, visible reply, parsed action, raw response
content, and call metadata (returned model/revision, timestamps, sampling
settings, finish reason, usage counts and raw-response references). Runner
policy metadata and the existing `student_calls` ledger retain their evidence.
Label request failures are recorded with sanitized errors and do not change
the greedy rollout. There are no automatic retries. These are extra **student**
calls, with zero teacher calls, environment transitions or training exposures;
account their usage from `label_samples[*].metadata`/the call ledger separately
from the runner's greedy-only episode usage metrics.

`rollouts.jsonl` contains one JSON object per episode; `episode_jsonl/*.jsonl`
contains the same per-episode records separately. Each `states` element contains
zero-based step, complete visible history, current observation/commands, exact
ReAct messages, student's reply/action, reward, success, next visible observation,
and termination fields. Environment verification is kept outside prompts. Each
row references the runner's original episode and provenance hashes. Readers
reconstruct the record from those artifacts and reject mismatches. Failures and
unknown outcomes remain in runner artifacts and `failures.jsonl`; they are not
silently relabeled as successful data.
Policy, environment-step and context-overflow failures also retain their visible
prompt/action in the JSONL `failed_attempts` list, with unknown rewards represented
by null. These infrastructure/terminal attempts are excluded from query sampling.

P3 retains only episodes whose final verification blob says an executable,
real-environment win. P4 makes one request per trajectory, saves raw structured
output, requires an identical action string in every turn, and preserves the
original `Action:` line bytes. Changed actions, turn counts, malformed JSON or
injected action/control lines cause a logged rejection without retry. Rewritten
thought truth is unknown; the executed action evidence belongs to its parent.
The rewrite output cap defaults to 8192 tokens for a whole trajectory and is
recorded separately from the correction teacher's configured per-state cap.
K20 yields 20 first-pass rewrite requests; the hard ceiling remains 40.

Correction examples contain the complete actual student prefix and one new
teacher action. The teacher uses exactly its existing collection ReAct prompt.
The visible teacher thought/raw reply is retained in the audit log, while the
training target is just `Action: COMMAND` plus the native turn-end. The final
label has no executed transition; its reward/success are unknown. The custom
loader supervises only this turn, regardless of the requested packing option.
Previous student turns are context, including empty replies, with zero loss.
P2/P4 are explicitly new labels/text, **not** new trusted executed seed counts.
No collector restores, branches, or pretends to execute a teacher correction.

`manifest.json` in each candidate root uses `format=r03_episode_dataset_v1` and
schema-2 episode files. It is a wrapper format, not a forged K-shot verified
collection. Provenance and collection logs record source IDs, parent rollouts or
seeds, raw outputs, rejections, token usage, and zero initial training exposure.
The training job must separately record all later exposures.

## Attempt caps, cache and dry runs

`teachers.capped.CappedTeacher` is a library, used by both collector CLIs. It
wraps `Teacher` with one attempt, SDK retries disabled, the existing cache and
usage ledger. Shared state hashes depend only on task identity and allowed
visible context/history. The existing cache additionally binds teacher alias,
prompt/version, and sampling settings, so R1 and R2 share identical corrections
without colliding with rewrite prompts.

Each actual transport invocation reserves one durable counter slot before
calling the transport. Failures and unknown billing consume slots conservatively.
The counter is reconciled with unique started requests in the ledger; deleting
it cannot recover spent budget. A lock serializes a collector run, including
concurrent workers; the existing cache lock deduplicates across runs. Use one
stable counter file, ledger and run ID for each collector. An interrupted request
or HTTP status other than **200** halts that run; no automatic retry or empty
training label is returned. Errors remain sanitized. Raw responses and known
input/output/reasoning-token usage are recorded by the original teacher.

Each generation re-reads the settled ledger before the next request. Cache hits
are allowed after the cap is exhausted and consume no attempt slot. Dry-run
mode never initializes a client, reads credentials, reserves counters, changes
the usage ledger, or writes training examples. It reports unique uncached calls,
remaining cap, and whether the run is halted/over budget in `dry_run.json`.
Dry-run is a point-in-time estimate; a concurrent writer can populate the cache.

R1 P2 uses at most 600 selected states; P4 at most 40 attempts; R2 at most 1500.
If 40 games contain over 600 unique states, `r1_selection.json` freezes the first
600 in pool-game/step order and reports the omitted count. This resolves the
preregistration's “every visited state” language against its hard cap without
silently exceeding the budget. Unparseable/inadmissible teacher actions are
logged and dropped, so accepted training-label counts can be below query counts.
A partially failed collector keeps its evidence; CLI outputs are immutable and
are not silently resumed into a different dataset.

## R2 allocation and interpretation

`d(s)` counts all p10 student policy-turn visits, normalized across the collected
R2 games. `k_s` counts K20 policy turns in the same buckets. A1 uniformly samples
unique visited states without replacement. A2 uses `W=d² sigma2`; A3 uses
`W=d sigma2`. The continuous solver applies
`n_s=max(sqrt(W_s/lambda)-k_s,0)` and largest-remainder integer rounding with a
stable cluster-name tie break.

Finite pools add an upper bound of available unique states per cluster. The
output discloses saturated clusters; a total budget exceeding all available
states is an error. Zero-weight ties use remaining-capacity proportions. The
reported error always uses the **Adam** weights, including for A0/A1/A2.

All arms/budgets share uniform hash priorities. A1 takes a global prefix; every
arm takes per-cluster prefixes of the same order. Thus any two arms share exactly
`min(n_s_arm1,n_s_arm2)` states within cluster s, the maximum possible overlap
for their allocations. No preference for already cached R1 states is introduced
into uniform R2 sampling. Output includes each arm's references, the R2 union,
the R1+R2 union, overlap and additional uncached-after-R1 upper bound. Cache
contents and rejected labels can change the actual billing/accepted totals.

`--statistics-only` outputs up to ten probe-state references per cluster before
allocation. `grad_stats --rollouts FILE... --states-per-cluster 10 --seed 0`
uses the identical deterministic selection, independent of sampled labels and
their gradients. It validates frozen R2-pool/evidence and the serving checkpoint
against the restored weights, then builds one diagnostic example per selected
state and sampled label. Only the sampled raw reply is supervised; visible
history and prompt are masked, and audit fields never enter model input. Raw
whitespace and native stop tokens are preserved; a cap finish does not gain a
synthetic stop token. An omitted stop token is assumed to be Gemma's `<turn|>`;
that assumption must match serving for the Fisher interpretation. Parsed
invalid/inadmissible samples remain in the estimate. Missing/failed labels or
unexpected thinking abort statistics rather than induce survivor filtering.
The state average is uniform even if label counts differ.

`grad_stats.py` now exposes **unclipped `sigma2`**, the identical
`sigma2_unclipped`, and `sigma2_clipped_DIAGNOSTIC`. `sample_count` counts
state/label examples; `state_count` counts sampled states. Old clipped,
token-mean sigma files must be recomputed. No teacher probes are needed.

## Exact CLI workflow

Run from the repository root with the existing environment (no installers or
downloaders). These are future experiment commands; only the builder and CPU
tests were executed for this implementation.

```bash
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=''
.venv/bin/python data/manifest/build_r03_sets.py

# Existing, separately validated p10 vLLM server; set these to its artifacts.
ENDPOINT=http://127.0.0.1:8000/v1
SERVER_PROVENANCE=/path/to/server/provenance.json
LORA_CHECK=/path/to/p10/check/report.json

.venv/bin/python tools/rollout_pool.py run \
  --subset R2 --endpoint "$ENDPOINT" --model-name p10 \
  --server-provenance "$SERVER_PROVENANCE" --lora-check "$LORA_CHECK" \
  --label-samples 2 --output data/raw/r03_p10

.venv/bin/python tools/rollout_pool.py filter \
  --rollouts data/raw/r03_p10/rollouts.jsonl --output data/verified/r03_p3

.venv/bin/python tools/r2_allocate.py --statistics-only \
  --rollouts data/raw/r03_p10/rollouts.jsonl --output reports/r03_probe_states

.venv/bin/python tools/collect_corrections.py \
  --source R1_P2 --run-id r03-r1-p2 --cap 600 \
  --rollouts data/raw/r03_p10/rollouts.jsonl \
  --selection reports/r03_probe_states/r1_selection.json \
  --output data/verified/r03_p2 --dry-run
# Remove --dry-run to collect after reviewing the estimate.

.venv/bin/python tools/collect_rewrites.py \
  --run-id r03-r1-p4 --cap 40 --output data/verified/r03_p4 --dry-run
# Remove --dry-run to collect; one call per K20 s0 trajectory.

# Run scoring in the training environment with local p10 weights.
# Choose --device explicitly; this CPU example keeps CUDA hidden.
.venv/bin/python tools/grad_stats.py \
  --snapshot /local/gemma-snapshot --checkpoint /runs/cold/checkpoint-tokens-104070 \
  --verified data/verified/seeds_k20_s0 \
  --rollouts data/raw/r03_p10/rollouts.jsonl --states-per-cluster 10 --seed 0 \
  --device cpu --output reports/r03_sigma.json

.venv/bin/python tools/r2_allocate.py \
  --rollouts data/raw/r03_p10/rollouts.jsonl --sigma reports/r03_sigma.json \
  --budgets 100 400 --seed 0 --output reports/r03_allocations

.venv/bin/python tools/collect_corrections.py \
  --source R2 --run-id r03-r2-union --cap 1500 \
  --rollouts data/raw/r03_p10/rollouts.jsonl \
  --selection reports/r03_allocations/allocation.json \
  --output data/verified/r03_r2_union --dry-run
# Remove --dry-run to collect the R2 union; existing R1 corrections hit cache.

# After union collection, materialize per-arm datasets using cache only.
for arm in A1_n100 A2_n100 A3_n100 A1_n400 A2_n400 A3_n400; do
  .venv/bin/python tools/collect_corrections.py \
    --source R2 --run-id "r03-r2-$arm" --cap 0 \
    --rollouts data/raw/r03_p10/rollouts.jsonl \
    --selection "reports/r03_allocations/$arm.json" \
    --output "data/verified/r03_$arm"
done

# Direct existing evaluation harness accepts the generated dev300 format.
.venv/bin/python eval/run_suite.py --config configs/base.yaml \
  --policy student --decoding greedy --endpoint "$ENDPOINT" --model-name p10 \
  --enable-thinking off --max-tokens 1024 --workers 4 \
  --tasks data/manifest/r03_dev300.json --server-provenance "$SERVER_PROVENANCE" \
  --lora-check "$LORA_CHECK" --output reports/r03_dev300_p10
```

Set `ALFWORLD_DATA_ROOT` to the installed data before real rollouts. The rollout
CLI optionally accepts `--games FILE.json` (a JSON list strictly within its
chosen pool subset). Both collectors default to the **same**
`data/raw/r03_teacher_cache` and `data/raw/r03_teacher_usage.jsonl`; preserve those
paths/settings across R1 and R2. `teachers/capped.py` has no separate executable
CLI; collector `--cap`, `--run-id`, `--cache`, `--usage-ledger`, and `--dry-run`
configure it.

The existing `scripts/eval_adapters.sh` refuses any list name besides dev40/dev140.
It was not edited (outside ownership). `eval/run_suite.py` accepts dev300 directly,
verified by an offline test; the command above bypasses that shell-only name
restriction while retaining harness checks. Do not rename dev300 to dev140:
that name triggers the exact original dev140-content validator.

## CPU validation and synthetic query counts

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/test_r03_sets.py tests/test_rollout_pool.py tests/test_capped_teacher.py \
  tests/test_collect_corrections.py tests/test_collect_rewrites.py tests/test_r2_allocate.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.:tests .venv/bin/python - <<'PY'
from test_r2_allocate import synthetic
from tools.r2_allocate import allocate
rows, demos, sigma, r1 = synthetic()
r = allocate(rows, demos, sigma, r1_states=r1['states'])
for k in ('r1_query_count', 'union_query_count', 'r1_r2_overlap',
          'r1_r2_union_query_count', 'r2_additional_after_r1'):
    print(k, r[k])
PY
```

The deterministic synthetic rollout has 120 games, 2880 states, 24 clusters,
and two budgets (100, 400). R1 selects 600 states and discloses 360 omissions;
R2's six arms need **593** unique states. Their overlap with R1 is **132**,
so R1+R2 requires **1061** distinct correction queries, with **461** additional
queries after R1. Twenty distinct first-pass K20 rewrites would make 1081 total
requests in this synthetic accounting, assuming successful responses and no
pre-existing cache or extra gradient-probe teacher calls. These are software
validation counts, not real-model experimental results.

Original data-tool validation: **35 new CPU tests passed**; all five executable CLI help paths
passed. The actual generated dev300 passed `eval.run_suite.frozen_tasks`, and all
80 copied surrogate episode hashes matched. This `.venv` has neither torch nor
transformers, so encoding/mask tests use a fake tokenizer and do not execute a
real `turn_loss_sum` forward pass. No installation, network request, GPU work,
real teacher request, commit, push, or hpg command was performed.

Follow-up validation: **82 CPU tests passed** across the four updated tools,
warmth/amplitude diagnostics and correction/rewrite collectors. Mock HTTP calls
verify label sampling, failures, accounting and unchanged greedy episodes;
a tiny model verifies the direct rollout-to-sigma CLI, full-reply gradient norms,
state/label averaging and the exact categorical Fisher trace. The same
`.venv/bin/python` used the existing read-only cached training dependencies via
`/tmp/r0r3_cpu_tests.py` (details in `R0R3_TOOLS.md`). No installs, network, GPU,
real teacher calls or commits were needed.
