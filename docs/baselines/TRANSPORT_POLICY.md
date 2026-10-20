# Baseline teacher transport policy (2026-10-19)

Engineering fix only. `teachers/baseline_transport.py` is the single explicit,
opt-in policy for teacher calls under `baselines/`. Seed collection does not opt
in: `teachers/collect_seeds.py`, `teachers/full_train.py`, and `configs/base.yaml`
retain their existing 60-second timeout and settings. Existing verified data,
its identities and loader are unchanged. New baseline identities bind this
policy; do not rewrite an old identity to resume it with new settings.

All paths below use `gpt-6-luna`, sampled text only. The reference files were
inspected locally; no upstream runtime or model was imported. No explicit
client timeout/retry override was found on the corresponding teacher paths.

| method | upstream file:line or "SDK default" | timeout | http attempts | method-level tries |
|---|---|---|---|---|
| LEAP extraction | `third_party/leap_llm/src/leap_llm/utils/openai.py:53-58` | 600 s | 3 | 3 per request (retained local contract) |
| LEAP correction, both variants | same client; `third_party/leap_llm/scripts/dataproc/correct_student_trajectory.py:102-119` | 600 s | 3 per try | 3 per request |
| STeCa reflection | `third_party/STeCa/data_collection/data_cali_alfworld.py:345-349` (no override; SDK default) | 600 s | 3 | 1 per reflection |
| GAD teacher | `third_party/LMOps/gad/deepscaler/utils.py:45-59` | 600 s | 3 | 1 per history |
| CoA, all five roles, TRAIN and DEV teacher | SDK default; `docs/baselines/NEW_METHOD_SPECS_1019.md`, Chain-of-Agents | 600 s | 3 | 1 per role invocation; 3 TRAIN system runs/task |
| AgentDistill manager/create/abstract/cluster/consolidate | SDK default; `docs/baselines/NEW_METHOD_SPECS_1019.md`, AgentDistill | 600 s | 3 | 1 per logical role request; 3 collection runs/task |
| HAD, with/without harness draws | SDK default; `docs/baselines/NEW_METHOD_SPECS_1019.md`, HAD | 600 s | 3 | 1 per draw |
| AD-FTP CoT and agent collection | `third_party/agent-distillation/src/smolagents/models.py:1079-1100` (`client_kwargs` defaults to empty; no caller override found) | 600 s | 3 | 1 per request; 3 environment attempts/task |

The 600-second SDK default has read/write/pool timeout 600 s and connection
timeout 5 s (`httpx.Timeout(600, connect=5)`). HTTP attempts here exclude the
separate shared-key rate-limit allowance described below. They are transport
attempts of an identical request, not additional labels, tasks, environment
attempts, or independent teacher draws.

Reference scope:

- LEAP extraction's upstream `extract_privileged_state_from_logs.py:73` calls
  once. The existing local three-try extraction contract is retained as
  explicitly required; it is not attributed to that upstream loop.
- STeCa uses legacy `openai.ChatCompletion.create` without settings; the
  requested modern SDK defaults apply to our Responses port.
- GAD's outer loop (`utils.py:49-68`) separately retries 429s with backoff. The
  shared-key retry429 policy below takes precedence over its very large bound.
- CoA's OAgents synthesis is not released. Its web-agent inference client
  (`AFM/evaluation/web_agent/inference_web_agent.py:76`) uses SDK defaults.
  `AFM/evaluation/web_agent/utils.py:101` sets 180 s for `retry_predict`, called
  at `inference_web_agent.py:348` for the external GRM answer judge. That judge,
  crawl summarizers and local model servers are not our five role teachers.
- `third_party/AgentDistillation` has no released MCP teacher/client path;
  HAD has no released implementation. Their local paper specs do not specify
  timeout or retries, so SDK defaults apply.
- ETO, IPR, AgentGym, Kang, SAD and WKM make no new teacher requests here.
  Their references to `TeacherResult` are return types, not teacher transport.
  GAD/HAD's loader-only `DrawTeacher` instances only reconstruct cache keys.

## Retry and failure semantics

The OpenAI 2.8.1 SDK implementation and constants were inspected locally.
The audited `Teacher` loop reproduces SDK `max_retries=2`: up to three HTTP
attempts on 408/409/5xx, timeouts and connection failures, honoring
`x-should-retry` for non-429 HTTP failures. Backoff starts at 0.5 s, doubles to
8 s, and applies the SDK's downward jitter factor `1 - 0.25 * random()`.
`retry-after-ms` or numeric/date `retry-after` in (0, 60] takes precedence.
The installed SDK's own retries remain disabled (`max_retries=0`) so every HTTP
attempt has its own durable start/end, sanitized failure, limiter acquisition,
and token reservation. There are no unlogged nested SDK requests.

Rate-limit 429s use the existing capped retry429 behavior: up to 20 rate-limit
attempts per logical request, numeric/date `retry-after` plus provider reset
headers, otherwise full jitter from zero to `min(5 * 2**n, 300)` seconds.
They do not decrement the ordinary three-HTTP-attempt allowance, consume another
capped logical slot, or advance a method-level try. Quota/billing 429s halt
immediately. Exhausting the 429 allowance halts the stage, including LEAP,
without proceeding to another method try. Interrupted/unknown attempts remain
retained and fail closed; no counter or token reservation is refunded.

LEAP retains three method tries. A completed parse/validation rejection consumes
one try; ordinary transport exhaustion inside a try consumes one try and moves
to the next, just as the upstream correction loop does. An explicit unsupported
sampling response also consumes a try and records the existing sampling fallback
before the next try. A later accepted response may complete the request, with
all failed evidence retained. If no response is accepted and any try has
transport-failed/unknown evidence, the stage fails loudly and publishes no
rejection outcome. Three completed malformed outputs remain format rejections.
All-transport exhaustion makes nine ordinary HTTP attempts and fails through
`teachers/stage.py`. Transport errors never become negative training labels.

## Provenance and accounting

New run identities and disclosures include `baseline_transport`, with the source,
timeout (including connection timeout), ordinary HTTP attempts, method tries,
429 bound/policy, backoff and retry owner. Capped counter identities and each
HTTP ledger row/returned metadata also bind the effective policy. AD-FTP binds
it in acquisition identity/provenance and its resulting collection manifest.
LEAP's reported caps distinguish method attempts from HTTP attempts, including
an upper bound for the independent 429 allowance. Existing capped counters keep
counting logical slots; detailed ledgers retain every actual HTTP attempt and
unknown usage, including failures recovered within a slot.

Offline regression tests cover actual client constructor timeouts, retry counts,
Retry-After and SDK retry hints, mixed 429/timeout sequences, persistent capped
recovery, LEAP try consumption, quota/429 halts, fail-loud stage reports, and
verified-collection loading with both original seed acquisition paths at 60 s.
No model, sampling, algorithm, prompt, training loss or environment protocol is
changed by this transport policy.
