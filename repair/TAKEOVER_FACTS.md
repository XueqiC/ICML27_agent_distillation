# Teacher takeover and public fact table (E0.1 / E0.2)

These tools implement the engineering seams for method-refinement plan v1,
sections 1–2. They produce TRAIN-side diagnostic evidence, never training data,
updates or checkpoint selection. No scientific feasibility claim follows from
unit tests or synthetic rollouts. No existing module, schema or template changes.

Run from the worktree root with `PYTHONPATH=src:.` and the environment Python.

## Prefix specification

A new JSON format: a nonempty list of entries with exactly these fields:

```json
[
  {
    "case_id": "cg-apple",
    "group": "pick_cool_then_place_in_recep-Apple-None-Microwave-19",
    "task_id": "train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_0",
    "source": {
      "failures_json": "/path/to/coverage/failures.json",
      "coverage_root": "/path/to/coverage",
      "episode_dir": "attempt-01-greedy/episodes/episode-directory"
    },
    "branch_index": 10
  }
]
```

`case_id` is a unique label; the generated default is the task ID. Task IDs and
source directories in the example are illustrative: use the generator for actual
identities. `branch_index` counts already executed policy turns, so 10 replays
steps 0–9, leaving 30 of H=40. It must leave room for continuation. Paths in
`failures_json` and `coverage_root` may be absolute or relative to the spec file;
`episode_dir` is relative to the coverage root and must remain inside it.

```bash
PYTHONPATH=src:. python -m tools.teacher_takeover spec-from-cg \
  --failures /path/to/coverage/failures.json \
  --coverage-root /path/to/coverage \
  --groups 'look_at_obj_in_light-Newspaper-None-DeskLamp-224,pick_and_place_simple-Kettle-None-Shelf-1,pick_clean_then_place_in_recep-Kettle-None-DiningTable-7,pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18,pick_cool_then_place_in_recep-Potato-None-SideTable-21,pick_cool_then_place_in_recep-Apple-None-Microwave-19,pick_cool_then_place_in_recep-Egg-None-Microwave-5' \
  --branch-index 10 --config configs/base_e4b.yaml --output /path/to/spec.json
```

Both generation and execution import `tools.cg_probe.load_cases` unchanged:
failure-list and actionable-summary hashes, episode/events/provenance hashes,
frozen training authorization, attempt-01 greedy provenance, seed, H=40 and
nonterminal ten-turn CG search-prefix checks all apply. The spec format can hold
other branch indices, but this first source adapter still requires CG-compatible
coverage evidence (including the loader's original first-ten-turn checks). A new
U_dev source adapter is outside E0.1. Group names are used for metric targets,
never for actor prompts. No game, PDDL, task-path contents, teacher demonstrations,
or future rollout discoveries enter prompts or facts. Real environments open
only their normal task inputs for actual reset/step execution.

## Execute and summarize

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m tools.teacher_takeover run \
  --spec /path/to/spec.json --config configs/base_e4b.yaml \
  --data-root /path/to/alfworld --runs 4 --workers 4 \
  --max-teacher-input-tokens 5000000 --max-teacher-output-tokens 500000 \
  --output /path/to/takeover

PYTHONPATH=src:. python -m tools.teacher_takeover summarize \
  --rollouts /path/to/takeover --output /path/to/takeover-summary
```

Every run resets and replays the original raw student replies through
`execute_actions(..., continuation=..., prefix_reference=...,
interaction_kind='evaluation_steps')`. The original horizon is retained. As in
CG, the imported reference is rebound to the current environment version and
fingerprint; original source provenance remains recorded. Initial observation,
prefix transitions, audit digests, rewards, termination, game identity and seed
must match before calling the actor. A mismatch is void, with a reason.

`--from-start` sets the effective branch to zero and runs the same checked-reset
seam with H=40. Use a separate output directory. It is a task-start diagnostic,
not evidence that the failed prefix was feasible.

The actor is the exact `teachers.openai_teacher.Teacher` class with collection
`models.teacher.sampling`, `max_attempts`, `timeout_seconds`, `secrets_file`,
`rate_limit_options`, and the unmodified ReAct template/version. No teacher or
student substitution is allowed. `build_messages` receives the public history;
replayed assistant messages are the student's visible replies. The CG loader's
extra structured-guidance `instruction` field is removed, so no duplicate task
instruction is prepended to baseline ReAct messages. The goal is already in the
initial observation. Hidden thinking stripping follows the shared visibility
helper. Facts are **not** injected into takeover prompts.

Each `(case_id, mode, run)` has a separate teacher cache, and interrupted attempts
have separate cache directories as well. Runs are independent API samples;
there is no fabricated sampling seed or reuse of another run's response. All
process workers use `output/data/teacher_rate_limit` (or the collection transport's
`TEACHER_RATE_LIMIT_DIR` override). Each process owns its environment. A narrow
client guard prevents quota 429 retries, which the default collection transport
would otherwise retry. Ordinary rate-limit 429s retain the existing transport's
backoff and configured maximum attempts. After transport retries end, the policy
classifies the sanitized usage receipt or exception as `TRANSIENT` (HTTP
5xx including 500/502/503/504, 408, 409, exhausted rate-limit 429, connection or timeout error),
`QUOTA` (429 rejected by the existing `_is_rate_limit_429` classifier), or `OTHER`.
Quota, other HTTP failures (including schema/authentication errors), unknown
usage, truncation, empty responses and unexpected cache hits halt immediately.

Transient failures get at most six additional attempts at the same logical
request with identical messages and fresh request/reservation IDs. The base
waits are 5, 10, 20, 40, 80 and 160 seconds, each with uniform 0–10% jitter;
their total is at most 346.5 seconds (under six minutes). This bounds the added
backoff, excluding the unchanged transport/rate-limiter waits and request time.
Budget checks precede retries and run every second during backoff, so a halt or
global stop from another worker ends the loop. Only exhausted transient retries
halt the invocation with nonzero exit status. No failed/empty response becomes an environment action.
Nonempty completed parser errors use the unchanged shared parser and turn cap.

### Global caps, interruption and resume

`GlobalTeacherBudget` accounting is reused behind a process lock and a durable
append-only checkpoint ledger. Before **each HTTP attempt**, including retries,
the adapter reserves a conservative UTF-8 byte input bound plus chat framing and
the configured `max_output_tokens`. The rate limiter's character/4 admission
estimate is not used as this global bound. Known actual input/output usage settles
the corresponding reservation; reasoning counts are metadata only. Unknown failed
or interrupted attempts retain their full bounds. An over-reservation usage
report halts further calls and preserves the overage.

Caps cover the whole invocation and all workers. On a cap stop, the active partial
execution is void and remaining runs are `not_run_budget`; teacher halts leave
remaining runs `not_run_halted`. Already in-flight requests may finish, and all
remain accounted for. An output cap too small to reserve even one configured
response makes no API request. A cap stop returns nonzero to distinguish an
incomplete diagnostic from a completed run.

Repeat the exact run command with `--resume`. Input, content, code, config,
dependency, mode, run-count and cap identity must match. Worker count can change.
Existing atomic records are skipped under per-key locks. An interrupted run with
no record resets/replays in a new attempt directory; its old calls and reservations
remain visible. Durable usage receipts settle idempotently after a crash between
receipt and settlement. A halted/budget-stopped invocation stays stopped on resume;
this flag never silently replenishes caps or clears teacher failures. Keep all
prior outputs when starting a separately authorized new invocation.

### Dry run and canary

Add `--dry-run` to spec generation and execution for coverage-shaped synthetic
fixtures: manifest authorization is bypassed exactly as in CG's dry loader, while
source hashes and episode checks still apply. The recorded-transition fake
replays only the selected prefix, then emits synthetic observations and never
wins. The fake API client runs through the same Teacher, prompt, reservation,
ledger and cache path; returned model ID is `FAKE_NO_MODEL`. It calls no SDK,
network, student model or GPU. Synthetic records cannot establish real replay or
feasibility.

For a real smoke diagnostic, add `--canary`: it selects exactly the first case,
one run, one worker, and clamps global caps to at most 100,000 input / 8,192 output
tokens without changing collection sampling. Smaller caller caps remain smaller.
It cannot be combined with `--dry-run`. It may stop before completion if its caps
are reached. Implementing this switch does not mean a real canary was executed.

### Outputs

- `plan.json`, `provenance.json`, `run_manifest.json`: effective settings, content
  and code hashes, dependency versions, frozen data inputs, source/student
  provenance, teacher alias, actual returned model IDs/timestamps, training
  exposure zero. `completion.json` records completeness and global accounting.
- `records/<key>.json`: immutable per-run record keyed by `(case_id, mode, run)`,
  execution `TrajectoryRecord` (null for unrun/setup failure), prefix verification,
  void reason, environment success (unknown for void), usage totals and call
  metadata, evaluation interaction ledger summary.
- `calls/<key>/<attempt>/`: isolated cache, exact sanitized prompt/output blobs,
  per-attempt `usage.jsonl`, and evaluation `budget.jsonl`. Failed attempts remain.
- `global_teacher_budget.jsonl`: atomic reservations and settlements across workers.
- `summary.json`, `summary.md`: per-case expected/valid runs, voids, successes,
  found/picked counts, median post-branch turns to success, feasible flag, usage.
  Feasible is true after at least one valid success, false after a complete valid
  set without successes, and unknown for incomplete sets without a success.

Finding/picking and other CG metrics are from `repair.guidance.derive_metrics`
with the group target and effective branch. They measure the suffix. Added
metrics include whole-episode legal non-refused target heat/cool/clean,
publicly supported placed count (unknown if unobserved), suffix policy turns,
turns to success and budget remaining at discovery. Transform/placement are
local progress; only environment execution success establishes task completion.
Summary distinguishes per-record usage from all-attempt usage (including orphan
attempts), conservative outstanding bounds and unknown quantities.

Usage rows include `repair_request_id`, `repair_logical_request_id`, zero-based
`repair_attempt_index`, and `failure_class` alongside the transport's original
IDs and attempt index. `transport_will_retry` preserves the original transport
decision; `repair_will_retry` and `repair_retry_wait_seconds` disclose the added
retry. `will_retry` covers either layer for existing resume reconciliation.
Append-only `repair_outcome` rows record failed policy attempts (including empty
completed replies) without adding usage or settlements; usage summaries exclude
those bookkeeping rows. Unknown failed attempts retain their full reservations.

## Harness fact table

```python
from repair.facts import FactTable

facts = FactTable.from_record(episode['initial'], episode['steps'])
facts.update({'action_parsed': 'inventory', 'action_outcome': 'legal',
              'observation': 'You are not carrying anything.'})
public_json = facts.to_json()
reviewer_text = facts.prompt_text()
```

Initial values can be text, `{observation: ...}` or `{visible: {observation: ...}}`;
steps support flat public records and runner/executor `visible` observations.
Only `action_parsed`, `action_outcome`, and observation are read. IDs are policy
turns, with -1 for initial evidence. Every known state/condition carries its
evidence IDs. Unmatched goals and insufficient observations stay unknown.

Supported goals: put a/some object in/on a receptacle; examine/look at an object
with/under the desklamp; clean/hot/cool adjective goals and clean/heat/cool-and-put
goals; put/find-two-and-put goals. Output includes object type, required state,
receptacle type and count. Regular two-object plurals and knives are normalized.
No task-name fallback exists.

Holding updates require a legal, non-refused take or move/put. Inventory is direct
evidence and reconciles the set; disagreements with previous known holdings are
retained, and the first inventory observation can have unknown prior agreement.
`Nothing happens.` and inadmissibility record refused actions and repetition
counts. Parser errors do not change holdings. Refused takes retain the held-object
evidence without asserting that holding was the cause of refusal.

Object sightings and latest locations come from public local lists, including
`On the Y`, `In it`, opening descriptions and arrivals. The initial room inventory
is not silently assigned to a hidden room container. Opening, closing, visits and
observed contents are separate fields. Opening without a recognized content list
records an observation opportunity but cannot clear the receptacle. Explicit
nothing or a recognized object list lacking the goal object clears that location
at that observation time. Later takes, placements or sightings update/invalidate
location evidence. Cleared means the publicly observed contents excluded the
goal type at that time, not that an object cannot ever be there.

State changes require legal, non-refused heat/cool/clean/use actions. Heating and
cooling supersede each other's current temperature evidence. Lamp use establishes
lamp-on evidence; a lit-object condition additionally needs contemporaneous
holding evidence. Conditions expose `satisfied`, `known_unsatisfied`, `refuted`,
or `undetermined` with evidence for location known, target held, hands free,
required state done, second instance and placed count. Inventory contradiction
can refute prior holdings; disappearance from an observed list can refute a known
location. No object location is inferred from household priors. Placement counts
distinct goal instances currently supported at the goal receptacle type, separate
from required transformation and overall success.

`prompt_text()` is compact, deterministic sorted JSON text with explicit
`unknown` values. It contains facts and evidence IDs only, not raw event rows,
paths, admissible-command-derived facts, rewards, success labels or audit fields.
This is the fact boundary for a later R-role; E0.2 does not implement that role.

## Offline audit and verification

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m tools.facts_audit \
  --episodes 'reports/runs/repair_pilot/cov_theta0_umech/coverage/attempt-*/episodes/*/episode.json' \
             'reports/runs/repair_pilot/cg_probe/run1/rollouts/records/*.json' \
  --output /tmp/facts-audit.json

CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q \
  tests/test_takeover.py tests/test_facts.py tests/test_cg_probe.py tests/test_repair_*.py
```

The audit accepts quoted globs (also `--glob`) or paths, unwraps `execution`, and
reports goal parse coverage, each inventory comparison, refused takes while
holding, cleared counts, skipped absent executions and exception types. No input
files are changed. Exceptions are reported without raw exception text; no matched
files or any exception yields nonzero status. It is CPU-only and makes no model,
API, network or environment calls. Coverage statistics alone do not establish
semantic accuracy; tests exercise public prose and state transitions explicitly.
