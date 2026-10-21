# Chain collection recovery for the 2026-10-20 quota incident

`teachers.chain_recovery.recover_chain_collection` and
`python -m tools.recover_chain_collection` implement an offline migration into a
**new** collection directory. The implementation shares the checked-copy,
read-only inventory, and separate-path primitives from
`teachers/full_train_recovery.py`. It never constructs a teacher, resets an
environment, or issues an API request. No real recovery or collection was
launched while implementing this change.

## Registered rule and evidence

Rule: `chain_quota_void_20261020_v1`; exhaustion cutoff:
`2026-10-20T22:45:00+00:00`. A credit requires a reserved logical call whose sole
terminal transport record is `failed`, HTTP 429, `RateLimitError`,
`retryable=false`, `will_retry=false`, `usage=null`, and no model-output reference.
Every credit lists the exact `call_id`, logical call ID, failure-record SHA256,
enqueue timestamp, and transport reservation timestamp.

The legacy ledger's `timestamp` precedes the rate-limiter wait. Eligibility uses
`reservation_timestamp` (UTC epoch, recorded when admitted by the limiter), or
`timestamp` when admission metadata is absent. All 32 AD-FTP incident failures
were admitted between 22:45:15.901623Z and 22:45:45.675488Z; eleven were queued
before 22:45. The legacy sanitized ledger does not retain the provider error
body/code. Classification therefore binds the registered incident and the frozen
transport's non-retryable quota signal; it does not invent a saved
`insufficient_quota` response body.

Non-quota terminal failures, unresolved starts, pending requests, changed
receipts, changed acknowledged ledger prefixes, inconsistent counters, and
changes during the audit/copy are refused. Successfully completed HTTP retry
chains remain spent. SmartAD's separately evidenced unsupported-temperature
fallback remains spent and retains its distinct sampling protocol. A recovery
of an already recovered source is refused, preventing repeated credits.

## Copy, counters, and publication

The source must be stopped and quiescent, including all SmartAD candidate
workers. No source locks or files are created. Every non-lock source file is
inventoried; copies are private, checked by SHA256, and published by renaming a
staging directory only after verification. Interrupted copies can resume.

Completed AD-FTP receipt bundles and SmartAD candidate/outcome files remain
byte-identical. Original stages, shard reports, identities, manifests, and
provenance are archived under `recovery/source/`. Failed and interrupted work is
retained. Imported final tasks bypass teacher construction; imported candidate
slots return their validated receipts. Missing or changed imported evidence
fails before calls can replace it.

The new identity binds the original identity and progress/publication manifest,
the complete source inventory, the immutable copied inventory, the recovery
rule, every cap-credit and cap-extension record, and current code/dependency hashes. For an
unfinished source with no publication manifest, `result.json` is the progress
manifest; receipt and stage hashes remain independently bound by the inventory.
Code/config/protocol changes on recovery resume are refused.

Each unfinished counter is initialized offline to:

```text
initial attempts = original reserved logical calls - exact quota credits
current floor    = initial attempts + newly reserved logical calls
```

The original counter is preserved, and the new counter and immutable
`recovery/credits/*.json` record bind the complete inherited ledger prefix.
The prefix stays intact. Runtime identity adoption changes only the run/path
namespace; teacher, sampling, prompt, and cap must match. Completed imports keep
their original counter bytes; their credits are disclosed but never spent.
The original caps remain in the counter identities: 123 calls per AD-FTP task
and 41 per SmartAD candidate (410 per task). Only the separately registered
`cap_extension` record below can increase the runtime cap. No successful calls,
rate-limit reservations, or billed tokens are refunded. Ordinary
`teachers.recover` acknowledgments preserve both the credited floor and the
registered effective cap on a subsequently interrupted migrated counter.

AD-FTP uses its existing `execution_attempt` transport-retry directories.
SmartAD retains the original slot and puts the interrupted slot's new execution
under `candidate-NN/transport_retries/attempt-1`, with a credited counter and
copied inherited ledger. Both restart with a real environment reset. Neither
fabricates a restored environment or resumes a partially executed episode.

`inherited_usage` and `new_usage` separate canonical calls, logical reservations,
known billed tokens, quota voids, unknown usage, and actual environment steps.
Archived stages and copied receipt bundles are not summed. `api_attempts`
retains total historical logical reservations, including void calls;
`new_teacher_calls` counts only new HTTP call IDs. Receipt-level historical
usage is retained for audit and must not be added to these totals again.
SmartAD selection carries this disclosure into its verified collection.

The existing collectors publish the K30/K100/K300 views using the original
`subset_views` bindings only when the whole K500 collection is final. Views
report zero additional calls/transitions; do not sum them as independent runs.
K continues to count trusted executed seed trajectories, not candidates or
retries. This recovery changes no model-visible fields, split, training labels,
task identity, success criteria, or training exposure.

## Commands

Offline audit (prints every task's final status, void call IDs, remaining cap,
full extension records with attempt IDs/calls/hashes, and any remaining
fresh-attempt cap warning; writes no collection):

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.
export SED_HASH_CACHE_DIR=/tmp/chainrecover-hash-cache
old=/home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/adftp-chain/chain_k500_s0
.venv/bin/python -m tools.recover_chain_collection \
  --method adftp --recover-from "$old" \
  --output /tmp/chainrecover-adftp-dry-run --dry-run \
  > /tmp/chainrecover-adftp-plan.json
```

The standalone CLI defaults to the exact source identity's config; relative
manifest paths resolve from the current project directory. `--config` may be
supplied but must match the frozen source config. To copy only, choose the new
real output directory and omit `--dry-run`; this still performs zero calls.

The shell launchers accept recovery on a single explicit cell. The following
prints a plan; omitting `--dry-run` first copies and then launches the remaining
collection work:

```bash
base=/home/xueqi/hq/projects/ICML27_agent_distillation
support="$base/data/verified/seeds_full_train_r2/subsets"
bash scripts/rai_collect_adftp.sh \
  --cell chain_k500_s0_r2 --source-view "$support/k500" \
  --subset-view "30=$support/k30" --subset-view "100=$support/k100" \
  --subset-view "300=$support/k300" \
  --recover-from "$base/reports/runs/adftp-chain/chain_k500_s0" \
  --output-root "$base/reports/runs/adftp-chain" \
  --rate-limit-dir "$base/data/teacher_rate_limit_chain" \
  --config configs/base.yaml --shards 32 --dry-run
```

For SmartAD, use `rai_collect_smartad.sh` and the corresponding old/new
`smartad-chain` paths, after stopping the source workers. Model/config, support
views, and limiter must match the old identity. On an existing migrated output,
add `--resume`. The separate offline CLI can be used without the launcher.

## Registered cap extension following the user's decision

`cap_extension` is a second record of `chain_quota_void_20261020_v1`, separate
from `cap_credit`. It requires at least one quota credit, an exhausted counter,
and a latest attempt that ended in **BudgetExceeded with no outcome**. The
error type is checked in the last event's content-addressed verification blob;
a warning, generic transport error, or collection-wide cap failure is not
sufficient. Every earlier AD-FTP method slot must have a completed effective
execution. Superseded quota-interrupted executions remain as inherited evidence;
their completed transport retries establish completion of those earlier slots.
Missing earlier attempts, incomplete earlier slots, final outcomes, and
insufficient or inconsistent evidence refuse the extension.

The extension is computed from the collectors' own constants:

```text
extension          = per-attempt call budget - quota credits
effective cap      = original cap + extension
initial counter    = original reserved calls - quota credits
fresh budget       = effective cap - initial counter = per-attempt call budget
cap + credits + extension = original cap + one fresh attempt
```

Each record binds the task/candidate, episode ID, method-attempt number, old and
fresh directories, reserved and interrupted-attempt call counts/IDs, credited
calls, extension, and counter/ledger/episode/event/evidence hashes. Records are
embedded in the collection identity and copied under `recovery/extensions/`;
the corresponding credit binds the extension hash. Changed or missing records
fail before new calls. Counter floors never decrease beyond the quota credit.
The AD-FTP collector permits only the registered fresh execution of the
interrupted method slot; it cannot turn the extension into additional method
attempts. Both collectors use real resets in fresh directories.

## Real read-only audit results

The read-only real AD-FTP audit reports **460 final imports and 40 tasks to
run**: 37 without an existing counter, plus the three exhausted tasks below.
There are 32 registered void calls: 29 belong to final imports and three to
unfinished tasks. The plan makes zero new calls and does not create its proposed
output directory or write under `reports/runs`.

All 12,581 historical logical/HTTP calls remain inherited, along with 11,979
executed environment steps, 49,663,569 input tokens, 883,415 output tokens, and
465,817 reasoning-token metadata counts. Reasoning counts are not added to
output tokens. The 32 quota voids are explicitly unbilled; there are zero other
unknown-usage calls in the canonical inherited ledgers.

Audited source anchors:

```text
source git commit:       d7ad7fe9c0bef0e771cbd8e033f59d84f42c16ca
identity (canonical JSON): 595a727e2f75770531585cc2a5d0d7d57a5c378bf621a86623af3f0fad7f8e96
result.json (file SHA256): 01d2fbd15ddab862767028c9a4f0dd81297ca2d3c311977a10bbc6372dc82c7e
347744-file inventory:    691b1632fcba2b53be00e626a21fc306509918caaf5a14a762920f3a6b4b713b
```

A one-call credit alone did not restore a full episode's budget. The three
eligible AD-FTP tasks have already reserved 123 calls. Each now receives one
credit and a **40-call extension**, leaving **41** calls for one fresh execution
of its interrupted third method attempt:

| Chain position (zero-based) | Work directory | Interrupted calls | Old / credit / extension / remaining |
| --- | --- | --- | --- |
| 19 | `b979abd14a4d8cd9b4b2` | 9 | 123 / 1 / 40 / 41 |
| 125 | `a524fd09469ce541762f` | 5 | 123 / 1 / 40 / 41 |
| 130 | `42167910dbe4c56c42ee` | 23 | 123 / 1 / 40 / 41 |

For example, position 130 made 17 successful calls in its interrupted second
attempt before the quota rejection, then re-executed that method attempt.
Those successful calls stay spent. The final interrupted third attempts also
retain their evidence, usage, and transitions. Their new execution directory is
`work/<key>/transport_retries/attempt-3/attempt-1`. Each effective cap is 163 and
each initial credited counter is 122; the difference is exactly 41. This changes
available execution budget, not success labels or a guarantee of task success.

| Source | Final tasks imported | Tasks to run | Final candidates imported | Candidate slots to run | Credits | Extensions | Extension calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AD-FTP | 460 | 40 | N/A | N/A | 32 | 3 | 120 |
| SmartAD | 18 | 482 | 265 | 4,735 | 32 | 0 | 0 |

The stopped SmartAD source contains **no candidate execution eligible for a cap
extension**. Its 32 interrupted candidate slots ended at quota failures rather
than `BudgetExceeded`. Thirty slots across twelve tasks still have fresh-attempt
cap warnings; the registered eligibility rule does not extend them. The 4,735
unfinished candidate slots include the still-dependent first-candidate reuse
decisions and therefore are not a promise of 4,735 new executions.

SmartAD inherits 4,533 logical/HTTP calls, 4,496 environment steps, 14,275,376
input tokens, 260,100 output tokens, and 125,612 reasoning-token metadata counts.
Its 32 quota voids are unbilled; there are zero other unknown-usage calls.
Source identity SHA256:
`8a27d50f7e888d2e59be1d5a5274894e65273a57b6988f4a6ef51bb99cf51dc5`.
The 34,887-file source inventory SHA256 is
`9ba60deda286a6d4ce27f39217a46d26e2f6823ac9070153f247b3bab4e0dd7a`.

Full dry-run plans, including all task/attempt IDs and extension hashes:

- `/tmp/chainrecover-adftp-cap-extension-plan.json`
- `/tmp/chainrecover-smartad-cap-extension-plan.json`

The audits compare complete source inventories before and after inspection.
Both proposed output directories remain absent, and both plans report zero new
teacher calls. All dry-run output and hash caching is under `/tmp`.

## Validation

CPU tests use fake clients/environments; `tests/conftest.py` blocks CUDA, network,
and the OpenAI API. Recovery tests exercise byte-identical imports, no calls for
final tasks/candidates, successful calls remaining spent, exact credits,
counter floors, interrupted-copy resume, source/import/credit/prefix tampering,
non-quota refusal, timestamp admission, launcher dry-run, SmartAD candidate
re-execution/selection, and exact support views.

The optional LEAP transport regression tests need the pinned upstream prompts,
which are absent from this worktree. They were supplied read-only from the
existing local checkout at
`/home/xueqi/hq/projects/ICML27_agent_distillation/third_party/leap_llm`, verified
at the repository's configured pin
`a3376d89a2eeee88f03555a4174a0bdf7faf8be6`. No upstream files were copied and no
network access was used. The pre-sharding AD-FTP report-path assertion was
updated to `shards/1-0/teacher_report.json`.

Validation results for the cap-extension follow-up:

- **290 affected tests passed across the main run and dependency-corrected
  rerun**: 282 passed in the main run; its only eight failures were missing
  LEAP prompt files, and all eight passed with the pinned local checkout above.
  Coverage includes AD-FTP, SmartAD, chain collections, recovery, candidate
  sharding/relocation, capped teachers, full-TRAIN recovery, fail-loud transport,
  baseline retry policy, and launchers.
- The recovery file now has **39 cases**, including **20 new cap-extension
  cases** for eligibility/refusals, exact terminal failure evidence, multiple
  credit arithmetic, hash tampering, fresh execution, inherited billing, repeat
  resume, and refusal of any call beyond the registered budget.
- Test logs: `/tmp/chainrecover-cap-extension-tests.log` and
  `/tmp/chainrecover-cap-extension-leap-tests.log`. The first deliberately
  retains the missing-dependency failures; the second records their successful
  rerun. Earlier test runs overlap and are not added to the 290 count.
- `git diff --check` passed.
- Both final real dry-runs exited successfully with identical before/after
  source inventories, zero calls, and absent proposed output/staging
  directories. Summary: `/tmp/chainrecover-cap-extension-summary.json`.

The main affected suite command was:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/test_chain_recovery.py tests/test_adftp.py tests/test_smartad.py \
  tests/test_capped_teacher.py tests/test_teacher_failloud.py \
  tests/test_full_train_recovery.py tests/test_chain_collections.py \
  tests/test_smartad_relocation.py tests/test_smartad_shards.py \
  tests/test_adftp_sad_launchers.py tests/test_baseline_transport.py
```

The eight missing-prompt cases were rerun without changing repository files:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.:tests .venv/bin/python - <<'PY'
from pathlib import Path
import pytest
from baselines.leap import protocol
protocol.UPSTREAM = Path('/home/xueqi/hq/projects/ICML27_agent_distillation/third_party/leap_llm')
raise SystemExit(pytest.main(['-q', 'tests/test_teacher_failloud.py', '-k', 'leap_stage']))
PY
```

The directly reproducible final recovery subset is:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/test_chain_recovery.py tests/test_smartad_relocation.py \
  tests/test_smartad_shards.py
```

No commit, push, real teacher request, real environment execution, or write into
`reports/runs` was performed for this task.
