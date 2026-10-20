# Full-TRAIN recovery without duplicate paid attempts

`tools/recover_full_train.py` migrates the interrupted `549503d` collection to
`data/verified/seeds_full_train_r2`. It does not construct a teacher, reset an
environment, or call an API. The source is read-only, must be quiescent, and stays
at its original path. Do not point the old launcher at the source.

Read-only classification:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/xueqi/tf5-env/bin/python \
  tools/recover_full_train.py --dry-run \
  --summary logs/full_train_recovery_dry_run.json
```

The default source is
`/home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train`.
`--source`, `--output`, and `--config` can be set explicitly. No output collection,
lock, SQLite journal, or cache is created by `--dry-run`. A requested summary must
be outside the source and destination; dry runs also forbid summaries in
`data/verified`.

For a later operator-run recovery and preflight, with **no billable calls**:

```bash
bash scripts/rai_collect_full_train_r2.sh --preflight
```

This command creates the new directory by verified copies, or validates an
existing recovery. Omitting `--preflight` then starts live collection. The launcher
always uses 32 shards, three real attempt slots, `all_train_v1`, and the original
160,000-call / 700,000,000-reserved-input-token caps. `--resume` is accepted for
clarity; recovery-or-resume is automatic. All workers disable CUDA. The launcher
does not issue a canary request. Keep code, configuration, dependencies, selection,
and shard count fixed after publishing the new identity.

## Evidence rule

`full_train_valid_complete_copy_v1` classifies each directory, including attempts
not yet present in the aggregate manifest:

* `valid_complete`: receipt/evidence hashes agree, the episode finished at a real
  environment termination or harness step cap, and every teacher call has a known
  completed outcome. Real losses consume slots just as real wins do.
* `transport_failed`: a terminal failed/incomplete teacher row, or a
  `TeacherCallError` recorded in the episode or its event verification. Both kinds
  of 429, timeouts, connection failures, 5xx and incomplete responses qualify.
* `unknown_interrupted`: a started episode, a teacher start with no end, malformed
  usage evidence, or insufficient evidence of a completed environment outcome.
  Unknown status takes precedence when an attempt also has earlier failures.

The old ledgers lack `logical_call_id`. Retry chains are reconstructed within the
same run/cache key using increasing HTTP attempt numbers; a new attempt 1 starts
a separate chain. A failed row is recovered only with `will_retry=true` and a
subsequent successful/rejected completion in that chain. A terminal
`will_retry=false` cannot be forgiven by another successful call. Classification
never uses the outage timestamp. HTTP/error counts include both all failed rows
and terminal rows, so recovered retries remain disclosed.

The audit reconciles excluded error receipts with the source manifest's error
count and reports unreceipted/other excluded attempts separately. Settled usage
uses each `call_id`'s last ledger state. Unknown billing stays unknown.

## Copy, publication, and resume

The new identity binds the source identity and manifest file SHA256, a content
inventory of the entire source (excluding lock files), the recovery rule, complete
import/exclusion lists, inherited usage, and current code/config/dependencies.
Excluded evidence stays at its source path, with its hash and classification
reason in the new identity. Teacher/student identity, sampling and environment
semantics must match; path relocation and current transport handling are recorded.

Each imported attempt is copied byte-for-byte, including episode files, teacher
ledger, cache and raw-output blobs. No symlinks, moves, or hardlinks to source
evidence are used. Original receipts and starts become `source_receipt.json` and
`source_started.json`; the new receipt carries the original attempt number, path
and hashes. Valid attempts preserve source order and become consecutive slots
1..n. A complete episode whose receipt was interrupted can receive a reconstructed
receipt, with the absence of its source receipt disclosed.

Copies are verified before publication. Interrupted copying resumes in
`.seeds_full_train_r2.recovering`; the collector only sees the final directory
after an atomic rename. Changed source files or changed partial copies cause a
refusal. Recovery and full preflight rehash the source and imported evidence.
Worker startup checks source metadata, the immutable import receipts, current
identity and inherited budget floor. Each reused receipt verifies its episode and
teacher hashes again. A missing imported receipt fails closed rather than paying
for a replacement. Import-aware resume bypasses legacy retry discovery only for
the identity's verified imports.

A valid win completes the task immediately. Otherwise, only n+1..3 may execute.
Future terminal transport incidents follow current main's stage-failure rules
and retain their usage/reservations; they never become negative task labels.
The three-slot limit counts completed real attempts, not terminal transport
incidents. No prefix restore is fabricated.

## Inherited caps and downstream use

Recovery copies the source SQLite reservation database **intact**. Its counters,
request IDs, logical call IDs and stopped flag are preserved. New reservations
are added to those totals; neither excluded attempts nor retries receive refunds.
The identity records the separate last-state settled usage. Do not add imported
receipt usage to inherited usage: it is a subset of already inherited cost.

The read-only audit found 2,189 valid attempts (1,406 wins), 2,353 transport
failures matching all 2,353 error receipts, and 32 additional interrupted attempts
without receipts. Settled usage is 57,342 calls / 292,888,980 input tokens.
The supplied incident summary's 1,907 attempted instances differs from this audit:
the on-disk manifest reports 2,417 and scanning all attempt directories finds
2,426. Both are retained explicitly in the dry-run report; no timestamp is used
to decide which attempts to import.

For the interrupted source, reservation accounting is 59,727 logical calls and
289,633,277 input tokens. This leaves 100,273 calls and 410,366,723 reserved input
tokens. The estimate used for reservations differs from billed input tokens;
settled input usage is not substituted for the durable reservation counter.
No claim is made that the remaining allowance will finish all tasks. Exhaustion
publishes `budget_exhausted`, makes no further requests, and leaves an incomplete
collection ineligible for training. Caps are never raised automatically.

Baseline `full`, AD-FTP's `full_train`, and Seed-SFT's `full` grid explicitly use
`seeds_full_train_r2`. The old name is not aliased or overwritten. Existing loaders
require a complete collection and actual environment evidence; `all_train_v1`
still counts candidate instances in the selection and trusted trajectories in
the completed collection.

Offline tests use stub transports and fake environments, including zero calls
over imports, legacy recovered retries, renumbering, first-win stopping, the
three-attempt bound, inherited caps, source hash refusal and interrupted copying.
These are engineering checks, not scientific results.
