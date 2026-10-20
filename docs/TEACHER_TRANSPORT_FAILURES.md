# Teacher transport failures — engineering fixes, 2026-10-19/20

A stage cannot complete with fewer labels because a teacher request failed.
The shared transport retains its existing bounded retry policy. Once that policy
fails, `TeacherCallError` (including budget/halted subclasses) propagates through
the runner and collector. No SDK exception body is printed or stored.

A returned response must have known input/output usage and either explicit
`status=completed` or a recognized output-cap truncation. A response with
`status=incomplete` is truncated when `incomplete_details.reason=max_output_tokens`,
or, only when `incomplete_details` is absent/null, its reported `output_tokens`
is at least the requested `max_output_tokens`. An explicit other reason (including
`content_filter`) never uses that fallback. Failed, cancelled, in-progress,
unknown-status and other incomplete responses, or missing usage, retain
output/usage blobs and raise with the existing transport policy.

2026-10-20: a response truncated at the step/output cap is a model output (upstream: parse error), not a transport failure.

Truncated responses return a normal `TeacherResult` with `truncated=True` (also
stored in metadata), retaining the raw text, original response status, incomplete
reason, requested cap and usage. The terminal ledger row is `status=rejected`,
`error_type=TruncatedOutput`, with the parser result recorded separately.
Truncations consume their logical slot and all token reservations, never enter
the reusable success cache, and do not halt a capped collector on restart.
Reasoning tokens can consume the whole cap with no visible output; these counts
remain accounting metadata only.

A completed or cap-truncated empty/malformed output reaches ordinary method
validation. H0 execution (including seed/full-TRAIN collection) records a parser
error/failed policy step and continues the episode; LEAP validation rejection
consumes its method try. Parseable returned text still follows the existing
parser/validator; truncation alone does not force a different label or mask.
Each method retains its existing rejection behavior, including fatal format
rejection for STeCa and
AgentDistill clustering/consolidation, and HAD/GAD retention of unparsable raw
labels with the existing downstream masks. This does not change algorithms,
prompts, K, splits, student/teacher identities or caps.

`teacher_report.json` belongs to a stage invocation. It separates
`transport_failures` (terminal failures), `transport_failed_attempts` (HTTP or
response failures, including recovered retries), `truncated_outputs`,
`parser_rejections`, and `rejections` (method-level output validation). Truncations
are counted once per returned response, with no transport/unknown increment.
An output can appear in truncation, parser and method counters; these counters
must not be summed as independent calls.
Existing `report.json`, `result.json`, and `summary.json` expose these counters
where present. Per-method disclosure `teacher_outcomes` counts historical failed,
incomplete and unknown ledger attempts separately from parser rejections and
`truncated_outputs`. A `TruncatedOutput` ledger row counts as a parser rejection
only when its separate `parser_error` field is set. Those attempt counts include
failed requests that eventually recovered. Billing and
reservations always come from the original ledgers, never from these reports.

## Recovery

Stage transactions keep their failed attempt directory and do not create
`complete.json`. Rerun the same stage after fixing the transport. LEAP resumes
within its original three reserved attempt slots. Each transport error aborts the
current invocation, including an unsupported-sampling HTTP response; the already
registered sampling fallback is recorded and applies on the next invocation.
Only output validation can create an exhausted/rejected LEAP outcome. If failed
or unknown requests leave insufficient slots for a valid outcome, it raises and
requires a separately accounted rerun.

Capped collectors deliberately remain halted until explicitly recovered:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/xueqi/tf5-env/bin/python -m teachers.recover \
  --counter /path/to/counter.json --reason transport_fixed
```

Use `interruption_reviewed` for a reviewed unknown/interrupted request. Recovery
records the previous counter and a hash of the entire acknowledged ledger prefix.
It never deletes evidence, refunds calls/tokens or raises a cap. Unknown usage
stays unknown. A modified or truncated ledger is rejected. If the remaining cap
is exhausted, use a new run with distinct provenance and retain/disclose the old
run; do not reset counters or edit caps in place.

A failed real execution is retried in a fresh `transport_retries` directory using
an actual reset and full execution (with normal request-cache reuse). No branch
state is fabricated. Unknown historical starts fail on discovery before they can
be retried. Successful or method-rejected completed/truncated attempts retain
the original method slot. Seed/full-TRAIN manifests bind historical failed executions
separately from method attempts. AgentDistill disclosures include retry resets.

Existing stages that were already incorrectly marked complete are not
retroactively repaired or trusted as new data. Preserve those runs and rerun
under the new code/provenance. Normal frozen identity checks still apply; this
fix does not bypass them.

The interrupted full-TRAIN `549503d` run has a dedicated
[offline migration](FULL_TRAIN_RECOVERY.md). It imports only complete attempts
into `seeds_full_train_r2` under a new identity, preserves recovered legacy retries,
and carries forward the entire reservation database. Its imported attempts are
reused without calls; excluded attempts remain hashed source evidence.

## Audited callers and offline validation

- LEAP: extract and correct; CoA: teacher and teacher-evaluate.
- AgentDistill: collect, abstract, cluster, consolidate.
- GAD/HAD: teacher-r; STeCa: reflect; AD-FTP: collect.
- Shared seeds/full TRAIN, corrections, rewrites, episode runner, teacher
  evaluation suite, mechanism takeover/intervention branches.
- Other imports of teacher results are student/scripted policy adapters and do
  not call a teacher transport.

Tests use CPU, fake environments and stub transports, with the repository's
network/API prohibition fixture. Regression coverage checks stage failure without
completion, cap-truncation classification and method handling, retained format
rejections, cache exclusion, separate report counters, real reset on retry,
explicit capped recovery, non-refundable caps, and resumption to completion.
No scientific result or faithful baseline reproduction is claimed by this
engineering validation.
