# Incident 2: V0 review and label transport

Worktree base: `3c4b3718c5b6815b43c0d7fe3bd0bbf02fdb5488`.
Validation uses fake clients/environments, CPU only, no network or teacher API.
Changes are unstaged; no git metadata or incident evidence was modified.

## Behavior

- V0 review/label clients opt into a 600-second read/write/pool timeout and
  5-second connection timeout. SDK internal retries remain disabled.
- Timeout and connection failures receive at most three audited attempts per
  logical call, with SDK-style exponential backoff (0.5s, 1s before jitter).
  Every attempt has its own reservation and terminal receipt. Unreported
  connection-failure usage stays `usage: null` in the receipt, with an explicit
  zero `reported_usage` settlement; the original reservation remains in history.
  This does not establish zero provider billing.
- Plain 429s retain their separate 20-attempt retry budget. Quota, authentication,
  budget stops, other HTTP failures and unknown response outcomes remain fatal.
- Exhausted timeouts produce a durable `teacher_timeout` review round, without
  a patch, execution or verification. Other failures continue, including with
  spawned workers. A timed-out label remains an audited action-only decision.
  Resuming a settled timeout does not issue another teacher request.
- The transport uses the existing strict V0 review/label JSON validators.
  Valid review receipts use the shared ledger's `success` status with
  `response_status: completed`; request receipts use `completed`.
  Invalid review JSON retains its raw output and consumes the existing one
  format retry, then records `format_retry_exhausted`. Output-cap retries stay
  separate. Invalid rationale labels retain the existing action-only fallback.
- Shared teacher changes are guarded by V0 policy flags; other collectors and
  baselines keep their existing timeout, parsing and retry behavior.

## Read-only incident accounting

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
  /home/xueqi/hq/projects/ICML27_agent_distillation/.venv/bin/python \
  -m tools.v0_usage_check \
  --review-dir /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1/review \
  --budget /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1/teacher_budget.jsonl
```

The original incident has 18 usage files, 19 attempts, 18 settlements and one
unsettled timeout. Eighteen terminal rows are `rejected / ActionParseError`;
the timeout is `APITimeoutError`, 60.901 seconds, nonretryable. The 18 state-link
rows are bookkeeping, not additional calls or usage. The check reconciles each
terminal receipt with its reservation/settlement and the final cumulative budget
snapshot, without summing repeated snapshots or recovered request copies.

| Settled usage | Tokens |
|---|---:|
| Input | 323,675 |
| Output | 42,528 |
| Reasoning (included in output) | 34,555 |

The existing incident's timeout remains unsettled: this audit does not rewrite
historical accounting or clear the halted run.
SHA-256 comparisons before and after reconciliation confirmed all 19 incident
ledger files (18 usage ledgers plus the shared budget ledger) were unchanged.

## Validation

Commands use `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.` and
`/home/xueqi/hq/projects/ICML27_agent_distillation/.venv/bin/python`:

```text
-m pytest -q tests/test_v0*
-m pytest -q
```

V0: **111 passed** in 52.10 seconds, including sequential/spawned worker
continuation, 600-second SDK construction, timeout/connection retry accounting,
quota/auth stops, strict JSON format retry exhaustion and durable label fallback.

Full CPU suite: **2,766 passed / 51 failed / 102 skipped** in 2,176.04 seconds
(36m16s). Compared with the requested 2,742 passed / 51 failed reference, this
adds **24 passing tests**, with no new or resolved failures. All 51 failure IDs
exactly match the saved baseline in
`spike_out/v0_work_logs/full_cpu_comparison.json`.

Logs: `/tmp/incident2-v0-final.log` and `/tmp/incident2-full-cpu-final.log`.
`git diff --check` passed; HEAD is unchanged and the index is empty.

## Unstaged files

- `repair/v0_data.py`
- `repair/v0_review.py`
- `repair/v0_transport.py`
- `teachers/capped.py`
- `teachers/openai_teacher.py`
- `tests/test_v0_transport.py` (new)
- `tests/test_v0_workers.py`
- `tools/v0_usage_check.py` (new)
- `reports/incident2_v0_transport.md` (new)
