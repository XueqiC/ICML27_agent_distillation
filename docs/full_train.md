Full TRAIN support (user decision 2026-10-19)
===========================================

This is one additional grid cell, `full`, for testing whether observations at
K <= 100 persist with ample training data. It does not change the few-shot rule,
the split freeze, teacher sampling, prompt, action parser, or termination rules.

`data/manifest/seeds_full_train.json` uses `all_train_v1`, seed 0: every eligible,
unsealed TRAIN game in the frozen seed pool, including sibling trials. There are
3,553 candidate instances in 1,465 groups: look 308, pick 790, clean 650, cool 533,
heat 459, two 813. Global order is ascending SHA256 of canonical JSON
`[seed, task_id]`; `candidate_order` records it explicitly. The grouped `selected`
lists retain that order within each type. All replacement lists are empty.
Selection `K` and `K_counts="candidates"` describe candidates, not trusted seeds.

Collection attempts each candidate at most three times, keeps the first real
environment-verified success, and excludes candidates without a verified success.
The complete collection's `K` and `trusted_trajectories` count successes; unique
instances and groups are reported separately. `failed_candidates` retains every
candidate with three completed real failures. Transport failures and unknown
executions remain separately disclosed and do not consume these task slots.
No replacement or external experience is
added. No collection or scientific result is claimed by adding this support.

For recovery of the interrupted collection on rai:

```bash
bash scripts/rai_collect_full_train_r2.sh --preflight  # Copy/validate; no API calls.
bash scripts/rai_collect_full_train_r2.sh             # Starts live collection.
```

The r2 launcher fixes the original 32 shards, disables CUDA, and uses the ALFWorld root from `rai_collect_parallel.sh`
(overridable with `ALFWORLD_DATA_ROOT`), takes a launcher flock, and preflights every
game's hashes before spawning processes. It writes one log per shard under `logs`.
Worker i owns global-order indices `i::N`. All workers write one collection at
`data/verified/seeds_full_train_r2` with the existing attempts/episode/teacher/receipt
layout and share `data/teacher_rate_limit` **inside that collection**, at the rate limits in base.yaml:
4,500 requests and 1,800,000 estimated input-plus-output tokens per rolling minute (90% of the
provider headers re-read 2026-10-19; they were 500 / 200,000 on 10-04).

The capped-teacher module's shared SQLite reservation ledger is
`teacher_budget/reservations.sqlite3` inside the collection. Before each transport
attempt, it atomically reserves the existing limiter's input estimate (canonical
prompt characters / 4 rounded up, plus message framing). Each uncached generation
counts as one logical call; its HTTP retries each reserve additional input tokens
without adding logical calls. Cache hits consume neither. The hard limits are
160,000 logical calls and 700,000,000 reserved input tokens. Reservations are
conservative admission accounting, not billed tokens; per-attempt usage ledgers
retain actual and unknown usage. Failed and interrupted reservations are never
refunded. The existing K100 teacher retry/sampling configuration is preserved.

On a cap stop, workers retain evidence and publish `budget_exhausted`; incomplete
collections are not trainable. Resume verifies selection/config/code/dependencies,
shard count and caps, reuses completed receipts, and preserves uncertain attempts
without resubmitting them. The caps cannot be reset or raised by `--resume`; a
cap-stopped run remains resumable for evidence recovery, with no further paid
requests. Interrupted runs below the caps continue the remaining attempts.
Receipts persist immediately; aggregate JSON checkpoints are batched every 25
candidates and on worker exit. After a hard kill, resume rebuilds the aggregate
from receipts. Keep the same worker count on resume.

`load_verified_collection` requires all candidates to be accounted for before
accepting a full collection. It verifies failures as well as wins, permits sibling
groups only for this rule, and checks actual trusted K rather than candidate K.
ETO and the shared IPR/AgentGym/LEAP/Kang loaders accept the rule. Seed-SFT uses the
same evidence loader. No full-train few-shot subset views are defined.

For Seed-SFT, prepare `configs/seedsft_dose_grid_full_train.json` with
`scripts/prepare_seedsft_dose_grid.py` and an existing local tokenizer snapshot
after collection. Preparation resolves trusted K and measured token exposure.
Use the resulting ready grid with `DOSE_GRID` and `--array=0-0` on
`scripts/hpg_seedsft_sweep.slurm`. Baseline job arrays accept `CELLS="full"` and
map it to `data/verified/seeds_full_train_r2`.

The interrupted `549503d` run is retained read-only. For its replacement identity,
use [the recovery tool and 32-shard launcher](FULL_TRAIN_RECOVERY.md); valid paid
attempts are copied and reused, transport failures do not consume task slots, and
all source reservations remain spent. The Seed-SFT full grid and AD-FTP full cell
also explicitly select `seeds_full_train_r2`.
