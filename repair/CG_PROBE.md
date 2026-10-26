# Condition-guidance probe

Implementation of the complete [2026-10-26 pre-registration](/home/xueqi/hq/projects/ICML27_agent_distillation/docs/direction/2026-10-26_condition_guidance_probe_prereg_ZH.md).
This is a TRAIN-side evaluation probe, with no training exports or updates.
The seven allowed groups are frozen in `repair.guidance.GROUPS`. A subset can
be selected for regeneration or a smoke run. No real experiment is implied by
the fake tests or `--dry-run`.

Run from the repository root with the environment Python and `PYTHONPATH=src:.`.
All new output directories are exclusive. Nothing modifies the existing repair
templates, schema, output caps, student implementation, or coverage artifacts.

## Generate and review hints

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m tools.cg_probe hints \
  --failures /path/to/coverage/failures.json \
  --coverage-root /path/to/coverage \
  --tasks 'look_at_obj_in_light-Newspaper-None-DeskLamp-224,pick_and_place_simple-Kettle-None-Shelf-1,pick_clean_then_place_in_recep-Kettle-None-DiningTable-7,pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18,pick_cool_then_place_in_recep-Potato-None-SideTable-21,pick_cool_then_place_in_recep-Apple-None-Microwave-19,pick_cool_then_place_in_recep-Egg-None-Microwave-5' \
  --branch-index 10 --config configs/base_e4b.yaml \
  --max-teacher-input-tokens 200000 --max-teacher-output-tokens 60000 \
  --output /path/to/cg/hints
```

The failure-list hash and actionable coverage summary must agree; episode,
events, and provenance hashes must match. Each selected case must be a complete
failed attempt-01-greedy training episode with at least 11 turns, a parseable
nonterminal ten-turn prefix, and no target instance observed through step 9.
The frozen manifest authorizes generation. The loader imports the runner's
transitions into `TrajectoryRecord`, preserving context-limit terminal status
and original usage counts as in `repair_collect`. It does **not** perform that
collector's full replay preflight: hints are CPU/public-record-only. Every
rollout later verifies its own real reset and prefix.

Only the task text from the public initial observation, that observation, ten
public replies/actions/observations, current admissible commands, and 10/40 step
counts enter the teacher envelope. Embedded thinking is stripped by the shared
`split_thinking` function. Task IDs, file paths, audit hashes, outcome labels,
future steps, teacher demonstrations, and E1 repairs never enter that envelope.
The public room inventory is parsed from the initial `you see a ...` list;
only legal `go to`, `open`, and `examine` prefix actions establish visited places.

`GuidanceTeacher` uses the same `teachers.capped.from_config` factory and
structured-response transport as `repair.runtime.make_teachers`, with its own
versioned envelope. It does not register a new repair stage or change
`OUTPUT_CAPS`. The cap is 8192 output tokens including reasoning. Every logical
call and every HTTP retry reserves a conservative UTF-8 input bound and the
output cap **before** transport. Known usage releases the completed reservation;
unknown failed attempts remain reserved. At most two format retries include the
validator reason. A transport failure halts further teacher work; bad labels or
budget stops leave C/D missing. No text is repaired by hand.

Artifacts include `hints.json`, `review.tsv`, per-case receipts, the append-only
budget/call ledgers, a separate `global_teacher_budget.jsonl`, provenance, and
`run_manifest.json` with template hash, returned model IDs/timestamps and usage.
The existing budget schema counts the initial guidance call under `diagnosis`
and later calls under `retry`; event details identify `condition_guidance`.
Unknown monetary cost remains unknown. Usage is not estimated from text.

Review every C and D for indirect location/action advice and unsupported
condition statuses. Copy `review.tsv` to a decisions file and fill `decision`
with exactly `approve` or `reject`; `note` can be empty. Extra review columns are
accepted. No empty decisions are accepted, including for missing C/D.

```bash
PYTHONPATH=src:. python -m tools.cg_probe approve \
  --hints /path/to/cg/hints/hints.json --decisions /path/to/decisions.tsv
```

This exclusively creates `hints_approval.json` beside the hints and binds it to
the exact SHA256 of `hints.json`. A rejection is retained. For the one permitted
human-review regeneration, rerun `hints` with that group's `--tasks` in a new
directory, retain both sets of evidence, and review again. The tool cannot infer
cross-directory regeneration lineage; the operator must retain/disclose it and
not exceed the preregistered one regeneration. Rejected or missing C/D cases run
A/B only and remain marked. Approval of valid C/D is required before their use.

## Roll out through the served theta0 adapter

The callable hook uses the same arguments passed by `repair_serve_stage.sh`:

```bash
repair_work() {
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$ENV_PYTHON" -m tools.cg_probe rollouts \
    --endpoint "$1" --server-provenance "$2" --lora-check "$3" \
    --model-name "$MODEL_NAME" --config "$CONFIG" --data-root "$ALFWORLD_DATA_ROOT" \
    --hints /path/to/cg/hints/hints.json --approval /path/to/cg/hints/hints_approval.json \
    --coverage-root /path/to/coverage --failures /path/to/coverage/failures.json \
    --workers 8 --output "$REPAIR_OUT/rollouts"
}
```

The caller sources the existing serve script after defining the hook. The probe
uses its passing LoRA report and server identity checks, verifies the configured
base/revision, and binds adapter weight/config hashes to the original coverage
adapter named `e4b-tsft-k20-e32-lr5e-5`. No new launcher is supplied.

For each enabled branch, run greedy temperature 0 and eight sampled continuations
with the unchanged `SampledStudent` (T=0.7, top_p=0.95, seeds 0–7). Every call uses
1024 max tokens and `enable_thinking=false`. Request seeds use the existing
`sha256([seed, task_id, turn])[:8]` rule, independent of branch. Guidance changes
only the observation of user turn 10, copied into later prompt histories. Stored
environment observations remain untouched; actual prompts are in student logs.

`execute_actions` resets and replays the ten original raw replies and compares
initial and prefix visible observations, commands, rewards, termination flags,
audit hashes, game hash and seed before continuation. Runner episodes lack the
repair executor's `environment_version` field, so the imported reference is
bound to the current adapter/dependency version before replay, as the collector
does when creating repair records. The original episode environment revision
and provenance remain retained. A mismatch voids that rollout and prevents the
student call. Policy context overflow remains a recorded failure; unknown
environment/transport execution errors are void, never false successes.

Each process writes an atomic record keyed by case/branch/decoding/seed. Use the
same command with `--resume` after interruption: exact input/config/code identity
is required, existing records are skipped, and old call logs are retained. A
crash before a result is atomically published requires a new real reset/replay;
its incomplete logs are preserved in a separate attempt directory. A per-key
lock prevents duplicate concurrent execution. `--limit-cases N` fixes a smoke
subset before running; changing it requires a new output directory.

`--dry-run` uses a fake teacher for hints or fake environment/policy for rollouts,
while still requiring coverage-shaped input files and explicit approval. It
marks all artifacts synthetic and cannot be mixed with real-mode hints. The fake
environment copies public prefix transitions with synthetic audit hashes, then
fails every continuation; it does not verify real ALFWorld replay.

## Summarize

```bash
PYTHONPATH=src:. python -m tools.cg_probe summarize \
  --rollouts /path/to/cg/rollouts --output /path/to/cg/summary
```

Writes `summary.json`, `summary.md`, and `per_case.tsv`. Sampled and greedy tables
are separate. Rates use equal case weights; missing/void cells remain unknown.
Paired differences use only cases with all eight valid shared seeds in both
branches and disclose exclusions. Cluster bootstrap resamples cases 10,000 times
with `numpy.random.default_rng(0)`. Found, picked and success use the preregistered
0.15 difference and net-two-case rule. The five requested comparisons are
reported, plus D−B to evaluate section 8's conclusion mapping. Incomplete,
subset, or synthetic results carry no mechanism conclusion.

Metrics use steps 10 onward. Finding requires the target instance regex in an
observation. Picking additionally requires a legal `take` and an observation
starting `You pick up`. Suggestion hit is null if never executed, otherwise true
if any execution of it directly revealed the target. Revisits count legal `go to`
actions returning to a previously visited receptacle whose earlier visit
observation showed no target (including earlier post-branch visits). This is the
predefined observation-based diagnostic, not a hidden-state claim that the
receptacle was exhaustively searched. First Thoughts remain verbatim public text;
no model is used to classify them. Diagnostic rows also compare the first action
with A under the same decoding/seed.

## Offline verification

```bash
CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests/test_cg_probe.py tests/test_repair_*.py
```

Tests cover public-field isolation, validators, two retries then missing,
pre-call/transport-retry caps, SHA-bound approval, actual HTTP request message
identity through a mock transport, later-turn persistence, real executor prefix
mismatch handling with fake environments, resume skips, metrics, deterministic
bootstrap, decision boundaries, and the full dry CLI through a process pool.
