# Method v1 splits and E0.3 joint guidance

This implementation adds a split freeze and a TRAIN-side diagnostic runner. It
produces execution evidence, never training examples or updates. Successful
assisted episodes do not establish independent student learning. Synthetic runs
and unit tests establish no scientific result.

## Freeze the two sets

From the repository root, with `PYTHONPATH=src:.`:

```bash
CUDA_VISIBLE_DEVICES='' python -m tools.method_v1_splits \
  --verified-root /path/to/ICML27_agent_distillation/data/verified/seeds_full_train_r2 \
  --output data/manifest/method_v1_splits.json
```

The default verified root is `data/verified/seeds_full_train_r2`. This worktree
has no such directory; the delivered freeze reads the existing collection in the
sibling `ICML27_agent_distillation` checkout. All reads and exclusions are bound
to recorded paths and SHA256 hashes. There are no new teacher calls, environment
steps, or training exposures. Existing manifests and verified collections are
never written. The output uses **exclusive create**, even for identical content.

The builder imports `data.manifest.build_repair_splits`, checks its recorded
builder hash, reproduces the original frozen result using its original sources,
and uses its frozen-input validation, exclusions, hash ordering, one-trial-per-
group selection and listing functions. TRAIN source and assignment are both
required; dev140, sealed and other evaluation groups remain excluded. Original
repair eligibility/exclusions remain in force. It additionally excludes every
group in D_probe, D_unseen_quick, U_mech and all six full-TRAIN subsets
k10/k20/k30/k100/k300/k500. Both selection and collection manifest identities are
included, including any named attempt/replacement groups.

U_dev requests ten groups per type with seed 20261101; its purpose is
`diagnosis_only`, `training_eligible=false`. V2_pool requests 200 groups with seed
20261102, purpose `method_v1_training_pool`, `training_eligible=true`. After U_dev
selection, V2 eligibility additionally requires a verified won teacher trial.
Hamilton/largest-remainder quotas use those eligible **group** counts, breaking
equal remainders by sorted task type. Hash ordering is ascending SHA256 of
canonical `[seed, task_id]`, with task ID tie break, exactly as repair_v01. V2
selects among verified trials so its recorded trajectory is for the **same trial**,
not an interchangeable sibling. Multiple verified trajectories for one trial
break ties by trajectory ID then path. Episode/event/provenance hashes and
complete, successful TRAIN execution are checked against collection receipts.

Shortfalls are explicit and never silently redistributed. The delivered inputs
leave only eight eligible `look_at_obj_in_light` groups: U_dev contains **58**
groups, with a shortfall of two in that type. V2_pool contains **200** groups;
all eight remaining light-task groups went to U_dev, so its light-task quota is
zero. These are the frozen constraints' consequences, not a completed 60-group
balanced diagnostic pool. Counts, quotas, both set list hashes, input hashes,
exclusion sources, trajectory IDs/paths/hashes and overlap assertions are in the
new manifest. K still counts trusted executed seed trajectories, not these task
groups.

## Inputs and modes

Use the exact prefix spec documented in [TAKEOVER_FACTS.md](TAKEOVER_FACTS.md),
created with `tools.teacher_takeover spec-from-cg`. The same unchanged loader
checks coverage hashes, greedy origin, task authorization, theta0 provenance,
seed, horizon and nonterminal prefix. Currently its source adapter accepts the
seven CG groups and their coverage-shaped evidence; accepting arbitrary new
U_dev coverage requires a separate source adapter. The new split freeze itself
does not claim that adapter exists.

`--from-start` still takes `--spec` to identify the cases and coverage adapter,
but uses branch index zero. Both starts reset the real environment and invoke
`execute_actions` with exact raw-prefix replay and a measured prefix reference.
Any observation/audit/reward/terminal divergence voids the episode before actor
calls. H remains 40, including the replayed prefix. No branch is restored from a
fabricated snapshot. Only normal reset/step reads environment inputs.

- U: the unchanged student ReAct prompt; R is never constructed or called.
- G: propose, review, optionally regenerate/demonstrate, execute, observe; repeat.
- G1: G until its first applied guide or demonstrate, then autonomous decisions.
  Its prior guidance remains in history, as in CG. Invalid reviews becoming
  accept do not consume the intervention. G/G1 use the same code and prompt;
  teacher calls are independent samples, not a claim of identical sampled text.
- T remains the existing teacher takeover runner.

Each case/mode runs one greedy rollout and the sampled seeds supplied by
`--seeds` (default 0–7). The served policies are imported directly from CG:
`Student` or `SampledStudent`, T=0.7/top_p=0.95 for sampled runs. Request seeds
use `sha256([sampling_seed, task_id, zero_based_turn])[:8]`; a regeneration keeps
the same turn and seed. Paired modes use the same seed list.

## Review and execution boundary

`joint-review/1` is an independent structured envelope using the same capped
factory and structured teacher transport as `repair.guidance`, with gpt-6-luna
and max output 4096 including reasoning. Existing templates are untouched.

R sees only task text extracted from the initial observation, public history
(reply/action/resulting observation), current observation and the same admissible
commands shown to the student, `FactTable.prompt_text()`, the unexecuted proposal,
and steps used/limit. The shared Gemma visibility helper strips hidden thinking;
visible `Thought:` prose remains. Task paths, game files, rewards, success labels,
audit dictionaries and future transitions are never prompt fields. R does not
receive private states or other branches' discoveries.

The JSON schema has exactly `decision`, `facts_cited`, `need`, `action`, `check`,
`guidance`. `facts_cited` entries have `key` (dot path, e.g.
`conditions.hands_free`) and `evidence_steps` (nonempty subset of that fact's
recorded evidence). Guide/demonstrate require citations, need, one exact admissible
command, check and guidance. Guidance is at most 60 whitespace-delimited words;
check is at most 30. A completed invalid response gets one format retry with the
validator reason. Two completed schema failures become accept with
`review_invalid`. Transient transport failures receive the bounded retries below;
exhausted retries, quota failures, other teacher failures, empty/incomplete
responses, unknown usage and budget stops void the episode and halt further reviews.

Guide copies the public history/current observation and appends exactly
`\n\nGuidance: ` plus guidance to the current observation. Later student prompts
reapply that suffix only to the corresponding historical observation. Original
environment records stay untouched. Regeneration happens once. A different
admissible command is accepted unless it repeats a previously refused action or
contradicts a cited known-unsatisfied/refuted prerequisite. Deterministic checks
cover take with hands occupied, target manipulation without target possession,
take after refuted target location, and target placement before a cited required
state. Unknown conditions are never converted to negative facts. Unsatisfied
counts do not veto constructive take/placement actions. This is a conservative
command prerequisite check, not a semantic proof of arbitrary prose.

Fallback or direct demonstrate executes the review's command with
`Thought: <single-line check, at most 30 words>\nAction: <command>` and executor
`teacher`. Refusal means inadmissible or public `Nothing happens.`; repeat
tracking includes the replayed prefix. Teacher claims and schema validity are
not environment truth. Success is solely the real executor's won result.

## Run, resume and serve

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m tools.joint_guidance run \
  --spec /path/to/prefix-spec.json --config configs/base_e4b.yaml \
  --endpoint "$endpoint" --server-provenance /path/to/server/provenance.json \
  --lora-check /path/to/check/report.json --model-name e4b-tsft-k20-e32-lr5e-5 \
  --data-root /path/to/alfworld --modes U G G1 --seeds 0 1 2 3 4 5 6 7 \
  --workers 4 --budget-ledger /path/to/shared/joint-budget.jsonl \
  --max-teacher-input-tokens 5000000 --max-teacher-output-tokens 500000 \
  --output /path/to/joint-run
```

The serving check is CG's unchanged `serving_spec`: endpoint/provenance/LoRA
report must agree, the adapter must be theta0 and its base/revision/weights/config
must match every coverage adapter. No checkpoint substitution is allowed.
The existing serving shell hook needs no modification:

```bash
repair_work() {
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$ENV_PYTHON" -m tools.joint_guidance run \
    --config "$CONFIG" --spec "$JOINT_SPEC" --data-root "$DATA_ROOT" \
    --endpoint "$1" --server-provenance "$2" --lora-check "$3" \
    --model-name "$MODEL_NAME" --workers 4 --budget-ledger "$JOINT_BUDGET" \
    --output "$REPAIR_OUT/joint"
}
source scripts/repair_serve_stage.sh
```

Set the existing script's required CONFIG, MODEL_NAME, ADAPTER, VERIFIED,
MODEL_PATH, REPAIR_OUT and Python variables, plus JOINT_SPEC, DATA_ROOT and
JOINT_BUDGET, before sourcing. This documentation does not launch serving.

`SharedTeacherBudget` uses the explicitly supplied ledger across all workers and
separate CLI invocations. Caps must match when reopening it. Every HTTP attempt,
including transport retries, repair retries and the format retry, reserves before transport.
Known usage settles idempotently; unknown/interrupted attempts keep their full
bounds. The capped factory retains its 20-attempt rate-limit policy. After that
transport returns a failed call, the reviewer classifies its sanitized receipt
or exception as `TRANSIENT` (HTTP 5xx including 500/502/503/504, 408, 409, exhausted rate-limit
429, connection or timeout error), `QUOTA` (the existing quota guard's 429
classification), or `OTHER`. Quota and other failures halt immediately; this
includes authentication/schema HTTP errors, empty output, truncation and cache
hits. Completed review-schema failures retain the separate format retry above.

Each transient logical request gets at most six repair retries with identical
messages, fresh request IDs and fresh transport reservations. Base waits are
5, 10, 20, 40, 80 and 160 seconds plus uniform jitter of 0–10% per wait: total
added wait is at most 346.5 seconds (under six minutes). This bound excludes
the unchanged transport's own retry/rate-limiter waits and request duration.
Budget checks run before each retry and every second during backoff; another
worker's global stop ends the loop. A successful retry does not halt the budget.

The local capped counter otherwise latches terminal failures. Only a fully
receipted transient failure from the current live request is acknowledged using
its existing hashed recovery boundary. `recoveries/<request_id>.json` retains
the prior counter and ledger hash; no reservation or consumed call slot is
refunded. Historical halts, interrupted calls and global stops are not recovered.
The worker cap remains 80 call slots, now including repair retries, alongside the
shared token caps. The limiter is shared beside the budget ledger (or through
the existing TEACHER_RATE_LIMIT_DIR override).

Resume with the same command plus `--resume`. Code/content/config/dependencies,
case list, modes, seeds, start point and ledger/caps are identity-bound; worker
count may change. Per-key locks and atomic immutable records prevent duplicate
execution. New attempts after an interruption use fresh caches while preserving
old reservations and logs. Durable usage receipts reconcile before resume. A
stopped ledger is never reset or replenished. U requires no review and can still
finish if the teacher budget has stopped; the invocation reports an incomplete
nonzero result. No raw SDK exception text is printed or persisted.

## Outputs and summary

Each `records/<key>.json` includes the full execution TrajectoryRecord, prefix
verification, success or unknown/void reason, and per-decision proposal, facts
hashes, review IDs/parsed response/validation/attempts, decision, separate guidance,
regenerated reply/action, final executed reply/action/executor, resulting public
observation and refusal/repeat flags. Failed review rows retain their evidence
and have no executed reply. `guidance_injections` identifies only appended
student-prompt suffixes; demonstration guidance is also retained in its step.
Usage includes retries and independent model ID/timestamp receipts. Capped
transport bookkeeping rows are not counted as unknown API attempts.
`reviews.jsonl` records each fresh request, its logical request ID, zero-based
`attempt_index`, `failure_class`, retry decision and delay; `retry` still denotes
the separate format retry. Usage rows carry `repair_request_id`,
`repair_logical_request_id`, `repair_attempt_index` and `failure_class` alongside
the native transport IDs/index. `transport_will_retry` preserves the transport's
original decision, `repair_will_retry` records the added retry, and `will_retry`
covers either so resume reconciliation accepts recovered failures. Append-only
`repair_outcome` rows retain policy-level rejections without duplicating usage.

`plan.json`, `provenance.json` and `run_manifest.json` bind effective config,
code/input hashes, dependencies, student revision/serving identity, teacher alias,
source evidence, environment seed/version, frozen manifest and split. Call caches,
review requests/raw responses/rejections, student call logs, usage and evaluation
interaction ledgers remain under `calls/<key>/<attempt>/`. Training exposure is
zero. Completion records include the shared budget, which may also include other
invocations; per-run all-attempt usage is reported separately.

```bash
PYTHONPATH=src:. python -m tools.joint_guidance summarize \
  --rollouts /path/to/joint-run --output /path/to/joint-summary
```

Summary JSON/Markdown contains per-case/mode and per-mode success/picked/found
rates, teacher-step share, follow rate, per-case paired differences and G−U,
G1−U, G−G1 case-cluster bootstrap intervals (10,000 resamples, NumPy seed 0).
Greedy and sampled are separate. Primary aggregate rates average per-case rates.
Aggregate teacher share and follow rate pool executed turns and guided attempts,
respectively; their numerators and denominators are reported explicitly.
Void/missing cells stay unknown and are excluded as entire paired cases, with
exclusions disclosed. Follow rate is undefined with no guided turns; the teacher
share denominator is executed suffix policy turns, including parser-error turns.
A guide falling back to teacher counts as a guide attempt and a demonstration,
not a student follow. Discovery times include explicit not-found counts; matched
turn action differences from U are descriptive because later states can diverge.
No significance threshold or mechanism conclusion is invented.

## Offline checks

`--dry-run` takes a synthetic coverage-shaped spec, uses scripted student replies,
a recorded-prefix fake environment and a deterministic fake reviewer through the
real capped transport/accounting interfaces. The reviewer guides with the first
admissible `go to` every third (one-based) policy turn and demonstrates on a
proposal repeating a refused action. Synthetic suffixes never win. Its separate
synthetic limiter permits rapid tests; no model, API, network or GPU is used.

```bash
CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q \
  tests/test_method_v1_splits.py tests/test_joint.py tests/test_takeover.py \
  tests/test_facts.py tests/test_cg_probe.py
```

## Opt-in transform preconditions (V2 preregistration §8)

`tools.joint_guidance run --review-template joint-review/2` selects the separate
transform-precondition experiment. Use a new output directory. The default is
`joint-review/1`; the V2 collection specified in the preregistration continues
using that default. This implementation does not launch the nine-heat-prefix
comparison or establish its adoption criterion.

`repair/joint_transform.py` adds a `TransformFactTable` used only with the new
template. It reads the same public observation/action/outcome whitelist as V1,
over the complete reset-to-current prefix (including replay). New conditions are:

- `required_transform`: parses the existing public goal's required state into
  hot → heat/microwave, cool → cool/fridge, clean → clean/sinkbasin, citing -1.
- `transform_done`: reuses the successful execution evidence in
  `required_state_done`. With no successful required transform in the complete
  prefix, it is `known_unsatisfied` and cites that prefix. This means the required
  execution has not been witnessed; it does **not** claim an unobserved initial
  temperature or cleanliness. V1's `required_state_done` remains undetermined
  until supported. Heating/cooling supersede each other, as in V1. An attempted
  transform whose outcome is missing remains undetermined until resolved by a
  subsequent executed transform.
- `at_required_appliance`: uses the latest legal, non-refused `go to` or explicit
  public arrival/location statement. An instance listed in the room, examined,
  opened or closed does not by itself establish arrival. Evidence includes the
  location step and the goal.
- `target_inside_appliance`: requires a legal, non-refused move/put into the
  required appliance type, a still-supported last object location there, and
  known non-possession. A take, subsequent move, or contradictory observation
  updates/invalidates this evidence. A sighting without a recorded placement
  does not assert this condition.
- `transform_ready`: target held AND at the required appliance AND the required
  transform not yet executed. All three must be supported to mark it satisfied;
  a known blocker makes it unsatisfied, otherwise it stays undetermined.

Unparsed goals and non-transform goals leave the added conditions undetermined.
Conditions keep the existing evidence-step format and enter the same explicit
prompt whitelist. No admissible-command lists, paths, hidden state, success
labels, teacher claims or future observations supply these facts.

`joint-review/2` is the exact V1 system template followed by one general
environment-rule paragraph: command forms, hold + arrival prerequisites, no
transform from placement/open/close, take an inserted target first, and use an
admissible transform next when ready. The JSON schema, citation validator,
regeneration/fallback checks, student prompts and execution loop are shared.
Readiness does not execute or force an action in the harness; the reviewer still
selects an admissible command and real execution determines its result.

Plan/manifests, review envelopes, transport template hashes and retries bind the
selected version. V2 episode records additionally store `template_version`, even
for U, void and unrun episodes. V1 retains its original episode/step schema: its
version is already in each review request and the run plan/manifest (U has no
review). Cross-version resume fails before any worker runs. Code/content hashes
continue to identify the actual checkout; they are not spoofed to the old HEAD,
and the existing code-identity resume restriction still applies to older runs.

`tests/fixtures/joint_v1b_step.json` is a small, attributed public prefix from
V1b at 7748da5, with source record/review-log hashes and a verified teacher prompt
blob reference. It is compatibility evidence only. Tests rebuild the V1 prompt,
facts, envelope and full decision-step record and compare their original hashes,
both with the omitted option and explicit `joint-review/1`. They require no
sibling checkout, model, API or network at test time.

```bash
CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q \
  tests/test_joint*.py tests/test_facts.py tests/test_takeover.py tests/test_supervision.py
```
