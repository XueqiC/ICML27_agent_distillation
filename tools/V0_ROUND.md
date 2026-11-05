# Framework V0, one K20 / seed-0 round

`tools/v0_round.py` adds `synth`, `attempt`, `probe`, `review`, `label`,
`build`, and `expand`. All implementation files are new; no existing harness,
policy, training, factory, or teacher module is modified. The controlling
documents are the read-only main checkout's November 2 preregistration §§2,
2b, 2c and end-to-end framework, with the explicit implementation request
taking precedence.

The first subcommand creates a fresh `--output` round root. Subsequent
subcommands add fresh phase directories there. `--resume` requires the same
inputs, effective configuration, code hashes and dependencies, verifies sealed
completed phases, and fills unfinished work. Completed records are reused;
raw failed calls and environment executions remain in their ledgers. Unknown
teacher expenditure or an interrupted request without durable completion
stops execution. No silent fresh budget or retry reset is allowed. Use a fresh
round for a changed protocol/configuration. All policy-facing fields are
explicitly whitelisted; audit manifests are never formatted as prompts.
The seed fixes candidate order, allocation, item identities and student request
seeds. Live black-box teacher completions are provider outputs, not promised
bitwise-deterministic responses; their raw text, returned model ID and time are
retained. Synthesis has a two-hour process limit and each native validation has
a 120-second limit in a separate killable CPU process. A timeout is logged as
unknown feasibility with its partial transition journal. Candidate ordering is
seeded; near a wall-time limit, feasibility availability also depends on runtime
load. No timeout is represented as unsolvability or successful validation.

## CPU synthesis and collection commands

Run from this worktree. Set the read-only source and environment paths:

```bash
P=/home/xueqi/hq/projects/ICML27_agent_distillation
PY=$P/.venv/bin/python
OUT=$PWD/spike_out/v0_round_example
SEEDS=$P/data/verified/seeds_full_train_r2/subsets/k20/manifest.json
GAMES=/home/xueqi/hq/projects/tc-alignment/envs/alfworld/data/json_2.1.1
export PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1

CUDA_VISIBLE_DEVICES='' "$PY" -m tools.v0_round synth \
  --seeds "$SEEDS" --data-root "$GAMES" --main-checkout "$P" \
  --output "$OUT" --workers 4 --seed 0
```

The factory makes no teacher/student calls and runs the existing real
ALFWorld/TextWorld validator in offline CPU workers. Its manifest is
`$OUT/synth/manifest.json`; authorized games are under
`$OUT/synth/data_root/train/synth/<id>/game.tw-pddl`. Every game ID, including
a G3 recovery situation, is `train/synth/<id>`. The manifest's raw-byte SHA256
is recorded in `stats.json` and every collection/dataset row.

Per seed, the targets are 6 near, 4 mid, 2 far, 1 very_far training items,
plus one probe in each bin. A probe is reserved first. Near training slots
alternate G1/G3; after a generator's valid candidates are exhausted the other
fills the slot and the allocation receipt records the fallback. Finite spaces
and held-out exclusions can leave slots unfilled. There is no acceptance-rate
tuning, substitution of other official training tasks, or quota padding.

Mid G2 keeps the seed's task type and changes object/receptacle; far G2 changes
task type. G5 combines a cross-type G2 goal with relocation of that new goal's
object. This follows the explicit request: **no extra distractor object is
added**, although preregistration §2b's earlier prose also mentioned distractors.
All templates, affordances and placement evidence come from K20 seeds.
G5 passes the same loading, task-text, executed-witness, repeatability and
held-out checks as G1/G2. Diagnostic normalized action edit distance compares
the executed witness with the parent seed; G3 includes prefix plus recovery.
Student witness NLL is explicitly unavailable because synthesis makes no
model calls.

The native task-name exclusion freeze is inherited from the existing factory,
including the conservative exclusion of all valid_seen names. It reads
held-out identities only. Synthetic train/probe groups use effective
state/goal fingerprints (G3 uses the actual recovery-state digest); native
semantic groups remain in `fingerprint.group`. Rephrases, repeated refusal
states and alternative witnesses do not create fresh instances. G3s are
reported separately as recovery situations, not new native task instances.

With an already served, equivalence-gated theta0 adapter:

```bash
SERVING=(--endpoint "$ENDPOINT" --server-provenance "$SERVER_PROVENANCE"
         --lora-check "$LORA_CHECK" --model-name e4b-tsft-k20-e32-lr5e-5)
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.v0_round attempt \
  --output "$OUT" "${SERVING[@]}"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.v0_round probe \
  --output "$OUT" "${SERVING[@]}"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.v0_round probe \
  --output "$OUT" --report-only
```

Both use the existing `Student`/`SampledStudent`, `build_messages`, parser,
`Environment`, and reset/replay executor: greedy plus sampled seeds 0/1/2,
T=.7, top_p=.95, H=40. There is no task-catalog fallback. `--task-ids` must
be exact authorized IDs; probes require the probe path (`attempt --probes`
is equivalent). G3 prefixes contain the actual seed replies and injected
actions with `train.dataset.EMPTY_THOUGHT`. The shared visibility renderer
removes the empty channel marker before prompting, as elsewhere in the repo.
Prefix actions are executed and checked against the recorded visible prefix;
the continuation follows another measured reset/replay. All prefix turns
count toward H and are context-only in every dataset.
An explicit task subset is marked incomplete coverage and cannot seal a V0
training dataset. Expansion's declared new-item set is its complete scope.

Probe output includes per-bin and per-decoding failure rates and the upper end
of an equal-tailed exact Clopper–Pearson 95% interval. The aggregate bin uses
one observation per probe task (failure if any of its four attempts fails),
not four independent observations. Unknowns are excluded and separately
reported; an incomplete bin cannot meet the stopping criterion. These are
diagnostic synthetic probes, never inputs to review, expansion or training.

## Whole-trajectory review and labels

```bash
# No teacher, student, network, or environment construction:
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.v0_round review --output "$OUT" --dry-run

# Production commands below use the configured teacher; do not run for a CPU smoke.
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.v0_round review \
  --output "$OUT" "${SERVING[@]}"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.v0_round label --output "$OUT"
```

`review --dry-run` retains public requests and prints byte sizes, conservative
first-request input bounds, a six-call-per-failure projection, output-token
limits and the 60M input / 6M output round caps. Later request sizes depend on
actual executions, so that input projection is not a hard bound.

Review selects at most two failures per item, greedy first, then sampled seed
order. A strict JSON object carries subgoal evidence, preservable choices,
breaking-point categories and at most two patch blocks over the supplied
trajectory's original zero-based turn indices. There is one format retry.
Patches cannot change the G3 context-only prefix. Public fact annotations
reuse `TransformFactTable`/joint-review/2, including transform prerequisites,
observed receptacle contents and repeats. Neither witness actions nor
unobserved locations enter requests. Prior rounds expose only executed public
records. The task's final won/unknown outcome is explicitly review evidence,
never a hidden-state prompt field.

Each round compiles the patch, resets, executes it with the student's original
suffix, and checks the unchanged prefix against the measured trajectory. If
that execution does not win, it resets/replays to immediately after the last
edited block and hands control to the same adapter with the failed attempt's
decoding. There are at most three rounds per failure. The latest hand-back
execution becomes the next reviewed trajectory. Success requires harness won;
environment/transport errors halt with their evidence retained. Every turn
records student / patch / handback-student ownership, and repairs retain edit
distance, token use, all execution branches and charged environment steps.

`label` uses V2's public-prefix rationale prompt, 64-token label cap, grounding
checks and action-only fallbacks. Only patch-produced decisions are relabeled.
`--action-only` explicitly skips label calls; those rows retain the V2
`action_only_no_label` marker. Invalid completed rationale outputs become
marked action-only rows; transport/cap failures cannot become fallback labels.
Review and labels use the existing capped gpt-6-luna client/accounting.
`teacher_budget.jsonl` is shared by the round and both expansion variants.
`TEACHER_RATE_LIMIT_DIR` retains the repository's cluster-wide override.
Token usage is settled from provider metadata; unknown usage is never zero.

## Dataset and train-worktree interchange

```bash
# Reuse a sealed V2 CONTINUE bundle exactly, or omit --replay and supply the
# original V2 builder's --verified-root, --theta0-training-dir and --snapshot.
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.v0_round build \
  --output "$OUT" --replay "$V2_CONTINUE_DATASET"
```

The dataset is `$OUT/dataset/{manifest.json,examples.jsonl,replay/}`. Root rows
have exactly the V2 decision fields plus required `v0`, with arm `V0`:

```json
{"bucket":"R","episode_id":"repair-...","generator":"G5",
 "bin":"very_far","edit_decision":true,
 "synthetic_manifest_sha256":"<raw manifest SHA256>"}
```

R contains verified successful repairs; S selects at most one student success
per item, by fewest turns then episode ID. G is empty until expansion. Every
accepted final-trajectory assistant decision is supervised, except G3 context
turns, environment-refused actions and repeated refused actions. Prefix
messages remain context, including refused actions. Successful legal repeated
actions are retained under the explicit whole-trajectory instruction. A
malformed parser-error prefix cannot be represented by V2's strict message
contract; later rows are excluded with an `unparseable_prefix` audit reason.
Only the final assistant message of each decision has a supervised index.
`edit_decision` means a patch produced that decision; a deletion alone creates
no artificial edited decision. Counts include bucket, bin and generator.

D exists only in the unchanged sealed V2 CONTINUE replay bundle: no duplicate
D rows in root `examples.jsonl`. Its 20 executed seed trajectories, exact bytes
and original supervision masks remain intact. D's virtual V0 metadata uses
bucket D, generator/bin seed, false edit flag and empty synthetic hash.
Every data input, rejection, failed attempt, transition, label, raw teacher
output and receipt is sealed in manifest inputs; none becomes message text.
K, distinct instances/groups and training exposure are distinct counts.

**Compatibility issue found in the requested sibling contract:** the inspected
`ICML27_wt_repair_train/tools/V0_DATASET.md` version 1 requires *exactly five*
v0 keys and only G1/G2/G3. This request requires the sixth `bin` key and G5.
This producer implements the explicit newer request and preserves every other
disk/replay/message contract. The older train reader must accept `bin`, G5 and
the virtual D bin before it can consume these datasets. No sibling file is
modified, and compatibility with that older reader is not claimed.

## Expansion

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.v0_round expand --output "$OUT" \
  --dataset "$OUT/dataset" --variant D-data \
  --groups "$GRAD_GROUPS/groups.json" --weights "$GRAD_GROUPS/weights.json" \
  --seeds "$SEEDS" --data-root "$GAMES" --main-checkout "$P" "${SERVING[@]}"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.v0_round expand --output "$OUT" \
  --dataset "$OUT/dataset" --variant D-param \
  --seeds "$SEEDS" --data-root "$GAMES" --main-checkout "$P" "${SERVING[@]}"
```

The target count equals the number of initial training items with at least one
known failed attempt. D-data verifies grouping/weight dataset and adapter
identities and allocates by group consistency. Groups without a repaired
episode are explicitly reported as ineligible. D-param divides equally over
observed breaking-point categories with repaired representatives. Largest
remainder allocation preserves the exact total; ties are lexical. Negative or
zero total allocation mass fails loudly rather than being clipped.

Each unit selects a repaired episode and applies its generator/bin/edit kind
to an original K20 seed, preferring that episode's seed then a deterministic
ordering of other K20 seeds. G2/G5 keep goal task type, G1 keeps destination
receptacle type and G3 keeps error type. Fresh fingerprints exclude all initial
training items and probes. Finite-space exhaustion writes the complete negative
accounting and fails; it never pads with duplicates or synthetic parents.
The generated items follow the same attempt/review/label pipeline, then append
G rows to **`$OUT/expand-D-data/dataset` or `expand-D-param/dataset`**.
The input dataset is reverified and never modified. The new combined synthetic
manifest includes both generations; inherited rows are rebound to its raw hash
in the new dataset only. Replay bytes remain identical.

## Rai launcher example

[`v0_round_rai.sh`](v0_round_rai.sh) follows the read-only main checkout's
`reports/runs/method_v1/run_v2_eval_rai.sh`: it sources `repair_stage_env.sh`,
defines `repair_work(endpoint, provenance, lora_check)`, then sources
`repair_serve_stage.sh`. The latter performs the existing HF/vLLM adapter gate
and passes the actual ready endpoint and gate artifacts. Its `repair_work`
process has CUDA hidden; the server follows the caller's GPU allocation.
Required cache exports and model/env paths are the same as `rai.env`.

```bash
export V0_ROUND_OUT="$OUT" ADAPTER=/path/to/theta0/checkpoint GPU=1
V0_STAGE=attempt bash tools/v0_round_rai.sh
V0_STAGE=probe bash tools/v0_round_rai.sh
V0_STAGE=review bash tools/v0_round_rai.sh
V0_STAGE=expand V0_VARIANT=D-data V0_DATASET="$OUT/dataset" \
  V0_GROUPS="$GRAD_GROUPS" bash tools/v0_round_rai.sh
```

These are production examples, not commands run during this implementation.
The task's real-environment dry run is synthesis only and stops before any
teacher or student client/model call. Unit tests use existing fake environments,
`FakeStudent`, and fake transport clients; they establish software behavior,
not a scientific finding or a faithful baseline reproduction.

To probe another adapter without writing to the existing round, use a fresh
`V0_ATTEMPT_DIR`. The launcher exclusively creates that directory and writes
serving/gate evidence under `serve/` and the probe phase under `probe/`:

```bash
cd /home/xueqi/hq/projects/ICML27_wt_repair_eval
V0_STAGE=probe V0_ROUND_OUT=/absolute/path/to/existing-round \
V0_ATTEMPT_DIR=/absolute/path/to/new-probe-dir \
ADAPTER=/absolute/path/to/adapter/checkpoint GPU=3 WORKERS=8 \
TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 \
  bash tools/v0_round_rai.sh
```

This uses the usual rai cache exports and `HPG_CONFIG`/`rai.env` settings.
`V0_ATTEMPT_DIR` must not exist, even as an empty directory; `V0_RESUME=1` is
incompatible. The direct CLI equivalent for an already gated server is
`probe --output ROUND --attempt-dir DIR` plus the existing endpoint, model-name,
server-provenance and lora-check arguments. The direct CLI puts the phase files,
attempt records and `summary.json` directly in `DIR`, exclusively. It never
creates or acquires the input round's lock. Both phase and summary bind the
adapter's resolved path and `adapter_model.safetensors` SHA-256. With no override,
the default remains `ROUND/probe_attempt`.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /home/xueqi/hq/projects/ICML27_agent_distillation/.venv/bin/python \
  -m pytest tests/test_v0_round.py tests/test_v0_pipeline.py -q
```
