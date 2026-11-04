# P0 pilot tooling

Implements the train-side white-box diagnosis pilot in
`ICML27_agent_distillation/docs/direction/2026-11-04_P0_whitebox_diagnosis_prereg_ZH.md`.
This is tooling, not a scientific result. No post-training or checkpoint export
is performed. Existing modules are imported without edits.

## Files and reused contracts

| New file | Responsibility |
|---|---|
| `repair/p0_pairs.py` | Read-only R-bucket extraction, first-divergence audit, instance-group split freeze, native prompt spans, S-decision sampling |
| `repair/p0_introspect.py` | Local merged bf16 student, nine JSON tool endpoints, content caches, reversible weight edits, checked harness rollouts |
| `repair/p0_agent.py` | Strict teacher protocol, diagnosis capabilities, budgets, existing V0 transport/settlement, submission freeze |
| `repair/p0_eval.py` | Teacher-free conditions, collateral/K20 evaluation, exact paired statistics, prediction scoring, JSON/Markdown reports |
| `tools/p0_pilot.py` | `extract`, `smoke`, `diagnose`, `evaluate`; scripted fake Responses client |
| `tests/p0_fakes.py`, `tests/test_p0_*.py` | CPU/offline native tiny Gemma4, fake env/teacher, real-record extraction tests |

The code reuses `TrajectoryRecord.decision_context`, `sed.react.build_messages`,
the base tokenizer's native OFF chat template, `sed.react.action_outcome`,
`sed.thinking.split_thinking`, `repair.execute.execute_actions`, and
`SyntheticManifest.environment` / the existing ALFWorld `Environment`.
Extractor checks relevant harness source hashes against the round's plan.
The original H=40 includes the replay prefix, parser-error turns, and the
continuation. Replays check visible observations, audit hashes, environment
version, seed, game hash, rewards, and terminal flags before continuation.

The only source records used are the supplied round's review/attempt records,
round-0 reviews, plan, and synthetic manifest. Only authorized train assets are
opened. K20 evaluation resolves exactly the 20 `seed_sources` in that manifest.
It does not load a general task/evaluation manifest or discover split folders.
No frozen evaluation assets or probe attempt folders are used.

## P0 pair freeze v3 (PI correction, 2026-11-04)

`reports/p0_tooling/extraction_v3/pairs.json` uses schema `p0-pairs/3`.
The v1 `extraction/` and v2 `extraction_v2/` directories remain untouched.
The unit is the first parsed-action divergence between the original trajectory
and `TrajectoryRecord.from_json(review_row['execution'])`, the final verified
execution. Both trajectories must have an action at t*. Eligibility requires
`executors[t*] == 'patch'`, t* at or after the attempt's context-only prefix,
and identical initial visible fields and `steps[i]['visible']` for every i < t*.
The state remains exactly the original `decision_context(t*)`; a_f and a_r are
the original and final execution's parsed actions at t*. Review-round count,
executor at t*, and the compiled round-0 action at t* are audit metadata only.
The round-0 action never supplies a_r.

Category resolution is unchanged: the first round-0 breaking point at t*, or
the closest step by absolute distance with ties going to the later step.
Resolved `other` is excluded. The teacher sees only the selected breaking
point's `step`, `category`, `reason`, and `decision_step=t*`; audit/patch rows
remain outside the prompt.

Split is unchanged: sort unique eligible task-instance `group_id` values,
permute using `random.Random(0).sample(groups, len(groups))`, and visit sorted
`CATEGORIES`. Until a category has at least four diagnosis pairs, select the
first unchosen group containing it. All eligible pairs from chosen groups
enter diagnosis; the remainder enter test. All load/diagnosis gates require
v3 and `alignment=final_execution_first_divergence`. V1/v2 are refused before
model or teacher calls.

| Resolved category | Eligible | Diagnosis | Test |
|---|---:|---:|---:|
| goal_misread | 27 | 4 | 23 |
| missing_precondition | 37 | 5 | 32 |
| refused_action | 19 | 5 | 14 |
| search_incomplete_or_loop | 12 | 4 | 8 |
| wrong_object_or_target | 18 | 5 | 13 |
| Total | 113 | 23 | 90 |

The PI's independent counts are reproduced exactly: 134 verified repairs,
113 eligible, 18 primary `divergence_not_teacher_patch` exclusions and 3
primary `category_other` exclusions. There are zero prefix or visible-state
mismatches. One student divergence also resolves to `other`, so the exhaustive
reason lists contain four category exclusions while the primary counts contain
three. This overlap does not change the eligible set.

| Category source | Eligible | Diagnosis | Test |
|---|---:|---:|---:|
| at_divergence | 69 | 13 | 56 |
| nearest_following | 35 | 5 | 30 |
| nearest_preceding | 9 | 5 | 4 |

The 113 pairs cover 74 unique instances/groups and 107 distinct rendered states.
Diagnosis covers 14 instances/groups and 21 states; test covers 60 and 86.
Instance, group and state overlap are zero. Nine parent seed task IDs are shared
between splits; 23 diagnosis and 35 test pairs descend from those shared parents.
K remains 20 trusted executed seed trajectories. Source attempts include 43
greedy and 70 sampled trajectories.

The example `55601f73e17b9863b9fdcd4683e327b4` illustrates that v3 can also move
the divergence earlier: t*=13, a_f=`go to bed 2`, a_r=`go to shelf 1`, with
round-0 audit action `examine bed 2`, across two review rounds. The action
mismatch cited for its old v2 unit was at a different index.

Greedy reproduction stays `not_run` at extraction. Diagnosis reproduction
never filters diagnosis. Test reproduction is measured only after all five
submissions are frozen. Decision flip rate uses only test pairs whose theta0
free greedy action reproduces a_f; reports preserve reproduction unknowns,
mismatches and the exact denominator. Zero reproductions yield unknown flip
rate. Primary rollout success and paired tests use all test pairs.

## Frozen implementation choices

The v3 final-execution unit, post-norm projection and Thought framing supersede
the earlier P0 implementation. The underlying harness and split rule are unchanged.

* Layers are **zero-based**. Residuals are block outputs after Gemma's per-layer
  input/scalar operations and before the final norm. The fixed control uses
  layers 20 through 41 inclusive. No ablation changes the original harness.
* Every decision-prefix tool (`action_scores`, `logit_lens`, `attention_summary`,
  `patch`, `hidden_mean`/`direction`/`projection`, `apply_edit`, and
  `counterfactual_context`) accepts `framing`: `original_thought` (default),
  `greedy_thought`, or `no_thought`. The suffix stays
  `Action: <command>` plus turn-end 106, including wrapper and terminator in
  the joint sequence score. It is not token-average NLL or a marginal over Thoughts.
* `original_thought` appends the exact original step's public reply prefix,
  split at the first `\nAction:`, with the newline retained before the candidate.
  No whitespace is normalized. Receipts retain both exact split parts, separator,
  character index, raw-text hash, requested/effective framing and fallback reason.
  A missing Thought or delimiter falls back to `no_thought`. R pairs and the
  200 frozen S collateral decisions carry their own `original_action_raw`;
  198 S decisions have Thought prefixes and two explicitly fall back.
* `greedy_thought` generates freely from the state under theta0 plus the active
  edit, cuts at the first `\nAction:`, and records the generated reply and split.
  Generation is cached per state, active edit, student identity and generation
  settings. Missing framing text/delimiter is recorded as a `no_thought` fallback.
  All tool results and cache keys record framing; default and explicit
  `original_thought` share a cache key. Different recorded Thoughts at the same
  visible state retain separate capabilities. Direction cache files are keyed
  by construction call, so identical vectors from different framings cannot
  overwrite each other's construction receipts.
* The appended prefix is labelled `current_thought`, including `Thought:` and
  its trailing newline. Attention reports its token mass. A counterfactual
  `replace` can rewrite it, and `delete` can remove it; the override and parent
  split remain recorded. Free greedy reproduction, flip-rate decisions,
  collateral, and rollouts always start from the state without a forced current
  Thought. Thought-only counterfactuals therefore affect conditional probes,
  while free rollouts continue from the original state text.
* Logit lens forces the common candidate-token prefix, then compares the first
  differing token after each block, using native final norm, unembedding, and
  logit softcap. Identical actions explicitly return no differing token.
* Direction vectors average residuals **at the teacher-forced command-token
  positions**, excluding `Action: ` and turn-end, first within each action and
  then equally across examples. The two group means are subtracted and each
  layer's vector is unit normalized. Zero vectors are rejected. Projection
  reports dot product and cosine on the same action-token means.
* Attention query is the final rendered prompt position. Hooks capture native
  q_norm/k_norm outputs; only one Q's RoPE, dot products, masks and softmax are
  reconstructed. KV-shared layers use actual donor keys. The main forward stays
  SDPA; it does not request eager full-sequence attention. Native sliding-window
  masks, GQA expansion and scaling are retained. Output gives per-layer span
  means/maxima over heads. Goal spans overlap observation spans, so totals across
  labels are not additive. JSON escaping is included in the native span map.
* Counterfactual edits operate on labelled rendered-text spans. They create a
  content-addressed state usable by all tools. Reordering requires disjoint
  spans; partially intersected annotations are removed. Reordering across the
  state/current-reply boundary is refused. A counterfactual
  rollout still replays the real original prefix; edited text changes only its
  first decision. Subsequent turns return to the ordinary visible-history
  renderer. This is not a claim of changed/restored physical state.
* Activation patches replace block-output prompt positions (explicit
  source/target token indices; negative indices count from prompt end).
  Candidate scoring sees the patch; continuation activations are not implicitly
  patched. Identity patches have zero score change in the tiny-model tests.
* Weight edits target `mlp.down_proj` and/or `self_attn.o_proj`. Directions d
  remain unit vectors in residual space. For each target, use the following
  native Gemma4RMSNorm weight g (`post_feedforward_layernorm.weight` or
  `post_attention_layernorm.weight`, respectively), without a +1 offset:
  `u=normalize(g*d)` and `W <- W-alpha*u*u.T*W`. Thus at alpha=1,
  `d.T @ postnorm(W' x)` vanishes up to fp32/bf16 rounding. Alpha is any finite
  scalar; alpha=0 skips the copy to preserve every byte, including signed zeros.
  Layers with either `post_feedforward_layernorm_1/_2`, an active MoE branch,
  or missing/non-native expected norms are refused with ValueError before edits.
  Arithmetic and normalization use fp32, followed by a cast to weight dtype.
  Receipts record each post-norm module name, ||g*d||, cos(u,d), the exact formula,
  targeted matrix elements/fraction and changed elements after rounding.
  Original CPU backups are restored byte-for-byte in `finally`, even on errors.
  Random, fixed-rule and teacher controls all call this same edit function.
* Original serving used vLLM; this pilot requires HF generate with in-memory
  merged LoRA. Native rendering, parser, visibility and stop IDs `[1,106,50]`
  are reused, but exact greedy agreement across inference kernels is measured,
  not assumed. Sampling is disabled, seeds fixed, deterministic algorithms
  enabled, and dtype is bf16. Full contexts are never truncated.
* Every valid, rejected, or cached tool invocation consumes one of 60 slots.
  Each rollout request consumes one of 10 slots, including a cached request.
  Up to five distinct edit specs may be explored, including edits used only in
  a rollout or in the final submission. The 400-real-step ceiling includes all
  prefix replay/continuation steps; cached results add no real steps. Parser
  errors consume horizon turns, as in the harness, but not environment steps.
* Numeric predictions use the registered metric vocabulary and `>`, `>=`, `<`,
  `<=`; values are fractions, differences of fractions, or exact p-values.
  Unsupported prediction metrics are rejected rather than judged after seeing
  outcomes. Missing verification yields unknown predictions, not a false label.
* Random control directions are independent normalized CPU Gaussian draws with
  seeds 0–4, matching teacher layers, targets and alpha. The fixed control uses
  diagnosis fail-minus-repair action-token means with `original_thought` framing,
  layers 20–41, teacher alpha
  and the teacher's selected projection target types.
* McNemar is two-sided exact, pooled and per category. A five-seed mean is not
  binary, so five separate paired tests are reported. The random point estimate
  is their mean; the conservative gate uses the maximum p and requires teacher
  superiority for every seed. It never treats the seeds as 5n independent pairs.
  This is stricter than testing only the mean and is an explicit resolution of
  an ambiguity in the draft. Per-category p-values are unadjusted/descriptive.
* S sampling is 200 individual non-prefix student decisions from successful
  attempts, sorted by attempt ID/step and sampled without replacement at seed
  zero. Every category-specific edit is evaluated on that same draw and on the
  original K20. Pooled collateral gate requires every category's teacher edit
  to be at most 5%; every category's K20 rate must not decline. There is no
  invented deploy-time classifier that knows a new failure's category.
* Multiple pairs from one instance violate a naive independent-task interpretation
  of paired tests even with instance-disjoint splits. Reports expose counts and use the field
  `preregistered_numeric_criterion`, not an automatic scientific-finding claim.

## Commands

Run from this worktree. All outputs below are new directories. Frozen artifacts
are exclusive writes; the agent/evaluator refuses restart that would reset
budgets. Tool caches can restore content-bound directions/states. V0 transport
retains its own durable request/usage receipts. A crashed agent is not silently
resumed with a fresh tool budget.

```bash
PY=/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" tools/p0_pilot.py extract --output reports/p0_tooling/extraction_v3

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -m pytest -q tests/test_p0_pairs.py tests/test_p0_introspect.py \
  tests/test_p0_agent.py tests/test_p0_eval.py
```

Future smoke command, **on rai GPU 1**. This v3 correction task authorizes no GPU,
teacher/API or network execution; the command below was not run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python \
  tools/p0_pilot.py smoke \
  --pairs reports/p0_tooling/extraction_v3/pairs.json \
  --output reports/p0_tooling/gpu_smoke_v3
```

The smoke selects the first two lexically sorted, aligned **diagnosis** IDs
before looking at any model output. It executes all nine tool types, two greedy
reproduction checks, a self-patch, one weight edit/revert, and exactly one
rollout. Layers `[0,20,23,24,41]` exercise ordinary and KV-shared layers. A scripted
fake Responses client goes through the real V0 transport and settlement path;
there is **no API request**. Its fixed usage numbers are labelled synthetic.
`smoke_report.json` records each tool's measured wall time, selected pairs,
model identity, checks, and reproduction mismatches. Mismatches remain failures
of the check; they are not filtered or retried with easier examples.

The future live `diagnose` command additionally requires `--category` and
`--allow-teacher-api`. **Do not use it for the tooling/smoke task.** It is gated
on the v3 final-execution first-divergence freeze and all category diagnosis pairs. Input/output runaway caps default to 2,000,000
and 500,000 tokens per category and are configurable. A sampled text/action
JSON response is the only teacher output requested. No logprobs or hidden
teacher internals are requested.

`make_transport` injects a lazy provider client beneath the existing
`StructuredTransport`, capped client, `SharedTeacherBudget`, reservations and
settled usage ledger. Provider order is Azure first when configured through
`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and
`AZURE_OPENAI_API_VERSION`; otherwise it uses the existing OpenAI credential
resolver. Partial Azure configuration fails instead of silently falling back.
There is no cross-provider failover after a request. Alias stays `gpt-6-luna`.
Provider, returned model ID, timestamp, raw sampled output, rejections, and
reported token usage are recorded. SDK retries remain disabled. V0 owns
connection/truncation retries; P0 tightens rate-limit retries to the original
attempt plus one probe. Quota, another 429, empty/truncated/unknown outcomes,
and exhausted caps fail loudly; raw exception text is never persisted.

Future evaluation requires five `--submission` arguments. The evaluator verifies
every submission and the shared pair freeze, writes `evaluation_freeze.json`,
then creates the evaluation runtime. It does not instantiate a teacher or write
training data. Outputs include `results.jsonl`, all real execution receipts,
condition/direction freezes, collateral/K20 decisions, per-category and pooled
McNemar tests, individual prediction results, `report.json`, and `report.md`.

## Validation status for v3

All **210 tests passed**: 116 P0 tests and 94 regressions, in 53.98 seconds.
JUnit results are at `/tmp/p0-v3-tests.xml`. No GPU, teacher/API or network call
was made; the future real-model smoke command above remains unrun.

Validation uses the four `tests/test_p0_*.py` modules and the same 94 regression
tests: `test_repair_patch_align.py` (33), `test_repair_execute.py` (20),
`test_repair_determinism.py` (3), `test_v0_transport.py` (23), and
`test_v0_truncation.py` (15). The existing autouse fixture blocks network and
live OpenAI transport and clears CUDA visibility. Unit tests use native randomly
initialized tiny Gemma4, fake clients and fake environments. The post-norm tests
use nonuniform norm weights (the old formula demonstrably fails), random inputs,
both projection targets, fp32/bf16 tolerances, alpha=0 byte preservation and exact
rollback. Additional tests cover final-round actions, student divergence,
initial/visible prefix mismatch, exact Thought splits, fallbacks, framing/edit
cache separation, current_thought rewriting/attention and original-Thought
fixed-control construction.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  tests/test_p0_pairs.py tests/test_p0_introspect.py tests/test_p0_agent.py tests/test_p0_eval.py \
  tests/test_repair_patch_align.py tests/test_repair_execute.py tests/test_repair_determinism.py \
  tests/test_v0_transport.py tests/test_v0_truncation.py --junitxml=/tmp/p0-v3-tests.xml
```

The earlier `reports/p0_tooling/validation.json`, `VALIDATION.md`, `tests.xml`,
`extraction/`, and `extraction_v2/` remain historical records. V3 counts and
exclusions are in `extraction_v3/extraction_report.json`. Real-model reproduction
and rollout success remain unmeasured; offline tooling validation establishes
no scientific finding. No tracked files, git index, commits or staging are changed.
