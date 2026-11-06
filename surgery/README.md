# P1 surgery tooling (settings A and B)

Implements §§2–8 of the frozen
`ICML27_agent_distillation/docs/direction/2026-11-05_P1_surgery_prereg_ZH.md`.
This is engineering tooling, not an experiment result. No live teacher call,
GPU experiment, E4 training, or dev140 evaluation is part of the CPU validation.
The reviewer commits; this change does not commit or alter the existing P0/V0 code.

## Files and environment imports

| File | Purpose |
|---|---|
| `surgery/common.py` | Immutable JSON, provenance, output guards, relocation |
| `surgery/pairs.py` | Exact A split; verified K20 evidence and teacher-forced B extraction |
| `surgery/runtime.py` | Pinned base, optional one in-memory merged adapter, recorded E4 identity |
| `surgery/student.py` | P0 student subclass; differentiable scores, gradient/neuron attribution, LoRA |
| `surgery/edits.py` | Ordered fp32 composition, exact rollback, safetensors deltas |
| `surgery/session.py` | Diagnosis-only capabilities, P0 tools, caches and receipts |
| `surgery/agent.py` | P0 agent/transport extension, budgets, compaction, integration |
| `surgery/conditions.py` | E/B condition construction and immutable freeze |
| `surgery/evaluation.py` | M1–M3, initial-state probe/K20 execution, shards/resume/CPU merge, M4 import |
| `surgery/export.py` | Standalone bf16 export, fixed-input reload, file manifest |
| `surgery/hpg_dev140.slurm` | New launcher using tree `46b8527`'s server/eval with explicit E4B config |
| `tools/p1_surgery.py` | All CLI entry points |
| `tests/test_p1_*.py` | CPU tiny-model, fake teacher/environment protocol tests |

The CLI imports the `surgery` modules above. Repository dependencies are
`repair.p0_pairs`, `repair.p0_introspect`, `repair.p0_agent`, `repair.p0_eval`,
`repair.adapter_eval`, `repair.records`, `repair.execute`, `repair.ledger`,
`repair.v0_common`, `repair.v0_transport`, `repair.takeover`, `sed.react`,
`sed.storage`, `sed.provenance`, `sed.seed_evidence`, `sed.config`, `sed.thinking`,
`sed.visibility`, `train.dataset`, `train.modeling`, `train.merge_lora`,
`teachers.openai_teacher`, `teachers.capped`, `teachers.rate_limit`, and
`harness.alfworld_env`. Their normal transitive imports remain unchanged.

Third-party packages: `torch`, `transformers` with native Gemma4 support,
`peft`, `safetensors`, `accelerate`, `tokenizers`, `numpy`, `PyYAML`, `openai`,
`httpx`, `alfworld`, `textworld`; `pytest` for checks. The separate M4 serving
Python also needs the installed compatible `vllm` and Gemma4 parser. These tools
never install packages or download weights. Check the hpg environments before
running; all dependency versions are captured in provenance.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$PY" -c \
 'import surgery.agent, surgery.conditions, surgery.evaluation, surgery.export; import torch, transformers, peft, safetensors, alfworld, textworld; print("imports OK")'
```

## Frozen protocol and artifacts

A uses exactly the 113 P0 v3 rows, changing only each row's `split`. It checks
the P0 content digest `cb4e16996b336e0acb30de7f31c16f05e931aec0c80f3ea747d6c8ecf71533f3`
and source file hash prefix `432fd178`. The new split is:

| Category | Diagnosis | Test |
|---|---:|---:|
| goal_misread | 13 | 14 |
| missing_precondition | 18 | 19 |
| refused_action | 9 | 10 |
| search_incomplete_or_loop | 6 | 6 |
| wrong_object_or_target | 9 | 9 |
| Total | 55 | 58 |

Diagnosis/test have 39/35 unique instances and task groups, with no overlap.
The split SHA256 is
`5dfde976ea4820bbc9b97e5f6c0f6f2b3c92eca31a80bf01590ce5d02a815715`.
K remains 20 trusted executed seed trajectories.

B checks `VERIFIED/{manifest,selection,identity}.json`, the referenced source
collection and every selected episode's hashed execution evidence. The previous
trajectory replies, observations and admissible commands are rendered through
P0's native tokenizer template, with thinking disabled. Each free base greedy
action is compared exactly to the parsed trajectory action. Every mismatch is
diagnosis/category `none`; matches form the collateral pool. All raw base replies
and usage are retained. No split or sample is chosen using base reproduction.
An invalid parsed base action remains a mismatch with `a_f=null`; score/gradient
tools that require a command reject that pair explicitly, rather than fabricating
a command or dropping it. Rollout and repair-target finetuning still accept it.

The seed runner used a different environment-version hash schema. B verifies
its original environment/parser/visibility source hashes and installed
ALFWorld/TextWorld/NumPy versions, then records both the source revision and
the equivalent P0 replay-version hash. It preserves every recorded transition.
Every branch still uses a real reset and prefix replay; P0 checks visible fields,
audit hashes, actions, rewards, seed, game bytes and terminal flags. No simulated
restore or teacher success claim is used.

Each condition is an immutable JSON file binding its ordered edits, fp32 dense
delta files and SHA256s, vectors, starting model, split, selection receipts and
code/dependency provenance. A requires E0/E1/E2_0…E2_4/E3/E4/E5 before evaluation;
B requires B0/B1/B2_0…B2_4/B3/B5. E4 comes from a separately trained checkpoint,
identified using `repair.adapter_eval.training_identity`; no E4 training is done
here. Test-pair training exposure in E4 is refused. Copy each condition directory
with its delta files, and relocate references when moving machines.

P0 tools and rendering are reused. The extra teacher tools are
`gradient_attribution(pair_ids)`, `neuron_attribution(pair_ids,layer,top_n)`, and
`local_finetune(pair_ids,modules,rank,lr,steps)`. Modules are strings such as
`20:mlp.down_proj`. All decision tools accept P0 `framing` values.

Backward tools (`gradient_statistics`/`gradient_attribution`, `neuron_attribution`,
and `local_finetune`, including E5/B5) temporarily checkpoint decoder forwards
with `use_reentrant=False`, preserve eval mode, isolate shared-KV replay, and
restore the exact original forwards even on exceptions. Weight gradients stream
to CPU one tensor at a time; the two margin branches run separately and combine
in the original dtype on CPU before the unchanged fp32 per-pair moments. Each
action graph is freed before the next forward, and logits remain limited to
scored positions. Forward-only tools retain their native execution. On CUDA,
`cuda_peak_memory_allocated_bytes` records `torch.cuda.max_memory_allocated`
after resetting peak statistics at call entry: at the tool-result top level,
on each `gradient_statistics` row, and on each E5/B5 finetune grid candidate.
It includes resident model allocations, excludes allocator reservations/other
processes, and is absent on CPU; cached receipts retain the original call's
peak. For an 8.3k-token sequence with a short action suffix, the pinned E4B bf16
model has about 14.8 GiB of resident weights, 1.7 GiB of residual checkpoints
plus 0.17 GiB of per-layer inputs, and at most about 50 MiB per projection
gradient (instead of a full 7.25 GiB set). Allowing for shared KV, logits,
recomputed layer activations and SDPA workspace gives a planning estimate of
roughly 24–34 GiB allocated, expected to fit 48 GB; this is not a GPU measurement
and depends on the SDPA kernel and suffix length. CPU pair gradients and fp32
moment accumulators need roughly 22 GiB of host RAM. Use the receipts to verify
headroom on the actual Ada/A100 run.

Teacher submissions use `{"mechanism":"...","edit":[...]}`. Edit schemas:

```json
[
  {"type":"project_out","direction_ids":["RETURNED_ID"],"layers":[20,21],"targets":["mlp.down_proj","self_attn.o_proj"],"alpha":0.5},
  {"type":"add_rank1","a":{"direction_id":"OUTPUT_SIZE_ID"},"b":{"direction_id":"INPUT_SIZE_ID"},"layers":[20],"targets":["mlp.down_proj"],"alpha":0.1},
  {"type":"scale_neurons","layers":[20],"neurons":{"20":[3,7]},"c":0.5},
  {"type":"local_finetune","delta_id":"RETURNED_DELTA_ID","alpha":1.0}
]
```

`add_rank1` accepts any existing q/k/v/o/gate/up/down projection with matching
vector dimensions. `project_out` accepts only down/o and 1–8 independent
directions, orthonormalized after multiplication by the native following norm's
weight, without a +1 offset. Edits compose in list order in fp32, then each final
matrix rounds once to its original native dtype (bf16) before installation.
Edited modules run natively without dtype hooks, using exactly the exported
weights. Local LoRA optimization and frozen delta artifacts retain fp32 precision.
Every context restores original parameter objects and verifies their
bytes, including exceptions. Frozen local deltas are loaded at evaluation;
there is no evaluation training path.

The yielded edit receipt, frozen condition's `edit_receipt`, tool receipts,
evaluation units, and export metadata record `rounding`: per-tensor and total
`fp32_changed_parameters`, `rounded_changed_parameters`, and
`rounded_relative_frobenius` = ||rounded weight - original||F / ||original||F.
Totals use the concatenated touched tensors; a zero denominator yields zero for
a zero delta and null otherwise. `per_edit` reports each ordered prefix against
the original weights on that edit's targets. Its `rounded_effect_changed_parameters`
compares consecutive rounded prefixes, and `effect_vanished` flags an fp32
effect lost at that step. Diagnostic rounding never feeds back into composition.
Nonfinite rounded prefixes have null norms and `rounded_finite=false`; only the
fully composed installation must be finite in the native dtype.
Final `vanished` flags a nonzero fp32 composition with zero rounded change.
No-op and vanished tensors retain their original parameter objects and bytes.

| Session | Tools | Rollouts | Distinct candidates | Real steps | Input/output cap |
|---|---:|---:|---:|---:|---|
| goal_misread (13 diagnosis pairs) | 300 | 102 | 25 | 4,080 | 60M / 3M |
| missing_precondition (18) | 300 | 122 | 25 | 4,880 | 60M / 3M |
| refused_action (9) | 300 | 86 | 25 | 3,440 | 60M / 3M |
| search_incomplete_or_loop (6) | 300 | 74 | 25 | 2,960 | 60M / 3M |
| wrong_object_or_target (9) | 300 | 86 | 25 | 3,440 | 60M / 3M |
| Integration (55) | 100 | 250 | 10 | 10,000 | 60M / 3M |
| Global B (90) | 500 | 440 | 40 | 17,600 | 100M / 5M |

**Prereg revision 2, before the remaining formal sessions:** category rollout
caps are `50 + 4 * n_pairs`; global B uses `80 + 4 * n_pairs`, giving
`80 + 4 * 90 = 440`. These preserve revision 1's 50 category and 80 global-B
exploratory rollouts and add two complete edited and unedited verifications.
Integration retains its 30 exploratory rollouts and adds `4 * 55`, for 250.
All real-step caps are `40 * rollout_cap`, including
prefix replay. `surgery.agent.LIMITS` registers the maximum category cap and
the integration/B caps; `CATEGORY_LIMITS` lists the five exact category caps.
`Agent` binds category and global-B budgets to the actual session pair count before use,
including CLI-created budgets; trial overrides may only reduce those caps.

Invalid/cached calls spend tool slots; cached rollout requests spend rollout
slots but no real steps. Local training spends a candidate; reusing its returned
unscaled delta counts as the same candidate. Compositions/rescalings are distinct.
Rollout reserves H=40 conservatively and settles actual replay/continuation steps.
Automatic submission checks spend rollout and real-step budgets through the
same session rollout path, without spending teacher tool slots or charging the
unedited baseline as an extra candidate. Reusing an already completed session
receipt makes no new rollout request and spends nothing; an explicit cached
teacher rollout request still spends a rollout and tool slot.
`agent.jsonl` and `tools/tools.jsonl` retain full requests, responses, failures,
usage and training exposures. The teacher-facing copy replaces tool results
older than the latest 20 with P0-style summaries. Within the latest 20, edit
receipts retain aggregate and per-edit rounding totals, norms, finite/vanished
flags, tensor counts and vanished-tensor counts; the per-tensor rounding tables
remain in the full disk receipts. Gradient rankings and other result fields
remain intact. Teacher transport remains a P1 subclass of `StructuredTransport`
with `SharedTeacherBudget` and the joint rate limiter, including
P0 JSON mode, retries, requested/returned model IDs and timestamps. All sessions
must use the same `--rate-limit-dir`. The teacher is always `gpt-6-luna` and needs
`--allow-teacher-api`.

### Submission verification and confirmation (revision 2)

In category, global-B and integration sessions, the first valid `submit` checks
coverage for the canonical SHA of the **resolved deployed ordered edit list**
(including frozen local-finetune delta references). Every diagnosis pair needs
both an unedited and an edited rollout from its original decision state.
`apply_edit` scores and counterfactual-context rollouts cannot satisfy this gate.
The automatic checks use the normal `Session.call('rollout', ...)` path: real
environment reset, P0 checked prefix replay, free greedy continuation, H=40.
Completed receipts for that exact list and baseline within the session are
reused; an empty edit list shares the baseline execution.

Before starting, the entire missing batch must fit remaining rollouts and
`40 * missing_rollouts` real steps. Actual steps settle after each call, including
failed replay. Execution errors remain fatal and retain their receipts; they
cannot become verified coverage. Real failures and unknown success labels stay
in the comparison table and are counted separately.

The agent returns a `submit_verification` tool result in the transcript: per-pair
unedited/edited success, receipt and execution hashes, and success/failure/unknown
and verified counts. It does not freeze yet. Submitting the same edit again
confirms; further tool calls or a different edit invalidate pending confirmation.
A revised list must have all-pair rollout coverage and return a new table before
confirmation. Existing receipts for an unchanged list may still be reused.
Integration retains its five-input and rescale/drop/reorder/combine restrictions;
its final exact composition now requires these real rollouts on all 55 pairs.

`submission.json` stores `verification`, full `verification_receipts`, and
`confirmation_status`; `agent.jsonl` records verification preflights, automatic
rollout requests/results/budgets, returned tables and the final status. If the
batch cannot fit, freeze the proposed edit with
`unverified_at_budget_exhaustion=true` and verification status
`unverified_at_budget_exhaustion` ("unverified at budget exhaustion"). Preserve
the partial table, missing entries as unknown without receipts, and the limiting
budgets. This outcome is reported distinctly from verified submissions.

### Submit-only token headroom

Before every category, integration or global-B teacher turn, the P1 guard reads
remaining caps minus **tokens or unsettled bounds**, not merely reported usage.
It mirrors the transport reservation exactly: input bound = canonical UTF-8
message bytes + `16 * (message_count + 1)` per attempt. A turn includes both the
16,384-output attempt and the possible 32,768-output truncation retry: two input
bounds and 49,152 output tokens, with no assumed settlement discount.

A normal turn is allowed only when both remaining budgets cover that turn
**plus one further submit-only exchange**, including its notice and retry.
Equality is allowed; a one-token deficit switches immediately to submit-only.
The extra exchange uses the current evidence and the same output caps: it does
not assume a short/cheap answer or discard the latest 20 results. Every following
reply is checked again against its actual prompt, including newly returned tool
results or rejection feedback; future result growth is not assumed bounded.

The notice states `Token budget nearly exhausted` and requires only `submit`.
Teacher tool calls are rejected without execution; automatic verification still
runs and the maximum three submit-only replies still applies. Insufficient
rollout/step budget freezes the explicit unverified submission described above.
If verification completed but tokens or the three-reply limit prevent teacher
confirmation, preserve the pending submission and complete table with
`confirmation_status=unconfirmed_at_budget_exhaustion`; this is distinct from
unverified execution and from an explicit teacher confirmation.
If even the submit-only exchange is unaffordable and there is no pending valid
submission, the agent records
`insufficient_token_budget_for_submit_only` and halts with `BudgetExceeded`
before transport, leaving no submission file. Existing fatal shared-budget
stops remain fatal; this guard does not override transport accounting.

`agent.jsonl` records a `token_budget_preflight` for every proposed turn: normal,
submit-only and selected reservation estimates, remaining input/output budgets,
required headroom, limiting budgets, trigger flag/reason and any stop reason.
The separate `submit_only` event records when and why the notice was issued.

### Tool-result payload sizes

The reported 2026-11-05 trial had approximately 48 KB gradient-attribution
results and 14–20 KB apply-edit results. Those trial receipts are not present in
this checkout, so these figures are user-reported, not remeasured here.

Reproducible **synthetic schema fixture** measurements from
`tests/test_p1_payloads.py`, in canonical UTF-8 bytes of each complete result:

| Tool | Before | Teacher view after |
|---|---:|---:|
| `gradient_attribution` | 36,806 | 36,806 |
| `rollout` | 10,773 | 1,459 |
| `apply_edit` | 10,716 | 1,402 |

This fixture uses a 32-layer × seven-projection gradient ranking, and an edit
touching 16 tensors with one ordered prefix. It measures serialization size,
not model behavior or a trial replay. Rounding tables dominate the edit receipts;
their teacher-only summary saves about 87%. The gradient ranking is the largest
remaining fixture result and is preserved because it guides edit selection.
Tests also run apply-edit/rollout on the CPU tiny student and fake environment,
verify cached and on-disk receipts remain byte-identical, and check the latest-20
window, aggregate values and vanished flags. These actual CPU tool results
measure 3,631 → 3,078 bytes for `apply_edit` and 2,262 → 1,709 for `rollout`
(one tensor); the larger fixture above isolates scaling with tensor count.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:. \
  "$PY" -m pytest -q -s tests/test_p1_payloads.py
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:. \
  "$PY" -m pytest -q tests/test_p1_*.py tests/test_p0_*.py
```

## Host setup

Run the following host-specific setup **on that host**, from this checkout.
These example functions are shorthand for the same CLI, not remote launchers.
All output roots are new P1 directories, outside P0 and V0 runs.

RAI:

```bash
export PYTHONPATH="$PWD/src:$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
PY=/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python
P1=/home/xueqi/hq/projects/ICML27_wt_repair_eval/reports/p1_runs
ROUND=/home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1
VERIFIED=/home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20
BASE=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
ADAPTER=/home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
ALF=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1
E4=/home/xueqi/hq/projects/ICML27_wt_repair_train/reports/p1_e4/update-000100/checkpoint
p1_rai() { "$PY" tools/p1_surgery.py "$@" --base "$BASE" --adapter "$ADAPTER" --round-dir "$ROUND" --verified "$VERIFIED" --alfworld-data "$ALF"; }
```

HPG (replace E4's example update with the recorded final checkpoint):

```bash
export PYTHONPATH="$PWD/src:$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
PY=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/.venv/bin/python
P1=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wt_repair_eval/reports/p1_runs
ROUND=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_v0_runs/v0_k20_r1
VERIFIED=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20
BASE=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
ADAPTER=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
ALF=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment-vllm/envs/alfworld/data/json_2.1.1
E4=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wt_repair_train/reports/p1_e4/update-000100/checkpoint
p1_hpg() { "$PY" tools/p1_surgery.py "$@" --base "$BASE" --adapter "$ADAPTER" --round-dir "$ROUND" --verified "$VERIFIED" --alfworld-data "$ALF" \
  --relocate /home/xueqi/hq/projects=/blue/fsu-compsci-dept/xc25.fsu/hq \
  --relocate /home/xueqi/.cache/huggingface=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache; }
```

Use explicit `--base`, `--adapter`, `--e4-checkpoint`, `--verified`,
`--round-dir`, `--manifest-path`, `--alfworld-data`, `--config`, and
`--rate-limit-dir` to override paths. Repeat `--relocate OLD=NEW` for paths read
from frozen inputs. It performs one lexical component-prefix substitution,
preserves the relative tail, forbids overlapping OLD prefixes and symlink
escapes, and verifies original bytes after relocation. It never rewrites frozen
JSON, hashes, prompt text or arbitrary audit strings. This path contract is a
small attributed port of the training worktree's `repair/v2_inputs.Relocation`,
used by `tools/train_v0_arm.py`.

## Every subcommand, on both hosts

Only `split-a`, `evaluate --merge`, and `verify-export` are CPU-only. Prefix
those commands with `CUDA_VISIBLE_DEVICES=''`. All other commands require the
single-GPU determinism environment above. Formal M1–M3 runs belong on HPG B200;
RAI GPU examples describe the portable interface and trial workflow.

1. Freeze A (`--pairs` here is the unchanged P0 input):

```bash
CUDA_VISIBLE_DEVICES='' p1_rai split-a --pairs reports/p0_tooling/extraction_v3/pairs.json --output "$P1/A/data"
CUDA_VISIBLE_DEVICES='' p1_hpg split-a --pairs reports/p0_tooling/extraction_v3/pairs.json --output "$P1/A/data"
```

2. Extract B (always raw base, no adapter; reads only verified K20 data):

```bash
p1_rai extract-b --output "$P1/B/data"
p1_hpg extract-b --output "$P1/B/data"
```

3. Diagnose A (one category per GPU/job; repeat for all five categories), or B:

```bash
p1_rai diagnose --pairs "$P1/A/data/pairs.json" --category goal_misread --output "$P1/A/category/goal_misread" --rate-limit-dir "$P1/joint-rate" --allow-teacher-api
p1_hpg diagnose --pairs "$P1/A/data/pairs.json" --category goal_misread --output "$P1/A/category/goal_misread" --rate-limit-dir "$P1/joint-rate" --allow-teacher-api
p1_rai diagnose --pairs "$P1/B/data/pairs.json" --output "$P1/B/global" --rate-limit-dir "$P1/joint-rate" --allow-teacher-api
p1_hpg diagnose --pairs "$P1/B/data/pairs.json" --output "$P1/B/global" --rate-limit-dir "$P1/joint-rate" --allow-teacher-api
```

On HPG, credentials belong only under
`/blue/fsu-compsci-dept/xc25.fsu/hq/secrets/` (directory 700, file 600). Use the
existing P0 provider environment/secret-file setup. Never echo/source-trace key
values into logs. No command here requests teacher internals or hidden reasoning.

4. Integrate the five frozen A submissions:

```bash
SUBMISSIONS=()
for c in goal_misread missing_precondition refused_action search_incomplete_or_loop wrong_object_or_target; do
  SUBMISSIONS+=(--submission "$P1/A/category/$c/submission.json")
done
p1_rai integrate --pairs "$P1/A/data/pairs.json" "${SUBMISSIONS[@]}" --output "$P1/A/integration" --rate-limit-dir "$P1/joint-rate" --allow-teacher-api
p1_hpg integrate --pairs "$P1/A/data/pairs.json" "${SUBMISSIONS[@]}" --output "$P1/A/integration" --rate-limit-dir "$P1/joint-rate" --allow-teacher-api
```

Integration verifies exact category, split and student identities. It can
rescale/drop/reorder/combine existing edits, but cannot train a new delta or
deploy newly invented directions/neurons. The exact final deployed list and the
unedited baseline require real rollouts on every diagnosis pair, followed by the
verification-table/confirmation exchange described above.

5. Build all non-E4 controls, including diagnosis-only grid searches:

```bash
p1_rai controls --pairs "$P1/A/data/pairs.json" --submission "$P1/A/integration/submission.json" --output "$P1/A/conditions"
p1_hpg controls --pairs "$P1/A/data/pairs.json" --submission "$P1/A/integration/submission.json" --output "$P1/A/conditions"
p1_rai controls --pairs "$P1/B/data/pairs.json" --submission "$P1/B/global/submission.json" --output "$P1/B/conditions"
p1_hpg controls --pairs "$P1/B/data/pairs.json" --submission "$P1/B/global/submission.json" --output "$P1/B/conditions"
```

E2/B2 use seeds 0–4 and preserve edit order, layers, modules, ranks and
coefficients. Neurons are sampled without replacement within each selected
layer; local deltas are independent Gaussian tensors with matched Frobenius
norms. E3/B3 use layers 20–41 down_proj, P0 original_thought action-token means,
and alpha `{0.25,0.5,1}`. E5/B5 choose the top 25% eligible projection tensors
by gradient SNR, then rank 16 LoRA over `{1e-5,3e-5,1e-4} × {50,200}`.
Every grid entry runs all diagnosis pairs. No test/probe/dev140 outcome can
select a condition. Unknown candidate verification cannot win selection.

6. Freeze the separately trained A E4 checkpoint:

```bash
p1_rai freeze-e4 --pairs "$P1/A/data/pairs.json" --e4-checkpoint "$E4" --output "$P1/A/conditions"
p1_hpg freeze-e4 --pairs "$P1/A/data/pairs.json" --e4-checkpoint "$E4" --output "$P1/A/conditions"
```

E4 training stays in the training repository: diagnosis repair episodes R plus
K20 D, probabilities 5/6 and 1/6, `ceil(4*rows/16)` updates, all LoRA tensors,
continued from theta0. Supply the resulting recorded V0 `update-N/checkpoint`;
never silently substitute a checkpoint with similar directory names.

7. Evaluate stable `i/n` shards, resume, and merge on CPU:

```bash
p1_rai evaluate --pairs "$P1/A/data/pairs.json" --conditions "$P1/A/conditions" --e4-checkpoint "$E4" --shard 0/4 --output "$P1/A/eval"
p1_hpg evaluate --pairs "$P1/A/data/pairs.json" --conditions "$P1/A/conditions" --e4-checkpoint "$E4" --shard 0/4 --output "$P1/A/eval"
p1_rai evaluate --pairs "$P1/A/data/pairs.json" --conditions "$P1/A/conditions" --e4-checkpoint "$E4" --shard 0/4 --resume --output "$P1/A/eval"
p1_hpg evaluate --pairs "$P1/A/data/pairs.json" --conditions "$P1/A/conditions" --e4-checkpoint "$E4" --shard 0/4 --resume --output "$P1/A/eval"
CUDA_VISIBLE_DEVICES='' p1_rai evaluate --pairs "$P1/A/data/pairs.json" --conditions "$P1/A/conditions" --merge --output "$P1/A/eval"
CUDA_VISIBLE_DEVICES='' p1_hpg evaluate --pairs "$P1/A/data/pairs.json" --conditions "$P1/A/conditions" --merge --output "$P1/A/eval"
```

Launch the other shard indices against the same directory and shard count.
For B substitute `B` for `A` and omit `--e4-checkpoint`. Resume verifies hashes
and execution receipts, then skips completed units; incomplete/extra units
prevent merge. Work units hash `(condition,metric,key)`, so all five random
seeds receive all metrics. Reports are `report.json` and `report.md`.
M1 flip denominator contains only pairs whose unedited greedy reproduces a_f.
M2 probes and original K20 start at the initial state, without G3 replay prefixes.
B labels original K20 as training tasks and has no M1. M3 uses the unchanged
P0 200-decision S sample for A, or Random(0)'s 200 matched steps for B.
Unknown outcomes and exact denominators remain explicit. McNemar appears only
in the JSON appendix. The random mean never treats five seeds as 5n tasks.

8. Export and verify each condition, including E0/B0 and all random seeds:

```bash
p1_rai export --pairs "$P1/A/data/pairs.json" --condition "$P1/A/conditions/E1.json" --output "$P1/A/export/E1"
p1_hpg export --pairs "$P1/A/data/pairs.json" --condition "$P1/A/conditions/E1.json" --output "$P1/A/export/E1"
CUDA_VISIBLE_DEVICES='' p1_rai verify-export --output "$P1/A/export/E1"
CUDA_VISIBLE_DEVICES='' p1_hpg verify-export --output "$P1/A/export/E1"
```

E4 export additionally takes `--e4-checkpoint "$E4"`. Export writes standalone
bf16 HF weights, snapshot config/tokenizer/chat assets, `consistency.json` and
`surgery_manifest.json`. Every touched tensor, including no-ops, must match the
installed tensor byte-for-byte in the serialized safetensors shards. Export
performs no additional dtype conversion. It compares every next-token vocabulary logit on two
fixed, non-task prompts, using merge_lora's bf16 `atol=.125, rtol=.01`; exact
reload equality is also reported. Failure writes a failed receipt and prevents
serving. Delta files retain fp32 even though standalone deployment requires bf16.

## Exact dev140 sbatch template

Read-only inspection of `git show 46b8527:scripts/hpg_eval_student.slurm` confirms
`MERGED_MODEL_PATH` is passed to `harness/serve_student.py`. The server supports
standalone HF directories. However, that launcher invokes the 12B default config
and tree `46b8527` has no `configs/base_e4b.yaml`. Therefore the new launcher
passes this P1 checkout's E4B config explicitly while using the historical tree's
server and `eval/run_suite.py`, greedy/off/1024/32768 semantics. The historical
server itself selects TRITON_ATTN for native Gemma4 with batch invariance.

Prepare an existing read-only checkout at exactly `46b8527` as `EVAL_ROOT`.
From the P1 checkout on HPG, submit the following template (one GPU per condition):

```bash
env PROJECT_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wt_repair_eval \
 EVAL_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_eval_46b8527 \
 P1_CONFIG=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wt_repair_eval/configs/base_e4b.yaml \
 ENV_PYTHON=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/.venv/bin/python \
 VLLM_PYTHON=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment-vllm/envs/vllm-serve/.venv/bin/python \
 MODEL_PATH=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 \
 MERGED_MODEL_PATH=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wt_repair_eval/reports/p1_runs/A/export/E1 \
 CONDITIONS=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wt_repair_eval/reports/p1_runs/A/conditions \
 TASKS=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/data/manifest/dev140.json \
 VERIFIED=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20 \
 WORKERS=16 BATCH_INVARIANT=1 ENABLE_THINKING=off MAX_TOKENS=1024 \
 OUT=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wt_repair_eval/reports/p1_runs/A/dev140/E1 \
 sbatch --export=ALL surgery/hpg_dev140.slurm
```

`HPG_CONFIG`, `ENV_PYTHON`, `VLLM_PYTHON`, `ALFWORLD_DATA_ROOT`, `SED_MANIFEST`
and `SED_SPLIT_FREEZE` remain overridable via the inherited job environment.
The historical evaluation does not consume VERIFIED for training; the wrapper
checks the complete frozen condition set before reading dev140 and hash-binds
the required K20 reference in `p1_launch.json`. E0 must follow this
same standalone path, supplying the merged theta0 path-consistency comparison
against the historical LoRA 78/140 result. B's merged-SFT reference is A E0,
not an extra B condition. Preserve all three reported historical LoRA seeds
as references only (55.7/55.7/64.3%).

After dev140 jobs finish, import results during CPU merge; repeat the flag for
every condition and every random seed:

```bash
CUDA_VISIBLE_DEVICES='' p1_rai evaluate --pairs "$P1/A/data/pairs.json" --conditions "$P1/A/conditions" --merge --m4-result "E1=$P1/A/dev140/E1" --output "$P1/A/eval"
CUDA_VISIBLE_DEVICES='' p1_hpg evaluate --pairs "$P1/A/data/pairs.json" --conditions "$P1/A/conditions" --merge --m4-result "E1=$P1/A/dev140/E1" --output "$P1/A/eval"
```

M4 ingestion verifies the condition/export identity, dev140 task-list hash,
140 episodes, greedy policy and standalone batch-invariant server. Until imported,
each condition's M4 is explicitly pending. Apply prereg §6/§8 point-estimate
criteria to the complete report; no significance threshold is a decision gate.

## §7 small-budget trial (excluded from formal results)

Use a new `trial` directory, one A category, one GPU. Budget reductions require
`--trial`; formal defaults cannot be accidentally changed by these switches.
Use the host's `p1_rai` or `p1_hpg` function:

```bash
p1_hpg diagnose --pairs "$P1/A/data/pairs.json" --category search_incomplete_or_loop \
  --trial --tool-calls 20 --rollouts 2 --candidate-edits 3 --environment-steps 80 \
  --input-token-cap 1000000 --output-token-cap 100000 \
  --rate-limit-dir "$P1/joint-rate" --allow-teacher-api --output "$P1/trial/category"
p1_hpg controls --pairs "$P1/A/data/pairs.json" --trial \
  --submission "$P1/trial/category/submission.json" --output "$P1/trial/conditions"
p1_hpg evaluate --pairs "$P1/A/data/pairs.json" --conditions "$P1/trial/conditions" \
  --trial --trial-limit 1 --shard 0/1 --output "$P1/trial/eval"
CUDA_VISIBLE_DEVICES='' p1_hpg evaluate --pairs "$P1/A/data/pairs.json" \
  --conditions "$P1/trial/conditions" --trial --merge --output "$P1/trial/eval"
p1_hpg export --pairs "$P1/A/data/pairs.json" --condition "$P1/trial/conditions/E1.json" \
  --output "$P1/trial/export/E1"
```

For the trial, test vLLM loading without opening dev140. On the allocated HPG
GPU, set `EVAL_ROOT`, `ENV_PYTHON`, and `VLLM_PYTHON` to the same paths as the
sbatch template, then run:

```bash
export VLLM_BATCH_INVARIANT=1
"$ENV_PYTHON" "$EVAL_ROOT/harness/serve_student.py" --config "$PWD/configs/base_e4b.yaml" \
  --python "$VLLM_PYTHON" --model-path "$BASE" --merged-model-path "$P1/trial/export/E1" \
  --port 0 --output "$P1/trial/vllm" &
trial_server_pid=$!
trap 'kill -TERM "$trial_server_pid" 2>/dev/null || true' EXIT
until [[ -f "$P1/trial/vllm/ready.json" ]]; do
  kill -0 "$trial_server_pid" || { wait "$trial_server_pid"; exit 1; }
  sleep 1
done
kill -TERM "$trial_server_pid"
wait "$trial_server_pid" || true
trap - EXIT
```

The M4 launcher refuses trial exports. All trial observations stay excluded from
formal findings. Formal category/integration/control runs must use fresh outputs
and the unchanged preregistered budgets/grids. CPU fake tests, without any model
download, teacher API or network, are:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:. \
 "$PY" -m pytest -q tests/test_p1_*.py tests/test_p0_*.py
```

## Assumptions resolving unspecified details

* Split: visit sorted categories, sort group IDs then use a fresh Random(0)
  shuffle in each category. Mixed-category groups count toward every affected
  category; skip a group if any target would overflow. Require all targets
  exactly, with no fallback split. This produces the stated 55/58 freeze.
* Prereg revision 2 replaces the old integration score-check/one-rollout gate
  with real edited and unedited rollouts on all 55 diagnosis pairs for the exact
  final list. Integration has 250 rollouts / 10,000 real steps. Category/global-B
  sessions use the same all-pair verification and confirmation protocol; their
  exact revised limits and budget-exhaustion outcomes are documented above.
* Gradient attribution ranks the Frobenius norm of the mean margin gradient.
  Neuron attribution is signed activation×gradient at the last decision-prefix
  token, averaged over pairs and ranked by absolute magnitude. Shared-KV tensors
  absent from the native model are omitted, never synthesized.
* Automatic SNR is `||mean(g)||F / max(sqrt(mean(||g-mean(g)||F²)),1e-12)` over
  the existing seven projection families. The top 25% count rounds up; tensor
  names break ties. All diagnosis pairs contribute equally. A single alpha
  is chosen for the ordered five-category fixed-rule composition; first grid
  entry wins success-rate ties. Unknown rollout success cannot win selection.
* Local LoRA uses AdamW, zero weight decay/dropout, scale 1 (`alpha=rank`),
  seed 0, microbatch one, and cyclic supplied pair order. Each step uses summed
  P0 action-sequence NLL (Action wrapper and turn-end included), with the selected
  framing. Targets are the chosen pairs' registered a_r, not arbitrary labels.
  Dense deltas are trained relative to the session's unedited starting student;
  composition then applies them in the submitted order. Training and failed
  attempts retain exposure/usage receipts.
* B original_thought uses the base greedy reply's public prefix; previous turns
  remain the trusted trajectory replies. Invalid parsed actions stay in the
  dataset and require explicit handling by command-scoring tools. If fewer than
  200 matched steps exist, collateral sampling fails instead of duplicating or
  silently reducing the registered sample.
* Standalone bf16 export requires byte-identical installed/exported tensors,
  uses merge_lora's default numerical tolerances for the fixed-input logits,
  and reports exact reload logits equality separately. All in-memory evaluation
  uses the same rounded weights; fp32 delta artifacts retain small updates.
